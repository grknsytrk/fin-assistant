"""Pluggable cache backend for RAG-Fin.

The application defaults to a process-local in-memory TTL cache. When the
operator wants to scale out (multi-worker deployment, response caching for a
public site, etc.) they can flip the backend to Redis without touching call
sites:

    RAGFIN_CACHE_BACKEND=redis
    RAGFIN_REDIS_URL=redis://localhost:6379/0

If the Redis backend is requested but unavailable (network down, library not
installed, etc.) we silently fall back to in-memory caching and keep the
application running. This way Redis is *additive*: it can only speed things
up, it can never break the existing flow.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


_CACHE_BACKEND_ENV = "RAGFIN_CACHE_BACKEND"
_CACHE_REDIS_URL_ENV = "RAGFIN_REDIS_URL"
_CACHE_NAMESPACE_ENV = "RAGFIN_CACHE_NAMESPACE"
_DEFAULT_NAMESPACE = "ragfin"


class CacheBackend:
    """Common interface every backend must implement.

    The contract is intentionally narrow so swapping backends is trivial.
    """

    name: str = "noop"

    def get(self, key: str) -> Any:
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def delete_prefix(self, prefix: str) -> int:
        return 0

    @contextmanager
    def lock(self, key: str, *, timeout: float = 30.0) -> Iterator[bool]:
        # The default no-op lock immediately yields ``True`` so call sites can be
        # written assuming a lock is always acquired.
        yield True


class InMemoryCache(CacheBackend):
    """Process-local TTL cache backed by a plain dict + threading lock."""

    name = "memory"

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._mutex = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()

    def get(self, key: str) -> Any:
        now = time.time()
        with self._mutex:
            payload = self._store.get(key)
            if payload is None:
                return None
            expires_at, value = payload
            if expires_at and expires_at < now:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds and ttl_seconds > 0 else 0
        with self._mutex:
            self._store[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        with self._mutex:
            self._store.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        with self._mutex:
            keys = [key for key in self._store if key.startswith(prefix)]
            for key in keys:
                self._store.pop(key, None)
            return len(keys)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_mutex:
            existing = self._locks.get(key)
            if existing is None:
                existing = threading.Lock()
                self._locks[key] = existing
            return existing

    @contextmanager
    def lock(self, key: str, *, timeout: float = 30.0) -> Iterator[bool]:
        lock_obj = self._lock_for(key)
        acquired = lock_obj.acquire(timeout=max(0.001, timeout))
        try:
            yield acquired
        finally:
            if acquired:
                lock_obj.release()


class RedisCache(CacheBackend):
    """Redis-backed cache. Falls back to in-memory if anything goes wrong.

    Values are JSON-serialised. Locks are implemented with the standard
    ``SET key value NX EX`` pattern + a UUID guard so that we don't release
    someone else's lock when our work runs longer than the lock TTL.
    """

    name = "redis"

    def __init__(self, url: str, namespace: str) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(f"redis client not available: {exc}") from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._namespace = namespace
        self.last_error: Optional[str] = None
        # A small in-memory fallback handles transient Redis blips so call
        # sites still see useful data while we wait for the network to recover.
        self._fallback = InMemoryCache()
        # Probe the connection up front so configuration mistakes surface
        # immediately at startup rather than masquerading as cache misses.
        self._client.ping()

    def _prefixed(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def _remember_error(self, exc: Exception) -> None:
        self.last_error = str(exc)

    def get(self, key: str) -> Any:
        try:
            raw = self._client.get(self._prefixed(key))
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._remember_error(exc)
            logger.warning("redis cache get failed for %s: %s", key, exc)
            return self._fallback.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:
            logger.debug("skipping non-serialisable cache value for %s: %s", key, exc)
            self._fallback.set(key, value, ttl_seconds=ttl_seconds)
            return
        try:
            if ttl_seconds and ttl_seconds > 0:
                self._client.set(self._prefixed(key), payload, ex=int(ttl_seconds))
            else:
                self._client.set(self._prefixed(key), payload)
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._remember_error(exc)
            logger.warning("redis cache set failed for %s: %s", key, exc)
            self._fallback.set(key, value, ttl_seconds=ttl_seconds)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._prefixed(key))
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._remember_error(exc)
            logger.warning("redis cache delete failed for %s: %s", key, exc)
        self._fallback.delete(key)

    def delete_prefix(self, prefix: str) -> int:
        deleted = self._fallback.delete_prefix(prefix)
        try:
            full_prefix = self._prefixed(prefix)
            keys = list(self._client.scan_iter(match=f"{full_prefix}*"))
            if keys:
                deleted += int(self._client.delete(*keys) or 0)
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._remember_error(exc)
            logger.warning("redis cache delete_prefix failed for %s: %s", prefix, exc)
        return deleted

    @contextmanager
    def lock(self, key: str, *, timeout: float = 30.0) -> Iterator[bool]:
        token = uuid.uuid4().hex
        full_key = self._prefixed(f"lock:{key}")
        ttl_ms = int(max(1.0, timeout) * 1000)
        acquired = False
        try:
            try:
                acquired = bool(self._client.set(full_key, token, nx=True, px=ttl_ms))
            except Exception as exc:  # pragma: no cover - depends on Redis
                self._remember_error(exc)
                logger.warning("redis lock failed for %s, falling back to memory: %s", key, exc)
                with self._fallback.lock(key, timeout=timeout) as got:
                    yield got
                return
            yield acquired
        finally:
            if acquired:
                try:
                    # Compare-and-delete via Lua so we never release a lock that
                    # has already expired and been re-acquired by another worker.
                    script = (
                        "if redis.call('get', KEYS[1]) == ARGV[1] then "
                        "return redis.call('del', KEYS[1]) else return 0 end"
                    )
                    self._client.eval(script, 1, full_key, token)
                except Exception as exc:  # pragma: no cover
                    self._remember_error(exc)
                    logger.warning("redis unlock failed for %s: %s", key, exc)


_BACKEND_LOCK = threading.Lock()
_BACKEND: Optional[CacheBackend] = None
_LAST_BACKEND_ERROR: Optional[str] = None
_REQUESTED_BACKEND: str = "memory"


def _build_backend() -> CacheBackend:
    global _LAST_BACKEND_ERROR, _REQUESTED_BACKEND
    backend_name = (os.getenv(_CACHE_BACKEND_ENV) or "memory").strip().lower()
    _REQUESTED_BACKEND = backend_name or "memory"
    _LAST_BACKEND_ERROR = None
    namespace = (os.getenv(_CACHE_NAMESPACE_ENV) or _DEFAULT_NAMESPACE).strip() or _DEFAULT_NAMESPACE
    if backend_name in {"", "memory", "in-memory", "inmemory", "local"}:
        return InMemoryCache()
    if backend_name == "redis":
        url = (os.getenv(_CACHE_REDIS_URL_ENV) or "redis://localhost:6379/0").strip()
        try:
            return RedisCache(url=url, namespace=namespace)
        except Exception as exc:
            _LAST_BACKEND_ERROR = str(exc)
            logger.warning("redis cache backend unavailable, falling back to memory: %s", exc)
            return InMemoryCache()
    _LAST_BACKEND_ERROR = f"unknown cache backend: {backend_name}"
    logger.warning("unknown cache backend %r, falling back to memory", backend_name)
    return InMemoryCache()


def get_cache() -> CacheBackend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = _build_backend()
    return _BACKEND


def reset_cache_for_tests() -> None:
    """Drop the cached backend instance and any in-memory state."""

    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = None


def cache_status() -> Dict[str, Any]:
    backend = get_cache()
    requested = _REQUESTED_BACKEND or (os.getenv(_CACHE_BACKEND_ENV) or "memory").strip().lower() or "memory"
    namespace = (os.getenv(_CACHE_NAMESPACE_ENV) or _DEFAULT_NAMESPACE).strip() or _DEFAULT_NAMESPACE
    redis_error = getattr(backend, "last_error", None)
    return {
        "cache_backend": backend.name,
        "cache_requested_backend": requested,
        "cache_namespace": namespace,
        "cache_redis_fallback": requested == "redis" and (backend.name != "redis" or bool(redis_error)),
        "cache_last_error": redis_error or _LAST_BACKEND_ERROR,
    }


def cached(
    *,
    key_fn: Callable[..., str],
    ttl_seconds: int,
    skip_when: Optional[Callable[..., bool]] = None,
    single_flight: bool = False,
    lock_timeout: float = 30.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Function-level response cache.

    ``key_fn`` builds a deterministic cache key from the wrapped function's
    arguments. ``ttl_seconds`` controls how long Redis (or memory) keeps the
    result. Pass ``skip_when`` to bypass the cache for specific inputs (e.g.
    very large payloads or admin-only endpoints).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if skip_when and skip_when(*args, **kwargs):
                return func(*args, **kwargs)
            try:
                key = key_fn(*args, **kwargs)
            except Exception:
                # If the key cannot be built we still want the wrapped function
                # to run; we just won't cache the result.
                return func(*args, **kwargs)
            backend = get_cache()
            cached_value = backend.get(key)
            if cached_value is not None:
                return cached_value
            if single_flight:
                with backend.lock(f"single-flight:{key}", timeout=lock_timeout) as acquired:
                    if acquired:
                        cached_value = backend.get(key)
                        if cached_value is not None:
                            return cached_value
                        result = func(*args, **kwargs)
                        if result is not None:
                            try:
                                backend.set(key, result, ttl_seconds=ttl_seconds)
                            except Exception as exc:  # pragma: no cover - defensive
                                logger.debug("cache set failed for %s: %s", key, exc)
                        return result
                    else:
                        deadline = time.time() + max(0.05, lock_timeout)
                        while time.time() < deadline:
                            time.sleep(0.05)
                            cached_value = backend.get(key)
                            if cached_value is not None:
                                return cached_value
                        result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            if result is not None:
                try:
                    backend.set(key, result, ttl_seconds=ttl_seconds)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("cache set failed for %s: %s", key, exc)
            return result

        return wrapper

    return decorator
