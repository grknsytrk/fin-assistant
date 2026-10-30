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
import copy
import logging
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

_RESPONSE_REFRESH_SLOTS = threading.BoundedSemaphore(4)
_RESPONSE_REFRESH_MUTEX = threading.Lock()
_RESPONSE_REFRESH_ACTIVE: set[str] = set()


_CACHE_BACKEND_ENV = "RAGFIN_CACHE_BACKEND"
_CACHE_REDIS_URL_ENV = "RAGFIN_REDIS_URL"
_CACHE_NAMESPACE_ENV = "RAGFIN_CACHE_NAMESPACE"
_REDIS_COOLDOWN_SECONDS_ENV = "RAGFIN_REDIS_COOLDOWN_SECONDS"
_DEFAULT_NAMESPACE = "ragfin"
_DEFAULT_REDIS_COOLDOWN_SECONDS = 120
_DEFAULT_REDIS_QUOTA_RETRY_SECONDS = 60 * 60

_CACHE_METRIC_FAMILIES = (
    "kap_flow_store",
    "snapshot-row-map",
    "kap",
    "funds",
    "market",
    "coordination",
    "other",
)
_CACHE_METRIC_FIELDS = (
    "get_calls",
    "get_remote_attempts",
    "get_hits",
    "get_misses",
    "get_errors",
    "get_fallbacks",
    "get_payload_bytes",
    "set_calls",
    "set_remote_attempts",
    "set_successes",
    "set_errors",
    "set_fallbacks",
    "set_payload_bytes",
    "set_attempted_payload_bytes",
    "cooldown_skips",
    "other_remote_attempts",
    "other_remote_errors",
    "other_fallbacks",
)


def _cache_key_family(key: str) -> str:
    """Map a cache key to a bounded, non-sensitive telemetry family."""

    normalized = str(key or "").strip().lower()
    if normalized.startswith("api:kap:flow:"):
        return "kap_flow_store"
    if normalized.startswith("api:funds:snapshot-row-map:"):
        return "snapshot-row-map"
    if normalized.startswith(("lock:", "swr-revalidate:", "single-flight:")):
        return "coordination"
    if normalized.startswith("api:kap:"):
        return "kap"
    if normalized.startswith("api:funds:"):
        return "funds"
    if normalized.startswith("api:market:"):
        return "market"
    return "other"


def _text_payload_bytes(value: Any) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, bytearray):
        return len(value)
    return len(str(value).encode("utf-8", errors="replace"))


class CacheBackend:
    """Common interface every backend must implement.

    The contract is intentionally narrow so swapping backends is trivial.
    """

    name: str = "noop"

    def get(self, key: str) -> Any:
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        return None

    def set_if_absent(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Store a value only when the key does not already exist."""

        if self.get(key) is not None:
            return False
        self.set(key, value, ttl_seconds=ttl_seconds)
        return True

    def delete(self, key: str) -> None:
        return None

    def renew_if_owner(self, key: str, owner: Any, *, ttl_seconds: int) -> bool:
        return False

    def release_if_owner(self, key: str, owner: Any) -> bool:
        return False

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

    def set_if_absent(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        expires_at = time.time() + ttl_seconds if ttl_seconds and ttl_seconds > 0 else 0
        with self._mutex:
            existing = self._store.get(key)
            if existing is not None:
                existing_expires_at, _ = existing
                if not existing_expires_at or existing_expires_at >= time.time():
                    return False
                self._store.pop(key, None)
            self._store[key] = (expires_at, value)
            return True

    def delete(self, key: str) -> None:
        with self._mutex:
            self._store.pop(key, None)

    def renew_if_owner(self, key: str, owner: Any, *, ttl_seconds: int) -> bool:
        with self._mutex:
            current = self._store.get(key)
            if current is None:
                return False
            expires_at, value = current
            if expires_at and expires_at < time.time():
                self._store.pop(key, None)
                return False
            if value != owner:
                return False
            self._store[key] = (time.time() + max(1, int(ttl_seconds)), value)
            return True

    def release_if_owner(self, key: str, owner: Any) -> bool:
        with self._mutex:
            current = self._store.get(key)
            if current is None:
                return False
            expires_at, value = current
            if expires_at and expires_at < time.time():
                self._store.pop(key, None)
                return False
            if value != owner:
                return False
            self._store.pop(key, None)
            return True

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
        self._cooldown_seconds = max(
            0,
            int(os.getenv(_REDIS_COOLDOWN_SECONDS_ENV, str(_DEFAULT_REDIS_COOLDOWN_SECONDS))),
        )
        self._circuit_lock = threading.Lock()
        self._circuit_open = False
        self._cooldown_until = 0.0
        self._half_open_probe = False
        self._quota_limited = False
        self._quota_limit_logged = False
        # A small in-memory fallback handles transient Redis blips so call
        # sites still see useful data while we wait for the network to recover.
        self._fallback = InMemoryCache()
        self._metrics_lock = threading.Lock()
        self._metrics_started_at = time.monotonic()
        self._metrics_totals = {field: 0 for field in _CACHE_METRIC_FIELDS}
        self._metrics_by_family = {
            family: {field: 0 for field in _CACHE_METRIC_FIELDS}
            for family in _CACHE_METRIC_FAMILIES
        }
        # Probe once up front. Transient errors retry after a short cooldown;
        # Upstash plan-limit errors retry less often to avoid filling logs.
        self._record_metrics("redis-health:ping", other_remote_attempts=1)
        try:
            self._client.ping()
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics("redis-health:ping", other_remote_errors=1)
            self._log_remote_error("ping", "redis-health:ping", exc)
        else:
            self._remember_success()

    def _prefixed(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def _record_metrics(self, key: str, **increments: int) -> None:
        family = _cache_key_family(key)
        with self._metrics_lock:
            for field, value in increments.items():
                if field not in self._metrics_totals:
                    continue
                amount = max(0, int(value))
                self._metrics_totals[field] += amount
                self._metrics_by_family[family][field] += amount

    def metrics_snapshot(self) -> Dict[str, Any]:
        with self._metrics_lock:
            return {
                "uptime_seconds": round(time.monotonic() - self._metrics_started_at, 1),
                "totals": dict(self._metrics_totals),
                "by_key_prefix": {
                    family: dict(metrics)
                    for family, metrics in self._metrics_by_family.items()
                    if any(metrics.values())
                },
                "payload_bytes_are_approximate": True,
            }

    def circuit_snapshot(self) -> Dict[str, Any]:
        with self._circuit_lock:
            remaining = max(0.0, self._cooldown_until - time.monotonic())
            return {
                "open": self._circuit_open,
                "cooldown_remaining_seconds": int(math.ceil(remaining)),
                "cooldown_seconds": self._cooldown_seconds,
                "quota_limited": self._quota_limited,
                "quota_retry_seconds": _DEFAULT_REDIS_QUOTA_RETRY_SECONDS,
                "half_open_probe_in_flight": self._half_open_probe,
            }

    def _should_attempt_redis(self) -> bool:
        with self._circuit_lock:
            if not self._circuit_open:
                return True
            if time.monotonic() < self._cooldown_until:
                return False
            # After cooldown, allow one request to probe recovery. Concurrent
            # callers use memory until that probe succeeds or reopens the circuit.
            if self._half_open_probe:
                return False
            self._half_open_probe = True
            return True

    def _remember_error(self, exc: Exception) -> Tuple[bool, bool]:
        now = time.monotonic()
        message = str(exc).casefold()
        is_plan_limit = "fixed plan limits" in message
        with self._circuit_lock:
            already_cooling_down = self._circuit_open and now < self._cooldown_until
            self.last_error = str(exc)
            if is_plan_limit:
                self._quota_limited = True
            if self._quota_limited:
                retry_seconds = _DEFAULT_REDIS_QUOTA_RETRY_SECONDS
            else:
                retry_seconds = self._cooldown_seconds
            self._circuit_open = True
            self._cooldown_until = now + retry_seconds
            self._half_open_probe = False
            if self._quota_limited:
                should_log = is_plan_limit and not self._quota_limit_logged
                self._quota_limit_logged = True
                return False, should_log
        return not already_cooling_down, False

    def _remember_success(self) -> None:
        with self._circuit_lock:
            self.last_error = None
            self._circuit_open = False
            self._cooldown_until = 0.0
            self._half_open_probe = False
            self._quota_limited = False

    def _log_remote_error(self, operation: str, key: str, exc: Exception) -> None:
        should_warn, should_log_quota = self._remember_error(exc)
        if should_log_quota:
            logger.info(
                "redis cache %s hit an Upstash plan limit for key_family=%s; "
                "using in-memory fallback and retrying in %d seconds",
                operation,
                _cache_key_family(key),
                _DEFAULT_REDIS_QUOTA_RETRY_SECONDS,
            )
        elif should_warn:
            logger.warning(
                "redis cache %s failed for key_family=%s; cooldown_seconds=%d: %s",
                operation,
                _cache_key_family(key),
                self._cooldown_seconds,
                exc,
            )

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value, default=str, separators=(",", ":"))

    def get(self, key: str) -> Any:
        self._record_metrics(key, get_calls=1)
        if not self._should_attempt_redis():
            self._record_metrics(key, get_fallbacks=1, cooldown_skips=1)
            return self._fallback.get(key)
        self._record_metrics(key, get_remote_attempts=1)
        try:
            raw = self._client.get(self._prefixed(key))
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(key, get_errors=1, get_fallbacks=1)
            self._log_remote_error("get", key, exc)
            return self._fallback.get(key)
        if raw is None:
            self._remember_success()
            self._record_metrics(key, get_misses=1)
            return None
        self._remember_success()
        self._record_metrics(key, get_hits=1, get_payload_bytes=_text_payload_bytes(raw))
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        self._record_metrics(key, set_calls=1)
        if not self._should_attempt_redis():
            self._record_metrics(key, set_fallbacks=1, cooldown_skips=1)
            self._fallback.set(key, value, ttl_seconds=ttl_seconds)
            return
        try:
            payload = self._serialize(value)
        except (TypeError, ValueError) as exc:
            logger.debug("skipping non-serialisable cache value for %s: %s", key, exc)
            self._record_metrics(key, set_fallbacks=1)
            self._fallback.set(key, value, ttl_seconds=ttl_seconds)
            return
        payload_bytes = _text_payload_bytes(payload)
        self._record_metrics(
            key,
            set_remote_attempts=1,
            set_attempted_payload_bytes=payload_bytes,
        )
        try:
            if ttl_seconds and ttl_seconds > 0:
                self._client.set(self._prefixed(key), payload, ex=int(ttl_seconds))
            else:
                self._client.set(self._prefixed(key), payload)
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(key, set_errors=1, set_fallbacks=1)
            self._log_remote_error("set", key, exc)
            self._fallback.set(key, value, ttl_seconds=ttl_seconds)
            return
        self._remember_success()
        self._record_metrics(key, set_successes=1, set_payload_bytes=payload_bytes)

    def set_if_absent(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        self._record_metrics(key, set_calls=1)
        if not self._should_attempt_redis():
            self._record_metrics(key, set_fallbacks=1, cooldown_skips=1)
            return self._fallback.set_if_absent(key, value, ttl_seconds=ttl_seconds)
        try:
            payload = self._serialize(value)
        except (TypeError, ValueError) as exc:
            logger.debug("skipping non-serialisable cache value for %s: %s", key, exc)
            self._record_metrics(key, set_fallbacks=1)
            return self._fallback.set_if_absent(key, value, ttl_seconds=ttl_seconds)
        payload_bytes = _text_payload_bytes(payload)
        self._record_metrics(
            key,
            set_remote_attempts=1,
            set_attempted_payload_bytes=payload_bytes,
        )
        try:
            inserted = bool(
                self._client.set(
                    self._prefixed(key),
                    payload,
                    ex=int(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else None,
                    nx=True,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(key, set_errors=1, set_fallbacks=1)
            self._log_remote_error("set_if_absent", key, exc)
            return self._fallback.set_if_absent(key, value, ttl_seconds=ttl_seconds)
        self._remember_success()
        self._record_metrics(
            key,
            set_successes=1,
            set_payload_bytes=payload_bytes if inserted else 0,
        )
        return inserted

    def delete(self, key: str) -> None:
        if not self._should_attempt_redis():
            self._record_metrics(key, other_fallbacks=1, cooldown_skips=1)
            self._fallback.delete(key)
            return
        self._record_metrics(key, other_remote_attempts=1)
        try:
            self._client.delete(self._prefixed(key))
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(key, other_remote_errors=1, other_fallbacks=1)
            self._log_remote_error("delete", key, exc)
        else:
            self._remember_success()
        self._fallback.delete(key)

    def renew_if_owner(self, key: str, owner: Any, *, ttl_seconds: int) -> bool:
        if not self._should_attempt_redis():
            self._record_metrics(key, other_fallbacks=1, cooldown_skips=1)
            return self._fallback.renew_if_owner(key, owner, ttl_seconds=ttl_seconds)
        self._record_metrics(key, other_remote_attempts=1)
        try:
            script = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
            )
            renewed = bool(
                self._client.eval(
                    script,
                    1,
                    self._prefixed(key),
                    self._serialize(owner),
                    max(1, int(ttl_seconds)),
                )
            )
            self._remember_success()
            if renewed:
                self._fallback.renew_if_owner(key, owner, ttl_seconds=ttl_seconds)
            return renewed
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(key, other_remote_errors=1, other_fallbacks=1)
            self._log_remote_error("lease_renew", key, exc)
            return self._fallback.renew_if_owner(key, owner, ttl_seconds=ttl_seconds)

    def release_if_owner(self, key: str, owner: Any) -> bool:
        if not self._should_attempt_redis():
            self._record_metrics(key, other_fallbacks=1, cooldown_skips=1)
            return self._fallback.release_if_owner(key, owner)
        self._record_metrics(key, other_remote_attempts=1)
        try:
            script = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end"
            )
            released = bool(self._client.eval(script, 1, self._prefixed(key), self._serialize(owner)))
            self._remember_success()
            self._fallback.release_if_owner(key, owner)
            return released
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(key, other_remote_errors=1, other_fallbacks=1)
            self._log_remote_error("lease_release", key, exc)
            return self._fallback.release_if_owner(key, owner)

    def delete_prefix(self, prefix: str) -> int:
        deleted = self._fallback.delete_prefix(prefix)
        if not self._should_attempt_redis():
            self._record_metrics(prefix, other_fallbacks=1, cooldown_skips=1)
            return deleted
        self._record_metrics(prefix, other_remote_attempts=1)
        try:
            full_prefix = self._prefixed(prefix)
            keys = list(self._client.scan_iter(match=f"{full_prefix}*"))
            if keys:
                deleted += int(self._client.delete(*keys) or 0)
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(prefix, other_remote_errors=1, other_fallbacks=1)
            self._log_remote_error("delete_prefix", prefix, exc)
        else:
            self._remember_success()
        return deleted

    @contextmanager
    def lock(self, key: str, *, timeout: float = 30.0) -> Iterator[bool]:
        token = uuid.uuid4().hex
        lock_key = f"lock:{key}"
        full_key = self._prefixed(lock_key)
        ttl_ms = int(max(1.0, timeout) * 1000)
        acquired = False
        token_bytes = _text_payload_bytes(token)
        self._record_metrics(lock_key, set_calls=1)
        if not self._should_attempt_redis():
            self._record_metrics(
                lock_key,
                set_fallbacks=1,
                cooldown_skips=1,
            )
            with self._fallback.lock(key, timeout=timeout) as got:
                yield got
            return
        self._record_metrics(
            lock_key,
            set_remote_attempts=1,
            set_attempted_payload_bytes=token_bytes,
        )
        try:
            acquired = bool(self._client.set(full_key, token, nx=True, px=ttl_ms))
            self._remember_success()
            self._record_metrics(
                lock_key,
                set_successes=1,
                set_payload_bytes=token_bytes if acquired else 0,
            )
        except Exception as exc:  # pragma: no cover - depends on Redis
            self._record_metrics(lock_key, set_errors=1, set_fallbacks=1)
            self._log_remote_error("lock", lock_key, exc)
            with self._fallback.lock(key, timeout=timeout) as got:
                yield got
            return

        try:
            yield acquired
        finally:
            if acquired and self._should_attempt_redis():
                self._record_metrics(lock_key, other_remote_attempts=1)
                try:
                    # Compare-and-delete via Lua so we never release a lock that
                    # has already expired and been re-acquired by another worker.
                    script = (
                        "if redis.call('get', KEYS[1]) == ARGV[1] then "
                        "return redis.call('del', KEYS[1]) else return 0 end"
                    )
                    self._client.eval(script, 1, full_key, token)
                    self._remember_success()
                except Exception as exc:  # pragma: no cover
                    self._record_metrics(lock_key, other_remote_errors=1)
                    self._log_remote_error("unlock", lock_key, exc)


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
    status = {
        "cache_backend": backend.name,
        "cache_requested_backend": requested,
        "cache_namespace": namespace,
        "cache_redis_fallback": requested == "redis" and (backend.name != "redis" or bool(redis_error)),
        "cache_last_error": redis_error or _LAST_BACKEND_ERROR,
    }
    circuit_snapshot = getattr(backend, "circuit_snapshot", None)
    if callable(circuit_snapshot):
        status["cache_redis_circuit"] = circuit_snapshot()
    metrics_snapshot = getattr(backend, "metrics_snapshot", None)
    if callable(metrics_snapshot):
        status["cache_redis_metrics"] = metrics_snapshot()
    return status


def get_json_dict(key: str) -> Optional[Dict[str, Any]]:
    """Return a cached JSON object, ignoring non-dict payloads."""

    try:
        cached_value = get_cache().get(key)
    except Exception:
        return None
    return dict(cached_value) if isinstance(cached_value, dict) else None


def set_json(key: str, value: Dict[str, Any], ttl_seconds: int) -> bool:
    """Store a JSON object and report whether the cache write was attempted."""

    try:
        get_cache().set(key, value, ttl_seconds=ttl_seconds)
    except Exception:
        return False
    return True


def get_or_set(
    key: str,
    *,
    ttl_seconds: int,
    factory: Callable[[], Any],
    single_flight: bool = False,
    lock_timeout: float = 30.0,
    cache_none: bool = False,
) -> tuple[Any, bool]:
    """Fetch a cached value or compute and store it.

    Returns ``(value, cache_hit)``. ``single_flight`` reuses the same lock
    semantics as ``cached`` so expensive upstream calls are only performed by
    one worker when Redis is enabled.
    """

    backend = get_cache()
    try:
        cached_value = backend.get(key)
    except Exception:
        cached_value = None
    if cached_value is not None:
        return cached_value, True

    def compute_and_store() -> Any:
        result = factory()
        if result is not None or cache_none:
            try:
                backend.set(key, result, ttl_seconds=ttl_seconds)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("cache set failed for %s: %s", key, exc)
        return result

    if not single_flight:
        return compute_and_store(), False

    with backend.lock(f"single-flight:{key}", timeout=lock_timeout) as acquired:
        if acquired:
            try:
                cached_value = backend.get(key)
            except Exception:
                cached_value = None
            if cached_value is not None:
                return cached_value, True
            return compute_and_store(), False

        deadline = time.time() + max(0.05, lock_timeout)
        while time.time() < deadline:
            time.sleep(0.05)
            try:
                cached_value = backend.get(key)
            except Exception:
                cached_value = None
            if cached_value is not None:
                return cached_value, True
        return compute_and_store(), False


def get_or_set_single_flight(
    key: str,
    *,
    ttl_seconds: int,
    factory: Callable[[], Any],
    lock_ttl_seconds: float,
    wait_timeout_seconds: float,
    poll_interval_seconds: float = 0.05,
    lock_key: Optional[str] = None,
    cache_usable: Optional[Callable[[Any], bool]] = None,
    allow_cached: bool = True,
) -> tuple[Any, str]:
    """Read or fill a cache entry without duplicate upstream work.

    Unlike :func:`get_or_set`, a waiter never invokes ``factory`` after its
    wait deadline.  It returns ``(None, "pending")`` instead, allowing the
    caller to serve stale data or an explicit unavailable response.  The lock
    TTL is intentionally independent from the waiter timeout: a slow owner
    keeps ownership long enough to finish, while callers only wait briefly.
    """

    backend = get_cache()

    def usable(value: Any) -> bool:
        if value is None:
            return False
        if cache_usable is None:
            return True
        try:
            return bool(cache_usable(value))
        except Exception:
            return False

    def read_usable() -> Any:
        try:
            value = backend.get(key)
        except Exception:
            return None
        return value if usable(value) else None

    if allow_cached:
        cached_value = read_usable()
        if cached_value is not None:
            return cached_value, "hit"

    owner = uuid.uuid4().hex
    actual_lock_key = lock_key or f"single-flight:{key}"
    acquired = False
    try:
        acquired = backend.set_if_absent(
            actual_lock_key,
            owner,
            ttl_seconds=max(1, int(math.ceil(float(lock_ttl_seconds)))),
        )
    except Exception:
        acquired = False

    if acquired:
        heartbeat_stop = threading.Event()
        ownership_lost = threading.Event()

        def renew_lock() -> None:
            interval = max(0.25, float(lock_ttl_seconds) / 3.0)
            while not heartbeat_stop.wait(interval):
                try:
                    renewed = backend.renew_if_owner(
                        actual_lock_key,
                        owner,
                        ttl_seconds=max(1, int(math.ceil(float(lock_ttl_seconds)))),
                    )
                except Exception:
                    renewed = False
                if not renewed:
                    ownership_lost.set()
                    return

        heartbeat_thread = threading.Thread(
            target=renew_lock,
            name=f"cache-single-flight-{actual_lock_key[-32:]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            if allow_cached:
                cached_value = read_usable()
                if cached_value is not None:
                    return cached_value, "hit"
            result = factory()
            if ownership_lost.is_set():
                replacement = read_usable()
                if replacement is not None:
                    return replacement, "coalesced"
                return None, "pending"
            if result is not None:
                try:
                    backend.set(key, result, ttl_seconds=max(1, int(ttl_seconds)))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("cache set failed for %s: %s", key, exc)
            return result, "miss"
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=max(0.25, float(lock_ttl_seconds) / 3.0))
            try:
                backend.release_if_owner(actual_lock_key, owner)
            except Exception:  # pragma: no cover - defensive
                logger.debug("single-flight unlock failed for %s", actual_lock_key)

    deadline = time.time() + max(0.0, float(wait_timeout_seconds))
    poll_seconds = max(0.01, float(poll_interval_seconds))
    while time.time() < deadline:
        time.sleep(poll_seconds)
        cached_value = read_usable()
        if cached_value is not None:
            return cached_value, "coalesced"
    return None, "pending"


def l1_l2_cached(
    local_cache: Dict[str, Dict[str, Any]],
    key: str,
    *,
    ttl_seconds: int,
    factory: Callable[[], Any],
    l2_key: Optional[str] = None,
    cache_none: bool = False,
) -> tuple[Any, bool]:
    """Small helper for process-local L1 + shared backend L2 caching.

    ``local_cache`` stores entries as ``{"_ts": ts, "data": value}``, matching
    the existing module-level cache pattern used throughout the app.
    """

    now = time.time()
    cached = local_cache.get(key)
    if cached and now - float(cached.get("_ts") or 0.0) < ttl_seconds:
        return cached.get("data"), True

    shared_key = l2_key or key
    value, shared_hit = get_or_set(
        shared_key,
        ttl_seconds=ttl_seconds,
        factory=factory,
        cache_none=cache_none,
    )
    if value is not None or cache_none:
        local_cache[key] = {"_ts": now, "data": value}
    return value, shared_hit


def cached_in_background(
    *,
    key_fn: Callable[..., str],
    ttl_seconds: int,
    stale_seconds: int,
    pending_fn: Callable[..., Dict[str, Any]],
) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
    """Serve the last response while a bounded, leased worker refreshes it.

    Cold reads return a pending payload. Companion keys keep the original
    prefix so existing prefix invalidation also removes stale/retry entries.
    Dates inside the source payload are never advanced by a cache read.
    """
    def decorator(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            key = key_fn(*args, **kwargs)
            backend = get_cache()
            fresh = backend.get(key)
            if fresh is not None:
                return copy.deepcopy(fresh)
            stale_key = f"{key}:stale"
            retry_key = f"{key}:retry"
            stale = backend.get(stale_key)

            def produce() -> Dict[str, Any]:
                started = time.perf_counter()
                result = func(*args, **kwargs)
                if stale is not None and (
                    result.get("status") == "unavailable"
                    or (stale.get("positions") and not result.get("positions") and result.get("status") != "ok")
                ):
                    raise RuntimeError("refresh returned unavailable; retaining last response")
                logging.getLogger("uvicorn.error").info(
                    "response_refresh key=%s duration_ms=%.1f", key,
                    (time.perf_counter() - started) * 1000,
                )
                return result

            def refresh() -> None:
                try:
                    result, state = get_or_set_single_flight(
                        key, ttl_seconds=ttl_seconds, factory=produce,
                        lock_ttl_seconds=120, wait_timeout_seconds=0,
                    )
                    if state == "miss" and result is not None and result.get("status") != "unavailable":
                        backend.set(stale_key, result, ttl_seconds=ttl_seconds + stale_seconds)
                except Exception:
                    logger.warning("response refresh failed for %s", key, exc_info=True)
                    # Do not hammer a failing upstream on every polling request.
                    backend.set(retry_key, True, ttl_seconds=30)
                finally:
                    with _RESPONSE_REFRESH_MUTEX:
                        _RESPONSE_REFRESH_ACTIVE.discard(key)
                    _RESPONSE_REFRESH_SLOTS.release()

            if not backend.get(retry_key):
                with _RESPONSE_REFRESH_MUTEX:
                    if key not in _RESPONSE_REFRESH_ACTIVE and _RESPONSE_REFRESH_SLOTS.acquire(blocking=False):
                        _RESPONSE_REFRESH_ACTIVE.add(key)
                        try:
                            threading.Thread(target=refresh, name="fund-response-refresh", daemon=True).start()
                        except Exception:
                            _RESPONSE_REFRESH_ACTIVE.discard(key)
                            _RESPONSE_REFRESH_SLOTS.release()
                            raise
            payload = copy.deepcopy(stale) if stale is not None else pending_fn(*args, **kwargs)
            payload["refresh_pending"] = True
            payload["response_cache_status"] = "stale" if stale is not None else "pending"
            return payload

        return wrapper
    return decorator


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
