"""Small, dependency-free request limiting for public API entrypoints.

The service can run behind either FastAPI directly or Gradio's FastAPI server
on Hugging Face Spaces.  Keeping the middleware here makes the policy usable
by both hosts without depending on an optional edge product.  Limits are
per-process; production deployments should still apply their edge limit as a
second, shared layer.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Deque, Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(str(os.getenv(name, default)).strip()))
    except (TypeError, ValueError):
        return default


class SlidingWindowRateLimiter:
    """Thread-safe, in-memory sliding-window limiter.

    The implementation deliberately has a very small surface: its state is
    local to the worker, while the public policy remains configurable through
    environment variables.  That prevents a missing Redis instance from
    silently disabling the basic abuse protection.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)

    def consume(self, key: str, *, limit: int, window_seconds: float = 60.0) -> Tuple[bool, int]:
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(math.ceil(bucket[0] + window_seconds - now)))
                return False, retry_after
            bucket.append(now)
            return True, 0


_LIMITER = SlidingWindowRateLimiter()


class SharedRedisSlidingWindowRateLimiter:
    """Use the configured Redis cache as a cross-worker rate-limit store.

    The local limiter remains the fail-safe path: an unavailable cache must
    never silently remove public request protection.  Redis is accessed only
    when the application's cache backend is already healthy and configured as
    Redis, so this class introduces no second connection configuration.
    """

    _SCRIPT = """
local now = tonumber(ARGV[1])
local cutoff = now - tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, cutoff)
local count = redis.call('ZCARD', KEYS[1])
if count >= limit then
  local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  local retry_after = 1
  if oldest[2] then
    retry_after = math.max(1, math.ceil(tonumber(oldest[2]) + tonumber(ARGV[2]) - now))
  end
  return {0, retry_after}
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('EXPIRE', KEYS[1], math.max(1, math.ceil(tonumber(ARGV[2])) + 1))
return {1, 0}
"""

    def consume(self, key: str, *, limit: int, window_seconds: float = 60.0) -> Tuple[bool, int] | None:
        if limit <= 0:
            return True, 0
        enabled = str(os.getenv("RAGFIN_RATE_LIMIT_SHARED_REDIS", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        try:
            from app.cache import get_cache

            backend = get_cache()
            if getattr(backend, "name", "") != "redis":
                return None
            client = getattr(backend, "_client", None)
            prefix = getattr(backend, "_prefixed", None)
            if client is None or not callable(prefix):
                return None
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            now = time.time()
            result = client.eval(
                self._SCRIPT,
                1,
                prefix(f"rate-limit:v1:{digest}"),
                str(now),
                str(max(1.0, window_seconds)),
                str(limit),
                f"{now:.9f}:{threading.get_ident()}",
            )
            allowed = bool(int(result[0]))
            retry_after = max(0, int(result[1]))
            return allowed, retry_after
        except Exception:
            return None


_SHARED_LIMITER = SharedRedisSlidingWindowRateLimiter()


def _consume_limit(key: str, *, limit: int) -> Tuple[bool, int]:
    shared_result = _SHARED_LIMITER.consume(key, limit=limit)
    if shared_result is not None:
        return shared_result
    return _LIMITER.consume(key, limit=limit)


def _client_identity(request: Request) -> str:
    """Use forwarded addresses only when an upstream proxy is trusted."""

    trust_proxy = str(os.getenv("RAGFIN_TRUST_PROXY_HEADERS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if trust_proxy:
        forwarded = str(request.headers.get("x-forwarded-for") or "").strip()
        if forwarded:
            return forwarded.split(",", 1)[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def _limit_for_request(request: Request) -> int:
    path = request.url.path.rstrip("/") or "/"
    if path.startswith("/admin/"):
        # Admin endpoints are authenticated separately.  Do not let a public
        # source address exhaust their operational refresh quota.
        return 0
    if path.endswith("/allocations/history"):
        return _positive_int_env("RAGFIN_RATE_LIMIT_ALLOCATION_HISTORY_PER_MINUTE", 12)
    if str(request.query_params.get("refresh") or "").strip().lower() in {"1", "true", "yes"}:
        return _positive_int_env("RAGFIN_RATE_LIMIT_REVALIDATE_PER_MINUTE", 24)
    return _positive_int_env("RAGFIN_RATE_LIMIT_PUBLIC_PER_MINUTE", 600)


def _shared_provider_limit_for_request(request: Request) -> int:
    """Bound expensive revalidations across clients when Redis is available."""

    path = request.url.path.rstrip("/") or "/"
    if path.startswith("/admin/"):
        return 0
    if path.endswith("/allocations/history"):
        return _positive_int_env("RAGFIN_RATE_LIMIT_ALLOCATION_HISTORY_GLOBAL_PER_MINUTE", 30)
    if str(request.query_params.get("refresh") or "").strip().lower() in {"1", "true", "yes"}:
        return _positive_int_env("RAGFIN_RATE_LIMIT_REVALIDATE_GLOBAL_PER_MINUTE", 60)
    return 0


class RequestRateLimitMiddleware(BaseHTTPMiddleware):
    """Return a standards-friendly 429 before expensive public work starts."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        limit = _limit_for_request(request)
        request_key = f"client:{request.method}:{request.url.path}:{_client_identity(request)}"
        allowed, retry_after = _consume_limit(request_key, limit=limit)
        provider_limit = _shared_provider_limit_for_request(request)
        if allowed and provider_limit > 0:
            allowed, retry_after = _consume_limit(
                f"provider:{request.method}:{request.url.path}",
                limit=provider_limit,
            )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "İstek limiti aşıldı. Lütfen daha sonra tekrar deneyin."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
