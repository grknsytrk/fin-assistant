"""Small, dependency-free request limiting for public API entrypoints.

The service can run behind either FastAPI directly or Gradio's FastAPI server
on Hugging Face Spaces.  Keeping the middleware here makes the policy usable
by both hosts without depending on an optional edge product.  Limits are
per-process; production deployments should still apply their edge limit as a
second, shared layer.
"""

from __future__ import annotations

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


class RequestRateLimitMiddleware(BaseHTTPMiddleware):
    """Return a standards-friendly 429 before expensive public work starts."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        limit = _limit_for_request(request)
        allowed, retry_after = _LIMITER.consume(
            f"{request.method}:{request.url.path}:{_client_identity(request)}",
            limit=limit,
        )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "İstek limiti aşıldı. Lütfen daha sonra tekrar deneyin."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
