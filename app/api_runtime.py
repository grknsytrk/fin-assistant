from __future__ import annotations

import asyncio

import bisect

import concurrent.futures

import html

from html.parser import HTMLParser

import json

import logging

import math

import os

import re

import secrets

import sys

import threading

import time

import uuid

from contextlib import asynccontextmanager

from datetime import date, datetime, timedelta, timezone

from functools import lru_cache

from pathlib import Path

from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response

from fastapi.responses import JSONResponse, StreamingResponse

from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.gzip import GZipMiddleware

from pydantic import BaseModel, Field

from src.config import load_config

from src.nvidia_commentary import MAX_REQUEST_BYTES, PayloadValidationError, generate_overview_commentary

from app.cache import cache_status as _cache_status

from app.cache import cached as _cached_response

from app.cache import cached_in_background as _background_cached_response

from app.cache import get_cache as _get_cache

from app.cache import get_or_set_single_flight as _get_or_set_single_flight

from app.cache import get_json_dict as _cache_get_dict

from app.cache import set_json as _cache_set_json

from app.rate_limit import RequestRateLimitMiddleware

from app import kap_flow_store as _kap_flow_store

from app.database import (
    acquire_refresh_lease,
    close_postgres_pool,
    database_enabled,
    ensure_kap_flow_schema,
    ensure_json_cache_schema,
    ensure_refresh_lease_schema,
    get_refresh_lease,
    hydrate_json_cache,
    release_refresh_lease,
    renew_refresh_lease,
)

from app.reference_data import (
    ensure_reference_data_schema,
    get_instruments,
    get_instrument_names,
    sync_reference_data_from_caches,
    upsert_instrument,
)

from app.financial_snapshot_store import (
    ensure_financial_snapshot_schema,
    extract_financial_snapshot_from_kap_payload,
    get_financial_snapshots_map,
    sync_financial_snapshots_from_kap_cache,
    upsert_financial_snapshots,
    upsert_provider_fundamentals_bulk,
)

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise RuntimeError(
        "RAG-Fin backend requires Python 3.10-3.12. "
        f"Current interpreter is Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Use .external\\venv311\\Scripts\\python.exe or run .\\run.ps1."
    )

ROOT = Path(__file__).resolve().parents[1]

CONFIG = load_config(ROOT / "config.yaml")

LOGGER = logging.getLogger("uvicorn.error")

_ADMIN_REFRESH_TOKEN_ENV = "RAGFIN_ADMIN_REFRESH_TOKEN"

_MARKET_SWR_STALE_TTL_SECONDS = int(os.getenv("RAGFIN_MARKET_SWR_STALE_TTL_SECONDS", "120"))

_MARKET_SWR_LOCK_TTL_SECONDS = float(os.getenv("RAGFIN_MARKET_SWR_LOCK_TTL_SECONDS", "60"))

_MARKET_SWR_WAIT_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_MARKET_SWR_WAIT_TIMEOUT_SECONDS", "3"))

_MARKET_SWR_REVALIDATION_LEASE_TTL_SECONDS = int(
    os.getenv("RAGFIN_MARKET_SWR_REVALIDATION_LEASE_TTL_SECONDS", "90")
)

_MARKET_SWR_REVALIDATION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("RAGFIN_MARKET_SWR_REVALIDATION_WORKERS", "2"))),
    thread_name_prefix="market-cache-revalidate",
)

def _truthy_env(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}

def _require_admin_refresh_access(request: Request) -> None:
    """Authenticate private refresh endpoints without exposing a frontend token."""

    expected = str(os.getenv(_ADMIN_REFRESH_TOKEN_ENV) or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Admin yenileme secret'i sunucuda tanımlı değil.")

    authorization = str(request.headers.get("authorization") or "").strip()
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip() or not secrets.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="Yetkisiz admin isteği.")

def _shared_cache_get_dict(key: str) -> Optional[Dict[str, Any]]:
    return _cache_get_dict(key)

def _shared_cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    if isinstance(value, dict):
        _cache_set_json(key, value, ttl_seconds=ttl_seconds)

def _swr_cache_entry(payload: Dict[str, Any], *, fresh_ttl_seconds: int, stale_ttl_seconds: int) -> Dict[str, Any]:
    """Wrap a response with explicit fresh and stale deadlines.

    Redis TTL keeps the entry only until the stale deadline.  A caller can
    therefore keep serving the last known-good response while one shared
    worker refreshes it, without treating a provider failure as a cache miss.
    """

    now = time.time()
    fresh_ttl = max(1, int(fresh_ttl_seconds))
    stale_ttl = max(fresh_ttl, int(stale_ttl_seconds))
    return {
        "payload": payload,
        "fresh_until": now + fresh_ttl,
        "stale_until": now + stale_ttl,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }

def _swr_entry_is_fresh(entry: Any) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return False
    try:
        return float(entry.get("fresh_until") or 0.0) > time.time()
    except (TypeError, ValueError):
        return False

def _swr_entry_is_stale(entry: Any) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return False
    try:
        return float(entry.get("stale_until") or 0.0) > time.time()
    except (TypeError, ValueError):
        return False

def _swr_entry_payload(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return None
    return dict(entry["payload"])

def _schedule_swr_revalidation(
    *,
    cache_key: str,
    fresh_ttl_seconds: int,
    stale_ttl_seconds: int,
    factory: Callable[[], Optional[Dict[str, Any]]],
    coordination_key: Optional[str] = None,
) -> bool:
    """Queue at most one cross-worker best-effort cache refresh.

    The small lease prevents request storms from filling this process's
    executor.  The actual refresh is still protected by the heartbeat-backed
    single-flight lock, so a lease expiry cannot create duplicate upstream
    work.
    """

    backend = _get_cache()
    owner = uuid.uuid4().hex
    lease_key = f"swr-revalidate:{coordination_key or cache_key}"
    try:
        acquired = backend.set_if_absent(
            lease_key,
            owner,
            ttl_seconds=max(1, _MARKET_SWR_REVALIDATION_LEASE_TTL_SECONDS),
        )
    except Exception:
        return False
    if not acquired:
        # Another worker already owns the queued revalidation; callers should
        # still present the result as pending rather than scheduling a storm.
        return True

    def refresh() -> None:
        def build_entry() -> Optional[Dict[str, Any]]:
            payload = factory()
            if not isinstance(payload, dict):
                return None
            return _swr_cache_entry(
                payload,
                fresh_ttl_seconds=fresh_ttl_seconds,
                stale_ttl_seconds=stale_ttl_seconds,
            )

        try:
            _get_or_set_single_flight(
                cache_key,
                ttl_seconds=max(1, stale_ttl_seconds),
                factory=build_entry,
                lock_key=f"single-flight:{coordination_key or cache_key}",
                lock_ttl_seconds=_MARKET_SWR_LOCK_TTL_SECONDS,
                wait_timeout_seconds=_MARKET_SWR_WAIT_TIMEOUT_SECONDS,
                cache_usable=lambda _entry: False,
                allow_cached=False,
            )
        except Exception:
            LOGGER.warning("market cache revalidation failed for %s", cache_key, exc_info=True)
        finally:
            try:
                backend.release_if_owner(lease_key, owner)
            except Exception:
                LOGGER.debug("market cache revalidation unlock failed for %s", cache_key)

    try:
        _MARKET_SWR_REVALIDATION_EXECUTOR.submit(refresh)
    except Exception:
        try:
            backend.release_if_owner(lease_key, owner)
        except Exception:
            pass
        return False
    return True

def _shared_swr_payload(
    *,
    cache_key: str,
    factory: Callable[[], Optional[Dict[str, Any]]],
    fresh_ttl_seconds: int,
    stale_ttl_seconds: int,
    local_cache: Optional[Dict[str, Any]] = None,
    local_key: Optional[str] = None,
    force_revalidate: bool = False,
) -> tuple[Optional[Dict[str, Any]], str, bool, bool]:
    """Read a shared stale-while-revalidate cache without public bypasses.

    Returns ``payload, status, stale, refresh_pending``.  A public
    ``refresh=true`` is only a revalidation request: it never performs an
    additional synchronous upstream fetch while usable data already exists.
    """

    local_entry = local_cache.get(local_key) if local_cache is not None and local_key else None
    shared_entry = _shared_cache_get_dict(cache_key)

    if _swr_entry_is_fresh(local_entry):
        pending = _schedule_swr_revalidation(
            cache_key=cache_key,
            fresh_ttl_seconds=fresh_ttl_seconds,
            stale_ttl_seconds=stale_ttl_seconds,
            factory=factory,
        ) if force_revalidate else False
        return _swr_entry_payload(local_entry), "local_hit", False, pending

    if _swr_entry_is_fresh(shared_entry):
        if local_cache is not None and local_key:
            local_cache[local_key] = shared_entry
        pending = _schedule_swr_revalidation(
            cache_key=cache_key,
            fresh_ttl_seconds=fresh_ttl_seconds,
            stale_ttl_seconds=stale_ttl_seconds,
            factory=factory,
        ) if force_revalidate else False
        return _swr_entry_payload(shared_entry), "shared_hit", False, pending

    stale_entry = shared_entry if _swr_entry_is_stale(shared_entry) else local_entry
    if _swr_entry_is_stale(stale_entry):
        if local_cache is not None and local_key:
            local_cache[local_key] = stale_entry
        pending = _schedule_swr_revalidation(
            cache_key=cache_key,
            fresh_ttl_seconds=fresh_ttl_seconds,
            stale_ttl_seconds=stale_ttl_seconds,
            factory=factory,
        )
        return _swr_entry_payload(stale_entry), "stale", True, pending

    def build_entry() -> Optional[Dict[str, Any]]:
        built = factory()
        if not isinstance(built, dict):
            return None
        return _swr_cache_entry(
            built,
            fresh_ttl_seconds=fresh_ttl_seconds,
            stale_ttl_seconds=stale_ttl_seconds,
        )

    entry, status = _get_or_set_single_flight(
        cache_key,
        ttl_seconds=max(1, stale_ttl_seconds),
        factory=build_entry,
        lock_key=f"single-flight:{cache_key}",
        lock_ttl_seconds=_MARKET_SWR_LOCK_TTL_SECONDS,
        wait_timeout_seconds=_MARKET_SWR_WAIT_TIMEOUT_SECONDS,
        cache_usable=_swr_entry_is_fresh,
    )
    payload = _swr_entry_payload(entry)
    if payload is not None and local_cache is not None and local_key:
        local_cache[local_key] = entry
    return payload, status, False, status == "pending"

def _with_market_cache_metadata(
    payload: Dict[str, Any],
    *,
    cache_status: str,
    stale: bool,
    refresh_pending: bool,
) -> Dict[str, Any]:
    out = dict(payload)
    out["cache_status"] = cache_status
    out["stale"] = stale
    out["refresh_pending"] = refresh_pending
    return out

_DEBUG_LOG_PATH = Path("debug-0cbd9f.log")

_DEBUG_SESSION_ID = "0cbd9f"

_DEBUG_RUN_ID = "market-flow-debug-v1"

def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": _DEBUG_RUN_ID,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass



def _empty_logo_payload() -> Dict[str, Optional[str]]:
    return {"logo_url": None, "logo_source": None}


__all__ = [name for name in globals() if not name.startswith("__")]
