from __future__ import annotations

import asyncio
import concurrent.futures
import html
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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise RuntimeError(
        "RAG-Fin backend requires Python 3.10-3.12. "
        f"Current interpreter is Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Use .external\\venv311\\Scripts\\python.exe or run .\\run.ps1."
    )

from src.config import load_config
from src.nvidia_commentary import MAX_REQUEST_BYTES, PayloadValidationError, generate_overview_commentary
from app.cache import cache_status as _cache_status
from app.cache import cached as _cached_response
from app.cache import get_cache as _get_cache
from app.cache import get_or_set_single_flight as _get_or_set_single_flight
from app.cache import get_json_dict as _cache_get_dict
from app.cache import set_json as _cache_set_json
from app.rate_limit import RequestRateLimitMiddleware
from app.database import (
    acquire_refresh_lease,
    close_postgres_pool,
    database_enabled,
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

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")
LOGGER = logging.getLogger("uvicorn.error")
_FUND_COLLECTOR_TASK: Optional[asyncio.Task[None]] = None
_KAP_SNAPSHOT_RESPONSE_CACHE_TTL = int(os.getenv("RAGFIN_KAP_SNAPSHOT_RESPONSE_CACHE_TTL_SECONDS", "300"))
_KAP_SNAPSHOT_RESPONSE_STALE_TTL = int(
    os.getenv("RAGFIN_KAP_SNAPSHOT_RESPONSE_STALE_TTL_SECONDS", str(30 * 60))
)
_KAP_SNAPSHOT_RESPONSE_LOCK_TIMEOUT = float(os.getenv("RAGFIN_KAP_SNAPSHOT_RESPONSE_LOCK_TIMEOUT_SECONDS", "120"))
_FUND_YIELD_SUMMARY_CACHE_TTL = int(os.getenv("RAGFIN_FUND_YIELD_SUMMARY_CACHE_TTL_SECONDS", "900"))
_FUND_HOLDINGS_RESPONSE_CACHE_TTL = int(os.getenv("RAGFIN_FUND_HOLDINGS_RESPONSE_CACHE_TTL_SECONDS", str(60 * 60)))
_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT = float(os.getenv("RAGFIN_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT_SECONDS", "120"))
_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION = 7
_FUND_HOLDINGS_LIVE_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_FUND_HOLDINGS_LIVE_RESPONSE_CACHE_TTL_SECONDS", "10")
)
_FUND_HOLDINGS_LIVE_RESPONSE_LOCK_TTL = float(
    os.getenv("RAGFIN_FUND_HOLDINGS_LIVE_RESPONSE_LOCK_TTL_SECONDS", "30")
)
_FUND_HOLDINGS_LIVE_RESPONSE_WAIT_TIMEOUT = float(
    os.getenv("RAGFIN_FUND_HOLDINGS_LIVE_RESPONSE_WAIT_TIMEOUT_SECONDS", "2")
)
_FUND_HOLDING_SECTOR_MAP_CACHE_TTL = 6 * 60 * 60
_FUND_REFRESH_JOB_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_REFRESH_JOB_TTL_SECONDS", str(30 * 60)))
_FUND_REFRESH_MAX_LOOKBACK_DAYS = 14
_FUND_REFRESH_RESOURCE_KEY = "funds_snapshot"
_FUND_REFRESH_LEASE_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_REFRESH_LEASE_TTL_SECONDS", "90"))
_FUND_REFRESH_ACTIVE_KEY = "api:funds-refresh:active"
_FUND_REFRESH_HEARTBEAT_INTERVAL_SECONDS = float(
    os.getenv("RAGFIN_FUND_REFRESH_HEARTBEAT_INTERVAL_SECONDS", "20")
)
_FUND_REFRESH_HEARTBEAT_TIMEOUT_SECONDS = float(
    os.getenv("RAGFIN_FUND_REFRESH_HEARTBEAT_TIMEOUT_SECONDS", "120")
)
_FUND_REFRESH_MAX_RUNTIME_SECONDS = float(
    os.getenv("RAGFIN_FUND_REFRESH_MAX_RUNTIME_SECONDS", str(15 * 60))
)
_FUND_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="fund-snapshot-refresh",
)
_FUND_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="fund-history-backfill",
)
_FUND_ALLOCATION_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_WORKERS", "1"))),
    thread_name_prefix="fund-allocation-history-refresh",
)
_FUND_REFRESH_STATE_LOCK = threading.Lock()
_FUND_HISTORY_STATE_LOCK = threading.Lock()
_FUND_HISTORY_JOB_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_HISTORY_JOB_TTL_SECONDS", str(30 * 60)))
_FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS = int(
    os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS", str(30 * 60))
)
_FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS = int(
    os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS", "120")
)
_FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS = float(
    os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS", "30")
)
_FUND_HISTORY_WARMUP_DAYS = int(os.getenv("RAGFIN_FUNDS_HISTORY_WARMUP_DAYS", "366"))
_FUND_HISTORY_LEASE_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_HISTORY_LEASE_TTL_SECONDS", "120"))
_FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS = float(
    os.getenv("RAGFIN_FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS", "30")
)
_FUND_HISTORY_MAX_PHASES = int(os.getenv("RAGFIN_FUND_HISTORY_MAX_PHASES", "2"))
_FUND_HISTORY_KEY_VERSION = 2
_ADMIN_REFRESH_TOKEN_ENV = "RAGFIN_ADMIN_REFRESH_TOKEN"
_ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS = int(
    os.getenv("RAGFIN_ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS", "370")
)
_MARKET_UNIVERSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_UNIVERSE_CACHE_TTL_SECONDS", str(6 * 60 * 60))
)
_MARKET_QUOTES_FRESH_TTL = int(os.getenv("RAGFIN_MARKET_QUOTES_FRESH_TTL_SECONDS", "5"))
_MARKET_QUOTES_STALE_TTL = int(os.getenv("RAGFIN_MARKET_QUOTES_STALE_TTL_SECONDS", "120"))
_MARKET_QUOTES_LOCK_TTL_SECONDS = float(
    os.getenv("RAGFIN_MARKET_QUOTES_LOCK_TTL_SECONDS", "60")
)
_MARKET_QUOTES_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("RAGFIN_MARKET_QUOTES_WAIT_TIMEOUT_SECONDS", "8")
)
_MARKET_QUOTES_POLL_INTERVAL_SECONDS = float(
    os.getenv("RAGFIN_MARKET_QUOTES_POLL_INTERVAL_SECONDS", "0.05")
)
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


def _log_market_cache_event(
    *,
    endpoint: str,
    index: str,
    cache_status: str,
    upstream_called: bool,
    stale: bool,
    started_at: float,
) -> None:
    elapsed_ms = int(max(0.0, (time.perf_counter() - started_at) * 1000))
    LOGGER.info(
        "market_cache endpoint=%s index=%s cache_status=%s upstream_called=%s stale=%s elapsed_ms=%s",
        endpoint,
        index,
        cache_status,
        str(bool(upstream_called)).lower(),
        str(bool(stale)).lower(),
        elapsed_ms,
    )


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
) -> bool:
    """Queue at most one cross-worker best-effort cache refresh.

    The small lease prevents request storms from filling this process's
    executor.  The actual refresh is still protected by the heartbeat-backed
    single-flight lock, so a lease expiry cannot create duplicate upstream
    work.
    """

    backend = _get_cache()
    owner = uuid.uuid4().hex
    lease_key = f"swr-revalidate:{cache_key}"
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
                lock_key=f"single-flight:{cache_key}",
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


def _normalized_cache_text(value: Optional[str], *, max_length: int = 120) -> str:
    """Keep user-controlled response-cache keys bounded and canonical."""

    return " ".join(str(value or "").split()).casefold()[:max_length]


def _fund_listing_cache_key(
    *,
    q: Optional[str],
    fund_type: Optional[str],
    founder: Optional[str],
    manager: Optional[str],
    risk: Optional[str],
    sort: str,
    order: str,
) -> str:
    return (
        "api:funds:v3:"
        f"q={_normalized_cache_text(q, max_length=80)}|type={_normalized_cache_text(fund_type, max_length=32)}"
        f"|founder={_normalized_cache_text(founder)}|manager={_normalized_cache_text(manager)}"
        f"|risk={_normalized_cache_text(risk, max_length=32)}|sort={_normalized_cache_text(sort, max_length=32)}"
        f"|order={_normalized_cache_text(order, max_length=8)}"
    )


def _fund_refresh_job_key(job_id: str) -> str:
    return f"api:funds-refresh:job:{job_id}"


def _fund_refresh_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_fund_refresh_job(job_id: str) -> Optional[Dict[str, Any]]:
    job = _get_cache().get(_fund_refresh_job_key(job_id))
    return dict(job) if isinstance(job, dict) else None


def _set_fund_refresh_job(job: Dict[str, Any]) -> None:
    _get_cache().set(
        _fund_refresh_job_key(str(job["job_id"])),
        job,
        ttl_seconds=_FUND_REFRESH_JOB_TTL_SECONDS,
    )


def _update_fund_refresh_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    with _FUND_REFRESH_STATE_LOCK:
        job = _get_fund_refresh_job(job_id)
        if job is None:
            return None
        job.update(updates)
        _set_fund_refresh_job(job)
        return job


def _active_fund_refresh_job() -> Optional[Dict[str, Any]]:
    backend = _get_cache()
    marker = backend.get(_FUND_REFRESH_ACTIVE_KEY)
    active_id = marker.get("job_id") if isinstance(marker, dict) else marker
    lease: Optional[Dict[str, Any]] = None
    if not active_id and database_enabled():
        lease = get_refresh_lease(_FUND_REFRESH_RESOURCE_KEY)
        lease_until = lease.get("lease_until") if isinstance(lease, dict) else None
        try:
            lease_active = bool(lease_until and lease_until > datetime.now(timezone.utc))
        except TypeError:
            try:
                lease_active = datetime.fromisoformat(str(lease_until).replace("Z", "+00:00")) > datetime.now(timezone.utc)
            except (TypeError, ValueError):
                lease_active = False
        active_id = lease.get("job_id") if lease_active and isinstance(lease, dict) else None
    if not active_id:
        return None
    job = _get_fund_refresh_job(str(active_id))
    if job is None and lease and active_id:
        # Redis/job keys can disappear during a process restart while the
        # authoritative Postgres lease is still alive. Recreate a status
        # record and let the lease expire before another generation starts.
        job = {
            "job_id": str(active_id),
            "status": "running",
            "requested_at": _fund_refresh_now_iso(),
            "started_at": None,
            "finished_at": None,
            "as_of": None,
            "row_count": None,
            "error": "Yenileme worker'ı yeniden başlatıldı; mevcut lease bekleniyor.",
            "resolution_status": None,
            "snapshot_action": None,
            "generation": lease.get("generation"),
        }
        _set_fund_refresh_job(job)
    if job and job.get("status") in {"queued", "running"}:
        if job.get("status") == "running":
            requested_text = str(job.get("requested_at") or "").strip()
            try:
                requested_at = datetime.fromisoformat(requested_text.replace("Z", "+00:00"))
                runtime = (datetime.now(timezone.utc) - requested_at).total_seconds()
            except (TypeError, ValueError):
                runtime = 0.0
            if runtime > _FUND_REFRESH_MAX_RUNTIME_SECONDS:
                return job
            heartbeat_text = str(job.get("heartbeat_at") or "").strip()
            try:
                heartbeat_at = datetime.fromisoformat(heartbeat_text.replace("Z", "+00:00"))
                heartbeat_age = (datetime.now(timezone.utc) - heartbeat_at).total_seconds()
            except (TypeError, ValueError):
                heartbeat_age = float("inf")
            if heartbeat_age > _FUND_REFRESH_HEARTBEAT_TIMEOUT_SECONDS:
                return job
        return job
    return None


def _run_fund_refresh_job(job_id: str, lookback_days: int) -> None:
    initial_job = _get_fund_refresh_job(job_id) or {}
    owner_token = str(initial_job.get("owner_token") or job_id)
    generation = int(initial_job.get("generation") or 0)
    owner_marker: Any = {
        "job_id": job_id,
        "owner_token": owner_token,
        # Redis is only an advisory lease; the Postgres generation is kept
        # separately and checked during the fenced commit transaction.
        "generation": 0,
    }
    current_marker = _get_cache().get(_FUND_REFRESH_ACTIVE_KEY)
    if current_marker == job_id:
        owner_marker = job_id
    _update_fund_refresh_job(
        job_id,
        status="running",
        started_at=_fund_refresh_now_iso(),
        heartbeat_at=_fund_refresh_now_iso(),
    )
    backend = _get_cache()
    heartbeat_stop = threading.Event()
    lease_lost = threading.Event()

    def heartbeat() -> None:
        interval = max(5.0, _FUND_REFRESH_HEARTBEAT_INTERVAL_SECONDS)
        while not heartbeat_stop.wait(interval):
            current_job = _get_fund_refresh_job(job_id) or {}
            started_text = str(current_job.get("started_at") or current_job.get("requested_at") or "")
            try:
                runtime = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(started_text.replace("Z", "+00:00"))
                ).total_seconds()
            except (TypeError, ValueError):
                runtime = 0.0
            if runtime > _FUND_REFRESH_MAX_RUNTIME_SECONDS:
                lease_lost.set()
                _update_fund_refresh_job(
                    job_id,
                    heartbeat_at=_fund_refresh_now_iso(),
                    error="Fon yenilemesi maksimum çalışma süresine ulaştı; commit engellenecek.",
                )
                continue
            renewed = backend.renew_if_owner(
                _FUND_REFRESH_ACTIVE_KEY,
                owner_marker,
                ttl_seconds=_FUND_REFRESH_LEASE_TTL_SECONDS,
            )
            db_renewed = renew_refresh_lease(
                _FUND_REFRESH_RESOURCE_KEY,
                job_id=job_id,
                owner_token=owner_token,
                ttl_seconds=_FUND_REFRESH_LEASE_TTL_SECONDS,
            )
            if database_enabled() and not db_renewed:
                lease_lost.set()
            if renewed or not database_enabled():
                _update_fund_refresh_job(job_id, heartbeat_at=_fund_refresh_now_iso())

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"fund-refresh-heartbeat-{job_id[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        from app.fund_service import refresh_funds_snapshot

        bounded_lookback_days = min(_FUND_REFRESH_MAX_LOOKBACK_DAYS, max(1, int(lookback_days)))
        refresh_kwargs = {
            "lookback_days": bounded_lookback_days,
            "persist_reference_data": False,
            "backfill_daily_returns": False,
            "persist_snapshot": False,
        }
        while True:
            try:
                result = refresh_funds_snapshot(CONFIG.paths.processed_dir, **refresh_kwargs)
                break
            except TypeError as exc:
                unexpected = str(exc)
                if "persist_snapshot" in unexpected and "persist_snapshot" in refresh_kwargs:
                    refresh_kwargs.pop("persist_snapshot")
                    continue
                if "persist_reference_data" in unexpected and "persist_reference_data" in refresh_kwargs:
                    refresh_kwargs.pop("persist_reference_data")
                    refresh_kwargs.pop("backfill_daily_returns", None)
                    continue
                if "lookback_days" in unexpected and len(refresh_kwargs) == 1:
                    result = refresh_funds_snapshot(CONFIG.paths.processed_dir)
                    break
                raise
        if not isinstance(result, dict):
            raise RuntimeError("Fon yenilemesi geçerli bir sonuç döndürmedi.")
        rows = list(result.get("rows") or []) if isinstance(result, dict) else []
        resolution_status = str(result.get("resolution_status") or ("available" if rows else "upstream_unavailable"))
        if lease_lost.is_set():
            _update_fund_refresh_job(
                job_id,
                status="superseded",
                finished_at=_fund_refresh_now_iso(),
                error="Fon yenileme lease'i başka bir generation tarafından devralındı.",
                resolution_status=resolution_status,
            )
            return
        if resolution_status == "upstream_unavailable":
            warnings = list(result.get("warnings") or [])
            _update_fund_refresh_job(
                job_id,
                status="failed",
                finished_at=_fund_refresh_now_iso(),
                resolution_status=resolution_status,
                snapshot_action="retained_existing",
                error=warnings[0] if warnings else "TEFAS güvenilir cevap döndürmedi; mevcut snapshot korundu.",
            )
            return
        from app.fund_service import commit_funds_snapshot

        result = dict(result)
        result["snapshot_generation"] = generation
        result_meta = dict(result.get("source_metadata") or {})
        result_meta["snapshot_generation"] = generation
        result["source_metadata"] = result_meta
        committed = commit_funds_snapshot(
            CONFIG.paths.processed_dir,
            result,
            job_id=job_id,
            owner_token=owner_token,
            generation=generation,
        )
        if not committed:
            _update_fund_refresh_job(
                job_id,
                status="superseded",
                finished_at=_fund_refresh_now_iso(),
                resolution_status=resolution_status,
                error="Snapshot commit'i lease/generation kontrolünden geçmedi.",
            )
            return
        _invalidate_fund_response_cache()
        _update_fund_refresh_job(
            job_id,
            status="succeeded",
            finished_at=_fund_refresh_now_iso(),
            as_of=result.get("as_of"),
            row_count=len(rows),
            resolution_status=resolution_status,
            resolved_as_of=(result.get("source_metadata") or {}).get("resolved_as_of"),
            snapshot_action=(result.get("source_metadata") or {}).get("snapshot_action") or result.get("snapshot_action"),
            error=None,
        )
    except Exception as exc:
        logging.getLogger("uvicorn.error").exception("fund snapshot refresh job failed")
        _update_fund_refresh_job(job_id, status="failed", finished_at=_fund_refresh_now_iso(), error=str(exc))
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        backend.release_if_owner(_FUND_REFRESH_ACTIVE_KEY, owner_marker)
        release_refresh_lease(
            _FUND_REFRESH_RESOURCE_KEY,
            job_id=job_id,
            owner_token=owner_token,
        )


def _start_fund_refresh_job(lookback_days: int) -> Dict[str, Any]:
    existing = _active_fund_refresh_job()
    if existing is not None:
        return existing

    backend = _get_cache()
    job_id = uuid.uuid4().hex
    owner_token = uuid.uuid4().hex
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "requested_at": _fund_refresh_now_iso(),
        "started_at": None,
        "finished_at": None,
        "as_of": None,
        "row_count": None,
        "error": None,
        "resolution_status": None,
        "snapshot_action": None,
        "owner_token": owner_token,
        "generation": None,
    }
    owner_marker = {"job_id": job_id, "owner_token": owner_token, "generation": 0}
    if not backend.set_if_absent(
        _FUND_REFRESH_ACTIVE_KEY,
        owner_marker,
        ttl_seconds=_FUND_REFRESH_LEASE_TTL_SECONDS,
    ):
        existing = _active_fund_refresh_job()
        if existing is not None:
            return existing
        job.update(
            status="failed",
            finished_at=_fund_refresh_now_iso(),
            error="Fon yenilemesi başka bir worker tarafından devralındı; tekrar deneyin.",
        )
        _set_fund_refresh_job(job)
        return job

    lease = acquire_refresh_lease(
        _FUND_REFRESH_RESOURCE_KEY,
        job_id=job_id,
        owner_token=owner_token,
        ttl_seconds=_FUND_REFRESH_LEASE_TTL_SECONDS,
    )
    if database_enabled() and not lease:
        backend.release_if_owner(_FUND_REFRESH_ACTIVE_KEY, owner_marker)
        existing = _active_fund_refresh_job()
        if existing is not None:
            return existing
        job.update(
            status="failed",
            finished_at=_fund_refresh_now_iso(),
            error="Fon yenileme Postgres lease'i alınamadı; mevcut snapshot korundu.",
        )
        _set_fund_refresh_job(job)
        return job
    if lease:
        generation = int(lease.get("generation") or 0)
        job["generation"] = generation
        owner_marker["generation"] = generation
    job["lease_until"] = str(lease.get("lease_until")) if lease else None
    _set_fund_refresh_job(job)
    _FUND_REFRESH_EXECUTOR.submit(_run_fund_refresh_job, job_id, lookback_days)
    return job


async def _fund_price_collector_loop() -> None:
    from app.fund_service import collect_daily_fund_prices

    startup_delay = float(os.getenv("RAGFIN_FUND_COLLECTOR_STARTUP_DELAY_SECONDS", "60"))
    interval = float(os.getenv("RAGFIN_FUND_COLLECTOR_INTERVAL_SECONDS", str(24 * 60 * 60)))
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)
    while True:
        try:
            result = await asyncio.to_thread(collect_daily_fund_prices, CONFIG.paths.processed_dir)
            LOGGER.info(
                "fund price collector completed: valid=%s skipped=%s source=%s",
                result.get("valid_point_count"),
                result.get("skipped_point_count"),
                result.get("source"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("fund price collector failed")
        await asyncio.sleep(max(60.0, interval))


async def _start_fund_price_collector() -> None:
    global _FUND_COLLECTOR_TASK
    if not _truthy_env("RAGFIN_FUND_COLLECTOR_ENABLED", "1"):
        return
    if _FUND_COLLECTOR_TASK and not _FUND_COLLECTOR_TASK.done():
        return
    _FUND_COLLECTOR_TASK = asyncio.create_task(_fund_price_collector_loop())


async def _stop_fund_price_collector() -> None:
    global _FUND_COLLECTOR_TASK
    task = _FUND_COLLECTOR_TASK
    _FUND_COLLECTOR_TASK = None
    if not task:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def bootstrap_application_storage() -> None:
    """Prepare persistent schemas before serving the first request.

    Gradio Spaces include this router without running the FastAPI app
    lifespan, so the HF entrypoint calls this function explicitly as well.
    Keeping all DDL here prevents normal API requests from racing on schema
    creation or waiting on Supabase statement timeouts.
    """

    ensure_json_cache_schema()
    ensure_refresh_lease_schema()
    ensure_reference_data_schema(CONFIG.paths.processed_dir)
    from app.fund_service import ensure_fund_prices_schema

    ensure_fund_prices_schema(CONFIG.paths.processed_dir)
    hydrate_json_cache(CONFIG.paths.processed_dir)
    sync_reference_data_from_caches(CONFIG.paths.processed_dir)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        await asyncio.to_thread(bootstrap_application_storage)
    except Exception:
        LOGGER.exception("application storage bootstrap failed")
        if database_enabled():
            raise
    await _start_fund_price_collector()
    try:
        yield
    finally:
        await _stop_fund_price_collector()
        close_postgres_pool()


app = FastAPI(title="RAG-Fin API", version="0.10.0", lifespan=_lifespan)

# region agent log helpers
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


# endregion


def _cors_allow_origins() -> List[str]:
    raw = os.getenv("RAGFIN_CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def _cors_allow_origin_regex() -> str:
    raw = os.getenv("RAGFIN_CORS_ALLOW_ORIGIN_REGEX", "").strip()
    if raw:
        return raw
    # Allow local dev hosts on any port (Vite may auto-switch ports such as 5174/5175).
    return r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$"


def configure_http_middleware(target: FastAPI) -> None:
    """Install the public HTTP policy on every production API host.

    Spaces serves this router from Gradio's FastAPI application rather than
    from ``app`` itself, so this function is also called by ``hf_entrypoint``.
    """

    target.add_middleware(RequestRateLimitMiddleware)
    # Middleware is evaluated in reverse registration order. CORS therefore
    # wraps the limiter too, so browser clients can read a 429/Retry-After.
    target.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_allow_origins(),
        allow_origin_regex=_cors_allow_origin_regex(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # The fund catalogue contains every visible TEFAS fund and is intentionally
    # returned in one response for client-side filtering.  Compressing responses
    # keeps that catalogue cheap to transfer without changing its API contract.
    target.add_middleware(GZipMiddleware, minimum_size=1024)


configure_http_middleware(app)


class MarketComparisonHistoryAsset(BaseModel):
    id: Optional[str] = None
    kind: Literal["fund", "stock", "index", "fx"]
    symbol: str = Field(..., min_length=1, max_length=32)
    label: Optional[str] = Field(default=None, max_length=160)


class MarketComparisonHistoryRequest(BaseModel):
    assets: List[MarketComparisonHistoryAsset] = Field(..., min_length=1, max_length=8)
    start_date: date
    end_date: date


def _quarter_label_sort_key(label: str) -> tuple[int, int, str]:
    normalized = str(label or "").strip().upper()
    if not normalized:
        return (0, 0, "")
    quarter_match = re.match(r"^(\d{4})Q([1-4])$", normalized)
    if quarter_match:
        return (int(quarter_match.group(1)), int(quarter_match.group(2)) * 3, normalized)
    period_match = re.match(r"^(\d{4})[/-](\d{1,2})$", normalized)
    if period_match:
        return (int(period_match.group(1)), int(period_match.group(2)), normalized)
    return (0, 0, normalized)


def _latest_quarter_label(quarters: List[str]) -> Optional[str]:
    candidates = [str(item or "").strip().upper() for item in quarters if str(item or "").strip()]
    if not candidates:
        return None
    return max(candidates, key=_quarter_label_sort_key)


_KAP_MARKET_METADATA_CACHE: Dict[str, Any] = {}
_KAP_MARKET_METADATA_CACHE_TTL = 6 * 60 * 60
_KAP_COMPANIES_RESPONSE_CACHE: Dict[str, Any] = {}
_KAP_COMPANIES_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_KAP_COMPANIES_RESPONSE_CACHE_TTL_SECONDS", str(60 * 60))
)
_KAP_COMPANIES_RESPONSE_CACHE_KEY = "api:kap:companies:v2"


def _load_cached_kap_market_metadata(cache_dir: Path, symbol: str) -> Dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return {}
    cache_file = cache_dir / f"{normalized_symbol}.json"
    if not cache_file.exists():
        return {}
    cache_key = ""
    signature = None
    shared_key = None
    try:
        stat = cache_file.stat()
        cache_key = str(cache_file.resolve())
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = _KAP_MARKET_METADATA_CACHE.get(cache_key)
        if cached and cached.get("signature") == signature:
            return dict(cached.get("data") or {})
        shared_key = f"api:kap:market-metadata:{normalized_symbol}:mtime={stat.st_mtime_ns}:size={stat.st_size}:v1"
        shared_cached = _shared_cache_get_dict(shared_key)
        if shared_cached is not None:
            metadata = dict(shared_cached)
            _KAP_MARKET_METADATA_CACHE[cache_key] = {
                "signature": signature,
                "data": metadata,
            }
            return metadata
    except Exception:
        cache_key = ""
        signature = None
        shared_key = None

    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    company_title = str(payload.get("company_title") or "").strip()
    company_code = str(payload.get("company") or payload.get("stock_code") or symbol).strip().upper()

    quarters_raw = payload.get("quarters")
    quarters = [
        str(row.get("quarter") or "").strip().upper()
        for row in (quarters_raw or [])
        if isinstance(row, dict) and str(row.get("quarter") or "").strip()
    ]
    quarter_rows = [row for row in (quarters_raw or []) if isinstance(row, dict)]
    latest_row = max(
        quarter_rows,
        key=lambda row: _quarter_label_sort_key(str(row.get("quarter") or "").strip().upper()),
        default=None,
    )
    shares_outstanding = None
    share_source = None
    if latest_row:
        for metric_key in ("odenmis_sermaye", "cikarilmis_sermaye"):
            for field in ("metrics", "metrics_ytd"):
                container = latest_row.get(field)
                if not isinstance(container, dict):
                    continue
                metric = container.get(metric_key)
                value = metric.get("value") if isinstance(metric, dict) else metric
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                    continue
                shares_outstanding = float(value)
                share_source = metric_key
                break
            if shares_outstanding is not None:
                break
    metadata = {
        "latest_quarter": _latest_quarter_label(quarters),
        "has_kap_cache": True,
        "shares_outstanding": shares_outstanding,
        "share_source": share_source,
        "company_title": company_title or None,
        "company": company_code or normalized_symbol,
    }
    if cache_key and signature:
        _KAP_MARKET_METADATA_CACHE[cache_key] = {
            "signature": signature,
            "data": metadata,
        }
        if shared_key:
            _shared_cache_set(shared_key, metadata, ttl_seconds=_KAP_MARKET_METADATA_CACHE_TTL)
    return metadata


def _stock_reference_record_from_kap_payload(symbol: str, payload: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    normalized = str(symbol or "").strip().upper()
    stock_code = str(payload.get("stock_code") or payload.get("company") or normalized).strip().upper()
    title = str(payload.get("company_title") or payload.get("title") or payload.get("companyName") or "").strip()
    member_oid = str(payload.get("member_oid") or payload.get("mkk_member_oid") or "").strip()
    return {
        "kind": "stock",
        "symbol": stock_code or normalized,
        "name": title or None,
        "short_name": stock_code or normalized,
        "source": source,
        "source_id": member_oid or None,
        "logo_url": f"https://www.kap.org.tr/tr/api/member/logo/{member_oid}" if member_oid else None,
        "logo_source": "kap" if member_oid else None,
        "as_of": str(payload.get("fetched_at") or "").strip() or None,
        "aliases": [normalized] if normalized and normalized != stock_code else [],
        "metadata": {
            "latest_quarter": _latest_quarter_label(
                [
                    str(row.get("quarter") or "").strip().upper()
                    for row in (payload.get("quarters") or [])
                    if isinstance(row, dict)
                ]
            ),
            "source_url": payload.get("source_url"),
        },
    }


def _upsert_stock_reference_from_kap_payload(symbol: str, payload: Dict[str, Any], *, source: str = "kap") -> None:
    if not isinstance(payload, dict) or not payload.get("ok", True):
        return
    record = _stock_reference_record_from_kap_payload(symbol, payload, source=source)
    if not record.get("symbol"):
        return
    try:
        upsert_instrument(CONFIG.paths.processed_dir, **record)
    except Exception:
        LOGGER.debug("stock reference upsert failed for %s", symbol, exc_info=True)


def _positive_float(raw: Any) -> Optional[float]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value <= 0:
        return None
    return value


def _market_cap_from_quote_and_meta(
    quote: Dict[str, Any],
    cached_meta: Dict[str, Any],
    basic_summary: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    price = quote.get("price")
    shares = cached_meta.get("shares_outstanding")
    price_value = _positive_float(price)
    shares_value = _positive_float(shares)
    if price_value is not None and shares_value is not None:
        return price_value * shares_value

    quote_market_cap = _positive_float(quote.get("market_cap"))
    if quote_market_cap is not None:
        return quote_market_cap

    summary = basic_summary or {}
    market_cap = _positive_float(summary.get("market_cap"))
    if market_cap is not None:
        return market_cap

    summary_shares = _positive_float(summary.get("shares_outstanding"))
    if price_value is not None and summary_shares is not None:
        return price_value * summary_shares
    return None


_UNIVERSE_CACHE: Dict[str, Any] = {}


def _market_universe_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    started_at = time.perf_counter()
    normalized_index = _normalize_stock_index(index_name)
    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key=f"api:market:universe:{normalized_index}:v3",
        factory=lambda: _build_market_universe_payload(index_name=normalized_index, force_refresh=True),
        fresh_ttl_seconds=_MARKET_UNIVERSE_CACHE_TTL,
        stale_ttl_seconds=max(_MARKET_SWR_STALE_TTL_SECONDS, _MARKET_UNIVERSE_CACHE_TTL * 2),
        local_cache=_UNIVERSE_CACHE,
        local_key=f"payload:{normalized_index}",
        force_revalidate=force_refresh,
    )
    if payload is None:
        raise HTTPException(status_code=503, detail="Piyasa evreni yenileniyor. Lütfen kısa süre sonra tekrar deneyin.")
    _log_market_cache_event(
        endpoint="market/universe",
        index=normalized_index,
        cache_status=cache_status,
        upstream_called=cache_status == "miss",
        stale=stale,
        started_at=started_at,
    )
    return _with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _build_market_universe_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    from app.kap_service import get_bist_index_universe

    normalized_index = _normalize_stock_index(index_name)
    universe = get_bist_index_universe(normalized_index, force_refresh=force_refresh)
    symbols = list(universe.get("symbols") or [])
    try:
        bist_all_count = (
            int(universe.get("count") or 0)
            if normalized_index == "XUTUM"
            else int(get_bist_index_universe("XUTUM").get("count") or 0)
        )
    except Exception:
        bist_all_count = int(universe.get("count") or len(symbols)) if normalized_index == "XUTUM" else 0
    try:
        instrument_map = get_instruments(CONFIG.paths.processed_dir, "stock", symbols)
    except Exception as exc:
        LOGGER.warning("reference instrument batch lookup failed: %s", exc)
        instrument_map = {}
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    rows: List[Dict[str, Any]] = []
    kap_cache_count = 0

    for symbol in symbols:
        normalized_symbol = str(symbol or "").strip().upper()
        instrument = instrument_map.get(normalized_symbol) or {}
        instrument_metadata = instrument.get("metadata") if isinstance(instrument.get("metadata"), dict) else {}
        cached_meta = _load_cached_kap_market_metadata(cache_dir, symbol)
        latest_quarter = (
            instrument_metadata.get("latest_quarter")
            or cached_meta.get("latest_quarter")
        )
        source = str(instrument.get("source") or "").strip().lower()
        has_kap_cache = bool(
            instrument_metadata.get("has_kap_cache")
            or source in {"kap", "kap_cache"}
            or cached_meta.get("has_kap_cache")
        )
        name = (
            str(instrument.get("name") or "").strip()
            or str(cached_meta.get("company_title") or "").strip()
            or normalized_symbol
        )
        logo_url = str(instrument.get("logo_url") or "").strip() or None
        logo_source = str(instrument.get("logo_source") or "").strip() or None
        if has_kap_cache:
            kap_cache_count += 1

        rows.append(
            {
                "symbol": normalized_symbol,
                "company": normalized_symbol,
                "name": name,
                "latest_quarter": latest_quarter,
                "has_kap_cache": has_kap_cache,
                "price": None,
                "price_currency": None,
                "change": None,
                "change_pct": None,
                "price_as_of": None,
                "market_cap": None,
                "logo_url": logo_url,
                "logo_source": logo_source,
            }
        )

    coverage_rows = [row for row in rows if row.get("has_kap_cache")]

    data = {
        "stats": {
            "index": normalized_index,
            "index_count": len(rows),
            "bist100_count": len(rows),
            "bist_all_count": bist_all_count,
            "kap_cache_count": kap_cache_count,
        },
        "universe": {
            "index": universe.get("index") or normalized_index,
            "count": int(universe.get("count") or len(rows)),
            "source": universe.get("source"),
            "source_url": universe.get("source_url"),
            "source_date": universe.get("source_date"),
            "fetched_at": universe.get("fetched_at"),
            "cache_hit": bool(universe.get("cache_hit")),
            "fallback_used": bool(universe.get("fallback_used")),
        },
        "rows": rows,
        "coverage_rows": coverage_rows,
    }
    return data


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", **_cache_status()}


@app.get("/market/universe")
def market_universe(index: str = Query("XUTUM"), refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_universe_payload(index_name=index, force_refresh=refresh)


@app.get("/market/stocks/search")
def market_stocks_search(
    q: str = Query("", max_length=80),
    index: str = Query("XUTUM"),
    limit: int = Query(20, ge=1, le=50),
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    normalized_index = _normalize_stock_index(index)
    query = str(q or "").strip().upper()
    payload = _market_universe_payload(index_name=normalized_index)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if query:
        rows = [
            row
            for row in rows
            if query in str(row.get("symbol") or row.get("company") or "").upper()
            or query in str(row.get("name") or "").upper()
        ]
    rows = rows[:limit]
    _log_market_cache_event(
        endpoint="market/stocks/search",
        index=normalized_index,
        cache_status="metadata",
        upstream_called=False,
        stale=False,
        started_at=started_at,
    )
    return {
        "index": normalized_index,
        "query": q,
        "count": len(rows),
        "rows": rows,
        "as_of": payload.get("universe", {}).get("fetched_at") if isinstance(payload.get("universe"), dict) else None,
    }


@app.get("/market/stocks")
def market_stocks(index: str = Query("XUTUM"), refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_stocks_payload(index_name=index, force_refresh=refresh)


@app.get("/market/stocks/cards")
def market_stock_cards(symbols: str = Query(""), refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_stock_cards_payload(symbols=symbols, force_refresh=refresh)


@app.get("/market/stocks/cards/chart")
def market_stock_card_chart(
    symbol: str = Query(""),
    range: str = Query("1d"),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    return _market_stock_card_chart_payload(symbol=symbol, chart_range=range, force_refresh=refresh)


@app.post("/market/comparison-history")
def market_comparison_history(request: MarketComparisonHistoryRequest) -> Dict[str, Any]:
    if request.start_date > request.end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    return _market_comparison_history_payload(request)


@app.get("/market/indices")
def market_indices(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_indices_payload(force_refresh=refresh)


@app.get("/market/indices/{index_code}")
def market_index_detail(index_code: str, refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_index_detail_payload(index_code, force_refresh=refresh)


@app.get("/funds")
def funds(
    q: Optional[str] = Query(None, max_length=80),
    fund_type: Optional[str] = Query(None, max_length=32),
    founder: Optional[str] = Query(None, max_length=120),
    manager: Optional[str] = Query(None, max_length=120),
    risk: Optional[str] = Query(None, max_length=32),
    sort: str = Query("fund_code", max_length=32),
    order: str = Query("asc", max_length=8),
) -> Dict[str, Any]:
    return _funds_listing_payload(
        q=q,
        fund_type=fund_type,
        founder=founder,
        manager=manager,
        risk=risk,
        sort=sort,
        order=order,
    )


@_cached_response(
    key_fn=_fund_listing_cache_key,
    ttl_seconds=45,
)
def _funds_listing_payload(
    *,
    q: Optional[str],
    fund_type: Optional[str],
    founder: Optional[str],
    manager: Optional[str],
    risk: Optional[str],
    sort: str,
    order: str,
) -> Dict[str, Any]:
    from app.fund_service import get_funds_payload

    return get_funds_payload(
        CONFIG.paths.processed_dir,
        q=q,
        fund_type=fund_type,
        founder=founder,
        manager=manager,
        risk=risk,
        sort=sort,
        order=order,
        # Keep the catalog read fast.  A cold/empty snapshot is recovered by
        # the explicit refresh flow so the ordinary list request cannot spend
        # its whole frontend timeout inside TEFAS.
        auto_refresh=False,
    )


@app.get("/funds/search")
def funds_search(q: str = Query("", min_length=0, max_length=80), limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    payload = _funds_search_payload(q=q)
    rows = list(payload.get("rows") or [])[:limit]
    # Trim and clone so each caller gets the right slice without mutating the
    # cached payload referenced by other concurrent requests.
    out = dict(payload)
    out["rows"] = rows
    out["count"] = len(rows)
    return out


@_cached_response(
    key_fn=lambda *, q: f"api:funds-search:v2:q={_normalized_cache_text(q, max_length=80)}",
    ttl_seconds=60,
)
def _funds_search_payload(*, q: str) -> Dict[str, Any]:
    from app.fund_service import get_funds_payload

    return get_funds_payload(
        CONFIG.paths.processed_dir,
        q=q,
        sort="fund_code",
        order="asc",
        min_aum=None,
        # Search must stay cache/snapshot-only as well; refresh is an explicit
        # background job started by the list page.
        auto_refresh=False,
    )


@app.get("/funds/categories")
def funds_categories() -> Dict[str, Any]:
    return _funds_categories_payload()


@_cached_response(key_fn=lambda: "api:funds-categories:v2", ttl_seconds=300)
def _funds_categories_payload() -> Dict[str, Any]:
    from app.fund_service import get_fund_categories_payload

    return get_fund_categories_payload(CONFIG.paths.processed_dir)


def _history_job_key(fund_code: str, job_id: str) -> str:
    return f"api:fund-history:job:v{_FUND_HISTORY_KEY_VERSION}:{fund_code}:{job_id}"


def _history_active_key(fund_code: str) -> str:
    return f"api:fund-history:active:v{_FUND_HISTORY_KEY_VERSION}:{fund_code}"


def _history_last_key(fund_code: str) -> str:
    return f"api:fund-history:last:v{_FUND_HISTORY_KEY_VERSION}:{fund_code}"


def _history_job_get(fund_code: str, job_id: str) -> Optional[Dict[str, Any]]:
    value = _get_cache().get(_history_job_key(fund_code, job_id))
    return dict(value) if isinstance(value, dict) else None


def _history_job_set(fund_code: str, job: Dict[str, Any]) -> None:
    _get_cache().set(
        _history_job_key(fund_code, str(job["job_id"])),
        job,
        ttl_seconds=_FUND_HISTORY_JOB_TTL_SECONDS,
    )


def _history_job_public(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job:
        return None
    return {
        key: job.get(key)
        for key in (
            "job_id",
            "fund_code",
            "requested_start",
            "requested_end",
            "effective_start",
            "effective_end",
            "status",
            "requested_at",
            "started_at",
            "finished_at",
            "heartbeat_at",
            "error",
            "resolution",
            "coverage_state",
            "daily_upgrade_state",
            "fintables_point_count",
            "history_source_used",
            "phase",
        )
        if key in job
    }


def _history_job_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _history_request_range(
    start_date: Optional[date],
    end_date: Optional[date],
) -> Dict[str, Any]:
    from app.fund_service import _fund_full_history_start_date

    effective_end = end_date or date.today()
    full_history = start_date is None and end_date is None
    requested_start = start_date or _fund_full_history_start_date()
    warmup_start = effective_end - timedelta(days=max(1, _FUND_HISTORY_WARMUP_DAYS))
    effective_start = requested_start if full_history else min(requested_start, warmup_start)
    return {
        "requested_start": requested_start,
        "requested_end": effective_end,
        "effective_start": effective_start,
        "effective_end": effective_end,
        "full_history": full_history,
    }


def _history_job_covers(job: Dict[str, Any], target: Dict[str, Any]) -> bool:
    job_start = _history_job_date(job.get("effective_start"))
    job_end = _history_job_date(job.get("effective_end"))
    return bool(
        job_start
        and job_end
        and job_start <= target["requested_start"]
        and job_end >= target["requested_end"]
    )


def _history_job_failed(job: Optional[Dict[str, Any]]) -> bool:
    if not job:
        return False
    return (
        str(job.get("status") or "").strip().lower() == "failed"
        or str(job.get("daily_upgrade_state") or "").strip().lower() == "failed"
    )


def _history_job_should_schedule(
    payload: Dict[str, Any],
    *,
    last_job: Optional[Dict[str, Any]],
    target: Dict[str, Any],
) -> bool:
    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    points = payload.get("points") if isinstance(payload.get("points"), list) else []
    if not points:
        return True
    # Long chart ranges are bootstrapped from Fintables' daily UDF series.
    # Jobs created before that source was made explicit can look successful
    # while containing only the shorter TEFAS detail history. Give each such
    # legacy job one retry; the worker records the probe result so an
    # unavailable upstream does not create a request loop.
    if _history_job_needs_fintables_probe(last_job, target):
        return True
    last_point = _history_job_date(metadata.get("available_end_date") or metadata.get("date_max"))
    if last_point and last_point < target["requested_end"]:
        from app.fund_service import _business_days_between

        if _business_days_between(last_point + timedelta(days=1), target["requested_end"]) > 0:
            return not (
                last_job
                and _history_job_covers(last_job, target)
                and not _history_job_failed(last_job)
            )
    if str(metadata.get("coverage_state") or "") == "range_incomplete":
        return not (
            last_job
            and _history_job_covers(last_job, target)
            and not _history_job_failed(last_job)
        )
    if int(metadata.get("internal_gap_count") or 0) > 0:
        return not (
            last_job
            and _history_job_covers(last_job, target)
            and not _history_job_failed(last_job)
        )
    resolution = str(metadata.get("resolution") or "unknown")
    if resolution != "daily":
        if last_job and _history_job_covers(last_job, target):
            return _history_job_failed(last_job) or str(last_job.get("daily_upgrade_state") or "") not in {
                "unavailable",
                "failed",
            }
        return True
    return False


def _history_job_needs_fintables_probe(
    last_job: Optional[Dict[str, Any]],
    target: Dict[str, Any],
) -> bool:
    target_span_days = (target["requested_end"] - target["requested_start"]).days
    return bool(
        target_span_days > 120
        and last_job
        and _history_job_covers(last_job, target)
        and "fintables_point_count" not in last_job
    )


def _history_attach_job(payload: Dict[str, Any], job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not job:
        return payload
    attached = dict(payload)
    metadata = dict(attached.get("source_metadata") or {})
    public_job = _history_job_public(job)
    metadata["history_job"] = public_job
    points = attached.get("points") if isinstance(attached.get("points"), list) else []
    status = str(job.get("status") or "").lower()
    resolution = str(metadata.get("resolution") or "unknown")
    if status in {"queued", "running"}:
        if points and resolution != "daily":
            if metadata.get("coverage_state") == "complete":
                metadata["coverage_state"] = "upgrading"
        elif not points:
            metadata["coverage_state"] = "upgrading"
        metadata["daily_upgrade_state"] = "pending"
    elif status == "failed":
        metadata["daily_upgrade_state"] = "failed"
    attached["source_metadata"] = metadata
    return attached


def _history_find_existing_job(fund_code: str, target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    backend = _get_cache()
    active_id = backend.get(_history_active_key(fund_code))
    if isinstance(active_id, dict):
        active_id = active_id.get("job_id")
    if active_id:
        active = _history_job_get(fund_code, str(active_id))
        if active and str(active.get("status") or "") in {"queued", "running"}:
            return active
        try:
            backend.delete(_history_active_key(fund_code))
        except Exception:
            pass
    last = backend.get(_history_last_key(fund_code))
    if isinstance(last, dict) and _history_job_covers(last, target):
        return dict(last)
    return None


def _history_merge_target(job: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    for field, chooser in (
        ("requested_start", min),
        ("effective_start", min),
    ):
        existing = _history_job_date(job.get(field))
        value = chooser(existing, target[field]) if existing else target[field]
        job[field] = value.isoformat()
    for field, chooser in (
        ("requested_end", max),
        ("effective_end", max),
    ):
        existing = _history_job_date(job.get(field))
        value = chooser(existing, target[field]) if existing else target[field]
        job[field] = value.isoformat()
    return job


def _history_start_or_extend_job(
    fund_code: str,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
    force_new: bool = False,
) -> Optional[Dict[str, Any]]:
    normalized = str(fund_code or "").strip().upper()
    if not normalized:
        return None
    target = _history_request_range(start_date, end_date)
    backend = _get_cache()
    job: Optional[Dict[str, Any]] = None
    submit = False
    with backend.lock(f"fund-history-state:{normalized}", timeout=5.0) as acquired:
        if not acquired:
            job = _history_find_existing_job(normalized, target)
        else:
            existing = _history_find_existing_job(normalized, target)
            if existing and str(existing.get("status") or "") in {"queued", "running"}:
                previous_start = _history_job_date(existing.get("effective_start"))
                previous_end = _history_job_date(existing.get("effective_end"))
                job = _history_merge_target(existing, target)
                if str(job.get("status")) == "running" and (
                    (previous_start and target["effective_start"] < previous_start)
                    or (previous_end and target["effective_end"] > previous_end)
                ):
                    job["extension_requested"] = True
                _history_job_set(normalized, job)
            elif existing and _history_job_covers(existing, target) and not _history_job_failed(existing) and not force_new:
                job = existing
            else:
                now = _fund_refresh_now_iso()
                job = {
                    "job_id": uuid.uuid4().hex,
                    "fund_code": normalized,
                    "requested_start": target["requested_start"].isoformat(),
                    "requested_end": target["requested_end"].isoformat(),
                    "effective_start": target["effective_start"].isoformat(),
                    "effective_end": target["effective_end"].isoformat(),
                    "status": "queued",
                    "requested_at": now,
                    "started_at": None,
                    "finished_at": None,
                    "heartbeat_at": None,
                    "error": None,
                    "phase": 0,
                }
                _history_job_set(normalized, job)
                backend.set(_history_active_key(normalized), job["job_id"], ttl_seconds=_FUND_HISTORY_JOB_TTL_SECONDS)
                submit = True
    if submit and job:
        try:
            _FUND_HISTORY_EXECUTOR.submit(_run_fund_history_job, normalized, str(job["job_id"]))
        except Exception as exc:
            job["status"] = "failed"
            job["finished_at"] = _fund_refresh_now_iso()
            job["error"] = f"history worker could not start: {exc}"
            _history_job_set(normalized, job)
    return job


def _history_job_owned(backend: Any, lease_key: str, owner_token: str) -> bool:
    return backend.get(lease_key) == owner_token


def _run_fund_history_job(fund_code: str, job_id: str) -> None:
    from app.fund_service import FundUpstreamError, get_fund_performance_payload, normalize_fund_code, refresh_fund_performance

    normalized = normalize_fund_code(fund_code)
    job = _history_job_get(normalized, job_id)
    if not job:
        return
    backend = _get_cache()
    lease_key = f"api:fund-history:lease:v{_FUND_HISTORY_KEY_VERSION}:{normalized}"
    owner_token = uuid.uuid4().hex
    if not backend.set_if_absent(lease_key, owner_token, ttl_seconds=_FUND_HISTORY_LEASE_TTL_SECONDS):
        job["status"] = "failed"
        job["finished_at"] = _fund_refresh_now_iso()
        job["error"] = "history job lease is already owned"
        _history_job_set(normalized, job)
        return

    heartbeat_stop = threading.Event()
    lease_lost = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(max(1.0, _FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS)):
            try:
                renewed = backend.renew_if_owner(
                    lease_key,
                    owner_token,
                    ttl_seconds=_FUND_HISTORY_LEASE_TTL_SECONDS,
                )
            except Exception:
                renewed = False
            if not renewed:
                lease_lost.set()
                return

    heartbeat_thread = threading.Thread(target=heartbeat, name=f"fund-history-heartbeat-{normalized}", daemon=True)
    heartbeat_thread.start()
    try:
        job["status"] = "running"
        job["started_at"] = job.get("started_at") or _fund_refresh_now_iso()
        job["heartbeat_at"] = _fund_refresh_now_iso()
        _history_job_set(normalized, job)
        for phase in range(max(1, _FUND_HISTORY_MAX_PHASES)):
            current = _history_job_get(normalized, job_id) or job
            phase_start = _history_job_date(current.get("effective_start"))
            phase_end = _history_job_date(current.get("effective_end"))
            if not phase_start or not phase_end:
                raise FundUpstreamError("history job range is invalid")
            current["phase"] = phase + 1
            current["heartbeat_at"] = _fund_refresh_now_iso()
            _history_job_set(normalized, current)
            refresh_fund_performance(
                CONFIG.paths.processed_dir,
                normalized,
                start_date=phase_start,
                end_date=phase_end,
                prefer_fast_long_range=(phase_end - phase_start).days > 120,
            )
            if lease_lost.is_set() or not _history_job_owned(backend, lease_key, owner_token):
                raise FundUpstreamError("history job lease was lost before commit")
            latest = _history_job_get(normalized, job_id) or current
            latest_start = _history_job_date(latest.get("effective_start"))
            latest_end = _history_job_date(latest.get("effective_end"))
            range_widened = bool(
                (latest_start and latest_start < phase_start)
                or (latest_end and latest_end > phase_end)
            )
            if phase + 1 < max(1, _FUND_HISTORY_MAX_PHASES) and range_widened:
                continue
            result = get_fund_performance_payload(
                CONFIG.paths.processed_dir,
                normalized,
                start_date=phase_start,
                end_date=phase_end,
                auto_refresh=False,
            )
            metadata = result.get("source_metadata") if isinstance(result.get("source_metadata"), dict) else {}
            latest.update(
                {
                    "status": "succeeded",
                    "finished_at": _fund_refresh_now_iso(),
                    "heartbeat_at": _fund_refresh_now_iso(),
                    "resolution": metadata.get("resolution"),
                    "coverage_state": metadata.get("coverage_state"),
                    "daily_upgrade_state": "unavailable" if metadata.get("resolution") != "daily" else "complete",
                    "fintables_point_count": int(metadata.get("cached_fallback_point_count") or 0),
                    "history_source_used": metadata.get("history_source_used"),
                    "error": None,
                }
            )
            _invalidate_single_fund_response_cache(normalized)
            _history_job_set(normalized, latest)
            backend.set(_history_last_key(normalized), latest, ttl_seconds=_FUND_HISTORY_JOB_TTL_SECONDS)
            return
    except Exception as exc:
        failed = _history_job_get(normalized, job_id) or job
        failed.update(
            {
                "status": "failed",
                "finished_at": _fund_refresh_now_iso(),
                "error": str(exc),
                "daily_upgrade_state": "failed",
            }
        )
        _history_job_set(normalized, failed)
        backend.set(_history_last_key(normalized), failed, ttl_seconds=_FUND_HISTORY_JOB_TTL_SECONDS)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=max(1.0, _FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS))
        try:
            backend.release_if_owner(lease_key, owner_token)
        except Exception:
            pass
        active = backend.get(_history_active_key(normalized))
        if active == job_id:
            backend.delete(_history_active_key(normalized))


@app.get("/funds/{fund_code}/performance")
def fund_performance(
    fund_code: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    fallback: bool = Query(False),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    return _fund_performance_payload(
        fund_code=fund_code,
        start_iso=start_date.isoformat() if start_date else None,
        end_iso=end_date.isoformat() if end_date else None,
        fallback=fallback,
        refresh=refresh,
    )


@_cached_response(
    key_fn=lambda *, fund_code, start_iso, end_iso, fallback, refresh: (
        f"api:fund-performance:v{_FUND_HISTORY_KEY_VERSION}:{fund_code}:{start_iso or 'full'}:{end_iso or 'today'}"
        f":fb={1 if fallback else 0}"
    ),
    ttl_seconds=60,
)
def _fund_performance_payload(
    *,
    fund_code: str,
    start_iso: Optional[str],
    end_iso: Optional[str],
    fallback: bool,
    refresh: bool,
) -> Dict[str, Any]:
    from app.fund_service import get_fund_performance_payload, normalize_fund_code

    start = date.fromisoformat(start_iso) if start_iso else None
    end = date.fromisoformat(end_iso) if end_iso else None
    normalized = normalize_fund_code(fund_code)
    target = _history_request_range(start, end)
    existing_job = _history_find_existing_job(normalized, target)
    payload = get_fund_performance_payload(
        CONFIG.paths.processed_dir,
        normalized,
        start_date=start,
        end_date=end,
        allow_upstream_fallback=fallback,
        auto_refresh=False,
    )
    if _history_job_should_schedule(payload, last_job=existing_job, target=target):
        existing_job = _history_start_or_extend_job(
            normalized,
            start_date=start,
            end_date=end,
            force_new=_history_job_needs_fintables_probe(existing_job, target),
        )
    return _history_attach_job(payload, existing_job)


@app.get("/funds/{fund_code}/performance/status")
def fund_performance_status(
    fund_code: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    target = _history_request_range(start_date, end_date)
    job = _history_find_existing_job(normalized, target)
    return {
        "fund_code": normalized,
        "status": str(job.get("status") if job else "idle"),
        "history_job": _history_job_public(job),
    }


@app.get("/funds/{fund_code}/yield-summary")
def fund_yield_summary(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import FintablesUpstreamError, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        return _fund_yield_summary_payload(normalized=normalized)
    except FintablesUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@_cached_response(
    key_fn=lambda *, normalized: f"api:fund-yield-summary:{normalized}",
    ttl_seconds=_FUND_YIELD_SUMMARY_CACHE_TTL,
    skip_when=lambda *, normalized: not normalized,
    single_flight=True,
    lock_timeout=20,
)
def _fund_yield_summary_payload(*, normalized: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_yield_summary_payload

    return get_fund_yield_summary_payload(normalized, processed_dir=CONFIG.paths.processed_dir)


@app.get("/funds/{fund_code}/holdings")
def fund_holdings(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    payload = _fund_holdings_static_payload(normalized=normalized)
    return _enrich_fund_holdings_with_daily_market_data(payload, normalized)


@_cached_response(
    key_fn=lambda *, normalized: f"api:fund-holdings:{normalized}:v{_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION}",
    ttl_seconds=_FUND_HOLDINGS_RESPONSE_CACHE_TTL,
    skip_when=lambda *, normalized: not normalized,
    single_flight=True,
    lock_timeout=_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT,
)
def _fund_holdings_static_payload(*, normalized: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_holdings_payload

    return get_fund_holdings_payload(CONFIG.paths.processed_dir, normalized)


_FUND_HOLDINGS_LIVE_POSITION_FIELDS = (
    "asset_code",
    "price",
    "price_currency",
    "return_pct",
    "return_source",
    "return_as_of",
    "estimated_exposure_value",
    "estimated_pnl_value",
    "estimated_fund_return_contribution_pct",
)


_FUND_HOLDING_SECTOR_LABELS: Dict[str, str] = {
    "XBANK": "Bankacılık",
    "XAKUR": "Aracı Kurum",
    "XBLSM": "Bilişim ve Yazılım",
    "XELKT": "Elektrik",
    "XFINK": "Finansal Kiralama Faktoring",
    "XGMYO": "Gayrimenkul",
    "XGIDA": "Gıda İçecek",
    "XHOLD": "Holding",
    "XILTM": "İletişim",
    "XINSA": "İnşaat",
    "XKAGT": "Orman Kağıt Basım",
    "XKMYA": "Kimya Petrol Plastik",
    "XMADN": "Madencilik",
    "XMANA": "Metal Ana",
    "XMESY": "Metal Eşya Makina",
    "XSGRT": "Sigorta",
    "XSPOR": "Spor",
    "XTAST": "Taş Toprak",
    "XTCRT": "Ticaret",
    "XTEKS": "Tekstil Deri",
    "XTRZM": "Turizm",
    "XULAS": "Ulaştırma",
    "XYORT": "Menkul Kıymet Y.O.",
    "XUSIN": "Sınai",
    "XUHIZ": "Hizmetler",
    "XUMAL": "Mali",
    "XUTEK": "Teknoloji",
}
_FUND_HOLDING_SECTOR_PRIORITY = (
    "XBANK",
    "XAKUR",
    "XBLSM",
    "XELKT",
    "XFINK",
    "XGMYO",
    "XGIDA",
    "XHOLD",
    "XILTM",
    "XINSA",
    "XKAGT",
    "XKMYA",
    "XMADN",
    "XMANA",
    "XMESY",
    "XSGRT",
    "XSPOR",
    "XTAST",
    "XTCRT",
    "XTEKS",
    "XTRZM",
    "XULAS",
    "XYORT",
    "XUSIN",
    "XUHIZ",
    "XUMAL",
    "XUTEK",
)
_FUND_HOLDING_SECTOR_MAP_CACHE: Dict[str, Any] = {}


def _fund_holding_sector_map() -> tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    cached = _FUND_HOLDING_SECTOR_MAP_CACHE.get("default")
    now = time.time()
    if cached and now - float(cached.get("_ts", 0)) < _FUND_HOLDING_SECTOR_MAP_CACHE_TTL:
        return dict(cached.get("map") or {}), {
            "cache_hit": True,
            "symbol_count": cached.get("symbol_count", 0),
            "source": cached.get("source"),
            "source_date": cached.get("source_date"),
            "warnings": list(cached.get("warnings") or []),
        }
    cache_key = "api:funds:holding-sector-map:v1"
    redis_cached = _shared_cache_get_dict(cache_key)
    if isinstance(redis_cached, dict):
        _FUND_HOLDING_SECTOR_MAP_CACHE["default"] = {
            "_ts": now,
            "map": dict(redis_cached.get("map") or {}),
            "symbol_count": redis_cached.get("symbol_count", 0),
            "source": redis_cached.get("source"),
            "source_date": redis_cached.get("source_date"),
            "warnings": list(redis_cached.get("warnings") or []),
        }
        return dict(redis_cached.get("map") or {}), {
            "cache_hit": True,
            "symbol_count": redis_cached.get("symbol_count", 0),
            "source": redis_cached.get("source"),
            "source_date": redis_cached.get("source_date"),
            "warnings": list(redis_cached.get("warnings") or []),
        }

    from app.kap_service import get_bist_index_universe

    sector_map: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []
    source = None
    source_date = None
    for sector_code in _FUND_HOLDING_SECTOR_PRIORITY:
        try:
            universe = get_bist_index_universe(sector_code)
        except Exception as exc:
            warnings.append(f"{sector_code}: {exc}")
            continue
        source = source or universe.get("source")
        source_date = source_date or universe.get("source_date")
        sector_label = _FUND_HOLDING_SECTOR_LABELS.get(sector_code, sector_code)
        for symbol in list(universe.get("symbols") or []):
            normalized_symbol = str(symbol or "").strip().upper().replace(".", "")
            if normalized_symbol and normalized_symbol not in sector_map:
                sector_map[normalized_symbol] = {
                    "sector_code": sector_code,
                    "sector_label": sector_label,
                }

    cache_payload = {
        "_ts": now,
        "map": sector_map,
        "symbol_count": len(sector_map),
        "source": source,
        "source_date": source_date,
        "warnings": warnings,
    }
    _FUND_HOLDING_SECTOR_MAP_CACHE["default"] = cache_payload
    _shared_cache_set(
        cache_key,
        {
            "map": sector_map,
            "symbol_count": len(sector_map),
            "source": source,
            "source_date": source_date,
            "warnings": warnings,
        },
        ttl_seconds=_FUND_HOLDING_SECTOR_MAP_CACHE_TTL,
    )
    return dict(sector_map), {
        "cache_hit": False,
        "symbol_count": len(sector_map),
        "source": source,
        "source_date": source_date,
        "warnings": warnings,
    }


@app.get("/funds/{fund_code}/holdings/live")
def fund_holdings_live(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    key = f"api:fund-holdings-live:{normalized}:v1"
    payload, cache_status = _get_or_set_single_flight(
        key,
        ttl_seconds=_FUND_HOLDINGS_LIVE_RESPONSE_CACHE_TTL,
        factory=lambda: _fund_holdings_live_payload(normalized=normalized),
        lock_ttl_seconds=_FUND_HOLDINGS_LIVE_RESPONSE_LOCK_TTL,
        wait_timeout_seconds=_FUND_HOLDINGS_LIVE_RESPONSE_WAIT_TIMEOUT,
    )
    if isinstance(payload, dict):
        return payload
    # Do not make a waiter repeat the live Yahoo/InfoYatirim fan-out. The
    # static response lets the client retain its existing values until the
    # single owner populates the short shared cache.
    static = _fund_holdings_static_payload(normalized=normalized)
    return {
        "fund_code": normalized,
        "status": static.get("status") or "pending",
        "positions": [],
        "portfolio_effect": None,
        "source": "daily_market_enrichment",
        "as_of": None,
        "source_metadata": {
            "source": "daily_market_enrichment",
            "live_cache_status": cache_status,
            "refresh_pending": True,
        },
    }


def _fund_holdings_live_payload(*, normalized: str) -> Dict[str, Any]:
    payload = _fund_holdings_static_payload(normalized=normalized)
    enriched = _enrich_fund_holdings_with_daily_market_data(payload, normalized)
    positions = []
    for position in enriched.get("positions") or []:
        if not isinstance(position, dict):
            continue
        positions.append({field: position.get(field) for field in _FUND_HOLDINGS_LIVE_POSITION_FIELDS})
    metadata = dict(enriched.get("source_metadata") or {})
    return {
        "fund_code": normalized,
        "status": enriched.get("status"),
        "positions": positions,
        "portfolio_effect": enriched.get("portfolio_effect"),
        "source": "daily_market_enrichment",
        "as_of": (enriched.get("portfolio_effect") or {}).get("as_of"),
        "source_metadata": {
            "source": "daily_market_enrichment",
            "static_cache_hit": metadata.get("cache_hit"),
            "disclosure_check": metadata.get("disclosure_check"),
            "daily_market_enrichment": metadata.get("daily_market_enrichment"),
            "market_enrichment": metadata.get("market_enrichment"),
        },
    }


def _api_number(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    try:
        return _parse_tr_decimal(raw)
    except Exception:
        return None


def _fund_snapshot_row_map() -> Dict[str, Dict[str, Any]]:
    rows, _meta = _fund_snapshot_row_map_with_meta()
    return rows


_FUND_SNAPSHOT_ROW_MAP_CACHE: Dict[str, Any] = {}


def _fund_snapshot_row_map_with_meta() -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    from app.fund_service import load_funds_snapshot, normalize_fund_code

    snapshot_path = CONFIG.paths.processed_dir / "funds_cache" / "funds_latest.json"
    stat = snapshot_path.stat() if snapshot_path.exists() else None
    cache_key = str(snapshot_path)
    cached = _FUND_SNAPSHOT_ROW_MAP_CACHE.get(cache_key)
    if cached and stat and cached.get("mtime") == stat.st_mtime:
        return dict(cached.get("rows") or {}), {
            "cache_hit": True,
            "row_count": cached.get("row_count", 0),
            "as_of": cached.get("as_of"),
        }
    stat_version = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)) if stat else None
    shared_key = f"api:funds:snapshot-row-map:mtime={stat_version}:v1" if stat_version is not None else None
    if shared_key:
        shared_cached = _shared_cache_get_dict(shared_key)
        if shared_cached is not None:
            rows = dict(shared_cached.get("rows") or {})
            cache_payload = {
                "mtime": stat.st_mtime if stat else None,
                "rows": rows,
                "row_count": shared_cached.get("row_count", len(rows)),
                "as_of": shared_cached.get("as_of"),
            }
            _FUND_SNAPSHOT_ROW_MAP_CACHE[cache_key] = cache_payload
            return rows, {
                "cache_hit": True,
                "row_count": cache_payload["row_count"],
                "as_of": cache_payload["as_of"],
                "shared_cache_hit": True,
            }

    try:
        snapshot = load_funds_snapshot(CONFIG.paths.processed_dir)
    except Exception:
        return {}, {"cache_hit": False, "row_count": 0, "error": "snapshot_unavailable"}
    rows: Dict[str, Dict[str, Any]] = {}
    for row in list(snapshot.get("rows") or []):
        if not isinstance(row, dict):
            continue
        code = normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        if code:
            rows[code] = row
    if stat:
        _FUND_SNAPSHOT_ROW_MAP_CACHE[cache_key] = {
            "mtime": stat.st_mtime,
            "rows": rows,
            "row_count": len(rows),
            "as_of": snapshot.get("as_of"),
        }
        if shared_key:
            _shared_cache_set(
                shared_key,
                {
                    "rows": rows,
                    "row_count": len(rows),
                    "as_of": snapshot.get("as_of"),
                },
                ttl_seconds=300,
            )
    return dict(rows), {"cache_hit": False, "row_count": len(rows), "as_of": snapshot.get("as_of")}


def _holding_code(position: Dict[str, Any]) -> str:
    from app.fund_service import normalize_fund_code

    return normalize_fund_code(str(position.get("asset_code") or position.get("asset_name") or "")).replace(".", "")


def _holding_type(position: Dict[str, Any]) -> str:
    return str(position.get("asset_type") or "").strip().lower()


_GEFAS_GYF_ALIAS_MAP: Dict[str, Dict[str, str]] = {
    "TPKGY": {
        "isin": "TRYTALP00036",
        "gefas_code": "TPKGY.F1",
        "label": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
    },
    "TPKGYF": {
        "isin": "TRYTALP00036",
        "gefas_code": "TPKGY.F1",
        "label": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
    },
    "TPKGYF1": {
        "isin": "TRYTALP00036",
        "gefas_code": "TPKGY.F1",
        "label": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
    },
}
_GEFAS_GYF_QUOTE_CACHE: Dict[str, Dict[str, Any]] = {}
_GEFAS_GYF_QUOTE_CACHE_TTL = 24 * 60 * 60
_FOREIGN_HOLDING_QUOTE_CACHE: Dict[str, Dict[str, Any]] = {}
_FOREIGN_HOLDING_QUOTE_CACHE_TTL = 15 * 60
_FOREIGN_HOLDING_KNOWN_NAMES_BY_ISIN = {
    "CH0183135992": "Swisscanto (CH) Silver ETF",
    "CH0118929048": "UBS Silver ETF USD acc",
    "CA37964K1012": "Global X Silver ETF",
    "US0032641088": "abrdn Physical Silver Shares ETF",
    "US46428Q1094": "iShares Silver Trust",
}
_FOREIGN_HOLDING_KNOWN_NAMES_BY_PROVIDER = {
    "ZSIGEU.SW": "Swisscanto (CH) Silver ETF",
    "SVUSA.SW": "UBS Silver ETF USD acc",
    "HUZ.TO": "Global X Silver ETF",
    "SIVR": "abrdn Physical Silver Shares ETF",
    "SLV": "iShares Silver Trust",
}


def _gefas_gyf_config(symbol: str) -> Optional[Dict[str, str]]:
    normalized = str(symbol or "").strip().upper().replace(".", "")
    return _GEFAS_GYF_ALIAS_MAP.get(normalized)


def _gefas_chart_date(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _fetch_gefas_gyf_chart(isin: str, metric: int) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    normalized_isin = str(isin or "").strip().upper()
    if not normalized_isin:
        return {}
    url = f"https://gefas.gov.tr/gyf/detay/grafik/{normalized_isin}/0/0/{metric}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://gefas.gov.tr/tr/gyf/detay",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception):
        return {}


def _fetch_gefas_gyf_quote(symbol: str) -> Optional[Dict[str, Any]]:
    config = _gefas_gyf_config(symbol)
    if not config:
        return None
    cache_key = config["isin"]
    now = time.time()
    cached = _GEFAS_GYF_QUOTE_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _GEFAS_GYF_QUOTE_CACHE_TTL:
        data = dict(cached.get("data") or {})
        if data:
            data["_cache_hit"] = True
        return data
    shared_key = f"api:funds:gefas-gyf-quote:{cache_key}:v1"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        _GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": shared_cached}
        data = dict(shared_cached)
        if data:
            data["_cache_hit"] = True
        return data

    price_chart = _fetch_gefas_gyf_chart(config["isin"], 0)
    return_chart = _fetch_gefas_gyf_chart(config["isin"], 2)
    prices = list(price_chart.get("datas") or [])
    price_labels = list(price_chart.get("labels") or [])
    returns = list(return_chart.get("datas") or [])
    return_labels = list(return_chart.get("labels") or [])
    price = _api_number(prices[-1]) if prices else None
    return_pct = _api_number(returns[-1]) if returns else None
    as_of = _gefas_chart_date(price_labels[-1] if price_labels else None) or _gefas_chart_date(return_labels[-1] if return_labels else None)
    if price is None and return_pct is None:
        _GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": {}}
        _shared_cache_set(shared_key, {}, ttl_seconds=_GEFAS_GYF_QUOTE_CACHE_TTL)
        return {}

    data = {
        "price": price,
        "currency": "TRY",
        "change_pct": return_pct,
        "as_of": as_of,
        "source": "gefas_gyf",
        "source_url": f"https://gefas.gov.tr/tr/gyf/detay/{config['gefas_code']}",
        "isin": config["isin"],
        "gefas_code": config["gefas_code"],
        "label": config.get("label"),
    }
    _GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": data}
    _shared_cache_set(shared_key, data, ttl_seconds=_GEFAS_GYF_QUOTE_CACHE_TTL)
    result = dict(data)
    result["_cache_hit"] = False
    return result


def _quote_map_for_holding_stocks(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    unique_symbols: List[str] = []
    seen_symbols: set[str] = set()
    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        if not normalized or normalized in seen_symbols:
            continue
        seen_symbols.add(normalized)
        unique_symbols.append(normalized)
    if not unique_symbols:
        return {}
    fetched_quotes = _fetch_market_price_map(unique_symbols, index_name="XUTUM")
    quotes = {symbol: fetched_quotes.get(symbol, {}) for symbol in unique_symbols if fetched_quotes.get(symbol)}
    missing_symbols = [
        symbol
        for symbol in unique_symbols
        if _api_number((quotes.get(symbol) or {}).get("price")) is None
        or _api_number((quotes.get(symbol) or {}).get("change_pct")) is None
    ]
    for symbol in missing_symbols[:_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT]:
        fallback = _fetch_infoyatirim_stock_page_quote(symbol)
        if fallback:
            quotes[symbol] = _merge_market_price_fallback(quotes.get(symbol, {}), fallback)
    return quotes


def _foreign_holding_provider_symbol(position: Dict[str, Any]) -> str:
    symbol = str(
        position.get("provider_symbol")
        or position.get("logo_symbol")
        or position.get("asset_code")
        or ""
    ).strip().upper()
    return symbol


def _quote_map_for_foreign_holdings(positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    symbols: List[str] = []
    seen: set[str] = set()
    for position in positions:
        if _holding_type(position) not in {"foreign_equity", "foreign_fund"}:
            continue
        symbol = _foreign_holding_provider_symbol(position)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    if not symbols:
        return {}

    now = time.time()
    quotes: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        cached = _FOREIGN_HOLDING_QUOTE_CACHE.get(symbol)
        if cached and now - float(cached.get("_ts") or 0.0) < _FOREIGN_HOLDING_QUOTE_CACHE_TTL:
            data = dict(cached.get("data") or {})
            if data:
                quotes[symbol] = data
            continue
        cache_key = f"api:funds:foreign-holding-quote:{re.sub(r'[^A-Z0-9_.-]+', '_', symbol)}:v4"
        shared_cached = _shared_cache_get_dict(cache_key)
        if shared_cached is not None:
            _FOREIGN_HOLDING_QUOTE_CACHE[symbol] = {"_ts": now, "data": shared_cached}
            if shared_cached:
                quotes[symbol] = dict(shared_cached)
            continue

        raw_quote = _fetch_yahoo_quote(symbol)
        data: Dict[str, Any] = {}
        if raw_quote.get("ok") and (
            _api_number(raw_quote.get("price")) is not None
            or _api_number(raw_quote.get("change_pct")) is not None
        ):
            data = {
                "price": _api_number(raw_quote.get("price")),
                "currency": raw_quote.get("currency"),
                "change_pct": _api_number(raw_quote.get("change_pct")),
                "as_of": raw_quote.get("as_of"),
                "source": "yahoo_finance_chart",
                "provider_symbol": symbol,
                "short_name": raw_quote.get("short_name"),
                "long_name": raw_quote.get("long_name"),
            }
            quotes[symbol] = data
        _FOREIGN_HOLDING_QUOTE_CACHE[symbol] = {"_ts": now, "data": data}
        _shared_cache_set(cache_key, data, ttl_seconds=_FOREIGN_HOLDING_QUOTE_CACHE_TTL)
    return quotes


def _foreign_holding_provider_name(quote: Dict[str, Any]) -> Optional[str]:
    for key in ("long_name", "short_name"):
        value = str(quote.get(key) or "").strip()
        if value:
            return value
    return None


def _foreign_holding_known_name(position: Dict[str, Any]) -> Optional[str]:
    isin = str(position.get("isin") or "").strip().upper()
    if isin and isin in _FOREIGN_HOLDING_KNOWN_NAMES_BY_ISIN:
        return _FOREIGN_HOLDING_KNOWN_NAMES_BY_ISIN[isin]
    provider_symbol = _foreign_holding_provider_symbol(position)
    if provider_symbol and provider_symbol in _FOREIGN_HOLDING_KNOWN_NAMES_BY_PROVIDER:
        return _FOREIGN_HOLDING_KNOWN_NAMES_BY_PROVIDER[provider_symbol]
    return None


def _foreign_holding_should_use_provider_name(position: Dict[str, Any], provider_name: str) -> bool:
    current = str(position.get("asset_name") or "").strip()
    if not provider_name or not current:
        return bool(provider_name and not current)
    code = _holding_code(position)
    provider_symbol = _foreign_holding_provider_symbol(position)
    normalized_current = re.sub(r"[^A-Z0-9]+", "", current.upper())
    if normalized_current in {code, re.sub(r"[^A-Z0-9]+", "", provider_symbol.upper())}:
        return True
    current_norm = current.upper()
    if " EQUITY" in current_norm or current_norm in {"CORP", "INC", "CMN", "AMERICA", "HOLDINGS", "MINERALS"}:
        return True
    letters = [char for char in current if char.isalpha()]
    return bool(letters and current == current.upper() and len(current) > 8)


def _position_daily_market_fields(
    position: Dict[str, Any],
    *,
    stock_quotes: Dict[str, Dict[str, Any]],
    gefas_quotes: Dict[str, Dict[str, Any]],
    foreign_quotes: Dict[str, Dict[str, Any]],
    fund_rows: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    code = _holding_code(position)
    asset_type = _holding_type(position)
    if asset_type == "local_equity":
        quote = stock_quotes.get(code) or {}
        return {
            "price": _api_number(quote.get("price")),
            "price_currency": quote.get("currency") or "TRY",
            "return_pct": _api_number(quote.get("change_pct")),
            "return_source": "infoyatirim_live_quote" if quote else None,
            "return_as_of": quote.get("as_of"),
        }
    if asset_type == "fund":
        row = fund_rows.get(code) or {}
        quote = stock_quotes.get(code) or {}
        if quote and _api_number(quote.get("change_pct")) is not None:
            return {
                "price": _api_number(quote.get("price")),
                "price_currency": quote.get("currency") or "TRY",
                "return_pct": _api_number(quote.get("change_pct")),
                "return_source": "infoyatirim_live_quote",
                "return_as_of": quote.get("as_of"),
            }
        gefas_quote = gefas_quotes.get(code) or {}
        if gefas_quote and (_api_number(gefas_quote.get("price")) is not None or _api_number(gefas_quote.get("change_pct")) is not None):
            return {
                "price": _api_number(gefas_quote.get("price")),
                "price_currency": gefas_quote.get("currency") or "TRY",
                "return_pct": _api_number(gefas_quote.get("change_pct")),
                "return_source": "gefas_gyf",
                "return_as_of": gefas_quote.get("as_of"),
            }
        return {
            "price": _api_number(row.get("price")),
            "price_currency": row.get("currency") or "TRY",
            "return_pct": _api_number(row.get("daily_return")),
            "return_source": "tefasfon_funds" if row else None,
            "return_as_of": row.get("as_of"),
        }
    if asset_type in {"foreign_equity", "foreign_fund"}:
        provider_symbol = _foreign_holding_provider_symbol(position)
        quote = foreign_quotes.get(provider_symbol) or {}
        return {
            "price": _api_number(quote.get("price")) if quote else _api_number(position.get("price")),
            "price_currency": quote.get("currency") if quote else None,
            "return_pct": _api_number(quote.get("change_pct")) if quote else None,
            "return_source": "yahoo_finance_chart" if quote else None,
            "return_as_of": quote.get("as_of") if quote else None,
        }
    return {
        "price": _api_number(position.get("price")),
        "price_currency": None,
        "return_pct": None,
        "return_source": None,
        "return_as_of": None,
    }


def _holding_weight_scale_context(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights = [
        weight
        for weight in (_api_number(position.get("weight")) for position in positions)
        if weight is not None and weight > 0
    ]
    total = sum(weights)
    if not weights:
        return {"action": "none", "factor": 1.0, "reason": None}
    max_weight = max(weights)
    if len(weights) >= 3 and 9000 <= total <= 11000 and max_weight <= 10000:
        return {
            "action": "basis_points_to_percent",
            "factor": 0.01,
            "reason": "positive weights sum near 10000 basis points",
        }
    # Fractional weights (0..1) reported instead of percents.
    if max_weight <= 1.05 and total <= 1.05 and len(weights) >= 2:
        return {
            "action": "fraction_to_percent",
            "factor": 100.0,
            "reason": "positive weights look like fractions of one",
        }
    return {"action": "none", "factor": 1.0, "reason": None}


def _per_position_basis_points_threshold() -> float:
    # Anything above 1000% is far beyond plausible leverage and almost
    # certainly reported in basis points.  Using a relatively conservative
    # threshold keeps real leveraged positions (<= a few hundred percent)
    # untouched.
    return 1000.0


def _validated_holding_weight(
    raw_value: Any,
    scale_factor: float,
    *,
    quality_hint: Any = None,
    warning_hint: Any = None,
) -> Dict[str, Any]:
    raw_number = _api_number(raw_value)
    if raw_number is None:
        return {"weight": None, "raw_weight": None, "quality": "missing", "warning": None}
    weight = raw_number * scale_factor
    hinted_quality = str(quality_hint or "").strip().lower()
    quality = "normalized" if scale_factor != 1.0 else ("fallback" if hinted_quality == "fallback" else "ok")
    warning = str(warning_hint or "").strip() or None
    # Per-position safety net: a single row well past 1000% with the rest of
    # the portfolio in normal percent territory is almost always a basis-point
    # leak.  Divide it down individually so other rows stay untouched.
    if scale_factor == 1.0 and abs(raw_number) > _per_position_basis_points_threshold() and abs(raw_number) <= 11000:
        weight = raw_number / 100.0
        quality = "normalized"
        warning = warning or "KAP ağırlığı yüzde ölçeğine normalize edildi."
    # A value beyond the normalization range is not a plausible holding
    # weight.  It is commonly a leaked borsa/sözleşme number (for example
    # 80100517), so fail closed instead of exposing an astronomical percent.
    if not math.isfinite(weight) or abs(weight) > _per_position_basis_points_threshold():
        return {
            "weight": None,
            "raw_weight": raw_number,
            "quality": "invalid",
            "warning": warning or "KAP ağırlığı geçersiz veya sözleşme numarası olarak algılandı.",
        }
    return {
        "weight": round(weight, 6),
        "raw_weight": raw_number if quality == "normalized" else None,
        "quality": quality,
        "warning": warning,
    }


def _validate_fund_holding_weights(positions: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scale_context = _holding_weight_scale_context(positions)
    scale_factor = float(scale_context.get("factor") or 1.0)
    validated: List[Dict[str, Any]] = []
    normalized_count = 0
    fallback_count = 0
    invalid_count = 0
    raw_positive_total = 0.0
    adjusted_positive_total = 0.0

    for position in positions:
        row = dict(position)
        raw_weight = _api_number(row.get("weight"))
        if raw_weight is not None and raw_weight > 0:
            raw_positive_total += raw_weight

        weight_result = _validated_holding_weight(
            row.get("weight"),
            scale_factor,
            quality_hint=row.get("weight_quality"),
            warning_hint=row.get("weight_warning"),
        )
        previous_result = _validated_holding_weight(
            row.get("previous_weight"),
            scale_factor,
            quality_hint=row.get("previous_weight_quality"),
            warning_hint=row.get("previous_weight_warning"),
        )

        row["weight"] = weight_result["weight"]
        row["weight_quality"] = weight_result["quality"]
        row["weight_warning"] = weight_result["warning"]
        if weight_result["raw_weight"] is not None:
            row["raw_weight"] = weight_result["raw_weight"]
        else:
            row.pop("raw_weight", None)

        row["previous_weight"] = previous_result["weight"]
        row["previous_weight_quality"] = previous_result["quality"]
        row["previous_weight_warning"] = previous_result["warning"]
        if previous_result["raw_weight"] is not None:
            row["raw_previous_weight"] = previous_result["raw_weight"]
        else:
            row.pop("raw_previous_weight", None)

        if weight_result["quality"] == "normalized" or previous_result["quality"] == "normalized":
            normalized_count += 1
        if weight_result["quality"] == "fallback" or previous_result["quality"] == "fallback":
            fallback_count += 1
        if weight_result["quality"] == "invalid" or previous_result["quality"] == "invalid":
            invalid_count += 1

        current_weight = weight_result["weight"]
        previous_weight = previous_result["weight"]
        if weight_result["quality"] == "invalid" or previous_result["quality"] == "invalid":
            row["weight_change"] = None
        elif current_weight is not None and previous_weight is not None and (
            weight_result["quality"] != "ok" or previous_result["quality"] != "ok"
        ):
            row["weight_change"] = round(current_weight - previous_weight, 6)
        elif current_weight is not None and previous_weight is None and weight_result["quality"] != "ok":
            row["weight_change"] = round(current_weight, 6)

        if row.get("weight") is not None and row["weight"] > 0:
            adjusted_positive_total += float(row["weight"])

        validated.append(row)

    status = "ok"
    if adjusted_positive_total > 100.5:
        status = "gross_exposure"

    quality = {
        "status": status,
        "normalized_position_count": normalized_count,
        "fallback_position_count": fallback_count,
        "invalid_position_count": invalid_count,
        "raw_total_weight": round(raw_positive_total, 6),
        "adjusted_total_weight": round(adjusted_positive_total, 6),
        "normalization": {
            "action": scale_context.get("action") or "none",
            "factor": scale_factor,
            "reason": scale_context.get("reason"),
        },
    }
    return validated, quality


def _enrich_fund_holdings_with_daily_market_data(payload: Dict[str, Any], fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized_fund = normalize_fund_code(fund_code)
    raw_positions = [dict(position) for position in list(payload.get("positions") or []) if isinstance(position, dict)]
    positions, holdings_quality = _validate_fund_holding_weights(raw_positions)
    fund_rows, fund_rows_meta = _fund_snapshot_row_map_with_meta()
    fund_row = fund_rows.get(normalized_fund) or {}
    fund_aum = _api_number(fund_row.get("aum"))
    stock_symbols = [
        _holding_code(position)
        for position in positions
        if _holding_code(position)
        and (
            _holding_type(position) == "local_equity"
            or (
                _holding_type(position) == "fund"
                and _holding_code(position) not in fund_rows
                and not _gefas_gyf_config(_holding_code(position))
            )
        )
    ]
    stock_quotes = _quote_map_for_holding_stocks(stock_symbols)
    gefas_quotes: Dict[str, Dict[str, Any]] = {}
    gefas_quote_cache_hits = 0
    for position in positions:
        code = _holding_code(position)
        if _holding_type(position) != "fund" or not _gefas_gyf_config(code):
            continue
        quote = _fetch_gefas_gyf_quote(code)
        if quote:
            if quote.get("_cache_hit"):
                gefas_quote_cache_hits += 1
            gefas_quotes[code] = quote
    foreign_quotes = _quote_map_for_foreign_holdings(positions)
    sector_map, sector_meta = _fund_holding_sector_map()

    enriched_positions: List[Dict[str, Any]] = []
    estimated_return_pct = 0.0
    estimated_pnl_value = 0.0
    has_pnl = False
    priced_weight = 0.0
    missing_weight = 0.0

    for position in positions:
        row = dict(position)
        row_type = _holding_type(row)
        if row_type in {"foreign_equity", "foreign_fund"}:
            provider_symbol = _foreign_holding_provider_symbol(row)
            row["asset_region"] = "foreign"
            row["provider_symbol"] = provider_symbol or None
            row["logo_symbol"] = row.get("logo_symbol") or provider_symbol or row.get("asset_code")
            row["detail_clickable"] = False
            quote = foreign_quotes.get(provider_symbol) if provider_symbol else None
            provider_name = _foreign_holding_provider_name(quote or {}) or _foreign_holding_known_name(row)
            if provider_name:
                row["provider_name"] = provider_name
                row["asset_name"] = provider_name
        elif row_type == "local_equity":
            row["asset_region"] = row.get("asset_region") or "TR"
            row["detail_clickable"] = True if row.get("detail_clickable") is not False else False
        elif row_type == "fund":
            row["asset_region"] = row.get("asset_region") or "TR"
            holding_code = _holding_code(row)
            fund_row = fund_rows.get(holding_code) or {}
            provider_name = str(
                row.get("provider_name")
                or fund_row.get("founder_company")
                or fund_row.get("manager_company")
                or ""
            ).strip()
            if provider_name:
                row["provider_name"] = provider_name
                row["logo_symbol"] = row.get("logo_symbol") or provider_name
            if not row.get("logo_url") and fund_row.get("logo_url"):
                row["logo_url"] = fund_row.get("logo_url")
                row["logo_source"] = fund_row.get("logo_source")

        sector_info = sector_map.get(_holding_code(row)) if row_type == "local_equity" else None
        row["sector_code"] = sector_info.get("sector_code") if sector_info else None
        row["sector_label"] = sector_info.get("sector_label") if sector_info else None
        daily_fields = _position_daily_market_fields(
            row,
            stock_quotes=stock_quotes,
            gefas_quotes=gefas_quotes,
            foreign_quotes=foreign_quotes,
            fund_rows=fund_rows,
        )
        row.update(daily_fields)
        # Mark fund-type positions as TEFAS-tradable when they exist in the
        # daily TEFAS funds snapshot (or have a GEFAS-GYF override).  The
        # frontend uses this flag to decide whether the row is clickable.
        if row_type == "fund":
            holding_code = _holding_code(row)
            row["tefas_tradable"] = bool(
                holding_code
                and (holding_code in fund_rows or _gefas_gyf_config(holding_code))
            )
            row["detail_clickable"] = bool(row["tefas_tradable"])
        elif row_type == "foreign_fund":
            row["tefas_tradable"] = False
            row["detail_clickable"] = False
        else:
            row["tefas_tradable"] = None

        weight = _api_number(row.get("weight"))
        return_pct = _api_number(row.get("return_pct"))
        exposure_value = (fund_aum * weight / 100.0) if fund_aum is not None and weight is not None and weight > 0 else None
        contribution_pct = (weight * return_pct / 100.0) if weight is not None and weight > 0 and return_pct is not None else None
        pnl_value = (exposure_value * return_pct / 100.0) if exposure_value is not None and return_pct is not None else None

        row["estimated_exposure_value"] = round(exposure_value, 2) if exposure_value is not None else None
        row["estimated_pnl_value"] = round(pnl_value, 2) if pnl_value is not None else None
        row["estimated_fund_return_contribution_pct"] = round(contribution_pct, 6) if contribution_pct is not None else None

        if weight is not None and weight > 0:
            if return_pct is not None:
                priced_weight += weight
                estimated_return_pct += contribution_pct or 0.0
                if pnl_value is not None:
                    estimated_pnl_value += pnl_value
                    has_pnl = True
            else:
                missing_weight += weight
        enriched_positions.append(row)

    enriched_payload = dict(payload)
    enriched_payload["positions"] = enriched_positions
    enriched_payload["portfolio_effect"] = {
        "period": "daily",
        "estimated_return_pct": round(estimated_return_pct, 6),
        "estimated_pnl_value": round(estimated_pnl_value, 2) if has_pnl else None,
        "priced_weight": round(priced_weight, 6),
        "missing_weight": round(missing_weight, 6),
        "aum": fund_aum,
        "as_of": fund_row.get("as_of") or (payload.get("source_metadata") or {}).get("as_of"),
    }
    metadata = dict(enriched_payload.get("source_metadata") or {})
    metadata["holdings_quality"] = holdings_quality
    metadata["daily_market_enrichment"] = {
        "period": "daily",
        "stock_quote_count": len(stock_quotes),
        "fund_snapshot_count": len(fund_rows),
        "gefas_gyf_quote_count": len(gefas_quotes),
        "gefas_gyf_quote_cache_hits": gefas_quote_cache_hits,
        "foreign_quote_count": len(foreign_quotes),
        "foreign_quote_missing_count": sum(
            1
            for position in positions
            if _holding_type(position) in {"foreign_equity", "foreign_fund"}
            and _foreign_holding_provider_symbol(position) not in foreign_quotes
        ),
        "sector_symbol_count": sector_meta.get("symbol_count"),
        "sector_cache_hit": sector_meta.get("cache_hit"),
        "sector_source": sector_meta.get("source"),
        "sector_source_date": sector_meta.get("source_date"),
        "daily_reference_cache_hit": bool(fund_rows_meta.get("cache_hit")),
        "daily_reference_row_count": fund_rows_meta.get("row_count"),
        "priced_weight": round(priced_weight, 6),
        "missing_weight": round(missing_weight, 6),
    }
    metadata["market_enrichment"] = {
        "stock_quote_live": True,
        "stock_quote_count": len(stock_quotes),
        "fund_daily_reference": "tefas_snapshot",
        "fund_daily_reference_cache_hit": bool(fund_rows_meta.get("cache_hit")),
        "gefas_gyf_cache_ttl_seconds": _GEFAS_GYF_QUOTE_CACHE_TTL,
        "foreign_daily_reference": "yahoo_finance_chart",
        "foreign_quote_count": len(foreign_quotes),
    }
    enriched_payload["source_metadata"] = metadata
    return enriched_payload


@app.get("/funds/{fund_code}/allocations")
def fund_allocations(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_allocations_payload

    return get_fund_allocations_payload(CONFIG.paths.processed_dir, fund_code)


def _allocation_history_job_key(fund_code: str, lookback_days: int) -> str:
    return f"api:fund-allocation-history:job:{fund_code}:{lookback_days}"


def _allocation_history_active_key(fund_code: str, lookback_days: int) -> str:
    return f"api:fund-allocation-history:active:{fund_code}:{lookback_days}"


def _allocation_history_job_get(fund_code: str, lookback_days: int, job_id: str) -> Optional[Dict[str, Any]]:
    value = _get_cache().get(f"{_allocation_history_job_key(fund_code, lookback_days)}:{job_id}")
    return dict(value) if isinstance(value, dict) else None


def _allocation_history_job_set(job: Dict[str, Any]) -> None:
    _get_cache().set(
        f"{_allocation_history_job_key(str(job['fund_code']), int(job['lookback_days']))}:{job['job_id']}",
        job,
        ttl_seconds=_FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS,
    )


def _allocation_history_job_public(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job:
        return None
    return {
        key: job.get(key)
        for key in ("job_id", "fund_code", "lookback_days", "status", "requested_at", "started_at", "finished_at", "error")
    }


def _run_allocation_history_refresh_job(fund_code: str, lookback_days: int, job_id: str) -> None:
    from app.fund_service import refresh_fund_allocations_history

    normalized = fund_code.strip().upper()
    backend = _get_cache()
    job = _allocation_history_job_get(normalized, lookback_days, job_id)
    if not job:
        return
    active_key = _allocation_history_active_key(normalized, lookback_days)
    lease_key = f"{active_key}:lease"
    owner = uuid.uuid4().hex
    if not backend.set_if_absent(lease_key, owner, ttl_seconds=_FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS):
        # A different worker has the shared lease. Its result will populate the
        # same file/cache; keep this job informational rather than duplicating
        # upstream TEFAS traffic.
        job.update({"status": "superseded", "finished_at": _fund_refresh_now_iso(), "error": None})
        _allocation_history_job_set(job)
        return

    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(max(1.0, _FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS)):
            backend.renew_if_owner(
                lease_key,
                owner,
                ttl_seconds=_FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS,
            )
            backend.renew_if_owner(
                active_key,
                job_id,
                ttl_seconds=_FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS,
            )

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"fund-allocation-history-heartbeat-{normalized}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        job.update({"status": "running", "started_at": _fund_refresh_now_iso(), "error": None})
        _allocation_history_job_set(job)
        refresh_fund_allocations_history(
            CONFIG.paths.processed_dir,
            normalized,
            lookback_days=lookback_days,
            allow_daily_fallback=True,
        )
        job.update({"status": "succeeded", "finished_at": _fund_refresh_now_iso(), "error": None})
        _allocation_history_job_set(job)
    except Exception as exc:
        job.update({"status": "failed", "finished_at": _fund_refresh_now_iso(), "error": str(exc)})
        _allocation_history_job_set(job)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=max(1.0, _FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS))
        backend.release_if_owner(lease_key, owner)
        if backend.get(active_key) == job_id:
            backend.delete(active_key)


def _start_allocation_history_refresh_job(fund_code: str, lookback_days: int) -> Dict[str, Any]:
    """Start at most one allocation-history refresh for a fund/range pair."""

    normalized = fund_code.strip().upper()
    bounded_lookback = max(1, min(365, int(lookback_days)))
    backend = _get_cache()
    active_key = _allocation_history_active_key(normalized, bounded_lookback)
    with backend.lock(f"fund-allocation-history-state:{normalized}:{bounded_lookback}", timeout=5.0) as acquired:
        if acquired:
            active_id = backend.get(active_key)
            if active_id:
                existing = _allocation_history_job_get(normalized, bounded_lookback, str(active_id))
                if existing and str(existing.get("status")) in {"queued", "running"}:
                    return existing
            job = {
                "job_id": uuid.uuid4().hex,
                "fund_code": normalized,
                "lookback_days": bounded_lookback,
                "status": "queued",
                "requested_at": _fund_refresh_now_iso(),
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            _allocation_history_job_set(job)
            backend.set(active_key, job["job_id"], ttl_seconds=_FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS)
            try:
                _FUND_ALLOCATION_HISTORY_EXECUTOR.submit(
                    _run_allocation_history_refresh_job,
                    normalized,
                    bounded_lookback,
                    str(job["job_id"]),
                )
            except Exception as exc:
                job.update({"status": "failed", "finished_at": _fund_refresh_now_iso(), "error": str(exc)})
                _allocation_history_job_set(job)
                backend.delete(active_key)
            return job
    active_id = backend.get(active_key)
    if active_id:
        existing = _allocation_history_job_get(normalized, bounded_lookback, str(active_id))
        if existing:
            return existing
    # A short lock contention should never turn into synchronous TEFAS work.
    return {
        "job_id": None,
        "fund_code": normalized,
        "lookback_days": bounded_lookback,
        "status": "pending",
        "requested_at": _fund_refresh_now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": "allocation refresh queue is busy",
    }


@app.get("/funds/{fund_code}/allocations/history")
def fund_allocations_history(
    fund_code: str,
    lookback_days: int = Query(30, ge=1, le=365),
) -> Any:
    from app.fund_service import get_fund_allocations_history_payload

    normalized = _require_known_fund_code(fund_code)
    payload = get_fund_allocations_history_payload(
        CONFIG.paths.processed_dir,
        normalized,
        lookback_days=lookback_days,
        auto_refresh=False,
    )
    needs_refresh = bool(payload.get("stale")) or str(payload.get("status")) == "pending"
    if not needs_refresh:
        payload["refresh_pending"] = False
        return payload
    job = _start_allocation_history_refresh_job(normalized, lookback_days)
    response = dict(payload)
    response["refresh_pending"] = str(job.get("status")) in {"queued", "running", "pending"}
    metadata = dict(response.get("source_metadata") or {})
    metadata["allocation_history_job"] = _allocation_history_job_public(job)
    response["source_metadata"] = metadata
    if not response.get("history"):
        return JSONResponse(status_code=202, content=response)
    return response


@app.get("/funds/{fund_code}")
def fund_detail(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        return _fund_detail_payload(normalized=normalized)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}") from exc


@_cached_response(
    key_fn=lambda *, normalized: f"api:fund-detail:{normalized}",
    ttl_seconds=60,
    skip_when=lambda *, normalized: not normalized,
)
def _fund_detail_payload(*, normalized: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_detail_payload

    return get_fund_detail_payload(CONFIG.paths.processed_dir, normalized)


@app.post("/admin/funds/refresh-snapshot")
def admin_refresh_funds_snapshot(
    request: Request,
    lookback_days: int = Query(_FUND_REFRESH_MAX_LOOKBACK_DAYS, ge=1, le=_FUND_REFRESH_MAX_LOOKBACK_DAYS)
) -> Dict[str, Any]:
    from app.fund_service import get_funds_payload

    _require_admin_refresh_access(request)
    # Start the upstream work outside the request path. The current snapshot
    # is returned immediately so a slow TEFAS/WAF response cannot freeze the UI.
    job = _start_fund_refresh_job(lookback_days)
    payload = get_funds_payload(
        CONFIG.paths.processed_dir,
        sort="fund_code",
        order="asc",
        auto_refresh=False,
    )
    payload["refresh_job"] = job
    return payload


def _require_known_fund_code(fund_code: str) -> str:
    """Reject arbitrary cache/job keys before an upstream refresh is queued."""

    from app.fund_service import get_fund_detail_payload, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        get_fund_detail_payload(CONFIG.paths.processed_dir, normalized)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}") from exc
    return normalized


@app.get("/admin/funds/refresh-snapshot/status")
def fund_refresh_snapshot_status(
    request: Request,
    job_id: str = Query(..., min_length=8, max_length=64),
) -> Dict[str, Any]:
    _require_admin_refresh_access(request)
    job = _get_fund_refresh_job(job_id.strip())
    if job is None:
        raise HTTPException(status_code=404, detail="Fon yenileme işi bulunamadı veya süresi doldu.")
    return {"refresh_job": job}


def _invalidate_fund_response_cache() -> None:
    """Drop cached fund responses after a snapshot refresh."""

    backend = _get_cache()
    for prefix in (
        "api:funds:",
        "api:funds-search:",
        "api:fund-performance:",
        "api:fund-detail:",
        "api:fund-yield-summary:",
        "api:fund-holdings:",
        "api:funds:snapshot-row-map:",
        "api:funds:holding-sector-map:",
    ):
        try:
            backend.delete_prefix(prefix)
        except Exception:
            continue
    try:
        backend.delete("api:funds-categories")
    except Exception:
        pass
    try:
        backend.delete("api:funds-categories:v2")
    except Exception:
        pass


def _invalidate_single_fund_response_cache(normalized: str) -> None:
    backend = _get_cache()
    for key_or_prefix in (
        f"api:fund-performance:{normalized}:",
        f"api:fund-performance:v{_FUND_HISTORY_KEY_VERSION}:{normalized}:",
        f"api:fund-detail:{normalized}",
        f"api:fund-yield-summary:{normalized}",
        f"api:fund-holdings:{normalized}:",
    ):
        try:
            if key_or_prefix.endswith(":"):
                backend.delete_prefix(key_or_prefix)
            else:
                backend.delete(key_or_prefix)
        except Exception:
            continue


@app.post("/admin/funds/collect-prices")
def admin_collect_fund_prices(
    request: Request,
    lookback_days: int = Query(10, ge=1, le=45),
    as_of: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import collect_daily_fund_prices

    _require_admin_refresh_access(request)
    return collect_daily_fund_prices(
        CONFIG.paths.processed_dir,
        as_of=as_of,
        lookback_days=lookback_days,
    )


@app.post("/admin/funds/{fund_code}/refresh-performance")
def admin_refresh_fund_performance(
    request: Request,
    fund_code: str,
    start_date: date = Query(...),
    end_date: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import FundUpstreamError, refresh_fund_performance, normalize_fund_code

    _require_admin_refresh_access(request)
    normalized = normalize_fund_code(fund_code)
    effective_end_date = end_date or date.today()
    if start_date > effective_end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    if (effective_end_date - start_date).days > _ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"start_date en fazla {_ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS} gun geriye gidebilir",
        )
    _require_known_fund_code(normalized)
    try:
        result = refresh_fund_performance(
            CONFIG.paths.processed_dir,
            normalized,
            start_date=start_date,
            end_date=effective_end_date,
        )
    except FundUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _invalidate_single_fund_response_cache(normalized)
    return result


@app.post("/admin/funds/{fund_code}/refresh-allocations")
def admin_refresh_fund_allocations(
    request: Request,
    fund_code: str,
    as_of: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import FundUpstreamError, normalize_fund_code, refresh_fund_allocations

    _require_admin_refresh_access(request)
    normalized = normalize_fund_code(fund_code)
    _require_known_fund_code(normalized)
    try:
        result = refresh_fund_allocations(CONFIG.paths.processed_dir, normalized, as_of=as_of)
    except FundUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _invalidate_single_fund_response_cache(normalized)
    return result


@app.post("/kap/overview-commentary")
async def kap_overview_commentary(request: Request) -> Dict[str, Any]:
    started_at = time.perf_counter()
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        LOGGER.warning(
            "[kap_overview_commentary] request body too large | bytes=%s limit=%s",
            len(body),
            MAX_REQUEST_BYTES,
        )
        raise HTTPException(status_code=413, detail="request body en fazla 64 KB olabilir")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "[kap_overview_commentary] invalid json body | bytes=%s error=%s",
            len(body),
            exc,
        )
        raise HTTPException(status_code=400, detail="gecerli JSON body gerekli") from exc
    LOGGER.info(
        "[kap_overview_commentary] request received | company=%s latest_period=%s bytes=%s",
        str(payload.get("company") or "").strip(),
        str(payload.get("latest_period") or "").strip(),
        len(body),
    )
    try:
        response = await _run_overview_commentary_until_done_or_disconnected(request, payload)
    except PayloadValidationError as exc:
        LOGGER.warning(
            "[kap_overview_commentary] payload validation failed | company=%s detail=%s",
            str(payload.get("company") or "").strip(),
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    LOGGER.info(
        "[kap_overview_commentary] completed | company=%s ok=%s model=%s score_source=%s elapsed_ms=%s error=%s",
        str(payload.get("company") or "").strip(),
        response.get("ok"),
        response.get("model_used"),
        ((response.get("scorecard") or {}) if isinstance(response.get("scorecard"), dict) else {}).get("score_source"),
        elapsed_ms,
        (str(response.get("error") or "")[:180] or None),
    )
    return response


async def _run_overview_commentary_until_done_or_disconnected(
    request: Request,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    task = asyncio.create_task(generate_overview_commentary(payload))
    company = str(payload.get("company") or "").strip()
    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                LOGGER.info(
                    "[kap_overview_commentary] client disconnected; cancelling NVIDIA request | company=%s",
                    company,
                )
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise HTTPException(status_code=499, detail="client disconnected")
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
            except asyncio.TimeoutError:
                continue
        return await task
    except asyncio.CancelledError:
        task.cancel()
        LOGGER.info(
            "[kap_overview_commentary] request task cancelled | company=%s",
            company,
        )
        raise


_FLOW_CACHE: Dict[str, Any] = {}
_FLOW_CACHE_TTL = 180
# When the VYK feed fetch budget is spent we still want the last successful
# payload served for a while even if the cache itself is stale.
_FLOW_STALE_SERVE_WINDOW = 15 * 60


def _parse_kap_publish_date(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    token = str(raw).strip()
    if not token:
        return None
    for fmt in (
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


# Map KAP disclosureType codes -> Turkish UI labels for the feed.
_KAP_TYPE_LABELS: Dict[str, str] = {
    "ODA": "Özel Durum",
    "FR": "Finansal Rapor",
    "FR_Consolidated": "Finansal Rapor",
    "FR_Solo": "Finansal Rapor",
    "KBR": "Kâr Payı",
    "DD": "Diğer Duyuru",
    "MD": "Mali Duyuru",
    "GK": "Genel Kurul",
    "FDR": "Faaliyet Raporu",
    "GR": "Geri Alım",
    "SR": "Sürdürülebilirlik",
    "CG": "Kurumsal Yönetim",
}


def _kap_category(disclosure_type: str, subject: str) -> str:
    dt = (disclosure_type or "").strip().upper()
    subj = (subject or "").lower()
    if dt.startswith("FR"):
        return "finansal_rapor"
    if "kar pay" in subj or dt == "KBR":
        return "kar_payi"
    if "geri alma" in subj or "geri alım" in subj or dt == "GR":
        return "geri_alim"
    if "genel kurul" in subj or dt == "GK":
        return "genel_kurul"
    if "kredi derec" in subj:
        return "kredi_derecelendirme"
    if "sürdürülebilir" in subj or dt == "SR":
        return "surdurulebilirlik"
    if "faaliyet rapor" in subj or dt == "FDR":
        return "faaliyet_raporu"
    return "bildirim"


def _kap_source_label(disclosure_type: str) -> str:
    dt = (disclosure_type or "").strip().upper()
    return _KAP_TYPE_LABELS.get(dt, "KAP")


_KAP_PUBLIC_LAST_ERROR: Dict[str, Any] = {"message": None, "ts": 0.0, "source": None}
_KAP_SESSION: Dict[str, Any] = {"opener": None, "bootstrapped_at": 0.0}
_KAP_SESSION_TTL = 15 * 60  # 15 dakika session yeniden kurulur


_KAP_DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    "Accept-Encoding": "identity",
}


def _kap_opener(force: bool = False) -> Any:
    """Build (and cache) a cookie-aware opener for kap.org.tr.

    Bootstraps a browser-like session by first requesting the HTML search page
    so KAP's WAF issues the cookies that later JSON calls require. Cached
    until TTL expires or `force=True`.
    """
    import http.cookiejar
    import urllib.error
    import urllib.request

    now = time.time()
    opener = _KAP_SESSION.get("opener")
    bootstrapped_at = float(_KAP_SESSION.get("bootstrapped_at") or 0.0)
    if opener is not None and not force and (now - bootstrapped_at) < _KAP_SESSION_TTL:
        return opener

    jar = http.cookiejar.CookieJar()
    new_opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPRedirectHandler(),
    )
    new_opener.addheaders = list(_KAP_DEFAULT_HEADERS.items())

    bootstrap_urls = [
        "https://www.kap.org.tr/tr/",
        "https://www.kap.org.tr/tr/bildirim-sorgu",
    ]
    bootstrap_results: List[Dict[str, Any]] = []
    for boot_url in bootstrap_urls:
        boot_started = time.time()
        try:
            req = urllib.request.Request(
                boot_url,
                headers={
                    **_KAP_DEFAULT_HEADERS,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Upgrade-Insecure-Requests": "1",
                },
            )
            with new_opener.open(req, timeout=8) as resp:
                resp.read(1024)
            bootstrap_results.append(
                {
                    "url": boot_url,
                    "ok": True,
                    "elapsed_ms": int((time.time() - boot_started) * 1000),
                    "cookies": len(list(jar)),
                }
            )
        except (urllib.error.URLError, TimeoutError, Exception) as exc:  # noqa: BLE001
            bootstrap_results.append(
                {
                    "url": boot_url,
                    "ok": False,
                    "elapsed_ms": int((time.time() - boot_started) * 1000),
                    "error": type(exc).__name__,
                    "cookies": len(list(jar)),
                }
            )
            continue

    _KAP_SESSION["opener"] = new_opener
    _KAP_SESSION["bootstrapped_at"] = now
    # region agent log
    _debug_log(
        "H1",
        "app/api.py:1230",
        "KAP bootstrap completed",
        {
            "force": force,
            "cookie_count": len(list(jar)),
            "results": bootstrap_results,
        },
    )
    # endregion
    return new_opener


def _fetch_kap_disclosures_via_url(
    url: str,
    timeout: float,
    opener: Any,
) -> tuple[Any, Optional[str]]:
    """Single attempt JSON fetch using the shared KAP session opener."""
    import urllib.error
    import urllib.request

    headers = {
        **_KAP_DEFAULT_HEADERS,
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    started = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        payload = json.loads(raw)
        # region agent log
        _debug_log(
            "H2",
            "app/api.py:1266",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": True,
                "elapsed_ms": int((time.time() - started) * 1000),
                "payload_type": type(payload).__name__,
                "item_count": len(payload) if isinstance(payload, list) else None,
            },
        )
        # endregion
        return payload, None
    except urllib.error.HTTPError as exc:
        # region agent log
        _debug_log(
            "H3",
            "app/api.py:1281",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": False,
                "elapsed_ms": int((time.time() - started) * 1000),
                "error": f"HTTPError {exc.code}",
            },
        )
        # endregion
        return None, f"HTTPError {exc.code}"
    except (urllib.error.URLError, TimeoutError, ValueError, Exception) as exc:  # noqa: BLE001
        error_text = f"{type(exc).__name__}: {exc}"
        # region agent log
        _debug_log(
            "H2",
            "app/api.py:1295",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": False,
                "elapsed_ms": int((time.time() - started) * 1000),
                "error": error_text,
            },
        )
        # endregion
        return None, error_text


def _fetch_kap_public_disclosures(max_items: int = 80) -> List[Dict[str, Any]]:
    """Fetch recent disclosures from KAP's public (UI-facing) endpoint.

    This endpoint does not require authentication; it backs the KAP.org.tr
    "Bildirim Sorgu" screen. Returns a list in publishedAt-descending order.
    In this environment KAP's public disclosure feed may be WAF-protected.
    We probe the fastest-blocking variant first so repeated refreshes do not
    spend ~15-20 seconds timing out before falling back.
    """
    attempts = [
        (
            "https://www.kap.org.tr/tr/api/disclosures?main-category=all&sub-category=all&memberType=IGS",
            10.0,
            True,
        ),
        ("https://www.kap.org.tr/tr/api/disclosures", 6.0, False),
        ("https://www.kap.org.tr/tr/api/disclosures", 9.0, True),
    ]
    payload: Any = None
    last_error: Optional[str] = None
    last_source: Optional[str] = None
    for idx, (url, timeout, force_bootstrap) in enumerate(attempts):
        opener = _kap_opener(force=force_bootstrap)
        payload, last_error = _fetch_kap_disclosures_via_url(url, timeout, opener)
        last_source = url
        if isinstance(payload, list):
            last_error = None
            break
        if last_error == "HTTPError 666":
            # region agent log
            _debug_log(
                "H6",
                "app/api.py:1368",
                "KAP public feed blocked, skipping slower retries",
                {
                    "url": url,
                    "attempt_index": idx,
                    "error": last_error,
                },
            )
            # endregion
            break
        if idx < len(attempts) - 1:
            time.sleep(0.35)

    _KAP_PUBLIC_LAST_ERROR["message"] = last_error
    _KAP_PUBLIC_LAST_ERROR["ts"] = time.time()
    _KAP_PUBLIC_LAST_ERROR["source"] = last_source

    if not isinstance(payload, list):
        return []

    results: List[Dict[str, Any]] = []
    for node in payload:
        if not isinstance(node, dict):
            continue
        basic = node.get("basic") if isinstance(node.get("basic"), dict) else node
        if not isinstance(basic, dict):
            continue
        disclosure_index = basic.get("disclosureIndex")
        publish_raw = basic.get("publishDate") or basic.get("submittedDate") or basic.get("disclosureClass")
        parsed_dt = _parse_kap_publish_date(publish_raw)
        if parsed_dt is None:
            continue
        stock_codes_raw = str(basic.get("stockCodes") or basic.get("stockCode") or "").strip()
        stock_codes = [s.strip().upper() for s in stock_codes_raw.replace(";", ",").split(",") if s.strip()]
        symbol = stock_codes[0] if stock_codes else ""
        if not symbol:
            # Non-listed disclosures (e.g. regulator notes) — skip in ticker-centric feed
            continue

        title_candidates = [
            basic.get("title"),
            basic.get("summary"),
            (basic.get("kapTitle") or {}).get("tr") if isinstance(basic.get("kapTitle"), dict) else None,
            basic.get("subject"),
        ]
        title = ""
        for candidate in title_candidates:
            if candidate and str(candidate).strip():
                title = str(candidate).strip()
                break
        if not title:
            title = "KAP Bildirimi"

        subject = str(basic.get("subject") or "").strip()
        disclosure_type = str(basic.get("disclosureType") or basic.get("type") or "").strip()

        results.append(
            {
                "id": f"{symbol}-{disclosure_index or parsed_dt.isoformat()}",
                "source": _kap_source_label(disclosure_type),
                "symbol": symbol,
                "stock_codes": stock_codes,
                "title": title,
                "subject": subject,
                "published_at": parsed_dt.isoformat(),
                "category": _kap_category(disclosure_type, subject),
                "kap_url": (
                    f"https://www.kap.org.tr/tr/Bildirim/{disclosure_index}"
                    if disclosure_index is not None
                    else None
                ),
            }
        )
        if len(results) >= max_items:
            break

    return results


def _local_flow_items_from_cache() -> List[Dict[str, Any]]:
    """Fallback feed constructed from locally cached financial reports."""
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    started = time.time()
    items: List[Dict[str, Any]] = []
    if not cache_dir.exists():
        # region agent log
        _debug_log(
            "H4",
            "app/api.py:1391",
            "Local flow cache scan finished",
            {"cache_dir_exists": False, "file_count": 0, "item_count": 0, "elapsed_ms": 0},
        )
        # endregion
        return items
    cache_files = list(cache_dir.glob("*.json"))
    for cache_file in cache_files:
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        symbol = str(
            payload.get("stock_code") or payload.get("company") or cache_file.stem
        ).strip().upper()
        quarters = payload.get("quarters")
        if not isinstance(quarters, list):
            continue
        for quarter in quarters[:2]:
            if not isinstance(quarter, dict):
                continue
            parsed_dt = _parse_kap_publish_date(quarter.get("publish_date"))
            if parsed_dt is None:
                continue
            title = str(quarter.get("title") or "Finansal Rapor").strip()
            quarter_label = str(quarter.get("quarter") or "").strip()
            disclosure_id = quarter.get("disclosure_index")
            items.append(
                {
                    "id": f"{symbol}-{disclosure_id or quarter_label or parsed_dt.isoformat()}",
                    "source": "Finansal Rapor",
                    "symbol": symbol,
                    "stock_codes": [symbol],
                    "title": f"{title}{' - ' + quarter_label if quarter_label else ''}",
                    "subject": "Finansal Rapor",
                    "published_at": parsed_dt.isoformat(),
                    "category": "finansal_rapor",
                    "kap_url": (
                        f"https://www.kap.org.tr/tr/Bildirim/{disclosure_id}"
                        if disclosure_id
                        else None
                    ),
                }
            )
    # region agent log
    _debug_log(
        "H4",
        "app/api.py:1432",
        "Local flow cache scan finished",
        {
            "cache_dir_exists": True,
            "file_count": len(cache_files),
            "item_count": len(items),
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    return items


_FLOW_DEGRADED_TTL = 25  # Tekrar canlı kaynağı dene: 25 sn sonra


# region VYK (resmi KAP REST) yardimcilari
# Son 24 saat disiplini icin varsayilan pencere. Kullanici akisin cok uzun
# bir zaman dilimini cekmesini istemedigi icin dar tutuluyor.
_VYK_DEFAULT_WINDOW_HOURS = 24
# `/disclosures` cagrisi 50 kayit dondurur; bu bugette `disclosureDetail`
# icin iki-sayfa disinda kalmalik ayirir.
_VYK_DEFAULT_LIST_BUDGET = 2
# Her refresh'te en fazla bu kadar `disclosureDetail` cagrisi yapilir;
# gerisi sessizce atlanir. Gateway'in "cok fazla istek" sikayetini onler.
_VYK_DEFAULT_DETAIL_BUDGET = 25
# Kullanici akisi genisletmek isteyebilir; bu sinir gateway'i bunaltmadan
# saglik sinirinda tutar.
_VYK_DEFAULT_DETAIL_BUDGET_MAX = 500
_VYK_DETAIL_WORKERS = 8


def _vyk_source_label(disclosure_class: str, disclosure_type: str, subject_tr: str) -> str:
    cls = (disclosure_class or "").upper().strip()
    typ = (disclosure_type or "").upper().strip()
    subj = (subject_tr or "").lower()
    if typ == "CA":
        if "kar pay" in subj:
            return "Kâr Payı"
        if "genel kurul" in subj:
            return "Genel Kurul"
        if "geri al" in subj:
            return "Geri Alım"
        if "sermaye" in subj:
            return "Sermaye Artırımı"
        return "Hak Kullanımı"
    if typ == "FON":
        return "Fon"
    if typ.startswith("FR") or cls == "FR":
        return "Finansal Rapor"
    if typ == "ODA" or cls == "ODA":
        return "Özel Durum"
    if typ == "DUY" or cls == "DUY":
        return "Düzenleyici Kurum"
    if typ == "DG" or cls == "DG":
        return "Diğer Bildirim"
    return "KAP"


def _vyk_category(disclosure_class: str, disclosure_type: str, subject_tr: str) -> str:
    cls = (disclosure_class or "").upper().strip()
    typ = (disclosure_type or "").upper().strip()
    subj = (subject_tr or "").lower()
    if typ.startswith("FR") or cls == "FR":
        return "finansal_rapor"
    if typ == "ODA" or cls == "ODA":
        return "ozel_durum"
    if "kar pay" in subj:
        return "kar_payi"
    if "genel kurul" in subj:
        return "genel_kurul"
    if "geri al" in subj:
        return "geri_alim"
    if "kredi derec" in subj:
        return "kredi_derecelendirme"
    if "sürdürülebilir" in subj:
        return "surdurulebilirlik"
    if "faaliyet rapor" in subj:
        return "faaliyet_raporu"
    return "bildirim"


def _fetch_kap_vyk_feed(
    *,
    window_hours: int = _VYK_DEFAULT_WINDOW_HOURS,
    list_pages: int = _VYK_DEFAULT_LIST_BUDGET,
    detail_budget: int = _VYK_DEFAULT_DETAIL_BUDGET,
) -> List[Dict[str, Any]]:
    """Build a flow feed from the official VYK REST endpoints.

    Returns feed items sorted by `published_at` desc, filtered to the last
    `window_hours` hours. Returns `[]` when credentials are missing or any
    upstream call fails so the caller can fall back to the local cache.
    """
    import concurrent.futures

    from src import kap_vyk_client

    cfg = getattr(CONFIG, "kap", None)
    if cfg is None or not kap_vyk_client.is_enabled(cfg):
        return []

    started = time.time()
    last_index = kap_vyk_client.get_last_disclosure_index(cfg)
    if not last_index or last_index <= 0:
        return []

    # `/disclosures` sayfalari 50 kayitlik. `list_pages` ile toplam pencereyi
    # kontrol altinda tutup upstream'e bindirmiyoruz.
    pages = max(1, min(int(list_pages or 1), 10))
    disclosures: List[Dict[str, Any]] = []
    cursor = int(last_index)
    seen_indexes: set[str] = set()
    for _ in range(pages):
        start_index = max(1, cursor - 49)
        rows = kap_vyk_client.list_disclosures_batch(cfg, start_index=start_index)
        if not rows:
            break
        added = 0
        for row in rows:
            idx = str(row.get("disclosureIndex") or "").strip()
            if not idx or idx in seen_indexes:
                continue
            seen_indexes.add(idx)
            disclosures.append(row)
            added += 1
        if added == 0:
            break
        cursor = start_index - 1
        if cursor <= 0:
            break

    if not disclosures:
        return []

    # Daha yeni kayitlari once isle; 24 saat penceresi disina taskinca
    # pahali detail cagrilarini kesmek icin bu sirali yaklasim sart.
    disclosures.sort(
        key=lambda node: int(str(node.get("disclosureIndex") or "0") or "0"),
        reverse=True,
    )

    members = kap_vyk_client.build_company_lookup(cfg)
    window_delta = timedelta(hours=max(1, int(window_hours)))
    now = datetime.now()
    cutoff = now - window_delta

    budget = max(1, min(int(detail_budget or 1), _VYK_DEFAULT_DETAIL_BUDGET_MAX))
    pending = disclosures[:budget]

    workers = max(1, min(_VYK_DETAIL_WORKERS, len(pending)))
    details: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                kap_vyk_client.get_disclosure_detail, cfg, row.get("disclosureIndex")
            ): str(row.get("disclosureIndex") or "")
            for row in pending
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                detail = future.result()
            except Exception:
                detail = None
            if detail and idx:
                details[idx] = detail

    items: List[Dict[str, Any]] = []
    fallback_items: List[Dict[str, Any]] = []
    for row in pending:
        idx_str = str(row.get("disclosureIndex") or "").strip()
        if not idx_str:
            continue
        detail = details.get(idx_str) or {}

        time_raw = str(detail.get("time") or "").strip()
        published_dt = _parse_kap_publish_date(time_raw)
        if published_dt is None:
            # Zaman damgasi yoksa akista saglikli konumlandiramayiz; atla.
            continue
        in_window = published_dt >= cutoff

        disclosure_class = str(
            row.get("disclosureClass") or detail.get("disclosureClass") or ""
        ).upper().strip()
        disclosure_type = str(
            row.get("disclosureType") or detail.get("disclosureType") or ""
        ).upper().strip()

        subject_obj = detail.get("subject")
        subject_tr = ""
        if isinstance(subject_obj, dict):
            subject_tr = str(subject_obj.get("tr") or "").strip()

        summary_obj = detail.get("summary")
        summary_tr = ""
        if isinstance(summary_obj, dict):
            summary_tr = str(summary_obj.get("tr") or "").strip()

        company_id = str(row.get("companyId") or detail.get("senderId") or "").strip()
        member = members.get(company_id, {})
        member_title = str(
            member.get("title") or row.get("title") or detail.get("senderTitle") or ""
        ).strip()
        member_stock_raw = str(member.get("stockCode") or "").strip().upper()
        member_stock = member_stock_raw.split(",")[0].strip() if member_stock_raw else ""

        sender_codes_raw = detail.get("senderExchCodes") or []
        stock_codes: List[str] = []
        primary_stock = ""
        if isinstance(sender_codes_raw, list) and sender_codes_raw:
            stock_codes = [
                str(code).strip().upper()
                for code in sender_codes_raw
                if str(code or "").strip()
            ]
            primary_stock = stock_codes[0] if stock_codes else ""
        if not primary_stock and member_stock:
            primary_stock = member_stock
            stock_codes = [member_stock]

        fund_code = str(
            detail.get("behalfFundCode") or row.get("fundCode") or ""
        ).strip().upper()
        symbol = primary_stock or fund_code or ""

        title = subject_tr or summary_tr or member_title or "KAP Bildirimi"
        subject = subject_tr or summary_tr

        built = {
            "id": f"vyk-{idx_str}",
            "source": _vyk_source_label(disclosure_class, disclosure_type, subject_tr),
            "symbol": symbol,
            "stock_codes": stock_codes,
            "title": title,
            "subject": subject,
            "published_at": published_dt.isoformat(),
            "category": _vyk_category(disclosure_class, disclosure_type, subject_tr),
            "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx_str}",
        }
        if in_window:
            items.append(built)
        else:
            # Pencere disi. Test gateway'i eski sabit veri dondurdugunde veya
            # KAP'ta uzun suredir yeni bildirim olmadiginda akisi bos birakmamak
            # icin sonra saglikli bir fallback kovasinda tutulur.
            fallback_items.append(built)

    if not items and fallback_items:
        # Pencere icinde hic kayit yoksa, zaten detay bedelini odedigimiz en
        # yeni kayitlari kullaniciya gosteriyoruz. Her senaryoda budget disinda
        # ekstra upstream istegi yok.
        items = fallback_items

    items.sort(key=lambda node: node.get("published_at") or "", reverse=True)

    used_fallback = not any(
        _parse_kap_publish_date(node.get("published_at")) and
        _parse_kap_publish_date(node.get("published_at")) >= cutoff  # type: ignore[operator]
        for node in items
    ) and bool(items)
    # region agent log
    _debug_log(
        "H13",
        "app/api.py:_fetch_kap_vyk_feed",
        "KAP VYK feed built",
        {
            "last_index": int(last_index),
            "list_pages": pages,
            "disclosures": len(disclosures),
            "detail_budget": budget,
            "detail_hits": len(details),
            "window_hours": int(window_hours),
            "items": len(items),
            "used_fallback": used_fallback,
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    return items


# endregion


# Member OID ve member feed cache'leri — canlı feed, BIST100 per-company
# endpoint'ini kullanarak KAP'ın listeleme endpoint'i engellendiğinde dahi
# canlı veri üretir.
_KAP_MEMBER_OID_CACHE: Dict[str, str] = {}
_KAP_MEMBER_OID_NEGATIVE_CACHE: Dict[str, float] = {}
_KAP_MEMBER_OID_NEGATIVE_TTL = 1800  # 30 dk
_KAP_MEMBER_OID_WARMED_FROM_CACHE = False
_MARKET_KAP_LOGO_CACHE: Dict[str, Any] = {}
_MARKET_KAP_LOGO_CACHE_TTL = 1800  # 30 dk
_KAP_MEMBER_FEED_CACHE: Dict[str, Any] = {"items": [], "ts": 0.0}
_KAP_MEMBER_FEED_TTL = 600  # 10 dk


def _fetch_kap_member_disclosures_for(symbol: str, year: int) -> List[Dict[str, Any]]:
    """Fetch recent financial disclosures for a single BIST company using the
    `listCompanyExcelMembers` endpoint which is NOT WAF-blocked.

    Returns [] on any error so the aggregator can continue for other companies.
    """
    import urllib.error
    import urllib.request

    oid = _KAP_MEMBER_OID_CACHE.get(symbol)
    if not oid:
        shared_oid = _shared_cache_get_dict(f"api:kap:member-oid:{symbol}:v1")
        if shared_oid and shared_oid.get("oid"):
            oid = str(shared_oid.get("oid") or "").strip()
            if oid:
                _KAP_MEMBER_OID_CACHE[symbol] = oid
    headers = {
        "Accept": "application/json",
        "Accept-Language": "tr",
        "User-Agent": _KAP_DEFAULT_HEADERS["User-Agent"],
    }
    if not oid:
        try:
            req = urllib.request.Request(
                f"https://www.kap.org.tr/tr/api/member/filter/{symbol}",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                rows = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(rows, list) and rows:
                oid = str(rows[0].get("mkkMemberOid") or "").strip()
                if oid:
                    _KAP_MEMBER_OID_CACHE[symbol] = oid
                    _shared_cache_set(
                        f"api:kap:member-oid:{symbol}:v1",
                        {"oid": oid},
                        ttl_seconds=24 * 60 * 60,
                    )
        except Exception:
            return []
    if not oid:
        return []
    try:
        req = urllib.request.Request(
            f"https://www.kap.org.tr/tr/api/financialTable/listCompanyExcelMembers/{oid}/{year}/T",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _resolve_kap_member_oid(symbol: str) -> Optional[str]:
    global _KAP_MEMBER_OID_WARMED_FROM_CACHE
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return None

    if not _KAP_MEMBER_OID_WARMED_FROM_CACHE:
        _KAP_MEMBER_OID_WARMED_FROM_CACHE = True
        kap_cache_dir = CONFIG.paths.processed_dir / "kap_cache"
        if kap_cache_dir.exists():
            for cache_file in kap_cache_dir.glob("*.json"):
                try:
                    with cache_file.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except Exception:
                    continue
                cache_symbol = str(payload.get("stock_code") or cache_file.stem).strip().upper()
                cache_oid = str(payload.get("member_oid") or "").strip()
                if cache_symbol and cache_oid and cache_symbol not in _KAP_MEMBER_OID_CACHE:
                    _KAP_MEMBER_OID_CACHE[cache_symbol] = cache_oid

    oid = _KAP_MEMBER_OID_CACHE.get(normalized)
    if oid:
        return oid
    shared_oid = _shared_cache_get_dict(f"api:kap:member-oid:{normalized}:v1")
    if shared_oid and shared_oid.get("oid"):
        oid = str(shared_oid.get("oid") or "").strip()
        if oid:
            _KAP_MEMBER_OID_CACHE[normalized] = oid
            return oid
    if _shared_cache_get_dict(f"api:kap:member-oid-miss:{normalized}:v1"):
        return None

    now = time.time()
    negative_ts = _KAP_MEMBER_OID_NEGATIVE_CACHE.get(normalized, 0.0)
    if negative_ts and now - negative_ts < _KAP_MEMBER_OID_NEGATIVE_TTL:
        return None

    import urllib.parse
    import urllib.request

    headers = {
        "Accept": "application/json",
        "Accept-Language": "tr",
        "User-Agent": _KAP_DEFAULT_HEADERS["User-Agent"],
    }
    try:
        encoded_symbol = urllib.parse.quote(normalized, safe="")
        req = urllib.request.Request(
            f"https://www.kap.org.tr/tr/api/member/filter/{encoded_symbol}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            rows = json.loads(resp.read().decode("utf-8", errors="ignore"))
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate = str(row.get("mkkMemberOid") or "").strip()
                if candidate:
                    _KAP_MEMBER_OID_CACHE[normalized] = candidate
                    _KAP_MEMBER_OID_NEGATIVE_CACHE.pop(normalized, None)
                    _shared_cache_set(
                        f"api:kap:member-oid:{normalized}:v1",
                        {"oid": candidate},
                        ttl_seconds=24 * 60 * 60,
                    )
                    return candidate
    except Exception:
        pass

    _KAP_MEMBER_OID_NEGATIVE_CACHE[normalized] = now
    _shared_cache_set(
        f"api:kap:member-oid-miss:{normalized}:v1",
        {"miss": True},
        ttl_seconds=_KAP_MEMBER_OID_NEGATIVE_TTL,
    )
    return None


def _kap_logo_payload_for_symbol(symbol: str) -> Dict[str, Optional[str]]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return {"logo_url": None, "logo_source": None}

    now = time.time()
    cached = _MARKET_KAP_LOGO_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_KAP_LOGO_CACHE_TTL:
        return dict(cached.get("data") or {"logo_url": None, "logo_source": None})
    shared_key = f"api:kap:logo:{normalized}:v1"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        _MARKET_KAP_LOGO_CACHE[normalized] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached or {"logo_url": None, "logo_source": None})

    oid = _resolve_kap_member_oid(normalized)
    data = {
        "logo_url": f"https://www.kap.org.tr/tr/api/member/logo/{oid}" if oid else None,
        "logo_source": "kap" if oid else None,
    }
    _MARKET_KAP_LOGO_CACHE[normalized] = {"_ts": now, "data": data}
    _shared_cache_set(shared_key, data, ttl_seconds=_MARKET_KAP_LOGO_CACHE_TTL)
    return dict(data)


def _empty_logo_payload() -> Dict[str, Optional[str]]:
    return {"logo_url": None, "logo_source": None}


def _synth_quarter_publish_dt(year: int, period: int) -> Optional[datetime]:
    """Approximate publish date from year+period when real publishDate absent."""
    approximate = {1: (5, 15), 2: (8, 15), 3: (11, 15), 4: (3, 15)}
    pair = approximate.get(int(period or 0))
    if not pair:
        return None
    month, day = pair
    y = int(year or 0) + (1 if int(period or 0) == 4 else 0)
    try:
        return datetime(y, month, day)
    except ValueError:
        return None


def _fetch_kap_member_feed(
    *,
    max_companies: int = 40,
    max_items: int = 160,
) -> List[Dict[str, Any]]:
    """Aggregate latest financial disclosures across BIST100 via working KAP endpoints.

    Returns feed items with best-effort `published_at` sourced from the local
    `kap_cache` (exact) when available, otherwise synthesized from year+period.
    Cached for `_KAP_MEMBER_FEED_TTL` seconds.
    """
    import concurrent.futures
    from app.kap_service import BIST100_SYMBOLS

    now = time.time()
    cache = _KAP_MEMBER_FEED_CACHE
    if cache.get("items") and (now - cache.get("ts", 0)) < _KAP_MEMBER_FEED_TTL:
        return list(cache["items"])
    shared_key = f"api:kap:member-feed:max={max_companies}:items={max_items}:v1"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        items = [dict(row) for row in list(shared_cached.get("items") or []) if isinstance(row, dict)]
        if items:
            cache["items"] = items
            cache["ts"] = now
            return list(items)

    started = time.time()
    symbols = list(BIST100_SYMBOLS[:max_companies])
    current_year = datetime.now(timezone.utc).year

    # Pre-load kap_cache publish dates keyed by disclosure_index for exact timestamps.
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    cache_publish: Dict[int, Dict[str, Any]] = {}
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            try:
                with cache_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            quarters = payload.get("quarters")
            if not isinstance(quarters, list):
                continue
            for q in quarters:
                if not isinstance(q, dict):
                    continue
                idx = q.get("disclosure_index")
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                cache_publish[idx_int] = {
                    "publish_date": q.get("publish_date"),
                    "quarter": q.get("quarter"),
                    "title": q.get("title"),
                }

    # Warm the OID cache from kap_cache files where available so the first
    # request doesn't pay a full `member/filter` lookup sweep.
    if not _KAP_MEMBER_OID_CACHE:
        kap_cache_dir = CONFIG.paths.processed_dir / "kap_cache"
        if kap_cache_dir.exists():
            for cache_file in kap_cache_dir.glob("*.json"):
                try:
                    with cache_file.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except Exception:
                    continue
                symbol = str(payload.get("stock_code") or cache_file.stem).strip().upper()
                oid = str(payload.get("member_oid") or "").strip()
                if symbol and oid:
                    _KAP_MEMBER_OID_CACHE[symbol] = oid

    def _gather(year_tasks: List[tuple[str, int]]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_fetch_kap_member_disclosures_for, sym, yr): (sym, yr)
                for sym, yr in year_tasks
            }
            for fut in concurrent.futures.as_completed(futures):
                sym, _ = futures[fut]
                try:
                    rows = fut.result()
                except Exception:
                    rows = []
                for row in rows:
                    row["_symbol"] = sym
                    collected.append(row)
        return collected

    current_tasks = [(sym, current_year) for sym in symbols]
    raw_rows: List[Dict[str, Any]] = _gather(current_tasks)
    # Backfill to the previous year only if current-year yield is thin.
    if len({row.get("_symbol") for row in raw_rows}) < max(8, len(symbols) // 4):
        raw_rows.extend(_gather([(sym, current_year - 1) for sym in symbols]))

    seen: set[int] = set()
    items: List[Dict[str, Any]] = []
    for row in raw_rows:
        try:
            idx = int(row.get("disclosureIndex") or 0)
        except Exception:
            continue
        if not idx or idx in seen:
            continue
        seen.add(idx)

        symbol = str(row.get("_symbol") or row.get("stockCode") or "").strip().upper()
        year = int(row.get("year") or 0)
        period = int(row.get("period") or 0)

        cache_meta = cache_publish.get(idx)
        publish_dt: Optional[datetime] = None
        quarter_label = ""
        title = ""
        if cache_meta:
            publish_dt = _parse_kap_publish_date(cache_meta.get("publish_date"))
            quarter_label = str(cache_meta.get("quarter") or "").strip()
            title = str(cache_meta.get("title") or "").strip()
        if publish_dt is None:
            publish_dt = _synth_quarter_publish_dt(year, period)
        if publish_dt is None:
            continue
        if not quarter_label and year and period:
            quarter_label = f"{year}Q{period}"
        if not title:
            title = "Finansal Rapor"

        items.append(
            {
                "id": f"{symbol}-{idx}",
                "source": "Finansal Rapor",
                "symbol": symbol,
                "stock_codes": [symbol],
                "title": f"{title}{' - ' + quarter_label if quarter_label else ''}",
                "subject": "Finansal Rapor",
                "published_at": publish_dt.isoformat(),
                "category": "finansal_rapor",
                "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx}",
            }
        )

    # Items without a kap_cache exact publish_date share the same synthesized
    # day; fall back to disclosureIndex as a tiebreaker so ordering matches
    # KAP's own allocation sequence (higher index = more recent).
    items.sort(
        key=lambda row: (
            row.get("published_at") or "",
            int(str(row.get("id") or "-0").rsplit("-", 1)[-1] or 0) if "-" in str(row.get("id") or "") else 0,
        ),
        reverse=True,
    )
    items = items[:max_items]

    # region agent log
    _debug_log(
        "H11",
        "app/api.py:_fetch_kap_member_feed",
        "KAP member feed built",
        {
            "symbol_count": len(symbols),
            "raw_rows": len(raw_rows),
            "unique_items": len(items),
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    cache["items"] = items
    cache["ts"] = now
    _shared_cache_set(shared_key, {"items": items}, ttl_seconds=_KAP_MEMBER_FEED_TTL)
    return list(items)


def _market_flow_payload(
    limit: int = 40,
    category: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    started = time.time()
    # Kullanici 'kac kayit' ayarini UI'dan degistirince backend'in VYK detay
    # butcesini de ona gore genisletmek istiyoruz; ayni zamanda cache'i bu
    # butceyle kademeli tutuyoruz ki kucuk secimle doldurulup buyukte kirilmasin.
    effective_budget = max(
        _VYK_DEFAULT_DETAIL_BUDGET,
        min(_VYK_DEFAULT_DETAIL_BUDGET_MAX, int(limit)),
    )
    effective_pages = max(1, min(10, (effective_budget + 49) // 50))
    cache_key = f"all::{category or ''}::b{effective_budget}"
    now = time.time()
    cached = _FLOW_CACHE.get(cache_key)
    if cached and not force_refresh:
        cached_data = cached["data"]
        ttl = _FLOW_DEGRADED_TTL if cached_data.get("degraded_mode") else _FLOW_CACHE_TTL
        if now - cached.get("_ts", 0) < ttl:
            return {**cached_data, "items": cached_data["items"][:limit]}
    shared_key = f"api:kap:market-flow:category={category or ''}:budget={effective_budget}:v1"
    if not force_refresh:
        shared_cached = _shared_cache_get_dict(shared_key)
        if shared_cached is not None:
            ttl = _FLOW_DEGRADED_TTL if shared_cached.get("degraded_mode") else _FLOW_CACHE_TTL
            _FLOW_CACHE[cache_key] = {"_ts": now, "data": shared_cached}
            return {**shared_cached, "items": list(shared_cached.get("items") or [])[:limit]}

    # Resmi KAP VYK akisi: credential'lar varsa en oncelikli kaynak.
    # UYARI: Kullanici kap.org.tr uzerinden canli veri istedigi icin VYK akisi gecici olarak devre disi birakildi.
    vyk_items: List[Dict[str, Any]] = []
    # vyk_items = _fetch_kap_vyk_feed(
    #     window_hours=_VYK_DEFAULT_WINDOW_HOURS,
    #     list_pages=effective_pages,
    #     detail_budget=effective_budget,
    # )
    
    # VYK boslarsa ya da credential yoksa resmi sitesi olan kap.org.tr uzerinden deneme yap.
    public_items: List[Dict[str, Any]] = []
    local_items: List[Dict[str, Any]] = []
    if not vyk_items:
        public_items = _fetch_kap_public_disclosures(max_items=limit)
        if not public_items:
            local_items = _local_flow_items_from_cache()

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in vyk_items + public_items + local_items:
        key = item.get("id") or ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(item)

    if category:
        merged = [row for row in merged if row.get("category") == category]

    merged.sort(key=lambda row: row.get("published_at") or "", reverse=True)

    public_error: Optional[str] = None
    if vyk_items:
        source = "kap_vyk"
        degraded = False
        multi_category_available = True
        warning: Optional[str] = None
    elif public_items:
        source = "kap_public_website"
        degraded = False
        multi_category_available = True
        warning = None
    else:
        source = "local_cache"
        degraded = True
        multi_category_available = False
        warning = (
            "Canlı KAP akışı şu an ulaşılamıyor; yalnızca yerel önbellekteki "
            "son finansal raporlar gösteriliyor."
        )

    data = {
        "items": merged[:_VYK_DEFAULT_DETAIL_BUDGET_MAX],
        "source": source,
        "degraded_mode": degraded,
        "multi_category": multi_category_available,
        "warning": warning,
        "public_error": public_error,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _FLOW_CACHE[cache_key] = {"_ts": now, "data": data}
    flow_ttl = _FLOW_DEGRADED_TTL if degraded else _FLOW_CACHE_TTL
    _shared_cache_set(shared_key, data, ttl_seconds=flow_ttl)
    # region agent log
    _debug_log(
        "H2",
        "app/api.py:_market_flow_payload",
        "Market flow payload built",
        {
            "limit": limit,
            "category": category,
            "force_refresh": force_refresh,
            "effective_budget": effective_budget,
            "vyk_items": len(vyk_items),
            "local_items": len(local_items),
            "merged_items": len(merged),
            "source": source,
            "degraded_mode": degraded,
            "multi_category": multi_category_available,
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    return {**data, "items": data["items"][:limit]}


@app.get("/market/flow")
def market_flow(
    limit: int = Query(40, ge=1, le=500),
    category: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    return _market_flow_payload(limit=limit, category=category, force_refresh=refresh)


@app.get("/kap/companies")
def kap_companies() -> Dict[str, Any]:
    from app.kap_service import get_kap_companies

    now_ts = time.time()
    cached = _KAP_COMPANIES_RESPONSE_CACHE.get("data")
    if cached and now_ts - float(_KAP_COMPANIES_RESPONSE_CACHE.get("_ts", 0)) < _KAP_COMPANIES_RESPONSE_CACHE_TTL:
        return cached

    shared_cached = _shared_cache_get_dict(_KAP_COMPANIES_RESPONSE_CACHE_KEY)
    if shared_cached and isinstance(shared_cached.get("companies"), list):
        _KAP_COMPANIES_RESPONSE_CACHE["_ts"] = now_ts
        _KAP_COMPANIES_RESPONSE_CACHE["data"] = shared_cached
        return shared_cached

    companies = get_kap_companies()
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    instrument_names = get_instrument_names(CONFIG.paths.processed_dir, "stock")
    items: List[Dict[str, Any]] = []
    for symbol in companies:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            continue
        cached_meta = _load_cached_kap_market_metadata(cache_dir, normalized)
        title = str(instrument_names.get(normalized) or cached_meta.get("company_title") or "").strip()
        company_code = str(cached_meta.get("company") or normalized).strip().upper()
        aliases = [normalized]
        if company_code and company_code != normalized:
            aliases.append(company_code)
        if title:
            aliases.append(title)
        items.append(
            {
                "symbol": normalized,
                "title": title or None,
                "aliases": aliases,
                "latest_quarter": cached_meta.get("latest_quarter"),
                "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
            }
        )
    payload = {"companies": companies, "items": items}
    _KAP_COMPANIES_RESPONSE_CACHE["_ts"] = now_ts
    _KAP_COMPANIES_RESPONSE_CACHE["data"] = payload
    _shared_cache_set(
        _KAP_COMPANIES_RESPONSE_CACHE_KEY,
        payload,
        ttl_seconds=_KAP_COMPANIES_RESPONSE_CACHE_TTL,
    )
    return payload


def _kap_snapshot_response_cache_key(company: str, max_quarters: int) -> str:
    from app.kap_service import normalize_kap_symbol
    from src.kap_fetcher import KAP_CACHE_SCHEMA_VERSION

    normalized = normalize_kap_symbol(str(company or "").strip().upper().replace(".", ""))
    return f"api:kap-snapshot:{normalized}:quarters={max_quarters}:schema={KAP_CACHE_SCHEMA_VERSION}"


def _annotate_kap_response_cache(
    payload: Dict[str, Any],
    *,
    cache_hit: bool,
    ttl_seconds: int = _KAP_SNAPSHOT_RESPONSE_CACHE_TTL,
) -> Dict[str, Any]:
    out = dict(payload)
    status = _cache_status()
    out["response_cache_hit"] = cache_hit
    out["response_cache_backend"] = status.get("cache_backend")
    out["response_cache_ttl_seconds"] = ttl_seconds
    return out


def _build_kap_snapshot_response(company: str, *, refresh: bool, max_quarters: int) -> Dict[str, Any]:
    from app.kap_service import get_kap_snapshot, normalize_snapshot_for_frontend

    raw = get_kap_snapshot(
        company=company,
        cfg=CONFIG.kap,
        processed_dir=CONFIG.paths.processed_dir,
        force_refresh=refresh,
        max_quarters=max_quarters,
        use_cache_when_complete=not refresh,
    )
    _upsert_stock_reference_from_kap_payload(company, raw, source="kap")
    normalized = normalize_snapshot_for_frontend(raw)

    price_payload = _fetch_kap_price_payload(normalized.get("stock_code") or company)
    isyatirim_payload = _fetch_isyatirim_multiples(normalized.get("stock_code") or company)
    normalized["valuation"] = _build_kap_valuation_payload(
        snapshot=normalized,
        price_payload=price_payload,
        isyatirim_payload=isyatirim_payload,
    )
    return normalized


@app.get("/kap/snapshot")
def kap_snapshot(
    company: str = Query(..., min_length=1),
    refresh: bool = Query(False),
    max_quarters: int = Query(10, ge=1, le=20),
) -> Dict[str, Any]:
    if not getattr(CONFIG, "kap", None) or not getattr(CONFIG.kap, "enabled", False):
        raise HTTPException(status_code=503, detail="KAP modülü devre dışı.")
    cache_key = _kap_snapshot_response_cache_key(company, max_quarters)
    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key=cache_key,
        # Snapshot cache expiry is the only time a public caller may trigger a
        # provider refresh.  The caller itself receives fresh/stale cached data
        # immediately, never an upstream cache-bypass request.
        factory=lambda: _build_kap_snapshot_response(company, refresh=True, max_quarters=max_quarters),
        fresh_ttl_seconds=_KAP_SNAPSHOT_RESPONSE_CACHE_TTL,
        stale_ttl_seconds=_KAP_SNAPSHOT_RESPONSE_STALE_TTL,
        force_revalidate=refresh,
    )
    if payload is None:
        raise HTTPException(status_code=503, detail="KAP snapshot yenileniyor. Lütfen kısa süre sonra tekrar deneyin.")
    response = _annotate_kap_response_cache(
        payload,
        cache_hit=cache_status in {"local_hit", "shared_hit", "coalesced", "stale"},
    )
    response["response_cache_status"] = cache_status
    response["response_cache_stale"] = stale
    response["refresh_pending"] = refresh_pending
    return response


def _quarter_sort_key(quarter: Dict[str, Any]) -> tuple[int, int]:
    return int(quarter.get("year") or 0), int(quarter.get("period") or 0)


def _extract_quarter_metric(
    quarter: Dict[str, Any],
    metric_key: str,
    priority: List[str],
) -> Optional[float]:
    for field in priority:
        container = quarter.get(field)
        if not isinstance(container, dict):
            continue
        metric = container.get(metric_key)
        if isinstance(metric, dict):
            value = metric.get("value")
        else:
            value = metric
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _build_ttm_sum(quarters_asc: List[Dict[str, Any]], metric_key: str) -> Optional[float]:
    if not quarters_asc:
        return None
    tail = quarters_asc[-4:]
    required = min(4, len(quarters_asc))
    values: List[float] = []
    for quarter in tail:
        value = _extract_quarter_metric(
            quarter,
            metric_key,
            priority=["metrics_quarterly", "metrics"],
        )
        if value is None:
            continue
        values.append(value)
    if len(values) != required:
        return None
    return float(sum(values))


def _build_kap_valuation_payload(
    *,
    snapshot: Dict[str, Any],
    price_payload: Dict[str, Any],
    isyatirim_payload: Dict[str, Any],
) -> Dict[str, Any]:
    quarters_raw = snapshot.get("quarters")
    quarters = [q for q in quarters_raw if isinstance(q, dict)] if isinstance(quarters_raw, list) else []
    quarters_sorted = sorted(quarters, key=_quarter_sort_key)
    latest = quarters_sorted[-1] if quarters_sorted else None

    ttm_net_kar = _build_ttm_sum(quarters_sorted, "net_kar")
    ttm_favok = _build_ttm_sum(quarters_sorted, "favok")

    ozkaynaklar = (
        _extract_quarter_metric(latest, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
        if latest
        else None
    )
    net_borc = (
        _extract_quarter_metric(latest, "net_borc", priority=["metrics", "metrics_ytd"])
        if latest
        else None
    )

    shares_outstanding = None
    share_source = None
    if latest:
        shares_outstanding = _extract_quarter_metric(
            latest,
            "odenmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        )
        if shares_outstanding is not None:
            share_source = "odenmis_sermaye"
        else:
            shares_outstanding = _extract_quarter_metric(
                latest,
                "cikarilmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            )
            if shares_outstanding is not None:
                share_source = "cikarilmis_sermaye"
    if shares_outstanding is not None and shares_outstanding <= 0:
        shares_outstanding = None
        share_source = None

    assumptions: List[str] = []
    share_nominal_value = None
    if shares_outstanding is not None:
        share_nominal_value = 1.0
        assumptions.append("Hisse adedi, nominal pay degeri 1 TL varsayimiyla hesaplandi.")

    price_ok = bool(price_payload.get("ok"))
    price = float(price_payload["price"]) if price_ok and isinstance(price_payload.get("price"), (int, float)) else None
    price_currency = str(price_payload.get("currency", "TRY")) if price_ok else None
    price_as_of = price_payload.get("as_of") if price_ok else None

    market_cap = price * shares_outstanding if price is not None and shares_outstanding is not None else None
    enterprise_value = market_cap + net_borc if market_cap is not None and net_borc is not None else None

    fk = _parse_tr_decimal(isyatirim_payload.get("fk")) if isyatirim_payload.get("ok") else None
    pd_dd = _parse_tr_decimal(isyatirim_payload.get("pd_dd")) if isyatirim_payload.get("ok") else None
    fd_favok = _parse_tr_decimal(isyatirim_payload.get("fd_favok")) if isyatirim_payload.get("ok") else None

    return {
        "price": price,
        "price_currency": price_currency,
        "price_as_of": price_as_of,
        "price_source": "yahoo_finance_chart",
        "shares_outstanding": shares_outstanding,
        "share_source": share_source,
        "share_nominal_value": share_nominal_value,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "ttm_net_kar": ttm_net_kar,
        "ttm_favok": ttm_favok,
        "fk": fk,
        "pd_dd": pd_dd,
        "fd_favok": fd_favok,
        "fk_prim_iskonto_pct": _parse_tr_decimal(isyatirim_payload.get("fk_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "fd_favok_prim_iskonto_pct": _parse_tr_decimal(isyatirim_payload.get("fd_favok_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "pd_dd_prim_iskonto_pct": _parse_tr_decimal(isyatirim_payload.get("pd_dd_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "multiples_source": isyatirim_payload.get("source") if isyatirim_payload.get("ok") else None,
        "multiples_note": isyatirim_payload.get("note") if isyatirim_payload.get("ok") else None,
        "multiples_as_of": isyatirim_payload.get("fetched_at") if isyatirim_payload.get("ok") else None,
        "multiples_error": isyatirim_payload.get("error") if not isyatirim_payload.get("ok") else None,
        "assumptions": assumptions,
    }


# ── Yahoo Finance price endpoint ──────────────────────────
_PRICE_CACHE: Dict[str, Any] = {}
_PRICE_CACHE_TTL = 300  # 5 minutes
_MARKET_PRICE_CACHE: Dict[str, Any] = {}
_MARKET_PRICE_CACHE_TTL = 3  # seconds; used by the live stocks table
_INFOYATIRIM_STOCK_PAGE_CACHE: Dict[str, Any] = {}
_INFOYATIRIM_STOCK_PAGE_CACHE_TTL = 60
_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT = 12
_STOCKS_CACHE: Dict[str, Any] = {}
_STOCKS_CACHE_TTL = 3
_MARKET_STOCK_CARD_CHART_CACHE: Dict[str, Any] = {}
_MARKET_STOCK_CARD_CHART_CACHE_TTL = 45
_MARKET_STOCK_CARD_LIMIT = 12
_MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL_SECONDS", "5")
)
_MARKET_STOCK_CARD_PREVIOUS_SESSION_LOOKBACK_DAYS = 10
_STOCK_CARD_VALUATION_CACHE: Dict[str, Any] = {}
_STOCK_CARD_VALUATION_CACHE_TTL = int(os.getenv("RAGFIN_STOCK_CARD_VALUATION_CACHE_TTL_SECONDS", str(6 * 60 * 60)))
_TURKEY_TIMEZONE = timezone(timedelta(hours=3))
# Borsa İstanbul Pay Piyasası tam iş günlerinde 09:40-18:10 arasındadır.
# 09:55 açılış fiyatının belirlendiği, 18:05 ise kapanış emir toplama
# aşamasının bittiği zamandır; bu saatler piyasanın genel açık/kapalı
# sınırları olarak kullanılmamalıdır.
_BIST_EQUITY_SESSION_OPEN_MINUTE = (9 * 60) + 40
_BIST_EQUITY_SESSION_CLOSE_MINUTE = (18 * 60) + 10
_MARKET_STOCK_CARD_CHART_RANGES: Dict[str, Dict[str, Any]] = {
    "1d": {"interval": "5m", "range": "1d", "ttl": 30},
    "1w": {"interval": "15m", "range": "5d", "ttl": 60},
    "1m": {"interval": "4h", "range": "1mo", "ttl": 600},
    "1y": {"interval": "1d", "range": "1y", "ttl": 3600},
}
_STOCK_RETURN_BASE_CACHE: Dict[str, Any] = {}
_STOCK_RETURN_BASE_CACHE_TTL = 900  # 15 minutes
_ISYATIRIM_CACHE: Dict[str, Any] = {}
_ISYATIRIM_CACHE_TTL = 900  # 15 minutes
_ISYATIRIM_BASIC_SUMMARY_CACHE: Dict[str, Any] = {}
_ISYATIRIM_BASIC_SUMMARY_CACHE_TTL = 60
_MARKET_STOCK_INDEX_ORDER = ["XUTUM", "XU100", "XU030"]
_MARKET_SECTOR_INDEX_ORDER = [
    "XUSIN",
    "XUHIZ",
    "XUMAL",
    "XUTEK",
    "XBANK",
    "XAKUR",
    "XBLSM",
    "XELKT",
    "XFINK",
    "XGMYO",
    "XGIDA",
    "XHOLD",
    "XILTM",
    "XINSA",
    "XKAGT",
    "XKMYA",
    "XMADN",
    "XMANA",
    "XMESY",
    "XSGRT",
    "XSPOR",
    "XTAST",
    "XTCRT",
    "XTEKS",
    "XTRZM",
    "XULAS",
    "XYORT",
]
_MARKET_INDEX_ORDER = _MARKET_STOCK_INDEX_ORDER + _MARKET_SECTOR_INDEX_ORDER
_MARKET_STOCK_INDEXES = set(_MARKET_STOCK_INDEX_ORDER)
_MARKET_INDICES_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_DETAIL_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_QUOTE_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_INTRADAY_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_RETURN_CACHE: Dict[str, Any] = {}
_MARKET_INDICES_CACHE_TTL = 3
_MARKET_INDEX_DETAIL_CACHE_TTL = 3
_MARKET_INDEX_QUOTE_CACHE_TTL = 3
_MARKET_INDEX_INTRADAY_CACHE_TTL = 3
_MARKET_INDEX_RETURN_CACHE_TTL = 900
_MARKET_INDEX_META: Dict[str, Dict[str, Any]] = {
    "XUTUM": {
        "symbol": "XUTUM",
        "label": "BIST Tüm",
        "yahoo_candidates": ["XUTUM.IS", "^XUTUM", "XUTUM"],
    },
    "XU100": {
        "symbol": "XU100",
        "label": "BIST 100",
        "yahoo_candidates": ["XU100.IS", "^XU100", "XU100"],
    },
    "XU030": {
        "symbol": "XU030",
        "label": "BIST 30",
        "yahoo_candidates": ["XU030.IS", "^XU030", "XU030"],
    },
}
_MARKET_SECTOR_INDEX_LABELS: Dict[str, str] = {
    "XUSIN": "BIST Sınai",
    "XUHIZ": "BIST Hizmetler",
    "XUMAL": "BIST Mali",
    "XUTEK": "BIST Teknoloji",
    "XBANK": "BIST Banka",
    "XAKUR": "BIST Aracı Kurumlar",
    "XBLSM": "BIST Bilişim",
    "XELKT": "BIST Elektrik",
    "XFINK": "BIST Fin. Kir. Faktoring",
    "XGMYO": "BIST Gayrimenkul Y.O.",
    "XGIDA": "BIST Gıda İçecek",
    "XHOLD": "BIST Holding ve Yatırım",
    "XILTM": "BIST İletişim",
    "XINSA": "BIST İnşaat",
    "XKAGT": "BIST Orman Kağıt Basım",
    "XKMYA": "BIST Kimya Petrol Plastik",
    "XMADN": "BIST Madencilik",
    "XMANA": "BIST Metal Ana",
    "XMESY": "BIST Metal Eşya Makina",
    "XSGRT": "BIST Sigorta",
    "XSPOR": "BIST Spor",
    "XTAST": "BIST Taş Toprak",
    "XTCRT": "BIST Ticaret",
    "XTEKS": "BIST Tekstil Deri",
    "XTRZM": "BIST Turizm",
    "XULAS": "BIST Ulaştırma",
    "XYORT": "BIST Menkul Kıym. Y.O.",
}
for _sector_index_code in _MARKET_SECTOR_INDEX_ORDER:
    _MARKET_INDEX_META[_sector_index_code] = {
        "symbol": _sector_index_code,
        "label": _MARKET_SECTOR_INDEX_LABELS[_sector_index_code],
        "yahoo_candidates": [
            f"{_sector_index_code}.IS",
            f"^{_sector_index_code}",
            _sector_index_code,
        ],
    }


def _supported_stock_indexes_text() -> str:
    return ", ".join(_MARKET_STOCK_INDEX_ORDER)


def _supported_market_indexes_text() -> str:
    return ", ".join(_MARKET_INDEX_ORDER)


def _normalize_stock_index(index_name: str) -> str:
    normalized = str(index_name or "XUTUM").strip().upper()
    if normalized not in _MARKET_STOCK_INDEXES:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen endeks. {_supported_stock_indexes_text()} kullanin.",
        )
    return normalized
_RETURN_BASE_FIELDS: List[tuple[str, str]] = [
    ("return_1w_pct", "base_1w"),
    ("return_1m_pct", "base_1m"),
    ("return_3m_pct", "base_3m"),
    ("return_6m_pct", "base_6m"),
    ("return_ytd_pct", "base_ytd"),
    ("return_1y_pct", "base_1y"),
]
_INDEX_RETURN_BASE_FIELDS: List[tuple[str, str]] = _RETURN_BASE_FIELDS + [
    ("return_5y_pct", "base_5y"),
]


def _isyatirim_company_card_url(symbol: str) -> str:
    return f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"


def _isyatirim_basic_summary_url() -> str:
    return "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx"


def _parse_tr_decimal(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        numeric = float(raw)
        if numeric != numeric:  # NaN guard
            return None
        return numeric

    token = str(raw).strip()
    if not token:
        return None

    token = token.replace("\xa0", "").replace(" ", "")
    token = token.replace("\u2212", "-")
    token = token.replace("%", "").replace("x", "").replace("X", "")
    if token in {"-", "--", "A/D", "N/A", "n/a"}:
        return None

    number_match = re.search(r"[-+]?\d[\d\.,]*", token)
    if not number_match:
        return None
    token = number_match.group(0)

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            # Turkish style: 1.234,56
            token = token.replace(".", "").replace(",", ".")
        else:
            # English style: 1,234.56
            token = token.replace(",", "")
    elif "," in token:
        if token.count(",") > 1:
            token = token.replace(",", "")
        else:
            left, right = token.split(",", 1)
            if len(right) == 3 and left.lstrip("+-").isdigit():
                token = token.replace(",", "")
            else:
                token = token.replace(",", ".")
    elif "." in token:
        if token.count(".") > 1:
            token = token.replace(".", "")
        else:
            left, right = token.split(".", 1)
            if len(right) == 3 and len(left.lstrip("+-")) > 3:
                token = token.replace(".", "")

    try:
        numeric = float(token)
        if numeric != numeric:  # NaN guard
            return None
        return numeric
    except Exception:
        return None


def _quote_ts_to_iso(raw: Any) -> Optional[str]:
    if not isinstance(raw, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _infoyatirim_stock_page_url(symbol: str) -> str:
    ticker = str(symbol or "").strip().lower()
    return f"https://infoyatirim.com/borsa/{ticker}-hisse"


def _infoyatirim_stock_page_text(html_text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", str(html_text or ""), flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text, flags=re.IGNORECASE))
    text = " ".join(text.split())
    return (
        text.upper()
        .replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )


def _extract_infoyatirim_stock_page_quote(symbol: str, html_text: str) -> Dict[str, Any]:
    ticker = str(symbol or "").strip().upper()
    if not ticker or not html_text:
        return {}

    text = _infoyatirim_stock_page_text(html_text)

    def value_after(label: str) -> Optional[float]:
        match = re.search(rf"{label}\s+([-+]?\d[\d\.,]*\s*(?:%|₺|TL)?)", text, flags=re.IGNORECASE)
        return _parse_tr_decimal(match.group(1)) if match else None

    price = value_after(r"SON ISLEM FIYATI")
    change_pct = value_after(r"GUNLUK DEGISIM\s+%")
    change = value_after(r"GUNLUK DEGISIM\s+\(TL\)")
    volume = value_after(r"GUNLUK HACIM\s+\(TL\)")
    if volume is None:
        volume = value_after(r"TOPLAM ISLEM HACMI")
    market_cap = value_after(r"PIYASA DEGERI")
    fk = value_after(r"F/K")
    pd_dd = value_after(r"PD/DD")
    fd_favok = value_after(r"FD/FAVOK")

    if price is None and change_pct is None and volume is None and market_cap is None and fk is None and pd_dd is None and fd_favok is None:
        return {}

    return {
        "price": price,
        "currency": "TRY",
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "market_cap": market_cap,
        "fk": fk,
        "pd_dd": pd_dd,
        "fd_favok": fd_favok,
        "market_state": "",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_infoyatirim_stock_page_quote(symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {}

    now = time.time()
    cached = _INFOYATIRIM_STOCK_PAGE_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < _INFOYATIRIM_STOCK_PAGE_CACHE_TTL:
        return dict(cached.get("data") or {})
    shared_key = f"api:market:infoyatirim-page-quote:{ticker}:v1"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        _INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached)

    request = urllib.request.Request(
        _infoyatirim_stock_page_url(ticker),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            html_text = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, Exception):
        _INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": {}}
        _shared_cache_set(shared_key, {}, ttl_seconds=_INFOYATIRIM_STOCK_PAGE_CACHE_TTL)
        return {}

    data = _extract_infoyatirim_stock_page_quote(ticker, html_text)
    _INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": data}
    _shared_cache_set(shared_key, data, ttl_seconds=_INFOYATIRIM_STOCK_PAGE_CACHE_TTL)
    return dict(data)


def _market_price_row_needs_fallback(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return True
    return row.get("price") is None or row.get("change_pct") is None or row.get("volume") is None


def _merge_market_price_fallback(base: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not fallback:
        return base
    merged = dict(base or {})
    for key in ("price", "currency", "change", "change_pct", "volume", "market_cap", "market_state", "as_of"):
        if key not in merged or merged.get(key) is None or (key in {"currency", "market_state"} and merged.get(key) == ""):
            value = fallback.get(key)
            if value is not None:
                merged[key] = value
    return merged


def _market_price_source_url(index_name: str) -> str:
    normalized = str(index_name or "XUTUM").strip().upper()
    if normalized == "XUTUM":
        return "https://infoyatirim.com/canli-borsa"
    if normalized == "XU030":
        return "https://infoyatirim.com/canli-borsa/xu100-bist-100-hisseleri"
    return "https://infoyatirim.com/canli-borsa/xu100-bist-100-hisseleri"


def _fetch_market_price_map(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
    import urllib.error
    import urllib.request

    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not normalized_symbols:
        return {}

    normalized_index = str(index_name or "XU100").strip().upper()
    cache_key = f"{normalized_index}:{','.join(normalized_symbols)}"
    now = time.time()
    cached = _MARKET_PRICE_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _MARKET_PRICE_CACHE_TTL:
        return cached.get("items", {})

    url = _market_price_source_url(normalized_index)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, Exception):
        return {}

    items: Dict[str, Dict[str, Any]] = {}
    try:
        row_pattern = re.compile(
            r'<tr[^>]+data-symbol="(?P<symbol>[A-Z0-9]+)"[^>]*>(?P<body>.*?)</tr>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        price_pattern = re.compile(r'<td[^>]+class="price"[^>]+data-val="(?P<value>[^"]+)"', re.IGNORECASE)
        change_pattern = re.compile(r'<td[^>]+class="change"[^>]+data-val="(?P<value>[^"]+)"', re.IGNORECASE)
        percent_pattern = re.compile(r'<td[^>]+class="percent"[^>]+data-val="(?P<value>[^"]+)"', re.IGNORECASE)
        fetched_at = datetime.now(timezone.utc).isoformat()
        for match in row_pattern.finditer(html_text):
            symbol = str(match.group("symbol") or "").strip().upper()
            if not symbol:
                continue
            body = match.group("body") or ""
            price_match = price_pattern.search(body)
            change_match = change_pattern.search(body)
            percent_match = percent_pattern.search(body)

            volume = None
            cells = re.findall(r"<td\b[^>]*>(.*?)</td>", body, flags=re.IGNORECASE | re.DOTALL)
            if len(cells) > 4:
                volume_raw = html.unescape(re.sub(r"<[^>]+>", " ", cells[4], flags=re.IGNORECASE))
                volume = _parse_tr_decimal(volume_raw)

            items[symbol] = {
                "price": _parse_tr_decimal(price_match.group("value") if price_match else None),
                "currency": "TRY",
                "change": _parse_tr_decimal(change_match.group("value") if change_match else None),
                "change_pct": _parse_tr_decimal(percent_match.group("value") if percent_match else None),
                "volume": volume,
                "market_state": "",
                "as_of": fetched_at,
            }
    except Exception:
        return {}

    missing_symbols = [
        symbol
        for symbol in normalized_symbols
        if _market_price_row_needs_fallback(items.get(symbol))
    ]
    if items and missing_symbols and normalized_index != "XUTUM":
        fallback_symbols = missing_symbols[:_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT]
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(6, len(fallback_symbols))) as pool:
                fallback_rows = list(pool.map(_fetch_infoyatirim_stock_page_quote, fallback_symbols))
        except Exception:
            fallback_rows = [_fetch_infoyatirim_stock_page_quote(symbol) for symbol in fallback_symbols]
        for symbol, fallback in zip(fallback_symbols, fallback_rows):
            if fallback:
                items[symbol] = _merge_market_price_fallback(items.get(symbol, {}), fallback)

    _MARKET_PRICE_CACHE[cache_key] = {"_ts": now, "items": items}
    return items


def _pick_series_value_at_or_before(
    points: List[tuple[datetime, float]],
    target: datetime,
) -> Optional[float]:
    candidate = None
    for point_dt, close in points:
        if point_dt <= target:
            candidate = close
        elif candidate is not None:
            break
    return candidate


def _pick_series_value_at_or_after(
    points: List[tuple[datetime, float]],
    target: datetime,
) -> Optional[float]:
    for point_dt, close in points:
        if point_dt >= target:
            return close
    return None


def _fetch_stock_return_bases(symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {}

    now = time.time()
    cached = _STOCK_RETURN_BASE_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < _STOCK_RETURN_BASE_CACHE_TTL:
        return dict(cached.get("data") or {})
    shared_cached = _shared_cache_get_dict(f"api:stock-return-bases:{ticker}")
    if shared_cached:
        _STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached)

    yahoo_symbol = f"{ticker}.IS"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        "?interval=1d&range=1y"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception):
        return {}

    try:
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
    except (KeyError, IndexError, TypeError):
        return {}

    series: List[Dict[str, Any]] = []
    for ts, close, high, low in zip(timestamps, closes, highs, lows):
        if close is None or not isinstance(ts, (int, float)):
            continue
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            series.append({
                "dt": dt,
                "close": float(close),
                "high": float(high) if high is not None else float(close),
                "low": float(low) if low is not None else float(close),
            })
        except (TypeError, ValueError):
            continue

    if not series:
        return {}

    series.sort(key=lambda x: x["dt"])
    latest = series[-1]
    latest_dt = latest["dt"]
    latest_close = latest["close"]
    year_start = datetime(latest_dt.year, 1, 1, tzinfo=timezone.utc)

    def _get_base_price(target_dt: datetime) -> Optional[float]:
        points_for_base = [(s["dt"], s["close"]) for s in series]
        return _pick_series_value_at_or_before(points_for_base, target_dt)

    def _get_range_stats(start_dt: datetime):
        relevant = [s for s in series if s["dt"] >= start_dt]
        if not relevant:
            return None, None, None
        
        base_val = _get_base_price(start_dt)
        high_val = max(s["high"] for s in relevant)
        low_val = min(s["low"] for s in relevant)
        return base_val, high_val, low_val

    b1w, h1w, l1w = _get_range_stats(latest_dt - timedelta(days=7))
    b1m, h1m, l1m = _get_range_stats(latest_dt - timedelta(days=30))
    b3m, h3m, l3m = _get_range_stats(latest_dt - timedelta(days=91))
    b6m, h6m, l6m = _get_range_stats(latest_dt - timedelta(days=182))
    bytd, hytd, lytd = _get_range_stats(year_start)
    b1y, h1y, l1y = _get_range_stats(latest_dt - timedelta(days=365))

    bases = {
        "base_1w": b1w, "high_1w": h1w, "low_1w": l1w,
        "base_1m": b1m, "high_1m": h1m, "low_1m": l1m,
        "base_3m": b3m, "high_3m": h3m, "low_3m": l3m,
        "base_6m": b6m, "high_6m": h6m, "low_6m": l6m,
        "base_ytd": bytd, "high_ytd": hytd, "low_ytd": lytd,
        "base_1y": b1y, "high_1y": h1y, "low_1y": l1y,
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
    }
    _STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": now, "data": bases}
    _shared_cache_set(f"api:stock-return-bases:{ticker}", bases, _STOCK_RETURN_BASE_CACHE_TTL)
    return dict(bases)


def _fetch_stock_return_bases_bulk(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    normalized_symbols = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not normalized_symbols:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    stale: List[str] = []
    now = time.time()
    for symbol in normalized_symbols:
        cached = _STOCK_RETURN_BASE_CACHE.get(symbol)
        if cached and now - cached.get("_ts", 0) < _STOCK_RETURN_BASE_CACHE_TTL:
            result[symbol] = dict(cached.get("data") or {})
            continue
        shared_cached = _shared_cache_get_dict(f"api:stock-return-bases:{symbol}")
        if shared_cached:
            _STOCK_RETURN_BASE_CACHE[symbol] = {"_ts": now, "data": shared_cached}
            result[symbol] = dict(shared_cached)
            continue
        stale.append(symbol)

    if not stale:
        return result

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for symbol, bases in zip(stale, pool.map(_fetch_stock_return_bases, stale)):
                result[symbol] = bases
    except Exception:
        for symbol in stale:
            result[symbol] = _fetch_stock_return_bases(symbol)
    return result


def _return_pct(current_price: Any, base_price: Any) -> Optional[float]:
    try:
        price = float(current_price)
        base = float(base_price)
    except (TypeError, ValueError):
        return None
    if price <= 0 or base <= 0:
        return None
    return round(((price - base) / base) * 100, 2)


def _returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Any]:
    res = {}
    for response_field, base_field in _RETURN_BASE_FIELDS:
        base_val = return_bases.get(base_field)
        res[response_field] = _return_pct(current_price, base_val)
        
        # Extract period (e.g. 1w from return_1w_pct)
        period = response_field.replace("return_", "").replace("_pct", "")
        res[f"base_{period}"] = base_val
        res[f"high_{period}"] = return_bases.get(f"high_{period}")
        res[f"low_{period}"] = return_bases.get(f"low_{period}")
    return res





def _market_stock_benchmarks() -> Dict[str, Dict[str, Any]]:
    base_map = _fetch_stock_return_bases_bulk(_MARKET_STOCK_INDEX_ORDER)
    benchmarks: Dict[str, Dict[str, Any]] = {}
    for index_name in _MARKET_STOCK_INDEX_ORDER:
        bases = base_map.get(index_name, {})
        current_for_returns = bases.get("latest_close")
        benchmarks[index_name] = {
            **_returns_from_bases(current_for_returns, bases),
            "as_of": bases.get("as_of"),
        }
    return benchmarks


def _market_stock_symbols_for_index(index_name: str) -> List[str]:
    from app.kap_service import get_bist100_companies, get_bist30_companies, get_bist_all_companies

    normalized = _normalize_stock_index(index_name)
    if normalized == "XUTUM":
        return get_bist_all_companies()
    if normalized == "XU100":
        return get_bist100_companies()
    return get_bist30_companies()


def _cached_stock_return_bases_bulk(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    now = time.time()
    result: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        cached = _STOCK_RETURN_BASE_CACHE.get(normalized)
        if cached and now - cached.get("_ts", 0) < _STOCK_RETURN_BASE_CACHE_TTL:
            result[normalized] = dict(cached.get("data") or {})
            continue
        shared_cached = _shared_cache_get_dict(f"api:stock-return-bases:{normalized}")
        if shared_cached:
            _STOCK_RETURN_BASE_CACHE[normalized] = {"_ts": now, "data": shared_cached}
            result[normalized] = dict(shared_cached)
    return result


def _market_stock_row(
    symbol: str,
    *,
    cached_meta: Dict[str, Any],
    quote: Dict[str, Any],
    return_bases: Dict[str, Any],
    basic_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    latest_quarter = cached_meta.get("latest_quarter")
    current_for_returns = quote.get("price") if quote.get("price") is not None else return_bases.get("latest_close")
    return {
        "company": symbol,
        "latest_quarter": latest_quarter,
        "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
        "price": quote.get("price"),
        "price_currency": quote.get("currency"),
        "change": quote.get("change"),
        "change_pct": quote.get("change_pct"),
        "price_as_of": quote.get("as_of"),
        "volume": quote.get("volume"),
        "market_cap": _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary),
        **_empty_logo_payload(),
        **_returns_from_bases(current_for_returns, return_bases),
    }


def _build_market_stocks_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    from app.kap_service import get_bist_index_universe

    normalized_index = _normalize_stock_index(index_name)
    symbols = _market_stock_symbols_for_index(normalized_index)
    try:
        universe = get_bist_index_universe(normalized_index, force_refresh=force_refresh)
    except Exception:
        universe = {
            "index": normalized_index,
            "count": len(symbols),
            "source": None,
            "source_url": None,
            "source_date": None,
            "fetched_at": None,
            "cache_hit": False,
            "fallback_used": False,
        }
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    price_map = _fetch_market_price_map(symbols, index_name=normalized_index)
    return_base_map = (
        _cached_stock_return_bases_bulk(symbols)
        if normalized_index == "XUTUM"
        else _fetch_stock_return_bases_bulk(symbols)
    )
    basic_summary_map = _fetch_isyatirim_basic_summary_map()
    rows = [
        _market_stock_row(
            symbol,
            cached_meta=_load_cached_kap_market_metadata(cache_dir, symbol),
            quote=price_map.get(symbol, {}),
            return_bases=return_base_map.get(symbol, {}),
            basic_summary=basic_summary_map.get(symbol),
        )
        for symbol in symbols
    ]
    data = {
        "index": normalized_index,
        "rows": rows,
        "benchmarks": _market_stock_benchmarks(),
        "source": "infoyatirim_yahoo",
        "universe": {
            "index": universe.get("index") or normalized_index,
            "count": int(universe.get("count") or len(symbols)),
            "source": universe.get("source"),
            "source_url": universe.get("source_url"),
            "source_date": universe.get("source_date"),
            "fetched_at": universe.get("fetched_at"),
            "cache_hit": bool(universe.get("cache_hit")),
            "fallback_used": bool(universe.get("fallback_used")),
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    return data


def _market_quote_entry_is_fresh(entry: Any) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return False
    try:
        return float(entry.get("fresh_until") or 0.0) > time.time()
    except (TypeError, ValueError):
        return False


def _market_quote_entry_is_stale(entry: Any) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return False
    try:
        return float(entry.get("stale_until") or 0.0) > time.time()
    except (TypeError, ValueError):
        return False


def _market_quote_entry_payload(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return None
    return dict(entry["payload"])


def _market_unavailable_stocks_payload(index_name: str) -> Dict[str, Any]:
    normalized_index = _normalize_stock_index(index_name)
    universe = _market_universe_payload(index_name=normalized_index)
    universe_info = universe.get("universe") if isinstance(universe.get("universe"), dict) else {}
    rows = []
    for row in universe.get("rows") if isinstance(universe.get("rows"), list) else []:
        rows.append(
            {
                **dict(row),
                "volume": None,
                "return_1w_pct": None,
                "return_1m_pct": None,
                "return_3m_pct": None,
                "return_6m_pct": None,
                "return_ytd_pct": None,
                "return_1y_pct": None,
            }
        )
    return {
        "index": normalized_index,
        "rows": rows,
        "benchmarks": _market_stock_benchmarks(),
        "source": "reference_data",
        "universe": dict(universe_info),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _market_stocks_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    normalized_index = _normalize_stock_index(index_name)
    started_at = time.perf_counter()
    local_key = f"payload:{normalized_index}"
    shared_key = f"api:market:stocks:{normalized_index}:v2"

    def revalidate_factory() -> Optional[Dict[str, Any]]:
        built = _build_market_stocks_payload(index_name=normalized_index, force_refresh=True)
        rows = built.get("rows") if isinstance(built.get("rows"), list) else []
        has_quote = not rows or any(row.get("price") is not None for row in rows if isinstance(row, dict))
        return built if has_quote else None

    refresh_pending = False
    if force_refresh:
        refresh_pending = _schedule_swr_revalidation(
            cache_key=shared_key,
            fresh_ttl_seconds=_MARKET_QUOTES_FRESH_TTL,
            stale_ttl_seconds=_MARKET_QUOTES_STALE_TTL,
            factory=revalidate_factory,
        )
        # A public refresh is a revalidation signal, never a cache bypass.
        force_refresh = False
    local_entry = _STOCKS_CACHE.get(local_key, {}).get("entry")

    if not force_refresh and _market_quote_entry_is_fresh(local_entry):
        payload = _market_quote_entry_payload(local_entry) or {}
        payload.update({
            "cache_status": "hit",
            "quote_status": "fresh",
            "stale": False,
            "quote_error": None,
            "refresh_pending": refresh_pending,
        })
        _log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status="hit",
            upstream_called=False,
            stale=False,
            started_at=started_at,
        )
        return payload

    shared_entry = _shared_cache_get_dict(shared_key)
    stale_entry = local_entry if _market_quote_entry_is_stale(local_entry) else None
    if _market_quote_entry_is_stale(shared_entry):
        stale_entry = shared_entry
    if not force_refresh and _market_quote_entry_is_fresh(shared_entry):
        _STOCKS_CACHE[local_key] = {"_ts": time.time(), "entry": shared_entry}
        payload = _market_quote_entry_payload(shared_entry) or {}
        payload.update({
            "cache_status": "shared_hit",
            "quote_status": "fresh",
            "stale": False,
            "quote_error": None,
            "refresh_pending": refresh_pending,
        })
        _log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status="shared_hit",
            upstream_called=False,
            stale=False,
            started_at=started_at,
        )
        return payload

    latest_build: Dict[str, Any] = {}

    def build_and_envelope() -> Optional[Dict[str, Any]]:
        nonlocal latest_build
        built = _build_market_stocks_payload(index_name=normalized_index, force_refresh=force_refresh)
        latest_build = dict(built)
        rows = built.get("rows") if isinstance(built.get("rows"), list) else []
        has_quote = not rows or any(row.get("price") is not None for row in rows if isinstance(row, dict))
        if not has_quote:
            return None
        now = time.time()
        return {
            "payload": built,
            "fresh_until": now + max(1, _MARKET_QUOTES_FRESH_TTL),
            "stale_until": now + max(1, _MARKET_QUOTES_STALE_TTL),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

    entry, cache_status = _get_or_set_single_flight(
        shared_key,
        ttl_seconds=max(1, _MARKET_QUOTES_STALE_TTL),
        factory=build_and_envelope,
        lock_key=f"api:market:stocks:{normalized_index}:lock:v2",
        lock_ttl_seconds=_MARKET_QUOTES_LOCK_TTL_SECONDS,
        wait_timeout_seconds=_MARKET_QUOTES_WAIT_TIMEOUT_SECONDS,
        poll_interval_seconds=_MARKET_QUOTES_POLL_INTERVAL_SECONDS,
        cache_usable=_market_quote_entry_is_fresh,
        allow_cached=not force_refresh,
    )

    if _market_quote_entry_is_fresh(entry):
        _STOCKS_CACHE[local_key] = {"_ts": time.time(), "entry": entry}
        payload = _market_quote_entry_payload(entry) or {}
        payload.update({
            "cache_status": cache_status,
            "quote_status": "fresh",
            "stale": False,
            "quote_error": None,
            "refresh_pending": refresh_pending,
        })
        _log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status=cache_status,
            upstream_called=cache_status == "miss",
            stale=False,
            started_at=started_at,
        )
        return payload

    stale_payload = _market_quote_entry_payload(stale_entry)
    if stale_payload is not None:
        stale_payload.update({
            "cache_status": "stale",
            "quote_status": "stale",
            "stale": True,
            "quote_error": "Piyasa fiyat kaynağına ulaşılamadı.",
            "refresh_pending": refresh_pending or cache_status == "pending",
        })
        _log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status="stale",
            upstream_called=cache_status == "miss",
            stale=True,
            started_at=started_at,
        )
        return stale_payload

    unavailable = latest_build or _market_unavailable_stocks_payload(normalized_index)
    unavailable.update({
        "cache_status": "unavailable",
        "quote_status": "unavailable",
        "stale": False,
        "quote_error": "Piyasa fiyat kaynağına ulaşılamadı.",
        "refresh_pending": refresh_pending or cache_status == "pending",
    })
    _log_market_cache_event(
        endpoint="market/stocks",
        index=normalized_index,
        cache_status="unavailable",
        upstream_called=cache_status == "miss",
        stale=False,
        started_at=started_at,
    )
    return unavailable


def _normalize_market_stock_card_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip()
    item = raw.upper()
    if item.endswith(".IS"):
        item = item[:-3]
    if not item or not re.fullmatch(r"[A-Z0-9]{2,12}", item):
        raise HTTPException(status_code=400, detail=f"Gecersiz hisse kodu: {raw or symbol}")
    return item


def _normalize_market_stock_card_symbols(symbols: str) -> List[str]:
    raw_items = re.split(r"[,\s]+", str(symbols or "").strip())
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not raw.strip():
            continue
        item = _normalize_market_stock_card_symbol(raw)
        if item in seen:
            continue
        normalized.append(item)
        seen.add(item)

    if len(normalized) > _MARKET_STOCK_CARD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"En fazla {_MARKET_STOCK_CARD_LIMIT} hisse karti secilebilir.",
        )
    return normalized


def _normalize_stock_card_chart_range(chart_range: str) -> str:
    normalized = str(chart_range or "1d").strip().lower()
    if normalized not in _MARKET_STOCK_CARD_CHART_RANGES:
        allowed = ", ".join(sorted(_MARKET_STOCK_CARD_CHART_RANGES))
        raise HTTPException(status_code=400, detail=f"Desteklenmeyen grafik araligi. Desteklenenler: {allowed}")
    return normalized


def _stock_card_chart_cache_key(symbol: str, chart_range: str) -> str:
    return f"stock-card-chart:v4:{symbol}:{chart_range}"


def _point_datetime(raw: Any) -> Optional[datetime]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _numeric_chart_value(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def _normalize_stock_card_line_points(raw_points: Any) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_points, list):
        return []

    for point in raw_points:
        if not isinstance(point, dict):
            continue
        point_dt = _point_datetime(point.get("time"))
        close = _numeric_chart_value(point.get("close"))
        if point_dt is None or close is None or close <= 0:
            continue
        time_key = point_dt.isoformat()
        row: Dict[str, Any] = {"time": time_key, "close": close}
        for key in ("open", "high", "low", "volume"):
            value = _numeric_chart_value(point.get(key))
            if value is not None:
                row[key] = value
        deduped[time_key] = row

    return [deduped[key] for key in sorted(deduped)]


def _stock_card_latest_point_dt(points: List[Dict[str, Any]]) -> Optional[datetime]:
    for point in reversed(points or []):
        point_dt = _point_datetime(point.get("time"))
        if point_dt is not None:
            return point_dt
    return None


def _bist_equity_session_status(now_local: datetime) -> str:
    """Return the BIST equity session phase for a Turkey-local timestamp."""
    if now_local.weekday() >= 5:
        return "closed"
    minute_of_day = (now_local.hour * 60) + now_local.minute
    if minute_of_day < _BIST_EQUITY_SESSION_OPEN_MINUTE:
        return "pre"
    if minute_of_day >= _BIST_EQUITY_SESSION_CLOSE_MINUTE:
        return "post"
    return "open"


def _stock_card_session_state(
    points: List[Dict[str, Any]],
    *,
    market_state: Any = None,
    source: Any = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    source_text = str(source or "").strip()
    now_local = (now or datetime.now(_TURKEY_TIMEZONE)).astimezone(_TURKEY_TIMEZONE)
    latest_dt = _stock_card_latest_point_dt(points)
    latest_local = latest_dt.astimezone(_TURKEY_TIMEZONE) if latest_dt else None
    last_trade_at = latest_dt.isoformat() if latest_dt else None
    last_trade_date = latest_local.date().isoformat() if latest_local else None
    previous_session = (
        source_text == "yahoo_previous_session"
        or (latest_local is not None and latest_local.date() < now_local.date())
    )
    bist_session = _bist_equity_session_status(now_local)

    if previous_session:
        status = "previous_session"
        label = "Piyasa kapalı"
        is_live = False
    elif bist_session == "open":
        status = "open"
        label = "Canlı"
        is_live = True
    elif bist_session == "pre":
        status = "pre"
        label = "Açılış öncesi"
        is_live = False
    elif bist_session == "post":
        status = "post"
        label = "Kapanış sonrası"
        is_live = False
    elif latest_local is not None:
        status = "closed"
        label = "Piyasa kapalı"
        is_live = False
    else:
        status = "unknown"
        label = "Veri bekleniyor"
        is_live = False

    return {
        "session_status": status,
        "session_label": label,
        "is_live": is_live,
        "last_trade_at": last_trade_at,
        "last_trade_date": last_trade_date,
        "is_stale": previous_session,
    }


def _fetch_previous_stock_card_intraday_chart(yahoo_symbol: str) -> Dict[str, Any]:
    today = datetime.now(_TURKEY_TIMEZONE).date()
    errors: List[str] = []
    for offset in range(1, _MARKET_STOCK_CARD_PREVIOUS_SESSION_LOOKBACK_DAYS + 1):
        session_date = today - timedelta(days=offset)
        if session_date.weekday() >= 5:
            continue
        chart = _fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="5m",
            start_date=session_date,
            end_date=session_date,
        )
        points = _normalize_stock_card_line_points(chart.get("points") if chart.get("ok") else [])
        if points:
            meta = dict(chart.get("meta") or {})
            meta["fallbackTradingDate"] = session_date.isoformat()
            chart = dict(chart)
            chart["meta"] = meta
            chart["points"] = points
            chart["fallback_trading_date"] = session_date.isoformat()
            return chart
        error = chart.get("error") if isinstance(chart, dict) else None
        if error:
            errors.append(str(error))
    return {
        "ok": False,
        "error": errors[-1] if errors else "previous_session_unavailable",
        "yahoo_symbol": yahoo_symbol,
        "points": [],
    }


def _fetch_stock_card_chart(symbol: str, chart_range: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    ticker = _normalize_market_stock_card_symbol(symbol)
    normalized_range = _normalize_stock_card_chart_range(chart_range)
    config = _MARKET_STOCK_CARD_CHART_RANGES[normalized_range]
    cache_key = _stock_card_chart_cache_key(ticker, normalized_range)
    shared_key = f"api:market:stock-card-chart:{ticker}:range={normalized_range}:v4"

    def build() -> Dict[str, Any]:
        yahoo_symbol = f"{ticker}.IS"
        fetched_at = datetime.now(timezone.utc).isoformat()
        chart = _fetch_yahoo_chart_raw(yahoo_symbol, interval=config["interval"], range_=config["range"])
        points = _normalize_stock_card_line_points(chart.get("points") if chart.get("ok") else [])
        source = "yahoo_live"
        if normalized_range == "1d" and chart.get("ok") and not points:
            fallback_chart = _fetch_previous_stock_card_intraday_chart(yahoo_symbol)
            fallback_points = _normalize_stock_card_line_points(fallback_chart.get("points") if fallback_chart.get("ok") else [])
            if fallback_points:
                chart = fallback_chart
                points = fallback_points
                source = "yahoo_previous_session"
        session_state = _stock_card_session_state(
            points,
            market_state=(chart.get("meta") or {}).get("marketState"),
            source=source,
        )
        return {
            "symbol": ticker,
            "range": normalized_range,
            "yahoo_symbol": yahoo_symbol,
            "line_points": points,
            "source": source,
            "as_of": fetched_at,
            "error": None if chart.get("ok") and points else chart.get("error") or "chart_unavailable",
            "meta": chart.get("meta") or {},
            **session_state,
        }

    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key=shared_key,
        factory=build,
        fresh_ttl_seconds=int(config["ttl"]),
        stale_ttl_seconds=max(_MARKET_SWR_STALE_TTL_SECONDS, int(config["ttl"]) * 2),
        local_cache=_MARKET_STOCK_CARD_CHART_CACHE,
        local_key=cache_key,
        force_revalidate=force_refresh,
    )
    if payload is None:
        return {
            "symbol": ticker,
            "range": normalized_range,
            "yahoo_symbol": f"{ticker}.IS",
            "line_points": [],
            "source": "yahoo_cache",
            "as_of": None,
            "error": "chart_refresh_pending",
            "meta": {},
            "cache_status": cache_status,
            "stale": False,
            "refresh_pending": True,
        }
    result = dict(payload)
    if cache_status in {"local_hit", "shared_hit", "coalesced", "stale"}:
        result["source"] = "yahoo_cache"
    result["cache_status"] = cache_status
    result["stale"] = stale
    result["refresh_pending"] = refresh_pending
    return result


def _market_stock_card_chart_payload(*, symbol: str, chart_range: str, force_refresh: bool = False) -> Dict[str, Any]:
    payload = _fetch_stock_card_chart(symbol, chart_range, force_refresh=force_refresh)
    return {
        "symbol": payload.get("symbol"),
        "range": payload.get("range"),
        "yahoo_symbol": payload.get("yahoo_symbol"),
        "line_points": payload.get("line_points") or [],
        "source": payload.get("source") or "yahoo_live",
        "as_of": payload.get("as_of"),
        "error": payload.get("error"),
        "session_status": payload.get("session_status"),
        "session_label": payload.get("session_label"),
        "is_live": payload.get("is_live"),
        "is_stale": payload.get("is_stale"),
        "last_trade_at": payload.get("last_trade_at"),
        "last_trade_date": payload.get("last_trade_date"),
    }


def _fetch_stock_card_intraday(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    chart_payload = _fetch_stock_card_chart(symbol, "1d", force_refresh=force_refresh)
    ticker = chart_payload.get("symbol") or _normalize_market_stock_card_symbol(symbol)
    yahoo_symbol = chart_payload.get("yahoo_symbol") or f"{ticker}.IS"
    points = chart_payload.get("line_points") or []
    if points:
        highs = [
            point.get("high")
            for point in points
            if isinstance(point.get("high"), (int, float))
        ]
        lows = [
            point.get("low")
            for point in points
            if isinstance(point.get("low"), (int, float))
        ]
        volumes = [
            point.get("volume")
            for point in points
            if isinstance(point.get("volume"), (int, float))
        ]
        meta_payload = chart_payload.get("meta") or {}
        last_close = points[-1].get("close") if points else None
        price = meta_payload.get("regularMarketPrice")
        if price is None and isinstance(last_close, (int, float)):
            price = last_close
        prev_close = meta_payload.get("chartPreviousClose") or meta_payload.get("previousClose")
        change = None
        change_pct = None
        if price is not None and prev_close:
            try:
                change = round(float(price) - float(prev_close), 4)
                change_pct = round((change / float(prev_close)) * 100, 2)
            except (TypeError, ValueError, ZeroDivisionError):
                change = None
                change_pct = None

        last_trade_at = _stock_card_latest_point_dt(points)
        session_state = _stock_card_session_state(
            points,
            market_state=meta_payload.get("marketState"),
            source=chart_payload.get("source"),
        )
        as_of = None
        rmt = meta_payload.get("regularMarketTime")
        if isinstance(rmt, (int, float)):
            try:
                as_of = datetime.fromtimestamp(float(rmt), tz=timezone.utc).isoformat()
            except Exception:
                as_of = None
        if session_state.get("is_stale") or not as_of:
            as_of = session_state.get("last_trade_at") or (last_trade_at.isoformat() if last_trade_at else None)

        payload = {
            "line_points": points,
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "high": meta_payload.get("regularMarketDayHigh") or (max(highs) if highs else None),
            "low": meta_payload.get("regularMarketDayLow") or (min(lows) if lows else None),
            "volume": meta_payload.get("regularMarketVolume") or (sum(volumes) if volumes else None),
            "volume_lot": meta_payload.get("regularMarketVolume") or (sum(volumes) if volumes else None),
            "currency": meta_payload.get("currency") or "TRY",
            "market_state": meta_payload.get("marketState") or "",
            "as_of": as_of or chart_payload.get("as_of"),
            "yahoo_symbol": yahoo_symbol,
            "error": None,
            **session_state,
        }
        return dict(payload)

    fallback = {
        "line_points": [],
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "high": None,
        "low": None,
        "volume": None,
        "volume_lot": None,
        "currency": "TRY",
        "market_state": "",
        "as_of": chart_payload.get("as_of"),
        "yahoo_symbol": yahoo_symbol,
        "error": chart_payload.get("error") or "chart_unavailable",
        **_stock_card_session_state([], market_state="", source=chart_payload.get("source")),
    }
    return dict(fallback)


def _stock_cards_response_cache_key(symbols: List[str]) -> str:
    normalized = ",".join(sorted(symbols))
    return f"api:market:stock-cards:symbols={normalized}:v2"


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_missing_market_ratio(value: Any) -> bool:
    ratio = _parse_tr_decimal(value)
    return ratio is None or abs(ratio) <= 1e-12


def _as_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _round_market_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    top = _as_finite_float(numerator)
    bottom = _as_finite_float(denominator)
    if top is None or bottom is None or abs(bottom) <= 1e-12:
        return None
    return round(top / bottom, 2)


def _resolve_market_card_multiples(symbol: str, multiples_payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    fk = _parse_tr_decimal(multiples_payload.get("fk")) if multiples_payload.get("ok") else None
    pd_dd = _parse_tr_decimal(multiples_payload.get("pd_dd")) if multiples_payload.get("ok") else None
    fd_favok = _parse_tr_decimal(multiples_payload.get("fd_favok")) if multiples_payload.get("ok") else None

    need_fallback = (
        _is_missing_market_ratio(fk)
        or _is_missing_market_ratio(pd_dd)
        or _is_missing_market_ratio(fd_favok)
    )
    if not need_fallback:
        return {"fk": fk, "pd_dd": pd_dd, "fd_favok": fd_favok}

    fallback = _fetch_infoyatirim_stock_page_quote(symbol)
    if _is_missing_market_ratio(fk):
        fallback_fk = _parse_tr_decimal(fallback.get("fk"))
        if fallback_fk is not None and abs(fallback_fk) > 1e-12:
            fk = fallback_fk
    if _is_missing_market_ratio(pd_dd):
        fallback_pd_dd = _parse_tr_decimal(fallback.get("pd_dd"))
        if fallback_pd_dd is not None and abs(fallback_pd_dd) > 1e-12:
            pd_dd = fallback_pd_dd
    if _is_missing_market_ratio(fd_favok):
        fallback_fd_favok = _parse_tr_decimal(fallback.get("fd_favok"))
        if fallback_fd_favok is not None and abs(fallback_fd_favok) > 1e-12:
            fd_favok = fallback_fd_favok

    return {"fk": fk, "pd_dd": pd_dd, "fd_favok": fd_favok}


_STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE: Dict[str, Any] = {}
_STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE_TTL = 6 * 60 * 60


def _stock_card_financial_snapshot_from_cache(symbol: str) -> Dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return {}
    cache_path = CONFIG.paths.processed_dir / "kap_cache" / f"{normalized_symbol}.json"
    if not cache_path.exists():
        return {}
    cache_key = ""
    signature = None
    shared_key = None
    try:
        stat = cache_path.stat()
        cache_key = str(cache_path.resolve())
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = _STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE.get(cache_key)
        if cached and cached.get("signature") == signature:
            return dict(cached.get("data") or {})
        shared_key = f"api:kap:financial-snapshot:{normalized_symbol}:mtime={stat.st_mtime_ns}:size={stat.st_size}:v1"
        shared_cached = _shared_cache_get_dict(shared_key)
        if shared_cached is not None:
            snapshot = dict(shared_cached)
            _STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE[cache_key] = {
                "signature": signature,
                "data": snapshot,
            }
            return snapshot
    except Exception:
        cache_key = ""
        signature = None
        shared_key = None

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    quarters_raw = payload.get("quarters")
    quarters = [q for q in quarters_raw if isinstance(q, dict)] if isinstance(quarters_raw, list) else []
    if not quarters:
        if cache_key and signature:
            _STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE[cache_key] = {
                "signature": signature,
                "data": {},
            }
            if shared_key:
                _shared_cache_set(shared_key, {}, ttl_seconds=_STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE_TTL)
        return {}

    quarters_sorted = sorted(quarters, key=_quarter_sort_key)
    latest = quarters_sorted[-1]
    ttm_net_kar = _build_ttm_sum(quarters_sorted, "net_kar") if len(quarters_sorted) >= 4 else None
    ttm_favok = _build_ttm_sum(quarters_sorted, "favok") if len(quarters_sorted) >= 4 else None
    ozkaynaklar = _extract_quarter_metric(latest, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
    net_borc = _extract_quarter_metric(latest, "net_borc", priority=["metrics", "metrics_ytd"])
    latest_quarter = str(latest.get("quarter") or "").strip().upper() or None
    snapshot = {
        "symbol": str(payload.get("company") or payload.get("stock_code") or normalized_symbol or "").strip().upper(),
        "latest_quarter": latest_quarter,
        "quarter_count": len(quarters_sorted),
        "ttm_net_kar": ttm_net_kar,
        "ttm_favok": ttm_favok,
        "ozkaynaklar": ozkaynaklar,
        "net_borc": net_borc,
        "source": "kap_cache",
        "as_of": payload.get("fetched_at"),
    }
    if cache_key and signature:
        _STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE[cache_key] = {
            "signature": signature,
            "data": snapshot,
        }
        if shared_key:
            _shared_cache_set(shared_key, snapshot, ttl_seconds=_STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE_TTL)
    return snapshot


def _stock_card_financial_ratios_from_snapshot(snapshot: Dict[str, Any], *, market_cap: Any) -> Dict[str, Any]:
    market_cap_value = _positive_float(market_cap)
    ttm_net_kar = _as_finite_float(snapshot.get("ttm_net_kar"))
    ttm_favok = _as_finite_float(snapshot.get("ttm_favok"))
    ozkaynaklar = _as_finite_float(snapshot.get("ozkaynaklar"))
    net_borc = _as_finite_float(snapshot.get("net_borc"))
    enterprise_value = (
        market_cap_value + net_borc
        if market_cap_value is not None and net_borc is not None
        else None
    )

    return {
        "fk": _round_market_ratio(market_cap_value, ttm_net_kar),
        "pd_dd": _round_market_ratio(market_cap_value, ozkaynaklar),
        "fd_favok": _round_market_ratio(enterprise_value, ttm_favok),
        "net_borc_favok": _round_market_ratio(net_borc, ttm_favok),
        "enterprise_value": enterprise_value,
    }


def _stock_card_financial_ratios_from_cache(symbol: str) -> Dict[str, Optional[float]]:
    snapshot = _stock_card_financial_snapshot_from_cache(symbol)
    ratios = _stock_card_financial_ratios_from_snapshot(snapshot, market_cap=None)
    net_borc_favok = ratios.get("net_borc_favok")
    if net_borc_favok is None and snapshot:
        try:
            net_borc = _as_finite_float(snapshot.get("net_borc"))
            ttm_favok = _as_finite_float(snapshot.get("ttm_favok"))
            if net_borc is not None and ttm_favok is not None and abs(ttm_favok) > 1e-12:
                net_borc_favok = round(net_borc / ttm_favok, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            net_borc_favok = None
    return {"net_borc_favok": net_borc_favok}


def _resolve_market_card_valuation_from_cached_data(
    cached: Dict[str, Any],
    *,
    market_cap: Any,
) -> Dict[str, Any]:
    snapshot = cached.get("financial_snapshot") if isinstance(cached.get("financial_snapshot"), dict) else {}
    computed = _stock_card_financial_ratios_from_snapshot(snapshot, market_cap=market_cap)
    provider = cached.get("provider_ratios") if isinstance(cached.get("provider_ratios"), dict) else {}

    values: Dict[str, Optional[float]] = {}
    sources: Dict[str, str] = {}
    for key in ("fk", "pd_dd", "fd_favok"):
        computed_value = computed.get(key)
        if not _is_missing_market_ratio(computed_value):
            values[key] = computed_value
            sources[key] = "kap_computed"
            continue
        provider_value = provider.get(key)
        values[key] = provider_value
        if not _is_missing_market_ratio(provider_value):
            sources[key] = str(cached.get("provider_source") or "external_ratio_provider")

    net_borc_favok = computed.get("net_borc_favok")
    if _is_missing_market_ratio(net_borc_favok):
        fallback_ratios = cached.get("fallback_financial_ratios")
        if isinstance(fallback_ratios, dict):
            net_borc_favok = fallback_ratios.get("net_borc_favok")

    return {
        **values,
        "net_borc_favok": net_borc_favok,
        "valuation_source": "kap_computed" if sources and set(sources.values()) == {"kap_computed"} else "mixed",
        "valuation_sources": sources,
        "valuation_as_of": cached.get("as_of"),
        "valuation_financial_period": snapshot.get("latest_quarter"),
        "enterprise_value": computed.get("enterprise_value"),
        "cache_hit": bool(cached.get("_cache_hit")),
    }


def _resolve_market_card_valuation(symbol: str, *, market_cap: Any) -> Dict[str, Any]:
    ticker = str(symbol or "").strip().upper()
    now = time.time()
    cached = _STOCK_CARD_VALUATION_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < _STOCK_CARD_VALUATION_CACHE_TTL:
        return _resolve_market_card_valuation_from_cached_data(
            {**cached, "_cache_hit": True},
            market_cap=market_cap,
        )
    shared_key = f"api:stock-card-valuation:{ticker}"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached:
        _STOCK_CARD_VALUATION_CACHE[ticker] = {**shared_cached, "_ts": now}
        return _resolve_market_card_valuation_from_cached_data(
            {**shared_cached, "_cache_hit": True},
            market_cap=market_cap,
        )

    snapshot = _stock_card_financial_snapshot_from_cache(ticker)
    computed = _stock_card_financial_ratios_from_snapshot(snapshot, market_cap=market_cap)
    needs_provider = any(_is_missing_market_ratio(computed.get(key)) for key in ("fk", "pd_dd", "fd_favok"))

    provider_payload: Dict[str, Any] = {}
    provider_ratios: Dict[str, Optional[float]] = {}
    if needs_provider:
        provider_payload = _fetch_isyatirim_multiples(ticker)
        provider_ratios = _resolve_market_card_multiples(ticker, provider_payload)

    cache_ratios = _stock_card_financial_ratios_from_cache(ticker)
    cached_payload = {
        "_ts": now,
        "financial_snapshot": snapshot,
        "provider_ratios": provider_ratios,
        "provider_source": provider_payload.get("source") if provider_payload.get("ok") else None,
        "provider_as_of": provider_payload.get("fetched_at"),
        "fallback_financial_ratios": cache_ratios,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _STOCK_CARD_VALUATION_CACHE[ticker] = cached_payload
    _shared_cache_set(shared_key, cached_payload, _STOCK_CARD_VALUATION_CACHE_TTL)
    return _resolve_market_card_valuation_from_cached_data(cached_payload, market_cap=market_cap)


def _market_stock_cards_payload(*, symbols: str, force_refresh: bool = False) -> Dict[str, Any]:
    normalized_symbols = _normalize_market_stock_card_symbols(symbols)
    if not normalized_symbols:
        return {
            "items": [],
            "source": "infoyatirim_yahoo_chart",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    response_cache_key = _stock_cards_response_cache_key(normalized_symbols)
    if not force_refresh:
        shared_cached = _shared_cache_get_dict(response_cache_key)
        if shared_cached is not None:
            return dict(shared_cached)

    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    price_map = _fetch_market_price_map(normalized_symbols)
    basic_summary_map = _fetch_isyatirim_basic_summary_map()
    return_base_map = _fetch_stock_return_bases_bulk(normalized_symbols)
    instrument_map = get_instruments(CONFIG.paths.processed_dir, "stock", normalized_symbols)

    items: List[Dict[str, Any]] = []
    for symbol in normalized_symbols:
        quote = price_map.get(symbol, {})
        intraday = _fetch_stock_card_intraday(symbol, force_refresh=force_refresh)
        cached_meta = _load_cached_kap_market_metadata(cache_dir, symbol)
        instrument = instrument_map.get(symbol)
        company_name = str((instrument or {}).get("name") or cached_meta.get("company_title") or "").strip() or symbol
        basic_summary = basic_summary_map.get(symbol)
        market_cap = _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary)
        valuation = _resolve_market_card_valuation(symbol, market_cap=market_cap)
        return_bases = return_base_map.get(symbol, {})

        price = _first_not_none(quote.get("price"), intraday.get("price"))
        currency = quote.get("currency") or intraday.get("currency") or "TRY"
        volume_tl = quote.get("volume")
        volume_lot = _first_not_none(intraday.get("volume_lot"), intraday.get("volume"))
        current_for_returns = price if price is not None else return_bases.get("latest_close")
        session_status = intraday.get("session_status") or "unknown"
        as_of = intraday.get("as_of") if session_status != "open" else (quote.get("as_of") or intraday.get("as_of"))
        item = {
            "symbol": symbol,
            "company": company_name,
            "yahoo_symbol": intraday.get("yahoo_symbol"),
            "price": price,
            "currency": currency,
            "change": _first_not_none(quote.get("change"), intraday.get("change")),
            "change_pct": _first_not_none(quote.get("change_pct"), intraday.get("change_pct")),
            "volume": _first_not_none(volume_tl, volume_lot),
            "volume_lot": volume_lot,
            "volume_tl": volume_tl,
            "market_cap": market_cap,
            "high": intraday.get("high"),
            "low": intraday.get("low"),
            "previous_close": intraday.get("prev_close"),
            "fk": valuation.get("fk"),
            "pd_dd": valuation.get("pd_dd"),
            "fd_favok": valuation.get("fd_favok"),
            "net_borc_favok": valuation.get("net_borc_favok"),
            "valuation_source": valuation.get("valuation_source"),
            "valuation_sources": valuation.get("valuation_sources"),
            "valuation_as_of": valuation.get("valuation_as_of"),
            "valuation_financial_period": valuation.get("valuation_financial_period"),
            "valuation_cache_hit": valuation.get("cache_hit"),
            "market_state": quote.get("market_state") or intraday.get("market_state") or "",
            "as_of": as_of,
            "session_status": session_status,
            "session_label": intraday.get("session_label"),
            "is_live": intraday.get("is_live"),
            "is_stale": intraday.get("is_stale"),
            "last_trade_at": intraday.get("last_trade_at"),
            "last_trade_date": intraday.get("last_trade_date"),
            "line_points": intraday.get("line_points") or [],
            "error": None if price is not None or intraday.get("line_points") else intraday.get("error"),
            "logo_url": (instrument or {}).get("logo_url"),
            "logo_source": (instrument or {}).get("logo_source"),
            **_returns_from_bases(current_for_returns, return_bases),
        }
        items.append(item)

    data = {
        "items": items,
        "source": "infoyatirim_yahoo_chart",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _shared_cache_set(
        response_cache_key,
        data,
        ttl_seconds=_MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL,
    )
    return data


def _strip_html_cell(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(raw or ""), flags=re.IGNORECASE)
    text = html.unescape(text)
    return " ".join(text.split())


def _norm_text(value: str) -> str:
    return (
        str(value or "")
        .upper()
        .replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )


def _extract_isyatirim_basic_summary_map(html_text: str) -> Dict[str, Dict[str, Any]]:
    table_match = re.search(
        r'<table[^>]+data-csvname="temelozet"[^>]*>(.*?)</table>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not table_match:
        return {}

    table_html = table_match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
    if not rows:
        return {}

    header_indexes: Dict[str, int] = {}
    items: Dict[str, Dict[str, Any]] = {}
    for row_html in rows:
        header_cells = [_strip_html_cell(cell) for cell in re.findall(r"<th[^>]*>(.*?)</th>", row_html, flags=re.IGNORECASE | re.DOTALL)]
        if header_cells:
            for idx, header in enumerate(header_cells):
                norm = _norm_text(header)
                if norm == "KOD":
                    header_indexes["symbol"] = idx
                elif "PIYASA DEGERI" in norm and "MN TL" in norm:
                    header_indexes["market_cap_mn_try"] = idx
                elif "HALKA ACIKLIK" in norm:
                    header_indexes["free_float_pct"] = idx
                elif norm.startswith("SERMAYE") and "MN TL" in norm:
                    header_indexes["capital_mn_try"] = idx
            continue

        cells = [_strip_html_cell(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)]
        if not cells:
            continue

        symbol_idx = header_indexes.get("symbol", 0)
        market_cap_idx = header_indexes.get("market_cap_mn_try", 4)
        free_float_idx = header_indexes.get("free_float_pct", 6)
        capital_idx = header_indexes.get("capital_mn_try", 7)
        if symbol_idx >= len(cells):
            continue
        symbol = str(cells[symbol_idx] or "").strip().upper()
        if not symbol:
            continue

        market_cap_mn_try = _parse_tr_decimal(cells[market_cap_idx]) if market_cap_idx < len(cells) else None
        free_float_pct = _parse_tr_decimal(cells[free_float_idx]) if free_float_idx < len(cells) else None
        capital_mn_try = _parse_tr_decimal(cells[capital_idx]) if capital_idx < len(cells) else None
        items[symbol] = {
            "market_cap": market_cap_mn_try * 1_000_000 if market_cap_mn_try is not None else None,
            "fdpo": round(free_float_pct / 100.0, 6) if free_float_pct is not None and free_float_pct > 0 else None,
            "shares_outstanding": capital_mn_try * 1_000_000 if capital_mn_try is not None else None,
            "source": "isyatirim_temelozet",
        }

    return items


def _fetch_isyatirim_basic_summary_map() -> Dict[str, Dict[str, Any]]:
    import urllib.error
    import urllib.request

    now = time.time()
    cached = _ISYATIRIM_BASIC_SUMMARY_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _ISYATIRIM_BASIC_SUMMARY_CACHE_TTL:
        return cached.get("items", {})
    shared_cached = _shared_cache_get_dict("api:isyatirim-basic-summary")
    if shared_cached:
        items = dict(shared_cached.get("items") or {})
        _ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": items}
        return items

    request = urllib.request.Request(
        url=_isyatirim_basic_summary_url(),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, Exception):
        _ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": {}}
        return {}

    items = _extract_isyatirim_basic_summary_map(html_text)
    _ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": items}
    _shared_cache_set("api:isyatirim-basic-summary", {"items": items}, _ISYATIRIM_BASIC_SUMMARY_CACHE_TTL)
    return items


def _extract_isyatirim_historical_averages(html_text: str, symbol: str) -> Dict[str, Any]:
    table_match = re.search(
        r'<table[^>]+data-csvname="tarihselortalamalar"[^>]*>(.*?)</table>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not table_match:
        return {"ok": False, "error": "İş Yatırım 'Tarihsel Ortalamalar' tablosu bulunamadı."}

    table_html = table_match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
    if not rows:
        return {"ok": False, "error": "İş Yatırım tablosunda satır bulunamadı."}

    header_cells: Optional[List[str]] = None
    selected_cells: Optional[List[str]] = None
    symbol_upper = str(symbol or "").strip().upper()
    for row_html in rows:
        header_raw = re.findall(r"<th[^>]*>(.*?)</th>", row_html, flags=re.IGNORECASE | re.DOTALL)
        if header_raw:
            header_cells = [_strip_html_cell(cell) for cell in header_raw]
            continue

        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)
        clean_cells = [_strip_html_cell(cell) for cell in cells]
        if not clean_cells:
            continue
        row_symbol = clean_cells[0].upper() if clean_cells else ""
        if row_symbol == symbol_upper:
            selected_cells = clean_cells
            break
        if selected_cells is None and len(clean_cells) >= 3:
            selected_cells = clean_cells

    if not selected_cells or len(selected_cells) < 3:
        return {"ok": False, "error": "İş Yatırım tablosundan çarpan verisi ayrıştırılamadı."}

    index_fk = None
    index_fd_favok = None
    index_pd_dd = None
    index_fk_prim = None
    index_fd_favok_prim = None
    index_pd_dd_prim = None

    if header_cells:
        for idx, header in enumerate(header_cells):
            norm = _norm_text(str(header))
            if "F/K" in norm and "TAHMIN" in norm:
                index_fk = idx
            elif ("FD/FAVOK" in norm or "FD/FAVÖK" in norm) and "TAHMIN" in norm:
                index_fd_favok = idx
            elif "PD/DD" in norm and "TAHMIN" in norm:
                index_pd_dd = idx
        if index_fk is not None and index_fk + 1 < len(header_cells) and "PRIM" in _norm_text(header_cells[index_fk + 1]):
            index_fk_prim = index_fk + 1
        if (
            index_fd_favok is not None
            and index_fd_favok + 1 < len(header_cells)
            and "PRIM" in _norm_text(header_cells[index_fd_favok + 1])
        ):
            index_fd_favok_prim = index_fd_favok + 1
        if index_pd_dd is not None and index_pd_dd + 1 < len(header_cells) and "PRIM" in _norm_text(header_cells[index_pd_dd + 1]):
            index_pd_dd_prim = index_pd_dd + 1

    if index_fk is None:
        index_fk = 1 if len(selected_cells) > 1 else None
    if index_fk_prim is None and index_fk is not None and len(selected_cells) > index_fk + 1:
        index_fk_prim = index_fk + 1

    if index_fd_favok is None and len(selected_cells) >= 7:
        # Non-bank layout fallback: KOD, F/K, Prim, FD/FAVOK, Prim, PD/DD, Prim
        index_fd_favok = 3
    if index_fd_favok_prim is None and index_fd_favok is not None and len(selected_cells) > index_fd_favok + 1:
        index_fd_favok_prim = index_fd_favok + 1

    if index_pd_dd is None:
        if len(selected_cells) >= 7:
            index_pd_dd = 5
        elif len(selected_cells) >= 5:
            # Bank/insurance layout: KOD, F/K, Prim, PD/DD, Prim
            index_pd_dd = 3
        else:
            index_pd_dd = None
    if index_pd_dd_prim is None and index_pd_dd is not None and len(selected_cells) > index_pd_dd + 1:
        index_pd_dd_prim = index_pd_dd + 1

    fk = _parse_tr_decimal(selected_cells[index_fk]) if index_fk is not None and len(selected_cells) > index_fk else None
    fd_favok = (
        _parse_tr_decimal(selected_cells[index_fd_favok])
        if index_fd_favok is not None and len(selected_cells) > index_fd_favok
        else None
    )
    pd_dd = _parse_tr_decimal(selected_cells[index_pd_dd]) if index_pd_dd is not None and len(selected_cells) > index_pd_dd else None

    fk_prim_isk = (
        _parse_tr_decimal(selected_cells[index_fk_prim])
        if index_fk_prim is not None and len(selected_cells) > index_fk_prim
        else None
    )
    fd_favok_prim_isk = (
        _parse_tr_decimal(selected_cells[index_fd_favok_prim])
        if index_fd_favok_prim is not None and len(selected_cells) > index_fd_favok_prim
        else None
    )
    pd_dd_prim_isk = (
        _parse_tr_decimal(selected_cells[index_pd_dd_prim])
        if index_pd_dd_prim is not None and len(selected_cells) > index_pd_dd_prim
        else None
    )

    note_match = re.search(
        r'<div[^>]+class="table-note"[^>]*>(.*?)</div>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    note = _strip_html_cell(note_match.group(1)) if note_match else None

    return {
        "ok": True,
        "fk": fk,
        "fd_favok": fd_favok,
        "pd_dd": pd_dd,
        "fk_prim_iskonto_pct": fk_prim_isk,
        "fd_favok_prim_iskonto_pct": fd_favok_prim_isk,
        "pd_dd_prim_iskonto_pct": pd_dd_prim_isk,
        "note": note,
    }


def _fetch_isyatirim_multiples(symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {
            "ok": False,
            "symbol": "",
            "error": "Sembol bos.",
        }

    cache_key = ticker
    now = time.time()
    cached = _ISYATIRIM_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _ISYATIRIM_CACHE_TTL:
        return cached
    shared_key = f"api:isyatirim-multiples:{cache_key}"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached:
        _ISYATIRIM_CACHE[cache_key] = {**shared_cached, "_ts": now}
        return dict(shared_cached)

    url = _isyatirim_company_card_url(ticker)
    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, Exception) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "source": "isyatirim_company_card",
            "url": url,
            "error": f"İş Yatırım bağlantı hatası: {exc}",
        }

    parsed = _extract_isyatirim_historical_averages(html_text, ticker)
    payload: Dict[str, Any] = {
        "ok": bool(parsed.get("ok")),
        "symbol": ticker,
        "source": "isyatirim_company_card",
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "_ts": now,
    }
    payload.update(parsed)
    _ISYATIRIM_CACHE[cache_key] = payload
    _shared_cache_set(shared_key, payload, _ISYATIRIM_CACHE_TTL)
    return payload


def _fetch_kap_price_payload(symbol: str) -> Dict[str, Any]:
    """Fetch latest stock price from Yahoo Finance for a BIST ticker."""
    import urllib.request
    import urllib.error

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {
            "ok": False,
            "symbol": "",
            "error": "Sembol bos.",
        }
    cache_key = ticker
    now = time.time()

    # Return cached if fresh
    cached = _PRICE_CACHE.get(cache_key)
    if cached and now - cached["_ts"] < _PRICE_CACHE_TTL:
        return cached
    shared_key = f"api:market:stock-price:{ticker}:v1"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        result = dict(shared_cached)
        result["_ts"] = now
        _PRICE_CACHE[cache_key] = result
        return result

    yahoo_symbol = f"{ticker}.IS"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval=1d&range=1d"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "error": f"Yahoo Finance bağlantı hatası: {exc}",
        }

    try:
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        currency = meta.get("currency", "TRY")
        market_state = meta.get("marketState", "")
        regular_market_time = meta.get("regularMarketTime")

        change = None
        change_pct = None
        if price is not None and prev_close:
            change = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2)
        as_of = None
        if isinstance(regular_market_time, (int, float)):
            try:
                as_of = datetime.fromtimestamp(float(regular_market_time), tz=timezone.utc).isoformat()
            except Exception:
                as_of = None

        result: Dict[str, Any] = {
            "ok": True,
            "symbol": ticker,
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "currency": currency,
            "market_state": market_state,
            "as_of": as_of,
            "_ts": now,
        }
        _PRICE_CACHE[cache_key] = result
        _shared_cache_set(shared_key, {k: v for k, v in result.items() if k != "_ts"}, ttl_seconds=_PRICE_CACHE_TTL)
        return result
    except (KeyError, IndexError, TypeError) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "error": f"Yahoo Finance veri parse hatası: {exc}",
        }


@app.get("/kap/price")
def kap_price(symbol: str = Query(..., min_length=1)) -> Dict[str, Any]:
    return _fetch_kap_price_payload(symbol)


# ── XU030 universe ────────────────────────────────────────
_XU030_CACHE: Dict[str, Any] = {}
_XU030_CACHE_TTL = 120  # 2 minutes


def _fill_prices_via_yahoo(symbols: List[str], base_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """For symbols missing a price in base_map, query Yahoo Finance individually."""
    result = dict(base_map)
    missing = [s for s in symbols if not (result.get(s) or {}).get("price")]
    if not missing:
        return result

    from concurrent.futures import ThreadPoolExecutor

    def _one(sym: str) -> tuple[str, Dict[str, Any]]:
        payload = _fetch_kap_price_payload(sym)
        if not payload.get("ok"):
            return sym, {}
        return sym, {
            "price": payload.get("price"),
            "currency": payload.get("currency") or "TRY",
            "change": payload.get("change"),
            "change_pct": payload.get("change_pct"),
            "market_state": payload.get("market_state") or "",
            "as_of": payload.get("as_of"),
        }

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for sym, quote in pool.map(_one, missing):
                if quote:
                    result[sym] = quote
    except Exception:
        for sym in missing:
            s, quote = _one(sym)
            if quote:
                result[s] = quote
    return result


def _xu030_payload() -> Dict[str, Any]:
    from app.kap_service import get_bist30_companies

    now = time.time()
    cached = _XU030_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _XU030_CACHE_TTL:
        return cached["data"]

    symbols = get_bist30_companies()
    base_map = _fetch_market_price_map(symbols)
    price_map = _fill_prices_via_yahoo(symbols, base_map)
    basic_summary_map = _fetch_isyatirim_basic_summary_map()

    cache_dir = CONFIG.paths.processed_dir / "kap_cache"

    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        cached_meta = _load_cached_kap_market_metadata(cache_dir, symbol)
        latest_quarter = cached_meta.get("latest_quarter")
        quote = price_map.get(symbol, {})
        rows.append(
            {
                "company": symbol,
                "latest_quarter": latest_quarter,
                "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
                "price": quote.get("price"),
                "price_currency": quote.get("currency"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "price_as_of": quote.get("as_of"),
                "market_cap": _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary_map.get(symbol)),
                **_empty_logo_payload(),
            }
        )

    data = {"index": "XU030", "rows": rows, "as_of": datetime.now(timezone.utc).isoformat()}
    _XU030_CACHE["payload"] = {"_ts": now, "data": data}
    return data


@app.get("/market/xu030")
def market_xu030() -> Dict[str, Any]:
    return _xu030_payload()


# ── Commodities (Yahoo-backed, provider-delayed) ──────────
_COMMODITY_CACHE: Dict[str, Any] = {}
_COMMODITY_CACHE_TTL = 3  # 3 seconds

# Display symbol -> (Yahoo ticker, Turkish label, override currency)
_COMMODITY_MAP: List[tuple[str, str, str, Optional[str]]] = [
    ("BRENT", "BZ=F", "Brent Petrol", "USD"),
    ("WTI", "CL=F", "WTI Ham Petrol", "USD"),
    ("DOGALGAZ", "NG=F", "Doğal Gaz", "USD"),
    ("ALTIN", "GC=F", "Altın (Ons)", "USD"),
    ("GUMUS", "SI=F", "Gümüş (Ons)", "USD"),
    ("BAKIR", "HG=F", "Bakır", "USD"),
    ("PLATIN", "PL=F", "Platin", "USD"),
    ("PALADYUM", "PA=F", "Paladyum", "USD"),
    ("KAHVE", "KC=F", "Kahve", "USD"),
    ("SEKER", "SB=F", "Şeker", "USD"),
    ("BUGDAY", "ZW=F", "Buğday", "USD"),
    ("MISIR", "ZC=F", "Mısır", "USD"),
    ("PAMUK", "CT=F", "Pamuk", "USD"),
    ("KAKAO", "CC=F", "Kakao", "USD"),
    ("SOYA", "ZS=F", "Soya Fasulyesi", "USD"),
]


def _fetch_yahoo_quote(yahoo_symbol: str) -> Dict[str, Any]:
    """Low-level Yahoo chart fetch for an arbitrary ticker (no cache)."""
    import urllib.error
    import urllib.request

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval=1d&range=1d"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {"ok": False, "error": f"yahoo_error: {exc}"}

    try:
        meta = data["chart"]["result"][0]["meta"]
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "error": f"yahoo_parse: {exc}"}

    price = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    currency = meta.get("currency")
    short_name = meta.get("shortName")
    long_name = meta.get("longName")
    market_state = meta.get("marketState", "")
    rmt = meta.get("regularMarketTime")
    high = meta.get("regularMarketDayHigh")
    low = meta.get("regularMarketDayLow")
    volume = meta.get("regularMarketVolume")

    change = None
    change_pct = None
    if price is not None and prev_close:
        try:
            change = round(float(price) - float(prev_close), 4)
            change_pct = round((change / float(prev_close)) * 100, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            change = None
            change_pct = None

    as_of = None
    if isinstance(rmt, (int, float)):
        try:
            as_of = datetime.fromtimestamp(float(rmt), tz=timezone.utc).isoformat()
        except Exception:
            as_of = None

    return {
        "ok": True,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "high": high,
        "low": low,
        "volume": volume,
        "currency": currency,
        "short_name": short_name,
        "long_name": long_name,
        "market_state": market_state,
        "as_of": as_of,
    }


def _normalize_market_index(index_code: str) -> str:
    normalized = str(index_code or "").strip().upper()
    if normalized not in _MARKET_INDEX_META:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen endeks. {_supported_market_indexes_text()} kullanin.",
        )
    return normalized


def _parse_yahoo_chart_payload(data: Dict[str, Any], yahoo_symbol: str) -> Dict[str, Any]:
    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "error": f"yahoo_parse: {exc}", "yahoo_symbol": yahoo_symbol}

    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    volumes = quote.get("volume") or []

    points: List[Dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        close = closes[idx] if idx < len(closes) else None
        if close is None or not isinstance(ts, (int, float)):
            continue
        try:
            numeric_close = float(close)
        except (TypeError, ValueError):
            continue
        if numeric_close <= 0:
            continue

        def _numeric_at(values: List[Any]) -> Optional[float]:
            if idx >= len(values):
                return None
            value = values[idx]
            if value is None or isinstance(value, bool):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        points.append(
            {
                "time": datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(),
                "close": numeric_close,
                "high": _numeric_at(highs),
                "low": _numeric_at(lows),
                "volume": _numeric_at(volumes),
            }
        )

    return {
        "ok": True,
        "yahoo_symbol": yahoo_symbol,
        "meta": meta,
        "points": points,
    }


def _fetch_yahoo_chart_url(url: str, yahoo_symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {"ok": False, "error": f"yahoo_error: {exc}", "yahoo_symbol": yahoo_symbol}

    return _parse_yahoo_chart_payload(data, yahoo_symbol)


def _fetch_yahoo_chart_raw(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval={interval}&range={range_}"
    )
    return _fetch_yahoo_chart_url(url, yahoo_symbol)


def _fetch_yahoo_chart_period_raw(
    yahoo_symbol: str,
    *,
    interval: str,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    period1 = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval={interval}&period1={period1}&period2={period2}"
    )
    return _fetch_yahoo_chart_url(url, yahoo_symbol)


def _format_isyatirim_chart_datetime(value: date) -> str:
    return f"{value.year:04d}{value.month:02d}{value.day:02d}000000"


def _parse_isyatirim_chart_datetime(raw: Any) -> Optional[datetime]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        try:
            timestamp = float(raw)
            if timestamp > 10_000_000_000:
                timestamp /= 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except Exception:
            return None
    return _point_datetime(raw)


def _fetch_isyatirim_index_history(index_code: str, *, start_date: date, end_date: date) -> List[tuple[datetime, float]]:
    import urllib.error
    import urllib.request

    normalized = _normalize_market_index(index_code)
    url = (
        "https://www.isyatirim.com.tr/_Layouts/15/IsYatirim.Website/Common/ChartData.aspx/"
        "IndexHistoricalAll"
        f"?period=1440&from={_format_isyatirim_chart_datetime(start_date)}"
        f"&to={_format_isyatirim_chart_datetime(end_date)}&endeks={normalized}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Endeksler.aspx",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8", errors="ignore"))
    except (urllib.error.URLError, Exception):
        return []

    raw_rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(raw_rows, list):
        return []

    points: List[tuple[datetime, float]] = []
    for row in raw_rows:
        raw_time = None
        raw_close = None
        if isinstance(row, dict):
            raw_time = row.get("d") or row.get("date") or row.get("time") or row.get("x")
            raw_close = row.get("c") or row.get("close") or row.get("last") or row.get("y")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            raw_time = row[0]
            raw_close = row[1]

        point_dt = _parse_isyatirim_chart_datetime(raw_time)
        close = _numeric_chart_value(raw_close)
        if point_dt is None or close is None or close <= 0:
            continue
        points.append((point_dt, close))

    points.sort(key=lambda item: item[0])
    return points


def _index_return_bases_from_points(
    points: List[tuple[datetime, float]],
    *,
    history_source: str,
    provider_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    cleaned: List[tuple[datetime, float]] = []
    for point_dt, close in points:
        if not isinstance(point_dt, datetime):
            continue
        if point_dt.tzinfo is None:
            point_dt = point_dt.replace(tzinfo=timezone.utc)
        cleaned.append((point_dt.astimezone(timezone.utc), float(close)))
    if not cleaned:
        return {}

    cleaned.sort(key=lambda item: item[0])
    latest_dt, latest_close = cleaned[-1]
    year_start = datetime(latest_dt.year, 1, 1, tzinfo=timezone.utc)
    bases = {
        "base_1w": _pick_series_value_at_or_before(cleaned, latest_dt - timedelta(days=7)),
        "base_1m": _pick_series_value_at_or_before(cleaned, latest_dt - timedelta(days=30)),
        "base_3m": _pick_series_value_at_or_before(cleaned, latest_dt - timedelta(days=91)),
        "base_6m": _pick_series_value_at_or_before(cleaned, latest_dt - timedelta(days=182)),
        "base_ytd": _pick_series_value_at_or_after(cleaned, year_start),
        "base_1y": _pick_series_value_at_or_before(cleaned, latest_dt - timedelta(days=365)),
        "base_5y": _pick_series_value_at_or_before(cleaned, latest_dt - timedelta(days=365 * 5)),
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
        "history_source": history_source,
    }
    if provider_symbol:
        bases["yahoo_symbol"] = provider_symbol
    return bases


def _index_return_bases_have_period_history(bases: Dict[str, Any]) -> bool:
    return any(
        bases.get(key) is not None
        for key in ("base_1w", "base_1m", "base_3m", "base_6m", "base_1y")
    )


def _fetch_index_quote(index_code: str) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    now = time.time()
    cached = _MARKET_INDEX_QUOTE_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_INDEX_QUOTE_CACHE_TTL:
        return dict(cached.get("data") or {})

    meta = _MARKET_INDEX_META[normalized]
    errors: List[str] = []
    for yahoo_symbol in meta["yahoo_candidates"]:
        quote = _fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            row = {
                "symbol": normalized,
                "label": meta["label"],
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price"),
                "prev_close": quote.get("prev_close"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "volume": quote.get("volume"),
                "currency": quote.get("currency") or "TRY",
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
            }
            _MARKET_INDEX_QUOTE_CACHE[normalized] = {"_ts": now, "data": row}
            return dict(row)
        errors.append(str(quote.get("error") or "quote_unavailable"))

    fallback = {
        "symbol": normalized,
        "label": meta["label"],
        "yahoo_symbol": None,
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "high": None,
        "low": None,
        "volume": None,
        "currency": "TRY",
        "market_state": "",
        "as_of": None,
        "error": "; ".join(errors[:3]) if errors else "quote_unavailable",
    }
    _MARKET_INDEX_QUOTE_CACHE[normalized] = {"_ts": now, "data": fallback}
    return dict(fallback)


def _fetch_index_return_bases(index_code: str) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    now = time.time()
    cached = _MARKET_INDEX_RETURN_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_INDEX_RETURN_CACHE_TTL:
        return dict(cached.get("data") or {})
    shared_key = f"api:market:index-return-bases:{normalized}:v2"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        _MARKET_INDEX_RETURN_CACHE[normalized] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached)

    meta = _MARKET_INDEX_META[normalized]
    yahoo_partial: Optional[Dict[str, Any]] = None
    for yahoo_symbol in meta["yahoo_candidates"]:
        chart = _fetch_yahoo_chart_raw(yahoo_symbol, interval="1d", range_="5y")
        if not chart.get("ok"):
            continue
        points = [
            (datetime.fromisoformat(str(point["time"])), float(point["close"]))
            for point in chart.get("points", [])
            if point.get("time") and isinstance(point.get("close"), (int, float))
        ]
        if not points:
            continue
        bases = _index_return_bases_from_points(points, history_source="yahoo", provider_symbol=yahoo_symbol)
        if _index_return_bases_have_period_history(bases):
            _MARKET_INDEX_RETURN_CACHE[normalized] = {"_ts": now, "data": bases}
            _shared_cache_set(shared_key, bases, ttl_seconds=_MARKET_INDEX_RETURN_CACHE_TTL)
            return dict(bases)
        if yahoo_partial is None:
            yahoo_partial = bases

    today = datetime.now(_TURKEY_TIMEZONE).date()
    history_start = today - timedelta(days=(365 * 5) + 14)
    history_points = _fetch_isyatirim_index_history(normalized, start_date=history_start, end_date=today + timedelta(days=1))
    bases = _index_return_bases_from_points(history_points, history_source="isyatirim")
    if _index_return_bases_have_period_history(bases):
        _MARKET_INDEX_RETURN_CACHE[normalized] = {"_ts": now, "data": bases}
        _shared_cache_set(shared_key, bases, ttl_seconds=_MARKET_INDEX_RETURN_CACHE_TTL)
        return dict(bases)

    if yahoo_partial and _index_return_bases_have_period_history(yahoo_partial):
        _MARKET_INDEX_RETURN_CACHE[normalized] = {"_ts": now, "data": yahoo_partial}
        _shared_cache_set(shared_key, yahoo_partial, ttl_seconds=_MARKET_INDEX_RETURN_CACHE_TTL)
        return dict(yahoo_partial)

    _MARKET_INDEX_RETURN_CACHE[normalized] = {"_ts": now, "data": {}}
    _shared_cache_set(shared_key, {}, ttl_seconds=_MARKET_INDEX_RETURN_CACHE_TTL)
    return {}


def _index_returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        response_field: _return_pct(current_price, return_bases.get(base_field))
        for response_field, base_field in _INDEX_RETURN_BASE_FIELDS
    }


def _market_index_row(index_code: str, *, quote: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    quote_row = dict(quote or _fetch_index_quote(normalized))
    bases = _fetch_index_return_bases(normalized)
    current_for_returns = quote_row.get("price") if quote_row.get("price") is not None else bases.get("latest_close")
    return {
        "symbol": normalized,
        "label": _MARKET_INDEX_META[normalized]["label"],
        "yahoo_symbol": quote_row.get("yahoo_symbol") or bases.get("yahoo_symbol"),
        "price": quote_row.get("price"),
        "prev_close": quote_row.get("prev_close"),
        "change": quote_row.get("change"),
        "change_pct": quote_row.get("change_pct"),
        "high": quote_row.get("high"),
        "low": quote_row.get("low"),
        "volume": quote_row.get("volume"),
        "currency": quote_row.get("currency") or "TRY",
        "market_state": quote_row.get("market_state") or "",
        "as_of": quote_row.get("as_of") or bases.get("as_of"),
        "error": quote_row.get("error"),
        **_index_returns_from_bases(current_for_returns, bases),
    }


def _comparison_error_message(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc) or exc.__class__.__name__


def _comparison_history_result(
    asset: MarketComparisonHistoryAsset,
    *,
    symbol: str,
    label: Optional[str],
    points: List[Dict[str, Any]],
    source: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": asset.id or f"{asset.kind}:{symbol}",
        "kind": asset.kind,
        "symbol": symbol,
        "label": label or asset.label or symbol,
        "points": points,
        "source": source,
        "error": error,
    }


def _comparison_history_points_from_chart(
    chart: Dict[str, Any],
    *,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for point in list(chart.get("points") or []):
        if not isinstance(point, dict):
            continue
        point_dt = _point_datetime(point.get("time"))
        close = _numeric_chart_value(point.get("close"))
        if point_dt is None or close is None or close <= 0:
            continue
        point_date = point_dt.date()
        if point_date < start_date or point_date > end_date:
            continue
        date_key = point_date.isoformat()
        deduped[date_key] = {"date": date_key, "value": close}
    return [deduped[key] for key in sorted(deduped)]


def _fund_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    from app.fund_service import get_fund_performance_payload, normalize_fund_code

    symbol = normalize_fund_code(asset.symbol)
    try:
        payload = get_fund_performance_payload(
            CONFIG.paths.processed_dir,
            symbol,
            start_date=start_date,
            end_date=end_date,
        )
        points: List[Dict[str, Any]] = []
        for point in list(payload.get("points") or []):
            try:
                price = float(point.get("price"))
            except (TypeError, ValueError):
                continue
            point_date = str(point.get("date") or "")
            if not point_date or price <= 0:
                continue
            if point_date < start_date.isoformat() or point_date > end_date.isoformat():
                continue
            points.append({"date": point_date, "value": price})
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=points,
            source=str(payload.get("source") or "sqlite"),
            error=None if points else str(payload.get("source_metadata", {}).get("warning") or payload.get("status") or "data_unavailable"),
        )
    except Exception as exc:
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="sqlite",
            error=_comparison_error_message(exc),
        )


def _stock_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    try:
        symbol = _normalize_market_stock_card_symbol(asset.symbol)
        yahoo_symbol = f"{symbol}.IS"
        chart = _fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date,
        )
        points = _comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=points,
            source="yahoo_finance_chart",
            error=None if chart.get("ok") and points else str(chart.get("error") or "data_unavailable"),
        )
    except Exception as exc:
        symbol = str(asset.symbol or "").strip().upper()
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error=_comparison_error_message(exc),
        )


def _index_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    try:
        symbol = _normalize_market_index(asset.symbol)
        label = asset.label or _MARKET_INDEX_META[symbol]["label"]
        errors: List[str] = []
        for yahoo_symbol in _MARKET_INDEX_META[symbol]["yahoo_candidates"]:
            chart = _fetch_yahoo_chart_period_raw(
                yahoo_symbol,
                interval="1d",
                start_date=start_date,
                end_date=end_date,
            )
            points = _comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
            if chart.get("ok") and points:
                return _comparison_history_result(
                    asset,
                    symbol=symbol,
                    label=label,
                    points=points,
                    source="yahoo_finance_chart",
                )
            errors.append(str(chart.get("error") or "data_unavailable"))
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=label,
            points=[],
            source="yahoo_finance_chart",
            error="; ".join(errors[:3]) if errors else "data_unavailable",
        )
    except Exception as exc:
        symbol = str(asset.symbol or "").strip().upper()
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error=_comparison_error_message(exc),
        )


def _fx_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    symbol = str(asset.symbol or "").strip().upper()
    direct_map = {entry[0]: entry for entry in _FX_DIRECT_MAP}
    entry = direct_map.get(symbol)
    if not entry:
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error="unsupported_fx_symbol",
        )

    _, yahoo_candidates, default_label = entry
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        chart = _fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date,
        )
        points = _comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
        if chart.get("ok") and points:
            return _comparison_history_result(
                asset,
                symbol=symbol,
                label=asset.label or default_label,
                points=points,
                source="yahoo_finance_chart",
            )
        errors.append(str(chart.get("error") or "data_unavailable"))

    return _comparison_history_result(
        asset,
        symbol=symbol,
        label=asset.label or default_label,
        points=[],
        source="yahoo_finance_chart",
        error="; ".join(errors[:3]) if errors else "data_unavailable",
    )


def _comparison_history_cache_key(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> str:
    """Build a bounded key that intentionally excludes presentation fields."""

    symbol = re.sub(r"[^A-Z0-9/_-]", "", str(asset.symbol or "").strip().upper())[:32]
    return (
        "api:market:comparison-history:v2:"
        f"kind={asset.kind}:symbol={symbol}:from={start_date.isoformat()}:to={end_date.isoformat()}"
    )


def _comparison_history_cached_asset(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
    handler: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    """Cache a single normalized series, not the user-specific POST body."""

    latest_result: Optional[Dict[str, Any]] = None

    def build() -> Optional[Dict[str, Any]]:
        nonlocal latest_result
        latest_result = handler(asset, start_date=start_date, end_date=end_date)
        # A first-request provider failure must not become a multi-day negative
        # cache.  Existing stale success data is still served by the wrapper.
        return latest_result if not latest_result.get("error") else None

    historical_range = end_date < date.today()
    fresh_ttl = 24 * 60 * 60 if historical_range else 60
    stale_ttl = 7 * 24 * 60 * 60 if historical_range else 10 * 60
    cached, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key=_comparison_history_cache_key(asset, start_date=start_date, end_date=end_date),
        factory=build,
        fresh_ttl_seconds=fresh_ttl,
        stale_ttl_seconds=stale_ttl,
    )
    if cached is not None:
        result = dict(cached)
    elif latest_result is not None:
        result = dict(latest_result)
    else:
        # A waiter must not repeat the provider request after the bounded
        # single-flight wait.  The client can retain its prior chart and retry.
        result = _comparison_history_result(
            asset,
            symbol=str(asset.symbol or "").strip().upper(),
            label=asset.label,
            points=[],
            source="mixed",
            error="refresh_pending",
        )
    # ``id`` and ``label`` are caller presentation choices, so they must not
    # fragment the expensive provider-series cache.
    symbol = str(result.get("symbol") or asset.symbol).strip().upper()
    result["id"] = asset.id or f"{asset.kind}:{symbol}"
    result["label"] = asset.label or result.get("label") or symbol
    result["cache_status"] = cache_status
    result["stale"] = stale
    result["refresh_pending"] = refresh_pending
    return result


def _market_comparison_history_payload(request: MarketComparisonHistoryRequest) -> Dict[str, Any]:
    handlers = {
        "fund": _fund_comparison_history,
        "stock": _stock_comparison_history,
        "index": _index_comparison_history,
        "fx": _fx_comparison_history,
    }
    return {
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "assets": [
            _comparison_history_cached_asset(
                asset,
                start_date=request.start_date,
                end_date=request.end_date,
                handler=handlers[asset.kind],
            )
            for asset in request.assets
        ],
        "source": "mixed",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _market_indices_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        try:
            with ThreadPoolExecutor(max_workers=min(8, len(_MARKET_INDEX_ORDER))) as pool:
                rows = list(pool.map(_market_index_row, _MARKET_INDEX_ORDER))
        except Exception:
            rows = [_market_index_row(index_code) for index_code in _MARKET_INDEX_ORDER]
        return {
            "rows": rows,
            "source": "yahoo_finance_chart",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key="api:market:indices:v2",
        factory=build,
        fresh_ttl_seconds=_MARKET_INDICES_CACHE_TTL,
        stale_ttl_seconds=_MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_MARKET_INDICES_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {
            "rows": [],
            "source": "yahoo_finance_chart",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "error": "market_indices_refresh_pending",
        }
    return _with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _index_intraday_payload(index_code: str) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    now = time.time()
    cached = _MARKET_INDEX_INTRADAY_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_INDEX_INTRADAY_CACHE_TTL:
        return dict(cached.get("data") or {})

    meta = _MARKET_INDEX_META[normalized]
    for yahoo_symbol in meta["yahoo_candidates"]:
        chart = _fetch_yahoo_chart_raw(yahoo_symbol, interval="5m", range_="1d")
        if chart.get("ok") and chart.get("points"):
            points = [
                {
                    "time": point["time"],
                    "open": point.get("open"),
                    "high": point.get("high"),
                    "low": point.get("low"),
                    "close": point["close"],
                }
                for point in chart.get("points", [])
                if point.get("time") and isinstance(point.get("close"), (int, float))
            ]
            highs = [
                point.get("high")
                for point in chart.get("points", [])
                if isinstance(point.get("high"), (int, float))
            ]
            lows = [
                point.get("low")
                for point in chart.get("points", [])
                if isinstance(point.get("low"), (int, float))
            ]
            meta_payload = chart.get("meta") or {}
            payload = {
                "line_points": points,
                "high": meta_payload.get("regularMarketDayHigh") or (max(highs) if highs else None),
                "low": meta_payload.get("regularMarketDayLow") or (min(lows) if lows else None),
                "prev_close": meta_payload.get("chartPreviousClose") or meta_payload.get("previousClose"),
                "yahoo_symbol": yahoo_symbol,
            }
            _MARKET_INDEX_INTRADAY_CACHE[normalized] = {"_ts": now, "data": payload}
            return dict(payload)

    fallback = {"line_points": [], "high": None, "low": None, "prev_close": None, "yahoo_symbol": None}
    _MARKET_INDEX_INTRADAY_CACHE[normalized] = {"_ts": now, "data": fallback}
    return dict(fallback)


def _latest_share_count_from_kap_cache(symbol: str) -> Optional[float]:
    metadata = _load_cached_kap_market_metadata(CONFIG.paths.processed_dir / "kap_cache", symbol)
    return _positive_float(metadata.get("shares_outstanding"))


def _index_weight_inputs_for_symbol(symbol: str) -> Dict[str, Optional[float]]:
    basic_summary = _fetch_isyatirim_basic_summary_map().get(str(symbol or "").strip().upper(), {})
    shares = _latest_share_count_from_kap_cache(symbol)
    if shares is None:
        shares = _positive_float(basic_summary.get("shares_outstanding"))
    fdpo = _positive_float(basic_summary.get("fdpo"))
    return {
        "shares_outstanding": shares,
        "fdpo": fdpo,
        "weight_coefficient": 1.0 if fdpo is not None else None,
    }


def _apply_index_weight_formula(
    rows: List[Dict[str, Any]],
    *,
    index_level: Any,
) -> tuple[List[Dict[str, Any]], str]:
    enriched: List[Dict[str, Any]] = []
    free_float_values: List[float] = []
    for row in rows:
        price = row.get("price")
        shares = row.get("shares_outstanding")
        fdpo = row.get("fdpo")
        coefficient = row.get("weight_coefficient")
        market_cap = row.get("market_cap")
        free_float_market_value = None
        try:
            if price is not None and shares is not None and fdpo is not None and coefficient is not None:
                free_float_market_value = float(price) * float(shares) * float(fdpo) * float(coefficient)
            elif market_cap is not None and fdpo is not None and coefficient is not None:
                free_float_market_value = float(market_cap) * float(fdpo) * float(coefficient)
        except (TypeError, ValueError):
            free_float_market_value = None
        enriched_row = {
            **row,
            "free_float_market_value": free_float_market_value if free_float_market_value and free_float_market_value > 0 else None,
            "weight_pct": None,
            "point_effect": None,
        }
        enriched.append(enriched_row)
        if enriched_row["free_float_market_value"] is not None:
            free_float_values.append(float(enriched_row["free_float_market_value"]))

    if len(free_float_values) != len(enriched) or not free_float_values:
        return [
            {
                **row,
                "free_float_market_value": None,
                "weight_pct": None,
                "point_effect": None,
            }
            for row in enriched
        ], "unavailable"

    total = sum(free_float_values)
    if total <= 0:
        return enriched, "unavailable"

    try:
        level = float(index_level)
    except (TypeError, ValueError):
        level = 0.0

    calculated: List[Dict[str, Any]] = []
    for row in enriched:
        ffmv = float(row["free_float_market_value"])
        weight_pct = (ffmv / total) * 100.0
        change_pct = row.get("change_pct")
        point_effect = None
        try:
            if change_pct is not None and level > 0:
                point_effect = level * (weight_pct / 100.0) * (float(change_pct) / 100.0)
        except (TypeError, ValueError):
            point_effect = None
        calculated.append(
            {
                **row,
                "weight_pct": round(weight_pct, 4),
                "point_effect": round(point_effect, 2) if point_effect is not None else None,
            }
        )
    calculated.sort(
        key=lambda item: (
            item.get("point_effect") is None,
            -abs(float(item.get("point_effect") or 0.0)),
            str(item.get("symbol") or ""),
        )
    )
    return calculated, "available"


def _market_index_constituent_stock_rows(index_code: str) -> List[Dict[str, Any]]:
    normalized = _normalize_market_index(index_code)
    if normalized in _MARKET_STOCK_INDEXES:
        return list(_market_stocks_payload(index_name=normalized).get("rows", []))

    from app.kap_service import get_bist_index_universe

    try:
        universe = get_bist_index_universe(normalized)
        symbols = [
            str(symbol or "").strip().upper()
            for symbol in universe.get("symbols", [])
            if str(symbol or "").strip()
        ]
    except Exception:
        symbols = []
    if not symbols:
        return []

    price_map = _fetch_market_price_map(symbols, index_name="XUTUM")
    basic_summary_map = _fetch_isyatirim_basic_summary_map()
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        quote = price_map.get(symbol, {})
        cached_meta = _load_cached_kap_market_metadata(cache_dir, symbol)
        basic_summary = basic_summary_map.get(symbol)
        rows.append(
            {
                "company": symbol,
                "price": quote.get("price"),
                "price_currency": quote.get("currency"),
                "change_pct": quote.get("change_pct"),
                "volume": quote.get("volume"),
                "market_cap": _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary),
                **_empty_logo_payload(),
            }
        )
    return rows


def _index_constituents(index_code: str, *, index_level: Any) -> tuple[List[Dict[str, Any]], str]:
    normalized = _normalize_market_index(index_code)
    rows: List[Dict[str, Any]] = []
    for stock in _market_index_constituent_stock_rows(normalized):
        symbol = str(stock.get("company") or "").strip().upper()
        if not symbol:
            continue
        weight_inputs = _index_weight_inputs_for_symbol(symbol)
        rows.append(
            {
                "symbol": symbol,
                "price": stock.get("price"),
                "price_currency": stock.get("price_currency"),
                "change_pct": stock.get("change_pct"),
                "volume": stock.get("volume"),
                "market_cap": stock.get("market_cap"),
                "logo_url": stock.get("logo_url"),
                "logo_source": stock.get("logo_source"),
                "shares_outstanding": weight_inputs.get("shares_outstanding"),
                "fdpo": weight_inputs.get("fdpo"),
                "weight_coefficient": weight_inputs.get("weight_coefficient"),
            }
        )
    weighted_rows, weight_status = _apply_index_weight_formula(rows, index_level=index_level)
    if weight_status != "available":
        weighted_rows.sort(
            key=lambda item: (
                item.get("change_pct") is None,
                -abs(float(item.get("change_pct") or 0.0)),
                str(item.get("symbol") or ""),
            )
        )
    return weighted_rows, weight_status


def _market_index_detail_payload(index_code: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)

    def build() -> Dict[str, Any]:
        quote = _fetch_index_quote(normalized)
        row = _market_index_row(normalized, quote=quote)
        intraday = _index_intraday_payload(normalized)
        constituents, weight_status = _index_constituents(normalized, index_level=row.get("price"))
        return {
            **row,
            "high": row.get("high") if row.get("high") is not None else intraday.get("high"),
            "low": row.get("low") if row.get("low") is not None else intraday.get("low"),
            "prev_close": row.get("prev_close") if row.get("prev_close") is not None else intraday.get("prev_close"),
            "line_points": intraday.get("line_points") or [],
            "constituents": constituents,
            "weight_status": weight_status,
            "weight_note": (
                "Tahmini ağırlık: İş Yatırım halka açıklık oranı ve ağırlık katsayısı 1 varsayımıyla hesaplandı."
                if weight_status == "available"
                else "Ağırlık verisi bulunamadı: pay sayısı, FDPO ve ağırlık katsayısı eksiksiz olmadığı için puan etkisi hesaplanamadı."
            ),
            "source": "yahoo_finance_chart",
            "as_of": row.get("as_of") or datetime.now(timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key=f"api:market:index-detail:{normalized}:v2",
        factory=build,
        fresh_ttl_seconds=_MARKET_INDEX_DETAIL_CACHE_TTL,
        stale_ttl_seconds=_MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_MARKET_INDEX_DETAIL_CACHE,
        local_key=normalized,
        force_revalidate=force_refresh,
    )
    if payload is None:
        raise HTTPException(status_code=503, detail="Endeks verisi yenileniyor. Lütfen kısa süre sonra tekrar deneyin.")
    return _with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _market_commodities_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        def _one(entry: tuple[str, str, str, Optional[str]]) -> Dict[str, Any]:
            symbol, yahoo_symbol, label, forced_currency = entry
            quote = _fetch_yahoo_quote(yahoo_symbol)
            return {
                "symbol": symbol,
                "label": label,
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price") if quote.get("ok") else None,
                "prev_close": quote.get("prev_close") if quote.get("ok") else None,
                "change": quote.get("change") if quote.get("ok") else None,
                "change_pct": quote.get("change_pct") if quote.get("ok") else None,
                "currency": forced_currency or quote.get("currency") or "USD",
                "market_state": quote.get("market_state") if quote.get("ok") else "",
                "as_of": quote.get("as_of") if quote.get("ok") else None,
                "error": None if quote.get("ok") else quote.get("error"),
                "logo_url": None,
                "logo_source": None,
            }

        items: List[Dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                for row in pool.map(_one, _COMMODITY_MAP):
                    items.append(row)
        except Exception:
            for entry in _COMMODITY_MAP:
                items.append(_one(entry))
        return {
            "items": items,
            "source": "yahoo_finance_chart",
            "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key="api:market:commodities:v2",
        factory=build,
        fresh_ttl_seconds=_COMMODITY_CACHE_TTL,
        stale_ttl_seconds=_MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_COMMODITY_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {
            "items": [],
            "source": "yahoo_finance_chart",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "error": "market_commodities_refresh_pending",
        }
    return _with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


@app.get("/market/commodities")
def market_commodities(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_commodities_payload(force_refresh=refresh)


# ── FX (Döviz) ────────────────────────────────────────────
_FX_CACHE: Dict[str, Any] = {}
_FX_CACHE_TTL = 3
_FX_RETURN_CACHE: Dict[str, Any] = {}
_FX_RETURN_CACHE_TTL = 15 * 60

_FX_DIRECT_MAP: List[tuple[str, List[str], str]] = [
    ("USD/TRY", ["USDTRY=X"], "Amerikan Doları / TL"),
    ("EUR/TRY", ["EURTRY=X"], "Euro / TL"),
    ("GBP/TRY", ["GBPTRY=X"], "İngiliz Sterlini / TL"),
    ("CHF/TRY", ["CHFTRY=X"], "İsviçre Frangı / TL"),
    ("AUD/TRY", ["AUDTRY=X"], "Avustralya Doları / TL"),
    ("CAD/TRY", ["CADTRY=X"], "Kanada Doları / TL"),
    ("JPY/TRY", ["JPYTRY=X"], "Japon Yeni / TL"),
    ("EUR/USD", ["EURUSD=X"], "Euro / Dolar"),
    ("GBP/USD", ["GBPUSD=X"], "Sterlin / Dolar"),
    ("USD/JPY", ["USDJPY=X", "JPY=X"], "Dolar / Japon Yeni"),
    ("EUR/JPY", ["EURJPY=X"], "Euro / Japon Yeni"),
    ("GBP/JPY", ["GBPJPY=X"], "Sterlin / Japon Yeni"),
    ("USD/CNY", ["USDCNY=X", "CNY=X"], "Dolar / Çin Yuanı"),
    ("EUR/CNY", ["EURCNY=X"], "Euro / Çin Yuanı"),
    ("GBP/CNY", ["GBPCNY=X"], "Sterlin / Çin Yuanı"),
    ("CNY/JPY", ["CNYJPY=X"], "Çin Yuanı / Japon Yeni"),
    ("CHF/JPY", ["CHFJPY=X"], "İsviçre Frangı / Japon Yeni"),
    ("DXY", ["DX-Y.NYB"], "Dolar Endeksi"),
]

_FX_DERIVED_MAP: List[tuple[str, str, str, str]] = [
    ("CNY/TRY", "Çin Yuanı / TL", "USD/TRY", "USD/CNY"),
]

_FX_ORDER: List[str] = [
    "USD/TRY",
    "EUR/TRY",
    "GBP/TRY",
    "CHF/TRY",
    "AUD/TRY",
    "CAD/TRY",
    "JPY/TRY",
    "CNY/TRY",
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "EUR/JPY",
    "GBP/JPY",
    "USD/CNY",
    "EUR/CNY",
    "GBP/CNY",
    "CNY/JPY",
    "CHF/JPY",
    "DXY",
]


def _fx_quote_currency(symbol: str) -> str:
    if "/" not in symbol:
        return ""
    return str(symbol or "").rsplit("/", 1)[-1].strip().upper()


def _fetch_fx_return_bases(yahoo_symbol: str) -> Dict[str, Any]:
    normalized = str(yahoo_symbol or "").strip()
    if not normalized:
        return {}
    now = time.time()
    cached = _FX_RETURN_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _FX_RETURN_CACHE_TTL:
        return dict(cached.get("data") or {})
    shared_key = f"api:market:fx-return-bases:{normalized}:v1"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        _FX_RETURN_CACHE[normalized] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached)

    chart = _fetch_yahoo_chart_raw(normalized, interval="1d", range_="1y")
    if not chart.get("ok"):
        _FX_RETURN_CACHE[normalized] = {"_ts": now, "data": {}}
        _shared_cache_set(shared_key, {}, ttl_seconds=_FX_RETURN_CACHE_TTL)
        return {}

    points = [
        (datetime.fromisoformat(str(point["time"])), float(point["close"]))
        for point in chart.get("points", [])
        if point.get("time") and isinstance(point.get("close"), (int, float))
    ]
    if not points:
        _FX_RETURN_CACHE[normalized] = {"_ts": now, "data": {}}
        _shared_cache_set(shared_key, {}, ttl_seconds=_FX_RETURN_CACHE_TTL)
        return {}

    points.sort(key=lambda item: item[0])
    latest_dt, latest_close = points[-1]
    year_start = datetime(latest_dt.year, 1, 1, tzinfo=timezone.utc)
    data = {
        "base_1w": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=7)),
        "base_1m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=30)),
        "base_3m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=91)),
        "base_6m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=182)),
        "base_ytd": _pick_series_value_at_or_after(points, year_start),
        "base_1y": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=365)),
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
    }
    _FX_RETURN_CACHE[normalized] = {"_ts": now, "data": data}
    _shared_cache_set(shared_key, data, ttl_seconds=_FX_RETURN_CACHE_TTL)
    return dict(data)


def _fx_returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Optional[float]]:
    current_for_returns = current_price if current_price is not None else return_bases.get("latest_close")
    return {
        response_field: _return_pct(current_for_returns, return_bases.get(base_field))
        for response_field, base_field in _RETURN_BASE_FIELDS
    }


def _fx_item_from_quote(symbol: str, yahoo_symbol: str, label: str, quote: Dict[str, Any], return_bases: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    period_returns = _fx_returns_from_bases(quote.get("price"), return_bases or {}) if quote.get("ok") else {}
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": yahoo_symbol,
        "price": quote.get("price") if quote.get("ok") else None,
        "prev_close": quote.get("prev_close") if quote.get("ok") else None,
        "change": quote.get("change") if quote.get("ok") else None,
        "change_pct": quote.get("change_pct") if quote.get("ok") else None,
        "currency": _fx_quote_currency(symbol),
        "market_state": quote.get("market_state") if quote.get("ok") else "",
        "as_of": quote.get("as_of") if quote.get("ok") else None,
        "error": None if quote.get("ok") else quote.get("error"),
        "logo_url": None,
        "logo_source": None,
        **period_returns,
    }


def _fx_direct_item(entry: tuple[str, List[str], str]) -> Dict[str, Any]:
    symbol, yahoo_candidates, label = entry
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        quote = _fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            return_bases = _fetch_fx_return_bases(yahoo_symbol) if symbol.endswith("/TRY") else None
            return _fx_item_from_quote(symbol, yahoo_symbol, label, quote, return_bases)
        errors.append(str(quote.get("error") or "quote_unavailable"))

    return _fx_item_from_quote(
        symbol,
        yahoo_candidates[0] if yahoo_candidates else "",
        label,
        {"ok": False, "error": "; ".join(errors[:3]) if errors else "quote_unavailable"},
    )


def _positive_number(raw: Any) -> Optional[float]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _fx_derived_item(
    symbol: str,
    label: str,
    numerator_symbol: str,
    denominator_symbol: str,
    items_by_symbol: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    numerator = items_by_symbol.get(numerator_symbol, {})
    denominator = items_by_symbol.get(denominator_symbol, {})

    numerator_price = _positive_number(numerator.get("price"))
    denominator_price = _positive_number(denominator.get("price"))
    numerator_prev = _positive_number(numerator.get("prev_close"))
    denominator_prev = _positive_number(denominator.get("prev_close"))

    price = numerator_price / denominator_price if numerator_price is not None and denominator_price is not None else None
    prev_close = numerator_prev / denominator_prev if numerator_prev is not None and denominator_prev is not None else None
    change = None
    change_pct = None
    if price is not None and prev_close is not None and prev_close > 0:
        change = round(price - prev_close, 6)
        change_pct = round((change / prev_close) * 100, 4)

    source_symbol = "/".join(
        item
        for item in (
            str(numerator.get("yahoo_symbol") or numerator_symbol),
            str(denominator.get("yahoo_symbol") or denominator_symbol),
        )
        if item
    )
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": source_symbol,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": _fx_quote_currency(symbol),
        "market_state": numerator.get("market_state") or denominator.get("market_state") or "",
        "as_of": numerator.get("as_of") or denominator.get("as_of"),
        "error": None if price is not None else "derived_quote_unavailable",
        "logo_url": None,
        "logo_source": None,
        "return_1w_pct": None,
        "return_1m_pct": None,
        "return_3m_pct": None,
        "return_6m_pct": None,
        "return_ytd_pct": None,
        "return_1y_pct": None,
    }


def _market_fx_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        items_by_symbol: Dict[str, Dict[str, Any]] = {}
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for row in pool.map(_fx_direct_item, _FX_DIRECT_MAP):
                    items_by_symbol[str(row.get("symbol") or "")] = row
        except Exception:
            for entry in _FX_DIRECT_MAP:
                row = _fx_direct_item(entry)
                items_by_symbol[str(row.get("symbol") or "")] = row

        for symbol, label, numerator_symbol, denominator_symbol in _FX_DERIVED_MAP:
            items_by_symbol[symbol] = _fx_derived_item(
                symbol,
                label,
                numerator_symbol,
                denominator_symbol,
                items_by_symbol,
            )

        items = [
            items_by_symbol[symbol]
            for symbol in _FX_ORDER
            if symbol in items_by_symbol
        ]
        return {
            "items": items,
            "source": "yahoo_finance_chart",
            "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key="api:market:fx:v2",
        factory=build,
        fresh_ttl_seconds=_FX_CACHE_TTL,
        stale_ttl_seconds=_MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_FX_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {
            "items": [],
            "source": "yahoo_finance_chart",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "error": "market_fx_refresh_pending",
        }
    return _with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


@app.get("/market/fx")
def market_fx(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_fx_payload(force_refresh=refresh)


# ── Market watch strip (single endpoint for Markets page) ────────────────
_WATCH_CACHE: Dict[str, Any] = {}
_WATCH_CACHE_TTL = 3
_WATCH_GLOBAL_CACHE: Dict[str, Any] = {}
_WATCH_GLOBAL_CACHE_TTL = 60
_WATCH_DELAY_NOTE = "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk)."

_WATCH_INDEX_CANDIDATES: List[tuple[str, str, List[str]]] = [
    ("XUTUM", "BIST Tüm", ["XUTUM.IS", "^XUTUM", "XUTUM"]),
    ("XU100", "BIST 100", ["XU100.IS", "^XU100", "XU100"]),
    ("XU030", "BIST 30", ["XU030.IS", "^XU030", "XU030"]),
]

_WATCH_GLOBAL_INDEX_CANDIDATES: List[tuple[str, str, List[str]]] = [
    ("SP500", "S&P 500", ["^GSPC", "SPY"]),
    ("NASDAQ", "Nasdaq", ["^IXIC", "QQQ"]),
    ("DOW", "Dow Jones", ["^DJI", "DIA"]),
    ("DAX", "DAX", ["^GDAXI", "DAX"]),
    ("FTSE", "FTSE 100", ["^FTSE", "ISF.L"]),
    ("NIKKEI", "Nikkei 225", ["^N225", "1321.T"]),
    ("HANGSENG", "Hang Seng", ["^HSI", "2800.HK"]),
    ("CAC40", "CAC 40", ["^FCHI", "CAC.PA"]),
]

_WATCH_FX_SYMBOLS: List[str] = ["USD/TRY", "EUR/TRY"]
_WATCH_FX_LABELS: Dict[str, str] = {
    "USD/TRY": "Amerikan Doları",
    "EUR/TRY": "Euro",
}

_WATCH_COMMODITY_SYMBOLS: List[str] = ["BRENT", "ALTIN", "GUMUS", "DOGALGAZ"]
_WATCH_COMMODITY_LABELS: Dict[str, str] = {
    "BRENT": "Brent Petrol",
    "ALTIN": "Altın (Ons)",
    "GUMUS": "Gümüş (Ons)",
    "DOGALGAZ": "Doğal Gaz",
}
_WATCH_RESPONSE_CACHE_KEY = "api:market:watch:v1"


def _empty_watch_item(
    symbol: str,
    label: str,
    *,
    currency: str = "",
    error: Optional[str] = None,
    yahoo_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": yahoo_symbol,
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "currency": currency,
        "market_state": "",
        "as_of": None,
        "error": error,
        "logo_url": None,
        "logo_source": None,
    }


def _normalize_watch_item(item: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(item.get("symbol") or "").strip()
    label = str(item.get("label") or symbol).strip() or symbol
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": item.get("yahoo_symbol"),
        "price": item.get("price"),
        "prev_close": item.get("prev_close"),
        "change": item.get("change"),
        "change_pct": item.get("change_pct"),
        "currency": item.get("currency") or "",
        "market_state": item.get("market_state") or "",
        "as_of": item.get("as_of"),
        "error": item.get("error"),
        "logo_url": item.get("logo_url"),
        "logo_source": item.get("logo_source"),
    }


def _pick_watch_items(
    items: List[Dict[str, Any]],
    symbols: List[str],
    fallback_labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    mapped = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in items
        if str(row.get("symbol") or "").strip()
    }
    selected: List[Dict[str, Any]] = []
    for symbol in symbols:
        row = mapped.get(symbol.upper())
        if row:
            selected.append(_normalize_watch_item(row))
            continue
        selected.append(
            _empty_watch_item(
                symbol=symbol,
                label=fallback_labels.get(symbol, symbol),
                error="instrument_not_found",
            )
        )
    return selected


def _watch_index_item(
    symbol: str,
    label: str,
    yahoo_candidates: List[str],
    *,
    fallback_currency: str = "TRY",
) -> Dict[str, Any]:
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        quote = _fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            return {
                "symbol": symbol,
                "label": label,
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price"),
                "prev_close": quote.get("prev_close"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "currency": quote.get("currency") or fallback_currency,
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
                "logo_url": None,
                "logo_source": None,
            }
        err = str(quote.get("error") or "quote_unavailable")
        errors.append(f"{yahoo_symbol}:{err}")

    return _empty_watch_item(
        symbol=symbol,
        label=label,
        currency=fallback_currency,
        error="; ".join(errors[:3]) if errors else "quote_unavailable",
    )


def _market_watch_global_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        def _one(entry: tuple[str, str, List[str]]) -> Dict[str, Any]:
            symbol, label, yahoo_candidates = entry
            return _watch_index_item(
                symbol=symbol,
                label=label,
                yahoo_candidates=yahoo_candidates,
                fallback_currency="",
            )

        try:
            with ThreadPoolExecutor(max_workers=min(8, len(_WATCH_GLOBAL_INDEX_CANDIDATES))) as pool:
                items = list(pool.map(_one, _WATCH_GLOBAL_INDEX_CANDIDATES))
        except Exception:
            items = [_one(entry) for entry in _WATCH_GLOBAL_INDEX_CANDIDATES]
        return {
            "items": items,
            "source": "yahoo_finance_chart",
            "delay_note": _WATCH_DELAY_NOTE,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key="api:market:watch-global:v2",
        factory=build,
        fresh_ttl_seconds=_WATCH_GLOBAL_CACHE_TTL,
        stale_ttl_seconds=max(_MARKET_SWR_STALE_TTL_SECONDS, _WATCH_GLOBAL_CACHE_TTL * 2),
        local_cache=_WATCH_GLOBAL_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {"items": [], "source": "yahoo_finance_chart", "as_of": datetime.now(timezone.utc).isoformat()}
    return _with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _market_watch_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    def build() -> Dict[str, Any]:
        fx_payload = _market_fx_payload()
        commodity_payload = _market_commodities_payload()
        indices = [
            _watch_index_item(symbol=symbol, label=label, yahoo_candidates=yahoo_candidates)
            for symbol, label, yahoo_candidates in _WATCH_INDEX_CANDIDATES
        ]
        fx_items = _pick_watch_items(
            items=list(fx_payload.get("items") or []),
            symbols=_WATCH_FX_SYMBOLS,
            fallback_labels=_WATCH_FX_LABELS,
        )
        commodity_items = _pick_watch_items(
            items=list(commodity_payload.get("items") or []),
            symbols=_WATCH_COMMODITY_SYMBOLS,
            fallback_labels=_WATCH_COMMODITY_LABELS,
        )
        return {
            "sections": {
                "indices": indices,
                "fx": fx_items,
                "commodities": commodity_items,
            },
            "source": "yahoo_finance_chart",
            "delay_note": _WATCH_DELAY_NOTE,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _shared_swr_payload(
        cache_key="api:market:watch:v2",
        factory=build,
        fresh_ttl_seconds=_WATCH_CACHE_TTL,
        stale_ttl_seconds=_MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_WATCH_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {"sections": {}, "source": "yahoo_finance_chart", "as_of": datetime.now(timezone.utc).isoformat()}
    return _with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


@app.get("/market/watch")
def market_watch(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_watch_payload(force_refresh=refresh)


@app.get("/market/watch/global")
def market_watch_global(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_watch_global_payload(force_refresh=refresh)
