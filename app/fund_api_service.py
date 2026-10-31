from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional

from app.api_runtime import *
from app.market_service import *

_FUND_COLLECTOR_TASK: Optional[asyncio.Task[None]] = None

_FUND_YIELD_SUMMARY_CACHE_TTL = int(os.getenv("RAGFIN_FUND_YIELD_SUMMARY_CACHE_TTL_SECONDS", "900"))

_FUND_HOLDINGS_RESPONSE_CACHE_TTL = int(os.getenv("RAGFIN_FUND_HOLDINGS_RESPONSE_CACHE_TTL_SECONDS", "60"))

_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT = float(os.getenv("RAGFIN_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT_SECONDS", "120"))

_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION = 13

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

_FUND_REFRESH_AUTO_COOLDOWN_SECONDS = int(
    os.getenv("RAGFIN_FUND_REFRESH_AUTO_COOLDOWN_SECONDS", str(30 * 60))
)

_FUND_REFRESH_AUTO_KEY_PREFIX = "api:funds-refresh:auto"

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

_FUND_HISTORY_KEY_VERSION = 7

# This cache is keyed by catalog and price revisions, which are updated in the
# same transaction as their source data. Keeping popular list payloads around
# longer lets the 5-minute edge revalidation reuse memory instead of rereading
# the full catalog from Supabase on every refresh.
_FUND_LISTING_CACHE_TTL_SECONDS = 600.0
_FUND_LISTING_CACHE_MAX_ENTRIES = 8
_FUND_LISTING_CACHE_MAX_BYTES = 8 * 1024 * 1024
_FUND_LISTING_CACHE_LOCK = threading.RLock()
_FUND_LISTING_CACHE: OrderedDict[str, tuple[float, int, Dict[str, Any]]] = OrderedDict()
_FUND_LISTING_CACHE_BYTES = 0
_FUND_LISTING_IN_FLIGHT: Dict[str, threading.Event] = {}

_ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS = int(
    os.getenv("RAGFIN_ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS", "370")
)

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


def _fund_listing_cache_get_locked(key: str) -> Optional[Dict[str, Any]]:
    global _FUND_LISTING_CACHE_BYTES
    entry = _FUND_LISTING_CACHE.get(key)
    if entry is None:
        return None
    expires_at, weight, payload = entry
    if expires_at <= time.monotonic():
        _FUND_LISTING_CACHE.pop(key, None)
        _FUND_LISTING_CACHE_BYTES = max(0, _FUND_LISTING_CACHE_BYTES - weight)
        return None
    _FUND_LISTING_CACHE.move_to_end(key)
    return copy.deepcopy(payload)


def _fund_listing_cache_store(key: str, payload: Dict[str, Any]) -> int:
    global _FUND_LISTING_CACHE_BYTES
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    weight = len(encoded)
    if weight > _FUND_LISTING_CACHE_MAX_BYTES:
        return weight
    with _FUND_LISTING_CACHE_LOCK:
        previous = _FUND_LISTING_CACHE.pop(key, None)
        if previous:
            _FUND_LISTING_CACHE_BYTES = max(0, _FUND_LISTING_CACHE_BYTES - previous[1])
        _FUND_LISTING_CACHE[key] = (
            time.monotonic() + _FUND_LISTING_CACHE_TTL_SECONDS,
            weight,
            copy.deepcopy(payload),
        )
        _FUND_LISTING_CACHE_BYTES += weight
        while (
            len(_FUND_LISTING_CACHE) > _FUND_LISTING_CACHE_MAX_ENTRIES
            or _FUND_LISTING_CACHE_BYTES > _FUND_LISTING_CACHE_MAX_BYTES
        ):
            _old_key, (_expiry, old_weight, _payload) = _FUND_LISTING_CACHE.popitem(last=False)
            _FUND_LISTING_CACHE_BYTES = max(0, _FUND_LISTING_CACHE_BYTES - old_weight)
    return weight


def _fund_listing_cacheable(payload: Dict[str, Any]) -> bool:
    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    return (
        str(payload.get("status") or "").strip().lower() == "ok"
        and not payload.get("stale")
        and not payload.get("degraded")
        and not payload.get("refresh_pending")
        and not metadata.get("refresh_pending")
        and not list(payload.get("warnings") or [])
    )


def _fund_listing_cache_get_or_build(
    key: str,
    factory: Callable[[], Dict[str, Any]],
    *,
    enabled: bool,
) -> Dict[str, Any]:
    if not enabled:
        logging.getLogger("uvicorn.error").info(
            "fund_listing_cache outcome=bypass key=%s",
            hashlib.sha256(key.encode()).hexdigest()[:12],
        )
        return factory()

    owner = False
    with _FUND_LISTING_CACHE_LOCK:
        cached = _fund_listing_cache_get_locked(key)
        if cached is not None:
            logging.getLogger("uvicorn.error").info("fund_listing_cache outcome=hit key=%s", hashlib.sha256(key.encode()).hexdigest()[:12])
            return cached
        event = _FUND_LISTING_IN_FLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _FUND_LISTING_IN_FLIGHT[key] = event
            owner = True

    if not owner:
        event.wait(timeout=10.0)
        with _FUND_LISTING_CACHE_LOCK:
            cached = _fund_listing_cache_get_locked(key)
        if cached is not None:
            logging.getLogger("uvicorn.error").info("fund_listing_cache outcome=coalesced_hit key=%s", hashlib.sha256(key.encode()).hexdigest()[:12])
            return cached

    logging.getLogger("uvicorn.error").info("fund_listing_cache outcome=miss key=%s", hashlib.sha256(key.encode()).hexdigest()[:12])
    try:
        payload = factory()
        if _fund_listing_cacheable(payload):
            weight = _fund_listing_cache_store(key, payload)
            logging.getLogger("uvicorn.error").info(
                "fund_listing_cache outcome=%s key=%s bytes=%d entries=%d total_bytes=%d",
                "store" if weight <= _FUND_LISTING_CACHE_MAX_BYTES else "oversize",
                hashlib.sha256(key.encode()).hexdigest()[:12],
                weight,
                len(_FUND_LISTING_CACHE),
                _FUND_LISTING_CACHE_BYTES,
            )
        return payload
    finally:
        if owner:
            with _FUND_LISTING_CACHE_LOCK:
                _FUND_LISTING_IN_FLIGHT.pop(key, None)
                event.set()


def _funds_listing_preflight() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load only compact catalogue status and revision counters before ETag validation."""

    from app.fund_service import (
        _mark_snapshot_current_for_market_closure,
        _snapshot_is_current_on_market_closure,
        FUNDS_LIST_MIN_AUM,
        fund_listing_revision,
        load_funds_snapshot,
    )

    snapshot = load_funds_snapshot(
        CONFIG.paths.processed_dir,
        include_rows=False,
        include_price_metadata=False,
    )
    if _snapshot_is_current_on_market_closure(snapshot):
        snapshot = _mark_snapshot_current_for_market_closure(snapshot)
    revision = fund_listing_revision(CONFIG.paths.processed_dir)
    revision["list_min_aum"] = FUNDS_LIST_MIN_AUM
    return snapshot, revision

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
            "cache_invalidation_status": None,
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
        result_meta = dict(result.get("source_metadata") or {})
        snapshot_action = str(
            result_meta.get("snapshot_action") or result.get("snapshot_action") or ""
        ).strip()
        if snapshot_action == "retained_existing" and result_meta.get("truncated_refresh_observed"):
            warnings = list(result.get("warnings") or [])
            _update_fund_refresh_job(
                job_id,
                status="failed",
                finished_at=_fund_refresh_now_iso(),
                resolution_status=resolution_status,
                snapshot_action=snapshot_action,
                error=warnings[-1] if warnings else "TEFAS eksik snapshot döndürdü; mevcut snapshot korundu.",
            )
            return
        from app.fund_service import commit_funds_snapshot

        result = dict(result)
        result["snapshot_generation"] = generation
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
        cache_invalidation_status = _invalidate_fund_response_cache()
        _update_fund_refresh_job(
            job_id,
            status="succeeded",
            finished_at=_fund_refresh_now_iso(),
            as_of=result.get("as_of"),
            row_count=len(rows),
            cache_invalidation_status=cache_invalidation_status,
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
        "cache_invalidation_status": None,
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

def _fund_snapshot_needs_background_refresh(payload: Dict[str, Any]) -> bool:
    """Return whether the canonical catalogue is behind its market target."""

    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    total_count = payload.get("total_count")
    try:
        has_catalogue = int(total_count) > 0
    except (TypeError, ValueError):
        has_catalogue = bool(payload.get("rows"))
    if not has_catalogue:
        return True

    # A previous deployment may have preserved a full snapshot after TEFAS
    # returned only its first page. Its as_of date can still match today's
    # target, so the date/lag checks below alone would never retry the repair.
    # Let the new paginated collector heal that specific legacy state once.
    if metadata.get("truncated_refresh_observed"):
        return True
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if any("snapshot looked truncated" in str(warning).casefold() for warning in warnings):
        return True

    lag = metadata.get("snapshot_as_of_lag_days")
    try:
        return int(lag) > 0
    except (TypeError, ValueError):
        pass

    snapshot_as_of = str(metadata.get("snapshot_as_of") or payload.get("as_of") or "").strip()
    target_date = str(metadata.get("snapshot_target_date") or "").strip()
    return bool(snapshot_as_of and target_date and snapshot_as_of < target_date)

def _public_fund_refresh_job(job: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "job_id",
        "status",
        "requested_at",
        "started_at",
        "finished_at",
        "as_of",
        "row_count",
        "error",
        "resolution_status",
        "snapshot_action",
    )
    return {field: job.get(field) for field in fields}

def _maybe_schedule_fund_snapshot_refresh(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Self-heal a missed external schedule without blocking the list request."""

    # A response still being assembled is not evidence of a missing snapshot.
    if payload.get("status") == "pending":
        return payload
    if not _fund_snapshot_needs_background_refresh(payload):
        return payload

    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    target_date = str(metadata.get("snapshot_target_date") or "unknown").strip()
    backend = _get_cache()
    job = _active_fund_refresh_job()
    if job is None:
        cooldown_key = f"{_FUND_REFRESH_AUTO_KEY_PREFIX}:{target_date}"
        cooldown_value = backend.get(cooldown_key)
        if isinstance(cooldown_value, dict):
            cooldown_job_id = str(cooldown_value.get("job_id") or "").strip()
            if cooldown_job_id:
                cooldown_job = _get_fund_refresh_job(cooldown_job_id)
                if cooldown_job is not None and str(cooldown_job.get("status") or "") not in {"queued", "running"}:
                    # A failed or superseded repair must not suppress the next
                    # automatic recovery attempt for the whole cooldown TTL.
                    backend.delete(cooldown_key)
        claimed = backend.set_if_absent(
            cooldown_key,
            {"scheduled_at": _fund_refresh_now_iso()},
            ttl_seconds=max(60, _FUND_REFRESH_AUTO_COOLDOWN_SECONDS),
        )
        if claimed:
            try:
                job = _start_fund_refresh_job(_FUND_REFRESH_MAX_LOOKBACK_DAYS)
                backend.set(
                    cooldown_key,
                    {
                        "scheduled_at": _fund_refresh_now_iso(),
                        "job_id": job.get("job_id"),
                    },
                    ttl_seconds=max(60, _FUND_REFRESH_AUTO_COOLDOWN_SECONDS),
                )
                LOGGER.info(
                    "stale fund catalogue scheduled background refresh: target=%s job_id=%s",
                    target_date,
                    job.get("job_id"),
                )
            except Exception:
                backend.delete(cooldown_key)
                LOGGER.exception("stale fund catalogue background refresh could not be scheduled")

    if job is None:
        return payload

    out = dict(payload)
    out["refresh_pending"] = bool(payload.get("refresh_pending")) or str(job.get("status") or "") in {"queued", "running"}
    out["refresh_job"] = _public_fund_refresh_job(job)
    return out

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

def _funds_listing_payload(
    *,
    q: Optional[str],
    fund_type: Optional[str],
    founder: Optional[str],
    manager: Optional[str],
    risk: Optional[str],
    sort: str,
    order: str,
    include_revision: bool = False,
    source_revision: Optional[Dict[str, Any]] = None,
    cacheable: bool = True,
) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Any]]:
    from app.fund_service import fund_listing_revision, get_funds_payload

    revision = source_revision or fund_listing_revision(CONFIG.paths.processed_dir)
    base_key = _fund_listing_cache_key(
        q=q,
        fund_type=fund_type,
        founder=founder,
        manager=manager,
        risk=risk,
        sort=sort,
        order=order,
    )
    revision_key = hashlib.sha256(
        json.dumps(revision, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    key = f"{base_key}|revision={revision_key}"

    def build() -> Dict[str, Any]:
        return get_funds_payload(
            CONFIG.paths.processed_dir,
            q=q,
            fund_type=fund_type,
            founder=founder,
            manager=manager,
            risk=risk,
            sort=sort,
            order=order,
            # Keep the catalog read fast. A cold/empty snapshot is recovered by
            # the explicit refresh flow so the ordinary list request cannot
            # spend its whole frontend timeout inside TEFAS.
            auto_refresh=False,
        )

    payload = _fund_listing_cache_get_or_build(key, build, enabled=cacheable)
    if include_revision:
        return payload, revision
    return payload


def _funds_search_payload(*, q: str, limit: int = 50) -> Dict[str, Any]:
    from app.fund_service import get_funds_payload

    return get_funds_payload(
        CONFIG.paths.processed_dir,
        q=q,
        sort="fund_code",
        order="asc",
        min_aum=None,
        limit=limit,
        # Search must stay cache/snapshot-only as well; refresh is an explicit
        # background job started by the list page.
        auto_refresh=False,
    )

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
    target_span_days = (target["requested_end"] - target["requested_start"]).days
    history_source = str(metadata.get("history_source_used") or "").strip().lower()
    if target_span_days > 120 and history_source and history_source != "fintables_udf_history":
        # A complete TEFAS cache can otherwise suppress the first Fintables
        # probe forever. Once a job records ``fintables_point_count`` (even
        # zero), the probe has been made and should not loop on every request.
        job_status = str(last_job.get("status") or "").strip().lower() if last_job else ""
        if job_status in {"queued", "running"}:
            return False
        if not last_job or "fintables_point_count" not in last_job:
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

@_background_cached_response(
    key_fn=lambda *, normalized: f"api:fund-yield-summary:{normalized}",
    ttl_seconds=_FUND_YIELD_SUMMARY_CACHE_TTL,
    stale_seconds=24 * 60 * 60,
    pending_fn=lambda *, normalized: {"fund_code": normalized, "status": "pending", "periods": {}},
)
def _fund_yield_summary_payload(*, normalized: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_yield_summary_payload

    return get_fund_yield_summary_payload(normalized, processed_dir=CONFIG.paths.processed_dir)

@_background_cached_response(
    key_fn=lambda *, normalized: f"api:fund-holdings:{normalized}:response:v{_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION}",
    ttl_seconds=15,
    stale_seconds=24 * 60 * 60,
    pending_fn=lambda *, normalized: {"fund_code": normalized, "status": "pending", "positions": []},
)
def _fund_holdings_response_payload(*, normalized: str) -> Dict[str, Any]:
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
    from app.foreign_instruments import is_isin

    if position.get("provider_symbol_status") == "unresolved":
        return ""
    symbol = str(
        position.get("provider_symbol")
        or position.get("logo_symbol")
        or position.get("asset_code")
        or ""
    ).strip().upper()
    return "" if is_isin(symbol) else symbol

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
    from app.foreign_instruments import resolve_foreign_holdings
    from app.fund_service import normalize_fund_code

    normalized_fund = normalize_fund_code(fund_code)
    raw_positions = [dict(position) for position in list(payload.get("positions") or []) if isinstance(position, dict)]
    raw_positions = resolve_foreign_holdings(
        raw_positions, cache_get=_shared_cache_get_dict, cache_set=_shared_cache_set,
    )
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

def _fund_detail_payload(*, normalized: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_detail_payload

    return get_fund_detail_payload(CONFIG.paths.processed_dir, normalized)

def _require_known_fund_code(fund_code: str) -> str:
    """Reject arbitrary cache/job keys before an upstream refresh is queued."""

    from app.fund_service import get_fund_detail_payload, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        get_fund_detail_payload(CONFIG.paths.processed_dir, normalized)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}") from exc
    return normalized

def _finish_fund_cache_invalidation(outcomes: List[bool], *, scope: str) -> str:
    """Summarize best-effort cache invalidation without failing the data refresh."""

    try:
        redis_fallback = bool(_cache_status().get("cache_redis_fallback"))
    except Exception:
        redis_fallback = True
    complete = bool(outcomes) and all(outcomes) and not redis_fallback
    status = "complete" if complete else "deferred"
    if not complete:
        logging.getLogger("uvicorn.error").warning(
            "fund_response_cache_invalidation status=%s scope=%s",
            status,
            scope,
        )
    return status


def _invalidate_fund_response_cache() -> str:
    """Drop cached fund responses and report whether shared invalidation completed."""

    backend = _get_cache()
    outcomes: List[bool] = []
    for prefix in (
        "api:funds:",
        "api:funds-search:",
        "api:fund-performance:",
        "api:fund-detail:",
        "api:fund-yield-summary:",
        "api:fund-holdings:",
    ):
        try:
            result = backend.delete_prefix_with_status(prefix)
            outcomes.append(bool(result.complete))
        except Exception:
            outcomes.append(False)
    try:
        result = backend.delete_with_status("api:funds-categories")
        outcomes.append(bool(result.complete))
    except Exception:
        outcomes.append(False)
    try:
        result = backend.delete_with_status("api:funds-categories:v2")
        outcomes.append(bool(result.complete))
    except Exception:
        outcomes.append(False)
    return _finish_fund_cache_invalidation(outcomes, scope="snapshot")


def _invalidate_single_fund_response_cache(normalized: str) -> str:
    """Drop one fund's cached responses and report shared invalidation status."""

    backend = _get_cache()
    outcomes: List[bool] = []
    for key_or_prefix in (
        f"api:fund-performance:{normalized}:",
        f"api:fund-performance:v{_FUND_HISTORY_KEY_VERSION}:{normalized}:",
        f"api:fund-detail:{normalized}",
        f"api:fund-yield-summary:{normalized}",
        f"api:fund-holdings:{normalized}:",
    ):
        try:
            if key_or_prefix.endswith(":"):
                result = backend.delete_prefix_with_status(key_or_prefix)
                outcomes.append(bool(result.complete))
            else:
                for key in (
                    key_or_prefix,
                    f"{key_or_prefix}:stale",
                    f"{key_or_prefix}:retry",
                ):
                    result = backend.delete_with_status(key)
                    outcomes.append(bool(result.complete))
        except Exception:
            outcomes.append(False)
    return _finish_fund_cache_invalidation(outcomes, scope="single_fund")

__all__ = [name for name in globals() if not name.startswith("__")]
