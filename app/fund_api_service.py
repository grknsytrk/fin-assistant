from __future__ import annotations

import copy, hashlib, json, threading, time, uuid
from collections import OrderedDict
from typing import Callable, Dict, Optional

from app.api_runtime import (
    Any, CONFIG, HTTPException, LOGGER, List, _background_cached_response, _cache_status,
    _cached_response, _shared_cache_get_dict, _shared_cache_set, _truthy_env,
    acquire_refresh_lease, asyncio, concurrent, database_enabled, date, datetime,
    get_refresh_lease, logging, math, os, re, release_refresh_lease, renew_refresh_lease,
    timedelta, timezone,
)
from app.cache import BoundedMemoryCache, get_cache as _get_cache
from app.etag import orjson_dumps
from app.market_service import (
    _FOREIGN_HOLDING_KNOWN_NAMES_BY_ISIN, _FOREIGN_HOLDING_KNOWN_NAMES_BY_PROVIDER,
    _FOREIGN_HOLDING_QUOTE_CACHE, _FOREIGN_HOLDING_QUOTE_CACHE_TTL, _GEFAS_GYF_ALIAS_MAP,
    _GEFAS_GYF_QUOTE_CACHE, _GEFAS_GYF_QUOTE_CACHE_TTL, _INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT,
    _fetch_infoyatirim_stock_page_quote, _fetch_market_price_map, _fetch_yahoo_quote,
    _merge_market_price_fallback, _parse_tr_decimal,
)

_FUND_COLLECTOR_TASK: Optional[asyncio.Task[None]] = None
_FUND_YIELD_SUMMARY_CACHE_TTL = int(os.getenv("RAGFIN_FUND_YIELD_SUMMARY_CACHE_TTL_SECONDS", "900"))
_FUND_HOLDINGS_RESPONSE_CACHE_TTL = int(os.getenv("RAGFIN_FUND_HOLDINGS_RESPONSE_CACHE_TTL_SECONDS", "60"))
_FUND_HOLDINGS_STATIC_CACHE_TTL = int(os.getenv("RAGFIN_FUND_HOLDINGS_STATIC_CACHE_TTL_SECONDS", "3600"))
_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT = float(os.getenv("RAGFIN_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT_SECONDS", "120"))
_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION = 13
_FUND_HOLDINGS_LIVE_RESPONSE_CACHE_TTL = int(os.getenv("RAGFIN_FUND_HOLDINGS_LIVE_RESPONSE_CACHE_TTL_SECONDS", "30"))
_FUND_HOLDINGS_LIVE_RESPONSE_LOCK_TTL = float(os.getenv("RAGFIN_FUND_HOLDINGS_LIVE_RESPONSE_LOCK_TTL_SECONDS", "30"))
_FUND_HOLDINGS_LIVE_RESPONSE_WAIT_TIMEOUT = float(os.getenv("RAGFIN_FUND_HOLDINGS_LIVE_RESPONSE_WAIT_TIMEOUT_SECONDS", "2"))
_FUND_HOLDING_SECTOR_MAP_CACHE_TTL = 6 * 60 * 60
_FUND_REFRESH_JOB_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_REFRESH_JOB_TTL_SECONDS", str(30 * 60)))
_FUND_REFRESH_MAX_LOOKBACK_DAYS = 14
_FUND_REFRESH_RESOURCE_KEY = "funds_snapshot"
_FUND_REFRESH_AUTO_COOLDOWN_SECONDS = int(os.getenv("RAGFIN_FUND_REFRESH_AUTO_COOLDOWN_SECONDS", str(30 * 60)))
_FUND_REFRESH_AUTO_KEY_PREFIX = "api:funds-refresh:auto"
_FUND_REFRESH_LEASE_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_REFRESH_LEASE_TTL_SECONDS", "90"))
_FUND_REFRESH_ACTIVE_KEY = "api:funds-refresh:active"
_FUND_REFRESH_HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("RAGFIN_FUND_REFRESH_HEARTBEAT_INTERVAL_SECONDS", "20"))
_FUND_REFRESH_HEARTBEAT_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_FUND_REFRESH_HEARTBEAT_TIMEOUT_SECONDS", "120"))
_FUND_REFRESH_MAX_RUNTIME_SECONDS = float(os.getenv("RAGFIN_FUND_REFRESH_MAX_RUNTIME_SECONDS", str(15 * 60)))

_FUND_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="fund-snapshot-refresh"
)
_FUND_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="fund-history-backfill"
)
_FUND_ALLOCATION_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_WORKERS", "1"))),
    thread_name_prefix="fund-allocation-history-refresh",
)


def shutdown_fund_executors() -> None:
    """Terminate all fund-related background thread pools."""
    for executor in (_FUND_REFRESH_EXECUTOR, _FUND_HISTORY_EXECUTOR, _FUND_ALLOCATION_HISTORY_EXECUTOR):
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


_FUND_REFRESH_STATE_LOCK = threading.Lock()
_FUND_HISTORY_STATE_LOCK = threading.Lock()
_FUND_HISTORY_JOB_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_HISTORY_JOB_TTL_SECONDS", str(30 * 60)))
_FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS", str(30 * 60)))
_FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS", "120"))
_FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS = float(
    os.getenv("RAGFIN_FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS", "30")
)
_FUND_HISTORY_WARMUP_DAYS = int(os.getenv("RAGFIN_FUNDS_HISTORY_WARMUP_DAYS", "366"))
_FUND_HISTORY_LEASE_TTL_SECONDS = int(os.getenv("RAGFIN_FUND_HISTORY_LEASE_TTL_SECONDS", "120"))
_FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("RAGFIN_FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS", "30"))
_FUND_HISTORY_MAX_PHASES = int(os.getenv("RAGFIN_FUND_HISTORY_MAX_PHASES", "2"))
_FUND_HISTORY_KEY_VERSION = 7

# This cache is keyed by catalog and price revisions, which are updated in the
# same transaction as their source data. Keeping popular list payloads around
# longer lets the 5-minute edge revalidation reuse memory instead of rereading
# the full catalog from Supabase on every refresh.
_FUND_LISTING_CACHE_TTL_SECONDS = float(os.getenv("RAGFIN_FUND_LISTING_CACHE_TTL_SECONDS", "3600"))
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
    *, q: Optional[str], fund_type: Optional[str], founder: Optional[str],
    manager: Optional[str], risk: Optional[str], sort: str, order: str,
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
    encoded = orjson_dumps(payload)
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
    key: str, factory: Callable[[], Dict[str, Any]], *, enabled: bool,
) -> Dict[str, Any]:
    if not enabled:
        logging.getLogger("uvicorn.error").info(
            "fund_listing_cache outcome=bypass key=%s", hashlib.sha256(key.encode()).hexdigest()[:12]
        )
        return factory()

    owner = False
    with _FUND_LISTING_CACHE_LOCK:
        cached = _fund_listing_cache_get_locked(key)
        if cached is not None:
            logging.getLogger("uvicorn.error").info(
                "fund_listing_cache outcome=hit key=%s", hashlib.sha256(key.encode()).hexdigest()[:12]
            )
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
            logging.getLogger("uvicorn.error").info(
                "fund_listing_cache outcome=coalesced_hit key=%s", hashlib.sha256(key.encode()).hexdigest()[:12]
            )
            return cached

    logging.getLogger("uvicorn.error").info(
        "fund_listing_cache outcome=miss key=%s", hashlib.sha256(key.encode()).hexdigest()[:12]
    )
    try:
        payload = factory()
        if _fund_listing_cacheable(payload):
            weight = _fund_listing_cache_store(key, payload)
            logging.getLogger("uvicorn.error").info(
                "fund_listing_cache outcome=%s key=%s bytes=%d entries=%d total_bytes=%d",
                "store" if weight <= _FUND_LISTING_CACHE_MAX_BYTES else "oversize",
                hashlib.sha256(key.encode()).hexdigest()[:12],
                weight, len(_FUND_LISTING_CACHE), _FUND_LISTING_CACHE_BYTES,
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
        _mark_snapshot_current_for_market_closure, _snapshot_is_current_on_market_closure,
        FUNDS_LIST_MIN_AUM, fund_listing_revision, load_funds_snapshot,
    )

    snapshot = load_funds_snapshot(
        CONFIG.paths.processed_dir, include_rows=False, include_price_metadata=False
    )
    if _snapshot_is_current_on_market_closure(snapshot):
        snapshot = _mark_snapshot_current_for_market_closure(snapshot)
    revision = fund_listing_revision(CONFIG.paths.processed_dir)
    revision["list_min_aum"] = FUNDS_LIST_MIN_AUM
    return snapshot, revision


def _funds_listing_payload(
    *, q: Optional[str], fund_type: Optional[str], founder: Optional[str],
    manager: Optional[str], risk: Optional[str], sort: str, order: str,
    include_revision: bool = False, source_revision: Optional[Dict[str, Any]] = None,
    cacheable: bool = True,
) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Any]]:
    from app.fund_service import fund_listing_revision, get_funds_payload

    revision = source_revision or fund_listing_revision(CONFIG.paths.processed_dir)
    base_key = _fund_listing_cache_key(
        q=q, fund_type=fund_type, founder=founder, manager=manager, risk=risk, sort=sort, order=order
    )
    revision_key = hashlib.sha256(orjson_dumps(revision, sort_keys=True)).hexdigest()
    key = f"{base_key}|revision={revision_key}"

    def build() -> Dict[str, Any]:
        return get_funds_payload(
            CONFIG.paths.processed_dir, q=q, fund_type=fund_type, founder=founder,
            manager=manager, risk=risk, sort=sort, order=order, auto_refresh=False,
        )

    payload = _fund_listing_cache_get_or_build(key, build, enabled=cacheable)
    if include_revision:
        return payload, revision
    return payload


_FUNDS_SEARCH_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=128, default_ttl=300.0)
_FUND_DETAIL_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=300.0)
_FUND_ALLOCATIONS_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=600.0)
_FUND_AUTO_REFRESH_PREFLIGHT_L1: BoundedMemoryCache = BoundedMemoryCache(max_entries=32, default_ttl=20.0)
_FOREIGN_IDENTITIES_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=512, default_ttl=24 * 60 * 60)

_FOREIGN_IDENTITIES_BULK_KEY = "api:funds:foreign-identities:v2"
_FOREIGN_IDENTITIES_BULK_TTL = 30 * 24 * 60 * 60
_FOREIGN_QUOTES_BULK_KEY = "api:funds:foreign-quotes:v1"


def _prune_bulk_ttl_dict(
    raw: Dict[str, Any], *, now: Optional[float] = None, max_entries: int = 500,
) -> Dict[str, Any]:
    current = now if now is not None else time.time()
    kept: List[tuple[str, Dict[str, Any]]] = []
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        exp = v.get("exp")
        if not isinstance(exp, (int, float)) or float(exp) <= current:
            continue
        kept.append((k, v))
    if len(kept) > max_entries:
        kept.sort(key=lambda item: float(item[1].get("exp") or 0.0), reverse=True)
        kept = kept[:max_entries]
    return dict(kept)


def _clear_single_fund_l1_caches(normalized: str) -> None:
    for cache_key in [
        k for k in list(_FUND_DETAIL_L1_CACHE.keys()) if isinstance(k, tuple) and len(k) >= 5 and k[4] == normalized
    ]:
        _FUND_DETAIL_L1_CACHE.pop(cache_key, None)
    for cache_key in [
        k for k in list(_FUND_ALLOCATIONS_L1_CACHE.keys()) if isinstance(k, tuple) and len(k) >= 4 and k[3] == normalized
    ]:
        _FUND_ALLOCATIONS_L1_CACHE.pop(cache_key, None)


def _funds_search_payload(*, q: str, limit: int = 50) -> Dict[str, Any]:
    from app.fund_service import fund_listing_revision, get_funds_payload

    try:
        revision = fund_listing_revision(CONFIG.paths.processed_dir)
    except Exception:
        revision = {}
    norm_q = _normalized_cache_text(q, max_length=80)
    rev_token = f"{int(revision.get('catalog_generation') or 0)}:{int(revision.get('price_generation') or 0)}"
    backend = _get_cache()
    l1_key = (id(backend), str(CONFIG.paths.processed_dir), rev_token, norm_q, int(limit))
    l1_hit = _FUNDS_SEARCH_L1_CACHE.get(l1_key)
    if isinstance(l1_hit, dict) and isinstance(l1_hit.get("rows"), list):
        return dict(l1_hit)

    cache_key = f"api:funds-search:v1:q={norm_q}|limit={int(limit)}|rev={rev_token}"
    if not norm_q or not database_enabled():
        cached = backend.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("rows"), list):
            _FUNDS_SEARCH_L1_CACHE[l1_key] = dict(cached)
            return dict(cached)

    payload = get_funds_payload(
        CONFIG.paths.processed_dir, q=q, sort="fund_code", order="asc",
        min_aum=None, limit=limit, auto_refresh=False,
    )
    if _fund_listing_cacheable(payload):
        _FUNDS_SEARCH_L1_CACHE[l1_key] = dict(payload)
        if not norm_q or not database_enabled():
            backend.set(cache_key, payload, ttl_seconds=600)
    return payload


_FUNDS_CATEGORIES_L1_LOCK = threading.Lock()
_FUNDS_CATEGORIES_L1_CACHE: Dict[str, Any] = {}


def _funds_categories_payload() -> Dict[str, Any]:
    from app.fund_service import fund_listing_revision, get_fund_categories_payload

    try:
        revision = fund_listing_revision(CONFIG.paths.processed_dir)
    except Exception:
        revision = {}
    cat_rev = f"{int(revision.get('catalog_generation') or 0)}:{int(revision.get('price_generation') or 0)}"
    now_mono = time.monotonic()
    with _FUNDS_CATEGORIES_L1_LOCK:
        l1 = _FUNDS_CATEGORIES_L1_CACHE.get(cat_rev)
        if l1 and l1.get("expires_at", 0.0) > now_mono and isinstance(l1.get("payload"), dict):
            return dict(l1["payload"])

    cache_key = "api:funds-categories:v2"
    backend = _get_cache()
    cached = backend.get(cache_key)
    if isinstance(cached, dict) and cached.get("fund_types") is not None and cached.get("_rev") == cat_rev:
        out = dict(cached)
        out.pop("_rev", None)
        with _FUNDS_CATEGORIES_L1_LOCK:
            _FUNDS_CATEGORIES_L1_CACHE.clear()
            _FUNDS_CATEGORIES_L1_CACHE[cat_rev] = {"expires_at": now_mono + 300.0, "payload": dict(out)}
        return out

    payload = get_fund_categories_payload(CONFIG.paths.processed_dir)
    if isinstance(payload, dict) and str(payload.get("status") or "").strip().lower() == "ok":
        with _FUNDS_CATEGORIES_L1_LOCK:
            _FUNDS_CATEGORIES_L1_CACHE.clear()
            _FUNDS_CATEGORIES_L1_CACHE[cat_rev] = {"expires_at": now_mono + 300.0, "payload": dict(payload)}
        backend.set(cache_key, {**payload, "_rev": cat_rev}, ttl_seconds=3600)
    return payload


def _fund_performance_payload(
    *, fund_code: str, start_iso: Optional[str], end_iso: Optional[str], fallback: bool, refresh: bool,
) -> Dict[str, Any]:
    from app import fund_service as _fund_service

    get_fund_performance_payload = globals().get(
        "get_fund_performance_payload", _fund_service.get_fund_performance_payload
    )
    normalize_fund_code = globals().get("normalize_fund_code", _fund_service.normalize_fund_code)

    start = date.fromisoformat(start_iso) if start_iso else None
    end = date.fromisoformat(end_iso) if end_iso else None
    normalized = normalize_fund_code(fund_code)
    target = _history_request_range(start, end)
    existing_job = _history_find_existing_job(normalized, target)
    payload = get_fund_performance_payload(
        CONFIG.paths.processed_dir, normalized, start_date=start, end_date=end,
        allow_upstream_fallback=fallback, auto_refresh=False,
    )
    if _history_job_should_schedule(payload, last_job=existing_job, target=target):
        existing_job = _history_start_or_extend_job(
            normalized, start_date=start, end_date=end,
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
    from app import fund_service as _fund_service

    get_fund_yield_summary_payload = globals().get(
        "get_fund_yield_summary_payload", _fund_service.get_fund_yield_summary_payload
    )
    return get_fund_yield_summary_payload(normalized, processed_dir=CONFIG.paths.processed_dir)


@_background_cached_response(
    key_fn=lambda *, normalized: f"api:fund-holdings:{normalized}:response:v{_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION}",
    ttl_seconds=90,
    stale_seconds=24 * 60 * 60,
    pending_fn=lambda *, normalized: {"fund_code": normalized, "status": "pending", "positions": []},
)
def _fund_holdings_response_payload(*, normalized: str) -> Dict[str, Any]:
    payload = _fund_holdings_static_payload(normalized=normalized)
    return _enrich_fund_holdings_with_daily_market_data(payload, normalized)


@_cached_response(
    key_fn=lambda *, normalized: f"api:fund-holdings:{normalized}:v{_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION}",
    ttl_seconds=_FUND_HOLDINGS_STATIC_CACHE_TTL,
    skip_when=lambda *, normalized: not normalized,
    single_flight=True,
    lock_timeout=_FUND_HOLDINGS_RESPONSE_LOCK_TIMEOUT,
)
def _fund_holdings_static_payload(*, normalized: str) -> Dict[str, Any]:
    from app import fund_service as _fund_service

    get_fund_holdings_payload = globals().get(
        "get_fund_holdings_payload", _fund_service.get_fund_holdings_payload
    )
    return get_fund_holdings_payload(CONFIG.paths.processed_dir, normalized)


_FUND_HOLDINGS_LIVE_POSITION_FIELDS = (
    "asset_code", "price", "price_currency", "return_pct", "return_source",
    "return_as_of", "estimated_exposure_value", "estimated_pnl_value",
    "estimated_fund_return_contribution_pct",
)

_FUND_HOLDING_SECTOR_LABELS: Dict[str, str] = {
    "XBANK": "Bankacılık", "XAKUR": "Aracı Kurum", "XBLSM": "Bilişim ve Yazılım",
    "XELKT": "Elektrik", "XFINK": "Finansal Kiralama Faktoring", "XGMYO": "Gayrimenkul",
    "XGIDA": "Gıda İçecek", "XHOLD": "Holding", "XILTM": "İletişim", "XINSA": "İnşaat",
    "XKAGT": "Orman Kağıt Basım", "XKMYA": "Kimya Petrol Plastik", "XMADN": "Madencilik",
    "XMANA": "Metal Ana", "XMESY": "Metal Eşya Makina", "XSGRT": "Sigorta",
    "XSPOR": "Spor", "XTAST": "Taş Toprak", "XTCRT": "Ticaret", "XTEKS": "Tekstil Deri",
    "XTRZM": "Turizm", "XULAS": "Ulaştırma", "XYORT": "Menkul Kıymet Y.O.",
    "XUSIN": "Sınai", "XUHIZ": "Hizmetler", "XUMAL": "Mali", "XUTEK": "Teknoloji",
}

_FUND_HOLDING_SECTOR_PRIORITY = (
    "XBANK", "XAKUR", "XBLSM", "XELKT", "XFINK", "XGMYO", "XGIDA", "XHOLD", "XILTM",
    "XINSA", "XKAGT", "XKMYA", "XMADN", "XMANA", "XMESY", "XSGRT", "XSPOR", "XTAST",
    "XTCRT", "XTEKS", "XTRZM", "XULAS", "XYORT", "XUSIN", "XUHIZ", "XUMAL", "XUTEK",
)

_FUND_HOLDING_SECTOR_MAP_CACHE: Dict[str, Any] = {}
_FUND_SNAPSHOT_ROW_MAP_CACHE: Dict[str, Any] = {}


def _fund_detail_payload(*, normalized: str) -> Dict[str, Any]:
    from app.fund_service import fund_listing_revision, get_fund_detail_payload

    try:
        revision = fund_listing_revision(CONFIG.paths.processed_dir)
    except Exception:
        revision = {}
    detail_rev = f"{int(revision.get('catalog_generation') or 0)}:{int(revision.get('price_generation') or 0)}"
    backend = _get_cache()
    l1_key = (id(backend), id(get_fund_detail_payload), str(CONFIG.paths.processed_dir), detail_rev, normalized)
    l1_hit = _FUND_DETAIL_L1_CACHE.get(l1_key)
    if isinstance(l1_hit, dict) and l1_hit.get("fund_code") == normalized:
        return dict(l1_hit)

    cache_key = f"api:fund-detail:{normalized}"
    cached = backend.get(cache_key)
    if isinstance(cached, dict) and cached.get("fund_code") == normalized and cached.get("_rev") == detail_rev:
        out = dict(cached)
        out.pop("_rev", None)
        _FUND_DETAIL_L1_CACHE[l1_key] = dict(out)
        return out

    payload = get_fund_detail_payload(CONFIG.paths.processed_dir, normalized)
    if isinstance(payload, dict) and str(payload.get("status") or "").strip().lower() not in {"unavailable", "pending"}:
        _FUND_DETAIL_L1_CACHE[l1_key] = dict(payload)
        backend.set(cache_key, {**payload, "_rev": detail_rev}, ttl_seconds=900)
    return payload


def _fund_allocations_cached(normalized: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_allocations_payload

    l1_key = (id(_get_cache()), id(get_fund_allocations_payload), str(CONFIG.paths.processed_dir), normalized, "allocations", 0)
    l1_hit = _FUND_ALLOCATIONS_L1_CACHE.get(l1_key)
    if isinstance(l1_hit, dict):
        return dict(l1_hit)
    payload = get_fund_allocations_payload(CONFIG.paths.processed_dir, normalized)
    if isinstance(payload, dict) and not payload.get("stale") and str(payload.get("status") or "") not in {"pending", "unavailable"}:
        _FUND_ALLOCATIONS_L1_CACHE[l1_key] = dict(payload)
    return payload


def _fund_allocations_history_cached(normalized: str, *, lookback_days: int) -> Dict[str, Any]:
    from app.fund_service import get_fund_allocations_history_payload

    l1_key = (
        id(_get_cache()), id(get_fund_allocations_history_payload),
        str(CONFIG.paths.processed_dir), normalized, "allocations_history", int(lookback_days),
    )
    l1_hit = _FUND_ALLOCATIONS_L1_CACHE.get(l1_key)
    if isinstance(l1_hit, dict):
        return dict(l1_hit)
    payload = get_fund_allocations_history_payload(
        CONFIG.paths.processed_dir, normalized, lookback_days=lookback_days, auto_refresh=False
    )
    needs_refresh = bool(payload.get("stale")) or str(payload.get("status")) == "pending"
    if isinstance(payload, dict) and not needs_refresh:
        _FUND_ALLOCATIONS_L1_CACHE[l1_key] = dict(payload)
    return payload


def _require_known_fund_code(fund_code: str) -> str:
    """Reject arbitrary cache/job keys before an upstream refresh is queued."""
    from app.fund_service import _connect_fund_prices_db, _peek_all_snapshot_rows, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    if not normalized:
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}")
    peeked = _peek_all_snapshot_rows(CONFIG.paths.processed_dir)
    if isinstance(peeked, dict) and isinstance(peeked.get("rows"), list):
        if any(
            normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or "")) == normalized
            for row in peeked["rows"]
            if isinstance(row, dict)
        ):
            return normalized
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}")
    if database_enabled():
        try:
            with _connect_fund_prices_db(CONFIG.paths.processed_dir) as conn:
                found = conn.execute(
                    "SELECT 1 FROM fund_snapshots WHERE fund_code = ? LIMIT 1",
                    (normalized,),
                ).fetchone()
            if found is not None:
                return normalized
            raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}")
        except HTTPException:
            raise
        except Exception:
            pass
    rows_by_code = _fund_snapshot_row_map()
    if normalized not in rows_by_code:
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}")
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
            "fund_response_cache_invalidation status=%s scope=%s", status, scope
        )
    return status


def _invalidate_fund_response_cache() -> str:
    """Drop cached fund responses and report whether shared invalidation completed."""
    global _FUND_LISTING_CACHE_BYTES
    with _FUND_LISTING_CACHE_LOCK:
        _FUND_LISTING_CACHE.clear()
        _FUND_LISTING_CACHE_BYTES = 0
    _FUND_SNAPSHOT_ROW_MAP_CACHE.clear()
    with _FUNDS_CATEGORIES_L1_LOCK:
        _FUNDS_CATEGORIES_L1_CACHE.clear()
    _FUNDS_SEARCH_L1_CACHE.clear()
    _FUND_DETAIL_L1_CACHE.clear()
    _FUND_ALLOCATIONS_L1_CACHE.clear()
    _FUND_AUTO_REFRESH_PREFLIGHT_L1.clear()
    backend = _get_cache()
    outcomes: List[bool] = []
    for prefix in (
        "api:funds:", "api:funds-search:", "api:fund-performance:",
        "api:fund-detail:", "api:fund-yield-summary:", "api:fund-holdings:",
    ):
        try:
            result = backend.delete_prefix_with_status(prefix)
            outcomes.append(bool(result.complete))
        except Exception:
            outcomes.append(False)
    for key in ("api:funds-categories", "api:funds-categories:v2"):
        try:
            result = backend.delete_with_status(key)
            outcomes.append(bool(result.complete))
        except Exception:
            outcomes.append(False)
    return _finish_fund_cache_invalidation(outcomes, scope="snapshot")


def _invalidate_single_fund_response_cache(normalized: str, *, include_holdings: bool = True) -> str:
    """Drop one fund's cached responses and report shared invalidation status."""
    _clear_single_fund_l1_caches(normalized)
    backend = _get_cache()
    outcomes: List[bool] = []
    targets = [
        f"api:fund-performance:{normalized}:",
        f"api:fund-performance:v{_FUND_HISTORY_KEY_VERSION}:{normalized}:",
        f"api:fund-detail:{normalized}",
        f"api:fund-yield-summary:{normalized}",
    ]
    if include_holdings:
        targets.append(f"api:fund-holdings:{normalized}:")
    for key_or_prefix in targets:
        try:
            if key_or_prefix.endswith(":"):
                result = backend.delete_prefix_with_status(key_or_prefix)
                outcomes.append(bool(result.complete))
            else:
                for key in (key_or_prefix, f"{key_or_prefix}:stale", f"{key_or_prefix}:retry"):
                    result = backend.delete_with_status(key)
                    outcomes.append(bool(result.complete))
        except Exception:
            outcomes.append(False)
    return _finish_fund_cache_invalidation(outcomes, scope="single_fund")


from app.funds_api.holdings_enrichment import (
    _api_number, _enrich_fund_holdings_with_daily_market_data, _fetch_gefas_gyf_chart,
    _fetch_gefas_gyf_quote, _foreign_holding_known_name, _foreign_holding_provider_name,
    _foreign_holding_provider_symbol, _foreign_holding_should_use_provider_name,
    _fund_holding_sector_map, _fund_holdings_live_payload, _fund_snapshot_row_map,
    _fund_snapshot_row_map_with_meta, _gefas_chart_date, _gefas_gyf_config, _holding_code,
    _holding_type, _holding_weight_scale_context, _per_position_basis_points_threshold,
    _position_daily_market_fields, _quote_map_for_foreign_holdings,
    _quote_map_for_holding_stocks, _validate_fund_holding_weights, _validated_holding_weight,
)
from app.funds_api.jobs_and_collector import (
    _active_fund_refresh_job, _allocation_history_active_key, _allocation_history_job_get,
    _allocation_history_job_key, _allocation_history_job_public, _allocation_history_job_set,
    _execute_fund_price_collector_cycle, _fund_price_collector_loop, _fund_refresh_job_key,
    _fund_refresh_now_iso, _fund_snapshot_needs_background_refresh, _get_fund_refresh_job,
    _history_active_key, _history_attach_job, _history_find_existing_job, _history_job_covers,
    _history_job_date, _history_job_failed, _history_job_get, _history_job_key,
    _history_job_needs_fintables_probe, _history_job_owned, _history_job_public,
    _history_job_set, _history_job_should_schedule, _history_last_key, _history_merge_target,
    _history_request_range, _history_start_or_extend_job, _maybe_schedule_fund_snapshot_refresh,
    _public_fund_refresh_job, _run_allocation_history_refresh_job, _run_fund_history_job,
    _run_fund_refresh_job, _set_fund_refresh_job, _start_allocation_history_refresh_job,
    _start_fund_price_collector, _start_fund_refresh_job, _stop_fund_price_collector,
    _update_fund_refresh_job,
)

__all__ = [name for name in globals() if not name.startswith("__")]
