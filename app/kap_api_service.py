from __future__ import annotations

from app.api_runtime import (
    Any,
    CONFIG,
    Dict,
    HTMLParser,
    HTTPException,
    LOGGER,
    List,
    Optional,
    Request,
    _MARKET_SWR_LOCK_TTL_SECONDS,
    _MARKET_SWR_STALE_TTL_SECONDS,
    _MARKET_SWR_WAIT_TIMEOUT_SECONDS,
    _cache_status,
    _debug_log,
    _get_cache,
    _get_or_set_single_flight,
    _kap_flow_store,
    _schedule_swr_revalidation,
    _shared_cache_get_dict,
    _shared_cache_set,
    _shared_swr_payload,
    _swr_cache_entry,
    _swr_entry_is_fresh,
    _swr_entry_is_stale,
    _swr_entry_payload,
    _with_market_cache_metadata,
    asyncio,
    bisect,
    concurrent,
    database_enabled,
    date,
    datetime,
    extract_financial_snapshot_from_kap_payload,
    generate_overview_commentary,
    json,
    math,
    os,
    re,
    time,
    timedelta,
    timezone,
    upsert_financial_snapshots,
    uuid,
)
from app.cache import BoundedMemoryCache, cache_reset_epoch
from app.kap_financials import (
    _build_ttm_from_reports,
    _extract_quarter_metric,
    _quarter_sort_key,
)
from app.market_service import (
    _TURKEY_TIMEZONE,
    _as_finite_float,
    _fetch_isyatirim_multiples,
    _fetch_kap_price_payload,
    _fetch_yahoo_chart_period_raw,
    _parse_tr_decimal,
    _round_market_ratio,
    _upsert_stock_reference_from_kap_payload,
)

_KAP_SNAPSHOT_RESPONSE_CACHE_TTL = int(os.getenv("RAGFIN_KAP_SNAPSHOT_RESPONSE_CACHE_TTL_SECONDS", "300"))

_KAP_SNAPSHOT_RESPONSE_STALE_TTL = int(
    os.getenv("RAGFIN_KAP_SNAPSHOT_RESPONSE_STALE_TTL_SECONDS", str(30 * 60))
)

_KAP_SNAPSHOT_RESPONSE_LOCK_TIMEOUT = float(os.getenv("RAGFIN_KAP_SNAPSHOT_RESPONSE_LOCK_TIMEOUT_SECONDS", "120"))

_KAP_FLOW_REFRESH_STALE_AFTER_SECONDS = max(
    60,
    int(os.getenv("RAGFIN_KAP_FLOW_REFRESH_STALE_AFTER_SECONDS", str(15 * 60))),
)

_KAP_FLOW_REFRESH_COOLDOWN_SECONDS = max(
    60,
    int(os.getenv("RAGFIN_KAP_FLOW_REFRESH_COOLDOWN_SECONDS", str(5 * 60))),
)

_KAP_FLOW_REFRESH_WATCHDOG_KEY = "api:kap:flow:refresh-watchdog:v1"

_KAP_FLOW_REFRESH_LIVE_AFTER_SECONDS = max(
    10,
    int(os.getenv("RAGFIN_KAP_FLOW_REFRESH_LIVE_AFTER_SECONDS", "30")),
)

_KAP_FLOW_REFRESH_LIVE_COOLDOWN_SECONDS = max(
    30,
    int(os.getenv("RAGFIN_KAP_FLOW_REFRESH_LIVE_COOLDOWN_SECONDS", "30")),
)

_KAP_FLOW_REFRESH_LIVE_WATCHDOG_KEY = "api:kap:flow:refresh-live-watchdog:v1"

_KAP_FLOW_WATCHDOG_L1_TTL_SECONDS = max(
    1,
    int(os.getenv("RAGFIN_KAP_FLOW_WATCHDOG_L1_TTL_SECONDS", "15")),
)

_KAP_FLOW_WATCHDOG_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(
    max_entries=16,
    default_ttl=_KAP_FLOW_WATCHDOG_L1_TTL_SECONDS,
)

_KAP_FLOW_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="kap-flow-refresh-watchdog",
)


def shutdown_background_refresh_executor() -> None:
    """Stop accepting new KAP refresh jobs during application shutdown."""
    _KAP_FLOW_REFRESH_EXECUTOR.shutdown(wait=False, cancel_futures=True)


_KAP_MARKET_METADATA_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=32)

_KAP_MARKET_METADATA_CACHE_TTL = 6 * 60 * 60

_KAP_COMPANIES_RESPONSE_CACHE: Dict[str, Any] = {}

_KAP_COMPANIES_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_KAP_COMPANIES_RESPONSE_CACHE_TTL_SECONDS", str(60 * 60))
)

_KAP_COMPANIES_RESPONSE_CACHE_KEY = "api:kap:companies:v2"


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
                    "[kap_overview_commentary] client disconnected; cancelling AI request | company=%s",
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


_FLOW_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64)

_FLOW_CACHE_TTL = 180

_FLOW_STALE_SERVE_WINDOW = 15 * 60

_KAP_FLOW_REFRESH_LOCK_TIMEOUT_SECONDS = float(
    os.getenv("RAGFIN_KAP_FLOW_REFRESH_LOCK_TIMEOUT_SECONDS", "0.1")
)

_KAP_SOURCE_TIMEZONE = timezone(timedelta(hours=3))

_KAP_TYPE_LABELS: Dict[str, str] = {
    "ODA": "Özel Durum",
    "ÖDA": "Özel Durum",
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
    "DG": "Diğer Bildirim",
    "DKB": "Diğer Bildirim",
}

_KAP_PUBLIC_LAST_ERROR: Dict[str, Any] = {"message": None, "ts": 0.0, "source": None}

_KAP_SESSION: Dict[str, Any] = {"opener": None, "bootstrapped_at": 0.0}

_KAP_SESSION_TTL = 15 * 60

_KAP_PUBLIC_SUBJECT_CACHE: Dict[str, Any] = {"items": None, "ts": 0.0}

_KAP_PUBLIC_SUBJECT_CACHE_TTL = 6 * 60 * 60

_KAP_DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    "Accept-Encoding": "identity",
}

_KAP_TABLE_MONTHS = {
    "ocak": 1,
    "şubat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "eylül": 9,
    "ekim": 10,
    "kasım": 11,
    "aralık": 12,
}

_FLOW_DEGRADED_TTL = 25

_VYK_DEFAULT_WINDOW_HOURS = 24

_VYK_DEFAULT_LIST_BUDGET = 2

_VYK_DEFAULT_DETAIL_BUDGET = 25

_VYK_DEFAULT_DETAIL_BUDGET_MAX = 2_000

_KAP_FLOW_CATEGORY_TARGET = 50

_KAP_FLOW_CATEGORY_BACKFILL_DAYS = 90

_KAP_FLOW_CATEGORY_BACKFILL_EXTRA = _KAP_FLOW_CATEGORY_TARGET * 3

_KAP_FLOW_STORAGE_LIMIT = _VYK_DEFAULT_DETAIL_BUDGET_MAX + _KAP_FLOW_CATEGORY_BACKFILL_EXTRA

_VYK_DETAIL_WORKERS = 8

_KAP_MEMBER_OID_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=1024)

_KAP_MEMBER_OID_NEGATIVE_CACHE: BoundedMemoryCache = BoundedMemoryCache(
    max_entries=512,
    default_ttl=1800,
)

_KAP_MEMBER_OID_NEGATIVE_TTL = 1800

_KAP_MEMBER_OID_WARMED_FROM_CACHE = False

_MARKET_KAP_LOGO_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=1024)

_MARKET_KAP_LOGO_CACHE_TTL = 1800

_KAP_MEMBER_FEED_CACHE: Dict[str, Any] = {"items": [], "ts": 0.0}

_KAP_MEMBER_FEED_TTL = 600

_KAP_VALUATION_HISTORY_PRICE_TTL = 24 * 60 * 60

_KAP_VALUATION_HISTORY_PRICE_STALE_TTL = 7 * 24 * 60 * 60

_KAP_VALUATION_HISTORY_PRICE_CACHE: BoundedMemoryCache = BoundedMemoryCache(
    max_entries=128,
    default_ttl=_KAP_VALUATION_HISTORY_PRICE_STALE_TTL,
)

from app.kap_api.flow_and_feeds import (  # noqa: E402
    _KapDisclosureTableParser,
    _build_market_flow_payload,
    _empty_logo_payload,
    _fetch_kap_category_backfill,
    _fetch_kap_disclosures_via_post,
    _fetch_kap_disclosures_via_url,
    _fetch_kap_member_disclosures_for,
    _fetch_kap_member_feed,
    _fetch_kap_public_disclosures,
    _fetch_kap_public_subjects,
    _fetch_kap_result_page_via_url,
    _fetch_kap_vyk_feed,
    _kap_category,
    _kap_category_source_label,
    _kap_flow_refresh_age_seconds,
    _kap_flow_refresh_metadata,
    _kap_flow_refresh_warning,
    _kap_logo_payload_for_symbol,
    _kap_opener,
    _kap_source_label,
    _legacy_market_flow_payload,
    _local_flow_items_from_cache,
    _market_flow_payload,
    _maybe_schedule_kap_flow_live_refresh,
    _maybe_schedule_kap_flow_refresh,
    _parse_kap_api_codes,
    _parse_kap_api_disclosures,
    _parse_kap_embedded_publish_dates,
    _parse_kap_public_result_page,
    _parse_kap_publish_date,
    _parse_kap_source_datetime,
    _parse_kap_table_codes,
    _parse_kap_table_datetime,
    _persisted_market_flow_payload,
    _refresh_kap_flow_store,
    _resolve_kap_member_oid,
    _synth_quarter_publish_dt,
    _vyk_category,
    _vyk_source_label,
)
from app.kap_api.valuation_and_snapshot import (  # noqa: E402
    _annotate_kap_response_cache,
    _build_kap_historical_multiples,
    _build_kap_historical_ttm,
    _build_kap_snapshot_response,
    _build_kap_valuation_payload,
    _cached_kap_snapshot_response,
    _fetch_kap_valuation_history_prices,
    _kap_history_price_date,
    _kap_history_price_start_date,
    _kap_history_trading_date,
    _kap_quarter_end_date,
    _kap_snapshot_response_cache_key,
)

__all__ = [name for name in globals() if not name.startswith("__")]
