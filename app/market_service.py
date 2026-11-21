from __future__ import annotations

import bisect

import hashlib

import json

import threading

from app.cache import BoundedMemoryCache, cache_reset_epoch

from app import database as _market_database

from app.api_runtime import (
    Any,
    BaseModel,
    Callable,
    CONFIG,
    CORSMiddleware,
    Dict,
    FastAPI,
    Field,
    GZipMiddleware,
    HTMLParser,
    HTTPException,
    JSONResponse,
    LOGGER,
    List,
    Literal,
    MAX_REQUEST_BYTES,
    Optional,
    Path,
    PayloadValidationError,
    Query,
    ROOT,
    Request,
    RequestRateLimitMiddleware,
    Response,
    StreamingResponse,
    _ADMIN_REFRESH_TOKEN_ENV,
    _DEBUG_LOG_PATH,
    _DEBUG_RUN_ID,
    _DEBUG_SESSION_ID,
    _MARKET_SWR_LOCK_TTL_SECONDS,
    _MARKET_SWR_REVALIDATION_EXECUTOR,
    _MARKET_SWR_REVALIDATION_LEASE_TTL_SECONDS,
    _MARKET_SWR_STALE_TTL_SECONDS,
    _MARKET_SWR_WAIT_TIMEOUT_SECONDS,
    _background_cached_response,
    _cache_get_dict,
    _cache_set_json,
    _cache_status,
    _cached_response,
    _debug_log,
    _empty_logo_payload,
    _get_cache,
    _get_or_set_single_flight,
    _kap_flow_store,
    _require_admin_refresh_access,
    _schedule_swr_revalidation,
    _shared_cache_get_dict,
    _shared_cache_set,
    _shared_swr_payload,
    _swr_cache_entry,
    _swr_entry_is_fresh,
    _swr_entry_is_stale,
    _swr_entry_payload,
    _truthy_env,
    _with_market_cache_metadata,
    acquire_refresh_lease,
    asynccontextmanager,
    asyncio,
    bisect,
    close_postgres_pool,
    concurrent,
    configure_application_logging,
    database_enabled,
    date,
    datetime,
    ensure_financial_snapshot_schema,
    ensure_json_cache_schema,
    ensure_kap_flow_schema,
    ensure_reference_data_schema,
    ensure_refresh_lease_schema,
    extract_financial_snapshot_from_kap_payload,
    generate_overview_commentary,
    get_financial_snapshots_map,
    get_instrument_names,
    get_instruments,
    get_refresh_lease,
    html,
    hydrate_json_cache,
    json,
    list_overview_models,
    load_config,
    logging,
    lru_cache,
    math,
    os,
    re,
    release_refresh_lease,
    renew_refresh_lease,
    secrets,
    shutdown_market_swr_executor,
    sync_financial_snapshots_from_kap_cache,
    sync_reference_data_from_caches,
    sys,
    threading,
    time,
    timedelta,
    timezone,
    upsert_financial_snapshots,
    upsert_instrument,
    upsert_provider_fundamentals_bulk,
    uuid,
)

from app.kap_financials import (
    _build_ttm_from_reports,
    _build_ttm_sum,
    _extract_quarter_metric,
    _latest_quarter_label,
    _quarter_label_sort_key,
    _quarter_sort_key,
)

from app.kap_service import (
    BIST30_SYMBOLS,
    BIST100_SYMBOLS,
    BIST_ALL_SYMBOLS_FALLBACK,
    BIST_INDEX_CACHE_TTL,
    BIST_INDEX_ORDER,
    BIST_INDEX_REPORT_URL,
    BIST_SECTOR_INDEX_ORDER,
    BIST_STOCK_INDEX_ORDER,
    Iterable,
    KAP_COMPANY_CANDIDATES,
    KAP_SYMBOL_ALIASES,
    KapConfig,
    classify_kap_company_kind,
    csv,
    fetch_kap_company_snapshot,
    get_bist30_companies,
    get_bist100_companies,
    get_bist_all_companies,
    get_bist_index_companies,
    get_bist_index_universe,
    get_kap_companies,
    get_kap_snapshot,
    io,
    normalize_kap_symbol,
    normalize_snapshot_for_frontend,
    parse_bist_index_report_csv,
    urllib,
)

_MARKET_UNIVERSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_UNIVERSE_CACHE_TTL_SECONDS", str(6 * 60 * 60))
)

_MARKET_QUOTES_FRESH_TTL = int(os.getenv("RAGFIN_MARKET_QUOTES_FRESH_TTL_SECONDS", "90"))

_MARKET_QUOTES_STALE_TTL = int(os.getenv("RAGFIN_MARKET_QUOTES_STALE_TTL_SECONDS", "900"))

_MARKET_QUOTES_LOCK_TTL_SECONDS = float(
    os.getenv("RAGFIN_MARKET_QUOTES_LOCK_TTL_SECONDS", "180")
)

_MARKET_QUOTES_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("RAGFIN_MARKET_QUOTES_WAIT_TIMEOUT_SECONDS", "8")
)

_MARKET_QUOTES_POLL_INTERVAL_SECONDS = float(
    os.getenv("RAGFIN_MARKET_QUOTES_POLL_INTERVAL_SECONDS", "0.05")
)

_MARKET_QUOTES_NOQUOTE_TTL_SECONDS = 3

_UNIVERSE_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=6 * 60 * 60)

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

_GEFAS_GYF_QUOTE_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=24 * 60 * 60)

_GEFAS_GYF_QUOTE_CACHE_TTL = 24 * 60 * 60

_FOREIGN_HOLDING_QUOTE_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=15 * 60)

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

_PRICE_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=512, default_ttl=300)

_PRICE_CACHE_TTL = 300

_MARKET_PRICE_SNAPSHOT_TRADING_TTL_SECONDS = int(
    os.getenv("RAGFIN_MARKET_PRICE_SNAPSHOT_TRADING_TTL_SECONDS", "60")
)

_MARKET_PRICE_SNAPSHOT_OFFHOURS_TTL_SECONDS = int(
    os.getenv("RAGFIN_MARKET_PRICE_SNAPSHOT_OFFHOURS_TTL_SECONDS", "300")
)

_MARKET_PRICE_SNAPSHOT_MAX_STALE_SECONDS = int(
    os.getenv("RAGFIN_MARKET_PRICE_SNAPSHOT_MAX_STALE_SECONDS", str(6 * 60 * 60))
)

_MARKET_PRICE_SNAPSHOT_RETRY_COOLDOWN_SECONDS = float(
    os.getenv("RAGFIN_MARKET_PRICE_SNAPSHOT_RETRY_COOLDOWN_SECONDS", "30")
)

_MARKET_PRICE_SNAPSHOT_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("RAGFIN_MARKET_PRICE_SNAPSHOT_WAIT_TIMEOUT_SECONDS", "1.0")
)

_MARKET_PRICE_CACHE: BoundedMemoryCache = BoundedMemoryCache(
    max_entries=16,
    default_ttl=max(3600, _MARKET_PRICE_SNAPSHOT_MAX_STALE_SECONDS),
)

_MARKET_QUOTE_SOURCE_INDEX = "XUTUM"

_MARKET_PRICE_SHARED_CACHE_KEY = "api:market:price-snapshot:XUTUM:v1"

_MARKET_PRICE_SNAPSHOT_MEMORY_KEY = f"snapshot:{_MARKET_QUOTE_SOURCE_INDEX}"

_MARKET_PRICE_DATABASE_CACHE_PATH = (
    CONFIG.paths.processed_dir / "_shared" / "market_price_snapshot_xutum.json"
)

_INFOYATIRIM_STOCK_PAGE_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=60)

_INFOYATIRIM_STOCK_PAGE_CACHE_TTL = 60

_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT = 12

_STOCKS_CACHE: BoundedMemoryCache = BoundedMemoryCache(
    max_entries=64,
    default_ttl=max(3600, _MARKET_QUOTES_STALE_TTL),
)

_MARKET_STOCK_CARD_CHART_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=512, default_ttl=3600)

_MARKET_STOCK_CARD_CHART_CACHE_TTL = 45

_MARKET_STOCK_CARD_LIMIT = 12

_MARKET_STOCK_CARD_CHART_MAX_WORKERS = 6

_MARKET_STOCK_CARDS_QUICK_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_STOCK_CARDS_QUICK_RESPONSE_CACHE_TTL_SECONDS", "3")
)

_MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL_SECONDS", "3")
)

_MARKET_STOCK_CARD_PREVIOUS_SESSION_LOOKBACK_DAYS = 10

_STOCK_CARD_VALUATION_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=512, default_ttl=6 * 60 * 60)

_STOCK_CARD_VALUATION_CACHE_TTL = int(os.getenv("RAGFIN_STOCK_CARD_VALUATION_CACHE_TTL_SECONDS", str(6 * 60 * 60)))

_TURKEY_TIMEZONE = timezone(timedelta(hours=3))

_BIST_EQUITY_SESSION_OPEN_MINUTE = (9 * 60) + 40

_BIST_EQUITY_SESSION_CLOSE_MINUTE = (18 * 60) + 10

_MARKET_STOCK_CARD_CHART_RANGES: Dict[str, Dict[str, Any]] = {
    "1d": {"interval": "5m", "range": "1d", "ttl": 30},
    "1w": {"interval": "15m", "range": "5d", "ttl": 60},
    "1m": {"interval": "4h", "range": "1mo", "ttl": 600},
    "ytd": {"interval": "1d", "range": "ytd", "ttl": 1800},
    "1y": {"interval": "1d", "range": "1y", "ttl": 3600},
    "5y": {"interval": "1wk", "range": "5y", "ttl": 3600},
}

_STOCK_RETURN_BASE_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=1024, default_ttl=21600)

_STOCK_RETURN_BASE_TRADING_TTL = int(os.getenv("RAGFIN_STOCK_RETURN_BASE_TTL_SECONDS", "3600"))

_STOCK_RETURN_BASE_OFFHOURS_TTL = int(os.getenv("RAGFIN_STOCK_RETURN_BASE_OFFHOURS_TTL_SECONDS", "21600"))

_STOCK_RETURN_BASE_CACHE_TTL = _STOCK_RETURN_BASE_TRADING_TTL

_STOCK_RETURN_BASE_STALE_TTL = int(os.getenv("RAGFIN_STOCK_RETURN_BASE_STALE_TTL_SECONDS", str(7 * 86400)))

_STOCK_RETURN_BASE_CACHE_VERSION = "v2"

_STOCK_RETURN_BASE_BULK_CACHE_KEY = "api:stock-return-bases:bulk:XUTUM:v3"

_STOCK_RETURN_BASE_SPARK_BATCH_SIZE = 20

_STOCK_RETURN_BASE_WARMUP_COOLDOWN_SECONDS = 120.0

_STOCK_RETURN_BASE_WARMUP_LOCK = threading.Lock()

_STOCK_RETURN_BASE_WARMUP_RUNNING = False

_STOCK_RETURN_BASE_WARMUP_LAST_FINISHED = 0.0

_STOCK_HIST_MULTIPLES_TTL = int(os.getenv("RAGFIN_STOCK_HIST_MULTIPLES_TTL_SECONDS", "86400"))

_STOCK_HIST_MULTIPLES_STALE_TTL = int(os.getenv("RAGFIN_STOCK_HIST_MULTIPLES_STALE_TTL_SECONDS", str(7 * 86400)))

_STOCK_HIST_MULTIPLES_CACHE_VERSION = "v2"

_STOCK_HIST_MULTIPLES_WARMUP_COOLDOWN_SECONDS = 300.0

_STOCK_HIST_MULTIPLES_LOCK = threading.Lock()

_STOCK_HIST_MULTIPLES_STATE: Dict[str, Any] = {"_ts": 0.0, "items": {}}

_STOCK_HIST_MULTIPLES_WARMUP_RUNNING = False

_STOCK_HIST_MULTIPLES_WARMUP_LAST_FINISHED = 0.0

_ISYATIRIM_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=900)

_ISYATIRIM_CACHE_TTL = 900

_ISYATIRIM_BASIC_SUMMARY_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=300)

_ISYATIRIM_BASIC_SUMMARY_CACHE_TTL = max(
    60, int(os.getenv("RAGFIN_ISYATIRIM_BASIC_SUMMARY_CACHE_TTL_SECONDS", "1800"))
)

_ISYATIRIM_BASIC_SUMMARY_PERSIST_LOCK = threading.Lock()

_ISYATIRIM_BASIC_SUMMARY_PERSISTED_SIGNATURES: BoundedMemoryCache = BoundedMemoryCache(max_entries=512, default_ttl=3600)

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

_MARKET_INDICES_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_MARKET_INDEX_DETAIL_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_MARKET_INDEX_QUOTE_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_MARKET_INDEX_INTRADAY_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_MARKET_INDEX_RETURN_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=1800)

_MARKET_INDICES_CACHE_TTL = 30

_MARKET_INDEX_DETAIL_CACHE_TTL = 30

_MARKET_INDEX_QUOTE_CACHE_TTL = 30

_MARKET_INDEX_INTRADAY_CACHE_TTL = 30

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

_MARKET_PRICE_SNAPSHOT_REFRESH_LOCK = threading.Lock()

_MARKET_PRICE_SNAPSHOT_REFRESH_EVENT = threading.Event()

_MARKET_PRICE_SNAPSHOT_REFRESH_EVENT.set()

_MARKET_PRICE_SNAPSHOT_STATE_LOCK = threading.Lock()

_MARKET_PRICE_SNAPSHOT_STATE: Dict[str, float] = {"last_failure_at": 0.0}

_BIST_STREAM_OPEN_MINUTE = (9 * 60) + 50

_BIST_STREAM_CLOSE_MINUTE = (18 * 60) + 20

_MARKET_QUOTES_OFFHOURS_FRESH_TTL = int(
    os.getenv("RAGFIN_MARKET_QUOTES_OFFHOURS_FRESH_TTL_SECONDS", "300")
)

_MARKET_STOCK_DELTA_TRACKED_KEYS = (
    "price", "change", "change_pct", "volume", "market_cap",
    "return_1w_pct", "return_1m_pct", "return_3m_pct", "return_6m_pct",
    "return_ytd_pct", "return_1y_pct",
    "sma_50", "sma_200", "high_1y", "low_1y",
    "sma_50_diff_pct", "sma_200_diff_pct", "high_52w_dist_pct", "low_52w_dist_pct",
    "latest_quarter", "fundamental_period", "sector", "has_kap_cache",
    "fk", "pd_dd", "fd_favok", "fd_satis", "net_borc_favok",
    "hist_fk_avg", "hist_pd_dd_avg", "hist_fd_favok_avg",
    "fk_hist_diff_pct", "pd_dd_hist_diff_pct", "fd_favok_hist_diff_pct",
    "hist_quarters", "hist_fk_quarters", "hist_pd_dd_quarters", "hist_fd_favok_quarters",
    "roe_pct",
    "favok_margin_pct", "net_margin_pct", "free_float_pct", "ttm_net_kar",
    "ttm_satis_gelirleri", "ttm_favok", "ttm_fcf", "ozkaynaklar", "net_borc",
)

_MARKET_STOCK_DELTA_PAYLOAD_KEYS = (
    "company",
    "name",
    "price",
    "price_currency",
    "change",
    "change_pct",
    "price_as_of",
    "volume",
    "market_cap",
    "return_1w_pct",
    "return_1m_pct",
    "return_3m_pct",
    "return_6m_pct",
    "return_ytd_pct",
    "return_1y_pct",
    "sma_50",
    "sma_200",
    "high_1y",
    "low_1y",
    "sma_50_diff_pct",
    "sma_200_diff_pct",
    "high_52w_dist_pct",
    "low_52w_dist_pct",
    "latest_quarter",
    "fundamental_period",
    "has_kap_cache",
    "sector",
    "fk",
    "pd_dd",
    "fd_favok",
    "fd_satis",
    "net_borc_favok",
    "hist_fk_avg",
    "hist_pd_dd_avg",
    "hist_fd_favok_avg",
    "fk_hist_diff_pct",
    "pd_dd_hist_diff_pct",
    "fd_favok_hist_diff_pct",
    "hist_quarters",
    "hist_fk_quarters",
    "hist_pd_dd_quarters",
    "hist_fd_favok_quarters",
    "roe_pct",
    "favok_margin_pct",
    "net_margin_pct",
    "free_float_pct",
    "ttm_net_kar",
    "ttm_satis_gelirleri",
    "ttm_favok",
    "ttm_fcf",
    "ozkaynaklar",
    "net_borc",
)

_MACRO_ACTIVE_TTL_SECONDS = int(os.getenv("RAGFIN_MACRO_FRESH_TTL_SECONDS", "60"))

_MACRO_OFFHOURS_TTL_SECONDS = int(os.getenv("RAGFIN_MACRO_OFFHOURS_FRESH_TTL_SECONDS", "300"))

_BIST_INDEX_ACTIVE_TTL_SECONDS = int(os.getenv("RAGFIN_BIST_INDEX_FRESH_TTL_SECONDS", "60"))

_BIST_INDEX_OFFHOURS_TTL_SECONDS = int(os.getenv("RAGFIN_BIST_INDEX_OFFHOURS_FRESH_TTL_SECONDS", "300"))

_STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=256, default_ttl=6 * 60 * 60)

_STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE_TTL = 6 * 60 * 60

_XU030_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=16, default_ttl=300)

_XU030_CACHE_TTL = 120

_COMMODITY_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_COMMODITY_CACHE_TTL = 30

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

_BULK_RETURN_BASES_LOCK = threading.Lock()

_INDEX_RETURN_BASES_BULK_KEY = "api:market:index-return-bases:bulk:v3"

_FX_RETURN_BASES_BULK_KEY = "api:market:fx-return-bases:bulk:v2"

_BULK_RETURN_BATCH_DEPTH: Dict[str, int] = {}

_BULK_RETURN_DIRTY: Dict[str, bool] = {}

_FX_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_FX_CACHE_TTL = 30

_FX_RETURN_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=15 * 60)

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

_COMPARISON_FX_EXTRA_MAP: Dict[str, tuple[List[str], str]] = {
    "USD/HKD": (["USDHKD=X", "HKD=X"], "Dolar / Hong Kong Doları"),
}

_GLOBAL_INDEX_NATIVE_CURRENCY: Dict[str, str] = {
    "SP500": "USD",
    "NASDAQ": "USD",
    "NASDAQ100": "USD",
    "DOW": "USD",
    "DAX": "EUR",
    "CAC40": "EUR",
    "FTSE": "GBP",
    "NIKKEI": "JPY",
    "HANGSENG": "HKD",
}

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

_WATCH_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_WATCH_CACHE_TTL = 30

_WATCH_GLOBAL_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=300)

_WATCH_GLOBAL_CACHE_TTL = 60

_WATCH_UNIFIED_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=32, default_ttl=300)

_WATCH_DELAY_NOTE = "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk)."

_WATCH_INDEX_CANDIDATES: List[tuple[str, str, List[str]]] = [
    ("XUTUM", "BIST Tüm", ["XUTUM.IS", "^XUTUM", "XUTUM"]),
    ("XU100", "BIST 100", ["XU100.IS", "^XU100", "XU100"]),
    ("XU030", "BIST 30", ["XU030.IS", "^XU030", "XU030"]),
    ("NASDAQ", "Nasdaq", ["^IXIC", "QQQ"]),
    ("SP500", "S&P 500", ["^GSPC", "SPY"]),
    ("DOW", "Dow Jones", ["^DJI", "DIA"]),
]

_WATCH_GLOBAL_INDEX_SYMBOLS = {"NASDAQ", "SP500", "DOW"}

_WATCH_BIST_INDEX_CANDIDATES: List[tuple[str, str, List[str]]] = [
    entry for entry in _WATCH_INDEX_CANDIDATES if entry[0] not in _WATCH_GLOBAL_INDEX_SYMBOLS
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

_WATCH_FX_SYMBOLS: List[str] = ["USD/TRY", "EUR/TRY", "GBP/TRY", "EUR/USD"]

_WATCH_FX_LABELS: Dict[str, str] = {
    "USD/TRY": "Amerikan Doları",
    "EUR/TRY": "Euro",
    "GBP/TRY": "İngiliz Sterlini",
    "EUR/USD": "Euro / Dolar",
}

_WATCH_COMMODITY_SYMBOLS: List[str] = ["BRENT", "ALTIN", "GUMUS", "DOGALGAZ"]

_WATCH_COMMODITY_LABELS: Dict[str, str] = {
    "BRENT": "Brent Petrol",
    "ALTIN": "Altın (Ons)",
    "GUMUS": "Gümüş (Ons)",
    "DOGALGAZ": "Doğal Gaz",
}

_WATCH_RESPONSE_CACHE_KEY = "api:market:watch:v1"

# Re-export modularized market functions while keeping caches/locks/seams on this module.
from app.market.isyatirim_client import (
    _isyatirim_company_card_url,
    _isyatirim_basic_summary_url,
    _parse_tr_decimal,
    _strip_html_cell,
    _norm_text,
    _extract_isyatirim_basic_summary_map,
    _fetch_isyatirim_basic_summary_map,
    _persist_isyatirim_basic_summary_if_changed,
    _extract_isyatirim_historical_averages,
    _fetch_isyatirim_multiples,
    _format_isyatirim_chart_datetime,
    _parse_isyatirim_chart_datetime,
    _fetch_isyatirim_index_history,
)

from app.market.historical_valuations import (
    _pick_series_value_at_or_before,
    _pick_series_value_at_or_after,
    _stock_return_base_cache_key,
    _stock_return_base_disk_path,
    _effective_stock_return_base_ttl,
    _stock_return_bases_from_series,
    _build_ohlc_series,
    _remember_stock_return_bases_bulk,
    _persist_stock_return_bases_disk,
    _hydrate_stock_return_bases_from_disk,
    _fetch_stock_return_bases,
    _fetch_stock_return_bases_spark_batch,
    _fetch_stock_return_bases_bulk,
    _return_pct,
    _returns_from_bases,
    _cached_stock_return_bases_bulk,
    _stock_return_base_stale_symbols,
    _warm_stock_return_bases_sync,
    _schedule_xutum_return_bases_warmup_if_needed,
    _stock_hist_multiples_disk_path,
    _fetch_stock_weekly_5y_spark_batch,
    _parse_split_events,
    _has_unmatched_capital_jump,
    _compute_kap_historical_averages_for_snapshot,
    _persist_stock_hist_multiples_disk,
    _hydrate_stock_hist_multiples_from_disk,
    _cached_stock_hist_multiples_map,
    _snapshot_has_capital_changes,
    _fetch_stock_split_events_cached,
    _warm_stock_hist_multiples_sync,
    _schedule_stock_hist_multiples_warmup_if_needed,
    _bulk_return_loaded_key,
    _warm_bulk_return_bases,
    _flush_bulk_return_bases,
    _begin_bulk_return_batch,
    _end_bulk_return_batch,
    _store_bulk_return_base_entry,
)

from app.market.universe_and_quotes import (
    _get_market_http_client,
    close_market_http_client,
    _log_market_cache_event,
    _load_cached_kap_market_metadata,
    _stock_reference_record_from_kap_payload,
    _upsert_stock_reference_from_kap_payload,
    _positive_float,
    _market_cap_from_quote_and_meta,
    _market_universe_payload,
    _build_market_universe_payload,
    _quote_ts_to_iso,
    _infoyatirim_stock_page_url,
    _infoyatirim_stock_page_text,
    _extract_infoyatirim_stock_page_quote,
    _fetch_infoyatirim_html,
    _fetch_infoyatirim_stock_page_quote,
    _market_price_row_needs_fallback,
    _merge_market_price_fallback,
    _market_price_source_url,
    _market_price_snapshot_created_at,
    _market_price_snapshot_age,
    _market_price_snapshot_fresh_ttl,
    _market_price_snapshot_is_usable,
    _market_price_snapshot_is_fresh,
    _market_price_snapshot_payload,
    _remember_market_price_snapshot,
    _market_price_snapshot_memory_entry,
    _market_price_snapshot_database_read,
    _market_price_snapshot_database_write,
    _market_price_redis_available,
    _market_price_snapshot_remaining_ttl,
    _market_price_snapshot_is_servable_stale,
    _begin_market_price_snapshot_refresh,
    _end_market_price_snapshot_refresh,
    _wait_for_market_price_snapshot_refresh,
    _market_price_snapshot_retry_allowed,
    _mark_market_price_snapshot_failure,
    _clear_market_price_snapshot_failure,
    _isyatirim_all_stocks_price_url,
    _extract_isyatirim_all_stocks_price_snapshot,
    _fetch_isyatirim_all_stocks_price_snapshot,
    _build_market_price_snapshot,
    _project_market_price_snapshot,
    _fetch_market_price_map,
    _fetch_kap_price_payload,
    _fill_prices_via_yahoo,
    _xu030_payload,
    _fetch_yahoo_quote,
    _parse_yahoo_chart_payload,
    _fetch_yahoo_chart_url,
    _fetch_yahoo_chart_raw,
    _fetch_yahoo_chart_period_raw,
)

from app.market.screener_stocks import (
    _supported_stock_indexes_text,
    _supported_market_indexes_text,
    _normalize_stock_index,
    _market_stock_benchmarks,
    _market_stock_symbols_for_index,
    _refresh_cached_xutum_stocks_payload,
    _market_stock_row,
    _build_market_stocks_payload,
    _market_quote_entry_is_fresh,
    _market_no_quote_cache_key,
    _market_no_quote_cache_active,
    _store_market_no_quote_cache,
    _clear_market_no_quote_cache,
    _market_payload_has_symbol_rows,
    _market_payload_has_quote,
    _market_payload_is_no_quote,
    _market_quote_entry_is_stale,
    _market_quote_entry_payload,
    _market_unavailable_stocks_payload,
    _is_bist_market_open,
    _effective_market_quotes_fresh_ttl,
    _is_global_fx_market_open,
    _effective_macro_fresh_ttl,
    _effective_bist_index_fresh_ttl,
    _compute_market_stocks_delta,
    _market_stocks_payload,
    market_stocks_etag_revision,
)

from app.market.cards_indices_fx import (
    _normalize_market_stock_card_symbol,
    _normalize_market_stock_card_symbols,
    _normalize_market_stock_card_tier,
    _normalize_stock_card_chart_range,
    _stock_card_chart_cache_key,
    _point_datetime,
    _numeric_chart_value,
    _normalize_stock_card_line_points,
    _stock_card_latest_point_dt,
    _bist_equity_session_status,
    _stock_card_session_state,
    _fetch_previous_stock_card_intraday_chart,
    _fetch_stock_card_chart,
    _market_stock_card_chart_payload,
    _fetch_stock_card_intraday,
    _stock_cards_response_cache_key,
    _first_not_none,
    _is_missing_market_ratio,
    _as_finite_float,
    _round_market_ratio,
    _resolve_market_card_multiples,
    _stock_card_financial_snapshot_from_cache,
    _stock_card_financial_ratios_from_snapshot,
    _stock_card_financial_ratios_from_cache,
    _resolve_market_card_valuation_from_cached_data,
    _resolve_market_card_valuation,
    _market_stock_cards_quick_payload,
    _market_stock_cards_payload,
    _normalize_market_index,
    _index_return_bases_from_points,
    _index_return_bases_have_period_history,
    _fetch_index_quote,
    _fetch_index_return_bases,
    _index_returns_from_bases,
    _market_index_row,
    _comparison_error_message,
    _comparison_asset_native_currency,
    _comparison_history_result,
    _comparison_history_points_from_chart,
    _fund_comparison_history,
    _stock_comparison_history,
    _index_comparison_history,
    _fx_comparison_history,
    _align_fx_cross_series,
    _resolve_fx_to_try_series,
    _convert_comparison_series_to_try,
    _comparison_history_cache_key,
    _comparison_history_cached_asset,
    _market_comparison_history_payload,
    _market_indices_payload,
    _index_intraday_payload,
    _latest_share_count_from_kap_cache,
    _index_weight_inputs_for_symbol,
    _apply_index_weight_formula,
    _market_index_constituent_stock_rows,
    _index_constituents,
    _market_index_detail_payload,
    _market_commodities_payload,
    _fx_quote_currency,
    _fetch_fx_return_bases,
    _fx_returns_from_bases,
    _fx_item_from_quote,
    _fx_direct_item,
    _positive_number,
    _fx_derived_item,
    _market_fx_payload,
    _empty_watch_item,
    _normalize_watch_item,
    _pick_watch_items,
    _watch_index_item,
    _market_watch_global_payload,
    _market_watch_payload,
    _market_watch_unified_payload,
)

__all__ = [name for name in globals() if not name.startswith("__")]
