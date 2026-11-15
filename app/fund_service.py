from __future__ import annotations

import atexit
import concurrent.futures
import copy
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import unicodedata
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlencode

import httpx

_ORIGINAL_HTTPX_CLIENT = httpx.Client
_CLIENT_LOCK = threading.Lock()
_KAP_CLIENT: Optional[httpx.Client] = None
_FINTABLES_PROXY_CLIENT: Optional[httpx.Client] = None

from app import json_cache as json_cache_store
from app.database import (
    connect_postgres,
    database_enabled,
    lock_down_postgres_tables,
    read_json_cache,
    read_json_cache_history_metadata,
    write_json_cache,
)
from app.reference_data import (
    get_instruments,
    get_instrument,
    get_instrument_name,
    get_instrument_names,
    upsert_instruments,
)

LOGGER = logging.getLogger(__name__)

try:
    from app.cache import get_json_dict as _cache_get_dict
    from app.cache import set_json as _cache_set_json
except Exception:  # pragma: no cover - cache layer is optional for import-time tools
    _cache_get_dict = None  # type: ignore[assignment]
    _cache_set_json = None  # type: ignore[assignment]

FINTABLES_USER_AGENT = os.getenv(
    "RAGFIN_FINTABLES_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)
FINTABLES_GATE_BASE_URL = os.getenv(
    "RAGFIN_FINTABLES_GATE_BASE_URL",
    "https://gate.fintables.com",
).rstrip("/")
FINTABLES_UDF_HISTORY_ENDPOINT = os.getenv(
    "RAGFIN_FINTABLES_UDF_HISTORY_ENDPOINT",
    f"{FINTABLES_GATE_BASE_URL}/barbar/udf/history",
)
FINTABLES_YIELD_SUMMARY_ENDPOINT = os.getenv(
    "RAGFIN_FINTABLES_YIELD_SUMMARY_ENDPOINT",
    f"{FINTABLES_GATE_BASE_URL}/barbar/server/yield",
)
FINTABLES_PROXY_BASE_URL = os.getenv("RAGFIN_FINTABLES_PROXY_BASE_URL", "").rstrip("/")
FINTABLES_PROXY_TOKEN = (
    os.getenv("RAGFIN_FINTABLES_PROXY_TOKEN", "").strip()
    or os.getenv("RAGFIN_ADMIN_REFRESH_TOKEN", "").strip()
)
FINTABLES_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_FINTABLES_TIMEOUT_SECONDS", "12"))
FINTABLES_FUND_BASE_URL = os.getenv(
    "RAGFIN_FINTABLES_FUND_BASE_URL",
    "https://fintables.com/fonlar",
).rstrip("/")
FINTABLES_GATE_BLOCKED_MESSAGE = "Fintables Gate blocked by WAF/Cloudflare"
FINTABLES_CURL_FALLBACK_ENABLED = os.getenv("RAGFIN_FINTABLES_CURL_FALLBACK_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
TEFAS_BASE_URL = os.getenv("RAGFIN_TEFAS_BASE_URL", "https://www.tefas.gov.tr").rstrip("/")
TEFAS_FUNDS_LIST_ENDPOINT = os.getenv(
    "RAGFIN_TEFAS_FUNDS_LIST_ENDPOINT",
    f"{TEFAS_BASE_URL}/api/funds/fonGnlBlgSiraliGetir",
)
TEFAS_PORTFOLIO_ENDPOINT = os.getenv(
    "RAGFIN_TEFAS_PORTFOLIO_ENDPOINT",
    f"{TEFAS_BASE_URL}/api/funds/dagilimSiraliGetirT",
)
TEFAS_RETURNS_RB_ENDPOINT = os.getenv(
    "RAGFIN_TEFAS_RETURNS_RB_ENDPOINT",
    f"{TEFAS_BASE_URL}/api/funds/fonGetiriBazliBilgiGetir",
)
TEFAS_RETURNS_SB_ENDPOINT = os.getenv(
    "RAGFIN_TEFAS_RETURNS_SB_ENDPOINT",
    f"{TEFAS_BASE_URL}/api/funds/fonBuyuklukBazliBilgiGetir",
)
TEFAS_RETURNS_MB_ENDPOINT = os.getenv(
    "RAGFIN_TEFAS_RETURNS_MB_ENDPOINT",
    f"{TEFAS_BASE_URL}/api/funds/fonYonetimBazliBilgiGetir",
)
TEFAS_FUNDS_LIST_PAGE_SIZE = int(os.getenv("RAGFIN_TEFAS_FUNDS_LIST_PAGE_SIZE", "5000"))
TEFAS_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_TEFAS_TIMEOUT_SECONDS", "60"))
TEFAS_HTTP_RETRY_ATTEMPTS = int(os.getenv("RAGFIN_TEFAS_HTTP_RETRY_ATTEMPTS", "5"))
TEFAS_HTTP_RETRY_BASE_SECONDS = float(os.getenv("RAGFIN_TEFAS_HTTP_RETRY_BASE_SECONDS", "5"))
TEFAS_HTTP_RETRY_MAX_SECONDS = float(os.getenv("RAGFIN_TEFAS_HTTP_RETRY_MAX_SECONDS", "45"))
TEFAS_HTTP_RETRY_AFTER_MAX_SECONDS = float(os.getenv("RAGFIN_TEFAS_HTTP_RETRY_AFTER_MAX_SECONDS", "300"))
TEFAS_FUND_HISTORY_CHUNK_DAYS = int(os.getenv("RAGFIN_TEFAS_FUND_HISTORY_CHUNK_DAYS", "31"))
TEFAS_HISTORY_CHUNK_MAX_DAYS = int(os.getenv("RAGFIN_TEFAS_HISTORY_CHUNK_MAX_DAYS", "90"))
TEFAS_DIRECT_PAGE_SIZE = int(os.getenv("RAGFIN_TEFAS_DIRECT_PAGE_SIZE", "1000"))
TEFAS_DIRECT_PAGE_DELAY_SECONDS = float(os.getenv("RAGFIN_TEFAS_DIRECT_PAGE_DELAY_SECONDS", "1"))
TEFAS_DIRECT_CHUNK_DELAY_SECONDS = float(os.getenv("RAGFIN_TEFAS_DIRECT_CHUNK_DELAY_SECONDS", "2"))
TEFASFON_SOURCE_URL = os.getenv("RAGFIN_TEFASFON_SOURCE_URL", "https://pypi.org/project/tefasfon/")
TEFASFON_FUNDS_SOURCE = "tefasfon_funds"
TEFASFON_RETURNS_SOURCE = "tefasfon_returns"
TEFASFON_PORTFOLIO_SOURCE = "tefasfon_portfolio"
TEFAS_LIST_SNAPSHOT_SOURCE = "tefas_list_snapshot"
TEFAS_DIRECT_FUNDS_SOURCE = "tefas_direct_funds"
TEFAS_DIRECT_RETURNS_SOURCE = "tefas_direct_returns"
TEFAS_DIRECT_PORTFOLIO_SOURCE = "tefas_direct_portfolio"
FINTABLES_UDF_HISTORY_SOURCE = "fintables_udf_history"
FINTABLES_YIELD_SUMMARY_SOURCE = "fintables_yield_summary"
DAILY_RETURN_LOCAL_PRICE_SOURCE = "local_price_diff"
FUND_HISTORY_SOURCE_POLICY = "fintables_daily_tefas_fallback"
_TEFAS_ALLOWED_FUND_TYPES = {"SEC", "PEN", "ETF", "RE", "VC"}
TEFAS_FUND_TYPES = tuple(
    item.strip().upper()
    for item in os.getenv("RAGFIN_TEFAS_FUND_TYPES", "SEC").split(",")
    if item.strip().upper() in _TEFAS_ALLOWED_FUND_TYPES
) or ("SEC",)
TEFAS_OPEN_ONLY = os.getenv("RAGFIN_TEFAS_OPEN_ONLY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
FUNDS_SNAPSHOT_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_SNAPSHOT_TTL_SECONDS", str(24 * 60 * 60)))
FUNDS_SNAPSHOT_INTRADAY_CHECK_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_SNAPSHOT_INTRADAY_CHECK_TTL_SECONDS", "300"))
FUNDS_MAX_LOOKBACK_CALENDAR_DAYS = int(os.getenv("RAGFIN_FUNDS_MAX_LOOKBACK_CALENDAR_DAYS", "14"))
FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS = int(
    os.getenv("RAGFIN_FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS", "900")
)
FUNDS_DAILY_SNAPSHOT_CACHE_VERSION = 2
FUNDS_HISTORY_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_HISTORY_TTL_SECONDS", str(24 * 60 * 60)))
FUNDS_ALLOCATION_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_ALLOCATION_TTL_SECONDS", str(24 * 60 * 60)))
# Empty allocation/history payloads are cached aggressively to avoid hammering
# TEFAS, but we keep the empty TTL much shorter than the populated one so the
# UI eventually recovers when upstream data becomes available again.
FUNDS_ALLOCATION_EMPTY_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_ALLOCATION_EMPTY_TTL_SECONDS", str(15 * 60)))
FUNDS_ALLOCATION_HISTORY_DAILY_FALLBACK_MAX_REQUESTS = int(
    os.getenv("RAGFIN_FUNDS_ALLOCATION_HISTORY_DAILY_FALLBACK_MAX_REQUESTS", "30")
)
FUNDS_ALLOCATION_HISTORY_DAILY_MAX_WORKERS = int(
    os.getenv("RAGFIN_FUNDS_ALLOCATION_HISTORY_DAILY_MAX_WORKERS", "2")
)
FUNDS_HOLDINGS_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_HOLDINGS_TTL_SECONDS", str(7 * 24 * 60 * 60)))
# KAP's portfolio PDF is monthly; revalidating its disclosure list every
# 2 hours catches newly published monthly reports without hammering KAP or
# rewriting Supabase JSON cache on unchanged reports.
FUNDS_HOLDINGS_DISCLOSURE_CHECK_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_HOLDINGS_DISCLOSURE_CHECK_TTL_SECONDS", "7200"))
FUNDS_HOLDINGS_NEGATIVE_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_HOLDINGS_NEGATIVE_TTL_SECONDS", str(30 * 60)))
FUNDS_HISTORY_CHUNK_DAYS = int(os.getenv("RAGFIN_FUNDS_HISTORY_CHUNK_DAYS", "60"))
FUNDS_WEB_HISTORY_CHUNK_DAYS = int(os.getenv("RAGFIN_FUNDS_WEB_HISTORY_CHUNK_DAYS", "30"))
FUNDS_WEB_HISTORY_SLEEP_SECONDS = float(os.getenv("RAGFIN_FUNDS_WEB_HISTORY_SLEEP_SECONDS", "0.35"))
FUNDS_DETAIL_MAX_WORKERS = int(os.getenv("RAGFIN_FUNDS_DETAIL_MAX_WORKERS", "16"))
FUNDS_COLLECTOR_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_COLLECTOR_LOOKBACK_DAYS", "30"))
FUNDS_AUTO_FETCH_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_AUTO_FETCH_LOOKBACK_DAYS", "30"))
FUNDS_AUTO_FETCH_NEGATIVE_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_AUTO_FETCH_NEGATIVE_TTL_SECONDS", "60"))
FUNDS_OVERVIEW_METRIC_MONTHS = int(os.getenv("RAGFIN_FUNDS_OVERVIEW_METRIC_MONTHS", "6"))
FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS", "10"))
FUNDS_FAST_LONG_RANGE_DAYS = int(os.getenv("RAGFIN_FUNDS_FAST_LONG_RANGE_DAYS", "120"))
FUNDS_RECENT_DETAIL_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_RECENT_DETAIL_LOOKBACK_DAYS", "35"))
FUNDS_HISTORY_WARMUP_DAYS = int(os.getenv("RAGFIN_FUNDS_HISTORY_WARMUP_DAYS", "366"))
FUNDS_FULL_HISTORY_START_DATE = os.getenv("RAGFIN_FUNDS_FULL_HISTORY_START_DATE", "2000-01-01").strip() or "2000-01-01"
FUNDS_LIST_MIN_AUM = float(os.getenv("RAGFIN_FUNDS_LIST_MIN_AUM", "0"))
FUND_PRICES_DB_FILENAME = os.getenv("RAGFIN_FUND_PRICES_DB_FILENAME", "fund_prices.sqlite3")
FUND_PRICE_POINTS_CACHE_TTL_SECONDS = max(
    0,
    int(os.getenv("RAGFIN_FUND_PRICE_POINTS_CACHE_TTL_SECONDS", "60")),
)
FUND_PRICE_POINTS_CACHE_MAX_ENTRIES = 64
FUND_PRICE_POINTS_CACHE_MAX_TOTAL_BYTES = 8 * 1024 * 1024
FUND_PRICE_POINTS_CACHE_MAX_ENTRY_BYTES = 2 * 1024 * 1024
FUND_PRICE_POINTS_CACHE_METRICS_FLUSH_SECONDS = 300.0
FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS = max(
    0,
    int(os.getenv("RAGFIN_FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS", str(7 * 24 * 60 * 60))),
)
KAP_BASE_URL = os.getenv("RAGFIN_KAP_BASE_URL", "https://www.kap.org.tr").rstrip("/")
KAP_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_KAP_TIMEOUT_SECONDS", "20"))
KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID = os.getenv(
    "RAGFIN_KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID",
    "8aca490d502e34b801502e380044002b",
).strip()
KAP_HOLDINGS_LOOKBACK_DAYS = int(os.getenv("RAGFIN_KAP_HOLDINGS_LOOKBACK_DAYS", "365"))
KAP_HOLDINGS_ATTACHMENT_TEXT_CACHE_VERSION = 2

_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
_SNAPSHOT_ALL_ROWS_CACHE: Dict[str, Dict[str, Any]] = {}
_LATEST_PRICE_POINTS_SUPERSET_CACHE: Dict[str, Dict[str, Any]] = {}
try:
    _MEMORY_CACHE_MAX_ENTRIES = max(
        16,
        int(os.getenv("RAGFIN_FUNDS_MEMORY_CACHE_MAX_ENTRIES", "256")),
    )
except ValueError:
    _MEMORY_CACHE_MAX_ENTRIES = 256

# The `/funds` ETag preflight reads the listing revision and catalogue summary
# before it can answer a 304, so those two values are memoised briefly instead
# of costing a Supabase round trip on every revalidation.
_FUND_LISTING_STATE_MEMO_TTL_SECONDS = float(
    os.getenv("RAGFIN_FUND_LISTING_STATE_MEMO_SECONDS", "30")
)


def _clear_all_fund_memory_caches() -> None:
    _MEMORY_CACHE.clear()
    _SNAPSHOT_ALL_ROWS_CACHE.clear()
    _LATEST_PRICE_POINTS_SUPERSET_CACHE.clear()


def _invalidate_fund_price_memory_caches() -> None:
    """Drop price/listing state entries while keeping catalog-only snapshot tables warm."""
    for cache_key in [
        key
        for key in list(_MEMORY_CACHE.keys())
        if str(key).startswith(
            (
                "fund-listing-state:",
                "latest-fund-price-points:",
                "snapshot:",
                "daily-return-overrides:",
            )
        )
    ]:
        _MEMORY_CACHE.pop(cache_key, None)
    _LATEST_PRICE_POINTS_SUPERSET_CACHE.clear()


def _peek_all_snapshot_rows(
    processed_dir: Path,
    generation: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not database_enabled():
        return None
    if generation is None:
        generation = fund_listing_revision(processed_dir).get("catalog_generation", 0)
    all_rows_cache_key = f"fund-snapshot-all-rows:{_fund_prices_db_path(processed_dir)}:{generation}"
    cached = _SNAPSHOT_ALL_ROWS_CACHE.get(all_rows_cache_key) or _MEMORY_CACHE.get(all_rows_cache_key)
    ttl = 21600.0 if generation else 3600.0
    if (
        isinstance(cached, dict)
        and isinstance(cached.get("cached_at"), (int, float))
        and time.time() - float(cached["cached_at"]) <= ttl
        and isinstance(cached.get("rows"), list)
    ):
        return cached
    return None


def _memory_cache_put(cache_key: str, value: Dict[str, Any]) -> None:
    """Insert into the process-local cache, evicting oldest entries first.

    This is FIFO, not LRU: reads do not refresh insertion order. 256 entries is
    far above the number of distinct fund list/detail views a single process
    serves between deploys, so eviction should stay rare.
    """
    _MEMORY_CACHE.pop(cache_key, None)
    _MEMORY_CACHE[cache_key] = value
    overflow = len(_MEMORY_CACHE) - _MEMORY_CACHE_MAX_ENTRIES
    if overflow <= 0:
        return
    for stale_key in list(_MEMORY_CACHE.keys())[:overflow]:
        _MEMORY_CACHE.pop(stale_key, None)


_SNAPSHOT_REFRESH_LOCK = threading.Lock()
_FUND_PRICES_SCHEMA_LOCK = threading.Lock()
_FUND_PRICES_SCHEMA_READY = False
_FUND_SNAPSHOT_MIGRATION_LOCK = threading.Lock()
_FUND_SNAPSHOT_COMMIT_LOCK = threading.Lock()
_AUTO_FETCH_LOCK = threading.Lock()
_AUTO_FETCH_IN_FLIGHT: Dict[str, threading.Event] = {}
_AUTO_FETCH_NEGATIVE_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_kap_client() -> httpx.Client:
    """Return a shared persistent HTTP client for KAP requests, or a fresh mock in tests."""
    if httpx.Client is not _ORIGINAL_HTTPX_CLIENT:
        return httpx.Client(timeout=KAP_TIMEOUT_SECONDS, follow_redirects=True)
    global _KAP_CLIENT
    with _CLIENT_LOCK:
        if _KAP_CLIENT is None or _KAP_CLIENT.is_closed:
            _KAP_CLIENT = httpx.Client(
                timeout=KAP_TIMEOUT_SECONDS,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0),
            )
        return _KAP_CLIENT


def _get_fintables_proxy_client(timeout_seconds: Optional[float] = None) -> httpx.Client:
    """Return a shared persistent HTTP client for Fintables proxy requests, or a fresh mock in tests."""
    effective_timeout = timeout_seconds if timeout_seconds is not None else FINTABLES_TIMEOUT_SECONDS
    if httpx.Client is not _ORIGINAL_HTTPX_CLIENT:
        return httpx.Client(timeout=effective_timeout, follow_redirects=True)
    global _FINTABLES_PROXY_CLIENT
    with _CLIENT_LOCK:
        if _FINTABLES_PROXY_CLIENT is None or _FINTABLES_PROXY_CLIENT.is_closed:
            _FINTABLES_PROXY_CLIENT = httpx.Client(
                timeout=effective_timeout,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0),
            )
        return _FINTABLES_PROXY_CLIENT


def close_shared_http_clients() -> None:
    """Close any shared HTTP connection pools cleanly."""
    global _KAP_CLIENT, _FINTABLES_PROXY_CLIENT
    with _CLIENT_LOCK:
        client_kap = _KAP_CLIENT
        _KAP_CLIENT = None
        client_proxy = _FINTABLES_PROXY_CLIENT
        _FINTABLES_PROXY_CLIENT = None

    if client_kap is not None:
        try:
            client_kap.close()
        except Exception:
            LOGGER.debug("error closing KAP HTTP client", exc_info=True)
    if client_proxy is not None:
        try:
            client_proxy.close()
        except Exception:
            LOGGER.debug("error closing Fintables proxy HTTP client", exc_info=True)


atexit.register(close_shared_http_clients)


@dataclass
class _FundPricePointsCacheEntry:
    payload: List[Dict[str, Any]]
    approx_bytes: int
    expires_at: float


@dataclass
class _FundPricePointsReadInFlight:
    event: threading.Event = field(default_factory=threading.Event)
    payload: Optional[List[Dict[str, Any]]] = None
    error: Optional[BaseException] = None


_FUND_PRICE_POINTS_CACHE: "OrderedDict[Any, _FundPricePointsCacheEntry]" = OrderedDict()
_FUND_PRICE_POINTS_CACHE_BYTES = 0
_FUND_PRICE_POINTS_CACHE_LOCK = threading.RLock()
_FUND_PRICE_POINTS_IN_FLIGHT: Dict[Any, _FundPricePointsReadInFlight] = {}
_FUND_PRICE_POINTS_CACHE_GENERATION = 0
_FUND_PRICE_POINTS_CACHE_METRICS_SINCE = time.monotonic()
_FUND_PRICE_POINTS_CACHE_METRICS: Dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "coalesced_hits": 0,
    "coalesced_errors": 0,
    "evictions": 0,
    "oversize_skips": 0,
    "invalidations": 0,
}

TARGET_FUND_MANAGER_KEYWORDS = tuple(
    item.strip()
    for item in os.getenv(
        "RAGFIN_TARGET_FUND_MANAGERS",
        "TERA,PUSULA,ATLAS,BULLS,VEGA,PARDUS,AKTIF",
    ).split(",")
    if item.strip()
)
TARGET_FUND_CODES = tuple(
    normalize_code.strip().upper()
    for normalize_code in os.getenv("RAGFIN_TARGET_FUND_CODES", "").split(",")
    if normalize_code.strip()
)

FUND_ALLOCATION_LABELS: Dict[str, str] = {
    "bb": "Banka Bonosu",
    "byf": "Borsa Yatırım Fonu",
    "d": "Diğer",
    "db": "Devlet Bonosu",
    "bpp": "Borsa Para Piyasası",
    "btaa": "Borsa Ters Repo",
    "btas": "Banka Tahvil/Bonosu",
    "dt": "Devlet Tahvili",
    "dot": "Döviz Ödemeli Tahvil",
    "eut": "Eurobond",
    "fb": "Finansman Bonosu",
    "fkb": "Finansman Kira Sertifikası",
    "gas": "Gayrimenkul Sertifikası",
    "gsykb": "Girişim Sermayesi YF Katılma Payları",
    "gsyy": "Girişim Sermayesi YF",
    "gykb": "Gayrimenkul YF Katılma Payları",
    "gyy": "Gayrimenkul YF",
    "hb": "Hazine Bonosu",
    "hs": "Hisse Senedi",
    "kba": "Kamu Borçlanma Araçları",
    "kh": "Kamu Kira Sertifikası",
    "khau": "Kamu Kira Sertifikası (Altın)",
    "khd": "Kamu Kira Sertifikası (Döviz)",
    "khtl": "Kamu Kira Sertifikası (TL)",
    "kks": "Kira Sertifikası",
    "kksd": "Kira Sertifikası (Döviz)",
    "kkstl": "Kira Sertifikası (TL)",
    "kksyd": "Kira Sertifikası (Yurt Dışı)",
    "km": "Kıymetli Maden",
    "kmbyf": "Kıymetli Maden BYF",
    "kmkba": "Kıymetli Maden Kamu Borçlanma Araçları",
    "kmkks": "Kıymetli Maden Kira Sertifikası",
    "kibd": "Kamu İç Borçlanma Senedi",
    "osks": "Özel Sektör Kira Sertifikası",
    "ost": "Özel Sektör Tahvili",
    "r": "Repo",
    "t": "Türev Araçlar",
    "tpp": "Takasbank Para Piyasası",
    "tr": "Ters Repo",
    "vdm": "Varlığa Dayalı Menkul Kıymet",
    "vm": "Vadeli Mevduat",
    "vmau": "Vadeli Mevduat (Altın)",
    "vmd": "Vadeli Mevduat (Döviz)",
    "vmtl": "Vadeli Mevduat (TL)",
    "vint": "Vadeli İşlem Nakit Teminatları",
    "yba": "Yabancı Borçlanma Araçları",
    "ybkb": "Yabancı Kamu Borçlanma Araçları",
    "ybosb": "Yabancı Özel Sektör Borçlanma Araçları",
    "ybyf": "Yabancı BYF",
    "yhs": "Yabancı Hisse Senedi",
    "ymk": "Yabancı Menkul Kıymet",
    "yyf": "Yatırım Fonları Katılma Payları",
    "oksyd": "Özel Sektör Yurt Dışı Kira Sertifikası",
    "osdb": "Özel Sektör Dış Borçlanma Araçları",
}

FUND_ALLOCATION_KEYS = tuple(FUND_ALLOCATION_LABELS)


class FundUpstreamError(RuntimeError):
    pass


class FundFormatError(FundUpstreamError):
    pass


class FintablesUpstreamError(FundUpstreamError):
    pass


class FintablesFormatError(FintablesUpstreamError):
    pass


class TefasUpstreamError(FundUpstreamError):
    pass


class TefasRateLimitError(TefasUpstreamError):
    def __init__(self, message: str, *, retry_after_seconds: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class TefasFormatError(TefasUpstreamError):
    pass


@dataclass
class FundSnapshotFetchResult:
    """Outcome of a bounded TEFAS catalogue probe.

    ``warnings`` is deliberately separate from ``upstream_errors``: an empty
    successful response means “not published”, while a failed request means
    that the upstream could not be trusted for that date.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    resolution_status: str = "not_published"
    requested_as_of: Optional[str] = None
    resolved_as_of: Optional[str] = None
    attempted_dates: List[str] = field(default_factory=list)
    empty_dates: List[str] = field(default_factory=list)
    upstream_errors: List[str] = field(default_factory=list)
    source: str = TEFASFON_FUNDS_SOURCE

    @property
    def all_requests_succeeded(self) -> bool:
        return not self.upstream_errors


def reset_fund_caches_for_tests() -> None:
    _clear_all_fund_memory_caches()
    with _AUTO_FETCH_LOCK:
        _AUTO_FETCH_IN_FLIGHT.clear()
        _AUTO_FETCH_NEGATIVE_CACHE.clear()


def _empty_fund_listing_revision() -> Dict[str, int]:
    return {"catalog_generation": 0, "price_generation": 0}


def _fund_listing_state_cache_key(processed_dir: Path) -> str:
    return f"fund-listing-state:{_fund_prices_db_path(processed_dir)}"


def invalidate_fund_listing_state_cache() -> None:
    """Drop memoised listing revision/summary pairs after a local write."""

    prefix = "fund-listing-state:"
    for cache_key in [key for key in list(_MEMORY_CACHE.keys()) if str(key).startswith(prefix)]:
        _MEMORY_CACHE.pop(cache_key, None)


def fund_listing_state(
    processed_dir: Path,
) -> Tuple[Dict[str, int], Optional[Dict[str, Any]]]:
    """Read the listing revision and the compact catalogue summary together.

    Both values live in the same ``fund_listing_state`` row and every ``/funds``
    request needs them, including ETag revalidations. PostgreSQL reads are
    memoised for a short window so the preflight does not pay a round trip per
    request; in-process writers invalidate the memo immediately. SQLite mode
    stays uncached because local tools rewrite the store directly.
    """

    if not database_enabled():
        return fund_listing_revision(processed_dir), _snapshot_metadata_from_store(processed_dir)

    cache_key = _fund_listing_state_cache_key(processed_dir)
    cached = _MEMORY_CACHE.get(cache_key)
    cached_at = cached.get("cached_at") if isinstance(cached, dict) else None
    if (
        isinstance(cached_at, (int, float))
        and time.time() - float(cached_at) <= _FUND_LISTING_STATE_MEMO_TTL_SECONDS
    ):
        cached_revision = cached.get("revision") if isinstance(cached.get("revision"), dict) else {}
        cached_summary = cached.get("summary")
        return (
            {
                "catalog_generation": int(cached_revision.get("catalog_generation") or 0),
                "price_generation": int(cached_revision.get("price_generation") or 0),
            },
            dict(cached_summary) if isinstance(cached_summary, dict) else None,
        )

    ensure_fund_prices_schema(processed_dir)
    with connect_postgres() as conn:
        row = conn.execute(
            "SELECT catalog_generation, price_generation, metadata_json "
            "FROM fund_listing_state WHERE id = 1"
        ).fetchone()

    revision = _empty_fund_listing_revision()
    summary: Optional[Dict[str, Any]] = None
    if isinstance(row, dict):
        revision = {
            "catalog_generation": int(row.get("catalog_generation") or 0),
            "price_generation": int(row.get("price_generation") or 0),
        }
        parsed = _safe_json_loads(row.get("metadata_json"), None)
        if isinstance(parsed, dict):
            summary = parsed
    elif row is not None:
        revision = {
            "catalog_generation": int(row["catalog_generation"] or 0),
            "price_generation": int(row["price_generation"] or 0),
        }
        parsed = _safe_json_loads(row["metadata_json"], None)
        if isinstance(parsed, dict):
            summary = parsed

    _memory_cache_put(
        cache_key,
        {"cached_at": time.time(), "revision": revision, "summary": summary},
    )
    return dict(revision), dict(summary) if summary is not None else None


def fund_listing_revision(processed_dir: Path) -> Dict[str, int]:
    """Read the compact, transactionally maintained list revision counters."""

    if database_enabled():
        revision, _summary = fund_listing_state(processed_dir)
        return revision
    with _connect_fund_prices_db(processed_dir) as conn:
        row = conn.execute(
            "SELECT catalog_generation, price_generation FROM fund_listing_state WHERE id = 1"
        ).fetchone()
    if isinstance(row, dict):
        return {
            "catalog_generation": int(row.get("catalog_generation") or 0),
            "price_generation": int(row.get("price_generation") or 0),
        }
    if row is not None:
        return {
            "catalog_generation": int(row["catalog_generation"] or 0),
            "price_generation": int(row["price_generation"] or 0),
        }
    return _empty_fund_listing_revision()


def _set_fund_listing_catalog_revision(conn: Any, generation: int, metadata: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO fund_listing_state (
            id, catalog_generation, price_generation, metadata_json, updated_at
        ) VALUES (1, ?, 0, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            catalog_generation = excluded.catalog_generation,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at
        """,
        (int(generation), _stable_json_dumps(metadata), _utc_now_iso()),
    )
    invalidate_fund_listing_state_cache()


def _bump_fund_listing_price_revision(conn: Any) -> None:
    conn.execute(
        """
        INSERT INTO fund_listing_state (
            id, catalog_generation, price_generation, metadata_json, updated_at
        ) VALUES (1, 0, 1, '{}', ?)
        ON CONFLICT (id) DO UPDATE SET
            price_generation = fund_listing_state.price_generation + 1,
            updated_at = excluded.updated_at
        """,
        (_utc_now_iso(),),
    )
    invalidate_fund_listing_state_cache()


def normalize_fund_code(raw: str | None) -> str:
    return "".join(str(raw or "").strip().upper().split())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _fund_full_history_start_date() -> date:
    try:
        return date.fromisoformat(FUNDS_FULL_HISTORY_START_DATE)
    except ValueError:
        return date(2000, 1, 1)


def _fund_cache_dir(processed_dir: Path) -> Path:
    return processed_dir / "funds_cache"


def _fund_daily_snapshot_dir(processed_dir: Path) -> Path:
    return _fund_cache_dir(processed_dir) / "daily_snapshots"


def _fund_daily_snapshot_path(processed_dir: Path, as_of: date) -> Path:
    return _fund_daily_snapshot_dir(processed_dir) / f"{as_of.isoformat()}.json"


def _snapshot_path(processed_dir: Path) -> Path:
    return _fund_cache_dir(processed_dir) / "funds_latest.json"


def _legacy_snapshot_path(processed_dir: Path) -> Path:
    return _fund_cache_dir(processed_dir) / "funds_latest.legacy.json"


def _history_dir(processed_dir: Path) -> Path:
    return _fund_cache_dir(processed_dir) / "history"


def _history_path(processed_dir: Path, fund_code: str) -> Path:
    return _history_dir(processed_dir) / f"{normalize_fund_code(fund_code)}.json"


def _allocations_dir(processed_dir: Path) -> Path:
    return _fund_cache_dir(processed_dir) / "allocations"


def _allocations_path(processed_dir: Path, fund_code: str) -> Path:
    return _allocations_dir(processed_dir) / f"{normalize_fund_code(fund_code)}.json"


def _allocations_history_path(processed_dir: Path, fund_code: str, lookback_days: int) -> Path:
    bounded = max(1, min(365, int(lookback_days)))
    return _allocations_dir(processed_dir) / f"{normalize_fund_code(fund_code)}_history_{bounded}d.json"


def _holdings_dir(processed_dir: Path) -> Path:
    return _fund_cache_dir(processed_dir) / "holdings"


def _holdings_path(processed_dir: Path, fund_code: str) -> Path:
    return _holdings_dir(processed_dir) / f"{normalize_fund_code(fund_code)}.json"


def _holdings_attachment_text_dir(processed_dir: Path) -> Path:
    return _holdings_dir(processed_dir) / "attachment_text"


def _holdings_attachment_text_path(processed_dir: Path, disclosure_index: Any, obj_id: Any) -> Path:
    index_text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(disclosure_index or "unknown").strip() or "unknown")
    obj_text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(obj_id or "unknown").strip() or "unknown")
    return _holdings_attachment_text_dir(processed_dir) / f"{index_text}_{obj_text}.json"


def _fund_prices_db_path(processed_dir: Path) -> Path:
    raw_path = os.getenv("RAGFIN_FUND_PRICES_DB_PATH", "").strip()
    if raw_path:
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else processed_dir / candidate
    return processed_dir / FUND_PRICES_DB_FILENAME


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _safe_json_loads(raw: Any, fallback: Any) -> Any:
    if raw is None:
        return fallback
    try:
        return json.loads(str(raw))
    except Exception:
        return fallback


def _connect_fund_prices_db(processed_dir: Path) -> Any:
    if database_enabled():
        return connect_postgres()
    path = _fund_prices_db_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _init_fund_prices_schema(conn)
    return conn


_FUND_SNAPSHOT_FIELDS = (
    "fund_code",
    "name",
    "fund_type",
    "founder_company",
    "manager_company",
    "price",
    "daily_return",
    "aum",
    "investor_count",
    "share_count",
    "risk_value",
    "currency",
    "as_of",
    "source",
    "period_returns",
)

# Keep the SQL result limited to values needed to reconstruct the public
# catalogue rows. Internal sort/search/generation columns remain in Postgres
# for filtering and ordering, but do not need to cross Supavisor on full reads.
_FUND_SNAPSHOT_READ_COLUMNS = (
    "fund_code",
    "name",
    "fund_type",
    "founder_company",
    "manager_company",
    "price",
    "daily_return",
    "aum",
    "investor_count",
    "share_count",
    "risk_value_json",
    "currency",
    "as_of",
    "source",
    "period_returns_json",
    "attributes_json",
    "present_fields_json",
)

# Bumping this prefix forces the next catalogue commit to rewrite every row.
# That is how stored-format changes (such as the compact absent-field list)
# reach databases whose row content has not otherwise changed.
_FUND_SNAPSHOT_CONTENT_HASH_VERSION = "v2"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return json_cache_store.read_json_cache_local_first(path, remote_reader=read_json_cache)


def _read_local_json(path: Path) -> Optional[Dict[str, Any]]:
    return json_cache_store.read_local_json(path)


def _should_clear_listing_memory_caches_for_path(path: Path) -> bool:
    if not database_enabled():
        return True
    parent_names = {path.parent.name.lower(), path.parent.parent.name.lower()}
    if parent_names & {"daily_snapshots", "history", "allocations", "holdings", "attachment_text"}:
        return False
    return True


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _write_local_json(path, payload)
    write_json_cache(path, payload)
    if _should_clear_listing_memory_caches_for_path(path):
        _clear_all_fund_memory_caches()


def _write_local_json(path: Path, payload: Dict[str, Any]) -> None:
    json_cache_store.write_local_json(path, payload)
    if _should_clear_listing_memory_caches_for_path(path):
        _clear_all_fund_memory_caches()


def _parse_iso_datetime(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        value = str(raw)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _cache_age_seconds(fetched_at: Any) -> Optional[float]:
    parsed = _parse_iso_datetime(fetched_at)
    if not parsed:
        return None
    return max(0.0, (_utc_now() - parsed).total_seconds())


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if result == result else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    text = text.replace("\xa0", "").replace("%", "").strip()
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        result = float(text)
        return result if result == result else None
    except ValueError:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _first_text(row: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _dedupe_warning_messages(warnings: Iterable[Any]) -> List[str]:
    """Keep warning payloads stable when repeated refreshes report the same error."""

    deduped: List[str] = []
    seen: set[str] = set()
    for warning in warnings:
        text = str(warning or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _first_present(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", text)


def _fund_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        try:
            return datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d").date().isoformat()
        except ValueError:
            return None
    if text.isdigit():
        return _fund_date(int(text))
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _split_date_range(start: date, end: date, chunk_days: int) -> Iterable[Tuple[date, date]]:
    if start > end or chunk_days <= 0:
        return []
    chunks: List[Tuple[date, date]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=chunk_days - 1))
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _web_history_windows(start: date, end: date) -> List[Tuple[date, date]]:
    if start > end:
        return []
    day_span = (end - start).days
    if day_span <= 120:
        return list(_split_date_range(start, end, FUNDS_WEB_HISTORY_CHUNK_DAYS))

    # Long ranges only need recent daily data plus month-end anchors for charts.
    # and monthly heatmaps we only need recent daily data plus month-end anchors.
    windows: List[Tuple[date, date]] = []
    seen: set[Tuple[date, date]] = set()

    def add_window(window_start: date, window_end: date) -> None:
        bounded_start = max(start, window_start)
        bounded_end = min(end, window_end)
        if bounded_start > bounded_end:
            return
        key = (bounded_start, bounded_end)
        if key in seen:
            return
        seen.add(key)
        windows.append(key)

    add_window(end - timedelta(days=35), end)

    current = date(start.year, start.month, 1)
    while current <= end:
        next_month = date(current.year + 1, 1, 1) if current.month == 12 else date(current.year, current.month + 1, 1)
        month_end = min(next_month - timedelta(days=1), end)
        add_window(month_end - timedelta(days=8), month_end)
        current = next_month

    windows.sort(key=lambda item: item[0])
    return windows


_FUND_PRICE_SOURCE_PRIORITY = {
    TEFASFON_FUNDS_SOURCE: 90,
    TEFAS_DIRECT_FUNDS_SOURCE: 85,
    FINTABLES_UDF_HISTORY_SOURCE: 70,
    "legacy_json": 10,
}
_TEFASFON_DAILY_PRICE_SOURCES = (TEFASFON_FUNDS_SOURCE,)
_TEFAS_DIRECT_DAILY_PRICE_SOURCES = (TEFAS_DIRECT_FUNDS_SOURCE,)
_FINTABLES_DAILY_PRICE_SOURCES = (FINTABLES_UDF_HISTORY_SOURCE,)
_DAILY_PRICE_SOURCES = (
    _TEFASFON_DAILY_PRICE_SOURCES
    + _TEFAS_DIRECT_DAILY_PRICE_SOURCES
    + _FINTABLES_DAILY_PRICE_SOURCES
)
_NON_DAILY_PRICE_SOURCES = {FINTABLES_YIELD_SUMMARY_SOURCE, TEFASFON_RETURNS_SOURCE, TEFASFON_PORTFOLIO_SOURCE}


# Maximum allowed gap (in days) between the latest snapshot point and the
# previous local close when back-filling a missing daily_return.  Keeps weekend
# / holiday transitions usable while filtering out long-stale points.
_DAILY_RETURN_LOCAL_FALLBACK_MAX_GAP_DAYS = int(
    os.getenv("RAGFIN_DAILY_RETURN_LOCAL_FALLBACK_MAX_GAP_DAYS", "5")
)


_TURKEY_MARKET_FULL_DAY_HOLIDAYS = {
    # Static full-day TEFAS/BIST holidays used only for history gap validation.
    # Half-days are intentionally excluded; one-day gaps stay below the warning threshold.
    date(2020, 1, 1),
    date(2020, 4, 23),
    date(2020, 5, 1),
    date(2020, 5, 19),
    date(2020, 5, 25),
    date(2020, 5, 26),
    date(2020, 7, 15),
    date(2020, 7, 31),
    date(2020, 8, 3),
    date(2020, 10, 29),
    date(2021, 1, 1),
    date(2021, 4, 23),
    date(2021, 5, 13),
    date(2021, 5, 14),
    date(2021, 5, 19),
    date(2021, 7, 15),
    date(2021, 7, 20),
    date(2021, 7, 21),
    date(2021, 7, 22),
    date(2021, 7, 23),
    date(2021, 8, 30),
    date(2021, 10, 29),
    date(2022, 4, 23),
    date(2022, 5, 2),
    date(2022, 5, 3),
    date(2022, 5, 4),
    date(2022, 5, 19),
    date(2022, 7, 11),
    date(2022, 7, 12),
    date(2022, 7, 15),
    date(2022, 8, 30),
    date(2023, 4, 21),
    date(2023, 5, 1),
    date(2023, 5, 19),
    date(2023, 6, 28),
    date(2023, 6, 29),
    date(2023, 6, 30),
    date(2023, 8, 30),
    date(2024, 1, 1),
    date(2024, 4, 10),
    date(2024, 4, 11),
    date(2024, 4, 12),
    date(2024, 4, 23),
    date(2024, 5, 1),
    date(2024, 6, 17),
    date(2024, 6, 18),
    date(2024, 6, 19),
    date(2024, 7, 15),
    date(2024, 8, 30),
    date(2024, 10, 29),
    date(2025, 1, 1),
    date(2025, 3, 31),
    date(2025, 4, 1),
    date(2025, 4, 23),
    date(2025, 5, 1),
    date(2025, 5, 19),
    date(2025, 6, 6),
    date(2025, 6, 9),
    date(2025, 7, 15),
    date(2025, 10, 29),
    date(2026, 1, 1),
    date(2026, 3, 20),
    date(2026, 4, 23),
    date(2026, 5, 1),
    date(2026, 5, 19),
    date(2026, 5, 27),
    date(2026, 5, 28),
    date(2026, 5, 29),
    date(2026, 7, 15),
    date(2026, 8, 30),
    date(2026, 10, 29),
}


def _is_turkey_market_business_day(day: date) -> bool:
    return day.weekday() < 5 and day not in _TURKEY_MARKET_FULL_DAY_HOLIDAYS


def _previous_turkey_market_business_day(day: date) -> date:
    current = day - timedelta(days=1)
    while not _is_turkey_market_business_day(current):
        current -= timedelta(days=1)
    return current


def _business_days_between(start: date, end: date) -> int:
    if start > end:
        return 0
    days = 0
    current = start
    while current <= end:
        if _is_turkey_market_business_day(current):
            days += 1
        current += timedelta(days=1)
    return days


def _latest_fund_snapshot_target_date(today: Optional[date] = None) -> date:
    current = today or _turkey_market_today()
    # The requested date is the latest Turkish market business day.  TEFAS
    # publication timing is resolved by the bounded probe below, not by a
    # fixed wall-clock cutoff that can permanently hide a newly published
    # close.
    if _is_turkey_market_business_day(current):
        return current
    while not _is_turkey_market_business_day(current):
        current -= timedelta(days=1)
    return current


def _turkey_market_today() -> date:
    """Return the current calendar date in the market's local timezone."""

    return datetime.now(timezone(timedelta(hours=3))).date()


# Re-export KAP holdings client and PDF parser symbols from app.funds.holdings_kap_parser
# after core definitions above to preserve module-level imports and monkeypatch seams.
from app.funds.holdings_kap_parser import (
    KAP_HOLDINGS_SOURCE,
    KAP_HOLDINGS_PARSE_VERSION,
    _KAP_NUMBER_PATTERN,
    _KAP_DATE_PATTERN,
    _KAP_ISIN_TAIL_PATTERN,
    _KAP_BORSA_CONTRACT_TAIL_PATTERN,
    _KAP_MAX_REASONABLE_WEIGHT,
    _KAP_LINE_PREFIX_PATTERNS,
    _KAP_NON_HOLDING_CONTEXT_TOKENS,
    _KAP_POSITION_STOPWORDS,
    _KAP_INCLUDED_HOLDING_TYPES,
    _KAP_FOREIGN_ISIN_PROVIDER_SYMBOLS,
    _KAP_FOREIGN_ISIN_PROVIDER_NAMES,
    _KAP_FOREIGN_SYMBOL_STOPWORDS,
    _KAP_FOREIGN_EXCHANGE_SUFFIXES,
    _KAP_FOREIGN_EQUITY_PREFIX_PATTERN,
    _KAP_ISIN_PATTERN,
    _KAP_NAME_FIRST_CATEGORY_PATTERN,
    _KAP_NAME_FIRST_PAGE_FOOTER_PATTERN,
    _KAP_NAME_FIRST_CATEGORY_WORDS,
    _KAP_NAME_FIRST_US_TICKER_SUFFIXES,
    _KAP_EXTRA_STOCK_SYMBOLS,
    _KAP_BIST_STOCK_SYMBOLS,
    _KAP_TURKISH_MONTH_WORDS,
    _kap_headers,
    _kap_url,
    _kap_request_get,
    _kap_get_json,
    _kap_post_json,
    _kap_get_text,
    _kap_get_bytes,
    _kap_search_fund_metadata,
    _kap_portfolio_subject_oid,
    _kap_disclosure_basic,
    _kap_disclosure_index,
    _parse_kap_datetime,
    _kap_list_portfolio_disclosures,
    _kap_fetch_report_detail,
    _kap_report_basic,
    _kap_report_attachments,
    _kap_download_attachment,
    _extract_kap_pdf_text,
    _month_end_iso,
    _kap_report_date,
    _holdings_report_index,
    _holdings_positions_hash,
    _holdings_meta_with_runtime_cache_flags,
    _holdings_disclosure_check_due,
    _holdings_disclosure_check_meta,
    _cached_holdings_matches_disclosures,
    _cached_holdings_with_disclosure_check,
    _kap_attachment_text_from_cache,
    _kap_source_url,
    _kap_number_matches,
    _coerce_kap_number_text,
    _kap_number_tokens,
    _kap_borsa_contract_tail_match,
    _kap_token_is_inside,
    _kap_weight_candidate_is_valid,
    _kap_extract_weight_candidate,
    _kap_extract_isin,
    _kap_foreign_provider_symbol_for_exchange,
    _kap_foreign_prefixed_security,
    _kap_stock_symbol_set,
    _kap_is_stock_symbol,
    _kap_stock_name_from_cache,
    _kap_fund_name_map,
    _kap_looks_like_equity_symbol,
    _kap_looks_like_local_equity_symbol,
    _kap_holding_name_is_header_noise,
    _kap_looks_like_fund_symbol,
    _kap_looks_like_foreign_symbol,
    _kap_is_included_holding_type,
    _kap_is_included_holding_context,
    _kap_context_is_foreign,
    _kap_with_foreign_context,
    _kap_asset_type_from_context,
    _kap_foreign_provider_symbol,
    _kap_category_marker,
    _kap_strip_inline_category,
    _kap_strip_continuation_prefix,
    _kap_line_starts_position,
    _kap_row_complete,
    _kap_buffer_is_header_noise,
    _kap_buffer_starts_with_fund_symbol,
    _kap_buffer_starts_with_foreign_symbol,
    _kap_buffer_starts_with_local_equity_symbol,
    _kap_line_starts_known_local_equity_symbol,
    _parse_kap_holding_block,
    _kap_aggregation_key,
    _kap_should_sum_duplicate_lots,
    _kap_sum_optional_number,
    _kap_deduplicate_positions,
    _parse_kap_column_holding_row,
    _kap_name_first_ticker_identity,
    _kap_name_first_row_parts,
    _kap_name_first_position,
    _kap_name_first_category_is_continuation,
    _kap_name_first_resolve_category,
    _kap_name_first_is_boilerplate,
    _kap_name_first_line_can_be_name,
    _kap_is_name_first_holdings_table,
    _parse_kap_name_first_holdings_pdf_text,
    _parse_kap_nominal_value_table,
    _parse_kap_holdings_pdf_text,
    _position_key,
    _kap_resolve_fund_code_ocr_variant,
    _normalize_holding_positions_for_response,
    _merge_holding_positions,
    _kap_report_positions,
    _unavailable_holdings_payload,
    _parse_iso_date,
    _holdings_cache_is_stale,
    refresh_fund_holdings,
    get_fund_holdings_payload,
)

# Re-export upstream TEFAS and Fintables provider clients/helpers from app.funds.providers
# after core definitions above to preserve module-level imports and monkeypatch seams.
from app.funds.providers import (
    _unix_timestamp_for_date,
    _looks_like_html_challenge,
    _decode_fintables_json_response,
    _curl_executable,
    _fintables_curl_payload,
    _fintables_curl_cffi_payload,
    _fintables_proxy_payload,
    _decode_tefas_json_response,
    _tefas_retry_after_seconds,
    _tefas_retry_delay_seconds,
    _is_tefas_rate_limit,
    _normalize_tefas_fund_list_payload,
    _normalize_fintables_udf_history_payload,
    _normalize_fintables_yield_summary_payload,
    FintablesClient,
    fetch_fintables_udf_history,
    fetch_fintables_yield_summary,
    _tefasfon_date,
    _tefasfon_adapter_version,
    _clean_dataframe_value,
    _dataframe_records,
    _tefasfon_rows,
    _merge_tefasfon_returns,
    _range_returns_by_code,
    _merge_tefasfon_range_returns,
    _coerce_tefas_percentage,
    _FUND_TAX_RULES,
    _fund_tax_info,
    _merge_tefasfon_management_fees,
    _yield_periods_from_points,
    _yield_periods_from_return_row,
    _is_tefasfon_adapter_unavailable,
    _generate_date_chunks,
    TefasFonClient,
    TefasClient,
)

# Re-export price store symbols from app.funds.price_store
# after core definitions above to preserve module-level imports and monkeypatch seams.
from app.funds.price_store import (
    _init_fund_prices_schema,
    ensure_fund_prices_schema,
    _normalize_history_row,
    _upsert_fund_reference_data,
    _fund_price_storage_row_to_point,
    _normalize_price_source,
    _public_price_source,
    _price_warning_from_row,
    _storage_point_from_row,
    _insert_fund_price_warning,
    upsert_fund_price_points,
    _fund_price_points_cache_key,
    _remove_fund_price_points_cache_entry_locked,
    _purge_expired_fund_price_points_cache_locked,
    _store_fund_price_points_cache_locked,
    _fund_price_points_cache_metrics_snapshot_locked,
    _log_fund_price_points_cache_metrics,
    invalidate_fund_price_points_cache,
    _reset_fund_price_points_cache_for_tests,
    _read_fund_price_points_cached,
    _read_fund_price_points_uncached,
    read_fund_price_points,
    read_latest_fund_price_points,
    read_fund_price_points_range,
    _daily_return_reconciliation_candidates,
    _daily_return_overrides_from_price_history,
    _carry_forward_resolved_daily_returns,
    _missing_daily_return_keys,
    _daily_return_override_values,
    _apply_daily_return_overrides,
    _persist_daily_return_overrides,
    _read_fintables_udf_price_points,
    _read_daily_fund_price_points,
    _read_preferred_fund_price_points,
    _backfill_daily_returns_from_local_prices,
)

# Re-export snapshot store symbols from app.funds.snapshot_store
# after core definitions above to preserve module-level imports and monkeypatch seams.
from app.funds.snapshot_store import (
    _encode_fund_snapshot_present_fields,
    _decode_fund_snapshot_present_fields,
    _fund_snapshot_record,
    _fund_snapshot_insert_sql,
    _fund_snapshot_records,
    _fund_snapshot_summary,
    _stage_fund_snapshot_codes,
    _delete_fund_snapshot_rows_missing_from_stage,
    _snapshot_row_from_record,
    _read_fund_snapshot_rows,
    _read_fund_snapshot_category_returns,
    _copy_legacy_snapshot,
    _snapshot_metadata_from_store,
    ensure_fund_snapshot_store,
    prune_fund_snapshot_rollback_copy,
    _commit_local_fund_snapshot_rows,
    commit_funds_snapshot,
    _daily_snapshot_cache_is_fresh,
    _cached_daily_funds_snapshot,
    _tefas_open_status,
    _is_tefas_open_row,
    _filter_tefas_open_rows,
    _founder_match_key,
    _target_manager_tokens,
    _is_target_fund_row,
    _filter_target_fund_rows,
    _fund_row_aum,
    _meets_min_aum,
    _infer_fund_type_from_name,
    _fund_name_match_candidates,
    _infer_founder_from_fund_name,
    _is_closed_fund,
    _snapshot_is_current_on_market_closure,
    _mark_snapshot_current_for_market_closure,
    _build_snapshot,
    _empty_snapshot_payload,
    _overlay_latest_price_history,
    _catalog_price_overlay_metadata,
    load_funds_snapshot,
    _target_fund_codes_from_snapshot_payload,
    _target_fund_codes_from_env,
    _tefas_open_codes_from_snapshot_payload,
    _snapshot_rows_by_code,
    _enrich_points_from_snapshot,
    _target_fund_codes_for_collection,
    _fetch_fintables_udf_history_for_codes,
    _fetch_tefasfon_history_for_codes,
    _fetch_tefasfon_daily_snapshots_for_codes,
    _merge_refresh_rows_with_existing,
    _legacy_fund_fetch_result,
    _fund_fetch_result,
    _snapshot_as_of_date,
    _add_snapshot_resolution_metadata,
    refresh_funds_snapshot,
    collect_daily_fund_prices,
    _row_matches,
    _sort_rows,
    _query_fund_snapshot_rows,
    _refresh_funds_snapshot_if_stale,
)

# Re-export performance and allocation symbols from app.funds.performance_and_allocations
# after core definitions above to preserve module-level imports and monkeypatch seams.
from app.funds.performance_and_allocations import (
    _normalize_allocation_row,
    _allocation_payload_report_date,
    _promote_latest_allocation_history,
    _return_between,
    _point_on_or_before,
    _period_returns,
    _period_returns_from_raw,
    _history_coverage_info,
    _history_internal_gap_warnings,
    _is_monthly_anchor_transition,
    _history_resolution,
    _history_is_usable_daily_range,
    _fetch_confirmed_tefas_tail,
    _history_needs_detail_fill,
    _dedupe_price_points,
    _dominant_price_source,
    _summary_from_points,
    get_funds_payload,
    get_fund_categories_payload,
    _fund_reference_metadata,
    _find_fund_row,
    _fund_return_rank_value,
    _fund_category_rankings,
    get_fund_detail_payload,
    get_fund_yield_summary_payload,
    _valid_performance_points,
    _legacy_history_points,
    _migrate_legacy_history_to_sqlite,
    _fund_performance_payload_from_points,
    _has_requested_price_coverage,
    _latest_fund_point_date,
    _recent_tail_refresh_target,
    _needs_recent_tail_refresh,
    _history_cache_covers_requested_span,
    _month_start,
    _shift_month,
    _month_end,
    _month_key,
    _fund_point_date,
    _fund_period_stats,
    _overview_metric_targets,
    _latest_point_for_month,
    _has_overview_metrics,
    _recent_detail_window,
    _recent_detail_missing_count,
    _fetch_recent_detail_rows,
    _missing_overview_metric_targets,
    _fetch_fund_overview_metric_rows,
    _auto_fetch_key,
    _recent_auto_fetch_failure,
    _remember_auto_fetch_failure,
    _auto_refresh_fund_performance,
    _auto_refresh_fund_overview_metrics,
    _auto_refresh_fund_recent_details,
    _long_range_metric_targets,
    _fetch_fast_long_fund_history,
    refresh_fund_performance,
    get_fund_performance_payload,
    refresh_fund_allocations,
    refresh_fund_allocations_history,
    get_fund_allocations_history_payload,
    get_fund_allocations_payload,
)
