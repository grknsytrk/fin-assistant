from __future__ import annotations

from app.api_runtime import *
from app.kap_financials import *
from app.kap_service import *

_MARKET_UNIVERSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_UNIVERSE_CACHE_TTL_SECONDS", str(6 * 60 * 60))
)

_MARKET_QUOTES_FRESH_TTL = int(os.getenv("RAGFIN_MARKET_QUOTES_FRESH_TTL_SECONDS", "3"))

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
    summary_shares = _positive_float(summary.get("shares_outstanding"))
    if price_value is not None and summary_shares is not None:
        # Use the live quote with the provider's share count; its cached market
        # cap may lag behind the live price.
        return price_value * summary_shares

    market_cap = _positive_float(summary.get("market_cap"))
    if market_cap is not None:
        return market_cap
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
    try:
        financial_snapshots = get_financial_snapshots_map(CONFIG.paths.processed_dir, symbols)
    except Exception as exc:
        LOGGER.warning("financial snapshot batch lookup failed: %s", exc)
        financial_snapshots = {}
    rows: List[Dict[str, Any]] = []
    kap_cache_count = 0

    for symbol in symbols:
        normalized_symbol = str(symbol or "").strip().upper()
        instrument = instrument_map.get(normalized_symbol) or {}
        instrument_metadata = instrument.get("metadata") if isinstance(instrument.get("metadata"), dict) else {}
        cached_meta = financial_snapshots.get(normalized_symbol) or {}
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

_PRICE_CACHE: Dict[str, Any] = {}

_PRICE_CACHE_TTL = 300

_MARKET_PRICE_CACHE: Dict[str, Any] = {}

_MARKET_PRICE_CACHE_TTL = 3

_MARKET_QUOTE_SOURCE_INDEX = "XUTUM"

_INFOYATIRIM_STOCK_PAGE_CACHE: Dict[str, Any] = {}

_INFOYATIRIM_STOCK_PAGE_CACHE_TTL = 60

_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT = 12

_STOCKS_CACHE: Dict[str, Any] = {}

_STOCKS_CACHE_TTL = 3

_MARKET_STOCK_CARD_CHART_CACHE: Dict[str, Any] = {}

_MARKET_STOCK_CARD_CHART_CACHE_TTL = 45

_MARKET_STOCK_CARD_LIMIT = 12

_MARKET_STOCK_CARDS_QUICK_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_STOCK_CARDS_QUICK_RESPONSE_CACHE_TTL_SECONDS", "3")
)

_MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL = int(
    os.getenv("RAGFIN_MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL_SECONDS", "3")
)

_MARKET_STOCK_CARD_PREVIOUS_SESSION_LOOKBACK_DAYS = 10

_STOCK_CARD_VALUATION_CACHE: Dict[str, Any] = {}

_STOCK_CARD_VALUATION_CACHE_TTL = int(os.getenv("RAGFIN_STOCK_CARD_VALUATION_CACHE_TTL_SECONDS", str(6 * 60 * 60)))

_TURKEY_TIMEZONE = timezone(timedelta(hours=3))

_BIST_EQUITY_SESSION_OPEN_MINUTE = (9 * 60) + 40

_BIST_EQUITY_SESSION_CLOSE_MINUTE = (18 * 60) + 10

_MARKET_STOCK_CARD_CHART_RANGES: Dict[str, Dict[str, Any]] = {
    "1d": {"interval": "5m", "range": "1d", "ttl": 30},
    "1w": {"interval": "15m", "range": "5d", "ttl": 60},
    "1m": {"interval": "4h", "range": "1mo", "ttl": 600},
    "1y": {"interval": "1d", "range": "1y", "ttl": 3600},
}

_STOCK_RETURN_BASE_CACHE: Dict[str, Any] = {}

_STOCK_RETURN_BASE_CACHE_TTL = 900

_STOCK_RETURN_BASE_CACHE_VERSION = "v2"

_ISYATIRIM_CACHE: Dict[str, Any] = {}

_ISYATIRIM_CACHE_TTL = 900

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

def _fetch_market_price_map(
    symbols: List[str],
    *,
    index_name: str = _MARKET_QUOTE_SOURCE_INDEX,
) -> Dict[str, Dict[str, Any]]:
    """Fetch one canonical InfoYatirim quote snapshot and project its symbols.

    Market lists and stock cards must not receive different prices because they
    requested different symbol subsets or index views.  The XUTUM live page
    contains the complete BIST equity universe, so it is used as the single
    quote source and cache snapshot for every consumer.
    """
    import urllib.error
    import urllib.request

    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not normalized_symbols:
        return {}

    # Keep the public argument for callers that select XU100/XU030 rows, but
    # deliberately use the same upstream page and cache entry for all of them.
    del index_name
    source_index = _MARKET_QUOTE_SOURCE_INDEX
    cache_key = f"snapshot:{source_index}"

    def project_items(items: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            symbol: dict(items[symbol])
            for symbol in normalized_symbols
            if symbol in items
        }

    now = time.time()
    cached = _MARKET_PRICE_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _MARKET_PRICE_CACHE_TTL:
        return project_items(cached.get("items", {}))

    url = _market_price_source_url(source_index)

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
    if items and missing_symbols:
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
    return project_items(items)

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

def _stock_return_base_cache_key(symbol: str) -> str:
    return f"api:stock-return-bases:{symbol}:{_STOCK_RETURN_BASE_CACHE_VERSION}"

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
    shared_key = _stock_return_base_cache_key(ticker)
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached:
        _STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached)

    yahoo_symbol = f"{ticker}.IS"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        # A one-year response can start on the first trading day after the
        # exact 365-day boundary, leaving no base price at-or-before it.
        # Two years keeps the displayed return truly one-year while providing
        # enough history to select the preceding trading session.
        "?interval=1d&range=2y"
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
    _shared_cache_set(shared_key, bases, _STOCK_RETURN_BASE_CACHE_TTL)
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
        shared_cached = _shared_cache_get_dict(_stock_return_base_cache_key(symbol))
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
        shared_cached = _shared_cache_get_dict(_stock_return_base_cache_key(normalized))
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
    market_cap = _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary)
    provider = basic_summary or {}
    company_kind = str(cached_meta.get("company_kind") or "generic").lower()
    if company_kind == "generic":
        from src.kap_fetcher import classify_kap_company_kind

        company_kind = classify_kap_company_kind(symbol, cached_meta.get("company_title"))
    financial_snapshot = {**cached_meta, "company_kind": company_kind}
    valuation = _resolve_market_card_valuation_from_cached_data(
        {
            "financial_snapshot": financial_snapshot,
            "provider_ratios": {
                "fk": provider.get("fk"),
                "pd_dd": provider.get("pd_dd"),
                "fd_favok": provider.get("fd_favok"),
            },
            "provider_source": provider.get("source") or "isyatirim_temelfinansal",
            "as_of": cached_meta.get("as_of") or provider.get("as_of"),
        },
        market_cap=market_cap,
    )
    ttm_sales = _as_finite_float(cached_meta.get("ttm_satis_gelirleri"))
    fd_satis = (
        _round_market_ratio(valuation.get("enterprise_value"), ttm_sales)
        if company_kind not in {"bank", "insurance"}
        else None
    )
    if _is_missing_market_ratio(fd_satis) and company_kind not in {"bank", "insurance"}:
        provider_fd_satis = _parse_tr_decimal(provider.get("fd_satis"))
        if not _is_missing_market_ratio(provider_fd_satis):
            fd_satis = provider_fd_satis
    free_float_pct = _as_finite_float(cached_meta.get("free_float_pct"))
    if free_float_pct is None:
        free_float_pct = _as_finite_float(provider.get("free_float_pct"))
    if free_float_pct is None:
        fdpo = _as_finite_float(provider.get("fdpo"))
        free_float_pct = fdpo * 100.0 if fdpo is not None else None
    net_borc_favok = _first_not_none(
        valuation.get("net_borc_favok"),
        _as_finite_float(cached_meta.get("net_borc_favok")),
    )
    ttm_favok = _as_finite_float(cached_meta.get("ttm_favok"))
    if company_kind in {"bank", "insurance"} or (ttm_favok is not None and ttm_favok <= 0):
        # Leverage is not meaningful for financial companies or when EBITDA is
        # zero or negative.
        net_borc_favok = None
    return {
        "company": symbol,
        "name": cached_meta.get("company_title") or symbol,
        "latest_quarter": latest_quarter,
        "fundamental_period": latest_quarter or cached_meta.get("provider_period") or provider.get("provider_period"),
        "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
        "price": quote.get("price"),
        "price_currency": quote.get("currency"),
        "change": quote.get("change"),
        "change_pct": quote.get("change_pct"),
        "price_as_of": quote.get("as_of"),
        "volume": quote.get("volume"),
        "market_cap": market_cap,
        "sector": cached_meta.get("sector") or provider.get("sector"),
        "fk": valuation.get("fk"),
        "pd_dd": valuation.get("pd_dd"),
        "fd_favok": valuation.get("fd_favok"),
        "fd_satis": fd_satis,
        "net_borc_favok": net_borc_favok,
        "roe_pct": _as_finite_float(cached_meta.get("roe_pct")),
        "favok_margin_pct": _as_finite_float(cached_meta.get("favok_margin_pct")),
        "net_margin_pct": _as_finite_float(cached_meta.get("net_margin_pct")),
        "free_float_pct": free_float_pct,
        "ttm_net_kar": _as_finite_float(cached_meta.get("ttm_net_kar")),
        "ttm_satis_gelirleri": _as_finite_float(cached_meta.get("ttm_satis_gelirleri")),
        "ttm_favok": ttm_favok,
        "ttm_fcf": _as_finite_float(cached_meta.get("serbest_nakit_akisi")),
        "ozkaynaklar": _as_finite_float(cached_meta.get("ozkaynaklar")),
        "net_borc": _as_finite_float(cached_meta.get("net_borc")),
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
    price_map = _fetch_market_price_map(symbols, index_name=normalized_index)
    quote_as_of = next(
        (
            quote.get("as_of")
            for quote in price_map.values()
            if isinstance(quote, dict) and quote.get("as_of")
        ),
        None,
    )
    return_base_map = (
        _cached_stock_return_bases_bulk(symbols)
        if normalized_index == "XUTUM"
        else _fetch_stock_return_bases_bulk(symbols)
    )
    basic_summary_map = _fetch_isyatirim_basic_summary_map()
    try:
        financial_snapshot_map = get_financial_snapshots_map(CONFIG.paths.processed_dir, symbols)
    except Exception as exc:
        LOGGER.warning("market stock financial snapshot lookup failed: %s", exc)
        financial_snapshot_map = {}
    rows = [
        _market_stock_row(
            symbol,
            cached_meta=financial_snapshot_map.get(str(symbol or "").strip().upper(), {}),
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
        # The quote timestamp is the shared live snapshot timestamp used by
        # both the market list and the stock cards.
        "as_of": quote_as_of or datetime.now(timezone.utc).isoformat(),
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

_BIST_STREAM_OPEN_MINUTE = (9 * 60) + 50

_BIST_STREAM_CLOSE_MINUTE = (18 * 60) + 20

_MARKET_QUOTES_OFFHOURS_FRESH_TTL = int(
    os.getenv("RAGFIN_MARKET_QUOTES_OFFHOURS_FRESH_TTL_SECONDS", "300")
)

_MARKET_STOCK_DELTA_TRACKED_KEYS = (
    "price", "change", "change_pct", "volume", "market_cap",
    "latest_quarter", "fundamental_period", "sector", "has_kap_cache",
    "fk", "pd_dd", "fd_favok", "fd_satis", "net_borc_favok", "roe_pct",
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
    "latest_quarter",
    "fundamental_period",
    "has_kap_cache",
    "sector",
    "fk",
    "pd_dd",
    "fd_favok",
    "fd_satis",
    "net_borc_favok",
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

def _is_bist_market_open(now: Optional[datetime] = None) -> bool:
    """Return True during BIST equity trading + auction window (Mon-Fri 09:50-18:20 TRT)."""
    current = (now or datetime.now(_TURKEY_TIMEZONE)).astimezone(_TURKEY_TIMEZONE)
    if current.weekday() >= 5:
        return False
    minute_of_day = (current.hour * 60) + current.minute
    return _BIST_STREAM_OPEN_MINUTE <= minute_of_day <= _BIST_STREAM_CLOSE_MINUTE

def _effective_market_quotes_fresh_ttl(now: Optional[datetime] = None) -> int:
    """Keep 3s quote freshness during BIST trading hours and extend to 300s off-hours."""
    if _is_bist_market_open(now):
        return max(1, int(_MARKET_QUOTES_FRESH_TTL))
    return max(1, int(_MARKET_QUOTES_FRESH_TTL), int(_MARKET_QUOTES_OFFHOURS_FRESH_TTL))

def _compute_market_stocks_delta(
    prev_snapshot: Dict[str, tuple[Any, ...]],
    rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, tuple[Any, ...]]]:
    """Return stock rows whose quote or screener fundamentals changed."""
    changed: List[Dict[str, Any]] = []
    next_snapshot: Dict[str, tuple[Any, ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("company") or row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        sig = tuple(row.get(k) for k in _MARKET_STOCK_DELTA_TRACKED_KEYS)
        next_snapshot[symbol] = sig
        if prev_snapshot and prev_snapshot.get(symbol) == sig:
            continue
        delta_item: Dict[str, Any] = {"company": symbol}
        for key in _MARKET_STOCK_DELTA_PAYLOAD_KEYS:
            fundamental_key = key in {
                "latest_quarter", "fundamental_period", "has_kap_cache", "sector", "fk", "pd_dd", "fd_favok",
                "fd_satis", "net_borc_favok", "roe_pct", "favok_margin_pct", "net_margin_pct",
                "free_float_pct", "ttm_net_kar", "ttm_satis_gelirleri", "ttm_favok",
                "ttm_fcf", "ozkaynaklar", "net_borc",
            }
            if key in row and (row[key] is not None or fundamental_key):
                delta_item[key] = row[key]
        changed.append(delta_item)
    return changed, next_snapshot

def _market_stocks_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    normalized_index = _normalize_stock_index(index_name)
    started_at = time.perf_counter()
    local_key = f"payload:{normalized_index}"
    shared_key = f"api:market:stocks:{normalized_index}:v4"
    effective_fresh_ttl = _effective_market_quotes_fresh_ttl()
    effective_stale_ttl = max(effective_fresh_ttl, _MARKET_QUOTES_STALE_TTL)

    def revalidate_factory() -> Optional[Dict[str, Any]]:
        built = _build_market_stocks_payload(index_name=normalized_index, force_refresh=True)
        rows = built.get("rows") if isinstance(built.get("rows"), list) else []
        has_quote = not rows or any(row.get("price") is not None for row in rows if isinstance(row, dict))
        return built if has_quote else None

    refresh_pending = False
    if force_refresh:
        refresh_pending = _schedule_swr_revalidation(
            cache_key=shared_key,
            fresh_ttl_seconds=effective_fresh_ttl,
            stale_ttl_seconds=effective_stale_ttl,
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
            "fresh_until": now + max(1, effective_fresh_ttl),
            "stale_until": now + max(1, effective_stale_ttl),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

    entry, cache_status = _get_or_set_single_flight(
        shared_key,
        ttl_seconds=max(1, effective_stale_ttl),
        factory=build_and_envelope,
        lock_key=f"api:market:stocks:{normalized_index}:lock:v4",
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


def market_stocks_etag_revision(*, index_name: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the generation of the cached stock representation, if available."""

    normalized_index = _normalize_stock_index(index_name)
    local_key = f"payload:{normalized_index}"
    shared_key = f"api:market:stocks:{normalized_index}:v4"
    local_entry = _STOCKS_CACHE.get(local_key, {}).get("entry")
    shared_entry = _shared_cache_get_dict(shared_key)
    source_revision = None
    for entry in (local_entry, shared_entry):
        if not isinstance(entry, dict):
            continue
        source_revision = entry.get("cached_at")
        if source_revision:
            break
        cached_payload = entry.get("payload")
        if isinstance(cached_payload, dict) and cached_payload.get("as_of"):
            source_revision = cached_payload.get("as_of")
            break
    if not source_revision:
        return None

    # Rows are tied to the cache generation above. The remaining response
    # fields can change independently (for example stale/refresh state), so
    # include them in the validator without hashing the large row array.
    visible_state = {key: value for key, value in payload.items() if key != "rows"}
    return {
        "representation": "market-stocks-v2",
        "index": normalized_index,
        "source_revision": source_revision,
        "visible_state": visible_state,
    }

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

def _normalize_market_stock_card_tier(tier: str) -> str:
    normalized = str(tier or "full").strip().lower()
    if normalized not in {"quick", "full"}:
        raise HTTPException(status_code=400, detail="Gecersiz hisse karti veri seviyesi: quick veya full")
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

def _stock_cards_response_cache_key(symbols: List[str], tier: str = "full") -> str:
    normalized = ",".join(sorted(symbols))
    return f"api:market:stock-cards:symbols={normalized}:tier={tier}:v3"

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
    try:
        stored_snapshot = get_financial_snapshots_map(
            CONFIG.paths.processed_dir,
            [normalized_symbol],
        ).get(normalized_symbol)
    except Exception as exc:
        LOGGER.warning("stock-card financial snapshot lookup failed for %s: %s", normalized_symbol, exc)
        stored_snapshot = None
    if stored_snapshot and stored_snapshot.get("has_kap_cache"):
        return {
            "symbol": normalized_symbol,
            "latest_quarter": stored_snapshot.get("latest_quarter"),
            "quarter_count": stored_snapshot.get("quarter_count"),
            "ttm_net_kar": stored_snapshot.get("ttm_net_kar"),
            "ttm_satis_gelirleri": stored_snapshot.get("ttm_satis_gelirleri"),
            "ttm_favok": stored_snapshot.get("ttm_favok"),
            "company_kind": stored_snapshot.get("company_kind") or "generic",
            "ozkaynaklar": stored_snapshot.get("ozkaynaklar"),
            "net_borc": stored_snapshot.get("net_borc"),
            "source": "financial_snapshot_sql",
            "as_of": stored_snapshot.get("as_of"),
        }
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
        shared_key = f"api:kap:financial-snapshot:{normalized_symbol}:mtime={stat.st_mtime_ns}:size={stat.st_size}:v2"
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

    from app.kap_service import normalize_snapshot_for_frontend

    normalized = normalize_snapshot_for_frontend(payload, allow_network=False)
    normalized_quarters = [
        quarter for quarter in normalized.get("quarters", []) if isinstance(quarter, dict)
    ]
    quarters_sorted = sorted(normalized_quarters, key=_quarter_sort_key)
    latest = quarters_sorted[-1]
    ttm_net_kar = _build_ttm_from_reports(quarters_sorted, "net_kar")
    ttm_favok = _build_ttm_from_reports(quarters_sorted, "favok")
    ozkaynaklar = _extract_quarter_metric(latest, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
    net_borc = _extract_quarter_metric(latest, "net_borc", priority=["metrics", "metrics_ytd"])
    latest_quarter = str(latest.get("quarter") or "").strip().upper() or None
    from src.kap_fetcher import classify_kap_company_kind
    snapshot = {
        "symbol": str(payload.get("company") or payload.get("stock_code") or normalized_symbol or "").strip().upper(),
        "latest_quarter": latest_quarter,
        "quarter_count": len(quarters_sorted),
        "ttm_net_kar": ttm_net_kar,
        "ttm_favok": ttm_favok,
        "company_kind": payload.get("company_kind") or classify_kap_company_kind(normalized_symbol, payload.get("company_title")),
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
    financial_company = snapshot.get("company_kind") in {"bank", "insurance"}
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
        "fd_favok": None if financial_company else _round_market_ratio(enterprise_value, ttm_favok),
        "net_borc_favok": None if financial_company else _round_market_ratio(net_borc, ttm_favok),
        "enterprise_value": None if financial_company else enterprise_value,
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

    if snapshot.get("company_kind") in {"bank", "insurance"}:
        values["fd_favok"] = None
        sources.pop("fd_favok", None)
        net_borc_favok = None

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
    shared_key = f"api:stock-card-valuation:{ticker}:v3"
    shared_cached = _shared_cache_get_dict(shared_key)
    if shared_cached:
        _STOCK_CARD_VALUATION_CACHE[ticker] = {**shared_cached, "_ts": now}
        return _resolve_market_card_valuation_from_cached_data(
            {**shared_cached, "_cache_hit": True},
            market_cap=market_cap,
        )

    snapshot = _stock_card_financial_snapshot_from_cache(ticker)
    from src.kap_fetcher import classify_kap_company_kind
    snapshot.setdefault("company_kind", classify_kap_company_kind(ticker))
    computed = _stock_card_financial_ratios_from_snapshot(snapshot, market_cap=market_cap)
    ratio_keys = ("fk", "pd_dd") if snapshot.get("company_kind") in {"bank", "insurance"} else ("fk", "pd_dd", "fd_favok")
    needs_provider = any(_is_missing_market_ratio(computed.get(key)) for key in ratio_keys)

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

def _market_stock_cards_quick_payload(
    normalized_symbols: List[str],
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Return the quote-first portion of stock cards without slow enrichments."""
    response_cache_key = _stock_cards_response_cache_key(normalized_symbols, tier="quick")
    if not force_refresh:
        shared_cached = _shared_cache_get_dict(response_cache_key)
        if shared_cached is not None:
            return dict(shared_cached)

    price_map = _fetch_market_price_map(normalized_symbols)
    quote_as_of = next(
        (
            quote.get("as_of")
            for quote in price_map.values()
            if isinstance(quote, dict) and quote.get("as_of")
        ),
        None,
    )
    instrument_map = get_instruments(CONFIG.paths.processed_dir, "stock", normalized_symbols)
    items: List[Dict[str, Any]] = []
    for symbol in normalized_symbols:
        quote = price_map.get(symbol, {})
        price = quote.get("price")
        volume = quote.get("volume")
        items.append({
            "symbol": symbol,
            "company": str((instrument_map.get(symbol) or {}).get("name") or symbol).strip() or symbol,
            "yahoo_symbol": f"{symbol}.IS",
            "price": price,
            "currency": quote.get("currency") or "TRY",
            "change": quote.get("change"),
            "change_pct": quote.get("change_pct"),
            "volume": volume,
            "volume_lot": None,
            "volume_tl": volume,
            "market_cap": quote.get("market_cap"),
            "high": None,
            "low": None,
            "previous_close": None,
            "fk": None,
            "pd_dd": None,
            "fd_favok": None,
            "net_borc_favok": None,
            "market_state": quote.get("market_state") or "",
            "as_of": quote.get("as_of"),
            "session_status": "unknown",
            "session_label": "Detaylar hazırlanıyor",
            "is_live": False,
            "is_stale": False,
            "last_trade_at": None,
            "last_trade_date": None,
            "line_points": [],
            "error": None if price is not None else "quote_unavailable",
            "logo_url": (instrument_map.get(symbol) or {}).get("logo_url"),
            "logo_source": (instrument_map.get(symbol) or {}).get("logo_source"),
            "card_stage": "quick",
            "return_1w_pct": None,
            "return_1m_pct": None,
            "return_3m_pct": None,
            "return_6m_pct": None,
            "return_ytd_pct": None,
            "return_1y_pct": None,
        })

    data = {
        "items": items,
        "source": "infoyatirim_live_quote",
        # The quote timestamp is the shared live snapshot timestamp used by
        # both the market list and the stock cards.
        "as_of": quote_as_of or datetime.now(timezone.utc).isoformat(),
    }
    _shared_cache_set(
        response_cache_key,
        data,
        ttl_seconds=_MARKET_STOCK_CARDS_QUICK_RESPONSE_CACHE_TTL,
    )
    return data

def _market_stock_cards_payload(
    *,
    symbols: str,
    force_refresh: bool = False,
    tier: str = "full",
) -> Dict[str, Any]:
    normalized_tier = _normalize_market_stock_card_tier(tier)
    normalized_symbols = _normalize_market_stock_card_symbols(symbols)
    if not normalized_symbols:
        return {
            "items": [],
            "source": "infoyatirim_live_quote" if normalized_tier == "quick" else "infoyatirim_yahoo_chart",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    if normalized_tier == "quick":
        return _market_stock_cards_quick_payload(normalized_symbols, force_refresh=force_refresh)

    response_cache_key = _stock_cards_response_cache_key(normalized_symbols, tier="full")
    if not force_refresh:
        shared_cached = _shared_cache_get_dict(response_cache_key)
        if shared_cached is not None:
            return dict(shared_cached)

    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    price_map = _fetch_market_price_map(normalized_symbols)
    quote_as_of = next(
        (
            quote.get("as_of")
            for quote in price_map.values()
            if isinstance(quote, dict) and quote.get("as_of")
        ),
        None,
    )
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
        # Price, change and volume all come from the canonical live quote;
        # keep its timestamp even when Yahoo supplies the chart/session data.
        as_of = quote.get("as_of") or intraday.get("as_of")
        item = {
            "symbol": symbol,
            "card_stage": "full",
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
        "as_of": quote_as_of or datetime.now(timezone.utc).isoformat(),
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
    items: Dict[str, Dict[str, Any]] = {}
    table_flags = re.IGNORECASE | re.DOTALL

    def rows_for_table(csv_name: str) -> List[str]:
        table_match = re.search(
            rf'<table\b(?=[^>]*\bdata-csvname\s*=\s*["\']{re.escape(csv_name)}["\'])[^>]*>(.*?)</table>',
            html_text,
            flags=table_flags,
        )
        return (
            re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_match.group(1), flags=table_flags)
            if table_match
            else []
        )

    def parse_cells(row_html: str, tag: str) -> List[str]:
        return [
            _strip_html_cell(cell)
            for cell in re.findall(rf"<{tag}\b[^>]*>(.*?)</{tag}\s*>", row_html, flags=table_flags)
        ]

    # The overview table is already fetched for market-cap/share-count fallback.
    overview_header: Dict[str, int] = {}
    for row_html in rows_for_table("temelozet"):
        headers = parse_cells(row_html, "th")
        if headers:
            for idx, header in enumerate(headers):
                norm = _norm_text(header)
                if norm == "KOD":
                    overview_header["symbol"] = idx
                elif norm == "SEKTOR":
                    overview_header["sector"] = idx
                elif "PIYASA DEGERI" in norm and "MN TL" in norm:
                    overview_header["market_cap_mn_try"] = idx
                elif "HALKA ACIKLIK" in norm:
                    overview_header["free_float_pct"] = idx
                elif norm.startswith("SERMAYE") and "MN TL" in norm:
                    overview_header["capital_mn_try"] = idx
            continue

        cells = parse_cells(row_html, "td")
        if not cells:
            continue
        symbol_idx = overview_header.get("symbol", 0)
        if symbol_idx >= len(cells):
            continue
        symbol = cells[symbol_idx].strip().upper()
        if not symbol:
            continue
        market_cap_mn_try = _parse_tr_decimal(cells[overview_header.get("market_cap_mn_try", 4)]) if overview_header.get("market_cap_mn_try", 4) < len(cells) else None
        free_float_pct = _parse_tr_decimal(cells[overview_header.get("free_float_pct", 6)]) if overview_header.get("free_float_pct", 6) < len(cells) else None
        capital_mn_try = _parse_tr_decimal(cells[overview_header.get("capital_mn_try", 7)]) if overview_header.get("capital_mn_try", 7) < len(cells) else None
        sector_idx = overview_header.get("sector", 2)
        items[symbol] = {
            "market_cap": market_cap_mn_try * 1_000_000 if market_cap_mn_try is not None else None,
            "fdpo": round(free_float_pct / 100.0, 6) if free_float_pct is not None and free_float_pct >= 0 else None,
            "free_float_pct": free_float_pct,
            "shares_outstanding": capital_mn_try * 1_000_000 if capital_mn_try is not None else None,
            "sector": cells[sector_idx].strip() or None if sector_idx < len(cells) else None,
            "source": "isyatirim_temelozet",
        }

    def normalized_period(raw: str) -> Optional[str]:
        text = str(raw or "").strip().upper()
        quarter_match = re.fullmatch(r"(\d{4})\s*Q([1-4])", text)
        if quarter_match:
            return f"{quarter_match.group(1)}Q{quarter_match.group(2)}"
        match = re.fullmatch(r"(\d{1,2})[./-](\d{4})", text)
        if match:
            month, year = int(match.group(1)), int(match.group(2))
        else:
            match = re.fullmatch(r"(\d{4})[./-](\d{1,2})", text)
            if not match:
                return None
            year, month = int(match.group(1)), int(match.group(2))
        if year < 1900 or month not in {3, 6, 9, 12}:
            return None
        return f"{year}Q{(month + 2) // 3}"

    financial_header: Dict[str, int] = {}
    for row_html in rows_for_table("temelfinansal"):
        headers = parse_cells(row_html, "th")
        if headers:
            for idx, header in enumerate(headers):
                norm = _norm_text(header).replace(" ", "")
                if norm == "KOD":
                    financial_header["symbol"] = idx
                elif norm in {"F/K", "FK"}:
                    financial_header["fk"] = idx
                elif "FD/FAVOK" in norm:
                    financial_header["fd_favok"] = idx
                elif "FD/SATIS" in norm:
                    financial_header["fd_satis"] = idx
                elif norm in {"PD/DD", "PDDD"}:
                    financial_header["pd_dd"] = idx
                elif "SONDONEM" in norm:
                    financial_header["provider_period"] = idx
            continue

        cells = parse_cells(row_html, "td")
        if not cells:
            continue
        symbol_idx = financial_header.get("symbol", 0)
        if symbol_idx >= len(cells):
            continue
        symbol = cells[symbol_idx].strip().upper()
        if not symbol:
            continue
        indexes = {
            "fk": financial_header.get("fk", 2),
            "fd_favok": financial_header.get("fd_favok", 3),
            "fd_satis": financial_header.get("fd_satis", 4),
            "pd_dd": financial_header.get("pd_dd", 5),
            "provider_period": financial_header.get("provider_period", 6),
        }
        record = items.setdefault(symbol, {"source": "isyatirim_temelfinansal"})
        record.update({
            "fk": _parse_tr_decimal(cells[indexes["fk"]]) if indexes["fk"] < len(cells) else None,
            "fd_favok": _parse_tr_decimal(cells[indexes["fd_favok"]]) if indexes["fd_favok"] < len(cells) else None,
            "fd_satis": _parse_tr_decimal(cells[indexes["fd_satis"]]) if indexes["fd_satis"] < len(cells) else None,
            "pd_dd": _parse_tr_decimal(cells[indexes["pd_dd"]]) if indexes["pd_dd"] < len(cells) else None,
            "provider_period": normalized_period(cells[indexes["provider_period"]]) if indexes["provider_period"] < len(cells) else None,
            "provider_source": "isyatirim_temelfinansal",
        })

    return items

def _fetch_isyatirim_basic_summary_map() -> Dict[str, Dict[str, Any]]:
    import urllib.error
    import urllib.request

    now = time.time()
    cached = _ISYATIRIM_BASIC_SUMMARY_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _ISYATIRIM_BASIC_SUMMARY_CACHE_TTL:
        return cached.get("items", {})
    shared_cached = _shared_cache_get_dict("api:isyatirim-basic-summary:v2")
    if shared_cached:
        items = dict(shared_cached.get("items") or {})
        _ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": items}
        try:
            upsert_provider_fundamentals_bulk(CONFIG.paths.processed_dir, items)
        except Exception:
            LOGGER.exception("shared İş Yatırım fundamentals could not be persisted")
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
    _shared_cache_set("api:isyatirim-basic-summary:v2", {"items": items}, _ISYATIRIM_BASIC_SUMMARY_CACHE_TTL)
    try:
        upsert_provider_fundamentals_bulk(CONFIG.paths.processed_dir, items)
    except Exception:
        LOGGER.exception("İş Yatırım fundamentals could not be persisted")
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

_XU030_CACHE: Dict[str, Any] = {}

_XU030_CACHE_TTL = 120

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

_COMMODITY_CACHE: Dict[str, Any] = {}

_COMMODITY_CACHE_TTL = 3

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

    splits: List[Dict[str, Any]] = []
    raw_events = result.get("events")
    raw_splits = (raw_events.get("splits") or {}) if isinstance(raw_events, dict) else {}
    for event in raw_splits.values():
        if not isinstance(event, dict):
            continue
        try:
            event_date = datetime.fromtimestamp(float(event.get("date")), tz=timezone.utc).date()
            numerator = float(event.get("numerator"))
            denominator = float(event.get("denominator"))
            ratio = numerator / denominator if denominator else None
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            continue
        if ratio is not None and math.isfinite(ratio) and ratio > 0:
            splits.append({"date": event_date.isoformat(), "ratio": ratio})

    return {
        "ok": True,
        "yahoo_symbol": yahoo_symbol,
        "meta": meta,
        "points": points,
        "splits": sorted(splits, key=lambda split: split["date"]),
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
        f"?interval={interval}&period1={period1}&period2={period2}&events=splits"
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

__all__ = [name for name in globals() if not name.startswith("__")]
