from __future__ import annotations

import copy
import html
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from app.database import read_json_cache, write_json_cache
from app.json_cache import read_json_cache_local_first, write_local_json

if TYPE_CHECKING:  # pragma: no cover
    from src.config import KapConfig

KAP_BASE_URL = "https://www.kap.org.tr/tr/api"
MEMBER_FILTER_ENDPOINT = "member/filter"
LIST_COMPANY_EXCEL_MEMBERS_ENDPOINT = "financialTable/listCompanyExcelMembers"
ATTACHMENT_DETAIL_ENDPOINT = "notification/attachment-detail"
PDF_ENDPOINT = "BildirimPdf"
DISCLOSURE_MEMBERS_BY_CRITERIA_ENDPOINT = "disclosure/members/byCriteria"
FILE_DOWNLOAD_ENDPOINT = "file/download"
KAP_CACHE_SCHEMA_VERSION = 15
KAP_LIVE_DISCLOSURE_CHECK_TTL_HOURS = 24.0
KAP_INSURANCE_PREMIUM_CHECK_TTL_HOURS = 24.0

# BIST'te kayıtlı ticari banka tickerleri. KAP banka raporlarında bilanço
# kalemleri için solo + konsolide sütunları yan yana basıldığından, bu
# tickerlarda konsolide sütunu önceliklendiriyoruz.
_KAP_BANK_TICKER_HINTS = frozenset(
    {
        "AKBNK",
        "ALBRK",
        "GARAN",
        "HALKB",
        "ICBCT",
        "ISCTR",
        "QNBFB",
        "SKBNK",
        "TSKB",
        "VAKBN",
        "YKBNK",
    }
)
_KAP_INSURANCE_TICKER_HINTS = frozenset(
    {
        "AGESA",
        "ANHYT",
        "AKGRT",
        "ANSGR",
        "RAYSG",
        "TURSG",
    }
)
KAP_INSURANCE_PREMIUM_CACHE_VERSION = 8
KAP_INSURANCE_PREMIUM_DISCLOSURE_LOOKBACK_DAYS = 1095
KAP_INSURANCE_PREMIUM_MAX_DISCLOSURES = 36
KAP_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
KAP_RETRYABLE_HTTP_CODES = {403, 429, 500, 502, 503, 504}

# Query aliases to improve company resolution against KAP ticker/search behavior.
COMPANY_QUERY_ALIASES: Dict[str, List[str]] = {
    "BIM": ["BIMAS", "BIM"],
    "BIMAS": ["BIMAS", "BIM"],
    "MIGROS": ["MGROS", "MIGROS"],
    "MGROS": ["MGROS", "MIGROS"],
    "SOK": ["SOKM", "SOK"],
    "SOKM": ["SOKM", "SOK"],
    "TAV": ["TAVHL", "TAV"],
    "TAVHL": ["TAVHL", "TAV"],
    "NETCAD": ["NETCD", "NETCAD"],
    "NETCD": ["NETCD", "NETCAD"],
    "ANHYT": ["ANHYT", "ANADOLU HAYAT EMEKLILIK", "ANADOLU HAYAT"],
    "RAYSG": ["RAYSG", "RAY SIGORTA", "RAY SİGORTA"],
    # BIST-30 core aliases (ticker <-> common short name)
    "AKBANK": ["AKBNK", "AKBANK"],
    "AKBNK": ["AKBNK", "AKBANK"],
    "ASELSAN": ["ASELS", "ASELSAN"],
    "ASELS": ["ASELS", "ASELSAN"],
    "EKGYO": ["EKGYO", "EMLAK KONUT"],
    "ENKA": ["ENKAI", "ENKA"],
    "ENKAI": ["ENKAI", "ENKA"],
    "EREGL": ["EREGL", "ERDEMIR", "EREGLI"],
    "GARAN": ["GARAN", "GARANTI"],
    "ISCTR": ["ISCTR", "IS BANKASI", "ISBANK"],
    "KCHOL": ["KCHOL", "KOC HOLDING", "KOCHOL"],
    "KOCHOL": ["KCHOL", "KOC HOLDING", "KOCHOL"],
    "KOZAL": ["KOZAL", "TRALT", "KOZA ALTIN", "TURK ALTIN ISLETMELERI"],
    "TRALT": ["TRALT", "KOZAL", "TURK ALTIN ISLETMELERI", "KOZA ALTIN"],
    "PETKM": ["PETKM", "PETKIM"],
    "SAHOL": ["SAHOL", "SABANCI"],
    "SISE": ["SISE", "SISECAM"],
    "THYAO": ["THYAO", "THY", "TURK HAVA YOLLARI"],
    "TOASO": ["TOASO", "TOFAS"],
    "TUPRS": ["TUPRS", "TUPRAS"],
    "YKBNK": ["YKBNK", "YAPI KREDI"],
    "KRDMA": ["KRDMA", "KRDMB", "KRDMD", "KARDEMIR"],
    "KRDMB": ["KRDMB", "KRDMA", "KRDMD", "KARDEMIR"],
    "KRDMD": ["KRDMD", "KRDMA", "KRDMB", "KARDEMIR"],
    "DENGE": ["DENGE", "DENGE YATIRIM"],
    "INVEO": ["INVEO", "INVEO YATIRIM"],
}

COMPANY_MEMBER_TITLE_HINTS: Dict[str, Tuple[str, ...]] = {
    # KAP member/filter/TERA returns MEDİTERA first because "tera" is a substring.
    # The ticker TERA is Tera Yatırım Menkul Değerler A.Ş.; prefer that exact issuer.
    "TERA": ("TERA YATIRIM MENKUL",),
    # KAP `member/filter/GARAN` (or GARANTI) returns several Garanti-affiliated
    # companies. For the BIST stock GARAN we want Türkiye Garanti Bankası A.Ş.,
    # not Garanti Emeklilik / Garanti Faktoring / Garanti Yatırım Ortaklığı.
    "GARAN": ("GARANTI BANKASI", "TURKIYE GARANTI BANKASI"),
    # Same disambiguation for other BIST-30 issuers whose KAP filter results
    # include affiliates with similar names. Each hint matches a single canonical
    # KAP issuer title (Turkish characters normalised).
    "AKBNK": ("AKBANK",),
    "ISCTR": ("TURKIYE IS BANKASI",),
    "YKBNK": ("YAPI VE KREDI BANKASI",),
    "HALKB": ("TURKIYE HALK BANKASI",),
    "VAKBN": ("TURKIYE VAKIFLAR BANKASI",),
    "AKSEN": ("AKSA ENERJI URETIM",),
    "ENKAI": ("ENKA INSAAT",),
    "EREGL": ("EREGLI DEMIR",),
    "FROTO": ("FORD OTOMOTIV",),
    "GUBRF": ("GUBRE FABRIKALARI",),
    "KCHOL": ("KOC HOLDING",),
    "KOZAL": ("TURK ALTIN ISLETMELERI", "KOZA ALTIN ISLETMELERI"),
    "KOZAA": ("KOZA ANADOLU METAL",),
    "KRDMA": ("KARDEMIR",),
    "KRDMB": ("KARDEMIR",),
    "KRDMD": ("KARDEMIR",),
    "DENGE": ("DENGE YATIRIM",),
    "INVEO": ("INVEO YATIRIM",),
    "MGROS": ("MIGROS TICARET",),
    "PETKM": ("PETKIM PETROKIMYA",),
    "SAHOL": ("HACI OMER SABANCI", "SABANCI HOLDING"),
    "SASA": ("SASA POLYESTER",),
    "SISE": ("TURKIYE SISE VE CAM",),
    "TCELL": ("TURKCELL ILETISIM",),
    "THYAO": ("TURK HAVA YOLLARI",),
    "TOASO": ("TOFAS",),
    "TTKOM": ("TURK TELEKOMUNIKASYON",),
    "TUPRS": ("TURKIYE PETROL RAFINERILERI", "TUPRAS",),
    "ULKER": ("ULKER BISKUVI",),
    "TKFEN": ("TEKFEN HOLDING",),
}

KAP_MEMBER_FALLBACKS: Dict[str, Dict[str, str]] = {
    "AGESA": {
        "company_code": "AGESA",
        "mkk_member_oid": "4028e4a140e95bea0140edeb7a54015d",
        "title": "AGESA HAYAT VE EMEKLİLİK A.Ş.",
        "permalink": "2370-agesa-hayat-ve-emeklilik-a-s",
        "query": "AGESA",
    },
    "ANHYT": {
        "company_code": "ANHYT",
        "mkk_member_oid": "4028e4a140e95be70140ed40cb930098",
        "title": "ANADOLU HAYAT EMEKLİLİK A.Ş.",
        "permalink": "860-anadolu-hayat-emeklilik-a-s",
        "query": "ANHYT",
    },
    "RAYSG": {
        "company_code": "RAYSG",
        "mkk_member_oid": "4028e4a241733d4201417ddf67f8288c",
        "title": "RAY SİGORTA A.Ş.",
        "permalink": "1063-ray-sigorta-a-s",
        "query": "RAYSG",
    },
    "KRDMA": {
        "company_code": "KRDMD",
        "mkk_member_oid": "4028e4a140f2ed7201412ace3ada0707",
        "title": "KARDEMİR KARABÜK DEMİR ÇELİK SANAYİ VE TİCARET A.Ş.",
        "permalink": "994-kardemir-karabuk-demir-celik-sanayi-ve-ticaret-a-s",
        "query": "KRDMD",
    },
    "KRDMB": {
        "company_code": "KRDMD",
        "mkk_member_oid": "4028e4a140f2ed7201412ace3ada0707",
        "title": "KARDEMİR KARABÜK DEMİR ÇELİK SANAYİ VE TİCARET A.Ş.",
        "permalink": "994-kardemir-karabuk-demir-celik-sanayi-ve-ticaret-a-s",
        "query": "KRDMD",
    },
    "KRDMD": {
        "company_code": "KRDMD",
        "mkk_member_oid": "4028e4a140f2ed7201412ace3ada0707",
        "title": "KARDEMİR KARABÜK DEMİR ÇELİK SANAYİ VE TİCARET A.Ş.",
        "permalink": "994-kardemir-karabuk-demir-celik-sanayi-ve-ticaret-a-s",
        "query": "KRDMD",
    },
    "DENGE": {
        "company_code": "1678",
        "mkk_member_oid": "4028e4a240ee866c0140f1f80b55002b",
        "title": "DENGE YATIRIM HOLDİNG A.Ş.",
        "permalink": "1678-denge-yatirim-holding-a-s",
        "query": "DENGE",
    },
    "INVEO": {
        "company_code": "962",
        "mkk_member_oid": "4028e4a140f2ed7201413b67b22f0929",
        "title": "INVEO YATIRIM HOLDİNG A.Ş.",
        "permalink": "962-inveo-yatirim-holding-a-s",
        "query": "INVEO",
    },
    "DOCO": {
        "company_code": "1530",
        "mkk_member_oid": "1DE05DAA82B4073AE0530A4A622A2EBD",
        "title": "DO & CO AKTIENGESELLSCHAFT",
        "permalink": "1530-do-co-aktiengesellschaft",
        "query": "DOCO",
    },
}


from src.kap.exceptions import KapPacingDeadlineExceeded
from src.kap.html_parser import (
    TR_NORMALIZE_MAP,
    _LABEL_PATTERN,
    _LegacyKapFinancialRowParser,
    _VALUE_PATTERN,
    _clean_html_text,
    _extract_rows_from_disclosure_body,
    _normalize,
    _parse_numeric_token,
    _parse_unit_info,
)
from src.kap.insurance_premium import (
    MARKETVISUALS_INSURANCE_COMPANY_ALIASES,
    MARKETVISUALS_INSURANCE_COMPANY_PAGES,
    MARKETVISUALS_INSURANCE_COMPANY_TITLES,
    MARKETVISUALS_INSURANCE_HAYATDISI_URL,
    MARKETVISUALS_INSURANCE_HAYAT_URL,
    MARKETVISUALS_TSB_SOURCE_LABEL,
    TURKISH_MONTH_NAME_TO_NUMBER,
    _derive_missing_monthly_premiums,
    _ensure_insurance_premium_disclosures,
    _extract_js_assignment_json,
    _extract_pdf_text,
    _fetch_insurance_premium_disclosures,
    _fetch_premium_only_snapshot,
    _fetch_tsb_marketvisuals_premium_disclosures,
    _has_insurance_premium_rows,
    _insurance_premium_company_key,
    _insurance_premium_company_title,
    _insurance_premium_disclosure_body,
    _is_insurance_like_payload,
    _is_insurance_premium_check_fresh,
    _is_premium_production_disclosure,
    _list_member_disclosures_by_criteria,
    _marketvisuals_company_aliases,
    _marketvisuals_matches_company,
    _marketvisuals_period_dates,
    _marketvisuals_rows_from_chart,
    _merge_insurance_premium_disclosures,
    _parse_insurance_premium_pdf_text,
    _parse_marketvisuals_insurance_premium_page,
    _parse_premium_number_token,
    _parse_premium_yoy_token,
    _pct_change,
    _premium_attachment_text,
    _premium_disclosure_text,
    _premium_pdf_unit_multiplier,
    _premium_period_from_text,
)
from src.kap.metric_scoring import (
    _col_preference,
    _derive_favok,
    _derive_finansal_borclar,
    _derive_net_borc,
    _extract_disclosure_metrics,
    _is_cash_equivalent_label,
    _is_favok_label,
    _pick_best_row,
    _pick_cash_like_financial_investment,
    _pick_income_row_value,
    _pick_metric_value,
    _score_metric_candidate,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def classify_kap_company_kind(company: Any = "", company_title: Any = "") -> str:
    """Classify a KAP issuer from identity only, never from parsed metrics."""
    company_norm = re.sub(r"[^A-Z0-9]", "", str(company or "").upper())
    title_norm = _normalize(str(company_title or ""))
    title_tokens = set(title_norm.split())

    if (
        company_norm in _KAP_BANK_TICKER_HINTS
        or "bankasi" in title_tokens
        or "bank" in title_tokens
    ):
        return "bank"
    if (
        company_norm in _KAP_INSURANCE_TICKER_HINTS
        or "sigorta" in title_norm
        or "emeklilik" in title_norm
    ):
        return "insurance"
    return "generic"


def _cache_file_for_company(processed_dir: Path, company: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(company or "").strip().upper()) or "UNKNOWN"
    return processed_dir / "kap_cache" / f"{slug}.json"


def _company_cache_keys(company: str) -> List[str]:
    company_key = str(company or "").strip().upper()
    keys: List[str] = []

    direct_aliases = COMPANY_QUERY_ALIASES.get(company_key)
    if direct_aliases:
        keys.extend(str(item or "").strip().upper() for item in direct_aliases)

    for canonical, aliases in COMPANY_QUERY_ALIASES.items():
        alias_keys = [str(item or "").strip().upper() for item in aliases]
        canonical_key = str(canonical or "").strip().upper()
        if company_key == canonical_key or company_key in alias_keys:
            keys.append(canonical_key)
            keys.extend(alias_keys)

    keys.append(company_key)

    result: List[str] = []
    for key in keys:
        if key and key not in result:
            result.append(key)
    return result or ["UNKNOWN"]


def _read_first_cache(processed_dir: Path, company: str) -> Tuple[Path, Optional[Dict[str, Any]]]:
    cache_keys = _company_cache_keys(company)
    primary_path = _cache_file_for_company(processed_dir, cache_keys[0])
    for key in cache_keys:
        path = _cache_file_for_company(processed_dir, key)
        cached = _read_cache(path)
        if cached and _cache_matches_requested_company(cached, company):
            # Snapshot refresh paths annotate and merge this payload in place.
            # Keep those changes out of the shared parsed-payload LRU, and
            # always return primary_path so updates write to the canonical key.
            return primary_path, copy.deepcopy(cached)
    return primary_path, None


def _cache_matches_requested_company(payload: Dict[str, Any], company: str) -> bool:
    company_key = str(company or "").strip().upper()
    if not company_key or not isinstance(payload, dict):
        return False
    acceptable = set(_company_cache_keys(company_key))
    acceptable.add(company_key)

    identifiers = [
        str(payload.get("company") or "").strip().upper(),
        str(payload.get("stock_code") or "").strip().upper(),
    ]
    for quarter in payload.get("quarters") or []:
        if isinstance(quarter, dict):
            identifiers.append(str(quarter.get("stock_code") or "").strip().upper())
    if any(identifier in acceptable for identifier in identifiers if identifier):
        return True

    hints = COMPANY_MEMBER_TITLE_HINTS.get(company_key)
    if hints:
        title_norm = _normalize(payload.get("company_title") or "")
        return any(_normalize(hint) in title_norm for hint in hints)
    return not any(identifiers)


def _cached_period_count(payload: Optional[Dict[str, Any]]) -> int:
    if not isinstance(payload, dict):
        return 0
    quarters = payload.get("quarters") or []
    if not isinstance(quarters, list):
        return 0
    periods: set[Tuple[int, int]] = set()
    for item in quarters:
        if not isinstance(item, dict):
            continue
        try:
            periods.add((int(item.get("year", 0)), int(item.get("period", 0))))
        except (TypeError, ValueError):
            continue
    return len(periods)


def _parse_cached_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_recent_timestamp(value: Any, ttl_hours: float) -> bool:
    parsed = _parse_cached_datetime(value)
    if parsed is None:
        return False
    return (_utc_now() - parsed) <= timedelta(hours=max(0.0, float(ttl_hours)))


def _latest_disclosure_key_from_cache(payload: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int, int]]:
    if not isinstance(payload, dict):
        return None
    quarters = payload.get("quarters") or []
    if not isinstance(quarters, list):
        return None
    keys: List[Tuple[int, int, int]] = []
    for item in quarters:
        if not isinstance(item, dict):
            continue
        try:
            keys.append(
                (
                    int(item.get("year", 0)),
                    int(item.get("period", 0)),
                    int(item.get("disclosure_index", 0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return max(keys) if keys else None


def _latest_disclosure_key_from_list(rows: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int]]:
    keys: List[Tuple[int, int, int]] = []
    for item in rows:
        try:
            keys.append(
                (
                    int(item.get("year", 0)),
                    int(item.get("period", 0)),
                    int(item.get("disclosure_index", 0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return max(keys) if keys else None


def _quarter_sort_tuple(row: Dict[str, Any]) -> Tuple[int, int, int]:
    try:
        return (
            int(row.get("year", 0)),
            int(row.get("period", 0)),
            int(row.get("disclosure_index", 0)),
        )
    except (TypeError, ValueError):
        return (0, 0, 0)


def _merge_live_and_cached_quarters(
    live_quarters: List[Dict[str, Any]],
    cached_payload: Optional[Dict[str, Any]],
    max_quarters: int,
) -> List[Dict[str, Any]]:
    if not isinstance(cached_payload, dict):
        return live_quarters
    try:
        cached_schema = int(cached_payload.get("schema_version") or 0)
    except (TypeError, ValueError):
        cached_schema = 0
    if cached_schema not in {0, KAP_CACHE_SCHEMA_VERSION}:
        return live_quarters[: max(1, int(max_quarters))]
    cached_quarters = cached_payload.get("quarters") or []
    if not isinstance(cached_quarters, list) or len(cached_quarters) <= len(live_quarters):
        return live_quarters

    by_period: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in cached_quarters:
        if not isinstance(row, dict):
            continue
        try:
            by_period[(int(row.get("year", 0)), int(row.get("period", 0)))] = row
        except (TypeError, ValueError):
            continue

    for row in live_quarters:
        try:
            by_period[(int(row.get("year", 0)), int(row.get("period", 0)))] = row
        except (TypeError, ValueError):
            continue

    merged = sorted(by_period.values(), key=_quarter_sort_tuple, reverse=True)
    return merged[: max(1, int(max_quarters))]


def _mark_live_disclosure_checked(
    cache_path: Path,
    payload: Dict[str, Any],
    latest_key: Optional[Tuple[int, int, int]] = None,
) -> None:
    payload["live_disclosure_checked_at"] = _utc_now().isoformat()
    payload.pop("live_disclosure_check_error", None)
    if latest_key:
        payload["live_disclosure_latest"] = {
            "year": latest_key[0],
            "period": latest_key[1],
            "disclosure_index": latest_key[2],
        }
    try:
        _write_cache(cache_path, payload)
    except Exception:
        pass


def _mark_live_disclosure_check_failed(cache_path: Path, payload: Dict[str, Any], error: Exception) -> None:
    payload["live_disclosure_checked_at"] = _utc_now().isoformat()
    payload["live_disclosure_check_error"] = str(error)
    try:
        _write_cache(cache_path, payload)
    except Exception:
        pass


def _read_cache(path: Path) -> Optional[Dict[str, Any]]:
    return read_json_cache_local_first(path, remote_reader=read_json_cache, reader="kap")


def _write_cache(path: Path, payload: Dict[str, Any]) -> None:
    # Update the local mirror and its L1 signature before the remote write so a
    # reader in this process sees the new payload immediately.
    write_local_json(path, payload)
    write_json_cache(path, payload)


def _is_cache_fresh(payload: Dict[str, Any], ttl_hours: float) -> bool:
    # ttl_hours <= 0: her zaman canlı KAP'a git (dosya önbelleği yalnızca yedek / yazma için).
    if float(ttl_hours) <= 0:
        return False
    return _is_recent_timestamp(payload.get("fetched_at"), ttl_hours)


def _is_live_disclosure_check_fresh(payload: Dict[str, Any]) -> bool:
    return _is_recent_timestamp(
        payload.get("live_disclosure_checked_at"),
        KAP_LIVE_DISCLOSURE_CHECK_TTL_HOURS,
    )


def _kap_request_headers(cfg: KapConfig, *, accept: str) -> Dict[str, str]:
    user_agent = str(getattr(cfg, "user_agent", "") or "").strip() or KAP_BROWSER_USER_AGENT
    return {
        "Accept": accept,
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        "User-Agent": user_agent,
        "Referer": "https://www.kap.org.tr/tr/",
        "X-Requested-With": "XMLHttpRequest",
    }


def _candidate_user_agents(cfg: KapConfig) -> List[str]:
    configured = str(getattr(cfg, "user_agent", "") or "").strip()
    if "mozilla/" in configured.lower():
        candidates = [configured, KAP_BROWSER_USER_AGENT]
    else:
        candidates = [KAP_BROWSER_USER_AGENT, configured]
    result: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in result:
            result.append(candidate)
    return result or [KAP_BROWSER_USER_AGENT]


_KAP_REQUEST_PACING_LOCK = threading.Lock()
_KAP_REQUEST_PACING_INTERVAL = 0.0
_KAP_REQUEST_PACING_LAST_AT = 0.0
_KAP_REQUEST_PACING_COOLDOWN_UNTIL = 0.0
_KAP_REQUEST_PACING_DEADLINE = 0.0
_KAP_429_FALLBACK_COOLDOWN_SECONDS = 120.0
_KAP_429_MAX_COOLDOWN_SECONDS = 3600.0


def set_kap_request_pacing(
    interval_seconds: float,
    *,
    deadline_monotonic: float = 0.0,
) -> None:
    """Throttle process-wide KAP JSON requests to a minimum start-to-start gap.

    KAP resets connections after a burst of statement attachment downloads, so
    batch jobs enable pacing for their duration. Interactive requests inherit
    the same pacing while a batch is running, which is desirable protection.
    """

    global _KAP_REQUEST_PACING_INTERVAL, _KAP_REQUEST_PACING_LAST_AT, _KAP_REQUEST_PACING_DEADLINE
    try:
        interval = max(0.0, float(interval_seconds))
    except (TypeError, ValueError):
        interval = 0.0
    try:
        deadline = max(0.0, float(deadline_monotonic)) if interval > 0 else 0.0
    except (TypeError, ValueError):
        deadline = 0.0
    with _KAP_REQUEST_PACING_LOCK:
        _KAP_REQUEST_PACING_INTERVAL = interval
        _KAP_REQUEST_PACING_DEADLINE = deadline
        _KAP_REQUEST_PACING_LAST_AT = (
            time.monotonic() - interval if interval > 0 else 0.0
        )


def _kap_batch_pacing_enabled() -> bool:
    return _KAP_REQUEST_PACING_INTERVAL > 0


def _is_kap_url(url: str) -> bool:
    return "marketvisuals.net" not in str(url or "").lower()


def _kap_retry_after_seconds(exc: urllib.error.HTTPError) -> float:
    headers = getattr(exc, "headers", None) or getattr(exc, "hdrs", None)
    retry_after = headers.get("Retry-After") if headers is not None else None
    if retry_after is not None:
        try:
            seconds = float(str(retry_after).strip())
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = _KAP_429_FALLBACK_COOLDOWN_SECONDS
    else:
        seconds = _KAP_429_FALLBACK_COOLDOWN_SECONDS
    return min(_KAP_429_MAX_COOLDOWN_SECONDS, max(0.0, seconds))


def _defer_kap_requests_after_rate_limit(exc: urllib.error.HTTPError) -> None:
    """Keep all KAP callers idle for the provider's Retry-After window."""

    global _KAP_REQUEST_PACING_COOLDOWN_UNTIL
    cooldown_seconds = _kap_retry_after_seconds(exc)
    if cooldown_seconds <= 0:
        return
    with _KAP_REQUEST_PACING_LOCK:
        _KAP_REQUEST_PACING_COOLDOWN_UNTIL = max(
            _KAP_REQUEST_PACING_COOLDOWN_UNTIL,
            time.monotonic() + cooldown_seconds,
        )


def _wait_for_kap_request_slot() -> None:
    global _KAP_REQUEST_PACING_LAST_AT
    with _KAP_REQUEST_PACING_LOCK:
        now = time.monotonic()
        next_slot = max(
            _KAP_REQUEST_PACING_LAST_AT + _KAP_REQUEST_PACING_INTERVAL,
            _KAP_REQUEST_PACING_COOLDOWN_UNTIL,
        )
        if _KAP_REQUEST_PACING_DEADLINE > 0 and (
            now >= _KAP_REQUEST_PACING_DEADLINE
            or next_slot > _KAP_REQUEST_PACING_DEADLINE
        ):
            raise KapPacingDeadlineExceeded("KAP istek sirasi is butcesini asti.")
        wait_seconds = next_slot - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
            if _KAP_REQUEST_PACING_DEADLINE > 0 and now >= _KAP_REQUEST_PACING_DEADLINE:
                raise KapPacingDeadlineExceeded("KAP istek sirasi is butcesini asti.")
        if _KAP_REQUEST_PACING_INTERVAL > 0:
            _KAP_REQUEST_PACING_LAST_AT = now


def _http_get_json(url: str, cfg: KapConfig) -> Any:
    last_error: Optional[BaseException] = None
    is_kap = _is_kap_url(url)
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept="application/json, text/plain, */*")
        headers["User-Agent"] = user_agent
        max_attempts = 3 if (is_kap and _kap_batch_pacing_enabled()) else 2
        for attempt in range(max_attempts):
            if is_kap:
                _wait_for_kap_request_slot()
            request = urllib.request.Request(url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429:
                    if is_kap:
                        _defer_kap_requests_after_rate_limit(exc)
                    if is_kap and _kap_batch_pacing_enabled() and attempt + 1 < max_attempts:
                        continue
                    raise
                time.sleep(0.8 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                break
        if isinstance(last_error, urllib.error.HTTPError) and int(getattr(last_error, "code", 0) or 0) == 429:
            raise last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("KAP JSON istegi basarisiz oldu.")


def _http_get_text(url: str, cfg: KapConfig) -> str:
    last_error: Optional[BaseException] = None
    is_kap = _is_kap_url(url)
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept="text/html,application/xhtml+xml,*/*")
        headers["User-Agent"] = user_agent
        max_attempts = 3 if (is_kap and _kap_batch_pacing_enabled()) else 2
        for attempt in range(max_attempts):
            if is_kap:
                _wait_for_kap_request_slot()
            request = urllib.request.Request(url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429:
                    if is_kap:
                        _defer_kap_requests_after_rate_limit(exc)
                    if is_kap and _kap_batch_pacing_enabled() and attempt + 1 < max_attempts:
                        continue
                    raise
                time.sleep(0.8 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                break
        if isinstance(last_error, urllib.error.HTTPError) and int(getattr(last_error, "code", 0) or 0) == 429:
            raise last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("KAP text istegi basarisiz oldu.")


def _http_post_json(url: str, payload: Dict[str, Any], cfg: KapConfig) -> Any:
    last_error: Optional[BaseException] = None
    body = json.dumps(payload).encode("utf-8")
    is_kap = _is_kap_url(url)
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept="application/json, text/plain, */*")
        headers["User-Agent"] = user_agent
        headers["Content-Type"] = "application/json"
        max_attempts = 3 if (is_kap and _kap_batch_pacing_enabled()) else 2
        for attempt in range(max_attempts):
            if is_kap:
                _wait_for_kap_request_slot()
            request = urllib.request.Request(url, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429:
                    if is_kap:
                        _defer_kap_requests_after_rate_limit(exc)
                    if is_kap and _kap_batch_pacing_enabled() and attempt + 1 < max_attempts:
                        continue
                    raise
                time.sleep(0.8 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                break
        if isinstance(last_error, urllib.error.HTTPError) and int(getattr(last_error, "code", 0) or 0) == 429:
            raise last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("KAP JSON POST istegi basarisiz oldu.")


def _http_get_bytes(url: str, cfg: KapConfig, *, accept: str = "*/*") -> bytes:
    last_error: Optional[BaseException] = None
    is_kap = _is_kap_url(url)
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept=accept)
        headers["User-Agent"] = user_agent
        max_attempts = 3 if (is_kap and _kap_batch_pacing_enabled()) else 2
        for attempt in range(max_attempts):
            if is_kap:
                _wait_for_kap_request_slot()
            request = urllib.request.Request(url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429:
                    if is_kap:
                        _defer_kap_requests_after_rate_limit(exc)
                    if is_kap and _kap_batch_pacing_enabled() and attempt + 1 < max_attempts:
                        continue
                    raise
                time.sleep(0.8 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                break
        if isinstance(last_error, urllib.error.HTTPError) and int(getattr(last_error, "code", 0) or 0) == 429:
            raise last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("KAP dosya istegi basarisiz oldu.")


def _member_filter_url(query: str) -> str:
    return f"{KAP_BASE_URL}/{MEMBER_FILTER_ENDPOINT}/{urllib.parse.quote(query)}"


def _select_member_filter_row(company_key: str, rows: List[Any]) -> Optional[Dict[str, Any]]:
    materialized = [dict(row or {}) for row in rows if isinstance(row, dict)]
    if not materialized:
        return None
    hints = COMPANY_MEMBER_TITLE_HINTS.get(company_key)
    if hints:
        normalized_hints = [_normalize(hint) for hint in hints if _normalize(hint)]
        for row in materialized:
            title_norm = _normalize(row.get("title") or "")
            permalink_norm = _normalize(str(row.get("permaLink") or "").replace("-", " "))
            if any(hint in title_norm or hint in permalink_norm for hint in normalized_hints):
                return row
    return materialized[0]


def _resolve_member(
    company: str,
    cfg: KapConfig,
    *,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    company_key = str(company or "").strip().upper()
    if not company_key:
        return None
    cached = _resolve_member_from_cache(company_key)
    if cached is not None:
        resolved_cached = _empty_member_to_none(cached)
        if resolved_cached is not None or not strict:
            return resolved_cached
    resolved = _resolve_member_uncached(company_key, cfg, strict=strict)
    if resolved is not None or not strict:
        _store_resolved_member(company_key, resolved)
    return resolved


def _resolve_member_uncached(
    company_key: str,
    cfg: KapConfig,
    *,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    candidate_queries = COMPANY_QUERY_ALIASES.get(company_key, [])
    if company_key not in candidate_queries:
        candidate_queries.append(company_key)
    # Fallback: split values like NETCAD_4Q -> NETCAD
    stripped = re.sub(r"[^A-Za-z0-9]+", " ", company_key).split(" ")
    if stripped:
        main_token = stripped[0].strip().upper()
        if main_token and main_token not in candidate_queries:
            candidate_queries.append(main_token)

    last_error: Optional[BaseException] = None
    for query in candidate_queries:
        try:
            rows = _http_get_json(_member_filter_url(query), cfg)
        except KapPacingDeadlineExceeded:
            raise
        except Exception as exc:
            last_error = exc
            continue
        if not isinstance(rows, list) or not rows:
            continue
        row = _select_member_filter_row(company_key, rows)
        if not row:
            continue
        mkk_oid = str(row.get("mkkMemberOid", "")).strip()
        if not mkk_oid:
            continue
        company_code = str(row.get("companyCode", "")).strip().upper()
        # KAP'ta KOZAL/KOZAA gibi BIST kodları yok; yan kuruluşlara (TRALT)
        # eşleşmesini engelliyoruz. Eşleşme yalnızca orijinal kod, tanınan
        # alias listesindeki kodlardan biri ya da KAP'ın numeric/empty
        # kodu olduğunda kabul edilir.
        looks_like_bist_ticker = bool(company_code) and company_code.isalpha() and 3 <= len(company_code) <= 6
        if looks_like_bist_ticker and company_code != company_key:
            allowed_codes = {alias.upper() for alias in candidate_queries if alias}
            fallback_meta = KAP_MEMBER_FALLBACKS.get(company_key) or {}
            fallback_code = str(fallback_meta.get("company_code", "")).strip().upper()
            if fallback_code:
                allowed_codes.add(fallback_code)
            if company_code not in allowed_codes:
                continue
        return {
            "company_code": str(row.get("companyCode", "")).strip(),
            "mkk_member_oid": mkk_oid,
            "title": str(row.get("title", "")).strip(),
            "permalink": str(row.get("permaLink", "")).strip(),
            "query": query,
        }
    if strict and last_error is not None:
        raise RuntimeError(
            f"KAP uye kaydi alinamadi: {company_key} error={last_error}"
        ) from last_error
    fallback = KAP_MEMBER_FALLBACKS.get(company_key)
    return dict(fallback) if fallback else None


# ---------------------------------------------------------------------------
# Member resolution cache (process-local + optional Redis backend)
# ---------------------------------------------------------------------------

# KAP'ın üye filtresi rate-limited; aynı ticker için defalarca arama atmak hem
# yavaş, hem 429'a neden oluyor. Bu yüzden başarılı/başarısız sonucu kısa
# süreli cache'e alıyoruz. ``app.cache`` katmanı default in-memory; Redis varsa
# tüm worker'lar arası paylaşıma açılır.
_RESOLVED_MEMBER_TTL_SECONDS = 24 * 60 * 60  # 24 saat


def _resolved_member_cache_key(company_key: str) -> str:
    return f"api:kap:resolve-member:{company_key}:v1"


def _empty_member_to_none(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payload or not payload.get("mkk_member_oid"):
        return None
    return dict(payload)


def _resolve_member_from_cache(company_key: str) -> Optional[Dict[str, Any]]:
    try:
        from app.cache import get_json_dict  # local import to avoid heavy deps at module import time
    except Exception:
        return None
    try:
        cached = get_json_dict(_resolved_member_cache_key(company_key))
    except Exception:
        return None
    return cached


def _store_resolved_member(company_key: str, resolved: Optional[Dict[str, Any]]) -> None:
    try:
        from app.cache import set_json
    except Exception:
        return
    payload: Dict[str, Any] = dict(resolved) if isinstance(resolved, dict) else {}
    # KAP'ta hiç bulunmayan ticker'ları (KOZAL, KOZAA gibi delisting'ler) negatif
    # cache'e alıyoruz; aksi halde her snapshot çağrısı KAP'ı yine yokluyor.
    try:
        set_json(
            _resolved_member_cache_key(company_key),
            payload,
            ttl_seconds=_RESOLVED_MEMBER_TTL_SECONDS,
        )
    except Exception:
        return


def _disclosure_text(row: Dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "title",
            "summary",
            "subject",
            "description",
            "financialStatementType",
            "financialStatementNature",
            "financialTableNature",
        )
    )
    return _normalize(text.replace("-", " "))


def _disclosure_consolidation_score(row: Dict[str, Any]) -> int:
    text = _disclosure_text(row)
    if not text:
        return 1
    if (
        "konsolide olmayan" in text
        or "konsolide degil" in text
        or "solo" in text
        or "unconsolidated" in text
        or "non consolidated" in text
    ):
        return 0
    if "konsolide" in text or "consolidated" in text:
        return 2
    return 1


def _disclosure_source_order(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("_source_order", 1_000_000))
    except (TypeError, ValueError):
        return 1_000_000


def _disclosure_index(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("disclosure_index") or row.get("disclosureIndex") or 0)
    except (TypeError, ValueError):
        return 0


def _disclosure_selection_key(row: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        _disclosure_consolidation_score(row),
        -_disclosure_source_order(row),
        _disclosure_index(row),
    )


def _preferred_disclosures_by_period(
    rows: List[Dict[str, Any]],
    max_periods: Optional[int] = None,
) -> List[Dict[str, Any]]:
    by_period: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in rows:
        try:
            key = (int(row.get("year") or 0), int(row.get("period") or 0))
        except (TypeError, ValueError):
            continue
        if key[0] <= 0 or key[1] < 1 or key[1] > 4:
            continue
        existing = by_period.get(key)
        if existing is None or _disclosure_selection_key(row) > _disclosure_selection_key(existing):
            by_period[key] = row

    ordered = sorted(
        by_period.values(),
        key=lambda item: (
            int(item.get("year") or 0),
            int(item.get("period") or 0),
            _disclosure_selection_key(item),
        ),
        reverse=True,
    )
    if max_periods is not None:
        ordered = ordered[: max(1, int(max_periods))]

    cleaned: List[Dict[str, Any]] = []
    for row in ordered:
        item = dict(row)
        item.pop("_source_order", None)
        cleaned.append(item)
    return cleaned


def _list_company_disclosures(
    member_oid: str,
    cfg: KapConfig,
    max_years_back: int = 10,
    max_periods: Optional[int] = None,
    *,
    allow_criteria_fallback: bool = True,
    strict_provider_errors: bool = False,
) -> List[Dict[str, Any]]:
    strict_errors = bool(strict_provider_errors) or not allow_criteria_fallback
    current_year = _utc_now().year
    rows: List[Dict[str, Any]] = []
    last_error: Optional[BaseException] = None
    provider_error: Optional[BaseException] = None
    requested_periods = max(1, int(max_periods)) if max_periods is not None else None
    empty_year_streak = 0

    for idx, year in enumerate(range(current_year, current_year - max_years_back - 1, -1)):
        if idx > 0:
            time.sleep(0.7)
        url = f"{KAP_BASE_URL}/{LIST_COMPANY_EXCEL_MEMBERS_ENDPOINT}/{member_oid}/{year}/T"
        try:
            payload = _http_get_json(url, cfg)
        except KapPacingDeadlineExceeded:
            raise
        except urllib.error.HTTPError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            if code == 429:
                if not rows and allow_criteria_fallback:
                    fallback_rows = _list_financial_report_disclosures_by_criteria(
                        member_oid=member_oid,
                        cfg=cfg,
                        max_years_back=max_years_back,
                        max_periods=requested_periods,
                    )
                    if fallback_rows:
                        return fallback_rows
                    raise RuntimeError(
                        "KAP istek limiti (429) nedeniyle finansal bildirimler şu anda alınamadı."
                    ) from exc
                if strict_errors:
                    raise RuntimeError(
                        f"KAP finansal bildirim listesi 429 ile sinirlandi: member_oid={member_oid}"
                    ) from exc
                break
            if code in KAP_RETRYABLE_HTTP_CODES:
                provider_error = exc
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            if strict_errors:
                # Connection resets and timeouts often arrive without an HTTP
                # status. Do not turn a partial year list into an "unchanged"
                # result in batch mode when one of those requests failed.
                provider_error = exc
            continue
        if not isinstance(payload, list):
            if rows:
                empty_year_streak += 1
                if empty_year_streak >= 2:
                    break
            continue
        rows_before_year = len(rows)
        for item in payload:
            if not isinstance(item, dict):
                continue
            disclosure_index = item.get("disclosureIndex")
            period = item.get("period")
            year_value = item.get("year", year)
            if disclosure_index is None or period is None:
                continue
            try:
                disclosure_index_int = int(disclosure_index)
                period_int = int(period)
                year_int = int(year_value)
            except Exception:
                continue
            if period_int < 1 or period_int > 4:
                continue
            rows.append(
                {
                    "year": year_int,
                    "period": period_int,
                    "disclosure_index": disclosure_index_int,
                    "stock_code": str(item.get("stockCode", "")).strip().upper(),
                    "title": str(item.get("title", "")).strip(),
                    "summary": str(item.get("summary", "")).strip(),
                    "subject": str(item.get("subject", "")).strip(),
                    "financialStatementType": str(item.get("financialStatementType", "")).strip(),
                    "financialStatementNature": str(item.get("financialStatementNature", "")).strip(),
                    "financialTableNature": str(item.get("financialTableNature", "")).strip(),
                    "pd_oid": str(item.get("pdOid", "")).strip(),
                    "mkk_member_oid": str(item.get("mkkMemberOid", "")).strip(),
                    "_source_order": len(rows),
                }
            )
        if len(rows) > rows_before_year:
            empty_year_streak = 0
        elif rows:
            empty_year_streak += 1
            if empty_year_streak >= 2:
                break
        if requested_periods is not None:
            period_count = len({(row["year"], row["period"]) for row in rows})
            if period_count >= requested_periods:
                break

    if strict_errors and provider_error is not None:
        raise RuntimeError(
            f"KAP finansal bildirim listesi saglayici hatasi: member_oid={member_oid} error={provider_error}"
        ) from provider_error

    if not rows and strict_errors and last_error is not None:
        raise RuntimeError(
            f"KAP finansal bildirim listesi alinamadi: member_oid={member_oid} error={last_error}"
        ) from last_error

    selected_rows = _preferred_disclosures_by_period(rows)
    current_period_count = len(selected_rows)
    needs_fallback = allow_criteria_fallback and (
        not selected_rows
        or (requested_periods is not None and current_period_count < requested_periods)
    )
    if needs_fallback:
        fallback_rows = _list_financial_report_disclosures_by_criteria(
            member_oid=member_oid,
            cfg=cfg,
            max_years_back=max_years_back,
            max_periods=requested_periods,
        )
        if fallback_rows:
            return _preferred_disclosures_by_period(
                selected_rows + fallback_rows,
                requested_periods,
            )
    return _preferred_disclosures_by_period(rows, requested_periods)


def _list_financial_report_disclosures_by_criteria(
    *,
    member_oid: str,
    cfg: KapConfig,
    max_years_back: int = 10,
    max_periods: Optional[int] = None,
) -> List[Dict[str, Any]]:
    lookback_days = max(365, int(max_years_back) * 370)
    rows = _list_member_disclosures_by_criteria(
        member_oid=member_oid,
        cfg=cfg,
        lookback_days=lookback_days,
    )
    candidates: List[Dict[str, Any]] = []
    for source_order, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        subject_norm = _normalize(str(row.get("subject") or ""))
        if "finansal rapor" not in subject_norm:
            continue
        try:
            disclosure_index = int(row.get("disclosureIndex") or 0)
            year = int(row.get("year") or 0)
            period = int(row.get("period") or 0)
        except (TypeError, ValueError):
            continue
        if disclosure_index <= 0 or year <= 0 or period < 1 or period > 4:
            continue
        candidates.append(
            {
                "year": year,
                "period": period,
                "disclosure_index": disclosure_index,
                "stock_code": "",
                "title": str(row.get("title") or row.get("summary") or "").strip(),
                "summary": str(row.get("summary", "")).strip(),
                "subject": str(row.get("subject", "")).strip(),
                "_source_order": source_order,
            }
        )

    return _preferred_disclosures_by_period(candidates, max_periods)


def _fetch_attachment_detail(disclosure_index: int, cfg: KapConfig) -> Optional[Dict[str, Any]]:
    url = f"{KAP_BASE_URL}/{ATTACHMENT_DETAIL_ENDPOINT}/{int(disclosure_index)}"
    payload = _http_get_json(url, cfg)
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    return dict(first) if isinstance(first, dict) else None


def _quarter_label(year: int, period: int) -> str:
    return f"{int(year)}Q{int(period)}"


def _normalize_disclosure_stock_code(raw: Any, requested_company: str) -> str:
    text = str(raw or "").strip().upper()
    requested = str(requested_company or "").strip().upper()
    if not text:
        return requested
    tokens = [token.strip().upper() for token in re.split(r"[,;/\s]+", text) if token.strip()]
    if requested and requested in tokens:
        return requested
    return tokens[0] if tokens else text


def fetch_kap_company_snapshot(
    *,
    company: str,
    cfg: KapConfig,
    processed_dir: Path,
    force_refresh: bool = False,
    max_quarters: int = 4,
    use_cache_when_complete: bool = False,
    allow_criteria_fallback: bool = True,
    strict_provider_errors: bool = False,
) -> Dict[str, Any]:
    strict_errors = bool(strict_provider_errors) or not allow_criteria_fallback
    company_norm = str(company or "").strip().upper()
    if not company_norm:
        return {
            "ok": False,
            "company": company_norm,
            "error": "Sirket bilgisi bos.",
            "quarters": [],
        }

    requested_max_quarters = max(1, int(max_quarters))

    cache_path, cached = _read_first_cache(processed_dir, company_norm)
    cache_version = int(cached.get("schema_version", 0)) if isinstance(cached, dict) else 0
    cached_period_count = _cached_period_count(cached)
    # A shallow request (the UI asks for five quarters first) must not truncate
    # an existing deeper history. Merge against the cached depth and persist the
    # full result so the next snapshot request can reuse the older quarters.
    cached_depth_to_preserve = (
        cached_period_count if cache_version == KAP_CACHE_SCHEMA_VERSION else 0
    )
    cache_write_depth = max(requested_max_quarters, cached_depth_to_preserve)
    try:
        cached_live_period_count = int(cached.get("live_period_count") or 0) if isinstance(cached, dict) else 0
    except (TypeError, ValueError):
        cached_live_period_count = 0
    try:
        cached_live_request_depth = (
            int(cached.get("live_period_request_depth") or 0)
            if isinstance(cached, dict)
            else 0
        )
    except (TypeError, ValueError):
        cached_live_request_depth = 0
    # `live_period_count` is the number of periods returned by KAP. It only
    # proves the provider has no more history when the request that produced it
    # asked for more periods than it got back. A cache built with a shallow
    # request must not satisfy a deeper request just because the two counts are
    # equal, otherwise the history can never grow beyond the first fetch.
    provider_total_known = (
        cached_live_period_count > 0
        and cached_live_request_depth > cached_live_period_count
        and cached_period_count >= cached_live_period_count
    )
    cache_has_requested_depth = cached_period_count >= requested_max_quarters or (
        provider_total_known and requested_max_quarters >= cached_live_period_count
    )
    if (
        cached
        and not force_refresh
        and cache_version == KAP_CACHE_SCHEMA_VERSION
        and cache_has_requested_depth
    ):
        if _is_cache_fresh(cached, cfg.cache_ttl_hours) or (
            use_cache_when_complete and _is_live_disclosure_check_fresh(cached)
        ):
            cached["cache_hit"] = True
            cached["cache_stale"] = False
            cached.pop("error", None)
            return _ensure_insurance_premium_disclosures(
                payload=cached,
                cache_path=cache_path,
                cfg=cfg,
                member=None,
                force_refresh=False,
                strict=not allow_criteria_fallback,
            )

        if use_cache_when_complete:
            cached_member_oid = str(cached.get("member_oid") or "").strip()
            if cached_member_oid:
                try:
                    latest_disclosures = _list_company_disclosures(
                        member_oid=cached_member_oid,
                        cfg=cfg,
                        max_periods=1,
                        allow_criteria_fallback=allow_criteria_fallback,
                    )
                    live_latest_key = _latest_disclosure_key_from_list(latest_disclosures)
                    cached_latest_key = _latest_disclosure_key_from_cache(cached)
                    if live_latest_key and cached_latest_key and live_latest_key <= cached_latest_key:
                        _mark_live_disclosure_checked(cache_path, cached, live_latest_key)
                        cached["cache_hit"] = True
                        cached["cache_stale"] = False
                        cached.pop("error", None)
                        return _ensure_insurance_premium_disclosures(
                            payload=cached,
                            cache_path=cache_path,
                            cfg=cfg,
                            member=None,
                            force_refresh=False,
                            strict=strict_errors,
                        )
                    if not live_latest_key:
                        raise RuntimeError("KAP finansal bildirim listesi bos dondu.")
                except Exception as exc:
                    # The UI can fall back to the cached snapshot, but a batch
                    # job must surface provider blocks instead of reporting a
                    # throttled check as "unchanged".
                    if strict_errors:
                        raise
                    _mark_live_disclosure_check_failed(cache_path, cached, exc)
                    cached["cache_hit"] = True
                    cached["cache_stale"] = False
                    cached.pop("error", None)
                    return _ensure_insurance_premium_disclosures(
                        payload=cached,
                        cache_path=cache_path,
                        cfg=cfg,
                        member=None,
                        force_refresh=False,
                    )

    member: Optional[Dict[str, Any]] = None

    try:
        # Keep the legacy two-argument call for interactive paths so existing
        # monkeypatches keep working; batch jobs opt into strict mode.
        member = (
            _resolve_member(company_norm, cfg, strict=True)
            if strict_errors
            else _resolve_member(company_norm, cfg)
        )
        if not member:
            raise RuntimeError(f"KAP uyelik kaydi bulunamadi: {company_norm}")

        # KAP banka raporları solo + konsolide sütunları yan yana basıyor.
        # BIST'teki yatırımcı ve veri sağlayıcılar konsolide rakamı kullanıyor;
        # bilanço kalemlerini bankada konsolide sütunundan seçiyoruz.
        member_title = str(member.get("title") or "").strip()
        company_kind = classify_kap_company_kind(company_norm, member_title)
        is_bank = company_kind == "bank"
        is_insurance = company_kind == "insurance"

        disclosures_kwargs: Dict[str, Any] = {
            "member_oid": member["mkk_member_oid"],
            "cfg": cfg,
            "max_periods": requested_max_quarters,
            "allow_criteria_fallback": allow_criteria_fallback,
        }
        if strict_provider_errors:
            disclosures_kwargs["strict_provider_errors"] = True
        disclosures = _list_company_disclosures(**disclosures_kwargs)
        if not disclosures:
            raise RuntimeError("KAP finansal bildirimleri şu anda alınamadı.")

        reusable_quarters_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        if (
            isinstance(cached, dict)
            and not force_refresh
            and cache_version == KAP_CACHE_SCHEMA_VERSION
        ):
            for cached_quarter in cached.get("quarters") or []:
                if not isinstance(cached_quarter, dict):
                    continue
                if not isinstance(cached_quarter.get("metrics"), dict) or not cached_quarter.get("metrics"):
                    continue
                try:
                    cached_key = (
                        int(cached_quarter.get("year") or 0),
                        int(cached_quarter.get("period") or 0),
                        int(cached_quarter.get("disclosure_index") or 0),
                    )
                except (TypeError, ValueError):
                    continue
                if cached_key[0] > 0 and 1 <= cached_key[1] <= 4 and cached_key[2] > 0:
                    reusable_quarters_by_key[cached_key] = cached_quarter

        quarter_rows: List[Dict[str, Any]] = []
        seen_periods: set[Tuple[int, int]] = set()
        for item in disclosures:
            if len(quarter_rows) >= requested_max_quarters:
                break
            period = int(item["period"])
            year = int(item["year"])
            period_key = (year, period)
            if period_key in seen_periods:
                continue
            disclosure_index = int(item["disclosure_index"])
            reusable = reusable_quarters_by_key.get((year, period, disclosure_index))
            if reusable is not None:
                quarter_rows.append(copy.deepcopy(reusable))
                seen_periods.add(period_key)
                continue
            detail = _fetch_attachment_detail(disclosure_index, cfg)
            if not detail:
                if strict_errors:
                    raise RuntimeError(
                        f"KAP bildirim detayi bos dondu: disclosure_index={disclosure_index}"
                    )
                continue
            basic = (
                detail.get("disclosure", {})
                .get("disclosureBasic", {})
                if isinstance(detail.get("disclosure"), dict)
                else {}
            )
            metrics, unit_info = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=False,
                comparison_mode="current",
                is_bank=is_bank,
                is_insurance=is_insurance,
            )
            metrics_ytd, _ = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=True,
                comparison_mode="current",
                is_bank=is_bank,
                is_insurance=is_insurance,
            )
            metrics_comparative, _ = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=False,
                comparison_mode="comparative",
                is_bank=is_bank,
                is_insurance=is_insurance,
            )
            metrics_ytd_comparative, _ = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=True,
                comparison_mode="comparative",
                is_bank=is_bank,
                is_insurance=is_insurance,
            )
            quarter_rows.append(
                {
                    "year": year,
                    "period": period,
                    "quarter": _quarter_label(year, period),
                    "disclosure_index": disclosure_index,
                    "publish_date": str(basic.get("publishDate", "")).strip(),
                    "title": str(basic.get("title", "") or item.get("title", "")).strip(),
                    "stock_code": _normalize_disclosure_stock_code(
                        basic.get("stockCode", "") or item.get("stock_code", ""),
                        company_norm,
                    ),
                    "pdf_url": f"{KAP_BASE_URL}/{PDF_ENDPOINT}/{disclosure_index}",
                    "unit_raw": str(unit_info.get("raw", "")),
                    "currency": str(unit_info.get("currency", "TL")).upper(),
                    "metrics": metrics,
                    "metrics_quarterly": metrics,
                    "metrics_ytd": metrics_ytd,
                    "metrics_comparative": metrics_comparative,
                    "metrics_quarterly_comparative": metrics_comparative,
                    "metrics_ytd_comparative": metrics_ytd_comparative,
                }
            )
            seen_periods.add(period_key)

        if not quarter_rows:
            raise RuntimeError("KAP bildirim detayi alindi ancak ceyrek verisi parse edilemedi.")

        quarter_rows = _merge_live_and_cached_quarters(quarter_rows, cached, cache_write_depth)

        if strict_errors:
            expected_live_periods = min(requested_max_quarters, len(disclosures))
            actual_periods = _cached_period_count({"quarters": quarter_rows})
            if actual_periods < expected_live_periods:
                raise RuntimeError(
                    "KAP bilanco derinligi yetersiz: "
                    f"beklenen={expected_live_periods} alinan={actual_periods}"
                )

        payload: Dict[str, Any] = {
            "ok": True,
            "cache_hit": False,
            "cache_stale": False,
            "schema_version": KAP_CACHE_SCHEMA_VERSION,
            "company": str(quarter_rows[0].get("stock_code", "") or company_norm).strip().upper(),
            "company_title": str(member.get("title", "")).strip(),
            "stock_code": str(quarter_rows[0].get("stock_code", "")).strip(),
            "company_kind": company_kind,
            "member_oid": str(member.get("mkk_member_oid", "")).strip(),
            "source_url": f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{member.get('permalink', '')}",
            "fetched_at": _utc_now().isoformat(),
            "live_period_count": len(disclosures),
            "live_period_request_depth": requested_max_quarters,
            "quarters": quarter_rows,
        }
        if isinstance(cached, dict):
            for key in (
                "insurance_premium_disclosures",
                "insurance_premium_disclosures_checked_at",
                "insurance_premium_disclosures_version",
            ):
                if key in cached:
                    payload[key] = copy.deepcopy(cached[key])
        payload = _ensure_insurance_premium_disclosures(
            payload=payload,
            cache_path=cache_path,
            cfg=cfg,
            member=member,
            force_refresh=force_refresh,
            strict=strict_errors,
        )
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        # A batch job must see the provider failure; serving the previous cache
        # here would report a throttled or partial refresh as success.
        if strict_errors:
            raise
        if cached:
            cached["ok"] = bool(cached.get("ok", False))
            cached["cache_hit"] = True
            cached["cache_stale"] = True
            cached["error"] = str(exc)
            if (
                cached["ok"]
                and _cached_period_count(cached) > 0
                and cache_has_requested_depth
                and cache_version == KAP_CACHE_SCHEMA_VERSION
            ):
                cached["cache_stale"] = False
                cached.pop("error", None)
            return _ensure_insurance_premium_disclosures(
                payload=cached,
                cache_path=cache_path,
                cfg=cfg,
                member=member,
                force_refresh=False,
            )
        premium_only = _fetch_premium_only_snapshot(
            company_norm=company_norm,
            cache_path=cache_path,
            cfg=cfg,
            member=member,
            error=exc,
        )
        if premium_only:
            return premium_only
        return {
            "ok": False,
            "cache_hit": False,
            "cache_stale": False,
            "company": company_norm,
            "company_title": str((member or {}).get("title") or "").strip(),
            "stock_code": company_norm,
            "error": str(exc),
            "quarters": [],
        }
