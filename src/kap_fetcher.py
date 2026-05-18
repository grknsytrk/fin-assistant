from __future__ import annotations

import html
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from src.config import KapConfig

KAP_BASE_URL = "https://www.kap.org.tr/tr/api"
MEMBER_FILTER_ENDPOINT = "member/filter"
LIST_COMPANY_EXCEL_MEMBERS_ENDPOINT = "financialTable/listCompanyExcelMembers"
ATTACHMENT_DETAIL_ENDPOINT = "notification/attachment-detail"
PDF_ENDPOINT = "BildirimPdf"
DISCLOSURE_MEMBERS_BY_CRITERIA_ENDPOINT = "disclosure/members/byCriteria"
FILE_DOWNLOAD_ENDPOINT = "file/download"
KAP_CACHE_SCHEMA_VERSION = 11
KAP_LIVE_DISCLOSURE_CHECK_TTL_HOURS = 24.0
KAP_INSURANCE_PREMIUM_CHECK_TTL_HOURS = 24.0
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
    "KOZAL": ["TRALT", "KOZAL", "KOZA ALTIN"],
    "TRALT": ["TRALT", "KOZAL", "KOZA ALTIN"],
    "PETKM": ["PETKM", "PETKIM"],
    "SAHOL": ["SAHOL", "SABANCI"],
    "SISE": ["SISE", "SISECAM"],
    "THYAO": ["THYAO", "THY", "TURK HAVA YOLLARI"],
    "TOASO": ["TOASO", "TOFAS"],
    "TUPRS": ["TUPRS", "TUPRAS"],
    "YKBNK": ["YKBNK", "YAPI KREDI"],
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
}

MARKETVISUALS_INSURANCE_HAYAT_URL = "https://marketvisuals.net/insurance_hayat.html"
MARKETVISUALS_INSURANCE_HAYATDISI_URL = "https://marketvisuals.net/insurance_hayatdisi.html"
MARKETVISUALS_TSB_SOURCE_LABEL = "TSB prim üretimi (MarketVisuals derlemesi)"
MARKETVISUALS_INSURANCE_COMPANY_PAGES: Dict[str, Tuple[str, ...]] = {
    "AGESA": (MARKETVISUALS_INSURANCE_HAYAT_URL,),
    "ANHYT": (MARKETVISUALS_INSURANCE_HAYAT_URL,),
    "AKGRT": (MARKETVISUALS_INSURANCE_HAYATDISI_URL,),
    "ANSGR": (MARKETVISUALS_INSURANCE_HAYATDISI_URL,),
    "RAYSG": (MARKETVISUALS_INSURANCE_HAYATDISI_URL,),
    "TURSG": (MARKETVISUALS_INSURANCE_HAYATDISI_URL,),
}
MARKETVISUALS_INSURANCE_COMPANY_ALIASES: Dict[str, List[str]] = {
    "AGESA": ["agesa hayat", "agesa"],
    "ANHYT": ["anadolu hayat emeklilik", "anadolu hayat"],
    "AKGRT": ["aksigorta"],
    "ANSGR": ["anadolu anonim turk sigorta", "anadolu sigorta"],
    "RAYSG": ["ray sigorta"],
    "TURSG": ["turkiye sigorta"],
}
MARKETVISUALS_INSURANCE_COMPANY_TITLES: Dict[str, str] = {
    "AGESA": "AgeSA Hayat ve Emeklilik AŞ",
    "ANHYT": "Anadolu Hayat Emeklilik AŞ",
    "AKGRT": "Aksigorta AŞ",
    "ANSGR": "Anadolu Anonim Türk Sigorta Şirketi",
    "RAYSG": "Ray Sigorta AŞ",
    "TURSG": "Türkiye Sigorta AŞ",
}

TURKISH_MONTH_NAME_TO_NUMBER = {
    "ocak": 1,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "eylul": 9,
    "ekim": 10,
    "kasim": 11,
    "aralik": 12,
}

TR_NORMALIZE_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)

_LABEL_PATTERN = re.compile(
    r'<div class="gwt-Label multi-language-content content-tr"[^>]*>(.*?)</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
_VALUE_PATTERN = re.compile(
    r'<td class="taxonomy-context-value col-order-class-(\d+)"[^>]*>\s*'
    r"<div>\s*<div[^>]*title=\"([^\"]+)\"",
    flags=re.IGNORECASE | re.DOTALL,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(text: str) -> str:
    lowered = str(text or "").strip().lower().translate(TR_NORMALIZE_MAP)
    lowered = unicodedata.normalize("NFKD", lowered)
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    return " ".join(lowered.split())


def _clean_html_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(raw or ""))
    text = html.unescape(text)
    return " ".join(text.split())


def _parse_numeric_token(raw: str) -> Optional[float]:
    token = str(raw or "").strip()
    if not token:
        return None

    token = token.replace("\xa0", "").replace(" ", "")
    negative = False
    if token.startswith("(") and token.endswith(")"):
        negative = True
        token = token[1:-1]
    if token.startswith("-"):
        negative = True
        token = token[1:]
    if token.startswith("+"):
        token = token[1:]

    token = re.sub(r"[^0-9\.,]", "", token)
    if not token:
        return None

    # Normalize decimal/thousand separators.
    if "." in token and "," in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        if token.count(",") > 1:
            token = token.replace(",", "")
        else:
            left, right = token.split(",", 1)
            if len(right) == 3 and left:
                token = left + right
            else:
                token = left + "." + right
    elif "." in token:
        if token.count(".") > 1:
            token = token.replace(".", "")
        else:
            left, right = token.split(".", 1)
            if len(right) == 3 and left:
                token = left + right

    try:
        value = float(token)
    except Exception:
        return None
    if negative:
        value = -abs(value)
    return value


def _is_favok_label(label_norm: str) -> bool:
    ln = str(label_norm or "")
    if not ln:
        return False
    # Exclude margin-style rows when present.
    if "favok marj" in ln or "fvaok marj" in ln or "ebitda marj" in ln:
        return False
    if "favok" in ln or "fvaok" in ln or "ebitda" in ln:
        return True

    # Expanded wording variants:
    # "faiz/vergi/amortisman/oncesi kar(kazanc)" and close forms.
    has_finance_tax = "faiz" in ln and "vergi" in ln
    has_depr = "amortisman" in ln or "itfa" in ln
    has_profit = "kar" in ln or "kazanc" in ln
    has_before = "oncesi" in ln or "oncesindeki" in ln
    return has_finance_tax and has_depr and has_profit and has_before


def _is_cash_equivalent_label(label_norm: str) -> bool:
    ln = str(label_norm or "")
    if not ln:
        return False
    if ln in {"nakit ve nakit benzerleri", "nakit ve nakit benzeri varliklar"}:
        return True
    if "nakit ve nakit benzer" not in ln:
        return False
    if any(
        token in ln
        for token in (
            "net artis",
            "net azalis",
            "donem basi",
            "donem sonu",
            "cevrim fark",
            "etkisi",
            "diger ",
        )
    ):
        return False
    return True


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
        if cached:
            return path, cached
    return primary_path, None


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
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _http_get_json(url: str, cfg: KapConfig) -> Any:
    last_error: Optional[BaseException] = None
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept="application/json, text/plain, */*")
        headers["User-Agent"] = user_agent
        for attempt in range(2):
            request = urllib.request.Request(url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429 and attempt >= 1:
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
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept="text/html,application/xhtml+xml,*/*")
        headers["User-Agent"] = user_agent
        for attempt in range(2):
            request = urllib.request.Request(url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429 and attempt >= 1:
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
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept="application/json, text/plain, */*")
        headers["User-Agent"] = user_agent
        headers["Content-Type"] = "application/json"
        for attempt in range(2):
            request = urllib.request.Request(url, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429 and attempt >= 1:
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
    for user_agent in _candidate_user_agents(cfg):
        headers = _kap_request_headers(cfg, accept=accept)
        headers["User-Agent"] = user_agent
        for attempt in range(2):
            request = urllib.request.Request(url, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = int(getattr(exc, "code", 0) or 0)
                if code not in KAP_RETRYABLE_HTTP_CODES:
                    raise
                if code == 429 and attempt >= 1:
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


def _resolve_member(company: str, cfg: KapConfig) -> Optional[Dict[str, Any]]:
    company_key = str(company or "").strip().upper()
    if not company_key:
        return None

    candidate_queries = COMPANY_QUERY_ALIASES.get(company_key, [])
    if company_key not in candidate_queries:
        candidate_queries.append(company_key)
    # Fallback: split values like NETCAD_4Q -> NETCAD
    stripped = re.sub(r"[^A-Za-z0-9]+", " ", company_key).split(" ")
    if stripped:
        main_token = stripped[0].strip().upper()
        if main_token and main_token not in candidate_queries:
            candidate_queries.append(main_token)

    for query in candidate_queries:
        try:
            rows = _http_get_json(_member_filter_url(query), cfg)
        except urllib.error.HTTPError:
            continue
        except Exception:
            continue
        if not isinstance(rows, list) or not rows:
            continue
        row = dict(rows[0] or {})
        mkk_oid = str(row.get("mkkMemberOid", "")).strip()
        if not mkk_oid:
            continue
        return {
            "company_code": str(row.get("companyCode", "")).strip(),
            "mkk_member_oid": mkk_oid,
            "title": str(row.get("title", "")).strip(),
            "permalink": str(row.get("permaLink", "")).strip(),
            "query": query,
        }
    fallback = KAP_MEMBER_FALLBACKS.get(company_key)
    return dict(fallback) if fallback else None


def _list_company_disclosures(
    member_oid: str,
    cfg: KapConfig,
    max_years_back: int = 6,
    max_periods: Optional[int] = None,
) -> List[Dict[str, Any]]:
    current_year = _utc_now().year
    rows: List[Dict[str, Any]] = []
    requested_periods = max(1, int(max_periods)) if max_periods is not None else None

    for idx, year in enumerate(range(current_year, current_year - max_years_back - 1, -1)):
        if idx > 0:
            time.sleep(0.7)
        url = f"{KAP_BASE_URL}/{LIST_COMPANY_EXCEL_MEMBERS_ENDPOINT}/{member_oid}/{year}/T"
        try:
            payload = _http_get_json(url, cfg)
        except urllib.error.HTTPError as exc:
            if int(getattr(exc, "code", 0) or 0) == 429:
                if not rows:
                    fallback_rows = _list_financial_report_disclosures_by_criteria(
                        member_oid=member_oid,
                        cfg=cfg,
                        max_years_back=max_years_back,
                        max_periods=requested_periods,
                    )
                    if fallback_rows:
                        return fallback_rows
                    raise RuntimeError("KAP istek limiti nedeniyle finansal bildirimler şu anda alınamadı.")
                break
            continue
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
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
                    "pd_oid": str(item.get("pdOid", "")).strip(),
                    "mkk_member_oid": str(item.get("mkkMemberOid", "")).strip(),
                }
            )
        if requested_periods is not None:
            period_count = len({(row["year"], row["period"]) for row in rows})
            if period_count >= requested_periods:
                break

    unique: Dict[int, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda x: (x["year"], x["period"], x["disclosure_index"]), reverse=True):
        if row["disclosure_index"] in unique:
            continue
        unique[row["disclosure_index"]] = row
    current_period_count = len({(row["year"], row["period"]) for row in unique.values()})
    needs_fallback = not unique or (
        requested_periods is not None and current_period_count < requested_periods
    )
    if needs_fallback:
        fallback_rows = _list_financial_report_disclosures_by_criteria(
            member_oid=member_oid,
            cfg=cfg,
            max_years_back=max_years_back,
            max_periods=requested_periods,
        )
        if fallback_rows:
            by_period: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for row in fallback_rows:
                try:
                    by_period[(int(row["year"]), int(row["period"]))] = row
                except (KeyError, TypeError, ValueError):
                    continue
            for row in unique.values():
                try:
                    by_period[(int(row["year"]), int(row["period"]))] = row
                except (KeyError, TypeError, ValueError):
                    continue
            merged = sorted(
                by_period.values(),
                key=lambda item: (
                    int(item.get("year") or 0),
                    int(item.get("period") or 0),
                    int(item.get("disclosure_index") or 0),
                ),
                reverse=True,
            )
            if requested_periods is not None:
                return merged[:requested_periods]
            return merged
    return list(unique.values())


def _list_financial_report_disclosures_by_criteria(
    *,
    member_oid: str,
    cfg: KapConfig,
    max_years_back: int = 6,
    max_periods: Optional[int] = None,
) -> List[Dict[str, Any]]:
    lookback_days = max(365, int(max_years_back) * 370)
    rows = _list_member_disclosures_by_criteria(
        member_oid=member_oid,
        cfg=cfg,
        lookback_days=lookback_days,
    )
    by_period: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in rows:
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
        key = (year, period)
        existing = by_period.get(key)
        if existing and int(existing.get("disclosure_index") or 0) >= disclosure_index:
            continue
        by_period[key] = {
            "year": year,
            "period": period,
            "disclosure_index": disclosure_index,
            "stock_code": "",
            "title": str(row.get("title") or row.get("summary") or "").strip(),
        }

    ordered = sorted(
        by_period.values(),
        key=lambda item: (
            int(item.get("year") or 0),
            int(item.get("period") or 0),
            int(item.get("disclosure_index") or 0),
        ),
        reverse=True,
    )
    if max_periods is not None:
        return ordered[: max(1, int(max_periods))]
    return ordered


def _parse_unit_info(html_block: str) -> Dict[str, Any]:
    raw = ""
    multiplier = 1.0
    currency = ""

    unit_match = re.search(
        r"Sunum Para Birimi</td>\s*<td>(.*?)</td>",
        html_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if unit_match:
        raw = _clean_html_text(unit_match.group(1))

    unit_norm = _normalize(raw)
    if "1.000.000.000" in raw:
        multiplier = 1_000_000_000.0
    elif "1.000.000" in raw:
        multiplier = 1_000_000.0
    elif "1.000" in raw:
        multiplier = 1_000.0
    elif "milyar" in unit_norm:
        multiplier = 1_000_000_000.0
    elif "milyon" in unit_norm:
        multiplier = 1_000_000.0
    elif "bin" in unit_norm:
        multiplier = 1_000.0

    upper_raw = raw.upper()
    for code in ("TRY", "TL", "EUR", "USD"):
        if code in upper_raw:
            currency = "TL" if code == "TRY" else code
            break
    if not currency:
        currency = "TL"

    return {"raw": raw, "multiplier": multiplier, "currency": currency}


def _extract_rows_from_disclosure_body(body_html: str, body_index: int, unit_multiplier: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in _VALUE_PATTERN.finditer(body_html):
        col_order = int(match.group(1))
        parsed_value = _parse_numeric_token(match.group(2))
        if parsed_value is None:
            continue
        raw_value = float(parsed_value)
        pre = body_html[max(0, match.start() - 2800) : match.start()]
        labels = [_clean_html_text(item) for item in _LABEL_PATTERN.findall(pre)]
        if not labels:
            continue
        label = labels[-1].strip()
        if not label:
            continue
        label_norm = _normalize(label)
        if label_norm in {
            "",
            "cari donem",
            "onceki donem",
            "dipnot referansi",
        }:
            continue
        rows.append(
            {
                "body_index": body_index,
                "label": label,
                "label_norm": label_norm,
                "col_order": col_order,
                "value": raw_value * unit_multiplier,
            }
        )
    return rows


def _col_preference(
    period: int,
    income_statement: bool,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
) -> Tuple[int, ...]:
    if comparison_mode == "comparative":
        if income_statement and int(period) > 1:
            return (5, 7, 4, 6) if prefer_income_statement_ytd else (7, 5, 6, 4)
        if income_statement:
            return (5, 7, 4, 6) if prefer_income_statement_ytd else (7, 5, 6, 4)
        return (5, 7, 4, 6)
    if income_statement and int(period) > 1:
        # Income statement rows usually expose: 4=YTD current, 5=YTD prev, 6=quarter current, 7=quarter prev.
        return (4, 6, 5, 7) if prefer_income_statement_ytd else (6, 4, 7, 5)
    if income_statement:
        return (4, 6, 5, 7)
    return (4, 6, 5, 7)


def _score_metric_candidate(
    metric_key: str,
    row: Dict[str, Any],
    period: int,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
) -> int:
    score = 0
    label_norm = str(row.get("label_norm", ""))
    body_index = int(row.get("body_index", -1))
    col_order = int(row.get("col_order", -1))

    if metric_key in {
        "net_kar",
        "satis_gelirleri",
        "brut_kar",
        "favok",
        "faiz_gelirleri",
        "faiz_giderleri",
        "net_ucret_komisyon_gelirleri",
        "net_faaliyet_kari",
        "esas_faaliyet_kari",
        "amortisman_itfa_gideri",
        "prim_uretimi",
        "alinan_net_primler",
        "teknik_gelirler",
        "teknik_denge",
    } and body_index == 1:
        score += 30
    if metric_key in {"faaliyet_nakit_akisi", "capex"} and body_index == 2:
        score += 30
    if metric_key in {
        "donen_varliklar",
        "duran_varliklar",
        "toplam_varliklar",
        "kisa_vadeli_yukumlulukler",
        "finansal_varliklar_net",
        "krediler",
        "mevduatlar",
        "beklenen_zarar_karsiliklari",
        "finansal_borclar",
        "net_borc",
        "ozkaynaklar",
        "nakit_ve_nakit_benzerleri",
        "finansal_varliklar_sigortacilik",
        "esas_faaliyetlerden_alacaklar",
        "teknik_karsiliklar",
        "esas_faaliyetlerden_borclar",
        "odenmis_sermaye",
        "cikarilmis_sermaye",
    } and body_index == 0:
        score += 30

    preferred_cols = _col_preference(
        period=period,
        income_statement=metric_key in {
            "net_kar",
            "satis_gelirleri",
            "brut_kar",
            "favok",
            "faiz_gelirleri",
            "faiz_giderleri",
            "net_ucret_komisyon_gelirleri",
            "net_faaliyet_kari",
            "esas_faaliyet_kari",
            "amortisman_itfa_gideri",
        },
        prefer_income_statement_ytd=prefer_income_statement_ytd,
        comparison_mode=comparison_mode,
    )
    if col_order in preferred_cols:
        score += max(1, 20 - preferred_cols.index(col_order) * 6)

    if metric_key == "net_kar":
        # Prefer attributable/explicit net-period profit rows over generic period-profit
        # lines, which may point to pre-attribution or continuing-operations subtotals.
        if "ana ortaklik paylari" in label_norm:
            if label_norm == "ana ortaklik paylari" and body_index == 1:
                score += 135
            elif "donem kari" in label_norm or "net donem kari" in label_norm:
                score += 125
            else:
                score -= 120
        if label_norm == "net donem kari veya zarari":
            score += 140
        elif "net donem kari veya zarari" in label_norm:
            score += 110
        elif "net donem kari" in label_norm:
            score += 90
        elif label_norm == "donem kari (zarari)":
            score += 35
        elif "donem kari (zarari)" in label_norm and "surdurulen faaliyetler" not in label_norm:
            score += 20
        if "kapsamli gelir" in label_norm:
            score -= 100

        # Consolidated statements often include multiple profit layers.
        # We prefer headline net/parent-profit rows over "continued operations".
        if "surdurulen faaliyetler donem kari" in label_norm:
            score -= 60
        if "kontrol gucu olmayan paylar" in label_norm:
            score -= 80
    elif metric_key == "satis_gelirleri":
        if label_norm == "toplam hasilat" or label_norm == "toplam satis gelirleri":
            score += 140
        elif "toplam hasilat" in label_norm or "toplam satis gelirleri" in label_norm:
            score += 115
        if "hasilat" in label_norm:
            score += 35
        if "satis gelirleri" in label_norm:
            score += 25
        if "finans sektoru faaliyetleri hasilati" in label_norm:
            score -= 40
    elif metric_key == "brut_kar":
        if "brut kar" in label_norm and "ticari faaliyetlerden" not in label_norm:
            score += 35
        elif "brut kar" in label_norm:
            score += 25
    elif metric_key == "favok":
        if _is_favok_label(label_norm):
            score += 40
        elif "faaliyet kari" in label_norm and "amortisman" in label_norm:
            score += 20
    elif metric_key == "faiz_gelirleri":
        if "faiz gelirleri" in label_norm:
            score += 45
        if "faiz giderleri" in label_norm:
            score -= 40
    elif metric_key == "faiz_giderleri":
        if "faiz giderleri" in label_norm:
            score += 45
        if "faiz gelirleri" in label_norm:
            score -= 40
    elif metric_key == "net_ucret_komisyon_gelirleri":
        if "net ucret ve komisyon gelirleri" in label_norm:
            score += 45
        elif "net ucret komisyon gelirleri" in label_norm:
            score += 40
    elif metric_key == "net_faaliyet_kari":
        if "net faaliyet kari (zarari)" in label_norm or "net faaliyet kari" in label_norm:
            score += 45
    elif metric_key == "esas_faaliyet_kari":
        if "esas faaliyet kari" in label_norm:
            score += 40
        elif "finansman geliri (gideri) oncesi faaliyet kari" in label_norm:
            score += 20
    elif metric_key == "amortisman_itfa_gideri":
        if "amortisman ve itfa gideri" in label_norm:
            score += 45
        elif "amortisman" in label_norm:
            score += 20
    elif metric_key == "prim_uretimi":
        if "brut yazilan primler" in label_norm:
            score += 110
        elif "yazilan primler" in label_norm and "reasuror payi dusulmus" in label_norm:
            score -= 30
    elif metric_key == "alinan_net_primler":
        if "yazilan primler" in label_norm and "reasuror payi dusulmus" in label_norm:
            score += 105
    elif metric_key == "teknik_gelirler":
        if "hayat disi teknik gelir" in label_norm:
            score += 110
        elif "teknik gelir" in label_norm and "diger teknik gelirler" not in label_norm:
            score += 40
    elif metric_key == "teknik_denge":
        if "teknik bolum dengesi - hayat disi" in label_norm:
            score += 110
        elif "genel teknik bolum dengesi" in label_norm:
            score += 95
    elif metric_key == "faaliyet_nakit_akisi":
        if "isletme faaliyetlerinden nakit akislari" in label_norm:
            score += 35
        elif "faaliyetlerden elde edilen nakit akis" in label_norm:
            score += 25
    elif metric_key == "capex":
        if "duran varliklarin alimindan kaynaklanan nakit cikislari" in label_norm:
            score += 40
        elif "maddi ve maddi olmayan duran varliklarin alimindan kaynaklanan nakit cikislari" in label_norm:
            score += 35
        elif "nakit cikis" in label_norm and "duran varlik" in label_norm:
            score += 20
    elif metric_key == "donen_varliklar":
        if "toplam donen varliklar" in label_norm:
            score += 80
        elif "donen varliklar" in label_norm:
            score += 20
        if "diger donen varliklar" in label_norm:
            score -= 35
    elif metric_key == "duran_varliklar":
        if "toplam duran varliklar" in label_norm:
            score += 80
        elif "duran varliklar" in label_norm:
            score += 20
        if "maddi duran varliklar" in label_norm or "maddi olmayan duran varliklar" in label_norm:
            score -= 35
    elif metric_key == "toplam_varliklar":
        if "toplam varliklar" in label_norm:
            score += 80
    elif metric_key == "finansal_varliklar_net":
        if "finansal varliklar (net)" in label_norm or "finansal varliklar net" in label_norm:
            score += 80
        elif "finansal varliklar" in label_norm:
            score += 30
    elif metric_key == "krediler":
        if label_norm == "krediler":
            score += 80
        elif "krediler" in label_norm:
            score += 35
    elif metric_key == "mevduatlar":
        if label_norm == "mevduatlar":
            score += 80
        elif "mevduatlar" in label_norm:
            score += 35
    elif metric_key == "beklenen_zarar_karsiliklari":
        if "beklenen zarar karsiliklari" in label_norm:
            score += 80
        elif "beklenen kredi zarar karsiliklari" in label_norm:
            score += 70
    elif metric_key == "nakit_ve_nakit_benzerleri":
        if _is_cash_equivalent_label(label_norm):
            score += 110
        if "diger nakit ve nakit benzeri varliklar" in label_norm:
            score -= 80
    elif metric_key == "finansal_varliklar_sigortacilik":
        if "finansal varliklar ile riski sigortalilara ait finansal yatirimlar" in label_norm:
            score += 110
    elif metric_key == "esas_faaliyetlerden_alacaklar":
        if label_norm == "esas faaliyetlerden alacaklar":
            score += 110
        elif "esas faaliyetlerden alacaklar" in label_norm:
            score += 55
    elif metric_key == "teknik_karsiliklar":
        if "sigortacilik teknik karsiliklari" in label_norm:
            score += 110
        elif "teknik karsilik" in label_norm and "diger teknik karsiliklar" not in label_norm:
            score += 50
    elif metric_key == "esas_faaliyetlerden_borclar":
        if label_norm == "esas faaliyetlerden borclar":
            score += 110
        elif "esas faaliyetlerden borclar" in label_norm:
            score += 55
    elif metric_key == "kisa_vadeli_yukumlulukler":
        if "toplam kisa vadeli yukumlulukler" in label_norm:
            score += 80
        elif "kisa vadeli yukumlulukler" in label_norm:
            score += 25
        if "ticari borc" in label_norm or "diger borc" in label_norm:
            score -= 30
    elif metric_key == "finansal_borclar":
        if "finansal borclar" in label_norm or "toplam finansal yukumlulukler" in label_norm:
            score += 40
        elif "kisa vadeli yukumlulukler" in label_norm or "uzun vadeli yukumlulukler" in label_norm:
            score += 15
    elif metric_key == "net_borc":
        if "net borc" in label_norm:
            score += 40
    elif metric_key == "ozkaynaklar":
        if "ana ortakliga ait ozkaynaklar" in label_norm:
            score += 95
        elif "ozsermaye toplami" in label_norm:
            score += 100
        elif "ana ortakliga ait" in label_norm and "ozkaynaklar" in label_norm:
            score += 85
        elif "toplam ozkaynaklar" in label_norm:
            score += 45
        elif "ozkaynaklar" in label_norm:
            score += 30
        elif "ozsermaye" in label_norm:
            score += 30
        if "kontrol gucu olmayan paylar" in label_norm:
            score -= 40
        if "toplam yukumlulukler ve ozsermaye" in label_norm:
            score -= 90
    elif metric_key == "odenmis_sermaye":
        if label_norm == "odenmis sermaye":
            score += 120
        elif "odenmis sermaye" in label_norm:
            score += 80
        if "sermaye duzeltme farklari" in label_norm:
            score -= 120
    elif metric_key == "cikarilmis_sermaye":
        if label_norm == "cikarilmis sermaye":
            score += 120
        elif "cikarilmis sermaye" in label_norm:
            score += 80
        if "sermaye duzeltme farklari" in label_norm:
            score -= 120

    if label_norm.startswith("toplam") or "ara toplam" in label_norm:
        if metric_key in {"donen_varliklar", "duran_varliklar", "toplam_varliklar", "ozkaynaklar"}:
            score += 20
        else:
            score -= 20
    return score


def _pick_metric_value(
    metric_key: str,
    rows: List[Dict[str, Any]],
    period: int,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
) -> Optional[float]:
    filtered: List[Dict[str, Any]] = []

    if metric_key == "net_kar":
        explicit_net_rows = [
            row
            for row in rows
            if "net donem kari veya zarari" in str(row.get("label_norm", ""))
        ]
        if explicit_net_rows:
            filtered = explicit_net_rows

    for row in rows:
        label_norm = str(row.get("label_norm", ""))
        if metric_key == "net_kar":
            if filtered:
                continue
            if "kontrol gucu olmayan paylar" in label_norm:
                continue
            is_bare_parent_share = "ana ortaklik paylari" in label_norm and (
                "donem kari" not in label_norm and "net donem kari" not in label_norm
            )
            if is_bare_parent_share and int(row.get("body_index", -1)) != 1:
                continue
            if (
                "ana ortaklik paylari" in label_norm
                or "net donem kari veya zarari" in label_norm
                or "net donem kari" in label_norm
                or "donem kari (zarari)" in label_norm
            ):
                filtered.append(row)
        elif metric_key == "satis_gelirleri":
            if "hasilat" in label_norm or "satis gelirleri" in label_norm:
                filtered.append(row)
        elif metric_key == "brut_kar":
            if "brut kar" in label_norm:
                filtered.append(row)
        elif metric_key == "favok":
            if _is_favok_label(label_norm):
                filtered.append(row)
        elif metric_key == "faiz_gelirleri":
            if "faiz gelirleri" in label_norm and "faiz giderleri" not in label_norm:
                filtered.append(row)
        elif metric_key == "faiz_giderleri":
            if "faiz giderleri" in label_norm:
                filtered.append(row)
        elif metric_key == "net_ucret_komisyon_gelirleri":
            if "net ucret ve komisyon gelirleri" in label_norm or "net ucret komisyon gelirleri" in label_norm:
                filtered.append(row)
        elif metric_key == "net_faaliyet_kari":
            if "net faaliyet kari" in label_norm:
                filtered.append(row)
        elif metric_key == "esas_faaliyet_kari":
            if "esas faaliyet kari" in label_norm or "finansman geliri (gideri) oncesi faaliyet kari" in label_norm:
                filtered.append(row)
        elif metric_key == "amortisman_itfa_gideri":
            if "amortisman ve itfa gideri" in label_norm or "amortisman" in label_norm:
                filtered.append(row)
        elif metric_key == "prim_uretimi":
            if "brut yazilan primler" in label_norm:
                filtered.append(row)
        elif metric_key == "alinan_net_primler":
            if "yazilan primler" in label_norm and "reasuror payi dusulmus" in label_norm:
                filtered.append(row)
        elif metric_key == "teknik_gelirler":
            if "hayat disi teknik gelir" in label_norm:
                filtered.append(row)
        elif metric_key == "teknik_denge":
            if "teknik bolum dengesi - hayat disi" in label_norm or "genel teknik bolum dengesi" in label_norm:
                filtered.append(row)
        elif metric_key == "faaliyet_nakit_akisi":
            if "isletme faaliyetlerinden nakit akis" in label_norm or "faaliyetlerden elde edilen nakit akis" in label_norm:
                filtered.append(row)
        elif metric_key == "capex":
            if "nakit cikis" in label_norm and "duran varlik" in label_norm and "alim" in label_norm:
                filtered.append(row)
        elif metric_key == "nakit_ve_nakit_benzerleri":
            if _is_cash_equivalent_label(label_norm):
                filtered.append(row)
        elif metric_key == "finansal_varliklar_sigortacilik":
            if "finansal varliklar ile riski sigortalilara ait finansal yatirimlar" in label_norm:
                filtered.append(row)
        elif metric_key == "esas_faaliyetlerden_alacaklar":
            if label_norm == "esas faaliyetlerden alacaklar" or "esas faaliyetlerden alacaklar" in label_norm:
                filtered.append(row)
        elif metric_key == "teknik_karsiliklar":
            if "sigortacilik teknik karsiliklari" in label_norm:
                filtered.append(row)
            elif "teknik karsilik" in label_norm and "diger teknik karsiliklar" not in label_norm:
                filtered.append(row)
        elif metric_key == "esas_faaliyetlerden_borclar":
            if label_norm == "esas faaliyetlerden borclar" or "esas faaliyetlerden borclar" in label_norm:
                filtered.append(row)
        elif metric_key == "donen_varliklar":
            if "toplam donen varliklar" in label_norm or label_norm == "donen varliklar":
                filtered.append(row)
        elif metric_key == "duran_varliklar":
            if "toplam duran varliklar" in label_norm or label_norm == "duran varliklar":
                filtered.append(row)
        elif metric_key == "toplam_varliklar":
            if "toplam varliklar" in label_norm:
                filtered.append(row)
        elif metric_key == "finansal_varliklar_net":
            if "finansal varliklar (net)" in label_norm or "finansal varliklar net" in label_norm:
                filtered.append(row)
            elif label_norm == "finansal varliklar":
                filtered.append(row)
        elif metric_key == "krediler":
            if label_norm == "krediler" or "krediler" in label_norm:
                filtered.append(row)
        elif metric_key == "mevduatlar":
            if label_norm == "mevduatlar" or "mevduatlar" in label_norm:
                filtered.append(row)
        elif metric_key == "beklenen_zarar_karsiliklari":
            if "beklenen zarar karsiliklari" in label_norm or "beklenen kredi zarar karsiliklari" in label_norm:
                filtered.append(row)
        elif metric_key == "kisa_vadeli_yukumlulukler":
            if "toplam kisa vadeli yukumlulukler" in label_norm or label_norm == "kisa vadeli yukumlulukler":
                filtered.append(row)
        elif metric_key == "finansal_borclar":
            if "finansal borclar" in label_norm or "toplam finansal yukumlulukler" in label_norm:
                filtered.append(row)
        elif metric_key == "net_borc":
            if "net borc" in label_norm:
                filtered.append(row)
        elif metric_key == "ozkaynaklar":
            if "ana ortakliga ait ozkaynaklar" in label_norm:
                filtered.append(row)
            elif "ozsermaye toplami" in label_norm:
                filtered.append(row)
            elif "toplam ozkaynaklar" in label_norm:
                filtered.append(row)
            elif "ozkaynaklar" in label_norm:
                filtered.append(row)
            elif "ozsermaye" in label_norm and "toplam yukumlulukler ve ozsermaye" not in label_norm:
                filtered.append(row)
        elif metric_key == "odenmis_sermaye":
            if label_norm == "odenmis sermaye" or "odenmis sermaye" in label_norm:
                filtered.append(row)
        elif metric_key == "cikarilmis_sermaye":
            if label_norm == "cikarilmis sermaye" or "cikarilmis sermaye" in label_norm:
                filtered.append(row)

    if not filtered:
        return None

    scored = sorted(
        filtered,
        key=lambda item: (
            _score_metric_candidate(
                metric_key,
                item,
                period,
                prefer_income_statement_ytd=prefer_income_statement_ytd,
                comparison_mode=comparison_mode,
            ),
            abs(float(item.get("value", 0.0))),
        ),
        reverse=True,
    )
    return float(scored[0].get("value", 0.0))


def _pick_best_row(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    includes: Tuple[str, ...],
    excludes: Tuple[str, ...] = (),
    body_index: Optional[int] = None,
    comparison_mode: str = "current",
) -> Optional[Dict[str, Any]]:
    preferred_cols = _col_preference(
        period=period,
        income_statement=False,
        comparison_mode=comparison_mode,
    )
    candidates: List[Tuple[int, float, Dict[str, Any]]] = []
    for row in rows:
        if body_index is not None and int(row.get("body_index", -1)) != int(body_index):
            continue
        label_norm = str(row.get("label_norm", ""))
        if not all(token in label_norm for token in includes):
            continue
        if any(token in label_norm for token in excludes):
            continue
        col_order = int(row.get("col_order", -1))
        col_rank = preferred_cols.index(col_order) if col_order in preferred_cols else 99
        value = float(row.get("value", 0.0))
        candidates.append((col_rank, -abs(value), row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _derive_finansal_borclar(
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str = "current",
) -> Optional[float]:
    def _bucket(*, includes: Tuple[str, ...], excludes: Tuple[str, ...] = ()) -> Optional[float]:
        non_related_excludes = excludes + ("iliskili taraf",)
        row = _pick_best_row(
            rows=rows,
            period=period,
            includes=includes,
            excludes=non_related_excludes,
            body_index=0,
            comparison_mode=comparison_mode,
        )
        if row is None:
            row = _pick_best_row(
                rows=rows,
                period=period,
                includes=includes,
                excludes=excludes,
                body_index=0,
                comparison_mode=comparison_mode,
            )
        if row is None:
            return None
        return float(row.get("value", 0.0))

    short_term = _bucket(
        includes=("kisa vadeli borclanmalar",),
        excludes=("uzun vadeli borclanmalarin kisa vadeli kisimlari",),
    )
    long_term_current = _bucket(includes=("uzun vadeli borclanmalarin kisa vadeli kisimlari",))
    long_term = _bucket(
        includes=("uzun vadeli borclanmalar",),
        excludes=("kisa vadeli kisim",),
    )

    pieces = [value for value in (short_term, long_term_current, long_term) if value is not None]
    if pieces:
        return float(sum(pieces))

    direct_total = _bucket(
        includes=("finansal", "borc"),
        excludes=("ticari", "diger borc"),
    )
    if direct_total is not None:
        return float(direct_total)
    return None


def _derive_net_borc(
    rows: List[Dict[str, Any]],
    period: int,
    finansal_borclar: Optional[float],
    comparison_mode: str = "current",
) -> Optional[float]:
    direct_row = _pick_best_row(
        rows=rows,
        period=period,
        includes=("net borc",),
        body_index=0,
        comparison_mode=comparison_mode,
    )
    if direct_row is not None:
        return float(direct_row.get("value", 0.0))
    if finansal_borclar is None:
        return None

    nakit_row = _pick_best_row(
        rows=rows,
        period=period,
        includes=("nakit ve nakit benzer",),
        body_index=0,
        comparison_mode=comparison_mode,
    )
    nakit = float(nakit_row.get("value", 0.0)) if nakit_row else 0.0

    cash_like_finansal_yatirim = _pick_cash_like_financial_investment(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
    )

    if nakit == 0.0 and cash_like_finansal_yatirim is None:
        return None
    return float(finansal_borclar) - nakit - float(cash_like_finansal_yatirim or 0.0)


def _pick_cash_like_financial_investment(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str = "current",
) -> Optional[float]:
    preferred_cols = _col_preference(
        period=period,
        income_statement=False,
        comparison_mode=comparison_mode,
    )
    primary_col = preferred_cols[0] if preferred_cols else 4
    generic_rows: List[Dict[str, Any]] = []
    explicit_rows: List[Dict[str, Any]] = []

    for row in rows:
        if int(row.get("body_index", -1)) != 0:
            continue
        col_order = int(row.get("col_order", -1))
        if col_order != primary_col:
            continue
        label_norm = str(row.get("label_norm", ""))
        if not label_norm:
            continue
        if any(
            token in label_norm
            for token in (
                "vadeli mevduat",
                "para piyasasi fon",
                "likit",
                "nakit benzeri",
                "gercege uygun deger farki kar/zarara yansitilan finansal varliklar",
            )
        ):
            explicit_rows.append(row)
            continue
        if label_norm == "finansal yatirimlar":
            generic_rows.append(row)

    if explicit_rows:
        best_explicit = max(explicit_rows, key=lambda item: abs(float(item.get("value", 0.0))))
        return float(best_explicit.get("value", 0.0))

    # Some issuers expose separate current/non-current investment totals with the same
    # label. In that case the larger current bucket is usually the liquid bucket used
    # in market net debt views. A single generic row is too ambiguous, so we leave it out.
    if len(generic_rows) >= 2:
        best_generic = max(generic_rows, key=lambda item: abs(float(item.get("value", 0.0))))
        return float(best_generic.get("value", 0.0))

    return None


def _pick_income_row_value(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str,
    includes: Tuple[str, ...],
    excludes: Tuple[str, ...] = (),
) -> Optional[float]:
    row = _pick_best_row(
        rows=rows,
        period=period,
        includes=includes,
        excludes=excludes,
        body_index=1,
        comparison_mode=comparison_mode,
    )
    if row is None:
        return None
    return float(row.get("value", 0.0))


def _derive_favok(
    *,
    rows: List[Dict[str, Any]],
    period: int,
    comparison_mode: str,
    explicit_favok: Optional[float],
    esas_faaliyet_kari: Optional[float],
    amortisman_itfa_gideri: Optional[float],
) -> Optional[float]:
    if explicit_favok is not None:
        return explicit_favok
    if esas_faaliyet_kari is None or amortisman_itfa_gideri is None:
        return None

    other_income = _pick_income_row_value(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
        includes=("esas faaliyetlerden diger gelir",),
    ) or 0.0
    other_expense = _pick_income_row_value(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
        includes=("esas faaliyetlerden diger gider",),
    ) or 0.0
    equity_share = _pick_income_row_value(
        rows=rows,
        period=period,
        comparison_mode=comparison_mode,
        includes=("ozkaynak yontemiyle degerlenen", "kar"),
        excludes=("diger kapsamli", "dagitilmamis"),
    ) or 0.0

    try:
        return (
            float(esas_faaliyet_kari)
            - float(other_income)
            - float(other_expense)
            - float(equity_share)
            + float(amortisman_itfa_gideri)
        )
    except Exception:
        return None


def _extract_disclosure_metrics(
    detail_payload: Dict[str, Any],
    period: int,
    *,
    prefer_income_statement_ytd: bool = False,
    comparison_mode: str = "current",
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    disclosure_body = detail_payload.get("disclosureBody", [])
    if not isinstance(disclosure_body, list):
        return {}, {"raw": "", "multiplier": 1.0, "currency": "TL"}

    all_rows: List[Dict[str, Any]] = []
    unit_info = {"raw": "", "multiplier": 1.0, "currency": "TL"}

    for body_index, body_item in enumerate(disclosure_body):
        body_html = str(body_item or "")
        if not body_html:
            continue
        body_unit = _parse_unit_info(body_html)
        if body_unit.get("raw"):
            unit_info = body_unit
        all_rows.extend(
            _extract_rows_from_disclosure_body(
                body_html=body_html,
                body_index=body_index,
                unit_multiplier=float(unit_info.get("multiplier", 1.0)),
            )
        )

    metrics: Dict[str, Optional[float]] = {
        "net_kar": _pick_metric_value(
            "net_kar",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "satis_gelirleri": _pick_metric_value(
            "satis_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "brut_kar": _pick_metric_value(
            "brut_kar",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "favok": _pick_metric_value(
            "favok",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "faiz_gelirleri": _pick_metric_value(
            "faiz_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "faiz_giderleri": _pick_metric_value(
            "faiz_giderleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "net_ucret_komisyon_gelirleri": _pick_metric_value(
            "net_ucret_komisyon_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "net_faaliyet_kari": _pick_metric_value(
            "net_faaliyet_kari",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "esas_faaliyet_kari": _pick_metric_value(
            "esas_faaliyet_kari",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "amortisman_itfa_gideri": _pick_metric_value(
            "amortisman_itfa_gideri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "prim_uretimi": _pick_metric_value(
            "prim_uretimi",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "alinan_net_primler": _pick_metric_value(
            "alinan_net_primler",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "teknik_gelirler": _pick_metric_value(
            "teknik_gelirler",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "teknik_denge": _pick_metric_value(
            "teknik_denge",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
            comparison_mode=comparison_mode,
        ),
        "faaliyet_nakit_akisi": _pick_metric_value("faaliyet_nakit_akisi", all_rows, period=period, comparison_mode=comparison_mode),
        "capex": _pick_metric_value("capex", all_rows, period=period, comparison_mode=comparison_mode),
        "nakit_ve_nakit_benzerleri": _pick_metric_value("nakit_ve_nakit_benzerleri", all_rows, period=period, comparison_mode=comparison_mode),
        "finansal_varliklar_sigortacilik": _pick_metric_value("finansal_varliklar_sigortacilik", all_rows, period=period, comparison_mode=comparison_mode),
        "esas_faaliyetlerden_alacaklar": _pick_metric_value("esas_faaliyetlerden_alacaklar", all_rows, period=period, comparison_mode=comparison_mode),
        "teknik_karsiliklar": _pick_metric_value("teknik_karsiliklar", all_rows, period=period, comparison_mode=comparison_mode),
        "esas_faaliyetlerden_borclar": _pick_metric_value("esas_faaliyetlerden_borclar", all_rows, period=period, comparison_mode=comparison_mode),
        "donen_varliklar": _pick_metric_value("donen_varliklar", all_rows, period=period, comparison_mode=comparison_mode),
        "duran_varliklar": _pick_metric_value("duran_varliklar", all_rows, period=period, comparison_mode=comparison_mode),
        "toplam_varliklar": _pick_metric_value("toplam_varliklar", all_rows, period=period, comparison_mode=comparison_mode),
        "kisa_vadeli_yukumlulukler": _pick_metric_value("kisa_vadeli_yukumlulukler", all_rows, period=period, comparison_mode=comparison_mode),
        "finansal_varliklar_net": _pick_metric_value("finansal_varliklar_net", all_rows, period=period, comparison_mode=comparison_mode),
        "krediler": _pick_metric_value("krediler", all_rows, period=period, comparison_mode=comparison_mode),
        "mevduatlar": _pick_metric_value("mevduatlar", all_rows, period=period, comparison_mode=comparison_mode),
        "beklenen_zarar_karsiliklari": _pick_metric_value("beklenen_zarar_karsiliklari", all_rows, period=period, comparison_mode=comparison_mode),
        "finansal_borclar": _pick_metric_value("finansal_borclar", all_rows, period=period, comparison_mode=comparison_mode),
        "net_borc": _pick_metric_value("net_borc", all_rows, period=period, comparison_mode=comparison_mode),
        "ozkaynaklar": _pick_metric_value("ozkaynaklar", all_rows, period=period, comparison_mode=comparison_mode),
        "odenmis_sermaye": _pick_metric_value("odenmis_sermaye", all_rows, period=period, comparison_mode=comparison_mode),
        "cikarilmis_sermaye": _pick_metric_value("cikarilmis_sermaye", all_rows, period=period, comparison_mode=comparison_mode),
    }
    if metrics["finansal_borclar"] is None:
        metrics["finansal_borclar"] = _derive_finansal_borclar(
            all_rows,
            period=period,
            comparison_mode=comparison_mode,
        )
    if metrics["net_borc"] is None:
        metrics["net_borc"] = _derive_net_borc(
            all_rows,
            period=period,
            finansal_borclar=metrics.get("finansal_borclar"),
            comparison_mode=comparison_mode,
        )
    metrics["favok"] = _derive_favok(
        rows=all_rows,
        period=period,
        comparison_mode=comparison_mode,
        explicit_favok=metrics.get("favok"),
        esas_faaliyet_kari=metrics.get("esas_faaliyet_kari"),
        amortisman_itfa_gideri=metrics.get("amortisman_itfa_gideri"),
    )
    if metrics["faaliyet_nakit_akisi"] is not None and metrics["capex"] is not None:
        metrics["serbest_nakit_akisi"] = float(metrics["faaliyet_nakit_akisi"]) + float(metrics["capex"])
    else:
        metrics["serbest_nakit_akisi"] = None
    cash_like = metrics.get("nakit_ve_nakit_benzerleri")
    insurance_assets = metrics.get("finansal_varliklar_sigortacilik")
    if cash_like is not None and insurance_assets is not None:
        metrics["nakit_benzeri_finansal_varliklar"] = float(cash_like) + float(insurance_assets)
    else:
        metrics["nakit_benzeri_finansal_varliklar"] = cash_like if cash_like is not None else insurance_assets
    return metrics, unit_info


def _fetch_attachment_detail(disclosure_index: int, cfg: KapConfig) -> Optional[Dict[str, Any]]:
    url = f"{KAP_BASE_URL}/{ATTACHMENT_DETAIL_ENDPOINT}/{int(disclosure_index)}"
    payload = _http_get_json(url, cfg)
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    return dict(first) if isinstance(first, dict) else None


def _is_insurance_like_payload(payload: Optional[Dict[str, Any]], member: Optional[Dict[str, Any]] = None) -> bool:
    title = " ".join(
        str(value or "")
        for value in (
            (member or {}).get("title"),
            (payload or {}).get("company_title") if isinstance(payload, dict) else "",
            (payload or {}).get("company") if isinstance(payload, dict) else "",
        )
    )
    title_norm = _normalize(title)
    if "sigorta" in title_norm or "emeklilik" in title_norm:
        return True

    quarters = (payload or {}).get("quarters") if isinstance(payload, dict) else []
    if not isinstance(quarters, list):
        return False
    insurance_keys = {
        "prim_uretimi",
        "alinan_net_primler",
        "teknik_gelirler",
        "teknik_denge",
        "teknik_karsiliklar",
    }
    for quarter in quarters:
        if not isinstance(quarter, dict):
            continue
        for bucket in ("metrics", "metrics_quarterly", "metrics_ytd"):
            metrics = quarter.get(bucket)
            if not isinstance(metrics, dict):
                continue
            if any(metrics.get(key) is not None for key in insurance_keys):
                return True
    return False


def _is_insurance_premium_check_fresh(payload: Dict[str, Any]) -> bool:
    if int(payload.get("insurance_premium_disclosures_version") or 0) != KAP_INSURANCE_PREMIUM_CACHE_VERSION:
        return False
    return _is_recent_timestamp(
        payload.get("insurance_premium_disclosures_checked_at"),
        KAP_INSURANCE_PREMIUM_CHECK_TTL_HOURS,
    )


def _premium_period_from_text(text: str) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    raw = str(text or "")
    matches = list(re.finditer(r"(\d{2})\.(\d{2})\.(\d{4})", raw))
    if matches:
        start = matches[0].group(0)
        end = matches[-1].group(0)
        try:
            month = int(matches[-1].group(2))
            year = int(matches[-1].group(3))
        except (TypeError, ValueError):
            return None, None, start, end
        return year, month, start, end

    short_range_matches = list(
        re.finditer(r"(\d{2})\.(\d{2})\s*[-/]\s*(\d{2})\.(\d{2})\.(\d{4})", raw)
    )
    if short_range_matches:
        match = short_range_matches[-1]
        try:
            month = int(match.group(4))
            year = int(match.group(5))
        except (TypeError, ValueError):
            return None, None, None, None
        period_start = f"{match.group(1)}.{match.group(2)}.{year}"
        period_end = f"{match.group(3)}.{match.group(4)}.{year}"
        return year, month, period_start, period_end

    month_year_matches = list(re.finditer(r"\b(\d{2})\.(\d{4})\b", raw))
    if not month_year_matches:
        norm = _normalize(raw)
        month_names = "|".join(TURKISH_MONTH_NAME_TO_NUMBER.keys())
        month_name_match = re.search(
            rf"\b(?P<year>20\d{{2}})\s+(?P<month>{month_names})\b",
            norm,
        ) or re.search(
            rf"\b(?P<month>{month_names})\s+(?P<year>20\d{{2}})\b",
            norm,
        )
        if not month_name_match:
            year_match = re.search(r"\b(?P<year>20\d{2})\b(?P<tail>.{0,120})", norm)
            tail = year_match.group("tail") if year_match else ""
            month_matches = list(re.finditer(rf"\b({month_names})\b", tail))
            if not year_match or not month_matches:
                return None, None, None, None
            try:
                year = int(year_match.group("year"))
                month = TURKISH_MONTH_NAME_TO_NUMBER[month_matches[-1].group(1)]
            except (KeyError, TypeError, ValueError):
                return None, None, None, None
        else:
            try:
                month = TURKISH_MONTH_NAME_TO_NUMBER[month_name_match.group("month")]
                year = int(month_name_match.group("year"))
            except (KeyError, TypeError, ValueError):
                return None, None, None, None
    else:
        match = month_year_matches[-1]
        try:
            month = int(match.group(1))
            year = int(match.group(2))
        except (TypeError, ValueError):
            return None, None, None, None
    if month < 1 or month > 12:
        return None, None, None, None
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    period_start = date(year, 1, 1).strftime("%d.%m.%Y")
    period_end = (next_month - timedelta(days=1)).strftime("%d.%m.%Y")
    return year, month, period_start, period_end


def _parse_premium_number_token(raw: Any) -> Optional[float]:
    token = str(raw or "").strip()
    if not token:
        return None
    negative = token.startswith("-")
    token = token.lstrip("+-")
    token = re.sub(r"[^0-9.,]", "", token)
    if not token:
        return None
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "." in token:
        token = token.replace(".", "")
    elif "," in token:
        token = token.replace(",", ".")
    try:
        value = float(token)
    except ValueError:
        return None
    return -abs(value) if negative else value


def _parse_premium_yoy_token(raw: Any) -> Optional[float]:
    text = str(raw or "").strip().lower()
    if not text or "a.d" in text or "n.m" in text:
        return None
    value = _parse_premium_number_token(text.replace("%", ""))
    return value


def _premium_pdf_unit_multiplier(text: str) -> float:
    norm = _normalize(text)
    if "bin tl" in norm or "thousand tl" in norm:
        return 1000.0
    if "milyon tl" in norm or "million tl" in norm:
        return 1_000_000.0
    return 1.0


def _parse_insurance_premium_pdf_text(text: str) -> Optional[Dict[str, Optional[float]]]:
    compact = " ".join(str(text or "").split())
    if not compact:
        return None

    patterns = (
        re.compile(
            r"(?P<monthly_prev>-?[\d.]+)\s+"
            r"(?P<monthly_current>-?[\d.]+)\s+"
            r"(?P<monthly_yoy>-?\d+%|a\.d\.?|n\.m\.?)\s+"
            r"GENEL\s+TOPLAM\s+"
            r"(?P<ytd_prev>-?[\d.]+)\s+"
            r"(?P<ytd_current>-?[\d.]+)\s+"
            r"(?P<ytd_yoy>-?\d+%|a\.d\.?|n\.m\.?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<monthly_prev>-?[\d.]+)\s+"
            r"(?P<monthly_current>-?[\d.]+)\s+"
            r"(?P<monthly_yoy>-?\d+%|a\.d\.?|n\.m\.?)\s+"
            r"TOTAL\s+"
            r"(?P<ytd_prev>-?[\d.]+)\s+"
            r"(?P<ytd_current>-?[\d.]+)\s+"
            r"(?P<ytd_yoy>-?\d+%|a\.d\.?|n\.m\.?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:GENEL\s+)?TOPLAM\s+"
            r"\d{2}\.\d{2}\.\d{4}\s+"
            r"(?P<ytd_prev>-?[\d.]+)\s+"
            r"\d{2}\.\d{2}\.\d{4}\s+"
            r"(?P<ytd_current>-?[\d.]+)\s+"
            r"(?:DEĞİŞİM\s*%?|DEGISIM\s*%?)?\s*"
            r"(?P<ytd_yoy>-?%?\d+(?:[,.]\d+)?%?|a\.d\.?|n\.m\.?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:GENEL\s+)?TOPLAM\s+"
            r"(?P<ytd_current>-?[\d.]+)\s+"
            r"(?P<ytd_prev>-?[\d.]+)\s+"
            r"(?P<ytd_yoy>-?%?\d+(?:[,.]\d+)?%?|a\.d\.?|n\.m\.?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"TOTAL\s+"
            r"(?P<ytd_current>-?[\d.]+)\s+"
            r"(?P<ytd_prev>-?[\d.]+)\s+"
            r"(?P<ytd_yoy>-?%?\d+(?:[,.]\d+)?%?|a\.d\.?|n\.m\.?)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"prim\s+üretimi\s+"
            r"(?P<ytd_current>[\d.]+)\s*(?:TL|TRY)?\b.*?"
            r"(?P<ytd_yoy>%\s*-?\d+(?:[,.]\d+)?|-?%?\d+(?:[,.]\d+)?\s*%)\s+"
            r"oranında.*?"
            r"geçen\s+yıl\s+aynı\s+dönem\s+"
            r"(?P<ytd_prev>[\d.]+)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"prim\s+üretimi\s+"
            r"(?P<ytd_current>[\d.,]+)\s*(?:TL|TRY)?\b.*?"
            r"(?:göre|compared).*?"
            r"(?P<ytd_yoy>%\s*-?\d+(?:[,.]\d+)?|-?%?\d+(?:[,.]\d+)?\s*%)\s+"
            r"(?:oranında\s+)?art",
            flags=re.IGNORECASE,
        ),
    )
    match: Optional[re.Match[str]] = None
    for pattern in patterns:
        matches = list(pattern.finditer(compact))
        if matches:
            match = matches[-1]
            break
    if match is None:
        return None

    multiplier = _premium_pdf_unit_multiplier(compact)
    monthly = _parse_premium_number_token(match.groupdict().get("monthly_current"))
    ytd = _parse_premium_number_token(match.group("ytd_current"))
    previous_monthly = _parse_premium_number_token(match.groupdict().get("monthly_prev"))
    previous_ytd = _parse_premium_number_token(match.groupdict().get("ytd_prev"))
    monthly_yoy = _parse_premium_yoy_token(match.groupdict().get("monthly_yoy"))
    ytd_yoy = _parse_premium_yoy_token(match.group("ytd_yoy"))
    if previous_ytd is None and ytd is not None and ytd_yoy is not None and ytd_yoy > -100:
        previous_ytd = ytd / (1 + (ytd_yoy / 100.0))
    if previous_monthly is None and monthly is not None and monthly_yoy is not None and monthly_yoy > -100:
        previous_monthly = monthly / (1 + (monthly_yoy / 100.0))
    return {
        "monthly_gross_premium": monthly * multiplier if monthly is not None else None,
        "ytd_gross_premium": ytd * multiplier if ytd is not None else None,
        "previous_year_monthly_gross_premium": previous_monthly * multiplier if previous_monthly is not None else None,
        "previous_year_ytd_gross_premium": previous_ytd * multiplier if previous_ytd is not None else None,
        "monthly_yoy_pct": monthly_yoy,
        "ytd_yoy_pct": ytd_yoy,
    }


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    pdf_start = data.find(b"%PDF")
    if pdf_start > 0:
        data = data[pdf_start:]
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        return ""
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(part for part in parts if part)


def _premium_attachment_text(detail: Dict[str, Any], cfg: KapConfig) -> str:
    attachments = detail.get("attachments") or []
    if not isinstance(attachments, list):
        return ""
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        obj_id = str(attachment.get("objId") or "").strip()
        extension = str(attachment.get("fileExtension") or "").strip().lower()
        if not obj_id or extension != "pdf":
            continue
        try:
            data = _http_get_bytes(
                f"{KAP_BASE_URL}/{FILE_DOWNLOAD_ENDPOINT}/{urllib.parse.quote(obj_id)}",
                cfg,
                accept="application/pdf,*/*",
            )
        except Exception:
            continue
        text = _extract_pdf_text(data)
        if text:
            return text
    return ""


def _premium_disclosure_text(detail: Dict[str, Any]) -> str:
    parts: List[str] = []
    basic = (
        detail.get("disclosure", {}).get("disclosureBasic", {})
        if isinstance(detail.get("disclosure"), dict)
        else {}
    )
    if isinstance(basic, dict):
        parts.extend(str(basic.get(key) or "") for key in ("summary", "title"))
    body = detail.get("disclosureBody")
    if isinstance(body, list):
        parts.extend(str(item or "") for item in body)
    elif body:
        parts.append(str(body))
    text = " ".join(parts)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(" ".join(text.split()))


def _insurance_premium_disclosure_body(
    *,
    member_oid: str,
    cfg: KapConfig,
    from_date: str,
    to_date: str,
) -> Dict[str, Any]:
    return {
        "fromDate": from_date,
        "toDate": to_date,
        "memberType": "IGS",
        "mkkMemberOidList": [member_oid],
        "inactiveMkkMemberOidList": [],
        "disclosureClass": "",
        "subjectList": [],
        "isLate": "",
        "mainSector": "",
        "sector": "",
        "subSector": "",
        "marketOid": "",
        "index": "",
        "bdkReview": "",
        "bdkMemberOidList": [],
        "year": "",
        "term": "",
        "ruleType": "",
        "period": "",
        "fromSrc": False,
        "srcCategory": "",
        "disclosureIndexList": [],
    }


def _list_member_disclosures_by_criteria(
    *,
    member_oid: str,
    cfg: KapConfig,
    lookback_days: int = 365,
) -> List[Dict[str, Any]]:
    end_dt = _utc_now().date()
    remaining_days = max(30, min(int(lookback_days or 365), 2555))
    cursor_end = end_dt
    rows: List[Dict[str, Any]] = []
    seen_indices: set[str] = set()

    while remaining_days > 0:
        window_days = min(365, remaining_days)
        start_dt = cursor_end - timedelta(days=window_days)
        payload = _http_post_json(
            f"{KAP_BASE_URL}/{DISCLOSURE_MEMBERS_BY_CRITERIA_ENDPOINT}",
            _insurance_premium_disclosure_body(
                member_oid=member_oid,
                cfg=cfg,
                from_date=start_dt.isoformat(),
                to_date=cursor_end.isoformat(),
            ),
            cfg,
        )
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                disclosure_index = str(row.get("disclosureIndex") or "").strip()
                if disclosure_index and disclosure_index in seen_indices:
                    continue
                if disclosure_index:
                    seen_indices.add(disclosure_index)
                rows.append(dict(row))

        remaining_days -= window_days
        cursor_end = start_dt - timedelta(days=1)

    return rows


def _is_premium_production_disclosure(row: Dict[str, Any]) -> bool:
    text = " ".join(str(row.get(key) or "") for key in ("summary", "subject", "title"))
    norm = _normalize(text)
    if not norm:
        return False
    if "prim" not in norm and "premium" not in norm:
        return False
    return (
        "brut yazilan prim" in norm
        or "prim uretim" in norm
        or "gross written premium" in norm
        or "premium production" in norm
    )


def _pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return round(((float(current) - float(previous)) / abs(float(previous))) * 100.0, 1)


def _extract_js_assignment_json(text: str, variable_name: str) -> Any:
    source = str(text or "")
    match = re.search(rf"\bvar\s+{re.escape(variable_name)}\s*=\s*", source)
    if not match:
        return None
    idx = match.end()
    while idx < len(source) and source[idx].isspace():
        idx += 1
    if idx >= len(source) or source[idx] not in "[{":
        return None

    opener = source[idx]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    escaped = False
    for pos in range(idx, len(source)):
        char = source[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return json.loads(source[idx : pos + 1])
    return None


def _marketvisuals_period_dates(year: int, month: int) -> Tuple[str, str]:
    start = date(int(year), int(month), 1)
    if int(month) == 12:
        next_month = date(int(year) + 1, 1, 1)
    else:
        next_month = date(int(year), int(month) + 1, 1)
    end = next_month - timedelta(days=1)
    return start.strftime("%d.%m.%Y"), end.strftime("%d.%m.%Y")


def _marketvisuals_company_aliases(company_key: str, company_title: str = "") -> List[str]:
    company_norm = str(company_key or "").strip().upper()
    aliases = list(MARKETVISUALS_INSURANCE_COMPANY_ALIASES.get(company_norm, []))
    title_norm = _normalize(company_title)
    if title_norm:
        aliases.append(title_norm)
    return [alias for alias in aliases if alias]


def _marketvisuals_matches_company(section_title: str, *, company_key: str, company_title: str = "") -> bool:
    title_norm = _normalize(section_title)
    if "aylik prim" not in title_norm:
        return False
    aliases = _marketvisuals_company_aliases(company_key, company_title)
    return any(_normalize(alias) in title_norm for alias in aliases)


def _marketvisuals_rows_from_chart(
    chart: Dict[str, Any],
    *,
    source_url: str,
) -> List[Dict[str, Any]]:
    data = chart.get("data") if isinstance(chart, dict) else None
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, list):
        return []

    by_year: Dict[int, List[float]] = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        try:
            year = int(str(dataset.get("label") or "").strip())
        except (TypeError, ValueError):
            continue
        values: List[float] = []
        for raw_value in dataset.get("data") or []:
            try:
                values.append(float(raw_value or 0.0))
            except (TypeError, ValueError):
                values.append(0.0)
        if values:
            by_year[year] = values

    rows: List[Dict[str, Any]] = []
    ytd_by_year: Dict[Tuple[int, int], float] = {}
    for year in sorted(by_year):
        cumulative = 0.0
        for idx, value_mn in enumerate(by_year[year][:12]):
            month = idx + 1
            if value_mn <= 0:
                continue
            cumulative += value_mn
            ytd_by_year[(year, month)] = cumulative
            previous_monthly_mn: Optional[float] = None
            previous_values = by_year.get(year - 1)
            if previous_values and idx < len(previous_values) and float(previous_values[idx] or 0.0) > 0:
                previous_monthly_mn = float(previous_values[idx])
            previous_ytd_mn = ytd_by_year.get((year - 1, month))
            period_start, period_end = _marketvisuals_period_dates(year, month)
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "period_label": f"{year}/{month}",
                    "period_start": period_start,
                    "period_end": period_end,
                    "published_at": "",
                    "disclosure_index": None,
                    "summary": MARKETVISUALS_TSB_SOURCE_LABEL,
                    "source_url": source_url,
                    "monthly_gross_premium": value_mn * 1_000_000.0,
                    "ytd_gross_premium": cumulative * 1_000_000.0,
                    "previous_year_monthly_gross_premium": (
                        previous_monthly_mn * 1_000_000.0 if previous_monthly_mn is not None else None
                    ),
                    "previous_year_ytd_gross_premium": (
                        previous_ytd_mn * 1_000_000.0 if previous_ytd_mn is not None else None
                    ),
                    "monthly_yoy_pct": _pct_change(value_mn, previous_monthly_mn),
                    "ytd_yoy_pct": _pct_change(cumulative, previous_ytd_mn),
                }
            )
    return rows


def _parse_marketvisuals_insurance_premium_page(
    html_text: str,
    *,
    company_key: str,
    company_title: str = "",
    source_url: str,
) -> List[Dict[str, Any]]:
    sections = _extract_js_assignment_json(html_text, "SECTIONS")
    charts = _extract_js_assignment_json(html_text, "CHARTS")
    if not isinstance(sections, list) or not isinstance(charts, list):
        return []

    chart_by_id = {
        str(chart.get("id") or ""): chart
        for chart in charts
        if isinstance(chart, dict) and chart.get("id") is not None
    }
    for section in sections:
        if not isinstance(section, dict):
            continue
        if not _marketvisuals_matches_company(
            str(section.get("title") or ""),
            company_key=company_key,
            company_title=company_title,
        ):
            continue
        chart_id = str(section.get("chartId") or "")
        chart = chart_by_id.get(chart_id)
        if not chart:
            continue
        return _marketvisuals_rows_from_chart(chart, source_url=source_url)
    return []


def _fetch_tsb_marketvisuals_premium_disclosures(
    *,
    company_key: str,
    cfg: KapConfig,
    company_title: str = "",
) -> List[Dict[str, Any]]:
    normalized_key = str(company_key or "").strip().upper()
    if normalized_key not in MARKETVISUALS_INSURANCE_COMPANY_ALIASES and not company_title:
        return []

    urls = MARKETVISUALS_INSURANCE_COMPANY_PAGES.get(
        normalized_key,
        (MARKETVISUALS_INSURANCE_HAYAT_URL, MARKETVISUALS_INSURANCE_HAYATDISI_URL),
    )
    rows: List[Dict[str, Any]] = []
    for url in urls:
        try:
            html_text = _http_get_text(url, cfg)
        except Exception:
            continue
        parsed = _parse_marketvisuals_insurance_premium_page(
            html_text,
            company_key=normalized_key,
            company_title=company_title,
            source_url=url,
        )
        if parsed:
            rows.extend(parsed)
    return _derive_missing_monthly_premiums(rows)


def _has_insurance_premium_rows(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            year = int(row.get("year") or 0)
            month = int(row.get("month") or 0)
        except (TypeError, ValueError):
            continue
        if year > 0 and 1 <= month <= 12 and (
            row.get("monthly_gross_premium") is not None or row.get("ytd_gross_premium") is not None
        ):
            return True
    return False


def _insurance_premium_company_key(payload: Dict[str, Any], member: Optional[Dict[str, Any]] = None) -> str:
    candidates = [
        str(payload.get("stock_code") or "").strip().upper(),
        str(payload.get("company") or "").strip().upper(),
        str((member or {}).get("company_code") or "").strip().upper(),
    ]
    for key in candidates:
        if key in MARKETVISUALS_INSURANCE_COMPANY_ALIASES:
            return key
    for key in candidates:
        if key:
            return key
    return ""


def _insurance_premium_company_title(payload: Dict[str, Any], member: Optional[Dict[str, Any]] = None) -> str:
    return str((member or {}).get("title") or payload.get("company_title") or "").strip()


def _derive_missing_monthly_premiums(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: (int(item.get("year") or 0), int(item.get("month") or 0)))
    by_period: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for row in ordered:
        try:
            year = int(row.get("year") or 0)
            month = int(row.get("month") or 0)
        except (TypeError, ValueError):
            continue
        if year <= 0 or month < 1 or month > 12:
            continue

        previous_month_row = by_period.get((year, month - 1))
        if row.get("monthly_gross_premium") is None and row.get("ytd_gross_premium") is not None:
            previous_ytd = (
                0.0
                if month == 1
                else previous_month_row.get("ytd_gross_premium") if previous_month_row else None
            )
            if previous_ytd is not None:
                row["monthly_gross_premium"] = float(row["ytd_gross_premium"]) - float(previous_ytd)

        if (
            row.get("previous_year_monthly_gross_premium") is None
            and row.get("previous_year_ytd_gross_premium") is not None
        ):
            previous_comparable_ytd = (
                0.0
                if month == 1
                else previous_month_row.get("previous_year_ytd_gross_premium") if previous_month_row else None
            )
            if previous_comparable_ytd is not None:
                row["previous_year_monthly_gross_premium"] = (
                    float(row["previous_year_ytd_gross_premium"]) - float(previous_comparable_ytd)
                )

        if row.get("monthly_yoy_pct") is None:
            row["monthly_yoy_pct"] = _pct_change(
                row.get("monthly_gross_premium"),
                row.get("previous_year_monthly_gross_premium"),
            )
        if row.get("ytd_yoy_pct") is None:
            row["ytd_yoy_pct"] = _pct_change(
                row.get("ytd_gross_premium"),
                row.get("previous_year_ytd_gross_premium"),
            )
        by_period[(year, month)] = row

    return ordered


def _fetch_insurance_premium_disclosures(
    *,
    member_oid: str,
    cfg: KapConfig,
    max_items: int = KAP_INSURANCE_PREMIUM_MAX_DISCLOSURES,
) -> List[Dict[str, Any]]:
    if not member_oid:
        return []
    rows = _list_member_disclosures_by_criteria(
        member_oid=member_oid,
        cfg=cfg,
        lookback_days=KAP_INSURANCE_PREMIUM_DISCLOSURE_LOOKBACK_DAYS,
    )
    premium_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not _is_premium_production_disclosure(row):
            continue
        try:
            disclosure_index = int(row.get("disclosureIndex") or 0)
        except (TypeError, ValueError):
            continue
        if disclosure_index <= 0:
            continue
        summary = str(row.get("summary") or row.get("subject") or "").strip()
        year, month, period_start, period_end = _premium_period_from_text(summary)
        premium_rows.append(
            {
                "row": row,
                "year": year,
                "month": month,
                "period_start": period_start,
                "period_end": period_end,
                "summary": summary,
                "disclosure_index": disclosure_index,
            }
        )
    latest_premium_year = max((int(item["year"]) for item in premium_rows if item["year"] is not None), default=0)
    premium_rows.sort(
        key=lambda item: (
            1 if item["year"] is None or item["month"] is None else 0,
            0 if latest_premium_year and int(item["year"] or 0) >= latest_premium_year - 1 else 1,
            int(item["year"] or 0) if latest_premium_year and int(item["year"] or 0) >= latest_premium_year - 1 else -int(item["year"] or 0),
            int(item["month"] or 0) if latest_premium_year and int(item["year"] or 0) >= latest_premium_year - 1 else -int(item["month"] or 0),
            int(item["disclosure_index"]),
        ),
    )

    results: List[Dict[str, Any]] = []
    seen_periods: set[Tuple[int, int]] = set()
    for item in premium_rows:
        if len(results) >= max(1, int(max_items)):
            break
        row = item["row"]
        disclosure_index = int(item["disclosure_index"])
        year = int(item["year"] or 0)
        month = int(item["month"] or 0)
        period_start = item["period_start"]
        period_end = item["period_end"]
        summary = str(item["summary"])
        try:
            detail = _fetch_attachment_detail(disclosure_index, cfg)
        except Exception:
            continue
        if not detail:
            continue
        disclosure_text = _premium_disclosure_text(detail)
        pdf_text = _premium_attachment_text(detail, cfg)
        combined_text = " ".join(part for part in (summary, disclosure_text, pdf_text) if part)
        if year <= 0 or month < 1 or month > 12:
            year, month, period_start, period_end = _premium_period_from_text(combined_text)
            year = int(year or 0)
            month = int(month or 0)
        if year <= 0 or month < 1 or month > 12:
            continue
        period_key = (year, month)
        if period_key in seen_periods:
            continue
        parsed = _parse_insurance_premium_pdf_text(combined_text)
        if not parsed:
            continue
        results.append(
            {
                "year": year,
                "month": month,
                "period_label": f"{year}/{month}",
                "period_start": period_start,
                "period_end": period_end,
                "published_at": str(row.get("publishDate") or "").strip(),
                "disclosure_index": disclosure_index,
                "summary": summary,
                "source_url": f"https://www.kap.org.tr/tr/Bildirim/{disclosure_index}",
                **parsed,
            }
        )
        seen_periods.add(period_key)

    return _derive_missing_monthly_premiums(results)


def _merge_insurance_premium_disclosures(
    existing_rows: Any,
    fetched_rows: Any,
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for collection in (existing_rows, fetched_rows):
        if not isinstance(collection, list):
            continue
        for row in collection:
            if not isinstance(row, dict):
                continue
            try:
                year = int(row.get("year") or 0)
                month = int(row.get("month") or 0)
            except (TypeError, ValueError):
                continue
            if year <= 0 or month < 1 or month > 12:
                continue
            merged[(year, month)] = dict(row)

    return [
        merged[key]
        for key in sorted(merged)
    ][-KAP_INSURANCE_PREMIUM_MAX_DISCLOSURES:]


def _ensure_insurance_premium_disclosures(
    *,
    payload: Dict[str, Any],
    cache_path: Path,
    cfg: KapConfig,
    member: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    if not _is_insurance_like_payload(payload, member):
        return payload
    if not force_refresh and _is_insurance_premium_check_fresh(payload):
        return payload

    member_oid = str(
        (member or {}).get("mkk_member_oid")
        or payload.get("member_oid")
        or ""
    ).strip()
    company_key = _insurance_premium_company_key(payload, member)
    company_title = _insurance_premium_company_title(payload, member)

    disclosures: List[Dict[str, Any]] = []
    fallback_disclosures: List[Dict[str, Any]] = []
    kap_error: Optional[BaseException] = None
    try:
        if member_oid:
            disclosures = _fetch_insurance_premium_disclosures(member_oid=member_oid, cfg=cfg)
    except Exception as exc:
        kap_error = exc

    should_fetch_marketvisuals = (
        company_key in MARKETVISUALS_INSURANCE_COMPANY_ALIASES
        or (not disclosures and not _has_insurance_premium_rows(payload.get("insurance_premium_disclosures")))
    )
    if should_fetch_marketvisuals:
        try:
            fallback_disclosures = _fetch_tsb_marketvisuals_premium_disclosures(
                company_key=company_key,
                company_title=company_title,
                cfg=cfg,
            )
        except Exception as exc:
            if kap_error is None:
                kap_error = exc

    if disclosures or fallback_disclosures or kap_error is None:
        if fallback_disclosures and company_key in MARKETVISUALS_INSURANCE_COMPANY_ALIASES:
            payload["insurance_premium_disclosures"] = _merge_insurance_premium_disclosures([], fallback_disclosures)
        else:
            merged = _merge_insurance_premium_disclosures(payload.get("insurance_premium_disclosures"), disclosures)
            payload["insurance_premium_disclosures"] = _merge_insurance_premium_disclosures(merged, fallback_disclosures)
        payload["insurance_premium_disclosures_checked_at"] = _utc_now().isoformat()
        payload["insurance_premium_disclosures_version"] = KAP_INSURANCE_PREMIUM_CACHE_VERSION
        payload.pop("insurance_premium_disclosures_error", None)
    else:
        payload["insurance_premium_disclosures_checked_at"] = _utc_now().isoformat()
        payload["insurance_premium_disclosures_error"] = str(kap_error)

    if payload.get("ok"):
        try:
            _write_cache(cache_path, payload)
        except Exception:
            pass
    return payload


def _fetch_premium_only_snapshot(
    *,
    company_norm: str,
    cache_path: Path,
    cfg: KapConfig,
    member: Optional[Dict[str, Any]],
    error: Optional[BaseException] = None,
) -> Optional[Dict[str, Any]]:
    member_oid = str((member or {}).get("mkk_member_oid") or "").strip()
    title = str(
        (member or {}).get("title")
        or MARKETVISUALS_INSURANCE_COMPANY_TITLES.get(company_norm)
        or ""
    ).strip()
    payload: Dict[str, Any] = {
        "ok": True,
        "cache_hit": False,
        "cache_stale": True,
        "schema_version": KAP_CACHE_SCHEMA_VERSION,
        "company": company_norm,
        "company_title": title,
        "stock_code": str((member or {}).get("company_code") or company_norm).strip().upper(),
        "member_oid": member_oid,
        "source_url": (
            f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{(member or {}).get('permalink', '')}"
            if member_oid
            else MARKETVISUALS_INSURANCE_COMPANY_PAGES.get(company_norm, ("",))[0]
        ),
        "fetched_at": _utc_now().isoformat(),
        "quarters": [],
        "financials_error": str(error) if error else None,
    }
    if not _is_insurance_like_payload(payload, member) and company_norm not in MARKETVISUALS_INSURANCE_COMPANY_ALIASES:
        return None
    disclosures: List[Dict[str, Any]] = []
    kap_error: Optional[BaseException] = None
    try:
        if member_oid:
            disclosures = _fetch_insurance_premium_disclosures(member_oid=member_oid, cfg=cfg)
    except Exception as exc:
        kap_error = exc
    fallback_disclosures: List[Dict[str, Any]] = []
    if company_norm in MARKETVISUALS_INSURANCE_COMPANY_ALIASES or not disclosures:
        try:
            fallback_disclosures = _fetch_tsb_marketvisuals_premium_disclosures(
                company_key=company_norm,
                company_title=title,
                cfg=cfg,
            )
        except Exception as exc:
            if kap_error is None:
                kap_error = exc
    if fallback_disclosures and company_norm in MARKETVISUALS_INSURANCE_COMPANY_ALIASES:
        disclosures = _merge_insurance_premium_disclosures([], fallback_disclosures)
    else:
        disclosures = _merge_insurance_premium_disclosures(disclosures, fallback_disclosures)
    if not disclosures:
        if kap_error is not None:
            payload["insurance_premium_disclosures_error"] = str(kap_error)
        return None
    payload["insurance_premium_disclosures"] = _merge_insurance_premium_disclosures([], disclosures)
    payload["insurance_premium_disclosures_checked_at"] = _utc_now().isoformat()
    payload["insurance_premium_disclosures_version"] = KAP_INSURANCE_PREMIUM_CACHE_VERSION
    try:
        _write_cache(cache_path, payload)
    except Exception:
        pass
    return payload


def _quarter_label(year: int, period: int) -> str:
    return f"{int(year)}Q{int(period)}"


def fetch_kap_company_snapshot(
    *,
    company: str,
    cfg: KapConfig,
    processed_dir: Path,
    force_refresh: bool = False,
    max_quarters: int = 4,
    use_cache_when_complete: bool = False,
) -> Dict[str, Any]:
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
    cache_has_requested_depth = cached_period_count >= requested_max_quarters
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
            )

        if use_cache_when_complete:
            cached_member_oid = str(cached.get("member_oid") or "").strip()
            if cached_member_oid:
                try:
                    latest_disclosures = _list_company_disclosures(
                        member_oid=cached_member_oid,
                        cfg=cfg,
                        max_periods=1,
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
                        )
                    if not live_latest_key:
                        raise RuntimeError("KAP finansal bildirim listesi bos dondu.")
                except Exception as exc:
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
        member = _resolve_member(company_norm, cfg)
        if not member:
            raise RuntimeError(f"KAP uyelik kaydi bulunamadi: {company_norm}")

        disclosures = _list_company_disclosures(
            member_oid=member["mkk_member_oid"],
            cfg=cfg,
            max_periods=requested_max_quarters,
        )
        if not disclosures:
            raise RuntimeError("KAP finansal bildirimleri şu anda alınamadı.")

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
            detail = _fetch_attachment_detail(disclosure_index, cfg)
            if not detail:
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
            )
            metrics_ytd, _ = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=True,
                comparison_mode="current",
            )
            metrics_comparative, _ = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=False,
                comparison_mode="comparative",
            )
            metrics_ytd_comparative, _ = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=True,
                comparison_mode="comparative",
            )
            quarter_rows.append(
                {
                    "year": year,
                    "period": period,
                    "quarter": _quarter_label(year, period),
                    "disclosure_index": disclosure_index,
                    "publish_date": str(basic.get("publishDate", "")).strip(),
                    "title": str(basic.get("title", "") or item.get("title", "")).strip(),
                    "stock_code": str(basic.get("stockCode", "") or item.get("stock_code", "")).strip().upper(),
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

        quarter_rows = _merge_live_and_cached_quarters(quarter_rows, cached, requested_max_quarters)

        payload: Dict[str, Any] = {
            "ok": True,
            "cache_hit": False,
            "cache_stale": False,
            "schema_version": KAP_CACHE_SCHEMA_VERSION,
            "company": str(quarter_rows[0].get("stock_code", "") or company_norm).strip().upper(),
            "company_title": str(member.get("title", "")).strip(),
            "stock_code": str(quarter_rows[0].get("stock_code", "")).strip(),
            "member_oid": str(member.get("mkk_member_oid", "")).strip(),
            "source_url": f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{member.get('permalink', '')}",
            "fetched_at": _utc_now().isoformat(),
            "quarters": quarter_rows,
        }
        payload = _ensure_insurance_premium_disclosures(
            payload=payload,
            cache_path=cache_path,
            cfg=cfg,
            member=member,
            force_refresh=True,
        )
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        if cached:
            cached["ok"] = bool(cached.get("ok", False))
            cached["cache_hit"] = True
            cached["cache_stale"] = True
            cached["error"] = str(exc)
            if cached["ok"] and _cached_period_count(cached) > 0:
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
