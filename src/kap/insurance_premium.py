"""Insurance premium disclosure parsing, PDF extraction, and TSB MarketVisuals fallback."""

from __future__ import annotations

import html
import io
import json
import re
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.kap.exceptions import KapPacingDeadlineExceeded
from src.kap.html_parser import _clean_html_text, _normalize

KAP_BASE_URL = "https://www.kap.org.tr/tr/api"
DISCLOSURE_MEMBERS_BY_CRITERIA_ENDPOINT = "disclosure/members/byCriteria"
FILE_DOWNLOAD_ENDPOINT = "file/download"
KAP_CACHE_SCHEMA_VERSION = 15
KAP_INSURANCE_PREMIUM_CHECK_TTL_HOURS = 24.0
KAP_INSURANCE_PREMIUM_CACHE_VERSION = 8
KAP_INSURANCE_PREMIUM_DISCLOSURE_LOOKBACK_DAYS = 1095
KAP_INSURANCE_PREMIUM_MAX_DISCLOSURES = 36

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


def _kf() -> Any:
    """Resolve src.kap_fetcher lazily so monkeypatching kap_fetcher attributes works."""
    import src.kap_fetcher as kf

    return kf


def _is_insurance_like_payload(payload: Optional[Dict[str, Any]], member: Optional[Dict[str, Any]] = None) -> bool:
    payload_dict = payload if isinstance(payload, dict) else {}
    company = (
        str((member or {}).get("company_code") or "").strip()
        or str(payload_dict.get("stock_code") or payload_dict.get("company") or "").strip()
    )
    title = str(
        (member or {}).get("title")
        or payload_dict.get("company_title")
        or ""
    ).strip()
    return _kf().classify_kap_company_kind(company, title) == "insurance"


def _is_insurance_premium_check_fresh(payload: Dict[str, Any]) -> bool:
    if int(payload.get("insurance_premium_disclosures_version") or 0) != KAP_INSURANCE_PREMIUM_CACHE_VERSION:
        return False
    return _kf()._is_recent_timestamp(
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
            data = _kf()._http_get_bytes(
                f"{KAP_BASE_URL}/{FILE_DOWNLOAD_ENDPOINT}/{urllib.parse.quote(obj_id)}",
                cfg,
                accept="application/pdf,*/*",
            )
        except KapPacingDeadlineExceeded:
            raise
        except urllib.error.HTTPError as exc:
            if int(getattr(exc, "code", 0) or 0) in {403, 429, 503}:
                raise
            continue
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
    end_dt = _kf()._utc_now().date()
    remaining_days = max(30, min(int(lookback_days or 365), 2555))
    cursor_end = end_dt
    rows: List[Dict[str, Any]] = []
    seen_indices: set[str] = set()

    while remaining_days > 0:
        window_days = min(365, remaining_days)
        start_dt = cursor_end - timedelta(days=window_days)
        payload = _kf()._http_post_json(
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
            html_text = _kf()._http_get_text(url, cfg)
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
    strict: bool = False,
) -> List[Dict[str, Any]]:
    if not member_oid:
        return []
    rows = _kf()._list_member_disclosures_by_criteria(
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
            detail = _kf()._fetch_attachment_detail(disclosure_index, cfg)
        except Exception:
            if strict:
                raise
            continue
        if not detail:
            if strict:
                raise RuntimeError(
                    f"KAP prim bildirim detayi bos dondu: disclosure_index={disclosure_index}"
                )
            continue
        disclosure_text = _kf()._premium_disclosure_text(detail)
        pdf_text = _kf()._premium_attachment_text(detail, cfg)
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
    strict: bool = False,
    persist: bool = True,
) -> Dict[str, Any]:
    if not _kf()._is_insurance_like_payload(payload, member):
        return payload
    if not force_refresh and _kf()._is_insurance_premium_check_fresh(payload):
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
    prefers_marketvisuals = company_key in MARKETVISUALS_INSURANCE_COMPANY_ALIASES

    if prefers_marketvisuals:
        try:
            fallback_disclosures = _kf()._fetch_tsb_marketvisuals_premium_disclosures(
                company_key=company_key,
                company_title=company_title,
                cfg=cfg,
            )
        except KapPacingDeadlineExceeded:
            raise
        except Exception as exc:
            kap_error = exc

    if not (prefers_marketvisuals and fallback_disclosures):
        try:
            if member_oid:
                disclosures = _kf()._fetch_insurance_premium_disclosures(
                    member_oid=member_oid,
                    cfg=cfg,
                    strict=strict,
                )
        except KapPacingDeadlineExceeded:
            raise
        except Exception as exc:
            if strict:
                raise
            kap_error = exc

    if not prefers_marketvisuals and not disclosures and not _kf()._has_insurance_premium_rows(
        payload.get("insurance_premium_disclosures")
    ):
        try:
            fallback_disclosures = _kf()._fetch_tsb_marketvisuals_premium_disclosures(
                company_key=company_key,
                company_title=company_title,
                cfg=cfg,
            )
        except KapPacingDeadlineExceeded:
            raise
        except Exception as exc:
            if kap_error is None:
                kap_error = exc

    if disclosures or fallback_disclosures or kap_error is None:
        if fallback_disclosures and company_key in MARKETVISUALS_INSURANCE_COMPANY_ALIASES:
            payload["insurance_premium_disclosures"] = _kf()._merge_insurance_premium_disclosures([], fallback_disclosures)
        else:
            merged = _kf()._merge_insurance_premium_disclosures(payload.get("insurance_premium_disclosures"), disclosures)
            payload["insurance_premium_disclosures"] = _kf()._merge_insurance_premium_disclosures(merged, fallback_disclosures)
        payload["insurance_premium_disclosures_checked_at"] = _kf()._utc_now().isoformat()
        payload["insurance_premium_disclosures_version"] = KAP_INSURANCE_PREMIUM_CACHE_VERSION
        payload.pop("insurance_premium_disclosures_error", None)
    else:
        payload["insurance_premium_disclosures_checked_at"] = _kf()._utc_now().isoformat()
        payload["insurance_premium_disclosures_error"] = str(kap_error)

    if persist and payload.get("ok"):
        try:
            _kf()._write_cache(cache_path, payload)
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
        "company_kind": _kf().classify_kap_company_kind(company_norm, title),
        "member_oid": member_oid,
        "source_url": (
            f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{(member or {}).get('permalink', '')}"
            if member_oid
            else MARKETVISUALS_INSURANCE_COMPANY_PAGES.get(company_norm, ("",))[0]
        ),
        "fetched_at": _kf()._utc_now().isoformat(),
        "quarters": [],
        "financials_error": str(error) if error else None,
    }
    if not _kf()._is_insurance_like_payload(payload, member) and company_norm not in MARKETVISUALS_INSURANCE_COMPANY_ALIASES:
        return None
    disclosures: List[Dict[str, Any]] = []
    kap_error: Optional[BaseException] = None
    try:
        if member_oid:
            disclosures = _kf()._fetch_insurance_premium_disclosures(member_oid=member_oid, cfg=cfg)
    except Exception as exc:
        kap_error = exc
    fallback_disclosures: List[Dict[str, Any]] = []
    if company_norm in MARKETVISUALS_INSURANCE_COMPANY_ALIASES or not disclosures:
        try:
            fallback_disclosures = _kf()._fetch_tsb_marketvisuals_premium_disclosures(
                company_key=company_norm,
                company_title=title,
                cfg=cfg,
            )
        except Exception as exc:
            if kap_error is None:
                kap_error = exc
    if fallback_disclosures and company_norm in MARKETVISUALS_INSURANCE_COMPANY_ALIASES:
        disclosures = _kf()._merge_insurance_premium_disclosures([], fallback_disclosures)
    else:
        disclosures = _kf()._merge_insurance_premium_disclosures(disclosures, fallback_disclosures)
    if not disclosures:
        if kap_error is not None:
            payload["insurance_premium_disclosures_error"] = str(kap_error)
        return None
    payload["insurance_premium_disclosures"] = _kf()._merge_insurance_premium_disclosures([], disclosures)
    payload["insurance_premium_disclosures_checked_at"] = _kf()._utc_now().isoformat()
    payload["insurance_premium_disclosures_version"] = KAP_INSURANCE_PREMIUM_CACHE_VERSION
    try:
        _kf()._write_cache(cache_path, payload)
    except Exception:
        pass
    return payload

