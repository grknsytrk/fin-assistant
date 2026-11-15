"""KAP portfolio holdings disclosure client and PDF parser."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import httpx

_ORIGINAL_EXPORTS: Dict[str, Any] = {}


def _fund_service() -> Any:
    """Resolve ``app.fund_service`` lazily to avoid import cycles and honor monkeypatches."""
    mod = sys.modules.get("app.fund_service")
    if mod is not None:
        return mod
    import app.fund_service as fund_service

    return fund_service


def _seam(name: str) -> Any:
    """Resolve a moved symbol via ``app.fund_service`` unless patched locally on this module."""
    local_obj = globals()[name]
    orig_obj = _ORIGINAL_EXPORTS.get(name, local_obj)
    if local_obj is not orig_obj:
        return local_obj
    return getattr(_fund_service(), name, local_obj)


KAP_HOLDINGS_SOURCE = "kap_portfolio_allocation_report"
KAP_HOLDINGS_PARSE_VERSION = 24
_KAP_NUMBER_PATTERN = re.compile(
    r"-?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?(?!\d)|\d+[.,]\d+|\d+)(?:\s*%)?"
)
_KAP_DATE_PATTERN = re.compile(r"\b\d{2}/\d{2}/\d{2,4}\b")
# A trailing ISIN may be glued directly to a borsa/sözleşme code that is
# itself glued to the currency token (e.g. ``14,41TL 80100511TRABTCIM91F5``),
# so we tolerate an optional numeric prefix between the currency and the ISIN.
_KAP_ISIN_TAIL_PATTERN = re.compile(
    r"(?P<weight>-?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?))\s*%?\s*(?:TL|TRY|USD|EUR|JPY|GBP)?\s*\d*[A-Z]{2}[A-Z0-9]{9}[0-9]\s*$",
    flags=re.IGNORECASE,
)
# Some KAP PDF text extractions omit the final FTD percentage and glue the
# borsa/sözleşme number directly to the ISIN (for example
# ``80100517TREDSTF00012`` or ``80_100_5TREDSTF00012``).  That number is not a
# portfolio weight and must never be selected as one.
_KAP_BORSA_CONTRACT_TAIL_PATTERN = re.compile(
    r"(?P<contract>(?:\d{2}(?:[_-]\d{3})+|\d{6,}))\s*"
    r"(?P<isin>[A-Z]{2}[A-Z0-9]{6,})\s*$",
    flags=re.IGNORECASE,
)
_KAP_MAX_REASONABLE_WEIGHT = 1000.0
# Lines that introduce continuation rows for an already-listed position
# (e.g. ``Tem.Ver.``/``Teminat Veren`` collateral lender prefixes) must not
# be treated as a new asset code, otherwise the parser invents a phantom row
# and swallows the next real position into its buffer.
_KAP_LINE_PREFIX_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:Tem\.?\s*Ver\.?|Teminat\s+Veren)\s+", flags=re.IGNORECASE), "Tem.Ver."),
    (re.compile(r"^Tem\.?\s+", flags=re.IGNORECASE), "Tem."),
)
# Category contexts that publish numeric rows whose leading token is a
# borsa-listed stock code (e.g. REPO/Mevduat collateral booked against
# ``BTCIM`` or a deposit booked against ``T.IS BANKASI``).  Such rows must
# never be promoted to ``local_equity``; doing so otherwise overwrites the
# real stock holding with the negative collateral leg.
_KAP_NON_HOLDING_CONTEXT_TOKENS = (
    "REPO",
    "TREPO",
    "T.REPO",
    "TPP",
    "BPP",
    "BORCLANMA",
    "BONO",
    "TAHVIL",
    "MEVDUAT",
    "NAKIT",
    "DOVIZ",
    "TEMINAT",
    "VIOP",
    "KIRA SERTIFIKA",
    "VARANT",
    "DIGER VARLIK",
    "DIGER VAR",
)
_KAP_POSITION_STOPWORDS = {
    "AÇIKLAMA",
    "BIRIM",
    "BİRİM",
    "BORSA",
    "DİĞER",
    "DÖVİZ",
    "FAIZ",
    "FAİZ",
    "FON",
    "FONU",
    "HISSE",
    "HİSSE",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "GRUP",
    "ISIN",
    "İÇ",
    "İHRAÇCI",
    "MENKUL",
    "NOMINAL",
    "NOMİNAL",
    "ORAN",
    "ORANI",
    "PORTFÖY",
    "SATIN",
    "SAYISI",
    "SIRKETIN",
    "ŞIRKETIN",
    "ŞİRKETİN",
    "TARIH",
    "TARİH",
    "TEM",
    "TEM.",
    "TEM.VER",
    "TEM.VER.",
    "TEMINAT",
    "TEMİNAT",
    "TOPLAM",
    "TUTAR",
    "TUTARI",
    "VADE",
    "VADEYE",
    "KALAN",
}
_KAP_INCLUDED_HOLDING_TYPES = {"local_equity", "fund", "foreign_equity", "foreign_fund"}
_KAP_FOREIGN_ISIN_PROVIDER_SYMBOLS = {
    "US0032641088": "SIVR",
    "US46428Q1094": "SLV",
}
_KAP_FOREIGN_ISIN_PROVIDER_NAMES = {
    "US0032641088": "abrdn Physical Silver Shares ETF",
    "US46428Q1094": "iShares Silver Trust",
    "CH0183135992": "Swisscanto (CH) Silver ETF",
    "CH0118929048": "UBS Silver ETF USD acc",
    "CA37964K1012": "Global X Silver ETF",
}
_KAP_FOREIGN_SYMBOL_STOPWORDS = {
    "AMERICA",
    "CMN",
    "CORP",
    "CORPORATION",
    "EQUITY",
    "HOLDINGS",
    "INC",
    "LIMITED",
    "MINERALS",
    "OF",
    "SE",
}
_KAP_FOREIGN_EXCHANGE_SUFFIXES = {
    "US": "",
    "FP": ".PA",
    "PA": ".PA",
    "SW": ".SW",
    "LN": ".L",
    "CN": ".TO",
    "GY": ".DE",
    "GR": ".DE",
    "NA": ".AS",
    "AS": ".AS",
}
_KAP_FOREIGN_EQUITY_PREFIX_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z0-9]{1,12})\s+(?P<exchange>US|FP|PA|SW|LN|CN|GY|GR|NA|AS)\b(?:\s+EQUITY\b)?",
    flags=re.IGNORECASE,
)
# Some KAP PDFs publish the portfolio table with the security name in the
# first column and the ISIN/fund code glued to the weight
# (``APPLE US 24,247.00 398,638,321.90 2.90US0378331005AAPL.O``).  Category
# headers in that layout are also split across lines and prefixed with a
# letter marker such as ``:Ğ) YABANCI HİSSE`` / ``SENETLERİ``.
_KAP_ISIN_PATTERN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_KAP_NAME_FIRST_CATEGORY_PATTERN = re.compile(
    r"^\s*:\s*[A-ZÇĞİÖŞÜ]{1,3}\s*\)\s*(?P<label>.*)$",
    flags=re.IGNORECASE,
)
_KAP_NAME_FIRST_PAGE_FOOTER_PATTERN = re.compile(
    r"^\s*\d{2}[./]\d{2}[./]\d{4}\s+\d{2}:\d{2}:\d{2}\s+Sayfa\s+\d+\s*/\s*\d+\s*$",
    flags=re.IGNORECASE,
)
_KAP_NAME_FIRST_CATEGORY_WORDS = {
    "ARACLARI",
    "BORCLANMA",
    "BORSA",
    "BYF",
    "DIGER",
    "DOVIZ",
    "FON",
    "FONLARI",
    "FONU",
    "HISSE",
    "KATILMA",
    "KIYMETLI",
    "MADEN",
    "MEVDUAT",
    "NAKIT",
    "PARA",
    "PAYI",
    "PAYLARI",
    "PIYASASI",
    "REPO",
    "SENEDI",
    "SENETLERI",
    "SERBEST",
    "SERTIFIKA",
    "SERTIFIKALAR",
    "SERMAYE",
    "TL",
    "TURK",
    "VADELI",
    "VARLIK",
    "VARLIKLAR",
    "YABANCI",
    "YATIRIM",
    "YP",
}
_KAP_NAME_FIRST_US_TICKER_SUFFIXES = {"K", "N", "O", "OQ", "P", "UQ", "US", "UN"}

_KAP_EXTRA_STOCK_SYMBOLS = {
    # Keep the parser tolerant of very recent KAP rows before the bundled
    # BIST universe fallback is refreshed.
    "DSTKF",
    "TERA",
    "TEHOL",
    "TRHOL",
}
_KAP_BIST_STOCK_SYMBOLS: Optional[set[str]] = None
_KAP_TURKISH_MONTH_WORDS = {
    "OCAK",
    "SUBAT",
    "ŞUBAT",
    "MART",
    "NISAN",
    "NİSAN",
    "MAYIS",
    "HAZIRAN",
    "HAZİRAN",
    "TEMMUZ",
    "AGUSTOS",
    "AĞUSTOS",
    "EYLUL",
    "EYLÜL",
    "EKIM",
    "EKİM",
    "KASIM",
    "ARALIK",
}


def _kap_headers(accept: str = "application/json, text/plain, */*") -> Dict[str, str]:
    return {
        "Accept": accept,
        "Accept-Language": "tr",
        "Content-Type": "application/json",
        "Referer": f"{_fund_service().KAP_BASE_URL}/tr/",
        "User-Agent": _fund_service().FINTABLES_USER_AGENT,
    }


def _kap_url(path: str) -> str:
    if str(path).startswith(("http://", "https://")):
        return str(path)
    return f"{_fund_service().KAP_BASE_URL}/{str(path).lstrip('/')}"


def _kap_request_get(url: str, headers: Dict[str, str]) -> httpx.Response:
    client = _fund_service()._get_kap_client()
    try:
        return client.get(url, headers=headers)
    except _fund_service().httpx.RemoteProtocolError:
        return client.get(url, headers=headers)


def _kap_get_json(path: str) -> Any:
    response = _seam("_kap_request_get")(_seam("_kap_url")(path), _seam("_kap_headers")())
    response.raise_for_status()
    return response.json()


def _kap_post_json(path: str, payload: Dict[str, Any]) -> Any:
    client = _fund_service()._get_kap_client()
    response = client.post(_seam("_kap_url")(path), json=payload, headers=_seam("_kap_headers")())
    response.raise_for_status()
    return response.json()


def _kap_get_text(path: str) -> str:
    response = _seam("_kap_request_get")(_seam("_kap_url")(path), _seam("_kap_headers")("text/html,*/*"))
    response.raise_for_status()
    return response.text


def _kap_get_bytes(path: str, accept: str = "application/pdf,*/*") -> bytes:
    response = _seam("_kap_request_get")(_seam("_kap_url")(path), _seam("_kap_headers")(accept))
    response.raise_for_status()
    return bytes(response.content)


def _kap_search_fund_metadata(fund_code: str) -> Optional[Dict[str, Any]]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    if not normalized:
        return None
    payload = _seam("_kap_post_json")("/tr/api/search/combined", {"keyword": normalized})
    candidates: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for group in payload:
            if not isinstance(group, dict):
                continue
            results = group.get("results")
            if isinstance(results, list):
                candidates.extend(row for row in results if isinstance(row, dict))
    for row in candidates:
        if str(row.get("searchType") or "").upper() != "F":
            continue
        candidate_code = _fund_service().normalize_fund_code(_fund_service()._normalize_match_text(row.get("cmpOrFundCode")))
        if candidate_code != normalized:
            continue
        fund_oid = str(row.get("memberOrFundOid") or "").strip()
        if not fund_oid:
            continue
        return {
            "fund_code": normalized,
            "fund_oid": fund_oid,
            "fund_name": str(row.get("searchValue") or "").strip() or None,
        }
    return None


def _kap_portfolio_subject_oid(fund_oid: str) -> str:
    oid = str(fund_oid or "").strip()
    if not oid:
        return _fund_service().KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID
    try:
        html = _seam("_kap_get_text")(f"/tr/fon-bildirimleri/{quote(oid)}")
    except Exception:
        return _fund_service().KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID
    match = re.search(
        r'\\?"value\\?"\s*:\s*\\?"([^"\\]+)\\?"\s*,\s*\\?"label\\?"\s*:\s*\\?"Portföy Dağılım Raporu',
        html,
    )
    if match:
        return match.group(1)
    return _fund_service().KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID


def _kap_disclosure_basic(row: Dict[str, Any]) -> Dict[str, Any]:
    basic = row.get("disclosureBasic")
    return dict(basic) if isinstance(basic, dict) else dict(row)


def _kap_disclosure_index(row: Dict[str, Any]) -> Optional[int]:
    basic = _seam("_kap_disclosure_basic")(row)
    try:
        value = int(basic.get("disclosureIndex") or row.get("disclosureIndex") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parse_kap_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%d.%m.%Y %H:%M:%S", "%Y.%m.%d %H:%M:%S", "%d.%m.%Y", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _kap_list_portfolio_disclosures(fund_oid: str, subject_oid: str) -> List[Dict[str, Any]]:
    oid = quote(str(fund_oid or "").strip())
    subject = quote(str(subject_oid or _fund_service().KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID).strip())
    if not oid:
        return []
    year = _fund_service()._utc_now().year
    ranges = [str(max(30, min(2555, _fund_service().KAP_HOLDINGS_LOOKBACK_DAYS))), str(year), str(year - 1)]
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    for range_value in ranges:
        try:
            payload = _seam("_kap_get_json")(f"/tr/api/disclosure/filter/FILTERYFBF/{oid}/{subject}/{quote(range_value)}")
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            disclosure_index = _seam("_kap_disclosure_index")(row)
            if disclosure_index:
                rows_by_index[disclosure_index] = dict(row)
        if len(rows_by_index) >= 2 and range_value.isdigit() and int(range_value) >= 365:
            break
    rows = list(rows_by_index.values())
    rows.sort(
        key=lambda row: (
            _seam("_parse_kap_datetime")(_seam("_kap_disclosure_basic")(row).get("publishDate")) or datetime.min,
            _seam("_kap_disclosure_index")(row) or 0,
        ),
        reverse=True,
    )
    return rows


def _kap_fetch_report_detail(disclosure_index: int) -> Optional[Dict[str, Any]]:
    payload = _seam("_kap_get_json")(f"/tr/api/notification/attachment-detail/{int(disclosure_index)}")
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return dict(payload[0])
    if isinstance(payload, dict):
        return dict(payload)
    return None


def _kap_report_basic(detail: Dict[str, Any]) -> Dict[str, Any]:
    disclosure = detail.get("disclosure")
    if isinstance(disclosure, dict):
        basic = disclosure.get("disclosureBasic")
        if isinstance(basic, dict):
            return dict(basic)
    basic = detail.get("disclosureBasic")
    return dict(basic) if isinstance(basic, dict) else {}


def _kap_report_attachments(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    attachments = detail.get("attachments")
    if not isinstance(attachments, list):
        return []
    return [dict(item) for item in attachments if isinstance(item, dict)]


def _kap_download_attachment(obj_id: str) -> bytes:
    return _seam("_kap_get_bytes")(f"/tr/api/file/download/{quote(str(obj_id or '').strip())}")


def _extract_kap_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    pdf_data = bytes(data or b"")
    pdf_start = pdf_data.find(b"%PDF")
    if pdf_start > 0:
        pdf_data = pdf_data[pdf_start:]
    try:
        reader = PdfReader(io.BytesIO(pdf_data))
    except Exception:
        return ""
    parts: List[str] = []
    image_pages: List[int] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        parts.append(text)
        if len(re.sub(r"\s+", "", text)) < 40:
            image_pages.append(index)
    used_ocr = False
    if image_pages:
        try:
            from app.kap_pdf_ocr import extract_pdf_ocr_pages

            for index, text in extract_pdf_ocr_pages(pdf_data, image_pages).items():
                if text.strip():
                    parts[index] = text
                    used_ocr = True
        except Exception:
            logging.getLogger(__name__).warning("KAP PDF OCR unavailable", exc_info=True)
    text = "\n".join(part for part in parts if part)
    return f"KAP_PDF_OCR\n{text}" if used_ocr else text


def _month_end_iso(year: int, month: int) -> Optional[str]:
    if year < 1900 or month < 1 or month > 12:
        return None
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    return (next_month - timedelta(days=1)).isoformat()


def _kap_report_date(detail: Dict[str, Any]) -> Optional[str]:
    basic = _seam("_kap_report_basic")(detail)
    for attachment in _seam("_kap_report_attachments")(detail):
        filename = str(attachment.get("fileName") or "")
        match = re.search(r"_(20\d{2})[._-](\d{1,2})", filename)
        if match:
            return _seam("_month_end_iso")(int(match.group(1)), int(match.group(2)))
    try:
        year = int(basic.get("year") or 0)
        month = int(basic.get("donem") or 0)
    except (TypeError, ValueError):
        return None
    return _seam("_month_end_iso")(year, month)


def _holdings_report_index(report: Any) -> Optional[int]:
    if not isinstance(report, dict):
        return None
    try:
        value = int(report.get("disclosure_index") or report.get("disclosureIndex") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _holdings_positions_hash(positions: Iterable[Dict[str, Any]]) -> str:
    static_keys = (
        "fund_code",
        "asset_code",
        "asset_name",
        "asset_type",
        "weight",
        "previous_weight",
        "weight_change",
        "change_status",
        "amount",
        "market_value",
        "report_date",
        "previous_report_date",
        "source_report_url",
        "source_type",
        "parse_confidence",
    )
    materialized: List[Dict[str, Any]] = []
    for position in positions:
        if not isinstance(position, dict):
            continue
        materialized.append({key: position.get(key) for key in static_keys})
    materialized.sort(key=lambda row: (str(row.get("asset_code") or ""), str(row.get("asset_name") or "")))
    encoded = _fund_service()._stable_json_dumps(materialized).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _holdings_meta_with_runtime_cache_flags(
    meta: Dict[str, Any],
    *,
    cache_hit: bool,
    stale: bool,
    static_cache_hit: bool,
) -> Dict[str, Any]:
    result = dict(meta)
    result["cache_hit"] = cache_hit
    result["stale"] = stale
    result["static_cache_hit"] = static_cache_hit
    result["cache_policy"] = "monthly_report"
    return result


def _holdings_disclosure_check_due(cached_meta: Dict[str, Any]) -> bool:
    check = cached_meta.get("disclosure_check")
    if not isinstance(check, dict):
        return True
    age = _fund_service()._cache_age_seconds(check.get("checked_at"))
    return age is None or age > _fund_service().FUNDS_HOLDINGS_DISCLOSURE_CHECK_TTL_SECONDS


def _holdings_disclosure_check_meta(disclosures: List[Dict[str, Any]]) -> Dict[str, Any]:
    latest_index = _seam("_kap_disclosure_index")(disclosures[0]) if disclosures else None
    previous_index = _seam("_kap_disclosure_index")(disclosures[1]) if len(disclosures) > 1 else None
    return {
        "checked_at": _fund_service()._utc_now_iso(),
        "ttl_seconds": _fund_service().FUNDS_HOLDINGS_DISCLOSURE_CHECK_TTL_SECONDS,
        "latest_disclosure_index": latest_index,
        "previous_disclosure_index": previous_index,
        "report_count": len(disclosures),
    }


def _cached_holdings_matches_disclosures(cached: Optional[Dict[str, Any]], disclosures: List[Dict[str, Any]]) -> bool:
    if not isinstance(cached, dict) or not list(cached.get("positions") or []):
        return False
    meta = cached.get("source_metadata") if isinstance(cached.get("source_metadata"), dict) else {}
    if _fund_service()._coerce_int(meta.get("parser_version")) != _seam("KAP_HOLDINGS_PARSE_VERSION"):
        return False

    latest_index = _seam("_kap_disclosure_index")(disclosures[0]) if disclosures else None
    previous_index = _seam("_kap_disclosure_index")(disclosures[1]) if len(disclosures) > 1 else None
    cached_latest = _seam("_holdings_report_index")(meta.get("latest_report"))
    cached_previous = _seam("_holdings_report_index")(meta.get("previous_report"))
    if latest_index and latest_index != cached_latest:
        return False
    if previous_index and previous_index != cached_previous:
        return False

    actual_hash = _seam("_holdings_positions_hash")(list(cached.get("positions") or []))
    stored_hash = str(meta.get("positions_hash") or "").strip()
    return not stored_hash or stored_hash == actual_hash


def _cached_holdings_with_disclosure_check(
    processed_dir: Path,
    fund_code: str,
    cached: Dict[str, Any],
    disclosures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = dict(cached)
    positions = list(payload.get("positions") or [])
    previous_meta = dict(payload.get("source_metadata") or {})
    meta = dict(previous_meta)
    meta["parser_version"] = _seam("KAP_HOLDINGS_PARSE_VERSION")
    meta["positions_hash"] = _seam("_holdings_positions_hash")(positions)
    meta["disclosure_check"] = _seam("_holdings_disclosure_check_meta")(disclosures)
    meta["last_static_cache_validated_at"] = _fund_service()._utc_now_iso()
    payload["source_metadata"] = _seam("_holdings_meta_with_runtime_cache_flags")(
        meta,
        cache_hit=True,
        stale=False,
        static_cache_hit=True,
    )
    holdings_path = _fund_service()._holdings_path(processed_dir, fund_code)
    prev_disclosure = dict(previous_meta.get("disclosure_check") or {})
    same_report_metadata = (
        previous_meta.get("parser_version") == _seam("KAP_HOLDINGS_PARSE_VERSION")
        and previous_meta.get("positions_hash") == meta["positions_hash"]
        and prev_disclosure.get("latest_disclosure_index")
        == meta["disclosure_check"].get("latest_disclosure_index")
        and holdings_path.exists()
    )
    if _fund_service().database_enabled() and same_report_metadata:
        # Only refreshed validation timestamps; avoid re-uploading unchanged holdings JSON to Supabase.
        _fund_service()._write_local_json(holdings_path, payload)
    else:
        _fund_service()._write_json(holdings_path, payload)
    return payload


def _kap_attachment_text_from_cache(
    processed_dir: Path,
    disclosure_index: Any,
    attachment: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    obj_id = str(attachment.get("objId") or "").strip()
    cache_path = _fund_service()._holdings_attachment_text_path(processed_dir, disclosure_index, obj_id)
    cached = _fund_service()._read_json(cache_path)
    if (
        cached
        and _fund_service()._coerce_int(cached.get("schema_version")) == _fund_service().KAP_HOLDINGS_ATTACHMENT_TEXT_CACHE_VERSION
        and str(cached.get("obj_id") or "").strip() == obj_id
        and isinstance(cached.get("text"), str)
        and str(cached.get("text") or "").strip()
    ):
        return str(cached.get("text") or ""), {
            "attachment_obj_id": obj_id,
            "attachment_text_cache_hit": True,
            "attachment_text_cache_path": str(cache_path),
        }

    data = _seam("_kap_download_attachment")(obj_id)
    text = _seam("_extract_kap_pdf_text")(data)
    if text.strip():
        _fund_service()._write_json(
            cache_path,
            {
                "schema_version": _fund_service().KAP_HOLDINGS_ATTACHMENT_TEXT_CACHE_VERSION,
                "disclosure_index": disclosure_index,
                "obj_id": obj_id,
                "file_name": attachment.get("fileName"),
                "fetched_at": _fund_service()._utc_now_iso(),
                "text": text,
            },
        )
    return text, {
        "attachment_obj_id": obj_id,
        "attachment_text_cache_hit": False,
        "attachment_text_cache_path": str(cache_path),
    }


def _kap_source_url(disclosure_index: Any) -> Optional[str]:
    try:
        idx = int(disclosure_index or 0)
    except (TypeError, ValueError):
        return None
    return f"{_fund_service().KAP_BASE_URL}/tr/Bildirim/{idx}" if idx > 0 else None


def _kap_number_matches(block: str) -> List[re.Match[str]]:
    date_spans = [match.span() for match in _KAP_DATE_PATTERN.finditer(block)]

    def inside_date(span: Tuple[int, int]) -> bool:
        return any(span[0] >= start and span[1] <= end for start, end in date_spans)

    def embedded_in_symbol(match: re.Match[str]) -> bool:
        start, _end = match.span()
        if start <= 0:
            return False
        previous = str(block or "")[start - 1 : start]
        return bool(previous and previous.isalnum())

    return [
        match
        for match in _KAP_NUMBER_PATTERN.finditer(block)
        if not inside_date(match.span()) and not embedded_in_symbol(match)
    ]


def _coerce_kap_number_text(raw: Any) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("\xa0", "").replace(" ", "").replace("%", "").strip()
    if not text or text in {"-", "+", "."}:
        return None
    sign = ""
    if text[:1] in {"+", "-"}:
        sign = text[:1]
        text = text[1:]
    if not text:
        return None
    comma_count = text.count(",")
    dot_count = text.count(".")
    if comma_count and dot_count:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif comma_count:
        parts = text.split(",")
        if comma_count > 1:
            text = "".join(parts)
        elif len(parts[-1]) == 3 and parts[0] != "0":
            text = "".join(parts)
        else:
            text = text.replace(",", ".")
    elif dot_count:
        parts = text.split(".")
        if dot_count > 1:
            text = "".join(parts)
        elif len(parts[-1]) == 3 and parts[0] != "0":
            text = "".join(parts)
    try:
        result = float(f"{sign}{text}")
        return result if result == result else None
    except ValueError:
        return None


def _kap_number_tokens(block: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for match in _kap_number_matches(block):
        raw = match.group(0)
        value = _coerce_kap_number_text(raw)
        if value is None:
            continue
        tokens.append(
            {
                "match": match,
                "text": raw,
                "value": value,
                "is_percent": "%" in raw,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def _kap_borsa_contract_tail_match(block: str) -> Optional[re.Match[str]]:
    return _KAP_BORSA_CONTRACT_TAIL_PATTERN.search(str(block or ""))


def _kap_token_is_inside(token: Dict[str, Any], match: Optional[re.Match[str]]) -> bool:
    if match is None:
        return False
    try:
        token_start = int(token.get("start"))
        token_end = int(token.get("end"))
    except (TypeError, ValueError):
        return False
    return token_start >= match.start("contract") and token_end <= match.end("contract")


def _kap_weight_candidate_is_valid(raw_value: Any, value: Optional[float]) -> bool:
    if value is None or not math.isfinite(value) or abs(value) > _KAP_MAX_REASONABLE_WEIGHT:
        return False
    raw = str(raw_value or "").replace("%", "").replace(" ", "").strip()
    # A plain 6+ digit integer directly before an ISIN is a borsa/sözleşme
    # number, not a percentage. Decimal and thousands-formatted percentages
    # are still handled by the numeric parser and magnitude guard above.
    if raw and not any(separator in raw for separator in (",", ".")) and len(re.sub(r"\D", "", raw)) >= 6:
        return False
    return True


def _kap_extract_weight_candidate(
    compact: str,
    number_tokens: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return the best row weight token and whether it used a fallback.

    KAP has both FPD and FTD percentage columns.  The trailing FTD value is
    preferred, but PDF text occasionally drops it and leaves the borsa number
    glued to the ISIN.  In that case use the last valid percentage-like token
    outside the contract-number span and mark the row as approximate.
    """
    contract_tail = _kap_borsa_contract_tail_match(compact)
    percent_tokens = [
        token
        for token in number_tokens
        if token.get("is_percent")
        and _kap_weight_candidate_is_valid(token.get("text"), float(token.get("value")))
        and not _kap_token_is_inside(token, contract_tail)
    ]
    if percent_tokens:
        return percent_tokens[-1], False

    tail = None if contract_tail else _KAP_ISIN_TAIL_PATTERN.search(compact)
    if tail:
        raw_weight = tail.group("weight")
        weight = _coerce_kap_number_text(raw_weight)
        if _kap_weight_candidate_is_valid(raw_weight, weight):
            return {
                "start": tail.start("weight"),
                "end": tail.end("weight"),
                "text": raw_weight,
                "value": weight,
            }, False

    fallback_tokens = [
        token
        for token in number_tokens
        if _kap_weight_candidate_is_valid(token.get("text"), float(token.get("value")))
        and not _kap_token_is_inside(token, contract_tail)
    ]
    if not fallback_tokens:
        return None, True
    return fallback_tokens[-1], True


def _kap_extract_isin(compact: str) -> Optional[str]:
    candidates = [
        match.group(1)
        for match in re.finditer(
            r"(?<![A-Z])([A-Z]{2}[A-Z0-9]{9}[0-9])(?![A-Z0-9])",
            str(compact or "").upper(),
        )
    ]
    return candidates[-1] if candidates else None


def _kap_foreign_provider_symbol_for_exchange(symbol: Any, exchange: Any) -> Optional[str]:
    base = _fund_service().normalize_fund_code(symbol).replace(".", "")
    exchange_code = _fund_service().normalize_fund_code(exchange).replace(".", "")
    if not base or base in _KAP_FOREIGN_SYMBOL_STOPWORDS or base in _KAP_POSITION_STOPWORDS:
        return None
    suffix = _KAP_FOREIGN_EXCHANGE_SUFFIXES.get(exchange_code)
    if suffix is None:
        return None
    return f"{base}{suffix}"


def _kap_foreign_prefixed_security(compact: str) -> Optional[Dict[str, Any]]:
    text = " ".join(str(compact or "").replace("\xa0", " ").split())
    match = _KAP_FOREIGN_EQUITY_PREFIX_PATTERN.match(text)
    if not match:
        return None
    symbol = _fund_service().normalize_fund_code(match.group("symbol")).replace(".", "")
    exchange = _fund_service().normalize_fund_code(match.group("exchange")).replace(".", "")
    provider_symbol = _seam("_kap_foreign_provider_symbol_for_exchange")(symbol, exchange)
    if not provider_symbol:
        return None
    return {
        "code": symbol,
        "exchange": exchange,
        "provider_symbol": provider_symbol,
        "prefix_end": match.end(),
    }


def _kap_stock_symbol_set() -> set[str]:
    global _KAP_BIST_STOCK_SYMBOLS
    fs = _fund_service()
    if hasattr(fs, "_KAP_BIST_STOCK_SYMBOLS"):
        if fs._KAP_BIST_STOCK_SYMBOLS is not None:
            return fs._KAP_BIST_STOCK_SYMBOLS
    elif _KAP_BIST_STOCK_SYMBOLS is not None:
        return _KAP_BIST_STOCK_SYMBOLS
    symbols: set[str] = set()
    try:
        from app.kap_service import BIST_ALL_SYMBOLS_FALLBACK

        symbols.update(fs.normalize_fund_code(symbol) for symbol in BIST_ALL_SYMBOLS_FALLBACK)
    except Exception:
        pass
    symbols.update(_KAP_EXTRA_STOCK_SYMBOLS)
    resolved = {symbol for symbol in symbols if symbol}
    _KAP_BIST_STOCK_SYMBOLS = resolved
    if hasattr(fs, "_KAP_BIST_STOCK_SYMBOLS"):
        fs._KAP_BIST_STOCK_SYMBOLS = resolved
    return resolved

def _kap_is_stock_symbol(code: str) -> bool:
    symbol = _fund_service().normalize_fund_code(code).replace(".", "")
    return bool(symbol and symbol in _seam("_kap_stock_symbol_set")())


def _kap_stock_name_from_cache(processed_dir: Path, code: str) -> Optional[str]:
    symbol = _fund_service().normalize_fund_code(code).replace(".", "")
    if not symbol:
        return None
    resolved = _fund_service().get_instrument_name(processed_dir, "stock", symbol)
    if resolved:
        return resolved
    payload = _fund_service()._read_json(processed_dir / "kap_cache" / f"{symbol}.json")
    if not payload:
        return None
    for key in ("company_title", "title", "companyName", "name"):
        value = str(payload.get(key) or "").strip()
        if value and _fund_service().normalize_fund_code(value) != symbol:
            return value
    return None


def _kap_fund_name_map(processed_dir: Path) -> Dict[str, str]:
    result: Dict[str, str] = _fund_service().get_instrument_names(processed_dir, "fund")
    payload = _fund_service()._read_json(_fund_service()._snapshot_path(processed_dir))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _fund_service().normalize_fund_code(row.get("fund_code")).replace(".", "")
        name = str(row.get("name") or "").strip()
        if code and name:
            result.setdefault(code, name)
    return result


def _kap_looks_like_equity_symbol(code: str) -> bool:
    symbol = _fund_service().normalize_fund_code(code).replace(".", "")
    return bool(re.fullmatch(r"[A-Z0-9]{3,6}", symbol))


def _kap_looks_like_local_equity_symbol(code: Any) -> bool:
    """Recognize a BIST-style ticker before the reference universe catches up."""
    raw = str(code or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,6}", raw):
        return False
    return (
        _kap_looks_like_equity_symbol(raw)
        and raw not in _KAP_POSITION_STOPWORDS
        and raw not in _KAP_NON_HOLDING_CONTEXT_TOKENS
    )


def _kap_holding_name_is_header_noise(name: Any) -> bool:
    """Reject a row whose parsed name is mostly the repeated PDF header."""
    norm = _fund_service()._normalize_match_text(name)
    if not norm:
        return False
    header_terms = (
        "ISIN KODU",
        "VADE",
        "IHRACCI",
        "NOMINAL",
        "FAIZ ORANI",
        "DOVIZ CINSI",
        "BORSA",
        "SOZLESM",
        "TUTARI",
        "ODEME SAYISI",
        "NET DONUS",
        "SATIN ALIS",
        "BIRIM ALIS",
    )
    matches = sum(1 for term in header_terms if term in norm)
    return matches >= 3


def _kap_looks_like_fund_symbol(code: str) -> bool:
    symbol = _fund_service().normalize_fund_code(code).replace(".", "")
    if not symbol or symbol in _KAP_POSITION_STOPWORDS:
        return False
    if symbol.startswith(("TRT", "TRF", "TRB", "TRD", "F_")):
        return False
    return bool(re.fullmatch(r"[A-Z0-9]{3,6}", symbol))


def _kap_looks_like_foreign_symbol(code: str) -> bool:
    symbol = _fund_service().normalize_fund_code(code).replace(".", "")
    if not symbol or symbol in _KAP_POSITION_STOPWORDS or symbol in _KAP_FOREIGN_SYMBOL_STOPWORDS:
        return False
    if symbol.startswith(("TRT", "TRF", "TRB", "TRD", "TRY")):
        return False
    return bool(re.fullmatch(r"[A-Z0-9]{1,12}", symbol))


def _kap_is_included_holding_type(asset_type: Any) -> bool:
    return str(asset_type or "").strip().lower() in _KAP_INCLUDED_HOLDING_TYPES


def _kap_is_included_holding_context(context: str) -> bool:
    norm = _fund_service()._normalize_match_text(context)
    return any(token in norm for token in ("HISSE", "Y.FONU", "YATIRIM FONU", "BORSA Y", "BYF", "FON SEPETI", "YABANCI"))


def _kap_context_is_foreign(context: str) -> bool:
    norm = _fund_service()._normalize_match_text(context)
    return any(token in norm for token in ("YABANCI", "YABANC", "YURT DISI", "YURTDISI", "YP "))


def _kap_with_foreign_context(marker: Optional[str], current_category: str = "") -> Optional[str]:
    if not marker:
        return marker
    if _kap_context_is_foreign(marker):
        return marker
    current_norm = _fund_service()._normalize_match_text(current_category)
    marker_norm = _fund_service()._normalize_match_text(marker)
    if not current_norm.startswith("YABANCI"):
        return marker
    if not any(token in marker_norm for token in ("BORSA", "HISSE")):
        return marker
    return f"Yabancı {marker}"


def _kap_asset_type_from_context(context: str, code: str, name: str) -> str:
    context_norm = _fund_service()._normalize_match_text(context)
    name_norm = _fund_service()._normalize_match_text(name)
    haystack = _fund_service()._normalize_match_text(f"{context} {code} {name}")
    symbol = _fund_service().normalize_fund_code(code)
    is_foreign_context = _kap_context_is_foreign(context_norm)
    has_equity_context = any(token in context_norm for token in ("HISSE SENEDI", "HISSE SENETLERI", "HISSE"))
    has_fund_context = any(token in context_norm for token in ("Y.FONU", "YATIRIM FONU", "BORSA Y", "BYF", "FON SEPETI"))
    has_excluded_context = any(token in context_norm for token in _KAP_NON_HOLDING_CONTEXT_TOKENS)
    if symbol in {"USD", "EUR", "JPY", "GBP"}:
        return "cash_fx"
    # When the table section explicitly belongs to a non-holding instrument
    # type (REPO, mevduat, borçlanma, etc.) we must never classify the row
    # as a stock/fund holding even if the leading token happens to match a
    # known BIST symbol.  The same ticker can appear as a deposit/repo
    # collateral leg with a negative weight, which would otherwise overwrite
    # the genuine stock holding picked up from the equities section.
    if has_excluded_context and not has_equity_context and not has_fund_context:
        if any(token in haystack for token in ("REPO", "TAHVIL", "BONO", "BORCLANMA", "HAZINE", "KIRA SERTIFIKASI")):
            return "debt"
        if any(token in haystack for token in ("DOVIZ", "NAKIT", "MEVDUAT", "TEMINAT")):
            return "cash_fx"
        return "other"
    if has_fund_context and (is_foreign_context and _kap_looks_like_foreign_symbol(symbol) or _kap_looks_like_fund_symbol(symbol)):
        return "foreign_fund" if is_foreign_context else "fund"
    if not has_excluded_context and _kap_looks_like_fund_symbol(symbol) and any(token in name_norm for token in ("YATIRIM FONU", "PORTFOY", "BYF", "FON", "ETF")):
        return "foreign_fund" if is_foreign_context else "fund"
    if _seam("_kap_is_stock_symbol")(symbol):
        return "local_equity"
    if has_equity_context and _kap_looks_like_equity_symbol(symbol):
        return "foreign_equity" if is_foreign_context else "local_equity"
    if is_foreign_context and has_fund_context and _kap_looks_like_foreign_symbol(symbol):
        return "foreign_fund"
    if is_foreign_context and _kap_looks_like_foreign_symbol(symbol):
        return "foreign_equity"
    if any(token in haystack for token in ("REPO", "TAHVIL", "BONO", "BORCLANMA", "HAZINE", "KIRA SERTIFIKASI")):
        return "debt"
    if any(token in haystack for token in ("DOVIZ", "NAKIT", "MEVDUAT", "TEMINAT")):
        return "cash_fx"
    if symbol.startswith(("TRT", "TRF", "TRB")):
        return "debt"
    return "other"


def _kap_foreign_provider_symbol(
    code: Any,
    isin: Any = None,
    name: Any = None,
    exchange: Any = None,
) -> Optional[str]:
    isin_text = str(isin or "").strip().upper()
    if isin_text and isin_text in _KAP_FOREIGN_ISIN_PROVIDER_SYMBOLS:
        return _KAP_FOREIGN_ISIN_PROVIDER_SYMBOLS[isin_text]
    exchange_symbol = _seam("_kap_foreign_provider_symbol_for_exchange")(code, exchange)
    if exchange_symbol:
        return exchange_symbol
    symbol = _fund_service().normalize_fund_code(code).replace(".", "")
    if not symbol or symbol in _KAP_POSITION_STOPWORDS or symbol in _KAP_FOREIGN_SYMBOL_STOPWORDS:
        return None
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", symbol):
        return None
    suffix_map = (
        ("US", ""),
        ("CN", ".TO"),
        ("SW", ".SW"),
        ("LN", ".L"),
        ("NA", ".AS"),
        ("AS", ".AS"),
        ("GY", ".DE"),
        ("GR", ".DE"),
        ("FP", ".PA"),
        ("PA", ".PA"),
    )
    for suffix, exchange_suffix in suffix_map:
        if symbol.endswith(suffix) and len(symbol) > len(suffix) + 1:
            base = symbol[: -len(suffix)]
            if base:
                return f"{base}{exchange_suffix}"
    return symbol


def _kap_category_marker(line: str) -> Optional[str]:
    norm = _fund_service()._normalize_match_text(line)
    if not norm or len(norm) > 80:
        return None
    norm = re.sub(r"^(?::\s*)?[A-Z]{1,3}\s*[\).:-]\s*", "", norm).strip()
    compact = re.sub(r"[\s.]", "", norm)
    if norm == "DOVIZ" and str(line or "").strip() == str(line or "").strip().upper():
        return None
    if "YABANCI" in norm and any(token in norm for token in ("SERMAYE", "MENKUL KIYMET", "PIYASASI")):
        return "Yabancı"
    if compact.startswith("YABSERPIY"):
        return "Yabancı"
    if norm in {"HISSE", "HISSE SENEDI", "HISSE SENETLERI", "PAY", "PAYLAR", "A.PAY", "YP HISSE", "YABANCI HISSE"}:
        return "Yabancı Hisse Senedi" if "YP " in norm or "YABANCI" in norm else "Hisse Senedi"
    if any(token in norm for token in ("HISSE SENEDI", "HISSE SENETLERI")):
        return "Yabancı Hisse Senedi" if "YABANCI" in norm else "Hisse Senedi"
    if norm in {"FON", "FONLAR", "Y.FONU", "YATIRIM FONU", "YATIRIM FONLARI", "BYF"}:
        return "Yabancı Yatırım Fonu/BYF" if "YABANCI" in norm else "Yatırım Fonu/BYF"
    if norm in {"BORSA Y.FONU", "BORSA YATIRIM FONU"} or (
        "FON" in norm and any(token in norm for token in ("YATIRIM", "BORSA Y", "BYF"))
    ):
        return "Yabancı Borsa Yatırım Fonu/BYF" if "YABANCI" in norm else "Borsa Yatırım Fonu/BYF"
    if "REPO" in norm and any(token in norm for token in ("TEM", "TUTAR")):
        return None
    if norm in {"T.REPO", "REPO", "BORCLANMA SENETLERI", "KIRA SERTIFIKALARI"} or (
        "REPO" in norm and "TEMINAT" not in norm and "TMNT" not in norm
    ):
        return "Borçlanma"
    if norm == "VIOP" or compact.startswith("TUREVARAC"):
        return "VIOP"
    if norm in {"MEVDUAT", "NAKIT", "DOVIZ", "DVZ", "DOVIZ/NAKIT", "DOVIZ NAKIT"} or ("MEV" in norm and "UAT" in norm):
        return "Döviz/Nakit"
    if "VARANT" in norm or "DIGERVAR" in compact:
        return "Diğer"
    return None


def _kap_strip_inline_category(line: str) -> Tuple[str, Optional[str]]:
    text = str(line or "").strip()
    patterns = (
        (r"^Hisse(?:\s+Senedi|\s+Senetleri)?(?:\s+(?:T.rk|Turk|Türk|Yabanc.|Yabanci|Yabancı))?(?:\s+|$)", "Hisse Senedi"),
        (r"^(?:(?:Borsa|Yabanc.|Yabanci|Yabancı|T.rk|Turk|Türk)\s+)?(?:Y\.?\s*Fonu|Yatırım\s+Fonu|BYF)(?:\s+(?:T.rk|Turk|Türk|Yabanc.|Yabanci|Yabancı))?(?:\s+|$)", "Yatırım Fonu/BYF"),
        (r"^T\.?\s*REPO(?:\s+|$)", "Borçlanma"),
        (r"^Döviz\s*/?\s*Nakit(?:\s+|$)", "Döviz/Nakit"),
        (r"^VIOP\s+Nakit\s+Teminatı(?:\s+|$)", "Döviz/Nakit"),
        (r"^(?:T.rk|Turk|Türk|Yabanc.|Yabanci|Yabancı)(?:\s+|$)", None),
    )
    for pattern, marker in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            stripped = text[match.end():].strip()
            if _fund_service()._normalize_match_text(stripped) in {"TURK", "YABANCI"}:
                stripped = ""
            if marker and re.search(r"Yabanc.|Yabanci|Yabancı", text[: match.end()], flags=re.IGNORECASE):
                marker = f"Yabancı {marker}"
            return stripped, marker
    return text, None


def _kap_strip_continuation_prefix(line: str) -> Tuple[str, Optional[str]]:
    """Strip leading collateral / lender prefixes (e.g. ``Tem.Ver.``).

    KAP portfolio PDFs often print collateral entries with a ``Teminat
    Veren`` prefix in front of the actual security code.  Without
    stripping, the parser would treat the prefix word as a brand new
    asset code and either invent a phantom row or swallow the next real
    position into its buffer.
    """
    text = str(line or "").strip()
    if not text:
        return line, None
    for pattern, marker in _KAP_LINE_PREFIX_PATTERNS:
        match = pattern.match(text)
        if match:
            return text[match.end():].strip(), marker
    return line, None


def _kap_line_starts_position(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", text, flags=re.IGNORECASE):
        return False
    if _kap_foreign_prefixed_security(text):
        return True
    norm = _fund_service()._normalize_match_text(text)
    if not _kap_number_matches(text) and "PORTFOY" in norm and "FON" in norm:
        return False
    first = text.split()[0].strip(":-,;")
    if not first or first.upper() in _KAP_POSITION_STOPWORDS:
        return False
    first_norm = _fund_service()._normalize_match_text(first)
    if any(first_norm.startswith(f"{month}-20") for month in _KAP_TURKISH_MONTH_WORDS):
        return False
    if first[:1].isdigit() or len(first) < 2 or len(first) > 16:
        return False
    return bool(re.match(r"^[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9._-]*$", first, flags=re.IGNORECASE))


def _kap_row_complete(block: str) -> bool:
    compact = " ".join(str(block or "").split())
    if _KAP_ISIN_TAIL_PATTERN.search(compact):
        return True
    tokens = _kap_number_tokens(compact)
    if len([token for token in tokens if token.get("is_percent")]) >= 2:
        return True
    if len(tokens) < 4:
        return False
    return any(token in compact.upper() for token in (" TL", "TRY", "USD", "EUR", "JPY", "GBP"))


def _kap_buffer_is_header_noise(buffer: List[str]) -> bool:
    compact = " ".join(str(part or "").strip() for part in buffer if str(part or "").strip())
    if not compact or _kap_number_matches(compact):
        return False
    if _kap_foreign_prefixed_security(compact):
        return False
    norm = _fund_service()._normalize_match_text(compact)
    header_terms = (
        "TOPLAM",
        "FPD",
        "FTD",
        "GORE",
        "GRUP",
        "ISIN KODU",
        "MENKUL KIYMET",
        "REPO TEMINAT",
        "DOVIZ",
        "BORSA",
        "SOZLESM",
    )
    if "PORTFOY" in norm and "FON" in norm and any(term in norm for term in header_terms):
        return True
    first_raw = compact.split()[0].strip(":-,;()") if compact.split() else ""
    if "-" in first_raw:
        first_left = _fund_service().normalize_fund_code(first_raw.split("-", 1)[0]).replace(".", "")
        if _kap_looks_like_fund_symbol(first_left):
            return False
    raw_tokens = compact.split()
    if len(raw_tokens) > 2 and raw_tokens[1] == "-":
        first_left = _fund_service().normalize_fund_code(raw_tokens[0].strip(":-,;()")).replace(".", "")
        if _kap_looks_like_fund_symbol(first_left):
            return False
    tokens = [_fund_service().normalize_fund_code(token.strip(":-,;()")).replace(".", "") for token in compact.split()]
    tokens = [token for token in tokens if token]
    if any(_seam("_kap_is_stock_symbol")(token) for token in tokens[:4]):
        return False
    return True


def _kap_buffer_starts_with_fund_symbol(buffer: List[str], context: str) -> bool:
    if "FON" not in _fund_service()._normalize_match_text(context) and "BYF" not in _fund_service()._normalize_match_text(context):
        return False
    first_line = str(buffer[0] if buffer else "").strip()
    first = first_line.split()[0].strip(":-,;()") if first_line.split() else ""
    candidate = first.split("-", 1)[0] if "-" in first else first
    candidate = _fund_service().normalize_fund_code(candidate).replace(".", "")
    return _kap_looks_like_fund_symbol(candidate)


def _kap_buffer_starts_with_foreign_symbol(buffer: List[str], context: str) -> bool:
    if _kap_foreign_prefixed_security(" ".join(buffer)):
        return True
    if not _kap_context_is_foreign(context):
        return False
    first_line = str(buffer[0] if buffer else "").strip()
    first = first_line.split()[0].strip(":-,;()") if first_line.split() else ""
    candidate = _fund_service().normalize_fund_code(first).replace(".", "")
    if len(candidate) < 2 or not _kap_looks_like_foreign_symbol(candidate):
        return False
    compact = " ".join(str(part or "").strip() for part in buffer if str(part or "").strip())
    return bool(compact and len(compact.split()) <= 4)


def _kap_buffer_starts_with_local_equity_symbol(buffer: List[str], context: str) -> bool:
    """Keep unknown local tickers while their multi-line company name unfolds."""
    context_norm = _fund_service()._normalize_match_text(context)
    if _kap_context_is_foreign(context_norm) or not any(
        token in context_norm for token in ("HISSE SENEDI", "HISSE SENETLERI", "HISSE")
    ):
        return False
    first_line = str(buffer[0] if buffer else "").strip()
    first = first_line.split()[0].strip(":-,;()") if first_line.split() else ""
    if not first or first != first.upper():
        return False
    return _kap_looks_like_local_equity_symbol(first)


def _kap_line_starts_known_local_equity_symbol(line: str) -> bool:
    """Identify a known ticker that starts a fresh local-equity row."""
    text = str(line or "").strip()
    first = text.split()[0].strip(" :-,;()") if text.split() else ""
    candidate = _fund_service().normalize_fund_code(first).replace(".", "")
    return bool(candidate and _seam("_kap_is_stock_symbol")(candidate))


def _parse_kap_holding_block(
    block: str,
    *,
    fund_code: str,
    category_context: str,
    report_date: Optional[str],
    source_url: Optional[str],
    continuation_marker: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    compact = " ".join(str(block or "").replace("\xa0", " ").split())
    norm = _fund_service()._normalize_match_text(compact)
    if not compact or re.match(r"^(?:ANA\s+GRUP|GRUP\s+TOPLAMI|TOPLAM)\b", norm) or "GRUP TOPLAMI" in norm:
        return None

    foreign_prefix = _kap_foreign_prefixed_security(compact)
    if foreign_prefix:
        code = str(foreign_prefix.get("code") or "").strip().upper()
        code_end = int(foreign_prefix.get("prefix_end") or 0)
    else:
        code_match = re.match(r"(?P<code>[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9._-]{1,15})\b", compact, flags=re.IGNORECASE)
        if not code_match:
            return None
        code = code_match.group("code").strip().upper()
        code_end = code_match.end()
        if "-" in code:
            left, right = code.split("-", 1)
            if _kap_looks_like_fund_symbol(left):
                code = left
                compact = f"{left} {right} {compact[code_match.end():]}".strip()
                code_match = re.match(r"(?P<code>[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9._-]{1,15})\b", compact, flags=re.IGNORECASE)
                if not code_match:
                    return None
                code_end = code_match.end()
    if code in _KAP_POSITION_STOPWORDS:
        return None

    number_tokens = _kap_number_tokens(compact)
    if len(number_tokens) < 3:
        return None

    first_number_start = int(number_tokens[0]["start"])
    raw_name = compact[code_end:first_number_start].strip(" -")
    raw_name = _KAP_DATE_PATTERN.sub(" ", raw_name)
    raw_name = re.sub(r"\s*/\s*", " ", raw_name)
    raw_name = re.sub(r"\s+-\s+", " ", raw_name)
    asset_name = " ".join(raw_name.split()) or code
    isin = _kap_extract_isin(compact)
    if isin:
        asset_name = " ".join(re.sub(rf"\b{re.escape(isin)}\b", " ", asset_name, flags=re.IGNORECASE).split()) or code
    weight_token, used_weight_fallback = _kap_extract_weight_candidate(compact, number_tokens)
    weight = float(weight_token["value"]) if weight_token else None
    numbers = [float(token["value"]) for token in number_tokens]
    if weight is None:
        return None

    amount = numbers[0] if numbers else None
    weight_start = int(weight_token.get("start")) if isinstance(weight_token, dict) and weight_token.get("start") is not None else None
    values_before_weight = [
        float(token["value"])
        for token in number_tokens
        if weight_start is None or int(token["start"]) < weight_start
    ]
    market_value = next((value for value in reversed(values_before_weight) if abs(value) > 100), None)
    asset_type = _kap_asset_type_from_context(category_context, code, asset_name)
    is_foreign = asset_type.startswith("foreign_")
    provider_symbol = (
        _seam("_kap_foreign_provider_symbol")(code, isin, asset_name, foreign_prefix.get("exchange") if foreign_prefix else None)
        if is_foreign
        else None
    )
    if is_foreign and isin and isin in _KAP_FOREIGN_ISIN_PROVIDER_NAMES:
        asset_name = _KAP_FOREIGN_ISIN_PROVIDER_NAMES[isin]
    return {
        "fund_code": fund_code,
        "asset_code": code,
        "asset_name": asset_name,
        "asset_type": asset_type,
        "asset_region": "foreign" if is_foreign else "TR",
        "isin": isin,
        "provider_symbol": provider_symbol,
        "logo_symbol": provider_symbol or code,
        "detail_clickable": False if is_foreign else None,
        "weight": round(float(weight), 6),
        "weight_quality": "fallback" if used_weight_fallback else "ok",
        "weight_warning": (
            "KAP satırındaki son FTD yüzdesi okunamadı; aynı satırdaki geçerli yüzde yaklaşık değer olarak kullanıldı."
            if used_weight_fallback
            else None
        ),
        "previous_weight": None,
        "weight_change": None,
        "change_status": "unchanged",
        "amount": amount,
        "market_value": market_value,
        "price": None,
        "report_date": report_date,
        "previous_report_date": None,
        "source_report_url": source_url,
        "source_type": "kap_pdf",
        "parse_confidence": 0.82,
        "continuation_marker": continuation_marker,
    }


def _kap_aggregation_key(position: Dict[str, Any]) -> str:
    code = _fund_service().normalize_fund_code(position.get("asset_code") or position.get("asset_name")).replace(".", "")
    asset_type = str(position.get("asset_type") or "").strip().lower()
    isin = str(position.get("isin") or "").strip().upper()
    return "|".join(part for part in (asset_type, code, isin) if part)


def _kap_should_sum_duplicate_lots(position: Dict[str, Any]) -> bool:
    asset_type = str(position.get("asset_type") or "").strip().lower()
    return asset_type in _KAP_INCLUDED_HOLDING_TYPES and not position.get("continuation_marker")


def _kap_sum_optional_number(left: Any, right: Any, *, digits: int) -> Optional[float]:
    left_number = _fund_service()._coerce_float(left)
    right_number = _fund_service()._coerce_float(right)
    if left_number is None and right_number is None:
        return None
    return round(float(left_number or 0.0) + float(right_number or 0.0), digits)


def _kap_deduplicate_positions(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_by_key: Dict[str, Dict[str, Any]] = {}
    for position in positions:
        key = _kap_aggregation_key(position)
        if not key:
            continue
        existing = merged_by_key.get(key)
        if not existing:
            merged_by_key[key] = dict(position)
            continue
        if _kap_should_sum_duplicate_lots(existing) and _kap_should_sum_duplicate_lots(position):
            existing["weight"] = _kap_sum_optional_number(existing.get("weight"), position.get("weight"), digits=6)
            existing["amount"] = _kap_sum_optional_number(existing.get("amount"), position.get("amount"), digits=6)
            existing["market_value"] = _kap_sum_optional_number(existing.get("market_value"), position.get("market_value"), digits=2)
            if str(existing.get("weight_quality") or "ok") != "ok" or str(position.get("weight_quality") or "ok") != "ok":
                existing["weight_quality"] = "fallback"
                existing["weight_warning"] = (
                    existing.get("weight_warning")
                    or position.get("weight_warning")
                    or "KAP satırındaki ağırlık yaklaşık değer olarak kullanıldı."
                )
            existing["parse_confidence"] = min(
                float(existing.get("parse_confidence") or 0.0),
                float(position.get("parse_confidence") or 0.0),
            )
            continue
        if abs(float(position.get("weight") or 0)) > abs(float(existing.get("weight") or 0)):
            merged_by_key[key] = dict(position)
    return sorted(merged_by_key.values(), key=lambda row: abs(float(row.get("weight") or 0)), reverse=True)


def _parse_kap_column_holding_row(
    line: str,
    *,
    fund_code: str,
    category_context: str,
    report_date: Optional[str],
    source_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Read ISIN / nominal / market value / group % / total % tables."""
    isin = _kap_extract_isin(line)
    if not isin:
        return None
    prefix, _isin, suffix = line.upper().partition(isin)
    tokens = _kap_number_tokens(suffix)
    # Require all four columns, so an unreadable weight cannot be replaced
    # with the group percentage or spill into the next security's row.
    if len(tokens) != 4 or _KAP_NUMBER_PATTERN.sub("", suffix).strip():
        return None
    decimal_match = re.search(r"([.,])\d{1,2}\s*%?$", tokens[-1]["text"])
    decimal = decimal_match.group(1) if decimal_match else None

    def number(token: Dict[str, Any]) -> Optional[float]:
        raw = token["text"].replace("%", "").strip()
        if decimal == ".":
            raw = raw.replace(",", "")
        elif decimal == ",":
            raw = raw.replace(".", "").replace(",", ".")
        else:
            return _coerce_kap_number_text(raw)
        try:
            value = float(raw)
            return value if math.isfinite(value) else None
        except ValueError:
            return None

    values = [number(token) for token in tokens]
    if any(value is None for value in values):
        return None
    amount, market_value, _group_weight, weight = values
    if not _kap_weight_candidate_is_valid(tokens[-1]["text"], weight):
        return None
    position = _seam("_parse_kap_holding_block")(
        f"{prefix.strip()} {amount} {market_value} {weight}% {isin}",
        fund_code=fund_code,
        category_context=category_context,
        report_date=report_date,
        source_url=source_url,
    )
    if position:
        position.update(amount=amount, market_value=market_value)
    return position


def _kap_name_first_ticker_identity(raw_ticker: Any) -> Tuple[Optional[str], Optional[str]]:
    """Split a KAP trailing ticker into a display code and provider symbol.

    The name-first KAP layout glues a Bloomberg-like ticker to the ISIN
    (``...2.90US0378331005AAPL.O``).  US exchange suffixes are not part of
    the Yahoo symbol, while country suffixes such as ``.L`` must stay.
    """
    compact = re.sub(r"\s+", "", str(raw_ticker or "")).strip().upper().strip(".,;:")
    if not compact or not re.fullmatch(r"[A-Z][A-Z0-9]{0,11}(?:\.[A-Z]{1,3})?", compact):
        return None, None
    if "." not in compact:
        return compact, compact
    base, suffix = compact.rsplit(".", 1)
    if suffix in _KAP_NAME_FIRST_US_TICKER_SUFFIXES:
        return base, base
    return base, compact


def _kap_name_first_row_parts(line: str) -> Optional[Dict[str, Any]]:
    """Read one ``name / nominal / market value / weight+ISIN`` row."""
    text = " ".join(str(line or "").replace("\xa0", " ").split())
    if not text or "TOPLAM" in _fund_service()._normalize_match_text(text):
        return None
    tokens = _kap_number_tokens(text)
    if len(tokens) < 3:
        return None
    weight_token = tokens[-1]
    weight = _fund_service()._coerce_float(weight_token.get("value"))
    if weight is None or not _kap_weight_candidate_is_valid(weight_token.get("text"), weight):
        return None
    weight_end = int(weight_token.get("end") or 0)
    if weight_end <= 0 or weight_end >= len(text):
        return None
    remainder = re.sub(r"\s+", "", text[weight_end:].strip()).upper()
    remainder = remainder.lstrip("%")
    if not remainder:
        return None

    amount_token = tokens[-3]
    market_value_token = tokens[-2]
    amount = _fund_service()._coerce_float(amount_token.get("value"))
    market_value = _fund_service()._coerce_float(market_value_token.get("value"))
    name_end = int(amount_token.get("start") or 0)
    name = text[:name_end].strip(" -:;")
    name = _KAP_DATE_PATTERN.sub(" ", name)
    name = " ".join(name.split()).strip(" -:;")
    if not name or not amount or market_value is None:
        return None

    isin: Optional[str] = None
    provider_symbol: Optional[str] = None
    isin_match = _KAP_ISIN_PATTERN.match(remainder)
    if isin_match:
        isin = isin_match.group(0)
        code, provider_symbol = _kap_name_first_ticker_identity(remainder[isin_match.end():])
        if not code:
            code = isin
    else:
        if not re.fullmatch(r"[A-Z0-9]{2,10}", remainder):
            return None
        code = _fund_service().normalize_fund_code(remainder).replace(".", "")
        if not code or code in _KAP_POSITION_STOPWORDS:
            return None
        if not (_kap_looks_like_fund_symbol(code) or _kap_looks_like_equity_symbol(code)):
            return None
    if not code or code in _KAP_POSITION_STOPWORDS:
        return None
    return {
        "code": code,
        "name": name,
        "isin": isin,
        "provider_symbol": provider_symbol,
        "amount": amount,
        "market_value": market_value,
        "weight": round(float(weight), 6),
    }


def _kap_name_first_position(
    parts: Dict[str, Any],
    *,
    fund_code: str,
    category: str,
    report_date: Optional[str],
    source_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    code = str(parts.get("code") or "").strip().upper()
    name = str(parts.get("name") or "").strip() or code
    asset_type = _kap_asset_type_from_context(category, code, name)
    if not code or not _kap_is_included_holding_type(asset_type):
        return None
    is_foreign = asset_type.startswith("foreign_")
    provider_symbol = str(parts.get("provider_symbol") or "").strip() or None
    return {
        "fund_code": fund_code,
        "asset_code": code,
        "asset_name": name,
        "asset_type": asset_type,
        "asset_region": "foreign" if is_foreign else "TR",
        "isin": parts.get("isin"),
        "provider_symbol": provider_symbol,
        "logo_symbol": provider_symbol or code,
        "detail_clickable": False if is_foreign else True,
        "weight": parts.get("weight"),
        "weight_quality": "ok",
        "weight_warning": None,
        "previous_weight": None,
        "weight_change": None,
        "change_status": "unchanged",
        "amount": parts.get("amount"),
        "market_value": parts.get("market_value"),
        "price": None,
        "report_date": report_date,
        "previous_report_date": None,
        "source_report_url": source_url,
        "source_type": "kap_pdf",
        "parse_confidence": 0.86,
    }


def _kap_name_first_category_is_continuation(line: str, parts: List[str]) -> bool:
    if len(parts) >= 3:
        return False
    text = " ".join(str(line or "").split())
    if not text or len(text) > 45 or _KAP_NAME_FIRST_PAGE_FOOTER_PATTERN.match(text):
        return False
    if _kap_number_tokens(text):
        return False
    words = re.findall(r"[A-Z]+", _fund_service()._normalize_match_text(text))
    if not words or len(words) > 4:
        return False
    return all(word in _KAP_NAME_FIRST_CATEGORY_WORDS for word in words)


def _kap_name_first_resolve_category(parts: List[str]) -> str:
    joined = " ".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not joined:
        return ""
    return _kap_category_marker(joined) or ""


def _kap_name_first_is_boilerplate(line: str) -> bool:
    text = " ".join(str(line or "").split())
    if not text or _KAP_NAME_FIRST_PAGE_FOOTER_PATTERN.match(text):
        return True
    norm = _fund_service()._normalize_match_text(text)
    if re.fullmatch(r"(?:BOLD\s*)+", norm):
        return True
    if "ISIN KODU" in norm and len(norm) <= 40:
        return True
    if "RAYIC DEGERI" in norm and "NOMINAL" in norm:
        return True
    return norm in {"TABLOSU", "FON PORTFOY DEGERI"}


def _kap_name_first_line_can_be_name(line: str) -> bool:
    text = " ".join(str(line or "").split())
    if not text or len(text) > 160 or _kap_name_first_is_boilerplate(text):
        return False
    if _kap_number_tokens(text):
        return False
    if not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]", text):
        return False
    norm = _fund_service()._normalize_match_text(text)
    if any(token in norm for token in ("TOPLAM", "SAYFA", "ISIN KODU", "NOMINAL DEGERI", "RAYIC DEGERI")):
        return False
    if re.match(r"^(?:IV|VI|VII|VIII|IX|V)\s*[-–]\s*", norm):
        return False
    return True


def _kap_is_name_first_holdings_table(text: str) -> bool:
    normalized = _fund_service()._normalize_match_text(text)
    if not (re.search(r"\bRAYIC\s+DEGERI\b", normalized) and re.search(r"\bISIN\s+KODU\b", normalized)):
        return False
    category_prefixes = 0
    for line in str(text or "").splitlines():
        if _KAP_NAME_FIRST_CATEGORY_PATTERN.match(line):
            category_prefixes += 1
        tokens = _kap_number_tokens(line)
        if len(tokens) < 3:
            continue
        remainder = re.sub(r"\s+", "", line[int(tokens[-1]["end"]):]).upper()
        if _KAP_ISIN_PATTERN.match(remainder):
            return True
    return category_prefixes >= 2


def _parse_kap_name_first_holdings_pdf_text(
    text: str,
    *,
    fund_code: str,
    report_date: Optional[str],
    source_url: Optional[str],
) -> List[Dict[str, Any]]:
    """Parse KAP PDFs that put the security name in the first column.

    Rows end with nominal value, market value and a percentage glued to the
    ISIN or fund code.  Category headers and security names may be split over
    several lines, so both are buffered until the next complete row.
    """
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    positions: List[Dict[str, Any]] = []
    pending_names: List[str] = []
    category_parts: List[str] = []
    category = ""
    category_open = False

    for line in lines:
        norm = _fund_service()._normalize_match_text(line)
        if category_open:
            if _kap_name_first_category_is_continuation(line, category_parts):
                category_parts.append(line)
                continue
            category = _kap_name_first_resolve_category(category_parts)
            category_parts = []
            category_open = False

        if re.match(r"^(?:IV|VI|VII|VIII|IX|V)\s*[-–]\s*", norm):
            break
        if _kap_name_first_is_boilerplate(line):
            pending_names = []
            continue
        category_match = _KAP_NAME_FIRST_CATEGORY_PATTERN.match(line)
        if category_match:
            category_parts = [category_match.group("label")]
            category_open = True
            category = ""
            pending_names = []
            continue
        if "TOPLAM" in norm:
            pending_names = []
            continue
        if not category:
            pending_names = []
            continue

        parts = _kap_name_first_row_parts(" ".join([*pending_names, line]) if pending_names else line)
        if parts:
            position = _kap_name_first_position(
                parts,
                fund_code=fund_code,
                category=category,
                report_date=report_date,
                source_url=source_url,
            )
            if position:
                positions.append(position)
            pending_names = []
            continue
        if _kap_name_first_line_can_be_name(line):
            pending_names.append(line)
            if len(pending_names) > 8:
                pending_names = pending_names[-8:]
        else:
            pending_names = []

    return _kap_deduplicate_positions(positions)


def _parse_kap_nominal_value_table(
    text: str,
    *,
    fund_code: str,
    report_date: Optional[str],
    source_url: Optional[str],
) -> List[Dict[str, Any]]:
    """Read numbered nominal / market value / ratio / optional cost tables."""
    positions: List[Dict[str, Any]] = []
    category = ""
    numeric_columns = 0
    currencies = {
        "TL", "TRY", "USD", "EUR", "GBP", "CHF", "CAD", "AUD", "JPY",
        "CNY", "HKD", "SGD", "AED", "SAR", "QAR", "KWD", "BHD", "OMR",
        "IDR", "MYR", "INR", "ZAR", "KRW",
    }
    for line in text.splitlines():
        line = line.strip()
        norm = _fund_service()._normalize_match_text(line)
        if numeric_columns and re.match(r"^(?:[4-9]|IV|V|VI|VII|VIII|IX)\s*[-–]\s*", norm):
            break
        header = re.search(r"\bNOMINAL\s+RAYIC\s+DEGER\s+ORAN\b", norm)
        if header:
            numeric_columns = 4 if "BIRIM ALIS FIYATI" in norm else 3
            category_text = norm[:header.start()].split("IHRAC", 1)[0].strip()
            if "KATILMA PAYLARI" in category_text:
                category = "Yatırım Fonu/BYF"
            else:
                category = _kap_category_marker(category_text) or "Diğer Varlıklar"
            continue
        if not numeric_columns:
            continue
        if re.match(r"^\d+\s*\.\s*", norm) and not _kap_number_matches(line)[1:]:
            category = _kap_category_marker(norm) or category
            continue
        if not _kap_is_included_holding_context(category):
            continue
        tokens = _kap_number_tokens(line)
        if len(tokens) < numeric_columns:
            continue
        values = tokens[-numeric_columns:]
        suffix = line[int(values[0]["start"]):]
        if _KAP_NUMBER_PATTERN.sub("", suffix).strip():
            continue
        prefix = re.sub(r"^\d+\s+", "", line[:int(values[0]["start"])].strip())
        fields = re.split(r"\s{2,}|\t+", prefix)
        label = fields[0].split()
        if not label:
            continue
        code = label[0].upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{1,15}", code) or code in _KAP_POSITION_STOPWORDS:
            continue
        currency_index = next((index for index in (1, 2) if index < len(label) and label[index] in currencies), None)
        currency = label[currency_index] if currency_index else None
        exchange = label[1] if currency_index == 2 else None
        name_parts = fields[1:] if len(fields) > 1 else label[(currency_index or 0) + 1:]
        name = " ".join(name_parts).strip() or code
        ratio_text = str(values[2]["text"]).replace("%", "").strip()
        decimal_match = re.search(r"([.,])\d+$", ratio_text)
        decimal = decimal_match.group(1) if decimal_match else None

        def number(token: Dict[str, Any]) -> float:
            raw = str(token["text"]).replace("%", "").strip()
            if decimal:
                raw = raw.replace("." if decimal == "," else ",", "").replace(decimal, ".")
            return float(raw) if decimal else float(token["value"])

        amount, market_value, weight = (number(token) for token in values[:3])
        if abs(weight) > _KAP_MAX_REASONABLE_WEIGHT:
            continue
        position = _seam("_parse_kap_holding_block")(
            f"{code} {name} {amount:.6f} {market_value:.6f} {weight:.6f}% {weight:.6f}%",
            fund_code=fund_code,
            category_context=category,
            report_date=report_date,
            source_url=source_url,
        )
        if not position or not _kap_is_included_holding_type(position.get("asset_type")):
            continue
        position.update(
            asset_name=name, amount=amount, market_value=market_value,
            weight=round(weight, 6), report_currency=currency,
        )
        if _kap_context_is_foreign(category):
            position.update(
                asset_type="foreign_fund" if "FON" in _fund_service()._normalize_match_text(category) else "foreign_equity",
                asset_region="foreign", detail_clickable=False,
            )
        if position["asset_type"].startswith("foreign_") and currency not in {None, "USD"}:
            # Report labels in other markets are not automatically US tickers
            # (e.g. a Saudi ticker can also belong to an unrelated US issuer).
            provider_symbol = _seam("_kap_foreign_provider_symbol_for_exchange")(code, exchange)
            position.update(
                provider_symbol=provider_symbol, logo_symbol=provider_symbol,
                provider_symbol_status="resolved" if provider_symbol else "unresolved",
            )
        positions.append(position)
    return _kap_deduplicate_positions(positions)


def _parse_kap_holdings_pdf_text(
    text: str,
    *,
    fund_code: str,
    report_date: Optional[str],
    source_url: Optional[str],
) -> List[Dict[str, Any]]:
    raw_text = str(text or "")
    start_match = re.search(r"(?:III\s*[-–]\s*)?FON\s*PORTF[ÖO]Y\s*[DC]E[ĞG]ER[İI]", raw_text, flags=re.IGNORECASE)
    if start_match:
        raw_text = raw_text[start_match.start():]
    if re.search(r"\bNOMINAL\s+RAYIC\s+DEGER\s+ORAN\b", _fund_service()._normalize_match_text(raw_text)):
        return _seam("_parse_kap_nominal_value_table")(
            raw_text, fund_code=fund_code, report_date=report_date, source_url=source_url,
        )
    if _kap_is_name_first_holdings_table(raw_text):
        parsed_name_first = _seam("_parse_kap_name_first_holdings_pdf_text")(
            raw_text, fund_code=fund_code, report_date=report_date, source_url=source_url,
        )
        if parsed_name_first:
            return parsed_name_first
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    positions: List[Dict[str, Any]] = []
    current_category = ""
    buffer: List[str] = []
    buffer_category = ""
    buffer_continuation_marker: Optional[str] = None
    skip_auxiliary_section = False
    auxiliary_seen_portfolio_header = False
    auxiliary_requires_portfolio_header = False
    column_table = False

    def flush() -> None:
        nonlocal buffer, buffer_category, buffer_continuation_marker
        if not buffer:
            return
        parsed = _seam("_parse_kap_holding_block")(
            " ".join(buffer),
            fund_code=fund_code,
            category_context=buffer_category,
            report_date=report_date,
            source_url=source_url,
            continuation_marker=buffer_continuation_marker,
        )
        if parsed and _kap_is_included_holding_type(parsed.get("asset_type")):
            positions.append(parsed)
        buffer = []
        buffer_category = ""
        buffer_continuation_marker = None

    for line_index, line in enumerate(lines):
        original_line = line
        norm = _fund_service()._normalize_match_text(line)
        line_continuation_marker: Optional[str] = None
        header_compact = re.sub(r"\s+", "", norm)
        if all(term in header_compact for term in ("ISIN", "NOMINAL", "TOPLAM", "GRUP")):
            flush()
            column_table = True
            continue
        section_match = re.match(r"^(?P<section>IV|VI|VII|VIII|IX|V)\s*[-–]\s*", norm)
        if section_match:
            flush()
            section = section_match.group("section")
            if section in {"IV", "V"}:
                skip_auxiliary_section = True
                auxiliary_seen_portfolio_header = False
                continue
            if section == "VI":
                skip_auxiliary_section = True
                auxiliary_seen_portfolio_header = False
                auxiliary_requires_portfolio_header = True
                continue
            break
        if skip_auxiliary_section:
            if "ISIN KODU" in norm or ("FON PORTFOY DEGERI" in norm and "TABLOSU" in norm):
                auxiliary_seen_portfolio_header = True
                continue
            preview_line, preview_inline_marker = _kap_strip_inline_category(line)
            preview_marker = _kap_with_foreign_context(_kap_category_marker(line), current_category)
            if preview_inline_marker:
                preview_marker = _kap_with_foreign_context(preview_inline_marker, current_category) or preview_marker
            # Expense/summary labels can resemble categories (e.g. "mevzuat"
            # or "diğer varlıklar"). They must not replace the portfolio's
            # category before the holdings table continues on the next page.
            can_resume = (preview_marker and _kap_is_included_holding_context(preview_marker)) or (
                auxiliary_seen_portfolio_header and _kap_line_starts_position(preview_line or line)
            )
            if auxiliary_requires_portfolio_header and not auxiliary_seen_portfolio_header:
                continue
            if can_resume:
                skip_auxiliary_section = False
                auxiliary_requires_portfolio_header = False
            else:
                continue
        if current_category and "FON PORTFOY DEGERI" in norm and "TABLOSU" not in norm:
            if _kap_number_matches(line):
                flush()
                break
            continue
        # Strip ``Tem.Ver.`` / ``Teminat Veren`` style collateral lender
        # prefixes so the parser sees the real security code as the
        # leading token instead of the prefix word.
        stripped_line, _continuation_marker = _kap_strip_continuation_prefix(line)
        if stripped_line != line:
            line = stripped_line
            line_continuation_marker = _continuation_marker
            if not line:
                continue
        first_token = line.split()[0].strip(":-,;()") if line.split() else ""
        if "-" in first_token and _fund_service().normalize_fund_code(first_token.split("-", 1)[0]) == _fund_service().normalize_fund_code(fund_code):
            buffer = []
            buffer_category = ""
            buffer_continuation_marker = None
            continue
        line, inline_marker = _kap_strip_inline_category(line)
        if (
            inline_marker
            and not line
            and buffer
            and not _kap_row_complete(" ".join(buffer))
            and _kap_buffer_starts_with_fund_symbol(buffer, buffer_category)
        ):
            line = original_line
            inline_marker = None
        if inline_marker and buffer and not _kap_row_complete(" ".join(buffer)):
            buffer = []
            buffer_category = ""
            buffer_continuation_marker = None
        if inline_marker:
            if buffer:
                flush()
            current_category = _kap_with_foreign_context(inline_marker, current_category) or inline_marker
            if not line:
                continue
        norm = _fund_service()._normalize_match_text(line)
        marker = _kap_with_foreign_context(_kap_category_marker(line), current_category)
        if marker and len(_kap_number_matches(line)) < 2 and buffer and not _kap_row_complete(" ".join(buffer)):
            if (
                _kap_buffer_is_header_noise(buffer)
                and not _kap_buffer_starts_with_fund_symbol(buffer, buffer_category)
                and not _kap_buffer_starts_with_foreign_symbol(buffer, buffer_category)
                and not _kap_buffer_starts_with_local_equity_symbol(buffer, buffer_category)
            ):
                buffer = []
                buffer_category = ""
                buffer_continuation_marker = None
            else:
                marker = None
        if marker and len(_kap_number_matches(line)) < 2 and (not buffer or _kap_row_complete(" ".join(buffer))):
            if buffer:
                flush()
            current_category = marker
            continue
        if not current_category:
            continue
        if column_table:
            parsed = _seam("_parse_kap_column_holding_row")(
                line,
                fund_code=fund_code,
                category_context=current_category,
                report_date=report_date,
                source_url=source_url,
            )
            if parsed and _kap_is_included_holding_type(parsed.get("asset_type")):
                positions.append(parsed)
            continue
        if "GRUP TOPLAMI" in norm:
            flush()
            continue
        if buffer and _kap_foreign_prefixed_security(line):
            # A ticker followed by an exchange starts a new security, even
            # when the preceding row was truncated or had a long issuer name.
            flush()
        if buffer and _kap_line_starts_position(line) and not _kap_row_complete(" ".join(buffer)):
            if (
                _kap_buffer_is_header_noise(buffer)
                and not _kap_buffer_starts_with_fund_symbol(buffer, buffer_category)
                and not _kap_buffer_starts_with_foreign_symbol(buffer, buffer_category)
                and (
                    not _kap_buffer_starts_with_local_equity_symbol(buffer, buffer_category)
                    or _kap_line_starts_known_local_equity_symbol(line)
                )
            ):
                buffer = []
                buffer_category = ""
                buffer_continuation_marker = None
        if not buffer and not _kap_line_starts_position(line):
            continue
        if buffer and _kap_line_starts_position(line) and _kap_row_complete(" ".join(buffer)):
            flush()
        if not buffer:
            buffer_category = current_category
            buffer_continuation_marker = line_continuation_marker
        buffer.append(line)
        if _kap_row_complete(" ".join(buffer)):
            next_line = lines[line_index + 1] if line_index + 1 < len(lines) else ""
            if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", next_line, flags=re.IGNORECASE):
                flush()
    flush()

    return _kap_deduplicate_positions(positions)


def _position_key(position: Dict[str, Any]) -> str:
    isin = str(position.get("isin") or "").strip().upper()
    if (
        str(position.get("asset_type") or "").startswith("foreign_")
        and re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", isin)
    ):
        return f"isin:{isin}"
    return _fund_service().normalize_fund_code(position.get("asset_code") or position.get("asset_name")).replace(".", "")


def _kap_resolve_fund_code_ocr_variant(code: str, fund_names: Dict[str, str]) -> str:
    normalized = _fund_service().normalize_fund_code(code).replace(".", "")
    if not normalized or fund_names.get(normalized):
        return normalized
    for index, char in enumerate(normalized):
        if char != "G":
            continue
        candidate = f"{normalized[:index]}L{normalized[index + 1:]}"
        if fund_names.get(candidate):
            return candidate
    return normalized


def _normalize_holding_positions_for_response(
    processed_dir: Path,
    positions: List[Dict[str, Any]],
    *,
    fund_code: str,
) -> List[Dict[str, Any]]:
    fund_names = _seam("_kap_fund_name_map")(processed_dir)
    current_fund_code = _fund_service().normalize_fund_code(fund_code).replace(".", "")
    candidates: List[Tuple[Dict[str, Any], str, str]] = []
    stock_codes: List[str] = []
    fund_codes: List[str] = []
    for position in positions:
        code = _fund_service().normalize_fund_code(position.get("asset_code")).replace(".", "")
        if not code or code in _KAP_POSITION_STOPWORDS:
            continue
        if "-" in code:
            left = code.split("-", 1)[0]
            if fund_names.get(left) or _kap_looks_like_fund_symbol(left):
                code = left
        if code == current_fund_code:
            continue
        row_type = str(position.get("asset_type") or "").strip().lower()
        if row_type == "local_equity" and _kap_holding_name_is_header_noise(position.get("asset_name")):
            continue
        if row_type == "fund":
            code = _seam("_kap_resolve_fund_code_ocr_variant")(code, fund_names)
            if code == current_fund_code:
                continue
        if row_type in {"foreign_equity", "foreign_fund"}:
            candidates.append((position, code, row_type))
            continue
        if _seam("_kap_is_stock_symbol")(code) or (row_type == "local_equity" and _kap_looks_like_local_equity_symbol(code)):
            candidates.append((position, code, "local_equity"))
            stock_codes.append(code)
            continue
        name_norm = _fund_service()._normalize_match_text(position.get("asset_name") or "")
        looks_like_named_fund = any(token in name_norm for token in ("YATIRIM FONU", "BORSA YATIRIM FONU", " FONU", "FON ", "BYF"))
        if not _kap_looks_like_fund_symbol(code) or not (row_type == "fund" or looks_like_named_fund or fund_names.get(code)):
            continue
        candidates.append((position, code, "fund"))
        fund_codes.append(code)

    stock_instruments = _fund_service().get_instruments(processed_dir, "stock", stock_codes)
    fund_instruments = _fund_service().get_instruments(processed_dir, "fund", fund_codes)
    normalized_rows: List[Dict[str, Any]] = []
    for position, code, asset_type in candidates:
        row = dict(position)
        row["asset_code"] = code
        row.pop("continuation_marker", None)
        if asset_type == "local_equity":
            row["asset_type"] = "local_equity"
            row["asset_region"] = "TR"
            row["detail_clickable"] = True
            instrument = stock_instruments.get(code)
            row["asset_name"] = (
                str((instrument or {}).get("name") or "").strip()
                or _seam("_kap_stock_name_from_cache")(processed_dir, code)
                or str(row.get("asset_name") or code).strip()
                or code
            )
            if instrument and instrument.get("logo_url"):
                row["logo_url"] = instrument.get("logo_url")
                row["logo_source"] = instrument.get("logo_source")
        elif asset_type == "fund":
            row["asset_type"] = "fund"
            row["asset_region"] = "TR"
            row["detail_clickable"] = True
            instrument = fund_instruments.get(code)
            instrument_meta = (instrument or {}).get("metadata") if isinstance((instrument or {}).get("metadata"), dict) else {}
            provider_name = (
                str(instrument_meta.get("founder_company") or instrument_meta.get("manager_company") or "").strip()
                if isinstance(instrument_meta, dict)
                else ""
            )
            if provider_name:
                row["provider_name"] = provider_name
                row["logo_symbol"] = row.get("logo_symbol") or provider_name
            if instrument and instrument.get("logo_url"):
                row["logo_url"] = instrument.get("logo_url")
                row["logo_source"] = instrument.get("logo_source")
            row["asset_name"] = (
                str((instrument or {}).get("name") or "").strip()
                or fund_names.get(code)
                or str(row.get("asset_name") or code).strip()
                or code
            )
        else:
            row["asset_type"] = asset_type
            row["asset_region"] = "foreign"
            row["detail_clickable"] = False
            row["tefas_tradable"] = False if asset_type == "foreign_fund" else None
            provider_symbol = (
                None if row.get("provider_symbol_status") == "unresolved"
                else row.get("provider_symbol") or _seam("_kap_foreign_provider_symbol")(code, row.get("isin"), row.get("asset_name"))
            )
            row["provider_symbol"] = provider_symbol
            row["logo_symbol"] = row.get("logo_symbol") or provider_symbol or code
            row["asset_name"] = str(row.get("asset_name") or code).strip() or code
        normalized_rows.append(row)
    return normalized_rows


def _merge_holding_positions(
    latest_positions: List[Dict[str, Any]],
    previous_positions: List[Dict[str, Any]],
    *,
    latest_report_date: Optional[str],
    previous_report_date: Optional[str],
) -> List[Dict[str, Any]]:
    previous_by_key = {_seam("_position_key")(position): position for position in previous_positions if _seam("_position_key")(position)}
    latest_keys: set[str] = set()
    merged: List[Dict[str, Any]] = []
    for position in latest_positions:
        key = _seam("_position_key")(position)
        latest_keys.add(key)
        previous = previous_by_key.get(key)
        current_weight = _fund_service()._coerce_float(position.get("weight"))
        previous_weight = _fund_service()._coerce_float(previous.get("weight")) if previous else None
        delta = None
        status = "new" if previous is None else "unchanged"
        if current_weight is not None and previous_weight is not None:
            delta = round(current_weight - previous_weight, 6)
            if delta > 0.005:
                status = "increased"
            elif delta < -0.005:
                status = "decreased"
        elif current_weight is not None and previous is None:
            delta = current_weight
        merged_position = dict(position)
        merged_position["previous_weight"] = previous_weight
        merged_position["previous_weight_quality"] = (
            str(previous.get("weight_quality") or "ok") if previous else "missing"
        )
        merged_position["previous_weight_warning"] = (
            previous.get("weight_warning") if previous else None
        )
        merged_position["previous_report_date"] = previous_report_date
        merged_position["weight_change"] = delta
        merged_position["change_status"] = status
        merged.append(merged_position)

    for key, previous in previous_by_key.items():
        if key in latest_keys:
            continue
        previous_weight = _fund_service()._coerce_float(previous.get("weight"))
        removed = dict(previous)
        removed["weight"] = 0.0
        removed["previous_weight"] = previous_weight
        removed["previous_weight_quality"] = str(previous.get("weight_quality") or "ok")
        removed["previous_weight_warning"] = previous.get("weight_warning")
        removed["previous_report_date"] = previous_report_date
        removed["weight_change"] = round(-previous_weight, 6) if previous_weight is not None else None
        removed["change_status"] = "removed"
        removed["report_date"] = latest_report_date or previous.get("report_date")
        merged.append(removed)

    return sorted(
        merged,
        key=lambda row: max(abs(float(row.get("weight") or 0)), abs(float(row.get("previous_weight") or 0))),
        reverse=True,
    )


def _kap_report_positions(
    processed_dir: Path,
    detail: Dict[str, Any],
    *,
    fund_code: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    basic = _seam("_kap_report_basic")(detail)
    disclosure_index = basic.get("disclosureIndex")
    report_date = _seam("_kap_report_date")(detail)
    source_url = _seam("_kap_source_url")(disclosure_index)
    attachments = _seam("_kap_report_attachments")(detail)
    pdf_attachment = next(
        (
            attachment
            for attachment in attachments
            if str(attachment.get("fileExtension") or "").strip().lower() == "pdf"
            and str(attachment.get("objId") or "").strip()
        ),
        None,
    )
    metadata = {
        "disclosure_index": disclosure_index,
        "file_name": pdf_attachment.get("fileName") if pdf_attachment else None,
        "report_date": report_date,
        "source_url": source_url,
    }
    if not pdf_attachment:
        metadata["warning"] = "KAP report has no PDF attachment"
        return [], metadata
    text, attachment_cache_meta = _seam("_kap_attachment_text_from_cache")(processed_dir, disclosure_index, pdf_attachment)
    metadata.update(attachment_cache_meta)
    if text.startswith("KAP_PDF_OCR\n"):
        metadata["text_extraction_method"] = "ocr"
    positions = _seam("_parse_kap_holdings_pdf_text")(
        text,
        fund_code=fund_code,
        report_date=report_date,
        source_url=source_url,
    )
    if not positions:
        metadata["warning"] = (
            "KAP PDF text parsed but no position rows were detected"
            if text.strip()
            else "KAP PDF text extraction returned no readable text"
        )
    return positions, metadata


def _unavailable_holdings_payload(
    fund_code: str,
    *,
    message: str,
    warnings: Optional[List[str]] = None,
    cache_hit: bool = False,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    return {
        "fund_code": normalized,
        "status": "unavailable",
        "positions": [],
        "source": _seam("KAP_HOLDINGS_SOURCE"),
        "message": message,
        "source_metadata": {
            "source": _seam("KAP_HOLDINGS_SOURCE"),
            "source_url": _fund_service().KAP_BASE_URL,
            "fetched_at": _fund_service()._utc_now_iso(),
            "as_of": None,
            "cache_hit": cache_hit,
            "stale": True,
            "static_cache_hit": False,
            "cache_policy": "monthly_report",
            "parse_status": "unavailable",
            "parser_version": _seam("KAP_HOLDINGS_PARSE_VERSION"),
            "warnings": list(warnings or [message]),
        },
    }


def _parse_iso_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _holdings_cache_is_stale(cached: Dict[str, Any]) -> bool:
    cached_meta = cached.get("source_metadata", {}) if isinstance(cached.get("source_metadata"), dict) else {}
    cached_parser_version = _fund_service()._coerce_int(cached_meta.get("parser_version"))
    if cached_parser_version != _seam("KAP_HOLDINGS_PARSE_VERSION"):
        return True

    positions = list(cached.get("positions") or [])
    stored_hash = str(cached_meta.get("positions_hash") or "").strip()
    if stored_hash and stored_hash != _seam("_holdings_positions_hash")(positions):
        return True

    parse_status = str(cached_meta.get("parse_status") or cached.get("status") or "").strip().lower()
    if parse_status in {"unavailable", "partial"} or str(cached.get("status") or "").strip().lower() in {"unavailable", "partial"}:
        age = _fund_service()._cache_age_seconds(cached_meta.get("fetched_at"))
        if age is None:
            return True
        if age <= _fund_service().FUNDS_HOLDINGS_NEGATIVE_TTL_SECONDS:
            return False
        return parse_status != "ok" and not positions

    latest_report = cached_meta.get("latest_report")
    latest_report_date = None
    if isinstance(latest_report, dict):
        latest_report_date = _seam("_parse_iso_date")(latest_report.get("report_date"))
    if latest_report_date is None:
        latest_report_date = _seam("_parse_iso_date")(cached_meta.get("as_of"))

    # KAP portfolio distribution reports are monthly. If the cached report is
    # from the current or immediately previous report month, there is no older
    # PDF content to refetch repeatedly during the same month.
    if latest_report_date is not None:
        today = _fund_service()._utc_now().date()
        month_age = (today.year - latest_report_date.year) * 12 + today.month - latest_report_date.month
        if month_age <= 1:
            return False

    age = _fund_service()._cache_age_seconds(cached_meta.get("fetched_at"))
    return age is None or age > _fund_service().FUNDS_HOLDINGS_TTL_SECONDS


def refresh_fund_holdings(
    processed_dir: Path,
    fund_code: str,
    *,
    cached_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    metadata = _seam("_kap_search_fund_metadata")(normalized)
    if not metadata:
        payload = _seam("_unavailable_holdings_payload")(
            normalized,
            message="KAP fon kaydı bulunamadı.",
            warnings=[f"KAP fund search returned no exact match for {normalized}"],
        )
        payload["source_metadata"]["positions_hash"] = _seam("_holdings_positions_hash")(payload.get("positions") or [])
        _fund_service()._write_json(_fund_service()._holdings_path(processed_dir, normalized), payload)
        return payload
    fund_oid = str(metadata.get("fund_oid") or "").strip()
    subject_oid = _seam("_kap_portfolio_subject_oid")(fund_oid)
    disclosures = _seam("_kap_list_portfolio_disclosures")(fund_oid, subject_oid)
    if not disclosures:
        payload = _seam("_unavailable_holdings_payload")(
            normalized,
            message="KAP portföy dağılım bildirimi bulunamadı.",
            warnings=[f"KAP portfolio allocation disclosures not found for {normalized}"],
        )
        payload["source_metadata"]["fund_oid"] = fund_oid
        payload["source_metadata"]["subject_oid"] = subject_oid
        payload["source_metadata"]["disclosure_check"] = _seam("_holdings_disclosure_check_meta")([])
        _fund_service()._write_json(_fund_service()._holdings_path(processed_dir, normalized), payload)
        return payload

    if _seam("_cached_holdings_matches_disclosures")(cached_payload, disclosures):
        return _seam("_cached_holdings_with_disclosure_check")(processed_dir, normalized, dict(cached_payload or {}), disclosures)

    warnings: List[str] = []
    report_payloads: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    for row in disclosures[:2]:
        disclosure_index = _seam("_kap_disclosure_index")(row)
        if not disclosure_index:
            continue
        detail = _seam("_kap_fetch_report_detail")(disclosure_index)
        if not detail:
            warnings.append(f"KAP disclosure detail unavailable: {disclosure_index}")
            continue
        positions, report_meta = _seam("_kap_report_positions")(processed_dir, detail, fund_code=normalized)
        if report_meta.get("warning"):
            warnings.append(str(report_meta["warning"]))
        report_payloads.append((positions, report_meta))

    latest_positions = report_payloads[0][0] if report_payloads else []
    latest_meta = report_payloads[0][1] if report_payloads else {}
    previous_positions = report_payloads[1][0] if len(report_payloads) > 1 else []
    previous_meta = report_payloads[1][1] if len(report_payloads) > 1 else {}
    if not latest_positions:
        payload = _seam("_unavailable_holdings_payload")(
            normalized,
            message="KAP portföy PDF'i parse edilemedi.",
            warnings=warnings or ["KAP PDF position parser returned no rows"],
        )
        payload["status"] = "partial"
        payload["source_metadata"]["parse_status"] = "partial"
        payload["source_metadata"]["fund_oid"] = fund_oid
        payload["source_metadata"]["subject_oid"] = subject_oid
        payload["source_metadata"]["latest_report"] = latest_meta or None
        payload["source_metadata"]["previous_report"] = previous_meta or None
        payload["source_metadata"]["disclosure_check"] = _seam("_holdings_disclosure_check_meta")(disclosures)
        payload["source_metadata"]["positions_hash"] = _seam("_holdings_positions_hash")(payload.get("positions") or [])
        _fund_service()._write_json(_fund_service()._holdings_path(processed_dir, normalized), payload)
        return payload

    if not previous_positions:
        warnings.append("Önceki ay portföy raporu parse edilemedi veya bulunamadı.")
    positions = _seam("_merge_holding_positions")(
        latest_positions,
        previous_positions,
        latest_report_date=latest_meta.get("report_date"),
        previous_report_date=previous_meta.get("report_date"),
    )
    positions = _seam("_normalize_holding_positions_for_response")(processed_dir, positions, fund_code=normalized)
    parse_status = "ok" if latest_positions and previous_positions else "partial"
    payload = {
        "fund_code": normalized,
        "status": parse_status,
        "positions": positions,
        "source": _seam("KAP_HOLDINGS_SOURCE"),
        "message": None,
        "source_metadata": {
            "source": _seam("KAP_HOLDINGS_SOURCE"),
            "source_url": latest_meta.get("source_url") or _fund_service().KAP_BASE_URL,
            "fetched_at": _fund_service()._utc_now_iso(),
            "as_of": latest_meta.get("report_date"),
            "cache_hit": False,
            "stale": False,
            "static_cache_hit": False,
            "cache_policy": "monthly_report",
            "parse_status": parse_status,
            "parser_version": _seam("KAP_HOLDINGS_PARSE_VERSION"),
            "fund_oid": fund_oid,
            "subject_oid": subject_oid,
            "fund_name": metadata.get("fund_name"),
            "latest_report": latest_meta or None,
            "previous_report": previous_meta or None,
            "disclosure_check": _seam("_holdings_disclosure_check_meta")(disclosures),
            "positions_hash": _seam("_holdings_positions_hash")(positions),
            "warnings": warnings,
        },
    }
    _fund_service()._write_json(_fund_service()._holdings_path(processed_dir, normalized), payload)
    return payload


def get_fund_holdings_payload(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    path = _fund_service()._holdings_path(processed_dir, normalized)
    cached = _fund_service()._read_json(path)
    if cached:
        cached_meta = cached.get("source_metadata", {}) if isinstance(cached.get("source_metadata"), dict) else {}
        stale = _seam("_holdings_cache_is_stale")(cached)
        parse_status = str(cached_meta.get("parse_status") or cached.get("status") or "").strip().lower()
        negative_cache_fresh = parse_status in {"unavailable", "partial"} and not stale and not list(cached.get("positions") or [])
        if not stale and (negative_cache_fresh or not _seam("_holdings_disclosure_check_due")(cached_meta)):
            payload = dict(cached)
            meta = dict(cached_meta)
            if payload.get("positions"):
                meta["positions_hash"] = _seam("_holdings_positions_hash")(list(payload.get("positions") or []))
            meta = _seam("_holdings_meta_with_runtime_cache_flags")(
                meta,
                cache_hit=True,
                stale=False,
                static_cache_hit=True,
            )
            payload["source_metadata"] = meta
            return payload
    try:
        return _seam("refresh_fund_holdings")(processed_dir, normalized, cached_payload=cached)
    except Exception as exc:
        if cached:
            payload = dict(cached)
            meta = dict(payload.get("source_metadata") or {})
            warnings = list(meta.get("warnings") or [])
            warnings.append(f"KAP holdings refresh failed: {exc}")
            meta["warnings"] = warnings
            meta["cache_hit"] = True
            meta["stale"] = True
            payload["source_metadata"] = meta
            payload["status"] = "partial" if payload.get("positions") else "unavailable"
            return payload
        return _seam("_unavailable_holdings_payload")(
            normalized,
            message="KAP portföy içeriği alınamadı.",
            warnings=[f"KAP holdings refresh failed: {exc}"],
        )

__all__ = [
    "KAP_HOLDINGS_SOURCE",
    "KAP_HOLDINGS_PARSE_VERSION",
    "_KAP_NUMBER_PATTERN",
    "_KAP_DATE_PATTERN",
    "_KAP_ISIN_TAIL_PATTERN",
    "_KAP_BORSA_CONTRACT_TAIL_PATTERN",
    "_KAP_MAX_REASONABLE_WEIGHT",
    "_KAP_LINE_PREFIX_PATTERNS",
    "_KAP_NON_HOLDING_CONTEXT_TOKENS",
    "_KAP_POSITION_STOPWORDS",
    "_KAP_INCLUDED_HOLDING_TYPES",
    "_KAP_FOREIGN_ISIN_PROVIDER_SYMBOLS",
    "_KAP_FOREIGN_ISIN_PROVIDER_NAMES",
    "_KAP_FOREIGN_SYMBOL_STOPWORDS",
    "_KAP_FOREIGN_EXCHANGE_SUFFIXES",
    "_KAP_FOREIGN_EQUITY_PREFIX_PATTERN",
    "_KAP_ISIN_PATTERN",
    "_KAP_NAME_FIRST_CATEGORY_PATTERN",
    "_KAP_NAME_FIRST_PAGE_FOOTER_PATTERN",
    "_KAP_NAME_FIRST_CATEGORY_WORDS",
    "_KAP_NAME_FIRST_US_TICKER_SUFFIXES",
    "_KAP_EXTRA_STOCK_SYMBOLS",
    "_KAP_BIST_STOCK_SYMBOLS",
    "_KAP_TURKISH_MONTH_WORDS",
    "_kap_headers",
    "_kap_url",
    "_kap_request_get",
    "_kap_get_json",
    "_kap_post_json",
    "_kap_get_text",
    "_kap_get_bytes",
    "_kap_search_fund_metadata",
    "_kap_portfolio_subject_oid",
    "_kap_disclosure_basic",
    "_kap_disclosure_index",
    "_parse_kap_datetime",
    "_kap_list_portfolio_disclosures",
    "_kap_fetch_report_detail",
    "_kap_report_basic",
    "_kap_report_attachments",
    "_kap_download_attachment",
    "_extract_kap_pdf_text",
    "_month_end_iso",
    "_kap_report_date",
    "_holdings_report_index",
    "_holdings_positions_hash",
    "_holdings_meta_with_runtime_cache_flags",
    "_holdings_disclosure_check_due",
    "_holdings_disclosure_check_meta",
    "_cached_holdings_matches_disclosures",
    "_cached_holdings_with_disclosure_check",
    "_kap_attachment_text_from_cache",
    "_kap_source_url",
    "_kap_number_matches",
    "_coerce_kap_number_text",
    "_kap_number_tokens",
    "_kap_borsa_contract_tail_match",
    "_kap_token_is_inside",
    "_kap_weight_candidate_is_valid",
    "_kap_extract_weight_candidate",
    "_kap_extract_isin",
    "_kap_foreign_provider_symbol_for_exchange",
    "_kap_foreign_prefixed_security",
    "_kap_stock_symbol_set",
    "_kap_is_stock_symbol",
    "_kap_stock_name_from_cache",
    "_kap_fund_name_map",
    "_kap_looks_like_equity_symbol",
    "_kap_looks_like_local_equity_symbol",
    "_kap_holding_name_is_header_noise",
    "_kap_looks_like_fund_symbol",
    "_kap_looks_like_foreign_symbol",
    "_kap_is_included_holding_type",
    "_kap_is_included_holding_context",
    "_kap_context_is_foreign",
    "_kap_with_foreign_context",
    "_kap_asset_type_from_context",
    "_kap_foreign_provider_symbol",
    "_kap_category_marker",
    "_kap_strip_inline_category",
    "_kap_strip_continuation_prefix",
    "_kap_line_starts_position",
    "_kap_row_complete",
    "_kap_buffer_is_header_noise",
    "_kap_buffer_starts_with_fund_symbol",
    "_kap_buffer_starts_with_foreign_symbol",
    "_kap_buffer_starts_with_local_equity_symbol",
    "_kap_line_starts_known_local_equity_symbol",
    "_parse_kap_holding_block",
    "_kap_aggregation_key",
    "_kap_should_sum_duplicate_lots",
    "_kap_sum_optional_number",
    "_kap_deduplicate_positions",
    "_parse_kap_column_holding_row",
    "_kap_name_first_ticker_identity",
    "_kap_name_first_row_parts",
    "_kap_name_first_position",
    "_kap_name_first_category_is_continuation",
    "_kap_name_first_resolve_category",
    "_kap_name_first_is_boilerplate",
    "_kap_name_first_line_can_be_name",
    "_kap_is_name_first_holdings_table",
    "_parse_kap_name_first_holdings_pdf_text",
    "_parse_kap_nominal_value_table",
    "_parse_kap_holdings_pdf_text",
    "_position_key",
    "_kap_resolve_fund_code_ocr_variant",
    "_normalize_holding_positions_for_response",
    "_merge_holding_positions",
    "_kap_report_positions",
    "_unavailable_holdings_payload",
    "_parse_iso_date",
    "_holdings_cache_is_stale",
    "refresh_fund_holdings",
    "get_fund_holdings_payload",
]

_ORIGINAL_EXPORTS.update({name: globals()[name] for name in __all__})
