from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from src.config import KapConfig

KAP_BASE_URL = "https://www.kap.org.tr/tr/api"
MEMBER_FILTER_ENDPOINT = "member/filter"
LIST_COMPANY_EXCEL_MEMBERS_ENDPOINT = "financialTable/listCompanyExcelMembers"
ATTACHMENT_DETAIL_ENDPOINT = "notification/attachment-detail"
PDF_ENDPOINT = "BildirimPdf"
KAP_CACHE_SCHEMA_VERSION = 4

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
    # BIST-30 core aliases (ticker <-> common short name)
    "AKBANK": ["AKBNK", "AKBANK"],
    "AKBNK": ["AKBNK", "AKBANK"],
    "ASELSAN": ["ASELS", "ASELSAN"],
    "ASELS": ["ASELS", "ASELSAN"],
    "EKGYO": ["EKGYO", "EMLAK KONUT"],
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


def _cache_file_for_company(processed_dir: Path, company: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(company or "").strip().upper()) or "UNKNOWN"
    return processed_dir / "kap_cache" / f"{slug}.json"


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
    fetched_at_raw = str(payload.get("fetched_at", "")).strip()
    if not fetched_at_raw:
        return False
    try:
        fetched_at = datetime.fromisoformat(fetched_at_raw.replace("Z", "+00:00"))
    except Exception:
        return False
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return (_utc_now() - fetched_at) <= timedelta(hours=max(0.0, float(ttl_hours)))


def _http_get_json(url: str, cfg: KapConfig) -> Any:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "Accept-Language": "tr",
            "User-Agent": cfg.user_agent,
        },
    )
    with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _http_get_text(url: str, cfg: KapConfig) -> str:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept-Language": "tr",
            "User-Agent": cfg.user_agent,
        },
    )
    with urllib.request.urlopen(request, timeout=float(cfg.timeout_seconds)) as response:
        return response.read().decode("utf-8", errors="replace")


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
    return None


def _list_company_disclosures(member_oid: str, cfg: KapConfig, max_years_back: int = 6) -> List[Dict[str, Any]]:
    current_year = _utc_now().year
    rows: List[Dict[str, Any]] = []

    for year in range(current_year, current_year - max_years_back - 1, -1):
        url = f"{KAP_BASE_URL}/{LIST_COMPANY_EXCEL_MEMBERS_ENDPOINT}/{member_oid}/{year}/T"
        try:
            payload = _http_get_json(url, cfg)
        except urllib.error.HTTPError:
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

    unique: Dict[int, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda x: (x["year"], x["period"], x["disclosure_index"]), reverse=True):
        if row["disclosure_index"] in unique:
            continue
        unique[row["disclosure_index"]] = row
    return list(unique.values())


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


def _col_preference(period: int, income_statement: bool, prefer_income_statement_ytd: bool = False) -> Tuple[int, ...]:
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
    )
    if col_order in preferred_cols:
        score += max(1, 20 - preferred_cols.index(col_order) * 6)

    if metric_key == "net_kar":
        if "ana ortaklik paylari" in label_norm:
            score += 120
        elif "net donem kari veya zarari" in label_norm:
            score += 110
        elif "donem kari (zarari)" in label_norm:
            score += 45

        # Consolidated statements often include multiple profit layers.
        # We prefer headline net/parent-profit rows over "continued operations".
        if "surdurulen faaliyetler donem kari" in label_norm:
            score -= 30
        if "kontrol gucu olmayan paylar" in label_norm:
            score -= 80
    elif metric_key == "satis_gelirleri":
        if "hasilat" in label_norm:
            score += 35
        if "satis gelirleri" in label_norm:
            score += 25
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
        elif "ana ortakliga ait" in label_norm and "ozkaynaklar" in label_norm:
            score += 85
        elif "toplam ozkaynaklar" in label_norm:
            score += 45
        elif "ozkaynaklar" in label_norm:
            score += 30
        if "kontrol gucu olmayan paylar" in label_norm:
            score -= 40

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
) -> Optional[float]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        label_norm = str(row.get("label_norm", ""))
        if metric_key == "net_kar":
            if "kontrol gucu olmayan paylar" in label_norm:
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
        elif metric_key == "faaliyet_nakit_akisi":
            if "isletme faaliyetlerinden nakit akis" in label_norm or "faaliyetlerden elde edilen nakit akis" in label_norm:
                filtered.append(row)
        elif metric_key == "capex":
            if "nakit cikis" in label_norm and "duran varlik" in label_norm and "alim" in label_norm:
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
            elif "toplam ozkaynaklar" in label_norm:
                filtered.append(row)
            elif "ozkaynaklar" in label_norm:
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
) -> Optional[Dict[str, Any]]:
    preferred_cols = _col_preference(period=period, income_statement=False)
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


def _derive_finansal_borclar(rows: List[Dict[str, Any]], period: int) -> Optional[float]:
    def _bucket(*, includes: Tuple[str, ...], excludes: Tuple[str, ...] = ()) -> Optional[float]:
        non_related_excludes = excludes + ("iliskili taraf",)
        row = _pick_best_row(
            rows=rows,
            period=period,
            includes=includes,
            excludes=non_related_excludes,
            body_index=0,
        )
        if row is None:
            row = _pick_best_row(
                rows=rows,
                period=period,
                includes=includes,
                excludes=excludes,
                body_index=0,
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


def _derive_net_borc(rows: List[Dict[str, Any]], period: int, finansal_borclar: Optional[float]) -> Optional[float]:
    direct_row = _pick_best_row(rows=rows, period=period, includes=("net borc",), body_index=0)
    if direct_row is not None:
        return float(direct_row.get("value", 0.0))
    if finansal_borclar is None:
        return None

    nakit_row = _pick_best_row(rows=rows, period=period, includes=("nakit ve nakit benzer",), body_index=0)
    yatirim_row = _pick_best_row(rows=rows, period=period, includes=("finansal yatirim",), body_index=0)
    nakit = float(nakit_row.get("value", 0.0)) if nakit_row else 0.0
    finansal_yatirim = float(yatirim_row.get("value", 0.0)) if yatirim_row else 0.0

    if nakit == 0.0 and finansal_yatirim == 0.0:
        return None
    return float(finansal_borclar) - nakit - finansal_yatirim


def _extract_disclosure_metrics(
    detail_payload: Dict[str, Any],
    period: int,
    *,
    prefer_income_statement_ytd: bool = False,
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
        ),
        "satis_gelirleri": _pick_metric_value(
            "satis_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "brut_kar": _pick_metric_value(
            "brut_kar",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "favok": _pick_metric_value(
            "favok",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "faiz_gelirleri": _pick_metric_value(
            "faiz_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "faiz_giderleri": _pick_metric_value(
            "faiz_giderleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "net_ucret_komisyon_gelirleri": _pick_metric_value(
            "net_ucret_komisyon_gelirleri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "net_faaliyet_kari": _pick_metric_value(
            "net_faaliyet_kari",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "esas_faaliyet_kari": _pick_metric_value(
            "esas_faaliyet_kari",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "amortisman_itfa_gideri": _pick_metric_value(
            "amortisman_itfa_gideri",
            all_rows,
            period=period,
            prefer_income_statement_ytd=prefer_income_statement_ytd,
        ),
        "faaliyet_nakit_akisi": _pick_metric_value("faaliyet_nakit_akisi", all_rows, period=period),
        "capex": _pick_metric_value("capex", all_rows, period=period),
        "donen_varliklar": _pick_metric_value("donen_varliklar", all_rows, period=period),
        "duran_varliklar": _pick_metric_value("duran_varliklar", all_rows, period=period),
        "toplam_varliklar": _pick_metric_value("toplam_varliklar", all_rows, period=period),
        "kisa_vadeli_yukumlulukler": _pick_metric_value("kisa_vadeli_yukumlulukler", all_rows, period=period),
        "finansal_varliklar_net": _pick_metric_value("finansal_varliklar_net", all_rows, period=period),
        "krediler": _pick_metric_value("krediler", all_rows, period=period),
        "mevduatlar": _pick_metric_value("mevduatlar", all_rows, period=period),
        "beklenen_zarar_karsiliklari": _pick_metric_value("beklenen_zarar_karsiliklari", all_rows, period=period),
        "finansal_borclar": _pick_metric_value("finansal_borclar", all_rows, period=period),
        "net_borc": _pick_metric_value("net_borc", all_rows, period=period),
        "ozkaynaklar": _pick_metric_value("ozkaynaklar", all_rows, period=period),
    }
    if metrics["finansal_borclar"] is None:
        metrics["finansal_borclar"] = _derive_finansal_borclar(all_rows, period=period)
    if metrics["net_borc"] is None:
        metrics["net_borc"] = _derive_net_borc(
            all_rows,
            period=period,
            finansal_borclar=metrics.get("finansal_borclar"),
        )
    if metrics["favok"] is None:
        esas = metrics.get("esas_faaliyet_kari")
        amortisman = metrics.get("amortisman_itfa_gideri")
        if esas is not None and amortisman is not None:
            try:
                metrics["favok"] = float(esas) + float(amortisman)
            except Exception:
                metrics["favok"] = None
    if metrics["faaliyet_nakit_akisi"] is not None and metrics["capex"] is not None:
        metrics["serbest_nakit_akisi"] = float(metrics["faaliyet_nakit_akisi"]) + float(metrics["capex"])
    else:
        metrics["serbest_nakit_akisi"] = None
    return metrics, unit_info


def _fetch_attachment_detail(disclosure_index: int, cfg: KapConfig) -> Optional[Dict[str, Any]]:
    url = f"{KAP_BASE_URL}/{ATTACHMENT_DETAIL_ENDPOINT}/{int(disclosure_index)}"
    payload = _http_get_json(url, cfg)
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    return dict(first) if isinstance(first, dict) else None


def _quarter_label(year: int, period: int) -> str:
    return f"{int(year)}Q{int(period)}"


def fetch_kap_company_snapshot(
    *,
    company: str,
    cfg: KapConfig,
    processed_dir: Path,
    force_refresh: bool = False,
    max_quarters: int = 4,
) -> Dict[str, Any]:
    company_norm = str(company or "").strip().upper()
    if not company_norm:
        return {
            "ok": False,
            "company": company_norm,
            "error": "Sirket bilgisi bos.",
            "quarters": [],
        }

    cache_path = _cache_file_for_company(processed_dir, company_norm)
    cached = _read_cache(cache_path)
    cache_version = int(cached.get("schema_version", 0)) if isinstance(cached, dict) else 0
    if (
        cached
        and not force_refresh
        and cache_version == KAP_CACHE_SCHEMA_VERSION
        and _is_cache_fresh(cached, cfg.cache_ttl_hours)
    ):
        cached["cache_hit"] = True
        cached["cache_stale"] = False
        return cached

    try:
        member = _resolve_member(company_norm, cfg)
        if not member:
            raise RuntimeError(f"KAP uyelik kaydi bulunamadi: {company_norm}")

        disclosures = _list_company_disclosures(member_oid=member["mkk_member_oid"], cfg=cfg)
        if not disclosures:
            raise RuntimeError("KAP finansal bildirim listesi bos dondu.")

        quarter_rows: List[Dict[str, Any]] = []
        for item in disclosures:
            if len(quarter_rows) >= max(1, int(max_quarters)):
                break
            period = int(item["period"])
            year = int(item["year"])
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
            )
            metrics_ytd, _ = _extract_disclosure_metrics(
                detail,
                period=period,
                prefer_income_statement_ytd=True,
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
                }
            )

        if not quarter_rows:
            raise RuntimeError("KAP bildirim detayi alindi ancak ceyrek verisi parse edilemedi.")

        payload: Dict[str, Any] = {
            "ok": True,
            "cache_hit": False,
            "cache_stale": False,
            "schema_version": KAP_CACHE_SCHEMA_VERSION,
            "company": company_norm,
            "company_title": str(member.get("title", "")).strip(),
            "stock_code": str(quarter_rows[0].get("stock_code", "")).strip(),
            "member_oid": str(member.get("mkk_member_oid", "")).strip(),
            "source_url": f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{member.get('permalink', '')}",
            "fetched_at": _utc_now().isoformat(),
            "quarters": quarter_rows,
        }
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        if cached:
            cached["ok"] = bool(cached.get("ok", False))
            cached["cache_hit"] = True
            cached["cache_stale"] = True
            cached["error"] = str(exc)
            return cached
        return {
            "ok": False,
            "cache_hit": False,
            "cache_stale": False,
            "company": company_norm,
            "error": str(exc),
            "quarters": [],
        }
