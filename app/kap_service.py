"""KAP service layer — decouples KAP logic from Streamlit and FastAPI."""
from __future__ import annotations

import csv
import io
import re
import time
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import KapConfig
from src.kap_fetcher import fetch_kap_company_snapshot

# Static BIST100 universe snapshot used by the market terminal.
# Source basis: current XU100 constituent pages cross-checked against the
# Q1 2026 Borsa Istanbul index change announcement.
BIST100_SYMBOLS: List[str] = [
    "AEFES", "AGHOL", "AKBNK", "AKSA", "AKSEN", "ALARK", "ALTNY", "ANSGR",
    "ARCLK", "ASELS", "ASTOR", "BALSU", "BIMAS", "BRSAN", "BRYAT", "BSOKE",
    "BTCIM", "CANTE", "CCOLA", "CIMSA", "CWENE", "DAPGM", "DOAS", "DOHOL",
    "DSTKF", "ECILC", "EFOR", "EGEEN", "EKGYO", "ENERY", "ENJSA", "ENKAI",
    "EREGL", "EUPWR", "FENER", "FROTO", "GARAN", "GENIL", "GESAN", "GLRMK",
    "GRSEL", "GRTHO", "GSRAY", "GUBRF", "HALKB", "HEKTS", "ISCTR", "ISMEN",
    "IZENR", "KCAER", "KCHOL", "KLRHO", "KONTR", "KRDMD", "KTLEV", "KUYAS",
    "MAGEN", "MAVI", "MGROS", "MIATK", "MPARK", "OBAMS", "ODAS", "OTKAR",
    "OYAKC", "PASEU", "PATEK", "PETKM", "PGSUS", "QUAGR", "RALYH", "REEDR",
    "SAHOL", "SASA", "SISE", "SKBNK", "SOKM", "TABGD", "TAVHL", "TCELL",
    "THYAO", "TKFEN", "TOASO", "TRALT", "TRENJ", "TRMET", "TSKB", "TSPOR",
    "TTKOM", "TTRAK", "TUKAS", "TUPRS", "TUREX", "TURSG", "ULKER", "VAKBN",
    "VESTL", "YEOTK", "YKBNK", "ZOREN",
]

# BIST30 (XU030) constituent snapshot. Ordered by most recent BIST30 membership
# list provided by the product owner.
BIST30_SYMBOLS: List[str] = [
    "TRALT", "AKBNK", "GARAN", "EKGYO", "PGSUS", "SAHOL", "YKBNK", "VAKBN",
    "ISCTR", "AEFES", "TTKOM", "ENKAI", "DSTKF", "THYAO", "EREGL", "KCHOL",
    "KRDMD", "FROTO", "SISE", "GUBRF", "TCELL", "TAVHL", "SASA", "BIMAS",
    "ASTOR", "TOASO", "MGROS", "ASELS", "PETKM", "TUPRS",
]

BIST_INDEX_REPORT_URL = "https://www.borsaistanbul.com/datum/hisse_endeks_ds.csv"
BIST_INDEX_CACHE_TTL = 6 * 60 * 60
BIST_STOCK_INDEX_ORDER: List[str] = ["XUTUM", "XU100", "XU030"]
BIST_SECTOR_INDEX_ORDER: List[str] = [
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
BIST_INDEX_ORDER: List[str] = BIST_STOCK_INDEX_ORDER + BIST_SECTOR_INDEX_ORDER

# Official Borsa Istanbul XUTUM snapshot from hisse_endeks_ds.csv, dated
# 2026-04-29. Used only when the live official CSV cannot be read.
_BIST_ALL_FALLBACK_RAW = """
A1CAP A1YEN AAGYO ACSEL ADEL ADESE ADGYO AEFES
AFYON AGESA AGHOL AGROT AGYO AHGAZ AHSGY AKBNK
AKCNS AKENR AKFGY AKFIS AKFYE AKGRT AKHAN AKMGY
AKSA AKSEN AKSGY AKSUE AKYHO ALARK ALBRK ALCAR
ALCTL ALFAS ALGYO ALKA ALKIM ALKLC ALTNY ALVES
ANELE ANGEN ANHYT ANSGR ARASE ARCLK ARDYZ ARENA
ARFYE ARMGD ARSAN ARTMS ARZUM ASELS ASGYO ASTOR
ASUZU ATAGY ATAKP ATATP ATATR AVGYO AVHOL AVOD
AVPGY AVTUR AYCES AYDEM AYEN AYGAZ AZTEK BAGFS
BAHKM BAKAB BALSU BANVT BARMA BASGZ BAYRK BEGYO
BERA BESLR BESTE BEYAZ BFREN BIENY BIGCH BIGEN
BIGTK BIMAS BINBN BINHO BIOEN BIZIM BJKAS BLCYT
BLUME BMSCH BMSTL BNTAS BOBET BORLS BORSK BOSSA
BRISA BRKSN BRKVY BRLSM BRSAN BRYAT BSOKE BTCIM
BUCIM BULGS BURCE BURVA BVSAN BYDNR CANTE CATES
CCOLA CELHA CEMAS CEMTS CEMZY CEOEM CGCAM CIMSA
CLEBI CMBTN CONSE COSMO CRDFA CRFSA CUSAN CVKMD
CWENE DAGI DAPGM DARDL DCTTR DENGE DERHL DERIM
DESA DESPC DEVA DGATE DGGYO DGNMO DITAS DMRGD
DMSAS DNISI DOAS DOCO DOFER DOFRB DOGUB DOHOL
DOKTA DSTKF DUNYH DURDO DURKN DYOBY DZGYO EBEBK
ECILC ECOGR ECZYT EDATA EDIP EFOR EGEEN EGEGY
EGEPO EGGUB EGPRO EGSER EKGYO EKOS EKSUN ELITE
EMKEL EMPAE ENDAE ENERY ENJSA ENKAI ENSRI ENTRA
EPLAS ERBOS ERCB EREGL ERSU ESCAR ESCOM ESEN
ETILR EUPWR EUREN EYGYO FADE FENER FLAP FMIZP
FONET FORMT FORTE FRIGO FRMPL FROTO FZLGY GARAN
GARFA GEDIK GEDZA GENIL GENKM GENTS GEREL GESAN
GIPTA GLBMD GLCVY GLRMK GLRYH GLYHO GMTAS GOKNR
GOLTS GOODY GOZDE GRSEL GRTHO GSDDE GSDHO GSRAY
GUBRF GUNDG GWIND GZNMI HALKB HATEK HATSN HDFGS
HEDEF HEKTS HKTM HLGYO HOROZ HRKET HTTBT HUBVC
HUNER HURGZ ICBCT ICUGS IDGYO IEYHO IHAAS IHEVA
IHGZT IHLAS IHLGM IHYAY IMASM INDES INFO INGRM
INTEM INVEO INVES ISATR ISBTR ISCTR ISDMR ISFIN
ISGSY ISGYO ISKPL ISMEN ISSEN IZENR IZFAS IZINV
IZMDC JANTS KAPLM KAREL KARSN KARTN KATMR KAYSE
KBORU KCAER KCHOL KFEIN KGYO KIMMR KLGYO KLKIM
KLMSN KLRHO KLSER KLSYN KLYPV KMPUR KNFRT KOCMT
KONKA KONTR KONYA KOPOL KORDS KOTON KRDMA KRDMB
KRDMD KRGYO KRONT KRPLS KRSTL KRTEK KRVGD KTLEV
KTSKR KUTPO KUVVA KUYAS KZBGY KZGYO LIDER LIDFA
LILAK LINK LKMNH LMKDC LOGO LRSHO LUKSK LXGYO
LYDHO LYDYE MAALT MACKO MAGEN MAKIM MAKTK MANAS
MARBL MARKA MARMR MARTI MAVI MCARD MEDTR MEGMT
MEKAG MEPET MERCN MERIT MERKO METRO MEYSU MGROS
MHRGY MIATK MNDRS MNDTR MOBTL MOGAN MOPAS MPARK
MRGYO MRSHL MSGYO MTRKS MZHLD NATEN NETAS NETCD
NIBAS NTGAZ NTHOL NUGYO NUHCM OBAMS OBASE ODAS
ODINE OFSYM ONCSM ONRYT ORCAY ORGE OSMEN OSTIM
OTKAR OTTO OYAKC OYLUM OYYAT OZATD OZGYO OZKGY
OZRDN OZSUB OZYSR PAGYO PAHOL PAMEL PAPIL PARSN
PASEU PATEK PCILT PEKGY PENGD PENTA PETKM PETUN
PGSUS PINSU PKART PKENT PLTUR PNLSN PNSUT POLHO
POLTK PRDGS PRKAB PRKME PRZMA PSDTC PSGYO QUAGR
RALYH RAYSG REEDR RGYAS RNPOL RODRG RTALB RUBNS
RUZYE RYGYO RYSAS SAFKR SAHOL SAMAT SANEL SANFM
SANKO SARKY SASA SAYAS SDTTR SEGMN SEGYO SEKFK
SEKUR SELEC SELVA SERNT SEYKM SILVR SISE SKBNK
SKTAS SKYLP SKYMD SMART SMRTG SMRVA SNGYO SNICA
SOKE SOKM SONME SRVGY SUNTK SURGY SUWEN SVGYO
TABGD TARKM TATEN TATGD TAVHL TBORG TCELL TCKRC
TDGYO TEHOL TEKTU TERA TEZOL TGSAS THYAO TKFEN
TKNSA TLMAN TMPOL TMSN TNZTP TOASO TRALT TRCAS
TRENJ TRGYO TRHOL TRILC TRMET TSGYO TSKB TSPOR
TTKOM TTRAK TUCLK TUKAS TUPRS TUREX TURGG TURSG
UCAYM UFUK ULAS ULKER ULUFA ULUSE ULUUN UNLU
USAK VAKBN VAKFA VAKFN VAKKO VANGD VBTYZ VERTU
VERUS VESBE VESTL VKGYO VKING VRGYO VSNMD YAPRK
YATAS YAYLA YEOTK YESIL YGGYO YIGIT YKBNK YKSLN
YUNSA YYLGD ZEDUR ZERGY ZGYO ZOREN ZRGYO
"""

BIST_ALL_SYMBOLS_FALLBACK: List[str] = _BIST_ALL_FALLBACK_RAW.split()
_BIST_FALLBACK_SYMBOLS: Dict[str, List[str]] = {
    "XUTUM": BIST_ALL_SYMBOLS_FALLBACK,
    "XU100": BIST100_SYMBOLS,
    "XU030": BIST30_SYMBOLS,
    **{code: [] for code in BIST_SECTOR_INDEX_ORDER},
}
_BIST_UNIVERSE_CACHE: Dict[str, Dict[str, Any]] = {}


def _dedupe_symbols(symbols: List[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = normalize_kap_symbol(symbol)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_bist_constituent_symbol(raw: str) -> str:
    symbol = str(raw or "").strip().upper()
    if symbol.endswith(".E"):
        symbol = symbol[:-2]
    return normalize_kap_symbol(symbol)


def parse_bist_index_report_csv(payload: str) -> Dict[str, Dict[str, Any]]:
    """Parse Borsa Istanbul equity index report CSV by index code."""
    parsed: Dict[str, Dict[str, Any]] = {}
    reader = csv.DictReader(io.StringIO(str(payload or "")), delimiter=";")
    for row in reader:
        raw_symbol = str(row.get("BILESEN KODU") or "").strip()
        raw_index = str(row.get("ENDEKS KODU") or "").strip().upper()
        if raw_symbol.upper() == "CONSTITUENT CODE" or not raw_index:
            continue
        symbol = _normalize_bist_constituent_symbol(raw_symbol)
        if not symbol:
            continue
        bucket = parsed.setdefault(raw_index, {"symbols": [], "source_date": ""})
        if symbol not in bucket["symbols"]:
            bucket["symbols"].append(symbol)
        source_date = str(row.get("TARIH(GG/AA/YYYY)") or "").strip()
        if source_date and not bucket["source_date"]:
            bucket["source_date"] = source_date
    return parsed


def _fallback_bist_universe(index_code: str, *, cache_hit: bool = False) -> Dict[str, Any]:
    normalized = str(index_code or "").strip().upper()
    symbols = _dedupe_symbols(_BIST_FALLBACK_SYMBOLS.get(normalized, []))
    return {
        "index": normalized,
        "symbols": symbols,
        "count": len(symbols),
        "source": "borsa_istanbul_csv_fallback_snapshot",
        "source_url": BIST_INDEX_REPORT_URL,
        "source_date": "2026-04-29" if normalized == "XUTUM" else "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "cache_hit": cache_hit,
        "fallback_used": True,
    }


def get_bist_index_universe(index_code: str = "XUTUM", *, force_refresh: bool = False) -> Dict[str, Any]:
    """Return official BIST index constituents with visible source metadata."""
    normalized = str(index_code or "XUTUM").strip().upper()
    if normalized not in _BIST_FALLBACK_SYMBOLS:
        raise ValueError(f"unsupported_index:{normalized}")

    now = time.time()
    cached = _BIST_UNIVERSE_CACHE.get(normalized)
    if cached and not force_refresh and now - cached.get("_ts", 0) < BIST_INDEX_CACHE_TTL:
        data = dict(cached["data"])
        data["symbols"] = list(data.get("symbols") or [])
        data["cache_hit"] = True
        return data
    redis_key = f"kap:bist-index-universe:{normalized}"
    if not force_refresh:
        try:
            from app.cache import get_cache

            redis_cached = get_cache().get(redis_key)
        except Exception:
            redis_cached = None
        if isinstance(redis_cached, dict):
            data = dict(redis_cached)
            data["symbols"] = list(data.get("symbols") or [])
            data["cache_hit"] = True
            _BIST_UNIVERSE_CACHE[normalized] = {"_ts": now, "data": data}
            return data

    try:
        request = urllib.request.Request(
            BIST_INDEX_REPORT_URL,
            headers={
                "User-Agent": "ragfin-bist-index-loader/1.0",
                "Accept": "text/csv,*/*",
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8-sig", errors="replace")
        parsed = parse_bist_index_report_csv(text)
    except Exception:
        fallback = _fallback_bist_universe(normalized)
        _BIST_UNIVERSE_CACHE[normalized] = {"_ts": now, "data": fallback}
        try:
            from app.cache import get_cache

            get_cache().set(redis_key, fallback, ttl_seconds=BIST_INDEX_CACHE_TTL)
        except Exception:
            pass
        return dict(fallback)

    fetched_at = datetime.now(timezone.utc).isoformat()
    for code in BIST_INDEX_ORDER:
        row = parsed.get(code, {})
        symbols = _dedupe_symbols(list(row.get("symbols") or []))
        fallback_used = False
        if not symbols:
            fallback_used = True
            symbols = _dedupe_symbols(_BIST_FALLBACK_SYMBOLS.get(code, []))
        data = {
            "index": code,
            "symbols": symbols,
            "count": len(symbols),
            "source": "borsa_istanbul_csv" if not fallback_used else "borsa_istanbul_csv_fallback_snapshot",
            "source_url": BIST_INDEX_REPORT_URL,
            "source_date": str(row.get("source_date") or ""),
            "fetched_at": fetched_at,
            "cache_hit": False,
            "fallback_used": fallback_used,
        }
        _BIST_UNIVERSE_CACHE[code] = {"_ts": now, "data": data}
        try:
            from app.cache import get_cache

            get_cache().set(f"kap:bist-index-universe:{code}", data, ttl_seconds=BIST_INDEX_CACHE_TTL)
        except Exception:
            pass

    result = dict(_BIST_UNIVERSE_CACHE[normalized]["data"])
    result["symbols"] = list(result.get("symbols") or [])
    return result


def get_bist_index_companies(index_code: str = "XUTUM", *, force_refresh: bool = False) -> List[str]:
    return list(get_bist_index_universe(index_code, force_refresh=force_refresh).get("symbols") or [])


def get_bist30_companies() -> List[str]:
    return get_bist_index_companies("XU030")


# Search candidates are kept as a separate concept from the market universe so
# header search can evolve independently without changing the terminal screen.
KAP_COMPANY_CANDIDATES: List[str] = list(BIST_ALL_SYMBOLS_FALLBACK)

# Prefer exchange symbols in dropdown and normalize common aliases from dataset names.
KAP_SYMBOL_ALIASES: Dict[str, str] = {
    "BIM": "BIMAS",
    "BIMAS": "BIMAS",
    "MIGROS": "MGROS",
    "MGROS": "MGROS",
    "SOK": "SOKM",
    "SOKM": "SOKM",
    "TAV": "TAVHL",
    "TAVHL": "TAVHL",
}


def normalize_kap_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    return KAP_SYMBOL_ALIASES.get(raw, raw)


def get_bist100_companies() -> List[str]:
    return get_bist_index_companies("XU100")


def get_bist_all_companies() -> List[str]:
    return get_bist_index_companies("XUTUM")


def get_kap_companies(indexed_companies: Optional[List[str]] = None) -> List[str]:
    """Return deduplicated, sorted company list (candidates + indexed)."""
    merged: List[str] = []
    seen: set[str] = set()

    def _append(symbol: str) -> None:
        normalized = normalize_kap_symbol(symbol)
        if not normalized:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        merged.append(normalized)

    for candidate in KAP_COMPANY_CANDIDATES:
        _append(candidate)

    if indexed_companies:
        for c in indexed_companies:
            _append(c)

    return sorted(merged)


def get_kap_snapshot(
    *,
    company: str,
    cfg: KapConfig,
    processed_dir: Path,
    force_refresh: bool = False,
    max_quarters: int = 10,
    use_cache_when_complete: bool = False,
) -> Dict[str, Any]:
    """Fetch a raw snapshot via kap_fetcher with correct keyword args."""
    normalized_company = normalize_kap_symbol(company)
    return fetch_kap_company_snapshot(
        company=normalized_company,
        cfg=cfg,
        processed_dir=processed_dir,
        force_refresh=force_refresh,
        max_quarters=max_quarters,
        use_cache_when_complete=use_cache_when_complete,
    )


# Display labels for frontend normalization
_METRIC_LABELS: Dict[str, str] = {
    "net_kar": "Net Kâr",
    "satis_gelirleri": "Hasılat",
    "brut_kar": "Brüt Kâr",
    "favok": "FAVÖK",
    "faiz_gelirleri": "Faiz Gelirleri",
    "faiz_giderleri": "Faiz Giderleri",
    "net_faaliyet_kari": "Net Faaliyet Kârı",
    "esas_faaliyet_kari": "Esas Faaliyet Kârı",
    "prim_uretimi": "Prim Üretimi",
    "alinan_net_primler": "Alınan Net Primler",
    "teknik_gelirler": "Teknik Gelirler",
    "teknik_denge": "Teknik Denge",
    "ozkaynaklar": "Özkaynaklar",
    "nakit_ve_nakit_benzerleri": "Nakit ve Nakit Benzerleri",
    "finansal_varliklar_sigortacilik": "Finansal Varlıklar (Sigortacılık)",
    "nakit_benzeri_finansal_varliklar": "Nakit Benzeri Finansal Varlıklar",
    "esas_faaliyetlerden_alacaklar": "Esas Faaliyetlerden Alacaklar",
    "teknik_karsiliklar": "Teknik Karşılıklar",
    "esas_faaliyetlerden_borclar": "Esas Faaliyetlerden Borçlar",
    "donen_varliklar": "Dönen Varlıklar",
    "kisa_vadeli_yukumlulukler": "Kısa Vadeli Yükümlülükler",
    "toplam_varliklar": "Toplam Varlıklar",
    "finansal_borclar": "Finansal Borçlar",
    "net_borc": "Net Borç",
    "faaliyet_nakit_akisi": "Faaliyet Nakit Akışı",
    "serbest_nakit_akisi": "Serbest Nakit Akışı",
    "odenmis_sermaye": "Ödenmiş Sermaye",
    "cikarilmis_sermaye": "Çıkarılmış Sermaye",
}

_SUMMARY_KEYS = [
    "net_kar", "satis_gelirleri", "brut_kar", "favok",
    "ozkaynaklar", "toplam_varliklar", "net_borc",
]


_FLOW_RESTATEMENT_KEYS = (
    "satis_gelirleri",
    "brut_kar",
    "favok",
    "net_kar",
    "faaliyet_nakit_akisi",
    "serbest_nakit_akisi",
    "faiz_gelirleri",
    "faiz_giderleri",
    "net_ucret_komisyon_gelirleri",
    "net_faaliyet_kari",
    "esas_faaliyet_kari",
    "prim_uretimi",
    "alinan_net_primler",
    "teknik_gelirler",
    "teknik_denge",
)

_POINT_IN_TIME_RESTATEMENT_KEYS = (
    "donen_varliklar",
    "duran_varliklar",
    "toplam_varliklar",
    "kisa_vadeli_yukumlulukler",
    "finansal_borclar",
    "net_borc",
    "ozkaynaklar",
    "krediler",
    "mevduatlar",
    "finansal_varliklar_net",
    "beklenen_zarar_karsiliklari",
    "esas_faaliyetlerden_alacaklar",
    "teknik_karsiliklar",
    "esas_faaliyetlerden_borclar",
    "nakit_ve_nakit_benzerleri",
    "finansal_varliklar_sigortacilik",
    "nakit_benzeri_finansal_varliklar",
)

_ANALYSIS_NOTE = (
    "Çeyreklik akışlarda ilk açıklanan değerler kullanılır. Karşılaştırılabilir "
    "bilanço ve analitik oranlarda önceki dönem tutarları güncel baza taşınabilir; "
    "ham değerler payload içindeki `*_original` alanlarında korunur."
)

_TCMB_CONSUMER_PRICES_URL = (
    "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%2BEN/Main%2BMenu/"
    "Statistics/Inflation%2BData/Consumer%2BPrices"
)
_TCMB_MONTHLY_ROW_PATTERN = re.compile(
    r"<tr>\s*<td[^>]*>(\d{2})-(\d{4})</td>\s*<td[^>]*>[^<]+</td>\s*<td[^>]*>([^<]+)</td>\s*</tr>",
    flags=re.IGNORECASE,
)


def _period_key(row: Dict[str, Any]) -> tuple[int, int]:
    return (int(row.get("year") or 0), int(row.get("period") or 0))


def _period_to_month(period: int) -> int:
    normalized_period = int(period or 0)
    if 1 <= normalized_period <= 4:
        return normalized_period * 3
    return normalized_period


def _safe_number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _metric_from_raw(row: Dict[str, Any], bucket: str, metric_key: str) -> Optional[float]:
    source = row.get(bucket) or {}
    if not isinstance(source, dict):
        return None
    return _safe_number(source.get(metric_key))


def _candidate_factor(restated_value: Optional[float], original_value: Optional[float]) -> Optional[float]:
    if restated_value is None or original_value in (None, 0):
        return None
    try:
        factor = float(restated_value) / float(original_value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if factor <= 0:
        return None
    if factor < 0.5 or factor > 3.0:
        return None
    return factor


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _derive_same_year_factor(newer: Dict[str, Any], older: Dict[str, Any]) -> Optional[float]:
    candidates: List[float] = []
    for metric_key in _FLOW_RESTATEMENT_KEYS:
        newer_ytd = _metric_from_raw(newer, "metrics_ytd", metric_key)
        newer_quarter = _metric_from_raw(newer, "metrics_quarterly", metric_key)
        older_ytd = _metric_from_raw(older, "metrics_ytd", metric_key)
        if newer_ytd is None or newer_quarter is None or older_ytd in (None, 0):
            continue
        restated_older_ytd = newer_ytd - newer_quarter
        factor = _candidate_factor(restated_older_ytd, older_ytd)
        if factor is not None:
            candidates.append(factor)
    return _median(candidates)


@lru_cache(maxsize=1)
def _load_monthly_inflation_rates() -> Dict[tuple[int, int], float]:
    request = urllib.request.Request(
        _TCMB_CONSUMER_PRICES_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8", errors="ignore")

    rates: Dict[tuple[int, int], float] = {}
    for month_raw, year_raw, pct_raw in _TCMB_MONTHLY_ROW_PATTERN.findall(payload):
        try:
            month = int(month_raw)
            year = int(year_raw)
            pct = float(str(pct_raw).replace(",", ".").strip())
        except ValueError:
            continue
        rates[(year, month)] = pct / 100.0
    return rates


def _next_month(year: int, month: int) -> tuple[int, int]:
    if month >= 12:
        return year + 1, 1
    return year, month + 1


def _derive_same_year_inflation_factor(newer: Dict[str, Any], older: Dict[str, Any]) -> Optional[float]:
    newer_year = int(newer.get("year") or 0)
    older_year = int(older.get("year") or 0)
    newer_month = _period_to_month(int(newer.get("period") or 0))
    older_month = _period_to_month(int(older.get("period") or 0))

    if newer_year != older_year or newer_month <= 0 or older_month <= 0 or newer_month <= older_month:
        return None

    try:
        monthly_rates = _load_monthly_inflation_rates()
    except Exception:
        return None

    factor = 1.0
    year, month = _next_month(older_year, older_month)
    while (year, month) <= (newer_year, newer_month):
        rate = monthly_rates.get((year, month))
        if rate is None:
            return None
        factor *= 1.0 + rate
        year, month = _next_month(year, month)
    return factor


def _derive_comparative_factor(newer: Dict[str, Any], older: Dict[str, Any]) -> Optional[float]:
    candidates: List[float] = []
    for metric_key in _POINT_IN_TIME_RESTATEMENT_KEYS:
        restated_value = _metric_from_raw(newer, "metrics_comparative", metric_key)
        original_value = _metric_from_raw(older, "metrics", metric_key)
        factor = _candidate_factor(restated_value, original_value)
        if factor is not None:
            candidates.append(factor)
    return _median(candidates)


def _derive_adjacent_factor(newer: Dict[str, Any], older: Dict[str, Any]) -> tuple[float, str]:
    same_year = int(newer.get("year") or 0) == int(older.get("year") or 0)
    if same_year:
        same_year_factor = _derive_same_year_factor(newer, older)
        if same_year_factor is not None:
            return same_year_factor, "same_year_ytd"
        inflation_factor = _derive_same_year_inflation_factor(newer, older)
        if inflation_factor is not None:
            return inflation_factor, "same_year_inflation"
        return 1.0, "reported_filing"

    comparative_factor = _derive_comparative_factor(newer, older)
    if comparative_factor is not None:
        return comparative_factor, "comparative_override"

    return 1.0, "reported_filing"


def _scale_metric_set(source: Dict[str, Any], factor: float) -> Dict[str, Optional[float]]:
    scaled: Dict[str, Optional[float]] = {}
    for key, value in source.items():
        numeric = _safe_number(value)
        scaled[key] = None if numeric is None else numeric * factor
    return scaled


def _merge_metric_override(
    target: Dict[str, Optional[float]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    if not override:
        return target
    merged = dict(target)
    for key, value in override.items():
        numeric = _safe_number(value)
        if numeric is not None:
            merged[key] = numeric
    return merged


def _build_analysis_state(
    quarters: List[Dict[str, Any]],
) -> tuple[Dict[tuple[int, int], float], Dict[tuple[int, int], str], Dict[tuple[int, int], Dict[str, Dict[str, Any]]]]:
    ordered = sorted(quarters, key=_period_key)
    if not ordered:
        return {}, {}, {}

    multiplier_map: Dict[tuple[int, int], float] = {}
    source_map: Dict[tuple[int, int], str] = {}
    latest_key = _period_key(ordered[-1])
    multiplier_map[latest_key] = 1.0
    source_map[latest_key] = "current_period"

    for idx in range(len(ordered) - 2, -1, -1):
        older = ordered[idx]
        newer = ordered[idx + 1]
        older_key = _period_key(older)
        newer_key = _period_key(newer)
        factor, source = _derive_adjacent_factor(newer, older)
        multiplier_map[older_key] = multiplier_map.get(newer_key, 1.0) * factor
        source_map[older_key] = source

    overrides: Dict[tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    latest = ordered[-1]
    latest_year = int(latest.get("year") or 0)
    latest_period = int(latest.get("period") or 0)

    prev_year_same_period_key = (latest_year - 1, latest_period)
    if any(_period_key(row) == prev_year_same_period_key for row in ordered):
        overrides.setdefault(prev_year_same_period_key, {})
        overrides[prev_year_same_period_key]["metrics_ytd"] = dict(latest.get("metrics_ytd_comparative") or {})
        overrides[prev_year_same_period_key]["metrics_quarterly"] = dict(latest.get("metrics_quarterly_comparative") or {})

    prev_year_end_key = (latest_year - 1, 4)
    prev_year_end_row = next((row for row in ordered if _period_key(row) == prev_year_end_key), None)
    if prev_year_end_row is not None:
        comparative_metrics = dict(latest.get("metrics_comparative") or {})
        # KAP HTML parse'ı bazı raporlarda comparative sütununu yanlış birim
        # (bin TL yerine TL) ya da yanlış satır olarak yakalayabiliyor. Eğer
        # comparative değer ham (filed) sayıya göre büyüklük mertebesinde
        # uyumsuzsa o anahtarı override'a almıyoruz; ham veri kullanılır.
        sanitized = _sanitize_comparative_against_filed(
            comparative_metrics,
            filed_metrics=dict(prev_year_end_row.get("metrics") or {}),
            keys=_POINT_IN_TIME_RESTATEMENT_KEYS,
        )
        if sanitized:
            overrides.setdefault(prev_year_end_key, {})
            overrides[prev_year_end_key]["metrics"] = sanitized

    return multiplier_map, source_map, overrides


def _sanitize_comparative_against_filed(
    comparative_metrics: Dict[str, Any],
    *,
    filed_metrics: Dict[str, Any],
    keys: Iterable[str],
    max_ratio: float = 8.0,
) -> Dict[str, Any]:
    """Filter out comparative values whose magnitude diverges suspiciously from
    the originally filed point-in-time metric. Bilanço (balance sheet) kalemleri
    iki ardışık çeyrek arasında 8 katından fazla değişmez; öyle bir farkı
    gördüğümüzde HTML parse hatası kabul edip override'ı düşürüyoruz. Diğer
    anahtarlar olduğu gibi korunur."""

    cleaned: Dict[str, Any] = {}
    for key, value in comparative_metrics.items():
        try:
            comparative_value = float(value) if value is not None else None
        except (TypeError, ValueError):
            comparative_value = None
        filed_raw = filed_metrics.get(key)
        try:
            filed_value = float(filed_raw) if filed_raw is not None else None
        except (TypeError, ValueError):
            filed_value = None
        if (
            key in keys
            and comparative_value is not None
            and filed_value is not None
            and filed_value != 0
        ):
            ratio = abs(filed_value) / max(1.0, abs(comparative_value))
            if ratio > max_ratio or (1.0 / max(1e-9, ratio)) > max_ratio:
                # Suspicious magnitude mismatch; keep filed value by skipping override.
                continue
        cleaned[key] = value
    return cleaned


def normalize_snapshot_for_frontend(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten raw snapshot into a frontend-friendly shape."""
    result: Dict[str, Any] = {
        "ok": raw.get("ok", False),
        "company": raw.get("company", ""),
        "company_title": raw.get("company_title", ""),
        "stock_code": raw.get("stock_code", ""),
        "fetched_at": raw.get("fetched_at", ""),
        "cache_hit": raw.get("cache_hit", False),
        "cache_stale": raw.get("cache_stale", False),
        "error": raw.get("error"),
        "analysis_basis": "latest_comparable",
        "analysis_note": _ANALYSIS_NOTE,
        "insurance_premium_disclosures": _normalize_insurance_premium_disclosures(
            raw.get("insurance_premium_disclosures") or []
        ),
    }

    quarters = raw.get("quarters") or []
    multiplier_map, source_map, override_map = _build_analysis_state(quarters)
    normalized_quarters = []
    for q in quarters:
        quarter_key = _period_key(q)
        factor = multiplier_map.get(quarter_key, 1.0)
        currency = str(q.get("currency", "TL"))
        raw_metrics = q.get("metrics") or {}
        raw_metrics_quarterly = q.get("metrics_quarterly") or raw_metrics
        raw_metrics_ytd = q.get("metrics_ytd") or raw_metrics
        raw_metrics_comparative = q.get("metrics_comparative") or {}
        raw_metrics_quarterly_comparative = q.get("metrics_quarterly_comparative") or raw_metrics_comparative
        raw_metrics_ytd_comparative = q.get("metrics_ytd_comparative") or raw_metrics_comparative

        analysis_metrics = _scale_metric_set(raw_metrics, factor)
        analysis_metrics_quarterly = _scale_metric_set(raw_metrics_quarterly, factor)
        analysis_metrics_ytd = _scale_metric_set(raw_metrics_ytd, factor)

        quarter_overrides = override_map.get(quarter_key, {})
        analysis_metrics = _merge_metric_override(analysis_metrics, quarter_overrides.get("metrics"))
        analysis_metrics_quarterly = _merge_metric_override(
            analysis_metrics_quarterly,
            quarter_overrides.get("metrics_quarterly"),
        )
        analysis_metrics_ytd = _merge_metric_override(
            analysis_metrics_ytd,
            quarter_overrides.get("metrics_ytd"),
        )

        metric_keys = sorted(
            set(_METRIC_LABELS.keys())
            | set(raw_metrics.keys())
            | set(raw_metrics_quarterly.keys())
            | set(raw_metrics_ytd.keys())
            | set(raw_metrics_comparative.keys())
            | set(raw_metrics_quarterly_comparative.keys())
            | set(raw_metrics_ytd_comparative.keys())
        )

        def _normalize_metric_set(source: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            normalized: Dict[str, Dict[str, Any]] = {}
            for key in metric_keys:
                label = _METRIC_LABELS.get(key, key.replace("_", " ").title())
                value = source.get(key)
                normalized[key] = {
                    "label": label,
                    "value": value,
                    "display": _fmt_number(value, currency),
                }
            return normalized

        nq: Dict[str, Any] = {
            "quarter": q.get("quarter", ""),
            "year": q.get("year"),
            "period": q.get("period"),
            "currency": currency,
            "publish_date": q.get("publish_date", ""),
            "metrics": _normalize_metric_set(analysis_metrics),
            "metrics_quarterly": _normalize_metric_set(analysis_metrics_quarterly),
            "metrics_ytd": _normalize_metric_set(analysis_metrics_ytd),
            "metrics_original": _normalize_metric_set(raw_metrics),
            "metrics_quarterly_original": _normalize_metric_set(raw_metrics_quarterly),
            "metrics_ytd_original": _normalize_metric_set(raw_metrics_ytd),
            "metrics_comparative": _normalize_metric_set(raw_metrics_comparative),
            "metrics_quarterly_comparative": _normalize_metric_set(raw_metrics_quarterly_comparative),
            "metrics_ytd_comparative": _normalize_metric_set(raw_metrics_ytd_comparative),
            "analysis_multiplier": factor,
            "analysis_factor_source": source_map.get(quarter_key, "reported_filing"),
        }
        normalized_quarters.append(nq)

    result["quarters"] = normalized_quarters

    # Summary = latest quarter's key metrics
    if normalized_quarters:
        latest = normalized_quarters[0]
        summary: Dict[str, Any] = {}
        for key in _SUMMARY_KEYS:
            m = latest["metrics"].get(key, {})
            summary[key] = {
                "label": m.get("label", key),
                "value": m.get("value"),
                "display": m.get("display", "-"),
            }
        result["summary"] = summary
        result["latest_quarter"] = latest["quarter"]
    else:
        result["summary"] = {}
        result["latest_quarter"] = None

    return result


def _fmt_number(val: Optional[float], currency: str = "TL") -> str:
    if val is None:
        return "-"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1_000_000_000:
        formatted = f"{sign}{abs_val / 1_000_000_000:,.2f} Milyar"
    elif abs_val >= 1_000_000:
        formatted = f"{sign}{abs_val / 1_000_000:,.2f} Milyon"
    elif abs_val >= 1_000:
        formatted = f"{sign}{abs_val / 1_000:,.1f} Bin"
    else:
        formatted = f"{sign}{abs_val:,.0f}"
    return f"{formatted} {currency}"


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "-"
    return f"% {val:.0f}"


def _normalize_insurance_premium_disclosures(rows: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        monthly = _safe_number(row.get("monthly_gross_premium"))
        ytd = _safe_number(row.get("ytd_gross_premium"))
        prev_monthly = _safe_number(row.get("previous_year_monthly_gross_premium"))
        prev_ytd = _safe_number(row.get("previous_year_ytd_gross_premium"))
        monthly_yoy = _safe_number(row.get("monthly_yoy_pct"))
        ytd_yoy = _safe_number(row.get("ytd_yoy_pct"))
        normalized.append(
            {
                "year": row.get("year"),
                "month": row.get("month"),
                "period_label": row.get("period_label"),
                "period_start": row.get("period_start"),
                "period_end": row.get("period_end"),
                "published_at": row.get("published_at"),
                "disclosure_index": row.get("disclosure_index"),
                "summary": row.get("summary"),
                "source_url": row.get("source_url"),
                "monthly_gross_premium": monthly,
                "monthly_gross_premium_display": _fmt_number(monthly, "TL"),
                "ytd_gross_premium": ytd,
                "ytd_gross_premium_display": _fmt_number(ytd, "TL"),
                "previous_year_monthly_gross_premium": prev_monthly,
                "previous_year_monthly_gross_premium_display": _fmt_number(prev_monthly, "TL"),
                "previous_year_ytd_gross_premium": prev_ytd,
                "previous_year_ytd_gross_premium_display": _fmt_number(prev_ytd, "TL"),
                "monthly_yoy_pct": monthly_yoy,
                "monthly_yoy_pct_display": _fmt_pct(monthly_yoy),
                "ytd_yoy_pct": ytd_yoy,
                "ytd_yoy_pct_display": _fmt_pct(ytd_yoy),
            }
        )
    normalized.sort(key=lambda item: (int(item.get("year") or 0), int(item.get("month") or 0)))
    return normalized
