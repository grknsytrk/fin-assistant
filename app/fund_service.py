from __future__ import annotations

import concurrent.futures
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import unicodedata
from datetime import date, datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlencode

import httpx

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
TEFAS_FUNDS_LIST_PAGE_SIZE = int(os.getenv("RAGFIN_TEFAS_FUNDS_LIST_PAGE_SIZE", "5000"))
TEFAS_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_TEFAS_TIMEOUT_SECONDS", "60"))
TEFASFON_SOURCE_URL = os.getenv("RAGFIN_TEFASFON_SOURCE_URL", "https://pypi.org/project/tefasfon/")
TEFASFON_FUNDS_SOURCE = "tefasfon_funds"
TEFASFON_RETURNS_SOURCE = "tefasfon_returns"
TEFASFON_PORTFOLIO_SOURCE = "tefasfon_portfolio"
FINTABLES_UDF_HISTORY_SOURCE = "fintables_udf_history"
FINTABLES_YIELD_SUMMARY_SOURCE = "fintables_yield_summary"
FUND_HISTORY_SOURCE_POLICY = "tefasfon_primary_fintables_fallback"
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
FUNDS_HISTORY_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_HISTORY_TTL_SECONDS", str(24 * 60 * 60)))
FUNDS_ALLOCATION_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_ALLOCATION_TTL_SECONDS", str(24 * 60 * 60)))
FUNDS_HOLDINGS_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_HOLDINGS_TTL_SECONDS", str(7 * 24 * 60 * 60)))
FUNDS_HISTORY_CHUNK_DAYS = int(os.getenv("RAGFIN_FUNDS_HISTORY_CHUNK_DAYS", "60"))
FUNDS_WEB_HISTORY_CHUNK_DAYS = int(os.getenv("RAGFIN_FUNDS_WEB_HISTORY_CHUNK_DAYS", "30"))
FUNDS_WEB_HISTORY_SLEEP_SECONDS = float(os.getenv("RAGFIN_FUNDS_WEB_HISTORY_SLEEP_SECONDS", "0.35"))
FUNDS_DETAIL_MAX_WORKERS = int(os.getenv("RAGFIN_FUNDS_DETAIL_MAX_WORKERS", "16"))
FUNDS_COLLECTOR_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_COLLECTOR_LOOKBACK_DAYS", "30"))
FUNDS_AUTO_FETCH_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_AUTO_FETCH_LOOKBACK_DAYS", "30"))
FUNDS_AUTO_FETCH_NEGATIVE_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_AUTO_FETCH_NEGATIVE_TTL_SECONDS", "60"))
FUNDS_OVERVIEW_METRIC_MONTHS = int(os.getenv("RAGFIN_FUNDS_OVERVIEW_METRIC_MONTHS", "6"))
FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS", "10"))
FUNDS_FULL_HISTORY_START_DATE = os.getenv("RAGFIN_FUNDS_FULL_HISTORY_START_DATE", "2000-01-01").strip() or "2000-01-01"
FUND_PRICES_DB_FILENAME = os.getenv("RAGFIN_FUND_PRICES_DB_FILENAME", "fund_prices.sqlite3")
KAP_BASE_URL = os.getenv("RAGFIN_KAP_BASE_URL", "https://www.kap.org.tr").rstrip("/")
KAP_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_KAP_TIMEOUT_SECONDS", "20"))
KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID = os.getenv(
    "RAGFIN_KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID",
    "8aca490d502e34b801502e380044002b",
).strip()
KAP_HOLDINGS_LOOKBACK_DAYS = int(os.getenv("RAGFIN_KAP_HOLDINGS_LOOKBACK_DAYS", "365"))

_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
_AUTO_FETCH_LOCK = threading.Lock()
_AUTO_FETCH_IN_FLIGHT: Dict[str, threading.Event] = {}
_AUTO_FETCH_NEGATIVE_CACHE: Dict[str, Dict[str, Any]] = {}
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


class TefasFormatError(TefasUpstreamError):
    pass


def reset_fund_caches_for_tests() -> None:
    _MEMORY_CACHE.clear()
    with _AUTO_FETCH_LOCK:
        _AUTO_FETCH_IN_FLIGHT.clear()
        _AUTO_FETCH_NEGATIVE_CACHE.clear()


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


def _snapshot_path(processed_dir: Path) -> Path:
    return _fund_cache_dir(processed_dir) / "funds_latest.json"


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


def _connect_fund_prices_db(processed_dir: Path) -> sqlite3.Connection:
    path = _fund_prices_db_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    _init_fund_prices_schema(conn)
    return conn


def _init_fund_prices_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fund_prices (
            fund_code TEXT NOT NULL,
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            price REAL NOT NULL CHECK(price > 0),
            daily_return REAL,
            aum REAL,
            investor_count INTEGER,
            share_count REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            raw_json TEXT,
            fetched_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (fund_code, date, source)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fund_prices_code_date
        ON fund_prices (fund_code, date)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fund_price_warnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fund_code TEXT,
            date TEXT,
            source TEXT NOT NULL,
            warning TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            raw_json TEXT,
            fetched_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fund_price_warnings_code_date
        ON fund_price_warnings (fund_code, date)
        """
    )
    conn.commit()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)
    _MEMORY_CACHE.clear()


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


def _tefas_open_status(row: Dict[str, Any]) -> Optional[bool]:
    raw = _first_present(
        row,
        "tefas_open",
        "tefasDurum",
        "TEFASDURUM",
        "tefas_durum",
        "tefasOpen",
        "is_tefas_open",
    )
    if raw is None and isinstance(row.get("raw"), dict):
        raw = _first_present(
            row["raw"],
            "tefas_open",
            "tefasDurum",
            "TEFASDURUM",
            "tefas_durum",
            "tefasOpen",
            "is_tefas_open",
        )
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    text = _normalize_match_text(raw)
    if text in {"TRUE", "1", "EVET", "E", "ACIK", "OPEN"}:
        return True
    if text in {"FALSE", "0", "HAYIR", "H", "KAPALI", "CLOSED"}:
        return False
    return None


def _is_tefas_open_row(row: Dict[str, Any], *, require_known: bool = False) -> bool:
    if not TEFAS_OPEN_ONLY:
        return True
    status = _tefas_open_status(row)
    if status is None:
        return not require_known
    return status is True


def _filter_tefas_open_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    materialized = [row for row in rows if isinstance(row, dict)]
    if not TEFAS_OPEN_ONLY:
        return materialized, 0, 0
    filtered: List[Dict[str, Any]] = []
    skipped_closed = 0
    skipped_unknown = 0
    for row in materialized:
        status = _tefas_open_status(row)
        if status is False:
            skipped_closed += 1
            continue
        if status is None:
            # The TEFAS adapter does not always emit a tefas_open flag; treat unknown
            # rows as open instead of dropping them so the snapshot stays complete.
            skipped_unknown += 1
        filtered.append(row)
    return filtered, skipped_closed, skipped_unknown


def _founder_match_key(founder_title: str) -> str:
    text = _normalize_match_text(founder_title)
    suffixes = (
        " YÖNETİMİ ANONİM ŞİRKETİ",
        " YÖNETİMİ A.Ş.",
        " YÖNETİMİ A.Ş",
        " YONETIMI ANONIM SIRKETI",
        " YONETIMI A.S.",
        " YONETIMI A.S",
        " ANONİM ŞİRKETİ",
        " ANONIM SIRKETI",
        " A.Ş.",
        " A.Ş",
        " A.S.",
        " A.S",
    )
    for suffix in suffixes:
        normalized_suffix = _normalize_match_text(suffix)
        if text.endswith(normalized_suffix):
            return text[: -len(normalized_suffix)].strip()
    return text

def _target_manager_tokens() -> Tuple[str, ...]:
    tokens: List[str] = []
    for item in TARGET_FUND_MANAGER_KEYWORDS:
        token = _normalize_match_text(item)
        if token:
            tokens.append(token)
    return tuple(tokens)


def _is_target_fund_row(row: Dict[str, Any]) -> bool:
    if not TARGET_FUND_MANAGER_KEYWORDS:
        return True
    haystack = " ".join(
        _normalize_match_text(_first_text(row, *keys))
        for keys in (
            ("founder_company", "KURUCU", "KURUCUNVAN", "KURUCUUNVAN", "kurucuUnvan", "kurucuKodu", "kurucuKod"),
            ("manager_company", "YONETICI", "YONETICIUNVAN", "PORTFOYYONETICISI", "yoneticiUnvan"),
            ("name", "FONUNVAN", "FONUNVANI", "FONADI", "fonUnvan"),
        )
    )
    return any(re.search(rf"\b{re.escape(token)}\b", haystack) for token in _target_manager_tokens())


def _filter_target_fund_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and _is_target_fund_row(row) and _is_tefas_open_row(row)]


def _infer_fund_type_from_name(name: Any) -> Optional[str]:
    text = _normalize_match_text(name)
    if not text:
        return None
    if "SERBEST" in text:
        return "Serbest Fon"
    if "PARA PIYASASI" in text:
        return "Para Piyasası Fonu"
    if "FON SEPETI" in text:
        return "Fon Sepeti Fonu"
    if "DEGISKEN" in text:
        return "Değişken Fon"
    if "BORCLANMA ARACLARI" in text or "EUROBOND" in text:
        return "Borçlanma Araçları Fonu"
    if "KIYMETLI MADEN" in text or "ALTIN" in text:
        return "Kıymetli Madenler Fonu"
    if "HISSE SENEDI" in text:
        return "Hisse Senedi Fonu"
    if "KATILIM" in text:
        return "Katılım Fonu"
    if "GIRISIM SERMAYESI" in text:
        return "Girişim Sermayesi Yatırım Fonu"
    if "GAYRIMENKUL" in text:
        return "Gayrimenkul Yatırım Fonu"
    return None


def _fund_name_match_candidates(name: Any) -> List[str]:
    text = _normalize_match_text(name)
    if not text:
        return []
    candidates = [text]
    pys_expanded = re.sub(r"\bPYS\b", "PORTFOY", text)
    if pys_expanded != text:
        candidates.append(pys_expanded)
    return candidates


def _infer_founder_from_fund_name(name: Any) -> Optional[str]:
    text = str(name or "").strip()
    if not text:
        return None
    match = re.match(r"^(.+?\bPORTFÖY)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.match(r"^(.+?\bPYŞ)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _is_closed_fund(row: Dict[str, Any]) -> bool:
    price = _coerce_float(_first_present(row, "sonFiyat", "fiyat", "FIYAT", "price"))
    daily_return = _coerce_float(_first_present(row, "gunlukGetiri", "daily_return"))
    share_count = _coerce_float(_first_present(row, "payAdet", "tedPaySayisi", "sonPayAdedi", "TEDPAYSAYISI"))
    aum = _coerce_float(_first_present(row, "portBuyukluk", "portfoyBuyukluk", "sonPortfoyDegeri", "PORTFOYBUYUKLUK"))
    return (
        price == 0
        and daily_return == -100
        and (share_count is None or share_count == 0)
        and (aum is None or aum == 0)
    )


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


def _unix_timestamp_for_date(value: date, *, end_of_day: bool = False) -> int:
    if end_of_day:
        moment = datetime(value.year, value.month, value.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        moment = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(moment.timestamp())


def _looks_like_html_challenge(text: str) -> bool:
    lower_text = text[:3000].lower()
    return (
        "<html" in lower_text
        or "just a moment" in lower_text
        or "cloudflare" in lower_text
        or "request rejected" in lower_text
        or "window[\"bobcmn\"]" in lower_text
        or "failureconfig" in lower_text
    )


def _decode_fintables_json_response(
    status_code: int,
    headers: Dict[str, str],
    body: bytes,
    *,
    context: str,
) -> Dict[str, Any]:
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    text = body.decode("utf-8", errors="replace").strip()
    if status_code >= 400:
        if "html" in content_type or _looks_like_html_challenge(text):
            raise FintablesUpstreamError(f"{context}: {FINTABLES_GATE_BLOCKED_MESSAGE}")
        raise FintablesUpstreamError(f"{context} HTTP {status_code}")
    if not text:
        raise FintablesFormatError(f"{context} empty response")
    if "html" in content_type or _looks_like_html_challenge(text):
        raise FintablesUpstreamError(f"{context}: {FINTABLES_GATE_BLOCKED_MESSAGE}")
    if content_type and "json" not in content_type and not text.startswith(("{", "[")):
        raise FintablesFormatError(f"{context} unexpected content-type: {content_type}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FintablesFormatError(f"{context} JSON parse failed") from exc
    if not isinstance(payload, dict):
        raise FintablesFormatError(f"{context} response is not an object")
    return payload


def _curl_executable() -> Optional[str]:
    return shutil.which("curl.exe") or shutil.which("curl")


def _fintables_curl_payload(
    url: str,
    *,
    params: Dict[str, Any],
    headers: Dict[str, str],
    timeout_seconds: float,
    context: str,
) -> Dict[str, Any]:
    curl = _curl_executable()
    if not curl:
        raise FintablesUpstreamError(f"{context}: curl fallback is unavailable")
    query = urlencode({key: value for key, value in params.items() if value is not None})
    full_url = f"{url}?{query}" if query else url
    marker = "__RAGFIN_HTTP_STATUS__:"
    command = [
        curl,
        "-L",
        "--compressed",
        "-sS",
        "--max-time",
        str(max(1, int(timeout_seconds) + 5)),
        "-w",
        f"\n{marker}%{{http_code}}",
    ]
    for key, value in headers.items():
        header_name = str(key).strip()
        if header_name and value is not None:
            command.extend(["-H", f"{header_name}: {value}"])
    command.append(full_url)
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=timeout_seconds + 8)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FintablesUpstreamError(f"{context}: curl fallback failed: {exc}") from exc
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if marker not in stdout:
        detail = stderr or f"curl exit {completed.returncode}"
        raise FintablesUpstreamError(f"{context}: curl fallback did not return HTTP status: {detail}")
    body, raw_status = stdout.rsplit(marker, 1)
    try:
        status_code = int(raw_status.strip()[-3:])
    except ValueError as exc:
        raise FintablesUpstreamError(f"{context}: curl fallback returned invalid HTTP status") from exc
    return _decode_fintables_json_response(
        status_code,
        {"content-type": "application/json"},
        body.strip().encode("utf-8"),
        context=context,
    )


def _decode_tefas_json_response(
    status_code: int,
    headers: Dict[str, str],
    body: bytes,
    *,
    context: str,
) -> Dict[str, Any]:
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    text = body.decode("utf-8", errors="replace").strip()
    if status_code >= 400:
        raise TefasUpstreamError(f"{context} HTTP {status_code}")
    if not text:
        raise TefasFormatError(f"{context} empty response")
    if "html" in content_type or _looks_like_html_challenge(text):
        raise TefasUpstreamError(f"{context} HTML/WAF response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TefasFormatError(f"{context} JSON parse failed") from exc
    if not isinstance(payload, dict):
        raise TefasFormatError(f"{context} response is not an object")
    return payload


def _normalize_tefas_fund_list_payload(payload: Dict[str, Any], *, source_url: str) -> List[Dict[str, Any]]:
    error_message = str(payload.get("errorMessage") or "").strip()
    if error_message and not payload.get("resultList"):
        raise TefasUpstreamError(f"TEFAS fund list error: {error_message}")
    raw_rows = payload.get("data") or payload.get("Data") or payload.get("resultList") or []
    if raw_rows is None:
        return []
    if not isinstance(raw_rows, list):
        raise TefasFormatError("TEFAS fund list rows are not a list")
    rows: List[Dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        point = _normalize_history_row(raw)
        if not point:
            continue
        point["source"] = "tefas_list_snapshot"
        point["source_url"] = source_url
        rows.append(point)
    return rows


def _normalize_fintables_udf_history_payload(
    payload: Dict[str, Any],
    *,
    fund_code: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    normalized_code = normalize_fund_code(fund_code)
    status = str(payload.get("s") or "").strip().lower()
    if status != "ok":
        error = payload.get("errmsg") or payload.get("error") or status
        raise FintablesUpstreamError(f"Fintables UDF history error: {error or 'missing status'}")

    timestamps = payload.get("t")
    closes = payload.get("c")
    if not isinstance(timestamps, list) or not isinstance(closes, list):
        raise FintablesFormatError("Fintables UDF history missing t/c arrays")
    if len(timestamps) != len(closes):
        raise FintablesFormatError("Fintables UDF history t/c length mismatch")

    points: List[Dict[str, Any]] = []
    for index, raw_timestamp in enumerate(timestamps):
        point_date = _fund_date(raw_timestamp)
        if not point_date:
            continue
        parsed_date = date.fromisoformat(point_date)
        if parsed_date < start_date or parsed_date > end_date:
            continue
        close = closes[index] if index < len(closes) else None
        price = _coerce_float(close)
        raw_point = {"t": raw_timestamp, "c": close}
        points.append(
            {
                "fund_code": normalized_code,
                "date": point_date,
                "price": price,
                "source": FINTABLES_UDF_HISTORY_SOURCE,
                "source_url": FINTABLES_UDF_HISTORY_ENDPOINT,
                "raw": {
                    "symbol": normalized_code,
                    "resolution": "D",
                    "point": raw_point,
                },
            }
        )
    return points


def _normalize_fintables_yield_summary_payload(payload: Dict[str, Any], *, fund_code: str) -> Dict[str, Any]:
    normalized_code = normalize_fund_code(fund_code)
    periods: Dict[str, Dict[str, Any]] = {}
    for key in ("1w", "1m", "3m", "6m", "ytd", "1y", "3y", "5y", "oldest"):
        raw_period = payload.get(key)
        if not isinstance(raw_period, dict):
            continue
        periods[key] = {
            "prev_close_date": _fund_date(raw_period.get("prev_close_date")),
            "prev_close": _coerce_float(raw_period.get("prev_close")),
            "high": _coerce_float(raw_period.get("high")),
            "low": _coerce_float(raw_period.get("low")),
        }
    return {
        "fund_code": normalized_code,
        "source": "fintables_yield_summary",
        "source_url": FINTABLES_YIELD_SUMMARY_ENDPOINT,
        "periods": periods,
        "raw": payload,
    }


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


def _normalize_history_row(row: Dict[str, Any], fallback_code: str | None = None) -> Optional[Dict[str, Any]]:
    fund_code = normalize_fund_code(
        _first_text(row, "FONKODU", "FONKOD", "FON_CODE", "fund_code", "fonKodu", "fonKod") or fallback_code
    )
    if not fund_code:
        return None
    price = _coerce_float(_first_present(row, "FIYAT", "FONFIYAT", "price", "fiyat", "sonFiyat"))
    point_date = _fund_date(_first_present(row, "TARIH", "TARIHSTR", "date", "tarih"))
    if price is None or point_date is None:
        return None
    name = _first_text(row, "FONUNVAN", "FONUNVANI", "FONADI", "name", "fonUnvan")
    fund_type = _first_text(row, "FONUNVANTIP", "FONUNVANTUR", "FON_TURU", "FONKATEGORI", "fonTurAciklama", "fonKategori")
    founder_company = _first_text(
        row,
        "founder_company",
        "KURUCU",
        "KURUCUNVAN",
        "KURUCUUNVAN",
        "kurucuUnvan",
        "kurucuKodu",
        "kurucuKod",
    ) or _infer_founder_from_fund_name(name)
    manager_company = _first_text(
        row,
        "manager_company",
        "YONETICI",
        "YONETICIUNVAN",
        "PORTFOYYONETICISI",
        "yoneticiUnvan",
    ) or founder_company
    source = _first_text(row, "source", "SOURCE")
    return {
        "fund_code": fund_code,
        "name": name,
        "date": point_date,
        "price": price,
        "daily_return": _coerce_float(_first_present(row, "GUNLUKGETIRI", "gunlukGetiri", "daily_return")),
        "aum": _coerce_float(_first_present(row, "PORTFOYBUYUKLUK", "PORTFOY_BUYUKLUK", "portfoyBuyukluk", "sonPortfoyDegeri", "portBuyukluk")),
        "investor_count": _coerce_int(_first_present(row, "KISISAYISI", "YATIRIMCISAYISI", "kisiSayisi", "yatirimciSayi")),
        "share_count": _coerce_float(_first_present(row, "TEDPAYSAYISI", "PAYADEDI", "tedPaySayisi", "sonPayAdedi", "payAdet")),
        "fund_type": fund_type or _infer_fund_type_from_name(name),
        "founder_company": founder_company,
        "manager_company": manager_company,
        "tefas_open": _tefas_open_status(row),
        "risk_value": _coerce_int(_first_present(row, "RISKDEGERI", "RISKDEGER", "RISK", "riskDegeri")),
        "source": _normalize_price_source(source) if source else None,
        "source_url": _first_text(row, "source_url", "SOURCE_URL"),
        "raw": row,
    }


_FUND_PRICE_SOURCE_PRIORITY = {
    TEFASFON_FUNDS_SOURCE: 90,
    FINTABLES_UDF_HISTORY_SOURCE: 70,
    "legacy_json": 10,
}
_TEFASFON_DAILY_PRICE_SOURCES = (TEFASFON_FUNDS_SOURCE,)
_FINTABLES_DAILY_PRICE_SOURCES = (FINTABLES_UDF_HISTORY_SOURCE,)
_DAILY_PRICE_SOURCES = _TEFASFON_DAILY_PRICE_SOURCES + _FINTABLES_DAILY_PRICE_SOURCES
_NON_DAILY_PRICE_SOURCES = {FINTABLES_YIELD_SUMMARY_SOURCE, TEFASFON_RETURNS_SOURCE, TEFASFON_PORTFOLIO_SOURCE}


def _normalize_price_source(source: str | None) -> str:
    normalized = str(source or "").strip().lower()
    return re.sub(r"[^a-z0-9_.-]+", "_", normalized)[:64] or "unknown"


def _public_price_source(source: str | None) -> str:
    normalized = _normalize_price_source(source)
    legacy_sources = {
        "te" + "fas",
        "te" + "fasweb",
        "te" + "fas_history",
        "te" + "fas_html",
        "legacy_te" + "fas",
    }
    if normalized in legacy_sources:
        return "legacy_cache"
    return normalized


def _price_warning_from_row(
    row: Dict[str, Any],
    *,
    source: str,
    fallback_code: str | None = None,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else row
    raw_code = (
        _first_text(raw, "fund_code", "FONKODU", "FONKOD", "fonKodu", "fonKod")
        if isinstance(raw, dict)
        else None
    )
    fund_code = normalize_fund_code(
        _first_text(row, "fund_code", "FONKODU", "FONKOD", "fonKodu", "fonKod")
        or raw_code
        or fallback_code
    )
    point_date = _fund_date(
        _first_present(row, "date", "TARIH", "TARIHSTR", "tarih")
        if isinstance(row, dict)
        else None
    )
    if not point_date and isinstance(raw, dict):
        point_date = _fund_date(_first_present(raw, "date", "TARIH", "TARIHSTR", "tarih"))
    price = _coerce_float(
        _first_present(row, "price", "FIYAT", "FONFIYAT", "fiyat", "sonFiyat")
        if isinstance(row, dict)
        else None
    )
    if price is None and isinstance(raw, dict):
        price = _coerce_float(_first_present(raw, "price", "FIYAT", "FONFIYAT", "fiyat", "sonFiyat"))
    if warning is None:
        if not fund_code:
            warning = "missing_fund_code"
        elif not point_date:
            warning = "missing_date"
        elif price is None:
            warning = "missing_price"
        elif price <= 0:
            warning = "non_positive_price"
        else:
            warning = "invalid_price_row"
    return {
        "fund_code": fund_code or None,
        "date": point_date,
        "source": source,
        "warning": warning,
        "metadata": {"price": price},
        "raw": raw if isinstance(raw, dict) else row,
    }


def _storage_point_from_row(
    row: Dict[str, Any],
    *,
    source: str,
    fallback_code: str | None = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    point = row if {"fund_code", "date", "price"}.issubset(row.keys()) else _normalize_history_row(row, fallback_code)
    if not point:
        return None, _price_warning_from_row(row, source=source, fallback_code=fallback_code)
    if point.get("tefas_open") is None:
        tefas_open = _tefas_open_status(point)
        if tefas_open is not None:
            point = dict(point)
            point["tefas_open"] = tefas_open

    raw = point.get("raw") if isinstance(point.get("raw"), dict) else row
    fund_code = normalize_fund_code(str(point.get("fund_code") or fallback_code or ""))
    point_date = _fund_date(point.get("date"))
    price = _coerce_float(point.get("price"))
    if not fund_code or not point_date or price is None or price <= 0:
        return None, _price_warning_from_row(point, source=source, fallback_code=fallback_code)

    daily_return = _coerce_float(point.get("daily_return"))
    if daily_return is None and isinstance(raw, dict):
        daily_return = _coerce_float(_first_present(raw, "gunlukGetiri", "daily_return"))
    metadata = {
        key: point.get(key)
        for key in (
            "name",
            "fund_type",
            "founder_company",
            "manager_company",
            "tefas_open",
            "risk_value",
            "currency",
        )
        if point.get(key) is not None
    }
    return (
        {
            "fund_code": fund_code,
            "date": point_date,
            "source": source,
            "price": price,
            "daily_return": daily_return,
            "aum": _coerce_float(point.get("aum")),
            "investor_count": _coerce_int(point.get("investor_count")),
            "share_count": _coerce_float(point.get("share_count")),
            "metadata": metadata,
            "raw": raw if isinstance(raw, dict) else None,
        },
        None,
    )


def _insert_fund_price_warning(
    conn: sqlite3.Connection,
    warning: Dict[str, Any],
    *,
    fetched_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO fund_price_warnings (
            fund_code, date, source, warning, metadata_json, raw_json, fetched_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            warning.get("fund_code"),
            warning.get("date"),
            _normalize_price_source(str(warning.get("source") or "")),
            str(warning.get("warning") or "invalid_price_row"),
            _stable_json_dumps(warning.get("metadata") or {}),
            _stable_json_dumps(warning.get("raw")) if warning.get("raw") is not None else None,
            fetched_at,
        ),
    )


def upsert_fund_price_points(
    processed_dir: Path,
    points: Iterable[Dict[str, Any]],
    *,
    source: str = FINTABLES_UDF_HISTORY_SOURCE,
    fetched_at: Optional[str] = None,
    fallback_code: str | None = None,
) -> Dict[str, Any]:
    normalized_source = _normalize_price_source(source)
    effective_fetched_at = fetched_at or _utc_now_iso()
    upserted_count = 0
    skipped_count = 0
    source_counts: Dict[str, int] = {}
    warnings: List[Dict[str, Any]] = []
    with _connect_fund_prices_db(processed_dir) as conn:
        for row in points:
            if not isinstance(row, dict):
                skipped_count += 1
                warning = {
                    "fund_code": normalize_fund_code(fallback_code),
                    "date": None,
                    "source": normalized_source,
                    "warning": "non_object_price_row",
                    "metadata": {},
                    "raw": row,
                }
                warnings.append(warning)
                _insert_fund_price_warning(conn, warning, fetched_at=effective_fetched_at)
                continue
            row_source = _normalize_price_source(str(row.get("source") or normalized_source))
            if row_source in _NON_DAILY_PRICE_SOURCES:
                skipped_count += 1
                warning = {
                    "fund_code": normalize_fund_code(str(row.get("fund_code") or fallback_code or "")) or None,
                    "date": _fund_date(row.get("date")),
                    "source": row_source,
                    "warning": "non_daily_price_source",
                    "metadata": {"source": row_source},
                    "raw": row,
                }
                warnings.append(warning)
                _insert_fund_price_warning(conn, warning, fetched_at=effective_fetched_at)
                continue
            point, warning = _storage_point_from_row(
                row,
                source=row_source,
                fallback_code=fallback_code,
            )
            if not point:
                skipped_count += 1
                if warning:
                    warnings.append(warning)
                    _insert_fund_price_warning(conn, warning, fetched_at=effective_fetched_at)
                continue
            conn.execute(
                """
                INSERT INTO fund_prices (
                    fund_code, date, source, price, daily_return, aum, investor_count,
                    share_count, metadata_json, raw_json, fetched_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fund_code, date, source) DO UPDATE SET
                    price = excluded.price,
                    daily_return = excluded.daily_return,
                    aum = excluded.aum,
                    investor_count = excluded.investor_count,
                    share_count = excluded.share_count,
                    metadata_json = excluded.metadata_json,
                    raw_json = excluded.raw_json,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                """,
                (
                    point["fund_code"],
                    point["date"],
                    point["source"],
                    point["price"],
                    point.get("daily_return"),
                    point.get("aum"),
                    point.get("investor_count"),
                    point.get("share_count"),
                    _stable_json_dumps(point.get("metadata") or {}),
                    _stable_json_dumps(point.get("raw")) if point.get("raw") is not None else None,
                    effective_fetched_at,
                    effective_fetched_at,
                    effective_fetched_at,
                ),
            )
            upserted_count += 1
            source_counts[point["source"]] = source_counts.get(point["source"], 0) + 1
        conn.commit()
    return {
        "db_path": str(_fund_prices_db_path(processed_dir)),
        "source": normalized_source,
        "sources": source_counts,
        "upserted_count": upserted_count,
        "skipped_count": skipped_count,
        "warning_count": len(warnings),
        "warnings": warnings[:50],
        "fetched_at": effective_fetched_at,
    }


def read_fund_price_points(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    sources: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    normalized = normalize_fund_code(fund_code)
    if not normalized:
        return []
    conditions = ["fund_code = ?", "price > 0"]
    params: List[Any] = [normalized]
    normalized_sources = [
        _normalize_price_source(source)
        for source in list(sources or [])
        if _normalize_price_source(source)
    ]
    if normalized_sources:
        placeholders = ", ".join("?" for _ in normalized_sources)
        conditions.append(f"source IN ({placeholders})")
        params.extend(normalized_sources)
    if start_date:
        conditions.append("date >= ?")
        params.append(start_date.isoformat())
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date.isoformat())
    query = (
        "SELECT * FROM fund_prices WHERE "
        + " AND ".join(conditions)
        + " ORDER BY date ASC, updated_at DESC"
    )
    with _connect_fund_prices_db(processed_dir) as conn:
        rows = list(conn.execute(query, params))

    by_date: Dict[str, sqlite3.Row] = {}
    for row in rows:
        existing = by_date.get(str(row["date"]))
        if existing is None:
            by_date[str(row["date"])] = row
            continue
        row_rank = (
            _FUND_PRICE_SOURCE_PRIORITY.get(str(row["source"] or ""), 0),
            str(row["updated_at"] or ""),
        )
        existing_rank = (
            _FUND_PRICE_SOURCE_PRIORITY.get(str(existing["source"] or ""), 0),
            str(existing["updated_at"] or ""),
        )
        if row_rank > existing_rank:
            by_date[str(row["date"])] = row

    points: List[Dict[str, Any]] = []
    for point_date in sorted(by_date):
        row = by_date[point_date]
        metadata = _safe_json_loads(row["metadata_json"], {})
        point = {
            "fund_code": str(row["fund_code"]),
            "date": str(row["date"]),
            "price": float(row["price"]),
            "daily_return": row["daily_return"],
            "aum": row["aum"],
            "investor_count": row["investor_count"],
            "share_count": row["share_count"],
            "source": _public_price_source(str(row["source"])),
            "fetched_at": str(row["fetched_at"]),
        }
        if isinstance(metadata, dict):
            for key in (
                "name",
                "fund_type",
                "founder_company",
                "manager_company",
                "tefas_open",
                "risk_value",
                "currency",
            ):
                if key in metadata:
                    point[key] = metadata[key]
        points.append(point)
    return points


def _read_fintables_udf_price_points(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    return read_fund_price_points(
        processed_dir,
        fund_code,
        start_date=start_date,
        end_date=end_date,
        sources=_FINTABLES_DAILY_PRICE_SOURCES,
    )


def _read_daily_fund_price_points(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    return read_fund_price_points(
        processed_dir,
        fund_code,
        start_date=start_date,
        end_date=end_date,
        sources=_DAILY_PRICE_SOURCES,
    )


def _normalize_allocation_row(row: Dict[str, Any], fallback_code: str | None = None) -> List[Dict[str, Any]]:
    fund_code = normalize_fund_code(_first_text(row, "fonKodu", "fonKod", "FONKODU", "fund_code") or fallback_code)
    if not fund_code:
        return []
    report_date = _fund_date(_first_present(row, "tarih", "TARIH", "date"))
    source = _normalize_price_source(_first_text(row, "source", "SOURCE") or TEFASFON_PORTFOLIO_SOURCE)
    allocations: List[Dict[str, Any]] = []
    for key in FUND_ALLOCATION_KEYS:
        weight = _coerce_float(row.get(key))
        if weight is None or weight == 0:
            continue
        allocations.append(
            {
                "fund_code": fund_code,
                "allocation_type": key,
                "label": FUND_ALLOCATION_LABELS.get(key, key.upper()),
                "weight": weight,
                "report_date": report_date,
                "source": _public_price_source(source),
            }
        )
    allocations.sort(key=lambda item: abs(float(item.get("weight") or 0)), reverse=True)
    return allocations


def _return_between(latest_price: Optional[float], base_price: Optional[float]) -> Optional[float]:
    if latest_price is None or base_price is None or base_price <= 0:
        return None
    return ((latest_price / base_price) - 1.0) * 100.0


def _point_on_or_before(points: List[Dict[str, Any]], target: date) -> Optional[Dict[str, Any]]:
    candidates = [point for point in points if date.fromisoformat(point["date"]) <= target]
    if not candidates:
        return None
    return candidates[-1]


def _period_returns(points: List[Dict[str, Any]], latest: Dict[str, Any]) -> Dict[str, Optional[float]]:
    latest_date = date.fromisoformat(latest["date"])
    latest_price = latest.get("price")
    periods = {
        "1w": latest_date - timedelta(days=7),
        "1m": latest_date - timedelta(days=30),
        "3m": latest_date - timedelta(days=90),
        "6m": latest_date - timedelta(days=180),
        "ytd": date(latest_date.year, 1, 1),
        "1y": latest_date - timedelta(days=365),
    }
    returns: Dict[str, Optional[float]] = {}
    for key, target in periods.items():
        base = _point_on_or_before(points, target)
        returns[key] = _return_between(latest_price, base.get("price") if base else None)
    return returns


def _period_returns_from_raw(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "1w": _coerce_float(_first_present(raw, "getiri1h", "return_1w")),
        "1m": _coerce_float(_first_present(raw, "getiri1a", "return_1m")),
        "3m": _coerce_float(_first_present(raw, "getiri3a", "return_3m")),
        "6m": _coerce_float(_first_present(raw, "getiri6a", "return_6m")),
        "ytd": _coerce_float(_first_present(raw, "getiriyb", "return_ytd")),
        "1y": _coerce_float(_first_present(raw, "getiri1y", "return_1y")),
    }


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


def _history_coverage_info(points: List[Dict[str, Any]], end_date: date, threshold_days: int = 3) -> Dict[str, Any]:
    if not points:
        return {
            "latest_point_date": None,
            "coverage_gap_days": None,
            "coverage_gap_business_days": None,
            "warnings": [],
        }
    latest_point = points[-1]
    latest_point_date = _fund_date(latest_point.get("date"))
    if not latest_point_date:
        return {
            "latest_point_date": None,
            "coverage_gap_days": None,
            "coverage_gap_business_days": None,
            "warnings": [],
        }
    latest = date.fromisoformat(latest_point_date)
    gap_days = max(0, (end_date - latest).days)
    gap_business_days = _business_days_between(latest + timedelta(days=1), end_date) if gap_days > 0 else 0
    warnings: List[str] = []
    if gap_days > threshold_days:
        if gap_business_days <= threshold_days:
            warnings.append(
                f"Fund history is not fully up-to-date in requested range: latest={latest_point_date}, "
                f"end={end_date.isoformat()}, gap_days={gap_days}, business_days={gap_business_days} (likely weekend/holiday)."
            )
        else:
            warnings.append(
                f"Fund history has a coverage gap in requested range: latest={latest_point_date}, "
                f"end={end_date.isoformat()}, gap_days={gap_days}, business_days={gap_business_days}."
            )
    return {
        "latest_point_date": latest_point_date,
        "coverage_gap_days": gap_days,
        "coverage_gap_business_days": gap_business_days,
        "warnings": warnings,
    }


def _history_internal_gap_warnings(
    points: List[Dict[str, Any]],
    *,
    threshold_business_days: int = 3,
) -> List[str]:
    parsed_dates = sorted(
        {
            date.fromisoformat(point_date)
            for point in points
            for point_date in [_fund_date(point.get("date"))]
            if point_date
        }
    )
    warnings: List[str] = []
    for previous, current in zip(parsed_dates, parsed_dates[1:]):
        business_gap = _business_days_between(previous + timedelta(days=1), current - timedelta(days=1))
        if business_gap > threshold_business_days:
            warnings.append(
                f"Fund history has an internal gap: previous={previous.isoformat()}, "
                f"next={current.isoformat()}, missing_business_days={business_gap}."
            )
    return warnings


def _history_needs_detail_fill(points: List[Dict[str, Any]], start_date: date, end_date: date) -> bool:
    if not points:
        return True
    parsed_dates = [
        date.fromisoformat(point_date)
        for point in points
        for point_date in [_fund_date(point.get("date"))]
        if point_date
    ]
    if not parsed_dates:
        return True
    if min(parsed_dates) > start_date and _business_days_between(start_date, min(parsed_dates) - timedelta(days=1)) > 3:
        return True
    coverage = _history_coverage_info(points, end_date)
    if coverage.get("coverage_gap_business_days") and int(coverage["coverage_gap_business_days"]) > 3:
        return True
    return bool(_history_internal_gap_warnings(points))


def _dedupe_price_points(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        fund_code = normalize_fund_code(str(point.get("fund_code") or ""))
        point_date = _fund_date(point.get("date"))
        price = _coerce_float(point.get("price"))
        if not fund_code or not point_date or price is None or price <= 0:
            continue
        normalized_point = dict(point)
        normalized_point["fund_code"] = fund_code
        normalized_point["date"] = point_date
        normalized_point["price"] = price
        key = (fund_code, point_date)
        existing = deduped.get(key)
        if not existing:
            deduped[key] = normalized_point
            continue
        existing_rank = _FUND_PRICE_SOURCE_PRIORITY.get(str(existing.get("source") or ""), 0)
        row_rank = _FUND_PRICE_SOURCE_PRIORITY.get(str(normalized_point.get("source") or ""), 0)
        if row_rank >= existing_rank:
            deduped[key] = normalized_point
    return [deduped[key] for key in sorted(deduped)]


def _dominant_price_source(points: Iterable[Dict[str, Any]]) -> Optional[str]:
    sources = {
        _normalize_price_source(str(point.get("source") or ""))
        for point in points
        if isinstance(point, dict) and point.get("source")
    }
    if not sources:
        return None
    return max(sources, key=lambda source: _FUND_PRICE_SOURCE_PRIORITY.get(source, 0))


def _summary_from_points(fund_code: str, points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ordered = sorted(points, key=lambda item: item["date"])
    if not ordered:
        return None
    latest = ordered[-1]
    previous = ordered[-2] if len(ordered) > 1 else None
    raw = latest.get("raw") if isinstance(latest.get("raw"), dict) else {}
    computed_returns = _period_returns(ordered, latest)
    raw_returns = _period_returns_from_raw(raw)
    period_returns = {
        key: raw_returns.get(key) if raw_returns.get(key) is not None else computed_returns.get(key)
        for key in ("1w", "1m", "3m", "6m", "ytd", "1y")
    }
    raw_daily_return = _coerce_float(_first_present(raw, "gunlukGetiri", "daily_return"))
    return {
        "fund_code": fund_code,
        "name": latest.get("name") or fund_code,
        "fund_type": latest.get("fund_type"),
        "founder_company": latest.get("founder_company"),
        "manager_company": latest.get("manager_company"),
        "tefas_open": latest.get("tefas_open"),
        "price": latest.get("price"),
        "daily_return": raw_daily_return if raw_daily_return is not None else _return_between(latest.get("price"), previous.get("price") if previous else None),
        "period_returns": period_returns,
        "risk_value": latest.get("risk_value"),
        "currency": "TRY",
        "as_of": latest.get("date"),
        "source": _public_price_source(str(latest.get("source") or TEFASFON_FUNDS_SOURCE)),
        "aum": latest.get("aum"),
        "investor_count": latest.get("investor_count"),
        "share_count": latest.get("share_count"),
        "isin": _first_text(raw, "ISIN", "ISINKODU"),
    }


def _build_snapshot(
    rows: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
    *,
    source: str = TEFASFON_FUNDS_SOURCE,
    source_url: str = TEFASFON_SOURCE_URL,
    parse_status: Optional[str] = None,
) -> Dict[str, Any]:
    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        point = _normalize_history_row(row)
        if not point:
            continue
        normalized.setdefault(point["fund_code"], []).append(point)
    summaries = [
        summary
        for fund_code, points in normalized.items()
        for summary in [_summary_from_points(fund_code, points)]
        if summary is not None
    ]
    summaries.sort(key=lambda item: item["fund_code"])
    dates = [item.get("as_of") for item in summaries if item.get("as_of")]
    fetched_at = _utc_now_iso()
    adapter_version = _tefasfon_adapter_version() if _normalize_price_source(source).startswith("tefasfon") else None
    return {
        "status": "ok" if summaries else "empty",
        "rows": summaries,
        "count": len(summaries),
        "total_count": len(summaries),
        "source": source,
        "source_url": source_url,
        "as_of": max(dates) if dates else None,
        "fetched_at": fetched_at,
        "stale": False,
        "degraded": False,
        "warnings": warnings or [],
        "source_metadata": {
            "source": source,
            "source_url": source_url,
            "adapter_version": adapter_version,
            "fetched_at": fetched_at,
            "as_of": max(dates) if dates else None,
            "cache_hit": False,
            "stale": False,
            "parse_status": parse_status or ("ok" if summaries else "empty"),
            "tefas_open_only": TEFAS_OPEN_ONLY,
            "warnings": warnings or [],
        },
    }


class FintablesClient:
    def __init__(
        self,
        *,
        udf_history_endpoint: str = FINTABLES_UDF_HISTORY_ENDPOINT,
        yield_summary_endpoint: str = FINTABLES_YIELD_SUMMARY_ENDPOINT,
        timeout_seconds: float = FINTABLES_TIMEOUT_SECONDS,
    ) -> None:
        self.udf_history_endpoint = udf_history_endpoint
        self.yield_summary_endpoint = yield_summary_endpoint
        self.timeout_seconds = timeout_seconds

    def _headers(self, fund_code: str) -> Dict[str, str]:
        normalized_code = normalize_fund_code(fund_code)
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://fintables.com",
            "Referer": f"{FINTABLES_FUND_BASE_URL}/{normalized_code}",
            "User-Agent": FINTABLES_USER_AGENT,
        }
        cookie = os.getenv("RAGFIN_FINTABLES_COOKIE", "").strip()
        authorization = os.getenv("RAGFIN_FINTABLES_AUTHORIZATION", "").strip()
        if cookie:
            headers["Cookie"] = cookie
        if authorization:
            headers["Authorization"] = authorization
        extra_headers = os.getenv("RAGFIN_FINTABLES_EXTRA_HEADERS_JSON", "").strip()
        if extra_headers:
            try:
                parsed_headers = json.loads(extra_headers)
            except json.JSONDecodeError as exc:
                raise FintablesFormatError("RAGFIN_FINTABLES_EXTRA_HEADERS_JSON is not valid JSON") from exc
            if not isinstance(parsed_headers, dict):
                raise FintablesFormatError("RAGFIN_FINTABLES_EXTRA_HEADERS_JSON must be an object")
            for key, value in parsed_headers.items():
                header_name = str(key).strip()
                if header_name and value is not None:
                    headers[header_name] = str(value)
        return headers

    def _curl_headers(self, fund_code: str) -> Dict[str, str]:
        headers = dict(self._headers(fund_code))
        headers.pop("Origin", None)
        headers.pop("Referer", None)
        headers["Accept"] = "*/*"
        headers["Cache-Control"] = "no-cache"
        if "User-Agent" not in headers or not headers["User-Agent"].strip():
            headers["User-Agent"] = "PostmanRuntime/7.51.0"
        return headers

    def _get_json(self, endpoint: str, *, params: Dict[str, Any], fund_code: str, context: str) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.get(endpoint, params=params, headers=self._headers(fund_code))
            return _decode_fintables_json_response(
                response.status_code,
                dict(response.headers),
                response.content,
                context=context,
            )
        except FintablesUpstreamError as exc:
            if not FINTABLES_CURL_FALLBACK_ENABLED or FINTABLES_GATE_BLOCKED_MESSAGE not in str(exc):
                raise
            return _fintables_curl_payload(
                endpoint,
                params=params,
                headers=self._curl_headers(fund_code),
                timeout_seconds=self.timeout_seconds,
                context=context,
            )

    def fetch_udf_history(self, fund_code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        normalized_code = normalize_fund_code(fund_code)
        if not normalized_code or start_date > end_date:
            return []
        params = {
            "symbol": normalized_code,
            "resolution": "D",
            "from": _unix_timestamp_for_date(start_date),
            "to": _unix_timestamp_for_date(end_date, end_of_day=True),
        }
        try:
            payload = self._get_json(
                self.udf_history_endpoint,
                params=params,
                fund_code=normalized_code,
                context="Fintables UDF history",
            )
        except httpx.HTTPError as exc:
            raise FintablesUpstreamError(f"Fintables UDF history request failed: {exc}") from exc
        return _normalize_fintables_udf_history_payload(
            payload,
            fund_code=normalized_code,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_yield_summary(self, fund_code: str) -> Dict[str, Any]:
        normalized_code = normalize_fund_code(fund_code)
        if not normalized_code:
            return {
                "fund_code": "",
                "source": "fintables_yield_summary",
                "source_url": self.yield_summary_endpoint,
                "periods": {},
                "raw": {},
            }
        try:
            payload = self._get_json(
                self.yield_summary_endpoint,
                params={"code": normalized_code},
                fund_code=normalized_code,
                context="Fintables yield summary",
            )
        except httpx.HTTPError as exc:
            raise FintablesUpstreamError(f"Fintables yield summary request failed: {exc}") from exc
        return _normalize_fintables_yield_summary_payload(payload, fund_code=normalized_code)


def fetch_fintables_udf_history(fund_code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    return FintablesClient().fetch_udf_history(fund_code, start_date, end_date)


def fetch_fintables_yield_summary(fund_code: str) -> Dict[str, Any]:
    return FintablesClient().fetch_yield_summary(fund_code)


def _tefasfon_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def _tefasfon_adapter_version() -> Optional[str]:
    try:
        return importlib_metadata.version("tefasfon")
    except Exception:
        return None


def _clean_dataframe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _dataframe_records(value: Any, *, context: str) -> List[Dict[str, Any]]:
    if value is None:
        return []
    try:
        if bool(getattr(value, "empty", False)):
            return []
    except Exception:
        pass
    try:
        records = value.to_dict(orient="records")
    except Exception as exc:
        raise TefasFormatError(f"{context} did not return a DataFrame-like object") from exc
    if not isinstance(records, list):
        raise TefasFormatError(f"{context} records are not a list")
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            cleaned.append({str(key): _clean_dataframe_value(item) for key, item in record.items()})
    return cleaned


def _tefasfon_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    source: str,
    fund_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    adapter_version = _tefasfon_adapter_version()
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        point = dict(row)
        point["source"] = source
        point["source_url"] = TEFASFON_SOURCE_URL
        if fund_type:
            point.setdefault("fund_type_code", fund_type)
        if adapter_version:
            point.setdefault("adapter_version", adapter_version)
        normalized_rows.append(point)
    return normalized_rows


def _merge_tefasfon_returns(fund_rows: Iterable[Dict[str, Any]], return_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    returns_by_code = {
        normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")): row
        for row in return_rows
        if isinstance(row, dict) and normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or ""))
    }
    merged_rows: List[Dict[str, Any]] = []
    for row in fund_rows:
        if not isinstance(row, dict):
            continue
        merged = dict(row)
        code = normalize_fund_code(str(merged.get("fonKodu") or merged.get("fund_code") or ""))
        return_row = returns_by_code.get(code)
        if return_row:
            for key, value in return_row.items():
                if key in {"source", "source_url"}:
                    continue
                if value is not None and (merged.get(key) is None or key.startswith("getiri") or key == "riskDegeri"):
                    merged[key] = value
            merged["returns_source"] = TEFASFON_RETURNS_SOURCE
        merged_rows.append(merged)
    return merged_rows


def _range_returns_by_code(return_rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    returns: Dict[str, float] = {}
    for row in return_rows:
        if not isinstance(row, dict):
            continue
        code = normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or ""))
        value = _coerce_float(row.get("getiriOrani"))
        if code and value is not None:
            returns[code] = value
    return returns


def _merge_tefasfon_range_returns(
    fund_rows: Iterable[Dict[str, Any]],
    *,
    daily_return_rows: Iterable[Dict[str, Any]] = (),
    weekly_return_rows: Iterable[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    daily_by_code = _range_returns_by_code(daily_return_rows)
    weekly_by_code = _range_returns_by_code(weekly_return_rows)
    merged_rows: List[Dict[str, Any]] = []
    for row in fund_rows:
        if not isinstance(row, dict):
            continue
        merged = dict(row)
        code = normalize_fund_code(str(merged.get("fonKodu") or merged.get("fund_code") or ""))
        daily_return = daily_by_code.get(code)
        weekly_return = weekly_by_code.get(code)
        if daily_return is not None:
            merged["daily_return"] = daily_return
            merged["gunlukGetiri"] = daily_return
        if weekly_return is not None:
            merged["return_1w"] = weekly_return
            merged["getiri1h"] = weekly_return
        if daily_return is not None or weekly_return is not None:
            merged["range_returns_source"] = TEFASFON_RETURNS_SOURCE
        merged_rows.append(merged)
    return merged_rows


def _yield_periods_from_points(points: Iterable[Dict[str, Any]], normalized_code: str) -> Dict[str, Dict[str, Any]]:
    ordered = _valid_performance_points(points, normalized_code)
    if not ordered:
        return {}
    latest = ordered[-1]
    latest_date = date.fromisoformat(latest["date"])
    period_targets = {
        "1w": latest_date - timedelta(days=7),
        "1m": latest_date - timedelta(days=30),
        "3m": latest_date - timedelta(days=90),
        "6m": latest_date - timedelta(days=180),
        "ytd": date(latest_date.year, 1, 1),
        "1y": latest_date - timedelta(days=365),
    }
    periods: Dict[str, Dict[str, Any]] = {}
    for key, target in period_targets.items():
        base = _point_on_or_before(ordered, target)
        period_points = [
            point
            for point in ordered
            if target <= date.fromisoformat(point["date"]) <= latest_date and _coerce_float(point.get("price")) is not None
        ]
        prices = [float(point["price"]) for point in period_points if _coerce_float(point.get("price")) is not None]
        if not base and not prices:
            continue
        periods[key] = {
            "prev_close_date": base.get("date") if base else None,
            "prev_close": _coerce_float(base.get("price")) if base else None,
            "high": max(prices) if prices else None,
            "low": min(prices) if prices else None,
        }
    oldest = ordered[0]
    periods["oldest"] = {
        "prev_close_date": oldest.get("date"),
        "prev_close": _coerce_float(oldest.get("price")),
        "high": max(float(point["price"]) for point in ordered),
        "low": min(float(point["price"]) for point in ordered),
    }
    return periods


def _yield_periods_from_return_row(
    row: Dict[str, Any],
    *,
    latest_price: Optional[float],
    latest_date: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    price = _coerce_float(latest_price)
    if price is None or price <= 0:
        return {}
    period_columns = {
        "1w": "getiri1h",
        "1m": "getiri1a",
        "3m": "getiri3a",
        "6m": "getiri6a",
        "ytd": "getiriyb",
        "1y": "getiri1y",
        "3y": "getiri3y",
        "5y": "getiri5y",
    }
    periods: Dict[str, Dict[str, Any]] = {}
    for period, column in period_columns.items():
        return_pct = _coerce_float(row.get(column))
        if return_pct is None:
            continue
        denominator = 1.0 + (return_pct / 100.0)
        if denominator <= 0:
            continue
        periods[period] = {
            "prev_close_date": None,
            "prev_close": price / denominator,
            "high": None,
            "low": None,
            "latest_date": latest_date,
            "return_pct": return_pct,
        }
    return periods


def _is_tefasfon_adapter_unavailable(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "timed out" in message or "not installed" in message or "cannot be imported" in message


class TefasFonClient:
    def __init__(self, *, fund_types: Iterable[str] = TEFAS_FUND_TYPES) -> None:
        normalized_types = [
            str(item).strip().upper()
            for item in fund_types
            if str(item).strip().upper() in _TEFAS_ALLOWED_FUND_TYPES
        ]
        self.fund_types = tuple(normalized_types) or ("SEC",)

    def _module(self) -> Any:
        try:
            import tefasfon  # type: ignore
        except Exception as exc:
            raise TefasUpstreamError("tefasfon package is not installed or cannot be imported") from exc
        return tefasfon

    def _call_dataframe(self, function_name: str, *, context: str, **kwargs: Any) -> List[Dict[str, Any]]:
        module = self._module()
        function = getattr(module, function_name, None)
        if function is None:
            raise TefasFormatError(f"tefasfon missing {function_name}")
        timeout_seconds = max(1.0, TEFAS_TIMEOUT_SECONDS)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tefasfon-{function_name}")
        future = executor.submit(function, **kwargs)
        try:
            frame = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TefasUpstreamError(f"{context} timed out after {timeout_seconds:g}s") from exc
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise TefasUpstreamError(f"{context} failed: {exc}") from exc
        executor.shutdown(wait=False)
        return _dataframe_records(frame, context=context)

    def _call_records(self, callback: Callable[[], List[Dict[str, Any]]], *, context: str) -> List[Dict[str, Any]]:
        timeout_seconds = max(1.0, TEFAS_TIMEOUT_SECONDS)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tefasfon-direct")
        future = executor.submit(callback)
        try:
            records = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TefasUpstreamError(f"{context} timed out after {timeout_seconds:g}s") from exc
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise TefasUpstreamError(f"{context} failed: {exc}") from exc
        executor.shutdown(wait=False)
        return records

    def fetch_funds(
        self,
        *,
        start_date: date,
        end_date: date,
        fund_codes: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if start_date > end_date:
            return []
        codes = sorted({normalize_fund_code(code) for code in list(fund_codes or []) if normalize_fund_code(code)})
        rows: List[Dict[str, Any]] = []
        for fund_type in self.fund_types:
            kwargs: Dict[str, Any] = {
                "fund_type": fund_type,
                "start_date": _tefasfon_date(start_date),
                "end_date": _tefasfon_date(end_date),
            }
            if codes:
                kwargs["fund_codes"] = codes
            records = self._call_dataframe("get_funds", context="tefasfon get_funds", **kwargs)
            rows.extend(_tefasfon_rows(records, source=TEFASFON_FUNDS_SOURCE, fund_type=fund_type))
        return rows

    def fetch_returns(
        self,
        *,
        fund_codes: Optional[Iterable[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        codes = sorted({normalize_fund_code(code) for code in list(fund_codes or []) if normalize_fund_code(code)})
        rows: List[Dict[str, Any]] = []
        for fund_type in self.fund_types:
            kwargs: Dict[str, Any] = {"fund_type": fund_type, "basis": "RB"}
            if start_date and end_date:
                kwargs["start_date"] = _tefasfon_date(start_date)
                kwargs["end_date"] = _tefasfon_date(end_date)
            if codes:
                kwargs["fund_codes"] = codes
            records = self._call_dataframe("get_returns", context="tefasfon get_returns", **kwargs)
            rows.extend(_tefasfon_rows(records, source=TEFASFON_RETURNS_SOURCE, fund_type=fund_type))
        return rows

    def fetch_daily_funds_snapshot(self, as_of: date) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        direct_errors: List[str] = []
        for fund_type in self.fund_types:
            try:
                rows.extend(self._fetch_funds_snapshot_direct(as_of, fund_type=fund_type))
            except TefasUpstreamError as exc:
                direct_errors.append(str(exc))
        if rows:
            return rows
        if direct_errors:
            raise TefasUpstreamError("; ".join(direct_errors))
        return []

    def fetch_portfolio(
        self,
        *,
        fund_code: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        normalized = normalize_fund_code(fund_code)
        if not normalized or start_date > end_date:
            return []
        rows: List[Dict[str, Any]] = []
        direct_errors: List[str] = []
        for fund_type in self.fund_types:
            try:
                rows.extend(self._fetch_portfolio_direct(normalized, start_date, end_date, fund_type=fund_type))
            except TefasUpstreamError as exc:
                direct_errors.append(str(exc))
        if rows or not direct_errors:
            return rows
        for fund_type in self.fund_types:
            records = self._call_dataframe(
                "get_portfolio",
                context="tefasfon get_portfolio",
                fund_type=fund_type,
                start_date=_tefasfon_date(start_date),
                end_date=_tefasfon_date(end_date),
                fund_codes=[normalized],
            )
            normalized_rows = _tefasfon_rows(records, source=TEFASFON_PORTFOLIO_SOURCE, fund_type=fund_type)
            coded_rows = [
                (row, normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")))
                for row in normalized_rows
            ]
            if any(code for _row, code in coded_rows):
                rows.extend(row for row, code in coded_rows if code == normalized)
            else:
                rows.extend(normalized_rows)
        return rows

    def _fetch_funds_snapshot_direct(
        self,
        as_of: date,
        *,
        fund_type: str,
    ) -> List[Dict[str, Any]]:
        fund_type_code = str(fund_type).strip().upper()

        def fetch_records() -> List[Dict[str, Any]]:
            from tefasfon import getter as tefas_getter  # type: ignore

            fon_tipi = tefas_getter._FUND_TIPI[fund_type_code]
            portal_url = tefas_getter._FUND_PORTAL[fund_type_code]
            fund_url_param = tefas_getter._FUND_URL_PARAM[fund_type_code]
            endpoint = tefas_getter._API_ENDPOINT["general_information"]
            as_of_iso = as_of.strftime("%Y-%m-%d")
            session = tefas_getter._new_session(portal_url, as_of_iso, as_of_iso, fund_url_param)
            base_payload = {
                "fonTipi": fon_tipi,
                "fonKodu": None,
                "aramaMetni": None,
                "fonTurKod": None,
                "fonGrubu": None,
                "sfonTurKod": None,
                "basTarih": as_of.strftime("%Y%m%d"),
                "bitTarih": as_of.strftime("%Y%m%d"),
                "basSira": 1,
                "bitSira": 1000,
                "fonTurAciklama": None,
                "dil": "TR",
                "kurucuKod": None,
            }
            return tefas_getter._get_all_pages(session, endpoint, base_payload, portal_url, as_of_iso, as_of_iso)

        records = self._call_records(
            fetch_records,
            context=f"tefasfon direct funds snapshot {as_of.isoformat()}",
        )
        return _tefasfon_rows(records, source=TEFASFON_FUNDS_SOURCE, fund_type=fund_type_code)

    def _fetch_portfolio_direct(
        self,
        fund_code: str,
        start_date: date,
        end_date: date,
        *,
        fund_type: str,
    ) -> List[Dict[str, Any]]:
        try:
            from tefasfon import getter as tefas_getter  # type: ignore
        except Exception as exc:
            raise TefasUpstreamError("tefasfon getter internals are unavailable") from exc
        try:
            fund_type_code = str(fund_type).strip().upper()
            fon_tipi = tefas_getter._FUND_TIPI[fund_type_code]
            portal_url = tefas_getter._FUND_PORTAL[fund_type_code]
            fund_url_param = tefas_getter._FUND_URL_PARAM[fund_type_code]
            endpoint = tefas_getter._API_ENDPOINT["portfolio_breakdown"]
            start_iso = start_date.strftime("%Y-%m-%d")
            end_iso = end_date.strftime("%Y-%m-%d")
            session = tefas_getter._new_session(portal_url, start_iso, end_iso, fund_url_param)
            base_payload = {
                "fonTipi": fon_tipi,
                "fonKodu": fund_code,
                "aramaMetni": fund_code,
                "fonTurKod": None,
                "fonGrubu": None,
                "sfonTurKod": None,
                "basTarih": start_date.strftime("%Y%m%d"),
                "bitTarih": end_date.strftime("%Y%m%d"),
                "basSira": 1,
                "bitSira": 1000,
                "fonTurAciklama": None,
                "dil": "TR",
                "kurucuKod": None,
            }
            records = tefas_getter._get_all_pages(session, endpoint, base_payload, portal_url, start_iso, end_iso)
        except Exception as exc:
            raise TefasUpstreamError(f"tefasfon direct portfolio failed: {exc}") from exc
        rows = _tefasfon_rows(records, source=TEFASFON_PORTFOLIO_SOURCE, fund_type=fund_type_code)
        coded_rows = [
            (row, normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")))
            for row in rows
        ]
        if any(code for _row, code in coded_rows):
            return [row for row, code in coded_rows if code == fund_code]
        return rows

    def fetch_history(self, fund_code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        normalized = normalize_fund_code(fund_code)
        if not normalized or start_date > end_date:
            return []

        def matching_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [
                row
                for row in rows
                if normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")) == normalized
            ]

        day_span = (end_date - start_date).days
        if day_span <= 120:
            return matching_rows(self.fetch_funds(start_date=start_date, end_date=end_date, fund_codes=[normalized]))

        rows_by_date: Dict[str, Dict[str, Any]] = {}

        def add_rows(rows: Iterable[Dict[str, Any]]) -> None:
            for row in matching_rows(rows):
                point_date = _fund_date(_first_present(row, "date", "tarih", "TARIH", "TARIHSTR"))
                if point_date:
                    rows_by_date[point_date] = row

        recent_start = max(start_date, end_date - timedelta(days=35))
        recent_error: Optional[TefasUpstreamError] = None
        try:
            add_rows(self.fetch_funds(start_date=recent_start, end_date=end_date, fund_codes=[normalized]))
        except TefasUpstreamError as exc:
            recent_error = exc

        current_month = _month_start(start_date)
        while current_month <= end_date:
            target_date = min(_month_end(current_month), end_date)
            if target_date >= recent_start:
                current_month = _shift_month(current_month, 1)
                continue
            lower_bound = max(current_month, target_date - timedelta(days=max(0, FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS)))
            current = target_date
            while current >= lower_bound:
                if current.weekday() >= 5:
                    current -= timedelta(days=1)
                    continue
                try:
                    snapshot_rows = self.fetch_daily_funds_snapshot(current)
                except TefasUpstreamError:
                    break
                matched = matching_rows(snapshot_rows)
                if matched:
                    add_rows(matched)
                    break
                current -= timedelta(days=1)
            current_month = _shift_month(current_month, 1)

        if not rows_by_date and recent_error is not None:
            raise recent_error
        return [rows_by_date[point_date] for point_date in sorted(rows_by_date)]

    def fetch_latest_fund_list_snapshot(self, *, as_of: date, lookback_days: int = 10) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        for offset in range(max(1, lookback_days)):
            target_date = as_of - timedelta(days=offset)
            try:
                fund_rows = self.fetch_funds(start_date=target_date, end_date=target_date)
            except TefasUpstreamError as exc:
                warnings.append(f"tefasfon_funds failed for {target_date.isoformat()}: {exc}")
                if _is_tefasfon_adapter_unavailable(exc):
                    break
                continue
            if not fund_rows:
                warnings.append(f"tefasfon_funds returned no rows for {target_date.isoformat()}")
                continue
            try:
                return_rows = self.fetch_returns()
                fund_rows = _merge_tefasfon_returns(fund_rows, return_rows)
            except TefasUpstreamError as exc:
                warnings.append(f"tefasfon_returns failed: {exc}")
            try:
                previous_business_day = target_date - timedelta(days=1)
                while previous_business_day.weekday() >= 5:
                    previous_business_day -= timedelta(days=1)
                daily_return_rows = self.fetch_returns(start_date=previous_business_day, end_date=target_date)
                weekly_return_rows = self.fetch_returns(start_date=target_date - timedelta(days=7), end_date=target_date)
                fund_rows = _merge_tefasfon_range_returns(
                    fund_rows,
                    daily_return_rows=daily_return_rows,
                    weekly_return_rows=weekly_return_rows,
                )
            except TefasUpstreamError as exc:
                warnings.append(f"tefasfon_range_returns failed: {exc}")
            open_rows, skipped_closed, skipped_unknown = _filter_tefas_open_rows(fund_rows)
            if skipped_closed:
                warnings.append(
                    f"tefas_open_only skipped {skipped_closed} closed fund rows"
                )
            if TEFAS_OPEN_ONLY and not open_rows:
                warnings.append(f"tefas_open_only returned no open rows for {target_date.isoformat()}")
                continue
            fund_rows = open_rows
            # Drop the per-day "no rows" lookback noise once we successfully resolved a
            # snapshot. Weekends and Turkish holidays naturally have no TEFAS data and
            # those messages would otherwise be surfaced as a fallback warning banner.
            kept_warnings = [
                w for w in warnings
                if "returned no rows" not in w and "tefas_open_only returned no open rows" not in w
            ]
            return fund_rows, kept_warnings
        return [], warnings

    def fetch_latest_portfolio(
        self,
        fund_code: str,
        *,
        as_of: date,
        lookback_days: int = 10,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        for offset in range(max(1, lookback_days)):
            target_date = as_of - timedelta(days=offset)
            try:
                rows = self.fetch_portfolio(fund_code=fund_code, start_date=target_date, end_date=target_date)
            except TefasUpstreamError as exc:
                warnings.append(f"tefasfon_portfolio failed for {target_date.isoformat()}: {exc}")
                if _is_tefasfon_adapter_unavailable(exc):
                    break
                continue
            if rows:
                return rows, warnings
            warnings.append(f"tefasfon_portfolio returned no rows for {target_date.isoformat()}")
        return [], warnings

    def fetch_yield_summary(
        self,
        fund_code: str,
        *,
        as_of: Optional[date] = None,
        latest_price: Optional[float] = None,
        latest_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized = normalize_fund_code(fund_code)
        raw_returns: List[Dict[str, Any]] = []
        periods: Dict[str, Dict[str, Any]] = {}
        points: List[Dict[str, Any]] = []
        history_error: Optional[BaseException] = None
        effective_end = as_of or date.today()
        raw_returns = self.fetch_returns(fund_codes=[normalized])
        if raw_returns:
            periods.update(
                _yield_periods_from_return_row(
                    raw_returns[0],
                    latest_price=latest_price,
                    latest_date=latest_date,
                )
            )
        if "1w" not in periods:
            try:
                weekly_returns = self.fetch_returns(
                    fund_codes=[normalized],
                    start_date=effective_end - timedelta(days=7),
                    end_date=effective_end,
                )
                if weekly_returns:
                    weekly_row = dict(weekly_returns[0])
                    weekly_row["getiri1h"] = weekly_row.get("getiriOrani")
                    periods.update(
                        _yield_periods_from_return_row(
                            weekly_row,
                            latest_price=latest_price,
                            latest_date=latest_date,
                        )
                    )
            except TefasUpstreamError as exc:
                history_error = exc

        if not periods or "1w" not in periods:
            effective_start = effective_end - timedelta(days=max(370, FUNDS_AUTO_FETCH_LOOKBACK_DAYS))
            try:
                points = self.fetch_history(normalized, effective_start, effective_end)
            except TefasUpstreamError as exc:
                history_error = exc
                points = []
            history_periods = _yield_periods_from_points(points, normalized)
            for key, value in history_periods.items():
                if key not in periods or periods[key].get("prev_close") is None:
                    periods[key] = value
        if not periods:
            if history_error is not None:
                raise history_error
            raise TefasFormatError("tefasfon_funds returned no usable history for yield summary")
        return {
            "fund_code": normalized,
            "source": TEFASFON_RETURNS_SOURCE if raw_returns else TEFASFON_FUNDS_SOURCE,
            "source_url": TEFASFON_SOURCE_URL,
            "periods": periods,
            "raw": {
                "points_count": len(points),
                "returns": raw_returns[:1],
            },
        }


class TefasClient:
    def __init__(
        self,
        *,
        funds_list_endpoint: str = TEFAS_FUNDS_LIST_ENDPOINT,
        timeout_seconds: float = TEFAS_TIMEOUT_SECONDS,
    ) -> None:
        self.funds_list_endpoint = funds_list_endpoint
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Content-Type": "application/json",
            "Origin": TEFAS_BASE_URL,
            "Referer": f"{TEFAS_BASE_URL}/fon-karsilastirma",
            "User-Agent": FINTABLES_USER_AGENT,
        }

    def _fund_list_body(self, *, target_date: date, start_row: int = 1, end_row: int = TEFAS_FUNDS_LIST_PAGE_SIZE) -> Dict[str, Any]:
        tefas_date = target_date.strftime("%Y%m%d")
        return {
            "fonTipi": "YAT",
            "fonKod": None,
            "fonGrup": 0,
            "basTarih": tefas_date,
            "bitTarih": tefas_date,
            "fonTurKod": None,
            "fonUnvanTip": None,
            "sfonTurKod": None,
            "basSira": start_row,
            "bitSira": end_row,
            "fonTurAciklama": None,
            "dil": "TR",
            "kurucuKod": None,
            "sira": None,
            "yon": None,
        }

    def fetch_fund_list_snapshot(self, *, target_date: date) -> List[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.post(
                    self.funds_list_endpoint,
                    json=self._fund_list_body(target_date=target_date),
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise TefasUpstreamError(f"TEFAS fund list request failed: {exc}") from exc
        payload = _decode_tefas_json_response(
            response.status_code,
            dict(response.headers),
            response.content,
            context="TEFAS fund list",
        )
        return _normalize_tefas_fund_list_payload(payload, source_url=self.funds_list_endpoint)

    def fetch_latest_fund_list_snapshot(self, *, as_of: date, lookback_days: int = 10) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        for offset in range(max(1, lookback_days)):
            target_date = as_of - timedelta(days=offset)
            try:
                rows = self.fetch_fund_list_snapshot(target_date=target_date)
            except TefasUpstreamError as exc:
                warnings.append(f"tefas_fund_list failed for {target_date.isoformat()}: {exc}")
                continue
            if rows:
                return rows, warnings
            warnings.append(f"tefas_fund_list returned no rows for {target_date.isoformat()}")
        return [], warnings


def _empty_snapshot_payload(reason: str) -> Dict[str, Any]:
    now = _utc_now_iso()
    return {
        "status": "unavailable",
        "rows": [],
        "count": 0,
        "total_count": 0,
        "source": TEFASFON_FUNDS_SOURCE,
        "source_url": TEFASFON_SOURCE_URL,
        "as_of": None,
        "fetched_at": None,
        "stale": True,
        "degraded": True,
        "warnings": [reason],
        "source_metadata": {
            "source": TEFASFON_FUNDS_SOURCE,
            "source_url": TEFASFON_SOURCE_URL,
            "fetched_at": None,
            "as_of": None,
            "cache_hit": False,
            "stale": True,
            "parse_status": "unavailable",
            "tefas_open_only": TEFAS_OPEN_ONLY,
            "warnings": [reason],
            "served_at": now,
        },
    }


def load_funds_snapshot(processed_dir: Path) -> Dict[str, Any]:
    path = _snapshot_path(processed_dir)
    cache_key = f"snapshot:{path}"
    stat = path.stat() if path.exists() else None
    cached = _MEMORY_CACHE.get(cache_key)
    if cached and stat and cached.get("mtime") == stat.st_mtime:
        payload = dict(cached["payload"])
    else:
        payload = _read_json(path) or _empty_snapshot_payload("fund snapshot cache is empty")
        if stat:
            _MEMORY_CACHE[cache_key] = {"mtime": stat.st_mtime, "payload": payload}
    fetched_at = payload.get("fetched_at")
    age = _cache_age_seconds(fetched_at)
    stale = bool(payload.get("stale")) or age is None or age > FUNDS_SNAPSHOT_TTL_SECONDS
    meta = dict(payload.get("source_metadata") or {})
    public_source = _public_price_source(str(payload.get("source") or meta.get("source") or TEFASFON_FUNDS_SOURCE))
    meta["source"] = public_source
    if public_source == "legacy_cache":
        meta["source_url"] = None
    meta["cache_hit"] = bool(stat)
    meta["stale"] = stale
    meta["tefas_open_only"] = TEFAS_OPEN_ONLY
    payload = dict(payload)
    payload["source"] = public_source
    if public_source == "legacy_cache":
        payload["source_url"] = None
    payload["rows"] = [
        {**row, "source": _public_price_source(str(row.get("source") or public_source))}
        if isinstance(row, dict)
        else row
        for row in list(payload.get("rows") or [])
    ]
    payload["stale"] = stale
    payload["source_metadata"] = meta
    return payload


def _target_fund_codes_from_snapshot_payload(snapshot: Dict[str, Any]) -> List[str]:
    codes = {
        normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict)
        and _is_target_fund_row(row)
        and _is_tefas_open_row(row, require_known=True)
    }
    return sorted(code for code in codes if code)


def _target_fund_codes_from_env() -> List[str]:
    return sorted({normalize_fund_code(code) for code in TARGET_FUND_CODES if normalize_fund_code(code)})


def _tefas_open_codes_from_snapshot_payload(snapshot: Dict[str, Any]) -> set[str]:
    return {
        normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict) and _is_tefas_open_row(row, require_known=True)
    }


def _snapshot_rows_by_code(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or "")): row
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict) and normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
    }


def _enrich_points_from_snapshot(points: Iterable[Dict[str, Any]], snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows_by_code = _snapshot_rows_by_code(snapshot)
    enriched: List[Dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        code = normalize_fund_code(str(point.get("fund_code") or ""))
        meta = rows_by_code.get(code, {})
        merged = dict(point)
        for key in (
            "name",
            "fund_type",
            "founder_company",
            "manager_company",
            "tefas_open",
            "risk_value",
            "currency",
            "isin",
        ):
            if merged.get(key) is None and meta.get(key) is not None:
                merged[key] = meta.get(key)
        enriched.append(merged)
    return enriched


def _target_fund_codes_for_collection(processed_dir: Path, *, lookback_days: int) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    snapshot = load_funds_snapshot(processed_dir)
    env_codes = set(_target_fund_codes_from_env())
    snapshot_codes = set(_target_fund_codes_from_snapshot_payload(snapshot))
    if TEFAS_OPEN_ONLY and env_codes and snapshot.get("rows"):
        open_codes = _tefas_open_codes_from_snapshot_payload(snapshot)
        skipped_env_codes = sorted(code for code in env_codes if code not in open_codes)
        if skipped_env_codes:
            warnings.append(
                "target fund codes skipped because they are not in the TEFAS-open snapshot: "
                + ", ".join(skipped_env_codes[:20])
            )
        env_codes = {code for code in env_codes if code in open_codes}
    codes = sorted(env_codes | snapshot_codes)
    if codes:
        return codes, warnings
    warnings.append("no target fund codes available; set RAGFIN_TARGET_FUND_CODES or keep a fund snapshot cache")
    return codes, warnings


def _fetch_fintables_udf_history_for_codes(
    fund_codes: Iterable[str],
    *,
    start_date: date,
    end_date: date,
    client: Optional[FintablesClient] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    fintables = client or FintablesClient()
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for fund_code in sorted({normalize_fund_code(code) for code in fund_codes if normalize_fund_code(code)}):
        try:
            rows.extend(fintables.fetch_udf_history(fund_code, start_date, end_date))
        except FintablesUpstreamError as exc:
            warnings.append(f"fintables_udf_history failed for {fund_code}: {exc}")
    return rows, warnings


def _fetch_tefasfon_history_for_codes(
    fund_codes: Iterable[str],
    *,
    start_date: date,
    end_date: date,
    client: Optional[TefasFonClient] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    codes = sorted({normalize_fund_code(code) for code in fund_codes if normalize_fund_code(code)})
    if not codes:
        return [], []
    tefas = client or TefasFonClient()
    try:
        return tefas.fetch_funds(start_date=start_date, end_date=end_date, fund_codes=codes), []
    except TefasUpstreamError as exc:
        return [], [f"tefasfon_funds failed: {exc}"]


def _fetch_tefasfon_daily_snapshots_for_codes(
    fund_codes: Iterable[str],
    *,
    start_date: date,
    end_date: date,
    client: Optional[TefasFonClient] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    codes = sorted({normalize_fund_code(code) for code in fund_codes if normalize_fund_code(code)})
    if not codes or start_date > end_date:
        return [], []
    wanted = set(codes)
    tefas = client or TefasFonClient()
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    current = start_date
    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        try:
            if hasattr(tefas, "fetch_daily_funds_snapshot"):
                day_rows = tefas.fetch_daily_funds_snapshot(current)  # type: ignore[attr-defined]
            else:
                day_rows = tefas.fetch_funds(start_date=current, end_date=current)
        except TefasUpstreamError as exc:
            warnings.append(f"tefasfon_funds daily snapshot failed for {current.isoformat()}: {exc}")
            current += timedelta(days=1)
            continue
        if not day_rows:
            warnings.append(f"tefasfon_funds daily snapshot returned no rows for {current.isoformat()}")
            current += timedelta(days=1)
            continue
        rows.extend(
            row
            for row in day_rows
            if normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")) in wanted
        )
        current += timedelta(days=1)
    return rows, warnings


def refresh_funds_snapshot(processed_dir: Path, *, lookback_days: int = 10) -> Dict[str, Any]:
    end_date = date.today()
    existing_snapshot = load_funds_snapshot(processed_dir)
    rows, warnings = TefasFonClient().fetch_latest_fund_list_snapshot(
        as_of=end_date,
        lookback_days=max(1, lookback_days),
    )
    rows, skipped_closed, skipped_unknown = _filter_tefas_open_rows(rows)
    if skipped_closed:
        warnings.append(
            f"tefas_open_only skipped {skipped_closed} closed fund rows"
        )
    if not rows:
        payload = dict(existing_snapshot)
        existing_warnings = list(payload.get("warnings") or [])
        all_warnings = existing_warnings + warnings
        meta = dict(payload.get("source_metadata") or {})
        meta["parse_status"] = "empty_tefasfon_funds"
        meta["warnings"] = all_warnings
        meta["stale"] = True
        meta["cache_hit"] = bool(payload.get("rows"))
        meta["source_policy"] = FUND_HISTORY_SOURCE_POLICY
        meta["fallback_used"] = bool(payload.get("rows"))
        meta["tefas_open_only"] = TEFAS_OPEN_ONLY
        payload["status"] = payload.get("status") or "unavailable"
        payload["stale"] = True
        payload["degraded"] = bool(payload.get("degraded")) or not bool(payload.get("rows"))
        payload["warnings"] = all_warnings
        payload["source_metadata"] = meta
        return payload
    snapshot = _build_snapshot(
        rows,
        warnings=warnings,
        source=TEFASFON_FUNDS_SOURCE,
        source_url=TEFASFON_SOURCE_URL,
        parse_status="ok_tefasfon_funds",
    )
    snapshot["source_metadata"]["source_policy"] = FUND_HISTORY_SOURCE_POLICY
    snapshot["source_metadata"]["fallback_used"] = False
    if not snapshot["rows"]:
        return {
            **existing_snapshot,
            "stale": True,
            "degraded": True,
            "warnings": list(existing_snapshot.get("warnings") or []) + warnings + ["tefasfon_funds returned no valid fund rows"],
        }
    upsert_fund_price_points(
        processed_dir,
        rows,
        source=TEFASFON_FUNDS_SOURCE,
        fetched_at=str(snapshot.get("fetched_at") or _utc_now_iso()),
    )
    _write_json(_snapshot_path(processed_dir), snapshot)
    return snapshot


def collect_daily_fund_prices(
    processed_dir: Path,
    *,
    as_of: Optional[date] = None,
    lookback_days: int = FUNDS_COLLECTOR_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    effective_as_of = as_of or date.today()
    start_date = effective_as_of - timedelta(days=max(1, lookback_days))
    existing_snapshot = load_funds_snapshot(processed_dir)
    fetched_at = _utc_now_iso()
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    source = TEFASFON_FUNDS_SOURCE
    source_url = TEFASFON_SOURCE_URL
    raw_row_count = 0
    skipped_by_manager_count = 0
    fallback_attempted = False
    large_universe_daily_snapshots = False

    target_codes, code_warnings = _target_fund_codes_for_collection(processed_dir, lookback_days=max(1, lookback_days))
    warnings.extend(code_warnings)
    if target_codes:
        if len(target_codes) > 100:
            large_universe_daily_snapshots = True
            tefas_rows, tefas_warnings = _fetch_tefasfon_daily_snapshots_for_codes(
                target_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
            warnings.append("tefasfon_funds used daily all-fund snapshots for large target universe")
        else:
            tefas_rows, tefas_warnings = _fetch_tefasfon_history_for_codes(
                target_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
        warnings.extend(tefas_warnings)
        if not tefas_rows:
            snapshot_rows, snapshot_warnings = _fetch_tefasfon_daily_snapshots_for_codes(
                target_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
            if snapshot_rows:
                warnings.append("tefasfon_funds code-filtered history was empty; used daily all-fund snapshots")
                tefas_rows = snapshot_rows
            warnings.extend(snapshot_warnings)
        tefas_codes = {
            point["fund_code"]
            for row in tefas_rows
            for point in [_normalize_history_row(row)]
            if point is not None and _coerce_float(point.get("price")) is not None and _coerce_float(point.get("price")) > 0
        }
        missing_codes = sorted(set(target_codes) - {code for code in tefas_codes if code})
        fintables_rows: List[Dict[str, Any]] = []
        if missing_codes and not large_universe_daily_snapshots:
            fallback_attempted = True
            fintables_rows, fintables_warnings = _fetch_fintables_udf_history_for_codes(
                missing_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
            warnings.extend(fintables_warnings)
        elif missing_codes:
            warnings.append(
                f"fintables_udf_history fallback skipped for {len(missing_codes)} missing codes in large TEFAS snapshot collection"
            )
        rows = list(tefas_rows) + list(fintables_rows)
        if not tefas_rows and fintables_rows:
            source = FINTABLES_UDF_HISTORY_SOURCE
            source_url = FINTABLES_UDF_HISTORY_ENDPOINT
        raw_row_count = len(tefas_rows) + len(fintables_rows)
        if missing_codes:
            warnings.append(f"fintables_udf_history fallback requested for {len(missing_codes)} missing TEFAS fund codes")
    else:
        warnings.append("no target fund codes available for TEFAS/Fintables collection")

    rows = _enrich_points_from_snapshot(rows, existing_snapshot)
    snapshot_rows = _dedupe_price_points(rows)

    storage_result = upsert_fund_price_points(
        processed_dir,
        rows,
        source=source,
        fetched_at=fetched_at,
    )
    snapshot = _build_snapshot(
        snapshot_rows,
        warnings=warnings,
        source=source,
        source_url=source_url,
        parse_status="ok_collector" if rows else "empty_collector",
    )
    snapshot["source_metadata"]["source_policy"] = FUND_HISTORY_SOURCE_POLICY
    snapshot["source_metadata"]["fallback_used"] = fallback_attempted
    if snapshot_rows:
        _write_json(_snapshot_path(processed_dir), snapshot)

    skipped_warnings = [
        str(item.get("warning") or "invalid_price_row")
        for item in list(storage_result.get("warnings") or [])
    ]
    all_warnings = warnings + skipped_warnings
    valid_dates = [
        point["date"]
        for row in rows
        for point in [_normalize_history_row(row)]
        if point is not None and _coerce_float(point.get("price")) is not None and _coerce_float(point.get("price")) > 0
    ]
    storage_as_of = max(valid_dates) if valid_dates else snapshot.get("as_of")
    return {
        "status": "ok" if storage_result.get("upserted_count") else ("empty" if rows else "unavailable"),
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": effective_as_of.isoformat(),
        "source": source,
        "source_url": source_url,
        "fetched_at": fetched_at,
        "row_count": len(rows),
        "raw_row_count": raw_row_count,
        "skipped_by_manager_count": skipped_by_manager_count,
        "valid_point_count": int(storage_result.get("upserted_count") or 0),
        "skipped_point_count": int(storage_result.get("skipped_count") or 0),
        "warning_count": len(all_warnings),
        "warnings": all_warnings[:50],
        "db_path": storage_result.get("db_path") or str(_fund_prices_db_path(processed_dir)),
        "as_of": storage_as_of,
        "source_metadata": {
            "source": source,
            "source_url": source_url,
            "db_path": storage_result.get("db_path") or str(_fund_prices_db_path(processed_dir)),
            "fetched_at": fetched_at,
            "as_of": storage_as_of,
            "parse_status": "ok" if storage_result.get("upserted_count") else "empty",
            "source_policy": FUND_HISTORY_SOURCE_POLICY,
            "fallback_used": fallback_attempted,
            "tefas_open_only": TEFAS_OPEN_ONLY,
            "warnings": all_warnings[:50],
            "storage": {
                "sources": storage_result.get("sources") or {},
                "upserted_count": storage_result.get("upserted_count"),
                "skipped_count": storage_result.get("skipped_count"),
                "warning_count": storage_result.get("warning_count"),
            },
        },
    }


def _row_matches(row: Dict[str, Any], *, q: Optional[str], fund_type: Optional[str], founder: Optional[str], manager: Optional[str], risk: Optional[str]) -> bool:
    if q:
        needle = q.strip().lower()
        haystack = " ".join(
            str(row.get(key) or "")
            for key in ("fund_code", "name", "fund_type", "founder_company", "manager_company")
        ).lower()
        if needle not in haystack:
            return False
    if fund_type and str(row.get("fund_type") or "").strip().lower() != fund_type.strip().lower():
        return False
    if founder and str(row.get("founder_company") or "").strip().lower() != founder.strip().lower():
        return False
    if manager and str(row.get("manager_company") or "").strip().lower() != manager.strip().lower():
        return False
    if risk and str(row.get("risk_value") or "").strip().lower() != risk.strip().lower():
        return False
    return True


def _sort_rows(rows: List[Dict[str, Any]], sort: str, order: str) -> List[Dict[str, Any]]:
    allowed = {
        "fund_code",
        "name",
        "fund_type",
        "founder_company",
        "manager_company",
        "price",
        "daily_return",
        "risk_value",
        "aum",
        "investor_count",
        "as_of",
    }
    key = sort if sort in allowed else "fund_code"
    descending = order.lower() == "desc"

    def sort_key(row: Dict[str, Any]) -> Tuple[int, Any]:
        value = row.get(key)
        if value is None:
            return (1, "")
        if isinstance(value, (int, float)):
            return (0, -value if descending else value)
        return (0, str(value).lower())

    ordered = sorted(rows, key=sort_key)
    if descending and ordered and not isinstance(next((row.get(key) for row in ordered if row.get(key) is not None), ""), (int, float)):
        valued = [row for row in ordered if row.get(key) is not None]
        missing = [row for row in ordered if row.get(key) is None]
        valued.reverse()
        return valued + missing
    return ordered


def get_funds_payload(
    processed_dir: Path,
    *,
    q: Optional[str] = None,
    fund_type: Optional[str] = None,
    founder: Optional[str] = None,
    manager: Optional[str] = None,
    risk: Optional[str] = None,
    sort: str = "fund_code",
    order: str = "asc",
) -> Dict[str, Any]:
    snapshot = load_funds_snapshot(processed_dir)
    rows = [
        row
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict)
        and _is_target_fund_row(row)
        and _is_tefas_open_row(row)
        and _row_matches(row, q=q, fund_type=fund_type, founder=founder, manager=manager, risk=risk)
    ]
    rows = _sort_rows(rows, sort, order)
    return {
        "status": snapshot.get("status") or ("ok" if rows else "empty"),
        "rows": rows,
        "count": len(rows),
        "total_count": len(snapshot.get("rows") or []),
        "source": _public_price_source(str(snapshot.get("source") or TEFASFON_FUNDS_SOURCE)),
        "source_url": snapshot.get("source_url", TEFASFON_SOURCE_URL),
        "as_of": snapshot.get("as_of"),
        "fetched_at": snapshot.get("fetched_at"),
        "stale": bool(snapshot.get("stale")),
        "degraded": bool(snapshot.get("degraded")),
        "warnings": list(snapshot.get("warnings") or []),
        "source_metadata": snapshot.get("source_metadata") or {},
    }


def get_fund_categories_payload(processed_dir: Path) -> Dict[str, Any]:
    snapshot = load_funds_snapshot(processed_dir)
    rows = [
        row
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict) and _is_target_fund_row(row) and _is_tefas_open_row(row)
    ]

    def unique(key: str) -> List[str]:
        values = sorted({str(row.get(key)).strip() for row in rows if row.get(key)})
        return values

    risk_values = sorted({int(row["risk_value"]) for row in rows if isinstance(row.get("risk_value"), int)})
    return {
        "status": snapshot.get("status") or "unavailable",
        "fund_types": unique("fund_type"),
        "founder_companies": unique("founder_company"),
        "manager_companies": unique("manager_company"),
        "risk_values": risk_values,
        "source_metadata": snapshot.get("source_metadata") or {},
    }


def _find_fund_row(processed_dir: Path, fund_code: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_fund_code(fund_code)
    snapshot = load_funds_snapshot(processed_dir)
    for row in list(snapshot.get("rows") or []):
        if (
            isinstance(row, dict)
            and _is_target_fund_row(row)
            and _is_tefas_open_row(row)
            and normalize_fund_code(str(row.get("fund_code") or "")) == normalized
        ):
            return row
    return None


def get_fund_detail_payload(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    row = _find_fund_row(processed_dir, normalized)
    if not row:
        raise KeyError(normalized)
    snapshot = load_funds_snapshot(processed_dir)
    return {
        **row,
        "isin": row.get("isin"),
        "strategy": None,
        "benchmark": None,
        "management_fee": None,
        "tax_info": None,
        "fintables_url": f"{FINTABLES_FUND_BASE_URL}/{normalized}",
        "kap_url": None,
        "source_metadata": snapshot.get("source_metadata") or {},
    }


def get_fund_yield_summary_payload(fund_code: str, processed_dir: Optional[Path] = None) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    warnings: List[str] = []
    fallback_used = False
    source = TEFASFON_FUNDS_SOURCE
    source_url = TEFASFON_SOURCE_URL
    summary_source_used = TEFASFON_FUNDS_SOURCE
    latest_price: Optional[float] = None
    latest_date: Optional[str] = None
    if processed_dir is not None:
        latest_row = _find_fund_row(processed_dir, normalized)
        if latest_row:
            latest_price = _coerce_float(latest_row.get("price"))
            latest_date = _fund_date(latest_row.get("as_of"))
    try:
        summary = TefasFonClient().fetch_yield_summary(
            normalized,
            latest_price=latest_price,
            latest_date=latest_date,
        )
        source = _public_price_source(str(summary.get("source") or source))
        source_url = str(summary.get("source_url") or source_url)
        summary_source_used = source
    except TefasUpstreamError as exc:
        warnings.append(f"tefasfon_yield_summary failed: {exc}")
        fallback_used = True
        source = FINTABLES_YIELD_SUMMARY_SOURCE
        source_url = FINTABLES_YIELD_SUMMARY_ENDPOINT
        summary_source_used = FINTABLES_YIELD_SUMMARY_SOURCE
        try:
            summary = fetch_fintables_yield_summary(normalized)
        except FintablesUpstreamError as fallback_exc:
            warnings.append(f"fintables_yield_summary failed: {fallback_exc}")
            summary = {
                "fund_code": normalized,
                "source": FINTABLES_YIELD_SUMMARY_SOURCE,
                "source_url": FINTABLES_YIELD_SUMMARY_ENDPOINT,
                "periods": {},
                "raw": {},
            }
    periods = summary.get("periods") or {}
    return {
        "fund_code": normalized,
        "status": "ok" if periods else ("unavailable" if warnings else "empty"),
        "source": source,
        "source_url": source_url,
        "periods": periods,
        "source_metadata": {
            "source": source,
            "source_url": source_url,
            "fetched_at": _utc_now_iso(),
            "purpose": "period_summary_only",
            "writes_fund_prices": False,
            "summary_source_used": summary_source_used,
            "source_policy": FUND_HISTORY_SOURCE_POLICY,
            "fallback_used": fallback_used,
            "adapter_version": _tefasfon_adapter_version() if source.startswith("tefasfon") else None,
            "warnings": warnings,
            "warning": warnings[0] if warnings else None,
        },
    }


def _valid_performance_points(points: Iterable[Dict[str, Any]], normalized_code: str) -> List[Dict[str, Any]]:
    valid: Dict[str, Dict[str, Any]] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        point_date = _fund_date(point.get("date"))
        price = _coerce_float(point.get("price"))
        if not point_date or price is None or price <= 0:
            continue
        valid[point_date] = {
            **point,
            "fund_code": normalize_fund_code(str(point.get("fund_code") or normalized_code)),
            "date": point_date,
            "price": price,
            "daily_return": _coerce_float(point.get("daily_return")),
            "aum": _coerce_float(point.get("aum")),
            "investor_count": _coerce_int(point.get("investor_count")),
            "share_count": _coerce_float(point.get("share_count")),
            "source": _public_price_source(str(point.get("source") or TEFASFON_FUNDS_SOURCE)),
        }
    ordered = [valid[key] for key in sorted(valid)]
    previous_price: Optional[float] = None
    for point in ordered:
        current_price = _coerce_float(point.get("price"))
        if point.get("daily_return") is None and previous_price is not None:
            point["daily_return"] = _return_between(current_price, previous_price)
        if current_price is not None:
            previous_price = current_price
    return ordered


def _legacy_history_points(processed_dir: Path, normalized_code: str) -> List[Dict[str, Any]]:
    payload = _read_json(_history_path(processed_dir, normalized_code)) or {}
    return _valid_performance_points(list(payload.get("points") or []), normalized_code)


def _migrate_legacy_history_to_sqlite(processed_dir: Path, normalized_code: str) -> Dict[str, Any]:
    legacy_points = _legacy_history_points(processed_dir, normalized_code)
    if not legacy_points:
        return {
            "upserted_count": 0,
            "skipped_count": 0,
            "warning_count": 0,
            "warnings": [],
        }
    return upsert_fund_price_points(
        processed_dir,
        legacy_points,
        source="legacy_json",
        fallback_code=normalized_code,
    )


def _fund_performance_payload_from_points(
    processed_dir: Path,
    normalized_code: str,
    points: List[Dict[str, Any]],
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    fetched_at: Optional[str] = None,
    cache_hit: bool = True,
    stale: Optional[bool] = None,
    parse_status: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    fetched_point_count: Optional[int] = None,
    storage_result: Optional[Dict[str, Any]] = None,
    backfill_used: bool = False,
    fallback_used: bool = False,
    fallback_reason: Optional[str] = None,
    overview_metric_backfill: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ordered = _valid_performance_points(points, normalized_code)
    latest_point_date = ordered[-1]["date"] if ordered else None
    latest_fetch = fetched_at or max(
        (str(point.get("fetched_at") or "") for point in ordered if point.get("fetched_at")),
        default=None,
    )
    if stale is None:
        stale = (_cache_age_seconds(latest_fetch) or (FUNDS_HISTORY_TTL_SECONDS + 1)) > FUNDS_HISTORY_TTL_SECONDS
    effective_end = end_date or (date.fromisoformat(latest_point_date) if latest_point_date else date.today())
    coverage = _history_coverage_info(ordered, effective_end)
    coverage_warnings = list(coverage.get("warnings") or [])
    internal_gap_warnings = _history_internal_gap_warnings(ordered)
    all_warnings = list(warnings or []) + coverage_warnings + internal_gap_warnings
    date_min = ordered[0]["date"] if ordered else None
    date_max = ordered[-1]["date"] if ordered else None
    history_source_used = _dominant_price_source(ordered) or TEFASFON_FUNDS_SOURCE
    fallback_point_count = sum(
        1
        for point in ordered
        if _normalize_price_source(str(point.get("source") or "")) == FINTABLES_UDF_HISTORY_SOURCE
    )
    metadata: Dict[str, Any] = {
        "source": "sqlite",
        "source_url": str(_fund_prices_db_path(processed_dir)),
        "db_path": str(_fund_prices_db_path(processed_dir)),
        "fetched_at": latest_fetch,
        "as_of": latest_point_date,
        "cache_hit": cache_hit,
        "stale": stale,
        "parse_status": parse_status or ("ok" if ordered else "empty"),
        "warnings": all_warnings,
        "warning": all_warnings[0] if all_warnings else None,
        "history_source_used": history_source_used,
        "history_source_policy": FUND_HISTORY_SOURCE_POLICY,
        "source_policy": FUND_HISTORY_SOURCE_POLICY,
        "primary_source": "tefasfon",
        "tefasfon_adapter_version": _tefasfon_adapter_version(),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "cached_fallback_points_present": fallback_point_count > 0,
        "cached_fallback_point_count": fallback_point_count,
        "final_points_count": len(ordered),
        "date_min": date_min,
        "date_max": date_max,
        "backfill_used": backfill_used,
        "latest_point_date": coverage.get("latest_point_date"),
        "coverage_gap_days": coverage.get("coverage_gap_days"),
        "coverage_gap_business_days": coverage.get("coverage_gap_business_days"),
        "internal_gap_count": len(internal_gap_warnings),
    }
    if start_date:
        metadata["requested_start_date"] = start_date.isoformat()
    if end_date:
        metadata["requested_end_date"] = end_date.isoformat()
    if fetched_point_count is not None:
        metadata["fetched_point_count"] = fetched_point_count
    if storage_result:
        metadata["storage"] = {
            "source": storage_result.get("source"),
            "sources": storage_result.get("sources") or {},
            "upserted_count": storage_result.get("upserted_count"),
            "skipped_count": storage_result.get("skipped_count"),
            "warning_count": storage_result.get("warning_count"),
        }
    if overview_metric_backfill is not None:
        metadata["overview_metric_backfill"] = overview_metric_backfill

    return {
        "fund_code": normalized_code,
        "status": "ok" if ordered else "empty",
        "points": ordered,
        "source": "sqlite",
        "source_url": str(_fund_prices_db_path(processed_dir)),
        "as_of": latest_point_date,
        "fetched_at": latest_fetch,
        "stale": bool(stale),
        "source_metadata": metadata,
    }


def _has_requested_price_coverage(
    points: List[Dict[str, Any]],
    *,
    start_date: Optional[date],
    end_date: Optional[date],
) -> bool:
    if not points:
        return False
    parsed_dates = [
        date.fromisoformat(point["date"])
        for point in points
        if isinstance(point.get("date"), str)
    ]
    if not parsed_dates:
        return False
    if start_date and min(parsed_dates) > start_date:
        if _business_days_between(start_date, min(parsed_dates) - timedelta(days=1)) > 3:
            return False
    if _history_internal_gap_warnings(points):
        return False
    if end_date:
        latest = max(parsed_dates)
        gap_days = max(0, (end_date - latest).days)
        if gap_days > 3 and _business_days_between(latest + timedelta(days=1), end_date) > 3:
            return False
    return True


def _history_cache_covers_requested_span(
    processed_dir: Path,
    normalized_code: str,
    *,
    start_date: date,
    end_date: date,
) -> bool:
    payload = _read_json(_history_path(processed_dir, normalized_code)) or {}
    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    if str(metadata.get("parse_status") or "").startswith("unavailable"):
        return False
    requested_start = _fund_date(metadata.get("requested_start_date"))
    if not requested_start:
        return False
    try:
        if date.fromisoformat(requested_start) > start_date:
            return False
    except ValueError:
        return False
    latest_point_date = _fund_date(metadata.get("date_max") or metadata.get("as_of") or payload.get("as_of"))
    if not latest_point_date:
        return False
    try:
        latest = date.fromisoformat(latest_point_date)
    except ValueError:
        return False
    gap_days = max(0, (end_date - latest).days)
    return gap_days <= 3 or _business_days_between(latest + timedelta(days=1), end_date) <= 3


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _shift_month(month_start: date, offset: int) -> date:
    month_index = month_start.year * 12 + (month_start.month - 1) + offset
    return date(month_index // 12, (month_index % 12) + 1, 1)


def _month_end(month_start: date) -> date:
    next_month = _shift_month(month_start, 1)
    return next_month - timedelta(days=1)


def _month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _fund_point_date(point: Dict[str, Any]) -> Optional[date]:
    raw_date = point.get("date")
    if not isinstance(raw_date, str):
        return None
    try:
        return date.fromisoformat(raw_date)
    except ValueError:
        return None


def _overview_metric_targets(end_date: date) -> List[Dict[str, Any]]:
    visible_months = max(1, FUNDS_OVERVIEW_METRIC_MONTHS)
    end_month = _month_start(end_date)
    targets: List[Dict[str, Any]] = []
    for offset in range(-visible_months, 1):
        month_start = _shift_month(end_month, offset)
        target_date = min(_month_end(month_start), end_date)
        targets.append(
            {
                "month": _month_key(month_start),
                "month_start": month_start,
                "target_date": target_date,
            }
        )
    return targets


def _latest_point_for_month(
    points: List[Dict[str, Any]],
    *,
    month: str,
    target_date: date,
) -> Optional[Dict[str, Any]]:
    latest: Optional[Tuple[date, Dict[str, Any]]] = None
    for point in points:
        point_date = _fund_point_date(point)
        if point_date is None or _month_key(point_date) != month or point_date > target_date:
            continue
        if latest is None or point_date > latest[0]:
            latest = (point_date, point)
    return latest[1] if latest else None


def _has_overview_metrics(point: Optional[Dict[str, Any]]) -> bool:
    if point is None:
        return False
    aum = _coerce_float(point.get("aum"))
    investor_count = _coerce_int(point.get("investor_count"))
    return aum is not None and aum > 0 and investor_count is not None


def _missing_overview_metric_targets(points: List[Dict[str, Any]], *, end_date: date) -> List[Dict[str, Any]]:
    missing: List[Dict[str, Any]] = []
    for target in _overview_metric_targets(end_date):
        point = _latest_point_for_month(
            points,
            month=str(target["month"]),
            target_date=target["target_date"],
        )
        if not _has_overview_metrics(point):
            missing.append(target)
    return missing


def _fetch_fund_overview_metric_rows(
    fund_code: str,
    targets: List[Dict[str, Any]],
    *,
    client: TefasFonClient,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    normalized = normalize_fund_code(fund_code)
    rows: List[Dict[str, Any]] = []
    fetched_months: Dict[str, str] = {}
    warnings: List[str] = []

    for target in targets:
        month = str(target["month"])
        target_date = target["target_date"]
        lower_bound = max(target["month_start"], target_date - timedelta(days=max(0, FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS)))
        current = target_date
        matched = False
        while current >= lower_bound:
            if current.weekday() >= 5:
                current -= timedelta(days=1)
                continue
            try:
                snapshot_rows = client.fetch_daily_funds_snapshot(current)
            except TefasUpstreamError as exc:
                warnings.append(f"tefasfon_funds overview metric snapshot failed for {current.isoformat()}: {exc}")
                break
            match = next(
                (
                    row
                    for row in snapshot_rows
                    if normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")) == normalized
                ),
                None,
            )
            if match:
                rows.append(match)
                fetched_months[month] = current.isoformat()
                matched = True
                break
            current -= timedelta(days=1)
        if not matched:
            warnings.append(
                f"tefasfon_funds overview metric row not found for {normalized} around {target_date.isoformat()}"
            )

    return rows, fetched_months, warnings


def _auto_fetch_key(
    processed_dir: Path,
    fund_code: str,
    start_date: date,
    end_date: date,
    *,
    namespace: str = "history",
) -> str:
    return ":".join(
        [
            str(_fund_prices_db_path(processed_dir)),
            namespace,
            normalize_fund_code(fund_code),
            start_date.isoformat(),
            end_date.isoformat(),
        ]
    )


def _recent_auto_fetch_failure(key: str) -> Optional[str]:
    with _AUTO_FETCH_LOCK:
        cached = _AUTO_FETCH_NEGATIVE_CACHE.get(key)
        if not cached:
            return None
        expires_at = cached.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at > _utc_now():
            return str(cached.get("error") or "recent fund history failure")
        _AUTO_FETCH_NEGATIVE_CACHE.pop(key, None)
    return None


def _remember_auto_fetch_failure(key: str, error: str) -> None:
    with _AUTO_FETCH_LOCK:
        _AUTO_FETCH_NEGATIVE_CACHE[key] = {
            "error": error,
            "expires_at": _utc_now() + timedelta(seconds=max(1, FUNDS_AUTO_FETCH_NEGATIVE_TTL_SECONDS)),
        }


def _auto_refresh_fund_performance(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: date,
    end_date: date,
) -> List[str]:
    normalized = normalize_fund_code(fund_code)
    key = _auto_fetch_key(processed_dir, normalized, start_date, end_date)
    recent_failure = _recent_auto_fetch_failure(key)
    if recent_failure:
        return [f"fund history auto fetch skipped after recent failure: {recent_failure}"]

    owner = False
    with _AUTO_FETCH_LOCK:
        event = _AUTO_FETCH_IN_FLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _AUTO_FETCH_IN_FLIGHT[key] = event
            owner = True

    if not owner:
        waited = event.wait(timeout=max(1.0, FINTABLES_TIMEOUT_SECONDS + 5.0))
        if not waited:
            return ["fund history auto fetch already in progress"]
        recent_failure = _recent_auto_fetch_failure(key)
        if recent_failure:
            return [f"fund history auto fetch skipped after recent failure: {recent_failure}"]
        return []

    try:
        payload = refresh_fund_performance(
            processed_dir,
            normalized,
            start_date=start_date,
            end_date=end_date,
        )
        if not payload.get("points"):
            error = "fund history returned no valid points"
            _remember_auto_fetch_failure(key, error)
            return [f"fund history auto fetch failed: {error}"]
        return []
    except FundUpstreamError as exc:
        error = str(exc)
        _remember_auto_fetch_failure(key, error)
        return [f"fund history auto fetch failed: {error}"]
    finally:
        with _AUTO_FETCH_LOCK:
            current = _AUTO_FETCH_IN_FLIGHT.get(key)
            if current is event:
                _AUTO_FETCH_IN_FLIGHT.pop(key, None)
            event.set()


def _auto_refresh_fund_overview_metrics(
    processed_dir: Path,
    fund_code: str,
    *,
    points: List[Dict[str, Any]],
    end_date: date,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    missing_targets = _missing_overview_metric_targets(points, end_date=end_date)
    metadata: Dict[str, Any] = {
        "attempted": False,
        "missing_months": [str(target["month"]) for target in missing_targets],
        "fetched_months": {},
        "upserted_count": 0,
        "skipped_count": 0,
        "warning_count": 0,
        "warnings": [],
    }
    if not normalized or not missing_targets:
        return metadata

    client = TefasFonClient()
    if not hasattr(client, "fetch_daily_funds_snapshot"):
        return metadata

    key = _auto_fetch_key(
        processed_dir,
        normalized,
        missing_targets[0]["target_date"],
        missing_targets[-1]["target_date"],
        namespace="overview-metrics:" + ",".join(str(target["month"]) for target in missing_targets),
    )
    recent_failure = _recent_auto_fetch_failure(key)
    if recent_failure:
        metadata["warnings"] = [f"fund overview metric backfill skipped after recent failure: {recent_failure}"]
        metadata["warning_count"] = 1
        metadata["skipped_recent_failure"] = True
        return metadata

    owner = False
    with _AUTO_FETCH_LOCK:
        event = _AUTO_FETCH_IN_FLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _AUTO_FETCH_IN_FLIGHT[key] = event
            owner = True

    if not owner:
        waited = event.wait(timeout=max(1.0, TEFAS_TIMEOUT_SECONDS + 5.0))
        if not waited:
            metadata["warnings"] = ["fund overview metric backfill already in progress"]
            metadata["warning_count"] = 1
            return metadata
        recent_failure = _recent_auto_fetch_failure(key)
        if recent_failure:
            metadata["warnings"] = [f"fund overview metric backfill skipped after recent failure: {recent_failure}"]
            metadata["warning_count"] = 1
        return metadata

    try:
        metadata["attempted"] = True
        rows, fetched_months, warnings = _fetch_fund_overview_metric_rows(
            normalized,
            missing_targets,
            client=client,
        )
        storage_result: Dict[str, Any] = {
            "upserted_count": 0,
            "skipped_count": 0,
            "warnings": [],
        }
        if rows:
            storage_result = upsert_fund_price_points(
                processed_dir,
                rows,
                source=TEFASFON_FUNDS_SOURCE,
                fallback_code=normalized,
            )
        storage_warnings = [
            str(item.get("warning") or "invalid_price_row")
            for item in list(storage_result.get("warnings") or [])
        ]
        all_warnings = warnings + storage_warnings
        metadata.update(
            {
                "fetched_months": fetched_months,
                "fetched_point_count": len(rows),
                "upserted_count": storage_result.get("upserted_count", 0),
                "skipped_count": storage_result.get("skipped_count", 0),
                "warning_count": len(all_warnings),
                "warnings": all_warnings,
            }
        )
        if not rows:
            error = "overview metric snapshots returned no matching fund rows"
            _remember_auto_fetch_failure(key, error)
            metadata["warnings"] = all_warnings + [error]
            metadata["warning_count"] = len(metadata["warnings"])
        return metadata
    except FundUpstreamError as exc:
        error = str(exc)
        _remember_auto_fetch_failure(key, error)
        metadata["attempted"] = True
        metadata["warnings"] = [f"fund overview metric backfill failed: {error}"]
        metadata["warning_count"] = 1
        return metadata
    finally:
        with _AUTO_FETCH_LOCK:
            current = _AUTO_FETCH_IN_FLIGHT.get(key)
            if current is event:
                _AUTO_FETCH_IN_FLIGHT.pop(key, None)
            event.set()


def refresh_fund_performance(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    warnings: List[str] = []
    points: List[Dict[str, Any]] = []
    source_used = TEFASFON_FUNDS_SOURCE
    fallback_used = False
    fallback_reason: Optional[str] = None
    tefas_failure_reason: Optional[str] = None
    try:
        points = TefasFonClient().fetch_history(normalized, start_date, end_date)
        if not _valid_performance_points(points, normalized):
            tefas_failure_reason = "tefasfon_funds returned no valid points"
            warnings.append(tefas_failure_reason)
            points = []
    except TefasUpstreamError as exc:
        tefas_failure_reason = f"tefasfon_funds failed: {exc}"
        warnings.append(tefas_failure_reason)
        points = []

    if not points:
        fallback_used = True
        fallback_reason = tefas_failure_reason or "tefasfon_funds returned no valid points"
        source_used = FINTABLES_UDF_HISTORY_SOURCE
        try:
            points = fetch_fintables_udf_history(normalized, start_date, end_date)
        except FintablesUpstreamError as exc:
            warnings.append(f"fintables_udf_history failed: {exc}")
            raise FundUpstreamError("; ".join(warnings)) from exc

    storage_result = upsert_fund_price_points(
        processed_dir,
        points,
        source=source_used,
        fallback_code=normalized,
    )
    merged_points = _read_daily_fund_price_points(
        processed_dir,
        normalized,
        start_date=start_date,
        end_date=end_date,
    )
    storage_warnings = [
        str(item.get("warning") or "invalid_price_row")
        for item in list(storage_result.get("warnings") or [])
    ]
    payload = _fund_performance_payload_from_points(
        processed_dir,
        normalized,
        merged_points,
        start_date=start_date,
        end_date=end_date,
        fetched_at=str(storage_result.get("fetched_at") or _utc_now_iso()),
        cache_hit=False,
        stale=False,
        parse_status="ok" if merged_points else "empty",
        warnings=warnings + storage_warnings,
        fetched_point_count=len(points),
        storage_result=storage_result,
        backfill_used=True,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    _write_json(_history_path(processed_dir, normalized), payload)
    return payload


def get_fund_performance_payload(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    allow_upstream_fallback: bool = False,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    full_history_requested = start_date is None and end_date is None
    effective_end = end_date or date.today()
    effective_start = start_date or (
        _fund_full_history_start_date()
        if full_history_requested
        else effective_end - timedelta(days=max(1, FUNDS_AUTO_FETCH_LOOKBACK_DAYS))
    )
    query_start = None if full_history_requested else effective_start
    query_end = end_date or effective_end
    auto_warnings: List[str] = []
    auto_fetch_attempted = False
    full_history_cache_hit = (
        not full_history_requested
        or _history_cache_covers_requested_span(
            processed_dir,
            normalized,
            start_date=effective_start,
            end_date=effective_end,
        )
    )

    points = _read_daily_fund_price_points(
        processed_dir,
        normalized,
        start_date=query_start,
        end_date=query_end,
    )
    has_requested_coverage = full_history_cache_hit if full_history_requested else _has_requested_price_coverage(
        points,
        start_date=query_start,
        end_date=effective_end,
    )
    if not has_requested_coverage:
        auto_fetch_attempted = True
        auto_warnings = _auto_refresh_fund_performance(
            processed_dir,
            normalized,
            start_date=effective_start,
            end_date=effective_end,
        )
        points = _read_daily_fund_price_points(
            processed_dir,
            normalized,
            start_date=query_start,
            end_date=query_end,
        )

    overview_metric_backfill: Optional[Dict[str, Any]] = None
    overview_metric_backfill_attempted = False
    if full_history_requested and points:
        overview_metric_backfill = _auto_refresh_fund_overview_metrics(
            processed_dir,
            normalized,
            points=points,
            end_date=effective_end,
        )
        overview_metric_backfill_attempted = bool(overview_metric_backfill.get("attempted"))
        if overview_metric_backfill_attempted:
            points = _read_daily_fund_price_points(
                processed_dir,
                normalized,
                start_date=query_start,
                end_date=query_end,
            )

    if not points:
        warnings = ["fund price database has no valid TEFAS/Fintables points for this fund/range"] + auto_warnings
        return {
            "fund_code": normalized,
            "status": "unavailable",
            "points": [],
            "source": "sqlite",
            "source_url": str(_fund_prices_db_path(processed_dir)),
            "as_of": None,
            "fetched_at": None,
            "stale": True,
            "source_metadata": {
                "source": "sqlite",
                "source_url": str(_fund_prices_db_path(processed_dir)),
                "db_path": str(_fund_prices_db_path(processed_dir)),
                "fetched_at": None,
                "as_of": None,
                "cache_hit": False,
                "stale": True,
                "parse_status": "unavailable",
                "warnings": warnings,
                "warning": warnings[0] if warnings else None,
                "history_source_used": None,
                "history_source_policy": FUND_HISTORY_SOURCE_POLICY,
                "source_policy": FUND_HISTORY_SOURCE_POLICY,
                "primary_source": "tefasfon",
                "tefasfon_adapter_version": _tefasfon_adapter_version(),
                "fallback_used": any("fintables_udf_history" in warning for warning in warnings),
                "fallback_reason": "tefasfon_and_fintables_unavailable" if any("fintables_udf_history" in warning for warning in warnings) else None,
                "final_points_count": 0,
                "date_min": None,
                "date_max": None,
                "backfill_used": auto_fetch_attempted,
                "requested_start_date": effective_start.isoformat(),
                "requested_end_date": effective_end.isoformat(),
                "auto_fetch_attempted": auto_fetch_attempted,
                "overview_metric_backfill": overview_metric_backfill,
            },
        }
    dominant_source = _dominant_price_source(points)
    cached_fallback_is_primary = dominant_source == FINTABLES_UDF_HISTORY_SOURCE
    return _fund_performance_payload_from_points(
        processed_dir,
        normalized,
        points,
        start_date=effective_start,
        end_date=effective_end,
        cache_hit=not auto_fetch_attempted and not overview_metric_backfill_attempted,
        stale=False if (auto_fetch_attempted or overview_metric_backfill_attempted) and not auto_warnings else None,
        parse_status=(
            "ok_fund_history_auto"
            if auto_fetch_attempted and not auto_warnings
            else "ok_fund_history_overview_metrics_auto"
            if overview_metric_backfill_attempted and not auto_warnings
            else None
        ),
        warnings=auto_warnings,
        backfill_used=auto_fetch_attempted or overview_metric_backfill_attempted,
        fallback_used=cached_fallback_is_primary,
        fallback_reason=(
            "cached_fintables_points_primary"
            if cached_fallback_is_primary
            else None
        ),
        overview_metric_backfill=overview_metric_backfill,
    )


def refresh_fund_allocations(
    processed_dir: Path,
    fund_code: str,
    *,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    fetched_at = _utc_now_iso()
    effective_as_of = as_of or date.today()
    warnings: List[str] = []
    try:
        rows, warnings = TefasFonClient().fetch_latest_portfolio(
            normalized,
            as_of=effective_as_of,
            lookback_days=10,
        )
    except TefasUpstreamError as exc:
        rows = []
        warnings = [f"tefasfon_portfolio failed: {exc}"]

    allocations = [
        allocation
        for row in rows
        for allocation in _normalize_allocation_row(row, fallback_code=normalized)
    ]
    if allocations:
        report_dates = [item.get("report_date") for item in allocations if item.get("report_date")]
        payload = {
            "fund_code": normalized,
            "status": "ok",
            "allocations": allocations,
            "source": TEFASFON_PORTFOLIO_SOURCE,
            "stale": False,
            "source_metadata": {
                "source": TEFASFON_PORTFOLIO_SOURCE,
                "source_url": TEFASFON_SOURCE_URL,
                "fetched_at": fetched_at,
                "as_of": max(report_dates) if report_dates else effective_as_of.isoformat(),
                "cache_hit": False,
                "stale": False,
                "parse_status": "ok_tefasfon_portfolio",
                "source_policy": "tefasfon_primary",
                "fallback_used": False,
                "warnings": warnings,
            },
        }
        _write_json(_allocations_path(processed_dir, normalized), payload)
        return payload

    cached = _read_json(_allocations_path(processed_dir, normalized))
    if cached and _public_price_source(str(cached.get("source") or "")) != "legacy_cache":
        payload = dict(cached)
        meta = dict(payload.get("source_metadata") or {})
        meta["cache_hit"] = True
        meta["stale"] = True
        meta["warnings"] = list(meta.get("warnings") or []) + warnings + ["tefasfon_portfolio returned no usable allocation rows"]
        meta["warning"] = meta["warnings"][0] if meta["warnings"] else None
        payload["stale"] = True
        payload["source_metadata"] = meta
        return payload

    payload = {
        "fund_code": normalized,
        "status": "unavailable",
        "allocations": [],
        "source": TEFASFON_PORTFOLIO_SOURCE,
        "stale": True,
        "source_metadata": {
            "source": TEFASFON_PORTFOLIO_SOURCE,
            "source_url": TEFASFON_SOURCE_URL,
            "fetched_at": fetched_at,
            "as_of": effective_as_of.isoformat(),
            "cache_hit": False,
            "stale": True,
            "parse_status": "unavailable",
            "source_policy": "tefasfon_primary",
            "fallback_used": False,
            "warnings": warnings + ["tefasfon_portfolio returned no usable allocation rows"],
        },
    }
    return payload


def refresh_fund_allocations_history(
    processed_dir: Path,
    fund_code: str,
    *,
    lookback_days: int = 30,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    bounded_lookback = max(1, min(365, int(lookback_days)))
    fetched_at = _utc_now_iso()
    effective_as_of = as_of or date.today()
    start_date = effective_as_of - timedelta(days=bounded_lookback - 1)
    warnings: List[str] = []
    client = TefasFonClient()
    try:
        rows = client.fetch_portfolio(
            fund_code=normalized,
            start_date=start_date,
            end_date=effective_as_of,
        )
    except TefasUpstreamError as exc:
        rows = []
        warnings = [f"tefasfon_portfolio_history failed: {exc}"]
    if not rows:
        daily_dates: List[date] = []
        target_date = start_date
        while target_date <= effective_as_of:
            if target_date.weekday() < 5:
                daily_dates.append(target_date)
            target_date += timedelta(days=1)
        daily_rows: List[Dict[str, Any]] = []
        daily_warnings: List[str] = []

        def fetch_daily_portfolio(day: date) -> Tuple[date, List[Dict[str, Any]], Optional[str]]:
            try:
                return day, client.fetch_portfolio(fund_code=normalized, start_date=day, end_date=day), None
            except TefasUpstreamError as exc:
                return day, [], f"tefasfon_portfolio_history failed for {day.isoformat()}: {exc}"

        max_workers = max(1, min(8, FUNDS_DETAIL_MAX_WORKERS, len(daily_dates) or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tefas-portfolio-history") as executor:
            futures = [executor.submit(fetch_daily_portfolio, day) for day in daily_dates]
            for future in concurrent.futures.as_completed(futures):
                _day, day_rows, warning = future.result()
                daily_rows.extend(day_rows)
                if warning:
                    daily_warnings.append(warning)

        if daily_rows:
            warnings.append("tefasfon_portfolio_history range query was empty; used daily snapshots")
            rows = daily_rows
        warnings.extend(daily_warnings[:20])

    allocations_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        for allocation in _normalize_allocation_row(row, fallback_code=normalized):
            report_date = _fund_date(allocation.get("report_date"))
            if not report_date:
                continue
            allocations_by_date.setdefault(report_date, []).append(allocation)

    history = [
        {
            "date": report_date,
            "allocations": sorted(
                allocations,
                key=lambda item: abs(float(item.get("weight") or 0)),
                reverse=True,
            ),
        }
        for report_date, allocations in sorted(allocations_by_date.items())
    ]
    payload = {
        "fund_code": normalized,
        "status": "ok" if history else "empty",
        "lookback_days": bounded_lookback,
        "history": history,
        "source": TEFASFON_PORTFOLIO_SOURCE,
        "stale": False,
        "source_metadata": {
            "source": TEFASFON_PORTFOLIO_SOURCE,
            "source_url": TEFASFON_SOURCE_URL,
            "fetched_at": fetched_at,
            "as_of": history[-1]["date"] if history else effective_as_of.isoformat(),
            "cache_hit": False,
            "stale": False,
            "parse_status": "ok_tefasfon_portfolio_history" if history else "empty_tefasfon_portfolio_history",
            "source_policy": "tefasfon_primary",
            "fallback_used": False,
            "warnings": warnings,
            "requested_start_date": start_date.isoformat(),
            "requested_end_date": effective_as_of.isoformat(),
        },
    }
    if history or os.getenv("RAGFIN_FUNDS_CACHE_EMPTY_ALLOCATION_HISTORY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        _write_json(_allocations_history_path(processed_dir, normalized, bounded_lookback), payload)
    return payload


def get_fund_allocations_history_payload(
    processed_dir: Path,
    fund_code: str,
    *,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    bounded_lookback = max(1, min(365, int(lookback_days)))
    path = _allocations_history_path(processed_dir, normalized, bounded_lookback)
    payload = _read_json(path)
    stale = True
    if payload:
        stale = (_cache_age_seconds(payload.get("source_metadata", {}).get("fetched_at")) or (FUNDS_ALLOCATION_TTL_SECONDS + 1)) > FUNDS_ALLOCATION_TTL_SECONDS
        if not stale:
            meta = dict(payload.get("source_metadata") or {})
            meta["cache_hit"] = True
            meta["stale"] = False
            payload = dict(payload)
            payload["stale"] = False
            payload["source_metadata"] = meta
            return payload

    fresh = refresh_fund_allocations_history(processed_dir, normalized, lookback_days=bounded_lookback)
    if fresh.get("history"):
        return fresh
    if payload:
        meta = dict(payload.get("source_metadata") or {})
        meta["cache_hit"] = True
        meta["stale"] = True
        meta["warnings"] = list(meta.get("warnings") or []) + list(fresh.get("source_metadata", {}).get("warnings") or [])
        meta["warning"] = meta["warnings"][0] if meta["warnings"] else None
        payload = dict(payload)
        payload["stale"] = True
        payload["source_metadata"] = meta
        return payload
    return fresh


def get_fund_allocations_payload(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    path = _allocations_path(processed_dir, normalized)
    payload = _read_json(path)
    if not payload or _public_price_source(str(payload.get("source") or "")) == "legacy_cache":
        return {
            "fund_code": normalized,
            "status": "unavailable",
            "allocations": [],
            "source": TEFASFON_PORTFOLIO_SOURCE,
            "source_metadata": {
                "source": TEFASFON_PORTFOLIO_SOURCE,
                "source_url": TEFASFON_SOURCE_URL,
                "fetched_at": None,
                "as_of": None,
                "cache_hit": False,
                "stale": True,
                "parse_status": "unavailable",
                "source_policy": "tefasfon_primary",
                "fallback_used": False,
                "warnings": ["fund allocation cache is empty or not available from TEFAS"],
            },
        }
    stale = (_cache_age_seconds(payload.get("source_metadata", {}).get("fetched_at") or payload.get("fetched_at")) or (FUNDS_ALLOCATION_TTL_SECONDS + 1)) > FUNDS_ALLOCATION_TTL_SECONDS
    meta = dict(payload.get("source_metadata") or {})
    meta["cache_hit"] = True
    meta["stale"] = stale
    payload = dict(payload)
    payload["stale"] = stale
    payload["source_metadata"] = meta
    return payload


KAP_HOLDINGS_SOURCE = "kap_portfolio_allocation_report"
KAP_HOLDINGS_PARSE_VERSION = 6
_KAP_NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:\.\d{3})*,\d+|-?\d+,\d+")
_KAP_DATE_PATTERN = re.compile(r"\b\d{2}/\d{2}/\d{2,4}\b")
_KAP_ISIN_TAIL_PATTERN = re.compile(
    r"(?P<weight>-?\d{1,3}(?:[,.]\d{1,6}))\s*(?:TL|TRY|USD|EUR|JPY|GBP)?\s*[A-Z]{2}[A-Z0-9]{6,}\s*$",
    flags=re.IGNORECASE,
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
    "PORTFÖY",
    "SATIN",
    "SIRKETIN",
    "ŞIRKETIN",
    "ŞİRKETİN",
    "TARIH",
    "TARİH",
    "TOPLAM",
    "TUTAR",
    "VADE",
}
_KAP_INCLUDED_HOLDING_TYPES = {"local_equity", "fund"}
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
        "Referer": f"{KAP_BASE_URL}/tr/",
        "User-Agent": FINTABLES_USER_AGENT,
    }


def _kap_url(path: str) -> str:
    if str(path).startswith(("http://", "https://")):
        return str(path)
    return f"{KAP_BASE_URL}/{str(path).lstrip('/')}"


def _kap_get_json(path: str) -> Any:
    with httpx.Client(timeout=KAP_TIMEOUT_SECONDS, follow_redirects=True, headers=_kap_headers()) as client:
        response = client.get(_kap_url(path))
        response.raise_for_status()
        return response.json()


def _kap_post_json(path: str, payload: Dict[str, Any]) -> Any:
    with httpx.Client(timeout=KAP_TIMEOUT_SECONDS, follow_redirects=True, headers=_kap_headers()) as client:
        response = client.post(_kap_url(path), json=payload)
        response.raise_for_status()
        return response.json()


def _kap_get_text(path: str) -> str:
    with httpx.Client(timeout=KAP_TIMEOUT_SECONDS, follow_redirects=True, headers=_kap_headers("text/html,*/*")) as client:
        response = client.get(_kap_url(path))
        response.raise_for_status()
        return response.text


def _kap_get_bytes(path: str, accept: str = "application/pdf,*/*") -> bytes:
    with httpx.Client(timeout=KAP_TIMEOUT_SECONDS, follow_redirects=True, headers=_kap_headers(accept)) as client:
        response = client.get(_kap_url(path))
        response.raise_for_status()
        return bytes(response.content)


def _kap_search_fund_metadata(fund_code: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_fund_code(fund_code)
    if not normalized:
        return None
    payload = _kap_post_json("/tr/api/search/combined", {"keyword": normalized})
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
        candidate_code = normalize_fund_code(_normalize_match_text(row.get("cmpOrFundCode")))
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
        return KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID
    try:
        html = _kap_get_text(f"/tr/fon-bildirimleri/{quote(oid)}")
    except Exception:
        return KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID
    match = re.search(
        r'\\?"value\\?"\s*:\s*\\?"([^"\\]+)\\?"\s*,\s*\\?"label\\?"\s*:\s*\\?"Portföy Dağılım Raporu',
        html,
    )
    if match:
        return match.group(1)
    return KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID


def _kap_disclosure_basic(row: Dict[str, Any]) -> Dict[str, Any]:
    basic = row.get("disclosureBasic")
    return dict(basic) if isinstance(basic, dict) else dict(row)


def _kap_disclosure_index(row: Dict[str, Any]) -> Optional[int]:
    basic = _kap_disclosure_basic(row)
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
    subject = quote(str(subject_oid or KAP_PORTFOLIO_ALLOCATION_SUBJECT_OID).strip())
    if not oid:
        return []
    year = _utc_now().year
    ranges = [str(max(30, min(2555, KAP_HOLDINGS_LOOKBACK_DAYS))), str(year), str(year - 1)]
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    for range_value in ranges:
        try:
            payload = _kap_get_json(f"/tr/api/disclosure/filter/FILTERYFBF/{oid}/{subject}/{quote(range_value)}")
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            disclosure_index = _kap_disclosure_index(row)
            if disclosure_index:
                rows_by_index[disclosure_index] = dict(row)
        if len(rows_by_index) >= 2 and range_value.isdigit() and int(range_value) >= 365:
            break
    rows = list(rows_by_index.values())
    rows.sort(
        key=lambda row: (
            _parse_kap_datetime(_kap_disclosure_basic(row).get("publishDate")) or datetime.min,
            _kap_disclosure_index(row) or 0,
        ),
        reverse=True,
    )
    return rows


def _kap_fetch_report_detail(disclosure_index: int) -> Optional[Dict[str, Any]]:
    payload = _kap_get_json(f"/tr/api/notification/attachment-detail/{int(disclosure_index)}")
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
    return _kap_get_bytes(f"/tr/api/file/download/{quote(str(obj_id or '').strip())}")


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
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(part for part in parts if part)


def _month_end_iso(year: int, month: int) -> Optional[str]:
    if year < 1900 or month < 1 or month > 12:
        return None
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    return (next_month - timedelta(days=1)).isoformat()


def _kap_report_date(detail: Dict[str, Any]) -> Optional[str]:
    basic = _kap_report_basic(detail)
    for attachment in _kap_report_attachments(detail):
        filename = str(attachment.get("fileName") or "")
        match = re.search(r"_(20\d{2})[._-](\d{1,2})", filename)
        if match:
            return _month_end_iso(int(match.group(1)), int(match.group(2)))
    try:
        year = int(basic.get("year") or 0)
        month = int(basic.get("donem") or 0)
    except (TypeError, ValueError):
        return None
    return _month_end_iso(year, month)


def _kap_source_url(disclosure_index: Any) -> Optional[str]:
    try:
        idx = int(disclosure_index or 0)
    except (TypeError, ValueError):
        return None
    return f"{KAP_BASE_URL}/tr/Bildirim/{idx}" if idx > 0 else None


def _kap_number_matches(block: str) -> List[re.Match[str]]:
    date_spans = [match.span() for match in _KAP_DATE_PATTERN.finditer(block)]

    def inside_date(span: Tuple[int, int]) -> bool:
        return any(span[0] >= start and span[1] <= end for start, end in date_spans)

    return [match for match in _KAP_NUMBER_PATTERN.finditer(block) if not inside_date(match.span())]


def _kap_stock_symbol_set() -> set[str]:
    global _KAP_BIST_STOCK_SYMBOLS
    if _KAP_BIST_STOCK_SYMBOLS is not None:
        return _KAP_BIST_STOCK_SYMBOLS
    symbols: set[str] = set()
    try:
        from app.kap_service import BIST_ALL_SYMBOLS_FALLBACK

        symbols.update(normalize_fund_code(symbol) for symbol in BIST_ALL_SYMBOLS_FALLBACK)
    except Exception:
        pass
    symbols.update(_KAP_EXTRA_STOCK_SYMBOLS)
    _KAP_BIST_STOCK_SYMBOLS = {symbol for symbol in symbols if symbol}
    return _KAP_BIST_STOCK_SYMBOLS


def _kap_is_stock_symbol(code: str) -> bool:
    symbol = normalize_fund_code(code).replace(".", "")
    return bool(symbol and symbol in _kap_stock_symbol_set())


def _kap_stock_name_from_cache(processed_dir: Path, code: str) -> Optional[str]:
    symbol = normalize_fund_code(code).replace(".", "")
    if not symbol:
        return None
    payload = _read_json(processed_dir / "kap_cache" / f"{symbol}.json")
    if not payload:
        return None
    for key in ("company_title", "title", "companyName", "name"):
        value = str(payload.get(key) or "").strip()
        if value and normalize_fund_code(value) != symbol:
            return value
    return None


def _kap_fund_name_map(processed_dir: Path) -> Dict[str, str]:
    payload = _read_json(_snapshot_path(processed_dir))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    result: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = normalize_fund_code(row.get("fund_code")).replace(".", "")
        name = str(row.get("name") or "").strip()
        if code and name:
            result[code] = name
    return result


def _kap_looks_like_equity_symbol(code: str) -> bool:
    symbol = normalize_fund_code(code).replace(".", "")
    return bool(re.fullmatch(r"[A-Z0-9]{3,6}", symbol))


def _kap_looks_like_fund_symbol(code: str) -> bool:
    symbol = normalize_fund_code(code).replace(".", "")
    if not symbol or symbol in _KAP_POSITION_STOPWORDS:
        return False
    if symbol.startswith(("TRT", "TRF", "TRB", "TRD", "F_")):
        return False
    return bool(re.fullmatch(r"[A-Z0-9]{3,6}", symbol))


def _kap_is_included_holding_type(asset_type: Any) -> bool:
    return str(asset_type or "").strip().lower() in _KAP_INCLUDED_HOLDING_TYPES


def _kap_asset_type_from_context(context: str, code: str, name: str) -> str:
    context_norm = _normalize_match_text(context)
    name_norm = _normalize_match_text(name)
    haystack = _normalize_match_text(f"{context} {code} {name}")
    symbol = normalize_fund_code(code)
    has_equity_context = any(token in context_norm for token in ("HISSE SENEDI", "HISSE SENETLERI", "HISSE"))
    has_fund_context = any(token in context_norm for token in ("Y.FONU", "YATIRIM FONU", "BORSA Y", "BYF", "FON SEPETI"))
    has_excluded_context = any(
        token in context_norm
        for token in ("REPO", "TAHVIL", "BONO", "BORCLANMA", "HAZINE", "KIRA SERTIFIKASI", "DOVIZ", "NAKIT", "MEVDUAT", "TEMINAT")
    )
    if _kap_is_stock_symbol(symbol):
        return "local_equity"
    if has_equity_context and _kap_looks_like_equity_symbol(symbol):
        return "local_equity"
    if has_fund_context and _kap_looks_like_fund_symbol(symbol):
        return "fund"
    if not has_excluded_context and _kap_looks_like_fund_symbol(symbol) and any(token in name_norm for token in ("YATIRIM FONU", "PORTFOY", "BYF", "FON")):
        return "fund"
    if any(token in haystack for token in ("REPO", "TAHVIL", "BONO", "BORCLANMA", "HAZINE", "KIRA SERTIFIKASI")):
        return "debt"
    if symbol in {"USD", "EUR", "JPY", "GBP"} or any(token in haystack for token in ("DOVIZ", "NAKIT", "MEVDUAT", "TEMINAT")):
        return "cash_fx"
    if symbol.startswith(("TRT", "TRF", "TRB")):
        return "debt"
    return "other"


def _kap_category_marker(line: str) -> Optional[str]:
    norm = _normalize_match_text(line)
    if not norm or len(norm) > 80:
        return None
    if norm in {"HISSE", "HISSE SENEDI", "HISSE SENETLERI"}:
        return "Hisse Senedi"
    if norm in {"Y.FONU", "YATIRIM FONU", "BORSA Y.FONU", "BYF"}:
        return "Yatırım Fonu/BYF"
    if norm in {"T.REPO", "REPO", "BORCLANMA SENETLERI", "KIRA SERTIFIKALARI"}:
        return "Borçlanma"
    if norm in {"MEVDUAT", "NAKIT", "DOVIZ/NAKIT", "DOVIZ NAKIT"}:
        return "Döviz/Nakit"
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
            if _normalize_match_text(stripped) in {"TURK", "YABANCI"}:
                stripped = ""
            return stripped, marker
    return text, None


def _kap_line_starts_position(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    norm = _normalize_match_text(text)
    if not _kap_number_matches(text) and "PORTFOY" in norm and "FON" in norm:
        return False
    first = text.split()[0].strip(":-,;")
    if not first or first.upper() in _KAP_POSITION_STOPWORDS:
        return False
    first_norm = _normalize_match_text(first)
    if any(first_norm.startswith(f"{month}-20") for month in _KAP_TURKISH_MONTH_WORDS):
        return False
    if first[:1].isdigit() or len(first) < 2 or len(first) > 16:
        return False
    return bool(re.match(r"^[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9._-]*$", first, flags=re.IGNORECASE))


def _kap_row_complete(block: str) -> bool:
    compact = " ".join(str(block or "").split())
    if _KAP_ISIN_TAIL_PATTERN.search(compact):
        return True
    numbers = _kap_number_matches(compact)
    if len(numbers) < 4:
        return False
    return any(token in compact.upper() for token in (" TL", "TRY", "USD", "EUR", "JPY", "GBP"))


def _kap_buffer_is_header_noise(buffer: List[str]) -> bool:
    compact = " ".join(str(part or "").strip() for part in buffer if str(part or "").strip())
    if not compact or _kap_number_matches(compact):
        return False
    norm = _normalize_match_text(compact)
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
        first_left = normalize_fund_code(first_raw.split("-", 1)[0]).replace(".", "")
        if _kap_looks_like_fund_symbol(first_left):
            return False
    raw_tokens = compact.split()
    if len(raw_tokens) > 2 and raw_tokens[1] == "-":
        first_left = normalize_fund_code(raw_tokens[0].strip(":-,;()")).replace(".", "")
        if _kap_looks_like_fund_symbol(first_left):
            return False
    tokens = [normalize_fund_code(token.strip(":-,;()")).replace(".", "") for token in compact.split()]
    tokens = [token for token in tokens if token]
    if any(_kap_is_stock_symbol(token) for token in tokens[:4]):
        return False
    return True


def _parse_kap_holding_block(
    block: str,
    *,
    fund_code: str,
    category_context: str,
    report_date: Optional[str],
    source_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    compact = " ".join(str(block or "").replace("\xa0", " ").split())
    norm = _normalize_match_text(compact)
    if not compact or "GRUP TOPLAMI" in norm or "TOPLAM" == norm:
        return None

    code_match = re.match(r"(?P<code>[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9._-]{1,15})\b", compact, flags=re.IGNORECASE)
    if not code_match:
        return None
    code = code_match.group("code").strip().upper()
    if "-" in code:
        left, right = code.split("-", 1)
        if _kap_looks_like_fund_symbol(left):
            code = left
            compact = f"{left} {right} {compact[code_match.end():]}".strip()
            code_match = re.match(r"(?P<code>[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9._-]{1,15})\b", compact, flags=re.IGNORECASE)
            if not code_match:
                return None
    if code in _KAP_POSITION_STOPWORDS:
        return None

    number_matches = _kap_number_matches(compact)
    if len(number_matches) < 3:
        return None

    first_number_start = number_matches[0].start()
    raw_name = compact[code_match.end():first_number_start].strip(" -")
    raw_name = _KAP_DATE_PATTERN.sub(" ", raw_name)
    asset_name = " ".join(raw_name.split()) or code
    tail = _KAP_ISIN_TAIL_PATTERN.search(compact)
    weight = _coerce_float(tail.group("weight")) if tail else None
    parsed_numbers = [_coerce_float(match.group(0)) for match in number_matches]
    numbers = [value for value in parsed_numbers if value is not None]
    if weight is None:
        small_values = [value for value in numbers[-4:] if abs(value) <= 100]
        weight = small_values[-1] if small_values else None
    if weight is None:
        return None

    amount = numbers[0] if numbers else None
    values_before_weight = numbers[:-1] if numbers and abs(numbers[-1] - float(weight)) < 0.0001 else numbers
    market_value = next((value for value in reversed(values_before_weight) if abs(value) > 100), None)
    asset_type = _kap_asset_type_from_context(category_context, code, asset_name)
    return {
        "fund_code": fund_code,
        "asset_code": code,
        "asset_name": asset_name,
        "asset_type": asset_type,
        "weight": round(float(weight), 6),
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
    }


def _parse_kap_holdings_pdf_text(
    text: str,
    *,
    fund_code: str,
    report_date: Optional[str],
    source_url: Optional[str],
) -> List[Dict[str, Any]]:
    raw_text = str(text or "")
    start_match = re.search(r"(?:III\s*[-–]\s*)?FON\s+PORTF[ÖO]Y\s+DE[ĞG]ER[İI]", raw_text, flags=re.IGNORECASE)
    if start_match:
        raw_text = raw_text[start_match.start():]
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    positions: List[Dict[str, Any]] = []
    current_category = ""
    buffer: List[str] = []
    buffer_category = ""

    def flush() -> None:
        nonlocal buffer, buffer_category
        if not buffer:
            return
        parsed = _parse_kap_holding_block(
            " ".join(buffer),
            fund_code=fund_code,
            category_context=buffer_category,
            report_date=report_date,
            source_url=source_url,
        )
        if parsed and _kap_is_included_holding_type(parsed.get("asset_type")):
            positions.append(parsed)
        buffer = []
        buffer_category = ""

    for line in lines:
        original_line = line
        line, inline_marker = _kap_strip_inline_category(line)
        if inline_marker and buffer and not _kap_row_complete(" ".join(buffer)):
            if _kap_buffer_is_header_noise(buffer):
                buffer = []
                buffer_category = ""
            else:
                line = original_line
                inline_marker = None
        if inline_marker:
            if buffer:
                flush()
            current_category = inline_marker
            if not line:
                continue
        norm = _normalize_match_text(line)
        marker = _kap_category_marker(line)
        if marker and len(_kap_number_matches(line)) < 2 and buffer and not _kap_row_complete(" ".join(buffer)):
            if _kap_buffer_is_header_noise(buffer):
                buffer = []
                buffer_category = ""
            else:
                marker = None
        if marker and len(_kap_number_matches(line)) < 2 and (not buffer or _kap_row_complete(" ".join(buffer))):
            if buffer:
                flush()
            current_category = marker
            continue
        if not current_category:
            continue
        if "GRUP TOPLAMI" in norm:
            flush()
            continue
        if buffer and _kap_line_starts_position(line) and not _kap_row_complete(" ".join(buffer)):
            if _kap_buffer_is_header_noise(buffer):
                buffer = []
                buffer_category = ""
        if not buffer and not _kap_line_starts_position(line):
            continue
        if buffer and _kap_line_starts_position(line) and _kap_row_complete(" ".join(buffer)):
            flush()
        if not buffer:
            buffer_category = current_category
        buffer.append(line)
        if _kap_row_complete(" ".join(buffer)):
            flush()
    flush()

    best_by_key: Dict[str, Dict[str, Any]] = {}
    for position in positions:
        key = normalize_fund_code(position.get("asset_code") or position.get("asset_name"))
        if not key:
            continue
        existing = best_by_key.get(key)
        if not existing or abs(float(position.get("weight") or 0)) > abs(float(existing.get("weight") or 0)):
            best_by_key[key] = position
    return sorted(best_by_key.values(), key=lambda row: abs(float(row.get("weight") or 0)), reverse=True)


def _position_key(position: Dict[str, Any]) -> str:
    return normalize_fund_code(position.get("asset_code") or position.get("asset_name"))


def _normalize_holding_positions_for_response(
    processed_dir: Path,
    positions: List[Dict[str, Any]],
    *,
    fund_code: str,
) -> List[Dict[str, Any]]:
    fund_names = _kap_fund_name_map(processed_dir)
    current_fund_code = normalize_fund_code(fund_code).replace(".", "")
    normalized_rows: List[Dict[str, Any]] = []
    for position in positions:
        code = normalize_fund_code(position.get("asset_code")).replace(".", "")
        if not code or code in _KAP_POSITION_STOPWORDS:
            continue
        if "-" in code:
            left = code.split("-", 1)[0]
            if fund_names.get(left) or _kap_looks_like_fund_symbol(left):
                code = left
        if code == current_fund_code:
            continue
        row = dict(position)
        row["asset_code"] = code
        if _kap_is_stock_symbol(code):
            row["asset_type"] = "local_equity"
            row["asset_name"] = _kap_stock_name_from_cache(processed_dir, code) or str(row.get("asset_name") or code).strip() or code
        else:
            row_type = str(row.get("asset_type") or "").strip().lower()
            name_norm = _normalize_match_text(row.get("asset_name") or "")
            looks_like_named_fund = any(token in name_norm for token in ("YATIRIM FONU", "BORSA YATIRIM FONU", " FONU", "FON ", "BYF"))
            if not _kap_looks_like_fund_symbol(code) or not (row_type == "fund" or looks_like_named_fund or fund_names.get(code)):
                continue
            row["asset_type"] = "fund"
            row["asset_name"] = fund_names.get(code) or str(row.get("asset_name") or code).strip() or code
        normalized_rows.append(row)
    return normalized_rows


def _merge_holding_positions(
    latest_positions: List[Dict[str, Any]],
    previous_positions: List[Dict[str, Any]],
    *,
    latest_report_date: Optional[str],
    previous_report_date: Optional[str],
) -> List[Dict[str, Any]]:
    previous_by_key = {_position_key(position): position for position in previous_positions if _position_key(position)}
    latest_keys: set[str] = set()
    merged: List[Dict[str, Any]] = []
    for position in latest_positions:
        key = _position_key(position)
        latest_keys.add(key)
        previous = previous_by_key.get(key)
        current_weight = _coerce_float(position.get("weight"))
        previous_weight = _coerce_float(previous.get("weight")) if previous else None
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
        merged_position["previous_report_date"] = previous_report_date
        merged_position["weight_change"] = delta
        merged_position["change_status"] = status
        merged.append(merged_position)

    for key, previous in previous_by_key.items():
        if key in latest_keys:
            continue
        previous_weight = _coerce_float(previous.get("weight"))
        removed = dict(previous)
        removed["weight"] = 0.0
        removed["previous_weight"] = previous_weight
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
    detail: Dict[str, Any],
    *,
    fund_code: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    basic = _kap_report_basic(detail)
    disclosure_index = basic.get("disclosureIndex")
    report_date = _kap_report_date(detail)
    source_url = _kap_source_url(disclosure_index)
    attachments = _kap_report_attachments(detail)
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
    data = _kap_download_attachment(str(pdf_attachment.get("objId") or ""))
    text = _extract_kap_pdf_text(data)
    positions = _parse_kap_holdings_pdf_text(
        text,
        fund_code=fund_code,
        report_date=report_date,
        source_url=source_url,
    )
    if not positions:
        metadata["warning"] = "KAP PDF text parsed but no position rows were detected"
    return positions, metadata


def _unavailable_holdings_payload(
    fund_code: str,
    *,
    message: str,
    warnings: Optional[List[str]] = None,
    cache_hit: bool = False,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    return {
        "fund_code": normalized,
        "status": "unavailable",
        "positions": [],
        "source": KAP_HOLDINGS_SOURCE,
        "message": message,
        "source_metadata": {
            "source": KAP_HOLDINGS_SOURCE,
            "source_url": KAP_BASE_URL,
            "fetched_at": _utc_now_iso(),
            "as_of": None,
            "cache_hit": cache_hit,
            "stale": True,
            "parse_status": "unavailable",
            "parser_version": KAP_HOLDINGS_PARSE_VERSION,
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


def _holdings_cache_is_stale(cached_meta: Dict[str, Any]) -> bool:
    cached_parser_version = _coerce_int(cached_meta.get("parser_version"))
    if cached_parser_version != KAP_HOLDINGS_PARSE_VERSION:
        return True

    latest_report = cached_meta.get("latest_report")
    latest_report_date = None
    if isinstance(latest_report, dict):
        latest_report_date = _parse_iso_date(latest_report.get("report_date"))
    if latest_report_date is None:
        latest_report_date = _parse_iso_date(cached_meta.get("as_of"))

    # KAP portfolio distribution reports are monthly. If the cached report is
    # from the current or immediately previous report month, there is no older
    # PDF content to refetch repeatedly during the same month.
    if latest_report_date is not None:
        today = _utc_now().date()
        month_age = (today.year - latest_report_date.year) * 12 + today.month - latest_report_date.month
        if month_age <= 1:
            return False

    age = _cache_age_seconds(cached_meta.get("fetched_at"))
    return age is None or age > FUNDS_HOLDINGS_TTL_SECONDS


def refresh_fund_holdings(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    metadata = _kap_search_fund_metadata(normalized)
    if not metadata:
        return _unavailable_holdings_payload(
            normalized,
            message="KAP fon kaydı bulunamadı.",
            warnings=[f"KAP fund search returned no exact match for {normalized}"],
        )
    fund_oid = str(metadata.get("fund_oid") or "").strip()
    subject_oid = _kap_portfolio_subject_oid(fund_oid)
    disclosures = _kap_list_portfolio_disclosures(fund_oid, subject_oid)
    if not disclosures:
        return _unavailable_holdings_payload(
            normalized,
            message="KAP portföy dağılım bildirimi bulunamadı.",
            warnings=[f"KAP portfolio allocation disclosures not found for {normalized}"],
        )

    warnings: List[str] = []
    report_payloads: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    for row in disclosures[:2]:
        disclosure_index = _kap_disclosure_index(row)
        if not disclosure_index:
            continue
        detail = _kap_fetch_report_detail(disclosure_index)
        if not detail:
            warnings.append(f"KAP disclosure detail unavailable: {disclosure_index}")
            continue
        positions, report_meta = _kap_report_positions(detail, fund_code=normalized)
        if report_meta.get("warning"):
            warnings.append(str(report_meta["warning"]))
        report_payloads.append((positions, report_meta))

    latest_positions = report_payloads[0][0] if report_payloads else []
    latest_meta = report_payloads[0][1] if report_payloads else {}
    previous_positions = report_payloads[1][0] if len(report_payloads) > 1 else []
    previous_meta = report_payloads[1][1] if len(report_payloads) > 1 else {}
    if not latest_positions:
        payload = _unavailable_holdings_payload(
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
        _write_json(_holdings_path(processed_dir, normalized), payload)
        return payload

    if not previous_positions:
        warnings.append("Önceki ay portföy raporu parse edilemedi veya bulunamadı.")
    positions = _merge_holding_positions(
        latest_positions,
        previous_positions,
        latest_report_date=latest_meta.get("report_date"),
        previous_report_date=previous_meta.get("report_date"),
    )
    positions = _normalize_holding_positions_for_response(processed_dir, positions, fund_code=normalized)
    parse_status = "ok" if latest_positions and previous_positions else "partial"
    payload = {
        "fund_code": normalized,
        "status": parse_status,
        "positions": positions,
        "source": KAP_HOLDINGS_SOURCE,
        "message": None,
        "source_metadata": {
            "source": KAP_HOLDINGS_SOURCE,
            "source_url": latest_meta.get("source_url") or KAP_BASE_URL,
            "fetched_at": _utc_now_iso(),
            "as_of": latest_meta.get("report_date"),
            "cache_hit": False,
            "stale": False,
            "parse_status": parse_status,
            "parser_version": KAP_HOLDINGS_PARSE_VERSION,
            "fund_oid": fund_oid,
            "subject_oid": subject_oid,
            "fund_name": metadata.get("fund_name"),
            "latest_report": latest_meta or None,
            "previous_report": previous_meta or None,
            "warnings": warnings,
        },
    }
    _write_json(_holdings_path(processed_dir, normalized), payload)
    return payload


def get_fund_holdings_payload(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    path = _holdings_path(processed_dir, normalized)
    cached = _read_json(path)
    if cached:
        cached_meta = cached.get("source_metadata", {}) if isinstance(cached.get("source_metadata"), dict) else {}
        stale = _holdings_cache_is_stale(cached_meta)
        if not stale:
            payload = dict(cached)
            meta = dict(cached_meta)
            meta["cache_hit"] = True
            meta["stale"] = False
            meta["cache_policy"] = "monthly_report"
            payload["source_metadata"] = meta
            return payload
    try:
        return refresh_fund_holdings(processed_dir, normalized)
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
        return _unavailable_holdings_payload(
            normalized,
            message="KAP portföy içeriği alınamadı.",
            warnings=[f"KAP holdings refresh failed: {exc}"],
        )
