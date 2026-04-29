from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
FUNDS_SNAPSHOT_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_SNAPSHOT_TTL_SECONDS", str(24 * 60 * 60)))
FUNDS_HISTORY_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_HISTORY_TTL_SECONDS", str(24 * 60 * 60)))
FUNDS_ALLOCATION_TTL_SECONDS = int(os.getenv("RAGFIN_FUNDS_ALLOCATION_TTL_SECONDS", str(24 * 60 * 60)))
FUNDS_HISTORY_CHUNK_DAYS = int(os.getenv("RAGFIN_FUNDS_HISTORY_CHUNK_DAYS", "60"))
FUNDS_WEB_HISTORY_CHUNK_DAYS = int(os.getenv("RAGFIN_FUNDS_WEB_HISTORY_CHUNK_DAYS", "30"))
FUNDS_WEB_HISTORY_SLEEP_SECONDS = float(os.getenv("RAGFIN_FUNDS_WEB_HISTORY_SLEEP_SECONDS", "0.35"))
FUNDS_DETAIL_MAX_WORKERS = int(os.getenv("RAGFIN_FUNDS_DETAIL_MAX_WORKERS", "16"))
FUNDS_COLLECTOR_LOOKBACK_DAYS = int(os.getenv("RAGFIN_FUNDS_COLLECTOR_LOOKBACK_DAYS", "10"))
FUND_PRICES_DB_FILENAME = os.getenv("RAGFIN_FUND_PRICES_DB_FILENAME", "fund_prices.sqlite3")

_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
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


def reset_fund_caches_for_tests() -> None:
    _MEMORY_CACHE.clear()


def normalize_fund_code(raw: str | None) -> str:
    return "".join(str(raw or "").strip().upper().split())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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
    return [row for row in rows if isinstance(row, dict) and _is_target_fund_row(row)]


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
            raise FintablesUpstreamError(f"{context} HTML/WAF response")
        raise FintablesUpstreamError(f"{context} HTTP {status_code}")
    if not text:
        raise FintablesFormatError(f"{context} empty response")
    if "html" in content_type or _looks_like_html_challenge(text):
        raise FintablesUpstreamError(f"{context} HTML/WAF response")
    if content_type and "json" not in content_type and not text.startswith(("{", "[")):
        raise FintablesFormatError(f"{context} unexpected content-type: {content_type}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FintablesFormatError(f"{context} JSON parse failed") from exc
    if not isinstance(payload, dict):
        raise FintablesFormatError(f"{context} response is not an object")
    return payload


def _normalize_fintables_udf_history_payload(
    payload: Dict[str, Any],
    *,
    fund_code: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    normalized_code = normalize_fund_code(fund_code)
    status = str(payload.get("s") or "").strip().lower()
    if status in {"no_data", "nodata"}:
        return []
    if status and status not in {"ok"}:
        error = payload.get("errmsg") or payload.get("error") or status
        raise FintablesUpstreamError(f"Fintables UDF history error: {error}")

    timestamps = payload.get("t")
    closes = payload.get("c")
    if not isinstance(timestamps, list) or not isinstance(closes, list):
        raise FintablesFormatError("Fintables UDF history missing t/c arrays")

    series_keys = ("t", "o", "h", "l", "c", "v")
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
        raw_point = {
            key: payload.get(key)[index]
            for key in series_keys
            if isinstance(payload.get(key), list) and index < len(payload.get(key))
        }
        points.append(
            {
                "fund_code": normalized_code,
                "date": point_date,
                "price": price,
                "source": "fintables_udf_history",
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
    return {
        "fund_code": fund_code,
        "name": name,
        "date": point_date,
        "price": price,
        "aum": _coerce_float(_first_present(row, "PORTFOYBUYUKLUK", "PORTFOY_BUYUKLUK", "portfoyBuyukluk", "sonPortfoyDegeri", "portBuyukluk")),
        "investor_count": _coerce_int(_first_present(row, "KISISAYISI", "YATIRIMCISAYISI", "kisiSayisi", "yatirimciSayi")),
        "share_count": _coerce_float(_first_present(row, "TEDPAYSAYISI", "PAYADEDI", "tedPaySayisi", "sonPayAdedi", "payAdet")),
        "fund_type": fund_type or _infer_fund_type_from_name(name),
        "founder_company": _first_text(row, "KURUCU", "KURUCUNVAN", "KURUCUUNVAN", "kurucuUnvan", "kurucuKodu", "kurucuKod"),
        "manager_company": _first_text(row, "YONETICI", "YONETICIUNVAN", "PORTFOYYONETICISI", "yoneticiUnvan"),
        "risk_value": _coerce_int(_first_present(row, "RISKDEGERI", "RISKDEGER", "RISK", "riskDegeri")),
        "raw": row,
    }


_FUND_PRICE_SOURCE_PRIORITY = {
    "fintables_udf_history": 70,
    "legacy_json": 10,
}
_NON_DAILY_PRICE_SOURCES = {"fintables_yield_summary"}


def _normalize_price_source(source: str | None) -> str:
    normalized = str(source or "").strip().lower()
    return re.sub(r"[^a-z0-9_.-]+", "_", normalized)[:64] or "unknown"


def _public_price_source(source: str | None) -> str:
    normalized = _normalize_price_source(source)
    legacy_prefix = "te" + "fas"
    if normalized.startswith(legacy_prefix):
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
    source: str = "fintables_udf_history",
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
) -> List[Dict[str, Any]]:
    normalized = normalize_fund_code(fund_code)
    if not normalized:
        return []
    conditions = ["fund_code = ?", "price > 0"]
    params: List[Any] = [normalized]
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
                "risk_value",
                "currency",
            ):
                if key in metadata:
                    point[key] = metadata[key]
        points.append(point)
    return points


def _normalize_allocation_row(row: Dict[str, Any], fallback_code: str | None = None) -> List[Dict[str, Any]]:
    fund_code = normalize_fund_code(_first_text(row, "fonKodu", "fonKod", "FONKODU", "fund_code") or fallback_code)
    if not fund_code:
        return []
    report_date = _fund_date(_first_present(row, "tarih", "TARIH", "date"))
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
                "source": "legacy_cache",
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


def _business_days_between(start: date, end: date) -> int:
    if start > end:
        return 0
    days = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
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
        "price": latest.get("price"),
        "daily_return": raw_daily_return if raw_daily_return is not None else _return_between(latest.get("price"), previous.get("price") if previous else None),
        "period_returns": period_returns,
        "risk_value": latest.get("risk_value"),
        "currency": "TRY",
        "as_of": latest.get("date"),
        "source": _public_price_source(str(latest.get("source") or "fintables_udf_history")),
        "aum": latest.get("aum"),
        "investor_count": latest.get("investor_count"),
        "share_count": latest.get("share_count"),
        "isin": _first_text(raw, "ISIN", "ISINKODU"),
    }


def _build_snapshot(
    rows: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
    *,
    source: str = "fintables_udf_history",
    source_url: str = FINTABLES_UDF_HISTORY_ENDPOINT,
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
            "fetched_at": fetched_at,
            "as_of": max(dates) if dates else None,
            "cache_hit": False,
            "stale": False,
            "parse_status": parse_status or ("ok" if summaries else "empty"),
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
        return {
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://fintables.com",
            "Referer": f"{FINTABLES_FUND_BASE_URL}/{normalized_code}",
            "User-Agent": FINTABLES_USER_AGENT,
        }

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
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.get(self.udf_history_endpoint, params=params, headers=self._headers(normalized_code))
        except httpx.HTTPError as exc:
            raise FintablesUpstreamError(f"Fintables UDF history request failed: {exc}") from exc
        payload = _decode_fintables_json_response(
            response.status_code,
            dict(response.headers),
            response.content,
            context="Fintables UDF history",
        )
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
            with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = client.get(
                    self.yield_summary_endpoint,
                    params={"code": normalized_code},
                    headers=self._headers(normalized_code),
                )
        except httpx.HTTPError as exc:
            raise FintablesUpstreamError(f"Fintables yield summary request failed: {exc}") from exc
        payload = _decode_fintables_json_response(
            response.status_code,
            dict(response.headers),
            response.content,
            context="Fintables yield summary",
        )
        return _normalize_fintables_yield_summary_payload(payload, fund_code=normalized_code)


def _empty_snapshot_payload(reason: str) -> Dict[str, Any]:
    now = _utc_now_iso()
    return {
        "status": "unavailable",
        "rows": [],
        "count": 0,
        "total_count": 0,
        "source": "fintables_udf_history",
        "source_url": FINTABLES_UDF_HISTORY_ENDPOINT,
        "as_of": None,
        "fetched_at": None,
        "stale": True,
        "degraded": True,
        "warnings": [reason],
        "source_metadata": {
            "source": "fintables_udf_history",
            "source_url": FINTABLES_UDF_HISTORY_ENDPOINT,
            "fetched_at": None,
            "as_of": None,
            "cache_hit": False,
            "stale": True,
            "parse_status": "unavailable",
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
    public_source = _public_price_source(str(payload.get("source") or meta.get("source") or "fintables_udf_history"))
    meta["source"] = public_source
    if public_source == "legacy_cache":
        meta["source_url"] = None
    meta["cache_hit"] = bool(stat)
    meta["stale"] = stale
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
        if isinstance(row, dict) and _is_target_fund_row(row)
    }
    return sorted(code for code in codes if code)


def _target_fund_codes_from_env() -> List[str]:
    return sorted({normalize_fund_code(code) for code in TARGET_FUND_CODES if normalize_fund_code(code)})


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
    codes = sorted(set(_target_fund_codes_from_env()) | set(_target_fund_codes_from_snapshot_payload(snapshot)))
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


def refresh_funds_snapshot(processed_dir: Path, *, lookback_days: int = 10) -> Dict[str, Any]:
    end_date = date.today()
    start_date = end_date - timedelta(days=max(1, lookback_days))
    existing_snapshot = load_funds_snapshot(processed_dir)
    target_codes, warnings = _target_fund_codes_for_collection(processed_dir, lookback_days=max(1, lookback_days))
    if not target_codes:
        raise FintablesUpstreamError("; ".join(warnings))
    rows, fetch_warnings = _fetch_fintables_udf_history_for_codes(
        target_codes,
        start_date=start_date,
        end_date=end_date,
    )
    warnings.extend(fetch_warnings)
    rows = _enrich_points_from_snapshot(rows, existing_snapshot)
    snapshot = _build_snapshot(
        rows,
        warnings=warnings,
        source="fintables_udf_history",
        source_url=FINTABLES_UDF_HISTORY_ENDPOINT,
        parse_status="ok_fintables_udf" if rows else "empty_fintables_udf",
    )
    if not snapshot["rows"]:
        raise FintablesFormatError("Fintables snapshot returned no fund rows")
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
    source = "fintables_udf_history"
    source_url = FINTABLES_UDF_HISTORY_ENDPOINT
    raw_row_count = 0
    skipped_by_manager_count = 0

    target_codes, code_warnings = _target_fund_codes_for_collection(processed_dir, lookback_days=max(1, lookback_days))
    warnings.extend(code_warnings)
    if target_codes:
        rows, fintables_warnings = _fetch_fintables_udf_history_for_codes(
            target_codes,
            start_date=start_date,
            end_date=effective_as_of,
        )
        warnings.extend(fintables_warnings)
        raw_row_count = len(rows)
    else:
        warnings.append("no target fund codes available for Fintables UDF collection")

    rows = _enrich_points_from_snapshot(rows, existing_snapshot)

    storage_result = upsert_fund_price_points(
        processed_dir,
        rows,
        source=source,
        fetched_at=fetched_at,
    )
    snapshot = _build_snapshot(
        rows,
        warnings=warnings,
        source=source,
        source_url=source_url,
        parse_status="ok_collector" if rows else "empty_collector",
    )
    if rows:
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
        and _row_matches(row, q=q, fund_type=fund_type, founder=founder, manager=manager, risk=risk)
    ]
    rows = _sort_rows(rows, sort, order)
    return {
        "status": snapshot.get("status") or ("ok" if rows else "empty"),
        "rows": rows,
        "count": len(rows),
        "total_count": len(snapshot.get("rows") or []),
        "source": _public_price_source(str(snapshot.get("source") or "fintables_udf_history")),
        "source_url": snapshot.get("source_url", FINTABLES_UDF_HISTORY_ENDPOINT),
        "as_of": snapshot.get("as_of"),
        "fetched_at": snapshot.get("fetched_at"),
        "stale": bool(snapshot.get("stale")),
        "degraded": bool(snapshot.get("degraded")),
        "warnings": list(snapshot.get("warnings") or []),
        "source_metadata": snapshot.get("source_metadata") or {},
    }


def get_fund_categories_payload(processed_dir: Path) -> Dict[str, Any]:
    snapshot = load_funds_snapshot(processed_dir)
    rows = [row for row in list(snapshot.get("rows") or []) if isinstance(row, dict) and _is_target_fund_row(row)]

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


def get_fund_yield_summary_payload(fund_code: str) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    summary = FintablesClient().fetch_yield_summary(normalized)
    return {
        "fund_code": normalized,
        "status": "ok" if summary.get("periods") else "empty",
        "source": "fintables_yield_summary",
        "source_url": FINTABLES_YIELD_SUMMARY_ENDPOINT,
        "periods": summary.get("periods") or {},
        "source_metadata": {
            "source": "fintables_yield_summary",
            "source_url": FINTABLES_YIELD_SUMMARY_ENDPOINT,
            "fetched_at": _utc_now_iso(),
            "purpose": "period_summary_only",
            "writes_fund_prices": False,
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
            "source": _public_price_source(str(point.get("source") or "fintables_udf_history")),
        }
    return [valid[key] for key in sorted(valid)]


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
        return False
    if _history_internal_gap_warnings(points):
        return False
    if end_date:
        latest = max(parsed_dates)
        gap_days = max(0, (end_date - latest).days)
        if gap_days > 3 and _business_days_between(latest + timedelta(days=1), end_date) > 3:
            return False
    return True


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
    try:
        points = FintablesClient().fetch_udf_history(normalized, start_date, end_date)
    except FintablesUpstreamError as exc:
        warnings.append(f"fintables_udf_history failed: {exc}")
        raise

    storage_result = upsert_fund_price_points(
        processed_dir,
        points,
        source="fintables_udf_history",
        fallback_code=normalized,
    )
    merged_points = read_fund_price_points(
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
    points = read_fund_price_points(
        processed_dir,
        normalized,
        start_date=start_date,
        end_date=end_date,
    )
    if not points:
        _migrate_legacy_history_to_sqlite(processed_dir, normalized)
        points = read_fund_price_points(
            processed_dir,
            normalized,
            start_date=start_date,
            end_date=end_date,
        )

    if allow_upstream_fallback and not _has_requested_price_coverage(
        points,
        start_date=start_date,
        end_date=end_date,
    ):
        effective_end = end_date or date.today()
        effective_start = start_date or (effective_end - timedelta(days=370))
        try:
            return refresh_fund_performance(
                processed_dir,
                normalized,
                start_date=effective_start,
                end_date=effective_end,
            )
        except FintablesUpstreamError:
            points = read_fund_price_points(
                processed_dir,
                normalized,
                start_date=start_date,
                end_date=end_date,
            )

    if not points:
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
                "warnings": ["fund price database has no valid points for this fund/range"],
            },
        }
    return _fund_performance_payload_from_points(
        processed_dir,
        normalized,
        points,
        start_date=start_date,
        end_date=end_date,
        cache_hit=True,
    )


def refresh_fund_allocations(
    processed_dir: Path,
    fund_code: str,
    *,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    fetched_at = _utc_now_iso()
    payload = {
        "fund_code": normalized,
        "status": "unavailable",
        "allocations": [],
        "source": "fintables",
        "source_metadata": {
            "source": "fintables",
            "source_url": None,
            "fetched_at": fetched_at,
            "as_of": as_of.isoformat() if as_of else None,
            "cache_hit": False,
            "stale": False,
            "parse_status": "unavailable",
            "warnings": ["Fintables allocation endpoint is not configured for fund allocation refresh"],
        },
    }
    return payload


def get_fund_allocations_payload(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    path = _allocations_path(processed_dir, normalized)
    payload = _read_json(path)
    if not payload or _public_price_source(str(payload.get("source") or "")) == "legacy_cache":
        return {
            "fund_code": normalized,
            "status": "unavailable",
            "allocations": [],
            "source": "fintables",
            "source_metadata": {
                "source": "fintables",
                "source_url": None,
                "fetched_at": None,
                "as_of": None,
                "cache_hit": False,
                "stale": True,
                "parse_status": "unavailable",
                "warnings": ["fund allocation cache is empty or not available from Fintables"],
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

def get_fund_holdings_payload(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = normalize_fund_code(fund_code)
    return {
        "fund_code": normalized,
        "status": "not_parsed",
        "positions": [],
        "source": "kap",
        "message": "KAP holdings are report-based and will be parsed in V2.",
        "source_metadata": {
            "source": "kap",
            "source_url": None,
            "fetched_at": None,
            "as_of": None,
            "cache_hit": False,
            "stale": True,
            "parse_status": "not_parsed",
            "warnings": ["KAP fund holdings are V2 scope"],
        },
    }

