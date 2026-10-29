from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from src.overview_scoring import (
    COMPANY_KINDS,
    SCORE_SOURCE_AI_ADJUSTED,
    SCORE_SOURCE_AI_FAILED_FALLBACK,
    SCORE_SOURCE_DETERMINISTIC_ONLY,
    SUBSCORE_KEYS,
    compute_base_analysis,
    merge_scorecard_with_adjustments,
)

DEFAULT_NVIDIA_MODEL = "minimaxai/minimax-m2.7"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_TIMEOUT_S = 300.0
DEFAULT_NVIDIA_MAX_TOKENS = 32000
DEFAULT_NVIDIA_RETRIES = 1
DEFAULT_IFM_MODEL = "IFM/K2-Horizon-375B-A23B"
DEFAULT_IFM_BASE_URL = "https://api.ifm.ai/v1"
DEFAULT_IFM_TIMEOUT_S = 300.0
DEFAULT_IFM_MAX_TOKENS = 32000
DEFAULT_IFM_RETRIES = 1

MAX_REQUEST_BYTES = 64 * 1024
MAX_SUMMARY_ROWS = 8
MAX_CHARTS = 9
MAX_SERIES_POINTS = 10
MAX_HISTORY_QUARTERS = 12
MAX_HISTORY_METRIC_KEYS = 32
MAX_HISTORY_RATIO_KEYS = 24
MAX_ABS_NUMBER = 1_000_000_000_000_000_000
MAX_DEBUG_TRACE_ITEMS = 40

TOP_LEVEL_KEYS = {"company", "company_title", "latest_period", "overview_payload", "history_context", "model"}
REQUIRED_TOP_LEVEL_KEYS = {"company", "company_title", "latest_period", "overview_payload", "history_context"}
OVERVIEW_KEYS = {"income_summary", "balance_summary", "charts", "period_analysis"}
SUMMARY_ROW_KEYS = {
    "key",
    "label",
    "basis",
    "current_period",
    "current_value",
    "current_display",
    "base_period",
    "base_value",
    "base_display",
    "pct_change",
    "pct_display",
}
CHART_KEYS = {"title", "kind", "series"}
SERIES_POINT_KEYS = {"label", "value", "display"}
HISTORY_CONTEXT_KEYS = {"company_kind", "quarters"}
HISTORY_QUARTER_KEYS = {"label", "year", "period", "metrics", "ratios"}
PERIOD_ANALYSIS_KEYS = {
    "latest_period",
    "year",
    "quarter",
    "currency",
    "quarterly_flow_comparisons",
    "balance_sheet_comparisons",
}
FLOW_COMPARISON_KEYS = {
    "key",
    "label",
    "current_period",
    "current_value",
    "previous_quarter_period",
    "previous_quarter_value",
    "previous_quarter_pct_change",
    "same_quarter_last_year_period",
    "same_quarter_last_year_value",
    "same_quarter_last_year_pct_change",
    "current_ytd_period",
    "current_ytd_value",
    "prior_year_ytd_period",
    "prior_year_ytd_value",
    "prior_year_ytd_pct_change",
}
BALANCE_COMPARISON_KEYS = {
    "key",
    "label",
    "current_period",
    "current_value",
    "previous_quarter_period",
    "previous_quarter_value",
    "previous_quarter_pct_change",
    "same_quarter_last_year_period",
    "same_quarter_last_year_value",
    "same_quarter_last_year_pct_change",
}

SUPPORTED_NVIDIA_MODELS = {
    "minimaxai/minimax-m2.7",
    "meta/llama-4-maverick-17b-128e-instruct",
}
SUPPORTED_IFM_MODELS = {DEFAULT_IFM_MODEL}
SUPPORTED_OVERVIEW_MODELS = SUPPORTED_NVIDIA_MODELS | SUPPORTED_IFM_MODELS
OVERVIEW_MODEL_ALIASES = {
    "minimax": "minimaxai/minimax-m2.7",
    "minimax m2.7": "minimaxai/minimax-m2.7",
    "minimax-m2.7": "minimaxai/minimax-m2.7",
    "llama": "meta/llama-4-maverick-17b-128e-instruct",
    "llama 4": "meta/llama-4-maverick-17b-128e-instruct",
    "llama-4-maverick-17b-128e-instruct": "meta/llama-4-maverick-17b-128e-instruct",
}

LOGGER = logging.getLogger("uvicorn.error")


class PayloadValidationError(ValueError):
    """Raised for client-controlled payload shape or size problems."""


class NvidiaCommentaryError(RuntimeError):
    """Raised for provider/network/model-output problems."""


def _nvidia_model() -> str:
    raw = os.getenv("NVIDIA_AI_MODEL", "").strip()
    if not raw:
        return DEFAULT_NVIDIA_MODEL
    resolved = OVERVIEW_MODEL_ALIASES.get(raw.lower(), raw)
    if resolved in SUPPORTED_NVIDIA_MODELS:
        return resolved
    return DEFAULT_NVIDIA_MODEL


def _ifm_model() -> str:
    raw = os.getenv("IFM_MODEL", "").strip()
    return raw or DEFAULT_IFM_MODEL


def _resolve_nvidia_model(model_override: Optional[str] = None) -> str:
    chosen = str(model_override or "").strip()
    if chosen in SUPPORTED_NVIDIA_MODELS:
        return chosen
    return _nvidia_model()


def _resolve_ifm_model(model_override: Optional[str] = None) -> str:
    chosen = str(model_override or "").strip()
    if chosen in SUPPORTED_IFM_MODELS:
        return chosen
    return _ifm_model()


def _nvidia_base_url() -> str:
    return (os.getenv("NVIDIA_AI_BASE_URL", "").strip() or DEFAULT_NVIDIA_BASE_URL).rstrip("/")


def _ifm_base_url() -> str:
    return (os.getenv("IFM_BASE_URL", "").strip() or DEFAULT_IFM_BASE_URL).rstrip("/")


def _nvidia_timeout_s() -> float:
    raw = os.getenv("NVIDIA_AI_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_NVIDIA_TIMEOUT_S
    try:
        return max(5.0, min(float(raw), 900.0))
    except ValueError:
        return DEFAULT_NVIDIA_TIMEOUT_S


def _nvidia_max_tokens() -> int:
    raw = os.getenv("NVIDIA_AI_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_NVIDIA_MAX_TOKENS
    try:
        return max(128, min(int(raw), DEFAULT_NVIDIA_MAX_TOKENS))
    except ValueError:
        return DEFAULT_NVIDIA_MAX_TOKENS


def _nvidia_retry_count() -> int:
    raw = os.getenv("NVIDIA_AI_MAX_RETRIES", "").strip()
    if not raw:
        return DEFAULT_NVIDIA_RETRIES
    try:
        return max(0, min(int(raw), 3))
    except ValueError:
        return DEFAULT_NVIDIA_RETRIES


def _nvidia_debug_enabled() -> bool:
    raw = os.getenv("NVIDIA_AI_DEBUG", "").strip().lower()
    return raw in {"1", "true", "yes", "on", "debug"}


def _ifm_timeout_s() -> float:
    raw = os.getenv("IFM_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_IFM_TIMEOUT_S
    try:
        return max(5.0, min(float(raw), 900.0))
    except ValueError:
        return DEFAULT_IFM_TIMEOUT_S


def _ifm_max_tokens() -> int:
    raw = os.getenv("IFM_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_IFM_MAX_TOKENS
    try:
        return max(128, min(int(raw), DEFAULT_IFM_MAX_TOKENS))
    except ValueError:
        return DEFAULT_IFM_MAX_TOKENS


def _ifm_retry_count() -> int:
    raw = os.getenv("IFM_MAX_RETRIES", "").strip()
    if not raw:
        return DEFAULT_IFM_RETRIES
    try:
        return max(0, min(int(raw), 3))
    except ValueError:
        return DEFAULT_IFM_RETRIES


def _ifm_debug_enabled() -> bool:
    raw = os.getenv("IFM_DEBUG", "").strip().lower()
    return raw in {"1", "true", "yes", "on", "debug"}


def _append_debug_trace(
    debug_trace: Optional[List[str]],
    stage: str,
    message: str,
    **fields: Any,
) -> None:
    if debug_trace is None:
        return
    normalized_fields = {
        key: value
        for key, value in fields.items()
        if value is not None and value != ""
    }
    if normalized_fields:
        suffix = ", ".join(f"{key}={normalized_fields[key]}" for key in sorted(normalized_fields))
        line = f"{stage}: {message} | {suffix}"
    else:
        line = f"{stage}: {message}"
    debug_trace.append(line[:600])
    if len(debug_trace) > MAX_DEBUG_TRACE_ITEMS:
        del debug_trace[:-MAX_DEBUG_TRACE_ITEMS]


def _log_debug(
    debug_trace: Optional[List[str]],
    stage: str,
    message: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    _append_debug_trace(debug_trace, stage, message, **fields)
    if _nvidia_debug_enabled() or _ifm_debug_enabled():
        LOGGER.log(level, "[ai_commentary] %s", debug_trace[-1] if debug_trace else f"{stage}: {message}")


def _trim_text(value: Any, max_len: int = 240) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    return text[:max_len]


def _clean_text(value: Any, *, max_len: Optional[int] = None, default: str = "") -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    if not text:
        return default
    return text[:max_len] if max_len is not None else text


def _unknown_keys(obj: Dict[str, Any], allowed: set[str], path: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise PayloadValidationError(f"{path} beklenmeyen alan iceriyor: {', '.join(extra)}")


def _require_object(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise PayloadValidationError(f"{path} object olmali")
    return value


def _require_list(value: Any, path: str, max_len: int) -> List[Any]:
    if not isinstance(value, list):
        raise PayloadValidationError(f"{path} array olmali")
    if len(value) > max_len:
        raise PayloadValidationError(f"{path} en fazla {max_len} eleman icerebilir")
    return value


def _text(value: Any, path: str, *, max_len: int, required: bool = True) -> str:
    if value is None:
        if required:
            raise PayloadValidationError(f"{path} zorunlu")
        return ""
    if not isinstance(value, str):
        value = str(value)
    normalized = " ".join(value.replace("\x00", " ").split())
    if required and not normalized:
        raise PayloadValidationError(f"{path} bos olamaz")
    if len(normalized) > max_len:
        raise PayloadValidationError(f"{path} en fazla {max_len} karakter olabilir")
    return normalized


def _number_or_none(value: Any, path: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PayloadValidationError(f"{path} sayi olmali")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        raise PayloadValidationError(f"{path} sayi olmali")
    if not math.isfinite(number) or abs(number) > MAX_ABS_NUMBER:
        raise PayloadValidationError(f"{path} gecersiz sayi")
    return number


def _integer_field(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    number = _number_or_none(value, path)
    if number is None:
        raise PayloadValidationError(f"{path} zorunlu")
    rounded = int(number)
    if rounded != number:
        raise PayloadValidationError(f"{path} tam sayi olmali")
    if rounded < minimum or rounded > maximum:
        raise PayloadValidationError(f"{path} {minimum} ile {maximum} arasinda olmali")
    return rounded


def _normalize_requested_model(value: Any, path: str) -> Optional[str]:
    raw = _text(value, path, max_len=120, required=False)
    if not raw:
        return None
    resolved = OVERVIEW_MODEL_ALIASES.get(raw.lower(), raw)
    if resolved not in SUPPORTED_OVERVIEW_MODELS:
        allowed = ", ".join(sorted(SUPPORTED_OVERVIEW_MODELS))
        raise PayloadValidationError(f"{path} desteklenmeyen model. Desteklenen modeller: {allowed}")
    return resolved


def _normalize_summary_row(row: Any, path: str, *, default_basis: str) -> Dict[str, Any]:
    item = _require_object(row, path)
    _unknown_keys(item, SUMMARY_ROW_KEYS, path)
    normalized = {
        "key": _text(item.get("key"), f"{path}.key", max_len=64),
        "label": _text(item.get("label"), f"{path}.label", max_len=80),
        "basis": _text(item.get("basis") or default_basis, f"{path}.basis", max_len=24, required=False),
        "current_period": _text(item.get("current_period"), f"{path}.current_period", max_len=24),
        "current_value": _number_or_none(item.get("current_value"), f"{path}.current_value"),
        "current_display": _text(item.get("current_display"), f"{path}.current_display", max_len=80),
        "base_period": _text(item.get("base_period"), f"{path}.base_period", max_len=24, required=False),
        "base_value": _number_or_none(item.get("base_value"), f"{path}.base_value"),
        "base_display": _text(item.get("base_display"), f"{path}.base_display", max_len=80, required=False),
        "pct_change": _number_or_none(item.get("pct_change"), f"{path}.pct_change"),
        "pct_display": _text(item.get("pct_display"), f"{path}.pct_display", max_len=24, required=False),
    }
    if normalized["current_value"] is None and normalized["base_value"] is None:
        raise PayloadValidationError(f"{path} en az bir sayisal deger icermeli")
    if normalized["basis"] not in {"", "year_to_date", "quarter_end"}:
        raise PayloadValidationError(f"{path}.basis desteklenmeyen deger")
    return normalized


def _normalize_period_comparison(row: Any, path: str, *, flow: bool) -> Dict[str, Any]:
    item = _require_object(row, path)
    _unknown_keys(item, FLOW_COMPARISON_KEYS if flow else BALANCE_COMPARISON_KEYS, path)
    normalized: Dict[str, Any] = {
        "key": _text(item.get("key"), f"{path}.key", max_len=64),
        "label": _text(item.get("label"), f"{path}.label", max_len=80),
        "current_period": _text(item.get("current_period"), f"{path}.current_period", max_len=32, required=False),
        "current_value": _number_or_none(item.get("current_value"), f"{path}.current_value"),
        "previous_quarter_period": _text(item.get("previous_quarter_period"), f"{path}.previous_quarter_period", max_len=32, required=False),
        "previous_quarter_value": _number_or_none(item.get("previous_quarter_value"), f"{path}.previous_quarter_value"),
        "previous_quarter_pct_change": _number_or_none(item.get("previous_quarter_pct_change"), f"{path}.previous_quarter_pct_change"),
        "same_quarter_last_year_period": _text(item.get("same_quarter_last_year_period"), f"{path}.same_quarter_last_year_period", max_len=32, required=False),
        "same_quarter_last_year_value": _number_or_none(item.get("same_quarter_last_year_value"), f"{path}.same_quarter_last_year_value"),
        "same_quarter_last_year_pct_change": _number_or_none(item.get("same_quarter_last_year_pct_change"), f"{path}.same_quarter_last_year_pct_change"),
    }
    if flow:
        normalized.update({
            "current_ytd_period": _text(item.get("current_ytd_period"), f"{path}.current_ytd_period", max_len=40, required=False),
            "current_ytd_value": _number_or_none(item.get("current_ytd_value"), f"{path}.current_ytd_value"),
            "prior_year_ytd_period": _text(item.get("prior_year_ytd_period"), f"{path}.prior_year_ytd_period", max_len=40, required=False),
            "prior_year_ytd_value": _number_or_none(item.get("prior_year_ytd_value"), f"{path}.prior_year_ytd_value"),
            "prior_year_ytd_pct_change": _number_or_none(item.get("prior_year_ytd_pct_change"), f"{path}.prior_year_ytd_pct_change"),
        })
    if all(value is None for key, value in normalized.items() if key.endswith("_value")):
        raise PayloadValidationError(f"{path} en az bir sayisal deger icermeli")
    return normalized


def _normalize_period_analysis(value: Any, fallback_period: str) -> Dict[str, Any]:
    if value is None:
        return {
            "latest_period": fallback_period,
            "year": 0,
            "quarter": 0,
            "currency": "",
            "quarterly_flow_comparisons": [],
            "balance_sheet_comparisons": [],
        }
    item = _require_object(value, "overview_payload.period_analysis")
    _unknown_keys(item, PERIOD_ANALYSIS_KEYS, "overview_payload.period_analysis")
    raw_flows = _require_list(item.get("quarterly_flow_comparisons"), "overview_payload.period_analysis.quarterly_flow_comparisons", 16)
    raw_balances = _require_list(item.get("balance_sheet_comparisons"), "overview_payload.period_analysis.balance_sheet_comparisons", 16)
    year = _integer_field(item.get("year"), "overview_payload.period_analysis.year", minimum=0, maximum=2100)
    quarter = _integer_field(item.get("quarter"), "overview_payload.period_analysis.quarter", minimum=0, maximum=4)
    if (year == 0) != (quarter == 0):
        raise PayloadValidationError("overview_payload.period_analysis.year ve quarter birlikte bos olmali")
    return {
        "latest_period": _text(item.get("latest_period"), "overview_payload.period_analysis.latest_period", max_len=32, required=False),
        "year": year,
        "quarter": quarter,
        "currency": _text(item.get("currency"), "overview_payload.period_analysis.currency", max_len=12, required=False),
        "quarterly_flow_comparisons": [
            _normalize_period_comparison(row, f"overview_payload.period_analysis.quarterly_flow_comparisons[{idx}]", flow=True)
            for idx, row in enumerate(raw_flows)
        ],
        "balance_sheet_comparisons": [
            _normalize_period_comparison(row, f"overview_payload.period_analysis.balance_sheet_comparisons[{idx}]", flow=False)
            for idx, row in enumerate(raw_balances)
        ],
    }


def _normalize_series_point(point: Any, path: str) -> Dict[str, Any]:
    item = _require_object(point, path)
    _unknown_keys(item, SERIES_POINT_KEYS, path)
    value = _number_or_none(item.get("value"), f"{path}.value")
    if value is None:
        raise PayloadValidationError(f"{path}.value zorunlu")
    return {
        "label": _text(item.get("label"), f"{path}.label", max_len=32),
        "value": value,
        "display": _text(item.get("display"), f"{path}.display", max_len=80),
    }


def _normalize_chart(chart: Any, path: str) -> Dict[str, Any]:
    item = _require_object(chart, path)
    _unknown_keys(item, CHART_KEYS, path)
    kind = _text(item.get("kind"), f"{path}.kind", max_len=12).lower()
    if kind not in {"bar", "line"}:
        raise PayloadValidationError(f"{path}.kind bar veya line olmali")
    raw_series = _require_list(item.get("series"), f"{path}.series", MAX_SERIES_POINTS)
    if not raw_series:
        raise PayloadValidationError(f"{path}.series bos olamaz")
    return {
        "title": _text(item.get("title"), f"{path}.title", max_len=100),
        "kind": kind,
        "series": [
            _normalize_series_point(point, f"{path}.series[{idx}]")
            for idx, point in enumerate(raw_series)
        ],
    }


def _normalize_value_map(value: Any, path: str, *, max_items: int) -> Dict[str, Optional[float]]:
    obj = _require_object(value, path)
    if len(obj) > max_items:
        raise PayloadValidationError(f"{path} en fazla {max_items} alan icerebilir")
    normalized: Dict[str, Optional[float]] = {}
    for raw_key, raw_value in obj.items():
        key = _text(raw_key, f"{path}.key", max_len=64)
        normalized[key] = _number_or_none(raw_value, f"{path}.{key}")
    return normalized


def _normalize_history_quarter(row: Any, path: str) -> Dict[str, Any]:
    item = _require_object(row, path)
    _unknown_keys(item, HISTORY_QUARTER_KEYS, path)
    return {
        "label": _text(item.get("label"), f"{path}.label", max_len=24),
        "year": _integer_field(item.get("year"), f"{path}.year", minimum=2000, maximum=2100),
        "period": _integer_field(item.get("period"), f"{path}.period", minimum=1, maximum=12),
        "metrics": _normalize_value_map(item.get("metrics"), f"{path}.metrics", max_items=MAX_HISTORY_METRIC_KEYS),
        "ratios": _normalize_value_map(item.get("ratios"), f"{path}.ratios", max_items=MAX_HISTORY_RATIO_KEYS),
    }


def _normalize_history_context(value: Any) -> Dict[str, Any]:
    history = _require_object(value, "history_context")
    _unknown_keys(history, HISTORY_CONTEXT_KEYS, "history_context")
    company_kind = _text(history.get("company_kind"), "history_context.company_kind", max_len=24).lower()
    if company_kind not in COMPANY_KINDS:
        allowed = ", ".join(sorted(COMPANY_KINDS))
        raise PayloadValidationError(f"history_context.company_kind desteklenmeyen deger. Desteklenenler: {allowed}")

    raw_quarters = _require_list(history.get("quarters"), "history_context.quarters", MAX_HISTORY_QUARTERS)
    if not raw_quarters:
        raise PayloadValidationError("history_context.quarters bos olamaz")
    quarters = [
        _normalize_history_quarter(row, f"history_context.quarters[{idx}]")
        for idx, row in enumerate(raw_quarters)
    ]
    quarters.sort(key=lambda row: (int(row["year"]), int(row["period"])))
    return {
        "company_kind": company_kind,
        "quarters": quarters,
    }


def validate_overview_commentary_request(payload: Any) -> Dict[str, Any]:
    root = _require_object(payload, "request")
    _unknown_keys(root, TOP_LEVEL_KEYS, "request")
    missing = sorted(key for key in REQUIRED_TOP_LEVEL_KEYS if key not in root)
    if missing:
        raise PayloadValidationError(f"zorunlu alan eksik: {', '.join(missing)}")

    overview = _require_object(root.get("overview_payload"), "overview_payload")
    _unknown_keys(overview, OVERVIEW_KEYS, "overview_payload")

    income_rows = _require_list(overview.get("income_summary"), "overview_payload.income_summary", MAX_SUMMARY_ROWS)
    balance_rows = _require_list(overview.get("balance_summary"), "overview_payload.balance_summary", MAX_SUMMARY_ROWS)
    charts = _require_list(overview.get("charts"), "overview_payload.charts", MAX_CHARTS)

    return {
        "company": _text(root.get("company"), "company", max_len=32),
        "company_title": _text(root.get("company_title"), "company_title", max_len=180),
        "latest_period": _text(root.get("latest_period"), "latest_period", max_len=24),
        "model": _normalize_requested_model(root.get("model"), "model"),
        "overview_payload": {
            "income_summary": [
                _normalize_summary_row(row, f"overview_payload.income_summary[{idx}]", default_basis="year_to_date")
                for idx, row in enumerate(income_rows)
            ],
            "balance_summary": [
                _normalize_summary_row(row, f"overview_payload.balance_summary[{idx}]", default_basis="quarter_end")
                for idx, row in enumerate(balance_rows)
            ],
            "period_analysis": _normalize_period_analysis(overview.get("period_analysis"), _text(root.get("latest_period"), "latest_period", max_len=24)),
            "charts": [
                _normalize_chart(chart, f"overview_payload.charts[{idx}]")
                for idx, chart in enumerate(charts)
            ],
        },
        "history_context": _normalize_history_context(root.get("history_context")),
    }


def _system_prompt() -> str:
    return (
        "Turkce finansal analiz ureten bir asistansin.\n"
        "Kurallar:\n"
        "- Sadece verilen JSON verisini kullan.\n"
        "- Analizin ana konusu en son KAP'ta aciklanan donemdir; yillik/TTM toplamlari son ceyregin yerine koyma. Donem etiketini acikca kullan (ornegin 2026 Q2).\n"
        "- period_analysis.quarterly_flow_comparisons gelir tablosu ve nakit akisinin tek ceyreklik akisini verir. Once son ceyregi ardil onceki ceyrekle, sonra gecen yilin ayni ceyregiyle karsilastir.\n"
        "- Ayni akista current_ytd_value ile prior_year_ytd_value, icinde bulunulan rapor yilinin basindan son aciklanan ceyrege kadarki birikimli tutarlari onceki yilin ayni donem birikimiyle karsilastirir. Bunlari tek ceyrek tutari gibi adlandirma. 2026 Q2 varsa 2026 YTD, ilk yari verisidir; Q3/Q4 aciklanmadiysa yil sonu sonucu veya tahmini uretme.\n"
        "- period_analysis.balance_sheet_comparisons bilanconun ceyrek sonu stok degerleridir. Son donemi onceki ceyrek sonu ve gecen yil ayni ceyrek sonuyla ayri ayri karsilastir; bunlari gelir/nakit akis toplami gibi yorumlama.\n"
        "- income_summary satirlari YTD, balance_summary satirlari ceyrek sonu degeridir; basis alanini ve donem etiketlerini dikkate al. period_analysis varsa karsilastirmalarda onu esas al. Tarihsel quarters dizisini mevsimsellik, trend ve tek ceyreklik sapmalari kontrol etmek icin kullan; tarihsel ortalama son aciklanan ceyregi gormezden getirmesin.\n"
        "- Sirket turune uygun metrikleri sec: genel sirketlerde satis, faaliyet karliligi/marjlar, net kar, nakit donusumu ve borc; bankalarda faiz/komisyon, kredi-mevduat, karsilik ve ozkaynak; sigortada prim, teknik denge, karsilik ve ozkaynak. Alakasiz sektorel metrikleri yorumlama.\n"
        "- Tarihsel ratios.roe degeri dort ardışık ceyreklik net karin ortalama ozkaynakla hesaplanan TTM ROE'sidir; null ise yillik getiri yorumu yapma. Diger marjlari son ceyrekteki akislardan, likidite/borc oranlarini ceyrek sonu bilancosundan hesapla.\n"
        "- Rakam listesini tekrar etme. Birbirine bagli 2-4 onemli etkene odaklan: buyume -> faaliyet kari ve marj -> net kar; kar -> faaliyet/serbest nakit akisina donusum; bilanco -> likidite, borcluluk veya sektorel risk. Olumlu ve olumsuz sinyalleri birlikte tart.\n"
        "- Yuzde degisimde ozellikle onceki deger sifira yakin ya da negatifse matematiksel yuzdeyi ekonomik buyume gibi sunma; tutar yonunu ve karliligin zarardan kara gecip gecmedigini acikla. Veri yoksa tahmin etme. Karsilastirma donemi yoksa bunu belirt; fiyat hedefi, rakip kiyaslamasi veya dis bilgi ekleme.\n"
        "- Veriler celiskiliyse, tek seferlik etki oldugu veriden anlasilmiyorsa bunu kesin neden gibi yazma. Sinyaller karisik/zayifsa score adjustment'i sifira yakin veya sifir tut.\n"
        "- Karlilik adjustment'inda son ceyrekteki net kar ve esas faaliyet kari artislarini ayni ceyrek yillik karsilastirmasi, YTD teyidi, marjlar ve tarihsel trendle birlikte tart. Pozitif ve maddi bir baz uzerinde net kar ile faaliyet kari birlikte gucleniyorsa, yalnizca mutlak marj seviyesi nedeniyle karlilik adjustment'ini otomatik sifirda birakma. Dusuk/sifira yakin baz veya zarardan kara geciste yuzde buyumeyi abartma. Net kar artisini bilanco gucu kaniti sayma; bilanco adjustment'ini borc, likidite, ozkaynak ve ilgili sektor bilancosi gostergelerine dayandir.\n"
        "- Final scorecard uretme; sadece bounded adjustment ve kisa gerekce uret.\n"
        "- Rakam uydurma, dis bilgi kullanma, fiyat hedefi verme.\n"
        "- Cumleleri tamamla; metin alanlarini yarim birakma.\n"
        "- Headline en son rapor donemini ve genel yonu soylesin. Bullets alaninda 3-5 kisa, kanita dayali bulgu yaz; mumkunse son ceyrek, ayni ceyrek yillik/YTD karsilastirmasi, nakit kalitesi ve bilancodan en anlamli konulari kapsa.\n"
        "- score_adjustments.overall_adjustment -0.5 ile 0.5 arasinda olsun.\n"
        "- score_adjustments.subscores yalniz buyume, karlilik, bilanco, nakit_akisi keylerini kullansin.\n"
        "- Her subscore adjustment -1.0 ile 1.0 arasinda olsun.\n"
        "- Yalniz su JSON semasini dondur:\n"
        '{"headline":"","bullets":[],"risk_note":"","watch_metrics":[],"summary":"","seasonality_note":"","score_adjustments":{"overall_adjustment":0,"subscores":[{"key":"buyume","adjustment":0,"summary":""},{"key":"karlilik","adjustment":0,"summary":""},{"key":"bilanco","adjustment":0,"summary":""},{"key":"nakit_akisi","adjustment":0,"summary":""}]}}'
    )


def _user_prompt(normalized_payload: Dict[str, Any], base_analysis: Dict[str, Any]) -> str:
    prompt_payload = {
        "company": normalized_payload["company"],
        "company_title": normalized_payload["company_title"],
        "latest_period": normalized_payload["latest_period"],
        "overview_payload": normalized_payload["overview_payload"],
        "history_context": normalized_payload["history_context"],
        "base_scorecard": base_analysis["scorecard"],
    }
    return (
        "Son aciklanan KAP finansal donemi, karsilastirma paketleri, 12 ceyrege kadar tarihce ve deterministic base scorecard asagidadir. Donem kiyaslari hazir hesaplanmistir; once bazlarini kontrol et:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)}\n\n"
        "Once period_analysis.latest_period donemini merkez al. Akis kalemleri icin standalone ceyrek karsilastirmalarini, yil icindeki birikimli performans icin YTD karsilastirmasini; bilanco kalemleri icin ceyrek sonu stok karsilastirmalarini birbirinden ayri yorumla. Ardindan tarihceyle mevsimsellik ve trendi dogrula. Base scorecard'i tamamen degistirme; sadece birbiriyle tutarli ve maddi kanit varsa sinirli adjustment uygula."
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    decoder = json.JSONDecoder()
    first_object: Optional[Dict[str, Any]] = None
    last_error: Optional[json.JSONDecodeError] = None
    cursor = 0

    while cursor < len(raw):
        start = raw.find("{", cursor)
        if start < 0:
            break
        try:
            parsed, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError as exc:
            last_error = exc
            cursor = start + 1
            continue

        if isinstance(parsed, dict):
            if first_object is None:
                first_object = parsed
            adjustments = parsed.get("score_adjustments")
            if isinstance(adjustments, dict) and isinstance(adjustments.get("subscores"), list):
                return parsed
        cursor = max(end, start + 1)

    if first_object is not None:
        return first_object
    if last_error is not None:
        raise NvidiaCommentaryError(f"AI JSON parse hatasi: {last_error}") from last_error
    raise NvidiaCommentaryError("AI modeli JSON formatinda yanit donmedi")


def _response_text_from_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise NvidiaCommentaryError("AI yaniti object degil")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise NvidiaCommentaryError("AI yanitinda choices yok")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message.get("content") or "")
    delta = first.get("delta") if isinstance(first, dict) else {}
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta.get("content") or "")
    raise NvidiaCommentaryError("AI bos icerik dondu")


def _text_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = " ".join(str(item or "").replace("\x00", " ").split())
        if text:
            result.append(text)
    return result


def _normalized_adjustment_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_adjustments = payload.get("score_adjustments")
    if not isinstance(raw_adjustments, dict):
        raise NvidiaCommentaryError("AI yanitinda score_adjustments object'i yok")

    overall_adjustment_raw = raw_adjustments.get("overall_adjustment", 0.0)
    if overall_adjustment_raw is None:
        overall_adjustment_raw = 0.0
    if isinstance(overall_adjustment_raw, bool) or not isinstance(overall_adjustment_raw, (int, float)):
        raise NvidiaCommentaryError("AI overall_adjustment sayi olmali")

    raw_subscores = raw_adjustments.get("subscores")
    if not isinstance(raw_subscores, list):
        raise NvidiaCommentaryError("AI score_adjustments.subscores array olmali")

    normalized_subscores: List[Dict[str, Any]] = []
    for raw_item in raw_subscores:
        if not isinstance(raw_item, dict):
            continue
        key = str(raw_item.get("key") or "").strip()
        if key not in SUBSCORE_KEYS:
            continue
        adjustment_raw = raw_item.get("adjustment", 0.0)
        if adjustment_raw is None:
            adjustment_raw = 0.0
        if isinstance(adjustment_raw, bool) or not isinstance(adjustment_raw, (int, float)):
            raise NvidiaCommentaryError(f"AI adjustment sayi olmali: {key}")
        normalized_subscores.append(
            {
                "key": key,
                "adjustment": float(adjustment_raw),
                "summary": _clean_text(raw_item.get("summary")),
            }
        )

    if not normalized_subscores:
        raise NvidiaCommentaryError("AI yanitinda gecerli subscore adjustment yok")

    return {
        "overall_adjustment": float(overall_adjustment_raw),
        "subscores": normalized_subscores,
        "summary": _clean_text(payload.get("summary")),
        "seasonality_note": _clean_text(payload.get("seasonality_note")),
    }


def _build_success_response(
    *,
    model: str,
    scorecard: Dict[str, Any],
    headline: str,
    bullets: List[str],
    risk_note: str,
    watch_metrics: List[str],
    error: Optional[str],
    debug_trace: Optional[List[str]] = None,
) -> Dict[str, Any]:
    response = {
        "ok": True,
        "headline": headline,
        "bullets": bullets,
        "risk_note": risk_note,
        "watch_metrics": watch_metrics,
        "model_used": model,
        "scorecard": scorecard,
        "error": error,
    }
    if debug_trace:
        response["debug_trace"] = debug_trace[:MAX_DEBUG_TRACE_ITEMS]
    return response


def _build_fallback_response(
    *,
    base_analysis: Dict[str, Any],
    model: str,
    score_source: str,
    error: Optional[str],
    debug_trace: Optional[List[str]] = None,
) -> Dict[str, Any]:
    scorecard = {
        **base_analysis["scorecard"],
        "score_source": score_source,
    }
    return _build_success_response(
        model=model,
        scorecard=scorecard,
        headline=base_analysis["headline"],
        bullets=list(base_analysis["bullets"]),
        risk_note=base_analysis["risk_note"],
        watch_metrics=list(base_analysis["watch_metrics"]),
        error=error,
        debug_trace=debug_trace,
    )


def _normalize_model_json(
    payload: Dict[str, Any],
    *,
    base_analysis: Dict[str, Any],
    model: str,
    debug_trace: Optional[List[str]] = None,
) -> Dict[str, Any]:
    adjustments = _normalized_adjustment_payload(payload)
    _log_debug(
        debug_trace,
        "scoring",
        "AI adjustment parse edildi",
        overall_adjustment=adjustments["overall_adjustment"],
        adjusted_subscores=",".join(
            f"{item['key']}={item['adjustment']}" for item in adjustments["subscores"]
        ),
    )
    scorecard = merge_scorecard_with_adjustments(
        base_analysis["scorecard"],
        adjustments,
        score_source=SCORE_SOURCE_AI_ADJUSTED,
    )

    headline = _clean_text(payload.get("headline"), default=base_analysis["headline"])
    bullets = _text_list(payload.get("bullets")) or list(base_analysis["bullets"])
    risk_note = _clean_text(payload.get("risk_note"), default=base_analysis["risk_note"])
    watch_metrics = _text_list(payload.get("watch_metrics")) or list(base_analysis["watch_metrics"])

    return _build_success_response(
        model=model,
        scorecard=scorecard,
        headline=headline,
        bullets=bullets,
        risk_note=risk_note,
        watch_metrics=watch_metrics,
        error=None,
        debug_trace=debug_trace,
    )


async def _call_chat_completion(
    normalized_payload: Dict[str, Any],
    base_analysis: Dict[str, Any],
    api_key: str,
    model: str,
    *,
    provider: str,
    base_url: str,
    timeout_s: float,
    max_tokens: int,
    retry_count: int,
    debug_trace: Optional[List[str]] = None,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(normalized_payload, base_analysis)},
        ],
        "temperature": 1.0 if provider == "IFM" else 0.4,
        "top_p": 0.95 if provider == "IFM" else 0.9,
        "max_tokens": max_tokens,
    }
    if provider == "IFM":
        body["chat_template_kwargs"] = {"reasoning_effort": "low"}
        body["response_format"] = {"type": "json_object"}
    else:
        body["chat_template_kwargs"] = {"thinking": False}
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Optional[BaseException] = None
    total_attempts = retry_count + 1
    timeout = httpx.Timeout(
        timeout_s,
        connect=min(30.0, timeout_s),
        read=timeout_s,
        write=min(30.0, timeout_s),
        pool=min(30.0, timeout_s),
    )
    _log_debug(
        debug_trace,
        "request",
        f"{provider} overview commentary request hazirlandi",
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        retries=retry_count,
        payload_bytes=len(body_bytes),
        company=normalized_payload.get("company"),
        latest_period=normalized_payload.get("latest_period"),
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, total_attempts + 1):
            started_at = time.perf_counter()
            _log_debug(
                debug_trace,
                "request",
                f"{provider} istegi gonderiliyor",
                attempt=attempt,
                total_attempts=total_attempts,
            )
            try:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    content=body_bytes,
                    headers=headers,
                )
                response.raise_for_status()
                response_payload = response.json()
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
                choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
                finish_reason = None
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    finish_reason = choices[0].get("finish_reason")
                _log_debug(
                    debug_trace,
                    "response",
                    f"{provider} yaniti alindi",
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    finish_reason=finish_reason,
                    usage=_trim_text(usage, 180) if usage is not None else None,
                )
                return _response_text_from_payload(response_payload)
            except asyncio.CancelledError:
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                _log_debug(
                    debug_trace,
                    "cancel",
                    f"{provider} istegi iptal edildi",
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                )
                raise
            except httpx.HTTPStatusError as exc:
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                detail = exc.response.text if exc.response is not None else str(exc)
                status_code = exc.response.status_code if exc.response is not None else 0
                last_error = exc
                _log_debug(
                    debug_trace,
                    "error",
                    f"{provider} HTTP hatasi",
                    level=logging.WARNING,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    status_code=status_code,
                    detail=_trim_text(detail),
                )
                if status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < total_attempts:
                    backoff_s = min(4.0, float(attempt))
                    _log_debug(
                        debug_trace,
                        "retry",
                        f"{provider} istegi yeniden denenecek",
                        level=logging.WARNING,
                        attempt=attempt,
                        backoff_s=backoff_s,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                raise NvidiaCommentaryError(f"{provider} HTTP hata: {status_code} {detail[:240]}".strip()) from exc
            except (httpx.RequestError, json.JSONDecodeError, ValueError) as exc:
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                last_error = exc
                _log_debug(
                    debug_trace,
                    "error",
                    f"{provider} baglanti/parsing hatasi",
                    level=logging.WARNING,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                    detail=_trim_text(exc),
                )
                if attempt < total_attempts:
                    backoff_s = min(4.0, float(attempt))
                    _log_debug(
                        debug_trace,
                        "retry",
                        f"{provider} istegi yeniden denenecek",
                        level=logging.WARNING,
                        attempt=attempt,
                        backoff_s=backoff_s,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                raise NvidiaCommentaryError(f"{provider} baglanti hatasi: {exc}") from exc
    raise NvidiaCommentaryError(f"{provider} baglanti hatasi: {last_error}")


async def _call_nvidia_chat(
    normalized_payload: Dict[str, Any],
    base_analysis: Dict[str, Any],
    api_key: str,
    model: str,
    *,
    debug_trace: Optional[List[str]] = None,
) -> str:
    return await _call_chat_completion(
        normalized_payload,
        base_analysis,
        api_key,
        model,
        provider="NVIDIA",
        base_url=_nvidia_base_url(),
        timeout_s=_nvidia_timeout_s(),
        max_tokens=_nvidia_max_tokens(),
        retry_count=_nvidia_retry_count(),
        debug_trace=debug_trace,
    )


async def _call_ifm_chat(
    normalized_payload: Dict[str, Any],
    base_analysis: Dict[str, Any],
    api_key: str,
    model: str,
    *,
    debug_trace: Optional[List[str]] = None,
) -> str:
    return await _call_chat_completion(
        normalized_payload,
        base_analysis,
        api_key,
        model,
        provider="IFM",
        base_url=_ifm_base_url(),
        timeout_s=_ifm_timeout_s(),
        max_tokens=_ifm_max_tokens(),
        retry_count=_ifm_retry_count(),
        debug_trace=debug_trace,
    )


async def generate_overview_commentary(payload: Dict[str, Any]) -> Dict[str, Any]:
    debug_trace: List[str] = []
    ifm_api_key = os.getenv("IFM_API_KEY", "").strip()
    legacy_nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    use_ifm = bool(ifm_api_key)
    default_model = _ifm_model() if use_ifm or not legacy_nvidia_api_key else _nvidia_model()
    _log_debug(
        debug_trace,
        "request",
        "Overview commentary istegi alindi",
        default_model=default_model,
        payload_bytes=len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
    )
    normalized_payload = validate_overview_commentary_request(payload)
    model = (
        _resolve_ifm_model(normalized_payload.get("model"))
        if use_ifm
        else _resolve_nvidia_model(normalized_payload.get("model"))
    )
    provider = "IFM" if use_ifm else "NVIDIA"
    _log_debug(
        debug_trace,
        "validation",
        "Overview commentary payload dogrulandi",
        model=model,
        income_rows=len(normalized_payload.get("overview_payload", {}).get("income_summary", [])),
        balance_rows=len(normalized_payload.get("overview_payload", {}).get("balance_summary", [])),
        charts=len(normalized_payload.get("overview_payload", {}).get("charts", [])),
        history_quarters=len(normalized_payload.get("history_context", {}).get("quarters", [])),
        company_kind=normalized_payload.get("history_context", {}).get("company_kind"),
    )

    base_analysis = compute_base_analysis(normalized_payload["history_context"])
    for line in base_analysis.get("debug_lines", []):
        _log_debug(debug_trace, "scoring", line)

    api_key = ifm_api_key or legacy_nvidia_api_key
    if not api_key:
        missing_key_error = "IFM_API_KEY bulunamadi; NVIDIA_API_KEY bulunamadi"
        _log_debug(debug_trace, "error", missing_key_error, level=logging.WARNING)
        return _build_fallback_response(
            base_analysis=base_analysis,
            model=model,
            score_source=SCORE_SOURCE_DETERMINISTIC_ONLY,
            error=missing_key_error,
            debug_trace=debug_trace,
        )

    try:
        chat_call = _call_ifm_chat if use_ifm else _call_nvidia_chat
        maybe_response_text = chat_call(
            normalized_payload,
            base_analysis,
            api_key=api_key,
            model=model,
            debug_trace=debug_trace,
        )
        if inspect.isawaitable(maybe_response_text):
            response_text = await maybe_response_text
        else:
            response_text = str(maybe_response_text)
        _log_debug(
            debug_trace,
            "response",
            f"{provider} metin yaniti alindi",
            response_preview=_trim_text(response_text, 180),
        )
        model_json = _extract_json_object(response_text)
        _log_debug(
            debug_trace,
            "parse",
            f"{provider} yaniti JSON object olarak parse edildi",
            keys=",".join(sorted(model_json.keys())),
        )
        return _normalize_model_json(
            model_json,
            base_analysis=base_analysis,
            model=model,
            debug_trace=debug_trace,
        )
    except asyncio.CancelledError:
        _log_debug(
            debug_trace,
            "cancel",
            "Overview commentary istegi iptal edildi",
            level=logging.INFO,
        )
        raise
    except NvidiaCommentaryError as exc:
        _log_debug(
            debug_trace,
            "error",
            "Overview commentary AI fallback tetiklendi",
            level=logging.WARNING,
            detail=_trim_text(exc),
        )
        return _build_fallback_response(
            base_analysis=base_analysis,
            model=model,
            score_source=SCORE_SOURCE_AI_FAILED_FALLBACK,
            error=str(exc),
            debug_trace=debug_trace,
        )
