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
OVERVIEW_KEYS = {"income_summary", "balance_summary", "charts"}
SUMMARY_ROW_KEYS = {
    "key",
    "label",
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

SUPPORTED_OVERVIEW_MODELS = {
    "minimaxai/minimax-m2.7",
    "meta/llama-4-maverick-17b-128e-instruct",
}
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
    if resolved in SUPPORTED_OVERVIEW_MODELS:
        return resolved
    return DEFAULT_NVIDIA_MODEL


def _resolve_nvidia_model(model_override: Optional[str] = None) -> str:
    chosen = str(model_override or "").strip()
    if chosen:
        return chosen
    return _nvidia_model()


def _nvidia_base_url() -> str:
    return (os.getenv("NVIDIA_AI_BASE_URL", "").strip() or DEFAULT_NVIDIA_BASE_URL).rstrip("/")


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
    if _nvidia_debug_enabled():
        LOGGER.log(level, "[nvidia_commentary] %s", debug_trace[-1] if debug_trace else f"{stage}: {message}")


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


def _normalize_summary_row(row: Any, path: str) -> Dict[str, Any]:
    item = _require_object(row, path)
    _unknown_keys(item, SUMMARY_ROW_KEYS, path)
    normalized = {
        "key": _text(item.get("key"), f"{path}.key", max_len=64),
        "label": _text(item.get("label"), f"{path}.label", max_len=80),
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
    return normalized


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
                _normalize_summary_row(row, f"overview_payload.income_summary[{idx}]")
                for idx, row in enumerate(income_rows)
            ],
            "balance_summary": [
                _normalize_summary_row(row, f"overview_payload.balance_summary[{idx}]")
                for idx, row in enumerate(balance_rows)
            ],
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
        "- Final scorecard uretme; sadece bounded adjustment ve kisa gerekce uret.\n"
        "- Rakam uydurma, dis bilgi kullanma, fiyat hedefi verme.\n"
        "- Cumleleri tamamla; metin alanlarini yarim birakma.\n"
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
        "Genel bakis ekranindan uretilen finansal veri ve deterministic base scorecard asagidadir:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)}\n\n"
        "Base scorecard'i tamamen degistirme. Sadece gerekiyorsa sinirli adjustment uygula ve kisa finansal yorum yaz."
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise NvidiaCommentaryError("NVIDIA modeli JSON formatinda yanit donmedi")
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise NvidiaCommentaryError(f"NVIDIA JSON parse hatasi: {exc}") from exc
    if not isinstance(parsed, dict):
        raise NvidiaCommentaryError("NVIDIA yaniti object olmali")
    return parsed


def _response_text_from_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise NvidiaCommentaryError("NVIDIA yaniti object degil")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise NvidiaCommentaryError("NVIDIA yanitinda choices yok")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message.get("content") or "")
    delta = first.get("delta") if isinstance(first, dict) else {}
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta.get("content") or "")
    raise NvidiaCommentaryError("NVIDIA bos icerik dondu")


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
        raise NvidiaCommentaryError("NVIDIA yanitinda score_adjustments object'i yok")

    overall_adjustment_raw = raw_adjustments.get("overall_adjustment", 0.0)
    if overall_adjustment_raw is None:
        overall_adjustment_raw = 0.0
    if isinstance(overall_adjustment_raw, bool) or not isinstance(overall_adjustment_raw, (int, float)):
        raise NvidiaCommentaryError("NVIDIA overall_adjustment sayi olmali")

    raw_subscores = raw_adjustments.get("subscores")
    if not isinstance(raw_subscores, list):
        raise NvidiaCommentaryError("NVIDIA score_adjustments.subscores array olmali")

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
            raise NvidiaCommentaryError(f"NVIDIA adjustment sayi olmali: {key}")
        normalized_subscores.append(
            {
                "key": key,
                "adjustment": float(adjustment_raw),
                "summary": _clean_text(raw_item.get("summary")),
            }
        )

    if not normalized_subscores:
        raise NvidiaCommentaryError("NVIDIA yanitinda gecerli subscore adjustment yok")

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


async def _call_nvidia_chat(
    normalized_payload: Dict[str, Any],
    base_analysis: Dict[str, Any],
    api_key: str,
    model: str,
    *,
    debug_trace: Optional[List[str]] = None,
) -> str:
    timeout_s = _nvidia_timeout_s()
    max_tokens = _nvidia_max_tokens()
    base_url = _nvidia_base_url()
    retry_count = _nvidia_retry_count()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(normalized_payload, base_analysis)},
        ],
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }
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
        "NVIDIA overview commentary request hazirlandi",
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
                "NVIDIA istegi gonderiliyor",
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
                    "NVIDIA yaniti alindi",
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
                    "NVIDIA istegi iptal edildi",
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
                    "NVIDIA HTTP hatasi",
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
                        "NVIDIA istegi yeniden denenecek",
                        level=logging.WARNING,
                        attempt=attempt,
                        backoff_s=backoff_s,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                raise NvidiaCommentaryError(f"NVIDIA HTTP hata: {status_code} {detail[:240]}".strip()) from exc
            except (httpx.RequestError, json.JSONDecodeError, ValueError) as exc:
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                last_error = exc
                _log_debug(
                    debug_trace,
                    "error",
                    "NVIDIA baglanti/parsing hatasi",
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
                        "NVIDIA istegi yeniden denenecek",
                        level=logging.WARNING,
                        attempt=attempt,
                        backoff_s=backoff_s,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                raise NvidiaCommentaryError(f"NVIDIA baglanti hatasi: {exc}") from exc
    raise NvidiaCommentaryError(f"NVIDIA baglanti hatasi: {last_error}")


async def generate_overview_commentary(payload: Dict[str, Any]) -> Dict[str, Any]:
    debug_trace: List[str] = []
    _log_debug(
        debug_trace,
        "request",
        "Overview commentary istegi alindi",
        default_model=_nvidia_model(),
        payload_bytes=len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
    )
    normalized_payload = validate_overview_commentary_request(payload)
    model = _resolve_nvidia_model(normalized_payload.get("model"))
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

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        _log_debug(debug_trace, "error", "NVIDIA_API_KEY bulunamadi", level=logging.WARNING)
        return _build_fallback_response(
            base_analysis=base_analysis,
            model=model,
            score_source=SCORE_SOURCE_DETERMINISTIC_ONLY,
            error="NVIDIA_API_KEY bulunamadi",
            debug_trace=debug_trace,
        )

    try:
        maybe_response_text = _call_nvidia_chat(
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
            "NVIDIA metin yaniti alindi",
            response_preview=_trim_text(response_text, 180),
        )
        model_json = _extract_json_object(response_text)
        _log_debug(
            debug_trace,
            "parse",
            "NVIDIA yaniti JSON object olarak parse edildi",
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
