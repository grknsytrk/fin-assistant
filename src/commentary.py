from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional

from src.openrouter_client import OpenRouterClient

SAFE_EMPTY_COMMENTARY: Dict[str, Any] = {
    "headline": "",
    "bullets": [],
    "risk_note": "",
    "next_question": "",
}


KNOWN_COMPANY_ALIASES: Dict[str, List[str]] = {
    "BIM": ["BIM", "BIMAS"],
    "MIGROS": ["MIGROS", "MGROS"],
    "SOK": ["SOK", "SOKM"],
    "TAV": ["TAV", "TAVHL"],
    "MAVI": ["MAVI"],
    "NETCAD": ["NETCAD"],
    "ORGE": ["ORGE"],
}

METRIC_LABELS: Dict[str, str] = {
    "net_kar": "Net donem kari",
    "satis_gelirleri": "Satis gelirleri",
    "brut_kar": "Brut kar",
    "favok": "FAVOK",
    "faaliyet_nakit_akisi": "Faaliyet nakit akisi",
    "serbest_nakit_akisi": "Serbest nakit akisi",
    "capex": "CAPEX",
    "ozkaynaklar": "Ozkaynaklar",
    "toplam_varliklar": "Toplam varliklar",
    "finansal_borclar": "Finansal borclar",
    "net_borc": "Net borc",
    "brut_kar_marji": "Brut kar marji",
    "favok_marji": "FAVOK marji",
    "net_kar_marji": "Net kar marji",
    "cari_oran": "Cari oran",
    "ozkaynak_karliligi": "Ozkaynak karliligi",
}


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except Exception:
        return False


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if _is_nan(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        fval = float(value)
        if math.isinf(fval):
            return None
        return round(fval, 6)
    return str(value).strip()


def _clean_map(payload: Dict[str, Any], max_items: int = 12) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key in list(payload.keys())[:max_items]:
        cleaned = _clean_scalar(payload.get(key))
        if cleaned is not None and cleaned != "":
            output[str(key)] = cleaned
    return output


def _trim_text(text: Any, max_chars: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _normalize_snippets(items: List[Any], max_items: int = 5) -> List[str]:
    output: List[str] = []
    for raw in items:
        line = _trim_text(raw, max_chars=180)
        if line and line not in output:
            output.append(line)
        if len(output) >= max_items:
            break
    return output


def _format_compact_number(value: Any) -> str:
    cleaned = _clean_scalar(value)
    if not isinstance(cleaned, float):
        return "-"
    abs_value = abs(cleaned)
    if abs_value >= 1_000_000_000:
        scaled = cleaned / 1_000_000_000.0
        return f"{scaled:.2f} mlr"
    if abs_value >= 1_000_000:
        scaled = cleaned / 1_000_000.0
        return f"{scaled:.2f} mn"
    if abs_value >= 1_000:
        scaled = cleaned / 1_000.0
        return f"{scaled:.2f} bin"
    return f"{cleaned:.2f}"


def _direction_text(direction: str) -> str:
    normalized = str(direction or "").strip().lower()
    if normalized == "artis":
        return "artis"
    if normalized == "azalis":
        return "azalis"
    return "yatay seyir"


def _find_low_quality(
    confidence_map: Dict[str, Any],
    verify_map: Dict[str, Any],
    threshold: float = 0.55,
) -> bool:
    for value in confidence_map.values():
        cleaned = _clean_scalar(value)
        if isinstance(cleaned, float) and cleaned < threshold:
            return True
    for value in verify_map.values():
        status = str(value or "").strip().upper()
        if status in {"LOW", "WARN", "FAIL"}:
            return True
    return False


def build_commentary_input(
    company: Any,
    period: Any,
    metrics: Dict[str, Any],
    ratios: Dict[str, Any],
    deltas: Dict[str, Any],
    confidence_map: Dict[str, Any],
    verify_map: Dict[str, Any],
    evidence_snippets: List[Any],
) -> Dict[str, Any]:
    cleaned_conf = _clean_map(confidence_map, max_items=16)
    cleaned_verify = _clean_map(verify_map, max_items=16)
    payload = {
        "company": str(company or "").strip(),
        "period": str(period or "").strip(),
        "metrics": _clean_map(metrics, max_items=12),
        "ratios": _clean_map(ratios, max_items=12),
        "deltas": _clean_map(deltas, max_items=12),
        "confidence_map": cleaned_conf,
        "verify_map": cleaned_verify,
        "evidence_snippets": _normalize_snippets(evidence_snippets, max_items=5),
    }
    payload["quality_flags"] = {
        "has_low_quality_signal": _find_low_quality(cleaned_conf, cleaned_verify),
    }
    return payload


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _optional_max_tokens(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    value = str(raw).strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _commentary_cfg(cfg: Any) -> Dict[str, Any]:
    container: Any = cfg
    if not isinstance(cfg, dict):
        container = getattr(cfg, "llm_assistant", None) or getattr(cfg, "llm_commentary", cfg)

    enabled = bool(_cfg_value(container, "enabled", False))
    provider = str(_cfg_value(container, "provider", "openrouter"))
    model = str(_cfg_value(container, "model", "arcee-ai/trinity-large-preview:free"))
    max_tokens = _optional_max_tokens(_cfg_value(container, "max_tokens", 220))
    timeout_s = float(_cfg_value(container, "timeout_s", 30.0))
    temperature = float(_cfg_value(container, "temperature", 0.2))
    reasoning_enabled = bool(_cfg_value(container, "reasoning_enabled", True))

    env_enabled = os.getenv("RAGFIN_LLM_ASSISTANT_ENABLED", "").strip() or os.getenv(
        "RAGFIN_LLM_COMMENTARY_ENABLED", ""
    ).strip()
    if env_enabled:
        enabled = env_enabled.lower() in {"1", "true", "yes", "on"}
    provider = (
        os.getenv("RAGFIN_LLM_ASSISTANT_PROVIDER", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_PROVIDER", "").strip()
        or provider
    )
    model = (
        os.getenv("RAGFIN_LLM_ASSISTANT_MODEL", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_MODEL", "").strip()
        or model
    )
    env_max_tokens = (
        os.getenv("RAGFIN_LLM_ASSISTANT_MAX_TOKENS", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_MAX_TOKENS", "").strip()
    )
    if env_max_tokens:
        max_tokens = _optional_max_tokens(env_max_tokens)
    env_timeout = (
        os.getenv("RAGFIN_LLM_ASSISTANT_TIMEOUT_S", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_TIMEOUT_S", "").strip()
    )
    if env_timeout:
        timeout_s = float(env_timeout)
    temperature = float(
        os.getenv(
            "RAGFIN_LLM_ASSISTANT_TEMPERATURE",
            os.getenv("RAGFIN_LLM_COMMENTARY_TEMPERATURE", str(temperature)),
        )
    )
    env_reasoning = (
        os.getenv("RAGFIN_LLM_ASSISTANT_REASONING_ENABLED", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_REASONING_ENABLED", "").strip()
    )
    if env_reasoning:
        reasoning_enabled = env_reasoning.lower() in {"1", "true", "yes", "on"}

    return {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "timeout_s": max(1.0, min(float(timeout_s), 120.0)),
        "temperature": max(0.0, min(float(temperature), 2.0)),
        "reasoning_enabled": reasoning_enabled,
    }


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        chunk = text[start : end + 1]
        try:
            payload = json.loads(chunk)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    return None


def _normalize_output(payload: Dict[str, Any], low_quality: bool) -> Dict[str, Any]:
    headline = str(payload.get("headline", "")).strip()[:90]
    bullet_items = payload.get("bullets", [])
    if not isinstance(bullet_items, list):
        bullet_items = []
    bullets: List[str] = []
    for item in bullet_items:
        line = str(item).strip()
        if line and line not in bullets:
            bullets.append(line[:140])
        if len(bullets) >= 5:
            break
    risk_note = str(payload.get("risk_note", "")).strip()[:220]
    next_question = str(payload.get("next_question", "")).strip()[:140]
    if low_quality and not risk_note:
        risk_note = "Bazi metriklerde guven/verify sinyali dusuk; kaniti kontrol edin."
    return {
        "headline": headline,
        "bullets": bullets,
        "risk_note": risk_note,
        "next_question": next_question,
    }


def _contains_forbidden_digits(value: Any) -> bool:
    text = str(value or "")
    # Allow quarter markers only (e.g. Q1/Q2/Q3/Q4, "3. ceyrek", "3 ceyrek").
    allowed_patterns = [
        r"\b[qQ][1-4]\b",
        r"\b[1-4]\.?\s*[cç]eyrek\b",
    ]
    sanitized = text
    for pattern in allowed_patterns:
        sanitized = re.sub(pattern, " ", sanitized, flags=re.IGNORECASE)
    return any(ch.isdigit() for ch in sanitized)


def _has_digit_in_output(payload: Dict[str, Any]) -> bool:
    if _contains_forbidden_digits(payload.get("headline", "")):
        return True
    if _contains_forbidden_digits(payload.get("risk_note", "")):
        return True
    if _contains_forbidden_digits(payload.get("next_question", "")):
        return True
    for item in payload.get("bullets", []) or []:
        if _contains_forbidden_digits(item):
            return True
    return False


def _contains_forbidden_placeholder_text(value: Any) -> bool:
    text = str(value or "").lower()
    patterns = [
        r"%",
        r"\byuzde\b",
        r"\bmilyar\s*tl\b",
        r"\bmn\s*tl\b",
        r"\bmlr\s*tl\b",
        r"\btl['’]?(ye|ya)?\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _has_placeholder_artifacts(payload: Dict[str, Any]) -> bool:
    if _contains_forbidden_placeholder_text(payload.get("headline", "")):
        return True
    if _contains_forbidden_placeholder_text(payload.get("risk_note", "")):
        return True
    if _contains_forbidden_placeholder_text(payload.get("next_question", "")):
        return True
    for item in payload.get("bullets", []) or []:
        if _contains_forbidden_placeholder_text(item):
            return True
    return False


def _empty_if_invalid(payload: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = {"headline", "bullets", "risk_note", "next_question"}
    if not isinstance(payload, dict):
        return dict(SAFE_EMPTY_COMMENTARY)
    if not required_keys.issubset(set(payload.keys())):
        return dict(SAFE_EMPTY_COMMENTARY)
    if not isinstance(payload.get("bullets"), list):
        return dict(SAFE_EMPTY_COMMENTARY)
    # Keep only the expected keys
    return {k: payload[k] for k in required_keys}


def _canonical_company_name(raw_company: Any) -> str:
    company = str(raw_company or "").strip().upper()
    if not company:
        return ""
    for canonical, aliases in KNOWN_COMPANY_ALIASES.items():
        if company == canonical or company in aliases:
            return canonical
    # Normalize suffix-heavy names, e.g. NETCAD_4Q -> NETCAD
    match = re.match(r"^([A-Z]+)", company)
    if match:
        return match.group(1)
    return company


def _contains_unexpected_company_mentions(payload: Dict[str, Any], expected_company: Any) -> bool:
    expected = _canonical_company_name(expected_company)
    allowed = set(KNOWN_COMPANY_ALIASES.get(expected, [expected])) if expected else set()

    text = " ".join(
        [
            str(payload.get("headline", "")),
            str(payload.get("risk_note", "")),
            str(payload.get("next_question", "")),
            " ".join(str(item) for item in (payload.get("bullets") or [])),
        ]
    ).upper()

    for canonical, aliases in KNOWN_COMPANY_ALIASES.items():
        for token in aliases:
            if not token:
                continue
            if not re.search(rf"\b{re.escape(token.upper())}\b", text):
                continue
            if expected and token.upper() in {alias.upper() for alias in allowed}:
                continue
            return True
    return False


def build_commentary_input_from_answer_payload(
    answer_payload: Dict[str, Any],
    *,
    question: str,
    company: Optional[str] = None,
    year: Optional[str] = None,
    quarter: Optional[str] = None,
) -> Dict[str, Any]:
    found = bool(answer_payload.get("found", (answer_payload.get("answer") or {}).get("found")))
    parsed = dict(answer_payload.get("parsed") or {})
    resolved_company = str(company or answer_payload.get("company") or parsed.get("company") or "").strip()
    resolved_period = str(quarter or parsed.get("quarter") or answer_payload.get("quarter") or "").strip()
    raw_metrics = dict(answer_payload.get("metrics") or answer_payload.get("kpis") or {})
    raw_ratios = dict(answer_payload.get("ratios") or {})
    raw_deltas = dict(answer_payload.get("deltas") or {})
    # LLM'e ham sayi vermiyoruz; sadece hangi metriklerin mevcut oldugunu ve yon bilgisini iletiyoruz.
    metrics = {str(key): _clean_scalar(value) for key, value in raw_metrics.items() if _clean_scalar(value) is not None}
    ratios = {str(key): _clean_scalar(value) for key, value in raw_ratios.items() if _clean_scalar(value) is not None}
    deltas: Dict[str, Any] = {}
    direction_signals: Dict[str, str] = {}
    qoq_signals: Dict[str, str] = {}
    yoy_signals: Dict[str, str] = {}
    for key, value in raw_deltas.items():
        cleaned = _clean_scalar(value)
        if not isinstance(cleaned, float):
            continue
        deltas[str(key)] = cleaned
        direction = "artis" if cleaned > 0 else ("azalis" if cleaned < 0 else "yatay")
        if str(key).endswith("_qoq"):
            base = str(key)[: -len("_qoq")]
            qoq_signals[base] = direction
            direction_signals[base] = direction  # backward compat: last wins
        elif str(key).endswith("_yoy"):
            base = str(key)[: -len("_yoy")]
            yoy_signals[base] = direction
            if base not in direction_signals:
                direction_signals[base] = direction
        else:
            metric_name = str(key).strip()
            direction_signals[metric_name] = direction
    confidence_map = dict(answer_payload.get("confidence_map") or {})
    verify_map = dict(answer_payload.get("verify_map") or {})
    answer_obj = dict(answer_payload.get("answer") or {})
    if "overall" not in verify_map and answer_obj.get("verify_status") is not None:
        verify_map["overall"] = answer_obj.get("verify_status")
    evidence_rows = list(answer_payload.get("evidence") or [])
    if not resolved_company:
        for row in evidence_rows:
            candidate_company = str((row or {}).get("company", "")).strip()
            if candidate_company:
                resolved_company = candidate_company
                break
    if not resolved_period:
        for row in evidence_rows:
            candidate_quarter = str((row or {}).get("quarter", "")).strip()
            if candidate_quarter:
                resolved_period = candidate_quarter
                break

    evidence_snippets = []
    for row in evidence_rows[:5]:
        excerpt = str((row or {}).get("excerpt", "")).strip()
        if excerpt:
            evidence_snippets.append(excerpt)
    payload = build_commentary_input(
        company=resolved_company,
        period=resolved_period,
        metrics=metrics,
        ratios=ratios,
        deltas=deltas,
        confidence_map=confidence_map,
        verify_map=verify_map,
        evidence_snippets=evidence_snippets,
    )
    payload["question"] = _trim_text(question, max_chars=180)
    if year:
        payload["year"] = str(year)
    payload["found"] = found
    payload["direction_signals"] = direction_signals
    payload["qoq_signals"] = qoq_signals
    payload["yoy_signals"] = yoy_signals
    payload["commentary_mode"] = str(
        answer_payload.get("commentary_mode") or answer_payload.get("source") or ""
    ).strip().lower()
    # Forward KAP-specific enrichments if present
    for extra_key in ("qoq_pct", "yoy_pct", "fcf_signal", "prev_quarter_label", "prev_year_label"):
        if extra_key in answer_payload:
            payload[extra_key] = answer_payload[extra_key]
    return payload


def _rule_based_commentary(
    commentary_input: Dict[str, Any],
    *,
    low_quality: bool,
    allow_numbers: bool,
) -> Dict[str, Any]:
    company = str(commentary_input.get("company", "")).strip().upper()
    period = str(commentary_input.get("period", "")).strip()
    direction_signals = dict(commentary_input.get("direction_signals") or {})
    qoq_signals = dict(commentary_input.get("qoq_signals") or {})
    yoy_signals = dict(commentary_input.get("yoy_signals") or {})
    metrics = dict(commentary_input.get("metrics") or {})
    evidence_snippets = list(commentary_input.get("evidence_snippets") or [])
    commentary_mode = str(commentary_input.get("commentary_mode", "")).strip().lower()
    is_kap = commentary_mode in {"kap", "kap_financials"}
    qoq_pct = dict(commentary_input.get("qoq_pct") or {})
    yoy_pct = dict(commentary_input.get("yoy_pct") or {})
    fcf_signal = dict(commentary_input.get("fcf_signal") or {})

    preferred_metrics = [
        "net_kar",
        "satis_gelirleri",
        "favok",
        "serbest_nakit_akisi",
        "brut_kar_marji",
        "favok_marji",
        "net_kar_marji",
        "cari_oran",
        "ozkaynak_karliligi",
    ]

    bullets: List[str] = []

    if is_kap and (qoq_signals or yoy_signals):
        # KAP mode: structured QoQ / YoY bullets
        for metric_key in preferred_metrics:
            if len(bullets) >= 4:
                break
            metric_label = METRIC_LABELS.get(metric_key, metric_key.replace("_", " "))
            parts: List[str] = []
            qoq_dir = qoq_signals.get(metric_key)
            yoy_dir = yoy_signals.get(metric_key)
            if qoq_dir:
                pct_str = ""
                if allow_numbers and metric_key in qoq_pct:
                    pct_str = f" (%{qoq_pct[metric_key]:+.1f})"
                parts.append(f"QoQ {_direction_text(qoq_dir)}{pct_str}")
            if yoy_dir:
                pct_str = ""
                if allow_numbers and metric_key in yoy_pct:
                    pct_str = f" (%{yoy_pct[metric_key]:+.1f})"
                parts.append(f"YoY {_direction_text(yoy_dir)}{pct_str}")
            if not parts:
                continue
            value_part = ""
            if allow_numbers and metric_key in metrics:
                value_part = f": {_format_compact_number(metrics.get(metric_key))}; "
            else:
                value_part = ": "
            bullets.append(f"{metric_label}{value_part}{', '.join(parts)}.")

        # FCF special bullet
        if fcf_signal and not any("nakit" in b.lower() for b in bullets):
            fcf_parts: List[str] = []
            sign = str(fcf_signal.get("sign", "")).strip()
            if sign:
                fcf_parts.append(f"Ceyreklik FCF {sign}")
            annual_sign = str(fcf_signal.get("annual_sign", "")).strip()
            if annual_sign:
                q_count = int(fcf_signal.get("quarters_in_sum", 0))
                if allow_numbers and "annual_sum" in fcf_signal:
                    fcf_parts.append(f"yillik toplam ({q_count}Q): {_format_compact_number(fcf_signal['annual_sum'])} ({annual_sign})")
                else:
                    fcf_parts.append(f"yillik toplam ({q_count}Q) {annual_sign}")
            if bool(fcf_signal.get("is_year_end")):
                fcf_parts.append("tam yil kapanisi")
            if fcf_parts:
                bullets.append(f"Serbest nakit akisi: {'; '.join(fcf_parts)}.")
    else:
        # Non-KAP generic path
        for metric_key in preferred_metrics:
            direction = direction_signals.get(metric_key)
            if direction is None:
                continue
            metric_label = METRIC_LABELS.get(metric_key, metric_key.replace("_", " "))
            if allow_numbers and metric_key in metrics:
                value_label = _format_compact_number(metrics.get(metric_key))
                bullets.append(f"{metric_label}: {value_label}; yon {_direction_text(direction)}.")
            else:
                bullets.append(f"{metric_label} tarafinda {_direction_text(direction)} goruluyor.")
            if len(bullets) >= 5:
                break

    # FCF fallback for KAP mode if no bullets yet captured it
    if is_kap and fcf_signal and not any("nakit" in b.lower() for b in bullets):
        sign = str(fcf_signal.get("sign", "")).strip()
        if sign:
            bullets.append(f"Serbest nakit akisi bu ceyrek {sign} bolgededir.")

    if not bullets:
        for snippet in evidence_snippets[:3]:
            line = _trim_text(snippet, max_chars=120)
            if line:
                bullets.append(line)

    if not bullets:
        bullets = ["Bu donemde metriklerin yonu sinirli veriyle izleniyor."]

    headline_parts = []
    if company:
        headline_parts.append(company)
    if period:
        headline_parts.append(period)
    headline_prefix = " ".join(headline_parts).strip()
    if headline_prefix:
        headline = f"{headline_prefix} finansal ozeti"
    else:
        headline = "Finansal ozet"

    risk_note = ""
    if low_quality:
        risk_note = "Bazi metriklerde guven/verify sinyali dusuk; kanitlari kontrol edin."

    next_question = "Bu egilimlerin bir sonraki ceyrekte devami beklenir mi?"
    return {
        "headline": headline[:90],
        "bullets": [item[:140] for item in bullets[:5]],
        "risk_note": risk_note[:220],
        "next_question": next_question[:140],
    }


def generate_commentary(
    answer_payload: Dict[str, Any],
    question: Any = "",
    cfg: Any = None,
    *,
    company: Optional[str] = None,
    year: Optional[str] = None,
    quarter: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    # Backward compatibility: generate_commentary(payload, cfg)
    if cfg is None and not isinstance(question, str):
        cfg = question
        question = ""
    question_text = str(question or "")

    if not isinstance(answer_payload, dict):
        return dict(SAFE_EMPTY_COMMENTARY)
    found = bool(answer_payload.get("found", (answer_payload.get("answer") or {}).get("found")))
    if not found:
        return dict(SAFE_EMPTY_COMMENTARY)

    conf = _commentary_cfg(cfg)
    override_model = str(model_override or "").strip()
    if override_model:
        conf["model"] = override_model
    if not conf["enabled"] or str(conf["provider"]).lower() != "openrouter":
        return dict(SAFE_EMPTY_COMMENTARY)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return dict(SAFE_EMPTY_COMMENTARY)

    try:
        commentary_input = build_commentary_input_from_answer_payload(
            answer_payload,
            question=question_text,
            company=company,
            year=year,
            quarter=quarter,
        )
    except Exception:
        commentary_input = {
            "company": str(company or "").strip(),
            "period": str(quarter or "").strip(),
            "metrics": {},
            "ratios": {},
            "deltas": {},
            "confidence_map": {},
            "verify_map": {},
            "evidence_snippets": [],
            "quality_flags": {},
            "direction_signals": {},
            "commentary_mode": str(answer_payload.get("commentary_mode", "")).strip().lower(),
        }
    low_quality = bool(((commentary_input.get("quality_flags") or {}).get("has_low_quality_signal")))
    commentary_mode = str(commentary_input.get("commentary_mode", "")).strip().lower()
    allow_numbers = commentary_mode in {"kap", "kap_financials"}
    strict_no_numbers = not allow_numbers

    number_rule = (
        "- Do not output numbers, percentages, or currency amounts.\n"
        if strict_no_numbers
        else "- Numbers are allowed ONLY if they already exist in answer_payload.\n"
    )
    if commentary_mode in {"kap", "kap_financials"}:
        system_prompt = (
            "You are a KAP financial commentary layer for a Turkish stock market assistant.\n"
            "You receive official KAP quarterly financial data that has already been calculated.\n"
            "STRICT RULES:\n"
            "- NEVER invent, fabricate, or hallucinate any number. All numbers come from answer_payload.\n"
            "- Numbers from answer_payload (metrics, ratios, deltas, qoq_pct, yoy_pct) may be quoted verbatim.\n"
            "- Your job is to write a SHORT Turkish commentary covering exactly these topics:\n"
            "  1) Son ceyrek ozeti: headline metric movements.\n"
            "  2) QoQ degisim: Use qoq_signals / qoq_pct to describe quarter-over-quarter changes.\n"
            "  3) YoY degisim: Use yoy_signals / yoy_pct to describe year-over-year changes.\n"
            "  4) Serbest nakit akisi (FCF) yorumu: Use fcf_signal to comment on FCF sign (pozitif/negatif),\n"
            "     and if annual_sum exists, mention the year-to-date or full-year FCF context.\n"
            "     If is_year_end is true, explicitly note this is the full-year FCF figure.\n"
            "- If company is provided, never mention another company name.\n"
            "- Do not contradict evidence. Keep it short and data-driven.\n"
            "- Write in Turkish.\n"
            "- Return only JSON with exactly these keys:\n"
            "{\"headline\":\"\", \"bullets\":[], \"risk_note\":\"\", \"next_question\":\"\"}"
        )
    else:
        system_prompt = (
            "You are a commentary layer for a financial assistant.\n"
            "STRICT RULES:\n"
            "- Only interpret provided answer_payload.\n"
            "- Never invent or change any number.\n"
            f"{number_rule}"
            "- Focus on direction (artis/azalis/yatay) and reliability signals.\n"
            "- If company is provided, never mention another company name.\n"
            "- Use metric names from direction_signals and verify_map.\n"
            "- Do not contradict evidence.\n"
            "- Keep it short and data-driven.\n"
            "- Write in Turkish.\n"
            "- Return only JSON with exactly these keys:\n"
            "{\"headline\":\"\", \"bullets\":[], \"risk_note\":\"\", \"next_question\":\"\"}"
        )
    try:
        user_prompt = (
            "Question:\n"
            f"{question_text}\n\n"
            "answer_payload (ground truth):\n"
            f"{json.dumps(commentary_input, ensure_ascii=False)}\n\n"
            "Return short commentary JSON."
        )
    except Exception:
        user_prompt = (
            "Question:\n"
            f"{question_text}\n\n"
            "Return short commentary JSON."
        )

    try:
        client = OpenRouterClient(api_key=api_key, timeout_sec=float(conf["timeout_s"]))
        kwargs: Dict[str, Any] = {
            "model": conf["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": conf.get("max_tokens"),
            "temperature": float(conf["temperature"]),
        }
        raw: Optional[str] = None
        # Not all models support response_format or reasoning; try with, fallback without
        try:
            raw = client.chat_completion(
                **kwargs,
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": bool(conf.get("reasoning_enabled", True))}},
            )
        except Exception:
            # Retry without response_format and reasoning for broader model compatibility
            try:
                raw = client.chat_completion(**kwargs)
            except Exception:
                return _rule_based_commentary(
                    commentary_input,
                    low_quality=low_quality,
                    allow_numbers=allow_numbers,
                )
        if raw is None:
            return _rule_based_commentary(
                commentary_input,
                low_quality=low_quality,
                allow_numbers=allow_numbers,
            )
        parsed = _extract_json_object(raw)
        if not parsed:
            return _rule_based_commentary(
                commentary_input,
                low_quality=low_quality,
                allow_numbers=allow_numbers,
            )
        normalized = _normalize_output(parsed, low_quality=low_quality)
        result = _empty_if_invalid(normalized)
        if strict_no_numbers and _has_digit_in_output(result):
            return _rule_based_commentary(
                commentary_input,
                low_quality=low_quality,
                allow_numbers=allow_numbers,
            )
        if strict_no_numbers and _has_placeholder_artifacts(result):
            return _rule_based_commentary(
                commentary_input,
                low_quality=low_quality,
                allow_numbers=allow_numbers,
            )
        if _contains_unexpected_company_mentions(result, commentary_input.get("company")):
            return _rule_based_commentary(
                commentary_input,
                low_quality=low_quality,
                allow_numbers=allow_numbers,
            )
        if not any(result.values()):
            return _rule_based_commentary(
                commentary_input,
                low_quality=low_quality,
                allow_numbers=allow_numbers,
            )
        return result
    except Exception:
        return _rule_based_commentary(
            commentary_input,
            low_quality=low_quality,
            allow_numbers=allow_numbers,
        )
