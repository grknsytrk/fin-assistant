from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence


TL_UNIT_HINTS = (" tl", "milyon", "milyar", "mn", "mlr")

ALTERNATE_PATTERNS = {
    "net_kar": re.compile(
        r"(?:net\s+kar|net\s+donem\s+kar[ıi]|net\s+donem\s+zarar[ıi])[^\d\-\(]{0,40}(\(?[\-]?\d[\d\.,]*\)?)",
        re.IGNORECASE,
    ),
    "favok": re.compile(
        r"(?:fvaok|favok|ebitda)[^\d\-\(]{0,40}(\(?[\-]?\d[\d\.,]*\)?)",
        re.IGNORECASE,
    ),
    "satis_gelirleri": re.compile(
        r"(?:net\s+sat[ıi]slar|sat[ıi]s\s+gelir(?:leri)?|ciro|has[ıi]lat)[^\d\-\(]{0,40}(\(?[\-]?\d[\d\.,]*\)?)",
        re.IGNORECASE,
    ),
    "magaza_sayisi": re.compile(
        r"(?:toplam\s+magaza|magaza\s+say[ıi]s[ıi]|sube\s+say[ıi]s[ıi])[^\d\-\(]{0,30}(\(?[\-]?\d[\d\.,]*\)?)",
        re.IGNORECASE,
    ),
    "net_kar_marji": re.compile(
        r"(?:net\s+kar\s+marj[ıi]|net\s+marj)[^%\d\-]{0,30}%?\s*(\(?[\-]?\d[\d\.,]*\)?)",
        re.IGNORECASE,
    ),
    "favok_marji": re.compile(
        r"(?:fvaok|favok|ebitda)\s+marj[ıi][^%\d\-]{0,30}%?\s*(\(?[\-]?\d[\d\.,]*\)?)",
        re.IGNORECASE,
    ),
    "brut_kar_marji": re.compile(
        r"(?:brut|brüt)\s+kar\s+marj[ıi][^%\d\-]{0,30}%?\s*(\(?[\-]?\d[\d\.,]*\)?)",
        re.IGNORECASE,
    ),
}


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _parse_tr_number(raw: str) -> Optional[float]:
    payload = str(raw).strip().replace(" ", "")
    if not payload:
        return None

    negative_by_parenthesis = payload.startswith("(") and payload.endswith(")")
    if negative_by_parenthesis:
        payload = payload[1:-1]

    sign = -1.0 if payload.startswith("-") else 1.0
    if negative_by_parenthesis:
        sign = -1.0

    payload = payload.lstrip("+-")
    if not payload:
        return None

    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", payload):
        normalized = payload.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", payload):
        normalized = payload.replace(",", "")
    elif payload.count(",") == 1 and payload.count(".") == 0:
        left, right = payload.split(",", 1)
        normalized = f"{left}.{right}" if len(right) <= 2 else f"{left}{right}"
    elif payload.count(".") > 1 and payload.count(",") == 0:
        normalized = payload.replace(".", "")
    elif "," in payload and "." in payload:
        if payload.rfind(",") > payload.rfind("."):
            normalized = payload.replace(".", "").replace(",", ".")
        else:
            normalized = payload.replace(",", "")
    else:
        normalized = payload.replace(",", ".")

    try:
        return sign * float(normalized)
    except ValueError:
        return None


def _tolerance(unit: str, baseline: float) -> float:
    if unit == "%":
        return 0.35
    return max(1.0, abs(baseline) * 0.02)


def _is_close(lhs: float, rhs: float, unit: str) -> bool:
    return abs(lhs - rhs) <= _tolerance(unit, rhs)


def _has_unit_ambiguity(selected: Dict[str, Any]) -> bool:
    unit = str(selected.get("unit", ""))
    if unit != "TL":
        return False
    multiplier = float(selected.get("multiplier", 1.0) or 1.0)
    excerpt = _normalize(str(selected.get("excerpt", "")))
    if multiplier != 1.0:
        return False
    return not any(hint in excerpt for hint in TL_UNIT_HINTS)


def _alternate_value(metric: str, excerpt: str, multiplier: float, unit: str) -> Optional[float]:
    pattern = ALTERNATE_PATTERNS.get(metric)
    if not pattern:
        return None
    match = pattern.search(excerpt)
    if not match:
        return None
    parsed = _parse_tr_number(match.group(1))
    if parsed is None:
        return None
    if unit == "TL":
        return float(parsed) * float(multiplier)
    return float(parsed)


def auto_verify_metric(
    metric: str,
    selected: Optional[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    quarter: Optional[str] = None,
) -> Dict[str, Any]:
    if not selected:
        return {
            "status": "FAIL",
            "checks": [],
            "warnings": ["candidate_yok"],
            "reasons": ["dogrulama_icin_secili_aday_yok"],
            "alternate_value": None,
        }

    warnings = []
    checks = []
    selected_value = float(selected.get("value", 0.0))
    unit = str(selected.get("unit", ""))
    excerpt = str(selected.get("excerpt", ""))
    multiplier = float(selected.get("multiplier", 1.0) or 1.0)

    if _has_unit_ambiguity(selected):
        warnings.append("unit_ambiguity")

    excerpt_norm = _normalize(excerpt)
    years = set(re.findall(r"20\d{2}", excerpt_norm))
    reasons = list(selected.get("reasons", []))
    if len(years) >= 2 and not any("year_" in item for item in reasons):
        warnings.append("year_mismatch_suspected")

    valid_candidates = [item for item in candidates if bool(item.get("validation_ok"))]
    valid_candidates = sorted(valid_candidates, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    if len(valid_candidates) >= 2:
        top_score = float(valid_candidates[0].get("score", 0.0))
        close_band = [
            item for item in valid_candidates[:3] if (top_score - float(item.get("score", 0.0))) <= 6.0
        ]
        if len(close_band) >= 2:
            values = [float(item.get("value", 0.0)) for item in close_band]
            spread = max(values) - min(values)
            if unit == "%" and spread > 2.5:
                warnings.append("multiple_strong_candidates_disagree")
            elif unit != "%" and min(abs(v) for v in values) > 0:
                ratio = max(abs(v) for v in values) / min(abs(v) for v in values)
                if ratio > 1.35:
                    warnings.append("multiple_strong_candidates_disagree")
            else:
                checks.append("candidate_consistency")
        else:
            checks.append("candidate_consistency")
    elif len(valid_candidates) == 1:
        checks.append("single_valid_candidate")

    alternate = _alternate_value(
        metric=metric,
        excerpt=excerpt,
        multiplier=multiplier,
        unit=unit,
    )
    if alternate is None:
        warnings.append("alternate_regex_not_found")
    elif _is_close(alternate, selected_value, unit):
        checks.append("alternate_regex_match")
    else:
        warnings.append("alternate_regex_disagree")

    if quarter:
        selected_quarter = str(selected.get("quarter", "")).upper()
        expected_quarter = str(quarter).upper()
        if selected_quarter and expected_quarter and selected_quarter != expected_quarter:
            warnings.append("quarter_mismatch_suspected")

    if not warnings and checks:
        status = "PASS"
    elif warnings and checks:
        status = "WARN"
    elif warnings:
        status = "WARN"
    else:
        status = "FAIL"

    reason_text = []
    if checks:
        reason_text.append("checks:" + ",".join(checks))
    if warnings:
        reason_text.append("warnings:" + ",".join(warnings))

    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "reasons": reason_text,
        "alternate_value": alternate,
    }
