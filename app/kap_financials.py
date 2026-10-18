from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

def _quarter_label_sort_key(label: str) -> tuple[int, int, str]:
    normalized = str(label or "").strip().upper()
    if not normalized:
        return (0, 0, "")
    quarter_match = re.match(r"^(\d{4})Q([1-4])$", normalized)
    if quarter_match:
        return (int(quarter_match.group(1)), int(quarter_match.group(2)) * 3, normalized)
    period_match = re.match(r"^(\d{4})[/-](\d{1,2})$", normalized)
    if period_match:
        return (int(period_match.group(1)), int(period_match.group(2)), normalized)
    return (0, 0, normalized)

def _latest_quarter_label(quarters: List[str]) -> Optional[str]:
    candidates = [str(item or "").strip().upper() for item in quarters if str(item or "").strip()]
    if not candidates:
        return None
    return max(candidates, key=_quarter_label_sort_key)

def _quarter_sort_key(quarter: Dict[str, Any]) -> tuple[int, int]:
    return int(quarter.get("year") or 0), int(quarter.get("period") or 0)

def _extract_quarter_metric(
    quarter: Dict[str, Any],
    metric_key: str,
    priority: List[str],
) -> Optional[float]:
    for field in priority:
        container = quarter.get(field)
        if not isinstance(container, dict):
            continue
        metric = container.get(metric_key)
        if isinstance(metric, dict):
            value = metric.get("value")
        else:
            value = metric
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None

def _build_ttm_sum(quarters_asc: List[Dict[str, Any]], metric_key: str) -> Optional[float]:
    if len(quarters_asc) < 4:
        return None
    tail = quarters_asc[-4:]
    periods = []
    values = []
    for quarter in tail:
        try:
            year, period = int(quarter.get("year") or 0), int(quarter.get("period") or 0)
        except (ValueError, TypeError):
            return None
        if year <= 0 or period not in (1, 2, 3, 4):
            return None
        periods.append(year * 4 + period)
        # Only standalone quarterly flows are additive. YTD/ambiguous metrics
        # must not silently replace a missing quarterly value.
        value = _extract_quarter_metric(quarter, metric_key, priority=["metrics_quarterly"])
        if value is None or not math.isfinite(value):
            return None
        values.append(value)
    if any(b != a + 1 for a, b in zip(periods, periods[1:])):
        return None
    return float(sum(values))

def _build_ttm_from_reports(quarters: List[Dict[str, Any]], metric_key: str) -> Optional[float]:
    """Build TTM from KAP YTD values, keeping the reporting period's price basis.

    KAP's Q4 income statement is annual YTD, not a standalone quarter. For Q1-Q3,
    prior-year annual flow values must first be normalized to the latest report's
    price basis; ``normalize_snapshot_for_frontend`` does that. The current and
    comparative YTD values come from the latest report and share its basis.
    """
    ordered = sorted(
        (quarter for quarter in quarters if isinstance(quarter, dict)),
        key=_quarter_sort_key,
    )
    if not ordered:
        return None

    latest = ordered[-1]
    try:
        year = int(latest.get("year") or 0)
        period = int(latest.get("period") or 0)
    except (TypeError, ValueError):
        return None
    if year <= 0 or period not in (1, 2, 3, 4):
        return None

    if period == 4:
        annual = _extract_quarter_metric(latest, metric_key, priority=["metrics_ytd", "metrics"])
        return annual if annual is not None and math.isfinite(annual) else None

    prior_year_end = next(
        (
            quarter
            for quarter in reversed(ordered[:-1])
            if int(quarter.get("year") or 0) == year - 1
            and int(quarter.get("period") or 0) == 4
        ),
        None,
    )
    if prior_year_end is None:
        return None

    prior_fy = _extract_quarter_metric(
        prior_year_end,
        metric_key,
        priority=["metrics_ytd", "metrics"],
    )
    current_ytd = _extract_quarter_metric(latest, metric_key, priority=["metrics_ytd"])
    comparative_ytd = _extract_quarter_metric(
        latest,
        metric_key,
        priority=["metrics_ytd_comparative"],
    )
    if any(value is None or not math.isfinite(value) for value in (prior_fy, current_ytd, comparative_ytd)):
        return None
    return float(prior_fy + current_ytd - comparative_ytd)

__all__ = [name for name in globals() if not name.startswith("__")]
