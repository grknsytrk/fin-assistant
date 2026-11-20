"""Fund listing, detail, performance history, and asset allocation payloads."""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from app.fund_service import FundSnapshotFetchResult

_UNSET: Any = object()
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


def _normalize_allocation_row(row: Dict[str, Any], fallback_code: str | None = None) -> List[Dict[str, Any]]:
    fund_code = _fund_service().normalize_fund_code(_fund_service()._first_text(row, "fonKodu", "fonKod", "FONKODU", "fund_code") or fallback_code)
    if not fund_code:
        return []
    report_date = _fund_service()._fund_date(_fund_service()._first_present(row, "tarih", "TARIH", "date"))
    source = _fund_service()._normalize_price_source(_fund_service()._first_text(row, "source", "SOURCE") or _fund_service().TEFASFON_PORTFOLIO_SOURCE)
    allocations: List[Dict[str, Any]] = []
    for key in _fund_service().FUND_ALLOCATION_KEYS:
        weight = _fund_service()._coerce_float(row.get(key))
        if weight is None or weight == 0:
            continue
        allocations.append(
            {
                "fund_code": fund_code,
                "allocation_type": key,
                "label": _fund_service().FUND_ALLOCATION_LABELS.get(key, key.upper()),
                "weight": weight,
                "report_date": report_date,
                "source": _fund_service()._public_price_source(source),
            }
        )
    allocations.sort(key=lambda item: abs(float(item.get("weight") or 0)), reverse=True)
    return allocations


def _allocation_payload_report_date(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the newest report date represented by an allocation payload."""

    if not isinstance(payload, dict):
        return None
    dates = [
        report_date
        for item in list(payload.get("allocations") or [])
        for report_date in [_fund_service()._fund_date(item.get("report_date"))]
        if report_date
    ]
    meta = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    dates.extend(
        report_date
        for report_date in [_fund_service()._fund_date(meta.get("as_of")), _fund_service()._fund_date(payload.get("as_of"))]
        if report_date
    )
    return max(dates) if dates else None


def _promote_latest_allocation_history(
    processed_dir: Path,
    fund_code: str,
    history_payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Make the newest fresh history day the canonical allocation snapshot.

    The overview card and the allocation-history screen used to have separate
    caches. A history refresh could therefore contain newer TEFAS data while
    the overview still displayed an older single-day snapshot. Promoting the
    newest history day keeps both views on the same report date without making
    another upstream request.
    """

    if not isinstance(history_payload, dict) or history_payload.get("stale"):
        return None
    normalized = _fund_service().normalize_fund_code(fund_code)
    history_days = [
        day
        for day in list(history_payload.get("history") or [])
        if isinstance(day, dict) and _fund_service()._fund_date(day.get("date")) and list(day.get("allocations") or [])
    ]
    if not history_days:
        return None
    latest_day = max(history_days, key=lambda day: _fund_service()._fund_date(day.get("date")) or "")
    latest_date = _fund_service()._fund_date(latest_day.get("date"))
    if not latest_date:
        return None
    latest_allocations = [
        dict(item)
        for item in list(latest_day.get("allocations") or [])
        if isinstance(item, dict)
    ]
    if not latest_allocations:
        return None

    current = _fund_service()._read_json(_fund_service()._allocations_path(processed_dir, normalized))
    current_date = _seam("_allocation_payload_report_date")(current)
    if current_date and current_date >= latest_date:
        return None

    history_meta = history_payload.get("source_metadata") if isinstance(history_payload.get("source_metadata"), dict) else {}
    fetched_at = history_meta.get("fetched_at") or _fund_service()._utc_now_iso()
    metadata = dict(history_meta)
    metadata.update(
        {
            "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
            "source_url": _fund_service().TEFASFON_SOURCE_URL,
            "fetched_at": fetched_at,
            "as_of": latest_date,
            "cache_hit": True,
            "stale": False,
            "parse_status": "ok_tefasfon_portfolio_history_latest",
            "source_policy": "tefasfon_primary",
            "promoted_from": "allocation_history",
            "history_lookback_days": history_payload.get("lookback_days"),
            "warnings": list(history_meta.get("warnings") or []),
        }
    )
    promoted = {
        "fund_code": normalized,
        "status": "ok",
        "allocations": latest_allocations,
        "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
        "stale": False,
        "source_metadata": metadata,
    }
    _fund_service()._write_json(_fund_service()._allocations_path(processed_dir, normalized), promoted)
    return promoted


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
        base = _seam("_point_on_or_before")(points, target)
        returns[key] = _seam("_return_between")(latest_price, base.get("price") if base else None)
    return returns


def _period_returns_from_raw(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "1w": _fund_service()._coerce_float(_fund_service()._first_present(raw, "getiri1h", "return_1w")),
        "1m": _fund_service()._coerce_float(_fund_service()._first_present(raw, "getiri1a", "return_1m")),
        "3m": _fund_service()._coerce_float(_fund_service()._first_present(raw, "getiri3a", "return_3m")),
        "6m": _fund_service()._coerce_float(_fund_service()._first_present(raw, "getiri6a", "return_6m")),
        "ytd": _fund_service()._coerce_float(_fund_service()._first_present(raw, "getiriyb", "return_ytd")),
        "1y": _fund_service()._coerce_float(_fund_service()._first_present(raw, "getiri1y", "return_1y")),
    }


def _history_coverage_info(points: List[Dict[str, Any]], end_date: date, threshold_days: int = 3) -> Dict[str, Any]:
    if not points:
        return {
            "latest_point_date": None,
            "coverage_gap_days": None,
            "coverage_gap_business_days": None,
            "warnings": [],
        }
    latest_point = points[-1]
    latest_point_date = _fund_service()._fund_date(latest_point.get("date"))
    if not latest_point_date:
        return {
            "latest_point_date": None,
            "coverage_gap_days": None,
            "coverage_gap_business_days": None,
            "warnings": [],
        }
    latest = date.fromisoformat(latest_point_date)
    gap_days = max(0, (end_date - latest).days)
    gap_business_days = _fund_service()._business_days_between(latest + timedelta(days=1), end_date) if gap_days > 0 else 0
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
    resolution: Optional[str] = None,
) -> List[str]:
    parsed_dates = sorted(
        {
            date.fromisoformat(point_date)
            for point in points
            for point_date in [_fund_service()._fund_date(point.get("date"))]
            if point_date
        }
    )
    resolved_resolution = resolution or _seam("_history_resolution")(points)
    if resolved_resolution == "monthly_anchor":
        return []

    warnings: List[str] = []
    for previous, current in zip(parsed_dates, parsed_dates[1:]):
        business_gap = _fund_service()._business_days_between(previous + timedelta(days=1), current - timedelta(days=1))
        if business_gap > threshold_business_days:
            if resolved_resolution == "mixed" and _seam("_is_monthly_anchor_transition")(previous, current):
                continue
            warnings.append(
                f"Fund history has an internal gap: previous={previous.isoformat()}, "
                f"next={current.isoformat()}, missing_business_days={business_gap}."
            )
    return warnings


def _is_monthly_anchor_transition(previous: date, current: date) -> bool:
    """Return whether a long gap looks like an intentional month-end sample.

    The long-range fallback keeps month-end anchors and recent daily points.
    Month-end-to-month-end gaps are therefore a resolution characteristic,
    not evidence that the fund lost daily records.
    """

    if previous.day < 25:
        return False
    return current.day >= 25 or current.day <= 7


def _history_resolution(points: List[Dict[str, Any]]) -> str:
    """Classify the stored series without treating sampling as data loss."""

    parsed_dates = sorted(
        {
            date.fromisoformat(point_date)
            for point in points
            for point_date in [_fund_service()._fund_date(point.get("date"))]
            if point_date
        }
    )
    if len(parsed_dates) < 2:
        return "unknown"

    large_gaps = []
    monthly_anchor_gaps = []
    for previous, current in zip(parsed_dates, parsed_dates[1:]):
        business_gap = _fund_service()._business_days_between(previous + timedelta(days=1), current - timedelta(days=1))
        if business_gap <= 3:
            continue
        large_gaps.append((previous, current))
        if _seam("_is_monthly_anchor_transition")(previous, current):
            monthly_anchor_gaps.append((previous, current))

    if large_gaps and len(monthly_anchor_gaps) == len(large_gaps):
        return "monthly_anchor"
    if monthly_anchor_gaps:
        return "mixed"
    return "daily"


def _history_is_usable_daily_range(
    points: List[Dict[str, Any]],
    *,
    start_date: date,
    end_date: date,
) -> bool:
    """Return whether a source covers the requested range as a daily series."""

    parsed_dates = sorted(
        {
            date.fromisoformat(point_date)
            for point in points
            for point_date in [_fund_service()._fund_date(point.get("date"))]
            if point_date
        }
    )
    if len(parsed_dates) < 2:
        return False
    if parsed_dates[0] > start_date and _fund_service()._business_days_between(
        start_date,
        parsed_dates[0] - timedelta(days=1),
    ) > 3:
        return False
    if parsed_dates[-1] < end_date and _fund_service()._business_days_between(
        parsed_dates[-1] + timedelta(days=1),
        end_date,
    ) > 3:
        return False
    return not _seam("_history_internal_gap_warnings")(points, resolution="daily")


def _fetch_confirmed_tefas_tail(
    fund_code: str,
    fintables_points: Iterable[Dict[str, Any]],
    *,
    end_date: date,
    max_business_gap: int = 3,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Fetch only a short, confirmed Fintables tail gap from TEFAS.

    A recent Fintables close can legitimately arrive after the requested
    range is read.  Treat the gap as missing data only when TEFAS already
    publishes a valid point for one of those dates.  Empty TEFAS responses
    therefore preserve the normal publication-delay behavior instead of
    manufacturing a warning or repeatedly replacing the Fintables series.
    """

    normalized = _fund_service().normalize_fund_code(fund_code)
    valid_fintables_points = _seam("_valid_performance_points")(fintables_points, normalized)
    if not normalized or not valid_fintables_points:
        return [], []

    latest_fintables_date = date.fromisoformat(valid_fintables_points[-1]["date"])
    if latest_fintables_date >= end_date:
        return [], []

    tail_start = latest_fintables_date + timedelta(days=1)
    missing_business_days = _fund_service()._business_days_between(tail_start, end_date)
    if missing_business_days <= 0 or missing_business_days > max(0, int(max_business_gap)):
        return [], []

    try:
        tefas_rows = _fund_service().TefasClient().fetch_fund_history(
            fund_codes=[normalized],
            start_date=tail_start,
            end_date=end_date,
        )
    except _fund_service().TefasUpstreamError as exc:
        return [], [f"tefas confirmed tail probe failed: {exc}"]

    valid_tefas_rows = _seam("_valid_performance_points")(tefas_rows, normalized)
    confirmed_tail = [
        row
        for row in valid_tefas_rows
        if latest_fintables_date < date.fromisoformat(row["date"]) <= end_date
    ]
    return confirmed_tail, []


def _history_needs_detail_fill(
    points: List[Dict[str, Any]],
    start_date: date,
    end_date: date,
    *,
    resolution: Optional[str] = None,
) -> bool:
    if not points:
        return True
    parsed_dates = [
        date.fromisoformat(point_date)
        for point in points
        for point_date in [_fund_service()._fund_date(point.get("date"))]
        if point_date
    ]
    if not parsed_dates:
        return True
    if min(parsed_dates) > start_date and _fund_service()._business_days_between(start_date, min(parsed_dates) - timedelta(days=1)) > 3:
        return True
    resolved_resolution = resolution or _seam("_history_resolution")(points)
    if resolved_resolution == "monthly_anchor":
        return False
    coverage = _seam("_history_coverage_info")(points, end_date)
    if coverage.get("coverage_gap_business_days") and int(coverage["coverage_gap_business_days"]) > 3:
        return True
    return bool(_seam("_history_internal_gap_warnings")(points, resolution=resolved_resolution))


def _dedupe_price_points(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        fund_code = _fund_service().normalize_fund_code(str(point.get("fund_code") or ""))
        point_date = _fund_service()._fund_date(point.get("date"))
        price = _fund_service()._coerce_float(point.get("price"))
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
        existing_rank = _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing.get("source") or ""), 0)
        row_rank = _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(normalized_point.get("source") or ""), 0)
        if row_rank >= existing_rank:
            deduped[key] = normalized_point
    return [deduped[key] for key in sorted(deduped)]


def _dominant_price_source(points: Iterable[Dict[str, Any]]) -> Optional[str]:
    source_counts: Dict[str, int] = {}
    for point in points:
        if not isinstance(point, dict) or not point.get("source"):
            continue
        source = _fund_service()._normalize_price_source(str(point.get("source") or ""))
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
    if not source_counts:
        return None
    return max(
        source_counts,
        key=lambda source: (
            source_counts[source],
            _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(source, 0),
        ),
    )


def _summary_from_points(fund_code: str, points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ordered = sorted(points, key=lambda item: item["date"])
    if not ordered:
        return None
    latest = ordered[-1]
    previous = ordered[-2] if len(ordered) > 1 else None
    raw = latest.get("raw") if isinstance(latest.get("raw"), dict) else {}
    computed_returns = _seam("_period_returns")(ordered, latest)
    raw_returns = _seam("_period_returns_from_raw")(raw)
    period_returns = {
        key: raw_returns.get(key) if raw_returns.get(key) is not None else computed_returns.get(key)
        for key in ("1w", "1m", "3m", "6m", "ytd", "1y")
    }
    raw_daily_return = _fund_service()._coerce_float(_fund_service()._first_present(raw, "gunlukGetiri", "daily_return"))
    daily_return_source = _fund_service()._first_text(raw, "daily_return_source")
    if raw.get("range_returns_source"):
        daily_return_source = _fund_service().TEFASFON_RETURNS_SOURCE
    management_fee_applied = (
        latest.get("management_fee_applied")
        if latest.get("management_fee_applied") is not None
        else _fund_service()._coerce_tefas_percentage(_fund_service()._first_present(raw, "uygulananYu1Y", "management_fee_applied"))
    )
    management_fee_prospectus = (
        latest.get("management_fee_prospectus")
        if latest.get("management_fee_prospectus") is not None
        else _fund_service()._coerce_tefas_percentage(_fund_service()._first_present(raw, "fonIcTuzukYu1G", "management_fee_prospectus"))
    )
    total_expense_ratio = (
        latest.get("total_expense_ratio")
        if latest.get("total_expense_ratio") is not None
        else _fund_service()._coerce_tefas_percentage(_fund_service()._first_present(raw, "fonTopGiderKesoran", "total_expense_ratio"))
    )
    # Önce uygulanan oranı seç; ama bu 0 (örn. performans ücretli serbest fonlar)
    # ise prospektus / iç tüzük üst sınırına ya da toplam gider oranına düş.
    if management_fee_applied is not None and management_fee_applied > 0:
        management_fee = management_fee_applied
    elif management_fee_prospectus is not None and management_fee_prospectus > 0:
        management_fee = management_fee_prospectus
    elif total_expense_ratio is not None and total_expense_ratio > 0:
        management_fee = total_expense_ratio
    else:
        management_fee = management_fee_applied
    return {
        "fund_code": fund_code,
        "name": latest.get("name") or fund_code,
        "fund_type": latest.get("fund_type"),
        "founder_company": latest.get("founder_company"),
        "manager_company": latest.get("manager_company"),
        "tefas_open": latest.get("tefas_open"),
        "price": latest.get("price"),
        "daily_return": raw_daily_return if raw_daily_return is not None else _seam("_return_between")(latest.get("price"), previous.get("price") if previous else None),
        "daily_return_source": daily_return_source,
        "period_returns": period_returns,
        "risk_value": latest.get("risk_value"),
        "currency": "TRY",
        "as_of": latest.get("date"),
        "source": _fund_service()._public_price_source(str(latest.get("source") or _fund_service().TEFASFON_FUNDS_SOURCE)),
        "aum": latest.get("aum"),
        "investor_count": latest.get("investor_count"),
        "share_count": latest.get("share_count"),
        "management_fee": management_fee,
        "management_fee_applied": management_fee_applied,
        "management_fee_prospectus": management_fee_prospectus,
        "total_expense_ratio": total_expense_ratio,
        "tax_info": _fund_service()._fund_tax_info(latest.get("fund_type")),
        "isin": _fund_service()._first_text(raw, "ISIN", "ISINKODU"),
    }


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
    min_aum: Optional[float] = _UNSET,
    auto_refresh: bool = False,
    include_revision: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]]:
    if min_aum is _UNSET:
        min_aum = _fund_service().FUNDS_LIST_MIN_AUM
    started = time.perf_counter()
    snapshot = _fund_service().load_funds_snapshot(processed_dir, include_rows=False)
    snapshot_loaded = time.perf_counter()
    # A previously failed weekend refresh may already be stored. Present the
    # Friday/last-business-day snapshot as current instead of alarming users.
    if _fund_service()._snapshot_is_current_on_market_closure(snapshot):
        snapshot = _fund_service()._mark_snapshot_current_for_market_closure(snapshot)
    if auto_refresh:
        snapshot = _fund_service()._refresh_funds_snapshot_if_stale(processed_dir, snapshot)
    database_rows = 0
    row_store_active = bool(
        isinstance(snapshot.get("source_metadata"), dict)
        and snapshot["source_metadata"].get("catalog_storage") == "fund_snapshots"
    )
    if row_store_active:
        rows, filtered_count = _fund_service()._query_fund_snapshot_rows(
            processed_dir,
            q=q,
            fund_type=fund_type,
            founder=founder,
            manager=manager,
            risk=risk,
            sort=sort,
            order=order,
            min_aum=min_aum,
            limit=limit,
        )
        database_rows = len(rows)
        rows = _fund_service()._overlay_latest_price_history(processed_dir, {"rows": rows}).get("rows") or []
    else:
        rows = [
            row
            for row in list(snapshot.get("rows") or [])
            if isinstance(row, dict)
            and _fund_service()._is_tefas_open_row(row)
            and _fund_service()._meets_min_aum(row, min_aum)
            and _fund_service()._row_matches(row, q=q, fund_type=fund_type, founder=founder, manager=manager, risk=risk)
        ]
        filtered_count = len(rows)
        database_rows = len(rows)
    filtered = time.perf_counter()
    rows = _fund_service()._apply_daily_return_overrides(processed_dir, rows, include_missing=True)
    # These fields can be replaced by newer fund_prices rows after the SQL
    # query, so restore the legacy order using the values the API returns.
    if row_store_active and sort in {"price", "daily_return", "as_of"}:
        rows = _fund_service()._sort_rows(rows, sort, order)
    reconciled = time.perf_counter()
    if not row_store_active:
        rows = _fund_service()._sort_rows(rows, sort, order)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    finished = time.perf_counter()
    _fund_service().logging.getLogger("uvicorn.error").info(
        "fund_catalog rows=%d db_rows=%d matched_rows=%d query_limit=%s snapshot_ms=%.1f filter_ms=%.1f returns_ms=%.1f sort_ms=%.1f total_ms=%.1f",
        len(rows), database_rows, filtered_count, limit, (snapshot_loaded - started) * 1000, (filtered - snapshot_loaded) * 1000,
        (reconciled - filtered) * 1000, (finished - reconciled) * 1000, (finished - started) * 1000,
    )
    meta = dict(snapshot.get("source_metadata") or {})
    meta["list_min_aum"] = min_aum
    payload = {
        "status": snapshot.get("status") or ("ok" if rows else "empty"),
        "rows": rows,
        "count": filtered_count,
        "total_count": int(snapshot.get("total_count") or snapshot.get("count") or filtered_count),
        "source": _fund_service()._public_price_source(str(snapshot.get("source") or _fund_service().TEFASFON_FUNDS_SOURCE)),
        "source_url": snapshot.get("source_url", _fund_service().TEFASFON_SOURCE_URL),
        "as_of": snapshot.get("as_of"),
        "fetched_at": snapshot.get("fetched_at"),
        "stale": bool(snapshot.get("stale")),
        "degraded": bool(snapshot.get("degraded")),
        "warnings": list(snapshot.get("warnings") or []),
        "source_metadata": meta,
    }
    if not include_revision:
        return payload

    source_revision = _fund_service().fund_listing_revision(processed_dir)
    return payload, source_revision


def get_fund_categories_payload(processed_dir: Path) -> Dict[str, Any]:
    snapshot = _fund_service().load_funds_snapshot(
        processed_dir,
        include_rows=False,
        include_price_metadata=False,
    )
    meta = dict(snapshot.get("source_metadata") or {})
    meta["list_min_aum"] = _fund_service().FUNDS_LIST_MIN_AUM
    row_store_active = meta.get("catalog_storage") == "fund_snapshots"
    if row_store_active:
        cached_all = _fund_service()._peek_all_snapshot_rows(
            processed_dir,
            snapshot.get("snapshot_generation"),
        )
        if cached_all is not None:
            rows = [
                row
                for row in list(cached_all.get("rows") or [])
                if isinstance(row, dict)
                and _fund_service()._is_tefas_open_row(row)
                and _fund_service()._meets_min_aum(row, _fund_service().FUNDS_LIST_MIN_AUM)
            ]
            category_values = {
                key: sorted({str(row.get(key)).strip() for row in rows if row.get(key)})
                for key in ("fund_type", "founder_company", "manager_company")
            }
            risk_values = sorted(
                {
                    int(row["risk_value"])
                    for row in rows
                    if isinstance(row.get("risk_value"), int)
                    and not isinstance(row.get("risk_value"), bool)
                }
            )
        else:
            conditions: List[str] = []
            params: List[Any] = []
            if _fund_service().TEFAS_OPEN_ONLY:
                conditions.append("(tefas_open IS NULL OR tefas_open = 1)")
            if _fund_service().FUNDS_LIST_MIN_AUM > 0:
                conditions.append("aum >= ?")
                params.append(_fund_service().FUNDS_LIST_MIN_AUM)
            where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
            with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
                records = conn.execute(
                    "SELECT DISTINCT fund_type, founder_company, manager_company, risk_value_json "
                    f"FROM fund_snapshots{where_sql}",
                    tuple(params),
                ).fetchall()
            unique_values: Dict[str, set[str]] = {
                "fund_type": set(),
                "founder_company": set(),
                "manager_company": set(),
            }
            risk_set: set[int] = set()
            for record in records:
                for key in unique_values:
                    value = record[key]
                    if value:
                        unique_values[key].add(str(value).strip())
                risk_value = _fund_service()._safe_json_loads(record["risk_value_json"], None)
                if isinstance(risk_value, int) and not isinstance(risk_value, bool):
                    risk_set.add(risk_value)
            risk_values = sorted(risk_set)
            category_values = {key: sorted(values) for key, values in unique_values.items()}
    else:
        rows = [
            row
            for row in list(snapshot.get("rows") or [])
            if isinstance(row, dict) and _fund_service()._is_tefas_open_row(row) and _fund_service()._meets_min_aum(row, _fund_service().FUNDS_LIST_MIN_AUM)
        ]
        category_values = {
            key: sorted({str(row.get(key)).strip() for row in rows if row.get(key)})
            for key in ("fund_type", "founder_company", "manager_company")
        }
        risk_values = sorted({int(row["risk_value"]) for row in rows if isinstance(row.get("risk_value"), int)})
    return {
        "status": snapshot.get("status") or "unavailable",
        "fund_types": category_values["fund_type"],
        "founder_companies": category_values["founder_company"],
        "manager_companies": category_values["manager_company"],
        "risk_values": risk_values,
        "source_metadata": meta,
    }


def _fund_reference_metadata(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    """Look up cached metadata (management fees, tax info, founder company, ...)
    from the reference_data SQLite. Returns an empty dict when nothing is found."""

    instrument = _fund_service().get_instrument(processed_dir, "fund", fund_code)
    if not instrument:
        return {}
    metadata = instrument.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _find_fund_row(processed_dir: Path, fund_code: str) -> Optional[Dict[str, Any]]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    snapshot = _fund_service().load_funds_snapshot(
        processed_dir,
        include_rows=False,
        include_price_metadata=False,
    )
    metadata = snapshot.get("source_metadata") if isinstance(snapshot.get("source_metadata"), dict) else {}
    if metadata.get("catalog_storage") == "fund_snapshots":
        cached_all = _fund_service()._peek_all_snapshot_rows(processed_dir, snapshot.get("snapshot_generation"))
        if cached_all is not None:
            for candidate in cached_all["rows"]:
                if not isinstance(candidate, dict):
                    continue
                if (
                    _fund_service().normalize_fund_code(str(candidate.get("fund_code") or candidate.get("fonKodu") or "")) == normalized
                    and _fund_service()._is_tefas_open_row(candidate)
                ):
                    return dict(candidate)
            return None
        where = "fund_code = ?"
        params: List[Any] = [normalized]
        if _fund_service().TEFAS_OPEN_ONLY:
            where += " AND (tefas_open IS NULL OR tefas_open = 1)"
        found = _fund_service()._read_fund_snapshot_rows(
            processed_dir,
            where=where,
            params=params,
        )
        row = found[0] if found else None
    else:
        # Legacy/empty catalogues have not been migrated into the row store;
        # keep their existing full-snapshot fallback until a valid refresh.
        legacy_snapshot = _fund_service().load_funds_snapshot(processed_dir, include_rows=True)
        row = next(
            (
                dict(candidate)
                for candidate in list(legacy_snapshot.get("rows") or [])
                if isinstance(candidate, dict)
                and _fund_service()._is_tefas_open_row(candidate)
                and _fund_service().normalize_fund_code(str(candidate.get("fund_code") or "")) == normalized
            ),
            None,
        )
    if row is None:
        return None
    overlaid = _fund_service()._overlay_latest_price_history(processed_dir, {"rows": [row]}).get("rows") or []
    if overlaid:
        row = overlaid[0]
    # Backfill management fee / tax fields from reference_data so a stale
    # snapshot keeps surfacing the last-known values on the detail endpoint.
    ref_meta = _seam("_fund_reference_metadata")(processed_dir, normalized)
    for key in (
        "management_fee_applied",
        "management_fee_prospectus",
        "total_expense_ratio",
        "tax_info",
    ):
        if row.get(key) is None and ref_meta.get(key) is not None:
            row[key] = ref_meta[key]
    return _fund_service()._apply_daily_return_overrides(processed_dir, [row], include_missing=True)[0]


def _fund_return_rank_value(row: Dict[str, Any], key: str) -> Optional[float]:
    period_returns = row.get("period_returns")
    if not isinstance(period_returns, dict):
        return None
    return _fund_service()._coerce_float(period_returns.get(key))


def _fund_category_rankings(
    snapshot: Dict[str, Any],
    row: Dict[str, Any],
    *,
    processed_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    category = str(row.get("fund_type") or "").strip()
    normalized_code = _fund_service().normalize_fund_code(str(row.get("fund_code") or ""))
    if not category or not normalized_code:
        return {"category": category or None, "category_total": 0, "as_of": snapshot.get("as_of"), "items": []}

    snapshot_meta = snapshot.get("source_metadata") if isinstance(snapshot.get("source_metadata"), dict) else {}
    if processed_dir is not None and snapshot_meta.get("catalog_storage") == "fund_snapshots":
        category_rows = _fund_service()._read_fund_snapshot_category_returns(
            processed_dir,
            fund_type=category,
            min_aum=_fund_service().FUNDS_LIST_MIN_AUM,
        )
    else:
        category_rows = []
        seen_codes: set[str] = set()
        for candidate in list(snapshot.get("rows") or []):
            if not isinstance(candidate, dict):
                continue
            candidate_code = _fund_service().normalize_fund_code(str(candidate.get("fund_code") or ""))
            if not candidate_code or candidate_code in seen_codes:
                continue
            if str(candidate.get("fund_type") or "").strip() != category:
                continue
            if not _fund_service()._is_tefas_open_row(candidate) or not _fund_service()._meets_min_aum(candidate, _fund_service().FUNDS_LIST_MIN_AUM):
                continue
            seen_codes.add(candidate_code)
            category_rows.append(candidate)

    metrics = [
        ("1m", "Aylık Getiri"),
        ("ytd", "YBB Getiri"),
        ("1y", "1 Yıllık Getiri"),
        ("6m", "6 Aylık Getiri"),
    ]
    items: List[Dict[str, Any]] = []
    for key, label in metrics:
        selected_value = _seam("_fund_return_rank_value")(row, key)
        if selected_value is None:
            continue
        values: List[float] = []
        for candidate in category_rows:
            candidate_value = _seam("_fund_return_rank_value")(candidate, key)
            if candidate_value is not None:
                values.append(candidate_value)
        total = len(values)
        if total <= 0:
            continue
        rank = 1 + sum(1 for value in values if value is not None and value > selected_value)
        top_percentile = ((total - rank + 1) / total) * 100 if total else None
        items.append(
            {
                "key": key,
                "label": label,
                "value": selected_value,
                "rank": rank,
                "total": total,
                "top_percentile": round(top_percentile) if top_percentile is not None else None,
                "direction": "higher_is_better",
            }
        )

    return {
        "category": category,
        "category_total": len(category_rows),
        "as_of": snapshot.get("as_of"),
        "items": items,
    }


def get_fund_detail_payload(processed_dir: Path, fund_code: str) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    row = _seam("_find_fund_row")(processed_dir, normalized)
    if not row:
        raise KeyError(normalized)
    snapshot = _fund_service().load_funds_snapshot(
        processed_dir,
        include_rows=False,
        include_price_metadata=False,
    )
    applied = row.get("management_fee_applied")
    prospectus = row.get("management_fee_prospectus")
    ter = row.get("total_expense_ratio")
    if row.get("management_fee") is not None and (
        not isinstance(row.get("management_fee"), (int, float))
        or row.get("management_fee") > 0
    ):
        management_fee = row.get("management_fee")
    elif applied is not None and applied > 0:
        management_fee = applied
    elif prospectus is not None and prospectus > 0:
        management_fee = prospectus
    elif ter is not None and ter > 0:
        management_fee = ter
    else:
        management_fee = applied if applied is not None else prospectus
    tax_info = row.get("tax_info") or _fund_service()._fund_tax_info(row.get("fund_type"))
    return {
        **row,
        "isin": row.get("isin"),
        "strategy": None,
        "benchmark": None,
        "management_fee": management_fee,
        "management_fee_applied": row.get("management_fee_applied"),
        "management_fee_prospectus": row.get("management_fee_prospectus"),
        "total_expense_ratio": row.get("total_expense_ratio"),
        "tax_info": tax_info,
        "fintables_url": f"{_fund_service().FINTABLES_FUND_BASE_URL}/{normalized}",
        "kap_url": None,
        "category_rankings": _seam("_fund_category_rankings")(snapshot, row, processed_dir=processed_dir),
        "source_metadata": snapshot.get("source_metadata") or {},
    }


def get_fund_yield_summary_payload(fund_code: str, processed_dir: Optional[Path] = None) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    warnings: List[str] = []
    fallback_used = False
    source = _fund_service().FINTABLES_YIELD_SUMMARY_SOURCE
    source_url = _fund_service().FINTABLES_YIELD_SUMMARY_ENDPOINT
    summary_source_used = _fund_service().FINTABLES_YIELD_SUMMARY_SOURCE
    latest_price: Optional[float] = None
    latest_date: Optional[str] = None
    if processed_dir is not None:
        latest_row = _seam("_find_fund_row")(processed_dir, normalized)
        if latest_row:
            latest_price = _fund_service()._coerce_float(latest_row.get("price"))
            latest_date = _fund_service()._fund_date(latest_row.get("as_of"))
    summary: Dict[str, Any] = {}
    try:
        summary = _fund_service().fetch_fintables_yield_summary(normalized)
        periods = summary.get("periods") if isinstance(summary, dict) else None
        if not isinstance(periods, dict) or not periods:
            warnings.append("fintables_yield_summary returned no usable periods")
            summary = {}
        else:
            source = _fund_service()._public_price_source(str(summary.get("source") or source))
            source_url = str(summary.get("source_url") or source_url)
            summary_source_used = source
    except _fund_service().FintablesUpstreamError as exc:
        warnings.append(f"fintables_yield_summary failed: {exc}")

    if not summary:
        try:
            summary = _fund_service().TefasFonClient().fetch_yield_summary(
                normalized,
                latest_price=latest_price,
                latest_date=latest_date,
            )
            periods = summary.get("periods") if isinstance(summary, dict) else None
            if isinstance(periods, dict) and periods:
                fallback_used = True
                source = _fund_service()._public_price_source(str(summary.get("source") or _fund_service().TEFASFON_FUNDS_SOURCE))
                source_url = str(summary.get("source_url") or _fund_service().TEFASFON_SOURCE_URL)
                summary_source_used = source
            else:
                warnings.append("tefasfon_yield_summary returned no usable periods")
                summary = {}
        except _fund_service().TefasUpstreamError as exc:
            warnings.append(f"tefasfon_yield_summary failed: {exc}")

    if not summary:
        summary = {
            "fund_code": normalized,
            "source": _fund_service().FINTABLES_YIELD_SUMMARY_SOURCE,
            "source_url": _fund_service().FINTABLES_YIELD_SUMMARY_ENDPOINT,
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
            "fetched_at": _fund_service()._utc_now_iso(),
            "purpose": "period_summary_only",
            "writes_fund_prices": False,
            "summary_source_used": summary_source_used,
            "source_policy": _fund_service().FUND_HISTORY_SOURCE_POLICY,
            "fallback_used": fallback_used,
            "adapter_version": _fund_service()._tefasfon_adapter_version() if source.startswith("tefasfon") else None,
            "warnings": warnings,
            "warning": warnings[0] if warnings else None,
        },
    }


def _valid_performance_points(points: Iterable[Dict[str, Any]], normalized_code: str) -> List[Dict[str, Any]]:
    valid: Dict[str, Dict[str, Any]] = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        point_date = _fund_service()._fund_date(point.get("date"))
        price = _fund_service()._coerce_float(point.get("price"))
        if not point_date or price is None or price <= 0:
            continue
        valid[point_date] = {
            **point,
            "fund_code": _fund_service().normalize_fund_code(str(point.get("fund_code") or normalized_code)),
            "date": point_date,
            "price": price,
            "daily_return": _fund_service()._coerce_float(point.get("daily_return")),
            "aum": _fund_service()._coerce_float(point.get("aum")),
            "investor_count": _fund_service()._coerce_int(point.get("investor_count")),
            "share_count": _fund_service()._coerce_float(point.get("share_count")),
            "source": _fund_service()._public_price_source(str(point.get("source") or _fund_service().TEFASFON_FUNDS_SOURCE)),
        }
    ordered = [valid[key] for key in sorted(valid)]
    previous_price: Optional[float] = None
    for point in ordered:
        current_price = _fund_service()._coerce_float(point.get("price"))
        if point.get("daily_return") is None and previous_price is not None:
            point["daily_return"] = _seam("_return_between")(current_price, previous_price)
        if current_price is not None:
            previous_price = current_price
    return ordered


def _legacy_history_points(processed_dir: Path, normalized_code: str) -> List[Dict[str, Any]]:
    payload = _fund_service()._read_json(_fund_service()._history_path(processed_dir, normalized_code)) or {}
    return _seam("_valid_performance_points")(list(payload.get("points") or []), normalized_code)


def _migrate_legacy_history_to_sqlite(processed_dir: Path, normalized_code: str) -> Dict[str, Any]:
    legacy_points = _seam("_legacy_history_points")(processed_dir, normalized_code)
    if not legacy_points:
        return {
            "upserted_count": 0,
            "skipped_count": 0,
            "warning_count": 0,
            "warnings": [],
        }
    return _fund_service().upsert_fund_price_points(
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
    recent_detail_backfill: Optional[Dict[str, Any]] = None,
    overview_metric_backfill: Optional[Dict[str, Any]] = None,
    full_history_requested: bool = False,
    history_job: Optional[Dict[str, Any]] = None,
    requested_resolution: str = "daily",
) -> Dict[str, Any]:
    ordered = _seam("_valid_performance_points")(points, normalized_code)
    latest_point_date = ordered[-1]["date"] if ordered else None
    latest_fetch = fetched_at or max(
        (str(point.get("fetched_at") or "") for point in ordered if point.get("fetched_at")),
        default=None,
    )
    if stale is None:
        stale = (_fund_service()._cache_age_seconds(latest_fetch) or (_fund_service().FUNDS_HISTORY_TTL_SECONDS + 1)) > _fund_service().FUNDS_HISTORY_TTL_SECONDS
    effective_end = end_date or (date.fromisoformat(latest_point_date) if latest_point_date else _fund_service().date.today())
    coverage = _seam("_history_coverage_info")(ordered, effective_end)
    coverage_warnings = list(coverage.get("warnings") or [])
    resolution = _seam("_history_resolution")(ordered)
    internal_gap_warnings = _seam("_history_internal_gap_warnings")(ordered, resolution=resolution)
    all_warnings = list(warnings or []) + coverage_warnings + internal_gap_warnings
    date_min = ordered[0]["date"] if ordered else None
    date_max = ordered[-1]["date"] if ordered else None
    coverage_state = "complete"
    if ordered:
        first_date = date.fromisoformat(date_min) if date_min else None
        start_gap = (
            _fund_service()._business_days_between(start_date, first_date - timedelta(days=1))
            if start_date and first_date and first_date > start_date
            else 0
        )
        # A late tail is a refresh/freshness concern, not evidence that the
        # fund's historical range starts later. The API can expose the local
        # points immediately while its background job fills that tail.
        end_gap = int(coverage.get("coverage_gap_business_days") or 0)
        if start_gap > 3 or end_gap > 3:
            coverage_state = "range_incomplete"
    else:
        coverage_state = "unavailable"
    daily_upgrade_state = "idle"
    if history_job:
        job_status = str(history_job.get("status") or "").strip().lower()
        if job_status in {"queued", "running"} and resolution != "daily":
            coverage_state = "upgrading" if coverage_state == "complete" else coverage_state
            daily_upgrade_state = "pending"
        elif job_status == "succeeded" and resolution != "daily":
            daily_upgrade_state = "unavailable"
        elif job_status == "failed":
            daily_upgrade_state = "failed"
    history_source_used = _seam("_dominant_price_source")(ordered) or _fund_service().TEFASFON_FUNDS_SOURCE
    fallback_point_count = sum(
        1
        for point in ordered
        if _fund_service()._normalize_price_source(str(point.get("source") or "")) == _fund_service().FINTABLES_UDF_HISTORY_SOURCE
    )
    metadata: Dict[str, Any] = {
        "source": "sqlite",
        "source_url": str(_fund_service()._fund_prices_db_path(processed_dir)),
        "db_path": str(_fund_service()._fund_prices_db_path(processed_dir)),
        "fetched_at": latest_fetch,
        "as_of": latest_point_date,
        "cache_hit": cache_hit,
        "stale": stale,
        "parse_status": parse_status or ("ok" if ordered else "empty"),
        "warnings": all_warnings,
        "warning": all_warnings[0] if all_warnings else None,
        "history_source_used": history_source_used,
        "history_source_policy": _fund_service().FUND_HISTORY_SOURCE_POLICY,
        "source_policy": _fund_service().FUND_HISTORY_SOURCE_POLICY,
        "primary_source": "fintables",
        "tefasfon_adapter_version": _fund_service()._tefasfon_adapter_version(),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "fintables_point_count": fallback_point_count,
        "cached_fallback_points_present": fallback_point_count > 0,
        "cached_fallback_point_count": fallback_point_count,
        "final_points_count": len(ordered),
        "date_min": date_min,
        "date_max": date_max,
        "backfill_used": backfill_used,
        "full_history_requested": bool(full_history_requested),
        "coverage_state": coverage_state,
        "resolution": resolution,
        "requested_resolution": requested_resolution,
        "available_start_date": date_min,
        "available_end_date": date_max,
        "daily_upgrade_state": daily_upgrade_state,
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
    if recent_detail_backfill is not None:
        metadata["recent_detail_backfill"] = recent_detail_backfill
    if overview_metric_backfill is not None:
        metadata["overview_metric_backfill"] = overview_metric_backfill
    if history_job is not None:
        metadata["history_job"] = history_job

    return {
        "fund_code": normalized_code,
        "status": "ok" if ordered else "empty",
        "points": ordered,
        "period_stats": _seam("_fund_period_stats")(ordered, as_of=effective_end),
        "source": "sqlite",
        "source_url": str(_fund_service()._fund_prices_db_path(processed_dir)),
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
        if _fund_service()._business_days_between(start_date, min(parsed_dates) - timedelta(days=1)) > 3:
            return False
    resolution = _seam("_history_resolution")(points)
    if _seam("_history_internal_gap_warnings")(points, resolution=resolution):
        return False
    if end_date:
        latest = max(parsed_dates)
        gap_days = max(0, (end_date - latest).days)
        if gap_days > 3 and _fund_service()._business_days_between(latest + timedelta(days=1), end_date) > 3:
            return False
    return True


def _latest_fund_point_date(points: List[Dict[str, Any]]) -> Optional[date]:
    latest: Optional[date] = None
    for point in points:
        raw_date = point.get("date") if isinstance(point, dict) else None
        if not isinstance(raw_date, str):
            continue
        try:
            point_date = date.fromisoformat(raw_date)
        except ValueError:
            continue
        if latest is None or point_date > latest:
            latest = point_date
    return latest


def _recent_tail_refresh_target(end_date: date) -> date:
    return min(end_date, _fund_service()._latest_fund_snapshot_target_date())


def _needs_recent_tail_refresh(points: List[Dict[str, Any]], *, end_date: date) -> bool:
    latest = _seam("_latest_fund_point_date")(points)
    if latest is None:
        return False
    target_date = _seam("_recent_tail_refresh_target")(end_date)
    if latest >= target_date:
        return False
    return _fund_service()._business_days_between(latest + timedelta(days=1), target_date) > 0


def _history_cache_covers_requested_span(
    processed_dir: Path,
    normalized_code: str,
    *,
    start_date: date,
    end_date: date,
) -> bool:
    history_path = _fund_service()._history_path(processed_dir, normalized_code)
    metadata = _fund_service().read_json_cache_history_metadata(history_path)
    if metadata is None:
        payload = _fund_service()._read_local_json(history_path) or {}
        metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
        if not metadata.get("as_of") and payload.get("as_of"):
            metadata = {**metadata, "as_of": payload["as_of"]}
    if str(metadata.get("parse_status") or "").startswith("unavailable"):
        return False
    requested_start = _fund_service()._fund_date(metadata.get("requested_start_date"))
    if not requested_start:
        return False
    try:
        if date.fromisoformat(requested_start) > start_date:
            return False
    except ValueError:
        return False
    latest_point_date = _fund_service()._fund_date(metadata.get("date_max") or metadata.get("as_of"))
    if not latest_point_date:
        return False
    try:
        latest = date.fromisoformat(latest_point_date)
    except ValueError:
        return False
    gap_days = max(0, (end_date - latest).days)
    return gap_days <= 3 or _fund_service()._business_days_between(latest + timedelta(days=1), end_date) <= 3


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _shift_month(month_start: date, offset: int) -> date:
    month_index = month_start.year * 12 + (month_start.month - 1) + offset
    return date(month_index // 12, (month_index % 12) + 1, 1)


def _month_end(month_start: date) -> date:
    next_month = _seam("_shift_month")(month_start, 1)
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


def _fund_period_stats(points: List[Dict[str, Any]], *, as_of: Optional[date] = None) -> Dict[str, Any]:
    dated_points: List[Tuple[date, Dict[str, Any], float]] = []
    for point in points:
        point_date = _seam("_fund_point_date")(point)
        price = _fund_service()._coerce_float(point.get("price"))
        if point_date is None or price is None or price <= 0:
            continue
        dated_points.append((point_date, point, price))
    dated_points.sort(key=lambda item: item[0])
    if not dated_points:
        return {"as_of": None, "periods": []}

    latest_available = dated_points[-1][0]
    anchor = min(as_of, latest_available) if as_of is not None else latest_available
    current_month_start = _seam("_month_start")(anchor)
    previous_month_start = _seam("_shift_month")(current_month_start, -1)
    period_defs = [
        ("current_month", _seam("_month_key")(current_month_start), current_month_start, min(anchor, _seam("_month_end")(current_month_start))),
        ("last_30_days", "last_30_days", anchor - timedelta(days=29), anchor),
        ("previous_month", _seam("_month_key")(previous_month_start), previous_month_start, _seam("_month_end")(previous_month_start)),
    ]

    periods: List[Dict[str, Any]] = []
    for key, label, start, end in period_defs:
        selected = [
            (idx, point_date, point, price)
            for idx, (point_date, point, price) in enumerate(dated_points)
            if start <= point_date <= end
        ]
        if not selected:
            periods.append(
                {
                    "key": key,
                    "label": label,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "trading_days": 0,
                    "return_days": 0,
                    "positive_days": 0,
                    "negative_days": 0,
                    "flat_days": 0,
                    "average_daily_return": None,
                    "cumulative_return": None,
                    "best_day_return": None,
                    "best_day_date": None,
                    "worst_day_return": None,
                    "worst_day_date": None,
                    "basis": "none",
                }
            )
            continue

        return_rows: List[Tuple[date, float]] = []
        for idx, point_date, point, price in selected:
            daily_return = _fund_service()._coerce_float(point.get("daily_return"))
            if daily_return is None and idx > 0:
                daily_return = _seam("_return_between")(price, dated_points[idx - 1][2])
            if daily_return is not None:
                return_rows.append((point_date, daily_return))

        first_idx = selected[0][0]
        base_price = dated_points[first_idx - 1][2] if first_idx > 0 else selected[0][3]
        basis = "previous_close" if first_idx > 0 else "first_point"
        cumulative_return = _seam("_return_between")(selected[-1][3], base_price)
        best = max(return_rows, key=lambda item: item[1], default=None)
        worst = min(return_rows, key=lambda item: item[1], default=None)
        positive_days = sum(1 for _, value in return_rows if value > 0)
        negative_days = sum(1 for _, value in return_rows if value < 0)
        flat_days = len(return_rows) - positive_days - negative_days
        average_daily_return = (
            sum(value for _, value in return_rows) / len(return_rows)
            if return_rows
            else None
        )

        periods.append(
            {
                "key": key,
                "label": label,
                "start_date": selected[0][1].isoformat(),
                "end_date": selected[-1][1].isoformat(),
                "trading_days": len(selected),
                "return_days": len(return_rows),
                "positive_days": positive_days,
                "negative_days": negative_days,
                "flat_days": flat_days,
                "average_daily_return": average_daily_return,
                "cumulative_return": cumulative_return,
                "best_day_return": best[1] if best else None,
                "best_day_date": best[0].isoformat() if best else None,
                "worst_day_return": worst[1] if worst else None,
                "worst_day_date": worst[0].isoformat() if worst else None,
                "basis": basis,
            }
        )

    return {"as_of": anchor.isoformat(), "periods": periods}


def _overview_metric_targets(end_date: date) -> List[Dict[str, Any]]:
    visible_months = max(1, _fund_service().FUNDS_OVERVIEW_METRIC_MONTHS)
    end_month = _seam("_month_start")(end_date)
    targets: List[Dict[str, Any]] = []
    for offset in range(-visible_months, 1):
        month_start = _seam("_shift_month")(end_month, offset)
        target_date = min(_seam("_month_end")(month_start), end_date)
        targets.append(
            {
                "month": _seam("_month_key")(month_start),
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
        point_date = _seam("_fund_point_date")(point)
        if point_date is None or _seam("_month_key")(point_date) != month or point_date > target_date:
            continue
        if latest is None or point_date > latest[0]:
            latest = (point_date, point)
    return latest[1] if latest else None


def _has_overview_metrics(point: Optional[Dict[str, Any]]) -> bool:
    if point is None:
        return False
    aum = _fund_service()._coerce_float(point.get("aum"))
    investor_count = _fund_service()._coerce_int(point.get("investor_count"))
    return aum is not None and aum > 0 and investor_count is not None


def _recent_detail_window(
    points: List[Dict[str, Any]],
    *,
    start_date: date,
    end_date: date,
) -> Optional[Tuple[date, date]]:
    if not points:
        return None
    latest = _seam("_latest_fund_point_date")(points)
    if latest is None:
        return None
    window_end = min(end_date, latest)
    lookback_days = max(1, _fund_service().FUNDS_RECENT_DETAIL_LOOKBACK_DAYS)
    window_start = max(start_date, window_end - timedelta(days=lookback_days - 1))
    if window_start > window_end:
        return None
    return window_start, window_end


def _recent_detail_missing_count(
    points: List[Dict[str, Any]],
    *,
    start_date: date,
    end_date: date,
) -> int:
    missing_count = 0
    for point in points:
        point_date = _seam("_fund_point_date")(point)
        if point_date is None or point_date < start_date or point_date > end_date:
            continue
        source = _fund_service()._normalize_price_source(str(point.get("source") or ""))
        if source != _fund_service().TEFASFON_FUNDS_SOURCE and not _seam("_has_overview_metrics")(point):
            missing_count += 1
    return missing_count


def _fetch_recent_detail_rows(
    fund_code: str,
    *,
    start_date: date,
    end_date: date,
    client: TefasFonClient,
    processed_dir: Optional[Path] = None,
    missing_dates: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    tefas_range_rate_limited = False
    try:
        rows = _fund_service().TefasClient().fetch_fund_history(
            fund_codes=[normalized],
            start_date=start_date,
            end_date=end_date,
        )
    except _fund_service().TefasUpstreamError as exc:
        tefas_range_rate_limited = _fund_service()._is_tefas_rate_limit(exc)
        warnings.append(f"tefas_fund_list recent detail range fallback failed: {exc}")
    else:
        if rows:
            valid_rows = [
                row
                for row in rows
                if isinstance(row, dict)
                and _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or row.get("FONKODU") or "")) == normalized
            ]
            return valid_rows, warnings
    if tefas_range_rate_limited:
        warnings.append("tefas_fund_list recent detail skipped secondary fallbacks after rate limit")
        return [], warnings

    use_daily_snapshot_fallback = False
    if processed_dir is not None and hasattr(client, "_module"):
        try:
            client._module()
        except _fund_service().TefasUpstreamError as exc:
            use_daily_snapshot_fallback = True
            warnings.append(f"tefasfon_funds recent detail range backfill failed: {exc}")
    if not use_daily_snapshot_fallback:
        try:
            rows = client.fetch_history(normalized, start_date, end_date)
        except _fund_service().TefasUpstreamError as exc:
            rows = []
            warnings.append(f"tefasfon_funds recent detail range backfill failed: {exc}")
    if not rows and processed_dir is not None and hasattr(client, "fetch_daily_funds_snapshot"):
        target_dates = set(missing_dates or [])
        # Prefer the indexed row store: one small query instead of one
        # multi-megabyte daily snapshot document per date. Dates the store
        # cannot fill still fall through to the snapshot documents below.
        stored_rows = [
            row
            for row in _fund_service().read_fund_price_points_range(
                processed_dir,
                normalized,
                start_date=start_date,
                end_date=end_date,
            )
            if (not target_dates or str(row.get("date") or "") in target_dates)
            and _seam("_has_overview_metrics")(row)
        ]
        if stored_rows:
            warnings.append("tefasfon_funds recent detail used fund_prices row store")
        stored_dates = {str(row.get("date") or "") for row in stored_rows}
        if stored_dates and (not target_dates or stored_dates >= target_dates):
            return stored_rows, warnings
        remaining_dates = target_dates - stored_dates

        snapshot_rows: List[Dict[str, Any]] = list(stored_rows)
        daily_warnings: List[str] = []
        used_snapshot_fallback = False
        current = start_date
        while current <= end_date:
            if current.weekday() >= 5 or (remaining_dates and current.isoformat() not in remaining_dates):
                current += timedelta(days=1)
                continue
            try:
                day_rows, _cache_hit = _fund_service()._cached_daily_funds_snapshot(processed_dir, client, current)
            except _fund_service().TefasUpstreamError as exc:
                daily_warnings.append(f"tefasfon_funds recent detail daily snapshot failed for {current.isoformat()}: {exc}")
                current += timedelta(days=1)
                continue
            if not day_rows:
                if _fund_service().FUNDS_WEB_HISTORY_SLEEP_SECONDS > 0:
                    time.sleep(_fund_service().FUNDS_WEB_HISTORY_SLEEP_SECONDS)
                try:
                    day_rows, _cache_hit = _fund_service()._cached_daily_funds_snapshot(
                        processed_dir,
                        client,
                        current,
                        force_refresh=True,
                    )
                except _fund_service().TefasUpstreamError as exc:
                    daily_warnings.append(f"tefasfon_funds recent detail daily snapshot retry failed for {current.isoformat()}: {exc}")
                    current += timedelta(days=1)
                    continue
            if not day_rows:
                daily_warnings.append(f"tefasfon_funds recent detail daily snapshot returned no rows for {current.isoformat()}")
                current += timedelta(days=1)
                continue
            match = next(
                (
                    row
                    for row in day_rows
                    if _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or row.get("FONKODU") or "")) == normalized
                ),
                None,
            )
            if match:
                normalized_match = dict(match)
                normalized_match["source"] = _fund_service().TEFASFON_FUNDS_SOURCE
                snapshot_rows.append(normalized_match)
                used_snapshot_fallback = True
            if not _cache_hit and _fund_service().FUNDS_WEB_HISTORY_SLEEP_SECONDS > 0:
                time.sleep(_fund_service().FUNDS_WEB_HISTORY_SLEEP_SECONDS)
            current += timedelta(days=1)
        if snapshot_rows:
            rows = snapshot_rows
            if used_snapshot_fallback:
                warnings.append("tefasfon_funds recent detail used daily snapshot fallback")
            warnings.extend(daily_warnings[:20])
        else:
            warnings.extend(daily_warnings[:20])
    valid_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or row.get("FONKODU") or "")) == normalized
    ]
    if not valid_rows:
        warnings.append(
            f"tefasfon_funds recent detail backfill returned no rows for {normalized} "
            f"between {start_date.isoformat()} and {end_date.isoformat()}"
        )
    return valid_rows, warnings


def _missing_overview_metric_targets(
    points: List[Dict[str, Any]],
    *,
    end_date: date,
    targets: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    missing: List[Dict[str, Any]] = []
    selected_targets = _seam("_overview_metric_targets")(end_date) if targets is None else targets
    has_fintables_points = any(
        isinstance(point, dict)
        and _fund_service()._normalize_price_source(str(point.get("source") or "")) == _fund_service().FINTABLES_UDF_HISTORY_SOURCE
        for point in points
    )
    for target in selected_targets:
        point = _seam("_latest_point_for_month")(
            points,
            month=str(target["month"]),
            target_date=target["target_date"],
        )
        if point is None and has_fintables_points:
            # A TEFAS-only monthly anchor without a Fintables price would be
            # discarded by the preferred performance reader. Do not fetch it
            # repeatedly or reintroduce a mixed history through the cache.
            continue
        if not _seam("_has_overview_metrics")(point):
            missing.append(target)
    return missing


def _fetch_fund_overview_metric_rows(
    fund_code: str,
    targets: List[Dict[str, Any]],
    *,
    client: TefasFonClient,
    processed_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    rows: List[Dict[str, Any]] = []
    fetched_months: Dict[str, str] = {}
    warnings: List[str] = []

    # The same metrics are already persisted in `fund_prices` when the daily
    # snapshot was captured. One indexed window read covers every target month,
    # so the slow per-date snapshot documents are only touched when a month is
    # genuinely missing from the row store.
    stored_by_month: Dict[str, Dict[str, Any]] = {}
    if processed_dir is not None and targets:
        lower_bounds = {
            str(target["month"]): max(
                target["month_start"],
                target["target_date"] - timedelta(days=max(0, _fund_service().FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS)),
            )
            for target in targets
        }
        stored_points = _fund_service().read_fund_price_points_range(
            processed_dir,
            normalized,
            start_date=min(lower_bounds.values()),
            end_date=max(target["target_date"] for target in targets),
        )
        for target in targets:
            month = str(target["month"])
            target_date = target["target_date"]
            lower_bound = lower_bounds[month]
            best: Optional[Tuple[date, Dict[str, Any]]] = None
            for point in stored_points:
                point_date = _seam("_fund_point_date")(point)
                if point_date is None or point_date < lower_bound or point_date > target_date:
                    continue
                if not _seam("_has_overview_metrics")(point):
                    continue
                if best is None or point_date > best[0]:
                    best = (point_date, point)
            if best is not None:
                stored_by_month[month] = best[1]

    for target in targets:
        month = str(target["month"])
        target_date = target["target_date"]
        stored_match = stored_by_month.get(month)
        if stored_match is not None:
            rows.append(stored_match)
            fetched_months[month] = str(stored_match.get("date") or "")
            continue
        lower_bound = max(target["month_start"], target_date - timedelta(days=max(0, _fund_service().FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS)))
        current = target_date
        matched = False
        while current >= lower_bound:
            if current.weekday() >= 5:
                current -= timedelta(days=1)
                continue
            try:
                if processed_dir is not None:
                    snapshot_rows, _cache_hit = _fund_service()._cached_daily_funds_snapshot(processed_dir, client, current)
                else:
                    snapshot_rows = client.fetch_daily_funds_snapshot(current)
            except _fund_service().TefasUpstreamError as exc:
                warnings.append(f"tefasfon_funds overview metric snapshot failed for {current.isoformat()}: {exc}")
                break
            match = next(
                (
                    row
                    for row in snapshot_rows
                    if _fund_service().normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")) == normalized
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
            str(_fund_service()._fund_prices_db_path(processed_dir)),
            namespace,
            _fund_service().normalize_fund_code(fund_code),
            start_date.isoformat(),
            end_date.isoformat(),
        ]
    )


def _recent_auto_fetch_failure(key: str) -> Optional[str]:
    with _fund_service()._AUTO_FETCH_LOCK:
        cached = _fund_service()._AUTO_FETCH_NEGATIVE_CACHE.get(key)
        if cached:
            expires_at = cached.get("expires_at")
            if isinstance(expires_at, datetime) and expires_at > _fund_service()._utc_now():
                return str(cached.get("error") or "recent fund history failure")
            _fund_service()._AUTO_FETCH_NEGATIVE_CACHE.pop(key, None)
    cache_get_dict = _fund_service()._cache_get_dict
    if cache_get_dict is None:
        return None
    try:
        shared = cache_get_dict(f"api:funds:auto-fetch-negative:{key}:v1")
    except Exception:
        shared = None
    if not isinstance(shared, dict):
        return None
    expires_at_epoch = shared.get("expires_at_epoch")
    try:
        expires_at = datetime.fromtimestamp(float(expires_at_epoch), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    if expires_at <= _fund_service()._utc_now():
        return None
    error = str(shared.get("error") or "recent fund history failure")
    with _fund_service()._AUTO_FETCH_LOCK:
        _fund_service()._AUTO_FETCH_NEGATIVE_CACHE[key] = {"error": error, "expires_at": expires_at}
    return error


def _remember_auto_fetch_failure(key: str, error: str) -> None:
    ttl = max(1, _fund_service().FUNDS_AUTO_FETCH_NEGATIVE_TTL_SECONDS)
    expires_at = _fund_service()._utc_now() + timedelta(seconds=ttl)
    with _fund_service()._AUTO_FETCH_LOCK:
        _fund_service()._AUTO_FETCH_NEGATIVE_CACHE[key] = {
            "error": error,
            "expires_at": expires_at,
        }
    cache_set_json = _fund_service()._cache_set_json
    if cache_set_json is not None:
        try:
            cache_set_json(
                f"api:funds:auto-fetch-negative:{key}:v1",
                {"error": error, "expires_at_epoch": expires_at.timestamp()},
                ttl_seconds=ttl,
            )
        except Exception:
            pass


def _auto_refresh_fund_performance(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: date,
    end_date: date,
    write_history_cache: bool = True,
    prefer_fast_long_range: bool = False,
) -> List[str]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    key = _seam("_auto_fetch_key")(processed_dir, normalized, start_date, end_date)
    recent_failure = _seam("_recent_auto_fetch_failure")(key)
    if recent_failure:
        return [f"fund history auto fetch skipped after recent failure: {recent_failure}"]

    owner = False
    with _fund_service()._AUTO_FETCH_LOCK:
        event = _fund_service()._AUTO_FETCH_IN_FLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _fund_service()._AUTO_FETCH_IN_FLIGHT[key] = event
            owner = True

    if not owner:
        waited = event.wait(timeout=max(1.0, _fund_service().FINTABLES_TIMEOUT_SECONDS + 5.0))
        if not waited:
            return ["fund history auto fetch already in progress"]
        recent_failure = _seam("_recent_auto_fetch_failure")(key)
        if recent_failure:
            return [f"fund history auto fetch skipped after recent failure: {recent_failure}"]
        return []

    try:
        payload = _seam("refresh_fund_performance")(
            processed_dir,
            normalized,
            start_date=start_date,
            end_date=end_date,
            write_history_cache=write_history_cache,
            prefer_fast_long_range=prefer_fast_long_range,
        )
        if not payload.get("points"):
            error = "fund history returned no valid points"
            _seam("_remember_auto_fetch_failure")(key, error)
            return [f"fund history auto fetch failed: {error}"]
        return []
    except _fund_service().FundUpstreamError as exc:
        error = str(exc)
        _seam("_remember_auto_fetch_failure")(key, error)
        return [f"fund history auto fetch failed: {error}"]
    finally:
        with _fund_service()._AUTO_FETCH_LOCK:
            current = _fund_service()._AUTO_FETCH_IN_FLIGHT.get(key)
            if current is event:
                _fund_service()._AUTO_FETCH_IN_FLIGHT.pop(key, None)
            event.set()


def _auto_refresh_fund_overview_metrics(
    processed_dir: Path,
    fund_code: str,
    *,
    points: List[Dict[str, Any]],
    end_date: date,
    targets: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    missing_targets = _seam("_missing_overview_metric_targets")(points, end_date=end_date, targets=targets)
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

    client = _fund_service().TefasFonClient()
    if not hasattr(client, "fetch_daily_funds_snapshot"):
        return metadata

    key = _seam("_auto_fetch_key")(
        processed_dir,
        normalized,
        missing_targets[0]["target_date"],
        missing_targets[-1]["target_date"],
        namespace="overview-metrics:" + ",".join(str(target["month"]) for target in missing_targets),
    )
    recent_failure = _seam("_recent_auto_fetch_failure")(key)
    if recent_failure:
        metadata["warnings"] = [f"fund overview metric backfill skipped after recent failure: {recent_failure}"]
        metadata["warning_count"] = 1
        metadata["skipped_recent_failure"] = True
        return metadata

    owner = False
    with _fund_service()._AUTO_FETCH_LOCK:
        event = _fund_service()._AUTO_FETCH_IN_FLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _fund_service()._AUTO_FETCH_IN_FLIGHT[key] = event
            owner = True

    if not owner:
        waited = event.wait(timeout=max(1.0, _fund_service().TEFAS_TIMEOUT_SECONDS + 5.0))
        if not waited:
            metadata["warnings"] = ["fund overview metric backfill already in progress"]
            metadata["warning_count"] = 1
            return metadata
        recent_failure = _seam("_recent_auto_fetch_failure")(key)
        if recent_failure:
            metadata["warnings"] = [f"fund overview metric backfill skipped after recent failure: {recent_failure}"]
            metadata["warning_count"] = 1
        return metadata

    try:
        metadata["attempted"] = True
        rows, fetched_months, warnings = _seam("_fetch_fund_overview_metric_rows")(
            normalized,
            missing_targets,
            client=client,
            processed_dir=processed_dir,
        )
        storage_result: Dict[str, Any] = {
            "upserted_count": 0,
            "skipped_count": 0,
            "warnings": [],
        }
        if rows:
            storage_result = _fund_service().upsert_fund_price_points(
                processed_dir,
                rows,
                source=_fund_service().TEFASFON_FUNDS_SOURCE,
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
            _seam("_remember_auto_fetch_failure")(key, error)
            metadata["warnings"] = all_warnings + [error]
            metadata["warning_count"] = len(metadata["warnings"])
        else:
            # The TEFAS overview metric snapshot only fills the most recent few
            # months, older months stay permanently empty. Without a positive
            # TTL we would re-trigger this expensive backfill on every request
            # that asks for a long range. Remember the success so we avoid the
            # round-trip until the TTL expires.
            _seam("_remember_auto_fetch_failure")(
                key,
                "overview metric backfill recently completed",
            )
        return metadata
    except _fund_service().FundUpstreamError as exc:
        error = str(exc)
        _seam("_remember_auto_fetch_failure")(key, error)
        metadata["attempted"] = True
        metadata["warnings"] = [f"fund overview metric backfill failed: {error}"]
        metadata["warning_count"] = 1
        return metadata
    finally:
        with _fund_service()._AUTO_FETCH_LOCK:
            current = _fund_service()._AUTO_FETCH_IN_FLIGHT.get(key)
            if current is event:
                _fund_service()._AUTO_FETCH_IN_FLIGHT.pop(key, None)
            event.set()


def _auto_refresh_fund_recent_details(
    processed_dir: Path,
    fund_code: str,
    *,
    points: List[Dict[str, Any]],
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    window = _seam("_recent_detail_window")(points, start_date=start_date, end_date=end_date)
    window_start, window_end = window if window else (None, None)
    missing_count = (
        _seam("_recent_detail_missing_count")(points, start_date=window_start, end_date=window_end)
        if window_start is not None and window_end is not None
        else 0
    )
    metadata: Dict[str, Any] = {
        "attempted": False,
        "start_date": window_start.isoformat() if window_start else None,
        "end_date": window_end.isoformat() if window_end else None,
        "lookback_days": max(1, _fund_service().FUNDS_RECENT_DETAIL_LOOKBACK_DAYS),
        "missing_count": missing_count,
        "fetched_point_count": 0,
        "upserted_count": 0,
        "skipped_count": 0,
        "warning_count": 0,
        "warnings": [],
    }
    if not normalized or window_start is None or window_end is None or missing_count <= 0:
        return metadata

    client = _fund_service().TefasFonClient()
    if not hasattr(client, "fetch_history"):
        metadata["skipped_unavailable"] = True
        return metadata

    key = _seam("_auto_fetch_key")(
        processed_dir,
        normalized,
        window_start,
        window_end,
        namespace="recent-details",
    )
    recent_failure = _seam("_recent_auto_fetch_failure")(key)
    if recent_failure:
        metadata["warnings"] = [f"fund recent detail backfill skipped after recent failure: {recent_failure}"]
        metadata["warning_count"] = 1
        metadata["skipped_recent_failure"] = True
        return metadata

    owner = False
    with _fund_service()._AUTO_FETCH_LOCK:
        event = _fund_service()._AUTO_FETCH_IN_FLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _fund_service()._AUTO_FETCH_IN_FLIGHT[key] = event
            owner = True

    if not owner:
        waited = event.wait(timeout=max(1.0, _fund_service().TEFAS_TIMEOUT_SECONDS + 5.0))
        if not waited:
            metadata["warnings"] = ["fund recent detail backfill already in progress"]
            metadata["warning_count"] = 1
            return metadata
        recent_failure = _seam("_recent_auto_fetch_failure")(key)
        if recent_failure:
            metadata["warnings"] = [f"fund recent detail backfill skipped after recent failure: {recent_failure}"]
            metadata["warning_count"] = 1
        return metadata

    try:
        metadata["attempted"] = True
        missing_dates = {
            point["date"]
            for point in points
            if isinstance(point.get("date"), str)
            and (window_start.isoformat() <= str(point.get("date")) <= window_end.isoformat())
            and _fund_service()._normalize_price_source(str(point.get("source") or "")) != _fund_service().TEFASFON_FUNDS_SOURCE
            and not _seam("_has_overview_metrics")(point)
        }
        rows, warnings = _seam("_fetch_recent_detail_rows")(
            normalized,
            start_date=window_start,
            end_date=window_end,
            client=client,
            processed_dir=processed_dir,
            missing_dates=missing_dates,
        )
        storage_result: Dict[str, Any] = {
            "upserted_count": 0,
            "skipped_count": 0,
            "warnings": [],
        }
        if rows:
            storage_result = _fund_service().upsert_fund_price_points(
                processed_dir,
                rows,
                source=_fund_service().TEFASFON_FUNDS_SOURCE,
                fallback_code=normalized,
            )
        storage_warnings = [
            str(item.get("warning") or "invalid_price_row")
            for item in list(storage_result.get("warnings") or [])
        ]
        all_warnings = warnings + storage_warnings
        metadata.update(
            {
                "fetched_point_count": len(rows),
                "upserted_count": storage_result.get("upserted_count", 0),
                "skipped_count": storage_result.get("skipped_count", 0),
                "warning_count": len(all_warnings),
                "warnings": all_warnings,
            }
        )
        if not rows:
            _seam("_remember_auto_fetch_failure")(key, "recent detail snapshots returned no matching fund rows")
        return metadata
    except _fund_service().FundUpstreamError as exc:
        error = str(exc)
        _seam("_remember_auto_fetch_failure")(key, error)
        metadata["attempted"] = True
        metadata["warnings"] = [f"fund recent detail backfill failed: {error}"]
        metadata["warning_count"] = 1
        return metadata
    finally:
        with _fund_service()._AUTO_FETCH_LOCK:
            current = _fund_service()._AUTO_FETCH_IN_FLIGHT.get(key)
            if current is event:
                _fund_service()._AUTO_FETCH_IN_FLIGHT.pop(key, None)
            event.set()


def _long_range_metric_targets(start_date: date, end_date: date) -> List[Dict[str, Any]]:
    return [
        target
        for target in _seam("_overview_metric_targets")(end_date)
        if target["target_date"] >= start_date and target["month_start"] <= end_date
    ]


def _fetch_fast_long_fund_history(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: date,
    end_date: date,
    client: TefasFonClient,
) -> Tuple[List[Dict[str, Any]], List[str], bool, Optional[str]]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    points: List[Dict[str, Any]] = []
    warnings: List[str] = []
    fallback_used = False
    fallback_reason: Optional[str] = None

    # Fintables is preferred for long chart ranges, but its response can be a
    # recent partial slice. Validate coverage before accepting it as the full
    # history; otherwise use TEFAS to recover older inception/history points.
    try:
        price_points = _fund_service().fetch_fintables_udf_history(normalized, start_date, end_date)
    except _fund_service().FintablesUpstreamError as exc:
        price_points = []
        warnings.append(f"fintables_udf_history fast bootstrap failed: {exc}")
    valid_fintables_rows = _seam("_valid_performance_points")(price_points, normalized)
    confirmed_tefas_tail, tail_warnings = _seam("_fetch_confirmed_tefas_tail")(
        normalized,
        valid_fintables_rows,
        end_date=end_date,
    )
    warnings.extend(tail_warnings)
    fintables_rows_with_tail = valid_fintables_rows + confirmed_tefas_tail
    if confirmed_tefas_tail:
        fallback_used = True
        fallback_reason = "fintables_udf_history_confirmed_tail_tefas_fallback"
    fintables_range_complete = bool(
        fintables_rows_with_tail
        and _seam("_history_is_usable_daily_range")(
            fintables_rows_with_tail,
            start_date=start_date,
            end_date=end_date,
        )
    )
    if fintables_range_complete:
        # Preserve the existing recent-metadata enrichment for price-only UDF
        # rows when the Fintables response really covers the requested range.
        points.extend(fintables_rows_with_tail)
    else:
        try:
            direct_rows = _fund_service().TefasClient().fetch_fund_history(
                fund_codes=[normalized],
                start_date=start_date,
                end_date=end_date,
            )
        except _fund_service().TefasUpstreamError as exc:
            direct_rows = []
            warnings.append(f"tefas_fund_list daily history bootstrap failed: {exc}")
        valid_direct_rows = _seam("_valid_performance_points")(direct_rows, normalized)
        if _seam("_history_is_usable_daily_range")(
            valid_direct_rows,
            start_date=start_date,
            end_date=end_date,
        ):
            fallback_used = True
            fallback_reason = (
                "fintables_udf_history_partial_tefas_fallback"
                if valid_fintables_rows
                else "fintables_udf_history_unavailable_tefas_fallback"
            )
            return valid_direct_rows, warnings, fallback_used, fallback_reason
        if valid_direct_rows:
            fallback_used = True
            fallback_reason = (
                "fintables_udf_history_partial_tefas_fallback"
                if valid_fintables_rows
                else "fintables_udf_history_unavailable_tefas_fallback"
            )
        points.extend(fintables_rows_with_tail)
        points.extend(valid_direct_rows)

    recent_start = max(start_date, end_date - timedelta(days=max(1, _fund_service().FUNDS_RECENT_DETAIL_LOOKBACK_DAYS) - 1))
    recent_rows, recent_warnings = _seam("_fetch_recent_detail_rows")(
        normalized,
        start_date=recent_start,
        end_date=end_date,
        client=client,
        processed_dir=processed_dir,
    )
    points.extend(recent_rows)
    warnings.extend(recent_warnings)

    metric_targets = _seam("_long_range_metric_targets")(start_date, end_date)
    if valid_fintables_rows:
        fintables_months = {str(point.get("date"))[:7] for point in valid_fintables_rows}
        metric_targets = [
            target for target in metric_targets
            if str(target.get("month")) in fintables_months
        ]
    metric_rows: List[Dict[str, Any]] = []
    metric_warnings: List[str] = []
    if metric_targets and hasattr(client, "fetch_daily_funds_snapshot"):
        metric_rows, _fetched_months, metric_warnings = _seam("_fetch_fund_overview_metric_rows")(
            normalized,
            metric_targets,
            client=client,
            processed_dir=processed_dir,
        )
    points.extend(metric_rows)
    warnings.extend(metric_warnings)

    if not points:
        warnings.append("fast long range bootstrap returned no points")
    return points, warnings, fallback_used, fallback_reason


def refresh_fund_performance(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: date,
    end_date: date,
    write_history_cache: bool = True,
    prefer_fast_long_range: bool = False,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    warnings: List[str] = []
    points: List[Dict[str, Any]] = []
    source_used = _fund_service().FINTABLES_UDF_HISTORY_SOURCE
    fallback_used = False
    fallback_reason: Optional[str] = None
    tefas_failure_reason: Optional[str] = None
    client = _fund_service().TefasFonClient()
    if prefer_fast_long_range and (end_date - start_date).days > _fund_service().FUNDS_FAST_LONG_RANGE_DAYS:
        points, fast_warnings, fallback_used, fallback_reason = _seam("_fetch_fast_long_fund_history")(
            processed_dir,
            normalized,
            start_date=start_date,
            end_date=end_date,
            client=client,
        )
        warnings.extend(fast_warnings)
        if not _seam("_valid_performance_points")(points, normalized):
            points = []
    else:
        try:
            fintables_points = _fund_service().fetch_fintables_udf_history(normalized, start_date, end_date)
            points = _seam("_valid_performance_points")(fintables_points, normalized)
            if not points:
                warnings.append("fintables_udf_history returned no valid points")
        except _fund_service().FintablesUpstreamError as exc:
            warnings.append(f"fintables_udf_history failed: {exc}")
            points = []

        confirmed_tefas_tail, tail_warnings = _seam("_fetch_confirmed_tefas_tail")(
            normalized,
            points,
            end_date=end_date,
        )
        warnings.extend(tail_warnings)
        if confirmed_tefas_tail:
            points.extend(confirmed_tefas_tail)
            fallback_used = True
            fallback_reason = "fintables_udf_history_confirmed_tail_tefas_fallback"

        if points and not _seam("_history_is_usable_daily_range")(
            points,
            start_date=start_date,
            end_date=end_date,
        ):
            try:
                tefas_points = client.fetch_history(normalized, start_date, end_date)
                valid_tefas_points = _seam("_valid_performance_points")(tefas_points, normalized)
                if _seam("_history_is_usable_daily_range")(
                    valid_tefas_points,
                    start_date=start_date,
                    end_date=end_date,
                ):
                    points = valid_tefas_points
                else:
                    points.extend(valid_tefas_points)
                if valid_tefas_points:
                    fallback_used = True
                    fallback_reason = "fintables_udf_history_partial_tefas_fallback"
            except _fund_service().TefasUpstreamError as exc:
                warnings.append(f"tefasfon_funds partial-history fallback failed: {exc}")

        if not points:
            fallback_used = True
            fallback_reason = "fintables_udf_history_unavailable_tefas_fallback"
            source_used = _fund_service().TEFASFON_FUNDS_SOURCE
            try:
                points = client.fetch_history(normalized, start_date, end_date)
                if not _seam("_valid_performance_points")(points, normalized):
                    tefas_failure_reason = "tefasfon_funds returned no valid points"
                    warnings.append(tefas_failure_reason)
                    points = []
            except _fund_service().TefasUpstreamError as exc:
                tefas_failure_reason = f"tefasfon_funds failed: {exc}"
                warnings.append(tefas_failure_reason)
                points = []

    if not points:
        raise _fund_service().FundUpstreamError("; ".join(warnings) or "fund history returned no valid points")

    source_used = _seam("_dominant_price_source")(points) or source_used

    storage_result = _fund_service().upsert_fund_price_points(
        processed_dir,
        points,
        source=source_used,
        fallback_code=normalized,
    )
    merged_points = _fund_service()._read_preferred_fund_price_points(
        processed_dir,
        normalized,
        start_date=start_date,
        end_date=end_date,
    )
    storage_warnings = [
        str(item.get("warning") or "invalid_price_row")
        for item in list(storage_result.get("warnings") or [])
    ]
    payload = _seam("_fund_performance_payload_from_points")(
        processed_dir,
        normalized,
        merged_points,
        start_date=start_date,
        end_date=end_date,
        fetched_at=str(storage_result.get("fetched_at") or _fund_service()._utc_now_iso()),
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
    if write_history_cache:
        history_path = _fund_service()._history_path(processed_dir, normalized)
        if _fund_service().database_enabled():
            # Full price series already reside in `fund_prices`; store full payload locally
            # and keep only span/parse metadata in `ragfin_json_cache` to minimize Postgres egress.
            _fund_service()._write_local_json(history_path, payload)
            compact_history_meta = {
                key: payload[key]
                for key in (
                    "fund_code",
                    "status",
                    "source",
                    "source_url",
                    "as_of",
                    "fetched_at",
                    "stale",
                    "source_metadata",
                )
                if key in payload
            }
            _fund_service().write_json_cache(history_path, compact_history_meta)
        else:
            _fund_service()._write_json(history_path, payload)
    return payload


def get_fund_performance_payload(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    allow_upstream_fallback: bool = False,
    auto_refresh: bool = True,
    history_job: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    full_history_requested = start_date is None and end_date is None
    effective_end = end_date or _fund_service().date.today()
    effective_start = start_date or (
        _fund_service()._fund_full_history_start_date()
        if full_history_requested
        else effective_end - timedelta(days=max(1, _fund_service().FUNDS_AUTO_FETCH_LOOKBACK_DAYS))
    )
    query_start = None if full_history_requested else effective_start
    query_end = end_date or effective_end
    auto_warnings: List[str] = []
    auto_fetch_attempted = False

    points = _fund_service()._read_preferred_fund_price_points(
        processed_dir,
        normalized,
        start_date=query_start,
        end_date=query_end,
    )
    if auto_refresh:
        full_history_cache_hit = (
            not full_history_requested
            or _seam("_history_cache_covers_requested_span")(
                processed_dir,
                normalized,
                start_date=effective_start,
                end_date=effective_end,
            )
        )
        has_requested_coverage = (
            full_history_cache_hit
            if full_history_requested
            else _seam("_has_requested_price_coverage")(
                points,
                start_date=query_start,
                end_date=effective_end,
            )
        )
    else:
        has_requested_coverage = True
    if auto_refresh and not has_requested_coverage:
        auto_fetch_attempted = True
        auto_warnings = _seam("_auto_refresh_fund_performance")(
            processed_dir,
            normalized,
            start_date=effective_start,
            end_date=effective_end,
            prefer_fast_long_range=not full_history_requested,
        )
        points = _fund_service()._read_preferred_fund_price_points(
            processed_dir,
            normalized,
            start_date=query_start,
            end_date=query_end,
        )
    elif auto_refresh and _seam("_needs_recent_tail_refresh")(points, end_date=effective_end):
        latest_point_date = _seam("_latest_fund_point_date")(points)
        if latest_point_date is not None:
            auto_fetch_attempted = True
            tail_start = latest_point_date + timedelta(days=1)
            auto_warnings = _seam("_auto_refresh_fund_performance")(
                processed_dir,
                normalized,
                start_date=tail_start,
                end_date=_seam("_recent_tail_refresh_target")(effective_end),
                write_history_cache=False,
            )
            points = _fund_service()._read_preferred_fund_price_points(
                processed_dir,
                normalized,
                start_date=query_start,
                end_date=query_end,
            )

    recent_detail_backfill: Optional[Dict[str, Any]] = None
    recent_detail_backfill_attempted = False
    if auto_refresh and points:
        recent_detail_backfill = _seam("_auto_refresh_fund_recent_details")(
            processed_dir,
            normalized,
            points=points,
            start_date=effective_start,
            end_date=effective_end,
        )
        recent_detail_backfill_attempted = bool(recent_detail_backfill.get("attempted"))
        if recent_detail_backfill_attempted:
            points = _fund_service()._read_preferred_fund_price_points(
                processed_dir,
                normalized,
                start_date=query_start,
                end_date=query_end,
            )

    overview_metric_backfill: Optional[Dict[str, Any]] = None
    overview_metric_backfill_attempted = False
    if auto_refresh and points and not (auto_fetch_attempted and not full_history_requested):
        overview_targets = (
            None
            if full_history_requested
            else _seam("_long_range_metric_targets")(effective_start, effective_end)
        )
        overview_metric_backfill = _seam("_auto_refresh_fund_overview_metrics")(
            processed_dir,
            normalized,
            points=points,
            end_date=effective_end,
            targets=overview_targets,
        )
        overview_metric_backfill_attempted = bool(overview_metric_backfill.get("attempted"))
        if overview_metric_backfill_attempted:
            points = _fund_service()._read_preferred_fund_price_points(
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
            "source_url": str(_fund_service()._fund_prices_db_path(processed_dir)),
            "as_of": None,
            "fetched_at": None,
            "stale": True,
            "source_metadata": {
                "source": "sqlite",
                "source_url": str(_fund_service()._fund_prices_db_path(processed_dir)),
                "db_path": str(_fund_service()._fund_prices_db_path(processed_dir)),
                "fetched_at": None,
                "as_of": None,
                "cache_hit": False,
                "stale": True,
                "parse_status": "unavailable",
                "warnings": warnings,
                "warning": warnings[0] if warnings else None,
                "history_source_used": None,
                "history_source_policy": _fund_service().FUND_HISTORY_SOURCE_POLICY,
                "source_policy": _fund_service().FUND_HISTORY_SOURCE_POLICY,
                "primary_source": "fintables",
                "tefasfon_adapter_version": _fund_service()._tefasfon_adapter_version(),
                "fallback_used": any("fintables_udf_history" in warning for warning in warnings),
                "fallback_reason": "tefasfon_and_fintables_unavailable" if any("fintables_udf_history" in warning for warning in warnings) else None,
                "final_points_count": 0,
                "date_min": None,
                "date_max": None,
                "backfill_used": auto_fetch_attempted,
                "full_history_requested": full_history_requested,
                "requested_start_date": effective_start.isoformat(),
                "requested_end_date": effective_end.isoformat(),
                "auto_fetch_attempted": auto_fetch_attempted,
                "recent_detail_backfill": recent_detail_backfill,
                "overview_metric_backfill": overview_metric_backfill,
                "coverage_state": "unavailable",
                "resolution": "unknown",
                "requested_resolution": "daily",
                "available_start_date": None,
                "available_end_date": None,
                "daily_upgrade_state": "failed" if history_job and history_job.get("status") == "failed" else "idle",
                "history_job": history_job,
            },
        }
    recent_detail_warnings = list((recent_detail_backfill or {}).get("warnings") or [])
    effective_backfill_used = (
        auto_fetch_attempted
        or recent_detail_backfill_attempted
        or overview_metric_backfill_attempted
    )
    return _seam("_fund_performance_payload_from_points")(
        processed_dir,
        normalized,
        points,
        start_date=effective_start,
        end_date=effective_end,
        cache_hit=not effective_backfill_used,
        stale=False if effective_backfill_used and not auto_warnings and not recent_detail_warnings else None,
        parse_status=(
            "ok_fund_history_auto"
            if auto_fetch_attempted and not auto_warnings
            else "ok_fund_history_recent_detail_auto"
            if recent_detail_backfill_attempted and not recent_detail_warnings
            else "ok_fund_history_overview_metrics_auto"
            if overview_metric_backfill_attempted and not auto_warnings
            else None
        ),
        warnings=auto_warnings + recent_detail_warnings,
        backfill_used=effective_backfill_used,
        fallback_used=False,
        fallback_reason=None,
        recent_detail_backfill=recent_detail_backfill,
        overview_metric_backfill=overview_metric_backfill,
        full_history_requested=full_history_requested,
        history_job=history_job,
    )


def refresh_fund_allocations(
    processed_dir: Path,
    fund_code: str,
    *,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    fetched_at = _fund_service()._utc_now_iso()
    effective_as_of = as_of or _fund_service().date.today()
    warnings: List[str] = []
    try:
        rows, warnings = _fund_service().TefasFonClient().fetch_latest_portfolio(
            normalized,
            as_of=effective_as_of,
            lookback_days=10,
        )
    except _fund_service().TefasUpstreamError as exc:
        rows = []
        warnings = [f"tefasfon_portfolio failed: {exc}"]

    allocations = [
        allocation
        for row in rows
        for allocation in _seam("_normalize_allocation_row")(row, fallback_code=normalized)
    ]
    if allocations:
        report_dates = [item.get("report_date") for item in allocations if item.get("report_date")]
        payload = {
            "fund_code": normalized,
            "status": "ok",
            "allocations": allocations,
            "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
            "stale": False,
            "source_metadata": {
                "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
                "source_url": _fund_service().TEFASFON_SOURCE_URL,
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
        _fund_service()._write_json(_fund_service()._allocations_path(processed_dir, normalized), payload)
        return payload

    cached = _fund_service()._read_json(_fund_service()._allocations_path(processed_dir, normalized))
    if cached and _fund_service()._public_price_source(str(cached.get("source") or "")) != "legacy_cache":
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
        "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
        "stale": True,
        "source_metadata": {
            "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
            "source_url": _fund_service().TEFASFON_SOURCE_URL,
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
    allow_daily_fallback: bool = True,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    bounded_lookback = max(1, min(365, int(lookback_days)))
    fetched_at = _fund_service()._utc_now_iso()
    effective_as_of = as_of or _fund_service().date.today()
    start_date = effective_as_of - timedelta(days=bounded_lookback - 1)
    warnings: List[str] = []
    client = _fund_service().TefasFonClient()
    rate_limited = False
    try:
        rows = client.fetch_portfolio(
            fund_code=normalized,
            start_date=start_date,
            end_date=effective_as_of,
        )
    except _fund_service().TefasUpstreamError as exc:
        rows = []
        warnings = [f"tefasfon_portfolio_history failed: {exc}"]
        rate_limited = _fund_service()._is_tefas_rate_limit(exc)
    if not rows and not rate_limited and allow_daily_fallback:
        daily_dates: List[date] = []
        target_date = start_date
        while target_date <= effective_as_of:
            if target_date.weekday() < 5:
                daily_dates.append(target_date)
            target_date += timedelta(days=1)
        max_daily_requests = max(0, _fund_service().FUNDS_ALLOCATION_HISTORY_DAILY_FALLBACK_MAX_REQUESTS)
        if len(daily_dates) > max_daily_requests:
            warnings.append(
                "tefasfon_portfolio_history range query was empty; "
                f"daily fallback skipped because it needs {len(daily_dates)} requests "
                f"(budget: {max_daily_requests})"
            )
            daily_dates = []
        daily_rows: List[Dict[str, Any]] = []
        daily_warnings: List[str] = []

        def fetch_daily_portfolio(day: date) -> Tuple[date, List[Dict[str, Any]], Optional[str]]:
            try:
                return day, client.fetch_portfolio(fund_code=normalized, start_date=day, end_date=day), None
            except _fund_service().TefasUpstreamError as exc:
                return day, [], f"tefasfon_portfolio_history failed for {day.isoformat()}: {exc}"

        if daily_dates:
            max_workers = max(
                1,
                min(
                    _fund_service().FUNDS_ALLOCATION_HISTORY_DAILY_MAX_WORKERS,
                    _fund_service().FUNDS_DETAIL_MAX_WORKERS,
                    len(daily_dates),
                ),
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="tefas-portfolio-history",
            ) as executor:
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
        for allocation in _seam("_normalize_allocation_row")(row, fallback_code=normalized):
            report_date = _fund_service()._fund_date(allocation.get("report_date"))
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
        "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
        "stale": False,
        "source_metadata": {
            "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
            "source_url": _fund_service().TEFASFON_SOURCE_URL,
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
        _fund_service()._write_json(_fund_service()._allocations_history_path(processed_dir, normalized, bounded_lookback), payload)
    if history:
        _seam("_promote_latest_allocation_history")(processed_dir, normalized, payload)
    return payload


def get_fund_allocations_history_payload(
    processed_dir: Path,
    fund_code: str,
    *,
    lookback_days: int = 30,
    auto_refresh: bool = True,
) -> Dict[str, Any]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    bounded_lookback = max(1, min(365, int(lookback_days)))
    path = _fund_service()._allocations_history_path(processed_dir, normalized, bounded_lookback)
    payload = _fund_service()._read_json(path)
    stale = True
    if payload:
        cache_age = _fund_service()._cache_age_seconds(payload.get("source_metadata", {}).get("fetched_at"))
        history_present = bool(payload.get("history"))
        ttl = _fund_service().FUNDS_ALLOCATION_TTL_SECONDS if history_present else _fund_service().FUNDS_ALLOCATION_EMPTY_TTL_SECONDS
        stale = (cache_age or (ttl + 1)) > ttl
        if not stale:
            meta = dict(payload.get("source_metadata") or {})
            meta["cache_hit"] = True
            meta["stale"] = False
            payload = dict(payload)
            payload["stale"] = False
            payload["source_metadata"] = meta
            return payload

    if not auto_refresh:
        if payload:
            meta = dict(payload.get("source_metadata") or {})
            meta["cache_hit"] = True
            meta["stale"] = True
            meta["warning"] = "allocation history refresh is pending"
            payload = dict(payload)
            payload["stale"] = True
            payload["source_metadata"] = meta
            return payload
        return {
            "fund_code": normalized,
            "status": "pending",
            "lookback_days": bounded_lookback,
            "history": [],
            "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
            "stale": True,
            "source_metadata": {
                "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
                "source_url": _fund_service().TEFASFON_SOURCE_URL,
                "fetched_at": None,
                "as_of": None,
                "cache_hit": False,
                "stale": True,
                "parse_status": "pending_allocation_history_refresh",
                "source_policy": "tefasfon_primary",
                "fallback_used": False,
                "warnings": ["allocation history refresh is pending"],
            },
        }

    fresh = _seam("refresh_fund_allocations_history")(processed_dir, normalized, lookback_days=bounded_lookback)
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
    normalized = _fund_service().normalize_fund_code(fund_code)
    path = _fund_service()._allocations_path(processed_dir, normalized)
    payload = _fund_service()._read_json(path)

    payload_stale = True
    if payload and _fund_service()._public_price_source(str(payload.get("source") or "")) != "legacy_cache":
        payload_age = _fund_service()._cache_age_seconds(
            payload.get("source_metadata", {}).get("fetched_at") or payload.get("fetched_at")
        )
        payload_stale = (
            payload_age if payload_age is not None else (_fund_service().FUNDS_ALLOCATION_TTL_SECONDS + 1)
        ) > _fund_service().FUNDS_ALLOCATION_TTL_SECONDS

    if payload_stale:
        # The history endpoint may already have a newer, valid TEFAS report even
        # when the legacy single-snapshot file is stale. Reuse that same latest
        # day so the overview and history screens cannot disagree.
        history_payload = _seam("get_fund_allocations_history_payload")(
            processed_dir,
            normalized,
            lookback_days=30,
            auto_refresh=False,
        )
        promoted = _seam("_promote_latest_allocation_history")(processed_dir, normalized, history_payload)
        if promoted is not None:
            payload = promoted

    if not payload or _fund_service()._public_price_source(str(payload.get("source") or "")) == "legacy_cache":
        return {
            "fund_code": normalized,
            "status": "unavailable",
            "allocations": [],
            "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
            "stale": True,
            "source_metadata": {
                "source": _fund_service().TEFASFON_PORTFOLIO_SOURCE,
                "source_url": _fund_service().TEFASFON_SOURCE_URL,
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
    stale = (_fund_service()._cache_age_seconds(payload.get("source_metadata", {}).get("fetched_at") or payload.get("fetched_at")) or (_fund_service().FUNDS_ALLOCATION_TTL_SECONDS + 1)) > _fund_service().FUNDS_ALLOCATION_TTL_SECONDS
    meta = dict(payload.get("source_metadata") or {})
    meta["cache_hit"] = True
    meta["stale"] = stale
    payload = dict(payload)
    payload["stale"] = stale
    payload["source_metadata"] = meta
    return payload


__all__ = [
    "_normalize_allocation_row",
    "_allocation_payload_report_date",
    "_promote_latest_allocation_history",
    "_return_between",
    "_point_on_or_before",
    "_period_returns",
    "_period_returns_from_raw",
    "_history_coverage_info",
    "_history_internal_gap_warnings",
    "_is_monthly_anchor_transition",
    "_history_resolution",
    "_history_is_usable_daily_range",
    "_fetch_confirmed_tefas_tail",
    "_history_needs_detail_fill",
    "_dedupe_price_points",
    "_dominant_price_source",
    "_summary_from_points",
    "get_funds_payload",
    "get_fund_categories_payload",
    "_fund_reference_metadata",
    "_find_fund_row",
    "_fund_return_rank_value",
    "_fund_category_rankings",
    "get_fund_detail_payload",
    "get_fund_yield_summary_payload",
    "_valid_performance_points",
    "_legacy_history_points",
    "_migrate_legacy_history_to_sqlite",
    "_fund_performance_payload_from_points",
    "_has_requested_price_coverage",
    "_latest_fund_point_date",
    "_recent_tail_refresh_target",
    "_needs_recent_tail_refresh",
    "_history_cache_covers_requested_span",
    "_month_start",
    "_shift_month",
    "_month_end",
    "_month_key",
    "_fund_point_date",
    "_fund_period_stats",
    "_overview_metric_targets",
    "_latest_point_for_month",
    "_has_overview_metrics",
    "_recent_detail_window",
    "_recent_detail_missing_count",
    "_fetch_recent_detail_rows",
    "_missing_overview_metric_targets",
    "_fetch_fund_overview_metric_rows",
    "_auto_fetch_key",
    "_recent_auto_fetch_failure",
    "_remember_auto_fetch_failure",
    "_auto_refresh_fund_performance",
    "_auto_refresh_fund_overview_metrics",
    "_auto_refresh_fund_recent_details",
    "_long_range_metric_targets",
    "_fetch_fast_long_fund_history",
    "refresh_fund_performance",
    "get_fund_performance_payload",
    "refresh_fund_allocations",
    "refresh_fund_allocations_history",
    "get_fund_allocations_history_payload",
    "get_fund_allocations_payload",
]

_ORIGINAL_EXPORTS.update({name: globals()[name] for name in __all__})
