from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.api_runtime import *
from app.etag import conditional_json_response, if_none_match_matches, weak_etag
from app.fund_api_service import *

router = APIRouter()

@router.get("/funds", response_model=Dict[str, Any], responses={304: {"description": "Not Modified"}})
def funds(
    request: Request,
    q: Optional[str] = Query(None, max_length=80),
    fund_type: Optional[str] = Query(None, max_length=32),
    founder: Optional[str] = Query(None, max_length=120),
    manager: Optional[str] = Query(None, max_length=120),
    risk: Optional[str] = Query(None, max_length=32),
    sort: str = Query("fund_code", max_length=32),
    order: str = Query("asc", max_length=8),
) -> Response:
    snapshot, source_revision = _funds_listing_preflight()
    snapshot = _maybe_schedule_fund_snapshot_refresh(snapshot)
    visible_state = {key: value for key, value in snapshot.items() if key != "rows"}
    allowed_sort_values = {
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

    def normalize_filter(value: Optional[str], *, empty_is_unfiltered: bool = False) -> Optional[str]:
        if value is None or value == "":
            return None
        normalized = value.strip().lower()
        return None if empty_is_unfiltered and not normalized else normalized

    revision = {
        "representation": "funds-list-v2",
        "query": {
            "q": normalize_filter(q, empty_is_unfiltered=True),
            "fund_type": normalize_filter(fund_type),
            "founder": normalize_filter(founder),
            "manager": normalize_filter(manager),
            "risk": normalize_filter(risk),
            "sort": sort if sort in allowed_sort_values else "fund_code",
            "order": "desc" if order.lower() == "desc" else "asc",
        },
        "source": source_revision,
        "visible_state": visible_state,
    }
    etag = weak_etag(revision)
    if if_none_match_matches(request.headers.get("if-none-match"), etag):
        logging.getLogger("uvicorn.error").info(
            "fund_listing_etag outcome=not_modified catalog_generation=%s price_generation=%s",
            source_revision.get("catalog_generation"),
            source_revision.get("price_generation"),
        )
        return Response(
            status_code=304,
            headers={"Cache-Control": "private, no-cache", "ETag": etag},
        )

    payload = _funds_listing_payload(
        q=q,
        fund_type=fund_type,
        founder=founder,
        manager=manager,
        risk=risk,
        sort=sort,
        order=order,
        source_revision=source_revision,
        cacheable=(
            not bool(snapshot.get("refresh_pending"))
            and not bool(snapshot.get("stale"))
            and not bool(snapshot.get("degraded"))
            and str(snapshot.get("status") or "") != "pending"
        ),
    )
    # Refresh state is re-evaluated per request and never frozen in the LRU.
    for field in ("refresh_pending", "refresh_job"):
        if field in snapshot:
            payload[field] = snapshot[field]
    return conditional_json_response(request, payload, revision)

@router.get("/funds/search")
def funds_search(q: str = Query("", min_length=0, max_length=80), limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    payload = _funds_search_payload(q=q, limit=limit)
    rows = list(payload.get("rows") or [])[:limit]
    # Trim and clone so each caller gets the right slice without mutating the
    # cached payload referenced by other concurrent requests.
    out = dict(payload)
    out["rows"] = rows
    out["count"] = len(rows)
    return out

@router.get("/funds/categories")
def funds_categories() -> Dict[str, Any]:
    return _funds_categories_payload()

@router.get("/funds/{fund_code}/performance")
def fund_performance(
    fund_code: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    fallback: bool = Query(False),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    return _fund_performance_payload(
        fund_code=fund_code,
        start_iso=start_date.isoformat() if start_date else None,
        end_iso=end_date.isoformat() if end_date else None,
        fallback=fallback,
        refresh=refresh,
    )

@router.get("/funds/{fund_code}/performance/status")
def fund_performance_status(
    fund_code: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    target = _history_request_range(start_date, end_date)
    job = _history_find_existing_job(normalized, target)
    return {
        "fund_code": normalized,
        "status": str(job.get("status") if job else "idle"),
        "history_job": _history_job_public(job),
    }

@router.get("/funds/{fund_code}/yield-summary")
def fund_yield_summary(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import FintablesUpstreamError, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        return _fund_yield_summary_payload(normalized=normalized)
    except FintablesUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@router.get("/funds/{fund_code}/holdings")
def fund_holdings(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    return _fund_holdings_response_payload(normalized=normalized)

@router.get("/funds/{fund_code}/holdings/live")
def fund_holdings_live(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    key = f"api:fund-holdings-live:{normalized}:v1"
    payload, cache_status = _get_or_set_single_flight(
        key,
        ttl_seconds=_FUND_HOLDINGS_LIVE_RESPONSE_CACHE_TTL,
        factory=lambda: _fund_holdings_live_payload(normalized=normalized),
        lock_ttl_seconds=_FUND_HOLDINGS_LIVE_RESPONSE_LOCK_TTL,
        wait_timeout_seconds=_FUND_HOLDINGS_LIVE_RESPONSE_WAIT_TIMEOUT,
    )
    if isinstance(payload, dict):
        return payload
    # Do not make a waiter repeat the live Yahoo/InfoYatirim fan-out. The
    # static response lets the client retain its existing values until the
    # single owner populates the short shared cache.
    static = _fund_holdings_static_payload(normalized=normalized)
    return {
        "fund_code": normalized,
        "status": static.get("status") or "pending",
        "positions": [],
        "portfolio_effect": None,
        "source": "daily_market_enrichment",
        "as_of": None,
        "source_metadata": {
            "source": "daily_market_enrichment",
            "live_cache_status": cache_status,
            "refresh_pending": True,
        },
    }

@router.get("/funds/{fund_code}/allocations")
def fund_allocations(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_allocations_payload, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    payload = get_fund_allocations_payload(CONFIG.paths.processed_dir, normalized)
    if not payload.get("stale"):
        response = dict(payload)
        response["refresh_pending"] = False
        return response

    # Keep serving the last usable snapshot, but use the same background
    # history refresh as the history screen. When it completes, the worker
    # promotes the newest history day into the canonical allocation snapshot.
    normalized = _require_known_fund_code(normalized)
    job = _start_allocation_history_refresh_job(normalized, 30)
    response = dict(payload)
    response["refresh_pending"] = str(job.get("status")) in {"queued", "running", "pending"}
    metadata = dict(response.get("source_metadata") or {})
    metadata["allocation_history_job"] = _allocation_history_job_public(job)
    response["source_metadata"] = metadata
    return response

@router.get("/funds/{fund_code}/allocations/history")
def fund_allocations_history(
    fund_code: str,
    lookback_days: int = Query(30, ge=1, le=365),
) -> Any:
    from app.fund_service import get_fund_allocations_history_payload

    normalized = _require_known_fund_code(fund_code)
    payload = get_fund_allocations_history_payload(
        CONFIG.paths.processed_dir,
        normalized,
        lookback_days=lookback_days,
        auto_refresh=False,
    )
    needs_refresh = bool(payload.get("stale")) or str(payload.get("status")) == "pending"
    if not needs_refresh:
        payload["refresh_pending"] = False
        return payload
    job = _start_allocation_history_refresh_job(normalized, lookback_days)
    response = dict(payload)
    response["refresh_pending"] = str(job.get("status")) in {"queued", "running", "pending"}
    metadata = dict(response.get("source_metadata") or {})
    metadata["allocation_history_job"] = _allocation_history_job_public(job)
    response["source_metadata"] = metadata
    if not response.get("history"):
        return JSONResponse(status_code=202, content=response)
    return response

@router.get("/funds/{fund_code}")
def fund_detail(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        return _fund_detail_payload(normalized=normalized)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}") from exc

@router.post("/admin/funds/refresh-snapshot")
def admin_refresh_funds_snapshot(
    request: Request,
    lookback_days: int = Query(_FUND_REFRESH_MAX_LOOKBACK_DAYS, ge=1, le=_FUND_REFRESH_MAX_LOOKBACK_DAYS)
) -> Dict[str, Any]:
    from app.fund_service import get_funds_payload

    _require_admin_refresh_access(request)
    # Start the upstream work outside the request path. The current snapshot
    # is returned immediately so a slow TEFAS/WAF response cannot freeze the UI.
    job = _start_fund_refresh_job(lookback_days)
    payload = get_funds_payload(
        CONFIG.paths.processed_dir,
        sort="fund_code",
        order="asc",
        auto_refresh=False,
    )
    payload["refresh_job"] = job
    return payload

@router.get("/admin/funds/refresh-snapshot/status")
def fund_refresh_snapshot_status(
    request: Request,
    job_id: str = Query(..., min_length=8, max_length=64),
) -> Dict[str, Any]:
    _require_admin_refresh_access(request)
    job = _get_fund_refresh_job(job_id.strip())
    if job is None:
        raise HTTPException(status_code=404, detail="Fon yenileme işi bulunamadı veya süresi doldu.")
    return {"refresh_job": job}

@router.post("/admin/funds/collect-prices")
def admin_collect_fund_prices(
    request: Request,
    lookback_days: int = Query(10, ge=1, le=45),
    as_of: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import collect_daily_fund_prices

    _require_admin_refresh_access(request)
    return collect_daily_fund_prices(
        CONFIG.paths.processed_dir,
        as_of=as_of,
        lookback_days=lookback_days,
    )

@router.post("/admin/funds/{fund_code}/refresh-performance")
def admin_refresh_fund_performance(
    request: Request,
    fund_code: str,
    start_date: date = Query(...),
    end_date: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import FundUpstreamError, refresh_fund_performance, normalize_fund_code

    _require_admin_refresh_access(request)
    normalized = normalize_fund_code(fund_code)
    effective_end_date = end_date or date.today()
    if start_date > effective_end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    if (effective_end_date - start_date).days > _ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"start_date en fazla {_ADMIN_FUND_PERFORMANCE_MAX_LOOKBACK_DAYS} gun geriye gidebilir",
        )
    _require_known_fund_code(normalized)
    try:
        result = refresh_fund_performance(
            CONFIG.paths.processed_dir,
            normalized,
            start_date=start_date,
            end_date=effective_end_date,
        )
    except FundUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    cache_invalidation_status = _invalidate_single_fund_response_cache(normalized)
    result = dict(result)
    result["cache_invalidation_status"] = cache_invalidation_status
    return result

@router.post("/admin/funds/{fund_code}/refresh-allocations")
def admin_refresh_fund_allocations(
    request: Request,
    fund_code: str,
    as_of: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import FundUpstreamError, normalize_fund_code, refresh_fund_allocations

    _require_admin_refresh_access(request)
    normalized = normalize_fund_code(fund_code)
    _require_known_fund_code(normalized)
    try:
        result = refresh_fund_allocations(CONFIG.paths.processed_dir, normalized, as_of=as_of)
    except FundUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    cache_invalidation_status = _invalidate_single_fund_response_cache(normalized)
    result = dict(result)
    result["cache_invalidation_status"] = cache_invalidation_status
    return result

__all__ = [name for name in globals() if not name.startswith("__")]
