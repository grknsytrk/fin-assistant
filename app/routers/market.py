from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.api_runtime import *
from app.etag import conditional_json_response
from app.market_service import *
from app.kap_api_service import *
from app.kap_service import *

router = APIRouter()

class MarketComparisonHistoryAsset(BaseModel):
    id: Optional[str] = None
    kind: Literal["fund", "stock", "index", "fx"]
    symbol: str = Field(..., min_length=1, max_length=32)
    label: Optional[str] = Field(default=None, max_length=160)

class MarketComparisonHistoryRequest(BaseModel):
    assets: List[MarketComparisonHistoryAsset] = Field(..., min_length=1, max_length=8)
    start_date: date
    end_date: date

@router.get("/market/universe")
def market_universe(index: str = Query("XUTUM"), refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_universe_payload(index_name=index, force_refresh=refresh)

@router.get("/market/stocks/search")
def market_stocks_search(
    q: str = Query("", max_length=80),
    index: str = Query("XUTUM"),
    limit: int = Query(20, ge=1, le=50),
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    normalized_index = _normalize_stock_index(index)
    query = str(q or "").strip().upper()
    payload = _market_universe_payload(index_name=normalized_index)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if query:
        rows = [
            row
            for row in rows
            if query in str(row.get("symbol") or row.get("company") or "").upper()
            or query in str(row.get("name") or "").upper()
        ]
    rows = rows[:limit]
    _log_market_cache_event(
        endpoint="market/stocks/search",
        index=normalized_index,
        cache_status="metadata",
        upstream_called=False,
        stale=False,
        started_at=started_at,
    )
    return {
        "index": normalized_index,
        "query": q,
        "count": len(rows),
        "rows": rows,
        "as_of": payload.get("universe", {}).get("fetched_at") if isinstance(payload.get("universe"), dict) else None,
    }

@router.get("/market/stocks", response_model=Dict[str, Any], responses={304: {"description": "Not Modified"}})
def market_stocks(request: Request, index: str = Query("XUTUM"), refresh: bool = Query(False)) -> Response:
    payload = _market_stocks_payload(index_name=index, force_refresh=refresh)
    source_revision = market_stocks_etag_revision(index_name=index, payload=payload)
    revision = None
    if source_revision is not None:
        revision = {
            "representation": "market-stocks-v1",
            "index": _normalize_stock_index(index),
            "source": source_revision,
        }
    return conditional_json_response(request, payload, revision)

@router.get("/market/screener")
def market_screener(
    index: str = Query("XUTUM"),
    sector: Optional[str] = Query(None),
    min_fk: Optional[float] = Query(None),
    max_fk: Optional[float] = Query(None),
    min_pd_dd: Optional[float] = Query(None),
    max_pd_dd: Optional[float] = Query(None),
    min_fd_favok: Optional[float] = Query(None),
    max_fd_favok: Optional[float] = Query(None),
    max_net_borc_favok: Optional[float] = Query(None),
    min_roe_pct: Optional[float] = Query(None),
    min_favok_margin_pct: Optional[float] = Query(None),
    positive_net_profit: bool = Query(False),
    net_cash_only: bool = Query(False),
    sort_by: str = Query("company"),
    sort_dir: str = Query("asc"),
) -> Dict[str, Any]:
    """Filter the cached live market rows by their KAP/provider fundamentals."""

    normalized_index = _normalize_stock_index(index)
    normalized_sort = str(sort_by or "company").strip().lower()
    normalized_direction = str(sort_dir or "asc").strip().lower()
    allowed_sort = {
        "company", "sector", "fundamental_period", "fk", "pd_dd", "fd_favok",
        "fd_satis", "net_borc_favok", "roe_pct", "free_float_pct", "price",
    }
    if normalized_sort not in allowed_sort:
        raise HTTPException(status_code=400, detail="Desteklenmeyen tarayıcı sıralama alanı.")
    if normalized_direction not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="Sıralama yönü asc veya desc olmalı.")

    market_payload = _market_stocks_payload(index_name=normalized_index)
    rows = [
        dict(row)
        for row in market_payload.get("rows", [])
        if isinstance(row, dict)
    ]
    all_sectors = sorted(
        {str(row.get("sector") or "").strip() for row in rows if str(row.get("sector") or "").strip()},
        key=lambda value: value.casefold(),
    )
    sector_filter = str(sector or "").strip().casefold()
    range_filters = (
        ("fk", min_fk, max_fk),
        ("pd_dd", min_pd_dd, max_pd_dd),
        ("fd_favok", min_fd_favok, max_fd_favok),
        ("net_borc_favok", None, max_net_borc_favok),
        ("roe_pct", min_roe_pct, None),
        ("favok_margin_pct", min_favok_margin_pct, None),
    )
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if sector_filter and str(row.get("sector") or "").strip().casefold() != sector_filter:
            continue
        matched = True
        for key, minimum, maximum in range_filters:
            if minimum is None and maximum is None:
                continue
            value = _as_finite_float(row.get(key))
            if value is None or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                matched = False
                break
        if not matched:
            continue
        if positive_net_profit:
            net_profit = _as_finite_float(row.get("ttm_net_kar"))
            if net_profit is None or net_profit <= 0:
                continue
        if net_cash_only:
            net_debt = _as_finite_float(row.get("net_borc"))
            if net_debt is None or net_debt >= 0:
                continue
        filtered.append(row)

    def sort_value(row: Dict[str, Any]) -> Any:
        value = row.get(normalized_sort)
        if isinstance(value, str):
            return value.casefold()
        return _as_finite_float(value)

    present = [row for row in filtered if sort_value(row) is not None]
    missing = [row for row in filtered if sort_value(row) is None]
    present.sort(key=lambda row: (sort_value(row), str(row.get("company") or "").casefold()), reverse=normalized_direction == "desc")
    missing.sort(key=lambda row: str(row.get("company") or "").casefold())
    filtered = present + missing
    return {
        "index": normalized_index,
        "count": len(filtered),
        "rows": filtered,
        "sectors": all_sectors,
        "as_of": market_payload.get("as_of"),
        "filters": {
            "sector": sector or None,
            "min_fk": min_fk,
            "max_fk": max_fk,
            "min_pd_dd": min_pd_dd,
            "max_pd_dd": max_pd_dd,
            "min_fd_favok": min_fd_favok,
            "max_fd_favok": max_fd_favok,
            "max_net_borc_favok": max_net_borc_favok,
            "min_roe_pct": min_roe_pct,
            "min_favok_margin_pct": min_favok_margin_pct,
            "positive_net_profit": positive_net_profit,
            "net_cash_only": net_cash_only,
            "sort_by": normalized_sort,
            "sort_dir": normalized_direction,
        },
    }

@router.get("/market/stream")
async def market_stream(
    request: Request,
    index: str = Query("XUTUM"),
    once: bool = Query(False),
) -> StreamingResponse:
    normalized_index = _normalize_stock_index(index)

    async def event_generator():
        prev_snapshot: Dict[str, tuple[Any, ...]] = {}
        last_heartbeat = time.monotonic()
        tick = 0
        while True:
            if not once and await request.is_disconnected():
                break
            market_open = _is_bist_market_open()
            try:
                payload = await asyncio.to_thread(_market_stocks_payload, index_name=normalized_index)
                rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
                changed, prev_snapshot = _compute_market_stocks_delta(prev_snapshot, rows)
                if changed or tick == 0:
                    data = {
                        "index": normalized_index,
                        "as_of": payload.get("as_of") or datetime.now(timezone.utc).isoformat(),
                        "market_open": market_open,
                        "count": len(changed),
                        "rows": changed,
                    }
                    yield f"event: stocks_delta\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    last_heartbeat = time.monotonic()
            except Exception as exc:
                LOGGER.debug("market stream tick failed: %s", exc)
            tick += 1
            if once:
                break
            now_mono = time.monotonic()
            if now_mono - last_heartbeat >= 15.0:
                yield ": heartbeat\n\n"
                last_heartbeat = now_mono
            sleep_seconds = 3.0 if market_open else 30.0
            await asyncio.sleep(sleep_seconds)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.get("/market/stocks/cards")
def market_stock_cards(
    symbols: str = Query(""),
    refresh: bool = Query(False),
    tier: str = Query("full"),
) -> Dict[str, Any]:
    return _market_stock_cards_payload(symbols=symbols, force_refresh=refresh, tier=tier)

@router.get("/market/stocks/cards/chart")
def market_stock_card_chart(
    symbol: str = Query(""),
    range: str = Query("1d"),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    return _market_stock_card_chart_payload(symbol=symbol, chart_range=range, force_refresh=refresh)

@router.post("/market/comparison-history")
def market_comparison_history(request: MarketComparisonHistoryRequest) -> Dict[str, Any]:
    if request.start_date > request.end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    return _market_comparison_history_payload(request)

@router.get("/market/indices")
def market_indices(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_indices_payload(force_refresh=refresh)

@router.get("/market/indices/{index_code}")
def market_index_detail(index_code: str, refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_index_detail_payload(index_code, force_refresh=refresh)

@router.get("/market/flow")
def market_flow(
    limit: int = Query(40, ge=1, le=500),
    category: Optional[str] = Query(None),
    refresh: bool = Query(False),
    before: Optional[str] = Query(None, max_length=512),
    after: Optional[str] = Query(None, max_length=512),
) -> Dict[str, Any]:
    if before and after:
        raise HTTPException(status_code=400, detail="Aynı istekte before ve after cursor birlikte kullanılamaz.")
    return _market_flow_payload(
        limit=limit,
        category=category,
        force_refresh=refresh,
        before=before,
        after=after,
    )

@router.get("/market/flow/head")
def market_flow_head(category: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Return the light-weight cursor used to detect newly ingested KAP rows."""

    store_configured = _kap_flow_store.store_is_configured()
    if store_configured:
        _maybe_schedule_kap_flow_live_refresh()
    response = _kap_flow_store.read_flow_head(category=category)
    if store_configured:
        refresh_metadata = _kap_flow_refresh_metadata()
        response.update(refresh_metadata)
        response["warning"] = _kap_flow_refresh_warning(refresh_metadata)
    return response

@router.get("/market/xu030")
def market_xu030() -> Dict[str, Any]:
    return _xu030_payload()

@router.get("/market/commodities")
def market_commodities(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_commodities_payload(force_refresh=refresh)

@router.get("/market/fx")
def market_fx(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_fx_payload(force_refresh=refresh)

@router.get("/market/watch")
def market_watch(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_watch_payload(force_refresh=refresh)

@router.get("/market/watch/global")
def market_watch_global(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_watch_global_payload(force_refresh=refresh)

__all__ = [name for name in globals() if not name.startswith("__")]
