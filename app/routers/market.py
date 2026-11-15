from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.api_runtime import (
    Any,
    BaseModel,
    Dict,
    Field,
    HTTPException,
    List,
    Literal,
    Optional,
    Query,
    StreamingResponse,
    _kap_flow_store,
    asyncio,
    date,
    time,
)
from app.etag import conditional_json_response
from app.kap_api_service import (
    _kap_flow_refresh_metadata,
    _kap_flow_refresh_warning,
    _market_flow_payload,
    _maybe_schedule_kap_flow_live_refresh,
)
from app.market_broadcaster import get_market_broadcaster
from app.market_service import (
    _compute_market_stocks_delta,
    _is_bist_market_open,
    _log_market_cache_event,
    _market_commodities_payload,
    _market_comparison_history_payload,
    _market_fx_payload,
    _market_index_detail_payload,
    _market_indices_payload,
    _market_stock_card_chart_payload,
    _market_stock_cards_payload,
    _market_stock_symbols_for_index,
    _market_stocks_payload,
    _market_universe_payload,
    _market_watch_global_payload,
    _market_watch_payload,
    _market_watch_unified_payload,
    _normalize_stock_index,
    _xu030_payload,
    market_stocks_etag_revision,
)

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
            "representation": "market-stocks-v2",
            "index": _normalize_stock_index(index),
            "source": source_revision,
        }
    return conditional_json_response(request, payload, revision)

@router.get("/market/stream")
async def market_stream(
    request: Request,
    index: str = Query("XUTUM"),
    once: bool = Query(False),
) -> StreamingResponse:
    broadcaster = get_market_broadcaster()

    return StreamingResponse(
        broadcaster.subscribe(index=index, once=once, request=request),
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

@router.get("/market/watch/unified")
def market_watch_unified(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_watch_unified_payload(force_refresh=refresh)

__all__ = [name for name in globals() if not name.startswith("__")]
