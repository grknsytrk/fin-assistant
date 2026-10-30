from __future__ import annotations

from app.api_runtime import *
from app.market_service import *
from app.model_portfolio_service import *

def _extract_model_portfolio_admin_token(request: Request) -> str:
    return str(request.headers.get("x-admin-token") or "").strip()

def _require_model_portfolio_admin(request: Request) -> None:
    from app.model_portfolio_service import verify_admin_session_token

    auth = str(request.headers.get("authorization") or "").strip()
    scheme, _, session_token = auth.partition(" ")
    if scheme.lower() != "bearer" or not verify_admin_session_token(session_token):
        raise HTTPException(
            status_code=401,
            detail="Yönetici oturumu geçersiz veya süresi dolmuş. Lütfen yeniden giriş yapın.",
        )

def _model_portfolio_cached_quote_map(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Read live stock quotes from warm _STOCKS_CACHE / shared cache first."""
    normalized = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
    if not normalized:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    target_set = set(normalized)

    for idx_name in ("XUTUM", "XU100", "XU030"):
        local_entry = _STOCKS_CACHE.get(f"payload:{idx_name}", {}).get("entry")
        payload = _market_quote_entry_payload(local_entry)
        if payload is None:
            shared_entry = _shared_cache_get_dict(f"api:market:stocks:{idx_name}:v4")
            payload = _market_quote_entry_payload(shared_entry)
        if payload is None:
            # Keep a prior cache generation useful during rolling deploys;
            # it still contains current quotes while the v4 snapshot warms.
            shared_entry = _shared_cache_get_dict(f"api:market:stocks:{idx_name}:v3")
            payload = _market_quote_entry_payload(shared_entry)
        if payload is None:
            shared_entry = _shared_cache_get_dict(f"api:market:stocks:{idx_name}:v2")
            payload = _market_quote_entry_payload(shared_entry)
        rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol") or row.get("company") or "").strip().upper()
            if sym in target_set and sym not in result and row.get("price") is not None:
                result[sym] = {
                    "price": row.get("price"),
                    "change": row.get("change"),
                    "change_pct": row.get("change_pct"),
                    "market_cap": row.get("market_cap"),
                    "price_as_of": row.get("price_as_of") or payload.get("as_of"),
                }
        if len(result) >= len(target_set):
            return result

    missing = [s for s in normalized if s not in result]
    if missing:
        fetched = _fetch_market_price_map(missing)
        try:
            snapshot_map = get_financial_snapshots_map(CONFIG.paths.processed_dir, missing)
        except Exception as exc:
            LOGGER.warning("model portfolio financial snapshot lookup failed: %s", exc)
            snapshot_map = {}
        for sym in missing:
            quote = fetched.get(sym) or {}
            if quote.get("price") is not None:
                cached_meta = snapshot_map.get(sym, {})
                result[sym] = {
                    "price": quote.get("price"),
                    "change": quote.get("change"),
                    "change_pct": quote.get("change_pct"),
                    "market_cap": _market_cap_from_quote_and_meta(quote, cached_meta, None),
                    "price_as_of": quote.get("as_of"),
                }
    return result

def _model_portfolio_cached_index(index_symbol: str) -> Optional[float]:
    """Read a current index level from already-warm caches before falling back."""
    normalized_symbol = str(index_symbol or "").strip().upper()
    indices_entry = _MARKET_INDICES_CACHE.get("payload")
    indices_payload = _swr_entry_payload(indices_entry)
    if indices_payload is None:
        indices_payload = _swr_entry_payload(_shared_cache_get_dict("api:market:indices:v2"))
    if isinstance(indices_payload, dict):
        for row in list(indices_payload.get("rows") or []):
            if isinstance(row, dict) and str(row.get("symbol") or "").upper() == normalized_symbol:
                price = _positive_float(row.get("price"))
                if price is not None:
                    return price

    watch_entry = _WATCH_CACHE.get("payload")
    watch_payload = _swr_entry_payload(watch_entry)
    if isinstance(watch_payload, dict):
        sections = watch_payload.get("sections") if isinstance(watch_payload.get("sections"), dict) else {}
        for row in list(sections.get("indices") or []):
            if isinstance(row, dict) and str(row.get("symbol") or "").upper() == normalized_symbol:
                price = _positive_float(row.get("price"))
                if price is not None:
                    return price

    base_entry = _STOCK_RETURN_BASE_CACHE.get(normalized_symbol)
    if isinstance(base_entry, dict) and isinstance(base_entry.get("data"), dict):
        price = _positive_float(base_entry["data"].get("latest_close"))
        if price is not None:
            return price

    try:
        row = _market_index_row(normalized_symbol)
        return _positive_float(row.get("price"))
    except Exception:
        return None

def _model_portfolio_cached_xu100() -> Optional[float]:
    return _model_portfolio_cached_index("XU100")

def _model_portfolio_cached_xu030() -> Optional[float]:
    return _model_portfolio_cached_index("XU030")

def _model_portfolio_comparison_series(
    kind: str,
    symbol: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    # This request model is declared by the market router, not market_service.
    # Import lazily so the router layer does not participate in module startup
    # and create a circular import while the API routers are being registered.
    from app.routers.market import MarketComparisonHistoryAsset

    handler = _index_comparison_history if kind == "index" else _stock_comparison_history
    asset = MarketComparisonHistoryAsset(
        id=f"{kind}:{symbol}",
        kind="index" if kind == "index" else "stock",
        symbol=symbol,
        label=symbol,
    )
    res = _comparison_history_cached_asset(
        asset,
        start_date=start_date,
        end_date=end_date,
        handler=handler,
    )
    pts = res.get("points")
    return list(pts) if isinstance(pts, list) else []

def _build_current_model_portfolio_response() -> Dict[str, Any]:
    from app.model_portfolio_service import (
        build_model_portfolio_payload,
        load_model_portfolio_state,
    )

    state = load_model_portfolio_state(CONFIG.paths.processed_dir)
    symbols = sorted(
        {
            str(p.get("symbol") or "").strip().upper()
            for p in state.get("positions", [])
            if str(p.get("symbol") or "").strip()
        }
    )
    quote_map = _model_portfolio_cached_quote_map(symbols)
    try:
        instrument_map = get_instruments(CONFIG.paths.processed_dir, "stock", symbols) if symbols else {}
    except Exception:
        instrument_map = {}

    current_xu100 = _model_portfolio_cached_xu100()
    return build_model_portfolio_payload(
        CONFIG.paths.processed_dir,
        quote_map=quote_map,
        instrument_map=instrument_map,
        current_xu100=current_xu100,
        valuation_resolver=lambda sym, mcap: _resolve_market_card_valuation(sym, market_cap=mcap),
    )

__all__ = [name for name in globals() if not name.startswith("__")]
