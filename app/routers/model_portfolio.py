from __future__ import annotations

from fastapi import APIRouter

from app.api_runtime import *
from app.model_portfolio_api_service import *
from app.model_portfolio_service import *

router = APIRouter()

@router.get("/model-portfolio")
def get_model_portfolio() -> Dict[str, Any]:
    return _build_current_model_portfolio_response()

@router.get("/model-portfolio/symbols")
def get_model_portfolio_symbols() -> Dict[str, Any]:
    from app.model_portfolio_service import get_active_model_portfolio_symbols

    return get_active_model_portfolio_symbols(CONFIG.paths.processed_dir)

@router.get("/model-portfolio/performance")
def get_model_portfolio_performance() -> Dict[str, Any]:
    from app.model_portfolio_service import (
        build_model_portfolio_performance_payload,
        load_model_portfolio_state,
    )

    state = load_model_portfolio_state(CONFIG.paths.processed_dir)
    active_symbols = sorted(
        {
            str(p.get("symbol") or "").strip().upper()
            for p in state.get("positions", [])
            if str(p.get("status") or "active") == "active" and str(p.get("symbol") or "").strip()
        }
    )
    quote_map = _model_portfolio_cached_quote_map(active_symbols)
    current_xu100 = _model_portfolio_cached_xu100()
    current_xu030 = _model_portfolio_cached_xu030()
    return build_model_portfolio_performance_payload(
        CONFIG.paths.processed_dir,
        fetch_series_fn=_model_portfolio_comparison_series,
        quote_map=quote_map,
        current_xu100=current_xu100,
        current_xu030=current_xu030,
        shared_cache_get=_shared_cache_get_dict,
        shared_cache_set=_shared_cache_set,
    )

@router.post("/model-portfolio/admin/verify")
def verify_model_portfolio_admin(request: Request, response: Response) -> Dict[str, Any]:
    from app.model_portfolio_service import (
        ADMIN_SESSION_TTL_SECONDS,
        create_admin_session_token,
        expected_admin_token,
        is_admin_locked_out,
        record_admin_auth_result,
        verify_admin_token,
    )
    from app.rate_limit import client_identity

    response.headers["Cache-Control"] = "no-store, private"
    response.headers["Pragma"] = "no-cache"

    if not expected_admin_token():
        raise HTTPException(
            status_code=503,
            detail="Yönetici anahtarı sunucu ortamında henüz yapılandırılmamış.",
        )

    client_ip = client_identity(request)
    if is_admin_locked_out(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Çok fazla başarısız giriş denemesi yapıldı. Lütfen 10 dakika sonra tekrar deneyin.",
            headers={"Retry-After": "600"},
        )
    admin_token = _extract_model_portfolio_admin_token(request)
    if verify_admin_token(admin_token):
        record_admin_auth_result(client_ip, True)
        session_token = create_admin_session_token()
        if not session_token:
            raise HTTPException(status_code=503, detail="Yönetici oturumu oluşturulamadı.")
        return {
            "ok": True,
            "session_token": session_token,
            "expires_in_seconds": ADMIN_SESSION_TTL_SECONDS,
        }

    allowed, _retry_after = record_admin_auth_result(client_ip, False)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Çok fazla başarısız giriş denemesi yapıldı. Lütfen 10 dakika sonra tekrar deneyin.",
            headers={"Retry-After": str(max(1, _retry_after))},
        )
    raise HTTPException(status_code=401, detail="Yönetici anahtarı geçersiz veya eksik.")

@router.get("/model-portfolio/prefill")
def prefill_model_portfolio_position(
    symbol: str = Query(..., min_length=1, max_length=16),
    entry_date: Optional[date] = Query(None),
) -> Dict[str, Any]:
    normalized_symbol = _normalize_market_stock_card_symbol(symbol)
    target_date = entry_date or date.today()
    today = date.today()

    quote_map = _model_portfolio_cached_quote_map([normalized_symbol])
    quote = quote_map.get(normalized_symbol) or {}
    try:
        instrument_map = get_instruments(CONFIG.paths.processed_dir, "stock", [normalized_symbol])
    except Exception:
        instrument_map = {}
    inst = instrument_map.get(normalized_symbol) or {}
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    cached_meta = _load_cached_kap_market_metadata(cache_dir, normalized_symbol)

    company_name = (
        str(inst.get("name") or "").strip()
        or str(cached_meta.get("company_title") or "").strip()
        or normalized_symbol
    )
    logo_url = str(inst.get("logo_url") or "").strip() or None
    logo_source = str(inst.get("logo_source") or "").strip() or None

    live_price = _positive_float(quote.get("price"))
    live_xu100 = _model_portfolio_cached_xu100()
    resolved_entry_price = live_price
    resolved_entry_xu100 = live_xu100

    if target_date < today:
        lookback_start = target_date - timedelta(days=10)
        stock_pts = _model_portfolio_comparison_series("stock", normalized_symbol, lookback_start, target_date)
        if stock_pts:
            hist_price = _positive_float(stock_pts[-1].get("value"))
            if hist_price is not None:
                resolved_entry_price = hist_price
        xu_pts = _model_portfolio_comparison_series("index", "XU100", lookback_start, target_date)
        if xu_pts:
            hist_xu = _positive_float(xu_pts[-1].get("value"))
            if hist_xu is not None:
                resolved_entry_xu100 = hist_xu

    market_cap = _positive_float(quote.get("market_cap"))
    if market_cap is None and resolved_entry_price is not None:
        shares = _positive_float(cached_meta.get("shares_outstanding"))
        if shares is not None:
            market_cap = resolved_entry_price * shares

    valuation = _resolve_market_card_valuation(normalized_symbol, market_cap=market_cap)

    return {
        "symbol": normalized_symbol,
        "company_name": company_name,
        "logo_url": logo_url,
        "logo_source": logo_source,
        "entry_date": target_date.isoformat(),
        "entry_price": round(resolved_entry_price, 4) if resolved_entry_price is not None else None,
        "current_price": round(live_price, 4) if live_price is not None else None,
        "entry_xu100": round(resolved_entry_xu100, 2) if resolved_entry_xu100 is not None else None,
        "entry_multiples": {
            "fk": valuation.get("fk"),
            "fd_favok": valuation.get("fd_favok"),
            "pd_dd": valuation.get("pd_dd"),
            "net_borc_favok": valuation.get("net_borc_favok"),
            "period": valuation.get("valuation_financial_period"),
        },
    }

@router.post("/model-portfolio/positions")
async def create_model_portfolio_position_endpoint(request: Request) -> Dict[str, Any]:
    from app.model_portfolio_service import create_position

    _require_model_portfolio_admin(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Geçersiz JSON gövdesi.") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Geçersiz istek formatı.")

    raw_symbol = str(body.get("symbol") or "").strip().upper()
    if not raw_symbol:
        raise HTTPException(status_code=400, detail="Hisse kodu (symbol) zorunludur.")

    normalized_symbol = _normalize_market_stock_card_symbol(raw_symbol)
    body["symbol"] = normalized_symbol

    quote_map = _model_portfolio_cached_quote_map([normalized_symbol])
    quote = quote_map.get(normalized_symbol) or {}
    try:
        instrument_map = get_instruments(CONFIG.paths.processed_dir, "stock", [normalized_symbol])
    except Exception:
        instrument_map = {}
    inst = instrument_map.get(normalized_symbol) or {}
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    cached_meta = _load_cached_kap_market_metadata(cache_dir, normalized_symbol)

    company_name = (
        str(inst.get("name") or "").strip()
        or str(cached_meta.get("company_title") or "").strip()
        or normalized_symbol
    )
    market_cap = _positive_float(quote.get("market_cap"))
    valuation = _resolve_market_card_valuation(normalized_symbol, market_cap=market_cap)
    current_xu100 = _model_portfolio_cached_xu100()

    if body.get("entry_price") is None and quote.get("price") is not None:
        body["entry_price"] = quote.get("price")

    try:
        create_position(
            CONFIG.paths.processed_dir,
            body,
            company_name=company_name,
            logo_url=str(inst.get("logo_url") or "").strip() or None,
            logo_source=str(inst.get("logo_source") or "").strip() or None,
            entry_multiples={
                "fk": valuation.get("fk"),
                "fd_favok": valuation.get("fd_favok"),
                "pd_dd": valuation.get("pd_dd"),
                "net_borc_favok": valuation.get("net_borc_favok"),
                "period": valuation.get("valuation_financial_period"),
            },
            entry_xu100=current_xu100,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_current_model_portfolio_response()

@router.put("/model-portfolio/positions/{position_id}")
async def update_model_portfolio_position_endpoint(position_id: str, request: Request) -> Dict[str, Any]:
    from app.model_portfolio_service import update_position

    _require_model_portfolio_admin(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Geçersiz JSON gövdesi.") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Geçersiz istek formatı.")

    try:
        update_position(CONFIG.paths.processed_dir, position_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_current_model_portfolio_response()

@router.post("/model-portfolio/positions/{position_id}/close")
async def close_model_portfolio_position_endpoint(position_id: str, request: Request) -> Dict[str, Any]:
    from app.model_portfolio_service import close_position, load_model_portfolio_state

    _require_model_portfolio_admin(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    state = load_model_portfolio_state(CONFIG.paths.processed_dir)
    pos = next((p for p in state.get("positions", []) if str(p.get("id")) == str(position_id)), None)
    sym = str((pos or {}).get("symbol") or "").strip().upper()
    quote = (_model_portfolio_cached_quote_map([sym]).get(sym) or {}) if sym else {}
    current_xu100 = _model_portfolio_cached_xu100()

    try:
        close_position(
            CONFIG.paths.processed_dir,
            position_id,
            body,
            current_price_fallback=_positive_float(quote.get("price")),
            current_xu100_fallback=current_xu100,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_current_model_portfolio_response()

@router.delete("/model-portfolio/positions/{position_id}")
def delete_model_portfolio_position_endpoint(position_id: str, request: Request) -> Dict[str, Any]:
    from app.model_portfolio_service import delete_position

    _require_model_portfolio_admin(request)
    try:
        delete_position(CONFIG.paths.processed_dir, position_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return _build_current_model_portfolio_response()

__all__ = [name for name in globals() if not name.startswith("__")]
