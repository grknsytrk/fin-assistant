from __future__ import annotations

from fastapi import APIRouter

from app.api_runtime import *
from app.kap_api_service import *
from app.market_service import *
from app.kap_service import *

router = APIRouter()


@router.get("/kap/overview-models")
async def kap_overview_models() -> Dict[str, Any]:
    return await list_overview_models()

class KapValuationHistoryRequest(BaseModel):
    company: str = Field(..., min_length=1, max_length=16)
    quarters: List[Dict[str, Any]] = Field(..., max_length=44)
    company_kind: Optional[Literal["generic", "bank", "insurance"]] = None

@router.post("/kap/overview-commentary")
async def kap_overview_commentary(request: Request) -> Dict[str, Any]:
    started_at = time.perf_counter()
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        LOGGER.warning(
            "[kap_overview_commentary] request body too large | bytes=%s limit=%s",
            len(body),
            MAX_REQUEST_BYTES,
        )
        raise HTTPException(status_code=413, detail="request body en fazla 64 KB olabilir")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "[kap_overview_commentary] invalid json body | bytes=%s error=%s",
            len(body),
            exc,
        )
        raise HTTPException(status_code=400, detail="gecerli JSON body gerekli") from exc
    LOGGER.info(
        "[kap_overview_commentary] request received | company=%s latest_period=%s bytes=%s",
        str(payload.get("company") or "").strip(),
        str(payload.get("latest_period") or "").strip(),
        len(body),
    )
    try:
        response = await _run_overview_commentary_until_done_or_disconnected(request, payload)
    except PayloadValidationError as exc:
        LOGGER.warning(
            "[kap_overview_commentary] payload validation failed | company=%s detail=%s",
            str(payload.get("company") or "").strip(),
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    LOGGER.info(
        "[kap_overview_commentary] completed | company=%s ok=%s model=%s score_source=%s elapsed_ms=%s error=%s",
        str(payload.get("company") or "").strip(),
        response.get("ok"),
        response.get("model_used"),
        ((response.get("scorecard") or {}) if isinstance(response.get("scorecard"), dict) else {}).get("score_source"),
        elapsed_ms,
        (str(response.get("error") or "")[:180] or None),
    )
    return response

@router.post("/admin/kap/refresh")
def admin_refresh_kap_flow(
    request: Request,
    fast: bool = Query(False),
) -> Dict[str, Any]:
    """Refresh KAP flow storage for the Cloudflare Cron or manual fallback."""

    _require_admin_refresh_access(request)
    return _refresh_kap_flow_store(fast=fast)

@router.get("/kap/companies")
def kap_companies() -> Dict[str, Any]:
    from app.kap_service import get_kap_companies

    now_ts = time.time()
    cached = _KAP_COMPANIES_RESPONSE_CACHE.get("data")
    if cached and now_ts - float(_KAP_COMPANIES_RESPONSE_CACHE.get("_ts", 0)) < _KAP_COMPANIES_RESPONSE_CACHE_TTL:
        return cached

    shared_cached = _shared_cache_get_dict(_KAP_COMPANIES_RESPONSE_CACHE_KEY)
    if shared_cached and isinstance(shared_cached.get("companies"), list):
        _KAP_COMPANIES_RESPONSE_CACHE["_ts"] = now_ts
        _KAP_COMPANIES_RESPONSE_CACHE["data"] = shared_cached
        return shared_cached

    companies = get_kap_companies()
    instrument_names = get_instrument_names(CONFIG.paths.processed_dir, "stock")
    try:
        financial_snapshots = get_financial_snapshots_map(CONFIG.paths.processed_dir, companies)
    except Exception as exc:
        LOGGER.warning("KAP company financial snapshot lookup failed: %s", exc)
        financial_snapshots = {}
    items: List[Dict[str, Any]] = []
    for symbol in companies:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            continue
        cached_meta = financial_snapshots.get(normalized) or {}
        title = str(instrument_names.get(normalized) or cached_meta.get("company_title") or "").strip()
        company_code = normalized
        aliases = [normalized]
        if company_code and company_code != normalized:
            aliases.append(company_code)
        if title:
            aliases.append(title)
        items.append(
            {
                "symbol": normalized,
                "title": title or None,
                "aliases": aliases,
                "latest_quarter": cached_meta.get("latest_quarter"),
                "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
            }
        )
    payload = {"companies": companies, "items": items}
    _KAP_COMPANIES_RESPONSE_CACHE["_ts"] = now_ts
    _KAP_COMPANIES_RESPONSE_CACHE["data"] = payload
    _shared_cache_set(
        _KAP_COMPANIES_RESPONSE_CACHE_KEY,
        payload,
        ttl_seconds=_KAP_COMPANIES_RESPONSE_CACHE_TTL,
    )
    return payload

@router.get("/kap/snapshot")
def kap_snapshot(
    company: str = Query(..., min_length=1),
    refresh: bool = Query(False),
    max_quarters: int = Query(10, ge=1, le=44),
) -> Dict[str, Any]:
    if not getattr(CONFIG, "kap", None) or not getattr(CONFIG.kap, "enabled", False):
        raise HTTPException(status_code=503, detail="KAP modülü devre dışı.")
    cache_key = _kap_snapshot_response_cache_key(company, max_quarters)
    # A missing response cache must never make the browser fetch KAP reports.
    # Disk snapshots survive response-cache expiry and are safe to serve while
    # the single-flight worker checks for new filings or additional quarters.
    started = time.perf_counter()
    entry = _shared_cache_get_dict(cache_key)
    payload = _swr_entry_payload(entry) if _swr_entry_is_stale(entry) else None
    stale = not _swr_entry_is_fresh(entry)
    cache_status = "shared_hit" if payload is not None and not stale else "stale"
    if payload is None:
        payload = _cached_kap_snapshot_response(company)
        cache_status = "disk_hit" if payload else "pending"
    refresh_pending = False
    if stale or refresh:
        refresh_pending = _schedule_swr_revalidation(
            cache_key=cache_key,
            coordination_key=cache_key.split(":quarters=")[0],
            factory=lambda: _build_kap_snapshot_response(company, refresh=False, max_quarters=max_quarters),
            fresh_ttl_seconds=_KAP_SNAPSHOT_RESPONSE_CACHE_TTL,
            stale_ttl_seconds=_KAP_SNAPSHOT_RESPONSE_STALE_TTL,
        )
    if payload is None:
        payload = {
            "ok": False, "company": company, "stock_code": company,
            "company_title": "", "quarters": [], "summary": {},
            "latest_quarter": None, "fetched_at": "", "cache_hit": False,
            "pending": refresh_pending,
            "error": "Finansal veriler hazırlanıyor. Kısa süre sonra yeniden deneyebilirsiniz.",
        }
    LOGGER.info("kap snapshot company=%s cache=%s elapsed_ms=%.1f",
                company, cache_status, (time.perf_counter() - started) * 1000)
    response = _annotate_kap_response_cache(
        payload,
        cache_hit=cache_status in {"local_hit", "shared_hit", "coalesced", "stale", "disk_hit"},
    )
    response["response_cache_status"] = cache_status
    response["response_cache_stale"] = stale
    response["refresh_pending"] = refresh_pending
    return response

@router.post("/kap/valuation-history")
def kap_valuation_history(request: KapValuationHistoryRequest) -> Dict[str, Any]:
    from app.kap_service import normalize_kap_symbol

    company = normalize_kap_symbol(request.company)
    if not company or not request.quarters:
        return {
            "ok": False,
            "pending": False,
            "company": company or request.company,
            "periods": [],
            "error": "Çarpan geçmişi için finansal dönem bulunamadı.",
        }
    return _build_kap_historical_multiples(
        snapshot={
            "company": company,
            "stock_code": company,
            "company_kind": request.company_kind,
            "quarters": request.quarters,
        },
    )

@router.get("/kap/price")
def kap_price(symbol: str = Query(..., min_length=1)) -> Dict[str, Any]:
    return _fetch_kap_price_payload(symbol)

__all__ = [name for name in globals() if not name.startswith("__")]
