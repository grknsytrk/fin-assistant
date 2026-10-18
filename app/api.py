"""FastAPI application composition root.

Domain endpoints and their supporting implementation live in dedicated
routers and services. This module keeps the stable ``app.api:app`` entrypoint,
application lifecycle, shared HTTP middleware, and a small compatibility
lookup for older imports of private API helpers.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from app.api_runtime import (
    CONFIG,
    LOGGER,
    RequestRateLimitMiddleware,
    _cache_status,
    close_postgres_pool,
    database_enabled,
    ensure_financial_snapshot_schema,
    ensure_json_cache_schema,
    ensure_kap_flow_schema,
    ensure_reference_data_schema,
    ensure_refresh_lease_schema,
    hydrate_json_cache,
    sync_financial_snapshots_from_kap_cache,
    sync_reference_data_from_caches,
)
from app import api_runtime as _api_runtime
from app import fund_api_service as _fund_api_service
from app import foreign_instruments as _foreign_instruments
from app import kap_api_service as _kap_api_service
from app import kap_financials as _kap_financials
from app import market_service as _market_service
from app import model_portfolio_api_service as _model_portfolio_api_service
from app import kap_service as _kap_service
from app.fund_api_service import _start_fund_price_collector, _stop_fund_price_collector


# The Gradio Space entrypoint imports this callable directly and invokes it
# because Gradio's FastAPI host does not execute this app's lifespan.
def bootstrap_application_storage() -> None:
    """Create durable schemas and hydrate caches before serving requests.

    Keeping the DDL here prevents normal API requests from racing on schema
    creation or waiting on Supabase statement timeouts.
    """

    ensure_json_cache_schema()
    ensure_refresh_lease_schema()
    ensure_kap_flow_schema()
    ensure_reference_data_schema(CONFIG.paths.processed_dir)
    from app.fund_service import ensure_fund_prices_schema

    ensure_fund_prices_schema(CONFIG.paths.processed_dir)
    hydrate_json_cache(CONFIG.paths.processed_dir)
    sync_reference_data_from_caches(CONFIG.paths.processed_dir)
    ensure_financial_snapshot_schema(CONFIG.paths.processed_dir)
    snapshot_sync = sync_financial_snapshots_from_kap_cache(CONFIG.paths.processed_dir)
    if snapshot_sync.get("upserted_count") or snapshot_sync.get("error_count"):
        LOGGER.info("KAP financial snapshot sync: %s", snapshot_sync)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        await asyncio.to_thread(bootstrap_application_storage)
    except Exception:
        LOGGER.exception("application storage bootstrap failed")
        if database_enabled():
            raise
    await _start_fund_price_collector()
    try:
        yield
    finally:
        await _stop_fund_price_collector()
        close_postgres_pool()


app = FastAPI(title="RAG-Fin API", version="0.10.0", lifespan=_lifespan)


def _cors_allow_origins() -> List[str]:
    raw = os.getenv("RAGFIN_CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173"]


def _cors_allow_origin_regex() -> str:
    raw = os.getenv("RAGFIN_CORS_ALLOW_ORIGIN_REGEX", "").strip()
    if raw:
        return raw
    # Allow local dev hosts on any port (Vite may auto-switch ports).
    return r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$"


def configure_http_middleware(target: FastAPI) -> None:
    """Install the public HTTP policy on every production API host."""

    target.add_middleware(RequestRateLimitMiddleware)
    # CORS wraps the limiter so browser clients can read 429/Retry-After.
    target.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_allow_origins(),
        allow_origin_regex=_cors_allow_origin_regex(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["ETag"],
    )
    # The fund catalogue is returned in one response; compression preserves
    # the API contract while reducing transfer size.
    target.add_middleware(GZipMiddleware, minimum_size=1024)


configure_http_middleware(app)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", **_cache_status()}


from app.routers.market import router as market_router
from app.routers.funds import router as funds_router
from app.routers.kap import router as kap_router
from app.routers.model_portfolio import router as model_portfolio_router

app.include_router(market_router)
app.include_router(funds_router)
app.include_router(kap_router)
app.include_router(model_portfolio_router)


_LEGACY_MODULES = (
    _api_runtime,
    _market_service,
    _fund_api_service,
    _kap_api_service,
    _kap_financials,
    _kap_service,
    _foreign_instruments,
    _model_portfolio_api_service,
)


def __getattr__(name: str) -> Any:
    """Resolve legacy private-helper imports to their new domain owner."""

    for module in _LEGACY_MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
