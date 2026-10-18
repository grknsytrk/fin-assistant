"""Compatibility facade for tests that predate the API router split.

The API implementation now lives across routers and domain services.  Keep
the existing focused tests readable while routing monkeypatches to every
module that owns or re-exports a helper under test.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from app import api as _api
from app import api_runtime as _api_runtime
from app import fund_api_service as _fund_api_service
from app import kap_api_service as _kap_api_service
from app import kap_financials as _kap_financials
from app import kap_service as _kap_service
from app import market_service as _market_service
from app import model_portfolio_api_service as _model_portfolio_api_service
from app.routers import funds as _funds_router
from app.routers import kap as _kap_router
from app.routers import market as _market_router
from app.routers import model_portfolio as _model_portfolio_router


_MODULES: tuple[ModuleType, ...] = (
    _api,
    _market_router,
    _funds_router,
    _kap_router,
    _model_portfolio_router,
    _market_service,
    _fund_api_service,
    _kap_api_service,
    _model_portfolio_api_service,
    _kap_financials,
    _api_runtime,
    _kap_service,
)


class _ApiModuleFacade:
    def __getattr__(self, name: str) -> Any:
        for module in _MODULES:
            try:
                return getattr(module, name)
            except AttributeError:
                continue
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        owners = [module for module in _MODULES if hasattr(module, name)]
        if not owners:
            raise AttributeError(name)
        for module in owners:
            setattr(module, name, value)


api_module = _ApiModuleFacade()
