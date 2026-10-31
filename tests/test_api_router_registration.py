import asyncio

from app import api as api_module
from app.api import app


EXPECTED_METHOD_PATHS = {
    ("GET", "/health"),
    ("GET", "/market/universe"),
    ("GET", "/market/stocks/search"),
    ("GET", "/market/stocks"),
    ("GET", "/market/screener"),
    ("GET", "/market/stream"),
    ("GET", "/market/stocks/cards"),
    ("GET", "/market/stocks/cards/chart"),
    ("POST", "/market/comparison-history"),
    ("GET", "/market/indices"),
    ("GET", "/market/indices/{index_code}"),
    ("GET", "/market/flow"),
    ("GET", "/market/flow/head"),
    ("GET", "/market/xu030"),
    ("GET", "/market/commodities"),
    ("GET", "/market/fx"),
    ("GET", "/market/watch"),
    ("GET", "/market/watch/global"),
    ("GET", "/funds"),
    ("GET", "/funds/search"),
    ("GET", "/funds/categories"),
    ("GET", "/funds/{fund_code}/performance"),
    ("GET", "/funds/{fund_code}/performance/status"),
    ("GET", "/funds/{fund_code}/yield-summary"),
    ("GET", "/funds/{fund_code}/holdings"),
    ("GET", "/funds/{fund_code}/holdings/live"),
    ("GET", "/funds/{fund_code}/allocations"),
    ("GET", "/funds/{fund_code}/allocations/history"),
    ("GET", "/funds/{fund_code}"),
    ("POST", "/admin/funds/refresh-snapshot"),
    ("GET", "/admin/funds/refresh-snapshot/status"),
    ("POST", "/admin/funds/collect-prices"),
    ("POST", "/admin/funds/{fund_code}/refresh-performance"),
    ("POST", "/admin/funds/{fund_code}/refresh-allocations"),
    ("POST", "/kap/overview-commentary"),
    ("POST", "/admin/kap/refresh"),
    ("GET", "/kap/overview-models"),
    ("GET", "/kap/companies"),
    ("GET", "/kap/snapshot"),
    ("POST", "/kap/valuation-history"),
    ("GET", "/kap/price"),
    ("GET", "/model-portfolio"),
    ("GET", "/model-portfolio/symbols"),
    ("GET", "/model-portfolio/performance"),
    ("POST", "/model-portfolio/admin/verify"),
    ("GET", "/model-portfolio/prefill"),
    ("POST", "/model-portfolio/improve-text"),
    ("POST", "/model-portfolio/positions"),
    ("PUT", "/model-portfolio/positions/{position_id}"),
    ("POST", "/model-portfolio/positions/{position_id}/close"),
    ("DELETE", "/model-portfolio/positions/{position_id}"),
}


def test_openapi_method_path_inventory_is_preserved():
    openapi = app.openapi()
    actual = {
        (method.upper(), path)
        for path, operations in openapi["paths"].items()
        for method in operations
    }

    assert actual == EXPECTED_METHOD_PATHS


def test_static_fund_paths_precede_the_dynamic_fund_detail_route():
    fund_routes = [
        route.path
        for route in app.routes
        if getattr(route, "path", "").startswith("/funds/")
    ]

    assert fund_routes.index("/funds/search") < fund_routes.index("/funds/{fund_code}")
    assert fund_routes.index("/funds/categories") < fund_routes.index("/funds/{fund_code}")


def test_application_lifecycle_and_gradio_bootstrap_exports(monkeypatch):
    calls = []

    monkeypatch.setattr(api_module, "bootstrap_application_storage", lambda: calls.append("bootstrap"))

    async def start_collector():
        calls.append("start")

    async def stop_collector():
        calls.append("stop")

    monkeypatch.setattr(api_module, "_start_fund_price_collector", start_collector)
    monkeypatch.setattr(api_module, "_stop_fund_price_collector", stop_collector)
    monkeypatch.setattr(api_module, "close_postgres_pool", lambda: calls.append("close"))

    async def exercise_lifespan():
        async with app.router.lifespan_context(app):
            assert calls == ["bootstrap", "start"]

    assert api_module.app is app
    assert callable(api_module.bootstrap_application_storage)
    assert callable(api_module.configure_http_middleware)
    asyncio.run(exercise_lifespan())
    assert calls == ["bootstrap", "start", "stop", "close"]
