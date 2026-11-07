from __future__ import annotations

import asyncio
import concurrent.futures
import pytest

from app.api import app
from app import api_runtime, fund_api_service, kap_statement_refresh, kap_api_service
from app.market_broadcaster import get_market_broadcaster


def test_individual_shutdown_helpers():
    # Test market SWR executor shutdown
    orig_market = api_runtime._MARKET_SWR_REVALIDATION_EXECUTOR
    try:
        api_runtime.shutdown_market_swr_executor()
        assert orig_market._shutdown is True
    finally:
        api_runtime._MARKET_SWR_REVALIDATION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # Test fund executors shutdown
    orig_refresh = fund_api_service._FUND_REFRESH_EXECUTOR
    orig_hist = fund_api_service._FUND_HISTORY_EXECUTOR
    orig_alloc = fund_api_service._FUND_ALLOCATION_HISTORY_EXECUTOR
    try:
        fund_api_service.shutdown_fund_executors()
        assert orig_refresh._shutdown is True
        assert orig_hist._shutdown is True
        assert orig_alloc._shutdown is True
    finally:
        fund_api_service._FUND_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fund_api_service._FUND_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        fund_api_service._FUND_ALLOCATION_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # Test KAP statement executor shutdown
    orig_kap_stmt = kap_statement_refresh._KAP_STATEMENT_EXECUTOR
    try:
        kap_statement_refresh.shutdown_kap_statement_executor()
        assert orig_kap_stmt._shutdown is True
    finally:
        kap_statement_refresh._KAP_STATEMENT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def test_lifespan_shuts_down_all_executors(monkeypatch):
    monkeypatch.setattr("app.api.bootstrap_application_storage", lambda: None)

    async def fake_start():
        pass

    async def fake_stop():
        pass

    monkeypatch.setattr("app.api._start_fund_price_collector", fake_start)
    monkeypatch.setattr("app.api._stop_fund_price_collector", fake_stop)
    monkeypatch.setattr("app.api.close_postgres_pool", lambda: None)
    monkeypatch.setattr("app.api.close_shared_http_clients", lambda: None)

    broadcaster = get_market_broadcaster()
    broadcaster_stopped = []
    orig_stop = broadcaster.stop

    async def spy_stop():
        broadcaster_stopped.append(True)
        await orig_stop()

    monkeypatch.setattr(broadcaster, "stop", spy_stop)

    saved_executors = {
        "market": api_runtime._MARKET_SWR_REVALIDATION_EXECUTOR,
        "fund_refresh": fund_api_service._FUND_REFRESH_EXECUTOR,
        "fund_history": fund_api_service._FUND_HISTORY_EXECUTOR,
        "fund_alloc": fund_api_service._FUND_ALLOCATION_HISTORY_EXECUTOR,
        "kap_stmt": kap_statement_refresh._KAP_STATEMENT_EXECUTOR,
        "kap_flow": kap_api_service._KAP_FLOW_REFRESH_EXECUTOR,
    }

    try:
        async def run_context():
            async with app.router.lifespan_context(app):
                pass

        asyncio.run(run_context())

        assert saved_executors["market"]._shutdown is True
        assert saved_executors["fund_refresh"]._shutdown is True
        assert saved_executors["fund_history"]._shutdown is True
        assert saved_executors["fund_alloc"]._shutdown is True
        assert saved_executors["kap_stmt"]._shutdown is True
        assert saved_executors["kap_flow"]._shutdown is True
        assert len(broadcaster_stopped) == 1
    finally:
        api_runtime._MARKET_SWR_REVALIDATION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        fund_api_service._FUND_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fund_api_service._FUND_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        fund_api_service._FUND_ALLOCATION_HISTORY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        kap_statement_refresh._KAP_STATEMENT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        kap_api_service._KAP_FLOW_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
