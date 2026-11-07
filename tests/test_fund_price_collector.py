from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
import pytest

from app import fund_api_service
from app.cache import InMemoryCache


def test_fund_price_collector_acquires_distributed_lock_and_runs(monkeypatch):
    cache = InMemoryCache()
    monkeypatch.setattr(fund_api_service, "_get_cache", lambda: cache)

    collect_mock = MagicMock(return_value={"valid_point_count": 50, "skipped_point_count": 0, "source": "test"})
    monkeypatch.setattr("app.fund_service.collect_daily_fund_prices", collect_mock)

    async def _test():
        executed = await fund_api_service._execute_fund_price_collector_cycle()
        assert executed is True
        assert collect_mock.called
        # Lock is released on exit
        assert cache.get("lock:fund-price-collector") is None
        # Cooldown key is set on success
        assert cache.get("lock:fund-price-collector:cooldown") is not None

    asyncio.run(_test())


def test_fund_price_collector_skips_when_lock_held_by_another_worker(monkeypatch):
    cache = InMemoryCache()
    cache.set_if_absent("lock:fund-price-collector", "other-worker-uuid", ttl_seconds=3600)
    monkeypatch.setattr(fund_api_service, "_get_cache", lambda: cache)

    collect_mock = MagicMock()
    monkeypatch.setattr("app.fund_service.collect_daily_fund_prices", collect_mock)

    async def _test():
        executed = await fund_api_service._execute_fund_price_collector_cycle()
        assert executed is False
        assert not collect_mock.called

    asyncio.run(_test())


def test_fund_price_collector_skips_when_cooldown_active(monkeypatch):
    cache = InMemoryCache()
    cache.set("lock:fund-price-collector:cooldown", "previous-worker-uuid", ttl_seconds=3600)
    monkeypatch.setattr(fund_api_service, "_get_cache", lambda: cache)

    collect_mock = MagicMock()
    monkeypatch.setattr("app.fund_service.collect_daily_fund_prices", collect_mock)

    async def _test():
        executed = await fund_api_service._execute_fund_price_collector_cycle()
        assert executed is False
        assert not collect_mock.called

    asyncio.run(_test())


def test_fund_price_collector_releases_lock_on_failure(monkeypatch):
    cache = InMemoryCache()
    monkeypatch.setattr(fund_api_service, "_get_cache", lambda: cache)

    collect_mock = MagicMock(side_effect=RuntimeError("TEFAS scrape error"))
    monkeypatch.setattr("app.fund_service.collect_daily_fund_prices", collect_mock)

    async def _test():
        executed = await fund_api_service._execute_fund_price_collector_cycle()
        assert executed is False
        assert collect_mock.called
        # Lock must be cleanly released even after exception
        assert cache.get("lock:fund-price-collector") is None
        # Cooldown must not be set on failure
        assert cache.get("lock:fund-price-collector:cooldown") is None

    asyncio.run(_test())


def test_fund_price_collector_heartbeat_renews_lease(monkeypatch):
    cache = InMemoryCache()
    monkeypatch.setattr(fund_api_service, "_get_cache", lambda: cache)

    renew_calls = []
    original_renew = cache.renew_if_owner

    def spy_renew(key, owner, *, ttl_seconds=600):
        renew_calls.append((key, owner, ttl_seconds))
        return original_renew(key, owner, ttl_seconds=ttl_seconds)

    monkeypatch.setattr(cache, "renew_if_owner", spy_renew)

    async def slow_collector(*args, **kwargs):
        # Simulate collection running while heartbeat fires
        await asyncio.sleep(0.06)
        return {"valid_point_count": 10, "skipped_point_count": 0, "source": "test"}

    monkeypatch.setattr(
        "app.fund_service.collect_daily_fund_prices",
        lambda *args, **kwargs: asyncio.run(slow_collector()),
    )

    async def _test():
        executed = await fund_api_service._execute_fund_price_collector_cycle(
            heartbeat_interval=0.02,
        )
        assert executed is True
        assert len(renew_calls) >= 1
        assert renew_calls[0][0] == "lock:fund-price-collector"

    asyncio.run(_test())
