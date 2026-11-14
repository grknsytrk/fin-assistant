from __future__ import annotations

import asyncio
import json
import pytest
from starlette.testclient import TestClient

from app.api import app
from app.market_broadcaster import MarketStreamBroadcaster, get_market_broadcaster


def test_market_watch_unified_returns_all_sections_and_timestamp():
    client = TestClient(app)
    response = client.get("/market/watch/unified")
    assert response.status_code == 200
    data = response.json()

    assert "indices" in data
    assert "fx" in data
    assert "commodities" in data
    assert "sections" in data
    assert "as_of" in data
    assert isinstance(data["as_of"], str)
    assert data["source"] == "yahoo_finance_chart"

    sections = data["sections"]
    assert "indices" in sections
    assert "fx" in sections
    assert "commodities" in sections
    assert isinstance(sections["indices"], list)
    assert isinstance(sections["fx"], list)
    assert isinstance(sections["commodities"], list)


def test_market_watch_unified_supports_force_refresh():
    client = TestClient(app)
    res1 = client.get("/market/watch/unified")
    assert res1.status_code == 200
    res2 = client.get("/market/watch/unified?refresh=true")
    assert res2.status_code == 200
    data2 = res2.json()
    assert "indices" in data2
    assert "as_of" in data2


def test_broadcaster_fans_out_deltas_to_multiple_queues():
    async def _test():
        broadcaster = MarketStreamBroadcaster()
        q1 = asyncio.Queue(maxsize=4)
        q2 = asyncio.Queue(maxsize=4)
        broadcaster._subscribers.add(q1)
        broadcaster._subscribers.add(q2)

        event_payload = "event: stocks_delta\ndata: {\"count\": 1}\n\n"
        broadcaster._fanout(event_payload)

        msg1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        msg2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert msg1 == event_payload
        assert msg2 == event_payload

    asyncio.run(_test())


def test_broadcaster_handles_slow_client_queue_full_without_blocking():
    async def _test():
        broadcaster = MarketStreamBroadcaster()
        slow_queue = asyncio.Queue(maxsize=1)
        fast_queue = asyncio.Queue(maxsize=4)
        broadcaster._subscribers.add(slow_queue)
        broadcaster._subscribers.add(fast_queue)

        slow_queue.put_nowait("old_message")  # now full
        event = "event: stocks_delta\ndata: {\"count\": 2}\n\n"
        broadcaster._fanout(event)  # must not raise QueueFull or block

        msg = await asyncio.wait_for(fast_queue.get(), timeout=1.0)
        assert msg == event
        assert slow_queue.qsize() == 1  # old message retained

    asyncio.run(_test())


def test_broadcaster_index_filtering():
    async def _test():
        broadcaster = MarketStreamBroadcaster()
        q_xutum = asyncio.Queue(maxsize=4)
        q_xu030 = asyncio.Queue(maxsize=4)
        broadcaster._subscribers.add(q_xutum)
        broadcaster._queue_indices[q_xutum] = "XUTUM"
        broadcaster._subscribers.add(q_xu030)
        broadcaster._queue_indices[q_xu030] = "XU030"

        event_xu030 = "event: stocks_delta\ndata: {\"index\": \"XU030\"}\n\n"
        broadcaster._fanout(event_xu030, target_index="XU030")

        msg_xu030 = await asyncio.wait_for(q_xu030.get(), timeout=1.0)
        assert msg_xu030 == event_xu030
        assert q_xutum.empty()

    asyncio.run(_test())


def test_market_stream_endpoint_once_mode():
    client = TestClient(app)
    res = client.get("/market/stream?index=XUTUM&once=true")
    assert res.status_code == 200
    assert "event: stocks_delta" in res.text
    assert "data: " in res.text


def test_fx_and_index_return_bases_use_bulk_redis_keys(monkeypatch):
    from app import cache as cache_module
    from app import market_service

    cache_module.reset_cache_for_tests()
    market_service._MARKET_INDEX_RETURN_CACHE.clear()
    market_service._FX_RETURN_CACHE.clear()
    market_service._MARKET_INDICES_CACHE.clear()
    market_service._FX_CACHE.clear()
    market_service._MARKET_INDEX_QUOTE_CACHE.clear()

    set_keys: list[str] = []
    orig_shared_set = market_service._shared_cache_set

    def recording_shared_set(key: str, value, ttl_seconds: int) -> None:
        set_keys.append(key)
        orig_shared_set(key, value, ttl_seconds)

    monkeypatch.setattr(market_service, "_shared_cache_set", recording_shared_set)
    monkeypatch.setattr(
        market_service,
        "_fetch_yahoo_quote",
        lambda sym: {"ok": True, "price": 100.0, "prev_close": 99.0, "change": 1.0, "change_pct": 1.0, "currency": "TRY"},
    )
    monkeypatch.setattr(
        market_service,
        "_fetch_yahoo_chart_raw",
        lambda sym, interval="1d", range_="1y": {
            "ok": True,
            "points": [
                {"time": "2026-01-05T00:00:00+00:00", "close": 90.0},
                {"time": "2026-11-10T00:00:00+00:00", "close": 100.0},
            ],
        },
    )

    fx_payload = market_service._market_fx_payload()
    assert len(fx_payload["items"]) > 0
    assert market_service._FX_RETURN_BASES_BULK_KEY in set_keys
    assert not any(k.startswith("api:market:fx-return-bases:") and k != market_service._FX_RETURN_BASES_BULK_KEY for k in set_keys)

    set_keys.clear()
    idx_payload = market_service._market_indices_payload()
    assert len(idx_payload["rows"]) == len(market_service._MARKET_INDEX_ORDER)
    assert market_service._INDEX_RETURN_BASES_BULK_KEY in set_keys
    assert not any(k.startswith("api:market:index-return-bases:") and k != market_service._INDEX_RETURN_BASES_BULK_KEY for k in set_keys)
    cache_module.reset_cache_for_tests()
