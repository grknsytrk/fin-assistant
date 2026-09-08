import threading
import time

import pytest

from app import cache


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "background worker did not finish"
        time.sleep(0.005)


@pytest.fixture
def backend(monkeypatch):
    backend = cache.InMemoryCache()
    monkeypatch.setattr(cache, "get_cache", lambda: backend)
    yield backend
    wait_until(lambda: not cache._RESPONSE_REFRESH_ACTIVE)


def background(func):
    return cache.cached_in_background(
        key_fn=lambda: "test:fund", ttl_seconds=45, stale_seconds=60,
        pending_fn=lambda: {"status": "pending", "rows": []},
    )(func)


def test_cold_reads_return_without_waiting_and_share_one_refresh(backend):
    entered, release = threading.Event(), threading.Event()
    calls = []

    @background
    def response():
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return {"status": "ok", "rows": [1]}

    try:
        assert response()["status"] == "pending"
        assert entered.wait(1)
        for _ in range(10):
            assert response()["refresh_pending"] is True
        assert calls == [1]
    finally:
        release.set()
    wait_until(lambda: backend.get("test:fund:stale") is not None)
    assert response() == {"status": "ok", "rows": [1]}


def test_stale_response_keeps_dates_and_cannot_mutate_stored_data(backend):
    old = {"status": "ok", "as_of": "2026-09-04", "rows": [{"price": 1}]}
    backend.set("test:fund:stale", old, ttl_seconds=60)
    release = threading.Event()

    @background
    def response():
        assert release.wait(3)
        return {"status": "ok", "as_of": "2026-09-07", "rows": [{"price": 2}]}

    try:
        result = response()
        assert result["as_of"] == "2026-09-04"
        assert result["response_cache_status"] == "stale"
        result["rows"][0]["price"] = 999
        assert old["rows"][0]["price"] == 1
    finally:
        release.set()
    wait_until(lambda: backend.get("test:fund") is not None)
    assert response()["as_of"] == "2026-09-07"


@pytest.mark.parametrize("raises", [True, False])
def test_failed_refresh_preserves_stale_data_and_backs_off(backend, raises):
    backend.set("test:fund:stale", {"status": "ok", "rows": [1]}, ttl_seconds=60)
    calls = []

    @background
    def response():
        calls.append(1)
        if raises:
            raise RuntimeError("upstream unavailable")
        return {"status": "unavailable", "rows": []}

    assert response()["rows"] == [1]
    wait_until(lambda: backend.get("test:fund:retry") is not None)
    for _ in range(5):
        assert response()["rows"] == [1]
    assert calls == [1]
    assert backend.get("test:fund") is None


def test_another_process_lease_prevents_duplicate_work(backend):
    backend.set("single-flight:test:fund", "other-worker", ttl_seconds=60)

    @background
    def response():
        pytest.fail("must not run while another worker owns the lease")

    assert response()["status"] == "pending"
    wait_until(lambda: not cache._RESPONSE_REFRESH_ACTIVE)
    assert backend.get("single-flight:test:fund") == "other-worker"


def test_worker_capacity_does_not_block_requests(backend):
    slots = cache._RESPONSE_REFRESH_SLOTS
    for _ in range(4):
        assert slots.acquire(blocking=False)
    try:
        @background
        def response():
            pytest.fail("no worker capacity")

        assert response()["status"] == "pending"
        assert not cache._RESPONSE_REFRESH_ACTIVE
    finally:
        for _ in range(4):
            slots.release()
