from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from api_module_facade import api_module
from app import cache as cache_module
from app import fund_service as fund_service_module
from app.api import app
from api_test_helpers import _reset_flow_state


def test_shared_swr_fresh_local_entry_skips_shared_cache_read(monkeypatch: pytest.MonkeyPatch) -> None:
    local_entry = api_module._swr_cache_entry(
        {"value": "local"},
        fresh_ttl_seconds=60,
        stale_ttl_seconds=120,
    )
    monkeypatch.setattr(
        api_module,
        "_shared_cache_get_dict",
        lambda _key: pytest.fail("fresh process-local cache should skip shared cache GET"),
    )

    payload, status, stale, pending = api_module._shared_swr_payload(
        cache_key="api:market:test",
        factory=lambda: pytest.fail("fresh cache should skip refresh"),
        fresh_ttl_seconds=60,
        stale_ttl_seconds=120,
        local_cache={"test": local_entry},
        local_key="test",
    )

    assert payload == {"value": "local"}
    assert status == "local_hit"
    assert stale is False
    assert pending is False


def test_cache_single_flight_never_runs_duplicate_factory() -> None:
    calls = 0
    calls_lock = threading.Lock()

    def factory() -> dict[str, str]:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.15)
        return {"value": "ready"}

    def request() -> tuple[dict[str, str] | None, str]:
        return cache_module.get_or_set_single_flight(
            "test:market:quote",
            ttl_seconds=30,
            factory=factory,
            lock_ttl_seconds=2,
            wait_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _index: request(), range(10)))

    assert calls == 1
    assert all(value == {"value": "ready"} for value, _status in results)
    assert any(status == "miss" for _value, status in results)


def test_admin_refresh_auth_distinguishes_missing_secret_and_bad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    monkeypatch.delenv("RAGFIN_ADMIN_REFRESH_TOKEN", raising=False)
    assert client.post("/admin/funds/refresh-snapshot").status_code == 503

    monkeypatch.setenv("RAGFIN_ADMIN_REFRESH_TOKEN", "test-admin-token")
    assert client.post("/admin/funds/refresh-snapshot").status_code == 401
    assert client.post(
        "/admin/funds/refresh-snapshot",
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 401


def test_legacy_admin_refresh_endpoints_require_the_same_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    for path in (
        "/admin/funds/collect-prices",
        "/admin/funds/TLY/refresh-performance?start_date=2026-01-01",
        "/admin/funds/TLY/refresh-allocations",
    ):
        assert client.post(path).status_code == 401

    monkeypatch.setattr(fund_service_module, "collect_daily_fund_prices", lambda *_args, **_kwargs: {"status": "ok"})
    response = client.post(
        "/admin/funds/collect-prices",
        headers={"Authorization": "Bearer test-admin-token"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
