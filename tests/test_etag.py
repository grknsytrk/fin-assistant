from __future__ import annotations

import threading
import time
from typing import Any

from fastapi.testclient import TestClient

from app.api import app
from app import fund_service as fund_service_module
from app import fund_api_service as fund_api_service_module
from app.routers import funds as funds_router
from app.routers import market as market_router
from app.etag import if_none_match_matches


def test_if_none_match_uses_weak_comparison_and_supports_lists() -> None:
    current = 'W/"current-tag"'

    assert if_none_match_matches('"older-tag", "current-tag"', current)
    assert if_none_match_matches('W/"current-tag"', current)
    assert if_none_match_matches("*", current)
    assert not if_none_match_matches('"different-tag"', current)
    assert not if_none_match_matches(None, current)


def test_daily_price_revision_changes_when_return_inputs_change(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    fund_service_module.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "AAA", "date": "2026-10-15", "price": 9.0},
            {"fund_code": "AAA", "date": "2026-10-16", "price": 10.0},
        ],
        source=fund_service_module.FINTABLES_UDF_HISTORY_SOURCE,
        fetched_at="2026-10-16T18:00:00+00:00",
    )
    candidates = [{"fund_code": "AAA", "as_of": "2026-10-16", "source": fund_service_module.TEFAS_LIST_SNAPSHOT_SOURCE}]
    first_overrides, first_revision = fund_service_module._daily_return_overrides_from_price_history(
        tmp_path,
        candidates,
        include_missing=True,
        include_revision=True,
    )

    fund_service_module.upsert_fund_price_points(
        tmp_path,
        [{"fund_code": "AAA", "date": "2026-10-16", "price": 10.5}],
        source=fund_service_module.FINTABLES_UDF_HISTORY_SOURCE,
        fetched_at="2026-10-16T18:00:00+00:00",
    )
    second_overrides, second_revision = fund_service_module._daily_return_overrides_from_price_history(
        tmp_path,
        candidates,
        include_missing=True,
        include_revision=True,
    )

    assert first_overrides[("AAA", "2026-10-16")] == 11.1111
    assert second_overrides[("AAA", "2026-10-16")] == 16.6667
    assert second_revision != first_revision


def test_market_stocks_returns_304_for_matching_etag_and_keeps_refresh(monkeypatch) -> None:
    payload = {"index": "XU100", "rows": [{"company": "AAA", "price": 10.0}], "refresh_pending": False}
    revision = {"source": "generation-1"}
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(market_router, "_market_stocks_payload", lambda **kwargs: calls.append(kwargs) or payload)
    monkeypatch.setattr(market_router, "market_stocks_etag_revision", lambda **_kwargs: revision)
    client = TestClient(app)

    first = client.get("/market/stocks?index=XU100")
    matching = client.get("/market/stocks?index=XU100", headers={"If-None-Match": first.headers["etag"]})
    revision["source"] = "generation-2"
    changed = client.get("/market/stocks?index=XU100", headers={"If-None-Match": first.headers["etag"]})
    refreshing = client.get(
        "/market/stocks?index=XU100&refresh=true",
        headers={"If-None-Match": changed.headers["etag"]},
    )

    assert first.status_code == 200
    assert first.json() == payload
    assert first.headers["cache-control"] == "private, no-cache"
    assert matching.status_code == 304
    assert matching.content == b""
    assert changed.status_code == 200
    assert changed.headers["etag"] != first.headers["etag"]
    assert refreshing.status_code == 304
    assert calls[-1]["force_refresh"] is True


def test_funds_etag_varies_by_query_and_snapshot_revision(monkeypatch) -> None:
    payload = {
        "status": "ok",
        "rows": [{"fund_code": "AAA", "price": 10.0}],
        "count": 1,
        "total_count": 1,
        "stale": False,
        "source_metadata": {"snapshot_as_of": "2026-10-16"},
    }
    revision = {"catalog_generation": 1, "price_generation": 10}
    snapshot = {
        "status": "ok",
        "stale": False,
        "degraded": False,
        "source_metadata": {"snapshot_as_of": "2026-10-16"},
    }
    build_calls: list[dict[str, Any]] = []
    schedule_calls = {"count": 0}
    monkeypatch.setattr(funds_router, "_funds_listing_preflight", lambda: (snapshot.copy(), revision.copy()))
    monkeypatch.setattr(funds_router, "_funds_listing_payload", lambda **kwargs: build_calls.append(kwargs) or payload)

    def schedule(value: dict[str, Any]) -> dict[str, Any]:
        schedule_calls["count"] += 1
        return value

    monkeypatch.setattr(funds_router, "_maybe_schedule_fund_snapshot_refresh", schedule)
    client = TestClient(app)

    first = client.get("/funds?sort=fund_code&order=asc")
    matching = client.get(
        "/funds?sort=fund_code&order=asc",
        headers={"If-None-Match": first.headers["etag"]},
    )
    normalized_query = client.get(
        "/funds?q=%20&sort=unsupported&order=ASC",
        headers={"If-None-Match": first.headers["etag"]},
    )
    different_query = client.get(
        "/funds?sort=name&order=asc",
        headers={"If-None-Match": first.headers["etag"]},
    )
    assert len(build_calls) == 2  # first 200 plus the distinct sort query; matching variants were early 304s
    revision["catalog_generation"] = 2
    changed = client.get(
        "/funds?sort=fund_code&order=asc",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert first.status_code == 200
    assert matching.status_code == 304
    assert matching.content == b""
    assert len(build_calls) == 3  # the changed catalogue revision requires a fresh payload
    assert normalized_query.status_code == 304
    assert different_query.status_code == 200
    assert different_query.headers["etag"] != first.headers["etag"]
    assert changed.status_code == 200
    assert changed.headers["etag"] != first.headers["etag"]
    assert schedule_calls["count"] == 5


def test_fund_listing_revisions_advance_atomically_for_catalog_and_prices(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    before = fund_service_module.fund_listing_revision(tmp_path)

    fund_service_module.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "AAA", "date": "2026-10-15", "price": 9.0},
            {"fund_code": "AAA", "date": "2026-10-16", "price": 10.0},
        ],
        source=fund_service_module.FINTABLES_UDF_HISTORY_SOURCE,
    )
    after_prices = fund_service_module.fund_listing_revision(tmp_path)
    assert after_prices["price_generation"] == before["price_generation"] + 1
    fund_service_module._persist_daily_return_overrides(tmp_path, {("AAA", "2026-10-16"): 11.1})
    after_returns = fund_service_module.fund_listing_revision(tmp_path)
    assert after_returns["price_generation"] == after_prices["price_generation"] + 1

    committed = fund_service_module.commit_funds_snapshot(
        tmp_path,
        {
            "status": "ok",
            "rows": [{"fund_code": "AAA", "name": "A Fon", "price": 10.0, "as_of": "2026-10-16"}],
            "as_of": "2026-10-16",
            "source_metadata": {"source": "test"},
        },
        job_id="local-test",
        owner_token="local-test",
        generation=1,
    )
    after_catalog = fund_service_module.fund_listing_revision(tmp_path)

    assert committed
    assert after_catalog["catalog_generation"] > before["catalog_generation"]
    assert after_catalog["price_generation"] == after_returns["price_generation"]


def test_fund_listing_lru_expires_evicts_and_coalesces(monkeypatch) -> None:
    service = fund_api_service_module
    service._FUND_LISTING_CACHE.clear()
    service._FUND_LISTING_CACHE_BYTES = 0
    service._FUND_LISTING_IN_FLIGHT.clear()
    monkeypatch.setattr(service, "_FUND_LISTING_CACHE_MAX_ENTRIES", 2)
    calls = {"count": 0}

    def build() -> dict[str, Any]:
        calls["count"] += 1
        return {"status": "ok", "rows": [{"fund_code": str(calls["count"])}]}

    first = service._fund_listing_cache_get_or_build("one", build, enabled=True)
    hit = service._fund_listing_cache_get_or_build("one", build, enabled=True)
    assert first == hit
    assert calls["count"] == 1

    # Force expiry without sleeping and make sure the next lookup rebuilds.
    with service._FUND_LISTING_CACHE_LOCK:
        _expires, weight, cached = service._FUND_LISTING_CACHE["one"]
        service._FUND_LISTING_CACHE["one"] = (0.0, weight, cached)
    service._fund_listing_cache_get_or_build("one", build, enabled=True)
    assert calls["count"] == 2

    service._fund_listing_cache_get_or_build("two", build, enabled=True)
    service._fund_listing_cache_get_or_build("three", build, enabled=True)
    assert len(service._FUND_LISTING_CACHE) == 2

    service._FUND_LISTING_CACHE.clear()
    service._FUND_LISTING_CACHE_BYTES = 0
    monkeypatch.setattr(service, "_FUND_LISTING_CACHE_MAX_BYTES", 1)
    service._fund_listing_cache_get_or_build("too-large", build, enabled=True)
    assert not service._FUND_LISTING_CACHE
    pending_calls = {"count": 0}

    def pending() -> dict[str, Any]:
        pending_calls["count"] += 1
        return {"status": "pending", "rows": []}

    service._fund_listing_cache_get_or_build("pending", pending, enabled=True)
    service._fund_listing_cache_get_or_build("pending", pending, enabled=True)
    assert pending_calls["count"] == 2

    service._FUND_LISTING_CACHE.clear()
    service._FUND_LISTING_CACHE_BYTES = 0
    monkeypatch.setattr(service, "_FUND_LISTING_CACHE_MAX_BYTES", 8 * 1024 * 1024)
    calls["count"] = 0
    barrier = threading.Barrier(2)
    results: list[dict[str, Any]] = []

    def slow_build() -> dict[str, Any]:
        calls["count"] += 1
        time.sleep(0.05)
        return {"status": "ok", "rows": []}

    def fetch() -> None:
        barrier.wait()
        results.append(service._fund_listing_cache_get_or_build("same", slow_build, enabled=True))

    threads = [threading.Thread(target=fetch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert calls["count"] == 1
    assert results == [{"status": "ok", "rows": []}] * 2


def test_cors_exposes_etag_and_allows_conditional_request_header(monkeypatch) -> None:
    monkeypatch.setattr(
        market_router,
        "_market_stocks_payload",
        lambda **_kwargs: {"index": "XU100", "rows": []},
    )
    monkeypatch.setattr(market_router, "market_stocks_etag_revision", lambda **_kwargs: {"source": "test"})
    client = TestClient(app)
    preflight = client.options(
        "/market/stocks",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "if-none-match",
        },
    )
    response = client.get("/market/stocks?index=XU100", headers={"Origin": "http://localhost:5173"})
    not_modified = client.get(
        "/market/stocks?index=XU100",
        headers={"Origin": "http://localhost:5173", "If-None-Match": response.headers["etag"]},
    )

    assert preflight.status_code == 200
    assert "if-none-match" in preflight.headers.get("access-control-allow-headers", "").lower()
    assert response.status_code == 200
    assert "etag" in response.headers.get("access-control-expose-headers", "").lower()
    assert not_modified.status_code == 304
    assert "etag" in not_modified.headers.get("access-control-expose-headers", "").lower()
