from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.api import app
from app import fund_service as fund_service_module
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
    revision = {"snapshot": "generation-1", "price_history": ("2026-10-16T10:00:00Z", 10)}
    monkeypatch.setattr(funds_router, "_funds_listing_payload", lambda **_kwargs: (payload, revision.copy()))
    monkeypatch.setattr(funds_router, "_maybe_schedule_fund_snapshot_refresh", lambda value: value)
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
    revision["snapshot"] = "generation-2"
    changed = client.get(
        "/funds?sort=fund_code&order=asc",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert first.status_code == 200
    assert matching.status_code == 304
    assert matching.content == b""
    assert normalized_query.status_code == 304
    assert different_query.status_code == 200
    assert different_query.headers["etag"] != first.headers["etag"]
    assert changed.status_code == 200
    assert changed.headers["etag"] != first.headers["etag"]


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
