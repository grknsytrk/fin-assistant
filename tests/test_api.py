from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import api as api_module
from app import cache as cache_module
from app import database as database_module
from app import fund_service as fund_service_module
from app import kap_service as kap_service_module
from app.api import app
from src import kap_vyk_client
from src.nvidia_commentary import NvidiaCommentaryError


@pytest.fixture(autouse=True)
def _reset_flow_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGFIN_CACHE_BACKEND", raising=False)
    monkeypatch.delenv("RAGFIN_REDIS_URL", raising=False)
    monkeypatch.delenv("RAGFIN_CACHE_NAMESPACE", raising=False)
    monkeypatch.setenv("RAGFIN_ADMIN_REFRESH_TOKEN", "test-admin-token")
    cache_module.reset_cache_for_tests()
    api_module._FLOW_CACHE.clear()
    api_module._WATCH_CACHE.clear()
    api_module._WATCH_GLOBAL_CACHE.clear()
    api_module._STOCKS_CACHE.clear()
    api_module._UNIVERSE_CACHE.clear()
    api_module._MARKET_PRICE_CACHE.clear()
    api_module._INFOYATIRIM_STOCK_PAGE_CACHE.clear()
    api_module._STOCK_RETURN_BASE_CACHE.clear()
    api_module._MARKET_STOCK_CARD_CHART_CACHE.clear()
    api_module._STOCK_CARD_VALUATION_CACHE.clear()
    api_module._MARKET_INDICES_CACHE.clear()
    api_module._MARKET_INDEX_DETAIL_CACHE.clear()
    api_module._MARKET_INDEX_QUOTE_CACHE.clear()
    api_module._MARKET_INDEX_INTRADAY_CACHE.clear()
    api_module._MARKET_INDEX_RETURN_CACHE.clear()
    api_module._KAP_MARKET_METADATA_CACHE.clear()
    api_module._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE.clear()
    api_module._FUND_HOLDING_SECTOR_MAP_CACHE.clear()
    api_module._ISYATIRIM_BASIC_SUMMARY_CACHE.clear()
    api_module._GEFAS_GYF_QUOTE_CACHE.clear()
    api_module._FOREIGN_HOLDING_QUOTE_CACHE.clear()
    api_module._FUND_SNAPSHOT_ROW_MAP_CACHE.clear()
    fund_service_module.reset_fund_caches_for_tests()
    kap_service_module._BIST_UNIVERSE_CACHE.clear()
    kap_vyk_client.reset_caches_for_tests()
    yield
    cache_module.reset_cache_for_tests()


def test_api_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["cache_backend"] in {"memory", "redis"}
    assert payload["cache_namespace"]


def test_market_universe_is_metadata_only_and_batch_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda index, force_refresh=False: {
            "index": str(index).upper(),
            "symbols": ["YKBNK", "BIMAS"],
            "count": 2,
            "source": "test",
            "source_url": None,
            "source_date": None,
            "fetched_at": "2026-08-21T10:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(
        api_module,
        "get_instruments",
        lambda *_args, **_kwargs: {
            "YKBNK": {
                "symbol": "YKBNK",
                "name": "Yapı ve Kredi Bankası A.Ş.",
                "source": "reference_data",
                "logo_url": "https://example.test/ykbnk.svg",
                "logo_source": "manual",
                "metadata": {"latest_quarter": "2026Q2", "has_kap_cache": True},
            }
        },
    )
    monkeypatch.setattr(api_module, "_load_cached_kap_market_metadata", lambda *_args: {})
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda *_args, **_kwargs: pytest.fail("universe must not fetch quotes"))
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: pytest.fail("universe must not fetch summaries"))

    payload = api_module._market_universe_payload(index_name="XU100", force_refresh=True)
    rows = {row["symbol"]: row for row in payload["rows"]}

    assert rows["YKBNK"]["name"] == "Yapı ve Kredi Bankası A.Ş."
    assert rows["YKBNK"]["logo_url"] == "https://example.test/ykbnk.svg"
    assert rows["YKBNK"]["latest_quarter"] == "2026Q2"
    assert rows["YKBNK"]["has_kap_cache"] is True
    assert rows["BIMAS"]["name"] == "BIMAS"
    assert rows["BIMAS"]["price"] is None


def test_market_stock_search_uses_metadata_without_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "_market_universe_payload", lambda **_kwargs: {
        "universe": {"fetched_at": "2026-08-21T10:00:00+00:00"},
        "rows": [
            {"symbol": "YKBNK", "company": "YKBNK", "name": "Yapı ve Kredi Bankası A.Ş."},
            {"symbol": "BIMAS", "company": "BIMAS", "name": "BİM Birleşik Mağazalar A.Ş."},
        ],
    })
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda *_args, **_kwargs: pytest.fail("search must not fetch quotes"))

    response = TestClient(app).get("/market/stocks/search", params={"q": "ykbnk", "limit": 20})

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["rows"][0]["symbol"] == "YKBNK"


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


def test_fund_history_requests_share_one_job_and_widen_range(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: List[tuple[Any, ...]] = []

    class FakeExecutor:
        def submit(self, *args: Any, **kwargs: Any) -> None:
            submitted.append(args)

    monkeypatch.setattr(api_module, "_FUND_HISTORY_EXECUTOR", FakeExecutor())

    seven_month_job = api_module._history_start_or_extend_job(
        "THF",
        start_date=date(2026, 1, 22),
        end_date=date(2026, 8, 18),
    )
    one_year_job = api_module._history_start_or_extend_job(
        "THF",
        start_date=date(2025, 8, 22),
        end_date=date(2026, 8, 18),
    )

    assert seven_month_job is not None
    assert one_year_job is not None
    assert one_year_job["job_id"] == seven_month_job["job_id"]
    assert one_year_job["requested_start"] == "2025-08-22"
    assert one_year_job["effective_start"] <= "2025-08-17"
    assert len(submitted) == 1

    running_job = dict(one_year_job)
    running_job["status"] = "running"
    api_module._history_job_set("THF", running_job)
    three_year_job = api_module._history_start_or_extend_job(
        "THF",
        start_date=date(2023, 8, 18),
        end_date=date(2026, 8, 21),
    )

    assert three_year_job is not None
    assert three_year_job["job_id"] == one_year_job["job_id"]
    assert three_year_job["effective_start"] == "2023-08-18"
    assert three_year_job["effective_end"] == "2026-08-21"
    assert three_year_job["extension_requested"] is True
    assert len(submitted) == 1


def test_fund_history_background_job_state_is_friendly_with_existing_points() -> None:
    payload = {
        "points": [{"date": "2026-08-18", "price": 10.0}],
        "source_metadata": {"resolution": "monthly_anchor", "coverage_state": "complete"},
    }
    job = {
        "job_id": "history-1",
        "fund_code": "THF",
        "status": "running",
        "requested_start": "2025-08-22",
        "requested_end": "2026-08-18",
    }

    attached = api_module._history_attach_job(payload, job)

    assert attached["points"] == payload["points"]
    assert attached["source_metadata"]["coverage_state"] == "upgrading"
    assert attached["source_metadata"]["daily_upgrade_state"] == "pending"


def test_failed_fund_history_job_can_be_retried_for_same_range(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: List[tuple[Any, ...]] = []

    class FakeExecutor:
        def submit(self, *args: Any, **kwargs: Any) -> None:
            submitted.append(args)

    monkeypatch.setattr(api_module, "_FUND_HISTORY_EXECUTOR", FakeExecutor())
    failed_job = {
        "job_id": "failed-history-1",
        "fund_code": "RETRYTHF",
        "requested_start": "2026-01-01",
        "requested_end": "2026-08-21",
        "effective_start": "2026-01-01",
        "effective_end": "2026-08-21",
        "status": "failed",
        "daily_upgrade_state": "failed",
    }
    backend = api_module._get_cache()
    api_module._history_job_set("RETRYTHF", failed_job)
    backend.set(api_module._history_last_key("RETRYTHF"), failed_job, ttl_seconds=300)

    retried = api_module._history_start_or_extend_job(
        "RETRYTHF",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 8, 21),
    )

    assert retried is not None
    assert retried["job_id"] != failed_job["job_id"]
    assert retried["status"] == "queued"
    assert len(submitted) == 1


def test_legacy_long_history_job_is_retried_for_fintables_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: List[tuple[Any, ...]] = []

    class FakeExecutor:
        def submit(self, *args: Any, **kwargs: Any) -> None:
            submitted.append(args)

    monkeypatch.setattr(api_module, "_FUND_HISTORY_EXECUTOR", FakeExecutor())
    legacy_job = {
        "job_id": "legacy-long-history",
        "fund_code": "THF",
        "requested_start": "2025-08-21",
        "requested_end": "2026-08-21",
        "effective_start": "2025-08-20",
        "effective_end": "2026-08-21",
        "status": "succeeded",
        "resolution": "mixed",
        "coverage_state": "complete",
        "daily_upgrade_state": "unavailable",
    }
    payload = {
        "points": [{"date": "2025-08-21", "price": 1.0}],
        "source_metadata": {
            "coverage_state": "complete",
            "resolution": "mixed",
            "internal_gap_count": 0,
        },
    }
    backend = api_module._get_cache()
    api_module._history_job_set("THF", legacy_job)
    backend.set(api_module._history_last_key("THF"), legacy_job, ttl_seconds=300)

    assert api_module._history_job_should_schedule(
        payload,
        last_job=legacy_job,
        target=api_module._history_request_range(date(2025, 8, 21), date(2026, 8, 21)),
    ) is True
    retried = api_module._history_start_or_extend_job(
        "THF",
        start_date=date(2025, 8, 21),
        end_date=date(2026, 8, 21),
        force_new=True,
    )

    assert retried is not None
    assert retried["job_id"] != legacy_job["job_id"]
    assert len(submitted) == 1


def test_fund_performance_returns_local_points_and_queues_background_history_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    submitted: List[tuple[Any, ...]] = []

    def fake_performance(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        assert kwargs["auto_refresh"] is False
        return {
            "fund_code": fund_code,
            "status": "ok",
            "points": [{"date": "2026-08-18", "price": 10.0}],
            "period_stats": {},
            "source_metadata": {
                "resolution": "monthly_anchor",
                "coverage_state": "complete",
                "available_start_date": "2025-11-28",
                "available_end_date": "2026-08-18",
                "internal_gap_count": 0,
            },
        }

    class FakeExecutor:
        def submit(self, *args: Any, **kwargs: Any) -> None:
            submitted.append(args)

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_performance)
    monkeypatch.setattr(api_module, "_FUND_HISTORY_EXECUTOR", FakeExecutor())

    response = TestClient(app).get(
        "/funds/THF/performance",
        params={"start_date": "2026-01-22", "end_date": "2026-08-18"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["points"][0]["price"] == 10.0
    assert payload["source_metadata"]["coverage_state"] == "upgrading"
    assert payload["source_metadata"]["history_job"]["status"] == "queued"
    assert len(calls) == 1
    assert len(submitted) == 1


def test_single_fund_history_invalidation_clears_versioned_performance_cache() -> None:
    backend = cache_module.get_cache()
    backend.set("api:fund-performance:v2:THF:2026-01-22:2026-08-18:fb=0:refresh=0", {"stale": True})

    api_module._invalidate_single_fund_response_cache("THF")

    assert backend.get("api:fund-performance:v2:THF:2026-01-22:2026-08-18:fb=0:refresh=0") is None


def test_market_stocks_serves_stale_quote_when_upstream_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_entry = {
        "payload": {"index": "XUTUM", "rows": [{"company": "YKBNK", "price": 25.0}], "benchmarks": {}, "as_of": "old"},
        "fresh_until": time.time() - 10,
        "stale_until": time.time() + 120,
    }
    backend = cache_module.get_cache()
    backend.set("api:market:stocks:XUTUM:v2", stale_entry, ttl_seconds=120)
    monkeypatch.setattr(api_module, "_build_market_stocks_payload", lambda **_kwargs: {
        "index": "XUTUM",
        "rows": [{"company": "YKBNK", "price": None}],
        "benchmarks": {},
        "as_of": "new",
    })

    payload = api_module._market_stocks_payload(index_name="XUTUM", force_refresh=True)

    assert payload["stale"] is True
    assert payload["quote_status"] == "stale"
    assert payload["rows"][0]["price"] == 25.0


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


def test_fenced_snapshot_commit_checks_ownership_before_upsert(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(database_module, "database_enabled", lambda: True)
    monkeypatch.setattr(database_module, "ensure_refresh_lease_schema", lambda: None)

    class Result:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self, lease):
            self.lease = lease
            self.queries = []
            self.commits = 0
            self.rollbacks = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, _params=()):
            self.queries.append(query)
            if "SELECT generation" in query:
                return Result(self.lease)
            if "INSERT INTO ragfin_json_cache" in query:
                return Result({"version": 7})
            raise AssertionError(f"unexpected query: {query}")

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    owner = {"generation": 7, "job_id": "job-a", "owner_token": "token-a", "lease_until": "future"}
    connection = Connection(owner)
    monkeypatch.setattr(database_module, "connect_postgres", lambda: connection)

    assert database_module.commit_json_cache_if_fenced(
        tmp_path / "funds_snapshot.json",
        {"rows": []},
        resource_key="funds_snapshot",
        job_id="job-a",
        owner_token="token-a",
        generation=7,
    ) is True
    assert len([query for query in connection.queries if "INSERT INTO ragfin_json_cache" in query]) == 1
    assert connection.commits == 1

    connection.lease = {**owner, "owner_token": "token-b"}
    assert database_module.commit_json_cache_if_fenced(
        tmp_path / "funds_snapshot.json",
        {"rows": ["late"]},
        resource_key="funds_snapshot",
        job_id="job-a",
        owner_token="token-a",
        generation=7,
    ) is False
    assert len([query for query in connection.queries if "INSERT INTO ragfin_json_cache" in query]) == 1


def test_kap_snapshot_response_cache_uses_schema_and_refresh_bypasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.kap_fetcher import KAP_CACHE_SCHEMA_VERSION

    calls: List[bool] = []

    def fake_get_kap_snapshot(**kwargs: Any) -> Dict[str, Any]:
        calls.append(bool(kwargs.get("force_refresh")))
        return {
            "ok": True,
            "stock_code": str(kwargs.get("company") or "").upper(),
            "company_title": "Akbank",
            "quarters": [],
        }

    def fake_normalize_snapshot(raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stock_code": raw["stock_code"],
            "latest_quarter": "2026/3",
            "source_metadata": {"source": "kap"},
        }

    monkeypatch.setattr(kap_service_module, "get_kap_snapshot", fake_get_kap_snapshot)
    monkeypatch.setattr(kap_service_module, "normalize_snapshot_for_frontend", fake_normalize_snapshot)
    monkeypatch.setattr(api_module, "_upsert_stock_reference_from_kap_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_module, "_fetch_kap_price_payload", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_fetch_isyatirim_multiples", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_build_kap_valuation_payload", lambda **_kwargs: {"ok": True})
    client = TestClient(app)

    first = client.get("/kap/snapshot", params={"company": "AKBNK", "max_quarters": 20})
    second = client.get("/kap/snapshot", params={"company": "AKBNK", "max_quarters": 20})
    refreshed = client.get("/kap/snapshot", params={"company": "AKBNK", "max_quarters": 20, "refresh": "true"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert refreshed.status_code == 200
    assert first.json()["response_cache_hit"] is False
    assert second.json()["response_cache_hit"] is True
    assert refreshed.json()["response_cache_hit"] is False
    assert calls == [False, True]
    assert f"schema={KAP_CACHE_SCHEMA_VERSION}" in api_module._kap_snapshot_response_cache_key("AKBNK", 20)


def test_kap_snapshot_normalization_exposes_identity_company_kind() -> None:
    payload = kap_service_module.normalize_snapshot_for_frontend(
        {
            "ok": True,
            "company": "TUPRS",
            "stock_code": "TUPRS",
            "company_title": "TÜPRAŞ-TÜRKİYE PETROL RAFİNERİLERİ A.Ş.",
            "quarters": [],
        }
    )

    assert payload["company_kind"] == "generic"


def test_api_funds_list_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_kwargs: Dict[str, Any] = {}

    def fake_get_funds_payload(processed_dir: Any, **kwargs: Any) -> Dict[str, Any]:
        seen_kwargs.update(kwargs)
        return {
            "status": "ok",
            "rows": [
                {
                    "fund_code": "YAC",
                    "name": "Yatirim Fonu",
                    "fund_type": "Hisse Senedi",
                    "price": 1.23,
                    "daily_return": 0.5,
                    "period_returns": {},
                    "risk_value": 5,
                    "currency": "TRY",
                    "as_of": "2026-04-28",
                    "source": "tefasfon_funds",
                }
            ],
            "count": 1,
            "total_count": 1,
            "source": "tefasfon_funds",
            "as_of": "2026-04-28",
            "fetched_at": "2026-04-28T09:00:00+00:00",
            "stale": False,
            "degraded": False,
            "warnings": [],
            "source_metadata": {"source": "tefasfon_funds", "parse_status": "ok"},
        }

    monkeypatch.setattr(fund_service_module, "get_funds_payload", fake_get_funds_payload)
    client = TestClient(app)

    response = client.get("/funds")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"][0]["fund_code"] == "YAC"
    assert payload["source"] == "tefasfon_funds"
    assert seen_kwargs["auto_refresh"] is False


def test_api_funds_search_keeps_full_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_kwargs: Dict[str, Any] = {}

    def fake_get_funds_payload(processed_dir: Any, **kwargs: Any) -> Dict[str, Any]:
        seen_kwargs.update(kwargs)
        return {
            "status": "ok",
            "rows": [
                {"fund_code": "LOW", "name": "Kucuk Fon", "aum": 100_000_000},
                {"fund_code": "BIG", "name": "Buyuk Fon", "aum": 500_000_000},
            ],
            "count": 2,
            "total_count": 2,
            "source": "tefasfon_funds",
            "as_of": "2026-05-20",
            "fetched_at": "2026-05-20T09:00:00+00:00",
            "stale": False,
            "degraded": False,
            "warnings": [],
            "source_metadata": {"source": "tefasfon_funds", "parse_status": "ok"},
        }

    monkeypatch.setattr(fund_service_module, "get_funds_payload", fake_get_funds_payload)
    client = TestClient(app)

    response = client.get("/funds/search", params={"q": "fon", "limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert seen_kwargs["min_aum"] is None


def test_api_fund_yield_summary_uses_response_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_payload(fund_code: str, *, processed_dir: Any) -> Dict[str, Any]:
        calls["count"] += 1
        return {
            "fund_code": fund_code,
            "status": "ok",
            "items": [{"period": "1A", "return_pct": 1.23}],
            "source_metadata": {"call": calls["count"]},
        }

    monkeypatch.setattr(fund_service_module, "get_fund_yield_summary_payload", fake_payload)
    client = TestClient(app)

    first = client.get("/funds/TLY/yield-summary")
    second = client.get("/funds/TLY/yield-summary")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["source_metadata"]["call"] == 1
    assert second.json()["source_metadata"]["call"] == 1
    assert calls["count"] == 1


def test_api_fund_holdings_reuses_static_response_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_holdings(_processed_dir: Any, fund_code: str) -> Dict[str, Any]:
        calls["count"] += 1
        return {
            "fund_code": fund_code,
            "status": "ok",
            "positions": [],
            "source_metadata": {"call": calls["count"]},
        }

    monkeypatch.setattr(fund_service_module, "get_fund_holdings_payload", fake_holdings)
    monkeypatch.setattr(
        api_module,
        "_enrich_fund_holdings_with_daily_market_data",
        lambda payload, fund_code: {**payload, "enriched_for": fund_code},
    )
    client = TestClient(app)

    first = client.get("/funds/TLY/holdings")
    second = client.get("/funds/TLY/holdings")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["source_metadata"]["call"] == 1
    assert second.json()["source_metadata"]["call"] == 1
    assert calls["count"] == 1


def test_admin_funds_refresh_invalidates_response_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cache_module.get_cache()
    backend.set("api:funds:q=|type=|founder=|manager=|risk=|sort=fund_code|order=asc", {"stale": True}, ttl_seconds=60)
    backend.set("api:funds:v2:q=|type=|founder=|manager=|risk=|sort=fund_code|order=asc", {"stale": True}, ttl_seconds=60)
    backend.set("api:funds-search:q=", {"stale": True}, ttl_seconds=60)
    backend.set("api:funds-categories", {"stale": True}, ttl_seconds=60)
    backend.set("api:funds-categories:v2", {"stale": True}, ttl_seconds=60)
    backend.set("api:fund-yield-summary:TLY", {"stale": True}, ttl_seconds=60)
    backend.set("api:fund-holdings:TLY", {"stale": True}, ttl_seconds=60)

    monkeypatch.setattr(
        fund_service_module,
        "refresh_funds_snapshot",
        lambda _processed_dir, *, lookback_days: {
            "status": "ok",
            "rows": [{"fund_code": "TLY"}],
            "stale": False,
            "degraded": False,
            "as_of": "2026-05-20",
            "lookback_days": lookback_days,
        },
    )
    monkeypatch.setattr(
        fund_service_module,
        "get_funds_payload",
        lambda _processed_dir, **kwargs: {
            "status": "ok",
            "rows": [{"fund_code": "TLY"}],
            "count": 1,
            "total_count": 1,
            "source": "tefasfon_funds",
            "as_of": "2026-05-20",
            "fetched_at": "2026-05-20T09:00:00+00:00",
            "stale": False,
            "degraded": False,
            "warnings": [],
            "source_metadata": {"source": "tefasfon_funds", "list_min_aum": 0},
        },
    )
    job = {
        "job_id": "test-refresh-job",
        "status": "queued",
        "requested_at": "2026-05-20T09:00:00+00:00",
        "started_at": None,
        "finished_at": None,
        "as_of": None,
        "row_count": None,
        "error": None,
    }
    api_module._set_fund_refresh_job(job)
    backend.set(api_module._FUND_REFRESH_ACTIVE_KEY, job["job_id"], ttl_seconds=600)
    monkeypatch.setattr(api_module, "_start_fund_refresh_job", lambda _lookback_days: job)
    client = TestClient(app)

    response = client.post(
        "/admin/funds/refresh-snapshot",
        params={"lookback_days": 1},
        headers={"Authorization": "Bearer test-admin-token"},
    )

    assert response.status_code == 200
    assert response.json()["rows"][0]["fund_code"] == "TLY"
    assert response.json()["source_metadata"]["list_min_aum"] == 0
    assert response.json()["refresh_job"]["job_id"] == "test-refresh-job"
    assert backend.get("api:funds:q=|type=|founder=|manager=|risk=|sort=fund_code|order=asc") == {"stale": True}

    api_module._run_fund_refresh_job("test-refresh-job", 1)

    assert backend.get("api:funds:q=|type=|founder=|manager=|risk=|sort=fund_code|order=asc") is None
    assert backend.get("api:funds:v2:q=|type=|founder=|manager=|risk=|sort=fund_code|order=asc") is None
    assert backend.get("api:funds-search:q=") is None
    assert backend.get("api:funds-categories") is None
    assert backend.get("api:funds-categories:v2") is None
    assert backend.get("api:fund-yield-summary:TLY") is None
    assert backend.get("api:fund-holdings:TLY") is None


def test_admin_funds_refresh_reuses_active_job_and_exposes_status(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cache_module.get_cache()
    job = {
        "job_id": "active-refresh-job",
        "status": "running",
        "requested_at": "2026-05-20T09:00:00+00:00",
        "started_at": "2026-05-20T09:00:01+00:00",
        "finished_at": None,
        "as_of": None,
        "row_count": None,
        "error": None,
    }
    api_module._set_fund_refresh_job(job)
    backend.set(api_module._FUND_REFRESH_ACTIVE_KEY, job["job_id"], ttl_seconds=600)
    monkeypatch.setattr(api_module, "_start_fund_refresh_job", lambda _lookback_days: job)
    monkeypatch.setattr(
        fund_service_module,
        "get_funds_payload",
        lambda _processed_dir, **_kwargs: {
            "status": "ok",
            "rows": [],
            "count": 0,
            "total_count": 0,
            "source": "tefasfon_funds",
            "as_of": "2026-05-20",
            "fetched_at": "2026-05-20T09:00:00+00:00",
            "stale": True,
            "degraded": False,
            "warnings": [],
            "source_metadata": {"source": "tefasfon_funds"},
        },
    )

    client = TestClient(app)
    headers = {"Authorization": "Bearer test-admin-token"}
    response = client.post("/admin/funds/refresh-snapshot", headers=headers)
    status = client.get(
        "/admin/funds/refresh-snapshot/status",
        params={"job_id": job["job_id"]},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["refresh_job"]["status"] == "running"
    assert status.status_code == 200
    assert status.json()["refresh_job"]["job_id"] == job["job_id"]


def _patch_holdings_cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(
        fund_service_module,
        "_holdings_path",
        lambda _processed_dir, fund_code: tmp_path / f"{fund_code}.json",
    )
    monkeypatch.setattr(
        fund_service_module,
        "_holdings_attachment_text_path",
        lambda _processed_dir, disclosure_index, obj_id: tmp_path / f"attachment_{disclosure_index}_{obj_id}.json",
    )
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_fetch_gefas_gyf_quote", lambda _symbol: None)
    monkeypatch.setattr(
        api_module,
        "_fund_holding_sector_map",
        lambda: ({}, {"cache_hit": False, "symbol_count": 0, "source": None, "source_date": None, "warnings": []}),
    )


def test_api_fund_holdings_unavailable_when_kap_fund_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(fund_service_module, "_kap_search_fund_metadata", lambda _fund_code: None)
    client = TestClient(app)

    response = client.get("/funds/YAC/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["positions"] == []
    assert payload["source_metadata"]["parse_status"] == "unavailable"


def test_api_fund_holdings_parses_kap_reports_and_deltas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    latest_text = """
FON PORTFÖY DEĞERİ TABLOSU
Hisse Senedi
DSTKF Destek Finans Faktoring A.S. 1.000,00 10,00 10.000,00 23,60TRYTRADSTKFXXX
TERA Tera Yatirim Menkul Degerler A.S. 2.000,00 5,00 20.000,00 12,86TRYTRATERAXXX
BIMAS Bim Birlesik Magazalar A.S. 100,00 5,00 500,00 1,25TRYTRABIMASXXX
"""
    previous_text = """
FON PORTFÖY DEĞERİ TABLOSU
Hisse Senedi
DSTKF Destek Finans Faktoring A.S. 1.000,00 10,00 10.000,00 20,45TRYTRADSTKFXXX
TERA Tera Yatirim Menkul Degerler A.S. 2.000,00 5,00 20.000,00 24,31TRYTRATERAXXX
AKBNK Akbank T.A.S. 300,00 2,00 600,00 3,00TRYTRAAKBNKXXX
"""

    def fake_detail(disclosure_index: int) -> Dict[str, Any]:
        if disclosure_index == 200:
            return {
                "disclosure": {"disclosureBasic": {"disclosureIndex": 200, "year": 2026, "donem": 4}},
                "attachments": [{"objId": "latest", "fileName": "TST_2026.04.pdf", "fileExtension": "pdf"}],
            }
        return {
            "disclosure": {"disclosureBasic": {"disclosureIndex": 100, "year": 2026, "donem": 3}},
            "attachments": [{"objId": "previous", "fileName": "TST_2026.03.pdf", "fileExtension": "pdf"}],
        }

    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "TST", "fund_oid": "fund-oid", "fund_name": "Test Fon"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "01.05.2026 10:00:00"}},
            {"disclosureBasic": {"disclosureIndex": 100, "publishDate": "01.04.2026 10:00:00"}},
        ],
    )
    monkeypatch.setattr(fund_service_module, "_kap_fetch_report_detail", fake_detail)
    monkeypatch.setattr(fund_service_module, "_kap_download_attachment", lambda obj_id: str(obj_id).encode("utf-8"))
    monkeypatch.setattr(
        fund_service_module,
        "_extract_kap_pdf_text",
        lambda data: latest_text if data == b"latest" else previous_text,
    )
    client = TestClient(app)

    response = client.get("/funds/TST/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    positions = {item["asset_code"]: item for item in payload["positions"]}
    assert positions["DSTKF"]["weight"] == 23.6
    assert positions["DSTKF"]["previous_weight"] == 20.45
    assert positions["DSTKF"]["weight_change"] == pytest.approx(3.15)
    assert positions["DSTKF"]["change_status"] == "increased"
    assert positions["TERA"]["change_status"] == "decreased"
    assert positions["BIMAS"]["change_status"] == "new"
    assert positions["AKBNK"]["change_status"] == "removed"
    assert payload["source_metadata"]["latest_report"]["file_name"] == "TST_2026.04.pdf"


def test_api_fund_holdings_adds_optional_sector_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 23.6,
                "previous_weight": None,
                "weight_change": None,
                "change_status": "unchanged",
                "amount": None,
                "market_value": None,
                "report_date": "2026-04-30",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.9,
            },
            {
                "fund_code": "TLY",
                "asset_code": "BIMAS",
                "asset_name": "Bilinmeyen Hisse",
                "asset_type": "local_equity",
                "weight": 1.0,
                "previous_weight": None,
                "weight_change": None,
                "change_status": "unchanged",
                "amount": None,
                "market_value": None,
                "report_date": "2026-04-30",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.9,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "parse_status": "ok",
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(
        api_module,
        "_fund_holding_sector_map",
        lambda: (
            {"DSTKF": {"sector_code": "XFINK", "sector_label": "Finansal Kiralama Faktoring"}},
            {"cache_hit": False, "symbol_count": 1, "source": "test", "source_date": "2026-04-30", "warnings": []},
        ),
    )
    client = TestClient(app)

    response = client.get("/funds/TLY/holdings")

    assert response.status_code == 200
    positions = {item["asset_code"]: item for item in response.json()["positions"]}
    assert positions["DSTKF"]["sector_code"] == "XFINK"
    assert positions["DSTKF"]["sector_label"] == "Finansal Kiralama Faktoring"

    direct_payload = dict(cache_payload)
    direct_payload["positions"] = [cache_payload["positions"][1]]
    direct = api_module._enrich_fund_holdings_with_daily_market_data(direct_payload, "TLY")
    assert direct["positions"][0]["sector_code"] is None
    assert direct["positions"][0]["sector_label"] is None


def test_api_fund_holdings_reuses_monthly_report_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 23.63,
                "previous_weight": 20.43,
                "weight_change": 3.2,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": False,
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "latest_report": {"disclosure_index": 1601574, "report_date": "2026-04-30", "source_url": "https://www.kap.org.tr/tr/Bildirim/1601574"},
            "previous_report": {"disclosure_index": 1583104, "report_date": "2026-03-31", "source_url": "https://www.kap.org.tr/tr/Bildirim/1583104"},
            "disclosure_check": {"checked_at": "2026-05-19T00:00:00+00:00", "ttl_seconds": 21600},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 19, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "refresh_fund_holdings",
        lambda *_args, **_kwargs: pytest.fail("monthly holdings cache should be reused"),
    )
    client = TestClient(app)

    response = client.get("/funds/TLY/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["positions"][0]["asset_code"] == "DSTKF"
    assert payload["source_metadata"]["cache_hit"] is True
    assert payload["source_metadata"]["cache_policy"] == "monthly_report"


def test_fund_holdings_disclosure_check_skips_pdf_when_report_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    positions = [
        {
            "fund_code": "TLY",
            "asset_code": "DSTKF",
            "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
            "asset_type": "local_equity",
            "weight": 23.63,
            "previous_weight": 20.43,
            "weight_change": 3.2,
            "change_status": "increased",
            "report_date": "2026-04-30",
            "previous_report_date": "2026-03-31",
            "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
            "source_type": "kap_pdf",
            "parse_confidence": 0.82,
        }
    ]
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": positions,
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "latest_report": {"disclosure_index": 1601574, "report_date": "2026-04-30"},
            "previous_report": {"disclosure_index": 1583104, "report_date": "2026-03-31"},
            "disclosure_check": {"checked_at": "2026-05-01T00:00:00+00:00", "ttl_seconds": 21600},
            "positions_hash": fund_service_module._holdings_positions_hash(positions),
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "TLY", "fund_oid": "fund-oid", "fund_name": "TLY"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 1601574, "publishDate": "06.05.2026 09:00:00"}},
            {"disclosureBasic": {"disclosureIndex": 1583104, "publishDate": "05.04.2026 09:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_fetch_report_detail",
        lambda _disclosure_index: pytest.fail("unchanged disclosure should not fetch attachment detail"),
    )

    payload = fund_service_module.get_fund_holdings_payload(tmp_path, "TLY")

    assert payload["positions"][0]["asset_code"] == "DSTKF"
    assert payload["source_metadata"]["cache_hit"] is True
    assert payload["source_metadata"]["static_cache_hit"] is True
    assert payload["source_metadata"]["disclosure_check"]["latest_disclosure_index"] == 1601574


def test_fund_holdings_parser_version_reuses_cached_attachment_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    (tmp_path / "TLY.json").write_text(
        json.dumps(
            {
                "fund_code": "TLY",
                "status": "ok",
                "positions": [],
                "source": "kap_portfolio_allocation_report",
                "message": None,
                "source_metadata": {
                    "fetched_at": "2026-05-01T00:00:00+00:00",
                    "parse_status": "ok",
                    "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION - 1,
                    "latest_report": {"disclosure_index": 1601574, "report_date": "2026-04-30"},
                },
            }
        ),
        encoding="utf-8",
    )
    text_cache_path = tmp_path / "attachment_text.json"
    text_cache_path.write_text(
        json.dumps(
            {
                "schema_version": fund_service_module.KAP_HOLDINGS_ATTACHMENT_TEXT_CACHE_VERSION,
                "disclosure_index": 1601574,
                "obj_id": "pdf-obj",
                "file_name": "TLY_2026.04.pdf",
                "fetched_at": "2026-05-06T09:00:00+00:00",
                "text": """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
AKBNK AKBANK T.A.Ş. 18.653.248,00 69,718033 30/04/26 73,200000 1.365.417.753,60 4,66 3,67TL 80100511 3,67TRAAKBNK91N6
""",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "TLY", "fund_oid": "fund-oid", "fund_name": "TLY"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 1601574, "publishDate": "06.05.2026 09:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_fetch_report_detail",
        lambda _disclosure_index: {
            "disclosureBasic": {"disclosureIndex": 1601574, "year": 2026, "donem": 4},
            "attachments": [{"fileExtension": "pdf", "objId": "pdf-obj", "fileName": "TLY_2026.04.pdf"}],
        },
    )
    monkeypatch.setattr(
        fund_service_module,
        "_holdings_attachment_text_path",
        lambda _processed_dir, _disclosure_index, _obj_id: text_cache_path,
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_download_attachment",
        lambda _obj_id: pytest.fail("cached attachment text should avoid PDF download"),
    )

    payload = fund_service_module.get_fund_holdings_payload(tmp_path, "TLY")

    assert payload["source_metadata"]["latest_report"]["attachment_text_cache_hit"] is True
    assert payload["positions"][0]["asset_code"] == "AKBNK"
    assert payload["source_metadata"]["parser_version"] == fund_service_module.KAP_HOLDINGS_PARSE_VERSION


def test_api_fund_holdings_enriches_daily_market_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "TLY",
                "asset_code": "PKZ",
                "asset_name": "PUSULA PORTFÖY KUZEY HİSSE SENEDİ SERBEST FON",
                "asset_type": "fund",
                "weight": 10.0,
                "previous_weight": 8.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "TLY",
                "asset_code": "MISS",
                "asset_name": "Eksik Hisse",
                "asset_type": "local_equity",
                "weight": 5.0,
                "previous_weight": 4.0,
                "weight_change": 1.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "TLY",
                "asset_code": "OLD",
                "asset_name": "Çıkan Hisse",
                "asset_type": "local_equity",
                "weight": 0.0,
                "previous_weight": 2.0,
                "weight_change": -2.0,
                "change_status": "removed",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": False,
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:00:00+00:00", "ttl_seconds": 21600},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))

    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols, **_kwargs: {
            "DSTKF": {"price": 40.0, "currency": "TRY", "change_pct": 2.5, "as_of": "2026-05-20T10:00:00+00:00"},
            "OLD": {"price": 9.0, "currency": "TRY", "change_pct": 10.0, "as_of": "2026-05-20T10:00:00+00:00"},
        },
    )
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {
            "rows": [
                {"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0},
                {"fund_code": "PKZ", "as_of": "2026-05-20", "price": 12.34, "daily_return": 1.2},
            ]
        },
    )
    client = TestClient(app)

    response = client.get("/funds/TLY/holdings")

    assert response.status_code == 200
    payload = response.json()
    positions = {item["asset_code"]: item for item in payload["positions"]}
    assert positions["DSTKF"]["price"] == 40.0
    assert positions["DSTKF"]["return_pct"] == 2.5
    assert positions["DSTKF"]["estimated_exposure_value"] == 20_000_000.0
    assert positions["DSTKF"]["estimated_pnl_value"] == 500_000.0
    assert positions["DSTKF"]["estimated_fund_return_contribution_pct"] == 0.5
    assert positions["PKZ"]["price"] == 12.34
    assert positions["PKZ"]["return_pct"] == 1.2
    assert positions["PKZ"]["estimated_pnl_value"] == 120_000.0
    assert positions["OLD"]["estimated_fund_return_contribution_pct"] is None
    assert payload["portfolio_effect"]["estimated_return_pct"] == pytest.approx(0.62)
    assert payload["portfolio_effect"]["estimated_pnl_value"] == 620_000.0
    assert payload["portfolio_effect"]["priced_weight"] == 30.0
    assert payload["portfolio_effect"]["missing_weight"] == 5.0


def test_api_fund_holdings_enriches_foreign_assets_with_yahoo_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"GTZ": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(api_module, "_quote_map_for_holding_stocks", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))

    def fake_yahoo_quote(symbol: str) -> Dict[str, Any]:
        if symbol == "SIVR":
            return {
                "ok": True,
                "price": 71.5,
                "currency": "USD",
                "change_pct": 1.4,
                "as_of": "2026-05-20T10:00:00+00:00",
                "long_name": "abrdn Physical Silver Shares ETF",
            }
        return {"ok": False, "error": "not_found"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_yahoo_quote)
    payload = {
        "fund_code": "GTZ",
        "status": "ok",
        "positions": [
            {
                "asset_code": "SIVRUS",
                "asset_name": "ABERDEEN STANDARD PHYSICAL SILVER SHARES ETF",
                "asset_type": "foreign_fund",
                "asset_region": "foreign",
                "provider_symbol": "SIVR",
                "weight": 10.0,
                "previous_weight": 8.0,
            },
            {
                "asset_code": "MISSUS",
                "asset_name": "Çözülemeyen Yabancı Hisse",
                "asset_type": "foreign_equity",
                "asset_region": "foreign",
                "provider_symbol": "MISS",
                "weight": 5.0,
                "previous_weight": 4.0,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "GTZ")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["SIVRUS"]["price"] == 71.5
    assert positions["SIVRUS"]["price_currency"] == "USD"
    assert positions["SIVRUS"]["return_pct"] == 1.4
    assert positions["SIVRUS"]["return_source"] == "yahoo_finance_chart"
    assert positions["SIVRUS"]["provider_name"] == "abrdn Physical Silver Shares ETF"
    assert positions["SIVRUS"]["asset_name"] == "abrdn Physical Silver Shares ETF"
    assert positions["SIVRUS"]["tefas_tradable"] is False
    assert positions["SIVRUS"]["detail_clickable"] is False
    assert positions["MISSUS"]["return_pct"] is None
    assert positions["MISSUS"]["return_source"] is None
    assert positions["MISSUS"]["detail_clickable"] is False
    assert enriched["portfolio_effect"]["estimated_return_pct"] == pytest.approx(0.14)
    assert enriched["portfolio_effect"]["priced_weight"] == 10.0
    assert enriched["portfolio_effect"]["missing_weight"] == 5.0
    enrichment_meta = enriched["source_metadata"]["daily_market_enrichment"]
    assert enrichment_meta["foreign_quote_count"] == 1
    assert enrichment_meta["foreign_quote_missing_count"] == 1


def test_api_fund_holdings_marks_inner_funds_as_tefas_tradable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Inner fund-type positions must expose ``tefas_tradable`` so the
    frontend can decide whether to make the row clickable."""
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "IIE",
        "status": "ok",
        "positions": [
            {
                "fund_code": "IIE",
                "asset_code": "PKZ",
                "asset_name": "PUSULA PORTFÖY KUZEY HİSSE SENEDİ SERBEST FON",
                "asset_type": "fund",
                "weight": 5.0,
                "previous_weight": 4.0,
                "weight_change": 1.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "IIE",
                "asset_code": "ITH",
                "asset_name": "İSTANBUL PORTFÖY YÖNETİMİ A.Ş. ONE LIFE VENTURES",
                "asset_type": "fund",
                "weight": 2.5,
                "previous_weight": 2.85,
                "weight_change": -0.35,
                "change_status": "decreased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "IIE",
                "asset_code": "BSOKE",
                "asset_name": "BATISÖKE SÖKE ÇİMENTO SANAYİİ T.A.Ş.",
                "asset_type": "local_equity",
                "weight": 71.0,
                "previous_weight": 58.2,
                "weight_change": 12.8,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": False,
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:00:00+00:00", "ttl_seconds": 21600},
            "warnings": [],
        },
    }
    (tmp_path / "IIE.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda symbols, **_kwargs: {})
    # PKZ is in the TEFAS-open snapshot, ITH is not.
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {
            "rows": [
                {"fund_code": "IIE", "as_of": "2026-05-20", "aum": 100_000_000.0},
                {
                    "fund_code": "PKZ",
                    "as_of": "2026-05-20",
                    "price": 12.34,
                    "daily_return": 1.2,
                    "founder_company": "PUSULA PORTFÖY",
                },
            ]
        },
    )
    client = TestClient(app)

    response = client.get("/funds/IIE/holdings")

    assert response.status_code == 200
    payload = response.json()
    positions = {item["asset_code"]: item for item in payload["positions"]}
    # PKZ is in the TEFAS funds snapshot → tradable.
    assert positions["PKZ"]["tefas_tradable"] is True
    assert positions["PKZ"]["provider_name"] == "PUSULA PORTFÖY"
    assert positions["PKZ"]["logo_symbol"] == "PUSULA PORTFÖY"
    # ITH is not in the snapshot → not tradable.
    assert positions["ITH"]["tefas_tradable"] is False
    # Stock positions must not carry the flag.
    assert positions["BSOKE"]["tefas_tradable"] is None


def test_api_fund_holdings_keeps_leveraged_weights_in_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {
            "GOOD": {"price": 20.0, "currency": "TRY", "change_pct": 2.5, "as_of": "2026-05-20T10:00:00+00:00"},
            "LEVR": {"price": 10.0, "currency": "TRY", "change_pct": 10.0, "as_of": "2026-05-20T10:00:00+00:00"},
        },
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "LEVR", "asset_name": "Kaldıraçlı Pozisyon", "asset_type": "local_equity", "weight": 127.5, "previous_weight": 45.2},
            {"asset_code": "GOOD", "asset_name": "Geçerli", "asset_type": "local_equity", "weight": 20.0, "previous_weight": 18.0},
            {"asset_code": "MISS", "asset_name": "Eksik", "asset_type": "local_equity", "weight": 5.0, "previous_weight": 4.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert enriched["status"] == "ok"
    assert positions["LEVR"]["weight"] == 127.5
    assert positions["LEVR"]["weight_quality"] == "ok"
    assert positions["LEVR"]["estimated_fund_return_contribution_pct"] == pytest.approx(12.75)
    assert positions["GOOD"]["estimated_fund_return_contribution_pct"] == 0.5
    assert enriched["portfolio_effect"]["estimated_return_pct"] == pytest.approx(13.25)
    assert enriched["portfolio_effect"]["priced_weight"] == 147.5
    assert enriched["portfolio_effect"]["missing_weight"] == 5.0
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "gross_exposure"
    assert quality["raw_total_weight"] == 152.5
    assert quality["adjusted_total_weight"] == 152.5


def test_api_fund_holdings_normalizes_high_confidence_basis_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {
            "AAA": {"price": 10.0, "currency": "TRY", "change_pct": 1.0, "as_of": "2026-05-20T10:00:00+00:00"},
            "BBB": {"price": 20.0, "currency": "TRY", "change_pct": 2.0, "as_of": "2026-05-20T10:00:00+00:00"},
            "CCC": {"price": 30.0, "currency": "TRY", "change_pct": 3.0, "as_of": "2026-05-20T10:00:00+00:00"},
        },
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "AAA", "asset_name": "AAA", "asset_type": "local_equity", "weight": 4000.0, "previous_weight": 3500.0},
            {"asset_code": "BBB", "asset_name": "BBB", "asset_type": "local_equity", "weight": 3500.0, "previous_weight": 3500.0},
            {"asset_code": "CCC", "asset_name": "CCC", "asset_type": "local_equity", "weight": 2500.0, "previous_weight": 3000.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert enriched["status"] == "ok"
    assert positions["AAA"]["weight"] == 40.0
    assert positions["AAA"]["raw_weight"] == 4000.0
    assert positions["AAA"]["weight_quality"] == "normalized"
    assert positions["AAA"]["weight_change"] == 5.0
    assert enriched["portfolio_effect"]["priced_weight"] == 100.0
    assert enriched["portfolio_effect"]["missing_weight"] == 0.0
    assert enriched["portfolio_effect"]["estimated_return_pct"] == pytest.approx(1.85)
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "ok"
    assert quality["normalized_position_count"] == 3
    assert quality["normalization"]["action"] == "basis_points_to_percent"


def test_api_fund_holdings_normalizes_fractional_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {},
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "AAA", "asset_name": "AAA", "asset_type": "local_equity", "weight": 0.40, "previous_weight": 0.35},
            {"asset_code": "BBB", "asset_name": "BBB", "asset_type": "local_equity", "weight": 0.35, "previous_weight": 0.35},
            {"asset_code": "CCC", "asset_name": "CCC", "asset_type": "local_equity", "weight": 0.25, "previous_weight": 0.30},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["AAA"]["weight"] == 40.0
    assert positions["AAA"]["raw_weight"] == 0.40
    assert positions["AAA"]["weight_quality"] == "normalized"
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "ok"
    assert quality["normalized_position_count"] == 3
    assert quality["normalization"]["action"] == "fraction_to_percent"
    assert quality["adjusted_total_weight"] == pytest.approx(100.0)


def test_api_fund_holdings_normalizes_per_position_basis_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {},
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    # Most rows are normal percents, but a single row leaks basis points.
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "GOOD1", "asset_name": "GOOD1", "asset_type": "local_equity", "weight": 30.0, "previous_weight": 30.0},
            {"asset_code": "GOOD2", "asset_name": "GOOD2", "asset_type": "local_equity", "weight": 30.0, "previous_weight": 28.0},
            {"asset_code": "BPLEAK", "asset_name": "BPLEAK", "asset_type": "local_equity", "weight": 4000.0, "previous_weight": 3500.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["GOOD1"]["weight"] == 30.0
    assert positions["GOOD1"]["weight_quality"] == "ok"
    assert positions["BPLEAK"]["weight"] == 40.0
    assert positions["BPLEAK"]["raw_weight"] == 4000.0
    assert positions["BPLEAK"]["weight_quality"] == "normalized"
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["normalization"]["action"] == "none"
    assert quality["normalized_position_count"] == 1
    assert quality["adjusted_total_weight"] == pytest.approx(100.0)
    assert quality["status"] == "ok"


def test_api_fund_holdings_hides_astronomical_contract_number_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(api_module, "_quote_map_for_holding_stocks", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 80100517.0,
                "previous_weight": 20.45,
                "weight_change": 80100496.55,
            },
            {
                "asset_code": "GOOD",
                "asset_name": "Geçerli",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["DSTKF"]["weight"] is None
    assert positions["DSTKF"]["weight_quality"] == "invalid"
    assert positions["DSTKF"]["raw_weight"] == 80100517.0
    assert positions["DSTKF"]["weight_change"] is None
    assert "sözleşme" in positions["DSTKF"]["weight_warning"]
    assert positions["GOOD"]["weight"] == 20.0
    assert enriched["source_metadata"]["holdings_quality"]["invalid_position_count"] == 1
    assert enriched["source_metadata"]["holdings_quality"]["adjusted_total_weight"] == pytest.approx(20.0)


def test_api_fund_holdings_recomputes_delta_for_fallback_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(api_module, "_quote_map_for_holding_stocks", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 6.90,
                "weight_quality": "fallback",
                "weight_warning": "KAP satırındaki son FTD yüzdesi okunamadı.",
                "previous_weight": 20.45,
                "previous_weight_quality": "ok",
                "weight_change": 80100496.55,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    position = enriched["positions"][0]
    assert position["weight"] == pytest.approx(6.90)
    assert position["previous_weight"] == pytest.approx(20.45)
    assert position["weight_change"] == pytest.approx(-13.55)
    assert position["weight_quality"] == "fallback"
    assert position["weight_warning"]
    assert enriched["source_metadata"]["holdings_quality"]["fallback_position_count"] == 1


def test_api_fund_holdings_flags_gross_exposure_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {},
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "AAA", "asset_name": "AAA", "asset_type": "local_equity", "weight": 71.0, "previous_weight": 60.0},
            {"asset_code": "BBB", "asset_name": "BBB", "asset_type": "local_equity", "weight": 66.0, "previous_weight": 50.0},
            {"asset_code": "CCC", "asset_name": "CCC", "asset_type": "local_equity", "weight": 20.0, "previous_weight": 30.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "gross_exposure"
    assert quality["adjusted_total_weight"] == pytest.approx(157.0)
    assert quality["normalization"]["action"] == "none"
    # Raw weights remain reported to the UI.
    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["AAA"]["weight"] == 71.0
    assert positions["BBB"]["weight"] == 66.0


def test_api_fund_holdings_enriches_tpkgy_from_gefas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "TPKGYF",
                "asset_name": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
                "asset_type": "fund",
                "weight": 10.0,
                "previous_weight": 8.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:00:00+00:00", "ttl_seconds": 21600},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {"rows": [{"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0}]},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_gefas_gyf_quote",
        lambda symbol: {
            "price": 8043.164427,
            "currency": "TRY",
            "change_pct": 0.21,
            "as_of": "2026-05-18",
            "source": "gefas_gyf",
        }
        if api_module._gefas_gyf_config(symbol)
        else None,
    )
    client = TestClient(app)

    response = client.get("/funds/TLY/holdings")

    assert response.status_code == 200
    payload = response.json()
    position = payload["positions"][0]
    assert position["price"] == 8043.164427
    assert position["return_pct"] == 0.21
    assert position["return_source"] == "gefas_gyf"
    assert position["return_as_of"] == "2026-05-18"
    assert position["estimated_exposure_value"] == 10_000_000.0
    assert position["estimated_pnl_value"] == 21_000.0
    assert position["estimated_fund_return_contribution_pct"] == pytest.approx(0.021)
    assert payload["portfolio_effect"]["estimated_return_pct"] == pytest.approx(0.021)
    assert payload["portfolio_effect"]["estimated_pnl_value"] == 21_000.0
    assert payload["portfolio_effect"]["priced_weight"] == 10.0
    assert payload["portfolio_effect"]["missing_weight"] == 0.0
    assert payload["source_metadata"]["daily_market_enrichment"]["gefas_gyf_quote_count"] == 1


def test_api_fund_holdings_does_not_cache_final_live_quote_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:00:00+00:00", "ttl_seconds": 21600},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {"rows": [{"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0}]},
    )
    calls = {"count": 0}

    def fake_price_map(_symbols: List[str], **_kwargs: Any) -> Dict[str, Dict[str, Any]]:
        calls["count"] += 1
        price = 40.0 + calls["count"]
        return {"DSTKF": {"price": price, "currency": "TRY", "change_pct": 1.0, "as_of": "2026-05-20T10:00:00+00:00"}}

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_price_map)
    client = TestClient(app)

    first = client.get("/funds/TLY/holdings").json()
    second = client.get("/funds/TLY/holdings").json()

    assert first["positions"][0]["price"] == 41.0
    assert second["positions"][0]["price"] == 42.0
    assert calls["count"] == 2


def test_api_fund_holdings_live_returns_small_market_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": True,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:00:00+00:00", "ttl_seconds": 21600},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "refresh_fund_holdings",
        lambda *_args, **_kwargs: pytest.fail("live holdings should reuse static KAP cache"),
    )
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {"rows": [{"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0}]},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols, **_kwargs: {
            "DSTKF": {"price": 40.0, "currency": "TRY", "change_pct": 2.5, "as_of": "2026-05-20T10:00:00+00:00"}
        },
    )
    client = TestClient(app)

    response = client.get("/funds/TLY/holdings/live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "daily_market_enrichment"
    assert payload["positions"] == [
        {
            "asset_code": "DSTKF",
            "price": 40.0,
            "price_currency": "TRY",
            "return_pct": 2.5,
            "return_source": "infoyatirim_live_quote",
            "return_as_of": "2026-05-20T10:00:00+00:00",
            "estimated_exposure_value": 20_000_000.0,
            "estimated_pnl_value": 500_000.0,
            "estimated_fund_return_contribution_pct": 0.5,
        }
    ]
    assert payload["portfolio_effect"]["estimated_return_pct"] == 0.5
    assert payload["source_metadata"]["static_cache_hit"] is True


def test_kap_holdings_parser_keeps_only_equities_and_funds() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
TRHOL Tera Finansal Yatirimlar Holding A.S. 1.000,00 10,00 10.000,00 5,98TRYTRATACYO91Q7
TEHOL Tera Yatirim Teknoloji Holding A.S. 1.000,00 10,00 10.000,00 2,46TRYTRAGLOBL91Q0
BORÇLANMA SENETLERİ
TRFTRYBE2614 Tera Yatirim Bankasi A.S. 05/10/26 154 0,00 3.540.000,00 100,248100 29/04/26 46,423707 102,968512 3.645.085,09 94,42 2,87TL 813652091 2,90TRFTRYBE2614
T.REPO
AC2 - (BORSA DISI) PARDUS PORTFÖY YÖNETİMİ A.Ş. 04/05/26 0 43,00 6.530.630,99 43,000000 30/04/26 1.828.523,76 6.530.630,99 43,000000 6.530.630,99 19,40 5,14TL 5,20TRYA1PY00081
MEVDUAT
TÜRKİYE VAKIFLAR BANKASI T.A.O. 04/05/26 0 40,25 2.019.927,00 30/04/26 2.028.837,29 40,250000 2.028.837,29 33,49 1,60TL 1,61
V-AY İÇİNDE YAPILAN GİDERLER
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert set(by_code) == {"TRHOL", "TEHOL"}
    assert by_code["TRHOL"]["asset_type"] == "local_equity"
    assert by_code["TEHOL"]["asset_type"] == "local_equity"
    assert "AC2" not in by_code
    assert "TÜRK" not in by_code
    assert "TÜRKİYE" not in by_code


def test_kap_holdings_parser_ignores_table_headers_and_splits_funds() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
TOPLAM
(FPD
GÖRE)
GRUP
(%)TOPLAM DEĞERGÜNLÜK BR
DEĞER
REPO TEMİNAT
TUTARI
DÖVİZ
CİNSİ
BORSA
SÖZLEŞM
E NO
ISIN KODU
HİSSE SENETLERİ
Hisse Türk
AKBNK AKBANK
T.A.Ş.
18.653.248,00 69,718033 30/04/26 73,200000 1.365.417.753,60 4,66 3,67TL 80100511 3,67TRAAKBNK91N6
T.REPO
AC2 - (BORSA DISI) PARDUS PORTFÖY YÖNETİMİ A.Ş. 04/05/26 0 43,00 6.530.630,99 43,000000 30/04/26 1.828.523,76 6.530.630,99 43,000000 6.530.630,99 19,40 5,14TL 5,20TRYA1PY00081
Y.Fonu Türk
PKZ-PUSULA
PORTFÖY KUZEY
HİSSE SENEDİ
SERBEST FON (TL)
(HİSSE SENEDİ
YOĞUN FON)
PUSULA
PORTFÖY
YÖNETİMİ
A.Ş.
296.567.139,00 7,259739 24/04/26 8,213626 2.435.891.563,64 37,90 6,55TL 6,55TRYPSLP00093
PNU - PUSULA
PORTFÖY İKİNCİ
PARA PİYASASI (TL)
FONU
PUSULA
PORTFÖY
YÖNETİMİ
A.Ş.
1.937.627.250,00 1,042159 29/04/26 1,065972 2.065.456.394,94 32,14 5,56TL 5,55TRYPSLP00168
TRT131027T36 HAZİNE 0 10.000,00 1,00 10.000,00 0,68TL 0,68TRT131027T36
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="PHE",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["AKBNK"]["asset_type"] == "local_equity"
    assert by_code["AKBNK"]["asset_name"] == "AKBANK T.A.Ş."
    assert by_code["PKZ"]["asset_type"] == "fund"
    assert by_code["PNU"]["asset_type"] == "fund"
    assert "CİNSİ" not in by_code
    assert "AC2" not in by_code
    assert "TRT131027T36" not in by_code


def test_kap_holdings_parser_stops_before_trade_flow_and_keeps_fund_rows() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FİNANS FAKTORİNG A.Ş.
297.000,00 2.100,000000 30/04/26 2.105,000000 625.185.000,00 8,90 7,42TL 80100511 7,42TREDSTK00016
ALKLC ALTINKILIÇ GIDA VE SÜT SANAYİ TİCARET A.Ş.
100.000,00 33,120000 30/04/26 34,000000 3.400.000,00 5,68 4,74TL 80100511 4,74TREALTC00018
28.628.328,84 616.050.828,67 84,88100,00 GRUP TOPLAMI 91,28
VIOP Nakit Teminatı
VIOP Nakit Teminatı 511.675,39 511.675,39 100,00 0,07
IV-FON TOPLAM DEĞERİ TABLOSU
AÇIKLAMA TUTAR ORAN%
A-)FON PORTFÖY DEĞERİ
B-)HAZIR DEĞERLER
725.771.076,08
36.919,65
FON TOPLAM DEĞERİ 674.990.430,71 100,00 %
DİĞER
Y.Fonu Türk
IDL AKTİF PORTFÖY
PARA PİYASASI (TL)
FONU
AKTİF PORTFÖY YÖNETİM A.Ş
10.721.000,00 5,296661 30/04/26 5,348747 57.343.916,59 52,51 7,90TL 8,50TRYMKFT00190
NKL AKTİF PORTFÖY
KISA VADELİ
SERBEST(TL) FON
AKTİF PORTFÖY YÖNETİM A.Ş
16.524.000,00 3,101976 27/04/26 3,138747 51.864.655,43 47,49 7,15TL 7,68TRYMKFT00281
725.771.076,08FON PORTFÖY DEĞERİ 100,00
Nisan-2026
HRZ-AKTİF PORTFÖY BIST HALKA ARZ ŞİRKETLERİ HİSSE SENEDİ (TL) FONU
VII-PORTFÖYDEN SATIŞLAR
KIYMET VADE IŞLEM TARIHI FIYAT İŞLEM DEĞERI NOMINAL DEĞERI
A) HİSSE SENETLERİ(SATIŞLAR)
HISSE TÜRK
06/04/26 30.072,00OBAMS 8,480 255.010,56OBA MAKARNACILIK SANAYI VE TICARET A.Ş.
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="HRZ",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1605671",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert set(by_code) == {"DSTKF", "ALKLC", "IDL", "NKL"}
    assert by_code["DSTKF"]["asset_type"] == "local_equity"
    assert by_code["IDL"]["asset_type"] == "fund"
    assert by_code["IDL"]["weight"] == 8.5
    assert by_code["NKL"]["asset_type"] == "fund"
    assert by_code["NKL"]["weight"] == 7.68
    assert "VII" not in by_code
    assert "OBAMS" not in by_code


def test_kap_holdings_parser_resumes_repeated_table_after_vi_section() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FAKTORİNG A.Ş.
10.414.435,00 799,526624 30/03/26 1.882,000000 19.599.966.670,00 33,07 20,29TL 80100517 20,43TREDSTF00012
VI-A-GEÇEN AY İÇİNDE RÜÇHAN HAKKI KULLANIMI
Sirketin Unvani BDL-BDZ Tarihi Nominal Değeriİşlem Tipi TUTAR
V-AY İÇİNDE YAPILAN GİDERLER
AÇIKLAMA TUTAR ORAN%
IV-FON TOPLAM DEĞERİ TABLOSU
A-)FON PORTFÖY DEĞERİ
96.581.119.176,39
FON PORTFÖY DEĞERİ TABLOSU
TOPLAM
(FPD
GÖRE)
GRUP
(%)TOPLAM DEĞERGÜNLÜK BR
DEĞER
ISIN KODU
DİĞER
Y.Fonu Türk
TPKGY TERA
Portföy A.Ş
49.559,00 117.828,44326
6
23/03/26 175.774,06000
0
8.711.186.639,54 82,52 9,02TL 9,08TRYTALP00036
TPKGY TERA
Portföy A.Ş
01/04/26 1 8,00 174.000,00000
0
30/03/26 175.774,06000
0
1.406.192,48 0,01 0,00TL 0,00TRYTALP00036
96.581.119.176,39FON PORTFÖY DEĞERİ 100,00
VII-PORTFÖYDEN SATIŞLAR
A) HİSSE SENETLERİ(SATIŞLAR)
HISSE TÜRK
06/04/26 30.072,00OBAMS 8,480 255.010,56OBA MAKARNACILIK SANAYI VE TICARET A.Ş.
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TLY",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1583104",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["DSTKF"]["asset_type"] == "local_equity"
    assert by_code["TPKGY"]["asset_type"] == "fund"
    assert by_code["TPKGY"]["weight"] == pytest.approx(9.08)
    assert "OBAMS" not in by_code


def test_kap_holdings_parser_uses_safe_fallback_when_borsa_number_replaces_ftd() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FİNANS FAKTORİNG A.Ş.
18.340,00 1.817,671842 28/04/26 2.730,000000 50.068.200,00 8,12 6,90TL 80100517TREDSTF00012
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url=None,
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["DSTKF"]["weight"] == pytest.approx(6.90)
    assert by_code["DSTKF"]["weight"] < 1000
    assert by_code["DSTKF"]["weight_quality"] == "fallback"
    assert "yaklaşık" in by_code["DSTKF"]["weight_warning"]


def test_kap_holdings_parser_prefers_real_ftd_after_separated_borsa_number() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FİNANS FAKTORİNG A.Ş.
18.340,00 1.817,671842 28/04/26 2.730,000000 50.068.200,00 8,12 6,90TL 80_100_5 7,42TREDSTF00012
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url=None,
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["DSTKF"]["weight"] == pytest.approx(7.42)
    assert by_code["DSTKF"]["weight_quality"] == "ok"
    assert by_code["DSTKF"]["weight_warning"] is None


def test_kap_holdings_parser_drops_orphan_page_header_before_position() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
KONTR Kontrolmatik
Teknoloji
1.627.564,00 17,215647 22/12/25 8,620000 14.029.601,68 0,02 0,01TL 80100511 0,01TREKNTR00013
Mart-2026
TLY-TERA PORTFÖY BİRİNCİ SERBEST FON
TOPLAM
(FPD
GÖRE)
GRUP
(%)TOPLAM DEĞERGÜNLÜK BR
DEĞER
REPO TEMİNAT
TUTARI
DÖVİZ
CİNSİ
BORSA
SÖZLEŞM
E NO
TOPLAM
(FTD
GÖRE)
ISIN KODU
Enerji ve
Mühendislik
A.Ş.
PEKGY PEKER
GAYRİMEN
KUL
YATIRIM
ORTAKLIĞI
A.Ş
506.822.154,00 12,515982 31/03/26 14,300000 7.247.556.802,20 12,23 7,50TL 80100511 7,55TREPEGY00022
PEKGY PEKER
GAYRİMEN
KUL
YATIRIM
ORTAKLIĞI
A.Ş
389.798,00 12,518149 31/03/26 14,300000 5.574.111,40 0,01 0,01TL 80100511 0,01TREPEGY00022
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TLY",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["PEKGY"]["weight"] == 7.56
    assert by_code["PEKGY"]["asset_type"] == "local_equity"
    assert "TLY" not in by_code
    assert "ENERJI" not in by_code


def test_kap_holdings_parser_skips_collateral_lender_prefix() -> None:
    """``Tem.Ver.``/``Teminat Veren`` rows must not invent a phantom code or
    swallow the next position into their buffer."""
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
HİSSE SENETLERİ
Hisse Türk
BSOKE BATISÖKE
SÖKE
ÇİMENTO
SANAYİİ
T.A.Ş.
1.325.828,00 22,365319 31/03/26 32,400000 42.956.827,20 2,77 3,67TL 80100511 3,88TRABSOKE91F5
BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
186.952,00 4,532224 31/03/26 5,950000 1.112.325,13 0,07 0,10TL 80100511 0,10TRABTCIM91F5
Tem.Ver. BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
28.363.048,00 4,532224 31/03/26 5,950000 168.760.174,87 10,89 14,41TL 80100511TRABTCIM91F5
GRTHO GRAINTUR
K HOLDİNG
A.Ş.
3.104.500,00 113,537810 31/03/26 226,000000 701.617.000,00 45,26 59,93TL 63,31TREGRNT00029
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="IIE",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1584779",
    )

    by_code = {position["asset_code"]: position for position in positions}
    # The ``Tem.Ver.`` continuation row and the main BSOKE/GRTHO rows must
    # all surface, none of them as a phantom ``TEM.VER`` code.
    assert "TEM.VER" not in by_code
    assert "TEM" not in by_code
    assert by_code["BSOKE"]["asset_type"] == "local_equity"
    assert by_code["BSOKE"]["weight"] == 3.88
    # The collateral leg has the larger absolute weight (14.41 vs 0.10) so
    # it must be the one kept after de-duplication, NOT a negative leg.
    assert by_code["BTCIM"]["asset_type"] == "local_equity"
    assert by_code["BTCIM"]["weight"] == 14.41
    # GRTHO must not have been swallowed into the previous buffer; its main
    # equity weight must remain intact.
    assert by_code["GRTHO"]["asset_type"] == "local_equity"
    assert by_code["GRTHO"]["weight"] == 63.31


def test_kap_holdings_parser_excludes_repo_collateral_using_known_stock_ticker() -> None:
    """REPO/Mevduat collateral rows must not be promoted to ``local_equity``
    even when the leading token is a known BIST stock code, otherwise the
    negative collateral leg overwrites the real holding."""
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
HİSSE SENETLERİ
Hisse Türk
BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
28.363.048,00 4,532224 31/03/26 5,950000 168.760.174,87 10,89 14,41TL 80100511TRABTCIM91F5
REPO
BTCIM 01/04/26 0 43,00 100.117.808,22 43,000000 31/03/26 16.207.456,00 100.117.808,22 43,000000 -100.117.808,22 57,14 -8,5581313833 -9,03
BTCIM 01/04/26 0 43,00 75.088.356,16 43,000000 31/03/26 12.155.592,00 75.088.356,16 43,000000 -75.088.356,16 42,86 -6,4181313833 -6,78
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="IIE",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    # Only the genuine equity holding may surface; the REPO collateral
    # legs must stay in non-holding bucket and therefore be filtered out.
    assert "BTCIM" in by_code
    assert by_code["BTCIM"]["asset_type"] == "local_equity"
    assert by_code["BTCIM"]["weight"] == 14.41
    # No negative-weight ghost row is allowed for the same ticker.
    assert all(
        position["weight"] is None or position["weight"] >= 0
        for position in positions
    ), positions


def test_kap_holdings_parser_handles_glued_borsa_code_before_isin() -> None:
    """ISIN tail recognition must tolerate borsa/sözleşme codes glued
    between the currency token and the ISIN (e.g. ``14,41TL 80100511TRABTCIM91F5``).

    Without this tolerance, ``_kap_row_complete`` returns False for the row
    and the buffer keeps absorbing the next position's lines."""
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
28.363.048,00 4,532224 30/04/26 5,950000 168.760.174,87 10,89 14,41TL 80100511TRABTCIM91F5
BSOKE BATISÖKE
SÖKE
ÇİMENTO
SANAYİİ
T.A.Ş.
1.325.828,00 22,365319 30/04/26 32,400000 42.956.827,20 2,77 3,67TL 80100511 3,88TRABSOKE91F5
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    # The ISIN tail with glued borsa code must still complete the BTCIM
    # row, otherwise its lines bleed into BSOKE and one of them disappears.
    assert {"BTCIM", "BSOKE"}.issubset(by_code.keys())
    assert by_code["BTCIM"]["weight"] == 14.41
    assert by_code["BSOKE"]["weight"] == 3.88


def test_kap_holdings_parser_sums_foreign_etf_lots_and_ignores_totals() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
A.PAY
GMSTR.F QNB FİNANS PORTFÖY GÜMÜŞ ETF TRYFNBK00030 0.00% 0 100,000.00 252.29 03.04.2025 0.00% 0 0 666.25 66,625,000.00 3.33% 0.44%
GMSTR.F QNB FİNANS PORTFÖY GÜMÜŞ ETF TRYFNBK00030 0.00% 0 30,000.00 252.29 03.04.2025 0.00% 0 0 666.25 19,987,500.00 1.00% 0.13%
Ana Grup Toplamı 130,000.00 86,612,500.00 4.33% 0.57%
c.YABANCI SERMAYE PİYASASI ARAÇLARI
BORSA YATIRIM FONgARI
SIVRUS ABERDEEN STANDARD INVESTMENTS US0032641088 0.00% 0 10,810.00 29.58 23.09.2024 0.00% 0 0 69.58 33,828,691.20 0.32% 0.22%
SIVRUS ABERDEEN STANDARD INVESTMENTS US0032641088 0.00% 0 15,612.00 30.27 03.10.2024 0.00% 0 0 69.58 48,856,015.45 0.46% 0.32%
Toplam 26,422.00 82,684,706.65 0.78% 0.54%
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="GTZ",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert "ANA" not in by_code
    assert by_code["GMSTR.F"]["asset_type"] == "fund"
    assert by_code["GMSTR.F"]["weight"] == pytest.approx(0.57)
    assert by_code["GMSTR.F"]["market_value"] == pytest.approx(86_612_500.0)
    assert by_code["SIVRUS"]["asset_type"] == "foreign_fund"
    assert by_code["SIVRUS"]["asset_region"] == "foreign"
    assert by_code["SIVRUS"]["provider_symbol"] == "SIVR"
    assert by_code["SIVRUS"]["detail_clickable"] is False
    assert by_code["SIVRUS"]["weight"] == pytest.approx(0.54)
    assert by_code["SIVRUS"]["market_value"] == pytest.approx(82_684_706.65)


def test_kap_holdings_parser_uses_known_foreign_etf_names_for_gtz() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
c.YABANCI SERMAYE PİYASASI ARAÇLARI
BORSA YATIRIM FONLARI
ZSIGEUSW SWISSCANTO FONcSgEITUNG AG CH0183135992 0.00% 0 1.00 0 0 0 0 2,499,057,068.78 16.34%
SVUSASW UBS FUNc MANAGEMENT(SWITZERgAN CH0118929048 0.00% 0 1.00 0 0 0 0 2,189,418,481.25 14.34%
HUZCN HORİZONS ETFS MANAGEMENT CANAc CA37964K1012 0.00% 0 1.00 0 0 0 0 302,340,570.63 1.98%
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="GTZ",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["ZSIGEUSW"]["asset_name"] == "Swisscanto (CH) Silver ETF"
    assert by_code["ZSIGEUSW"]["provider_symbol"] == "ZSIGEU.SW"
    assert by_code["SVUSASW"]["asset_name"] == "UBS Silver ETF USD acc"
    assert by_code["SVUSASW"]["provider_symbol"] == "SVUSA.SW"
    assert by_code["HUZCN"]["asset_name"] == "Global X Silver ETF"
    assert by_code["HUZCN"]["provider_symbol"] == "HUZ.TO"


def _pps_foreign_holdings_text(
    *,
    xom_weight: float = 16.59,
    ovv_weight: float = 8.84,
    aem_weight: float = 6.38,
    b_weight: float = 5.33,
    aa_weight: float = 4.93,
    fcx_weight: float = 0.78,
    ewz_weight: float = 9.79,
    sqqq_weight: float = 4.23,
    pjl_weight: float = 18.74,
    ptn_weight: float = 5.74,
) -> str:
    return f"""
III-FON PORTFOY DEGERI TABLOSU
Hisse Senedi Yabanci
AA US EQUITY UNITED
STATAES
OF
AMERICA
34.500,000 75,660000 30/04/26 75,660000 89.101.014,30 1,35 {aa_weight:.2f}US0138721065
AEM US EQUITY AGNICO-
EAGLE
MINES
LIMITED
CMN
14.950,000 179,560000 30/04/26 179,560000 91.980.683,85 1,39 {aem_weight:.2f}CA0084741085
B US EQUITY BARRICK
MINING
CORP
60.000,000 42,240000 30/04/26 42,240000 86.753.088,00 1,31 {b_weight:.2f}CA06849F1080
FCX US EQUITY FREEPORT
-
MCMORAN
INC
6.000,000 65,350000 30/04/26 65,350000 13.416.798,00 0,20 {fcx_weight:.2f}US35671D8570
OVV US EQUITY OVINTIV
INC
64.000,000 55,860000 30/04/26 55,860000 122.462.784,00 1,85 {ovv_weight:.2f}US69047Q1022
XOM US EQUITY EXXON
MOBIL
CORP
47.550,000 147,190000 30/04/26 147,190000 239.585.281,35 3,62 {xom_weight:.2f}US30231G1022
V-AY ICINDE YAPILAN GIDERLER
ACIKLAMA TUTAR ORAN%
Satimlarda Odenen Komisyonlar 359.188,88 0,1776 %
IV-FON TOPLAM DEGERI TABLOSU
DIGER
Borsa Y.Fonu Yabanci
EWZ US EQUITY ISHARES MSCI BRAZIL ETF 102.875,000 36,370000 30/04/26 36,370000 128.117.496,88 1,94 {ewz_weight:.2f}US4642864007
SQQQ US EQUITY PROSHARES ULTRAPRO SHORT QQQ 32.500,000 38,320000 30/04/26 38,320000 42.665.480,00 0,65 {sqqq_weight:.2f}US74347G4322
Y.Fonu Turk
PJL PHILLIP PORTFOY PARA PIYASASI FONU 1.250.000,000 4,742700 30/04/26 4,742700 5.928.375,00 0,09 {pjl_weight:.2f}TRYPHPY00016
PTN PHILLIP PORTFOY ALTIN FONU 540.000,000 1,618300 30/04/26 1,618300 873.882,00 0,01 {ptn_weight:.2f}TRYPHPY00107
Doviz
USD FED 7.855.647,60 44,758437 30/04/26 44,969200 353.262.188,05 0,00 0,00USD 17,86USD
VI-ALIM SATIM ISLEMLERI
"""


def test_kap_holdings_parser_handles_pps_foreign_equity_and_continued_sections() -> None:
    positions = fund_service_module._parse_kap_holdings_pdf_text(
        _pps_foreign_holdings_text(),
        fund_code="PPS",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert {"XOM", "OVV", "AEM", "B", "AA", "FCX", "EWZ", "SQQQ", "PJL", "PTN"}.issubset(by_code)
    assert not {"CORP", "INC", "CMN", "AMERICA", "HOLDINGS", "MINERALS", "SATIMLARDA", "USD"}.intersection(by_code)
    assert by_code["XOM"]["asset_type"] == "foreign_equity"
    assert by_code["B"]["asset_type"] == "foreign_equity"
    assert by_code["XOM"]["provider_symbol"] == "XOM"
    assert by_code["B"]["provider_symbol"] == "B"
    assert by_code["XOM"]["detail_clickable"] is False
    assert by_code["EWZ"]["asset_type"] == "foreign_fund"
    assert by_code["SQQQ"]["asset_type"] == "foreign_fund"
    assert by_code["EWZ"]["provider_symbol"] == "EWZ"
    assert by_code["PJL"]["asset_type"] == "fund"
    assert by_code["PTN"]["asset_type"] == "fund"
    assert by_code["XOM"]["weight"] == pytest.approx(16.59)
    assert by_code["EWZ"]["weight"] == pytest.approx(9.79)
    assert by_code["PJL"]["weight"] == pytest.approx(18.74)


def test_kap_holdings_parser_keeps_foreign_symbol_split_from_isin() -> None:
    text = """
III-FON PORTFOY DEGERI TABLOSU
Hisse Yabanci
QCOM
US7475251036
QUALCOMM INC
6.500,00 147,549223 24/04/26 178,445000 52.159.437,81 5,90 5,03USD 5,03US7475251036
VI-ALIM SATIM ISLEMLERI
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="CPU",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["QCOM"]["asset_type"] == "foreign_equity"
    assert by_code["QCOM"]["provider_symbol"] == "QCOM"
    assert by_code["QCOM"]["asset_name"] == "QUALCOMM INC"
    assert by_code["QCOM"]["weight"] == pytest.approx(5.03)
    assert "US7475251036" not in by_code


def test_kap_holdings_parser_merges_pps_previous_month_without_bogus_removed() -> None:
    latest = fund_service_module._parse_kap_holdings_pdf_text(
        _pps_foreign_holdings_text(),
        fund_code="PPS",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )
    previous = fund_service_module._parse_kap_holdings_pdf_text(
        _pps_foreign_holdings_text(
            xom_weight=13.61,
            ovv_weight=6.67,
            aem_weight=5.68,
            b_weight=5.55,
            aa_weight=3.83,
            fcx_weight=0.52,
            ewz_weight=3.44,
            sqqq_weight=0.0,
            pjl_weight=21.02,
            ptn_weight=3.90,
        ),
        fund_code="PPS",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/2",
    )

    merged = fund_service_module._merge_holding_positions(
        latest,
        previous,
        latest_report_date="2026-04-30",
        previous_report_date="2026-03-31",
    )

    by_code = {position["asset_code"]: position for position in merged}
    assert by_code["XOM"]["previous_weight"] == pytest.approx(13.61)
    assert by_code["XOM"]["weight_change"] == pytest.approx(2.98)
    assert by_code["XOM"]["change_status"] == "increased"
    assert by_code["PJL"]["previous_weight"] == pytest.approx(21.02)
    assert by_code["PJL"]["weight_change"] == pytest.approx(-2.28)
    assert not {"CORP", "INC", "CMN", "AMERICA", "HOLDINGS", "MINERALS", "SATIMLARDA", "USD"}.intersection(by_code)


def test_kap_holdings_normalizer_corrects_fund_code_ocr_with_reference_data(tmp_path: Any) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "fund_code": "GTL",
                        "name": "GARANTİ PORTFÖY BİRİNCİ PARA PİYASASI (TL) FONU",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    positions = [
        {
            "fund_code": "GTZ",
            "asset_code": "GTG",
            "asset_name": "GARANTİ PORTFÖY BİRİNCİ PARA PİYASASI (TL) FONU",
            "asset_type": "fund",
            "weight": 2.73,
        }
    ]

    normalized = fund_service_module._normalize_holding_positions_for_response(
        tmp_path,
        positions,
        fund_code="GTZ",
    )

    assert normalized[0]["asset_code"] == "GTL"
    assert normalized[0]["asset_name"] == "GARANTİ PORTFÖY BİRİNCİ PARA PİYASASI (TL) FONU"


def test_api_fund_holdings_partial_when_pdf_not_parsed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "BAD", "fund_oid": "fund-oid", "fund_name": "Bad Fon"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "01.05.2026 10:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_fetch_report_detail",
        lambda _idx: {
            "disclosure": {"disclosureBasic": {"disclosureIndex": 200, "year": 2026, "donem": 4}},
            "attachments": [{"objId": "latest", "fileName": "BAD_2026.04.pdf", "fileExtension": "pdf"}],
        },
    )
    monkeypatch.setattr(fund_service_module, "_kap_download_attachment", lambda _obj_id: b"latest")
    monkeypatch.setattr(fund_service_module, "_extract_kap_pdf_text", lambda _data: "parse edilemeyen metin")
    client = TestClient(app)

    response = client.get("/funds/BAD/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["positions"] == []
    assert payload["source_metadata"]["parse_status"] == "partial"


def test_api_fund_allocations_history_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_history(processed_dir: Any, fund_code: str, *, lookback_days: int = 30) -> Dict[str, Any]:
        return {
            "fund_code": fund_code,
            "status": "ok",
            "lookback_days": lookback_days,
            "history": [
                {
                    "date": "2026-04-30",
                    "allocations": [
                        {
                            "fund_code": fund_code,
                            "allocation_type": "hs",
                            "label": "Hisse Senedi",
                            "weight": 58.0,
                            "report_date": "2026-04-30",
                            "source": "tefasfon_portfolio",
                        }
                    ],
                }
            ],
            "source": "tefasfon_portfolio",
            "stale": False,
            "source_metadata": {"source": "tefasfon_portfolio", "parse_status": "ok"},
        }

    monkeypatch.setattr(fund_service_module, "get_fund_allocations_history_payload", fake_history)
    client = TestClient(app)

    response = client.get("/funds/TLY/allocations/history?lookback_days=30")

    assert response.status_code == 200
    payload = response.json()
    assert payload["fund_code"] == "TLY"
    assert payload["history"][0]["allocations"][0]["label"] == "Hisse Senedi"


def _overview_commentary_payload() -> Dict[str, Any]:
    return {
        "company": "BIMAS",
        "company_title": "BIM BIRLESIK MAGAZALAR A.S.",
        "latest_period": "2025/12",
        "history_context": {
            "company_kind": "generic",
            "quarters": [
                {
                    "label": "2024/3",
                    "year": 2024,
                    "period": 1,
                    "metrics": {
                        "satis_gelirleri": 150_000_000_000.0,
                        "favok": 9_800_000_000.0,
                        "net_kar": 4_100_000_000.0,
                        "faaliyet_nakit_akisi": 3_500_000_000.0,
                        "serbest_nakit_akisi": 2_100_000_000.0,
                        "ozkaynaklar": 52_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.53,
                        "net_kar_marji": 2.73,
                        "roe": 7.88,
                        "cari_oran": 1.28,
                        "net_borc_ozkaynak": 0.42,
                        "nakit_donusum": 0.51,
                    },
                },
                {
                    "label": "2024/6",
                    "year": 2024,
                    "period": 2,
                    "metrics": {
                        "satis_gelirleri": 162_000_000_000.0,
                        "favok": 10_200_000_000.0,
                        "net_kar": 4_600_000_000.0,
                        "faaliyet_nakit_akisi": 3_800_000_000.0,
                        "serbest_nakit_akisi": 2_300_000_000.0,
                        "ozkaynaklar": 54_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.3,
                        "net_kar_marji": 2.84,
                        "roe": 8.52,
                        "cari_oran": 1.3,
                        "net_borc_ozkaynak": 0.4,
                        "nakit_donusum": 0.5,
                    },
                },
                {
                    "label": "2024/9",
                    "year": 2024,
                    "period": 3,
                    "metrics": {
                        "satis_gelirleri": 171_000_000_000.0,
                        "favok": 10_900_000_000.0,
                        "net_kar": 5_000_000_000.0,
                        "faaliyet_nakit_akisi": 4_200_000_000.0,
                        "serbest_nakit_akisi": 2_500_000_000.0,
                        "ozkaynaklar": 56_500_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.37,
                        "net_kar_marji": 2.92,
                        "roe": 8.85,
                        "cari_oran": 1.31,
                        "net_borc_ozkaynak": 0.39,
                        "nakit_donusum": 0.5,
                    },
                },
                {
                    "label": "2024/12",
                    "year": 2024,
                    "period": 4,
                    "metrics": {
                        "satis_gelirleri": 180_000_000_000.0,
                        "favok": 11_500_000_000.0,
                        "net_kar": 5_300_000_000.0,
                        "faaliyet_nakit_akisi": 4_500_000_000.0,
                        "serbest_nakit_akisi": 2_700_000_000.0,
                        "ozkaynaklar": 59_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.39,
                        "net_kar_marji": 2.94,
                        "roe": 8.98,
                        "cari_oran": 1.33,
                        "net_borc_ozkaynak": 0.38,
                        "nakit_donusum": 0.51,
                    },
                },
                {
                    "label": "2025/3",
                    "year": 2025,
                    "period": 1,
                    "metrics": {
                        "satis_gelirleri": 158_000_000_000.0,
                        "favok": 10_400_000_000.0,
                        "net_kar": 4_500_000_000.0,
                        "faaliyet_nakit_akisi": 3_900_000_000.0,
                        "serbest_nakit_akisi": 2_250_000_000.0,
                        "ozkaynaklar": 63_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.58,
                        "net_kar_marji": 2.85,
                        "roe": 7.14,
                        "cari_oran": 1.36,
                        "net_borc_ozkaynak": 0.4,
                        "nakit_donusum": 0.5,
                    },
                },
                {
                    "label": "2025/6",
                    "year": 2025,
                    "period": 2,
                    "metrics": {
                        "satis_gelirleri": 168_000_000_000.0,
                        "favok": 10_950_000_000.0,
                        "net_kar": 4_900_000_000.0,
                        "faaliyet_nakit_akisi": 4_050_000_000.0,
                        "serbest_nakit_akisi": 2_350_000_000.0,
                        "ozkaynaklar": 66_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.52,
                        "net_kar_marji": 2.92,
                        "roe": 7.42,
                        "cari_oran": 1.34,
                        "net_borc_ozkaynak": 0.45,
                        "nakit_donusum": 0.48,
                    },
                },
                {
                    "label": "2025/9",
                    "year": 2025,
                    "period": 3,
                    "metrics": {
                        "satis_gelirleri": 190_000_000_000.0,
                        "favok": 11_900_000_000.0,
                        "net_kar": 5_450_000_000.0,
                        "faaliyet_nakit_akisi": 4_500_000_000.0,
                        "serbest_nakit_akisi": 2_600_000_000.0,
                        "ozkaynaklar": 69_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.26,
                        "net_kar_marji": 2.87,
                        "roe": 7.9,
                        "cari_oran": 1.31,
                        "net_borc_ozkaynak": 0.46,
                        "nakit_donusum": 0.48,
                    },
                },
                {
                    "label": "2025/12",
                    "year": 2025,
                    "period": 4,
                    "metrics": {
                        "satis_gelirleri": 210_000_000_000.0,
                        "favok": 13_700_000_000.0,
                        "net_kar": 6_100_000_000.0,
                        "faaliyet_nakit_akisi": 4_900_000_000.0,
                        "serbest_nakit_akisi": 3_050_000_000.0,
                        "ozkaynaklar": 72_500_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.52,
                        "net_kar_marji": 2.9,
                        "roe": 8.41,
                        "cari_oran": 1.29,
                        "net_borc_ozkaynak": 0.53,
                        "nakit_donusum": 0.5,
                    },
                },
            ],
        },
        "overview_payload": {
            "income_summary": [
                {
                    "key": "satis_gelirleri",
                    "label": "Satislar",
                    "current_period": "2025/12",
                    "current_value": 721_060_000_000.0,
                    "current_display": "721,06 Milyar TL",
                    "base_period": "2024/12",
                    "base_value": 680_070_000_000.0,
                    "base_display": "680,07 Milyar TL",
                    "pct_change": 6.03,
                    "pct_display": "% 6",
                }
            ],
            "balance_summary": [
                {
                    "key": "net_borc",
                    "label": "Net Borc",
                    "current_period": "2025/12",
                    "current_value": 38_570_000_000.0,
                    "current_display": "38,57 Milyar TL",
                    "base_period": "2025/9",
                    "base_value": 31_120_000_000.0,
                    "base_display": "31,12 Milyar TL",
                    "pct_change": 23.94,
                    "pct_display": "% 24",
                }
            ],
            "charts": [
                {
                    "title": "Ceyreklik Satislar",
                    "kind": "bar",
                    "series": [
                        {"label": "2025/9", "value": 190_000_000_000.0, "display": "190,00 Milyar TL"},
                        {"label": "2025/12", "value": 210_000_000_000.0, "display": "210,00 Milyar TL"},
                    ],
                }
            ],
        },
    }


def test_kap_overview_commentary_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        assert api_key == "dummy-nvidia-key"
        assert model == "minimaxai/minimax-m2.7"
        assert normalized_payload["company"] == "BIMAS"
        assert base_analysis["scorecard"]["overall_score"] > 0
        assert debug_trace is not None
        return (
            '{"headline":"Satis ivmesi korunuyor",'
            '"bullets":["Satislar yillik bazda artarken net borc yukselmis."],'
            '"risk_note":"Nakit akisi ve borc trendi izlenmeli.",'
            '"watch_metrics":["Net borc","FAVOK marji"],'
            '"summary":"Genel gorunum dengeli, karlilik tarafi daha destekleyici.",'
            '"seasonality_note":"Son ceyrek kendi mevsimsel bandindan belirgin kopmuyor.",'
            '"score_adjustments":{"overall_adjustment":0.3,"subscores":['
            '{"key":"buyume","adjustment":0.4,"summary":"Satis ve kar buyumesi ayni ceyrek bazina gore destekleyici."},'
            '{"key":"karlilik","adjustment":0.2,"summary":"Marjlar baz etkisine ragmen korunuyor."},'
            '{"key":"bilanco","adjustment":-0.2,"summary":"Borcluluk tarafi biraz daha dikkat gerektiriyor."},'
            '{"key":"nakit_akisi","adjustment":0.1,"summary":"Nakit donusumu zayiflamadan suruyor."}'
            ']}}'
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.delenv("NVIDIA_AI_MODEL", raising=False)
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["headline"] == "Satis ivmesi korunuyor"
    assert payload["watch_metrics"] == ["Net borc", "FAVOK marji"]
    assert payload["model_used"] == "minimaxai/minimax-m2.7"
    assert payload["scorecard"]["score_source"] == "ai_adjusted"
    assert len(payload["scorecard"]["subscores"]) == 4
    assert any("validation:" in item for item in payload.get("debug_trace", []))


def test_kap_overview_commentary_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        assert api_key == "dummy-nvidia-key"
        assert model == "meta/llama-4-maverick-17b-128e-instruct"
        assert normalized_payload["model"] == "meta/llama-4-maverick-17b-128e-instruct"
        assert base_analysis["scorecard"]["score_source"] == "deterministic_only"
        return (
            '{"headline":"Model override calisti",'
            '"bullets":["Secilen model backend tarafina ulasti."],'
            '"risk_note":"",'
            '"watch_metrics":["Net borc"],'
            '"summary":"Model override ile AI adjustment akisi calisti.",'
            '"seasonality_note":"Mevsimsellik etkisi sinirli.",'
            '"score_adjustments":{"overall_adjustment":0.1,"subscores":['
            '{"key":"buyume","adjustment":0.1,"summary":"Buyume sinyalleri hafif olumlu."},'
            '{"key":"karlilik","adjustment":0.0,"summary":"Karlilik notu buyuk olcude korundu."},'
            '{"key":"bilanco","adjustment":0.0,"summary":"Bilanco notu sabit kaldi."},'
            '{"key":"nakit_akisi","adjustment":0.0,"summary":"Nakit akisi notu sabit kaldi."}'
            ']}}'
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    payload = _overview_commentary_payload()
    payload["model"] = "meta/llama-4-maverick-17b-128e-instruct"

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model_used"] == "meta/llama-4-maverick-17b-128e-instruct"
    assert body["scorecard"]["score_source"] == "ai_adjusted"


def test_kap_overview_commentary_does_not_truncate_model_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_headline = "Uzun baslik " + ("tam cumle devam ediyor " * 20) + "ve burada bitiyor."
    long_bullet = "Uzun madde " + ("kesilmeden devam eden finansal yorum " * 20) + "tamamlandi."
    long_risk = "Uzun risk notu " + ("cumle bolunmeden ilerliyor " * 20) + "tamamlandi."
    long_summary = "Uzun skor ozeti " + ("modelin gerekcesi kesilmeden aktariliyor " * 20) + "tamamlandi."
    long_subscore = "Uzun alt skor yorumu " + ("detayli nedenler kesilmeden korunuyor " * 20) + "tamamlandi."

    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        return json.dumps(
            {
                "headline": long_headline,
                "bullets": [long_bullet],
                "risk_note": long_risk,
                "watch_metrics": ["Net borc"],
                "summary": long_summary,
                "seasonality_note": long_summary,
                "score_adjustments": {
                    "overall_adjustment": 0.1,
                    "subscores": [
                        {"key": "buyume", "adjustment": 0.0, "summary": long_subscore},
                        {"key": "karlilik", "adjustment": 0.0, "summary": long_subscore},
                        {"key": "bilanco", "adjustment": 0.0, "summary": long_subscore},
                        {"key": "nakit_akisi", "adjustment": 0.0, "summary": long_subscore},
                    ],
                },
            },
            ensure_ascii=False,
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["headline"] == long_headline
    assert body["bullets"] == [long_bullet]
    assert body["risk_note"] == long_risk
    assert body["scorecard"]["summary"] == long_summary
    assert body["scorecard"]["seasonality_note"] == long_summary
    assert all(item["summary"] == long_subscore for item in body["scorecard"]["subscores"])


def test_kap_overview_commentary_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "NVIDIA_API_KEY" in payload["error"]
    assert payload["scorecard"]["score_source"] == "deterministic_only"
    assert any("NVIDIA_API_KEY bulunamadi" in item for item in payload.get("debug_trace", []))


def test_kap_overview_commentary_provider_error_includes_debug_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        raise NvidiaCommentaryError("simule provider hatasi")

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["error"] == "simule provider hatasi"
    assert payload["scorecard"]["score_source"] == "ai_failed_fallback"
    assert any("AI fallback" in item for item in payload.get("debug_trace", []))


def test_kap_overview_commentary_invalid_ai_adjustments_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        return (
            '{"headline":"Eksik adjustment",'
            '"bullets":["Model yorum yazdi ama adjustment semasi eksik."],'
            '"risk_note":"",'
            '"watch_metrics":["Net borc"]}'
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["scorecard"]["score_source"] == "ai_failed_fallback"
    assert "score_adjustments" in payload["error"]


def test_kap_overview_commentary_rejects_large_body() -> None:
    client = TestClient(app)
    response = client.post(
        "/kap/overview-commentary",
        content=b'{"company":"' + (b"x" * (api_module.MAX_REQUEST_BYTES + 1)) + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_kap_overview_commentary_rejects_unexpected_top_level() -> None:
    payload = _overview_commentary_payload()
    payload["unexpected"] = True

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "beklenmeyen alan" in response.json()["detail"]


def test_kap_overview_commentary_requires_history_context() -> None:
    payload = _overview_commentary_payload()
    payload.pop("history_context")

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "history_context" in response.json()["detail"]


def test_kap_overview_commentary_rejects_unsupported_model() -> None:
    payload = _overview_commentary_payload()
    payload["model"] = "unsupported/model"

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "desteklenmeyen model" in response.json()["detail"]


def test_kap_overview_commentary_rejects_series_limit() -> None:
    payload = _overview_commentary_payload()
    payload["overview_payload"]["charts"][0]["series"] = [
        {"label": f"2025/{idx}", "value": float(idx), "display": str(idx)}
        for idx in range(11)
    ]

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "en fazla 10" in response.json()["detail"]


def test_kap_overview_commentary_cancels_provider_task_on_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False

    async def fake_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal cancelled
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            await asyncio.sleep(0)
            return True

    async def run_case() -> None:
        with pytest.raises(HTTPException) as exc_info:
            await api_module._run_overview_commentary_until_done_or_disconnected(
                DisconnectingRequest(),  # type: ignore[arg-type]
                {"company": "BIMAS"},
            )
        assert exc_info.value.status_code == 499

    monkeypatch.setattr(api_module, "generate_overview_commentary", fake_generate)
    asyncio.run(run_case())
    assert cancelled is True


def _flow_item(
    *,
    idx: str,
    category: str,
    published_at: str,
    source: str = "Finansal Rapor",
    symbol: str = "BIMAS",
    id_prefix: str = "vyk",
) -> Dict[str, Any]:
    return {
        "id": f"{id_prefix}-{idx}",
        "source": source,
        "symbol": symbol,
        "stock_codes": [symbol],
        "title": f"Bildirim {idx}",
        "subject": f"Konu {idx}",
        "published_at": published_at,
        "category": category,
        "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx}",
    }


def _stub_flow_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    public: Optional[List[Dict[str, Any]]] = None,
    local: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[int]]:
    """Replace both upstream flow sources with deterministic stubs."""
    calls: Dict[str, List[int]] = {"public": [0], "local": [0]}

    def fake_public(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        calls["public"][0] += 1
        return list(public or [])

    def fake_local() -> List[Dict[str, Any]]:
        calls["local"][0] += 1
        return list(local or [])

    monkeypatch.setattr(api_module, "_fetch_kap_public_disclosures", fake_public)
    monkeypatch.setattr(api_module, "_local_flow_items_from_cache", fake_local)
    return calls


def test_market_flow_prefers_public_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[
            _flow_item(
                idx="1230901",
                category="finansal_rapor",
                published_at="2026-04-19T11:00:00",
                id_prefix="public",
            ),
            _flow_item(
                idx="1230900",
                category="ozel_durum",
                published_at="2026-04-19T10:00:00",
                source="Özel Durum",
                symbol="ASELS",
                id_prefix="public",
            ),
        ],
        local=[
            _flow_item(
                idx="9",
                category="finansal_rapor",
                published_at="2026-04-18T09:00:00",
                id_prefix="local",
            )
        ],
    )

    client = TestClient(app)
    response = client.get("/market/flow", params={"limit": 10})
    assert response.status_code == 200
    payload = response.json()

    assert payload["source"] == "kap_public_website"
    assert payload["degraded_mode"] is False
    assert payload["multi_category"] is True
    assert payload.get("warning") in (None, "")

    ids = [row["id"] for row in payload["items"]]
    assert ids == ["public-1230901", "public-1230900"]

    assert calls["public"][0] == 1
    assert calls["local"][0] == 0


def test_market_flow_falls_back_to_local_when_public_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[],
        local=[
            _flow_item(
                idx="42",
                category="finansal_rapor",
                published_at="2026-04-19T08:00:00",
                id_prefix="local",
            )
        ],
    )

    client = TestClient(app)
    response = client.get("/market/flow", params={"limit": 10})
    assert response.status_code == 200
    payload = response.json()

    assert payload["source"] == "local_cache"
    assert payload["degraded_mode"] is True
    assert payload["multi_category"] is False
    assert payload["warning"]
    assert calls["public"][0] == 1
    assert calls["local"][0] == 1
    assert [row["id"] for row in payload["items"]] == ["local-42"]


def test_market_flow_reuses_cache_on_repeat_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[
            _flow_item(
                idx="1",
                category="finansal_rapor",
                published_at="2026-04-19T11:00:00",
            )
        ],
    )

    client = TestClient(app)
    for _ in range(3):
        response = client.get("/market/flow", params={"limit": 5})
        assert response.status_code == 200

    # Flow cache 180 sn, degraded_mode=False oldugu icin ikinci ve ucuncu
    # istekler upstream'e hic ulasmamali.
    assert calls["public"][0] == 1


def test_market_flow_scales_detail_budget_with_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Kullanici UI'dan daha fazla kayit istedikçe backend VYK'ye gonderdigi
    # detay butcesini de buyutmeli; bu sayede 'kayit sayisi' secicisi gercekten
    # feed'i genisletiyor.
    seen_limits: List[int] = []

    def fake_public(**kwargs: Any) -> List[Dict[str, Any]]:
        seen_limits.append(int(kwargs.get("max_items") or 0))
        return [
            _flow_item(
                idx=str(i),
                category="finansal_rapor",
                published_at=f"2026-04-19T{10 + (i % 6):02d}:00:00",
            )
            for i in range(1, 6)
        ]

    monkeypatch.setattr(api_module, "_fetch_kap_public_disclosures", fake_public)
    monkeypatch.setattr(api_module, "_local_flow_items_from_cache", lambda: [])

    client = TestClient(app)
    client.get("/market/flow", params={"limit": 25})
    client.get("/market/flow", params={"limit": 500})

    assert seen_limits, "Public feed cagrilmadi"
    assert len(seen_limits) == 2
    assert seen_limits[0] < seen_limits[1]
    assert seen_limits[1] >= 500


def test_market_flow_refresh_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[
            _flow_item(
                idx="1",
                category="finansal_rapor",
                published_at="2026-04-19T11:00:00",
            )
        ],
    )

    client = TestClient(app)
    client.get("/market/flow", params={"limit": 5})
    client.get("/market/flow", params={"limit": 5, "refresh": "true"})

    assert calls["public"][0] == 2


def test_market_flow_category_filter_applies_to_public_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_flow_sources(
        monkeypatch,
        public=[
            _flow_item(idx="1", category="ozel_durum", published_at="2026-04-19T12:00:00"),
            _flow_item(idx="2", category="finansal_rapor", published_at="2026-04-19T13:00:00"),
            _flow_item(idx="3", category="genel_kurul", published_at="2026-04-19T14:00:00"),
        ],
    )

    client = TestClient(app)
    response = client.get("/market/flow", params={"limit": 10, "category": "ozel_durum"})
    assert response.status_code == 200
    payload = response.json()
    cats = {row["category"] for row in payload["items"]}
    assert cats == {"ozel_durum"}


def test_fetch_market_price_map_parses_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <table><tbody id="tableBody">
      <tr data-symbol="A1CAP">
        <th scope="row">A1CAP</th>
        <td>A1 CAPITAL</td>
        <td class="price" data-val="14.18">14.18</td>
        <td class="change" data-val="-0.21">-0.21</td>
        <td class="percent" data-val="-1.46">-1.46 %</td>
        <td>108.750.000,50</td>
        <td class="previousClose" data-val="14.39">14.39</td>
      </tr>
    </tbody></table>
    """

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return html.encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())

    payload = api_module._fetch_market_price_map(["A1CAP"])

    assert payload["A1CAP"]["price"] == 14.18
    assert payload["A1CAP"]["change_pct"] == -1.46
    assert payload["A1CAP"]["volume"] == 108750000.50


def test_extract_infoyatirim_stock_page_quote_parses_single_stock_page() -> None:
    html = """
    <section>
      <h1>TRABZONSPOR SPORTİF (TSPOR)</h1>
      <div>Son İşlem Fiyatı 1.03₺</div>
      <div>Satış 1.04₺</div>
      <div>Günlük Değişim % 0.00%</div>
      <div>Günlük Hacim (TL) 358,697,300₺</div>
      <div>Günlük Değişim (TL) 0.00₺</div>
      <div>Piyasa Değeri 7,724,999,785</div>
      <div>F/K 23.86</div>
      <div>PD/DD 0.77</div>
      <div>FD/FAVÖK 11.20</div>
    </section>
    """

    payload = api_module._extract_infoyatirim_stock_page_quote("TSPOR", html)

    assert payload["price"] == 1.03
    assert payload["change_pct"] == 0.0
    assert payload["change"] == 0.0
    assert payload["volume"] == 358_697_300.0
    assert payload["market_cap"] == 7_724_999_785.0
    assert payload["fk"] == 23.86
    assert payload["pd_dd"] == 0.77
    assert payload["fd_favok"] == 11.2
    assert payload["currency"] == "TRY"
    assert api_module._extract_infoyatirim_stock_page_quote("TSPOR", "<html></html>") == {}


def test_fetch_market_price_map_uses_stock_page_fallback_for_missing_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = """
    <table><tbody id="tableBody">
      <tr data-symbol="A1CAP">
        <th scope="row">A1CAP</th>
        <td>A1 CAPITAL</td>
        <td class="price" data-val="14.18">14.18</td>
        <td class="change" data-val="-0.21">-0.21</td>
        <td class="percent" data-val="-1.46">-1.46 %</td>
        <td>108.750.000,50</td>
      </tr>
    </tbody></table>
    """

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return html.encode("utf-8")

    fallback_calls: List[str] = []

    def fake_fallback(symbol: str) -> Dict[str, Any]:
        fallback_calls.append(symbol)
        if symbol == "EGEEN":
            return {
                "price": 6665.0,
                "currency": "TRY",
                "change": -17.35,
                "change_pct": -0.26,
                "volume": 212_043_600.0,
                "market_state": "",
                "as_of": "2026-04-25T09:00:00+00:00",
            }
        return {}

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", fake_fallback)

    payload = api_module._fetch_market_price_map(["A1CAP", "EGEEN", "TSPOR"])

    assert fallback_calls == ["EGEEN", "TSPOR"]
    assert payload["A1CAP"]["price"] == 14.18
    assert payload["EGEEN"]["price"] == 6665.0
    assert payload["EGEEN"]["change_pct"] == -0.26
    assert payload["EGEEN"]["volume"] == 212_043_600.0
    assert "TSPOR" not in payload


def test_extract_isyatirim_basic_summary_map_parses_market_caps() -> None:
    html = """
    <table class="dataTable" data-csvname="temelozet">
      <thead>
        <tr>
          <th>Kod</th>
          <th>Hisse Adı</th>
          <th>Sektör</th>
          <th>Kapanış (TL)</th>
          <th>Piyasa Değeri (mn TL)</th>
          <th>Piyasa Değeri (mn $)</th>
          <th>Halka Açıklık Oranı (%)</th>
          <th>Sermaye (mn TL)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>ASTOR</td><td>Astor Enerji</td><td>Elektrik</td><td>221,50</td><td>221.057,0</td><td>4.919,5</td><td>42,7</td><td>998,0</td></tr>
        <tr><td>GARAN</td><td>Garanti Bankası</td><td>Bankacılık</td><td>138,00</td><td>579.600,0</td><td>12.898,7</td><td>14,0</td><td>4.200,0</td></tr>
        <tr><td>MGROS</td><td>Migros</td><td>Perakende</td><td>638,50</td><td>115.603,1</td><td>2.572,7</td><td>50,8</td><td>181,1</td></tr>
        <tr><td>TUPRS</td><td>Tüpraş</td><td>Petrol</td><td>269,00</td><td>518.308,0</td><td>11.534,6</td><td>48,6</td><td>1.926,8</td></tr>
      </tbody>
    </table>
    """

    payload = api_module._extract_isyatirim_basic_summary_map(html)

    assert payload["ASTOR"]["market_cap"] == 221_057_000_000.0
    assert payload["GARAN"]["market_cap"] == 579_600_000_000.0
    assert payload["MGROS"]["market_cap"] == 115_603_100_000.0
    assert payload["TUPRS"]["market_cap"] == 518_308_000_000.0
    assert payload["ASTOR"]["fdpo"] == 0.427
    assert payload["ASTOR"]["shares_outstanding"] == 998_000_000.0
    assert api_module._market_cap_from_quote_and_meta(
        {"price": 2.0},
        {},
        {"shares_outstanding": 10.0},
    ) == 20.0


def test_index_weight_inputs_use_isyatirim_fdpo_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "_latest_share_count_from_kap_cache", lambda _symbol: None)
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "ASTOR": {
                "shares_outstanding": 998_000_000.0,
                "fdpo": 0.427,
            },
        },
    )

    payload = api_module._index_weight_inputs_for_symbol("ASTOR")

    assert payload["shares_outstanding"] == 998_000_000.0
    assert payload["fdpo"] == 0.427
    assert payload["weight_coefficient"] == 1.0


def test_bist_index_report_parser_and_fallback_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    csv_payload = """BILESEN KODU;BULTEN_ADI;ENDEKS KODU;ENDEKS ADI;ENDEKS INGILIZCE ADI;TARIH(GG/AA/YYYY)
CONSTITUENT CODE;CONSTITUENT NAME;INDEX CODE;INDEX NAME IN TURKISH;INDEX NAME IN ENGLISH;DATE(DD/MM/YYYY)
AAA.E;AAA TEST;XUTUM;BIST TUM;BIST ALL SHARES;29/04/2026
BBB.E;BBB TEST;XUTUM;BIST TUM;BIST ALL SHARES;29/04/2026
AAA.E;AAA TEST;XU100;BIST 100;BIST 100;29/04/2026
BNK.E;BANK TEST;XBANK;BIST BANKA;BIST BANKS;29/04/2026
"""
    parsed = kap_service_module.parse_bist_index_report_csv(csv_payload)

    assert parsed["XUTUM"]["symbols"] == ["AAA", "BBB"]
    assert parsed["XUTUM"]["source_date"] == "29/04/2026"
    assert parsed["XU100"]["symbols"] == ["AAA"]
    assert parsed["XBANK"]["symbols"] == ["BNK"]

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return csv_payload.encode("utf-8-sig")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    sector_payload = kap_service_module.get_bist_index_universe("XBANK", force_refresh=True)

    assert sector_payload["index"] == "XBANK"
    assert sector_payload["symbols"] == ["BNK"]
    assert sector_payload["fallback_used"] is False
    kap_service_module._BIST_UNIVERSE_CACHE.clear()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("shared BIST universe cache should satisfy request"),
    )
    cached_sector = kap_service_module.get_bist_index_universe("XBANK")
    assert cached_sector["symbols"] == ["BNK"]
    assert cached_sector["cache_hit"] is True

    def fail_urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    payload = kap_service_module.get_bist_index_universe("XUTUM", force_refresh=True)

    assert payload["index"] == "XUTUM"
    assert payload["fallback_used"] is True
    assert payload["source"] == "borsa_istanbul_csv_fallback_snapshot"
    assert payload["count"] == len(payload["symbols"])
    assert "AEFES" in payload["symbols"]


def test_market_stocks_payload_extends_rows_and_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    price_calls: List[tuple[str, ...]] = []

    monkeypatch.setattr("app.kap_service.get_bist100_companies", lambda: ["A1CAP", "AEFES"])
    monkeypatch.setattr("app.kap_service.get_bist30_companies", lambda: ["AEFES"])
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda index, force_refresh=False: {
            "index": str(index).upper(),
            "symbols": ["AEFES"] if str(index).upper() == "XU030" else ["A1CAP", "AEFES"],
            "count": 1 if str(index).upper() == "XU030" else 2,
            "source": "test",
            "source_url": "test://bist",
            "source_date": "29/04/2026",
            "fetched_at": "2026-04-29T12:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, symbol: {
            "latest_quarter": "2026Q1",
            "has_kap_cache": symbol == "A1CAP",
            "shares_outstanding": 100_000_000.0 if symbol == "A1CAP" else None,
        },
    )

    def fake_prices(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
        price_calls.append(tuple(symbols))
        return {
            "A1CAP": {
                "price": 14.18,
                "currency": "TRY",
                "change": -0.21,
                "change_pct": -1.46,
                "volume": 108750000.0,
                "as_of": "2026-04-24T12:00:00+00:00",
            },
            "AEFES": {
                "price": 19.0,
                "currency": "TRY",
                "change": 0.2,
                "change_pct": 1.06,
                "volume": None,
                "as_of": "2026-04-24T12:00:00+00:00",
            },
        }

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_prices)
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "A1CAP": {"market_cap": 999_000_000_000.0, "shares_outstanding": 1_000_000.0},
            "AEFES": {"market_cap": 38_000_000_000.0, "shares_outstanding": 2_000_000_000.0},
        },
    )

    def fake_return_bases(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            if symbol == "A1CAP":
                result[symbol] = {
                    "base_1w": 14.0,
                    "base_1m": 13.0,
                    "base_3m": 12.0,
                    "base_6m": 10.0,
                    "base_ytd": 11.0,
                    "base_1y": 7.0,
                }
            elif symbol == "XU100":
                result[symbol] = {
                    "base_1w": 100.0,
                    "base_1m": 95.0,
                    "base_3m": 90.0,
                    "base_6m": 80.0,
                    "base_ytd": 85.0,
                    "base_1y": 70.0,
                    "latest_close": 110.0,
                    "as_of": "2026-04-24T12:00:00+00:00",
                }
            elif symbol == "XU030":
                result[symbol] = {
                    "base_1w": 200.0,
                    "base_1m": 190.0,
                    "base_3m": 180.0,
                    "base_6m": 160.0,
                    "base_ytd": 170.0,
                    "base_1y": 140.0,
                    "latest_close": 220.0,
                    "as_of": "2026-04-24T12:00:00+00:00",
                }
        return result

    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", fake_return_bases)
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"market stock rows should not resolve logos over KAP: {symbol}"),
    )
    api_module._STOCKS_CACHE.clear()

    client = TestClient(app)
    first = client.get("/market/stocks?index=XU100")
    second = client.get("/market/stocks?index=XU100")
    third = client.get("/market/stocks?index=XU030")
    fourth = client.get("/market/stocks?index=XU030")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert fourth.status_code == 200
    assert price_calls == [("A1CAP", "AEFES"), ("AEFES",)]

    first_payload = first.json()
    assert first_payload["index"] == "XU100"
    assert first_payload["benchmarks"]["XU100"]["return_1w_pct"] == 10.0
    assert first_payload["benchmarks"]["XU030"]["return_1w_pct"] == 10.0

    rows = first_payload["rows"]
    a1cap = rows[0]
    assert a1cap["company"] == "A1CAP"
    assert a1cap["volume"] == 108750000.0
    assert a1cap["market_cap"] == 1418000000.0
    assert a1cap["logo_url"] is None
    assert a1cap["logo_source"] is None
    assert a1cap["return_1w_pct"] == 1.29
    assert a1cap["return_1y_pct"] == 102.57
    assert rows[1]["market_cap"] == 38000000000.0
    assert rows[1]["return_1w_pct"] is None
    assert third.json()["index"] == "XU030"
    assert [row["company"] for row in third.json()["rows"]] == ["AEFES"]


def test_xutum_market_stocks_uses_cache_only_side_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.kap_service.get_bist_all_companies", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda _index, force_refresh=False: {
            "index": "XUTUM",
            "symbols": ["AAA", "BBB"],
            "count": 2,
            "source": "test",
            "source_url": "test://bist",
            "source_date": "29/04/2026",
            "fetched_at": "2026-04-29T12:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})

    def fake_price_map(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
        assert index_name == "XUTUM"
        return {
            symbol: {
                "price": 10.0,
                "currency": "TRY",
                "change": 0.1,
                "change_pct": 1.0,
                "volume": 1000.0,
                "as_of": "2026-04-29T12:00:00+00:00",
            }
            for symbol in symbols
        }

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_price_map)

    def fake_return_bases(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        assert symbols == ["XUTUM", "XU100", "XU030"]
        return {}

    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", fake_return_bases)
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"XUTUM should not resolve logos over KAP: {symbol}"),
    )

    payload = api_module._market_stocks_payload(index_name="XUTUM", force_refresh=True)

    assert payload["index"] == "XUTUM"
    assert [row["company"] for row in payload["rows"]] == ["AAA", "BBB"]
    assert payload["rows"][0]["logo_url"] is None
    assert payload["rows"][0]["return_1w_pct"] is None
    assert payload["universe"]["fallback_used"] is False


def test_legacy_xu030_payload_does_not_resolve_kap_logos(monkeypatch: pytest.MonkeyPatch) -> None:
    api_module._XU030_CACHE.clear()
    monkeypatch.setattr("app.kap_service.get_bist30_companies", lambda: ["AKBNK"])
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols: {
            "AKBNK": {
                "price": 64.0,
                "currency": "TRY",
                "change": 1.0,
                "change_pct": 1.59,
                "as_of": "2026-04-25T11:00:00+00:00",
            }
        },
    )
    monkeypatch.setattr(api_module, "_fill_prices_via_yahoo", lambda _symbols, base_map: base_map)
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"legacy XU030 should not resolve logos over KAP: {symbol}"),
    )

    payload = api_module._xu030_payload()

    assert payload["index"] == "XU030"
    assert payload["rows"][0]["company"] == "AKBNK"
    assert payload["rows"][0]["logo_url"] is None
    assert payload["rows"][0]["logo_source"] is None


def test_infoyatirim_stock_page_fallback_populates_xu100_rows_and_index_impact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_html = """
    <table><tbody id="tableBody">
      <tr data-symbol="A1CAP">
        <td>A1 CAPITAL</td>
        <td class="price" data-val="14.18">14.18</td>
        <td class="change" data-val="-0.21">-0.21</td>
        <td class="percent" data-val="-1.46">-1.46 %</td>
        <td>108.750.000</td>
      </tr>
    </tbody></table>
    """

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return list_html.encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr("app.kap_service.get_bist100_companies", lambda: ["EGEEN"])
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_infoyatirim_stock_page_quote",
        lambda symbol: {
            "price": 6665.0,
            "currency": "TRY",
            "change": -17.35,
            "change_pct": -0.26,
            "volume": 212_043_600.0,
            "market_cap": 20_994_800_000.0,
            "market_state": "",
            "as_of": "2026-04-25T09:00:00+00:00",
        }
        if symbol == "EGEEN"
        else {},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "EGEEN": {
                "market_cap": 20_994_800_000.0,
                "shares_outstanding": 3_200_000.0,
                "fdpo": 0.358,
            },
        },
    )
    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_latest_share_count_from_kap_cache", lambda _symbol: None)
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XU100",
            "label": "BIST 100",
            "yahoo_symbol": "XU100.IS",
            "price": 1000.0,
            "prev_close": 1002.6,
            "change": -2.6,
            "change_pct": -0.26,
            "high": 1010.0,
            "low": 990.0,
            "volume": 1_000_000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-25T09:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(api_module, "_fetch_index_return_bases", lambda _index_code: {})
    monkeypatch.setattr(api_module, "_index_intraday_payload", lambda _index_code: {"line_points": []})

    client = TestClient(app)
    stocks_response = client.get("/market/stocks?index=XU100&refresh=true")
    index_response = client.get("/market/indices/XU100?refresh=true")

    assert stocks_response.status_code == 200
    stock_row = stocks_response.json()["rows"][0]
    assert stock_row["company"] == "EGEEN"
    assert stock_row["price"] == 6665.0
    assert stock_row["change_pct"] == -0.26
    assert stock_row["volume"] == 212_043_600.0

    assert index_response.status_code == 200
    index_row = index_response.json()["constituents"][0]
    assert index_row["symbol"] == "EGEEN"
    assert index_row["weight_pct"] == 100.0
    assert index_row["point_effect"] == -2.6


def test_market_stock_cards_returns_quotes_and_line_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols: {
            symbol: {
                "price": 760.0 if symbol == "BIMAS" else 325.0,
                "currency": "TRY",
                "change": -3.0 if symbol == "BIMAS" else 1.5,
                "change_pct": -0.39 if symbol == "BIMAS" else 0.46,
                "volume": 2_813_888_143.0 if symbol == "BIMAS" else 12_228_796_146.0,
                "market_state": "REGULAR",
                "as_of": "2026-04-25T11:00:00+00:00",
            }
            for symbol in symbols
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "BIMAS": {"market_cap": 456_000_000_000.0},
            "THYAO": {"market_cap": 448_500_000_000.0},
        },
    )
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_multiples",
        lambda symbol: {
            "ok": True,
            "fk": 24.47 if symbol == "BIMAS" else 3.79,
            "pd_dd": 2.75 if symbol == "BIMAS" else 0.49,
            "fd_favok": 11.37 if symbol == "BIMAS" else 5.97,
        },
    )
    monkeypatch.setattr(api_module, "_stock_card_financial_snapshot_from_cache", lambda _symbol: {})
    monkeypatch.setattr(
        api_module,
        "_stock_card_financial_ratios_from_cache",
        lambda symbol: {"net_borc_favok": 0.89 if symbol == "BIMAS" else 3.13},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_return_bases_bulk",
        lambda _symbols: {
            "BIMAS": {
                "base_1w": 765.0,
                "base_1m": 680.0,
                "base_3m": 628.5,
                "base_6m": 574.5,
                "base_ytd": 537.0,
                "base_1y": 452.0,
            },
            "THYAO": {},
        },
    )

    def fake_intraday(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
        return {
            "line_points": [
                {"time": "2026-04-25T08:00:00+00:00", "close": 750.0},
                {"time": "2026-04-25T08:05:00+00:00", "close": 760.0},
            ],
            "high": 761.5,
            "low": 747.0,
            "prev_close": 763.0,
            "volume": 3_729_933,
            "volume_lot": 3_729_933,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-25T11:00:00+00:00",
            "yahoo_symbol": f"{symbol}.IS",
            "error": None,
            "force_refresh_seen": force_refresh,
        }

    monkeypatch.setattr(api_module, "_fetch_stock_card_intraday", fake_intraday)
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"stock cards should not resolve logos over KAP: {symbol}"),
    )
    monkeypatch.setattr(api_module, "get_instruments", lambda *_args, **_kwargs: {})

    client = TestClient(app)
    response = client.get("/market/stocks/cards?symbols=BIMAS,THYAO&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "infoyatirim_yahoo_chart"
    assert [item["symbol"] for item in payload["items"]] == ["BIMAS", "THYAO"]
    assert payload["items"][0]["price"] == 760.0
    assert payload["items"][0]["change_pct"] == -0.39
    assert payload["items"][0]["volume"] == 2_813_888_143.0
    assert payload["items"][0]["volume_tl"] == 2_813_888_143.0
    assert payload["items"][0]["volume_lot"] == 3_729_933
    assert payload["items"][0]["market_cap"] == 456_000_000_000.0
    assert payload["items"][0]["logo_url"] is None
    assert payload["items"][0]["logo_source"] is None
    assert payload["items"][0]["previous_close"] == 763.0
    assert payload["items"][0]["fk"] == 24.47
    assert payload["items"][0]["pd_dd"] == 2.75
    assert payload["items"][0]["fd_favok"] == 11.37
    assert payload["items"][0]["net_borc_favok"] == 0.89
    assert payload["items"][0]["return_1w_pct"] == -0.65
    assert payload["items"][0]["return_1m_pct"] == 11.76
    assert payload["items"][0]["line_points"][1]["close"] == 760.0


def test_market_stock_cards_falls_back_to_infoyatirim_multiples_when_missing_or_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols: {
            symbol: {
                "price": 207.0 if symbol == "KCHOL" else 760.0,
                "currency": "TRY",
                "change": 2.1 if symbol == "KCHOL" else -3.0,
                "change_pct": 1.02 if symbol == "KCHOL" else -0.39,
                "volume": 3_308_601_000.0 if symbol == "KCHOL" else 2_813_888_143.0,
                "market_state": "REGULAR",
                "as_of": "2026-04-25T11:00:00+00:00",
            }
            for symbol in symbols
        },
    )
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )

    def fake_multiples(symbol: str) -> Dict[str, Any]:
        if symbol == "KCHOL":
            return {"ok": True, "fk": 0.0, "pd_dd": None, "fd_favok": 0.0}
        return {"ok": True, "fk": 24.47, "pd_dd": 2.75, "fd_favok": 11.37}

    monkeypatch.setattr(api_module, "_fetch_isyatirim_multiples", fake_multiples)
    monkeypatch.setattr(api_module, "_stock_card_financial_snapshot_from_cache", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_stock_card_financial_ratios_from_cache", lambda _symbol: {"net_borc_favok": None})
    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", lambda _symbols: {})
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_card_intraday",
        lambda symbol, force_refresh=False: {
            "line_points": [],
            "high": None,
            "low": None,
            "prev_close": None,
            "volume": None,
            "volume_lot": None,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-25T11:00:00+00:00",
            "yahoo_symbol": f"{symbol}.IS",
            "error": None,
        },
    )

    fallback_calls: List[str] = []

    def fake_infoyatirim_quote(symbol: str) -> Dict[str, Any]:
        fallback_calls.append(symbol)
        if symbol == "KCHOL":
            return {"fk": 23.86, "pd_dd": 0.77, "fd_favok": 11.2}
        return {}

    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", fake_infoyatirim_quote)

    client = TestClient(app)
    response = client.get("/market/stocks/cards?symbols=KCHOL,BIMAS&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    items = {item["symbol"]: item for item in payload["items"]}
    assert items["KCHOL"]["fk"] == 23.86
    assert items["KCHOL"]["pd_dd"] == 0.77
    assert items["KCHOL"]["fd_favok"] == 11.2
    assert items["BIMAS"]["fk"] == 24.47
    assert items["BIMAS"]["pd_dd"] == 2.75
    assert items["BIMAS"]["fd_favok"] == 11.37
    assert fallback_calls == ["KCHOL"]


def test_market_stock_card_valuation_prefers_kap_snapshot_and_reprices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_stock_card_financial_snapshot_from_cache",
        lambda _symbol: {
            "latest_quarter": "2026Q1",
            "ttm_net_kar": 10.0,
            "ozkaynaklar": 20.0,
            "ttm_favok": 5.0,
            "net_borc": 2.0,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_multiples",
        lambda symbol: pytest.fail(f"external multiples should not be fetched for complete KAP snapshot: {symbol}"),
    )
    monkeypatch.setattr(api_module, "_stock_card_financial_ratios_from_cache", lambda _symbol: {"net_borc_favok": 0.4})

    first = api_module._resolve_market_card_valuation("BIMAS", market_cap=100.0)
    second = api_module._resolve_market_card_valuation("BIMAS", market_cap=120.0)

    assert first["fk"] == 10.0
    assert first["pd_dd"] == 5.0
    assert first["fd_favok"] == 20.4
    assert first["net_borc_favok"] == 0.4
    assert first["valuation_source"] == "kap_computed"
    assert second["fk"] == 12.0
    assert second["pd_dd"] == 6.0
    assert second["fd_favok"] == 24.4
    assert second["cache_hit"] is True


def test_market_stock_card_valuation_caches_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    monkeypatch.setattr(api_module, "_stock_card_financial_snapshot_from_cache", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_stock_card_financial_ratios_from_cache", lambda _symbol: {"net_borc_favok": None})

    def fake_multiples(symbol: str) -> Dict[str, Any]:
        calls.append(symbol)
        return {"ok": True, "source": "isyatirim_company_card", "fk": 11.0, "pd_dd": 2.0, "fd_favok": 8.0}

    monkeypatch.setattr(api_module, "_fetch_isyatirim_multiples", fake_multiples)
    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", lambda symbol: pytest.fail(symbol))

    first = api_module._resolve_market_card_valuation("BIMAS", market_cap=100.0)
    second = api_module._resolve_market_card_valuation("BIMAS", market_cap=120.0)

    assert first["fk"] == 11.0
    assert second["fk"] == 11.0
    assert second["cache_hit"] is True
    assert calls == ["BIMAS"]


def test_market_stock_cards_rejects_more_than_twelve_symbols() -> None:
    client = TestClient(app)
    symbols = ",".join(f"SYM{i}" for i in range(13))

    response = client.get(f"/market/stocks/cards?symbols={symbols}")

    assert response.status_code == 400
    assert "En fazla 12" in response.json()["detail"]


def test_market_stock_card_chart_maps_range_and_normalizes_points(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chart(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        assert yahoo_symbol == "BIMAS.IS"
        assert interval == "15m"
        assert range_ == "5d"
        return {
            "ok": True,
            "meta": {},
            "points": [
                {"time": "2026-04-25T08:30:00+00:00", "close": 752.0},
                {"time": "2026-04-25T08:00:00+00:00", "close": 750.0},
                {"time": "2026-04-25T08:15:00+00:00", "close": None},
                {"time": "2026-04-25T08:30:00+00:00", "close": 753.0, "high": 754.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    client = TestClient(app)
    response = client.get("/market/stocks/cards/chart?symbol=bimas.IS&range=1w&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BIMAS"
    assert payload["range"] == "1w"
    assert payload["yahoo_symbol"] == "BIMAS.IS"
    assert payload["source"] == "yahoo_live"
    assert [point["time"] for point in payload["line_points"]] == [
        "2026-04-25T08:00:00+00:00",
        "2026-04-25T08:30:00+00:00",
    ]
    assert payload["line_points"][1]["close"] == 753.0


@pytest.mark.parametrize(
    ("chart_range", "expected_interval", "expected_yahoo_range"),
    [("1m", "4h", "1mo"), ("1y", "1d", "1y")],
)
def test_market_stock_card_chart_uses_requested_long_range_granularity(
    monkeypatch: pytest.MonkeyPatch,
    chart_range: str,
    expected_interval: str,
    expected_yahoo_range: str,
) -> None:
    calls: List[str] = []

    def fake_chart(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        calls.append(f"{yahoo_symbol}:{interval}:{range_}")
        return {
            "ok": True,
            "meta": {},
            "points": [{"time": "2026-04-25T08:00:00+00:00", "close": 750.0}],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    client = TestClient(app)
    response = client.get(f"/market/stocks/cards/chart?symbol=BIMAS&range={chart_range}&refresh=true")

    assert response.status_code == 200
    assert response.json()["range"] == chart_range
    assert calls == [f"BIMAS.IS:{expected_interval}:{expected_yahoo_range}"]


def test_market_stock_card_chart_uses_normalized_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_chart(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        calls.append(f"{yahoo_symbol}:{interval}:{range_}")
        return {
            "ok": True,
            "meta": {},
            "points": [{"time": "2026-04-25T08:00:00+00:00", "close": 750.0}],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    client = TestClient(app)
    first = client.get("/market/stocks/cards/chart?symbol=BIMAS&range=1d")
    second = client.get("/market/stocks/cards/chart?symbol=bimas.IS&range=1d")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == ["BIMAS.IS:5m:1d"]
    assert first.json()["source"] == "yahoo_live"
    assert second.json()["source"] == "yahoo_cache"
    assert first.json()["as_of"] == second.json()["as_of"]


def test_market_stock_card_chart_falls_back_to_previous_session_when_daily_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    period_calls: List[date] = []

    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_raw",
        lambda *_args, **_kwargs: {"ok": True, "meta": {}, "points": []},
    )

    def fake_period_chart(
        yahoo_symbol: str,
        *,
        interval: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        assert yahoo_symbol == "BIMAS.IS"
        assert interval == "5m"
        assert start_date == end_date
        assert start_date.weekday() < 5
        period_calls.append(start_date)
        return {
            "ok": True,
            "meta": {"chartPreviousClose": 100.0},
            "points": [
                {"time": f"{start_date.isoformat()}T07:00:00+00:00", "close": 101.0, "high": 102.0, "low": 99.0},
                {"time": f"{start_date.isoformat()}T07:05:00+00:00", "close": 103.0, "high": 104.0, "low": 100.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_period_chart)

    client = TestClient(app)
    response = client.get("/market/stocks/cards/chart?symbol=BIMAS&range=1d&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "yahoo_previous_session"
    assert payload["error"] is None
    assert len(payload["line_points"]) == 2
    assert payload["line_points"][-1]["close"] == 103.0
    assert period_calls


def test_stock_card_session_state_marks_old_intraday_points_as_closed() -> None:
    state = api_module._stock_card_session_state(
        [{"time": "2026-05-26T09:40:00+00:00", "close": 373.0}],
        market_state="REGULAR",
        source="yahoo_live",
        now=datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc),
    )

    assert state["session_status"] == "previous_session"
    assert state["session_label"] == "Piyasa kapalı"
    assert state["is_live"] is False
    assert state["is_stale"] is True
    assert state["last_trade_at"] == "2026-05-26T09:40:00+00:00"
    assert state["last_trade_date"] == "2026-05-26"


def test_market_stock_card_chart_rejects_invalid_symbol_and_range() -> None:
    client = TestClient(app)

    invalid_symbol = client.get("/market/stocks/cards/chart?symbol=!!!&range=1d")
    invalid_range = client.get("/market/stocks/cards/chart?symbol=BIMAS&range=2y")

    assert invalid_symbol.status_code == 400
    assert invalid_range.status_code == 400


def test_market_stock_card_chart_returns_error_for_no_data_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_raw",
        lambda *_args, **_kwargs: {"ok": False, "error": "not_found"},
    )

    client = TestClient(app)
    response = client.get("/market/stocks/cards/chart?symbol=NOTREAL&range=1m")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "NOTREAL"
    assert payload["line_points"] == []
    assert payload["error"] == "not_found"


def test_market_comparison_history_returns_mixed_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fund_history(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        assert fund_code == "TLY"
        assert kwargs["start_date"].isoformat() == "2026-04-01"
        assert kwargs["end_date"].isoformat() == "2026-04-05"
        return {
            "status": "ok",
            "source": "sqlite",
            "points": [
                {"date": "2026-03-31", "price": 9.8},
                {"date": "2026-04-01", "price": 10.0},
                {"date": "2026-04-05", "price": 10.5},
            ],
            "source_metadata": {},
        }

    def fake_chart(yahoo_symbol: str, *, interval: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
        assert interval == "1d"
        assert start_date.isoformat() == "2026-04-01"
        assert end_date.isoformat() == "2026-04-05"
        values = {
            "BIMAS.IS": 750.0,
            "XU100.IS": 9500.0,
            "USDTRY=X": 32.0,
        }
        base = values[yahoo_symbol]
        return {
            "ok": True,
            "points": [
                {"time": "2026-04-01T08:00:00+00:00", "close": base},
                {"time": "2026-04-05T08:00:00+00:00", "close": base + 5.0},
            ],
        }

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_fund_history)
    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_chart)

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-05",
            "assets": [
                {"id": "fund:TLY", "kind": "fund", "symbol": "TLY", "label": "TLY"},
                {"id": "stock:BIMAS", "kind": "stock", "symbol": "BIMAS", "label": "BIMAS"},
                {"id": "index:XU100", "kind": "index", "symbol": "XU100", "label": "BIST 100"},
                {"id": "fx:USD/TRY", "kind": "fx", "symbol": "USD/TRY", "label": "Dolar"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["start_date"] == "2026-04-01"
    assert [asset["id"] for asset in payload["assets"]] == ["fund:TLY", "stock:BIMAS", "index:XU100", "fx:USD/TRY"]
    assert payload["assets"][0]["points"] == [
        {"date": "2026-04-01", "value": 10.0},
        {"date": "2026-04-05", "value": 10.5},
    ]
    assert payload["assets"][1]["points"][0] == {"date": "2026-04-01", "value": 750.0}
    assert all(asset["error"] is None for asset in payload["assets"])


def test_market_comparison_history_rejects_reversed_dates() -> None:
    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-05",
            "end_date": "2026-04-01",
            "assets": [{"kind": "index", "symbol": "XU100"}],
        },
    )

    assert response.status_code == 400
    assert "start_date" in response.json()["detail"]


def test_market_comparison_history_returns_partial_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fund_history(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "ok",
            "source": "sqlite",
            "points": [{"date": "2026-04-01", "price": 10.0}],
            "source_metadata": {},
        }

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_fund_history)
    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_period_raw",
        lambda *_args, **_kwargs: {"ok": False, "error": "not_found", "points": []},
    )

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-05",
            "assets": [
                {"id": "fund:TLY", "kind": "fund", "symbol": "TLY"},
                {"id": "stock:NOTREAL", "kind": "stock", "symbol": "NOTREAL"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"][0]["points"] == [{"date": "2026-04-01", "value": 10.0}]
    assert payload["assets"][0]["error"] is None
    assert payload["assets"][1]["points"] == []
    assert payload["assets"][1]["error"] == "not_found"


def test_fetch_stock_card_intraday_maps_chart_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chart(_yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        assert interval == "5m"
        assert range_ == "1d"
        return {
            "ok": True,
            "meta": {
                "regularMarketPrice": 760.0,
                "chartPreviousClose": 763.0,
                "regularMarketDayHigh": 761.5,
                "regularMarketDayLow": 747.0,
                "regularMarketVolume": 3_729_933,
                "currency": "TRY",
                "marketState": "REGULAR",
                "regularMarketTime": 1_777_110_000,
            },
            "points": [
                {"time": "2026-04-25T08:00:00+00:00", "close": 750.0, "high": 752.0, "low": 748.0},
                {"time": "2026-04-25T08:05:00+00:00", "close": 760.0, "high": 761.0, "low": 754.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    payload = api_module._fetch_stock_card_intraday("BIMAS")

    assert payload["yahoo_symbol"] == "BIMAS.IS"
    assert payload["price"] == 760.0
    assert payload["change_pct"] == -0.39
    assert payload["high"] == 761.5
    assert payload["low"] == 747.0
    assert payload["prev_close"] == 763.0
    assert payload["volume"] == 3_729_933
    assert payload["volume_lot"] == 3_729_933
    assert len(payload["line_points"]) == 2


def test_fetch_stock_card_intraday_returns_empty_on_chart_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_raw",
        lambda *_args, **_kwargs: {"ok": False, "error": "provider_down"},
    )

    payload = api_module._fetch_stock_card_intraday("BIMAS")

    assert payload["line_points"] == []
    assert payload["price"] is None
    assert payload["error"] == "provider_down"


def test_market_stocks_rejects_unknown_index() -> None:
    client = TestClient(app)
    response = client.get("/market/stocks?index=XU050")

    assert response.status_code == 400
    assert "Desteklenmeyen endeks" in response.json()["detail"]


def test_market_indices_list_returns_main_and_sector_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(index_code: str) -> Dict[str, Any]:
        prices = {"XUTUM": 330.0, "XU100": 110.0, "XU030": 220.0, "XBANK": 440.0}
        prevs = {"XUTUM": 300.0, "XU100": 100.0, "XU030": 200.0, "XBANK": 400.0}
        labels = {"XUTUM": "BIST Tüm", "XU100": "BIST 100", "XU030": "BIST 30", "XBANK": "BIST Banka"}
        return {
            "symbol": index_code,
            "label": labels[index_code],
            "yahoo_symbol": f"{index_code}.IS",
            "price": prices[index_code],
            "prev_close": prevs[index_code],
            "change": prices[index_code] - prevs[index_code],
            "change_pct": 10.0,
            "high": 112.0,
            "low": 99.0,
            "volume": 123000000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        }

    def fake_bases(index_code: str) -> Dict[str, Any]:
        latest = {"XUTUM": 330.0, "XU100": 110.0, "XU030": 220.0, "XBANK": 440.0}[index_code]
        return {
            "base_1w": latest / 1.1,
            "base_1m": latest - 20.0,
            "base_3m": latest - 30.0,
            "base_6m": latest - 40.0,
            "base_ytd": latest - 50.0,
            "base_1y": latest - 60.0,
            "base_5y": latest - 70.0,
            "latest_close": latest,
            "as_of": "2026-04-24T12:00:00+00:00",
            "yahoo_symbol": f"{index_code}.IS",
        }

    monkeypatch.setattr(api_module, "_fetch_index_quote", fake_quote)
    monkeypatch.setattr(api_module, "_fetch_index_return_bases", fake_bases)
    monkeypatch.setattr(api_module, "_MARKET_INDEX_ORDER", ["XUTUM", "XU100", "XU030", "XBANK"])

    client = TestClient(app)
    response = client.get("/market/indices")

    assert response.status_code == 200
    payload = response.json()
    assert [row["symbol"] for row in payload["rows"]] == ["XUTUM", "XU100", "XU030", "XBANK"]
    assert payload["rows"][0]["label"] == "BIST Tüm"
    assert payload["rows"][0]["return_1w_pct"] == 10.0
    assert payload["rows"][0]["volume"] == 123000000.0
    assert "XPTIC" not in {row["symbol"] for row in payload["rows"]}


def test_market_sector_index_order_excludes_capped_and_removed_codes() -> None:
    assert "XBANK" in api_module._MARKET_INDEX_ORDER
    assert "XUSIN" in api_module._MARKET_INDEX_ORDER
    for code in {"XSINS", "XGYOS", "XTKJS", "XTTIC", "XPTIC", "XKNKL", "XYIHZ"}:
        assert code not in api_module._MARKET_INDEX_ORDER
        assert code not in api_module._MARKET_INDEX_META


def test_market_index_return_bases_fall_back_to_isyatirim_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_yahoo_chart(_yahoo_symbol: str, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "points": [{"time": "2026-05-27T00:00:00+00:00", "close": 200.0}],
        }

    def fake_isyatirim_history(
        index_code: str,
        *,
        start_date: date,
        end_date: date,
    ) -> List[tuple[datetime, float]]:
        assert index_code == "XAKUR"
        assert start_date < end_date
        return [
            (datetime(2021, 5, 17, tzinfo=timezone.utc), 70.0),
            (datetime(2025, 5, 27, tzinfo=timezone.utc), 100.0),
            (datetime(2025, 11, 26, tzinfo=timezone.utc), 110.0),
            (datetime(2026, 1, 2, tzinfo=timezone.utc), 120.0),
            (datetime(2026, 2, 25, tzinfo=timezone.utc), 130.0),
            (datetime(2026, 4, 27, tzinfo=timezone.utc), 150.0),
            (datetime(2026, 5, 20, tzinfo=timezone.utc), 160.0),
            (datetime(2026, 5, 27, tzinfo=timezone.utc), 200.0),
        ]

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_yahoo_chart)
    monkeypatch.setattr(api_module, "_fetch_isyatirim_index_history", fake_isyatirim_history)

    bases = api_module._fetch_index_return_bases("XAKUR")

    assert bases["history_source"] == "isyatirim"
    assert bases["base_1w"] == 160.0
    assert bases["base_1m"] == 150.0
    assert bases["base_3m"] == 130.0
    assert bases["base_6m"] == 110.0
    assert bases["base_ytd"] == 120.0
    assert bases["base_1y"] == 100.0

    row = api_module._market_index_row(
        "XAKUR",
        quote={
            "symbol": "XAKUR",
            "label": "BIST Aracı Kurumlar",
            "yahoo_symbol": "XAKUR.IS",
            "price": 220.0,
            "prev_close": 200.0,
            "change": 20.0,
            "change_pct": 10.0,
            "high": 225.0,
            "low": 198.0,
            "volume": None,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-05-27T12:00:00+00:00",
            "error": None,
        },
    )
    assert row["return_1w_pct"] == 37.5
    assert row["return_1m_pct"] == 46.67
    assert row["return_ytd_pct"] == 83.33


def test_market_index_detail_returns_line_points_and_weighted_constituents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XU100",
            "label": "BIST 100",
            "yahoo_symbol": "XU100.IS",
            "price": 1000.0,
            "prev_close": 990.0,
            "change": 10.0,
            "change_pct": 1.01,
            "high": 1010.0,
            "low": 980.0,
            "volume": 1000000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_index_return_bases",
        lambda _index_code: {
            "base_1w": 900.0,
            "base_1m": 850.0,
            "base_3m": 800.0,
            "base_6m": 750.0,
            "base_ytd": 700.0,
            "base_1y": 650.0,
            "base_5y": 500.0,
            "latest_close": 1000.0,
            "as_of": "2026-04-24T12:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        api_module,
        "_index_intraday_payload",
        lambda _index_code: {
            "line_points": [
                {"time": "2026-04-24T09:00:00+00:00", "close": 990.0},
                {"time": "2026-04-24T10:00:00+00:00", "close": 1000.0},
            ],
            "high": 1005.0,
            "low": 985.0,
            "prev_close": 990.0,
            "yahoo_symbol": "XU100.IS",
        },
    )
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: {
            "rows": [
                {
                    "company": "AAA",
                    "price": 10.0,
                    "price_currency": "TRY",
                    "change_pct": 2.0,
                    "volume": 100.0,
                },
                {
                    "company": "BBB",
                    "price": 30.0,
                    "price_currency": "TRY",
                    "change_pct": -1.0,
                    "volume": 200.0,
                },
            ]
        },
    )
    monkeypatch.setattr(
        api_module,
        "_index_weight_inputs_for_symbol",
        lambda symbol: {
            "shares_outstanding": 100.0,
            "fdpo": 0.5,
            "weight_coefficient": 1.0,
        },
    )

    client = TestClient(app)
    response = client.get("/market/indices/XU100")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XU100"
    assert payload["line_points"][1]["close"] == 1000.0
    assert payload["return_5y_pct"] == 100.0
    assert payload["weight_status"] == "available"
    rows = {row["symbol"]: row for row in payload["constituents"]}
    assert rows["AAA"]["weight_pct"] == 25.0
    assert rows["AAA"]["point_effect"] == 5.0
    assert rows["BBB"]["weight_pct"] == 75.0
    assert rows["BBB"]["point_effect"] == -7.5


def test_market_sector_index_detail_uses_bist_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XBANK",
            "label": "BIST Banka",
            "yahoo_symbol": "XBANK.IS",
            "price": 1500.0,
            "prev_close": 1450.0,
            "change": 50.0,
            "change_pct": 3.45,
            "high": 1510.0,
            "low": 1440.0,
            "volume": 5000000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_index_return_bases",
        lambda _index_code: {
            "base_1w": 1400.0,
            "base_1m": 1300.0,
            "base_3m": 1200.0,
            "base_6m": 1100.0,
            "base_ytd": 1000.0,
            "base_1y": 900.0,
            "base_5y": 750.0,
            "latest_close": 1500.0,
            "as_of": "2026-04-24T12:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        api_module,
        "_index_intraday_payload",
        lambda _index_code: {
            "line_points": [{"time": "2026-04-24T10:00:00+00:00", "close": 1500.0}],
            "high": 1510.0,
            "low": 1440.0,
            "prev_close": 1450.0,
            "yahoo_symbol": "XBANK.IS",
        },
    )
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda index, force_refresh=False: {
            "index": str(index).upper(),
            "symbols": ["AAA", "BBB"],
            "count": 2,
            "source": "test",
            "source_url": "test://bist",
            "source_date": "29/04/2026",
            "fetched_at": "2026-04-29T12:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )

    def fake_price_map(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
        assert symbols == ["AAA", "BBB"]
        assert index_name == "XUTUM"
        return {
            "AAA": {"price": 10.0, "currency": "TRY", "change_pct": 2.0, "volume": 100.0},
            "BBB": {"price": 30.0, "currency": "TRY", "change_pct": -1.0, "volume": 200.0},
        }

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_price_map)
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(api_module, "_load_cached_kap_market_metadata", lambda _cache_dir, _symbol: {})
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: pytest.fail("sector index detail should use the BIST index universe directly"),
    )
    monkeypatch.setattr(
        api_module,
        "_index_weight_inputs_for_symbol",
        lambda symbol: {
            "shares_outstanding": 100.0,
            "fdpo": 0.5,
            "weight_coefficient": 1.0,
        },
    )

    client = TestClient(app)
    response = client.get("/market/indices/XBANK")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XBANK"
    assert payload["return_5y_pct"] == 100.0
    assert payload["weight_status"] == "available"
    assert [row["symbol"] for row in payload["constituents"]] == ["BBB", "AAA"]


def test_market_index_detail_leaves_weights_empty_when_inputs_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XU030",
            "label": "BIST 30",
            "yahoo_symbol": "XU030.IS",
            "price": 2000.0,
            "prev_close": 1980.0,
            "change": 20.0,
            "change_pct": 1.01,
            "high": None,
            "low": None,
            "volume": None,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(api_module, "_fetch_index_return_bases", lambda _index_code: {})
    monkeypatch.setattr(api_module, "_index_intraday_payload", lambda _index_code: {"line_points": []})
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: {
            "rows": [
                {"company": "AAA", "price": 10.0, "price_currency": "TRY", "change_pct": 2.0, "volume": 100.0},
            ]
        },
    )
    monkeypatch.setattr(
        api_module,
        "_index_weight_inputs_for_symbol",
        lambda _symbol: {"shares_outstanding": 100.0, "fdpo": None, "weight_coefficient": None},
    )

    client = TestClient(app)
    response = client.get("/market/indices/XU030")

    assert response.status_code == 200
    payload = response.json()
    assert payload["weight_status"] == "unavailable"
    assert payload["constituents"][0]["weight_pct"] is None
    assert payload["constituents"][0]["point_effect"] is None
    assert "Ağırlık" in payload["weight_note"]


def test_market_index_detail_rejects_unknown_index() -> None:
    client = TestClient(app)
    response = client.get("/market/indices/XU050")

    assert response.status_code == 400
    assert "Desteklenmeyen endeks" in response.json()["detail"]


def _watch_quote(
    *,
    price: float,
    prev_close: float,
    currency: str = "TRY",
    as_of: str = "2026-04-20T09:30:00+00:00",
) -> Dict[str, Any]:
    change = round(price - prev_close, 4)
    change_pct = round((change / prev_close) * 100, 2)
    return {
        "ok": True,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": currency,
        "market_state": "REGULAR",
        "as_of": as_of,
    }


def _fx_payload_for_watch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "items": items,
        "source": "yahoo_finance_chart",
        "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
        "as_of": "2026-04-20T09:30:00+00:00",
    }


def _commodity_payload_for_watch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "items": items,
        "source": "yahoo_finance_chart",
        "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
        "as_of": "2026-04-20T09:30:00+00:00",
    }


def test_market_watch_success_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        if yahoo_symbol == "XU100.IS":
            return _watch_quote(price=10235.0, prev_close=10195.0)
        if yahoo_symbol == "XU030.IS":
            return _watch_quote(price=11312.0, prev_close=11300.0)
        return {"ok": False, "error": "unsupported_symbol"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(
        api_module,
        "_market_fx_payload",
        lambda: _fx_payload_for_watch(
            [
                {
                    "symbol": "USD/TRY",
                    "label": "Amerikan Doları",
                    "yahoo_symbol": "USDTRY=X",
                    "price": 38.52,
                    "prev_close": 38.40,
                    "change": 0.12,
                    "change_pct": 0.31,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "EUR/TRY",
                    "label": "Euro",
                    "yahoo_symbol": "EURTRY=X",
                    "price": 41.80,
                    "prev_close": 41.72,
                    "change": 0.08,
                    "change_pct": 0.19,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        api_module,
        "_market_commodities_payload",
        lambda: _commodity_payload_for_watch(
            [
                {
                    "symbol": "BRENT",
                    "label": "Brent Petrol",
                    "yahoo_symbol": "BZ=F",
                    "price": 86.12,
                    "prev_close": 85.70,
                    "change": 0.42,
                    "change_pct": 0.49,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "ALTIN",
                    "label": "Altın (Ons)",
                    "yahoo_symbol": "GC=F",
                    "price": 2412.5,
                    "prev_close": 2404.0,
                    "change": 8.5,
                    "change_pct": 0.35,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "GUMUS",
                    "label": "Gümüş (Ons)",
                    "yahoo_symbol": "SI=F",
                    "price": 30.2,
                    "prev_close": 30.0,
                    "change": 0.2,
                    "change_pct": 0.67,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "DOGALGAZ",
                    "label": "Doğal Gaz",
                    "yahoo_symbol": "NG=F",
                    "price": 2.45,
                    "prev_close": 2.41,
                    "change": 0.04,
                    "change_pct": 1.66,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()

    assert payload["source"] == "yahoo_finance_chart"
    assert "15dk" in payload["delay_note"]

    sections = payload["sections"]
    assert [row["symbol"] for row in sections["indices"]] == ["XUTUM", "XU100", "XU030"]
    assert [row["symbol"] for row in sections["fx"]] == ["USD/TRY", "EUR/TRY"]
    assert [row["symbol"] for row in sections["commodities"]] == ["BRENT", "ALTIN", "GUMUS", "DOGALGAZ"]


def test_market_watch_index_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        calls.append(yahoo_symbol)
        if yahoo_symbol == "XUTUM.IS":
            return _watch_quote(price=10000.0, prev_close=9950.0)
        if yahoo_symbol == "XU100.IS":
            return {"ok": False, "error": "primary_failed"}
        if yahoo_symbol == "^XU100":
            return _watch_quote(price=10235.0, prev_close=10195.0)
        if yahoo_symbol == "XU030.IS":
            return _watch_quote(price=11312.0, prev_close=11300.0)
        return {"ok": False, "error": "unsupported_symbol"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", lambda: _fx_payload_for_watch([]))
    monkeypatch.setattr(api_module, "_market_commodities_payload", lambda: _commodity_payload_for_watch([]))

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()

    indices = {row["symbol"]: row for row in payload["sections"]["indices"]}
    assert indices["XU100"]["yahoo_symbol"] == "^XU100"
    assert "XU100.IS" in calls
    assert "^XU100" in calls


def test_market_watch_reuses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    counters: Dict[str, List[int]] = {
        "quote": [0],
        "fx": [0],
        "commodity": [0],
    }

    def fake_quote(_yahoo_symbol: str) -> Dict[str, Any]:
        counters["quote"][0] += 1
        return _watch_quote(price=100.0, prev_close=99.0)

    def fake_fx() -> Dict[str, Any]:
        counters["fx"][0] += 1
        return _fx_payload_for_watch([])

    def fake_commodity() -> Dict[str, Any]:
        counters["commodity"][0] += 1
        return _commodity_payload_for_watch([])

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", fake_fx)
    monkeypatch.setattr(api_module, "_market_commodities_payload", fake_commodity)

    client = TestClient(app)
    first = client.get("/market/watch")
    second = client.get("/market/watch")

    assert first.status_code == 200
    assert second.status_code == 200
    assert counters["quote"][0] == 3
    assert counters["fx"][0] == 1
    assert counters["commodity"][0] == 1


def test_market_watch_handles_partial_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(_yahoo_symbol: str) -> Dict[str, Any]:
        return {"ok": False, "error": "provider_down"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(
        api_module,
        "_market_fx_payload",
        lambda: _fx_payload_for_watch(
            [
                {
                    "symbol": "USD/TRY",
                    "label": "Amerikan Doları",
                    "yahoo_symbol": "USDTRY=X",
                    "price": 38.52,
                    "prev_close": 38.40,
                    "change": 0.12,
                    "change_pct": 0.31,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        api_module,
        "_market_commodities_payload",
        lambda: _commodity_payload_for_watch(
            [
                {
                    "symbol": "BRENT",
                    "label": "Brent Petrol",
                    "yahoo_symbol": "BZ=F",
                    "price": 86.12,
                    "prev_close": 85.70,
                    "change": 0.42,
                    "change_pct": 0.49,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                }
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()
    sections = payload["sections"]

    index_xu100 = next(row for row in sections["indices"] if row["symbol"] == "XU100")
    fx_eur = next(row for row in sections["fx"] if row["symbol"] == "EUR/TRY")
    commodity_altin = next(row for row in sections["commodities"] if row["symbol"] == "ALTIN")

    assert index_xu100["price"] is None
    assert index_xu100["error"]
    assert fx_eur["price"] is None
    assert fx_eur["error"] == "instrument_not_found"
    assert commodity_altin["price"] is None
    assert commodity_altin["error"] == "instrument_not_found"


def test_market_watch_global_returns_world_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        if yahoo_symbol == "^GSPC":
            return _watch_quote(price=5100.0, prev_close=5000.0, currency="USD")
        if yahoo_symbol == "^IXIC":
            return _watch_quote(price=16000.0, prev_close=15920.0, currency="USD")
        return {"ok": False, "error": "provider_down"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)

    client = TestClient(app)
    response = client.get("/market/watch/global")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "yahoo_finance_chart"
    symbols = [row["symbol"] for row in payload["items"]]
    assert symbols[:2] == ["SP500", "NASDAQ"]
    assert "DAX" in symbols
    items = {row["symbol"]: row for row in payload["items"]}
    assert items["SP500"]["price"] == 5100.0
    assert items["SP500"]["currency"] == "USD"
    assert items["DAX"]["price"] is None
    assert items["DAX"]["error"]


def test_market_watch_global_reuses_own_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        calls.append(yahoo_symbol)
        return _watch_quote(price=100.0, prev_close=99.0, currency="USD")

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)

    client = TestClient(app)
    first = client.get("/market/watch/global")
    second = client.get("/market/watch/global")

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == len(api_module._WATCH_GLOBAL_INDEX_CANDIDATES)
    assert first.json()["as_of"] == second.json()["as_of"]


# region VYK feed unit tests ------------------------------------------------


def _iso_hours_ago(hours: float) -> str:
    dt = datetime.now() - timedelta(hours=hours)
    # VYK `time` format: "dd.mm.YYYY HH:MM:SS"
    return dt.strftime("%d.%m.%Y %H:%M:%S")


def _stub_vyk_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    last_index: int,
    disclosures: List[Dict[str, Any]],
    details: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """Replace kap_vyk_client helpers so we can drive _fetch_kap_vyk_feed."""
    calls: Dict[str, List[Any]] = {"last": [0], "list": [], "detail": [], "members": [0]}

    def fake_last(_cfg: Any) -> int:
        calls["last"][0] += 1
        return last_index

    def fake_list(_cfg: Any, *, start_index: int, **_kwargs: Any) -> List[Dict[str, Any]]:
        calls["list"].append(start_index)
        return list(disclosures)

    def fake_detail(_cfg: Any, disclosure_index: Any, *, file_type: str = "data") -> Optional[Dict[str, Any]]:
        idx = str(disclosure_index or "").strip()
        calls["detail"].append(idx)
        return details.get(idx)

    def fake_members(_cfg: Any) -> Dict[str, Dict[str, Any]]:
        calls["members"][0] += 1
        return {}

    monkeypatch.setattr(
        "src.kap_vyk_client.get_last_disclosure_index", fake_last, raising=False
    )
    monkeypatch.setattr(
        "src.kap_vyk_client.list_disclosures_batch", fake_list, raising=False
    )
    monkeypatch.setattr(
        "src.kap_vyk_client.get_disclosure_detail", fake_detail, raising=False
    )
    monkeypatch.setattr(
        "src.kap_vyk_client.build_company_lookup", fake_members, raising=False
    )
    monkeypatch.setattr("src.kap_vyk_client.is_enabled", lambda _cfg: True, raising=False)
    return calls


def _list_row(idx: int, disclosure_class: str = "ODA", disclosure_type: str = "ODA") -> Dict[str, Any]:
    return {
        "disclosureIndex": str(idx),
        "disclosureClass": disclosure_class,
        "disclosureType": disclosure_type,
        "title": "Örnek Şirket A.Ş.",
        "companyId": "1",
    }


def _detail(
    idx: int,
    *,
    hours_ago: float,
    subject_tr: str = "Örnek Konu",
    stock: str = "BIMAS",
    disclosure_class: str = "ODA",
    disclosure_type: str = "ODA",
) -> Dict[str, Any]:
    return {
        "disclosureIndex": str(idx),
        "disclosureClass": disclosure_class,
        "disclosureType": disclosure_type,
        "senderExchCodes": [stock],
        "senderTitle": "Örnek Şirket A.Ş.",
        "subject": {"tr": subject_tr, "en": "Sample"},
        "summary": {"tr": subject_tr, "en": "Sample"},
        "time": _iso_hours_ago(hours_ago),
    }


def test_vyk_feed_drops_items_outside_24h_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pencere icinde en az 1 kayit varsa pencere disi kalanlar atilir.
    disclosures = [_list_row(101), _list_row(100)]
    details = {
        "101": _detail(101, hours_ago=2),
        "100": _detail(100, hours_ago=48),
    }
    _stub_vyk_client(
        monkeypatch,
        last_index=101,
        disclosures=disclosures,
        details=details,
    )

    items = api_module._fetch_kap_vyk_feed()

    ids = [row["id"] for row in items]
    assert ids == ["vyk-101"]


def test_vyk_feed_falls_back_when_window_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pencere icinde hic kayit yoksa, detayi alinmis en yeni kayitlari yine
    # gosteriyoruz. Test gateway'i eski sabit veri dondurdugunde akisi bos
    # birakmamak icin bu davranis kritik.
    disclosures = [_list_row(101), _list_row(100)]
    details = {
        "101": _detail(101, hours_ago=48),
        "100": _detail(100, hours_ago=72),
    }
    _stub_vyk_client(
        monkeypatch,
        last_index=101,
        disclosures=disclosures,
        details=details,
    )

    items = api_module._fetch_kap_vyk_feed()

    ids = [row["id"] for row in items]
    assert sorted(ids) == ["vyk-100", "vyk-101"]


def test_vyk_feed_respects_detail_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    disclosures = [_list_row(i) for i in range(110, 100, -1)]
    details = {str(i): _detail(i, hours_ago=1) for i in range(101, 111)}
    calls = _stub_vyk_client(
        monkeypatch,
        last_index=110,
        disclosures=disclosures,
        details=details,
    )

    items = api_module._fetch_kap_vyk_feed(detail_budget=3, list_pages=1)

    # Budget is the hard cap on detail fetches; feed length must not exceed it.
    assert len(items) <= 3
    assert len(calls["detail"]) <= 3


def test_vyk_feed_empty_when_credentials_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Disable the client entirely and make sure no network helpers are invoked.
    monkeypatch.setattr("src.kap_vyk_client.is_enabled", lambda _cfg: False, raising=False)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("VYK upstream cagrilmamali")

    monkeypatch.setattr(
        "src.kap_vyk_client.get_last_disclosure_index", _boom, raising=False
    )
    assert api_module._fetch_kap_vyk_feed() == []


# endregion
