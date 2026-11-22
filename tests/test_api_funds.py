from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api_module_facade import api_module
from app import cache as cache_module
from app import database as database_module
from app import fund_service as fund_service_module
from app.api import app
from app.routers import funds as funds_router
from api_test_helpers import (
    _get_ready_fund,
    _reset_flow_state,
)


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


def test_complete_tefas_history_schedules_one_fintables_probe() -> None:
    payload = {
        "points": [{"date": "2026-03-10", "price": 10.0}, {"date": "2026-09-10", "price": 11.0}],
        "source_metadata": {
            "history_source_used": "tefasfon_funds",
            "resolution": "daily",
            "coverage_state": "complete",
            "available_start_date": "2026-03-10",
            "available_end_date": "2026-09-10",
            "internal_gap_count": 0,
        },
    }
    target = api_module._history_request_range(date(2026, 3, 10), date(2026, 9, 10))

    assert api_module._history_job_should_schedule(payload, last_job=None, target=target) is True
    completed_probe = {
        "job_id": "fintables-probe-1",
        "status": "succeeded",
        "effective_start": "2026-03-10",
        "effective_end": "2026-09-10",
        "requested_start": "2026-03-10",
        "requested_end": "2026-09-10",
        "fintables_point_count": 0,
    }
    assert api_module._history_job_should_schedule(payload, last_job=completed_probe, target=target) is False


def test_head_limited_fintables_history_schedules_recovery_backfill() -> None:
    payload = {
        "points": [
            {"date": "2026-04-01", "price": 1.0},
            {"date": "2026-09-10", "price": 2.0},
        ],
        "source_metadata": {
            "history_source_used": "fintables_udf_history",
            "resolution": "daily",
            "coverage_state": "range_incomplete",
            "available_start_date": "2026-04-01",
            "available_end_date": "2026-09-10",
            "internal_gap_count": 0,
        },
    }
    target = api_module._history_request_range(date(2026, 3, 10), date(2026, 9, 10))

    assert api_module._history_job_should_schedule(payload, last_job=None, target=target) is True


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


def test_fund_performance_refresh_bypasses_response_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_performance(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {
            "fund_code": fund_code,
            "status": "ok",
            "points": [{"date": "2026-08-18", "price": 10.0}],
            "period_stats": {},
            "source_metadata": {
                "resolution": "daily",
                "coverage_state": "complete",
                "available_start_date": "2026-01-22",
                "available_end_date": "2026-08-18",
                "internal_gap_count": 0,
            },
        }

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_performance)

    client = TestClient(app)
    params = {"start_date": "2026-01-22", "end_date": "2026-08-18"}
    first = client.get("/funds/REFRESHTHF/performance", params=params)
    refreshed = client.get(
        "/funds/REFRESHTHF/performance",
        params={**params, "refresh": "true"},
    )

    assert first.status_code == 200
    assert refreshed.status_code == 200
    assert len(calls) == 2


def test_single_fund_history_invalidation_clears_versioned_performance_cache() -> None:
    backend = cache_module.get_cache()
    backend.set("api:fund-performance:v3:THF:2026-01-22:2026-08-18:fb=0:refresh=0", {"stale": True})

    api_module._invalidate_single_fund_response_cache("THF")

    assert backend.get("api:fund-performance:v2:THF:2026-01-22:2026-08-18:fb=0:refresh=0") is None


def test_snapshot_refresh_still_returns_the_current_payload_and_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_start_fund_refresh_job",
        lambda _lookback_days: {"job_id": "refresh-job", "status": "queued"},
    )
    monkeypatch.setattr(
        fund_service_module,
        "get_funds_payload",
        lambda *_args, **_kwargs: {"status": "ok", "rows": []},
    )

    response = TestClient(app).post(
        "/admin/funds/refresh-snapshot",
        headers={"Authorization": "Bearer test-admin-token"},
    )

    assert response.status_code == 200
    assert response.json()["refresh_job"]["job_id"] == "refresh-job"


def test_snapshot_refresh_include_rows_false_skips_catalogue_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_start_fund_refresh_job",
        lambda _lookback_days: {"job_id": "refresh-job-lite", "status": "queued"},
    )
    monkeypatch.setattr(
        fund_service_module,
        "get_funds_payload",
        lambda *_args, **_kwargs: pytest.fail("include_rows=false must skip get_funds_payload"),
    )

    response = TestClient(app).post(
        "/admin/funds/refresh-snapshot",
        params={"lookback_days": 5, "include_rows": "false"},
        headers={"Authorization": "Bearer test-admin-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "refresh_job": {"job_id": "refresh-job-lite", "status": "queued"},
    }


def test_run_fund_refresh_job_skips_commit_and_invalidation_when_retained_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = cache_module.get_cache()
    backend.set("api:funds:keep", {"ok": True}, ttl_seconds=60)
    job = {
        "job_id": "retained-current-job",
        "status": "queued",
        "requested_at": "2026-05-20T09:00:00+00:00",
        "started_at": None,
        "finished_at": None,
        "as_of": None,
        "row_count": None,
        "cache_invalidation_status": None,
        "error": None,
        "owner_token": "owner-retained",
        "generation": 5,
    }
    api_module._set_fund_refresh_job(job)
    backend.set(api_module._FUND_REFRESH_ACTIVE_KEY, job["job_id"], ttl_seconds=600)
    monkeypatch.setattr(
        fund_service_module,
        "refresh_funds_snapshot",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "rows": [{"fund_code": "TLY"}],
            "resolution_status": "available",
            "as_of": "2026-05-20",
            "source_metadata": {"snapshot_action": "retained_current", "resolved_as_of": "2026-05-20"},
        },
    )
    monkeypatch.setattr(
        fund_service_module,
        "commit_funds_snapshot",
        lambda *_args, **_kwargs: pytest.fail("retained_current must not re-commit snapshot"),
    )

    api_module._run_fund_refresh_job(job["job_id"], 5)

    completed_job = api_module._get_fund_refresh_job(job["job_id"])
    assert completed_job["status"] == "succeeded"
    assert completed_job["snapshot_action"] == "retained_current"
    assert completed_job["cache_invalidation_status"] == "skipped_unchanged"
    assert backend.get("api:funds:keep") == {"ok": True}


def test_api_allocation_history_returns_pending_without_synchronous_upstream_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "_require_known_fund_code", lambda fund_code: "TLY")
    monkeypatch.setattr(
        fund_service_module,
        "get_fund_allocations_history_payload",
        lambda *_args, **_kwargs: {
            "fund_code": "TLY",
            "status": "pending",
            "lookback_days": 30,
            "history": [],
            "source": "tefasfon_portfolio",
            "stale": True,
            "source_metadata": {"source": "tefasfon_portfolio"},
        },
    )
    monkeypatch.setattr(
        api_module,
        "_start_allocation_history_refresh_job",
        lambda *_args, **_kwargs: {
            "job_id": "job-1",
            "fund_code": "TLY",
            "lookback_days": 30,
            "status": "queued",
            "requested_at": "2026-09-06T12:00:00+00:00",
            "started_at": None,
            "finished_at": None,
            "error": None,
        },
    )

    response = TestClient(app).get("/funds/TLY/allocations/history?lookback_days=30")

    assert response.status_code == 202
    assert response.json()["refresh_pending"] is True
    assert response.json()["source_metadata"]["allocation_history_job"]["job_id"] == "job-1"


def test_api_allocations_queues_refresh_for_stale_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "_require_known_fund_code", lambda _fund_code: "TLY")
    monkeypatch.setattr(
        fund_service_module,
        "get_fund_allocations_payload",
        lambda *_args, **_kwargs: {
            "fund_code": "TLY",
            "status": "ok",
            "allocations": [{"allocation_type": "hs", "weight": 80.3}],
            "source": "tefasfon_portfolio",
            "stale": True,
            "source_metadata": {"as_of": "2026-09-02"},
        },
    )
    monkeypatch.setattr(
        api_module,
        "_start_allocation_history_refresh_job",
        lambda *_args, **_kwargs: {
            "job_id": "job-2",
            "fund_code": "TLY",
            "lookback_days": 30,
            "status": "queued",
            "requested_at": "2026-09-09T12:00:00+00:00",
            "started_at": None,
            "finished_at": None,
            "error": None,
        },
    )

    response = TestClient(app).get("/funds/TLY/allocations")

    assert response.status_code == 200
    payload = response.json()
    assert payload["refresh_pending"] is True
    assert payload["source_metadata"]["allocation_history_job"]["job_id"] == "job-2"


def test_allocation_history_refresh_job_is_coalesced_per_fund_and_range(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[tuple[Any, ...]] = []

    class FakeExecutor:
        def submit(self, *args: Any) -> None:
            submitted.append(args)

    monkeypatch.setattr(api_module, "_FUND_ALLOCATION_HISTORY_EXECUTOR", FakeExecutor())

    first = api_module._start_allocation_history_refresh_job("TLY", 30)
    second = api_module._start_allocation_history_refresh_job("TLY", 30)

    assert first["job_id"] == second["job_id"]
    assert len(submitted) == 1


def test_unknown_fund_cannot_enqueue_allocation_history_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    def unknown(_fund_code: str) -> str:
        raise HTTPException(status_code=404, detail="Fon bulunamadi")

    monkeypatch.setattr(api_module, "_require_known_fund_code", unknown)
    monkeypatch.setattr(
        api_module,
        "_start_allocation_history_refresh_job",
        lambda *_args, **_kwargs: pytest.fail("unknown funds must not enqueue upstream work"),
    )

    response = TestClient(app).get("/funds/NOTREAL/allocations/history")

    assert response.status_code == 404


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


def test_api_funds_list_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_kwargs: Dict[str, Any] = {}
    payload_calls = {"count": 0}

    def fake_get_funds_payload(processed_dir: Any, **kwargs: Any) -> Dict[str, Any]:
        payload_calls["count"] += 1
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
    monkeypatch.setattr(
        funds_router,
        "_funds_listing_preflight",
        lambda: (
            {
                "status": "ok",
                "total_count": 1,
                "stale": False,
                "degraded": False,
                "source_metadata": {
                    "source": "tefasfon_funds",
                    "snapshot_as_of": "2026-04-28",
                    "snapshot_as_of_lag_days": 0,
                },
            },
            {"catalog_generation": 1, "price_generation": 0},
        ),
    )
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds")
    repeated = client.get("/funds")

    assert response.status_code == 200
    assert repeated.status_code == 200
    assert repeated.json() == response.json()
    assert payload_calls["count"] == 1
    payload = response.json()
    assert payload["rows"][0]["fund_code"] == "YAC"
    assert payload["source"] == "tefasfon_funds"
    assert seen_kwargs["auto_refresh"] is False


def test_pending_catalogue_response_does_not_trigger_upstream_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module, "_start_fund_refresh_job",
        lambda _: pytest.fail("pending response must not trigger snapshot recovery"),
    )
    payload = {"status": "pending", "rows": [], "refresh_pending": True}
    assert api_module._maybe_schedule_fund_snapshot_refresh(payload) is payload


def test_stale_fund_catalogue_schedules_one_background_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs: list[int] = []
    monkeypatch.setattr(
        api_module,
        "_start_fund_refresh_job",
        lambda lookback_days: jobs.append(lookback_days) or {
            "job_id": "auto-refresh-job",
            "status": "queued",
            "requested_at": "2026-09-07T12:00:00+00:00",
        },
    )
    backend = cache_module.get_cache()
    backend.delete(f"{api_module._FUND_REFRESH_AUTO_KEY_PREFIX}:2026-09-07")
    payload = {
        "status": "ok",
        "rows": [{"fund_code": "TLY"}],
        "total_count": 1,
        "as_of": "2026-09-04",
        "stale": True,
        "source_metadata": {
            "snapshot_as_of": "2026-09-04",
            "snapshot_target_date": "2026-09-07",
            "snapshot_as_of_lag_days": 3,
        },
    }

    first = api_module._maybe_schedule_fund_snapshot_refresh(payload)
    second = api_module._maybe_schedule_fund_snapshot_refresh(payload)

    assert jobs == [api_module._FUND_REFRESH_MAX_LOOKBACK_DAYS]
    assert first["refresh_pending"] is True
    assert first["refresh_job"]["job_id"] == "auto-refresh-job"
    assert "owner_token" not in first["refresh_job"]
    assert "refresh_job" not in second


def test_current_weekend_fund_catalogue_does_not_schedule_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_start_fund_refresh_job",
        lambda _lookback_days: pytest.fail("current market snapshot should not schedule a refresh"),
    )
    payload = {
        "status": "ok",
        "rows": [{"fund_code": "TLY"}],
        "total_count": 1,
        "as_of": "2026-09-04",
        "stale": True,
        "source_metadata": {
            "snapshot_as_of": "2026-09-04",
            "snapshot_target_date": "2026-09-04",
            "snapshot_as_of_lag_days": 0,
        },
    }

    assert api_module._maybe_schedule_fund_snapshot_refresh(payload) is payload


def test_truncated_current_fund_catalogue_schedules_repair_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs: list[int] = []
    monkeypatch.setattr(
        api_module,
        "_start_fund_refresh_job",
        lambda lookback_days: jobs.append(lookback_days) or {
            "job_id": "truncated-repair-job",
            "status": "queued",
            "requested_at": "2026-09-08T12:00:00+00:00",
        },
    )
    backend = cache_module.get_cache()
    backend.delete(f"{api_module._FUND_REFRESH_AUTO_KEY_PREFIX}:2026-09-08")
    payload = {
        "status": "ok",
        "rows": [{"fund_code": "TLY"}],
        "total_count": 2041,
        "as_of": "2026-09-08",
        "stale": False,
        "warnings": ["tefasfon snapshot looked truncated (1000 rows vs cached 2041)"],
        "source_metadata": {
            "snapshot_as_of": "2026-09-08",
            "snapshot_target_date": "2026-09-08",
            "snapshot_as_of_lag_days": 0,
            "truncated_refresh_observed": True,
        },
    }

    refreshed = api_module._maybe_schedule_fund_snapshot_refresh(payload)

    assert jobs == [api_module._FUND_REFRESH_MAX_LOOKBACK_DAYS]
    assert refreshed["refresh_pending"] is True
    assert refreshed["refresh_job"]["job_id"] == "truncated-repair-job"


def test_failed_auto_refresh_does_not_hold_repair_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs: list[int] = []
    backend = cache_module.get_cache()
    cooldown_key = f"{api_module._FUND_REFRESH_AUTO_KEY_PREFIX}:2026-09-08"
    backend.delete(cooldown_key)
    first_job = {
        "job_id": "failed-auto-refresh-job",
        "status": "failed",
        "requested_at": "2026-09-08T12:00:00+00:00",
    }
    api_module._set_fund_refresh_job(first_job)
    backend.set(cooldown_key, {"job_id": first_job["job_id"]}, ttl_seconds=3600)
    monkeypatch.setattr(
        api_module,
        "_start_fund_refresh_job",
        lambda lookback_days: jobs.append(lookback_days) or {
            "job_id": "retry-auto-refresh-job",
            "status": "queued",
            "requested_at": "2026-09-08T12:01:00+00:00",
        },
    )
    payload = {
        "status": "ok",
        "rows": [{"fund_code": "TLY"}],
        "total_count": 2041,
        "as_of": "2026-09-08",
        "stale": False,
        "warnings": ["tefasfon snapshot looked truncated (1000 rows vs cached 2041)"],
        "source_metadata": {
            "snapshot_as_of": "2026-09-08",
            "snapshot_target_date": "2026-09-08",
            "truncated_refresh_observed": True,
        },
    }

    refreshed = api_module._maybe_schedule_fund_snapshot_refresh(payload)

    assert jobs == [api_module._FUND_REFRESH_MAX_LOOKBACK_DAYS]
    assert refreshed["refresh_job"]["job_id"] == "retry-auto-refresh-job"


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
    assert payload["total_count"] == 2
    assert seen_kwargs["min_aum"] is None
    assert seen_kwargs["limit"] == 1


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

    first = _get_ready_fund(client, "/funds/TLY/yield-summary")
    second = _get_ready_fund(client, "/funds/TLY/yield-summary")

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

    first = _get_ready_fund(client, "/funds/TLY/holdings")
    second = _get_ready_fund(client, "/funds/TLY/holdings")

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
    monkeypatch.setattr(fund_service_module, "commit_funds_snapshot", lambda *_args, **_kwargs: True)
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

    completed_job = api_module._get_fund_refresh_job("test-refresh-job")
    assert completed_job["status"] == "succeeded"
    assert completed_job["cache_invalidation_status"] == "complete"
    assert backend.get("api:funds:q=|type=|founder=|manager=|risk=|sort=fund_code|order=asc") is None
    assert backend.get("api:funds:v2:q=|type=|founder=|manager=|risk=|sort=fund_code|order=asc") is None
    assert backend.get("api:funds-search:q=") is None
    assert backend.get("api:funds-categories") is None
    assert backend.get("api:funds-categories:v2") is None
    assert backend.get("api:fund-yield-summary:TLY") is None
    assert backend.get("api:fund-holdings:TLY") is None


def test_admin_snapshot_refresh_job_reports_deferred_cache_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cache_module.get_cache()
    job = {
        "job_id": "deferred-refresh-job",
        "status": "queued",
        "requested_at": "2026-05-20T09:00:00+00:00",
        "started_at": None,
        "finished_at": None,
        "as_of": None,
        "row_count": None,
        "cache_invalidation_status": None,
        "error": None,
        "owner_token": "deferred-refresh-owner",
        "generation": 0,
    }
    api_module._set_fund_refresh_job(job)
    backend.set(api_module._FUND_REFRESH_ACTIVE_KEY, job["job_id"], ttl_seconds=600)
    monkeypatch.setattr(
        fund_service_module,
        "refresh_funds_snapshot",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "rows": [{"fund_code": "TLY"}],
            "resolution_status": "available",
            "as_of": "2026-05-20",
            "source_metadata": {"snapshot_action": "updated"},
        },
    )
    monkeypatch.setattr(fund_service_module, "commit_funds_snapshot", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(api_module, "_invalidate_fund_response_cache", lambda: "deferred")

    api_module._run_fund_refresh_job(job["job_id"], 1)

    completed_job = api_module._get_fund_refresh_job(job["job_id"])
    assert completed_job["status"] == "succeeded"
    assert completed_job["cache_invalidation_status"] == "deferred"


def test_fund_response_invalidation_reports_deferred_when_redis_fell_back_to_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = cache_module.InMemoryCache()
    backend.set("api:funds:stale", {"old": True}, ttl_seconds=60)
    monkeypatch.setattr(api_module, "_get_cache", lambda: backend)
    monkeypatch.setattr(api_module, "_cache_status", lambda: {"cache_redis_fallback": True})

    status = api_module._invalidate_fund_response_cache()

    assert status == "deferred"
    assert backend.get("api:funds:stale") is None


def test_nested_fund_maps_are_invalidated_by_the_parent_prefix_without_extra_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TrackingCache(cache_module.InMemoryCache):
        def __init__(self) -> None:
            super().__init__()
            self.deleted_prefixes: List[str] = []

        def delete_prefix_with_status(self, prefix: str):
            self.deleted_prefixes.append(prefix)
            return super().delete_prefix_with_status(prefix)

    backend = _TrackingCache()
    backend.set("api:funds:snapshot-row-map:mtime=1:v1", {"stale": True}, ttl_seconds=60)
    backend.set("api:funds:holding-sector-map:v1", {"stale": True}, ttl_seconds=60)
    monkeypatch.setattr(api_module, "_get_cache", lambda: backend)
    monkeypatch.setattr(api_module, "_cache_status", lambda: {"cache_redis_fallback": False})

    status = api_module._invalidate_fund_response_cache()

    assert status == "complete"
    assert backend.get("api:funds:snapshot-row-map:mtime=1:v1") is None
    assert backend.get("api:funds:holding-sector-map:v1") is None
    assert backend.deleted_prefixes.count("api:funds:") == 1
    assert "api:funds:snapshot-row-map:" not in backend.deleted_prefixes
    assert "api:funds:holding-sector-map:" not in backend.deleted_prefixes


@pytest.mark.parametrize(
    ("path", "params", "service_name"),
    [
        (
            "/admin/funds/THF/refresh-performance",
            {"start_date": "2026-01-01", "end_date": "2026-01-02"},
            "refresh_fund_performance",
        ),
        (
            "/admin/funds/THF/refresh-allocations",
            {},
            "refresh_fund_allocations",
        ),
    ],
)
def test_single_fund_refresh_responses_report_cache_invalidation_status(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    params: Dict[str, str],
    service_name: str,
) -> None:
    monkeypatch.setattr(funds_router, "_require_known_fund_code", lambda fund_code: fund_code)
    monkeypatch.setattr(
        fund_service_module,
        service_name,
        lambda *_args, **_kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(funds_router, "_invalidate_single_fund_response_cache", lambda _code: "deferred")

    response = TestClient(app).post(
        path,
        params=params,
        headers={"Authorization": "Bearer test-admin-token"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["cache_invalidation_status"] == "deferred"


def test_truncated_fund_refresh_is_failed_without_committing(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cache_module.get_cache()
    job = {
        "job_id": "truncated-refresh-job",
        "status": "queued",
        "requested_at": "2026-09-08T10:00:00+00:00",
        "started_at": None,
        "finished_at": None,
        "as_of": None,
        "row_count": None,
        "error": None,
    }
    api_module._set_fund_refresh_job(job)
    backend.set(api_module._FUND_REFRESH_ACTIVE_KEY, job["job_id"], ttl_seconds=600)
    monkeypatch.setattr(
        fund_service_module,
        "refresh_funds_snapshot",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "rows": [{"fund_code": "TLY"}],
            "resolution_status": "available",
            "warnings": ["tefasfon snapshot looked truncated (1000 rows vs cached 2041)"],
            "source_metadata": {
                "snapshot_action": "retained_existing",
                "truncated_refresh_observed": True,
            },
        },
    )
    monkeypatch.setattr(
        fund_service_module,
        "commit_funds_snapshot",
        lambda *_args, **_kwargs: pytest.fail("truncated snapshot must not be committed"),
    )

    api_module._run_fund_refresh_job(job["job_id"], 14)

    final_job = api_module._get_fund_refresh_job(job["job_id"])
    assert final_job is not None
    assert final_job["status"] == "failed"
    assert final_job["snapshot_action"] == "retained_existing"
    assert "truncated" in final_job["error"]


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


def test_api_fund_allocations_history_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_history(
        processed_dir: Any,
        fund_code: str,
        *,
        lookback_days: int = 30,
        auto_refresh: bool = True,
    ) -> Dict[str, Any]:
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

    monkeypatch.setattr(api_module, "_require_known_fund_code", lambda fund_code: str(fund_code).upper())
    monkeypatch.setattr(fund_service_module, "get_fund_allocations_history_payload", fake_history)
    client = TestClient(app)

    response = client.get("/funds/TLY/allocations/history?lookback_days=30")

    assert response.status_code == 200
    payload = response.json()
    assert payload["fund_code"] == "TLY"
    assert payload["history"][0]["allocations"][0]["label"] == "Hisse Senedi"


def test_fund_detail_categories_search_payload_caches_and_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fund_service as fund_service_module

    monkeypatch.setattr(
        fund_service_module,
        "fund_listing_revision",
        lambda _dir: {"catalog_generation": 7, "price_generation": 11},
    )

    detail_calls = 0
    categories_calls = 0
    search_calls = 0

    def fake_detail(_dir, fund_code: str):
        nonlocal detail_calls
        detail_calls += 1
        return {"status": "ok", "fund_code": fund_code, "name": "TERA FON"}

    def fake_categories(_dir):
        nonlocal categories_calls
        categories_calls += 1
        return {
            "status": "ok",
            "fund_types": [{"code": "SEC", "label": "Menkul"}],
            "categories": ["Menkul"],
        }

    def fake_search(_dir, **_kwargs):
        nonlocal search_calls
        search_calls += 1
        return {
            "status": "ok",
            "rows": [{"fund_code": "TLY", "name": "TERA FON"}],
            "total": 1,
            "stale": False,
        }

    monkeypatch.setattr(fund_service_module, "get_fund_detail_payload", fake_detail)
    monkeypatch.setattr(fund_service_module, "get_fund_categories_payload", fake_categories)
    monkeypatch.setattr(fund_service_module, "get_funds_payload", fake_search)

    assert api_module._fund_detail_payload(normalized="TLY")["fund_code"] == "TLY"
    assert api_module._fund_detail_payload(normalized="TLY")["fund_code"] == "TLY"
    assert detail_calls == 1

    assert api_module._funds_categories_payload()["categories"] == ["Menkul"]
    assert api_module._funds_categories_payload()["categories"] == ["Menkul"]
    assert categories_calls == 1

    assert api_module._funds_search_payload(q="TLY", limit=10)["total"] == 1
    assert api_module._funds_search_payload(q="TLY", limit=10)["total"] == 1
    assert search_calls == 1

    api_module._invalidate_fund_response_cache()

    assert api_module._fund_detail_payload(normalized="TLY")["fund_code"] == "TLY"
    assert api_module._funds_categories_payload()["categories"] == ["Menkul"]
    assert api_module._funds_search_payload(q="TLY", limit=10)["total"] == 1
    assert detail_calls == 2
    assert categories_calls == 2
    assert search_calls == 2


def test_fund_performance_omits_etag_for_incomplete_or_running_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_performance(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "fund_code": fund_code,
            "status": "ok",
            "points": [{"date": "2025-09-10", "price": 100.0}],
            "period_stats": {},
            "stale": False,
            "source_metadata": {
                "resolution": "daily",
                "coverage_state": "range_incomplete",
                "available_start_date": "2025-09-10",
                "available_end_date": "2026-11-20",
                "internal_gap_count": 0,
            },
        }

    class FakeExecutor:
        def submit(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_performance)
    monkeypatch.setattr(api_module, "_FUND_HISTORY_EXECUTOR", FakeExecutor())

    response = TestClient(app).get(
        "/funds/YAY/performance",
        params={"start_date": "2023-11-22", "end_date": "2026-11-22"},
    )

    assert response.status_code == 200
    assert "etag" not in {k.lower(): v for k, v in response.headers.items()}
    assert response.json()["source_metadata"]["coverage_state"] == "range_incomplete"
    assert response.json()["source_metadata"]["history_job"]["status"] == "queued"


def test_fund_performance_long_range_waits_inline_and_returns_completed_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import concurrent.futures
    import threading
    import time

    ready = threading.Event()
    calls: List[str] = []

    def fake_get_performance(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        start_d = kwargs.get("start_date")
        calls.append(start_d.isoformat() if start_d else "full")
        if ready.is_set():
            return {
                "fund_code": fund_code,
                "status": "ok",
                "points": [
                    {"date": "2021-11-22", "price": 249.17}
                    if (start_d and start_d <= date(2021, 11, 22))
                    else {"date": "2023-11-22", "price": 551.13},
                    {"date": "2026-11-20", "price": 1634.29},
                ],
                "period_stats": {},
                "stale": False,
                "source_metadata": {
                    "resolution": "daily",
                    "coverage_state": "complete",
                    "available_start_date": "2021-11-22",
                    "available_end_date": "2026-11-20",
                    "internal_gap_count": 0,
                    "chart_granularity": "weekly",
                },
            }
        return {
            "fund_code": fund_code,
            "status": "ok",
            "points": [{"date": "2025-09-10", "price": 1076.64}, {"date": "2026-11-20", "price": 1634.29}],
            "period_stats": {},
            "stale": False,
            "source_metadata": {
                "resolution": "daily",
                "coverage_state": "range_incomplete",
                "available_start_date": "2025-09-10",
                "available_end_date": "2026-11-20",
                "internal_gap_count": 0,
                "chart_granularity": "weekly",
            },
        }

    def fake_refresh(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        time.sleep(0.05)
        ready.set()
        return {"fund_code": fund_code, "status": "ok", "points": [{"date": "2023-11-22", "price": 551.13}]}

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_get_performance)
        monkeypatch.setattr(fund_service_module, "refresh_fund_performance", fake_refresh)
        monkeypatch.setattr(api_module, "_FUND_HISTORY_EXECUTOR", executor)
        monkeypatch.setattr(api_module, "_FUND_HISTORY_INLINE_WAIT_SECONDS", 2.0)

        client = TestClient(app)
        response = client.get(
            "/funds/YAYINLINE/performance",
            params={"start_date": "2023-11-22", "end_date": "2026-11-22"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["source_metadata"]["coverage_state"] == "complete"
        assert payload["source_metadata"]["history_job"]["status"] == "succeeded"
        # Widened 5Y effective_start lets a subsequent 5Y request skip re-queuing
        assert payload["source_metadata"]["history_job"]["effective_start"] <= "2021-11-22"
        assert payload["points"][0]["date"] == "2023-11-22"
        assert "etag" in {k.lower() for k in response.headers}
    finally:
        executor.shutdown(wait=True)

