from __future__ import annotations

import json
import types
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app import cache as cache_module
from app import financial_snapshot_store as snapshot_store
from app import kap_statement_refresh as statement_module
from app.api import app
from app.routers import kap as kap_router


ADMIN_HEADERS = {"Authorization": "Bearer test-admin-token"}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGFIN_CACHE_BACKEND", raising=False)
    monkeypatch.delenv("RAGFIN_REDIS_URL", raising=False)
    monkeypatch.setenv("RAGFIN_ADMIN_REFRESH_TOKEN", "test-admin-token")
    cache_module.reset_cache_for_tests()
    yield
    cache_module.reset_cache_for_tests()


@pytest.fixture
def statement_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    processed = tmp_path / "processed"
    cache_dir = processed / "kap_cache"
    cache_dir.mkdir(parents=True)
    config = types.SimpleNamespace(
        paths=types.SimpleNamespace(processed_dir=processed),
        kap=types.SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(statement_module, "CONFIG", config)
    monkeypatch.setattr(statement_module, "_KAP_STATEMENT_REQUEST_GAP_SECONDS", 0.0)
    # Snapshot persistence must never touch a real database in these tests.
    monkeypatch.setattr(
        snapshot_store,
        "extract_financial_snapshot_from_kap_payload",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        snapshot_store,
        "upsert_financial_snapshots",
        lambda *_args, **_kwargs: 0,
    )
    return types.SimpleNamespace(processed=processed, cache_dir=cache_dir)


def _write_statement_cache(
    cache_dir,
    symbol: str,
    *,
    quarters: int = 5,
    schema_version: int = 15,
) -> Dict[str, Any]:
    payload = {
        "ok": True,
        "company": symbol,
        "company_title": symbol,
        "schema_version": schema_version,
        "quarters": [
            {"year": 2026, "period": period, "disclosure_index": period}
            for period in range(1, quarters + 1)
        ],
    }
    (cache_dir / f"{symbol}.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _seed_and_run_job(
    *,
    mode: str,
    symbols: List[str],
    max_quarters: int = 5,
    budget_seconds: int = 1500,
    concurrency: int = 1,
    job_id: str = "job-1",
) -> Dict[str, Any]:
    job = {
        "job_id": job_id,
        "status": "queued",
        "mode": mode,
        "universe": "symbols",
        "symbols": symbols,
        "max_quarters": max_quarters,
        "budget_seconds": budget_seconds,
        "concurrency": concurrency,
        "requested_at": statement_module._kap_statement_now_iso(),
        "owner_token": "owner-1",
    }
    statement_module._set_kap_statement_refresh_job(job)
    backend = statement_module._get_cache()
    assert backend.set_if_absent(
        statement_module._KAP_STATEMENT_ACTIVE_KEY,
        {"job_id": job_id, "owner_token": "owner-1"},
        ttl_seconds=600,
    )
    statement_module._run_kap_statement_refresh_job(job_id)
    stored = statement_module._get_kap_statement_refresh_job(job_id)
    assert stored is not None
    return stored


class _AdvancingClock:
    """Deterministic clock: every read moves time forward by ``step`` seconds."""

    def __init__(self, step: float) -> None:
        self.value = 0.0
        self.step = step

    def monotonic(self) -> float:
        self.value += self.step
        return self.value

    def sleep(self, _seconds: float) -> None:
        return None


def test_incremental_only_rechecks_ready_caches(statement_env, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_statement_cache(statement_env.cache_dir, "SHAL", quarters=1)
    _write_statement_cache(statement_env.cache_dir, "OLD1", quarters=5, schema_version=14)
    ready = _write_statement_cache(statement_env.cache_dir, "READ", quarters=5)
    calls: List[Dict[str, Any]] = []

    def fake_snapshot(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {**ready, "ok": True}

    monkeypatch.setattr("app.kap_service.get_kap_snapshot", fake_snapshot)

    job = _seed_and_run_job(mode="incremental", symbols=["MISS", "SHAL", "OLD1", "READ"])

    assert [call["company"] for call in calls] == ["READ"]
    assert calls[0]["use_cache_when_complete"] is True
    assert calls[0]["allow_criteria_fallback"] is False
    assert job["status"] == "succeeded"
    assert job["skipped_missing_cache"] == 1
    assert job["skipped_shallow_cache"] == 1
    assert job["skipped_schema"] == 1
    assert job["skipped_ready_cache"] == 0
    assert job["fetched"] == 1
    assert job["unchanged"] == 1
    assert job["coverage"] == {
        "ready": 1,
        "missing": 1,
        "shallow": 1,
        "schema_outdated": 1,
        "total": 4,
    }


def test_full_backfill_fills_gaps_and_skips_ready_caches(statement_env, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_statement_cache(statement_env.cache_dir, "SHAL", quarters=1)
    _write_statement_cache(statement_env.cache_dir, "READ", quarters=5)
    calls: List[str] = []

    def fake_snapshot(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs["company"])
        return {
            "ok": True,
            "schema_version": 15,
            "quarters": [
                {"year": 2026, "period": period, "disclosure_index": period}
                for period in range(1, 6)
            ],
        }

    monkeypatch.setattr("app.kap_service.get_kap_snapshot", fake_snapshot)

    job = _seed_and_run_job(mode="full", symbols=["MISS", "SHAL", "READ"])

    assert sorted(calls) == ["MISS", "SHAL"]
    assert job["status"] == "succeeded"
    assert job["skipped_ready_cache"] == 1
    assert job["skipped_missing_cache"] == 0
    assert job["skipped_shallow_cache"] == 0
    assert job["updated"] == 2
    assert job["fetched"] == 2


def test_incremental_rejects_snapshot_with_outdated_schema(
    statement_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_statement_cache(statement_env.cache_dir, "READ", quarters=5)
    monkeypatch.setattr(
        "app.kap_service.get_kap_snapshot",
        lambda **_kwargs: {
            "ok": True,
            "schema_version": 14,
            "quarters": [{"year": 2026, "period": 1, "disclosure_index": 1}],
        },
    )

    job = _seed_and_run_job(mode="incremental", symbols=["READ"])

    assert job["status"] == "partial"
    assert job["stop_reason"] == "completed_with_errors"
    assert job["failed"] == 1
    assert job["updated"] == 0
    assert job["errors"][0]["outcome"] == "failed"
    assert "semasi" in job["errors"][0]["error"]


def test_incremental_rejects_stale_cache_snapshot(
    statement_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_statement_cache(statement_env.cache_dir, "READ", quarters=5)
    monkeypatch.setattr(
        "app.kap_service.get_kap_snapshot",
        lambda **_kwargs: {
            "ok": True,
            "schema_version": 15,
            "cache_stale": True,
            "quarters": [{"year": 2026, "period": 1, "disclosure_index": 1}],
        },
    )

    job = _seed_and_run_job(mode="incremental", symbols=["READ"])

    assert job["status"] == "partial"
    assert job["stop_reason"] == "completed_with_errors"
    assert job["failed"] == 1
    assert "bayat" in job["errors"][0]["error"]


def test_full_backfill_rejects_shallow_live_depth(
    statement_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_statement_cache(statement_env.cache_dir, "SHAL", quarters=1)
    monkeypatch.setattr(
        "app.kap_service.get_kap_snapshot",
        lambda **_kwargs: {
            "ok": True,
            "schema_version": 15,
            "live_period_count": 5,
            "quarters": [
                {"year": 2026, "period": period, "disclosure_index": period}
                for period in range(1, 3)
            ],
        },
    )

    job = _seed_and_run_job(mode="full", symbols=["SHAL"])

    assert job["status"] == "partial"
    assert job["stop_reason"] == "completed_with_errors"
    assert job["failed"] == 1
    assert "derinligi" in job["errors"][0]["error"]


def test_full_backfill_accepts_company_with_fewer_available_periods(
    statement_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_statement_cache(statement_env.cache_dir, "NEWCO", quarters=1)
    monkeypatch.setattr(
        "app.kap_service.get_kap_snapshot",
        lambda **_kwargs: {
            "ok": True,
            "schema_version": 15,
            "live_period_count": 2,
            "quarters": [
                {"year": 2026, "period": period, "disclosure_index": period}
                for period in range(1, 3)
            ],
        },
    )

    job = _seed_and_run_job(mode="full", symbols=["NEWCO"])

    assert job["status"] == "succeeded"
    assert job["failed"] == 0
    assert job["updated"] == 1


def test_full_backfill_does_not_inflate_fetch_depth_from_outdated_schema_cache(
    statement_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_statement_cache(statement_env.cache_dir, "OLD20", quarters=20, schema_version=11)
    calls: List[Dict[str, Any]] = []

    def fake_snapshot(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {
            "ok": True,
            "schema_version": 15,
            "live_period_count": 5,
            "quarters": [
                {"year": 2026, "period": period, "disclosure_index": period}
                for period in range(1, 6)
            ],
        }

    monkeypatch.setattr("app.kap_service.get_kap_snapshot", fake_snapshot)

    job = _seed_and_run_job(mode="full", symbols=["OLD20"], max_quarters=5)

    assert len(calls) == 1
    assert calls[0]["max_quarters"] == 5
    assert job["status"] == "succeeded"
    assert job["updated"] == 1


def test_pacing_deadline_exceeded_stops_job_as_budget_exhausted_without_blocked_count(
    statement_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.kap_fetcher import KapPacingDeadlineExceeded

    calls = 0

    def fake_process(symbol: str, **_kwargs: Any) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"symbol": symbol, "outcome": "updated"}
        raise KapPacingDeadlineExceeded("KAP istek sirasi is butcesini asti.")

    monkeypatch.setattr(statement_module, "_process_kap_statement_symbol", fake_process)

    job = _seed_and_run_job(mode="full", symbols=["SYM1", "SYM2", "SYM3"], budget_seconds=300)

    assert job["status"] == "partial"
    assert job["stop_reason"] == "budget_exhausted"
    assert job["processed"] == 1
    assert job["updated"] == 1
    assert job["blocked"] == 0
    assert job["failed"] == 0
    assert job["remaining"] == 2
    assert job["cursor"] == 1


def test_budget_exhaustion_marks_job_partial(statement_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(statement_module, "time", _AdvancingClock(step=20.0))
    monkeypatch.setattr(
        statement_module,
        "_guarded_process_kap_statement_symbol",
        lambda symbol, **_kwargs: {"symbol": symbol, "outcome": "unchanged", "blocked": False},
    )

    job = _seed_and_run_job(
        mode="incremental",
        symbols=[f"S{index}" for index in range(1, 11)],
        budget_seconds=30,
    )

    assert job["status"] == "partial"
    assert job["stop_reason"] == "budget_exhausted"
    assert 0 < job["processed"] < 10
    assert job["remaining"] == 10 - job["processed"]
    assert job["cursor"] == job["processed"]


def test_repeated_provider_blocks_abort_the_job(statement_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(statement_module, "_KAP_STATEMENT_BLOCK_STREAK_LIMIT", 2)
    monkeypatch.setattr(statement_module, "_KAP_STATEMENT_BLOCK_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(
        statement_module,
        "_guarded_process_kap_statement_symbol",
        lambda symbol, **_kwargs: {
            "symbol": symbol,
            "outcome": "blocked",
            "blocked": True,
            "error": "connection reset by peer",
        },
    )

    job = _seed_and_run_job(
        mode="incremental",
        symbols=[f"B{index}" for index in range(1, 6)],
    )

    assert job["status"] == "aborted"
    assert job["stop_reason"] == "provider_blocked"
    assert job["blocked"] == 2
    assert job["processed"] == 2
    assert job["remaining"] == 3
    assert job["errors"][0]["outcome"] == "blocked"


def test_statement_refresh_endpoint_requires_admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kap_router,
        "CONFIG",
        types.SimpleNamespace(kap=types.SimpleNamespace(enabled=True)),
    )

    response = TestClient(app).post("/admin/kap/refresh-statements")

    assert response.status_code == 401


def test_statement_refresh_endpoint_requires_enabled_kap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kap_router,
        "CONFIG",
        types.SimpleNamespace(kap=types.SimpleNamespace(enabled=False)),
    )

    response = TestClient(app).post(
        "/admin/kap/refresh-statements",
        headers=ADMIN_HEADERS,
        params={"universe": "symbols", "symbols": "THYAO"},
    )

    assert response.status_code == 503


def test_statement_refresh_endpoint_maps_validation_and_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kap_router,
        "CONFIG",
        types.SimpleNamespace(kap=types.SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(
        kap_router,
        "_start_kap_statement_refresh_job",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("gecersiz mode")),
    )
    client = TestClient(app)

    invalid = client.post(
        "/admin/kap/refresh-statements",
        headers=ADMIN_HEADERS,
        params={"universe": "symbols", "symbols": "THYAO"},
    )

    monkeypatch.setattr(
        kap_router,
        "_start_kap_statement_refresh_job",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("is baska worker tarafindan devralindi")),
    )
    conflict = client.post(
        "/admin/kap/refresh-statements",
        headers=ADMIN_HEADERS,
        params={"universe": "symbols", "symbols": "THYAO"},
    )

    assert invalid.status_code == 400
    assert conflict.status_code == 409


def test_statement_refresh_endpoint_parses_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kap_router,
        "CONFIG",
        types.SimpleNamespace(kap=types.SimpleNamespace(enabled=True)),
    )
    calls: List[Dict[str, Any]] = []

    def fake_start(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {"job_id": "job-abcdef12", "status": "queued"}

    monkeypatch.setattr(kap_router, "_start_kap_statement_refresh_job", fake_start)

    response = TestClient(app).post(
        "/admin/kap/refresh-statements",
        headers=ADMIN_HEADERS,
        params={
            "mode": "full",
            "universe": "symbols",
            "symbols": "THYAO, GARAN;AKBNK",
            "max_quarters": 6,
            "budget_seconds": 600,
            "concurrency": 2,
        },
    )

    assert response.status_code == 200
    assert calls == [
        {
            "mode": "full",
            "universe": "symbols",
            "symbols": ["THYAO", "GARAN", "AKBNK"],
            "max_quarters": 6,
            "budget_seconds": 600,
            "concurrency": 2,
        }
    ]
    assert response.json()["refresh_job"]["job_id"] == "job-abcdef12"


def test_statement_refresh_status_returns_404_for_unknown_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kap_router,
        "CONFIG",
        types.SimpleNamespace(kap=types.SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(kap_router, "_get_kap_statement_refresh_job", lambda _job_id: None)

    response = TestClient(app).get(
        "/admin/kap/refresh-statements/status",
        headers=ADMIN_HEADERS,
        params={"job_id": "unknown-job-id"},
    )

    assert response.status_code == 404


def test_job_tracks_current_and_last_symbol_and_detects_interrupted_worker(
    statement_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_current: List[Any] = []

    def fake_snapshot(**kwargs: Any) -> Dict[str, Any]:
        stored = statement_module._get_kap_statement_refresh_job("job-track")
        assert stored is not None
        observed_current.append(stored.get("current_symbol"))
        return {
            "ok": True,
            "schema_version": 15,
            "live_period_count": 5,
            "quarters": [
                {"year": 2026, "period": period, "disclosure_index": period}
                for period in range(1, 6)
            ],
        }

    monkeypatch.setattr("app.kap_service.get_kap_snapshot", fake_snapshot)
    job = _seed_and_run_job(mode="full", symbols=["MGROS"], job_id="job-track")

    assert observed_current == ["MGROS"]
    assert job["current_symbol"] is None
    assert job["last_symbol"] == "MGROS:updated"

    # Simulate a container restart leaving a running job with a stale heartbeat
    stale_job = {
        "job_id": "job-stale",
        "status": "running",
        "mode": "full",
        "universe": "bist100",
        "budget_seconds": 5400,
        "started_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:01:00+00:00",
        "current_symbol": "MGROS",
        "last_symbol": "MAVI:updated",
        "processed": 62,
        "remaining": 38,
        "owner_token": "owner-stale",
    }
    statement_module._set_kap_statement_refresh_job(stale_job)
    recovered = statement_module._get_kap_statement_refresh_job("job-stale")
    assert recovered is not None
    assert recovered["status"] == "partial"
    assert recovered["stop_reason"] == "worker_interrupted"
    assert recovered["current_symbol"] is None
    assert recovered["last_symbol"] == "MAVI:updated"
