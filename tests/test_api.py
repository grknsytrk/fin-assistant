from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import database as database_module
from app.api import app
from api_test_helpers import _reset_flow_state


def test_api_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["cache_backend"] in {"memory", "redis"}
    assert payload["cache_namespace"]
    assert "cache_redis_fallback" in payload
    assert "cache_last_error" in payload
    if payload["cache_backend"] == "redis":
        assert "cache_redis_circuit" in payload
    assert "supabase_json_cache_metrics" in payload
    assert "totals" in payload["supabase_json_cache_metrics"]
    assert "supabase_db_metrics" in payload
    db_metrics = payload["supabase_db_metrics"]
    assert "window_seconds" in db_metrics
    assert isinstance(db_metrics["window_seconds"], (int, float))
    assert "tables" in db_metrics
    expected_tables = {
        "ragfin_json_cache",
        "ragfin_kap_flow_events",
        "fund_snapshots",
        "fund_prices",
        "company_financial_snapshots",
        "instruments",
    }
    assert set(db_metrics["tables"].keys()) == expected_tables
    for table_name in expected_tables:
        assert isinstance(db_metrics["tables"][table_name]["reads"], int)
        assert isinstance(db_metrics["tables"][table_name]["bytes"], int)


def test_api_health_supabase_db_metrics_contract() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "supabase_db_metrics" in payload
    db_metrics = payload["supabase_db_metrics"]
    assert isinstance(db_metrics["window_seconds"], (int, float))
    assert db_metrics["window_seconds"] >= 0.0

    tables = db_metrics["tables"]
    expected_tables = (
        "ragfin_json_cache",
        "ragfin_kap_flow_events",
        "fund_snapshots",
        "fund_prices",
        "company_financial_snapshots",
        "instruments",
    )
    for table in expected_tables:
        assert table in tables
        assert tables[table]["reads"] >= 0
        assert tables[table]["bytes"] >= 0


def test_api_health_db_metrics_increments_on_target_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import database as database_module
    from app.etag import orjson_dumps

    database_module.reset_db_metrics()

    mock_payload = {"test_key": "test_value_12345"}
    mock_bytes = len(orjson_dumps([mock_payload]))

    database_module._record_db_metric("ragfin_json_cache", "SELECT", mock_bytes)

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    metrics = response.json()["supabase_db_metrics"]["tables"]

    assert metrics["ragfin_json_cache"]["reads"] == 1
    assert metrics["ragfin_json_cache"]["bytes"] == mock_bytes

    for other in ("ragfin_kap_flow_events", "fund_snapshots", "fund_prices", "company_financial_snapshots", "instruments"):
        assert metrics[other]["reads"] == 0
        assert metrics[other]["bytes"] == 0


def test_api_health_db_metrics_ignores_non_targeted_tables() -> None:
    from app import database as database_module

    database_module.reset_db_metrics()

    database_module._record_db_metric("refresh_leases", "SELECT", 500)
    database_module._record_db_metric("kap_flow_events", "SELECT", 2000)

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    tables = response.json()["supabase_db_metrics"]["tables"]

    for target in (
        "ragfin_json_cache",
        "ragfin_kap_flow_events",
        "fund_snapshots",
        "fund_prices",
        "company_financial_snapshots",
        "instruments",
    ):
        assert tables[target]["reads"] == 0
        assert tables[target]["bytes"] == 0


def test_api_health_db_metrics_reset() -> None:
    from app import database as database_module

    database_module._record_db_metric("fund_snapshots", "SELECT", 1024)
    assert database_module.db_metrics_status()["tables"]["fund_snapshots"]["reads"] >= 1

    database_module.reset_db_metrics()
    status = database_module.db_metrics_status()
    for table in status["tables"].values():
        assert table["reads"] == 0
        assert table["bytes"] == 0


def test_postgres_connection_metered_cursor_tracking() -> None:
    from app import database as database_module
    from app.etag import orjson_dumps

    database_module.reset_db_metrics()

    class FakeCursor:
        def __init__(self, rows: list[Any]) -> None:
            self.rows = list(rows)
            self._idx = 0
            self.description = [("col", 1)]
            self.rowcount = len(rows)

        def fetchone(self) -> Any:
            if self._idx < len(self.rows):
                val = self.rows[self._idx]
                self._idx += 1
                return val
            return None

        def fetchall(self) -> list[Any]:
            remaining = self.rows[self._idx :]
            self._idx = len(self.rows)
            return remaining

        def fetchmany(self, size: int | None = None) -> list[Any]:
            n = 1 if size is None else size
            batch = self.rows[self._idx : self._idx + n]
            self._idx += len(batch)
            return batch

        def __iter__(self) -> Any:
            return self

        def __next__(self) -> Any:
            val = self.fetchone()
            if val is None:
                raise StopIteration
            return val

    class FakeConn:
        def __init__(self) -> None:
            self.executed_queries: list[str] = []

        def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any:
            self.executed_queries.append(query)
            if "fund_prices" in query:
                return FakeCursor([{"fund_code": "NNF", "price": 10.5}, {"fund_code": "TI2", "price": 20.0}])
            return FakeCursor([])

    fake_conn = FakeConn()
    pg_conn = database_module.PostgresConnection(fake_conn)
    cur = pg_conn.execute("SELECT * FROM fund_prices WHERE date = ?", ("2026-01-01",))
    rows = cur.fetchall()
    assert len(rows) == 2

    status = database_module.db_metrics_status()
    fp_metrics = status["tables"]["fund_prices"]
    assert fp_metrics["reads"] == 1
    expected_bytes = len(orjson_dumps(rows))
    assert fp_metrics["bytes"] == expected_bytes

    database_module.reset_db_metrics()
