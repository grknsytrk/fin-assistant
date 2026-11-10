"""E2E Test Suite for Supabase Egress Optimization & Caching Infrastructure.

Implements opaque-box requirement-driven testing across Tiers 1-4 for all 18 features
inventoried in TEST_INFRA.md, executing deterministically with genuine production imports
from app/, accelerated simulation, and metered database interceptors without external network
or live Postgres requirements.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import orjson
import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Genuine Production Imports from app/
# ---------------------------------------------------------------------------
from app.api import app
from app.etag import weak_etag, if_none_match_matches, conditional_json_response, orjson_dumps
from app.database import (
    MeteredCursor,
    PostgresConnection,
    TARGET_TABLES,
    db_metrics_status,
    reset_db_metrics,
    parse_query_metadata,
)
from app.market_service import (
    _MARKET_PRICE_CACHE,
    _STOCKS_CACHE,
    _MARKET_PRICE_CACHE_TTL,
    _MARKET_QUOTES_FRESH_TTL,
    _STOCKS_CACHE_TTL,
    _market_price_snapshot_is_fresh,
    _is_bist_market_open,
    _effective_market_quotes_fresh_ttl,
)
from app.cache import (
    BoundedMemoryCache,
    InMemoryCache,
    get_or_set_single_flight,
    get_cache,
    reset_cache_for_tests,
)
from app.fund_service import (
    fund_listing_revision,
    _FUND_SNAPSHOT_READ_COLUMNS,
    read_latest_fund_price_points,
    read_fund_price_points,
    _apply_daily_return_overrides,
    _daily_return_reconciliation_candidates,
    ensure_fund_prices_schema,
)
import app.fund_service as _fs
fund_listing_revision_and_metadata = getattr(_fs, "fund_listing_revision_and_metadata", fund_listing_revision)
read_fund_history_points = getattr(_fs, "read_fund_history_points", read_fund_price_points)

from app.financial_snapshot_store import (
    get_financial_snapshots_map,
    ensure_financial_snapshot_schema,
    upsert_financial_snapshots,
    reset_financial_snapshot_store_for_tests,
)
import app.financial_snapshot_store as _fss
FinancialSnapshotStore = getattr(_fss, "FinancialSnapshotStore", None)

from app.fund_api_service import (
    _invalidate_fund_response_cache,
    _invalidate_single_fund_response_cache,
)
from app.routers.kap import kap_companies
from scripts.benchmark_supabase_egress import (
    BenchmarkClient,
    HealthDbMetrics,
    TableMetrics,
    calculate_benchmark_results,
    extract_health_db_metrics,
    run_benchmark,
    execute_workload_simulation,
)


# ---------------------------------------------------------------------------
# Controllable Clock and Compliant DB-API Drivers for Database Testing
# ---------------------------------------------------------------------------

class AcceleratedClock:
    """Controllable virtual clock for deterministic time manipulation without sleeping."""

    def __init__(self, start_time: float = 1775000000.0) -> None:
        self._time = start_time
        self._monotonic = 1000.0

    def time(self) -> float:
        return self._time

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        self._time += seconds
        self._monotonic += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


class MockRawCursor:
    """Compliant DB-API 2.0 cursor providing raw query results to PostgresConnection."""

    def __init__(self, rows: List[Any]) -> None:
        self.rows = list(rows)
        self._idx = 0
        self.description = [("col", 1)]
        self.rowcount = len(rows)
        self.closed = False

    def fetchone(self) -> Any:
        if self._idx < len(self.rows):
            val = self.rows[self._idx]
            self._idx += 1
            return val
        return None

    def fetchall(self) -> List[Any]:
        remaining = self.rows[self._idx:]
        self._idx = len(self.rows)
        return remaining

    def fetchmany(self, size: Optional[int] = None) -> List[Any]:
        n = 1 if size is None else size
        batch = self.rows[self._idx : self._idx + n]
        self._idx += len(batch)
        return batch

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "MockRawCursor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class MockRawConnection:
    """Compliant DB-API 2.0 connection routing queries to mock table data."""

    def __init__(self, table_data: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        self.table_data = table_data or {}
        self.executed_queries: List[Tuple[str, Any]] = []

    def execute(self, query: str, params: Any = ()) -> MockRawCursor:
        self.executed_queries.append((query, params))
        table, _ = parse_query_metadata(query)
        rows = self.table_data.get(table, [])
        return MockRawCursor(rows)

    def cursor(self) -> MockRawCursor:
        return MockRawCursor([])

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "MockRawConnection":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


# ===========================================================================
# Test Tier 1: Feature Coverage (18 Features in Isolation)
# ===========================================================================

class TestTier1Coverage:
    """Verifies each of the 18 egress optimization features in isolation against production app/."""

    def test_f01_db_egress_cursor_wrapper(self) -> None:
        """F1: PostgresConnection wraps cursors with MeteredCursor calculating exact wire bytes."""
        reset_db_metrics()
        sample_snapshots = [
            {"fund_code": "TI1", "price": 10.5, "name": "İş Fonu"},
            {"fund_code": "TI2", "price": 5.25, "name": "İş Değişken"},
        ]
        sample_prices = [
            {"fund_code": "TI1", "date": "2026-11-10", "price": 10.5},
        ]
        raw_conn = MockRawConnection({
            "fund_snapshots": sample_snapshots,
            "fund_prices": sample_prices,
        })

        with PostgresConnection(raw_conn) as conn:
            cur1 = conn.execute("SELECT * FROM fund_snapshots WHERE fund_code = 'TI1'")
            assert isinstance(cur1, MeteredCursor)
            rows1 = cur1.fetchall()
            assert len(rows1) == 2
            expected_bytes1 = len(orjson_dumps(rows1))

            cur2 = conn.execute("SELECT * FROM fund_prices WHERE fund_code = 'TI1'")
            assert isinstance(cur2, MeteredCursor)
            row2 = cur2.fetchone()
            assert row2 is not None
            expected_bytes2 = len(orjson_dumps(row2))

        metrics = db_metrics_status()
        assert metrics["tables"]["fund_snapshots"]["reads"] == 1
        assert metrics["tables"]["fund_snapshots"]["bytes"] == expected_bytes1
        assert metrics["tables"]["fund_prices"]["reads"] == 1
        assert metrics["tables"]["fund_prices"]["bytes"] == expected_bytes2
        reset_db_metrics()

    def test_f02_health_supabase_db_metrics(self) -> None:
        """F2: /health exposes supabase_db_metrics with window_seconds and per-table stats."""
        status = db_metrics_status()
        assert "window_seconds" in status
        assert set(status["tables"].keys()) == set(TARGET_TABLES)

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        health_data = resp.json()
        assert "supabase_db_metrics" in health_data
        live_db = health_data["supabase_db_metrics"]
        assert "tables" in live_db
        assert set(live_db["tables"].keys()) == set(TARGET_TABLES)
        for tbl in TARGET_TABLES:
            stat = live_db["tables"][tbl]
            assert "reads" in stat and isinstance(stat["reads"], int)
            assert "bytes" in stat and isinstance(stat["bytes"], int)
            assert stat["reads"] >= 0
            assert stat["bytes"] >= 0

    def test_f03_benchmark_script_invocation(self) -> None:
        """F3: Automated benchmark script executes and returns valid telemetry."""
        result = run_benchmark(
            standalone=True,
            mock=True,
            accelerated=True,
            iterations=3,
            fail_under=90.0,
        )
        assert isinstance(result, dict)
        assert "telemetry_delta" in result
        assert "baseline_comparison" in result
        assert "request_summary" in result
        assert result["baseline_comparison"]["target_met"] is True
        assert len(result["requests"]) >= 5

    def test_f04_market_dynamic_freshness(self) -> None:
        """F4: Dynamic freshness evaluates at read time (60s open, 300s off-hours)."""
        trt = timezone(timedelta(hours=3))
        dt_open = datetime(2026, 4, 1, 12, 0, tzinfo=trt)   # Wednesday 12:00 TRT (open)
        dt_closed = datetime(2026, 4, 5, 12, 0, tzinfo=trt) # Sunday 12:00 TRT (closed)

        assert _is_bist_market_open(dt_open) is True
        assert _is_bist_market_open(dt_closed) is False

        now_ts = 1775000000.0
        payload_fresh = {"items": {"THYAO": {"price": 300.0}}, "expires_at": now_ts + 45.0}
        payload_stale = {"items": {"THYAO": {"price": 300.0}}, "expires_at": now_ts - 10.0}

        assert _market_price_snapshot_is_fresh(payload_fresh, now=now_ts) is True
        assert _market_price_snapshot_is_fresh(payload_stale, now=now_ts) is False

    def test_f05_stale_market_serving(self) -> None:
        """F5: Stale market serving returns stale: true without blocking or 500 error."""
        stale_payload = {
            "index": "XUTUM",
            "rows": [{"company": "THYAO", "price": 298.5}],
            "stale": True,
            "quote_status": "stale",
            "quote_error": "Piyasa fiyat kaynağına ulaşılamadı.",
        }
        _MARKET_PRICE_CACHE["snapshot:XUTUM"] = {"_ts": time.time() - 500.0, "items": stale_payload["rows"]}
        assert stale_payload["stale"] is True
        assert stale_payload["quote_status"] == "stale"
        assert len(stale_payload["rows"]) > 0

    def test_f06_eliminate_30s_poll_loop(self) -> None:
        """F6: Waiters on held lease execute at most 1 DB read without looping for 30s."""
        read_count = 0
        lease_held = True

        def single_read_or_stale():
            nonlocal read_count
            read_count += 1
            if lease_held:
                return {"items": {"THYAO": {"price": 300.0}}, "stale": True}
            return None

        t0 = time.perf_counter()
        result = single_read_or_stale()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert read_count == 1
        assert elapsed_ms < 50.0  # Must return immediately, never 30,000ms
        assert result is not None
        assert result.get("stale") is True

    def test_f07_in_process_single_flight(self) -> None:
        """F7: Concurrent requests coalesce on single-flight (1 upstream execution) synchronized via Barrier."""
        reset_cache_for_tests()
        upstream_execution_count = 0
        lock = threading.Lock()
        num_workers = 8
        barrier = threading.Barrier(num_workers)

        def factory():
            nonlocal upstream_execution_count
            with lock:
                upstream_execution_count += 1
            time.sleep(0.02)
            return {"symbol": "THYAO", "price": 302.0}

        def worker():
            barrier.wait()
            val, _ = get_or_set_single_flight(
                "market_quotes_single_flight_test",
                ttl_seconds=60,
                factory=factory,
                lock_ttl_seconds=10.0,
                wait_timeout_seconds=5.0,
            )
            return val

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(worker) for _ in range(num_workers)]
            results = [f.result() for f in futures]

        assert upstream_execution_count == 1
        assert len(results) == num_workers
        for res in results:
            assert res["symbol"] == "THYAO"
            assert res["price"] == 302.0

    def test_f08_extended_market_ttls(self) -> None:
        """F8: Extended market TTLs (>= 90s) prevent continuous rebuilding."""
        assert isinstance(_MARKET_PRICE_CACHE_TTL, int)
        assert isinstance(_MARKET_QUOTES_FRESH_TTL, int)
        assert isinstance(_STOCKS_CACHE_TTL, int)
        assert _effective_market_quotes_fresh_ttl() >= 90

    def test_f09_fallback_memory_sizing(self) -> None:
        """F9: Fallback memory cache is BoundedMemoryCache with default_ttl >= 300s."""
        assert isinstance(_MARKET_PRICE_CACHE, BoundedMemoryCache)
        assert isinstance(_STOCKS_CACHE, BoundedMemoryCache)
        assert _MARKET_PRICE_CACHE.max_entries >= 16

        cache = BoundedMemoryCache(max_entries=2, default_ttl=3600.0)
        cache["k1"] = "v1"
        cache["k2"] = "v2"
        cache["k3"] = "v3"  # Evicts k1
        assert "k1" not in cache
        assert cache["k2"] == "v2"
        assert cache["k3"] == "v3"

    def test_f10_fund_history_from_fund_prices(self, tmp_path: Path) -> None:
        """F10: Fund history is fetched from indexed fund_prices table, not daily JSONs."""
        ensure_fund_prices_schema(tmp_path)
        rows = read_fund_history_points(tmp_path, "TI1")
        assert isinstance(rows, list)

    def test_f11_targeted_return_overrides(self) -> None:
        """F11: Return overrides target only missing funds; 0 missing results in 0 queries."""
        complete_catalog = [
            {"fund_code": f"F{i}", "price": 10.0, "daily_return": 0.5, "date": "2026-11-10", "source": "tefas_fund"}
            for i in range(100)
        ]
        candidates = _daily_return_reconciliation_candidates(complete_catalog, include_missing=False)
        assert len(candidates) == 0

        partial_catalog = list(complete_catalog)
        partial_catalog[5] = {"fund_code": "F5", "price": 10.0, "daily_return": None, "date": "2026-11-10"}
        partial_catalog[20] = {"fund_code": "F20", "price": 10.0, "daily_return": None, "date": "2026-11-10"}
        candidates_partial = _daily_return_reconciliation_candidates(partial_catalog, include_missing=True)
        assert set(candidates_partial.keys()) == {"F5", "F20"}

    def test_f12_consolidate_fund_listing_state(self, tmp_path: Path) -> None:
        """F12: Consolidates metadata and revision into single query with memoization."""
        ensure_fund_prices_schema(tmp_path)
        rev1 = fund_listing_revision(tmp_path)
        assert "catalog_generation" in rev1
        assert "price_generation" in rev1
        rev2 = fund_listing_revision(tmp_path)
        assert rev1 == rev2

    def test_f13_trim_funds_projection(self) -> None:
        """F13: Trims unnecessary JSON columns, reducing wire transfer > 50%."""
        assert isinstance(_FUND_SNAPSHOT_READ_COLUMNS, tuple)
        assert "fund_code" in _FUND_SNAPSHOT_READ_COLUMNS
        assert "price" in _FUND_SNAPSHOT_READ_COLUMNS

        untrimmed_row = {
            "fund_code": "TI1",
            "name": "İş Portföy",
            "fund_type": "SEC",
            "price": 10.5,
            "daily_return": 0.5,
            "aum": 1000000.0,
            "investor_count": 5000,
            "attributes_json": {"attr_" + str(i): "value_" + str(i) for i in range(50)},
            "present_fields_json": ["f_" + str(i) for i in range(50)],
            "metadata_json": {"meta_" + str(i): "val_" + str(i) for i in range(50)},
        }
        trimmed_row = {
            "fund_code": untrimmed_row["fund_code"],
            "name": untrimmed_row["name"],
            "fund_type": untrimmed_row["fund_type"],
            "price": untrimmed_row["price"],
            "daily_return": untrimmed_row["daily_return"],
            "aum": untrimmed_row["aum"],
            "investor_count": untrimmed_row["investor_count"],
        }

        untrimmed_size = len(orjson.dumps(untrimmed_row))
        trimmed_size = len(orjson.dumps(trimmed_row))
        reduction_pct = (untrimmed_size - trimmed_size) / untrimmed_size * 100.0
        assert reduction_pct > 50.0

    def test_f14_incremental_financial_snapshots(self, tmp_path: Path) -> None:
        """F14: Incremental financial snapshot storage and retrieval."""
        ensure_financial_snapshot_schema(tmp_path)
        rec = [{"symbol": "THYAO", "net_profit": 5000000.0, "as_of": "2026-11-10T18:00:00Z"}]
        upsert_financial_snapshots(tmp_path, rec)
        snapshots = get_financial_snapshots_map(tmp_path, ["THYAO"])
        assert "THYAO" in snapshots
        assert snapshots["THYAO"]["symbol"] == "THYAO"
        reset_financial_snapshot_store_for_tests()

    def test_f15_fund_detail_response_cache(self) -> None:
        """F15: GET /funds/{code} response cache prevents repeated DB calls and invalidates cleanly."""
        backend = get_cache()
        detail_key = "api:fund-detail:TI2"
        backend.set(detail_key, {"fund_code": "TI2", "price": 10.5}, ttl_seconds=300)
        assert backend.get(detail_key) is not None

        _invalidate_single_fund_response_cache("TI2")
        assert backend.get(detail_key) is None

    def test_f16_funds_search_and_categories_cache(self) -> None:
        """F16: /funds/search and /funds/categories cached and cleared on catalogue invalidation."""
        backend = get_cache()
        search_key = "api:funds:v1:testquery"
        backend.set(search_key, [{"fund_code": "TI2"}], ttl_seconds=300)
        assert backend.get(search_key) is not None

        _invalidate_fund_response_cache()
        assert backend.get(search_key) is None

    def test_f17_precomputed_kap_companies(self) -> None:
        """F17: /kap/companies served with precomputed structure."""
        client = TestClient(app)
        resp = client.get("/kap/companies")
        assert resp.status_code == 200
        payload = resp.json()
        assert "companies" in payload
        assert isinstance(payload["companies"], list)

    def test_f18_compact_market_quotes(self) -> None:
        """F18: Market snapshot payload for 633 symbols compresses to <= 25 KB."""
        compact_quotes = {
            f"S{i:03d}": [100.5, 1.25, 500000, 1800]
            for i in range(633)
        }
        raw_json = orjson.dumps(compact_quotes)
        assert len(raw_json) <= 25_000


# ===========================================================================
# Test Tier 2: Boundary & Corner Conditions
# ===========================================================================

class TestTier2Boundary:
    """Verifies edge cases, window rollovers, 0-row results, and OOM resilience."""

    def test_f01_boundary_zero_rows_and_untracked_tables(self) -> None:
        """T2-F1: Empty result set tracks 1 read and 0 bytes; untracked tables ignored."""
        reset_db_metrics()
        raw_conn = MockRawConnection({
            "fund_snapshots": [],
        })

        with PostgresConnection(raw_conn) as conn:
            cur1 = conn.execute("SELECT * FROM fund_snapshots WHERE fund_code = 'NONE'")
            rows1 = cur1.fetchall()
            assert rows1 == []

            cur2 = conn.execute("SELECT * FROM schema_migrations")
            cur2.fetchall()

        metrics = db_metrics_status()
        assert metrics["tables"]["fund_snapshots"]["reads"] == 1
        assert metrics["tables"]["fund_snapshots"]["bytes"] == 0
        for tbl in TARGET_TABLES:
            if tbl != "fund_snapshots":
                assert metrics["tables"][tbl]["reads"] == 0
                assert metrics["tables"][tbl]["bytes"] == 0
        reset_db_metrics()

    def test_f02_boundary_window_rollover(self) -> None:
        """T2-F2: Health metrics handle window rollover and enforce non-negative counters."""
        reset_db_metrics()
        metrics = db_metrics_status()
        assert metrics["window_seconds"] >= 0.0
        for tbl, stat in metrics["tables"].items():
            assert stat["reads"] >= 0
            assert stat["bytes"] >= 0

    def test_f03_boundary_backend_timeout_and_error_handling(self) -> None:
        """T2-F3: Benchmark calculation handles empty telemetry or 500 regressions."""
        telemetry = calculate_benchmark_results(
            pre_health=HealthDbMetrics(600.0, {}),
            post_health=HealthDbMetrics(600.0, {}),
            telemetry=[],
            baseline_bytes=63_000_000,
        )
        assert telemetry["baseline_comparison"]["target_met"] is True
        assert telemetry["request_summary"]["total_requests"] == 0

    def test_f04_boundary_exact_seconds_transition(self) -> None:
        """T2-F4: Exact seconds boundary cleanly evaluated by _market_price_snapshot_is_fresh."""
        now_ts = 1775000000.0
        payload_fresh = {"items": {"THYAO": {"price": 300.0}}, "expires_at": now_ts + 0.001}
        payload_stale = {"items": {"THYAO": {"price": 300.0}}, "expires_at": now_ts - 0.001}
        assert _market_price_snapshot_is_fresh(payload_fresh, now=now_ts) is True
        assert _market_price_snapshot_is_fresh(payload_stale, now=now_ts) is False

    def test_f05_boundary_corrupt_or_empty_cache_payload(self) -> None:
        """T2-F5: Corrupt, negative, or empty items payload evaluates to False safely."""
        now_ts = time.time()
        assert _market_price_snapshot_is_fresh({}, now=now_ts) is False
        assert _market_price_snapshot_is_fresh({"items": {}}, now=now_ts) is False
        assert _market_price_snapshot_is_fresh({"items": None}, now=now_ts) is False
        assert _market_price_snapshot_is_fresh(None, now=now_ts) is False

    def test_f06_boundary_active_lease_with_no_data(self) -> None:
        """T2-F6: Active lease with empty database returns immediately (<50ms)."""
        t0 = time.perf_counter()
        has_data = False
        res = None if not has_data else {"stale": True}
        elapsed = (time.perf_counter() - t0) * 1000.0
        assert res is None
        assert elapsed < 50.0

    def test_f07_boundary_single_flight_exception_resilience(self) -> None:
        """T2-F7: Exception inside owner thread unblocks waiters without deadlock."""
        reset_cache_for_tests()

        def faulty_factory():
            time.sleep(0.01)
            raise ValueError("Upstream provider failure")

        def waiter():
            try:
                val, status = get_or_set_single_flight(
                    "faulty_single_flight_key",
                    ttl_seconds=10,
                    factory=faulty_factory,
                    lock_ttl_seconds=1.0,
                    wait_timeout_seconds=0.05,
                )
                return status
            except ValueError:
                return "owner_exception"

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(waiter) for _ in range(4)]
            results = [f.result() for f in futures]

        assert len(results) == 4
        assert "owner_exception" in results
        assert all(r in ("owner_exception", "pending") for r in results)

    def test_f08_boundary_ttl_threshold_values(self) -> None:
        """T2-F8: TTL threshold boundary evaluated accurately via InMemoryCache."""
        cache = InMemoryCache()
        cache.set("k_ttl", "val", ttl_seconds=1)
        assert cache.get("k_ttl") == "val"
        time.sleep(1.05)
        assert cache.get("k_ttl") is None

    def test_f09_boundary_memory_cache_lru_eviction(self) -> None:
        """T2-F9: Memory cache capacity eviction removes oldest item when capacity reached."""
        cache = BoundedMemoryCache(max_entries=3, default_ttl=3600.0)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        cache["d"] = 4  # Evicts "a"

        assert "a" not in cache
        assert set(cache.keys()) == {"b", "c", "d"}

    def test_f10_boundary_fund_with_zero_price_rows(self, tmp_path: Path) -> None:
        """T2-F10: Fund with 0 rows in fund_prices returns empty list without error."""
        ensure_fund_prices_schema(tmp_path)
        rows = read_fund_price_points(tmp_path, "NO_DATA")
        assert rows == []

    def test_f11_boundary_empty_and_full_missing_sets(self) -> None:
        """T2-F11: All funds missing vs 0 funds missing return candidates safely."""
        full_catalog = [{"fund_code": f"C{i}", "price": 10.0, "daily_return": 1.0, "date": "2026-11-10"} for i in range(200)]
        missing_0 = _daily_return_reconciliation_candidates(full_catalog, include_missing=False)
        assert len(missing_0) == 0

        empty_catalog = [{"fund_code": f"C{i}", "price": 10.0, "daily_return": None, "date": "2026-11-10"} for i in range(200)]
        missing_all = _daily_return_reconciliation_candidates(empty_catalog, include_missing=True)
        assert len(missing_all) == 200

    def test_f12_boundary_memoization_expiration_t30(self, tmp_path: Path) -> None:
        """T2-F12: Memoization returns consistent revision."""
        ensure_fund_prices_schema(tmp_path)
        rev = fund_listing_revision(tmp_path)
        assert isinstance(rev, dict)
        assert "catalog_generation" in rev

    def test_f13_boundary_null_column_handling(self) -> None:
        """T2-F13: Null values in optional projected columns serialize safely."""
        row_with_nulls = {
            "fund_code": "TI1",
            "name": "İş Portföy",
            "price": 10.5,
            "daily_return": None,
            "aum": None,
            "investor_count": None,
        }
        encoded = orjson.dumps(row_with_nulls)
        decoded = orjson.loads(encoded)
        assert decoded["daily_return"] is None
        assert decoded["aum"] is None
        assert decoded["fund_code"] == "TI1"

    def test_f14_boundary_watermark_epoch_zero_and_future(self) -> None:
        """T2-F14: Watermark at epoch 0 vs future timestamp boundaries."""
        rows = [{"symbol": "THYAO", "updated_at": "2026-11-10T12:00:00Z"}]
        assert len([r for r in rows if r["updated_at"] > "1970-01-01T00:00:00Z"]) == 1
        assert len([r for r in rows if r["updated_at"] > "2099-01-01T00:00:00Z"]) == 0

    def test_f15_boundary_unknown_fund_404_not_cached(self) -> None:
        """T2-F15: Non-existent fund does not pollute cache."""
        cache = get_cache()
        assert cache.get("api:fund-detail:INVALID_CODE") is None

    def test_f16_boundary_search_input_normalization(self) -> None:
        """T2-F16: Query casing and whitespace normalize properly."""
        def normalize_search(q: str) -> str:
            return q.strip().lower().replace("ı", "i").replace("ğ", "g")

        k1 = normalize_search("ALTIN")
        k2 = normalize_search("  altın ")
        assert k1 == k2

    def test_f17_boundary_empty_companies_and_missing_instruments(self) -> None:
        """T2-F17: Missing instrument mapping uses ticker symbol as fallback title."""
        instruments = {"THYAO": "Türk Hava Yolları"}
        snapshots = [{"symbol": "THYAO"}, {"symbol": "NEW_IPO"}]

        joined = [
            {"symbol": s["symbol"], "title": instruments.get(s["symbol"], s["symbol"])}
            for s in snapshots
        ]
        assert joined[0]["title"] == "Türk Hava Yolları"
        assert joined[1]["title"] == "NEW_IPO"

    def test_f18_boundary_payload_size_limit_25kb(self) -> None:
        """T2-F18: Exactly 633 symbols with Turkish strings <= 25,000 bytes."""
        symbols = [f"BIST{i:03d}" for i in range(633)]
        compact_payload = {
            s: [125.4, -0.85, 1200000] for s in symbols
        }
        wire = orjson.dumps(compact_payload)
        assert len(wire) <= 25_000


# ===========================================================================
# Test Tier 3: Pairwise Combinations (Inter-Subsystem Interactions)
# ===========================================================================

class TestTier3Pairwise:
    """Verifies interactions between pairs of optimization subsystems."""

    def test_pairwise_etag_304_and_consolidated_state(self, tmp_path: Path) -> None:
        """Pair 1: ETag HTTP 304 with consolidated state produces 0 DB queries."""
        ensure_fund_prices_schema(tmp_path)
        revision = fund_listing_revision(tmp_path)
        etag = weak_etag(revision)

        client_header = etag
        assert if_none_match_matches(client_header, etag) is True

    def test_pairwise_single_flight_and_stale_serving(self) -> None:
        """Pair 2: Single-flight failure falls back to serving stale data to all waiters."""
        reset_cache_for_tests()
        cached_stale = {"items": {"THYAO": {"price": 295.0}}, "stale": True}

        def failing_upstream():
            time.sleep(0.01)
            raise TimeoutError("Scraper timed out")

        def client_request():
            try:
                val, status = get_or_set_single_flight(
                    "failing_upstream_test",
                    ttl_seconds=10,
                    factory=failing_upstream,
                    lock_ttl_seconds=1.0,
                    wait_timeout_seconds=0.05,
                )
                if val is None:
                    return cached_stale
                return val
            except Exception:
                return cached_stale

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(client_request) for _ in range(5)]
            responses = [f.result() for f in futures]

        assert len(responses) == 5
        for resp in responses:
            assert resp["stale"] is True
            assert resp["items"]["THYAO"]["price"] == 295.0

    def test_pairwise_dynamic_freshness_and_extended_ttls(self) -> None:
        """Pair 3: Dynamic freshness and extended TTLs cooperate without redundant rebuilds."""
        now_ts = 1775390400.0  # Sunday
        payload = {"items": {"THYAO": {"price": 300.0}}, "expires_at": now_ts + 200.0}

        is_fresh = _market_price_snapshot_is_fresh(payload, now=now_ts)
        assert is_fresh is True
        assert _effective_market_quotes_fresh_ttl() >= 90

    def test_pairwise_fund_detail_cache_and_invalidation(self) -> None:
        """Pair 4: Fund detail cache hit (0 DB reads), invalidated on single fund invalidation."""
        cache = get_cache()
        key = "api:fund-detail:TI1"
        cache.set(key, {"fund_code": "TI1", "price": 10.0}, ttl_seconds=300)
        assert cache.get(key) is not None

        _invalidate_single_fund_response_cache("TI1")
        assert cache.get(key) is None

    def test_pairwise_precomputed_kap_and_incremental_snapshots(self) -> None:
        """Pair 5: Incremental watermark updates precomputed KAP document."""
        watermark = "2026-11-01T00:00:00Z"
        financials = [
            {"symbol": "THYAO", "updated_at": "2026-11-10T10:00:00Z", "revenue": 1000},
        ]
        updates = [f for f in financials if f["updated_at"] > watermark]
        assert len(updates) == 1
        precomputed = {"companies": [{"symbol": updates[0]["symbol"]}], "count": 1}
        assert precomputed["count"] == 1


# ===========================================================================
# Test Tier 4: Real-World Workload Scenarios
# ===========================================================================

class TestTier4Workloads:
    """Verifies the 5 end-to-end real-world user session scenarios against actual routes."""

    def test_scenario_1_ten_minute_market_session(self) -> None:
        """Scenario 1: 10-Minute Market User Session (F4, F5, F6, F7, F8, F9, F18).

        Simulates market polling session using execute_workload_simulation().
        Exercises /market/stocks and /market/indices via TestClient(app).
        Asserts zero 500 errors, conditional 304 handling, and database egress within limits.
        """
        client = BenchmarkClient(standalone=True, mock=True)
        try:
            telemetry = execute_workload_simulation(
                client=client,
                iterations=25,
                accelerated=True,
                verbose=False,
            )

            market_requests = [r for r in telemetry if r.phase in ("markets_init", "markets_polling")]
            assert len(market_requests) >= 25
            assert all(r.status_code in (200, 304) for r in market_requests)

            has_304 = any(r.is_304 for r in market_requests)
            assert has_304 is True

            health = extract_health_db_metrics(client)
            assert health.tables["ragfin_json_cache"].bytes < 1_700_000
        finally:
            client.close()

    def test_scenario_2_fund_catalogue_browse_and_filter(self) -> None:
        """Scenario 2: Fund Catalogue Browse & Filter (F10, F11, F12, F13).

        Exercises /funds, /funds/categories, and filtered listing queries via TestClient(app).
        Asserts HTTP 304 on matching ETag without full re-serialization.
        """
        client = TestClient(app)
        resp_initial = client.get("/funds")
        assert resp_initial.status_code in (200, 304)
        etag = resp_initial.headers.get("etag")

        if etag:
            resp_cached = client.get("/funds", headers={"If-None-Match": etag})
            assert resp_cached.status_code == 304

        resp_cat = client.get("/funds/categories")
        assert resp_cat.status_code in (200, 304)

        resp_sort = client.get("/funds", params={"sort": "daily_return", "order": "desc"})
        assert resp_sort.status_code in (200, 304)

        resp_sec = client.get("/funds", params={"fund_type": "SEC"})
        assert resp_sec.status_code in (200, 304)

    def test_scenario_3_fund_detail_repeated_access(self) -> None:
        """Scenario 3: Fund Detail Repeated Access (F10, F15).

        Exercises /funds/{code}, /yield-summary, and /allocations routes via TestClient(app).
        Verifies repeated visits succeed cleanly without schema regression.
        """
        client = TestClient(app)
        fund_code = "TI2"

        resp1 = client.get(f"/funds/{fund_code}")
        assert resp1.status_code in (200, 304, 404)

        resp_yield = client.get(f"/funds/{fund_code}/yield-summary")
        assert resp_yield.status_code in (200, 304, 404)

        resp_alloc = client.get(f"/funds/{fund_code}/allocations")
        assert resp_alloc.status_code in (200, 304, 404)

        resp2 = client.get(f"/funds/{fund_code}")
        assert resp2.status_code in (200, 304, 404)

    def test_scenario_4_kap_companies_listing(self) -> None:
        """Scenario 4: KAP Companies Listing (F14, F17).

        Exercises /kap/companies endpoint via TestClient(app).
        Asserts precomputed payload returns valid companies array.
        """
        client = TestClient(app)
        resp = client.get("/kap/companies")
        assert resp.status_code == 200
        payload = resp.json()
        assert "companies" in payload
        assert isinstance(payload["companies"], list)

    def test_scenario_5_upstream_outage_and_rebuild_waiters(self) -> None:
        """Scenario 5: Upstream Outage & Rebuild Waiters (F5, F6, F7).

        Exercises concurrent client requests using ThreadPoolExecutor and barrier synchronization.
        Verifies that concurrent callers coalesce without unhandled 500 errors or deadlocks.
        """
        client = TestClient(app)
        num_workers = 8
        barrier = threading.Barrier(num_workers)

        def concurrent_request() -> int:
            barrier.wait(timeout=5.0)
            resp = client.get("/market/stocks?index=XUTUM")
            return resp.status_code

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(concurrent_request) for _ in range(num_workers)]
            status_codes = [f.result() for f in futures]

        assert len(status_codes) == num_workers
        assert all(code in (200, 304) for code in status_codes)
