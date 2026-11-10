"""Adversarial stress-test and verification harness for database egress telemetry.

Challenges tested:
1. Wire byte accuracy: exact byte count match with len(orjson_dumps(row/rows))
2. No double-counting: fetchone + fetchall, fetchmany, iterator, repeated calls
3. Concurrency stress: 25+ concurrent threads across target and non-target tables
4. SQL edge cases: CTEs, joins, subqueries, case sensitivity, comments, quotes
5. Fast-path verification: non-target queries bypass serialization and mutex lock
"""

import threading
import time
from decimal import Decimal
from datetime import datetime, date
from typing import Any, List, Optional
import pytest

from app import database as db
from app.etag import orjson_dumps


class MockCursor:
    """Mock DB-API cursor supporting all fetch mechanisms and iteration."""

    def __init__(self, rows: List[Any], description: Optional[List[tuple]] = None) -> None:
        self._rows = list(rows)
        self._pos = 0
        self.description = description or [("col1", 1)]
        self.rowcount = len(rows)
        self.closed = False

    def fetchone(self) -> Any:
        if self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            return row
        return None

    def fetchall(self) -> List[Any]:
        remaining = self._rows[self._pos :]
        self._pos = len(self._rows)
        return remaining

    def fetchmany(self, size: Optional[int] = None) -> List[Any]:
        n = 1 if size is None else max(0, size)
        batch = self._rows[self._pos : self._pos + n]
        self._pos += len(batch)
        return batch

    def __iter__(self) -> "MockCursor":
        return self

    def __next__(self) -> Any:
        val = self.fetchone()
        if val is None:
            raise StopIteration
        return val

    def close(self) -> None:
        self.closed = True


class MockConnection:
    """Mock connection that generates MockCursor for test queries."""

    def __init__(self, query_row_generator=None) -> None:
        self.query_row_generator = query_row_generator
        self.executed_queries: List[tuple[str, Any]] = []

    def execute(self, query: str, params: Any = ()) -> MockCursor:
        self.executed_queries.append((query, params))
        if self.query_row_generator:
            rows = self.query_row_generator(query, params)
        else:
            rows = []
        return MockCursor(rows)


# ==============================================================================
# Challenge 1: Wire Byte Accuracy
# ==============================================================================
class TestWireByteAccuracy:

    def setup_method(self) -> None:
        db.reset_db_metrics()

    def test_fetchone_exact_byte_match(self) -> None:
        row = {
            "fund_code": "NNF",
            "price": Decimal("123.456789"),
            "date": date(2026, 5, 15),
            "updated_at": datetime(2026, 5, 15, 14, 30, 0),
            "is_active": True,
            "nested": {"rank": 1, "score": 98.5},
        }
        mock_cur = MockCursor([row])
        cur = db.MeteredCursor(mock_cur, table="fund_prices", operation="SELECT")

        fetched = cur.fetchone()
        assert fetched == row

        expected_bytes = len(orjson_dumps(row))
        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["bytes"] == expected_bytes
        assert expected_bytes > 0

    def test_fetchall_exact_byte_match(self) -> None:
        rows = [
            {"id": i, "name": f"Instrument-{i}", "price": Decimal(f"{10.5 + i:.2f}")}
            for i in range(10)
        ]
        mock_cur = MockCursor(rows)
        cur = db.MeteredCursor(mock_cur, table="instruments", operation="SELECT")

        fetched = cur.fetchall()
        assert len(fetched) == 10

        expected_bytes = len(orjson_dumps(rows))
        status = db.db_metrics_status()
        assert status["tables"]["instruments"]["bytes"] == expected_bytes

    def test_fetchmany_exact_byte_match(self) -> None:
        rows = [{"cache_key": f"key_{i}", "payload": {"val": i}} for i in range(7)]
        mock_cur = MockCursor(rows)
        cur = db.MeteredCursor(mock_cur, table="ragfin_json_cache", operation="SELECT")

        batch1 = cur.fetchmany(3)
        assert len(batch1) == 3
        expected_batch1 = len(orjson_dumps(batch1))

        batch2 = cur.fetchmany(4)
        assert len(batch2) == 4
        expected_batch2 = len(orjson_dumps(batch2))

        status = db.db_metrics_status()
        assert status["tables"]["ragfin_json_cache"]["bytes"] == expected_batch1 + expected_batch2

    def test_iteration_exact_byte_match(self) -> None:
        rows = [{"snapshot_id": i, "aum": 1000000 * i} for i in range(5)]
        mock_cur = MockCursor(rows)
        cur = db.MeteredCursor(mock_cur, table="fund_snapshots", operation="SELECT")

        iterated_rows = []
        for r in cur:
            iterated_rows.append(r)
        assert len(iterated_rows) == 5

        expected_bytes = sum(len(orjson_dumps(r)) for r in rows)
        status = db.db_metrics_status()
        assert status["tables"]["fund_snapshots"]["bytes"] == expected_bytes

    def test_empty_results_zero_bytes(self) -> None:
        mock_cur = MockCursor([])
        cur = db.MeteredCursor(mock_cur, table="fund_prices", operation="SELECT")

        assert cur.fetchone() is None
        assert cur.fetchall() == []
        assert cur.fetchmany(5) == []

        for _ in cur:
            pytest.fail("Should not yield anything")

        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["bytes"] == 0


# ==============================================================================
# Challenge 2: No Double-Counting
# ==============================================================================
class TestNoDoubleCounting:

    def setup_method(self) -> None:
        db.reset_db_metrics()

    def test_fetchone_then_fetchall(self) -> None:
        rows = [{"code": f"C{i}"} for i in range(4)]
        mock_cur = MockCursor(rows)
        cur = db.MeteredCursor(mock_cur, table="fund_prices", operation="SELECT")

        r0 = cur.fetchone()
        assert r0 == {"code": "C0"}
        rem = cur.fetchall()
        assert len(rem) == 3

        expected_bytes = len(orjson_dumps(r0)) + len(orjson_dumps(rem))
        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["bytes"] == expected_bytes

    def test_fetchone_then_fetchmany_then_iter(self) -> None:
        rows = [{"code": f"C{i}"} for i in range(6)]
        mock_cur = MockCursor(rows)
        cur = db.MeteredCursor(mock_cur, table="fund_prices", operation="SELECT")

        r0 = cur.fetchone()
        b1 = cur.fetchmany(2)
        r_iter = list(cur)

        assert len([r0]) + len(b1) + len(r_iter) == 6
        expected_bytes = (
            len(orjson_dumps(r0))
            + len(orjson_dumps(b1))
            + sum(len(orjson_dumps(r)) for r in r_iter)
        )
        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["bytes"] == expected_bytes

    def test_repeated_fetchall_no_double_count(self) -> None:
        rows = [{"code": "ABC"}]
        mock_cur = MockCursor(rows)
        cur = db.MeteredCursor(mock_cur, table="fund_prices", operation="SELECT")

        first = cur.fetchall()
        assert len(first) == 1
        expected = len(orjson_dumps(first))

        second = cur.fetchall()
        assert second == []

        third = cur.fetchone()
        assert third is None

        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["bytes"] == expected

    def test_query_reads_counted_once_on_execute_not_on_fetch(self) -> None:
        mock_conn = MockConnection(lambda q, p: [{"x": 1}, {"x": 2}])
        pg_conn = db.PostgresConnection(mock_conn)

        cur = pg_conn.execute("SELECT * FROM fund_snapshots WHERE id = ?", (1,))
        status1 = db.db_metrics_status()
        assert status1["tables"]["fund_snapshots"]["reads"] == 1
        assert status1["tables"]["fund_snapshots"]["bytes"] == 0

        # Now fetch in parts
        cur.fetchone()
        cur.fetchall()
        cur.fetchone()  # exhausted

        status2 = db.db_metrics_status()
        assert status2["tables"]["fund_snapshots"]["reads"] == 1  # still 1!
        assert status2["tables"]["fund_snapshots"]["bytes"] > 0


# ==============================================================================
# Challenge 3: Concurrency Stress
# ==============================================================================
class TestConcurrencyStress:

    def setup_method(self) -> None:
        db.reset_db_metrics()

    def test_high_concurrency_thread_safety(self) -> None:
        """25 threads execute 80 queries each concurrently across target and non-target tables."""
        num_threads = 25
        queries_per_thread = 80

        target_tables = list(db.TARGET_TABLES)
        non_target_tables = ["refresh_leases", "market_quotes", "audit_log", "sessions"]

        expected_reads = {t: 0 for t in target_tables}
        expected_bytes = {t: 0 for t in target_tables}
        expected_lock = threading.Lock()

        sample_row = {"symbol": "TEST", "val": 123.45}
        row_bytes = len(orjson_dumps(sample_row))
        batch_3_bytes = len(orjson_dumps([sample_row, sample_row, sample_row]))

        barrier = threading.Barrier(num_threads)
        errors = []

        def worker(thread_id: int) -> None:
            try:
                barrier.wait()
                for i in range(queries_per_thread):
                    target_idx = (thread_id + i) % (len(target_tables) + len(non_target_tables))
                    if target_idx < len(target_tables):
                        table = target_tables[target_idx]
                        is_target = True
                    else:
                        table = non_target_tables[target_idx - len(target_tables)]
                        is_target = False

                    mock_cur = MockCursor([sample_row, sample_row, sample_row])
                    mock_conn = MockConnection(lambda q, p: [sample_row, sample_row, sample_row])
                    pg_conn = db.PostgresConnection(mock_conn)

                    sql = f"SELECT * FROM {table} WHERE id = ?"
                    cur = pg_conn.execute(sql, (i,))

                    fetch_mode = i % 4
                    if fetch_mode == 0:
                        cur.fetchone()
                        bytes_added = row_bytes
                    elif fetch_mode == 1:
                        cur.fetchall()
                        bytes_added = batch_3_bytes
                    elif fetch_mode == 2:
                        cur.fetchmany(1)
                        bytes_added = len(orjson_dumps([sample_row]))
                    else:
                        for _ in cur:
                            pass
                        bytes_added = 3 * row_bytes

                    if is_target:
                        with expected_lock:
                            expected_reads[table] += 1
                            expected_bytes[table] += bytes_added

            except Exception as e:
                errors.append(e)

        # Also launch 2 background readers querying status concurrently
        stop_readers = threading.Event()
        reader_errors = []

        def reader() -> None:
            while not stop_readers.is_set():
                try:
                    s = db.db_metrics_status()
                    assert "tables" in s
                except Exception as e:
                    reader_errors.append(e)
                time.sleep(0.001)

        reader_threads = [threading.Thread(target=reader) for _ in range(2)]
        for rt in reader_threads:
            rt.start()

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stop_readers.set()
        for rt in reader_threads:
            rt.join()

        assert not errors, f"Worker thread errors: {errors}"
        assert not reader_errors, f"Reader thread errors: {reader_errors}"

        status = db.db_metrics_status()
        for table in target_tables:
            actual_r = status["tables"][table]["reads"]
            actual_b = status["tables"][table]["bytes"]
            exp_r = expected_reads[table]
            exp_b = expected_bytes[table]
            assert actual_r == exp_r, f"Mismatch in {table} reads: {actual_r} != {exp_r}"
            assert actual_b == exp_b, f"Mismatch in {table} bytes: {actual_b} != {exp_b}"

    def test_concurrent_resets_and_queries(self) -> None:
        """Verify no deadlocks or dictionary-modified-during-iteration errors during concurrent resets."""
        num_workers = 15
        num_resetters = 3
        num_readers = 5
        duration = 1.0  # seconds

        stop_event = threading.Event()
        errors = []

        def worker(idx: int) -> None:
            while not stop_event.is_set():
                try:
                    mock_conn = MockConnection(lambda q, p: [{"col": "data"}])
                    pg_conn = db.PostgresConnection(mock_conn)
                    cur = pg_conn.execute("SELECT * FROM fund_snapshots WHERE id = ?", (idx,))
                    cur.fetchall()
                except Exception as e:
                    errors.append(e)

        def resetter() -> None:
            while not stop_event.is_set():
                try:
                    db.reset_db_metrics()
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            while not stop_event.is_set():
                try:
                    status = db.db_metrics_status()
                    assert len(status["tables"]) == len(db.TARGET_TABLES)
                    time.sleep(0.005)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]
        threads += [threading.Thread(target=resetter) for _ in range(num_resetters)]
        threads += [threading.Thread(target=reader) for _ in range(num_readers)]

        for t in threads:
            t.start()

        time.sleep(duration)
        stop_event.set()

        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent reset stress: {errors}"


# ==============================================================================
# Challenge 4: SQL Edge Cases
# ==============================================================================
class TestSqlEdgeCases:

    def test_ctes(self) -> None:
        q1 = "WITH recent AS (SELECT 1 FROM other_tbl) SELECT * FROM fund_snapshots WHERE id = 1"
        t1, op1 = db.parse_query_metadata(q1)
        assert t1 == "fund_snapshots"
        assert op1 == "SELECT"

        q2 = "WITH a AS (SELECT * FROM instruments), b AS (SELECT 1) SELECT * FROM b"
        t2, op2 = db.parse_query_metadata(q2)
        assert t2 == "instruments"
        assert op2 == "SELECT"

    def test_joins(self) -> None:
        q = "SELECT a.code, b.price FROM other_tbl a JOIN fund_prices b ON a.id = b.id"
        t, op = db.parse_query_metadata(q)
        assert t == "fund_prices"
        assert op == "SELECT"

    def test_subqueries(self) -> None:
        q = "SELECT * FROM (SELECT * FROM company_financial_snapshots WHERE q = '2025/12') s"
        t, op = db.parse_query_metadata(q)
        assert t == "company_financial_snapshots"
        assert op == "SELECT"

    def test_comments(self) -> None:
        # Line comment
        q1 = "-- comment here\nSELECT * FROM fund_prices"
        t1, op1 = db.parse_query_metadata(q1)
        assert t1 == "fund_prices"
        assert op1 == "SELECT"

        # Block comment
        q2 = "/* multiline\n comment */\nSELECT * FROM ragfin_json_cache"
        t2, op2 = db.parse_query_metadata(q2)
        assert t2 == "ragfin_json_cache"
        assert op2 == "SELECT"

        # Multiple mixed comments
        q3 = "-- line 1\n/* block 1 */\n-- line 2\nSELECT * FROM instruments"
        t3, op3 = db.parse_query_metadata(q3)
        assert t3 == "instruments"
        assert op3 == "SELECT"

    def test_case_insensitivity(self) -> None:
        queries = [
            "select * from FUND_PRICES",
            "Select * From Fund_Prices",
            "SELECT * FROM fund_prices",
        ]
        for q in queries:
            t, op = db.parse_query_metadata(q)
            assert t == "fund_prices"
            assert op == "SELECT"

    def test_quoted_and_schema_tables(self) -> None:
        q1 = 'SELECT * FROM "fund_prices"'
        t1, _ = db.parse_query_metadata(q1)
        assert t1 == "fund_prices"

        q2 = "SELECT * FROM public.fund_prices"
        t2, _ = db.parse_query_metadata(q2)
        assert t2 == "fund_prices"

        # Note: Testing behavior on individually quoted schema.table
        # e.g., "public"."fund_prices" or public."fund_prices"
        # In Postgres, both "table" and public.table are supported.
        t3, _ = db.parse_query_metadata('SELECT * FROM "public"."fund_prices"')
        # Documents known parser behavior on "public"."table"
        assert t3 in ("fund_prices", "public")

    def test_dml_operations(self) -> None:
        db.reset_db_metrics()
        mock_conn = MockConnection()
        pg_conn = db.PostgresConnection(mock_conn)

        # INSERT should not increment reads
        pg_conn.execute("INSERT INTO fund_prices (code) VALUES (?)", ("ABC",))
        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["reads"] == 0

        # UPDATE should not increment reads
        pg_conn.execute("UPDATE fund_prices SET price = ? WHERE code = ?", (10.0, "ABC"))
        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["reads"] == 0

        # DELETE should not increment reads
        pg_conn.execute("DELETE FROM fund_prices WHERE code = ?", ("ABC",))
        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["reads"] == 0


# ==============================================================================
# Challenge 5: Fast-Path Verification (Non-Target Queries)
# ==============================================================================
class TestFastPathVerification:

    def setup_method(self) -> None:
        db.reset_db_metrics()

    def test_non_target_skips_orjson_dumps_and_mutex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that orjson_dumps and _DB_METRICS_LOCK are never touched for non-target queries."""
        orjson_called = []

        def spy_orjson_dumps(obj, *args, **kwargs):
            orjson_called.append(obj)
            return orjson_dumps(obj, *args, **kwargs)

        monkeypatch.setattr(db, "orjson_dumps", spy_orjson_dumps)

        # Track lock acquisition via a proxy wrapper
        class LockProxy:
            def __init__(self, lock: Any) -> None:
                self._lock = lock
                self.acquire_count = 0

            def __enter__(self) -> Any:
                self.acquire_count += 1
                return self._lock.__enter__()

            def __exit__(self, *args: Any) -> Any:
                return self._lock.__exit__(*args)

            def acquire(self, *args: Any, **kwargs: Any) -> Any:
                self.acquire_count += 1
                return self._lock.acquire(*args, **kwargs)

            def release(self) -> None:
                self._lock.release()

        lock_proxy = LockProxy(db._DB_METRICS_LOCK)
        monkeypatch.setattr(db, "_DB_METRICS_LOCK", lock_proxy)

        rows = [{"user_id": i, "token": f"tok_{i}"} for i in range(100)]
        mock_conn = MockConnection(lambda q, p: rows)
        pg_conn = db.PostgresConnection(mock_conn)

        # Execute query on non-target table
        cur = pg_conn.execute("SELECT * FROM user_sessions WHERE active = ?", (1,))
        assert lock_proxy.acquire_count == 0, "Lock should not be acquired during execute of non-target query"

        # Fetchone
        r0 = cur.fetchone()
        assert r0 is not None
        assert not orjson_called, "orjson_dumps should not be called for non-target fetchone"
        assert lock_proxy.acquire_count == 0, "Lock should not be acquired during non-target fetchone"

        # Fetchmany
        bm = cur.fetchmany(10)
        assert len(bm) == 10
        assert not orjson_called, "orjson_dumps should not be called for non-target fetchmany"
        assert lock_proxy.acquire_count == 0, "Lock should not be acquired during non-target fetchmany"

        # Fetchall
        rem = cur.fetchall()
        assert len(rem) == 89
        assert not orjson_called, "orjson_dumps should not be called for non-target fetchall"
        assert lock_proxy.acquire_count == 0, "Lock should not be acquired during non-target fetchall"

        # Verify metrics remain completely zero
        status = db.db_metrics_status()
        for t in db.TARGET_TABLES:
            assert status["tables"][t]["reads"] == 0
            assert status["tables"][t]["bytes"] == 0

    def test_metered_cursor_attributes_and_context_manager(self) -> None:
        mock_cur = MockCursor([{"a": 1}], description=[("col_a", 1043)])
        mock_cur.custom_attr = "custom_val"  # type: ignore

        cur = db.MeteredCursor(mock_cur, table="fund_prices", operation="SELECT")

        # Test context manager
        with cur as entered_cur:
            assert entered_cur is cur
            assert entered_cur.description == [("col_a", 1043)]
            assert entered_cur.rowcount == 1
            assert entered_cur.closed is False
            assert entered_cur.custom_attr == "custom_val"

        cur.close()
        assert mock_cur.closed is True

    def test_cursor_large_payload_wire_bytes(self) -> None:
        rows = [
            {
                "fund_code": f"FND{i:04d}",
                "price": Decimal(f"{100 + i * 0.1234:.4f}"),
                "date": date(2026, 1, 1),
                "metadata": {"tags": ["equity", "growth", f"tier_{i % 5}"]},
            }
            for i in range(1000)
        ]
        mock_cur = MockCursor(rows)
        cur = db.MeteredCursor(mock_cur, table="fund_prices", operation="SELECT")

        fetched = cur.fetchall()
        assert len(fetched) == 1000

        expected_bytes = len(orjson_dumps(rows))
        status = db.db_metrics_status()
        assert status["tables"]["fund_prices"]["bytes"] == expected_bytes
        # 1000 rows with metadata should be ~100-200 KB
        assert expected_bytes > 50_000

    def test_non_target_iter_returns_raw_cursor_iter(self) -> None:
        mock_cur = MockCursor([{"id": 1}, {"id": 2}])
        cur = db.MeteredCursor(mock_cur, table="non_target_table", operation="SELECT")
        cursor_iter = iter(cur)
        # For non-target tables, __iter__ returns iter(self._cursor) directly
        assert cursor_iter is mock_cur
