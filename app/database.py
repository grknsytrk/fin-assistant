"""Optional PostgreSQL persistence used by deployed RAG-Fin instances.

Local development remains SQLite/file-cache based.  When
``RAGFIN_DATABASE_URL`` (or the conventional ``DATABASE_URL``) is present,
the small database wrapper in this module lets the existing repository code
use Supabase PostgreSQL without leaking provider-specific code into every
call site.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from app.etag import orjson_dumps

LOGGER = logging.getLogger(__name__)

TARGET_TABLES = (
    "ragfin_json_cache",
    "ragfin_kap_flow_events",
    "fund_snapshots",
    "fund_prices",
    "company_financial_snapshots",
    "instruments",
)

OPERATION_RE = re.compile(
    r"^\s*(?:--[^\n]*\n|\/\*.*?\*\/\s*)*\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP|DO)\b",
    re.IGNORECASE | re.DOTALL,
)

TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+(?:(?:IF\s+NOT\s+EXISTS|ONLY)\s+)?[\"\'`]?([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)[\"\'`]?",
    re.IGNORECASE,
)

_DB_METRICS_LOCK = threading.Lock()
_DB_PROCESS_STARTED = time.monotonic()
_DB_METRICS_STARTED = time.monotonic()

_DB_TABLE_METRICS: dict[str, dict[str, int]] = {
    table: {"reads": 0, "bytes": 0} for table in TARGET_TABLES
}
_DB_DETAILED_METRICS: dict[tuple[str, str], dict[str, int]] = {}


def parse_query_metadata(query: str) -> tuple[str, str]:
    """Extract (table_name, operation) from SQL string."""
    op_match = OPERATION_RE.search(query)
    operation = op_match.group(1).upper() if op_match else "UNKNOWN"

    if operation == "WITH":
        tail_op = re.search(r"\)\s*(SELECT|INSERT|UPDATE|DELETE)\b", query, re.IGNORECASE)
        operation = tail_op.group(1).upper() if tail_op else "SELECT"

    matches = TABLE_RE.findall(query)
    raw_tables = [m.split(".")[-1].lower() for m in matches]

    target = next((t for t in raw_tables if t in TARGET_TABLES), None)
    if target is None:
        target = raw_tables[0] if raw_tables else "unknown"

    return target, operation


def _record_db_query(table: str, operation: str) -> None:
    if table not in TARGET_TABLES:
        return
    with _DB_METRICS_LOCK:
        if operation == "SELECT":
            _DB_TABLE_METRICS[table]["reads"] += 1
        key = (table, operation)
        if key not in _DB_DETAILED_METRICS:
            _DB_DETAILED_METRICS[key] = {"queries": 0, "rows": 0, "bytes": 0}
        _DB_DETAILED_METRICS[key]["queries"] += 1


def _record_db_bytes(table: str, operation: str, byte_count: int, row_count: int = 1) -> None:
    if table not in TARGET_TABLES or byte_count <= 0:
        return
    with _DB_METRICS_LOCK:
        _DB_TABLE_METRICS[table]["bytes"] += byte_count
        key = (table, operation)
        if key not in _DB_DETAILED_METRICS:
            _DB_DETAILED_METRICS[key] = {"queries": 0, "rows": 0, "bytes": 0}
        _DB_DETAILED_METRICS[key]["bytes"] += byte_count
        _DB_DETAILED_METRICS[key]["rows"] += row_count


def _record_db_metric(table: str, operation: str = "SELECT", byte_count: int = 0) -> None:
    """Helper for internal recording and testing."""
    if table not in TARGET_TABLES:
        return
    _record_db_query(table, operation)
    if byte_count > 0:
        _record_db_bytes(table, operation, byte_count)


def db_metrics_status() -> dict[str, Any]:
    """Return current window telemetry for Supabase PostgreSQL egress."""
    now = time.monotonic()
    with _DB_METRICS_LOCK:
        return {
            "window_seconds": round(now - _DB_METRICS_STARTED, 1),
            "uptime_seconds": round(now - _DB_PROCESS_STARTED, 1),
            "tables": {
                table: dict(_DB_TABLE_METRICS[table]) for table in TARGET_TABLES
            },
        }


def reset_db_metrics() -> None:
    """Reset the telemetry window and counters for benchmarking and tests."""
    global _DB_METRICS_STARTED
    with _DB_METRICS_LOCK:
        _DB_METRICS_STARTED = time.monotonic()
        for table in TARGET_TABLES:
            _DB_TABLE_METRICS[table]["reads"] = 0
            _DB_TABLE_METRICS[table]["bytes"] = 0
        _DB_DETAILED_METRICS.clear()


def _exact_byte_metering_enabled() -> bool:
    return str(os.getenv("RAGFIN_DB_EXACT_BYTE_METERING", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _estimate_row_bytes(row: Any) -> int:
    if row is None:
        return 0
    if isinstance(row, dict):
        total = 2
        for key, value in row.items():
            total += (len(key) if isinstance(key, str) else 8) + 4
            if isinstance(value, (str, bytes, bytearray)):
                total += len(value)
            elif isinstance(value, (dict, list, tuple)):
                total += max(16, len(value) * 64)
            else:
                total += 16
        return total
    if isinstance(row, (list, tuple)):
        total = 2
        for value in row:
            if isinstance(value, (str, bytes, bytearray)):
                total += len(value) + 2
            elif isinstance(value, (dict, list, tuple)):
                total += max(16, len(value) * 64)
            else:
                total += 16
        return total
    if isinstance(row, (str, bytes, bytearray)):
        return len(row)
    return 64


def _estimate_rows_bytes(rows: list[Any]) -> int:
    if not rows:
        return 0
    if len(rows) <= 16:
        return sum(_estimate_row_bytes(r) for r in rows)
    sample_count = 16
    sample_bytes = sum(_estimate_row_bytes(r) for r in rows[:sample_count])
    return int((sample_bytes / sample_count) * len(rows))


class MeteredCursor:
    """Thin wrapper around psycopg cursor to measure row egress wire bytes."""

    def __init__(self, cursor: Any, table: str, operation: str) -> None:
        self._cursor = cursor
        self._table = table
        self._operation = operation
        self._is_target = table in TARGET_TABLES

    def _measure_row(self, row: Any) -> int:
        if _exact_byte_metering_enabled():
            return len(orjson_dumps(row))
        return _estimate_row_bytes(row)

    def _measure_rows(self, rows: list[Any]) -> int:
        if _exact_byte_metering_enabled():
            return len(orjson_dumps(rows))
        return _estimate_rows_bytes(rows)

    def fetchone(self) -> Any:
        row = self._cursor.fetchone()
        if row is not None and self._is_target:
            _record_db_bytes(self._table, self._operation, self._measure_row(row), 1)
        return row

    def fetchall(self) -> list[Any]:
        rows = self._cursor.fetchall()
        if rows and self._is_target:
            _record_db_bytes(self._table, self._operation, self._measure_rows(rows), len(rows))
        return rows

    def fetchmany(self, size: Optional[int] = None) -> list[Any]:
        if size is None:
            rows = self._cursor.fetchmany()
        else:
            rows = self._cursor.fetchmany(size)
        if rows and self._is_target:
            _record_db_bytes(self._table, self._operation, self._measure_rows(rows), len(rows))
        return rows

    def __iter__(self) -> Any:
        if not self._is_target:
            return iter(self._cursor)
        return self

    def __next__(self) -> Any:
        row = self._cursor.fetchone()
        if row is None:
            raise StopIteration
        if self._is_target:
            _record_db_bytes(self._table, self._operation, self._measure_row(row), 1)
        return row

    def __enter__(self) -> "MeteredCursor":
        if hasattr(self._cursor, "__enter__"):
            self._cursor.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if hasattr(self._cursor, "__exit__"):
            return self._cursor.__exit__(exc_type, exc_val, exc_tb)
        return None

    @property
    def description(self) -> Any:
        return getattr(self._cursor, "description", None)

    @property
    def rowcount(self) -> int:
        return getattr(self._cursor, "rowcount", -1)

    @property
    def closed(self) -> bool:
        return getattr(self._cursor, "closed", False)

    def close(self) -> None:
        if hasattr(self._cursor, "close"):
            self._cursor.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


_SCHEMA_LOCK = threading.Lock()
_JSON_SCHEMA_READY = False
_REFRESH_LEASE_SCHEMA_READY = False
_KAP_FLOW_SCHEMA_READY = False
_POOL_LOCK = threading.Lock()
_POSTGRES_POOL: Any = None
_POSTGRES_POOL_URL = ""
_DATABASE_STATEMENT_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("RAGFIN_DATABASE_STATEMENT_TIMEOUT_SECONDS", "15")),
)


def database_url() -> str:
    """Return the configured PostgreSQL URL, if deployment persistence is on."""

    return (os.getenv("RAGFIN_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip()


def database_enabled() -> bool:
    return bool(database_url())


def _require_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - exercised only misconfigured deploys
        raise RuntimeError(
            "RAGFIN_DATABASE_URL is set but psycopg is not installed; "
            "install the project requirements"
        ) from exc
    return psycopg, dict_row


class PostgresConnection:
    """Small DB-API compatibility wrapper for the repository's SQLite calls."""

    is_postgres = True

    def __init__(self, resource: Any) -> None:
        # ``resource`` is either a psycopg connection or a pool connection
        # context manager.  Keeping the context manager here lets callers use
        # the existing ``with connect_postgres()`` contract while returning a
        # pooled connection to psycopg-pool on exit.
        self._resource = resource
        self._connection: Any = None
        self._entered = False

    def _active_connection(self) -> Any:
        if self._connection is not None:
            return self._connection
        # Preserve the old direct-connection behavior for small maintenance
        # scripts that instantiate the wrapper without ``with``.  Pooled
        # resources don't expose ``execute`` until they are entered.
        if hasattr(self._resource, "execute"):
            return self._resource
        raise RuntimeError("PostgreSQL connection must be used as a context manager")

    def execute(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        # Existing local SQL uses SQLite's ``?`` placeholders.  The SQL used
        # by the application has no literal question marks, so this simple
        # conversion keeps both backends on the same query definitions.
        normalized = re.sub(r"\?", "%s", query)
        raw_cursor = self._active_connection().execute(normalized, params)
        table, operation = parse_query_metadata(query)
        if table in TARGET_TABLES:
            _record_db_query(table, operation)
        return MeteredCursor(raw_cursor, table=table, operation=operation)

    def executemany(self, query: str, params_seq: Any) -> Any:
        normalized = re.sub(r"\?", "%s", query)
        table, operation = parse_query_metadata(query)
        if table in TARGET_TABLES:
            _record_db_query(table, operation)
        with self._active_connection().cursor() as cursor:
            raw_cursor = cursor.executemany(normalized, params_seq)
            return MeteredCursor(raw_cursor or cursor, table=table, operation=operation)

    def commit(self) -> None:
        self._active_connection().commit()

    def rollback(self) -> None:
        self._active_connection().rollback()

    def close(self) -> None:
        if self._entered:
            self.__exit__(None, None, None)
            return
        close = getattr(self._resource, "close", None)
        if close:
            close()

    def __enter__(self) -> "PostgresConnection":
        if not self._entered:
            self._connection = self._resource.__enter__()
            self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        if not self._entered:
            return None
        try:
            return self._resource.__exit__(exc_type, exc_value, traceback)
        finally:
            self._connection = None
            self._entered = False


def _postgres_pool() -> Any:
    """Return the process-local psycopg pool used by the deployed API."""

    global _POSTGRES_POOL, _POSTGRES_POOL_URL
    url = database_url()
    if not url:
        raise RuntimeError("RAGFIN_DATABASE_URL is not configured")
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError(
            "RAGFIN_DATABASE_URL is set but psycopg-pool is not installed; "
            "install the project requirements"
        ) from exc

    with _POOL_LOCK:
        if _POSTGRES_POOL is not None and _POSTGRES_POOL_URL == url:
            return _POSTGRES_POOL
        if _POSTGRES_POOL is not None:
            _POSTGRES_POOL.close()
        max_size = max(1, int(os.getenv("RAGFIN_DATABASE_POOL_MAX_SIZE", "4")))
        timeout = max(1.0, float(os.getenv("RAGFIN_DATABASE_POOL_TIMEOUT_SECONDS", "10")))
        max_idle = max(30.0, float(os.getenv("RAGFIN_DATABASE_POOL_MAX_IDLE_SECONDS", "300")))
        max_lifetime = max(60.0, float(os.getenv("RAGFIN_DATABASE_POOL_MAX_LIFETIME_SECONDS", "1800")))
        _POSTGRES_POOL = ConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=max_size,
            timeout=timeout,
            max_idle=max_idle,
            max_lifetime=max_lifetime,
            kwargs={
                "row_factory": _require_psycopg()[1],
                # Safe for both Supavisor session mode and transaction mode.
                "prepare_threshold": None,
                # Remote cache persistence is an enhancement; a locked or
                # overloaded database must not keep the API refresh worker
                # waiting indefinitely.
                "options": f"-c statement_timeout={int(_DATABASE_STATEMENT_TIMEOUT_SECONDS * 1000)}",
            },
            open=True,
        )
        _POSTGRES_POOL_URL = url
        return _POSTGRES_POOL


def connect_postgres() -> PostgresConnection:
    return PostgresConnection(_postgres_pool().connection())


def close_postgres_pool() -> None:
    """Close the process-local pool, primarily for tests and graceful shutdown."""

    global _POSTGRES_POOL, _POSTGRES_POOL_URL
    with _POOL_LOCK:
        if _POSTGRES_POOL is not None:
            _POSTGRES_POOL.close()
        _POSTGRES_POOL = None
        _POSTGRES_POOL_URL = ""


_KNOWN_CACHE_ROOTS = ("_shared", "model_portfolio", "funds_cache", "kap_cache")


def _cache_key(path: Path) -> str:
    """Create a portable key relative to the application's processed folder."""

    normalized = Path(path)
    parts = [part for part in normalized.parts if str(part)]
    lowered = [str(part).lower() for part in parts]
    try:
        processed_index = next(i for i, part in enumerate(lowered) if part == "processed")
        relative = parts[processed_index + 1 :]
    except StopIteration:
        try:
            known_index = next(i for i, part in enumerate(lowered) if part in _KNOWN_CACHE_ROOTS)
            relative = parts[known_index:]
        except StopIteration:
            is_abs_or_os = normalized.is_absolute() or (
                len(parts) > 0 and (":" in str(parts[0]) or str(parts[0]).startswith(("/", "\\")))
            )
            relative = parts[-1:] if is_abs_or_os and len(parts) > 1 else parts
    return "/".join(str(part).replace("\\", "/") for part in relative if str(part))


def lock_down_postgres_tables(conn: Any, *table_names: str) -> None:
    """Enable RLS and revoke browser/API role grants on internal Postgres tables."""

    if str(os.getenv("RAGFIN_AUTO_MIGRATE", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    if not getattr(conn, "is_postgres", False):
        return
    for raw_name in table_names:
        table = str(raw_name or "").strip()
        if not table or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", table):
            continue
        conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        conn.execute(f"REVOKE ALL ON TABLE {table} FROM PUBLIC")
        conn.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                    EXECUTE 'REVOKE ALL ON TABLE {table} FROM anon';
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                    EXECUTE 'REVOKE ALL ON TABLE {table} FROM authenticated';
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
                    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {table} TO service_role';
                END IF;
            END
            $$
            """
        )


def ensure_json_cache_schema() -> None:
    """Create the remote JSON cache table once per process."""

    global _JSON_SCHEMA_READY
    if not database_enabled() or _JSON_SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _JSON_SCHEMA_READY:
            return
        with connect_postgres() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ragfin_json_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    version BIGINT NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            # Existing deployments predate fenced snapshot commits.  Keep
            # this migration idempotent so a restart upgrades them without a
            # separate migration command.
            conn.execute(
                "ALTER TABLE ragfin_json_cache ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ragfin_json_cache_updated_at
                ON ragfin_json_cache (updated_at)
                """
            )
            lock_down_postgres_tables(conn, "ragfin_json_cache")
            conn.commit()
        _JSON_SCHEMA_READY = True


def ensure_refresh_lease_schema() -> None:
    """Create the Postgres lease table used as the fencing authority."""

    global _REFRESH_LEASE_SCHEMA_READY
    if not database_enabled() or _REFRESH_LEASE_SCHEMA_READY:
        return
    ensure_json_cache_schema()
    with _SCHEMA_LOCK:
        if _REFRESH_LEASE_SCHEMA_READY:
            return
        with connect_postgres() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ragfin_refresh_lease (
                    resource_key TEXT PRIMARY KEY,
                    generation BIGINT NOT NULL DEFAULT 0,
                    job_id TEXT,
                    owner_token TEXT,
                    lease_until TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ragfin_refresh_lease_until
                ON ragfin_refresh_lease (lease_until)
                """
            )
            lock_down_postgres_tables(conn, "ragfin_refresh_lease")
            conn.commit()
        _REFRESH_LEASE_SCHEMA_READY = True


def ensure_kap_flow_schema() -> None:
    """Create the durable KAP flow event store used by cursor pagination."""

    global _KAP_FLOW_SCHEMA_READY
    if not database_enabled() or _KAP_FLOW_SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _KAP_FLOW_SCHEMA_READY:
            return
        with connect_postgres() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ragfin_kap_flow_events (
                    event_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    disclosure_id TEXT,
                    published_at TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL DEFAULT '',
                    stock_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
                    related_symbols JSONB NOT NULL DEFAULT '[]'::jsonb,
                    title TEXT NOT NULL,
                    subject TEXT,
                    category TEXT NOT NULL DEFAULT 'bildirim',
                    kap_url TEXT,
                    payload JSONB NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ragfin_kap_flow_events_published
                ON ragfin_kap_flow_events (published_at DESC, event_key DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ragfin_kap_flow_events_category_published
                ON ragfin_kap_flow_events (category, published_at DESC, event_key DESC)
                """
            )
            lock_down_postgres_tables(conn, "ragfin_kap_flow_events")
            conn.commit()
        _KAP_FLOW_SCHEMA_READY = True


def upsert_kap_flow_events(events: list[dict[str, Any]]) -> int:
    """Atomically upsert changed KAP flow events into Postgres in batches."""

    if not database_enabled() or not events:
        return 0
    ensure_kap_flow_schema()
    rows = [
        (
            str(event["event_key"]),
            str(event.get("source") or "KAP"),
            str(event.get("disclosure_id") or "") or None,
            str(event["published_at"]),
            str(event.get("symbol") or ""),
            json.dumps(event.get("stock_codes") or [], ensure_ascii=False),
            json.dumps(event.get("related_symbols") or [], ensure_ascii=False),
            str(event.get("title") or "KAP Bildirimi"),
            str(event.get("subject") or "") or None,
            str(event.get("category") or "bildirim"),
            str(event.get("kap_url") or "") or None,
            json.dumps(event.get("payload") or {}, ensure_ascii=False, default=str),
        )
        for event in events
        if event.get("event_key") and event.get("published_at")
    ]
    if not rows:
        return 0
    value_tuple = "(?, ?, ?, ?, ?, ?::jsonb, ?::jsonb, ?, ?, ?, ?, ?::jsonb, NOW(), NOW())"
    chunk_size = 150
    modified_count = 0
    with connect_postgres() as conn:
        for offset in range(0, len(rows), chunk_size):
            chunk = rows[offset : offset + chunk_size]
            placeholders = ", ".join(value_tuple for _ in chunk)
            flat_params = [col for row in chunk for col in row]
            cursor = conn.execute(
                f"""
                INSERT INTO ragfin_kap_flow_events (
                    event_key, source, disclosure_id, published_at, symbol,
                    stock_codes, related_symbols, title, subject, category,
                    kap_url, payload, ingested_at, updated_at
                )
                VALUES {placeholders}
                ON CONFLICT (event_key) DO UPDATE SET
                    source = EXCLUDED.source,
                    disclosure_id = EXCLUDED.disclosure_id,
                    published_at = EXCLUDED.published_at,
                    symbol = EXCLUDED.symbol,
                    stock_codes = EXCLUDED.stock_codes,
                    related_symbols = EXCLUDED.related_symbols,
                    title = EXCLUDED.title,
                    subject = EXCLUDED.subject,
                    category = EXCLUDED.category,
                    kap_url = EXCLUDED.kap_url,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                WHERE ragfin_kap_flow_events.payload IS DISTINCT FROM EXCLUDED.payload
                   OR ragfin_kap_flow_events.source IS DISTINCT FROM EXCLUDED.source
                   OR ragfin_kap_flow_events.disclosure_id IS DISTINCT FROM EXCLUDED.disclosure_id
                   OR ragfin_kap_flow_events.stock_codes IS DISTINCT FROM EXCLUDED.stock_codes
                   OR ragfin_kap_flow_events.related_symbols IS DISTINCT FROM EXCLUDED.related_symbols
                   OR ragfin_kap_flow_events.title IS DISTINCT FROM EXCLUDED.title
                   OR ragfin_kap_flow_events.subject IS DISTINCT FROM EXCLUDED.subject
                   OR ragfin_kap_flow_events.category IS DISTINCT FROM EXCLUDED.category
                   OR ragfin_kap_flow_events.published_at IS DISTINCT FROM EXCLUDED.published_at
                   OR ragfin_kap_flow_events.symbol IS DISTINCT FROM EXCLUDED.symbol
                   OR ragfin_kap_flow_events.kap_url IS DISTINCT FROM EXCLUDED.kap_url
                RETURNING 1
                """,
                tuple(flat_params),
            )
            if hasattr(cursor, "fetchall"):
                modified_count += len(cursor.fetchall() or [])
            else:
                modified_count += len(chunk)
        conn.commit()
    return modified_count


def read_kap_flow_head(*, category: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Read only the newest (event_key, published_at) cursor without fetching payload JSONB."""

    if not database_enabled():
        return None
    ensure_kap_flow_schema()
    where = "WHERE category = ?" if category else ""
    params: tuple[Any, ...] = (category,) if category else ()
    with connect_postgres() as conn:
        row = conn.execute(
            f"""
            SELECT event_key, published_at
            FROM ragfin_kap_flow_events
            {where}
            ORDER BY published_at DESC, event_key DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    if not isinstance(row, dict):
        return None
    event_key = str(row.get("event_key") or "").strip()
    published_at = row.get("published_at")
    if not event_key or not published_at:
        return None
    return {"event_key": event_key, "published_at": published_at}


def read_kap_flow_events(
    *,
    limit: int,
    before: Optional[tuple[str, str]] = None,
    after: Optional[tuple[str, str]] = None,
    category: Optional[str] = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Read a keyset-paginated KAP flow page from the durable store."""

    if not database_enabled():
        return [], False
    ensure_kap_flow_schema()
    page_size = max(1, min(int(limit), 500))
    conditions: list[str] = []
    params: list[Any] = []
    if category:
        conditions.append("category = ?")
        params.append(category)
    if before:
        conditions.append("(published_at, event_key) < (?, ?)")
        params.extend(before)
    if after:
        conditions.append("(published_at, event_key) > (?, ?)")
        params.extend(after)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with connect_postgres() as conn:
        rows = conn.execute(
            f"""
            SELECT event_key, published_at, payload
            FROM ragfin_kap_flow_events
            {where}
            ORDER BY published_at DESC, event_key DESC
            LIMIT ?
            """,
            (*params, page_size + 1),
        ).fetchall()
    has_more = len(rows) > page_size
    output: list[dict[str, Any]] = []
    for row in rows[:page_size]:
        if not isinstance(row, dict) or not isinstance(row.get("payload"), dict):
            continue
        output.append(
            {
                "event_key": str(row.get("event_key") or ""),
                "published_at": row.get("published_at"),
                "payload": dict(row["payload"]),
            }
        )
    return output, has_more


def reset_kap_flow_schema_for_tests() -> None:
    global _KAP_FLOW_SCHEMA_READY
    _KAP_FLOW_SCHEMA_READY = False


def acquire_refresh_lease(
    resource_key: str,
    *,
    job_id: str,
    owner_token: str,
    ttl_seconds: int,
) -> Optional[dict[str, Any]]:
    """Atomically claim a generation lease, returning its generation."""

    if not database_enabled():
        return None
    try:
        ensure_refresh_lease_schema()
        with connect_postgres() as conn:
            row = conn.execute(
                """
                INSERT INTO ragfin_refresh_lease
                    (resource_key, generation, job_id, owner_token, lease_until, updated_at)
                VALUES (?, 1, ?, ?, NOW() + (? * INTERVAL '1 second'), NOW())
                ON CONFLICT (resource_key) DO UPDATE SET
                    generation = ragfin_refresh_lease.generation + 1,
                    job_id = EXCLUDED.job_id,
                    owner_token = EXCLUDED.owner_token,
                    lease_until = EXCLUDED.lease_until,
                    updated_at = NOW()
                WHERE ragfin_refresh_lease.lease_until IS NULL
                   OR ragfin_refresh_lease.lease_until <= NOW()
                RETURNING generation, job_id, owner_token, lease_until
                """,
                (resource_key, job_id, owner_token, max(1, int(ttl_seconds))),
            ).fetchone()
            conn.commit()
        return dict(row) if isinstance(row, dict) else None
    except Exception:
        LOGGER.warning("refresh lease acquire failed for %s", resource_key, exc_info=True)
        return None


def renew_refresh_lease(
    resource_key: str,
    *,
    job_id: str,
    owner_token: str,
    ttl_seconds: int,
) -> bool:
    if not database_enabled():
        return True
    try:
        ensure_refresh_lease_schema()
        with connect_postgres() as conn:
            row = conn.execute(
                """
                UPDATE ragfin_refresh_lease
                SET lease_until = NOW() + (? * INTERVAL '1 second'), updated_at = NOW()
                WHERE resource_key = ? AND job_id = ? AND owner_token = ?
                  AND lease_until > NOW()
                RETURNING generation
                """,
                (max(1, int(ttl_seconds)), resource_key, job_id, owner_token),
            ).fetchone()
            conn.commit()
        return isinstance(row, dict)
    except Exception:
        LOGGER.warning("refresh lease renew failed for %s", resource_key, exc_info=True)
        return False


def release_refresh_lease(resource_key: str, *, job_id: str, owner_token: str) -> bool:
    if not database_enabled():
        return True
    try:
        ensure_refresh_lease_schema()
        with connect_postgres() as conn:
            row = conn.execute(
                """
                UPDATE ragfin_refresh_lease
                SET lease_until = NOW(), updated_at = NOW()
                WHERE resource_key = ? AND job_id = ? AND owner_token = ?
                RETURNING generation
                """,
                (resource_key, job_id, owner_token),
            ).fetchone()
            conn.commit()
        return isinstance(row, dict)
    except Exception:
        LOGGER.warning("refresh lease release failed for %s", resource_key, exc_info=True)
        return False


def get_refresh_lease(resource_key: str) -> Optional[dict[str, Any]]:
    if not database_enabled():
        return None
    try:
        ensure_refresh_lease_schema()
        with connect_postgres() as conn:
            row = conn.execute(
                "SELECT generation, job_id, owner_token, lease_until FROM ragfin_refresh_lease WHERE resource_key = ?",
                (resource_key,),
            ).fetchone()
        return dict(row) if isinstance(row, dict) else None
    except Exception:
        LOGGER.warning("refresh lease read failed for %s", resource_key, exc_info=True)
        return None


def commit_json_cache_if_fenced(
    path: Path,
    payload: dict[str, Any],
    *,
    resource_key: str,
    job_id: str,
    owner_token: str,
    generation: int,
) -> bool:
    """Commit a cache payload only while the caller owns the current lease.

    The lease row lock and versioned cache upsert intentionally share one
    transaction.  A late worker can therefore never pass a separate
    check-then-write gap after another generation takes over.
    """

    if not database_enabled():
        return False
    try:
        ensure_refresh_lease_schema()
        with connect_postgres() as conn:
            lease = conn.execute(
                """
                SELECT generation, job_id, owner_token, lease_until
                FROM ragfin_refresh_lease
                WHERE resource_key = ? AND lease_until > NOW()
                FOR UPDATE
                """,
                (resource_key,),
            ).fetchone()
            if not isinstance(lease, dict):
                conn.rollback()
                return False
            if (
                int(lease.get("generation") or -1) != int(generation)
                or str(lease.get("job_id") or "") != job_id
                or str(lease.get("owner_token") or "") != owner_token
                or lease.get("lease_until") is None
            ):
                conn.rollback()
                return False
            row = conn.execute(
                """
                INSERT INTO ragfin_json_cache (cache_key, payload, version, updated_at)
                VALUES (?, ?::jsonb, ?, NOW())
                ON CONFLICT (cache_key) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    version = EXCLUDED.version,
                    updated_at = EXCLUDED.updated_at
                WHERE ragfin_json_cache.version < EXCLUDED.version
                RETURNING version, updated_at::text AS updated_at
                """,
                (_cache_key(path), json.dumps(payload, ensure_ascii=False, default=str), int(generation)),
            ).fetchone()
            if not isinstance(row, dict):
                conn.rollback()
                return False
            conn.commit()
        updated_at = str(row.get("updated_at") or "").strip()
        if updated_at:
            try:
                from app.json_cache import remember_json_cache_remote_stamp

                remember_json_cache_remote_stamp(path, (updated_at, int(row.get("version") or 0)))
            except Exception:
                pass
        return True
    except Exception:
        LOGGER.warning("fenced JSON cache commit failed for %s", path, exc_info=True)
        return False


def read_json_cache(path: Path) -> Optional[dict[str, Any]]:
    """Read a JSON cache document from Supabase, returning ``None`` on miss."""

    if not database_enabled():
        return None
    try:
        ensure_json_cache_schema()
        with connect_postgres() as conn:
            row = conn.execute(
                "SELECT payload FROM ragfin_json_cache WHERE cache_key = ?",
                (_cache_key(path),),
            ).fetchone()
        payload = row.get("payload") if isinstance(row, dict) else None
        return dict(payload) if isinstance(payload, dict) else None
    except Exception:
        LOGGER.warning("remote JSON cache read failed for %s", path, exc_info=True)
        return None


def read_json_cache_conditional(
    path: Path,
    *,
    known_stamp: Optional[tuple[str, int]] = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[tuple[str, int]]]:
    """Revalidate a JSON cache document against its remote ``(updated_at, version)`` stamp.

    Returns ``(status, payload, stamp)`` where ``status`` is one of:
    - ``"not_modified"``: remote row matches ``known_stamp``; ``payload`` is ``None``
      and no JSONB body is transferred over the wire.
    - ``"hit"``: remote row exists and differs from ``known_stamp`` (or ``known_stamp``
      was not provided); ``payload`` contains the full document.
    - ``"miss"``: remote row does not exist or database is unavailable.
    """

    if not database_enabled():
        return ("miss", None, None)
    try:
        ensure_json_cache_schema()
        cache_key = _cache_key(path)
        with connect_postgres() as conn:
            if known_stamp is not None and known_stamp[0]:
                known_updated_at, known_version = str(known_stamp[0]), int(known_stamp[1])
                row = conn.execute(
                    """
                    SELECT
                        updated_at::text AS updated_at,
                        COALESCE(version, 0) AS version,
                        CASE
                            WHEN updated_at::text = ? AND COALESCE(version, 0) = ? THEN FALSE
                            ELSE TRUE
                        END AS modified,
                        CASE
                            WHEN updated_at::text = ? AND COALESCE(version, 0) = ? THEN NULL
                            ELSE payload
                        END AS payload
                    FROM ragfin_json_cache
                    WHERE cache_key = ?
                    """,
                    (known_updated_at, known_version, known_updated_at, known_version, cache_key),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT
                        updated_at::text AS updated_at,
                        COALESCE(version, 0) AS version,
                        TRUE AS modified,
                        payload
                    FROM ragfin_json_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
        if not isinstance(row, dict):
            return ("miss", None, None)
        updated_at = str(row.get("updated_at") or "").strip()
        version = int(row.get("version") or 0)
        stamp = (updated_at, version) if updated_at else None
        if known_stamp is not None and not bool(row.get("modified", True)):
            return ("not_modified", None, stamp or known_stamp)
        payload = row.get("payload")
        if not isinstance(payload, dict):
            return ("miss", None, None)
        return ("hit", dict(payload), stamp)
    except Exception:
        LOGGER.warning("remote JSON cache conditional read failed for %s", path, exc_info=True)
        return ("miss", None, None)


def read_json_cache_history_metadata(path: Path) -> Optional[dict[str, Any]]:
    """Read only the small coverage fields needed for a fund-history cache check.

    Historical points are also stored row-by-row in ``fund_prices``. Returning
    the legacy JSON document here would transfer the entire history just to
    inspect a handful of metadata values.
    """

    if not database_enabled():
        return None
    try:
        ensure_json_cache_schema()
        with connect_postgres() as conn:
            row = conn.execute(
                """
                SELECT
                    payload #>> '{source_metadata,parse_status}' AS parse_status,
                    payload #>> '{source_metadata,requested_start_date}' AS requested_start_date,
                    payload #>> '{source_metadata,date_max}' AS date_max,
                    COALESCE(
                        payload #>> '{source_metadata,as_of}',
                        payload #>> '{as_of}'
                    ) AS as_of
                FROM ragfin_json_cache
                WHERE cache_key = ?
                """,
                (_cache_key(path),),
            ).fetchone()
        return dict(row) if isinstance(row, dict) else None
    except Exception:
        LOGGER.warning("remote JSON cache metadata read failed for %s", path, exc_info=True)
        return None


def read_json_cache_kap_metadata() -> dict[str, dict[str, Any]]:
    """Read compact headers for cached KAP company documents without transferring financial tables."""

    if not database_enabled():
        return {}
    try:
        ensure_json_cache_schema()
        with connect_postgres() as conn:
            rows = conn.execute(
                """
                SELECT
                    cache_key,
                    payload->>'ok' AS ok,
                    payload->>'company' AS company,
                    payload->>'stock_code' AS stock_code,
                    payload->>'company_title' AS company_title,
                    payload->>'member_oid' AS member_oid,
                    payload->>'source_url' AS source_url,
                    payload->>'schema_version' AS schema_version,
                    payload->>'live_period_count' AS live_period_count,
                    (
                        SELECT COALESCE(
                            jsonb_agg(
                                jsonb_build_object(
                                    'year', q->'year',
                                    'period', q->'period',
                                    'disclosure_index', q->'disclosure_index',
                                    'stock_code', q->'stock_code'
                                )
                            ),
                            '[]'::jsonb
                        )
                        FROM jsonb_array_elements(
                            CASE
                                WHEN jsonb_typeof(payload->'quarters') = 'array' THEN payload->'quarters'
                                ELSE '[]'::jsonb
                            END
                        ) AS q
                    ) AS quarter_headers
                FROM ragfin_json_cache
                WHERE cache_key LIKE ?
                """,
                ("kap_cache/%.json",),
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            cache_key = str(row.get("cache_key") or "").strip()
            if not cache_key:
                continue
            stem = Path(cache_key).stem.strip().upper()
            if not stem:
                continue
            raw_ok = row.get("ok")
            ok_val = False if str(raw_ok).strip().lower() == "false" else True
            try:
                schema_version = int(row.get("schema_version") or 0)
            except (TypeError, ValueError):
                schema_version = 0
            try:
                live_period_count = int(row.get("live_period_count") or 0)
            except (TypeError, ValueError):
                live_period_count = 0
            raw_headers = row.get("quarter_headers")
            if isinstance(raw_headers, str):
                try:
                    raw_headers = json.loads(raw_headers)
                except ValueError:
                    raw_headers = []
            quarters = [q for q in (raw_headers or []) if isinstance(q, dict)]
            out[stem] = {
                "ok": ok_val,
                "company": row.get("company") or stem,
                "stock_code": row.get("stock_code") or stem,
                "company_title": row.get("company_title"),
                "member_oid": row.get("member_oid"),
                "source_url": row.get("source_url"),
                "schema_version": schema_version,
                "live_period_count": live_period_count,
                "quarters": quarters,
            }
        return out
    except Exception:
        LOGGER.warning("remote KAP metadata batch read failed", exc_info=True)
        return {}


def write_json_cache(path: Path, payload: dict[str, Any]) -> None:
    """Upsert one JSON cache document in Supabase."""

    if not database_enabled():
        return
    try:
        ensure_json_cache_schema()
        with connect_postgres() as conn:
            key = _cache_key(path)
            cursor = conn.execute(
                """
                INSERT INTO ragfin_json_cache (cache_key, payload, updated_at)
                VALUES (?, ?::jsonb, NOW())
                ON CONFLICT (cache_key) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = EXCLUDED.updated_at
                WHERE ragfin_json_cache.payload IS DISTINCT FROM EXCLUDED.payload
                RETURNING updated_at::text AS updated_at, COALESCE(version, 0) AS version
                """,
                (key, json.dumps(payload, ensure_ascii=False, default=str)),
            )
            row = cursor.fetchone() if hasattr(cursor, "fetchone") else None
            if not isinstance(row, dict):
                existing_cur = conn.execute(
                    """
                    SELECT updated_at::text AS updated_at, COALESCE(version, 0) AS version
                    FROM ragfin_json_cache
                    WHERE cache_key = ?
                    """,
                    (key,),
                )
                row = existing_cur.fetchone() if hasattr(existing_cur, "fetchone") else None
            conn.commit()
        if isinstance(row, dict):
            updated_at = str(row.get("updated_at") or "").strip()
            if updated_at:
                try:
                    from app.json_cache import remember_json_cache_remote_stamp

                    remember_json_cache_remote_stamp(path, (updated_at, int(row.get("version") or 0)))
                except Exception:
                    pass
    except Exception:
        # Local cache writes must remain successful even if Supabase is
        # temporarily unavailable.  The next refresh can repair the remote
        # copy without taking the API down.
        LOGGER.warning("remote JSON cache write failed for %s", path, exc_info=True)


def hydrate_json_cache(
    processed_dir: Path,
    *,
    exclude_key_prefixes: tuple[str, ...] = (),
    include_key_prefixes: tuple[str, ...] = (),
) -> int:
    """Download selected remote JSON cache documents into the local folder.

    Large daily fund snapshots and KAP company caches are fetched on demand.
    When ``include_key_prefixes`` is supplied, only those prefixes are hydrated,
    preventing unrelated or stale cache rows from causing startup egress.
    """

    if not database_enabled():
        return 0
    try:
        ensure_json_cache_schema()
        normalized_excludes: list[str] = []
        for raw_prefix in exclude_key_prefixes:
            prefix = str(raw_prefix or "").replace("\\", "/").strip("/")
            if prefix:
                normalized_excludes.append(f"{prefix}/")
        excludes = tuple(dict.fromkeys(normalized_excludes))

        normalized_includes: list[str] = []
        for raw_prefix in include_key_prefixes:
            prefix = str(raw_prefix or "").replace("\\", "/").strip("/")
            if prefix:
                normalized_includes.append(f"{prefix}/")
        includes = tuple(dict.fromkeys(normalized_includes))

        conditions: list[str] = [
            "POSITION('://' IN cache_key) = 0",
            "SUBSTR(cache_key, 1, 1) <> '/'",
        ]
        query_params: list[Any] = []
        if includes:
            conditions.append("(" + " OR ".join("cache_key LIKE ? ESCAPE '\\'" for _ in includes) + ")")
            for prefix in includes:
                escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                query_params.append(f"{escaped}%")
        if excludes:
            for prefix in excludes:
                escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                conditions.append("cache_key NOT LIKE ? ESCAPE '\\'")
                query_params.append(f"{escaped}%")

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        with connect_postgres() as conn:
            rows = conn.execute(
                f"SELECT cache_key, payload, updated_at::text AS updated_at, COALESCE(version, 0) AS version FROM ragfin_json_cache {where_clause} ORDER BY cache_key",
                tuple(query_params),
            ).fetchall()
        hydrated = 0
        measured_bytes = 0
        root = Path(processed_dir).resolve()
        from app.json_cache import remember_json_cache_read

        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("cache_key") or "").replace("\\", "/").strip("/")
            payload = row.get("payload")
            if not key or not isinstance(payload, dict) or ".." in Path(key).parts:
                continue
            measured_bytes += len(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
            )
            target = (root / key).resolve()
            if root != target and root not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            existing_matches = False
            if target.exists():
                try:
                    existing_matches = target.read_text(encoding="utf-8") == serialized
                except OSError:
                    existing_matches = False
            if not existing_matches:
                temporary = target.with_suffix(target.suffix + ".remote.tmp")
                temporary.write_text(
                    serialized,
                    encoding="utf-8",
                )
                temporary.replace(target)
            updated_at = str(row.get("updated_at") or "").strip()
            stamp = (updated_at, int(row.get("version") or 0)) if updated_at else None
            remember_json_cache_read(target, remote_stamp=stamp)
            hydrated += 1
        LOGGER.info(
            "json_cache_hydrate rows=%d hydrated=%d excluded_in_sql=%s excluded_prefixes=%s approximate_payload_bytes=%d over_1mib=%s",
            len(rows),
            hydrated,
            bool(excludes),
            ",".join(excludes),
            measured_bytes,
            measured_bytes > 1024 * 1024,
        )
        return hydrated
    except Exception:
        LOGGER.warning("remote JSON cache hydration failed", exc_info=True)
        return 0


def reset_database_state_for_tests() -> None:
    global _JSON_SCHEMA_READY, _REFRESH_LEASE_SCHEMA_READY, _KAP_FLOW_SCHEMA_READY
    close_postgres_pool()
    _JSON_SCHEMA_READY = False
    _REFRESH_LEASE_SCHEMA_READY = False
    _KAP_FLOW_SCHEMA_READY = False
    reset_db_metrics()
