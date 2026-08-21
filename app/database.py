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
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

_SCHEMA_LOCK = threading.Lock()
_JSON_SCHEMA_READY = False
_REFRESH_LEASE_SCHEMA_READY = False
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
        return self._active_connection().execute(normalized, params)

    def executemany(self, query: str, params_seq: Any) -> Any:
        normalized = re.sub(r"\?", "%s", query)
        with self._active_connection().cursor() as cursor:
            return cursor.executemany(normalized, params_seq)

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
        _POSTGRES_POOL = ConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=max_size,
            timeout=timeout,
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


def _cache_key(path: Path) -> str:
    """Create a portable key relative to the application's processed folder."""

    normalized = Path(path)
    parts = list(normalized.parts)
    try:
        processed_index = next(i for i, part in enumerate(parts) if part.lower() == "processed")
        relative = parts[processed_index + 1 :]
    except StopIteration:
        relative = parts
    return "/".join(str(part).replace("\\", "/") for part in relative if str(part))


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
            conn.commit()
        _REFRESH_LEASE_SCHEMA_READY = True


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
                RETURNING version
                """,
                (_cache_key(path), json.dumps(payload, ensure_ascii=False, default=str), int(generation)),
            ).fetchone()
            if not isinstance(row, dict):
                conn.rollback()
                return False
            conn.commit()
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


def write_json_cache(path: Path, payload: dict[str, Any]) -> None:
    """Upsert one JSON cache document in Supabase."""

    if not database_enabled():
        return
    try:
        ensure_json_cache_schema()
        with connect_postgres() as conn:
            conn.execute(
                """
                INSERT INTO ragfin_json_cache (cache_key, payload, updated_at)
                VALUES (?, ?::jsonb, NOW())
                ON CONFLICT (cache_key) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = EXCLUDED.updated_at
                """,
                (_cache_key(path), json.dumps(payload, ensure_ascii=False, default=str)),
            )
            conn.commit()
    except Exception:
        # Local cache writes must remain successful even if Supabase is
        # temporarily unavailable.  The next refresh can repair the remote
        # copy without taking the API down.
        LOGGER.warning("remote JSON cache write failed for %s", path, exc_info=True)


def hydrate_json_cache(processed_dir: Path) -> int:
    """Download remote JSON cache documents into the local ephemeral folder."""

    if not database_enabled():
        return 0
    try:
        ensure_json_cache_schema()
        with connect_postgres() as conn:
            rows = conn.execute(
                "SELECT cache_key, payload FROM ragfin_json_cache ORDER BY cache_key"
            ).fetchall()
        hydrated = 0
        root = Path(processed_dir).resolve()
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("cache_key") or "").replace("\\", "/").strip("/")
            payload = row.get("payload")
            if not key or not isinstance(payload, dict) or ".." in Path(key).parts:
                continue
            target = (root / key).resolve()
            if root != target and root not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".remote.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            temporary.replace(target)
            hydrated += 1
        return hydrated
    except Exception:
        LOGGER.warning("remote JSON cache hydration failed", exc_info=True)
        return 0


def reset_database_state_for_tests() -> None:
    global _JSON_SCHEMA_READY, _REFRESH_LEASE_SCHEMA_READY
    close_postgres_pool()
    _JSON_SCHEMA_READY = False
    _REFRESH_LEASE_SCHEMA_READY = False
