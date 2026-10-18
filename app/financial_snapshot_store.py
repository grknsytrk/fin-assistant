"""Persist compact company fundamentals derived from KAP and provider data."""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from app.database import connect_postgres, database_enabled

LOGGER = logging.getLogger(__name__)

FINANCIAL_SNAPSHOT_DB_FILENAME = os.getenv(
    "RAGFIN_REFERENCE_DATA_DB_FILENAME",
    "reference_data.sqlite3",
)

_SCHEMA_LOCK = threading.Lock()
_POSTGRES_SCHEMA_READY = False
_LOCAL_SCHEMA_READY: set[str] = set()
_CACHE_LOCK = threading.RLock()
_SNAPSHOT_MEMORY_CACHE: Dict[str, tuple[float, Dict[str, Dict[str, Any]]]] = {}
_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = 60.0

_SNAPSHOT_COLUMNS = (
    "symbol",
    "company_title",
    "company_kind",
    "sector",
    "latest_quarter",
    "quarter_count",
    "has_kap_cache",
    "shares_outstanding",
    "share_source",
    "free_float_pct",
    "ttm_net_kar",
    "ttm_satis_gelirleri",
    "ttm_favok",
    "ozkaynaklar",
    "net_borc",
    "toplam_varliklar",
    "serbest_nakit_akisi",
    "roe_pct",
    "favok_margin_pct",
    "net_margin_pct",
    "net_borc_favok",
    "provider_fk",
    "provider_pd_dd",
    "provider_fd_favok",
    "provider_fd_satis",
    "provider_period",
    "source_mtime_ns",
    "as_of",
    "updated_at",
)


def financial_snapshot_db_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / FINANCIAL_SNAPSHOT_DB_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(processed_dir: Path) -> str:
    if database_enabled():
        return "postgres"
    return str(financial_snapshot_db_path(processed_dir).resolve())


def _invalidate_memory_cache(processed_dir: Path) -> None:
    key = _cache_key(processed_dir)
    with _CACHE_LOCK:
        _SNAPSHOT_MEMORY_CACHE.pop(key, None)


def _sqlite_connection(processed_dir: Path) -> sqlite3.Connection:
    path = financial_snapshot_db_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _create_snapshot_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS company_financial_snapshots (
            symbol TEXT PRIMARY KEY,
            company_title TEXT,
            company_kind TEXT NOT NULL DEFAULT 'generic',
            sector TEXT,
            latest_quarter TEXT,
            quarter_count INTEGER NOT NULL DEFAULT 0,
            has_kap_cache INTEGER NOT NULL DEFAULT 0,
            shares_outstanding REAL,
            share_source TEXT,
            free_float_pct REAL,
            ttm_net_kar REAL,
            ttm_satis_gelirleri REAL,
            ttm_favok REAL,
            ozkaynaklar REAL,
            net_borc REAL,
            toplam_varliklar REAL,
            serbest_nakit_akisi REAL,
            roe_pct REAL,
            favok_margin_pct REAL,
            net_margin_pct REAL,
            net_borc_favok REAL,
            provider_fk REAL,
            provider_pd_dd REAL,
            provider_fd_favok REAL,
            provider_fd_satis REAL,
            provider_period TEXT,
            source_mtime_ns BIGINT,
            as_of TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def ensure_financial_snapshot_schema(processed_dir: Path) -> None:
    """Create the local or shared financial snapshot table once per process."""

    global _POSTGRES_SCHEMA_READY
    if database_enabled():
        if _POSTGRES_SCHEMA_READY:
            return
        with _SCHEMA_LOCK:
            if _POSTGRES_SCHEMA_READY:
                return
            with connect_postgres() as conn:
                _create_snapshot_schema(conn)
                # This table is only used by the application server. Keep it
                # unavailable to browser/API roles even on Supabase projects
                # that still grant new public tables to anon/authenticated.
                conn.execute(
                    "ALTER TABLE company_financial_snapshots ENABLE ROW LEVEL SECURITY"
                )
                conn.execute(
                    "REVOKE ALL ON TABLE company_financial_snapshots FROM PUBLIC"
                )
                conn.execute(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                            EXECUTE 'REVOKE ALL ON TABLE company_financial_snapshots FROM anon';
                        END IF;
                        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                            EXECUTE 'REVOKE ALL ON TABLE company_financial_snapshots FROM authenticated';
                        END IF;
                        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
                            EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE company_financial_snapshots TO service_role';
                        END IF;
                    END
                    $$
                    """
                )
                conn.commit()
            _POSTGRES_SCHEMA_READY = True
        return

    key = str(financial_snapshot_db_path(processed_dir).resolve())
    if key in _LOCAL_SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if key in _LOCAL_SCHEMA_READY:
            return
        conn = _sqlite_connection(processed_dir)
        try:
            _create_snapshot_schema(conn)
        finally:
            conn.close()
        _LOCAL_SCHEMA_READY.add(key)


@contextmanager
def _connection(processed_dir: Path) -> Iterator[Any]:
    ensure_financial_snapshot_schema(processed_dir)
    if database_enabled():
        with connect_postgres() as conn:
            yield conn
        return

    conn = _sqlite_connection(processed_dir)
    try:
        yield conn
    finally:
        conn.close()


def _finite_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _number_from_metric_set(
    quarter: Dict[str, Any],
    metric_key: str,
    priority: Iterable[str],
) -> Optional[float]:
    for container_name in priority:
        container = quarter.get(container_name)
        if not isinstance(container, dict):
            continue
        metric = container.get(metric_key)
        value = metric.get("value") if isinstance(metric, dict) else metric
        number = _finite_number(value)
        if number is not None:
            return number
    return None


def extract_financial_snapshot_from_kap_payload(
    symbol: str,
    payload: Dict[str, Any],
    *,
    source_mtime_ns: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a KAP cache payload and extract its compact latest snapshot."""

    if not isinstance(payload, dict):
        return None
    normalized_symbol = str(symbol or payload.get("stock_code") or payload.get("company") or "").strip().upper()
    if not normalized_symbol:
        return None

    from app.kap_financials import _build_ttm_from_reports, _quarter_sort_key
    from app.kap_service import normalize_snapshot_for_frontend

    normalized = normalize_snapshot_for_frontend(payload, allow_network=False)
    quarters_raw = normalized.get("quarters")
    quarters = sorted(
        [quarter for quarter in quarters_raw if isinstance(quarter, dict)]
        if isinstance(quarters_raw, list)
        else [],
        key=_quarter_sort_key,
    )
    latest = quarters[-1] if quarters else {}
    shares = _number_from_metric_set(
        latest,
        "odenmis_sermaye",
        ("metrics", "metrics_ytd"),
    )
    share_source = "odenmis_sermaye" if shares is not None else None
    if shares is None:
        shares = _number_from_metric_set(
            latest,
            "cikarilmis_sermaye",
            ("metrics", "metrics_ytd"),
        )
        if shares is not None:
            share_source = "cikarilmis_sermaye"
    if shares is not None and shares <= 0:
        shares = None
        share_source = None

    ttm_net_kar = _build_ttm_from_reports(quarters, "net_kar") if quarters else None
    ttm_satis = _build_ttm_from_reports(quarters, "satis_gelirleri") if quarters else None
    ttm_favok = _build_ttm_from_reports(quarters, "favok") if quarters else None
    ttm_fcf = _build_ttm_from_reports(quarters, "serbest_nakit_akisi") if quarters else None
    equity = _number_from_metric_set(latest, "ozkaynaklar", ("metrics", "metrics_ytd"))
    net_debt = _number_from_metric_set(latest, "net_borc", ("metrics", "metrics_ytd"))
    total_assets = _number_from_metric_set(latest, "toplam_varliklar", ("metrics", "metrics_ytd"))

    roe_pct = 100.0 * ttm_net_kar / equity if ttm_net_kar is not None and equity not in (None, 0) else None
    favok_margin_pct = 100.0 * ttm_favok / ttm_satis if ttm_favok is not None and ttm_satis not in (None, 0) else None
    net_margin_pct = 100.0 * ttm_net_kar / ttm_satis if ttm_net_kar is not None and ttm_satis not in (None, 0) else None
    net_debt_favok = net_debt / ttm_favok if net_debt is not None and ttm_favok not in (None, 0) else None

    if source_mtime_ns is None:
        cache_path = Path(payload.get("_cache_path") or "")
        if not cache_path.is_absolute():
            cache_path = Path(payload.get("_processed_dir") or "") / "kap_cache" / f"{normalized_symbol}.json"
        try:
            source_mtime_ns = cache_path.stat().st_mtime_ns
        except OSError:
            source_mtime_ns = None

    quarter_name = str(latest.get("quarter") or "").strip().upper() or None
    return {
        "symbol": normalized_symbol,
        "company_title": str(normalized.get("company_title") or payload.get("company_title") or "").strip() or None,
        "company_kind": str(normalized.get("company_kind") or payload.get("company_kind") or "generic").strip().lower(),
        "sector": None,
        "latest_quarter": quarter_name,
        "quarter_count": len(quarters),
        "has_kap_cache": True,
        "shares_outstanding": shares,
        "share_source": share_source,
        "free_float_pct": None,
        "ttm_net_kar": _finite_number(ttm_net_kar),
        "ttm_satis_gelirleri": _finite_number(ttm_satis),
        "ttm_favok": _finite_number(ttm_favok),
        "ozkaynaklar": equity,
        "net_borc": net_debt,
        "toplam_varliklar": total_assets,
        "serbest_nakit_akisi": _finite_number(ttm_fcf),
        "roe_pct": _finite_number(roe_pct),
        "favok_margin_pct": _finite_number(favok_margin_pct),
        "net_margin_pct": _finite_number(net_margin_pct),
        "net_borc_favok": _finite_number(net_debt_favok),
        "provider_fk": None,
        "provider_pd_dd": None,
        "provider_fd_favok": None,
        "provider_fd_satis": None,
        "provider_period": None,
        "source_mtime_ns": source_mtime_ns,
        "as_of": str(payload.get("fetched_at") or payload.get("as_of") or "").strip() or None,
        "updated_at": _utc_now_iso(),
    }


def _normalized_snapshot_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    symbol = str(record.get("symbol") or "").strip().upper()
    if not symbol:
        return None
    normalized: Dict[str, Any] = {key: None for key in _SNAPSHOT_COLUMNS}
    normalized.update(record)
    normalized["symbol"] = symbol
    normalized["company_title"] = str(normalized.get("company_title") or "").strip() or None
    company_kind = str(normalized.get("company_kind") or "generic").strip().lower()
    normalized["company_kind"] = company_kind if company_kind in {"generic", "bank", "insurance"} else "generic"
    normalized["sector"] = str(normalized.get("sector") or "").strip() or None
    normalized["has_kap_cache"] = 1 if normalized.get("has_kap_cache") else 0
    try:
        normalized["quarter_count"] = max(0, int(normalized.get("quarter_count") or 0))
    except (TypeError, ValueError):
        normalized["quarter_count"] = 0
    for key in _SNAPSHOT_COLUMNS:
        if key in {
            "symbol", "company_title", "company_kind", "sector", "latest_quarter",
            "share_source", "provider_period", "as_of", "updated_at", "has_kap_cache", "quarter_count",
            "source_mtime_ns",
        }:
            continue
        normalized[key] = _finite_number(normalized.get(key))
    source_mtime_ns = normalized.get("source_mtime_ns")
    try:
        normalized["source_mtime_ns"] = int(source_mtime_ns) if source_mtime_ns is not None else None
    except (TypeError, ValueError):
        normalized["source_mtime_ns"] = None
    for key in ("latest_quarter", "share_source", "provider_period", "as_of"):
        normalized[key] = str(normalized.get(key) or "").strip() or None
    normalized["updated_at"] = str(normalized.get("updated_at") or _utc_now_iso())
    return normalized


def upsert_financial_snapshots(processed_dir: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Batch upsert KAP-derived fields while leaving provider fields intact."""

    normalized_records = [
        normalized
        for record in records
        if isinstance(record, dict)
        and (normalized := _normalized_snapshot_record(record)) is not None
    ]
    if not normalized_records:
        return 0
    columns = _SNAPSHOT_COLUMNS
    placeholders = ", ".join("?" for _ in columns)
    updates = [
        f"{column} = excluded.{column}"
        for column in columns
        if column not in {"symbol", "sector", "free_float_pct", "provider_fk", "provider_pd_dd", "provider_fd_favok", "provider_fd_satis", "provider_period"}
    ]
    updates[updates.index("source_mtime_ns = excluded.source_mtime_ns")] = (
        "source_mtime_ns = COALESCE(excluded.source_mtime_ns, company_financial_snapshots.source_mtime_ns)"
    )
    updates[updates.index("shares_outstanding = excluded.shares_outstanding")] = (
        "shares_outstanding = COALESCE(excluded.shares_outstanding, company_financial_snapshots.shares_outstanding)"
    )
    updates[updates.index("share_source = excluded.share_source")] = (
        "share_source = COALESCE(excluded.share_source, company_financial_snapshots.share_source)"
    )
    with _connection(processed_dir) as conn:
        conn.executemany(
            f"""
            INSERT INTO company_financial_snapshots ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(symbol) DO UPDATE SET {', '.join(updates)}
            """,
            [tuple(record.get(column) for column in columns) for record in normalized_records],
        )
        conn.commit()
    _invalidate_memory_cache(processed_dir)
    return len(normalized_records)


def sync_financial_snapshots_from_kap_cache(processed_dir: Path) -> Dict[str, int]:
    """Parse only new or changed KAP JSON files and persist their compact data."""

    ensure_financial_snapshot_schema(processed_dir)
    cache_dir = Path(processed_dir) / "kap_cache"
    if not cache_dir.is_dir():
        return {"scanned_count": 0, "upserted_count": 0, "skipped_unchanged_count": 0, "error_count": 0}

    with _connection(processed_dir) as conn:
        existing_rows = conn.execute(
            "SELECT symbol, source_mtime_ns FROM company_financial_snapshots WHERE has_kap_cache = 1"
        ).fetchall()
    existing_mtimes = {
        str(row.get("symbol") if isinstance(row, dict) else row["symbol"]).upper(): (
            row.get("source_mtime_ns") if isinstance(row, dict) else row["source_mtime_ns"]
        )
        for row in existing_rows
    }

    records: List[Dict[str, Any]] = []
    scanned_count = 0
    skipped_count = 0
    error_count = 0
    for cache_path in sorted(cache_dir.glob("*.json")):
        scanned_count += 1
        symbol = cache_path.stem.strip().upper()
        try:
            source_mtime_ns = cache_path.stat().st_mtime_ns
            if existing_mtimes.get(symbol) == source_mtime_ns:
                skipped_count += 1
                continue
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                error_count += 1
                continue
            record = extract_financial_snapshot_from_kap_payload(
                symbol,
                payload,
                source_mtime_ns=source_mtime_ns,
            )
            if record:
                records.append(record)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            error_count += 1
            LOGGER.warning("could not sync KAP financial snapshot for %s", symbol, exc_info=True)
        except Exception:
            error_count += 1
            LOGGER.exception("unexpected KAP financial snapshot sync error for %s", symbol)

    upserted_count = upsert_financial_snapshots(processed_dir, records)
    return {
        "scanned_count": scanned_count,
        "upserted_count": upserted_count,
        "skipped_unchanged_count": skipped_count,
        "error_count": error_count,
    }


def _normalize_database_row(row: Any) -> Dict[str, Any]:
    normalized = dict(row)
    normalized["symbol"] = str(normalized.get("symbol") or "").strip().upper()
    normalized["has_kap_cache"] = bool(normalized.get("has_kap_cache"))
    for key in _SNAPSHOT_COLUMNS:
        if key in {
            "symbol", "company_title", "company_kind", "sector", "latest_quarter",
            "has_kap_cache", "quarter_count", "share_source", "provider_period",
            "source_mtime_ns", "as_of", "updated_at",
        }:
            continue
        normalized[key] = _finite_number(normalized.get(key))
    return normalized


def get_financial_snapshots_map(
    processed_dir: Path,
    symbols: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Read the full compact snapshot table once and serve maps from memory."""

    requested = None if symbols is None else {
        str(symbol or "").strip().upper()
        for symbol in symbols
        if str(symbol or "").strip()
    }
    if requested == set():
        return {}

    key = _cache_key(processed_dir)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _SNAPSHOT_MEMORY_CACHE.get(key)
        if cached and cached[0] > now:
            full_map = cached[1]
        else:
            full_map = None

    if full_map is None:
        with _connection(processed_dir) as conn:
            rows = conn.execute(
                "SELECT * FROM company_financial_snapshots ORDER BY symbol"
            ).fetchall()
        full_map = {
            normalized["symbol"]: normalized
            for row in rows
            if (normalized := _normalize_database_row(row)).get("symbol")
        }
        with _CACHE_LOCK:
            _SNAPSHOT_MEMORY_CACHE[key] = (
                time.monotonic() + _SNAPSHOT_MEMORY_CACHE_TTL_SECONDS,
                full_map,
            )

    if requested is None:
        return {symbol: dict(record) for symbol, record in full_map.items()}
    return {
        symbol: dict(full_map[symbol])
        for symbol in requested
        if symbol in full_map
    }


def _provider_record(symbol: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    normalized_symbol = str(symbol or payload.get("symbol") or "").strip().upper()
    if not normalized_symbol:
        return None
    free_float = _finite_number(payload.get("free_float_pct"))
    if free_float is None:
        fdpo = _finite_number(payload.get("fdpo"))
        free_float = fdpo * 100.0 if fdpo is not None else None
    shares = _finite_number(payload.get("shares_outstanding"))
    return {
        "symbol": normalized_symbol,
        "shares_outstanding": shares if shares is not None and shares > 0 else None,
        "share_source": "isyatirim_temelozet" if shares is not None and shares > 0 else None,
        "sector": str(payload.get("sector") or "").strip() or None,
        "free_float_pct": free_float,
        "provider_fk": _finite_number(payload.get("provider_fk", payload.get("fk"))),
        "provider_pd_dd": _finite_number(payload.get("provider_pd_dd", payload.get("pd_dd"))),
        "provider_fd_favok": _finite_number(payload.get("provider_fd_favok", payload.get("fd_favok"))),
        "provider_fd_satis": _finite_number(payload.get("provider_fd_satis", payload.get("fd_satis"))),
        "provider_period": str(payload.get("provider_period") or "").strip() or None,
        "updated_at": _utc_now_iso(),
    }


def upsert_provider_fundamentals_bulk(
    processed_dir: Path,
    provider_map: Dict[str, Dict[str, Any]],
) -> int:
    """Batch store provider sectors, free float and backup valuation ratios."""

    if not isinstance(provider_map, dict) or not provider_map:
        return 0
    records = [
        record
        for symbol, payload in provider_map.items()
        if isinstance(payload, dict)
        and (record := _provider_record(str(symbol), payload)) is not None
    ]
    if not records:
        return 0
    with _connection(processed_dir) as conn:
        conn.executemany(
            """
            INSERT INTO company_financial_snapshots (
                symbol, company_kind, quarter_count, has_kap_cache,
                shares_outstanding, share_source, sector, free_float_pct,
                provider_fk, provider_pd_dd, provider_fd_favok, provider_fd_satis,
                provider_period, updated_at
            )
            VALUES (?, 'generic', 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                shares_outstanding = CASE
                    WHEN company_financial_snapshots.share_source IN ('odenmis_sermaye', 'cikarilmis_sermaye')
                    THEN company_financial_snapshots.shares_outstanding
                    ELSE excluded.shares_outstanding
                END,
                share_source = CASE
                    WHEN company_financial_snapshots.share_source IN ('odenmis_sermaye', 'cikarilmis_sermaye')
                    THEN company_financial_snapshots.share_source
                    ELSE excluded.share_source
                END,
                sector = excluded.sector,
                free_float_pct = excluded.free_float_pct,
                provider_fk = excluded.provider_fk,
                provider_pd_dd = excluded.provider_pd_dd,
                provider_fd_favok = excluded.provider_fd_favok,
                provider_fd_satis = excluded.provider_fd_satis,
                provider_period = excluded.provider_period,
                updated_at = excluded.updated_at
            """,
            [
                (
                    record["symbol"],
                    record["shares_outstanding"],
                    record["share_source"],
                    record["sector"],
                    record["free_float_pct"],
                    record["provider_fk"],
                    record["provider_pd_dd"],
                    record["provider_fd_favok"],
                    record["provider_fd_satis"],
                    record["provider_period"],
                    record["updated_at"],
                )
                for record in records
            ],
        )
        conn.commit()
    _invalidate_memory_cache(processed_dir)
    return len(records)


def reset_financial_snapshot_store_for_tests() -> None:
    """Reset process-local schema and row caches used by isolated tests."""

    global _POSTGRES_SCHEMA_READY
    _POSTGRES_SCHEMA_READY = False
    _LOCAL_SCHEMA_READY.clear()
    with _CACHE_LOCK:
        _SNAPSHOT_MEMORY_CACHE.clear()
