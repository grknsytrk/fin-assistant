"""Fund catalogue snapshot persistence, refresh, filtering, and daily price collection."""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from app.fund_service import FundSnapshotFetchResult

_UNSET: Any = object()
_ORIGINAL_EXPORTS: Dict[str, Any] = {}


def _fund_service() -> Any:
    """Resolve ``app.fund_service`` lazily to avoid import cycles and honor monkeypatches."""
    mod = sys.modules.get("app.fund_service")
    if mod is not None:
        return mod
    import app.fund_service as fund_service

    return fund_service


def _seam(name: str) -> Any:
    """Resolve a moved symbol via ``app.fund_service`` unless patched locally on this module."""
    local_obj = globals()[name]
    orig_obj = _ORIGINAL_EXPORTS.get(name, local_obj)
    if local_obj is not orig_obj:
        return local_obj
    return getattr(_fund_service(), name, local_obj)


def _encode_fund_snapshot_present_fields(row: Dict[str, Any]) -> str:
    """Store the absent snapshot fields instead of the full key list.

    A full key list is dominated by the sixteen well-known snapshot fields and
    costs roughly 350 bytes per catalogue row. Nearly every row carries all of
    them, so an explicit ``{"absent": [...]}`` payload is far smaller. Legacy
    databases still hold a plain list, which the reader keeps accepting.
    """

    present = {str(key) for key in row}
    absent = [field for field in _fund_service()._FUND_SNAPSHOT_FIELDS if field not in present]
    if not absent:
        return "{}"
    return _fund_service()._stable_json_dumps({"absent": absent})


def _decode_fund_snapshot_present_fields(raw: Any) -> set[str]:
    parse = _fund_service()._safe_json_loads(raw, None)
    if isinstance(parse, list):
        return {str(field) for field in parse}
    if isinstance(parse, dict):
        absent = parse.get("absent")
        absent_fields = {str(field) for field in absent} if isinstance(absent, list) else set()
        return set(_fund_service()._FUND_SNAPSHOT_FIELDS) - absent_fields
    return set()


def _fund_snapshot_record(row: Dict[str, Any], row_order: int, generation: int) -> Optional[Tuple[Any, ...]]:
    code = _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
    if not code:
        return None
    present_fields = _seam("_encode_fund_snapshot_present_fields")(row)
    attributes = {key: value for key, value in row.items() if key not in _fund_service()._FUND_SNAPSHOT_FIELDS}
    period_returns = row.get("period_returns")
    risk_value = row.get("risk_value")
    risk_number = _fund_service()._coerce_float(risk_value)
    try:
        tefas_open = _seam("_tefas_open_status")(row)
    except NameError:  # schema backfill can run during early module initialization
        tefas_open = None
    search_text = " ".join(
        str(row.get(key) or "")
        for key in ("fund_code", "name", "fund_type", "founder_company", "manager_company")
    ).lower()
    normalized_fields = (
        str(row.get("fund_type") or "").strip().lower(),
        str(row.get("founder_company") or "").strip().lower(),
        str(row.get("manager_company") or "").strip().lower(),
        str(risk_value or "").strip().lower(),
    )
    stable_row = _fund_service()._stable_json_dumps(row)
    return (
        code,
        row.get("name"),
        row.get("fund_type"),
        row.get("founder_company"),
        row.get("manager_company"),
        _fund_service()._coerce_float(row.get("price")),
        _fund_service()._coerce_float(row.get("daily_return")),
        _seam("_fund_row_aum")(row),
        _fund_service()._coerce_int(row.get("investor_count")),
        _fund_service()._coerce_float(row.get("share_count")),
        None if risk_value is None else str(risk_value),
        None if risk_value is None else _fund_service()._stable_json_dumps(risk_value),
        risk_number,
        None if tefas_open is None else int(tefas_open),
        row.get("currency"),
        _fund_service()._fund_date(row.get("as_of")),
        row.get("source"),
        _fund_service()._stable_json_dumps(period_returns),
        _fund_service()._stable_json_dumps(attributes),
        present_fields,
        search_text,
        *normalized_fields,
        str(code).lower(),
        str(row.get("name") or "").lower(),
        str(row.get("fund_type") or "").lower(),
        str(row.get("founder_company") or "").lower(),
        str(row.get("manager_company") or "").lower(),
        int(row_order),
        int(generation),
        hashlib.sha256(f"{_fund_service()._FUND_SNAPSHOT_CONTENT_HASH_VERSION}\n{stable_row}".encode("utf-8")).hexdigest(),
    )


def _fund_snapshot_insert_sql() -> str:
    columns = (
        "fund_code, name, fund_type, founder_company, manager_company, price, daily_return, aum, "
        "investor_count, share_count, risk_value, risk_value_json, risk_value_num, tefas_open, "
        "currency, as_of, source, period_returns_json, attributes_json, present_fields_json, "
        "search_text, fund_type_key, founder_key, manager_key, risk_key, fund_code_sort, name_sort, "
        "fund_type_sort, founder_sort, manager_sort, row_order, generation, content_hash"
    )
    values = ", ".join("?" for _ in range(33))
    updates = ", ".join(
        f"{column} = excluded.{column}"
        for column in columns.split(", ")
        if column != "fund_code"
    )
    return (
        f"INSERT INTO fund_snapshots ({columns}) VALUES ({values}) "
        f"ON CONFLICT (fund_code) DO UPDATE SET {updates} "
        "WHERE fund_snapshots.content_hash <> excluded.content_hash"
    )


def _fund_snapshot_records(rows: Iterable[Any], generation: int) -> List[Tuple[Any, ...]]:
    records_by_code: Dict[str, Tuple[Any, ...]] = {}
    for row_order, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        record = _seam("_fund_snapshot_record")(row, row_order, generation)
        if record is None:
            continue
        # Keep the latest occurrence to match the existing reference-data
        # batch upsert behavior when a legacy payload contains duplicate codes.
        records_by_code[str(record[0])] = record
    return sorted(records_by_code.values(), key=lambda record: int(record[30]))


def _fund_snapshot_summary(payload: Dict[str, Any], generation: int) -> Dict[str, Any]:
    summary = {key: value for key, value in payload.items() if key != "rows"}
    count = len(
        {
            _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
            for row in list(payload.get("rows") or [])
            if isinstance(row, dict)
            and _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        }
    )
    summary["count"] = count
    summary["total_count"] = count
    summary["snapshot_generation"] = int(generation)
    metadata = dict(summary.get("source_metadata") or {})
    metadata["snapshot_generation"] = int(generation)
    metadata["catalog_storage"] = "fund_snapshots"
    summary["source_metadata"] = metadata
    return summary


def _stage_fund_snapshot_codes(conn: Any, records: List[Tuple[Any, ...]]) -> None:
    """Track the complete incoming set while allowing unchanged rows to stay put."""

    conn.execute(
        "CREATE TEMP TABLE fund_snapshot_commit_codes (fund_code TEXT PRIMARY KEY)"
    )
    conn.executemany(
        "INSERT INTO fund_snapshot_commit_codes (fund_code) VALUES (?)",
        [(str(record[0]),) for record in records],
    )


def _delete_fund_snapshot_rows_missing_from_stage(conn: Any) -> None:
    conn.execute(
        """
        DELETE FROM fund_snapshots
        WHERE NOT EXISTS (
            SELECT 1 FROM fund_snapshot_commit_codes staged
            WHERE staged.fund_code = fund_snapshots.fund_code
        )
        """
    )
    conn.execute("DROP TABLE fund_snapshot_commit_codes")


def _snapshot_row_from_record(record: Any) -> Dict[str, Any]:
    present = _seam("_decode_fund_snapshot_present_fields")(record["present_fields_json"])
    attributes = _fund_service()._safe_json_loads(record["attributes_json"], {})
    row = dict(attributes) if isinstance(attributes, dict) else {}
    for field in _fund_service()._FUND_SNAPSHOT_FIELDS:
        if field not in present:
            continue
        if field == "period_returns":
            value = _fund_service()._safe_json_loads(record["period_returns_json"], {})
        elif field == "risk_value":
            value = _fund_service()._safe_json_loads(record["risk_value_json"], None)
        else:
            value = record[field]
        row[field] = value
    if "fund_code" not in row:
        row["fund_code"] = record["fund_code"]
    return row


def _read_fund_snapshot_rows(
    processed_dir: Path,
    *,
    where: str = "",
    params: Iterable[Any] = (),
    order_by: str = "row_order ASC",
) -> List[Dict[str, Any]]:
    use_all_rows_cache = (
        not where
        and not tuple(params)
        and order_by == "row_order ASC"
        and _fund_service().database_enabled()
    )
    all_rows_cache_key = ""
    if use_all_rows_cache:
        catalog_gen = _fund_service().fund_listing_revision(processed_dir).get("catalog_generation", 0)
        all_rows_cache_key = f"fund-snapshot-all-rows:{_fund_service()._fund_prices_db_path(processed_dir)}:{catalog_gen}"
        cached = _fund_service()._peek_all_snapshot_rows(processed_dir, catalog_gen)
        if cached is not None:
            return [dict(row) for row in cached["rows"]]

    columns = ", ".join(_fund_service()._FUND_SNAPSHOT_READ_COLUMNS)
    query = f"SELECT {columns} FROM fund_snapshots{(' WHERE ' + where) if where else ''} ORDER BY {order_by}"
    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        records = conn.execute(query, tuple(params)).fetchall()
    rows = [_seam("_snapshot_row_from_record")(record) for record in records]
    if use_all_rows_cache and all_rows_cache_key:
        catalog_dates = {
            str(row.get("fund_code") or ""): _fund_service()._fund_date(row.get("as_of"))
            for row in rows
            if row.get("fund_code")
        }
        entry = {
            "cached_at": time.time(),
            "rows": [dict(row) for row in rows],
            "catalog_dates": catalog_dates,
        }
        if len(_fund_service()._SNAPSHOT_ALL_ROWS_CACHE) >= 2:
            oldest_key = next(iter(_fund_service()._SNAPSHOT_ALL_ROWS_CACHE))
            _fund_service()._SNAPSHOT_ALL_ROWS_CACHE.pop(oldest_key, None)
        _fund_service()._SNAPSHOT_ALL_ROWS_CACHE[all_rows_cache_key] = entry
    return rows


def _read_fund_snapshot_category_returns(
    processed_dir: Path,
    *,
    fund_type: str,
    min_aum: Optional[float],
) -> List[Dict[str, Any]]:
    normalized_type = fund_type.strip().lower()
    cache_key = (
        f"snapshot-category-returns:{_fund_service()._fund_prices_db_path(processed_dir)}:"
        f"{normalized_type}:{int(bool(_fund_service().TEFAS_OPEN_ONLY))}:{min_aum}"
    )
    cached = _fund_service()._MEMORY_CACHE.get(cache_key)
    if (
        isinstance(cached, dict)
        and isinstance(cached.get("cached_at"), (int, float))
        and time.time() - float(cached["cached_at"]) <= 300.0
    ):
        return [
            {"period_returns": dict(item.get("period_returns") or {})}
            for item in list(cached.get("rows") or [])
            if isinstance(item, dict)
        ]

    cached_all = _fund_service()._peek_all_snapshot_rows(processed_dir)
    if cached_all is not None:
        rows = [
            {"period_returns": dict(item.get("period_returns") or {})}
            for item in cached_all["rows"]
            if isinstance(item, dict)
            and str(item.get("fund_type") or "").strip().lower() == normalized_type
            and _seam("_is_tefas_open_row")(item)
            and _seam("_meets_min_aum")(item, min_aum)
        ]
        _fund_service()._memory_cache_put(cache_key, {"cached_at": time.time(), "rows": rows})
        return [
            {"period_returns": dict(item.get("period_returns") or {})}
            for item in rows
        ]

    conditions = ["fund_type_key = ?"]
    params: List[Any] = [normalized_type]
    if _fund_service().TEFAS_OPEN_ONLY:
        conditions.append("(tefas_open IS NULL OR tefas_open = 1)")
    if min_aum is not None and min_aum > 0:
        conditions.append("aum >= ?")
        params.append(float(min_aum))
    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        records = conn.execute(
            "SELECT period_returns_json FROM fund_snapshots WHERE "
            + " AND ".join(conditions)
            + " ORDER BY row_order ASC",
            tuple(params),
        ).fetchall()
    rows = [
        {"period_returns": _fund_service()._safe_json_loads(record["period_returns_json"], {})}
        for record in records
    ]
    _fund_service()._memory_cache_put(cache_key, {"cached_at": time.time(), "rows": rows})
    return [
        {"period_returns": dict(item.get("period_returns") or {})}
        for item in rows
    ]


def _copy_legacy_snapshot(processed_dir: Path, payload: Dict[str, Any]) -> None:
    if not list(payload.get("rows") or []):
        return
    legacy = _fund_service()._legacy_snapshot_path(processed_dir)
    if legacy.exists():
        return
    _fund_service()._write_local_json(legacy, payload)


def _snapshot_metadata_from_store(processed_dir: Path) -> Optional[Dict[str, Any]]:
    if _fund_service().database_enabled():
        _revision, summary = _fund_service().fund_listing_state(processed_dir)
        metadata = summary.get("source_metadata") if isinstance(summary, dict) else None
        if isinstance(metadata, dict) and metadata.get("catalog_storage") == "fund_snapshots":
            return summary

        # One-time backfill for databases that already have row snapshots but
        # predate fund_listing_state. The existing JSON summary is compact after
        # the catalogue migration and becomes unnecessary on later requests.
        summary = _fund_service()._read_json(_fund_service()._snapshot_path(processed_dir))
        metadata = summary.get("source_metadata") if isinstance(summary, dict) else None
        if summary is not None and isinstance(metadata, dict) and metadata.get("catalog_storage") == "fund_snapshots":
            with _fund_service().connect_postgres() as conn:
                conn.execute(
                    "UPDATE fund_listing_state SET metadata_json = ?, catalog_generation = GREATEST(catalog_generation, ?), updated_at = NOW() WHERE id = 1 AND metadata_json = '{}'",
                    (_fund_service()._stable_json_dumps(summary), int(summary.get("snapshot_generation") or 0)),
                )
                conn.commit()
            _fund_service().invalidate_fund_listing_state_cache()
            return summary
        return None
    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        state = conn.execute("SELECT metadata_json FROM fund_snapshot_state WHERE id = 1").fetchone()
    if state is None:
        return None
    raw = state["metadata_json"] if not isinstance(state, dict) else state.get("metadata_json")
    summary = _fund_service()._safe_json_loads(raw, None)
    return summary if isinstance(summary, dict) else None


def ensure_fund_snapshot_store(processed_dir: Path) -> Dict[str, Any]:
    """Idempotently migrate the legacy JSON catalogue into the row store."""

    with _fund_service()._FUND_SNAPSHOT_MIGRATION_LOCK:
        _fund_service().ensure_fund_prices_schema(processed_dir)
        summary = _seam("_snapshot_metadata_from_store")(processed_dir)
        if summary is not None:
            if _fund_service().database_enabled():
                return summary
            # Local tools and legacy tests may replace the JSON file directly.
            # Treat a full document as a new import while the normal on-disk
            # representation is the compact metadata summary.
            disk_payload = _fund_service()._read_json(_fund_service()._snapshot_path(processed_dir)) or {}
            if not isinstance(disk_payload.get("rows"), list):
                return summary

        legacy = _fund_service()._read_json(_fund_service()._snapshot_path(processed_dir)) or _fund_service()._read_json(_fund_service()._legacy_snapshot_path(processed_dir)) or {}
        rows = [row for row in list(legacy.get("rows") or []) if isinstance(row, dict)]
        if not rows:
            return legacy

        if _fund_service().database_enabled():
            from app.database import _cache_key

            path = _fund_service()._snapshot_path(processed_dir)
            cache_key = _cache_key(path)
            legacy_key = _cache_key(_fund_service()._legacy_snapshot_path(processed_dir))
            with _fund_service().connect_postgres() as conn:
                # Serialize concurrent first-start migrations across API replicas.
                conn.execute("SELECT pg_advisory_xact_lock(hashtext('fund_snapshots_migration_v1'))")
                # Coordinate with refresh commits, which lock the same lease row.
                conn.execute(
                    "SELECT resource_key FROM ragfin_refresh_lease WHERE resource_key = ? FOR UPDATE",
                    ("funds_snapshot",),
                ).fetchone()
                count_row = conn.execute("SELECT COUNT(*) AS row_count FROM fund_snapshots").fetchone()
                count = int(count_row.get("row_count") or 0) if isinstance(count_row, dict) else 0
                cache_row = conn.execute(
                    "SELECT payload, version FROM ragfin_json_cache WHERE cache_key = ? FOR UPDATE",
                    (cache_key,),
                ).fetchone()
                current_payload = cache_row.get("payload") if isinstance(cache_row, dict) else None
                current_meta = current_payload.get("source_metadata") if isinstance(current_payload, dict) else None
                if (
                    count
                    and isinstance(current_meta, dict)
                    and current_meta.get("catalog_storage") == "fund_snapshots"
                ):
                    conn.rollback()
                    _fund_service()._clear_all_fund_memory_caches()
                    return dict(current_payload)
                cache_version = int(cache_row.get("version") or 0) if isinstance(cache_row, dict) else 0
                if count:
                    conn.execute("DELETE FROM fund_snapshots")
                max_generation_row = conn.execute(
                    "SELECT COALESCE(MAX(generation), 0) AS generation FROM fund_snapshots"
                ).fetchone()
                max_generation = (
                    int(max_generation_row.get("generation") or 0)
                    if isinstance(max_generation_row, dict)
                    else 0
                )
                generation = max(
                    1,
                    cache_version + 1,
                    max_generation + 1,
                    int(legacy.get("snapshot_generation") or 0),
                )
                summary_payload = _seam("_fund_snapshot_summary")(legacy, generation)
                conn.execute(
                    """
                    INSERT INTO ragfin_json_cache (cache_key, payload, version, updated_at)
                    SELECT ?, payload, version, NOW() FROM ragfin_json_cache WHERE cache_key = ?
                    ON CONFLICT (cache_key) DO NOTHING
                    """,
                    (legacy_key, cache_key),
                )
                conn.executemany(_seam("_fund_snapshot_insert_sql")(), _seam("_fund_snapshot_records")(rows, generation))
                conn.execute(
                    """
                    INSERT INTO ragfin_json_cache (cache_key, payload, version, updated_at)
                    VALUES (?, ?::jsonb, ?, NOW())
                    ON CONFLICT (cache_key) DO UPDATE SET
                        payload = EXCLUDED.payload,
                        version = EXCLUDED.version,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (cache_key, json.dumps(summary_payload, ensure_ascii=False, default=str), generation),
                )
                _fund_service()._set_fund_listing_catalog_revision(conn, generation, summary_payload)
                conn.commit()
            _seam("_copy_legacy_snapshot")(processed_dir, legacy)
            _fund_service()._write_local_json(path, summary_payload)
        else:
            _seam("_copy_legacy_snapshot")(processed_dir, legacy)
            with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
                conn.execute("BEGIN IMMEDIATE")
                state = conn.execute("SELECT generation FROM fund_snapshot_state WHERE id = 1").fetchone()
                current_generation = int(state["generation"]) if state is not None else 0
                generation = max(1, current_generation + 1, int(legacy.get("snapshot_generation") or 0))
                summary_payload = _seam("_fund_snapshot_summary")(legacy, generation)
                records = _seam("_fund_snapshot_records")(rows, generation)
                _seam("_stage_fund_snapshot_codes")(conn, records)
                conn.executemany(_seam("_fund_snapshot_insert_sql")(), records)
                _seam("_delete_fund_snapshot_rows_missing_from_stage")(conn)
                conn.execute(
                    """
                    INSERT INTO fund_snapshot_state (id, generation, row_count, metadata_json, updated_at)
                    VALUES (1, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        generation = excluded.generation,
                        row_count = excluded.row_count,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        generation,
                        len(records),
                        _fund_service()._stable_json_dumps(summary_payload),
                        _fund_service()._utc_now_iso(),
                    ),
                )
                _fund_service()._set_fund_listing_catalog_revision(conn, generation, summary_payload)
            _fund_service()._write_local_json(_fund_service()._snapshot_path(processed_dir), summary_payload)
        _fund_service()._clear_all_fund_memory_caches()
        _fund_service().logging.getLogger("uvicorn.error").info(
            "fund_catalog migration rows=%d generation=%d", len(rows), generation
        )
        return summary_payload


def prune_fund_snapshot_rollback_copy(processed_dir: Path) -> bool:
    """Replace the one-time full JSON rollback copy after its retention window."""

    if _fund_service().FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS <= 0:
        return False
    path = _fund_service()._legacy_snapshot_path(processed_dir)
    if _fund_service().database_enabled():
        from app.database import _cache_key

        try:
            with _fund_service().connect_postgres() as conn:
                record = conn.execute(
                    """
                    SELECT
                        updated_at,
                        CASE
                            WHEN jsonb_typeof(payload->'rows') = 'array'
                            THEN jsonb_array_length(payload->'rows')
                            ELSE 0
                        END AS row_count,
                        COALESCE((payload->>'rollback_copy_expired')::boolean, FALSE) AS expired
                    FROM ragfin_json_cache
                    WHERE cache_key = ?
                    """,
                    (_cache_key(path),),
                ).fetchone()
            if not isinstance(record, dict):
                return False
            if "expired" in record or "row_count" in record:
                if bool(record.get("expired")):
                    return False
                age = _fund_service()._cache_age_seconds(record.get("updated_at"))
                if age is None or age < _fund_service().FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS:
                    return False
                row_count = max(0, int(record.get("row_count") or 0))
            else:
                payload = record.get("payload")
                if not isinstance(payload, dict) or payload.get("rollback_copy_expired"):
                    return False
                age = _fund_service()._cache_age_seconds(record.get("updated_at"))
                if age is None or age < _fund_service().FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS:
                    return False
                row_count = len(list(payload.get("rows") or []))
        except Exception:
            _fund_service().LOGGER.warning("fund snapshot rollback retention check failed", exc_info=True)
            return False
    else:
        if not path.exists():
            return False
        try:
            payload = _fund_service()._read_json(path) or {}
            if payload.get("rollback_copy_expired"):
                return False
            age = max(0.0, time.time() - path.stat().st_mtime)
            if age < _fund_service().FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS:
                return False
            row_count = len(list(payload.get("rows") or []))
        except Exception:
            _fund_service().LOGGER.warning("local fund snapshot rollback retention check failed", exc_info=True)
            return False

    compact = {
        "status": "rollback_copy_expired",
        "row_count": row_count,
        "rollback_copy_expired": True,
        "expired_at": _fund_service()._utc_now_iso(),
        "source_metadata": {"catalog_storage": "fund_snapshots"},
    }
    if _fund_service().database_enabled():
        _fund_service().write_json_cache(path, compact)
    _fund_service()._write_local_json(path, compact)
    _fund_service().LOGGER.info("fund_catalog legacy rollback copy compacted rows=%d", row_count)
    return True


def _commit_local_fund_snapshot_rows(
    processed_dir: Path,
    rows: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    with _fund_service()._FUND_SNAPSHOT_COMMIT_LOCK:
        with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
            conn.execute("BEGIN IMMEDIATE")
            state = conn.execute(
                "SELECT generation FROM fund_snapshot_state WHERE id = 1"
            ).fetchone()
            current_generation = int(state["generation"]) if state is not None else 0
            generation = max(current_generation + 1, 1)
            records = _seam("_fund_snapshot_records")(rows, generation)
            summary = _seam("_fund_snapshot_summary")(payload, generation)
            _seam("_stage_fund_snapshot_codes")(conn, records)
            conn.executemany(_seam("_fund_snapshot_insert_sql")(), records)
            _seam("_delete_fund_snapshot_rows_missing_from_stage")(conn)
            conn.execute(
                """
                INSERT INTO fund_snapshot_state (id, generation, row_count, metadata_json, updated_at)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    generation = excluded.generation,
                    row_count = excluded.row_count,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (generation, len(records), _fund_service()._stable_json_dumps(summary), _fund_service()._utc_now_iso()),
            )
            _fund_service()._set_fund_listing_catalog_revision(conn, generation, summary)
        _fund_service()._write_local_json(_fund_service()._snapshot_path(processed_dir), summary)
        return summary


def commit_funds_snapshot(
    processed_dir: Path,
    payload: Dict[str, Any],
    *,
    job_id: str,
    owner_token: str,
    generation: int,
) -> bool:
    """Publish the entire catalogue and its compact metadata atomically."""

    path = _fund_service()._snapshot_path(processed_dir)
    rows = [row for row in list(payload.get("rows") or []) if isinstance(row, dict)]
    records = _seam("_fund_snapshot_records")(rows, generation)
    has_invalid_codes = any(
        not _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        for row in rows
    )
    if not records or has_invalid_codes:
        _fund_service().LOGGER.warning(
            "fund snapshot commit rejected: invalid or empty catalogue rows=%d valid=%d",
            len(rows),
            len(records),
        )
        return False
    _fund_service().ensure_fund_prices_schema(processed_dir)
    previous = _fund_service()._read_json(path) or {}
    _seam("_copy_legacy_snapshot")(processed_dir, previous)

    if _fund_service().database_enabled():
        from app.database import _cache_key, ensure_refresh_lease_schema

        ensure_refresh_lease_schema()
        cache_key = _cache_key(path)
        legacy_key = _cache_key(_fund_service()._legacy_snapshot_path(processed_dir))
        summary = _seam("_fund_snapshot_summary")(payload, generation)
        try:
            with _fund_service().connect_postgres() as conn:
                lease = conn.execute(
                    """
                    SELECT generation, job_id, owner_token, lease_until
                    FROM ragfin_refresh_lease
                    WHERE resource_key = ? AND lease_until > NOW()
                    FOR UPDATE
                    """,
                    ("funds_snapshot",),
                ).fetchone()
                if not isinstance(lease, dict) or (
                    int(lease.get("generation") or -1) != int(generation)
                    or str(lease.get("job_id") or "") != job_id
                    or str(lease.get("owner_token") or "") != owner_token
                    or lease.get("lease_until") is None
                ):
                    conn.rollback()
                    return False

                # Preserve the original monolithic document once for a controlled
                # rollback window; later generations write only compact metadata.
                conn.execute(
                    """
                    INSERT INTO ragfin_json_cache (cache_key, payload, version, updated_at)
                    SELECT ?, payload, version, NOW()
                    FROM ragfin_json_cache
                    WHERE cache_key = ? AND jsonb_typeof(payload->'rows') = 'array'
                    ON CONFLICT (cache_key) DO NOTHING
                    """,
                    (legacy_key, cache_key),
                )
                _seam("_stage_fund_snapshot_codes")(conn, records)
                conn.executemany(_seam("_fund_snapshot_insert_sql")(), records)
                _seam("_delete_fund_snapshot_rows_missing_from_stage")(conn)
                meta_row = conn.execute(
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
                    (cache_key, json.dumps(summary, ensure_ascii=False, default=str), int(generation)),
                ).fetchone()
                if not isinstance(meta_row, dict):
                    conn.rollback()
                    return False
                _fund_service()._set_fund_listing_catalog_revision(conn, generation, summary)
                conn.commit()
        except Exception:
            _fund_service().LOGGER.warning("fund snapshot row commit failed", exc_info=True)
            return False
    else:
        try:
            summary = _seam("_commit_local_fund_snapshot_rows")(processed_dir, rows, payload)
        except Exception:
            _fund_service().LOGGER.warning("local fund snapshot row commit failed", exc_info=True)
            return False

    # Mirror only after the durable rows and metadata have committed.
    if _fund_service().database_enabled():
        try:
            _fund_service()._write_local_json(path, summary)
        except Exception:
            _fund_service().LOGGER.warning("fund snapshot local metadata mirror failed", exc_info=True)
    _fund_service()._clear_all_fund_memory_caches()
    return True


def _daily_snapshot_cache_is_fresh(payload: Dict[str, Any], as_of: date) -> bool:
    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    cache_version = _fund_service()._coerce_int(payload.get("cache_version") or metadata.get("cache_version"))
    if cache_version != _fund_service().FUNDS_DAILY_SNAPSHOT_CACHE_VERSION:
        return False
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False
    if not rows:
        fetched_at = payload.get("fetched_at") or (payload.get("source_metadata") or {}).get("fetched_at")
        age = _fund_service()._cache_age_seconds(fetched_at)
        return age is not None and age <= max(1, _fund_service().FUNDS_ALLOCATION_EMPTY_TTL_SECONDS)
    target_date = _fund_service()._latest_fund_snapshot_target_date()
    if as_of < target_date:
        return True
    fetched_at = payload.get("fetched_at") or (payload.get("source_metadata") or {}).get("fetched_at")
    age = _fund_service()._cache_age_seconds(fetched_at)
    return age is not None and age <= max(1, _fund_service().FUNDS_SNAPSHOT_INTRADAY_CHECK_TTL_SECONDS)


def _cached_daily_funds_snapshot(
    processed_dir: Path,
    client: "TefasFonClient",
    as_of: date,
    *,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    path = _fund_service()._fund_daily_snapshot_path(processed_dir, as_of)
    # In Postgres mode the snapshot rows live in `fund_prices` and `fund_snapshots`;
    # keep multi-MB daily snapshot blobs on local disk only to avoid egress churn.
    cached = (_fund_service()._read_local_json(path) if _fund_service().database_enabled() else _fund_service()._read_json(path)) or {}
    if not force_refresh and cached and _seam("_daily_snapshot_cache_is_fresh")(cached, as_of):
        rows = [row for row in list(cached.get("rows") or []) if isinstance(row, dict)]
        return rows, True

    fetched_at = _fund_service()._utc_now_iso()
    rows = client.fetch_daily_funds_snapshot(as_of)
    payload = {
        "status": "ok" if rows else "empty",
        "cache_version": _fund_service().FUNDS_DAILY_SNAPSHOT_CACHE_VERSION,
        "as_of": as_of.isoformat(),
        "fetched_at": fetched_at,
        "source": _fund_service().TEFASFON_FUNDS_SOURCE,
        "rows": rows,
        "source_metadata": {
            "source": _fund_service().TEFASFON_FUNDS_SOURCE,
            "as_of": as_of.isoformat(),
            "fetched_at": fetched_at,
            "cache_policy": "daily_snapshot",
            "cache_version": _fund_service().FUNDS_DAILY_SNAPSHOT_CACHE_VERSION,
            "row_count": len(rows),
        },
    }
    if _fund_service().database_enabled():
        _fund_service()._write_local_json(path, payload)
    else:
        _fund_service()._write_json(path, payload)
    if rows:
        _fund_service().upsert_fund_price_points(processed_dir, rows, source=_fund_service().TEFASFON_FUNDS_SOURCE, fetched_at=fetched_at)
        _fund_service()._upsert_fund_reference_data(processed_dir, rows)
    return rows, False


def _tefas_open_status(row: Dict[str, Any]) -> Optional[bool]:
    raw = _fund_service()._first_present(
        row,
        "tefas_open",
        "tefasDurum",
        "TEFASDURUM",
        "tefas_durum",
        "tefasOpen",
        "is_tefas_open",
    )
    if raw is None and isinstance(row.get("raw"), dict):
        raw = _fund_service()._first_present(
            row["raw"],
            "tefas_open",
            "tefasDurum",
            "TEFASDURUM",
            "tefas_durum",
            "tefasOpen",
            "is_tefas_open",
        )
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    text = _fund_service()._normalize_match_text(raw)
    if text in {"TRUE", "1", "EVET", "E", "ACIK", "OPEN"}:
        return True
    if text in {"FALSE", "0", "HAYIR", "H", "KAPALI", "CLOSED"}:
        return False
    return None


def _is_tefas_open_row(row: Dict[str, Any], *, require_known: bool = False) -> bool:
    if not _fund_service().TEFAS_OPEN_ONLY:
        return True
    status = _seam("_tefas_open_status")(row)
    if status is None:
        return not require_known
    return status is True


def _filter_tefas_open_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    materialized = [row for row in rows if isinstance(row, dict)]
    if not _fund_service().TEFAS_OPEN_ONLY:
        return materialized, 0, 0
    filtered: List[Dict[str, Any]] = []
    skipped_closed = 0
    skipped_unknown = 0
    for row in materialized:
        status = _seam("_tefas_open_status")(row)
        if status is False:
            skipped_closed += 1
            continue
        if status is None:
            # The TEFAS adapter does not always emit a tefas_open flag; treat unknown
            # rows as open instead of dropping them so the snapshot stays complete.
            skipped_unknown += 1
        filtered.append(row)
    return filtered, skipped_closed, skipped_unknown


def _founder_match_key(founder_title: str) -> str:
    text = _fund_service()._normalize_match_text(founder_title)
    suffixes = (
        " YÖNETİMİ ANONİM ŞİRKETİ",
        " YÖNETİMİ A.Ş.",
        " YÖNETİMİ A.Ş",
        " YONETIMI ANONIM SIRKETI",
        " YONETIMI A.S.",
        " YONETIMI A.S",
        " ANONİM ŞİRKETİ",
        " ANONIM SIRKETI",
        " A.Ş.",
        " A.Ş",
        " A.S.",
        " A.S",
    )
    for suffix in suffixes:
        normalized_suffix = _fund_service()._normalize_match_text(suffix)
        if text.endswith(normalized_suffix):
            return text[: -len(normalized_suffix)].strip()
    return text


def _target_manager_tokens() -> Tuple[str, ...]:
    tokens: List[str] = []
    for item in _fund_service().TARGET_FUND_MANAGER_KEYWORDS:
        token = _fund_service()._normalize_match_text(item)
        if token:
            tokens.append(token)
    return tuple(tokens)


def _is_target_fund_row(row: Dict[str, Any]) -> bool:
    if not _fund_service().TARGET_FUND_MANAGER_KEYWORDS:
        return True
    haystack = " ".join(
        _fund_service()._normalize_match_text(_fund_service()._first_text(row, *keys))
        for keys in (
            ("founder_company", "KURUCU", "KURUCUNVAN", "KURUCUUNVAN", "kurucuUnvan", "kurucuKodu", "kurucuKod"),
            ("manager_company", "YONETICI", "YONETICIUNVAN", "PORTFOYYONETICISI", "yoneticiUnvan"),
            ("name", "FONUNVAN", "FONUNVANI", "FONADI", "fonUnvan"),
        )
    )
    return any(re.search(rf"\b{re.escape(token)}\b", haystack) for token in _seam("_target_manager_tokens")())


def _filter_target_fund_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict) and _seam("_is_target_fund_row")(row) and _seam("_is_tefas_open_row")(row)]


def _fund_row_aum(row: Dict[str, Any]) -> Optional[float]:
    return _fund_service()._coerce_float(
        _fund_service()._first_present(
            row,
            "aum",
            "portfoyBuyukluk",
            "portBuyukluk",
            "sonPortfoyDegeri",
            "PORTFOYBUYUKLUK",
            "PORTFOY_BUYUKLUK",
        )
    )


def _meets_min_aum(row: Dict[str, Any], min_aum: Optional[float]) -> bool:
    if min_aum is None or min_aum <= 0:
        return True
    aum = _seam("_fund_row_aum")(row)
    return aum is not None and aum >= min_aum


def _infer_fund_type_from_name(name: Any) -> Optional[str]:
    text = _fund_service()._normalize_match_text(name)
    if not text:
        return None
    if "SERBEST" in text:
        return "Serbest Fon"
    if "PARA PIYASASI" in text:
        return "Para Piyasası Fonu"
    if "FON SEPETI" in text:
        return "Fon Sepeti Fonu"
    if "DEGISKEN" in text:
        return "Değişken Fon"
    if "BORCLANMA ARACLARI" in text or "EUROBOND" in text:
        return "Borçlanma Araçları Fonu"
    if "KIYMETLI MADEN" in text or "ALTIN" in text:
        return "Kıymetli Madenler Fonu"
    if "HISSE SENEDI" in text:
        return "Hisse Senedi Fonu"
    if "KATILIM" in text:
        return "Katılım Fonu"
    if "GIRISIM SERMAYESI" in text:
        return "Girişim Sermayesi Yatırım Fonu"
    if "GAYRIMENKUL" in text:
        return "Gayrimenkul Yatırım Fonu"
    return None


def _fund_name_match_candidates(name: Any) -> List[str]:
    text = _fund_service()._normalize_match_text(name)
    if not text:
        return []
    candidates = [text]
    pys_expanded = re.sub(r"\bPYS\b", "PORTFOY", text)
    if pys_expanded != text:
        candidates.append(pys_expanded)
    return candidates


def _infer_founder_from_fund_name(name: Any) -> Optional[str]:
    text = str(name or "").strip()
    if not text:
        return None
    match = re.match(r"^(.+?\bPORTFÖY)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.match(r"^(.+?\bPYŞ)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _is_closed_fund(row: Dict[str, Any]) -> bool:
    price = _fund_service()._coerce_float(_fund_service()._first_present(row, "sonFiyat", "fiyat", "FIYAT", "price"))
    daily_return = _fund_service()._coerce_float(_fund_service()._first_present(row, "gunlukGetiri", "daily_return"))
    share_count = _fund_service()._coerce_float(_fund_service()._first_present(row, "payAdet", "tedPaySayisi", "sonPayAdedi", "TEDPAYSAYISI"))
    aum = _fund_service()._coerce_float(_fund_service()._first_present(row, "portBuyukluk", "portfoyBuyukluk", "sonPortfoyDegeri", "PORTFOYBUYUKLUK"))
    return (
        price == 0
        and daily_return == -100
        and (share_count is None or share_count == 0)
        and (aum is None or aum == 0)
    )


def _snapshot_is_current_on_market_closure(
    snapshot: Dict[str, Any],
    *,
    today: Optional[date] = None,
) -> bool:
    """Whether a snapshot already covers the latest published market session.

    On weekends and full-day holidays TEFAS has no newer session to publish.
    Retrying a Friday snapshot on those days only creates a false upstream-error
    warning, so a non-empty snapshot for the latest business day is current.
    """

    current = today or _fund_service()._turkey_market_today()
    target = _fund_service()._latest_fund_snapshot_target_date(current)
    snapshot_as_of = _seam("_snapshot_as_of_date")(snapshot)
    return bool(
        current > target
        and snapshot_as_of is not None
        and snapshot_as_of >= target
        and (
            list(snapshot.get("rows") or [])
            or int(snapshot.get("total_count") or snapshot.get("count") or 0) > 0
        )
    )


def _mark_snapshot_current_for_market_closure(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Clear a stale transport warning when the market is expected to be closed."""

    payload = dict(snapshot)
    prior_warnings = list(payload.get("warnings") or [])
    meta = dict(payload.get("source_metadata") or {})
    target = _fund_service()._latest_fund_snapshot_target_date()
    if prior_warnings:
        meta["last_refresh_warnings"] = prior_warnings
    meta.update(
        {
            "resolution_status": "market_closed",
            "requested_as_of": target.isoformat(),
            "resolved_as_of": payload.get("as_of"),
            "snapshot_action": "retained_current_market_closed",
            "market_closed": True,
            "market_closed_target_date": target.isoformat(),
            "warnings": [],
        }
    )
    has_rows = bool(payload.get("rows")) or int(payload.get("total_count") or payload.get("count") or 0) > 0
    payload["status"] = "ok" if has_rows else payload.get("status")
    payload["stale"] = False
    payload["degraded"] = False
    payload["warnings"] = []
    payload["source_metadata"] = meta
    payload["snapshot_action"] = "retained_current_market_closed"
    payload["resolution_status"] = "market_closed"
    return payload


def _build_snapshot(
    rows: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
    *,
    source: str = _UNSET,
    source_url: str = _UNSET,
    parse_status: Optional[str] = None,
) -> Dict[str, Any]:
    if source is _UNSET:
        source = _fund_service().TEFASFON_FUNDS_SOURCE
    if source_url is _UNSET:
        source_url = _fund_service().TEFASFON_SOURCE_URL
    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        point = _fund_service()._normalize_history_row(row)
        if not point:
            continue
        normalized.setdefault(point["fund_code"], []).append(point)
    summaries = [
        summary
        for fund_code, points in normalized.items()
        for summary in [_fund_service()._summary_from_points(fund_code, points)]
        if summary is not None
    ]
    summaries.sort(key=lambda item: item["fund_code"])
    dates = [item.get("as_of") for item in summaries if item.get("as_of")]
    fetched_at = _fund_service()._utc_now_iso()
    adapter_version = _fund_service()._tefasfon_adapter_version() if _fund_service()._normalize_price_source(source).startswith("tefasfon") else None
    return {
        "status": "ok" if summaries else "empty",
        "rows": summaries,
        "count": len(summaries),
        "total_count": len(summaries),
        "source": source,
        "source_url": source_url,
        "as_of": max(dates) if dates else None,
        "fetched_at": fetched_at,
        "stale": False,
        "degraded": False,
        "warnings": warnings or [],
        "source_metadata": {
            "source": source,
            "source_url": source_url,
            "adapter_version": adapter_version,
            "fetched_at": fetched_at,
            "as_of": max(dates) if dates else None,
            "cache_hit": False,
            "stale": False,
            "parse_status": parse_status or ("ok" if summaries else "empty"),
            "tefas_open_only": _fund_service().TEFAS_OPEN_ONLY,
            "warnings": warnings or [],
        },
    }


def _empty_snapshot_payload(reason: str) -> Dict[str, Any]:
    now = _fund_service()._utc_now_iso()
    return {
        "status": "unavailable",
        "rows": [],
        "count": 0,
        "total_count": 0,
        "source": _fund_service().TEFASFON_FUNDS_SOURCE,
        "source_url": _fund_service().TEFASFON_SOURCE_URL,
        "as_of": None,
        "fetched_at": None,
        "stale": True,
        "degraded": True,
        "warnings": [reason],
        "source_metadata": {
            "source": _fund_service().TEFASFON_FUNDS_SOURCE,
            "source_url": _fund_service().TEFASFON_SOURCE_URL,
            "fetched_at": None,
            "as_of": None,
            "cache_hit": False,
            "stale": True,
            "parse_status": "unavailable",
            "tefas_open_only": _fund_service().TEFAS_OPEN_ONLY,
            "warnings": [reason],
            "served_at": now,
        },
    }


def _overlay_latest_price_history(
    processed_dir: Path,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge newer daily points into the point-in-time fund catalogue.

    TEFAS list metadata and daily price history are persisted independently.
    A failed catalogue refresh must not hide a newer valid price that is
    already present in the history store.
    """

    rows = [row for row in list(payload.get("rows") or []) if isinstance(row, dict)]
    if not rows:
        return payload

    snapshot_as_of = _fund_service()._fund_date(payload.get("as_of") or (payload.get("source_metadata") or {}).get("as_of"))
    target_date = _fund_service()._latest_fund_snapshot_target_date()
    latest_points = _fund_service().read_latest_fund_price_points(
        processed_dir,
        (str(row.get("fund_code") or "") for row in rows),
        end_date=target_date,
        include_metadata=False,
    )
    if not latest_points:
        return payload

    merged_rows: List[Dict[str, Any]] = []
    overlay_count = 0
    overlay_dates: List[str] = []
    overlay_fetched_at: List[str] = []
    for row in rows:
        code = _fund_service().normalize_fund_code(str(row.get("fund_code") or ""))
        point = latest_points.get(code)
        if not point:
            merged_rows.append(dict(row))
            continue
        row_date = _fund_service()._fund_date(row.get("as_of"))
        point_date = _fund_service()._fund_date(point.get("date"))
        if not point_date or (row_date and point_date < row_date):
            merged_rows.append(dict(row))
            continue

        merged = dict(row)
        is_newer_point = not row_date or point_date > row_date
        for key in ("price", "daily_return", "aum", "investor_count", "share_count"):
            if point.get(key) is not None:
                merged[key] = point[key]
        if is_newer_point:
            merged["as_of"] = point_date
            merged["source"] = point.get("source") or merged.get("source")
            overlay_count += 1
            overlay_dates.append(point_date)
            if point.get("fetched_at"):
                overlay_fetched_at.append(str(point["fetched_at"]))
        merged_rows.append(merged)

    if not overlay_count:
        return payload

    result = dict(payload)
    result["rows"] = merged_rows
    effective_as_of = max([value for value in [snapshot_as_of, *overlay_dates] if value], default=None)
    if effective_as_of:
        result["as_of"] = effective_as_of
    metadata = dict(result.get("source_metadata") or {})
    metadata["snapshot_as_of"] = snapshot_as_of
    metadata["price_history_as_of"] = max(overlay_dates) if overlay_dates else None
    metadata["price_history_overlay_count"] = overlay_count
    metadata["price_history_overlay_source"] = "fund_prices"
    metadata["price_history_fetched_at"] = max(overlay_fetched_at) if overlay_fetched_at else None
    result["source_metadata"] = metadata
    return result


def _catalog_price_overlay_metadata(processed_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    is_postgres = _fund_service().database_enabled()
    generation = payload.get("snapshot_generation")
    ttl = (
        max(21600, _fund_service().FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS)
        if (generation and is_postgres and _fund_service().FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS >= 900)
        else max(1, _fund_service().FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS)
    )
    original_as_of = _fund_service()._fund_date(payload.get("as_of") or (payload.get("source_metadata") or {}).get("as_of"))
    dates_cache_key = (
        f"catalog-dates:{_fund_service()._fund_prices_db_path(processed_dir)}:{generation}:{original_as_of}"
    )
    cached_dates_entry = _fund_service()._MEMORY_CACHE.get(dates_cache_key)
    all_rows_cached = (
        _fund_service()._peek_all_snapshot_rows(processed_dir, generation)
        if (generation is not None and is_postgres)
        else None
    )
    if (
        isinstance(cached_dates_entry, dict)
        and isinstance(cached_dates_entry.get("cached_at"), (int, float))
        and time.time() - float(cached_dates_entry["cached_at"]) <= ttl
        and isinstance(cached_dates_entry.get("catalog_dates"), dict)
    ):
        catalog_dates: Dict[str, Optional[str]] = dict(cached_dates_entry["catalog_dates"])
    elif (
        isinstance(all_rows_cached, dict)
        and isinstance(all_rows_cached.get("cached_at"), (int, float))
        and time.time() - float(all_rows_cached["cached_at"]) <= ttl
        and isinstance(all_rows_cached.get("catalog_dates"), dict)
    ):
        catalog_dates = dict(all_rows_cached["catalog_dates"])
        _fund_service()._memory_cache_put(
            dates_cache_key,
            {"cached_at": time.time(), "catalog_dates": catalog_dates},
        )
    else:
        with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
            records = conn.execute("SELECT fund_code, as_of FROM fund_snapshots ORDER BY fund_code").fetchall()
        catalog_dates = {}
        for record in records:
            code = str(record["fund_code"] or "")
            catalog_dates[code] = _fund_service()._fund_date(record["as_of"])
        _fund_service()._memory_cache_put(
            dates_cache_key,
            {"cached_at": time.time(), "catalog_dates": catalog_dates},
        )
    if not catalog_dates:
        return payload
    latest = _fund_service().read_latest_fund_price_points(
        processed_dir,
        catalog_dates.keys(),
        end_date=_fund_service()._latest_fund_snapshot_target_date(),
        include_metadata=False,
    )
    newer_dates: List[str] = []
    fetched_at: List[str] = []
    overlay_count = 0
    for code, point in latest.items():
        point_date = _fund_service()._fund_date(point.get("date"))
        current_date = catalog_dates.get(code)
        if not point_date or (current_date and point_date < current_date):
            continue
        if not current_date or point_date > current_date:
            overlay_count += 1
            newer_dates.append(point_date)
            if point.get("fetched_at"):
                fetched_at.append(str(point["fetched_at"]))
    result = dict(payload)
    original_as_of = _fund_service()._fund_date(payload.get("as_of") or (payload.get("source_metadata") or {}).get("as_of"))
    effective_as_of = max([value for value in [original_as_of, *newer_dates] if value], default=None)
    if effective_as_of:
        result["as_of"] = effective_as_of
    metadata = dict(result.get("source_metadata") or {})
    metadata["snapshot_as_of"] = original_as_of
    metadata["price_history_as_of"] = max(newer_dates) if newer_dates else None
    metadata["price_history_overlay_count"] = overlay_count
    metadata["price_history_overlay_source"] = "fund_prices"
    metadata["price_history_fetched_at"] = max(fetched_at) if fetched_at else None
    result["source_metadata"] = metadata
    return result


def load_funds_snapshot(
    processed_dir: Path,
    *,
    include_rows: bool = True,
    include_price_metadata: bool = True,
) -> Dict[str, Any]:
    path = _fund_service()._snapshot_path(processed_dir)
    cache_key = f"snapshot:{path}"
    stat = path.stat() if path.exists() else None
    file_signature = (
        getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
        int(stat.st_size),
    ) if stat else None
    stored_summary = _seam("ensure_fund_snapshot_store")(processed_dir)
    stored_metadata = stored_summary.get("source_metadata") if isinstance(stored_summary, dict) else None
    row_store_active = (
        isinstance(stored_metadata, dict)
        and stored_metadata.get("catalog_storage") == "fund_snapshots"
    )
    if row_store_active:
        payload = dict(stored_summary)
        payload["total_count"] = int(stored_summary.get("total_count") or stored_summary.get("count") or 0)
        payload["count"] = int(stored_summary.get("count") or 0)
        if include_rows:
            payload["rows"] = _seam("_read_fund_snapshot_rows")(processed_dir)
        else:
            payload.pop("rows", None)
            if include_price_metadata:
                payload = _seam("_catalog_price_overlay_metadata")(processed_dir, payload)
    else:
        cached = _fund_service()._MEMORY_CACHE.get(cache_key)
        if cached and stat and cached.get("file_signature") == file_signature:
            payload = dict(cached["payload"])
        else:
            payload = dict(_fund_service()._read_json(path) or _seam("_empty_snapshot_payload")("fund snapshot cache is empty"))
            if stat:
                _fund_service()._memory_cache_put(cache_key, {
                    "file_signature": file_signature,
                    "mtime": stat.st_mtime,
                    "payload": payload,
                })
        if not include_rows:
            payload.pop("rows", None)
    initial_metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    original_snapshot_as_of = _fund_service()._fund_date(
        initial_metadata.get("snapshot_as_of")
        or payload.get("as_of")
        or initial_metadata.get("as_of")
    )
    payload = _seam("_overlay_latest_price_history")(processed_dir, payload)
    fetched_at = payload.get("fetched_at")
    age = _fund_service()._cache_age_seconds(fetched_at)
    meta = dict(payload.get("source_metadata") or {})
    target_date = _fund_service()._latest_fund_snapshot_target_date()
    snapshot_as_of = original_snapshot_as_of or _fund_service()._fund_date(payload.get("as_of") or meta.get("as_of"))
    snapshot_lag_days: Optional[int] = None
    if snapshot_as_of:
        try:
            snapshot_lag_days = max(0, (target_date - date.fromisoformat(snapshot_as_of)).days)
        except ValueError:
            snapshot_lag_days = None
    is_behind_target = snapshot_lag_days is not None and snapshot_lag_days > 0
    resolution_status = str(meta.get("resolution_status") or payload.get("resolution_status") or "")
    last_checked_at = meta.get("last_checked_at")
    last_check_age = _fund_service()._cache_age_seconds(last_checked_at)
    successful_current_probe = (
        resolution_status in {"not_published", "resolved_previous_date", "available"}
        and str(meta.get("requested_as_of") or meta.get("last_checked_requested_date") or "")
        == target_date.isoformat()
        and last_check_age is not None
        and last_check_age <= max(_fund_service().FUNDS_SNAPSHOT_TTL_SECONDS, _fund_service().FUNDS_SNAPSHOT_INTRADAY_CHECK_TTL_SECONDS)
    )
    recently_checked_current_day = (
        is_behind_target
        and age is not None
        and age <= max(0, _fund_service().FUNDS_SNAPSHOT_INTRADAY_CHECK_TTL_SECONDS)
    )
    ttl_stale = age is None or age > _fund_service().FUNDS_SNAPSHOT_TTL_SECONDS
    stale = (
        bool(payload.get("stale")) or ttl_stale or (is_behind_target and not recently_checked_current_day)
    ) and not successful_current_probe
    recovered_current_probe = (
        successful_current_probe
        and resolution_status == "available"
        and snapshot_as_of == target_date.isoformat()
        and not list(meta.get("upstream_errors") or [])
    )
    public_source = _fund_service()._public_price_source(str(payload.get("source") or meta.get("source") or _fund_service().TEFASFON_FUNDS_SOURCE))
    meta["source"] = public_source
    if public_source == "legacy_cache":
        meta["source_url"] = None
    meta["cache_hit"] = bool(stat) or row_store_active
    meta["stale"] = stale
    meta["snapshot_target_date"] = target_date.isoformat()
    meta["snapshot_as_of_lag_days"] = snapshot_lag_days
    meta["snapshot_intraday_check_ttl_seconds"] = _fund_service().FUNDS_SNAPSHOT_INTRADAY_CHECK_TTL_SECONDS
    meta["snapshot_as_of"] = snapshot_as_of
    meta["resolution_status"] = resolution_status or None
    meta["last_checked_at"] = last_checked_at
    meta["as_of"] = payload.get("as_of")
    if is_behind_target:
        meta["awaiting_current_snapshot"] = recently_checked_current_day or successful_current_probe
    meta["tefas_open_only"] = _fund_service().TEFAS_OPEN_ONLY
    payload = dict(payload)
    visible_warnings = _fund_service()._dedupe_warning_messages(
        list(payload.get("warnings") or []) + list(meta.get("warnings") or [])
    )
    if recovered_current_probe:
        # A successful current-day probe supersedes warnings left by an older
        # failed attempt. Keep the warning only while the current snapshot is
        # actually degraded or the current probe has an upstream error.
        visible_warnings = []
    payload["warnings"] = visible_warnings
    meta["warnings"] = visible_warnings
    payload["source"] = public_source
    if public_source == "legacy_cache":
        payload["source_url"] = None
    if include_rows:
        payload["rows"] = [
            {**row, "source": _fund_service()._public_price_source(str(row.get("source") or public_source))}
            if isinstance(row, dict)
            else row
            for row in list(payload.get("rows") or [])
        ]
    payload["stale"] = stale
    payload["source_metadata"] = meta
    return payload


def _target_fund_codes_from_snapshot_payload(snapshot: Dict[str, Any]) -> List[str]:
    codes = {
        _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict)
        and _seam("_is_target_fund_row")(row)
        and _seam("_is_tefas_open_row")(row, require_known=True)
    }
    return sorted(code for code in codes if code)


def _target_fund_codes_from_env() -> List[str]:
    return sorted({_fund_service().normalize_fund_code(code) for code in _fund_service().TARGET_FUND_CODES if _fund_service().normalize_fund_code(code)})


def _tefas_open_codes_from_snapshot_payload(snapshot: Dict[str, Any]) -> set[str]:
    return {
        _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict) and _seam("_is_tefas_open_row")(row, require_known=True)
    }


def _snapshot_rows_by_code(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or "")): row
        for row in list(snapshot.get("rows") or [])
        if isinstance(row, dict) and _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
    }


def _enrich_points_from_snapshot(points: Iterable[Dict[str, Any]], snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows_by_code = _seam("_snapshot_rows_by_code")(snapshot)
    enriched: List[Dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        code = _fund_service().normalize_fund_code(str(point.get("fund_code") or ""))
        meta = rows_by_code.get(code, {})
        merged = dict(point)
        for key in (
            "name",
            "fund_type",
            "founder_company",
            "manager_company",
            "tefas_open",
            "risk_value",
            "currency",
            "isin",
        ):
            if merged.get(key) is None and meta.get(key) is not None:
                merged[key] = meta.get(key)
        enriched.append(merged)
    return enriched


def _target_fund_codes_for_collection(processed_dir: Path, *, lookback_days: int) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    snapshot = _seam("load_funds_snapshot")(processed_dir)
    env_codes = set(_seam("_target_fund_codes_from_env")())
    snapshot_codes = set(_seam("_target_fund_codes_from_snapshot_payload")(snapshot))
    if _fund_service().TEFAS_OPEN_ONLY and env_codes and snapshot.get("rows"):
        open_codes = _seam("_tefas_open_codes_from_snapshot_payload")(snapshot)
        skipped_env_codes = sorted(code for code in env_codes if code not in open_codes)
        if skipped_env_codes:
            warnings.append(
                "target fund codes skipped because they are not in the TEFAS-open snapshot: "
                + ", ".join(skipped_env_codes[:20])
            )
        env_codes = {code for code in env_codes if code in open_codes}
    codes = sorted(env_codes | snapshot_codes)
    if codes:
        return codes, warnings
    warnings.append("no target fund codes available; set RAGFIN_TARGET_FUND_CODES or keep a fund snapshot cache")
    return codes, warnings


def _fetch_fintables_udf_history_for_codes(
    fund_codes: Iterable[str],
    *,
    start_date: date,
    end_date: date,
    client: Optional[FintablesClient] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    fintables = client or _fund_service().FintablesClient()
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for fund_code in sorted({_fund_service().normalize_fund_code(code) for code in fund_codes if _fund_service().normalize_fund_code(code)}):
        try:
            rows.extend(fintables.fetch_udf_history(fund_code, start_date, end_date))
        except _fund_service().FintablesUpstreamError as exc:
            warnings.append(f"fintables_udf_history failed for {fund_code}: {exc}")
    return rows, warnings


def _fetch_tefasfon_history_for_codes(
    fund_codes: Iterable[str],
    *,
    start_date: date,
    end_date: date,
    client: Optional[TefasFonClient] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    codes = sorted({_fund_service().normalize_fund_code(code) for code in fund_codes if _fund_service().normalize_fund_code(code)})
    if not codes or start_date > end_date:
        return [], []
    tefas = client or _fund_service().TefasFonClient()
    chunks = _fund_service()._generate_date_chunks(start_date, end_date, max_chunk_days=_fund_service().TEFAS_HISTORY_CHUNK_MAX_DAYS)
    all_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for chunk_start, chunk_end in chunks:
        try:
            batch = tefas.fetch_funds(start_date=chunk_start, end_date=chunk_end, fund_codes=codes)
            all_rows.extend(batch)
        except _fund_service().TefasRateLimitError:
            raise
        except _fund_service().TefasUpstreamError as exc:
            warnings.append(
                f"tefasfon_funds chunk [{chunk_start.isoformat()} - {chunk_end.isoformat()}] failed: {exc}"
            )

    return all_rows, warnings


def _fetch_tefasfon_daily_snapshots_for_codes(
    fund_codes: Iterable[str],
    *,
    start_date: date,
    end_date: date,
    client: Optional[TefasFonClient] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    codes = sorted({_fund_service().normalize_fund_code(code) for code in fund_codes if _fund_service().normalize_fund_code(code)})
    if not codes or start_date > end_date:
        return [], []
    wanted = set(codes)
    tefas = client or _fund_service().TefasFonClient()
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    current = start_date
    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        try:
            if hasattr(tefas, "fetch_daily_funds_snapshot"):
                day_rows = tefas.fetch_daily_funds_snapshot(current)  # type: ignore[attr-defined]
            else:
                day_rows = tefas.fetch_funds(start_date=current, end_date=current)
        except _fund_service().TefasUpstreamError as exc:
            warnings.append(f"tefasfon_funds daily snapshot failed for {current.isoformat()}: {exc}")
            current += timedelta(days=1)
            continue
        if not day_rows:
            warnings.append(f"tefasfon_funds daily snapshot returned no rows for {current.isoformat()}")
            current += timedelta(days=1)
            continue
        rows.extend(
            row
            for row in day_rows
            if _fund_service().normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")) in wanted
        )
        current += timedelta(days=1)
    return rows, warnings


def _merge_refresh_rows_with_existing(
    existing_rows: Iterable[Dict[str, Any]],
    fresh_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep optional catalog fields when the fast TEFAS list omits them."""

    existing_by_code = {
        _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or "")): row
        for row in existing_rows
        if isinstance(row, dict)
        and _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
    }
    # Fields that describe the current quote. A previous value must never
    # survive into a new row: a stale ``daily_return_source`` would claim a
    # return that no longer exists and would also hide a fresh TEFAS range
    # return from the reconciliation step.
    volatile_fields = {
        "price",
        "daily_return",
        "daily_return_source",
        "aum",
        "investor_count",
        "share_count",
        "as_of",
        "fetched_at",
        "source",
        "source_url",
    }
    merged_rows: List[Dict[str, Any]] = []
    for row in fresh_rows:
        if not isinstance(row, dict):
            continue
        code = _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        previous = existing_by_code.get(code, {})
        merged = {
            key: value
            for key, value in previous.items()
            if key not in volatile_fields
        }
        merged.update(row)
        merged_rows.append(merged)
    return merged_rows


def _legacy_fund_fetch_result(
    client: Any,
    *,
    as_of: date,
    lookback_days: int,
    enrich: bool = False,
    source: str,
) -> FundSnapshotFetchResult:
    """Adapt older test doubles/adapters to the explicit outcome contract."""

    try:
        value = client.fetch_latest_fund_list_snapshot(
            as_of=as_of,
            lookback_days=lookback_days,
            enrich=enrich,
        )
    except TypeError as exc:
        if "enrich" not in str(exc):
            raise
        value = client.fetch_latest_fund_list_snapshot(as_of=as_of, lookback_days=lookback_days)
    rows, warnings = value if isinstance(value, tuple) and len(value) == 2 else (value, [])
    rows = list(rows or [])
    warnings = list(warnings or [])
    if rows:
        dates = [
            _fund_service()._fund_date(_fund_service()._first_present(row, "date", "tarih", "TARIH", "TARIHSTR"))
            for row in rows
            if isinstance(row, dict)
        ]
        resolved = max((item for item in dates if item), default=as_of.isoformat())
        return _fund_service().FundSnapshotFetchResult(
            rows=rows,
            warnings=warnings,
            resolution_status="available" if resolved == as_of.isoformat() else "resolved_previous_date",
            requested_as_of=as_of.isoformat(),
            resolved_as_of=resolved,
            attempted_dates=[as_of.isoformat()],
            source=source,
        )
    # Legacy adapters used warnings to signal transport errors. Preserve that
    # meaning while a warning-free empty response is a valid not-published
    # result under the new contract.
    has_error = bool(warnings)
    return _fund_service().FundSnapshotFetchResult(
        warnings=warnings,
        resolution_status="upstream_unavailable" if has_error else "not_published",
        requested_as_of=as_of.isoformat(),
        attempted_dates=[as_of.isoformat()],
        upstream_errors=list(warnings) if has_error else [],
        source=source,
    )


def _fund_fetch_result(client: Any, *, as_of: date, lookback_days: int, source: str) -> FundSnapshotFetchResult:
    method = getattr(client, "fetch_latest_fund_list_snapshot_result", None)
    if callable(method):
        try:
            result = method(as_of=as_of, lookback_days=lookback_days, enrich=False)
        except TypeError as exc:
            if "enrich" not in str(exc):
                raise
            result = method(as_of=as_of, lookback_days=lookback_days)
        if isinstance(result, _fund_service().FundSnapshotFetchResult):
            return result
    return _seam("_legacy_fund_fetch_result")(
        client,
        as_of=as_of,
        lookback_days=lookback_days,
        enrich=False,
        source=source,
    )


def _snapshot_as_of_date(snapshot: Dict[str, Any]) -> Optional[date]:
    value = _fund_service()._fund_date(snapshot.get("as_of") or (snapshot.get("source_metadata") or {}).get("as_of"))
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _add_snapshot_resolution_metadata(
    payload: Dict[str, Any],
    *,
    result: FundSnapshotFetchResult,
    action: str,
    warnings: List[str],
) -> Dict[str, Any]:
    normalized_warnings = _fund_service()._dedupe_warning_messages(warnings)
    meta = dict(payload.get("source_metadata") or {})
    meta.update(
        {
            "resolution_status": result.resolution_status,
            "requested_as_of": result.requested_as_of,
            "resolved_as_of": result.resolved_as_of,
            "last_checked_requested_date": result.requested_as_of,
            "last_checked_at": _fund_service()._utc_now_iso(),
            "snapshot_action": action,
            "upstream_errors": list(result.upstream_errors),
            "attempted_dates": list(result.attempted_dates),
            "empty_dates": list(result.empty_dates),
            "warnings": normalized_warnings,
        }
    )
    payload["source_metadata"] = meta
    payload["warnings"] = normalized_warnings
    payload["snapshot_action"] = action
    payload["resolution_status"] = result.resolution_status
    return payload


def refresh_funds_snapshot(
    processed_dir: Path,
    *,
    lookback_days: int = _UNSET,
    persist_reference_data: bool = True,
    backfill_daily_returns: bool = True,
    persist_snapshot: bool = True,
) -> Dict[str, Any]:
    if lookback_days is _UNSET:
        lookback_days = _fund_service().FUNDS_MAX_LOOKBACK_CALENDAR_DAYS
    end_date = _fund_service()._latest_fund_snapshot_target_date()
    existing_snapshot = _seam("load_funds_snapshot")(processed_dir)
    # A Friday close is already the latest valid snapshot throughout the
    # weekend/holiday. Do not call providers merely to turn it into a false
    # "refresh failed" warning.
    if _seam("_snapshot_is_current_on_market_closure")(existing_snapshot):
        retained = _seam("_mark_snapshot_current_for_market_closure")(existing_snapshot)
        if persist_snapshot and not _fund_service().database_enabled():
            _seam("commit_funds_snapshot")(
                processed_dir, retained, job_id="local-refresh", owner_token="local-refresh", generation=0
            )
        return retained
    bounded_lookback_days = min(_fund_service().FUNDS_MAX_LOOKBACK_CALENDAR_DAYS, max(1, int(lookback_days)))
    warnings: List[str] = []
    primary = _seam("_fund_fetch_result")(
        _fund_service().TefasFonClient(),
        as_of=end_date,
        lookback_days=bounded_lookback_days,
        source=_fund_service().TEFASFON_FUNDS_SOURCE,
    )
    result = primary
    warnings.extend(primary.warnings)
    # Direct TEFAS is a limited fallback for an unreliable primary adapter.
    # A reliable all-empty response is deliberately not reclassified as an
    # upstream failure merely because another adapter might behave differently.
    if not primary.rows and primary.resolution_status == "upstream_unavailable":
        direct = _seam("_fund_fetch_result")(
            _fund_service().TefasClient(),
            as_of=end_date,
            lookback_days=bounded_lookback_days,
            source=_fund_service().TEFAS_DIRECT_FUNDS_SOURCE,
        )
        warnings.extend(direct.warnings)
        if direct.rows:
            warnings.append("tefas_direct_funds fallback used")
            direct.upstream_errors = list(primary.upstream_errors) + list(direct.upstream_errors)
            result = direct
        else:
            result.upstream_errors = list(primary.upstream_errors) + list(direct.upstream_errors)
            result.resolution_status = "upstream_unavailable"
    existing_rows = list(existing_snapshot.get("rows") or [])
    existing_count = len(existing_rows)
    candidate_rows, _, _ = _seam("_filter_tefas_open_rows")(list(result.rows or []))
    candidate_count = len(candidate_rows)
    # Check the provider result before merging with the cache. Merging is
    # intentionally conservative and would otherwise hide a same-day
    # 1000-row response by filling it with the existing 2041 rows.
    if existing_count >= 100 and candidate_count and candidate_count + 50 < existing_count:
        truncation_warning = (
            f"tefasfon snapshot looked truncated ({candidate_count} rows vs cached {existing_count}); "
            "preserving existing snapshot and marking it stale"
        )
        merged = dict(existing_snapshot)
        merged_warnings = list(merged.get("warnings") or []) + warnings + [truncation_warning]
        merged_meta = dict(merged.get("source_metadata") or {})
        merged_meta["warnings"] = merged_warnings
        merged_meta["stale"] = True
        merged_meta["truncated_refresh_observed"] = True
        merged_meta["truncated_refresh_row_count"] = candidate_count
        merged_meta["truncated_refresh_total_count"] = existing_count
        merged["warnings"] = merged_warnings
        merged["stale"] = True
        merged["source_metadata"] = merged_meta
        merged = _seam("_add_snapshot_resolution_metadata")(
            merged,
            result=result,
            action="retained_existing",
            warnings=merged_warnings,
        )
        return merged

    rows = _seam("_merge_refresh_rows_with_existing")(existing_rows, result.rows)
    rows, skipped_closed, skipped_unknown = _seam("_filter_tefas_open_rows")(rows)
    carried_daily_returns = _fund_service()._carry_forward_resolved_daily_returns(existing_rows, rows)
    if skipped_closed:
        warnings.append(
            f"tefas_open_only skipped {skipped_closed} closed fund rows"
        )
    if not rows:
        payload = dict(existing_snapshot)
        all_warnings = list(existing_snapshot.get("warnings") or []) + warnings
        action = "retained_existing"
        payload["status"] = (
            "unavailable"
            if result.resolution_status == "upstream_unavailable"
            else ("ok" if existing_snapshot.get("rows") else "empty")
        )
        payload["stale"] = result.resolution_status == "upstream_unavailable"
        payload["degraded"] = result.resolution_status == "upstream_unavailable" or not bool(payload.get("rows"))
        payload = _seam("_add_snapshot_resolution_metadata")(
            payload,
            result=result,
            action=action,
            warnings=all_warnings,
        )
        payload.setdefault("source_metadata", {})["parse_status"] = (
            "upstream_unavailable" if result.resolution_status == "upstream_unavailable" else "not_published"
        )
        payload["source_metadata"]["source_policy"] = _fund_service().FUND_HISTORY_SOURCE_POLICY
        payload["source_metadata"]["fallback_used"] = bool(existing_snapshot.get("rows"))
        payload["source_metadata"]["tefas_open_only"] = _fund_service().TEFAS_OPEN_ONLY
        if persist_snapshot and not _fund_service().database_enabled():
            _seam("commit_funds_snapshot")(
                processed_dir, payload, job_id="local-refresh", owner_token="local-refresh", generation=0
            )
        return payload
    snapshot = _seam("_build_snapshot")(
        rows,
        warnings=warnings,
        source=result.source,
        source_url=_fund_service().TEFAS_BASE_URL if result.source == _fund_service().TEFAS_DIRECT_FUNDS_SOURCE else _fund_service().TEFASFON_SOURCE_URL,
        parse_status="ok_tefas_funds",
    )
    snapshot["source_metadata"]["source_policy"] = _fund_service().FUND_HISTORY_SOURCE_POLICY
    snapshot["source_metadata"]["fallback_used"] = result.source == _fund_service().TEFAS_DIRECT_FUNDS_SOURCE
    if not snapshot["rows"]:
        return {
            **existing_snapshot,
            "stale": True,
            "degraded": True,
            "warnings": list(existing_snapshot.get("warnings") or []) + warnings + ["tefasfon_funds returned no valid fund rows"],
        }
    found_as_of = _seam("_snapshot_as_of_date")(snapshot)
    existing_as_of = _seam("_snapshot_as_of_date")(existing_snapshot)
    if found_as_of is not None and existing_as_of is not None and found_as_of < existing_as_of:
        # A genuinely older response must not downgrade the catalogue. An
        # equal-date response is still a new refresh: TEFAS can publish a
        # different subset of rows on the same day, so retaining the previous
        # rows here would keep stale per-fund prices indefinitely.
        action = "retained_newer"
        retained = dict(existing_snapshot)
        retained["stale"] = False if result.resolution_status != "upstream_unavailable" else True
        retained["degraded"] = result.resolution_status == "upstream_unavailable"
        retained_warnings = _fund_service()._dedupe_warning_messages(
            list(retained.get("warnings") or []) + warnings
        )
        retained = _seam("_add_snapshot_resolution_metadata")(
            retained,
            result=result,
            action=action,
            warnings=retained_warnings,
        )
        retained.setdefault("source_metadata", {})["source_policy"] = _fund_service().FUND_HISTORY_SOURCE_POLICY
        if persist_snapshot and not _fund_service().database_enabled():
            _seam("commit_funds_snapshot")(
                processed_dir, retained, job_id="local-refresh", owner_token="local-refresh", generation=0
            )
        return retained
    snapshot = _seam("_add_snapshot_resolution_metadata")(
        snapshot,
        result=result,
        action="updated",
        warnings=warnings,
    )
    snapshot["stale"] = False
    snapshot["degraded"] = False
    # Guard: do not overwrite a healthy snapshot with a clearly-truncated refresh
    # (e.g. TEFAS pagination cut short and we end up with the first page only).
    new_count = len(snapshot["rows"])
    if existing_count >= 100 and new_count + 50 < existing_count:
        truncation_warning = (
            f"tefasfon snapshot looked truncated ({new_count} rows vs cached {existing_count}); "
            "preserving existing snapshot and marking it stale"
        )
        merged = dict(existing_snapshot)
        merged_warnings = list(merged.get("warnings") or []) + warnings + [truncation_warning]
        merged_meta = dict(merged.get("source_metadata") or {})
        merged_meta["warnings"] = merged_warnings
        merged_meta["stale"] = True
        merged_meta["truncated_refresh_observed"] = True
        merged_meta["truncated_refresh_row_count"] = new_count
        merged_meta["truncated_refresh_total_count"] = existing_count
        merged["warnings"] = merged_warnings
        merged["stale"] = True
        merged["source_metadata"] = merged_meta
        merged = _seam("_add_snapshot_resolution_metadata")(
            merged,
            result=result,
            action="retained_existing",
            warnings=merged_warnings,
        )
        return merged
    _fund_service().upsert_fund_price_points(
        processed_dir,
        rows,
        source=result.source,
        fetched_at=str(snapshot.get("fetched_at") or _fund_service()._utc_now_iso()),
    )
    # Resolve every catalogue return once at write time. Previous-close returns
    # from price history replace range-return values and record their source, so
    # a later `/funds` rebuild no longer scans the 5-day window for every fund.
    missing_return_keys = _fund_service()._missing_daily_return_keys(snapshot["rows"])
    daily_return_overrides = _fund_service()._daily_return_overrides_from_price_history(
        processed_dir,
        snapshot["rows"],
        include_missing=True,
    )
    local_fallback_count = sum(
        1 for key in daily_return_overrides if key in missing_return_keys
    )
    if daily_return_overrides:
        snapshot["rows"] = _fund_service()._daily_return_override_values(
            snapshot["rows"],
            daily_return_overrides,
            mark_price_history_source=True,
        )
        _fund_service()._persist_daily_return_overrides(processed_dir, daily_return_overrides)
    backfilled = (
        _fund_service()._backfill_daily_returns_from_local_prices(processed_dir, snapshot["rows"])
        if backfill_daily_returns
        else 0
    )
    local_fallback_count += backfilled
    if local_fallback_count:
        meta = snapshot.setdefault("source_metadata", {})
        meta["daily_return_local_fallback_count"] = local_fallback_count
    if carried_daily_returns:
        snapshot.setdefault("source_metadata", {})[
            "daily_return_carried_forward_count"
        ] = carried_daily_returns
    if persist_reference_data:
        snapshot["source_metadata"]["reference_data"] = _fund_service()._upsert_fund_reference_data(processed_dir, snapshot["rows"])
    else:
        # The catalog snapshot already contains the searchable fund metadata.
        # Reference-data upserts are intentionally deferred during the fast
        # refresh path because writing thousands of optional records to a
        # remote database can otherwise keep the background job running for
        # minutes. Existing fee/reference values remain available on detail
        # pages and can be synchronized by a separate maintenance flow.
        snapshot["source_metadata"]["reference_data"] = {"deferred": True}
    if persist_snapshot:
        # The deployed refresh job owns the latest catalogue through the
        # Postgres generation lease. The collector may still populate the
        # price-history store, but must not race that fenced snapshot write.
        if not _fund_service().database_enabled():
            _seam("commit_funds_snapshot")(
                processed_dir, snapshot, job_id="local-refresh", owner_token="local-refresh", generation=0
            )
    return snapshot


def collect_daily_fund_prices(
    processed_dir: Path,
    *,
    as_of: Optional[date] = None,
    lookback_days: int = _UNSET,
) -> Dict[str, Any]:
    if lookback_days is _UNSET:
        lookback_days = _fund_service().FUNDS_COLLECTOR_LOOKBACK_DAYS
    effective_as_of = as_of or _fund_service()._latest_fund_snapshot_target_date()
    start_date = effective_as_of - timedelta(days=max(1, lookback_days))
    existing_snapshot = _seam("load_funds_snapshot")(processed_dir)
    fetched_at = _fund_service()._utc_now_iso()
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    source = _fund_service().TEFASFON_FUNDS_SOURCE
    source_url = _fund_service().TEFASFON_SOURCE_URL
    raw_row_count = 0
    skipped_by_manager_count = 0
    fallback_attempted = False
    large_universe_daily_snapshots = False

    target_codes, code_warnings = _seam("_target_fund_codes_for_collection")(processed_dir, lookback_days=max(1, lookback_days))
    warnings.extend(code_warnings)
    if target_codes:
        if len(target_codes) > 100:
            large_universe_daily_snapshots = True
            tefas_rows, tefas_warnings = _seam("_fetch_tefasfon_daily_snapshots_for_codes")(
                target_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
            warnings.append("tefasfon_funds used daily all-fund snapshots for large target universe")
        else:
            tefas_rows, tefas_warnings = _seam("_fetch_tefasfon_history_for_codes")(
                target_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
        warnings.extend(tefas_warnings)
        if not tefas_rows and not large_universe_daily_snapshots:
            snapshot_rows, snapshot_warnings = _seam("_fetch_tefasfon_daily_snapshots_for_codes")(
                target_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
            if snapshot_rows:
                warnings.append("tefasfon_funds code-filtered history was empty; used daily all-fund snapshots")
                tefas_rows = snapshot_rows
            warnings.extend(snapshot_warnings)
        tefas_codes = {
            point["fund_code"]
            for row in tefas_rows
            for point in [_fund_service()._normalize_history_row(row)]
            if point is not None and _fund_service()._coerce_float(point.get("price")) is not None and _fund_service()._coerce_float(point.get("price")) > 0
        }
        missing_codes = sorted(set(target_codes) - {code for code in tefas_codes if code})
        fintables_rows: List[Dict[str, Any]] = []
        if missing_codes and not large_universe_daily_snapshots:
            fallback_attempted = True
            fintables_rows, fintables_warnings = _seam("_fetch_fintables_udf_history_for_codes")(
                missing_codes,
                start_date=start_date,
                end_date=effective_as_of,
            )
            warnings.extend(fintables_warnings)
        elif missing_codes:
            warnings.append(
                f"fintables_udf_history fallback skipped for {len(missing_codes)} missing codes in large TEFAS snapshot collection"
            )
        rows = list(tefas_rows) + list(fintables_rows)
        if not tefas_rows and fintables_rows:
            source = _fund_service().FINTABLES_UDF_HISTORY_SOURCE
            source_url = _fund_service().FINTABLES_UDF_HISTORY_ENDPOINT
        raw_row_count = len(tefas_rows) + len(fintables_rows)
        if missing_codes:
            warnings.append(f"fintables_udf_history fallback requested for {len(missing_codes)} missing TEFAS fund codes")
    else:
        warnings.append("no target fund codes available for TEFAS/Fintables collection")

    rows = _seam("_enrich_points_from_snapshot")(rows, existing_snapshot)
    snapshot_rows = _fund_service()._dedupe_price_points(rows)

    storage_result = _fund_service().upsert_fund_price_points(
        processed_dir,
        rows,
        source=source,
        fetched_at=fetched_at,
    )
    snapshot = _seam("_build_snapshot")(
        snapshot_rows,
        warnings=warnings,
        source=source,
        source_url=source_url,
        parse_status="ok_collector" if rows else "empty_collector",
    )
    snapshot["source_metadata"]["source_policy"] = _fund_service().FUND_HISTORY_SOURCE_POLICY
    snapshot["source_metadata"]["fallback_used"] = fallback_attempted
    if snapshot_rows:
        daily_return_overrides = _fund_service()._daily_return_overrides_from_price_history(processed_dir, snapshot["rows"])
        if daily_return_overrides:
            snapshot["rows"] = _fund_service()._daily_return_override_values(snapshot["rows"], daily_return_overrides)
            _fund_service()._persist_daily_return_overrides(processed_dir, daily_return_overrides)
        backfilled = _fund_service()._backfill_daily_returns_from_local_prices(processed_dir, snapshot["rows"])
        if backfilled:
            snapshot["source_metadata"]["daily_return_local_fallback_count"] = backfilled
        snapshot["source_metadata"]["reference_data"] = _fund_service()._upsert_fund_reference_data(processed_dir, snapshot_rows)
        # The collector writes history only. Catalogue refreshes own the row
        # snapshot, while current list prices are overlaid from fund_prices.

    skipped_warnings = [
        str(item.get("warning") or "invalid_price_row")
        for item in list(storage_result.get("warnings") or [])
    ]
    all_warnings = warnings + skipped_warnings
    valid_dates = [
        point["date"]
        for row in rows
        for point in [_fund_service()._normalize_history_row(row)]
        if point is not None and _fund_service()._coerce_float(point.get("price")) is not None and _fund_service()._coerce_float(point.get("price")) > 0
    ]
    storage_as_of = max(valid_dates) if valid_dates else snapshot.get("as_of")
    return {
        "status": "ok" if storage_result.get("upserted_count") else ("empty" if rows else "unavailable"),
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": effective_as_of.isoformat(),
        "source": source,
        "source_url": source_url,
        "fetched_at": fetched_at,
        "row_count": len(rows),
        "raw_row_count": raw_row_count,
        "skipped_by_manager_count": skipped_by_manager_count,
        "valid_point_count": int(storage_result.get("upserted_count") or 0),
        "skipped_point_count": int(storage_result.get("skipped_count") or 0),
        "warning_count": len(all_warnings),
        "warnings": all_warnings[:50],
        "db_path": storage_result.get("db_path") or str(_fund_service()._fund_prices_db_path(processed_dir)),
        "as_of": storage_as_of,
        "source_metadata": {
            "source": source,
            "source_url": source_url,
            "db_path": storage_result.get("db_path") or str(_fund_service()._fund_prices_db_path(processed_dir)),
            "fetched_at": fetched_at,
            "as_of": storage_as_of,
            "parse_status": "ok" if storage_result.get("upserted_count") else "empty",
            "source_policy": _fund_service().FUND_HISTORY_SOURCE_POLICY,
            "fallback_used": fallback_attempted,
            "tefas_open_only": _fund_service().TEFAS_OPEN_ONLY,
            "warnings": all_warnings[:50],
            "storage": {
                "sources": storage_result.get("sources") or {},
                "upserted_count": storage_result.get("upserted_count"),
                "skipped_count": storage_result.get("skipped_count"),
                "warning_count": storage_result.get("warning_count"),
            },
        },
    }


def _row_matches(row: Dict[str, Any], *, q: Optional[str], fund_type: Optional[str], founder: Optional[str], manager: Optional[str], risk: Optional[str]) -> bool:
    if q:
        needle = q.strip().lower()
        haystack = " ".join(
            str(row.get(key) or "")
            for key in ("fund_code", "name", "fund_type", "founder_company", "manager_company")
        ).lower()
        if needle not in haystack:
            return False
    if fund_type and str(row.get("fund_type") or "").strip().lower() != fund_type.strip().lower():
        return False
    if founder and str(row.get("founder_company") or "").strip().lower() != founder.strip().lower():
        return False
    if manager and str(row.get("manager_company") or "").strip().lower() != manager.strip().lower():
        return False
    if risk and str(row.get("risk_value") or "").strip().lower() != risk.strip().lower():
        return False
    return True


def _sort_rows(rows: List[Dict[str, Any]], sort: str, order: str) -> List[Dict[str, Any]]:
    allowed = {
        "fund_code",
        "name",
        "fund_type",
        "founder_company",
        "manager_company",
        "price",
        "daily_return",
        "risk_value",
        "aum",
        "investor_count",
        "as_of",
    }
    key = sort if sort in allowed else "fund_code"
    descending = order.lower() == "desc"

    def _typed_value(row: Dict[str, Any]) -> Any:
        value = row.get(key)
        if key == "risk_value" and value is not None:
            numeric = _fund_service()._coerce_float(value)
            if numeric is not None:
                return numeric
        return value

    def sort_key(row: Dict[str, Any]) -> Tuple[int, Any]:
        value = _typed_value(row)
        if value is None:
            return (1, "")
        if isinstance(value, (int, float)):
            return (0, -value if descending else value)
        return (0, str(value).lower())

    ordered = sorted(rows, key=sort_key)
    first_valued = next((_typed_value(row) for row in ordered if _typed_value(row) is not None), "")
    if descending and ordered and not isinstance(first_valued, (int, float)):
        valued = [row for row in ordered if _typed_value(row) is not None]
        missing = [row for row in ordered if _typed_value(row) is None]
        valued.reverse()
        return valued + missing
    return ordered


def _query_fund_snapshot_rows(
    processed_dir: Path,
    *,
    q: Optional[str],
    fund_type: Optional[str],
    founder: Optional[str],
    manager: Optional[str],
    risk: Optional[str],
    sort: str,
    order: str,
    min_aum: Optional[float],
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter and sort the current catalogue in SQL, returning only matches."""

    if _fund_service().database_enabled():
        all_rows = _seam("_read_fund_snapshot_rows")(processed_dir)
        matched = [
            row
            for row in all_rows
            if _seam("_is_tefas_open_row")(row)
            and _seam("_meets_min_aum")(row, min_aum)
            and _seam("_row_matches")(
                row,
                q=q,
                fund_type=fund_type,
                founder=founder,
                manager=manager,
                risk=risk,
            )
        ]
        matching_count = len(matched)
        ordered = _seam("_sort_rows")(matched, sort=sort, order=order)
        if limit is not None:
            ordered = ordered[: max(0, int(limit))]
        return ordered, matching_count

    conditions: List[str] = []
    params: List[Any] = []
    if _fund_service().TEFAS_OPEN_ONLY:
        conditions.append("(tefas_open IS NULL OR tefas_open = 1)")
    if min_aum is not None and min_aum > 0:
        conditions.append("aum >= ?")
        params.append(float(min_aum))
    for value, column in (
        (fund_type, "fund_type_key"),
        (founder, "founder_key"),
        (manager, "manager_key"),
        (risk, "risk_key"),
    ):
        if value:
            conditions.append(f"{column} = ?")
            params.append(str(value).strip().lower())
    needle = q.strip().lower() if q else ""
    if needle:
        if _fund_service().database_enabled():
            conditions.append("strpos(search_text, ?) > 0")
        else:
            conditions.append("instr(search_text, ?) > 0")
        params.append(needle)

    where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
    sort_columns = {
        "fund_code": "fund_code_sort",
        "name": "name_sort",
        "fund_type": "fund_type_sort",
        "founder_company": "founder_sort",
        "manager_company": "manager_sort",
        "price": "price",
        "daily_return": "daily_return",
        "risk_value": "risk_value_num",
        "aum": "aum",
        "investor_count": "investor_count",
        "as_of": "as_of",
    }
    sort_column = sort_columns.get(sort, "fund_code_sort")
    direction = "DESC" if order.lower() == "desc" else "ASC"
    text_sort = sort in {
        "fund_code", "name", "fund_type", "founder_company", "manager_company"
    }
    if text_sort:
        collation = '"C"' if _fund_service().database_enabled() else "BINARY"
        sort_expression = f"{sort_column} COLLATE {collation}"
    else:
        sort_expression = sort_column
    tie_order = "DESC" if text_sort and direction == "DESC" else "ASC"
    order_by = f"({sort_column} IS NULL) ASC, {sort_expression} {direction}, row_order {tie_order}"
    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        matching_row = conn.execute(
            f"SELECT COUNT(*) AS row_count FROM fund_snapshots{where_sql}",
            tuple(params),
        ).fetchone()
        matching_count = int(
            matching_row.get("row_count") or 0
            if isinstance(matching_row, dict)
            else matching_row["row_count"]
        )
        projection = (
            "fund_code, name, fund_type, founder_company, manager_company, price, daily_return, "
            "aum, investor_count, share_count, risk_value, risk_value_json, currency, as_of, source, "
            "period_returns_json, attributes_json, present_fields_json, row_order"
        )
        limit_sql = " LIMIT ?" if limit is not None else ""
        record_params = (*params, max(0, int(limit))) if limit is not None else tuple(params)
        records = conn.execute(
            f"SELECT {projection} FROM fund_snapshots{where_sql} ORDER BY {order_by}{limit_sql}",
            record_params,
        ).fetchall()
    return [_seam("_snapshot_row_from_record")(record) for record in records], matching_count


def _refresh_funds_snapshot_if_stale(processed_dir: Path, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not snapshot.get("stale"):
        return snapshot
    # In a deployed Postgres-backed instance, refreshes must go through the
    # API job so the generation/ownership transaction fences late workers.
    # Synchronous stale refresh remains useful for local file-cache mode.
    if _fund_service().database_enabled():
        return snapshot
    source = _fund_service()._public_price_source(str(snapshot.get("source") or (snapshot.get("source_metadata") or {}).get("source") or ""))
    if source != _fund_service().TEFASFON_FUNDS_SOURCE:
        return snapshot
    # First the cheap process-local guard so a busy uvicorn worker doesn't kick
    # off two refreshes at once.
    acquired_local = _fund_service()._SNAPSHOT_REFRESH_LOCK.acquire(blocking=False)
    if not acquired_local:
        return snapshot
    try:
        # When Redis is wired up the distributed lock prevents *other* workers
        # (or replicas) from also calling TEFAS at the same time. With the
        # default in-memory backend this is just an immediate ``True``.
        from app.cache import get_cache

        cache = get_cache()
        with cache.lock("funds-snapshot-refresh", timeout=120) as acquired_remote:
            if not acquired_remote:
                return snapshot
            try:
                _seam("refresh_funds_snapshot")(processed_dir, lookback_days=10)
                return _seam("load_funds_snapshot")(processed_dir)
            except Exception as exc:
                payload = dict(snapshot)
                warnings = list(payload.get("warnings") or [])
                warnings.append(f"fund snapshot auto-refresh failed: {exc}")
                meta = dict(payload.get("source_metadata") or {})
                meta["auto_refresh_failed"] = True
                meta["auto_refresh_error"] = str(exc)
                meta["warnings"] = list(meta.get("warnings") or []) + [f"fund snapshot auto-refresh failed: {exc}"]
                payload["warnings"] = warnings
                payload["source_metadata"] = meta
                return payload
    finally:
        _fund_service()._SNAPSHOT_REFRESH_LOCK.release()


__all__ = [
    "_encode_fund_snapshot_present_fields",
    "_decode_fund_snapshot_present_fields",
    "_fund_snapshot_record",
    "_fund_snapshot_insert_sql",
    "_fund_snapshot_records",
    "_fund_snapshot_summary",
    "_stage_fund_snapshot_codes",
    "_delete_fund_snapshot_rows_missing_from_stage",
    "_snapshot_row_from_record",
    "_read_fund_snapshot_rows",
    "_read_fund_snapshot_category_returns",
    "_copy_legacy_snapshot",
    "_snapshot_metadata_from_store",
    "ensure_fund_snapshot_store",
    "prune_fund_snapshot_rollback_copy",
    "_commit_local_fund_snapshot_rows",
    "commit_funds_snapshot",
    "_daily_snapshot_cache_is_fresh",
    "_cached_daily_funds_snapshot",
    "_tefas_open_status",
    "_is_tefas_open_row",
    "_filter_tefas_open_rows",
    "_founder_match_key",
    "_target_manager_tokens",
    "_is_target_fund_row",
    "_filter_target_fund_rows",
    "_fund_row_aum",
    "_meets_min_aum",
    "_infer_fund_type_from_name",
    "_fund_name_match_candidates",
    "_infer_founder_from_fund_name",
    "_is_closed_fund",
    "_snapshot_is_current_on_market_closure",
    "_mark_snapshot_current_for_market_closure",
    "_build_snapshot",
    "_empty_snapshot_payload",
    "_overlay_latest_price_history",
    "_catalog_price_overlay_metadata",
    "load_funds_snapshot",
    "_target_fund_codes_from_snapshot_payload",
    "_target_fund_codes_from_env",
    "_tefas_open_codes_from_snapshot_payload",
    "_snapshot_rows_by_code",
    "_enrich_points_from_snapshot",
    "_target_fund_codes_for_collection",
    "_fetch_fintables_udf_history_for_codes",
    "_fetch_tefasfon_history_for_codes",
    "_fetch_tefasfon_daily_snapshots_for_codes",
    "_merge_refresh_rows_with_existing",
    "_legacy_fund_fetch_result",
    "_fund_fetch_result",
    "_snapshot_as_of_date",
    "_add_snapshot_resolution_metadata",
    "refresh_funds_snapshot",
    "collect_daily_fund_prices",
    "_row_matches",
    "_sort_rows",
    "_query_fund_snapshot_rows",
    "_refresh_funds_snapshot_if_stale",
]

_ORIGINAL_EXPORTS.update({name: globals()[name] for name in __all__})
