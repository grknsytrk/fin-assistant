from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.database import connect_postgres, database_enabled, lock_down_postgres_tables

REFERENCE_DATA_DB_FILENAME = os.getenv("RAGFIN_REFERENCE_DATA_DB_FILENAME", "reference_data.sqlite3")

_SOURCE_PRIORITY = {
    "manual": 100,
    "kap": 80,
    "kap_cache": 75,
    "tefasfon_funds": 70,
    "tefas": 60,
    "tefas_list_snapshot": 60,
    "legacy_json": 20,
}

_MANUAL_INSTRUMENTS: List[Dict[str, Any]] = [
    {
        "kind": "stock",
        "symbol": "TERA",
        "name": "TERA YATIRIM MENKUL DEĞERLER A.Ş.",
        "source": "manual",
        "logo_url": "https://storage.fintables.com/media/uploads/company-logos/tera_icon.png",
        "logo_source": "manual",
    },
    {
        "kind": "stock",
        "symbol": "TRHOL",
        "name": "TERA FİNANSAL YATIRIMLAR HOLDİNG A.Ş.",
        "source": "manual",
        "logo_url": "https://s3-symbol-logo.tradingview.com/dagi-yatirim-holding--big.svg",
        "logo_source": "manual",
    },
    {
        "kind": "stock",
        "symbol": "TEHOL",
        "name": "TERA YATIRIM TEKNOLOJİ HOLDİNG A.Ş.",
        "source": "manual",
        "logo_url": "https://storage.fintables.com/media/uploads/company-logos/tera_icon.png",
        "logo_source": "manual",
    },
]

_SEEDED_PROCESSED_DIRS: set[str] = set()
_BOOTSTRAPPED_PROCESSED_DIRS: set[str] = set()
_REFERENCE_SCHEMA_LOCK = threading.Lock()
_REFERENCE_SCHEMA_READY = False
_INSTRUMENTS_CACHE_TTL_SECONDS = float(os.getenv("RAGFIN_INSTRUMENTS_CACHE_TTL_SECONDS", "1800"))
_INSTRUMENTS_CACHE_LOCK = threading.RLock()
_INSTRUMENTS_CACHE: Dict[
    str,
    tuple[float, Dict[tuple[str, str], Dict[str, Any]], Dict[tuple[str, str], str]],
] = {}
_INSTRUMENTS_STAMP_CACHE: Dict[str, tuple[str, str]] = {}
_INSTRUMENT_SELECT_COLUMNS = (
    "kind, symbol, name, short_name, source, source_id, "
    "logo_url, logo_source, active, as_of, metadata_json, "
    "created_at, updated_at"
)


def _instruments_cache_key(processed_dir: Path) -> str:
    if database_enabled():
        db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL") or ""
        return f"postgres:{hash(db_url)}"
    return str(Path(processed_dir).resolve())


def clear_instruments_cache(processed_dir: Optional[Path] = None) -> None:
    """Clear the process-local instruments cache (all scopes or a single scope)."""
    with _INSTRUMENTS_CACHE_LOCK:
        if processed_dir is None:
            _INSTRUMENTS_CACHE.clear()
            _INSTRUMENTS_STAMP_CACHE.clear()
        else:
            cache_key = _instruments_cache_key(processed_dir)
            _INSTRUMENTS_CACHE.pop(cache_key, None)
            _INSTRUMENTS_STAMP_CACHE.pop(cache_key, None)


def _read_instruments_stamp(conn: Any) -> Optional[tuple[str, str]]:
    row = conn.execute(
        """
        SELECT
            (SELECT COALESCE(MAX(updated_at), '') || ':' || COUNT(*) FROM instruments) AS instruments_stamp,
            (SELECT COALESCE(MAX(updated_at), '') || ':' || COUNT(*) FROM instrument_aliases) AS aliases_stamp
        """
    ).fetchone()
    if row is None:
        return None
    return (str(row["instruments_stamp"] or ""), str(row["aliases_stamp"] or ""))


def _copy_instrument_dict(inst: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(inst)
    metadata = inst.get("metadata")
    if isinstance(metadata, dict):
        copied["metadata"] = dict(metadata)
    return copied


def _ensure_instruments_cache(
    processed_dir: Path,
) -> tuple[Dict[tuple[str, str], Dict[str, Any]], Dict[tuple[str, str], str]]:
    cache_key = _instruments_cache_key(processed_dir)
    with _INSTRUMENTS_CACHE_LOCK:
        now = time.monotonic()
        cached = _INSTRUMENTS_CACHE.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1], cached[2]

        by_kind_symbol: Dict[tuple[str, str], Dict[str, Any]] = {}
        aliases_by_kind_alias: Dict[tuple[str, str], str] = {}
        stamp: Optional[tuple[str, str]] = None
        with _connect(processed_dir) as conn:
            if getattr(conn, "is_postgres", False):
                stamp = _read_instruments_stamp(conn)
                if (
                    cached is not None
                    and stamp is not None
                    and _INSTRUMENTS_STAMP_CACHE.get(cache_key) == stamp
                ):
                    if _INSTRUMENTS_CACHE_TTL_SECONDS > 0:
                        _INSTRUMENTS_CACHE[cache_key] = (
                            time.monotonic() + _INSTRUMENTS_CACHE_TTL_SECONDS,
                            cached[1],
                            cached[2],
                        )
                    return cached[1], cached[2]
            rows = conn.execute(f"SELECT {_INSTRUMENT_SELECT_COLUMNS} FROM instruments").fetchall()
            alias_rows = conn.execute("SELECT alias, kind, symbol FROM instrument_aliases").fetchall()
        for row in rows:
            instrument = _row_to_dict(row)
            norm_kind = normalize_instrument_kind(instrument.get("kind"))
            norm_symbol = normalize_instrument_symbol(instrument.get("symbol"))
            if norm_kind and norm_symbol:
                by_kind_symbol[(norm_kind, norm_symbol)] = instrument
        for row in alias_rows:
            norm_kind = normalize_instrument_kind(row["kind"])
            norm_alias = normalize_instrument_symbol(row["alias"])
            norm_symbol = normalize_instrument_symbol(row["symbol"])
            if norm_kind and norm_alias and norm_symbol:
                aliases_by_kind_alias[(norm_kind, norm_alias)] = norm_symbol

        if _INSTRUMENTS_CACHE_TTL_SECONDS > 0:
            _INSTRUMENTS_CACHE[cache_key] = (
                time.monotonic() + _INSTRUMENTS_CACHE_TTL_SECONDS,
                by_kind_symbol,
                aliases_by_kind_alias,
            )
            if stamp is not None:
                _INSTRUMENTS_STAMP_CACHE[cache_key] = stamp
        return by_kind_symbol, aliases_by_kind_alias


def reference_data_db_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / REFERENCE_DATA_DB_FILENAME


def normalize_instrument_symbol(raw: Any) -> str:
    normalized = str(raw or "").strip().upper()
    if normalized.endswith(".E"):
        normalized = normalized[:-2]
    normalized = re.sub(r"\.[A-Z]{1,4}$", "", normalized)
    return re.sub(r"[^A-Z0-9_/.-]+", "", normalized)


def normalize_instrument_kind(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in {"equity", "local_equity", "hisse"}:
        return "stock"
    if value in {"fon", "byf", "etf"}:
        return "fund"
    return re.sub(r"[^a-z0-9_-]+", "_", value)[:32] or "other"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_json_loads(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _source_priority(source: Any) -> int:
    normalized = str(source or "").strip().lower()
    return _SOURCE_PRIORITY.get(normalized, 50 if normalized else 0)


def _connect(processed_dir: Path) -> Any:
    if database_enabled():
        return connect_postgres()
    path = reference_data_db_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _init_schema(conn)
    return conn


def _init_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instruments (
            kind TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            short_name TEXT,
            source TEXT,
            source_id TEXT,
            logo_url TEXT,
            logo_source TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            as_of TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (kind, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_instruments_symbol
        ON instruments (symbol)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_instruments_kind_active
        ON instruments (kind, active)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instrument_aliases (
            alias TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            symbol TEXT NOT NULL,
            source TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_instrument_aliases_target
        ON instrument_aliases (kind, symbol)
        """
    )
    if getattr(conn, "is_postgres", False):
        lock_down_postgres_tables(conn, "instruments", "instrument_aliases")
    conn.commit()


def ensure_reference_data_schema(processed_dir: Path) -> None:
    """Create the remote reference-data schema once during app bootstrap."""

    del processed_dir  # The Postgres schema is shared; the path is SQLite-only.
    global _REFERENCE_SCHEMA_READY
    if not database_enabled() or _REFERENCE_SCHEMA_READY:
        return
    with _REFERENCE_SCHEMA_LOCK:
        if _REFERENCE_SCHEMA_READY:
            return
        with connect_postgres() as conn:
            _init_schema(conn)
        _REFERENCE_SCHEMA_READY = True


def _normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = normalize_instrument_kind(record.get("kind"))
    symbol = normalize_instrument_symbol(record.get("symbol"))
    if not kind or not symbol:
        return None
    normalized: Dict[str, Any] = {
        "kind": kind,
        "symbol": symbol,
        "name": str(record.get("name") or "").strip() or None,
        "short_name": str(record.get("short_name") or "").strip() or None,
        "source": str(record.get("source") or "").strip() or None,
        "source_id": str(record.get("source_id") or "").strip() or None,
        "logo_url": str(record.get("logo_url") or "").strip() or None,
        "logo_source": str(record.get("logo_source") or "").strip() or None,
        "active": 0 if record.get("active") is False else 1,
        "as_of": str(record.get("as_of") or "").strip() or None,
        "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
        "aliases": [
            normalize_instrument_symbol(alias)
            for alias in (record.get("aliases") or [])
            if normalize_instrument_symbol(alias) and normalize_instrument_symbol(alias) != symbol
        ],
    }
    return normalized


def _merge_record(existing: Optional[sqlite3.Row], incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not existing:
        return dict(incoming)

    existing_priority = _source_priority(existing["source"])
    incoming_priority = _source_priority(incoming.get("source"))
    incoming_can_override = incoming_priority >= existing_priority
    merged = dict(incoming)
    for key in ("name", "short_name", "source_id", "logo_url", "logo_source", "as_of"):
        existing_value = existing[key]
        incoming_value = incoming.get(key)
        if existing_value and (not incoming_value or not incoming_can_override):
            merged[key] = existing_value
    if existing["source"] and (not incoming.get("source") or not incoming_can_override):
        merged["source"] = existing["source"]
    existing_meta = _safe_json_loads(existing["metadata_json"])
    incoming_meta = dict(incoming.get("metadata") or {})
    merged["metadata"] = {**existing_meta, **incoming_meta}
    return merged


def _merge_normalized_records(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge duplicate records before sending a PostgreSQL batch upsert.

    PostgreSQL rejects a single ``INSERT ... ON CONFLICT DO UPDATE`` statement
    when its VALUES list contains the same constrained key more than once.
    Cache bootstrap data can legitimately contain duplicate fund or stock
    rows, so collapse them in memory first while preserving source priority.
    """

    existing_priority = _source_priority(existing.get("source"))
    incoming_priority = _source_priority(incoming.get("source"))
    incoming_can_override = incoming_priority >= existing_priority
    merged = dict(incoming)
    for key in ("name", "short_name", "source_id", "logo_url", "logo_source", "as_of"):
        existing_value = existing.get(key)
        incoming_value = incoming.get(key)
        if existing_value and (not incoming_value or not incoming_can_override):
            merged[key] = existing_value
    if existing.get("source") and (not incoming.get("source") or not incoming_can_override):
        merged["source"] = existing["source"]
    merged["metadata"] = {
        **dict(existing.get("metadata") or {}),
        **dict(incoming.get("metadata") or {}),
    }
    merged["aliases"] = list(
        dict.fromkeys(
            [
                *list(existing.get("aliases") or []),
                *list(incoming.get("aliases") or []),
            ]
        )
    )
    return merged


def _update_instruments_cache_incremental(
    processed_dir: Path,
    upserted_instruments: List[Dict[str, Any]],
    upserted_aliases: List[tuple[str, str, str]],
) -> None:
    if not upserted_instruments and not upserted_aliases:
        return
    cache_key = _instruments_cache_key(processed_dir)
    with _INSTRUMENTS_CACHE_LOCK:
        cached = _INSTRUMENTS_CACHE.get(cache_key)
        if cached is None:
            return
        _, by_kind_symbol, aliases_by_kind_alias = cached
        for inst in upserted_instruments:
            norm_kind = normalize_instrument_kind(inst.get("kind"))
            norm_symbol = normalize_instrument_symbol(inst.get("symbol"))
            if norm_kind and norm_symbol:
                by_kind_symbol[(norm_kind, norm_symbol)] = _copy_instrument_dict(inst)
        for alias, kind, symbol in upserted_aliases:
            norm_kind = normalize_instrument_kind(kind)
            norm_alias = normalize_instrument_symbol(alias)
            norm_symbol = normalize_instrument_symbol(symbol)
            if norm_kind and norm_alias and norm_symbol:
                aliases_by_kind_alias[(norm_kind, norm_alias)] = norm_symbol
        if _INSTRUMENTS_CACHE_TTL_SECONDS > 0:
            _INSTRUMENTS_CACHE[cache_key] = (
                time.monotonic() + _INSTRUMENTS_CACHE_TTL_SECONDS,
                by_kind_symbol,
                aliases_by_kind_alias,
            )


def _instrument_row_matches_existing(
    existing: Any,
    row: Dict[str, Any],
    metadata_json: str,
) -> bool:
    if existing is None:
        return False
    existing_active = int(existing["active"]) if existing["active"] is not None else 1
    return (
        existing["name"] == row.get("name")
        and existing["short_name"] == row.get("short_name")
        and existing["source"] == row.get("source")
        and existing["source_id"] == row.get("source_id")
        and existing["logo_url"] == row.get("logo_url")
        and existing["logo_source"] == row.get("logo_source")
        and existing_active == int(row.get("active", 1))
        and existing["as_of"] == row.get("as_of")
        and (existing["metadata_json"] or "{}") == metadata_json
    )


def upsert_instruments(
    processed_dir: Path,
    records: Iterable[Dict[str, Any]],
    *,
    seed_manual: bool = True,
) -> Dict[str, Any]:
    normalized_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        normalized = _normalize_record(record)
        if not normalized:
            continue
        key = (str(normalized["kind"]), str(normalized["symbol"]))
        previous = normalized_by_key.get(key)
        normalized_by_key[key] = (
            _merge_normalized_records(previous, normalized) if previous else normalized
        )
    normalized_records = list(normalized_by_key.values())
    if seed_manual:
        seed_manual_instruments(processed_dir)
    if not normalized_records:
        return {"db_path": str(reference_data_db_path(processed_dir)), "upserted_count": 0}

    now = _utc_now_iso()
    with _connect(processed_dir) as conn:
        # A fund catalog refresh can contain thousands of rows.  The SQLite
        # path intentionally keeps its simple row-by-row behavior, but doing a
        # SELECT and INSERT round-trip for every row through Supabase's pooler
        # makes a refresh take minutes.  Read existing rows in small batches,
        # merge priorities in memory, then send the upserts with executemany.
        # This keeps the same conflict/metadata semantics without making the
        # user-facing refresh job wait on thousands of network round-trips.
        if getattr(conn, "is_postgres", False):
            existing_by_key: Dict[tuple[str, str], Any] = {}
            symbols_by_kind: Dict[str, List[str]] = {}
            for incoming in normalized_records:
                symbols_by_kind.setdefault(str(incoming["kind"]), []).append(str(incoming["symbol"]))
            for kind, symbols in symbols_by_kind.items():
                unique_symbols = list(dict.fromkeys(symbols))
                for offset in range(0, len(unique_symbols), 500):
                    chunk = unique_symbols[offset : offset + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT {_INSTRUMENT_SELECT_COLUMNS} FROM instruments WHERE kind = ? AND symbol IN ({placeholders})",
                        (kind, *chunk),
                    ).fetchall()
                    for existing in rows:
                        existing_by_key[(str(existing["kind"]), str(existing["symbol"]))] = existing

            instrument_rows: List[tuple[Any, ...]] = []
            upserted_dicts: List[Dict[str, Any]] = []
            alias_by_key: Dict[str, tuple[Any, ...]] = {}
            for incoming in normalized_records:
                existing = existing_by_key.get((incoming["kind"], incoming["symbol"]))
                row = _merge_record(existing, incoming)
                created_at = existing["created_at"] if existing else now
                metadata_dict = dict(row.get("metadata") or {})
                metadata_json = _stable_json(metadata_dict)
                if not _instrument_row_matches_existing(existing, row, metadata_json):
                    instrument_rows.append(
                        (
                            row["kind"],
                            row["symbol"],
                            row.get("name"),
                            row.get("short_name"),
                            row.get("source"),
                            row.get("source_id"),
                            row.get("logo_url"),
                            row.get("logo_source"),
                            int(row.get("active", 1)),
                            row.get("as_of"),
                            metadata_json,
                            created_at,
                            now,
                        )
                    )
                    upserted_dicts.append(
                        {
                            "kind": row["kind"],
                            "symbol": row["symbol"],
                            "name": row.get("name"),
                            "short_name": row.get("short_name"),
                            "source": row.get("source"),
                            "source_id": row.get("source_id"),
                            "logo_url": row.get("logo_url"),
                            "logo_source": row.get("logo_source"),
                            "active": bool(row.get("active", 1)),
                            "as_of": row.get("as_of"),
                            "metadata": metadata_dict,
                            "created_at": created_at,
                            "updated_at": now,
                        }
                    )
                for alias in row.get("aliases") or []:
                    alias_by_key[str(alias)] = (alias, row["kind"], row["symbol"], row.get("source"), now)

            alias_rows: List[tuple[Any, ...]] = []
            if alias_by_key:
                existing_aliases: Dict[str, tuple[Any, ...]] = {}
                alias_keys = list(alias_by_key.keys())
                for offset in range(0, len(alias_keys), 500):
                    chunk = alias_keys[offset : offset + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    for existing_alias in conn.execute(
                        f"SELECT alias, kind, symbol, source FROM instrument_aliases WHERE alias IN ({placeholders})",
                        tuple(chunk),
                    ).fetchall():
                        existing_aliases[str(existing_alias["alias"])] = (
                            str(existing_alias["alias"]),
                            str(existing_alias["kind"]),
                            str(existing_alias["symbol"]),
                            existing_alias["source"],
                        )
                alias_rows = [
                    candidate
                    for alias, candidate in alias_by_key.items()
                    if existing_aliases.get(alias) != candidate[:4]
                ]

            instrument_insert_sql = """
                INSERT INTO instruments (
                    kind, symbol, name, short_name, source, source_id,
                    logo_url, logo_source, active, as_of, metadata_json,
                    created_at, updated_at
                )
                VALUES {values}
                ON CONFLICT(kind, symbol) DO UPDATE SET
                    name = excluded.name,
                    short_name = excluded.short_name,
                    source = excluded.source,
                    source_id = excluded.source_id,
                    logo_url = excluded.logo_url,
                    logo_source = excluded.logo_source,
                    active = excluded.active,
                    as_of = excluded.as_of,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """
            for offset in range(0, len(instrument_rows), 500):
                chunk = instrument_rows[offset : offset + 500]
                value_group = "(" + ", ".join("?" for _ in range(13)) + ")"
                conn.execute(
                    instrument_insert_sql.format(values=", ".join(value_group for _ in chunk)),
                    [value for row in chunk for value in row],
                )
            if alias_rows:
                alias_insert_sql = """
                    INSERT INTO instrument_aliases (alias, kind, symbol, source, updated_at)
                    VALUES {values}
                    ON CONFLICT(alias) DO UPDATE SET
                        kind = excluded.kind,
                        symbol = excluded.symbol,
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """
                for offset in range(0, len(alias_rows), 500):
                    chunk = alias_rows[offset : offset + 500]
                    value_group = "(" + ", ".join("?" for _ in range(5)) + ")"
                    conn.execute(
                        alias_insert_sql.format(values=", ".join(value_group for _ in chunk)),
                        [value for row in chunk for value in row],
                    )
            if instrument_rows or alias_rows:
                conn.commit()
                _update_instruments_cache_incremental(
                    processed_dir,
                    upserted_dicts,
                    [(str(a[0]), str(a[1]), str(a[2])) for a in alias_rows],
                )
            return {
                "db_path": str(reference_data_db_path(processed_dir)),
                "upserted_count": len(instrument_rows),
                "alias_count": len(alias_rows),
            }

        upserted = 0
        alias_count = 0
        sqlite_upserted_dicts: List[Dict[str, Any]] = []
        sqlite_upserted_aliases: List[tuple[str, str, str]] = []
        for incoming in normalized_records:
            existing = conn.execute(
                f"SELECT {_INSTRUMENT_SELECT_COLUMNS} FROM instruments WHERE kind = ? AND symbol = ?",
                (incoming["kind"], incoming["symbol"]),
            ).fetchone()
            row = _merge_record(existing, incoming)
            created_at = existing["created_at"] if existing else now
            metadata_dict = dict(row.get("metadata") or {})
            conn.execute(
                """
                INSERT INTO instruments (
                    kind, symbol, name, short_name, source, source_id,
                    logo_url, logo_source, active, as_of, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, symbol) DO UPDATE SET
                    name = excluded.name,
                    short_name = excluded.short_name,
                    source = excluded.source,
                    source_id = excluded.source_id,
                    logo_url = excluded.logo_url,
                    logo_source = excluded.logo_source,
                    active = excluded.active,
                    as_of = excluded.as_of,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    row["kind"],
                    row["symbol"],
                    row.get("name"),
                    row.get("short_name"),
                    row.get("source"),
                    row.get("source_id"),
                    row.get("logo_url"),
                    row.get("logo_source"),
                    int(row.get("active", 1)),
                    row.get("as_of"),
                    _stable_json(metadata_dict),
                    created_at,
                    now,
                ),
            )
            sqlite_upserted_dicts.append(
                {
                    "kind": row["kind"],
                    "symbol": row["symbol"],
                    "name": row.get("name"),
                    "short_name": row.get("short_name"),
                    "source": row.get("source"),
                    "source_id": row.get("source_id"),
                    "logo_url": row.get("logo_url"),
                    "logo_source": row.get("logo_source"),
                    "active": bool(row.get("active", 1)),
                    "as_of": row.get("as_of"),
                    "metadata": metadata_dict,
                    "created_at": created_at,
                    "updated_at": now,
                }
            )
            for alias in row.get("aliases") or []:
                conn.execute(
                    """
                    INSERT INTO instrument_aliases (alias, kind, symbol, source, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(alias) DO UPDATE SET
                        kind = excluded.kind,
                        symbol = excluded.symbol,
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (alias, row["kind"], row["symbol"], row.get("source"), now),
                )
                sqlite_upserted_aliases.append((str(alias), str(row["kind"]), str(row["symbol"])))
                alias_count += 1
            upserted += 1
        conn.commit()
    _update_instruments_cache_incremental(processed_dir, sqlite_upserted_dicts, sqlite_upserted_aliases)
    return {
        "db_path": str(reference_data_db_path(processed_dir)),
        "upserted_count": upserted,
        "alias_count": alias_count,
    }


def upsert_instrument(processed_dir: Path, **record: Any) -> Dict[str, Any]:
    return upsert_instruments(processed_dir, [record])


def seed_manual_instruments(processed_dir: Path) -> None:
    path_key = str(Path(processed_dir).resolve())
    if path_key in _SEEDED_PROCESSED_DIRS:
        return
    _SEEDED_PROCESSED_DIRS.add(path_key)
    upsert_instruments(processed_dir, _MANUAL_INSTRUMENTS, seed_manual=False)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    result = dict(row)
    result["active"] = bool(result.get("active"))
    result["metadata"] = _safe_json_loads(result.pop("metadata_json", "{}"))
    return result


def get_instrument(processed_dir: Path, kind: str, symbol: Any) -> Optional[Dict[str, Any]]:
    seed_manual_instruments(processed_dir)
    normalized_kind = normalize_instrument_kind(kind)
    normalized_symbol = normalize_instrument_symbol(symbol)
    if not normalized_kind or not normalized_symbol:
        return None
    by_kind_symbol, aliases_by_kind_alias = _ensure_instruments_cache(processed_dir)
    instrument = by_kind_symbol.get((normalized_kind, normalized_symbol))
    if instrument is None:
        target_symbol = aliases_by_kind_alias.get((normalized_kind, normalized_symbol))
        if target_symbol:
            instrument = by_kind_symbol.get((normalized_kind, target_symbol))
    return _copy_instrument_dict(instrument) if instrument else None


def get_instruments(processed_dir: Path, kind: str, symbols: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    seed_manual_instruments(processed_dir)
    normalized_kind = normalize_instrument_kind(kind)
    requested_symbols: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized_symbol = normalize_instrument_symbol(symbol)
        if not normalized_symbol or normalized_symbol in seen:
            continue
        seen.add(normalized_symbol)
        requested_symbols.append(normalized_symbol)
    if not normalized_kind or not requested_symbols:
        return {}

    by_kind_symbol, aliases_by_kind_alias = _ensure_instruments_cache(processed_dir)
    result: Dict[str, Dict[str, Any]] = {}
    for requested_symbol in requested_symbols:
        instrument = by_kind_symbol.get((normalized_kind, requested_symbol))
        if instrument is None:
            target_symbol = aliases_by_kind_alias.get((normalized_kind, requested_symbol))
            if target_symbol:
                instrument = by_kind_symbol.get((normalized_kind, target_symbol))
        if instrument is not None:
            result[requested_symbol] = _copy_instrument_dict(instrument)
    return result


def get_instrument_name(processed_dir: Path, kind: str, symbol: Any) -> Optional[str]:
    instrument = get_instrument(processed_dir, kind, symbol)
    if not instrument:
        return None
    return str(instrument.get("name") or instrument.get("short_name") or "").strip() or None


def get_instrument_names(processed_dir: Path, kind: str) -> Dict[str, str]:
    seed_manual_instruments(processed_dir)
    normalized_kind = normalize_instrument_kind(kind)
    if not normalized_kind:
        return {}
    by_kind_symbol, _ = _ensure_instruments_cache(processed_dir)
    result: Dict[str, str] = {}
    for (row_kind, symbol), instrument in by_kind_symbol.items():
        if row_kind != normalized_kind or not instrument.get("active"):
            continue
        name = str(instrument.get("name") or instrument.get("short_name") or "").strip()
        if symbol and name:
            result[symbol] = name
    return result


def reset_reference_data_state_for_tests() -> None:
    global _REFERENCE_SCHEMA_READY
    _SEEDED_PROCESSED_DIRS.clear()
    _BOOTSTRAPPED_PROCESSED_DIRS.clear()
    _REFERENCE_SCHEMA_READY = False
    clear_instruments_cache()


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _fund_record_from_cache_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    symbol = normalize_instrument_symbol(row.get("fund_code") or row.get("fonKodu"))
    if not symbol:
        return None
    name = str(row.get("name") or row.get("fonUnvan") or "").strip()
    return {
        "kind": "fund",
        "symbol": symbol,
        "name": name or None,
        "short_name": symbol,
        "source": str(row.get("source") or "legacy_json"),
        "as_of": str(row.get("as_of") or row.get("date") or row.get("tarih") or "").strip() or None,
        "active": row.get("tefas_open") is not False,
        "metadata": {
            "fund_type": row.get("fund_type"),
            "founder_company": row.get("founder_company"),
            "manager_company": row.get("manager_company"),
            "risk_value": row.get("risk_value"),
            "aum": row.get("aum"),
        },
    }


def _stock_record_from_kap_cache(symbol_hint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    symbol = normalize_instrument_symbol(payload.get("stock_code") or payload.get("company") or symbol_hint)
    if not symbol:
        return None
    title = str(payload.get("company_title") or payload.get("title") or payload.get("companyName") or "").strip()
    member_oid = str(payload.get("member_oid") or payload.get("mkk_member_oid") or "").strip()
    hint = normalize_instrument_symbol(symbol_hint)
    return {
        "kind": "stock",
        "symbol": symbol,
        "name": title or None,
        "short_name": symbol,
        "source": "kap_cache",
        "source_id": member_oid or None,
        "logo_url": f"https://www.kap.org.tr/tr/api/member/logo/{member_oid}" if member_oid else None,
        "logo_source": "kap" if member_oid else None,
        "as_of": str(payload.get("fetched_at") or "").strip() or None,
        "aliases": [hint] if hint and hint != symbol else [],
        "metadata": {"source_url": payload.get("source_url")},
    }


def sync_reference_data_from_caches(processed_dir: Path) -> Dict[str, Any]:
    path_key = str(Path(processed_dir).resolve())
    if path_key in _BOOTSTRAPPED_PROCESSED_DIRS:
        return {"db_path": str(reference_data_db_path(processed_dir)), "skipped": True}
    _BOOTSTRAPPED_PROCESSED_DIRS.add(path_key)

    need_funds = True
    need_stocks = True
    if database_enabled():
        try:
            with _connect(processed_dir) as conn:
                count_rows = conn.execute(
                    "SELECT kind, COUNT(*) AS row_count FROM instruments GROUP BY kind"
                ).fetchall()
            counts = {
                str(r.get("kind") if isinstance(r, dict) else r["kind"]): int(
                    (r.get("row_count") if isinstance(r, dict) else r["row_count"]) or 0
                )
                for r in count_rows
            }
            need_funds = counts.get("fund", 0) < 100
            need_stocks = counts.get("stock", 0) < 50
        except Exception:
            need_funds = True
            need_stocks = True

    if not need_funds and not need_stocks:
        seed_manual_instruments(processed_dir)
        return {
            "db_path": str(reference_data_db_path(processed_dir)),
            "upserted_count": 0,
            "skipped": True,
            "record_count": 0,
        }

    records: List[Dict[str, Any]] = []
    if need_funds:
        # The fund catalogue is stored as normalized rows after the snapshot-store
        # migration, so do not depend on a monolithic JSON file here.
        from app.fund_service import load_funds_snapshot

        funds_payload = load_funds_snapshot(processed_dir)
        rows = funds_payload.get("rows") if isinstance(funds_payload, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    record = _fund_record_from_cache_row(row)
                    if record:
                        records.append(record)

    if need_stocks:
        kap_dir = Path(processed_dir) / "kap_cache"
        if kap_dir.exists():
            for path in kap_dir.glob("*.json"):
                payload = _read_json_dict(path)
                if not payload:
                    continue
                record = _stock_record_from_kap_cache(path.stem, payload)
                if record:
                    records.append(record)

    result = upsert_instruments(processed_dir, records)
    result["skipped"] = False
    result["record_count"] = len(records)
    return result
