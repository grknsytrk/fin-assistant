from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.database import connect_postgres, database_enabled

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
                        f"SELECT * FROM instruments WHERE kind = ? AND symbol IN ({placeholders})",
                        (kind, *chunk),
                    ).fetchall()
                    for existing in rows:
                        existing_by_key[(str(existing["kind"]), str(existing["symbol"]))] = existing

            instrument_rows: List[tuple[Any, ...]] = []
            alias_by_key: Dict[str, tuple[Any, ...]] = {}
            for incoming in normalized_records:
                existing = existing_by_key.get((incoming["kind"], incoming["symbol"]))
                row = _merge_record(existing, incoming)
                created_at = existing["created_at"] if existing else now
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
                        _stable_json(row.get("metadata") or {}),
                        created_at,
                        now,
                    )
                )
                for alias in row.get("aliases") or []:
                    alias_by_key[str(alias)] = (alias, row["kind"], row["symbol"], row.get("source"), now)

            alias_rows = list(alias_by_key.values())

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
            conn.commit()
            return {
                "db_path": str(reference_data_db_path(processed_dir)),
                "upserted_count": len(instrument_rows),
                "alias_count": len(alias_rows),
            }

        upserted = 0
        alias_count = 0
        for incoming in normalized_records:
            existing = conn.execute(
                "SELECT * FROM instruments WHERE kind = ? AND symbol = ?",
                (incoming["kind"], incoming["symbol"]),
            ).fetchone()
            row = _merge_record(existing, incoming)
            created_at = existing["created_at"] if existing else now
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
                    _stable_json(row.get("metadata") or {}),
                    created_at,
                    now,
                ),
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
                alias_count += 1
            upserted += 1
        conn.commit()
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
    with _connect(processed_dir) as conn:
        row = conn.execute(
            "SELECT * FROM instruments WHERE kind = ? AND symbol = ?",
            (normalized_kind, normalized_symbol),
        ).fetchone()
        if not row:
            alias = conn.execute(
                "SELECT kind, symbol FROM instrument_aliases WHERE alias = ?",
                (normalized_symbol,),
            ).fetchone()
            if alias and alias["kind"] == normalized_kind:
                row = conn.execute(
                    "SELECT * FROM instruments WHERE kind = ? AND symbol = ?",
                    (alias["kind"], alias["symbol"]),
                ).fetchone()
        return _row_to_dict(row) if row else None


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

    placeholders = ",".join("?" for _ in requested_symbols)
    result: Dict[str, Dict[str, Any]] = {}
    missing = set(requested_symbols)
    with _connect(processed_dir) as conn:
        rows = conn.execute(
            f"SELECT * FROM instruments WHERE kind = ? AND symbol IN ({placeholders})",
            (normalized_kind, *requested_symbols),
        ).fetchall()
        for row in rows:
            instrument = _row_to_dict(row)
            symbol = str(instrument.get("symbol") or "")
            if symbol:
                result[symbol] = instrument
                missing.discard(symbol)

        if missing:
            alias_placeholders = ",".join("?" for _ in missing)
            alias_rows = conn.execute(
                f"SELECT alias, kind, symbol FROM instrument_aliases WHERE alias IN ({alias_placeholders})",
                tuple(missing),
            ).fetchall()
            alias_targets = {
                str(row["alias"]): str(row["symbol"])
                for row in alias_rows
                if row["kind"] == normalized_kind and row["symbol"]
            }
            target_symbols = sorted(set(alias_targets.values()))
            if target_symbols:
                target_placeholders = ",".join("?" for _ in target_symbols)
                target_rows = conn.execute(
                    f"SELECT * FROM instruments WHERE kind = ? AND symbol IN ({target_placeholders})",
                    (normalized_kind, *target_symbols),
                ).fetchall()
                target_map: Dict[str, Dict[str, Any]] = {}
                for row in target_rows:
                    instrument = _row_to_dict(row)
                    target_symbol = str(instrument.get("symbol") or "")
                    if target_symbol:
                        target_map[target_symbol] = instrument
                for alias, target_symbol in alias_targets.items():
                    instrument = target_map.get(target_symbol)
                    if instrument:
                        result[alias] = instrument
    return result


def get_instrument_name(processed_dir: Path, kind: str, symbol: Any) -> Optional[str]:
    instrument = get_instrument(processed_dir, kind, symbol)
    if not instrument:
        return None
    return str(instrument.get("name") or instrument.get("short_name") or "").strip() or None


def get_instrument_names(processed_dir: Path, kind: str) -> Dict[str, str]:
    seed_manual_instruments(processed_dir)
    normalized_kind = normalize_instrument_kind(kind)
    with _connect(processed_dir) as conn:
        rows = conn.execute(
            "SELECT symbol, name, short_name FROM instruments WHERE kind = ? AND active = 1",
            (normalized_kind,),
        ).fetchall()
    result: Dict[str, str] = {}
    for row in rows:
        name = str(row["name"] or row["short_name"] or "").strip()
        if row["symbol"] and name:
            result[str(row["symbol"])] = name
    return result


def reset_reference_data_state_for_tests() -> None:
    global _REFERENCE_SCHEMA_READY
    _SEEDED_PROCESSED_DIRS.clear()
    _BOOTSTRAPPED_PROCESSED_DIRS.clear()
    _REFERENCE_SCHEMA_READY = False


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

    records: List[Dict[str, Any]] = []
    funds_payload = _read_json_dict(Path(processed_dir) / "funds_cache" / "funds_latest.json")
    rows = funds_payload.get("rows") if isinstance(funds_payload, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                record = _fund_record_from_cache_row(row)
                if record:
                    records.append(record)

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
