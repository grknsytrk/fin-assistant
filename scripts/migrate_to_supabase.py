"""Copy the existing local caches into the configured Supabase database.

Usage (PowerShell):

    $env:RAGFIN_DATABASE_URL = 'postgresql://...'
    python scripts/migrate_to_supabase.py

The URL is intentionally read from the environment so database credentials do
not end up in shell history, source code, or git.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import connect_postgres, database_enabled, ensure_json_cache_schema, write_json_cache
from app.fund_service import _init_fund_prices_schema
from app.reference_data import _init_schema


PROCESSED_DIR = ROOT / "data" / "processed"


def _copy_json_caches() -> int:
    count = 0
    for path in sorted(PROCESSED_DIR.rglob("*.json")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            write_json_cache(path, payload)
            count += 1
    return count


def _sqlite_rows(path: Path, table: str, columns: Iterable[str]) -> list[tuple[Any, ...]]:
    if not path.exists():
        return []
    selected = ", ".join(columns)
    with sqlite3.connect(path) as conn:
        return [tuple(row) for row in conn.execute(f"SELECT {selected} FROM {table}").fetchall()]


def _insert_rows(conn: Any, table: str, columns: list[str], rows: list[tuple[Any, ...]]) -> int:
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    names = ", ".join(columns)
    conn.executemany(
        f"INSERT INTO {table} ({names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
        rows,
    )
    return len(rows)


def _copy_sqlite_databases() -> dict[str, int]:
    fund_db = PROCESSED_DIR / "fund_prices.sqlite3"
    reference_db = PROCESSED_DIR / "reference_data.sqlite3"
    copied: dict[str, int] = {}
    with connect_postgres() as conn:
        _init_fund_prices_schema(conn)
        copied["fund_prices"] = _insert_rows(
            conn,
            "fund_prices",
            [
                "fund_code", "date", "source", "price", "daily_return", "aum",
                "investor_count", "share_count", "metadata_json", "raw_json",
                "fetched_at", "created_at", "updated_at",
            ],
            _sqlite_rows(
                fund_db,
                "fund_prices",
                [
                    "fund_code", "date", "source", "price", "daily_return", "aum",
                    "investor_count", "share_count", "metadata_json", "raw_json",
                    "fetched_at", "created_at", "updated_at",
                ],
            ),
        )
        copied["fund_price_warnings"] = _insert_rows(
            conn,
            "fund_price_warnings",
            ["fund_code", "date", "source", "warning", "metadata_json", "raw_json", "fetched_at"],
            _sqlite_rows(
                fund_db,
                "fund_price_warnings",
                ["fund_code", "date", "source", "warning", "metadata_json", "raw_json", "fetched_at"],
            ),
        )
        _init_schema(conn)
        copied["instruments"] = _insert_rows(
            conn,
            "instruments",
            [
                "kind", "symbol", "name", "short_name", "source", "source_id", "logo_url",
                "logo_source", "active", "as_of", "metadata_json", "created_at", "updated_at",
            ],
            _sqlite_rows(
                reference_db,
                "instruments",
                [
                    "kind", "symbol", "name", "short_name", "source", "source_id", "logo_url",
                    "logo_source", "active", "as_of", "metadata_json", "created_at", "updated_at",
                ],
            ),
        )
        copied["instrument_aliases"] = _insert_rows(
            conn,
            "instrument_aliases",
            ["alias", "kind", "symbol", "source", "updated_at"],
            _sqlite_rows(
                reference_db,
                "instrument_aliases",
                ["alias", "kind", "symbol", "source", "updated_at"],
            ),
        )
        conn.commit()
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-json", action="store_true", help="do not copy JSON cache documents")
    parser.add_argument("--skip-sqlite", action="store_true", help="do not copy SQLite tables")
    args = parser.parse_args()
    if not database_enabled():
        raise SystemExit("RAGFIN_DATABASE_URL must be set in the environment")

    ensure_json_cache_schema()
    json_count = 0 if args.skip_json else _copy_json_caches()
    sqlite_counts = {} if args.skip_sqlite else _copy_sqlite_databases()
    print(json.dumps({"json_cache_documents": json_count, "sqlite_rows": sqlite_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
