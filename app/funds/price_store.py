"""SQLite/PostgreSQL price history store, LRU/TTL cache, and daily return backfill."""
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


def _init_fund_prices_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fund_prices (
            fund_code TEXT NOT NULL,
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            price REAL NOT NULL CHECK(price > 0),
            daily_return REAL,
            aum REAL,
            investor_count INTEGER,
            share_count REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            raw_json TEXT,
            fetched_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (fund_code, date, source)
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_fund_prices_code_date")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fund_prices_code_date_updated
        ON fund_prices (fund_code, date, updated_at DESC)
        """
    )
    warning_id_type = "BIGSERIAL PRIMARY KEY" if getattr(conn, "is_postgres", False) else "INTEGER PRIMARY KEY AUTOINCREMENT"
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS fund_price_warnings (
            id {warning_id_type},
            fund_code TEXT,
            date TEXT,
            source TEXT NOT NULL,
            warning TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{{}}',
            raw_json TEXT,
            fetched_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fund_price_warnings_code_date
        ON fund_price_warnings (fund_code, date)
        """
    )
    float_type = "DOUBLE PRECISION" if getattr(conn, "is_postgres", False) else "REAL"
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fund_snapshots (
            fund_code TEXT PRIMARY KEY,
            name TEXT,
            fund_type TEXT,
            founder_company TEXT,
            manager_company TEXT,
            price __FLOAT_TYPE__,
            daily_return __FLOAT_TYPE__,
            aum __FLOAT_TYPE__,
            investor_count INTEGER,
            share_count __FLOAT_TYPE__,
            risk_value TEXT,
            risk_value_json TEXT,
            risk_value_num __FLOAT_TYPE__,
            tefas_open INTEGER,
            currency TEXT,
            as_of TEXT,
            source TEXT,
            period_returns_json TEXT NOT NULL DEFAULT '{}',
            attributes_json TEXT NOT NULL DEFAULT '{}',
            present_fields_json TEXT NOT NULL DEFAULT '[]',
            search_text TEXT NOT NULL DEFAULT '',
            fund_type_key TEXT NOT NULL DEFAULT '',
            founder_key TEXT NOT NULL DEFAULT '',
            manager_key TEXT NOT NULL DEFAULT '',
            risk_key TEXT NOT NULL DEFAULT '',
            fund_code_sort TEXT NOT NULL DEFAULT '',
            name_sort TEXT NOT NULL DEFAULT '',
            fund_type_sort TEXT NOT NULL DEFAULT '',
            founder_sort TEXT NOT NULL DEFAULT '',
            manager_sort TEXT NOT NULL DEFAULT '',
            row_order INTEGER NOT NULL DEFAULT 0,
            generation BIGINT NOT NULL DEFAULT 0,
            content_hash TEXT NOT NULL DEFAULT ''
        )
        """.replace("__FLOAT_TYPE__", float_type)
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fund_snapshots_type_aum ON fund_snapshots (fund_type_key, aum)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fund_snapshots_fund_type ON fund_snapshots (fund_type, row_order)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fund_snapshots_aum ON fund_snapshots (aum) WHERE aum IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fund_snapshots_founder ON fund_snapshots (founder_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fund_snapshots_manager ON fund_snapshots (manager_key)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fund_listing_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            catalog_generation BIGINT NOT NULL DEFAULT 0,
            price_generation BIGINT NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO fund_listing_state (
            id, catalog_generation, price_generation, metadata_json, updated_at
        )
        SELECT 1, COALESCE(MAX(generation), 0), 0, '{}', CURRENT_TIMESTAMP
        FROM fund_snapshots
        WHERE TRUE
        ON CONFLICT (id) DO NOTHING
        """
    )
    if getattr(conn, "is_postgres", False):
        _fund_service().lock_down_postgres_tables(
            conn,
            "fund_prices",
            "fund_price_warnings",
            "fund_snapshots",
            "fund_listing_state",
        )
    if not getattr(conn, "is_postgres", False):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_snapshot_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                generation BIGINT NOT NULL DEFAULT 0,
                row_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            )
            """
        )
    conn.commit()


def ensure_fund_prices_schema(processed_dir: Path) -> None:
    """Create the remote fund-price schema once during app bootstrap."""

    del processed_dir  # The Postgres schema is shared; the path is SQLite-only.
    if not _fund_service().database_enabled() or _fund_service()._FUND_PRICES_SCHEMA_READY:
        return
    with _fund_service()._FUND_PRICES_SCHEMA_LOCK:
        if _fund_service()._FUND_PRICES_SCHEMA_READY:
            return
        with _fund_service().connect_postgres() as conn:
            _seam("_init_fund_prices_schema")(conn)
        _fund_service()._FUND_PRICES_SCHEMA_READY = True


def _normalize_history_row(row: Dict[str, Any], fallback_code: str | None = None) -> Optional[Dict[str, Any]]:
    fund_code = _fund_service().normalize_fund_code(
        _fund_service()._first_text(row, "FONKODU", "FONKOD", "FON_CODE", "fund_code", "fonKodu", "fonKod") or fallback_code
    )
    if not fund_code:
        return None
    price = _fund_service()._coerce_float(_fund_service()._first_present(row, "FIYAT", "FONFIYAT", "price", "fiyat", "sonFiyat"))
    point_date = _fund_service()._fund_date(_fund_service()._first_present(row, "TARIH", "TARIHSTR", "date", "tarih"))
    if price is None or point_date is None:
        return None
    name = _fund_service()._first_text(row, "FONUNVAN", "FONUNVANI", "FONADI", "name", "fonUnvan")
    fund_type = _fund_service()._first_text(row, "FONUNVANTIP", "FONUNVANTUR", "FON_TURU", "FONKATEGORI", "fonTurAciklama", "fonKategori")
    founder_company = _fund_service()._first_text(
        row,
        "founder_company",
        "KURUCU",
        "KURUCUNVAN",
        "KURUCUUNVAN",
        "kurucuUnvan",
        "kurucuKodu",
        "kurucuKod",
    ) or _fund_service()._infer_founder_from_fund_name(name)
    manager_company = _fund_service()._first_text(
        row,
        "manager_company",
        "YONETICI",
        "YONETICIUNVAN",
        "PORTFOYYONETICISI",
        "yoneticiUnvan",
    ) or founder_company
    source = _fund_service()._first_text(row, "source", "SOURCE")
    return {
        "fund_code": fund_code,
        "name": name,
        "date": point_date,
        "price": price,
        "daily_return": _fund_service()._coerce_float(_fund_service()._first_present(row, "GUNLUKGETIRI", "gunlukGetiri", "daily_return")),
        "aum": _fund_service()._coerce_float(_fund_service()._first_present(row, "PORTFOYBUYUKLUK", "PORTFOY_BUYUKLUK", "portfoyBuyukluk", "sonPortfoyDegeri", "portBuyukluk")),
        "investor_count": _fund_service()._coerce_int(_fund_service()._first_present(row, "KISISAYISI", "YATIRIMCISAYISI", "kisiSayisi", "yatirimciSayi")),
        "share_count": _fund_service()._coerce_float(_fund_service()._first_present(row, "TEDPAYSAYISI", "PAYADEDI", "tedPaySayisi", "sonPayAdedi", "payAdet")),
        "fund_type": fund_type or _fund_service()._infer_fund_type_from_name(name),
        "founder_company": founder_company,
        "manager_company": manager_company,
        "tefas_open": _fund_service()._tefas_open_status(row),
        "risk_value": _fund_service()._coerce_int(_fund_service()._first_present(row, "RISKDEGERI", "RISKDEGER", "RISK", "riskDegeri")),
        "management_fee_applied": _fund_service()._coerce_tefas_percentage(
            _fund_service()._first_present(row, "management_fee_applied", "uygulananYu1Y")
        ),
        "management_fee_prospectus": _fund_service()._coerce_tefas_percentage(
            _fund_service()._first_present(row, "management_fee_prospectus", "fonIcTuzukYu1G")
        ),
        "total_expense_ratio": _fund_service()._coerce_tefas_percentage(
            _fund_service()._first_present(row, "total_expense_ratio", "fonTopGiderKesoran")
        ),
        "source": _seam("_normalize_price_source")(source) if source else None,
        "source_url": _fund_service()._first_text(row, "source_url", "SOURCE_URL"),
        "raw": row,
    }


def _upsert_fund_reference_data(
    processed_dir: Path,
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        point = _seam("_normalize_history_row")(row) or {
            "fund_code": _fund_service().normalize_fund_code(_fund_service()._first_text(row, "fund_code", "fonKodu", "FONKODU")),
            "name": _fund_service()._first_text(row, "name", "fonUnvan", "FONUNVAN"),
            "date": _fund_service()._fund_date(_fund_service()._first_present(row, "date", "as_of", "tarih", "TARIH")),
            "fund_type": _fund_service()._first_text(row, "fund_type", "fonTurAciklama", "FON_TURU"),
            "founder_company": _fund_service()._first_text(row, "founder_company", "KURUCU", "kurucuUnvan"),
            "manager_company": _fund_service()._first_text(row, "manager_company", "YONETICI", "yoneticiUnvan"),
            "risk_value": _fund_service()._coerce_int(_fund_service()._first_present(row, "risk_value", "riskDegeri", "RISKDEGERI")),
            "tefas_open": _fund_service()._tefas_open_status(row),
            "source": _fund_service()._first_text(row, "source", "SOURCE"),
        }
        if not point.get("fund_code"):
            continue
        # Snapshot rows may have already-coerced fee fields; otherwise pull them
        # straight from the raw tefasfon MB payload.
        management_fee_applied = (
            point.get("management_fee_applied")
            if point.get("management_fee_applied") is not None
            else _fund_service()._coerce_tefas_percentage(_fund_service()._first_present(row, "uygulananYu1Y", "management_fee_applied"))
        )
        management_fee_prospectus = (
            point.get("management_fee_prospectus")
            if point.get("management_fee_prospectus") is not None
            else _fund_service()._coerce_tefas_percentage(_fund_service()._first_present(row, "fonIcTuzukYu1G", "management_fee_prospectus"))
        )
        total_expense_ratio = (
            point.get("total_expense_ratio")
            if point.get("total_expense_ratio") is not None
            else _fund_service()._coerce_tefas_percentage(_fund_service()._first_present(row, "fonTopGiderKesoran", "total_expense_ratio"))
        )
        tax_info = _fund_service()._fund_tax_info(point.get("fund_type"))
        metadata: Dict[str, Any] = {
            "fund_type": point.get("fund_type"),
            "founder_company": point.get("founder_company"),
            "manager_company": point.get("manager_company"),
            "risk_value": point.get("risk_value"),
            "tefas_open": point.get("tefas_open"),
        }
        # Only persist fee values when we actually have them so we don't clobber
        # a previously cached payload with ``None`` on a partial refresh.
        if management_fee_applied is not None:
            metadata["management_fee_applied"] = management_fee_applied
        if management_fee_prospectus is not None:
            metadata["management_fee_prospectus"] = management_fee_prospectus
        if total_expense_ratio is not None:
            metadata["total_expense_ratio"] = total_expense_ratio
        if tax_info is not None:
            metadata["tax_info"] = tax_info
        records.append(
            {
                "kind": "fund",
                "symbol": point["fund_code"],
                "name": point.get("name"),
                "short_name": point["fund_code"],
                "source": point.get("source") or _fund_service().TEFASFON_FUNDS_SOURCE,
                "as_of": point.get("date"),
                "active": point.get("tefas_open") is not False,
                "metadata": metadata,
            }
        )
    return _fund_service().upsert_instruments(processed_dir, records)


def _fund_price_storage_row_to_point(row: Any, *, include_metadata: bool = True) -> Dict[str, Any]:
    metadata = _fund_service()._safe_json_loads(row["metadata_json"], {}) if include_metadata else None
    point = {
        "fund_code": str(row["fund_code"]),
        "date": str(row["date"]),
        "price": float(row["price"]),
        "daily_return": row["daily_return"],
        "aum": row["aum"],
        "investor_count": row["investor_count"],
        "share_count": row["share_count"],
        "source": _seam("_public_price_source")(str(row["source"])),
        "fetched_at": str(row["fetched_at"]),
    }
    if isinstance(metadata, dict):
        for key in (
            "name",
            "fund_type",
            "founder_company",
            "manager_company",
            "tefas_open",
            "risk_value",
            "currency",
        ):
            if key in metadata:
                point[key] = metadata[key]
    return point


def _normalize_price_source(source: str | None) -> str:
    normalized = str(source or "").strip().lower()
    return re.sub(r"[^a-z0-9_.-]+", "_", normalized)[:64] or "unknown"


def _public_price_source(source: str | None) -> str:
    normalized = _seam("_normalize_price_source")(source)
    legacy_sources = {
        "te" + "fas",
        "te" + "fasweb",
        "te" + "fas_history",
        "te" + "fas_html",
        "legacy_te" + "fas",
    }
    if normalized in legacy_sources:
        return "legacy_cache"
    return normalized


def _price_warning_from_row(
    row: Dict[str, Any],
    *,
    source: str,
    fallback_code: str | None = None,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else row
    raw_code = (
        _fund_service()._first_text(raw, "fund_code", "FONKODU", "FONKOD", "fonKodu", "fonKod")
        if isinstance(raw, dict)
        else None
    )
    fund_code = _fund_service().normalize_fund_code(
        _fund_service()._first_text(row, "fund_code", "FONKODU", "FONKOD", "fonKodu", "fonKod")
        or raw_code
        or fallback_code
    )
    point_date = _fund_service()._fund_date(
        _fund_service()._first_present(row, "date", "TARIH", "TARIHSTR", "tarih")
        if isinstance(row, dict)
        else None
    )
    if not point_date and isinstance(raw, dict):
        point_date = _fund_service()._fund_date(_fund_service()._first_present(raw, "date", "TARIH", "TARIHSTR", "tarih"))
    price = _fund_service()._coerce_float(
        _fund_service()._first_present(row, "price", "FIYAT", "FONFIYAT", "fiyat", "sonFiyat")
        if isinstance(row, dict)
        else None
    )
    if price is None and isinstance(raw, dict):
        price = _fund_service()._coerce_float(_fund_service()._first_present(raw, "price", "FIYAT", "FONFIYAT", "fiyat", "sonFiyat"))
    if warning is None:
        if not fund_code:
            warning = "missing_fund_code"
        elif not point_date:
            warning = "missing_date"
        elif price is None:
            warning = "missing_price"
        elif price <= 0:
            warning = "non_positive_price"
        else:
            warning = "invalid_price_row"
    return {
        "fund_code": fund_code or None,
        "date": point_date,
        "source": source,
        "warning": warning,
        "metadata": {"price": price},
        "raw": None,
    }


def _storage_point_from_row(
    row: Dict[str, Any],
    *,
    source: str,
    fallback_code: str | None = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    point = row if {"fund_code", "date", "price"}.issubset(row.keys()) else _seam("_normalize_history_row")(row, fallback_code)
    if not point:
        return None, _seam("_price_warning_from_row")(row, source=source, fallback_code=fallback_code)
    if point.get("tefas_open") is None:
        tefas_open = _fund_service()._tefas_open_status(point)
        if tefas_open is not None:
            point = dict(point)
            point["tefas_open"] = tefas_open

    raw = point.get("raw") if isinstance(point.get("raw"), dict) else row
    fund_code = _fund_service().normalize_fund_code(str(point.get("fund_code") or fallback_code or ""))
    point_date = _fund_service()._fund_date(point.get("date"))
    price = _fund_service()._coerce_float(point.get("price"))
    if not fund_code or not point_date or price is None or price <= 0:
        return None, _seam("_price_warning_from_row")(point, source=source, fallback_code=fallback_code)

    daily_return = _fund_service()._coerce_float(point.get("daily_return"))
    if daily_return is None and isinstance(raw, dict):
        daily_return = _fund_service()._coerce_float(_fund_service()._first_present(raw, "gunlukGetiri", "daily_return"))
    metadata = (
        {"tefas_open": point["tefas_open"]}
        if point.get("tefas_open") is not None
        else {}
    )
    return (
        {
            "fund_code": fund_code,
            "date": point_date,
            "source": source,
            "price": price,
            "daily_return": daily_return,
            "aum": _fund_service()._coerce_float(point.get("aum")),
            "investor_count": _fund_service()._coerce_int(point.get("investor_count")),
            "share_count": _fund_service()._coerce_float(point.get("share_count")),
            "metadata": metadata,
            "raw": None,
        },
        None,
    )


def _insert_fund_price_warning(
    conn: sqlite3.Connection,
    warning: Dict[str, Any],
    *,
    fetched_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO fund_price_warnings (
            fund_code, date, source, warning, metadata_json, raw_json, fetched_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            warning.get("fund_code"),
            warning.get("date"),
            _seam("_normalize_price_source")(str(warning.get("source") or "")),
            str(warning.get("warning") or "invalid_price_row"),
            _fund_service()._stable_json_dumps(warning.get("metadata") or {}),
            _fund_service()._stable_json_dumps(warning.get("raw")) if warning.get("raw") is not None else None,
            fetched_at,
        ),
    )


def upsert_fund_price_points(
    processed_dir: Path,
    points: Iterable[Dict[str, Any]],
    *,
    source: str = _UNSET,
    fetched_at: Optional[str] = None,
    fallback_code: str | None = None,
) -> Dict[str, Any]:
    if source is _UNSET:
        source = _fund_service().FINTABLES_UDF_HISTORY_SOURCE
    normalized_source = _seam("_normalize_price_source")(source)
    effective_fetched_at = fetched_at or _fund_service()._utc_now_iso()
    upserted_count = 0
    skipped_count = 0
    source_counts: Dict[str, int] = {}
    warnings: List[Dict[str, Any]] = []

    if _fund_service().database_enabled():
        # The deployed price store is PostgreSQL.  A catalog refresh contains
        # one current point for roughly two thousand funds; sending those
        # writes one by one through Supavisor turns a fast TEFAS response into
        # a multi-minute job. Prepare the same rows in memory and send them in
        # a few multi-value statements instead.
        prepared_rows_by_key: Dict[Tuple[str, str, str], tuple[Any, ...]] = {}
        warning_rows: List[tuple[Any, ...]] = []
        for row in points:
            if not isinstance(row, dict):
                skipped_count += 1
                warning = {
                    "fund_code": _fund_service().normalize_fund_code(fallback_code),
                    "date": None,
                    "source": normalized_source,
                    "warning": "non_object_price_row",
                    "metadata": {},
                    "raw": row,
                }
                warnings.append(warning)
                warning_rows.append(
                    (
                        warning.get("fund_code") or None,
                        None,
                        normalized_source,
                        warning["warning"],
                        _fund_service()._stable_json_dumps(warning.get("metadata") or {}),
                        _fund_service()._stable_json_dumps(warning.get("raw")) if warning.get("raw") is not None else None,
                        effective_fetched_at,
                    )
                )
                continue
            row_source = _seam("_normalize_price_source")(str(row.get("source") or normalized_source))
            if row_source in _fund_service()._NON_DAILY_PRICE_SOURCES:
                skipped_count += 1
                warning = {
                    "fund_code": _fund_service().normalize_fund_code(str(row.get("fund_code") or fallback_code or "")) or None,
                    "date": _fund_service()._fund_date(row.get("date")),
                    "source": row_source,
                    "warning": "non_daily_price_source",
                    "metadata": {"source": row_source},
                    "raw": row,
                }
                warnings.append(warning)
                warning_rows.append(
                    (
                        warning.get("fund_code"),
                        warning.get("date"),
                        row_source,
                        warning["warning"],
                        _fund_service()._stable_json_dumps(warning.get("metadata") or {}),
                        _fund_service()._stable_json_dumps(warning.get("raw")) if warning.get("raw") is not None else None,
                        effective_fetched_at,
                    )
                )
                continue
            point, warning = _seam("_storage_point_from_row")(
                row,
                source=row_source,
                fallback_code=fallback_code,
            )
            if not point:
                skipped_count += 1
                if warning:
                    warnings.append(warning)
                    warning_rows.append(
                        (
                            warning.get("fund_code") or None,
                            warning.get("date"),
                            row_source,
                            str(warning.get("warning") or "invalid_price_row"),
                            _fund_service()._stable_json_dumps(warning.get("metadata") or {}),
                            _fund_service()._stable_json_dumps(warning.get("raw")) if warning.get("raw") is not None else None,
                            effective_fetched_at,
                        )
                    )
                continue
            prepared_row = (
                point["fund_code"],
                point["date"],
                point["source"],
                point["price"],
                point.get("daily_return"),
                point.get("aum"),
                point.get("investor_count"),
                point.get("share_count"),
                _fund_service()._stable_json_dumps(point.get("metadata") or {}),
                _fund_service()._stable_json_dumps(point.get("raw")) if point.get("raw") is not None else None,
                effective_fetched_at,
                effective_fetched_at,
                effective_fetched_at,
            )
            # A long-range refresh can combine overlapping TEFAS/Fintables
            # windows. PostgreSQL rejects a multi-value INSERT when the same
            # conflict key appears twice in that one statement, so keep the
            # last (most recently merged) row for each fund/date/source key.
            prepared_rows_by_key[(point["fund_code"], point["date"], point["source"])] = prepared_row

        prepared_rows = list(prepared_rows_by_key.values())
        upserted_count = len(prepared_rows)
        source_counts = {}
        for prepared_row in prepared_rows:
            row_source = str(prepared_row[2])
            source_counts[row_source] = source_counts.get(row_source, 0) + 1

        with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
            insert_sql = """
                INSERT INTO fund_prices (
                    fund_code, date, source, price, daily_return, aum, investor_count,
                    share_count, metadata_json, raw_json, fetched_at, created_at, updated_at
                )
                VALUES {values}
                ON CONFLICT(fund_code, date, source) DO UPDATE SET
                    price = excluded.price,
                    daily_return = excluded.daily_return,
                    aum = excluded.aum,
                    investor_count = excluded.investor_count,
                    share_count = excluded.share_count,
                    metadata_json = excluded.metadata_json,
                    raw_json = excluded.raw_json,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                WHERE fund_prices.price IS DISTINCT FROM excluded.price
                   OR fund_prices.daily_return IS DISTINCT FROM excluded.daily_return
                   OR fund_prices.aum IS DISTINCT FROM excluded.aum
                   OR fund_prices.investor_count IS DISTINCT FROM excluded.investor_count
                   OR fund_prices.share_count IS DISTINCT FROM excluded.share_count
                   OR fund_prices.metadata_json IS DISTINCT FROM excluded.metadata_json
                   OR fund_prices.raw_json IS DISTINCT FROM excluded.raw_json
            """
            changed_count = 0
            for offset in range(0, len(prepared_rows), 250):
                chunk = prepared_rows[offset : offset + 250]
                value_group = "(" + ", ".join("?" for _ in range(13)) + ")"
                exec_result = conn.execute(
                    insert_sql.format(values=", ".join(value_group for _ in chunk)),
                    [value for row in chunk for value in row],
                )
                rowcount = getattr(exec_result, "rowcount", None)
                if isinstance(rowcount, int) and rowcount >= 0:
                    changed_count += rowcount
                else:
                    changed_count += len(chunk)
            if warning_rows:
                warning_sql = """
                    INSERT INTO fund_price_warnings (
                        fund_code, date, source, warning, metadata_json, raw_json, fetched_at
                    )
                    VALUES {values}
                """
                for offset in range(0, len(warning_rows), 250):
                    chunk = warning_rows[offset : offset + 250]
                    value_group = "(" + ", ".join("?" for _ in range(7)) + ")"
                    conn.execute(
                        warning_sql.format(values=", ".join(value_group for _ in chunk)),
                        [value for row in chunk for value in row],
                    )
            if changed_count:
                _fund_service()._bump_fund_listing_price_revision(conn)
            conn.commit()
        if changed_count:
            _fund_service()._invalidate_fund_price_memory_caches()
            _seam("invalidate_fund_price_points_cache")()
        return {
            "db_path": str(_fund_service()._fund_prices_db_path(processed_dir)),
            "source": normalized_source,
            "sources": source_counts,
            "upserted_count": upserted_count,
            "skipped_count": skipped_count,
            "warning_count": len(warnings),
            "warnings": warnings[:50],
            "fetched_at": effective_fetched_at,
        }

    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        for row in points:
            if not isinstance(row, dict):
                skipped_count += 1
                warning = {
                    "fund_code": _fund_service().normalize_fund_code(fallback_code),
                    "date": None,
                    "source": normalized_source,
                    "warning": "non_object_price_row",
                    "metadata": {},
                    "raw": row,
                }
                warnings.append(warning)
                _seam("_insert_fund_price_warning")(conn, warning, fetched_at=effective_fetched_at)
                continue
            row_source = _seam("_normalize_price_source")(str(row.get("source") or normalized_source))
            if row_source in _fund_service()._NON_DAILY_PRICE_SOURCES:
                skipped_count += 1
                warning = {
                    "fund_code": _fund_service().normalize_fund_code(str(row.get("fund_code") or fallback_code or "")) or None,
                    "date": _fund_service()._fund_date(row.get("date")),
                    "source": row_source,
                    "warning": "non_daily_price_source",
                    "metadata": {"source": row_source},
                    "raw": row,
                }
                warnings.append(warning)
                _seam("_insert_fund_price_warning")(conn, warning, fetched_at=effective_fetched_at)
                continue
            point, warning = _seam("_storage_point_from_row")(
                row,
                source=row_source,
                fallback_code=fallback_code,
            )
            if not point:
                skipped_count += 1
                if warning:
                    warnings.append(warning)
                    _seam("_insert_fund_price_warning")(conn, warning, fetched_at=effective_fetched_at)
                continue
            conn.execute(
                """
                INSERT INTO fund_prices (
                    fund_code, date, source, price, daily_return, aum, investor_count,
                    share_count, metadata_json, raw_json, fetched_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fund_code, date, source) DO UPDATE SET
                    price = excluded.price,
                    daily_return = excluded.daily_return,
                    aum = excluded.aum,
                    investor_count = excluded.investor_count,
                    share_count = excluded.share_count,
                    metadata_json = excluded.metadata_json,
                    raw_json = excluded.raw_json,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                """,
                (
                    point["fund_code"],
                    point["date"],
                    point["source"],
                    point["price"],
                    point.get("daily_return"),
                    point.get("aum"),
                    point.get("investor_count"),
                    point.get("share_count"),
                    _fund_service()._stable_json_dumps(point.get("metadata") or {}),
                    _fund_service()._stable_json_dumps(point.get("raw")) if point.get("raw") is not None else None,
                    effective_fetched_at,
                    effective_fetched_at,
                    effective_fetched_at,
                ),
            )
            upserted_count += 1
            source_counts[point["source"]] = source_counts.get(point["source"], 0) + 1
        if upserted_count:
            _fund_service()._bump_fund_listing_price_revision(conn)
        conn.commit()
    if upserted_count:
        _fund_service()._invalidate_fund_price_memory_caches()
        _seam("invalidate_fund_price_points_cache")()
    return {
        "db_path": str(_fund_service()._fund_prices_db_path(processed_dir)),
        "source": normalized_source,
        "sources": source_counts,
        "upserted_count": upserted_count,
        "skipped_count": skipped_count,
        "warning_count": len(warnings),
        "warnings": warnings[:50],
        "fetched_at": effective_fetched_at,
    }


def _effective_fund_price_points_cache_ttl_seconds() -> int:
    ttl = int(_fund_service().FUND_PRICE_POINTS_CACHE_TTL_SECONDS)
    if ttl <= 0:
        return 0
    if _fund_service().database_enabled() and ttl >= 600:
        # Generation-fenced keys invalidate via fund_listing_revision when new prices land,
        # so warm Postgres workers can safely hold historical series for 6 hours.
        return max(21600, ttl)
    return ttl


def _price_source_priority_sql(column: str = "source") -> Tuple[str, List[Any]]:
    priority_items = list(_fund_service()._FUND_PRICE_SOURCE_PRIORITY.items())
    if not priority_items:
        return "0", []
    priority_cases = " ".join(f"WHEN ? THEN {int(priority)}" for _source, priority in priority_items)
    priority_params: List[Any] = [source for source, _priority in priority_items]
    return f"CASE {column} {priority_cases} ELSE 0 END", priority_params


def _fund_price_points_cache_key(
    processed_dir: Path,
    normalized: str,
    start_date: Optional[date],
    end_date: Optional[date],
    normalized_sources: Tuple[str, ...],
) -> Tuple[Any, ...]:
    if _fund_service().database_enabled():
        try:
            price_gen = int(_fund_service().fund_listing_revision(processed_dir).get("price_generation") or 0)
        except Exception:
            price_gen = 0
        scope: Any = f"postgres:gen={price_gen}"
    else:
        scope = str(_fund_service()._fund_prices_db_path(processed_dir))
    return (
        scope,
        normalized,
        start_date.isoformat() if start_date else None,
        end_date.isoformat() if end_date else None,
        normalized_sources,
    )


def _remove_fund_price_points_cache_entry_locked(key: Any) -> None:
    entry = _fund_service()._FUND_PRICE_POINTS_CACHE.pop(key, None)
    if entry is not None:
        _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES = max(
            0,
            _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES - entry.approx_bytes,
        )


def _purge_expired_fund_price_points_cache_locked(now: float) -> None:
    for key in list(_fund_service()._FUND_PRICE_POINTS_CACHE):
        entry = _fund_service()._FUND_PRICE_POINTS_CACHE.get(key)
        if entry is not None and entry.expires_at <= now:
            _seam("_remove_fund_price_points_cache_entry_locked")(key)


def _store_fund_price_points_cache_locked(
    key: Any,
    payload: List[Dict[str, Any]],
    approx_bytes: int,
    now: float,
) -> None:
    _seam("_remove_fund_price_points_cache_entry_locked")(key)
    _seam("_purge_expired_fund_price_points_cache_locked")(now)
    _fund_service()._FUND_PRICE_POINTS_CACHE[key] = _fund_service()._FundPricePointsCacheEntry(
        payload=payload,
        approx_bytes=max(0, int(approx_bytes)),
        expires_at=now + _effective_fund_price_points_cache_ttl_seconds(),
    )
    _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES += max(0, int(approx_bytes))
    while (
        len(_fund_service()._FUND_PRICE_POINTS_CACHE) > _fund_service().FUND_PRICE_POINTS_CACHE_MAX_ENTRIES
        or _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES > _fund_service().FUND_PRICE_POINTS_CACHE_MAX_TOTAL_BYTES
    ):
        _oldest_key, oldest = _fund_service()._FUND_PRICE_POINTS_CACHE.popitem(last=False)
        _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES = max(
            0,
            _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES - oldest.approx_bytes,
        )
        _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS["evictions"] += 1


def _fund_price_points_cache_metrics_snapshot_locked(
    now: float,
) -> Optional[Tuple[float, Dict[str, int]]]:
    elapsed = now - _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS_SINCE
    if elapsed < _fund_service().FUND_PRICE_POINTS_CACHE_METRICS_FLUSH_SECONDS:
        return None
    snapshot = (
        dict(_fund_service()._FUND_PRICE_POINTS_CACHE_METRICS)
        if any(_fund_service()._FUND_PRICE_POINTS_CACHE_METRICS.values())
        else None
    )
    if snapshot is not None:
        snapshot["entries"] = len(_fund_service()._FUND_PRICE_POINTS_CACHE)
        snapshot["approx_bytes"] = _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES
        for key in _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS:
            _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS[key] = 0
    _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS_SINCE = now
    return (elapsed, snapshot) if snapshot is not None else None


def _log_fund_price_points_cache_metrics(
    pending: Optional[Tuple[float, Dict[str, int]]],
) -> None:
    if pending is None:
        return
    elapsed, snapshot = pending
    _fund_service().LOGGER.info(
        "fund price cache window_s=%.0f hits=%d misses=%d coalesced_hits=%d "
        "coalesced_errors=%d evictions=%d oversize_skips=%d invalidations=%d "
        "entries=%d approx_kib=%.1f",
        elapsed,
        snapshot["hits"],
        snapshot["misses"],
        snapshot["coalesced_hits"],
        snapshot["coalesced_errors"],
        snapshot["evictions"],
        snapshot["oversize_skips"],
        snapshot["invalidations"],
        snapshot["entries"],
        snapshot["approx_bytes"] / 1024.0,
    )


def invalidate_fund_price_points_cache() -> None:
    """Drop all in-process price history entries after a successful write."""

    with _fund_service()._FUND_PRICE_POINTS_CACHE_LOCK:
        _fund_service()._FUND_PRICE_POINTS_CACHE.clear()
        _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES = 0
        _fund_service()._FUND_PRICE_POINTS_CACHE_GENERATION += 1
        _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS["invalidations"] += 1
        pending = _seam("_fund_price_points_cache_metrics_snapshot_locked")(time.monotonic())
    _seam("_log_fund_price_points_cache_metrics")(pending)


def _reset_fund_price_points_cache_for_tests() -> None:
    with _fund_service()._FUND_PRICE_POINTS_CACHE_LOCK:
        _fund_service()._FUND_PRICE_POINTS_CACHE.clear()
        _fund_service()._FUND_PRICE_POINTS_IN_FLIGHT.clear()
        _fund_service()._FUND_PRICE_POINTS_CACHE_BYTES = 0
        _fund_service()._FUND_PRICE_POINTS_CACHE_GENERATION += 1
        for key in _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS:
            _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS[key] = 0
        _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS_SINCE = time.monotonic()


def _read_fund_price_points_cached(
    key: Tuple[Any, ...],
    loader: Callable[[], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    ttl_seconds = _effective_fund_price_points_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return loader()

    while True:
        now = time.monotonic()
        pending: Optional[Tuple[float, Dict[str, int]]] = None
        entry: Optional[_FundPricePointsCacheEntry] = None
        inflight: Optional[_FundPricePointsReadInFlight] = None
        leader = False
        generation = 0

        with _fund_service()._FUND_PRICE_POINTS_CACHE_LOCK:
            entry = _fund_service()._FUND_PRICE_POINTS_CACHE.get(key)
            if entry is not None:
                if entry.expires_at > now:
                    _fund_service()._FUND_PRICE_POINTS_CACHE.move_to_end(key)
                    _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS["hits"] += 1
                    pending = _seam("_fund_price_points_cache_metrics_snapshot_locked")(now)
                else:
                    _seam("_remove_fund_price_points_cache_entry_locked")(key)
                    entry = None

            if entry is None:
                current = _fund_service()._FUND_PRICE_POINTS_IN_FLIGHT.get(key)
                if current is None:
                    current = _fund_service()._FundPricePointsReadInFlight()
                    _fund_service()._FUND_PRICE_POINTS_IN_FLIGHT[key] = current
                    leader = True
                    generation = _fund_service()._FUND_PRICE_POINTS_CACHE_GENERATION
                    _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS["misses"] += 1
                    pending = _seam("_fund_price_points_cache_metrics_snapshot_locked")(now)
                inflight = current

        _seam("_log_fund_price_points_cache_metrics")(pending)

        if entry is not None:
            return copy.deepcopy(entry.payload)

        if inflight is None:  # pragma: no cover - defensive
            continue

        if not leader:
            inflight.event.wait()
            if inflight.error is not None:
                with _fund_service()._FUND_PRICE_POINTS_CACHE_LOCK:
                    _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS["coalesced_errors"] += 1
                    pending = _seam("_fund_price_points_cache_metrics_snapshot_locked")(
                        time.monotonic()
                    )
                _seam("_log_fund_price_points_cache_metrics")(pending)
                raise inflight.error
            with _fund_service()._FUND_PRICE_POINTS_CACHE_LOCK:
                _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS["coalesced_hits"] += 1
                pending = _seam("_fund_price_points_cache_metrics_snapshot_locked")(time.monotonic())
            _seam("_log_fund_price_points_cache_metrics")(pending)
            return copy.deepcopy(inflight.payload or [])

        try:
            points = loader()
            cached_payload = copy.deepcopy(points)
            approx_bytes = len(_fund_service()._stable_json_dumps(cached_payload).encode("utf-8"))
            with _fund_service()._FUND_PRICE_POINTS_CACHE_LOCK:
                if approx_bytes > _fund_service().FUND_PRICE_POINTS_CACHE_MAX_ENTRY_BYTES:
                    _fund_service()._FUND_PRICE_POINTS_CACHE_METRICS["oversize_skips"] += 1
                    inflight.payload = points
                else:
                    inflight.payload = cached_payload
                    if generation == _fund_service()._FUND_PRICE_POINTS_CACHE_GENERATION:
                        _seam("_store_fund_price_points_cache_locked")(
                            key,
                            cached_payload,
                            approx_bytes,
                            time.monotonic(),
                        )
            return points
        except BaseException as exc:
            inflight.error = exc
            raise
        finally:
            with _fund_service()._FUND_PRICE_POINTS_CACHE_LOCK:
                _fund_service()._FUND_PRICE_POINTS_IN_FLIGHT.pop(key, None)
                inflight.event.set()
                pending = _seam("_fund_price_points_cache_metrics_snapshot_locked")(time.monotonic())
            _seam("_log_fund_price_points_cache_metrics")(pending)


def _read_fund_price_points_uncached(
    processed_dir: Path,
    normalized: str,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
    normalized_sources: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    is_postgres = _fund_service().database_enabled()
    conditions = ["fund_code = ?", "price > 0"]
    params: List[Any] = [normalized]
    if normalized_sources:
        placeholders = ", ".join("?" for _ in normalized_sources)
        conditions.append(f"source IN ({placeholders})")
        params.extend(normalized_sources)
    if start_date:
        conditions.append("date >= ?")
        params.append(start_date.isoformat())
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date.isoformat())
    where_clause = " AND ".join(conditions)
    if is_postgres:
        source_priority_sql, priority_params = _price_source_priority_sql("source")
        query = (
            # Keep raw_json in storage for provenance, but do not transfer it to
            # this reader: chart/API points only use the normalized columns below.
            "SELECT DISTINCT ON (date) "
            "fund_code, date, source, price, daily_return, aum, investor_count, "
            f"share_count, metadata_json, fetched_at, updated_at FROM fund_prices WHERE {where_clause} "
            f"ORDER BY date ASC, {source_priority_sql} DESC, updated_at DESC"
        )
        query_params: List[Any] = [*params, *priority_params]
    else:
        query = (
            "SELECT fund_code, date, source, price, daily_return, aum, investor_count, "
            f"share_count, metadata_json, fetched_at, updated_at FROM fund_prices WHERE {where_clause} "
            "ORDER BY date ASC, updated_at DESC"
        )
        query_params = params
    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        rows = list(conn.execute(query, query_params))

    by_date: Dict[str, Any] = {}
    for row in rows:
        existing = by_date.get(str(row["date"]))
        if existing is None:
            by_date[str(row["date"])] = row
            continue
        row_rank = (
            _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(row["source"] or ""), 0),
            str(row["updated_at"] or ""),
        )
        existing_rank = (
            _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing["source"] or ""), 0),
            str(existing["updated_at"] or ""),
        )
        if row_rank > existing_rank:
            by_date[str(row["date"])] = row

    return [
        _seam("_fund_price_storage_row_to_point")(by_date[point_date])
        for point_date in sorted(by_date)
    ]


def read_fund_price_points(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    sources: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    normalized = _fund_service().normalize_fund_code(fund_code)
    if not normalized:
        return []
    normalized_sources = tuple(
        sorted({_seam("_normalize_price_source")(source) for source in list(sources or [])})
    )
    cache_key = _seam("_fund_price_points_cache_key")(
        processed_dir,
        normalized,
        start_date,
        end_date,
        normalized_sources,
    )
    return _seam("_read_fund_price_points_cached")(
        cache_key,
        lambda: _seam("_read_fund_price_points_uncached")(
            processed_dir,
            normalized,
            start_date=start_date,
            end_date=end_date,
            normalized_sources=normalized_sources,
        ),
    )


def read_latest_fund_price_points(
    processed_dir: Path,
    fund_codes: Iterable[str],
    *,
    end_date: Optional[date] = None,
    include_metadata: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Read the latest daily price point for many funds in one query.

    The fund catalogue is a point-in-time snapshot, while daily history can
    receive newer rows independently. This bulk reader keeps the catalogue
    overlay cheap enough to use for the list and detail endpoints.

    ``include_metadata=False`` skips the descriptive ``metadata_json`` blob for
    callers that only overlay quote columns. Both cache namespaces stay
    separate so a lean read never hides metadata from a later full read.
    """

    normalized_codes = sorted({_fund_service().normalize_fund_code(code) for code in fund_codes if _fund_service().normalize_fund_code(code)})
    if not normalized_codes:
        return {}

    is_postgres = _fund_service().database_enabled()
    price_gen = _fund_service().fund_listing_revision(processed_dir).get("price_generation", 0) if is_postgres else 0
    ttl = (
        max(21600, _fund_service().FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS)
        if (is_postgres and _fund_service().FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS >= 900)
        else max(1, _fund_service().FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS)
    )
    now = time.time()
    projection_mode = "full" if include_metadata else "quote"
    db_key_prefix = (
        f"latest-fund-price-points:{projection_mode}:{_fund_service()._fund_prices_db_path(processed_dir)}:"
        f"gen={price_gen}:{end_date.isoformat() if end_date else 'latest'}"
    )
    cache_key = f"{db_key_prefix}:{','.join(normalized_codes)}"
    cached = _fund_service()._MEMORY_CACHE.get(cache_key)
    cached_at = cached.get("cached_at") if isinstance(cached, dict) else None
    if isinstance(cached_at, (int, float)) and now - float(cached_at) <= ttl:
        return {
            str(code): dict(point)
            for code, point in dict(cached.get("points") or {}).items()
            if isinstance(point, dict)
        }

    superset_key = f"{db_key_prefix}:__superset__"
    superset = _fund_service()._LATEST_PRICE_POINTS_SUPERSET_CACHE.get(superset_key) or _fund_service()._MEMORY_CACHE.get(superset_key)
    superset_at = superset.get("cached_at") if isinstance(superset, dict) else None
    superset_codes = superset.get("covered_codes") if isinstance(superset, dict) else None
    if (
        isinstance(superset_at, (int, float))
        and now - float(superset_at) <= ttl
        and isinstance(superset_codes, set)
        and superset_codes.issuperset(normalized_codes)
    ):
        superset_points = dict(superset.get("points") or {})
        return {
            code: dict(superset_points[code])
            for code in normalized_codes
            if isinstance(superset_points.get(code), dict)
        }

    is_postgres = _fund_service().database_enabled()
    source_placeholders = ", ".join("?" for _ in _fund_service()._DAILY_PRICE_SOURCES)
    code_placeholders = ", ".join("?" for _ in normalized_codes)
    inner_conditions = [
        f"fund_code IN ({code_placeholders})",
        "price > 0",
        f"source IN ({source_placeholders})",
    ]
    inner_params: List[Any] = [*normalized_codes, *_fund_service()._DAILY_PRICE_SOURCES]
    if end_date:
        inner_conditions.append("date <= ?")
        inner_params.append(end_date.isoformat())
    metadata_column = ", fp.metadata_json" if include_metadata else ""
    if is_postgres:
        priority_items = list(_fund_service()._FUND_PRICE_SOURCE_PRIORITY.items())
        priority_cases = " ".join(f"WHEN ? THEN {int(priority)}" for _source, priority in priority_items)
        priority_params = [source for source, _priority in priority_items]
        source_priority_sql = f"CASE fp.source {priority_cases} ELSE 0 END" if priority_cases else "0"
        query = f"""
            -- raw_json and created_at are intentionally omitted; the public point
            -- mapper does not use them, and raw_json can dominate these result rows.
            SELECT DISTINCT ON (fp.fund_code)
                   fp.fund_code, fp.date, fp.source, fp.price, fp.daily_return,
                   fp.aum, fp.investor_count, fp.share_count,
                   fp.fetched_at, fp.updated_at{metadata_column}
            FROM fund_prices fp
            WHERE {' AND '.join(inner_conditions)}
            ORDER BY fp.fund_code,
                     fp.date DESC,
                     {source_priority_sql} DESC,
                     fp.updated_at DESC
        """
        params = [*inner_params, *priority_params]
    else:
        query = f"""
            -- raw_json and created_at are intentionally omitted; the public point
            -- mapper does not use them, and raw_json can dominate these result rows.
            SELECT fp.fund_code, fp.date, fp.source, fp.price, fp.daily_return,
                   fp.aum, fp.investor_count, fp.share_count,
                   fp.fetched_at, fp.updated_at{metadata_column}
            FROM fund_prices fp
            INNER JOIN (
                SELECT fund_code, MAX(date) AS latest_date
                FROM fund_prices
                WHERE {' AND '.join(inner_conditions)}
                GROUP BY fund_code
            ) latest
              ON latest.fund_code = fp.fund_code
             AND latest.latest_date = fp.date
            WHERE fp.fund_code IN ({code_placeholders})
              AND fp.price > 0
              AND fp.source IN ({source_placeholders})
            ORDER BY fp.fund_code, fp.updated_at DESC
        """
        params = [*inner_params, *normalized_codes, *_fund_service()._DAILY_PRICE_SOURCES]

    selected: Dict[str, Any] = {}
    try:
        with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
            for row in conn.execute(query, params):
                code = _fund_service().normalize_fund_code(str(row["fund_code"]))
                existing = selected.get(code)
                if existing is None:
                    selected[code] = row
                    continue
                row_rank = (
                    _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(row["source"] or ""), 0),
                    str(row["updated_at"] or ""),
                )
                existing_rank = (
                    _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing["source"] or ""), 0),
                    str(existing["updated_at"] or ""),
                )
                if row_rank > existing_rank:
                    selected[code] = row
    except Exception:
        # A history overlay is an enhancement. If the optional price store is
        # unavailable, the catalogue snapshot remains the safe fallback.
        return {}

    points = {
        code: _seam("_fund_price_storage_row_to_point")(row, include_metadata=include_metadata)
        for code, row in selected.items()
    }
    now_stored = time.time()
    _fund_service()._memory_cache_put(cache_key, {"cached_at": now_stored, "points": points})
    if (
        not isinstance(superset_at, (int, float))
        or now_stored - float(superset_at) > ttl
        or not isinstance(superset_codes, set)
        or len(normalized_codes) >= len(superset_codes)
    ):
        if len(_fund_service()._LATEST_PRICE_POINTS_SUPERSET_CACHE) >= 4:
            oldest_key = next(iter(_fund_service()._LATEST_PRICE_POINTS_SUPERSET_CACHE))
            _fund_service()._LATEST_PRICE_POINTS_SUPERSET_CACHE.pop(oldest_key, None)
        _fund_service()._LATEST_PRICE_POINTS_SUPERSET_CACHE[superset_key] = {
            "cached_at": now_stored,
            "covered_codes": set(normalized_codes),
            "points": points,
        }
    return {code: dict(point) for code, point in points.items()}


def read_fund_price_points_range(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """Read one fund's daily price points for a date window from the row store.

    Fund detail fallbacks need a single fund's rows for a handful of dates.
    Reading them here avoids pulling the multi-megabyte
    ``daily_snapshots/<date>.json`` documents, which contain every fund in the
    market just to extract one code.
    """

    normalized = _fund_service().normalize_fund_code(fund_code)
    daily_sources = tuple(_fund_service()._DAILY_PRICE_SOURCES)
    if not normalized or start_date > end_date or not daily_sources:
        return []

    def load_range() -> List[Dict[str, Any]]:
        is_postgres = _fund_service().database_enabled()
        source_placeholders = ", ".join("?" for _ in daily_sources)
        where_sql = (
            "WHERE fund_code = ? "
            "AND date BETWEEN ? AND ? "
            "AND price > 0 "
            f"AND source IN ({source_placeholders})"
        )
        base_params: List[Any] = [
            normalized,
            start_date.isoformat(),
            end_date.isoformat(),
            *daily_sources,
        ]
        if is_postgres:
            source_priority_sql, priority_params = _price_source_priority_sql("source")
            query = f"""
                -- metadata_json, raw_json and created_at are omitted; range readers only use price/overview columns.
                SELECT DISTINCT ON (date)
                       fund_code, date, source, price, daily_return, aum,
                       investor_count, share_count, fetched_at, updated_at
                FROM fund_prices
                {where_sql}
                ORDER BY date ASC, {source_priority_sql} DESC, updated_at DESC
            """
            params = [*base_params, *priority_params]
        else:
            query = f"""
                SELECT fund_code, date, source, price, daily_return, aum,
                       investor_count, share_count, fetched_at, updated_at
                FROM fund_prices
                {where_sql}
                ORDER BY date, updated_at DESC
            """
            params = base_params

        by_date: Dict[str, Any] = {}
        with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
            for row in conn.execute(query, params):
                row_date = str(row["date"])
                existing = by_date.get(row_date)
                if existing is None or (
                    _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(row["source"] or ""), 0),
                    str(row["updated_at"] or ""),
                ) > (
                    _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing["source"] or ""), 0),
                    str(existing["updated_at"] or ""),
                ):
                    by_date[row_date] = row

        return [
            _seam("_fund_price_storage_row_to_point")(by_date[day], include_metadata=False)
            for day in sorted(by_date)
        ]

    cache_key = (
        *_seam("_fund_price_points_cache_key")(
            processed_dir,
            normalized,
            start_date,
            end_date,
            tuple(sorted(daily_sources)),
        ),
        "range",
    )
    try:
        return _seam("_read_fund_price_points_cached")(cache_key, load_range)
    except Exception:
        # The row store is an optional optimization; callers fall back to the
        # daily snapshot documents when it is unavailable. Keep exceptions
        # outside _read_fund_price_points_cached so transient failures are never cached.
        return []


def _daily_return_reconciliation_candidates(
    rows: Iterable[Dict[str, Any]],
    *,
    include_missing: bool = False,
) -> Dict[str, str]:
    """Return fund/date pairs whose snapshot return may use a stale range.

    The TEFAS list snapshot can contain a ``getiriOrani`` value calculated for
    a wider range than the previous market business day.  Those rows are
    marked either by the list-snapshot source or by the returns merge that
    supplied the value.  A normal official ``gunlukGetiri`` is left intact.
    When ``include_missing`` is true, rows without a daily return are also
    candidates so public payloads can derive the previous-market-close change
    from the daily price history.
    """

    candidates: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = _seam("_normalize_price_source")(str(row.get("source") or ""))
        return_source = _seam("_normalize_price_source")(str(row.get("daily_return_source") or ""))
        has_daily_return = _fund_service()._coerce_float(_fund_service()._first_present(row, "daily_return", "gunlukGetiri")) is not None
        # A catalogue row that carries a zero price has no price history either,
        # so it can never produce a previous-close return. Keeping it in the
        # candidate set only adds funds to the rebuild's window query.
        row_price = _fund_service()._coerce_float(row.get("price"))
        if row_price is not None and row_price <= 0:
            continue
        if not (
            source == _fund_service().TEFAS_LIST_SNAPSHOT_SOURCE
            or return_source == _fund_service().TEFASFON_RETURNS_SOURCE
            or (include_missing and not has_daily_return)
        ):
            continue
        code = _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        point_date = _fund_service()._fund_date(row.get("as_of") or row.get("date") or row.get("tarih"))
        if code and point_date:
            candidates[code] = point_date
    return candidates


def _daily_return_overrides_from_price_history(
    processed_dir: Path,
    rows: Iterable[Dict[str, Any]],
    *,
    max_gap_days: Optional[int] = None,
    include_missing: bool = False,
    include_revision: bool = False,
) -> Dict[Tuple[str, str], float] | Tuple[Dict[Tuple[str, str], float], Optional[Tuple[str, int]]]:
    """Compute previous-market-business-day returns for eligible snapshots."""

    effective_max_gap_days = (
        _fund_service()._DAILY_RETURN_LOCAL_FALLBACK_MAX_GAP_DAYS
        if max_gap_days is None
        else max_gap_days
    )
    candidates = _seam("_daily_return_reconciliation_candidates")(rows, include_missing=include_missing)
    if not candidates or effective_max_gap_days <= 0:
        return ({}, None) if include_revision else {}
    candidate_dates = [date.fromisoformat(value) for value in candidates.values()]
    start_date = min(candidate_dates) - timedelta(days=effective_max_gap_days)
    end_date = max(candidate_dates)
    normalized_sources = tuple(_fund_service()._DAILY_PRICE_SOURCES)
    placeholders = ", ".join("?" for _ in candidates)
    source_placeholders = ", ".join("?" for _ in normalized_sources)
    is_postgres = _fund_service().database_enabled()
    if is_postgres:
        priority_items = list(_fund_service()._FUND_PRICE_SOURCE_PRIORITY.items())
        priority_cases = " ".join(f"WHEN ? THEN {int(priority)}" for _source, priority in priority_items)
        priority_params = [source for source, _priority in priority_items]
        source_priority_sql = f"CASE source {priority_cases} ELSE 0 END" if priority_cases else "0"
        query = (
            "SELECT DISTINCT ON (fund_code, date) fund_code, date, source, price, updated_at FROM fund_prices "
            f"WHERE fund_code IN ({placeholders}) "
            f"AND date >= ? AND date <= ? AND source IN ({source_placeholders}) AND price > 0 "
            f"ORDER BY fund_code, date ASC, {source_priority_sql} DESC, updated_at DESC"
        )
        params: List[Any] = [
            *candidates.keys(),
            start_date.isoformat(),
            end_date.isoformat(),
            *normalized_sources,
            *priority_params,
        ]
    else:
        query = (
            "SELECT fund_code, date, source, price, updated_at FROM fund_prices "
            f"WHERE fund_code IN ({placeholders}) "
            f"AND date >= ? AND date <= ? AND source IN ({source_placeholders}) AND price > 0 "
            "ORDER BY fund_code, date ASC, updated_at DESC"
        )
        params = [*candidates.keys(), start_date.isoformat(), end_date.isoformat(), *normalized_sources]

    selected: Dict[Tuple[str, str], sqlite3.Row] = {}
    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        for row in conn.execute(query, params):
            key = (str(row["fund_code"]), str(row["date"]))
            existing = selected.get(key)
            if existing is None:
                selected[key] = row
                continue
            row_rank = (
                _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(row["source"] or ""), 0),
                str(row["updated_at"] or ""),
            )
            existing_rank = (
                _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing["source"] or ""), 0),
                str(existing["updated_at"] or ""),
            )
            if row_rank > existing_rank:
                selected[key] = row

    if include_revision:
        revision_digest = hashlib.sha256()
        for (code, point_date), row in sorted(selected.items()):
            revision_fields = (
                code,
                point_date,
                str(row["source"] or ""),
                _fund_service()._coerce_float(row["price"]),
                str(row["updated_at"] or ""),
            )
            revision_digest.update(
                json.dumps(revision_fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            revision_digest.update(b"\n")
        source_revision = (revision_digest.hexdigest(), len(selected))
    else:
        source_revision = None
    by_code: Dict[str, List[Tuple[str, float]]] = {}
    for (code, point_date), row in selected.items():
        price = _fund_service()._coerce_float(row["price"])
        if price is not None and price > 0:
            by_code.setdefault(code, []).append((point_date, price))

    overrides: Dict[Tuple[str, str], float] = {}
    for code, target_date in candidates.items():
        points = sorted(by_code.get(code) or [], key=lambda item: item[0])
        current = next((price for point_date, price in points if point_date == target_date), None)
        previous = next(
            ((point_date, price) for point_date, price in reversed(points) if point_date < target_date),
            None,
        )
        if current is None or previous is None:
            continue
        previous_date, previous_price = previous
        gap = (date.fromisoformat(target_date) - date.fromisoformat(previous_date)).days
        if gap < 1 or gap > effective_max_gap_days:
            continue
        computed = _fund_service()._return_between(current, previous_price)
        if computed is not None:
            overrides[(code, target_date)] = round(computed, 4)
    if include_revision:
        return overrides, source_revision
    return overrides


def _carry_forward_resolved_daily_returns(
    existing_rows: Iterable[Dict[str, Any]],
    fresh_rows: Iterable[Dict[str, Any]],
) -> int:
    """Reuse a resolved previous-close return while the price is unchanged.

    TEFAS publishes fund prices once per trading day, but the catalogue is
    refreshed every 30 minutes. Recomputing the same previous-close difference
    from ``fund_prices`` for all ~2,000 funds on every refresh reads about
    1.3 MB. When the date and price are unchanged the stored return is still
    valid, so only genuinely changed rows remain reconciliation candidates.
    """

    previous_by_code = {
        _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or "")): row
        for row in existing_rows
        if isinstance(row, dict)
    }
    carried = 0
    for row in fresh_rows:
        if not isinstance(row, dict):
            continue
        code = _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        previous = previous_by_code.get(code)
        if not previous:
            continue
        if str(previous.get("daily_return_source") or "") != _fund_service().DAILY_RETURN_LOCAL_PRICE_SOURCE:
            continue
        prev_date = _fund_service()._fund_date(
            _fund_service()._first_present(previous, "as_of", "TARIH", "TARIHSTR", "date", "tarih")
        )
        curr_date = _fund_service()._fund_date(
            _fund_service()._first_present(row, "as_of", "TARIH", "TARIHSTR", "date", "tarih")
        )
        if prev_date != curr_date:
            continue
        previous_price = _fund_service()._coerce_float(
            _fund_service()._first_present(previous, "price", "FIYAT", "FONFIYAT", "fiyat", "sonFiyat")
        )
        current_price = _fund_service()._coerce_float(
            _fund_service()._first_present(row, "price", "FIYAT", "FONFIYAT", "fiyat", "sonFiyat")
        )
        if previous_price is None or current_price is None:
            continue
        if abs(previous_price - current_price) > max(1e-9, abs(previous_price) * 1e-9):
            continue
        resolved = _fund_service()._coerce_float(previous.get("daily_return"))
        if resolved is None:
            continue
        row["daily_return"] = resolved
        if "gunlukGetiri" in row and row.get("gunlukGetiri") is None:
            row["gunlukGetiri"] = resolved
        row["daily_return_source"] = _fund_service().DAILY_RETURN_LOCAL_PRICE_SOURCE
        if previous.get("daily_return_basis_date"):
            row["daily_return_basis_date"] = previous["daily_return_basis_date"]
        carried += 1
    return carried


def _missing_daily_return_keys(rows: Iterable[Dict[str, Any]]) -> set[Tuple[str, str]]:
    """Return (fund, date) pairs whose catalogue row has no daily return."""

    missing: set[Tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _fund_service()._coerce_float(_fund_service()._first_present(row, "daily_return", "gunlukGetiri")) is not None:
            continue
        code = _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        point_date = _fund_service()._fund_date(row.get("as_of") or row.get("date") or row.get("tarih"))
        if code and point_date:
            missing.add((code, point_date))
    return missing


def _daily_return_override_values(
    rows: Iterable[Dict[str, Any]],
    overrides: Dict[Tuple[str, str], float],
    *,
    mark_price_history_source: bool = False,
) -> List[Dict[str, Any]]:
    """Apply precomputed previous-close returns to catalogue rows.

    When ``mark_price_history_source`` is true the row records that its return
    no longer comes from the range-return merge. Readers can then skip the
    price-history reconciliation query for that row on every catalogue rebuild.
    """

    materialized = [dict(row) for row in rows if isinstance(row, dict)]
    if not overrides:
        return materialized
    for row in materialized:
        code = _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        point_date = _fund_service()._fund_date(row.get("as_of") or row.get("date") or row.get("tarih"))
        override = overrides.get((code, point_date or ""))
        if override is None:
            continue
        row["daily_return"] = override
        if mark_price_history_source:
            row["daily_return_source"] = _fund_service().DAILY_RETURN_LOCAL_PRICE_SOURCE
    return materialized


def _apply_daily_return_overrides(
    processed_dir: Path,
    rows: Iterable[Dict[str, Any]],
    *,
    include_missing: bool = False,
    include_revision: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Optional[Tuple[str, int]]]:
    materialized = [dict(row) for row in rows if isinstance(row, dict)]
    cache_key: Optional[str] = None
    overrides: Optional[Dict[Tuple[str, str], float]] = None
    source_revision: Optional[Tuple[str, int]] = None
    if _fund_service().database_enabled() and not include_revision and len(materialized) >= 10:
        candidates = _seam("_daily_return_reconciliation_candidates")(
            materialized,
            include_missing=include_missing,
        )
        if not candidates:
            return materialized
        price_gen = _fund_service().fund_listing_revision(processed_dir).get("price_generation", 0)
        digest_input = "|".join(f"{code}:{point_date}" for code, point_date in sorted(candidates.items()))
        candidates_hash = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:16]
        cache_key = (
            f"daily-return-overrides:{_fund_service()._fund_prices_db_path(processed_dir)}:"
            f"{price_gen}:{int(include_missing)}:{candidates_hash}"
        )
        cached = _fund_service()._MEMORY_CACHE.get(cache_key)
        overlay_ttl = _fund_service().FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS
        memo_ttl = float(
            max(21600, overlay_ttl)
            if overlay_ttl >= 900
            else max(1, overlay_ttl)
        )
        if (
            isinstance(cached, dict)
            and isinstance(cached.get("cached_at"), (int, float))
            and time.time() - float(cached["cached_at"]) <= memo_ttl
            and isinstance(cached.get("overrides"), dict)
        ):
            overrides = cached["overrides"]

    if overrides is None:
        override_result = _seam("_daily_return_overrides_from_price_history")(
            processed_dir,
            materialized,
            include_missing=include_missing,
            include_revision=include_revision,
        )
        if include_revision:
            overrides, source_revision = override_result
        else:
            overrides = override_result
            source_revision = None
        if cache_key is not None and isinstance(overrides, dict):
            _fund_service()._memory_cache_put(
                cache_key,
                {
                    "cached_at": time.time(),
                    "overrides": overrides,
                },
            )
    materialized = _seam("_daily_return_override_values")(materialized, overrides)
    return (materialized, source_revision) if include_revision else materialized


def _persist_daily_return_overrides(
    processed_dir: Path,
    overrides: Dict[Tuple[str, str], float],
) -> None:
    if not overrides:
        return
    source_placeholders = ", ".join("?" for _ in _fund_service()._DAILY_PRICE_SOURCES)
    query = (
        "UPDATE fund_prices SET daily_return = ?, updated_at = ? "
        "WHERE fund_code = ? AND date = ? "
        f"AND source IN ({source_placeholders}) "
        "AND (daily_return IS NULL OR ABS(daily_return - ?) > 1e-9)"
    )
    now = _fund_service()._utc_now_iso()
    params_seq = [
        (value, now, code, point_date, *_fund_service()._DAILY_PRICE_SOURCES, value)
        for (code, point_date), value in overrides.items()
    ]
    changed_count = 0
    with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
        if hasattr(conn, "executemany"):
            cur = conn.executemany(query, params_seq)
            rc = getattr(cur, "rowcount", None)
            changed_count = rc if isinstance(rc, int) and rc >= 0 else len(params_seq)
        else:
            for row_params in params_seq:
                cur = conn.execute(query, row_params)
                rc = getattr(cur, "rowcount", None)
                changed_count += rc if isinstance(rc, int) and rc >= 0 else 1
        if changed_count:
            _fund_service()._bump_fund_listing_price_revision(conn)
        conn.commit()
    if changed_count:
        _fund_service()._invalidate_fund_price_memory_caches()
        _seam("invalidate_fund_price_points_cache")()


def _read_fintables_udf_price_points(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    return _seam("read_fund_price_points")(
        processed_dir,
        fund_code,
        start_date=start_date,
        end_date=end_date,
        sources=_fund_service()._FINTABLES_DAILY_PRICE_SOURCES,
    )


def _read_daily_fund_price_points(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    return _seam("read_fund_price_points")(
        processed_dir,
        fund_code,
        start_date=start_date,
        end_date=end_date,
        sources=_fund_service()._DAILY_PRICE_SOURCES,
    )


def _read_preferred_fund_price_points(
    processed_dir: Path,
    fund_code: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Read performance prices with Fintables preferred where it is complete.

    Fintables remains authoritative for overlapping dates, but a partial
    Fintables response must not hide older TEFAS history. This is especially
    important for full-history reads, where the upstream UDF endpoint may
    return only a recent slice even though the fund has an older inception
    date.
    """

    normalized = _fund_service().normalize_fund_code(fund_code)
    if not normalized:
        return []
    daily_sources = tuple(_fund_service()._DAILY_PRICE_SOURCES)
    fintables_sources = set(_fund_service()._FINTABLES_DAILY_PRICE_SOURCES)

    def load_preferred() -> List[Dict[str, Any]]:
        is_postgres = _fund_service().database_enabled()
        conditions = ["fund_code = ?", "price > 0"]
        params: List[Any] = [normalized]
        if daily_sources:
            placeholders = ", ".join("?" for _ in daily_sources)
            conditions.append(f"source IN ({placeholders})")
            params.extend(daily_sources)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date.isoformat())
        where_clause = " AND ".join(conditions)
        if is_postgres:
            source_priority_sql, priority_params = _price_source_priority_sql("source")
            if fintables_sources:
                fintables_list = sorted(fintables_sources)
                fintables_placeholders = ", ".join("?" for _ in fintables_list)
                query = (
                    "SELECT DISTINCT ON (date, is_fintables) "
                    "fund_code, date, source, price, daily_return, aum, investor_count, "
                    "share_count, fetched_at, updated_at FROM ("
                    "SELECT fund_code, date, source, price, daily_return, aum, investor_count, "
                    "share_count, fetched_at, updated_at, "
                    f"CASE WHEN source IN ({fintables_placeholders}) THEN 1 ELSE 0 END AS is_fintables "
                    f"FROM fund_prices WHERE {where_clause}"
                    ") AS fp "
                    f"ORDER BY date ASC, is_fintables DESC, {source_priority_sql} DESC, updated_at DESC"
                )
                query_params: List[Any] = [
                    *fintables_list,
                    *params,
                    *priority_params,
                ]
            else:
                query = (
                    "SELECT DISTINCT ON (date) "
                    "fund_code, date, source, price, daily_return, aum, investor_count, "
                    f"share_count, fetched_at, updated_at FROM fund_prices WHERE {where_clause} "
                    f"ORDER BY date ASC, {source_priority_sql} DESC, updated_at DESC"
                )
                query_params = [*params, *priority_params]
        else:
            query = (
                "SELECT fund_code, date, source, price, daily_return, aum, investor_count, "
                f"share_count, fetched_at, updated_at FROM fund_prices WHERE {where_clause} "
                "ORDER BY date ASC, updated_at DESC"
            )
            query_params = list(params)
        with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
            rows = list(conn.execute(query, query_params))

        fintables_rows_by_date: Dict[str, Any] = {}
        daily_rows_by_date: Dict[str, Any] = {}
        for row in rows:
            point_date = str(row["date"])
            row_source = _seam("_normalize_price_source")(str(row["source"] or ""))
            row_rank = (
                _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(row["source"] or ""), 0),
                str(row["updated_at"] or ""),
            )
            if row_source in fintables_sources:
                existing_f = fintables_rows_by_date.get(point_date)
                if existing_f is None:
                    fintables_rows_by_date[point_date] = row
                else:
                    existing_f_rank = (
                        _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing_f["source"] or ""), 0),
                        str(existing_f["updated_at"] or ""),
                    )
                    if row_rank > existing_f_rank:
                        fintables_rows_by_date[point_date] = row
            existing_d = daily_rows_by_date.get(point_date)
            if existing_d is None:
                daily_rows_by_date[point_date] = row
            else:
                existing_d_rank = (
                    _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing_d["source"] or ""), 0),
                    str(existing_d["updated_at"] or ""),
                )
                if row_rank > existing_d_rank:
                    daily_rows_by_date[point_date] = row

        fintables_points = [
            _seam("_fund_price_storage_row_to_point")(fintables_rows_by_date[d], include_metadata=False)
            for d in sorted(fintables_rows_by_date)
        ]
        daily_points = [
            _seam("_fund_price_storage_row_to_point")(daily_rows_by_date[d], include_metadata=False)
            for d in sorted(daily_rows_by_date)
        ]
        if not fintables_points:
            return daily_points

        daily_by_date = {
            str(point.get("date")): point
            for point in daily_points
            if isinstance(point, dict) and point.get("date")
        }
        metadata_fields = (
            "aum",
            "investor_count",
            "share_count",
            "name",
            "fund_type",
            "founder_company",
            "manager_company",
            "tefas_open",
            "risk_value",
            "currency",
        )

        def enrich(point: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(point)
            supplemental = daily_by_date.get(str(point.get("date"))) or {}
            for field in metadata_fields:
                if merged.get(field) is None and supplemental.get(field) is not None:
                    merged[field] = supplemental[field]
            return merged

        fintables_is_complete = bool(
            start_date
            and end_date
            and _fund_service()._history_is_usable_daily_range(
                fintables_points,
                start_date=start_date,
                end_date=end_date,
            )
        )
        if fintables_is_complete:
            preferred_by_date = {
                str(point.get("date")): enrich(point)
                for point in fintables_points
                if point.get("date")
            }
            latest_fintables_date = max(preferred_by_date, default="")
            # The daily collector can publish a confirmed TEFAS close before the
            # Fintables UDF catches up. Keep Fintables authoritative on overlaps,
            # but do not hide a newer, non-overlapping TEFAS tail.
            for point in daily_points:
                point_date = str(point.get("date") or "")
                if latest_fintables_date < point_date <= end_date.isoformat():
                    preferred_by_date[point_date] = point
            return [preferred_by_date[point_date] for point_date in sorted(preferred_by_date)]

        # A partial Fintables response is preferred only on dates it actually
        # covers. Keep all other locally cached dates, including older TEFAS
        # inception/history points, and then let Fintables win on overlaps.
        preferred_by_date = dict(daily_by_date)
        for point in fintables_points:
            preferred_by_date[str(point.get("date"))] = enrich(point)
        return [preferred_by_date[point_date] for point_date in sorted(preferred_by_date)]

    cache_key = (
        *_seam("_fund_price_points_cache_key")(
            processed_dir,
            normalized,
            start_date,
            end_date,
            tuple(sorted(daily_sources)),
        ),
        "preferred",
    )
    return _seam("_read_fund_price_points_cached")(cache_key, load_preferred)


def _backfill_daily_returns_from_local_prices(
    processed_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    max_gap_days: int = _UNSET,
) -> int:
    """Fallback: when TEFAS does not publish a daily return for a fund (e.g.
    qualified-investor / TEFAS-closed funds), compute it from the last two
    locally cached price points if they are close enough in time.

    Mutates ``rows`` in place.  Only fills rows whose ``daily_return`` and
    ``gunlukGetiri`` are both ``None``; existing values are preserved.
    Returns the number of rows that were back-filled.
    """

    if max_gap_days is _UNSET:
        max_gap_days = _fund_service()._DAILY_RETURN_LOCAL_FALLBACK_MAX_GAP_DAYS
    if not rows or max_gap_days <= 0:
        return 0
    read_points_fn = _seam("read_fund_price_points")
    pending: List[Tuple[Dict[str, Any], str, date]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        existing_daily = row.get("daily_return")
        if existing_daily is None:
            existing_daily = row.get("gunlukGetiri")
        if _fund_service()._coerce_float(existing_daily) is not None:
            continue
        code = _fund_service().normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        if not code:
            continue
        as_of_text = _fund_service()._first_text(row, "as_of", "tarih", "TARIH", "date")
        as_of_iso = _fund_service()._fund_date(as_of_text) if as_of_text else None
        if not as_of_iso:
            continue
        try:
            as_of = date.fromisoformat(as_of_iso)
        except ValueError:
            continue
        pending.append((row, code, as_of))

    if not pending:
        return 0

    points_by_code: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if len(pending) > 1 and read_points_fn is read_fund_price_points:
        codes = sorted({code for _r, code, _d in pending})
        min_date = min(as_of for _r, _c, as_of in pending) - timedelta(days=max(1, max_gap_days))
        max_date = max(as_of for _r, _c, as_of in pending)
        normalized_sources = tuple(_fund_service()._DAILY_PRICE_SOURCES)
        placeholders = ", ".join("?" for _ in codes)
        source_placeholders = ", ".join("?" for _ in normalized_sources)
        if _fund_service().database_enabled():
            priority_items = list(_fund_service()._FUND_PRICE_SOURCE_PRIORITY.items())
            priority_cases = " ".join(f"WHEN ? THEN {int(priority)}" for _source, priority in priority_items)
            priority_params = [source for source, _priority in priority_items]
            source_priority_sql = f"CASE source {priority_cases} ELSE 0 END" if priority_cases else "0"
            query = (
                "SELECT DISTINCT ON (fund_code, date) fund_code, date, price FROM fund_prices "
                f"WHERE fund_code IN ({placeholders}) "
                f"AND date >= ? AND date <= ? AND source IN ({source_placeholders}) AND price > 0 "
                f"ORDER BY fund_code, date ASC, {source_priority_sql} DESC, updated_at DESC"
            )
            params: List[Any] = [
                *codes,
                min_date.isoformat(),
                max_date.isoformat(),
                *normalized_sources,
                *priority_params,
            ]
            selected_points: Dict[str, List[Dict[str, Any]]] = {}
            with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
                for db_row in conn.execute(query, params):
                    selected_points.setdefault(str(db_row["fund_code"]), []).append(
                        {"date": str(db_row["date"]), "price": db_row["price"]}
                    )
            points_by_code = selected_points
        else:
            query = (
                "SELECT fund_code, date, source, price, updated_at FROM fund_prices "
                f"WHERE fund_code IN ({placeholders}) "
                f"AND date >= ? AND date <= ? AND source IN ({source_placeholders}) AND price > 0 "
                "ORDER BY fund_code, date ASC, updated_at DESC"
            )
            params = [*codes, min_date.isoformat(), max_date.isoformat(), *normalized_sources]
            selected_by_key: Dict[Tuple[str, str], Any] = {}
            with _fund_service()._connect_fund_prices_db(processed_dir) as conn:
                for db_row in conn.execute(query, params):
                    key = (str(db_row["fund_code"]), str(db_row["date"]))
                    existing = selected_by_key.get(key)
                    if existing is None:
                        selected_by_key[key] = db_row
                        continue
                    row_rank = (
                        _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(db_row["source"] or ""), 0),
                        str(db_row["updated_at"] or ""),
                    )
                    existing_rank = (
                        _fund_service()._FUND_PRICE_SOURCE_PRIORITY.get(str(existing["source"] or ""), 0),
                        str(existing["updated_at"] or ""),
                    )
                    if row_rank > existing_rank:
                        selected_by_key[key] = db_row
            selected_points = {}
            for (c_code, c_date), db_row in sorted(selected_by_key.items()):
                selected_points.setdefault(c_code, []).append(
                    {"date": c_date, "price": db_row["price"]}
                )
            points_by_code = selected_points

    filled = 0
    for row, code, as_of in pending:
        if points_by_code is not None:
            lower_iso = (as_of - timedelta(days=max(1, max_gap_days))).isoformat()
            upper_iso = as_of.isoformat()
            points = [
                pt
                for pt in (points_by_code.get(code) or [])
                if lower_iso <= str(pt.get("date") or "") <= upper_iso
            ]
        else:
            points = read_points_fn(
                processed_dir,
                code,
                start_date=as_of - timedelta(days=max(1, max_gap_days)),
                end_date=as_of,
                sources=_fund_service()._DAILY_PRICE_SOURCES,
            )
        if len(points) < 2:
            continue
        latest = points[-1]
        previous = points[-2]
        latest_iso = _fund_service()._fund_date(latest.get("date"))
        previous_iso = _fund_service()._fund_date(previous.get("date"))
        if not latest_iso or not previous_iso:
            continue
        try:
            latest_date = date.fromisoformat(latest_iso)
            previous_date = date.fromisoformat(previous_iso)
        except ValueError:
            continue
        if latest_date != as_of:
            # Local cache is missing today's point; skip rather than guess.
            continue
        gap = (latest_date - previous_date).days
        if gap < 1 or gap > max_gap_days:
            continue
        latest_price = _fund_service()._coerce_float(latest.get("price"))
        previous_price = _fund_service()._coerce_float(previous.get("price"))
        computed = _fund_service()._return_between(latest_price, previous_price)
        if computed is None:
            continue
        rounded = round(computed, 4)
        row["daily_return"] = rounded
        row["gunlukGetiri"] = rounded
        row["daily_return_source"] = _fund_service().DAILY_RETURN_LOCAL_PRICE_SOURCE
        row["daily_return_basis_date"] = previous_date.isoformat()
        filled += 1
    return filled


__all__ = [
    "_init_fund_prices_schema",
    "ensure_fund_prices_schema",
    "_normalize_history_row",
    "_upsert_fund_reference_data",
    "_fund_price_storage_row_to_point",
    "_normalize_price_source",
    "_public_price_source",
    "_price_warning_from_row",
    "_storage_point_from_row",
    "_insert_fund_price_warning",
    "upsert_fund_price_points",
    "_fund_price_points_cache_key",
    "_remove_fund_price_points_cache_entry_locked",
    "_purge_expired_fund_price_points_cache_locked",
    "_store_fund_price_points_cache_locked",
    "_fund_price_points_cache_metrics_snapshot_locked",
    "_log_fund_price_points_cache_metrics",
    "invalidate_fund_price_points_cache",
    "_reset_fund_price_points_cache_for_tests",
    "_read_fund_price_points_cached",
    "_read_fund_price_points_uncached",
    "read_fund_price_points",
    "read_latest_fund_price_points",
    "read_fund_price_points_range",
    "_daily_return_reconciliation_candidates",
    "_daily_return_overrides_from_price_history",
    "_carry_forward_resolved_daily_returns",
    "_missing_daily_return_keys",
    "_daily_return_override_values",
    "_apply_daily_return_overrides",
    "_persist_daily_return_overrides",
    "_read_fintables_udf_price_points",
    "_read_daily_fund_price_points",
    "_read_preferred_fund_price_points",
    "_backfill_daily_returns_from_local_prices",
]

_ORIGINAL_EXPORTS.update({name: globals()[name] for name in __all__})
