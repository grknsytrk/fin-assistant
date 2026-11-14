from __future__ import annotations

import json

from app.reference_data import get_instrument, sync_reference_data_from_caches, upsert_instrument


def test_reference_data_keeps_higher_priority_static_fields(tmp_path) -> None:
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="TRHOL",
        name="Manual TRHOL",
        logo_url="https://example.test/trhol.svg",
        logo_source="manual",
        source="manual",
    )
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="TRHOL",
        name="KAP TRHOL",
        logo_url="https://example.test/kap.svg",
        logo_source="kap",
        source="kap",
    )

    row = get_instrument(tmp_path, "stock", "TRHOL")

    assert row is not None
    assert row["name"] == "Manual TRHOL"
    assert row["logo_url"] == "https://example.test/trhol.svg"


def test_reference_data_resolves_alias(tmp_path) -> None:
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="BIMAS",
        name="BİM BİRLEŞİK MAĞAZALAR A.Ş.",
        source="kap",
        aliases=["BIM"],
    )

    row = get_instrument(tmp_path, "stock", "BIM")

    assert row is not None
    assert row["symbol"] == "BIMAS"


def test_reference_data_bootstraps_existing_json_caches(tmp_path) -> None:
    funds_dir = tmp_path / "funds_cache"
    funds_dir.mkdir()
    (funds_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "fund_code": "TLY",
                        "name": "TERA PORTFÖY BİRİNCİ SERBEST FON",
                        "founder_company": "TERA PORTFÖY",
                        "as_of": "2026-05-20",
                        "source": "tefasfon_funds",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    kap_dir = tmp_path / "kap_cache"
    kap_dir.mkdir()
    (kap_dir / "DSTKF.json").write_text(
        json.dumps(
            {
                "company": "DSTKF",
                "stock_code": "DSTKF",
                "company_title": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "member_oid": "member-dstkf",
                "fetched_at": "2026-05-20T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = sync_reference_data_from_caches(tmp_path)

    assert result["record_count"] == 2
    assert get_instrument(tmp_path, "fund", "TLY")["name"] == "TERA PORTFÖY BİRİNCİ SERBEST FON"
    assert get_instrument(tmp_path, "stock", "DSTKF")["name"] == "DESTEK FİNANS FAKTORİNG A.Ş."


def test_reference_data_deduplicates_batch_keys(tmp_path) -> None:
    funds_dir = tmp_path / "funds_cache"
    funds_dir.mkdir()
    (funds_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"fund_code": "TLY", "name": "TLY", "source": "tefasfon_funds"},
                    {"fund_code": "TLY", "name": "TLY updated", "source": "tefasfon_funds"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = sync_reference_data_from_caches(tmp_path)

    # The normalized catalogue has one current row per fund code.
    assert result["record_count"] == 1
    assert result["upserted_count"] == 1
    assert get_instrument(tmp_path, "fund", "TLY")["name"] == "TLY updated"


def test_instruments_lookup_uses_memory_cache_and_invalidates_on_upsert(tmp_path, monkeypatch) -> None:
    import app.reference_data as ref_data

    ref_data.reset_reference_data_state_for_tests()
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="THYAO",
        name="TÜRK HAVA YOLLARI",
        aliases=["THYAO_ALT"],
        source="kap",
        active=1,
    )

    original_connect = ref_data._connect
    connect_calls = 0

    def tracked_connect(processed_dir):
        nonlocal connect_calls
        connect_calls += 1
        return original_connect(processed_dir)

    monkeypatch.setattr(ref_data, "_connect", tracked_connect)

    by_symbol = get_instrument(tmp_path, "stock", "THYAO")
    by_alias = get_instrument(tmp_path, "stock", "THYAO_ALT")
    batch = ref_data.get_instruments(tmp_path, "stock", ["THYAO", "THYAO_ALT"])
    names = ref_data.get_instrument_names(tmp_path, "stock")

    assert by_symbol is not None and by_symbol["name"] == "TÜRK HAVA YOLLARI"
    assert by_alias is not None and by_alias["name"] == "TÜRK HAVA YOLLARI"
    assert batch["THYAO"]["name"] == "TÜRK HAVA YOLLARI"
    assert batch["THYAO_ALT"]["name"] == "TÜRK HAVA YOLLARI"
    assert names["THYAO"] == "TÜRK HAVA YOLLARI"
    assert connect_calls == 1

    # Updating an instrument must invalidate the in-memory cache cleanly.
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="THYAO",
        name="THYAO UPDATED",
        source="manual",
        active=1,
    )
    refreshed = get_instrument(tmp_path, "stock", "THYAO")
    assert refreshed is not None and refreshed["name"] == "THYAO UPDATED"
    assert connect_calls == 3
    ref_data.reset_reference_data_state_for_tests()


def test_postgres_instruments_stamp_skips_full_scan_and_noop_upsert(tmp_path, monkeypatch) -> None:
    from app import reference_data as ref_data

    ref_data.reset_reference_data_state_for_tests()
    queries: list[str] = []
    instruments_rows = [
        {
            "kind": "stock",
            "symbol": "THYAO",
            "name": "Türk Hava Yolları",
            "short_name": "THYAO",
            "source": "kap",
            "source_id": "100",
            "logo_url": None,
            "logo_source": None,
            "active": 1,
            "as_of": "2026-11-14",
            "metadata_json": "{}",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-11-14T10:00:00+00:00",
        }
    ]
    alias_rows: list[dict] = []

    class _FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return list(self._rows)

        def fetchone(self):
            return self._rows[0] if self._rows else None

    class _FakePgConn:
        is_postgres = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql: str, params=()):
            del params
            normalized = " ".join(sql.split())
            queries.append(normalized)
            if "AS instruments_stamp" in normalized:
                return _FakeResult([{"instruments_stamp": "2026-11-14T10:00:00+00:00:1", "aliases_stamp": ":0"}])
            if normalized.startswith("SELECT * FROM instruments WHERE kind ="):
                return _FakeResult(instruments_rows)
            if normalized.startswith("SELECT * FROM instruments"):
                return _FakeResult(instruments_rows)
            if normalized.startswith("SELECT alias, kind, symbol FROM instrument_aliases"):
                return _FakeResult(alias_rows)
            if normalized.startswith("SELECT alias, kind, symbol, source FROM instrument_aliases"):
                return _FakeResult([])
            return _FakeResult([])

        def commit(self):
            return None

    monkeypatch.setattr(ref_data, "database_enabled", lambda: True)
    monkeypatch.setattr(ref_data, "ensure_reference_data_schema", lambda _p: None)
    monkeypatch.setattr(ref_data, "_connect", lambda _p: _FakePgConn())
    monkeypatch.setattr(ref_data, "_INSTRUMENTS_CACHE_TTL_SECONDS", 60.0)
    monkeypatch.setattr(ref_data, "_MANUAL_INSTRUMENTS", [])

    first = get_instrument(tmp_path, "stock", "THYAO")
    assert first is not None and first["name"] == "Türk Hava Yolları"
    assert sum(1 for q in queries if q == "SELECT * FROM instruments") == 1

    # Simulate TTL expiry: Postgres stamp matches, so full-table SELECT * FROM instruments must NOT rerun.
    cache_key = ref_data._instruments_cache_key(tmp_path)
    old_exp, sym_map, alias_map = ref_data._INSTRUMENTS_CACHE[cache_key]
    ref_data._INSTRUMENTS_CACHE[cache_key] = (old_exp - 1000.0, sym_map, alias_map)

    second = get_instrument(tmp_path, "stock", "THYAO")
    assert second is not None and second["name"] == "Türk Hava Yolları"
    assert sum(1 for q in queries if q == "SELECT * FROM instruments") == 1
    assert sum(1 for q in queries if "AS instruments_stamp" in q) == 2

    # Unchanged upsert must skip INSERT ... ON CONFLICT DO UPDATE and keep cache intact.
    noop = ref_data.upsert_instruments(
        tmp_path,
        [
            {
                "kind": "stock",
                "symbol": "THYAO",
                "name": "Türk Hava Yolları",
                "short_name": "THYAO",
                "source": "kap",
                "source_id": "100",
                "active": 1,
                "as_of": "2026-11-14",
                "metadata": {},
            }
        ],
        seed_manual=False,
    )
    assert noop["upserted_count"] == 0
    assert not any(q.startswith("INSERT INTO instruments") for q in queries)
    assert cache_key in ref_data._INSTRUMENTS_CACHE
    ref_data.reset_reference_data_state_for_tests()
