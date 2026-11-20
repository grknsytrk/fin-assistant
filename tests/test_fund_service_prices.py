from __future__ import annotations

import json
import threading
import time
from datetime import date

import pytest

from app import fund_service


def test_fund_prices_db_uses_wal_and_updated_index(tmp_path) -> None:
    with fund_service._connect_fund_prices_db(tmp_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
        indexes = {
            row["name"]
            for row in conn.execute("PRAGMA index_list('fund_prices')").fetchall()
        }

    assert str(journal_mode).lower() == "wal"
    assert int(synchronous) == 1
    assert "idx_fund_prices_code_date_updated" in indexes


def test_postgres_fund_price_upsert_deduplicates_same_batch_key(monkeypatch, tmp_path) -> None:
    executions = []

    class FakePostgresConnection:
        is_postgres = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, query, params=()):
            executions.append((query, list(params)))

        def commit(self):
            return None

    monkeypatch.setattr(fund_service, "database_enabled", lambda: True)
    monkeypatch.setattr(fund_service, "connect_postgres", lambda: FakePostgresConnection())

    result = fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "THF", "date": "2026-04-30", "price": 10.0, "source": "tefasfon_funds"},
            {"fund_code": "THF", "date": "2026-04-30", "price": 11.0, "source": "tefasfon_funds"},
        ],
        source="tefasfon_funds",
    )

    fund_inserts = [item for item in executions if "INSERT INTO fund_prices" in item[0]]
    assert result["upserted_count"] == 1
    assert len(fund_inserts) == 1
    assert len(fund_inserts[0][1]) == 13
    assert fund_inserts[0][1][3] == 11.0


def test_fund_snapshot_sort_uses_overlaid_prices_and_daily_returns(monkeypatch, tmp_path) -> None:
    from datetime import date

    monkeypatch.setattr(
        fund_service,
        "_latest_fund_snapshot_target_date",
        lambda today=None: date(2026, 10, 16),
    )
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "AAA", "date": "2026-10-15", "price": 100.0, "source": "tefasfon_funds"},
            {
                "fund_code": "AAA", "date": "2026-10-16", "price": 110.0,
                "daily_return": 10.0, "source": "tefasfon_funds",
            },
            {"fund_code": "BBB", "date": "2026-10-15", "price": 100.0, "source": "tefasfon_funds"},
            {
                "fund_code": "BBB", "date": "2026-10-16", "price": 105.0,
                "daily_return": 5.0, "source": "tefasfon_funds",
            },
        ],
        source="tefasfon_funds",
    )
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "rows": [
                    {
                        "fund_code": "AAA", "name": "Fund A", "price": 1.0,
                        "daily_return": 1.0, "as_of": "2026-10-15",
                        "source": "tefas_list_snapshot", "tefas_open": True,
                    },
                    {
                        "fund_code": "BBB", "name": "Fund B", "price": 200.0,
                        "daily_return": 20.0, "as_of": "2026-10-15",
                        "source": "tefas_list_snapshot", "tefas_open": True,
                    },
                ],
                "as_of": "2026-10-15",
                "source_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    by_price = fund_service.get_funds_payload(tmp_path, min_aum=None, sort="price", order="desc")
    by_return = fund_service.get_funds_payload(tmp_path, min_aum=None, sort="daily_return", order="desc")

    assert [row["fund_code"] for row in by_price["rows"]] == ["AAA", "BBB"]
    assert [row["fund_code"] for row in by_return["rows"]] == ["AAA", "BBB"]
    assert by_return["rows"][0]["daily_return"] == pytest.approx(10.0)


def test_fund_public_daily_return_uses_previous_business_day_price(tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "TLY", "date": "2026-08-14", "price": 8591.591231, "source": "tefasfon_funds"},
            {"fund_code": "TLY", "date": "2026-08-17", "price": 8611.792448, "source": "tefasfon_funds"},
        ],
        source="tefasfon_funds",
    )
    snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY BIRINCI SERBEST FON",
                "price": 8611.792448,
                "daily_return": 1.8110046568,
                "tefas_open": True,
                "as_of": "2026-08-17",
                "source": "tefas_list_snapshot",
            }
        ],
        "source": "tefasfon_funds",
        "as_of": "2026-08-17",
        "fetched_at": "2026-08-17T14:44:39+00:00",
        "stale": False,
        "degraded": False,
        "warnings": [],
        "source_metadata": {"source": "tefasfon_funds"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    list_payload = fund_service.get_funds_payload(tmp_path, min_aum=None)
    detail_payload = fund_service.get_fund_detail_payload(tmp_path, "TLY")

    assert list_payload["rows"][0]["daily_return"] == pytest.approx(0.2351, abs=0.0001)
    assert detail_payload["daily_return"] == pytest.approx(0.2351, abs=0.0001)


def test_fund_public_daily_return_fills_missing_direct_snapshot_return(tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "AAL", "date": "2026-09-09", "price": 3.5556, "source": "tefasfon_funds"},
            {"fund_code": "AAL", "date": "2026-09-10", "price": 3.559048, "source": "tefas_direct_funds"},
        ],
        source="tefas_direct_funds",
    )
    snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "AAL",
                "name": "ATA PORTFOY PARA PIYASASI FONU",
                "price": 3.559048,
                "daily_return": None,
                "tefas_open": True,
                "as_of": "2026-09-10",
                "source": "tefas_direct_funds",
            }
        ],
        "source": "tefas_direct_funds",
        "as_of": "2026-09-10",
        "fetched_at": "2026-09-10T13:31:07+03:00",
        "stale": False,
        "degraded": False,
        "warnings": [],
        "source_metadata": {"source": "tefas_direct_funds"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    list_payload = fund_service.get_funds_payload(tmp_path, min_aum=None)
    detail_payload = fund_service.get_fund_detail_payload(tmp_path, "AAL")

    expected = ((3.559048 / 3.5556) - 1.0) * 100.0
    assert list_payload["rows"][0]["daily_return"] == pytest.approx(expected, abs=0.0001)
    assert detail_payload["daily_return"] == pytest.approx(expected, abs=0.0001)


def test_fund_payload_overlays_newer_history_point_on_stale_snapshot(tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {
                "fund_code": "TLY",
                "date": "2026-08-18",
                "price": 8690.327521,
                "daily_return": 0.912,
                "aum": 279_794_000_000,
                "investor_count": 107_078,
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "TLY",
                "date": "2026-08-20",
                "price": 8819.35,
                "daily_return": 0.9869,
                "aum": 278_824_000_000,
                "investor_count": 108_848,
                "source": "tefasfon_funds",
            },
        ],
        source="tefasfon_funds",
    )
    snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY BIRINCI SERBEST FON",
                "price": 8690.327521,
                "daily_return": 0.912,
                "aum": 279_794_000_000,
                "investor_count": 107_078,
                "tefas_open": True,
                "as_of": "2026-08-18",
                "source": "tefasfon_funds",
            }
        ],
        "source": "tefasfon_funds",
        "as_of": "2026-08-18",
        "fetched_at": "2026-08-18T13:09:47+00:00",
        "stale": True,
        "degraded": False,
        "warnings": [],
        "source_metadata": {"source": "tefasfon_funds"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    list_payload = fund_service.get_funds_payload(tmp_path, min_aum=None)
    detail_payload = fund_service.get_fund_detail_payload(tmp_path, "TLY")

    assert list_payload["as_of"] == "2026-08-20"
    assert list_payload["rows"][0]["as_of"] == "2026-08-20"
    assert list_payload["rows"][0]["price"] == pytest.approx(8819.35)
    assert list_payload["rows"][0]["daily_return"] == pytest.approx(0.9869)
    assert list_payload["rows"][0]["investor_count"] == 108_848
    assert list_payload["source_metadata"]["snapshot_as_of"] == "2026-08-18"
    assert list_payload["source_metadata"]["price_history_as_of"] == "2026-08-20"
    assert detail_payload["as_of"] == "2026-08-20"
    assert detail_payload["price"] == pytest.approx(8819.35)


def test_collect_daily_fund_prices_uses_tefasfon_then_fintables_for_missing_codes(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"fund_code": "TLY", "name": "TERA PORTFOY TEST FONU", "founder_company": "TERA PORTFOY", "tefas_open": True},
                    {"fund_code": "PHE", "name": "PUSULA PORTFOY TEST FONU", "founder_company": "PUSULA PORTFOY", "tefas_open": True},
                ],
                "fetched_at": "2026-04-28T09:00:00+00:00",
                "source_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    class FakeTefasFonClient:
        def fetch_funds(self, *, start_date, end_date, fund_codes=None):
            assert set(fund_codes or []) == {"PHE", "TLY"}
            return [
                {
                    "fund_code": "TLY",
                    "date": "2026-04-29",
                    "price": 3.17,
                    "source": "tefasfon_funds",
                }
            ]

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            assert fund_code == "PHE"
            return [
                {
                    "fund_code": fund_code,
                    "date": "2026-04-29",
                    "price": 0,
                    "source": "fintables_udf_history",
                }
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    payload = fund_service.collect_daily_fund_prices(
        tmp_path,
        as_of=date(2026, 4, 29),
        lookback_days=1,
    )

    assert payload["status"] == "ok"
    assert payload["source"] == "tefasfon_funds"
    assert payload["valid_point_count"] == 1
    assert payload["skipped_point_count"] == 1
    assert payload["source_metadata"]["fallback_used"] is True
    assert fund_service.read_fund_price_points(tmp_path, "TLY")[0]["source"] == "tefasfon_funds"
    assert fund_service.read_fund_price_points(tmp_path, "PHE") == []


def test_target_fund_codes_for_collection_skips_env_codes_outside_tefas_open_snapshot(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"fund_code": "TLY", "name": "TERA PORTFOY TEST FONU", "founder_company": "TERA PORTFOY", "tefas_open": True},
                    {"fund_code": "PHE", "name": "PUSULA PORTFOY TEST FONU", "founder_company": "PUSULA PORTFOY", "tefas_open": False},
                ],
                "fetched_at": "2026-04-28T09:00:00+00:00",
                "source_metadata": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fund_service, "TARGET_FUND_CODES", ("TLY", "PHE", "ZZZ"))

    codes, warnings = fund_service._target_fund_codes_for_collection(tmp_path, lookback_days=1)

    assert codes == ["TLY"]
    assert "PHE" in " ".join(warnings)
    assert "ZZZ" in " ".join(warnings)


def test_collect_daily_fund_prices_uses_daily_snapshot_when_code_filtered_tefas_is_empty(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"fund_code": "TLY", "name": "TERA PORTFOY TEST FONU", "founder_company": "TERA PORTFOY", "tefas_open": True},
                ],
                "fetched_at": "2026-04-28T09:00:00+00:00",
                "source_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    class FakeTefasFonClient:
        def fetch_funds(self, *, start_date, end_date, fund_codes=None):
            if fund_codes:
                return []
            return [
                {
                    "fund_code": "TLY",
                    "date": start_date.isoformat(),
                    "price": 3.17,
                    "aum": 1_000_000,
                    "investor_count": 123,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.collect_daily_fund_prices(
        tmp_path,
        as_of=date(2026, 4, 29),
        lookback_days=1,
    )

    assert payload["status"] == "ok"
    assert "daily all-fund snapshots" in " ".join(payload["warnings"])
    stored = fund_service.read_fund_price_points(tmp_path, "TLY")
    assert stored[0]["aum"] == 1_000_000
    assert stored[0]["investor_count"] == 123


def test_collect_daily_fund_prices_skips_code_filtered_call_for_large_universe(monkeypatch, tmp_path) -> None:
    rows = [
        {"fund_code": f"T{i:03d}", "name": f"TERA PORTFOY TEST FONU {i}", "founder_company": "TERA PORTFOY", "tefas_open": True}
        for i in range(101)
    ]
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps({"rows": rows, "fetched_at": "2026-04-28T09:00:00+00:00", "source_metadata": {}}),
        encoding="utf-8",
    )

    class FakeTefasFonClient:
        def fetch_funds(self, *, start_date, end_date, fund_codes=None):
            assert fund_codes is None
            return [
                {
                    "fund_code": "T000",
                    "date": start_date.isoformat(),
                    "price": 3.17,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "_fetch_fintables_udf_history_for_codes", lambda *args, **kwargs: ([], []))

    payload = fund_service.collect_daily_fund_prices(
        tmp_path,
        as_of=date(2026, 4, 29),
        lookback_days=1,
    )

    assert payload["status"] == "ok"
    assert "large target universe" in " ".join(payload["warnings"])
    assert "fallback skipped" in " ".join(payload["warnings"])


def test_daily_snapshot_backfill_skips_weekends(monkeypatch) -> None:
    calls = []

    class FakeTefasFonClient:
        def fetch_funds(self, *, start_date, end_date, fund_codes=None):
            calls.append(start_date.isoformat())
            return [
                {
                    "fund_code": "TLY",
                    "date": start_date.isoformat(),
                    "price": 3.17,
                    "source": "tefasfon_funds",
                }
            ]

    rows, warnings = fund_service._fetch_tefasfon_daily_snapshots_for_codes(
        ["TLY"],
        start_date=date(2026, 5, 8),
        end_date=date(2026, 5, 11),
        client=FakeTefasFonClient(),
    )

    assert calls == ["2026-05-08", "2026-05-11"]
    assert len(rows) == 2
    assert warnings == []


def test_read_fund_price_points_hides_legacy_source_name(tmp_path) -> None:
    legacy_source = "te" + "fas"
    fund_service.upsert_fund_price_points(
        tmp_path,
        [{"fund_code": "TLY", "date": "2026-04-01", "price": 100.0}],
        source=legacy_source,
        fetched_at="2026-04-01T10:00:00+00:00",
    )

    points = fund_service.read_fund_price_points(tmp_path, "TLY")

    assert points[0]["source"] == "legacy_cache"


def test_read_fund_price_points_prefers_tefasfon_over_fintables_for_same_date(tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [{"fund_code": "TLY", "date": "2026-04-01", "price": 100.0, "source": "fintables_udf_history"}],
        source="fintables_udf_history",
    )
    fund_service.upsert_fund_price_points(
        tmp_path,
        [{"fund_code": "TLY", "date": "2026-04-01", "price": 101.0, "source": "tefasfon_funds"}],
        source="tefasfon_funds",
    )

    points = fund_service.read_fund_price_points(tmp_path, "TLY")

    assert points[0]["price"] == 101.0
    assert points[0]["source"] == "tefasfon_funds"


def test_read_preferred_fund_price_points_keeps_fintables_prices_without_mixing(tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "GEN", "date": "2026-04-01", "price": 100.0, "source": "fintables_udf_history"},
            {"fund_code": "GEN", "date": "2026-04-02", "price": 102.0, "source": "fintables_udf_history"},
        ],
        source="fintables_udf_history",
    )
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {
                "fund_code": "GEN",
                "date": "2026-04-01",
                "price": 101.0,
                "aum": 1_000_000,
                "investor_count": 12,
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "GEN",
                "date": "2026-04-02",
                "price": 103.0,
                "aum": 1_100_000,
                "investor_count": 13,
                "source": "tefasfon_funds",
            },
        ],
        source="tefasfon_funds",
    )

    points = fund_service._read_preferred_fund_price_points(tmp_path, "GEN")

    assert [point["price"] for point in points] == [100.0, 102.0]
    assert {point["source"] for point in points} == {"fintables_udf_history"}
    assert [point["aum"] for point in points] == [1_000_000, 1_100_000]
    assert [point["investor_count"] for point in points] == [12, 13]

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "GEN",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        auto_refresh=False,
    )

    assert [point["price"] for point in payload["points"]] == [100.0, 102.0]
    assert payload["source_metadata"]["history_source_used"] == "fintables_udf_history"
    assert payload["source_metadata"]["primary_source"] == "fintables"


def test_read_preferred_fund_price_points_keeps_newer_tefas_tail(tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "TAIL", "date": "2026-09-10", "price": 10.0, "source": "fintables_udf_history"},
            {"fund_code": "TAIL", "date": "2026-09-11", "price": 10.1, "source": "fintables_udf_history"},
        ],
        source="fintables_udf_history",
    )
    fund_service.upsert_fund_price_points(
        tmp_path,
        [{"fund_code": "TAIL", "date": "2026-09-14", "price": 10.2, "source": "tefasfon_funds"}],
        source="tefasfon_funds",
    )

    points = fund_service._read_preferred_fund_price_points(
        tmp_path,
        "TAIL",
        start_date=date(2026, 9, 10),
        end_date=date(2026, 9, 14),
    )

    assert [point["date"] for point in points] == ["2026-09-10", "2026-09-11", "2026-09-14"]
    assert points[-1]["source"] == "tefasfon_funds"


def test_read_preferred_fund_price_points_does_not_hide_older_tefas_history(tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "DOH", "date": "2026-04-02", "price": 2.0, "source": "fintables_udf_history"},
            {"fund_code": "DOH", "date": "2026-09-10", "price": 2.5, "source": "fintables_udf_history"},
        ],
        source="fintables_udf_history",
    )
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "DOH", "date": "2025-04-29", "price": 1.0, "source": "tefasfon_funds"},
            {"fund_code": "DOH", "date": "2025-05-30", "price": 1.2, "source": "tefasfon_funds"},
        ],
        source="tefasfon_funds",
    )

    points = fund_service._read_preferred_fund_price_points(
        tmp_path,
        "DOH",
        start_date=date(2025, 4, 29),
        end_date=date(2026, 9, 10),
    )

    assert points[0]["date"] == "2025-04-29"
    assert points[0]["source"] == "tefasfon_funds"
    assert points[-1]["date"] == "2026-09-10"
    assert points[-1]["source"] == "fintables_udf_history"


def test_fund_price_upsert_does_not_write_fintables_yield_summary(tmp_path) -> None:
    result = fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {
                "fund_code": "TLY",
                "date": "2026-01-02",
                "price": 100.0,
                "source": "fintables_yield_summary",
            }
        ],
        source="fintables_yield_summary",
    )

    assert result["upserted_count"] == 0
    assert result["skipped_count"] == 1
    assert result["warnings"][0]["warning"] == "non_daily_price_source"
    assert fund_service.read_fund_price_points(tmp_path, "TLY") == []


def test_write_time_return_overrides_leave_no_read_time_candidates() -> None:
    rows = [
        {
            "fund_code": "AAA",
            "as_of": "2026-11-12",
            "daily_return": 1.5,
            "daily_return_source": "tefasfon_returns",
        },
        {"fund_code": "BBB", "as_of": "2026-11-12", "daily_return": None},
        {"fund_code": "CCC", "as_of": "2026-11-12", "daily_return": 2.0},
    ]
    overrides = {("AAA", "2026-11-12"): 0.75, ("BBB", "2026-11-12"): -0.5}

    applied = fund_service._daily_return_override_values(
        rows,
        overrides,
        mark_price_history_source=True,
    )

    assert applied[0]["daily_return"] == 0.75
    assert applied[0]["daily_return_source"] == fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE
    assert applied[1]["daily_return"] == -0.5
    assert applied[1]["daily_return_source"] == fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE
    assert applied[2]["daily_return"] == 2.0
    assert "daily_return_source" not in applied[2]

    # Marked rows are no longer candidates, so a later catalogue rebuild does
    # not scan the 5-day price window for them again.
    assert fund_service._daily_return_reconciliation_candidates(applied, include_missing=True) == {}


def test_fund_price_write_preserves_snapshot_all_rows_and_postgres_queries_reuse_cache(tmp_path, monkeypatch) -> None:
    from app import fund_service

    fund_service.reset_fund_caches_for_tests()
    sample_rows = [
        {
            "fund_code": "AAA",
            "name": "Alpha Equity Fund",
            "fund_type": "Hisse Senedi Fonu",
            "founder_company": "Alpha PY",
            "manager_company": "Alpha PY",
            "price": 12.5,
            "daily_return": 1.2,
            "aum": 500_000_000.0,
            "investor_count": 1000,
            "share_count": 40_000_000.0,
            "risk_value": 6,
            "currency": "TRY",
            "as_of": "2026-11-13",
            "source": "tefasfon_funds",
            "period_returns": {"1m": 4.2},
            "tefas_open": True,
        },
        {
            "fund_code": "BBB",
            "name": "Beta Bond Fund",
            "fund_type": "Borçlanma Araçları Fonu",
            "founder_company": "Beta PY",
            "manager_company": "Beta PY",
            "price": 4.1,
            "daily_return": 0.4,
            "aum": 900_000_000.0,
            "investor_count": 2000,
            "share_count": 200_000_000.0,
            "risk_value": 3,
            "currency": "TRY",
            "as_of": "2026-11-13",
            "source": "tefasfon_funds",
            "period_returns": {"1m": 2.1},
            "tefas_open": True,
        },
    ]

    monkeypatch.setattr(fund_service, "database_enabled", lambda: True)
    monkeypatch.setattr(
        fund_service,
        "fund_listing_revision",
        lambda _p: {"catalog_generation": 7, "price_generation": 3},
    )
    monkeypatch.setattr(
        fund_service,
        "ensure_fund_snapshot_store",
        lambda _p: {"source_metadata": {"catalog_storage": "fund_snapshots"}, "snapshot_generation": 7},
    )

    read_calls = 0

    class _FakeRowsConn:
        is_postgres = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql: str, params=()):
            nonlocal read_calls
            del params
            assert "FROM fund_snapshots" in sql
            read_calls += 1

            class _Res:
                def fetchall(self):
                    return [
                        {
                            **row,
                            "period_returns_json": json.dumps(row["period_returns"]),
                            "risk_value_json": json.dumps(row["risk_value"]),
                            "attributes_json": "{}",
                            "present_fields_json": "{}",
                        }
                        for row in sample_rows
                    ]

            return _Res()

    monkeypatch.setattr(fund_service, "_connect_fund_prices_db", lambda _p: _FakeRowsConn())

    # First query hydrates _SNAPSHOT_ALL_ROWS_CACHE once; all filter/sort/detail/category
    # lookups for catalog_generation=7 must reuse RAM with 0 additional SQL reads.
    matched, total = fund_service._query_fund_snapshot_rows(
        tmp_path,
        q="beta",
        fund_type=None,
        founder=None,
        manager=None,
        risk=None,
        sort="aum",
        order="desc",
        min_aum=None,
        limit=10,
    )
    assert read_calls == 1
    assert total == 1 and [r["fund_code"] for r in matched] == ["BBB"]

    matched_sorted, total_sorted = fund_service._query_fund_snapshot_rows(
        tmp_path,
        q=None,
        fund_type=None,
        founder=None,
        manager=None,
        risk=None,
        sort="risk_value",
        order="desc",
        min_aum=600_000_000.0,
        limit=10,
    )
    assert total_sorted == 1 and [r["fund_code"] for r in matched_sorted] == ["BBB"]
    assert fund_service._find_fund_row(tmp_path, "AAA")["name"] == "Alpha Equity Fund"
    cat_rows = fund_service._read_fund_snapshot_category_returns(
        tmp_path, fund_type="Hisse Senedi Fonu", min_aum=None
    )
    assert cat_rows == [{"period_returns": {"1m": 4.2}}]
    assert read_calls == 1

    # Price-only cache invalidation must preserve the catalog-only _SNAPSHOT_ALL_ROWS_CACHE.
    fund_service._MEMORY_CACHE["fund-listing-state:dummy"] = {"cached_at": 1.0}
    fund_service._MEMORY_CACHE["latest-fund-price-points:quote:dummy"] = {"cached_at": 1.0}
    fund_service._invalidate_fund_price_memory_caches()
    assert "fund-listing-state:dummy" not in fund_service._MEMORY_CACHE
    assert "latest-fund-price-points:quote:dummy" not in fund_service._MEMORY_CACHE
    assert fund_service._peek_all_snapshot_rows(tmp_path, 7) is not None

    fund_service.reset_fund_caches_for_tests()


def test_fetch_tefasfon_daily_snapshots_skips_dates_already_covered_in_db(tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()
    target_codes = [f"T{idx:02d}" for idx in range(10)]
    seed_points = []
    # 2026-05-06 (Wed): 9/10 target codes covered -> >= 90% threshold -> should skip!
    for code in target_codes[:9]:
        seed_points.append({"fund_code": code, "date": "2026-05-06", "price": 10.0, "source": "tefasfon_funds"})
    # 2026-05-07 (Thu): only 8/10 target codes + 5 unrelated codes -> < 90% -> must NOT skip!
    for code in target_codes[:8]:
        seed_points.append({"fund_code": code, "date": "2026-05-07", "price": 10.5, "source": "tefasfon_funds"})
    for idx in range(5):
        seed_points.append({"fund_code": f"Z{idx:02d}", "date": "2026-05-07", "price": 20.0, "source": "tefasfon_funds"})
    # 2026-05-08 (Fri): 0/10 covered -> must fetch!
    fund_service.upsert_fund_price_points(tmp_path, seed_points, source="tefasfon_funds")

    snapshot_calls: list[str] = []

    class FakeTefasFonClient:
        def fetch_daily_funds_snapshot(self, as_of):
            snapshot_calls.append(as_of.isoformat())
            return [{"fonKodu": target_codes[0], "tarih": as_of.isoformat(), "fiyat": 11.0}]

    rows, warnings = fund_service._fetch_tefasfon_daily_snapshots_for_codes(
        target_codes,
        start_date=date(2026, 5, 6),
        end_date=date(2026, 5, 8),
        client=FakeTefasFonClient(),
        processed_dir=tmp_path,
    )

    assert snapshot_calls == ["2026-05-07", "2026-05-08"]
    assert len(rows) == 2
    assert warnings == []


def test_fetch_fintables_udf_history_for_codes_runs_in_parallel_with_deterministic_order(
    monkeypatch,
) -> None:
    fund_service.reset_fund_caches_for_tests()
    lock = threading.Lock()
    active = 0
    peak = 0
    order_seen: list[str] = []

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                order_seen.append(fund_code)
            try:
                if fund_code == "AAA" and fund_service._get_fintables_sticky_transport() is None:
                    # First synchronous warmup call locks sticky transport before fanout.
                    fund_service._set_fintables_sticky_transport("proxy")
                time.sleep(0.03)
                if fund_code == "CCC":
                    raise fund_service.FintablesUpstreamError("blocked CCC")
                return [{"fund_code": fund_code, "date": start_date.isoformat(), "price": 1.0}]
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(fund_service, "FUNDS_FINTABLES_FETCH_MAX_WORKERS", 4)
    try:
        rows, warnings = fund_service._fetch_fintables_udf_history_for_codes(
            ["HHH", "CCC", "AAA", "DDD", "EEE", "BBB", "FFF", "GGG"],
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 10),
            client=FakeFintablesClient(),
        )

        # Warmup ran AAA first, then remaining 7 codes fanned out across workers.
        assert order_seen[0] == "AAA"
        assert peak >= 2
        assert [row["fund_code"] for row in rows] == ["AAA", "BBB", "DDD", "EEE", "FFF", "GGG", "HHH"]
        assert warnings == ["fintables_udf_history failed for CCC: blocked CCC"]
    finally:
        fund_service.reset_fund_caches_for_tests()
