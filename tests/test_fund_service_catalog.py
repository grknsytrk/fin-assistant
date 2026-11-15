from __future__ import annotations

import json
from datetime import date

import pytest

from app import fund_service
from app.reference_data import get_instrument, upsert_instrument


def test_fund_snapshot_search_pushes_limit_into_sql_and_keeps_unknown_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert fund_service.commit_funds_snapshot(
        tmp_path,
        {
            "status": "ok",
            "as_of": "2026-10-16",
            "rows": [
                {"fund_code": "AAA", "name": "Tefas Alpha", "as_of": "2026-10-16", "custom_tefas_field": "keep-a"},
                {"fund_code": "AAB", "name": "Tefas Beta", "as_of": "2026-10-16", "custom_tefas_field": "keep-b"},
            ],
            "source_metadata": {"source": "test"},
        },
        job_id="local-test",
        owner_token="local-test",
        generation=1,
    )

    rows, total_count = fund_service._query_fund_snapshot_rows(
        tmp_path,
        q="tefas",
        fund_type=None,
        founder=None,
        manager=None,
        risk=None,
        sort="fund_code",
        order="asc",
        min_aum=None,
        limit=1,
    )

    assert total_count == 2
    assert len(rows) == 1
    assert rows[0]["custom_tefas_field"] == "keep-a"


def test_normalize_tefas_fund_list_payload_maps_list_snapshot_rows() -> None:
    rows = fund_service._normalize_tefas_fund_list_payload(
        {
            "resultList": [
                {
                    "fonKodu": "TLY",
                    "fonUnvan": "TERA PORTFÖY TEST FONU",
                    "tarih": "2026-04-29",
                    "fiyat": 3.17,
                    "portfoyBuyukluk": 1000000,
                }
            ]
        },
        source_url="https://www.tefas.gov.tr/api/funds/fonGnlBlgSiraliGetir",
    )

    assert rows[0]["fund_code"] == "TLY"
    assert rows[0]["name"] == "TERA PORTFÖY TEST FONU"
    assert rows[0]["founder_company"] == "TERA PORTFÖY"
    assert rows[0]["source"] == "tefas_list_snapshot"
    assert rows[0]["date"] == "2026-04-29"


def test_funds_payload_reads_stale_first_snapshot(tmp_path) -> None:
    snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY Yatirim Fonu",
                "fund_type": "Hisse Senedi",
                "founder_company": "TERA PORTFOY YONETIMI A.S.",
                "manager_company": "TERA PORTFOY YONETIMI A.S.",
                "price": 1.23,
                "daily_return": 0.5,
                "period_returns": {"1w": None, "1m": None, "3m": None, "6m": None, "ytd": None, "1y": None},
                "risk_value": 5,
                "currency": "TRY",
                "aum": 400_000_000,
                "as_of": "2026-04-28",
                "source": "fintables_udf_history",
            }
        ],
        "source": "fintables_udf_history",
        "source_url": "https://gate.fintables.com/barbar/udf/history",
        "as_of": "2026-04-28",
        "fetched_at": "2026-04-28T09:00:00+00:00",
        "stale": False,
        "degraded": False,
        "warnings": [],
        "source_metadata": {"source": "fintables_udf_history", "parse_status": "ok"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    payload = fund_service.get_funds_payload(tmp_path, q="yatirim", sort="fund_code", order="asc")

    assert payload["count"] == 1
    assert payload["rows"][0]["fund_code"] == "TLY"
    assert payload["source_metadata"]["cache_hit"] is True


def test_fund_snapshot_json_migrates_to_rows_and_keeps_unknown_fields(tmp_path) -> None:
    source_snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY TEST FONU",
                "fund_type": "Hisse Senedi",
                "founder_company": "TERA PORTFOY",
                "manager_company": "TERA PORTFOY",
                "price": 12.345,
                "daily_return": 0.25,
                "period_returns": {"1m": 4.5, "1y": 38.2},
                "risk_value": 5,
                "aum": 1_250_000_000,
                "investor_count": 1200,
                "share_count": 100_000,
                "as_of": "2026-04-28",
                "source": "tefasfon_funds",
                "tefas_open": True,
                "future_tefas_field": {"new_key": "preserved"},
            }
        ],
        "count": 1,
        "total_count": 1,
        "source": "tefasfon_funds",
        "as_of": "2026-04-28",
        "fetched_at": "2026-04-28T09:00:00+00:00",
        "source_metadata": {"parse_status": "ok", "provider_revision": "v1"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    snapshot_path = cache_dir / "funds_latest.json"
    snapshot_path.write_text(json.dumps(source_snapshot), encoding="utf-8")

    migrated = fund_service.load_funds_snapshot(tmp_path)
    migrated_again = fund_service.load_funds_snapshot(tmp_path)

    assert migrated["rows"][0]["fund_code"] == "TLY"
    assert migrated_again["snapshot_generation"] == migrated["snapshot_generation"]
    assert migrated["rows"][0]["future_tefas_field"] == {"new_key": "preserved"}
    assert migrated["rows"][0]["period_returns"] == {"1m": 4.5, "1y": 38.2}
    assert migrated["source_metadata"]["provider_revision"] == "v1"
    assert migrated["source_metadata"]["catalog_storage"] == "fund_snapshots"
    assert "rows" not in json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert len(json.loads((cache_dir / "funds_latest.legacy.json").read_text(encoding="utf-8"))["rows"]) == 1

    with fund_service._connect_fund_prices_db(tmp_path) as conn:
        row = conn.execute(
            "SELECT price, aum, investor_count, generation FROM fund_snapshots WHERE fund_code = ?",
            ("TLY",),
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) AS row_count FROM fund_snapshots").fetchone()["row_count"]
    assert row["price"] == pytest.approx(12.345)
    assert row["aum"] == pytest.approx(1_250_000_000)
    assert row["investor_count"] == 1200
    assert row["generation"] == migrated["snapshot_generation"]
    assert count == 1


def test_fund_snapshot_rollback_copy_is_compacted_after_retention(monkeypatch, tmp_path) -> None:
    import os
    import time

    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps({"rows": [{"fund_code": "TLY", "price": 1.0}], "source_metadata": {}}),
        encoding="utf-8",
    )
    fund_service.load_funds_snapshot(tmp_path)
    legacy_path = cache_dir / "funds_latest.legacy.json"
    expired_time = time.time() - 10
    os.utime(legacy_path, (expired_time, expired_time))
    monkeypatch.setattr(fund_service, "FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS", 5)

    assert fund_service.prune_fund_snapshot_rollback_copy(tmp_path) is True
    compact_copy = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert compact_copy["rollback_copy_expired"] is True
    assert compact_copy["row_count"] == 1
    assert "rows" not in compact_copy


def test_fund_snapshot_rollback_copy_postgres_check_avoids_full_payload_read(
    monkeypatch, tmp_path
) -> None:
    executed_queries: list[str] = []
    written_payloads: list[dict] = []

    class FakeResult:
        def fetchone(self):
            return {
                "updated_at": "2020-01-01T00:00:00+00:00",
                "row_count": 2124,
                "expired": False,
            }

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, _params=()):
            executed_queries.append(" ".join(str(query).split()))
            return FakeResult()

    monkeypatch.setattr(fund_service, "database_enabled", lambda: True)
    monkeypatch.setattr(fund_service, "connect_postgres", lambda: FakeConn())
    monkeypatch.setattr(
        fund_service,
        "write_json_cache",
        lambda _path, payload: written_payloads.append(dict(payload)),
    )
    monkeypatch.setattr(fund_service, "FUND_SNAPSHOT_ROLLBACK_TTL_SECONDS", 5)

    assert fund_service.prune_fund_snapshot_rollback_copy(tmp_path) is True
    assert len(executed_queries) == 1
    assert "jsonb_array_length" in executed_queries[0]
    assert "SELECT payload," not in executed_queries[0]
    assert len(written_payloads) == 1
    assert written_payloads[0]["row_count"] == 2124
    assert written_payloads[0]["rollback_copy_expired"] is True


def test_fund_snapshot_sql_list_filter_sort_and_detail_preserve_api_values(tmp_path) -> None:
    rows = [
        {
            "fund_code": "AAA",
            "name": "PUSULA PORTFOY HISSE FONU",
            "fund_type": "Hisse Senedi",
            "founder_company": "PUSULA PORTFOY",
            "manager_company": "PUSULA PORTFOY",
            "risk_value": 5,
            "price": 2.0,
            "aum": 100_000_000,
            "period_returns": {"1m": 10.0, "ytd": 18.0, "1y": 34.0, "6m": 20.0},
            "as_of": "2026-04-28",
            "tefas_open": True,
            "extended_label": "preserve this",
        },
        {
            "fund_code": "BBB",
            "name": "TERA PORTFOY HISSE FONU",
            "fund_type": "Hisse Senedi",
            "founder_company": "TERA PORTFOY",
            "manager_company": "TERA PORTFOY",
            "risk_value": 4,
            "price": 3.0,
            "aum": 500_000_000,
            "period_returns": {"1m": 5.0, "ytd": 9.0, "1y": 22.0, "6m": 11.0},
            "as_of": "2026-04-28",
            "tefas_open": True,
        },
        {
            "fund_code": "CCC",
            "name": "KAPALI FON",
            "fund_type": "Hisse Senedi",
            "risk_value": 3,
            "price": 9.0,
            "aum": 900_000_000,
            "tefas_open": False,
        },
    ]
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps({"status": "ok", "rows": rows, "as_of": "2026-04-28", "source_metadata": {}}),
        encoding="utf-8",
    )

    filtered = fund_service.get_funds_payload(
        tmp_path,
        q="pusula",
        fund_type="Hisse Senedi",
        risk="5",
        min_aum=50_000_000,
        sort="price",
        order="desc",
    )
    ordered = fund_service.get_funds_payload(tmp_path, sort="price", order="desc")
    detail = fund_service.get_fund_detail_payload(tmp_path, "AAA")

    assert [row["fund_code"] for row in filtered["rows"]] == ["AAA"]
    assert [row["fund_code"] for row in ordered["rows"]] == ["BBB", "AAA"]
    assert ordered["total_count"] == 3
    assert detail["extended_label"] == "preserve this"
    assert detail["category_rankings"]["category_total"] == 2


def test_fund_snapshot_present_fields_store_only_absent_keys(tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    initial = {"status": "ok", "rows": [], "as_of": "2026-04-28", "source_metadata": {}}
    (cache_dir / "funds_latest.json").write_text(json.dumps(initial), encoding="utf-8")
    fund_service.load_funds_snapshot(tmp_path)

    payload = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "CMP",
                "name": "Compact fund",
                "price": 3.5,
                "as_of": "2026-04-29",
                "period_returns": {"1y": 10.0},
                "isin": "TRACMP001",
            }
        ],
        "as_of": "2026-04-29",
        "source_metadata": {},
    }
    assert fund_service.commit_funds_snapshot(
        tmp_path, payload, job_id="local", owner_token="local", generation=0
    ) is True

    with fund_service._connect_fund_prices_db(tmp_path) as conn:
        stored = conn.execute(
            "SELECT present_fields_json FROM fund_snapshots WHERE fund_code = 'CMP'"
        ).fetchone()["present_fields_json"]
    decoded = json.loads(stored)
    assert isinstance(decoded, dict)
    assert "fund_code" not in decoded.get("absent", [])
    assert "daily_return" in decoded["absent"]
    assert len(stored) < 200
    dense = {field: 1 for field in fund_service._FUND_SNAPSHOT_FIELDS}
    assert fund_service._encode_fund_snapshot_present_fields(dense) == "{}"

    rows = fund_service.load_funds_snapshot(tmp_path)["rows"]
    row = next(item for item in rows if item["fund_code"] == "CMP")
    assert row["period_returns"] == {"1y": 10.0}
    assert row["isin"] == "TRACMP001"
    assert "daily_return" not in row


def test_fund_snapshot_record_accepts_legacy_present_fields_list() -> None:
    record = {
        "fund_code": "OLD",
        "present_fields_json": json.dumps(["fund_code", "name", "price"]),
        "attributes_json": "{}",
        "period_returns_json": "{}",
        "risk_value_json": "null",
        "name": "Legacy fund",
        "price": 1.25,
    }

    assert fund_service._snapshot_row_from_record(record) == {
        "fund_code": "OLD",
        "name": "Legacy fund",
        "price": 1.25,
    }


def test_fund_snapshot_commit_replaces_all_rows_atomically_and_serializes_local_writers(tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    initial = {
        "status": "ok",
        "rows": [{"fund_code": "OLD", "name": "Old fund", "price": 1.0, "as_of": "2026-04-28"}],
        "as_of": "2026-04-28",
        "source_metadata": {},
    }
    (cache_dir / "funds_latest.json").write_text(json.dumps(initial), encoding="utf-8")
    fund_service.load_funds_snapshot(tmp_path)

    replacement_rows = [
        {"fund_code": "NEW", "name": "New fund", "price": 5.0, "as_of": "2026-04-29"},
        {"fund_code": "KEEP", "name": "Keep fund", "price": 7.0, "as_of": "2026-04-29"},
    ]
    replacement = {"status": "ok", "rows": replacement_rows, "as_of": "2026-04-29", "source_metadata": {}}
    assert fund_service.commit_funds_snapshot(
        tmp_path, replacement, job_id="local", owner_token="local", generation=0
    ) is True
    assert fund_service.commit_funds_snapshot(
        tmp_path, {"status": "ok", "rows": [], "source_metadata": {}},
        job_id="local", owner_token="local", generation=0,
    ) is False
    current_codes = {row["fund_code"] for row in fund_service.load_funds_snapshot(tmp_path)["rows"]}
    assert current_codes == {"NEW", "KEEP"}
    with fund_service._connect_fund_prices_db(tmp_path) as conn:
        before_unchanged_commit = {
            row["fund_code"]: row["generation"]
            for row in conn.execute("SELECT fund_code, generation FROM fund_snapshots").fetchall()
        }
    assert fund_service.commit_funds_snapshot(
        tmp_path, replacement, job_id="local", owner_token="local", generation=0
    ) is True
    with fund_service._connect_fund_prices_db(tmp_path) as conn:
        after_unchanged_commit = {
            row["fund_code"]: row["generation"]
            for row in conn.execute("SELECT fund_code, generation FROM fund_snapshots").fetchall()
        }
    assert after_unchanged_commit == before_unchanged_commit

    from concurrent.futures import ThreadPoolExecutor

    def write_price(price: float) -> bool:
        concurrent_payload = {
            "status": "ok",
            "rows": [{"fund_code": "NEW", "name": "New fund", "price": price, "as_of": "2026-04-29"}],
            "as_of": "2026-04-29",
            "source_metadata": {},
        }
        return fund_service.commit_funds_snapshot(
            tmp_path, concurrent_payload, job_id="local", owner_token="local", generation=0
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(write_price, (6.0, 7.0, 8.0, 9.0)))
    with fund_service._connect_fund_prices_db(tmp_path) as conn:
        records = conn.execute("SELECT fund_code, price, generation FROM fund_snapshots").fetchall()
    assert len(records) == 1
    assert records[0]["fund_code"] == "NEW"
    assert records[0]["price"] in {6.0, 7.0, 8.0, 9.0}
    assert records[0]["generation"] == json.loads(
        (cache_dir / "funds_latest.json").read_text(encoding="utf-8")
    )["snapshot_generation"]


def test_fund_snapshot_commit_keeps_latest_duplicate_code(tmp_path) -> None:
    payload = {
        "status": "ok",
        "rows": [
            {"fund_code": "TLY", "name": "Old name", "price": 1.0},
            {"fund_code": "TLY", "name": "Current name", "price": 2.0},
        ],
        "source_metadata": {},
    }

    assert fund_service.commit_funds_snapshot(
        tmp_path, payload, job_id="local", owner_token="local", generation=0
    ) is True
    detail = fund_service.get_fund_detail_payload(tmp_path, "TLY")

    assert detail["name"] == "Current name"
    assert detail["price"] == pytest.approx(2.0)


def test_funds_payload_lists_all_open_funds_without_aum_threshold(tmp_path) -> None:
    snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "BIG",
                "name": "XYZ PORTFOY BUYUK FON",
                "founder_company": "XYZ PORTFOY",
                "price": 1.0,
                "aum": 500_000_000,
                "as_of": "2026-05-20",
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "LOW",
                "name": "TERA PORTFOY KUCUK FON",
                "founder_company": "TERA PORTFOY",
                "price": 1.0,
                "aum": 299_999_999,
                "as_of": "2026-05-20",
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "MISSING",
                "name": "TERA PORTFOY AUM BILGISIZ FON",
                "founder_company": "TERA PORTFOY",
                "price": 1.0,
                "as_of": "2026-05-20",
                "source": "tefasfon_funds",
            },
        ],
        "source": "tefasfon_funds",
        "source_url": "https://pypi.org/project/tefasfon/",
        "as_of": "2026-05-20",
        "fetched_at": "2026-05-20T09:00:00+00:00",
        "stale": False,
        "degraded": False,
        "warnings": [],
        "source_metadata": {"source": "tefasfon_funds", "parse_status": "ok"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    payload = fund_service.get_funds_payload(tmp_path, sort="fund_code", order="asc")
    search_payload = fund_service.get_funds_payload(tmp_path, sort="fund_code", order="asc", min_aum=None)

    assert [row["fund_code"] for row in payload["rows"]] == ["BIG", "LOW", "MISSING"]
    assert payload["source_metadata"]["list_min_aum"] == 0
    assert [row["fund_code"] for row in search_payload["rows"]] == ["BIG", "LOW", "MISSING"]


def test_funds_payload_keeps_explicit_aum_threshold(tmp_path) -> None:
    snapshot = {
        "status": "ok",
        "rows": [
            {"fund_code": "BIG", "name": "BUYUK FON", "aum": 500_000_000, "source": "tefasfon_funds"},
            {"fund_code": "LOW", "name": "KUCUK FON", "aum": 100_000_000, "source": "tefasfon_funds"},
        ],
        "source": "tefasfon_funds",
        "fetched_at": "2026-05-20T09:00:00+00:00",
        "stale": False,
        "degraded": False,
        "warnings": [],
        "source_metadata": {"source": "tefasfon_funds"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    payload = fund_service.get_funds_payload(
        tmp_path,
        sort="fund_code",
        order="asc",
        min_aum=300_000_000,
    )

    assert [row["fund_code"] for row in payload["rows"]] == ["BIG"]
    assert payload["source_metadata"]["list_min_aum"] == 300_000_000


def test_fund_categories_use_same_open_fund_visibility_rule(tmp_path) -> None:
    snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "OPEN",
                "name": "ACIK FON",
                "fund_type": "Hisse Senedi",
                "founder_company": "ACIK PORTFOY",
                "manager_company": "ACIK PORTFOY",
                "tefas_open": True,
                "risk_value": 5,
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "CLOSED",
                "name": "KAPALI FON",
                "fund_type": "Serbest Fon",
                "founder_company": "KAPALI PORTFOY",
                "manager_company": "KAPALI PORTFOY",
                "tefas_open": False,
                "risk_value": 7,
                "source": "tefasfon_funds",
            },
        ],
        "source": "tefasfon_funds",
        "fetched_at": "2026-05-20T09:00:00+00:00",
        "stale": False,
        "degraded": False,
        "warnings": [],
        "source_metadata": {"source": "tefasfon_funds"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    payload = fund_service.get_fund_categories_payload(tmp_path)

    assert payload["fund_types"] == ["Hisse Senedi"]
    assert payload["founder_companies"] == ["ACIK PORTFOY"]
    assert payload["risk_values"] == [5]
    assert payload["source_metadata"]["list_min_aum"] == 0


def test_funds_payload_auto_refreshes_stale_tefas_snapshot(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "rows": [
                    {
                        "fund_code": "OLD",
                        "name": "OLD FUND",
                        "price": 1.0,
                        "aum": 500_000_000,
                        "as_of": "2026-05-20",
                        "source": "tefasfon_funds",
                    }
                ],
                "source": "tefasfon_funds",
                "as_of": "2026-05-20",
                "fetched_at": "2026-05-20T09:00:00+00:00",
                "stale": False,
                "degraded": False,
                "warnings": [],
                "source_metadata": {"source": "tefasfon_funds"},
            }
        ),
        encoding="utf-8",
    )

    def fake_refresh(processed_dir, *, lookback_days):
        refreshed = {
            "status": "ok",
            "rows": [
                {
                    "fund_code": "TLY",
                    "name": "TERA PORTFOY TEST FONU",
                    "price": 5518.5,
                    "daily_return": 0.04,
                    "aum": 500_000_000,
                    "as_of": fund_service._latest_fund_snapshot_target_date().isoformat(),
                    "source": "tefasfon_funds",
                }
            ],
            "source": "tefasfon_funds",
            "as_of": fund_service._latest_fund_snapshot_target_date().isoformat(),
            "fetched_at": fund_service._utc_now_iso(),
            "stale": False,
            "degraded": False,
            "warnings": [],
            "source_metadata": {"source": "tefasfon_funds"},
        }
        (processed_dir / "funds_cache" / "funds_latest.json").write_text(json.dumps(refreshed), encoding="utf-8")
        return refreshed

    monkeypatch.setattr(fund_service, "refresh_funds_snapshot", fake_refresh)

    payload = fund_service.get_funds_payload(tmp_path, auto_refresh=True, min_aum=None)

    assert [row["fund_code"] for row in payload["rows"]] == ["TLY"]
    assert payload["rows"][0]["daily_return"] == 0.04


def test_funds_payload_empty_cache_degraded(tmp_path) -> None:
    payload = fund_service.get_funds_payload(tmp_path)

    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["rows"] == []
    assert payload["source"] == "tefasfon_funds"


def test_build_snapshot_uses_tefasfon_source_by_default() -> None:
    snapshot = fund_service._build_snapshot(
        [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY TEST FONU",
                "date": "2026-04-29",
                "price": 3.17,
                "source": "tefasfon_funds",
            }
        ]
    )

    assert snapshot["source"] == "tefasfon_funds"
    assert snapshot["rows"][0]["source"] == "tefasfon_funds"


def test_build_snapshot_computes_daily_and_weekly_returns_from_tefasfon_history() -> None:
    snapshot = fund_service._build_snapshot(
        [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY TEST FONU",
                "date": "2026-05-04",
                "price": 100.0,
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY TEST FONU",
                "date": "2026-05-08",
                "price": 105.0,
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY TEST FONU",
                "date": "2026-05-11",
                "price": 110.0,
                "source": "tefasfon_funds",
            },
        ]
    )

    row = snapshot["rows"][0]
    assert row["daily_return"] == pytest.approx(4.7619047619)
    assert row["period_returns"]["1w"] == pytest.approx(10.0)


def test_latest_tefasfon_snapshot_daily_return_skips_turkey_market_holidays() -> None:
    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            super().__init__(fund_types=["SEC"])
            self.range_calls = []

        def fetch_daily_funds_snapshot(self, as_of):
            assert as_of == date(2026, 5, 20)
            return [
                {
                    "fonKodu": "TLY",
                    "fonUnvan": "TERA PORTFOY TEST FONU",
                    "tarih": "2026-05-20",
                    "fiyat": 5516.308655,
                }
            ]

        def fetch_management_fees(self, *, fund_codes=None, lookback_days=21, as_of=None):
            return []

        def fetch_returns(self, *, start_date=None, end_date=None, fund_codes=None):
            if start_date is None or end_date is None:
                return []
            self.range_calls.append((start_date, end_date))
            if start_date == date(2026, 5, 18) and end_date == date(2026, 5, 20):
                return [{"fonKodu": "TLY", "getiriOrani": 0.4995}]
            return []

    client = FakeTefasFonClient()

    rows, warnings = client.fetch_latest_fund_list_snapshot(as_of=date(2026, 5, 20), lookback_days=1)

    assert warnings == []
    assert rows[0]["daily_return"] == pytest.approx(0.4995)
    assert (date(2026, 5, 18), date(2026, 5, 20)) in client.range_calls
    assert (date(2026, 5, 19), date(2026, 5, 20)) not in client.range_calls


def test_refresh_funds_snapshot_backfills_daily_return_from_local_prices(monkeypatch, tmp_path) -> None:
    """When TEFAS does not publish a daily return for a fund (typical for
    qualified-investor / TEFAS-closed funds), the snapshot should still pick up
    a daily_return computed from the last two locally cached price points and
    flag the fallback in source_metadata.
    """

    seed_rows = [
        {"fund_code": "UCP", "date": "2026-05-25", "price": 1.206595, "source": "tefasfon_funds"},
        {"fund_code": "UCP", "date": "2026-05-26", "price": 1.207886, "source": "tefasfon_funds"},
    ]
    fund_service.upsert_fund_price_points(tmp_path, seed_rows, source="tefasfon_funds", fetched_at="2026-05-26T16:00:00+00:00")

    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [
                {
                    "fund_code": "UCP",
                    "name": "FAKE PORTFOY UCP FONU",
                    "founder_company": "FAKE PORTFOY",
                    "manager_company": "FAKE PORTFOY",
                    "date": "2026-05-26",
                    "price": 1.207886,
                    "aum": 5_285_049_438.0,
                    "investor_count": 27,
                    "tefasDurum": True,
                    "source": "tefasfon_funds",
                    # TEFAS returns no getiriOrani for this fund
                    "gunlukGetiri": None,
                    "daily_return": None,
                }
            ], []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    row = payload["rows"][0]
    assert row["fund_code"] == "UCP"
    assert row["daily_return"] == pytest.approx(0.107, rel=1e-2)
    assert row["daily_return_source"] == fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE
    assert payload["source_metadata"]["daily_return_local_fallback_count"] == 1


def test_carry_forward_resolved_daily_returns_only_for_unchanged_price() -> None:
    existing = [
        {
            "fund_code": "CAR",
            "as_of": "2026-05-26",
            "price": 10.0,
            "daily_return": 2.0,
            "daily_return_source": fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE,
        },
        {
            "fund_code": "MOV",
            "as_of": "2026-05-26",
            "price": 10.0,
            "daily_return": 2.0,
            "daily_return_source": fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE,
        },
    ]
    fresh = [
        {"fund_code": "CAR", "as_of": "2026-05-26", "price": 10.0, "source": "tefasfon_funds"},
        {"fund_code": "MOV", "as_of": "2026-05-26", "price": 10.5, "source": "tefasfon_funds"},
    ]

    carried = fund_service._carry_forward_resolved_daily_returns(existing, fresh)

    assert carried == 1
    assert fresh[0]["daily_return"] == 2.0
    assert fresh[0]["daily_return_source"] == fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE
    # A moved price must be recomputed, so the row stays a reconciliation candidate.
    assert "daily_return" not in fresh[1]
    assert set(
        fund_service._daily_return_reconciliation_candidates(fresh, include_missing=True)
    ) == {"MOV"}


def test_merge_refresh_rows_keeps_daily_return_source_out_of_previous_quote() -> None:
    """The stored marker must describe the current quote, not the previous one."""

    existing = [
        {
            "fund_code": "STL",
            "name": "STABLE PORTFOY FONU",
            "price": 10.0,
            "daily_return": 2.0,
            "daily_return_source": fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE,
        }
    ]

    merged = fund_service._merge_refresh_rows_with_existing(
        existing,
        [{"fund_code": "STL", "price": None, "daily_return": None}],
    )

    assert merged[0]["name"] == "STABLE PORTFOY FONU"
    assert "daily_return_source" not in merged[0]

    # A fresh range return must keep its own marker so reconciliation still
    # rewrites it to the previous-close value.
    merged = fund_service._merge_refresh_rows_with_existing(
        existing,
        [
            {
                "fund_code": "STL",
                "as_of": "2026-05-26",
                "price": 10.5,
                "daily_return": 1.5,
                "daily_return_source": fund_service.TEFASFON_RETURNS_SOURCE,
            }
        ],
    )

    assert merged[0]["daily_return_source"] == fund_service.TEFASFON_RETURNS_SOURCE
    assert set(fund_service._daily_return_reconciliation_candidates(merged)) == {"STL"}


def test_reconciliation_candidates_skip_zero_price_rows() -> None:
    """Zero-price placeholder rows can never be reconciled from price history."""

    rows = [
        {"fund_code": "ZRO", "as_of": "2026-05-26", "price": 0.0},
        {"fund_code": "NUL", "as_of": "2026-05-26", "price": None},
        {"fund_code": "OKP", "as_of": "2026-05-26", "price": 1.5},
    ]

    candidates = fund_service._daily_return_reconciliation_candidates(
        rows,
        include_missing=True,
    )

    assert set(candidates) == {"NUL", "OKP"}


def test_refresh_funds_snapshot_reuses_resolved_return_for_unchanged_price(
    monkeypatch, tmp_path
) -> None:
    """An unchanged price must not trigger the all-fund price-history read again."""

    assert fund_service.commit_funds_snapshot(
        tmp_path,
        {
            "status": "ok",
            "as_of": "2026-05-26",
            "rows": [
                {
                    "fund_code": "STL",
                    "name": "STABLE PORTFOY FONU",
                    "date": "2026-05-26",
                    "price": 1.207886,
                    "daily_return": 0.107,
                    "daily_return_source": fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE,
                    "daily_return_basis_date": "2026-05-25",
                    "tefasDurum": True,
                    "source": "tefasfon_funds",
                }
            ],
        },
        job_id="seed",
        owner_token="seed",
        generation=1,
    )

    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [
                {
                    "fund_code": "STL",
                    "name": "STABLE PORTFOY FONU",
                    "founder_company": "STABLE PORTFOY",
                    "manager_company": "STABLE PORTFOY",
                    "date": "2026-05-26",
                    "price": 1.207886,
                    "aum": 5_000_000.0,
                    "investor_count": 42,
                    "tefasDurum": True,
                    "source": "tefasfon_funds",
                    "gunlukGetiri": None,
                    "daily_return": None,
                }
            ], []

    candidate_snapshots = []

    def record_candidates(_processed_dir, rows_arg, **kwargs):
        candidate_snapshots.append(
            fund_service._daily_return_reconciliation_candidates(
                rows_arg,
                include_missing=bool(kwargs.get("include_missing")),
            )
        )
        return {}

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(
        fund_service,
        "_daily_return_overrides_from_price_history",
        record_candidates,
    )

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    # The carried return leaves no reconciliation candidate, so the price
    # history window is never read for an unchanged catalogue.
    assert candidate_snapshots == [{}]
    row = payload["rows"][0]
    assert row["daily_return"] == pytest.approx(0.107)
    assert row["daily_return_source"] == fund_service.DAILY_RETURN_LOCAL_PRICE_SOURCE
    assert payload["source_metadata"]["daily_return_carried_forward_count"] == 1


def test_refresh_funds_snapshot_keeps_official_daily_return_over_local_fallback(monkeypatch, tmp_path) -> None:
    """If TEFAS does publish a daily return, the local fallback must not run."""

    seed_rows = [
        {"fund_code": "TLY", "date": "2026-05-25", "price": 3.10, "source": "tefasfon_funds"},
        {"fund_code": "TLY", "date": "2026-05-26", "price": 3.17, "source": "tefasfon_funds"},
    ]
    fund_service.upsert_fund_price_points(tmp_path, seed_rows, source="tefasfon_funds", fetched_at="2026-05-26T16:00:00+00:00")

    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [
                {
                    "fund_code": "TLY",
                    "name": "TERA PORTFOY TEST FONU",
                    "founder_company": "TERA PORTFOY",
                    "manager_company": "TERA PORTFOY",
                    "date": "2026-05-26",
                    "price": 3.17,
                    "aum": 1_000_000,
                    "investor_count": 123,
                    "tefasDurum": True,
                    "source": "tefasfon_funds",
                    "gunlukGetiri": 0.42,  # official value should win
                }
            ], []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    row = payload["rows"][0]
    assert row["daily_return"] == pytest.approx(0.42)
    assert "daily_return_local_fallback_count" not in payload["source_metadata"]


def test_refresh_funds_snapshot_skips_fallback_when_local_gap_too_large(monkeypatch, tmp_path) -> None:
    """If the previous local point is older than the configured gap window,
    we leave daily_return None rather than emitting a stale value."""

    seed_rows = [
        {"fund_code": "STAL", "date": "2026-05-01", "price": 1.10, "source": "tefasfon_funds"},
        {"fund_code": "STAL", "date": "2026-05-26", "price": 1.20, "source": "tefasfon_funds"},
    ]
    fund_service.upsert_fund_price_points(tmp_path, seed_rows, source="tefasfon_funds", fetched_at="2026-05-26T16:00:00+00:00")

    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [
                {
                    "fund_code": "STAL",
                    "name": "STALE FUND",
                    "founder_company": "STALE",
                    "manager_company": "STALE",
                    "date": "2026-05-26",
                    "price": 1.20,
                    "aum": 1_000_000,
                    "investor_count": 1,
                    "tefasDurum": True,
                    "source": "tefasfon_funds",
                    "gunlukGetiri": None,
                }
            ], []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    row = payload["rows"][0]
    assert row["daily_return"] is None
    assert "daily_return_local_fallback_count" not in payload["source_metadata"]


def test_refresh_funds_snapshot_uses_tefasfon_source(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [
                {
                    "fund_code": "TLY",
                    "name": "TERA PORTFOY TEST FONU",
                    "founder_company": "TERA PORTFOY YONETIMI A.S.",
                    "manager_company": "TERA PORTFOY YONETIMI A.S.",
                    "date": "2026-04-29",
                    "price": 3.17,
                    "aum": 1_000_000,
                    "investor_count": 123,
                    "tefasDurum": True,
                    "source": "tefasfon_funds",
                }
            ], []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    assert payload["source"] == "tefasfon_funds"
    assert payload["rows"][0]["source"] == "tefasfon_funds"
    assert payload["rows"][0]["name"] == "TERA PORTFOY TEST FONU"
    assert payload["rows"][0]["price"] == 3.17
    assert payload["rows"][0]["tefas_open"] is True
    stored = fund_service.read_fund_price_points(tmp_path, "TLY")
    assert stored[0]["source"] == "tefasfon_funds"
    assert stored[0]["aum"] == 1_000_000
    assert stored[0]["investor_count"] == 123
    assert stored[0]["tefas_open"] is True
    reference = get_instrument(tmp_path, "fund", "TLY")
    assert reference is not None
    assert reference["name"] == "TERA PORTFOY TEST FONU"
    assert reference["metadata"]["founder_company"] == "TERA PORTFOY YONETIMI A.S."


def test_refresh_funds_snapshot_filters_tefas_closed_rows(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [
                {
                    "fund_code": "TLY",
                    "name": "TERA PORTFOY ACIK FONU",
                    "date": "2026-04-29",
                    "price": 3.17,
                    "tefasDurum": True,
                    "source": "tefasfon_funds",
                },
                {
                    "fund_code": "PHE",
                    "name": "PUSULA PORTFOY KAPALI FONU",
                    "date": "2026-04-29",
                    "price": 4.2,
                    "tefasDurum": False,
                    "source": "tefasfon_funds",
                },
                {
                    "fund_code": "ABG",
                    "name": "ATLAS PORTFOY DURUMSUZ FONU",
                    "date": "2026-04-29",
                    "price": 5.2,
                    "source": "tefasfon_funds",
                },
            ], []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    assert sorted(row["fund_code"] for row in payload["rows"]) == ["ABG", "TLY"]
    assert "tefas_open_only skipped 1 closed fund rows" in " ".join(payload["warnings"])
    assert fund_service.read_fund_price_points(tmp_path, "PHE") == []


def test_refresh_funds_snapshot_noops_when_tefas_returns_empty(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [], ["empty"]

    class EmptyDirectTefasClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [], ["direct-empty"]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "TefasClient", lambda: EmptyDirectTefasClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["rows"] == []
    assert payload["source_metadata"]["parse_status"] == "upstream_unavailable"
    assert "empty" in " ".join(payload["warnings"])


def test_refresh_funds_snapshot_does_not_downgrade_newer_snapshot(monkeypatch, tmp_path) -> None:
    existing = fund_service._build_snapshot(
        [{"fund_code": "TLY", "date": "2026-08-21", "price": 8.865, "tefasDurum": True}]
    )
    fund_service._write_json(fund_service._snapshot_path(tmp_path), existing)

    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot_result(self, *, as_of, lookback_days, enrich=False):
            return fund_service.FundSnapshotFetchResult(
                rows=[{"fund_code": "TLY", "date": "2026-08-20", "price": 8.819, "tefasDurum": True}],
                resolution_status="resolved_previous_date",
                requested_as_of="2026-08-21",
                resolved_as_of="2026-08-20",
                attempted_dates=["2026-08-21", "2026-08-20"],
            )

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    payload = fund_service.refresh_funds_snapshot(
        tmp_path, lookback_days=14, persist_reference_data=False, backfill_daily_returns=False
    )

    assert payload["as_of"] == "2026-08-21"
    assert payload["snapshot_action"] == "retained_newer"
    assert payload["resolution_status"] == "resolved_previous_date"


def test_refresh_funds_snapshot_replaces_stale_rows_on_same_date(monkeypatch, tmp_path) -> None:
    target = date(2026, 8, 21)
    monkeypatch.setattr(fund_service, "_latest_fund_snapshot_target_date", lambda *_args: target)
    monkeypatch.setattr(fund_service, "_turkey_market_today", lambda: target)
    existing = fund_service._build_snapshot(
        [
            {"fund_code": "TLY", "date": target.isoformat(), "price": 8.865, "tefasDurum": True},
            {"fund_code": "STALE", "date": target.isoformat(), "price": 4.2, "tefasDurum": True},
        ],
        warnings=["previous current-day refresh failed"],
    )
    fund_service._write_json(fund_service._snapshot_path(tmp_path), existing)

    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot_result(self, *, as_of, lookback_days, enrich=False):
            return fund_service.FundSnapshotFetchResult(
                rows=[
                    {
                        "fund_code": "TLY",
                        "date": target.isoformat(),
                        "price": 9.1,
                        "tefasDurum": True,
                        "source": "tefasfon_funds",
                    }
                ],
                resolution_status="available",
                requested_as_of=as_of.isoformat(),
                resolved_as_of=target.isoformat(),
                attempted_dates=[target.isoformat()],
                source="tefasfon_funds",
            )

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    refreshed = fund_service.refresh_funds_snapshot(
        tmp_path,
        lookback_days=1,
        persist_reference_data=False,
        backfill_daily_returns=False,
    )

    assert [row["fund_code"] for row in refreshed["rows"]] == ["TLY"]
    assert refreshed["rows"][0]["price"] == pytest.approx(9.1)
    assert refreshed["warnings"] == []
    assert refreshed["source_metadata"]["warnings"] == []
    assert refreshed["snapshot_action"] == "updated"


def test_load_funds_snapshot_hides_warnings_after_successful_current_probe(monkeypatch, tmp_path) -> None:
    target = date(2026, 8, 21)
    monkeypatch.setattr(fund_service, "_latest_fund_snapshot_target_date", lambda *_args: target)
    now = fund_service._utc_now_iso()
    payload = fund_service._build_snapshot(
        [{"fund_code": "TLY", "date": target.isoformat(), "price": 8.865, "tefasDurum": True}],
        warnings=["old TEFAS list error", "old TEFAS list error"],
    )
    payload["source_metadata"].update(
        {
            "resolution_status": "available",
            "requested_as_of": target.isoformat(),
            "last_checked_at": now,
            "upstream_errors": [],
        }
    )
    fund_service._write_json(fund_service._snapshot_path(tmp_path), payload)

    loaded = fund_service.load_funds_snapshot(tmp_path)

    assert loaded["warnings"] == []
    assert loaded["source_metadata"]["warnings"] == []


def test_weekend_snapshot_is_not_refreshed_or_reported_stale(monkeypatch, tmp_path) -> None:
    existing = fund_service._build_snapshot(
        [{"fund_code": "TLY", "date": "2026-09-04", "price": 8.865, "tefasDurum": True}]
    )
    existing.update({"stale": True, "degraded": True, "warnings": ["temporary upstream error"]})
    fund_service._write_json(fund_service._snapshot_path(tmp_path), existing)
    monkeypatch.setattr(fund_service, "_turkey_market_today", lambda: date(2026, 9, 6))

    class UnexpectedTefasClient:
        def fetch_latest_fund_list_snapshot_result(self, **_kwargs):
            pytest.fail("weekend should not call TEFAS")

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: UnexpectedTefasClient())

    refreshed = fund_service.refresh_funds_snapshot(
        tmp_path, lookback_days=14, persist_reference_data=False, backfill_daily_returns=False
    )
    listed = fund_service.get_funds_payload(tmp_path)

    assert refreshed["resolution_status"] == "market_closed"
    assert refreshed["snapshot_action"] == "retained_current_market_closed"
    assert refreshed["stale"] is False
    assert refreshed["warnings"] == []
    assert listed["stale"] is False
    assert listed["degraded"] is False
    assert listed["warnings"] == []


def test_refresh_funds_snapshot_uses_bounded_direct_fallback(monkeypatch, tmp_path) -> None:
    calls = {"lookback_days": None}

    class EmptyTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days, enrich=True):
            return [], ["tefasfon-empty"]

    class DirectFallbackClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            calls["lookback_days"] = lookback_days
            return [
                {
                    "fund_code": "TLY",
                    "name": "DIRECT FALLBACK FUND",
                    "date": "2026-05-20",
                    "price": 10.0,
                    "tefasDurum": True,
                }
            ], []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: EmptyTefasFonClient())
    monkeypatch.setattr(fund_service, "TefasClient", lambda: DirectFallbackClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=10)

    assert payload["rows"][0]["fund_code"] == "TLY"
    assert calls["lookback_days"] == 10
    assert "tefas_direct_funds fallback used" in payload["warnings"]


def test_fund_tax_info_maps_known_fund_categories() -> None:
    assert fund_service._fund_tax_info("Hisse Senedi Şemsiye Fonu") == "%0"
    assert fund_service._fund_tax_info("Borsa Yatırım Fonu") == "%0"
    assert fund_service._fund_tax_info("Serbest Şemsiye Fonu") == "%10"
    assert fund_service._fund_tax_info("Para Piyasası Şemsiye Fonu") == "%10"
    assert fund_service._fund_tax_info("Bireysel Emeklilik Fonu") == "—"
    assert fund_service._fund_tax_info("") is None
    assert fund_service._fund_tax_info(None) is None


def test_merge_tefasfon_management_fees_attaches_fee_columns() -> None:
    fund_rows = [
        {"fonKodu": "TLY", "fonUnvan": "TERA"},
        {"fonKodu": "PHE", "fonUnvan": "PUSULA"},
    ]
    fee_rows = [
        {
            "fonKodu": "TLY",
            "uygulananYu1Y": "0",
            "fonIcTuzukYu1G": "2",
            "fonTopGiderKesoran": "2",
        },
        {
            "fonKodu": "PHE",
            "uygulananYu1Y": "2,5",
            "fonIcTuzukYu1G": "2,5",
            "fonTopGiderKesoran": "1,65",
        },
    ]
    merged = fund_service._merge_tefasfon_management_fees(fund_rows, fee_rows)
    by_code = {row["fonKodu"]: row for row in merged}
    assert by_code["TLY"]["management_fee_applied"] == 0
    assert by_code["TLY"]["management_fee_prospectus"] == 2
    assert by_code["TLY"]["total_expense_ratio"] == 2
    assert by_code["PHE"]["management_fee_applied"] == 2.5
    assert by_code["PHE"]["total_expense_ratio"] == pytest.approx(1.65)


def test_get_fund_detail_payload_falls_back_to_reference_data(tmp_path) -> None:
    snapshot = {
        "status": "ok",
        "rows": [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFÖY BİRİNCİ SERBEST FON",
                "fund_type": "Serbest Şemsiye Fonu",
                "founder_company": "TERA PORTFÖY",
                "manager_company": "TERA PORTFÖY",
                "tefas_open": True,
                "price": 5479.81,
                "daily_return": 3.6,
                "period_returns": {},
                "risk_value": 7,
                "currency": "TRY",
                "as_of": "2026-05-25",
                "source": "tefasfon_funds",
                "aum": 137_000_000_000.0,
                "investor_count": 82_659,
                "share_count": 25_000_000.0,
                # Snapshot intentionally lacks the management fee columns to simulate
                # a partial refresh. Reference data should plug them back in.
            },
        ],
        "as_of": "2026-05-25",
        "fetched_at": "2026-05-25T13:11:01+00:00",
        "stale": False,
        "source": "tefasfon_funds",
        "source_url": "https://pypi.org/project/tefasfon/",
        "source_metadata": {"source": "tefasfon_funds", "parse_status": "ok"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    upsert_instrument(
        tmp_path,
        kind="fund",
        symbol="TLY",
        name="TERA PORTFÖY BİRİNCİ SERBEST FON",
        source="tefasfon_funds",
        metadata={
            "fund_type": "Serbest Şemsiye Fonu",
            "management_fee_applied": 0.0,
            "management_fee_prospectus": 2.0,
            "total_expense_ratio": 2.0,
            "tax_info": "%10",
        },
    )

    payload = fund_service.get_fund_detail_payload(tmp_path, "TLY")

    assert payload["management_fee_applied"] == 0.0
    assert payload["management_fee_prospectus"] == 2.0
    assert payload["total_expense_ratio"] == 2.0
    # `management_fee` should fall back to the prospectus value when the applied
    # rate is 0 (typical for performance-fee serbest funds like TLY).
    assert payload["management_fee"] == 2.0
    assert payload["tax_info"] == "%10"


def test_get_fund_detail_payload_includes_category_rankings(tmp_path) -> None:
    def row(
        code: str,
        one_month: float,
        ytd: float,
        one_year: float,
        six_month: float,
        *,
        aum: float = 1_000_000_000,
    ) -> dict:
        return {
            "fund_code": code,
            "name": f"{code} TEST FONU",
            "fund_type": "Hisse Senedi Şemsiye Fonu",
            "founder_company": "TEST PORTFÖY",
            "manager_company": "TEST PORTFÖY",
            "tefas_open": True,
            "price": 1.0,
            "daily_return": 0.1,
            "period_returns": {"1m": one_month, "ytd": ytd, "1y": one_year, "6m": six_month},
            "risk_value": 6,
            "currency": "TRY",
            "as_of": "2026-05-26",
            "source": "tefasfon_funds",
            "aum": aum,
        }

    snapshot = {
        "status": "ok",
        "rows": [
            row("TLY", 5.0, 8.0, 10.0, 6.0),
            row("AAA", 10.0, 12.0, 8.0, 5.0),
            row("BBB", 7.0, 7.0, 12.0, 4.0),
            row("CCC", -1.0, 4.0, 6.0, 9.0),
            row("LOW", 20.0, 20.0, 20.0, 20.0, aum=1_000),
            {**row("PPF", 30.0, 30.0, 30.0, 30.0), "fund_type": "Para Piyasası Şemsiye Fonu"},
        ],
        "as_of": "2026-05-26",
        "fetched_at": "2026-05-26T13:11:01+00:00",
        "stale": False,
        "source": "tefasfon_funds",
        "source_url": "https://pypi.org/project/tefasfon/",
        "source_metadata": {"source": "tefasfon_funds", "parse_status": "ok"},
    }
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(json.dumps(snapshot), encoding="utf-8")

    payload = fund_service.get_fund_detail_payload(tmp_path, "TLY")

    rankings = payload["category_rankings"]
    assert rankings["category"] == "Hisse Senedi Şemsiye Fonu"
    assert rankings["category_total"] == 5
    monthly = next(item for item in rankings["items"] if item["key"] == "1m")
    assert monthly["rank"] == 4
    assert monthly["total"] == 5
    assert monthly["top_percentile"] == 40
