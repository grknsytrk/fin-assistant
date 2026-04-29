from __future__ import annotations

import json
from datetime import date

from app import fund_service


def test_fintables_defaults_match_current_history_contract() -> None:
    assert fund_service.FINTABLES_UDF_HISTORY_ENDPOINT.endswith("/barbar/udf/history")
    assert fund_service.FINTABLES_YIELD_SUMMARY_ENDPOINT.endswith("/barbar/server/yield")


def test_normalize_fintables_udf_history_payload_uses_close_series() -> None:
    rows = fund_service._normalize_fintables_udf_history_payload(
        {
            "s": "ok",
            "t": [1775001600, 1775088000],
            "o": [99.0, 100.0],
            "h": [101.0, 102.0],
            "l": [98.0, 99.0],
            "c": [100.0, 0.0],
            "v": [10, 11],
        },
        fund_code="tly",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
    )

    assert [row["date"] for row in rows] == ["2026-04-01", "2026-04-02"]
    assert [row["price"] for row in rows] == [100.0, 0.0]
    assert rows[0]["source"] == "fintables_udf_history"


def test_normalize_fintables_yield_summary_keeps_periods_outside_daily_series() -> None:
    summary = fund_service._normalize_fintables_yield_summary_payload(
        {
            "ytd": {
                "prev_close_date": "2026-01-02",
                "prev_close": 1.2,
                "high": 1.5,
                "low": 1.1,
            }
        },
        fund_code="PHE",
    )

    assert summary["source"] == "fintables_yield_summary"
    assert summary["periods"]["ytd"]["prev_close"] == 1.2
    assert summary["periods"]["ytd"]["high"] == 1.5


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


def test_funds_payload_empty_cache_degraded(tmp_path) -> None:
    payload = fund_service.get_funds_payload(tmp_path)

    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["rows"] == []
    assert payload["source"] == "fintables_udf_history"


def test_build_snapshot_uses_fintables_source_by_default() -> None:
    snapshot = fund_service._build_snapshot(
        [
            {
                "fund_code": "TLY",
                "name": "TERA PORTFOY TEST FONU",
                "date": "2026-04-29",
                "price": 3.17,
                "source": "fintables_udf_history",
            }
        ]
    )

    assert snapshot["source"] == "fintables_udf_history"
    assert snapshot["rows"][0]["source"] == "fintables_udf_history"


def test_refresh_funds_snapshot_uses_fintables_codes_from_existing_snapshot(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "fund_code": "TLY",
                        "name": "TERA PORTFOY TEST FONU",
                        "founder_company": "TERA PORTFOY YONETIMI A.S.",
                        "source": "legacy_cache",
                    }
                ],
                "fetched_at": "2026-04-28T09:00:00+00:00",
                "source_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            assert fund_code == "TLY"
            return [
                {
                    "fund_code": "TLY",
                    "date": "2026-04-29",
                    "price": 3.17,
                    "source": "fintables_udf_history",
                }
            ]

    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    assert payload["source"] == "fintables_udf_history"
    assert payload["rows"][0]["name"] == "TERA PORTFOY TEST FONU"
    assert payload["rows"][0]["price"] == 3.17


def test_refresh_funds_snapshot_requires_known_target_codes(tmp_path) -> None:
    try:
        fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)
    except fund_service.FintablesUpstreamError as exc:
        assert "no target fund codes" in str(exc)
    else:
        raise AssertionError("expected Fintables target-code error")


def test_collect_daily_fund_prices_writes_valid_fintables_udf_rows_only(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir()
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"fund_code": "TLY", "name": "TERA PORTFOY TEST FONU", "founder_company": "TERA PORTFOY"},
                    {"fund_code": "PHE", "name": "PUSULA PORTFOY TEST FONU", "founder_company": "PUSULA PORTFOY"},
                ],
                "fetched_at": "2026-04-28T09:00:00+00:00",
                "source_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            return [
                {
                    "fund_code": fund_code,
                    "date": "2026-04-29",
                    "price": 3.17 if fund_code == "TLY" else 0,
                    "source": "fintables_udf_history",
                }
            ]

    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    payload = fund_service.collect_daily_fund_prices(
        tmp_path,
        as_of=date(2026, 4, 29),
        lookback_days=1,
    )

    assert payload["status"] == "ok"
    assert payload["source"] == "fintables_udf_history"
    assert payload["valid_point_count"] == 1
    assert payload["skipped_point_count"] == 1
    assert fund_service.read_fund_price_points(tmp_path, "TLY")[0]["source"] == "fintables_udf_history"
    assert fund_service.read_fund_price_points(tmp_path, "PHE") == []


def test_refresh_fund_performance_does_not_fallback_when_fintables_fails(monkeypatch, tmp_path) -> None:
    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            raise fund_service.FintablesUpstreamError("blocked")

    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    try:
        fund_service.refresh_fund_performance(
            tmp_path,
            "TLY",
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 29),
        )
    except fund_service.FintablesUpstreamError as exc:
        assert "blocked" in str(exc)
    else:
        raise AssertionError("expected Fintables error")


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


def test_get_fund_allocations_payload_hides_legacy_cache(tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache" / "allocations"
    cache_dir.mkdir(parents=True)
    (cache_dir / "TLY.json").write_text(
        json.dumps({"fund_code": "TLY", "source": "te" + "fas", "allocations": [{"allocation_type": "hs"}]}),
        encoding="utf-8",
    )

    payload = fund_service.get_fund_allocations_payload(tmp_path, "TLY")

    assert payload["status"] == "unavailable"
    assert payload["source"] == "fintables"
    assert payload["allocations"] == []
