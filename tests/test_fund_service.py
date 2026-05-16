from __future__ import annotations

import json
from datetime import date

import pytest

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
    assert rows[0]["raw"]["point"] == {"t": 1775001600, "c": 100.0}
    assert [row["date"] for row in fund_service._valid_performance_points(rows, "TLY")] == ["2026-04-01"]


def test_normalize_fintables_udf_history_requires_ok_and_matching_t_c() -> None:
    for payload in (
        {"s": "no_data", "t": [], "c": []},
        {"s": "ok", "t": [1775001600], "c": []},
        {"s": "ok", "t": "bad", "c": []},
    ):
        try:
            fund_service._normalize_fintables_udf_history_payload(
                payload,
                fund_code="TLY",
                start_date=date(2026, 4, 1),
                end_date=date(2026, 4, 2),
            )
        except fund_service.FintablesUpstreamError:
            continue
        else:
            raise AssertionError("expected Fintables UDF validation error")


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


def test_fintables_headers_include_optional_env_values(monkeypatch) -> None:
    monkeypatch.setenv("RAGFIN_FINTABLES_COOKIE", "cf_clearance=abc")
    monkeypatch.setenv("RAGFIN_FINTABLES_AUTHORIZATION", "Bearer token")
    monkeypatch.setenv("RAGFIN_FINTABLES_EXTRA_HEADERS_JSON", json.dumps({"X-Test": "yes"}))

    headers = fund_service.FintablesClient()._headers("PHE")

    assert headers["Cookie"] == "cf_clearance=abc"
    assert headers["Authorization"] == "Bearer token"
    assert headers["X-Test"] == "yes"


def test_fintables_waf_html_is_clear_upstream_error() -> None:
    try:
        fund_service._decode_fintables_json_response(
            403,
            {"content-type": "text/html"},
            b"<html><title>Just a moment...</title>cloudflare</html>",
            context="Fintables UDF history",
        )
    except fund_service.FintablesUpstreamError as exc:
        assert "Fintables Gate blocked by WAF/Cloudflare" in str(exc)
    else:
        raise AssertionError("expected WAF upstream error")


def test_fintables_client_uses_curl_fallback_after_waf(monkeypatch) -> None:
    class FakeResponse:
        status_code = 403
        headers = {"content-type": "text/html"}
        content = b"<html><title>Just a moment...</title>cloudflare</html>"

    class FakeHttpxClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, *args, **kwargs):
            return FakeResponse()

    fallback_calls = []

    def fake_curl_payload(url, *, params, headers, timeout_seconds, context):
        fallback_calls.append((url, params, headers, context))
        return {
            "1w": {
                "prev_close_date": "2026-04-23T21:00:00Z",
                "prev_close": 10,
                "high": 12,
                "low": 9,
            }
        }

    monkeypatch.setattr(fund_service.httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(fund_service, "_fintables_curl_payload", fake_curl_payload)

    payload = fund_service.FintablesClient().fetch_yield_summary("TLY")

    assert payload["periods"]["1w"]["prev_close"] == 10
    assert fallback_calls
    assert fallback_calls[0][3] == "Fintables yield summary"


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


def test_valid_performance_points_computes_missing_daily_returns() -> None:
    points = fund_service._valid_performance_points(
        [
            {"fund_code": "TLY", "date": "2026-05-08", "price": 100.0, "source": "fintables_udf_history"},
            {"fund_code": "TLY", "date": "2026-05-11", "price": 103.0, "source": "fintables_udf_history"},
        ],
        "TLY",
    )

    assert points[0]["daily_return"] is None
    assert points[1]["daily_return"] == pytest.approx(3.0)


def test_fund_history_gap_ignores_turkey_market_holidays() -> None:
    warnings = fund_service._history_internal_gap_warnings(
        [
            {"fund_code": "TLY", "date": "2021-07-19", "price": 1.0},
            {"fund_code": "TLY", "date": "2021-07-26", "price": 1.1},
        ]
    )

    assert warnings == []
    assert fund_service._business_days_between(date(2021, 7, 20), date(2021, 7, 23)) == 0


def test_fund_history_gap_still_warns_for_real_business_day_gap() -> None:
    warnings = fund_service._history_internal_gap_warnings(
        [
            {"fund_code": "TLY", "date": "2021-08-02", "price": 1.0},
            {"fund_code": "TLY", "date": "2021-08-09", "price": 1.1},
        ]
    )

    assert warnings == [
        "Fund history has an internal gap: previous=2021-08-02, next=2021-08-09, missing_business_days=4."
    ]


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

    assert [row["fund_code"] for row in payload["rows"]] == ["TLY"]
    assert "tefas_open_only skipped 2 non-open fund rows" in " ".join(payload["warnings"])
    assert fund_service.read_fund_price_points(tmp_path, "PHE") == []


def test_refresh_funds_snapshot_noops_when_tefas_returns_empty(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_latest_fund_list_snapshot(self, *, as_of, lookback_days):
            return [], ["empty"]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_funds_snapshot(tmp_path, lookback_days=1)

    assert payload["status"] == "unavailable"
    assert payload["degraded"] is True
    assert payload["rows"] == []
    assert payload["source_metadata"]["parse_status"] == "empty_tefasfon_funds"
    assert "empty" in " ".join(payload["warnings"])


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


def test_refresh_fund_performance_uses_tefasfon_primary(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            return [
                {"fund_code": fund_code, "date": "2026-04-28", "price": 3.1, "source": "tefasfon_funds"},
                {"fund_code": fund_code, "date": "2026-04-29", "price": 3.2, "source": "tefasfon_funds"},
            ]

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            raise AssertionError("Fintables should not be called when TEFAS has data")

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    payload = fund_service.refresh_fund_performance(
        tmp_path,
        "TLY",
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 29),
    )

    assert payload["status"] == "ok"
    assert payload["source_metadata"]["history_source_used"] == "tefasfon_funds"
    assert payload["source_metadata"]["history_source_policy"] == "tefasfon_primary_fintables_fallback"
    assert payload["source_metadata"]["fallback_used"] is False


def test_refresh_fund_performance_falls_back_to_fintables_when_tefasfon_fails(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            raise fund_service.TefasUpstreamError("blocked")

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            return [
                {"fund_code": fund_code, "date": "2026-04-29", "price": 3.2, "source": "fintables_udf_history"}
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    payload = fund_service.refresh_fund_performance(
        tmp_path,
        "TLY",
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 29),
    )

    assert payload["status"] == "ok"
    assert payload["source_metadata"]["history_source_used"] == "fintables_udf_history"
    assert payload["source_metadata"]["fallback_used"] is True
    assert "blocked" in " ".join(payload["source_metadata"]["warnings"])


def test_get_fund_performance_payload_fetches_tefasfon_on_cache_miss(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            calls.append((fund_code, start_date, end_date))
            return [
                {
                    "fund_code": fund_code,
                    "date": "2026-04-28",
                    "price": 3.1,
                    "source": "tefasfon_funds",
                },
                {
                    "fund_code": fund_code,
                    "date": "2026-04-29",
                    "price": 3.2,
                    "source": "tefasfon_funds",
                },
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "TLY",
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 29),
    )
    cached_payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "TLY",
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 29),
    )

    assert len(calls) == 1
    assert payload["status"] == "ok"
    assert [point["price"] for point in payload["points"]] == [3.1, 3.2]
    assert payload["source_metadata"]["history_source_used"] == "tefasfon_funds"
    assert payload["source_metadata"]["history_source_policy"] == "tefasfon_primary_fintables_fallback"
    assert payload["source_metadata"]["final_points_count"] == 2
    assert payload["source_metadata"]["date_min"] == "2026-04-28"
    assert payload["source_metadata"]["date_max"] == "2026-04-29"
    assert payload["source_metadata"]["backfill_used"] is True
    assert payload["source_metadata"]["fallback_used"] is False
    assert cached_payload["source_metadata"]["cache_hit"] is True
    assert cached_payload["source_metadata"]["backfill_used"] is False
    assert fund_service.read_fund_price_points(tmp_path, "TLY")[0]["source"] == "tefasfon_funds"


def test_get_fund_performance_payload_defaults_to_full_history(monkeypatch, tmp_path) -> None:
    calls = []
    today = date.today()

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            calls.append((fund_code, start_date, end_date))
            return [
                {"fund_code": fund_code, "date": "2020-01-02", "price": 1.0, "source": "tefasfon_funds"},
                {"fund_code": fund_code, "date": today.isoformat(), "price": 3.2, "source": "tefasfon_funds"},
            ]

    monkeypatch.setattr(fund_service, "FUNDS_FULL_HISTORY_START_DATE", "2020-01-01")
    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")
    cached_payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")

    assert calls
    assert calls[0][1] == date(2020, 1, 1)
    assert [point["date"] for point in payload["points"]] == ["2020-01-02", today.isoformat()]
    assert payload["source_metadata"]["requested_start_date"] == "2020-01-01"
    assert cached_payload["source_metadata"]["cache_hit"] is True
    assert len(calls) == 1


def test_get_fund_performance_payload_accepts_weekend_range_start(monkeypatch, tmp_path) -> None:
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "TLY", "date": "2026-04-13", "price": 100.0, "source": "tefasfon_funds"},
            {"fund_code": "TLY", "date": "2026-04-14", "price": 101.0, "source": "tefasfon_funds"},
        ],
        source="tefasfon_funds",
    )

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            raise AssertionError("cache should satisfy weekend-start coverage")

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "TLY",
        start_date=date(2026, 4, 11),
        end_date=date(2026, 4, 14),
    )

    assert payload["status"] == "ok"
    assert [point["date"] for point in payload["points"]] == ["2026-04-13", "2026-04-14"]
    assert payload["source_metadata"]["backfill_used"] is False


def test_get_fund_performance_payload_does_not_use_legacy_when_sources_fail(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache" / "history"
    cache_dir.mkdir(parents=True)
    (cache_dir / "TLY.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "fund_code": "TLY",
                        "date": "2026-04-28",
                        "price": 9.9,
                        "source": "te" + "fas",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            raise fund_service.TefasUpstreamError("tefas blocked")

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            raise fund_service.FintablesUpstreamError("fintables blocked")

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "TLY",
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 29),
    )

    assert payload["status"] == "unavailable"
    assert payload["points"] == []
    assert "tefas blocked" in " ".join(payload["source_metadata"]["warnings"])
    assert "fintables blocked" in " ".join(payload["source_metadata"]["warnings"])
    assert payload["source_metadata"]["warning"]
    assert payload["source_metadata"]["history_source_used"] is None
    assert payload["source_metadata"]["history_source_policy"] == "tefasfon_primary_fintables_fallback"
    assert payload["source_metadata"]["final_points_count"] == 0
    assert payload["source_metadata"]["backfill_used"] is True
    assert fund_service.read_fund_price_points(tmp_path, "TLY") == []


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


def test_get_fund_yield_summary_payload_returns_unavailable_without_price_writes(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_yield_summary(self, fund_code, **kwargs):
            raise fund_service.TefasUpstreamError("tefas blocked")

    def fail_summary(fund_code):
        raise fund_service.FintablesUpstreamError("Fintables Gate blocked by WAF/Cloudflare")

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "fetch_fintables_yield_summary", fail_summary)

    payload = fund_service.get_fund_yield_summary_payload("TLY")

    assert payload["status"] == "unavailable"
    assert payload["periods"] == {}
    assert payload["source_metadata"]["summary_source_used"] == "fintables_yield_summary"
    assert payload["source_metadata"]["writes_fund_prices"] is False
    assert "WAF" in " ".join(payload["source_metadata"]["warnings"])
    assert fund_service.read_fund_price_points(tmp_path, "TLY") == []


def test_get_fund_yield_summary_payload_uses_tefasfon_first(monkeypatch) -> None:
    class FakeTefasFonClient:
        def fetch_yield_summary(self, fund_code, **kwargs):
            return {
                "fund_code": fund_code,
                "source": "tefasfon_funds",
                "source_url": "https://pypi.org/project/tefasfon/",
                "periods": {
                    "1m": {
                        "prev_close_date": "2026-04-01",
                        "prev_close": 10.0,
                        "high": 12.0,
                        "low": 9.5,
                    }
                },
                "raw": {},
            }

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.get_fund_yield_summary_payload("TLY")

    assert payload["status"] == "ok"
    assert payload["source"] == "tefasfon_funds"
    assert payload["periods"]["1m"]["prev_close"] == 10.0
    assert payload["source_metadata"]["summary_source_used"] == "tefasfon_funds"
    assert payload["source_metadata"]["fallback_used"] is False


def test_tefasfon_yield_summary_fills_weekly_period_from_history() -> None:
    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            pass

        def fetch_returns(self, *, fund_codes=None, start_date=None, end_date=None):
            return [
                {
                    "fonKodu": "TLY",
                    "getiri1a": 25.0,
                    "source": "tefasfon_returns",
                }
            ]

        def fetch_history(self, fund_code, start_date, end_date):
            return [
                {"fund_code": fund_code, "date": "2026-05-04", "price": 100.0, "source": "tefasfon_funds"},
                {"fund_code": fund_code, "date": "2026-05-11", "price": 110.0, "source": "tefasfon_funds"},
            ]

    summary = FakeTefasFonClient().fetch_yield_summary(
        "TLY",
        as_of=date(2026, 5, 11),
        latest_price=110.0,
        latest_date="2026-05-11",
    )

    assert summary["source"] == "tefasfon_returns"
    assert summary["periods"]["1m"]["prev_close"] == pytest.approx(88.0)
    assert summary["periods"]["1w"]["prev_close"] == pytest.approx(100.0)


def test_tefasfon_portfolio_filters_requested_fund_code() -> None:
    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            self.fund_types = ("SEC",)

        def _fetch_portfolio_direct(self, fund_code, start_date, end_date, *, fund_type):
            raise fund_service.TefasUpstreamError("direct path disabled in test")

        def _call_dataframe(self, function_name, *, context, **kwargs):
            assert function_name == "get_portfolio"
            return [
                {"fonKodu": "AAA", "tarih": "2026-04-30", "hs": 10.0},
                {"fonKodu": "TLY", "tarih": "2026-04-30", "hs": 58.0},
            ]

    rows = FakeTefasFonClient().fetch_portfolio(
        fund_code="TLY",
        start_date=date(2026, 4, 30),
        end_date=date(2026, 4, 30),
    )

    assert [row["fonKodu"] for row in rows] == ["TLY"]


def test_get_fund_allocations_payload_hides_legacy_cache(tmp_path) -> None:
    cache_dir = tmp_path / "funds_cache" / "allocations"
    cache_dir.mkdir(parents=True)
    (cache_dir / "TLY.json").write_text(
        json.dumps({"fund_code": "TLY", "source": "te" + "fas", "allocations": [{"allocation_type": "hs"}]}),
        encoding="utf-8",
    )

    payload = fund_service.get_fund_allocations_payload(tmp_path, "TLY")

    assert payload["status"] == "unavailable"
    assert payload["source"] == "tefasfon_portfolio"
    assert payload["allocations"] == []


def test_refresh_fund_allocations_uses_tefasfon_portfolio(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_latest_portfolio(self, fund_code, *, as_of, lookback_days):
            return [
                {
                    "fonKodu": fund_code,
                    "tarih": "2026-04-29",
                    "hs": 55.5,
                    "tr": 4.5,
                    "source": "tefasfon_portfolio",
                }
            ], []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_fund_allocations(tmp_path, "TLY", as_of=date(2026, 4, 29))

    assert payload["status"] == "ok"
    assert payload["source"] == "tefasfon_portfolio"
    assert [item["allocation_type"] for item in payload["allocations"]] == ["hs", "tr"]
    assert payload["allocations"][0]["label"] == "Hisse Senedi"
    assert payload["source_metadata"]["fallback_used"] is False


def test_refresh_fund_allocations_history_groups_last_days(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_portfolio(self, *, fund_code, start_date, end_date):
            assert fund_code == "TLY"
            assert start_date == date(2026, 4, 1)
            assert end_date == date(2026, 4, 30)
            return [
                {"fonKodu": fund_code, "tarih": "2026-04-01", "hs": 50.0, "yyf": 20.0, "source": "tefasfon_portfolio"},
                {"fonKodu": fund_code, "tarih": "2026-04-30", "hs": 58.0, "yyf": 17.0, "source": "tefasfon_portfolio"},
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_fund_allocations_history(
        tmp_path,
        "TLY",
        lookback_days=30,
        as_of=date(2026, 4, 30),
    )

    assert payload["status"] == "ok"
    assert payload["lookback_days"] == 30
    assert [row["date"] for row in payload["history"]] == ["2026-04-01", "2026-04-30"]
    assert payload["history"][0]["allocations"][0]["allocation_type"] == "hs"
    assert payload["history"][1]["allocations"][0]["weight"] == 58.0


def test_refresh_fund_allocations_history_falls_back_to_daily_snapshots(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeTefasFonClient:
        def fetch_portfolio(self, *, fund_code, start_date, end_date):
            calls.append((start_date, end_date))
            if start_date != end_date:
                return []
            if start_date == date(2026, 4, 27):
                return [{"fonKodu": fund_code, "tarih": "2026-04-27", "hs": 50.0, "source": "tefasfon_portfolio"}]
            if start_date == date(2026, 4, 30):
                return [{"fonKodu": fund_code, "tarih": "2026-04-30", "hs": 58.0, "source": "tefasfon_portfolio"}]
            return []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.refresh_fund_allocations_history(
        tmp_path,
        "TLY",
        lookback_days=4,
        as_of=date(2026, 4, 30),
    )

    assert calls[0] == (date(2026, 4, 27), date(2026, 4, 30))
    assert (date(2026, 4, 27), date(2026, 4, 27)) in calls
    assert [row["date"] for row in payload["history"]] == ["2026-04-27", "2026-04-30"]
    assert "daily snapshots" in " ".join(payload["source_metadata"]["warnings"])


def test_get_fund_allocations_history_payload_uses_cache(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeTefasFonClient:
        def fetch_portfolio(self, *, fund_code, start_date, end_date):
            calls.append((fund_code, start_date, end_date))
            return [
                {"fonKodu": fund_code, "tarih": end_date.isoformat(), "hs": 55.0, "source": "tefasfon_portfolio"},
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    first = fund_service.get_fund_allocations_history_payload(tmp_path, "TLY", lookback_days=30)
    second = fund_service.get_fund_allocations_history_payload(tmp_path, "TLY", lookback_days=30)

    assert first["status"] == "ok"
    assert second["source_metadata"]["cache_hit"] is True
    assert len(calls) == 1


def test_get_fund_allocations_history_payload_caches_empty_result(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeTefasFonClient:
        def fetch_portfolio(self, *, fund_code, start_date, end_date):
            calls.append((fund_code, start_date, end_date))
            return []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    first = fund_service.get_fund_allocations_history_payload(tmp_path, "PHE", lookback_days=30)
    first_call_count = len(calls)
    second = fund_service.get_fund_allocations_history_payload(tmp_path, "PHE", lookback_days=30)

    assert first["status"] == "empty"
    assert second["source_metadata"]["cache_hit"] is True
    assert first_call_count > 1
    assert len(calls) == first_call_count
