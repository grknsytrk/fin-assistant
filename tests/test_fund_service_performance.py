from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from app import fund_service
from tests.test_fund_service import _stub_direct_tefas_empty


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


def test_fund_period_stats_summarizes_current_month_returns() -> None:
    stats = fund_service._fund_period_stats(
        [
            {"date": "2026-04-30", "price": 100.0, "daily_return": None},
            {"date": "2026-05-04", "price": 101.0, "daily_return": 1.0},
            {"date": "2026-05-05", "price": 99.0, "daily_return": -2.0},
            {"date": "2026-05-06", "price": 99.0, "daily_return": 0.0},
            {"date": "2026-05-07", "price": 102.0, "daily_return": 3.0},
        ],
        as_of=date(2026, 5, 31),
    )

    current_month = next(row for row in stats["periods"] if row["key"] == "current_month")
    assert current_month["label"] == "2026-05"
    assert current_month["trading_days"] == 4
    assert current_month["return_days"] == 4
    assert current_month["positive_days"] == 2
    assert current_month["negative_days"] == 1
    assert current_month["flat_days"] == 1
    assert current_month["average_daily_return"] == pytest.approx(0.5)
    assert current_month["cumulative_return"] == pytest.approx(2.0)
    assert current_month["basis"] == "previous_close"


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


def test_fund_history_monthly_anchor_is_resolution_not_partial_gap() -> None:
    points = [
        {"fund_code": "THF", "date": point_date, "price": float(index + 1)}
        for index, point_date in enumerate(
            [
                "2025-11-28",
                "2025-12-31",
                "2026-01-30",
                "2026-02-27",
                "2026-03-31",
                "2026-04-30",
                "2026-05-29",
                "2026-06-30",
                "2026-07-31",
            ]
        )
    ]

    assert fund_service._history_resolution(points) == "monthly_anchor"
    assert fund_service._history_internal_gap_warnings(points) == []


def test_fund_history_mixed_resolution_keeps_real_daily_gap_warning() -> None:
    points = [
        {"fund_code": "IIH", "date": point_date, "price": float(index + 1)}
        for index, point_date in enumerate(
            [
                "2026-05-29",
                "2026-06-30",
                "2026-07-31",
                "2026-08-03",
                "2026-08-10",
                "2026-08-11",
            ]
        )
    ]

    assert fund_service._history_resolution(points) == "mixed"
    warnings = fund_service._history_internal_gap_warnings(points)
    assert len(warnings) == 1
    assert "previous=2026-08-03, next=2026-08-10" in warnings[0]


def test_fund_performance_metadata_exposes_monthly_anchor_resolution(tmp_path) -> None:
    points = [
        {"fund_code": "THF", "date": "2026-06-30", "price": 10.0},
        {"fund_code": "THF", "date": "2026-07-31", "price": 11.0},
    ]

    payload = fund_service._fund_performance_payload_from_points(
        tmp_path,
        "THF",
        points,
        start_date=date(2026, 6, 30),
        end_date=date(2026, 7, 31),
    )

    metadata = payload["source_metadata"]
    assert metadata["resolution"] == "monthly_anchor"
    assert metadata["coverage_state"] == "complete"
    assert metadata["internal_gap_count"] == 0


def test_fund_performance_metadata_marks_material_tail_gap_incomplete(tmp_path) -> None:
    points = [
        {"fund_code": "AFT", "date": "2026-05-25", "price": 10.0},
        {"fund_code": "AFT", "date": "2026-05-26", "price": 11.0},
    ]

    payload = fund_service._fund_performance_payload_from_points(
        tmp_path,
        "AFT",
        points,
        start_date=date(2026, 3, 10),
        end_date=date(2026, 9, 10),
    )

    metadata = payload["source_metadata"]
    assert metadata["coverage_state"] == "range_incomplete"
    assert metadata["coverage_gap_business_days"] > 3
    assert metadata["available_end_date"] == "2026-05-26"


def test_fund_performance_metadata_keeps_partial_fintables_head_incomplete(tmp_path) -> None:
    points = []
    current = date(2026, 4, 1)
    price = 1.0
    while current <= date(2026, 9, 10):
        if current.weekday() < 5:
            points.append(
                {
                    "fund_code": "BYZ",
                    "date": current.isoformat(),
                    "price": price,
                    "source": "fintables_udf_history",
                }
            )
            price += 0.01
        current += timedelta(days=1)

    payload = fund_service._fund_performance_payload_from_points(
        tmp_path,
        "BYZ",
        points,
        start_date=date(2026, 3, 10),
        end_date=date(2026, 9, 10),
    )

    metadata = payload["source_metadata"]
    assert metadata["coverage_state"] == "range_incomplete"
    assert metadata["coverage_gap_business_days"] == 0
    assert metadata["internal_gap_count"] == 0


@pytest.mark.parametrize("fund_code", ["TLY", "TI2", "BOH"])
def test_refresh_fund_performance_prefers_fintables_for_all_funds(monkeypatch, tmp_path, fund_code) -> None:
    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            raise AssertionError("TEFAS should not be called when Fintables has a valid history")

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            return [
                {"fund_code": fund_code, "date": "2026-04-28", "price": 3.1, "source": "fintables_udf_history"},
                {"fund_code": fund_code, "date": "2026-04-29", "price": 3.2, "source": "fintables_udf_history"},
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "FintablesClient", lambda: FakeFintablesClient())

    payload = fund_service.refresh_fund_performance(
        tmp_path,
        fund_code,
        start_date=date(2026, 4, 28),
        end_date=date(2026, 4, 29),
    )

    assert payload["status"] == "ok"
    assert payload["source_metadata"]["history_source_used"] == "fintables_udf_history"
    assert payload["source_metadata"]["history_source_policy"] == "fintables_daily_tefas_fallback"
    assert payload["source_metadata"]["primary_source"] == "fintables"
    assert payload["source_metadata"]["fallback_used"] is False
    assert payload["source_metadata"]["fallback_reason"] is None
    assert payload["source_metadata"]["tefasfon_adapter_version"] == fund_service._tefasfon_adapter_version()


def test_refresh_fund_performance_falls_back_to_tefas_when_fintables_fails(monkeypatch, tmp_path) -> None:
    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            return [
                {"fund_code": fund_code, "date": "2026-04-28", "price": 3.1, "source": "tefasfon_funds"},
                {"fund_code": fund_code, "date": "2026-04-29", "price": 3.2, "source": "tefasfon_funds"},
            ]

    class FakeFintablesClient:
        def fetch_udf_history(self, fund_code, start_date, end_date):
            raise fund_service.FintablesUpstreamError("blocked")

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
    assert payload["source_metadata"]["fallback_used"] is True
    assert payload["source_metadata"]["fallback_reason"] == "fintables_udf_history_unavailable_tefas_fallback"
    assert "blocked" in " ".join(payload["source_metadata"]["warnings"])


def test_tefasfon_fetch_history_uses_anchor_strategy_for_long_ranges(monkeypatch) -> None:
    client = fund_service.TefasFonClient(fund_types=["SEC"])
    fetch_ranges = []
    snapshot_calls = []

    def fake_fetch_funds(*, start_date, end_date, fund_codes=None):
        fetch_ranges.append((start_date, end_date, tuple(fund_codes or [])))
        return [
            {"fonKodu": "TLY", "tarih": start_date.isoformat(), "fiyat": 5.0, "source": "tefasfon_funds"},
            {"fonKodu": "TLY", "tarih": end_date.isoformat(), "fiyat": 6.0, "source": "tefasfon_funds"},
            {"fonKodu": "ABC", "tarih": end_date.isoformat(), "fiyat": 7.0, "source": "tefasfon_funds"},
        ]

    def fake_snapshot(as_of):
        snapshot_calls.append(as_of)
        return []

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            return []

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())
    monkeypatch.setattr(client, "fetch_funds", fake_fetch_funds)
    monkeypatch.setattr(client, "fetch_daily_funds_snapshot", fake_snapshot)

    rows = client.fetch_history("TLY", date(2026, 1, 1), date(2026, 5, 18))

    assert fetch_ranges == [
        (date(2026, 1, 1), date(2026, 3, 31), ("TLY",)),
        (date(2026, 4, 1), date(2026, 5, 18), ("TLY",)),
    ]
    assert snapshot_calls == []
    for c_start, c_end, codes in fetch_ranges:
        assert (c_end - c_start).days < 90
        assert (c_end - c_start).days + 1 <= 90
    assert [row["tarih"] for row in rows] == [
        "2026-01-01",
        "2026-03-31",
        "2026-04-01",
        "2026-05-18",
    ]
    assert all(row["fonKodu"] == "TLY" for row in rows)


def test_tefasfon_fetch_history_chunks_wide_ranges_to_90_days(monkeypatch) -> None:
    client = fund_service.TefasFonClient(fund_types=["SEC"])
    direct_calls = []

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            direct_calls.append((tuple(fund_codes), start_date, end_date))
            return [
                {
                    "fonKodu": fund_codes[0],
                    "tarih": start_date.isoformat(),
                    "fiyat": 10.0,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())

    # Primary direct path must issue a single full-range request without 90-day double-chunking.
    rows = client.fetch_history("TLY", date(2025, 1, 1), date(2025, 12, 31))
    assert direct_calls == [(("TLY",), date(2025, 1, 1), date(2025, 12, 31))]
    assert len(rows) == 1

    # Fallback path (when direct returns empty) must still chunk wide ranges into <= 90 days.
    class EmptyTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            return []

    fallback_calls = []

    def fake_fetch_funds(*, start_date, end_date, fund_codes=None):
        fallback_calls.append((tuple(fund_codes or ()), start_date, end_date))
        return [
            {
                "fonKodu": fund_codes[0],
                "tarih": start_date.isoformat(),
                "fiyat": 10.0,
                "source": "tefasfon_funds",
            }
        ]

    monkeypatch.setattr(fund_service, "TefasClient", lambda: EmptyTefasClient())
    monkeypatch.setattr(client, "fetch_funds", fake_fetch_funds)

    fallback_rows = client.fetch_history("TLY", date(2025, 1, 1), date(2025, 12, 31))

    assert len(fallback_calls) == 5
    for codes, c_start, c_end in fallback_calls:
        assert codes == ("TLY",)
        assert (c_end - c_start).days + 1 <= 90
    assert fallback_calls[0][1] == date(2025, 1, 1)
    assert fallback_calls[0][2] == date(2025, 3, 31)
    assert fallback_calls[1][1] == date(2025, 4, 1)
    assert fallback_calls[1][2] == date(2025, 6, 29)
    assert fallback_calls[2][1] == date(2025, 6, 30)
    assert fallback_calls[2][2] == date(2025, 9, 27)
    assert fallback_calls[3][1] == date(2025, 9, 28)
    assert fallback_calls[3][2] == date(2025, 12, 26)
    assert fallback_calls[4][1] == date(2025, 12, 27)
    assert fallback_calls[4][2] == date(2025, 12, 31)
    assert len(fallback_rows) == 5


def test_tefasfon_fetch_history_recovers_from_single_chunk_failure(monkeypatch) -> None:
    # Part 1: Direct calendar-month fetch keeps surviving months when one month fails.
    direct = fund_service.TefasClient()

    def fake_range(*, fund_code, start_date, end_date):
        if start_date.month == 2:
            raise fund_service.TefasUpstreamError("Temporary Feb error")
        return [
            {
                "fund_code": fund_code,
                "date": start_date.isoformat(),
                "price": float(start_date.month),
                "source": "tefasfon_funds",
            }
        ]

    monkeypatch.setattr(direct, "fetch_fund_range", fake_range)
    direct_rows = direct.fetch_fund_history(
        fund_codes=["TLY"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
    )
    assert [row["date"] for row in direct_rows] == ["2026-01-01", "2026-03-01"]

    # Part 2: Fallback chunk loop also survives a single 90-day chunk failure.
    client = fund_service.TefasFonClient(fund_types=["SEC"])
    fallback_count = 0

    class EmptyTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            return []

    def fake_fallback_funds(*, start_date, end_date, fund_codes=None):
        nonlocal fallback_count
        fallback_count += 1
        if fallback_count == 2:
            raise fund_service.TefasUpstreamError("Temporary fallback chunk error")
        return [
            {
                "fonKodu": fund_codes[0],
                "tarih": start_date.isoformat(),
                "fiyat": 1.0,
                "source": "tefasfon_funds",
            }
        ]

    monkeypatch.setattr(fund_service, "TefasClient", lambda: EmptyTefasClient())
    monkeypatch.setattr(client, "fetch_funds", fake_fallback_funds)

    rows = client.fetch_history("TLY", date(2026, 1, 1), date(2026, 6, 29))
    assert len(rows) == 1
    assert rows[0]["tarih"] == "2026-01-01"


def test_tefasfon_fetch_history_aborts_immediately_on_rate_limit(monkeypatch) -> None:
    client = fund_service.TefasFonClient(fund_types=["SEC"])
    attempts = 0

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            nonlocal attempts
            attempts += 1
            raise fund_service.TefasRateLimitError("TEFAS HTTP 429 Too Many Requests")

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())

    with pytest.raises(fund_service.TefasRateLimitError):
        client.fetch_history("TLY", date(2025, 1, 1), date(2025, 12, 31))

    # Must abort after chunk 1 without hammering remaining 4 chunks
    assert attempts == 1


def test_tefasfon_fetch_history_raises_when_all_chunks_fail(monkeypatch) -> None:
    client = fund_service.TefasFonClient(fund_types=["SEC"])

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            raise fund_service.TefasUpstreamError("Direct upstream down")

    def fake_fallback(*, start_date, end_date, fund_codes=None):
        raise fund_service.TefasUpstreamError("Fallback upstream down")

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())
    monkeypatch.setattr(client, "fetch_funds", fake_fallback)

    with pytest.raises(fund_service.TefasUpstreamError) as exc_info:
        client.fetch_history("TLY", date(2026, 1, 1), date(2026, 3, 31))

    assert "Fallback upstream down" in str(exc_info.value)


def test_tefasfon_fetch_history_preserves_unmapped_fields(monkeypatch) -> None:
    client = fund_service.TefasFonClient(fund_types=["SEC"])

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            return [
                {
                    "fonKodu": "TLY",
                    "tarih": "2026-01-15",
                    "fiyat": 12.34,
                    "tedPaySayisi": 1000000,
                    "kisiSayisi": 500,
                    "portfoyBuyukluk": 50000000.0,
                    "custom_future_field": "preserved_value",
                }
            ]

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())

    rows = client.fetch_history("TLY", date(2026, 1, 1), date(2026, 1, 31))
    assert len(rows) == 1
    assert rows[0]["fonKodu"] == "TLY"
    assert rows[0]["fiyat"] == 12.34
    assert rows[0]["tedPaySayisi"] == 1000000
    assert rows[0]["kisiSayisi"] == 500
    assert rows[0]["portfoyBuyukluk"] == 50000000.0
    assert rows[0]["custom_future_field"] == "preserved_value"


def test_generate_date_chunks_properties() -> None:
    # Inverted range
    assert fund_service._generate_date_chunks(date(2026, 2, 1), date(2026, 1, 1)) == []

    # Single day
    single = fund_service._generate_date_chunks(date(2026, 1, 1), date(2026, 1, 1))
    assert single == [(date(2026, 1, 1), date(2026, 1, 1))]
    assert (single[0][1] - single[0][0]).days == 0

    # Exactly 90 days: 2026-01-01 to 2026-03-31
    exact_90 = fund_service._generate_date_chunks(date(2026, 1, 1), date(2026, 3, 31))
    assert len(exact_90) == 1
    assert (exact_90[0][1] - exact_90[0][0]).days == 89  # 90 calendar days

    # 91 days: 2026-01-01 to 2026-04-01
    split_91 = fund_service._generate_date_chunks(date(2026, 1, 1), date(2026, 4, 1))
    assert len(split_91) == 2
    assert split_91[0] == (date(2026, 1, 1), date(2026, 3, 31))
    assert split_91[1] == (date(2026, 4, 1), date(2026, 4, 1))


def test_fast_long_history_prefers_complete_daily_fintables_history(monkeypatch) -> None:
    fintables_calls = []
    direct_calls = []

    def fake_fintables_history(fund_code, start_date, end_date):
        fintables_calls.append((fund_code, start_date, end_date))
        rows = []
        current = start_date
        price = 1.0
        while current <= end_date:
            if current.weekday() < 5:
                rows.append(
                    {
                        "fund_code": fund_code,
                        "date": current.isoformat(),
                        "price": price,
                        "source": "fintables_udf_history",
                    }
                )
                price += 0.01
            current += timedelta(days=1)
        return rows

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            direct_calls.append((start_date, end_date))
            raise AssertionError("TEFAS should not be used for a complete Fintables daily range")

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())
    monkeypatch.setattr(fund_service, "fetch_fintables_udf_history", fake_fintables_history)
    monkeypatch.setattr(fund_service, "_fetch_recent_detail_rows", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(fund_service, "_fetch_fund_overview_metric_rows", lambda *_args, **_kwargs: ([], [], []))

    points, warnings, fallback_used, fallback_reason = fund_service._fetch_fast_long_fund_history(
        None,
        "TI2",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 5, 18),
        client=object(),
    )

    assert fintables_calls == [("TI2", date(2026, 1, 1), date(2026, 5, 18))]
    assert direct_calls == []
    assert len(points) == 98
    assert points[0]["date"] == "2026-01-01"
    assert points[-1]["date"] == "2026-05-18"
    assert points[0]["source"] == "fintables_udf_history"
    assert warnings == []
    assert fallback_used is False
    assert fallback_reason is None


def test_fast_long_history_adds_confirmed_tefas_tail(monkeypatch) -> None:
    tefas_calls = []

    def partial_fintables_history(fund_code, start_date, end_date):
        rows = []
        current = start_date
        price = 1.0
        while current <= date(2026, 9, 11):
            if current.weekday() < 5:
                rows.append(
                    {
                        "fund_code": fund_code,
                        "date": current.isoformat(),
                        "price": price,
                        "source": "fintables_udf_history",
                    }
                )
                price += 0.01
            current += timedelta(days=1)
        return rows

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            tefas_calls.append((start_date, end_date))
            return [
                {
                    "fund_code": fund_codes[0],
                    "date": "2026-09-14",
                    "price": 9.0,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())
    monkeypatch.setattr(fund_service, "fetch_fintables_udf_history", partial_fintables_history)
    monkeypatch.setattr(fund_service, "_fetch_recent_detail_rows", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(fund_service, "_fetch_fund_overview_metric_rows", lambda *_args, **_kwargs: ([], [], []))

    points, warnings, fallback_used, fallback_reason = fund_service._fetch_fast_long_fund_history(
        None,
        "TLY",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 9, 14),
        client=object(),
    )

    assert tefas_calls == [(date(2026, 9, 12), date(2026, 9, 14))]
    assert points[-1]["date"] == "2026-09-14"
    assert points[-1]["source"] == "tefasfon_funds"
    assert any(point["source"] == "fintables_udf_history" for point in points)
    assert warnings == []
    assert fallback_used is True
    assert fallback_reason == "fintables_udf_history_confirmed_tail_tefas_fallback"


def test_confirmed_tefas_tail_preserves_publication_delay(monkeypatch) -> None:
    tefas_calls = []

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            tefas_calls.append((start_date, end_date))
            return []

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())

    tail, warnings = fund_service._fetch_confirmed_tefas_tail(
        "TLY",
        [
            {
                "fund_code": "TLY",
                "date": "2026-09-11",
                "price": 9805.33,
                "source": "fintables_udf_history",
            }
        ],
        end_date=date(2026, 9, 14),
    )

    assert tefas_calls == [(date(2026, 9, 12), date(2026, 9, 14))]
    assert tail == []
    assert warnings == []


def test_fast_long_history_uses_tefas_when_fintables_range_is_incomplete(monkeypatch) -> None:
    calls = []

    def partial_fintables_history(fund_code, start_date, end_date):
        calls.append("fintables")
        return [
            {
                "fund_code": fund_code,
                "date": end_date.isoformat(),
                "price": 2.0,
                "source": "fintables_udf_history",
            }
        ]

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            calls.append("tefas")
            rows = []
            current = start_date
            price = 1.0
            while current <= end_date:
                if current.weekday() < 5:
                    rows.append(
                        {
                            "fund_code": fund_codes[0],
                            "date": current.isoformat(),
                            "price": price,
                            "source": "tefasfon_funds",
                        }
                    )
                    price += 0.01
                current += timedelta(days=1)
            return rows

    monkeypatch.setattr(fund_service, "fetch_fintables_udf_history", partial_fintables_history)
    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())
    monkeypatch.setattr(fund_service, "_fetch_recent_detail_rows", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(fund_service, "_fetch_fund_overview_metric_rows", lambda *_args, **_kwargs: ([], [], []))

    points, warnings, fallback_used, fallback_reason = fund_service._fetch_fast_long_fund_history(
        None,
        "TI2",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 5, 18),
        client=object(),
    )

    assert calls == ["fintables", "tefas"]
    assert len(points) == 98
    assert points[0]["date"] == "2026-01-01"
    assert points[-1]["date"] == "2026-05-18"
    assert points[0]["source"] == "tefasfon_funds"
    assert points[-1]["source"] == "tefasfon_funds"
    assert warnings == []
    assert fallback_used is True
    assert fallback_reason == "fintables_udf_history_partial_tefas_fallback"


def test_get_fund_performance_payload_uses_tefasfon_only_when_fintables_unavailable(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        fund_service,
        "fetch_fintables_udf_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(fund_service.FintablesUpstreamError("unavailable")),
    )

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
    assert payload["source_metadata"]["history_source_policy"] == "fintables_daily_tefas_fallback"
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
    monkeypatch.setattr(
        fund_service,
        "fetch_fintables_udf_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(fund_service.FintablesUpstreamError("unavailable")),
    )

    payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")
    cached_payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")

    assert calls
    assert calls[0][1] == date(2020, 1, 1)
    assert [point["date"] for point in payload["points"]] == ["2020-01-02", today.isoformat()]
    assert payload["source_metadata"]["requested_start_date"] == "2020-01-01"
    assert payload["source_metadata"]["full_history_requested"] is True
    assert cached_payload["source_metadata"]["cache_hit"] is True
    assert len(calls) == 1


def test_get_fund_performance_payload_refreshes_only_missing_recent_tail(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 5, 21)

    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {
                "fund_code": "TLY",
                "date": "2026-05-20",
                "price": 5516.308655,
                "daily_return": 0.50,
                "source": "tefasfon_funds",
            }
        ],
        source="tefasfon_funds",
    )
    history_path = fund_service._history_path(tmp_path, "TLY")
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "source_metadata": {
                    "requested_start_date": "2026-01-01",
                    "date_max": "2026-05-20",
                    "parse_status": "ok",
                },
                "as_of": "2026-05-20",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            calls.append((fund_code, start_date, end_date))
            return [
                {
                    "fund_code": fund_code,
                    "date": "2026-05-21",
                    "price": 5518.513245,
                    "daily_return": 0.04,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "date", FixedDate)
    monkeypatch.setattr(fund_service, "FUNDS_FULL_HISTORY_START_DATE", "2026-01-01")
    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")
    dates = [point["date"] for point in payload["points"]]
    cached_metadata = json.loads(history_path.read_text(encoding="utf-8"))["source_metadata"]

    assert calls == [("TLY", date(2026, 5, 21), date(2026, 5, 21))]
    assert dates == ["2026-05-20", "2026-05-21"]
    assert payload["points"][-1]["daily_return"] == 0.04
    assert payload["source_metadata"]["backfill_used"] is True
    assert payload["source_metadata"]["full_history_requested"] is True
    assert cached_metadata["requested_start_date"] == "2026-01-01"
    assert cached_metadata["date_max"] == "2026-05-20"

    second_payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")

    assert len(calls) == 1
    assert second_payload["source_metadata"]["cache_hit"] is True


def test_get_fund_performance_payload_fast_bootstraps_long_partial_range(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 5, 21)

    history_calls = []
    snapshot_calls = []

    def fake_fintables_history(fund_code, start_date, end_date):
        rows = []
        current = start_date
        price = 1.0
        while current <= end_date:
            if current.weekday() < 5:
                rows.append({
                    "fund_code": fund_code,
                    "date": current.isoformat(),
                    "price": price,
                    "source": "fintables_udf_history",
                })
                price += 0.01
            current += timedelta(days=1)
        return rows

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            history_calls.append((fund_code, start_date, end_date))
            rows = []
            current = start_date
            price = 2.0
            while current <= end_date:
                if current.weekday() < 5:
                    rows.append(
                        {
                            "fonKodu": fund_code,
                            "tarih": current.isoformat(),
                            "fiyat": price,
                            "portfoyBuyukluk": 1_000_000_000 + len(rows),
                            "kisiSayisi": 10_000 + len(rows),
                            "tedPaySayisi": 500_000 + len(rows),
                            "source": "tefasfon_funds",
                        }
                    )
                    price += 0.01
                current += timedelta(days=1)
            return rows

        def fetch_daily_funds_snapshot(self, as_of):
            snapshot_calls.append(as_of.isoformat())
            if as_of.isoformat() == "2026-04-30":
                return [
                    {
                        "fonKodu": "KHA",
                        "tarih": "2026-04-30",
                        "fiyat": 2.0,
                        "portfoyBuyukluk": 1_000_000_000,
                        "kisiSayisi": 10_000,
                        "source": "tefasfon_funds",
                    },
                    {
                        "fonKodu": "DGR",
                        "tarih": "2026-04-30",
                        "fiyat": 1.0,
                        "portfoyBuyukluk": 2_000_000_000,
                        "kisiSayisi": 20_000,
                        "source": "tefasfon_funds",
                    },
                ]
            if as_of.isoformat() == "2026-05-21":
                return [
                    {
                        "fonKodu": "KHA",
                        "tarih": "2026-05-21",
                        "fiyat": 2.2,
                        "portfoyBuyukluk": 1_250_000_000,
                        "kisiSayisi": 11_000,
                        "source": "tefasfon_funds",
                    }
                ]
            return []

    monkeypatch.setattr(fund_service, "date", FixedDate)
    monkeypatch.setattr(fund_service, "fetch_fintables_udf_history", fake_fintables_history)
    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    _stub_direct_tefas_empty(monkeypatch)

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "KHA",
        start_date=date(2025, 10, 21),
        end_date=date(2026, 5, 21),
    )
    by_date = {point["date"]: point for point in payload["points"]}

    assert history_calls == [("KHA", date(2026, 4, 17), date(2026, 5, 21))]
    assert "2026-04-30" in snapshot_calls
    assert "2026-05-21" in snapshot_calls
    assert by_date["2026-04-30"]["aum"] == 1_000_000_000
    assert by_date["2026-05-21"]["investor_count"] == 11_000
    assert by_date["2026-05-20"]["aum"] is not None
    assert fund_service.read_fund_price_points(tmp_path, "DGR")[0]["investor_count"] == 20_000
    assert payload["source_metadata"]["full_history_requested"] is False
    assert payload["source_metadata"]["cached_fallback_points_present"] is True
    assert payload["source_metadata"]["recent_detail_backfill"]["attempted"] is False

    snapshot_calls.clear()
    second_payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "KHA",
        start_date=date(2025, 10, 21),
        end_date=date(2026, 5, 21),
    )

    assert snapshot_calls == []
    assert len(history_calls) == 1
    assert second_payload["source_metadata"]["full_history_requested"] is False


def test_fast_long_history_does_not_treat_monthly_metrics_as_daily_coverage(monkeypatch) -> None:
    direct_calls = []

    def unavailable_fintables_history(fund_code, start_date, end_date):
        raise fund_service.FintablesUpstreamError("unavailable")

    class FakeTefasClient:
        def fetch_fund_history(self, *, fund_codes, start_date, end_date):
            direct_calls.append((start_date, end_date))
            return [
                {
                    "fonKodu": fund_codes[0],
                    "tarih": end_date.isoformat(),
                    "fiyat": 2.0,
                    "source": "tefasfon_funds",
                }
            ]

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            return [
                {
                    "fonKodu": fund_code,
                    "tarih": end_date.isoformat(),
                    "fiyat": 1.0,
                    "source": "tefasfon_funds",
                }
            ]

        def fetch_daily_funds_snapshot(self, as_of):
            return [
                {
                    "fonKodu": "THF",
                    "tarih": as_of.isoformat(),
                    "fiyat": 1.0,
                    "portfoyBuyukluk": 1_000_000,
                    "kisiSayisi": 100,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "fetch_fintables_udf_history", unavailable_fintables_history)
    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeTefasClient())

    points, warnings, _fallback_used, _fallback_reason = fund_service._fetch_fast_long_fund_history(
        None,
        "THF",
        start_date=date(2025, 11, 28),
        end_date=date(2026, 8, 21),
        client=FakeTefasFonClient(),
    )

    assert points
    assert warnings[0] == "fintables_udf_history fast bootstrap failed: unavailable"
    assert (date(2025, 11, 28), date(2026, 8, 21)) in direct_calls


def test_get_fund_performance_payload_backfills_missing_overview_metrics(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 5, 18)

    price_only_points = [
        {"fund_code": "TLY", "date": "2025-12-31", "price": 2930.287064, "source": "fintables_udf_history"},
        {"fund_code": "TLY", "date": "2026-01-30", "price": 3393.579372, "source": "fintables_udf_history"},
        {"fund_code": "TLY", "date": "2026-02-27", "price": 3899.452408, "source": "fintables_udf_history"},
        {"fund_code": "TLY", "date": "2026-03-31", "price": 4638.074312, "source": "fintables_udf_history"},
        {"fund_code": "TLY", "date": "2026-04-30", "price": 5223.812438, "source": "fintables_udf_history"},
        {"fund_code": "TLY", "date": "2026-05-18", "price": 5488.8942, "source": "fintables_udf_history"},
    ]
    fund_service.upsert_fund_price_points(tmp_path, price_only_points, source="fintables_udf_history")
    history_path = fund_service._history_path(tmp_path, "TLY")
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "source_metadata": {
                    "requested_start_date": "2025-11-01",
                    "date_max": "2026-05-18",
                    "parse_status": "ok",
                },
                "as_of": "2026-05-18",
            }
        ),
        encoding="utf-8",
    )

    snapshot_by_date = {
        "2025-11-28": ("2025-11-28", 2800.0, 35_000_000_000.0, 40_000, 12_500_000),
        "2025-12-31": ("2025-12-31", 2930.287064, 39_896_268_611.8, 44_740, 13_615_140),
        "2026-01-30": ("2026-01-30", 3393.579372, 51_076_423_774.42, 47_788, 15_050_900),
        "2026-02-27": ("2026-02-27", 3899.452408, 60_607_168_862.97, 55_293, 15_542_482),
        "2026-03-31": ("2026-03-31", 4638.074312, 93_949_227_501.05, 70_964, 20_256_085),
        "2026-04-30": ("2026-04-30", 5223.812438, 124_111_347_800.34, 82_223, 23_758_768),
        "2026-05-18": ("2026-05-18", 5488.8942, 134_932_745_112.65, 85_009, 24_582_865),
    }
    calls = []

    class FakeTefasFonClient:
        def fetch_daily_funds_snapshot(self, as_of):
            calls.append(as_of.isoformat())
            row = snapshot_by_date.get(as_of.isoformat())
            if not row:
                return []
            row_date, price, aum, investors, shares = row
            return [
                {
                    "fonKodu": "TLY",
                    "tarih": row_date,
                    "fiyat": price,
                    "portfoyBuyukluk": aum,
                    "kisiSayisi": investors,
                    "tedPaySayisi": shares,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "date", FixedDate)
    monkeypatch.setattr(fund_service, "FUNDS_FULL_HISTORY_START_DATE", "2025-11-01")
    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")
    by_date = {point["date"]: point for point in payload["points"]}

    assert calls == [
        "2025-12-31",
        "2026-01-30",
        "2026-02-27",
        "2026-03-31",
        "2026-04-30",
        "2026-05-18",
    ]
    assert by_date["2025-12-31"]["aum"] == 39_896_268_611.8
    assert by_date["2026-01-30"]["investor_count"] == 47_788
    assert by_date["2026-02-27"]["aum"] == 60_607_168_862.97
    assert by_date["2026-03-31"]["investor_count"] == 70_964
    assert by_date["2026-04-30"]["source"] == "fintables_udf_history"
    assert payload["source_metadata"]["overview_metric_backfill"]["attempted"] is True
    assert payload["source_metadata"]["overview_metric_backfill"]["upserted_count"] == 6
    assert payload["source_metadata"]["backfill_used"] is True

    cached_payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")

    assert len(calls) == 6
    assert cached_payload["source_metadata"]["cache_hit"] is True
    assert cached_payload["source_metadata"]["overview_metric_backfill"]["attempted"] is False


def test_fetch_fund_overview_metric_rows_reads_fund_prices_first(tmp_path) -> None:
    """One indexed row-store read replaces the per-date snapshot documents."""
    fund_service.reset_fund_caches_for_tests()
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {
                "fund_code": "TLY",
                "date": "2026-04-30",
                "price": 10.0,
                "aum": 100_000_000.0,
                "investor_count": 500,
                "source": "tefasfon_funds",
            }
        ],
        source="tefasfon_funds",
    )

    class FakeTefasFonClient:
        def fetch_daily_funds_snapshot(self, as_of):
            pytest.fail(f"daily snapshot {as_of.isoformat()} should not be read")

    rows, fetched_months, warnings = fund_service._fetch_fund_overview_metric_rows(
        "TLY",
        [{"month": "2026-04", "month_start": date(2026, 4, 1), "target_date": date(2026, 4, 30)}],
        client=FakeTefasFonClient(),
        processed_dir=tmp_path,
    )

    assert fetched_months == {"2026-04": "2026-04-30"}
    assert rows[0]["aum"] == 100_000_000.0
    assert warnings == []


def test_fetch_recent_detail_rows_reads_fund_prices_first(tmp_path, monkeypatch) -> None:
    """Missing detail dates are resolved from the row store, not snapshots."""
    fund_service.reset_fund_caches_for_tests()
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {
                "fund_code": "TLY",
                "date": "2026-04-29",
                "price": 11.0,
                "aum": 110_000_000.0,
                "investor_count": 510,
                "source": "tefasfon_funds",
            },
            {
                "fund_code": "TLY",
                "date": "2026-04-30",
                "price": 12.0,
                "aum": 120_000_000.0,
                "investor_count": 520,
                "source": "tefasfon_funds",
            },
        ],
        source="tefasfon_funds",
    )

    class FakeTefasClient:
        def fetch_fund_history(self, **_kwargs):
            return []

    class FakeTefasFonClient:
        def fetch_history(self, *_args):
            return []

        def fetch_daily_funds_snapshot(self, as_of):
            pytest.fail(f"daily snapshot {as_of.isoformat()} should not be read")

    monkeypatch.setattr(fund_service, "TefasClient", FakeTefasClient)
    rows, warnings = fund_service._fetch_recent_detail_rows(
        "TLY",
        start_date=date(2026, 4, 27),
        end_date=date(2026, 4, 30),
        client=FakeTefasFonClient(),
        processed_dir=tmp_path,
        missing_dates={"2026-04-30"},
    )

    assert [row["date"] for row in rows] == ["2026-04-30"]
    assert "tefasfon_funds recent detail used fund_prices row store" in warnings


def test_fetch_recent_detail_rows_fills_remaining_dates_from_snapshots(tmp_path, monkeypatch) -> None:
    """A partial row-store hit must not skip the remaining missing dates."""
    fund_service.reset_fund_caches_for_tests()
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {
                "fund_code": "TLY",
                "date": "2026-04-29",
                "price": 11.0,
                "aum": 110_000_000.0,
                "investor_count": 510,
                "source": "tefasfon_funds",
            }
        ],
        source="tefasfon_funds",
    )

    class FakeTefasClient:
        def fetch_fund_history(self, **_kwargs):
            return []

    class FakeTefasFonClient:
        def fetch_history(self, *_args):
            return []

        def fetch_daily_funds_snapshot(self, _as_of):
            raise AssertionError("daily snapshot should be served by the patched helper")

    snapshot_calls = []

    def fake_daily_snapshot(_processed_dir, _client, as_of, *, force_refresh=False):
        snapshot_calls.append(as_of.isoformat())
        return (
            [
                {
                    "fund_code": "TLY",
                    "date": as_of.isoformat(),
                    "price": 12.0,
                    "aum": 120_000_000.0,
                    "investor_count": 520,
                }
            ],
            False,
        )

    monkeypatch.setattr(fund_service, "TefasClient", FakeTefasClient)
    monkeypatch.setattr(fund_service, "_cached_daily_funds_snapshot", fake_daily_snapshot)
    monkeypatch.setattr(fund_service, "FUNDS_WEB_HISTORY_SLEEP_SECONDS", 0.0)

    rows, warnings = fund_service._fetch_recent_detail_rows(
        "TLY",
        start_date=date(2026, 4, 27),
        end_date=date(2026, 4, 30),
        client=FakeTefasFonClient(),
        processed_dir=tmp_path,
        missing_dates={"2026-04-29", "2026-04-30"},
    )

    assert [row["date"] for row in rows] == ["2026-04-29", "2026-04-30"]
    assert snapshot_calls == ["2026-04-30"]
    assert "tefasfon_funds recent detail used fund_prices row store" in warnings
    assert "tefasfon_funds recent detail used daily snapshot fallback" in warnings


def test_get_fund_performance_payload_backfills_partial_range_overview_metrics(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 5, 21)

    current = date(2025, 10, 21)
    price_only_points = []
    price = 1.0
    while current <= date(2026, 5, 21):
        if current.weekday() < 5:
            price_only_points.append({
                "fund_code": "PBR",
                "date": current.isoformat(),
                "price": price,
                "source": "fintables_udf_history",
            })
            price += 0.01
        current += timedelta(days=1)
    fund_service.upsert_fund_price_points(tmp_path, price_only_points, source="fintables_udf_history")

    snapshot_by_date = {
        "2025-11-28": (100_000_000.0, 1_000),
        "2025-12-31": (200_000_000.0, 2_000),
        "2026-01-30": (300_000_000.0, 3_000),
        "2026-02-27": (400_000_000.0, 4_000),
        "2026-03-31": (500_000_000.0, 5_000),
        "2026-04-30": (600_000_000.0, 6_000),
        "2026-05-21": (700_000_000.0, 7_000),
    }
    calls = []
    history_calls = []

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            history_calls.append((fund_code, start_date, end_date))
            rows = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    rows.append(
                        {
                            "fonKodu": fund_code,
                            "tarih": current.isoformat(),
                            "fiyat": 1.0,
                            "portfoyBuyukluk": 800_000_000.0,
                            "kisiSayisi": 8_000,
                            "source": "tefasfon_funds",
                        }
                    )
                current += timedelta(days=1)
            return rows

        def fetch_daily_funds_snapshot(self, as_of):
            calls.append(as_of.isoformat())
            values = snapshot_by_date.get(as_of.isoformat())
            if values is None:
                return []
            aum, investors = values
            return [
                {
                    "fonKodu": "PBR",
                    "tarih": as_of.isoformat(),
                    "fiyat": 1.0,
                    "portfoyBuyukluk": aum,
                    "kisiSayisi": investors,
                    "source": "tefasfon_funds",
                }
            ]

    monkeypatch.setattr(fund_service, "date", FixedDate)
    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    _stub_direct_tefas_empty(monkeypatch)

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "PBR",
        start_date=date(2025, 10, 21),
        end_date=date(2026, 5, 21),
    )
    by_date = {point["date"]: point for point in payload["points"]}

    assert "2026-02-27" in calls
    assert "2026-03-31" in calls
    assert by_date["2026-02-27"]["aum"] == 400_000_000.0
    assert by_date["2026-03-31"]["investor_count"] == 5_000
    assert by_date["2026-05-20"]["aum"] == 800_000_000.0
    assert history_calls == [("PBR", date(2026, 4, 17), date(2026, 5, 21))]
    assert payload["source_metadata"]["recent_detail_backfill"]["attempted"] is True
    assert payload["source_metadata"]["overview_metric_backfill"]["attempted"] is True
    assert payload["source_metadata"]["full_history_requested"] is False


def test_get_fund_performance_payload_backfills_recent_details_from_cache(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()
    current = date(2026, 5, 4)
    price = 100.0
    price_only_points = []
    while current <= date(2026, 5, 21):
        if current.weekday() < 5:
            price_only_points.append(
                {
                    "fund_code": "RDT",
                    "date": current.isoformat(),
                    "price": price,
                    "source": "fintables_udf_history",
                }
            )
            price += 1.0
        current += timedelta(days=1)
    fund_service.upsert_fund_price_points(tmp_path, price_only_points, source="fintables_udf_history")

    history_calls = []

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            history_calls.append((fund_code, start_date, end_date))
            rows = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    rows.append(
                        {
                            "fonKodu": fund_code,
                            "tarih": current.isoformat(),
                            "fiyat": 100.0,
                            "portfoyBuyukluk": 900_000_000.0,
                            "kisiSayisi": 9_000,
                            "tedPaySayisi": 1_000_000,
                            "source": "tefasfon_funds",
                        }
                    )
                current += timedelta(days=1)
            return rows

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    _stub_direct_tefas_empty(monkeypatch)

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "RDT",
        start_date=date(2026, 5, 4),
        end_date=date(2026, 5, 21),
    )
    by_date = {point["date"]: point for point in payload["points"]}

    assert history_calls == [("RDT", date(2026, 5, 4), date(2026, 5, 21))]
    assert by_date["2026-05-20"]["aum"] == 900_000_000.0
    assert by_date["2026-05-20"]["investor_count"] == 9_000
    assert by_date["2026-05-20"]["source"] == "fintables_udf_history"
    assert payload["source_metadata"]["cache_hit"] is False
    assert payload["source_metadata"]["recent_detail_backfill"]["attempted"] is True


def test_get_fund_performance_payload_recent_detail_failure_keeps_prices(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "RDF", "date": "2026-05-20", "price": 100.0, "source": "fintables_udf_history"},
            {"fund_code": "RDF", "date": "2026-05-21", "price": 101.0, "source": "fintables_udf_history"},
        ],
        source="fintables_udf_history",
    )

    class FakeTefasFonClient:
        def fetch_history(self, fund_code, start_date, end_date):
            raise fund_service.TefasUpstreamError("blocked")

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    _stub_direct_tefas_empty(monkeypatch)

    payload = fund_service.get_fund_performance_payload(
        tmp_path,
        "RDF",
        start_date=date(2026, 5, 20),
        end_date=date(2026, 5, 21),
    )

    assert [point["date"] for point in payload["points"]] == ["2026-05-20", "2026-05-21"]
    assert payload["source_metadata"]["recent_detail_backfill"]["attempted"] is True
    assert payload["source_metadata"]["recent_detail_backfill"]["warning_count"] == 2
    assert "recent detail range backfill failed" in payload["source_metadata"]["warnings"][0]


def test_overview_metric_backfill_negative_cache_skips_repeat(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 5, 18)

    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "TLY", "date": "2026-04-30", "price": 5223.812438, "source": "fintables_udf_history"},
            {"fund_code": "TLY", "date": "2026-05-18", "price": 5488.8942, "source": "fintables_udf_history"},
        ],
        source="fintables_udf_history",
    )
    history_path = fund_service._history_path(tmp_path, "TLY")
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "source_metadata": {
                    "requested_start_date": "2025-11-01",
                    "date_max": "2026-05-18",
                    "parse_status": "ok",
                },
                "as_of": "2026-05-18",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class FakeTefasFonClient:
        def fetch_daily_funds_snapshot(self, as_of):
            calls.append(as_of.isoformat())
            raise fund_service.TefasUpstreamError("blocked")

    fake_client = FakeTefasFonClient()
    monkeypatch.setattr(fund_service, "date", FixedDate)
    monkeypatch.setattr(fund_service, "FUNDS_FULL_HISTORY_START_DATE", "2025-11-01")
    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: fake_client)

    first_payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")
    first_call_count = len(calls)
    second_payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")

    assert first_call_count > 0
    assert len(calls) == first_call_count
    assert first_payload["source_metadata"]["overview_metric_backfill"]["attempted"] is True
    assert second_payload["source_metadata"]["overview_metric_backfill"]["skipped_recent_failure"] is True


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
    assert payload["source_metadata"]["history_source_policy"] == "fintables_daily_tefas_fallback"
    assert payload["source_metadata"]["final_points_count"] == 0
    assert payload["source_metadata"]["backfill_used"] is True
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


def test_get_fund_yield_summary_payload_uses_fintables_first(monkeypatch) -> None:
    class FakeTefasFonClient:
        def fetch_yield_summary(self, fund_code, **kwargs):
            raise AssertionError("TEFAS should not be called when Fintables has a valid summary")

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(
        fund_service,
        "fetch_fintables_yield_summary",
        lambda _fund_code: {
            "fund_code": "TLY",
            "source": "fintables_yield_summary",
            "source_url": "https://fintables.example/yield",
            "periods": {
                "1m": {
                    "prev_close_date": "2026-04-01",
                    "prev_close": 10.0,
                    "high": 12.0,
                    "low": 9.5,
                }
            },
            "raw": {},
        },
    )

    payload = fund_service.get_fund_yield_summary_payload("TLY")

    assert payload["status"] == "ok"
    assert payload["source"] == "fintables_yield_summary"
    assert payload["periods"]["1m"]["prev_close"] == 10.0
    assert payload["source_metadata"]["summary_source_used"] == "fintables_yield_summary"
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


def test_fetch_fund_overview_metric_rows_uses_single_fund_range_before_daily_snapshot(
    monkeypatch, tmp_path
) -> None:
    fund_service.reset_fund_caches_for_tests()
    range_calls: list[tuple[str, date, date]] = []
    snapshot_calls: list[date] = []

    class FakeDirectClient:
        def fetch_fund_range(self, *, fund_code, start_date, end_date):
            range_calls.append((fund_code, start_date, end_date))
            if start_date.month == 4:
                return [
                    {
                        "fund_code": "TLY",
                        "date": "2026-04-30",
                        "price": 12.5,
                        "aum": 250_000_000.0,
                        "investor_count": 1250,
                        "share_count": 20_000_000.0,
                        "source": "tefasfon_funds",
                    }
                ]
            return []

    def fake_daily_snapshot(_self, as_of: date):
        snapshot_calls.append(as_of)
        if as_of == date(2026, 3, 31):
            return [
                {
                    "fonKodu": "TLY",
                    "tarih": "2026-03-31",
                    "fiyat": 11.0,
                    "portfoyBuyukluk": 200_000_000.0,
                    "kisiSayisi": 1000,
                    "tedPaySayisi": 18_000_000.0,
                }
            ]
        return []

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeDirectClient())
    monkeypatch.setattr(fund_service.TefasFonClient, "fetch_daily_funds_snapshot", fake_daily_snapshot)

    rows, fetched_months, warnings = fund_service._fetch_fund_overview_metric_rows(
        "TLY",
        [
            {"month": "2026-03", "month_start": date(2026, 3, 1), "target_date": date(2026, 3, 31)},
            {"month": "2026-04", "month_start": date(2026, 4, 1), "target_date": date(2026, 4, 30)},
        ],
        client=fund_service.TefasFonClient(),
        processed_dir=tmp_path,
    )

    assert sorted(call[1].month for call in range_calls) == [3, 4]
    # April was resolved via single-fund range so daily snapshot was NEVER requested for April!
    assert snapshot_calls == [date(2026, 3, 31)]
    assert fetched_months == {"2026-03": "2026-03-31", "2026-04": "2026-04-30"}
    assert len(rows) == 2
    assert rows[1]["aum"] == 250_000_000.0
    assert warnings == []


def test_get_fund_yield_summary_payload_uses_local_snapshot_fast_path_zero_http(
    monkeypatch, tmp_path
) -> None:
    fund_service.reset_fund_caches_for_tests()
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    target_day = date(2026, 5, 20)

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 5, 20)

    monkeypatch.setattr(fund_service, "date", FixedDate)
    monkeypatch.setattr(fund_service, "_turkey_market_today", lambda: target_day)
    assert fund_service.commit_funds_snapshot(
        tmp_path,
        {
            "status": "ok",
            "as_of": target_day.isoformat(),
            "rows": [
                {
                    "fund_code": "TLY",
                    "name": "TERA TEST FONU",
                    "as_of": target_day.isoformat(),
                    "price": 120.0,
                    "tefas_open": True,
                    "period_returns": {
                        "1w": 2.0,
                        "1m": 20.0,
                        "3m": 50.0,
                        "6m": 80.0,
                        "ytd": 60.0,
                        "1y": 100.0,
                    },
                }
            ],
            "source_metadata": {"source": "tefasfon_funds"},
        },
        job_id="local-fast-test",
        owner_token="local-fast-test",
        generation=1,
    )
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "TLY", "date": "2026-04-20", "price": 95.0, "source": "fintables_udf_history"},
            {"fund_code": "TLY", "date": "2026-05-10", "price": 125.0, "source": "fintables_udf_history"},
            {"fund_code": "TLY", "date": "2026-05-20", "price": 120.0, "source": "fintables_udf_history"},
        ],
        source="fintables_udf_history",
    )

    monkeypatch.setattr(
        fund_service,
        "fetch_fintables_yield_summary",
        lambda *_a, **_k: pytest.fail("Fintables should not be called when local snapshot is complete"),
    )
    monkeypatch.setattr(
        fund_service,
        "TefasFonClient",
        lambda: pytest.fail("TEFAS should not be called when local snapshot is complete"),
    )

    payload = fund_service.get_fund_yield_summary_payload("TLY", processed_dir=tmp_path)

    assert payload["status"] == "ok"
    assert payload["source"] == "local_fund_snapshot"
    assert payload["source_metadata"]["local_fast_path"] is True
    assert payload["source_metadata"]["fallback_used"] is False
    assert payload["periods"]["1m"]["prev_close"] == pytest.approx(100.0)
    assert payload["periods"]["1m"]["high"] == pytest.approx(125.0)
    assert payload["periods"]["1m"]["low"] == pytest.approx(95.0)
    assert payload["periods"]["1m"]["prev_close_date"] == "2026-04-20"
    assert payload["periods"]["oldest"]["prev_close"] == pytest.approx(95.0)

    # Stale snapshot (> 5 days old) cleanly falls back to Fintables.
    stale_dir = tmp_path / "stale_case"
    fintables_calls: list[str] = []
    assert fund_service.commit_funds_snapshot(
        stale_dir,
        {
            "status": "ok",
            "as_of": "2026-05-01",
            "rows": [
                {
                    "fund_code": "TLY",
                    "name": "TERA TEST FONU",
                    "as_of": "2026-05-01",
                    "price": 105.0,
                    "tefas_open": True,
                    "period_returns": {"1m": 10.0, "3m": 20.0, "6m": 30.0, "1y": 40.0},
                }
            ],
            "source_metadata": {"source": "tefasfon_funds"},
        },
        job_id="stale-test",
        owner_token="stale-test",
        generation=1,
    )
    fund_service.upsert_fund_price_points(
        stale_dir,
        [{"fund_code": "TLY", "date": "2026-05-01", "price": 105.0, "source": "fintables_udf_history"}],
        source="fintables_udf_history",
    )
    monkeypatch.setattr(
        fund_service,
        "fetch_fintables_yield_summary",
        lambda code: fintables_calls.append(code)
        or {
            "fund_code": code,
            "source": "fintables_yield_summary",
            "source_url": "https://fintables.example/yield",
            "periods": {"1m": {"prev_close_date": "2026-04-20", "prev_close": 90.0, "high": 110.0, "low": 88.0}},
            "raw": {},
        },
    )
    stale_fallback = fund_service.get_fund_yield_summary_payload("TLY", processed_dir=stale_dir)
    assert fintables_calls == ["TLY"]
    assert stale_fallback["source"] == "fintables_yield_summary"


def test_period_returns_and_summary_handle_3y_5y_and_zero_sentinel() -> None:
    points_5y = [
        {"date": "2021-11-19", "price": 10.0},
        {"date": "2023-11-20", "price": 25.0},
        {"date": "2025-11-20", "price": 40.0},
        {"date": "2026-11-20", "price": 50.0},
    ]
    summary_5y = fund_service._summary_from_points("TMG", points_5y)
    assert summary_5y is not None
    assert summary_5y["period_returns"]["3y"] == pytest.approx(100.0)
    assert summary_5y["period_returns"]["5y"] == pytest.approx(400.0)

    points_3y_only = points_5y[1:]
    summary_3y = fund_service._summary_from_points("BUY", points_3y_only)
    assert summary_3y is not None
    assert summary_3y["period_returns"]["3y"] == pytest.approx(100.0)
    assert "5y" not in summary_3y["period_returns"]

    points_young = points_5y[2:]
    summary_young = fund_service._summary_from_points("YNG", points_young)
    assert summary_young is not None
    assert "3y" not in summary_young["period_returns"]
    assert "5y" not in summary_young["period_returns"]

    # TEFAS 0.0 sentinel across 1y/3y/5y must not leak into extended returns
    sentinel_raw = {
        "getiri1a": 5.0,
        "getiri3a": 10.0,
        "getiri6a": 12.0,
        "getiri1y": 0.0,
        "getiri3y": 0.0,
        "getiri5y": 0.0,
    }
    parsed_sentinel = fund_service._period_returns_from_raw(sentinel_raw)
    assert parsed_sentinel["3y"] is None
    assert parsed_sentinel["5y"] is None


def test_local_yield_summary_and_detail_enrich_3y_5y_zero_http(monkeypatch, tmp_path) -> None:
    fund_service.reset_fund_caches_for_tests()
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    target_day = date(2026, 11, 20)

    class FixedDate(date):
        @classmethod
        def today(cls) -> "FixedDate":
            return cls(2026, 11, 20)

    monkeypatch.setattr(fund_service, "date", FixedDate)
    monkeypatch.setattr(fund_service, "_turkey_market_today", lambda: target_day)
    assert fund_service.commit_funds_snapshot(
        tmp_path,
        {
            "status": "ok",
            "as_of": target_day.isoformat(),
            "rows": [
                {
                    "fund_code": "TLY",
                    "name": "TERA BIRINCI",
                    "as_of": target_day.isoformat(),
                    "price": 100.0,
                    "tefas_open": True,
                    "period_returns": {
                        "1w": 2.0,
                        "1m": 10.0,
                        "3m": 25.0,
                        "6m": 50.0,
                        "ytd": 75.0,
                        "1y": 0.0,
                        "3y": 0.0,
                        "5y": 0.0,
                    },
                }
            ],
            "source_metadata": {"source": "tefasfon_funds"},
        },
        job_id="local-ext-test",
        owner_token="local-ext-test",
        generation=1,
    )
    fund_service.upsert_fund_price_points(
        tmp_path,
        [
            {"fund_code": "TLY", "date": "2021-11-19", "price": 10.0, "source": "fintables_udf_history"},
            {"fund_code": "TLY", "date": "2023-11-20", "price": 20.0, "source": "fintables_udf_history"},
            {"fund_code": "TLY", "date": "2026-10-20", "price": 90.0, "source": "fintables_udf_history"},
            {"fund_code": "TLY", "date": "2026-11-20", "price": 100.0, "source": "fintables_udf_history"},
        ],
        source="fintables_udf_history",
    )

    monkeypatch.setattr(
        fund_service,
        "fetch_fintables_yield_summary",
        lambda *_a, **_k: pytest.fail("Fintables HTTP must not be called"),
    )
    monkeypatch.setattr(
        fund_service,
        "TefasFonClient",
        lambda: pytest.fail("TEFAS HTTP must not be called"),
    )

    detail = fund_service.get_fund_detail_payload(tmp_path, "TLY")
    assert detail["history_start_date"] == "2021-11-19"
    assert detail["period_returns"]["3y"] == pytest.approx(400.0)
    assert detail["period_returns"]["5y"] == pytest.approx(900.0)

    yield_payload = fund_service.get_fund_yield_summary_payload("TLY", processed_dir=tmp_path)
    assert yield_payload["source"] == "local_fund_snapshot"
    assert yield_payload["periods"]["3y"]["prev_close_date"] == "2023-11-20"
    assert yield_payload["periods"]["3y"]["prev_close"] == pytest.approx(20.0)
    assert yield_payload["periods"]["3y"]["return_pct"] == pytest.approx(400.0)
    assert yield_payload["periods"]["5y"]["prev_close_date"] == "2021-11-19"
    assert yield_payload["periods"]["5y"]["prev_close"] == pytest.approx(10.0)
    assert yield_payload["periods"]["5y"]["return_pct"] == pytest.approx(900.0)
    assert yield_payload["periods"]["oldest"]["prev_close_date"] == "2021-11-19"


def test_history_job_should_schedule_allows_widening_running_job_for_3y_5y() -> None:
    from app import fund_api_service

    running_1y_job = {
        "job_id": "job-1y",
        "fund_code": "TMG",
        "effective_start": "2025-11-20",
        "effective_end": "2026-11-21",
        "status": "running",
    }
    payload = {
        "points": [{"date": "2025-11-20", "price": 1.1}, {"date": "2026-11-20", "price": 1.44}],
        "source_metadata": {"history_source_used": "tefasfon_funds", "available_end_date": "2026-11-20"},
    }
    target_1y = {"requested_start": date(2025, 11, 21), "requested_end": date(2026, 11, 21)}
    target_5y = {"requested_start": date(2021, 11, 21), "requested_end": date(2026, 11, 21)}

    assert fund_api_service._history_job_should_schedule(payload, last_job=running_1y_job, target=target_1y) is False
    assert fund_api_service._history_job_should_schedule(payload, last_job=running_1y_job, target=target_5y) is True
