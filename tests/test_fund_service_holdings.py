from __future__ import annotations

import json
from datetime import date

import pytest

from app import fund_service
from app.reference_data import upsert_instrument


def test_kap_stock_name_prefers_reference_data(tmp_path) -> None:
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="TERA",
        name="TERA YATIRIM MENKUL DEĞERLER A.Ş.",
        source="kap",
        source_id="member-1",
    )
    cache_dir = tmp_path / "kap_cache"
    cache_dir.mkdir()
    (cache_dir / "TERA.json").write_text(
        json.dumps({"company_title": "MEDITERA TIBBİ MALZEME SANAYİ VE TİCARET A.Ş."}),
        encoding="utf-8",
    )

    assert fund_service._kap_stock_name_from_cache(tmp_path, "TERA") == "TERA YATIRIM MENKUL DEĞERLER A.Ş."


def test_tefasfon_portfolio_filters_requested_fund_code() -> None:
    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            self.fund_types = ("SEC",)

        def _fetch_portfolio_direct_request(self, fund_code, start_date, end_date, *, fund_type):
            raise fund_service.TefasUpstreamError("direct request path disabled in test")

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


def test_tefasfon_portfolio_prefers_direct_request_before_package_fallback() -> None:
    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            self.fund_types = ("SEC",)

        def _fetch_portfolio_direct_request(self, fund_code, start_date, end_date, *, fund_type):
            return [
                {
                    "fonKodu": fund_code,
                    "tarih": "2026-04-30",
                    "hs": 58.0,
                    "source": fund_service.TEFAS_DIRECT_PORTFOLIO_SOURCE,
                }
            ]

        def _fetch_portfolio_direct(self, fund_code, start_date, end_date, *, fund_type):
            raise AssertionError("tefasfon getter fallback should not be called")

    rows = FakeTefasFonClient().fetch_portfolio(
        fund_code="TLY",
        start_date=date(2026, 4, 30),
        end_date=date(2026, 4, 30),
    )

    assert rows[0]["hs"] == 58.0
    assert rows[0]["source"] == fund_service.TEFAS_DIRECT_PORTFOLIO_SOURCE


def test_tefasfon_portfolio_stops_on_direct_rate_limit_without_package_retry() -> None:
    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            self.fund_types = ("SEC",)

        def _fetch_portfolio_direct_request(self, fund_code, start_date, end_date, *, fund_type):
            raise fund_service.TefasRateLimitError("direct portfolio HTTP 429")

        def _fetch_portfolio_direct(self, fund_code, start_date, end_date, *, fund_type):
            raise AssertionError("rate-limited direct request must not immediately retry through tefasfon")

    with pytest.raises(fund_service.TefasRateLimitError):
        FakeTefasFonClient().fetch_portfolio(
            fund_code="TLY",
            start_date=date(2026, 4, 30),
            end_date=date(2026, 4, 30),
        )


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
    current = fund_service.get_fund_allocations_payload(tmp_path, "TLY")
    assert current["source_metadata"]["as_of"] == "2026-04-30"
    assert current["allocations"][0]["weight"] == 58.0


def test_get_fund_allocations_payload_promotes_newer_fresh_history(monkeypatch, tmp_path) -> None:
    allocations_dir = tmp_path / "funds_cache" / "allocations"
    allocations_dir.mkdir(parents=True)
    (allocations_dir / "TLY.json").write_text(
        json.dumps(
            {
                "fund_code": "TLY",
                "status": "ok",
                "allocations": [
                    {
                        "fund_code": "TLY",
                        "allocation_type": "hs",
                        "label": "Hisse Senedi",
                        "weight": 80.3,
                        "report_date": "2026-09-02",
                        "source": "tefasfon_portfolio",
                    }
                ],
                "source": "tefasfon_portfolio",
                "stale": True,
                "source_metadata": {
                    "as_of": "2026-09-02",
                    "fetched_at": "2026-09-02T18:00:00+00:00",
                },
            }
        ),
        encoding="utf-8",
    )
    (allocations_dir / "TLY_history_30d.json").write_text(
        json.dumps(
            {
                "fund_code": "TLY",
                "status": "ok",
                "lookback_days": 30,
                "history": [
                    {
                        "date": "2026-09-08",
                        "allocations": [
                            {
                                "fund_code": "TLY",
                                "allocation_type": "hs",
                                "label": "Hisse Senedi",
                                "weight": 84.57,
                                "report_date": "2026-09-08",
                                "source": "tefas_direct_portfolio",
                            }
                        ],
                    }
                ],
                "source": "tefasfon_portfolio",
                "stale": False,
                "source_metadata": {
                    "as_of": "2026-09-08",
                    "fetched_at": fund_service._utc_now_iso(),
                    "warnings": [],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = fund_service.get_fund_allocations_payload(tmp_path, "TLY")

    assert payload["stale"] is False
    assert payload["source_metadata"]["as_of"] == "2026-09-08"
    assert payload["source_metadata"]["promoted_from"] == "allocation_history"
    assert payload["allocations"][0]["weight"] == 84.57


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


def test_tefasfon_portfolio_authoritative_empty_skips_redundant_fallbacks() -> None:
    calls: list[str] = []

    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            self.fund_types = ("SEC",)

        def _fetch_portfolio_direct_request(self, fund_code, start_date, end_date, *, fund_type):
            calls.append("direct_request")
            return []

        def _fetch_portfolio_direct(self, fund_code, start_date, end_date, *, fund_type):
            calls.append("tefasfon_direct")
            raise AssertionError("authoritative empty direct response must not trigger tefasfon direct fallback")

        def _call_dataframe(self, function_name, *, context, **kwargs):
            calls.append("tefasfon_dataframe")
            raise AssertionError("authoritative empty direct response must not trigger tefasfon dataframe fallback")

    rows = FakeTefasFonClient().fetch_portfolio(
        fund_code="TLY",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
    )

    assert rows == []
    assert calls == ["direct_request"]


def test_refresh_fund_allocations_history_daily_fallback_early_exits_after_consecutive_empty_days(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[date, date]] = []

    class FakeTefasFonClient:
        def fetch_portfolio(self, *, fund_code, start_date, end_date):
            calls.append((start_date, end_date))
            return []

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())
    monkeypatch.setattr(fund_service, "FUNDS_ALLOCATION_HISTORY_DAILY_FALLBACK_MAX_CONSECUTIVE_EMPTY", 3)

    payload = fund_service.refresh_fund_allocations_history(
        tmp_path,
        "EMPTYFUND",
        lookback_days=30,
        as_of=date(2026, 4, 30),
        allow_daily_fallback=True,
    )

    assert payload["status"] == "empty"
    # 1 full range query + 3 newest consecutive empty business days (Apr 30, Apr 29, Apr 28) -> stops!
    assert calls == [
        (date(2026, 4, 1), date(2026, 4, 30)),
        (date(2026, 4, 30), date(2026, 4, 30)),
        (date(2026, 4, 29), date(2026, 4, 29)),
        (date(2026, 4, 28), date(2026, 4, 28)),
    ]


def test_get_fund_allocations_history_payload_slices_smaller_lookback_from_canonical_store_zero_http(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[date, date]] = []

    class FakeTefasFonClient:
        def fetch_portfolio(self, *, fund_code, start_date, end_date):
            calls.append((start_date, end_date))
            return [
                {"fonKodu": fund_code, "tarih": "2026-02-15", "hs": 45.0, "source": "tefasfon_portfolio"},
                {"fonKodu": fund_code, "tarih": "2026-04-10", "hs": 52.0, "source": "tefasfon_portfolio"},
                {"fonKodu": fund_code, "tarih": "2026-04-28", "hs": 58.0, "source": "tefasfon_portfolio"},
            ]

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    seeded = fund_service.refresh_fund_allocations_history(
        tmp_path,
        "TLY",
        lookback_days=90,
        as_of=date(2026, 4, 30),
    )
    assert len(calls) == 1
    assert [row["date"] for row in seeded["history"]] == ["2026-02-15", "2026-04-10", "2026-04-28"]

    # Shorter lookback windows slice directly from the canonical cache with 0 TEFAS calls.
    sliced_30 = fund_service.get_fund_allocations_history_payload(tmp_path, "TLY", lookback_days=30)
    assert sliced_30["source_metadata"]["cache_hit"] is True
    assert sliced_30["lookback_days"] == 30
    assert [row["date"] for row in sliced_30["history"]] == ["2026-04-10", "2026-04-28"]

    sliced_7 = fund_service.get_fund_allocations_history_payload(tmp_path, "TLY", lookback_days=7)
    assert sliced_7["source_metadata"]["cache_hit"] is True
    assert sliced_7["lookback_days"] == 7
    assert [row["date"] for row in sliced_7["history"]] == ["2026-04-28"]
    assert len(calls) == 1


def test_refresh_fund_allocations_history_fetches_only_delta_span_on_tail_and_head_expansion(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[date, date]] = []

    class FakeTefasFonClient:
        def fetch_portfolio(self, *, fund_code, start_date, end_date):
            calls.append((start_date, end_date))
            rows = []
            if start_date <= date(2026, 4, 15) <= end_date:
                rows.append({"fonKodu": fund_code, "tarih": "2026-04-15", "hs": 50.0, "source": "tefasfon_portfolio"})
            if start_date <= date(2026, 4, 28) <= end_date:
                rows.append({"fonKodu": fund_code, "tarih": "2026-04-28", "hs": 56.0, "source": "tefasfon_portfolio"})
            if start_date <= date(2026, 3, 10) <= end_date:
                rows.append({"fonKodu": fund_code, "tarih": "2026-03-10", "hs": 42.0, "source": "tefasfon_portfolio"})
            return rows

    monkeypatch.setattr(fund_service, "TefasFonClient", lambda: FakeTefasFonClient())

    # 1. Seed 30d ending 2026-04-25 -> [2026-03-27 .. 2026-04-25]
    seed_payload = fund_service.refresh_fund_allocations_history(
        tmp_path,
        "TLY",
        lookback_days=30,
        as_of=date(2026, 4, 25),
    )
    assert calls[-1] == (date(2026, 3, 27), date(2026, 4, 25))
    assert [row["date"] for row in seed_payload["history"]] == ["2026-04-15"]

    # 2. Advance as_of to 2026-04-30 with lookback=30 -> fetches ONLY tail delta (2026-04-26..2026-04-30)
    tail_payload = fund_service.refresh_fund_allocations_history(
        tmp_path,
        "TLY",
        lookback_days=30,
        as_of=date(2026, 4, 30),
    )
    assert calls[-1] == (date(2026, 4, 26), date(2026, 4, 30))
    assert [row["date"] for row in tail_payload["history"]] == ["2026-04-15", "2026-04-28"]

    # 3. Expand lookback to 60d ending 2026-04-30 -> fetches ONLY head delta (2026-03-02..2026-03-26)
    head_payload = fund_service.refresh_fund_allocations_history(
        tmp_path,
        "TLY",
        lookback_days=60,
        as_of=date(2026, 4, 30),
    )
    assert calls[-1] == (date(2026, 3, 2), date(2026, 3, 26))
    assert [row["date"] for row in head_payload["history"]] == ["2026-03-10", "2026-04-15", "2026-04-28"]


def test_fund_allocation_history_executor_defaults_to_three_workers() -> None:
    from app import fund_api_service

    assert fund_api_service._FUND_ALLOCATION_HISTORY_EXECUTOR._max_workers == 3


def test_refresh_fund_holdings_reuses_cached_kap_oids_and_falls_back_when_stale(
    monkeypatch, tmp_path
) -> None:
    search_calls: list[str] = []
    subject_calls: list[str] = []
    list_calls: list[tuple[str, str]] = []

    cached_positions = [
        {
            "fund_code": "TLY",
            "asset_code": "DSTKF",
            "asset_name": "DESTEK FINANS FAKTORING A.S.",
            "asset_type": "local_equity",
            "weight": 23.6,
            "previous_weight": 20.4,
            "weight_change": 3.2,
            "change_status": "increased",
            "report_date": "2026-04-30",
            "previous_report_date": "2026-03-31",
            "source_report_url": "https://www.kap.org.tr/tr/Bildirim/200",
            "source_type": "kap_pdf",
            "parse_confidence": 0.9,
        }
    ]
    cached_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": cached_positions,
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-05T10:00:00+00:00",
            "as_of": "2026-04-30",
            "parse_status": "ok",
            "parser_version": fund_service.KAP_HOLDINGS_PARSE_VERSION,
            "fund_oid": "cached-fund-oid",
            "subject_oid": "cached-subject-oid",
            "fund_name": "Tera Portfoy Birinci Serbest Fon",
            "latest_report": {"disclosure_index": 200, "report_date": "2026-04-30", "file_name": "TLY_2026.04.pdf"},
            "previous_report": {"disclosure_index": 100, "report_date": "2026-03-31", "file_name": "TLY_2026.03.pdf"},
            "positions_hash": fund_service._holdings_positions_hash(cached_positions),
        },
    }

    monkeypatch.setattr(
        fund_service,
        "_kap_search_fund_metadata",
        lambda code: (
            search_calls.append(code)
            or {"fund_code": code, "fund_oid": "fresh-fund-oid", "fund_name": "Tera Portfoy"}
        ),
    )
    monkeypatch.setattr(
        fund_service,
        "_kap_portfolio_subject_oid",
        lambda f_oid: subject_calls.append(f_oid) or "fresh-subject-oid",
    )

    # Case 1: Cached OIDs return disclosures -> skips search POST and subject HTML GET completely!
    monkeypatch.setattr(
        fund_service,
        "_kap_list_portfolio_disclosures",
        lambda f_oid, s_oid: (
            list_calls.append((f_oid, s_oid))
            or [
                {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "05.05.2026 09:00:00"}},
                {"disclosureBasic": {"disclosureIndex": 100, "publishDate": "05.04.2026 09:00:00"}},
            ]
        ),
    )

    res1 = fund_service.refresh_fund_holdings(tmp_path, "TLY", cached_payload=cached_payload)
    assert res1["status"] == "ok"
    assert search_calls == []
    assert subject_calls == []
    assert list_calls == [("cached-fund-oid", "cached-subject-oid")]

    # Case 2: Cached OIDs are stale (return empty) -> falls back to _kap_search_fund_metadata + _kap_portfolio_subject_oid!
    list_calls.clear()
    monkeypatch.setattr(
        fund_service,
        "_kap_list_portfolio_disclosures",
        lambda f_oid, s_oid: (
            list_calls.append((f_oid, s_oid))
            or (
                []
                if f_oid == "cached-fund-oid"
                else [
                    {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "05.05.2026 09:00:00"}},
                    {"disclosureBasic": {"disclosureIndex": 100, "publishDate": "05.04.2026 09:00:00"}},
                ]
            )
        ),
    )

    res2 = fund_service.refresh_fund_holdings(tmp_path, "TLY", cached_payload=cached_payload)
    assert res2["status"] == "ok"
    assert search_calls == ["TLY"]
    assert subject_calls == ["fresh-fund-oid"]
    assert list_calls == [
        ("cached-fund-oid", "cached-subject-oid"),
        ("fresh-fund-oid", "fresh-subject-oid"),
    ]


def test_effective_holdings_disclosure_check_ttl_is_calendar_aware(monkeypatch) -> None:
    from datetime import datetime, timezone

    monkeypatch.setattr(fund_service, "FUNDS_HOLDINGS_DISCLOSURE_CHECK_TTL_SECONDS", 7200)
    monkeypatch.setattr(fund_service, "FUNDS_HOLDINGS_DISCLOSURE_CHECK_SETTLED_TTL_SECONDS", 86400)

    cached_meta = {
        "as_of": "2026-04-30",
        "latest_report": {"disclosure_index": 200, "report_date": "2026-04-30"},
        "disclosure_check": {
            "checked_at": "2026-05-18T08:00:00+00:00",
            "ttl_seconds": 86400,
        },
    }

    # On May 18 (day > 10, previous month's April report already cached), TTL is 24h (86400s).
    monkeypatch.setattr(
        fund_service,
        "_utc_now",
        lambda: datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc),
    )
    assert fund_service._effective_holdings_disclosure_check_ttl_seconds(cached_meta) == 86400
    # 6 hours elapsed (< 24h) -> check is NOT due!
    assert fund_service._holdings_disclosure_check_due(cached_meta) is False

    # On May 5 (day <= 10, publication window for April report if March report is cached), TTL is 2h (7200s).
    march_cached_meta = {
        "as_of": "2026-03-31",
        "latest_report": {"disclosure_index": 100, "report_date": "2026-03-31"},
        "disclosure_check": {
            "checked_at": "2026-04-05T08:00:00+00:00",
            "ttl_seconds": 7200,
        },
    }
    monkeypatch.setattr(
        fund_service,
        "_utc_now",
        lambda: datetime(2026, 4, 5, 11, 0, tzinfo=timezone.utc),
    )
    assert fund_service._effective_holdings_disclosure_check_ttl_seconds(march_cached_meta) == 7200
    # 3 hours elapsed (> 2h) -> check IS due!
    assert fund_service._holdings_disclosure_check_due(march_cached_meta) is True


def test_refresh_fund_holdings_reuses_previous_month_positions_on_new_report_rollover(
    monkeypatch, tmp_path
) -> None:
    latest_may_text = """
FON PORTFÖY DEĞERİ TABLOSU
Hisse Senedi
DSTKF Destek Finans Faktoring A.S. 1.000,00 10,00 10.000,00 26,00TRYTRADSTKFXXX
BIMAS Bim Birlesik Magazalar A.S. 100,00 5,00 500,00 4,50TRYTRABIMASXXX
"""
    cached_april_positions = [
        {
            "fund_code": "TLY",
            "asset_code": "DSTKF",
            "asset_name": "DESTEK FINANS FAKTORING A.S.",
            "asset_type": "local_equity",
            "weight": 20.0,
            "previous_weight": 18.0,
            "weight_change": 2.0,
            "change_status": "increased",
            "report_date": "2026-04-30",
            "previous_report_date": "2026-03-31",
            "source_report_url": "https://www.kap.org.tr/tr/Bildirim/200",
            "source_type": "kap_pdf",
            "parse_confidence": 0.9,
        },
        {
            "fund_code": "TLY",
            "asset_code": "AKBNK",
            "asset_name": "AKBANK T.A.S.",
            "asset_type": "local_equity",
            "weight": 5.0,
            "previous_weight": 5.0,
            "weight_change": 0.0,
            "change_status": "unchanged",
            "report_date": "2026-04-30",
            "previous_report_date": "2026-03-31",
            "source_report_url": "https://www.kap.org.tr/tr/Bildirim/200",
            "source_type": "kap_pdf",
            "parse_confidence": 0.9,
        },
        {
            "fund_code": "TLY",
            "asset_code": "OLDREMOVED",
            "asset_name": "ESKI HISSE",
            "asset_type": "local_equity",
            "weight": 0.0,
            "previous_weight": 3.0,
            "weight_change": -3.0,
            "change_status": "removed",
            "report_date": "2026-04-30",
            "previous_report_date": "2026-03-31",
            "source_report_url": "https://www.kap.org.tr/tr/Bildirim/100",
            "source_type": "kap_pdf",
            "parse_confidence": 0.9,
        },
    ]
    cached_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": cached_april_positions,
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-05T10:00:00+00:00",
            "as_of": "2026-04-30",
            "parse_status": "ok",
            "parser_version": fund_service.KAP_HOLDINGS_PARSE_VERSION,
            "fund_oid": "fund-oid",
            "subject_oid": "subject-oid",
            "fund_name": "Tera Portfoy",
            "latest_report": {
                "disclosure_index": 200,
                "report_date": "2026-04-30",
                "file_name": "TLY_2026.04.pdf",
                "attachment_obj_id": "april-pdf",
                "source_url": "https://www.kap.org.tr/tr/Bildirim/200",
            },
            "previous_report": {
                "disclosure_index": 100,
                "report_date": "2026-03-31",
                "file_name": "TLY_2026.03.pdf",
            },
            "positions_hash": fund_service._holdings_positions_hash(cached_april_positions),
        },
    }

    detail_calls: list[int] = []
    download_calls: list[str] = []

    monkeypatch.setattr(
        fund_service,
        "_kap_list_portfolio_disclosures",
        lambda _f_oid, _s_oid: [
            {"disclosureBasic": {"disclosureIndex": 300, "publishDate": "04.06.2026 09:00:00"}},
            {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "05.05.2026 09:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service,
        "_kap_fetch_report_detail",
        lambda idx: (
            detail_calls.append(idx)
            or {
                "disclosure": {"disclosureBasic": {"disclosureIndex": idx, "year": 2026, "donem": 5}},
                "attachments": [{"objId": "may-pdf", "fileName": "TLY_2026.05.pdf", "fileExtension": "pdf"}],
            }
        ),
    )
    monkeypatch.setattr(
        fund_service,
        "_kap_download_attachment",
        lambda obj_id: download_calls.append(obj_id) or b"may-pdf-bytes",
    )
    monkeypatch.setattr(fund_service, "_extract_kap_pdf_text", lambda _data: latest_may_text)

    payload = fund_service.refresh_fund_holdings(tmp_path, "TLY", cached_payload=cached_payload)

    assert payload["status"] == "ok"
    # Only the new report (300) was fetched & parsed; report 200 was reused from cached_payload!
    assert detail_calls == [300]
    assert download_calls == ["may-pdf"]
    by_code = {item["asset_code"]: item for item in payload["positions"]}
    assert by_code["DSTKF"]["weight"] == 26.0
    assert by_code["DSTKF"]["previous_weight"] == 20.0
    assert by_code["DSTKF"]["weight_change"] == 6.0
    assert by_code["DSTKF"]["change_status"] == "increased"
    assert by_code["BIMAS"]["change_status"] == "new"
    assert by_code["AKBNK"]["change_status"] == "removed"
    assert by_code["AKBNK"]["previous_weight"] == 5.0
    assert "OLDREMOVED" not in by_code


def test_refresh_fund_holdings_parses_two_reports_in_parallel_on_cold_cache(
    monkeypatch, tmp_path
) -> None:
    import threading
    import time

    latest_text = """
FON PORTFÖY DEĞERİ TABLOSU
Hisse Senedi
DSTKF Destek Finans Faktoring A.S. 1.000,00 10,00 10.000,00 24,00TRYTRADSTKFXXX
"""
    previous_text = """
FON PORTFÖY DEĞERİ TABLOSU
Hisse Senedi
DSTKF Destek Finans Faktoring A.S. 1.000,00 10,00 10.000,00 19,00TRYTRADSTKFXXX
"""
    lock = threading.Lock()
    active = 0
    peak_concurrency = 0

    monkeypatch.setattr(
        fund_service,
        "_kap_search_fund_metadata",
        lambda _code: {"fund_code": "TLY", "fund_oid": "fund-oid", "fund_name": "Tera Portfoy"},
    )
    monkeypatch.setattr(fund_service, "_kap_portfolio_subject_oid", lambda _f_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service,
        "_kap_list_portfolio_disclosures",
        lambda _f_oid, _s_oid: [
            {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "05.05.2026 09:00:00"}},
            {"disclosureBasic": {"disclosureIndex": 100, "publishDate": "05.04.2026 09:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service,
        "_kap_fetch_report_detail",
        lambda idx: {
            "disclosure": {"disclosureBasic": {"disclosureIndex": idx, "year": 2026, "donem": 4 if idx == 200 else 3}},
            "attachments": [{"objId": f"obj-{idx}", "fileName": f"TLY_{idx}.pdf", "fileExtension": "pdf"}],
        },
    )

    def fake_download(obj_id: str) -> bytes:
        nonlocal active, peak_concurrency
        with lock:
            active += 1
            if active > peak_concurrency:
                peak_concurrency = active
        try:
            time.sleep(0.04)
            return obj_id.encode("utf-8")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(fund_service, "_kap_download_attachment", fake_download)
    monkeypatch.setattr(
        fund_service,
        "_extract_kap_pdf_text",
        lambda data: latest_text if data == b"obj-200" else previous_text,
    )

    payload = fund_service.refresh_fund_holdings(tmp_path, "TLY")

    assert payload["status"] == "ok"
    assert peak_concurrency == 2
    assert payload["positions"][0]["weight"] == 24.0
    assert payload["positions"][0]["previous_weight"] == 19.0
    assert payload["positions"][0]["weight_change"] == 5.0
