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
