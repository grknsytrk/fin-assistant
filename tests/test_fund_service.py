from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from app import fund_service
from app.reference_data import get_instrument, upsert_instrument


def test_fintables_defaults_match_current_history_contract() -> None:
    assert fund_service.FINTABLES_UDF_HISTORY_ENDPOINT.endswith("/barbar/udf/history")
    assert fund_service.FINTABLES_YIELD_SUMMARY_ENDPOINT.endswith("/barbar/server/yield")


def test_tefas_client_fetch_fund_range_retries_rate_limit(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __init__(self, status_code, headers, payload):
            self.status_code = status_code
            self.headers = headers
            self.content = payload

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, **kwargs):
            calls.append(kwargs.get("json"))
            if len(calls) == 1:
                return FakeResponse(429, {"Retry-After": "0"}, b"")
            return FakeResponse(
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "data": [
                            {
                                "fonKodu": "TLY",
                                "tarih": "2026-06-24",
                                "fiyat": 6686.9308,
                                "portfoyBuyukluk": 188_308_204_646.0,
                                "kisiSayisi": 90_110,
                                "tedPaySayisi": 28_160_633,
                            }
                        ]
                    }
                ).encode("utf-8"),
            )

    monkeypatch.setattr(fund_service.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(fund_service, "TEFAS_HTTP_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(fund_service, "TEFAS_HTTP_RETRY_BASE_SECONDS", 0.0)
    monkeypatch.setattr(fund_service, "TEFAS_HTTP_RETRY_MAX_SECONDS", 0.0)

    rows = fund_service.TefasClient().fetch_fund_range(
        fund_code="TLY",
        start_date=date(2026, 6, 24),
        end_date=date(2026, 6, 24),
    )

    assert len(calls) == 2
    assert rows[0]["source"] == "tefasfon_funds"
    assert rows[0]["aum"] == 188_308_204_646.0
    assert rows[0]["investor_count"] == 90_110


def test_tefas_client_history_does_not_daily_fanout_after_rate_limit(monkeypatch) -> None:
    client = fund_service.TefasClient()

    def fake_range(**kwargs):
        raise fund_service.TefasRateLimitError("TEFAS fund range HTTP 429")

    def fail_snapshot(**kwargs):
        raise AssertionError("rate-limited range should not fan out into daily snapshots")

    monkeypatch.setattr(client, "fetch_fund_range", fake_range)
    monkeypatch.setattr(client, "fetch_fund_list_snapshot", fail_snapshot)

    with pytest.raises(fund_service.TefasRateLimitError):
        client.fetch_fund_history(
            fund_codes=["TLY"],
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 25),
        )


def test_direct_tefas_adapter_fetches_all_supported_payload_families(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        def __init__(self, payload):
            self.content = json.dumps(payload).encode("utf-8")

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get(self, url, **kwargs):
            calls.append(("GET", url, kwargs))
            return FakeResponse({})

        def post(self, url, **kwargs):
            payload = kwargs["json"]
            calls.append(("POST", url, payload))
            if url.endswith("fonGnlBlgSiraliGetir"):
                body = {
                    "fonKodu": "TLY",
                    "fonUnvan": "TERA PORTFÖY TEST FONU",
                    "tarih": "2026-06-24",
                    "fiyat": 3.17,
                    "portfoyBuyukluk": 100.0,
                }
            elif url.endswith("fonGetiriBazliBilgiGetir"):
                body = {"fonKodu": "TLY", "getiri1h": "1,25", "getiriOrani": "1,25"}
            elif url.endswith("fonYonetimBazliBilgiGetir"):
                body = {"fonKodu": "TLY", "uygulananYu1Y": "2,10", "fonTopGiderKesoran": "2,40"}
            elif url.endswith("dagilimSiraliGetirT"):
                body = {"fonKodu": "TLY", "tarih": "2026-06-24", "hs": 42.0}
            else:
                raise AssertionError(f"unexpected endpoint: {url}")
            return FakeResponse({"resultList": [body], "toplamSayfa": 1})

    monkeypatch.setattr(fund_service.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(fund_service, "TEFAS_DIRECT_PAGE_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(fund_service, "TEFAS_DIRECT_CHUNK_DELAY_SECONDS", 0.0)

    client = fund_service.TefasClient()
    funds = client.fetch_funds_direct(
        start_date=date(2026, 6, 24),
        end_date=date(2026, 6, 24),
        fund_codes=["TLY"],
    )
    returns = client.fetch_returns_direct(
        basis="RB",
        fund_codes=["TLY"],
        start_date=date(2026, 6, 24),
        end_date=date(2026, 6, 24),
    )
    fees = client.fetch_management_fees_direct(
        fund_codes=["TLY"],
        as_of=date(2026, 6, 24),
        lookback_days=7,
    )
    portfolio = client.fetch_portfolio_direct(
        fund_code="TLY",
        start_date=date(2026, 6, 24),
        end_date=date(2026, 6, 24),
    )

    assert funds[0]["fonKodu"] == "TLY"
    assert funds[0]["source"] == fund_service.TEFAS_DIRECT_FUNDS_SOURCE
    assert returns[0]["getiriOrani"] == "1,25"
    assert fees[0]["uygulananYu1Y"] == "2,10"
    assert portfolio[0]["hs"] == 42.0
    posted_endpoints = [url for method, url, _payload in calls if method == "POST"]
    assert any(url.endswith("fonGnlBlgSiraliGetir") for url in posted_endpoints)
    assert any(url.endswith("fonGetiriBazliBilgiGetir") for url in posted_endpoints)
    assert any(url.endswith("fonYonetimBazliBilgiGetir") for url in posted_endpoints)
    assert any(url.endswith("dagilimSiraliGetirT") for url in posted_endpoints)


def _stub_direct_tefas_empty(monkeypatch) -> None:
    class FakeDirectTefasClient:
        def fetch_fund_history(self, **kwargs):
            return []

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeDirectTefasClient())


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


def test_latest_tefasfon_snapshot_daily_return_skips_turkey_market_holidays() -> None:
    class FakeTefasFonClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            super().__init__(fund_types=["SEC"])
            self.range_calls = []

        def fetch_funds(self, *, start_date, end_date, fund_codes=None):
            assert start_date == date(2026, 5, 20)
            assert end_date == date(2026, 5, 20)
            return [
                {
                    "fonKodu": "TLY",
                    "fonUnvan": "TERA PORTFOY TEST FONU",
                    "tarih": "2026-05-20",
                    "fiyat": 5516.308655,
                }
            ]

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
    assert payload["source_metadata"]["daily_return_local_fallback_count"] == 1


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
    assert payload["source_metadata"]["primary_source"] == "tefasfon"
    assert payload["source_metadata"]["fallback_used"] is False
    assert payload["source_metadata"]["fallback_reason"] is None
    assert payload["source_metadata"]["tefasfon_adapter_version"] == fund_service._tefasfon_adapter_version()


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
    assert payload["source_metadata"]["fallback_reason"] == "tefasfon_funds failed: blocked"
    assert "blocked" in " ".join(payload["source_metadata"]["warnings"])


def test_tefasfon_fetch_history_uses_anchor_strategy_for_long_ranges(monkeypatch) -> None:
    client = fund_service.TefasFonClient(fund_types=["SEC"])
    fetch_ranges = []
    snapshot_calls = []

    def fake_fetch_funds(*, start_date, end_date, fund_codes=None):
        fetch_ranges.append((start_date, end_date, tuple(fund_codes or [])))
        return [
            {"fonKodu": "TLY", "tarih": "2026-04-30", "fiyat": 5.0, "source": "tefasfon_funds"},
            {"fonKodu": "TLY", "tarih": "2026-05-18", "fiyat": 6.0, "source": "tefasfon_funds"},
            {"fonKodu": "ABC", "tarih": "2026-05-18", "fiyat": 7.0, "source": "tefasfon_funds"},
        ]

    def fake_snapshot(as_of):
        snapshot_calls.append(as_of)
        rows_by_date = {
            date(2026, 1, 30): {"fonKodu": "TLY", "tarih": "2026-01-30", "fiyat": 2.0, "source": "tefasfon_funds"},
            date(2026, 2, 27): {"fonKodu": "TLY", "tarih": "2026-02-27", "fiyat": 3.0, "source": "tefasfon_funds"},
            date(2026, 3, 31): {"fonKodu": "TLY", "tarih": "2026-03-31", "fiyat": 4.0, "source": "tefasfon_funds"},
        }
        row = rows_by_date.get(as_of)
        return [row] if row else []

    monkeypatch.setattr(client, "fetch_funds", fake_fetch_funds)
    monkeypatch.setattr(client, "fetch_daily_funds_snapshot", fake_snapshot)

    rows = client.fetch_history("TLY", date(2026, 1, 1), date(2026, 5, 18))

    assert fetch_ranges == [(date(2026, 4, 13), date(2026, 5, 18), ("TLY",))]
    assert snapshot_calls == [date(2026, 1, 30), date(2026, 2, 27), date(2026, 3, 31)]
    assert [row["tarih"] for row in rows] == [
        "2026-01-30",
        "2026-02-27",
        "2026-03-31",
        "2026-04-30",
        "2026-05-18",
    ]


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
        "2025-11-28",
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
    assert by_date["2026-04-30"]["source"] == "tefasfon_funds"
    assert payload["source_metadata"]["overview_metric_backfill"]["attempted"] is True
    assert payload["source_metadata"]["overview_metric_backfill"]["upserted_count"] == 7
    assert payload["source_metadata"]["backfill_used"] is True

    cached_payload = fund_service.get_fund_performance_payload(tmp_path, "TLY")

    assert len(calls) == 7
    assert cached_payload["source_metadata"]["cache_hit"] is True
    assert cached_payload["source_metadata"]["overview_metric_backfill"]["attempted"] is False


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
    assert by_date["2026-05-20"]["source"] == "tefasfon_funds"
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


def test_fund_tax_info_maps_known_fund_categories() -> None:
    assert fund_service._fund_tax_info("Hisse Senedi Şemsiye Fonu") == "%0"
    assert fund_service._fund_tax_info("Borsa Yatırım Fonu") == "%0"
    assert fund_service._fund_tax_info("Serbest Şemsiye Fonu") == "%10"
    assert fund_service._fund_tax_info("Para Piyasası Şemsiye Fonu") == "%10"
    assert fund_service._fund_tax_info("Bireysel Emeklilik Fonu") == "—"
    assert fund_service._fund_tax_info("") is None
    assert fund_service._fund_tax_info(None) is None


def test_coerce_tefas_percentage_handles_turkish_decimal_strings() -> None:
    assert fund_service._coerce_tefas_percentage("1,65") == pytest.approx(1.65)
    assert fund_service._coerce_tefas_percentage("2") == pytest.approx(2.0)
    assert fund_service._coerce_tefas_percentage("% 0,5") == pytest.approx(0.5)
    assert fund_service._coerce_tefas_percentage(0) == 0.0
    assert fund_service._coerce_tefas_percentage(None) is None
    assert fund_service._coerce_tefas_percentage("nan") is None


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
