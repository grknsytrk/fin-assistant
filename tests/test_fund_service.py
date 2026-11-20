from __future__ import annotations

import json
import threading
import time
from datetime import date

import pytest

from app import fund_service


def _stub_direct_tefas_empty(monkeypatch) -> None:
    class FakeDirectTefasClient:
        def fetch_fund_history(self, **kwargs):
            return []

    monkeypatch.setattr(fund_service, "TefasClient", lambda: FakeDirectTefasClient())


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


def test_tefas_retry_after_is_honored_with_a_separate_safety_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(fund_service, "TEFAS_HTTP_RETRY_AFTER_MAX_SECONDS", 300.0)
    monkeypatch.setattr(fund_service.random, "uniform", lambda *_args: 1.0)

    assert fund_service._tefas_retry_delay_seconds(1, 120.0) == 120.0
    assert fund_service._tefas_retry_delay_seconds(1, 999.0) == 300.0


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


def test_tefas_client_history_splits_at_calendar_month_boundaries(monkeypatch) -> None:
    client = fund_service.TefasClient()
    ranges = []

    def fake_range(*, fund_code, start_date, end_date):
        ranges.append((start_date, end_date))
        return [
            {
                "fund_code": fund_code,
                "date": start_date.isoformat(),
                "price": 1.0,
                "source": "tefasfon_funds",
            }
        ]

    monkeypatch.setattr(client, "fetch_fund_range", fake_range)
    monkeypatch.setattr(fund_service, "TEFAS_FUND_HISTORY_CHUNK_DAYS", 31)
    monkeypatch.setattr(fund_service, "TEFAS_DIRECT_CHUNK_DELAY_SECONDS", 0.0)

    rows = client.fetch_fund_history(
        fund_codes=["TLY"],
        start_date=date(2026, 2, 1),
        end_date=date(2026, 3, 31),
    )

    assert sorted(ranges) == [
        (date(2026, 2, 1), date(2026, 2, 28)),
        (date(2026, 3, 1), date(2026, 3, 31)),
    ]
    assert [row["date"] for row in rows] == ["2026-02-01", "2026-03-01"]


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


def test_tefasfon_snapshot_direct_probes_past_misreported_single_page(monkeypatch) -> None:
    from tefasfon import getter

    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.text = json.dumps(payload)

        def json(self):
            return json.loads(self.text)

    class FakeSession:
        def post(self, _url, *, json, headers, timeout):
            del headers, timeout
            calls.append(json)
            if json["basSira"] == 1:
                rows = [
                    {"fonKodu": f"F{index:04d}", "tarih": "2026-06-24", "fiyat": 1.0}
                    for index in range(1000)
                ]
            else:
                rows = [{"fonKodu": "F1000", "tarih": "2026-06-24", "fiyat": 1.0}]
            return FakeResponse({"resultList": rows, "toplamSayfa": 1})

    monkeypatch.setattr(getter, "_new_session", lambda *_args, **_kwargs: FakeSession())
    monkeypatch.setattr(getter, "_PAGE_SIZE", 1000)
    monkeypatch.setattr(getter, "_PAGE_DELAY", 0.0)

    rows = fund_service.TefasFonClient(fund_types=["SEC"])._fetch_funds_snapshot_direct(
        date(2026, 6, 24),
        fund_type="SEC",
    )

    assert len(rows) == 1001
    assert [payload["basSira"] for payload in calls] == [1, 1001]


def test_tefasfon_snapshot_direct_rejects_unverified_full_page(monkeypatch) -> None:
    from tefasfon import getter

    class FakeResponse:
        def __init__(self, payload):
            self.text = json.dumps(payload)

        def json(self):
            return json.loads(self.text)

    class FakeSession:
        def post(self, _url, *, json, headers, timeout):
            del headers, timeout
            if json["basSira"] == 1:
                rows = [
                    {"fonKodu": f"F{index:04d}", "tarih": "2026-06-24", "fiyat": 1.0}
                    for index in range(1000)
                ]
            else:
                rows = []
            return FakeResponse({"resultList": rows, "toplamSayfa": 1})

    monkeypatch.setattr(getter, "_new_session", lambda *_args, **_kwargs: FakeSession())
    monkeypatch.setattr(getter, "_PAGE_SIZE", 1000)
    monkeypatch.setattr(getter, "_PAGE_DELAY", 0.0)

    with pytest.raises(fund_service.TefasUpstreamError, match="full page"):
        fund_service.TefasFonClient(fund_types=["SEC"])._fetch_funds_snapshot_direct(
            date(2026, 6, 24),
            fund_type="SEC",
        )


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


def test_fintables_client_uses_curl_cffi_after_shell_curl_fails(monkeypatch) -> None:
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

    cffi_calls = []

    def fail_shell_curl(*_args, **_kwargs):
        raise fund_service.FintablesUpstreamError("shell curl unavailable")

    def fake_curl_cffi_payload(url, *, params, headers, timeout_seconds, context):
        cffi_calls.append((url, params, headers, timeout_seconds, context))
        return {"1w": {"prev_close_date": "2026-04-23T21:00:00Z", "prev_close": 10, "high": 12, "low": 9}}

    monkeypatch.setattr(fund_service.httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(fund_service, "_fintables_curl_payload", fail_shell_curl)
    monkeypatch.setattr(fund_service, "_fintables_curl_cffi_payload", fake_curl_cffi_payload)

    payload = fund_service.FintablesClient().fetch_yield_summary("TLY")

    assert payload["periods"]["1w"]["prev_close"] == 10
    assert cffi_calls
    assert cffi_calls[0][4] == "Fintables yield summary"


def test_fintables_client_uses_authenticated_proxy_after_local_fallbacks_fail(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, status_code, content, headers):
            self.status_code = status_code
            self.content = content
            self.headers = headers

    calls = []

    class FakeHttpxClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, *args, **kwargs):
            calls.append((url, kwargs))
            if len(calls) == 1:
                return FakeResponse(
                    403,
                    b"<html><title>Just a moment...</title>cloudflare</html>",
                    {"content-type": "text/html"},
                )
            return FakeResponse(
                200,
                b'{"1w":{"prev_close_date":"2026-04-23T21:00:00Z","prev_close":10,"high":12,"low":9}}',
                {"content-type": "application/json"},
            )

    def fail_fallback(*_args, **_kwargs):
        raise fund_service.FintablesUpstreamError("fallback unavailable")

    monkeypatch.setattr(fund_service.httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(fund_service, "_fintables_curl_payload", fail_fallback)
    monkeypatch.setattr(fund_service, "_fintables_curl_cffi_payload", fail_fallback)
    monkeypatch.setattr(fund_service, "FINTABLES_PROXY_BASE_URL", "https://fin-assistant.example.workers.dev")
    monkeypatch.setattr(fund_service, "FINTABLES_PROXY_TOKEN", "proxy-token")

    payload = fund_service.FintablesClient().fetch_yield_summary("TLY")

    assert payload["periods"]["1w"]["prev_close"] == 10
    assert calls[1][0] == "https://fin-assistant.example.workers.dev/internal/fintables/yield-summary"
    assert calls[1][1]["headers"]["Authorization"] == "Bearer proxy-token"
    assert calls[1][1]["params"] == {"code": "TLY"}


def test_tefasfon_fast_snapshot_skips_returns_and_fee_enrichment(monkeypatch) -> None:
    client = fund_service.TefasFonClient()
    calls = {"snapshots": 0, "returns": 0, "fees": 0}

    monkeypatch.setattr(
        client,
        "fetch_daily_funds_snapshot",
        lambda _as_of: calls.__setitem__("snapshots", calls["snapshots"] + 1) or [
            {
                "fund_code": "TLY",
                "name": "FAST FUND",
                "date": "2026-05-20",
                "price": 10.0,
                "tefasDurum": True,
            }
        ],
    )
    monkeypatch.setattr(
        client,
        "fetch_returns",
        lambda **_kwargs: calls.__setitem__("returns", calls["returns"] + 1) or pytest.fail("returns must not run"),
    )
    monkeypatch.setattr(
        client,
        "fetch_management_fees",
        lambda **_kwargs: calls.__setitem__("fees", calls["fees"] + 1) or pytest.fail("fees must not run"),
    )

    rows, warnings = client.fetch_latest_fund_list_snapshot(
        as_of=date(2026, 5, 20),
        lookback_days=3,
        enrich=False,
    )

    assert len(rows) == 1
    assert warnings == []
    assert calls == {"snapshots": 1, "returns": 0, "fees": 0}


def test_tefasfon_snapshot_resolution_finds_previous_date_within_fourteen_days(monkeypatch) -> None:
    client = fund_service.TefasFonClient()
    calls = []

    def fetch_daily_funds_snapshot(as_of):
        calls.append(as_of)
        if as_of == date(2026, 8, 20):
            return [{"fund_code": "TLY", "date": "2026-08-20", "price": 8.819, "tefasDurum": True}]
        return []

    monkeypatch.setattr(client, "fetch_daily_funds_snapshot", fetch_daily_funds_snapshot)
    result = client.fetch_latest_fund_list_snapshot_result(
        as_of=date(2026, 8, 21), lookback_days=40, enrich=False
    )

    assert result.resolution_status == "resolved_previous_date"
    assert result.resolved_as_of == "2026-08-20"
    assert calls == [date(2026, 8, 21), date(2026, 8, 20)]


def test_tefasfon_snapshot_resolution_distinguishes_empty_from_upstream_error(monkeypatch) -> None:
    client = fund_service.TefasFonClient()
    monkeypatch.setattr(client, "fetch_daily_funds_snapshot", lambda _as_of: [])
    empty = client.fetch_latest_fund_list_snapshot_result(
        as_of=date(2026, 8, 21), lookback_days=40, enrich=False
    )
    assert empty.resolution_status == "not_published"
    assert len(empty.attempted_dates) == 14

    def unavailable(**_kwargs):
        raise fund_service.TefasUpstreamError("429")

    monkeypatch.setattr(client, "fetch_daily_funds_snapshot", lambda _as_of: unavailable())
    failed = client.fetch_latest_fund_list_snapshot_result(
        as_of=date(2026, 8, 21), lookback_days=14, enrich=False
    )
    assert failed.resolution_status == "upstream_unavailable"
    assert len(failed.upstream_errors) == 3


def test_coerce_tefas_percentage_handles_turkish_decimal_strings() -> None:
    assert fund_service._coerce_tefas_percentage("1,65") == pytest.approx(1.65)
    assert fund_service._coerce_tefas_percentage("2") == pytest.approx(2.0)
    assert fund_service._coerce_tefas_percentage("% 0,5") == pytest.approx(0.5)
    assert fund_service._coerce_tefas_percentage(0) == 0.0
    assert fund_service._coerce_tefas_percentage(None) is None
    assert fund_service._coerce_tefas_percentage("nan") is None


def test_memory_cache_evicts_oldest_entries(monkeypatch) -> None:
    from app import fund_service

    fund_service.reset_fund_caches_for_tests()
    monkeypatch.setattr(fund_service, "_MEMORY_CACHE_MAX_ENTRIES", 4)

    for index in range(6):
        fund_service._memory_cache_put(f"key-{index}", {"value": index})

    assert len(fund_service._MEMORY_CACHE) == 4
    assert "key-0" not in fund_service._MEMORY_CACHE
    assert "key-1" not in fund_service._MEMORY_CACHE
    assert "key-2" in fund_service._MEMORY_CACHE
    assert "key-5" in fund_service._MEMORY_CACHE

    # Updating existing key does not evict extra entries
    fund_service._memory_cache_put("key-2", {"value": 999})
    assert len(fund_service._MEMORY_CACHE) == 4
    assert fund_service._MEMORY_CACHE["key-2"]["value"] == 999

    fund_service.reset_fund_caches_for_tests()


def test_tefas_client_fetch_fund_history_parallel_chunks_and_rate_limit_early_stop(
    monkeypatch,
) -> None:
    client = fund_service.TefasClient()
    lock = threading.Lock()
    active = 0
    peak = 0
    visited_months: list[int] = []

    def fake_range(*, fund_code, start_date, end_date):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            visited_months.append(start_date.month)
        try:
            time.sleep(0.03)
            if start_date.month == 3:
                raise fund_service.TefasUpstreamError("temporary month 3 glitch")
            return [
                {
                    "fund_code": fund_code,
                    "date": start_date.isoformat(),
                    "price": float(start_date.month),
                    "source": "tefasfon_funds",
                }
            ]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(client, "fetch_fund_range", fake_range)
    monkeypatch.setattr(fund_service, "TEFAS_FUND_HISTORY_CHUNK_DAYS", 31)
    monkeypatch.setattr(fund_service, "TEFAS_HISTORY_MAX_WORKERS", 3)

    rows = client.fetch_fund_history(
        fund_codes=["TLY"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 5, 31),
    )

    assert 2 <= peak <= 3
    assert sorted(visited_months) == [1, 2, 3, 4, 5]
    assert [row["date"] for row in rows] == [
        "2026-01-01",
        "2026-02-01",
        "2026-04-01",
        "2026-05-01",
    ]

    # Rate limit aborts queued months early instead of hammering all remaining chunks.
    rate_calls: list[int] = []

    def fake_rate_limited(*, fund_code, start_date, end_date):
        with lock:
            rate_calls.append(start_date.month)
        if start_date.month <= 3:
            raise fund_service.TefasRateLimitError("TEFAS 429")
        time.sleep(0.05)
        return []

    monkeypatch.setattr(client, "fetch_fund_range", fake_rate_limited)
    monkeypatch.setattr(fund_service, "TEFAS_HISTORY_MAX_WORKERS", 2)

    with pytest.raises(fund_service.TefasRateLimitError):
        client.fetch_fund_history(
            fund_codes=["TLY"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 8, 31),
        )
    assert len(rate_calls) < 8


def test_tefasfon_enrich_runs_four_upstream_calls_in_parallel() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    class ParallelFakeClient(fund_service.TefasFonClient):
        def __init__(self) -> None:
            pass

        def _enter(self) -> None:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)

        def _leave(self) -> None:
            nonlocal active
            with lock:
                active -= 1

        def fetch_daily_funds_snapshot(self, as_of):
            return [
                {
                    "fonKodu": "TLY",
                    "fonUnvan": "TERA TEST FONU",
                    "tarih": as_of.isoformat(),
                    "fiyat": 100.0,
                    "tefasDurum": True,
                }
            ]

        def fetch_returns(self, *, fund_codes=None, start_date=None, end_date=None):
            self._enter()
            try:
                time.sleep(0.03)
                if start_date is None and end_date is None:
                    return [{"fonKodu": "TLY", "getiri1a": 12.5}]
                if (end_date - start_date).days <= 3:
                    return [{"fonKodu": "TLY", "getiriOrani": 0.45}]
                return [{"fonKodu": "TLY", "getiriOrani": 2.1}]
            finally:
                self._leave()

        def fetch_management_fees(self, *, fund_codes=None, lookback_days=21, as_of=None):
            self._enter()
            try:
                time.sleep(0.03)
                raise fund_service.TefasUpstreamError("fees timeout")
            finally:
                self._leave()

    result = ParallelFakeClient().fetch_latest_fund_list_snapshot_result(
        as_of=date(2026, 5, 20),
        lookback_days=1,
        enrich=True,
    )

    assert peak >= 2
    assert len(result.rows) == 1
    assert result.rows[0]["getiri1a"] == pytest.approx(12.5)
    assert result.rows[0]["returns_source"] == "tefasfon_returns"
    assert result.rows[0]["daily_return"] == pytest.approx(0.45)
    assert result.rows[0]["return_1w"] == pytest.approx(2.1)
    assert any("tefasfon_management_fees failed" in msg for msg in result.warnings)
