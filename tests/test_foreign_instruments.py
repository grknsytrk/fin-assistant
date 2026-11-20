from __future__ import annotations

import json

import httpx
import pytest

from app import foreign_instruments
from app import fund_service


def _row(isin: str, **fields) -> dict:
    return {"asset_type": "foreign_equity", "asset_code": isin, "asset_name": isin,
            "isin": isin, "provider_symbol": isin, "logo_symbol": isin, **fields}


def _mapping(ticker: str, name: str, exchange: str = "US") -> dict:
    return {"marketSector": "Equity", "ticker": ticker, "name": name, "exchCode": exchange}


def _mock_client(monkeypatch, handler) -> None:
    original = httpx.Client
    monkeypatch.setattr(foreign_instruments.httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs,
    ))


def test_identifier_mapping_batches_unique_isins_and_reuses_cache(monkeypatch) -> None:
    instruments = {
        "US5949181045": ("MSFT", "MICROSOFT CORP"),
        "US0079031078": ("AMD", "ADVANCED MICRO DEVICES"),
        "US67066G1040": ("NVDA", "NVIDIA CORP"),
        "US02079K3059": ("GOOGL", "ALPHABET INC-CL A"),
        "US30303M1027": ("META", "META PLATFORMS INC"),
        "US0382221051": ("AMAT", "APPLIED MATERIALS INC"),
    }
    requests = []

    def handler(request):
        jobs = json.loads(request.content)
        requests.append(jobs)
        return httpx.Response(200, json=[{"data": [_mapping(*instruments[job["idValue"]])]} for job in jobs])

    _mock_client(monkeypatch, handler)
    cache, writes = {}, []

    def save(key, value, ttl_seconds):
        cache[key] = value
        writes.append(ttl_seconds)

    original = [_row(isin) for isin in instruments] + [_row("US5949181045")]
    resolved = foreign_instruments.resolve_foreign_holdings(original, cache_get=cache.get, cache_set=save)
    assert [len(batch) for batch in requests] == [5, 1]
    assert [row["provider_symbol"] for row in resolved] == ["MSFT", "AMD", "NVDA", "GOOGL", "META", "AMAT", "MSFT"]
    assert resolved[0]["asset_name"] == "MICROSOFT CORP"
    assert resolved[1]["logo_symbol"] == "AMD"
    assert original[0]["provider_symbol"] == "US5949181045"
    assert writes == [30 * 24 * 60 * 60] * 6
    assert foreign_instruments.resolve_foreign_holdings(original, cache_get=cache.get, cache_set=save) == resolved
    assert len(requests) == 2


def test_mapping_prefers_home_listing_and_rejects_ambiguous_or_non_equity_results() -> None:
    identity = foreign_instruments._identity_from_mapping({"data": [
        _mapping("MSFT", "MICROSOFT CORP"),
        _mapping("MSF", "MICROSOFT CORP", "GY"),
    ]}, "US5949181045")
    assert identity["provider_symbol"] == "MSFT"
    assert foreign_instruments._identity_from_mapping({"data": [
        _mapping("ONE", "ONE"), _mapping("TWO", "TWO"),
    ]}, "US5949181045") == {}
    assert foreign_instruments._identity_from_mapping({"data": [
        {**_mapping("MSFT", "MICROSOFT CORP"), "marketSector": "Corp"},
    ]}, "US5949181045") == {}


def test_mapping_outage_clears_invalid_quote_symbols_and_uses_short_negative_cache(monkeypatch) -> None:
    from app.api import _foreign_holding_provider_symbol

    _mock_client(monkeypatch, lambda _: httpx.Response(429, json={"error": "rate limit"}))
    writes = []
    resolved = foreign_instruments.resolve_foreign_holdings(
        [_row("US5949181045")], cache_get=lambda _: None,
        cache_set=lambda key, value, ttl_seconds: writes.append((value, ttl_seconds)),
    )
    assert resolved[0]["provider_symbol"] is None
    assert _foreign_holding_provider_symbol(resolved[0]) == ""
    assert writes == [({}, 60)]


def test_valid_tickers_skip_mapping_and_foreign_month_comparison_uses_isin(monkeypatch) -> None:
    def unexpected_request(_):
        pytest.fail("Existing foreign tickers must not need an identifier lookup")

    _mock_client(monkeypatch, unexpected_request)
    current = _row("US5949181045", asset_code="MSFT", provider_symbol="MSFT", asset_name="Microsoft")
    previous = _row("US5949181045")
    assert foreign_instruments.resolve_foreign_holdings(
        [current], cache_get=lambda _: None, cache_set=lambda *args, **kwargs: None,
    ) == [current]
    assert fund_service._position_key(current) == fund_service._position_key(previous)


def test_mapping_uses_report_isin_when_the_original_code_is_another_identifier() -> None:
    cached = {"provider_symbol": "GOOGL", "name": "ALPHABET INC-CL A", "source": "openfigi"}
    keys = []

    def read(key):
        keys.append(key)
        return cached

    resolved = foreign_instruments.resolve_foreign_holdings(
        [_row("US02079K3059", asset_code="US21873S1087", provider_symbol="US21873S1087")],
        cache_get=read, cache_set=lambda *args, **kwargs: None,
    )
    assert keys == ["api:funds:foreign-identity:US02079K3059:v1"]
    assert resolved[0]["provider_symbol"] == "GOOGL"


def test_fund_api_service_uses_bulk_redis_keys_for_foreign_quotes_and_identities(monkeypatch) -> None:
    from app import cache as cache_module
    from app import fund_api_service

    cache_module.reset_cache_for_tests()
    fund_api_service._FOREIGN_HOLDING_QUOTE_CACHE.clear()
    fund_api_service._FOREIGN_IDENTITIES_L1_CACHE.clear()
    monkeypatch.setattr(
        fund_api_service,
        "_fetch_yahoo_quote",
        lambda sym: {"ok": True, "price": 100.0, "currency": "USD", "change_pct": 1.25},
    )

    positions = [
        _row("US5949181045", asset_code="MSFT", provider_symbol="MSFT"),
        _row("US0079031078", asset_code="AMD", provider_symbol="AMD"),
    ]
    quotes = fund_api_service._quote_map_for_foreign_holdings(positions)
    assert sorted(quotes) == ["AMD", "MSFT"]

    backend = cache_module.get_cache()
    bulk_quotes = backend.get(fund_api_service._FOREIGN_QUOTES_BULK_KEY)
    assert isinstance(bulk_quotes, dict)
    assert set(bulk_quotes) == {"AMD", "MSFT"}
    # No per-symbol quote keys should be written when bulk cache is active.
    assert backend.get("api:funds:foreign-holding-quote:MSFT:v4") is None
    assert backend.get("api:funds:foreign-holding-quote:AMD:v4") is None

    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json=[{"data": [_mapping("MSFT", "MICROSOFT CORP")]}],
        ),
    )
    enriched = fund_api_service._enrich_fund_holdings_with_daily_market_data(
        {"fund_code": "TLY", "positions": [_row("US5949181045")]},
        "TLY",
    )
    assert enriched["positions"][0]["provider_symbol"] == "MSFT"
    bulk_identities = backend.get(fund_api_service._FOREIGN_IDENTITIES_BULK_KEY)
    assert isinstance(bulk_identities, dict)
    assert "api:funds:foreign-identity:US5949181045:v1" in bulk_identities
    assert backend.get("api:funds:foreign-identity:US5949181045:v1") is None
    cache_module.reset_cache_for_tests()


def test_openfigi_client_is_pooled_reusable_and_closed_on_teardown(monkeypatch) -> None:
    foreign_instruments.close_openfigi_client()
    try:
        client1, ephemeral1 = foreign_instruments._get_openfigi_client(headers={"Content-Type": "application/json"})
        client2, ephemeral2 = foreign_instruments._get_openfigi_client(headers={"Content-Type": "application/json"})
        assert client1 is client2
        assert ephemeral1 is False
        assert ephemeral2 is False
        assert isinstance(client1, httpx.Client)
        assert not client1.is_closed

        # close_shared_http_clients delegates teardown to OpenFIGI pool cleanly.
        fund_service.close_shared_http_clients()
        assert client1.is_closed
        assert foreign_instruments._OPENFIGI_CLIENT is None

        # Monkeypatched httpx.Client always creates fresh ephemeral instances.
        _mock_client(monkeypatch, lambda _req: httpx.Response(200, json=[]))
        mock1, mock_ephem1 = foreign_instruments._get_openfigi_client(headers={"Content-Type": "application/json"})
        mock2, mock_ephem2 = foreign_instruments._get_openfigi_client(headers={"Content-Type": "application/json"})
        assert mock1 is not mock2
        assert mock_ephem1 is True
        assert mock_ephem2 is True
        assert foreign_instruments._OPENFIGI_CLIENT is None
        mock1.close()
        mock2.close()
    finally:
        foreign_instruments.close_openfigi_client()


def test_openfigi_retries_once_on_remote_protocol_error(monkeypatch) -> None:
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError("stale keepalive socket")
        return httpx.Response(200, json=[{"data": [_mapping("MSFT", "MICROSOFT CORP")]}])

    _mock_client(monkeypatch, handler)
    resolved = foreign_instruments.resolve_foreign_holdings(
        [_row("US5949181045")],
        cache_get=lambda _key: None,
        cache_set=lambda *args, **kwargs: None,
    )
    assert attempts == 2
    assert resolved[0]["provider_symbol"] == "MSFT"
    assert resolved[0]["asset_name"] == "MICROSOFT CORP"
