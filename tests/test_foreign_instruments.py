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
