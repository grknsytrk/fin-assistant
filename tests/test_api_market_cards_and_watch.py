from __future__ import annotations

import json
import threading
from datetime import date, datetime, timezone
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api_module_facade import api_module
from app import fund_service as fund_service_module
from app.api import app
from api_test_helpers import (
    _commodity_payload_for_watch,
    _fx_payload_for_watch,
    _reset_flow_state,
    _watch_quote,
)


def test_market_stock_cards_returns_quotes_and_line_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols: {
            symbol: {
                "price": 760.0 if symbol == "BIMAS" else 325.0,
                "currency": "TRY",
                "change": -3.0 if symbol == "BIMAS" else 1.5,
                "change_pct": -0.39 if symbol == "BIMAS" else 0.46,
                "volume": 2_813_888_143.0 if symbol == "BIMAS" else 12_228_796_146.0,
                "market_state": "REGULAR",
                "as_of": "2026-04-25T11:00:00+00:00",
            }
            for symbol in symbols
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "BIMAS": {"market_cap": 456_000_000_000.0},
            "THYAO": {"market_cap": 448_500_000_000.0},
        },
    )
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_multiples",
        lambda symbol: {
            "ok": True,
            "fk": 24.47 if symbol == "BIMAS" else 3.79,
            "pd_dd": 2.75 if symbol == "BIMAS" else 0.49,
            "fd_favok": 11.37 if symbol == "BIMAS" else 5.97,
        },
    )
    monkeypatch.setattr(api_module, "_stock_card_financial_snapshot_from_cache", lambda _symbol: {})
    monkeypatch.setattr(
        api_module,
        "_stock_card_financial_ratios_from_cache",
        lambda symbol: {"net_borc_favok": 0.89 if symbol == "BIMAS" else 3.13},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_return_bases_bulk",
        lambda _symbols: {
            "BIMAS": {
                "base_1w": 765.0,
                "base_1m": 680.0,
                "base_3m": 628.5,
                "base_6m": 574.5,
                "base_ytd": 537.0,
                "base_1y": 452.0,
            },
            "THYAO": {},
        },
    )

    def fake_intraday(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
        return {
            "line_points": [
                {"time": "2026-04-25T08:00:00+00:00", "close": 750.0},
                {"time": "2026-04-25T08:05:00+00:00", "close": 760.0},
            ],
            "high": 761.5,
            "low": 747.0,
            "prev_close": 763.0,
            "volume": 3_729_933,
            "volume_lot": 3_729_933,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-25T11:00:00+00:00",
            "yahoo_symbol": f"{symbol}.IS",
            "error": None,
            "force_refresh_seen": force_refresh,
        }

    monkeypatch.setattr(api_module, "_fetch_stock_card_intraday", fake_intraday)
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"stock cards should not resolve logos over KAP: {symbol}"),
    )
    monkeypatch.setattr(api_module, "get_instruments", lambda *_args, **_kwargs: {})

    client = TestClient(app)
    response = client.get("/market/stocks/cards?symbols=BIMAS,THYAO&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "infoyatirim_yahoo_chart"
    assert [item["symbol"] for item in payload["items"]] == ["BIMAS", "THYAO"]
    assert payload["items"][0]["price"] == 760.0
    assert payload["items"][0]["change_pct"] == -0.39
    assert payload["items"][0]["volume"] == 2_813_888_143.0
    assert payload["items"][0]["volume_tl"] == 2_813_888_143.0
    assert payload["items"][0]["volume_lot"] == 3_729_933
    assert payload["items"][0]["market_cap"] == 456_000_000_000.0
    assert payload["items"][0]["logo_url"] is None
    assert payload["items"][0]["logo_source"] is None
    assert payload["items"][0]["previous_close"] == 763.0
    assert payload["items"][0]["fk"] == 24.47
    assert payload["items"][0]["pd_dd"] == 2.75
    assert payload["items"][0]["fd_favok"] == 11.37
    assert payload["items"][0]["net_borc_favok"] == 0.89
    assert payload["items"][0]["return_1w_pct"] == -0.65
    assert payload["items"][0]["return_1m_pct"] == 11.76
    assert payload["items"][0]["line_points"][1]["close"] == 760.0


def test_full_stock_cards_fetch_charts_concurrently_with_bounded_workers_and_stable_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbols = [f"S{index:02d}" for index in range(12)]
    barrier = threading.Barrier(6)
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    calls = []
    refresh_values = []

    monkeypatch.setattr(api_module, "_shared_cache_get_dict", lambda _key: None)
    monkeypatch.setattr(api_module, "_shared_cache_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda requested: {symbol: {} for symbol in requested})
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", lambda _requested: {})
    monkeypatch.setattr(api_module, "get_instruments", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api_module, "_load_cached_kap_market_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api_module, "_resolve_market_card_valuation", lambda *_args, **_kwargs: {})

    def fake_intraday(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(symbol)
            refresh_values.append(force_refresh)
        try:
            barrier.wait(timeout=2)
        finally:
            with active_lock:
                active -= 1
        if symbol == "S03":
            return {"line_points": [], "error": "chart_unavailable", "yahoo_symbol": f"{symbol}.IS"}
        return {
            "line_points": [{"time": "2026-10-31T10:00:00+00:00", "close": 100.0}],
            "price": 100.0,
            "error": None,
            "yahoo_symbol": f"{symbol}.IS",
            "session_status": "closed",
        }

    monkeypatch.setattr(api_module, "_fetch_stock_card_intraday", fake_intraday)

    payload = api_module._market_stock_cards_payload(
        symbols=",".join(symbols),
        force_refresh=True,
        tier="full",
    )

    assert [item["symbol"] for item in payload["items"]] == symbols
    assert sorted(calls) == symbols
    assert len(calls) == len(symbols)
    assert set(refresh_values) == {True}
    assert 2 <= max_active <= 6
    assert payload["items"][3]["error"] == "chart_unavailable"
    assert payload["items"][4]["error"] is None


def test_market_stock_cards_quick_returns_quote_before_detail_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols: {
            "BIMAS": {
                "price": 760.0,
                "currency": "TRY",
                "change": -3.0,
                "change_pct": -0.39,
                "volume": 2_813_888_143.0,
                "market_state": "REGULAR",
                "as_of": "2026-04-25T11:00:00+00:00",
            }
        },
    )
    monkeypatch.setattr(
        api_module,
        "get_instruments",
        lambda *_args, **_kwargs: {"BIMAS": {"name": "BIM BIRLESIK MAGAZALAR"}},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: pytest.fail("quick stock cards should not fetch basic summaries"),
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_return_bases_bulk",
        lambda _symbols: pytest.fail("quick stock cards should not fetch return history"),
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_card_intraday",
        lambda *_args, **_kwargs: pytest.fail("quick stock cards should not fetch intraday data"),
    )
    monkeypatch.setattr(
        api_module,
        "_resolve_market_card_valuation",
        lambda *_args, **_kwargs: pytest.fail("quick stock cards should not fetch valuation data"),
    )

    client = TestClient(app)
    response = client.get("/market/stocks/cards?symbols=BIMAS&tier=quick&refresh=true")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["card_stage"] == "quick"
    assert item["price"] == 760.0
    assert item["change_pct"] == -0.39
    assert item["volume_tl"] == 2_813_888_143.0
    assert item["line_points"] == []
    assert item["fk"] is None


def test_stock_return_bases_keep_history_before_one_year_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: List[str] = []
    timestamps = [
        datetime(2025, 9, 3, tzinfo=timezone.utc).timestamp(),
        datetime(2025, 9, 8, tzinfo=timezone.utc).timestamp(),
        datetime(2026, 9, 4, tzinfo=timezone.utc).timestamp(),
    ]
    chart_payload = {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "close": [100.0, 105.0, 150.0],
                                "high": [101.0, 106.0, 151.0],
                                "low": [99.0, 104.0, 149.0],
                            }
                        ]
                    },
                }
            ]
        }
    }

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(chart_payload).encode("utf-8")

    def fake_urlopen(request: Any, **_kwargs: Any) -> FakeResponse:
        requested_urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    bases = api_module._fetch_stock_return_bases("BIMAS")
    returns = api_module._returns_from_bases(150.0, bases)

    assert requested_urls == [
        "https://query1.finance.yahoo.com/v8/finance/chart/BIMAS.IS?interval=1d&range=2y"
    ]
    assert bases["base_1y"] == 100.0
    assert returns["return_1y_pct"] == 50.0


def test_market_stock_cards_falls_back_to_infoyatirim_multiples_when_missing_or_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols: {
            symbol: {
                "price": 207.0 if symbol == "KCHOL" else 760.0,
                "currency": "TRY",
                "change": 2.1 if symbol == "KCHOL" else -3.0,
                "change_pct": 1.02 if symbol == "KCHOL" else -0.39,
                "volume": 3_308_601_000.0 if symbol == "KCHOL" else 2_813_888_143.0,
                "market_state": "REGULAR",
                "as_of": "2026-04-25T11:00:00+00:00",
            }
            for symbol in symbols
        },
    )
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )

    def fake_multiples(symbol: str) -> Dict[str, Any]:
        if symbol == "KCHOL":
            return {"ok": True, "fk": 0.0, "pd_dd": None, "fd_favok": 0.0}
        return {"ok": True, "fk": 24.47, "pd_dd": 2.75, "fd_favok": 11.37}

    monkeypatch.setattr(api_module, "_fetch_isyatirim_multiples", fake_multiples)
    monkeypatch.setattr(api_module, "_stock_card_financial_snapshot_from_cache", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_stock_card_financial_ratios_from_cache", lambda _symbol: {"net_borc_favok": None})
    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", lambda _symbols: {})
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_card_intraday",
        lambda symbol, force_refresh=False: {
            "line_points": [],
            "high": None,
            "low": None,
            "prev_close": None,
            "volume": None,
            "volume_lot": None,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-25T11:00:00+00:00",
            "yahoo_symbol": f"{symbol}.IS",
            "error": None,
        },
    )

    fallback_calls: List[str] = []

    def fake_infoyatirim_quote(symbol: str) -> Dict[str, Any]:
        fallback_calls.append(symbol)
        if symbol == "KCHOL":
            return {"fk": 23.86, "pd_dd": 0.77, "fd_favok": 11.2}
        return {}

    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", fake_infoyatirim_quote)

    client = TestClient(app)
    response = client.get("/market/stocks/cards?symbols=KCHOL,BIMAS&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    items = {item["symbol"]: item for item in payload["items"]}
    assert items["KCHOL"]["fk"] == 23.86
    assert items["KCHOL"]["pd_dd"] == 0.77
    assert items["KCHOL"]["fd_favok"] == 11.2
    assert items["BIMAS"]["fk"] == 24.47
    assert items["BIMAS"]["pd_dd"] == 2.75
    assert items["BIMAS"]["fd_favok"] == 11.37
    assert fallback_calls == ["KCHOL"]


def test_market_stock_card_valuation_prefers_kap_snapshot_and_reprices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_stock_card_financial_snapshot_from_cache",
        lambda _symbol: {
            "latest_quarter": "2026Q1",
            "ttm_net_kar": 10.0,
            "ozkaynaklar": 20.0,
            "ttm_favok": 5.0,
            "net_borc": 2.0,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_multiples",
        lambda symbol: pytest.fail(f"external multiples should not be fetched for complete KAP snapshot: {symbol}"),
    )
    monkeypatch.setattr(api_module, "_stock_card_financial_ratios_from_cache", lambda _symbol: {"net_borc_favok": 0.4})

    first = api_module._resolve_market_card_valuation("BIMAS", market_cap=100.0)
    second = api_module._resolve_market_card_valuation("BIMAS", market_cap=120.0)

    assert first["fk"] == 10.0
    assert first["pd_dd"] == 5.0
    assert first["fd_favok"] == 20.4
    assert first["net_borc_favok"] == 0.4
    assert first["valuation_source"] == "kap_computed"
    assert second["fk"] == 12.0
    assert second["pd_dd"] == 6.0
    assert second["fd_favok"] == 24.4
    assert second["cache_hit"] is True


def test_market_stock_card_valuation_caches_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    monkeypatch.setattr(api_module, "_stock_card_financial_snapshot_from_cache", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_stock_card_financial_ratios_from_cache", lambda _symbol: {"net_borc_favok": None})

    def fake_multiples(symbol: str) -> Dict[str, Any]:
        calls.append(symbol)
        return {"ok": True, "source": "isyatirim_company_card", "fk": 11.0, "pd_dd": 2.0, "fd_favok": 8.0}

    monkeypatch.setattr(api_module, "_fetch_isyatirim_multiples", fake_multiples)
    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", lambda symbol: pytest.fail(symbol))

    first = api_module._resolve_market_card_valuation("BIMAS", market_cap=100.0)
    second = api_module._resolve_market_card_valuation("BIMAS", market_cap=120.0)

    assert first["fk"] == 11.0
    assert second["fk"] == 11.0
    assert second["cache_hit"] is True
    assert calls == ["BIMAS"]


def test_market_stock_cards_rejects_more_than_twelve_symbols() -> None:
    client = TestClient(app)
    symbols = ",".join(f"SYM{i}" for i in range(13))

    response = client.get(f"/market/stocks/cards?symbols={symbols}")

    assert response.status_code == 400
    assert "En fazla 12" in response.json()["detail"]


def test_market_stock_card_chart_maps_range_and_normalizes_points(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chart(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        assert yahoo_symbol == "BIMAS.IS"
        assert interval == "15m"
        assert range_ == "5d"
        return {
            "ok": True,
            "meta": {},
            "points": [
                {"time": "2026-04-25T08:30:00+00:00", "close": 752.0},
                {"time": "2026-04-25T08:00:00+00:00", "close": 750.0},
                {"time": "2026-04-25T08:15:00+00:00", "close": None},
                {"time": "2026-04-25T08:30:00+00:00", "close": 753.0, "high": 754.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    client = TestClient(app)
    response = client.get("/market/stocks/cards/chart?symbol=bimas.IS&range=1w&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BIMAS"
    assert payload["range"] == "1w"
    assert payload["yahoo_symbol"] == "BIMAS.IS"
    assert payload["source"] == "yahoo_live"
    assert [point["time"] for point in payload["line_points"]] == [
        "2026-04-25T08:00:00+00:00",
        "2026-04-25T08:30:00+00:00",
    ]
    assert payload["line_points"][1]["close"] == 753.0


@pytest.mark.parametrize(
    ("chart_range", "expected_interval", "expected_yahoo_range"),
    [("1m", "4h", "1mo"), ("ytd", "1d", "ytd"), ("1y", "1d", "1y"), ("5y", "1wk", "5y")],
)
def test_market_stock_card_chart_uses_requested_long_range_granularity(
    monkeypatch: pytest.MonkeyPatch,
    chart_range: str,
    expected_interval: str,
    expected_yahoo_range: str,
) -> None:
    calls: List[str] = []

    def fake_chart(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        calls.append(f"{yahoo_symbol}:{interval}:{range_}")
        return {
            "ok": True,
            "meta": {},
            "points": [{"time": "2026-04-25T08:00:00+00:00", "close": 750.0}],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    client = TestClient(app)
    response = client.get(f"/market/stocks/cards/chart?symbol=BIMAS&range={chart_range}&refresh=true")

    assert response.status_code == 200
    assert response.json()["range"] == chart_range
    assert calls == [f"BIMAS.IS:{expected_interval}:{expected_yahoo_range}"]


def test_market_stock_card_chart_uses_normalized_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_chart(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        calls.append(f"{yahoo_symbol}:{interval}:{range_}")
        return {
            "ok": True,
            "meta": {},
            "points": [{"time": "2026-04-25T08:00:00+00:00", "close": 750.0}],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    client = TestClient(app)
    first = client.get("/market/stocks/cards/chart?symbol=BIMAS&range=1d")
    second = client.get("/market/stocks/cards/chart?symbol=bimas.IS&range=1d")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == ["BIMAS.IS:5m:1d"]
    assert first.json()["source"] == "yahoo_live"
    assert second.json()["source"] == "yahoo_cache"
    assert first.json()["as_of"] == second.json()["as_of"]


def test_market_stock_card_chart_falls_back_to_previous_session_when_daily_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    period_calls: List[date] = []

    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_raw",
        lambda *_args, **_kwargs: {"ok": True, "meta": {}, "points": []},
    )

    def fake_period_chart(
        yahoo_symbol: str,
        *,
        interval: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        assert yahoo_symbol == "BIMAS.IS"
        assert interval == "5m"
        assert start_date == end_date
        assert start_date.weekday() < 5
        period_calls.append(start_date)
        return {
            "ok": True,
            "meta": {"chartPreviousClose": 100.0},
            "points": [
                {"time": f"{start_date.isoformat()}T07:00:00+00:00", "close": 101.0, "high": 102.0, "low": 99.0},
                {"time": f"{start_date.isoformat()}T07:05:00+00:00", "close": 103.0, "high": 104.0, "low": 100.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_period_chart)

    client = TestClient(app)
    response = client.get("/market/stocks/cards/chart?symbol=BIMAS&range=1d&refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "yahoo_previous_session"
    assert payload["error"] is None
    assert len(payload["line_points"]) == 2
    assert payload["line_points"][-1]["close"] == 103.0
    assert period_calls


def test_stock_card_session_state_marks_old_intraday_points_as_closed() -> None:
    state = api_module._stock_card_session_state(
        [{"time": "2026-05-26T09:40:00+00:00", "close": 373.0}],
        market_state="REGULAR",
        source="yahoo_live",
        now=datetime(2026, 5, 28, 15, 0, tzinfo=timezone.utc),
    )

    assert state["session_status"] == "previous_session"
    assert state["session_label"] == "Piyasa kapalı"
    assert state["is_live"] is False
    assert state["is_stale"] is True
    assert state["last_trade_at"] == "2026-05-26T09:40:00+00:00"
    assert state["last_trade_date"] == "2026-05-26"


def test_market_stock_card_chart_rejects_invalid_symbol_and_range() -> None:
    client = TestClient(app)

    invalid_symbol = client.get("/market/stocks/cards/chart?symbol=!!!&range=1d")
    invalid_range = client.get("/market/stocks/cards/chart?symbol=BIMAS&range=2y")

    assert invalid_symbol.status_code == 400
    assert invalid_range.status_code == 400


def test_market_stock_card_chart_returns_error_for_no_data_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_raw",
        lambda *_args, **_kwargs: {"ok": False, "error": "not_found"},
    )

    client = TestClient(app)
    response = client.get("/market/stocks/cards/chart?symbol=NOTREAL&range=1m")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "NOTREAL"
    assert payload["line_points"] == []
    assert payload["error"] == "not_found"


def test_market_comparison_history_returns_mixed_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fund_history(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        assert fund_code == "TLY"
        assert kwargs["start_date"].isoformat() == "2026-04-01"
        assert kwargs["end_date"].isoformat() == "2026-04-05"
        return {
            "status": "ok",
            "source": "sqlite",
            "points": [
                {"date": "2026-03-31", "price": 9.8},
                {"date": "2026-04-01", "price": 10.0},
                {"date": "2026-04-05", "price": 10.5},
            ],
            "source_metadata": {},
        }

    def fake_chart(yahoo_symbol: str, *, interval: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
        assert interval == "1d"
        assert start_date.isoformat() == "2026-04-01"
        assert end_date.isoformat() == "2026-04-05"
        values = {
            "BIMAS.IS": 750.0,
            "XU100.IS": 9500.0,
            "USDTRY=X": 32.0,
        }
        base = values[yahoo_symbol]
        return {
            "ok": True,
            "points": [
                {"time": "2026-04-01T08:00:00+00:00", "close": base},
                {"time": "2026-04-05T08:00:00+00:00", "close": base + 5.0},
            ],
        }

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_fund_history)
    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_chart)

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-05",
            "assets": [
                {"id": "fund:TLY", "kind": "fund", "symbol": "TLY", "label": "TLY"},
                {"id": "stock:BIMAS", "kind": "stock", "symbol": "BIMAS", "label": "BIMAS"},
                {"id": "index:XU100", "kind": "index", "symbol": "XU100", "label": "BIST 100"},
                {"id": "fx:USD/TRY", "kind": "fx", "symbol": "USD/TRY", "label": "Dolar"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["start_date"] == "2026-04-01"
    assert [asset["id"] for asset in payload["assets"]] == ["fund:TLY", "stock:BIMAS", "index:XU100", "fx:USD/TRY"]
    assert payload["assets"][0]["points"] == [
        {"date": "2026-04-01", "value": 10.0},
        {"date": "2026-04-05", "value": 10.5},
    ]
    assert payload["assets"][1]["points"][0] == {"date": "2026-04-01", "value": 750.0}
    assert all(asset["error"] is None for asset in payload["assets"])


def test_market_comparison_history_rejects_reversed_dates() -> None:
    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-05",
            "end_date": "2026-04-01",
            "assets": [{"kind": "index", "symbol": "XU100"}],
        },
    )

    assert response.status_code == 400
    assert "start_date" in response.json()["detail"]


def test_market_comparison_history_returns_partial_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fund_history(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "ok",
            "source": "sqlite",
            "points": [{"date": "2026-04-01", "price": 10.0}],
            "source_metadata": {},
        }

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_fund_history)
    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_period_raw",
        lambda *_args, **_kwargs: {"ok": False, "error": "not_found", "points": []},
    )

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-05",
            "assets": [
                {"id": "fund:TLY", "kind": "fund", "symbol": "TLY"},
                {"id": "stock:NOTREAL", "kind": "stock", "symbol": "NOTREAL"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"][0]["points"] == [{"date": "2026-04-01", "value": 10.0}]
    assert payload["assets"][0]["error"] is None
    assert payload["assets"][1]["points"] == []
    assert payload["assets"][1]["error"] == "not_found"


def test_market_comparison_history_fund_skips_overview_metrics_and_serves_local_db_when_swr_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_fund_history(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append({"fund_code": fund_code, **kwargs})
        return {
            "status": "ok",
            "source": "sqlite",
            "points": [
                {"date": "2025-11-20", "price": 1.15},
                {"date": "2026-11-20", "price": 1.68},
            ],
            "source_metadata": {},
        }

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_fund_history)

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2025-11-10",
            "end_date": "2026-11-20",
            "assets": [{"id": "fund:GUH", "kind": "fund", "symbol": "GUH", "label": "GUH"}],
        },
    )
    assert response.status_code == 200
    asset_payload = response.json()["assets"][0]
    assert asset_payload["error"] is None
    assert len(asset_payload["points"]) == 2
    assert calls[0]["include_overview_metrics"] is False

    # Simulate single-flight waiter timeout (shared SWR returns pending) while DB already has points
    monkeypatch.setattr(
        api_module,
        "_shared_swr_payload",
        lambda **_kwargs: (None, "pending", False, True),
    )
    response_pending = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2025-11-11",
            "end_date": "2026-11-20",
            "assets": [{"id": "fund:GUH", "kind": "fund", "symbol": "GUH", "label": "GUH"}],
        },
    )
    assert response_pending.status_code == 200
    pending_asset = response_pending.json()["assets"][0]
    assert pending_asset["error"] is None
    assert pending_asset["refresh_pending"] is True
    assert len(pending_asset["points"]) == 2
    assert calls[-1]["auto_refresh"] is False
    assert calls[-1]["include_overview_metrics"] is False


def test_market_comparison_history_supports_global_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_symbols: List[str] = []

    def fake_chart(yahoo_symbol: str, *, interval: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
        assert interval == "1d"
        requested_symbols.append(yahoo_symbol)
        base = 5800.0 if yahoo_symbol == "^GSPC" else 20500.0
        return {
            "ok": True,
            "points": [
                {"time": "2026-04-01T14:00:00+00:00", "close": base},
                {"time": "2026-04-05T14:00:00+00:00", "close": base + 50.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_chart)

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-05",
            "assets": [
                {"id": "index:SP500", "kind": "index", "symbol": "SP500", "label": "S&P 500 (ABD)"},
                {"id": "index:NASDAQ100", "kind": "index", "symbol": "NASDAQ100", "label": "Nasdaq 100 (ABD)"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert requested_symbols == ["^GSPC", "^NDX"]
    assert payload["assets"][0]["symbol"] == "SP500"
    assert payload["assets"][0]["points"] == [
        {"date": "2026-04-01", "value": 5800.0},
        {"date": "2026-04-05", "value": 5850.0},
    ]
    assert payload["assets"][1]["symbol"] == "NASDAQ100"
    assert payload["assets"][1]["points"] == [
        {"date": "2026-04-01", "value": 20500.0},
        {"date": "2026-04-05", "value": 20550.0},
    ]
    assert all(asset["error"] is None for asset in payload["assets"])


def test_market_comparison_history_converts_foreign_indices_to_try(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx_calls: List[str] = []

    def fake_chart(yahoo_symbol: str, *, interval: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
        assert interval == "1d"
        if yahoo_symbol in {"USDTRY=X", "EURTRY=X", "USDJPY=X", "USDHKD=X"}:
            fx_calls.append(yahoo_symbol)
        series_map: Dict[str, List[Dict[str, Any]]] = {
            "^IXIC": [
                {"time": "2026-04-06T14:00:00+00:00", "close": 100.0},
                {"time": "2026-04-07T14:00:00+00:00", "close": 105.0},
                {"time": "2026-04-08T14:00:00+00:00", "close": 110.0},
            ],
            "^GSPC": [
                {"time": "2026-04-06T14:00:00+00:00", "close": 5000.0},
                {"time": "2026-04-08T14:00:00+00:00", "close": 5100.0},
            ],
            "^GDAXI": [
                {"time": "2026-04-06T14:00:00+00:00", "close": 20000.0},
                {"time": "2026-04-08T14:00:00+00:00", "close": 20200.0},
            ],
            "^N225": [
                {"time": "2026-04-06T06:00:00+00:00", "close": 40000.0},
                {"time": "2026-04-08T06:00:00+00:00", "close": 42000.0},
            ],
            "^HSI": [
                {"time": "2026-04-06T06:00:00+00:00", "close": 20000.0},
                {"time": "2026-04-08T06:00:00+00:00", "close": 21000.0},
            ],
            # USDTRY intentionally omits 2026-04-07 to verify causal carry-forward
            # uses 2026-04-06 (40.0), never future 2026-04-08 (42.0).
            "USDTRY=X": [
                {"time": "2026-04-06T08:00:00+00:00", "close": 40.0},
                {"time": "2026-04-08T08:00:00+00:00", "close": 42.0},
            ],
            "EURTRY=X": [
                {"time": "2026-04-06T08:00:00+00:00", "close": 44.0},
                {"time": "2026-04-08T08:00:00+00:00", "close": 45.0},
            ],
            "USDJPY=X": [
                {"time": "2026-04-06T08:00:00+00:00", "close": 160.0},
                {"time": "2026-04-08T08:00:00+00:00", "close": 150.0},
            ],
            "USDHKD=X": [
                {"time": "2026-04-06T08:00:00+00:00", "close": 8.0},
                {"time": "2026-04-08T08:00:00+00:00", "close": 7.5},
            ],
        }
        return {"ok": True, "points": series_map[yahoo_symbol]}

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_chart)

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-06",
            "end_date": "2026-04-08",
            "target_currency": "TRY",
            "assets": [
                {"id": "index:NASDAQ", "kind": "index", "symbol": "NASDAQ"},
                {"id": "index:SP500", "kind": "index", "symbol": "SP500"},
                {"id": "index:DAX", "kind": "index", "symbol": "DAX"},
                {"id": "index:NIKKEI", "kind": "index", "symbol": "NIKKEI"},
                {"id": "index:HANGSENG", "kind": "index", "symbol": "HANGSENG"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_currency"] == "TRY"
    assert fx_calls.count("USDTRY=X") == 1

    by_id = {item["id"]: item for item in payload["assets"]}
    nasdaq = by_id["index:NASDAQ"]
    assert nasdaq["native_currency"] == "USD"
    assert nasdaq["currency"] == "TRY"
    assert nasdaq["fx_converted"] is True
    assert nasdaq["fx_pair"] == "USD/TRY"
    assert nasdaq["points"] == [
        {"date": "2026-04-06", "value": 4000.0},
        {"date": "2026-04-07", "value": 4200.0},
        {"date": "2026-04-08", "value": 4620.0},
    ]

    dax = by_id["index:DAX"]
    assert dax["native_currency"] == "EUR"
    assert dax["fx_converted"] is True
    assert dax["fx_pair"] == "EUR/TRY"
    assert dax["points"] == [
        {"date": "2026-04-06", "value": 880000.0},
        {"date": "2026-04-08", "value": 909000.0},
    ]

    nikkei = by_id["index:NIKKEI"]
    assert nikkei["native_currency"] == "JPY"
    assert nikkei["fx_converted"] is True
    assert nikkei["fx_pair"] == "JPY/TRY"
    assert nikkei["points"] == [
        {"date": "2026-04-06", "value": 10000.0},
        {"date": "2026-04-08", "value": 11760.0},
    ]

    hsi = by_id["index:HANGSENG"]
    assert hsi["native_currency"] == "HKD"
    assert hsi["fx_converted"] is True
    assert hsi["fx_pair"] == "HKD/TRY"
    assert hsi["points"] == [
        {"date": "2026-04-06", "value": 100000.0},
        {"date": "2026-04-08", "value": 117600.0},
    ]


def test_market_comparison_history_try_fx_failure_does_not_leak_unconverted_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_chart(yahoo_symbol: str, *, interval: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
        if yahoo_symbol == "USDTRY=X":
            return {"ok": False, "error": "upstream_down", "points": []}
        return {
            "ok": True,
            "points": [
                {"time": "2026-04-10T14:00:00+00:00", "close": 100.0},
                {"time": "2026-04-12T14:00:00+00:00", "close": 105.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_chart)

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-10",
            "end_date": "2026-04-12",
            "target_currency": "TRY",
            "assets": [
                {"id": "index:NASDAQ", "kind": "index", "symbol": "NASDAQ"},
            ],
        },
    )

    assert response.status_code == 200
    asset = response.json()["assets"][0]
    assert asset["points"] == []
    assert asset["fx_converted"] is False
    assert asset["fx_pair"] == "USD/TRY"
    assert asset["error"] == "fx_rate_unavailable:USD/TRY"


def test_market_comparison_history_keeps_try_assets_unconverted_when_target_try(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fund_history(processed_dir: Any, fund_code: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "ok",
            "source": "sqlite",
            "currency": "TRY",
            "points": [
                {"date": "2026-04-15", "price": 12.0},
                {"date": "2026-04-17", "price": 12.6},
            ],
            "source_metadata": {},
        }

    def fake_chart(yahoo_symbol: str, *, interval: str, start_date: Any, end_date: Any) -> Dict[str, Any]:
        base = {"BIMAS.IS": 400.0, "XU100.IS": 9800.0, "USDTRY=X": 41.0}[yahoo_symbol]
        return {
            "ok": True,
            "points": [
                {"time": "2026-04-15T10:00:00+00:00", "close": base},
                {"time": "2026-04-17T10:00:00+00:00", "close": base + 2.0},
            ],
        }

    monkeypatch.setattr(fund_service_module, "get_fund_performance_payload", fake_fund_history)
    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_period_raw", fake_chart)

    client = TestClient(app)
    response = client.post(
        "/market/comparison-history",
        json={
            "start_date": "2026-04-15",
            "end_date": "2026-04-17",
            "target_currency": "TRY",
            "assets": [
                {"id": "fund:TLY", "kind": "fund", "symbol": "TLY"},
                {"id": "stock:BIMAS", "kind": "stock", "symbol": "BIMAS"},
                {"id": "index:XU100", "kind": "index", "symbol": "XU100"},
                {"id": "fx:USD/TRY", "kind": "fx", "symbol": "USD/TRY"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    by_id = {item["id"]: item for item in payload["assets"]}
    assert all(item["fx_converted"] is False for item in payload["assets"])
    assert all(item["error"] is None for item in payload["assets"])
    assert by_id["fund:TLY"]["points"] == [
        {"date": "2026-04-15", "value": 12.0},
        {"date": "2026-04-17", "value": 12.6},
    ]
    assert by_id["stock:BIMAS"]["points"] == [
        {"date": "2026-04-15", "value": 400.0},
        {"date": "2026-04-17", "value": 402.0},
    ]
    assert by_id["index:XU100"]["points"] == [
        {"date": "2026-04-15", "value": 9800.0},
        {"date": "2026-04-17", "value": 9802.0},
    ]
    assert by_id["fx:USD/TRY"]["points"] == [
        {"date": "2026-04-15", "value": 41.0},
        {"date": "2026-04-17", "value": 43.0},
    ]


def test_fetch_stock_card_intraday_maps_chart_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chart(_yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
        assert interval == "5m"
        assert range_ == "1d"
        return {
            "ok": True,
            "meta": {
                "regularMarketPrice": 760.0,
                "chartPreviousClose": 763.0,
                "regularMarketDayHigh": 761.5,
                "regularMarketDayLow": 747.0,
                "regularMarketVolume": 3_729_933,
                "currency": "TRY",
                "marketState": "REGULAR",
                "regularMarketTime": 1_777_110_000,
            },
            "points": [
                {"time": "2026-04-25T08:00:00+00:00", "close": 750.0, "high": 752.0, "low": 748.0},
                {"time": "2026-04-25T08:05:00+00:00", "close": 760.0, "high": 761.0, "low": 754.0},
            ],
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_chart)

    payload = api_module._fetch_stock_card_intraday("BIMAS")

    assert payload["yahoo_symbol"] == "BIMAS.IS"
    assert payload["price"] == 760.0
    assert payload["change_pct"] == -0.39
    assert payload["high"] == 761.5
    assert payload["low"] == 747.0
    assert payload["prev_close"] == 763.0
    assert payload["volume"] == 3_729_933
    assert payload["volume_lot"] == 3_729_933
    assert len(payload["line_points"]) == 2


def test_fetch_stock_card_intraday_returns_empty_on_chart_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_yahoo_chart_raw",
        lambda *_args, **_kwargs: {"ok": False, "error": "provider_down"},
    )

    payload = api_module._fetch_stock_card_intraday("BIMAS")

    assert payload["line_points"] == []
    assert payload["price"] is None
    assert payload["error"] == "provider_down"


def test_market_indices_list_returns_main_and_sector_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(index_code: str) -> Dict[str, Any]:
        prices = {"XUTUM": 330.0, "XU100": 110.0, "XU030": 220.0, "XBANK": 440.0}
        prevs = {"XUTUM": 300.0, "XU100": 100.0, "XU030": 200.0, "XBANK": 400.0}
        labels = {"XUTUM": "BIST Tüm", "XU100": "BIST 100", "XU030": "BIST 30", "XBANK": "BIST Banka"}
        return {
            "symbol": index_code,
            "label": labels[index_code],
            "yahoo_symbol": f"{index_code}.IS",
            "price": prices[index_code],
            "prev_close": prevs[index_code],
            "change": prices[index_code] - prevs[index_code],
            "change_pct": 10.0,
            "high": 112.0,
            "low": 99.0,
            "volume": 123000000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        }

    def fake_bases(index_code: str) -> Dict[str, Any]:
        latest = {"XUTUM": 330.0, "XU100": 110.0, "XU030": 220.0, "XBANK": 440.0}[index_code]
        return {
            "base_1w": latest / 1.1,
            "base_1m": latest - 20.0,
            "base_3m": latest - 30.0,
            "base_6m": latest - 40.0,
            "base_ytd": latest - 50.0,
            "base_1y": latest - 60.0,
            "base_5y": latest - 70.0,
            "latest_close": latest,
            "as_of": "2026-04-24T12:00:00+00:00",
            "yahoo_symbol": f"{index_code}.IS",
        }

    monkeypatch.setattr(api_module, "_fetch_index_quote", fake_quote)
    monkeypatch.setattr(api_module, "_fetch_index_return_bases", fake_bases)
    monkeypatch.setattr(api_module, "_MARKET_INDEX_ORDER", ["XUTUM", "XU100", "XU030", "XBANK"])

    client = TestClient(app)
    response = client.get("/market/indices")

    assert response.status_code == 200
    payload = response.json()
    assert [row["symbol"] for row in payload["rows"]] == ["XUTUM", "XU100", "XU030", "XBANK"]
    assert payload["rows"][0]["label"] == "BIST Tüm"
    assert payload["rows"][0]["return_1w_pct"] == 10.0
    assert payload["rows"][0]["volume"] == 123000000.0
    assert "XPTIC" not in {row["symbol"] for row in payload["rows"]}


def test_market_sector_index_order_excludes_capped_and_removed_codes() -> None:
    assert "XBANK" in api_module._MARKET_INDEX_ORDER
    assert "XUSIN" in api_module._MARKET_INDEX_ORDER
    for code in {"XSINS", "XGYOS", "XTKJS", "XTTIC", "XPTIC", "XKNKL", "XYIHZ"}:
        assert code not in api_module._MARKET_INDEX_ORDER
        assert code not in api_module._MARKET_INDEX_META


def test_market_index_return_bases_fall_back_to_isyatirim_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_yahoo_chart(_yahoo_symbol: str, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "points": [{"time": "2026-05-27T00:00:00+00:00", "close": 200.0}],
        }

    def fake_isyatirim_history(
        index_code: str,
        *,
        start_date: date,
        end_date: date,
    ) -> List[tuple[datetime, float]]:
        assert index_code == "XAKUR"
        assert start_date < end_date
        return [
            (datetime(2021, 5, 17, tzinfo=timezone.utc), 70.0),
            (datetime(2025, 5, 27, tzinfo=timezone.utc), 100.0),
            (datetime(2025, 11, 26, tzinfo=timezone.utc), 110.0),
            (datetime(2026, 1, 2, tzinfo=timezone.utc), 120.0),
            (datetime(2026, 2, 25, tzinfo=timezone.utc), 130.0),
            (datetime(2026, 4, 27, tzinfo=timezone.utc), 150.0),
            (datetime(2026, 5, 20, tzinfo=timezone.utc), 160.0),
            (datetime(2026, 5, 27, tzinfo=timezone.utc), 200.0),
        ]

    monkeypatch.setattr(api_module, "_fetch_yahoo_chart_raw", fake_yahoo_chart)
    monkeypatch.setattr(api_module, "_fetch_isyatirim_index_history", fake_isyatirim_history)

    bases = api_module._fetch_index_return_bases("XAKUR")

    assert bases["history_source"] == "isyatirim"
    assert bases["base_1w"] == 160.0
    assert bases["base_1m"] == 150.0
    assert bases["base_3m"] == 130.0
    assert bases["base_6m"] == 110.0
    assert bases["base_ytd"] == 120.0
    assert bases["base_1y"] == 100.0

    row = api_module._market_index_row(
        "XAKUR",
        quote={
            "symbol": "XAKUR",
            "label": "BIST Aracı Kurumlar",
            "yahoo_symbol": "XAKUR.IS",
            "price": 220.0,
            "prev_close": 200.0,
            "change": 20.0,
            "change_pct": 10.0,
            "high": 225.0,
            "low": 198.0,
            "volume": None,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-05-27T12:00:00+00:00",
            "error": None,
        },
    )
    assert row["return_1w_pct"] == 37.5
    assert row["return_1m_pct"] == 46.67
    assert row["return_ytd_pct"] == 83.33


def test_market_index_detail_returns_line_points_and_weighted_constituents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XU100",
            "label": "BIST 100",
            "yahoo_symbol": "XU100.IS",
            "price": 1000.0,
            "prev_close": 990.0,
            "change": 10.0,
            "change_pct": 1.01,
            "high": 1010.0,
            "low": 980.0,
            "volume": 1000000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_index_return_bases",
        lambda _index_code: {
            "base_1w": 900.0,
            "base_1m": 850.0,
            "base_3m": 800.0,
            "base_6m": 750.0,
            "base_ytd": 700.0,
            "base_1y": 650.0,
            "base_5y": 500.0,
            "latest_close": 1000.0,
            "as_of": "2026-04-24T12:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        api_module,
        "_index_intraday_payload",
        lambda _index_code: {
            "line_points": [
                {"time": "2026-04-24T09:00:00+00:00", "close": 990.0},
                {"time": "2026-04-24T10:00:00+00:00", "close": 1000.0},
            ],
            "high": 1005.0,
            "low": 985.0,
            "prev_close": 990.0,
            "yahoo_symbol": "XU100.IS",
        },
    )
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: {
            "rows": [
                {
                    "company": "AAA",
                    "price": 10.0,
                    "price_currency": "TRY",
                    "change_pct": 2.0,
                    "volume": 100.0,
                },
                {
                    "company": "BBB",
                    "price": 30.0,
                    "price_currency": "TRY",
                    "change_pct": -1.0,
                    "volume": 200.0,
                },
            ]
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "AAA": {"fdpo": 0.5, "shares_outstanding": 100.0},
            "BBB": {"fdpo": 0.5, "shares_outstanding": 100.0},
        },
    )
    monkeypatch.setattr(
        api_module,
        "_index_weight_inputs_for_symbol",
        lambda symbol, **_kwargs: {
            "shares_outstanding": 100.0,
            "fdpo": 0.5,
            "weight_coefficient": 1.0,
        },
    )

    client = TestClient(app)
    response = client.get("/market/indices/XU100")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XU100"
    assert payload["line_points"][1]["close"] == 1000.0
    assert payload["return_5y_pct"] == 100.0
    assert payload["weight_status"] == "available"
    rows = {row["symbol"]: row for row in payload["constituents"]}
    assert rows["AAA"]["weight_pct"] == 25.0
    assert rows["AAA"]["point_effect"] == 5.0
    assert rows["BBB"]["weight_pct"] == 75.0
    assert rows["BBB"]["point_effect"] == -7.5


def test_market_sector_index_detail_uses_bist_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XBANK",
            "label": "BIST Banka",
            "yahoo_symbol": "XBANK.IS",
            "price": 1500.0,
            "prev_close": 1450.0,
            "change": 50.0,
            "change_pct": 3.45,
            "high": 1510.0,
            "low": 1440.0,
            "volume": 5000000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_index_return_bases",
        lambda _index_code: {
            "base_1w": 1400.0,
            "base_1m": 1300.0,
            "base_3m": 1200.0,
            "base_6m": 1100.0,
            "base_ytd": 1000.0,
            "base_1y": 900.0,
            "base_5y": 750.0,
            "latest_close": 1500.0,
            "as_of": "2026-04-24T12:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        api_module,
        "_index_intraday_payload",
        lambda _index_code: {
            "line_points": [{"time": "2026-04-24T10:00:00+00:00", "close": 1500.0}],
            "high": 1510.0,
            "low": 1440.0,
            "prev_close": 1450.0,
            "yahoo_symbol": "XBANK.IS",
        },
    )
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda index, force_refresh=False: {
            "index": str(index).upper(),
            "symbols": ["AAA", "BBB"],
            "count": 2,
            "source": "test",
            "source_url": "test://bist",
            "source_date": "29/04/2026",
            "fetched_at": "2026-04-29T12:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )

    def fake_price_map(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
        assert symbols == ["AAA", "BBB"]
        assert index_name == "XUTUM"
        return {
            "AAA": {"price": 10.0, "currency": "TRY", "change_pct": 2.0, "volume": 100.0},
            "BBB": {"price": 30.0, "currency": "TRY", "change_pct": -1.0, "volume": 200.0},
        }

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_price_map)
    basic_summary_calls: List[int] = []
    metadata_calls: List[str] = []

    def fake_basic_summary_map() -> Dict[str, Dict[str, Any]]:
        basic_summary_calls.append(1)
        return {
            "AAA": {"fdpo": 0.5, "shares_outstanding": 1000.0},
            "BBB": {"fdpo": 0.5, "shares_outstanding": 50.0},
        }

    def fake_metadata(_cache_dir: Any, symbol: str) -> Dict[str, Any]:
        metadata_calls.append(symbol)
        return {"shares_outstanding": 100.0}

    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", fake_basic_summary_map)
    monkeypatch.setattr(api_module, "_load_cached_kap_market_metadata", fake_metadata)
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: pytest.fail("sector index detail should use the BIST index universe directly"),
    )
    monkeypatch.setattr(
        api_module,
        "_latest_share_count_from_kap_cache",
        lambda _symbol: pytest.fail("preloaded KAP metadata should be reused for index weights"),
    )

    client = TestClient(app)
    response = client.get("/market/indices/XBANK")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XBANK"
    assert payload["return_5y_pct"] == 100.0
    assert payload["weight_status"] == "available"
    assert [row["symbol"] for row in payload["constituents"]] == ["BBB", "AAA"]
    weights = {row["symbol"]: row["weight_pct"] for row in payload["constituents"]}
    assert weights == {"AAA": 25.0, "BBB": 75.0}
    assert all("_kap_market_metadata" not in row for row in payload["constituents"])
    assert basic_summary_calls == [1]
    assert metadata_calls == ["AAA", "BBB"]


def test_market_index_detail_leaves_weights_empty_when_inputs_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XU030",
            "label": "BIST 30",
            "yahoo_symbol": "XU030.IS",
            "price": 2000.0,
            "prev_close": 1980.0,
            "change": 20.0,
            "change_pct": 1.01,
            "high": None,
            "low": None,
            "volume": None,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(api_module, "_fetch_index_return_bases", lambda _index_code: {})
    monkeypatch.setattr(api_module, "_index_intraday_payload", lambda _index_code: {"line_points": []})
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: {
            "rows": [
                {"company": "AAA", "price": 10.0, "price_currency": "TRY", "change_pct": 2.0, "volume": 100.0},
            ]
        },
    )
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(
        api_module,
        "_index_weight_inputs_for_symbol",
        lambda _symbol, **_kwargs: {"shares_outstanding": 100.0, "fdpo": None, "weight_coefficient": None},
    )

    client = TestClient(app)
    response = client.get("/market/indices/XU030")

    assert response.status_code == 200
    payload = response.json()
    assert payload["weight_status"] == "unavailable"
    assert payload["constituents"][0]["weight_pct"] is None
    assert payload["constituents"][0]["point_effect"] is None
    assert "Ağırlık" in payload["weight_note"]


def test_market_index_detail_rejects_unknown_index() -> None:
    client = TestClient(app)
    response = client.get("/market/indices/XU050")

    assert response.status_code == 400
    assert "Desteklenmeyen endeks" in response.json()["detail"]


def test_market_watch_success_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        if yahoo_symbol == "XU100.IS":
            return _watch_quote(price=10235.0, prev_close=10195.0)
        if yahoo_symbol == "XU030.IS":
            return _watch_quote(price=11312.0, prev_close=11300.0)
        return {"ok": False, "error": "unsupported_symbol"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(
        api_module,
        "_market_fx_payload",
        lambda: _fx_payload_for_watch(
            [
                {
                    "symbol": "USD/TRY",
                    "label": "Amerikan Doları",
                    "yahoo_symbol": "USDTRY=X",
                    "price": 38.52,
                    "prev_close": 38.40,
                    "change": 0.12,
                    "change_pct": 0.31,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "EUR/TRY",
                    "label": "Euro",
                    "yahoo_symbol": "EURTRY=X",
                    "price": 41.80,
                    "prev_close": 41.72,
                    "change": 0.08,
                    "change_pct": 0.19,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "GBP/TRY",
                    "label": "?ngiliz Sterlini",
                    "yahoo_symbol": "GBPTRY=X",
                    "price": 49.10,
                    "prev_close": 49.00,
                    "change": 0.10,
                    "change_pct": 0.20,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "EUR/USD",
                    "label": "Euro / Dolar",
                    "yahoo_symbol": "EURUSD=X",
                    "price": 1.085,
                    "prev_close": 1.082,
                    "change": 0.003,
                    "change_pct": 0.28,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        api_module,
        "_market_commodities_payload",
        lambda: _commodity_payload_for_watch(
            [
                {
                    "symbol": "BRENT",
                    "label": "Brent Petrol",
                    "yahoo_symbol": "BZ=F",
                    "price": 86.12,
                    "prev_close": 85.70,
                    "change": 0.42,
                    "change_pct": 0.49,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "ALTIN",
                    "label": "Altın (Ons)",
                    "yahoo_symbol": "GC=F",
                    "price": 2412.5,
                    "prev_close": 2404.0,
                    "change": 8.5,
                    "change_pct": 0.35,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "GUMUS",
                    "label": "Gümüş (Ons)",
                    "yahoo_symbol": "SI=F",
                    "price": 30.2,
                    "prev_close": 30.0,
                    "change": 0.2,
                    "change_pct": 0.67,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "DOGALGAZ",
                    "label": "Doğal Gaz",
                    "yahoo_symbol": "NG=F",
                    "price": 2.45,
                    "prev_close": 2.41,
                    "change": 0.04,
                    "change_pct": 1.66,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()

    assert payload["source"] == "yahoo_finance_chart"
    assert "15dk" in payload["delay_note"]

    sections = payload["sections"]
    assert [row["symbol"] for row in sections["indices"]] == [
        "XUTUM",
        "XU100",
        "XU030",
        "NASDAQ",
        "SP500",
        "DOW",
    ]
    assert [row["symbol"] for row in sections["fx"]] == ["USD/TRY", "EUR/TRY", "GBP/TRY", "EUR/USD"]
    assert [row["symbol"] for row in sections["commodities"]] == ["BRENT", "ALTIN", "GUMUS", "DOGALGAZ"]


def test_market_watch_reuses_global_quote_for_us_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        calls.append(yahoo_symbol)
        prices = {
            "^GSPC": 5100.0,
            "^IXIC": 16000.0,
            "^DJI": 39000.0,
        }
        if yahoo_symbol in prices:
            return _watch_quote(price=prices[yahoo_symbol], prev_close=prices[yahoo_symbol] - 100, currency="USD")
        return {"ok": False, "error": "unsupported_symbol"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", lambda: _fx_payload_for_watch([]))
    monkeypatch.setattr(api_module, "_market_commodities_payload", lambda: _commodity_payload_for_watch([]))

    client = TestClient(app)
    watch_payload = client.get("/market/watch").json()
    global_payload = client.get("/market/watch/global").json()

    watch_indices = {row["symbol"]: row for row in watch_payload["sections"]["indices"]}
    global_indices = {row["symbol"]: row for row in global_payload["items"]}

    for symbol in ("SP500", "NASDAQ", "DOW"):
        assert watch_indices[symbol]["price"] == global_indices[symbol]["price"]
        assert watch_indices[symbol]["as_of"] == global_indices[symbol]["as_of"]

    # Each US index is fetched once and then shared by both endpoints.
    assert calls.count("^GSPC") == 1
    assert calls.count("^IXIC") == 1
    assert calls.count("^DJI") == 1


def test_market_watch_index_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []
    calls_lock = threading.Lock()

    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        with calls_lock:
            calls.append(yahoo_symbol)
        if yahoo_symbol == "XUTUM.IS":
            return _watch_quote(price=10000.0, prev_close=9950.0)
        if yahoo_symbol == "XU100.IS":
            return {"ok": False, "error": "primary_failed"}
        if yahoo_symbol == "^XU100":
            return _watch_quote(price=10235.0, prev_close=10195.0)
        if yahoo_symbol == "XU030.IS":
            return _watch_quote(price=11312.0, prev_close=11300.0)
        return {"ok": False, "error": "unsupported_symbol"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", lambda: _fx_payload_for_watch([]))
    monkeypatch.setattr(api_module, "_market_commodities_payload", lambda: _commodity_payload_for_watch([]))

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()

    indices = {row["symbol"]: row for row in payload["sections"]["indices"]}
    assert indices["XU100"]["yahoo_symbol"] == "^XU100"
    assert "XU100.IS" in calls
    assert "^XU100" in calls
    assert calls.index("XU100.IS") < calls.index("^XU100")


def test_market_watch_index_quotes_run_concurrently_and_keep_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_symbols = [entry[0] for entry in api_module._WATCH_INDEX_CANDIDATES]
    bist_candidates = api_module._WATCH_BIST_INDEX_CANDIDATES
    candidate_count = len(bist_candidates)
    bist_yahoo_symbols = {
        yahoo_symbol
        for _symbol, _label, yahoo_candidates in bist_candidates
        for yahoo_symbol in yahoo_candidates
    }
    quote_barrier = threading.Barrier(candidate_count)
    active_bist_calls = 0
    max_active_bist_calls = 0
    state_lock = threading.Lock()

    def _quote_for(yahoo_symbol: str) -> Dict[str, Any]:
        index = yahoo_symbol.split(".", maxsplit=1)[0].replace("^", "")
        price_map = {
            "XUTUM": 10000.0,
            "XU100": 1000.0,
            "XU030": 3000.0,
            "IXIC": 16000.0,
            "GSPC": 5100.0,
            "DJI": 39000.0,
        }
        return _watch_quote(
            price=price_map.get(index, 100.0),
            prev_close=99.0,
        )

    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        nonlocal active_bist_calls, max_active_bist_calls
        if yahoo_symbol not in bist_yahoo_symbols:
            # SP500/NASDAQ/DOW come from the shared global cache now, so the
            # world-index fetches must not join the BIST barrier.
            return _quote_for(yahoo_symbol)

        with state_lock:
            active_bist_calls += 1
            max_active_bist_calls = max(max_active_bist_calls, active_bist_calls)
        try:
            quote_barrier.wait(timeout=3)
            return _quote_for(yahoo_symbol)
        finally:
            with state_lock:
                active_bist_calls -= 1

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", lambda: _fx_payload_for_watch([]))
    monkeypatch.setattr(api_module, "_market_commodities_payload", lambda: _commodity_payload_for_watch([]))

    response = TestClient(app).get("/market/watch")

    assert response.status_code == 200
    indices = response.json()["sections"]["indices"]
    assert [row["symbol"] for row in indices] == expected_symbols
    assert [row["price"] for row in indices][:3] == [10000.0, 1000.0, 3000.0]
    assert [row["price"] for row in indices][3:] == [16000.0, 5100.0, 39000.0]
    assert max_active_bist_calls == candidate_count


def test_market_watch_reuses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    counters: Dict[str, List[int]] = {
        "quote": [0],
        "fx": [0],
        "commodity": [0],
    }

    def fake_quote(_yahoo_symbol: str) -> Dict[str, Any]:
        counters["quote"][0] += 1
        return _watch_quote(price=100.0, prev_close=99.0)

    def fake_fx() -> Dict[str, Any]:
        counters["fx"][0] += 1
        return _fx_payload_for_watch([])

    def fake_commodity() -> Dict[str, Any]:
        counters["commodity"][0] += 1
        return _commodity_payload_for_watch([])

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", fake_fx)
    monkeypatch.setattr(api_module, "_market_commodities_payload", fake_commodity)

    client = TestClient(app)
    first = client.get("/market/watch")
    second = client.get("/market/watch")

    assert first.status_code == 200
    assert second.status_code == 200
    assert counters["quote"][0] == (
        len(api_module._WATCH_BIST_INDEX_CANDIDATES)
        + len(api_module._WATCH_GLOBAL_INDEX_CANDIDATES)
    )
    assert counters["fx"][0] == 1
    assert counters["commodity"][0] == 1


def test_market_watch_handles_partial_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(_yahoo_symbol: str) -> Dict[str, Any]:
        return {"ok": False, "error": "provider_down"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(
        api_module,
        "_market_fx_payload",
        lambda: _fx_payload_for_watch(
            [
                {
                    "symbol": "USD/TRY",
                    "label": "Amerikan Doları",
                    "yahoo_symbol": "USDTRY=X",
                    "price": 38.52,
                    "prev_close": 38.40,
                    "change": 0.12,
                    "change_pct": 0.31,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        api_module,
        "_market_commodities_payload",
        lambda: _commodity_payload_for_watch(
            [
                {
                    "symbol": "BRENT",
                    "label": "Brent Petrol",
                    "yahoo_symbol": "BZ=F",
                    "price": 86.12,
                    "prev_close": 85.70,
                    "change": 0.42,
                    "change_pct": 0.49,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                }
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()
    sections = payload["sections"]

    index_xu100 = next(row for row in sections["indices"] if row["symbol"] == "XU100")
    fx_eur = next(row for row in sections["fx"] if row["symbol"] == "EUR/TRY")
    commodity_altin = next(row for row in sections["commodities"] if row["symbol"] == "ALTIN")

    assert index_xu100["price"] is None
    assert index_xu100["error"]
    assert fx_eur["price"] is None
    assert fx_eur["error"] == "instrument_not_found"
    assert commodity_altin["price"] is None
    assert commodity_altin["error"] == "instrument_not_found"


def test_market_watch_global_returns_world_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        if yahoo_symbol == "^GSPC":
            return _watch_quote(price=5100.0, prev_close=5000.0, currency="USD")
        if yahoo_symbol == "^IXIC":
            return _watch_quote(price=16000.0, prev_close=15920.0, currency="USD")
        return {"ok": False, "error": "provider_down"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)

    client = TestClient(app)
    response = client.get("/market/watch/global")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "yahoo_finance_chart"
    symbols = [row["symbol"] for row in payload["items"]]
    assert symbols[:2] == ["SP500", "NASDAQ"]
    assert "DAX" in symbols
    items = {row["symbol"]: row for row in payload["items"]}
    assert items["SP500"]["price"] == 5100.0
    assert items["SP500"]["currency"] == "USD"
    assert items["DAX"]["price"] is None
    assert items["DAX"]["error"]


def test_market_watch_global_reuses_own_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        calls.append(yahoo_symbol)
        return _watch_quote(price=100.0, prev_close=99.0, currency="USD")

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)

    client = TestClient(app)
    first = client.get("/market/watch/global")
    second = client.get("/market/watch/global")

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == len(api_module._WATCH_GLOBAL_INDEX_CANDIDATES)
    assert first.json()["as_of"] == second.json()["as_of"]
