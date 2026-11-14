from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api_module_facade import api_module


def test_isyatirim_tables_include_sector_and_current_provider_multiples() -> None:
    html = """
    <table data-csvname="temelozet">
      <thead><tr>
        <th>Kod</th><th>Hisse Adı</th><th>Sektör</th><th>Kapanış (TL)</th>
        <th>Piyasa Değeri (mn TL)</th><th>Piyasa Değeri (mn $)</th>
        <th>Halka Açıklık Oranı (%)</th><th>Sermaye (mn TL)</th>
      </tr></thead>
      <tbody><tr><td>AAA</td><td>AAA Teknoloji</td><td>Teknoloji</td><td>18,00</td>
        <td>18.000,0</td><td>500,0</td><td>35,0</td><td>1.000,0</td></tr></tbody>
    </table>
    <table data-csvname="temelfinansal">
      <thead><tr>
        <th>Kod</th><th>Kapanış (TL)</th><th>F/K</th><th>FD/FAVÖK</th>
        <th>FD/Satışlar</th><th>PD/DD</th><th>Son Dönem</th>
      </tr></thead>
      <tbody><tr><td>AAA</td><td>18,00</td><td>12,4</td><td>5,6</td>
        <td>1,7</td><td>2,4</td><td>6/2026</td></tr></tbody>
    </table>
    """

    row = api_module._extract_isyatirim_basic_summary_map(html)["AAA"]

    assert row["sector"] == "Teknoloji"
    assert row["market_cap"] == 18_000_000_000.0
    assert row["shares_outstanding"] == 1_000_000_000.0
    assert row["free_float_pct"] == 35.0
    assert row["fdpo"] == 0.35
    assert row["fk"] == 12.4
    assert row["fd_favok"] == 5.6
    assert row["fd_satis"] == 1.7
    assert row["pd_dd"] == 2.4
    assert row["provider_period"] == "2026Q2"


def test_market_stock_row_computes_current_valuation_from_snapshot() -> None:
    row = api_module._market_stock_row(
        "AAA",
        cached_meta={
            "latest_quarter": "2026Q2",
            "has_kap_cache": True,
            "shares_outstanding": 100.0,
            "company_kind": "generic",
            "ttm_net_kar": 100.0,
            "ttm_satis_gelirleri": 400.0,
            "ttm_favok": 100.0,
            "ozkaynaklar": 500.0,
            "net_borc": 50.0,
            "roe_pct": 20.0,
            "favok_margin_pct": 25.0,
            "net_margin_pct": 25.0,
        },
        quote={"price": 10.0, "currency": "TRY", "change_pct": 1.0},
        return_bases={},
        basic_summary={"sector": "Enerji", "free_float_pct": 42.7},
    )

    assert row["market_cap"] == 1_000.0
    assert row["fk"] == 10.0
    assert row["pd_dd"] == 2.0
    assert row["fd_favok"] == 10.5
    assert row["fd_satis"] == pytest.approx(2.62)
    assert row["net_borc_favok"] == 0.5
    assert row["roe_pct"] == 20.0
    assert row["favok_margin_pct"] == 25.0
    assert row["sector"] == "Enerji"
    assert row["fundamental_period"] == "2026Q2"
    assert row["free_float_pct"] == 42.7


def test_market_stock_row_does_not_treat_negative_ebitda_as_low_leverage() -> None:
    row = api_module._market_stock_row(
        "AAA",
        cached_meta={
            "shares_outstanding": 100.0,
            "ttm_net_kar": 20.0,
            "ttm_satis_gelirleri": 400.0,
            "ttm_favok": -10.0,
            "ozkaynaklar": 500.0,
            "net_borc": 50.0,
            "net_borc_favok": -5.0,
        },
        quote={"price": 10.0},
        return_bases={},
    )

    assert row["net_borc_favok"] is None


def test_market_cap_uses_live_price_with_provider_share_count() -> None:
    market_cap = api_module._market_cap_from_quote_and_meta(
        {"price": 3.0},
        {},
        {"market_cap": 2_000.0, "shares_outstanding": 10.0},
    )

    assert market_cap == 30.0


def test_market_stock_row_uses_provider_ratio_fallback_without_kap_snapshot() -> None:
    row = api_module._market_stock_row(
        "AAA",
        cached_meta={},
        quote={"price": 10.0},
        return_bases={},
        basic_summary={
            "sector": "Teknoloji",
            "shares_outstanding": 100.0,
            "fk": 12.0,
            "pd_dd": 2.0,
            "fd_favok": 6.0,
            "fd_satis": 1.5,
            "provider_period": "2026Q2",
        },
    )

    assert row["market_cap"] == 1_000.0
    assert row["sector"] == "Teknoloji"
    assert row["fk"] == 12.0
    assert row["pd_dd"] == 2.0
    assert row["fd_favok"] == 6.0
    assert row["fd_satis"] == 1.5
    assert row["fundamental_period"] == "2026Q2"


def test_market_stock_row_omits_ebitda_multiples_for_banks() -> None:
    row = api_module._market_stock_row(
        "GARAN",
        cached_meta={},
        quote={"price": 10.0},
        return_bases={},
        basic_summary={
            "sector": "Bankacılık",
            "shares_outstanding": 100.0,
            "fk": 12.0,
            "pd_dd": 2.0,
            "fd_favok": 6.0,
            "fd_satis": 1.5,
        },
    )

    assert row["fk"] == 12.0
    assert row["pd_dd"] == 2.0
    assert row["fd_favok"] is None
    assert row["fd_satis"] is None
    assert row["net_borc_favok"] is None


def test_market_screener_filters_rows_and_returns_all_sectors(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[Dict[str, Any]] = [
        {"company": "AAA", "sector": "Enerji", "fk": 10.0, "fd_favok": 6.5, "pd_dd": 2.0, "roe_pct": 30.0, "ttm_net_kar": 20.0, "net_borc": 5.0, "net_borc_favok": 0.5},
        {"company": "BBB", "sector": "Bankacılık", "fk": 15.0, "fd_favok": None, "pd_dd": 1.5, "roe_pct": 10.0, "ttm_net_kar": 30.0, "net_borc": -20.0, "net_borc_favok": -0.4},
        {"company": "CCC", "sector": "Enerji", "fk": 8.0, "fd_favok": 7.0, "pd_dd": 1.0, "roe_pct": 40.0, "ttm_net_kar": -5.0, "net_borc": 10.0, "net_borc_favok": -0.2},
        {"company": "DDD", "sector": "Teknoloji", "fk": 2.0, "fd_favok": 9.0, "pd_dd": 0.8, "roe_pct": 20.0, "ttm_net_kar": 60.0, "net_borc": 15.0, "net_borc_favok": 0.4},
    ]
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: {"rows": rows, "as_of": "2026-10-18T12:00:00+00:00"},
    )

    response = TestClient(api_module.app).get(
        "/market/screener",
        params={
            "index": "XU100",
            "max_fk": 12,
            "max_fd_favok": 8,
            "min_roe_pct": 25,
            "positive_net_profit": "true",
            "sort_by": "fk",
            "sort_dir": "asc",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["index"] == "XU100"
    assert payload["count"] == 1
    assert [row["company"] for row in payload["rows"]] == ["AAA"]
    assert payload["sectors"] == ["Bankacılık", "Enerji", "Teknoloji"]

    net_cash = TestClient(api_module.app).get(
        "/market/screener",
        params={"index": "XU030", "net_cash_only": "true"},
    )
    assert net_cash.status_code == 200
    assert [row["company"] for row in net_cash.json()["rows"]] == ["BBB"]


def test_kap_companies_uses_batched_snapshot_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    api_module._KAP_COMPANIES_RESPONSE_CACHE.clear()
    monkeypatch.setattr("app.kap_service.get_kap_companies", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(api_module, "get_instrument_names", lambda *_args: {"AAA": "Alpha A.Ş."})
    monkeypatch.setattr(
        api_module,
        "get_financial_snapshots_map",
        lambda _processed_dir, symbols: {
            "BBB": {
                "company_title": "Beta A.Ş.",
                "latest_quarter": "2026Q2",
                "has_kap_cache": True,
            }
        },
    )
    monkeypatch.setattr(api_module, "_shared_cache_get_dict", lambda _key: None)
    monkeypatch.setattr(api_module, "_shared_cache_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda *_args: pytest.fail("KAP companies should use the SQL snapshot map"),
    )

    payload = api_module.kap_companies()
    rows = {row["symbol"]: row for row in payload["items"]}

    assert rows["AAA"]["title"] == "Alpha A.Ş."
    assert rows["BBB"]["title"] == "Beta A.Ş."
    assert rows["BBB"]["latest_quarter"] == "2026Q2"
    assert rows["BBB"]["has_kap_cache"] is True
    api_module._KAP_COMPANIES_RESPONSE_CACHE.clear()


def test_stock_return_bases_and_market_row_compute_sma_and_52w_distances() -> None:
    from datetime import datetime, timedelta, timezone

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # 210 daily bars smoothly rising from 100 to 309
    series = [
        {
            "dt": start + timedelta(days=i),
            "close": float(100 + i),
            "high": float(100 + i + 2),
            "low": float(100 + i - 2),
        }
        for i in range(210)
    ]
    bases = api_module._stock_return_bases_from_series(series)
    # Last 50 closes: 260..309 -> avg = 284.5; Last 200 closes: 110..309 -> avg = 209.5
    assert bases["sma_50"] == pytest.approx(284.5)
    assert bases["sma_200"] == pytest.approx(209.5)

    row = api_module._market_stock_row(
        "AAA",
        cached_meta={
            "shares_outstanding": 100.0,
            "ttm_net_kar": 100.0,
            "ttm_satis_gelirleri": 400.0,
            "ttm_favok": 100.0,
            "ozkaynaklar": 500.0,
            "net_borc": 50.0,
        },
        quote={"price": 300.0, "currency": "TRY"},
        return_bases=bases,
        historical_multiples={
            "hist_fk_avg": 400.0,
            "hist_pd_dd_avg": 80.0,
            "hist_fd_favok_avg": 300.0,
            "hist_quarters": 8,
        },
    )

    assert row["sma_50"] == 284.5
    assert row["sma_200"] == 209.5
    assert row["sma_50_diff_pct"] == pytest.approx(round(((300.0 - 284.5) / 284.5) * 100, 2))
    assert row["sma_200_diff_pct"] == pytest.approx(round(((300.0 - 209.5) / 209.5) * 100, 2))
    assert row["high_52w_dist_pct"] is not None
    assert row["low_52w_dist_pct"] is not None
    assert row["hist_fk_avg"] == 400.0
    assert row["fk_hist_diff_pct"] == pytest.approx(-25.0)
    assert row["pd_dd_hist_diff_pct"] == pytest.approx(-25.0)
    assert row["hist_quarters"] == 8


def test_compute_kap_historical_averages_for_snapshot_and_bank_exclusion() -> None:
    snapshot = {
        "stock_code": "AAA",
        "company_kind": "generic",
        "quarters": [
            {
                "year": 2024,
                "period": 4,
                "publish_date": "2025-03-03T18:00:00+03:00",
                "metrics": {
                    "net_kar": 100.0,
                    "ozkaynaklar": 500.0,
                    "favok": 120.0,
                    "net_borc": 100.0,
                    "odenmis_sermaye": 100.0,
                },
            },
            {
                "year": 2025,
                "period": 4,
                "publish_date": "2026-03-02T18:00:00+03:00",
                "metrics": {
                    "net_kar": 200.0,
                    "ozkaynaklar": 1000.0,
                    "favok": 250.0,
                    "net_borc": 0.0,
                    "odenmis_sermaye": 100.0,
                },
            },
        ],
    }
    price_points = [
        ("2025-03-03", 10.0),  # MC = 1000 -> F/K=10, PD/DD=2, FD/FAVÖK=1100/120=9.17
        ("2026-03-02", 20.0),  # MC = 2000 -> F/K=10, PD/DD=2, FD/FAVÖK=2000/250=8.0
    ]

    summary = api_module._compute_kap_historical_averages_for_snapshot(snapshot, price_points)
    assert summary["hist_fk_avg"] == pytest.approx(10.0)
    assert summary["hist_pd_dd_avg"] == pytest.approx(2.0)
    assert summary["hist_fd_favok_avg"] == pytest.approx(round((9.17 + 8.0) / 2, 2))
    assert summary["hist_quarters"] == 2

    bank_summary = api_module._compute_kap_historical_averages_for_snapshot(
        {**snapshot, "stock_code": "GARAN", "company_kind": "bank"},
        price_points,
    )
    assert bank_summary["hist_fk_avg"] == pytest.approx(10.0)
    assert bank_summary["hist_pd_dd_avg"] == pytest.approx(2.0)
    assert bank_summary["hist_fd_favok_avg"] is None


def test_market_screener_supports_historical_and_technical_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[Dict[str, Any]] = [
        {"company": "AAA", "sector": "Sanayi", "fk_hist_diff_pct": -22.0, "sma_200_diff_pct": 14.0},
        {"company": "BBB", "sector": "Sanayi", "fk_hist_diff_pct": 15.0, "sma_200_diff_pct": 12.0},
        {"company": "CCC", "sector": "Teknoloji", "fk_hist_diff_pct": -30.0, "sma_200_diff_pct": -5.0},
    ]
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda **_kwargs: {"rows": rows, "as_of": "2026-10-18T12:00:00+00:00"},
    )

    response = TestClient(api_module.app).get(
        "/market/screener",
        params={
            "max_fk_hist_diff_pct": -10,
            "min_sma_200_diff_pct": 5,
            "sort_by": "fk_hist_diff_pct",
            "sort_dir": "asc",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert [row["company"] for row in payload["rows"]] == ["AAA"]
