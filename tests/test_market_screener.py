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


def test_compute_kap_historical_averages_adjusts_paid_and_bonus_capital_with_split_events() -> None:
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
                    "net_borc": 0.0,
                    "odenmis_sermaye": 50.0,
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
                    "odenmis_sermaye": 200.0,
                },
            },
        ],
    }
    price_points = [
        ("2025-03-03", 10.0),  # MC with 2x split after 2024Q4: 10 * (50 * 2) = 1000 -> F/K=10
        ("2026-03-02", 10.0),  # MC after 2025Q4 (includes paid increase): 10 * 200 = 2000 -> F/K=10
    ]

    summary = api_module._compute_kap_historical_averages_for_snapshot(
        snapshot,
        price_points,
        current_shares=200.0,
        split_events=[{"date": "2025-06-01", "ratio": 2.0}],
    )
    assert summary["hist_fk_avg"] == pytest.approx(10.0)
    assert summary["hist_pd_dd_avg"] == pytest.approx(2.0)
    assert summary["hist_quarters"] == 2
    assert summary["hist_fk_quarters"] == 2
    assert summary["hist_pd_dd_quarters"] == 2


def test_warm_stock_hist_multiples_fetches_splits_only_for_capital_changes_and_tracks_per_metric_quarters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from app import market_service

    kap_dir = tmp_path / "kap_cache"
    kap_dir.mkdir(parents=True, exist_ok=True)
    # AAA: constant capital (0 split calls expected), 3 quarters with positive equity but only 2 positive net profit.
    aaa_snapshot = {
        "stock_code": "AAA",
        "company_kind": "generic",
        "quarters": [
            {
                "year": 2023,
                "period": 4,
                "publish_date": "2024-03-04T18:00:00+03:00",
                "metrics": {"net_kar": -50.0, "ozkaynaklar": 500.0, "favok": 120.0, "net_borc": 0.0, "odenmis_sermaye": 100.0},
            },
            {
                "year": 2024,
                "period": 4,
                "publish_date": "2025-03-03T18:00:00+03:00",
                "metrics": {"net_kar": 100.0, "ozkaynaklar": 500.0, "favok": 120.0, "net_borc": 0.0, "odenmis_sermaye": 100.0},
            },
            {
                "year": 2025,
                "period": 4,
                "publish_date": "2026-03-02T18:00:00+03:00",
                "metrics": {"net_kar": 200.0, "ozkaynaklar": 1000.0, "favok": 250.0, "net_borc": 0.0, "odenmis_sermaye": 100.0},
            },
        ],
    }
    # BBB: capital doubled between 2024Q4 and 2025Q4 (1 split lookup expected).
    bbb_snapshot = {
        "stock_code": "BBB",
        "company_kind": "generic",
        "quarters": [
            {
                "year": 2024,
                "period": 4,
                "publish_date": "2025-03-03T18:00:00+03:00",
                "metrics": {"net_kar": 100.0, "ozkaynaklar": 500.0, "favok": 120.0, "net_borc": 0.0, "odenmis_sermaye": 50.0},
            },
            {
                "year": 2025,
                "period": 4,
                "publish_date": "2026-03-02T18:00:00+03:00",
                "metrics": {"net_kar": 200.0, "ozkaynaklar": 1000.0, "favok": 250.0, "net_borc": 0.0, "odenmis_sermaye": 100.0},
            },
        ],
    }
    (kap_dir / "AAA.json").write_text(json.dumps(aaa_snapshot), encoding="utf-8")
    (kap_dir / "BBB.json").write_text(json.dumps(bbb_snapshot), encoding="utf-8")

    from types import SimpleNamespace

    split_calls: list[str] = []
    monkeypatch.setattr(
        market_service,
        "CONFIG",
        SimpleNamespace(paths=SimpleNamespace(processed_dir=tmp_path)),
    )
    market_service._STOCK_HIST_MULTIPLES_STATE.update({"_ts": 0.0, "items": {}})
    monkeypatch.setattr(
        market_service,
        "_fetch_stock_weekly_5y_spark_batch",
        lambda _syms: {
            "AAA": [("2024-03-04", 10.0), ("2025-03-03", 10.0), ("2026-03-02", 20.0)],
            "BBB": [("2025-03-03", 10.0), ("2026-03-02", 20.0)],
        },
    )

    def fake_split_fetch(symbol: str, *, start_date, end_date):
        del start_date, end_date
        split_calls.append(symbol)
        return [{"date": "2025-06-01", "ratio": 2.0}]

    monkeypatch.setattr(market_service, "_fetch_stock_split_events_cached", fake_split_fetch)
    monkeypatch.setattr(market_service, "_refresh_cached_xutum_stocks_payload", lambda: None)

    result = market_service._warm_stock_hist_multiples_sync(symbols=["AAA", "BBB"])
    assert result == {"requested": 2, "populated": 2}
    assert split_calls == ["BBB"]

    items = market_service._STOCK_HIST_MULTIPLES_STATE["items"]
    assert items["AAA"]["hist_quarters"] == 3
    assert items["AAA"]["hist_fk_quarters"] == 2
    assert items["AAA"]["hist_pd_dd_quarters"] == 3
    assert items["BBB"]["hist_fk_avg"] == pytest.approx(10.0)


def test_parse_yahoo_chart_payload_extracts_sorted_splits() -> None:
    from app import market_service

    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "TEST.IS"},
                    "timestamp": [1735689600],
                    "indicators": {"quote": [{"close": [25.0], "high": [26.0], "low": [24.0], "volume": [1000]}]},
                    "events": {
                        "splits": {
                            "later": {"date": 1751328000, "numerator": 3, "denominator": 1},
                            "earlier": {"date": 1740787200, "numerator": 2, "denominator": 1},
                            "broken": {"date": "bad", "numerator": 2, "denominator": 0},
                        }
                    },
                }
            ]
        }
    }
    parsed = market_service._parse_yahoo_chart_payload(payload, "TEST.IS")
    assert parsed["ok"] is True
    assert parsed["splits"] == [
        {"date": "2025-03-01", "ratio": 2.0},
        {"date": "2025-07-01", "ratio": 3.0},
    ]


def test_compute_kap_historical_averages_falls_back_to_current_shares_on_missing_split() -> None:
    fallback_hits: list[bool] = []
    snapshot = {
        "stock_code": "SPLT",
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
                    "net_borc": 0.0,
                    "odenmis_sermaye": 10.0,
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
        ("2025-03-03", 10.0),  # Yahoo split-adjusted historical price expects current shares=100 -> MC=1000 (F/K=10)
        ("2026-03-02", 20.0),  # MC=2000 (F/K=10)
    ]

    summary = api_module._compute_kap_historical_averages_for_snapshot(
        snapshot,
        price_points,
        current_shares=100.0,
        split_events=[],  # Yahoo missed the 10x bonus split event
        on_missing_split_fallback=lambda: fallback_hits.append(True),
    )
    assert fallback_hits == [True]
    assert summary["hist_fk_avg"] == pytest.approx(10.0)
    assert summary["hist_pd_dd_avg"] == pytest.approx(2.0)


def test_hydrate_stock_hist_multiples_tolerates_v1_as_stale_and_empty_split_short_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import time
    from types import SimpleNamespace
    from app import market_service

    monkeypatch.setattr(
        market_service,
        "CONFIG",
        SimpleNamespace(paths=SimpleNamespace(processed_dir=tmp_path)),
    )
    cache_path = market_service._stock_hist_multiples_disk_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "updated_at": time.time(),
                "items": {"GARAN": {"hist_fk_avg": 5.5, "hist_quarters": 8}},
            }
        ),
        encoding="utf-8",
    )
    market_service._STOCK_HIST_MULTIPLES_STATE.update({"_ts": 123.0, "items": {}})

    loaded = market_service._hydrate_stock_hist_multiples_from_disk()
    assert loaded == 1
    assert market_service._STOCK_HIST_MULTIPLES_STATE["_ts"] == 0.0
    assert market_service._STOCK_HIST_MULTIPLES_STATE["items"]["GARAN"]["hist_fk_avg"] == 5.5

    recorded_ttls: list[int] = []
    monkeypatch.setattr(market_service, "_shared_cache_get_dict", lambda _key: None)
    monkeypatch.setattr(
        market_service,
        "_fetch_yahoo_chart_period_raw",
        lambda *_args, **_kwargs: {"ok": True, "splits": []},
    )
    monkeypatch.setattr(
        market_service,
        "_shared_cache_set",
        lambda _key, _val, ttl: recorded_ttls.append(ttl),
    )

    from datetime import date

    splits = market_service._fetch_stock_split_events_cached(
        "EMPTYTEST",
        start_date=date(2020, 1, 1),
        end_date=date(2026, 1, 1),
    )
    assert splits == []
    assert recorded_ttls == [3600]


def test_warm_stock_hist_multiples_only_missing_and_remote_stem_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace
    from app import market_service

    monkeypatch.setattr(
        market_service,
        "CONFIG",
        SimpleNamespace(paths=SimpleNamespace(processed_dir=tmp_path)),
    )
    market_service._STOCK_HIST_MULTIPLES_STATE.update(
        {
            "_ts": 9999999999.0,
            "items": {"AAA": {"hist_fk_avg": 7.5, "hist_quarters": 8}},
        }
    )

    monkeypatch.setattr(
        market_service._market_database,
        "read_json_cache_kap_metadata",
        lambda: pytest.fail("only_missing must not read remote KAP metadata when all targets exist"),
    )
    monkeypatch.setattr(
        market_service,
        "_fetch_stock_weekly_5y_spark_batch",
        lambda _syms: pytest.fail("only_missing must not call Yahoo when all targets exist"),
    )

    skipped = market_service._warm_stock_hist_multiples_sync(symbols=["AAA"], only_missing=True)
    assert skipped == {"requested": 0, "populated": 0, "skipped_existing": 1}

    # Verify stem keys returned by read_json_cache_kap_metadata (without '.json') are resolved.
    monkeypatch.setattr(market_service._market_database, "database_enabled", lambda: True)
    monkeypatch.setattr(
        market_service._market_database,
        "read_json_cache_kap_metadata",
        lambda: {"CCC": {"ok": True}},
    )
    monkeypatch.setattr(
        market_service,
        "_fetch_stock_weekly_5y_spark_batch",
        lambda syms: {s: [("2025-03-03", 10.0), ("2026-03-02", 20.0)] for s in syms},
    )
    import app.json_cache as _jc

    monkeypatch.setattr(
        _jc,
        "read_json_cache_local_first",
        lambda _path: {
            "ok": True,
            "stock_code": "CCC",
            "company_kind": "generic",
            "quarters": [
                {
                    "year": 2024,
                    "period": 4,
                    "publish_date": "2025-03-03T18:00:00+03:00",
                    "metrics": {"net_kar": 100.0, "ozkaynaklar": 500.0, "favok": 120.0, "net_borc": 0.0, "odenmis_sermaye": 50.0},
                },
                {
                    "year": 2025,
                    "period": 4,
                    "publish_date": "2026-03-02T18:00:00+03:00",
                    "metrics": {"net_kar": 200.0, "ozkaynaklar": 1000.0, "favok": 240.0, "net_borc": 0.0, "odenmis_sermaye": 50.0},
                },
            ],
        },
    )
    monkeypatch.setattr(
        market_service,
        "get_financial_snapshots_map",
        lambda _p: {"CCC": {"shares_outstanding": 50.0}},
    )
    monkeypatch.setattr(market_service, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(market_service, "_refresh_cached_xutum_stocks_payload", lambda: None)

    pop = market_service._warm_stock_hist_multiples_sync(symbols=["AAA", "CCC"], only_missing=True)
    assert pop == {"requested": 1, "populated": 1, "skipped_existing": 1}
    assert "CCC" in market_service._STOCK_HIST_MULTIPLES_STATE["items"]
