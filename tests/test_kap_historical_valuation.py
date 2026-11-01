from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pytest

from app import api as api_module
from app import cache as cache_module
from app import kap_api_service
from app import kap_service as kap_service_module


def test_ttm_from_q3_uses_prior_fy_and_comparable_ytd() -> None:
    quarters = [
        {
            "year": 2024,
            "period": 4,
            "metrics_ytd": {"net_kar": 120.0},
        },
        {
            "year": 2025,
            "period": 3,
            "metrics_ytd": {"net_kar": 30.0},
            "metrics_ytd_comparative": {"net_kar": 20.0},
        },
    ]

    assert api_module._build_ttm_from_reports(quarters, "net_kar") == 130.0


def test_ttm_from_q4_uses_annual_ytd_without_summing_quarters() -> None:
    quarters = [
        {
            "year": 2025,
            "period": 4,
            "metrics": {"net_kar": 100.0},
            "metrics_quarterly": {"net_kar": 100.0},
            "metrics_ytd": {"net_kar": 100.0},
        }
    ]

    assert api_module._build_ttm_from_reports(quarters, "net_kar") == 100.0


def test_comparative_quarterly_profit_does_not_replace_prior_fy_profit() -> None:
    payload = {
        "ok": True,
        "company": "TEST",
        "quarters": [
            {
                "year": 2024,
                "period": 4,
                "quarter": "2024Q4",
                "metrics": {"net_kar": 100.0, "ozkaynaklar": 100.0},
                "metrics_ytd": {"net_kar": 100.0},
            },
            {
                "year": 2025,
                "period": 3,
                "quarter": "2025Q3",
                "metrics": {"net_kar": 8.0, "ozkaynaklar": 120.0},
                "metrics_quarterly": {"net_kar": 8.0},
                "metrics_ytd": {"net_kar": 25.0},
                "metrics_ytd_comparative": {"net_kar": 19.0},
                "metrics_comparative": {"net_kar": 5.0, "ozkaynaklar": 120.0},
            },
        ],
    }

    normalized = kap_service_module.normalize_snapshot_for_frontend(payload, allow_network=False)
    prior_fy = next(q for q in normalized["quarters"] if q["quarter"] == "2024Q4")

    assert prior_fy["metrics"]["net_kar"]["value"] == 120.0
    assert prior_fy["metrics"]["ozkaynaklar"]["value"] == 120.0


def test_valuation_history_prices_populates_on_cold_miss_and_reuses_across_depths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_module.reset_cache_for_tests()
    kap_api_service._KAP_VALUATION_HISTORY_PRICE_CACHE.clear()
    calls: List[Dict[str, Any]] = []

    def fake_chart(symbol: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append({"symbol": symbol, **kwargs})
        return {
            "ok": True,
            "points": [
                {"time": "2025-03-31", "close": 100.0},
                {"time": "2026-03-31", "close": 150.0},
            ],
            "splits": [],
        }

    monkeypatch.setattr(kap_api_service, "_fetch_yahoo_chart_period_raw", fake_chart)

    # First call for 5-quarter window populates synchronously (not pending)
    first_payload, first_status, first_stale, first_pending = (
        kap_api_service._fetch_kap_valuation_history_prices(
            "THYAO",
            start_date=date(2025, 3, 31),
        )
    )
    # Second call for 20-quarter window reuses the anchored 11-year series
    second_payload, second_status, second_stale, second_pending = (
        kap_api_service._fetch_kap_valuation_history_prices(
            "THYAO",
            start_date=date(2021, 3, 31),
        )
    )

    assert first_payload is not None
    assert first_pending is False
    assert first_stale is False
    assert len(first_payload["points"]) == 2
    assert second_payload == first_payload
    assert second_status == "local_hit"
    assert second_pending is False
    assert len(calls) == 1
