from __future__ import annotations

from app import api as api_module
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
