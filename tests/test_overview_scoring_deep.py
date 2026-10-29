"""Targeted tests verifying the refined scoring logic and scenario edge cases."""
from typing import Any, Dict

import pytest

from src.overview_scoring import (
    AI_SUBSCORE_ADJUSTMENT_LIMIT,
    compute_base_analysis,
    merge_scorecard_with_adjustments,
)

B = 1_000_000_000.0


def _quarter(
    year: int,
    period: int,
    sales: float,
    favok: float,
    efk: float,
    net: float,
    cfo: float = 3.0,
    capex: float = 1.0,
    equity: float = 50.0,
    net_debt: float = 10.0,
    current_ratio: float = 1.4,
) -> Dict[str, Any]:
    fcf = cfo - capex
    return {
        "label": f"{year} Q{period}",
        "year": year,
        "period": period,
        "metrics": {
            "satis_gelirleri": sales * B,
            "favok": favok * B,
            "esas_faaliyet_kari": efk * B,
            "net_kar": net * B,
            "faaliyet_nakit_akisi": cfo * B,
            "capex": capex * B,
            "serbest_nakit_akisi": fcf * B,
            "ozkaynaklar": equity * B,
            "net_borc": net_debt * B,
        },
        "ratios": {
            "favok_marji": (favok / sales * 100) if sales else None,
            "net_kar_marji": (net / sales * 100) if sales else None,
            "roe": 12.0,
            "cari_oran": current_ratio,
            "net_borc_ozkaynak": (net_debt / equity) if equity else None,
            "nakit_donusum": (fcf / net) if net else None,
        },
    }


def _base_history():
    return [
        _quarter(2024, 3, 30, 4.5, 4.0, 1.5),
        _quarter(2024, 4, 35, 5.0, 4.5, 1.8),
        _quarter(2025, 1, 32, 4.8, 4.2, 1.6),
        _quarter(2025, 2, 40, 5.5, 5.0, 2.0),
        _quarter(2025, 3, 42, 5.8, 5.2, 2.1),
        _quarter(2025, 4, 45, 6.2, 5.5, 2.3),
        _quarter(2026, 1, 44, 6.0, 5.4, 2.2),
    ]


def test_non_operating_profit_jump_discounts_both_growth_and_profitability():
    """When net profit jumps 2B -> 8.5B but operating profit is flat,

    profitability bucket must reflect the earnings quality haircut.
    """
    history = _base_history()
    # 2026 Q2: Sales grew mildly, FAVÖK flat, net profit jumps to 8.5B (non-operating)
    latest_non_op = _quarter(2026, 2, 45, 5.5, 5.0, 8.5, cfo=4.0, capex=1.0)
    result = compute_base_analysis({"company_kind": "generic", "quarters": history + [latest_non_op]})
    card = result["scorecard"]

    karlilik = next(s for s in card["subscores"] if s["key"] == "karlilik")
    buyume = next(s for s in card["subscores"] if s["key"] == "buyume")

    # Profitability must be discounted, not sitting at 9.0+
    assert karlilik["score"] < 7.0, f"Expected discounted profitability, got {karlilik['score']}"
    assert buyume["score"] < 8.0, f"Expected discounted growth, got {buyume['score']}"
    assert any(flag["code"] == "faaliyet_disi_kar_artisi" for flag in card["flags"])


def test_operating_backed_profit_jump_scores_high():
    """When net profit jumps 2B -> 8.5B AND operating profit jumps 5.5B -> 9.5B,

    both growth and profitability should be scored strongly.
    """
    history = _base_history()
    latest_op = _quarter(2026, 2, 52, 9.5, 8.8, 8.5, cfo=7.0, capex=1.5)
    result = compute_base_analysis({"company_kind": "generic", "quarters": history + [latest_op]})
    card = result["scorecard"]

    karlilik = next(s for s in card["subscores"] if s["key"] == "karlilik")
    buyume = next(s for s in card["subscores"] if s["key"] == "buyume")

    assert karlilik["score"] >= 8.0, f"Expected high profitability, got {karlilik['score']}"
    assert buyume["score"] >= 8.5, f"Expected high growth, got {buyume['score']}"
    assert not any(flag["code"] == "faaliyet_disi_kar_artisi" for flag in card["flags"])


def test_tiny_turnaround_from_loss_stays_neutral_not_forced_to_6_5():
    """A turnaround that is economically negligible relative to turnover

    should score around neutral (5.0), not jump to 6.5.
    """
    # 2025 Q2: -10K net loss on 40B sales. 2026 Q2: +1K profit on 40B sales.
    history = [
        _quarter(2024, 3, 30, 4.0, 3.5, 0.1),
        _quarter(2024, 4, 35, 4.5, 4.0, 0.1),
        _quarter(2025, 1, 32, 4.0, 3.5, 0.1),
        _quarter(2025, 2, 40, 4.0, 3.5, -0.00001),  # tiny loss
        _quarter(2025, 3, 40, 4.0, 3.5, 0.1),
        _quarter(2025, 4, 40, 4.0, 3.5, 0.1),
        _quarter(2026, 1, 40, 4.0, 3.5, 0.1),
    ]
    latest = _quarter(2026, 2, 40, 4.0, 3.5, 0.000001)  # tiny profit
    result = compute_base_analysis({"company_kind": "generic", "quarters": history + [latest]})
    karlilik = next(s for s in result["scorecard"]["subscores"] if s["key"] == "karlilik")
    turnaround_signal = next((sig for sig in karlilik["signals"] if "Net kâr değişimi" in sig["label"]), None)
    assert turnaround_signal is not None
    # Score should be around 5.0, not >= 6.5
    assert turnaround_signal["score"] <= 5.5, f"Expected neutral score for tiny turnaround, got {turnaround_signal['score']}"


def test_negative_baseline_does_not_emit_bogus_percentage_flag():
    """When prior year or YTD profit was negative, the cross-period flag

    must not emit misleading percentage texts like 'net kârı %150 arttı'.
    """
    history = [
        _quarter(2024, 3, 30, 4.0, 3.5, 1.0),
        _quarter(2024, 4, 35, 4.5, 4.0, 1.0),
        _quarter(2025, 1, 32, 4.0, 3.5, -2.0),
        _quarter(2025, 2, 40, 4.0, 3.5, -1.0),  # Negative baseline
        _quarter(2025, 3, 40, 4.0, 3.5, 1.0),
        _quarter(2025, 4, 40, 4.0, 3.5, 1.0),
        _quarter(2026, 1, 40, 4.0, 3.5, 1.0),
    ]
    latest = _quarter(2026, 2, 40, 4.0, 3.5, 1.5)
    result = compute_base_analysis({"company_kind": "generic", "quarters": history + [latest]})
    flags = result["scorecard"]["flags"]
    # Neither ceyrek_guclu_ytd_zayif nor ceyrek_zayif_ytd_guclu should emit bogus % from negative base
    for f in flags:
        if f["code"] in ("ceyrek_guclu_ytd_zayif", "ceyrek_zayif_ytd_guclu"):
            pytest.fail(f"Unexpected flag with negative base: {f}")


def test_ai_subscore_adjustment_limit_is_0_8():
    assert AI_SUBSCORE_ADJUSTMENT_LIMIT == 0.8


def test_merge_scorecard_synthesizes_summary_when_empty():
    history = _base_history()
    latest = _quarter(2026, 2, 45, 6.0, 5.5, 2.5)
    base = compute_base_analysis({"company_kind": "generic", "quarters": history + [latest]})

    # AI returns empty summary and empty overall_adjustment, but adjusts subscores
    ai_adjustments = {
        "overall_adjustment": 0.0,
        "subscores": [
            {"key": "buyume", "adjustment": 0.5, "summary": "Büyüme ivmesi hızlandı"},
            {"key": "karlilik", "adjustment": 0.4, "summary": "Marjlar iyileşti"},
            {"key": "bilanco", "adjustment": 0.0, "summary": ""},
            {"key": "nakit_akisi", "adjustment": 0.0, "summary": ""},
        ],
        "summary": "",  # Empty
        "seasonality_note": "",
    }
    merged = merge_scorecard_with_adjustments(
        base["scorecard"],
        ai_adjustments,
        score_source="ai_adjusted",
    )
    # Summary should not be empty, and should be regenerated based on updated scores
    assert merged["summary"] != ""
    assert merged["overall_score"] >= base["scorecard"]["overall_score"]


def test_reasoning_effort_validation_and_default():
    from src.nvidia_commentary import PayloadValidationError, validate_overview_commentary_request

    base_req = {
        "company": "THYAO",
        "company_title": "Turk Hava Yollari",
        "latest_period": "2026/2",
        "history_context": {"company_kind": "generic", "quarters": [_base_history()[0]]},
        "overview_payload": {
            "income_summary": [],
            "balance_summary": [],
            "charts": [],
        },
    }

    # Default should be low
    validated_default = validate_overview_commentary_request(base_req)
    assert validated_default["reasoning_effort"] == "low"

    # Explicit medium and high should be accepted
    for effort in ("low", "medium", "high"):
        req = dict(base_req, reasoning_effort=effort)
        assert validate_overview_commentary_request(req)["reasoning_effort"] == effort

    # Invalid should raise PayloadValidationError
    with pytest.raises(PayloadValidationError):
        validate_overview_commentary_request(dict(base_req, reasoning_effort="extreme"))


