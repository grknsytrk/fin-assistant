from src.validators import validate_metric_value, validate_ratios


def test_validate_metric_rejects_impossible_margin() -> None:
    verdict = validate_metric_value(metric="favok_marji", value=250.0, unit="%")
    assert verdict["ok"] is False


def test_validate_ratios_rejects_out_of_bounds() -> None:
    verdict = validate_ratios(
        {
            "net_margin": 280.0,
            "favok_margin": 4.0,
            "brut_kar_marji": 35.0,
        }
    )
    assert verdict["ok"] is False
    assert "net_marj_aralik_disi" in verdict["flags"]


def test_validate_ratios_allows_negative_margin_in_reasonable_bounds() -> None:
    verdict = validate_ratios(
        {
            "net_margin": -45.0,
            "favok_margin": -12.0,
            "brut_kar_marji": 18.0,
        }
    )
    assert verdict["ok"] is True


def test_validate_metric_rejects_tiny_capex_values() -> None:
    verdict = validate_metric_value(metric="capex", value=14231.0, unit="TL")
    assert verdict["ok"] is False
    assert verdict["reason"] == "nakit_akisi_cok_dusuk_olasi_olcek_hatasi"


def test_validate_metric_allows_reasonable_capex_values() -> None:
    verdict = validate_metric_value(metric="capex", value=1_423_100_000.0, unit="TL")
    assert verdict["ok"] is True
