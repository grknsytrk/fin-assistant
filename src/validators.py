from __future__ import annotations

from typing import Any, Dict, List, Optional


MARGIN_METRICS = {"net_kar_marji", "favok_marji", "brut_kar_marji", "net_margin", "favok_margin"}
TL_METRICS = {
    "net_kar",
    "favok",
    "satis_gelirleri",
    "faaliyet_nakit_akisi",
    "serbest_nakit_akisi",
    "capex",
}
MARGIN_MIN_DEFAULT = -200.0
MARGIN_MAX_DEFAULT = 200.0


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def validate_metric_value(
    metric: str,
    value: Any,
    unit: str,
    expected_range: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    numeric = _as_float(value)
    if numeric is None:
        return {"ok": False, "reason": "numeric_deger_yok"}

    if expected_range:
        min_expected = _as_float(expected_range.get("min"))
        max_expected = _as_float(expected_range.get("max"))
        if min_expected is not None and numeric < min_expected:
            return {"ok": False, "reason": "config_expected_range_min_altinda"}
        if max_expected is not None and numeric > max_expected:
            return {"ok": False, "reason": "config_expected_range_max_ustunde"}

    if unit == "%":
        if numeric < MARGIN_MIN_DEFAULT or numeric > MARGIN_MAX_DEFAULT:
            return {"ok": False, "reason": "marj_beklenen_aralik_disinda"}
        return {"ok": True, "reason": "ok"}

    if metric in TL_METRICS or unit == "TL":
        abs_numeric = abs(numeric)
        if metric == "satis_gelirleri" and abs_numeric < 10_000_000.0:
            return {"ok": False, "reason": "satis_geliri_cok_dusuk_olasi_olcek_hatasi"}
        if metric in {"net_kar", "favok"} and abs_numeric < 100_000.0 and abs_numeric != 0.0:
            return {"ok": False, "reason": "kar_favok_cok_dusuk_olasi_olcek_hatasi"}
        if metric in {"faaliyet_nakit_akisi", "serbest_nakit_akisi", "capex"} and abs_numeric < 100_000.0 and abs_numeric != 0.0:
            return {"ok": False, "reason": "nakit_akisi_cok_dusuk_olasi_olcek_hatasi"}
        # Billion-level values are possible, but trillion-scale for quarterly
        # single-company retail metrics usually indicates extraction/scaling errors.
        if abs(numeric) > 5_000_000_000_000.0:
            return {"ok": False, "reason": "tl_degeri_olasi_degilden_buyuk"}
        return {"ok": True, "reason": "ok"}

    if metric == "magaza_sayisi" or unit == "count":
        if numeric < 10 or numeric > 300_000:
            return {"ok": False, "reason": "magaza_sayisi_beklenen_aralik_disinda"}
        return {"ok": True, "reason": "ok"}

    return {"ok": True, "reason": "ok"}


def validate_ratios(row: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []

    brut = _as_float(row.get("brut_kar_marji"))
    if brut is not None and (brut < MARGIN_MIN_DEFAULT or brut > MARGIN_MAX_DEFAULT):
        flags.append("brut_kar_marji_aralik_disi")

    net = _as_float(row.get("net_margin"))
    if net is not None and (net < MARGIN_MIN_DEFAULT or net > MARGIN_MAX_DEFAULT):
        flags.append("net_marj_aralik_disi")

    favok = _as_float(row.get("favok_margin"))
    if favok is not None and (favok < MARGIN_MIN_DEFAULT or favok > MARGIN_MAX_DEFAULT):
        flags.append("favok_marji_aralik_disi")

    return {"ok": len(flags) == 0, "flags": flags}
