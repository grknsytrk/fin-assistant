from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

SCORE_SOURCE_DETERMINISTIC_ONLY = "deterministic_only"
SCORE_SOURCE_AI_ADJUSTED = "ai_adjusted"
SCORE_SOURCE_AI_FAILED_FALLBACK = "ai_failed_fallback"

SUBSCORE_KEYS = ("buyume", "karlilik", "bilanco", "nakit_akisi")
COMPANY_KINDS = {"generic", "bank", "insurance"}

BUCKET_LABELS = {
    "generic": {
        "buyume": "Büyüme",
        "karlilik": "Karlılık",
        "bilanco": "Bilanço",
        "nakit_akisi": "Nakit Akışı",
    },
    "bank": {
        "buyume": "Büyüme",
        "karlilik": "Karlılık",
        "bilanco": "Sermaye Kalitesi",
        "nakit_akisi": "Likidite ve Fonlama",
    },
    "insurance": {
        "buyume": "Büyüme",
        "karlilik": "Karlılık",
        "bilanco": "Sermaye Dayanıklılığı",
        "nakit_akisi": "Likidite ve Karşılıklar",
    },
}

WATCH_METRICS = {
    "generic": {
        "buyume": ["Satışlar", "FAVÖK", "Net Kar"],
        "karlilik": ["FAVÖK Marjı", "Net Kar Marjı", "ROE"],
        "bilanco": ["Net Borç / Özkaynak", "Cari Oran", "Özkaynaklar"],
        "nakit_akisi": ["Faaliyet Nakit Akışı", "Serbest Nakit Akışı", "Nakit Dönüşümü"],
    },
    "bank": {
        "buyume": ["Net Ücret Komisyon Gelirleri", "Net Faaliyet Karı", "Krediler"],
        "karlilik": ["ROE", "Net Kar", "Net Faaliyet Karı"],
        "bilanco": ["Özkaynaklar", "Karşılık / Kredi", "Finansal Varlıklar"],
        "nakit_akisi": ["Kredi / Mevduat", "Mevduatlar", "Finansal Varlıklar"],
    },
    "insurance": {
        "buyume": ["Prim Üretimi", "Alınan Net Primler", "Teknik Gelirler"],
        "karlilik": ["Teknik Denge Marjı", "ROE", "Net Kar"],
        "bilanco": ["Özkaynaklar", "Özkaynak / Teknik Karşılık", "Borç / Özkaynak"],
        "nakit_akisi": ["Nakit / Teknik Karşılık", "Teknik Denge", "Nakit Benzeri Varlıklar"],
    },
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round1(value: float) -> float:
    return round(_clamp(float(value), 0.0, 10.0) + 1e-9, 1)


def _pct_change(current: Optional[float], base: Optional[float]) -> Optional[float]:
    if current is None or base is None or base == 0:
        return None
    return ((current - base) / abs(base)) * 100.0


def _score_centered(change_pct: Optional[float], scale: float) -> Optional[float]:
    if change_pct is None:
        return None
    return _clamp(5.0 + (change_pct / scale) * 5.0, 0.0, 10.0)


def _score_higher(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    if high <= low:
        return None
    return _clamp(((value - low) / (high - low)) * 10.0, 0.0, 10.0)


def _score_lower(value: Optional[float], best: float, worst: float) -> Optional[float]:
    if value is None:
        return None
    if worst <= best:
        return None
    return _clamp((1.0 - ((value - best) / (worst - best))) * 10.0, 0.0, 10.0)


def _score_band(
    value: Optional[float],
    soft_low: float,
    ideal_low: float,
    ideal_high: float,
    soft_high: float,
) -> Optional[float]:
    if value is None:
        return None
    if soft_low >= ideal_low or ideal_high >= soft_high:
        return None
    if ideal_low <= value <= ideal_high:
        midpoint = (ideal_low + ideal_high) / 2.0
        if midpoint == ideal_low == ideal_high:
            return 9.0
        distance = abs(value - midpoint)
        span = max((ideal_high - ideal_low) / 2.0, 1e-9)
        return _clamp(9.5 - (distance / span) * 1.5, 8.0, 10.0)
    if value < ideal_low:
        return _score_higher(value, soft_low, ideal_low)
    return _score_lower(value, ideal_high, soft_high)


def _score_sign(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value > 0:
        return 7.6
    if value == 0:
        return 5.0
    return 2.2


def _bucket_tone(score: float) -> str:
    if score >= 7.5:
        return "güçlü"
    if score >= 6.0:
        return "dengeli"
    return "zayıf"


def _overall_label(score: float) -> str:
    if score >= 7.5:
        return "Güçlü"
    if score >= 6.0:
        return "Dengeli"
    return "Zayıf"


def _series_snapshot(history_context: Dict[str, Any], source: str, key: str) -> Dict[str, Any]:
    quarters = history_context.get("quarters") or []
    latest = quarters[-1] if quarters else {}
    current = ((latest.get(source) or {}) if isinstance(latest, dict) else {}).get(key)
    same_period_values: List[float] = []
    trailing_values: List[float] = []

    latest_period = int((latest or {}).get("period") or 0)
    for row in quarters[:-1]:
        if not isinstance(row, dict):
            continue
        value = ((row.get(source) or {}) if isinstance(row.get(source), dict) else {}).get(key)
        if isinstance(value, (int, float)):
            trailing_values.append(float(value))
            if int(row.get("period") or 0) == latest_period:
                same_period_values.append(float(value))

    trailing_window = trailing_values[-4:]
    current_value = float(current) if isinstance(current, (int, float)) else None
    same_period_median = median(same_period_values) if same_period_values else None
    same_period_last = same_period_values[-1] if same_period_values else None
    trend_baseline = mean(trailing_window) if trailing_window else None

    seasonal_delta = _pct_change(current_value, same_period_median)
    trend_delta = _pct_change(current_value, trend_baseline)
    yoy_change = _pct_change(current_value, same_period_last)

    if seasonal_delta is None:
        season_flag = "unknown"
    elif abs(seasonal_delta) <= 10:
        season_flag = "within"
    elif seasonal_delta > 10 and (trend_delta is None or trend_delta >= -5):
        season_flag = "above"
    elif seasonal_delta < -10 and (trend_delta is None or trend_delta <= 5):
        season_flag = "below"
    else:
        season_flag = "mixed"

    return {
        "current": current_value,
        "yoy_change": yoy_change,
        "seasonal_delta": seasonal_delta,
        "trend_delta": trend_delta,
        "same_period_count": len(same_period_values),
        "trailing_count": len(trailing_window),
        "season_flag": season_flag,
    }


def _signal(label: str, score: Optional[float], weight: float) -> Optional[Dict[str, Any]]:
    if score is None:
        return None
    return {"label": label, "score": _clamp(float(score), 0.0, 10.0), "weight": float(weight)}


def _weighted_score(signals: List[Dict[str, Any]]) -> float:
    total_weight = sum(item["weight"] for item in signals)
    if total_weight <= 0:
        return 5.0
    weighted = sum(item["score"] * item["weight"] for item in signals)
    return weighted / total_weight


def _bucket_summary(label: str, score: float, signals: List[Dict[str, Any]]) -> str:
    if not signals:
        return f"{label} tarafında veri sınırlı; puan nötr tutuldu."
    strongest = max(signals, key=lambda item: item["score"])
    weakest = min(signals, key=lambda item: item["score"])
    tone = _bucket_tone(score)
    if strongest["label"] == weakest["label"]:
        return f"{label} görünümü {tone}; değerlendirme büyük ölçüde {strongest['label'].lower()} üzerinden oluştu."
    return (
        f"{label} görünümü {tone}; {strongest['label'].lower()} destek verirken "
        f"{weakest['label'].lower()} baskı yaratıyor."
    )


def _seasonality_note(flags: List[str]) -> str:
    within_count = sum(1 for item in flags if item == "within")
    above_count = sum(1 for item in flags if item == "above")
    below_count = sum(1 for item in flags if item == "below")
    mixed_count = sum(1 for item in flags if item == "mixed")

    if within_count >= 2 and below_count == 0:
        return "Son dönem verileri kendi mevsimsel bandından belirgin kopmuyor; takvim etkisi nedeniyle zayıf görünen kalemler sert cezalandırılmadı."
    if below_count >= 2 and below_count >= above_count:
        return "Son dönem bazı ana kalemlerde hem kendi sezon bandının hem yakın trendin altında; zayıflık yalnız mevsimsellik ile açıklanamıyor."
    if above_count >= 2 and above_count > below_count:
        return "Bazı ana kalemler kendi tarihsel çeyrek bandının üzerinde; dönemsel güç sadece baz etkisine dayanmıyor."
    if mixed_count:
        return "Mevsimsellik etkisi karışık; bazı kalemler kendi dönem bandına yakınken bazıları trendden belirgin sapıyor."
    return "Mevsimsellik etkisi sınırlı; skor ağırlıkla yakın dönem operasyonel eğilimlerden türetildi."


def _overall_summary(overall_score: float, subscores: List[Dict[str, Any]]) -> str:
    best = max(subscores, key=lambda item: item["score"])
    worst = min(subscores, key=lambda item: item["score"])
    label = _overall_label(overall_score)
    return f"Genel görünüm {label.lower()}; en güçlü alan {best['label']}, en zayıf alan {worst['label']}."


def _watch_metrics(company_kind: str, ordered_subscores: List[Dict[str, Any]]) -> List[str]:
    result: List[str] = []
    watch_map = WATCH_METRICS.get(company_kind, WATCH_METRICS["generic"])
    for subscore in ordered_subscores:
        for label in watch_map.get(subscore["key"], []):
            if label not in result:
                result.append(label)
            if len(result) >= 4:
                return result
    return result


def _deterministic_commentary(
    scorecard: Dict[str, Any],
    company_kind: str,
) -> Dict[str, Any]:
    subscores = list(scorecard.get("subscores") or [])
    best = max(subscores, key=lambda item: item["score"])
    worst = min(subscores, key=lambda item: item["score"])
    ordered = sorted(subscores, key=lambda item: item["score"])
    headline = f"{scorecard['overall_label']} görünüm: {best['label']} önde, {worst['label']} izlenmeli."
    bullets = [
        f"{best['label']} skoru {best['score']}/10 seviyesinde. {best['summary']}",
        f"{worst['label']} skoru {worst['score']}/10 seviyesinde. {worst['summary']}",
        scorecard["seasonality_note"],
    ]
    risk_note = f"En belirgin kırılganlık {worst['label'].lower()} tarafında. Bir sonraki dönemde bu başlık teyit edilmeli."
    return {
        "headline": headline,
        "bullets": bullets,
        "risk_note": risk_note,
        "watch_metrics": _watch_metrics(company_kind, ordered),
    }


def _generic_bucket_signals(
    bucket_key: str,
    history_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    sales = _series_snapshot(history_context, "metrics", "satis_gelirleri")
    favok = _series_snapshot(history_context, "metrics", "favok")
    net_profit = _series_snapshot(history_context, "metrics", "net_kar")
    cfo = _series_snapshot(history_context, "metrics", "faaliyet_nakit_akisi")
    fcf = _series_snapshot(history_context, "metrics", "serbest_nakit_akisi")
    favok_margin = _series_snapshot(history_context, "ratios", "favok_marji")
    net_margin = _series_snapshot(history_context, "ratios", "net_kar_marji")
    roe = _series_snapshot(history_context, "ratios", "roe")
    current_ratio = _series_snapshot(history_context, "ratios", "cari_oran")
    debt_equity = _series_snapshot(history_context, "ratios", "net_borc_ozkaynak")
    cash_conversion = _series_snapshot(history_context, "ratios", "nakit_donusum")
    equity = _series_snapshot(history_context, "metrics", "ozkaynaklar")

    if bucket_key == "buyume":
        return (
            [
                item
                for item in [
                    _signal("Satış büyümesi", _score_centered(sales["yoy_change"], 35.0), 1.2),
                    _signal("FAVÖK büyümesi", _score_centered(favok["yoy_change"], 40.0), 1.0),
                    _signal("Net kar büyümesi", _score_centered(net_profit["yoy_change"], 45.0), 1.0),
                    _signal("Satışların mevsimsel konumu", _score_centered(sales["seasonal_delta"], 18.0), 0.7),
                    _signal("Net kar trendi", _score_centered(net_profit["trend_delta"], 25.0), 0.7),
                ]
                if item
            ],
            [sales["season_flag"], net_profit["season_flag"]],
        )
    if bucket_key == "karlilik":
        return (
            [
                item
                for item in [
                    _signal("FAVÖK marjı", _score_higher(favok_margin["current"], 4.0, 20.0), 1.0),
                    _signal("Net kar marjı", _score_higher(net_margin["current"], 0.0, 12.0), 1.1),
                    _signal("ROE", _score_higher(roe["current"], 4.0, 24.0), 0.8),
                    _signal("Net marjın mevsimsel sapması", _score_centered(net_margin["seasonal_delta"], 18.0), 0.6),
                ]
                if item
            ],
            [net_margin["season_flag"], favok_margin["season_flag"]],
        )
    if bucket_key == "bilanco":
        debt_season_score = None
        if debt_equity["seasonal_delta"] is not None:
            debt_season_score = _score_centered(-debt_equity["seasonal_delta"], 25.0)
        return (
            [
                item
                for item in [
                    _signal("Net borç / özkaynak", _score_lower(debt_equity["current"], 0.2, 2.5), 1.0),
                    _signal("Cari oran", _score_higher(current_ratio["current"], 0.8, 1.8), 0.9),
                    _signal("Özkaynak büyümesi", _score_centered(equity["yoy_change"], 30.0), 0.8),
                    _signal("Borç yapısının mevsimsel konumu", debt_season_score, 0.7),
                ]
                if item
            ],
            [debt_equity["season_flag"], current_ratio["season_flag"]],
        )
    return (
        [
            item
            for item in [
                _signal("Faaliyet nakit akışı", _score_sign(cfo["current"]), 0.9),
                _signal("Serbest nakit akışı", _score_sign(fcf["current"]), 1.1),
                _signal("Nakit dönüşümü", _score_higher(cash_conversion["current"], 0.3, 1.2), 1.0),
                _signal("Serbest nakit akışının mevsimsel konumu", _score_centered(fcf["seasonal_delta"], 25.0), 0.7),
            ]
            if item
        ],
        [fcf["season_flag"], cfo["season_flag"]],
    )


def _bank_bucket_signals(
    bucket_key: str,
    history_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    fees = _series_snapshot(history_context, "metrics", "net_ucret_komisyon_gelirleri")
    operating = _series_snapshot(history_context, "metrics", "net_faaliyet_kari")
    net_profit = _series_snapshot(history_context, "metrics", "net_kar")
    loans = _series_snapshot(history_context, "metrics", "krediler")
    deposits = _series_snapshot(history_context, "metrics", "mevduatlar")
    provisions_ratio = _series_snapshot(history_context, "ratios", "karsilik_kredi_orani")
    loan_deposit = _series_snapshot(history_context, "ratios", "kredi_mevduat_orani")
    roe = _series_snapshot(history_context, "ratios", "roe")
    equity = _series_snapshot(history_context, "metrics", "ozkaynaklar")
    financial_assets = _series_snapshot(history_context, "metrics", "finansal_varliklar_net")

    if bucket_key == "buyume":
        return (
            [
                item
                for item in [
                    _signal("Ücret komisyon büyümesi", _score_centered(fees["yoy_change"], 30.0), 1.0),
                    _signal("Faaliyet karı büyümesi", _score_centered(operating["yoy_change"], 35.0), 1.0),
                    _signal("Net kar büyümesi", _score_centered(net_profit["yoy_change"], 40.0), 1.0),
                    _signal("Kredi büyümesinin mevsimsel konumu", _score_centered(loans["seasonal_delta"], 15.0), 0.7),
                ]
                if item
            ],
            [loans["season_flag"], net_profit["season_flag"]],
        )
    if bucket_key == "karlilik":
        return (
            [
                item
                for item in [
                    _signal("ROE", _score_higher(roe["current"], 6.0, 24.0), 1.0),
                    _signal("Net kar eğilimi", _score_centered(net_profit["trend_delta"], 25.0), 0.9),
                    _signal("Faaliyet karı eğilimi", _score_centered(operating["trend_delta"], 25.0), 0.9),
                    _signal("Net karın işaret kalitesi", _score_sign(net_profit["current"]), 0.8),
                ]
                if item
            ],
            [net_profit["season_flag"], operating["season_flag"]],
        )
    if bucket_key == "bilanco":
        return (
            [
                item
                for item in [
                    _signal("Özkaynak büyümesi", _score_centered(equity["yoy_change"], 20.0), 1.0),
                    _signal("Karşılık / kredi", _score_lower(provisions_ratio["current"], 0.01, 0.08), 1.0),
                    _signal("Finansal varlık eğilimi", _score_centered(financial_assets["trend_delta"], 20.0), 0.7),
                    _signal("Kredilerin sezon bandı", _score_centered(loans["seasonal_delta"], 15.0), 0.6),
                ]
                if item
            ],
            [loans["season_flag"], equity["season_flag"]],
        )
    return (
        [
            item
            for item in [
                _signal("Kredi / mevduat dengesi", _score_band(loan_deposit["current"], 0.55, 0.8, 1.15, 1.45), 1.2),
                _signal("Mevduat büyümesi", _score_centered(deposits["yoy_change"], 18.0), 0.9),
                _signal("Finansal varlıkların mevsimsel konumu", _score_centered(financial_assets["seasonal_delta"], 18.0), 0.7),
                _signal("Mevduat trendi", _score_centered(deposits["trend_delta"], 15.0), 0.7),
            ]
            if item
        ],
        [deposits["season_flag"], financial_assets["season_flag"]],
    )


def _insurance_bucket_signals(
    bucket_key: str,
    history_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    gross_premium = _series_snapshot(history_context, "metrics", "prim_uretimi")
    net_premium = _series_snapshot(history_context, "metrics", "alinan_net_primler")
    technical_income = _series_snapshot(history_context, "metrics", "teknik_gelirler")
    technical_balance = _series_snapshot(history_context, "metrics", "teknik_denge")
    net_profit = _series_snapshot(history_context, "metrics", "net_kar")
    equity = _series_snapshot(history_context, "metrics", "ozkaynaklar")
    cash_like = _series_snapshot(history_context, "metrics", "nakit_benzeri_finansal_varliklar")
    tech_balance_margin = _series_snapshot(history_context, "ratios", "teknik_denge_marji")
    cash_reserve_ratio = _series_snapshot(history_context, "ratios", "nakit_karsilik_orani")
    equity_reserve_ratio = _series_snapshot(history_context, "ratios", "ozkaynak_karsilik_orani")
    debt_equity = _series_snapshot(history_context, "ratios", "borc_ozkaynak_orani")
    roe = _series_snapshot(history_context, "ratios", "roe")

    if bucket_key == "buyume":
        return (
            [
                item
                for item in [
                    _signal("Prim üretimi büyümesi", _score_centered(gross_premium["yoy_change"], 25.0), 1.0),
                    _signal("Net prim büyümesi", _score_centered(net_premium["yoy_change"], 25.0), 1.0),
                    _signal("Teknik gelir büyümesi", _score_centered(technical_income["yoy_change"], 25.0), 0.9),
                    _signal("Prim üretiminin mevsimsel konumu", _score_centered(gross_premium["seasonal_delta"], 15.0), 0.7),
                ]
                if item
            ],
            [gross_premium["season_flag"], net_premium["season_flag"]],
        )
    if bucket_key == "karlilik":
        return (
            [
                item
                for item in [
                    _signal("Teknik denge marjı", _score_higher(tech_balance_margin["current"], 0.0, 15.0), 1.0),
                    _signal("ROE", _score_higher(roe["current"], 4.0, 22.0), 0.8),
                    _signal("Net kar büyümesi", _score_centered(net_profit["yoy_change"], 35.0), 1.0),
                    _signal("Net kar trendi", _score_centered(net_profit["trend_delta"], 25.0), 0.7),
                ]
                if item
            ],
            [net_profit["season_flag"], technical_balance["season_flag"]],
        )
    if bucket_key == "bilanco":
        return (
            [
                item
                for item in [
                    _signal("Özkaynak büyümesi", _score_centered(equity["yoy_change"], 20.0), 1.0),
                    _signal("Özkaynak / teknik karşılık", _score_higher(equity_reserve_ratio["current"], 0.3, 1.2), 0.9),
                    _signal("Borç / özkaynak", _score_lower(debt_equity["current"], 0.2, 2.0), 0.9),
                    _signal("Özkaynakların mevsimsel konumu", _score_centered(equity["seasonal_delta"], 15.0), 0.6),
                ]
                if item
            ],
            [equity["season_flag"], debt_equity["season_flag"]],
        )
    return (
        [
            item
            for item in [
                _signal("Nakit / teknik karşılık", _score_higher(cash_reserve_ratio["current"], 0.4, 1.2), 1.1),
                _signal("Teknik denge işaret kalitesi", _score_sign(technical_balance["current"]), 0.9),
                _signal("Nakit benzeri varlık eğilimi", _score_centered(cash_like["trend_delta"], 20.0), 0.8),
                _signal("Teknik dengenin mevsimsel konumu", _score_centered(technical_balance["seasonal_delta"], 18.0), 0.7),
            ]
            if item
        ],
        [technical_balance["season_flag"], cash_like["season_flag"]],
    )


def _bucket_signals(company_kind: str, bucket_key: str, history_context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    if company_kind == "bank":
        return _bank_bucket_signals(bucket_key, history_context)
    if company_kind == "insurance":
        return _insurance_bucket_signals(bucket_key, history_context)
    return _generic_bucket_signals(bucket_key, history_context)


def compute_base_analysis(history_context: Dict[str, Any]) -> Dict[str, Any]:
    company_kind = str(history_context.get("company_kind") or "generic").strip().lower()
    if company_kind not in COMPANY_KINDS:
        company_kind = "generic"

    bucket_labels = BUCKET_LABELS.get(company_kind, BUCKET_LABELS["generic"])
    subscores: List[Dict[str, Any]] = []
    debug_lines = [
        f"score history_quarters={len(history_context.get('quarters') or [])}",
        f"score company_kind={company_kind}",
    ]
    seasonality_flags: List[str] = []

    for bucket_key in SUBSCORE_KEYS:
        signals, flags = _bucket_signals(company_kind, bucket_key, history_context)
        score = _round1(_weighted_score(signals))
        label = bucket_labels.get(bucket_key, bucket_key)
        subscores.append(
            {
                "key": bucket_key,
                "label": label,
                "score": score,
                "summary": _bucket_summary(label, score, signals),
            }
        )
        seasonality_flags.extend([flag for flag in flags if flag and flag != "unknown"])
        signal_debug = ", ".join(f"{item['label']}={item['score']:.1f}" for item in signals[:4]) or "veri_yok"
        debug_lines.append(f"base {bucket_key}={score:.1f} signals={signal_debug}")

    overall_score = _round1(mean([item["score"] for item in subscores]) if subscores else 5.0)
    scorecard = {
        "overall_score": overall_score,
        "overall_label": _overall_label(overall_score),
        "summary": _overall_summary(overall_score, subscores),
        "seasonality_note": _seasonality_note(seasonality_flags),
        "score_source": SCORE_SOURCE_DETERMINISTIC_ONLY,
        "subscores": subscores,
    }
    commentary = _deterministic_commentary(scorecard, company_kind)
    return {
        "company_kind": company_kind,
        "scorecard": scorecard,
        "headline": commentary["headline"],
        "bullets": commentary["bullets"],
        "risk_note": commentary["risk_note"],
        "watch_metrics": commentary["watch_metrics"],
        "debug_lines": debug_lines,
    }


def merge_scorecard_with_adjustments(
    base_scorecard: Dict[str, Any],
    ai_adjustments: Dict[str, Any],
    *,
    score_source: str,
) -> Dict[str, Any]:
    overall_adjustment = float(ai_adjustments.get("overall_adjustment") or 0.0)
    bounded_overall_adjustment = _clamp(overall_adjustment, -0.5, 0.5)

    ai_subscores = {
        str(item.get("key") or "").strip(): item
        for item in (ai_adjustments.get("subscores") or [])
        if isinstance(item, dict) and str(item.get("key") or "").strip()
    }

    merged_subscores: List[Dict[str, Any]] = []
    for item in list(base_scorecard.get("subscores") or []):
        raw_adjustment = ai_subscores.get(item["key"], {}).get("adjustment")
        adjustment = float(raw_adjustment or 0.0) if isinstance(raw_adjustment, (int, float)) else 0.0
        bounded_adjustment = _clamp(adjustment, -1.0, 1.0)
        ai_summary = str(ai_subscores.get(item["key"], {}).get("summary") or "").strip()
        merged_subscores.append(
            {
                **item,
                "score": _round1(float(item["score"]) + bounded_adjustment),
                "summary": ai_summary or item["summary"],
            }
        )

    overall_score = _round1(float(base_scorecard["overall_score"]) + bounded_overall_adjustment)
    summary = str(ai_adjustments.get("summary") or "").strip() or base_scorecard["summary"]
    seasonality_note = str(ai_adjustments.get("seasonality_note") or "").strip() or base_scorecard["seasonality_note"]
    return {
        **base_scorecard,
        "overall_score": overall_score,
        "overall_label": _overall_label(overall_score),
        "summary": summary,
        "seasonality_note": seasonality_note,
        "score_source": score_source,
        "subscores": merged_subscores,
    }
