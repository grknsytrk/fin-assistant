from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Callable, Dict, List, Optional, Tuple

SCORE_SOURCE_DETERMINISTIC_ONLY = "deterministic_only"
SCORE_SOURCE_AI_ADJUSTED = "ai_adjusted"
SCORE_SOURCE_AI_FAILED_FALLBACK = "ai_failed_fallback"

SUBSCORE_KEYS = ("buyume", "karlilik", "bilanco", "nakit_akisi")
COMPANY_KINDS = {"generic", "bank", "insurance"}

# AI may only nudge the deterministic score. The overall score is recomputed
# from the adjusted subscores, so the overall adjustment is intentionally small.
AI_SUBSCORE_ADJUSTMENT_LIMIT = 0.8

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

# Overall score is a weighted mean, not a flat mean: profitability and its
# quality matter more than a single noisy quarter of cash flow.
BUCKET_WEIGHTS = {
    "generic": {"buyume": 0.25, "karlilik": 0.30, "bilanco": 0.20, "nakit_akisi": 0.25},
    "bank": {"buyume": 0.25, "karlilik": 0.35, "bilanco": 0.25, "nakit_akisi": 0.15},
    "insurance": {"buyume": 0.25, "karlilik": 0.35, "bilanco": 0.25, "nakit_akisi": 0.15},
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

SCORE_BANDS: Tuple[Tuple[float, str, str], ...] = (
    (8.5, "Çok Güçlü", "çok güçlü"),
    (7.5, "Güçlü", "güçlü"),
    (6.5, "İyi", "iyi"),
    (5.5, "Dengeli", "dengeli"),
    (4.5, "Karışık", "karışık"),
    (3.0, "Zayıf", "zayıf"),
    (0.0, "Çok Zayıf", "çok zayıf"),
)

PROFIT_GROWTH_REGIMES = {"kar_artti", "kar_azaldi", "kar_yatay"}
NON_OPERATING_QUALITY_CODES = {"faaliyet_disi_kar_artisi", "faaliyet_zarari_net_kar"}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round1(value: float) -> float:
    return round(_clamp(float(value), 0.0, 10.0) + 1e-9, 1)


def _pct_change(current: Optional[float], base: Optional[float]) -> Optional[float]:
    if current is None or base is None or base == 0:
        return None
    return ((current - base) / abs(base)) * 100.0


def _safe_ratio(numerator: Optional[float], denominator: Optional[float], scale: float = 1.0) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator) * scale


def _score_centered(change_pct: Optional[float], scale: float) -> Optional[float]:
    if change_pct is None:
        return None
    return _clamp(5.0 + (change_pct / scale) * 5.0, 0.0, 10.0)


def _score_growth(change: Optional[float], scale: float) -> Optional[float]:
    """Smooth, saturating growth score: 0 -> 5, +scale -> ~8.8, -scale -> ~1.2.

    Unlike a linear clamp, +60% and +300% still score differently and a single
    extreme value does not jump straight to 0 or 10.
    """
    if change is None:
        return None
    return _clamp(5.0 + 5.0 * math.tanh(change / scale), 0.0, 10.0)


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
        distance = abs(value - midpoint)
        span = max((ideal_high - ideal_low) / 2.0, 1e-9)
        return _clamp(9.5 - (distance / span) * 1.5, 8.0, 10.0)
    if value < ideal_low:
        return _score_higher(value, soft_low, ideal_low)
    return _score_lower(value, ideal_high, soft_high)


def _shrink(score: Optional[float], factor: float) -> Optional[float]:
    if score is None:
        return None
    return 5.0 + (score - 5.0) * factor


def _score_margin_level(
    margin: Optional[float],
    *,
    abs_low: float,
    abs_high: float,
    hist_median: Optional[float] = None,
    roe: Optional[float] = None,
) -> Optional[float]:
    """Score a margin level while respecting structurally low-margin, high-turnover models."""
    abs_score = _score_higher(margin, abs_low, abs_high)
    if abs_score is None or margin is None or margin <= 0:
        return abs_score
    if hist_median is None or hist_median <= 0:
        return abs_score
    hist_low = max(0.4, hist_median * 0.55)
    hist_high = max(hist_median * 1.25, hist_median + 1.2)
    hist_score = _score_higher(margin, hist_low, hist_high)
    if hist_score is None:
        return abs_score
    if (roe is not None and roe >= 10.0) or margin >= hist_median:
        return max(abs_score, hist_score)
    if hist_median >= 1.5:
        return max(abs_score, 0.30 * abs_score + 0.70 * hist_score)
    return abs_score


def _margin_pp_scale(
    hist_median: Optional[float],
    current_margin: Optional[float],
    *,
    default_scale: float,
    min_scale: float,
) -> float:
    """Scale pp-change thresholds proportionally for low-margin businesses."""
    ref = hist_median if (hist_median is not None and hist_median > 0) else current_margin
    if ref is None or ref <= 0:
        return default_scale
    return _clamp(ref * 0.30, min_scale, default_scale)


# ---------------------------------------------------------------------------
# Display helpers (Turkish formatting for signal values / flags)
# ---------------------------------------------------------------------------


def _fmt_amount(value: Optional[float]) -> str:
    if value is None:
        return "-"
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.2f} mr".replace(".", ",")
    if magnitude >= 1e6:
        return f"{value / 1e6:.1f} mn".replace(".", ",")
    return f"{value:,.0f}".replace(",", ".")


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    digits = 0 if abs(value) >= 10 else 1
    return f"{sign}%{abs(value):.{digits}f}".replace(".", ",")


def _fmt_level_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"%{value:.1f}".replace(".", ",")


def _fmt_pp(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f} puan".replace(".", ",")


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x".replace(".", ",")


def _fmt_change(current: Optional[float], base: Optional[float]) -> str:
    if current is None and base is None:
        return "-"
    text = f"{_fmt_amount(base)} → {_fmt_amount(current)}"
    if current is not None and base is not None and current > 0 and base > 0:
        text += f" ({_fmt_pct(_pct_change(current, base))})"
    return text


def _score_band_entry(score: float) -> Tuple[float, str, str]:
    for entry in SCORE_BANDS:
        if score >= entry[0]:
            return entry
    return SCORE_BANDS[-1]


def _overall_label(score: float) -> str:
    return _score_band_entry(score)[1]


def _bucket_tone(score: float) -> str:
    return _score_band_entry(score)[2]


def _confidence_label(coverage: float) -> str:
    if coverage >= 0.75:
        return "yüksek"
    if coverage >= 0.5:
        return "orta"
    return "düşük"


# ---------------------------------------------------------------------------
# Quarter history access
# ---------------------------------------------------------------------------


class _History:
    """Indexes history quarters by (year, quarter) so YoY/TTM/YTD lookups are
    calendar based instead of positional (missing quarters stay missing)."""

    def __init__(self, history_context: Dict[str, Any]):
        raw_rows = [row for row in (history_context.get("quarters") or []) if isinstance(row, dict)]
        periods = [self._int(row.get("period")) for row in raw_rows]
        month_based = any(period > 4 for period in periods)
        self._rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for row in raw_rows:
            year = self._int(row.get("year"))
            period = self._int(row.get("period"))
            if month_based:
                if period % 3 != 0:
                    continue
                quarter = period // 3
            else:
                quarter = period
            if year <= 0 or quarter not in (1, 2, 3, 4):
                continue
            self._rows[(year, quarter)] = row
        self.keys = sorted(self._rows)
        self.latest: Optional[Tuple[int, int]] = self.keys[-1] if self.keys else None

    @staticmethod
    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def shift(key: Tuple[int, int], back: int) -> Tuple[int, int]:
        year, quarter = key
        index = year * 4 + (quarter - 1) - back
        return (index // 4, index % 4 + 1)

    @staticmethod
    def label(key: Optional[Tuple[int, int]]) -> str:
        if key is None:
            return "-"
        return f"{key[0]} Q{key[1]}"

    def value(self, key: Optional[Tuple[int, int]], source: str, metric: str) -> Optional[float]:
        if key is None:
            return None
        row = self._rows.get(key)
        if not row:
            return None
        bucket = row.get(source)
        if not isinstance(bucket, dict):
            return None
        return _num(bucket.get(metric))

    def window_sum(self, end_key: Optional[Tuple[int, int]], count: int, source: str, metric: str) -> Optional[float]:
        if end_key is None or count <= 0:
            return None
        total = 0.0
        for back in range(count):
            value = self.value(self.shift(end_key, back), source, metric)
            if value is None:
                return None
            total += value
        return total

    def ytd_sum(self, end_key: Optional[Tuple[int, int]], source: str, metric: str) -> Optional[float]:
        if end_key is None:
            return None
        return self.window_sum(end_key, end_key[1], source, metric)

    def prior_keys(self, limit: int = 8) -> List[Tuple[int, int]]:
        if self.latest is None:
            return []
        return self.keys[:-1][-limit:]


def _flow(h: _History, metric: str, source: str = "metrics") -> Dict[str, Optional[float]]:
    key = h.latest
    if key is None:
        return {name: None for name in ("cur", "prev", "ly", "ttm", "ttm_ly", "ytd", "ytd_ly", "yoy", "qoq", "ttm_yoy", "ytd_yoy")}
    ly_key = h.shift(key, 4)
    cur = h.value(key, source, metric)
    prev = h.value(h.shift(key, 1), source, metric)
    ly = h.value(ly_key, source, metric)
    ttm = h.window_sum(key, 4, source, metric)
    ttm_ly = h.window_sum(ly_key, 4, source, metric)
    ytd = h.ytd_sum(key, source, metric)
    ytd_ly = h.ytd_sum(ly_key, source, metric)
    return {
        "cur": cur,
        "prev": prev,
        "ly": ly,
        "ttm": ttm,
        "ttm_ly": ttm_ly,
        "ytd": ytd,
        "ytd_ly": ytd_ly,
        "yoy": _pct_change(cur, ly),
        "qoq": _pct_change(cur, prev),
        "ttm_yoy": _pct_change(ttm, ttm_ly),
        "ytd_yoy": _pct_change(ytd, ytd_ly),
    }


def _season(h: _History, metric: str, source: str = "metrics") -> Dict[str, Any]:
    key = h.latest
    if key is None:
        return {"seasonal_delta": None, "trend_delta": None, "season_flag": "unknown"}
    current = h.value(key, source, metric)
    same_period = [h.value(h.shift(key, 4 * idx), source, metric) for idx in (1, 2, 3)]
    same_period_values = [value for value in same_period if value is not None]
    trailing = [h.value(h.shift(key, idx), source, metric) for idx in (1, 2, 3, 4)]
    trailing_values = [value for value in trailing if value is not None]
    seasonal_delta = _pct_change(current, median(same_period_values) if same_period_values else None)
    trend_delta = _pct_change(current, mean(trailing_values) if trailing_values else None)

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
    return {"seasonal_delta": seasonal_delta, "trend_delta": trend_delta, "season_flag": season_flag}


def _margin(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return (numerator / denominator) * 100.0


def _margin_for_period(
    h: _History,
    period: Optional[Tuple[int, int]],
    numerator_key: str,
    denominator_key: str,
) -> Optional[float]:
    return _margin(
        h.value(period, "metrics", numerator_key),
        h.value(period, "metrics", denominator_key),
    )


def _margin_delta_pp(
    h: _History,
    period: Optional[Tuple[int, int]],
    numerator_key: str,
    denominator_key: str,
    *,
    periods_back: int,
) -> Optional[float]:
    current = _margin_for_period(h, period, numerator_key, denominator_key)
    comparison_period = _History.shift(period, periods_back) if period is not None else None
    comparison = _margin_for_period(h, comparison_period, numerator_key, denominator_key)
    return current - comparison if current is not None and comparison is not None else None


def _margin_trend_label(qoq_change_pp: Optional[float]) -> str:
    if qoq_change_pp is None:
        return "veri_yok"
    if qoq_change_pp >= 0.3:
        return "ceyreklik_iyilesme"
    if qoq_change_pp <= -0.3:
        return "ceyreklik_daralma"
    return "ceyreklik_yatay"


def _margin_history(h: _History, numerator_key: str, denominator_key: str, limit: int = 8) -> List[float]:
    values: List[float] = []
    for key in h.prior_keys(limit):
        margin = _margin(h.value(key, "metrics", numerator_key), h.value(key, "metrics", denominator_key))
        if margin is not None:
            values.append(margin)
    return values


# ---------------------------------------------------------------------------
# Scenario logic shared by all company kinds
# ---------------------------------------------------------------------------


def _profit_regime(current: Optional[float], base: Optional[float]) -> Optional[str]:
    if current is None or base is None:
        return None
    if base > 0 and current > 0:
        change = (current - base) / base
        if change >= 0.05:
            return "kar_artti"
        if change <= -0.05:
            return "kar_azaldi"
        return "kar_yatay"
    if base <= 0 < current:
        return "zarardan_kara"
    if current <= 0 < base:
        return "kardan_zarara"
    if current > base:
        return "zarar_daraldi"
    if current < base:
        return "zarar_buyudu"
    return "zarar_yatay"


def _score_profit_change(
    current: Optional[float],
    base: Optional[float],
    ref_current: Optional[float],
    ref_base: Optional[float],
    *,
    pct_scale: float,
    delta_scale: float,
    materiality: float,
) -> Tuple[Optional[float], Optional[str]]:
    """Score a profit line change with sign-aware regimes.

    - Positive -> positive: percentage growth, blended with the change relative
      to the reference (sales or equity) when the base was immaterial.
    - Sign changes / losses: percentage is meaningless, so we score the size of
      the change relative to the reference and keep each regime in its band.
    """
    regime = _profit_regime(current, base)
    if regime is None or current is None or base is None:
        return None, None

    delta_score: Optional[float] = None
    if ref_current is not None and ref_current > 0:
        delta_pp = ((current - base) / ref_current) * 100.0
        delta_score = _score_growth(delta_pp, delta_scale)

    if regime in PROFIT_GROWTH_REGIMES:
        growth_score = _score_growth(_pct_change(current, base), pct_scale)
        if growth_score is None:
            growth_score = 5.0
        confidence = 1.0
        if ref_base is not None and ref_base > 0:
            base_margin = (base / ref_base) * 100.0
            confidence = _clamp(base_margin / materiality, 0.3, 1.0)
        if delta_score is None:
            return 5.0 + (growth_score - 5.0) * confidence, regime
        return growth_score * confidence + delta_score * (1.0 - confidence), regime
    if regime == "zarardan_kara":
        # A sign change is positive, but its score must still reflect how large
        # the improvement is relative to the business. Without a usable
        # denominator, keep it neutral instead of granting an automatic uplift.
        return (_clamp(delta_score, 0.0, 9.5) if delta_score is not None else 5.0), regime
    if regime == "kardan_zarara":
        return _clamp(delta_score if delta_score is not None else 2.0, 0.5, 3.0), regime
    if regime == "zarar_daraldi":
        return _clamp(delta_score if delta_score is not None else 5.0, 4.0, 6.0), regime
    if regime == "zarar_buyudu":
        return _clamp(delta_score if delta_score is not None else 2.0, 0.5, 3.0), regime
    return 3.0, regime


def _earnings_quality(
    net: Dict[str, Optional[float]],
    operating: Dict[str, Optional[float]],
    *,
    ebitda_current: Optional[float],
    ebitda_ly: Optional[float] = None,
    reference_current: Optional[float],
    materiality_pp: float,
) -> Dict[str, Any]:
    """Checks whether the net profit change is backed by operating profit."""
    result: Dict[str, Any] = {
        "score": None,
        "code": None,
        "explained_share": None,
        "net_delta": None,
        "operating_delta": None,
        "non_operating_current": None,
        "net_exceeds_ebitda": False,
    }
    net_cur, net_ly = net.get("cur"), net.get("ly")
    op_cur, op_ly = operating.get("cur"), operating.get("ly")
    if net_cur is None or op_cur is None:
        return result

    result["non_operating_current"] = net_cur - op_cur
    score: Optional[float] = None
    code: Optional[str] = None

    if net_ly is not None and op_ly is not None:
        net_delta = net_cur - net_ly
        op_delta = op_cur - op_ly
        result["net_delta"] = net_delta
        result["operating_delta"] = op_delta
        if reference_current is not None and reference_current > 0:
            material = abs(net_delta) / reference_current * 100.0 >= materiality_pp
        else:
            material = abs(net_delta) >= 0.05 * max(abs(net_ly), 1.0)

        if not material:
            score, code = 6.0, "degisim_sinirli"
        elif net_delta > 0:
            explained = op_delta / net_delta
            result["explained_share"] = explained
            if reference_current is not None and reference_current > 0:
                op_material = abs(op_delta) / reference_current * 100.0 >= materiality_pp
            else:
                op_material = abs(op_delta) >= 0.05 * max(abs(op_ly or 0.0), 1.0)
            ebitda_explained: Optional[float] = None
            if ebitda_current is not None and ebitda_ly is not None:
                ebitda_explained = (ebitda_current - ebitda_ly) / net_delta
            op_yoy_pct = ((op_cur - op_ly) / op_ly * 100.0) if (op_ly is not None and op_ly > 0) else None
            # A material increase in operating profit is positive evidence
            # even when net profit rises faster because of other items. Treat
            # its share as context, not as an automatic quality penalty.
            if op_delta > 0 and op_material and explained >= 0.6:
                score, code = 9.0, "faaliyet_destekli"
            elif (
                op_delta > 0
                and op_material
                and ebitda_explained is not None
                and ebitda_explained >= 0.6
                and (op_yoy_pct is None or op_yoy_pct >= 25.0)
            ):
                score, code = 8.5, "kismen_faaliyet_destekli"
            elif op_delta > 0 and op_material and op_yoy_pct is not None and op_yoy_pct >= 35.0:
                score, code = 8.2, "kismen_faaliyet_destekli"
            elif op_delta > 0 and op_material:
                score, code = 7.5, "kismen_faaliyet_destekli"
            elif op_delta >= 0:
                score, code = 4.0, "faaliyet_disi_kar_artisi"
            else:
                score, code = 3.0, "faaliyet_disi_kar_artisi"
        else:
            if op_delta > 0:
                score, code = 6.0, "faaliyet_guclu_net_zayif"
            else:
                result["explained_share"] = op_delta / net_delta
                score, code = 5.0, "faaliyet_kaynakli_dusus"

    if net_cur > 0 and op_cur <= 0:
        score, code = 2.5, "faaliyet_zarari_net_kar"
    if ebitda_current is not None and ebitda_current > 0 and net_cur > ebitda_current:
        # Net profit above EBITDA means the current level leans on
        # non-operating items. Keep this as a quality caution, but do not let
        # it erase a material improvement in operating profit.
        result["net_exceeds_ebitda"] = True
        ceiling = 7.5 if code == "faaliyet_destekli" else 6.5 if code == "kismen_faaliyet_destekli" else 4.0
        score = min(score if score is not None else 5.0, ceiling)
        if code is None:
            code = "net_kar_favoku_asiyor"

    result["score"] = score
    result["code"] = code
    return result


def _flag(code: str, tone: str, title: str, detail: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    return {"code": code, "tone": tone, "title": title, "detail": detail, "bucket": bucket}


def _spec(label: str, score: Optional[float], weight: float, value: str = "", *, optional: bool = False) -> Dict[str, Any]:
    return {"label": label, "score": score, "weight": weight, "value": value, "optional": optional}


def _empty_guards() -> Dict[str, Dict[str, float]]:
    return {
        key: {"min": -AI_SUBSCORE_ADJUSTMENT_LIMIT, "max": AI_SUBSCORE_ADJUSTMENT_LIMIT}
        for key in SUBSCORE_KEYS
    }


def _tighten(guards: Dict[str, Dict[str, float]], key: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> None:
    guard = guards[key]
    if minimum is not None:
        guard["min"] = max(guard["min"], minimum)
    if maximum is not None:
        guard["max"] = min(guard["max"], maximum)
    if guard["min"] > guard["max"]:
        guard["min"] = guard["max"]


def _profit_regime_flags(
    regime: Optional[str],
    net: Dict[str, Optional[float]],
    *,
    period_label: str,
    ly_label: str,
    base_margin: Optional[float],
    low_base_threshold: float,
) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    change_text = _fmt_change(net.get("cur"), net.get("ly"))
    yoy = net.get("yoy")
    if regime == "kar_artti" and yoy is not None and yoy >= 30:
        flags.append(_flag(
            "guclu_net_kar_artisi", "positive", "Güçlü net kâr artışı",
            f"{period_label} net kârı {ly_label} ile kıyasla {change_text}.", "karlilik",
        ))
    elif regime == "kar_artti":
        flags.append(_flag(
            "net_kar_artisi", "positive", "Net kâr arttı",
            f"{period_label} net kârı {ly_label} ile kıyasla {change_text}.", "karlilik",
        ))
    elif regime == "kar_azaldi" and yoy is not None and yoy <= -20:
        flags.append(_flag(
            "net_kar_dususu", "negative", "Net kâr belirgin geriledi",
            f"{period_label} net kârı {ly_label} ile kıyasla {change_text}.", "karlilik",
        ))
    elif regime == "kar_azaldi":
        flags.append(_flag(
            "net_kar_hafif_dusus", "warning", "Net kâr hafif geriledi",
            f"{period_label} net kârı {ly_label} ile kıyasla {change_text}.", "karlilik",
        ))
    elif regime == "zarardan_kara":
        flags.append(_flag(
            "zarardan_kara", "positive", "Zarardan kâra geçiş",
            f"{ly_label} döneminde {_fmt_amount(net.get('ly'))} olan sonuç {period_label} döneminde {_fmt_amount(net.get('cur'))} kâra döndü; yüzde değişim anlamlı değil.",
            "karlilik",
        ))
    elif regime == "kardan_zarara":
        flags.append(_flag(
            "kardan_zarara", "negative", "Kârdan zarara geçiş",
            f"{ly_label} döneminde {_fmt_amount(net.get('ly'))} kâr varken {period_label} döneminde {_fmt_amount(net.get('cur'))} sonuç oluştu.",
            "karlilik",
        ))
    elif regime == "zarar_daraldi":
        flags.append(_flag(
            "zarar_daraldi", "warning", "Zarar daraldı",
            f"Net zarar {_fmt_amount(net.get('ly'))} seviyesinden {_fmt_amount(net.get('cur'))} seviyesine geriledi; iyileşme var ama şirket hâlâ zararda.",
            "karlilik",
        ))
    elif regime == "zarar_buyudu":
        flags.append(_flag(
            "zarar_buyudu", "negative", "Zarar büyüdü",
            f"Net zarar {_fmt_amount(net.get('ly'))} seviyesinden {_fmt_amount(net.get('cur'))} seviyesine çıktı.",
            "karlilik",
        ))

    if regime == "kar_artti" and base_margin is not None and base_margin < low_base_threshold:
        flags.append(_flag(
            "dusuk_baz", "warning", "Düşük baz etkisi",
            f"Geçen yılın aynı çeyreğindeki kâr çok düşük bir bazdaydı; {_fmt_pct(yoy)} artış ekonomik büyümeyi abartabilir, mutlak tutar artışı daha anlamlı.",
            "buyume",
        ))

    ytd_yoy = net.get("ytd_yoy")
    quarter_change_comparable = _both_positive(net.get("cur"), net.get("ly"))
    ytd_change_comparable = _both_positive(net.get("ytd"), net.get("ytd_ly"))
    if quarter_change_comparable and ytd_change_comparable and yoy is not None and ytd_yoy is not None:
        if yoy >= 20 and ytd_yoy < 0:
            flags.append(_flag(
                "ceyrek_guclu_ytd_zayif", "neutral", "Son çeyrek güçlü, yıl toplamı hâlâ geride",
                f"Son çeyrek net kârı {_fmt_pct(yoy)} değişirken yıl başından beri toplam {_fmt_pct(ytd_yoy)}.",
                "buyume",
            ))
        elif yoy <= -20 and ytd_yoy > 0:
            flags.append(_flag(
                "ceyrek_zayif_ytd_guclu", "neutral", "Son çeyrek zayıf, yıl toplamı hâlâ önde",
                f"Son çeyrek net kârı {_fmt_pct(yoy)} değişirken yıl başından beri toplam {_fmt_pct(ytd_yoy)}.",
                "buyume",
            ))
    return flags


def _quality_flags(quality: Dict[str, Any], *, operating_label: str, period_label: str) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    code = quality.get("code")
    net_delta = quality.get("net_delta")
    op_delta = quality.get("operating_delta")
    explained = quality.get("explained_share")
    delta_text = ""
    if net_delta is not None and op_delta is not None:
        delta_text = f"Net kâr {_fmt_amount(net_delta)} değişirken {operating_label} {_fmt_amount(op_delta)} değişti."
    if code == "faaliyet_destekli":
        share = f" (artışın yaklaşık %{min(explained, 9.99) * 100:.0f}'i)" if explained is not None else ""
        flags.append(_flag(
            "faaliyet_destekli", "positive", "Kâr artışı faaliyetlerle destekleniyor",
            f"{delta_text} Net kâr artışını büyük ölçüde {operating_label} artışı açıklıyor{share}.".strip(),
            "karlilik",
        ))
    elif code == "kismen_faaliyet_destekli":
        share = f" ({operating_label} artışı net kâr artışının yaklaşık %{min(explained, 9.99) * 100:.0f}'ine denk geliyor)" if explained is not None else ""
        flags.append(_flag(
            "kismen_faaliyet_destekli", "neutral", f"{operating_label.capitalize()} arttı",
            f"{delta_text} {operating_label.capitalize()} artışı net kâr artışına katkı sağlıyor{share}. Net kârdaki ek artışın niteliği alt kalem dökümü olmadan değerlendirilemez; bu fark tek başına faaliyet performansına eksi yazılmaz.".strip(),
            "karlilik",
        ))
    elif code == "faaliyet_disi_kar_artisi":
        flags.append(_flag(
            "faaliyet_disi_kar_artisi", "warning", f"Net kâr artışını {operating_label} desteklemiyor",
            f"{delta_text} Faaliyet kârı net kâr artışını açıklamıyor. Alt kalem dökümü olmadığı için artışın kaynağı belirlenemiyor; sürdürülebilirlik izlenmeli.".strip(),
            "karlilik",
        ))
    elif code == "faaliyet_guclu_net_zayif":
        flags.append(_flag(
            "faaliyet_guclu_net_zayif", "neutral", "Faaliyet kârı artarken net kâr geriledi",
            f"{delta_text} {operating_label} artarken net kâr geriledi; verideki kalemler net kârdaki baskının kaynağını göstermiyor.".strip(),
            "karlilik",
        ))
    elif code == "faaliyet_zarari_net_kar":
        flags.append(_flag(
            "faaliyet_zarari_net_kar", "warning", "Faaliyet zararına rağmen net kâr",
            f"{period_label} döneminde {operating_label} negatifken net kâr pozitif. Net kârı oluşturan diğer kalemlerin dökümü verilmediği için kaynak belirlenemiyor.",
            "karlilik",
        ))
    if quality.get("net_exceeds_ebitda"):
        flags.append(_flag(
            "net_kar_favoku_asiyor", "warning", "Net kâr FAVÖK'ü aşıyor",
            "Net kâr FAVÖK'ü aşıyor; bu farkın kaynağını belirlemek için gereken gelir/gider alt kalemleri veride bulunmuyor.",
            "karlilik",
        ))
    return flags


# ---------------------------------------------------------------------------
# Generic (industrial / commercial) companies
# ---------------------------------------------------------------------------


def _analyze_generic(h: _History) -> Dict[str, Any]:
    sales = _flow(h, "satis_gelirleri")
    favok = _flow(h, "favok")
    efk = _flow(h, "esas_faaliyet_kari")
    net = _flow(h, "net_kar")
    cfo = _flow(h, "faaliyet_nakit_akisi")
    fcf = _flow(h, "serbest_nakit_akisi")
    equity = _flow(h, "ozkaynaklar")
    net_debt = _flow(h, "net_borc")
    latest = h.latest
    period_label = _History.label(latest)
    ly_label = _History.label(_History.shift(latest, 4)) if latest else "-"

    roe = h.value(latest, "ratios", "roe")
    current_ratio = h.value(latest, "ratios", "cari_oran")

    gross_margin = _margin_for_period(h, latest, "brut_kar", "satis_gelirleri")
    gross_margin_prev = _margin_for_period(h, _History.shift(latest, 1) if latest else None, "brut_kar", "satis_gelirleri")
    gross_margin_ly = _margin_for_period(h, _History.shift(latest, 4) if latest else None, "brut_kar", "satis_gelirleri")
    gross_margin_qoq_pp = _margin_delta_pp(h, latest, "brut_kar", "satis_gelirleri", periods_back=1)
    gross_margin_yoy_pp = _margin_delta_pp(h, latest, "brut_kar", "satis_gelirleri", periods_back=4)
    gross_margin_hist = _margin_history(h, "brut_kar", "satis_gelirleri")
    gross_margin_median = median(gross_margin_hist) if len(gross_margin_hist) >= 3 else None

    favok_margin = _margin(favok["cur"], sales["cur"])
    if favok_margin is None:
        favok_margin = h.value(latest, "ratios", "favok_marji")
    favok_margin_prev = _margin_for_period(h, _History.shift(latest, 1) if latest else None, "favok", "satis_gelirleri")
    favok_margin_qoq_pp = _margin_delta_pp(h, latest, "favok", "satis_gelirleri", periods_back=1)
    favok_margin_ly = _margin(favok["ly"], sales["ly"])
    favok_margin_hist = _margin_history(h, "favok", "satis_gelirleri")
    favok_margin_median = median(favok_margin_hist) if len(favok_margin_hist) >= 3 else None
    net_margin = _margin(net["cur"], sales["cur"])
    if net_margin is None:
        net_margin = h.value(latest, "ratios", "net_kar_marji")
    net_margin_prev = _margin_for_period(h, _History.shift(latest, 1) if latest else None, "net_kar", "satis_gelirleri")
    net_margin_ly = _margin_for_period(h, _History.shift(latest, 4) if latest else None, "net_kar", "satis_gelirleri")
    net_margin_qoq_pp = _margin_delta_pp(h, latest, "net_kar", "satis_gelirleri", periods_back=1)
    net_margin_yoy_pp = _margin_delta_pp(h, latest, "net_kar", "satis_gelirleri", periods_back=4)
    net_margin_hist = _margin_history(h, "net_kar", "satis_gelirleri")
    net_margin_median = median(net_margin_hist) if len(net_margin_hist) >= 3 else None
    net_base_margin = _margin(net["ly"], sales["ly"])

    operating = efk if efk["cur"] is not None and efk["ly"] is not None else favok
    operating_label = "esas faaliyet kârı" if operating is efk else "FAVÖK"
    quality = _earnings_quality(
        net,
        operating,
        ebitda_current=favok["cur"],
        ebitda_ly=favok["ly"],
        reference_current=sales["cur"],
        materiality_pp=0.3,
    )
    non_operating = quality["code"] in NON_OPERATING_QUALITY_CODES or quality["net_exceeds_ebitda"]

    net_change_score, regime = _score_profit_change(
        net["cur"], net["ly"], sales["cur"], sales["ly"], pct_scale=60.0, delta_scale=4.0, materiality=3.0,
    )
    net_ttm_score, _ = _score_profit_change(
        net["ttm"], net["ttm_ly"], sales["ttm"], sales["ttm_ly"], pct_scale=50.0, delta_scale=3.0, materiality=3.0,
    )
    op_metric = favok if favok["cur"] is not None else efk
    op_metric_label = "FAVÖK" if op_metric is favok else "Esas faaliyet kârı"
    op_change_score, _ = _score_profit_change(
        op_metric["cur"], op_metric["ly"], sales["cur"], sales["ly"], pct_scale=45.0, delta_scale=3.0, materiality=4.0,
    )
    op_karlilik_metric = efk if efk["cur"] is not None and efk["ly"] is not None else favok
    op_karlilik_label = "Esas faaliyet kârı" if op_karlilik_metric is efk else "FAVÖK"
    op_karlilik_change_score, _ = _score_profit_change(
        op_karlilik_metric["cur"],
        op_karlilik_metric["ly"],
        sales["cur"],
        sales["ly"],
        pct_scale=45.0,
        delta_scale=1.5 if op_karlilik_metric is efk else 3.0,
        materiality=1.5 if op_karlilik_metric is efk else 4.0,
    )
    op_ytd_metric = efk if efk["ytd"] is not None and efk["ytd_ly"] is not None else favok
    op_ytd_label = "Esas faaliyet kârı" if op_ytd_metric is efk else "FAVÖK"
    op_ytd_change_score, _ = _score_profit_change(
        op_ytd_metric["ytd"], op_ytd_metric["ytd_ly"], sales["ytd"], sales["ytd_ly"],
        pct_scale=45.0,
        delta_scale=1.5 if op_ytd_metric is efk else 3.0,
        materiality=1.5 if op_ytd_metric is efk else 4.0,
    )
    # Unbacked net profit growth counts less in both growth and profitability.
    net_growth_for_growth_bucket = _shrink(net_change_score, 0.5) if non_operating else net_change_score
    net_ttm_growth_for_growth_bucket = _shrink(net_ttm_score, 0.5) if non_operating else net_ttm_score
    net_growth_for_profitability = net_growth_for_growth_bucket
    sales_season = _season(h, "satis_gelirleri")

    buyume = [
        _spec("Satış büyümesi (yıllık, çeyrek)", _score_growth(sales["yoy"], 30.0), 1.2, _fmt_pct(sales["yoy"])),
        _spec("Satış büyümesi (yıl başından beri)", _score_growth(sales["ytd_yoy"], 30.0), 0.6, _fmt_pct(sales["ytd_yoy"])),
        _spec("Satış büyümesi (son 12 ay)", _score_growth(sales["ttm_yoy"], 30.0), 0.5, _fmt_pct(sales["ttm_yoy"])),
        _spec(f"{op_metric_label} değişimi (yıllık)", op_change_score, 1.0, _fmt_change(op_metric["cur"], op_metric["ly"])),
        _spec(
            "Net kâr değişimi (yıllık)" + (" · kalite indirimi" if non_operating else ""),
            net_growth_for_growth_bucket,
            0.9,
            _fmt_change(net["cur"], net["ly"]),
        ),
        _spec("Net kâr değişimi (son 12 ay)" + (" · kalite indirimi" if non_operating else ""), net_ttm_growth_for_growth_bucket, 0.6, _fmt_change(net["ttm"], net["ttm_ly"])),
        _spec("Satışların mevsimsel konumu", _score_centered(sales_season["seasonal_delta"], 18.0), 0.4, _fmt_pct(sales_season["seasonal_delta"])),
    ]

    favok_margin_vs_hist = (favok_margin - favok_margin_median) if favok_margin is not None and favok_margin_median is not None else None
    favok_margin_yoy_pp = (favok_margin - favok_margin_ly) if favok_margin is not None and favok_margin_ly is not None else None
    favok_hist_scale = _margin_pp_scale(favok_margin_median, favok_margin, default_scale=3.0, min_scale=1.2)
    favok_qoq_scale = _margin_pp_scale(favok_margin_median, favok_margin, default_scale=2.0, min_scale=1.0)
    favok_yoy_scale = _margin_pp_scale(favok_margin_median, favok_margin, default_scale=3.0, min_scale=1.2)
    net_qoq_scale = _margin_pp_scale(net_margin_median, net_margin, default_scale=2.0, min_scale=0.8)
    net_yoy_scale = _margin_pp_scale(net_margin_median, net_margin, default_scale=2.5, min_scale=1.0)
    net_margin_level_score = _score_margin_level(
        net_margin, abs_low=0.0, abs_high=12.0, hist_median=None if non_operating else net_margin_median, roe=roe,
    )
    net_margin_yoy_score = _score_growth(net_margin_yoy_pp, net_yoy_scale)
    if non_operating:
        net_margin_level_score = _shrink(net_margin_level_score, 0.5)
        net_margin_yoy_score = _shrink(net_margin_yoy_score, 0.5)
    karlilik = [
        _spec("Brüt kâr marjı (çeyreklik değişim)", _score_growth(gross_margin_qoq_pp, 2.0), 0.35, _fmt_pp(gross_margin_qoq_pp)),
        _spec("FAVÖK marjı seviyesi", _score_margin_level(favok_margin, abs_low=4.0, abs_high=20.0, hist_median=favok_margin_median, roe=roe), 0.6, _fmt_level_pct(favok_margin)),
        _spec("FAVÖK marjı (kendi geçmişine göre)", _score_growth(favok_margin_vs_hist, favok_hist_scale), 0.7, _fmt_pp(favok_margin_vs_hist)),
        _spec("FAVÖK marjı (çeyreklik değişim)", _score_growth(favok_margin_qoq_pp, favok_qoq_scale), 0.55, _fmt_pp(favok_margin_qoq_pp)),
        _spec("FAVÖK marjı yıllık değişim", _score_growth(favok_margin_yoy_pp, favok_yoy_scale), 0.6, _fmt_pp(favok_margin_yoy_pp)),
        _spec("Net kâr marjı seviyesi", net_margin_level_score, 0.3 if non_operating else 0.6, _fmt_level_pct(net_margin)),
        _spec("Net kâr marjı (çeyreklik değişim)" + (" · kalite indirimi" if non_operating else ""), _score_growth(net_margin_qoq_pp, net_qoq_scale), 0.2 if non_operating else 0.35, _fmt_pp(net_margin_qoq_pp)),
        _spec("Net kâr marjı yıllık değişim" + (" · kalite indirimi" if non_operating else ""), net_margin_yoy_score, 0.2 if non_operating else 0.45, _fmt_pp(net_margin_yoy_pp)),
        _spec("ROE (son 12 ay)", _score_higher(roe, 2.0, 18.0), 0.8, _fmt_level_pct(roe)),
        _spec("Net kâr değişimi (aynı çeyrek)" + (" · kalite indirimi" if non_operating else ""), net_growth_for_profitability, 0.6, _fmt_change(net["cur"], net["ly"])),
        _spec(f"{op_karlilik_label} değişimi (aynı çeyrek)", op_karlilik_change_score, 0.6, _fmt_change(op_karlilik_metric["cur"], op_karlilik_metric["ly"])),
        _spec(f"{op_ytd_label} değişimi (YTD)", op_ytd_change_score, 0.7, _fmt_change(op_ytd_metric["ytd"], op_ytd_metric["ytd_ly"])),
        _spec(f"Kâr kalitesi ({operating_label} desteği)", quality["score"], 1.0, str(quality["code"] or "-")),
    ]

    # Cash flow: trailing 12 months first, single quarter as secondary signal.
    cfo_ttm, net_ttm, fcf_ttm, sales_ttm = cfo["ttm"], net["ttm"], fcf["ttm"], sales["ttm"]
    use_ttm = cfo_ttm is not None and net_ttm is not None
    cfo_base = cfo_ttm if use_ttm else cfo["cur"]
    net_base = net_ttm if use_ttm else net["cur"]
    conversion: Optional[float] = None
    conversion_score: Optional[float] = None
    if cfo_base is not None and net_base is not None:
        if net_base > 0:
            conversion = cfo_base / net_base
            conversion_score = _score_higher(conversion, 0.2, 1.0)
        elif cfo_base > 0:
            conversion_score = 6.5
        else:
            conversion_score = 1.5
    fcf_base = fcf_ttm if fcf_ttm is not None and sales_ttm is not None else fcf["cur"]
    sales_base = sales_ttm if fcf_ttm is not None and sales_ttm is not None else sales["cur"]
    fcf_margin = _margin(fcf_base, sales_base)
    fcf_margin_score = _score_higher(fcf_margin, -6.0, 8.0)
    investment_phase = bool(cfo_base is not None and cfo_base > 0 and fcf_base is not None and fcf_base < 0)
    if investment_phase and fcf_margin_score is not None:
        fcf_margin_score = max(fcf_margin_score, 3.5)
    cfo_quarter_score: Optional[float] = None
    if cfo["cur"] is not None:
        if cfo["cur"] > 0:
            cfo_quarter_score = 7.5
        else:
            cfo_quarter_score = 4.0 if (cfo_ttm is not None and cfo_ttm > 0) else 2.0
    fcf_quarter_score: Optional[float] = None
    if fcf["cur"] is not None:
        if fcf["cur"] > 0:
            fcf_quarter_score = 7.0
        else:
            fcf_quarter_score = 4.0 if (cfo["cur"] is not None and cfo["cur"] > 0) else 2.0
    window_text = "son 12 ay" if use_ttm else "çeyrek"
    nakit = [
        _spec(f"Nakit dönüşümü (FNA / net kâr, {window_text})", conversion_score, 1.2, _fmt_ratio(conversion) if conversion is not None else _fmt_amount(cfo_base)),
        _spec("Faaliyet nakit akışı (çeyrek)", cfo_quarter_score, 0.6, _fmt_amount(cfo["cur"])),
        _spec("Serbest nakit akışı marjı" + (" (son 12 ay)" if fcf_ttm is not None and sales_ttm is not None else " (çeyrek)"), fcf_margin_score, 1.0, _fmt_level_pct(fcf_margin)),
        _spec("Serbest nakit akışı (çeyrek)", fcf_quarter_score, 0.4, _fmt_amount(fcf["cur"])),
    ]

    eq = equity["cur"]
    negative_equity = eq is not None and eq <= 0
    nd = net_debt["cur"]
    nd_equity: Optional[float] = None
    if nd is not None and eq is not None and eq > 0:
        nd_equity = nd / eq
    elif not negative_equity:
        nd_equity = h.value(latest, "ratios", "net_borc_ozkaynak")
    nd_ebitda: Optional[float] = None
    nd_ebitda_score: Optional[float] = None
    if nd is not None:
        if nd <= 0:
            nd_ebitda_score = 10.0
        elif favok["ttm"] is not None and favok["ttm"] > 0:
            nd_ebitda = nd / favok["ttm"]
            nd_ebitda_score = _score_lower(nd_ebitda, 0.5, 4.0)
        elif favok["ttm"] is not None:
            nd_ebitda_score = 1.0
    nd_change_pp: Optional[float] = None
    if nd is not None and net_debt["ly"] is not None and eq is not None and eq > 0:
        nd_change_pp = (nd - net_debt["ly"]) / eq * 100.0
    bilanco = [
        _spec("Net borç / özkaynak", None if negative_equity else _score_lower(nd_equity, 0.0, 2.0), 1.0, _fmt_ratio(nd_equity)),
        _spec("Net borç / FAVÖK (son 12 ay)", nd_ebitda_score, 1.0, "net nakit" if nd is not None and nd <= 0 else _fmt_ratio(nd_ebitda)),
        _spec("Cari oran", _score_higher(current_ratio, 0.8, 1.8), 0.8, _fmt_ratio(current_ratio)),
        _spec("Özkaynak büyümesi (yıllık)", _score_growth(equity["yoy"], 30.0), 0.6, _fmt_pct(equity["yoy"])),
        _spec("Net borç değişimi (yıllık, özkaynağa göre)", _score_growth(-nd_change_pp, 15.0) if nd_change_pp is not None else None, 0.4, _fmt_pp(nd_change_pp)),
    ]
    if negative_equity:
        bilanco.append(_spec("Özkaynak negatif", 0.0, 3.0, _fmt_amount(eq), optional=True))

    # ---- flags --------------------------------------------------------------
    flags: List[Dict[str, Any]] = []
    flags.extend(_profit_regime_flags(
        regime, net, period_label=period_label, ly_label=ly_label,
        base_margin=net_base_margin, low_base_threshold=1.5,
    ))
    flags.extend(_quality_flags(quality, operating_label=operating_label, period_label=period_label))
    if (
        op_ytd_metric["ytd"] is not None
        and op_ytd_metric["ytd"] > 0
        and op_ytd_metric["ytd_ly"] is not None
        and op_ytd_metric["ytd"] > op_ytd_metric["ytd_ly"]
    ):
        flags.append(_flag(
            "faaliyet_kar_ytd_artisi", "positive", f"{op_ytd_label} YTD arttı",
            f"{period_label} YTD {op_ytd_label.lower()} {_fmt_change(op_ytd_metric['ytd'], op_ytd_metric['ytd_ly'])}; yıl başından beri faaliyet kârlılığı geçen yılın aynı döneminin üzerinde.",
            "karlilik",
        ))
    if favok_margin_yoy_pp is not None and favok_margin_yoy_pp >= 2.0:
        flags.append(_flag("marj_genislemesi", "positive", "FAVÖK marjı genişledi", f"FAVÖK marjı yıllık {_fmt_pp(favok_margin_yoy_pp)} değişimle {_fmt_level_pct(favok_margin)} oldu.", "karlilik"))
    elif favok_margin_yoy_pp is not None and favok_margin_yoy_pp <= -2.0:
        flags.append(_flag("marj_daralmasi", "negative", "FAVÖK marjı daraldı", f"FAVÖK marjı yıllık {_fmt_pp(favok_margin_yoy_pp)} değişimle {_fmt_level_pct(favok_margin)} oldu.", "karlilik"))
    if favok_margin_qoq_pp is not None and favok_margin_qoq_pp >= 0.3:
        margin_comparisons = [
            f"FAVÖK marjı {_fmt_level_pct(favok_margin)} oldu ({_fmt_pp(favok_margin_qoq_pp)} çeyreklik, {_fmt_pp(favok_margin_yoy_pp)} yıllık değişim)"
        ]
        if gross_margin is not None:
            margin_comparisons.append(f"brüt kâr marjı {_fmt_level_pct(gross_margin)} ({_fmt_pp(gross_margin_qoq_pp)} çeyreklik)")
        if net_margin is not None:
            margin_comparisons.append(f"net kâr marjı {_fmt_level_pct(net_margin)} ({_fmt_pp(net_margin_qoq_pp)} çeyreklik)")
        flags.append(_flag(
            "marj_ceyreklik_toparlandi", "positive", "FAVÖK marjı son çeyrekte toparlandı",
            f"{period_label}: " + "; ".join(margin_comparisons) + ". Mutlak seviye ile son çeyrekteki yön ayrı değerlendirilmelidir.",
            "karlilik",
        ))
    elif favok_margin_qoq_pp is not None and favok_margin_qoq_pp <= -0.3:
        flags.append(_flag(
            "marj_ceyreklik_daraldi", "negative", "FAVÖK marjı son çeyrekte daraldı",
            f"{period_label} FAVÖK marjı {_fmt_level_pct(favok_margin)} oldu; önceki çeyreğe göre {_fmt_pp(favok_margin_qoq_pp)} değişti.",
            "karlilik",
        ))
    if sales["yoy"] is not None and sales["yoy"] <= -10:
        if favok_margin_yoy_pp is not None and favok_margin_yoy_pp > 0:
            flags.append(_flag("satis_dusus_marj_artis", "neutral", "Satış düşerken marj genişledi", f"Satışlar {_fmt_pct(sales['yoy'])} değişirken FAVÖK marjı {_fmt_pp(favok_margin_yoy_pp)} iyileşti.", "buyume"))
        else:
            flags.append(_flag("satis_dususu", "negative", "Satışlar geriledi", f"Satışlar geçen yılın aynı çeyreğine göre {_fmt_pct(sales['yoy'])} değişti.", "buyume"))
    elif sales["yoy"] is not None and sales["yoy"] > 0 and favok_margin_yoy_pp is not None and favok_margin_yoy_pp <= -2.0:
        flags.append(_flag("karsiz_buyume", "warning", "Büyüme marj kaybıyla geldi", f"Satışlar {_fmt_pct(sales['yoy'])} artarken FAVÖK marjı {_fmt_pp(favok_margin_yoy_pp)} geriledi.", "karlilik"))
    if sales["qoq"] is not None and sales["qoq"] <= -10 and sales["yoy"] is not None and sales["yoy"] > 0:
        flags.append(_flag("ceyreklik_satis_dusus_yillik_artis", "neutral", "Çeyreklik ve yıllık satış eğilimleri farklı", f"Satışlar önceki çeyreğe göre {_fmt_pct(sales['qoq'])}, geçen yılın aynı çeyreğine göre {_fmt_pct(sales['yoy'])} değişti.", "buyume"))

    cfo_negative_ttm = cfo_ttm is not None and cfo_ttm < 0
    if conversion is not None and conversion >= 0.8:
        flags.append(_flag("nakit_teyitli_kar", "positive", "Kâr nakitle teyit ediliyor", f"Faaliyet nakit akışı net kârın {_fmt_ratio(conversion)} katı ({window_text}).", "nakit_akisi"))
    elif conversion is not None and conversion < 0.3:
        flags.append(_flag("nakit_donusumu_zayif", "warning", "Kârın nakde dönüşümü zayıf", f"Faaliyet nakit akışı net kârın yalnızca {_fmt_ratio(conversion)} katı ({window_text}); bu veriler farkın nedenini göstermiyor.", "nakit_akisi"))
    if cfo_negative_ttm:
        flags.append(_flag("faaliyet_nakit_negatif", "negative", "Son 12 ayda faaliyet nakit akışı negatif", f"Son 12 ay faaliyet nakit akışı {_fmt_amount(cfo_ttm)}.", "nakit_akisi"))
    if investment_phase:
        flags.append(_flag("faaliyet_pozitif_fcf_negatif", "neutral", "Faaliyet nakdi pozitif, serbest nakit akışı negatif", "Faaliyet nakit akışı pozitifken serbest nakit akışı negatif. Yatırım harcamalarının ayrıntısı bu görünümde verilmediği için neden belirlenemiyor.", "nakit_akisi"))

    if negative_equity:
        flags.append(_flag("negatif_ozkaynak", "negative", "Özkaynak negatif", f"Özkaynaklar {_fmt_amount(eq)}; bilanço riski kâr artışıyla telafi edilmez.", "bilanco"))
    if nd is not None and nd <= 0:
        flags.append(_flag("net_nakit", "positive", "Net nakit pozisyonu", f"Nakit, finansal borçları {_fmt_amount(-nd)} aşıyor.", "bilanco"))
    elif nd_ebitda is not None and nd_ebitda >= 3.0:
        flags.append(_flag("yuksek_kaldirac", "negative", "Yüksek kaldıraç", f"Net borç / son 12 ay FAVÖK {_fmt_ratio(nd_ebitda)}.", "bilanco"))
    if current_ratio is not None and current_ratio < 1.0:
        flags.append(_flag("likidite_zayif", "warning", "Cari oran 1'in altında", f"Cari oran {_fmt_ratio(current_ratio)}; kısa vadeli yükümlülükler dönen varlıkları aşıyor.", "bilanco"))

    # ---- guards / caps -------------------------------------------------------
    guards = _empty_guards()
    supported_growth = (
        quality["code"] in ("faaliyet_destekli", "kismen_faaliyet_destekli")
        and not non_operating
        and regime in ("kar_artti", "zarardan_kara")
    )
    if supported_growth:
        _tighten(guards, "karlilik", minimum=-0.3)
        _tighten(guards, "buyume", minimum=-0.3)
        if (favok_margin_qoq_pp or 0.0) >= 0.3 and (favok_margin_vs_hist or 0.0) >= 0.0:
            _tighten(guards, "karlilik", minimum=-0.2)
    if non_operating:
        _tighten(guards, "karlilik", maximum=0.3)
        _tighten(guards, "buyume", maximum=0.3)
        # The base already discounts quality; AI should not double-punish.
        _tighten(guards, "karlilik", minimum=-0.5)
        if favok_margin_qoq_pp is not None and favok_margin_qoq_pp >= 0.5:
            # Recent operating margin recovery is positive evidence even when
            # the net-profit increase itself is not explained by operations.
            _tighten(guards, "karlilik", minimum=-0.3)
    if quality["code"] == "faaliyet_guclu_net_zayif":
        _tighten(guards, "buyume", minimum=-0.3)
    if negative_equity:
        _tighten(guards, "bilanco", maximum=0.0)
    if conversion is not None and conversion >= 0.8:
        _tighten(guards, "nakit_akisi", minimum=-0.3)
    if cfo_negative_ttm:
        _tighten(guards, "nakit_akisi", maximum=0.3)

    bucket_caps: Dict[str, float] = {}
    overall_cap: Optional[float] = None
    if negative_equity:
        bucket_caps["bilanco"] = 2.0
        overall_cap = 3.5
    if regime == "kardan_zarara" and cfo_negative_ttm:
        overall_cap = min(overall_cap or 10.0, 4.0)
    if net["cur"] is not None and net["cur"] < 0 and net_ttm is not None and net_ttm < 0:
        overall_cap = min(overall_cap or 10.0, 5.0)

    facts = {
        "latest_period": period_label,
        "same_quarter_last_year": ly_label,
        "profit_regime": regime,
        "earnings_quality": quality["code"],
        "operating_metric_for_quality": operating_label,
        "satis": _flow_fact(sales),
        "favok": _flow_fact(favok),
        "esas_faaliyet_kari": _flow_fact(efk),
        "net_kar": _flow_fact(net),
        "net_kar_eksi_faaliyet_kari": _fmt_amount(quality["non_operating_current"]),
        "faaliyet_karinin_aciklama_payi": None if quality["explained_share"] is None else round(quality["explained_share"], 2),
        "marj_trendleri": {
            "brut_kar": {
                "current": _fmt_level_pct(gross_margin),
                "previous_quarter": _fmt_level_pct(gross_margin_prev),
                "same_quarter_last_year": _fmt_level_pct(gross_margin_ly),
                "qoq_change_pp": _fmt_pp(gross_margin_qoq_pp),
                "yoy_change_pp": _fmt_pp(gross_margin_yoy_pp),
                "historical_median": _fmt_level_pct(gross_margin_median),
                "historical_median_sample_quarters": len(gross_margin_hist),
                "trend": _margin_trend_label(gross_margin_qoq_pp),
            },
            "favok": {
                "current": _fmt_level_pct(favok_margin),
                "previous_quarter": _fmt_level_pct(favok_margin_prev),
                "same_quarter_last_year": _fmt_level_pct(favok_margin_ly),
                "qoq_change_pp": _fmt_pp(favok_margin_qoq_pp),
                "yoy_change_pp": _fmt_pp(favok_margin_yoy_pp),
                "historical_median": _fmt_level_pct(favok_margin_median),
                "historical_median_sample_quarters": len(favok_margin_hist),
                "trend": _margin_trend_label(favok_margin_qoq_pp),
            },
            "net_kar": {
                "current": _fmt_level_pct(net_margin),
                "previous_quarter": _fmt_level_pct(net_margin_prev),
                "same_quarter_last_year": _fmt_level_pct(net_margin_ly),
                "qoq_change_pp": _fmt_pp(net_margin_qoq_pp),
                "yoy_change_pp": _fmt_pp(net_margin_yoy_pp),
                "historical_median": _fmt_level_pct(net_margin_median),
                "historical_median_sample_quarters": len(net_margin_hist),
                "trend": _margin_trend_label(net_margin_qoq_pp),
            },
        },
        "favok_marji": _fmt_level_pct(favok_margin),
        "favok_marji_yillik_degisim": _fmt_pp(favok_margin_yoy_pp),
        "favok_marji_gecmis_medyan": _fmt_level_pct(favok_margin_median),
        "net_kar_marji": _fmt_level_pct(net_margin),
        "nakit_donusumu": _fmt_ratio(conversion),
        "serbest_nakit_akisi_son_12_ay": _fmt_amount(fcf_ttm),
        "net_borc_favok": "net nakit" if nd is not None and nd <= 0 else _fmt_ratio(nd_ebitda),
    }

    return {
        "buckets": {"buyume": buyume, "karlilik": karlilik, "bilanco": bilanco, "nakit_akisi": nakit},
        "flags": flags,
        "facts": facts,
        "guards": guards,
        "bucket_caps": bucket_caps,
        "overall_cap": overall_cap,
        "season_flags": [
            sales_season["season_flag"],
            _season(h, "net_kar")["season_flag"],
            _season(h, "favok")["season_flag"],
        ],
    }


def _flow_fact(flow: Dict[str, Optional[float]]) -> Dict[str, str]:
    return {
        "ceyrek": _fmt_amount(flow.get("cur")),
        "onceki_ceyrek": _fmt_amount(flow.get("prev")),
        "gecen_yil_ayni_ceyrek": _fmt_amount(flow.get("ly")),
        "yillik_degisim": _fmt_pct(flow.get("yoy")) if _both_positive(flow.get("cur"), flow.get("ly")) else "anlamsiz/yok",
        "ceyreklik_degisim": _fmt_pct(flow.get("qoq")) if _both_positive(flow.get("cur"), flow.get("prev")) else "anlamsiz/yok",
        "ytd": _fmt_amount(flow.get("ytd")),
        "ytd_degisim": _fmt_pct(flow.get("ytd_yoy")) if _both_positive(flow.get("ytd"), flow.get("ytd_ly")) else "anlamsiz/yok",
        "son_12_ay": _fmt_amount(flow.get("ttm")),
        "son_12_ay_degisim": _fmt_pct(flow.get("ttm_yoy")) if _both_positive(flow.get("ttm"), flow.get("ttm_ly")) else "anlamsiz/yok",
    }


def _both_positive(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and a > 0 and b > 0


# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------


def _analyze_bank(h: _History) -> Dict[str, Any]:
    nii = _flow(h, "net_faiz_geliri_gideri")
    fees = _flow(h, "net_ucret_komisyon_gelirleri")
    operating = _flow(h, "net_faaliyet_kari")
    net = _flow(h, "net_kar")
    loans = _flow(h, "krediler")
    deposits = _flow(h, "mevduatlar")
    equity = _flow(h, "ozkaynaklar")
    fin_assets = _season(h, "finansal_varliklar_net")
    latest = h.latest
    period_label = _History.label(latest)
    ly_label = _History.label(_History.shift(latest, 4)) if latest else "-"
    roe = h.value(latest, "ratios", "roe")
    loan_deposit = h.value(latest, "ratios", "kredi_mevduat_orani")
    provisions_ratio = h.value(latest, "ratios", "karsilik_kredi_orani")

    net_change_score, regime = _score_profit_change(
        net["cur"], net["ly"], equity["cur"], equity["ly"], pct_scale=50.0, delta_scale=1.0, materiality=1.0,
    )
    net_ttm_score, _ = _score_profit_change(
        net["ttm"], net["ttm_ly"], equity["cur"], equity["ly"], pct_scale=45.0, delta_scale=3.0, materiality=4.0,
    )
    op_change_score, _ = _score_profit_change(
        operating["cur"], operating["ly"], equity["cur"], equity["ly"], pct_scale=50.0, delta_scale=1.0, materiality=1.0,
    )
    quality = _earnings_quality(net, operating, ebitda_current=None, reference_current=equity["cur"], materiality_pp=0.2)
    non_operating = quality["code"] in NON_OPERATING_QUALITY_CODES
    net_change_score_quality_adjusted = _shrink(net_change_score, 0.5) if non_operating else net_change_score
    net_ttm_score_quality_adjusted = _shrink(net_ttm_score, 0.5) if non_operating else net_ttm_score

    buyume = [
        _spec("Net faiz geliri büyümesi", _score_growth(nii["yoy"], 35.0), 0.8, _fmt_pct(nii["yoy"])),
        _spec("Ücret-komisyon büyümesi", _score_growth(fees["yoy"], 35.0), 1.0, _fmt_pct(fees["yoy"])),
        _spec("Net faaliyet kârı değişimi", op_change_score, 1.0, _fmt_change(operating["cur"], operating["ly"])),
        _spec("Net kâr değişimi (yıllık)" + (" · kalite indirimi" if non_operating else ""), net_change_score_quality_adjusted, 1.0, _fmt_change(net["cur"], net["ly"])),
        _spec("Kredi büyümesi (yıllık)", _score_growth(loans["yoy"], 35.0), 0.6, _fmt_pct(loans["yoy"])),
    ]
    karlilik = [
        _spec("ROE (son 12 ay)", _score_higher(roe, 8.0, 30.0), 1.2, _fmt_level_pct(roe)),
        _spec("Net kâr değişimi (aynı çeyrek)" + (" · kalite indirimi" if non_operating else ""), net_change_score_quality_adjusted, 0.9, _fmt_change(net["cur"], net["ly"])),
        _spec("Net kâr değişimi (son 12 ay)" + (" · kalite indirimi" if non_operating else ""), net_ttm_score_quality_adjusted, 0.7, _fmt_change(net["ttm"], net["ttm_ly"])),
        _spec("Kâr kalitesi (faaliyet kârı desteği)", quality["score"], 0.6, str(quality["code"] or "-")),
    ]
    eq = equity["cur"]
    negative_equity = eq is not None and eq <= 0
    bilanco = [
        _spec("Özkaynak büyümesi (yıllık)", _score_growth(equity["yoy"], 30.0), 1.0, _fmt_pct(equity["yoy"])),
        _spec("Karşılık / kredi", _score_lower(provisions_ratio, 0.01, 0.08), 1.0, _fmt_ratio(provisions_ratio)),
        _spec("Finansal varlık eğilimi", _score_centered(fin_assets["trend_delta"], 20.0), 0.4, _fmt_pct(fin_assets["trend_delta"])),
    ]
    if negative_equity:
        bilanco.append(_spec("Özkaynak negatif", 0.0, 3.0, _fmt_amount(eq), optional=True))
    nakit = [
        _spec("Kredi / mevduat dengesi", _score_band(loan_deposit, 0.55, 0.8, 1.15, 1.45), 1.2, _fmt_ratio(loan_deposit)),
        _spec("Mevduat büyümesi (yıllık)", _score_growth(deposits["yoy"], 30.0), 0.9, _fmt_pct(deposits["yoy"])),
        _spec("Mevduat değişimi (çeyreklik)", _score_growth(deposits["qoq"], 10.0), 0.5, _fmt_pct(deposits["qoq"])),
    ]

    flags = _profit_regime_flags(
        regime, net, period_label=period_label, ly_label=ly_label,
        base_margin=_safe_ratio(net["ly"], equity["ly"], 100.0), low_base_threshold=0.5,
    )
    flags.extend(_quality_flags(quality, operating_label="net faaliyet kârı", period_label=period_label))
    if loan_deposit is not None and loan_deposit > 1.3:
        flags.append(_flag("yuksek_kredi_mevduat", "warning", "Kredi / mevduat oranı yüksek", f"Kredi / mevduat oranı {_fmt_ratio(loan_deposit)}; krediler mevduat tutarını aşıyor.", "nakit_akisi"))
    if negative_equity:
        flags.append(_flag("negatif_ozkaynak", "negative", "Özkaynak negatif", f"Özkaynaklar {_fmt_amount(eq)}.", "bilanco"))

    guards = _empty_guards()
    if quality["code"] == "faaliyet_destekli" and regime in ("kar_artti", "zarardan_kara"):
        _tighten(guards, "karlilik", minimum=-0.3)
        _tighten(guards, "buyume", minimum=-0.3)
    if non_operating:
        _tighten(guards, "karlilik", maximum=0.3)
    if negative_equity:
        _tighten(guards, "bilanco", maximum=0.0)

    overall_cap: Optional[float] = 3.5 if negative_equity else None
    if net["cur"] is not None and net["cur"] < 0 and net["ttm"] is not None and net["ttm"] < 0:
        overall_cap = min(overall_cap or 10.0, 5.0)

    facts = {
        "latest_period": period_label,
        "same_quarter_last_year": ly_label,
        "profit_regime": regime,
        "earnings_quality": quality["code"],
        "net_faiz_geliri": _flow_fact(nii),
        "ucret_komisyon": _flow_fact(fees),
        "net_faaliyet_kari": _flow_fact(operating),
        "net_kar": _flow_fact(net),
        "roe": _fmt_level_pct(roe),
        "kredi_mevduat": _fmt_ratio(loan_deposit),
        "karsilik_kredi": _fmt_ratio(provisions_ratio),
    }
    return {
        "buckets": {"buyume": buyume, "karlilik": karlilik, "bilanco": bilanco, "nakit_akisi": nakit},
        "flags": flags,
        "facts": facts,
        "guards": guards,
        "bucket_caps": {"bilanco": 2.0} if negative_equity else {},
        "overall_cap": overall_cap,
        "season_flags": [
            _season(h, "krediler")["season_flag"],
            _season(h, "net_kar")["season_flag"],
            _season(h, "mevduatlar")["season_flag"],
        ],
    }


# ---------------------------------------------------------------------------
# Insurance
# ---------------------------------------------------------------------------


def _analyze_insurance(h: _History) -> Dict[str, Any]:
    gross_premium = _flow(h, "prim_uretimi")
    net_premium = _flow(h, "alinan_net_primler")
    technical_income = _flow(h, "teknik_gelirler")
    technical_balance = _flow(h, "teknik_denge")
    net = _flow(h, "net_kar")
    equity = _flow(h, "ozkaynaklar")
    cash_like = _flow(h, "nakit_benzeri_finansal_varliklar")
    latest = h.latest
    period_label = _History.label(latest)
    ly_label = _History.label(_History.shift(latest, 4)) if latest else "-"
    tech_margin = h.value(latest, "ratios", "teknik_denge_marji")
    cash_reserve = h.value(latest, "ratios", "nakit_karsilik_orani")
    equity_reserve = h.value(latest, "ratios", "ozkaynak_karsilik_orani")
    debt_equity = h.value(latest, "ratios", "borc_ozkaynak_orani")
    roe = h.value(latest, "ratios", "roe")

    net_change_score, regime = _score_profit_change(
        net["cur"], net["ly"], equity["cur"], equity["ly"], pct_scale=50.0, delta_scale=1.0, materiality=1.0,
    )
    net_ttm_score, _ = _score_profit_change(
        net["ttm"], net["ttm_ly"], equity["cur"], equity["ly"], pct_scale=45.0, delta_scale=3.0, materiality=4.0,
    )
    premium_ref = net_premium if net_premium["cur"] is not None else gross_premium
    tech_change_score, _ = _score_profit_change(
        technical_balance["cur"], technical_balance["ly"], premium_ref["cur"], premium_ref["ly"],
        pct_scale=50.0, delta_scale=4.0, materiality=3.0,
    )
    quality = _earnings_quality(
        net,
        technical_balance,
        ebitda_current=None,
        reference_current=premium_ref["cur"],
        materiality_pp=0.3,
    )
    non_operating = quality["code"] in NON_OPERATING_QUALITY_CODES
    net_change_score_quality_adjusted = _shrink(net_change_score, 0.5) if non_operating else net_change_score
    net_ttm_score_quality_adjusted = _shrink(net_ttm_score, 0.5) if non_operating else net_ttm_score

    buyume = [
        _spec("Prim üretimi büyümesi", _score_growth(gross_premium["yoy"], 30.0), 1.0, _fmt_pct(gross_premium["yoy"])),
        _spec("Net prim büyümesi", _score_growth(net_premium["yoy"], 30.0), 1.0, _fmt_pct(net_premium["yoy"])),
        _spec("Teknik gelir büyümesi", _score_growth(technical_income["yoy"], 30.0), 0.8, _fmt_pct(technical_income["yoy"])),
        _spec("Teknik denge değişimi", tech_change_score, 0.7, _fmt_change(technical_balance["cur"], technical_balance["ly"])),
        _spec("Net kâr değişimi (yıllık)" + (" · kalite indirimi" if non_operating else ""), net_change_score_quality_adjusted, 0.8, _fmt_change(net["cur"], net["ly"])),
    ]
    karlilik = [
        _spec("Teknik denge marjı", _score_higher(tech_margin, 0.0, 15.0), 1.0, _fmt_level_pct(tech_margin)),
        _spec("ROE (son 12 ay)", _score_higher(roe, 4.0, 25.0), 0.9, _fmt_level_pct(roe)),
        _spec("Net kâr değişimi (aynı çeyrek)" + (" · kalite indirimi" if non_operating else ""), net_change_score_quality_adjusted, 1.0, _fmt_change(net["cur"], net["ly"])),
        _spec("Net kâr değişimi (son 12 ay)" + (" · kalite indirimi" if non_operating else ""), net_ttm_score_quality_adjusted, 0.6, _fmt_change(net["ttm"], net["ttm_ly"])),
        _spec("Kâr kalitesi (teknik denge desteği)", quality["score"], 0.8, str(quality["code"] or "-")),
    ]
    eq = equity["cur"]
    negative_equity = eq is not None and eq <= 0
    bilanco = [
        _spec("Özkaynak büyümesi (yıllık)", _score_growth(equity["yoy"], 30.0), 1.0, _fmt_pct(equity["yoy"])),
        _spec("Özkaynak / teknik karşılık", _score_higher(equity_reserve, 0.3, 1.2), 0.9, _fmt_ratio(equity_reserve)),
        _spec("Borç / özkaynak", None if negative_equity else _score_lower(debt_equity, 0.2, 2.0), 0.9, _fmt_ratio(debt_equity)),
    ]
    if negative_equity:
        bilanco.append(_spec("Özkaynak negatif", 0.0, 3.0, _fmt_amount(eq), optional=True))
    tech_sign_score: Optional[float] = None
    if technical_balance["cur"] is not None:
        tech_sign_score = 7.6 if technical_balance["cur"] > 0 else 2.2
    nakit = [
        _spec("Nakit / teknik karşılık", _score_higher(cash_reserve, 0.4, 1.2), 1.1, _fmt_ratio(cash_reserve)),
        _spec("Teknik denge işareti", tech_sign_score, 0.9, _fmt_amount(technical_balance["cur"])),
        _spec("Nakit benzeri varlık büyümesi", _score_growth(cash_like["yoy"], 30.0), 0.6, _fmt_pct(cash_like["yoy"])),
    ]

    flags = _profit_regime_flags(
        regime, net, period_label=period_label, ly_label=ly_label,
        base_margin=_safe_ratio(net["ly"], equity["ly"], 100.0), low_base_threshold=0.5,
    )
    flags.extend(_quality_flags(quality, operating_label="teknik denge", period_label=period_label))
    if technical_balance["cur"] is not None and technical_balance["cur"] < 0 and quality["code"] != "faaliyet_zarari_net_kar":
        flags.append(_flag("teknik_zarar", "negative", "Teknik denge negatif", f"{period_label} teknik denge {_fmt_amount(technical_balance['cur'])}; temel sigortacılık sonucu negatiftir. Diğer kâr kalemlerinin dökümü verilmediği için net sonuca etkisi belirlenemiyor.", "karlilik"))
    if negative_equity:
        flags.append(_flag("negatif_ozkaynak", "negative", "Özkaynak negatif", f"Özkaynaklar {_fmt_amount(eq)}.", "bilanco"))

    guards = _empty_guards()
    if non_operating:
        _tighten(guards, "karlilik", maximum=0.3)
        _tighten(guards, "buyume", maximum=0.3)
    if negative_equity:
        _tighten(guards, "bilanco", maximum=0.0)
    overall_cap: Optional[float] = 3.5 if negative_equity else None
    if net["cur"] is not None and net["cur"] < 0 and net["ttm"] is not None and net["ttm"] < 0:
        overall_cap = min(overall_cap or 10.0, 5.0)

    facts = {
        "latest_period": period_label,
        "same_quarter_last_year": ly_label,
        "profit_regime": regime,
        "earnings_quality": quality["code"],
        "prim_uretimi": _flow_fact(gross_premium),
        "alinan_net_primler": _flow_fact(net_premium),
        "teknik_denge": _flow_fact(technical_balance),
        "net_kar": _flow_fact(net),
        "teknik_denge_marji": _fmt_level_pct(tech_margin),
        "roe": _fmt_level_pct(roe),
    }
    return {
        "buckets": {"buyume": buyume, "karlilik": karlilik, "bilanco": bilanco, "nakit_akisi": nakit},
        "flags": flags,
        "facts": facts,
        "guards": guards,
        "bucket_caps": {"bilanco": 2.0} if negative_equity else {},
        "overall_cap": overall_cap,
        "season_flags": [
            _season(h, "prim_uretimi")["season_flag"],
            _season(h, "net_kar")["season_flag"],
            _season(h, "teknik_denge")["season_flag"],
        ],
    }


_ANALYZERS: Dict[str, Callable[[_History], Dict[str, Any]]] = {
    "generic": _analyze_generic,
    "bank": _analyze_bank,
    "insurance": _analyze_insurance,
}


# ---------------------------------------------------------------------------
# Scorecard assembly
# ---------------------------------------------------------------------------


def _bucket_from_specs(specs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    possible = sum(spec["weight"] for spec in specs if not spec["optional"] and spec["weight"] > 0)
    present = [spec for spec in specs if spec["score"] is not None and spec["weight"] > 0]
    covered = sum(spec["weight"] for spec in present if not spec["optional"])
    coverage = covered / possible if possible > 0 else 0.0
    signals = [
        {
            "label": spec["label"],
            "score": round(_clamp(float(spec["score"]), 0.0, 10.0), 2),
            "weight": round(float(spec["weight"]), 2),
            "value": spec["value"] or "",
        }
        for spec in present
    ]
    return signals, coverage


def _weighted_score(signals: List[Dict[str, Any]]) -> float:
    total_weight = sum(item["weight"] for item in signals)
    if total_weight <= 0:
        return 5.0
    weighted = sum(item["score"] * item["weight"] for item in signals)
    return weighted / total_weight


def _weighted_overall(subscores: List[Dict[str, Any]], weights: Dict[str, float]) -> float:
    if not subscores:
        return 5.0
    total = 0.0
    weight_sum = 0.0
    for item in subscores:
        weight = float(weights.get(item["key"], 1.0))
        total += float(item["score"]) * weight
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else 5.0


def _bucket_summary(label: str, score: float, signals: List[Dict[str, Any]], flags: List[Dict[str, Any]]) -> str:
    if not signals:
        return f"{label} tarafında veri sınırlı; puan nötr tutuldu."
    ordered = sorted(signals, key=lambda item: item["score"])
    strongest = ordered[-1]
    weakest = ordered[0]
    text = f"{label} görünümü {_bucket_tone(score)} ({score:.1f}/10)"
    parts: List[str] = []
    if strongest["score"] >= 6.0:
        parts.append(f"{strongest['label']} destekliyor")
    if weakest["score"] <= 4.5 and weakest["label"] != strongest["label"]:
        parts.append(f"{weakest['label']} baskı yaratıyor")
    if parts:
        text += "; " + ", ".join(parts)
    text += "."
    if flags:
        text += f" {flags[0]['title']}."
    return text


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


def _overall_summary(overall_score: float, subscores: List[Dict[str, Any]], flags: List[Dict[str, Any]]) -> str:
    best = max(subscores, key=lambda item: item["score"])
    worst = min(subscores, key=lambda item: item["score"])
    text = (
        f"Genel görünüm {_bucket_tone(overall_score)}; en güçlü alan {best['label']} ({best['score']:.1f}), "
        f"en zayıf alan {worst['label']} ({worst['score']:.1f})."
    )
    key_flag = next((flag for flag in flags if flag["tone"] in {"negative", "warning"}), None)
    if key_flag:
        text += f" Dikkat: {key_flag['title']}."
    return text


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


def _deterministic_commentary(scorecard: Dict[str, Any], company_kind: str) -> Dict[str, Any]:
    subscores = list(scorecard.get("subscores") or [])
    flags = list(scorecard.get("flags") or [])
    best = max(subscores, key=lambda item: item["score"])
    worst = min(subscores, key=lambda item: item["score"])
    ordered = sorted(subscores, key=lambda item: item["score"])
    lead_flag = next((flag for flag in flags if flag["tone"] in {"positive", "negative"}), None)
    headline = f"{scorecard['overall_label']} görünüm: {best['label']} önde, {worst['label']} izlenmeli."
    if lead_flag:
        headline = f"{scorecard['overall_label']} görünüm: {lead_flag['title']}."
    bullets: List[str] = [f"{flag['title']}: {flag['detail']}" for flag in flags[:3]]
    bullets.append(f"{best['label']} skoru {best['score']}/10. {best['summary']}")
    if worst["key"] != best["key"]:
        bullets.append(f"{worst['label']} skoru {worst['score']}/10. {worst['summary']}")
    bullets.append(scorecard["seasonality_note"])
    risk_flag = next((flag for flag in flags if flag["tone"] in {"negative", "warning"}), None)
    if risk_flag:
        risk_note = f"{risk_flag['title']}: {risk_flag['detail']}"
    else:
        risk_note = f"En belirgin kırılganlık {worst['label']} tarafında. Bir sonraki dönemde bu başlık teyit edilmeli."
    return {
        "headline": headline,
        "bullets": bullets[:5],
        "risk_note": risk_note,
        "watch_metrics": _watch_metrics(company_kind, ordered),
    }


def compute_base_analysis(history_context: Dict[str, Any]) -> Dict[str, Any]:
    company_kind = str(history_context.get("company_kind") or "generic").strip().lower()
    if company_kind not in COMPANY_KINDS:
        company_kind = "generic"

    history = _History(history_context)
    analysis = _ANALYZERS[company_kind](history)
    bucket_labels = BUCKET_LABELS.get(company_kind, BUCKET_LABELS["generic"])
    weights = BUCKET_WEIGHTS.get(company_kind, BUCKET_WEIGHTS["generic"])
    flags: List[Dict[str, Any]] = analysis["flags"]
    guards: Dict[str, Dict[str, float]] = analysis["guards"]
    bucket_caps: Dict[str, float] = analysis["bucket_caps"]

    debug_lines = [
        f"score history_quarters={len(history_context.get('quarters') or [])}",
        f"score company_kind={company_kind}",
        f"score latest={_History.label(history.latest)}",
        "score flags=" + (",".join(flag["code"] for flag in flags) or "yok"),
    ]
    subscores: List[Dict[str, Any]] = []
    coverages: Dict[str, float] = {}
    for bucket_key in SUBSCORE_KEYS:
        signals, coverage = _bucket_from_specs(analysis["buckets"].get(bucket_key, []))
        coverages[bucket_key] = coverage
        raw = _weighted_score(signals)
        # Thin evidence pulls the bucket toward neutral instead of letting one
        # signal define the whole score.
        score = 5.0 + (raw - 5.0) * _clamp(0.4 + coverage, 0.4, 1.0)
        cap = bucket_caps.get(bucket_key)
        if cap is not None:
            score = min(score, cap)
        score = _round1(score)
        if score >= 8.0:
            _tighten(guards, bucket_key, minimum=-0.6)
        if score <= 2.5:
            _tighten(guards, bucket_key, maximum=0.6)
        label = bucket_labels.get(bucket_key, bucket_key)
        bucket_flags = [flag for flag in flags if flag.get("bucket") == bucket_key]
        subscores.append(
            {
                "key": bucket_key,
                "label": label,
                "score": score,
                "base_score": score,
                "ai_adjustment": 0.0,
                "weight": weights.get(bucket_key, 0.25),
                "confidence": _confidence_label(coverage),
                "summary": _bucket_summary(label, score, signals, bucket_flags),
                "signals": signals,
            }
        )
        signal_debug = ", ".join(f"{item['label']}={item['score']:.1f}" for item in signals[:6]) or "veri_yok"
        debug_lines.append(f"base {bucket_key}={score:.1f} coverage={coverage:.2f} signals={signal_debug}")

    overall_raw = _weighted_overall(subscores, weights)
    overall_cap = analysis.get("overall_cap")
    if overall_cap is not None:
        overall_raw = min(overall_raw, overall_cap)
    overall_score = _round1(overall_raw)
    overall_coverage = sum(coverages[key] * weights.get(key, 0.25) for key in SUBSCORE_KEYS) / max(sum(weights.values()), 1e-9)
    if len(history.keys) < 5:
        overall_coverage = min(overall_coverage, 0.49)

    scorecard = {
        "overall_score": overall_score,
        "overall_label": _overall_label(overall_score),
        "summary": _overall_summary(overall_score, subscores, flags),
        "seasonality_note": _seasonality_note([flag for flag in analysis["season_flags"] if flag and flag != "unknown"]),
        "score_source": SCORE_SOURCE_DETERMINISTIC_ONLY,
        "confidence": _confidence_label(overall_coverage),
        "overall_cap": overall_cap,
        "weights": dict(weights),
        "flags": flags,
        "adjustment_guards": {key: dict(value) for key, value in guards.items()},
        "subscores": subscores,
    }
    commentary = _deterministic_commentary(scorecard, company_kind)
    return {
        "company_kind": company_kind,
        "scorecard": scorecard,
        "facts": analysis["facts"],
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
    guards = base_scorecard.get("adjustment_guards") or {}
    weights = base_scorecard.get("weights") or {key: 1.0 for key in SUBSCORE_KEYS}

    ai_subscores = {
        str(item.get("key") or "").strip(): item
        for item in (ai_adjustments.get("subscores") or [])
        if isinstance(item, dict) and str(item.get("key") or "").strip()
    }

    merged_subscores: List[Dict[str, Any]] = []
    for item in list(base_scorecard.get("subscores") or []):
        raw_adjustment = ai_subscores.get(item["key"], {}).get("adjustment")
        adjustment = float(raw_adjustment or 0.0) if isinstance(raw_adjustment, (int, float)) and not isinstance(raw_adjustment, bool) else 0.0
        bounded_adjustment = _clamp(adjustment, -AI_SUBSCORE_ADJUSTMENT_LIMIT, AI_SUBSCORE_ADJUSTMENT_LIMIT)
        guard = guards.get(item["key"])
        if isinstance(guard, dict):
            bounded_adjustment = _clamp(
                bounded_adjustment,
                float(guard.get("min", -AI_SUBSCORE_ADJUSTMENT_LIMIT)),
                float(guard.get("max", AI_SUBSCORE_ADJUSTMENT_LIMIT)),
            )
        base_score = float(item.get("base_score", item["score"]))
        ai_summary = str(ai_subscores.get(item["key"], {}).get("summary") or "").strip()
        merged_subscores.append(
            {
                **item,
                "base_score": base_score,
                "ai_adjustment": round(bounded_adjustment, 2),
                "score": _round1(base_score + bounded_adjustment),
                "summary": ai_summary or item["summary"],
            }
        )

    # The overall score is derived only from the displayed adjusted subscores;
    # the model's overall_adjustment field is retained for schema compatibility.
    overall_raw = _weighted_overall(merged_subscores, weights)
    overall_cap = base_scorecard.get("overall_cap")
    if isinstance(overall_cap, (int, float)) and not isinstance(overall_cap, bool):
        overall_raw = min(overall_raw, float(overall_cap))
    overall_score = _round1(overall_raw)
    summary = str(ai_adjustments.get("summary") or "").strip()
    if not summary:
        summary = _overall_summary(overall_score, merged_subscores, base_scorecard.get("flags") or [])
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
