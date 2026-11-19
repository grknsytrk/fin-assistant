from __future__ import annotations

from datetime import date, datetime
import os
from typing import Any, Dict, List, Optional

from app.kap_api import _kas

_KAP_SNAPSHOT_RESPONSE_CACHE_TTL = int(os.getenv("RAGFIN_KAP_SNAPSHOT_RESPONSE_CACHE_TTL_SECONDS", "300"))


def _kap_snapshot_response_cache_key(company: str, max_quarters: int) -> str:
    from app.kap_service import normalize_kap_symbol
    from src.kap_fetcher import KAP_CACHE_SCHEMA_VERSION

    normalized = normalize_kap_symbol(str(company or "").strip().upper().replace(".", ""))
    return f"api:kap-snapshot:{normalized}:quarters={max_quarters}:schema={KAP_CACHE_SCHEMA_VERSION}:v2"


def _annotate_kap_response_cache(
    payload: Dict[str, Any],
    *,
    cache_hit: bool,
    ttl_seconds: int = _KAP_SNAPSHOT_RESPONSE_CACHE_TTL,
) -> Dict[str, Any]:
    kas = _kas()
    out = dict(payload)
    status = kas._cache_status()
    out["response_cache_hit"] = cache_hit
    out["response_cache_backend"] = status.get("cache_backend")
    out["response_cache_ttl_seconds"] = ttl_seconds
    return out


def _build_kap_snapshot_response(company: str, *, refresh: bool, max_quarters: int) -> Dict[str, Any]:
    from app.kap_service import get_kap_snapshot, normalize_snapshot_for_frontend

    kas = _kas()
    raw = get_kap_snapshot(
        company=company,
        cfg=kas.CONFIG.kap,
        processed_dir=kas.CONFIG.paths.processed_dir,
        force_refresh=refresh,
        max_quarters=max_quarters,
        use_cache_when_complete=not refresh,
    )
    if not raw.get("ok"):
        raise RuntimeError(raw.get("error") or "KAP snapshot unavailable")
    snapshot_symbol = str(raw.get("stock_code") or raw.get("company") or company).strip().upper()
    snapshot_mtime_ns = None
    try:
        snapshot_mtime_ns = (
            kas.CONFIG.paths.processed_dir / "kap_cache" / f"{snapshot_symbol}.json"
        ).stat().st_mtime_ns
    except OSError:
        pass
    normalized = normalize_snapshot_for_frontend(raw)
    try:
        financial_snapshot = kas.extract_financial_snapshot_from_kap_payload(
            snapshot_symbol,
            normalized,
            source_mtime_ns=snapshot_mtime_ns,
        )
        if financial_snapshot:
            kas.upsert_financial_snapshots(kas.CONFIG.paths.processed_dir, [financial_snapshot])
    except Exception:
        kas.LOGGER.exception("KAP financial snapshot upsert failed for %s", snapshot_symbol)
    kas._upsert_stock_reference_from_kap_payload(company, raw, source="kap")

    symbol = normalized.get("stock_code") or company
    with kas.concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        price_future = executor.submit(kas._fetch_kap_price_payload, symbol)
        multiples_future = executor.submit(kas._fetch_isyatirim_multiples, symbol)
        price_payload = price_future.result()
        isyatirim_payload = multiples_future.result()
    normalized["valuation"] = kas._build_kap_valuation_payload(
        snapshot=normalized,
        price_payload=price_payload,
        isyatirim_payload=isyatirim_payload,
    )
    return normalized


def _cached_kap_snapshot_response(company: str) -> Optional[Dict[str, Any]]:
    """Read persisted statements and quote caches without contacting providers."""
    from app.kap_service import normalize_kap_symbol, normalize_snapshot_for_frontend
    from src.kap_fetcher import _read_first_cache, KAP_CACHE_SCHEMA_VERSION

    kas = _kas()
    symbol = normalize_kap_symbol(company)
    _, raw = _read_first_cache(kas.CONFIG.paths.processed_dir, symbol)
    if not raw or raw.get("schema_version") != KAP_CACHE_SCHEMA_VERSION or not raw.get("quarters"):
        return None
    normalized = normalize_snapshot_for_frontend(raw, allow_network=False)
    normalized["cache_hit"] = True
    normalized["cache_stale"] = True
    price = kas._shared_cache_get_dict(f"api:market:stock-price:{symbol}:v1") or {}
    multiples = kas._shared_cache_get_dict(f"api:isyatirim-multiples:{symbol}") or {}
    normalized["valuation"] = kas._build_kap_valuation_payload(
        snapshot=normalized, price_payload=price, isyatirim_payload=multiples,
    )
    return normalized


def _build_kap_valuation_payload(
    *,
    snapshot: Dict[str, Any],
    price_payload: Dict[str, Any],
    isyatirim_payload: Dict[str, Any],
) -> Dict[str, Any]:
    kas = _kas()
    financial_company = snapshot.get("company_kind") in {"bank", "insurance"}
    quarters_raw = snapshot.get("quarters")
    quarters = [q for q in quarters_raw if isinstance(q, dict)] if isinstance(quarters_raw, list) else []
    quarters_sorted = sorted(quarters, key=kas._quarter_sort_key)
    latest = quarters_sorted[-1] if quarters_sorted else None

    ttm_net_kar = kas._build_ttm_from_reports(quarters_sorted, "net_kar")
    ttm_favok = kas._build_ttm_from_reports(quarters_sorted, "favok")

    ozkaynaklar = (
        kas._extract_quarter_metric(latest, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
        if latest
        else None
    )
    net_borc = (
        kas._extract_quarter_metric(latest, "net_borc", priority=["metrics", "metrics_ytd"])
        if latest
        else None
    )

    shares_outstanding = None
    share_source = None
    if latest:
        shares_outstanding = kas._extract_quarter_metric(
            latest,
            "odenmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        )
        if shares_outstanding is not None:
            share_source = "odenmis_sermaye"
        else:
            shares_outstanding = kas._extract_quarter_metric(
                latest,
                "cikarilmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            )
            if shares_outstanding is not None:
                share_source = "cikarilmis_sermaye"
    if shares_outstanding is not None and shares_outstanding <= 0:
        shares_outstanding = None
        share_source = None

    assumptions: List[str] = []
    share_nominal_value = None
    if shares_outstanding is not None:
        share_nominal_value = 1.0
        assumptions.append("Hisse adedi, nominal pay degeri 1 TL varsayimiyla hesaplandi.")

    price_ok = bool(price_payload.get("ok"))
    price = float(price_payload["price"]) if price_ok and isinstance(price_payload.get("price"), (int, float)) else None
    price_currency = str(price_payload.get("currency", "TRY")) if price_ok else None
    price_as_of = price_payload.get("as_of") if price_ok else None

    market_cap = price * shares_outstanding if price is not None and shares_outstanding is not None else None
    enterprise_value = market_cap + net_borc if market_cap is not None and net_borc is not None else None

    fk = kas._parse_tr_decimal(isyatirim_payload.get("fk")) if isyatirim_payload.get("ok") else None
    pd_dd = kas._parse_tr_decimal(isyatirim_payload.get("pd_dd")) if isyatirim_payload.get("ok") else None
    fd_favok = kas._parse_tr_decimal(isyatirim_payload.get("fd_favok")) if isyatirim_payload.get("ok") else None

    if financial_company:
        fd_favok = None
        enterprise_value = None
        ttm_favok = None

    return {
        "price": price,
        "price_currency": price_currency,
        "price_as_of": price_as_of,
        "price_source": "yahoo_finance_chart",
        "shares_outstanding": shares_outstanding,
        "share_source": share_source,
        "share_nominal_value": share_nominal_value,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "ttm_net_kar": ttm_net_kar,
        "ttm_favok": ttm_favok,
        "fk": fk,
        "pd_dd": pd_dd,
        "fd_favok": fd_favok,
        "fk_prim_iskonto_pct": kas._parse_tr_decimal(isyatirim_payload.get("fk_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "fd_favok_prim_iskonto_pct": kas._parse_tr_decimal(isyatirim_payload.get("fd_favok_prim_iskonto_pct"))
        if isyatirim_payload.get("ok") and not financial_company
        else None,
        "pd_dd_prim_iskonto_pct": kas._parse_tr_decimal(isyatirim_payload.get("pd_dd_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "multiples_source": isyatirim_payload.get("source") if isyatirim_payload.get("ok") else None,
        "multiples_note": isyatirim_payload.get("note") if isyatirim_payload.get("ok") else None,
        "multiples_as_of": isyatirim_payload.get("fetched_at") if isyatirim_payload.get("ok") else None,
        "multiples_error": isyatirim_payload.get("error") if not isyatirim_payload.get("ok") else None,
        "assumptions": assumptions,
    }


def _kap_quarter_end_date(quarter: Dict[str, Any]) -> Optional[date]:
    kas = _kas()
    try:
        year = int(quarter.get("year") or 0)
        period = int(quarter.get("period") or 0)
    except (TypeError, ValueError):
        return None
    if year <= 0 or period not in (1, 2, 3, 4):
        return None
    return kas.date(year, period * 3, (31, 30, 30, 31)[period - 1])


def _kap_history_price_start_date(quarters: List[Dict[str, Any]]) -> Optional[date]:
    kas = _kas()
    dates = [
        quarter_end
        for quarter in quarters
        if (quarter_end := kas._kap_quarter_end_date(quarter)) is not None
    ]
    return min(dates) if dates else None


def _kap_history_trading_date(publish_date: Any) -> Optional[date]:
    kas = _kas()
    value = str(publish_date or "").strip()
    if not value:
        return None
    parsed: Optional[datetime] = None
    try:
        parsed = kas.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for pattern in (
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y",
            "%Y.%m.%d %H:%M:%S",
            "%Y.%m.%d",
            "%Y/%m/%d",
        ):
            try:
                parsed = kas.datetime.strptime(value, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=kas._TURKEY_TIMEZONE)
    else:
        parsed = parsed.astimezone(kas._TURKEY_TIMEZONE)
    trading_date = parsed.date()
    if parsed.hour * 60 + parsed.minute > 18 * 60 + 10:
        trading_date += kas.timedelta(days=1)
    return trading_date


def _kap_history_price_date(value: Any) -> Optional[date]:
    kas = _kas()
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = kas.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return kas.date.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(kas._TURKEY_TIMEZONE)
    return parsed.date()


def _build_kap_historical_ttm(
    quarters: List[Dict[str, Any]],
    index: int,
    metric_key: str,
) -> Optional[float]:
    kas = _kas()
    latest = quarters[index]
    try:
        year = int(latest.get("year") or 0)
        period = int(latest.get("period") or 0)
    except (TypeError, ValueError):
        return None
    if period == 4:
        annual = kas._extract_quarter_metric(latest, metric_key, priority=["metrics_ytd", "metrics"])
        return annual if annual is not None and kas.math.isfinite(annual) else None
    if period not in (1, 2, 3):
        return None

    def is_prior_year_end(quarter: Dict[str, Any]) -> bool:
        try:
            return int(quarter.get("year") or 0) == year - 1 and int(quarter.get("period") or 0) == 4
        except (TypeError, ValueError):
            return False

    prior_year_end = next(
        (
            quarter
            for quarter in reversed(quarters[:index])
            if is_prior_year_end(quarter)
        ),
        None,
    )
    if prior_year_end is None:
        return None

    prior_fy = kas._extract_quarter_metric(
        prior_year_end,
        metric_key,
        priority=["metrics_ytd", "metrics"],
    )
    current_ytd = kas._extract_quarter_metric(latest, metric_key, priority=["metrics_ytd"])
    comparative_ytd = kas._extract_quarter_metric(
        latest,
        metric_key,
        priority=["metrics_ytd_comparative"],
    )
    if any(
        value is None or not kas.math.isfinite(value)
        for value in (prior_fy, current_ytd, comparative_ytd)
    ):
        return None

    # KAP's prior-year annual flow and the current report's comparative YTD
    # can be in different TMS 29 price bases. ``analysis_multiplier`` converts
    # each period's reported values to the latest snapshot basis, so convert
    # the prior annual value to the current report basis using the prior/current
    # multiplier ratio before replacing the comparable YTD amount.
    current_factor = kas._as_finite_float(latest.get("analysis_multiplier")) or 1.0
    prior_factor = kas._as_finite_float(prior_year_end.get("analysis_multiplier")) or 1.0
    if current_factor <= 0 or prior_factor <= 0:
        scale = 1.0
    else:
        scale = prior_factor / current_factor
    return float(prior_fy * scale + current_ytd - comparative_ytd)


def _fetch_kap_valuation_history_prices(
    symbol: str,
    *,
    start_date: date,
) -> tuple[Optional[Dict[str, Any]], str, bool, bool]:
    kas = _kas()
    normalized = str(symbol or "").strip().upper()
    end_date = kas.date.today() - kas.timedelta(days=1)
    if not normalized or start_date > end_date:
        return None, "unavailable", False, False

    # Anchor the start year to cover the full 44-quarter (~11-year) window so
    # progressive 5Q -> 20Q -> 44Q loads reuse one series per stock, and omit
    # end_date from the cache key so the 7-day SWR cache survives midnight.
    clamped_start_year = max(2010, start_date.year)
    default_history_start_year = max(2014, end_date.year - 11)
    start_of_year = kas.date(min(clamped_start_year, default_history_start_year), 1, 1)
    cache_key = (
        f"api:kap:valuation-history-prices:v2:{normalized}:"
        f"from={start_of_year.isoformat()}"
    )
    failure_cache_key = f"{cache_key}:failure"
    local_key = f"{normalized}:{start_of_year.isoformat()}"

    def build() -> Optional[Dict[str, Any]]:
        chart = kas._fetch_yahoo_chart_period_raw(
            f"{normalized}.IS",
            interval="1d",
            start_date=start_of_year,
            end_date=end_date,
        )
        if not chart.get("ok"):
            kas._shared_cache_set(
                failure_cache_key,
                {"error": "historical_price_unavailable"},
                ttl_seconds=60,
            )
            return None
        points = []
        for point in chart.get("points") or []:
            point_date = kas._kap_history_price_date(point.get("time"))
            close = kas._as_finite_float(point.get("close"))
            if point_date is None or close is None or close <= 0:
                continue
            points.append({"date": point_date.isoformat(), "close": close})
        if not points:
            kas._shared_cache_set(
                failure_cache_key,
                {"error": "historical_price_unavailable"},
                ttl_seconds=60,
            )
            return None
        try:
            kas._get_cache().delete(failure_cache_key)
        except Exception:
            pass
        splits = []
        for event in chart.get("splits") or []:
            event_date = kas._kap_history_price_date(event.get("date"))
            ratio = kas._as_finite_float(event.get("ratio"))
            if event_date is not None and ratio is not None and ratio > 0:
                splits.append({"date": event_date.isoformat(), "ratio": ratio})
        return {
            "points": sorted(points, key=lambda point: point["date"]),
            "splits": sorted(splits, key=lambda split: split["date"]),
            "source": "yahoo_finance_chart",
            "as_of": end_date.isoformat(),
        }

    local_entry = kas._KAP_VALUATION_HISTORY_PRICE_CACHE.get(local_key)
    if kas._swr_entry_is_fresh(local_entry):
        return kas._swr_entry_payload(local_entry), "local_hit", False, False
    shared_entry = kas._shared_cache_get_dict(cache_key)
    if kas._swr_entry_is_fresh(shared_entry):
        kas._KAP_VALUATION_HISTORY_PRICE_CACHE[local_key] = shared_entry
        return kas._swr_entry_payload(shared_entry), "shared_hit", False, False

    stale_entry = shared_entry if kas._swr_entry_is_stale(shared_entry) else local_entry
    if kas._swr_entry_is_stale(stale_entry):
        kas._KAP_VALUATION_HISTORY_PRICE_CACHE[local_key] = stale_entry
        pending = False
        if not kas._shared_cache_get_dict(failure_cache_key):
            pending = kas._schedule_swr_revalidation(
                cache_key=cache_key,
                factory=build,
                fresh_ttl_seconds=kas._KAP_VALUATION_HISTORY_PRICE_TTL,
                stale_ttl_seconds=kas._KAP_VALUATION_HISTORY_PRICE_STALE_TTL,
            )
        return kas._swr_entry_payload(stale_entry), "stale", True, pending

    if kas._shared_cache_get_dict(failure_cache_key):
        return None, "unavailable", False, False

    def build_entry() -> Optional[Dict[str, Any]]:
        built = build()
        if not isinstance(built, dict):
            return None
        return kas._swr_cache_entry(
            built,
            fresh_ttl_seconds=kas._KAP_VALUATION_HISTORY_PRICE_TTL,
            stale_ttl_seconds=kas._KAP_VALUATION_HISTORY_PRICE_STALE_TTL,
        )

    entry, status = kas._get_or_set_single_flight(
        cache_key,
        ttl_seconds=max(1, kas._KAP_VALUATION_HISTORY_PRICE_STALE_TTL),
        factory=build_entry,
        lock_key=f"single-flight:{cache_key}",
        lock_ttl_seconds=kas._MARKET_SWR_LOCK_TTL_SECONDS,
        wait_timeout_seconds=kas._MARKET_SWR_WAIT_TIMEOUT_SECONDS,
        cache_usable=kas._swr_entry_is_fresh,
    )
    payload = kas._swr_entry_payload(entry)
    if payload is not None:
        kas._KAP_VALUATION_HISTORY_PRICE_CACHE[local_key] = entry
        return payload, status, False, False
    if status == "pending":
        return None, "pending", False, True
    return None, "unavailable", False, False


def _build_kap_historical_multiples(*, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    kas = _kas()
    company = str(snapshot.get("stock_code") or snapshot.get("company") or "").strip().upper()
    quarters_raw = snapshot.get("quarters")
    quarters = sorted(
        [quarter for quarter in quarters_raw if isinstance(quarter, dict)]
        if isinstance(quarters_raw, list)
        else [],
        key=kas._quarter_sort_key,
    )
    start_date = kas._kap_history_price_start_date(quarters)
    if not company or not quarters or start_date is None:
        return {
            "ok": False,
            "pending": False,
            "company": company,
            "periods": [],
            "available_quarters": len(quarters),
            "error": "Tarihsel çarpanlar için rapor dönemi bulunamadı.",
        }

    prices_payload, price_cache_status, price_cache_stale, price_refresh_pending = (
        kas._fetch_kap_valuation_history_prices(company, start_date=start_date)
    )
    if not prices_payload:
        return {
            "ok": False,
            "pending": price_refresh_pending or price_cache_status == "pending",
            "refresh_pending": price_refresh_pending,
            "company": company,
            "periods": [],
            "available_quarters": len(quarters),
            "price_cache_status": price_cache_status,
            "error": (
                "Tarihsel fiyat verisi şu an alınamıyor."
                if price_cache_status == "unavailable"
                else "Tarihsel fiyat verisi hazırlanıyor."
            ),
        }

    price_points = prices_payload.get("points") or []
    split_events = prices_payload.get("splits") or []
    price_dates = [str(point.get("date") or "") for point in price_points]
    reporting_quarters = []
    for quarter in quarters:
        # The original KAP values keep each filing's own nominal basis. Using
        # today's TMS 29 restatement with that period's price would distort
        # historical multiples.
        reporting_quarters.append({
            **quarter,
            "metrics": quarter.get("metrics_original") or quarter.get("metrics") or {},
            "metrics_ytd": quarter.get("metrics_ytd_original") or quarter.get("metrics_ytd") or {},
            "metrics_quarterly": quarter.get("metrics_quarterly_original")
            or quarter.get("metrics_quarterly")
            or {},
            "metrics_comparative": quarter.get("metrics_comparative") or {},
            "metrics_ytd_comparative": quarter.get("metrics_ytd_comparative") or {},
        })

    periods = []
    for index, quarter in enumerate(reporting_quarters):
        report_date = kas._kap_history_trading_date(quarter.get("publish_date"))
        price_index = None
        if report_date is not None:
            target = report_date.isoformat()
            candidate = kas.bisect.bisect_left(price_dates, target)
            if candidate < len(price_dates):
                candidate_date = kas._kap_history_price_date(price_dates[candidate])
                if candidate_date and (candidate_date - report_date).days <= 7:
                    price_index = candidate

        ttm_net_kar = kas._build_kap_historical_ttm(reporting_quarters, index, "net_kar")
        ttm_satis = kas._build_kap_historical_ttm(reporting_quarters, index, "satis_gelirleri")
        ttm_favok = kas._build_kap_historical_ttm(reporting_quarters, index, "favok")
        equity = kas._extract_quarter_metric(quarter, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
        net_debt = kas._extract_quarter_metric(quarter, "net_borc", priority=["metrics", "metrics_ytd"])
        shares = kas._extract_quarter_metric(
            quarter,
            "odenmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        )
        share_source = "odenmis_sermaye" if shares is not None else None
        if shares is None:
            shares = kas._extract_quarter_metric(
                quarter,
                "cikarilmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            )
            if shares is not None:
                share_source = "cikarilmis_sermaye"
        if shares is not None and shares <= 0:
            shares = None

        price_date = price_dates[price_index] if price_index is not None else None
        price = (
            kas._as_finite_float(price_points[price_index].get("close"))
            if price_index is not None
            else None
        )
        quarter_end = kas._kap_quarter_end_date(quarter)
        split_factor = 1.0
        if quarter_end is not None:
            for event in split_events:
                event_date = kas._kap_history_price_date(event.get("date"))
                ratio = kas._as_finite_float(event.get("ratio"))
                if event_date and event_date > quarter_end and ratio and ratio > 0:
                    split_factor *= ratio
        adjusted_shares = shares * split_factor if shares is not None else None
        market_cap = (
            price * adjusted_shares
            if price is not None and adjusted_shares is not None
            else None
        )
        financial_company = snapshot.get("company_kind") in {"bank", "insurance"}
        enterprise_value = (
            market_cap + net_debt
            if market_cap is not None and net_debt is not None and not financial_company
            else None
        )
        period_key = str(quarter.get("quarter") or "")
        periods.append({
            "quarter": period_key,
            "year": int(quarter.get("year") or 0),
            "period": int(quarter.get("period") or 0),
            "publish_date": str(quarter.get("publish_date") or ""),
            "price_date": price_date,
            "price": round(price, 4) if price is not None else None,
            "shares_outstanding": round(adjusted_shares, 4) if adjusted_shares is not None else None,
            "share_source": share_source,
            "split_adjustment": round(split_factor, 8),
            "market_cap": round(market_cap, 2) if market_cap is not None else None,
            "ttm_net_kar": round(ttm_net_kar, 2) if ttm_net_kar is not None else None,
            "ttm_satis": round(ttm_satis, 2) if ttm_satis is not None else None,
            "equity": round(equity, 2) if equity is not None else None,
            "ttm_favok": round(ttm_favok, 2) if ttm_favok is not None else None,
            "net_debt": round(net_debt, 2) if net_debt is not None else None,
            "fk": kas._round_market_ratio(market_cap, ttm_net_kar)
            if ttm_net_kar is not None and ttm_net_kar > 0
            else None,
            "pd_satis": kas._round_market_ratio(market_cap, ttm_satis)
            if ttm_satis is not None and ttm_satis > 0
            else None,
            "pd_dd": kas._round_market_ratio(market_cap, equity)
            if equity is not None and equity > 0
            else None,
            "fd_favok": kas._round_market_ratio(enterprise_value, ttm_favok)
            if ttm_favok is not None and ttm_favok > 0
            else None,
        })

    def average(metric: str) -> Optional[float]:
        values = [
            float(period[metric])
            for period in periods
            if isinstance(period.get(metric), (int, float))
            and not isinstance(period.get(metric), bool)
            and kas.math.isfinite(float(period[metric]))
            and float(period[metric]) > 0
        ]
        return round(sum(values) / len(values), 2) if values else None

    return {
        "ok": True,
        "pending": False,
        "refresh_pending": price_refresh_pending,
        "company": company,
        "available_quarters": len(quarters),
        "calculated_counts": {
            metric: sum(1 for period in periods if period.get(metric) is not None)
            for metric in ("fk", "pd_satis", "pd_dd", "fd_favok")
        },
        "averages": {
            "fk": average("fk"),
            "pd_satis": average("pd_satis"),
            "pd_dd": average("pd_dd"),
            "fd_favok": average("fd_favok"),
        },
        "periods": periods,
        "price_source": prices_payload.get("source") or "yahoo_finance_chart",
        "price_as_of": prices_payload.get("as_of"),
        "price_cache_status": price_cache_status,
        "price_cache_stale": price_cache_stale,
        "method_note": (
            "Çarpanlar, KAP raporunun yayımlandığı tarihten sonraki ilk kapanış fiyatı ve "
            "o raporda bildirilen finansallarla hesaplanır. TTM kâr/FAVÖK rapor döneminin "
            "enflasyon bazına uyarlanır; "
            "hisse bölünmeleri fiyat ve pay adedinde aynı temele getirilir. Ödenmiş sermaye "
            "nominal pay değeri 1 TL kabul edilerek pay adedi hesabında kullanılır."
        ),
    }
