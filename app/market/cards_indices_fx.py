from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.market import _market_service


def _normalize_market_stock_card_symbol(symbol: str) -> str:
    _ms = _market_service()
    raw = str(symbol or "").strip()
    item = raw.upper()
    if item.endswith(".IS"):
        item = item[:-3]
    if not item or not _ms.re.fullmatch(r"[A-Z0-9]{2,12}", item):
        raise _ms.HTTPException(status_code=400, detail=f"Gecersiz hisse kodu: {raw or symbol}")
    return item


def _normalize_market_stock_card_symbols(symbols: str) -> List[str]:
    _ms = _market_service()
    raw_items = _ms.re.split(r"[,\s]+", str(symbols or "").strip())
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not raw.strip():
            continue
        item = _ms._normalize_market_stock_card_symbol(raw)
        if item in seen:
            continue
        normalized.append(item)
        seen.add(item)

    if len(normalized) > _ms._MARKET_STOCK_CARD_LIMIT:
        raise _ms.HTTPException(
            status_code=400,
            detail=f"En fazla {_ms._MARKET_STOCK_CARD_LIMIT} hisse karti secilebilir.",
        )
    return normalized


def _normalize_market_stock_card_tier(tier: str) -> str:
    _ms = _market_service()
    normalized = str(tier or "full").strip().lower()
    if normalized not in {"quick", "full"}:
        raise _ms.HTTPException(status_code=400, detail="Gecersiz hisse karti veri seviyesi: quick veya full")
    return normalized


def _normalize_stock_card_chart_range(chart_range: str) -> str:
    _ms = _market_service()
    normalized = str(chart_range or "1d").strip().lower()
    if normalized not in _ms._MARKET_STOCK_CARD_CHART_RANGES:
        allowed = ", ".join(sorted(_ms._MARKET_STOCK_CARD_CHART_RANGES))
        raise _ms.HTTPException(status_code=400, detail=f"Desteklenmeyen grafik araligi. Desteklenenler: {allowed}")
    return normalized


def _stock_card_chart_cache_key(symbol: str, chart_range: str) -> str:
    return f"stock-card-chart:v4:{symbol}:{chart_range}"


def _point_datetime(raw: Any) -> Optional[datetime]:
    _ms = _market_service()
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            return _ms.datetime.fromtimestamp(float(raw), tz=_ms.timezone.utc)
        except Exception:
            return None
    if not raw:
        return None
    try:
        parsed = _ms.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_ms.timezone.utc)
    return parsed.astimezone(_ms.timezone.utc)


def _numeric_chart_value(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def _normalize_stock_card_line_points(raw_points: Any) -> List[Dict[str, Any]]:
    _ms = _market_service()
    deduped: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_points, list):
        return []

    for point in raw_points:
        if not isinstance(point, dict):
            continue
        point_dt = _ms._point_datetime(point.get("time"))
        close = _ms._numeric_chart_value(point.get("close"))
        if point_dt is None or close is None or close <= 0:
            continue
        time_key = point_dt.isoformat()
        row: Dict[str, Any] = {"time": time_key, "close": close}
        for key in ("open", "high", "low", "volume"):
            value = _ms._numeric_chart_value(point.get(key))
            if value is not None:
                row[key] = value
        deduped[time_key] = row

    return [deduped[key] for key in sorted(deduped)]


def _stock_card_latest_point_dt(points: List[Dict[str, Any]]) -> Optional[datetime]:
    _ms = _market_service()
    for point in reversed(points or []):
        point_dt = _ms._point_datetime(point.get("time"))
        if point_dt is not None:
            return point_dt
    return None


def _bist_equity_session_status(now_local: datetime) -> str:
    """Return the BIST equity session phase for a Turkey-local timestamp."""
    _ms = _market_service()
    if now_local.weekday() >= 5:
        return "closed"
    minute_of_day = (now_local.hour * 60) + now_local.minute
    if minute_of_day < _ms._BIST_EQUITY_SESSION_OPEN_MINUTE:
        return "pre"
    if minute_of_day >= _ms._BIST_EQUITY_SESSION_CLOSE_MINUTE:
        return "post"
    return "open"


def _stock_card_session_state(
    points: List[Dict[str, Any]],
    *,
    market_state: Any = None,
    source: Any = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    _ms = _market_service()
    source_text = str(source or "").strip()
    now_local = (now or _ms.datetime.now(_ms._TURKEY_TIMEZONE)).astimezone(_ms._TURKEY_TIMEZONE)
    latest_dt = _ms._stock_card_latest_point_dt(points)
    latest_local = latest_dt.astimezone(_ms._TURKEY_TIMEZONE) if latest_dt else None
    last_trade_at = latest_dt.isoformat() if latest_dt else None
    last_trade_date = latest_local.date().isoformat() if latest_local else None
    previous_session = (
        source_text == "yahoo_previous_session"
        or (latest_local is not None and latest_local.date() < now_local.date())
    )
    bist_session = _ms._bist_equity_session_status(now_local)

    if previous_session:
        status = "previous_session"
        label = "Piyasa kapalı"
        is_live = False
    elif bist_session == "open":
        status = "open"
        label = "Canlı"
        is_live = True
    elif bist_session == "pre":
        status = "pre"
        label = "Açılış öncesi"
        is_live = False
    elif bist_session == "post":
        status = "post"
        label = "Kapanış sonrası"
        is_live = False
    elif latest_local is not None:
        status = "closed"
        label = "Piyasa kapalı"
        is_live = False
    else:
        status = "unknown"
        label = "Veri bekleniyor"
        is_live = False

    return {
        "session_status": status,
        "session_label": label,
        "is_live": is_live,
        "last_trade_at": last_trade_at,
        "last_trade_date": last_trade_date,
        "is_stale": previous_session,
    }


def _fetch_previous_stock_card_intraday_chart(yahoo_symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    today = _ms.datetime.now(_ms._TURKEY_TIMEZONE).date()
    errors: List[str] = []
    for offset in range(1, _ms._MARKET_STOCK_CARD_PREVIOUS_SESSION_LOOKBACK_DAYS + 1):
        session_date = today - _ms.timedelta(days=offset)
        if session_date.weekday() >= 5:
            continue
        chart = _ms._fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="5m",
            start_date=session_date,
            end_date=session_date,
        )
        points = _ms._normalize_stock_card_line_points(chart.get("points") if chart.get("ok") else [])
        if points:
            meta = dict(chart.get("meta") or {})
            meta["fallbackTradingDate"] = session_date.isoformat()
            chart = dict(chart)
            chart["meta"] = meta
            chart["points"] = points
            chart["fallback_trading_date"] = session_date.isoformat()
            return chart
        error = chart.get("error") if isinstance(chart, dict) else None
        if error:
            errors.append(str(error))
    return {
        "ok": False,
        "error": errors[-1] if errors else "previous_session_unavailable",
        "yahoo_symbol": yahoo_symbol,
        "points": [],
    }


def _fetch_stock_card_chart(symbol: str, chart_range: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    ticker = _ms._normalize_market_stock_card_symbol(symbol)
    normalized_range = _ms._normalize_stock_card_chart_range(chart_range)
    config = _ms._MARKET_STOCK_CARD_CHART_RANGES[normalized_range]
    cache_key = _ms._stock_card_chart_cache_key(ticker, normalized_range)
    shared_key = f"api:market:stock-card-chart:{ticker}:range={normalized_range}:v4"

    def build() -> Dict[str, Any]:
        yahoo_symbol = f"{ticker}.IS"
        fetched_at = _ms.datetime.now(_ms.timezone.utc).isoformat()
        chart = _ms._fetch_yahoo_chart_raw(yahoo_symbol, interval=config["interval"], range_=config["range"])
        points = _ms._normalize_stock_card_line_points(chart.get("points") if chart.get("ok") else [])
        source = "yahoo_live"
        if normalized_range == "1d" and chart.get("ok") and not points:
            fallback_chart = _ms._fetch_previous_stock_card_intraday_chart(yahoo_symbol)
            fallback_points = _ms._normalize_stock_card_line_points(fallback_chart.get("points") if fallback_chart.get("ok") else [])
            if fallback_points:
                chart = fallback_chart
                points = fallback_points
                source = "yahoo_previous_session"
        session_state = _ms._stock_card_session_state(
            points,
            market_state=(chart.get("meta") or {}).get("marketState"),
            source=source,
        )
        return {
            "symbol": ticker,
            "range": normalized_range,
            "yahoo_symbol": yahoo_symbol,
            "line_points": points,
            "source": source,
            "as_of": fetched_at,
            "error": None if chart.get("ok") and points else chart.get("error") or "chart_unavailable",
            "meta": chart.get("meta") or {},
            **session_state,
        }

    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key=shared_key,
        factory=build,
        fresh_ttl_seconds=int(config["ttl"]),
        stale_ttl_seconds=max(_ms._MARKET_SWR_STALE_TTL_SECONDS, int(config["ttl"]) * 2),
        local_cache=_ms._MARKET_STOCK_CARD_CHART_CACHE,
        local_key=cache_key,
        force_revalidate=force_refresh,
    )
    if payload is None:
        return {
            "symbol": ticker,
            "range": normalized_range,
            "yahoo_symbol": f"{ticker}.IS",
            "line_points": [],
            "source": "yahoo_cache",
            "as_of": None,
            "error": "chart_refresh_pending",
            "meta": {},
            "cache_status": cache_status,
            "stale": False,
            "refresh_pending": True,
        }
    result = dict(payload)
    if cache_status in {"local_hit", "shared_hit", "coalesced", "stale"}:
        result["source"] = "yahoo_cache"
    result["cache_status"] = cache_status
    result["stale"] = stale
    result["refresh_pending"] = refresh_pending
    return result


def _market_stock_card_chart_payload(*, symbol: str, chart_range: str, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    payload = _ms._fetch_stock_card_chart(symbol, chart_range, force_refresh=force_refresh)
    return {
        "symbol": payload.get("symbol"),
        "range": payload.get("range"),
        "yahoo_symbol": payload.get("yahoo_symbol"),
        "line_points": payload.get("line_points") or [],
        "source": payload.get("source") or "yahoo_live",
        "as_of": payload.get("as_of"),
        "error": payload.get("error"),
        "session_status": payload.get("session_status"),
        "session_label": payload.get("session_label"),
        "is_live": payload.get("is_live"),
        "is_stale": payload.get("is_stale"),
        "last_trade_at": payload.get("last_trade_at"),
        "last_trade_date": payload.get("last_trade_date"),
    }


def _fetch_stock_card_intraday(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    chart_payload = _ms._fetch_stock_card_chart(symbol, "1d", force_refresh=force_refresh)
    ticker = chart_payload.get("symbol") or _ms._normalize_market_stock_card_symbol(symbol)
    yahoo_symbol = chart_payload.get("yahoo_symbol") or f"{ticker}.IS"
    points = chart_payload.get("line_points") or []
    if points:
        highs = [
            point.get("high")
            for point in points
            if isinstance(point.get("high"), (int, float))
        ]
        lows = [
            point.get("low")
            for point in points
            if isinstance(point.get("low"), (int, float))
        ]
        volumes = [
            point.get("volume")
            for point in points
            if isinstance(point.get("volume"), (int, float))
        ]
        meta_payload = chart_payload.get("meta") or {}
        last_close = points[-1].get("close") if points else None
        price = meta_payload.get("regularMarketPrice")
        if price is None and isinstance(last_close, (int, float)):
            price = last_close
        prev_close = meta_payload.get("chartPreviousClose") or meta_payload.get("previousClose")
        change = None
        change_pct = None
        if price is not None and prev_close:
            try:
                change = round(float(price) - float(prev_close), 4)
                change_pct = round((change / float(prev_close)) * 100, 2)
            except (TypeError, ValueError, ZeroDivisionError):
                change = None
                change_pct = None

        last_trade_at = _ms._stock_card_latest_point_dt(points)
        session_state = _ms._stock_card_session_state(
            points,
            market_state=meta_payload.get("marketState"),
            source=chart_payload.get("source"),
        )
        as_of = None
        rmt = meta_payload.get("regularMarketTime")
        if isinstance(rmt, (int, float)):
            try:
                as_of = _ms.datetime.fromtimestamp(float(rmt), tz=_ms.timezone.utc).isoformat()
            except Exception:
                as_of = None
        if session_state.get("is_stale") or not as_of:
            as_of = session_state.get("last_trade_at") or (last_trade_at.isoformat() if last_trade_at else None)

        payload = {
            "line_points": points,
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "high": meta_payload.get("regularMarketDayHigh") or (max(highs) if highs else None),
            "low": meta_payload.get("regularMarketDayLow") or (min(lows) if lows else None),
            "volume": meta_payload.get("regularMarketVolume") or (sum(volumes) if volumes else None),
            "volume_lot": meta_payload.get("regularMarketVolume") or (sum(volumes) if volumes else None),
            "currency": meta_payload.get("currency") or "TRY",
            "market_state": meta_payload.get("marketState") or "",
            "as_of": as_of or chart_payload.get("as_of"),
            "yahoo_symbol": yahoo_symbol,
            "error": None,
            **session_state,
        }
        return dict(payload)

    fallback = {
        "line_points": [],
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "high": None,
        "low": None,
        "volume": None,
        "volume_lot": None,
        "currency": "TRY",
        "market_state": "",
        "as_of": chart_payload.get("as_of"),
        "yahoo_symbol": yahoo_symbol,
        "error": chart_payload.get("error") or "chart_unavailable",
        **_ms._stock_card_session_state([], market_state="", source=chart_payload.get("source")),
    }
    return dict(fallback)


def _stock_cards_response_cache_key(symbols: List[str], tier: str = "full") -> str:
    normalized = ",".join(sorted(symbols))
    return f"api:market:stock-cards:symbols={normalized}:tier={tier}:v3"


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_missing_market_ratio(value: Any) -> bool:
    _ms = _market_service()
    ratio = _ms._parse_tr_decimal(value)
    return ratio is None or abs(ratio) <= 1e-12


def _as_finite_float(value: Any) -> Optional[float]:
    _ms = _market_service()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not _ms.math.isfinite(numeric):
        return None
    return numeric


def _round_market_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    _ms = _market_service()
    top = _ms._as_finite_float(numerator)
    bottom = _ms._as_finite_float(denominator)
    if top is None or bottom is None or abs(bottom) <= 1e-12:
        return None
    return round(top / bottom, 2)


def _resolve_market_card_multiples(symbol: str, multiples_payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    _ms = _market_service()
    fk = _ms._parse_tr_decimal(multiples_payload.get("fk")) if multiples_payload.get("ok") else None
    pd_dd = _ms._parse_tr_decimal(multiples_payload.get("pd_dd")) if multiples_payload.get("ok") else None
    fd_favok = _ms._parse_tr_decimal(multiples_payload.get("fd_favok")) if multiples_payload.get("ok") else None

    need_fallback = (
        _ms._is_missing_market_ratio(fk)
        or _ms._is_missing_market_ratio(pd_dd)
        or _ms._is_missing_market_ratio(fd_favok)
    )
    if not need_fallback:
        return {"fk": fk, "pd_dd": pd_dd, "fd_favok": fd_favok}

    fallback = _ms._fetch_infoyatirim_stock_page_quote(symbol)
    if _ms._is_missing_market_ratio(fk):
        fallback_fk = _ms._parse_tr_decimal(fallback.get("fk"))
        if fallback_fk is not None and abs(fallback_fk) > 1e-12:
            fk = fallback_fk
    if _ms._is_missing_market_ratio(pd_dd):
        fallback_pd_dd = _ms._parse_tr_decimal(fallback.get("pd_dd"))
        if fallback_pd_dd is not None and abs(fallback_pd_dd) > 1e-12:
            pd_dd = fallback_pd_dd
    if _ms._is_missing_market_ratio(fd_favok):
        fallback_fd_favok = _ms._parse_tr_decimal(fallback.get("fd_favok"))
        if fallback_fd_favok is not None and abs(fallback_fd_favok) > 1e-12:
            fd_favok = fallback_fd_favok

    return {"fk": fk, "pd_dd": pd_dd, "fd_favok": fd_favok}


def _stock_card_financial_snapshot_from_cache(symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return {}
    try:
        stored_snapshot = _ms.get_financial_snapshots_map(
            _ms.CONFIG.paths.processed_dir,
            [normalized_symbol],
        ).get(normalized_symbol)
    except Exception as exc:
        _ms.LOGGER.warning("stock-card financial snapshot lookup failed for %s: %s", normalized_symbol, exc)
        stored_snapshot = None
    if stored_snapshot and stored_snapshot.get("has_kap_cache"):
        return {
            "symbol": normalized_symbol,
            "latest_quarter": stored_snapshot.get("latest_quarter"),
            "quarter_count": stored_snapshot.get("quarter_count"),
            "ttm_net_kar": stored_snapshot.get("ttm_net_kar"),
            "ttm_satis_gelirleri": stored_snapshot.get("ttm_satis_gelirleri"),
            "ttm_favok": stored_snapshot.get("ttm_favok"),
            "company_kind": stored_snapshot.get("company_kind") or "generic",
            "ozkaynaklar": stored_snapshot.get("ozkaynaklar"),
            "net_borc": stored_snapshot.get("net_borc"),
            "source": "financial_snapshot_sql",
            "as_of": stored_snapshot.get("as_of"),
        }
    cache_path = _ms.CONFIG.paths.processed_dir / "kap_cache" / f"{normalized_symbol}.json"
    if not cache_path.exists():
        return {}
    cache_key = ""
    signature = None
    shared_key = None
    try:
        stat = cache_path.stat()
        cache_key = str(cache_path.resolve())
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = _ms._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE.get(cache_key)
        if cached and cached.get("signature") == signature:
            return dict(cached.get("data") or {})
        shared_key = f"api:kap:financial-snapshot:{normalized_symbol}:mtime={stat.st_mtime_ns}:size={stat.st_size}:v2"
        shared_cached = _ms._shared_cache_get_dict(shared_key)
        if shared_cached is not None:
            snapshot = dict(shared_cached)
            _ms._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE[cache_key] = {
                "signature": signature,
                "data": snapshot,
            }
            return snapshot
    except Exception:
        cache_key = ""
        signature = None
        shared_key = None

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = _ms.json.load(handle)
    except Exception:
        return {}

    quarters_raw = payload.get("quarters")
    quarters = [q for q in quarters_raw if isinstance(q, dict)] if isinstance(quarters_raw, list) else []
    if not quarters:
        if cache_key and signature:
            _ms._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE[cache_key] = {
                "signature": signature,
                "data": {},
            }
            if shared_key:
                _ms._shared_cache_set(shared_key, {}, ttl_seconds=_ms._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE_TTL)
        return {}

    from app.kap_service import normalize_snapshot_for_frontend

    normalized = normalize_snapshot_for_frontend(payload, allow_network=False)
    normalized_quarters = [
        quarter for quarter in normalized.get("quarters", []) if isinstance(quarter, dict)
    ]
    quarters_sorted = sorted(normalized_quarters, key=_ms._quarter_sort_key)
    latest = quarters_sorted[-1]
    ttm_net_kar = _ms._build_ttm_from_reports(quarters_sorted, "net_kar")
    ttm_favok = _ms._build_ttm_from_reports(quarters_sorted, "favok")
    ozkaynaklar = _ms._extract_quarter_metric(latest, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
    net_borc = _ms._extract_quarter_metric(latest, "net_borc", priority=["metrics", "metrics_ytd"])
    latest_quarter = str(latest.get("quarter") or "").strip().upper() or None
    from src.kap_fetcher import classify_kap_company_kind
    snapshot = {
        "symbol": str(payload.get("company") or payload.get("stock_code") or normalized_symbol or "").strip().upper(),
        "latest_quarter": latest_quarter,
        "quarter_count": len(quarters_sorted),
        "ttm_net_kar": ttm_net_kar,
        "ttm_favok": ttm_favok,
        "company_kind": payload.get("company_kind") or classify_kap_company_kind(normalized_symbol, payload.get("company_title")),
        "ozkaynaklar": ozkaynaklar,
        "net_borc": net_borc,
        "source": "kap_cache",
        "as_of": payload.get("fetched_at"),
    }
    if cache_key and signature:
        _ms._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE[cache_key] = {
            "signature": signature,
            "data": snapshot,
        }
        if shared_key:
            _ms._shared_cache_set(shared_key, snapshot, ttl_seconds=_ms._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE_TTL)
    return snapshot


def _stock_card_financial_ratios_from_snapshot(snapshot: Dict[str, Any], *, market_cap: Any) -> Dict[str, Any]:
    _ms = _market_service()
    financial_company = snapshot.get("company_kind") in {"bank", "insurance"}
    market_cap_value = _ms._positive_float(market_cap)
    ttm_net_kar = _ms._as_finite_float(snapshot.get("ttm_net_kar"))
    ttm_favok = _ms._as_finite_float(snapshot.get("ttm_favok"))
    ozkaynaklar = _ms._as_finite_float(snapshot.get("ozkaynaklar"))
    net_borc = _ms._as_finite_float(snapshot.get("net_borc"))
    enterprise_value = (
        market_cap_value + net_borc
        if market_cap_value is not None and net_borc is not None
        else None
    )

    return {
        "fk": _ms._round_market_ratio(market_cap_value, ttm_net_kar),
        "pd_dd": _ms._round_market_ratio(market_cap_value, ozkaynaklar),
        "fd_favok": None if financial_company else _ms._round_market_ratio(enterprise_value, ttm_favok),
        "net_borc_favok": None if financial_company else _ms._round_market_ratio(net_borc, ttm_favok),
        "enterprise_value": None if financial_company else enterprise_value,
    }


def _stock_card_financial_ratios_from_cached_snapshot(
    snapshot: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    _ms = _market_service()
    ratios = _ms._stock_card_financial_ratios_from_snapshot(snapshot, market_cap=None)
    net_borc_favok = ratios.get("net_borc_favok")
    if net_borc_favok is None and snapshot:
        try:
            net_borc = _ms._as_finite_float(snapshot.get("net_borc"))
            ttm_favok = _ms._as_finite_float(snapshot.get("ttm_favok"))
            if net_borc is not None and ttm_favok is not None and abs(ttm_favok) > 1e-12:
                net_borc_favok = round(net_borc / ttm_favok, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            net_borc_favok = None
    return {"net_borc_favok": net_borc_favok}


def _stock_card_financial_ratios_from_cache(symbol: str) -> Dict[str, Optional[float]]:
    _ms = _market_service()
    snapshot = _ms._stock_card_financial_snapshot_from_cache(symbol)
    return _stock_card_financial_ratios_from_cached_snapshot(snapshot)


def _resolve_market_card_valuation_from_cached_data(
    cached: Dict[str, Any],
    *,
    market_cap: Any,
) -> Dict[str, Any]:
    _ms = _market_service()
    snapshot = cached.get("financial_snapshot") if isinstance(cached.get("financial_snapshot"), dict) else {}
    computed = _ms._stock_card_financial_ratios_from_snapshot(snapshot, market_cap=market_cap)
    provider = cached.get("provider_ratios") if isinstance(cached.get("provider_ratios"), dict) else {}

    values: Dict[str, Optional[float]] = {}
    sources: Dict[str, str] = {}
    for key in ("fk", "pd_dd", "fd_favok"):
        computed_value = computed.get(key)
        if not _ms._is_missing_market_ratio(computed_value):
            values[key] = computed_value
            sources[key] = "kap_computed"
            continue
        provider_value = provider.get(key)
        values[key] = provider_value
        if not _ms._is_missing_market_ratio(provider_value):
            sources[key] = str(cached.get("provider_source") or "external_ratio_provider")

    net_borc_favok = computed.get("net_borc_favok")
    if _ms._is_missing_market_ratio(net_borc_favok):
        fallback_ratios = cached.get("fallback_financial_ratios")
        if isinstance(fallback_ratios, dict):
            net_borc_favok = fallback_ratios.get("net_borc_favok")

    if snapshot.get("company_kind") in {"bank", "insurance"}:
        values["fd_favok"] = None
        sources.pop("fd_favok", None)
        net_borc_favok = None

    return {
        **values,
        "net_borc_favok": net_borc_favok,
        "valuation_source": "kap_computed" if sources and set(sources.values()) == {"kap_computed"} else "mixed",
        "valuation_sources": sources,
        "valuation_as_of": cached.get("as_of"),
        "valuation_financial_period": snapshot.get("latest_quarter"),
        "enterprise_value": computed.get("enterprise_value"),
        "cache_hit": bool(cached.get("_cache_hit")),
    }


def _resolve_market_card_valuation(symbol: str, *, market_cap: Any) -> Dict[str, Any]:
    _ms = _market_service()
    ticker = str(symbol or "").strip().upper()
    now = _ms.time.time()
    cached = _ms._STOCK_CARD_VALUATION_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < _ms._STOCK_CARD_VALUATION_CACHE_TTL:
        return _ms._resolve_market_card_valuation_from_cached_data(
            {**cached, "_cache_hit": True},
            market_cap=market_cap,
        )
    shared_key = f"api:stock-card-valuation:{ticker}:v3"
    shared_cached = _ms._shared_cache_get_dict(shared_key)
    if shared_cached:
        _ms._STOCK_CARD_VALUATION_CACHE[ticker] = {**shared_cached, "_ts": now}
        return _ms._resolve_market_card_valuation_from_cached_data(
            {**shared_cached, "_cache_hit": True},
            market_cap=market_cap,
        )

    snapshot = _ms._stock_card_financial_snapshot_from_cache(ticker)
    from src.kap_fetcher import classify_kap_company_kind
    snapshot.setdefault("company_kind", classify_kap_company_kind(ticker))
    computed = _ms._stock_card_financial_ratios_from_snapshot(snapshot, market_cap=market_cap)
    ratio_keys = ("fk", "pd_dd") if snapshot.get("company_kind") in {"bank", "insurance"} else ("fk", "pd_dd", "fd_favok")
    needs_provider = any(_ms._is_missing_market_ratio(computed.get(key)) for key in ratio_keys)

    provider_payload: Dict[str, Any] = {}
    provider_ratios: Dict[str, Optional[float]] = {}
    if needs_provider:
        provider_payload = _ms._fetch_isyatirim_multiples(ticker)
        provider_ratios = _ms._resolve_market_card_multiples(ticker, provider_payload)

    if _ms._stock_card_financial_ratios_from_cache is not _stock_card_financial_ratios_from_cache:
        cache_ratios = _ms._stock_card_financial_ratios_from_cache(ticker)
    else:
        cache_ratios = _stock_card_financial_ratios_from_cached_snapshot(snapshot)
    cached_payload = {
        "_ts": now,
        "financial_snapshot": snapshot,
        "provider_ratios": provider_ratios,
        "provider_source": provider_payload.get("source") if provider_payload.get("ok") else None,
        "provider_as_of": provider_payload.get("fetched_at"),
        "fallback_financial_ratios": cache_ratios,
        "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
    }
    _ms._STOCK_CARD_VALUATION_CACHE[ticker] = cached_payload
    _ms._shared_cache_set(shared_key, cached_payload, _ms._STOCK_CARD_VALUATION_CACHE_TTL)
    return _ms._resolve_market_card_valuation_from_cached_data(cached_payload, market_cap=market_cap)


def _market_stock_cards_quick_payload(
    normalized_symbols: List[str],
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Return the quote-first portion of stock cards without slow enrichments."""
    _ms = _market_service()
    response_cache_key = _ms._stock_cards_response_cache_key(normalized_symbols, tier="quick")
    if not force_refresh:
        shared_cached = _ms._shared_cache_get_dict(response_cache_key)
        if shared_cached is not None:
            return dict(shared_cached)

    price_map = _ms._fetch_market_price_map(normalized_symbols)
    quote_as_of = next(
        (
            quote.get("as_of")
            for quote in price_map.values()
            if isinstance(quote, dict) and quote.get("as_of")
        ),
        None,
    )
    instrument_map = _ms.get_instruments(_ms.CONFIG.paths.processed_dir, "stock", normalized_symbols)
    items: List[Dict[str, Any]] = []
    for symbol in normalized_symbols:
        quote = price_map.get(symbol, {})
        price = quote.get("price")
        volume = quote.get("volume")
        items.append({
            "symbol": symbol,
            "company": str((instrument_map.get(symbol) or {}).get("name") or symbol).strip() or symbol,
            "yahoo_symbol": f"{symbol}.IS",
            "price": price,
            "currency": quote.get("currency") or "TRY",
            "change": quote.get("change"),
            "change_pct": quote.get("change_pct"),
            "volume": volume,
            "volume_lot": quote.get("volume_lot"),
            "volume_tl": volume,
            "market_cap": quote.get("market_cap"),
            "high": None,
            "low": None,
            "previous_close": None,
            "fk": None,
            "pd_dd": None,
            "fd_favok": None,
            "net_borc_favok": None,
            "market_state": quote.get("market_state") or "",
            "as_of": quote.get("as_of"),
            "session_status": "unknown",
            "session_label": "Detaylar hazırlanıyor",
            "is_live": False,
            "is_stale": False,
            "last_trade_at": None,
            "last_trade_date": None,
            "line_points": [],
            "error": None if price is not None else "quote_unavailable",
            "logo_url": (instrument_map.get(symbol) or {}).get("logo_url"),
            "logo_source": (instrument_map.get(symbol) or {}).get("logo_source"),
            "card_stage": "quick",
            "return_1w_pct": None,
            "return_1m_pct": None,
            "return_3m_pct": None,
            "return_6m_pct": None,
            "return_ytd_pct": None,
            "return_1y_pct": None,
        })

    data = {
        "items": items,
        "source": "infoyatirim_live_quote",
        # The quote timestamp is the shared live snapshot timestamp used by
        # both the market list and the stock cards.
        "as_of": quote_as_of or _ms.datetime.now(_ms.timezone.utc).isoformat(),
    }
    _ms._shared_cache_set(
        response_cache_key,
        data,
        ttl_seconds=_ms._MARKET_STOCK_CARDS_QUICK_RESPONSE_CACHE_TTL,
    )
    return data


def _market_stock_cards_payload(
    *,
    symbols: str,
    force_refresh: bool = False,
    tier: str = "full",
) -> Dict[str, Any]:
    _ms = _market_service()
    normalized_tier = _ms._normalize_market_stock_card_tier(tier)
    normalized_symbols = _ms._normalize_market_stock_card_symbols(symbols)
    if not normalized_symbols:
        return {
            "items": [],
            "source": "infoyatirim_live_quote" if normalized_tier == "quick" else "infoyatirim_yahoo_chart",
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    if normalized_tier == "quick":
        return _ms._market_stock_cards_quick_payload(normalized_symbols, force_refresh=force_refresh)

    response_cache_key = _ms._stock_cards_response_cache_key(normalized_symbols, tier="full")
    if not force_refresh:
        shared_cached = _ms._shared_cache_get_dict(response_cache_key)
        if shared_cached is not None:
            return dict(shared_cached)

    cache_dir = _ms.CONFIG.paths.processed_dir / "kap_cache"
    price_map = _ms._fetch_market_price_map(normalized_symbols)
    quote_as_of = next(
        (
            quote.get("as_of")
            for quote in price_map.values()
            if isinstance(quote, dict) and quote.get("as_of")
        ),
        None,
    )
    basic_summary_map = _ms._fetch_isyatirim_basic_summary_map()
    return_base_map = _ms._fetch_stock_return_bases_bulk(normalized_symbols)
    instrument_map = _ms.get_instruments(_ms.CONFIG.paths.processed_dir, "stock", normalized_symbols)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers=min(_ms._MARKET_STOCK_CARD_CHART_MAX_WORKERS, len(normalized_symbols))
    ) as pool:
        intraday_results = list(
            pool.map(
                lambda symbol: _ms._fetch_stock_card_intraday(symbol, force_refresh=force_refresh),
                normalized_symbols,
            )
        )
    intraday_by_symbol = dict(zip(normalized_symbols, intraday_results))

    items: List[Dict[str, Any]] = []
    for symbol in normalized_symbols:
        quote = price_map.get(symbol, {})
        intraday = intraday_by_symbol[symbol]
        cached_meta = _ms._load_cached_kap_market_metadata(cache_dir, symbol)
        instrument = instrument_map.get(symbol)
        company_name = str((instrument or {}).get("name") or cached_meta.get("company_title") or "").strip() or symbol
        basic_summary = basic_summary_map.get(symbol)
        market_cap = _ms._market_cap_from_quote_and_meta(quote, cached_meta, basic_summary)
        valuation = _ms._resolve_market_card_valuation(symbol, market_cap=market_cap)
        return_bases = return_base_map.get(symbol, {})

        price = _ms._first_not_none(quote.get("price"), intraday.get("price"))
        currency = quote.get("currency") or intraday.get("currency") or "TRY"
        volume_tl = quote.get("volume")
        volume_lot = _ms._first_not_none(intraday.get("volume_lot"), intraday.get("volume"), quote.get("volume_lot"))
        current_for_returns = price if price is not None else return_bases.get("latest_close")
        session_status = intraday.get("session_status") or "unknown"
        # Price, change and volume all come from the canonical live quote;
        # keep its timestamp even when Yahoo supplies the chart/session data.
        as_of = quote.get("as_of") or intraday.get("as_of")
        item = {
            "symbol": symbol,
            "card_stage": "full",
            "company": company_name,
            "yahoo_symbol": intraday.get("yahoo_symbol"),
            "price": price,
            "currency": currency,
            "change": _ms._first_not_none(quote.get("change"), intraday.get("change")),
            "change_pct": _ms._first_not_none(quote.get("change_pct"), intraday.get("change_pct")),
            "volume": _ms._first_not_none(volume_tl, volume_lot),
            "volume_lot": volume_lot,
            "volume_tl": volume_tl,
            "market_cap": market_cap,
            "high": intraday.get("high"),
            "low": intraday.get("low"),
            "previous_close": intraday.get("prev_close"),
            "fk": valuation.get("fk"),
            "pd_dd": valuation.get("pd_dd"),
            "fd_favok": valuation.get("fd_favok"),
            "net_borc_favok": valuation.get("net_borc_favok"),
            "valuation_source": valuation.get("valuation_source"),
            "valuation_sources": valuation.get("valuation_sources"),
            "valuation_as_of": valuation.get("valuation_as_of"),
            "valuation_financial_period": valuation.get("valuation_financial_period"),
            "valuation_cache_hit": valuation.get("cache_hit"),
            "market_state": quote.get("market_state") or intraday.get("market_state") or "",
            "as_of": as_of,
            "session_status": session_status,
            "session_label": intraday.get("session_label"),
            "is_live": intraday.get("is_live"),
            "is_stale": intraday.get("is_stale"),
            "last_trade_at": intraday.get("last_trade_at"),
            "last_trade_date": intraday.get("last_trade_date"),
            "line_points": intraday.get("line_points") or [],
            "error": None if price is not None or intraday.get("line_points") else intraday.get("error"),
            "logo_url": (instrument or {}).get("logo_url"),
            "logo_source": (instrument or {}).get("logo_source"),
            **_ms._returns_from_bases(current_for_returns, return_bases),
        }
        items.append(item)

    data = {
        "items": items,
        "source": "infoyatirim_yahoo_chart",
        "as_of": quote_as_of or _ms.datetime.now(_ms.timezone.utc).isoformat(),
    }
    _ms._shared_cache_set(
        response_cache_key,
        data,
        ttl_seconds=_ms._MARKET_STOCK_CARDS_RESPONSE_CACHE_TTL,
    )
    return data


def _normalize_market_index(index_code: str) -> str:
    _ms = _market_service()
    normalized = str(index_code or "").strip().upper()
    if normalized not in _ms._MARKET_INDEX_META:
        raise _ms.HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen endeks. {_ms._supported_market_indexes_text()} kullanin.",
        )
    return normalized


def _index_return_bases_from_points(
    points: List[tuple[datetime, float]],
    *,
    history_source: str,
    provider_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    _ms = _market_service()
    cleaned: List[tuple[datetime, float]] = []
    for point_dt, close in points:
        if not isinstance(point_dt, _ms.datetime):
            continue
        if point_dt.tzinfo is None:
            point_dt = point_dt.replace(tzinfo=_ms.timezone.utc)
        cleaned.append((point_dt.astimezone(_ms.timezone.utc), float(close)))
    if not cleaned:
        return {}

    cleaned.sort(key=lambda item: item[0])
    latest_dt, latest_close = cleaned[-1]
    year_start = _ms.datetime(latest_dt.year, 1, 1, tzinfo=_ms.timezone.utc)
    bases = {
        "base_1w": _ms._pick_series_value_at_or_before(cleaned, latest_dt - _ms.timedelta(days=7)),
        "base_1m": _ms._pick_series_value_at_or_before(cleaned, latest_dt - _ms.timedelta(days=30)),
        "base_3m": _ms._pick_series_value_at_or_before(cleaned, latest_dt - _ms.timedelta(days=91)),
        "base_6m": _ms._pick_series_value_at_or_before(cleaned, latest_dt - _ms.timedelta(days=182)),
        "base_ytd": _ms._pick_series_value_at_or_after(cleaned, year_start),
        "base_1y": _ms._pick_series_value_at_or_before(cleaned, latest_dt - _ms.timedelta(days=365)),
        "base_5y": _ms._pick_series_value_at_or_before(cleaned, latest_dt - _ms.timedelta(days=365 * 5)),
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
        "history_source": history_source,
    }
    if provider_symbol:
        bases["yahoo_symbol"] = provider_symbol
    return bases


def _index_return_bases_have_period_history(bases: Dict[str, Any]) -> bool:
    return any(
        bases.get(key) is not None
        for key in ("base_1w", "base_1m", "base_3m", "base_6m", "base_1y")
    )


def _fetch_index_quote(index_code: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized = _ms._normalize_market_index(index_code)
    now = _ms.time.time()
    cached = _ms._MARKET_INDEX_QUOTE_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _ms._effective_bist_index_fresh_ttl():
        return dict(cached.get("data") or {})

    meta = _ms._MARKET_INDEX_META[normalized]
    errors: List[str] = []
    for yahoo_symbol in meta["yahoo_candidates"]:
        quote = _ms._fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            row = {
                "symbol": normalized,
                "label": meta["label"],
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price"),
                "prev_close": quote.get("prev_close"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "volume": quote.get("volume"),
                "currency": quote.get("currency") or "TRY",
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
            }
            _ms._MARKET_INDEX_QUOTE_CACHE[normalized] = {"_ts": now, "data": row}
            return dict(row)
        errors.append(str(quote.get("error") or "quote_unavailable"))

    fallback = {
        "symbol": normalized,
        "label": meta["label"],
        "yahoo_symbol": None,
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "high": None,
        "low": None,
        "volume": None,
        "currency": "TRY",
        "market_state": "",
        "as_of": None,
        "error": "; ".join(errors[:3]) if errors else "quote_unavailable",
    }
    _ms._MARKET_INDEX_QUOTE_CACHE[normalized] = {"_ts": now, "data": fallback}
    return dict(fallback)


def _fetch_index_return_bases(index_code: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized = _ms._normalize_market_index(index_code)
    now = _ms.time.time()
    cached = _ms._MARKET_INDEX_RETURN_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _ms._MARKET_INDEX_RETURN_CACHE_TTL:
        return dict(cached.get("data") or {})
    _ms._warm_bulk_return_bases(
        _ms._MARKET_INDEX_RETURN_CACHE,
        _ms._INDEX_RETURN_BASES_BULK_KEY,
        _ms._MARKET_INDEX_RETURN_CACHE_TTL,
    )
    cached_after_warm = _ms._MARKET_INDEX_RETURN_CACHE.get(normalized)
    if cached_after_warm and now - cached_after_warm.get("_ts", 0) < _ms._MARKET_INDEX_RETURN_CACHE_TTL:
        return dict(cached_after_warm.get("data") or {})

    meta = _ms._MARKET_INDEX_META[normalized]
    yahoo_partial: Optional[Dict[str, Any]] = None
    for yahoo_symbol in meta["yahoo_candidates"]:
        chart = _ms._fetch_yahoo_chart_raw(yahoo_symbol, interval="1d", range_="5y")
        if not chart.get("ok"):
            continue
        points = [
            (_ms.datetime.fromisoformat(str(point["time"])), float(point["close"]))
            for point in chart.get("points", [])
            if point.get("time") and isinstance(point.get("close"), (int, float))
        ]
        if not points:
            continue
        bases = _ms._index_return_bases_from_points(points, history_source="yahoo", provider_symbol=yahoo_symbol)
        if _ms._index_return_bases_have_period_history(bases):
            return _ms._store_bulk_return_base_entry(
                _ms._MARKET_INDEX_RETURN_CACHE,
                _ms._INDEX_RETURN_BASES_BULK_KEY,
                normalized,
                bases,
                _ms._MARKET_INDEX_RETURN_CACHE_TTL,
                now,
            )
        if yahoo_partial is None:
            yahoo_partial = bases

    today = _ms.datetime.now(_ms._TURKEY_TIMEZONE).date()
    history_start = today - _ms.timedelta(days=(365 * 5) + 14)
    history_points = _ms._fetch_isyatirim_index_history(normalized, start_date=history_start, end_date=today + _ms.timedelta(days=1))
    bases = _ms._index_return_bases_from_points(history_points, history_source="isyatirim")
    if _ms._index_return_bases_have_period_history(bases):
        return _ms._store_bulk_return_base_entry(
            _ms._MARKET_INDEX_RETURN_CACHE,
            _ms._INDEX_RETURN_BASES_BULK_KEY,
            normalized,
            bases,
            _ms._MARKET_INDEX_RETURN_CACHE_TTL,
            now,
        )

    if yahoo_partial and _ms._index_return_bases_have_period_history(yahoo_partial):
        return _ms._store_bulk_return_base_entry(
            _ms._MARKET_INDEX_RETURN_CACHE,
            _ms._INDEX_RETURN_BASES_BULK_KEY,
            normalized,
            yahoo_partial,
            _ms._MARKET_INDEX_RETURN_CACHE_TTL,
            now,
        )

    return _ms._store_bulk_return_base_entry(
        _ms._MARKET_INDEX_RETURN_CACHE,
        _ms._INDEX_RETURN_BASES_BULK_KEY,
        normalized,
        {},
        _ms._MARKET_INDEX_RETURN_CACHE_TTL,
        now,
    )


def _index_returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Optional[float]]:
    _ms = _market_service()
    return {
        response_field: _ms._return_pct(current_price, return_bases.get(base_field))
        for response_field, base_field in _ms._INDEX_RETURN_BASE_FIELDS
    }


def _market_index_row(index_code: str, *, quote: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _ms = _market_service()
    normalized = _ms._normalize_market_index(index_code)
    quote_row = dict(quote or _ms._fetch_index_quote(normalized))
    bases = _ms._fetch_index_return_bases(normalized)
    current_for_returns = quote_row.get("price") if quote_row.get("price") is not None else bases.get("latest_close")
    return {
        "symbol": normalized,
        "label": _ms._MARKET_INDEX_META[normalized]["label"],
        "yahoo_symbol": quote_row.get("yahoo_symbol") or bases.get("yahoo_symbol"),
        "price": quote_row.get("price"),
        "prev_close": quote_row.get("prev_close"),
        "change": quote_row.get("change"),
        "change_pct": quote_row.get("change_pct"),
        "high": quote_row.get("high"),
        "low": quote_row.get("low"),
        "volume": quote_row.get("volume"),
        "currency": quote_row.get("currency") or "TRY",
        "market_state": quote_row.get("market_state") or "",
        "as_of": quote_row.get("as_of") or bases.get("as_of"),
        "error": quote_row.get("error"),
        **_ms._index_returns_from_bases(current_for_returns, bases),
    }


def _comparison_error_message(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc) or exc.__class__.__name__


def _comparison_history_result(
    asset: MarketComparisonHistoryAsset,
    *,
    symbol: str,
    label: Optional[str],
    points: List[Dict[str, Any]],
    source: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": asset.id or f"{asset.kind}:{symbol}",
        "kind": asset.kind,
        "symbol": symbol,
        "label": label or asset.label or symbol,
        "points": points,
        "source": source,
        "error": error,
    }


def _comparison_history_points_from_chart(
    chart: Dict[str, Any],
    *,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    _ms = _market_service()
    deduped: Dict[str, Dict[str, Any]] = {}
    for point in list(chart.get("points") or []):
        if not isinstance(point, dict):
            continue
        point_dt = _ms._point_datetime(point.get("time"))
        close = _ms._numeric_chart_value(point.get("close"))
        if point_dt is None or close is None or close <= 0:
            continue
        point_date = point_dt.date()
        if point_date < start_date or point_date > end_date:
            continue
        date_key = point_date.isoformat()
        deduped[date_key] = {"date": date_key, "value": close}
    return [deduped[key] for key in sorted(deduped)]


def _fund_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    _ms = _market_service()
    from app.fund_service import get_fund_performance_payload, normalize_fund_code

    symbol = normalize_fund_code(asset.symbol)
    try:
        payload = get_fund_performance_payload(
            _ms.CONFIG.paths.processed_dir,
            symbol,
            start_date=start_date,
            end_date=end_date,
        )
        points: List[Dict[str, Any]] = []
        for point in list(payload.get("points") or []):
            try:
                price = float(point.get("price"))
            except (TypeError, ValueError):
                continue
            point_date = str(point.get("date") or "")
            if not point_date or price <= 0:
                continue
            if point_date < start_date.isoformat() or point_date > end_date.isoformat():
                continue
            points.append({"date": point_date, "value": price})
        return _ms._comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=points,
            source=str(payload.get("source") or "sqlite"),
            error=None if points else str(payload.get("source_metadata", {}).get("warning") or payload.get("status") or "data_unavailable"),
        )
    except Exception as exc:
        return _ms._comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="sqlite",
            error=_ms._comparison_error_message(exc),
        )


def _stock_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    _ms = _market_service()
    try:
        symbol = _ms._normalize_market_stock_card_symbol(asset.symbol)
        yahoo_symbol = f"{symbol}.IS"
        chart = _ms._fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date,
        )
        points = _ms._comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
        return _ms._comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=points,
            source="yahoo_finance_chart",
            error=None if chart.get("ok") and points else str(chart.get("error") or "data_unavailable"),
        )
    except Exception as exc:
        symbol = str(asset.symbol or "").strip().upper()
        return _ms._comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error=_ms._comparison_error_message(exc),
        )


def _index_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    _ms = _market_service()
    try:
        symbol = _ms._normalize_market_index(asset.symbol)
        label = asset.label or _ms._MARKET_INDEX_META[symbol]["label"]
        errors: List[str] = []
        for yahoo_symbol in _ms._MARKET_INDEX_META[symbol]["yahoo_candidates"]:
            chart = _ms._fetch_yahoo_chart_period_raw(
                yahoo_symbol,
                interval="1d",
                start_date=start_date,
                end_date=end_date,
            )
            points = _ms._comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
            if chart.get("ok") and points:
                return _ms._comparison_history_result(
                    asset,
                    symbol=symbol,
                    label=label,
                    points=points,
                    source="yahoo_finance_chart",
                )
            errors.append(str(chart.get("error") or "data_unavailable"))
        return _ms._comparison_history_result(
            asset,
            symbol=symbol,
            label=label,
            points=[],
            source="yahoo_finance_chart",
            error="; ".join(errors[:3]) if errors else "data_unavailable",
        )
    except Exception as exc:
        symbol = str(asset.symbol or "").strip().upper()
        return _ms._comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error=_ms._comparison_error_message(exc),
        )


def _fx_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    _ms = _market_service()
    symbol = str(asset.symbol or "").strip().upper()
    direct_map = {entry[0]: entry for entry in _ms._FX_DIRECT_MAP}
    entry = direct_map.get(symbol)
    if not entry:
        return _ms._comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error="unsupported_fx_symbol",
        )

    _, yahoo_candidates, default_label = entry
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        chart = _ms._fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date,
        )
        points = _ms._comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
        if chart.get("ok") and points:
            return _ms._comparison_history_result(
                asset,
                symbol=symbol,
                label=asset.label or default_label,
                points=points,
                source="yahoo_finance_chart",
            )
        errors.append(str(chart.get("error") or "data_unavailable"))

    return _ms._comparison_history_result(
        asset,
        symbol=symbol,
        label=asset.label or default_label,
        points=[],
        source="yahoo_finance_chart",
        error="; ".join(errors[:3]) if errors else "data_unavailable",
    )


def _comparison_history_cache_key(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> str:
    """Build a bounded key that intentionally excludes presentation fields."""

    _ms = _market_service()
    symbol = _ms.re.sub(r"[^A-Z0-9/_-]", "", str(asset.symbol or "").strip().upper())[:32]
    return (
        "api:market:comparison-history:v2:"
        f"kind={asset.kind}:symbol={symbol}:from={start_date.isoformat()}:to={end_date.isoformat()}"
    )


def _comparison_history_cached_asset(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
    handler: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    """Cache a single normalized series, not the user-specific POST body."""

    _ms = _market_service()
    latest_result: Optional[Dict[str, Any]] = None

    def build() -> Optional[Dict[str, Any]]:
        nonlocal latest_result
        latest_result = handler(asset, start_date=start_date, end_date=end_date)
        # A first-request provider failure must not become a multi-day negative
        # cache.  Existing stale success data is still served by the wrapper.
        return latest_result if not latest_result.get("error") else None

    historical_range = end_date < _ms.date.today()
    fresh_ttl = 24 * 60 * 60 if historical_range else 60
    stale_ttl = 7 * 24 * 60 * 60 if historical_range else 10 * 60
    cached, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key=_ms._comparison_history_cache_key(asset, start_date=start_date, end_date=end_date),
        factory=build,
        fresh_ttl_seconds=fresh_ttl,
        stale_ttl_seconds=stale_ttl,
    )
    if cached is not None:
        result = dict(cached)
    elif latest_result is not None:
        result = dict(latest_result)
    else:
        # A waiter must not repeat the provider request after the bounded
        # single-flight wait.  The client can retain its prior chart and retry.
        result = _ms._comparison_history_result(
            asset,
            symbol=str(asset.symbol or "").strip().upper(),
            label=asset.label,
            points=[],
            source="mixed",
            error="refresh_pending",
        )
    # ``id`` and ``label`` are caller presentation choices, so they must not
    # fragment the expensive provider-series cache.
    symbol = str(result.get("symbol") or asset.symbol).strip().upper()
    result["id"] = asset.id or f"{asset.kind}:{symbol}"
    result["label"] = asset.label or result.get("label") or symbol
    result["cache_status"] = cache_status
    result["stale"] = stale
    result["refresh_pending"] = refresh_pending
    return result


def _market_comparison_history_payload(request: MarketComparisonHistoryRequest) -> Dict[str, Any]:
    _ms = _market_service()
    handlers = {
        "fund": _ms._fund_comparison_history,
        "stock": _ms._stock_comparison_history,
        "index": _ms._index_comparison_history,
        "fx": _ms._fx_comparison_history,
    }
    assets_list = list(request.assets)
    if len(assets_list) > 1:
        with ThreadPoolExecutor(
            max_workers=min(6, len(assets_list)),
            thread_name_prefix="market-compare",
        ) as executor:
            futures = [
                executor.submit(
                    _ms._comparison_history_cached_asset,
                    asset,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    handler=handlers[asset.kind],
                )
                for asset in assets_list
            ]
            resolved_assets = [future.result() for future in futures]
    else:
        resolved_assets = [
            _ms._comparison_history_cached_asset(
                asset,
                start_date=request.start_date,
                end_date=request.end_date,
                handler=handlers[asset.kind],
            )
            for asset in assets_list
        ]
    return {
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "assets": resolved_assets,
        "source": "mixed",
        "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
    }


def _market_indices_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        _ms._begin_bulk_return_batch(
            _ms._MARKET_INDEX_RETURN_CACHE,
            _ms._INDEX_RETURN_BASES_BULK_KEY,
            _ms._MARKET_INDEX_RETURN_CACHE_TTL,
        )
        try:
            try:
                with ThreadPoolExecutor(max_workers=min(8, len(_ms._MARKET_INDEX_ORDER))) as pool:
                    rows = list(pool.map(_ms._market_index_row, _ms._MARKET_INDEX_ORDER))
            except Exception:
                rows = [_ms._market_index_row(index_code) for index_code in _ms._MARKET_INDEX_ORDER]
        finally:
            _ms._end_bulk_return_batch(
                _ms._MARKET_INDEX_RETURN_CACHE,
                _ms._INDEX_RETURN_BASES_BULK_KEY,
                _ms._MARKET_INDEX_RETURN_CACHE_TTL,
            )
        return {
            "rows": rows,
            "source": "yahoo_finance_chart",
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key="api:market:indices:v2",
        factory=build,
        fresh_ttl_seconds=_ms._effective_bist_index_fresh_ttl(),
        stale_ttl_seconds=_ms._MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_ms._MARKET_INDICES_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {
            "rows": [],
            "source": "yahoo_finance_chart",
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
            "error": "market_indices_refresh_pending",
        }
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _index_intraday_payload(index_code: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized = _ms._normalize_market_index(index_code)
    now = _ms.time.time()
    cached = _ms._MARKET_INDEX_INTRADAY_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _ms._MARKET_INDEX_INTRADAY_CACHE_TTL:
        return dict(cached.get("data") or {})

    meta = _ms._MARKET_INDEX_META[normalized]
    for yahoo_symbol in meta["yahoo_candidates"]:
        chart = _ms._fetch_yahoo_chart_raw(yahoo_symbol, interval="5m", range_="1d")
        if chart.get("ok") and chart.get("points"):
            points = [
                {
                    "time": point["time"],
                    "open": point.get("open"),
                    "high": point.get("high"),
                    "low": point.get("low"),
                    "close": point["close"],
                }
                for point in chart.get("points", [])
                if point.get("time") and isinstance(point.get("close"), (int, float))
            ]
            highs = [
                point.get("high")
                for point in chart.get("points", [])
                if isinstance(point.get("high"), (int, float))
            ]
            lows = [
                point.get("low")
                for point in chart.get("points", [])
                if isinstance(point.get("low"), (int, float))
            ]
            meta_payload = chart.get("meta") or {}
            payload = {
                "line_points": points,
                "high": meta_payload.get("regularMarketDayHigh") or (max(highs) if highs else None),
                "low": meta_payload.get("regularMarketDayLow") or (min(lows) if lows else None),
                "prev_close": meta_payload.get("chartPreviousClose") or meta_payload.get("previousClose"),
                "yahoo_symbol": yahoo_symbol,
            }
            _ms._MARKET_INDEX_INTRADAY_CACHE[normalized] = {"_ts": now, "data": payload}
            return dict(payload)

    fallback = {"line_points": [], "high": None, "low": None, "prev_close": None, "yahoo_symbol": None}
    _ms._MARKET_INDEX_INTRADAY_CACHE[normalized] = {"_ts": now, "data": fallback}
    return dict(fallback)


def _latest_share_count_from_kap_cache(symbol: str) -> Optional[float]:
    _ms = _market_service()
    metadata = _ms._load_cached_kap_market_metadata(_ms.CONFIG.paths.processed_dir / "kap_cache", symbol)
    return _ms._positive_float(metadata.get("shares_outstanding"))


def _index_weight_inputs_for_symbol(
    symbol: str,
    *,
    cached_metadata: Optional[Dict[str, Any]] = None,
    basic_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    _ms = _market_service()
    normalized_symbol = str(symbol or "").strip().upper()
    if basic_summary is None:
        basic_summary = _ms._fetch_isyatirim_basic_summary_map().get(normalized_symbol, {})
    shares = (
        _ms._positive_float(cached_metadata.get("shares_outstanding"))
        if cached_metadata is not None
        else _ms._latest_share_count_from_kap_cache(normalized_symbol)
    )
    if shares is None:
        shares = _ms._positive_float(basic_summary.get("shares_outstanding"))
    fdpo = _ms._positive_float(basic_summary.get("fdpo"))
    return {
        "shares_outstanding": shares,
        "fdpo": fdpo,
        "weight_coefficient": 1.0 if fdpo is not None else None,
    }


def _apply_index_weight_formula(
    rows: List[Dict[str, Any]],
    *,
    index_level: Any,
) -> tuple[List[Dict[str, Any]], str]:
    enriched: List[Dict[str, Any]] = []
    free_float_values: List[float] = []
    for row in rows:
        price = row.get("price")
        shares = row.get("shares_outstanding")
        fdpo = row.get("fdpo")
        coefficient = row.get("weight_coefficient")
        market_cap = row.get("market_cap")
        free_float_market_value = None
        try:
            if price is not None and shares is not None and fdpo is not None and coefficient is not None:
                free_float_market_value = float(price) * float(shares) * float(fdpo) * float(coefficient)
            elif market_cap is not None and fdpo is not None and coefficient is not None:
                free_float_market_value = float(market_cap) * float(fdpo) * float(coefficient)
        except (TypeError, ValueError):
            free_float_market_value = None
        enriched_row = {
            **row,
            "free_float_market_value": free_float_market_value if free_float_market_value and free_float_market_value > 0 else None,
            "weight_pct": None,
            "point_effect": None,
        }
        enriched.append(enriched_row)
        if enriched_row["free_float_market_value"] is not None:
            free_float_values.append(float(enriched_row["free_float_market_value"]))

    if len(free_float_values) != len(enriched) or not free_float_values:
        return [
            {
                **row,
                "free_float_market_value": None,
                "weight_pct": None,
                "point_effect": None,
            }
            for row in enriched
        ], "unavailable"

    total = sum(free_float_values)
    if total <= 0:
        return enriched, "unavailable"

    try:
        level = float(index_level)
    except (TypeError, ValueError):
        level = 0.0

    calculated: List[Dict[str, Any]] = []
    for row in enriched:
        ffmv = float(row["free_float_market_value"])
        weight_pct = (ffmv / total) * 100.0
        change_pct = row.get("change_pct")
        point_effect = None
        try:
            if change_pct is not None and level > 0:
                point_effect = level * (weight_pct / 100.0) * (float(change_pct) / 100.0)
        except (TypeError, ValueError):
            point_effect = None
        calculated.append(
            {
                **row,
                "weight_pct": round(weight_pct, 4),
                "point_effect": round(point_effect, 2) if point_effect is not None else None,
            }
        )
    calculated.sort(
        key=lambda item: (
            item.get("point_effect") is None,
            -abs(float(item.get("point_effect") or 0.0)),
            str(item.get("symbol") or ""),
        )
    )
    return calculated, "available"


def _market_index_constituent_stock_rows(
    index_code: str,
    *,
    basic_summary_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    _ms = _market_service()
    normalized = _ms._normalize_market_index(index_code)
    if basic_summary_map is None:
        basic_summary_map = _ms._fetch_isyatirim_basic_summary_map()
    if normalized in _ms._MARKET_STOCK_INDEXES:
        return list(
            _ms._market_stocks_payload(
                index_name=normalized,
                basic_summary_map=basic_summary_map,
            ).get("rows", [])
        )

    from app.kap_service import get_bist_index_universe

    try:
        universe = get_bist_index_universe(normalized)
        symbols = [
            str(symbol or "").strip().upper()
            for symbol in universe.get("symbols", [])
            if str(symbol or "").strip()
        ]
    except Exception:
        symbols = []
    if not symbols:
        return []

    price_map = _ms._fetch_market_price_map(symbols, index_name="XUTUM")
    cache_dir = _ms.CONFIG.paths.processed_dir / "kap_cache"
    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        quote = price_map.get(symbol, {})
        cached_meta = _ms._load_cached_kap_market_metadata(cache_dir, symbol)
        basic_summary = basic_summary_map.get(symbol)
        rows.append(
            {
                "company": symbol,
                "price": quote.get("price"),
                "price_currency": quote.get("currency"),
                "change_pct": quote.get("change_pct"),
                "volume": quote.get("volume"),
                "market_cap": _ms._market_cap_from_quote_and_meta(quote, cached_meta, basic_summary),
                "_kap_market_metadata": cached_meta,
                **_ms._empty_logo_payload(),
            }
        )
    return rows


def _index_constituents(index_code: str, *, index_level: Any) -> tuple[List[Dict[str, Any]], str]:
    _ms = _market_service()
    normalized = _ms._normalize_market_index(index_code)
    basic_summary_map = _ms._fetch_isyatirim_basic_summary_map()
    rows: List[Dict[str, Any]] = []
    for stock in _ms._market_index_constituent_stock_rows(
        normalized,
        basic_summary_map=basic_summary_map,
    ):
        symbol = str(stock.get("company") or "").strip().upper()
        if not symbol:
            continue
        cached_metadata = stock.get("_kap_market_metadata")
        weight_inputs = _ms._index_weight_inputs_for_symbol(
            symbol,
            cached_metadata=cached_metadata if isinstance(cached_metadata, dict) else None,
            basic_summary=basic_summary_map.get(symbol) or {},
        )
        rows.append(
            {
                "symbol": symbol,
                "price": stock.get("price"),
                "price_currency": stock.get("price_currency"),
                "change_pct": stock.get("change_pct"),
                "volume": stock.get("volume"),
                "market_cap": stock.get("market_cap"),
                "logo_url": stock.get("logo_url"),
                "logo_source": stock.get("logo_source"),
                "shares_outstanding": weight_inputs.get("shares_outstanding"),
                "fdpo": weight_inputs.get("fdpo"),
                "weight_coefficient": weight_inputs.get("weight_coefficient"),
            }
        )
    weighted_rows, weight_status = _ms._apply_index_weight_formula(rows, index_level=index_level)
    if weight_status != "available":
        weighted_rows.sort(
            key=lambda item: (
                item.get("change_pct") is None,
                -abs(float(item.get("change_pct") or 0.0)),
                str(item.get("symbol") or ""),
            )
        )
    return weighted_rows, weight_status


def _market_index_detail_payload(index_code: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    normalized = _ms._normalize_market_index(index_code)

    def build() -> Dict[str, Any]:
        quote = _ms._fetch_index_quote(normalized)
        row = _ms._market_index_row(normalized, quote=quote)
        intraday = _ms._index_intraday_payload(normalized)
        constituents, weight_status = _ms._index_constituents(normalized, index_level=row.get("price"))
        return {
            **row,
            "high": row.get("high") if row.get("high") is not None else intraday.get("high"),
            "low": row.get("low") if row.get("low") is not None else intraday.get("low"),
            "prev_close": row.get("prev_close") if row.get("prev_close") is not None else intraday.get("prev_close"),
            "line_points": intraday.get("line_points") or [],
            "constituents": constituents,
            "weight_status": weight_status,
            "weight_note": (
                "Tahmini ağırlık: İş Yatırım halka açıklık oranı ve ağırlık katsayısı 1 varsayımıyla hesaplandı."
                if weight_status == "available"
                else "Ağırlık verisi bulunamadı: pay sayısı, FDPO ve ağırlık katsayısı eksiksiz olmadığı için puan etkisi hesaplanamadı."
            ),
            "source": "yahoo_finance_chart",
            "as_of": row.get("as_of") or _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key=f"api:market:index-detail:{normalized}:v2",
        factory=build,
        fresh_ttl_seconds=_ms._effective_bist_index_fresh_ttl(),
        stale_ttl_seconds=_ms._MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_ms._MARKET_INDEX_DETAIL_CACHE,
        local_key=normalized,
        force_revalidate=force_refresh,
    )
    if payload is None:
        raise _ms.HTTPException(status_code=503, detail="Endeks verisi yenileniyor. Lütfen kısa süre sonra tekrar deneyin.")
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _market_commodities_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        def _one(entry: tuple[str, str, str, Optional[str]]) -> Dict[str, Any]:
            symbol, yahoo_symbol, label, forced_currency = entry
            quote = _ms._fetch_yahoo_quote(yahoo_symbol)
            return {
                "symbol": symbol,
                "label": label,
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price") if quote.get("ok") else None,
                "prev_close": quote.get("prev_close") if quote.get("ok") else None,
                "change": quote.get("change") if quote.get("ok") else None,
                "change_pct": quote.get("change_pct") if quote.get("ok") else None,
                "currency": forced_currency or quote.get("currency") or "USD",
                "market_state": quote.get("market_state") if quote.get("ok") else "",
                "as_of": quote.get("as_of") if quote.get("ok") else None,
                "error": None if quote.get("ok") else quote.get("error"),
                "logo_url": None,
                "logo_source": None,
            }

        items: List[Dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                for row in pool.map(_one, _ms._COMMODITY_MAP):
                    items.append(row)
        except Exception:
            for entry in _ms._COMMODITY_MAP:
                items.append(_one(entry))
        return {
            "items": items,
            "source": "yahoo_finance_chart",
            "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key="api:market:commodities:v2",
        factory=build,
        fresh_ttl_seconds=_ms._effective_macro_fresh_ttl(),
        stale_ttl_seconds=_ms._MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_ms._COMMODITY_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {
            "items": [],
            "source": "yahoo_finance_chart",
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
            "error": "market_commodities_refresh_pending",
        }
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _fx_quote_currency(symbol: str) -> str:
    if "/" not in symbol:
        return ""
    return str(symbol or "").rsplit("/", 1)[-1].strip().upper()


def _fetch_fx_return_bases(yahoo_symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized = str(yahoo_symbol or "").strip()
    if not normalized:
        return {}
    now = _ms.time.time()
    cached = _ms._FX_RETURN_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _ms._FX_RETURN_CACHE_TTL:
        return dict(cached.get("data") or {})
    _ms._warm_bulk_return_bases(
        _ms._FX_RETURN_CACHE,
        _ms._FX_RETURN_BASES_BULK_KEY,
        _ms._FX_RETURN_CACHE_TTL,
    )
    cached_after_warm = _ms._FX_RETURN_CACHE.get(normalized)
    if cached_after_warm and now - cached_after_warm.get("_ts", 0) < _ms._FX_RETURN_CACHE_TTL:
        return dict(cached_after_warm.get("data") or {})

    chart = _ms._fetch_yahoo_chart_raw(normalized, interval="1d", range_="1y")
    if not chart.get("ok"):
        return _ms._store_bulk_return_base_entry(
            _ms._FX_RETURN_CACHE,
            _ms._FX_RETURN_BASES_BULK_KEY,
            normalized,
            {},
            _ms._FX_RETURN_CACHE_TTL,
            now,
        )

    points = [
        (_ms.datetime.fromisoformat(str(point["time"])), float(point["close"]))
        for point in chart.get("points", [])
        if point.get("time") and isinstance(point.get("close"), (int, float))
    ]
    if not points:
        return _ms._store_bulk_return_base_entry(
            _ms._FX_RETURN_CACHE,
            _ms._FX_RETURN_BASES_BULK_KEY,
            normalized,
            {},
            _ms._FX_RETURN_CACHE_TTL,
            now,
        )

    points.sort(key=lambda item: item[0])
    latest_dt, latest_close = points[-1]
    year_start = _ms.datetime(latest_dt.year, 1, 1, tzinfo=_ms.timezone.utc)
    data = {
        "base_1w": _ms._pick_series_value_at_or_before(points, latest_dt - _ms.timedelta(days=7)),
        "base_1m": _ms._pick_series_value_at_or_before(points, latest_dt - _ms.timedelta(days=30)),
        "base_3m": _ms._pick_series_value_at_or_before(points, latest_dt - _ms.timedelta(days=91)),
        "base_6m": _ms._pick_series_value_at_or_before(points, latest_dt - _ms.timedelta(days=182)),
        "base_ytd": _ms._pick_series_value_at_or_after(points, year_start),
        "base_1y": _ms._pick_series_value_at_or_before(points, latest_dt - _ms.timedelta(days=365)),
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
    }
    return _ms._store_bulk_return_base_entry(
        _ms._FX_RETURN_CACHE,
        _ms._FX_RETURN_BASES_BULK_KEY,
        normalized,
        data,
        _ms._FX_RETURN_CACHE_TTL,
        now,
    )


def _fx_returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Optional[float]]:
    _ms = _market_service()
    current_for_returns = current_price if current_price is not None else return_bases.get("latest_close")
    return {
        response_field: _ms._return_pct(current_for_returns, return_bases.get(base_field))
        for response_field, base_field in _ms._RETURN_BASE_FIELDS
    }


def _fx_item_from_quote(symbol: str, yahoo_symbol: str, label: str, quote: Dict[str, Any], return_bases: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _ms = _market_service()
    period_returns = _ms._fx_returns_from_bases(quote.get("price"), return_bases or {}) if quote.get("ok") else {}
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": yahoo_symbol,
        "price": quote.get("price") if quote.get("ok") else None,
        "prev_close": quote.get("prev_close") if quote.get("ok") else None,
        "change": quote.get("change") if quote.get("ok") else None,
        "change_pct": quote.get("change_pct") if quote.get("ok") else None,
        "currency": _ms._fx_quote_currency(symbol),
        "market_state": quote.get("market_state") if quote.get("ok") else "",
        "as_of": quote.get("as_of") if quote.get("ok") else None,
        "error": None if quote.get("ok") else quote.get("error"),
        "logo_url": None,
        "logo_source": None,
        **period_returns,
    }


def _fx_direct_item(entry: tuple[str, List[str], str]) -> Dict[str, Any]:
    _ms = _market_service()
    symbol, yahoo_candidates, label = entry
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        quote = _ms._fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            return_bases = _ms._fetch_fx_return_bases(yahoo_symbol) if symbol.endswith("/TRY") else None
            return _ms._fx_item_from_quote(symbol, yahoo_symbol, label, quote, return_bases)
        errors.append(str(quote.get("error") or "quote_unavailable"))

    return _ms._fx_item_from_quote(
        symbol,
        yahoo_candidates[0] if yahoo_candidates else "",
        label,
        {"ok": False, "error": "; ".join(errors[:3]) if errors else "quote_unavailable"},
    )


def _positive_number(raw: Any) -> Optional[float]:
    _ms = _market_service()
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not _ms.math.isfinite(value) or value <= 0:
        return None
    return value


def _fx_derived_item(
    symbol: str,
    label: str,
    numerator_symbol: str,
    denominator_symbol: str,
    items_by_symbol: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    _ms = _market_service()
    numerator = items_by_symbol.get(numerator_symbol, {})
    denominator = items_by_symbol.get(denominator_symbol, {})

    numerator_price = _ms._positive_number(numerator.get("price"))
    denominator_price = _ms._positive_number(denominator.get("price"))
    numerator_prev = _ms._positive_number(numerator.get("prev_close"))
    denominator_prev = _ms._positive_number(denominator.get("prev_close"))

    price = numerator_price / denominator_price if numerator_price is not None and denominator_price is not None else None
    prev_close = numerator_prev / denominator_prev if numerator_prev is not None and denominator_prev is not None else None
    change = None
    change_pct = None
    if price is not None and prev_close is not None and prev_close > 0:
        change = round(price - prev_close, 6)
        change_pct = round((change / prev_close) * 100, 4)

    source_symbol = "/".join(
        item
        for item in (
            str(numerator.get("yahoo_symbol") or numerator_symbol),
            str(denominator.get("yahoo_symbol") or denominator_symbol),
        )
        if item
    )
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": source_symbol,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": _ms._fx_quote_currency(symbol),
        "market_state": numerator.get("market_state") or denominator.get("market_state") or "",
        "as_of": numerator.get("as_of") or denominator.get("as_of"),
        "error": None if price is not None else "derived_quote_unavailable",
        "logo_url": None,
        "logo_source": None,
        "return_1w_pct": None,
        "return_1m_pct": None,
        "return_3m_pct": None,
        "return_6m_pct": None,
        "return_ytd_pct": None,
        "return_1y_pct": None,
    }


def _market_fx_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        items_by_symbol: Dict[str, Dict[str, Any]] = {}
        _ms._begin_bulk_return_batch(
            _ms._FX_RETURN_CACHE,
            _ms._FX_RETURN_BASES_BULK_KEY,
            _ms._FX_RETURN_CACHE_TTL,
        )
        try:
            try:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for row in pool.map(_ms._fx_direct_item, _ms._FX_DIRECT_MAP):
                        items_by_symbol[str(row.get("symbol") or "")] = row
            except Exception:
                for entry in _ms._FX_DIRECT_MAP:
                    row = _ms._fx_direct_item(entry)
                    items_by_symbol[str(row.get("symbol") or "")] = row
        finally:
            _ms._end_bulk_return_batch(
                _ms._FX_RETURN_CACHE,
                _ms._FX_RETURN_BASES_BULK_KEY,
                _ms._FX_RETURN_CACHE_TTL,
            )

        for symbol, label, numerator_symbol, denominator_symbol in _ms._FX_DERIVED_MAP:
            items_by_symbol[symbol] = _ms._fx_derived_item(
                symbol,
                label,
                numerator_symbol,
                denominator_symbol,
                items_by_symbol,
            )

        items = [
            items_by_symbol[symbol]
            for symbol in _ms._FX_ORDER
            if symbol in items_by_symbol
        ]
        return {
            "items": items,
            "source": "yahoo_finance_chart",
            "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key="api:market:fx:v2",
        factory=build,
        fresh_ttl_seconds=_ms._effective_macro_fresh_ttl(),
        stale_ttl_seconds=_ms._MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_ms._FX_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {
            "items": [],
            "source": "yahoo_finance_chart",
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
            "error": "market_fx_refresh_pending",
        }
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _empty_watch_item(
    symbol: str,
    label: str,
    *,
    currency: str = "",
    error: Optional[str] = None,
    yahoo_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": yahoo_symbol,
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "currency": currency,
        "market_state": "",
        "as_of": None,
        "error": error,
        "logo_url": None,
        "logo_source": None,
    }


def _normalize_watch_item(item: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(item.get("symbol") or "").strip()
    label = str(item.get("label") or symbol).strip() or symbol
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": item.get("yahoo_symbol"),
        "price": item.get("price"),
        "prev_close": item.get("prev_close"),
        "change": item.get("change"),
        "change_pct": item.get("change_pct"),
        "currency": item.get("currency") or "",
        "market_state": item.get("market_state") or "",
        "as_of": item.get("as_of"),
        "error": item.get("error"),
        "logo_url": item.get("logo_url"),
        "logo_source": item.get("logo_source"),
    }


def _pick_watch_items(
    items: List[Dict[str, Any]],
    symbols: List[str],
    fallback_labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    _ms = _market_service()
    mapped = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in items
        if str(row.get("symbol") or "").strip()
    }
    selected: List[Dict[str, Any]] = []
    for symbol in symbols:
        row = mapped.get(symbol.upper())
        if row:
            selected.append(_ms._normalize_watch_item(row))
            continue
        selected.append(
            _ms._empty_watch_item(
                symbol=symbol,
                label=fallback_labels.get(symbol, symbol),
                error="instrument_not_found",
            )
        )
    return selected


def _watch_index_item(
    symbol: str,
    label: str,
    yahoo_candidates: List[str],
    *,
    fallback_currency: str = "TRY",
) -> Dict[str, Any]:
    _ms = _market_service()
    normalized_symbol = str(symbol or "").strip().upper()
    if normalized_symbol in _ms._MARKET_INDEX_META:
        quote = _ms._fetch_index_quote(normalized_symbol)
        if quote.get("price") is not None:
            return {
                "symbol": symbol,
                "label": label or quote.get("label") or symbol,
                "yahoo_symbol": quote.get("yahoo_symbol"),
                "price": quote.get("price"),
                "prev_close": quote.get("prev_close"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "currency": quote.get("currency") or fallback_currency,
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
                "logo_url": None,
                "logo_source": None,
            }
        return _ms._empty_watch_item(
            symbol=symbol,
            label=label or quote.get("label") or symbol,
            currency=fallback_currency,
            error=quote.get("error") or "quote_unavailable",
            yahoo_symbol=quote.get("yahoo_symbol"),
        )

    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        quote = _ms._fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            return {
                "symbol": symbol,
                "label": label,
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price"),
                "prev_close": quote.get("prev_close"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "currency": quote.get("currency") or fallback_currency,
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
                "logo_url": None,
                "logo_source": None,
            }
        err = str(quote.get("error") or "quote_unavailable")
        errors.append(f"{yahoo_symbol}:{err}")

    return _ms._empty_watch_item(
        symbol=symbol,
        label=label,
        currency=fallback_currency,
        error="; ".join(errors[:3]) if errors else "quote_unavailable",
    )


def _market_watch_global_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        def _one(entry: tuple[str, str, List[str]]) -> Dict[str, Any]:
            symbol, label, yahoo_candidates = entry
            return _ms._watch_index_item(
                symbol=symbol,
                label=label,
                yahoo_candidates=yahoo_candidates,
                fallback_currency="USD" if symbol in _ms._WATCH_GLOBAL_INDEX_SYMBOLS else "",
            )

        try:
            with ThreadPoolExecutor(max_workers=min(8, len(_ms._WATCH_GLOBAL_INDEX_CANDIDATES))) as pool:
                items = list(pool.map(_one, _ms._WATCH_GLOBAL_INDEX_CANDIDATES))
        except Exception:
            items = [_one(entry) for entry in _ms._WATCH_GLOBAL_INDEX_CANDIDATES]
        return {
            "items": items,
            "source": "yahoo_finance_chart",
            "delay_note": _ms._WATCH_DELAY_NOTE,
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    fresh_ttl = _ms._effective_macro_fresh_ttl()
    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key="api:market:watch-global:v2",
        factory=build,
        fresh_ttl_seconds=fresh_ttl,
        stale_ttl_seconds=max(_ms._MARKET_SWR_STALE_TTL_SECONDS, fresh_ttl * 2),
        local_cache=_ms._WATCH_GLOBAL_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {"items": [], "source": "yahoo_finance_chart", "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat()}
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _market_watch_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    def build() -> Dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        fx_payload = _ms._market_fx_payload()
        commodity_payload = _ms._market_commodities_payload()

        def _watch_index(entry: tuple[str, str, List[str]]) -> Dict[str, Any]:
            symbol, label, yahoo_candidates = entry
            fallback_curr = "USD" if symbol in _ms._WATCH_GLOBAL_INDEX_SYMBOLS else "TRY"
            return _ms._watch_index_item(
                symbol=symbol,
                label=label,
                yahoo_candidates=yahoo_candidates,
                fallback_currency=fallback_curr,
            )

        with ThreadPoolExecutor(max_workers=min(6, len(_ms._WATCH_BIST_INDEX_CANDIDATES))) as pool:
            local_indices = list(pool.map(_watch_index, _ms._WATCH_BIST_INDEX_CANDIDATES))
        local_indices_by_symbol = {row["symbol"]: row for row in local_indices}

        # Reuse the global-index cache for SP500/NASDAQ/DOW instead of hitting
        # Yahoo a second time. Both market-watch endpoints then expose the
        # exact same quote for the same instrument.
        global_indices_by_symbol = {
            str(item.get("symbol") or "").strip().upper(): item
            for item in (_ms._market_watch_global_payload().get("items") or [])
            if str(item.get("symbol") or "").strip()
        }

        indices: List[Dict[str, Any]] = []
        for symbol, label, yahoo_candidates in _ms._WATCH_INDEX_CANDIDATES:
            row = local_indices_by_symbol.get(symbol) or global_indices_by_symbol.get(symbol)
            if row is None:
                row = _watch_index((symbol, label, yahoo_candidates))
            indices.append(row)
        fx_items = _ms._pick_watch_items(
            items=list(fx_payload.get("items") or []),
            symbols=_ms._WATCH_FX_SYMBOLS,
            fallback_labels=_ms._WATCH_FX_LABELS,
        )
        commodity_items = _ms._pick_watch_items(
            items=list(commodity_payload.get("items") or []),
            symbols=_ms._WATCH_COMMODITY_SYMBOLS,
            fallback_labels=_ms._WATCH_COMMODITY_LABELS,
        )
        return {
            "sections": {
                "indices": indices,
                "fx": fx_items,
                "commodities": commodity_items,
            },
            "source": "yahoo_finance_chart",
            "delay_note": _ms._WATCH_DELAY_NOTE,
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key="api:market:watch:v4",
        factory=build,
        fresh_ttl_seconds=_ms._effective_macro_fresh_ttl(),
        stale_ttl_seconds=_ms._MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_ms._WATCH_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {"sections": {}, "source": "yahoo_finance_chart", "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat()}
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _market_watch_unified_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    def build() -> Dict[str, Any]:
        fx_payload = _ms._market_fx_payload(force_refresh=force_refresh)
        commodities_payload = _ms._market_commodities_payload(force_refresh=force_refresh)
        global_indices_payload = _ms._market_watch_global_payload(force_refresh=force_refresh)

        from concurrent.futures import ThreadPoolExecutor

        def _watch_index(entry: tuple[str, str, List[str]]) -> Dict[str, Any]:
            symbol, label, yahoo_candidates = entry
            return _ms._watch_index_item(
                symbol=symbol,
                label=label,
                yahoo_candidates=yahoo_candidates,
                fallback_currency="TRY",
            )

        try:
            with ThreadPoolExecutor(max_workers=min(4, len(_ms._WATCH_BIST_INDEX_CANDIDATES))) as pool:
                bist_indices = list(pool.map(_watch_index, _ms._WATCH_BIST_INDEX_CANDIDATES))
        except Exception:
            bist_indices = [_watch_index(entry) for entry in _ms._WATCH_BIST_INDEX_CANDIDATES]

        seen_symbols: set[str] = set()
        unified_indices: List[Dict[str, Any]] = []
        for item in bist_indices + list(global_indices_payload.get("items") or []):
            sym = str(item.get("symbol") or "").strip().upper()
            if sym and sym not in seen_symbols:
                seen_symbols.add(sym)
                unified_indices.append(item)

        fx_items = list(fx_payload.get("items") or [])
        commodity_items = list(commodities_payload.get("items") or [])
        unified_as_of = _ms.datetime.now(_ms.timezone.utc).isoformat()

        return {
            "indices": unified_indices,
            "fx": fx_items,
            "commodities": commodity_items,
            "sections": {
                "indices": unified_indices,
                "fx": fx_items,
                "commodities": commodity_items,
            },
            "source": "yahoo_finance_chart",
            "delay_note": _ms._WATCH_DELAY_NOTE,
            "as_of": unified_as_of,
        }

    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key="api:market:watch:unified:v1",
        factory=build,
        fresh_ttl_seconds=_ms._effective_macro_fresh_ttl(),
        stale_ttl_seconds=_ms._MARKET_SWR_STALE_TTL_SECONDS,
        local_cache=_ms._WATCH_UNIFIED_CACHE,
        local_key="payload",
        force_revalidate=force_refresh,
    )
    if payload is None:
        payload = {
            "indices": [],
            "fx": [],
            "commodities": [],
            "sections": {"indices": [], "fx": [], "commodities": []},
            "source": "yahoo_finance_chart",
            "delay_note": _ms._WATCH_DELAY_NOTE,
            "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )
