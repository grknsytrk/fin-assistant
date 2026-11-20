from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.market import _market_service


def _pick_series_value_at_or_before(
    points: List[tuple[datetime, float]],
    target: datetime,
) -> Optional[float]:
    candidate = None
    for point_dt, close in points:
        if point_dt <= target:
            candidate = close
        elif candidate is not None:
            break
    return candidate


def _pick_series_value_at_or_after(
    points: List[tuple[datetime, float]],
    target: datetime,
) -> Optional[float]:
    for point_dt, close in points:
        if point_dt >= target:
            return close
    return None


def _stock_return_base_cache_key(symbol: str) -> str:
    _ms = _market_service()
    return f"api:stock-return-bases:{symbol}:{_ms._STOCK_RETURN_BASE_CACHE_VERSION}"


def _stock_return_base_disk_path() -> Path:
    _ms = _market_service()
    return _ms.CONFIG.paths.processed_dir / "_shared" / "stock_return_bases_xutum.json"


def _effective_stock_return_base_ttl(now_ts: Optional[float] = None) -> int:
    """Keep 1h base-price freshness in session and 6h off-hours/weekends."""
    _ms = _market_service()
    dt = (
        _ms.datetime.fromtimestamp(now_ts, tz=_ms._TURKEY_TIMEZONE)
        if now_ts is not None
        else None
    )
    if _ms._is_bist_market_open(dt):
        return max(60, int(_ms._STOCK_RETURN_BASE_TRADING_TTL), int(_ms._STOCK_RETURN_BASE_CACHE_TTL))
    return max(
        60,
        int(_ms._STOCK_RETURN_BASE_TRADING_TTL),
        int(_ms._STOCK_RETURN_BASE_OFFHOURS_TTL),
        int(_ms._STOCK_RETURN_BASE_CACHE_TTL),
    )


def _stock_return_bases_from_series(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    _ms = _market_service()
    if not series:
        return {}

    ordered = sorted(series, key=lambda x: x["dt"])
    latest = ordered[-1]
    latest_dt = latest["dt"]
    latest_close = latest["close"]
    year_start = _ms.datetime(latest_dt.year, 1, 1, tzinfo=_ms.timezone.utc)
    points_for_base = [(s["dt"], s["close"]) for s in ordered]

    def _get_range_stats(start_dt: datetime) -> tuple[Optional[float], Optional[float], Optional[float]]:
        relevant = [s for s in ordered if s["dt"] >= start_dt]
        if not relevant:
            return None, None, None

        base_val = _ms._pick_series_value_at_or_before(points_for_base, start_dt)
        high_val = max(s["high"] for s in relevant)
        low_val = min(s["low"] for s in relevant)
        return base_val, high_val, low_val

    b1w, h1w, l1w = _get_range_stats(latest_dt - _ms.timedelta(days=7))
    b1m, h1m, l1m = _get_range_stats(latest_dt - _ms.timedelta(days=30))
    b3m, h3m, l3m = _get_range_stats(latest_dt - _ms.timedelta(days=91))
    b6m, h6m, l6m = _get_range_stats(latest_dt - _ms.timedelta(days=182))
    bytd, hytd, lytd = _get_range_stats(year_start)
    b1y, h1y, l1y = _get_range_stats(latest_dt - _ms.timedelta(days=365))

    closes = [
        float(s["close"])
        for s in ordered
        if _ms._as_finite_float(s.get("close")) is not None and float(s["close"]) > 0
    ]
    sma_50 = round(sum(closes[-50:]) / 50, 4) if len(closes) >= 50 else None
    sma_200 = round(sum(closes[-200:]) / 200, 4) if len(closes) >= 200 else None

    return {
        "base_1w": b1w, "high_1w": h1w, "low_1w": l1w,
        "base_1m": b1m, "high_1m": h1m, "low_1m": l1m,
        "base_3m": b3m, "high_3m": h3m, "low_3m": l3m,
        "base_6m": b6m, "high_6m": h6m, "low_6m": l6m,
        "base_ytd": bytd, "high_ytd": hytd, "low_ytd": lytd,
        "base_1y": b1y, "high_1y": h1y, "low_1y": l1y,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
    }


def _build_ohlc_series(
    timestamps: List[Any],
    closes: List[Any],
    highs: Optional[List[Any]] = None,
    lows: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    _ms = _market_service()
    series: List[Dict[str, Any]] = []
    high_seq = highs if highs else [None] * len(timestamps)
    low_seq = lows if lows else [None] * len(timestamps)
    for ts, close, high, low in zip(timestamps, closes, high_seq, low_seq):
        if close is None or not isinstance(ts, (int, float)):
            continue
        try:
            dt = _ms.datetime.fromtimestamp(float(ts), tz=_ms.timezone.utc)
            close_val = float(close)
            if not _ms.math.isfinite(close_val) or close_val <= 0:
                continue
            high_val = float(high) if high is not None and _ms.math.isfinite(float(high)) else close_val
            low_val = float(low) if low is not None and _ms.math.isfinite(float(low)) else close_val
            series.append({
                "dt": dt,
                "close": close_val,
                "high": high_val,
                "low": low_val,
            })
        except (TypeError, ValueError):
            continue
    return series


def _remember_stock_return_bases_bulk(
    items: Dict[str, Dict[str, Any]],
    *,
    now_ts: Optional[float] = None,
    write_shared_bulk: bool = True,
    write_disk: bool = True,
) -> None:
    _ms = _market_service()
    if not items:
        return
    now = _ms.time.time() if now_ts is None else now_ts
    ttl = _ms._effective_stock_return_base_ttl(now)
    stale_ttl = max(ttl, int(_ms._STOCK_RETURN_BASE_STALE_TTL))

    clean_items: Dict[str, Dict[str, Any]] = {}
    for sym, bases in items.items():
        ticker = str(sym or "").strip().upper()
        if not ticker or not isinstance(bases, dict) or not bases:
            continue
        payload = dict(bases)
        _ms._STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": now, "data": payload}
        clean_items[ticker] = payload
    if not clean_items:
        return

    if write_shared_bulk:
        existing_bulk = _ms._shared_cache_get_dict(_ms._STOCK_RETURN_BASE_BULK_CACHE_KEY)
        merged_items: Dict[str, Any] = (
            dict(existing_bulk.get("items") or {})
            if isinstance(existing_bulk, dict) and isinstance(existing_bulk.get("items"), dict)
            else {}
        )
        for ticker, payload in clean_items.items():
            merged_items[ticker] = {"_ts": now, "data": payload}
        _ms._shared_cache_set(
            _ms._STOCK_RETURN_BASE_BULK_CACHE_KEY,
            {"created_at": now, "items": merged_items},
            stale_ttl,
        )

    if write_disk and not _ms.os.getenv("PYTEST_CURRENT_TEST"):
        _ms._persist_stock_return_bases_disk(now_ts=now)


def _persist_stock_return_bases_disk(*, now_ts: Optional[float] = None) -> None:
    """Persist return bases ONLY to local disk (never Supabase) for fast restarts."""
    _ms = _market_service()
    try:
        path = _ms._stock_return_base_disk_path()
        items: Dict[str, Dict[str, Any]] = {}
        for ticker in list(_ms._STOCK_RETURN_BASE_CACHE.keys()):
            entry = _ms._STOCK_RETURN_BASE_CACHE.get(ticker)
            if not isinstance(entry, dict):
                continue
            data = entry.get("data")
            if not isinstance(data, dict) or not data:
                continue
            items[str(ticker).upper()] = {
                "_ts": float(entry.get("_ts") or 0.0),
                "data": data,
            }
        if not items:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _ms._STOCK_RETURN_BASE_CACHE_VERSION,
            "updated_at": _ms.time.time() if now_ts is None else now_ts,
            "items": items,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(_ms.json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        _ms.LOGGER.debug("stock return base local disk persist skipped: %s", exc)


def _hydrate_stock_return_bases_from_disk() -> int:
    """Load local disk return bases into process memory/Redis without touching Supabase."""
    _ms = _market_service()
    path = _ms._stock_return_base_disk_path()
    if not path.exists():
        return 0
    try:
        raw = _ms.json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(raw, dict):
        return 0
    items = raw.get("items")
    if not isinstance(items, dict):
        return 0

    now = _ms.time.time()
    max_age = max(3600, int(_ms._STOCK_RETURN_BASE_STALE_TTL))
    hydrated = 0
    bulk_items: Dict[str, Dict[str, Any]] = {}
    for sym, entry in items.items():
        ticker = str(sym or "").strip().upper()
        if not ticker or not isinstance(entry, dict):
            continue
        data = entry.get("data")
        try:
            ts = float(entry.get("_ts") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        if not isinstance(data, dict) or not data or (ts > 0 and (now - ts) > max_age):
            continue
        _ms._STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": ts, "data": dict(data)}
        bulk_items[ticker] = {"_ts": ts, "data": dict(data)}
        hydrated += 1
    if bulk_items:
        existing_bulk = _ms._shared_cache_get_dict(_ms._STOCK_RETURN_BASE_BULK_CACHE_KEY)
        existing_items = (existing_bulk or {}).get("items") if isinstance(existing_bulk, dict) else None
        if isinstance(existing_items, dict) and len(existing_items) >= len(bulk_items):
            return hydrated
        _ms._shared_cache_set(
            _ms._STOCK_RETURN_BASE_BULK_CACHE_KEY,
            {"created_at": now, "items": bulk_items},
            max_age,
        )
    return hydrated


def _fetch_stock_return_bases(symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {}

    now = _ms.time.time()
    ttl = _ms._effective_stock_return_base_ttl(now)
    cached = _ms._STOCK_RETURN_BASE_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < ttl:
        return dict(cached.get("data") or {})
    shared_key = _ms._stock_return_base_cache_key(ticker)
    shared_cached = _ms._shared_cache_get_dict(shared_key)
    if shared_cached:
        _ms._STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached)

    yahoo_symbol = f"{ticker}.IS"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        # A one-year response can start on the first trading day after the
        # exact 365-day boundary, leaving no base price at-or-before it.
        # Two years keeps the displayed return truly one-year while providing
        # enough history to select the preceding trading session.
        "?interval=1d&range=2y"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _ms.json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception):
        return {}

    try:
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
    except (KeyError, IndexError, TypeError):
        return {}

    series = _ms._build_ohlc_series(timestamps, closes, highs, lows)
    bases = _ms._stock_return_bases_from_series(series)
    if not bases:
        return {}

    _ms._STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": now, "data": bases}
    _ms._shared_cache_set(shared_key, bases, ttl)
    return dict(bases)


def _fetch_stock_return_bases_spark_batch(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch multi-symbol 2y daily closes in one Yahoo v7 spark call (~20 tickers/req)."""
    _ms = _market_service()
    import urllib.error
    import urllib.parse
    import urllib.request

    cleaned = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
    if not cleaned:
        return {}

    yahoo_to_ticker = {f"{ticker}.IS": ticker for ticker in cleaned}
    encoded_syms = urllib.parse.quote(",".join(yahoo_to_ticker.keys()), safe=",")
    url = f"https://query1.finance.yahoo.com/v7/finance/spark?symbols={encoded_syms}&range=2y&interval=1d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _ms.json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception):
        return {}

    items = ((data.get("spark") or {}).get("result") or []) if isinstance(data, dict) else []
    parsed: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_sym = str(item.get("symbol") or "").strip().upper()
        ticker = yahoo_to_ticker.get(raw_sym) or raw_sym.removesuffix(".IS")
        responses = item.get("response") or []
        if not ticker or not responses or not isinstance(responses[0], dict):
            continue
        r0 = responses[0]
        timestamps = r0.get("timestamp") or []
        quote = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        highs = quote.get("high") or None
        lows = quote.get("low") or None
        series = _ms._build_ohlc_series(timestamps, closes, highs, lows)
        bases = _ms._stock_return_bases_from_series(series)
        if bases:
            parsed[ticker] = bases
    return parsed


def _fetch_stock_return_bases_bulk(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    _ms = _market_service()
    normalized_symbols = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not normalized_symbols:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    stale: List[str] = []
    now = _ms.time.time()
    ttl = _ms._effective_stock_return_base_ttl(now)

    missing_from_memory: List[str] = []
    for symbol in normalized_symbols:
        cached = _ms._STOCK_RETURN_BASE_CACHE.get(symbol)
        if cached and now - cached.get("_ts", 0) < ttl:
            result[symbol] = dict(cached.get("data") or {})
        else:
            missing_from_memory.append(symbol)

    if missing_from_memory:
        bulk_cached = _ms._shared_cache_get_dict(_ms._STOCK_RETURN_BASE_BULK_CACHE_KEY)
        bulk_items = (
            bulk_cached.get("items")
            if isinstance(bulk_cached, dict) and isinstance(bulk_cached.get("items"), dict)
            else {}
        )
        still_missing: List[str] = []
        for symbol in missing_from_memory:
            bulk_entry = bulk_items.get(symbol)
            if (
                isinstance(bulk_entry, dict)
                and isinstance(bulk_entry.get("data"), dict)
                and bulk_entry["data"]
                and now - float(bulk_entry.get("_ts") or 0.0) < ttl
            ):
                _ms._STOCK_RETURN_BASE_CACHE[symbol] = {
                    "_ts": float(bulk_entry.get("_ts") or now),
                    "data": dict(bulk_entry["data"]),
                }
                result[symbol] = dict(bulk_entry["data"])
                continue
            shared_cached = _ms._shared_cache_get_dict(_ms._stock_return_base_cache_key(symbol))
            if shared_cached:
                _ms._STOCK_RETURN_BASE_CACHE[symbol] = {"_ts": now, "data": shared_cached}
                result[symbol] = dict(shared_cached)
                continue
            still_missing.append(symbol)
        stale = still_missing

    if not stale:
        return result

    if len(stale) == 1:
        symbol = stale[0]
        bases = _ms._fetch_stock_return_bases(symbol)
        result[symbol] = bases
        if bases:
            _ms._remember_stock_return_bases_bulk({symbol: bases}, now_ts=now)
        return result

    from concurrent.futures import ThreadPoolExecutor

    batches = [
        stale[i : i + _ms._STOCK_RETURN_BASE_SPARK_BATCH_SIZE]
        for i in range(0, len(stale), _ms._STOCK_RETURN_BASE_SPARK_BATCH_SIZE)
    ]
    fetched_map: Dict[str, Dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=min(6, len(batches))) as pool:
            for batch_res in pool.map(_ms._fetch_stock_return_bases_spark_batch, batches):
                fetched_map.update(batch_res)
    except Exception:
        for batch in batches:
            fetched_map.update(_ms._fetch_stock_return_bases_spark_batch(batch))

    missed = [symbol for symbol in stale if symbol not in fetched_map]
    if missed:
        try:
            with ThreadPoolExecutor(max_workers=min(6, len(missed))) as pool:
                for symbol, bases in zip(missed, pool.map(_ms._fetch_stock_return_bases, missed)):
                    if bases:
                        fetched_map[symbol] = bases
        except Exception:
            for symbol in missed:
                bases = _ms._fetch_stock_return_bases(symbol)
                if bases:
                    fetched_map[symbol] = bases

    if fetched_map:
        _ms._remember_stock_return_bases_bulk(fetched_map, now_ts=now)

    for symbol in stale:
        bases = fetched_map.get(symbol, {})
        result[symbol] = bases
        if not bases and symbol not in _ms._STOCK_RETURN_BASE_CACHE:
            # Avoid repeatedly hammering upstream for delisted/suspended tickers.
            _ms._STOCK_RETURN_BASE_CACHE[symbol] = {"_ts": 0.0, "_neg_ts": now, "data": {}}
    return result


def _return_pct(current_price: Any, base_price: Any) -> Optional[float]:
    try:
        price = float(current_price)
        base = float(base_price)
    except (TypeError, ValueError):
        return None
    if price <= 0 or base <= 0:
        return None
    return round(((price - base) / base) * 100, 2)


def _returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Any]:
    _ms = _market_service()
    res = {}
    for response_field, base_field in _ms._RETURN_BASE_FIELDS:
        base_val = return_bases.get(base_field)
        res[response_field] = _ms._return_pct(current_price, base_val)

        # Extract period (e.g. 1w from return_1w_pct)
        period = response_field.replace("return_", "").replace("_pct", "")
        res[f"base_{period}"] = base_val
        res[f"high_{period}"] = return_bases.get(f"high_{period}")
        res[f"low_{period}"] = return_bases.get(f"low_{period}")

    sma_50 = _ms._as_finite_float(return_bases.get("sma_50"))
    sma_200 = _ms._as_finite_float(return_bases.get("sma_200"))
    high_1y = _ms._as_finite_float(return_bases.get("high_1y"))
    low_1y = _ms._as_finite_float(return_bases.get("low_1y"))
    res["sma_50"] = round(sma_50, 2) if sma_50 is not None else None
    res["sma_200"] = round(sma_200, 2) if sma_200 is not None else None
    res["sma_50_diff_pct"] = _ms._return_pct(current_price, sma_50)
    res["sma_200_diff_pct"] = _ms._return_pct(current_price, sma_200)
    res["high_52w_dist_pct"] = _ms._return_pct(current_price, high_1y)
    res["low_52w_dist_pct"] = _ms._return_pct(current_price, low_1y)
    return res


def _cached_stock_return_bases_bulk(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    _ms = _market_service()
    now = _ms.time.time()
    max_stale = max(_ms._effective_stock_return_base_ttl(now), int(_ms._STOCK_RETURN_BASE_STALE_TTL))
    result: Dict[str, Dict[str, Any]] = {}
    missing_from_memory: List[str] = []

    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            continue
        cached = _ms._STOCK_RETURN_BASE_CACHE.get(normalized)
        if cached and now - float(cached.get("_ts", 0.0)) < max_stale:
            result[normalized] = dict(cached.get("data") or {})
        else:
            missing_from_memory.append(normalized)

    if missing_from_memory:
        bulk_cached = _ms._shared_cache_get_dict(_ms._STOCK_RETURN_BASE_BULK_CACHE_KEY)
        bulk_items = (
            bulk_cached.get("items")
            if isinstance(bulk_cached, dict) and isinstance(bulk_cached.get("items"), dict)
            else {}
        )
        for normalized in missing_from_memory:
            bulk_entry = bulk_items.get(normalized)
            if (
                isinstance(bulk_entry, dict)
                and isinstance(bulk_entry.get("data"), dict)
                and bulk_entry["data"]
                and now - float(bulk_entry.get("_ts") or 0.0) < max_stale
            ):
                _ms._STOCK_RETURN_BASE_CACHE[normalized] = {
                    "_ts": float(bulk_entry.get("_ts") or now),
                    "data": dict(bulk_entry["data"]),
                }
                result[normalized] = dict(bulk_entry["data"])
                continue
            shared_cached = _ms._shared_cache_get_dict(_ms._stock_return_base_cache_key(normalized))
            if shared_cached:
                _ms._STOCK_RETURN_BASE_CACHE[normalized] = {"_ts": now, "data": shared_cached}
                result[normalized] = dict(shared_cached)
    return result


def _stock_return_base_stale_symbols(symbols: List[str], *, now_ts: Optional[float] = None) -> List[str]:
    _ms = _market_service()
    now = _ms.time.time() if now_ts is None else now_ts
    ttl = _ms._effective_stock_return_base_ttl(now)
    stale: List[str] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            continue
        cached = _ms._STOCK_RETURN_BASE_CACHE.get(symbol)
        if cached and not cached.get("data") and (now - float(cached.get("_neg_ts", 0.0))) < 1800:
            continue
        if not cached or (now - float(cached.get("_ts", 0.0))) >= ttl or not cached.get("data") or "sma_50" not in cached.get("data", {}):
            stale.append(symbol)
    return stale


def _warm_stock_return_bases_sync(symbols: Optional[List[str]] = None) -> Dict[str, int]:
    """Fetch stale/missing return bases via batched Spark calls and refresh XUTUM cache."""
    _ms = _market_service()
    target_symbols = (
        [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
        if symbols is not None
        else _ms._market_stock_symbols_for_index("XUTUM")
    )
    stale_symbols = _ms._stock_return_base_stale_symbols(target_symbols)
    if not stale_symbols:
        return {"requested": 0, "populated": 0}
    fetched = _ms._fetch_stock_return_bases_bulk(stale_symbols)
    populated = sum(1 for val in fetched.values() if isinstance(val, dict) and bool(val))
    if populated > 0:
        _ms._refresh_cached_xutum_stocks_payload()
    return {"requested": len(stale_symbols), "populated": populated}


def _schedule_xutum_return_bases_warmup_if_needed(symbols: Optional[List[str]] = None) -> bool:
    """Kick off a single-flight non-blocking batch warmup when XUTUM bases are cold/stale."""
    _ms = _market_service()

    if _ms.os.getenv("PYTEST_CURRENT_TEST"):
        return False

    target_symbols = (
        list(symbols)
        if symbols is not None
        else _ms._market_stock_symbols_for_index("XUTUM")
    )
    if not target_symbols:
        return False

    now = _ms.time.time()
    stale_symbols = _ms._stock_return_base_stale_symbols(target_symbols, now_ts=now)
    if not stale_symbols:
        return False

    with _ms._STOCK_RETURN_BASE_WARMUP_LOCK:
        if _ms._STOCK_RETURN_BASE_WARMUP_RUNNING:
            return False
        if (now - _ms._STOCK_RETURN_BASE_WARMUP_LAST_FINISHED) < _ms._STOCK_RETURN_BASE_WARMUP_COOLDOWN_SECONDS:
            return False
        _ms._STOCK_RETURN_BASE_WARMUP_RUNNING = True

    def _run_warmup() -> None:
        try:
            summary = _ms._warm_stock_return_bases_sync(stale_symbols)
            _ms.LOGGER.info("XUTUM return-bases warmup finished: %s", summary)
        except Exception as exc:
            _ms.LOGGER.warning("XUTUM return-bases warmup failed: %s", exc)
        finally:
            with _ms._STOCK_RETURN_BASE_WARMUP_LOCK:
                _ms._STOCK_RETURN_BASE_WARMUP_LAST_FINISHED = _ms.time.time()
                _ms._STOCK_RETURN_BASE_WARMUP_RUNNING = False

    try:
        _ms._MARKET_SWR_REVALIDATION_EXECUTOR.submit(_run_warmup)
        return True
    except Exception:
        with _ms._STOCK_RETURN_BASE_WARMUP_LOCK:
            _ms._STOCK_RETURN_BASE_WARMUP_RUNNING = False
        return False


def _stock_hist_multiples_disk_path() -> Path:
    _ms = _market_service()
    return _ms.CONFIG.paths.processed_dir / "_shared" / "stock_historical_averages.json"


def _fetch_stock_weekly_5y_spark_batch(symbols: List[str]) -> Dict[str, List[tuple[str, float]]]:
    """Fetch multi-symbol 5Y weekly split-adjusted closes in one Yahoo v7 spark call (~20 tickers/req)."""
    _ms = _market_service()
    import urllib.error
    import urllib.parse
    import urllib.request

    cleaned = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
    if not cleaned:
        return {}

    yahoo_to_ticker = {f"{ticker}.IS": ticker for ticker in cleaned}
    encoded_syms = urllib.parse.quote(",".join(yahoo_to_ticker.keys()), safe=",")
    url = f"https://query1.finance.yahoo.com/v7/finance/spark?symbols={encoded_syms}&range=5y&interval=1wk"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = _ms.json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception):
        return {}

    items = ((data.get("spark") or {}).get("result") or []) if isinstance(data, dict) else []
    parsed: Dict[str, List[tuple[str, float]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_sym = str(item.get("symbol") or "").strip().upper()
        ticker = yahoo_to_ticker.get(raw_sym) or raw_sym.removesuffix(".IS")
        responses = item.get("response") or []
        if not ticker or not responses or not isinstance(responses[0], dict):
            continue
        r0 = responses[0]
        timestamps = r0.get("timestamp") or []
        quote = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        by_date: Dict[str, float] = {}
        for ts, close in zip(timestamps, closes):
            close_val = _ms._as_finite_float(close)
            if close_val is None or close_val <= 0 or not isinstance(ts, (int, float)):
                continue
            try:
                day_iso = _ms.datetime.fromtimestamp(float(ts), tz=_ms.timezone.utc).date().isoformat()
                by_date[day_iso] = close_val
            except Exception:
                continue
        if by_date:
            parsed[ticker] = sorted(by_date.items(), key=lambda entry: entry[0])
    return parsed


def _parse_split_events(split_events: Optional[List[Dict[str, Any]]]) -> List[tuple[date, float]]:
    _ms = _market_service()
    parsed: List[tuple[date, float]] = []
    for event in split_events or []:
        if not isinstance(event, dict):
            continue
        try:
            event_dt = _ms.date.fromisoformat(str(event.get("date") or "")[:10])
        except ValueError:
            continue
        ratio = _ms._as_finite_float(event.get("ratio"))
        if ratio is not None and ratio > 0:
            parsed.append((event_dt, ratio))
    return parsed


def _has_unmatched_capital_jump(
    reporting_quarters: List[Dict[str, Any]],
    parsed_splits: List[tuple[date, float]],
    *,
    quarter_end_fn: Any,
) -> bool:
    """Detect >=1.5x consecutive-quarter capital jumps lacking a nearby Yahoo split event."""
    _ms = _market_service()
    prev_shares: Optional[float] = None
    prev_end: Optional[date] = None
    for quarter in reporting_quarters:
        q_shares = _ms._extract_quarter_metric(
            quarter,
            "odenmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        ) or _ms._extract_quarter_metric(
            quarter,
            "cikarilmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        )
        q_end = quarter_end_fn(quarter)
        if q_shares is None or q_shares <= 0:
            continue
        if prev_shares is not None and prev_shares > 0:
            jump_ratio = float(q_shares) / float(prev_shares)
            if jump_ratio >= 1.5 or jump_ratio <= (1.0 / 1.5):
                if prev_end is not None and q_end is not None:
                    win_start = prev_end - _ms.timedelta(days=45)
                    win_end = q_end + _ms.timedelta(days=60)
                    matched = any(win_start <= ev_dt <= win_end for ev_dt, _ in parsed_splits)
                elif q_end is not None:
                    win_start = q_end - _ms.timedelta(days=135)
                    win_end = q_end + _ms.timedelta(days=60)
                    matched = any(win_start <= ev_dt <= win_end for ev_dt, _ in parsed_splits)
                else:
                    matched = bool(parsed_splits)
                if not matched:
                    return True
        prev_shares = float(q_shares)
        prev_end = q_end
    return False


def _compute_kap_historical_averages_for_snapshot(
    snapshot: Dict[str, Any],
    price_points: List[tuple[str, float]],
    *,
    current_shares: Optional[float] = None,
    split_events: Optional[List[Dict[str, Any]]] = None,
    on_missing_split_fallback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute historical valuation multiple averages from a KAP snapshot + weekly prices."""
    _ms = _market_service()
    from app.kap_api_service import (
        _build_kap_historical_ttm,
        _kap_history_trading_date,
        _kap_quarter_end_date,
    )

    if not isinstance(snapshot, dict) or not price_points:
        return {}
    quarters_raw = snapshot.get("quarters")
    if not isinstance(quarters_raw, list) or not quarters_raw:
        return {}

    quarters = sorted(
        [quarter for quarter in quarters_raw if isinstance(quarter, dict)],
        key=_ms._quarter_sort_key,
    )
    if not quarters:
        return {}

    reporting_quarters = [
        {
            **quarter,
            "metrics": quarter.get("metrics_original") or quarter.get("metrics") or {},
            "metrics_ytd": quarter.get("metrics_ytd_original") or quarter.get("metrics_ytd") or {},
            "metrics_quarterly": (
                quarter.get("metrics_quarterly_original")
                or quarter.get("metrics_quarterly")
                or {}
            ),
            "metrics_comparative": quarter.get("metrics_comparative") or {},
            "metrics_ytd_comparative": quarter.get("metrics_ytd_comparative") or {},
        }
        for quarter in quarters
    ]

    cur_shares = _ms._as_finite_float(current_shares)
    if cur_shares is None or cur_shares <= 0:
        for quarter in reversed(reporting_quarters):
            candidate = _ms._extract_quarter_metric(
                quarter,
                "odenmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            ) or _ms._extract_quarter_metric(
                quarter,
                "cikarilmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            )
            if candidate is not None and candidate > 0:
                cur_shares = float(candidate)
                break
    if cur_shares is None or cur_shares <= 0:
        return {}

    kind = str(snapshot.get("company_kind") or "generic").lower()
    if kind == "generic":
        from src.kap_fetcher import classify_kap_company_kind

        kind = classify_kap_company_kind(
            str(snapshot.get("stock_code") or snapshot.get("company") or ""),
            snapshot.get("company_title"),
        )
    financial_company = kind in {"bank", "insurance"}

    ordered_prices = sorted(price_points, key=lambda item: item[0])
    price_dates = [item[0] for item in ordered_prices]
    parsed_splits = _ms._parse_split_events(split_events) if split_events is not None else []
    use_split_adjusted_shares = split_events is not None
    if use_split_adjusted_shares and _ms._has_unmatched_capital_jump(
        reporting_quarters,
        parsed_splits,
        quarter_end_fn=_kap_quarter_end_date,
    ):
        # Fall back to current shares when KAP reports a >=1.5x capital jump
        # without a matching Yahoo split event, avoiding split-adjusted price vs raw share mismatch.
        use_split_adjusted_shares = False
        if callable(on_missing_split_fallback):
            on_missing_split_fallback()
    fk_vals: List[float] = []
    pd_dd_vals: List[float] = []
    fd_favok_vals: List[float] = []

    for index, quarter in enumerate(reporting_quarters):
        report_date = _kap_history_trading_date(quarter.get("publish_date"))
        if report_date is None:
            continue
        pos = _ms.bisect.bisect_left(price_dates, report_date.isoformat())
        if pos >= len(ordered_prices):
            continue
        try:
            point_dt = _ms.date.fromisoformat(price_dates[pos])
        except ValueError:
            continue
        if (point_dt - report_date).days > 10:
            continue
        price = _ms._as_finite_float(ordered_prices[pos][1])
        if price is None or price <= 0:
            continue

        quarter_shares = cur_shares
        if use_split_adjusted_shares:
            raw_quarter_shares = _ms._extract_quarter_metric(
                quarter,
                "odenmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            ) or _ms._extract_quarter_metric(
                quarter,
                "cikarilmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            )
            if raw_quarter_shares is not None and raw_quarter_shares > 0:
                quarter_end = _kap_quarter_end_date(quarter)
                split_factor = 1.0
                if quarter_end is not None:
                    for event_dt, ratio in parsed_splits:
                        if event_dt > quarter_end:
                            split_factor *= ratio
                quarter_shares = float(raw_quarter_shares) * split_factor

        market_cap = price * quarter_shares
        ttm_net_kar = _build_kap_historical_ttm(reporting_quarters, index, "net_kar")
        equity = _ms._extract_quarter_metric(quarter, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
        if ttm_net_kar is not None and ttm_net_kar > 0:
            ratio = _ms._round_market_ratio(market_cap, ttm_net_kar)
            if ratio is not None and ratio > 0:
                fk_vals.append(ratio)
        if equity is not None and equity > 0:
            ratio = _ms._round_market_ratio(market_cap, equity)
            if ratio is not None and ratio > 0:
                pd_dd_vals.append(ratio)
        if not financial_company:
            ttm_favok = _build_kap_historical_ttm(reporting_quarters, index, "favok")
            net_debt = _ms._extract_quarter_metric(quarter, "net_borc", priority=["metrics", "metrics_ytd"])
            if (
                ttm_favok is not None
                and ttm_favok > 0
                and net_debt is not None
                and _ms.math.isfinite(net_debt)
            ):
                ratio = _ms._round_market_ratio(market_cap + net_debt, ttm_favok)
                if ratio is not None and ratio > 0:
                    fd_favok_vals.append(ratio)

    def _avg_positive(vals: List[float]) -> Optional[float]:
        if len(vals) < 2:
            return None
        return round(sum(vals) / len(vals), 2)

    hist_fk_avg = _avg_positive(fk_vals)
    hist_pd_dd_avg = _avg_positive(pd_dd_vals)
    hist_fd_favok_avg = None if financial_company else _avg_positive(fd_favok_vals)
    hist_fk_quarters = len(fk_vals) if hist_fk_avg is not None else None
    hist_pd_dd_quarters = len(pd_dd_vals) if hist_pd_dd_avg is not None else None
    hist_fd_favok_quarters = len(fd_favok_vals) if hist_fd_favok_avg is not None else None
    hist_quarters = max(len(fk_vals), len(pd_dd_vals), len(fd_favok_vals), 0)
    if hist_fk_avg is None and hist_pd_dd_avg is None and hist_fd_favok_avg is None:
        return {}
    return {
        "hist_fk_avg": hist_fk_avg,
        "hist_pd_dd_avg": hist_pd_dd_avg,
        "hist_fd_favok_avg": hist_fd_favok_avg,
        "hist_quarters": hist_quarters,
        "hist_fk_quarters": hist_fk_quarters,
        "hist_pd_dd_quarters": hist_pd_dd_quarters,
        "hist_fd_favok_quarters": hist_fd_favok_quarters,
    }


def _persist_stock_hist_multiples_disk(items: Dict[str, Dict[str, Any]], *, now_ts: Optional[float] = None) -> None:
    """Persist compact historical multiple averages locally and to _shared/ Supabase cache."""
    _ms = _market_service()
    try:
        path = _ms._stock_hist_multiples_disk_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _ms._STOCK_HIST_MULTIPLES_CACHE_VERSION,
            "updated_at": _ms.time.time() if now_ts is None else now_ts,
            "items": items,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(_ms.json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(path)
        if _ms._market_database.database_enabled():
            _ms._market_database.write_json_cache(path, payload)
    except Exception as exc:
        _ms.LOGGER.warning("stock historical multiples disk persist skipped: %s", exc)


def _hydrate_stock_hist_multiples_from_disk() -> int:
    _ms = _market_service()
    path = _ms._stock_hist_multiples_disk_path()
    if not path.exists():
        return 0
    try:
        raw = _ms.json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(raw, dict):
        return 0
    version = raw.get("version")
    if version not in {"v1", _ms._STOCK_HIST_MULTIPLES_CACHE_VERSION}:
        return 0
    items = raw.get("items")
    if not isinstance(items, dict) or not items:
        return 0
    updated_at = float(raw.get("updated_at") or 0.0)
    if updated_at > 0 and (_ms.time.time() - updated_at) > max(3600, int(_ms._STOCK_HIST_MULTIPLES_STALE_TTL)):
        return 0
    clean: Dict[str, Dict[str, Any]] = {}
    for sym, entry in items.items():
        ticker = str(sym or "").strip().upper()
        if ticker and isinstance(entry, dict):
            clean[ticker] = dict(entry)
    if not clean:
        return 0
    hydrated_ts = updated_at if version == _ms._STOCK_HIST_MULTIPLES_CACHE_VERSION else 0.0
    with _ms._STOCK_HIST_MULTIPLES_LOCK:
        _ms._STOCK_HIST_MULTIPLES_STATE["_ts"] = hydrated_ts
        _ms._STOCK_HIST_MULTIPLES_STATE["items"] = clean
    return len(clean)


def _cached_stock_hist_multiples_map() -> Dict[str, Dict[str, Any]]:
    _ms = _market_service()
    with _ms._STOCK_HIST_MULTIPLES_LOCK:
        existing = _ms._STOCK_HIST_MULTIPLES_STATE.get("items")
        if isinstance(existing, dict) and existing:
            return dict(existing)
    if not _ms.os.getenv("PYTEST_CURRENT_TEST"):
        _ms._hydrate_stock_hist_multiples_from_disk()
    with _ms._STOCK_HIST_MULTIPLES_LOCK:
        items = _ms._STOCK_HIST_MULTIPLES_STATE.get("items")
        return dict(items) if isinstance(items, dict) else {}


def _snapshot_has_capital_changes(snapshot: Dict[str, Any]) -> bool:
    _ms = _market_service()
    capitals: set[float] = set()
    for quarter in snapshot.get("quarters") or []:
        if not isinstance(quarter, dict):
            continue
        q_metrics = {
            **quarter,
            "metrics": quarter.get("metrics_original") or quarter.get("metrics") or {},
            "metrics_ytd": quarter.get("metrics_ytd_original") or quarter.get("metrics_ytd") or {},
        }
        val = _ms._extract_quarter_metric(
            q_metrics,
            "odenmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        ) or _ms._extract_quarter_metric(
            q_metrics,
            "cikarilmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        )
        if val is not None and val > 0:
            capitals.add(round(float(val), 4))
            if len(capitals) > 1:
                return True
    return False


def _fetch_stock_split_events_cached(
    symbol: str,
    *,
    start_date: date,
    end_date: date,
) -> Optional[List[Dict[str, Any]]]:
    _ms = _market_service()
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return None
    cache_key = f"api:stock:splits:v1:{normalized}"
    cached = _ms._shared_cache_get_dict(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("splits"), list):
        return cached["splits"]
    chart = _ms._fetch_yahoo_chart_period_raw(
        f"{normalized}.IS",
        interval="1mo",
        start_date=start_date,
        end_date=end_date,
    )
    if not chart.get("ok"):
        return None
    splits = chart.get("splits") if isinstance(chart.get("splits"), list) else []
    split_ttl = max(3600, int(_ms._STOCK_HIST_MULTIPLES_TTL)) if splits else 3600
    _ms._shared_cache_set(cache_key, {"splits": splits}, split_ttl)
    return splits


def _warm_stock_hist_multiples_sync(
    symbols: Optional[List[str]] = None,
    *,
    financial_snapshot_map: Optional[Dict[str, Dict[str, Any]]] = None,
    basic_summary_map: Optional[Dict[str, Dict[str, Any]]] = None,
    only_missing: bool = False,
) -> Dict[str, int]:
    """Compute historical valuation averages from kap_cache JSONs (local + Supabase) + batched 5Y weekly spark."""
    _ms = _market_service()
    from concurrent.futures import ThreadPoolExecutor

    kap_cache_dir = _ms.CONFIG.paths.processed_dir / "kap_cache"
    kap_cache_dir.mkdir(parents=True, exist_ok=True)

    target_symbols = (
        [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
        if symbols is not None
        else _ms._market_stock_symbols_for_index("XUTUM")
    )
    skipped_existing = 0
    if only_missing:
        existing_hist = _ms._cached_stock_hist_multiples_map()
        skipped_existing = sum(1 for sym in target_symbols if sym in existing_hist)
        target_symbols = [sym for sym in target_symbols if sym not in existing_hist]
        if not target_symbols:
            return {"requested": 0, "populated": 0, "skipped_existing": skipped_existing}
    local_kap_set = {sym for sym in target_symbols if (kap_cache_dir / f"{sym}.json").exists()}
    missing_local_symbols = [sym for sym in target_symbols if sym not in local_kap_set]
    remote_kap_set: Set[str] = set()
    if missing_local_symbols and _ms._market_database.database_enabled():
        try:
            remote_meta = _ms._market_database.read_json_cache_kap_metadata()
            if isinstance(remote_meta, dict):
                for key in remote_meta.keys():
                    # read_json_cache_kap_metadata() returns ticker stems (e.g. "THYAO"),
                    # though paths like "kap_cache/THYAO.json" are also accepted safely.
                    name = str(key).rsplit("/", 1)[-1].strip()
                    sym_name = name[:-5].strip().upper() if name.lower().endswith(".json") else name.upper()
                    if sym_name:
                        remote_kap_set.add(sym_name)
        except Exception:
            _ms.LOGGER.debug("read_json_cache_kap_metadata failed during historical multiples warmup", exc_info=True)

    available_kap_set = local_kap_set | remote_kap_set
    kap_symbols = [sym for sym in target_symbols if sym in available_kap_set]
    if not kap_symbols:
        out: Dict[str, int] = {"requested": 0, "populated": 0}
        if only_missing:
            out["skipped_existing"] = skipped_existing
        return out

    batches = [
        kap_symbols[i : i + _ms._STOCK_RETURN_BASE_SPARK_BATCH_SIZE]
        for i in range(0, len(kap_symbols), _ms._STOCK_RETURN_BASE_SPARK_BATCH_SIZE)
    ]
    price_map: Dict[str, List[tuple[str, float]]] = {}
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            for batch_res in pool.map(_ms._fetch_stock_weekly_5y_spark_batch, batches):
                price_map.update(batch_res)
    except Exception:
        for batch in batches:
            price_map.update(_ms._fetch_stock_weekly_5y_spark_batch(batch))

    if not price_map:
        out = {"requested": len(kap_symbols), "populated": 0}
        if only_missing:
            out["skipped_existing"] = skipped_existing
        return out

    if financial_snapshot_map is None:
        try:
            financial_snapshot_map = _ms.get_financial_snapshots_map(_ms.CONFIG.paths.processed_dir)
        except Exception:
            financial_snapshot_map = {}
    if basic_summary_map is None:
        try:
            basic_summary_map = _ms._fetch_isyatirim_basic_summary_map()
        except Exception:
            basic_summary_map = {}

    fin_map = financial_snapshot_map or {}
    bs_map = basic_summary_map or {}
    computed: Dict[str, Dict[str, Any]] = {}
    missing_split_fallback_syms: List[str] = []

    def _compute_for_symbol(
        sym: str,
        raw_kap: Dict[str, Any],
        sym_splits: Optional[List[Dict[str, Any]]],
    ) -> None:
        pts = price_map.get(sym) or []
        provider_shares = _ms._as_finite_float((bs_map.get(sym) or {}).get("shares_outstanding"))
        meta_shares = _ms._as_finite_float((fin_map.get(sym) or {}).get("shares_outstanding"))
        cur_shares = provider_shares if (provider_shares and provider_shares > 0) else meta_shares
        summary = _ms._compute_kap_historical_averages_for_snapshot(
            raw_kap,
            pts,
            current_shares=cur_shares,
            split_events=sym_splits,
            on_missing_split_fallback=lambda s=sym: missing_split_fallback_syms.append(s),
        )
        if summary:
            computed[sym] = summary

    from app.json_cache import read_json_cache_local_first

    capital_changed_snapshots: Dict[str, Dict[str, Any]] = {}
    for sym in kap_symbols:
        if not price_map.get(sym):
            continue
        cache_path = kap_cache_dir / f"{sym}.json"
        raw_kap = None
        if cache_path.exists():
            try:
                raw_kap = _ms.json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                raw_kap = None
        if raw_kap is None and sym in remote_kap_set:
            try:
                raw_kap = read_json_cache_local_first(cache_path)
            except Exception:
                raw_kap = None
        if not isinstance(raw_kap, dict):
            continue
        if _ms._snapshot_has_capital_changes(raw_kap):
            capital_changed_snapshots[sym] = raw_kap
        else:
            # Constant-capital snapshots pass [] directly (zero HTTP calls) and release immediately.
            _compute_for_symbol(sym, raw_kap, [])

    today = _ms.date.today()
    splits_start = today - _ms.timedelta(days=6 * 366)
    splits_by_sym: Dict[str, Optional[List[Dict[str, Any]]]] = {}
    capital_changed_syms = list(capital_changed_snapshots.keys())
    if capital_changed_syms:
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(capital_changed_syms))) as pool:
                results = pool.map(
                    lambda s: _ms._fetch_stock_split_events_cached(s, start_date=splits_start, end_date=today),
                    capital_changed_syms,
                )
                for sym, splits in zip(capital_changed_syms, results):
                    splits_by_sym[sym] = splits
        except Exception:
            for sym in capital_changed_syms:
                splits_by_sym[sym] = _ms._fetch_stock_split_events_cached(
                    sym, start_date=splits_start, end_date=today
                )

    for sym, raw_kap in capital_changed_snapshots.items():
        # Changed-capital snapshots pass fetched split events, or None to fall back safely if Yahoo failed.
        sym_splits = splits_by_sym.get(sym) if sym in splits_by_sym else []
        _compute_for_symbol(sym, raw_kap, sym_splits)
    capital_changed_snapshots.clear()

    if missing_split_fallback_syms:
        _ms.LOGGER.warning(
            "stock historical multiples fell back to current_shares due to unmatched capital jumps for %d symbols (%s)",
            len(missing_split_fallback_syms),
            ", ".join(missing_split_fallback_syms[:10]),
        )

    now = _ms.time.time()
    if computed:
        with _ms._STOCK_HIST_MULTIPLES_LOCK:
            merged = dict(_ms._STOCK_HIST_MULTIPLES_STATE.get("items") or {})
            merged.update(computed)
            _ms._STOCK_HIST_MULTIPLES_STATE["_ts"] = now
            _ms._STOCK_HIST_MULTIPLES_STATE["items"] = merged
        if not _ms.os.getenv("PYTEST_CURRENT_TEST"):
            _ms._persist_stock_hist_multiples_disk(merged, now_ts=now)
        _ms._refresh_cached_xutum_stocks_payload()
    out = {"requested": len(kap_symbols), "populated": len(computed)}
    if only_missing:
        out["skipped_existing"] = skipped_existing
    return out


def _schedule_stock_hist_multiples_warmup_if_needed(
    symbols: Optional[List[str]] = None,
    *,
    financial_snapshot_map: Optional[Dict[str, Dict[str, Any]]] = None,
    basic_summary_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Kick off background warmup for historical valuation averages if cold/expired."""
    _ms = _market_service()

    if _ms.os.getenv("PYTEST_CURRENT_TEST"):
        return False

    now = _ms.time.time()
    with _ms._STOCK_HIST_MULTIPLES_LOCK:
        ts = float(_ms._STOCK_HIST_MULTIPLES_STATE.get("_ts") or 0.0)
        items = _ms._STOCK_HIST_MULTIPLES_STATE.get("items") or {}
        if items and (now - ts) < max(3600, int(_ms._STOCK_HIST_MULTIPLES_TTL)):
            return False
        if _ms._STOCK_HIST_MULTIPLES_WARMUP_RUNNING:
            return False
        if (now - _ms._STOCK_HIST_MULTIPLES_WARMUP_LAST_FINISHED) < _ms._STOCK_HIST_MULTIPLES_WARMUP_COOLDOWN_SECONDS:
            return False
        _ms._STOCK_HIST_MULTIPLES_WARMUP_RUNNING = True

    def _run_warmup() -> None:
        try:
            summary = _ms._warm_stock_hist_multiples_sync(
                symbols,
                financial_snapshot_map=financial_snapshot_map,
                basic_summary_map=basic_summary_map,
            )
            _ms.LOGGER.info("Stock historical valuation multiples warmup finished: %s", summary)
        except Exception as exc:
            _ms.LOGGER.warning("Stock historical valuation multiples warmup failed: %s", exc)
        finally:
            with _ms._STOCK_HIST_MULTIPLES_LOCK:
                _ms._STOCK_HIST_MULTIPLES_WARMUP_LAST_FINISHED = _ms.time.time()
                _ms._STOCK_HIST_MULTIPLES_WARMUP_RUNNING = False

    try:
        _ms._MARKET_SWR_REVALIDATION_EXECUTOR.submit(_run_warmup)
        return True
    except Exception:
        with _ms._STOCK_HIST_MULTIPLES_LOCK:
            _ms._STOCK_HIST_MULTIPLES_WARMUP_RUNNING = False
        return False


def _bulk_return_loaded_key() -> str:
    _ms = _market_service()
    return f"__bulk_loaded__:{_ms.cache_reset_epoch()}:{id(_ms._get_cache())}"


def _warm_bulk_return_bases(
    local_cache: BoundedMemoryCache,
    shared_key: str,
    ttl_seconds: int,
) -> None:
    _ms = _market_service()
    loaded_key = _ms._bulk_return_loaded_key()
    if local_cache.get(loaded_key):
        return
    with _ms._BULK_RETURN_BASES_LOCK:
        if local_cache.get(loaded_key):
            return
        now = _ms.time.time()
        shared_map = _ms._shared_cache_get_dict(shared_key)
        if isinstance(shared_map, dict):
            for item_key, entry in shared_map.items():
                if not isinstance(entry, dict):
                    continue
                ts = entry.get("_ts")
                data = entry.get("data")
                if isinstance(ts, (int, float)) and isinstance(data, dict) and now - float(ts) < ttl_seconds:
                    local_cache[str(item_key)] = {"_ts": float(ts), "data": dict(data)}
        local_cache[loaded_key] = now


def _flush_bulk_return_bases(
    local_cache: BoundedMemoryCache,
    shared_key: str,
    ttl_seconds: int,
) -> None:
    _ms = _market_service()
    with _ms._BULK_RETURN_BASES_LOCK:
        now = _ms.time.time()
        payload: Dict[str, Any] = {}
        existing = _ms._shared_cache_get_dict(shared_key)
        if isinstance(existing, dict):
            for item_key, entry in existing.items():
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("_ts"), (int, float))
                    and isinstance(entry.get("data"), dict)
                    and now - float(entry["_ts"]) < ttl_seconds
                ):
                    payload[str(item_key)] = entry
        changed = False
        for item_key in list(local_cache.keys()):
            if str(item_key).startswith("__bulk_loaded__:"):
                continue
            entry = local_cache.get(item_key)
            if not isinstance(entry, dict):
                continue
            ts = entry.get("_ts")
            data = entry.get("data")
            if isinstance(ts, (int, float)) and isinstance(data, dict) and now - float(ts) < ttl_seconds:
                prev = payload.get(str(item_key))
                if not isinstance(prev, dict) or float(prev.get("_ts", 0)) < float(ts):
                    payload[str(item_key)] = {"_ts": float(ts), "data": dict(data)}
                    changed = True
        if changed and payload:
            _ms._shared_cache_set(shared_key, payload, ttl_seconds=ttl_seconds)


def _begin_bulk_return_batch(local_cache: BoundedMemoryCache, shared_key: str, ttl_seconds: int) -> None:
    _ms = _market_service()
    _ms._warm_bulk_return_bases(local_cache, shared_key, ttl_seconds)
    with _ms._BULK_RETURN_BASES_LOCK:
        _ms._BULK_RETURN_BATCH_DEPTH[shared_key] = _ms._BULK_RETURN_BATCH_DEPTH.get(shared_key, 0) + 1


def _end_bulk_return_batch(local_cache: BoundedMemoryCache, shared_key: str, ttl_seconds: int) -> None:
    _ms = _market_service()
    should_flush = False
    with _ms._BULK_RETURN_BASES_LOCK:
        next_depth = max(0, _ms._BULK_RETURN_BATCH_DEPTH.get(shared_key, 0) - 1)
        if next_depth == 0:
            _ms._BULK_RETURN_BATCH_DEPTH.pop(shared_key, None)
            should_flush = bool(_ms._BULK_RETURN_DIRTY.pop(shared_key, False))
        else:
            _ms._BULK_RETURN_BATCH_DEPTH[shared_key] = next_depth
    if should_flush:
        _ms._flush_bulk_return_bases(local_cache, shared_key, ttl_seconds)


def _store_bulk_return_base_entry(
    local_cache: BoundedMemoryCache,
    shared_key: str,
    item_key: str,
    data: Dict[str, Any],
    ttl_seconds: int,
    now: float,
) -> Dict[str, Any]:
    _ms = _market_service()
    local_cache[item_key] = {"_ts": now, "data": dict(data)}
    flush_now = False
    with _ms._BULK_RETURN_BASES_LOCK:
        if _ms._BULK_RETURN_BATCH_DEPTH.get(shared_key, 0) > 0:
            _ms._BULK_RETURN_DIRTY[shared_key] = True
        else:
            flush_now = True
    if flush_now:
        _ms._flush_bulk_return_bases(local_cache, shared_key, ttl_seconds)
    return dict(data)
