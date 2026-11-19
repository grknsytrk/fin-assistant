from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.market import _market_service


def _supported_stock_indexes_text() -> str:
    _ms = _market_service()
    return ", ".join(_ms._MARKET_STOCK_INDEX_ORDER)


def _supported_market_indexes_text() -> str:
    _ms = _market_service()
    return ", ".join(_ms._MARKET_INDEX_ORDER)


def _normalize_stock_index(index_name: str) -> str:
    _ms = _market_service()
    normalized = str(index_name or "XUTUM").strip().upper()
    if normalized not in _ms._MARKET_STOCK_INDEXES:
        raise _ms.HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen endeks. {_ms._supported_stock_indexes_text()} kullanin.",
        )
    return normalized


def _market_stock_benchmarks() -> Dict[str, Dict[str, Any]]:
    _ms = _market_service()
    base_map = _ms._fetch_stock_return_bases_bulk(_ms._MARKET_STOCK_INDEX_ORDER)
    benchmarks: Dict[str, Dict[str, Any]] = {}
    for index_name in _ms._MARKET_STOCK_INDEX_ORDER:
        bases = base_map.get(index_name, {})
        try:
            live_quote = _ms._fetch_index_quote(index_name)
        except Exception as exc:
            _ms.LOGGER.debug("market stock benchmark live quote lookup failed for %s: %s", index_name, exc)
            live_quote = {}

        live_price_raw = live_quote.get("price") if isinstance(live_quote, dict) else None
        live_price: Optional[float] = None
        if live_price_raw is not None and not isinstance(live_price_raw, bool):
            try:
                candidate = float(live_price_raw)
                if _ms.math.isfinite(candidate) and candidate > 0:
                    live_price = candidate
            except (TypeError, ValueError):
                live_price = None

        if live_price is not None:
            current_for_returns = live_price
        else:
            current_for_returns = bases.get("latest_close")

        price_val: Optional[float] = None
        if current_for_returns is not None and not isinstance(current_for_returns, bool):
            try:
                numeric_price = float(current_for_returns)
                if _ms.math.isfinite(numeric_price):
                    price_val = numeric_price
            except (TypeError, ValueError):
                price_val = None

        prev_close_val: Optional[float] = None
        raw_prev = live_quote.get("prev_close") if isinstance(live_quote, dict) else None
        if raw_prev is None:
            raw_prev = bases.get("latest_close")
        if raw_prev is not None and not isinstance(raw_prev, bool):
            try:
                numeric_prev = float(raw_prev)
                if _ms.math.isfinite(numeric_prev):
                    prev_close_val = numeric_prev
            except (TypeError, ValueError):
                prev_close_val = None

        change_val: Optional[float] = None
        raw_change = live_quote.get("change") if isinstance(live_quote, dict) else None
        if raw_change is not None and not isinstance(raw_change, bool):
            try:
                numeric_change = float(raw_change)
                if _ms.math.isfinite(numeric_change):
                    change_val = numeric_change
            except (TypeError, ValueError):
                change_val = None

        change_pct_val: Optional[float] = None
        raw_change_pct = live_quote.get("change_pct") if isinstance(live_quote, dict) else None
        if raw_change_pct is not None and not isinstance(raw_change_pct, bool):
            try:
                numeric_change_pct = float(raw_change_pct)
                if _ms.math.isfinite(numeric_change_pct):
                    change_pct_val = numeric_change_pct
            except (TypeError, ValueError):
                change_pct_val = None

        if live_price is not None and prev_close_val is not None:
            if change_val is None:
                change_val = round(live_price - prev_close_val, 2)
            if change_pct_val is None and prev_close_val > 0:
                change_pct_val = round(((live_price - prev_close_val) / prev_close_val) * 100, 2)

        as_of_val = (
            (live_quote.get("as_of") if isinstance(live_quote, dict) else None)
            or bases.get("as_of")
        )

        benchmarks[index_name] = {
            "price": price_val,
            "prev_close": prev_close_val,
            "change": change_val,
            "change_pct": change_pct_val,
            "as_of": as_of_val,
            **_ms._returns_from_bases(current_for_returns, bases),
        }
    return benchmarks


def _market_stock_symbols_for_index(index_name: str) -> List[str]:
    _ms = _market_service()
    from app.kap_service import get_bist100_companies, get_bist30_companies, get_bist_all_companies

    normalized = _ms._normalize_stock_index(index_name)
    if normalized == "XUTUM":
        return get_bist_all_companies()
    if normalized == "XU100":
        return get_bist100_companies()
    return get_bist30_companies()


def _refresh_cached_xutum_stocks_payload() -> None:
    """Update cached XUTUM stocks payload once return bases finish warming."""
    _ms = _market_service()
    local_key = "payload:XUTUM"
    shared_key = "api:market:stocks:XUTUM:v4"
    if _ms._STOCKS_CACHE.get(local_key) is None and _ms._shared_cache_get_dict(shared_key) is None:
        return
    try:
        built = _ms._build_market_stocks_payload(index_name="XUTUM", force_refresh=False)
        quote_stale = bool(built.pop("quote_stale", False))
        if not _ms._market_payload_has_quote(built):
            return
        now = _ms.time.time()
        effective_fresh_ttl = _ms._effective_market_quotes_fresh_ttl()
        effective_stale_ttl = max(effective_fresh_ttl, _ms._MARKET_QUOTES_STALE_TTL)
        entry = {
            "payload": built,
            "quote_stale": quote_stale,
            "fresh_until": now + max(1, effective_fresh_ttl),
            "stale_until": now + max(1, effective_stale_ttl),
            "cached_at": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }
        _ms._STOCKS_CACHE[local_key] = {"_ts": now, "entry": entry}
        _ms._shared_cache_set(shared_key, entry, max(1, effective_stale_ttl))
    except Exception as exc:
        _ms.LOGGER.debug("refreshing cached XUTUM payload after return warmup failed: %s", exc)


def _market_stock_row(
    symbol: str,
    *,
    cached_meta: Dict[str, Any],
    quote: Dict[str, Any],
    return_bases: Dict[str, Any],
    basic_summary: Optional[Dict[str, Any]] = None,
    historical_multiples: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ms = _market_service()
    latest_quarter = cached_meta.get("latest_quarter")
    current_for_returns = quote.get("price") if quote.get("price") is not None else return_bases.get("latest_close")
    market_cap = _ms._market_cap_from_quote_and_meta(quote, cached_meta, basic_summary)
    provider = basic_summary or {}
    company_kind = str(cached_meta.get("company_kind") or "generic").lower()
    if company_kind == "generic":
        from src.kap_fetcher import classify_kap_company_kind

        company_kind = classify_kap_company_kind(symbol, cached_meta.get("company_title"))
    financial_snapshot = {**cached_meta, "company_kind": company_kind}
    valuation = _ms._resolve_market_card_valuation_from_cached_data(
        {
            "financial_snapshot": financial_snapshot,
            "provider_ratios": {
                "fk": provider.get("fk"),
                "pd_dd": provider.get("pd_dd"),
                "fd_favok": provider.get("fd_favok"),
            },
            "provider_source": provider.get("source") or "isyatirim_temelfinansal",
            "as_of": cached_meta.get("as_of") or provider.get("as_of"),
        },
        market_cap=market_cap,
    )
    ttm_sales = _ms._as_finite_float(cached_meta.get("ttm_satis_gelirleri"))
    fd_satis = (
        _ms._round_market_ratio(valuation.get("enterprise_value"), ttm_sales)
        if company_kind not in {"bank", "insurance"}
        else None
    )
    if _ms._is_missing_market_ratio(fd_satis) and company_kind not in {"bank", "insurance"}:
        provider_fd_satis = _ms._parse_tr_decimal(provider.get("fd_satis"))
        if not _ms._is_missing_market_ratio(provider_fd_satis):
            fd_satis = provider_fd_satis
    free_float_pct = _ms._as_finite_float(cached_meta.get("free_float_pct"))
    if free_float_pct is None:
        free_float_pct = _ms._as_finite_float(provider.get("free_float_pct"))
    if free_float_pct is None:
        fdpo = _ms._as_finite_float(provider.get("fdpo"))
        free_float_pct = fdpo * 100.0 if fdpo is not None else None
    net_borc_favok = _ms._first_not_none(
        valuation.get("net_borc_favok"),
        _ms._as_finite_float(cached_meta.get("net_borc_favok")),
    )
    ttm_favok = _ms._as_finite_float(cached_meta.get("ttm_favok"))
    if company_kind in {"bank", "insurance"} or (ttm_favok is not None and ttm_favok <= 0):
        # Leverage is not meaningful for financial companies or when EBITDA is
        # zero or negative.
        net_borc_favok = None
    hist = historical_multiples or {}
    hist_fk_avg = _ms._as_finite_float(hist.get("hist_fk_avg"))
    hist_pd_dd_avg = _ms._as_finite_float(hist.get("hist_pd_dd_avg"))
    hist_fd_favok_avg = (
        None
        if company_kind in {"bank", "insurance"}
        else _ms._as_finite_float(hist.get("hist_fd_favok_avg"))
    )
    def _opt_int(key: str) -> Optional[int]:
        val = hist.get(key)
        return int(val) if isinstance(val, int) and not isinstance(val, bool) else None

    hist_quarters = _opt_int("hist_quarters")
    hist_fk_quarters = _opt_int("hist_fk_quarters")
    hist_pd_dd_quarters = _opt_int("hist_pd_dd_quarters")
    hist_fd_favok_quarters = (
        None if company_kind in {"bank", "insurance"} else _opt_int("hist_fd_favok_quarters")
    )
    return {
        "company": symbol,
        "name": cached_meta.get("company_title") or symbol,
        "latest_quarter": latest_quarter,
        "fundamental_period": latest_quarter or cached_meta.get("provider_period") or provider.get("provider_period"),
        "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
        "price": quote.get("price"),
        "price_currency": quote.get("currency"),
        "change": quote.get("change"),
        "change_pct": quote.get("change_pct"),
        "price_as_of": quote.get("as_of"),
        "volume": quote.get("volume"),
        "market_cap": market_cap,
        "sector": cached_meta.get("sector") or provider.get("sector"),
        "fk": valuation.get("fk"),
        "pd_dd": valuation.get("pd_dd"),
        "fd_favok": valuation.get("fd_favok"),
        "fd_satis": fd_satis,
        "net_borc_favok": net_borc_favok,
        "hist_fk_avg": hist_fk_avg,
        "hist_pd_dd_avg": hist_pd_dd_avg,
        "hist_fd_favok_avg": hist_fd_favok_avg,
        "fk_hist_diff_pct": _ms._return_pct(valuation.get("fk"), hist_fk_avg),
        "pd_dd_hist_diff_pct": _ms._return_pct(valuation.get("pd_dd"), hist_pd_dd_avg),
        "fd_favok_hist_diff_pct": _ms._return_pct(valuation.get("fd_favok"), hist_fd_favok_avg),
        "hist_quarters": hist_quarters,
        "hist_fk_quarters": hist_fk_quarters,
        "hist_pd_dd_quarters": hist_pd_dd_quarters,
        "hist_fd_favok_quarters": hist_fd_favok_quarters,
        "roe_pct": _ms._as_finite_float(cached_meta.get("roe_pct")),
        "favok_margin_pct": _ms._as_finite_float(cached_meta.get("favok_margin_pct")),
        "net_margin_pct": _ms._as_finite_float(cached_meta.get("net_margin_pct")),
        "free_float_pct": free_float_pct,
        "ttm_net_kar": _ms._as_finite_float(cached_meta.get("ttm_net_kar")),
        "ttm_satis_gelirleri": _ms._as_finite_float(cached_meta.get("ttm_satis_gelirleri")),
        "ttm_favok": ttm_favok,
        "ttm_fcf": _ms._as_finite_float(cached_meta.get("serbest_nakit_akisi")),
        "ozkaynaklar": _ms._as_finite_float(cached_meta.get("ozkaynaklar")),
        "net_borc": _ms._as_finite_float(cached_meta.get("net_borc")),
        **_ms._empty_logo_payload(),
        **_ms._returns_from_bases(current_for_returns, return_bases),
    }


def _build_market_stocks_payload(
    *,
    index_name: str = "XUTUM",
    force_refresh: bool = False,
    basic_summary_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    _ms = _market_service()
    from app.kap_service import get_bist_index_universe

    normalized_index = _ms._normalize_stock_index(index_name)
    symbols = _ms._market_stock_symbols_for_index(normalized_index)
    try:
        universe = get_bist_index_universe(normalized_index, force_refresh=force_refresh)
    except Exception:
        universe = {
            "index": normalized_index,
            "count": len(symbols),
            "source": None,
            "source_url": None,
            "source_date": None,
            "fetched_at": None,
            "cache_hit": False,
            "fallback_used": False,
        }
    price_map = _ms._fetch_market_price_map(symbols, index_name=normalized_index)
    quote_as_of = next(
        (
            quote.get("as_of")
            for quote in price_map.values()
            if isinstance(quote, dict) and quote.get("as_of")
        ),
        None,
    )
    if normalized_index == "XUTUM":
        return_base_map = _ms._cached_stock_return_bases_bulk(symbols)
        _ms._schedule_xutum_return_bases_warmup_if_needed(symbols)
    else:
        return_base_map = _ms._fetch_stock_return_bases_bulk(symbols)
    if basic_summary_map is None:
        basic_summary_map = _ms._fetch_isyatirim_basic_summary_map()
    try:
        financial_snapshot_map = _ms.get_financial_snapshots_map(_ms.CONFIG.paths.processed_dir, symbols)
    except Exception as exc:
        _ms.LOGGER.warning("market stock financial snapshot lookup failed: %s", exc)
        financial_snapshot_map = {}
    hist_multiples_map = _ms._cached_stock_hist_multiples_map()
    _ms._schedule_stock_hist_multiples_warmup_if_needed(
        symbols,
        financial_snapshot_map=financial_snapshot_map,
        basic_summary_map=basic_summary_map,
    )
    rows = [
        _ms._market_stock_row(
            symbol,
            cached_meta=financial_snapshot_map.get(str(symbol or "").strip().upper(), {}),
            quote=price_map.get(symbol, {}),
            return_bases=return_base_map.get(symbol, {}),
            basic_summary=basic_summary_map.get(symbol),
            historical_multiples=hist_multiples_map.get(str(symbol or "").strip().upper()),
        )
        for symbol in symbols
    ]
    data = {
        "index": normalized_index,
        "rows": rows,
        # Internal signal consumed by `_market_stocks_payload`, which turns it
        # into the public `quote_status`/`stale` envelope fields and removes it
        # from the response payload.
        "quote_stale": any(
            bool(quote.get("stale"))
            for quote in price_map.values()
            if isinstance(quote, dict)
        ),
        "benchmarks": _ms._market_stock_benchmarks(),
        "source": "infoyatirim_yahoo",
        "universe": {
            "index": universe.get("index") or normalized_index,
            "count": int(universe.get("count") or len(symbols)),
            "source": universe.get("source"),
            "source_url": universe.get("source_url"),
            "source_date": universe.get("source_date"),
            "fetched_at": universe.get("fetched_at"),
            "cache_hit": bool(universe.get("cache_hit")),
            "fallback_used": bool(universe.get("fallback_used")),
        },
        # The quote timestamp is the shared live snapshot timestamp used by
        # both the market list and the stock cards.
        "as_of": quote_as_of or _ms.datetime.now(_ms.timezone.utc).isoformat(),
    }
    return data


def _market_quote_entry_is_fresh(entry: Any) -> bool:
    _ms = _market_service()
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return False
    try:
        return float(entry.get("fresh_until") or 0.0) > _ms.time.time()
    except (TypeError, ValueError):
        return False


def _market_no_quote_cache_key(index_name: str) -> str:
    return f"api:market:stocks:{index_name}:no-quote:v1"


def _market_no_quote_cache_active(index_name: str) -> bool:
    _ms = _market_service()
    entry = _ms._shared_cache_get_dict(_ms._market_no_quote_cache_key(index_name))
    return isinstance(entry, dict) and entry.get("status") == "no_quote"


def _store_market_no_quote_cache(index_name: str) -> None:
    _ms = _market_service()
    _ms._shared_cache_set(
        _ms._market_no_quote_cache_key(index_name),
        {"status": "no_quote"},
        _ms._MARKET_QUOTES_NOQUOTE_TTL_SECONDS,
    )


def _clear_market_no_quote_cache(index_name: str) -> None:
    _ms = _market_service()
    _ms._get_cache().delete(_ms._market_no_quote_cache_key(index_name))


def _market_payload_has_symbol_rows(payload: Dict[str, Any]) -> bool:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return any(
        isinstance(row, dict)
        and str(row.get("company") or row.get("symbol") or "").strip()
        for row in rows
    )


def _market_payload_has_quote(payload: Dict[str, Any]) -> bool:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    if not rows:
        return True
    return any(
        isinstance(row, dict)
        and str(row.get("company") or row.get("symbol") or "").strip()
        and row.get("price") is not None
        for row in rows
    )


def _market_payload_is_no_quote(payload: Dict[str, Any]) -> bool:
    _ms = _market_service()
    return _ms._market_payload_has_symbol_rows(payload) and not _ms._market_payload_has_quote(payload)


def _market_quote_entry_is_stale(entry: Any) -> bool:
    _ms = _market_service()
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return False
    try:
        return float(entry.get("stale_until") or 0.0) > _ms.time.time()
    except (TypeError, ValueError):
        return False


def _market_quote_entry_payload(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
        return None
    return dict(entry["payload"])


def _market_unavailable_stocks_payload(index_name: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized_index = _ms._normalize_stock_index(index_name)
    universe = _ms._market_universe_payload(index_name=normalized_index)
    universe_info = universe.get("universe") if isinstance(universe.get("universe"), dict) else {}
    rows = []
    for row in universe.get("rows") if isinstance(universe.get("rows"), list) else []:
        rows.append(
            {
                **dict(row),
                "volume": None,
                "return_1w_pct": None,
                "return_1m_pct": None,
                "return_3m_pct": None,
                "return_6m_pct": None,
                "return_ytd_pct": None,
                "return_1y_pct": None,
            }
        )
    return {
        "index": normalized_index,
        "rows": rows,
        "benchmarks": _ms._market_stock_benchmarks(),
        "source": "reference_data",
        "universe": dict(universe_info),
        "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
    }


def _is_bist_market_open(now: Optional[datetime] = None) -> bool:
    """Return True during BIST equity trading + auction window (Mon-Fri 09:50-18:20 TRT)."""
    _ms = _market_service()
    if now is None:
        current = _ms.datetime.now(_ms._TURKEY_TIMEZONE)
    elif now.tzinfo is None:
        current = now.replace(tzinfo=_ms._TURKEY_TIMEZONE)
    else:
        current = now.astimezone(_ms._TURKEY_TIMEZONE)
    if current.weekday() >= 5:
        return False
    minute_of_day = (current.hour * 60) + current.minute
    return _ms._BIST_STREAM_OPEN_MINUTE <= minute_of_day <= _ms._BIST_STREAM_CLOSE_MINUTE


def _effective_market_quotes_fresh_ttl(now: Optional[datetime] = None) -> int:
    """Keep 90s quote freshness during BIST trading hours and extend to 300s off-hours."""
    _ms = _market_service()
    if _ms._is_bist_market_open(now):
        return max(1, int(_ms._MARKET_QUOTES_FRESH_TTL))
    return max(1, int(_ms._MARKET_QUOTES_FRESH_TTL), int(_ms._MARKET_QUOTES_OFFHOURS_FRESH_TTL))


def _is_global_fx_market_open(now: Optional[datetime] = None) -> bool:
    """Return True during 24/5 Global FX/Commodities/Macro window (Sun 23:55 - Sat 01:00 TRT)."""
    _ms = _market_service()
    if now is None:
        current = _ms.datetime.now(_ms._TURKEY_TIMEZONE)
    elif now.tzinfo is None:
        current = now.replace(tzinfo=_ms._TURKEY_TIMEZONE)
    else:
        current = now.astimezone(_ms._TURKEY_TIMEZONE)

    weekday = current.weekday()
    minute_of_day = (current.hour * 60) + current.minute

    # Saturday: closed from 01:00 TRT (minute 60) onwards
    if weekday == 5 and minute_of_day >= 60:
        return False
    # Sunday: closed until 23:55 TRT (minute 1435)
    if weekday == 6 and minute_of_day < (23 * 60 + 55):
        return False
    return True


def _effective_macro_fresh_ttl(now: Optional[datetime] = None) -> int:
    """Return 30s during active trading hours and 300s during off-hours/weekends."""
    _ms = _market_service()
    if _ms._is_global_fx_market_open(now):
        return max(1, _ms._MACRO_ACTIVE_TTL_SECONDS)
    return max(1, _ms._MACRO_ACTIVE_TTL_SECONDS, _ms._MACRO_OFFHOURS_TTL_SECONDS)


def _effective_bist_index_fresh_ttl(now: Optional[datetime] = None) -> int:
    """Return 30s during BIST trading hours and 300s during off-hours/weekends."""
    _ms = _market_service()
    if _ms._is_bist_market_open(now):
        return max(1, _ms._BIST_INDEX_ACTIVE_TTL_SECONDS)
    return max(1, _ms._BIST_INDEX_ACTIVE_TTL_SECONDS, _ms._BIST_INDEX_OFFHOURS_TTL_SECONDS)


def _compute_market_stocks_delta(
    prev_snapshot: Dict[str, tuple[Any, ...]],
    rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, tuple[Any, ...]]]:
    """Return stock rows whose quote or screener fundamentals changed."""
    _ms = _market_service()
    changed: List[Dict[str, Any]] = []
    next_snapshot: Dict[str, tuple[Any, ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("company") or row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        sig = tuple(row.get(k) for k in _ms._MARKET_STOCK_DELTA_TRACKED_KEYS)
        next_snapshot[symbol] = sig
        if prev_snapshot and prev_snapshot.get(symbol) == sig:
            continue
        delta_item: Dict[str, Any] = {"company": symbol}
        for key in _ms._MARKET_STOCK_DELTA_PAYLOAD_KEYS:
            fundamental_key = key in {
                "latest_quarter", "fundamental_period", "has_kap_cache", "sector", "fk", "pd_dd", "fd_favok",
                "fd_satis", "net_borc_favok", "roe_pct", "favok_margin_pct", "net_margin_pct",
                "free_float_pct", "ttm_net_kar", "ttm_satis_gelirleri", "ttm_favok",
                "ttm_fcf", "ozkaynaklar", "net_borc",
            }
            if key in row and (row[key] is not None or fundamental_key):
                delta_item[key] = row[key]
        changed.append(delta_item)
    return changed, next_snapshot


def _market_stocks_payload(
    *,
    index_name: str = "XUTUM",
    force_refresh: bool = False,
    basic_summary_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    _ms = _market_service()
    normalized_index = _ms._normalize_stock_index(index_name)
    started_at = _ms.time.perf_counter()
    local_key = f"payload:{normalized_index}"
    shared_key = f"api:market:stocks:{normalized_index}:v4"
    no_quote_cached = _ms._market_no_quote_cache_active(normalized_index)
    effective_fresh_ttl = _ms._effective_market_quotes_fresh_ttl()
    effective_stale_ttl = max(effective_fresh_ttl, _ms._MARKET_QUOTES_STALE_TTL)

    def freshness(entry: Any) -> tuple[str, bool]:
        """Report quotes served from an aged snapshot instead of calling them fresh."""

        if isinstance(entry, dict) and entry.get("quote_stale"):
            return "stale", True
        return "fresh", False

    def revalidate_factory() -> Optional[Dict[str, Any]]:
        if _ms._market_no_quote_cache_active(normalized_index):
            return None
        built = _ms._build_market_stocks_payload(
            index_name=normalized_index,
            force_refresh=True,
            basic_summary_map=basic_summary_map,
        )
        if _ms._market_payload_is_no_quote(built):
            _ms._store_market_no_quote_cache(normalized_index)
            return None
        if not _ms._market_payload_has_quote(built):
            return None
        _ms._clear_market_no_quote_cache(normalized_index)
        # The SWR entry schema has no per-build metadata slot, so drop the
        # signal rather than leaking a non-contract field into the payload.
        built.pop("quote_stale", None)
        return built

    refresh_pending = False
    if force_refresh:
        if not no_quote_cached:
            refresh_pending = _ms._schedule_swr_revalidation(
                cache_key=shared_key,
                fresh_ttl_seconds=effective_fresh_ttl,
                stale_ttl_seconds=effective_stale_ttl,
                factory=revalidate_factory,
            )
        # A public refresh is a revalidation signal, never a cache bypass.
        force_refresh = False
    local_entry = _ms._STOCKS_CACHE.get(local_key, {}).get("entry")

    if not force_refresh and _ms._market_quote_entry_is_fresh(local_entry):
        payload = _ms._market_quote_entry_payload(local_entry) or {}
        quote_status, quote_stale = freshness(local_entry)
        payload.update({
            "cache_status": "hit",
            "quote_status": quote_status,
            "stale": quote_stale,
            "quote_error": None,
            "refresh_pending": refresh_pending,
        })
        _ms._log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status="hit",
            upstream_called=False,
            stale=quote_stale,
            started_at=started_at,
        )
        return payload

    shared_entry = _ms._shared_cache_get_dict(shared_key)
    stale_entry = local_entry if _ms._market_quote_entry_is_stale(local_entry) else None
    if _ms._market_quote_entry_is_stale(shared_entry):
        stale_entry = shared_entry
    if not force_refresh and _ms._market_quote_entry_is_fresh(shared_entry):
        _ms._STOCKS_CACHE[local_key] = {"_ts": _ms.time.time(), "entry": shared_entry}
        payload = _ms._market_quote_entry_payload(shared_entry) or {}
        quote_status, quote_stale = freshness(shared_entry)
        payload.update({
            "cache_status": "shared_hit",
            "quote_status": quote_status,
            "stale": quote_stale,
            "quote_error": None,
            "refresh_pending": refresh_pending,
        })
        _ms._log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status="shared_hit",
            upstream_called=False,
            stale=quote_stale,
            started_at=started_at,
        )
        return payload

    latest_build: Dict[str, Any] = {}

    def build_and_envelope() -> Optional[Dict[str, Any]]:
        nonlocal latest_build
        built = _ms._build_market_stocks_payload(
            index_name=normalized_index,
            force_refresh=force_refresh,
            basic_summary_map=basic_summary_map,
        )
        # Pop before the fallback copy so the internal signal can never reach a
        # public payload, including the `unavailable` response below.
        quote_stale = bool(built.pop("quote_stale", False))
        latest_build = dict(built)
        if _ms._market_payload_is_no_quote(built):
            _ms._store_market_no_quote_cache(normalized_index)
            return None
        if not _ms._market_payload_has_quote(built):
            return None
        _ms._clear_market_no_quote_cache(normalized_index)
        now = _ms.time.time()
        return {
            "payload": built,
            # Kept beside the payload so the response envelope can flag aged
            # quotes without storing a non-contract field in the payload.
            "quote_stale": quote_stale,
            "fresh_until": now + max(1, effective_fresh_ttl),
            "stale_until": now + max(1, effective_stale_ttl),
            "cached_at": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        }

    if no_quote_cached:
        entry, cache_status = None, "unavailable"
    else:
        entry, cache_status = _ms._get_or_set_single_flight(
            shared_key,
            ttl_seconds=max(1, effective_stale_ttl),
            factory=build_and_envelope,
            lock_key=f"api:market:stocks:{normalized_index}:lock:v4",
            lock_ttl_seconds=_ms._MARKET_QUOTES_LOCK_TTL_SECONDS,
            wait_timeout_seconds=_ms._MARKET_QUOTES_WAIT_TIMEOUT_SECONDS,
            poll_interval_seconds=_ms._MARKET_QUOTES_POLL_INTERVAL_SECONDS,
            cache_usable=_ms._market_quote_entry_is_fresh,
            allow_cached=not force_refresh,
        )

    if _ms._market_quote_entry_is_fresh(entry):
        _ms._STOCKS_CACHE[local_key] = {"_ts": _ms.time.time(), "entry": entry}
        payload = _ms._market_quote_entry_payload(entry) or {}
        quote_status, quote_stale = freshness(entry)
        payload.update({
            "cache_status": cache_status,
            "quote_status": quote_status,
            "stale": quote_stale,
            "quote_error": None,
            "refresh_pending": refresh_pending,
        })
        _ms._log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status=cache_status,
            upstream_called=cache_status == "miss",
            stale=quote_stale,
            started_at=started_at,
        )
        return payload

    stale_payload = _ms._market_quote_entry_payload(stale_entry)
    if stale_payload is not None:
        stale_payload.update({
            "cache_status": "stale",
            "quote_status": "stale",
            "stale": True,
            "quote_error": "Piyasa fiyat kaynağına ulaşılamadı.",
            "refresh_pending": refresh_pending or cache_status == "pending",
        })
        _ms._log_market_cache_event(
            endpoint="market/stocks",
            index=normalized_index,
            cache_status="stale",
            upstream_called=cache_status == "miss",
            stale=True,
            started_at=started_at,
        )
        return stale_payload

    unavailable = latest_build or _ms._market_unavailable_stocks_payload(normalized_index)
    unavailable.update({
        "cache_status": "unavailable",
        "quote_status": "unavailable",
        "stale": False,
        "quote_error": "Piyasa fiyat kaynağına ulaşılamadı.",
        "refresh_pending": refresh_pending or cache_status == "pending",
    })
    _ms._log_market_cache_event(
        endpoint="market/stocks",
        index=normalized_index,
        cache_status="unavailable",
        upstream_called=cache_status == "miss",
        stale=False,
        started_at=started_at,
    )
    return unavailable


def market_stocks_etag_revision(*, index_name: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the generation of the cached stock representation, if available."""

    _ms = _market_service()
    normalized_index = _ms._normalize_stock_index(index_name)
    local_key = f"payload:{normalized_index}"
    shared_key = f"api:market:stocks:{normalized_index}:v4"
    def _extract_revision(entry: Any) -> Any:
        if not isinstance(entry, dict):
            return None
        rev = entry.get("cached_at")
        if rev:
            return rev
        cached_payload = entry.get("payload")
        if isinstance(cached_payload, dict) and cached_payload.get("as_of"):
            return cached_payload.get("as_of")
        return None

    local_entry = _ms._STOCKS_CACHE.get(local_key, {}).get("entry")
    source_revision = _extract_revision(local_entry)
    if not source_revision:
        shared_entry = _ms._shared_cache_get_dict(shared_key)
        source_revision = _extract_revision(shared_entry)
    if not source_revision:
        return None

    # Rows are tied to the cache generation above. The remaining response
    # fields can change independently (for example stale/refresh state), so
    # include them in the validator without hashing the large row array.
    visible_state = {key: value for key, value in payload.items() if key != "rows"}
    return {
        "representation": "market-stocks-v2",
        "index": normalized_index,
        "source_revision": source_revision,
        "visible_state": visible_state,
    }
