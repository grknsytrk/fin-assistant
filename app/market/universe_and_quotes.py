from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.market import _market_service


def _log_market_cache_event(
    *,
    endpoint: str,
    index: str,
    cache_status: str,
    upstream_called: bool,
    stale: bool,
    started_at: float,
) -> None:
    _ms = _market_service()
    elapsed_ms = int(max(0.0, (_ms.time.perf_counter() - started_at) * 1000))
    _ms.LOGGER.info(
        "market_cache endpoint=%s index=%s cache_status=%s upstream_called=%s stale=%s elapsed_ms=%s",
        endpoint,
        index,
        cache_status,
        str(bool(upstream_called)).lower(),
        str(bool(stale)).lower(),
        elapsed_ms,
    )


def _load_cached_kap_market_metadata(cache_dir: Path, symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return {}
    cache_file = cache_dir / f"{normalized_symbol}.json"
    if not cache_file.exists():
        try:
            stored = _ms.get_financial_snapshots_map(cache_dir.parent, [normalized_symbol]).get(normalized_symbol)
        except Exception:
            stored = None
        if stored and stored.get("has_kap_cache"):
            return {
                "latest_quarter": stored.get("latest_quarter"),
                "has_kap_cache": True,
                "shares_outstanding": stored.get("shares_outstanding"),
                "share_source": stored.get("share_source"),
                "company_title": stored.get("company_title"),
                "company": stored.get("symbol") or normalized_symbol,
            }
        return {}
    cache_key = ""
    signature = None
    shared_key = None
    try:
        stat = cache_file.stat()
        cache_key = str(cache_file.resolve())
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = _ms._KAP_MARKET_METADATA_CACHE.get(cache_key)
        if cached and cached.get("signature") == signature:
            return dict(cached.get("data") or {})
        shared_key = f"api:kap:market-metadata:{normalized_symbol}:mtime={stat.st_mtime_ns}:size={stat.st_size}:v1"
        shared_cached = _ms._shared_cache_get_dict(shared_key)
        if shared_cached is not None:
            metadata = dict(shared_cached)
            _ms._KAP_MARKET_METADATA_CACHE[cache_key] = {
                "signature": signature,
                "data": metadata,
            }
            return metadata
    except Exception:
        cache_key = ""
        signature = None
        shared_key = None

    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            payload = _ms.json.load(handle)
    except Exception:
        return {}

    company_title = str(payload.get("company_title") or "").strip()
    company_code = str(payload.get("company") or payload.get("stock_code") or symbol).strip().upper()

    quarters_raw = payload.get("quarters")
    quarters = [
        str(row.get("quarter") or "").strip().upper()
        for row in (quarters_raw or [])
        if isinstance(row, dict) and str(row.get("quarter") or "").strip()
    ]
    quarter_rows = [row for row in (quarters_raw or []) if isinstance(row, dict)]
    latest_row = max(
        quarter_rows,
        key=lambda row: _ms._quarter_label_sort_key(str(row.get("quarter") or "").strip().upper()),
        default=None,
    )
    shares_outstanding = None
    share_source = None
    if latest_row:
        for metric_key in ("odenmis_sermaye", "cikarilmis_sermaye"):
            for field in ("metrics", "metrics_ytd"):
                container = latest_row.get(field)
                if not isinstance(container, dict):
                    continue
                metric = container.get(metric_key)
                value = metric.get("value") if isinstance(metric, dict) else metric
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                    continue
                shares_outstanding = float(value)
                share_source = metric_key
                break
            if shares_outstanding is not None:
                break
    metadata = {
        "latest_quarter": _ms._latest_quarter_label(quarters),
        "has_kap_cache": True,
        "shares_outstanding": shares_outstanding,
        "share_source": share_source,
        "company_title": company_title or None,
        "company": company_code or normalized_symbol,
    }
    if cache_key and signature:
        _ms._KAP_MARKET_METADATA_CACHE[cache_key] = {
            "signature": signature,
            "data": metadata,
        }
        if shared_key:
            _ms._shared_cache_set(shared_key, metadata, ttl_seconds=_ms._KAP_MARKET_METADATA_CACHE_TTL)
    return metadata


def _stock_reference_record_from_kap_payload(symbol: str, payload: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    _ms = _market_service()
    normalized = str(symbol or "").strip().upper()
    stock_code = str(payload.get("stock_code") or payload.get("company") or normalized).strip().upper()
    title = str(payload.get("company_title") or payload.get("title") or payload.get("companyName") or "").strip()
    member_oid = str(payload.get("member_oid") or payload.get("mkk_member_oid") or "").strip()
    return {
        "kind": "stock",
        "symbol": stock_code or normalized,
        "name": title or None,
        "short_name": stock_code or normalized,
        "source": source,
        "source_id": member_oid or None,
        "logo_url": f"https://www.kap.org.tr/tr/api/member/logo/{member_oid}" if member_oid else None,
        "logo_source": "kap" if member_oid else None,
        "as_of": str(payload.get("fetched_at") or "").strip() or None,
        "aliases": [normalized] if normalized and normalized != stock_code else [],
        "metadata": {
            "latest_quarter": _ms._latest_quarter_label(
                [
                    str(row.get("quarter") or "").strip().upper()
                    for row in (payload.get("quarters") or [])
                    if isinstance(row, dict)
                ]
            ),
            "source_url": payload.get("source_url"),
        },
    }


def _upsert_stock_reference_from_kap_payload(symbol: str, payload: Dict[str, Any], *, source: str = "kap") -> None:
    _ms = _market_service()
    if not isinstance(payload, dict) or not payload.get("ok", True):
        return
    record = _ms._stock_reference_record_from_kap_payload(symbol, payload, source=source)
    if not record.get("symbol"):
        return
    try:
        _ms.upsert_instrument(_ms.CONFIG.paths.processed_dir, **record)
    except Exception:
        _ms.LOGGER.debug("stock reference upsert failed for %s", symbol, exc_info=True)


def _positive_float(raw: Any) -> Optional[float]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value <= 0:
        return None
    return value


def _market_cap_from_quote_and_meta(
    quote: Dict[str, Any],
    cached_meta: Dict[str, Any],
    basic_summary: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    _ms = _market_service()
    price = quote.get("price")
    shares = cached_meta.get("shares_outstanding")
    price_value = _ms._positive_float(price)
    shares_value = _ms._positive_float(shares)
    if price_value is not None and shares_value is not None:
        return price_value * shares_value

    quote_market_cap = _ms._positive_float(quote.get("market_cap"))
    if quote_market_cap is not None:
        return quote_market_cap

    summary = basic_summary or {}
    summary_shares = _ms._positive_float(summary.get("shares_outstanding"))
    if price_value is not None and summary_shares is not None:
        # Use the live quote with the provider's share count; its cached market
        # cap may lag behind the live price.
        return price_value * summary_shares

    market_cap = _ms._positive_float(summary.get("market_cap"))
    if market_cap is not None:
        return market_cap
    return None


def _market_universe_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    started_at = _ms.time.perf_counter()
    normalized_index = _ms._normalize_stock_index(index_name)
    payload, cache_status, stale, refresh_pending = _ms._shared_swr_payload(
        cache_key=f"api:market:universe:{normalized_index}:v3",
        factory=lambda: _ms._build_market_universe_payload(index_name=normalized_index, force_refresh=True),
        fresh_ttl_seconds=_ms._MARKET_UNIVERSE_CACHE_TTL,
        stale_ttl_seconds=max(_ms._MARKET_SWR_STALE_TTL_SECONDS, _ms._MARKET_UNIVERSE_CACHE_TTL * 2),
        local_cache=_ms._UNIVERSE_CACHE,
        local_key=f"payload:{normalized_index}",
        force_revalidate=force_refresh,
    )
    if payload is None:
        raise _ms.HTTPException(status_code=503, detail="Piyasa evreni yenileniyor. Lütfen kısa süre sonra tekrar deneyin.")
    _ms._log_market_cache_event(
        endpoint="market/universe",
        index=normalized_index,
        cache_status=cache_status,
        upstream_called=cache_status == "miss",
        stale=stale,
        started_at=started_at,
    )
    return _ms._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )


def _build_market_universe_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    _ms = _market_service()
    from app.kap_service import get_bist_index_universe

    normalized_index = _ms._normalize_stock_index(index_name)
    universe = get_bist_index_universe(normalized_index, force_refresh=force_refresh)
    symbols = list(universe.get("symbols") or [])
    try:
        bist_all_count = (
            int(universe.get("count") or 0)
            if normalized_index == "XUTUM"
            else int(get_bist_index_universe("XUTUM").get("count") or 0)
        )
    except Exception:
        bist_all_count = int(universe.get("count") or len(symbols)) if normalized_index == "XUTUM" else 0
    try:
        instrument_map = _ms.get_instruments(_ms.CONFIG.paths.processed_dir, "stock", symbols)
    except Exception as exc:
        _ms.LOGGER.warning("reference instrument batch lookup failed: %s", exc)
        instrument_map = {}
    try:
        financial_snapshots = _ms.get_financial_snapshots_map(_ms.CONFIG.paths.processed_dir, symbols)
    except Exception as exc:
        _ms.LOGGER.warning("financial snapshot batch lookup failed: %s", exc)
        financial_snapshots = {}
    rows: List[Dict[str, Any]] = []
    kap_cache_count = 0

    for symbol in symbols:
        normalized_symbol = str(symbol or "").strip().upper()
        instrument = instrument_map.get(normalized_symbol) or {}
        instrument_metadata = instrument.get("metadata") if isinstance(instrument.get("metadata"), dict) else {}
        cached_meta = financial_snapshots.get(normalized_symbol) or {}
        latest_quarter = (
            cached_meta.get("latest_quarter")
            or instrument_metadata.get("latest_quarter")
        )
        source = str(instrument.get("source") or "").strip().lower()
        has_kap_cache = bool(
            instrument_metadata.get("has_kap_cache")
            or source in {"kap", "kap_cache"}
            or cached_meta.get("has_kap_cache")
        )
        name = (
            str(instrument.get("name") or "").strip()
            or str(cached_meta.get("company_title") or "").strip()
            or normalized_symbol
        )
        logo_url = str(instrument.get("logo_url") or "").strip() or None
        logo_source = str(instrument.get("logo_source") or "").strip() or None
        if has_kap_cache:
            kap_cache_count += 1

        rows.append(
            {
                "symbol": normalized_symbol,
                "company": normalized_symbol,
                "name": name,
                "latest_quarter": latest_quarter,
                "has_kap_cache": has_kap_cache,
                "price": None,
                "price_currency": None,
                "change": None,
                "change_pct": None,
                "price_as_of": None,
                "market_cap": None,
                "logo_url": logo_url,
                "logo_source": logo_source,
            }
        )

    coverage_rows = [row for row in rows if row.get("has_kap_cache")]

    data = {
        "stats": {
            "index": normalized_index,
            "index_count": len(rows),
            "bist100_count": len(rows),
            "bist_all_count": bist_all_count,
            "kap_cache_count": kap_cache_count,
        },
        "universe": {
            "index": universe.get("index") or normalized_index,
            "count": int(universe.get("count") or len(rows)),
            "source": universe.get("source"),
            "source_url": universe.get("source_url"),
            "source_date": universe.get("source_date"),
            "fetched_at": universe.get("fetched_at"),
            "cache_hit": bool(universe.get("cache_hit")),
            "fallback_used": bool(universe.get("fallback_used")),
        },
        "rows": rows,
        "coverage_rows": coverage_rows,
    }
    return data


def _quote_ts_to_iso(raw: Any) -> Optional[str]:
    _ms = _market_service()
    if not isinstance(raw, (int, float)):
        return None
    try:
        return _ms.datetime.fromtimestamp(float(raw), tz=_ms.timezone.utc).isoformat()
    except Exception:
        return None


def _infoyatirim_stock_page_url(symbol: str) -> str:
    ticker = str(symbol or "").strip().lower()
    return f"https://infoyatirim.com/borsa/{ticker}-hisse"


def _infoyatirim_stock_page_text(html_text: str) -> str:
    _ms = _market_service()
    text = _ms.re.sub(r"<script\b[^>]*>.*?</script>", " ", str(html_text or ""), flags=_ms.re.IGNORECASE | _ms.re.DOTALL)
    text = _ms.re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=_ms.re.IGNORECASE | _ms.re.DOTALL)
    text = _ms.html.unescape(_ms.re.sub(r"<[^>]+>", " ", text, flags=_ms.re.IGNORECASE))
    text = " ".join(text.split())
    return (
        text.upper()
        .replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )


def _extract_infoyatirim_stock_page_quote(symbol: str, html_text: str) -> Dict[str, Any]:
    _ms = _market_service()
    ticker = str(symbol or "").strip().upper()
    if not ticker or not html_text:
        return {}

    text = _ms._infoyatirim_stock_page_text(html_text)

    def value_after(label: str) -> Optional[float]:
        match = _ms.re.search(rf"{label}\s+([-+]?\d[\d\.,]*\s*(?:%|₺|TL)?)", text, flags=_ms.re.IGNORECASE)
        return _ms._parse_tr_decimal(match.group(1)) if match else None

    price = value_after(r"SON ISLEM FIYATI")
    change_pct = value_after(r"GUNLUK DEGISIM\s+%")
    change = value_after(r"GUNLUK DEGISIM\s+\(TL\)")
    volume = value_after(r"GUNLUK HACIM\s+\(TL\)")
    if volume is None:
        volume = value_after(r"TOPLAM ISLEM HACMI")
    market_cap = value_after(r"PIYASA DEGERI")
    fk = value_after(r"F/K")
    pd_dd = value_after(r"PD/DD")
    fd_favok = value_after(r"FD/FAVOK")

    if price is None and change_pct is None and volume is None and market_cap is None and fk is None and pd_dd is None and fd_favok is None:
        return {}

    return {
        "price": price,
        "currency": "TRY",
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "market_cap": market_cap,
        "fk": fk,
        "pd_dd": pd_dd,
        "fd_favok": fd_favok,
        "market_state": "",
        "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat(),
    }


def _fetch_infoyatirim_html(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
) -> Optional[str]:
    import urllib.error
    import urllib.request

    req_headers = headers or {"User-Agent": "Mozilla/5.0"}
    request = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, Exception) as exc:
        # Windows system trust store may lack InfoYatirim's intermediate CA; fall back to httpx/certifi.
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSL" in type(exc).__name__.upper():
            try:
                import httpx

                resp = httpx.get(url, headers=req_headers, follow_redirects=True, timeout=timeout)
                if resp.status_code == 200 and resp.text:
                    return resp.text
            except Exception:
                return None
        return None


def _fetch_infoyatirim_stock_page_quote(symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {}

    now = _ms.time.time()
    cached = _ms._INFOYATIRIM_STOCK_PAGE_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < _ms._INFOYATIRIM_STOCK_PAGE_CACHE_TTL:
        return dict(cached.get("data") or {})
    shared_key = f"api:market:infoyatirim-page-quote:{ticker}:v1"
    shared_cached = _ms._shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        _ms._INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached)

    html_text = _ms._fetch_infoyatirim_html(
        _ms._infoyatirim_stock_page_url(ticker),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
        timeout=8,
    )
    if html_text is None:
        _ms._INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": {}}
        _ms._shared_cache_set(shared_key, {}, ttl_seconds=_ms._INFOYATIRIM_STOCK_PAGE_CACHE_TTL)
        return {}

    data = _ms._extract_infoyatirim_stock_page_quote(ticker, html_text)
    _ms._INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": data}
    _ms._shared_cache_set(shared_key, data, ttl_seconds=_ms._INFOYATIRIM_STOCK_PAGE_CACHE_TTL)
    return dict(data)


def _market_price_row_needs_fallback(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return True
    return row.get("price") is None or row.get("change_pct") is None or row.get("volume") is None


def _merge_market_price_fallback(base: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not fallback:
        return base
    merged = dict(base or {})
    for key in ("price", "currency", "change", "change_pct", "volume", "market_cap", "market_state", "as_of"):
        if key not in merged or merged.get(key) is None or (key in {"currency", "market_state"} and merged.get(key) == ""):
            value = fallback.get(key)
            if value is not None:
                merged[key] = value
    return merged


def _market_price_source_url(index_name: str) -> str:
    normalized = str(index_name or "XUTUM").strip().upper()
    if normalized == "XUTUM":
        return "https://infoyatirim.com/canli-borsa"
    if normalized == "XU030":
        return "https://infoyatirim.com/canli-borsa/xu100-bist-100-hisseleri"
    return "https://infoyatirim.com/canli-borsa/xu100-bist-100-hisseleri"


def _market_price_snapshot_created_at(payload: Any) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    try:
        created_at = float(payload.get("created_at") or 0.0)
    except (TypeError, ValueError):
        return None
    return created_at if created_at > 0 else None


def _market_price_snapshot_age(payload: Any, *, now: Optional[float] = None) -> Optional[float]:
    _ms = _market_service()
    created_at = _ms._market_price_snapshot_created_at(payload)
    if created_at is None:
        return None
    return max(0.0, (_ms.time.time() if now is None else now) - created_at)


def _market_price_snapshot_fresh_ttl(now: Optional[float] = None) -> int:
    """Freshness window evaluated at read time: 60s in session, 300s off-hours."""
    _ms = _market_service()
    current = _ms.time.time() if now is None else now
    try:
        market_open = _ms._is_bist_market_open(_ms.datetime.fromtimestamp(current, _ms._TURKEY_TIMEZONE))
    except Exception:
        market_open = True
    if market_open:
        return max(1, int(_ms._MARKET_PRICE_SNAPSHOT_TRADING_TTL_SECONDS))
    return max(
        1,
        int(_ms._MARKET_PRICE_SNAPSHOT_TRADING_TTL_SECONDS),
        int(_ms._MARKET_PRICE_SNAPSHOT_OFFHOURS_TTL_SECONDS),
    )


def _market_price_snapshot_is_usable(payload: Any) -> bool:
    """Accept any snapshot that carries rows, regardless of its `expires_at`.

    Rows persisted before the freshness model changed still carry an
    `expires_at` in the past. They stay usable so a deploy does not start by
    rebuilding everything (and re-reading the blob) from scratch.
    """
    if not isinstance(payload, dict):
        return False
    items = payload.get("items")
    return isinstance(items, dict) and bool(items)


def _market_price_snapshot_is_fresh(payload: Any, *, now: Optional[float] = None) -> bool:
    """Whether the snapshot should be rebuilt from upstream right now.

    Freshness comes from ``created_at`` evaluated at read time. The legacy
    ``expires_at`` stamp is still honoured for rows written before this model
    existed, so an older persisted blob keeps working.
    """
    _ms = _market_service()
    if not _ms._market_price_snapshot_is_usable(payload):
        return False
    current = _ms.time.time() if now is None else now
    created_at = _ms._market_price_snapshot_created_at(payload)
    if created_at is not None:
        return (current - created_at) <= _ms._market_price_snapshot_fresh_ttl(current)
    try:
        expires_at = float(payload.get("expires_at") or 0.0)
    except (TypeError, ValueError):
        return False
    return expires_at > current


def _market_price_snapshot_payload(items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    _ms = _market_service()
    created_at = _ms.time.time()
    return {
        "items": items,
        "created_at": created_at,
    }


def _remember_market_price_snapshot(payload: Dict[str, Any]) -> None:
    """Keep the newest snapshot in process memory.

    The cache TTL already bounds its lifetime, so no extra load timestamp is
    stored: freshness is derived from ``created_at`` on every read.
    """
    _ms = _market_service()
    if not _ms._market_price_snapshot_is_usable(payload):
        return
    _ms._MARKET_PRICE_CACHE[_ms._MARKET_PRICE_SNAPSHOT_MEMORY_KEY] = {
        "created_at": _ms._market_price_snapshot_created_at(payload),
        "items": payload["items"],
    }


def _market_price_snapshot_memory_entry() -> Optional[Dict[str, Any]]:
    _ms = _market_service()
    entry = _ms._MARKET_PRICE_CACHE.get(_ms._MARKET_PRICE_SNAPSHOT_MEMORY_KEY)
    if not isinstance(entry, dict) or not isinstance(entry.get("items"), dict):
        return None
    return entry


def _market_price_snapshot_database_read() -> Optional[Dict[str, Any]]:
    """Read the persisted snapshot once. Usability, not freshness, gates this."""
    _ms = _market_service()
    if not _ms._market_database.database_enabled():
        return None
    try:
        payload = _ms._market_database.read_json_cache(_ms._MARKET_PRICE_DATABASE_CACHE_PATH)
    except Exception:
        return None
    return payload if _ms._market_price_snapshot_is_usable(payload) else None


def _market_price_snapshot_database_write(payload: Dict[str, Any]) -> None:
    _ms = _market_service()
    if not _ms._market_database.database_enabled():
        return
    try:
        _ms._market_database.write_json_cache(_ms._MARKET_PRICE_DATABASE_CACHE_PATH, payload)
    except Exception:
        _ms.LOGGER.warning("market price snapshot database write failed", exc_info=True)


def _market_price_redis_available() -> bool:
    _ms = _market_service()
    try:
        status = _ms._cache_status()
    except Exception:
        return False
    return status.get("cache_backend") == "redis" and not bool(status.get("cache_redis_fallback"))


def _market_price_snapshot_remaining_ttl(payload: Dict[str, Any]) -> int:
    """TTL for the shared cache entry, always at least one second."""
    _ms = _market_service()
    age = _ms._market_price_snapshot_age(payload)
    if age is not None:
        remaining = _ms._market_price_snapshot_fresh_ttl() - age
        return max(1, int(_ms.math.ceil(remaining)))
    try:
        legacy_remaining = float(payload.get("expires_at") or 0.0) - _ms.time.time()
    except (TypeError, ValueError):
        legacy_remaining = 0.0
    return max(1, int(_ms.math.ceil(legacy_remaining)))


def _market_price_snapshot_is_servable_stale(payload: Any) -> bool:
    """A usable snapshot that is old but still within the stale tolerance."""
    _ms = _market_service()
    if not _ms._market_price_snapshot_is_usable(payload):
        return False
    age = _ms._market_price_snapshot_age(payload)
    if age is None:
        # Legacy row with no `created_at`; there is no timestamp to judge, so
        # serve it once while the rebuild runs.
        return True
    return age <= max(1, int(_ms._MARKET_PRICE_SNAPSHOT_MAX_STALE_SECONDS))


def _begin_market_price_snapshot_refresh() -> bool:
    """Claim the process-local rebuild slot. Only one request may build."""
    _ms = _market_service()
    with _ms._MARKET_PRICE_SNAPSHOT_REFRESH_LOCK:
        if _ms._MARKET_PRICE_SNAPSHOT_REFRESH_EVENT.is_set():
            _ms._MARKET_PRICE_SNAPSHOT_REFRESH_EVENT.clear()
            return True
    return False


def _end_market_price_snapshot_refresh() -> None:
    _ms = _market_service()
    _ms._MARKET_PRICE_SNAPSHOT_REFRESH_EVENT.set()


def _wait_for_market_price_snapshot_refresh() -> None:
    _ms = _market_service()
    _ms._MARKET_PRICE_SNAPSHOT_REFRESH_EVENT.wait(
        timeout=max(0.05, float(_ms._MARKET_PRICE_SNAPSHOT_WAIT_TIMEOUT_SECONDS))
    )


def _market_price_snapshot_retry_allowed(*, has_stale: bool) -> bool:
    """Never hammer a failing upstream while a stale snapshot can be served."""
    _ms = _market_service()
    if not has_stale:
        return True
    with _ms._MARKET_PRICE_SNAPSHOT_STATE_LOCK:
        last_failure_at = _ms._MARKET_PRICE_SNAPSHOT_STATE["last_failure_at"]
    if last_failure_at <= 0.0:
        return True
    return (_ms.time.time() - last_failure_at) >= _ms._MARKET_PRICE_SNAPSHOT_RETRY_COOLDOWN_SECONDS


def _mark_market_price_snapshot_failure() -> None:
    _ms = _market_service()
    with _ms._MARKET_PRICE_SNAPSHOT_STATE_LOCK:
        _ms._MARKET_PRICE_SNAPSHOT_STATE["last_failure_at"] = _ms.time.time()


def _clear_market_price_snapshot_failure() -> None:
    _ms = _market_service()
    with _ms._MARKET_PRICE_SNAPSHOT_STATE_LOCK:
        _ms._MARKET_PRICE_SNAPSHOT_STATE["last_failure_at"] = 0.0


def _build_market_price_snapshot(
    normalized_symbols: List[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    _ms = _market_service()
    url = _ms._market_price_source_url(_ms._MARKET_QUOTE_SOURCE_INDEX)
    html_text = _ms._fetch_infoyatirim_html(url, timeout=10)
    if html_text is None:
        return None

    items: Dict[str, Dict[str, Any]] = {}
    try:
        row_pattern = _ms.re.compile(
            r'<tr[^>]+data-symbol="(?P<symbol>[A-Z0-9]+)"[^>]*>(?P<body>.*?)</tr>',
            flags=_ms.re.IGNORECASE | _ms.re.DOTALL,
        )
        price_pattern = _ms.re.compile(r'<td[^>]+class="price"[^>]+data-val="(?P<value>[^"]+)"', _ms.re.IGNORECASE)
        change_pattern = _ms.re.compile(r'<td[^>]+class="change"[^>]+data-val="(?P<value>[^"]+)"', _ms.re.IGNORECASE)
        percent_pattern = _ms.re.compile(r'<td[^>]+class="percent"[^>]+data-val="(?P<value>[^"]+)"', _ms.re.IGNORECASE)
        fetched_at = _ms.datetime.now(_ms.timezone.utc).isoformat()
        for match in row_pattern.finditer(html_text):
            symbol = str(match.group("symbol") or "").strip().upper()
            if not symbol:
                continue
            body = match.group("body") or ""
            price_match = price_pattern.search(body)
            change_match = change_pattern.search(body)
            percent_match = percent_pattern.search(body)

            volume = None
            cells = _ms.re.findall(r"<td\b[^>]*>(.*?)</td>", body, flags=_ms.re.IGNORECASE | _ms.re.DOTALL)
            if len(cells) > 4:
                volume_raw = _ms.html.unescape(_ms.re.sub(r"<[^>]+>", " ", cells[4], flags=_ms.re.IGNORECASE))
                volume = _ms._parse_tr_decimal(volume_raw)

            items[symbol] = {
                "price": _ms._parse_tr_decimal(price_match.group("value") if price_match else None),
                "currency": "TRY",
                "change": _ms._parse_tr_decimal(change_match.group("value") if change_match else None),
                "change_pct": _ms._parse_tr_decimal(percent_match.group("value") if percent_match else None),
                "volume": volume,
                "market_state": "",
                "as_of": fetched_at,
            }
    except Exception:
        return None

    missing_symbols = [
        symbol
        for symbol in normalized_symbols
        if _ms._market_price_row_needs_fallback(items.get(symbol))
    ]
    if items and missing_symbols:
        fallback_symbols = missing_symbols[:_ms._INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT]
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(6, len(fallback_symbols))) as pool:
                fallback_rows = list(pool.map(_ms._fetch_infoyatirim_stock_page_quote, fallback_symbols))
        except Exception:
            fallback_rows = [_ms._fetch_infoyatirim_stock_page_quote(symbol) for symbol in fallback_symbols]
        for symbol, fallback in zip(fallback_symbols, fallback_rows):
            if fallback:
                items[symbol] = _ms._merge_market_price_fallback(items.get(symbol, {}), fallback)
    return items


def _project_market_price_snapshot(
    items: Dict[str, Dict[str, Any]],
    symbols: List[str],
    *,
    stale: bool = False,
) -> Dict[str, Dict[str, Any]]:
    projected = {
        symbol: dict(items[symbol])
        for symbol in symbols
        if isinstance(items.get(symbol), dict)
    }
    if stale:
        for row in projected.values():
            # Additive flag only; existing keys are untouched so the public
            # response contracts stay stable.
            row["stale"] = True
    return projected


def _fetch_market_price_map(
    symbols: List[str],
    *,
    index_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fetch one canonical XUTUM quote snapshot and project the requested symbols.

    The snapshot is served from process memory, then the shared cache, then the
    persisted row. Only one request per process may rebuild it; every other
    request either waits briefly for that rebuild or is served the previous
    snapshot marked with ``stale=True``.
    """
    _ms = _market_service()
    if index_name is None:
        index_name = _ms._MARKET_QUOTE_SOURCE_INDEX
    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not normalized_symbols:
        return {}

    # Keep the public argument for callers that select XU100/XU030 rows, but
    # deliberately use the same upstream page and cache entry for all of them.
    del index_name
    cache_key = _ms._MARKET_PRICE_SHARED_CACHE_KEY

    def project(items: Dict[str, Dict[str, Any]], *, stale: bool = False) -> Dict[str, Dict[str, Any]]:
        return _ms._project_market_price_snapshot(items, normalized_symbols, stale=stale)

    now = _ms.time.time()
    fresh_ttl = _ms._market_price_snapshot_fresh_ttl(now)

    # Stale candidates are collected on the way down so a failed or in-flight
    # rebuild can still answer with the previous snapshot.
    stale_snapshot: Optional[Dict[str, Any]] = None

    local_entry = _ms._market_price_snapshot_memory_entry()
    if local_entry is not None:
        created_at = local_entry.get("created_at")
        age = max(0.0, now - float(created_at)) if created_at else None
        if age is not None and age <= fresh_ttl:
            return project(local_entry["items"])
        # Old but still cached: keep it as the fallback for this request so a
        # slow rebuild never turns into a Postgres read per call.
        candidate = {
            "items": local_entry["items"],
            "created_at": created_at,
        }
        if _ms._market_price_snapshot_is_servable_stale(candidate):
            stale_snapshot = candidate

    redis_available = _ms._market_price_redis_available()
    if redis_available:
        shared_cached = _ms._shared_cache_get_dict(cache_key)
        if _ms._market_price_snapshot_is_fresh(shared_cached, now=now):
            _ms._remember_market_price_snapshot(shared_cached)
            return project(shared_cached["items"])

    # Only hit Postgres when process memory has nothing usable. During an
    # upstream outage a stale memory snapshot is enough to answer, so the
    # 147 KB blob is not re-read on every request.
    if stale_snapshot is None:
        database_cached = _ms._market_price_snapshot_database_read()
        if _ms._market_price_snapshot_is_fresh(database_cached, now=now):
            _ms._remember_market_price_snapshot(database_cached)
            if redis_available:
                _ms._shared_cache_set(
                    cache_key,
                    database_cached,
                    ttl_seconds=_ms._market_price_snapshot_remaining_ttl(database_cached),
                )
            return project(database_cached["items"])
        if _ms._market_price_snapshot_is_servable_stale(database_cached):
            _ms._remember_market_price_snapshot(database_cached)
            stale_snapshot = database_cached

    if not _ms._market_price_snapshot_retry_allowed(has_stale=stale_snapshot is not None):
        _ms.LOGGER.info("Serving stale market snapshot; upstream rebuild is cooling down")
        return project(stale_snapshot["items"], stale=True)

    if not _ms._begin_market_price_snapshot_refresh():
        # Another request owns the rebuild. Serve the previous snapshot instead
        # of waiting, otherwise a slow upstream would stack up request threads.
        if stale_snapshot is not None:
            _ms.LOGGER.debug("Serving stale market snapshot while another request refreshes it")
            return project(stale_snapshot["items"], stale=True)
        _ms._wait_for_market_price_snapshot_refresh()
        refreshed_entry = _ms._market_price_snapshot_memory_entry()
        if refreshed_entry is not None and _ms._market_price_snapshot_is_servable_stale(refreshed_entry):
            is_fresh = _ms._market_price_snapshot_is_fresh(refreshed_entry, now=_ms.time.time())
            return project(refreshed_entry["items"], stale=not is_fresh)
        return {}

    try:
        items = _ms._build_market_price_snapshot(normalized_symbols)
        if items is None:
            _ms._mark_market_price_snapshot_failure()
            if stale_snapshot is not None:
                _ms.LOGGER.warning("Serving stale market snapshot after a failed rebuild")
                return project(stale_snapshot["items"], stale=True)
            return {}
        snapshot = _ms._market_price_snapshot_payload(items)
        _ms._market_price_snapshot_database_write(snapshot)
        _ms._remember_market_price_snapshot(snapshot)
        _ms._clear_market_price_snapshot_failure()
        if redis_available:
            _ms._shared_cache_set(
                cache_key,
                snapshot,
                ttl_seconds=_ms._market_price_snapshot_remaining_ttl(snapshot),
            )
        return project(snapshot["items"])
    finally:
        _ms._end_market_price_snapshot_refresh()


def _fetch_kap_price_payload(symbol: str) -> Dict[str, Any]:
    """Fetch latest stock price from Yahoo Finance for a BIST ticker."""
    _ms = _market_service()
    import urllib.request
    import urllib.error

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {
            "ok": False,
            "symbol": "",
            "error": "Sembol bos.",
        }
    cache_key = ticker
    now = _ms.time.time()

    # Return cached if fresh
    cached = _ms._PRICE_CACHE.get(cache_key)
    if cached and now - cached["_ts"] < _ms._PRICE_CACHE_TTL:
        return cached
    shared_key = f"api:market:stock-price:{ticker}:v1"
    shared_cached = _ms._shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        result = dict(shared_cached)
        result["_ts"] = now
        _ms._PRICE_CACHE[cache_key] = result
        return result

    yahoo_symbol = f"{ticker}.IS"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval=1d&range=1d"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _ms.json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "error": f"Yahoo Finance bağlantı hatası: {exc}",
        }

    try:
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        currency = meta.get("currency", "TRY")
        market_state = meta.get("marketState", "")
        regular_market_time = meta.get("regularMarketTime")

        change = None
        change_pct = None
        if price is not None and prev_close:
            change = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2)
        as_of = None
        if isinstance(regular_market_time, (int, float)):
            try:
                as_of = _ms.datetime.fromtimestamp(float(regular_market_time), tz=_ms.timezone.utc).isoformat()
            except Exception:
                as_of = None

        result: Dict[str, Any] = {
            "ok": True,
            "symbol": ticker,
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "currency": currency,
            "market_state": market_state,
            "as_of": as_of,
            "_ts": now,
        }
        _ms._PRICE_CACHE[cache_key] = result
        _ms._shared_cache_set(shared_key, {k: v for k, v in result.items() if k != "_ts"}, ttl_seconds=_ms._PRICE_CACHE_TTL)
        return result
    except (KeyError, IndexError, TypeError) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "error": f"Yahoo Finance veri parse hatası: {exc}",
        }


def _fill_prices_via_yahoo(symbols: List[str], base_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """For symbols missing a price in base_map, query Yahoo Finance individually."""
    _ms = _market_service()
    result = dict(base_map)
    missing = [s for s in symbols if not (result.get(s) or {}).get("price")]
    if not missing:
        return result

    from concurrent.futures import ThreadPoolExecutor

    def _one(sym: str) -> tuple[str, Dict[str, Any]]:
        payload = _ms._fetch_kap_price_payload(sym)
        if not payload.get("ok"):
            return sym, {}
        return sym, {
            "price": payload.get("price"),
            "currency": payload.get("currency") or "TRY",
            "change": payload.get("change"),
            "change_pct": payload.get("change_pct"),
            "market_state": payload.get("market_state") or "",
            "as_of": payload.get("as_of"),
        }

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for sym, quote in pool.map(_one, missing):
                if quote:
                    result[sym] = quote
    except Exception:
        for sym in missing:
            s, quote = _one(sym)
            if quote:
                result[s] = quote
    return result


def _xu030_payload() -> Dict[str, Any]:
    _ms = _market_service()
    from app.kap_service import get_bist30_companies

    now = _ms.time.time()
    cached = _ms._XU030_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _ms._XU030_CACHE_TTL:
        return cached["data"]

    symbols = get_bist30_companies()
    base_map = _ms._fetch_market_price_map(symbols)
    price_map = _ms._fill_prices_via_yahoo(symbols, base_map)
    basic_summary_map = _ms._fetch_isyatirim_basic_summary_map()

    cache_dir = _ms.CONFIG.paths.processed_dir / "kap_cache"

    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        cached_meta = _ms._load_cached_kap_market_metadata(cache_dir, symbol)
        latest_quarter = cached_meta.get("latest_quarter")
        quote = price_map.get(symbol, {})
        rows.append(
            {
                "company": symbol,
                "latest_quarter": latest_quarter,
                "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
                "price": quote.get("price"),
                "price_currency": quote.get("currency"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "price_as_of": quote.get("as_of"),
                "market_cap": _ms._market_cap_from_quote_and_meta(quote, cached_meta, basic_summary_map.get(symbol)),
                **_ms._empty_logo_payload(),
            }
        )

    data = {"index": "XU030", "rows": rows, "as_of": _ms.datetime.now(_ms.timezone.utc).isoformat()}
    _ms._XU030_CACHE["payload"] = {"_ts": now, "data": data}
    return data


def _fetch_yahoo_quote(yahoo_symbol: str) -> Dict[str, Any]:
    """Low-level Yahoo chart fetch for an arbitrary ticker (no cache)."""
    _ms = _market_service()
    import urllib.error
    import urllib.request

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval=1d&range=1d"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _ms.json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {"ok": False, "error": f"yahoo_error: {exc}"}

    try:
        meta = data["chart"]["result"][0]["meta"]
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "error": f"yahoo_parse: {exc}"}

    price = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    currency = meta.get("currency")
    short_name = meta.get("shortName")
    long_name = meta.get("longName")
    market_state = meta.get("marketState", "")
    rmt = meta.get("regularMarketTime")
    high = meta.get("regularMarketDayHigh")
    low = meta.get("regularMarketDayLow")
    volume = meta.get("regularMarketVolume")

    change = None
    change_pct = None
    if price is not None and prev_close:
        try:
            change = round(float(price) - float(prev_close), 4)
            change_pct = round((change / float(prev_close)) * 100, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            change = None
            change_pct = None

    as_of = None
    if isinstance(rmt, (int, float)):
        try:
            as_of = _ms.datetime.fromtimestamp(float(rmt), tz=_ms.timezone.utc).isoformat()
        except Exception:
            as_of = None

    return {
        "ok": True,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "high": high,
        "low": low,
        "volume": volume,
        "currency": currency,
        "short_name": short_name,
        "long_name": long_name,
        "market_state": market_state,
        "as_of": as_of,
    }


def _parse_yahoo_chart_payload(data: Dict[str, Any], yahoo_symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "error": f"yahoo_parse: {exc}", "yahoo_symbol": yahoo_symbol}

    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    volumes = quote.get("volume") or []

    points: List[Dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        close = closes[idx] if idx < len(closes) else None
        if close is None or not isinstance(ts, (int, float)):
            continue
        try:
            numeric_close = float(close)
        except (TypeError, ValueError):
            continue
        if numeric_close <= 0:
            continue

        def _numeric_at(values: List[Any]) -> Optional[float]:
            if idx >= len(values):
                return None
            value = values[idx]
            if value is None or isinstance(value, bool):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        points.append(
            {
                "time": _ms.datetime.fromtimestamp(float(ts), tz=_ms.timezone.utc).isoformat(),
                "close": numeric_close,
                "high": _numeric_at(highs),
                "low": _numeric_at(lows),
                "volume": _numeric_at(volumes),
            }
        )

    splits: List[Dict[str, Any]] = []
    raw_events = result.get("events")
    raw_splits = (raw_events.get("splits") or {}) if isinstance(raw_events, dict) else {}
    for event in raw_splits.values():
        if not isinstance(event, dict):
            continue
        try:
            event_date = _ms.datetime.fromtimestamp(float(event.get("date")), tz=_ms.timezone.utc).date()
            numerator = float(event.get("numerator"))
            denominator = float(event.get("denominator"))
            ratio = numerator / denominator if denominator else None
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            continue
        if ratio is not None and _ms.math.isfinite(ratio) and ratio > 0:
            splits.append({"date": event_date.isoformat(), "ratio": ratio})

    return {
        "ok": True,
        "yahoo_symbol": yahoo_symbol,
        "meta": meta,
        "points": points,
        "splits": sorted(splits, key=lambda split: split["date"]),
    }


def _fetch_yahoo_chart_url(url: str, yahoo_symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _ms.json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {"ok": False, "error": f"yahoo_error: {exc}", "yahoo_symbol": yahoo_symbol}

    return _ms._parse_yahoo_chart_payload(data, yahoo_symbol)


def _fetch_yahoo_chart_raw(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
    _ms = _market_service()
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval={interval}&range={range_}"
    )
    return _ms._fetch_yahoo_chart_url(url, yahoo_symbol)


def _fetch_yahoo_chart_period_raw(
    yahoo_symbol: str,
    *,
    interval: str,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    _ms = _market_service()
    period1 = int(_ms.datetime.combine(start_date, _ms.datetime.min.time(), tzinfo=_ms.timezone.utc).timestamp())
    period2 = int(_ms.datetime.combine(end_date + _ms.timedelta(days=1), _ms.datetime.min.time(), tzinfo=_ms.timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval={interval}&period1={period1}&period2={period2}&events=splits"
    )
    return _ms._fetch_yahoo_chart_url(url, yahoo_symbol)
