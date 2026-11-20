from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

import httpx

from app.funds_api import _fas

_ORIGINAL_URLOPEN = urllib.request.urlopen
_ORIGINAL_HTTPX_CLIENT = httpx.Client
_GEFAS_CLIENT_LOCK = threading.Lock()
_GEFAS_HTTP_CLIENT: Optional[httpx.Client] = None


def _get_gefas_http_client(timeout_seconds: float = 8.0) -> Tuple[httpx.Client, bool]:
    """Return a shared persistent HTTP client for GEFAS requests, or a fresh mock in tests."""
    if httpx.Client is not _ORIGINAL_HTTPX_CLIENT:
        return httpx.Client(timeout=timeout_seconds, follow_redirects=True), True
    global _GEFAS_HTTP_CLIENT
    with _GEFAS_CLIENT_LOCK:
        if _GEFAS_HTTP_CLIENT is None or _GEFAS_HTTP_CLIENT.is_closed:
            _GEFAS_HTTP_CLIENT = httpx.Client(
                timeout=timeout_seconds,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
        return _GEFAS_HTTP_CLIENT, False


def close_gefas_http_client() -> None:
    """Close the shared GEFAS HTTP client cleanly."""
    global _GEFAS_HTTP_CLIENT
    with _GEFAS_CLIENT_LOCK:
        client = _GEFAS_HTTP_CLIENT
        _GEFAS_HTTP_CLIENT = None
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


atexit.register(close_gefas_http_client)


def _fund_holding_sector_map() -> tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    fas = _fas()
    cached = fas._FUND_HOLDING_SECTOR_MAP_CACHE.get("default")
    now = fas.time.time()
    if cached and now - float(cached.get("_ts", 0)) < fas._FUND_HOLDING_SECTOR_MAP_CACHE_TTL:
        return dict(cached.get("map") or {}), {
            "cache_hit": True,
            "symbol_count": cached.get("symbol_count", 0),
            "source": cached.get("source"),
            "source_date": cached.get("source_date"),
            "warnings": list(cached.get("warnings") or []),
        }
    cache_key = "api:funds:holding-sector-map:v1"
    redis_cached = fas._shared_cache_get_dict(cache_key)
    if isinstance(redis_cached, dict):
        fas._FUND_HOLDING_SECTOR_MAP_CACHE["default"] = {
            "_ts": now,
            "map": dict(redis_cached.get("map") or {}),
            "symbol_count": redis_cached.get("symbol_count", 0),
            "source": redis_cached.get("source"),
            "source_date": redis_cached.get("source_date"),
            "warnings": list(redis_cached.get("warnings") or []),
        }
        return dict(redis_cached.get("map") or {}), {
            "cache_hit": True,
            "symbol_count": redis_cached.get("symbol_count", 0),
            "source": redis_cached.get("source"),
            "source_date": redis_cached.get("source_date"),
            "warnings": list(redis_cached.get("warnings") or []),
        }

    from app import kap_service as _kap_service

    get_bist_index_universe = getattr(
        fas, "get_bist_index_universe", _kap_service.get_bist_index_universe
    )

    priority_codes = list(fas._FUND_HOLDING_SECTOR_PRIORITY)
    sector_results: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[Exception]]] = {}
    if priority_codes:
        first_code = priority_codes[0]
        try:
            sector_results[first_code] = (get_bist_index_universe(first_code), None)
        except Exception as exc:
            sector_results[first_code] = (None, exc)
        remaining_codes = priority_codes[1:]
        if len(remaining_codes) > 1:
            with ThreadPoolExecutor(
                max_workers=min(6, len(remaining_codes)),
                thread_name_prefix="holdings-sector",
            ) as executor:
                future_by_code = {
                    sector_code: executor.submit(get_bist_index_universe, sector_code)
                    for sector_code in remaining_codes
                }
                for sector_code in remaining_codes:
                    try:
                        sector_results[sector_code] = (future_by_code[sector_code].result(), None)
                    except Exception as exc:
                        sector_results[sector_code] = (None, exc)
        else:
            for sector_code in remaining_codes:
                try:
                    sector_results[sector_code] = (get_bist_index_universe(sector_code), None)
                except Exception as exc:
                    sector_results[sector_code] = (None, exc)

    sector_map: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []
    source = None
    source_date = None
    for sector_code in priority_codes:
        universe, exc = sector_results.get(sector_code, (None, None))
        if exc is not None or not isinstance(universe, dict):
            if exc is not None:
                warnings.append(f"{sector_code}: {exc}")
            continue
        source = source or universe.get("source")
        source_date = source_date or universe.get("source_date")
        sector_label = fas._FUND_HOLDING_SECTOR_LABELS.get(sector_code, sector_code)
        for symbol in list(universe.get("symbols") or []):
            normalized_symbol = str(symbol or "").strip().upper().replace(".", "")
            if normalized_symbol and normalized_symbol not in sector_map:
                sector_map[normalized_symbol] = {
                    "sector_code": sector_code,
                    "sector_label": sector_label,
                }

    cache_payload = {
        "_ts": now,
        "map": sector_map,
        "symbol_count": len(sector_map),
        "source": source,
        "source_date": source_date,
        "warnings": warnings,
    }
    fas._FUND_HOLDING_SECTOR_MAP_CACHE["default"] = cache_payload
    fas._shared_cache_set(
        cache_key,
        {
            "map": sector_map,
            "symbol_count": len(sector_map),
            "source": source,
            "source_date": source_date,
            "warnings": warnings,
        },
        ttl_seconds=fas._FUND_HOLDING_SECTOR_MAP_CACHE_TTL,
    )
    return dict(sector_map), {
        "cache_hit": False,
        "symbol_count": len(sector_map),
        "source": source,
        "source_date": source_date,
        "warnings": warnings,
    }


def _fund_holdings_live_payload(*, normalized: str) -> Dict[str, Any]:
    fas = _fas()
    payload = fas._fund_holdings_static_payload(normalized=normalized)
    enriched = fas._enrich_fund_holdings_with_daily_market_data(payload, normalized)
    positions = []
    for position in enriched.get("positions") or []:
        if not isinstance(position, dict):
            continue
        positions.append({field: position.get(field) for field in fas._FUND_HOLDINGS_LIVE_POSITION_FIELDS})
    metadata = dict(enriched.get("source_metadata") or {})
    return {
        "fund_code": normalized,
        "status": enriched.get("status"),
        "positions": positions,
        "portfolio_effect": enriched.get("portfolio_effect"),
        "source": "daily_market_enrichment",
        "as_of": (enriched.get("portfolio_effect") or {}).get("as_of"),
        "source_metadata": {
            "source": "daily_market_enrichment",
            "static_cache_hit": metadata.get("cache_hit"),
            "disclosure_check": metadata.get("disclosure_check"),
            "daily_market_enrichment": metadata.get("daily_market_enrichment"),
            "market_enrichment": metadata.get("market_enrichment"),
        },
    }


def _api_number(raw: Any) -> Optional[float]:
    fas = _fas()
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if fas.math.isfinite(value) else None
    try:
        return fas._parse_tr_decimal(raw)
    except Exception:
        return None


def _fund_snapshot_row_map() -> Dict[str, Dict[str, Any]]:
    fas = _fas()
    rows, _meta = fas._fund_snapshot_row_map_with_meta()
    return rows


def _fund_snapshot_row_map_with_meta() -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    fas = _fas()
    from app import fund_service as _fund_service

    fund_listing_revision = getattr(
        fas, "fund_listing_revision", _fund_service.fund_listing_revision
    )
    load_funds_snapshot = getattr(fas, "load_funds_snapshot", _fund_service.load_funds_snapshot)
    normalize_fund_code = getattr(fas, "normalize_fund_code", _fund_service.normalize_fund_code)

    snapshot_path = fas.CONFIG.paths.processed_dir / "funds_cache" / "funds_latest.json"
    stat = snapshot_path.stat() if snapshot_path.exists() else None
    cache_key = str(snapshot_path)
    now_mono = fas.time.monotonic()
    cached = fas._FUND_SNAPSHOT_ROW_MAP_CACHE.get(cache_key)
    if cached and stat and cached.get("mtime") == stat.st_mtime:
        return dict(cached.get("rows") or {}), {
            "cache_hit": True,
            "row_count": cached.get("row_count", 0),
            "as_of": cached.get("as_of"),
        }
    if stat is not None:
        stat_version = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        version_token: Optional[str] = f"mtime={stat_version}"
    else:
        try:
            rev = fund_listing_revision(fas.CONFIG.paths.processed_dir)
            version_token = f"gen={int(rev.get('catalog_generation') or 0)}:price={int(rev.get('price_generation') or 0)}"
        except Exception:
            version_token = None
    if cached and stat is None and version_token and cached.get("version_token") == version_token:
        cached["expires_at"] = now_mono + 600.0
        return dict(cached.get("rows") or {}), {
            "cache_hit": True,
            "row_count": cached.get("row_count", 0),
            "as_of": cached.get("as_of"),
        }
    if (
        cached
        and stat is None
        and cached.get("mtime") is None
        and version_token is None
        and float(cached.get("expires_at") or 0.0) > now_mono
    ):
        return dict(cached.get("rows") or {}), {
            "cache_hit": True,
            "row_count": cached.get("row_count", 0),
            "as_of": cached.get("as_of"),
        }
    shared_key = f"api:funds:snapshot-row-map:{version_token}:v1" if version_token is not None else None
    if shared_key:
        shared_cached = fas._shared_cache_get_dict(shared_key)
        if shared_cached is not None:
            rows = dict(shared_cached.get("rows") or {})
            l1_ttl = 600.0 if (stat is None and version_token) else 60.0
            cache_payload = {
                "mtime": stat.st_mtime if stat else None,
                "version_token": version_token,
                "expires_at": now_mono + l1_ttl,
                "rows": rows,
                "row_count": shared_cached.get("row_count", len(rows)),
                "as_of": shared_cached.get("as_of"),
            }
            fas._FUND_SNAPSHOT_ROW_MAP_CACHE[cache_key] = cache_payload
            return rows, {
                "cache_hit": True,
                "row_count": cache_payload["row_count"],
                "as_of": cache_payload["as_of"],
                "shared_cache_hit": True,
            }

    try:
        snapshot = load_funds_snapshot(fas.CONFIG.paths.processed_dir)
    except Exception:
        return {}, {"cache_hit": False, "row_count": 0, "error": "snapshot_unavailable"}
    rows: Dict[str, Dict[str, Any]] = {}
    for row in list(snapshot.get("rows") or []):
        if not isinstance(row, dict):
            continue
        code = normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        if code:
            rows[code] = row
    if stat or rows:
        l1_ttl = 600.0 if (stat is None and version_token) else 60.0
        fas._FUND_SNAPSHOT_ROW_MAP_CACHE[cache_key] = {
            "mtime": stat.st_mtime if stat else None,
            "version_token": version_token,
            "expires_at": now_mono + l1_ttl,
            "rows": rows,
            "row_count": len(rows),
            "as_of": snapshot.get("as_of"),
        }
        if shared_key:
            fas._shared_cache_set(
                shared_key,
                {
                    "rows": rows,
                    "row_count": len(rows),
                    "as_of": snapshot.get("as_of"),
                },
                ttl_seconds=3600 if version_token else 300,
            )
    return dict(rows), {"cache_hit": False, "row_count": len(rows), "as_of": snapshot.get("as_of")}


def _holding_code(position: Dict[str, Any]) -> str:
    fas = _fas()
    from app import fund_service as _fund_service

    normalize_fund_code = getattr(fas, "normalize_fund_code", _fund_service.normalize_fund_code)
    return normalize_fund_code(str(position.get("asset_code") or position.get("asset_name") or "")).replace(
        ".", ""
    )


def _holding_type(position: Dict[str, Any]) -> str:
    return str(position.get("asset_type") or "").strip().lower()


def _gefas_gyf_config(symbol: str) -> Optional[Dict[str, str]]:
    fas = _fas()
    normalized = str(symbol or "").strip().upper().replace(".", "")
    return fas._GEFAS_GYF_ALIAS_MAP.get(normalized)


def _gefas_chart_date(raw: Any) -> Optional[str]:
    fas = _fas()
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return fas.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _fetch_gefas_gyf_chart(isin: str, metric: int) -> Dict[str, Any]:
    fas = _fas()
    normalized_isin = str(isin or "").strip().upper()
    if not normalized_isin:
        return {}
    url = f"https://gefas.gov.tr/gyf/detay/grafik/{normalized_isin}/0/0/{metric}"
    headers = {
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://gefas.gov.tr/tr/gyf/detay",
        "User-Agent": "Mozilla/5.0",
    }
    if urllib.request.urlopen is not _ORIGINAL_URLOPEN:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                return fas.json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, fas.json.JSONDecodeError, Exception):
            return {}

    client, is_ephemeral = _get_gefas_http_client(timeout_seconds=8.0)
    try:
        try:
            response = client.get(url, headers=headers, timeout=8.0)
        except httpx.RemoteProtocolError:
            response = client.get(url, headers=headers, timeout=8.0)
        response.raise_for_status()
        return fas.json.loads(response.content.decode("utf-8", errors="replace"))
    except (httpx.HTTPError, TimeoutError, fas.json.JSONDecodeError, Exception):
        return {}
    finally:
        if is_ephemeral:
            try:
                client.close()
            except Exception:
                pass


def _fetch_gefas_gyf_quote(symbol: str) -> Optional[Dict[str, Any]]:
    fas = _fas()
    config = fas._gefas_gyf_config(symbol)
    if not config:
        return None
    cache_key = config["isin"]
    now = fas.time.time()
    cached = fas._GEFAS_GYF_QUOTE_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < fas._GEFAS_GYF_QUOTE_CACHE_TTL:
        data = dict(cached.get("data") or {})
        if data:
            data["_cache_hit"] = True
        return data
    shared_key = f"api:funds:gefas-gyf-quote:{cache_key}:v1"
    shared_cached = fas._shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        fas._GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": shared_cached}
        data = dict(shared_cached)
        if data:
            data["_cache_hit"] = True
        return data

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gefas-chart") as executor:
        price_future = executor.submit(fas._fetch_gefas_gyf_chart, config["isin"], 0)
        return_future = executor.submit(fas._fetch_gefas_gyf_chart, config["isin"], 2)
        price_chart = price_future.result()
        return_chart = return_future.result()
    prices = list(price_chart.get("datas") or [])
    price_labels = list(price_chart.get("labels") or [])
    returns = list(return_chart.get("datas") or [])
    return_labels = list(return_chart.get("labels") or [])
    price = fas._api_number(prices[-1]) if prices else None
    return_pct = fas._api_number(returns[-1]) if returns else None
    as_of = fas._gefas_chart_date(price_labels[-1] if price_labels else None) or fas._gefas_chart_date(
        return_labels[-1] if return_labels else None
    )
    if price is None and return_pct is None:
        fas._GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": {}}
        fas._shared_cache_set(shared_key, {}, ttl_seconds=fas._GEFAS_GYF_QUOTE_CACHE_TTL)
        return {}

    data = {
        "price": price,
        "currency": "TRY",
        "change_pct": return_pct,
        "as_of": as_of,
        "source": "gefas_gyf",
        "source_url": f"https://gefas.gov.tr/tr/gyf/detay/{config['gefas_code']}",
        "isin": config["isin"],
        "gefas_code": config["gefas_code"],
        "label": config.get("label"),
    }
    fas._GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": data}
    fas._shared_cache_set(shared_key, data, ttl_seconds=fas._GEFAS_GYF_QUOTE_CACHE_TTL)
    result = dict(data)
    result["_cache_hit"] = False
    return result


def _quote_map_for_holding_stocks(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    fas = _fas()
    unique_symbols: List[str] = []
    seen_symbols: set[str] = set()
    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        if not normalized or normalized in seen_symbols:
            continue
        seen_symbols.add(normalized)
        unique_symbols.append(normalized)
    if not unique_symbols:
        return {}
    fetched_quotes = fas._fetch_market_price_map(unique_symbols, index_name="XUTUM")
    quotes = {symbol: fetched_quotes.get(symbol, {}) for symbol in unique_symbols if fetched_quotes.get(symbol)}
    missing_symbols = [
        symbol
        for symbol in unique_symbols
        if fas._api_number((quotes.get(symbol) or {}).get("price")) is None
        or fas._api_number((quotes.get(symbol) or {}).get("change_pct")) is None
    ]
    fallback_targets = list(missing_symbols[: fas._INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT])
    if len(fallback_targets) > 1:
        with ThreadPoolExecutor(
            max_workers=min(4, len(fallback_targets)),
            thread_name_prefix="holdings-stock-fb",
        ) as executor:
            fallback_futures = {
                symbol: executor.submit(fas._fetch_infoyatirim_stock_page_quote, symbol)
                for symbol in fallback_targets
            }
            for symbol in fallback_targets:
                fallback = fallback_futures[symbol].result()
                if fallback:
                    quotes[symbol] = fas._merge_market_price_fallback(quotes.get(symbol, {}), fallback)
    else:
        for symbol in fallback_targets:
            fallback = fas._fetch_infoyatirim_stock_page_quote(symbol)
            if fallback:
                quotes[symbol] = fas._merge_market_price_fallback(quotes.get(symbol, {}), fallback)
    return quotes


def _foreign_holding_provider_symbol(position: Dict[str, Any]) -> str:
    fas = _fas()
    from app import foreign_instruments as _foreign_instruments

    is_isin = getattr(fas, "is_isin", _foreign_instruments.is_isin)
    if position.get("provider_symbol_status") == "unresolved":
        return ""
    symbol = str(
        position.get("provider_symbol")
        or position.get("logo_symbol")
        or position.get("asset_code")
        or ""
    ).strip().upper()
    return "" if is_isin(symbol) else symbol


def _quote_map_for_foreign_holdings(positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    fas = _fas()
    symbols: List[str] = []
    seen: set[str] = set()
    for position in positions:
        if fas._holding_type(position) not in {"foreign_equity", "foreign_fund"}:
            continue
        symbol = fas._foreign_holding_provider_symbol(position)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    if not symbols:
        return {}

    now = fas.time.time()
    quotes: Dict[str, Dict[str, Any]] = {}
    l1_missing: List[str] = []
    for symbol in symbols:
        cached = fas._FOREIGN_HOLDING_QUOTE_CACHE.get(symbol)
        if cached and now - float(cached.get("_ts") or 0.0) < fas._FOREIGN_HOLDING_QUOTE_CACHE_TTL:
            data = dict(cached.get("data") or {})
            if data:
                quotes[symbol] = data
            continue
        l1_missing.append(symbol)

    if not l1_missing:
        return quotes

    raw_bulk = fas._shared_cache_get_dict(fas._FOREIGN_QUOTES_BULK_KEY)
    bulk_quotes: Dict[str, Any] = dict(raw_bulk) if isinstance(raw_bulk, dict) else {}
    bulk_dirty = False
    yahoo_missing: List[str] = []
    for symbol in l1_missing:
        shared_entry = bulk_quotes.get(symbol)
        if (
            isinstance(shared_entry, dict)
            and isinstance(shared_entry.get("exp"), (int, float))
            and float(shared_entry["exp"]) > now
            and isinstance(shared_entry.get("d"), dict)
        ):
            shared_cached = dict(shared_entry["d"])
            fas._FOREIGN_HOLDING_QUOTE_CACHE[symbol] = {"_ts": now, "data": shared_cached}
            if shared_cached:
                quotes[symbol] = dict(shared_cached)
            continue
        yahoo_missing.append(symbol)

    if yahoo_missing:
        raw_quotes_by_symbol: Dict[str, Dict[str, Any]] = {}
        if len(yahoo_missing) > 1:
            with ThreadPoolExecutor(
                max_workers=min(6, len(yahoo_missing)),
                thread_name_prefix="holdings-yahoo",
            ) as executor:
                yahoo_futures = {
                    symbol: executor.submit(fas._fetch_yahoo_quote, symbol)
                    for symbol in yahoo_missing
                }
                for symbol in yahoo_missing:
                    raw_quotes_by_symbol[symbol] = yahoo_futures[symbol].result() or {}
        else:
            only_symbol = yahoo_missing[0]
            raw_quotes_by_symbol[only_symbol] = fas._fetch_yahoo_quote(only_symbol) or {}

        for symbol in yahoo_missing:
            raw_quote = raw_quotes_by_symbol.get(symbol) or {}
            data: Dict[str, Any] = {}
            if raw_quote.get("ok") and (
                fas._api_number(raw_quote.get("price")) is not None
                or fas._api_number(raw_quote.get("change_pct")) is not None
            ):
                data = {
                    "price": fas._api_number(raw_quote.get("price")),
                    "currency": raw_quote.get("currency"),
                    "change_pct": fas._api_number(raw_quote.get("change_pct")),
                    "as_of": raw_quote.get("as_of"),
                    "source": "yahoo_finance_chart",
                    "provider_symbol": symbol,
                    "short_name": raw_quote.get("short_name"),
                    "long_name": raw_quote.get("long_name"),
                }
                quotes[symbol] = data
            fas._FOREIGN_HOLDING_QUOTE_CACHE[symbol] = {"_ts": now, "data": data}
            bulk_quotes[symbol] = {
                "d": data,
                "exp": now + fas._FOREIGN_HOLDING_QUOTE_CACHE_TTL,
            }
            bulk_dirty = True

    if bulk_dirty:
        fas._shared_cache_set(
            fas._FOREIGN_QUOTES_BULK_KEY,
            fas._prune_bulk_ttl_dict(bulk_quotes, now=now, max_entries=500),
            ttl_seconds=fas._FOREIGN_HOLDING_QUOTE_CACHE_TTL,
        )
    return quotes


def _foreign_holding_provider_name(quote: Dict[str, Any]) -> Optional[str]:
    for key in ("long_name", "short_name"):
        value = str(quote.get(key) or "").strip()
        if value:
            return value
    return None


def _foreign_holding_known_name(position: Dict[str, Any]) -> Optional[str]:
    fas = _fas()
    isin = str(position.get("isin") or "").strip().upper()
    if isin and isin in fas._FOREIGN_HOLDING_KNOWN_NAMES_BY_ISIN:
        return fas._FOREIGN_HOLDING_KNOWN_NAMES_BY_ISIN[isin]
    provider_symbol = fas._foreign_holding_provider_symbol(position)
    if provider_symbol and provider_symbol in fas._FOREIGN_HOLDING_KNOWN_NAMES_BY_PROVIDER:
        return fas._FOREIGN_HOLDING_KNOWN_NAMES_BY_PROVIDER[provider_symbol]
    return None


def _foreign_holding_should_use_provider_name(position: Dict[str, Any], provider_name: str) -> bool:
    fas = _fas()
    current = str(position.get("asset_name") or "").strip()
    if not provider_name or not current:
        return bool(provider_name and not current)
    code = fas._holding_code(position)
    provider_symbol = fas._foreign_holding_provider_symbol(position)
    normalized_current = fas.re.sub(r"[^A-Z0-9]+", "", current.upper())
    if normalized_current in {code, fas.re.sub(r"[^A-Z0-9]+", "", provider_symbol.upper())}:
        return True
    current_norm = current.upper()
    if " EQUITY" in current_norm or current_norm in {"CORP", "INC", "CMN", "AMERICA", "HOLDINGS", "MINERALS"}:
        return True
    letters = [char for char in current if char.isalpha()]
    return bool(letters and current == current.upper() and len(current) > 8)


def _position_daily_market_fields(
    position: Dict[str, Any],
    *,
    stock_quotes: Dict[str, Dict[str, Any]],
    gefas_quotes: Dict[str, Dict[str, Any]],
    foreign_quotes: Dict[str, Dict[str, Any]],
    fund_rows: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    fas = _fas()
    code = fas._holding_code(position)
    asset_type = fas._holding_type(position)
    if asset_type == "local_equity":
        quote = stock_quotes.get(code) or {}
        return {
            "price": fas._api_number(quote.get("price")),
            "price_currency": quote.get("currency") or "TRY",
            "return_pct": fas._api_number(quote.get("change_pct")),
            "return_source": "infoyatirim_live_quote" if quote else None,
            "return_as_of": quote.get("as_of"),
        }
    if asset_type == "fund":
        row = fund_rows.get(code) or {}
        quote = stock_quotes.get(code) or {}
        if quote and fas._api_number(quote.get("change_pct")) is not None:
            return {
                "price": fas._api_number(quote.get("price")),
                "price_currency": quote.get("currency") or "TRY",
                "return_pct": fas._api_number(quote.get("change_pct")),
                "return_source": "infoyatirim_live_quote",
                "return_as_of": quote.get("as_of"),
            }
        gefas_quote = gefas_quotes.get(code) or {}
        if gefas_quote and (
            fas._api_number(gefas_quote.get("price")) is not None
            or fas._api_number(gefas_quote.get("change_pct")) is not None
        ):
            return {
                "price": fas._api_number(gefas_quote.get("price")),
                "price_currency": gefas_quote.get("currency") or "TRY",
                "return_pct": fas._api_number(gefas_quote.get("change_pct")),
                "return_source": "gefas_gyf",
                "return_as_of": gefas_quote.get("as_of"),
            }
        return {
            "price": fas._api_number(row.get("price")),
            "price_currency": row.get("currency") or "TRY",
            "return_pct": fas._api_number(row.get("daily_return")),
            "return_source": "tefasfon_funds" if row else None,
            "return_as_of": row.get("as_of"),
        }
    if asset_type in {"foreign_equity", "foreign_fund"}:
        provider_symbol = fas._foreign_holding_provider_symbol(position)
        quote = foreign_quotes.get(provider_symbol) or {}
        return {
            "price": fas._api_number(quote.get("price")) if quote else fas._api_number(position.get("price")),
            "price_currency": quote.get("currency") if quote else None,
            "return_pct": fas._api_number(quote.get("change_pct")) if quote else None,
            "return_source": "yahoo_finance_chart" if quote else None,
            "return_as_of": quote.get("as_of") if quote else None,
        }
    return {
        "price": fas._api_number(position.get("price")),
        "price_currency": None,
        "return_pct": None,
        "return_source": None,
        "return_as_of": None,
    }


def _holding_weight_scale_context(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    fas = _fas()
    weights = [
        weight
        for weight in (fas._api_number(position.get("weight")) for position in positions)
        if weight is not None and weight > 0
    ]
    total = sum(weights)
    if not weights:
        return {"action": "none", "factor": 1.0, "reason": None}
    max_weight = max(weights)
    if len(weights) >= 3 and 9000 <= total <= 11000 and max_weight <= 10000:
        return {
            "action": "basis_points_to_percent",
            "factor": 0.01,
            "reason": "positive weights sum near 10000 basis points",
        }
    # Fractional weights (0..1) reported instead of percents.
    if max_weight <= 1.05 and total <= 1.05 and len(weights) >= 2:
        return {
            "action": "fraction_to_percent",
            "factor": 100.0,
            "reason": "positive weights look like fractions of one",
        }
    return {"action": "none", "factor": 1.0, "reason": None}


def _per_position_basis_points_threshold() -> float:
    # Anything above 1000% is far beyond plausible leverage and almost
    # certainly reported in basis points.  Using a relatively conservative
    # threshold keeps real leveraged positions (<= a few hundred percent)
    # untouched.
    return 1000.0


def _validated_holding_weight(
    raw_value: Any,
    scale_factor: float,
    *,
    quality_hint: Any = None,
    warning_hint: Any = None,
) -> Dict[str, Any]:
    fas = _fas()
    raw_number = fas._api_number(raw_value)
    if raw_number is None:
        return {"weight": None, "raw_weight": None, "quality": "missing", "warning": None}
    weight = raw_number * scale_factor
    hinted_quality = str(quality_hint or "").strip().lower()
    quality = "normalized" if scale_factor != 1.0 else ("fallback" if hinted_quality == "fallback" else "ok")
    warning = str(warning_hint or "").strip() or None
    # Per-position safety net: a single row well past 1000% with the rest of
    # the portfolio in normal percent territory is almost always a basis-point
    # leak.  Divide it down individually so other rows stay untouched.
    if (
        scale_factor == 1.0
        and abs(raw_number) > fas._per_position_basis_points_threshold()
        and abs(raw_number) <= 11000
    ):
        weight = raw_number / 100.0
        quality = "normalized"
        warning = warning or "KAP ağırlığı yüzde ölçeğine normalize edildi."
    # A value beyond the normalization range is not a plausible holding
    # weight.  It is commonly a leaked borsa/sözleşme number (for example
    # 80100517), so fail closed instead of exposing an astronomical percent.
    if not fas.math.isfinite(weight) or abs(weight) > fas._per_position_basis_points_threshold():
        return {
            "weight": None,
            "raw_weight": raw_number,
            "quality": "invalid",
            "warning": warning or "KAP ağırlığı geçersiz veya sözleşme numarası olarak algılandı.",
        }
    return {
        "weight": round(weight, 6),
        "raw_weight": raw_number if quality == "normalized" else None,
        "quality": quality,
        "warning": warning,
    }


def _validate_fund_holding_weights(
    positions: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fas = _fas()
    scale_context = fas._holding_weight_scale_context(positions)
    scale_factor = float(scale_context.get("factor") or 1.0)
    validated: List[Dict[str, Any]] = []
    normalized_count = 0
    fallback_count = 0
    invalid_count = 0
    raw_positive_total = 0.0
    adjusted_positive_total = 0.0

    for position in positions:
        row = dict(position)
        raw_weight = fas._api_number(row.get("weight"))
        if raw_weight is not None and raw_weight > 0:
            raw_positive_total += raw_weight

        weight_result = fas._validated_holding_weight(
            row.get("weight"),
            scale_factor,
            quality_hint=row.get("weight_quality"),
            warning_hint=row.get("weight_warning"),
        )
        previous_result = fas._validated_holding_weight(
            row.get("previous_weight"),
            scale_factor,
            quality_hint=row.get("previous_weight_quality"),
            warning_hint=row.get("previous_weight_warning"),
        )

        row["weight"] = weight_result["weight"]
        row["weight_quality"] = weight_result["quality"]
        row["weight_warning"] = weight_result["warning"]
        if weight_result["raw_weight"] is not None:
            row["raw_weight"] = weight_result["raw_weight"]
        else:
            row.pop("raw_weight", None)

        row["previous_weight"] = previous_result["weight"]
        row["previous_weight_quality"] = previous_result["quality"]
        row["previous_weight_warning"] = previous_result["warning"]
        if previous_result["raw_weight"] is not None:
            row["raw_previous_weight"] = previous_result["raw_weight"]
        else:
            row.pop("raw_previous_weight", None)

        if weight_result["quality"] == "normalized" or previous_result["quality"] == "normalized":
            normalized_count += 1
        if weight_result["quality"] == "fallback" or previous_result["quality"] == "fallback":
            fallback_count += 1
        if weight_result["quality"] == "invalid" or previous_result["quality"] == "invalid":
            invalid_count += 1

        current_weight = weight_result["weight"]
        previous_weight = previous_result["weight"]
        if weight_result["quality"] == "invalid" or previous_result["quality"] == "invalid":
            row["weight_change"] = None
        elif current_weight is not None and previous_weight is not None and (
            weight_result["quality"] != "ok" or previous_result["quality"] != "ok"
        ):
            row["weight_change"] = round(current_weight - previous_weight, 6)
        elif current_weight is not None and previous_weight is None and weight_result["quality"] != "ok":
            row["weight_change"] = round(current_weight, 6)

        if row.get("weight") is not None and row["weight"] > 0:
            adjusted_positive_total += float(row["weight"])

        validated.append(row)

    status = "ok"
    if adjusted_positive_total > 100.5:
        status = "gross_exposure"

    quality = {
        "status": status,
        "normalized_position_count": normalized_count,
        "fallback_position_count": fallback_count,
        "invalid_position_count": invalid_count,
        "raw_total_weight": round(raw_positive_total, 6),
        "adjusted_total_weight": round(adjusted_positive_total, 6),
        "normalization": {
            "action": scale_context.get("action") or "none",
            "factor": scale_factor,
            "reason": scale_context.get("reason"),
        },
    }
    return validated, quality


def _enrich_fund_holdings_with_daily_market_data(
    payload: Dict[str, Any], fund_code: str
) -> Dict[str, Any]:
    fas = _fas()
    from app import foreign_instruments as _foreign_instruments
    from app import fund_service as _fund_service

    resolve_foreign_holdings = getattr(
        fas, "resolve_foreign_holdings", _foreign_instruments.resolve_foreign_holdings
    )
    normalize_fund_code = getattr(fas, "normalize_fund_code", _fund_service.normalize_fund_code)

    normalized_fund = normalize_fund_code(fund_code)
    raw_positions = [
        dict(position) for position in list(payload.get("positions") or []) if isinstance(position, dict)
    ]
    backend_id = id(fas._get_cache())
    bulk_identities_loaded = False
    bulk_identities: Dict[str, Any] = {}
    bulk_identities_dirty = False
    now_epoch = fas.time.time()

    def _foreign_identity_cache_get(cache_key: str) -> Optional[Dict[str, Any]]:
        nonlocal bulk_identities_loaded, bulk_identities
        l1_key = (backend_id, cache_key)
        l1_hit = fas._FOREIGN_IDENTITIES_L1_CACHE.get(l1_key)
        now_check = fas.time.time()
        if (
            isinstance(l1_hit, dict)
            and float(l1_hit.get("_exp") or 0.0) > now_check
            and isinstance(l1_hit.get("d"), dict)
        ):
            return dict(l1_hit["d"])
        if not bulk_identities_loaded:
            bulk_identities_loaded = True
            loaded = fas._shared_cache_get_dict(fas._FOREIGN_IDENTITIES_BULK_KEY)
            if isinstance(loaded, dict):
                bulk_identities = dict(loaded)
        entry = bulk_identities.get(cache_key)
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("exp"), (int, float))
            and float(entry["exp"]) > now_epoch
            and isinstance(entry.get("d"), dict)
        ):
            data = dict(entry["d"])
            fas._FOREIGN_IDENTITIES_L1_CACHE[l1_key] = {"d": dict(data), "_exp": float(entry["exp"])}
            return data
        return None

    def _foreign_identity_cache_set(
        cache_key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        **_: Any,
    ) -> None:
        nonlocal bulk_identities_loaded, bulk_identities, bulk_identities_dirty
        effective_ttl = max(30, int(ttl_seconds or fas._FOREIGN_IDENTITIES_BULK_TTL))
        data = dict(value) if isinstance(value, dict) else {}
        expires_at = fas.time.time() + effective_ttl
        fas._FOREIGN_IDENTITIES_L1_CACHE[(backend_id, cache_key)] = {"d": data, "_exp": expires_at}
        if not bulk_identities_loaded:
            bulk_identities_loaded = True
            loaded = fas._shared_cache_get_dict(fas._FOREIGN_IDENTITIES_BULK_KEY)
            if isinstance(loaded, dict):
                bulk_identities = dict(loaded)
        bulk_identities[cache_key] = {
            "d": data,
            "exp": expires_at,
        }
        bulk_identities_dirty = True

    raw_positions = resolve_foreign_holdings(
        raw_positions, cache_get=_foreign_identity_cache_get, cache_set=_foreign_identity_cache_set
    )
    if bulk_identities_dirty:
        fas._shared_cache_set(
            fas._FOREIGN_IDENTITIES_BULK_KEY,
            fas._prune_bulk_ttl_dict(bulk_identities, now=fas.time.time(), max_entries=1000),
            ttl_seconds=fas._FOREIGN_IDENTITIES_BULK_TTL,
        )
    positions, holdings_quality = fas._validate_fund_holding_weights(raw_positions)
    fund_rows, fund_rows_meta = fas._fund_snapshot_row_map_with_meta()
    fund_row = fund_rows.get(normalized_fund) or {}
    fund_aum = fas._api_number(fund_row.get("aum"))
    stock_symbols = [
        fas._holding_code(position)
        for position in positions
        if fas._holding_code(position)
        and (
            fas._holding_type(position) == "local_equity"
            or (
                fas._holding_type(position) == "fund"
                and fas._holding_code(position) not in fund_rows
                and not fas._gefas_gyf_config(fas._holding_code(position))
            )
        )
    ]
    gefas_position_codes = [
        fas._holding_code(position)
        for position in positions
        if fas._holding_type(position) == "fund" and fas._gefas_gyf_config(fas._holding_code(position))
    ]

    def _run_gefas_leg() -> Tuple[Dict[str, Dict[str, Any]], int]:
        local_gefas_quotes: Dict[str, Dict[str, Any]] = {}
        local_gefas_hits = 0
        if not gefas_position_codes:
            return local_gefas_quotes, local_gefas_hits
        unique_gefas_codes = list(dict.fromkeys(gefas_position_codes))
        first_fetch_by_code: Dict[str, Optional[Dict[str, Any]]] = {}
        if len(unique_gefas_codes) > 1:
            with ThreadPoolExecutor(
                max_workers=min(4, len(unique_gefas_codes)),
                thread_name_prefix="holdings-gefas",
            ) as gefas_pool:
                gefas_futures = {
                    code: gefas_pool.submit(fas._fetch_gefas_gyf_quote, code)
                    for code in unique_gefas_codes
                }
                for code in unique_gefas_codes:
                    first_fetch_by_code[code] = gefas_futures[code].result()
        else:
            only_code = unique_gefas_codes[0]
            first_fetch_by_code[only_code] = fas._fetch_gefas_gyf_quote(only_code)

        seen_gefas_codes: set[str] = set()
        for code in gefas_position_codes:
            if code not in seen_gefas_codes:
                seen_gefas_codes.add(code)
                quote = first_fetch_by_code.get(code)
            else:
                quote = fas._fetch_gefas_gyf_quote(code)
            if quote:
                if quote.get("_cache_hit"):
                    local_gefas_hits += 1
                local_gefas_quotes[code] = quote
        return local_gefas_quotes, local_gefas_hits

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="holdings-enrich") as executor:
        stock_future = executor.submit(fas._quote_map_for_holding_stocks, stock_symbols)
        gefas_future = executor.submit(_run_gefas_leg)
        foreign_future = executor.submit(fas._quote_map_for_foreign_holdings, positions)
        sector_future = executor.submit(fas._fund_holding_sector_map)

        stock_quotes = stock_future.result()
        gefas_quotes, gefas_quote_cache_hits = gefas_future.result()
        foreign_quotes = foreign_future.result()
        sector_map, sector_meta = sector_future.result()

    enriched_positions: List[Dict[str, Any]] = []
    estimated_return_pct = 0.0
    estimated_pnl_value = 0.0
    has_pnl = False
    priced_weight = 0.0
    missing_weight = 0.0

    for position in positions:
        row = dict(position)
        row_type = fas._holding_type(row)
        if row_type in {"foreign_equity", "foreign_fund"}:
            provider_symbol = fas._foreign_holding_provider_symbol(row)
            row["asset_region"] = "foreign"
            row["provider_symbol"] = provider_symbol or None
            row["logo_symbol"] = row.get("logo_symbol") or provider_symbol or row.get("asset_code")
            row["detail_clickable"] = False
            quote = foreign_quotes.get(provider_symbol) if provider_symbol else None
            provider_name = fas._foreign_holding_provider_name(
                quote or {}
            ) or fas._foreign_holding_known_name(row)
            if provider_name:
                row["provider_name"] = provider_name
                row["asset_name"] = provider_name
        elif row_type == "local_equity":
            row["asset_region"] = row.get("asset_region") or "TR"
            row["detail_clickable"] = True if row.get("detail_clickable") is not False else False
        elif row_type == "fund":
            row["asset_region"] = row.get("asset_region") or "TR"
            holding_code = fas._holding_code(row)
            fund_row = fund_rows.get(holding_code) or {}
            provider_name = str(
                row.get("provider_name")
                or fund_row.get("founder_company")
                or fund_row.get("manager_company")
                or ""
            ).strip()
            if provider_name:
                row["provider_name"] = provider_name
                row["logo_symbol"] = row.get("logo_symbol") or provider_name
            if not row.get("logo_url") and fund_row.get("logo_url"):
                row["logo_url"] = fund_row.get("logo_url")
                row["logo_source"] = fund_row.get("logo_source")

        sector_info = sector_map.get(fas._holding_code(row)) if row_type == "local_equity" else None
        row["sector_code"] = sector_info.get("sector_code") if sector_info else None
        row["sector_label"] = sector_info.get("sector_label") if sector_info else None
        daily_fields = fas._position_daily_market_fields(
            row,
            stock_quotes=stock_quotes,
            gefas_quotes=gefas_quotes,
            foreign_quotes=foreign_quotes,
            fund_rows=fund_rows,
        )
        row.update(daily_fields)
        # Mark fund-type positions as TEFAS-tradable when they exist in the
        # daily TEFAS funds snapshot (or have a GEFAS-GYF override).  The
        # frontend uses this flag to decide whether the row is clickable.
        if row_type == "fund":
            holding_code = fas._holding_code(row)
            row["tefas_tradable"] = bool(
                holding_code and (holding_code in fund_rows or fas._gefas_gyf_config(holding_code))
            )
            row["detail_clickable"] = bool(row["tefas_tradable"])
        elif row_type == "foreign_fund":
            row["tefas_tradable"] = False
            row["detail_clickable"] = False
        else:
            row["tefas_tradable"] = None

        weight = fas._api_number(row.get("weight"))
        return_pct = fas._api_number(row.get("return_pct"))
        exposure_value = (
            (fund_aum * weight / 100.0)
            if fund_aum is not None and weight is not None and weight > 0
            else None
        )
        contribution_pct = (
            (weight * return_pct / 100.0)
            if weight is not None and weight > 0 and return_pct is not None
            else None
        )
        pnl_value = (
            (exposure_value * return_pct / 100.0)
            if exposure_value is not None and return_pct is not None
            else None
        )

        row["estimated_exposure_value"] = round(exposure_value, 2) if exposure_value is not None else None
        row["estimated_pnl_value"] = round(pnl_value, 2) if pnl_value is not None else None
        row["estimated_fund_return_contribution_pct"] = (
            round(contribution_pct, 6) if contribution_pct is not None else None
        )

        if weight is not None and weight > 0:
            if return_pct is not None:
                priced_weight += weight
                estimated_return_pct += contribution_pct or 0.0
                if pnl_value is not None:
                    estimated_pnl_value += pnl_value
                    has_pnl = True
            else:
                missing_weight += weight
        enriched_positions.append(row)

    enriched_payload = dict(payload)
    enriched_payload["positions"] = enriched_positions
    enriched_payload["portfolio_effect"] = {
        "period": "daily",
        "estimated_return_pct": round(estimated_return_pct, 6),
        "estimated_pnl_value": round(estimated_pnl_value, 2) if has_pnl else None,
        "priced_weight": round(priced_weight, 6),
        "missing_weight": round(missing_weight, 6),
        "aum": fund_aum,
        "as_of": fund_row.get("as_of") or (payload.get("source_metadata") or {}).get("as_of"),
    }
    metadata = dict(enriched_payload.get("source_metadata") or {})
    metadata["holdings_quality"] = holdings_quality
    metadata["daily_market_enrichment"] = {
        "period": "daily",
        "stock_quote_count": len(stock_quotes),
        "fund_snapshot_count": len(fund_rows),
        "gefas_gyf_quote_count": len(gefas_quotes),
        "gefas_gyf_quote_cache_hits": gefas_quote_cache_hits,
        "foreign_quote_count": len(foreign_quotes),
        "foreign_quote_missing_count": sum(
            1
            for position in positions
            if fas._holding_type(position) in {"foreign_equity", "foreign_fund"}
            and fas._foreign_holding_provider_symbol(position) not in foreign_quotes
        ),
        "sector_symbol_count": sector_meta.get("symbol_count"),
        "sector_cache_hit": sector_meta.get("cache_hit"),
        "sector_source": sector_meta.get("source"),
        "sector_source_date": sector_meta.get("source_date"),
        "daily_reference_cache_hit": bool(fund_rows_meta.get("cache_hit")),
        "daily_reference_row_count": fund_rows_meta.get("row_count"),
        "priced_weight": round(priced_weight, 6),
        "missing_weight": round(missing_weight, 6),
    }
    metadata["market_enrichment"] = {
        "stock_quote_live": True,
        "stock_quote_count": len(stock_quotes),
        "fund_daily_reference": "tefas_snapshot",
        "fund_daily_reference_cache_hit": bool(fund_rows_meta.get("cache_hit")),
        "gefas_gyf_cache_ttl_seconds": fas._GEFAS_GYF_QUOTE_CACHE_TTL,
        "foreign_daily_reference": "yahoo_finance_chart",
        "foreign_quote_count": len(foreign_quotes),
    }
    enriched_payload["source_metadata"] = metadata
    return enriched_payload
