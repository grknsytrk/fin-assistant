from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.market import _market_service


def _isyatirim_company_card_url(symbol: str) -> str:
    return f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"


def _isyatirim_basic_summary_url() -> str:
    return "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx"


def _parse_tr_decimal(raw: Any) -> Optional[float]:
    _ms = _market_service()
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        numeric = float(raw)
        if numeric != numeric:  # NaN guard
            return None
        return numeric

    token = str(raw).strip()
    if not token:
        return None

    token = token.replace("\xa0", "").replace(" ", "")
    token = token.replace("\u2212", "-")
    token = token.replace("%", "").replace("x", "").replace("X", "")
    if token in {"-", "--", "A/D", "N/A", "n/a"}:
        return None

    number_match = _ms.re.search(r"[-+]?\d[\d\.,]*", token)
    if not number_match:
        return None
    token = number_match.group(0)

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            # Turkish style: 1.234,56
            token = token.replace(".", "").replace(",", ".")
        else:
            # English style: 1,234.56
            token = token.replace(",", "")
    elif "," in token:
        if token.count(",") > 1:
            token = token.replace(",", "")
        else:
            left, right = token.split(",", 1)
            if len(right) == 3 and left.lstrip("+-").isdigit():
                token = token.replace(",", "")
            else:
                token = token.replace(",", ".")
    elif "." in token:
        if token.count(".") > 1:
            token = token.replace(".", "")
        else:
            left, right = token.split(".", 1)
            if len(right) == 3 and len(left.lstrip("+-")) > 3:
                token = token.replace(".", "")

    try:
        numeric = float(token)
        if numeric != numeric:  # NaN guard
            return None
        return numeric
    except Exception:
        return None


def _strip_html_cell(raw: str) -> str:
    _ms = _market_service()
    text = _ms.re.sub(r"<[^>]+>", " ", str(raw or ""), flags=_ms.re.IGNORECASE)
    text = _ms.html.unescape(text)
    return " ".join(text.split())


def _norm_text(value: str) -> str:
    return (
        str(value or "")
        .upper()
        .replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )


def _extract_isyatirim_basic_summary_map(html_text: str) -> Dict[str, Dict[str, Any]]:
    _ms = _market_service()
    items: Dict[str, Dict[str, Any]] = {}
    table_flags = _ms.re.IGNORECASE | _ms.re.DOTALL

    def rows_for_table(csv_name: str) -> List[str]:
        table_match = _ms.re.search(
            rf'<table\b(?=[^>]*\bdata-csvname\s*=\s*["\']{_ms.re.escape(csv_name)}["\'])[^>]*>(.*?)</table>',
            html_text,
            flags=table_flags,
        )
        return (
            _ms.re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_match.group(1), flags=table_flags)
            if table_match
            else []
        )

    def parse_cells(row_html: str, tag: str) -> List[str]:
        return [
            _ms._strip_html_cell(cell)
            for cell in _ms.re.findall(rf"<{tag}\b[^>]*>(.*?)</{tag}\s*>", row_html, flags=table_flags)
        ]

    # The overview table is already fetched for market-cap/share-count fallback.
    overview_header: Dict[str, int] = {}
    for row_html in rows_for_table("temelozet"):
        headers = parse_cells(row_html, "th")
        if headers:
            for idx, header in enumerate(headers):
                norm = _ms._norm_text(header)
                if norm == "KOD":
                    overview_header["symbol"] = idx
                elif norm == "SEKTOR":
                    overview_header["sector"] = idx
                elif "PIYASA DEGERI" in norm and "MN TL" in norm:
                    overview_header["market_cap_mn_try"] = idx
                elif "HALKA ACIKLIK" in norm:
                    overview_header["free_float_pct"] = idx
                elif norm.startswith("SERMAYE") and "MN TL" in norm:
                    overview_header["capital_mn_try"] = idx
            continue

        cells = parse_cells(row_html, "td")
        if not cells:
            continue
        symbol_idx = overview_header.get("symbol", 0)
        if symbol_idx >= len(cells):
            continue
        symbol = cells[symbol_idx].strip().upper()
        if not symbol:
            continue
        market_cap_mn_try = _ms._parse_tr_decimal(cells[overview_header.get("market_cap_mn_try", 4)]) if overview_header.get("market_cap_mn_try", 4) < len(cells) else None
        free_float_pct = _ms._parse_tr_decimal(cells[overview_header.get("free_float_pct", 6)]) if overview_header.get("free_float_pct", 6) < len(cells) else None
        capital_mn_try = _ms._parse_tr_decimal(cells[overview_header.get("capital_mn_try", 7)]) if overview_header.get("capital_mn_try", 7) < len(cells) else None
        sector_idx = overview_header.get("sector", 2)
        items[symbol] = {
            "market_cap": market_cap_mn_try * 1_000_000 if market_cap_mn_try is not None else None,
            "fdpo": round(free_float_pct / 100.0, 6) if free_float_pct is not None and free_float_pct >= 0 else None,
            "free_float_pct": free_float_pct,
            "shares_outstanding": capital_mn_try * 1_000_000 if capital_mn_try is not None else None,
            "sector": cells[sector_idx].strip() or None if sector_idx < len(cells) else None,
            "source": "isyatirim_temelozet",
        }

    def normalized_period(raw: str) -> Optional[str]:
        text = str(raw or "").strip().upper()
        quarter_match = _ms.re.fullmatch(r"(\d{4})\s*Q([1-4])", text)
        if quarter_match:
            return f"{quarter_match.group(1)}Q{quarter_match.group(2)}"
        match = _ms.re.fullmatch(r"(\d{1,2})[./-](\d{4})", text)
        if match:
            month, year = int(match.group(1)), int(match.group(2))
        else:
            match = _ms.re.fullmatch(r"(\d{4})[./-](\d{1,2})", text)
            if not match:
                return None
            year, month = int(match.group(1)), int(match.group(2))
        if year < 1900 or month not in {3, 6, 9, 12}:
            return None
        return f"{year}Q{(month + 2) // 3}"

    financial_header: Dict[str, int] = {}
    for row_html in rows_for_table("temelfinansal"):
        headers = parse_cells(row_html, "th")
        if headers:
            for idx, header in enumerate(headers):
                norm = _ms._norm_text(header).replace(" ", "")
                if norm == "KOD":
                    financial_header["symbol"] = idx
                elif norm in {"F/K", "FK"}:
                    financial_header["fk"] = idx
                elif "FD/FAVOK" in norm:
                    financial_header["fd_favok"] = idx
                elif "FD/SATIS" in norm:
                    financial_header["fd_satis"] = idx
                elif norm in {"PD/DD", "PDDD"}:
                    financial_header["pd_dd"] = idx
                elif "SONDONEM" in norm:
                    financial_header["provider_period"] = idx
            continue

        cells = parse_cells(row_html, "td")
        if not cells:
            continue
        symbol_idx = financial_header.get("symbol", 0)
        if symbol_idx >= len(cells):
            continue
        symbol = cells[symbol_idx].strip().upper()
        if not symbol:
            continue
        indexes = {
            "fk": financial_header.get("fk", 2),
            "fd_favok": financial_header.get("fd_favok", 3),
            "fd_satis": financial_header.get("fd_satis", 4),
            "pd_dd": financial_header.get("pd_dd", 5),
            "provider_period": financial_header.get("provider_period", 6),
        }
        record = items.setdefault(symbol, {"source": "isyatirim_temelfinansal"})
        record.update({
            "fk": _ms._parse_tr_decimal(cells[indexes["fk"]]) if indexes["fk"] < len(cells) else None,
            "fd_favok": _ms._parse_tr_decimal(cells[indexes["fd_favok"]]) if indexes["fd_favok"] < len(cells) else None,
            "fd_satis": _ms._parse_tr_decimal(cells[indexes["fd_satis"]]) if indexes["fd_satis"] < len(cells) else None,
            "pd_dd": _ms._parse_tr_decimal(cells[indexes["pd_dd"]]) if indexes["pd_dd"] < len(cells) else None,
            "provider_period": normalized_period(cells[indexes["provider_period"]]) if indexes["provider_period"] < len(cells) else None,
            "provider_source": "isyatirim_temelfinansal",
        })

    return items


def _fetch_isyatirim_basic_summary_map() -> Dict[str, Dict[str, Any]]:
    _ms = _market_service()
    import urllib.error
    import urllib.request

    now = _ms.time.time()
    cached = _ms._ISYATIRIM_BASIC_SUMMARY_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _ms._ISYATIRIM_BASIC_SUMMARY_CACHE_TTL:
        return cached.get("items", {})
    shared_cached = _ms._shared_cache_get_dict("api:isyatirim-basic-summary:v2")
    if shared_cached:
        items = dict(shared_cached.get("items") or {})
        _ms._ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": items}
        try:
            _ms._persist_isyatirim_basic_summary_if_changed(items)
        except Exception:
            _ms.LOGGER.exception("shared İş Yatırım fundamentals could not be persisted")
        return items

    request = urllib.request.Request(
        url=_ms._isyatirim_basic_summary_url(),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, Exception):
        retry_offset = max(0, _ms._ISYATIRIM_BASIC_SUMMARY_CACHE_TTL - 60)
        _ms._ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now - retry_offset, "items": {}}
        return {}

    items = _ms._extract_isyatirim_basic_summary_map(html_text)
    _ms._ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": items}
    _ms._shared_cache_set("api:isyatirim-basic-summary:v2", {"items": items}, _ms._ISYATIRIM_BASIC_SUMMARY_CACHE_TTL)
    try:
        _ms._persist_isyatirim_basic_summary_if_changed(items)
    except Exception:
        _ms.LOGGER.exception("İş Yatırım fundamentals could not be persisted")
    return items


def _persist_isyatirim_basic_summary_if_changed(items: Dict[str, Dict[str, Any]]) -> None:
    """Persist each distinct provider summary once per process and data directory."""

    _ms = _market_service()
    signature_payload = _ms.json.dumps(
        items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    signature = _ms.hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    processed_dir = _ms.CONFIG.paths.processed_dir
    persistence_scope = str(processed_dir.resolve())

    with _ms._ISYATIRIM_BASIC_SUMMARY_PERSIST_LOCK:
        if _ms._ISYATIRIM_BASIC_SUMMARY_PERSISTED_SIGNATURES.get(persistence_scope) == signature:
            return
        _ms.upsert_provider_fundamentals_bulk(processed_dir, items)
        _ms._ISYATIRIM_BASIC_SUMMARY_PERSISTED_SIGNATURES[persistence_scope] = signature


def _extract_isyatirim_historical_averages(html_text: str, symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    table_match = _ms.re.search(
        r'<table[^>]+data-csvname="tarihselortalamalar"[^>]*>(.*?)</table>',
        html_text,
        flags=_ms.re.IGNORECASE | _ms.re.DOTALL,
    )
    if not table_match:
        return {"ok": False, "error": "İş Yatırım 'Tarihsel Ortalamalar' tablosu bulunamadı."}

    table_html = table_match.group(1)
    rows = _ms.re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=_ms.re.IGNORECASE | _ms.re.DOTALL)
    if not rows:
        return {"ok": False, "error": "İş Yatırım tablosunda satır bulunamadı."}

    header_cells: Optional[List[str]] = None
    selected_cells: Optional[List[str]] = None
    symbol_upper = str(symbol or "").strip().upper()
    for row_html in rows:
        header_raw = _ms.re.findall(r"<th[^>]*>(.*?)</th>", row_html, flags=_ms.re.IGNORECASE | _ms.re.DOTALL)
        if header_raw:
            header_cells = [_ms._strip_html_cell(cell) for cell in header_raw]
            continue

        cells = _ms.re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=_ms.re.IGNORECASE | _ms.re.DOTALL)
        clean_cells = [_ms._strip_html_cell(cell) for cell in cells]
        if not clean_cells:
            continue
        row_symbol = clean_cells[0].upper() if clean_cells else ""
        if row_symbol == symbol_upper:
            selected_cells = clean_cells
            break
        if selected_cells is None and len(clean_cells) >= 3:
            selected_cells = clean_cells

    if not selected_cells or len(selected_cells) < 3:
        return {"ok": False, "error": "İş Yatırım tablosundan çarpan verisi ayrıştırılamadı."}

    index_fk = None
    index_fd_favok = None
    index_pd_dd = None
    index_fk_prim = None
    index_fd_favok_prim = None
    index_pd_dd_prim = None

    if header_cells:
        for idx, header in enumerate(header_cells):
            norm = _ms._norm_text(str(header))
            if "F/K" in norm and "TAHMIN" in norm:
                index_fk = idx
            elif ("FD/FAVOK" in norm or "FD/FAVÖK" in norm) and "TAHMIN" in norm:
                index_fd_favok = idx
            elif "PD/DD" in norm and "TAHMIN" in norm:
                index_pd_dd = idx
        if index_fk is not None and index_fk + 1 < len(header_cells) and "PRIM" in _ms._norm_text(header_cells[index_fk + 1]):
            index_fk_prim = index_fk + 1
        if (
            index_fd_favok is not None
            and index_fd_favok + 1 < len(header_cells)
            and "PRIM" in _ms._norm_text(header_cells[index_fd_favok + 1])
        ):
            index_fd_favok_prim = index_fd_favok + 1
        if index_pd_dd is not None and index_pd_dd + 1 < len(header_cells) and "PRIM" in _ms._norm_text(header_cells[index_pd_dd + 1]):
            index_pd_dd_prim = index_pd_dd + 1

    if index_fk is None:
        index_fk = 1 if len(selected_cells) > 1 else None
    if index_fk_prim is None and index_fk is not None and len(selected_cells) > index_fk + 1:
        index_fk_prim = index_fk + 1

    if index_fd_favok is None and len(selected_cells) >= 7:
        # Non-bank layout fallback: KOD, F/K, Prim, FD/FAVOK, Prim, PD/DD, Prim
        index_fd_favok = 3
    if index_fd_favok_prim is None and index_fd_favok is not None and len(selected_cells) > index_fd_favok + 1:
        index_fd_favok_prim = index_fd_favok + 1

    if index_pd_dd is None:
        if len(selected_cells) >= 7:
            index_pd_dd = 5
        elif len(selected_cells) >= 5:
            # Bank/insurance layout: KOD, F/K, Prim, PD/DD, Prim
            index_pd_dd = 3
        else:
            index_pd_dd = None
    if index_pd_dd_prim is None and index_pd_dd is not None and len(selected_cells) > index_pd_dd + 1:
        index_pd_dd_prim = index_pd_dd + 1

    fk = _ms._parse_tr_decimal(selected_cells[index_fk]) if index_fk is not None and len(selected_cells) > index_fk else None
    fd_favok = (
        _ms._parse_tr_decimal(selected_cells[index_fd_favok])
        if index_fd_favok is not None and len(selected_cells) > index_fd_favok
        else None
    )
    pd_dd = _ms._parse_tr_decimal(selected_cells[index_pd_dd]) if index_pd_dd is not None and len(selected_cells) > index_pd_dd else None

    fk_prim_isk = (
        _ms._parse_tr_decimal(selected_cells[index_fk_prim])
        if index_fk_prim is not None and len(selected_cells) > index_fk_prim
        else None
    )
    fd_favok_prim_isk = (
        _ms._parse_tr_decimal(selected_cells[index_fd_favok_prim])
        if index_fd_favok_prim is not None and len(selected_cells) > index_fd_favok_prim
        else None
    )
    pd_dd_prim_isk = (
        _ms._parse_tr_decimal(selected_cells[index_pd_dd_prim])
        if index_pd_dd_prim is not None and len(selected_cells) > index_pd_dd_prim
        else None
    )

    note_match = _ms.re.search(
        r'<div[^>]+class="table-note"[^>]*>(.*?)</div>',
        html_text,
        flags=_ms.re.IGNORECASE | _ms.re.DOTALL,
    )
    note = _ms._strip_html_cell(note_match.group(1)) if note_match else None

    return {
        "ok": True,
        "fk": fk,
        "fd_favok": fd_favok,
        "pd_dd": pd_dd,
        "fk_prim_iskonto_pct": fk_prim_isk,
        "fd_favok_prim_iskonto_pct": fd_favok_prim_isk,
        "pd_dd_prim_iskonto_pct": pd_dd_prim_isk,
        "note": note,
    }


def _fetch_isyatirim_multiples(symbol: str) -> Dict[str, Any]:
    _ms = _market_service()
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {
            "ok": False,
            "symbol": "",
            "error": "Sembol bos.",
        }

    cache_key = ticker
    now = _ms.time.time()
    cached = _ms._ISYATIRIM_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _ms._ISYATIRIM_CACHE_TTL:
        return cached
    shared_key = f"api:isyatirim-multiples:{cache_key}"
    shared_cached = _ms._shared_cache_get_dict(shared_key)
    if shared_cached:
        _ms._ISYATIRIM_CACHE[cache_key] = {**shared_cached, "_ts": now}
        return dict(shared_cached)

    url = _ms._isyatirim_company_card_url(ticker)
    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, Exception) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "source": "isyatirim_company_card",
            "url": url,
            "error": f"İş Yatırım bağlantı hatası: {exc}",
        }

    parsed = _ms._extract_isyatirim_historical_averages(html_text, ticker)
    payload: Dict[str, Any] = {
        "ok": bool(parsed.get("ok")),
        "symbol": ticker,
        "source": "isyatirim_company_card",
        "url": url,
        "fetched_at": _ms.datetime.now(_ms.timezone.utc).isoformat(),
        "_ts": now,
    }
    payload.update(parsed)
    _ms._ISYATIRIM_CACHE[cache_key] = payload
    _ms._shared_cache_set(shared_key, payload, _ms._ISYATIRIM_CACHE_TTL)
    return payload


def _format_isyatirim_chart_datetime(value: date) -> str:
    return f"{value.year:04d}{value.month:02d}{value.day:02d}000000"


def _parse_isyatirim_chart_datetime(raw: Any) -> Optional[datetime]:
    _ms = _market_service()
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        try:
            timestamp = float(raw)
            if timestamp > 10_000_000_000:
                timestamp /= 1000.0
            return _ms.datetime.fromtimestamp(timestamp, tz=_ms.timezone.utc)
        except Exception:
            return None
    return _ms._point_datetime(raw)


def _fetch_isyatirim_index_history(index_code: str, *, start_date: date, end_date: date) -> List[tuple[datetime, float]]:
    _ms = _market_service()
    import urllib.error
    import urllib.request

    normalized = _ms._normalize_market_index(index_code)
    url = (
        "https://www.isyatirim.com.tr/_Layouts/15/IsYatirim.Website/Common/ChartData.aspx/"
        "IndexHistoricalAll"
        f"?period=1440&from={_ms._format_isyatirim_chart_datetime(start_date)}"
        f"&to={_ms._format_isyatirim_chart_datetime(end_date)}&endeks={normalized}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Endeksler.aspx",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = _ms.json.loads(response.read().decode("utf-8", errors="ignore"))
    except (urllib.error.URLError, Exception):
        return []

    raw_rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(raw_rows, list):
        return []

    points: List[tuple[datetime, float]] = []
    for row in raw_rows:
        raw_time = None
        raw_close = None
        if isinstance(row, dict):
            raw_time = row.get("d") or row.get("date") or row.get("time") or row.get("x")
            raw_close = row.get("c") or row.get("close") or row.get("last") or row.get("y")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            raw_time = row[0]
            raw_close = row[1]

        point_dt = _ms._parse_isyatirim_chart_datetime(raw_time)
        close = _ms._numeric_chart_value(raw_close)
        if point_dt is None or close is None or close <= 0:
            continue
        points.append((point_dt, close))

    points.sort(key=lambda item: item[0])
    return points
