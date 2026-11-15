from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import re
from typing import Any, Dict, List, Optional

from app.kap_api import _kas

_VYK_DEFAULT_WINDOW_HOURS = 24
_VYK_DEFAULT_LIST_BUDGET = 2
_VYK_DEFAULT_DETAIL_BUDGET = 25


def _kap_flow_refresh_age_seconds(
    status: Optional[Dict[str, Any]] = None,
    *,
    now: Optional[datetime] = None,
) -> Optional[int]:
    """Return the age of the last successful flow refresh in seconds."""

    kas = _kas()
    current = status if status is not None else kas._kap_flow_store.get_flow_status()
    raw = current.get("last_successful_refresh")
    if not raw:
        return None
    try:
        parsed = kas.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=kas.timezone.utc)
    current_time = now or kas.datetime.now(kas.timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=kas.timezone.utc)
    age = (current_time.astimezone(kas.timezone.utc) - parsed.astimezone(kas.timezone.utc)).total_seconds()
    return max(0, int(age))


def _kap_flow_refresh_metadata() -> Dict[str, Any]:
    kas = _kas()
    status = kas._kap_flow_store.get_flow_status()
    age_seconds = kas._kap_flow_refresh_age_seconds(status)
    refresh_status = str(status.get("refresh_status") or "unknown")
    stale = age_seconds is None or age_seconds > kas._KAP_FLOW_REFRESH_STALE_AFTER_SECONDS
    if stale and refresh_status in {"", "unknown", "ok"}:
        refresh_status = "stale"
    refresh_pending = bool(status.get("refresh_pending")) or refresh_status in {"queued", "running"}
    return {
        "refresh_status": refresh_status,
        "refresh_pending": refresh_pending,
        "refresh_age_seconds": age_seconds,
        "last_successful_refresh": status.get("last_successful_refresh"),
        "last_refresh_attempt": status.get("last_refresh_attempt"),
    }


def _kap_flow_refresh_warning(metadata: Dict[str, Any]) -> Optional[str]:
    kas = _kas()
    age_seconds = metadata.get("refresh_age_seconds")
    if metadata.get("refresh_pending"):
        return "KAP akışı güncelleniyor; son başarılı veri gösteriliyor."
    if age_seconds is None or int(age_seconds) > kas._KAP_FLOW_REFRESH_STALE_AFTER_SECONDS:
        return "KAP akışı beklenenden eski; otomatik yenileme yeniden denenecek."
    return None


def _maybe_schedule_kap_flow_refresh(
    *,
    refresh_after_seconds: Optional[int] = None,
    cooldown_seconds: Optional[int] = None,
    watchdog_key: Optional[str] = None,
    fast: bool = False,
) -> bool:
    """Queue one non-blocking refresh when the durable flow is too old."""

    kas = _kas()
    if not kas._kap_flow_store.store_is_configured():
        return False
    refresh_after = max(
        1,
        int(
            kas._KAP_FLOW_REFRESH_STALE_AFTER_SECONDS
            if refresh_after_seconds is None
            else refresh_after_seconds
        ),
    )
    cooldown = max(
        1,
        int(
            kas._KAP_FLOW_REFRESH_COOLDOWN_SECONDS
            if cooldown_seconds is None
            else cooldown_seconds
        ),
    )
    lease_key = watchdog_key or kas._KAP_FLOW_REFRESH_WATCHDOG_KEY
    metadata = kas._kap_flow_refresh_metadata()
    age_seconds = metadata.get("refresh_age_seconds")
    if age_seconds is not None and int(age_seconds) <= refresh_after:
        return False

    backend = kas._get_cache()
    watchdog_l1_key = (kas.cache_reset_epoch(), id(backend), lease_key)
    l1_cooldown_until = kas._KAP_FLOW_WATCHDOG_L1_CACHE.get(watchdog_l1_key)
    if isinstance(l1_cooldown_until, (int, float)) and kas.time.time() < l1_cooldown_until:
        return False
    owner = kas.uuid.uuid4().hex
    requested_at = kas.datetime.now(kas.timezone.utc).isoformat()
    lease_value = {"owner": owner, "requested_at": requested_at}
    try:
        claimed = backend.set_if_absent(
            lease_key,
            lease_value,
            ttl_seconds=cooldown,
        )
    except Exception:
        kas.LOGGER.warning("KAP flow watchdog lease could not be acquired", exc_info=True)
        return False
    kas._KAP_FLOW_WATCHDOG_L1_CACHE[watchdog_l1_key] = kas.time.time() + min(
        kas._KAP_FLOW_WATCHDOG_L1_TTL_SECONDS,
        cooldown,
    )
    if not claimed:
        return False

    kas._kap_flow_store.update_flow_status(
        refresh_status="queued",
        refresh_pending=True,
        last_refresh_attempt=requested_at,
    )

    def refresh() -> None:
        try:
            kas._refresh_kap_flow_store(fast=fast)
        except Exception:
            # The last successful data remains authoritative. The cooldown
            # prevents a visible head poll from retrying on every request.
            kas.LOGGER.warning("KAP flow watchdog refresh failed", exc_info=True)

    try:
        kas._KAP_FLOW_REFRESH_EXECUTOR.submit(refresh)
    except Exception:
        kas._KAP_FLOW_WATCHDOG_L1_CACHE.pop(watchdog_l1_key, None)
        try:
            backend.release_if_owner(lease_key, lease_value)
        except Exception:
            kas.LOGGER.debug("KAP flow watchdog lease release failed", exc_info=True)
        kas._kap_flow_store.update_flow_status(
            refresh_status="failed",
            refresh_pending=False,
            last_refresh_attempt=requested_at,
        )
        kas.LOGGER.warning("KAP flow watchdog refresh could not be queued", exc_info=True)
        return False
    return True


def _maybe_schedule_kap_flow_live_refresh() -> bool:
    """Refresh active flow viewers without waiting for the stale threshold."""

    kas = _kas()
    return kas._maybe_schedule_kap_flow_refresh(
        refresh_after_seconds=kas._KAP_FLOW_REFRESH_LIVE_AFTER_SECONDS,
        cooldown_seconds=kas._KAP_FLOW_REFRESH_LIVE_COOLDOWN_SECONDS,
        watchdog_key=kas._KAP_FLOW_REFRESH_LIVE_WATCHDOG_KEY,
        fast=True,
    )


def _parse_kap_publish_date(raw: Any) -> Optional[datetime]:
    kas = _kas()
    if not raw:
        return None
    token = str(raw).strip()
    if not token:
        return None
    for fmt in (
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            return kas.datetime.strptime(token, fmt)
        except ValueError:
            continue
    try:
        return kas.datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_kap_source_datetime(raw: Any) -> Optional[datetime]:
    """Parse an absolute KAP timestamp as Europe/Istanbul time."""
    kas = _kas()
    parsed = kas._parse_kap_publish_date(raw)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=kas._KAP_SOURCE_TIMEZONE)
    return parsed


def _kap_category(disclosure_type: str, subject: str, summary: str = "") -> str:
    dt = (disclosure_type or "").strip().upper()
    subj = f"{subject or ''} {summary or ''}".lower()
    if dt.startswith("FR"):
        return "finansal_rapor"
    if "kar pay" in subj or dt == "KBR":
        return "kar_payi"
    if "geri alma" in subj or "geri alım" in subj or dt == "GR":
        return "geri_alim"
    if "genel kurul" in subj or dt == "GK":
        return "genel_kurul"
    if "kredi derec" in subj:
        return "kredi_derecelendirme"
    if "sürdürülebilir" in subj or dt == "SR":
        return "surdurulebilirlik"
    if "faaliyet rapor" in subj or dt == "FDR":
        return "faaliyet_raporu"
    if dt in {"ODA", "ÖDA"}:
        return "ozel_durum"
    if dt in {"DG", "DKB", "DK"}:
        return "diger"
    return "bildirim"


def _kap_source_label(disclosure_type: str) -> str:
    kas = _kas()
    dt = (disclosure_type or "").strip().upper()
    return kas._KAP_TYPE_LABELS.get(dt, "KAP")


def _kap_opener(force: bool = False) -> Any:
    """Build (and cache) a cookie-aware opener for kap.org.tr.

    Bootstraps a browser-like session by first requesting the HTML search page
    so KAP's WAF issues the cookies that later JSON calls require. Cached
    until TTL expires or `force=True`.
    """
    import http.cookiejar
    import urllib.error
    import urllib.request

    kas = _kas()
    now = kas.time.time()
    opener = kas._KAP_SESSION.get("opener")
    bootstrapped_at = float(kas._KAP_SESSION.get("bootstrapped_at") or 0.0)
    if opener is not None and not force and (now - bootstrapped_at) < kas._KAP_SESSION_TTL:
        return opener

    jar = http.cookiejar.CookieJar()
    new_opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPRedirectHandler(),
    )
    new_opener.addheaders = list(kas._KAP_DEFAULT_HEADERS.items())

    bootstrap_urls = [
        "https://www.kap.org.tr/tr/",
        "https://www.kap.org.tr/tr/bildirim-sorgu",
    ]
    bootstrap_results: List[Dict[str, Any]] = []
    for boot_url in bootstrap_urls:
        boot_started = kas.time.time()
        try:
            req = urllib.request.Request(
                boot_url,
                headers={
                    **kas._KAP_DEFAULT_HEADERS,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Upgrade-Insecure-Requests": "1",
                },
            )
            with new_opener.open(req, timeout=8) as resp:
                resp.read(1024)
            bootstrap_results.append(
                {
                    "url": boot_url,
                    "ok": True,
                    "elapsed_ms": int((kas.time.time() - boot_started) * 1000),
                    "cookies": len(list(jar)),
                }
            )
        except (urllib.error.URLError, TimeoutError, Exception) as exc:  # noqa: BLE001
            bootstrap_results.append(
                {
                    "url": boot_url,
                    "ok": False,
                    "elapsed_ms": int((kas.time.time() - boot_started) * 1000),
                    "error": type(exc).__name__,
                    "cookies": len(list(jar)),
                }
            )
            continue

    kas._KAP_SESSION["opener"] = new_opener
    kas._KAP_SESSION["bootstrapped_at"] = now
    # region agent log
    kas._debug_log(
        "H1",
        "app/api.py:1230",
        "KAP bootstrap completed",
        {
            "force": force,
            "cookie_count": len(list(jar)),
            "results": bootstrap_results,
        },
    )
    # endregion
    return new_opener


def _fetch_kap_disclosures_via_url(
    url: str,
    timeout: float,
    opener: Any,
) -> tuple[Any, Optional[str]]:
    """Single attempt JSON fetch using the shared KAP session opener."""
    import urllib.error
    import urllib.request

    kas = _kas()
    headers = {
        **kas._KAP_DEFAULT_HEADERS,
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    started = kas.time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        payload = kas.json.loads(raw)
        # region agent log
        kas._debug_log(
            "H2",
            "app/api.py:1266",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": True,
                "elapsed_ms": int((kas.time.time() - started) * 1000),
                "payload_type": type(payload).__name__,
                "item_count": len(payload) if isinstance(payload, list) else None,
            },
        )
        # endregion
        return payload, None
    except urllib.error.HTTPError as exc:
        # region agent log
        kas._debug_log(
            "H3",
            "app/api.py:1281",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": False,
                "elapsed_ms": int((kas.time.time() - started) * 1000),
                "error": f"HTTPError {exc.code}",
            },
        )
        # endregion
        return None, f"HTTPError {exc.code}"
    except (urllib.error.URLError, TimeoutError, ValueError, Exception) as exc:  # noqa: BLE001
        error_text = f"{type(exc).__name__}: {exc}"
        # region agent log
        kas._debug_log(
            "H2",
            "app/api.py:1295",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": False,
                "elapsed_ms": int((kas.time.time() - started) * 1000),
                "error": error_text,
            },
        )
        # endregion
        return None, error_text


class _KapDisclosureTableParser(HTMLParser):
    """Parse notification rows rendered by KAP's public result page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[Dict[str, Any]] = []
        self._row: Optional[Dict[str, Any]] = None
        self._cell_index: Optional[int] = None

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag == "tr" and self._row is None:
            attr_map = dict(attrs)
            row_id = str(attr_map.get("id") or "")
            if re.fullmatch(r"notification\d+", row_id):
                self._row = {"id": row_id, "disclosure_id": None, "cells": []}
                self._cell_index = None
            return

        if self._row is None:
            return
        if tag == "td":
            cells = self._row["cells"]
            cells.append([])
            self._cell_index = len(cells) - 1
        elif tag == "input" and self._cell_index == 0:
            attr_map = dict(attrs)
            input_id = str(attr_map.get("id") or "").strip()
            if input_id:
                self._row["disclosure_id"] = input_id

    def handle_endtag(self, tag: str) -> None:
        if self._row is None:
            return
        if tag == "td":
            self._cell_index = None
        elif tag == "tr":
            self.rows.append(self._row)
            self._row = None
            self._cell_index = None

    def handle_data(self, data: str) -> None:
        if self._row is None or self._cell_index is None:
            return
        self._row["cells"][self._cell_index].append(data)


def _parse_kap_table_datetime(raw: str) -> Optional[datetime]:
    """Parse the Turkish date/time text used in KAP notification cells."""
    kas = _kas()
    text = " ".join(str(raw or "").split())
    if not text:
        return None
    time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    hour = int(time_match.group(1)) if time_match else 0
    minute = int(time_match.group(2)) if time_match else 0

    relative_day: Optional[int] = None
    lowered = text.casefold()
    if "bugün" in lowered:
        relative_day = 0
    elif "dün" in lowered:
        relative_day = -1
    if relative_day is not None:
        base = kas.datetime.now(kas._KAP_SOURCE_TIMEZONE) + kas.timedelta(days=relative_day)
        return base.replace(hour=hour, minute=minute, second=0, microsecond=0)

    month_pattern = (
        r"\b(\d{1,2})\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|"
        r"ağustos|eylül|ekim|kasım|aralık)\s+(\d{4})\b"
    )
    date_match = re.search(month_pattern, lowered)
    if date_match:
        try:
            return kas.datetime(
                int(date_match.group(3)),
                kas._KAP_TABLE_MONTHS[date_match.group(2)],
                int(date_match.group(1)),
                hour,
                minute,
                tzinfo=kas._KAP_SOURCE_TIMEZONE,
            )
        except ValueError:
            return None

    return kas._parse_kap_source_datetime(text)


def _parse_kap_embedded_publish_dates(page: str) -> Dict[str, datetime]:
    """Extract absolute publish dates embedded in KAP's result page state."""
    kas = _kas()
    pattern = re.compile(
        r'publishDate[\\]?"\s*:[\\]?"([^"\\]+)[\\]?"\s*,\s*'
        r'[\\]?"disclosureIndex[\\]?"\s*:\s*(\d+)',
        re.IGNORECASE,
    )
    dates: Dict[str, datetime] = {}
    for match in pattern.finditer(page):
        disclosure_id = match.group(2)
        published_dt = kas._parse_kap_source_datetime(match.group(1))
        if disclosure_id and published_dt is not None:
            dates[disclosure_id] = published_dt
    return dates


def _parse_kap_table_codes(raw: str) -> List[str]:
    """Extract ticker/fund codes while ignoring KAP's '(+N)' marker."""
    values: List[str] = []
    for token in re.split(r"[,;\s]+", str(raw or "").strip()):
        normalized = token.strip().upper()
        if not normalized or normalized == "-" or re.fullmatch(r"\(\+\d+\)", normalized):
            continue
        if normalized not in values:
            values.append(normalized)
    return values


def _parse_kap_public_result_page(page: str, max_items: int) -> List[Dict[str, Any]]:
    kas = _kas()
    parser = kas._KapDisclosureTableParser()
    try:
        parser.feed(page)
        parser.close()
    except Exception:
        return []

    absolute_publish_dates = kas._parse_kap_embedded_publish_dates(page)
    results: List[Dict[str, Any]] = []
    for row in parser.rows:
        cells = [
            " ".join(" ".join(str(part) for part in cell).split())
            for cell in row.get("cells") or []
        ]
        if len(cells) < 8:
            continue

        disclosure_id = str(row.get("disclosure_id") or "").strip()
        published_dt = absolute_publish_dates.get(disclosure_id)
        if published_dt is None:
            published_dt = kas._parse_kap_table_datetime(cells[2])
        if published_dt is None:
            continue

        member_codes = kas._parse_kap_table_codes(cells[3])
        member_cell = cells[4] if len(cells) > 4 else ""
        related_tail = member_cell.split("/", 1)[1] if "/" in member_cell else ""
        related_codes = kas._parse_kap_table_codes(related_tail)
        if not related_codes and len(cells) > 8:
            related_codes = kas._parse_kap_table_codes(cells[8])
        display_codes = member_codes or related_codes
        symbol = related_codes[0] if related_codes else (display_codes[0] if display_codes else "")
        if not symbol:
            continue

        disclosure_type = cells[5].strip()
        subject = cells[6].strip()
        summary = cells[7].strip()
        title = subject or summary or "KAP Bildirimi"
        item: Dict[str, Any] = {
            "id": f"kap-{disclosure_id or row.get('id') or published_dt.isoformat()}",
            "disclosure_id": disclosure_id or None,
            "source": kas._kap_source_label(disclosure_type),
            "symbol": symbol,
            "stock_codes": display_codes,
            "related_symbols": related_codes,
            "title": title,
            "subject": subject,
            "published_at": published_dt.isoformat(),
            "category": kas._kap_category(disclosure_type, subject, summary),
            "kap_url": (
                f"https://www.kap.org.tr/tr/Bildirim/{disclosure_id}"
                if disclosure_id
                else None
            ),
        }
        results.append(item)
        if len(results) >= max(1, int(max_items)):
            break
    return results


def _fetch_kap_result_page_via_url(
    url: str,
    timeout: float,
    opener: Any,
) -> tuple[Optional[str], Optional[str]]:
    """Fetch KAP's browser-rendered disclosure result page."""
    import urllib.error
    import urllib.request

    kas = _kas()
    headers = {
        **kas._KAP_DEFAULT_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    started = kas.time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=timeout) as resp:
            page = resp.read().decode("utf-8", errors="ignore")
        kas._debug_log(
            "H2",
            "app/api.py:_fetch_kap_result_page_via_url",
            "KAP result page fetched",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": True,
                "elapsed_ms": int((kas.time.time() - started) * 1000),
                "bytes": len(page),
            },
        )
        return page, None
    except urllib.error.HTTPError as exc:
        return None, f"HTTPError {exc.code}"
    except (urllib.error.URLError, TimeoutError, Exception) as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _fetch_kap_disclosures_via_post(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    opener: Any,
) -> tuple[Any, Optional[str]]:
    """POST one criteria query to KAP's public disclosure API."""
    import urllib.error
    import urllib.request

    kas = _kas()
    headers = {
        **kas._KAP_DEFAULT_HEADERS,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://www.kap.org.tr",
        "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        request = urllib.request.Request(
            url,
            data=kas.json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with opener.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        return kas.json.loads(raw), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTPError {exc.code}"
    except (urllib.error.URLError, TimeoutError, ValueError, Exception) as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _fetch_kap_public_subjects(opener: Any) -> List[Dict[str, Any]]:
    """Fetch and cache KAP's subject catalog used for category backfills."""
    kas = _kas()
    now = kas.time.time()
    cached_items = kas._KAP_PUBLIC_SUBJECT_CACHE.get("items")
    if (
        isinstance(cached_items, list)
        and now - float(kas._KAP_PUBLIC_SUBJECT_CACHE.get("ts") or 0.0) < kas._KAP_PUBLIC_SUBJECT_CACHE_TTL
    ):
        return list(cached_items)

    payload, _error = kas._fetch_kap_disclosures_via_url(
        "https://www.kap.org.tr/tr/api/disclosure/subjects/ALL/Y",
        12.0,
        opener,
    )
    if not isinstance(payload, list):
        return []
    subjects = [
        {
            "subject": str(row.get("subject") or "").strip(),
            "subjectOid": str(row.get("subjectOid") or "").strip(),
            "disclosureClass": str(row.get("disclosureClass") or "").strip().upper(),
        }
        for row in payload
        if isinstance(row, dict) and row.get("subjectOid")
    ]
    kas._KAP_PUBLIC_SUBJECT_CACHE["items"] = subjects
    kas._KAP_PUBLIC_SUBJECT_CACHE["ts"] = now
    return list(subjects)


def _parse_kap_api_codes(raw: Any) -> List[str]:
    kas = _kas()
    if isinstance(raw, (list, tuple, set)):
        raw = " ".join(str(value) for value in raw)
    return kas._parse_kap_table_codes(str(raw or ""))


def _kap_category_source_label(category: str, disclosure_type: str) -> str:
    kas = _kas()
    if category == "finansal_rapor":
        return "Finansal Rapor"
    if category == "kar_payi":
        return "Kâr Payı"
    if category == "genel_kurul":
        return "Genel Kurul"
    return kas._kap_source_label(disclosure_type)


def _parse_kap_api_disclosures(payload: Any, max_items: int) -> List[Dict[str, Any]]:
    """Normalize rows returned by KAP's criteria API into flow items."""
    kas = _kas()
    if not isinstance(payload, list):
        return []

    results: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        published_dt = kas._parse_kap_source_datetime(row.get("publishDate"))
        disclosure_index = str(row.get("disclosureIndex") or "").strip()
        if published_dt is None or not disclosure_index:
            continue

        stock_codes = kas._parse_kap_api_codes(row.get("stockCodes") or row.get("fundCode"))
        related_codes = kas._parse_kap_api_codes(row.get("relatedStocks"))
        display_codes = stock_codes or related_codes
        symbol = related_codes[0] if related_codes else (display_codes[0] if display_codes else "")
        if not symbol:
            continue

        disclosure_type = str(row.get("disclosureType") or row.get("disclosureClass") or "").strip()
        subject = str(row.get("subject") or "").strip()
        summary = str(row.get("summary") or "").strip()
        category = kas._kap_category(disclosure_type, subject, summary)
        results.append(
            {
                "id": f"kap-{disclosure_index}",
                "disclosure_id": disclosure_index,
                "source": kas._kap_category_source_label(category, disclosure_type),
                "symbol": symbol,
                "stock_codes": display_codes,
                "related_symbols": related_codes,
                "title": subject or summary or "KAP Bildirimi",
                "subject": subject,
                "summary": summary,
                "published_at": published_dt.isoformat(),
                "category": category,
                "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{disclosure_index}",
            }
        )

    results.sort(key=lambda row: row.get("published_at") or "", reverse=True)
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in results:
        row_id = str(row.get("id") or "")
        if row_id and row_id in seen:
            continue
        if row_id:
            seen.add(row_id)
        deduped.append(row)
        if len(deduped) >= max(1, int(max_items)):
            break
    return deduped


def _fetch_kap_category_backfill(opener: Any) -> List[Dict[str, Any]]:
    """Fetch a small recent history window for sparse flow categories."""
    kas = _kas()
    topics = kas._fetch_kap_public_subjects(opener)
    if not topics:
        return []

    topic_groups = {
        "genel_kurul": [
            row["subjectOid"]
            for row in topics
            if "genel kurul" in row["subject"].casefold()
        ],
        "kar_payi": [
            row["subjectOid"]
            for row in topics
            if "kar pay" in row["subject"].casefold()
        ],
        "finansal_rapor": [
            row["subjectOid"]
            for row in topics
            if row["disclosureClass"] == "FR"
        ],
    }
    start_date = kas.date.today() - kas.timedelta(days=kas._KAP_FLOW_CATEGORY_BACKFILL_DAYS)
    end_date = kas.date.today()
    base_payload: Dict[str, Any] = {
        "fromDate": start_date.isoformat(),
        "toDate": end_date.isoformat(),
        "memberType": "IGS",
        "mkkMemberOidList": [],
        "inactiveMkkMemberOidList": [],
        "disclosureClass": "",
        "isLate": "",
        "mainSector": "",
        "sector": "",
        "subSector": "",
        "marketOid": "",
        "index": "",
        "bdkReview": "",
        "bdkMemberOidList": [],
        "year": "",
        "term": "",
        "ruleType": "",
        "period": "",
        "fromSrc": False,
        "srcCategory": "",
        "disclosureIndexList": [],
    }
    endpoint = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"
    results: List[Dict[str, Any]] = []
    for expected_category, subject_oids in topic_groups.items():
        if not subject_oids:
            continue
        query = dict(base_payload)
        query["subjectList"] = subject_oids
        payload, _error = kas._fetch_kap_disclosures_via_post(endpoint, query, 20.0, opener)
        parsed = kas._parse_kap_api_disclosures(payload, kas._KAP_FLOW_CATEGORY_TARGET * 3)
        category_rows = [row for row in parsed if row.get("category") == expected_category]
        results.extend(category_rows[:kas._KAP_FLOW_CATEGORY_TARGET])
    return results


def _fetch_kap_public_disclosures(
    max_items: int = 80,
    *,
    include_category_backfill: bool = True,
) -> List[Dict[str, Any]]:
    """Fetch recent company and fund disclosures from KAP's public UI page.

    KAP's public JSON list endpoint is WAF-protected for this deployment, but
    the same data is rendered by the official result page. We read the first
    page for both traded companies and funds, then merge by publication time.
    The public criteria API also supplies a bounded recent backfill for sparse
    category filters, so their latest 50 rows are not lost behind the general
    page window.
    The licensed KAP REST API remains the preferred option when credentials
    are configured.
    """
    kas = _kas()
    requested = max(1, min(int(max_items), kas._VYK_DEFAULT_DETAIL_BUDGET_MAX))
    page_urls = [
        "https://www.kap.org.tr/tr/bildirim-sorgu-sonuc?srcbar=Y&cmp=Y&cat=1&slf=ALL",
        "https://www.kap.org.tr/tr/bildirim-sorgu-sonuc?srcbar=Y&cmp=N&cat=6&slf=ALL",
    ]
    last_error: Optional[str] = None
    page_results: List[Dict[str, Any]] = []
    opener = kas._kap_opener()
    successful_pages = 0
    for url in page_urls:
        page, error = kas._fetch_kap_result_page_via_url(url, 12.0, opener)
        if page:
            successful_pages += 1
            page_results.extend(kas._parse_kap_public_result_page(page, requested))
        elif error:
            last_error = error

    if successful_pages:
        last_error = None

    page_results.sort(key=lambda row: row.get("published_at") or "", reverse=True)
    category_results = kas._fetch_kap_category_backfill(opener) if include_category_backfill else []
    results = [*page_results[:requested], *category_results]
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in results:
        row_id = str(row.get("id") or "")
        if row_id and row_id in seen:
            continue
        if row_id:
            seen.add(row_id)
        deduped.append(row)
    deduped.sort(key=lambda row: row.get("published_at") or "", reverse=True)
    kas._KAP_PUBLIC_LAST_ERROR["message"] = last_error
    kas._KAP_PUBLIC_LAST_ERROR["ts"] = kas.time.time()
    kas._KAP_PUBLIC_LAST_ERROR["source"] = page_urls[0]
    return deduped


def _local_flow_items_from_cache() -> List[Dict[str, Any]]:
    """Fallback feed constructed from locally cached financial reports."""
    kas = _kas()
    cache_dir = kas.CONFIG.paths.processed_dir / "kap_cache"
    started = kas.time.time()
    items: List[Dict[str, Any]] = []
    if not cache_dir.exists():
        # region agent log
        kas._debug_log(
            "H4",
            "app/api.py:1391",
            "Local flow cache scan finished",
            {"cache_dir_exists": False, "file_count": 0, "item_count": 0, "elapsed_ms": 0},
        )
        # endregion
        return items
    cache_files = list(cache_dir.glob("*.json"))
    for cache_file in cache_files:
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                payload = kas.json.load(handle)
        except Exception:
            continue
        symbol = str(
            payload.get("stock_code") or payload.get("company") or cache_file.stem
        ).strip().upper()
        quarters = payload.get("quarters")
        if not isinstance(quarters, list):
            continue
        for quarter in quarters[:2]:
            if not isinstance(quarter, dict):
                continue
            parsed_dt = kas._parse_kap_publish_date(quarter.get("publish_date"))
            if parsed_dt is None:
                continue
            title = str(quarter.get("title") or "Finansal Rapor").strip()
            quarter_label = str(quarter.get("quarter") or "").strip()
            disclosure_id = quarter.get("disclosure_index")
            items.append(
                {
                    "id": f"{symbol}-{disclosure_id or quarter_label or parsed_dt.isoformat()}",
                    "disclosure_id": str(disclosure_id).strip() if disclosure_id else None,
                    "source": "Finansal Rapor",
                    "symbol": symbol,
                    "stock_codes": [symbol],
                    "title": f"{title}{' - ' + quarter_label if quarter_label else ''}",
                    "subject": "Finansal Rapor",
                    "published_at": parsed_dt.isoformat(),
                    "category": "finansal_rapor",
                    "kap_url": (
                        f"https://www.kap.org.tr/tr/Bildirim/{disclosure_id}"
                        if disclosure_id
                        else None
                    ),
                }
            )
    # region agent log
    kas._debug_log(
        "H4",
        "app/api.py:1432",
        "Local flow cache scan finished",
        {
            "cache_dir_exists": True,
            "file_count": len(cache_files),
            "item_count": len(items),
            "elapsed_ms": int((kas.time.time() - started) * 1000),
        },
    )
    # endregion
    return items


def _vyk_source_label(disclosure_class: str, disclosure_type: str, subject_tr: str) -> str:
    cls = (disclosure_class or "").upper().strip()
    typ = (disclosure_type or "").upper().strip()
    subj = (subject_tr or "").lower()
    if typ == "CA":
        if "kar pay" in subj:
            return "Kâr Payı"
        if "genel kurul" in subj:
            return "Genel Kurul"
        if "geri al" in subj:
            return "Geri Alım"
        if "sermaye" in subj:
            return "Sermaye Artırımı"
        return "Hak Kullanımı"
    if typ == "FON":
        return "Fon"
    if typ.startswith("FR") or cls == "FR":
        return "Finansal Rapor"
    if typ == "ODA" or cls == "ODA":
        return "Özel Durum"
    if typ == "DUY" or cls == "DUY":
        return "Düzenleyici Kurum"
    if typ == "DG" or cls == "DG":
        return "Diğer Bildirim"
    return "KAP"


def _vyk_category(disclosure_class: str, disclosure_type: str, subject_tr: str) -> str:
    cls = (disclosure_class or "").upper().strip()
    typ = (disclosure_type or "").upper().strip()
    subj = (subject_tr or "").lower()
    if typ.startswith("FR") or cls == "FR":
        return "finansal_rapor"
    if typ == "ODA" or cls == "ODA":
        return "ozel_durum"
    if "kar pay" in subj:
        return "kar_payi"
    if "genel kurul" in subj:
        return "genel_kurul"
    if "geri al" in subj:
        return "geri_alim"
    if "kredi derec" in subj:
        return "kredi_derecelendirme"
    if "sürdürülebilir" in subj:
        return "surdurulebilirlik"
    if "faaliyet rapor" in subj:
        return "faaliyet_raporu"
    return "bildirim"


def _fetch_kap_vyk_feed(
    *,
    window_hours: int = _VYK_DEFAULT_WINDOW_HOURS,
    list_pages: int = _VYK_DEFAULT_LIST_BUDGET,
    detail_budget: int = _VYK_DEFAULT_DETAIL_BUDGET,
) -> List[Dict[str, Any]]:
    """Build a flow feed from the official VYK REST endpoints.

    Returns feed items sorted by `published_at` desc, filtered to the last
    `window_hours` hours. Returns `[]` when credentials are missing or any
    upstream call fails so the caller can fall back to the local cache.
    """
    import concurrent.futures

    from src import kap_vyk_client

    kas = _kas()
    cfg = getattr(kas.CONFIG, "kap", None)
    if cfg is None or not kap_vyk_client.is_enabled(cfg):
        return []

    started = kas.time.time()
    last_index = kap_vyk_client.get_last_disclosure_index(cfg)
    if not last_index or last_index <= 0:
        return []

    # `/disclosures` sayfalari 50 kayitlik. `list_pages` ile toplam pencereyi
    # kontrol altinda tutup upstream'e bindirmiyoruz.
    pages = max(1, min(int(list_pages or 1), 10))
    disclosures: List[Dict[str, Any]] = []
    cursor = int(last_index)
    seen_indexes: set[str] = set()
    for _ in range(pages):
        start_index = max(1, cursor - 49)
        rows = kap_vyk_client.list_disclosures_batch(cfg, start_index=start_index)
        if not rows:
            break
        added = 0
        for row in rows:
            idx = str(row.get("disclosureIndex") or "").strip()
            if not idx or idx in seen_indexes:
                continue
            seen_indexes.add(idx)
            disclosures.append(row)
            added += 1
        if added == 0:
            break
        cursor = start_index - 1
        if cursor <= 0:
            break

    if not disclosures:
        return []

    # Daha yeni kayitlari once isle; 24 saat penceresi disina taskinca
    # pahali detail cagrilarini kesmek icin bu sirali yaklasim sart.
    disclosures.sort(
        key=lambda node: int(str(node.get("disclosureIndex") or "0") or "0"),
        reverse=True,
    )

    members = kap_vyk_client.build_company_lookup(cfg)
    window_delta = kas.timedelta(hours=max(1, int(window_hours)))
    now = kas.datetime.now()
    cutoff = now - window_delta

    budget = max(1, min(int(detail_budget or 1), kas._VYK_DEFAULT_DETAIL_BUDGET_MAX))
    pending = disclosures[:budget]

    workers = max(1, min(kas._VYK_DETAIL_WORKERS, len(pending)))
    details: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                kap_vyk_client.get_disclosure_detail, cfg, row.get("disclosureIndex")
            ): str(row.get("disclosureIndex") or "")
            for row in pending
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                detail = future.result()
            except Exception:
                detail = None
            if detail and idx:
                details[idx] = detail

    items: List[Dict[str, Any]] = []
    fallback_items: List[Dict[str, Any]] = []
    for row in pending:
        idx_str = str(row.get("disclosureIndex") or "").strip()
        if not idx_str:
            continue
        detail = details.get(idx_str) or {}

        time_raw = str(detail.get("time") or "").strip()
        published_dt = kas._parse_kap_publish_date(time_raw)
        if published_dt is None:
            # Zaman damgasi yoksa akista saglikli konumlandiramayiz; atla.
            continue
        in_window = published_dt >= cutoff

        disclosure_class = str(
            row.get("disclosureClass") or detail.get("disclosureClass") or ""
        ).upper().strip()
        disclosure_type = str(
            row.get("disclosureType") or detail.get("disclosureType") or ""
        ).upper().strip()

        subject_obj = detail.get("subject")
        subject_tr = ""
        if isinstance(subject_obj, dict):
            subject_tr = str(subject_obj.get("tr") or "").strip()

        summary_obj = detail.get("summary")
        summary_tr = ""
        if isinstance(summary_obj, dict):
            summary_tr = str(summary_obj.get("tr") or "").strip()

        company_id = str(row.get("companyId") or detail.get("senderId") or "").strip()
        member = members.get(company_id, {})
        member_title = str(
            member.get("title") or row.get("title") or detail.get("senderTitle") or ""
        ).strip()
        member_stock_raw = str(member.get("stockCode") or "").strip().upper()
        member_stock = member_stock_raw.split(",")[0].strip() if member_stock_raw else ""

        sender_codes_raw = detail.get("senderExchCodes") or []
        stock_codes: List[str] = []
        primary_stock = ""
        if isinstance(sender_codes_raw, list) and sender_codes_raw:
            stock_codes = [
                str(code).strip().upper()
                for code in sender_codes_raw
                if str(code or "").strip()
            ]
            primary_stock = stock_codes[0] if stock_codes else ""
        if not primary_stock and member_stock:
            primary_stock = member_stock
            stock_codes = [member_stock]

        fund_code = str(
            detail.get("behalfFundCode") or row.get("fundCode") or ""
        ).strip().upper()
        symbol = primary_stock or fund_code or ""

        title = subject_tr or summary_tr or member_title or "KAP Bildirimi"
        subject = subject_tr or summary_tr

        built = {
            "id": f"vyk-{idx_str}",
            "disclosure_id": idx_str,
            "source": kas._vyk_source_label(disclosure_class, disclosure_type, subject_tr),
            "symbol": symbol,
            "stock_codes": stock_codes,
            "title": title,
            "subject": subject,
            "published_at": published_dt.isoformat(),
            "category": kas._vyk_category(disclosure_class, disclosure_type, subject_tr),
            "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx_str}",
        }
        if in_window:
            items.append(built)
        else:
            # Pencere disi. Test gateway'i eski sabit veri dondurdugunde veya
            # KAP'ta uzun suredir yeni bildirim olmadiginda akisi bos birakmamak
            # icin sonra saglikli bir fallback kovasinda tutulur.
            fallback_items.append(built)

    if not items and fallback_items:
        # Pencere icinde hic kayit yoksa, zaten detay bedelini odedigimiz en
        # yeni kayitlari kullaniciya gosteriyoruz. Her senaryoda budget disinda
        # ekstra upstream istegi yok.
        items = fallback_items

    items.sort(key=lambda node: node.get("published_at") or "", reverse=True)

    used_fallback = not any(
        kas._parse_kap_publish_date(node.get("published_at")) and
        kas._parse_kap_publish_date(node.get("published_at")) >= cutoff  # type: ignore[operator]
        for node in items
    ) and bool(items)
    # region agent log
    kas._debug_log(
        "H13",
        "app/api.py:_fetch_kap_vyk_feed",
        "KAP VYK feed built",
        {
            "last_index": int(last_index),
            "list_pages": pages,
            "disclosures": len(disclosures),
            "detail_budget": budget,
            "detail_hits": len(details),
            "window_hours": int(window_hours),
            "items": len(items),
            "used_fallback": used_fallback,
            "elapsed_ms": int((kas.time.time() - started) * 1000),
        },
    )
    # endregion
    return items


def _fetch_kap_member_disclosures_for(symbol: str, year: int) -> List[Dict[str, Any]]:
    """Fetch recent financial disclosures for a single BIST company using the
    `listCompanyExcelMembers` endpoint which is NOT WAF-blocked.

    Returns [] on any error so the aggregator can continue for other companies.
    """
    import urllib.error
    import urllib.request

    kas = _kas()
    oid = kas._KAP_MEMBER_OID_CACHE.get(symbol)
    if not oid:
        shared_oid = kas._shared_cache_get_dict(f"api:kap:member-oid:{symbol}:v1")
        if shared_oid and shared_oid.get("oid"):
            oid = str(shared_oid.get("oid") or "").strip()
            if oid:
                kas._KAP_MEMBER_OID_CACHE[symbol] = oid
    headers = {
        "Accept": "application/json",
        "Accept-Language": "tr",
        "User-Agent": kas._KAP_DEFAULT_HEADERS["User-Agent"],
    }
    if not oid:
        try:
            req = urllib.request.Request(
                f"https://www.kap.org.tr/tr/api/member/filter/{symbol}",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                rows = kas.json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(rows, list) and rows:
                oid = str(rows[0].get("mkkMemberOid") or "").strip()
                if oid:
                    kas._KAP_MEMBER_OID_CACHE[symbol] = oid
                    kas._shared_cache_set(
                        f"api:kap:member-oid:{symbol}:v1",
                        {"oid": oid},
                        ttl_seconds=24 * 60 * 60,
                    )
        except Exception:
            return []
    if not oid:
        return []
    try:
        req = urllib.request.Request(
            f"https://www.kap.org.tr/tr/api/financialTable/listCompanyExcelMembers/{oid}/{year}/T",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = kas.json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _resolve_kap_member_oid(symbol: str) -> Optional[str]:
    kas = _kas()
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return None

    if not kas._KAP_MEMBER_OID_WARMED_FROM_CACHE:
        kas._KAP_MEMBER_OID_WARMED_FROM_CACHE = True
        kap_cache_dir = kas.CONFIG.paths.processed_dir / "kap_cache"
        if kap_cache_dir.exists():
            for cache_file in kap_cache_dir.glob("*.json"):
                try:
                    with cache_file.open("r", encoding="utf-8") as handle:
                        payload = kas.json.load(handle)
                except Exception:
                    continue
                cache_symbol = str(payload.get("stock_code") or cache_file.stem).strip().upper()
                cache_oid = str(payload.get("member_oid") or "").strip()
                if cache_symbol and cache_oid and cache_symbol not in kas._KAP_MEMBER_OID_CACHE:
                    kas._KAP_MEMBER_OID_CACHE[cache_symbol] = cache_oid

    oid = kas._KAP_MEMBER_OID_CACHE.get(normalized)
    if oid:
        return oid
    shared_oid = kas._shared_cache_get_dict(f"api:kap:member-oid:{normalized}:v1")
    if shared_oid and shared_oid.get("oid"):
        oid = str(shared_oid.get("oid") or "").strip()
        if oid:
            kas._KAP_MEMBER_OID_CACHE[normalized] = oid
            return oid
    if kas._shared_cache_get_dict(f"api:kap:member-oid-miss:{normalized}:v1"):
        return None

    now = kas.time.time()
    negative_ts = kas._KAP_MEMBER_OID_NEGATIVE_CACHE.get(normalized, 0.0)
    if negative_ts and now - negative_ts < kas._KAP_MEMBER_OID_NEGATIVE_TTL:
        return None

    import urllib.parse
    import urllib.request

    headers = {
        "Accept": "application/json",
        "Accept-Language": "tr",
        "User-Agent": kas._KAP_DEFAULT_HEADERS["User-Agent"],
    }
    try:
        encoded_symbol = urllib.parse.quote(normalized, safe="")
        req = urllib.request.Request(
            f"https://www.kap.org.tr/tr/api/member/filter/{encoded_symbol}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            rows = kas.json.loads(resp.read().decode("utf-8", errors="ignore"))
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate = str(row.get("mkkMemberOid") or "").strip()
                if candidate:
                    kas._KAP_MEMBER_OID_CACHE[normalized] = candidate
                    kas._KAP_MEMBER_OID_NEGATIVE_CACHE.pop(normalized, None)
                    kas._shared_cache_set(
                        f"api:kap:member-oid:{normalized}:v1",
                        {"oid": candidate},
                        ttl_seconds=24 * 60 * 60,
                    )
                    return candidate
    except Exception:
        pass

    kas._KAP_MEMBER_OID_NEGATIVE_CACHE[normalized] = now
    kas._shared_cache_set(
        f"api:kap:member-oid-miss:{normalized}:v1",
        {"miss": True},
        ttl_seconds=kas._KAP_MEMBER_OID_NEGATIVE_TTL,
    )
    return None


def _kap_logo_payload_for_symbol(symbol: str) -> Dict[str, Optional[str]]:
    kas = _kas()
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return {"logo_url": None, "logo_source": None}

    now = kas.time.time()
    cached = kas._MARKET_KAP_LOGO_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < kas._MARKET_KAP_LOGO_CACHE_TTL:
        return dict(cached.get("data") or {"logo_url": None, "logo_source": None})
    shared_key = f"api:kap:logo:{normalized}:v1"
    shared_cached = kas._shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        kas._MARKET_KAP_LOGO_CACHE[normalized] = {"_ts": now, "data": shared_cached}
        return dict(shared_cached or {"logo_url": None, "logo_source": None})

    oid = kas._resolve_kap_member_oid(normalized)
    data = {
        "logo_url": f"https://www.kap.org.tr/tr/api/member/logo/{oid}" if oid else None,
        "logo_source": "kap" if oid else None,
    }
    kas._MARKET_KAP_LOGO_CACHE[normalized] = {"_ts": now, "data": data}
    kas._shared_cache_set(shared_key, data, ttl_seconds=kas._MARKET_KAP_LOGO_CACHE_TTL)
    return dict(data)


def _empty_logo_payload() -> Dict[str, Optional[str]]:
    return {"logo_url": None, "logo_source": None}


def _synth_quarter_publish_dt(year: int, period: int) -> Optional[datetime]:
    """Approximate publish date from year+period when real publishDate absent."""
    kas = _kas()
    approximate = {1: (5, 15), 2: (8, 15), 3: (11, 15), 4: (3, 15)}
    pair = approximate.get(int(period or 0))
    if not pair:
        return None
    month, day = pair
    y = int(year or 0) + (1 if int(period or 0) == 4 else 0)
    try:
        return kas.datetime(y, month, day)
    except ValueError:
        return None


def _fetch_kap_member_feed(
    *,
    max_companies: int = 40,
    max_items: int = 160,
) -> List[Dict[str, Any]]:
    """Aggregate latest financial disclosures across BIST100 via working KAP endpoints.

    Returns feed items with best-effort `published_at` sourced from the local
    `kap_cache` (exact) when available, otherwise synthesized from year+period.
    Cached for `_KAP_MEMBER_FEED_TTL` seconds.
    """
    import concurrent.futures
    from app.kap_service import BIST100_SYMBOLS

    kas = _kas()
    now = kas.time.time()
    cache = kas._KAP_MEMBER_FEED_CACHE
    if cache.get("items") and (now - cache.get("ts", 0)) < kas._KAP_MEMBER_FEED_TTL:
        return list(cache["items"])
    shared_key = f"api:kap:member-feed:max={max_companies}:items={max_items}:v1"
    shared_cached = kas._shared_cache_get_dict(shared_key)
    if shared_cached is not None:
        items = [dict(row) for row in list(shared_cached.get("items") or []) if isinstance(row, dict)]
        if items:
            cache["items"] = items
            cache["ts"] = now
            return list(items)

    started = kas.time.time()
    symbols = list(BIST100_SYMBOLS[:max_companies])
    current_year = kas.datetime.now(kas.timezone.utc).year

    # Pre-load kap_cache publish dates keyed by disclosure_index for exact timestamps.
    cache_dir = kas.CONFIG.paths.processed_dir / "kap_cache"
    cache_publish: Dict[int, Dict[str, Any]] = {}
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            try:
                with cache_file.open("r", encoding="utf-8") as handle:
                    payload = kas.json.load(handle)
            except Exception:
                continue
            quarters = payload.get("quarters")
            if not isinstance(quarters, list):
                continue
            for q in quarters:
                if not isinstance(q, dict):
                    continue
                idx = q.get("disclosure_index")
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                cache_publish[idx_int] = {
                    "publish_date": q.get("publish_date"),
                    "quarter": q.get("quarter"),
                    "title": q.get("title"),
                }

    # Warm the OID cache from kap_cache files where available so the first
    # request doesn't pay a full `member/filter` lookup sweep.
    if not kas._KAP_MEMBER_OID_CACHE:
        kap_cache_dir = kas.CONFIG.paths.processed_dir / "kap_cache"
        if kap_cache_dir.exists():
            for cache_file in kap_cache_dir.glob("*.json"):
                try:
                    with cache_file.open("r", encoding="utf-8") as handle:
                        payload = kas.json.load(handle)
                except Exception:
                    continue
                symbol = str(payload.get("stock_code") or cache_file.stem).strip().upper()
                oid = str(payload.get("member_oid") or "").strip()
                if symbol and oid:
                    kas._KAP_MEMBER_OID_CACHE[symbol] = oid

    def _gather(year_tasks: List[tuple[str, int]]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(kas._fetch_kap_member_disclosures_for, sym, yr): (sym, yr)
                for sym, yr in year_tasks
            }
            for fut in concurrent.futures.as_completed(futures):
                sym, _ = futures[fut]
                try:
                    rows = fut.result()
                except Exception:
                    rows = []
                for row in rows:
                    row["_symbol"] = sym
                    collected.append(row)
        return collected

    current_tasks = [(sym, current_year) for sym in symbols]
    raw_rows: List[Dict[str, Any]] = _gather(current_tasks)
    # Backfill to the previous year only if current-year yield is thin.
    if len({row.get("_symbol") for row in raw_rows}) < max(8, len(symbols) // 4):
        raw_rows.extend(_gather([(sym, current_year - 1) for sym in symbols]))

    seen: set[int] = set()
    items: List[Dict[str, Any]] = []
    for row in raw_rows:
        try:
            idx = int(row.get("disclosureIndex") or 0)
        except Exception:
            continue
        if not idx or idx in seen:
            continue
        seen.add(idx)

        symbol = str(row.get("_symbol") or row.get("stockCode") or "").strip().upper()
        year = int(row.get("year") or 0)
        period = int(row.get("period") or 0)

        cache_meta = cache_publish.get(idx)
        publish_dt: Optional[datetime] = None
        quarter_label = ""
        title = ""
        if cache_meta:
            publish_dt = kas._parse_kap_publish_date(cache_meta.get("publish_date"))
            quarter_label = str(cache_meta.get("quarter") or "").strip()
            title = str(cache_meta.get("title") or "").strip()
        if publish_dt is None:
            publish_dt = kas._synth_quarter_publish_dt(year, period)
        if publish_dt is None:
            continue
        if not quarter_label and year and period:
            quarter_label = f"{year}Q{period}"
        if not title:
            title = "Finansal Rapor"

        items.append(
            {
                "id": f"{symbol}-{idx}",
                "disclosure_id": str(idx),
                "source": "Finansal Rapor",
                "symbol": symbol,
                "stock_codes": [symbol],
                "title": f"{title}{' - ' + quarter_label if quarter_label else ''}",
                "subject": "Finansal Rapor",
                "published_at": publish_dt.isoformat(),
                "category": "finansal_rapor",
                "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx}",
            }
        )

    # Items without a kap_cache exact publish_date share the same synthesized
    # day; fall back to disclosureIndex as a tiebreaker so ordering matches
    # KAP's own allocation sequence (higher index = more recent).
    items.sort(
        key=lambda row: (
            row.get("published_at") or "",
            int(str(row.get("id") or "-0").rsplit("-", 1)[-1] or 0) if "-" in str(row.get("id") or "") else 0,
        ),
        reverse=True,
    )
    items = items[:max_items]

    # region agent log
    kas._debug_log(
        "H11",
        "app/api.py:_fetch_kap_member_feed",
        "KAP member feed built",
        {
            "symbol_count": len(symbols),
            "raw_rows": len(raw_rows),
            "unique_items": len(items),
            "elapsed_ms": int((kas.time.time() - started) * 1000),
        },
    )
    # endregion
    cache["items"] = items
    cache["ts"] = now
    kas._shared_cache_set(shared_key, {"items": items}, ttl_seconds=kas._KAP_MEMBER_FEED_TTL)
    return list(items)


def _legacy_market_flow_payload(
    limit: int = 40,
    category: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    kas = _kas()
    # Use the expanded budget in the cache key, then trim the shared result per
    # caller.  This prevents a small-list request from poisoning a larger one.
    effective_budget = max(
        kas._VYK_DEFAULT_DETAIL_BUDGET,
        min(kas._VYK_DEFAULT_DETAIL_BUDGET_MAX, int(limit)),
    )
    shared_key = f"api:kap:market-flow:category={category or ''}:budget={effective_budget}:v3"
    payload, cache_status, stale, refresh_pending = kas._shared_swr_payload(
        cache_key=shared_key,
        factory=lambda: kas._build_market_flow_payload(
            limit=effective_budget,
            category=category,
            force_refresh=True,
        ),
        fresh_ttl_seconds=kas._FLOW_CACHE_TTL,
        stale_ttl_seconds=max(kas._MARKET_SWR_STALE_TTL_SECONDS, kas._FLOW_CACHE_TTL * 2),
        local_cache=kas._FLOW_CACHE,
        local_key=f"all::{category or ''}::b{effective_budget}",
        force_revalidate=force_refresh,
    )
    if payload is None:
        raise kas.HTTPException(status_code=503, detail="KAP piyasa akışı yenileniyor. Lütfen kısa süre sonra tekrar deneyin.")
    response = kas._with_market_cache_metadata(
        payload,
        cache_status=cache_status,
        stale=stale,
        refresh_pending=refresh_pending,
    )
    response["items"] = list(response.get("items") or [])[:limit]
    return response


def _persisted_market_flow_payload(
    *,
    limit: int,
    category: Optional[str],
    before: Optional[str],
    after: Optional[str],
) -> Optional[Dict[str, Any]]:
    kas = _kas()
    try:
        page = kas._kap_flow_store.read_flow_page(
            limit=limit,
            before=before,
            after=after,
            category=category,
        )
    except ValueError as exc:
        raise kas.HTTPException(status_code=400, detail=str(exc)) from exc
    if page is None:
        return None
    kas._maybe_schedule_kap_flow_live_refresh()
    response = dict(page)
    response.setdefault("source", "kap_flow_store")
    response.setdefault("as_of", None)
    response.setdefault("degraded_mode", False)
    response.setdefault("multi_category", True)
    response.setdefault("warning", None)
    response.setdefault("public_error", None)
    response["cache_status"] = "shared_hit"
    refresh_metadata = kas._kap_flow_refresh_metadata()
    response.update(refresh_metadata)
    response["stale"] = (
        refresh_metadata.get("refresh_age_seconds") is None
        or int(refresh_metadata["refresh_age_seconds"]) > kas._KAP_FLOW_REFRESH_STALE_AFTER_SECONDS
    )
    response["warning"] = response.get("warning") or kas._kap_flow_refresh_warning(refresh_metadata)
    return response


def _market_flow_payload(
    limit: int = 40,
    category: Optional[str] = None,
    *,
    force_refresh: bool = False,
    before: Optional[str] = None,
    after: Optional[str] = None,
) -> Dict[str, Any]:
    kas = _kas()
    # Once the durable feed has rows, reads never call KAP.  The first cold
    # read seeds the store through the legacy source path so local development
    # and an unprimed deployment remain usable before the first cron run.
    store_requested = bool(before or after) or kas.database_enabled() or kas._kap_flow_store.store_is_configured()
    if store_requested:
        persisted = kas._persisted_market_flow_payload(
            limit=limit,
            category=category,
            before=before,
            after=after,
        )
        if persisted is not None:
            return persisted
        if before or after:
            return {
                "items": [],
                "source": "kap_flow_store",
                "as_of": None,
                "has_more": False,
                "next_cursor": None,
                "latest_cursor": None,
                "last_successful_refresh": None,
                "refresh_status": "empty",
                "degraded_mode": False,
                "multi_category": True,
                "warning": None,
                "public_error": None,
                "cache_status": "miss",
                "stale": False,
                "refresh_pending": False,
            }
        seed_payload = kas._legacy_market_flow_payload(
            limit=kas._KAP_FLOW_STORAGE_LIMIT,
            category=None,
            force_refresh=force_refresh,
        )
        if seed_payload.get("items"):
            kas._kap_flow_store.persist_flow_items(
                list(seed_payload.get("items") or []),
                source=str(seed_payload.get("source") or "kap"),
                as_of=seed_payload.get("as_of"),
            )
            persisted = kas._persisted_market_flow_payload(
                limit=limit,
                category=category,
                before=before,
                after=after,
            )
            if persisted is not None:
                return persisted
            if category:
                empty_category_payload = dict(seed_payload)
                empty_category_payload.update(
                    {
                        "items": [],
                        "has_more": False,
                        "next_cursor": None,
                        "latest_cursor": None,
                        "cache_status": "miss",
                        "stale": False,
                        "refresh_pending": False,
                    }
                )
                return empty_category_payload
        return seed_payload
    return kas._legacy_market_flow_payload(limit=limit, category=category, force_refresh=force_refresh)


def _refresh_kap_flow_store(*, fast: bool = False) -> Dict[str, Any]:
    """Fetch the current KAP feed once and publish it to durable storage."""

    kas = _kas()
    backend = kas._get_cache()
    with backend.lock("kap-flow-refresh", timeout=max(0.001, kas._KAP_FLOW_REFRESH_LOCK_TIMEOUT_SECONDS)) as acquired:
        if not acquired:
            return {
                "status": "already_running",
                "refresh_status": "running",
                "source": "kap_flow_store",
            }

        refresh_started_at = kas.datetime.now(kas.timezone.utc).isoformat()
        kas._kap_flow_store.update_flow_status(
            refresh_status="running",
            refresh_pending=True,
            last_refresh_attempt=refresh_started_at,
            last_refresh_error=None,
        )
        try:
            fast_limit = max(60, int(kas.os.getenv("RAGFIN_KAP_VYK_FAST_REFRESH_LIMIT", "150")))
            payload = kas._build_market_flow_payload(
                limit=fast_limit if fast else kas._VYK_DEFAULT_DETAIL_BUDGET_MAX,
                category=None,
                force_refresh=True,
                include_category_backfill=not fast,
            )
            items = list(payload.get("items") or [])
            if not items:
                kas._kap_flow_store.update_flow_status(
                    refresh_status="empty",
                    refresh_pending=False,
                    last_refresh_attempt=refresh_started_at,
                    last_refresh_error=payload.get("warning"),
                )
                return {
                    "status": "empty",
                    "refresh_status": "empty",
                    "source": payload.get("source") or "kap_flow_store",
                    "as_of": payload.get("as_of"),
                    "stored_count": 0,
                    "warning": payload.get("warning"),
                }

            result = kas._kap_flow_store.persist_flow_items(
                items,
                source=str(payload.get("source") or "kap"),
                as_of=payload.get("as_of"),
            )
            return {
                "status": "ok",
                **result,
                "received_count": len(items),
                "degraded_mode": bool(payload.get("degraded_mode")),
                "warning": payload.get("warning"),
            }
        except Exception as exc:
            kas._kap_flow_store.update_flow_status(
                refresh_status="failed",
                refresh_pending=False,
                last_refresh_attempt=refresh_started_at,
                last_refresh_error=str(exc),
            )
            raise


def _build_market_flow_payload(
    limit: int = 40,
    category: Optional[str] = None,
    *,
    force_refresh: bool = False,
    include_category_backfill: bool = True,
) -> Dict[str, Any]:
    kas = _kas()
    started = kas.time.time()
    # Kullanici 'kac kayit' ayarini UI'dan degistirince backend'in VYK detay
    # butcesini de ona gore genisletmek istiyoruz; ayni zamanda cache'i bu
    # butceyle kademeli tutuyoruz ki kucuk secimle doldurulup buyukte kirilmasin.
    effective_budget = max(
        kas._VYK_DEFAULT_DETAIL_BUDGET,
        min(kas._VYK_DEFAULT_DETAIL_BUDGET_MAX, int(limit)),
    )
    effective_pages = max(1, min(10, (effective_budget + 49) // 50))
    # Resmi KAP VYK akisi: credential'lar varsa en oncelikli kaynak.
    # UYARI: Kullanici kap.org.tr uzerinden canli veri istedigi icin VYK akisi gecici olarak devre disi birakildi.
    vyk_items: List[Dict[str, Any]] = []
    # vyk_items = _fetch_kap_vyk_feed(
    #     window_hours=_VYK_DEFAULT_WINDOW_HOURS,
    #     list_pages=effective_pages,
    #     detail_budget=effective_budget,
    # )

    # VYK boslarsa ya da credential yoksa resmi sitesi olan kap.org.tr uzerinden deneme yap.
    public_items: List[Dict[str, Any]] = []
    local_items: List[Dict[str, Any]] = []
    if not vyk_items:
        public_items = kas._fetch_kap_public_disclosures(
            max_items=limit,
            include_category_backfill=include_category_backfill,
        )
        if not public_items:
            local_items = kas._local_flow_items_from_cache()

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in vyk_items + public_items + local_items:
        key = item.get("id") or ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(item)

    if category:
        merged = [row for row in merged if row.get("category") == category]

    merged.sort(key=lambda row: row.get("published_at") or "", reverse=True)

    public_error: Optional[str] = None
    if vyk_items:
        source = "kap_vyk"
        degraded = False
        multi_category_available = True
        warning: Optional[str] = None
    elif public_items:
        source = "kap_public_website"
        degraded = False
        multi_category_available = True
        warning = None
    else:
        source = "local_cache"
        degraded = True
        multi_category_available = False
        warning = (
            "Canlı KAP akışı şu an ulaşılamıyor; yalnızca yerel önbellekteki "
            "son finansal raporlar gösteriliyor."
        )

    data = {
        "items": merged[:kas._KAP_FLOW_STORAGE_LIMIT],
        "source": source,
        "degraded_mode": degraded,
        "multi_category": multi_category_available,
        "warning": warning,
        "public_error": public_error,
        "as_of": kas.datetime.now(kas.timezone.utc).isoformat(),
    }
    # region agent log
    kas._debug_log(
        "H2",
        "app/api.py:_market_flow_payload",
        "Market flow payload built",
        {
            "limit": limit,
            "category": category,
            "force_refresh": force_refresh,
            "effective_budget": effective_budget,
            "vyk_items": len(vyk_items),
            "local_items": len(local_items),
            "merged_items": len(merged),
            "source": source,
            "degraded_mode": degraded,
            "multi_category": multi_category_available,
            "elapsed_ms": int((kas.time.time() - started) * 1000),
        },
    )
    # endregion
    return data
