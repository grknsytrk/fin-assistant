"""Durable and cache-first storage helpers for the KAP market flow."""

from __future__ import annotations

import base64
import binascii
import copy
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from app.cache import get_cache
from app.database import (
    database_enabled,
    read_kap_flow_events,
    upsert_kap_flow_events,
)

LOGGER = logging.getLogger(__name__)

FLOW_HEAD_CACHE_PREFIX = "api:kap:flow:head:v1:"
FLOW_STATUS_CACHE_KEY = "api:kap:flow:status:v1"
FLOW_REDIS_STORE_KEY = "api:kap:flow:events:v1"
FLOW_HEAD_CACHE_TTL_SECONDS = int(os.getenv("RAGFIN_KAP_FLOW_HEAD_CACHE_TTL_SECONDS", "60"))
FLOW_RETAINED_EVENTS = max(100, int(os.getenv("RAGFIN_KAP_FLOW_RETAINED_EVENTS", "2000")))


def _normalize_datetime(raw: Any) -> Optional[str]:
    if isinstance(raw, datetime):
        parsed = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _event_key(item: dict[str, Any]) -> tuple[str, Optional[str]]:
    raw_id = str(item.get("id") or "").strip()
    disclosure_id = str(item.get("disclosure_id") or "").strip() or None
    if not disclosure_id and raw_id.lower().startswith("kap-"):
        disclosure_id = raw_id[4:].strip() or None
    if disclosure_id:
        return f"kap:{disclosure_id}", disclosure_id
    source = str(item.get("source") or "kap").strip().casefold() or "kap"
    return f"{source}:{raw_id or 'unknown'}", None


def normalize_flow_item(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Normalize one upstream row and attach its stable persistence key."""

    if not isinstance(item, dict):
        return None
    published_at = _normalize_datetime(item.get("published_at"))
    if not published_at:
        return None
    event_key, disclosure_id = _event_key(item)
    stock_codes = [str(code).strip().upper() for code in item.get("stock_codes") or [] if str(code).strip()]
    related_symbols = [
        str(code).strip().upper()
        for code in item.get("related_symbols") or []
        if str(code).strip()
    ]
    payload = {
        "id": str(item.get("id") or event_key),
        "source": str(item.get("source") or "KAP"),
        "symbol": str(item.get("symbol") or "").strip().upper(),
        "stock_codes": stock_codes,
        "related_symbols": related_symbols,
        "title": str(item.get("title") or "KAP Bildirimi"),
        "subject": str(item.get("subject") or "") or None,
        "published_at": published_at,
        "category": str(item.get("category") or "bildirim"),
        "kap_url": item.get("kap_url"),
    }
    if disclosure_id:
        payload["disclosure_id"] = disclosure_id
    return {
        "event_key": event_key,
        "disclosure_id": disclosure_id,
        "published_at": published_at,
        "source": payload["source"],
        "symbol": payload["symbol"],
        "stock_codes": stock_codes,
        "related_symbols": related_symbols,
        "title": payload["title"],
        "subject": payload["subject"],
        "category": payload["category"],
        "kap_url": payload["kap_url"],
        "payload": payload,
    }


def _sort_key(event: dict[str, Any]) -> tuple[str, str]:
    return (str(event.get("published_at") or ""), str(event.get("event_key") or ""))


def encode_cursor(published_at: Any, event_key: str) -> str:
    normalized = _normalize_datetime(published_at) or str(published_at or "")
    raw = json.dumps(
        {"published_at": normalized, "event_key": str(event_key)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(value: Optional[str]) -> Optional[tuple[str, str]]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        padding = "=" * (-len(text) % 4)
        payload = json.loads(base64.urlsafe_b64decode(f"{text}{padding}").decode("utf-8"))
        published_at = _normalize_datetime(payload.get("published_at"))
        event_key = str(payload.get("event_key") or "").strip()
    except (ValueError, TypeError, binascii.Error, json.JSONDecodeError):
        return None
    if not published_at or not event_key:
        return None
    return published_at, event_key


def _head_key(limit: int, category: Optional[str]) -> str:
    return f"{FLOW_HEAD_CACHE_PREFIX}{max(1, int(limit))}:{str(category or '').strip()}"


def _row_to_event(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    event_key = str(row.get("event_key") or "").strip()
    published_at = _normalize_datetime(row.get("published_at") or payload.get("published_at"))
    if not event_key or not published_at:
        return None
    item = copy.deepcopy(payload)
    item["published_at"] = published_at
    return {
        "event_key": event_key,
        "published_at": published_at,
        "payload": item,
    }


def _redis_events() -> list[dict[str, Any]]:
    cached = get_cache().get(FLOW_REDIS_STORE_KEY)
    if not isinstance(cached, list):
        return []
    result: list[dict[str, Any]] = []
    for row in cached:
        if not isinstance(row, dict):
            continue
        normalized = _row_to_event(row)
        if normalized:
            result.append(normalized)
    return result


def _write_redis_events(events: list[dict[str, Any]]) -> None:
    get_cache().set(FLOW_REDIS_STORE_KEY, events[:FLOW_RETAINED_EVENTS])


def _page_from_events(
    events: list[dict[str, Any]],
    *,
    limit: int,
    before: Optional[tuple[str, str]],
    after: Optional[tuple[str, str]],
    category: Optional[str],
) -> tuple[list[dict[str, Any]], bool]:
    filtered = sorted(events, key=_sort_key, reverse=True)
    if category:
        filtered = [row for row in filtered if row["payload"].get("category") == category]
    if before:
        filtered = [row for row in filtered if _sort_key(row) < before]
    if after:
        filtered = [row for row in filtered if _sort_key(row) > after]
    page_size = max(1, min(int(limit), 500))
    return filtered[:page_size], len(filtered) > page_size


def _build_page(
    rows: list[dict[str, Any]],
    *,
    has_more: bool,
    status: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    items = [row["payload"] for row in rows]
    return {
        "items": items,
        "has_more": bool(has_more),
        "next_cursor": (
            encode_cursor(rows[-1]["published_at"], rows[-1]["event_key"])
            if rows
            else None
        ),
        "latest_cursor": (
            encode_cursor(rows[0]["published_at"], rows[0]["event_key"])
            if rows
            else None
        ),
        "as_of": (status or {}).get("as_of") or (rows[0]["published_at"] if rows else None),
        "source": (status or {}).get("source") or "kap_flow_store",
        "last_successful_refresh": (status or {}).get("last_successful_refresh"),
        "refresh_status": (status or {}).get("refresh_status") or "unknown",
    }


def persist_flow_items(items: list[dict[str, Any]], *, source: str, as_of: Optional[str] = None) -> dict[str, Any]:
    """Persist feed rows and invalidate only the affected head cache."""

    normalized_by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        normalized = normalize_flow_item(item)
        if normalized:
            normalized_by_key[normalized["event_key"]] = normalized
    normalized = sorted(normalized_by_key.values(), key=_sort_key, reverse=True)
    database_written = False
    if normalized and database_enabled():
        try:
            upsert_kap_flow_events(normalized)
            database_written = True
        except Exception:
            LOGGER.warning("durable KAP flow upsert failed", exc_info=True)

    # Redis remains a useful local fallback when database persistence is not
    # configured, and always provides the fast head cache.
    existing = {row["event_key"]: row for row in _redis_events()}
    for row in normalized:
        existing[row["event_key"]] = {
            "event_key": row["event_key"],
            "published_at": row["published_at"],
            "payload": row["payload"],
        }
    merged = sorted(existing.values(), key=_sort_key, reverse=True)[:FLOW_RETAINED_EVENTS]
    _write_redis_events(merged)

    backend = get_cache()
    try:
        backend.delete_prefix(FLOW_HEAD_CACHE_PREFIX)
    except Exception:
        LOGGER.debug("KAP flow head cache invalidation failed", exc_info=True)

    latest = merged[0] if merged else None
    status = {
        "source": source,
        "as_of": as_of or (latest or {}).get("published_at"),
        "last_successful_refresh": datetime.now(timezone.utc).isoformat(),
        "refresh_status": "ok" if normalized else "empty",
        "stored_count": len(normalized),
        "database_written": database_written,
    }
    backend.set(FLOW_STATUS_CACHE_KEY, status)
    return status


def read_flow_page(
    *,
    limit: int,
    before: Optional[str] = None,
    after: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Read a page from Postgres, falling back to the bounded Redis store."""

    before_tuple = decode_cursor(before)
    after_tuple = decode_cursor(after)
    if before and before_tuple is None:
        raise ValueError("before cursor geçersiz")
    if after and after_tuple is None:
        raise ValueError("after cursor geçersiz")

    cache_key = _head_key(limit, category)
    if not before_tuple and not after_tuple:
        cached_head = get_cache().get(cache_key)
        if isinstance(cached_head, dict) and isinstance(cached_head.get("items"), list):
            return dict(cached_head)

    rows: list[dict[str, Any]] = []
    has_more = False
    if database_enabled():
        try:
            db_rows, has_more = read_kap_flow_events(
                limit=limit,
                before=before_tuple,
                after=after_tuple,
                category=category,
            )
            rows = [row for row in (_row_to_event(item) for item in db_rows) if row]
        except Exception:
            # A Postgres blip must not hide the bounded Redis copy from users.
            LOGGER.warning("durable KAP flow read failed; using Redis fallback", exc_info=True)

    if not rows:
        rows, has_more = _page_from_events(
            _redis_events(),
            limit=limit,
            before=before_tuple,
            after=after_tuple,
            category=category,
        )

    if not rows:
        if before_tuple or after_tuple:
            return _build_page([], has_more=False, status=get_flow_status())
        return None

    page = _build_page(rows, has_more=has_more, status=get_flow_status())
    if not before_tuple and not after_tuple:
        get_cache().set(cache_key, page, ttl_seconds=max(1, FLOW_HEAD_CACHE_TTL_SECONDS))
    return page


def get_flow_status() -> dict[str, Any]:
    cached = get_cache().get(FLOW_STATUS_CACHE_KEY)
    return dict(cached) if isinstance(cached, dict) else {}


def store_is_configured() -> bool:
    """Return whether the durable or bounded Redis feed store can serve rows."""

    # A configured Redis backend must count as available even before the first
    # refresh has written the bounded fallback list.  Otherwise the first cold
    # request would bypass persistence and every worker would independently
    # call KAP until the scheduled job happened to run.
    return database_enabled() or get_cache().name == "redis" or bool(_redis_events())


def read_flow_head(*, limit: int = 1, category: Optional[str] = None) -> dict[str, Any]:
    page = read_flow_page(limit=limit, category=category)
    if page is None:
        return {
            "latest_cursor": None,
            "as_of": None,
            "last_successful_refresh": None,
            "refresh_status": "empty",
        }
    return {
        "latest_cursor": page.get("latest_cursor"),
        "as_of": page.get("as_of"),
        "last_successful_refresh": page.get("last_successful_refresh"),
        "refresh_status": page.get("refresh_status"),
        "source": page.get("source"),
    }
