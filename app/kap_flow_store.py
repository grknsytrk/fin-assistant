"""Durable and cache-first storage helpers for the KAP market flow."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from app.cache import BoundedMemoryCache, cache_reset_epoch, get_cache
from app.database import (
    database_enabled,
    read_kap_flow_events,
    read_kap_flow_head,
    upsert_kap_flow_events,
)

LOGGER = logging.getLogger(__name__)

_PERSISTED_FLOW_HASH_LOCK = threading.Lock()
_PERSISTED_FLOW_HASHES: dict[tuple[Any, ...], str] = {}
_PERSISTED_FLOW_HASH_MAX = 5000
_LOCAL_REDIS_EVENTS_BY_SCOPE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

FLOW_HEAD_CACHE_PREFIX = "api:kap:flow:head:v1:"
FLOW_STATUS_CACHE_KEY = "api:kap:flow:status:v1"
FLOW_REDIS_STORE_KEY = "api:kap:flow:events:v1"
# New disclosures explicitly invalidate FLOW_HEAD_CACHE_PREFIX, so unchanged
# windows can safely stay in Redis longer without polling Postgres every minute.
FLOW_HEAD_CACHE_TTL_SECONDS = int(os.getenv("RAGFIN_KAP_FLOW_HEAD_CACHE_TTL_SECONDS", "300"))
FLOW_HEAD_L1_TTL_SECONDS = max(1, int(os.getenv("RAGFIN_KAP_FLOW_HEAD_L1_TTL_SECONDS", "10")))
FLOW_PAGE_L1_TTL_SECONDS = max(1, int(os.getenv("RAGFIN_KAP_FLOW_PAGE_L1_TTL_SECONDS", "30")))
FLOW_STATUS_L1_TTL_SECONDS = max(1, int(os.getenv("RAGFIN_KAP_FLOW_STATUS_L1_TTL_SECONDS", "10")))

_FLOW_HEAD_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=32, default_ttl=FLOW_HEAD_L1_TTL_SECONDS)
_FLOW_PAGE_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=64, default_ttl=FLOW_PAGE_L1_TTL_SECONDS)
_FLOW_STATUS_L1_CACHE: BoundedMemoryCache = BoundedMemoryCache(max_entries=8, default_ttl=FLOW_STATUS_L1_TTL_SECONDS)
# Keep the Redis fallback slightly wider than the 2,000-row public window when no
# database is configured; when Postgres is active, keep a compact fallback tail.
FLOW_RETAINED_EVENTS = max(100, int(os.getenv("RAGFIN_KAP_FLOW_RETAINED_EVENTS", "2500")))
FLOW_DB_REDIS_FALLBACK_EVENTS = max(
    50, int(os.getenv("RAGFIN_KAP_FLOW_REDIS_FALLBACK_EVENTS", "300"))
)


def _retained_redis_limit() -> int:
    return min(FLOW_RETAINED_EVENTS, FLOW_DB_REDIS_FALLBACK_EVENTS) if database_enabled() else FLOW_RETAINED_EVENTS


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


ALLOWED_FLOW_CATEGORIES = frozenset(
    {
        "finansal_rapor",
        "kar_payi",
        "geri_alim",
        "genel_kurul",
        "kredi_derecelendirme",
        "surdurulebilirlik",
        "faaliyet_raporu",
        "ozel_durum",
        "diger",
        "bildirim",
    }
)


def normalize_flow_category(category: Optional[str]) -> Optional[str]:
    if category is None:
        return None
    cleaned = str(category).strip().lower()
    if not cleaned or cleaned == "all":
        return None
    if cleaned not in ALLOWED_FLOW_CATEGORIES:
        raise ValueError(f"Geçersiz akış kategorisi: {category}")
    return cleaned


def _head_key(limit: int, category: Optional[str]) -> str:
    norm_category = normalize_flow_category(category)
    return f"{FLOW_HEAD_CACHE_PREFIX}{max(1, int(limit))}:{norm_category or ''}"


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
    trimmed = list(events[: _retained_redis_limit()])
    get_cache().set(FLOW_REDIS_STORE_KEY, trimmed)
    scope = _flow_hash_scope()
    with _PERSISTED_FLOW_HASH_LOCK:
        _LOCAL_REDIS_EVENTS_BY_SCOPE[scope] = trimmed


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


def _invalidate_flow_l1_read_caches() -> None:
    _FLOW_HEAD_L1_CACHE.clear()
    _FLOW_PAGE_L1_CACHE.clear()


def _hashed_database_scope() -> str:
    raw = os.getenv("RAGFIN_DATABASE_URL", "") or os.getenv("DATABASE_URL", "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] if raw else ""


def _flow_l1_scope(backend: Any = None) -> tuple[Any, ...]:
    resolved = get_cache() if backend is None else backend
    return (
        cache_reset_epoch(),
        id(resolved),
        bool(database_enabled()),
        _hashed_database_scope(),
    )


def _flow_status_l1_key(backend: Any = None) -> tuple[Any, ...]:
    resolved = get_cache() if backend is None else backend
    return (cache_reset_epoch(), id(resolved))


def clear_persisted_flow_hashes() -> None:
    """Clear the process-local dedup table (primarily for tests)."""

    with _PERSISTED_FLOW_HASH_LOCK:
        _PERSISTED_FLOW_HASHES.clear()
        _LOCAL_REDIS_EVENTS_BY_SCOPE.clear()
    _FLOW_HEAD_L1_CACHE.clear()
    _FLOW_PAGE_L1_CACHE.clear()
    _FLOW_STATUS_L1_CACHE.clear()


def _flow_hash_scope() -> tuple[Any, ...]:
    return (
        bool(database_enabled()),
        _hashed_database_scope(),
        id(get_cache()),
    )


def _flow_item_digest(row: dict[str, Any]) -> str:
    body = json.dumps(row.get("payload") or {}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def persist_flow_items(items: list[dict[str, Any]], *, source: str, as_of: Optional[str] = None) -> dict[str, Any]:
    """Persist new or changed feed rows and invalidate head cache only when changed."""

    normalized_by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        normalized = normalize_flow_item(item)
        if normalized:
            normalized_by_key[normalized["event_key"]] = normalized
    normalized = sorted(normalized_by_key.values(), key=_sort_key, reverse=True)

    scope = _flow_hash_scope()
    previous_status = get_flow_status()
    cache_cold = not previous_status
    changed_rows: list[dict[str, Any]] = []
    row_digests: list[tuple[tuple[Any, ...], str]] = []
    with _PERSISTED_FLOW_HASH_LOCK:
        for row in normalized:
            digest = _flow_item_digest(row)
            key = (*scope, row["event_key"])
            row_digests.append((key, digest))
            if cache_cold or _PERSISTED_FLOW_HASHES.get(key) != digest:
                changed_rows.append(row)

    database_written = False
    if changed_rows and database_enabled():
        try:
            upsert_kap_flow_events(changed_rows)
            database_written = True
        except Exception:
            LOGGER.warning("durable KAP flow upsert failed", exc_info=True)

    db_ok = (not database_enabled()) or database_written
    if changed_rows and db_ok:
        with _PERSISTED_FLOW_HASH_LOCK:
            if len(_PERSISTED_FLOW_HASHES) > _PERSISTED_FLOW_HASH_MAX:
                _PERSISTED_FLOW_HASHES.clear()
            for key, digest in row_digests:
                _PERSISTED_FLOW_HASHES[key] = digest

    # Redis remains a useful local fallback when database persistence is not
    # configured, and always provides the fast head cache. Avoid re-reading or
    # rewriting the full multi-hundred-KB Redis list on every cron tick when the
    # process already synced it and nothing changed.
    with _PERSISTED_FLOW_HASH_LOCK:
        local_existing_list = _LOCAL_REDIS_EVENTS_BY_SCOPE.get(scope)
    if not changed_rows and local_existing_list is not None:
        merged = list(local_existing_list[: _retained_redis_limit()])
        redis_changed = False
    else:
        base_events = (
            local_existing_list
            if local_existing_list is not None and database_enabled()
            else _redis_events()
        )
        existing = {row["event_key"]: row for row in base_events}
        redis_changed = False
        for row in changed_rows if existing else normalized:
            event_key = row["event_key"]
            candidate = {
                "event_key": event_key,
                "published_at": row["published_at"],
                "payload": row["payload"],
            }
            if existing.get(event_key) != candidate:
                existing[event_key] = candidate
                redis_changed = True
        merged = sorted(existing.values(), key=_sort_key, reverse=True)[: _retained_redis_limit()]
        if redis_changed or (normalized and not existing):
            _write_redis_events(merged)
        else:
            with _PERSISTED_FLOW_HASH_LOCK:
                _LOCAL_REDIS_EVENTS_BY_SCOPE[scope] = list(merged)

    backend = get_cache()
    if changed_rows or redis_changed:
        _invalidate_flow_l1_read_caches()
        try:
            backend.delete_prefix(FLOW_HEAD_CACHE_PREFIX)
        except Exception:
            LOGGER.debug("KAP flow head cache invalidation failed", exc_info=True)

    latest = merged[0] if merged else (normalized[0] if normalized else None)
    status = {
        "source": source,
        "as_of": as_of or (latest or {}).get("published_at"),
        "last_successful_refresh": datetime.now(timezone.utc).isoformat(),
        "last_refresh_attempt": previous_status.get("last_refresh_attempt"),
        "last_refresh_error": None,
        "refresh_status": "ok" if normalized else "empty",
        "refresh_pending": False,
        "stored_count": len(normalized),
        "changed_count": len(changed_rows),
        "database_written": database_written,
    }
    backend.set(FLOW_STATUS_CACHE_KEY, status)
    _FLOW_STATUS_L1_CACHE[_flow_status_l1_key(backend)] = dict(status)
    return status


def read_flow_page(
    *,
    limit: int,
    before: Optional[str] = None,
    after: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Read a page from Postgres, falling back to the bounded Redis store."""

    norm_category = normalize_flow_category(category)
    before_tuple = decode_cursor(before)
    after_tuple = decode_cursor(after)
    if before and before_tuple is None:
        raise ValueError("before cursor geçersiz")
    if after and after_tuple is None:
        raise ValueError("after cursor geçersiz")

    backend = get_cache()
    cache_key = _head_key(limit, norm_category)
    l1_key = (*_flow_l1_scope(backend), cache_key)
    if not before_tuple and not after_tuple:
        cached_l1 = _FLOW_PAGE_L1_CACHE.get(l1_key)
        if isinstance(cached_l1, dict) and isinstance(cached_l1.get("items"), list):
            return dict(cached_l1)
        cached_head = backend.get(cache_key)
        if isinstance(cached_head, dict) and isinstance(cached_head.get("items"), list):
            _FLOW_PAGE_L1_CACHE[l1_key] = dict(cached_head)
            return dict(cached_head)

    rows: list[dict[str, Any]] = []
    has_more = False
    db_active = database_enabled()
    db_failed = False
    if db_active:
        try:
            db_rows, has_more = read_kap_flow_events(
                limit=limit,
                before=before_tuple,
                after=after_tuple,
                category=norm_category,
            )
            rows = [row for row in (_row_to_event(item) for item in db_rows) if row]
        except Exception:
            # A Postgres blip must not hide the bounded Redis copy from users.
            db_failed = True
            LOGGER.warning("durable KAP flow read failed; using Redis fallback", exc_info=True)

    if not db_active or db_failed:
        rows, has_more = _page_from_events(
            _redis_events(),
            limit=limit,
            before=before_tuple,
            after=after_tuple,
            category=norm_category,
        )

    if not rows:
        if before_tuple or after_tuple:
            return _build_page([], has_more=False, status=get_flow_status())
        return None

    page = _build_page(rows, has_more=has_more, status=get_flow_status())
    if not before_tuple and not after_tuple:
        backend.set(cache_key, page, ttl_seconds=max(1, FLOW_HEAD_CACHE_TTL_SECONDS))
        _FLOW_PAGE_L1_CACHE[l1_key] = dict(page)
    return page


def get_flow_status() -> dict[str, Any]:
    backend = get_cache()
    l1_key = _flow_status_l1_key(backend)
    cached_l1 = _FLOW_STATUS_L1_CACHE.get(l1_key)
    if isinstance(cached_l1, dict):
        return dict(cached_l1)
    cached = backend.get(FLOW_STATUS_CACHE_KEY)
    status = dict(cached) if isinstance(cached, dict) else {}
    _FLOW_STATUS_L1_CACHE[l1_key] = dict(status)
    return status


def update_flow_status(**updates: Any) -> dict[str, Any]:
    """Merge refresh lifecycle metadata without replacing the stored feed."""

    backend = get_cache()
    current = get_flow_status()
    current.update(updates)
    backend.set(FLOW_STATUS_CACHE_KEY, current)
    _FLOW_STATUS_L1_CACHE[_flow_status_l1_key(backend)] = dict(current)
    return current


def store_is_configured() -> bool:
    """Return whether the durable or bounded Redis feed store can serve rows."""

    # A configured Redis backend must count as available even before the first
    # refresh has written the bounded fallback list.  Otherwise the first cold
    # request would bypass persistence and every worker would independently
    # call KAP until the scheduled job happened to run.
    return database_enabled() or get_cache().name == "redis" or bool(_redis_events())


def _is_cached_flow_head(value: Any) -> bool:
    return isinstance(value, dict) and (
        bool(value.get("latest_cursor")) or value.get("refresh_status") == "empty"
    )


def read_flow_head(*, limit: int = 1, category: Optional[str] = None) -> dict[str, Any]:
    del limit  # Head queries only inspect the newest cursor.
    norm_category = normalize_flow_category(category)
    backend = get_cache()
    cache_key = f"{FLOW_HEAD_CACHE_PREFIX}cursor:{norm_category or ''}"
    l1_key = (*_flow_l1_scope(backend), cache_key)
    cached_l1 = _FLOW_HEAD_L1_CACHE.get(l1_key)
    if _is_cached_flow_head(cached_l1):
        return dict(cached_l1)
    cached = backend.get(cache_key)
    if _is_cached_flow_head(cached):
        _FLOW_HEAD_L1_CACHE[l1_key] = dict(cached)
        return dict(cached)

    head_row: Optional[dict[str, Any]] = None
    db_active = database_enabled()
    db_failed = False
    if db_active:
        try:
            db_head = read_kap_flow_head(category=norm_category)
            if db_head:
                normalized_dt = _normalize_datetime(db_head.get("published_at"))
                if normalized_dt and db_head.get("event_key"):
                    head_row = {
                        "event_key": str(db_head["event_key"]),
                        "published_at": normalized_dt,
                    }
        except Exception:
            db_failed = True
            LOGGER.warning("durable KAP flow head read failed; using Redis fallback", exc_info=True)

    if not db_active or db_failed:
        redis_rows, _has_more = _page_from_events(
            _redis_events(),
            limit=1,
            before=None,
            after=None,
            category=norm_category,
        )
        if redis_rows:
            head_row = {
                "event_key": redis_rows[0]["event_key"],
                "published_at": redis_rows[0]["published_at"],
            }

    if head_row is None:
        empty_result = {
            "latest_cursor": None,
            "as_of": None,
            "last_successful_refresh": None,
            "refresh_status": "empty",
        }
        empty_ttl = min(30, max(1, FLOW_HEAD_CACHE_TTL_SECONDS))
        backend.set(cache_key, empty_result, ttl_seconds=empty_ttl)
        _FLOW_HEAD_L1_CACHE[l1_key] = dict(empty_result)
        return empty_result

    status = get_flow_status()
    result = {
        "latest_cursor": encode_cursor(head_row["published_at"], head_row["event_key"]),
        "as_of": status.get("as_of") or head_row["published_at"],
        "last_successful_refresh": status.get("last_successful_refresh"),
        "refresh_status": status.get("refresh_status") or "unknown",
        "source": status.get("source") or "kap_flow_store",
    }
    backend.set(cache_key, result, ttl_seconds=max(1, FLOW_HEAD_CACHE_TTL_SECONDS))
    _FLOW_HEAD_L1_CACHE[l1_key] = dict(result)
    return result
