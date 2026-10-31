"""Process-local L1 cache for JSON documents mirrored in Supabase.

A bounded LRU keeps parsed payloads in memory while their file signature and
read TTL remain valid. Callers keep their own data freshness rules (for
example KAP's ``fetched_at`` and live-disclosure checks).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from app.database import read_json_cache

LOGGER = logging.getLogger(__name__)

JSON_CACHE_LOCAL_READ_TTL_SECONDS = max(
    0,
    int(os.getenv("RAGFIN_JSON_CACHE_LOCAL_READ_TTL_SECONDS", "60")),
)
JSON_CACHE_METRICS_FLUSH_SECONDS = max(
    1.0,
    float(os.getenv("RAGFIN_JSON_CACHE_METRICS_FLUSH_SECONDS", "300")),
)
JSON_CACHE_READ_STATE_MAX_ENTRIES = 4096
JSON_CACHE_PAYLOAD_CACHE_MAX_ENTRIES = 256
JSON_CACHE_PAYLOAD_CACHE_MAX_BYTES = 16 * 1024 * 1024
_JSON_CACHE_TELEMETRY_FIELDS = (
    "local_hits",
    "coalesced_hits",
    "remote_reads",
    "remote_hits",
    "remote_misses",
    "fallback_local",
    "approx_remote_payload_bytes",
)
_JSON_CACHE_TELEMETRY_PREFIXES = ("funds_cache", "kap_cache", "other")

_JSON_CACHE_READ_STATE: Dict[str, Tuple[float, Optional[Tuple[int, int]]]] = {}
_JSON_CACHE_READ_STATE_LOCK = threading.RLock()


@dataclass(frozen=True)
class _JsonPayloadCacheEntry:
    read_at: float
    signature: Optional[Tuple[int, int]]
    payload: Optional[Dict[str, Any]]
    file_bytes: int


_JSON_CACHE_PAYLOAD_CACHE: OrderedDict[str, _JsonPayloadCacheEntry] = OrderedDict()
_JSON_CACHE_PAYLOAD_CACHE_BYTES = 0
_JSON_CACHE_PAYLOAD_CACHE_LOCK = threading.RLock()
_JSON_CACHE_REMOTE_READ_LOCKS = tuple(threading.RLock() for _ in range(16))
_JSON_CACHE_METRICS_LOCK = threading.Lock()
_JSON_CACHE_METRICS_SINCE = time.monotonic()
_JSON_CACHE_METRICS: Dict[str, int] = {
    "local_hits": 0,
    "coalesced_hits": 0,
    "remote_reads": 0,
    "remote_hits": 0,
    "remote_misses": 0,
    "fallback_local": 0,
    "remote_bytes": 0,
}
_JSON_CACHE_METRICS_BY_READER: Dict[str, Dict[str, int]] = {}
_JSON_CACHE_TELEMETRY_STARTED = time.monotonic()
_JSON_CACHE_TELEMETRY_TOTALS = {key: 0 for key in _JSON_CACHE_TELEMETRY_FIELDS}
_JSON_CACHE_TELEMETRY_BY_PREFIX = {
    prefix: {key: 0 for key in _JSON_CACHE_TELEMETRY_FIELDS}
    for prefix in _JSON_CACHE_TELEMETRY_PREFIXES
}


def json_cache_key_prefix(path: Path) -> str:
    """Return a bounded path family, never the actual database cache key."""

    parts = [str(part).replace("\\", "/").strip("/").lower() for part in path.parts]
    try:
        processed_index = parts.index("processed")
    except ValueError:
        return "other"
    if processed_index + 1 >= len(parts):
        return "other"
    prefix = parts[processed_index + 1]
    return prefix if prefix in {"funds_cache", "kap_cache"} else "other"


def _record_json_cache_telemetry_locked(prefix: str, **increments: int) -> None:
    family = prefix if prefix in _JSON_CACHE_TELEMETRY_BY_PREFIX else "other"
    for key, value in increments.items():
        if key not in _JSON_CACHE_TELEMETRY_TOTALS:
            continue
        amount = max(0, int(value))
        _JSON_CACHE_TELEMETRY_TOTALS[key] += amount
        _JSON_CACHE_TELEMETRY_BY_PREFIX[family][key] += amount


def json_cache_metrics_status() -> Dict[str, Any]:
    """Return process-lifetime, bounded-cardinality cache counters for health."""

    now = time.monotonic()
    with _JSON_CACHE_METRICS_LOCK:
        return {
            "uptime_seconds": round(now - _JSON_CACHE_TELEMETRY_STARTED, 1),
            "totals": dict(_JSON_CACHE_TELEMETRY_TOTALS),
            "by_key_prefix": {
                prefix: dict(metrics)
                for prefix, metrics in _JSON_CACHE_TELEMETRY_BY_PREFIX.items()
                if any(metrics.values())
            },
            "payload_bytes_are_approximate": True,
        }


def _reader_metrics_locked(reader: str) -> Dict[str, int]:
    normalized_reader = reader if reader in {"funds", "kap"} else "other"
    return _JSON_CACHE_METRICS_BY_READER.setdefault(
        normalized_reader,
        {key: 0 for key in _JSON_CACHE_METRICS},
    )


def json_cache_file_signature(path: Path) -> Optional[Tuple[int, int]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
    return int(mtime_ns), int(stat.st_size)


def json_cache_read_is_fresh(path: Path, *, ttl_seconds: Optional[int] = None) -> bool:
    ttl = JSON_CACHE_LOCAL_READ_TTL_SECONDS if ttl_seconds is None else max(0, int(ttl_seconds))
    if ttl <= 0:
        return False
    key = str(path)
    signature = json_cache_file_signature(path)
    return _json_cache_read_timestamp(key, signature, ttl) is not None


def _json_cache_read_timestamp(
    key: str,
    signature: Optional[Tuple[int, int]],
    ttl_seconds: int,
) -> Optional[float]:
    if ttl_seconds <= 0:
        return None
    with _JSON_CACHE_READ_STATE_LOCK:
        state = _JSON_CACHE_READ_STATE.get(key)
    if state is None or state[1] != signature:
        return None
    if time.monotonic() - state[0] > ttl_seconds:
        return None
    return state[0]


def _remove_json_cache_payload_locked(key: str) -> None:
    global _JSON_CACHE_PAYLOAD_CACHE_BYTES
    entry = _JSON_CACHE_PAYLOAD_CACHE.pop(key, None)
    if entry is not None:
        _JSON_CACHE_PAYLOAD_CACHE_BYTES = max(0, _JSON_CACHE_PAYLOAD_CACHE_BYTES - entry.file_bytes)


def _get_json_cache_payload(
    key: str,
    signature: Optional[Tuple[int, int]],
    *,
    ttl_seconds: int,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if ttl_seconds <= 0:
        return False, None
    read_at = _json_cache_read_timestamp(key, signature, ttl_seconds)
    with _JSON_CACHE_PAYLOAD_CACHE_LOCK:
        entry = _JSON_CACHE_PAYLOAD_CACHE.get(key)
        if entry is None:
            return False, None
        if (
            read_at is None
            or entry.signature != signature
            or time.monotonic() - entry.read_at > ttl_seconds
        ):
            _remove_json_cache_payload_locked(key)
            return False, None
        _JSON_CACHE_PAYLOAD_CACHE.move_to_end(key)
        return True, entry.payload


def _remember_json_cache_payload(
    key: str,
    payload: Optional[Dict[str, Any]],
    signature: Optional[Tuple[int, int]],
    read_at: float,
    *,
    ttl_seconds: int,
    copy_payload: bool = False,
) -> None:
    global _JSON_CACHE_PAYLOAD_CACHE_BYTES
    max_entries = max(0, int(JSON_CACHE_PAYLOAD_CACHE_MAX_ENTRIES))
    max_bytes = max(0, int(JSON_CACHE_PAYLOAD_CACHE_MAX_BYTES))
    file_bytes = max(0, int(signature[1])) if signature is not None else 0
    if (
        ttl_seconds <= 0
        or max_entries <= 0
        or max_bytes <= 0
        or file_bytes > max_bytes
    ):
        with _JSON_CACHE_PAYLOAD_CACHE_LOCK:
            _remove_json_cache_payload_locked(key)
        return

    try:
        stored_payload = copy.deepcopy(payload) if copy_payload and payload is not None else payload
    except Exception:
        LOGGER.debug("Could not retain JSON cache payload in memory: %s", key, exc_info=True)
        with _JSON_CACHE_PAYLOAD_CACHE_LOCK:
            _remove_json_cache_payload_locked(key)
        return

    with _JSON_CACHE_PAYLOAD_CACHE_LOCK:
        _remove_json_cache_payload_locked(key)
        _JSON_CACHE_PAYLOAD_CACHE[key] = _JsonPayloadCacheEntry(
            read_at=read_at,
            signature=signature,
            payload=stored_payload,
            file_bytes=file_bytes,
        )
        _JSON_CACHE_PAYLOAD_CACHE_BYTES += file_bytes
        while (
            len(_JSON_CACHE_PAYLOAD_CACHE) > max_entries
            or _JSON_CACHE_PAYLOAD_CACHE_BYTES > max_bytes
        ):
            oldest_key = next(iter(_JSON_CACHE_PAYLOAD_CACHE))
            _remove_json_cache_payload_locked(oldest_key)


def _remember_json_cache_read_state(
    path: Path,
) -> Tuple[float, Optional[Tuple[int, int]]]:
    key = str(path)
    state = (time.monotonic(), json_cache_file_signature(path))
    evicted_key: Optional[str] = None
    with _JSON_CACHE_READ_STATE_LOCK:
        _JSON_CACHE_READ_STATE.pop(key, None)
        if len(_JSON_CACHE_READ_STATE) >= JSON_CACHE_READ_STATE_MAX_ENTRIES:
            evicted_key = next(iter(_JSON_CACHE_READ_STATE))
            _JSON_CACHE_READ_STATE.pop(evicted_key, None)
        _JSON_CACHE_READ_STATE[key] = state
    if evicted_key is not None:
        with _JSON_CACHE_PAYLOAD_CACHE_LOCK:
            _remove_json_cache_payload_locked(evicted_key)
    return state


def remember_json_cache_read(path: Path) -> None:
    _remember_json_cache_read_state(path)


def json_cache_metrics_snapshot_locked(
    now: float,
) -> Optional[Tuple[float, Dict[str, int], Dict[str, Dict[str, int]]]]:
    """Return and reset the counter window when it is due, else ``None``."""

    global _JSON_CACHE_METRICS_SINCE
    elapsed = now - _JSON_CACHE_METRICS_SINCE
    if elapsed < JSON_CACHE_METRICS_FLUSH_SECONDS:
        return None
    snapshot = dict(_JSON_CACHE_METRICS) if any(_JSON_CACHE_METRICS.values()) else None
    if snapshot is not None:
        for key in _JSON_CACHE_METRICS:
            _JSON_CACHE_METRICS[key] = 0
    _JSON_CACHE_METRICS_SINCE = now
    reader_snapshots = {
        reader: dict(metrics)
        for reader, metrics in _JSON_CACHE_METRICS_BY_READER.items()
        if any(metrics.values())
    }
    for metrics in _JSON_CACHE_METRICS_BY_READER.values():
        for key in metrics:
            metrics[key] = 0
    return (elapsed, snapshot, reader_snapshots) if snapshot is not None else None


def log_json_cache_metrics(
    pending: Optional[Tuple[float, Dict[str, int], Dict[str, Dict[str, int]]]],
) -> None:
    if pending is None:
        return
    elapsed, snapshot, reader_snapshots = pending
    LOGGER.info(
        "json cache reads window_s=%.0f local_hits=%d coalesced_hits=%d "
        "remote_reads=%d remote_hits=%d remote_misses=%d fallback_local=%d "
        "approx_payload_kib=%.1f",
        elapsed,
        snapshot["local_hits"],
        snapshot["coalesced_hits"],
        snapshot["remote_reads"],
        snapshot["remote_hits"],
        snapshot["remote_misses"],
        snapshot["fallback_local"],
        snapshot["remote_bytes"] / 1024.0,
    )
    for reader, reader_snapshot in reader_snapshots.items():
        LOGGER.info(
            "json cache reads by_reader window_s=%.0f reader=%s local_hits=%d "
            "coalesced_hits=%d remote_reads=%d remote_hits=%d remote_misses=%d "
            "fallback_local=%d approx_payload_kib=%.1f",
            elapsed,
            reader,
            reader_snapshot["local_hits"],
            reader_snapshot["coalesced_hits"],
            reader_snapshot["remote_reads"],
            reader_snapshot["remote_hits"],
            reader_snapshot["remote_misses"],
            reader_snapshot["fallback_local"],
            reader_snapshot["remote_bytes"] / 1024.0,
        )


def record_json_cache_local_hit(
    *,
    coalesced: bool,
    reader: str,
    prefix: str = "other",
) -> None:
    now = time.monotonic()
    with _JSON_CACHE_METRICS_LOCK:
        _JSON_CACHE_METRICS["local_hits"] += 1
        reader_metrics = _reader_metrics_locked(reader)
        reader_metrics["local_hits"] += 1
        telemetry_increments = {"local_hits": 1}
        if coalesced:
            _JSON_CACHE_METRICS["coalesced_hits"] += 1
            reader_metrics["coalesced_hits"] += 1
            telemetry_increments["coalesced_hits"] = 1
        _record_json_cache_telemetry_locked(prefix, **telemetry_increments)
        pending = json_cache_metrics_snapshot_locked(now)
    log_json_cache_metrics(pending)


def record_json_cache_remote_result(
    *,
    hit: bool,
    fallback_local: bool,
    approx_bytes: int = 0,
    reader: str,
    prefix: str = "other",
) -> None:
    now = time.monotonic()
    with _JSON_CACHE_METRICS_LOCK:
        _JSON_CACHE_METRICS["remote_reads"] += 1
        reader_metrics = _reader_metrics_locked(reader)
        reader_metrics["remote_reads"] += 1
        telemetry_increments = {"remote_reads": 1}
        if hit:
            _JSON_CACHE_METRICS["remote_hits"] += 1
            _JSON_CACHE_METRICS["remote_bytes"] += max(0, int(approx_bytes))
            reader_metrics["remote_hits"] += 1
            reader_metrics["remote_bytes"] += max(0, int(approx_bytes))
            telemetry_increments["remote_hits"] = 1
            telemetry_increments["approx_remote_payload_bytes"] = max(0, int(approx_bytes))
        else:
            _JSON_CACHE_METRICS["remote_misses"] += 1
            reader_metrics["remote_misses"] += 1
            telemetry_increments["remote_misses"] = 1
            if fallback_local:
                _JSON_CACHE_METRICS["fallback_local"] += 1
                reader_metrics["fallback_local"] += 1
                telemetry_increments["fallback_local"] = 1
        _record_json_cache_telemetry_locked(prefix, **telemetry_increments)
        pending = json_cache_metrics_snapshot_locked(now)
    log_json_cache_metrics(pending)


def read_local_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def write_local_json(path: Path, payload: Dict[str, Any]) -> None:
    key = str(path)
    write_lock = _JSON_CACHE_REMOTE_READ_LOCKS[hash(key) % len(_JSON_CACHE_REMOTE_READ_LOCKS)]
    with write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp.replace(path)
        read_at, signature = _remember_json_cache_read_state(path)
        _remember_json_cache_payload(
            key,
            payload,
            signature,
            read_at,
            ttl_seconds=JSON_CACHE_LOCAL_READ_TTL_SECONDS,
            copy_payload=True,
        )


def read_json_cache_local_first(
    path: Path,
    *,
    remote_reader: Optional[Callable[[Path], Optional[Dict[str, Any]]]] = None,
    ttl_seconds: Optional[int] = None,
    reader: str = "funds",
) -> Optional[Dict[str, Any]]:
    """Read a local JSON mirror, revalidating it remotely only when stale.

    Parsed payloads are held in a bounded process-local LRU. A hit is valid
    only while both the existing read TTL and local file signature match.
    Callers must treat returned payloads as read-only and copy before mutation.
    Concurrent revalidations for the same path are coalesced with a striped
    lock.
    """

    ttl = JSON_CACHE_LOCAL_READ_TTL_SECONDS if ttl_seconds is None else max(0, int(ttl_seconds))
    read_remote = remote_reader or read_json_cache
    key_prefix = json_cache_key_prefix(path)
    metric_reader = "kap" if key_prefix == "kap_cache" else reader
    key = str(path)
    signature = json_cache_file_signature(path)
    payload_hit, cached_payload = _get_json_cache_payload(key, signature, ttl_seconds=ttl)
    if payload_hit:
        record_json_cache_local_hit(coalesced=False, reader=metric_reader, prefix=key_prefix)
        return cached_payload

    local_payload = read_local_json(path)
    signature = json_cache_file_signature(path)
    read_at = _json_cache_read_timestamp(key, signature, ttl)
    if read_at is not None:
        _remember_json_cache_payload(
            key, local_payload, signature, read_at, ttl_seconds=ttl
        )
        record_json_cache_local_hit(coalesced=False, reader=metric_reader, prefix=key_prefix)
        return local_payload

    read_lock = _JSON_CACHE_REMOTE_READ_LOCKS[hash(key) % len(_JSON_CACHE_REMOTE_READ_LOCKS)]
    with read_lock:
        signature = json_cache_file_signature(path)
        payload_hit, cached_payload = _get_json_cache_payload(key, signature, ttl_seconds=ttl)
        if payload_hit:
            record_json_cache_local_hit(coalesced=True, reader=metric_reader, prefix=key_prefix)
            return cached_payload

        local_payload = read_local_json(path)
        signature = json_cache_file_signature(path)
        read_at = _json_cache_read_timestamp(key, signature, ttl)
        if read_at is not None:
            _remember_json_cache_payload(
                key, local_payload, signature, read_at, ttl_seconds=ttl
            )
            record_json_cache_local_hit(coalesced=True, reader=metric_reader, prefix=key_prefix)
            return local_payload

        try:
            remote_payload = read_remote(path)
        except Exception:
            # app.database.read_json_cache already returns None on failures, but
            # keep this boundary defensive so a remote problem never hides the
            # local copy from callers.
            LOGGER.warning("remote JSON cache read failed; using local copy", exc_info=True)
            remote_payload = None
        if remote_payload is None:
            # Remember misses and temporary remote failures too, so a hot key
            # cannot turn into a Supabase query on every concurrent request.
            read_at, state_signature = _remember_json_cache_read_state(path)
            if state_signature == signature:
                _remember_json_cache_payload(
                    key, local_payload, state_signature, read_at, ttl_seconds=ttl
                )
            record_json_cache_remote_result(
                hit=False,
                fallback_local=local_payload is not None,
                reader=metric_reader,
                prefix=key_prefix,
            )
            return local_payload

        if remote_payload != local_payload:
            try:
                write_local_json(path, remote_payload)
            except OSError:
                LOGGER.debug("Could not mirror remote JSON cache locally: %s", path, exc_info=True)
        else:
            read_at, state_signature = _remember_json_cache_read_state(path)
            if state_signature == signature:
                _remember_json_cache_payload(
                    key, local_payload, state_signature, read_at, ttl_seconds=ttl
                )

        # Compact JSON length estimates payload size without exposing content.
        # It is not the exact Postgres wire size and is used only for telemetry.
        try:
            approx_bytes = len(
                json.dumps(
                    remote_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
        except (TypeError, ValueError):
            approx_bytes = 0
        record_json_cache_remote_result(
            hit=True,
            fallback_local=False,
            approx_bytes=approx_bytes,
            reader=metric_reader,
            prefix=key_prefix,
        )
        return remote_payload
