"""Process-local L1 cache for JSON documents mirrored in Supabase.

The helpers keep only a file signature and a read timestamp in memory; the
payloads stay on disk. A short TTL prevents a hot cache document from issuing
a Supabase read on every request, while callers keep their own freshness rules
(for example KAP's ``fetched_at`` and live-disclosure checks).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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

_JSON_CACHE_READ_STATE: Dict[str, Tuple[float, Optional[Tuple[int, int]]]] = {}
_JSON_CACHE_READ_STATE_LOCK = threading.RLock()
_JSON_CACHE_REMOTE_READ_LOCKS = tuple(threading.Lock() for _ in range(16))
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
    with _JSON_CACHE_READ_STATE_LOCK:
        state = _JSON_CACHE_READ_STATE.get(key)
    if state is None or time.monotonic() - state[0] > ttl:
        return False
    return state[1] == json_cache_file_signature(path)


def remember_json_cache_read(path: Path) -> None:
    key = str(path)
    state = (time.monotonic(), json_cache_file_signature(path))
    with _JSON_CACHE_READ_STATE_LOCK:
        _JSON_CACHE_READ_STATE.pop(key, None)
        if len(_JSON_CACHE_READ_STATE) >= JSON_CACHE_READ_STATE_MAX_ENTRIES:
            _JSON_CACHE_READ_STATE.pop(next(iter(_JSON_CACHE_READ_STATE)))
        _JSON_CACHE_READ_STATE[key] = state


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
            "fallback_local=%d",
            elapsed,
            reader,
            reader_snapshot["local_hits"],
            reader_snapshot["coalesced_hits"],
            reader_snapshot["remote_reads"],
            reader_snapshot["remote_hits"],
            reader_snapshot["remote_misses"],
            reader_snapshot["fallback_local"],
        )


def record_json_cache_local_hit(*, coalesced: bool, reader: str) -> None:
    now = time.monotonic()
    with _JSON_CACHE_METRICS_LOCK:
        _JSON_CACHE_METRICS["local_hits"] += 1
        reader_metrics = _reader_metrics_locked(reader)
        reader_metrics["local_hits"] += 1
        if coalesced:
            _JSON_CACHE_METRICS["coalesced_hits"] += 1
            reader_metrics["coalesced_hits"] += 1
        pending = json_cache_metrics_snapshot_locked(now)
    log_json_cache_metrics(pending)


def record_json_cache_remote_result(
    *,
    hit: bool,
    fallback_local: bool,
    approx_bytes: int = 0,
    reader: str,
) -> None:
    now = time.monotonic()
    with _JSON_CACHE_METRICS_LOCK:
        _JSON_CACHE_METRICS["remote_reads"] += 1
        reader_metrics = _reader_metrics_locked(reader)
        reader_metrics["remote_reads"] += 1
        if hit:
            _JSON_CACHE_METRICS["remote_hits"] += 1
            _JSON_CACHE_METRICS["remote_bytes"] += max(0, int(approx_bytes))
            reader_metrics["remote_hits"] += 1
            reader_metrics["remote_bytes"] += max(0, int(approx_bytes))
        else:
            _JSON_CACHE_METRICS["remote_misses"] += 1
            reader_metrics["remote_misses"] += 1
            if fallback_local:
                _JSON_CACHE_METRICS["fallback_local"] += 1
                reader_metrics["fallback_local"] += 1
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)
    remember_json_cache_read(path)


def read_json_cache_local_first(
    path: Path,
    *,
    remote_reader: Optional[Callable[[Path], Optional[Dict[str, Any]]]] = None,
    ttl_seconds: Optional[int] = None,
    reader: str = "funds",
) -> Optional[Dict[str, Any]]:
    """Read a local JSON mirror, revalidating it remotely only when stale.

    Only timestamps and file signatures are kept per cache key; payloads stay
    on disk so a cold container does not hydrate every document into memory.
    Concurrent revalidations for the same path are coalesced with a striped
    lock, keeping bookkeeping bounded even when the cache contains one file
    per fund.
    """

    ttl = JSON_CACHE_LOCAL_READ_TTL_SECONDS if ttl_seconds is None else max(0, int(ttl_seconds))
    read_remote = remote_reader or read_json_cache
    local_payload = read_local_json(path)
    if json_cache_read_is_fresh(path, ttl_seconds=ttl):
        record_json_cache_local_hit(coalesced=False, reader=reader)
        return local_payload

    key = str(path)
    read_lock = _JSON_CACHE_REMOTE_READ_LOCKS[hash(key) % len(_JSON_CACHE_REMOTE_READ_LOCKS)]
    with read_lock:
        local_payload = read_local_json(path)
        if json_cache_read_is_fresh(path, ttl_seconds=ttl):
            record_json_cache_local_hit(coalesced=True, reader=reader)
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
            remember_json_cache_read(path)
            record_json_cache_remote_result(
                hit=False,
                fallback_local=local_payload is not None,
                reader=reader,
            )
            return local_payload

        mirrored = True
        if remote_payload != local_payload:
            try:
                write_local_json(path, remote_payload)
            except OSError:
                LOGGER.debug("Could not mirror remote JSON cache locally: %s", path, exc_info=True)
                mirrored = False
        else:
            remember_json_cache_read(path)

        # The mirrored file size is a cheap proxy for the payload bytes; it is
        # never used for cache decisions and is reported only in aggregate.
        approx_bytes = 0
        if mirrored:
            signature = json_cache_file_signature(path)
            if signature is not None:
                approx_bytes = signature[1]
        record_json_cache_remote_result(
            hit=True,
            fallback_local=False,
            approx_bytes=approx_bytes,
            reader=reader,
        )
        return remote_payload
