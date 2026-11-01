"""Batch refresh for KAP financial statements.

The public KAP site throttles bursts on ``notification/attachment-detail``:
each statement table is roughly 2 MB and the provider temporarily resets
connections when a single client downloads many of them in a row. This runner
is therefore deliberately conservative:

* one company at a time by default, with a small gap between requests,
* a wall-clock budget per run; the job stops cleanly as ``partial`` and the
  cursor lets the next run continue where this one stopped,
* a circuit breaker that aborts after repeated provider blocks instead of
  hammering KAP,
* ``incremental`` only re-checks companies whose cache is already current
  (schema version and requested depth). A missing, shallow, or outdated cache
  is reported as skipped and is *never* expanded into a full history download,
* ``full`` is the explicit, operator-triggered backfill that fills depth.

Cached depth above ``_KAP_STATEMENT_DEPTH_CEILING`` is normalized to that
ceiling so a refresh cannot balloon into an unbounded download.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.api_runtime import *
from app.kap_financials import _quarter_sort_key


def _kap_statement_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


_KAP_STATEMENT_ACTIVE_KEY = "api:kap:statements:active:v1"

_KAP_STATEMENT_JOB_KEY_PREFIX = "api:kap:statements:job:"

_KAP_STATEMENT_JOB_TTL_SECONDS = _kap_statement_env_int(
    "RAGFIN_KAP_STATEMENT_JOB_TTL_SECONDS", 6 * 60 * 60, minimum=1800
)

_KAP_STATEMENT_DEFAULT_BUDGET_SECONDS = _kap_statement_env_int(
    "RAGFIN_KAP_STATEMENT_DEFAULT_BUDGET_SECONDS", 5400, minimum=60
)

_KAP_STATEMENT_MAX_BUDGET_SECONDS = _kap_statement_env_int(
    "RAGFIN_KAP_STATEMENT_MAX_BUDGET_SECONDS", 10800, minimum=60
)

_KAP_STATEMENT_MAX_QUARTERS = 20

_KAP_STATEMENT_DEPTH_CEILING = 20

_KAP_STATEMENT_MAX_CONCURRENCY = 2

_KAP_STATEMENT_MAX_SYMBOLS = 800

_KAP_STATEMENT_REQUEST_GAP_SECONDS = float(
    os.getenv("RAGFIN_KAP_STATEMENT_REQUEST_GAP_SECONDS", "0.5") or 0.5
)

# Minimum start-to-start gap between KAP JSON requests while a batch job runs.
# Keep batch traffic around 9 JSON requests per minute by default so sustained
# runs stay below KAP's ~35-40 request / 5-minute sliding window.
_KAP_STATEMENT_REQUEST_INTERVAL_SECONDS = max(
    0.0,
    float(os.getenv("RAGFIN_KAP_STATEMENT_REQUEST_INTERVAL_SECONDS", "6.5") or 0.0),
)

_KAP_STATEMENT_BLOCK_COOLDOWN_SECONDS = max(
    0.0,
    float(os.getenv("RAGFIN_KAP_STATEMENT_BLOCK_COOLDOWN_SECONDS", "45") or 0.0),
)

_KAP_STATEMENT_BLOCK_STREAK_LIMIT = _kap_statement_env_int(
    "RAGFIN_KAP_STATEMENT_BLOCK_STREAK_LIMIT", 3, minimum=1
)

_KAP_STATEMENT_FAILURE_STREAK_LIMIT = _kap_statement_env_int(
    "RAGFIN_KAP_STATEMENT_FAILURE_STREAK_LIMIT", 12, minimum=1
)

_KAP_STATEMENT_MAX_ERRORS = 15

_KAP_STATEMENT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="kap-statement-refresh",
)

_SKIP_OUTCOME_FIELDS = {
    "missing": "skipped_missing_cache",
    "shallow": "skipped_shallow_cache",
    "schema_outdated": "skipped_schema",
    "skipped_ready": "skipped_ready_cache",
}

_STATEMENT_JOB_PUBLIC_FIELDS = (
    "job_id",
    "status",
    "mode",
    "universe",
    "max_quarters",
    "budget_seconds",
    "concurrency",
    "request_interval_seconds",
    "requested_at",
    "started_at",
    "finished_at",
    "duration_seconds",
    "total",
    "processed",
    "fetched",
    "updated",
    "unchanged",
    "skipped_missing_cache",
    "skipped_shallow_cache",
    "skipped_schema",
    "skipped_ready_cache",
    "failed",
    "blocked",
    "remaining",
    "cursor",
    "coverage",
    "stop_reason",
    "errors",
    "error",
)


def _kap_statement_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _kap_statement_parse_iso(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _kap_statement_job_key(job_id: str) -> str:
    return f"{_KAP_STATEMENT_JOB_KEY_PREFIX}{job_id}"


def _set_kap_statement_refresh_job(job: Dict[str, Any]) -> None:
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        return
    _shared_cache_set(
        _kap_statement_job_key(job_id),
        job,
        ttl_seconds=_KAP_STATEMENT_JOB_TTL_SECONDS,
    )


def _get_kap_statement_refresh_job(job_id: str) -> Optional[Dict[str, Any]]:
    normalized = str(job_id or "").strip()
    if not normalized:
        return None
    payload = _shared_cache_get_dict(_kap_statement_job_key(normalized))
    return payload if isinstance(payload, dict) else None


def _public_kap_statement_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {field: job.get(field) for field in _STATEMENT_JOB_PUBLIC_FIELDS}


def _active_kap_statement_refresh_job() -> Optional[Dict[str, Any]]:
    marker = _shared_cache_get_dict(_KAP_STATEMENT_ACTIVE_KEY)
    if not isinstance(marker, dict):
        return None
    job_id = str(marker.get("job_id") or "").strip()
    if not job_id:
        return None
    job = _get_kap_statement_refresh_job(job_id)
    if not isinstance(job, dict):
        return None
    if str(job.get("status") or "") not in {"queued", "running"}:
        return None
    return job


def _kap_statement_job_is_stale(job: Dict[str, Any]) -> bool:
    started = _kap_statement_parse_iso(job.get("started_at") or job.get("requested_at"))
    if started is None:
        return True
    try:
        budget = int(job.get("budget_seconds") or _KAP_STATEMENT_DEFAULT_BUDGET_SECONDS)
    except (TypeError, ValueError):
        budget = _KAP_STATEMENT_DEFAULT_BUDGET_SECONDS
    age = (datetime.now(timezone.utc) - started).total_seconds()
    return age > max(300, budget + 600)


def _owns_kap_statement_active_marker(job_id: str, owner_token: str) -> bool:
    marker = _shared_cache_get_dict(_KAP_STATEMENT_ACTIVE_KEY)
    if not isinstance(marker, dict):
        return False
    return (
        str(marker.get("job_id") or "") == str(job_id or "")
        and str(marker.get("owner_token") or "") == str(owner_token or "")
    )


def _parse_kap_statement_symbols(raw: Any) -> List[str]:
    """Accept a comma/whitespace separated symbol list from query input."""

    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        tokens = [str(item or "") for item in raw]
    else:
        tokens = str(raw or "").replace(";", ",").split(",")
    symbols: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        for part in str(token or "").split():
            normalized = str(part or "").strip().upper()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            symbols.append(normalized)
    return symbols


def _resolve_kap_statement_symbols(universe: str, symbols: List[str]) -> List[str]:
    from app.kap_service import (
        get_bist30_companies,
        get_bist100_companies,
        get_kap_companies,
        normalize_kap_symbol,
    )

    normalized_symbols = _parse_kap_statement_symbols(symbols)
    if normalized_symbols:
        resolved = [normalize_kap_symbol(symbol) for symbol in normalized_symbols]
    else:
        code = str(universe or "all").strip().lower()
        if code in {"bist30", "bist100"}:
            source = get_bist30_companies() if code == "bist30" else get_bist100_companies()
        else:
            source = get_kap_companies()
        resolved = [normalize_kap_symbol(symbol) for symbol in source]

    result: List[str] = []
    seen: set[str] = set()
    for symbol in resolved:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
        if len(result) >= _KAP_STATEMENT_MAX_SYMBOLS:
            break
    if not result:
        raise ValueError("Bilanço yenilemesi için sembol listesi boş.")
    return result


def _statement_cache_path(symbol: str) -> Path:
    from src.kap_fetcher import _cache_file_for_company

    return _cache_file_for_company(CONFIG.paths.processed_dir, symbol)


def _read_local_statement_cache(symbol: str) -> Optional[Dict[str, Any]]:
    """Read only the local mirror; remote hydration happens at bootstrap."""

    path = _statement_cache_path(symbol)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _statement_cached_depth(payload: Optional[Dict[str, Any]]) -> int:
    from src.kap_fetcher import _cached_period_count

    return _cached_period_count(payload)


def _statement_cache_state(symbol: str, max_quarters: int) -> str:
    from src.kap_fetcher import KAP_CACHE_SCHEMA_VERSION

    payload = _read_local_statement_cache(symbol)
    if not isinstance(payload, dict) or payload.get("ok") is False:
        return "missing"
    try:
        schema_version = int(payload.get("schema_version") or 0)
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != KAP_CACHE_SCHEMA_VERSION:
        return "schema_outdated"
    if _statement_cached_depth(payload) < max(1, int(max_quarters)):
        return "shallow"
    return "ready"


def _latest_statement_key(payload: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int, int]]:
    if not isinstance(payload, dict):
        return None
    quarters = [row for row in payload.get("quarters") or [] if isinstance(row, dict)]
    if not quarters:
        return None
    try:
        latest = max(quarters, key=_quarter_sort_key)
        return (
            int(latest.get("year") or 0),
            int(latest.get("period") or 0),
            int(latest.get("disclosure_index") or 0),
        )
    except (TypeError, ValueError):
        return None


def _is_provider_block_error(exc: BaseException) -> bool:
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        try:
            return int(getattr(exc, "code", 0) or 0) in {403, 429, 503}
        except (TypeError, ValueError):
            return False
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, TimeoutError)):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (ConnectionResetError, ConnectionAbortedError, TimeoutError)):
        return True
    text = str(exc or "").lower()
    return any(
        token in text
        for token in (
            "connection reset",
            "connection aborted",
            "remote end closed",
            "urlopen error",
            "timed out",
            "temporary failure",
            " 403",
            " 429",
            "http error 403",
            "http error 429",
            "http error 503",
        )
    )


def _seed_resolved_member_from_statement_cache(
    symbol: str,
    payload: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(payload, dict):
        return
    from src.kap_fetcher import (
        _cache_matches_requested_company,
        _resolve_member_from_cache,
        _store_resolved_member,
    )

    if not _cache_matches_requested_company(payload, symbol):
        return
    member_oid = str(payload.get("member_oid") or "").strip()
    title = str(payload.get("company_title") or "").strip()
    if not member_oid or not title:
        return
    if _resolve_member_from_cache(symbol) is not None:
        return
    source_url = str(payload.get("source_url") or "").strip()
    permalink = (
        source_url.rsplit("/ozet/", 1)[-1].strip("/")
        if "/ozet/" in source_url
        else ""
    )
    _store_resolved_member(
        symbol,
        {
            "company_code": str(payload.get("stock_code") or symbol).strip().upper(),
            "mkk_member_oid": member_oid,
            "title": title,
            "permalink": permalink,
            "query": symbol,
        },
    )


def _process_kap_statement_symbol(
    symbol: str,
    *,
    mode: str,
    max_quarters: int,
) -> Dict[str, Any]:
    from app.financial_snapshot_store import (
        extract_financial_snapshot_from_kap_payload,
        upsert_financial_snapshots,
    )
    from app.kap_service import get_kap_snapshot, normalize_kap_symbol
    from src.kap_fetcher import KAP_CACHE_SCHEMA_VERSION

    normalized = normalize_kap_symbol(symbol)
    state = _statement_cache_state(normalized, max_quarters)
    if mode == "full":
        # Backfill only fills gaps; caches that already meet the requested
        # depth are deliberately left alone so the job stays idempotent.
        if state == "ready":
            return {"symbol": normalized, "outcome": "skipped_ready"}
    elif state != "ready":
        return {"symbol": normalized, "outcome": state}

    previous = _read_local_statement_cache(normalized)
    _seed_resolved_member_from_statement_cache(normalized, previous)
    previous_is_current_schema = (
        isinstance(previous, dict)
        and int(previous.get("schema_version") or 0) == KAP_CACHE_SCHEMA_VERSION
    )
    cached_depth = (
        min(_KAP_STATEMENT_DEPTH_CEILING, _statement_cached_depth(previous))
        if previous_is_current_schema
        else 0
    )
    fetch_depth = min(
        _KAP_STATEMENT_DEPTH_CEILING,
        max(int(max_quarters), cached_depth),
    )
    raw = get_kap_snapshot(
        company=normalized,
        cfg=CONFIG.kap,
        processed_dir=CONFIG.paths.processed_dir,
        force_refresh=False,
        max_quarters=fetch_depth,
        use_cache_when_complete=(mode == "incremental"),
        allow_criteria_fallback=False,
    )
    if not isinstance(raw, dict) or not raw.get("ok"):
        error = str((raw or {}).get("error") or "KAP snapshot unavailable")
        raise RuntimeError(error)

    schema_version = int(raw.get("schema_version") or 0)
    if schema_version != KAP_CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"KAP snapshot semasi guncel degil: {schema_version} != {KAP_CACHE_SCHEMA_VERSION}"
        )
    if raw.get("cache_stale"):
        raise RuntimeError("KAP snapshot bayat cache'ten dondu.")
    if mode == "full":
        try:
            live_period_count = int(raw.get("live_period_count") or 0)
        except (TypeError, ValueError):
            live_period_count = 0
        expected_depth = min(fetch_depth, live_period_count) if live_period_count > 0 else 0
        actual_depth = _statement_cached_depth(raw)
        if expected_depth > 0 and actual_depth < expected_depth:
            raise RuntimeError(
                "KAP bilanco derinligi yetersiz: "
                f"beklenen={expected_depth} alinan={actual_depth}"
            )

    snapshot_mtime_ns: Optional[int] = None
    try:
        snapshot_mtime_ns = _statement_cache_path(normalized).stat().st_mtime_ns
    except OSError:
        snapshot_mtime_ns = None
    financial_snapshot = extract_financial_snapshot_from_kap_payload(
        normalized,
        raw,
        source_mtime_ns=snapshot_mtime_ns,
    )
    if financial_snapshot:
        upsert_financial_snapshots(CONFIG.paths.processed_dir, [financial_snapshot])

    changed = _latest_statement_key(previous) != _latest_statement_key(raw)
    return {
        "symbol": normalized,
        "outcome": "updated" if changed else "unchanged",
        "quarter_count": _statement_cached_depth(raw),
        "schema_version": raw.get("schema_version") or KAP_CACHE_SCHEMA_VERSION,
    }


def _guarded_process_kap_statement_symbol(
    symbol: str,
    *,
    mode: str,
    max_quarters: int,
) -> Dict[str, Any]:
    from src.kap_fetcher import KapPacingDeadlineExceeded

    try:
        result = _process_kap_statement_symbol(
            symbol,
            mode=mode,
            max_quarters=max_quarters,
        )
        return {**result, "blocked": False}
    except KapPacingDeadlineExceeded:
        return {
            "symbol": symbol,
            "outcome": "budget_exhausted",
            "blocked": False,
        }
    except Exception as exc:  # noqa: BLE001 - one symbol must not stop the job
        blocked = _is_provider_block_error(exc)
        return {
            "symbol": symbol,
            "outcome": "blocked" if blocked else "failed",
            "blocked": blocked,
            "error": (str(exc or "")[:240] or exc.__class__.__name__),
        }


def _statement_coverage(symbols: List[str], max_quarters: int) -> Dict[str, int]:
    coverage = {
        "ready": 0,
        "missing": 0,
        "shallow": 0,
        "schema_outdated": 0,
        "total": len(symbols),
    }
    for symbol in symbols:
        state = _statement_cache_state(symbol, max_quarters)
        if state in coverage:
            coverage[state] += 1
        else:
            coverage["missing"] += 1
    return coverage


def _run_kap_statement_refresh_job(job_id: str) -> None:
    job = _get_kap_statement_refresh_job(job_id)
    if not isinstance(job, dict):
        return

    backend = _get_cache()
    owner_token = str(job.get("owner_token") or "")
    mode = str(job.get("mode") or "incremental")
    try:
        max_quarters = max(1, min(_KAP_STATEMENT_MAX_QUARTERS, int(job.get("max_quarters") or 5)))
        budget_seconds = max(
            30,
            min(
                _KAP_STATEMENT_MAX_BUDGET_SECONDS,
                int(job.get("budget_seconds") or _KAP_STATEMENT_DEFAULT_BUDGET_SECONDS),
            ),
        )
        concurrency = max(1, min(_KAP_STATEMENT_MAX_CONCURRENCY, int(job.get("concurrency") or 1)))
    except (TypeError, ValueError):
        max_quarters, budget_seconds, concurrency = 5, _KAP_STATEMENT_DEFAULT_BUDGET_SECONDS, 1

    started_monotonic = time.monotonic()
    started_at = _kap_statement_now_iso()
    job.update(
        status="running",
        started_at=started_at,
        total=0,
        processed=0,
        fetched=0,
        updated=0,
        unchanged=0,
        skipped_missing_cache=0,
        skipped_shallow_cache=0,
        skipped_schema=0,
        skipped_ready_cache=0,
        failed=0,
        blocked=0,
        remaining=0,
        cursor=0,
        coverage=None,
        stop_reason=None,
        errors=[],
        duration_seconds=None,
    )
    _set_kap_statement_refresh_job(job)

    try:
        symbols = _resolve_kap_statement_symbols(
            str(job.get("universe") or "all"),
            list(job.get("symbols") or []),
        )
    except Exception as exc:  # noqa: BLE001 - surface the reason to the operator
        job.update(
            status="failed",
            finished_at=_kap_statement_now_iso(),
            stop_reason="symbol_resolution_failed",
            error=str(exc or exc.__class__.__name__),
            duration_seconds=round(time.monotonic() - started_monotonic, 1),
        )
        _set_kap_statement_refresh_job(job)
        backend.release_if_owner(
            _KAP_STATEMENT_ACTIVE_KEY,
            {"job_id": job_id, "owner_token": owner_token},
        )
        return

    total = len(symbols)
    job.update(total=total, remaining=total)
    _set_kap_statement_refresh_job(job)

    deadline = started_monotonic + budget_seconds
    cursor = 0
    block_streak = 0
    failure_streak = 0
    last_flush = started_monotonic
    stop_reason = "completed"
    superseded = False

    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix=f"kap-statement-batch-{job_id[:8]}",
    )
    from src.kap_fetcher import set_kap_request_pacing

    try:
        set_kap_request_pacing(
            _KAP_STATEMENT_REQUEST_INTERVAL_SECONDS,
            deadline_monotonic=deadline,
        )
        while cursor < total:
            if not _owns_kap_statement_active_marker(job_id, owner_token):
                stop_reason = "superseded"
                superseded = True
                break
            if time.monotonic() >= deadline:
                stop_reason = "budget_exhausted"
                break

            window = symbols[cursor:cursor + concurrency]
            results = list(
                pool.map(
                    lambda symbol: _guarded_process_kap_statement_symbol(
                        symbol,
                        mode=mode,
                        max_quarters=max_quarters,
                    ),
                    window,
                )
            )
            if any(str(result.get("outcome") or "") == "budget_exhausted" for result in results):
                stop_reason = "budget_exhausted"
                break

            window_blocked = 0
            window_failed = 0
            for result in results:
                outcome = str(result.get("outcome") or "failed")
                skip_field = _SKIP_OUTCOME_FIELDS.get(outcome)
                if skip_field:
                    job[skip_field] = int(job.get(skip_field) or 0) + 1
                elif outcome == "updated":
                    job["updated"] = int(job.get("updated") or 0) + 1
                    job["fetched"] = int(job.get("fetched") or 0) + 1
                elif outcome == "unchanged":
                    job["unchanged"] = int(job.get("unchanged") or 0) + 1
                    job["fetched"] = int(job.get("fetched") or 0) + 1
                elif outcome == "blocked":
                    job["blocked"] = int(job.get("blocked") or 0) + 1
                    job["fetched"] = int(job.get("fetched") or 0) + 1
                    window_blocked += 1
                else:
                    job["failed"] = int(job.get("failed") or 0) + 1
                    job["fetched"] = int(job.get("fetched") or 0) + 1
                    window_failed += 1
                if result.get("error"):
                    errors = list(job.get("errors") or [])
                    if len(errors) < _KAP_STATEMENT_MAX_ERRORS:
                        errors.append(
                            {
                                "symbol": result.get("symbol"),
                                "outcome": outcome,
                                "error": str(result.get("error") or "")[:240],
                            }
                        )
                        job["errors"] = errors

            block_streak = block_streak + window_blocked if window_blocked else 0
            failure_streak = failure_streak + window_failed if window_failed else 0
            if not window_blocked and not window_failed:
                block_streak = 0
                failure_streak = 0

            cursor += len(window)
            job["cursor"] = cursor
            job["processed"] = cursor
            job["remaining"] = max(0, total - cursor)

            if block_streak >= _KAP_STATEMENT_BLOCK_STREAK_LIMIT:
                stop_reason = "provider_blocked"
                break
            if failure_streak >= _KAP_STATEMENT_FAILURE_STREAK_LIMIT:
                stop_reason = "provider_failures"
                break

            now = time.monotonic()
            if now - last_flush >= 1.0:
                _set_kap_statement_refresh_job(job)
                last_flush = now
            pause_seconds = (
                _KAP_STATEMENT_BLOCK_COOLDOWN_SECONDS
                if window_blocked
                else _KAP_STATEMENT_REQUEST_GAP_SECONDS
            )
            if now < deadline and pause_seconds > 0:
                time.sleep(min(pause_seconds, max(0.0, deadline - now)))
    except Exception as exc:  # noqa: BLE001 - never leave the job stuck in "running"
        LOGGER.exception("KAP statement refresh job failed: %s", job_id)
        job["failed"] = int(job.get("failed") or 0) + 1
        job["errors"] = list(job.get("errors") or [])[:_KAP_STATEMENT_MAX_ERRORS - 1] + [
            {"symbol": None, "outcome": "job_error", "error": str(exc or "")[:240]}
        ]
        stop_reason = "job_error"
    finally:
        pool.shutdown(wait=True)
        set_kap_request_pacing(0.0)

    if superseded:
        status = "superseded"
    elif stop_reason == "completed":
        if int(job.get("failed") or 0) > 0 or int(job.get("blocked") or 0) > 0:
            status = "partial"
            stop_reason = "completed_with_errors"
        else:
            status = "succeeded"
    elif stop_reason == "budget_exhausted":
        status = "partial"
    elif stop_reason in {"provider_blocked", "provider_failures"}:
        status = "aborted"
    else:
        status = "failed"

    job["status"] = status
    job["stop_reason"] = stop_reason
    job["finished_at"] = _kap_statement_now_iso()
    job["duration_seconds"] = round(time.monotonic() - started_monotonic, 1)
    job["processed"] = cursor
    job["remaining"] = max(0, total - cursor)
    if int(job.get("updated") or 0) > 0:
        try:
            from app.kap_api_service import (
                _KAP_COMPANIES_RESPONSE_CACHE,
                _KAP_COMPANIES_RESPONSE_CACHE_KEY,
            )
            from app.market_service import (
                _MARKET_STOCK_INDEX_ORDER,
                _STOCKS_CACHE,
                _UNIVERSE_CACHE,
            )

            _KAP_COMPANIES_RESPONSE_CACHE.clear()
            _UNIVERSE_CACHE.clear()
            _STOCKS_CACHE.clear()
            backend.delete(_KAP_COMPANIES_RESPONSE_CACHE_KEY)
            for idx_name in _MARKET_STOCK_INDEX_ORDER:
                backend.delete(f"api:market:universe:{idx_name}:v3")
                backend.delete(f"api:market:stocks:{idx_name}:v4")
        except Exception:  # noqa: BLE001 - cache invalidation must not fail the job
            LOGGER.warning("Market fundamental cache invalidation failed", exc_info=True)
    try:
        job["coverage"] = _statement_coverage(symbols, max_quarters)
    except Exception:  # noqa: BLE001 - coverage must not fail the job
        LOGGER.warning("KAP statement coverage scan failed", exc_info=True)
    _set_kap_statement_refresh_job(job)
    backend.release_if_owner(
        _KAP_STATEMENT_ACTIVE_KEY,
        {"job_id": job_id, "owner_token": owner_token},
    )


def _start_kap_statement_refresh_job(
    *,
    mode: str,
    universe: str,
    symbols: Optional[List[str]] = None,
    max_quarters: int = 5,
    budget_seconds: int = _KAP_STATEMENT_DEFAULT_BUDGET_SECONDS,
    concurrency: int = 1,
) -> Dict[str, Any]:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"incremental", "full"}:
        raise ValueError("mode 'incremental' veya 'full' olmalı.")
    normalized_universe = str(universe or "all").strip().lower()
    if normalized_universe not in {"all", "bist30", "bist100", "symbols"}:
        raise ValueError("universe 'all', 'bist30', 'bist100' veya 'symbols' olmalı.")
    parsed_symbols = _parse_kap_statement_symbols(symbols or [])
    if normalized_universe == "symbols" and not parsed_symbols:
        raise ValueError("universe=symbols için en az bir sembol gerekli.")

    try:
        effective_quarters = max(1, min(_KAP_STATEMENT_MAX_QUARTERS, int(max_quarters)))
        effective_budget = max(
            30,
            min(_KAP_STATEMENT_MAX_BUDGET_SECONDS, int(budget_seconds)),
        )
        effective_concurrency = max(1, min(_KAP_STATEMENT_MAX_CONCURRENCY, int(concurrency)))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_quarters, budget_seconds ve concurrency sayı olmalı.") from exc

    backend = _get_cache()
    existing = _active_kap_statement_refresh_job()
    if isinstance(existing, dict):
        if not _kap_statement_job_is_stale(existing):
            return _public_kap_statement_job(existing)
        stale_job = dict(existing)
        stale_job.update(
            status="failed",
            finished_at=_kap_statement_now_iso(),
            stop_reason="stale_job_replaced",
            error="Önceki bilanço yenileme işi zaman aşımına uğradı; yeni iş başlatıldı.",
        )
        _set_kap_statement_refresh_job(stale_job)
        backend.delete(_KAP_STATEMENT_ACTIVE_KEY)

    job_id = uuid.uuid4().hex
    owner_token = uuid.uuid4().hex
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "mode": normalized_mode,
        "universe": normalized_universe,
        "symbols": parsed_symbols,
        "max_quarters": effective_quarters,
        "budget_seconds": effective_budget,
        "concurrency": effective_concurrency,
        "request_interval_seconds": _KAP_STATEMENT_REQUEST_INTERVAL_SECONDS,
        "requested_at": _kap_statement_now_iso(),
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "total": 0,
        "processed": 0,
        "fetched": 0,
        "updated": 0,
        "unchanged": 0,
        "skipped_missing_cache": 0,
        "skipped_shallow_cache": 0,
        "skipped_schema": 0,
        "skipped_ready_cache": 0,
        "failed": 0,
        "blocked": 0,
        "remaining": 0,
        "cursor": 0,
        "coverage": None,
        "stop_reason": None,
        "errors": [],
        "owner_token": owner_token,
    }
    active_ttl = max(120, min(_KAP_STATEMENT_JOB_TTL_SECONDS, effective_budget + 900))
    owner_marker = {"job_id": job_id, "owner_token": owner_token}
    if not backend.set_if_absent(
        _KAP_STATEMENT_ACTIVE_KEY,
        owner_marker,
        ttl_seconds=active_ttl,
    ):
        existing = _active_kap_statement_refresh_job()
        if isinstance(existing, dict):
            return _public_kap_statement_job(existing)
        # The marker points at a job that already finished (or expired) but was
        # never released. Reclaim it so the operator is not locked out until the
        # marker TTL elapses.
        backend.delete(_KAP_STATEMENT_ACTIVE_KEY)
        if not backend.set_if_absent(
            _KAP_STATEMENT_ACTIVE_KEY,
            owner_marker,
            ttl_seconds=active_ttl,
        ):
            existing = _active_kap_statement_refresh_job()
            if isinstance(existing, dict):
                return _public_kap_statement_job(existing)
            raise RuntimeError("Bilanço yenilemesi başka bir worker tarafından devralındı; tekrar deneyin.")

    _set_kap_statement_refresh_job(job)
    _KAP_STATEMENT_EXECUTOR.submit(_run_kap_statement_refresh_job, job_id)
    return _public_kap_statement_job(job)


__all__ = [name for name in globals() if not name.startswith("__")]
