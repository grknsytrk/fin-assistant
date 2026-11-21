from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from app.funds_api import _fas


def _fund_refresh_job_key(job_id: str) -> str:
    return f"api:funds-refresh:job:{job_id}"


def _fund_refresh_now_iso() -> str:
    fas = _fas()
    return fas.datetime.now(fas.timezone.utc).isoformat()


def _get_fund_refresh_job(job_id: str) -> Optional[Dict[str, Any]]:
    fas = _fas()
    job = fas._get_cache().get(fas._fund_refresh_job_key(job_id))
    return dict(job) if isinstance(job, dict) else None


def _set_fund_refresh_job(job: Dict[str, Any]) -> None:
    fas = _fas()
    fas._FUND_AUTO_REFRESH_PREFLIGHT_L1.clear()
    fas._get_cache().set(
        fas._fund_refresh_job_key(str(job["job_id"])),
        job,
        ttl_seconds=fas._FUND_REFRESH_JOB_TTL_SECONDS,
    )


def _update_fund_refresh_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    fas = _fas()
    with fas._FUND_REFRESH_STATE_LOCK:
        job = fas._get_fund_refresh_job(job_id)
        if job is None:
            return None
        job.update(updates)
        fas._set_fund_refresh_job(job)
        return job


def _active_fund_refresh_job() -> Optional[Dict[str, Any]]:
    fas = _fas()
    backend = fas._get_cache()
    marker = backend.get(fas._FUND_REFRESH_ACTIVE_KEY)
    active_id = marker.get("job_id") if isinstance(marker, dict) else marker
    lease: Optional[Dict[str, Any]] = None
    if not active_id and fas.database_enabled():
        lease = fas.get_refresh_lease(fas._FUND_REFRESH_RESOURCE_KEY)
        lease_until = lease.get("lease_until") if isinstance(lease, dict) else None
        try:
            lease_active = bool(lease_until and lease_until > fas.datetime.now(fas.timezone.utc))
        except TypeError:
            try:
                lease_active = (
                    fas.datetime.fromisoformat(str(lease_until).replace("Z", "+00:00"))
                    > fas.datetime.now(fas.timezone.utc)
                )
            except (TypeError, ValueError):
                lease_active = False
        active_id = lease.get("job_id") if lease_active and isinstance(lease, dict) else None
    if not active_id:
        return None
    job = fas._get_fund_refresh_job(str(active_id))
    if job is None and lease and active_id:
        # Redis/job keys can disappear during a process restart while the
        # authoritative Postgres lease is still alive. Recreate a status
        # record and let the lease expire before another generation starts.
        job = {
            "job_id": str(active_id),
            "status": "running",
            "requested_at": fas._fund_refresh_now_iso(),
            "started_at": None,
            "finished_at": None,
            "as_of": None,
            "row_count": None,
            "cache_invalidation_status": None,
            "error": "Yenileme worker'ı yeniden başlatıldı; mevcut lease bekleniyor.",
            "resolution_status": None,
            "snapshot_action": None,
            "generation": lease.get("generation"),
        }
        fas._set_fund_refresh_job(job)
    if job and job.get("status") in {"queued", "running"}:
        if job.get("status") == "running":
            requested_text = str(job.get("requested_at") or "").strip()
            try:
                requested_at = fas.datetime.fromisoformat(requested_text.replace("Z", "+00:00"))
                runtime = (fas.datetime.now(fas.timezone.utc) - requested_at).total_seconds()
            except (TypeError, ValueError):
                runtime = 0.0
            if runtime > fas._FUND_REFRESH_MAX_RUNTIME_SECONDS:
                return job
            heartbeat_text = str(job.get("heartbeat_at") or "").strip()
            try:
                heartbeat_at = fas.datetime.fromisoformat(heartbeat_text.replace("Z", "+00:00"))
                heartbeat_age = (fas.datetime.now(fas.timezone.utc) - heartbeat_at).total_seconds()
            except (TypeError, ValueError):
                heartbeat_age = float("inf")
            if heartbeat_age > fas._FUND_REFRESH_HEARTBEAT_TIMEOUT_SECONDS:
                return job
        return job
    return None


def _run_fund_refresh_job(job_id: str, lookback_days: int) -> None:
    fas = _fas()
    initial_job = fas._get_fund_refresh_job(job_id) or {}
    owner_token = str(initial_job.get("owner_token") or job_id)
    generation = int(initial_job.get("generation") or 0)
    owner_marker: Any = {
        "job_id": job_id,
        "owner_token": owner_token,
        # Redis is only an advisory lease; the Postgres generation is kept
        # separately and checked during the fenced commit transaction.
        "generation": 0,
    }
    current_marker = fas._get_cache().get(fas._FUND_REFRESH_ACTIVE_KEY)
    if current_marker == job_id:
        owner_marker = job_id
    fas._update_fund_refresh_job(
        job_id,
        status="running",
        started_at=fas._fund_refresh_now_iso(),
        heartbeat_at=fas._fund_refresh_now_iso(),
    )
    backend = fas._get_cache()
    heartbeat_stop = fas.threading.Event()
    lease_lost = fas.threading.Event()

    def heartbeat() -> None:
        interval = max(5.0, fas._FUND_REFRESH_HEARTBEAT_INTERVAL_SECONDS)
        while not heartbeat_stop.wait(interval):
            current_job = fas._get_fund_refresh_job(job_id) or {}
            started_text = str(current_job.get("started_at") or current_job.get("requested_at") or "")
            try:
                runtime = (
                    fas.datetime.now(fas.timezone.utc)
                    - fas.datetime.fromisoformat(started_text.replace("Z", "+00:00"))
                ).total_seconds()
            except (TypeError, ValueError):
                runtime = 0.0
            if runtime > fas._FUND_REFRESH_MAX_RUNTIME_SECONDS:
                lease_lost.set()
                fas._update_fund_refresh_job(
                    job_id,
                    heartbeat_at=fas._fund_refresh_now_iso(),
                    error="Fon yenilemesi maksimum çalışma süresine ulaştı; commit engellenecek.",
                )
                continue
            renewed = backend.renew_if_owner(
                fas._FUND_REFRESH_ACTIVE_KEY,
                owner_marker,
                ttl_seconds=fas._FUND_REFRESH_LEASE_TTL_SECONDS,
            )
            db_renewed = fas.renew_refresh_lease(
                fas._FUND_REFRESH_RESOURCE_KEY,
                job_id=job_id,
                owner_token=owner_token,
                ttl_seconds=fas._FUND_REFRESH_LEASE_TTL_SECONDS,
            )
            if fas.database_enabled() and not db_renewed:
                lease_lost.set()
            if renewed or not fas.database_enabled():
                fas._update_fund_refresh_job(job_id, heartbeat_at=fas._fund_refresh_now_iso())

    heartbeat_thread = fas.threading.Thread(
        target=heartbeat,
        name=f"fund-refresh-heartbeat-{job_id[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        from app import fund_service as _fund_service

        refresh_funds_snapshot = getattr(
            fas, "refresh_funds_snapshot", _fund_service.refresh_funds_snapshot
        )
        bounded_lookback_days = min(fas._FUND_REFRESH_MAX_LOOKBACK_DAYS, max(1, int(lookback_days)))
        refresh_kwargs = {
            "lookback_days": bounded_lookback_days,
            "persist_reference_data": False,
            "backfill_daily_returns": False,
            "persist_snapshot": False,
        }
        while True:
            try:
                result = refresh_funds_snapshot(fas.CONFIG.paths.processed_dir, **refresh_kwargs)
                break
            except TypeError as exc:
                unexpected = str(exc)
                if "persist_snapshot" in unexpected and "persist_snapshot" in refresh_kwargs:
                    refresh_kwargs.pop("persist_snapshot")
                    continue
                if "persist_reference_data" in unexpected and "persist_reference_data" in refresh_kwargs:
                    refresh_kwargs.pop("persist_reference_data")
                    refresh_kwargs.pop("backfill_daily_returns", None)
                    continue
                if "lookback_days" in unexpected and len(refresh_kwargs) == 1:
                    result = refresh_funds_snapshot(fas.CONFIG.paths.processed_dir)
                    break
                raise
        if not isinstance(result, dict):
            raise RuntimeError("Fon yenilemesi geçerli bir sonuç döndürmedi.")
        rows = list(result.get("rows") or []) if isinstance(result, dict) else []
        resolution_status = str(
            result.get("resolution_status") or ("available" if rows else "upstream_unavailable")
        )
        if lease_lost.is_set():
            fas._update_fund_refresh_job(
                job_id,
                status="superseded",
                finished_at=fas._fund_refresh_now_iso(),
                error="Fon yenileme lease'i başka bir generation tarafından devralındı.",
                resolution_status=resolution_status,
            )
            return
        if resolution_status == "upstream_unavailable":
            warnings = list(result.get("warnings") or [])
            fas._update_fund_refresh_job(
                job_id,
                status="failed",
                finished_at=fas._fund_refresh_now_iso(),
                resolution_status=resolution_status,
                snapshot_action="retained_existing",
                error=warnings[0] if warnings else "TEFAS güvenilir cevap döndürmedi; mevcut snapshot korundu.",
            )
            return
        result_meta = dict(result.get("source_metadata") or {})
        snapshot_action = str(
            result_meta.get("snapshot_action") or result.get("snapshot_action") or ""
        ).strip()
        if snapshot_action == "retained_existing" and result_meta.get("truncated_refresh_observed"):
            warnings = list(result.get("warnings") or [])
            fas._update_fund_refresh_job(
                job_id,
                status="failed",
                finished_at=fas._fund_refresh_now_iso(),
                resolution_status=resolution_status,
                snapshot_action=snapshot_action,
                error=warnings[-1] if warnings else "TEFAS eksik snapshot döndürdü; mevcut snapshot korundu.",
            )
            return
        commit_funds_snapshot = getattr(
            fas, "commit_funds_snapshot", _fund_service.commit_funds_snapshot
        )
        if (
            snapshot_action in {"retained_current", "retained_newer", "retained_current_market_closed"}
            and commit_funds_snapshot is _fund_service.commit_funds_snapshot
        ):
            fas._update_fund_refresh_job(
                job_id,
                status="succeeded",
                finished_at=fas._fund_refresh_now_iso(),
                as_of=result.get("as_of"),
                row_count=len(rows),
                cache_invalidation_status="skipped_unchanged",
                resolution_status=resolution_status,
                resolved_as_of=result_meta.get("resolved_as_of"),
                snapshot_action=snapshot_action,
                error=None,
            )
            return

        result = dict(result)
        result["snapshot_generation"] = generation
        result_meta["snapshot_generation"] = generation
        result["source_metadata"] = result_meta
        committed = commit_funds_snapshot(
            fas.CONFIG.paths.processed_dir,
            result,
            job_id=job_id,
            owner_token=owner_token,
            generation=generation,
        )
        if not committed:
            fas._update_fund_refresh_job(
                job_id,
                status="superseded",
                finished_at=fas._fund_refresh_now_iso(),
                resolution_status=resolution_status,
                error="Snapshot commit'i lease/generation kontrolünden geçmedi.",
            )
            return
        cache_invalidation_status = fas._invalidate_fund_response_cache()
        fas._update_fund_refresh_job(
            job_id,
            status="succeeded",
            finished_at=fas._fund_refresh_now_iso(),
            as_of=result.get("as_of"),
            row_count=len(rows),
            cache_invalidation_status=cache_invalidation_status,
            resolution_status=resolution_status,
            resolved_as_of=(result.get("source_metadata") or {}).get("resolved_as_of"),
            snapshot_action=(result.get("source_metadata") or {}).get("snapshot_action")
            or result.get("snapshot_action"),
            error=None,
        )
    except Exception as exc:
        fas.logging.getLogger("uvicorn.error").exception("fund snapshot refresh job failed")
        fas._update_fund_refresh_job(
            job_id, status="failed", finished_at=fas._fund_refresh_now_iso(), error=str(exc)
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        backend.release_if_owner(fas._FUND_REFRESH_ACTIVE_KEY, owner_marker)
        fas.release_refresh_lease(
            fas._FUND_REFRESH_RESOURCE_KEY,
            job_id=job_id,
            owner_token=owner_token,
        )


def _start_fund_refresh_job(lookback_days: int) -> Dict[str, Any]:
    fas = _fas()
    existing = fas._active_fund_refresh_job()
    if existing is not None:
        return existing

    backend = fas._get_cache()
    job_id = fas.uuid.uuid4().hex
    owner_token = fas.uuid.uuid4().hex
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "requested_at": fas._fund_refresh_now_iso(),
        "started_at": None,
        "finished_at": None,
        "as_of": None,
        "row_count": None,
        "cache_invalidation_status": None,
        "error": None,
        "resolution_status": None,
        "snapshot_action": None,
        "owner_token": owner_token,
        "generation": None,
    }
    owner_marker = {"job_id": job_id, "owner_token": owner_token, "generation": 0}
    if not backend.set_if_absent(
        fas._FUND_REFRESH_ACTIVE_KEY,
        owner_marker,
        ttl_seconds=fas._FUND_REFRESH_LEASE_TTL_SECONDS,
    ):
        existing = fas._active_fund_refresh_job()
        if existing is not None:
            return existing
        job.update(
            status="failed",
            finished_at=fas._fund_refresh_now_iso(),
            error="Fon yenilemesi başka bir worker tarafından devralındı; tekrar deneyin.",
        )
        fas._set_fund_refresh_job(job)
        return job

    lease = fas.acquire_refresh_lease(
        fas._FUND_REFRESH_RESOURCE_KEY,
        job_id=job_id,
        owner_token=owner_token,
        ttl_seconds=fas._FUND_REFRESH_LEASE_TTL_SECONDS,
    )
    if fas.database_enabled() and not lease:
        backend.release_if_owner(fas._FUND_REFRESH_ACTIVE_KEY, owner_marker)
        existing = fas._active_fund_refresh_job()
        if existing is not None:
            return existing
        job.update(
            status="failed",
            finished_at=fas._fund_refresh_now_iso(),
            error="Fon yenileme Postgres lease'i alınamadı; mevcut snapshot korundu.",
        )
        fas._set_fund_refresh_job(job)
        return job
    if lease:
        generation = int(lease.get("generation") or 0)
        job["generation"] = generation
        owner_marker["generation"] = generation
    job["lease_until"] = str(lease.get("lease_until")) if lease else None
    fas._set_fund_refresh_job(job)
    fas._FUND_REFRESH_EXECUTOR.submit(fas._run_fund_refresh_job, job_id, lookback_days)
    return job


def _fund_snapshot_needs_background_refresh(payload: Dict[str, Any]) -> bool:
    """Return whether the canonical catalogue is behind its market target."""

    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    total_count = payload.get("total_count")
    try:
        has_catalogue = int(total_count) > 0
    except (TypeError, ValueError):
        has_catalogue = bool(payload.get("rows"))
    if not has_catalogue:
        return True

    # A previous deployment may have preserved a full snapshot after TEFAS
    # returned only its first page. Its as_of date can still match today's
    # target, so the date/lag checks below alone would never retry the repair.
    # Let the new paginated collector heal that specific legacy state once.
    if metadata.get("truncated_refresh_observed"):
        return True
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if any("snapshot looked truncated" in str(warning).casefold() for warning in warnings):
        return True

    lag = metadata.get("snapshot_as_of_lag_days")
    try:
        return int(lag) > 0
    except (TypeError, ValueError):
        pass

    snapshot_as_of = str(metadata.get("snapshot_as_of") or payload.get("as_of") or "").strip()
    target_date = str(metadata.get("snapshot_target_date") or "").strip()
    return bool(snapshot_as_of and target_date and snapshot_as_of < target_date)


def _public_fund_refresh_job(job: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "job_id",
        "status",
        "requested_at",
        "started_at",
        "finished_at",
        "as_of",
        "row_count",
        "error",
        "resolution_status",
        "snapshot_action",
    )
    return {field: job.get(field) for field in fields}


def _maybe_schedule_fund_snapshot_refresh(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Self-heal a missed external schedule without blocking the list request."""

    fas = _fas()
    # A response still being assembled is not evidence of a missing snapshot.
    if payload.get("status") == "pending":
        return payload
    if not fas._fund_snapshot_needs_background_refresh(payload):
        return payload

    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    target_date = str(metadata.get("snapshot_target_date") or "unknown").strip()
    snapshot_as_of = str(metadata.get("snapshot_as_of") or payload.get("as_of") or "").strip()
    truncated_flag = bool(metadata.get("truncated_refresh_observed"))
    backend = fas._get_cache()
    preflight_key = (id(backend), target_date, snapshot_as_of, truncated_flag)
    l1_preflight = fas._FUND_AUTO_REFRESH_PREFLIGHT_L1.get(preflight_key)
    now_mono = fas.time.monotonic()
    if isinstance(l1_preflight, dict) and float(l1_preflight.get("_exp") or 0.0) > now_mono:
        cached_job = l1_preflight.get("job")
        if cached_job is None:
            return payload
        if isinstance(cached_job, dict):
            out = dict(payload)
            out["refresh_pending"] = bool(payload.get("refresh_pending")) or str(
                cached_job.get("status") or ""
            ) in {"queued", "running"}
            out["refresh_job"] = cached_job
            return out

    job = fas._active_fund_refresh_job()
    started_new_job = False
    if job is None:
        cooldown_key = f"{fas._FUND_REFRESH_AUTO_KEY_PREFIX}:{target_date}"
        cooldown_value = backend.get(cooldown_key)
        if isinstance(cooldown_value, dict):
            cooldown_job_id = str(cooldown_value.get("job_id") or "").strip()
            if cooldown_job_id:
                cooldown_job = fas._get_fund_refresh_job(cooldown_job_id)
                if cooldown_job is not None and str(cooldown_job.get("status") or "") not in {
                    "queued",
                    "running",
                }:
                    # A failed or superseded repair must not suppress the next
                    # automatic recovery attempt for the whole cooldown TTL.
                    backend.delete(cooldown_key)
        claimed = backend.set_if_absent(
            cooldown_key,
            {"scheduled_at": fas._fund_refresh_now_iso()},
            ttl_seconds=max(60, fas._FUND_REFRESH_AUTO_COOLDOWN_SECONDS),
        )
        if claimed:
            started_new_job = True
            try:
                job = fas._start_fund_refresh_job(fas._FUND_REFRESH_MAX_LOOKBACK_DAYS)
                backend.set(
                    cooldown_key,
                    {
                        "scheduled_at": fas._fund_refresh_now_iso(),
                        "job_id": job.get("job_id"),
                    },
                    ttl_seconds=max(60, fas._FUND_REFRESH_AUTO_COOLDOWN_SECONDS),
                )
                fas.LOGGER.info(
                    "stale fund catalogue scheduled background refresh: target=%s job_id=%s",
                    target_date,
                    job.get("job_id"),
                )
            except Exception:
                backend.delete(cooldown_key)
                fas.LOGGER.exception("stale fund catalogue background refresh could not be scheduled")

    if job is None:
        fas._FUND_AUTO_REFRESH_PREFLIGHT_L1[preflight_key] = {
            "job": None,
            "_exp": fas.time.monotonic() + 20.0,
        }
        return payload

    public_job = fas._public_fund_refresh_job(job)
    if not started_new_job:
        fas._FUND_AUTO_REFRESH_PREFLIGHT_L1[preflight_key] = {
            "job": public_job,
            "_exp": fas.time.monotonic() + 5.0,
        }
    out = dict(payload)
    out["refresh_pending"] = bool(payload.get("refresh_pending")) or str(job.get("status") or "") in {
        "queued",
        "running",
    }
    out["refresh_job"] = public_job
    return out


async def _execute_fund_price_collector_cycle(
    *,
    interval: Optional[float] = None,
    lease_ttl: int = 600,
    heartbeat_interval: float = 30.0,
) -> bool:
    """Execute a single cycle of the fund price collector protected by distributed Redis lease."""
    fas = _fas()
    from app import fund_service as _fund_service

    collect_daily_fund_prices = getattr(
        fas, "collect_daily_fund_prices", _fund_service.collect_daily_fund_prices
    )
    backend = fas._get_cache()
    if interval is None:
        interval = float(fas.os.getenv("RAGFIN_FUND_COLLECTOR_INTERVAL_SECONDS", str(24 * 60 * 60)))
    lease_key = "lock:fund-price-collector"
    cooldown_key = "lock:fund-price-collector:cooldown"
    worker_id = f"worker:{fas.os.getpid()}:{fas.uuid.uuid4().hex[:8]}"

    # 1. Check if cycle was already completed within current interval cooldown
    if backend.get(cooldown_key) is not None:
        fas.LOGGER.debug("Fund price collector cooldown active; skipping cycle")
        return False

    # 2. Attempt to acquire distributed execution lock
    if not backend.set_if_absent(lease_key, worker_id, ttl_seconds=lease_ttl):
        fas.LOGGER.debug("Fund price collector lease held by another worker; skipping cycle")
        return False

    # 3. Background lease heartbeat renewal
    heartbeat_stop = fas.asyncio.Event()

    async def _lease_heartbeat() -> None:
        while not heartbeat_stop.is_set():
            try:
                await fas.asyncio.sleep(heartbeat_interval)
                if heartbeat_stop.is_set():
                    break
                backend.renew_if_owner(lease_key, worker_id, ttl_seconds=lease_ttl)
            except fas.asyncio.CancelledError:
                break
            except Exception:
                pass

    heartbeat_task = fas.asyncio.create_task(_lease_heartbeat())
    try:
        fas.LOGGER.info(
            "Acquired fund price collector lease (%s); executing collect_daily_fund_prices", worker_id
        )
        result = await fas.asyncio.to_thread(collect_daily_fund_prices, fas.CONFIG.paths.processed_dir)
        fas.LOGGER.info(
            "fund price collector completed: valid=%s skipped=%s source=%s",
            result.get("valid_point_count") if isinstance(result, dict) else None,
            result.get("skipped_point_count") if isinstance(result, dict) else None,
            result.get("source") if isinstance(result, dict) else None,
        )
        backend.set(cooldown_key, worker_id, ttl_seconds=max(60, int(interval) - 30))
        return True
    except fas.asyncio.CancelledError:
        raise
    except Exception:
        fas.LOGGER.exception("fund price collector failed")
        return False
    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except fas.asyncio.CancelledError:
            pass
        backend.release_if_owner(lease_key, worker_id)


async def _fund_price_collector_loop() -> None:
    fas = _fas()
    startup_delay = float(fas.os.getenv("RAGFIN_FUND_COLLECTOR_STARTUP_DELAY_SECONDS", "60"))
    interval = float(fas.os.getenv("RAGFIN_FUND_COLLECTOR_INTERVAL_SECONDS", str(24 * 60 * 60)))
    if startup_delay > 0:
        await fas.asyncio.sleep(startup_delay)
    while True:
        try:
            await fas._execute_fund_price_collector_cycle(interval=interval)
        except fas.asyncio.CancelledError:
            raise
        except Exception:
            fas.LOGGER.exception("fund price collector failed")
        await fas.asyncio.sleep(max(60.0, interval))


async def _start_fund_price_collector() -> None:
    fas = _fas()
    if not fas._truthy_env("RAGFIN_FUND_COLLECTOR_ENABLED", "1"):
        return
    if fas._FUND_COLLECTOR_TASK and not fas._FUND_COLLECTOR_TASK.done():
        return
    fas._FUND_COLLECTOR_TASK = fas.asyncio.create_task(fas._fund_price_collector_loop())


async def _stop_fund_price_collector() -> None:
    fas = _fas()
    task = fas._FUND_COLLECTOR_TASK
    fas._FUND_COLLECTOR_TASK = None
    if not task:
        return
    task.cancel()
    try:
        await task
    except fas.asyncio.CancelledError:
        pass


def _history_job_key(fund_code: str, job_id: str) -> str:
    fas = _fas()
    return f"api:fund-history:job:v{fas._FUND_HISTORY_KEY_VERSION}:{fund_code}:{job_id}"


def _history_active_key(fund_code: str) -> str:
    fas = _fas()
    return f"api:fund-history:active:v{fas._FUND_HISTORY_KEY_VERSION}:{fund_code}"


def _history_last_key(fund_code: str) -> str:
    fas = _fas()
    return f"api:fund-history:last:v{fas._FUND_HISTORY_KEY_VERSION}:{fund_code}"


def _history_job_get(fund_code: str, job_id: str) -> Optional[Dict[str, Any]]:
    fas = _fas()
    value = fas._get_cache().get(fas._history_job_key(fund_code, job_id))
    return dict(value) if isinstance(value, dict) else None


def _history_job_set(fund_code: str, job: Dict[str, Any]) -> None:
    fas = _fas()
    fas._get_cache().set(
        fas._history_job_key(fund_code, str(job["job_id"])),
        job,
        ttl_seconds=fas._FUND_HISTORY_JOB_TTL_SECONDS,
    )


def _history_job_public(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job:
        return None
    return {
        key: job.get(key)
        for key in (
            "job_id",
            "fund_code",
            "requested_start",
            "requested_end",
            "effective_start",
            "effective_end",
            "status",
            "requested_at",
            "started_at",
            "finished_at",
            "heartbeat_at",
            "error",
            "resolution",
            "coverage_state",
            "daily_upgrade_state",
            "fintables_point_count",
            "history_source_used",
            "phase",
        )
        if key in job
    }


def _history_job_date(value: Any) -> Optional[date]:
    fas = _fas()
    try:
        return fas.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _history_request_range(
    start_date: Optional[date],
    end_date: Optional[date],
) -> Dict[str, Any]:
    fas = _fas()
    from app import fund_service as _fund_service

    _fund_full_history_start_date = getattr(
        fas, "_fund_full_history_start_date", _fund_service._fund_full_history_start_date
    )
    effective_end = end_date or fas.date.today()
    full_history = start_date is None and end_date is None
    requested_start = start_date or _fund_full_history_start_date()
    warmup_start = effective_end - fas.timedelta(days=max(1, fas._FUND_HISTORY_WARMUP_DAYS))
    effective_start = requested_start if full_history else min(requested_start, warmup_start)
    return {
        "requested_start": requested_start,
        "requested_end": effective_end,
        "effective_start": effective_start,
        "effective_end": effective_end,
        "full_history": full_history,
    }


def _history_job_covers(job: Dict[str, Any], target: Dict[str, Any]) -> bool:
    fas = _fas()
    job_start = fas._history_job_date(job.get("effective_start"))
    job_end = fas._history_job_date(job.get("effective_end"))
    return bool(
        job_start
        and job_end
        and job_start <= target["requested_start"]
        and job_end >= target["requested_end"]
    )


def _history_job_failed(job: Optional[Dict[str, Any]]) -> bool:
    if not job:
        return False
    return (
        str(job.get("status") or "").strip().lower() == "failed"
        or str(job.get("daily_upgrade_state") or "").strip().lower() == "failed"
    )


def _history_job_should_schedule(
    payload: Dict[str, Any],
    *,
    last_job: Optional[Dict[str, Any]],
    target: Dict[str, Any],
) -> bool:
    fas = _fas()
    metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    points = payload.get("points") if isinstance(payload.get("points"), list) else []
    if not points:
        return True
    # Long chart ranges are bootstrapped from Fintables' daily UDF series.
    # Jobs created before that source was made explicit can look successful
    # while containing only the shorter TEFAS detail history. Give each such
    # legacy job one retry; the worker records the probe result so an
    # unavailable upstream does not create a request loop.
    if fas._history_job_needs_fintables_probe(last_job, target):
        return True
    target_span_days = (target["requested_end"] - target["requested_start"]).days
    history_source = str(metadata.get("history_source_used") or "").strip().lower()
    if target_span_days > 120 and history_source and history_source != "fintables_udf_history":
        # A complete TEFAS cache can otherwise suppress the first Fintables
        # probe forever. Once a job records ``fintables_point_count`` (even
        # zero), the probe has been made and should not loop on every request.
        job_status = str(last_job.get("status") or "").strip().lower() if last_job else ""
        if job_status in {"queued", "running"} and fas._history_job_covers(last_job, target):
            return False
        if not last_job or "fintables_point_count" not in last_job:
            return True
    last_point = fas._history_job_date(metadata.get("available_end_date") or metadata.get("date_max"))
    if last_point and last_point < target["requested_end"]:
        from app import fund_service as _fund_service

        _business_days_between = getattr(
            fas, "_business_days_between", _fund_service._business_days_between
        )
        if _business_days_between(last_point + fas.timedelta(days=1), target["requested_end"]) > 0:
            return not (
                last_job
                and fas._history_job_covers(last_job, target)
                and not fas._history_job_failed(last_job)
            )
    if str(metadata.get("coverage_state") or "") == "range_incomplete":
        return not (
            last_job
            and fas._history_job_covers(last_job, target)
            and not fas._history_job_failed(last_job)
        )
    if int(metadata.get("internal_gap_count") or 0) > 0:
        return not (
            last_job
            and fas._history_job_covers(last_job, target)
            and not fas._history_job_failed(last_job)
        )
    resolution = str(metadata.get("resolution") or "unknown")
    if resolution != "daily":
        if last_job and fas._history_job_covers(last_job, target):
            return fas._history_job_failed(last_job) or str(last_job.get("daily_upgrade_state") or "") not in {
                "unavailable",
                "failed",
            }
        return True
    return False


def _history_job_needs_fintables_probe(
    last_job: Optional[Dict[str, Any]],
    target: Dict[str, Any],
) -> bool:
    fas = _fas()
    if not last_job:
        return False
    job_status = str(last_job.get("status") or "").strip().lower()
    if job_status in {"queued", "running"}:
        return False
    target_span_days = (target["requested_end"] - target["requested_start"]).days
    return bool(
        target_span_days > 120
        and fas._history_job_covers(last_job, target)
        and "fintables_point_count" not in last_job
    )


def _history_attach_job(payload: Dict[str, Any], job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    fas = _fas()
    if not job:
        return payload
    attached = dict(payload)
    metadata = dict(attached.get("source_metadata") or {})
    public_job = fas._history_job_public(job)
    metadata["history_job"] = public_job
    points = attached.get("points") if isinstance(attached.get("points"), list) else []
    status = str(job.get("status") or "").lower()
    resolution = str(metadata.get("resolution") or "unknown")
    if status in {"queued", "running"}:
        if points and resolution != "daily":
            if metadata.get("coverage_state") == "complete":
                metadata["coverage_state"] = "upgrading"
        elif not points:
            metadata["coverage_state"] = "upgrading"
        metadata["daily_upgrade_state"] = "pending"
    elif status == "failed":
        metadata["daily_upgrade_state"] = "failed"
    attached["source_metadata"] = metadata
    return attached


def _history_find_existing_job(fund_code: str, target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fas = _fas()
    backend = fas._get_cache()
    active_id = backend.get(fas._history_active_key(fund_code))
    if isinstance(active_id, dict):
        active_id = active_id.get("job_id")
    if active_id:
        active = fas._history_job_get(fund_code, str(active_id))
        if active and str(active.get("status") or "") in {"queued", "running"}:
            return active
        try:
            backend.delete(fas._history_active_key(fund_code))
        except Exception:
            pass
    last = backend.get(fas._history_last_key(fund_code))
    if isinstance(last, dict) and fas._history_job_covers(last, target):
        return dict(last)
    return None


def _history_merge_target(job: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    fas = _fas()
    for field, chooser in (
        ("requested_start", min),
        ("effective_start", min),
    ):
        existing = fas._history_job_date(job.get(field))
        value = chooser(existing, target[field]) if existing else target[field]
        job[field] = value.isoformat()
    for field, chooser in (
        ("requested_end", max),
        ("effective_end", max),
    ):
        existing = fas._history_job_date(job.get(field))
        value = chooser(existing, target[field]) if existing else target[field]
        job[field] = value.isoformat()
    return job


def _history_start_or_extend_job(
    fund_code: str,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
    force_new: bool = False,
) -> Optional[Dict[str, Any]]:
    fas = _fas()
    normalized = str(fund_code or "").strip().upper()
    if not normalized:
        return None
    target = fas._history_request_range(start_date, end_date)
    backend = fas._get_cache()
    job: Optional[Dict[str, Any]] = None
    submit = False
    with backend.lock(f"fund-history-state:{normalized}", timeout=5.0) as acquired:
        if not acquired:
            job = fas._history_find_existing_job(normalized, target)
        else:
            existing = fas._history_find_existing_job(normalized, target)
            if existing and str(existing.get("status") or "") in {"queued", "running"}:
                previous_start = fas._history_job_date(existing.get("effective_start"))
                previous_end = fas._history_job_date(existing.get("effective_end"))
                job = fas._history_merge_target(existing, target)
                if str(job.get("status")) == "running" and (
                    (previous_start and target["effective_start"] < previous_start)
                    or (previous_end and target["effective_end"] > previous_end)
                ):
                    job["extension_requested"] = True
                fas._history_job_set(normalized, job)
            elif (
                existing
                and fas._history_job_covers(existing, target)
                and not fas._history_job_failed(existing)
                and not force_new
            ):
                job = existing
            else:
                now = fas._fund_refresh_now_iso()
                job = {
                    "job_id": fas.uuid.uuid4().hex,
                    "fund_code": normalized,
                    "requested_start": target["requested_start"].isoformat(),
                    "requested_end": target["requested_end"].isoformat(),
                    "effective_start": target["effective_start"].isoformat(),
                    "effective_end": target["effective_end"].isoformat(),
                    "status": "queued",
                    "requested_at": now,
                    "started_at": None,
                    "finished_at": None,
                    "heartbeat_at": None,
                    "error": None,
                    "phase": 0,
                }
                fas._history_job_set(normalized, job)
                backend.set(
                    fas._history_active_key(normalized),
                    job["job_id"],
                    ttl_seconds=fas._FUND_HISTORY_JOB_TTL_SECONDS,
                )
                submit = True
    if submit and job:
        try:
            fas._FUND_HISTORY_EXECUTOR.submit(fas._run_fund_history_job, normalized, str(job["job_id"]))
        except Exception as exc:
            job["status"] = "failed"
            job["finished_at"] = fas._fund_refresh_now_iso()
            job["error"] = f"history worker could not start: {exc}"
            fas._history_job_set(normalized, job)
    return job


def _history_job_owned(backend: Any, lease_key: str, owner_token: str) -> bool:
    return backend.get(lease_key) == owner_token


def _run_fund_history_job(fund_code: str, job_id: str) -> None:
    fas = _fas()
    from app import fund_service as _fund_service

    FundUpstreamError = getattr(fas, "FundUpstreamError", _fund_service.FundUpstreamError)
    get_fund_performance_payload = getattr(
        fas, "get_fund_performance_payload", _fund_service.get_fund_performance_payload
    )
    normalize_fund_code = getattr(fas, "normalize_fund_code", _fund_service.normalize_fund_code)
    refresh_fund_performance = getattr(
        fas, "refresh_fund_performance", _fund_service.refresh_fund_performance
    )

    normalized = normalize_fund_code(fund_code)
    job = fas._history_job_get(normalized, job_id)
    if not job:
        return
    backend = fas._get_cache()
    lease_key = f"api:fund-history:lease:v{fas._FUND_HISTORY_KEY_VERSION}:{normalized}"
    owner_token = fas.uuid.uuid4().hex
    if not backend.set_if_absent(lease_key, owner_token, ttl_seconds=fas._FUND_HISTORY_LEASE_TTL_SECONDS):
        job["status"] = "failed"
        job["finished_at"] = fas._fund_refresh_now_iso()
        job["error"] = "history job lease is already owned"
        fas._history_job_set(normalized, job)
        return

    heartbeat_stop = fas.threading.Event()
    lease_lost = fas.threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(max(1.0, fas._FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS)):
            try:
                renewed = backend.renew_if_owner(
                    lease_key,
                    owner_token,
                    ttl_seconds=fas._FUND_HISTORY_LEASE_TTL_SECONDS,
                )
            except Exception:
                renewed = False
            if not renewed:
                lease_lost.set()
                return

    heartbeat_thread = fas.threading.Thread(
        target=heartbeat, name=f"fund-history-heartbeat-{normalized}", daemon=True
    )
    heartbeat_thread.start()
    try:
        job["status"] = "running"
        job["started_at"] = job.get("started_at") or fas._fund_refresh_now_iso()
        job["heartbeat_at"] = fas._fund_refresh_now_iso()
        fas._history_job_set(normalized, job)
        for phase in range(max(1, fas._FUND_HISTORY_MAX_PHASES)):
            current = fas._history_job_get(normalized, job_id) or job
            phase_start = fas._history_job_date(current.get("effective_start"))
            phase_end = fas._history_job_date(current.get("effective_end"))
            if not phase_start or not phase_end:
                raise FundUpstreamError("history job range is invalid")
            current["phase"] = phase + 1
            current["heartbeat_at"] = fas._fund_refresh_now_iso()
            fas._history_job_set(normalized, current)
            refresh_fund_performance(
                fas.CONFIG.paths.processed_dir,
                normalized,
                start_date=phase_start,
                end_date=phase_end,
                prefer_fast_long_range=(phase_end - phase_start).days > 120,
            )
            if lease_lost.is_set() or not fas._history_job_owned(backend, lease_key, owner_token):
                raise FundUpstreamError("history job lease was lost before commit")
            latest = fas._history_job_get(normalized, job_id) or current
            latest_start = fas._history_job_date(latest.get("effective_start"))
            latest_end = fas._history_job_date(latest.get("effective_end"))
            range_widened = bool(
                (latest_start and latest_start < phase_start)
                or (latest_end and latest_end > phase_end)
            )
            if phase + 1 < max(1, fas._FUND_HISTORY_MAX_PHASES) and range_widened:
                continue
            result = get_fund_performance_payload(
                fas.CONFIG.paths.processed_dir,
                normalized,
                start_date=phase_start,
                end_date=phase_end,
                auto_refresh=False,
            )
            metadata = result.get("source_metadata") if isinstance(result.get("source_metadata"), dict) else {}
            latest.update(
                {
                    "status": "succeeded",
                    "finished_at": fas._fund_refresh_now_iso(),
                    "heartbeat_at": fas._fund_refresh_now_iso(),
                    "resolution": metadata.get("resolution"),
                    "coverage_state": metadata.get("coverage_state"),
                    "daily_upgrade_state": "unavailable"
                    if metadata.get("resolution") != "daily"
                    else "complete",
                    "fintables_point_count": int(metadata.get("cached_fallback_point_count") or 0),
                    "history_source_used": metadata.get("history_source_used"),
                    "error": None,
                }
            )
            fas._invalidate_single_fund_response_cache(normalized, include_holdings=False)
            fas._history_job_set(normalized, latest)
            backend.set(
                fas._history_last_key(normalized),
                latest,
                ttl_seconds=fas._FUND_HISTORY_JOB_TTL_SECONDS,
            )
            return
    except Exception as exc:
        failed = fas._history_job_get(normalized, job_id) or job
        failed.update(
            {
                "status": "failed",
                "finished_at": fas._fund_refresh_now_iso(),
                "error": str(exc),
                "daily_upgrade_state": "failed",
            }
        )
        fas._history_job_set(normalized, failed)
        backend.set(
            fas._history_last_key(normalized),
            failed,
            ttl_seconds=fas._FUND_HISTORY_JOB_TTL_SECONDS,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=max(1.0, fas._FUND_HISTORY_HEARTBEAT_INTERVAL_SECONDS))
        try:
            backend.release_if_owner(lease_key, owner_token)
        except Exception:
            pass
        active = backend.get(fas._history_active_key(normalized))
        if active == job_id:
            backend.delete(fas._history_active_key(normalized))


def _allocation_history_job_key(fund_code: str, lookback_days: int) -> str:
    return f"api:fund-allocation-history:job:{fund_code}:{lookback_days}"


def _allocation_history_active_key(fund_code: str, lookback_days: int) -> str:
    return f"api:fund-allocation-history:active:{fund_code}:{lookback_days}"


def _allocation_history_job_get(fund_code: str, lookback_days: int, job_id: str) -> Optional[Dict[str, Any]]:
    fas = _fas()
    value = fas._get_cache().get(f"{fas._allocation_history_job_key(fund_code, lookback_days)}:{job_id}")
    return dict(value) if isinstance(value, dict) else None


def _allocation_history_job_set(job: Dict[str, Any]) -> None:
    fas = _fas()
    fas._get_cache().set(
        f"{fas._allocation_history_job_key(str(job['fund_code']), int(job['lookback_days']))}:{job['job_id']}",
        job,
        ttl_seconds=fas._FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS,
    )


def _allocation_history_job_public(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job:
        return None
    return {
        key: job.get(key)
        for key in (
            "job_id",
            "fund_code",
            "lookback_days",
            "status",
            "requested_at",
            "started_at",
            "finished_at",
            "error",
        )
    }


def _run_allocation_history_refresh_job(fund_code: str, lookback_days: int, job_id: str) -> None:
    fas = _fas()
    from app import fund_service as _fund_service

    refresh_fund_allocations_history = getattr(
        fas, "refresh_fund_allocations_history", _fund_service.refresh_fund_allocations_history
    )

    normalized = fund_code.strip().upper()
    backend = fas._get_cache()
    job = fas._allocation_history_job_get(normalized, lookback_days, job_id)
    if not job:
        return
    active_key = fas._allocation_history_active_key(normalized, lookback_days)
    lease_key = f"{active_key}:lease"
    owner = fas.uuid.uuid4().hex
    if not backend.set_if_absent(lease_key, owner, ttl_seconds=fas._FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS):
        # A different worker has the shared lease. Its result will populate the
        # same file/cache; keep this job informational rather than duplicating
        # upstream TEFAS traffic.
        job.update({"status": "superseded", "finished_at": fas._fund_refresh_now_iso(), "error": None})
        fas._allocation_history_job_set(job)
        return

    heartbeat_stop = fas.threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(max(1.0, fas._FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS)):
            backend.renew_if_owner(
                lease_key,
                owner,
                ttl_seconds=fas._FUND_ALLOCATION_HISTORY_LEASE_TTL_SECONDS,
            )
            backend.renew_if_owner(
                active_key,
                job_id,
                ttl_seconds=fas._FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS,
            )

    heartbeat_thread = fas.threading.Thread(
        target=heartbeat,
        name=f"fund-allocation-history-heartbeat-{normalized}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        job.update({"status": "running", "started_at": fas._fund_refresh_now_iso(), "error": None})
        fas._allocation_history_job_set(job)
        refresh_fund_allocations_history(
            fas.CONFIG.paths.processed_dir,
            normalized,
            lookback_days=lookback_days,
            allow_daily_fallback=True,
        )
        job.update({"status": "succeeded", "finished_at": fas._fund_refresh_now_iso(), "error": None})
        fas._allocation_history_job_set(job)
        fas._clear_single_fund_l1_caches(normalized)
    except Exception as exc:
        job.update({"status": "failed", "finished_at": fas._fund_refresh_now_iso(), "error": str(exc)})
        fas._allocation_history_job_set(job)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=max(1.0, fas._FUND_ALLOCATION_HISTORY_HEARTBEAT_INTERVAL_SECONDS))
        backend.release_if_owner(lease_key, owner)
        if backend.get(active_key) == job_id:
            backend.delete(active_key)


def _start_allocation_history_refresh_job(fund_code: str, lookback_days: int) -> Dict[str, Any]:
    """Start at most one allocation-history refresh for a fund/range pair."""

    fas = _fas()
    normalized = fund_code.strip().upper()
    bounded_lookback = max(1, min(365, int(lookback_days)))
    backend = fas._get_cache()
    active_key = fas._allocation_history_active_key(normalized, bounded_lookback)
    with backend.lock(f"fund-allocation-history-state:{normalized}:{bounded_lookback}", timeout=5.0) as acquired:
        if acquired:
            active_id = backend.get(active_key)
            if active_id:
                existing = fas._allocation_history_job_get(normalized, bounded_lookback, str(active_id))
                if existing and str(existing.get("status")) in {"queued", "running"}:
                    return existing
            job = {
                "job_id": fas.uuid.uuid4().hex,
                "fund_code": normalized,
                "lookback_days": bounded_lookback,
                "status": "queued",
                "requested_at": fas._fund_refresh_now_iso(),
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            fas._allocation_history_job_set(job)
            backend.set(active_key, job["job_id"], ttl_seconds=fas._FUND_ALLOCATION_HISTORY_JOB_TTL_SECONDS)
            try:
                fas._FUND_ALLOCATION_HISTORY_EXECUTOR.submit(
                    fas._run_allocation_history_refresh_job,
                    normalized,
                    bounded_lookback,
                    str(job["job_id"]),
                )
            except Exception as exc:
                job.update({"status": "failed", "finished_at": fas._fund_refresh_now_iso(), "error": str(exc)})
                fas._allocation_history_job_set(job)
                backend.delete(active_key)
            return job
    active_id = backend.get(active_key)
    if active_id:
        existing = fas._allocation_history_job_get(normalized, bounded_lookback, str(active_id))
        if existing:
            return existing
    # A short lock contention should never turn into synchronous TEFAS work.
    return {
        "job_id": None,
        "fund_code": normalized,
        "lookback_days": bounded_lookback,
        "status": "pending",
        "requested_at": fas._fund_refresh_now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": "allocation refresh queue is busy",
    }
