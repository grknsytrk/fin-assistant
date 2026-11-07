"""Centralized Server-Sent Events (SSE) broadcaster for market stock updates.

Replaces per-client worker thread polling loops with a single shared broadcaster
task fanning out stock deltas to connected clients via bounded asyncio queues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger("ragfin.market_broadcaster")


def _resolve_market_fn(name: str) -> Callable[..., Any]:
    market_router = sys.modules.get("app.routers.market")
    if market_router and hasattr(market_router, name):
        return getattr(market_router, name)
    import app.market_service as ms
    return getattr(ms, name)


def _resolve_sleep_fn() -> Callable[[float], Any]:
    market_router = sys.modules.get("app.routers.market")
    if market_router and hasattr(market_router, "asyncio"):
        router_asyncio = getattr(market_router, "asyncio")
        if hasattr(router_asyncio, "sleep"):
            return getattr(router_asyncio, "sleep")
    return asyncio.sleep


class MarketStreamBroadcaster:
    """Singleton background broadcaster for streaming BIST stock market deltas."""

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._queue_indices: Dict[asyncio.Queue, str] = {}
        self._queue_requests: Dict[asyncio.Queue, Any] = {}
        self._last_rows: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._latest_event: Dict[str, str] = {}
        self._prev_snapshots: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
        self._prev_benchmarks: Dict[str, Tuple[Any, ...]] = {}

    @property
    def total_subscribers(self) -> int:
        return len(self._subscribers)

    def _fanout(self, event_msg: str, target_index: Optional[str] = None) -> None:
        """Deliver an SSE message to active subscriber queues without blocking."""
        for q in list(self._subscribers):
            if target_index is not None:
                q_index = self._queue_indices.get(q)
                if q_index is not None and q_index != target_index:
                    continue
            try:
                q.put_nowait(event_msg)
            except asyncio.QueueFull:
                pass

    async def subscribe(
        self,
        index: str = "XUTUM",
        once: bool = False,
        request: Optional[Any] = None,
    ) -> AsyncIterator[str]:
        """Subscribe to market delta events for the specified stock index."""
        normalize_fn = _resolve_market_fn("_normalize_stock_index")
        norm_index = normalize_fn(index)

        if once:
            if norm_index in self._latest_event:
                yield self._latest_event[norm_index]
                return
            market_stocks_payload_fn = _resolve_market_fn("_market_stocks_payload")
            compute_delta_fn = _resolve_market_fn("_compute_market_stocks_delta")
            is_market_open_fn = _resolve_market_fn("_is_bist_market_open")
            payload = await asyncio.to_thread(market_stocks_payload_fn, index_name=norm_index)
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            benchmarks = payload.get("benchmarks") if isinstance(payload.get("benchmarks"), dict) else {}
            changed, _ = compute_delta_fn({}, rows)
            market_open = is_market_open_fn()
            data = {
                "index": norm_index,
                "as_of": payload.get("as_of") or datetime.now(timezone.utc).isoformat(),
                "market_open": market_open,
                "count": len(changed),
                "rows": changed,
                "benchmarks": benchmarks,
            }
            yield f"event: stocks_delta\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            return

        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        async with self._lock:
            self._subscribers.add(queue)
            self._queue_indices[queue] = norm_index
            if request is not None:
                self._queue_requests[queue] = request
            if norm_index in self._latest_event:
                try:
                    queue.put_nowait(self._latest_event[norm_index])
                except asyncio.QueueFull:
                    pass
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._broadcaster_loop())

        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield msg
        finally:
            async with self._lock:
                self._subscribers.discard(queue)
                self._queue_indices.pop(queue, None)
                self._queue_requests.pop(queue, None)
                if len(self._subscribers) == 0 and self._task and not self._task.done():
                    self._task.cancel()
                    self._task = None

    async def _broadcaster_loop(self) -> None:
        """Single background task fetching stock data and broadcasting deltas."""
        last_heartbeat = time.monotonic()
        tick = 0
        try:
            while True:
                async with self._lock:
                    if len(self._subscribers) == 0:
                        break
                    # Check disconnected clients per tick
                    for q in list(self._subscribers):
                        req = self._queue_requests.get(q)
                        if req is not None:
                            try:
                                if await req.is_disconnected():
                                    self._subscribers.discard(q)
                                    self._queue_indices.pop(q, None)
                                    self._queue_requests.pop(q, None)
                                    q.put_nowait(None)
                            except Exception:
                                pass
                    if len(self._subscribers) == 0:
                        break

                market_stocks_payload_fn = _resolve_market_fn("_market_stocks_payload")
                compute_delta_fn = _resolve_market_fn("_compute_market_stocks_delta")
                is_market_open_fn = _resolve_market_fn("_is_bist_market_open")
                symbols_for_index_fn = _resolve_market_fn("_market_stock_symbols_for_index")

                market_open = is_market_open_fn()
                try:
                    # Single worker thread execution shared by all subscribers
                    payload = await asyncio.to_thread(market_stocks_payload_fn, index_name="XUTUM")
                    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
                    benchmarks = payload.get("benchmarks") if isinstance(payload.get("benchmarks"), dict) else {}

                    benchmarks_snapshot = {
                        k: (v.get("price"), v.get("change"), v.get("change_pct"), v.get("as_of"))
                        for k, v in benchmarks.items()
                        if isinstance(v, dict)
                    }
                    benchmarks_changed = benchmarks_snapshot != self._prev_benchmarks
                    if benchmarks_changed:
                        self._prev_benchmarks = benchmarks_snapshot

                    async with self._lock:
                        active_indices = {self._queue_indices.get(q, "XUTUM") for q in self._subscribers}
                    if not active_indices:
                        active_indices = {"XUTUM"}

                    for target_index in active_indices:
                        if target_index == "XUTUM":
                            target_rows = rows
                        else:
                            allowed_syms = set(symbols_for_index_fn(target_index))
                            target_rows = [
                                r for r in rows
                                if str(r.get("company") or r.get("symbol") or "").upper() in allowed_syms
                            ]

                        last_rows_for_index = self._last_rows.get(target_index)
                        if target_rows is last_rows_for_index:
                            changed = []
                        else:
                            prev_snap = self._prev_snapshots.get(target_index, {})
                            changed, next_snap = compute_delta_fn(prev_snap, target_rows)
                            self._prev_snapshots[target_index] = next_snap
                            self._last_rows[target_index] = target_rows

                        if changed or benchmarks_changed or tick == 0:
                            data = {
                                "index": target_index,
                                "as_of": payload.get("as_of") or datetime.now(timezone.utc).isoformat(),
                                "market_open": market_open,
                                "count": len(changed),
                                "rows": changed,
                                "benchmarks": benchmarks,
                            }
                            event_msg = f"event: stocks_delta\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                            self._latest_event[target_index] = event_msg
                            self._fanout(event_msg, target_index=target_index)
                            last_heartbeat = time.monotonic()

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOGGER.debug("broadcaster loop iteration error: %s", exc)

                tick += 1
                now_mono = time.monotonic()
                if now_mono - last_heartbeat >= 15.0:
                    heartbeat_msg = ": heartbeat\n\n"
                    self._fanout(heartbeat_msg)
                    last_heartbeat = now_mono

                sleep_seconds = 3.0 if market_open else 30.0
                sleep_fn = _resolve_sleep_fn()
                await sleep_fn(sleep_seconds)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Cancel the broadcaster background loop and clear all subscribers."""
        async with self._lock:
            task = self._task
            self._task = None
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._subscribers.clear()
            self._queue_indices.clear()
            self._queue_requests.clear()
            self._last_rows.clear()
            self._prev_snapshots.clear()
            self._latest_event.clear()


_BROADCASTER = MarketStreamBroadcaster()


def get_market_broadcaster() -> MarketStreamBroadcaster:
    """Return the global singleton MarketStreamBroadcaster instance."""
    return _BROADCASTER
