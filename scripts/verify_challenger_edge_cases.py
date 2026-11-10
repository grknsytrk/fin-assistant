#!/usr/bin/env python3
"""Empirical Challenger Verification Script.

Tests edge cases and adversarial scenarios against:
- SingleFlightCoordinator (thread crashes, timeouts, reentrancy, concurrency)
- MeteredCursorHarness (Turkish characters, None values, empty sets, multiple fetches)
- AcceleratedClock (negative jumps, large increments, concurrency)
- Benchmark and E2E components
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List

import orjson
import pytest

from tests.test_e2e_egress import (
    AcceleratedClock,
    MeteredConnectionHarness,
    MeteredCursorHarness,
    SingleFlightCoordinator,
    TRACKED_EGRESS_TABLES,
    evaluate_market_dynamic_freshness,
)


def log_result(suite: str, test_name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] [{suite}] {test_name}: {details}")


# ===========================================================================
# 1. SingleFlightCoordinator Adversarial Tests
# ===========================================================================

def test_single_flight_exception_propagation():
    """Verify standard Exception propagates to waiter threads."""
    coord = SingleFlightCoordinator()
    def faulty():
        time.sleep(0.02)
        raise ValueError("synthetic error")

    def waiter():
        return coord.execute("k1", faulty)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(waiter) for _ in range(5)]
        exceptions = []
        for f in futures:
            try:
                f.result()
            except ValueError as e:
                exceptions.append(str(e))
    assert len(exceptions) == 5, f"Expected 5 exceptions, got {len(exceptions)}"
    log_result("SingleFlight", "exception_propagation", True, "All 5 threads caught ValueError")


def test_single_flight_timeout_behavior():
    """Verify waiter behavior when factory takes longer than wait timeout.
    
    NOTE: Default timeout in SingleFlightCoordinator is 10.0s.
    We test what happens if event.wait() times out.
    """
    coord = SingleFlightCoordinator()
    
    # Check if execute allows timeout or if waiter returns None on timeout
    def slow_factory():
        time.sleep(0.2)
        return "slow_result"

    # We patch event.wait temporarily to timeout after 0.01s to observe behavior
    original_wait = threading.Event.wait
    try:
        def short_wait(self, timeout=None):
            return original_wait(self, timeout=0.01)
        
        threading.Event.wait = short_wait
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_owner = pool.submit(coord.execute, "slow_key", slow_factory)
            time.sleep(0.005) # ensure owner started
            f_waiter = pool.submit(coord.execute, "slow_key", slow_factory)
            
            waiter_res = f_waiter.result()
            owner_res = f_owner.result()

        # Did waiter return None because timeout wasn't checked?
        # When event.wait(0.01) returns False, box[0] is still None!
        timed_out_without_error = (waiter_res is None and owner_res == "slow_result")
        log_result(
            "SingleFlight",
            "timeout_silent_none_issue",
            not timed_out_without_error,
            f"Waiter returned {waiter_res!r} while owner was computing (silent timeout flaw: {timed_out_without_error})"
        )
    finally:
        threading.Event.wait = original_wait


def test_single_flight_base_exception():
    """Verify behavior when factory raises BaseException (e.g. KeyboardInterrupt)."""
    coord = SingleFlightCoordinator()
    class CustomBaseException(BaseException):
        pass

    def crash_factory():
        time.sleep(0.02)
        raise CustomBaseException("fatal interrupt")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_owner = pool.submit(coord.execute, "base_exc_key", crash_factory)
        time.sleep(0.005)
        f_waiters = [pool.submit(coord.execute, "base_exc_key", crash_factory) for _ in range(2)]

        owner_exc = None
        try:
            f_owner.result()
        except CustomBaseException as e:
            owner_exc = e

        waiter_results = []
        waiter_exceptions = []
        for w in f_waiters:
            try:
                waiter_results.append(w.result())
            except Exception as e:
                waiter_exceptions.append(e)

    # In SingleFlightCoordinator:
    # except Exception as exc: -> BaseException is NOT caught by except Exception!
    # So box[1] remains None! finally sets event.
    # Waiter wakes up: box[1] is None, box[0] is None -> waiter returns None!
    flaw_detected = any(r is None for r in waiter_results)
    log_result(
        "SingleFlight",
        "base_exception_handling",
        not flaw_detected,
        f"Owner raised BaseException; waiters returned {waiter_results} (silent None: {flaw_detected})"
    )


def test_single_flight_reentrancy_deadlock():
    """Verify what happens if factory calls execute with the same key (re-entrancy)."""
    coord = SingleFlightCoordinator()

    def recursive_factory():
        # Re-entrant call on same key
        return coord.execute("reentrant_key", lambda: "inner")

    # In SingleFlightCoordinator, recursive call with same key on same thread sees key in _in_flight
    # and calls event.wait(timeout=10.0), blocking until timeout!
    t0 = time.perf_counter()
    original_wait = threading.Event.wait
    try:
        def fast_wait(self, timeout=None):
            return original_wait(self, timeout=0.05)
        threading.Event.wait = fast_wait
        
        res = coord.execute("reentrant_key", recursive_factory)
        elapsed = time.perf_counter() - t0
        log_result(
            "SingleFlight",
            "reentrant_deadlock",
            False,
            f"Reentrant call blocked for {elapsed:.3f}s and returned {res!r} (deadlock detected!)"
        )
    except Exception as e:
        log_result("SingleFlight", "reentrant_deadlock", True, f"Reentrant call raised {e}")
    finally:
        threading.Event.wait = original_wait


def test_single_flight_high_concurrency():
    """Verify 50 threads across 10 keys simultaneously."""
    coord = SingleFlightCoordinator()
    counter = 0
    lock = threading.Lock()

    def work(key_id):
        nonlocal counter
        with lock:
            counter += 1
        time.sleep(0.01)
        return f"result_{key_id}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
        futures = [
            pool.submit(coord.execute, f"key_{i % 10}", lambda i=i: work(i % 10))
            for i in range(100)
        ]
        results = [f.result() for f in futures]

    assert len(results) == 100
    for i, r in enumerate(results):
        assert r == f"result_{i % 10}"
    log_result("SingleFlight", "high_concurrency_100_requests", True, f"Total underlying factory calls: {counter} <= 100")


# ===========================================================================
# 2. MeteredCursorHarness Tests
# ===========================================================================

def test_metered_cursor_turkish_characters():
    """Verify Turkish characters: ç, ğ, ı, ö, ş, ü, İ, Ğ, Ö, Ü, Ş, Ç."""
    turkish_data = [
        {"fund_code": "TI1", "name": "İş Portföy Şemsiye Fonu Çeşitlendirilmiş", "detail": "Öğleden sonra ığdır güneşi"},
        {"fund_code": "TI2", "name": "Ak Portföy BIST 100 Dışı Şirketler", "detail": "Çok yüksek riskli ürün"},
    ]
    harness = MeteredConnectionHarness({"fund_snapshots": turkish_data})
    with harness as conn:
        cur = conn.execute("SELECT * FROM fund_snapshots")
        rows = cur.fetchall()

    direct_bytes = len(orjson.dumps(turkish_data))
    measured_bytes = harness.measured_bytes["fund_snapshots"]
    match = (direct_bytes == measured_bytes)
    log_result("MeteredCursor", "turkish_characters", match, f"Direct: {direct_bytes} B, Harness: {measured_bytes} B")


def test_metered_cursor_none_values():
    """Verify None values in rows."""
    none_data = [
        {"fund_code": "TI1", "price": None, "daily_return": None, "metadata": None},
        {"fund_code": None, "price": None, "daily_return": 0.0, "metadata": {"note": None}},
    ]
    harness = MeteredConnectionHarness({"fund_snapshots": none_data})
    with harness as conn:
        cur = conn.execute("SELECT * FROM fund_snapshots")
        rows = cur.fetchall()

    direct_bytes = len(orjson.dumps(none_data))
    measured_bytes = harness.measured_bytes["fund_snapshots"]
    match = (direct_bytes == measured_bytes)
    log_result("MeteredCursor", "none_values", match, f"Direct: {direct_bytes} B, Harness: {measured_bytes} B")


def test_metered_cursor_empty_set_in_row():
    """Verify serialization when row contains a set object."""
    set_data = [
        {"fund_code": "TI1", "tags": {"equity", "bist30"}},
    ]
    harness = MeteredConnectionHarness({"fund_snapshots": set_data})
    with harness as conn:
        cur = conn.execute("SELECT * FROM fund_snapshots")
        rows = cur.fetchall()

    measured_bytes = harness.measured_bytes["fund_snapshots"]
    # orjson fails on set, so fallback to json.dumps(default=str) must trigger
    expected_fallback_bytes = len(json.dumps(set_data, default=str).encode("utf-8"))
    match = (measured_bytes == expected_fallback_bytes)
    log_result("MeteredCursor", "set_fallback_serialization", match, f"Fallback bytes: {measured_bytes} B (expected: {expected_fallback_bytes} B)")


def test_metered_cursor_empty_rows_and_multiple_fetchall():
    """Verify behavior on empty queries and repeated fetchall calls."""
    harness = MeteredConnectionHarness({"fund_prices": []})
    with harness as conn:
        cur = conn.execute("SELECT * FROM fund_prices")
        r1 = cur.fetchall()
        r2 = cur.fetchall()

    # Did second fetchall recount reads and empty list bytes?
    reads = harness.measured_reads["fund_prices"]
    bytes_count = harness.measured_bytes["fund_prices"]
    empty_list_bytes = len(orjson.dumps([]))
    log_result(
        "MeteredCursor",
        "empty_rows_multiple_fetchall",
        True,
        f"Calls: 2, Reads counted: {reads}, Bytes: {bytes_count} (each empty fetchall added {empty_list_bytes} B)"
    )


def test_metered_cursor_fetchone_vs_fetchall():
    """Compare byte counting between fetchone loop vs fetchall."""
    data = [{"id": i, "val": "abc"} for i in range(5)]
    harness1 = MeteredConnectionHarness({"fund_snapshots": data})
    with harness1 as conn:
        cur = conn.execute("SELECT * FROM fund_snapshots")
        cur.fetchall()
    fetchall_bytes = harness1.measured_bytes["fund_snapshots"]
    fetchall_reads = harness1.measured_reads["fund_snapshots"]

    harness2 = MeteredConnectionHarness({"fund_snapshots": data})
    with harness2 as conn:
        cur = conn.execute("SELECT * FROM fund_snapshots")
        for _ in range(5):
            cur.fetchone()
    fetchone_bytes = harness2.measured_bytes["fund_snapshots"]
    fetchone_reads = harness2.measured_reads["fund_snapshots"]

    # In fetchone: each call does len(orjson.dumps([row])) and reads += 1
    # fetchone_reads will be 5, whereas fetchall_reads will be 1!
    log_result(
        "MeteredCursor",
        "fetchone_vs_fetchall_discrepancy",
        True,
        f"Fetchall: {fetchall_reads} reads, {fetchall_bytes} B; Fetchone loop: {fetchone_reads} reads, {fetchone_bytes} B"
    )


# ===========================================================================
# 3. AcceleratedClock Adversarial Tests
# ===========================================================================

def test_accelerated_clock_negative_jumps():
    """Verify behavior on negative time jumps."""
    clock = AcceleratedClock(1000.0)
    assert clock.monotonic() == 1000.0
    
    clock.advance(-200.0)
    t_after = clock.time()
    m_after = clock.monotonic()
    
    monotonic_decreased = (m_after < 1000.0)
    log_result(
        "AcceleratedClock",
        "negative_time_jump",
        True,
        f"Time: {t_after}, Monotonic: {m_after} (monotonic decreased: {monotonic_decreased})"
    )


def test_accelerated_clock_large_increments():
    """Verify behavior on extreme time increments (e.g. 1e12 seconds)."""
    clock = AcceleratedClock(1000.0)
    clock.advance(1e12)
    assert clock.time() == 1000.0 + 1e12
    assert clock.monotonic() == 1000.0 + 1e12
    
    # Check if small delta can still be resolved
    clock.advance(0.001)
    diff = clock.time() - (1000.0 + 1e12)
    precision_ok = abs(diff - 0.001) < 1e-6
    log_result(
        "AcceleratedClock",
        "large_increments_precision",
        precision_ok,
        f"Added 0.001s at 1e12 scale; measured diff = {diff:.6f}s (precision retained: {precision_ok})"
    )


def test_accelerated_clock_concurrent_advance():
    """Verify thread-safety of AcceleratedClock under concurrent advance."""
    clock = AcceleratedClock(0.0)
    
    def advance_worker():
        for _ in range(1000):
            clock.advance(1.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(advance_worker) for _ in range(10)]
        for f in futures:
            f.result()

    expected = 10000.0
    actual = clock.time()
    match = (actual == expected)
    log_result(
        "AcceleratedClock",
        "concurrent_advance_thread_safety",
        match,
        f"Expected: {expected}, Actual: {actual} (Lost updates: {expected - actual})"
    )


# ===========================================================================
# Main Execution
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EMPIRICAL CHALLENGER: ADVERSARIAL STRESS HARNESS")
    print("=" * 70)

    print("\n--- Testing SingleFlightCoordinator ---")
    test_single_flight_exception_propagation()
    test_single_flight_timeout_behavior()
    test_single_flight_base_exception()
    test_single_flight_reentrancy_deadlock()
    test_single_flight_high_concurrency()

    print("\n--- Testing MeteredCursorHarness ---")
    test_metered_cursor_turkish_characters()
    test_metered_cursor_none_values()
    test_metered_cursor_empty_set_in_row()
    test_metered_cursor_empty_rows_and_multiple_fetchall()
    test_metered_cursor_fetchone_vs_fetchall()

    print("\n--- Testing AcceleratedClock ---")
    test_accelerated_clock_negative_jumps()
    test_accelerated_clock_large_increments()
    test_accelerated_clock_concurrent_advance()

    print("\n" + "=" * 70)
    print("EMPIRICAL CHALLENGER VERIFICATION COMPLETE")
    print("=" * 70)
