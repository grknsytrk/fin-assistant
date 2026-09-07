"""Offline request-path benchmark; no provider requests or production cache writes.

Run: .external/venv311/Scripts/python.exe scripts/benchmark_kap_opening.py
Baseline models the old cold-response path with a controlled 150 ms factory.
Disk/warm cases use a synthetic statement fixture. Results exclude browser/network.
"""
import json
from dataclasses import replace
import os
import statistics
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["RAGFIN_CACHE_BACKEND"] = "memory"
from app import api, cache


def measure(fn, count=12):
    durations = []
    for _ in range(count):
        start = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - start) * 1000)
    ordered = sorted(durations)
    return {"p50_ms": round(statistics.median(durations), 2), "p95_ms": round(ordered[-1], 2), "samples": count}


def main():
    cache.reset_cache_for_tests()
    symbol = "BENCH"
    from src.kap_fetcher import KAP_CACHE_SCHEMA_VERSION
    workspace = Path(__file__).resolve().parents[1]
    fixture_dir = tempfile.TemporaryDirectory(prefix="kap-bench-", dir=workspace / "tmp")
    fixture_root = Path(fixture_dir.name).resolve()
    assert fixture_root.is_relative_to(workspace / "tmp")
    (fixture_root / "kap_cache").mkdir()
    raw = {"ok": True, "company": symbol, "stock_code": symbol,
           "schema_version": KAP_CACHE_SCHEMA_VERSION, "quarters": [
               {"year": 2021 + i // 4, "period": 1 + i % 4,
                "quarter": f"{2021 + i // 4}Q{1 + i % 4}",
                "metrics": {"ozkaynaklar": 100},
                "metrics_quarterly": {"net_kar": i, "favok": i + 1}}
               for i in range(20)]}
    (fixture_root / "kap_cache" / f"{symbol}.json").write_text(json.dumps(raw), encoding="utf-8")
    config_patch = patch.object(api, "CONFIG", replace(api.CONFIG, paths=replace(api.CONFIG.paths, processed_dir=fixture_root)))
    config_patch.start()
    payload = api._cached_kap_snapshot_response(symbol)
    assert payload is not None
    def delayed_factory():
        time.sleep(0.15)
        return payload
    def baseline():
        cache.reset_cache_for_tests()
        return api._shared_swr_payload(cache_key="bench:baseline", factory=delayed_factory,
            fresh_ttl_seconds=300, stale_ttl_seconds=1800)
    results = {"baseline_cold_with_150ms_factory": measure(baseline)}
    cache.reset_cache_for_tests()
    with patch.object(api, "_schedule_swr_revalidation", return_value=True):
        results["disk_statement_response"] = measure(lambda: api.kap_snapshot(symbol, False, 5))
        entry = api._swr_cache_entry(payload, fresh_ttl_seconds=300, stale_ttl_seconds=1800)
        api._shared_cache_set(api._kap_snapshot_response_cache_key(symbol, 5), entry, 1800)
        results["warm_response"] = measure(lambda: api.kap_snapshot(symbol, False, 5))
        with patch.object(api, "_cached_kap_snapshot_response", return_value=None):
            results["empty_cache_pending_response"] = measure(lambda: api.kap_snapshot("MISSING", False, 5))
    config_patch.stop()
    fixture_dir.cleanup()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
