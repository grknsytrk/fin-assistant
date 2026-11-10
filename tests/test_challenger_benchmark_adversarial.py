"""Adversarial stress-test suite for scripts/benchmark_supabase_egress.py.

Empirically validates:
1. Standalone simulation execution without hanging, producing valid telemetry diffs
   for both supabase_db_metrics and supabase_json_cache_metrics.
2. Threshold assertions: --assert-egress-under exits code 2 on exceeded, code 0 within limit.
3. Offline backend handling: Exit code 1 with clean fatal error (0 python traceback lines).
4. Report generation: JSON (--output) and Markdown (--markdown-output) validity and directory creation.
5. ETag tracking and HTTP 304 handling across iterated requests.
6. Stress / boundary conditions (large iteration count, whitespace in flags, extreme fail-under).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from app.api import app
from scripts.benchmark_supabase_egress import (
    BenchmarkClient,
    BenchmarkRunner,
    HealthDbMetrics,
    TableMetrics,
    calculate_benchmark_results,
    extract_health_db_metrics,
    main,
    run_benchmark,
)

PYTHON_EXE = sys.executable
BENCHMARK_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "benchmark_supabase_egress.py")


def run_benchmark_cli(args: List[str], timeout: float = 20.0) -> subprocess.CompletedProcess:
    """Helper to run benchmark CLI in a subprocess with repo PYTHONPATH."""
    cmd = [PYTHON_EXE, BENCHMARK_SCRIPT] + args
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


class TestStandaloneSimulationExecution:
    """Empirical verification of standalone simulation execution & telemetry diffs."""

    def test_standalone_cli_execution_completes_and_reports(self) -> None:
        """Standalone simulation runs reliably without hanging and exits code 0."""
        proc = run_benchmark_cli(["--standalone", "--mock", "--accelerated", "--iterations", "3"])
        assert proc.returncode == 0, f"Process failed with code {proc.returncode}:\n{proc.stderr}"
        stdout = proc.stdout
        assert "SUPABASE EGRESS & WORKLOAD BENCHMARK REPORT" in stdout
        assert "Database Egress by Table (`supabase_db_metrics`)" in stdout
        assert "JSON Cache Counters (`supabase_json_cache_metrics`)" in stdout
        assert "HTTP Client Traffic Summary" in stdout

    def test_standalone_programmatic_telemetry_diffs(self) -> None:
        """Verify valid telemetry diffs for both supabase_db_metrics and supabase_json_cache_metrics."""
        results = run_benchmark(
            standalone=True,
            mock=True,
            accelerated=True,
            iterations=3,
        )

        # 1. DB Metrics Diffs
        assert "db_egress" in results
        assert "telemetry_delta" in results
        db_egress = results["db_egress"]
        assert db_egress["total_reads_delta"] >= 0
        assert db_egress["total_bytes_delta"] >= 0
        assert "by_table" in db_egress

        expected_tables = (
            "ragfin_json_cache",
            "fund_snapshots",
            "fund_prices",
            "company_financial_snapshots",
            "instruments",
        )
        for tbl in expected_tables:
            assert tbl in db_egress["by_table"], f"Missing table {tbl} in db_egress"
            stat = db_egress["by_table"][tbl]
            assert "reads_delta" in stat
            assert "bytes_delta" in stat
            assert stat["reads_delta"] >= 0
            assert stat["bytes_delta"] >= 0
            assert "bytes_delta_formatted" in stat
            assert "bytes_per_hour_projected_formatted" in stat

        # 2. JSON Cache Metrics Diffs
        assert "json_cache" in results
        jc = results["json_cache"]
        expected_jc_keys = (
            "remote_reads",
            "remote_hits",
            "remote_misses",
            "local_hits",
            "coalesced_hits",
            "approx_remote_payload_bytes",
        )
        for k in expected_jc_keys:
            assert k in jc, f"Missing json_cache metric {k}"
            assert "before" in jc[k]
            assert "after" in jc[k]
            assert "delta" in jc[k]
            assert jc[k]["delta"] >= 0


class TestThresholdAssertions:
    """Empirical verification of --assert-egress-under behavior."""

    def test_assert_egress_under_passes_when_within_limit(self) -> None:
        """Exits code 0 when total DB egress is strictly below limit."""
        # Mock run produces ~31.74 KB (~0.03 MB), limit of 10.0 MB passes cleanly
        proc = run_benchmark_cli([
            "--standalone",
            "--mock",
            "--accelerated",
            "--iterations",
            "2",
            "--assert-egress-under",
            "10.0",
        ])
        assert proc.returncode == 0, f"Expected code 0, got {proc.returncode}. Stderr: {proc.stderr}"
        assert "[ASSERTION PASSED]" in proc.stdout
        assert "is within limit 10.00 MB" in proc.stdout

    def test_assert_egress_under_fails_with_code_2_when_exceeded(self) -> None:
        """Exits code 2 when total DB egress exceeds limit."""
        # 1. CLI verification: When threshold is set below actual egress (e.g. -0.000001 MB),
        # actual_mb (>= 0.0 MB) strictly exceeds limit -> exits with returncode 2.
        proc = run_benchmark_cli([
            "--standalone",
            "--mock",
            "--accelerated",
            "--iterations",
            "2",
            "--assert-egress-under",
            "-0.000001",
        ])
        assert proc.returncode == 2, f"Expected code 2, got {proc.returncode}. Output: {proc.stdout}\nStderr: {proc.stderr}"
        assert "[ASSERTION FAILED]" in proc.stderr
        assert "exceeds limit" in proc.stderr

        # 2. Programmatic verification: Mocking health telemetry with 15 MB egress and threshold 10.0 MB
        class MockHighEgressClient:
            standalone = True
            mock = True

            def __init__(self):
                self._call_count = 0

            def get(self, path: str, params: Any = None, headers: Any = None) -> Any:
                class MockResp:
                    def __init__(self, data: Dict[str, Any]):
                        self.status_code = 200
                        self.content = json.dumps(data).encode("utf-8")
                        self.headers = {}

                    def json(self):
                        return json.loads(self.content.decode("utf-8"))

                if path == "/health":
                    self._call_count += 1
                    # Before call: 0 bytes. After call: 15 MB (15,728,640 bytes)
                    bytes_val = 15 * 1024 * 1024 if self._call_count > 1 else 0
                    return MockResp({
                        "supabase_db_metrics": {
                            "window_seconds": 600.0,
                            "tables": {
                                "ragfin_json_cache": {"reads": 5, "bytes": bytes_val},
                                "fund_snapshots": {"reads": 0, "bytes": 0},
                                "fund_prices": {"reads": 0, "bytes": 0},
                                "company_financial_snapshots": {"reads": 0, "bytes": 0},
                                "instruments": {"reads": 0, "bytes": 0},
                            },
                        },
                        "supabase_json_cache_metrics": {"totals": {}},
                    })
                return MockResp({"status": "ok"})

            def close(self):
                pass

        runner = BenchmarkRunner(
            client=MockHighEgressClient(),
            iterations=1,
            accelerated=True,
            assert_egress_under=10.0,
        )
        res = runner.run()
        actual_mb = res["db_egress"]["total_bytes_delta"] / (1024 * 1024)
        assert actual_mb == 15.0
        # When actual (15.0 MB) > assert_egress_under (10.0 MB), main() returns code 2
        assert actual_mb > 10.0

    def test_assert_egress_under_exact_boundary(self) -> None:
        """Verify boundary condition behavior (actual == threshold)."""
        # Run first to get exact byte count
        results = run_benchmark(standalone=True, mock=True, accelerated=True, iterations=1)
        actual_mb = results["db_egress"]["total_bytes_delta"] / (1024 * 1024)

        # Equal threshold should PASS (code 0) because condition is >
        proc = run_benchmark_cli([
            "--standalone",
            "--mock",
            "--accelerated",
            "--iterations",
            "1",
            "--assert-egress-under",
            str(actual_mb),
        ])
        # In mock mode, the byte count is deterministic for iterations=1
        assert proc.returncode == 0
        assert "[ASSERTION PASSED]" in proc.stdout


class TestOfflineBackendHandling:
    """Empirical verification of offline backend resilience with 0 tracebacks."""

    def test_offline_backend_refused_connection_exits_code_1_no_traceback(self) -> None:
        """Unreachable port exits code 1 with clean fatal error and zero python tracebacks."""
        # Target an inactive local port
        proc = run_benchmark_cli([
            "--base-url",
            "http://127.0.0.1:59998",
            "--timeout",
            "0.5",
        ], timeout=5.0)

        assert proc.returncode == 1, f"Expected code 1, got {proc.returncode}"
        combined_output = proc.stdout + proc.stderr
        assert "[FATAL]" in combined_output or "[ERROR]" in combined_output
        assert "Backend at http://127.0.0.1:59998 is unreachable." in combined_output or "Connection failed" in combined_output

        # Zero python traceback lines
        assert "Traceback (most recent call last):" not in proc.stderr
        assert "Traceback (most recent call last):" not in proc.stdout

    def test_offline_backend_nonexistent_host_exits_code_1_no_traceback(self) -> None:
        """Unresolvable domain exits code 1 with clean error and zero tracebacks."""
        proc = run_benchmark_cli([
            "--base-url",
            "http://nonexistent.ragfin.invalid:8000",
            "--timeout",
            "0.5",
        ], timeout=5.0)

        assert proc.returncode == 1
        combined_output = proc.stdout + proc.stderr
        assert "[FATAL]" in combined_output or "[ERROR]" in combined_output
        assert "Traceback (most recent call last):" not in proc.stderr
        assert "Traceback (most recent call last):" not in proc.stdout


class TestReportGeneration:
    """Empirical verification of JSON and Markdown report generation."""

    def test_json_and_markdown_output_generation(self, tmp_path: Path) -> None:
        """Generates valid JSON and Markdown files with complete schemas in nested dirs."""
        out_json = tmp_path / "reports" / "nested" / "egress_benchmark.json"
        out_md = tmp_path / "reports" / "nested" / "egress_benchmark.md"

        proc = run_benchmark_cli([
            "--standalone",
            "--mock",
            "--accelerated",
            "--iterations",
            "2",
            "--output",
            str(out_json),
            "--markdown-output",
            str(out_md),
        ])
        assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
        assert out_json.exists(), f"JSON output not created at {out_json}"
        assert out_md.exists(), f"Markdown output not created at {out_md}"

        # 1. Inspect JSON Structure
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "telemetry_delta" in data
        assert "db_egress" in data
        assert "json_cache" in data
        assert "baseline_comparison" in data
        assert "request_summary" in data
        assert "http_traffic" in data
        assert "raw_snapshots" in data
        assert "requests" in data

        assert isinstance(data["requests"], list)
        assert len(data["requests"]) > 0
        sample_req = data["requests"][0]
        assert "path" in sample_req
        assert "status_code" in sample_req
        assert "latency_ms" in sample_req
        assert "wire_bytes" in sample_req
        assert "is_304" in sample_req

        # 2. Inspect Markdown Structure
        md_text = out_md.read_text(encoding="utf-8")
        assert md_text.startswith("# Benchmark Report:")
        assert "## Database Egress by Table" in md_text
        assert "| Table | Reads (Before) | Reads (After) | Delta Reads |" in md_text
        assert "## JSON Cache Counters" in md_text
        assert "## HTTP Traffic Summary" in md_text
        assert "| Endpoint | Calls | 200 OK | 304 NotMod |" in md_text


class TestETagTrackingAnd304Handling:
    """Empirical verification of ETag tracking and HTTP 304 conditional requests."""

    def test_etag_tracking_across_iterated_requests(self) -> None:
        """Iterated requests send If-None-Match when ETag received and record 304 responses."""
        captured_headers: List[Dict[str, str]] = []

        class MockETagClient:
            def __init__(self) -> None:
                self.call_count: Dict[str, int] = {}
                self.standalone = True
                self.mock = True

            def get(self, path: str, params: Any = None, headers: Any = None) -> Any:
                h = dict(headers or {})
                captured_headers.append(h)
                self.call_count[path] = self.call_count.get(path, 0) + 1

                class MockResponse:
                    def __init__(self, status_code: int, content: bytes, resp_headers: Dict[str, str]):
                        self.status_code = status_code
                        self.content = content
                        self.headers = resp_headers

                    def json(self):
                        return json.loads(self.content.decode("utf-8"))

                # /health endpoint returns mock metrics
                if path == "/health":
                    return MockResponse(
                        200,
                        json.dumps({
                            "supabase_db_metrics": {
                                "window_seconds": 600.0,
                                "tables": {t: {"reads": 1, "bytes": 100} for t in (
                                    "ragfin_json_cache", "fund_snapshots", "fund_prices",
                                    "company_financial_snapshots", "instruments"
                                )},
                            },
                            "supabase_json_cache_metrics": {"totals": {"remote_reads": 1}},
                        }).encode("utf-8"),
                        {"content-type": "application/json"},
                    )

                etag_tag = f'W/"version-{path}"'
                if h.get("If-None-Match") == etag_tag:
                    # Return 304 Not Modified
                    return MockResponse(304, b"", {"etag": etag_tag, "content-length": "0"})
                else:
                    # Return 200 OK with ETag
                    body = json.dumps({"status": "ok", "path": path}).encode("utf-8")
                    return MockResponse(200, body, {"etag": etag_tag, "content-length": str(len(body))})

            def close(self) -> None:
                pass

        mock_client = MockETagClient()
        runner = BenchmarkRunner(
            client=mock_client,
            iterations=2,
            accelerated=True,
            use_etags=True,
        )
        results = runner.run()

        # In iteration 1, endpoints return 200 and set ETag.
        # In iteration 2, runner sends If-None-Match, resulting in 304 Not Modified!
        assert results["request_summary"]["status_304_count"] > 0
        assert results["http_traffic"]["status_304"] > 0

        # Verify If-None-Match headers were sent in captured headers
        if_none_matches = [h["If-None-Match"] for h in captured_headers if "If-None-Match" in h]
        assert len(if_none_matches) > 0

    def test_no_etags_flag_disables_conditional_headers(self) -> None:
        """--no-etags flag sets use_etags: False and suppresses If-None-Match."""
        results = run_benchmark(
            standalone=True,
            mock=True,
            accelerated=True,
            iterations=2,
            no_etags=True,
        )
        assert results["metadata"]["use_etags"] is False
        assert results["request_summary"]["status_304_count"] == 0


class TestAdversarialEdgeCases:
    """Stress testing edge cases and input variations."""

    def test_large_iteration_stress_does_not_hang_or_leak(self) -> None:
        """Executes 50 iterations in accelerated mode in under 3 seconds."""
        t0 = time.perf_counter()
        results = run_benchmark(
            standalone=True,
            mock=True,
            accelerated=True,
            iterations=50,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"50 iterations took too long: {elapsed:.2f}s"
        assert len(results["requests"]) == 50 * 9  # 9 requests per iteration
        assert results["baseline_comparison"]["target_met"] is True

    def test_fund_codes_with_whitespace_and_mixed_case(self) -> None:
        """Handles malformed fund code strings gracefully."""
        runner = BenchmarkRunner(
            standalone=True,
            mock=True,
            accelerated=True,
            fund_codes="  nnf , , ti2 , MAC  ",
            iterations=1,
        )
        assert runner.fund_codes == ["NNF", "TI2", "MAC"]

    def test_fail_under_threshold_enforcement(self) -> None:
        """When fail-under threshold is unmet, run exits code 1."""
        # Force fail-under to 100.01% (impossible since max savings is 100.00%)
        proc = run_benchmark_cli([
            "--standalone",
            "--mock",
            "--accelerated",
            "--iterations",
            "1",
            "--fail-under",
            "100.01",
        ])
        assert proc.returncode == 1, f"Expected code 1 when target not met, got {proc.returncode}"
        assert "Verdict               : FAILED (< 100.0%)" in proc.stdout
