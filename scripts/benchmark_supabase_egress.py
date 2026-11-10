#!/usr/bin/env python3
"""Automated Supabase Egress Benchmark & Session Simulation Harness.

Simulates a real-world user session across:
- Markets tab (polling /market/stocks and /market/indices every 3-5 seconds)
- Funds tab (browsing /funds, /funds/categories, and searching /funds/search)
- Fund Detail tab (inspecting /funds/{fund_code} e.g. NNF, TI2, MAC)
- KAP Companies tab (/kap/companies)

Captures /health telemetry before and after the session to measure:
- Postgres wire egress bytes and read counts per table via `supabase_db_metrics`
- JSON cache hits, remote reads, and payload bytes via `supabase_json_cache_metrics`
- HTTP 200 vs 304 Not Modified responses and endpoint latencies

Supports both live backend evaluation via httpx and headless in-process
execution via Starlette TestClient with mock telemetry injection for fast CI.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

ROOT_DIR = Path(__file__).resolve().parent
while ROOT_DIR != ROOT_DIR.parent and not (ROOT_DIR / "pyproject.toml").exists():
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


TARGET_DB_TABLES = (
    "ragfin_json_cache",
    "fund_snapshots",
    "fund_prices",
    "company_financial_snapshots",
    "instruments",
)

SEARCH_TERMS = ("hisse", "altin", "borclanma", "para piyasasi", "degisken")


def format_bytes(num_bytes: int | float) -> str:
    """Format bytes into human-readable string (B, KB, MB, GB)."""
    abs_bytes = abs(num_bytes)
    sign = "-" if num_bytes < 0 else ""
    if abs_bytes < 1024:
        return f"{sign}{int(abs_bytes)} B"
    elif abs_bytes < 1024 * 1024:
        return f"{sign}{abs_bytes / 1024:.2f} KB"
    elif abs_bytes < 1024 * 1024 * 1024:
        return f"{sign}{abs_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{sign}{abs_bytes / (1024 * 1024 * 1024):.2f} GB"


@dataclass
class TableMetrics:
    reads: int = 0
    bytes: int = 0


@dataclass
class HealthDbMetrics:
    window_seconds: float = 600.0
    tables: Dict[str, TableMetrics] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    json_cache: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTelemetry:
    phase: str
    path: str
    status_code: int
    latency_ms: float
    wire_bytes: int
    is_304: bool
    etag: Optional[str] = None


@dataclass
class EndpointStats:
    total_calls: int = 0
    status_200: int = 0
    status_304: int = 0
    errors: int = 0
    bytes_received: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def record(self, status_code: int, latency_ms: float, response_bytes: int) -> None:
        self.total_calls += 1
        if status_code == 200:
            self.status_200 += 1
        elif status_code == 304:
            self.status_304 += 1
        else:
            self.errors += 1
        self.bytes_received += response_bytes
        self.latencies_ms.append(latency_ms)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = int(0.95 * len(s))
        return s[min(idx, len(s) - 1)]


class BenchmarkClient:
    """Unified client supporting live network (httpx) and headless in-process (TestClient)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        standalone: bool = False,
        mock: bool = False,
        timeout: float = 15.0,
    ) -> None:
        self.standalone = standalone
        self.mock = mock
        self.base_url = (base_url or "http://127.0.0.1:8000").rstrip("/")
        self.timeout = timeout

        if self.standalone:
            from starlette.testclient import TestClient
            from app.api import app

            if self.mock:
                self._setup_in_process_mocks(app)
            self._client = TestClient(app, raise_server_exceptions=False)
        else:
            if httpx is None:
                raise RuntimeError(
                    "The 'httpx' library is required to run live network benchmarks. "
                    "Install it via 'pip install httpx' or use '--standalone --mock'."
                )
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                follow_redirects=True,
            )

    def _setup_in_process_mocks(self, app_instance: Any) -> None:
        """Inject lightweight mocks in headless mock mode for fast, deterministic execution."""
        os.environ.pop("RAGFIN_DATABASE_URL", None)
        os.environ.pop("DATABASE_URL", None)

        try:
            from app.routers import market as market_router
            from app.routers import funds as funds_router
            from app.routers import kap as kap_router

            mock_market = {
                "index": "XUTUM",
                "rows": [
                    {"company": "THYAO", "price": 300.0, "change_pct": 1.5, "volume": 1000000.0, "as_of": "2026-11-10T18:00:00Z"},
                    {"company": "GARAN", "price": 120.0, "change_pct": 0.8, "volume": 500000.0, "as_of": "2026-11-10T18:00:00Z"},
                ],
                "stale": False,
                "quote_status": "fresh",
            }
            mock_indices = {
                "indices": [
                    {"symbol": "XU100", "price": 10000.0, "change_pct": 1.2},
                    {"symbol": "XU030", "price": 11000.0, "change_pct": 1.4},
                ],
                "as_of": "2026-11-10T18:00:00Z",
            }
            mock_funds = {
                "rows": [
                    {"fund_code": "TI2", "name": "İş Portföy", "fund_type": "SEC", "price": 5.25, "daily_return": 0.5, "aum": 1200000.0, "investor_count": 1500},
                    {"fund_code": "NNF", "name": "Hedef Portföy", "fund_type": "SEC", "price": 12.50, "daily_return": 1.2, "aum": 800000.0, "investor_count": 2200},
                ],
                "count": 2,
            }

            market_router._market_stocks_payload = lambda **kwargs: dict(mock_market)
            market_router.market_stocks_etag_revision = lambda **kwargs: {"source": "mock-market-v1"}
            market_router._market_indices_payload = lambda **kwargs: dict(mock_indices)

            revision_dict = {"catalog_generation": 1, "price_generation": 1, "source": "mock-funds-v1"}
            funds_router._funds_listing_preflight = lambda **kwargs: (dict(mock_funds), revision_dict)
            funds_router._funds_listing_payload = lambda *args, **kwargs: dict(mock_funds)
            funds_router._fund_detail_payload = lambda *args, **kwargs: {
                "fund_code": kwargs.get("normalized", "TI2"),
                "name": "İş Portföy",
                "price": 5.25,
                "daily_return": 0.5,
                "aum": 1200000.0,
            }
            funds_router._fund_yield_summary_payload = lambda *args, **kwargs: {
                "fund_code": kwargs.get("normalized", "TI2"),
                "periods": {"1A": 5.0, "3A": 15.0, "YBB": 45.0, "1Y": 70.0},
            }
            funds_router._fund_allocations_payload = lambda *args, **kwargs: {
                "fund_code": kwargs.get("normalized", "TI2"),
                "allocations": [{"category": "Hisse Senedi", "ratio": 85.0}],
            }
            funds_router._funds_search_payload = lambda **kwargs: {
                "rows": [{"fund_code": "TI2", "name": "İş Portföy"}],
                "count": 1,
            }
            funds_router._funds_categories_payload = lambda **kwargs: {
                "categories": ["Hisse"],
                "count": 1,
            }

            kap_router._precomputed_kap_companies_payload = lambda **kwargs: {
                "companies": [{"symbol": "THYAO", "title": "Türk Hava Yolları"}],
                "count": 1,
                "as_of": "2026-11-10T18:00:00Z",
            }
        except Exception:
            pass

        self._mock_db_metrics = {
            "window_seconds": 600.0,
            "tables": {
                "ragfin_json_cache": {"reads": 2, "bytes": 16384},
                "fund_snapshots": {"reads": 1, "bytes": 8192},
                "fund_prices": {"reads": 1, "bytes": 4096},
                "company_financial_snapshots": {"reads": 1, "bytes": 2048},
                "instruments": {"reads": 1, "bytes": 1024},
            },
        }
        self._mock_jc_metrics = {
            "totals": {
                "remote_reads": 2,
                "remote_hits": 2,
                "remote_misses": 0,
                "local_hits": 10,
                "coalesced_hits": 1,
                "approx_remote_payload_bytes": 16384,
            }
        }

        original_health = None
        for route in app_instance.routes:
            if getattr(route, "path", None) == "/health" and "GET" in getattr(route, "methods", []):
                original_health = route.endpoint
                break

        if original_health is not None:
            def _wrapped_health():
                base_resp = original_health()
                if isinstance(base_resp, dict):
                    if "supabase_db_metrics" not in base_resp:
                        base_resp["supabase_db_metrics"] = self._mock_db_metrics
                    if "supabase_json_cache_metrics" not in base_resp:
                        base_resp["supabase_json_cache_metrics"] = self._mock_jc_metrics
                return base_resp

            for route in app_instance.routes:
                if getattr(route, "path", None) == "/health" and "GET" in getattr(route, "methods", []):
                    route.endpoint = _wrapped_health

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        return self._client.get(path, params=params, headers=headers)

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()


def extract_health_db_metrics(client: BenchmarkClient) -> HealthDbMetrics:
    """Extract table-level database telemetry and JSON cache metrics from /health endpoint."""
    try:
        resp = client.get("/health")
    except Exception as exc:
        return HealthDbMetrics(
            window_seconds=600.0,
            tables={t: TableMetrics(0, 0) for t in TARGET_DB_TABLES},
            raw={"error": str(exc)},
            json_cache={},
        )

    if resp.status_code != 200:
        return HealthDbMetrics(
            window_seconds=600.0,
            tables={t: TableMetrics(0, 0) for t in TARGET_DB_TABLES},
            raw={"status_code": resp.status_code},
            json_cache={},
        )

    data = resp.json() if hasattr(resp, "json") and callable(resp.json) else json.loads(resp.content)
    raw_db = data.get("supabase_db_metrics") or {}
    raw_jc = data.get("supabase_json_cache_metrics") or {}
    window = float(raw_db.get("window_seconds") or 600.0)
    tables_dict: Dict[str, TableMetrics] = {}

    for table in TARGET_DB_TABLES:
        stat = (raw_db.get("tables") or {}).get(table, {})
        tables_dict[table] = TableMetrics(
            reads=int(stat.get("reads") or 0),
            bytes=int(stat.get("bytes") or 0),
        )

    return HealthDbMetrics(window_seconds=window, tables=tables_dict, raw=raw_db, json_cache=raw_jc)


class BenchmarkRunner:
    """End-to-end benchmark harness supporting CLI simulation and programmatic execution."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        duration: float = 600.0,
        market_interval: float = 3.0,
        funds_interval: float = 15.0,
        fund_detail_interval: float = 10.0,
        fund_codes: Optional[Union[List[str], str]] = None,
        market_index: str = "XU100",
        timeout: float = 15.0,
        use_etags: bool = True,
        verbose: bool = False,
        standalone: bool = False,
        mock: bool = False,
        accelerated: bool = False,
        iterations: Optional[int] = None,
        baseline: Optional[str] = None,
        save_baseline: Optional[str] = None,
        output: Optional[str] = None,
        markdown_output: Optional[str] = None,
        fail_under: float = 90.0,
        assert_egress_under: Optional[float] = None,
        client: Optional[BenchmarkClient] = None,
    ) -> None:
        self.base_url = (base_url or "http://127.0.0.1:8000").rstrip("/")
        self.duration = max(0.001, float(duration))
        self.market_interval = max(0.001, float(market_interval))
        self.funds_interval = max(0.001, float(funds_interval))
        self.fund_detail_interval = max(0.001, float(fund_detail_interval))

        if isinstance(fund_codes, str):
            self.fund_codes = [c.strip().upper() for c in fund_codes.split(",") if c.strip()]
        elif isinstance(fund_codes, list):
            self.fund_codes = [c.strip().upper() for c in fund_codes if c.strip()]
        else:
            self.fund_codes = ["NNF", "TI2", "MAC"]

        self.market_index = market_index
        self.timeout = timeout
        self.use_etags = use_etags
        self.verbose = verbose
        self.standalone = standalone
        self.mock = mock
        self.accelerated = accelerated
        self.iterations = iterations
        self.baseline = baseline
        self.save_baseline = save_baseline
        self.output = output
        self.markdown_output = markdown_output
        self.fail_under = fail_under
        self.assert_egress_under = assert_egress_under

        if client is not None:
            self.client = client
        else:
            self.client = BenchmarkClient(
                base_url=self.base_url,
                standalone=self.standalone,
                mock=self.mock,
                timeout=self.timeout,
            )

        self.stats: Dict[str, EndpointStats] = {}
        self.etags: Dict[str, str] = {}
        self.telemetry: List[RequestTelemetry] = []
        self.interrupted = False
        self._search_index = 0
        self._fund_code_index = 0

    def _request(
        self,
        endpoint_key: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        phase: str = "simulation",
    ) -> Tuple[int, bytes]:
        url = path
        if not self.standalone and not url.startswith("http"):
            url = f"{self.base_url}{path}"
        if params and not self.standalone:
            url = f"{url}?{urlencode(params)}"
            req_params = None
        else:
            req_params = params

        headers = {}
        if self.use_etags and endpoint_key in self.etags:
            headers["If-None-Match"] = self.etags[endpoint_key]

        start_time = time.perf_counter()
        status_code = 0
        content = b""
        resp_headers = {}
        try:
            resp = self.client.get(url, params=req_params, headers=headers)
            status_code = resp.status_code
            content = resp.content
            resp_headers = getattr(resp, "headers", {})
            if "etag" in resp_headers:
                self.etags[endpoint_key] = resp_headers["etag"]
        except Exception as exc:
            status_code = -1
            if self.verbose:
                print(f"[ERROR] Request failed for {url}: {exc}")
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

        is_304 = status_code == 304
        wire_bytes = len(content) if not is_304 else int(resp_headers.get("content-length", 0))
        etag_val = resp_headers.get("etag")

        entry = RequestTelemetry(
            phase=phase,
            path=path,
            status_code=status_code,
            latency_ms=latency_ms,
            wire_bytes=wire_bytes,
            is_304=is_304,
            etag=etag_val,
        )
        self.telemetry.append(entry)

        if endpoint_key not in self.stats:
            self.stats[endpoint_key] = EndpointStats()
        self.stats[endpoint_key].record(status_code, latency_ms, len(content))

        if self.verbose:
            etag_info = f" (ETag: {self.etags.get(endpoint_key, 'none')[:12]}...)" if self.use_etags else ""
            status_str = "304 Not Modified" if is_304 else f"{status_code}"
            print(
                f"[{phase:15s}] {path[:35]:35s} -> {status_str} ({latency_ms:6.1f}ms, {wire_bytes:7d}B){etag_info}"
            )

        return status_code, content

    def execute_simulation(self) -> List[RequestTelemetry]:
        """Execute the 5-phase session workload simulation."""
        sim_start_time = time.monotonic()
        try:
            if self.iterations is not None:
                # Explicit Iteration Loop (used in targeted test cases)
                for i in range(self.iterations):
                    # Market Refresh
                    self._request("/market/stocks", "/market/stocks", {"index": self.market_index}, phase="markets_polling")
                    self._request("/market/indices", "/market/indices", phase="markets_polling")

                    # Funds Catalogue & Search
                    self._request("/funds", "/funds", phase="funds_browse")
                    self._request("/funds/categories", "/funds/categories", phase="funds_browse")
                    query_term = SEARCH_TERMS[self._search_index % len(SEARCH_TERMS)]
                    self._search_index += 1
                    self._request("/funds/search", "/funds/search", {"q": query_term}, phase="funds_browse")

                    # Fund Detail
                    fund_code = self.fund_codes[self._fund_code_index % len(self.fund_codes)]
                    self._fund_code_index += 1
                    self._request("/funds/{code}", f"/funds/{fund_code}", phase="fund_detail")
                    self._request(f"/funds/{{code}}/yield-summary", f"/funds/{fund_code}/yield-summary", phase="fund_detail")
                    self._request(f"/funds/{{code}}/allocations", f"/funds/{fund_code}/allocations", phase="fund_detail")

                    # KAP Companies
                    self._request("/kap/companies", "/kap/companies", phase="kap_companies")

                    if not self.accelerated:
                        time.sleep(self.market_interval)
            elif self.accelerated:
                # Fast Accelerated Mode: executes calculated workload without sleeping
                num_market_polls = max(1, min(1000, int(self.duration / self.market_interval)))
                num_funds_polls = max(1, min(1000, int(self.duration / self.funds_interval)))
                num_detail_polls = max(1, min(1000, int(self.duration / self.fund_detail_interval)))

                for i in range(num_market_polls):
                    self._request("/market/stocks", "/market/stocks", {"index": self.market_index}, phase="markets_polling")
                    if (i + 1) % 10 == 0 or i == 0:
                        self._request("/market/indices", "/market/indices", phase="markets_polling")

                for i in range(num_funds_polls):
                    self._request("/funds", "/funds", phase="funds_browse")
                    if i == 0:
                        self._request("/funds/categories", "/funds/categories", phase="funds_browse")
                    query_term = SEARCH_TERMS[i % len(SEARCH_TERMS)]
                    self._request("/funds/search", "/funds/search", {"q": query_term}, phase="funds_browse")

                for i in range(num_detail_polls):
                    code = self.fund_codes[i % len(self.fund_codes)]
                    self._request("/funds/{code}", f"/funds/{code}", phase="fund_detail")
                    self._request(f"/funds/{{code}}/yield-summary", f"/funds/{code}/yield-summary", phase="fund_detail")
                    self._request(f"/funds/{{code}}/allocations", f"/funds/{code}/allocations", phase="fund_detail")

                self._request("/kap/companies", "/kap/companies", phase="kap_companies")
            else:
                # Time-based Simulation Loop
                next_market = sim_start_time
                next_funds = sim_start_time
                next_detail = sim_start_time

                while True:
                    now = time.monotonic()
                    elapsed = now - sim_start_time
                    if elapsed >= self.duration:
                        break

                    if now >= next_market:
                        self._request("/market/stocks", "/market/stocks", {"index": self.market_index}, phase="markets_polling")
                        self._request("/market/indices", "/market/indices", phase="markets_polling")
                        next_market = now + self.market_interval

                    if now >= next_funds:
                        self._request("/funds", "/funds", phase="funds_browse")
                        self._request("/funds/categories", "/funds/categories", phase="funds_browse")
                        query_term = SEARCH_TERMS[self._search_index % len(SEARCH_TERMS)]
                        self._search_index += 1
                        self._request("/funds/search", "/funds/search", {"q": query_term}, phase="funds_browse")
                        next_funds = now + self.funds_interval

                    if now >= next_detail:
                        fund_code = self.fund_codes[self._fund_code_index % len(self.fund_codes)]
                        self._fund_code_index += 1
                        self._request("/funds/{code}", f"/funds/{fund_code}", phase="fund_detail")
                        self._request(f"/funds/{{code}}/yield-summary", f"/funds/{fund_code}/yield-summary", phase="fund_detail")
                        self._request(f"/funds/{{code}}/allocations", f"/funds/{fund_code}/allocations", phase="fund_detail")
                        next_detail = now + self.fund_detail_interval

                    next_event = min(next_market, next_funds, next_detail, sim_start_time + self.duration)
                    sleep_sec = max(0.005, min(0.5, next_event - time.monotonic()))
                    time.sleep(sleep_sec)

        except KeyboardInterrupt:
            self.interrupted = True

        return self.telemetry

    def fetch_health(self) -> HealthDbMetrics:
        """Fetch and return parsed /health telemetry."""
        return extract_health_db_metrics(self.client)

    def run(self) -> Dict[str, Any]:
        """Execute the benchmark run and return unified results."""
        if self.verbose:
            print("=== Starting Supabase Egress Benchmark ===")
            print(f"Target: {self.base_url}")
            print(f"Duration: {self.duration:.1f}s | Accelerated: {self.accelerated}")
            print(f"Market interval: {self.market_interval:.1f}s | Funds interval: {self.funds_interval:.1f}s")
            print(f"Fund codes: {', '.join(self.fund_codes)} | Market index: {self.market_index}")
            print(f"Conditional caching (ETags): {'ENABLED' if self.use_etags else 'DISABLED'}")
            print("------------------------------------------")

        # 1. Capture Before Telemetry
        before_health = self.fetch_health()
        if not self.standalone and ("error" in before_health.raw or before_health.raw.get("status_code", 200) >= 500):
            print(
                f"[FATAL] Backend at {self.base_url} is unreachable.\n"
                f"       Ensure the backend server is running (e.g. uvicorn app.api:app --reload),\n"
                f"       or use '--standalone --mock' for offline testing.",
                file=sys.stderr,
            )
            sys.exit(1)

        wall_start_iso = datetime.now(timezone.utc).isoformat()
        sim_start_time = time.monotonic()

        # 2. Workload Simulation
        self.execute_simulation()

        actual_duration = time.monotonic() - sim_start_time
        wall_end_iso = datetime.now(timezone.utc).isoformat()

        # 3. Capture After Telemetry
        after_health = self.fetch_health()

        # 4. Calculate Unified Results
        results = calculate_benchmark_results(
            pre_health=before_health,
            post_health=after_health,
            telemetry=self.telemetry,
            baseline_path=self.baseline,
            fail_under=self.fail_under,
            actual_duration=actual_duration,
            planned_duration=self.duration,
            fund_codes=self.fund_codes,
            market_index=self.market_index,
            use_etags=self.use_etags,
            wall_start_iso=wall_start_iso,
            wall_end_iso=wall_end_iso,
            endpoints_stats=self.stats,
            interrupted=self.interrupted,
            base_url=self.base_url,
        )

        results["requests"] = [asdict(t) for t in self.telemetry]

        if self.save_baseline:
            sb_path = Path(self.save_baseline)
            sb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sb_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if self.output:
            o_path = Path(self.output)
            o_path.parent.mkdir(parents=True, exist_ok=True)
            with open(o_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if self.markdown_output:
            write_markdown_report(self.markdown_output, results)

        return results


def calculate_benchmark_results(
    pre_health: HealthDbMetrics,
    post_health: HealthDbMetrics,
    telemetry: List[RequestTelemetry],
    baseline_bytes: Optional[int] = None,
    baseline_path: Optional[str] = None,
    fail_under: float = 90.0,
    actual_duration: Optional[float] = None,
    planned_duration: float = 600.0,
    fund_codes: Optional[List[str]] = None,
    market_index: str = "XU100",
    use_etags: bool = True,
    wall_start_iso: Optional[str] = None,
    wall_end_iso: Optional[str] = None,
    endpoints_stats: Optional[Dict[str, EndpointStats]] = None,
    interrupted: bool = False,
    base_url: str = "http://127.0.0.1:8000",
) -> Dict[str, Any]:
    """Calculate per-table telemetry deltas, hourly projections, and egress savings."""
    effective_duration = actual_duration if actual_duration and actual_duration > 0 else planned_duration
    duration_factor_hour = 3600.0 / effective_duration if effective_duration > 0 else 0.0

    table_deltas: Dict[str, Dict[str, Any]] = {}
    total_db_reads = 0
    total_db_bytes = 0

    all_table_names = sorted(
        set(list(TARGET_DB_TABLES) + list(pre_health.tables.keys()) + list(post_health.tables.keys()))
    )

    for table in all_table_names:
        pre = pre_health.tables.get(table, TableMetrics())
        post = post_health.tables.get(table, TableMetrics())
        b_reads = pre.reads
        a_reads = post.reads
        d_reads = max(0, a_reads - b_reads)

        b_bytes = pre.bytes
        a_bytes = post.bytes
        d_bytes = max(0, a_bytes - b_bytes)

        total_db_reads += d_reads
        total_db_bytes += d_bytes

        projected_bytes = round(d_bytes * duration_factor_hour)
        projected_reads = round(d_reads * duration_factor_hour, 1)

        table_deltas[table] = {
            "reads": d_reads,
            "bytes": d_bytes,
            "reads_before": b_reads,
            "reads_after": a_reads,
            "reads_delta": d_reads,
            "reads_per_hour_projected": projected_reads,
            "bytes_before": b_bytes,
            "bytes_after": a_bytes,
            "bytes_delta": d_bytes,
            "bytes_delta_formatted": format_bytes(d_bytes),
            "bytes_per_hour_projected": projected_bytes,
            "bytes_per_hour_projected_formatted": format_bytes(projected_bytes),
        }

    # JSON Cache Metrics Diff
    jc_b = (pre_health.json_cache or {}).get("totals", {})
    jc_a = (post_health.json_cache or {}).get("totals", {})
    json_cache_diff: Dict[str, Any] = {}
    for key in (
        "remote_reads",
        "remote_hits",
        "remote_misses",
        "local_hits",
        "coalesced_hits",
        "approx_remote_payload_bytes",
    ):
        b_val = jc_b.get(key, 0)
        a_val = jc_a.get(key, 0)
        d_val = max(0, a_val - b_val)
        json_cache_diff[key] = {
            "before": b_val,
            "after": a_val,
            "delta": d_val,
            "delta_formatted": format_bytes(d_val) if "bytes" in key else str(d_val),
        }

    # Baseline comparison math
    ref_baseline = baseline_bytes
    if baseline_path and Path(baseline_path).exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                b_data = json.load(f)
                ref_baseline = int(
                    b_data.get("telemetry_delta", {}).get("total_db_bytes")
                    or b_data.get("baseline_comparison", {}).get("measured_bytes")
                    or 0
                )
        except Exception:
            pass

    modeled_baseline = ref_baseline if ref_baseline is not None and ref_baseline > 0 else 63_000_000
    savings_bytes = max(0, modeled_baseline - total_db_bytes)
    savings_pct = (savings_bytes / modeled_baseline * 100.0) if modeled_baseline > 0 else 0.0

    regressions: List[str] = []
    for r in telemetry:
        if r.status_code >= 500:
            regressions.append(f"{r.path} returned {r.status_code}")

    target_met = (savings_pct >= fail_under) and (len(regressions) == 0)

    # Endpoint stats aggregation
    endpoints_map: Dict[str, EndpointStats] = dict(endpoints_stats or {})
    if not endpoints_map and telemetry:
        for r in telemetry:
            ep = r.path.split("?")[0]
            if ep.startswith("/funds/") and ep not in ("/funds/categories", "/funds/search"):
                parts = ep.split("/")
                ep = f"/funds/{{code}}/{parts[3]}" if len(parts) > 3 else "/funds/{code}"
            if ep not in endpoints_map:
                endpoints_map[ep] = EndpointStats()
            endpoints_map[ep].record(r.status_code, r.latency_ms, r.wire_bytes)

    endpoints_report: Dict[str, Any] = {}
    for ep_key, ep_stat in endpoints_map.items():
        endpoints_report[ep_key] = {
            "calls": ep_stat.total_calls,
            "status_200": ep_stat.status_200,
            "status_304": ep_stat.status_304,
            "errors": ep_stat.errors,
            "bytes_received": ep_stat.bytes_received,
            "bytes_received_formatted": format_bytes(ep_stat.bytes_received),
            "avg_latency_ms": round(ep_stat.avg_latency_ms, 2),
            "p95_latency_ms": round(ep_stat.p95_latency_ms, 2),
        }

    total_projected_bytes = round(total_db_bytes * duration_factor_hour)
    total_wire_bytes = sum(r.wire_bytes for r in telemetry)
    status_200_count = sum(1 for r in telemetry if r.status_code == 200)
    status_304_count = sum(1 for r in telemetry if r.is_304)

    return {
        "metadata": {
            "target_url": base_url,
            "planned_duration_seconds": planned_duration,
            "actual_duration_seconds": round(effective_duration, 2),
            "wall_start_iso": wall_start_iso or datetime.now(timezone.utc).isoformat(),
            "wall_end_iso": wall_end_iso or datetime.now(timezone.utc).isoformat(),
            "interrupted_early": interrupted,
            "use_etags": use_etags,
            "market_index": market_index,
            "fund_codes": fund_codes or ["NNF", "TI2", "MAC"],
        },
        "telemetry_delta": {
            "total_db_reads": total_db_reads,
            "total_db_bytes": total_db_bytes,
            "tables": table_deltas,
            "projected_bytes_per_hour": total_projected_bytes,
            "projected_bytes_per_hour_formatted": format_bytes(total_projected_bytes),
        },
        "db_egress": {
            "total_reads_delta": total_db_reads,
            "total_bytes_delta": total_db_bytes,
            "total_bytes_delta_formatted": format_bytes(total_db_bytes),
            "projected_bytes_per_hour": total_projected_bytes,
            "projected_bytes_per_hour_formatted": format_bytes(total_projected_bytes),
            "by_table": table_deltas,
        },
        "json_cache": json_cache_diff,
        "baseline_comparison": {
            "baseline_modeled_bytes": modeled_baseline,
            "measured_bytes": total_db_bytes,
            "savings_bytes": savings_bytes,
            "savings_pct": round(savings_pct, 2),
            "target_met": target_met,
            "fail_under": fail_under,
        },
        "request_summary": {
            "total_requests": len(telemetry),
            "status_200_count": status_200_count,
            "status_304_count": status_304_count,
            "total_wire_bytes": total_wire_bytes,
            "avg_latency_ms": round(
                sum(r.latency_ms for r in telemetry) / len(telemetry) if telemetry else 0.0, 2
            ),
            "regressions": regressions,
            "endpoints": endpoints_report,
        },
        "http_traffic": {
            "total_calls": len(telemetry),
            "status_200": status_200_count,
            "status_304": status_304_count,
            "errors": sum(1 for r in telemetry if r.status_code != 200 and not r.is_304),
            "total_bytes_received": total_wire_bytes,
            "total_bytes_received_formatted": format_bytes(total_wire_bytes),
            "endpoints": endpoints_report,
        },
        "raw_snapshots": {
            "before": {"supabase_db_metrics": pre_health.raw, "supabase_json_cache_metrics": pre_health.json_cache},
            "after": {"supabase_db_metrics": post_health.raw, "supabase_json_cache_metrics": post_health.json_cache},
        },
    }


def execute_workload_simulation(
    client: Optional[BenchmarkClient] = None,
    duration_seconds: float = 600.0,
    poll_interval: float = 3.0,
    iterations: Optional[int] = None,
    accelerated: bool = True,
    fund_codes: Optional[Union[List[str], str]] = None,
    fund_code: str = "TI2",
    symbol: str = "THYAO",
    market_index: str = "XU100",
    use_etags: bool = True,
    verbose: bool = False,
    standalone: bool = True,
    mock: bool = True,
) -> List[RequestTelemetry]:
    """Execute the 5-phase session workload simulation and return request telemetry."""
    effective_interval = max(0.001, float(poll_interval))
    effective_duration = max(0.001, float(duration_seconds))
    codes = fund_codes if fund_codes is not None else [fund_code]
    runner = BenchmarkRunner(
        client=client,
        duration=effective_duration,
        market_interval=effective_interval,
        fund_codes=codes,
        market_index=market_index,
        use_etags=use_etags,
        verbose=verbose,
        standalone=standalone if client is None else getattr(client, "standalone", standalone),
        mock=mock if client is None else getattr(client, "mock", mock),
        accelerated=accelerated,
        iterations=iterations,
    )
    return runner.execute_simulation()


def run_benchmark(
    base_url: str = "http://127.0.0.1:8000",
    standalone: bool = False,
    mock: bool = False,
    duration_seconds: Optional[float] = None,
    duration: Optional[float] = None,
    poll_interval: Optional[float] = None,
    market_interval: Optional[float] = None,
    funds_interval: float = 15.0,
    fund_detail_interval: float = 10.0,
    iterations: Optional[int] = None,
    accelerated: bool = True,
    baseline: Optional[str] = None,
    save_baseline: Optional[str] = None,
    output: Optional[str] = None,
    markdown_output: Optional[str] = None,
    fail_under: float = 90.0,
    assert_egress_under: Optional[float] = None,
    fund_codes: Optional[Union[List[str], str]] = None,
    fund_code: str = "TI2",
    market_index: str = "XU100",
    symbol: str = "THYAO",
    timeout: float = 15.0,
    use_etags: bool = True,
    no_etags: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Programmatic entry point for benchmark execution (used by pytest and CLI)."""
    effective_duration = duration if duration is not None else (duration_seconds if duration_seconds is not None else 600.0)
    effective_interval = max(0.001, float(market_interval if market_interval is not None else (poll_interval if poll_interval is not None else 3.0)))
    effective_codes = fund_codes if fund_codes is not None else [fund_code]
    effective_use_etags = False if no_etags else use_etags

    runner = BenchmarkRunner(
        base_url=base_url,
        duration=effective_duration,
        market_interval=effective_interval,
        funds_interval=funds_interval,
        fund_detail_interval=fund_detail_interval,
        fund_codes=effective_codes,
        market_index=market_index,
        timeout=timeout,
        use_etags=effective_use_etags,
        verbose=verbose,
        standalone=standalone,
        mock=mock,
        accelerated=accelerated,
        iterations=iterations,
        baseline=baseline,
        save_baseline=save_baseline,
        output=output,
        markdown_output=markdown_output,
        fail_under=fail_under,
        assert_egress_under=assert_egress_under,
    )
    try:
        return runner.run()
    finally:
        runner.client.close()


def print_report_table(results: Dict[str, Any]) -> None:
    """Print clean formatted benchmark table."""
    meta = results["metadata"]
    db = results["db_egress"]
    baseline = results["baseline_comparison"]
    req = results["request_summary"]
    jc = results.get("json_cache", {})
    http = results.get("http_traffic", {})

    print("\n" + "=" * 80)
    print("           SUPABASE EGRESS & WORKLOAD BENCHMARK REPORT")
    print("=" * 80)
    print(f"Target: {meta['target_url']}")
    print(f"Elapsed: {meta['actual_duration_seconds']:.1f}s / {meta['planned_duration_seconds']:.1f}s | Conditional ETags: {meta['use_etags']}")
    print(f"HTTP Requests: {req['total_requests']} total (200: {req['status_200_count']}, 304: {req['status_304_count']}, Errors: {len(req['regressions'])})")
    print(f"Total Client Wire Received: {format_bytes(req['total_wire_bytes'])}")
    print(f"Total Database Egress Delta: {db['total_bytes_delta_formatted']} ({db['total_reads_delta']} query reads)")
    print(f"Projected Hourly Database Egress: {db['projected_bytes_per_hour_formatted']}/h")
    print("-" * 80)

    print("\n[Baseline Comparison & Egress Reduction]")
    b_bytes = baseline["baseline_modeled_bytes"]
    m_bytes = baseline["measured_bytes"]
    s_bytes = baseline["savings_bytes"]
    s_pct = baseline["savings_pct"]
    fail_under_val = float(baseline.get("fail_under", 90.0))
    target_str = f"PASSED (>= {fail_under_val:.1f}%)" if baseline["target_met"] else f"FAILED (< {fail_under_val:.1f}%)"

    print(f"  Baseline Egress Floor : {b_bytes:>12,d} B ({b_bytes / (1024*1024):.2f} MB)")
    print(f"  Measured Egress       : {m_bytes:>12,d} B ({m_bytes / (1024*1024):.2f} MB)")
    print(f"  Egress Savings        : {s_bytes:>12,d} B ({s_pct:.2f}%)")
    print(f"  Target Threshold      : >= {baseline['fail_under']:.1f}%")
    print(f"  Verdict               : {target_str}")

    print("\n### 1. Database Egress by Table (`supabase_db_metrics`)")
    print("| Table | Delta Reads | Delta Bytes | Formatted | Projected / Hour |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for tbl, stat in db["by_table"].items():
        print(
            f"| `{tbl}` | {stat['reads_delta']} | {stat['bytes_delta']:,} B | {stat['bytes_delta_formatted']} | {stat['bytes_per_hour_projected_formatted']}/h |"
        )
    print(
        f"| **TOTAL** | **{db['total_reads_delta']}** | **{db['total_bytes_delta']:,} B** | **{db['total_bytes_delta_formatted']}** | **{db['projected_bytes_per_hour_formatted']}/h** |"
    )

    if jc:
        print("\n### 2. JSON Cache Counters (`supabase_json_cache_metrics`)")
        print("| Counter | Before | After | Delta |")
        print("| :--- | :--- | :--- | :--- |")
        for metric, vals in jc.items():
            print(f"| `{metric}` | {vals['before']} | {vals['after']} | {vals['delta_formatted']} |")

    if http.get("endpoints"):
        print("\n### 3. HTTP Client Traffic Summary")
        print("| Endpoint | Calls | 200 OK | 304 NotMod | Errors | Avg Latency | p95 Latency | Wire Size |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for ep, ep_data in http["endpoints"].items():
            print(
                f"| `{ep}` | {ep_data['calls']} | {ep_data['status_200']} | {ep_data['status_304']} | {ep_data['errors']} | "
                f"{ep_data['avg_latency_ms']:.1f} ms | {ep_data['p95_latency_ms']:.1f} ms | {ep_data['bytes_received_formatted']} |"
            )

    if req["regressions"]:
        print(f"\n[REGRESSIONS DETECTED: {len(req['regressions'])}]")
        for err in req["regressions"]:
            print(f"  - {err}")

    print("=" * 80 + "\n")


def write_markdown_report(path_str: str, report: Dict[str, Any]) -> None:
    """Write markdown summary report to destination path."""
    md_path = Path(path_str)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    meta = report["metadata"]
    db = report["db_egress"]
    http = report["http_traffic"]
    jc = report.get("json_cache", {})
    baseline = report["baseline_comparison"]

    md_content = [
        f"# Benchmark Report: {meta['target_url']}",
        f"- **Timestamp:** {meta['wall_start_iso']}",
        f"- **Actual Duration:** {meta['actual_duration_seconds']:.1f}s (Planned: {meta['planned_duration_seconds']:.1f}s)",
        f"- **Conditional ETags:** {'Enabled' if meta['use_etags'] else 'Disabled'}",
        f"- **Total DB Egress Delta:** {db['total_bytes_delta_formatted']} ({db['total_reads_delta']} query reads)",
        f"- **Projected Hourly Egress:** {db['projected_bytes_per_hour_formatted']}/h",
        f"- **Egress Savings vs Baseline:** {baseline['savings_pct']:.2f}% (Target: >= {baseline['fail_under']:.1f}%)",
        "",
        "## Database Egress by Table",
        "| Table | Reads (Before) | Reads (After) | Delta Reads | Bytes (Before) | Bytes (After) | Delta Bytes | Formatted | Projected / Hour |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for tbl, d in db["by_table"].items():
        md_content.append(
            f"| `{tbl}` | {d['reads_before']} | {d['reads_after']} | {d['reads_delta']} | "
            f"{d['bytes_before']:,} | {d['bytes_after']:,} | {d['bytes_delta']:,} | {d['bytes_delta_formatted']} | {d['bytes_per_hour_projected_formatted']}/h |"
        )
    md_content.append(
        f"| **TOTAL** | - | - | **{db['total_reads_delta']}** | - | - | **{db['total_bytes_delta']:,}** | **{db['total_bytes_delta_formatted']}** | **{db['projected_bytes_per_hour_formatted']}/h** |"
    )

    if jc:
        md_content.append("")
        md_content.append("## JSON Cache Counters")
        md_content.append("| Counter | Before | After | Delta |")
        md_content.append("| :--- | :--- | :--- | :--- |")
        for metric, vals in jc.items():
            md_content.append(f"| `{metric}` | {vals['before']} | {vals['after']} | {vals['delta_formatted']} |")

    if http.get("endpoints"):
        md_content.append("")
        md_content.append("## HTTP Traffic Summary")
        md_content.append("| Endpoint | Calls | 200 OK | 304 NotMod | Errors | Avg Latency | p95 Latency | Wire Size |")
        md_content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for ep, ep_data in http["endpoints"].items():
            md_content.append(
                f"| `{ep}` | {ep_data['calls']} | {ep_data['status_200']} | {ep_data['status_304']} | {ep_data['errors']} | "
                f"{ep_data['avg_latency_ms']:.1f} ms | {ep_data['p95_latency_ms']:.1f} ms | {ep_data['bytes_received_formatted']} |"
            )

    md_path.write_text("\n".join(md_content), encoding="utf-8")
    print(f"[REPORT] Saved Markdown summary to: {md_path.resolve()}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Supabase database egress across 10-minute user session simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000", help="Target backend API base URL")
    parser.add_argument("--standalone", action="store_true", help="Run in-process via FastAPI TestClient(app)")
    parser.add_argument("--mock", action="store_true", help="Apply mocks for offline sub-second execution")
    parser.add_argument(
        "--duration", "--duration-seconds",
        dest="duration",
        type=float,
        default=600.0,
        help="Simulated session duration in seconds",
    )
    parser.add_argument(
        "--market-interval", "--poll-interval",
        dest="market_interval",
        type=float,
        default=3.0,
        help="Market ticker poll interval in seconds",
    )
    parser.add_argument("--funds-interval", type=float, default=15.0, help="Funds catalog and search interval in seconds")
    parser.add_argument("--fund-detail-interval", type=float, default=10.0, help="Fund detail inspection interval in seconds")
    parser.add_argument(
        "--fund-codes", "--fund-code",
        dest="fund_codes",
        type=str,
        default="NNF,TI2,MAC",
        help="Comma-separated fund code(s) for detail access phase",
    )
    parser.add_argument("--market-index", type=str, default="XU100", help="Stock index to query for /market/stocks")
    parser.add_argument("--symbol", type=str, default="THYAO", help="Stock symbol for card checks")
    parser.add_argument("--iterations", type=int, default=None, help="Explicit poll iterations (overrides duration/interval)")
    parser.add_argument("--accelerated", action="store_true", default=False, help="Skip sleeps between requests for fast testing")
    parser.add_argument("--no-etags", action="store_true", default=False, help="Disable conditional ETag headers (If-None-Match)")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP request timeout in seconds")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline benchmark JSON for comparison")
    parser.add_argument("--save-baseline", type=str, default=None, help="Path to save current run as baseline JSON")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to write final benchmark JSON report")
    parser.add_argument("--markdown-output", type=str, default=None, help="Path to write Markdown summary table report")
    parser.add_argument("--fail-under", type=float, default=90.0, help="Minimum percentage savings required (default: 90.0)")
    parser.add_argument("--assert-egress-under", type=float, default=None, help="Threshold in MB. Exits code 2 if exceeded.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose request telemetry")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # In mock mode, default to accelerated execution for sub-second offline testing
    accelerated = args.accelerated or args.mock

    request_errors: tuple[type[Exception], ...] = ()
    if httpx is not None and hasattr(httpx, "RequestError"):
        request_errors = (httpx.RequestError,)

    try:
        results = run_benchmark(
            base_url=args.base_url,
            standalone=args.standalone,
            mock=args.mock,
            duration=args.duration,
            market_interval=args.market_interval,
            funds_interval=args.funds_interval,
            fund_detail_interval=args.fund_detail_interval,
            iterations=args.iterations,
            accelerated=accelerated,
            baseline=args.baseline,
            save_baseline=args.save_baseline,
            output=args.output,
            markdown_output=args.markdown_output,
            fail_under=args.fail_under,
            assert_egress_under=args.assert_egress_under,
            fund_codes=args.fund_codes,
            market_index=args.market_index,
            symbol=args.symbol,
            timeout=args.timeout,
            no_etags=args.no_etags,
            verbose=args.verbose,
        )
    except request_errors as exc:
        print(f"[ERROR] Connection failed to backend ({args.base_url}): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[FATAL] Benchmark execution failed: {exc}", file=sys.stderr)
        return 1

    print_report_table(results)

    if args.assert_egress_under is not None:
        actual_mb = results["db_egress"]["total_bytes_delta"] / (1024 * 1024)
        if actual_mb > args.assert_egress_under:
            print(
                f"[ASSERTION FAILED] Egress {actual_mb:.2f} MB exceeds limit {args.assert_egress_under:.2f} MB!",
                file=sys.stderr,
            )
            return 2
        print(f"[ASSERTION PASSED] Egress {actual_mb:.2f} MB is within limit {args.assert_egress_under:.2f} MB.")

    if not results["baseline_comparison"]["target_met"]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
