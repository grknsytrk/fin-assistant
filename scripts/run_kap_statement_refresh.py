"""CLI runner for KAP statement refresh with automatic resumption and error tolerance.

Can run against a remote FIN API deployment or locally in-process.

Usage:
    python scripts/run_kap_statement_refresh.py --mode incremental --universe bist100
    python scripts/run_kap_statement_refresh.py --mode full --universe all --budget-seconds 21600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


def _request_refresh_api(
    base_url: str,
    admin_token: str,
    *,
    mode: str,
    universe: str,
    symbols: Optional[str],
    max_quarters: int,
    budget_seconds: int,
    concurrency: int,
) -> Dict[str, Any]:
    params = {
        "mode": mode,
        "universe": universe,
        "max_quarters": str(max_quarters),
        "budget_seconds": str(budget_seconds),
        "concurrency": str(concurrency),
    }
    if symbols:
        params["symbols"] = symbols
    query = urllib.parse.urlencode(params)
    url = f"{base_url.rstrip('/')}/admin/kap/refresh-statements?{query}"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _poll_status_api(
    base_url: str,
    admin_token: str,
    job_id: str,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/admin/kap/refresh-statements/status?job_id={job_id}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_local(
    *,
    mode: str,
    universe: str,
    symbols: Optional[str],
    max_quarters: int,
    budget_seconds: int,
    concurrency: int,
) -> Dict[str, Any]:
    from app.kap_statement_refresh import (
        _get_kap_statement_refresh_job,
        _public_kap_statement_job,
        _start_kap_statement_refresh_job,
    )

    parsed_symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    job = _start_kap_statement_refresh_job(
        mode=mode,
        universe=universe,
        symbols=parsed_symbols,
        max_quarters=max_quarters,
        budget_seconds=budget_seconds,
        concurrency=concurrency,
    )
    job_id = str(job.get("job_id") or "")
    while True:
        time.sleep(2)
        stored = _get_kap_statement_refresh_job(job_id)
        if not stored:
            raise RuntimeError(f"Job {job_id} not found in storage.")
        public_job = _public_kap_statement_job(stored)
        status = str(public_job.get("status") or "")
        processed = public_job.get("processed", 0)
        total = public_job.get("total", 0)
        current = public_job.get("current_symbol") or "-"
        last = public_job.get("last_symbol") or "-"
        print(
            f"statement refresh: job_id={job_id} status={status} "
            f"processed={processed}/{total} current={current} last={last}"
        )
        if status in {"succeeded", "partial", "aborted", "failed", "superseded"}:
            return {"refresh_job": public_job}


def main() -> int:
    parser = argparse.ArgumentParser(description="KAP statement refresh runner.")
    parser.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    parser.add_argument("--universe", choices=["all", "bist30", "bist100", "symbols"], default="bist100")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols for universe=symbols")
    parser.add_argument("--max-quarters", type=int, default=5)
    parser.add_argument("--budget-seconds", type=int, default=5400)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-allowed-failures", type=int, default=10)
    parser.add_argument("--max-resumes", type=int, default=1)
    parser.add_argument("--api-base-url", type=str, default=os.getenv("FIN_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--admin-token", type=str, default=os.getenv("FIN_API_ADMIN_TOKEN", os.getenv("RAGFIN_ADMIN_REFRESH_TOKEN", "")))
    parser.add_argument("--local", action="store_true", help="Run locally in-process without API server")

    args = parser.parse_args()

    if args.local:
        result = _run_local(
            mode=args.mode,
            universe=args.universe,
            symbols=args.symbols,
            max_quarters=args.max_quarters,
            budget_seconds=args.budget_seconds,
            concurrency=args.concurrency,
        )
        job = result.get("refresh_job", {})
        status = str(job.get("status") or "")
        failed = int(job.get("failed") or 0)
        blocked = int(job.get("blocked") or 0)
        if status == "succeeded" or (status == "partial" and failed <= args.max_allowed_failures and blocked == 0):
            return 0
        return 1

    resumes_left = args.max_resumes
    while True:
        print(f"Requesting KAP statement refresh (mode={args.mode}, universe={args.universe}, budget={args.budget_seconds}s)...")
        resp = _request_refresh_api(
            args.api_base_url,
            args.admin_token,
            mode=args.mode,
            universe=args.universe,
            symbols=args.symbols,
            max_quarters=args.max_quarters,
            budget_seconds=args.budget_seconds,
            concurrency=args.concurrency,
        )
        job_id = str(resp.get("refresh_job", {}).get("job_id") or "")
        if not job_id:
            print("Error: endpoint did not return job_id", file=sys.stderr)
            return 1

        print(f"Statement refresh started: job_id={job_id}")
        poll_deadline = time.time() + args.budget_seconds + 300
        while time.time() < poll_deadline:
            time.sleep(15)
            try:
                status_payload = _poll_status_api(args.api_base_url, args.admin_token, job_id)
            except Exception as exc:
                print(f"Warning: status poll failed: {exc}", file=sys.stderr)
                continue

            job = status_payload.get("refresh_job", {})
            status = str(job.get("status") or "")
            processed = job.get("processed", 0)
            total = job.get("total", 0)
            updated = job.get("updated", 0)
            skipped_ready = job.get("skipped_ready_cache", 0)
            failed = int(job.get("failed") or 0)
            blocked = int(job.get("blocked") or 0)
            remaining = job.get("remaining", 0)
            current = job.get("current_symbol") or "-"
            last = job.get("last_symbol") or "-"
            stop_reason = job.get("stop_reason") or "running"

            print(
                f"statement refresh: job_id={job_id} status={status} processed={processed}/{total} "
                f"current={current} last={last} updated={updated} skipped_ready={skipped_ready} "
                f"failed={failed} blocked={blocked} remaining={remaining} stop={stop_reason}"
            )

            if status == "succeeded":
                if blocked > 0 or failed > args.max_allowed_failures:
                    print(f"Error: succeeded with excessive errors: failed={failed} blocked={blocked}", file=sys.stderr)
                    return 1
                if failed > 0:
                    print(f"Warning: succeeded with {failed} non-fatal company failure(s) (within tolerance of {args.max_allowed_failures})")
                print("Statement refresh succeeded.")
                return 0

            if status == "partial":
                if blocked > 0:
                    print(f"Error: statement refresh aborted due to provider blocking (blocked={blocked})", file=sys.stderr)
                    return 1
                if failed > args.max_allowed_failures:
                    print(f"Error: statement refresh failed with errors exceeding threshold ({failed} > {args.max_allowed_failures})", file=sys.stderr)
                    return 1
                if stop_reason == "worker_interrupted":
                    if resumes_left > 0:
                        resumes_left -= 1
                        print(f"Warning: worker was interrupted; starting resume pass ({resumes_left} resumes left)...")
                        break
                    print("Warning: worker was interrupted and resume passes exhausted. Next run will continue.")
                    return 0
                if stop_reason in {"completed_with_errors", "budget_exhausted"}:
                    print(f"Statement refresh stopped ({stop_reason}): processed={processed}/{total} remaining={remaining}. Progress saved.")
                    return 0

            if status in {"aborted", "failed"}:
                print(f"Error: statement refresh failed: status={status} stop_reason={stop_reason}", file=sys.stderr)
                return 1

            if status == "superseded":
                print("Statement refresh was superseded by another job.")
                return 0

        else:
            print("Error: statement refresh polling deadline reached.", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
