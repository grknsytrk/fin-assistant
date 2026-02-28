from __future__ import annotations

import json
import sys
from pathlib import Path


def _coverage_after(row: dict) -> float:
    for key in ("coverage_rate_after", "extraction_accuracy_after", "coverage_rate"):
        if key in row and row.get(key) is not None:
            return float(row.get(key))
    return 0.0


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    summary_path = Path(args[0]) if args else Path("data/demo/processed/eval_metrics_summary.json")
    if not summary_path.exists():
        print(f"[smoke] summary bulunamadi: {summary_path}")
        return 1

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    by_company = (
        payload.get("multi_company_extraction", {}).get("by_company", {})
        if isinstance(payload, dict)
        else {}
    )

    expected = ["BIM", "MIGROS", "SOK"]
    failed = False
    for company in expected:
        if company not in by_company:
            print(f"[smoke] eksik sirket: {company}")
            failed = True
            continue
        coverage = _coverage_after(by_company[company])
        print(f"[smoke] {company} coverage_after={coverage:.4f}")
        if coverage <= 0.0:
            failed = True

    if failed:
        print("[smoke] metrics_report demo smoke BASARISIZ")
        return 1
    print("[smoke] metrics_report demo smoke BASARILI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
