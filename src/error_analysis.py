from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.query_parser import parse_query

DEFAULT_DETAILED_FILE = Path("data/processed/eval_metrics_detailed.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/processed/error_analysis.jsonl")


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _is_failed_retrieval(retriever_row: Dict[str, object]) -> bool:
    rank = retriever_row.get("first_match_rank")
    return rank is None


def run_error_report(
    detailed_file: Path = DEFAULT_DETAILED_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    retrievers: Optional[List[str]] = None,
) -> Dict[str, object]:
    if not detailed_file.exists():
        raise FileNotFoundError(f"Detayli metrics dosyasi bulunamadi: {detailed_file}")

    rows = _load_jsonl(detailed_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_errors = 0
    per_retriever: Dict[str, int] = {}

    with output_file.open("w", encoding="utf-8") as out_file:
        for row in rows:
            question = str(row.get("question", ""))
            parsed = parse_query(question)
            query_type = str(parsed.get("signals", {}).get("query_type", "other"))
            quarter_detected = parsed.get("quarter")
            expected = row.get("expected", [])
            retriever_rows = row.get("retrievers", {})
            if not isinstance(retriever_rows, dict):
                continue

            for retriever_name, retriever_row in retriever_rows.items():
                if retrievers and retriever_name not in retrievers:
                    continue
                if not isinstance(retriever_row, dict):
                    continue
                if not _is_failed_retrieval(retriever_row):
                    continue

                top_results = retriever_row.get("top_results", [])
                output_row = {
                    "id": row.get("id"),
                    "question": question,
                    "retriever_used": retriever_name,
                    "query_type": query_type,
                    "quarter_detected": quarter_detected,
                    "expected": expected,
                    "retrieved": top_results,
                }
                out_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                total_errors += 1
                per_retriever[retriever_name] = per_retriever.get(retriever_name, 0) + 1

    return {
        "detailed_file": str(detailed_file),
        "output_file": str(output_file),
        "total_errors": total_errors,
        "errors_per_retriever": per_retriever,
        "questions_count": len(rows),
    }

