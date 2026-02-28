from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from src.metrics import _find_first_match_rank, load_gold_questions
from src.retrieve import RetrieverV3, RetrieverV5Hybrid, RetrieverV6Cross


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = max(0, min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _format_table(rows: List[Dict[str, object]]) -> str:
    headers = [("retriever", 12), ("avg_ms", 12), ("p95_ms", 12), ("hit@1", 10)]
    sep = " | "

    def _render(row: List[str]) -> str:
        return sep.join(value.ljust(width) for value, (_, width) in zip(row, headers))

    lines = [_render([name for name, _ in headers])]
    lines.append("-" * (sum(width for _, width in headers) + len(sep) * (len(headers) - 1)))
    for row in rows:
        lines.append(
            _render(
                [
                    str(row["retriever"]),
                    f"{float(row['avg_ms']):.2f}",
                    f"{float(row['p95_ms']):.2f}",
                    f"{float(row['hit@1']):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def run_latency_benchmark(
    gold_file: Path,
    output_file: Path,
    sample_size: int = 20,
    top_k: int = 5,
    top_k_initial_v3: int = 20,
    top_k_initial_v5_vector: int = 20,
    top_k_initial_v5_bm25: int = 20,
    top_k_candidates_v6: int = 15,
    alpha_v3: float = 0.35,
    beta_v5: float = 0.6,
) -> Dict[str, object]:
    questions = load_gold_questions(gold_file)[:sample_size]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    retriever_v3 = RetrieverV3()
    retriever_v5 = RetrieverV5Hybrid()
    retriever_v6 = RetrieverV6Cross()

    latency_ms: Dict[str, List[float]] = {"v3": [], "v5": [], "v6": []}
    hit1_count: Dict[str, float] = {"v3": 0.0, "v5": 0.0, "v6": 0.0}

    for row in questions:
        question = str(row.get("question", ""))
        expected = row.get("expected", [])
        expected_list = expected if isinstance(expected, list) else []

        started = time.perf_counter()
        v3_results = retriever_v3.retrieve_with_query_awareness(
            query=question,
            top_k_initial=top_k_initial_v3,
            top_k_final=top_k,
            alpha=alpha_v3,
        )
        latency_ms["v3"].append((time.perf_counter() - started) * 1000.0)
        v3_rank = _find_first_match_rank(v3_results, expected_list, max_rank=top_k)
        hit1_count["v3"] += 1.0 if v3_rank == 1 else 0.0

        started = time.perf_counter()
        v5_results = retriever_v5.retrieve_with_hybrid(
            query=question,
            top_k_vector=top_k_initial_v5_vector,
            top_k_bm25=top_k_initial_v5_bm25,
            top_k_final=top_k,
            beta=beta_v5,
            alpha_v3=alpha_v3,
        )
        latency_ms["v5"].append((time.perf_counter() - started) * 1000.0)
        v5_rank = _find_first_match_rank(v5_results, expected_list, max_rank=top_k)
        hit1_count["v5"] += 1.0 if v5_rank == 1 else 0.0

        started = time.perf_counter()
        v6_results = retriever_v6.retrieve_with_cross_encoder(
            query=question,
            top_k_candidates=top_k_candidates_v6,
            top_k_final=top_k,
            top_k_vector=top_k_initial_v5_vector,
            top_k_bm25=top_k_initial_v5_bm25,
            beta=beta_v5,
            alpha_v3=alpha_v3,
        )
        latency_ms["v6"].append((time.perf_counter() - started) * 1000.0)
        v6_rank = _find_first_match_rank(v6_results, expected_list, max_rank=top_k)
        hit1_count["v6"] += 1.0 if v6_rank == 1 else 0.0

    rows: List[Dict[str, object]] = []
    for retriever_name in ("v3", "v5", "v6"):
        values = latency_ms[retriever_name]
        sorted_values = sorted(values)
        total = len(values) if values else 1
        rows.append(
            {
                "retriever": retriever_name,
                "avg_ms": round(statistics.fmean(values), 2) if values else 0.0,
                "p95_ms": round(_percentile(sorted_values, 95), 2),
                "hit@1": round(hit1_count[retriever_name] / total, 4),
                "n": total,
            }
        )

    payload = {
        "gold_file": str(gold_file),
        "output_file": str(output_file),
        "sample_size": len(questions),
        "rows": rows,
        "table": _format_table(rows),
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload

