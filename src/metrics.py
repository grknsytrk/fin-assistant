from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.retrieve import (
    RetrievedChunk,
    Retriever,
    RetrieverBM25,
    RetrieverV2,
    RetrieverV3,
    RetrieverV5Hybrid,
    RetrieverV6Cross,
    normalize_for_match,
)
from src.metrics_extractor import extract_metric_with_candidates
from src.query_parser import parse_query

DEFAULT_GOLD_FILE = Path("eval/gold_questions.jsonl")
DEFAULT_DETAILED_OUTPUT = Path("data/processed/eval_metrics_detailed.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/eval_metrics_summary.json")
DEFAULT_WEEK6_SUMMARY_OUTPUT = Path("data/processed/eval_metrics_week6.json")
DEFAULT_MULTI_COMPANY_GOLD_FILE = Path("eval/gold_questions_multicompany.jsonl")
RETRIEVER_ORDER = ["v1", "v2", "v3", "v4_bm25", "v5_hybrid", "v6_cross"]


def load_gold_questions(gold_file: Path) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    with gold_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            questions.append(json.loads(line))
    return questions


def _normalize_quarter_label(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip().upper()
    if not candidate:
        return None
    if candidate.endswith("Q1") or candidate == "Q1":
        return "Q1"
    if candidate.endswith("Q2") or candidate == "Q2":
        return "Q2"
    if candidate.endswith("Q3") or candidate == "Q3":
        return "Q3"
    if candidate.endswith("Q4") or candidate == "Q4":
        return "Q4"
    return None


def _serialize_result_row(item: RetrievedChunk) -> Dict[str, object]:
    return {
        "doc_id": item.doc_id,
        "company": item.company,
        "quarter": item.quarter,
        "year": item.year,
        "page": item.page,
        "section_title": item.section_title,
        "block_type": item.block_type,
        "distance": item.distance,
        "raw_score": item.score,
        "vector_score": item.vector_score,
        "lexical_boost": item.lexical_boost,
        "final_score": item.final_score,
    }


def _entry_matches_expected(result: RetrievedChunk, expected_entry: Dict[str, object]) -> bool:
    expected_quarter = _normalize_quarter_label(expected_entry.get("quarter"))
    result_quarter = _normalize_quarter_label(result.quarter)
    if expected_quarter and expected_quarter != result_quarter:
        return False

    expected_doc_contains = expected_entry.get("doc_contains")
    if expected_doc_contains:
        if normalize_for_match(str(expected_doc_contains)) not in normalize_for_match(result.doc_id):
            return False

    expected_page = expected_entry.get("page")
    if expected_page is not None and int(expected_page) != result.page:
        return False

    expected_section_contains = expected_entry.get("section_contains")
    if expected_section_contains:
        if normalize_for_match(str(expected_section_contains)) not in normalize_for_match(result.section_title):
            return False

    return True


def _find_first_match_rank(
    results: Sequence[RetrievedChunk],
    expected: Sequence[Dict[str, object]],
    max_rank: int = 5,
) -> Optional[int]:
    for idx, result in enumerate(results[:max_rank], start=1):
        if any(_entry_matches_expected(result, expected_entry) for expected_entry in expected):
            return idx
    return None


def _quarter_accuracy_at_1(results: Sequence[RetrievedChunk], expected: Sequence[Dict[str, object]]) -> float:
    if not results:
        return 0.0

    top_quarter = _normalize_quarter_label(results[0].quarter)
    expected_quarters = {
        normalized
        for normalized in (_normalize_quarter_label(entry.get("quarter")) for entry in expected)
        if normalized
    }
    if not expected_quarters or not top_quarter:
        return 0.0
    return 1.0 if top_quarter in expected_quarters else 0.0


def _metrics_from_rank(rank: Optional[int]) -> Dict[str, float]:
    hit1 = 1.0 if rank == 1 else 0.0
    hit3 = 1.0 if rank is not None and rank <= 3 else 0.0
    hit5 = 1.0 if rank is not None and rank <= 5 else 0.0
    mrr5 = 1.0 / rank if rank is not None and rank <= 5 else 0.0
    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,
        "MRR@5": mrr5,
    }


def _empty_aggregate() -> Dict[str, float]:
    return {
        "hit@1": 0.0,
        "hit@3": 0.0,
        "hit@5": 0.0,
        "MRR@5": 0.0,
        "quarter_accuracy@1": 0.0,
    }


def _add_metrics(target: Dict[str, float], source: Dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def _finalize_average(metrics_sum: Dict[str, float], total: int) -> Dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in metrics_sum}
    return {key: round(value / total, 4) for key, value in metrics_sum.items()}


def format_metrics_table(summary: Dict[str, object]) -> str:
    retrievers = summary.get("retrievers", {})
    headers = [
        ("retriever", 10),
        ("hit@1", 10),
        ("hit@3", 10),
        ("hit@5", 10),
        ("MRR@5", 10),
        ("quarter@1", 12),
    ]

    def _format_row(values: List[str]) -> str:
        padded = [value.ljust(width) for value, (_, width) in zip(values, headers)]
        return " | ".join(padded)

    lines = []
    lines.append(_format_row([name for name, _ in headers]))
    lines.append("-" * (sum(width for _, width in headers) + (3 * (len(headers) - 1))))

    for retriever_name in RETRIEVER_ORDER:
        if retriever_name not in retrievers:
            continue
        metrics = retrievers.get(retriever_name, {})
        lines.append(
            _format_row(
                [
                    retriever_name,
                    f"{metrics.get('hit@1', 0.0):.4f}",
                    f"{metrics.get('hit@3', 0.0):.4f}",
                    f"{metrics.get('hit@5', 0.0):.4f}",
                    f"{metrics.get('MRR@5', 0.0):.4f}",
                    f"{metrics.get('quarter_accuracy@1', 0.0):.4f}",
                ]
            )
        )
    return "\n".join(lines)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _run_multicompany_extraction_eval(
    gold_file: Path,
    top_k_initial_v3: int,
    top_k: int,
    alpha_v3: float,
) -> Dict[str, object]:
    if not gold_file.exists():
        return {
            "dataset": str(gold_file),
            "num_questions": 0,
            "by_company": {},
            "by_metric": {},
        }

    rows = load_gold_questions(gold_file)
    retriever_v3 = RetrieverV3()
    by_company_metric: Dict[str, Dict[str, Dict[str, int]]] = {}
    detailed_rows: List[Dict[str, object]] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        company = str(row.get("company", "")).strip().upper()
        metric = str(row.get("metric", "")).strip()
        quarter = str(row.get("quarter", "")).strip().upper()
        if not question or not company or not metric or not quarter:
            continue

        chunks = retriever_v3.retrieve_with_query_awareness(
            query=question,
            top_k_initial=top_k_initial_v3,
            top_k_final=top_k,
            alpha=alpha_v3,
            quarter_override=quarter,
            company_override=company,
        )
        baseline_extracted = extract_metric_with_candidates(
            chunks=chunks,
            metric=metric,
            quarter=quarter,
            top_n=5,
            use_structured_table_reconstruction=False,
            use_expected_range_sanity=False,
        )
        extracted = extract_metric_with_candidates(
            chunks=chunks,
            metric=metric,
            quarter=quarter,
            top_n=5,
        )
        baseline_selected = baseline_extracted.get("selected")
        selected = extracted.get("selected")
        candidates = list(extracted.get("candidates", []))
        found_before = baseline_selected is not None
        found_after = selected is not None
        found = found_after
        invalid = bool(candidates) and not found and all(not bool(c.get("validation_ok")) for c in candidates)

        company_bucket = by_company_metric.setdefault(company, {})
        metric_bucket = company_bucket.setdefault(
            metric,
            {
                "total": 0,
                "found_before": 0,
                "found": 0,
                "invalid": 0,
                "verified_pass": 0,
                "verified_warn": 0,
                "verified_fail": 0,
            },
        )
        metric_bucket["total"] += 1
        if found_before:
            metric_bucket["found_before"] += 1
        if found:
            metric_bucket["found"] += 1
            verify_status = str((selected or {}).get("verify_status", "FAIL")).upper()
            if verify_status == "PASS":
                metric_bucket["verified_pass"] += 1
            elif verify_status == "WARN":
                metric_bucket["verified_warn"] += 1
            else:
                metric_bucket["verified_fail"] += 1
        if invalid:
            metric_bucket["invalid"] += 1

        parsed = parse_query(question)
        detailed_rows.append(
            {
                "id": row.get("id"),
                "company": company,
                "metric": metric,
                "quarter": quarter,
                "question": question,
                "query_type": parsed.get("signals", {}).get("query_type"),
                "found_before": found_before,
                "found_after": found_after,
                "found": found_after,
                "invalid": invalid,
                "verify_status": (selected or {}).get("verify_status"),
                "selected_confidence": (selected or {}).get("confidence"),
                "selected_source": {
                    "doc_id": (selected or {}).get("doc_id"),
                    "page": (selected or {}).get("page"),
                    "section_title": (selected or {}).get("section_title"),
                },
            }
        )

    by_company_summary: Dict[str, Dict[str, object]] = {}
    by_metric_totals: Dict[str, Dict[str, int]] = {}
    for company, metric_map in by_company_metric.items():
        company_total = 0
        company_found_before = 0
        company_found = 0
        company_invalid = 0
        company_verified_pass = 0
        metrics_payload: Dict[str, Dict[str, float]] = {}
        for metric, stats in metric_map.items():
            total = int(stats["total"])
            found_before = int(stats["found_before"])
            found = int(stats["found"])
            invalid = int(stats["invalid"])
            verified_pass = int(stats["verified_pass"])
            verified_warn = int(stats["verified_warn"])
            verified_fail = int(stats["verified_fail"])
            company_total += total
            company_found_before += found_before
            company_found += found
            company_invalid += invalid
            company_verified_pass += verified_pass
            metrics_payload[metric] = {
                "coverage_rate_before": _safe_rate(found_before, total),
                "coverage_rate": _safe_rate(found, total),
                "coverage_rate_after": _safe_rate(found, total),
                "coverage_delta": round(_safe_rate(found, total) - _safe_rate(found_before, total), 4),
                "invalid_rate": _safe_rate(invalid, total),
                "verified_pass_rate": _safe_rate(verified_pass, found),
                "verified_warn_rate": _safe_rate(verified_warn, found),
                "verified_fail_rate": _safe_rate(verified_fail, found),
                "count": total,
            }
            metric_totals = by_metric_totals.setdefault(
                metric,
                {
                    "total": 0,
                    "found_before": 0,
                    "found": 0,
                    "invalid": 0,
                    "verified_pass": 0,
                    "verified_warn": 0,
                    "verified_fail": 0,
                },
            )
            metric_totals["total"] += total
            metric_totals["found_before"] += found_before
            metric_totals["found"] += found
            metric_totals["invalid"] += invalid
            metric_totals["verified_pass"] += verified_pass
            metric_totals["verified_warn"] += verified_warn
            metric_totals["verified_fail"] += verified_fail

        by_company_summary[company] = {
            "extraction_accuracy_before": _safe_rate(company_found_before, company_total),
            "extraction_accuracy_after": _safe_rate(company_found, company_total),
            "extraction_accuracy_delta": round(
                _safe_rate(company_found, company_total) - _safe_rate(company_found_before, company_total),
                4,
            ),
            "coverage_rate_before": _safe_rate(company_found_before, company_total),
            "coverage_rate": _safe_rate(company_found, company_total),
            "coverage_rate_after": _safe_rate(company_found, company_total),
            "invalid_rate": _safe_rate(company_invalid, company_total),
            "verified_pass_rate": _safe_rate(company_verified_pass, company_found),
            "count": company_total,
            "metrics": metrics_payload,
        }

    by_metric_summary: Dict[str, Dict[str, float]] = {}
    for metric, stats in by_metric_totals.items():
        by_metric_summary[metric] = {
            "coverage_rate_before": _safe_rate(int(stats["found_before"]), int(stats["total"])),
            "coverage_rate": _safe_rate(int(stats["found"]), int(stats["total"])),
            "coverage_rate_after": _safe_rate(int(stats["found"]), int(stats["total"])),
            "coverage_delta": round(
                _safe_rate(int(stats["found"]), int(stats["total"]))
                - _safe_rate(int(stats["found_before"]), int(stats["total"])),
                4,
            ),
            "invalid_rate": _safe_rate(int(stats["invalid"]), int(stats["total"])),
            "verified_pass_rate": _safe_rate(int(stats["verified_pass"]), int(stats["found"])),
            "verified_warn_rate": _safe_rate(int(stats["verified_warn"]), int(stats["found"])),
            "verified_fail_rate": _safe_rate(int(stats["verified_fail"]), int(stats["found"])),
            "count": int(stats["total"]),
        }

    return {
        "dataset": str(gold_file),
        "num_questions": len(detailed_rows),
        "by_company": by_company_summary,
        "by_metric": by_metric_summary,
        "details": detailed_rows,
    }


def run_metrics_report(
    gold_file: Path = DEFAULT_GOLD_FILE,
    multi_company_gold_file: Path = DEFAULT_MULTI_COMPANY_GOLD_FILE,
    detailed_output: Path = DEFAULT_DETAILED_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    week6_summary_output: Path = DEFAULT_WEEK6_SUMMARY_OUTPUT,
    top_k: int = 5,
    top_k_initial_v2: int = 15,
    top_k_initial_v3: int = 20,
    top_k_initial_v5_vector: int = 20,
    top_k_initial_v5_bm25: int = 20,
    top_k_candidates_v6: int = 15,
    alpha_v2: float = 0.35,
    alpha_v3: float = 0.35,
    beta_v5: float = 0.6,
) -> Dict[str, object]:
    questions = load_gold_questions(gold_file)
    detailed_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    week6_summary_output.parent.mkdir(parents=True, exist_ok=True)

    retriever_v1 = Retriever()
    retriever_v2 = RetrieverV2()
    retriever_v3 = RetrieverV3()
    retriever_v4_bm25 = RetrieverBM25()
    retriever_v5 = RetrieverV5Hybrid()
    retriever_v6 = RetrieverV6Cross()

    aggregates = {name: _empty_aggregate() for name in RETRIEVER_ORDER}

    with detailed_output.open("w", encoding="utf-8") as out_file:
        for question_row in questions:
            question_id = str(question_row.get("id", ""))
            question = str(question_row.get("question", ""))
            question_type = str(question_row.get("type", "unknown"))
            expected = question_row.get("expected", [])
            expected_list = expected if isinstance(expected, list) else []

            v1_results = retriever_v1.retrieve(question, top_k=top_k)
            v2_results = retriever_v2.retrieve_with_boost(
                query=question,
                top_k_initial=top_k_initial_v2,
                top_k_final=top_k,
                alpha=alpha_v2,
            )
            v3_results = retriever_v3.retrieve_with_query_awareness(
                query=question,
                top_k_initial=top_k_initial_v3,
                top_k_final=top_k,
                alpha=alpha_v3,
            )
            v4_results = retriever_v4_bm25.retrieve(
                query=question,
                top_k=top_k,
            )
            v5_results = retriever_v5.retrieve_with_hybrid(
                query=question,
                top_k_vector=top_k_initial_v5_vector,
                top_k_bm25=top_k_initial_v5_bm25,
                top_k_final=top_k,
                beta=beta_v5,
                alpha_v3=alpha_v3,
            )
            v6_results = retriever_v6.retrieve_with_cross_encoder(
                query=question,
                top_k_candidates=top_k_candidates_v6,
                top_k_final=top_k,
                top_k_vector=top_k_initial_v5_vector,
                top_k_bm25=top_k_initial_v5_bm25,
                beta=beta_v5,
                alpha_v3=alpha_v3,
            )

            per_retriever_rows: Dict[str, Dict[str, object]] = {}
            for retriever_name, results in (
                ("v1", v1_results),
                ("v2", v2_results),
                ("v3", v3_results),
                ("v4_bm25", v4_results),
                ("v5_hybrid", v5_results),
                ("v6_cross", v6_results),
            ):
                rank = _find_first_match_rank(results=results, expected=expected_list, max_rank=top_k)
                metrics = _metrics_from_rank(rank)
                quarter_acc = _quarter_accuracy_at_1(results=results, expected=expected_list)
                metrics["quarter_accuracy@1"] = quarter_acc
                _add_metrics(aggregates[retriever_name], metrics)

                per_retriever_rows[retriever_name] = {
                    "first_match_rank": rank,
                    "metrics": metrics,
                    "top_results": [_serialize_result_row(row) for row in results[:top_k]],
                }

            detail_row = {
                "id": question_id,
                "question": question,
                "type": question_type,
                "expected": expected_list,
                "retrievers": per_retriever_rows,
            }
            out_file.write(json.dumps(detail_row, ensure_ascii=False) + "\n")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(gold_file),
        "multi_company_dataset": str(multi_company_gold_file),
        "num_questions": len(questions),
        "top_k": top_k,
        "top_k_initial_v2": top_k_initial_v2,
        "top_k_initial_v3": top_k_initial_v3,
        "top_k_initial_v5_vector": top_k_initial_v5_vector,
        "top_k_initial_v5_bm25": top_k_initial_v5_bm25,
        "top_k_candidates_v6": top_k_candidates_v6,
        "alpha_v2": alpha_v2,
        "alpha_v3": alpha_v3,
        "beta_v5": beta_v5,
        "retrievers": {
            name: _finalize_average(aggregates[name], len(questions)) for name in RETRIEVER_ORDER
        },
        "multi_company_extraction": _run_multicompany_extraction_eval(
            gold_file=multi_company_gold_file,
            top_k_initial_v3=top_k_initial_v3,
            top_k=top_k,
            alpha_v3=alpha_v3,
        ),
        "detailed_output": str(detailed_output),
    }

    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with week6_summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "summary": summary,
        "summary_output": str(summary_output),
        "week6_summary_output": str(week6_summary_output),
        "detailed_output": str(detailed_output),
        "table": format_metrics_table(summary),
    }
