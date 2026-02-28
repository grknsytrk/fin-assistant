from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.metrics import load_gold_questions
from src.metrics_extractor import METRIC_DEFINITIONS, extract_metric_with_candidates, normalize_for_match
from src.retrieve import RetrievedChunk, RetrieverV3

DEFAULT_MULTI_COMPANY_GOLD = Path("eval/gold_questions_multicompany.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed")

NUMBER_PATTERN = re.compile(r"\(?[\-]?\d[\d\.,]*\)?")
LINE_PHRASE_PATTERN = re.compile(r"\(?[\-]?\d[\d\.,]*\)?")
STOP_TOKENS = {
    "ve",
    "ile",
    "icin",
    "olarak",
    "bir",
    "bu",
    "de",
    "da",
    "q1",
    "q2",
    "q3",
    "2024",
    "2025",
}


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _matches_synonym(text: str, metric: str) -> bool:
    metric_row = METRIC_DEFINITIONS.get(metric, {})
    synonyms = metric_row.get("synonyms", ())
    normalized = normalize_for_match(text)
    return any(str(synonym) in normalized for synonym in synonyms)


def _missing_reason(
    metric: str,
    chunks: Sequence[RetrievedChunk],
    candidates: Sequence[Dict[str, Any]],
) -> str:
    if candidates and all(not bool(item.get("validation_ok")) for item in candidates):
        return "unit_parse_or_scaling_fail"
    if candidates:
        return "candidate_conflict_or_ranking_miss"

    all_text = " ".join(str(chunk.text) for chunk in chunks[:8])
    has_numbers = bool(NUMBER_PATTERN.search(all_text))
    has_metric_label = _matches_synonym(all_text, metric)

    if not has_numbers:
        return "table_split_or_no_numeric_evidence"
    if has_numbers and not has_metric_label:
        return "label_mismatch"
    return "pattern_miss_or_table_split"


def _collect_line_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not NUMBER_PATTERN.search(line):
            continue
        norm_line = normalize_for_match(line)
        for match in LINE_PHRASE_PATTERN.finditer(norm_line):
            prefix = norm_line[: match.start()].strip(" -:;|")
            tokens = [tok for tok in re.findall(r"[a-z0-9]+", prefix) if tok not in STOP_TOKENS]
            if not tokens:
                continue
            phrase = " ".join(tokens[-5:])
            if len(phrase) < 4:
                continue
            if any(ch.isdigit() for ch in phrase):
                continue
            phrases.append(phrase)
    return phrases


def _iter_company_rows(
    rows: Iterable[Dict[str, Any]],
    company: Optional[str],
) -> Iterable[Dict[str, Any]]:
    company_filter = str(company or "").strip().upper()
    for row in rows:
        if not company_filter:
            yield row
            continue
        row_company = str(row.get("company", "")).strip().upper()
        if row_company == company_filter:
            yield row


def run_coverage_audit(
    company: Optional[str] = None,
    gold_file: Path = DEFAULT_MULTI_COMPANY_GOLD,
    output_file: Optional[Path] = None,
    top_k_initial_v3: int = 20,
    top_k: int = 5,
    alpha_v3: float = 0.35,
) -> Dict[str, Any]:
    if not gold_file.exists():
        raise FileNotFoundError(f"Gold dosyasi bulunamadi: {gold_file}")

    rows = list(_iter_company_rows(load_gold_questions(gold_file), company=company))
    retriever = RetrieverV3()

    by_metric = defaultdict(lambda: {"total": 0, "found": 0, "invalid": 0, "verified_pass": 0, "reasons": Counter()})
    missing_details: List[Dict[str, Any]] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        metric = str(row.get("metric", "")).strip()
        quarter = str(row.get("quarter", "")).strip().upper()
        company_row = str(row.get("company", "")).strip().upper()
        if not (question and metric and quarter and company_row):
            continue

        chunks = retriever.retrieve_with_query_awareness(
            query=question,
            top_k_initial=top_k_initial_v3,
            top_k_final=top_k,
            alpha=alpha_v3,
            quarter_override=quarter,
            company_override=company_row,
        )
        extraction = extract_metric_with_candidates(
            chunks=chunks,
            metric=metric,
            quarter=quarter,
            top_n=5,
        )
        selected = extraction.get("selected")
        candidates = list(extraction.get("candidates", []))

        bucket = by_metric[metric]
        bucket["total"] += 1
        if selected:
            bucket["found"] += 1
            if str(selected.get("verify_status")) == "PASS":
                bucket["verified_pass"] += 1
            continue

        invalid = bool(candidates) and all(not bool(item.get("validation_ok")) for item in candidates)
        if invalid:
            bucket["invalid"] += 1

        reason = _missing_reason(metric=metric, chunks=chunks, candidates=candidates)
        bucket["reasons"][reason] += 1
        missing_details.append(
            {
                "id": row.get("id"),
                "company": company_row,
                "metric": metric,
                "quarter": quarter,
                "question": question,
                "reason": reason,
                "top_sections": [str(chunk.section_title) for chunk in chunks[:5]],
                "top_doc_pages": [
                    {
                        "doc_id": str(chunk.doc_id),
                        "quarter": str(chunk.quarter),
                        "page": int(chunk.page),
                    }
                    for chunk in chunks[:5]
                ],
                "candidate_count": len(candidates),
                "candidate_reasons": [str(item.get("validation_reason", "")) for item in candidates[:3]],
            }
        )

    coverage_by_metric: Dict[str, Dict[str, Any]] = {}
    for metric, stats in sorted(by_metric.items()):
        total = int(stats["total"])
        found = int(stats["found"])
        invalid = int(stats["invalid"])
        verified_pass = int(stats["verified_pass"])
        reason_counter: Counter = stats["reasons"]
        coverage_by_metric[metric] = {
            "coverage_rate": _safe_rate(found, total),
            "invalid_rate": _safe_rate(invalid, total),
            "verified_pass_rate": _safe_rate(verified_pass, found),
            "count": total,
            "missing": total - found,
            "likely_reasons": dict(reason_counter.most_common(5)),
        }

    missing_rank = sorted(
        (
            {
                "metric": metric,
                "missing_count": payload["missing"],
                "coverage_rate": payload["coverage_rate"],
                "likely_reasons": payload["likely_reasons"],
            }
            for metric, payload in coverage_by_metric.items()
        ),
        key=lambda item: int(item["missing_count"]),
        reverse=True,
    )

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(gold_file),
        "company": str(company).upper() if company else "ALL",
        "num_questions": sum(int(item["count"]) for item in coverage_by_metric.values()),
        "coverage_by_metric": coverage_by_metric,
        "top_missing_metrics": missing_rank[:8],
        "missing_examples": missing_details[:40],
    }

    if output_file is None:
        suffix = str(company).upper() if company else "ALL"
        output_file = DEFAULT_OUTPUT_DIR / f"coverage_audit_{suffix}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    payload["output_file"] = str(output_file)
    return payload


def suggest_dictionary_phrases(
    company: str,
    gold_file: Path = DEFAULT_MULTI_COMPANY_GOLD,
    top_n: int = 30,
    top_k_initial_v3: int = 20,
    top_k: int = 5,
    alpha_v3: float = 0.35,
) -> Dict[str, Any]:
    company_upper = str(company).strip().upper()
    if not company_upper:
        raise ValueError("company zorunlu")
    rows = list(_iter_company_rows(load_gold_questions(gold_file), company=company_upper))
    retriever = RetrieverV3()
    phrases = Counter()

    for row in rows:
        question = str(row.get("question", "")).strip()
        metric = str(row.get("metric", "")).strip()
        quarter = str(row.get("quarter", "")).strip().upper()
        if not (question and metric and quarter):
            continue

        chunks = retriever.retrieve_with_query_awareness(
            query=question,
            top_k_initial=top_k_initial_v3,
            top_k_final=top_k,
            alpha=alpha_v3,
            quarter_override=quarter,
            company_override=company_upper,
        )
        extraction = extract_metric_with_candidates(
            chunks=chunks,
            metric=metric,
            quarter=quarter,
            top_n=5,
        )
        if extraction.get("selected") is not None:
            continue

        for chunk in chunks[:8]:
            for phrase in _collect_line_phrases(str(chunk.text)):
                phrases[phrase] += 1

    suggestions = [{"phrase": phrase, "count": count} for phrase, count in phrases.most_common(top_n)]
    return {
        "company": company_upper,
        "dataset": str(gold_file),
        "top_n": top_n,
        "suggestions": suggestions,
    }
