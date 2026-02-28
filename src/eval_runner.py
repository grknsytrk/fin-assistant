from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.retrieve import Retriever, RetrieverV2


def load_questions(questions_file: Path) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    with questions_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            questions.append(json.loads(line))
    return questions


def run_retrieval_eval(
    questions_file: Path = Path("eval/questions.jsonl"),
    output_file: Path = Path("data/processed/eval_retrieval.jsonl"),
    top_k: int = 5,
) -> Dict[str, object]:
    retriever = Retriever()
    questions = load_questions(questions_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for row in questions:
            question = str(row.get("question", ""))
            qid = row.get("id")
            results = retriever.retrieve(question, top_k=top_k)
            retrieved_pages = []
            seen = set()
            for item in results:
                key = (item.doc_id, item.quarter, item.page)
                if key in seen:
                    continue
                seen.add(key)
                retrieved_pages.append(
                    {
                        "doc_id": item.doc_id,
                        "company": item.company,
                        "quarter": item.quarter,
                        "year": item.year,
                        "page": item.page,
                    }
                )
            out = {
                "id": qid,
                "question": question,
                "retrieved_pages": retrieved_pages,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {
        "questions_count": len(questions),
        "output_file": str(output_file),
        "top_k": top_k,
    }


def _extract_unique_pages(results) -> List[Dict[str, object]]:
    pages = []
    seen = set()
    for item in results:
        key = (item.doc_id, item.quarter, item.page)
        if key in seen:
            continue
        seen.add(key)
        pages.append(
            {
                "doc_id": item.doc_id,
                "company": item.company,
                "quarter": item.quarter,
                "year": item.year,
                "page": item.page,
            }
        )
    return pages


def _extract_unique_sections(results) -> List[Dict[str, object]]:
    sections = []
    seen = set()
    for item in results:
        key = (item.doc_id, item.quarter, item.page, item.section_title)
        if key in seen:
            continue
        seen.add(key)
        sections.append(
            {
                "doc_id": item.doc_id,
                "company": item.company,
                "quarter": item.quarter,
                "year": item.year,
                "page": item.page,
                "section_title": item.section_title,
            }
        )
    return sections


def run_retrieval_eval_comparison(
    questions_file: Path = Path("eval/questions.jsonl"),
    output_file: Path = Path("data/processed/eval_retrieval_comparison.jsonl"),
    top_k: int = 5,
    top_k_initial_v2: int = 15,
    alpha: float = 0.35,
) -> Dict[str, object]:
    retriever_v1 = Retriever()
    retriever_v2 = RetrieverV2()
    questions = load_questions(questions_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for row in questions:
            question = str(row.get("question", ""))
            qid = row.get("id")

            results_v1 = retriever_v1.retrieve(question, top_k=top_k)
            results_v2 = retriever_v2.retrieve_with_boost(
                query=question,
                top_k_initial=top_k_initial_v2,
                top_k_final=top_k,
                alpha=alpha,
            )

            out = {
                "id": qid,
                "question": question,
                "v1_top_pages": _extract_unique_pages(results_v1),
                "v2_top_pages": _extract_unique_pages(results_v2),
                "v2_top_section_titles": _extract_unique_sections(results_v2),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {
        "questions_count": len(questions),
        "output_file": str(output_file),
        "top_k": top_k,
        "top_k_initial_v2": top_k_initial_v2,
        "alpha": alpha,
    }
