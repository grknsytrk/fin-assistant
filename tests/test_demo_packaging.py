from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.index import build_index_v2_from_pages
from src.metrics import run_metrics_report
from src.retrieve import RetrievedChunk


class _FakeCollection:
    def __init__(self) -> None:
        self._rows: Dict[str, Dict[str, Any]] = {}

    def count(self) -> int:
        return len(self._rows)

    def get(self, limit: Optional[int] = None, include: Optional[List[str]] = None) -> Dict[str, List[str]]:
        ids = list(self._rows.keys())
        if limit is not None:
            ids = ids[: int(limit)]
        return {"ids": ids}

    def delete(self, ids: List[str]) -> None:
        for item in ids:
            self._rows.pop(item, None)

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        for idx, item in enumerate(ids):
            self._rows[item] = {
                "id": item,
                "document": documents[idx],
                "metadata": metadatas[idx],
                "embedding": embeddings[idx],
            }


class _FakeClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self.collections: Dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> _FakeCollection:
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]


class _FakeEmbedder:
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_demo_bundle_index_from_pages(monkeypatch, tmp_path: Path) -> None:
    from src import index as index_module

    fake_client = _FakeClient(path=str(tmp_path / "chroma"))
    monkeypatch.setattr(index_module, "E5Embedder", _FakeEmbedder)
    monkeypatch.setattr(index_module.chromadb, "PersistentClient", lambda path: fake_client)

    fixture_file = Path("data/demo_bundle/pages_fixture.jsonl")
    summary = build_index_v2_from_pages(
        pages_file=fixture_file,
        processed_dir=tmp_path / "processed",
        collection_name="demo_test_v2",
        chunk_size=700,
        overlap=100,
    )

    assert summary["num_pdfs"] == 9
    assert int(summary["indexed_chunks"]) > 0
    assert int(summary["collection_count"]) == int(summary["indexed_chunks"])


def _quarter_from_query(text: str) -> str:
    normalized = text.lower()
    if "q1" in normalized or "birinci" in normalized:
        return "2025Q1"
    if "q2" in normalized or "ikinci" in normalized:
        return "2025Q2"
    if "q3" in normalized or "ucuncu" in normalized:
        return "2025Q3"
    return "2025Q1"


def _company_from_query(text: str, fallback: Optional[str] = None) -> str:
    if fallback:
        return str(fallback).upper()
    normalized = text.upper()
    for company in ("BIM", "MIGROS", "SOK"):
        if company in normalized:
            return company
    return "BIM"


def _fake_chunk(query: str, company: Optional[str] = None, quarter: Optional[str] = None) -> RetrievedChunk:
    selected_company = _company_from_query(query, fallback=company)
    selected_quarter = quarter or _quarter_from_query(query)
    doc_id = f"{selected_company}_{selected_quarter}_DEMO"
    page = 5
    page_match = re.search(r"page\s*(\d+)", query.lower())
    if page_match:
        page = int(page_match.group(1))
    return RetrievedChunk(
        text=f"{selected_company} {selected_quarter} demo chunk",
        distance=0.01,
        score=0.99,
        doc_id=doc_id,
        company=selected_company,
        quarter=selected_quarter,
        year=2025,
        page=page,
        chunk_id=f"{doc_id}_c1",
        section_title="FINANSAL",
        block_type="table_like",
        chunk_version="v2",
        vector_score=0.99,
        lexical_boost=0.4,
        final_score=0.99,
    )


class _FakeRetriever:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def retrieve(self, query: str, top_k: int = 5, company: Optional[str] = None) -> List[RetrievedChunk]:
        return [_fake_chunk(query, company=company)]

    def retrieve_with_boost(self, query: str, *args, **kwargs) -> List[RetrievedChunk]:
        return [_fake_chunk(query)]

    def retrieve_with_query_awareness(
        self,
        query: str,
        *args,
        quarter_override: Optional[str] = None,
        company_override: Optional[str] = None,
        **kwargs,
    ) -> List[RetrievedChunk]:
        quarter = f"2025{quarter_override}" if quarter_override and not str(quarter_override).startswith("20") else quarter_override
        return [_fake_chunk(query, company=company_override, quarter=quarter)]

    def retrieve_with_hybrid(self, query: str, *args, **kwargs) -> List[RetrievedChunk]:
        return [_fake_chunk(query)]

    def retrieve_with_cross_encoder(self, query: str, *args, **kwargs) -> List[RetrievedChunk]:
        return [_fake_chunk(query)]


def _fake_extract_metric_with_candidates(chunks: List[RetrievedChunk], metric: str, quarter: str, **kwargs) -> Dict[str, Any]:
    selected = None
    if chunks:
        base = chunks[0]
        selected = {
            "quarter": quarter,
            "metric": metric,
            "value": 1.0,
            "unit": "TL",
            "confidence": 0.95,
            "verify_status": "PASS",
            "doc_id": base.doc_id,
            "page": base.page,
            "section_title": base.section_title,
            "validation_ok": True,
        }
    return {"selected": selected, "candidates": [{"validation_ok": True}]}


def test_metrics_report_demo_smoke(monkeypatch, tmp_path: Path) -> None:
    from src import metrics as metrics_module

    monkeypatch.setattr(metrics_module, "Retriever", _FakeRetriever)
    monkeypatch.setattr(metrics_module, "RetrieverV2", _FakeRetriever)
    monkeypatch.setattr(metrics_module, "RetrieverV3", _FakeRetriever)
    monkeypatch.setattr(metrics_module, "RetrieverBM25", _FakeRetriever)
    monkeypatch.setattr(metrics_module, "RetrieverV5Hybrid", _FakeRetriever)
    monkeypatch.setattr(metrics_module, "RetrieverV6Cross", _FakeRetriever)
    monkeypatch.setattr(metrics_module, "extract_metric_with_candidates", _fake_extract_metric_with_candidates)

    report = run_metrics_report(
        gold_file=Path("data/demo_bundle/gold_questions_demo.jsonl"),
        multi_company_gold_file=Path("data/demo_bundle/gold_questions_demo_multicompany.jsonl"),
        detailed_output=tmp_path / "metrics_detailed.jsonl",
        summary_output=tmp_path / "metrics_summary.json",
        week6_summary_output=tmp_path / "metrics_week6.json",
        top_k=1,
        top_k_initial_v2=1,
        top_k_initial_v3=1,
        top_k_initial_v5_vector=1,
        top_k_initial_v5_bm25=1,
        top_k_candidates_v6=1,
        alpha_v2=0.35,
        alpha_v3=0.35,
        beta_v5=0.6,
    )

    by_company = ((report.get("summary") or {}).get("multi_company_extraction") or {}).get("by_company", {})
    assert set(by_company.keys()) >= {"BIM", "MIGROS", "SOK"}
    for company in ("BIM", "MIGROS", "SOK"):
        row = by_company[company]
        assert float(row.get("coverage_rate_after", 0.0)) > 0.0
        assert int(row.get("count", 0)) > 0
