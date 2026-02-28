from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pandas import isna
from pydantic import BaseModel, Field

from src.answer import AnswerEngine, RulesBasedAnswerAdapter
from src.config import AppConfig, load_config
from src.commentary import SAFE_EMPTY_COMMENTARY, generate_commentary
from src.index import build_index, build_index_v2
from src.ingest import ingest_raw_pdfs, list_pdf_files
from src.metrics_extractor import (
    QUARTER_ORDER,
    aggregate_metric_across_quarters,
    build_metric_query,
    collect_top_sources,
    compute_overall_change,
    infer_metric_from_question,
    is_comparison_query,
    metric_display_name,
)
from src.query_parser import parse_query
from src.ratio_engine import (
    build_ratio_table,
    detect_company_mentions,
    is_cross_company_query,
    run_cross_company_comparison,
)
from src.retrieve import RetrievedChunk, Retriever, RetrieverV2, RetrieverV3, RetrieverV5Hybrid, RetrieverV6Cross

CONFIG = load_config(Path("config.yaml"))
app = FastAPI(title="RAG-Fin API", version="0.10.0")
FEEDBACK_FILE = CONFIG.paths.processed_dir / "feedback.jsonl"


class IndexRequest(BaseModel):
    version: Literal["v1", "v2"] = "v2"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retriever: Literal["v1", "v2", "v3", "v5", "v6"] = "v3"
    mode: Literal["single", "trend"] = "single"
    company: Optional[str] = None


class CommentaryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer_payload: Dict[str, Any]
    company: Optional[str] = None
    year: Optional[str] = None
    quarter: Optional[str] = None
    model: Optional[str] = None


class FeedbackRequest(BaseModel):
    timestamp: Optional[str] = None
    company: Optional[str] = None
    quarter: Optional[str] = None
    metric: str = Field(..., min_length=1)
    extracted_value: Optional[str] = None
    user_value: Optional[str] = None
    evidence_ref: Optional[str] = None
    verdict: Literal["dogru", "yanlis"] = "yanlis"


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _collection_count(collection_name: str) -> Optional[int]:
    try:
        import chromadb
    except Exception:
        return None

    try:
        client = chromadb.PersistentClient(path=str(CONFIG.chroma.dir))
        collection = client.get_collection(name=collection_name)
        return int(collection.count())
    except Exception:
        return 0


def _stats_payload() -> Dict[str, Any]:
    pdf_files = list_pdf_files(CONFIG.paths.raw_dir)
    companies = _available_companies_from_chunks(CONFIG.paths.chunks_v2_file)
    return {
        "pdf_count": len(pdf_files),
        "page_count": _count_jsonl_rows(CONFIG.paths.pages_file),
        "chunk_count_v1": _count_jsonl_rows(CONFIG.paths.chunks_v1_file),
        "chunk_count_v2": _count_jsonl_rows(CONFIG.paths.chunks_v2_file),
        "collection_count_v1": _collection_count(CONFIG.chroma.collection_v1),
        "collection_count_v2": _collection_count(CONFIG.chroma.collection_v2),
        "companies": companies,
    }


def _available_companies_from_chunks(chunks_file: Path) -> List[str]:
    if not chunks_file.exists():
        return []
    companies = set()
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            company = str(payload.get("company", "")).strip()
            if company:
                companies.add(company.upper())
    return sorted(companies)


@lru_cache(maxsize=1)
def _answer_engine() -> AnswerEngine:
    return AnswerEngine(adapter=RulesBasedAnswerAdapter(max_distance=0.45))


@lru_cache(maxsize=1)
def _retriever_v1() -> Retriever:
    return Retriever(
        chroma_path=CONFIG.chroma.dir,
        collection_name=CONFIG.chroma.collection_v1,
        model_name=CONFIG.models.embedding,
    )


@lru_cache(maxsize=1)
def _retriever_v2() -> RetrieverV2:
    return RetrieverV2(
        chroma_path=CONFIG.chroma.dir,
        collection_name=CONFIG.chroma.collection_v2,
        model_name=CONFIG.models.embedding,
    )


@lru_cache(maxsize=1)
def _retriever_v3() -> RetrieverV3:
    return RetrieverV3(
        chroma_path=CONFIG.chroma.dir,
        collection_name=CONFIG.chroma.collection_v2,
        model_name=CONFIG.models.embedding,
    )


@lru_cache(maxsize=1)
def _retriever_v5() -> RetrieverV5Hybrid:
    return RetrieverV5Hybrid(
        chroma_path=CONFIG.chroma.dir,
        collection_name=CONFIG.chroma.collection_v2,
        model_name=CONFIG.models.embedding,
        chunks_file=CONFIG.paths.chunks_v2_file,
    )


@lru_cache(maxsize=1)
def _retriever_v6() -> RetrieverV6Cross:
    return RetrieverV6Cross(
        chroma_path=CONFIG.chroma.dir,
        collection_name=CONFIG.chroma.collection_v2,
        model_name=CONFIG.models.embedding,
        chunks_file=CONFIG.paths.chunks_v2_file,
        cross_encoder_model=CONFIG.models.cross_encoder,
    )


def _clear_cached_components() -> None:
    _answer_engine.cache_clear()
    _retriever_v1.cache_clear()
    _retriever_v2.cache_clear()
    _retriever_v3.cache_clear()
    _retriever_v5.cache_clear()
    _retriever_v6.cache_clear()


def _extract_summary_bullets(answer_text: str) -> List[str]:
    bullets: List[str] = []
    for line in answer_text.splitlines():
        stripped = line.strip()
        if stripped == "Evidence":
            break
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def _is_found_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return "dokümanda bulunamadı" not in lowered and "dokumanda bulunamadi" not in lowered


def _short_excerpt(text: str, max_chars: int = 320) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _frame_to_csv(frame) -> str:
    if frame is None:
        return ""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def _sanitize_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for row in rows:
        clean: Dict[str, Any] = {}
        for key, value in row.items():
            try:
                clean[key] = None if isna(value) else value
            except Exception:
                clean[key] = value
        sanitized.append(clean)
    return sanitized


def _llm_assistant_enabled() -> bool:
    llm_cfg = getattr(CONFIG, "llm_assistant", None) or getattr(CONFIG, "llm_commentary", None)
    return bool(getattr(llm_cfg, "enabled", False))


def _empty_commentary() -> Dict[str, Any]:
    return dict(SAFE_EMPTY_COMMENTARY)


def _append_feedback(payload: Dict[str, Any]) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _serialize_evidence_from_chunks(chunks: List[RetrievedChunk], limit: int = 5) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for chunk in chunks[:limit]:
        confidence = float(chunk.final_score) if chunk.final_score is not None else float(chunk.score)
        confidence = max(0.0, min(1.0, confidence))
        evidence.append(
            {
                "doc_id": chunk.doc_id,
                "company": chunk.company,
                "quarter": chunk.quarter,
                "year": chunk.year,
                "page": chunk.page,
                "section_title": chunk.section_title,
                "excerpt": _short_excerpt(chunk.text, max_chars=360),
                "block_type": chunk.block_type,
                "confidence": round(confidence, 4),
                "verify_status": None,
                "verify_warnings": [],
            }
        )
    return evidence


def _retrieve_single(question: str, retriever_name: str, company: Optional[str] = None) -> List[RetrievedChunk]:
    cfg = CONFIG.retrieval
    if retriever_name == "v1":
        return _retriever_v1().retrieve(question, top_k=cfg.top_k_final, company=company)
    if retriever_name == "v2":
        return _retriever_v2().retrieve_with_boost(
            query=question,
            top_k_initial=cfg.v2_top_k_initial,
            top_k_final=cfg.top_k_final,
            alpha=cfg.alpha_v2,
            company=company,
        )
    if retriever_name == "v5":
        return _retriever_v5().retrieve_with_hybrid(
            query=question,
            top_k_vector=cfg.v5_top_k_vector,
            top_k_bm25=cfg.v5_top_k_bm25,
            top_k_final=cfg.top_k_final,
            beta=cfg.beta_v5,
            alpha_v3=cfg.alpha_v3,
            company_override=company,
        )
    if retriever_name == "v6":
        return _retriever_v6().retrieve_with_cross_encoder(
            query=question,
            top_k_candidates=cfg.v6_cross_top_n,
            top_k_final=cfg.top_k_final,
            top_k_vector=cfg.v5_top_k_vector,
            top_k_bm25=cfg.v5_top_k_bm25,
            beta=cfg.beta_v5,
            alpha_v3=cfg.alpha_v3,
            company_override=company,
        )
    return _retriever_v3().retrieve_with_query_awareness(
        query=question,
        top_k_initial=cfg.v3_top_k_initial,
        top_k_final=cfg.top_k_final,
        alpha=cfg.alpha_v3,
        company_override=company,
    )


def _retrieve_for_quarter(
    question: str,
    retriever_name: str,
    quarter: str,
    company: Optional[str] = None,
) -> List[RetrievedChunk]:
    cfg = CONFIG.retrieval
    if retriever_name == "v1":
        return _retriever_v1().retrieve(
            question,
            top_k=cfg.top_k_final,
            quarter=quarter,
            company=company,
        )
    if retriever_name == "v2":
        return _retriever_v2().retrieve_with_boost(
            query=question,
            top_k_initial=cfg.v2_top_k_initial,
            top_k_final=cfg.top_k_final,
            alpha=cfg.alpha_v2,
            quarter=quarter,
            company=company,
        )
    if retriever_name == "v5":
        return _retriever_v5().retrieve_with_hybrid(
            query=question,
            top_k_vector=cfg.v5_top_k_vector,
            top_k_bm25=cfg.v5_top_k_bm25,
            top_k_final=cfg.top_k_final,
            beta=cfg.beta_v5,
            alpha_v3=cfg.alpha_v3,
            quarter_override=quarter,
            company_override=company,
        )
    if retriever_name == "v6":
        return _retriever_v6().retrieve_with_cross_encoder(
            query=question,
            top_k_candidates=cfg.v6_cross_top_n,
            top_k_final=cfg.top_k_final,
            top_k_vector=cfg.v5_top_k_vector,
            top_k_bm25=cfg.v5_top_k_bm25,
            beta=cfg.beta_v5,
            alpha_v3=cfg.alpha_v3,
            quarter_override=quarter,
            company_override=company,
        )
    return _retriever_v3().retrieve_with_query_awareness(
        query=question,
        top_k_initial=cfg.v3_top_k_initial,
        top_k_final=cfg.top_k_final,
        alpha=cfg.alpha_v3,
        quarter_override=quarter,
        company_override=company,
    )


def _run_trend_mode(question: str, retriever_name: str, company: Optional[str] = None) -> Dict[str, Any]:
    metric = infer_metric_from_question(question)
    quarter_chunks: Dict[str, List[RetrievedChunk]] = {}

    for quarter in QUARTER_ORDER:
        q = build_metric_query(metric, quarter, question) if metric else question
        quarter_chunks[quarter] = _retrieve_for_quarter(
            q,
            retriever_name=retriever_name,
            quarter=quarter,
            company=company,
        )

    if not metric:
        sources = collect_top_sources(quarter_chunks=quarter_chunks)
        evidence = [
            {
                "doc_id": source["doc_id"],
                "company": source.get("company", company),
                "quarter": source["quarter"],
                "page": source["page"],
                "section_title": source["section_title"],
                "excerpt": "",
                "block_type": "unknown",
                "confidence": None,
                "reasons": [],
            }
            for source in sources
        ]
        return {
            "found": False,
            "bullets": [
                "Dokümanda bulunamadı",
                "Trend metrik tipi tespit edilemedi.",
            ],
            "evidence": evidence,
            "top_k": CONFIG.retrieval.top_k_final,
        }

    frame, records = aggregate_metric_across_quarters(quarter_chunks=quarter_chunks, metric=metric)
    overall = compute_overall_change(frame)
    missing_quarters = [str(row["quarter"]) for _, row in frame.iterrows() if isna(row["value"])]
    found = bool(records)

    bullets = [f"Metrik: {metric_display_name(metric)}"]
    for _, row in frame.iterrows():
        bullets.append(f"{row['quarter']}: {row['value_display']}")
    if overall.get("abs_change") is not None:
        bullets.append(
            f"Q1->Q3 degisim: {overall['abs_change']:.2f} ({overall['pct_change']:.2f}% | {overall['direction']})"
        )
    if missing_quarters:
        bullets.append(f"Eksik ceyrekler: {', '.join(missing_quarters)}")
    if not found:
        bullets.insert(0, "Dokümanda bulunamadı")

    evidence = [
        {
            "doc_id": record["doc_id"],
            "company": record.get("company", company),
            "quarter": record["quarter"],
            "page": record["page"],
            "section_title": record["section_title"],
            "excerpt": _short_excerpt(str(record.get("excerpt", ""))),
            "block_type": record.get("block_type", "text"),
            "confidence": record.get("confidence"),
            "reasons": record.get("reasons", []),
            "verify_status": record.get("verify_status"),
            "verify_warnings": record.get("verify_warnings", []),
        }
        for record in records
    ]

    if not evidence:
        for source in collect_top_sources(quarter_chunks=quarter_chunks):
            evidence.append(
                {
                    "doc_id": source["doc_id"],
                    "company": source.get("company", company),
                    "quarter": source["quarter"],
                    "page": source["page"],
                    "section_title": source["section_title"],
                    "excerpt": "",
                    "block_type": "unknown",
                    "confidence": None,
                    "reasons": [],
                    "verify_status": "FAIL",
                    "verify_warnings": ["veri_bulunamadi"],
                }
            )

    return {
        "found": found,
        "bullets": bullets,
        "evidence": evidence,
        "frame": frame,
        "confidence": min(
            [float(record.get("confidence")) for record in records if record.get("confidence") is not None],
            default=0.0 if not found else 0.5,
        ),
        "top_k": CONFIG.retrieval.top_k_final,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return _stats_payload()


@app.post("/ingest")
def ingest() -> Dict[str, Any]:
    pages, summary = ingest_raw_pdfs(
        raw_dir=CONFIG.paths.raw_dir,
        output_file=CONFIG.paths.pages_file,
    )
    return {
        "message": "ingest_completed",
        "pages_written": len(pages),
        "summary": summary,
    }


@app.post("/index")
def index(request: IndexRequest) -> Dict[str, Any]:
    if request.version == "v1":
        summary = build_index(
            raw_dir=CONFIG.paths.raw_dir,
            processed_dir=CONFIG.paths.processed_dir,
            collection_name=CONFIG.chroma.collection_v1,
            chunk_size=CONFIG.chunking.v1.chunk_size,
            overlap=CONFIG.chunking.v1.overlap,
        )
    elif request.version == "v2":
        summary = build_index_v2(
            raw_dir=CONFIG.paths.raw_dir,
            processed_dir=CONFIG.paths.processed_dir,
            collection_name=CONFIG.chroma.collection_v2,
            chunk_size=CONFIG.chunking.v2.chunk_size,
            overlap=CONFIG.chunking.v2.overlap,
        )
    else:
        raise HTTPException(status_code=400, detail="version v1 veya v2 olmali")
    _clear_cached_components()
    return {"message": "index_completed", "version": request.version, "summary": summary}


@app.post("/ask")
def ask(request: AskRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    parsed = parse_query(request.question)
    use_trend = request.mode == "trend" or is_comparison_query(request.question)
    available_companies = _available_companies_from_chunks(CONFIG.paths.chunks_v2_file)
    mentioned_companies = detect_company_mentions(request.question, available_companies=available_companies)
    forced_company = request.company.upper().strip() if request.company else None
    comparison_companies = list(dict.fromkeys(([forced_company] if forced_company else []) + mentioned_companies))
    use_cross_company = is_cross_company_query(
        request.question,
        available_companies=available_companies,
    ) and (len(comparison_companies) >= 2 or len(available_companies) >= 2)

    if use_cross_company:
        if len(comparison_companies) < 2:
            comparison_companies = available_companies[:3]
        comparison = run_cross_company_comparison(
            question=request.question,
            retriever=_retriever_v3(),
            companies=comparison_companies,
            top_k_initial=CONFIG.retrieval.v3_top_k_initial,
            top_k_final=CONFIG.retrieval.top_k_final,
            alpha=CONFIG.retrieval.alpha_v3,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        evidence = []
        for record in comparison.get("evidence", [])[:12]:
            evidence.append(
                {
                    "doc_id": record.get("doc_id"),
                    "company": record.get("company"),
                    "quarter": record.get("quarter"),
                    "page": record.get("page"),
                    "section_title": record.get("section_title"),
                    "excerpt": _short_excerpt(str(record.get("excerpt", ""))),
                    "block_type": record.get("block_type", "text"),
                    "confidence": record.get("confidence"),
                    "reasons": record.get("reasons", []),
                    "verify_status": record.get("verify_status"),
                    "verify_warnings": record.get("verify_warnings", []),
                }
            )
        frame = comparison.get("frame")
        table_rows = (
            _sanitize_records(frame.to_dict(orient="records")) if frame is not None and not frame.empty else []
        )
        bullets = [
            comparison.get("message", "Karsilastirma tamamlandi."),
            f"Hedef metrik: {comparison.get('target')}",
        ]
        if comparison.get("best_company"):
            bullets.append(f"En iyi sirket: {comparison['best_company']}")
        if not comparison.get("found"):
            bullets.insert(0, "Dokümanda bulunamadı")
        verify_values = [str(item.get("verify_status")) for item in evidence if item.get("verify_status")]
        if "FAIL" in verify_values:
            cross_verify = "FAIL"
        elif "WARN" in verify_values:
            cross_verify = "WARN"
        elif verify_values:
            cross_verify = "PASS"
        else:
            cross_verify = "WARN"
        response_payload = {
            "answer": {
                "bullets": bullets,
                "found": bool(comparison.get("found")),
                "confidence": comparison.get("best_confidence"),
                "verify_status": cross_verify,
            },
            "parsed": {
                "quarter": parsed.get("quarter"),
                "query_type": parsed.get("signals", {}).get("query_type"),
                "company": forced_company,
                "mentioned_companies": mentioned_companies,
            },
            "comparison": {
                "mode": "cross_company",
                "target": comparison.get("target"),
                "best_company": comparison.get("best_company"),
                "best_value": comparison.get("best_value"),
                "best_confidence": comparison.get("best_confidence"),
                "rows": table_rows,
            },
            "evidence": evidence,
            "debug": {
                "retriever": "v3",
                "latency_ms": round(latency_ms, 2),
                "top_k": CONFIG.retrieval.top_k_final,
            },
        }
        return response_payload

    if use_trend:
        trend = _run_trend_mode(
            question=request.question,
            retriever_name=request.retriever,
            company=forced_company,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        frame = trend.get("frame")
        verify_values = [
            str(item.get("verify_status"))
            for item in trend.get("evidence", [])
            if item.get("verify_status")
        ]
        if "FAIL" in verify_values:
            trend_verify_status = "FAIL"
        elif "WARN" in verify_values:
            trend_verify_status = "WARN"
        elif verify_values:
            trend_verify_status = "PASS"
        else:
            trend_verify_status = "FAIL" if not trend["found"] else "WARN"
        response_payload = {
            "answer": {
                "bullets": trend["bullets"],
                "found": trend["found"],
                "confidence": trend.get("confidence"),
                "verify_status": trend_verify_status,
            },
            "parsed": {
                "quarter": parsed.get("quarter"),
                "query_type": parsed.get("signals", {}).get("query_type"),
                "company": forced_company,
            },
            "trend": {
                "rows": _sanitize_records(frame.to_dict(orient="records")) if frame is not None and not frame.empty else [],
            },
            "evidence": trend["evidence"],
            "debug": {
                "retriever": request.retriever,
                "latency_ms": round(latency_ms, 2),
                "top_k": trend["top_k"],
            },
        }
        return response_payload

    chunks = _retrieve_single(
        question=request.question,
        retriever_name=request.retriever,
        company=forced_company,
    )
    answer_text = _answer_engine().answer(question=request.question, chunks=chunks)
    found = _is_found_answer(answer_text)
    bullets = _extract_summary_bullets(answer_text)
    if not bullets:
        bullets = ["Dokümanda bulunamadı" if not found else answer_text.strip()]
    evidence = _serialize_evidence_from_chunks(chunks, limit=CONFIG.retrieval.top_k_final)
    latency_ms = (time.perf_counter() - started) * 1000.0
    top_confidence = evidence[0]["confidence"] if evidence else 0.0
    answer_confidence = float(top_confidence) if found else 0.0

    response_payload = {
        "answer": {
            "bullets": bullets,
            "found": found,
            "confidence": round(answer_confidence, 4),
            "verify_status": "WARN" if found else "FAIL",
        },
        "parsed": {
            "quarter": parsed.get("quarter"),
            "query_type": parsed.get("signals", {}).get("query_type"),
            "company": forced_company,
        },
        "evidence": evidence,
        "debug": {
            "retriever": request.retriever,
            "latency_ms": round(latency_ms, 2),
            "top_k": CONFIG.retrieval.top_k_final,
        },
    }
    return response_payload


@app.post("/commentary")
def commentary(request: CommentaryRequest) -> Dict[str, Any]:
    if not _llm_assistant_enabled():
        return _empty_commentary()
    if not isinstance(request.answer_payload, dict):
        raise HTTPException(status_code=400, detail="answer_payload object olmali")

    found = bool(
        request.answer_payload.get("found", (request.answer_payload.get("answer") or {}).get("found"))
    )
    if not found:
        return _empty_commentary()

    try:
        commentary_payload = generate_commentary(
            answer_payload=dict(request.answer_payload),
            question=request.question,
            cfg=CONFIG,
            company=request.company,
            year=request.year,
            quarter=request.quarter,
            model_override=request.model,
        )
    except TypeError:
        # Backward compatibility for older generate_commentary(commentary_input, cfg) signatures.
        commentary_payload = generate_commentary(dict(request.answer_payload), CONFIG)
    if not any(commentary_payload.values()):
        return _empty_commentary()
    return commentary_payload


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> Dict[str, Any]:
    payload = {
        "timestamp": request.timestamp or datetime.now(timezone.utc).isoformat(),
        "company": request.company.upper().strip() if request.company else None,
        "quarter": request.quarter,
        "metric": request.metric,
        "extracted_value": request.extracted_value,
        "user_value": request.user_value,
        "evidence_ref": request.evidence_ref,
        "verdict": request.verdict,
    }
    _append_feedback(payload)
    return {"message": "feedback_saved", "path": str(FEEDBACK_FILE), "feedback": payload}


@app.get("/export")
def export_table(
    type: Literal["trend", "ratio"] = Query(..., alias="type"),
    company: Optional[str] = None,
) -> PlainTextResponse:
    company_norm = company.upper().strip() if company else None

    if type == "trend":
        trend = _run_trend_mode(
            question="Q1 Q2 Q3 net kar trendi",
            retriever_name="v3",
            company=company_norm,
        )
        frame = trend.get("frame")
        csv_text = _frame_to_csv(frame)
        filename = f"trend_{company_norm or 'ALL'}.csv"
    else:
        ratio = build_ratio_table(
            question="Q1 Q2 Q3 finansal oranlar",
            retriever=_retriever_v3(),
            company=company_norm,
            top_k_initial=CONFIG.retrieval.v3_top_k_initial,
            top_k_final=CONFIG.retrieval.top_k_final,
            alpha=CONFIG.retrieval.alpha_v3,
        )
        frame = ratio.get("frame")
        csv_text = _frame_to_csv(frame)
        filename = f"ratio_{company_norm or 'ALL'}.csv"

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return PlainTextResponse(content=csv_text, media_type="text/csv", headers=headers)
