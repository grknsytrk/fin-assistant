from __future__ import annotations

import io
import html
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")
app = FastAPI(title="RAG-Fin API", version="0.10.0")
FEEDBACK_FILE = CONFIG.paths.processed_dir / "feedback.jsonl"

# region agent log helpers
_DEBUG_LOG_PATH = Path("debug-0cbd9f.log")
_DEBUG_SESSION_ID = "0cbd9f"
_DEBUG_RUN_ID = "market-flow-debug-v1"


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": _DEBUG_RUN_ID,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion


def _cors_allow_origins() -> List[str]:
    raw = os.getenv("RAGFIN_CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def _cors_allow_origin_regex() -> str:
    raw = os.getenv("RAGFIN_CORS_ALLOW_ORIGIN_REGEX", "").strip()
    if raw:
        return raw
    # Allow local dev hosts on any port (Vite may auto-switch ports such as 5174/5175).
    return r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$"


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_origin_regex=_cors_allow_origin_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _company_breakdown_from_chunks(chunks_file: Path) -> List[Dict[str, Any]]:
    if not chunks_file.exists():
        return []

    counts: Dict[str, Dict[str, Any]] = {}
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue

            company_raw = str(payload.get("company", "")).strip().upper()
            if not company_raw:
                continue
            company = company_raw
            quarter = str(payload.get("quarter", "")).strip().upper()
            row = counts.setdefault(company, {"company": company, "chunks": 0, "quarters": set()})
            row["chunks"] += 1
            if quarter:
                row["quarters"].add(quarter)

    rows: List[Dict[str, Any]] = []
    for row in counts.values():
        rows.append(
            {
                "company": row["company"],
                "chunks": int(row["chunks"]),
                "quarters": sorted(list(row["quarters"])),
                "quarter_count": len(row["quarters"]),
            }
        )
    rows.sort(key=lambda item: (-int(item["chunks"]), str(item["company"])))
    return rows


def _quarter_label_sort_key(label: str) -> tuple[int, int, str]:
    normalized = str(label or "").strip().upper()
    if not normalized:
        return (0, 0, "")
    quarter_match = re.match(r"^(\d{4})Q([1-4])$", normalized)
    if quarter_match:
        return (int(quarter_match.group(1)), int(quarter_match.group(2)) * 3, normalized)
    period_match = re.match(r"^(\d{4})[/-](\d{1,2})$", normalized)
    if period_match:
        return (int(period_match.group(1)), int(period_match.group(2)), normalized)
    return (0, 0, normalized)


def _latest_quarter_label(quarters: List[str]) -> Optional[str]:
    candidates = [str(item or "").strip().upper() for item in quarters if str(item or "").strip()]
    if not candidates:
        return None
    return max(candidates, key=_quarter_label_sort_key)


def _load_cached_kap_market_metadata(cache_dir: Path, symbol: str) -> Dict[str, Any]:
    cache_file = cache_dir / f"{symbol}.json"
    if not cache_file.exists():
        return {}
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    quarters_raw = payload.get("quarters")
    quarters = [
        str(row.get("quarter") or "").strip().upper()
        for row in (quarters_raw or [])
        if isinstance(row, dict) and str(row.get("quarter") or "").strip()
    ]
    return {
        "latest_quarter": _latest_quarter_label(quarters),
        "has_kap_cache": True,
    }


_UNIVERSE_CACHE: Dict[str, Any] = {}
_UNIVERSE_CACHE_TTL = 120  # 2 minutes


def _market_universe_payload() -> Dict[str, Any]:
    from app.kap_service import get_bist100_companies

    now_ts = time.time()
    cached = _UNIVERSE_CACHE.get("payload")
    if cached and now_ts - cached.get("_ts", 0) < _UNIVERSE_CACHE_TTL:
        return cached["data"]

    stats = _stats_payload()
    symbols = get_bist100_companies()
    breakdown_rows = _company_breakdown_from_chunks(CONFIG.paths.chunks_v2_file)
    breakdown_map = {
        str(row.get("company") or "").strip().upper(): row
        for row in breakdown_rows
        if str(row.get("company") or "").strip()
    }
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    base_price_map = _fetch_market_price_map(symbols)
    price_map = _fill_prices_via_yahoo(symbols, base_price_map)

    rows: List[Dict[str, Any]] = []
    rag_ready_count = 0
    kap_cache_count = 0

    for symbol in symbols:
        breakdown_row = breakdown_map.get(symbol, {})
        cached_meta = _load_cached_kap_market_metadata(cache_dir, symbol)
        quarters = [
            str(item or "").strip().upper()
            for item in (breakdown_row.get("quarters") or [])
            if str(item or "").strip()
        ]
        has_rag = bool(breakdown_row)
        latest_quarter = _latest_quarter_label(
            quarters + ([cached_meta["latest_quarter"]] if cached_meta.get("latest_quarter") else [])
        )
        has_kap_cache = bool(cached_meta.get("has_kap_cache"))
        quote = price_map.get(symbol, {})
        if has_rag:
            rag_ready_count += 1
        if has_kap_cache:
            kap_cache_count += 1

        rows.append(
            {
                "company": symbol,
                "chunks": int(breakdown_row.get("chunks") or 0),
                "quarter_count": int(breakdown_row.get("quarter_count") or 0),
                "latest_quarter": latest_quarter,
                "has_rag": has_rag,
                "has_kap_cache": has_kap_cache,
                "price": quote.get("price"),
                "price_currency": quote.get("currency"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "price_as_of": quote.get("as_of"),
            }
        )

    coverage_rows = sorted(
        [row for row in rows if row["has_rag"]],
        key=lambda item: (-int(item["quarter_count"]), -int(item["chunks"]), str(item["company"])),
    )[:8]

    data = {
        "stats": {
            "bist100_count": len(rows),
            "rag_ready_count": rag_ready_count,
            "kap_only_count": len(rows) - rag_ready_count,
            "kap_cache_count": kap_cache_count,
            "pdf_count": int(stats.get("pdf_count") or 0),
            "page_count": int(stats.get("page_count") or 0),
        },
        "rows": rows,
        "coverage_rows": coverage_rows,
    }
    _UNIVERSE_CACHE["payload"] = {"_ts": now_ts, "data": data}
    return data


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


class _ComparisonRetrieverAdapter:
    """Bridge ratio-engine comparison calls to the selected retriever implementation."""

    def __init__(self, retriever_name: str) -> None:
        self.retriever_name = str(retriever_name or "v3")
        if self.retriever_name == "v1":
            self.base_retriever = _retriever_v1()
        elif self.retriever_name == "v2":
            self.base_retriever = _retriever_v2()
        elif self.retriever_name == "v5":
            self.base_retriever = _retriever_v5()
        elif self.retriever_name == "v6":
            self.base_retriever = _retriever_v6()
        else:
            self.base_retriever = _retriever_v3()
            self.retriever_name = "v3"

        # Preserve local-fallback behavior in ratio_engine.build_ratio_table().
        self.collection = getattr(self.base_retriever, "collection", None)
        self.client = getattr(self.base_retriever, "client", None)

    def retrieve_with_query_awareness(
        self,
        query: str,
        top_k_initial: int = 20,
        top_k_final: int = 5,
        alpha: float = 0.35,
        quarter_override: Optional[str] = None,
        company_override: Optional[str] = None,
        allow_quarter_fallback: bool = True,
    ) -> List[RetrievedChunk]:
        if self.retriever_name == "v1":
            return self.base_retriever.retrieve(
                query=query,
                top_k=top_k_final,
                quarter=quarter_override,
                company=company_override,
            )
        if self.retriever_name == "v2":
            return self.base_retriever.retrieve_with_boost(
                query=query,
                top_k_initial=top_k_initial,
                top_k_final=top_k_final,
                alpha=alpha,
                quarter=quarter_override,
                company=company_override,
            )
        if self.retriever_name == "v5":
            return self.base_retriever.retrieve_with_hybrid(
                query=query,
                top_k_vector=top_k_initial,
                top_k_bm25=top_k_initial,
                top_k_final=top_k_final,
                beta=CONFIG.retrieval.beta_v5,
                alpha_v3=alpha,
                quarter_override=quarter_override,
                company_override=company_override,
            )
        if self.retriever_name == "v6":
            return self.base_retriever.retrieve_with_cross_encoder(
                query=query,
                top_k_candidates=max(CONFIG.retrieval.v6_cross_top_n, top_k_final),
                top_k_final=top_k_final,
                top_k_vector=top_k_initial,
                top_k_bm25=top_k_initial,
                beta=CONFIG.retrieval.beta_v5,
                alpha_v3=alpha,
                quarter_override=quarter_override,
                company_override=company_override,
            )
        return self.base_retriever.retrieve_with_query_awareness(
            query=query,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            alpha=alpha,
            quarter_override=quarter_override,
            company_override=company_override,
            allow_quarter_fallback=allow_quarter_fallback,
        )


def _comparison_retriever(retriever_name: str) -> _ComparisonRetrieverAdapter:
    return _ComparisonRetrieverAdapter(retriever_name)


def _comparison_top_k_initial(retriever_name: str) -> int:
    if retriever_name == "v2":
        return CONFIG.retrieval.v2_top_k_initial
    if retriever_name in {"v5", "v6"}:
        return max(CONFIG.retrieval.v5_top_k_vector, CONFIG.retrieval.v5_top_k_bm25)
    return CONFIG.retrieval.v3_top_k_initial


def _comparison_alpha(retriever_name: str) -> float:
    if retriever_name == "v2":
        return CONFIG.retrieval.alpha_v2
    return CONFIG.retrieval.alpha_v3


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


@app.get("/stats/company-breakdown")
def stats_company_breakdown() -> Dict[str, Any]:
    return {"rows": _company_breakdown_from_chunks(CONFIG.paths.chunks_v2_file)}


@app.get("/market/universe")
def market_universe() -> Dict[str, Any]:
    return _market_universe_payload()


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
            retriever=_comparison_retriever(request.retriever),
            companies=comparison_companies,
            top_k_initial=_comparison_top_k_initial(request.retriever),
            top_k_final=CONFIG.retrieval.top_k_final,
            alpha=_comparison_alpha(request.retriever),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        evidence = []
        for record in comparison.get("evidence", [])[:12]:
            evidence.append(
                {
                    "doc_id": record.get("doc_id"),
                    "company": record.get("company"),
                    "year": record.get("year"),
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
                "retriever": request.retriever,
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
            "answer_text": answer_text.strip() if found else "",
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


_FLOW_CACHE: Dict[str, Any] = {}
_FLOW_CACHE_TTL = 180
# When the VYK feed fetch budget is spent we still want the last successful
# payload served for a while even if the cache itself is stale.
_FLOW_STALE_SERVE_WINDOW = 15 * 60


def _parse_kap_publish_date(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    token = str(raw).strip()
    if not token:
        return None
    for fmt in (
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None


# Map KAP disclosureType codes -> Turkish UI labels for the feed.
_KAP_TYPE_LABELS: Dict[str, str] = {
    "ODA": "Özel Durum",
    "FR": "Finansal Rapor",
    "FR_Consolidated": "Finansal Rapor",
    "FR_Solo": "Finansal Rapor",
    "KBR": "Kâr Payı",
    "DD": "Diğer Duyuru",
    "MD": "Mali Duyuru",
    "GK": "Genel Kurul",
    "FDR": "Faaliyet Raporu",
    "GR": "Geri Alım",
    "SR": "Sürdürülebilirlik",
    "CG": "Kurumsal Yönetim",
}


def _kap_category(disclosure_type: str, subject: str) -> str:
    dt = (disclosure_type or "").strip().upper()
    subj = (subject or "").lower()
    if dt.startswith("FR"):
        return "finansal_rapor"
    if "kar pay" in subj or dt == "KBR":
        return "kar_payi"
    if "geri alma" in subj or "geri alım" in subj or dt == "GR":
        return "geri_alim"
    if "genel kurul" in subj or dt == "GK":
        return "genel_kurul"
    if "kredi derec" in subj:
        return "kredi_derecelendirme"
    if "sürdürülebilir" in subj or dt == "SR":
        return "surdurulebilirlik"
    if "faaliyet rapor" in subj or dt == "FDR":
        return "faaliyet_raporu"
    return "bildirim"


def _kap_source_label(disclosure_type: str) -> str:
    dt = (disclosure_type or "").strip().upper()
    return _KAP_TYPE_LABELS.get(dt, "KAP")


_KAP_PUBLIC_LAST_ERROR: Dict[str, Any] = {"message": None, "ts": 0.0, "source": None}
_KAP_SESSION: Dict[str, Any] = {"opener": None, "bootstrapped_at": 0.0}
_KAP_SESSION_TTL = 15 * 60  # 15 dakika session yeniden kurulur


_KAP_DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    "Accept-Encoding": "identity",
}


def _kap_opener(force: bool = False) -> Any:
    """Build (and cache) a cookie-aware opener for kap.org.tr.

    Bootstraps a browser-like session by first requesting the HTML search page
    so KAP's WAF issues the cookies that later JSON calls require. Cached
    until TTL expires or `force=True`.
    """
    import http.cookiejar
    import urllib.error
    import urllib.request

    now = time.time()
    opener = _KAP_SESSION.get("opener")
    bootstrapped_at = float(_KAP_SESSION.get("bootstrapped_at") or 0.0)
    if opener is not None and not force and (now - bootstrapped_at) < _KAP_SESSION_TTL:
        return opener

    jar = http.cookiejar.CookieJar()
    new_opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPRedirectHandler(),
    )
    new_opener.addheaders = list(_KAP_DEFAULT_HEADERS.items())

    bootstrap_urls = [
        "https://www.kap.org.tr/tr/",
        "https://www.kap.org.tr/tr/bildirim-sorgu",
    ]
    bootstrap_results: List[Dict[str, Any]] = []
    for boot_url in bootstrap_urls:
        boot_started = time.time()
        try:
            req = urllib.request.Request(
                boot_url,
                headers={
                    **_KAP_DEFAULT_HEADERS,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Upgrade-Insecure-Requests": "1",
                },
            )
            with new_opener.open(req, timeout=8) as resp:
                resp.read(1024)
            bootstrap_results.append(
                {
                    "url": boot_url,
                    "ok": True,
                    "elapsed_ms": int((time.time() - boot_started) * 1000),
                    "cookies": len(list(jar)),
                }
            )
        except (urllib.error.URLError, TimeoutError, Exception) as exc:  # noqa: BLE001
            bootstrap_results.append(
                {
                    "url": boot_url,
                    "ok": False,
                    "elapsed_ms": int((time.time() - boot_started) * 1000),
                    "error": type(exc).__name__,
                    "cookies": len(list(jar)),
                }
            )
            continue

    _KAP_SESSION["opener"] = new_opener
    _KAP_SESSION["bootstrapped_at"] = now
    # region agent log
    _debug_log(
        "H1",
        "app/api.py:1230",
        "KAP bootstrap completed",
        {
            "force": force,
            "cookie_count": len(list(jar)),
            "results": bootstrap_results,
        },
    )
    # endregion
    return new_opener


def _fetch_kap_disclosures_via_url(
    url: str,
    timeout: float,
    opener: Any,
) -> tuple[Any, Optional[str]]:
    """Single attempt JSON fetch using the shared KAP session opener."""
    import urllib.error
    import urllib.request

    headers = {
        **_KAP_DEFAULT_HEADERS,
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    started = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        payload = json.loads(raw)
        # region agent log
        _debug_log(
            "H2",
            "app/api.py:1266",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": True,
                "elapsed_ms": int((time.time() - started) * 1000),
                "payload_type": type(payload).__name__,
                "item_count": len(payload) if isinstance(payload, list) else None,
            },
        )
        # endregion
        return payload, None
    except urllib.error.HTTPError as exc:
        # region agent log
        _debug_log(
            "H3",
            "app/api.py:1281",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": False,
                "elapsed_ms": int((time.time() - started) * 1000),
                "error": f"HTTPError {exc.code}",
            },
        )
        # endregion
        return None, f"HTTPError {exc.code}"
    except (urllib.error.URLError, TimeoutError, ValueError, Exception) as exc:  # noqa: BLE001
        error_text = f"{type(exc).__name__}: {exc}"
        # region agent log
        _debug_log(
            "H2",
            "app/api.py:1295",
            "KAP disclosures attempt finished",
            {
                "url": url,
                "timeout_s": timeout,
                "ok": False,
                "elapsed_ms": int((time.time() - started) * 1000),
                "error": error_text,
            },
        )
        # endregion
        return None, error_text


def _fetch_kap_public_disclosures(max_items: int = 80) -> List[Dict[str, Any]]:
    """Fetch recent disclosures from KAP's public (UI-facing) endpoint.

    This endpoint does not require authentication; it backs the KAP.org.tr
    "Bildirim Sorgu" screen. Returns a list in publishedAt-descending order.
    In this environment KAP's public disclosure feed may be WAF-protected.
    We probe the fastest-blocking variant first so repeated refreshes do not
    spend ~15-20 seconds timing out before falling back.
    """
    attempts = [
        (
            "https://www.kap.org.tr/tr/api/disclosures?main-category=all&sub-category=all&memberType=IGS",
            10.0,
            True,
        ),
        ("https://www.kap.org.tr/tr/api/disclosures", 6.0, False),
        ("https://www.kap.org.tr/tr/api/disclosures", 9.0, True),
    ]
    payload: Any = None
    last_error: Optional[str] = None
    last_source: Optional[str] = None
    for idx, (url, timeout, force_bootstrap) in enumerate(attempts):
        opener = _kap_opener(force=force_bootstrap)
        payload, last_error = _fetch_kap_disclosures_via_url(url, timeout, opener)
        last_source = url
        if isinstance(payload, list):
            last_error = None
            break
        if last_error == "HTTPError 666":
            # region agent log
            _debug_log(
                "H6",
                "app/api.py:1368",
                "KAP public feed blocked, skipping slower retries",
                {
                    "url": url,
                    "attempt_index": idx,
                    "error": last_error,
                },
            )
            # endregion
            break
        if idx < len(attempts) - 1:
            time.sleep(0.35)

    _KAP_PUBLIC_LAST_ERROR["message"] = last_error
    _KAP_PUBLIC_LAST_ERROR["ts"] = time.time()
    _KAP_PUBLIC_LAST_ERROR["source"] = last_source

    if not isinstance(payload, list):
        return []

    results: List[Dict[str, Any]] = []
    for node in payload:
        if not isinstance(node, dict):
            continue
        basic = node.get("basic") if isinstance(node.get("basic"), dict) else node
        if not isinstance(basic, dict):
            continue
        disclosure_index = basic.get("disclosureIndex")
        publish_raw = basic.get("publishDate") or basic.get("submittedDate") or basic.get("disclosureClass")
        parsed_dt = _parse_kap_publish_date(publish_raw)
        if parsed_dt is None:
            continue
        stock_codes_raw = str(basic.get("stockCodes") or basic.get("stockCode") or "").strip()
        stock_codes = [s.strip().upper() for s in stock_codes_raw.replace(";", ",").split(",") if s.strip()]
        symbol = stock_codes[0] if stock_codes else ""
        if not symbol:
            # Non-listed disclosures (e.g. regulator notes) — skip in ticker-centric feed
            continue

        title_candidates = [
            basic.get("title"),
            basic.get("summary"),
            (basic.get("kapTitle") or {}).get("tr") if isinstance(basic.get("kapTitle"), dict) else None,
            basic.get("subject"),
        ]
        title = ""
        for candidate in title_candidates:
            if candidate and str(candidate).strip():
                title = str(candidate).strip()
                break
        if not title:
            title = "KAP Bildirimi"

        subject = str(basic.get("subject") or "").strip()
        disclosure_type = str(basic.get("disclosureType") or basic.get("type") or "").strip()

        results.append(
            {
                "id": f"{symbol}-{disclosure_index or parsed_dt.isoformat()}",
                "source": _kap_source_label(disclosure_type),
                "symbol": symbol,
                "stock_codes": stock_codes,
                "title": title,
                "subject": subject,
                "published_at": parsed_dt.isoformat(),
                "category": _kap_category(disclosure_type, subject),
                "kap_url": (
                    f"https://www.kap.org.tr/tr/Bildirim/{disclosure_index}"
                    if disclosure_index is not None
                    else None
                ),
            }
        )
        if len(results) >= max_items:
            break

    return results


def _local_flow_items_from_cache() -> List[Dict[str, Any]]:
    """Fallback feed constructed from locally cached financial reports."""
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    started = time.time()
    items: List[Dict[str, Any]] = []
    if not cache_dir.exists():
        # region agent log
        _debug_log(
            "H4",
            "app/api.py:1391",
            "Local flow cache scan finished",
            {"cache_dir_exists": False, "file_count": 0, "item_count": 0, "elapsed_ms": 0},
        )
        # endregion
        return items
    cache_files = list(cache_dir.glob("*.json"))
    for cache_file in cache_files:
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        symbol = str(
            payload.get("stock_code") or payload.get("company") or cache_file.stem
        ).strip().upper()
        quarters = payload.get("quarters")
        if not isinstance(quarters, list):
            continue
        for quarter in quarters[:2]:
            if not isinstance(quarter, dict):
                continue
            parsed_dt = _parse_kap_publish_date(quarter.get("publish_date"))
            if parsed_dt is None:
                continue
            title = str(quarter.get("title") or "Finansal Rapor").strip()
            quarter_label = str(quarter.get("quarter") or "").strip()
            disclosure_id = quarter.get("disclosure_index")
            items.append(
                {
                    "id": f"{symbol}-{disclosure_id or quarter_label or parsed_dt.isoformat()}",
                    "source": "Finansal Rapor",
                    "symbol": symbol,
                    "stock_codes": [symbol],
                    "title": f"{title}{' - ' + quarter_label if quarter_label else ''}",
                    "subject": "Finansal Rapor",
                    "published_at": parsed_dt.isoformat(),
                    "category": "finansal_rapor",
                    "kap_url": (
                        f"https://www.kap.org.tr/tr/Bildirim/{disclosure_id}"
                        if disclosure_id
                        else None
                    ),
                }
            )
    # region agent log
    _debug_log(
        "H4",
        "app/api.py:1432",
        "Local flow cache scan finished",
        {
            "cache_dir_exists": True,
            "file_count": len(cache_files),
            "item_count": len(items),
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    return items


_FLOW_DEGRADED_TTL = 25  # Tekrar canlı kaynağı dene: 25 sn sonra


# region VYK (resmi KAP REST) yardimcilari
# Son 24 saat disiplini icin varsayilan pencere. Kullanici akisin cok uzun
# bir zaman dilimini cekmesini istemedigi icin dar tutuluyor.
_VYK_DEFAULT_WINDOW_HOURS = 24
# `/disclosures` cagrisi 50 kayit dondurur; bu bugette `disclosureDetail`
# icin iki-sayfa disinda kalmalik ayirir.
_VYK_DEFAULT_LIST_BUDGET = 2
# Her refresh'te en fazla bu kadar `disclosureDetail` cagrisi yapilir;
# gerisi sessizce atlanir. Gateway'in "cok fazla istek" sikayetini onler.
_VYK_DEFAULT_DETAIL_BUDGET = 25
# Kullanici akisi genisletmek isteyebilir; bu sinir gateway'i bunaltmadan
# saglik sinirinda tutar.
_VYK_DEFAULT_DETAIL_BUDGET_MAX = 500
_VYK_DETAIL_WORKERS = 8


def _vyk_source_label(disclosure_class: str, disclosure_type: str, subject_tr: str) -> str:
    cls = (disclosure_class or "").upper().strip()
    typ = (disclosure_type or "").upper().strip()
    subj = (subject_tr or "").lower()
    if typ == "CA":
        if "kar pay" in subj:
            return "Kâr Payı"
        if "genel kurul" in subj:
            return "Genel Kurul"
        if "geri al" in subj:
            return "Geri Alım"
        if "sermaye" in subj:
            return "Sermaye Artırımı"
        return "Hak Kullanımı"
    if typ == "FON":
        return "Fon"
    if typ.startswith("FR") or cls == "FR":
        return "Finansal Rapor"
    if typ == "ODA" or cls == "ODA":
        return "Özel Durum"
    if typ == "DUY" or cls == "DUY":
        return "Düzenleyici Kurum"
    if typ == "DG" or cls == "DG":
        return "Diğer Bildirim"
    return "KAP"


def _vyk_category(disclosure_class: str, disclosure_type: str, subject_tr: str) -> str:
    cls = (disclosure_class or "").upper().strip()
    typ = (disclosure_type or "").upper().strip()
    subj = (subject_tr or "").lower()
    if typ.startswith("FR") or cls == "FR":
        return "finansal_rapor"
    if typ == "ODA" or cls == "ODA":
        return "ozel_durum"
    if "kar pay" in subj:
        return "kar_payi"
    if "genel kurul" in subj:
        return "genel_kurul"
    if "geri al" in subj:
        return "geri_alim"
    if "kredi derec" in subj:
        return "kredi_derecelendirme"
    if "sürdürülebilir" in subj:
        return "surdurulebilirlik"
    if "faaliyet rapor" in subj:
        return "faaliyet_raporu"
    return "bildirim"


def _fetch_kap_vyk_feed(
    *,
    window_hours: int = _VYK_DEFAULT_WINDOW_HOURS,
    list_pages: int = _VYK_DEFAULT_LIST_BUDGET,
    detail_budget: int = _VYK_DEFAULT_DETAIL_BUDGET,
) -> List[Dict[str, Any]]:
    """Build a flow feed from the official VYK REST endpoints.

    Returns feed items sorted by `published_at` desc, filtered to the last
    `window_hours` hours. Returns `[]` when credentials are missing or any
    upstream call fails so the caller can fall back to the local cache.
    """
    import concurrent.futures

    from src import kap_vyk_client

    cfg = getattr(CONFIG, "kap", None)
    if cfg is None or not kap_vyk_client.is_enabled(cfg):
        return []

    started = time.time()
    last_index = kap_vyk_client.get_last_disclosure_index(cfg)
    if not last_index or last_index <= 0:
        return []

    # `/disclosures` sayfalari 50 kayitlik. `list_pages` ile toplam pencereyi
    # kontrol altinda tutup upstream'e bindirmiyoruz.
    pages = max(1, min(int(list_pages or 1), 10))
    disclosures: List[Dict[str, Any]] = []
    cursor = int(last_index)
    seen_indexes: set[str] = set()
    for _ in range(pages):
        start_index = max(1, cursor - 49)
        rows = kap_vyk_client.list_disclosures_batch(cfg, start_index=start_index)
        if not rows:
            break
        added = 0
        for row in rows:
            idx = str(row.get("disclosureIndex") or "").strip()
            if not idx or idx in seen_indexes:
                continue
            seen_indexes.add(idx)
            disclosures.append(row)
            added += 1
        if added == 0:
            break
        cursor = start_index - 1
        if cursor <= 0:
            break

    if not disclosures:
        return []

    # Daha yeni kayitlari once isle; 24 saat penceresi disina taskinca
    # pahali detail cagrilarini kesmek icin bu sirali yaklasim sart.
    disclosures.sort(
        key=lambda node: int(str(node.get("disclosureIndex") or "0") or "0"),
        reverse=True,
    )

    members = kap_vyk_client.build_company_lookup(cfg)
    window_delta = timedelta(hours=max(1, int(window_hours)))
    now = datetime.now()
    cutoff = now - window_delta

    budget = max(1, min(int(detail_budget or 1), _VYK_DEFAULT_DETAIL_BUDGET_MAX))
    pending = disclosures[:budget]

    workers = max(1, min(_VYK_DETAIL_WORKERS, len(pending)))
    details: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                kap_vyk_client.get_disclosure_detail, cfg, row.get("disclosureIndex")
            ): str(row.get("disclosureIndex") or "")
            for row in pending
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                detail = future.result()
            except Exception:
                detail = None
            if detail and idx:
                details[idx] = detail

    items: List[Dict[str, Any]] = []
    fallback_items: List[Dict[str, Any]] = []
    for row in pending:
        idx_str = str(row.get("disclosureIndex") or "").strip()
        if not idx_str:
            continue
        detail = details.get(idx_str) or {}

        time_raw = str(detail.get("time") or "").strip()
        published_dt = _parse_kap_publish_date(time_raw)
        if published_dt is None:
            # Zaman damgasi yoksa akista saglikli konumlandiramayiz; atla.
            continue
        in_window = published_dt >= cutoff

        disclosure_class = str(
            row.get("disclosureClass") or detail.get("disclosureClass") or ""
        ).upper().strip()
        disclosure_type = str(
            row.get("disclosureType") or detail.get("disclosureType") or ""
        ).upper().strip()

        subject_obj = detail.get("subject")
        subject_tr = ""
        if isinstance(subject_obj, dict):
            subject_tr = str(subject_obj.get("tr") or "").strip()

        summary_obj = detail.get("summary")
        summary_tr = ""
        if isinstance(summary_obj, dict):
            summary_tr = str(summary_obj.get("tr") or "").strip()

        company_id = str(row.get("companyId") or detail.get("senderId") or "").strip()
        member = members.get(company_id, {})
        member_title = str(
            member.get("title") or row.get("title") or detail.get("senderTitle") or ""
        ).strip()
        member_stock_raw = str(member.get("stockCode") or "").strip().upper()
        member_stock = member_stock_raw.split(",")[0].strip() if member_stock_raw else ""

        sender_codes_raw = detail.get("senderExchCodes") or []
        stock_codes: List[str] = []
        primary_stock = ""
        if isinstance(sender_codes_raw, list) and sender_codes_raw:
            stock_codes = [
                str(code).strip().upper()
                for code in sender_codes_raw
                if str(code or "").strip()
            ]
            primary_stock = stock_codes[0] if stock_codes else ""
        if not primary_stock and member_stock:
            primary_stock = member_stock
            stock_codes = [member_stock]

        fund_code = str(
            detail.get("behalfFundCode") or row.get("fundCode") or ""
        ).strip().upper()
        symbol = primary_stock or fund_code or ""

        title = subject_tr or summary_tr or member_title or "KAP Bildirimi"
        subject = subject_tr or summary_tr

        built = {
            "id": f"vyk-{idx_str}",
            "source": _vyk_source_label(disclosure_class, disclosure_type, subject_tr),
            "symbol": symbol,
            "stock_codes": stock_codes,
            "title": title,
            "subject": subject,
            "published_at": published_dt.isoformat(),
            "category": _vyk_category(disclosure_class, disclosure_type, subject_tr),
            "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx_str}",
        }
        if in_window:
            items.append(built)
        else:
            # Pencere disi. Test gateway'i eski sabit veri dondurdugunde veya
            # KAP'ta uzun suredir yeni bildirim olmadiginda akisi bos birakmamak
            # icin sonra saglikli bir fallback kovasinda tutulur.
            fallback_items.append(built)

    if not items and fallback_items:
        # Pencere icinde hic kayit yoksa, zaten detay bedelini odedigimiz en
        # yeni kayitlari kullaniciya gosteriyoruz. Her senaryoda budget disinda
        # ekstra upstream istegi yok.
        items = fallback_items

    items.sort(key=lambda node: node.get("published_at") or "", reverse=True)

    used_fallback = not any(
        _parse_kap_publish_date(node.get("published_at")) and
        _parse_kap_publish_date(node.get("published_at")) >= cutoff  # type: ignore[operator]
        for node in items
    ) and bool(items)
    # region agent log
    _debug_log(
        "H13",
        "app/api.py:_fetch_kap_vyk_feed",
        "KAP VYK feed built",
        {
            "last_index": int(last_index),
            "list_pages": pages,
            "disclosures": len(disclosures),
            "detail_budget": budget,
            "detail_hits": len(details),
            "window_hours": int(window_hours),
            "items": len(items),
            "used_fallback": used_fallback,
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    return items


# endregion


# Member OID ve member feed cache'leri — canlı feed, BIST100 per-company
# endpoint'ini kullanarak KAP'ın listeleme endpoint'i engellendiğinde dahi
# canlı veri üretir.
_KAP_MEMBER_OID_CACHE: Dict[str, str] = {}
_KAP_MEMBER_FEED_CACHE: Dict[str, Any] = {"items": [], "ts": 0.0}
_KAP_MEMBER_FEED_TTL = 600  # 10 dk


def _fetch_kap_member_disclosures_for(symbol: str, year: int) -> List[Dict[str, Any]]:
    """Fetch recent financial disclosures for a single BIST company using the
    `listCompanyExcelMembers` endpoint which is NOT WAF-blocked.

    Returns [] on any error so the aggregator can continue for other companies.
    """
    import urllib.error
    import urllib.request

    oid = _KAP_MEMBER_OID_CACHE.get(symbol)
    headers = {
        "Accept": "application/json",
        "Accept-Language": "tr",
        "User-Agent": _KAP_DEFAULT_HEADERS["User-Agent"],
    }
    if not oid:
        try:
            req = urllib.request.Request(
                f"https://www.kap.org.tr/tr/api/member/filter/{symbol}",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                rows = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(rows, list) and rows:
                oid = str(rows[0].get("mkkMemberOid") or "").strip()
                if oid:
                    _KAP_MEMBER_OID_CACHE[symbol] = oid
        except Exception:
            return []
    if not oid:
        return []
    try:
        req = urllib.request.Request(
            f"https://www.kap.org.tr/tr/api/financialTable/listCompanyExcelMembers/{oid}/{year}/T",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _synth_quarter_publish_dt(year: int, period: int) -> Optional[datetime]:
    """Approximate publish date from year+period when real publishDate absent."""
    approximate = {1: (5, 15), 2: (8, 15), 3: (11, 15), 4: (3, 15)}
    pair = approximate.get(int(period or 0))
    if not pair:
        return None
    month, day = pair
    y = int(year or 0) + (1 if int(period or 0) == 4 else 0)
    try:
        return datetime(y, month, day)
    except ValueError:
        return None


def _fetch_kap_member_feed(
    *,
    max_companies: int = 40,
    max_items: int = 160,
) -> List[Dict[str, Any]]:
    """Aggregate latest financial disclosures across BIST100 via working KAP endpoints.

    Returns feed items with best-effort `published_at` sourced from the local
    `kap_cache` (exact) when available, otherwise synthesized from year+period.
    Cached for `_KAP_MEMBER_FEED_TTL` seconds.
    """
    import concurrent.futures
    from app.kap_service import BIST100_SYMBOLS

    now = time.time()
    cache = _KAP_MEMBER_FEED_CACHE
    if cache.get("items") and (now - cache.get("ts", 0)) < _KAP_MEMBER_FEED_TTL:
        return list(cache["items"])

    started = time.time()
    symbols = list(BIST100_SYMBOLS[:max_companies])
    current_year = datetime.now(timezone.utc).year

    # Pre-load kap_cache publish dates keyed by disclosure_index for exact timestamps.
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    cache_publish: Dict[int, Dict[str, Any]] = {}
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            try:
                with cache_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            quarters = payload.get("quarters")
            if not isinstance(quarters, list):
                continue
            for q in quarters:
                if not isinstance(q, dict):
                    continue
                idx = q.get("disclosure_index")
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                cache_publish[idx_int] = {
                    "publish_date": q.get("publish_date"),
                    "quarter": q.get("quarter"),
                    "title": q.get("title"),
                }

    # Warm the OID cache from kap_cache files where available so the first
    # request doesn't pay a full `member/filter` lookup sweep.
    if not _KAP_MEMBER_OID_CACHE:
        kap_cache_dir = CONFIG.paths.processed_dir / "kap_cache"
        if kap_cache_dir.exists():
            for cache_file in kap_cache_dir.glob("*.json"):
                try:
                    with cache_file.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except Exception:
                    continue
                symbol = str(payload.get("stock_code") or cache_file.stem).strip().upper()
                oid = str(payload.get("member_oid") or "").strip()
                if symbol and oid:
                    _KAP_MEMBER_OID_CACHE[symbol] = oid

    def _gather(year_tasks: List[tuple[str, int]]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_fetch_kap_member_disclosures_for, sym, yr): (sym, yr)
                for sym, yr in year_tasks
            }
            for fut in concurrent.futures.as_completed(futures):
                sym, _ = futures[fut]
                try:
                    rows = fut.result()
                except Exception:
                    rows = []
                for row in rows:
                    row["_symbol"] = sym
                    collected.append(row)
        return collected

    current_tasks = [(sym, current_year) for sym in symbols]
    raw_rows: List[Dict[str, Any]] = _gather(current_tasks)
    # Backfill to the previous year only if current-year yield is thin.
    if len({row.get("_symbol") for row in raw_rows}) < max(8, len(symbols) // 4):
        raw_rows.extend(_gather([(sym, current_year - 1) for sym in symbols]))

    seen: set[int] = set()
    items: List[Dict[str, Any]] = []
    for row in raw_rows:
        try:
            idx = int(row.get("disclosureIndex") or 0)
        except Exception:
            continue
        if not idx or idx in seen:
            continue
        seen.add(idx)

        symbol = str(row.get("_symbol") or row.get("stockCode") or "").strip().upper()
        year = int(row.get("year") or 0)
        period = int(row.get("period") or 0)

        cache_meta = cache_publish.get(idx)
        publish_dt: Optional[datetime] = None
        quarter_label = ""
        title = ""
        if cache_meta:
            publish_dt = _parse_kap_publish_date(cache_meta.get("publish_date"))
            quarter_label = str(cache_meta.get("quarter") or "").strip()
            title = str(cache_meta.get("title") or "").strip()
        if publish_dt is None:
            publish_dt = _synth_quarter_publish_dt(year, period)
        if publish_dt is None:
            continue
        if not quarter_label and year and period:
            quarter_label = f"{year}Q{period}"
        if not title:
            title = "Finansal Rapor"

        items.append(
            {
                "id": f"{symbol}-{idx}",
                "source": "Finansal Rapor",
                "symbol": symbol,
                "stock_codes": [symbol],
                "title": f"{title}{' - ' + quarter_label if quarter_label else ''}",
                "subject": "Finansal Rapor",
                "published_at": publish_dt.isoformat(),
                "category": "finansal_rapor",
                "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx}",
            }
        )

    # Items without a kap_cache exact publish_date share the same synthesized
    # day; fall back to disclosureIndex as a tiebreaker so ordering matches
    # KAP's own allocation sequence (higher index = more recent).
    items.sort(
        key=lambda row: (
            row.get("published_at") or "",
            int(str(row.get("id") or "-0").rsplit("-", 1)[-1] or 0) if "-" in str(row.get("id") or "") else 0,
        ),
        reverse=True,
    )
    items = items[:max_items]

    # region agent log
    _debug_log(
        "H11",
        "app/api.py:_fetch_kap_member_feed",
        "KAP member feed built",
        {
            "symbol_count": len(symbols),
            "raw_rows": len(raw_rows),
            "unique_items": len(items),
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    cache["items"] = items
    cache["ts"] = now
    return list(items)


def _market_flow_payload(
    limit: int = 40,
    category: Optional[str] = None,
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    started = time.time()
    # Kullanici 'kac kayit' ayarini UI'dan degistirince backend'in VYK detay
    # butcesini de ona gore genisletmek istiyoruz; ayni zamanda cache'i bu
    # butceyle kademeli tutuyoruz ki kucuk secimle doldurulup buyukte kirilmasin.
    effective_budget = max(
        _VYK_DEFAULT_DETAIL_BUDGET,
        min(_VYK_DEFAULT_DETAIL_BUDGET_MAX, int(limit)),
    )
    effective_pages = max(1, min(10, (effective_budget + 49) // 50))
    cache_key = f"all::{category or ''}::b{effective_budget}"
    now = time.time()
    cached = _FLOW_CACHE.get(cache_key)
    if cached and not force_refresh:
        cached_data = cached["data"]
        ttl = _FLOW_DEGRADED_TTL if cached_data.get("degraded_mode") else _FLOW_CACHE_TTL
        if now - cached.get("_ts", 0) < ttl:
            return {**cached_data, "items": cached_data["items"][:limit]}

    # Resmi KAP VYK akisi: credential'lar varsa en oncelikli kaynak.
    # UYARI: Kullanici kap.org.tr uzerinden canli veri istedigi icin VYK akisi gecici olarak devre disi birakildi.
    vyk_items: List[Dict[str, Any]] = []
    # vyk_items = _fetch_kap_vyk_feed(
    #     window_hours=_VYK_DEFAULT_WINDOW_HOURS,
    #     list_pages=effective_pages,
    #     detail_budget=effective_budget,
    # )
    
    # VYK boslarsa ya da credential yoksa resmi sitesi olan kap.org.tr uzerinden deneme yap.
    public_items: List[Dict[str, Any]] = []
    local_items: List[Dict[str, Any]] = []
    if not vyk_items:
        public_items = _fetch_kap_public_disclosures(max_items=limit)
        if not public_items:
            local_items = _local_flow_items_from_cache()

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in vyk_items + public_items + local_items:
        key = item.get("id") or ""
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(item)

    if category:
        merged = [row for row in merged if row.get("category") == category]

    merged.sort(key=lambda row: row.get("published_at") or "", reverse=True)

    public_error: Optional[str] = None
    if vyk_items:
        source = "kap_vyk"
        degraded = False
        multi_category_available = True
        warning: Optional[str] = None
    elif public_items:
        source = "kap_public_website"
        degraded = False
        multi_category_available = True
        warning = None
    else:
        source = "local_cache"
        degraded = True
        multi_category_available = False
        warning = (
            "Canlı KAP akışı şu an ulaşılamıyor; yalnızca yerel önbellekteki "
            "son finansal raporlar gösteriliyor."
        )

    data = {
        "items": merged[:_VYK_DEFAULT_DETAIL_BUDGET_MAX],
        "source": source,
        "degraded_mode": degraded,
        "multi_category": multi_category_available,
        "warning": warning,
        "public_error": public_error,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _FLOW_CACHE[cache_key] = {"_ts": now, "data": data}
    # region agent log
    _debug_log(
        "H2",
        "app/api.py:_market_flow_payload",
        "Market flow payload built",
        {
            "limit": limit,
            "category": category,
            "force_refresh": force_refresh,
            "effective_budget": effective_budget,
            "vyk_items": len(vyk_items),
            "local_items": len(local_items),
            "merged_items": len(merged),
            "source": source,
            "degraded_mode": degraded,
            "multi_category": multi_category_available,
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
    # endregion
    return {**data, "items": data["items"][:limit]}


@app.get("/market/flow")
def market_flow(
    limit: int = Query(40, ge=1, le=500),
    category: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    return _market_flow_payload(limit=limit, category=category, force_refresh=refresh)


@app.get("/kap/companies")
def kap_companies() -> Dict[str, Any]:
    from app.kap_service import get_kap_companies

    indexed = _available_companies_from_chunks(CONFIG.paths.chunks_v2_file)
    return {"companies": get_kap_companies(indexed)}


@app.get("/kap/snapshot")
def kap_snapshot(
    company: str = Query(..., min_length=1),
    refresh: bool = Query(False),
    max_quarters: int = Query(10, ge=1, le=20),
) -> Dict[str, Any]:
    from app.kap_service import get_kap_snapshot, normalize_snapshot_for_frontend

    if not getattr(CONFIG, "kap", None) or not getattr(CONFIG.kap, "enabled", False):
        raise HTTPException(status_code=503, detail="KAP modülü devre dışı.")

    raw = get_kap_snapshot(
        company=company,
        cfg=CONFIG.kap,
        processed_dir=CONFIG.paths.processed_dir,
        force_refresh=refresh,
        max_quarters=max_quarters,
    )
    normalized = normalize_snapshot_for_frontend(raw)

    price_payload = _fetch_kap_price_payload(normalized.get("stock_code") or company)
    isyatirim_payload = _fetch_isyatirim_multiples(normalized.get("stock_code") or company)
    normalized["valuation"] = _build_kap_valuation_payload(
        snapshot=normalized,
        price_payload=price_payload,
        isyatirim_payload=isyatirim_payload,
    )
    return normalized


def _quarter_sort_key(quarter: Dict[str, Any]) -> tuple[int, int]:
    return int(quarter.get("year") or 0), int(quarter.get("period") or 0)


def _extract_quarter_metric(
    quarter: Dict[str, Any],
    metric_key: str,
    priority: List[str],
) -> Optional[float]:
    for field in priority:
        container = quarter.get(field)
        if not isinstance(container, dict):
            continue
        metric = container.get(metric_key)
        if isinstance(metric, dict):
            value = metric.get("value")
        else:
            value = metric
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _build_ttm_sum(quarters_asc: List[Dict[str, Any]], metric_key: str) -> Optional[float]:
    if not quarters_asc:
        return None
    tail = quarters_asc[-4:]
    required = min(4, len(quarters_asc))
    values: List[float] = []
    for quarter in tail:
        value = _extract_quarter_metric(
            quarter,
            metric_key,
            priority=["metrics_quarterly", "metrics"],
        )
        if value is None:
            continue
        values.append(value)
    if len(values) != required:
        return None
    return float(sum(values))


def _build_kap_valuation_payload(
    *,
    snapshot: Dict[str, Any],
    price_payload: Dict[str, Any],
    isyatirim_payload: Dict[str, Any],
) -> Dict[str, Any]:
    quarters_raw = snapshot.get("quarters")
    quarters = [q for q in quarters_raw if isinstance(q, dict)] if isinstance(quarters_raw, list) else []
    quarters_sorted = sorted(quarters, key=_quarter_sort_key)
    latest = quarters_sorted[-1] if quarters_sorted else None

    ttm_net_kar = _build_ttm_sum(quarters_sorted, "net_kar")
    ttm_favok = _build_ttm_sum(quarters_sorted, "favok")

    ozkaynaklar = (
        _extract_quarter_metric(latest, "ozkaynaklar", priority=["metrics", "metrics_ytd"])
        if latest
        else None
    )
    net_borc = (
        _extract_quarter_metric(latest, "net_borc", priority=["metrics", "metrics_ytd"])
        if latest
        else None
    )

    shares_outstanding = None
    share_source = None
    if latest:
        shares_outstanding = _extract_quarter_metric(
            latest,
            "odenmis_sermaye",
            priority=["metrics", "metrics_ytd"],
        )
        if shares_outstanding is not None:
            share_source = "odenmis_sermaye"
        else:
            shares_outstanding = _extract_quarter_metric(
                latest,
                "cikarilmis_sermaye",
                priority=["metrics", "metrics_ytd"],
            )
            if shares_outstanding is not None:
                share_source = "cikarilmis_sermaye"
    if shares_outstanding is not None and shares_outstanding <= 0:
        shares_outstanding = None
        share_source = None

    assumptions: List[str] = []
    share_nominal_value = None
    if shares_outstanding is not None:
        share_nominal_value = 1.0
        assumptions.append("Hisse adedi, nominal pay degeri 1 TL varsayimiyla hesaplandi.")

    price_ok = bool(price_payload.get("ok"))
    price = float(price_payload["price"]) if price_ok and isinstance(price_payload.get("price"), (int, float)) else None
    price_currency = str(price_payload.get("currency", "TRY")) if price_ok else None
    price_as_of = price_payload.get("as_of") if price_ok else None

    market_cap = price * shares_outstanding if price is not None and shares_outstanding is not None else None
    enterprise_value = market_cap + net_borc if market_cap is not None and net_borc is not None else None

    fk = _parse_tr_decimal(isyatirim_payload.get("fk")) if isyatirim_payload.get("ok") else None
    pd_dd = _parse_tr_decimal(isyatirim_payload.get("pd_dd")) if isyatirim_payload.get("ok") else None
    fd_favok = _parse_tr_decimal(isyatirim_payload.get("fd_favok")) if isyatirim_payload.get("ok") else None

    return {
        "price": price,
        "price_currency": price_currency,
        "price_as_of": price_as_of,
        "price_source": "yahoo_finance_chart",
        "shares_outstanding": shares_outstanding,
        "share_source": share_source,
        "share_nominal_value": share_nominal_value,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "ttm_net_kar": ttm_net_kar,
        "ttm_favok": ttm_favok,
        "fk": fk,
        "pd_dd": pd_dd,
        "fd_favok": fd_favok,
        "fk_prim_iskonto_pct": _parse_tr_decimal(isyatirim_payload.get("fk_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "fd_favok_prim_iskonto_pct": _parse_tr_decimal(isyatirim_payload.get("fd_favok_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "pd_dd_prim_iskonto_pct": _parse_tr_decimal(isyatirim_payload.get("pd_dd_prim_iskonto_pct"))
        if isyatirim_payload.get("ok")
        else None,
        "multiples_source": isyatirim_payload.get("source") if isyatirim_payload.get("ok") else None,
        "multiples_note": isyatirim_payload.get("note") if isyatirim_payload.get("ok") else None,
        "multiples_as_of": isyatirim_payload.get("fetched_at") if isyatirim_payload.get("ok") else None,
        "multiples_error": isyatirim_payload.get("error") if not isyatirim_payload.get("ok") else None,
        "assumptions": assumptions,
    }


# ── Yahoo Finance price endpoint ──────────────────────────
_PRICE_CACHE: Dict[str, Any] = {}
_PRICE_CACHE_TTL = 300  # 5 minutes
_MARKET_PRICE_CACHE: Dict[str, Any] = {}
_MARKET_PRICE_CACHE_TTL = 300  # 5 minutes
_ISYATIRIM_CACHE: Dict[str, Any] = {}
_ISYATIRIM_CACHE_TTL = 900  # 15 minutes


def _isyatirim_company_card_url(symbol: str) -> str:
    return f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"


def _parse_tr_decimal(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        numeric = float(raw)
        if numeric != numeric:  # NaN guard
            return None
        return numeric

    token = str(raw).strip()
    if not token:
        return None

    token = token.replace("\xa0", "").replace(" ", "")
    token = token.replace("\u2212", "-")
    token = token.replace("%", "").replace("x", "").replace("X", "")
    if token in {"-", "--", "A/D", "N/A", "n/a"}:
        return None

    number_match = re.search(r"[-+]?\d[\d\.,]*", token)
    if not number_match:
        return None
    token = number_match.group(0)

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            # Turkish style: 1.234,56
            token = token.replace(".", "").replace(",", ".")
        else:
            # English style: 1,234.56
            token = token.replace(",", "")
    elif "," in token:
        if token.count(",") > 1:
            token = token.replace(",", "")
        else:
            left, right = token.split(",", 1)
            if len(right) == 3 and left.lstrip("+-").isdigit():
                token = token.replace(",", "")
            else:
                token = token.replace(",", ".")
    elif "." in token:
        if token.count(".") > 1:
            token = token.replace(".", "")
        else:
            left, right = token.split(".", 1)
            if len(right) == 3 and len(left.lstrip("+-")) > 3:
                token = token.replace(".", "")

    try:
        numeric = float(token)
        if numeric != numeric:  # NaN guard
            return None
        return numeric
    except Exception:
        return None


def _quote_ts_to_iso(raw: Any) -> Optional[str]:
    if not isinstance(raw, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _fetch_market_price_map(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    import urllib.error
    import urllib.request

    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not normalized_symbols:
        return {}

    cache_key = ",".join(normalized_symbols)
    now = time.time()
    cached = _MARKET_PRICE_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _MARKET_PRICE_CACHE_TTL:
        return cached.get("items", {})

    url = "https://infoyatirim.com/canli-borsa/xu100-bist-100-hisseleri"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, Exception):
        return {}

    items: Dict[str, Dict[str, Any]] = {}
    try:
        row_pattern = re.compile(
            r'<tr[^>]+data-symbol="(?P<symbol>[A-Z0-9]+)"[^>]*>.*?'
            r'<td class="price" data-val="(?P<price>[^"]+)">.*?</td>.*?'
            r'<td class="change" data-val="(?P<change>[^"]+)".*?>.*?</td>.*?'
            r'<td class="percent" data-val="(?P<change_pct>[^"]+)".*?>.*?</td>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        fetched_at = datetime.now(timezone.utc).isoformat()
        for match in row_pattern.finditer(html_text):
            symbol = str(match.group("symbol") or "").strip().upper()
            if not symbol:
                continue

            items[symbol] = {
                "price": _parse_tr_decimal(match.group("price")),
                "currency": "TRY",
                "change": _parse_tr_decimal(match.group("change")),
                "change_pct": _parse_tr_decimal(match.group("change_pct")),
                "market_state": "",
                "as_of": fetched_at,
            }
    except Exception:
        return {}

    _MARKET_PRICE_CACHE[cache_key] = {"_ts": now, "items": items}
    return items


def _strip_html_cell(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(raw or ""), flags=re.IGNORECASE)
    text = html.unescape(text)
    return " ".join(text.split())


def _norm_text(value: str) -> str:
    return (
        str(value or "")
        .upper()
        .replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )


def _extract_isyatirim_historical_averages(html_text: str, symbol: str) -> Dict[str, Any]:
    table_match = re.search(
        r'<table[^>]+data-csvname="tarihselortalamalar"[^>]*>(.*?)</table>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not table_match:
        return {"ok": False, "error": "İş Yatırım 'Tarihsel Ortalamalar' tablosu bulunamadı."}

    table_html = table_match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
    if not rows:
        return {"ok": False, "error": "İş Yatırım tablosunda satır bulunamadı."}

    header_cells: Optional[List[str]] = None
    selected_cells: Optional[List[str]] = None
    symbol_upper = str(symbol or "").strip().upper()
    for row_html in rows:
        header_raw = re.findall(r"<th[^>]*>(.*?)</th>", row_html, flags=re.IGNORECASE | re.DOTALL)
        if header_raw:
            header_cells = [_strip_html_cell(cell) for cell in header_raw]
            continue

        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)
        clean_cells = [_strip_html_cell(cell) for cell in cells]
        if not clean_cells:
            continue
        row_symbol = clean_cells[0].upper() if clean_cells else ""
        if row_symbol == symbol_upper:
            selected_cells = clean_cells
            break
        if selected_cells is None and len(clean_cells) >= 3:
            selected_cells = clean_cells

    if not selected_cells or len(selected_cells) < 3:
        return {"ok": False, "error": "İş Yatırım tablosundan çarpan verisi ayrıştırılamadı."}

    index_fk = None
    index_fd_favok = None
    index_pd_dd = None
    index_fk_prim = None
    index_fd_favok_prim = None
    index_pd_dd_prim = None

    if header_cells:
        for idx, header in enumerate(header_cells):
            norm = _norm_text(str(header))
            if "F/K" in norm and "TAHMIN" in norm:
                index_fk = idx
            elif ("FD/FAVOK" in norm or "FD/FAVÖK" in norm) and "TAHMIN" in norm:
                index_fd_favok = idx
            elif "PD/DD" in norm and "TAHMIN" in norm:
                index_pd_dd = idx
        if index_fk is not None and index_fk + 1 < len(header_cells) and "PRIM" in _norm_text(header_cells[index_fk + 1]):
            index_fk_prim = index_fk + 1
        if (
            index_fd_favok is not None
            and index_fd_favok + 1 < len(header_cells)
            and "PRIM" in _norm_text(header_cells[index_fd_favok + 1])
        ):
            index_fd_favok_prim = index_fd_favok + 1
        if index_pd_dd is not None and index_pd_dd + 1 < len(header_cells) and "PRIM" in _norm_text(header_cells[index_pd_dd + 1]):
            index_pd_dd_prim = index_pd_dd + 1

    if index_fk is None:
        index_fk = 1 if len(selected_cells) > 1 else None
    if index_fk_prim is None and index_fk is not None and len(selected_cells) > index_fk + 1:
        index_fk_prim = index_fk + 1

    if index_fd_favok is None and len(selected_cells) >= 7:
        # Non-bank layout fallback: KOD, F/K, Prim, FD/FAVOK, Prim, PD/DD, Prim
        index_fd_favok = 3
    if index_fd_favok_prim is None and index_fd_favok is not None and len(selected_cells) > index_fd_favok + 1:
        index_fd_favok_prim = index_fd_favok + 1

    if index_pd_dd is None:
        if len(selected_cells) >= 7:
            index_pd_dd = 5
        elif len(selected_cells) >= 5:
            # Bank/insurance layout: KOD, F/K, Prim, PD/DD, Prim
            index_pd_dd = 3
        else:
            index_pd_dd = None
    if index_pd_dd_prim is None and index_pd_dd is not None and len(selected_cells) > index_pd_dd + 1:
        index_pd_dd_prim = index_pd_dd + 1

    fk = _parse_tr_decimal(selected_cells[index_fk]) if index_fk is not None and len(selected_cells) > index_fk else None
    fd_favok = (
        _parse_tr_decimal(selected_cells[index_fd_favok])
        if index_fd_favok is not None and len(selected_cells) > index_fd_favok
        else None
    )
    pd_dd = _parse_tr_decimal(selected_cells[index_pd_dd]) if index_pd_dd is not None and len(selected_cells) > index_pd_dd else None

    fk_prim_isk = (
        _parse_tr_decimal(selected_cells[index_fk_prim])
        if index_fk_prim is not None and len(selected_cells) > index_fk_prim
        else None
    )
    fd_favok_prim_isk = (
        _parse_tr_decimal(selected_cells[index_fd_favok_prim])
        if index_fd_favok_prim is not None and len(selected_cells) > index_fd_favok_prim
        else None
    )
    pd_dd_prim_isk = (
        _parse_tr_decimal(selected_cells[index_pd_dd_prim])
        if index_pd_dd_prim is not None and len(selected_cells) > index_pd_dd_prim
        else None
    )

    note_match = re.search(
        r'<div[^>]+class="table-note"[^>]*>(.*?)</div>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    note = _strip_html_cell(note_match.group(1)) if note_match else None

    return {
        "ok": True,
        "fk": fk,
        "fd_favok": fd_favok,
        "pd_dd": pd_dd,
        "fk_prim_iskonto_pct": fk_prim_isk,
        "fd_favok_prim_iskonto_pct": fd_favok_prim_isk,
        "pd_dd_prim_iskonto_pct": pd_dd_prim_isk,
        "note": note,
    }


def _fetch_isyatirim_multiples(symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {
            "ok": False,
            "symbol": "",
            "error": "Sembol bos.",
        }

    cache_key = ticker
    now = time.time()
    cached = _ISYATIRIM_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _ISYATIRIM_CACHE_TTL:
        return cached

    url = _isyatirim_company_card_url(ticker)
    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, Exception) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "source": "isyatirim_company_card",
            "url": url,
            "error": f"İş Yatırım bağlantı hatası: {exc}",
        }

    parsed = _extract_isyatirim_historical_averages(html_text, ticker)
    payload: Dict[str, Any] = {
        "ok": bool(parsed.get("ok")),
        "symbol": ticker,
        "source": "isyatirim_company_card",
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "_ts": now,
    }
    payload.update(parsed)
    _ISYATIRIM_CACHE[cache_key] = payload
    return payload


def _fetch_kap_price_payload(symbol: str) -> Dict[str, Any]:
    """Fetch latest stock price from Yahoo Finance for a BIST ticker."""
    import urllib.request
    import urllib.error

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {
            "ok": False,
            "symbol": "",
            "error": "Sembol bos.",
        }
    cache_key = ticker
    now = time.time()

    # Return cached if fresh
    cached = _PRICE_CACHE.get(cache_key)
    if cached and now - cached["_ts"] < _PRICE_CACHE_TTL:
        return cached

    yahoo_symbol = f"{ticker}.IS"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval=1d&range=1d"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "error": f"Yahoo Finance bağlantı hatası: {exc}",
        }

    try:
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        currency = meta.get("currency", "TRY")
        market_state = meta.get("marketState", "")
        regular_market_time = meta.get("regularMarketTime")

        change = None
        change_pct = None
        if price is not None and prev_close:
            change = round(price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2)
        as_of = None
        if isinstance(regular_market_time, (int, float)):
            try:
                as_of = datetime.fromtimestamp(float(regular_market_time), tz=timezone.utc).isoformat()
            except Exception:
                as_of = None

        result: Dict[str, Any] = {
            "ok": True,
            "symbol": ticker,
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "currency": currency,
            "market_state": market_state,
            "as_of": as_of,
            "_ts": now,
        }
        _PRICE_CACHE[cache_key] = result
        return result
    except (KeyError, IndexError, TypeError) as exc:
        return {
            "ok": False,
            "symbol": ticker,
            "error": f"Yahoo Finance veri parse hatası: {exc}",
        }


@app.get("/kap/price")
def kap_price(symbol: str = Query(..., min_length=1)) -> Dict[str, Any]:
    return _fetch_kap_price_payload(symbol)


# ── XU030 universe ────────────────────────────────────────
_XU030_CACHE: Dict[str, Any] = {}
_XU030_CACHE_TTL = 120  # 2 minutes


def _fill_prices_via_yahoo(symbols: List[str], base_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """For symbols missing a price in base_map, query Yahoo Finance individually."""
    result = dict(base_map)
    missing = [s for s in symbols if not (result.get(s) or {}).get("price")]
    if not missing:
        return result

    from concurrent.futures import ThreadPoolExecutor

    def _one(sym: str) -> tuple[str, Dict[str, Any]]:
        payload = _fetch_kap_price_payload(sym)
        if not payload.get("ok"):
            return sym, {}
        return sym, {
            "price": payload.get("price"),
            "currency": payload.get("currency") or "TRY",
            "change": payload.get("change"),
            "change_pct": payload.get("change_pct"),
            "market_state": payload.get("market_state") or "",
            "as_of": payload.get("as_of"),
        }

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for sym, quote in pool.map(_one, missing):
                if quote:
                    result[sym] = quote
    except Exception:
        for sym in missing:
            s, quote = _one(sym)
            if quote:
                result[s] = quote
    return result


def _xu030_payload() -> Dict[str, Any]:
    from app.kap_service import get_bist30_companies

    now = time.time()
    cached = _XU030_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _XU030_CACHE_TTL:
        return cached["data"]

    symbols = get_bist30_companies()
    base_map = _fetch_market_price_map(symbols)
    price_map = _fill_prices_via_yahoo(symbols, base_map)

    breakdown_rows = _company_breakdown_from_chunks(CONFIG.paths.chunks_v2_file)
    breakdown_map = {
        str(row.get("company") or "").strip().upper(): row
        for row in breakdown_rows
        if str(row.get("company") or "").strip()
    }
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"

    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        breakdown_row = breakdown_map.get(symbol, {})
        cached_meta = _load_cached_kap_market_metadata(cache_dir, symbol)
        quarters = [
            str(item or "").strip().upper()
            for item in (breakdown_row.get("quarters") or [])
            if str(item or "").strip()
        ]
        latest_quarter = _latest_quarter_label(
            quarters + ([cached_meta["latest_quarter"]] if cached_meta.get("latest_quarter") else [])
        )
        quote = price_map.get(symbol, {})
        rows.append(
            {
                "company": symbol,
                "chunks": int(breakdown_row.get("chunks") or 0),
                "quarter_count": int(breakdown_row.get("quarter_count") or 0),
                "latest_quarter": latest_quarter,
                "has_rag": bool(breakdown_row),
                "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
                "price": quote.get("price"),
                "price_currency": quote.get("currency"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "price_as_of": quote.get("as_of"),
            }
        )

    data = {"index": "XU030", "rows": rows, "as_of": datetime.now(timezone.utc).isoformat()}
    _XU030_CACHE["payload"] = {"_ts": now, "data": data}
    return data


@app.get("/market/xu030")
def market_xu030() -> Dict[str, Any]:
    return _xu030_payload()


# ── Commodities (Yahoo-backed, provider-delayed) ──────────
_COMMODITY_CACHE: Dict[str, Any] = {}
_COMMODITY_CACHE_TTL = 90  # 90 seconds

# Display symbol -> (Yahoo ticker, Turkish label, override currency)
_COMMODITY_MAP: List[tuple[str, str, str, Optional[str]]] = [
    ("BRENT", "BZ=F", "Brent Petrol", "USD"),
    ("WTI", "CL=F", "WTI Ham Petrol", "USD"),
    ("DOGALGAZ", "NG=F", "Doğal Gaz", "USD"),
    ("ALTIN", "GC=F", "Altın (Ons)", "USD"),
    ("GUMUS", "SI=F", "Gümüş (Ons)", "USD"),
    ("BAKIR", "HG=F", "Bakır", "USD"),
    ("PLATIN", "PL=F", "Platin", "USD"),
    ("PALADYUM", "PA=F", "Paladyum", "USD"),
]


def _fetch_yahoo_quote(yahoo_symbol: str) -> Dict[str, Any]:
    """Low-level Yahoo chart fetch for an arbitrary ticker (no cache)."""
    import urllib.error
    import urllib.request

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval=1d&range=1d"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {"ok": False, "error": f"yahoo_error: {exc}"}

    try:
        meta = data["chart"]["result"][0]["meta"]
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "error": f"yahoo_parse: {exc}"}

    price = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    currency = meta.get("currency")
    market_state = meta.get("marketState", "")
    rmt = meta.get("regularMarketTime")

    change = None
    change_pct = None
    if price is not None and prev_close:
        try:
            change = round(float(price) - float(prev_close), 4)
            change_pct = round((change / float(prev_close)) * 100, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            change = None
            change_pct = None

    as_of = None
    if isinstance(rmt, (int, float)):
        try:
            as_of = datetime.fromtimestamp(float(rmt), tz=timezone.utc).isoformat()
        except Exception:
            as_of = None

    return {
        "ok": True,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": currency,
        "market_state": market_state,
        "as_of": as_of,
    }


def _market_commodities_payload() -> Dict[str, Any]:
    now = time.time()
    cached = _COMMODITY_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _COMMODITY_CACHE_TTL:
        return cached["data"]

    from concurrent.futures import ThreadPoolExecutor

    def _one(entry: tuple[str, str, str, Optional[str]]) -> Dict[str, Any]:
        symbol, yahoo_symbol, label, forced_currency = entry
        quote = _fetch_yahoo_quote(yahoo_symbol)
        return {
            "symbol": symbol,
            "label": label,
            "yahoo_symbol": yahoo_symbol,
            "price": quote.get("price") if quote.get("ok") else None,
            "prev_close": quote.get("prev_close") if quote.get("ok") else None,
            "change": quote.get("change") if quote.get("ok") else None,
            "change_pct": quote.get("change_pct") if quote.get("ok") else None,
            "currency": forced_currency or quote.get("currency") or "USD",
            "market_state": quote.get("market_state") if quote.get("ok") else "",
            "as_of": quote.get("as_of") if quote.get("ok") else None,
            "error": None if quote.get("ok") else quote.get("error"),
        }

    items: List[Dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for row in pool.map(_one, _COMMODITY_MAP):
                items.append(row)
    except Exception:
        for entry in _COMMODITY_MAP:
            items.append(_one(entry))

    data = {
        "items": items,
        "source": "yahoo_finance_chart",
        "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _COMMODITY_CACHE["payload"] = {"_ts": now, "data": data}
    return data


@app.get("/market/commodities")
def market_commodities() -> Dict[str, Any]:
    return _market_commodities_payload()


# ── FX (Döviz) ────────────────────────────────────────────
_FX_CACHE: Dict[str, Any] = {}
_FX_CACHE_TTL = 90

_FX_MAP: List[tuple[str, str, str]] = [
    ("USD/TRY", "USDTRY=X", "Amerikan Doları"),
    ("EUR/TRY", "EURTRY=X", "Euro"),
    ("GBP/TRY", "GBPTRY=X", "İngiliz Sterlini"),
    ("EUR/USD", "EURUSD=X", "Euro / Dolar"),
    ("DXY", "DX-Y.NYB", "Dolar Endeksi"),
]


def _market_fx_payload() -> Dict[str, Any]:
    now = time.time()
    cached = _FX_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _FX_CACHE_TTL:
        return cached["data"]

    from concurrent.futures import ThreadPoolExecutor

    def _one(entry: tuple[str, str, str]) -> Dict[str, Any]:
        symbol, yahoo_symbol, label = entry
        quote = _fetch_yahoo_quote(yahoo_symbol)
        currency = "TRY" if symbol.endswith("/TRY") else ("USD" if symbol.endswith("/USD") else "")
        return {
            "symbol": symbol,
            "label": label,
            "yahoo_symbol": yahoo_symbol,
            "price": quote.get("price") if quote.get("ok") else None,
            "prev_close": quote.get("prev_close") if quote.get("ok") else None,
            "change": quote.get("change") if quote.get("ok") else None,
            "change_pct": quote.get("change_pct") if quote.get("ok") else None,
            "currency": currency,
            "market_state": quote.get("market_state") if quote.get("ok") else "",
            "as_of": quote.get("as_of") if quote.get("ok") else None,
            "error": None if quote.get("ok") else quote.get("error"),
        }

    items: List[Dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for row in pool.map(_one, _FX_MAP):
                items.append(row)
    except Exception:
        for entry in _FX_MAP:
            items.append(_one(entry))

    data = {
        "items": items,
        "source": "yahoo_finance_chart",
        "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _FX_CACHE["payload"] = {"_ts": now, "data": data}
    return data


@app.get("/market/fx")
def market_fx() -> Dict[str, Any]:
    return _market_fx_payload()


# ── Market watch strip (single endpoint for Markets page) ────────────────
_WATCH_CACHE: Dict[str, Any] = {}
_WATCH_CACHE_TTL = 60
_WATCH_DELAY_NOTE = "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk)."

_WATCH_INDEX_CANDIDATES: List[tuple[str, str, List[str]]] = [
    ("XU100", "BIST 100", ["XU100.IS", "^XU100", "XU100"]),
    ("XU030", "BIST 30", ["XU030.IS", "^XU030", "XU030"]),
]

_WATCH_FX_SYMBOLS: List[str] = ["USD/TRY", "EUR/TRY"]
_WATCH_FX_LABELS: Dict[str, str] = {
    "USD/TRY": "Amerikan Doları",
    "EUR/TRY": "Euro",
}

_WATCH_COMMODITY_SYMBOLS: List[str] = ["BRENT", "ALTIN", "GUMUS", "DOGALGAZ"]
_WATCH_COMMODITY_LABELS: Dict[str, str] = {
    "BRENT": "Brent Petrol",
    "ALTIN": "Altın (Ons)",
    "GUMUS": "Gümüş (Ons)",
    "DOGALGAZ": "Doğal Gaz",
}


def _empty_watch_item(
    symbol: str,
    label: str,
    *,
    currency: str = "",
    error: Optional[str] = None,
    yahoo_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": yahoo_symbol,
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "currency": currency,
        "market_state": "",
        "as_of": None,
        "error": error,
    }


def _normalize_watch_item(item: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(item.get("symbol") or "").strip()
    label = str(item.get("label") or symbol).strip() or symbol
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": item.get("yahoo_symbol"),
        "price": item.get("price"),
        "prev_close": item.get("prev_close"),
        "change": item.get("change"),
        "change_pct": item.get("change_pct"),
        "currency": item.get("currency") or "",
        "market_state": item.get("market_state") or "",
        "as_of": item.get("as_of"),
        "error": item.get("error"),
    }


def _pick_watch_items(
    items: List[Dict[str, Any]],
    symbols: List[str],
    fallback_labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    mapped = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in items
        if str(row.get("symbol") or "").strip()
    }
    selected: List[Dict[str, Any]] = []
    for symbol in symbols:
        row = mapped.get(symbol.upper())
        if row:
            selected.append(_normalize_watch_item(row))
            continue
        selected.append(
            _empty_watch_item(
                symbol=symbol,
                label=fallback_labels.get(symbol, symbol),
                error="instrument_not_found",
            )
        )
    return selected


def _watch_index_item(symbol: str, label: str, yahoo_candidates: List[str]) -> Dict[str, Any]:
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        quote = _fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            return {
                "symbol": symbol,
                "label": label,
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price"),
                "prev_close": quote.get("prev_close"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "currency": quote.get("currency") or "TRY",
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
            }
        err = str(quote.get("error") or "quote_unavailable")
        errors.append(f"{yahoo_symbol}:{err}")

    return _empty_watch_item(
        symbol=symbol,
        label=label,
        currency="TRY",
        error="; ".join(errors[:3]) if errors else "quote_unavailable",
    )


def _market_watch_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    now = time.time()
    cached = _WATCH_CACHE.get("payload")
    if cached and not force_refresh and now - cached.get("_ts", 0) < _WATCH_CACHE_TTL:
        return cached["data"]

    fx_payload = _market_fx_payload()
    commodity_payload = _market_commodities_payload()

    indices = [
        _watch_index_item(symbol=symbol, label=label, yahoo_candidates=yahoo_candidates)
        for symbol, label, yahoo_candidates in _WATCH_INDEX_CANDIDATES
    ]
    fx_items = _pick_watch_items(
        items=list(fx_payload.get("items") or []),
        symbols=_WATCH_FX_SYMBOLS,
        fallback_labels=_WATCH_FX_LABELS,
    )
    commodity_items = _pick_watch_items(
        items=list(commodity_payload.get("items") or []),
        symbols=_WATCH_COMMODITY_SYMBOLS,
        fallback_labels=_WATCH_COMMODITY_LABELS,
    )

    data = {
        "sections": {
            "indices": indices,
            "fx": fx_items,
            "commodities": commodity_items,
        },
        "source": "yahoo_finance_chart",
        "delay_note": _WATCH_DELAY_NOTE,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _WATCH_CACHE["payload"] = {"_ts": now, "data": data}
    return data


@app.get("/market/watch")
def market_watch(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_watch_payload(force_refresh=refresh)
