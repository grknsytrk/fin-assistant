from __future__ import annotations

import asyncio
import io
import html
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pandas import isna
from pydantic import BaseModel, Field

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise RuntimeError(
        "RAG-Fin backend requires Python 3.10-3.12. "
        f"Current interpreter is Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Use .external\\venv311\\Scripts\\python.exe or run .\\run.ps1."
    )

from src.answer import AnswerEngine, RulesBasedAnswerAdapter
from src.config import AppConfig, load_config
from src.commentary import SAFE_EMPTY_COMMENTARY, generate_commentary
from src.index import build_index, build_index_v2
from src.ingest import ingest_raw_pdfs, list_pdf_files
from src.nvidia_commentary import MAX_REQUEST_BYTES, PayloadValidationError, generate_overview_commentary
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
from app.reference_data import (
    get_instrument,
    get_instrument_name,
    sync_reference_data_from_caches,
    upsert_instrument,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")
FEEDBACK_FILE = CONFIG.paths.processed_dir / "feedback.jsonl"
LOGGER = logging.getLogger("uvicorn.error")
_FUND_COLLECTOR_TASK: Optional[asyncio.Task[None]] = None


def _truthy_env(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


async def _fund_price_collector_loop() -> None:
    from app.fund_service import collect_daily_fund_prices

    startup_delay = float(os.getenv("RAGFIN_FUND_COLLECTOR_STARTUP_DELAY_SECONDS", "60"))
    interval = float(os.getenv("RAGFIN_FUND_COLLECTOR_INTERVAL_SECONDS", str(24 * 60 * 60)))
    if startup_delay > 0:
        await asyncio.sleep(startup_delay)
    while True:
        try:
            result = await asyncio.to_thread(collect_daily_fund_prices, CONFIG.paths.processed_dir)
            LOGGER.info(
                "fund price collector completed: valid=%s skipped=%s source=%s",
                result.get("valid_point_count"),
                result.get("skipped_point_count"),
                result.get("source"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("fund price collector failed")
        await asyncio.sleep(max(60.0, interval))


async def _start_fund_price_collector() -> None:
    global _FUND_COLLECTOR_TASK
    if not _truthy_env("RAGFIN_FUND_COLLECTOR_ENABLED", "1"):
        return
    if _FUND_COLLECTOR_TASK and not _FUND_COLLECTOR_TASK.done():
        return
    _FUND_COLLECTOR_TASK = asyncio.create_task(_fund_price_collector_loop())


async def _stop_fund_price_collector() -> None:
    global _FUND_COLLECTOR_TASK
    task = _FUND_COLLECTOR_TASK
    _FUND_COLLECTOR_TASK = None
    if not task:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        await asyncio.to_thread(sync_reference_data_from_caches, CONFIG.paths.processed_dir)
    except Exception:
        LOGGER.debug("reference data bootstrap failed", exc_info=True)
    await _start_fund_price_collector()
    try:
        yield
    finally:
        await _stop_fund_price_collector()


app = FastAPI(title="RAG-Fin API", version="0.10.0", lifespan=_lifespan)

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


class MarketComparisonHistoryAsset(BaseModel):
    id: Optional[str] = None
    kind: Literal["fund", "stock", "index", "fx"]
    symbol: str = Field(..., min_length=1)
    label: Optional[str] = None


class MarketComparisonHistoryRequest(BaseModel):
    assets: List[MarketComparisonHistoryAsset] = Field(..., min_length=1, max_length=8)
    start_date: date
    end_date: date


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


_AVAILABLE_COMPANIES_CACHE: Dict[str, Any] = {}


def _available_companies_from_chunks(chunks_file: Path) -> List[str]:
    if not chunks_file.exists():
        return []
    try:
        stat = chunks_file.stat()
        cache_key = str(chunks_file.resolve())
        cached = _AVAILABLE_COMPANIES_CACHE.get(cache_key)
        signature = (stat.st_mtime_ns, stat.st_size)
        if cached and cached.get("signature") == signature:
            return list(cached.get("companies") or [])
    except Exception:
        cache_key = ""
        signature = None

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
    result = sorted(companies)
    if cache_key and signature:
        _AVAILABLE_COMPANIES_CACHE[cache_key] = {
            "signature": signature,
            "companies": result,
        }
    return result


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

    company_title = str(payload.get("company_title") or "").strip()
    company_code = str(payload.get("company") or payload.get("stock_code") or symbol).strip().upper()

    quarters_raw = payload.get("quarters")
    quarters = [
        str(row.get("quarter") or "").strip().upper()
        for row in (quarters_raw or [])
        if isinstance(row, dict) and str(row.get("quarter") or "").strip()
    ]
    quarter_rows = [row for row in (quarters_raw or []) if isinstance(row, dict)]
    latest_row = max(
        quarter_rows,
        key=lambda row: _quarter_label_sort_key(str(row.get("quarter") or "").strip().upper()),
        default=None,
    )
    shares_outstanding = None
    share_source = None
    if latest_row:
        for metric_key in ("odenmis_sermaye", "cikarilmis_sermaye"):
            for field in ("metrics", "metrics_ytd"):
                container = latest_row.get(field)
                if not isinstance(container, dict):
                    continue
                metric = container.get(metric_key)
                value = metric.get("value") if isinstance(metric, dict) else metric
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                    continue
                shares_outstanding = float(value)
                share_source = metric_key
                break
            if shares_outstanding is not None:
                break
    return {
        "latest_quarter": _latest_quarter_label(quarters),
        "has_kap_cache": True,
        "shares_outstanding": shares_outstanding,
        "share_source": share_source,
        "company_title": company_title or None,
        "company": company_code or symbol,
    }


def _stock_reference_record_from_kap_payload(symbol: str, payload: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    normalized = str(symbol or "").strip().upper()
    stock_code = str(payload.get("stock_code") or payload.get("company") or normalized).strip().upper()
    title = str(payload.get("company_title") or payload.get("title") or payload.get("companyName") or "").strip()
    member_oid = str(payload.get("member_oid") or payload.get("mkk_member_oid") or "").strip()
    return {
        "kind": "stock",
        "symbol": stock_code or normalized,
        "name": title or None,
        "short_name": stock_code or normalized,
        "source": source,
        "source_id": member_oid or None,
        "logo_url": f"https://www.kap.org.tr/tr/api/member/logo/{member_oid}" if member_oid else None,
        "logo_source": "kap" if member_oid else None,
        "as_of": str(payload.get("fetched_at") or "").strip() or None,
        "aliases": [normalized] if normalized and normalized != stock_code else [],
        "metadata": {
            "latest_quarter": _latest_quarter_label(
                [
                    str(row.get("quarter") or "").strip().upper()
                    for row in (payload.get("quarters") or [])
                    if isinstance(row, dict)
                ]
            ),
            "source_url": payload.get("source_url"),
        },
    }


def _upsert_stock_reference_from_kap_payload(symbol: str, payload: Dict[str, Any], *, source: str = "kap") -> None:
    if not isinstance(payload, dict) or not payload.get("ok", True):
        return
    record = _stock_reference_record_from_kap_payload(symbol, payload, source=source)
    if not record.get("symbol"):
        return
    try:
        upsert_instrument(CONFIG.paths.processed_dir, **record)
    except Exception:
        LOGGER.debug("stock reference upsert failed for %s", symbol, exc_info=True)


def _positive_float(raw: Any) -> Optional[float]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value <= 0:
        return None
    return value


def _market_cap_from_quote_and_meta(
    quote: Dict[str, Any],
    cached_meta: Dict[str, Any],
    basic_summary: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    price = quote.get("price")
    shares = cached_meta.get("shares_outstanding")
    price_value = _positive_float(price)
    shares_value = _positive_float(shares)
    if price_value is not None and shares_value is not None:
        return price_value * shares_value

    quote_market_cap = _positive_float(quote.get("market_cap"))
    if quote_market_cap is not None:
        return quote_market_cap

    summary = basic_summary or {}
    market_cap = _positive_float(summary.get("market_cap"))
    if market_cap is not None:
        return market_cap

    summary_shares = _positive_float(summary.get("shares_outstanding"))
    if price_value is not None and summary_shares is not None:
        return price_value * summary_shares
    return None


_UNIVERSE_CACHE: Dict[str, Any] = {}
_UNIVERSE_CACHE_TTL = 120  # 2 minutes


def _market_universe_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    from app.kap_service import get_bist_index_universe

    now_ts = time.time()
    normalized_index = _normalize_stock_index(index_name)
    cache_key = f"payload:{normalized_index}"
    cached = _UNIVERSE_CACHE.get(cache_key)
    if cached and not force_refresh and now_ts - cached.get("_ts", 0) < _UNIVERSE_CACHE_TTL:
        return cached["data"]

    stats = _stats_payload()
    universe = get_bist_index_universe(normalized_index, force_refresh=force_refresh)
    symbols = list(universe.get("symbols") or [])
    try:
        bist_all_count = (
            int(universe.get("count") or 0)
            if normalized_index == "XUTUM"
            else int(get_bist_index_universe("XUTUM").get("count") or 0)
        )
    except Exception:
        bist_all_count = int(universe.get("count") or len(symbols)) if normalized_index == "XUTUM" else 0
    breakdown_rows = _company_breakdown_from_chunks(CONFIG.paths.chunks_v2_file)
    breakdown_map = {
        str(row.get("company") or "").strip().upper(): row
        for row in breakdown_rows
        if str(row.get("company") or "").strip()
    }
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    base_price_map = _fetch_market_price_map(symbols, index_name=normalized_index)
    price_map = (
        base_price_map
        if normalized_index == "XUTUM"
        else _fill_prices_via_yahoo(symbols, base_price_map)
    )
    basic_summary_map = _fetch_isyatirim_basic_summary_map()
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
                "market_cap": _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary_map.get(symbol)),
                **_empty_logo_payload(),
            }
        )

    coverage_rows = sorted(
        [row for row in rows if row["has_rag"]],
        key=lambda item: (-int(item["quarter_count"]), -int(item["chunks"]), str(item["company"])),
    )[:8]

    data = {
        "stats": {
            "index": normalized_index,
            "index_count": len(rows),
            "bist100_count": len(rows),
            "bist_all_count": bist_all_count,
            "rag_ready_count": rag_ready_count,
            "kap_only_count": len(rows) - rag_ready_count,
            "kap_cache_count": kap_cache_count,
            "pdf_count": int(stats.get("pdf_count") or 0),
            "page_count": int(stats.get("page_count") or 0),
        },
        "universe": {
            "index": universe.get("index") or normalized_index,
            "count": int(universe.get("count") or len(rows)),
            "source": universe.get("source"),
            "source_url": universe.get("source_url"),
            "source_date": universe.get("source_date"),
            "fetched_at": universe.get("fetched_at"),
            "cache_hit": bool(universe.get("cache_hit")),
            "fallback_used": bool(universe.get("fallback_used")),
        },
        "rows": rows,
        "coverage_rows": coverage_rows,
    }
    _UNIVERSE_CACHE[cache_key] = {"_ts": now_ts, "data": data}
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
def market_universe(index: str = Query("XUTUM"), refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_universe_payload(index_name=index, force_refresh=refresh)


@app.get("/market/stocks")
def market_stocks(index: str = Query("XUTUM"), refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_stocks_payload(index_name=index, force_refresh=refresh)


@app.get("/market/stocks/cards")
def market_stock_cards(symbols: str = Query(""), refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_stock_cards_payload(symbols=symbols, force_refresh=refresh)


@app.get("/market/stocks/cards/chart")
def market_stock_card_chart(
    symbol: str = Query(""),
    range: str = Query("1d"),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    return _market_stock_card_chart_payload(symbol=symbol, chart_range=range, force_refresh=refresh)


@app.post("/market/comparison-history")
def market_comparison_history(request: MarketComparisonHistoryRequest) -> Dict[str, Any]:
    if request.start_date > request.end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    return _market_comparison_history_payload(request)


@app.get("/market/indices")
def market_indices(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_indices_payload(force_refresh=refresh)


@app.get("/market/indices/{index_code}")
def market_index_detail(index_code: str, refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_index_detail_payload(index_code, force_refresh=refresh)


@app.get("/funds")
def funds(
    q: Optional[str] = Query(None),
    fund_type: Optional[str] = Query(None),
    founder: Optional[str] = Query(None),
    manager: Optional[str] = Query(None),
    risk: Optional[str] = Query(None),
    sort: str = Query("fund_code"),
    order: str = Query("asc"),
) -> Dict[str, Any]:
    from app.fund_service import get_funds_payload

    return get_funds_payload(
        CONFIG.paths.processed_dir,
        q=q,
        fund_type=fund_type,
        founder=founder,
        manager=manager,
        risk=risk,
        sort=sort,
        order=order,
        auto_refresh=True,
    )


@app.get("/funds/search")
def funds_search(q: str = Query("", min_length=0), limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    from app.fund_service import get_funds_payload

    payload = get_funds_payload(
        CONFIG.paths.processed_dir,
        q=q,
        sort="fund_code",
        order="asc",
        min_aum=None,
        auto_refresh=True,
    )
    payload["rows"] = list(payload.get("rows") or [])[:limit]
    payload["count"] = len(payload["rows"])
    return payload


@app.get("/funds/categories")
def funds_categories() -> Dict[str, Any]:
    from app.fund_service import get_fund_categories_payload

    return get_fund_categories_payload(CONFIG.paths.processed_dir)


@app.get("/funds/{fund_code}/performance")
def fund_performance(
    fund_code: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    fallback: bool = Query(False),
) -> Dict[str, Any]:
    from app.fund_service import get_fund_performance_payload

    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    return get_fund_performance_payload(
        CONFIG.paths.processed_dir,
        fund_code,
        start_date=start_date,
        end_date=end_date,
        allow_upstream_fallback=fallback,
    )


@app.get("/funds/{fund_code}/yield-summary")
def fund_yield_summary(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import FintablesUpstreamError, get_fund_yield_summary_payload, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        return get_fund_yield_summary_payload(normalized, processed_dir=CONFIG.paths.processed_dir)
    except FintablesUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/funds/{fund_code}/holdings")
def fund_holdings(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_holdings_payload

    payload = get_fund_holdings_payload(CONFIG.paths.processed_dir, fund_code)
    return _enrich_fund_holdings_with_daily_market_data(payload, fund_code)


def _api_number(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    try:
        return _parse_tr_decimal(raw)
    except Exception:
        return None


def _fund_snapshot_row_map() -> Dict[str, Dict[str, Any]]:
    rows, _meta = _fund_snapshot_row_map_with_meta()
    return rows


_FUND_SNAPSHOT_ROW_MAP_CACHE: Dict[str, Any] = {}


def _fund_snapshot_row_map_with_meta() -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    from app.fund_service import load_funds_snapshot, normalize_fund_code

    snapshot_path = CONFIG.paths.processed_dir / "funds_cache" / "funds_latest.json"
    stat = snapshot_path.stat() if snapshot_path.exists() else None
    cache_key = str(snapshot_path)
    cached = _FUND_SNAPSHOT_ROW_MAP_CACHE.get(cache_key)
    if cached and stat and cached.get("mtime") == stat.st_mtime:
        return dict(cached.get("rows") or {}), {
            "cache_hit": True,
            "row_count": cached.get("row_count", 0),
            "as_of": cached.get("as_of"),
        }

    try:
        snapshot = load_funds_snapshot(CONFIG.paths.processed_dir)
    except Exception:
        return {}, {"cache_hit": False, "row_count": 0, "error": "snapshot_unavailable"}
    rows: Dict[str, Dict[str, Any]] = {}
    for row in list(snapshot.get("rows") or []):
        if not isinstance(row, dict):
            continue
        code = normalize_fund_code(str(row.get("fund_code") or row.get("fonKodu") or ""))
        if code:
            rows[code] = row
    if stat:
        _FUND_SNAPSHOT_ROW_MAP_CACHE[cache_key] = {
            "mtime": stat.st_mtime,
            "rows": rows,
            "row_count": len(rows),
            "as_of": snapshot.get("as_of"),
        }
    return dict(rows), {"cache_hit": False, "row_count": len(rows), "as_of": snapshot.get("as_of")}


def _holding_code(position: Dict[str, Any]) -> str:
    from app.fund_service import normalize_fund_code

    return normalize_fund_code(str(position.get("asset_code") or position.get("asset_name") or "")).replace(".", "")


def _holding_type(position: Dict[str, Any]) -> str:
    return str(position.get("asset_type") or "").strip().lower()


_GEFAS_GYF_ALIAS_MAP: Dict[str, Dict[str, str]] = {
    "TPKGY": {
        "isin": "TRYTALP00036",
        "gefas_code": "TPKGY.F1",
        "label": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
    },
    "TPKGYF": {
        "isin": "TRYTALP00036",
        "gefas_code": "TPKGY.F1",
        "label": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
    },
    "TPKGYF1": {
        "isin": "TRYTALP00036",
        "gefas_code": "TPKGY.F1",
        "label": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
    },
}
_GEFAS_GYF_QUOTE_CACHE: Dict[str, Dict[str, Any]] = {}
_GEFAS_GYF_QUOTE_CACHE_TTL = 24 * 60 * 60


def _gefas_gyf_config(symbol: str) -> Optional[Dict[str, str]]:
    normalized = str(symbol or "").strip().upper().replace(".", "")
    return _GEFAS_GYF_ALIAS_MAP.get(normalized)


def _gefas_chart_date(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _fetch_gefas_gyf_chart(isin: str, metric: int) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    normalized_isin = str(isin or "").strip().upper()
    if not normalized_isin:
        return {}
    url = f"https://gefas.gov.tr/gyf/detay/grafik/{normalized_isin}/0/0/{metric}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://gefas.gov.tr/tr/gyf/detay",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception):
        return {}


def _fetch_gefas_gyf_quote(symbol: str) -> Optional[Dict[str, Any]]:
    config = _gefas_gyf_config(symbol)
    if not config:
        return None
    cache_key = config["isin"]
    now = time.time()
    cached = _GEFAS_GYF_QUOTE_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _GEFAS_GYF_QUOTE_CACHE_TTL:
        data = dict(cached.get("data") or {})
        if data:
            data["_cache_hit"] = True
        return data

    price_chart = _fetch_gefas_gyf_chart(config["isin"], 0)
    return_chart = _fetch_gefas_gyf_chart(config["isin"], 2)
    prices = list(price_chart.get("datas") or [])
    price_labels = list(price_chart.get("labels") or [])
    returns = list(return_chart.get("datas") or [])
    return_labels = list(return_chart.get("labels") or [])
    price = _api_number(prices[-1]) if prices else None
    return_pct = _api_number(returns[-1]) if returns else None
    as_of = _gefas_chart_date(price_labels[-1] if price_labels else None) or _gefas_chart_date(return_labels[-1] if return_labels else None)
    if price is None and return_pct is None:
        _GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": {}}
        return {}

    data = {
        "price": price,
        "currency": "TRY",
        "change_pct": return_pct,
        "as_of": as_of,
        "source": "gefas_gyf",
        "source_url": f"https://gefas.gov.tr/tr/gyf/detay/{config['gefas_code']}",
        "isin": config["isin"],
        "gefas_code": config["gefas_code"],
        "label": config.get("label"),
    }
    _GEFAS_GYF_QUOTE_CACHE[cache_key] = {"_ts": now, "data": data}
    result = dict(data)
    result["_cache_hit"] = False
    return result


def _quote_map_for_holding_stocks(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    unique_symbols: List[str] = []
    seen_symbols: set[str] = set()
    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        if not normalized or normalized in seen_symbols:
            continue
        seen_symbols.add(normalized)
        unique_symbols.append(normalized)
    if not unique_symbols:
        return {}
    fetched_quotes = _fetch_market_price_map(unique_symbols, index_name="XUTUM")
    quotes = {symbol: fetched_quotes.get(symbol, {}) for symbol in unique_symbols if fetched_quotes.get(symbol)}
    missing_symbols = [
        symbol
        for symbol in unique_symbols
        if _api_number((quotes.get(symbol) or {}).get("price")) is None
        or _api_number((quotes.get(symbol) or {}).get("change_pct")) is None
    ]
    for symbol in missing_symbols[:_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT]:
        fallback = _fetch_infoyatirim_stock_page_quote(symbol)
        if fallback:
            quotes[symbol] = _merge_market_price_fallback(quotes.get(symbol, {}), fallback)
    return quotes


def _position_daily_market_fields(
    position: Dict[str, Any],
    *,
    stock_quotes: Dict[str, Dict[str, Any]],
    gefas_quotes: Dict[str, Dict[str, Any]],
    fund_rows: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    code = _holding_code(position)
    asset_type = _holding_type(position)
    if asset_type == "local_equity":
        quote = stock_quotes.get(code) or {}
        return {
            "price": _api_number(quote.get("price")),
            "price_currency": quote.get("currency") or "TRY",
            "return_pct": _api_number(quote.get("change_pct")),
            "return_source": "infoyatirim_live_quote" if quote else None,
            "return_as_of": quote.get("as_of"),
        }
    if asset_type == "fund":
        row = fund_rows.get(code) or {}
        quote = stock_quotes.get(code) or {}
        if quote and _api_number(quote.get("change_pct")) is not None:
            return {
                "price": _api_number(quote.get("price")),
                "price_currency": quote.get("currency") or "TRY",
                "return_pct": _api_number(quote.get("change_pct")),
                "return_source": "infoyatirim_live_quote",
                "return_as_of": quote.get("as_of"),
            }
        gefas_quote = gefas_quotes.get(code) or {}
        if gefas_quote and (_api_number(gefas_quote.get("price")) is not None or _api_number(gefas_quote.get("change_pct")) is not None):
            return {
                "price": _api_number(gefas_quote.get("price")),
                "price_currency": gefas_quote.get("currency") or "TRY",
                "return_pct": _api_number(gefas_quote.get("change_pct")),
                "return_source": "gefas_gyf",
                "return_as_of": gefas_quote.get("as_of"),
            }
        return {
            "price": _api_number(row.get("price")),
            "price_currency": row.get("currency") or "TRY",
            "return_pct": _api_number(row.get("daily_return")),
            "return_source": "tefasfon_funds" if row else None,
            "return_as_of": row.get("as_of"),
        }
    return {
        "price": _api_number(position.get("price")),
        "price_currency": None,
        "return_pct": None,
        "return_source": None,
        "return_as_of": None,
    }


def _enrich_fund_holdings_with_daily_market_data(payload: Dict[str, Any], fund_code: str) -> Dict[str, Any]:
    from app.fund_service import normalize_fund_code

    normalized_fund = normalize_fund_code(fund_code)
    positions = [dict(position) for position in list(payload.get("positions") or []) if isinstance(position, dict)]
    fund_rows, fund_rows_meta = _fund_snapshot_row_map_with_meta()
    fund_row = fund_rows.get(normalized_fund) or {}
    fund_aum = _api_number(fund_row.get("aum"))
    stock_symbols = [
        _holding_code(position)
        for position in positions
        if _holding_code(position)
        and (
            _holding_type(position) == "local_equity"
            or (
                _holding_type(position) == "fund"
                and _holding_code(position) not in fund_rows
                and not _gefas_gyf_config(_holding_code(position))
            )
        )
    ]
    stock_quotes = _quote_map_for_holding_stocks(stock_symbols)
    gefas_quotes: Dict[str, Dict[str, Any]] = {}
    gefas_quote_cache_hits = 0
    for position in positions:
        code = _holding_code(position)
        if _holding_type(position) != "fund" or not _gefas_gyf_config(code):
            continue
        quote = _fetch_gefas_gyf_quote(code)
        if quote:
            if quote.get("_cache_hit"):
                gefas_quote_cache_hits += 1
            gefas_quotes[code] = quote

    enriched_positions: List[Dict[str, Any]] = []
    estimated_return_pct = 0.0
    estimated_pnl_value = 0.0
    has_pnl = False
    priced_weight = 0.0
    missing_weight = 0.0

    for position in positions:
        row = dict(position)
        daily_fields = _position_daily_market_fields(row, stock_quotes=stock_quotes, gefas_quotes=gefas_quotes, fund_rows=fund_rows)
        row.update(daily_fields)

        weight = _api_number(row.get("weight"))
        return_pct = _api_number(row.get("return_pct"))
        exposure_value = (fund_aum * weight / 100.0) if fund_aum is not None and weight is not None and weight > 0 else None
        contribution_pct = (weight * return_pct / 100.0) if weight is not None and weight > 0 and return_pct is not None else None
        pnl_value = (exposure_value * return_pct / 100.0) if exposure_value is not None and return_pct is not None else None

        row["estimated_exposure_value"] = round(exposure_value, 2) if exposure_value is not None else None
        row["estimated_pnl_value"] = round(pnl_value, 2) if pnl_value is not None else None
        row["estimated_fund_return_contribution_pct"] = round(contribution_pct, 6) if contribution_pct is not None else None

        if weight is not None and weight > 0:
            if return_pct is not None:
                priced_weight += weight
                estimated_return_pct += contribution_pct or 0.0
                if pnl_value is not None:
                    estimated_pnl_value += pnl_value
                    has_pnl = True
            else:
                missing_weight += weight
        enriched_positions.append(row)

    enriched_payload = dict(payload)
    enriched_payload["positions"] = enriched_positions
    enriched_payload["portfolio_effect"] = {
        "period": "daily",
        "estimated_return_pct": round(estimated_return_pct, 6),
        "estimated_pnl_value": round(estimated_pnl_value, 2) if has_pnl else None,
        "priced_weight": round(priced_weight, 6),
        "missing_weight": round(missing_weight, 6),
        "aum": fund_aum,
        "as_of": fund_row.get("as_of") or (payload.get("source_metadata") or {}).get("as_of"),
    }
    metadata = dict(enriched_payload.get("source_metadata") or {})
    metadata["daily_market_enrichment"] = {
        "period": "daily",
        "stock_quote_count": len(stock_quotes),
        "fund_snapshot_count": len(fund_rows),
        "gefas_gyf_quote_count": len(gefas_quotes),
        "gefas_gyf_quote_cache_hits": gefas_quote_cache_hits,
        "daily_reference_cache_hit": bool(fund_rows_meta.get("cache_hit")),
        "daily_reference_row_count": fund_rows_meta.get("row_count"),
        "priced_weight": round(priced_weight, 6),
        "missing_weight": round(missing_weight, 6),
    }
    metadata["market_enrichment"] = {
        "stock_quote_live": True,
        "stock_quote_count": len(stock_quotes),
        "fund_daily_reference": "tefas_snapshot",
        "fund_daily_reference_cache_hit": bool(fund_rows_meta.get("cache_hit")),
        "gefas_gyf_cache_ttl_seconds": _GEFAS_GYF_QUOTE_CACHE_TTL,
    }
    enriched_payload["source_metadata"] = metadata
    return enriched_payload


@app.get("/funds/{fund_code}/allocations")
def fund_allocations(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_allocations_payload

    return get_fund_allocations_payload(CONFIG.paths.processed_dir, fund_code)


@app.get("/funds/{fund_code}/allocations/history")
def fund_allocations_history(
    fund_code: str,
    lookback_days: int = Query(30, ge=1, le=365),
) -> Dict[str, Any]:
    from app.fund_service import get_fund_allocations_history_payload

    return get_fund_allocations_history_payload(
        CONFIG.paths.processed_dir,
        fund_code,
        lookback_days=lookback_days,
    )


@app.get("/funds/{fund_code}")
def fund_detail(fund_code: str) -> Dict[str, Any]:
    from app.fund_service import get_fund_detail_payload, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    try:
        return get_fund_detail_payload(CONFIG.paths.processed_dir, normalized)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Fon bulunamadi: {normalized}") from exc


@app.post("/admin/funds/refresh-snapshot")
def admin_refresh_funds_snapshot(lookback_days: int = Query(10, ge=1, le=45)) -> Dict[str, Any]:
    from app.fund_service import FundUpstreamError, refresh_funds_snapshot

    try:
        return refresh_funds_snapshot(CONFIG.paths.processed_dir, lookback_days=lookback_days)
    except FundUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/admin/funds/collect-prices")
def admin_collect_fund_prices(
    lookback_days: int = Query(10, ge=1, le=45),
    as_of: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import collect_daily_fund_prices

    return collect_daily_fund_prices(
        CONFIG.paths.processed_dir,
        as_of=as_of,
        lookback_days=lookback_days,
    )


@app.post("/admin/funds/{fund_code}/refresh-performance")
def admin_refresh_fund_performance(
    fund_code: str,
    start_date: date = Query(...),
    end_date: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import FundUpstreamError, refresh_fund_performance, normalize_fund_code

    normalized = normalize_fund_code(fund_code)
    effective_end_date = end_date or date.today()
    if start_date > effective_end_date:
        raise HTTPException(status_code=400, detail="start_date end_date sonrasinda olamaz")
    try:
        return refresh_fund_performance(
            CONFIG.paths.processed_dir,
            normalized,
            start_date=start_date,
            end_date=effective_end_date,
        )
    except FundUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/admin/funds/{fund_code}/refresh-allocations")
def admin_refresh_fund_allocations(
    fund_code: str,
    as_of: Optional[date] = Query(None),
) -> Dict[str, Any]:
    from app.fund_service import FundUpstreamError, normalize_fund_code, refresh_fund_allocations

    normalized = normalize_fund_code(fund_code)
    try:
        return refresh_fund_allocations(CONFIG.paths.processed_dir, normalized, as_of=as_of)
    except FundUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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


@app.post("/kap/overview-commentary")
async def kap_overview_commentary(request: Request) -> Dict[str, Any]:
    started_at = time.perf_counter()
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        LOGGER.warning(
            "[kap_overview_commentary] request body too large | bytes=%s limit=%s",
            len(body),
            MAX_REQUEST_BYTES,
        )
        raise HTTPException(status_code=413, detail="request body en fazla 64 KB olabilir")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "[kap_overview_commentary] invalid json body | bytes=%s error=%s",
            len(body),
            exc,
        )
        raise HTTPException(status_code=400, detail="gecerli JSON body gerekli") from exc
    LOGGER.info(
        "[kap_overview_commentary] request received | company=%s latest_period=%s bytes=%s",
        str(payload.get("company") or "").strip(),
        str(payload.get("latest_period") or "").strip(),
        len(body),
    )
    try:
        response = await _run_overview_commentary_until_done_or_disconnected(request, payload)
    except PayloadValidationError as exc:
        LOGGER.warning(
            "[kap_overview_commentary] payload validation failed | company=%s detail=%s",
            str(payload.get("company") or "").strip(),
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    LOGGER.info(
        "[kap_overview_commentary] completed | company=%s ok=%s model=%s score_source=%s elapsed_ms=%s error=%s",
        str(payload.get("company") or "").strip(),
        response.get("ok"),
        response.get("model_used"),
        ((response.get("scorecard") or {}) if isinstance(response.get("scorecard"), dict) else {}).get("score_source"),
        elapsed_ms,
        (str(response.get("error") or "")[:180] or None),
    )
    return response


async def _run_overview_commentary_until_done_or_disconnected(
    request: Request,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    task = asyncio.create_task(generate_overview_commentary(payload))
    company = str(payload.get("company") or "").strip()
    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                LOGGER.info(
                    "[kap_overview_commentary] client disconnected; cancelling NVIDIA request | company=%s",
                    company,
                )
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise HTTPException(status_code=499, detail="client disconnected")
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
            except asyncio.TimeoutError:
                continue
        return await task
    except asyncio.CancelledError:
        task.cancel()
        LOGGER.info(
            "[kap_overview_commentary] request task cancelled | company=%s",
            company,
        )
        raise


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
_KAP_MEMBER_OID_NEGATIVE_CACHE: Dict[str, float] = {}
_KAP_MEMBER_OID_NEGATIVE_TTL = 1800  # 30 dk
_KAP_MEMBER_OID_WARMED_FROM_CACHE = False
_MARKET_KAP_LOGO_CACHE: Dict[str, Any] = {}
_MARKET_KAP_LOGO_CACHE_TTL = 1800  # 30 dk
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


def _resolve_kap_member_oid(symbol: str) -> Optional[str]:
    global _KAP_MEMBER_OID_WARMED_FROM_CACHE
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return None

    if not _KAP_MEMBER_OID_WARMED_FROM_CACHE:
        _KAP_MEMBER_OID_WARMED_FROM_CACHE = True
        kap_cache_dir = CONFIG.paths.processed_dir / "kap_cache"
        if kap_cache_dir.exists():
            for cache_file in kap_cache_dir.glob("*.json"):
                try:
                    with cache_file.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except Exception:
                    continue
                cache_symbol = str(payload.get("stock_code") or cache_file.stem).strip().upper()
                cache_oid = str(payload.get("member_oid") or "").strip()
                if cache_symbol and cache_oid and cache_symbol not in _KAP_MEMBER_OID_CACHE:
                    _KAP_MEMBER_OID_CACHE[cache_symbol] = cache_oid

    oid = _KAP_MEMBER_OID_CACHE.get(normalized)
    if oid:
        return oid

    now = time.time()
    negative_ts = _KAP_MEMBER_OID_NEGATIVE_CACHE.get(normalized, 0.0)
    if negative_ts and now - negative_ts < _KAP_MEMBER_OID_NEGATIVE_TTL:
        return None

    import urllib.parse
    import urllib.request

    headers = {
        "Accept": "application/json",
        "Accept-Language": "tr",
        "User-Agent": _KAP_DEFAULT_HEADERS["User-Agent"],
    }
    try:
        encoded_symbol = urllib.parse.quote(normalized, safe="")
        req = urllib.request.Request(
            f"https://www.kap.org.tr/tr/api/member/filter/{encoded_symbol}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            rows = json.loads(resp.read().decode("utf-8", errors="ignore"))
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate = str(row.get("mkkMemberOid") or "").strip()
                if candidate:
                    _KAP_MEMBER_OID_CACHE[normalized] = candidate
                    _KAP_MEMBER_OID_NEGATIVE_CACHE.pop(normalized, None)
                    return candidate
    except Exception:
        pass

    _KAP_MEMBER_OID_NEGATIVE_CACHE[normalized] = now
    return None


def _kap_logo_payload_for_symbol(symbol: str) -> Dict[str, Optional[str]]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return {"logo_url": None, "logo_source": None}

    now = time.time()
    cached = _MARKET_KAP_LOGO_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_KAP_LOGO_CACHE_TTL:
        return dict(cached.get("data") or {"logo_url": None, "logo_source": None})

    oid = _resolve_kap_member_oid(normalized)
    data = {
        "logo_url": f"https://www.kap.org.tr/tr/api/member/logo/{oid}" if oid else None,
        "logo_source": "kap" if oid else None,
    }
    _MARKET_KAP_LOGO_CACHE[normalized] = {"_ts": now, "data": data}
    return dict(data)


def _empty_logo_payload() -> Dict[str, Optional[str]]:
    return {"logo_url": None, "logo_source": None}


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
    companies = get_kap_companies(indexed)
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    items: List[Dict[str, Any]] = []
    for symbol in companies:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            continue
        cached_meta = _load_cached_kap_market_metadata(cache_dir, normalized)
        title = str(get_instrument_name(CONFIG.paths.processed_dir, "stock", normalized) or cached_meta.get("company_title") or "").strip()
        company_code = str(cached_meta.get("company") or normalized).strip().upper()
        aliases = [normalized]
        if company_code and company_code != normalized:
            aliases.append(company_code)
        if title:
            aliases.append(title)
        items.append(
            {
                "symbol": normalized,
                "title": title or None,
                "aliases": aliases,
                "latest_quarter": cached_meta.get("latest_quarter"),
                "has_kap_cache": bool(cached_meta.get("has_kap_cache")),
            }
        )
    return {"companies": companies, "items": items}


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
        use_cache_when_complete=not refresh,
    )
    _upsert_stock_reference_from_kap_payload(company, raw, source="kap")
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
_MARKET_PRICE_CACHE_TTL = 3  # seconds; used by the live stocks table
_INFOYATIRIM_STOCK_PAGE_CACHE: Dict[str, Any] = {}
_INFOYATIRIM_STOCK_PAGE_CACHE_TTL = 60
_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT = 12
_STOCKS_CACHE: Dict[str, Any] = {}
_STOCKS_CACHE_TTL = 3
_MARKET_STOCK_CARD_CHART_CACHE: Dict[str, Any] = {}
_MARKET_STOCK_CARD_CHART_CACHE_TTL = 45
_MARKET_STOCK_CARD_LIMIT = 12
_MARKET_STOCK_CARD_PREVIOUS_SESSION_LOOKBACK_DAYS = 10
_TURKEY_TIMEZONE = timezone(timedelta(hours=3))
_MARKET_STOCK_CARD_CHART_RANGES: Dict[str, Dict[str, Any]] = {
    "1d": {"interval": "5m", "range": "1d", "ttl": 30},
    "1w": {"interval": "15m", "range": "5d", "ttl": 60},
    "1m": {"interval": "1d", "range": "1mo", "ttl": 600},
    "1y": {"interval": "1wk", "range": "1y", "ttl": 3600},
}
_STOCK_RETURN_BASE_CACHE: Dict[str, Any] = {}
_STOCK_RETURN_BASE_CACHE_TTL = 900  # 15 minutes
_ISYATIRIM_CACHE: Dict[str, Any] = {}
_ISYATIRIM_CACHE_TTL = 900  # 15 minutes
_ISYATIRIM_BASIC_SUMMARY_CACHE: Dict[str, Any] = {}
_ISYATIRIM_BASIC_SUMMARY_CACHE_TTL = 60
_MARKET_STOCK_INDEX_ORDER = ["XUTUM", "XU100", "XU030"]
_MARKET_STOCK_INDEXES = set(_MARKET_STOCK_INDEX_ORDER)
_MARKET_INDICES_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_DETAIL_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_QUOTE_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_INTRADAY_CACHE: Dict[str, Any] = {}
_MARKET_INDEX_RETURN_CACHE: Dict[str, Any] = {}
_MARKET_INDICES_CACHE_TTL = 3
_MARKET_INDEX_DETAIL_CACHE_TTL = 3
_MARKET_INDEX_QUOTE_CACHE_TTL = 3
_MARKET_INDEX_INTRADAY_CACHE_TTL = 3
_MARKET_INDEX_RETURN_CACHE_TTL = 900
_MARKET_INDEX_META: Dict[str, Dict[str, Any]] = {
    "XUTUM": {
        "symbol": "XUTUM",
        "label": "BIST Tüm",
        "yahoo_candidates": ["XUTUM.IS", "^XUTUM", "XUTUM"],
    },
    "XU100": {
        "symbol": "XU100",
        "label": "BIST 100",
        "yahoo_candidates": ["XU100.IS", "^XU100", "XU100"],
    },
    "XU030": {
        "symbol": "XU030",
        "label": "BIST 30",
        "yahoo_candidates": ["XU030.IS", "^XU030", "XU030"],
    },
}


def _supported_market_indexes_text() -> str:
    return ", ".join(_MARKET_STOCK_INDEX_ORDER)


def _normalize_stock_index(index_name: str) -> str:
    normalized = str(index_name or "XUTUM").strip().upper()
    if normalized not in _MARKET_STOCK_INDEXES:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen endeks. {_supported_market_indexes_text()} kullanin.",
        )
    return normalized
_RETURN_BASE_FIELDS: List[tuple[str, str]] = [
    ("return_1w_pct", "base_1w"),
    ("return_1m_pct", "base_1m"),
    ("return_3m_pct", "base_3m"),
    ("return_6m_pct", "base_6m"),
    ("return_ytd_pct", "base_ytd"),
    ("return_1y_pct", "base_1y"),
]
_INDEX_RETURN_BASE_FIELDS: List[tuple[str, str]] = _RETURN_BASE_FIELDS + [
    ("return_5y_pct", "base_5y"),
]


def _isyatirim_company_card_url(symbol: str) -> str:
    return f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"


def _isyatirim_basic_summary_url() -> str:
    return "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx"


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


def _infoyatirim_stock_page_url(symbol: str) -> str:
    ticker = str(symbol or "").strip().lower()
    return f"https://infoyatirim.com/borsa/{ticker}-hisse"


def _infoyatirim_stock_page_text(html_text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", str(html_text or ""), flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text, flags=re.IGNORECASE))
    text = " ".join(text.split())
    return (
        text.upper()
        .replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )


def _extract_infoyatirim_stock_page_quote(symbol: str, html_text: str) -> Dict[str, Any]:
    ticker = str(symbol or "").strip().upper()
    if not ticker or not html_text:
        return {}

    text = _infoyatirim_stock_page_text(html_text)

    def value_after(label: str) -> Optional[float]:
        match = re.search(rf"{label}\s+([-+]?\d[\d\.,]*\s*(?:%|₺|TL)?)", text, flags=re.IGNORECASE)
        return _parse_tr_decimal(match.group(1)) if match else None

    price = value_after(r"SON ISLEM FIYATI")
    change_pct = value_after(r"GUNLUK DEGISIM\s+%")
    change = value_after(r"GUNLUK DEGISIM\s+\(TL\)")
    volume = value_after(r"GUNLUK HACIM\s+\(TL\)")
    if volume is None:
        volume = value_after(r"TOPLAM ISLEM HACMI")
    market_cap = value_after(r"PIYASA DEGERI")
    fk = value_after(r"F/K")
    pd_dd = value_after(r"PD/DD")
    fd_favok = value_after(r"FD/FAVOK")

    if price is None and change_pct is None and volume is None and market_cap is None and fk is None and pd_dd is None and fd_favok is None:
        return {}

    return {
        "price": price,
        "currency": "TRY",
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "market_cap": market_cap,
        "fk": fk,
        "pd_dd": pd_dd,
        "fd_favok": fd_favok,
        "market_state": "",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_infoyatirim_stock_page_quote(symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {}

    now = time.time()
    cached = _INFOYATIRIM_STOCK_PAGE_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < _INFOYATIRIM_STOCK_PAGE_CACHE_TTL:
        return dict(cached.get("data") or {})

    request = urllib.request.Request(
        _infoyatirim_stock_page_url(ticker),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            html_text = response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, Exception):
        _INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": {}}
        return {}

    data = _extract_infoyatirim_stock_page_quote(ticker, html_text)
    _INFOYATIRIM_STOCK_PAGE_CACHE[ticker] = {"_ts": now, "data": data}
    return dict(data)


def _market_price_row_needs_fallback(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return True
    return row.get("price") is None or row.get("change_pct") is None or row.get("volume") is None


def _merge_market_price_fallback(base: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not fallback:
        return base
    merged = dict(base or {})
    for key in ("price", "currency", "change", "change_pct", "volume", "market_cap", "market_state", "as_of"):
        if key not in merged or merged.get(key) is None or (key in {"currency", "market_state"} and merged.get(key) == ""):
            value = fallback.get(key)
            if value is not None:
                merged[key] = value
    return merged


def _market_price_source_url(index_name: str) -> str:
    normalized = str(index_name or "XUTUM").strip().upper()
    if normalized == "XUTUM":
        return "https://infoyatirim.com/canli-borsa"
    if normalized == "XU030":
        return "https://infoyatirim.com/canli-borsa/xu100-bist-100-hisseleri"
    return "https://infoyatirim.com/canli-borsa/xu100-bist-100-hisseleri"


def _fetch_market_price_map(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
    import urllib.error
    import urllib.request

    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not normalized_symbols:
        return {}

    normalized_index = str(index_name or "XU100").strip().upper()
    cache_key = f"{normalized_index}:{','.join(normalized_symbols)}"
    now = time.time()
    cached = _MARKET_PRICE_CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _MARKET_PRICE_CACHE_TTL:
        return cached.get("items", {})

    url = _market_price_source_url(normalized_index)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, Exception):
        return {}

    items: Dict[str, Dict[str, Any]] = {}
    try:
        row_pattern = re.compile(
            r'<tr[^>]+data-symbol="(?P<symbol>[A-Z0-9]+)"[^>]*>(?P<body>.*?)</tr>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        price_pattern = re.compile(r'<td[^>]+class="price"[^>]+data-val="(?P<value>[^"]+)"', re.IGNORECASE)
        change_pattern = re.compile(r'<td[^>]+class="change"[^>]+data-val="(?P<value>[^"]+)"', re.IGNORECASE)
        percent_pattern = re.compile(r'<td[^>]+class="percent"[^>]+data-val="(?P<value>[^"]+)"', re.IGNORECASE)
        fetched_at = datetime.now(timezone.utc).isoformat()
        for match in row_pattern.finditer(html_text):
            symbol = str(match.group("symbol") or "").strip().upper()
            if not symbol:
                continue
            body = match.group("body") or ""
            price_match = price_pattern.search(body)
            change_match = change_pattern.search(body)
            percent_match = percent_pattern.search(body)

            volume = None
            cells = re.findall(r"<td\b[^>]*>(.*?)</td>", body, flags=re.IGNORECASE | re.DOTALL)
            if len(cells) > 4:
                volume_raw = html.unescape(re.sub(r"<[^>]+>", " ", cells[4], flags=re.IGNORECASE))
                volume = _parse_tr_decimal(volume_raw)

            items[symbol] = {
                "price": _parse_tr_decimal(price_match.group("value") if price_match else None),
                "currency": "TRY",
                "change": _parse_tr_decimal(change_match.group("value") if change_match else None),
                "change_pct": _parse_tr_decimal(percent_match.group("value") if percent_match else None),
                "volume": volume,
                "market_state": "",
                "as_of": fetched_at,
            }
    except Exception:
        return {}

    missing_symbols = [
        symbol
        for symbol in normalized_symbols
        if _market_price_row_needs_fallback(items.get(symbol))
    ]
    if items and missing_symbols and normalized_index != "XUTUM":
        fallback_symbols = missing_symbols[:_INFOYATIRIM_STOCK_PAGE_FALLBACK_LIMIT]
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(6, len(fallback_symbols))) as pool:
                fallback_rows = list(pool.map(_fetch_infoyatirim_stock_page_quote, fallback_symbols))
        except Exception:
            fallback_rows = [_fetch_infoyatirim_stock_page_quote(symbol) for symbol in fallback_symbols]
        for symbol, fallback in zip(fallback_symbols, fallback_rows):
            if fallback:
                items[symbol] = _merge_market_price_fallback(items.get(symbol, {}), fallback)

    _MARKET_PRICE_CACHE[cache_key] = {"_ts": now, "items": items}
    return items


def _pick_series_value_at_or_before(
    points: List[tuple[datetime, float]],
    target: datetime,
) -> Optional[float]:
    candidate = None
    for point_dt, close in points:
        if point_dt <= target:
            candidate = close
        elif candidate is not None:
            break
    return candidate


def _pick_series_value_at_or_after(
    points: List[tuple[datetime, float]],
    target: datetime,
) -> Optional[float]:
    for point_dt, close in points:
        if point_dt >= target:
            return close
    return None


def _fetch_stock_return_bases(symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        return {}

    now = time.time()
    cached = _STOCK_RETURN_BASE_CACHE.get(ticker)
    if cached and now - cached.get("_ts", 0) < _STOCK_RETURN_BASE_CACHE_TTL:
        return dict(cached.get("data") or {})

    yahoo_symbol = f"{ticker}.IS"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        "?interval=1d&range=1y"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception):
        return {}

    try:
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
    except (KeyError, IndexError, TypeError):
        return {}

    series: List[Dict[str, Any]] = []
    for ts, close, high, low in zip(timestamps, closes, highs, lows):
        if close is None or not isinstance(ts, (int, float)):
            continue
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            series.append({
                "dt": dt,
                "close": float(close),
                "high": float(high) if high is not None else float(close),
                "low": float(low) if low is not None else float(close),
            })
        except (TypeError, ValueError):
            continue

    if not series:
        return {}

    series.sort(key=lambda x: x["dt"])
    latest = series[-1]
    latest_dt = latest["dt"]
    latest_close = latest["close"]
    year_start = datetime(latest_dt.year, 1, 1, tzinfo=timezone.utc)

    def _get_base_price(target_dt: datetime) -> Optional[float]:
        points_for_base = [(s["dt"], s["close"]) for s in series]
        return _pick_series_value_at_or_before(points_for_base, target_dt)

    def _get_range_stats(start_dt: datetime):
        relevant = [s for s in series if s["dt"] >= start_dt]
        if not relevant:
            return None, None, None
        
        base_val = _get_base_price(start_dt)
        high_val = max(s["high"] for s in relevant)
        low_val = min(s["low"] for s in relevant)
        return base_val, high_val, low_val

    b1w, h1w, l1w = _get_range_stats(latest_dt - timedelta(days=7))
    b1m, h1m, l1m = _get_range_stats(latest_dt - timedelta(days=30))
    b3m, h3m, l3m = _get_range_stats(latest_dt - timedelta(days=91))
    b6m, h6m, l6m = _get_range_stats(latest_dt - timedelta(days=182))
    bytd, hytd, lytd = _get_range_stats(year_start)
    b1y, h1y, l1y = _get_range_stats(latest_dt - timedelta(days=365))

    bases = {
        "base_1w": b1w, "high_1w": h1w, "low_1w": l1w,
        "base_1m": b1m, "high_1m": h1m, "low_1m": l1m,
        "base_3m": b3m, "high_3m": h3m, "low_3m": l3m,
        "base_6m": b6m, "high_6m": h6m, "low_6m": l6m,
        "base_ytd": bytd, "high_ytd": hytd, "low_ytd": lytd,
        "base_1y": b1y, "high_1y": h1y, "low_1y": l1y,
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
    }
    _STOCK_RETURN_BASE_CACHE[ticker] = {"_ts": now, "data": bases}
    return dict(bases)


def _fetch_stock_return_bases_bulk(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    normalized_symbols = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not normalized_symbols:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    stale: List[str] = []
    now = time.time()
    for symbol in normalized_symbols:
        cached = _STOCK_RETURN_BASE_CACHE.get(symbol)
        if cached and now - cached.get("_ts", 0) < _STOCK_RETURN_BASE_CACHE_TTL:
            result[symbol] = dict(cached.get("data") or {})
        else:
            stale.append(symbol)

    if not stale:
        return result

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for symbol, bases in zip(stale, pool.map(_fetch_stock_return_bases, stale)):
                result[symbol] = bases
    except Exception:
        for symbol in stale:
            result[symbol] = _fetch_stock_return_bases(symbol)
    return result


def _return_pct(current_price: Any, base_price: Any) -> Optional[float]:
    try:
        price = float(current_price)
        base = float(base_price)
    except (TypeError, ValueError):
        return None
    if price <= 0 or base <= 0:
        return None
    return round(((price - base) / base) * 100, 2)


def _returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Any]:
    res = {}
    for response_field, base_field in _RETURN_BASE_FIELDS:
        base_val = return_bases.get(base_field)
        res[response_field] = _return_pct(current_price, base_val)
        
        # Extract period (e.g. 1w from return_1w_pct)
        period = response_field.replace("return_", "").replace("_pct", "")
        res[f"base_{period}"] = base_val
        res[f"high_{period}"] = return_bases.get(f"high_{period}")
        res[f"low_{period}"] = return_bases.get(f"low_{period}")
    return res





def _market_stock_benchmarks() -> Dict[str, Dict[str, Any]]:
    base_map = _fetch_stock_return_bases_bulk(_MARKET_STOCK_INDEX_ORDER)
    benchmarks: Dict[str, Dict[str, Any]] = {}
    for index_name in _MARKET_STOCK_INDEX_ORDER:
        bases = base_map.get(index_name, {})
        current_for_returns = bases.get("latest_close")
        benchmarks[index_name] = {
            **_returns_from_bases(current_for_returns, bases),
            "as_of": bases.get("as_of"),
        }
    return benchmarks


def _market_stock_symbols_for_index(index_name: str) -> List[str]:
    from app.kap_service import get_bist100_companies, get_bist30_companies, get_bist_all_companies

    normalized = _normalize_stock_index(index_name)
    if normalized == "XUTUM":
        return get_bist_all_companies()
    if normalized == "XU100":
        return get_bist100_companies()
    return get_bist30_companies()


def _cached_stock_return_bases_bulk(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    now = time.time()
    result: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        cached = _STOCK_RETURN_BASE_CACHE.get(normalized)
        if cached and now - cached.get("_ts", 0) < _STOCK_RETURN_BASE_CACHE_TTL:
            result[normalized] = dict(cached.get("data") or {})
    return result


def _market_stock_row(
    symbol: str,
    *,
    breakdown_row: Dict[str, Any],
    cached_meta: Dict[str, Any],
    quote: Dict[str, Any],
    return_bases: Dict[str, Any],
    basic_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    quarters = [
        str(item or "").strip().upper()
        for item in (breakdown_row.get("quarters") or [])
        if str(item or "").strip()
    ]
    latest_quarter = _latest_quarter_label(
        quarters + ([cached_meta["latest_quarter"]] if cached_meta.get("latest_quarter") else [])
    )
    current_for_returns = quote.get("price") if quote.get("price") is not None else return_bases.get("latest_close")
    return {
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
        "volume": quote.get("volume"),
        "market_cap": _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary),
        **_empty_logo_payload(),
        **_returns_from_bases(current_for_returns, return_bases),
    }


def _market_stocks_payload(*, index_name: str = "XUTUM", force_refresh: bool = False) -> Dict[str, Any]:
    from app.kap_service import get_bist_index_universe

    normalized_index = _normalize_stock_index(index_name)
    now_ts = time.time()
    cache_key = f"payload:{normalized_index}"
    cached = _STOCKS_CACHE.get(cache_key)
    if cached and not force_refresh and now_ts - cached.get("_ts", 0) < _STOCKS_CACHE_TTL:
        return cached["data"]

    symbols = _market_stock_symbols_for_index(normalized_index)
    try:
        universe = get_bist_index_universe(normalized_index, force_refresh=force_refresh)
    except Exception:
        universe = {
            "index": normalized_index,
            "count": len(symbols),
            "source": None,
            "source_url": None,
            "source_date": None,
            "fetched_at": None,
            "cache_hit": False,
            "fallback_used": False,
        }
    breakdown_rows = _company_breakdown_from_chunks(CONFIG.paths.chunks_v2_file)
    breakdown_map = {
        str(row.get("company") or "").strip().upper(): row
        for row in breakdown_rows
        if str(row.get("company") or "").strip()
    }
    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    price_map = _fetch_market_price_map(symbols, index_name=normalized_index)
    return_base_map = (
        _cached_stock_return_bases_bulk(symbols)
        if normalized_index == "XUTUM"
        else _fetch_stock_return_bases_bulk(symbols)
    )
    basic_summary_map = _fetch_isyatirim_basic_summary_map()
    rows = [
        _market_stock_row(
            symbol,
            breakdown_row=breakdown_map.get(symbol, {}),
            cached_meta=_load_cached_kap_market_metadata(cache_dir, symbol),
            quote=price_map.get(symbol, {}),
            return_bases=return_base_map.get(symbol, {}),
            basic_summary=basic_summary_map.get(symbol),
        )
        for symbol in symbols
    ]
    data = {
        "index": normalized_index,
        "rows": rows,
        "benchmarks": _market_stock_benchmarks(),
        "source": "infoyatirim_yahoo",
        "universe": {
            "index": universe.get("index") or normalized_index,
            "count": int(universe.get("count") or len(symbols)),
            "source": universe.get("source"),
            "source_url": universe.get("source_url"),
            "source_date": universe.get("source_date"),
            "fetched_at": universe.get("fetched_at"),
            "cache_hit": bool(universe.get("cache_hit")),
            "fallback_used": bool(universe.get("fallback_used")),
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _STOCKS_CACHE[cache_key] = {"_ts": now_ts, "data": data}
    return data


def _normalize_market_stock_card_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip()
    item = raw.upper()
    if item.endswith(".IS"):
        item = item[:-3]
    if not item or not re.fullmatch(r"[A-Z0-9]{2,12}", item):
        raise HTTPException(status_code=400, detail=f"Gecersiz hisse kodu: {raw or symbol}")
    return item


def _normalize_market_stock_card_symbols(symbols: str) -> List[str]:
    raw_items = re.split(r"[,\s]+", str(symbols or "").strip())
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not raw.strip():
            continue
        item = _normalize_market_stock_card_symbol(raw)
        if item in seen:
            continue
        normalized.append(item)
        seen.add(item)

    if len(normalized) > _MARKET_STOCK_CARD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"En fazla {_MARKET_STOCK_CARD_LIMIT} hisse karti secilebilir.",
        )
    return normalized


def _normalize_stock_card_chart_range(chart_range: str) -> str:
    normalized = str(chart_range or "1d").strip().lower()
    if normalized not in _MARKET_STOCK_CARD_CHART_RANGES:
        allowed = ", ".join(sorted(_MARKET_STOCK_CARD_CHART_RANGES))
        raise HTTPException(status_code=400, detail=f"Desteklenmeyen grafik araligi. Desteklenenler: {allowed}")
    return normalized


def _stock_card_chart_cache_key(symbol: str, chart_range: str) -> str:
    return f"stock-card-chart:{symbol}:{chart_range}"


def _point_datetime(raw: Any) -> Optional[datetime]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _numeric_chart_value(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def _normalize_stock_card_line_points(raw_points: Any) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_points, list):
        return []

    for point in raw_points:
        if not isinstance(point, dict):
            continue
        point_dt = _point_datetime(point.get("time"))
        close = _numeric_chart_value(point.get("close"))
        if point_dt is None or close is None or close <= 0:
            continue
        time_key = point_dt.isoformat()
        row: Dict[str, Any] = {"time": time_key, "close": close}
        for key in ("open", "high", "low", "volume"):
            value = _numeric_chart_value(point.get(key))
            if value is not None:
                row[key] = value
        deduped[time_key] = row

    return [deduped[key] for key in sorted(deduped)]


def _fetch_previous_stock_card_intraday_chart(yahoo_symbol: str) -> Dict[str, Any]:
    today = datetime.now(_TURKEY_TIMEZONE).date()
    errors: List[str] = []
    for offset in range(1, _MARKET_STOCK_CARD_PREVIOUS_SESSION_LOOKBACK_DAYS + 1):
        session_date = today - timedelta(days=offset)
        if session_date.weekday() >= 5:
            continue
        chart = _fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="5m",
            start_date=session_date,
            end_date=session_date,
        )
        points = _normalize_stock_card_line_points(chart.get("points") if chart.get("ok") else [])
        if points:
            meta = dict(chart.get("meta") or {})
            meta["fallbackTradingDate"] = session_date.isoformat()
            chart = dict(chart)
            chart["meta"] = meta
            chart["points"] = points
            chart["fallback_trading_date"] = session_date.isoformat()
            return chart
        error = chart.get("error") if isinstance(chart, dict) else None
        if error:
            errors.append(str(error))
    return {
        "ok": False,
        "error": errors[-1] if errors else "previous_session_unavailable",
        "yahoo_symbol": yahoo_symbol,
        "points": [],
    }


def _fetch_stock_card_chart(symbol: str, chart_range: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    ticker = _normalize_market_stock_card_symbol(symbol)
    normalized_range = _normalize_stock_card_chart_range(chart_range)
    config = _MARKET_STOCK_CARD_CHART_RANGES[normalized_range]
    cache_key = _stock_card_chart_cache_key(ticker, normalized_range)
    now = time.time()
    cached = _MARKET_STOCK_CARD_CHART_CACHE.get(cache_key)
    if cached and not force_refresh and now - cached.get("_ts", 0) < config["ttl"]:
        payload = dict(cached.get("data") or {})
        payload["source"] = "yahoo_cache"
        return payload

    yahoo_symbol = f"{ticker}.IS"
    fetched_at = datetime.now(timezone.utc).isoformat()
    chart = _fetch_yahoo_chart_raw(yahoo_symbol, interval=config["interval"], range_=config["range"])
    points = _normalize_stock_card_line_points(chart.get("points") if chart.get("ok") else [])
    source = "yahoo_live"
    if normalized_range == "1d" and chart.get("ok") and not points:
        fallback_chart = _fetch_previous_stock_card_intraday_chart(yahoo_symbol)
        fallback_points = _normalize_stock_card_line_points(fallback_chart.get("points") if fallback_chart.get("ok") else [])
        if fallback_points:
            chart = fallback_chart
            points = fallback_points
            source = "yahoo_previous_session"
    payload = {
        "symbol": ticker,
        "range": normalized_range,
        "yahoo_symbol": yahoo_symbol,
        "line_points": points,
        "source": source,
        "as_of": fetched_at,
        "error": None if chart.get("ok") and points else chart.get("error") or "chart_unavailable",
        "meta": chart.get("meta") or {},
    }
    _MARKET_STOCK_CARD_CHART_CACHE[cache_key] = {"_ts": now, "data": payload}
    return dict(payload)


def _market_stock_card_chart_payload(*, symbol: str, chart_range: str, force_refresh: bool = False) -> Dict[str, Any]:
    payload = _fetch_stock_card_chart(symbol, chart_range, force_refresh=force_refresh)
    return {
        "symbol": payload.get("symbol"),
        "range": payload.get("range"),
        "yahoo_symbol": payload.get("yahoo_symbol"),
        "line_points": payload.get("line_points") or [],
        "source": payload.get("source") or "yahoo_live",
        "as_of": payload.get("as_of"),
        "error": payload.get("error"),
    }


def _fetch_stock_card_intraday(symbol: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    chart_payload = _fetch_stock_card_chart(symbol, "1d", force_refresh=force_refresh)
    ticker = chart_payload.get("symbol") or _normalize_market_stock_card_symbol(symbol)
    yahoo_symbol = chart_payload.get("yahoo_symbol") or f"{ticker}.IS"
    points = chart_payload.get("line_points") or []
    if points:
        highs = [
            point.get("high")
            for point in points
            if isinstance(point.get("high"), (int, float))
        ]
        lows = [
            point.get("low")
            for point in points
            if isinstance(point.get("low"), (int, float))
        ]
        volumes = [
            point.get("volume")
            for point in points
            if isinstance(point.get("volume"), (int, float))
        ]
        meta_payload = chart_payload.get("meta") or {}
        last_close = points[-1].get("close") if points else None
        price = meta_payload.get("regularMarketPrice")
        if price is None and isinstance(last_close, (int, float)):
            price = last_close
        prev_close = meta_payload.get("chartPreviousClose") or meta_payload.get("previousClose")
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
        rmt = meta_payload.get("regularMarketTime")
        if isinstance(rmt, (int, float)):
            try:
                as_of = datetime.fromtimestamp(float(rmt), tz=timezone.utc).isoformat()
            except Exception:
                as_of = None

        payload = {
            "line_points": points,
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "high": meta_payload.get("regularMarketDayHigh") or (max(highs) if highs else None),
            "low": meta_payload.get("regularMarketDayLow") or (min(lows) if lows else None),
            "volume": meta_payload.get("regularMarketVolume") or (sum(volumes) if volumes else None),
            "volume_lot": meta_payload.get("regularMarketVolume") or (sum(volumes) if volumes else None),
            "currency": meta_payload.get("currency") or "TRY",
            "market_state": meta_payload.get("marketState") or "",
            "as_of": as_of or chart_payload.get("as_of"),
            "yahoo_symbol": yahoo_symbol,
            "error": None,
        }
        return dict(payload)

    fallback = {
        "line_points": [],
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "high": None,
        "low": None,
        "volume": None,
        "volume_lot": None,
        "currency": "TRY",
        "market_state": "",
        "as_of": chart_payload.get("as_of"),
        "yahoo_symbol": yahoo_symbol,
        "error": chart_payload.get("error") or "chart_unavailable",
    }
    return dict(fallback)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _is_missing_market_ratio(value: Any) -> bool:
    ratio = _parse_tr_decimal(value)
    return ratio is None or abs(ratio) <= 1e-12


def _resolve_market_card_multiples(symbol: str, multiples_payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    fk = _parse_tr_decimal(multiples_payload.get("fk")) if multiples_payload.get("ok") else None
    pd_dd = _parse_tr_decimal(multiples_payload.get("pd_dd")) if multiples_payload.get("ok") else None
    fd_favok = _parse_tr_decimal(multiples_payload.get("fd_favok")) if multiples_payload.get("ok") else None

    need_fallback = (
        _is_missing_market_ratio(fk)
        or _is_missing_market_ratio(pd_dd)
        or _is_missing_market_ratio(fd_favok)
    )
    if not need_fallback:
        return {"fk": fk, "pd_dd": pd_dd, "fd_favok": fd_favok}

    fallback = _fetch_infoyatirim_stock_page_quote(symbol)
    if _is_missing_market_ratio(fk):
        fallback_fk = _parse_tr_decimal(fallback.get("fk"))
        if fallback_fk is not None and abs(fallback_fk) > 1e-12:
            fk = fallback_fk
    if _is_missing_market_ratio(pd_dd):
        fallback_pd_dd = _parse_tr_decimal(fallback.get("pd_dd"))
        if fallback_pd_dd is not None and abs(fallback_pd_dd) > 1e-12:
            pd_dd = fallback_pd_dd
    if _is_missing_market_ratio(fd_favok):
        fallback_fd_favok = _parse_tr_decimal(fallback.get("fd_favok"))
        if fallback_fd_favok is not None and abs(fallback_fd_favok) > 1e-12:
            fd_favok = fallback_fd_favok

    return {"fk": fk, "pd_dd": pd_dd, "fd_favok": fd_favok}


def _stock_card_financial_ratios_from_cache(symbol: str) -> Dict[str, Optional[float]]:
    cache_path = CONFIG.paths.processed_dir / "kap_cache" / f"{str(symbol or '').strip().upper()}.json"
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {"net_borc_favok": None}

    quarters_raw = payload.get("quarters")
    quarters = [q for q in quarters_raw if isinstance(q, dict)] if isinstance(quarters_raw, list) else []
    if not quarters:
        return {"net_borc_favok": None}

    quarters_sorted = sorted(quarters, key=_quarter_sort_key)
    latest = quarters_sorted[-1]
    net_borc = _extract_quarter_metric(latest, "net_borc", priority=["metrics", "metrics_ytd"])
    ttm_favok = _build_ttm_sum(quarters_sorted, "favok")
    net_borc_favok = None
    try:
        if net_borc is not None and ttm_favok is not None and float(ttm_favok) != 0:
            net_borc_favok = round(float(net_borc) / float(ttm_favok), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        net_borc_favok = None
    return {"net_borc_favok": net_borc_favok}


def _market_stock_cards_payload(*, symbols: str, force_refresh: bool = False) -> Dict[str, Any]:
    normalized_symbols = _normalize_market_stock_card_symbols(symbols)
    if not normalized_symbols:
        return {
            "items": [],
            "source": "infoyatirim_yahoo_chart",
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    cache_dir = CONFIG.paths.processed_dir / "kap_cache"
    price_map = _fetch_market_price_map(normalized_symbols)
    basic_summary_map = _fetch_isyatirim_basic_summary_map()
    return_base_map = _fetch_stock_return_bases_bulk(normalized_symbols)

    items: List[Dict[str, Any]] = []
    for symbol in normalized_symbols:
        quote = price_map.get(symbol, {})
        intraday = _fetch_stock_card_intraday(symbol, force_refresh=force_refresh)
        cached_meta = _load_cached_kap_market_metadata(cache_dir, symbol)
        instrument = get_instrument(CONFIG.paths.processed_dir, "stock", symbol)
        company_name = str((instrument or {}).get("name") or cached_meta.get("company_title") or "").strip() or symbol
        basic_summary = basic_summary_map.get(symbol)
        market_cap = _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary)
        multiples = _fetch_isyatirim_multiples(symbol)
        resolved_multiples = _resolve_market_card_multiples(symbol, multiples)
        cache_ratios = _stock_card_financial_ratios_from_cache(symbol)
        return_bases = return_base_map.get(symbol, {})

        price = _first_not_none(quote.get("price"), intraday.get("price"))
        currency = quote.get("currency") or intraday.get("currency") or "TRY"
        volume_tl = quote.get("volume")
        volume_lot = _first_not_none(intraday.get("volume_lot"), intraday.get("volume"))
        current_for_returns = price if price is not None else return_bases.get("latest_close")
        item = {
            "symbol": symbol,
            "company": company_name,
            "yahoo_symbol": intraday.get("yahoo_symbol"),
            "price": price,
            "currency": currency,
            "change": _first_not_none(quote.get("change"), intraday.get("change")),
            "change_pct": _first_not_none(quote.get("change_pct"), intraday.get("change_pct")),
            "volume": _first_not_none(volume_tl, volume_lot),
            "volume_lot": volume_lot,
            "volume_tl": volume_tl,
            "market_cap": market_cap,
            "high": intraday.get("high"),
            "low": intraday.get("low"),
            "previous_close": intraday.get("prev_close"),
            "fk": resolved_multiples.get("fk"),
            "pd_dd": resolved_multiples.get("pd_dd"),
            "fd_favok": resolved_multiples.get("fd_favok"),
            "net_borc_favok": cache_ratios.get("net_borc_favok"),
            "market_state": quote.get("market_state") or intraday.get("market_state") or "",
            "as_of": quote.get("as_of") or intraday.get("as_of"),
            "line_points": intraday.get("line_points") or [],
            "error": None if price is not None or intraday.get("line_points") else intraday.get("error"),
            "logo_url": (instrument or {}).get("logo_url"),
            "logo_source": (instrument or {}).get("logo_source"),
            **_returns_from_bases(current_for_returns, return_bases),
        }
        items.append(item)

    return {
        "items": items,
        "source": "infoyatirim_yahoo_chart",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


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


def _extract_isyatirim_basic_summary_map(html_text: str) -> Dict[str, Dict[str, Any]]:
    table_match = re.search(
        r'<table[^>]+data-csvname="temelozet"[^>]*>(.*?)</table>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not table_match:
        return {}

    table_html = table_match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
    if not rows:
        return {}

    header_indexes: Dict[str, int] = {}
    items: Dict[str, Dict[str, Any]] = {}
    for row_html in rows:
        header_cells = [_strip_html_cell(cell) for cell in re.findall(r"<th[^>]*>(.*?)</th>", row_html, flags=re.IGNORECASE | re.DOTALL)]
        if header_cells:
            for idx, header in enumerate(header_cells):
                norm = _norm_text(header)
                if norm == "KOD":
                    header_indexes["symbol"] = idx
                elif "PIYASA DEGERI" in norm and "MN TL" in norm:
                    header_indexes["market_cap_mn_try"] = idx
                elif "HALKA ACIKLIK" in norm:
                    header_indexes["free_float_pct"] = idx
                elif norm.startswith("SERMAYE") and "MN TL" in norm:
                    header_indexes["capital_mn_try"] = idx
            continue

        cells = [_strip_html_cell(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.IGNORECASE | re.DOTALL)]
        if not cells:
            continue

        symbol_idx = header_indexes.get("symbol", 0)
        market_cap_idx = header_indexes.get("market_cap_mn_try", 4)
        free_float_idx = header_indexes.get("free_float_pct", 6)
        capital_idx = header_indexes.get("capital_mn_try", 7)
        if symbol_idx >= len(cells):
            continue
        symbol = str(cells[symbol_idx] or "").strip().upper()
        if not symbol:
            continue

        market_cap_mn_try = _parse_tr_decimal(cells[market_cap_idx]) if market_cap_idx < len(cells) else None
        free_float_pct = _parse_tr_decimal(cells[free_float_idx]) if free_float_idx < len(cells) else None
        capital_mn_try = _parse_tr_decimal(cells[capital_idx]) if capital_idx < len(cells) else None
        items[symbol] = {
            "market_cap": market_cap_mn_try * 1_000_000 if market_cap_mn_try is not None else None,
            "fdpo": round(free_float_pct / 100.0, 6) if free_float_pct is not None and free_float_pct > 0 else None,
            "shares_outstanding": capital_mn_try * 1_000_000 if capital_mn_try is not None else None,
            "source": "isyatirim_temelozet",
        }

    return items


def _fetch_isyatirim_basic_summary_map() -> Dict[str, Dict[str, Any]]:
    import urllib.error
    import urllib.request

    now = time.time()
    cached = _ISYATIRIM_BASIC_SUMMARY_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _ISYATIRIM_BASIC_SUMMARY_CACHE_TTL:
        return cached.get("items", {})

    request = urllib.request.Request(
        url=_isyatirim_basic_summary_url(),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            html_text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, Exception):
        _ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": {}}
        return {}

    items = _extract_isyatirim_basic_summary_map(html_text)
    _ISYATIRIM_BASIC_SUMMARY_CACHE["payload"] = {"_ts": now, "items": items}
    return items


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
    basic_summary_map = _fetch_isyatirim_basic_summary_map()

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
                "market_cap": _market_cap_from_quote_and_meta(quote, cached_meta, basic_summary_map.get(symbol)),
                **_empty_logo_payload(),
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
_COMMODITY_CACHE_TTL = 3  # 3 seconds

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
    ("KAHVE", "KC=F", "Kahve", "USD"),
    ("SEKER", "SB=F", "Şeker", "USD"),
    ("BUGDAY", "ZW=F", "Buğday", "USD"),
    ("MISIR", "ZC=F", "Mısır", "USD"),
    ("PAMUK", "CT=F", "Pamuk", "USD"),
    ("KAKAO", "CC=F", "Kakao", "USD"),
    ("SOYA", "ZS=F", "Soya Fasulyesi", "USD"),
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
    high = meta.get("regularMarketDayHigh")
    low = meta.get("regularMarketDayLow")
    volume = meta.get("regularMarketVolume")

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
        "high": high,
        "low": low,
        "volume": volume,
        "currency": currency,
        "market_state": market_state,
        "as_of": as_of,
    }


def _normalize_market_index(index_code: str) -> str:
    normalized = str(index_code or "").strip().upper()
    if normalized not in _MARKET_INDEX_META:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen endeks. {_supported_market_indexes_text()} kullanin.",
        )
    return normalized


def _parse_yahoo_chart_payload(data: Dict[str, Any], yahoo_symbol: str) -> Dict[str, Any]:
    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "error": f"yahoo_parse: {exc}", "yahoo_symbol": yahoo_symbol}

    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    volumes = quote.get("volume") or []

    points: List[Dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        close = closes[idx] if idx < len(closes) else None
        if close is None or not isinstance(ts, (int, float)):
            continue
        try:
            numeric_close = float(close)
        except (TypeError, ValueError):
            continue
        if numeric_close <= 0:
            continue

        def _numeric_at(values: List[Any]) -> Optional[float]:
            if idx >= len(values):
                return None
            value = values[idx]
            if value is None or isinstance(value, bool):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        points.append(
            {
                "time": datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(),
                "close": numeric_close,
                "high": _numeric_at(highs),
                "low": _numeric_at(lows),
                "volume": _numeric_at(volumes),
            }
        )

    return {
        "ok": True,
        "yahoo_symbol": yahoo_symbol,
        "meta": meta,
        "points": points,
    }


def _fetch_yahoo_chart_url(url: str, yahoo_symbol: str) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, Exception) as exc:
        return {"ok": False, "error": f"yahoo_error: {exc}", "yahoo_symbol": yahoo_symbol}

    return _parse_yahoo_chart_payload(data, yahoo_symbol)


def _fetch_yahoo_chart_raw(yahoo_symbol: str, *, interval: str, range_: str) -> Dict[str, Any]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval={interval}&range={range_}"
    )
    return _fetch_yahoo_chart_url(url, yahoo_symbol)


def _fetch_yahoo_chart_period_raw(
    yahoo_symbol: str,
    *,
    interval: str,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    period1 = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        f"?interval={interval}&period1={period1}&period2={period2}"
    )
    return _fetch_yahoo_chart_url(url, yahoo_symbol)


def _fetch_index_quote(index_code: str) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    now = time.time()
    cached = _MARKET_INDEX_QUOTE_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_INDEX_QUOTE_CACHE_TTL:
        return dict(cached.get("data") or {})

    meta = _MARKET_INDEX_META[normalized]
    errors: List[str] = []
    for yahoo_symbol in meta["yahoo_candidates"]:
        quote = _fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            row = {
                "symbol": normalized,
                "label": meta["label"],
                "yahoo_symbol": yahoo_symbol,
                "price": quote.get("price"),
                "prev_close": quote.get("prev_close"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "volume": quote.get("volume"),
                "currency": quote.get("currency") or "TRY",
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
            }
            _MARKET_INDEX_QUOTE_CACHE[normalized] = {"_ts": now, "data": row}
            return dict(row)
        errors.append(str(quote.get("error") or "quote_unavailable"))

    fallback = {
        "symbol": normalized,
        "label": meta["label"],
        "yahoo_symbol": None,
        "price": None,
        "prev_close": None,
        "change": None,
        "change_pct": None,
        "high": None,
        "low": None,
        "volume": None,
        "currency": "TRY",
        "market_state": "",
        "as_of": None,
        "error": "; ".join(errors[:3]) if errors else "quote_unavailable",
    }
    _MARKET_INDEX_QUOTE_CACHE[normalized] = {"_ts": now, "data": fallback}
    return dict(fallback)


def _fetch_index_return_bases(index_code: str) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    now = time.time()
    cached = _MARKET_INDEX_RETURN_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_INDEX_RETURN_CACHE_TTL:
        return dict(cached.get("data") or {})

    meta = _MARKET_INDEX_META[normalized]
    for yahoo_symbol in meta["yahoo_candidates"]:
        chart = _fetch_yahoo_chart_raw(yahoo_symbol, interval="1d", range_="5y")
        if not chart.get("ok"):
            continue
        points = [
            (datetime.fromisoformat(str(point["time"])), float(point["close"]))
            for point in chart.get("points", [])
            if point.get("time") and isinstance(point.get("close"), (int, float))
        ]
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        latest_dt, latest_close = points[-1]
        year_start = datetime(latest_dt.year, 1, 1, tzinfo=timezone.utc)
        bases = {
            "base_1w": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=7)),
            "base_1m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=30)),
            "base_3m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=91)),
            "base_6m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=182)),
            "base_ytd": _pick_series_value_at_or_after(points, year_start),
            "base_1y": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=365)),
            "base_5y": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=365 * 5)),
            "latest_close": latest_close,
            "as_of": latest_dt.isoformat(),
            "yahoo_symbol": yahoo_symbol,
        }
        _MARKET_INDEX_RETURN_CACHE[normalized] = {"_ts": now, "data": bases}
        return dict(bases)

    _MARKET_INDEX_RETURN_CACHE[normalized] = {"_ts": now, "data": {}}
    return {}


def _index_returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        response_field: _return_pct(current_price, return_bases.get(base_field))
        for response_field, base_field in _INDEX_RETURN_BASE_FIELDS
    }


def _market_index_row(index_code: str, *, quote: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    quote_row = dict(quote or _fetch_index_quote(normalized))
    bases = _fetch_index_return_bases(normalized)
    current_for_returns = quote_row.get("price") if quote_row.get("price") is not None else bases.get("latest_close")
    return {
        "symbol": normalized,
        "label": _MARKET_INDEX_META[normalized]["label"],
        "yahoo_symbol": quote_row.get("yahoo_symbol") or bases.get("yahoo_symbol"),
        "price": quote_row.get("price"),
        "prev_close": quote_row.get("prev_close"),
        "change": quote_row.get("change"),
        "change_pct": quote_row.get("change_pct"),
        "high": quote_row.get("high"),
        "low": quote_row.get("low"),
        "volume": quote_row.get("volume"),
        "currency": quote_row.get("currency") or "TRY",
        "market_state": quote_row.get("market_state") or "",
        "as_of": quote_row.get("as_of") or bases.get("as_of"),
        "error": quote_row.get("error"),
        **_index_returns_from_bases(current_for_returns, bases),
    }


def _comparison_error_message(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc) or exc.__class__.__name__


def _comparison_history_result(
    asset: MarketComparisonHistoryAsset,
    *,
    symbol: str,
    label: Optional[str],
    points: List[Dict[str, Any]],
    source: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": asset.id or f"{asset.kind}:{symbol}",
        "kind": asset.kind,
        "symbol": symbol,
        "label": label or asset.label or symbol,
        "points": points,
        "source": source,
        "error": error,
    }


def _comparison_history_points_from_chart(
    chart: Dict[str, Any],
    *,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for point in list(chart.get("points") or []):
        if not isinstance(point, dict):
            continue
        point_dt = _point_datetime(point.get("time"))
        close = _numeric_chart_value(point.get("close"))
        if point_dt is None or close is None or close <= 0:
            continue
        point_date = point_dt.date()
        if point_date < start_date or point_date > end_date:
            continue
        date_key = point_date.isoformat()
        deduped[date_key] = {"date": date_key, "value": close}
    return [deduped[key] for key in sorted(deduped)]


def _fund_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    from app.fund_service import get_fund_performance_payload, normalize_fund_code

    symbol = normalize_fund_code(asset.symbol)
    try:
        payload = get_fund_performance_payload(
            CONFIG.paths.processed_dir,
            symbol,
            start_date=start_date,
            end_date=end_date,
        )
        points: List[Dict[str, Any]] = []
        for point in list(payload.get("points") or []):
            try:
                price = float(point.get("price"))
            except (TypeError, ValueError):
                continue
            point_date = str(point.get("date") or "")
            if not point_date or price <= 0:
                continue
            if point_date < start_date.isoformat() or point_date > end_date.isoformat():
                continue
            points.append({"date": point_date, "value": price})
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=points,
            source=str(payload.get("source") or "sqlite"),
            error=None if points else str(payload.get("source_metadata", {}).get("warning") or payload.get("status") or "data_unavailable"),
        )
    except Exception as exc:
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="sqlite",
            error=_comparison_error_message(exc),
        )


def _stock_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    try:
        symbol = _normalize_market_stock_card_symbol(asset.symbol)
        yahoo_symbol = f"{symbol}.IS"
        chart = _fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date,
        )
        points = _comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=points,
            source="yahoo_finance_chart",
            error=None if chart.get("ok") and points else str(chart.get("error") or "data_unavailable"),
        )
    except Exception as exc:
        symbol = str(asset.symbol or "").strip().upper()
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error=_comparison_error_message(exc),
        )


def _index_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    try:
        symbol = _normalize_market_index(asset.symbol)
        label = asset.label or _MARKET_INDEX_META[symbol]["label"]
        errors: List[str] = []
        for yahoo_symbol in _MARKET_INDEX_META[symbol]["yahoo_candidates"]:
            chart = _fetch_yahoo_chart_period_raw(
                yahoo_symbol,
                interval="1d",
                start_date=start_date,
                end_date=end_date,
            )
            points = _comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
            if chart.get("ok") and points:
                return _comparison_history_result(
                    asset,
                    symbol=symbol,
                    label=label,
                    points=points,
                    source="yahoo_finance_chart",
                )
            errors.append(str(chart.get("error") or "data_unavailable"))
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=label,
            points=[],
            source="yahoo_finance_chart",
            error="; ".join(errors[:3]) if errors else "data_unavailable",
        )
    except Exception as exc:
        symbol = str(asset.symbol or "").strip().upper()
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error=_comparison_error_message(exc),
        )


def _fx_comparison_history(
    asset: MarketComparisonHistoryAsset,
    *,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    symbol = str(asset.symbol or "").strip().upper()
    direct_map = {entry[0]: entry for entry in _FX_DIRECT_MAP}
    entry = direct_map.get(symbol)
    if not entry:
        return _comparison_history_result(
            asset,
            symbol=symbol,
            label=asset.label or symbol,
            points=[],
            source="yahoo_finance_chart",
            error="unsupported_fx_symbol",
        )

    _, yahoo_candidates, default_label = entry
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        chart = _fetch_yahoo_chart_period_raw(
            yahoo_symbol,
            interval="1d",
            start_date=start_date,
            end_date=end_date,
        )
        points = _comparison_history_points_from_chart(chart, start_date=start_date, end_date=end_date)
        if chart.get("ok") and points:
            return _comparison_history_result(
                asset,
                symbol=symbol,
                label=asset.label or default_label,
                points=points,
                source="yahoo_finance_chart",
            )
        errors.append(str(chart.get("error") or "data_unavailable"))

    return _comparison_history_result(
        asset,
        symbol=symbol,
        label=asset.label or default_label,
        points=[],
        source="yahoo_finance_chart",
        error="; ".join(errors[:3]) if errors else "data_unavailable",
    )


def _market_comparison_history_payload(request: MarketComparisonHistoryRequest) -> Dict[str, Any]:
    handlers = {
        "fund": _fund_comparison_history,
        "stock": _stock_comparison_history,
        "index": _index_comparison_history,
        "fx": _fx_comparison_history,
    }
    return {
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "assets": [
            handlers[asset.kind](asset, start_date=request.start_date, end_date=request.end_date)
            for asset in request.assets
        ],
        "source": "mixed",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _market_indices_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    now = time.time()
    cached = _MARKET_INDICES_CACHE.get("payload")
    if cached and not force_refresh and now - cached.get("_ts", 0) < _MARKET_INDICES_CACHE_TTL:
        return cached["data"]

    rows = [_market_index_row(index_code) for index_code in _MARKET_STOCK_INDEX_ORDER]
    data = {
        "rows": rows,
        "source": "yahoo_finance_chart",
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _MARKET_INDICES_CACHE["payload"] = {"_ts": now, "data": data}
    return data


def _index_intraday_payload(index_code: str) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    now = time.time()
    cached = _MARKET_INDEX_INTRADAY_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _MARKET_INDEX_INTRADAY_CACHE_TTL:
        return dict(cached.get("data") or {})

    meta = _MARKET_INDEX_META[normalized]
    for yahoo_symbol in meta["yahoo_candidates"]:
        chart = _fetch_yahoo_chart_raw(yahoo_symbol, interval="5m", range_="1d")
        if chart.get("ok") and chart.get("points"):
            points = [
                {
                    "time": point["time"],
                    "open": point.get("open"),
                    "high": point.get("high"),
                    "low": point.get("low"),
                    "close": point["close"],
                }
                for point in chart.get("points", [])
                if point.get("time") and isinstance(point.get("close"), (int, float))
            ]
            highs = [
                point.get("high")
                for point in chart.get("points", [])
                if isinstance(point.get("high"), (int, float))
            ]
            lows = [
                point.get("low")
                for point in chart.get("points", [])
                if isinstance(point.get("low"), (int, float))
            ]
            meta_payload = chart.get("meta") or {}
            payload = {
                "line_points": points,
                "high": meta_payload.get("regularMarketDayHigh") or (max(highs) if highs else None),
                "low": meta_payload.get("regularMarketDayLow") or (min(lows) if lows else None),
                "prev_close": meta_payload.get("chartPreviousClose") or meta_payload.get("previousClose"),
                "yahoo_symbol": yahoo_symbol,
            }
            _MARKET_INDEX_INTRADAY_CACHE[normalized] = {"_ts": now, "data": payload}
            return dict(payload)

    fallback = {"line_points": [], "high": None, "low": None, "prev_close": None, "yahoo_symbol": None}
    _MARKET_INDEX_INTRADAY_CACHE[normalized] = {"_ts": now, "data": fallback}
    return dict(fallback)


def _latest_share_count_from_kap_cache(symbol: str) -> Optional[float]:
    cache_path = CONFIG.paths.processed_dir / "kap_cache" / f"{symbol}.json"
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    quarters_raw = payload.get("quarters")
    quarters = [q for q in quarters_raw if isinstance(q, dict)] if isinstance(quarters_raw, list) else []
    if not quarters:
        return None
    latest = sorted(quarters, key=_quarter_sort_key)[-1]
    shares = _extract_quarter_metric(latest, "odenmis_sermaye", priority=["metrics", "metrics_ytd"])
    if shares is None:
        shares = _extract_quarter_metric(latest, "cikarilmis_sermaye", priority=["metrics", "metrics_ytd"])
    if shares is None or shares <= 0:
        return None
    return shares


def _index_weight_inputs_for_symbol(symbol: str) -> Dict[str, Optional[float]]:
    basic_summary = _fetch_isyatirim_basic_summary_map().get(str(symbol or "").strip().upper(), {})
    shares = _latest_share_count_from_kap_cache(symbol)
    if shares is None:
        shares = _positive_float(basic_summary.get("shares_outstanding"))
    fdpo = _positive_float(basic_summary.get("fdpo"))
    return {
        "shares_outstanding": shares,
        "fdpo": fdpo,
        "weight_coefficient": 1.0 if fdpo is not None else None,
    }


def _apply_index_weight_formula(
    rows: List[Dict[str, Any]],
    *,
    index_level: Any,
) -> tuple[List[Dict[str, Any]], str]:
    enriched: List[Dict[str, Any]] = []
    free_float_values: List[float] = []
    for row in rows:
        price = row.get("price")
        shares = row.get("shares_outstanding")
        fdpo = row.get("fdpo")
        coefficient = row.get("weight_coefficient")
        market_cap = row.get("market_cap")
        free_float_market_value = None
        try:
            if price is not None and shares is not None and fdpo is not None and coefficient is not None:
                free_float_market_value = float(price) * float(shares) * float(fdpo) * float(coefficient)
            elif market_cap is not None and fdpo is not None and coefficient is not None:
                free_float_market_value = float(market_cap) * float(fdpo) * float(coefficient)
        except (TypeError, ValueError):
            free_float_market_value = None
        enriched_row = {
            **row,
            "free_float_market_value": free_float_market_value if free_float_market_value and free_float_market_value > 0 else None,
            "weight_pct": None,
            "point_effect": None,
        }
        enriched.append(enriched_row)
        if enriched_row["free_float_market_value"] is not None:
            free_float_values.append(float(enriched_row["free_float_market_value"]))

    if len(free_float_values) != len(enriched) or not free_float_values:
        return [
            {
                **row,
                "free_float_market_value": None,
                "weight_pct": None,
                "point_effect": None,
            }
            for row in enriched
        ], "unavailable"

    total = sum(free_float_values)
    if total <= 0:
        return enriched, "unavailable"

    try:
        level = float(index_level)
    except (TypeError, ValueError):
        level = 0.0

    calculated: List[Dict[str, Any]] = []
    for row in enriched:
        ffmv = float(row["free_float_market_value"])
        weight_pct = (ffmv / total) * 100.0
        change_pct = row.get("change_pct")
        point_effect = None
        try:
            if change_pct is not None and level > 0:
                point_effect = level * (weight_pct / 100.0) * (float(change_pct) / 100.0)
        except (TypeError, ValueError):
            point_effect = None
        calculated.append(
            {
                **row,
                "weight_pct": round(weight_pct, 4),
                "point_effect": round(point_effect, 2) if point_effect is not None else None,
            }
        )
    calculated.sort(
        key=lambda item: (
            item.get("point_effect") is None,
            -abs(float(item.get("point_effect") or 0.0)),
            str(item.get("symbol") or ""),
        )
    )
    return calculated, "available"


def _index_constituents(index_code: str, *, index_level: Any) -> tuple[List[Dict[str, Any]], str]:
    normalized = _normalize_market_index(index_code)
    stocks_payload = _market_stocks_payload(index_name=normalized)
    rows: List[Dict[str, Any]] = []
    for stock in stocks_payload.get("rows", []):
        symbol = str(stock.get("company") or "").strip().upper()
        if not symbol:
            continue
        weight_inputs = _index_weight_inputs_for_symbol(symbol)
        rows.append(
            {
                "symbol": symbol,
                "price": stock.get("price"),
                "price_currency": stock.get("price_currency"),
                "change_pct": stock.get("change_pct"),
                "volume": stock.get("volume"),
                "market_cap": stock.get("market_cap"),
                "logo_url": stock.get("logo_url"),
                "logo_source": stock.get("logo_source"),
                "shares_outstanding": weight_inputs.get("shares_outstanding"),
                "fdpo": weight_inputs.get("fdpo"),
                "weight_coefficient": weight_inputs.get("weight_coefficient"),
            }
        )
    weighted_rows, weight_status = _apply_index_weight_formula(rows, index_level=index_level)
    if weight_status != "available":
        weighted_rows.sort(
            key=lambda item: (
                item.get("change_pct") is None,
                -abs(float(item.get("change_pct") or 0.0)),
                str(item.get("symbol") or ""),
            )
        )
    return weighted_rows, weight_status


def _market_index_detail_payload(index_code: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    normalized = _normalize_market_index(index_code)
    now = time.time()
    cached = _MARKET_INDEX_DETAIL_CACHE.get(normalized)
    if cached and not force_refresh and now - cached.get("_ts", 0) < _MARKET_INDEX_DETAIL_CACHE_TTL:
        return cached["data"]

    quote = _fetch_index_quote(normalized)
    row = _market_index_row(normalized, quote=quote)
    intraday = _index_intraday_payload(normalized)
    constituents, weight_status = _index_constituents(normalized, index_level=row.get("price"))
    data = {
        **row,
        "high": row.get("high") if row.get("high") is not None else intraday.get("high"),
        "low": row.get("low") if row.get("low") is not None else intraday.get("low"),
        "prev_close": row.get("prev_close") if row.get("prev_close") is not None else intraday.get("prev_close"),
        "line_points": intraday.get("line_points") or [],
        "constituents": constituents,
        "weight_status": weight_status,
        "weight_note": (
            "Tahmini ağırlık: İş Yatırım halka açıklık oranı ve ağırlık katsayısı 1 varsayımıyla hesaplandı."
            if weight_status == "available"
            else "Ağırlık verisi bulunamadı: pay sayısı, FDPO ve ağırlık katsayısı eksiksiz olmadığı için puan etkisi hesaplanamadı."
        ),
        "source": "yahoo_finance_chart",
        "as_of": row.get("as_of") or datetime.now(timezone.utc).isoformat(),
    }
    _MARKET_INDEX_DETAIL_CACHE[normalized] = {"_ts": now, "data": data}
    return data


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
            "logo_url": None,
            "logo_source": None,
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
_FX_CACHE_TTL = 3
_FX_RETURN_CACHE: Dict[str, Any] = {}
_FX_RETURN_CACHE_TTL = 15 * 60

_FX_DIRECT_MAP: List[tuple[str, List[str], str]] = [
    ("USD/TRY", ["USDTRY=X"], "Amerikan Doları / TL"),
    ("EUR/TRY", ["EURTRY=X"], "Euro / TL"),
    ("GBP/TRY", ["GBPTRY=X"], "İngiliz Sterlini / TL"),
    ("CHF/TRY", ["CHFTRY=X"], "İsviçre Frangı / TL"),
    ("AUD/TRY", ["AUDTRY=X"], "Avustralya Doları / TL"),
    ("CAD/TRY", ["CADTRY=X"], "Kanada Doları / TL"),
    ("JPY/TRY", ["JPYTRY=X"], "Japon Yeni / TL"),
    ("EUR/USD", ["EURUSD=X"], "Euro / Dolar"),
    ("GBP/USD", ["GBPUSD=X"], "Sterlin / Dolar"),
    ("USD/JPY", ["USDJPY=X", "JPY=X"], "Dolar / Japon Yeni"),
    ("EUR/JPY", ["EURJPY=X"], "Euro / Japon Yeni"),
    ("GBP/JPY", ["GBPJPY=X"], "Sterlin / Japon Yeni"),
    ("USD/CNY", ["USDCNY=X", "CNY=X"], "Dolar / Çin Yuanı"),
    ("EUR/CNY", ["EURCNY=X"], "Euro / Çin Yuanı"),
    ("GBP/CNY", ["GBPCNY=X"], "Sterlin / Çin Yuanı"),
    ("CNY/JPY", ["CNYJPY=X"], "Çin Yuanı / Japon Yeni"),
    ("CHF/JPY", ["CHFJPY=X"], "İsviçre Frangı / Japon Yeni"),
    ("DXY", ["DX-Y.NYB"], "Dolar Endeksi"),
]

_FX_DERIVED_MAP: List[tuple[str, str, str, str]] = [
    ("CNY/TRY", "Çin Yuanı / TL", "USD/TRY", "USD/CNY"),
]

_FX_ORDER: List[str] = [
    "USD/TRY",
    "EUR/TRY",
    "GBP/TRY",
    "CHF/TRY",
    "AUD/TRY",
    "CAD/TRY",
    "JPY/TRY",
    "CNY/TRY",
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "EUR/JPY",
    "GBP/JPY",
    "USD/CNY",
    "EUR/CNY",
    "GBP/CNY",
    "CNY/JPY",
    "CHF/JPY",
    "DXY",
]


def _fx_quote_currency(symbol: str) -> str:
    if "/" not in symbol:
        return ""
    return str(symbol or "").rsplit("/", 1)[-1].strip().upper()


def _fetch_fx_return_bases(yahoo_symbol: str) -> Dict[str, Any]:
    normalized = str(yahoo_symbol or "").strip()
    if not normalized:
        return {}
    now = time.time()
    cached = _FX_RETURN_CACHE.get(normalized)
    if cached and now - cached.get("_ts", 0) < _FX_RETURN_CACHE_TTL:
        return dict(cached.get("data") or {})

    chart = _fetch_yahoo_chart_raw(normalized, interval="1d", range_="1y")
    if not chart.get("ok"):
        _FX_RETURN_CACHE[normalized] = {"_ts": now, "data": {}}
        return {}

    points = [
        (datetime.fromisoformat(str(point["time"])), float(point["close"]))
        for point in chart.get("points", [])
        if point.get("time") and isinstance(point.get("close"), (int, float))
    ]
    if not points:
        _FX_RETURN_CACHE[normalized] = {"_ts": now, "data": {}}
        return {}

    points.sort(key=lambda item: item[0])
    latest_dt, latest_close = points[-1]
    year_start = datetime(latest_dt.year, 1, 1, tzinfo=timezone.utc)
    data = {
        "base_1w": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=7)),
        "base_1m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=30)),
        "base_3m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=91)),
        "base_6m": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=182)),
        "base_ytd": _pick_series_value_at_or_after(points, year_start),
        "base_1y": _pick_series_value_at_or_before(points, latest_dt - timedelta(days=365)),
        "latest_close": latest_close,
        "as_of": latest_dt.isoformat(),
    }
    _FX_RETURN_CACHE[normalized] = {"_ts": now, "data": data}
    return dict(data)


def _fx_returns_from_bases(current_price: Any, return_bases: Dict[str, Any]) -> Dict[str, Optional[float]]:
    current_for_returns = current_price if current_price is not None else return_bases.get("latest_close")
    return {
        response_field: _return_pct(current_for_returns, return_bases.get(base_field))
        for response_field, base_field in _RETURN_BASE_FIELDS
    }


def _fx_item_from_quote(symbol: str, yahoo_symbol: str, label: str, quote: Dict[str, Any], return_bases: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    period_returns = _fx_returns_from_bases(quote.get("price"), return_bases or {}) if quote.get("ok") else {}
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": yahoo_symbol,
        "price": quote.get("price") if quote.get("ok") else None,
        "prev_close": quote.get("prev_close") if quote.get("ok") else None,
        "change": quote.get("change") if quote.get("ok") else None,
        "change_pct": quote.get("change_pct") if quote.get("ok") else None,
        "currency": _fx_quote_currency(symbol),
        "market_state": quote.get("market_state") if quote.get("ok") else "",
        "as_of": quote.get("as_of") if quote.get("ok") else None,
        "error": None if quote.get("ok") else quote.get("error"),
        "logo_url": None,
        "logo_source": None,
        **period_returns,
    }


def _fx_direct_item(entry: tuple[str, List[str], str]) -> Dict[str, Any]:
    symbol, yahoo_candidates, label = entry
    errors: List[str] = []
    for yahoo_symbol in yahoo_candidates:
        quote = _fetch_yahoo_quote(yahoo_symbol)
        if quote.get("ok") and quote.get("price") is not None:
            return_bases = _fetch_fx_return_bases(yahoo_symbol) if symbol.endswith("/TRY") else None
            return _fx_item_from_quote(symbol, yahoo_symbol, label, quote, return_bases)
        errors.append(str(quote.get("error") or "quote_unavailable"))

    return _fx_item_from_quote(
        symbol,
        yahoo_candidates[0] if yahoo_candidates else "",
        label,
        {"ok": False, "error": "; ".join(errors[:3]) if errors else "quote_unavailable"},
    )


def _positive_number(raw: Any) -> Optional[float]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _fx_derived_item(
    symbol: str,
    label: str,
    numerator_symbol: str,
    denominator_symbol: str,
    items_by_symbol: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    numerator = items_by_symbol.get(numerator_symbol, {})
    denominator = items_by_symbol.get(denominator_symbol, {})

    numerator_price = _positive_number(numerator.get("price"))
    denominator_price = _positive_number(denominator.get("price"))
    numerator_prev = _positive_number(numerator.get("prev_close"))
    denominator_prev = _positive_number(denominator.get("prev_close"))

    price = numerator_price / denominator_price if numerator_price is not None and denominator_price is not None else None
    prev_close = numerator_prev / denominator_prev if numerator_prev is not None and denominator_prev is not None else None
    change = None
    change_pct = None
    if price is not None and prev_close is not None and prev_close > 0:
        change = round(price - prev_close, 6)
        change_pct = round((change / prev_close) * 100, 4)

    source_symbol = "/".join(
        item
        for item in (
            str(numerator.get("yahoo_symbol") or numerator_symbol),
            str(denominator.get("yahoo_symbol") or denominator_symbol),
        )
        if item
    )
    return {
        "symbol": symbol,
        "label": label,
        "yahoo_symbol": source_symbol,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": _fx_quote_currency(symbol),
        "market_state": numerator.get("market_state") or denominator.get("market_state") or "",
        "as_of": numerator.get("as_of") or denominator.get("as_of"),
        "error": None if price is not None else "derived_quote_unavailable",
        "logo_url": None,
        "logo_source": None,
        "return_1w_pct": None,
        "return_1m_pct": None,
        "return_3m_pct": None,
        "return_6m_pct": None,
        "return_ytd_pct": None,
        "return_1y_pct": None,
    }


def _market_fx_payload() -> Dict[str, Any]:
    now = time.time()
    cached = _FX_CACHE.get("payload")
    if cached and now - cached.get("_ts", 0) < _FX_CACHE_TTL:
        return cached["data"]

    from concurrent.futures import ThreadPoolExecutor

    items_by_symbol: Dict[str, Dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row in pool.map(_fx_direct_item, _FX_DIRECT_MAP):
                items_by_symbol[str(row.get("symbol") or "")] = row
    except Exception:
        for entry in _FX_DIRECT_MAP:
            row = _fx_direct_item(entry)
            items_by_symbol[str(row.get("symbol") or "")] = row

    for symbol, label, numerator_symbol, denominator_symbol in _FX_DERIVED_MAP:
        items_by_symbol[symbol] = _fx_derived_item(
            symbol,
            label,
            numerator_symbol,
            denominator_symbol,
            items_by_symbol,
        )

    items = [
        items_by_symbol[symbol]
        for symbol in _FX_ORDER
        if symbol in items_by_symbol
    ]

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
_WATCH_CACHE_TTL = 3
_WATCH_GLOBAL_CACHE: Dict[str, Any] = {}
_WATCH_GLOBAL_CACHE_TTL = 60
_WATCH_DELAY_NOTE = "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk)."

_WATCH_INDEX_CANDIDATES: List[tuple[str, str, List[str]]] = [
    ("XUTUM", "BIST Tüm", ["XUTUM.IS", "^XUTUM", "XUTUM"]),
    ("XU100", "BIST 100", ["XU100.IS", "^XU100", "XU100"]),
    ("XU030", "BIST 30", ["XU030.IS", "^XU030", "XU030"]),
]

_WATCH_GLOBAL_INDEX_CANDIDATES: List[tuple[str, str, List[str]]] = [
    ("SP500", "S&P 500", ["^GSPC", "SPY"]),
    ("NASDAQ", "Nasdaq", ["^IXIC", "QQQ"]),
    ("DOW", "Dow Jones", ["^DJI", "DIA"]),
    ("DAX", "DAX", ["^GDAXI", "DAX"]),
    ("FTSE", "FTSE 100", ["^FTSE", "ISF.L"]),
    ("NIKKEI", "Nikkei 225", ["^N225", "1321.T"]),
    ("HANGSENG", "Hang Seng", ["^HSI", "2800.HK"]),
    ("CAC40", "CAC 40", ["^FCHI", "CAC.PA"]),
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
        "logo_url": None,
        "logo_source": None,
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
        "logo_url": item.get("logo_url"),
        "logo_source": item.get("logo_source"),
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


def _watch_index_item(
    symbol: str,
    label: str,
    yahoo_candidates: List[str],
    *,
    fallback_currency: str = "TRY",
) -> Dict[str, Any]:
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
                "currency": quote.get("currency") or fallback_currency,
                "market_state": quote.get("market_state") or "",
                "as_of": quote.get("as_of"),
                "error": None,
                "logo_url": None,
                "logo_source": None,
            }
        err = str(quote.get("error") or "quote_unavailable")
        errors.append(f"{yahoo_symbol}:{err}")

    return _empty_watch_item(
        symbol=symbol,
        label=label,
        currency=fallback_currency,
        error="; ".join(errors[:3]) if errors else "quote_unavailable",
    )


def _market_watch_global_payload(*, force_refresh: bool = False) -> Dict[str, Any]:
    now = time.time()
    cached = _WATCH_GLOBAL_CACHE.get("payload")
    if cached and not force_refresh and now - cached.get("_ts", 0) < _WATCH_GLOBAL_CACHE_TTL:
        return cached["data"]

    from concurrent.futures import ThreadPoolExecutor

    def _one(entry: tuple[str, str, List[str]]) -> Dict[str, Any]:
        symbol, label, yahoo_candidates = entry
        return _watch_index_item(
            symbol=symbol,
            label=label,
            yahoo_candidates=yahoo_candidates,
            fallback_currency="",
        )

    items: List[Dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=min(8, len(_WATCH_GLOBAL_INDEX_CANDIDATES))) as pool:
            items = list(pool.map(_one, _WATCH_GLOBAL_INDEX_CANDIDATES))
    except Exception:
        items = [_one(entry) for entry in _WATCH_GLOBAL_INDEX_CANDIDATES]

    data = {
        "items": items,
        "source": "yahoo_finance_chart",
        "delay_note": _WATCH_DELAY_NOTE,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    _WATCH_GLOBAL_CACHE["payload"] = {"_ts": now, "data": data}
    return data


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


@app.get("/market/watch/global")
def market_watch_global(refresh: bool = Query(False)) -> Dict[str, Any]:
    return _market_watch_global_payload(force_refresh=refresh)
