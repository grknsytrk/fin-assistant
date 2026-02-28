from __future__ import annotations

import html
import json
import math
import re
import sys
import textwrap
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest import (
    ingest_raw_pdfs,
    list_pdf_files,
    parse_company_from_name,
    parse_quarter_from_name,
    parse_year_from_name,
)
from src.query_parser import parse_query
from src.config import AppConfig, load_config
from src.commentary import generate_commentary
from src.kap_fetcher import fetch_kap_company_snapshot
from src.ratio_engine import (
    build_ratio_table,
    build_executive_summary,
    detect_company_mentions,
    detect_last_quarter_changes,
    is_cross_company_query,
    run_cross_company_comparison,
)
from app.ui_components import trust_badge_html, trust_level

MIN_PY = (3, 9)
MAX_PY_EXCLUSIVE = (3, 13)
CONFIG: AppConfig = load_config(ROOT / "config.yaml")
DEFAULT_COLLECTION_NAME = CONFIG.chroma.collection_v1
DEFAULT_COLLECTION_NAME_V2 = CONFIG.chroma.collection_v2
RAW_DIR = CONFIG.paths.raw_dir
PROCESSED_DIR = CONFIG.paths.processed_dir
PAGES_FILE = CONFIG.paths.pages_file
CHUNKS_V1_FILE = CONFIG.paths.chunks_v1_file
CHUNKS_V2_FILE = CONFIG.paths.chunks_v2_file
CHROMA_DIR = CONFIG.chroma.dir
DEFAULT_DETAILED_OUTPUT = CONFIG.evaluation.detailed_output
DEFAULT_SUMMARY_OUTPUT = CONFIG.evaluation.summary_output
UI_LOG_FILE = CONFIG.paths.ui_log_file
FEEDBACK_FILE = CONFIG.paths.processed_dir / "feedback.jsonl"
INGEST_LOG_FILE = CONFIG.paths.processed_dir / "ingest_logs.jsonl"
RETRIEVER_HELP = {
    "v6": "v6 - cross-encoder rerank (en agir, daha hassas)",
    "v5": "v5 - hybrid (vector + BM25)",
    "v4": "v4 - BM25 lexical retrieval",
    "v3": "v3 (Onerilen) - query parser + akilli boost + otomatik ceyrek filtresi",
    "v2": "v2 - vector retrieval + lexical rerank",
    "v1": "v1 - temel vector retrieval",
}
EXAMPLE_QUESTIONS = [
    "2025 ucuncu ceyrek net kar kac?",
    "ilk yariyilda FAVOK marji yuzde kac?",
    "Q1 Q2 Q3 net kar trendi nasil?",
    "dokuz aylik satislar gecen yila gore artmis mi?",
    "riskler neler?",
    "2025 ikinci ceyrek magaza sayisi kac?",
    "uzay madenciligi gelir hedefi nedir?",
]
ASK_NOT_FOUND_SUGGESTIONS = [
    "2025 ucuncu ceyrek net kar kac?",
    "Q1 Q2 Q3 net kar trendi nasil?",
    "2025 ikinci ceyrek magaza sayisi kac?",
]
AI_MODEL_OPTIONS = [
    "arcee-ai/trinity-large-preview:free",
    "stepfun/step-3.5-flash:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-235b-a22b-thinking-2507",
]
AI_MODEL_CUSTOM_SENTINEL = "Custom..."
KAP_BANK_TICKERS = {"AKBNK", "GARAN", "HALKB", "ISCTR", "TSKB", "VAKBN", "YKBNK"}
KAP_DEFAULT_BIST30_COMPANIES = [
    "AEFES",
    "AKBNK",
    "ALARK",
    "ARCLK",
    "ASELS",
    "ASTOR",
    "BIMAS",
    "BRSAN",
    "BTCIM",
    "CCOLA",
    "CIMSA",
    "DOAS",
    "DOHOL",
    "DSTKF",
    "EKGYO",
    "ENKAI",
    "EREGL",
    "FROTO",
    "GARAN",
    "GUBRF",
    "HALKB",
    "HEKTS",
    "ISCTR",
    "KCHOL",
    "KONTR",
    "KRDMD",
    "KUYAS",
    "MAVI",
    "MGROS",
    "MIATK",
    "OYAKC",
    "PASEU",
    "PETKM",
    "PGSUS",
    "SAHOL",
    "SASA",
    "SISE",
    "SOKM",
    "TAVHL",
    "TCELL",
    "THYAO",
    "TOASO",
    "TRALT",
    "TRMET",
    "TSKB",
    "TTKOM",
    "TUPRS",
    "ULKER",
    "VAKBN",
    "YKBNK",
]

DEFAULT_UI_SETTINGS: Dict[str, Any] = {
    "retriever": "v3",
    "top_k_final": int(CONFIG.retrieval.top_k_final),
    "top_k_initial_v2": int(CONFIG.retrieval.v2_top_k_initial),
    "top_k_initial_v3": int(CONFIG.retrieval.v3_top_k_initial),
    "top_k_vector_v5": int(CONFIG.retrieval.v5_top_k_vector),
    "top_k_bm25_v5": int(CONFIG.retrieval.v5_top_k_bm25),
    "top_k_candidates_v6": int(CONFIG.retrieval.v6_cross_top_n),
    "alpha_v2": float(CONFIG.retrieval.alpha_v2),
    "alpha_v3": float(CONFIG.retrieval.alpha_v3),
    "beta_v5": float(CONFIG.retrieval.beta_v5),
    "show_debug_candidates": False,
    "llm_model": str(
        getattr(getattr(CONFIG, "llm_assistant", None) or getattr(CONFIG, "llm_commentary", None), "model", "")
        or "arcee-ai/trinity-large-preview:free"
    ),
    "llm_model_custom": "",
}


def _ensure_paths() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    UI_LOG_FILE.touch(exist_ok=True)
    FEEDBACK_FILE.touch(exist_ok=True)
    INGEST_LOG_FILE.touch(exist_ok=True)


def _get_ui_settings() -> Dict[str, Any]:
    if "ui_settings" not in st.session_state:
        st.session_state["ui_settings"] = dict(DEFAULT_UI_SETTINGS)
    settings = st.session_state["ui_settings"]
    llm_model = str(settings.get("llm_model", "")).strip()
    if not llm_model:
        llm_model = str(DEFAULT_UI_SETTINGS["llm_model"])
    settings["llm_model"] = llm_model
    if llm_model not in AI_MODEL_OPTIONS and not str(settings.get("llm_model_custom", "")).strip():
        settings["llm_model_custom"] = llm_model
    return settings


def _append_ingest_log(action: str, status: str, details: Dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "status": status,
        "details": details,
    }
    with INGEST_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_recent_ingest_logs(limit: int = 20) -> List[Dict[str, Any]]:
    if not INGEST_LOG_FILE.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with INGEST_LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return list(reversed(rows[-limit:]))


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_supported_python() -> bool:
    current = sys.version_info[:2]
    return MIN_PY <= current < MAX_PY_EXCLUSIVE


def _python_version_string() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _runtime_help_message() -> str:
    return (
        "Bu UI icin Python 3.9-3.12 gerekiyor.\n"
        f"Mevcut surum: {_python_version_string()}\n\n"
        "Ornek calistirma:\n"
        ".\\.venv39\\Scripts\\python.exe -m streamlit run app/ui.py"
    )


def _render_quick_start_card() -> None:
    st.info(
        "Hizli baslangic:\n"
        "1) Data sekmesinde PDF yukleyin (gerekirse)\n"
        "2) Ingest ve Index v2 calistirin\n"
        "3) Ask sekmesinde soruyu yazip 'Yanit Uret' tusuna basin"
    )


def _collection_count(collection_name: str) -> Optional[int]:
    try:
        import chromadb
    except Exception:
        return None

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=collection_name)
        return int(collection.count())
    except Exception:
        return 0


def _data_stats() -> Dict[str, Optional[int]]:
    pdf_files = list_pdf_files(RAW_DIR)
    return {
        "pdf_count": len(pdf_files),
        "page_count": _count_jsonl_rows(PAGES_FILE),
        "chunk_count_v1": _count_jsonl_rows(CHUNKS_V1_FILE),
        "chunk_count_v2": _count_jsonl_rows(CHUNKS_V2_FILE),
        "collection_count_v1": _collection_count(DEFAULT_COLLECTION_NAME),
        "collection_count_v2": _collection_count(DEFAULT_COLLECTION_NAME_V2),
    }


def _available_companies() -> List[str]:
    raw_companies: List[str] = []
    if CHUNKS_V2_FILE.exists():
        seen = set()
        with CHUNKS_V2_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                company = str(payload.get("company", "")).strip().upper()
                if company and company not in seen:
                    seen.add(company)
                    raw_companies.append(company)
    if not raw_companies:
        return ["BIM"]

    # Keep raw values for filter compatibility, but collapse visually equivalent names
    # (e.g. NETCAD_4Q -> NETCAD) so selector labels stay clean.
    canonical_to_raw: Dict[str, str] = {}
    for raw_company in raw_companies:
        canonical = _company_display_name(raw_company)
        current = canonical_to_raw.get(canonical)
        if current is None or len(raw_company) < len(current):
            canonical_to_raw[canonical] = raw_company
    return [canonical_to_raw[key] for key in sorted(canonical_to_raw.keys())]


def _company_display_name(company: Any) -> str:
    raw = str(company or "").strip()
    if not raw:
        return "-"
    if raw.upper() == "TUMU":
        return "TUMU"
    parsed = parse_company_from_name(raw)
    parsed_norm = str(parsed or "").strip().upper()
    if parsed_norm and parsed_norm != "UNKNOWN":
        return parsed_norm
    return raw.upper()


def _build_indexed_docs_rows() -> List[Dict[str, Any]]:
    page_counts: Dict[str, int] = {}
    if PAGES_FILE.exists():
        with PAGES_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                doc_id = str(row.get("doc_id", "")).strip()
                if not doc_id:
                    continue
                page_counts[doc_id] = page_counts.get(doc_id, 0) + 1

    indexed_docs = set()
    if CHUNKS_V2_FILE.exists():
        with CHUNKS_V2_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                doc_id = str(row.get("doc_id", "")).strip()
                if doc_id:
                    indexed_docs.add(doc_id)

    rows: List[Dict[str, Any]] = []
    for pdf in sorted(list_pdf_files(RAW_DIR), key=lambda p: p.name.lower()):
        stem = pdf.stem
        company = parse_company_from_name(stem)
        quarter = parse_quarter_from_name(stem)
        year = parse_year_from_name(stem)
        page_count = page_counts.get(stem, 0)
        indexed_at = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        status = "Indexed" if stem in indexed_docs else ("Ingested" if page_count > 0 else "Raw")
        rows.append(
            {
                "company": company or "-",
                "year": year or "-",
                "quarter": quarter or "-",
                "filename": pdf.name,
                "pages": page_count,
                "indexed_at": indexed_at,
                "status": status,
            }
        )
    return rows


def _save_uploaded_pdfs(files: Sequence[object]) -> List[str]:
    saved: List[str] = []
    for file_obj in files:
        file_name = getattr(file_obj, "name", "").strip()
        if not file_name.lower().endswith(".pdf"):
            continue
        target = RAW_DIR / Path(file_name).name
        target.write_bytes(file_obj.getbuffer())
        saved.append(target.name)
    return saved


def _detect_pdf_metadata_from_name(filename: str) -> Dict[str, object]:
    stem = Path(filename).stem
    return {
        "file": filename,
        "company": parse_company_from_name(stem),
        "quarter": parse_quarter_from_name(stem),
        "year": parse_year_from_name(stem),
    }


def _has_indexed_reports() -> bool:
    stats = _data_stats()
    collection_count_v2 = stats.get("collection_count_v2")
    if isinstance(collection_count_v2, int):
        return collection_count_v2 > 0
    return any(row.get("status") == "Indexed" for row in _build_indexed_docs_rows())


def _find_sample_pdf() -> Optional[Path]:
    candidates: List[Path] = []
    for directory in (ROOT / "data" / "samples", ROOT / "data" / "sample", RAW_DIR):
        if not directory.exists():
            continue
        for pdf in sorted(directory.glob("*.pdf"), key=lambda p: p.name.lower()):
            candidates.append(pdf)
    return candidates[0] if candidates else None


def _run_ingest_and_index_v2(saved_files: Optional[Sequence[str]] = None) -> Tuple[bool, str]:
    try:
        used_incremental = True
        try:
            from src.index import build_index_v2_incremental as _index_v2_fn
        except Exception:
            # Backward compatibility: if running process has old module state,
            # gracefully fallback to full v2 rebuild instead of crashing.
            from src.index import build_index_v2 as _index_v2_fn

            used_incremental = False

        index_summary = _index_v2_fn(
            raw_dir=RAW_DIR,
            processed_dir=PROCESSED_DIR,
            collection_name=DEFAULT_COLLECTION_NAME_V2,
            chunk_size=CONFIG.chunking.v2.chunk_size,
            overlap=CONFIG.chunking.v2.overlap,
        )
        st.session_state["last_index_v2_summary"] = index_summary
        st.session_state["last_ingest_summary"] = {
            "mode": "incremental" if used_incremental else "full_v2_fallback",
            "num_pdfs": index_summary.get("num_pdfs", 0),
            "pdf_files": index_summary.get("pdf_files", []),
            "pages_per_pdf": index_summary.get("pages_per_pdf", {}),
            "total_pages": index_summary.get("total_pages", 0),
            "companies": index_summary.get("companies", []),
            "changed_files": index_summary.get("changed_files", []),
            "removed_files": index_summary.get("removed_files", []),
        }
        _clear_cached_retrievers()
        changed_files = list(index_summary.get("changed_files", []))
        removed_files = list(index_summary.get("removed_files", []))
        if used_incremental:
            if changed_files or removed_files:
                message = (
                    f"Incremental index tamamlandi. Degisen dosya: {len(changed_files)}"
                    f", silinen dosya: {len(removed_files)}."
                )
            else:
                message = "Incremental index tamamlandi. Degisen dosya bulunamadi."
        else:
            message = "Incremental import bulunamadi; full reindex v2 ile tamamlandi."
        _append_ingest_log(
            action="ingest_index_v2",
            status="ok",
            details={
                "mode": "incremental" if used_incremental else "full_v2_fallback",
                "saved_files": list(saved_files or []),
                "changed_files": changed_files,
                "removed_files": removed_files,
                "pages": index_summary.get("total_pages", 0),
                "collection_count": index_summary.get("collection_count"),
            },
        )
        return True, message
    except Exception as exc:
        _append_ingest_log(
            action="ingest_index_v2",
            status="error",
            details={"saved_files": list(saved_files or []), "error": str(exc)},
        )
        return False, f"Islem basarisiz: {exc}"


def _post_feedback_to_api(payload: Dict[str, Any], endpoint: str = "http://127.0.0.1:8000/feedback") -> bool:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            return 200 <= int(getattr(response, "status", 0)) < 300
    except Exception:
        return False


def _commentary_has_content(commentary: Dict[str, Any]) -> bool:
    if not isinstance(commentary, dict):
        return False
    headline = str(commentary.get("headline", "")).strip()
    risk_note = str(commentary.get("risk_note", "")).strip()
    next_question = str(commentary.get("next_question", "")).strip()
    bullets_raw = commentary.get("bullets", [])
    bullets: List[str] = []
    if isinstance(bullets_raw, list):
        bullets = [str(item).strip() for item in bullets_raw if str(item).strip()]
    return bool(headline or risk_note or next_question or bullets)


def _post_commentary_to_api(
    *,
    question: str,
    answer_payload: Dict[str, Any],
    company: Optional[str] = None,
    year: Optional[str] = None,
    quarter: Optional[str] = None,
    model_override: Optional[str] = None,
    endpoint: str = "http://127.0.0.1:8000/commentary",
) -> Dict[str, Any]:
    found = bool(answer_payload.get("found", (answer_payload.get("answer") or {}).get("found")))
    if not found:
        return {}

    requested_model = str(model_override or "").strip()

    def _finalize_commentary_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        result = dict(payload)
        if requested_model and not str(result.get("_model", "")).strip():
            result["_model"] = requested_model
        if _commentary_has_content(result):
            return result
        if not str(result.get("_error", "")).strip():
            result["_error"] = "Bu model bu denemede yorum uretemedi. Farkli bir modelle tekrar deneyin."
        return result

    if bool(st.session_state.get("commentary_api_unreachable", False)):
        try:
            return _finalize_commentary_payload(
                generate_commentary(
                    answer_payload=answer_payload,
                    question=question,
                    cfg=CONFIG,
                    company=company,
                    year=year,
                    quarter=quarter,
                    model_override=model_override,
                )
            )
        except Exception:
            pass  # fall through to API attempt / local fallback

    def _local_commentary_fallback() -> Dict[str, Any]:
        try:
            payload = generate_commentary(
                answer_payload=answer_payload,
                question=question,
                cfg=CONFIG,
                company=company,
                year=year,
                quarter=quarter,
                model_override=model_override,
            )
            return _finalize_commentary_payload(payload)
        except TypeError:
            # Backward compatibility for older generate_commentary(commentary_input, cfg) signatures.
            try:
                payload = generate_commentary(answer_payload, CONFIG)
                return _finalize_commentary_payload(payload)
            except Exception:
                return _finalize_commentary_payload({})
        except Exception:
            return _finalize_commentary_payload({})

    body = json.dumps(
        {
            "question": question,
            "company": company,
            "year": year,
            "quarter": quarter,
            "model": model_override,
            "answer_payload": answer_payload,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1.2) as response:
            if int(getattr(response, "status", 0)) < 200 or int(getattr(response, "status", 0)) >= 300:
                return _local_commentary_fallback()
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            if isinstance(payload, dict):
                st.session_state["commentary_api_unreachable"] = False
                return _finalize_commentary_payload(payload)
            return _local_commentary_fallback()
    except urllib.error.URLError:
        st.session_state["commentary_api_unreachable"] = True
        return _local_commentary_fallback()
    except Exception:
        return _local_commentary_fallback()


def _qa_to_commentary_answer_payload(
    qa: Dict[str, Any],
    *,
    primary_answer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mode = str(qa.get("mode", "standard"))
    parsed = dict(qa.get("parsed") or {})
    chunks = list(qa.get("chunks") or [])

    inferred_company = str(qa.get("company") or "").strip().upper()
    if not inferred_company and chunks:
        inferred_company = str(getattr(chunks[0], "company", "") or "").strip().upper()
    if not inferred_company and chunks:
        inferred_company = parse_company_from_name(str(getattr(chunks[0], "doc_id", "")).strip())
    inferred_company = inferred_company or None

    inferred_quarter = str(parsed.get("quarter") or "").strip()
    if not inferred_quarter and chunks:
        inferred_quarter = str(getattr(chunks[0], "quarter", "") or "").strip()
    inferred_quarter = inferred_quarter or None

    payload: Dict[str, Any] = {
        "found": bool(qa.get("found")),
        "parsed": parsed,
        "company": inferred_company,
    }
    if mode == "cross_company":
        result = dict(qa.get("cross_company_result") or {})
        payload["answer"] = {
            "found": bool(result.get("found", qa.get("found"))),
            "verify_status": result.get("verify_status"),
            "bullets": [str(result.get("message", "Karsilastirma tamamlandi."))],
        }
        payload["comparison"] = {
            "target": result.get("target"),
            "best_company": result.get("best_company"),
            "best_value": result.get("best_value"),
            "best_confidence": result.get("best_confidence"),
        }
        payload["evidence"] = list(result.get("evidence") or [])[:8]
        return payload

    if mode == "comparison":
        result = dict(qa.get("comparison_result") or {})
        payload["answer"] = {
            "found": bool(result.get("found", qa.get("found"))),
            "verify_status": result.get("verify_status"),
            "bullets": [line.replace("- ", "").strip() for line in _comparison_markdown_lines(result)[:6]],
        }
        payload["comparison"] = {"target": result.get("target_metric"), "period": result.get("period")}
        payload["evidence"] = list(result.get("records") or [])[:8]
        return payload

    answer_text = str(qa.get("answer_text", ""))
    bullets = _summary_items_for_display(answer_text)[:6]
    chunks = chunks[:8]
    evidence: List[Dict[str, Any]] = []
    for chunk in chunks:
        evidence.append(
            {
                "doc_id": str(getattr(chunk, "doc_id", "")),
                "company": str(getattr(chunk, "company", "")),
                "quarter": str(getattr(chunk, "quarter", "")),
                "year": str(getattr(chunk, "year", "")),
                "page": getattr(chunk, "page", None),
                "section_title": str(getattr(chunk, "section_title", "")),
                "excerpt": _short_excerpt(str(getattr(chunk, "text", "")), max_chars=260),
                "confidence": getattr(chunk, "final_score", None) or getattr(chunk, "score", None),
            }
        )
    payload["answer"] = {"found": bool(qa.get("found")), "verify_status": "WARN", "bullets": bullets}
    payload["evidence"] = evidence
    if inferred_quarter:
        payload["parsed"]["quarter"] = inferred_quarter

    if primary_answer:
        metric_map = {
            "Net kar": "net_kar",
            "Net zarar": "net_kar",
            "Brut kar": "brut_kar",
            "Satis Gelirleri": "satis_gelirleri",
            "FAVOK": "favok",
            "Net marj": "net_margin",
            "FAVOK marji": "favok_margin",
            "Brut kar marji": "brut_kar_marji",
            "Magaza sayisi": "magaza_sayisi",
            "Faaliyet nakit akisi": "faaliyet_nakit_akisi",
            "CAPEX": "capex",
            "Serbest nakit akisi": "serbest_nakit_akisi",
        }
        label = str(primary_answer.get("label", "")).strip()
        metric_key = metric_map.get(label)
        value_text = str(primary_answer.get("value", "")).strip()
        source_chunk = primary_answer.get("source")
        if metric_key and value_text:
            payload["metrics"] = {metric_key: value_text}
        if source_chunk is not None:
            payload["answer"]["source_doc_id"] = str(getattr(source_chunk, "doc_id", ""))
            payload["answer"]["source_quarter"] = str(getattr(source_chunk, "quarter", ""))
            payload["answer"]["source_company"] = str(getattr(source_chunk, "company", ""))
            if not payload.get("company"):
                source_company = str(getattr(source_chunk, "company", "")).strip().upper()
                if source_company:
                    payload["company"] = source_company
            if not payload["parsed"].get("quarter"):
                source_quarter = str(getattr(source_chunk, "quarter", "")).strip()
                if source_quarter:
                    payload["parsed"]["quarter"] = source_quarter
    return payload


def _overview_to_commentary_answer_payload(
    *,
    ratio_result: Dict[str, Any],
    period_hint: Optional[str],
    company: str,
) -> Dict[str, Any]:
    frame = ratio_result.get("frame")
    if frame is None or frame.empty:
        return {"found": False}

    metric_keys = (
        "net_kar",
        "brut_kar",
        "satis_gelirleri",
        "favok",
        "faaliyet_nakit_akisi",
        "capex",
        "serbest_nakit_akisi",
        "magaza_sayisi",
    )
    ratio_keys = (
        "net_margin",
        "favok_margin",
        "brut_kar_marji",
        "revenue_growth_qoq",
        "store_growth_qoq",
        "cash_flow_growth_qoq",
    )
    value_keys = list(metric_keys + ratio_keys)
    valid_frame = frame.dropna(subset=value_keys, how="all")
    if valid_frame.empty:
        return {"found": False}

    available_quarters = [str(item) for item in frame["quarter"].tolist()]
    if period_hint and period_hint in available_quarters:
        requested_row = frame[frame["quarter"] == period_hint]
        requested_valid = requested_row.dropna(subset=value_keys, how="all")
        if requested_valid.empty:
            return {
                "found": False,
                "company": company,
                "parsed": {"quarter": period_hint},
            }
        row = requested_valid.iloc[[-1]]
        period = str(row.iloc[-1]["quarter"])
    else:
        row = valid_frame.iloc[[-1]]
        period = str(row.iloc[-1]["quarter"])

    current = row.iloc[-1]
    prev_frame = valid_frame[valid_frame["quarter"] < period]
    previous = prev_frame.iloc[-1] if not prev_frame.empty else None

    metrics = {key: current.get(key) for key in metric_keys}
    ratios = {key: current.get(key) for key in ratio_keys}
    deltas: Dict[str, Any] = {}
    if previous is not None:
        for key in metric_keys + ratio_keys:
            cur_val = current.get(key)
            prev_val = previous.get(key)
            if cur_val is None or prev_val is None:
                continue
            if _is_nan(cur_val) or _is_nan(prev_val):
                continue
            try:
                deltas[f"{key}_qoq"] = float(cur_val) - float(prev_val)
            except Exception:
                continue

    confidence_map_raw = ratio_result.get("confidence_map", {})
    confidence_map: Dict[str, Any] = {}
    verify_map: Dict[str, Any] = {}
    evidence_rows: List[Dict[str, Any]] = []
    for key in metric_keys + ratio_keys:
        detail = dict(confidence_map_raw.get(key, {}).get(period, {}) or {})
        confidence_map[key] = detail.get("confidence")
        verify_map[key] = detail.get("verify_status")
        for ev in list(detail.get("evidence", []))[:1]:
            evidence_rows.append({"excerpt": str(ev)})

    return {
        "found": True,
        "company": company,
        "parsed": {"quarter": period},
        "metrics": metrics,
        "ratios": ratios,
        "deltas": deltas,
        "confidence_map": confidence_map,
        "verify_map": verify_map,
        "evidence": evidence_rows[:5],
        "answer": {"found": True, "verify_status": "WARN", "bullets": ["Ceyrek ozeti hazir."]},
    }


def _extract_year_from_doc_id(doc_id: str) -> Optional[str]:
    match = re.search(r"(20\d{2})", str(doc_id))
    return match.group(1) if match else None


DISPLAY_SAFE_LEADING_TOKENS = {
    "ve",
    "bu",
    "bir",
    "ile",
    "ancak",
    "fakat",
    "ama",
    "gibi",
    "hem",
    "ise",
    "sonra",
    "icin",
    "için",
    "olarak",
}


def _clean_chunk_text_for_display(text: str) -> str:
    payload = text.lstrip()
    if not payload:
        return payload

    match = re.match(r"^([^\s]+)(\s+.*)$", payload, flags=re.DOTALL)
    if not match:
        return payload

    first_token = match.group(1)
    rest = match.group(2).lstrip()
    token_alpha = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", first_token)
    token_lower = token_alpha.lower()
    if not token_lower:
        return payload

    looks_fragment = (
        len(token_lower) <= 7
        and token_lower not in DISPLAY_SAFE_LEADING_TOKENS
        and token_alpha[:1].islower()
        and first_token.endswith((",", ".", ";", ":"))
    )
    if looks_fragment and rest:
        return rest
    return payload


def _short_excerpt(text: str, max_chars: int = 240) -> str:
    display_text = _clean_chunk_text_for_display(text)
    compact = " ".join(line.strip() for line in display_text.splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _summary_lines(answer_text: str) -> List[str]:
    lines: List[str] = []
    for line in answer_text.splitlines():
        stripped = line.strip()
        if stripped == "Evidence":
            break
        if stripped.startswith("- "):
            lines.append(stripped)
    return lines


def _summary_items_for_display(answer_text: str) -> List[str]:
    items: List[str] = []
    for line in _summary_lines(answer_text):
        item = line[2:].strip() if line.startswith("- ") else line.strip()
        norm = _normalize_for_match(item)
        if norm.startswith("soru"):
            continue
        if norm.startswith("sayisal adaylar") or ("adaylar" in norm and "say" in norm):
            continue
        if norm.startswith("yanit"):
            item = item.split(":", 1)[1].strip() if ":" in item else item
        elif ":" in item:
            head_norm = _normalize_for_match(item.split(":", 1)[0])
            if head_norm.startswith("yan"):
                item = item.split(":", 1)[1].strip()
        items.append(item)
    return items


def _is_low_information_summary(item: str) -> bool:
    norm = _normalize_for_match(item)
    low_info_patterns = (
        "ilgili icerik asagidaki kanitlarda bulundu",
        "icerik asagidaki kanitlarda bulundu",
        "asagidaki kanitlarda bulundu",
        "kanitlarda bulundu",
    )
    return any(pattern in norm for pattern in low_info_patterns)


def _technical_lines(answer_text: str) -> List[str]:
    details: List[str] = []
    for line in _summary_lines(answer_text):
        item = line[2:].strip() if line.startswith("- ") else line.strip()
        norm = _normalize_for_match(item)
        if norm.startswith("sayisal adaylar") or ("adaylar" in norm and "say" in norm):
            details.append(item)
    return details


def _is_found_answer(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return "dokümanda bulunamadı" not in lowered and "dokumanda bulunamadi" not in lowered


def _source_label(doc_id: str, quarter: str, page: object, section_title: str) -> str:
    return f"{doc_id} | {quarter} | s.{page} | {section_title}"


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _format_pct(value: object) -> str:
    if value is None or _is_nan(value):
        return "-"
    return f"%{float(value):.2f}".replace(".", ",")


def _format_delta(value: object, unit: str) -> str:
    if value is None or _is_nan(value):
        return "-"
    number = float(value)
    if unit == "count":
        return f"{number:,.0f}".replace(",", ".")
    if unit == "TL":
        abs_number = abs(number)
        if abs_number >= 1_000_000_000:
            return f"{number / 1_000_000_000:.2f}".replace(".", ",") + " milyar TL"
        if abs_number >= 1_000_000:
            return f"{number / 1_000_000:.2f}".replace(".", ",") + " milyon TL"
        return f"{number:,.0f}".replace(",", ".") + " TL"
    return f"{number:.2f}".replace(".", ",")


TR_NORMALIZE_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)


def _normalize_for_match(text: str) -> str:
    lowered = text.lower().translate(TR_NORMALIZE_MAP)
    lowered = re.sub(r"[^\w%\s\.,:]", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _clean_numeric(raw: str) -> str:
    return raw.strip().strip(".,;: ")


def _parse_tr_number(raw: str) -> Optional[float]:
    payload = raw.strip().replace(" ", "")
    if not payload:
        return None

    sign = -1.0 if payload.startswith("-") else 1.0
    payload = payload.lstrip("+-")
    if not payload:
        return None

    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", payload):
        normalized = payload.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", payload):
        normalized = payload.replace(",", "")
    elif payload.count(",") == 1 and payload.count(".") == 0:
        left, right = payload.split(",", 1)
        normalized = f"{left}.{right}" if len(right) <= 2 else f"{left}{right}"
    elif payload.count(".") > 1 and payload.count(",") == 0:
        normalized = payload.replace(".", "")
    elif "," in payload and "." in payload:
        if payload.rfind(",") > payload.rfind("."):
            normalized = payload.replace(".", "").replace(",", ".")
        else:
            normalized = payload.replace(",", "")
    else:
        normalized = payload.replace(",", ".")

    try:
        return sign * float(normalized)
    except ValueError:
        return None


def _format_money_compact(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}".replace(".", ",") + " mlr TL"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}".replace(".", ",") + " mn TL"
    return f"{value:,.0f}".replace(",", ".") + " TL"


def _format_tl_from_match(raw: str, text_norm: str, match: re.Match[str]) -> str:
    number = _parse_tr_number(raw)
    if number is None:
        return raw

    near_start = max(0, match.start() - 48)
    near_end = min(len(text_norm), match.end() + 48)
    near_context = text_norm[near_start:near_end]
    wide_start = max(0, match.start() - 220)
    wide_end = min(len(text_norm), match.end() + 220)
    wide_context = text_norm[wide_start:wide_end]

    multiplier = 1.0
    if "milyar" in near_context:
        multiplier = 1_000_000_000.0
    elif "milyon" in near_context:
        multiplier = 1_000_000.0
    elif "bin" in near_context:
        multiplier = 1_000.0
    elif "milyar" in wide_context:
        multiplier = 1_000_000_000.0
    elif "milyon" in wide_context:
        multiplier = 1_000_000.0
    elif "bin" in wide_context:
        multiplier = 1_000.0
    elif "milyar tl" in text_norm:
        multiplier = 1_000_000_000.0
    elif "milyon tl" in text_norm:
        multiplier = 1_000_000.0
    elif "bin tl" in text_norm:
        multiplier = 1_000.0

    return _format_money_compact(number * multiplier)


def _is_negative_display_value(value_text: str) -> bool:
    payload = str(value_text or "").strip().replace(" ", "")
    if not payload:
        return False
    if payload.startswith("-") or payload.startswith("%-"):
        return True
    if payload.startswith("(") and payload.endswith(")"):
        return True
    return False


def _primary_value_chip_html(value_text: str) -> str:
    safe_value = html.escape(str(value_text or "-"))
    if _is_negative_display_value(str(value_text)):
        bg = "#3a0f15"
        fg = "#ff6b6b"
        border = "#7f1d1d"
    else:
        bg = "#0f2f22"
        fg = "#4ade80"
        border = "#166534"
    return (
        "<span style='display:inline-block;padding:2px 10px;border-radius:8px;"
        f"border:1px solid {border};background:{bg};color:{fg};font-family:ui-monospace,Consolas,monospace'>"
        f"{safe_value}</span>"
    )


def _best_extractor_primary_answer(question: str, chunks: Sequence[Any]) -> Optional[Dict[str, Any]]:
    if not chunks:
        return None
    try:
        from src.metrics_extractor import (
            extract_metric_with_candidates,
            format_metric_value,
            infer_metric_from_question,
            metric_display_name,
            metric_unit,
        )
    except Exception:
        return None

    metric = infer_metric_from_question(question)
    q_norm = _normalize_for_match(question)
    if metric is None and "kar" in q_norm and "marj" not in q_norm and "favok" not in q_norm and "brut" not in q_norm:
        metric = "net_kar"
    if metric is None:
        return None

    parsed = parse_query(question)
    parsed_quarter = str(parsed.get("quarter") or "").upper() or None

    available_quarters: List[str] = []
    for chunk in chunks:
        chunk_quarter = str(getattr(chunk, "quarter", "")).upper()
        match = re.search(r"Q([1-4])", chunk_quarter)
        if match:
            available_quarters.append(f"Q{match.group(1)}")
    if not available_quarters:
        available_quarters = ["Q4", "Q3", "Q2", "Q1"]

    seen = set()
    ordered_quarters: List[str] = []
    if parsed_quarter:
        ordered_quarters.append(parsed_quarter)
        seen.add(parsed_quarter)
    for quarter in available_quarters:
        if quarter not in seen:
            seen.add(quarter)
            ordered_quarters.append(quarter)

    # If question has an explicit quarter, prefer it strictly first.
    if parsed_quarter:
        extracted = extract_metric_with_candidates(
            chunks=chunks,
            metric=metric,
            quarter=parsed_quarter,
            top_n=6,
        )
        selected = extracted.get("selected")
        if selected:
            unit = str(selected.get("unit") or metric_unit(metric))
            currency = str(selected.get("currency", "TL"))
            value_raw = selected.get("value")
            if value_raw is not None:
                try:
                    formatted_value = format_metric_value(float(value_raw), unit, currency)
                    source = chunks[0]
                    for chunk in chunks:
                        same_doc = str(getattr(chunk, "doc_id", "")) == str(selected.get("doc_id", ""))
                        same_page = int(getattr(chunk, "page", -1)) == int(selected.get("page", -2))
                        if same_doc and same_page:
                            source = chunk
                            break
                    return {
                        "label": metric_display_name(metric),
                        "value": formatted_value,
                        "source": source,
                    }
                except Exception:
                    pass

    best: Optional[Dict[str, Any]] = None
    best_rank: Tuple[float, float] = (-1.0, -1.0)
    for quarter in ordered_quarters:
        extracted = extract_metric_with_candidates(
            chunks=chunks,
            metric=metric,
            quarter=quarter,
            top_n=6,
        )
        selected = extracted.get("selected")
        if not selected:
            continue
        confidence = float(selected.get("confidence") or 0.0)
        score = float(selected.get("score") or 0.0)
        rank = (confidence, score)
        if rank > best_rank:
            best_rank = rank
            best = selected

    if not best:
        return None

    unit = str(best.get("unit") or metric_unit(metric))
    currency = str(best.get("currency", "TL"))
    value_raw = best.get("value")
    if value_raw is None:
        return None
    try:
        formatted_value = format_metric_value(float(value_raw), unit, currency)
    except Exception:
        return None

    source = chunks[0]
    for chunk in chunks:
        same_doc = str(getattr(chunk, "doc_id", "")) == str(best.get("doc_id", ""))
        same_page = int(getattr(chunk, "page", -1)) == int(best.get("page", -2))
        if same_doc and same_page:
            source = chunk
            break

    return {
        "label": metric_display_name(metric),
        "value": formatted_value,
        "source": source,
    }


def _extract_primary_answer_with_fallback(
    question: str,
    chunks: Sequence[Any],
    company: Optional[str],
) -> Optional[Dict[str, Any]]:
    primary = _extract_primary_answer(question=question, chunks=chunks)
    if primary:
        return primary

    try:
        from src.metrics_extractor import build_metric_query, infer_metric_from_question
    except Exception:
        return None

    metric = infer_metric_from_question(question)
    q_norm = _normalize_for_match(question)
    if metric is None and "kar" in q_norm and "marj" not in q_norm and "favok" not in q_norm and "brut" not in q_norm:
        metric = "net_kar"
    if metric is None:
        return None

    parsed = parse_query(question)
    parsed_quarter = str(parsed.get("quarter") or "").upper() or None
    settings = _get_ui_settings()
    top_k_initial = max(int(settings.get("top_k_initial_v3", CONFIG.retrieval.v3_top_k_initial)), 30)
    top_k_final = max(int(settings.get("top_k_final", CONFIG.retrieval.top_k_final)), 12)
    alpha = float(settings.get("alpha_v3", CONFIG.retrieval.alpha_v3))

    retriever = _retriever_v3()
    search_quarters = [parsed_quarter] if parsed_quarter else ["Q4", "Q3", "Q2", "Q1"]
    for quarter in search_quarters:
        metric_query = build_metric_query(metric, quarter, question)
        fallback_chunks = retriever.retrieve_with_query_awareness(
            query=metric_query,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            alpha=alpha,
            quarter_override=quarter,
            company_override=company,
        )
        fallback_primary = _extract_primary_answer(question=question, chunks=fallback_chunks)
        if fallback_primary:
            return fallback_primary

    # Final robust fallback: ratio_engine table (company-specific, cross-quarter aware).
    if company:
        try:
            from src.metrics_extractor import format_metric_value, metric_display_name, metric_unit
            from src.ratio_engine import build_ratio_table
        except Exception:
            return None

        parsed_quarter = str(parsed.get("quarter") or "").upper() or None
        settings = _get_ui_settings()
        top_k_initial = max(int(settings.get("top_k_initial_v3", CONFIG.retrieval.v3_top_k_initial)), 30)
        top_k_final = max(int(settings.get("top_k_final", CONFIG.retrieval.top_k_final)), 12)
        alpha = float(settings.get("alpha_v3", CONFIG.retrieval.alpha_v3))

        ratio_seed_query = (
            "Q1 Q2 Q3 Q4 net kar favok satis gelirleri net kar marji "
            "favok marji brut kar marji magaza sayisi trend"
        )
        ratio = build_ratio_table(
            question=ratio_seed_query,
            retriever=_retriever_v3(),
            company=company,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            alpha=alpha,
        )
        frame = ratio.get("frame")
        if frame is None or frame.empty:
            return None

        column_map = {
            "net_kar": "net_kar",
            "favok": "favok",
            "satis_gelirleri": "satis_gelirleri",
            "magaza_sayisi": "magaza_sayisi",
            "net_kar_marji": "net_margin",
            "favok_marji": "favok_margin",
            "brut_kar_marji": "brut_kar_marji",
        }
        record_metric_map = {
            "net_margin": "net_kar_marji",
            "favok_margin": "favok_marji",
        }
        value_column = column_map.get(metric)
        if not value_column or value_column not in frame.columns:
            return None

        candidate_rows = frame.dropna(subset=[value_column])
        if candidate_rows.empty:
            return None
        if parsed_quarter:
            quarter_rows = candidate_rows[candidate_rows["quarter"] == parsed_quarter]
            selected_row = quarter_rows.iloc[-1] if not quarter_rows.empty else candidate_rows.iloc[-1]
        else:
            selected_row = candidate_rows.iloc[-1]

        value = selected_row.get(value_column)
        if value is None or _is_nan(value):
            return None

        display_metric = metric
        if value_column == "net_margin":
            display_metric = "net_kar_marji"
        elif value_column == "favok_margin":
            display_metric = "favok_marji"
        selected_quarter = str(selected_row.get("quarter", parsed_quarter or ""))

        metric_records = ratio.get("metric_records", {}) or {}
        record_metric = record_metric_map.get(value_column, display_metric)
        selected_record = None
        for record in metric_records.get(record_metric, []):
            if str(record.get("quarter")) == selected_quarter:
                selected_record = record
                break
        unit = metric_unit(display_metric)
        selected_currency = str((selected_record or {}).get("currency", "TL"))
        formatted = format_metric_value(float(value), unit, selected_currency)
        label = metric_display_name(display_metric)

        if selected_record:
            source = SimpleNamespace(
                doc_id=str(selected_record.get("doc_id", "")),
                quarter=str(selected_record.get("quarter", selected_quarter)),
                page=int(selected_record.get("page", 0) or 0),
                section_title=str(selected_record.get("section_title", "(no heading)")),
                text=str(selected_record.get("excerpt", "")),
            )
        else:
            source = chunks[0] if chunks else SimpleNamespace(
                doc_id="-",
                quarter=selected_quarter,
                page=0,
                section_title="(no heading)",
                text="",
            )

        return {
            "label": label,
            "value": formatted,
            "source": source,
        }
    return None


def _extract_primary_answer(question: str, chunks: Sequence[Any]) -> Optional[Dict[str, Any]]:
    if not chunks:
        return None

    extractor_answer = _best_extractor_primary_answer(question=question, chunks=chunks)
    if extractor_answer:
        return extractor_answer

    q_norm = _normalize_for_match(question)
    normalized_chunks: List[Dict[str, Any]] = []
    for chunk in chunks:
        normalized_chunks.append(
            {
                "chunk": chunk,
                "text_norm": _normalize_for_match(getattr(chunk, "text", "")),
                "section_norm": _normalize_for_match(getattr(chunk, "section_title", "")),
            }
        )

    def search(
        label: str,
        patterns: Sequence[re.Pattern],
        predicate: Optional[Any] = None,
        formatter: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        for row in normalized_chunks:
            chunk = row["chunk"]
            text_norm = row["text_norm"]
            section_norm = row["section_norm"]
            if predicate and not predicate(text_norm, section_norm):
                continue
            for pattern in patterns:
                match = pattern.search(text_norm)
                if not match:
                    continue
                raw = _clean_numeric(match.group(1))
                if not raw:
                    continue
                if formatter:
                    try:
                        value = formatter(raw, text_norm, match)
                    except TypeError:
                        value = formatter(raw)
                else:
                    value = raw
                return {"label": label, "value": value, "source": chunk}
        return None

    if "favok" in q_norm and "marj" in q_norm:
        result = search(
            label="FAVOK marji",
            patterns=[
                re.compile(r"(?:fvaok|favok)\s+marji\s*%?\s*([\-]?\d[\d\.,]*)%?"),
                re.compile(r"(?:fvaok|favok)\s+marji[^%]{0,120}%\s*([\-]?\d[\d\.,]*)"),
            ],
            formatter=lambda raw: raw if raw.endswith("%") else f"%{raw}",
        )
        if result:
            return result

    def _search_favok(label: str = "FAVOK") -> Optional[Dict[str, Any]]:
        return search(
            label=label,
            patterns=[
                re.compile(r"(?:fvaok|favok)[uü]?\s*[:\-]?\s*([\-]?\d[\d\.,]*)"),
                re.compile(
                    r"faaliyet\s+kar[ıi]\s+amortisman\s+vergi\s+oncesi\s+kar[ıi]\s*[:\-]?\s*([\-]?\d[\d\.,]*)"
                ),
            ],
            formatter=_format_tl_from_match,
        )

    if "favok" in q_norm and "marj" not in q_norm:
        result = _search_favok(label="FAVOK")
        if result:
            return result

    if "brut" in q_norm and "kar" in q_norm and "marj" in q_norm:
        result = search(
            label="Brut kar marji",
            patterns=[
                re.compile(r"brut\s+kar\s+marji\s*%?\s*([\-]?\d[\d\.,]*)%?"),
                re.compile(r"brut\s+kar\s+marji[^%]{0,120}%\s*([\-]?\d[\d\.,]*)"),
            ],
            formatter=lambda raw: raw if raw.endswith("%") else f"%{raw}",
        )
        if result:
            return result

    if "net" in q_norm and "kar" in q_norm and "marj" in q_norm:
        result = search(
            label="Net kar marji",
            patterns=[
                re.compile(r"net\s+kar\s+marji\s*%?\s*([\-]?\d[\d\.,]*)%?"),
                re.compile(r"net\s+kar\s+marji[^%]{0,120}%\s*([\-]?\d[\d\.,]*)"),
            ],
            formatter=lambda raw: raw if raw.endswith("%") else f"%{raw}",
        )
        if result:
            return result

    def _search_net_kar(label: str = "Net kar") -> Optional[Dict[str, Any]]:
        return search(
            label=label,
            patterns=[
                re.compile(r"net\s+kar[ıi]?\s*[:\-]?\s*([\-]?\d[\d\.,]*)"),
                re.compile(r"net\s+donem\s+kar[ıi]\s*[:\-]?\s*([\-]?\d[\d\.,]*)"),
                re.compile(r"net\s+donem\s+(?:kari|karı)\s*[:\-]?\s*([\-]?\d[\d\.,]*)"),
            ],
            formatter=_format_tl_from_match,
        )

    if "net" in q_norm and "kar" in q_norm:
        result = _search_net_kar(label="Net kar")
        if result:
            return result

    if (
        "kar" in q_norm
        and "marj" not in q_norm
        and "favok" not in q_norm
        and "fvaok" not in q_norm
        and "brut" not in q_norm
    ):
        result = _search_net_kar(label="Net kar")
        if result:
            return result

    if ("satis" in q_norm) or ("hasilat" in q_norm) or ("ciro" in q_norm):
        result = search(
            label="Satislar",
            patterns=[
                re.compile(r"satislar?\s+([\-]?\d[\d\.,]*)"),
                re.compile(r"satis\s+gelir(?:leri)?\s+([\-]?\d[\d\.,]*)"),
                re.compile(r"hasilat\s+([\-]?\d[\d\.,]*)"),
                re.compile(r"ciro[^\d]{0,20}([\-]?\d[\d\.,]*)"),
                re.compile(r"(?:net\s+)?satislar?[^0-9]{0,80}([\-]?\d[\d\.,]*)\s*(?:milyon|milyar)?\s*tl"),
                re.compile(r"ciro[^0-9]{0,80}([\-]?\d[\d\.,]*)\s*(?:milyon|milyar)?\s*tl"),
            ],
            formatter=_format_tl_from_match,
        )
        if result:
            return result

    if "yatirim" in q_norm:
        result = search(
            label="Yatirim tutari",
            patterns=[re.compile(r"([\-]?\d[\d\.,]*)\s*milyar\s*tl\s*yatirim")],
            formatter=lambda raw: f"{raw} milyar TL",
        )
        if result:
            return result
        result = search(
            label="Yatirim tutari",
            patterns=[re.compile(r"([\-]?\d[\d\.,]*)\s*milyon\s*tl\s*yatirim")],
            formatter=lambda raw: f"{raw} milyon TL",
        )
        if result:
            return result

    if "magaza" in q_norm and ("sayi" in q_norm or "kac" in q_norm or "adet" in q_norm or "toplam" in q_norm):
        # Prefer table-like "Mağaza Sayıları Özet Tablo" chunks when available.
        magaza_table_predicate = lambda text_norm, section_norm: (
            ("magaza sayilari" in section_norm)
            or ("bim turkiye" in text_norm and "toplam" in text_norm)
        )
        result = search(
            label="Toplam magaza sayisi",
            patterns=[re.compile(r"toplam\s+([\-]?\d[\d\.,]*)")],
            predicate=magaza_table_predicate,
            formatter=lambda raw: f"{raw} adet",
        )
        if result:
            return result

        magaza_predicate = lambda text_norm, section_norm: ("magaza" in text_norm) or ("magaza" in section_norm)
        result = search(
            label="Toplam magaza sayisi",
            patterns=[
                re.compile(r"([\-]?\d[\d\.,]*)\s*magazasi\s+bulunmaktadir"),
            ],
            predicate=magaza_predicate,
            formatter=lambda raw: f"{raw} adet",
        )
        if result:
            return result

        for row in normalized_chunks:
            text_norm = row["text_norm"]
            if "magazasi bulunmaktadir" not in text_norm:
                continue
            values = re.findall(r"\d[\d\.,]*", text_norm)
            candidates: List[Tuple[int, str]] = []
            for raw in values:
                digits = re.sub(r"[^\d]", "", raw)
                if not digits:
                    continue
                number = int(digits)
                if 1000 <= number <= 50000:
                    candidates.append((number, raw))
            if candidates:
                best_raw = max(candidates, key=lambda item: item[0])[1]
                return {
                    "label": "Toplam magaza sayisi",
                    "value": f"{best_raw} adet",
                    "source": row["chunk"],
                }

    if "calisan" in q_norm or "personel" in q_norm:
        result = search(
            label="Calisan sayisi",
            patterns=[
                re.compile(r"(?:calisan|personel)\s+sayisi\s*([\-]?\d[\d\.,]*)"),
                re.compile(r"toplam\s+(?:calisan|personel)\s*([\-]?\d[\d\.,]*)"),
            ],
            formatter=lambda raw: f"{raw} kisi",
        )
        if result:
            return result

    if "online" in q_norm or "eticaret" in q_norm or "e ticaret" in q_norm:
        result = search(
            label="Online satis orani",
            patterns=[
                re.compile(r"(?:online|eticaret|e ticaret)[^%]{0,50}([\-]?\d[\d\.,]*)\s*%"),
            ],
            formatter=lambda raw: raw if raw.endswith("%") else f"%{raw}",
        )
        if result:
            return result

    return None


def _is_comparison_mode(question: str, query_type: str) -> bool:
    if query_type == "trend":
        return True
    try:
        from src.metrics_extractor import is_comparison_query

        return bool(is_comparison_query(question))
    except Exception:
        return False


def _run_comparison_pipeline(question: str, company: Optional[str] = None) -> Dict[str, Any]:
    from src.metrics_extractor import run_cross_quarter_comparison

    settings = _get_ui_settings()
    return run_cross_quarter_comparison(
        question=question,
        retriever=_retriever_v3(),
        top_k_initial=int(settings.get("top_k_initial_v3", CONFIG.retrieval.v3_top_k_initial)),
        top_k_final=int(settings.get("top_k_final", CONFIG.retrieval.top_k_final)),
        alpha=float(settings.get("alpha_v3", CONFIG.retrieval.alpha_v3)),
        company=company,
    )


def _comparison_table_rows(frame: Any) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if frame is None or frame.empty:
        return rows
    for _, row in frame.iterrows():
        unit = str(row.get("unit", "TL") or "TL")
        rows.append(
            {
                "Ceyrek": row.get("quarter"),
                "Metrik": row.get("metric"),
                "Deger": row.get("value_display"),
                "Degisim": _format_delta(row.get("abs_change"), unit),
                "Degisim %": _format_pct(row.get("pct_change")),
                "Yon": row.get("direction") or "-",
                "Kaynak": row.get("citation") or "-",
            }
        )
    return rows


def _comparison_series_rows(frame: Any) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if frame is None or frame.empty:
        return rows
    for _, row in frame.iterrows():
        value = row.get("value")
        if value is None or _is_nan(value):
            continue
        rows.append(
            {
                "quarter": row.get("quarter"),
                "value": float(value),
            }
        )
    return rows


def _frame_to_csv_bytes(frame: Any) -> bytes:
    if frame is None:
        return b""
    try:
        csv_text = frame.to_csv(index=False)
    except Exception:
        return b""
    return csv_text.encode("utf-8")


def _comparison_markdown_lines(result: Dict[str, Any]) -> List[str]:
    from src.metrics_extractor import format_metric_value, metric_display_name

    metric = str(result.get("metric") or "")
    frame = result.get("frame")
    overall = result.get("overall_change", {})
    lines: List[str] = []

    if not metric:
        return ["- Yanit: Karsilastirma metrik tipi tespit edilemedi."]

    lines.append(f"- Metrik: {metric_display_name(metric)}")
    if frame is not None and not frame.empty:
        for _, row in frame.iterrows():
            lines.append(f"- {row['quarter']}: {row['value_display']}")

    abs_change = overall.get("abs_change")
    pct_change = overall.get("pct_change")
    direction = overall.get("direction")
    if abs_change is not None:
        unit_hint = str(frame.iloc[-1]["unit"]) if frame is not None and not frame.empty and "unit" in frame.columns else "TL"
        currency_hint = (
            str(frame.iloc[-1]["currency"])
            if frame is not None and not frame.empty and "currency" in frame.columns
            else "TL"
        )
        lines.append(
            "- Q1->Q3 Degisim: "
            + f"{format_metric_value(float(abs_change), unit_hint, currency_hint)}"
            + (f" ({_format_pct(pct_change)})" if pct_change is not None else "")
            + (f" [{direction}]" if direction else "")
        )
    return lines


def _top_sources(chunks: Sequence[Any], limit: int = 5) -> List[Dict[str, object]]:
    sources: List[Dict[str, object]] = []
    seen = set()
    for chunk in chunks:
        doc_id = getattr(chunk, "doc_id", "")
        quarter = getattr(chunk, "quarter", "")
        page = getattr(chunk, "page", 0)
        section_title = getattr(chunk, "section_title", "(no heading)")
        key = (doc_id, quarter, page, section_title)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "doc_id": doc_id,
                "company": getattr(chunk, "company", ""),
                "quarter": quarter,
                "page": page,
                "section_title": section_title,
            }
        )
        if len(sources) >= limit:
            break
    return sources


def _append_ui_log(
    question: str,
    retriever_name: str,
    parsed: Dict[str, object],
    sources: Sequence[Dict[str, object]],
    found: bool,
    company: Optional[str] = None,
) -> None:
    log_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retriever": retriever_name,
        "parsed": {
            "quarter": parsed.get("quarter"),
            "query_type": parsed.get("signals", {}).get("query_type"),
            "company": company,
        },
        "top_sources": list(sources),
        "found": bool(found),
    }
    with UI_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_row, ensure_ascii=False) + "\n")


def _append_feedback_log(
    *,
    company: Optional[str],
    quarter: Optional[str],
    metric: str,
    extracted_value: Optional[str],
    user_value: Optional[str],
    evidence_ref: Optional[str],
    verdict: str,
    note: Optional[str] = None,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "company": company,
        "quarter": quarter,
        "metric": metric,
        "extracted_value": extracted_value,
        "user_value": user_value,
        "evidence_ref": evidence_ref,
        "verdict": verdict,
        "note": note,
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _markdown_export(
    question: str,
    retriever_name: str,
    parsed: Dict[str, object],
    answer_text: str,
    chunks: Sequence[Any],
    comparison_result: Optional[Dict[str, Any]] = None,
    mode: str = "standard",
) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [
        "# RAG-Fin Answer Export",
        "",
        f"- Timestamp: {timestamp}",
        f"- Retriever: {retriever_name}",
        f"- Question: {question}",
        f"- Parsed quarter: {parsed.get('quarter')}",
        f"- Parsed query_type: {parsed.get('signals', {}).get('query_type')}",
        "",
        "## Answer",
    ]

    summary = _summary_lines(answer_text)
    if summary:
        lines.extend(summary)
    else:
        lines.append(answer_text.strip() or "- (empty)")

    if mode == "comparison" and comparison_result:
        lines.extend(["", "## Comparison"])
        lines.extend(_comparison_markdown_lines(comparison_result))

    lines.extend(["", "## Evidence"])
    if mode == "comparison" and comparison_result:
        records = comparison_result.get("records", [])
        if not records:
            lines.append("- Uygun kanıt bulunamadı.")
        else:
            for record in records:
                lines.append(
                    f"- [{record['doc_id']} | {record['quarter']} | {record['page']} | {record['section_title']}] "
                    f"{_clean_chunk_text_for_display(str(record.get('excerpt', '')))}"
                )
    else:
        if not chunks:
            lines.append("- Uygun kanıt bulunamadı.")
        else:
            for chunk in chunks:
                lines.append(
                    f"- [{chunk.doc_id} | {chunk.quarter} | {chunk.page} | {chunk.section_title}] "
                    f"{_short_excerpt(chunk.text)}"
                )
    return "\n".join(lines) + "\n"


@st.cache_resource(show_spinner=False)
def _answer_engine() -> Any:
    from src.answer import AnswerEngine, RulesBasedAnswerAdapter

    return AnswerEngine(adapter=RulesBasedAnswerAdapter(max_distance=0.45))


@st.cache_resource(show_spinner=False)
def _retriever_v1() -> Any:
    from src.retrieve import Retriever

    return Retriever(
        chroma_path=CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        model_name=CONFIG.models.embedding,
    )


@st.cache_resource(show_spinner=False)
def _retriever_v2() -> Any:
    from src.retrieve import RetrieverV2

    return RetrieverV2(
        chroma_path=CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION_NAME_V2,
        model_name=CONFIG.models.embedding,
    )


@st.cache_resource(show_spinner=False)
def _retriever_v3() -> Any:
    from src.retrieve import RetrieverV3

    return RetrieverV3(
        chroma_path=CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION_NAME_V2,
        model_name=CONFIG.models.embedding,
    )


@st.cache_resource(show_spinner=False)
def _retriever_v4_bm25() -> Any:
    from src.retrieve import RetrieverBM25

    return RetrieverBM25(chunks_file=CHUNKS_V2_FILE)


@st.cache_resource(show_spinner=False)
def _retriever_v5_hybrid() -> Any:
    from src.retrieve import RetrieverV5Hybrid

    return RetrieverV5Hybrid(
        chroma_path=CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION_NAME_V2,
        model_name=CONFIG.models.embedding,
        chunks_file=CHUNKS_V2_FILE,
    )


@st.cache_resource(show_spinner=False)
def _retriever_v6_cross() -> Any:
    from src.retrieve import RetrieverV6Cross

    return RetrieverV6Cross(
        chroma_path=CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION_NAME_V2,
        model_name=CONFIG.models.embedding,
        chunks_file=CHUNKS_V2_FILE,
        cross_encoder_model=CONFIG.models.cross_encoder,
    )


def _clear_cached_retrievers() -> None:
    _answer_engine.clear()
    _retriever_v1.clear()
    _retriever_v2.clear()
    _retriever_v3.clear()
    _retriever_v4_bm25.clear()
    _retriever_v5_hybrid.clear()
    _retriever_v6_cross.clear()


def _retrieve(question: str, retriever_name: str, company: Optional[str] = None) -> List[Any]:
    settings = _get_ui_settings()
    top_k_final = int(settings.get("top_k_final", CONFIG.retrieval.top_k_final))
    top_k_initial_v2 = int(settings.get("top_k_initial_v2", CONFIG.retrieval.v2_top_k_initial))
    top_k_initial_v3 = int(settings.get("top_k_initial_v3", CONFIG.retrieval.v3_top_k_initial))
    top_k_vector_v5 = int(settings.get("top_k_vector_v5", CONFIG.retrieval.v5_top_k_vector))
    top_k_bm25_v5 = int(settings.get("top_k_bm25_v5", CONFIG.retrieval.v5_top_k_bm25))
    top_k_candidates_v6 = int(settings.get("top_k_candidates_v6", CONFIG.retrieval.v6_cross_top_n))
    alpha_v2 = float(settings.get("alpha_v2", CONFIG.retrieval.alpha_v2))
    alpha_v3 = float(settings.get("alpha_v3", CONFIG.retrieval.alpha_v3))
    beta_v5 = float(settings.get("beta_v5", CONFIG.retrieval.beta_v5))

    if retriever_name == "v1":
        return _retriever_v1().retrieve(
            question,
            top_k=top_k_final,
            company=company,
        )
    if retriever_name == "v2":
        return _retriever_v2().retrieve_with_boost(
            query=question,
            top_k_initial=top_k_initial_v2,
            top_k_final=top_k_final,
            alpha=alpha_v2,
            company=company,
        )
    if retriever_name == "v4":
        return _retriever_v4_bm25().retrieve(
            query=question,
            top_k=top_k_final,
            company=company,
        )
    if retriever_name == "v5":
        return _retriever_v5_hybrid().retrieve_with_hybrid(
            query=question,
            top_k_vector=top_k_vector_v5,
            top_k_bm25=top_k_bm25_v5,
            top_k_final=top_k_final,
            beta=beta_v5,
            alpha_v3=alpha_v3,
            company_override=company,
        )
    if retriever_name == "v6":
        return _retriever_v6_cross().retrieve_with_cross_encoder(
            query=question,
            top_k_candidates=top_k_candidates_v6,
            top_k_final=top_k_final,
            top_k_vector=top_k_vector_v5,
            top_k_bm25=top_k_bm25_v5,
            beta=beta_v5,
            alpha_v3=alpha_v3,
            company_override=company,
        )
    return _retriever_v3().retrieve_with_query_awareness(
        query=question,
        top_k_initial=top_k_initial_v3,
        top_k_final=top_k_final,
        alpha=alpha_v3,
        company_override=company,
    )


def _metrics_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    retrievers = summary.get("retrievers", {})
    rows: List[Dict[str, object]] = []
    preferred_order = ["v1", "v2", "v3", "v4_bm25", "v5_hybrid", "v6_cross"]
    for name in preferred_order:
        if name not in retrievers:
            continue
        metrics = retrievers.get(name, {})
        rows.append(
            {
                "retriever": name,
                "hit@1": metrics.get("hit@1", 0.0),
                "hit@3": metrics.get("hit@3", 0.0),
                "hit@5": metrics.get("hit@5", 0.0),
                "MRR@5": metrics.get("MRR@5", 0.0),
                "quarter@1": metrics.get("quarter_accuracy@1", 0.0),
            }
        )
    return rows


def _load_metrics_summary_from_disk() -> Optional[Dict[str, object]]:
    if not DEFAULT_SUMMARY_OUTPUT.exists():
        return None
    with DEFAULT_SUMMARY_OUTPUT.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_pdf_file_list() -> None:
    st.markdown("**Mevcut PDF Dosyalari (`data/raw`)**")
    pdf_files = list_pdf_files(RAW_DIR)
    if not pdf_files:
        st.warning("Henüz PDF yok. Asagidan PDF yukleyebilirsiniz.")
        return
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        st.write(f"- {pdf.name} ({size_mb:.2f} MB)")


def _render_stats_panel() -> None:
    st.subheader("Sistem Durumu")
    stats = _data_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("PDF sayisi", stats["pdf_count"] or 0)
    col2.metric("Sayfa kaydi", stats["page_count"] or 0)
    col3.metric(
        "Toplam chunk",
        (stats["chunk_count_v1"] or 0) + (stats["chunk_count_v2"] or 0),
    )

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Chunk v1", stats["chunk_count_v1"] or 0)
    col5.metric("Chunk v2", stats["chunk_count_v2"] or 0)
    col6.metric(
        "Collection v1",
        "N/A" if stats["collection_count_v1"] is None else stats["collection_count_v1"],
    )
    col7.metric(
        "Collection v2",
        "N/A" if stats["collection_count_v2"] is None else stats["collection_count_v2"],
    )

    if stats["collection_count_v1"] is None or stats["collection_count_v2"] is None:
        st.warning("Collection sayilari okunamadi. Python 3.9-3.12 ortaminda calistirdiginizdan emin olun.")


def _render_data_tab() -> None:
    st.subheader("1) Veri Hazirlama")
    st.info("Akilli yukleme: PDF yukleyin, sistem sirket/ceyrek/yili otomatik algilar, ingest + index_v2 otomatik calisir.")
    _render_pdf_file_list()

    uploaded = st.file_uploader(
        "Yeni PDF yukle",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        detected_rows = [_detect_pdf_metadata_from_name(getattr(item, "name", "")) for item in uploaded]
        st.markdown("**Algilanan Dosya Bilgisi**")
        st.dataframe(detected_rows, use_container_width=True, hide_index=True)

    if uploaded:
        upload_signature = "|".join(
            f"{getattr(item, 'name', '')}:{getattr(item, 'size', 0)}"
            for item in sorted(uploaded, key=lambda x: getattr(x, "name", ""))
        )
        if st.session_state.get("last_auto_upload_signature") != upload_signature:
            saved = _save_uploaded_pdfs(uploaded)
            if saved:
                st.success(f"{len(saved)} PDF kaydedildi.")
                for name in saved:
                    st.write(f"- {name}")
                with st.spinner("Incremental ingest + index v2 otomatik calisiyor..."):
                    ok, message = _run_ingest_and_index_v2(saved_files=saved)
                if ok:
                    st.session_state["last_auto_upload_signature"] = upload_signature
                    st.success(message)
                    st.info("Artik Ask veya Dashboard sekmesinden soru sorabilirsiniz.")
                else:
                    st.error(message)
            else:
                st.warning("Gecerli PDF bulunamadi.")

    with st.expander("Advanced Mode (Muhendislik Ayarlari)", expanded=False):
        _render_quick_start_card()
        col_ingest, col_index_v1, col_index_v2 = st.columns(3)

        if col_ingest.button("1. Ingest Baslat"):
            with st.spinner("Ingest calisiyor..."):
                pages, summary = ingest_raw_pdfs(raw_dir=RAW_DIR, output_file=PAGES_FILE)
            st.session_state["last_ingest_summary"] = summary
            st.success(f"Ingest tamamlandi. Kayit sayisi: {len(pages)}")
            st.info("Sonraki adim: '2. Incremental Index v2 Baslat' (onerilen) calistirin.")

        if col_index_v1.button("2. Index v1 Baslat"):
            try:
                from src.index import build_index

                with st.spinner("Index v1 calisiyor..."):
                    summary = build_index(
                        raw_dir=RAW_DIR,
                        processed_dir=PROCESSED_DIR,
                        collection_name=DEFAULT_COLLECTION_NAME,
                        chunk_size=CONFIG.chunking.v1.chunk_size,
                        overlap=CONFIG.chunking.v1.overlap,
                    )
                _clear_cached_retrievers()
                st.session_state["last_index_v1_summary"] = summary
                st.success("Index v1 tamamlandi.")
                st.info("Ask sekmesine gecip soru sorabilirsiniz.")
            except Exception as exc:
                st.error(f"Index v1 baslatilamadi: {exc}")

        if col_index_v2.button("2. Incremental Index v2 Baslat (Onerilen)"):
            with st.spinner("Incremental index v2 calisiyor..."):
                ok, message = _run_ingest_and_index_v2(saved_files=[])
            if ok:
                st.success(message)
                st.info("Ask sekmesine gecip soru sorabilirsiniz.")
            else:
                st.error(f"Index v2 baslatilamadi: {message}")

    if "last_ingest_summary" in st.session_state:
        with st.expander("Son ingest ozeti"):
            st.json(st.session_state["last_ingest_summary"])
    if "last_index_v1_summary" in st.session_state:
        with st.expander("Son index v1 ozeti"):
            st.json(st.session_state["last_index_v1_summary"])
    if "last_index_v2_summary" in st.session_state:
        with st.expander("Son index v2 ozeti"):
            st.json(st.session_state["last_index_v2_summary"])

    _render_stats_panel()


def _render_ask_tab() -> None:
    st.subheader("2) Soru Sor")
    st.caption("Sistem sadece dokuman iceriginden yanit uretir ve her yanitta kanit gosterir.")
    available_companies = _available_companies()

    st.markdown("**Ornek Sorular**")
    sample_cols = st.columns(3)
    for idx, sample in enumerate(EXAMPLE_QUESTIONS):
        col = sample_cols[idx % 3]
        if col.button(sample, key=f"sample_q_{idx}"):
            st.session_state["ask_question_input"] = sample

    question = st.text_area(
        "Sorunuz",
        placeholder="Ornek: 2025 ucuncu ceyrek net kar kac?",
        height=100,
        key="ask_question_input",
    )
    company_options = ["TUMU"] + available_companies
    selected_company = st.selectbox(
        "Sirket Filtresi",
        options=company_options,
        index=0,
        format_func=_company_display_name,
    )
    company_filter = None if selected_company == "TUMU" else selected_company
    retriever_name = "v3"
    with st.expander("Advanced Mode", expanded=False):
        retriever_name = st.radio(
            "Arama Modu",
            options=["v6", "v5", "v4", "v3", "v2", "v1"],
            horizontal=True,
            format_func=lambda x: RETRIEVER_HELP[x],
            index=3,
        )

    parsed = parse_query(question) if question.strip() else {"quarter": None, "signals": {"query_type": "other"}}
    quarter = parsed.get("quarter")
    query_type = parsed.get("signals", {}).get("query_type")
    mentioned_companies = detect_company_mentions(question, available_companies=available_companies)
    cross_company_mode = is_cross_company_query(
        question,
        available_companies=available_companies,
    ) and (len(mentioned_companies) >= 2 or (len(mentioned_companies) == 0 and len(available_companies) >= 2))
    comparison_mode = _is_comparison_mode(question, str(query_type)) and not cross_company_mode

    st.markdown("**Algilanan Soru Bilgisi**")
    col_q, col_t, col_m, col_c = st.columns(4)
    col_q.metric("Ceyrek", quarter if quarter else "Belirsiz")
    col_t.metric("Soru Tipi", str(query_type))
    if cross_company_mode:
        mode_label = "Cross-Company"
    elif comparison_mode:
        mode_label = "v4 Karsilastirma"
    else:
        mode_label = "Standart"
    col_m.metric("Mod", mode_label)
    col_c.metric("Sirket", company_filter or "TUMU")

    if st.button("Yanit Uret"):
        if not question.strip():
            st.warning("Lutfen bir soru girin.")
        else:
            retrieval_error: Optional[str] = None
            chunks: List[Any] = []
            comparison_result: Optional[Dict[str, Any]] = None
            cross_company_result: Optional[Dict[str, Any]] = None
            answer_text = ""
            mode = "cross_company" if cross_company_mode else ("comparison" if comparison_mode else "standard")
            with st.spinner("Sorgu calisiyor..."):
                try:
                    if cross_company_mode:
                        comparison_companies = list(mentioned_companies)
                        if company_filter and company_filter not in comparison_companies:
                            comparison_companies.insert(0, company_filter)
                        if len(comparison_companies) < 2:
                            comparison_companies = available_companies[:3]
                        cross_company_result = run_cross_company_comparison(
                            question=question,
                            retriever=_retriever_v3(),
                            companies=comparison_companies,
                            top_k_initial=CONFIG.retrieval.v3_top_k_initial,
                            top_k_final=CONFIG.retrieval.top_k_final,
                            alpha=CONFIG.retrieval.alpha_v3,
                        )
                        evidence_records = cross_company_result.get("evidence", [])
                        for record in evidence_records:
                            chunks.append(
                                type(
                                    "TmpChunk",
                                    (),
                                    {
                                        "doc_id": record.get("doc_id", ""),
                                        "company": record.get("company", ""),
                                        "quarter": record.get("quarter", ""),
                                        "page": record.get("page", 0),
                                        "section_title": record.get("section_title", "(no heading)"),
                                        "text": record.get("excerpt", ""),
                                    },
                                )()
                            )
                        if cross_company_result.get("found"):
                            answer_text = (
                                f"- {cross_company_result.get('message', 'Karsilastirma tamamlandi.')}\n"
                                f"- Hedef metrik: {cross_company_result.get('target_label', cross_company_result.get('target'))}"
                            )
                        else:
                            answer_text = (
                                "- Dokümanda bulunamadı.\n"
                                f"- {cross_company_result.get('message', 'Karsilastirma icin yeterli veri bulunamadi.')}"
                            )
                    elif comparison_mode:
                        comparison_result = _run_comparison_pipeline(question=question, company=company_filter)
                        chunks = []
                        for quarter_chunks in comparison_result.get("quarter_chunks", {}).values():
                            chunks.extend(list(quarter_chunks))
                        if comparison_result.get("found"):
                            lines = _comparison_markdown_lines(comparison_result)
                            answer_text = "\n".join(lines) if lines else "- Yanit: Karsilastirma tamamlandi."
                        else:
                            searched = comparison_result.get("top_sources", [])
                            searched_str = ", ".join(
                                _source_label(
                                    str(src.get("doc_id", "")),
                                    str(src.get("quarter", "")),
                                    src.get("page", ""),
                                    str(src.get("section_title", "(no heading)")),
                                )
                                for src in searched
                            )
                            answer_text = (
                                "- Dokümanda bulunamadı.\n"
                                f"- Aranan sayfalar: {searched_str if searched_str else 'Yok'}\n\n"
                                "Evidence\n"
                                "- Uygun kanıt bulunamadı."
                            )
                    else:
                        chunks = _retrieve(
                            question=question,
                            retriever_name=retriever_name,
                            company=company_filter,
                        )
                        answer_text = _answer_engine().answer(question=question, chunks=chunks)
                except Exception as exc:
                    chunks = []
                    comparison_result = None
                    cross_company_result = None
                    retrieval_error = str(exc)
                    answer_text = (
                        "- Dokümanda bulunamadı.\n"
                        "- Aranan sayfalar: Yok\n\n"
                        "Evidence\n"
                        "- Uygun kanıt bulunamadı."
                    )

            if cross_company_result is not None:
                found = bool(cross_company_result.get("found"))
            elif comparison_result is not None:
                found = bool(comparison_result.get("found"))
            else:
                found = _is_found_answer(answer_text)

            if cross_company_result is not None:
                frame = cross_company_result.get("frame")
                sources = []
                if frame is not None and not frame.empty:
                    for _, row in frame.iterrows():
                        sources.append(
                            {
                                "doc_id": str(row.get("company", "")),
                                "company": str(row.get("company", "")),
                                "quarter": str(row.get("quarter", "")),
                                "page": "-",
                                "section_title": str(cross_company_result.get("target", "")),
                            }
                        )
            elif comparison_result is not None:
                sources = list(comparison_result.get("top_sources", []))
            else:
                sources = _top_sources(chunks, limit=5)
            _append_ui_log(
                question=question,
                retriever_name=retriever_name,
                parsed=parsed,
                sources=sources,
                found=found,
                company=company_filter,
            )

            st.session_state["last_qa"] = {
                "question": question,
                "retriever": retriever_name,
                "company": company_filter,
                "parsed": parsed,
                "answer_text": answer_text,
                "found": found,
                "chunks": chunks,
                "retrieval_error": retrieval_error,
                "mode": mode,
                "comparison_result": comparison_result,
                "cross_company_result": cross_company_result,
            }

    qa = st.session_state.get("last_qa")
    if not qa:
        return

    if qa.get("retrieval_error"):
        st.error(f"Retrieval hatasi: {qa['retrieval_error']}")

    st.subheader("Sonuc")
    status_col1, status_col2, status_col3 = st.columns(3)
    status_col1.metric("Durum", "Bulundu" if qa["found"] else "Bulunamadi")
    if qa.get("mode") == "comparison":
        retriever_label = "V4"
    elif qa.get("mode") == "cross_company":
        retriever_label = "V4-COMP"
    else:
        retriever_label = str(qa["retriever"]).upper()
    status_col2.metric("Retriever", retriever_label)
    status_col3.metric("Kanit adedi", len(qa["chunks"]))

    with st.container(border=True):
        if qa["found"]:
            st.success("Soruya uygun kanit bulundu.")
        else:
            st.error("Dokümanda bulunamadı")
            st.info("Isterseniz soruyu daha spesifik yazin: ceyrek, KPI ya da metrik adi ekleyin.")

        if qa.get("mode") == "cross_company":
            result = qa.get("cross_company_result") or {}
            frame = result.get("frame")
            st.markdown("**Cross-Company Karsilastirma**")
            st.markdown(f"- Hedef metrik: `{result.get('target_label', result.get('target', '-'))}`")
            if result.get("best_company"):
                st.success(f"En iyi performans: {result['best_company']}")
            if frame is not None and not frame.empty:
                display_rows = []
                for _, row in frame.iterrows():
                    value = row.get("value")
                    display_rows.append(
                        {
                            "Sirket": row.get("company"),
                            "Ceyrek": row.get("quarter"),
                            "Metrik": row.get("target"),
                            "Deger": "-" if value is None or _is_nan(value) else f"{float(value):.2f}",
                        }
                    )
                st.dataframe(display_rows, use_container_width=True, hide_index=True)
                st.download_button(
                    label="Karsilastirma CSV Indir",
                    data=_frame_to_csv_bytes(frame),
                    file_name=f"cross_company_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info(result.get("message", "Karsilastirma sonucu yok."))
        elif qa.get("mode") == "comparison":
            result = qa.get("comparison_result") or {}
            from src.metrics_extractor import format_metric_value, metric_display_name

            metric = result.get("metric")
            frame = result.get("frame")
            overall = result.get("overall_change", {})
            missing_quarters = result.get("missing_quarters", [])

            if qa["found"] and metric:
                st.markdown("**Karsilastirma Ozeti**")
                st.markdown(f"### {metric_display_name(str(metric))} (Q1/Q2/Q3)")
                for line in _comparison_markdown_lines(result):
                    st.markdown(line)

                abs_change = overall.get("abs_change")
                pct_change = overall.get("pct_change")
                direction = overall.get("direction")
                if abs_change is not None:
                    unit_hint = str(frame.iloc[-1]["unit"]) if frame is not None and not frame.empty and "unit" in frame.columns else "TL"
                    currency_hint = (
                        str(frame.iloc[-1]["currency"])
                        if frame is not None and not frame.empty and "currency" in frame.columns
                        else "TL"
                    )
                    st.info(
                        "Toplam degisim (Q1->Q3): "
                        + f"{format_metric_value(float(abs_change), unit_hint, currency_hint)}"
                        + (f" | %{float(pct_change):.2f}".replace(".", ",") if pct_change is not None else "")
                        + (f" | Yon: {direction}" if direction else "")
                    )
            elif not metric:
                st.warning(
                    "Metrik tipi anlasilamadi. Lutfen net kar, net kar marji, FAVOK, brut kar marji, FAVOK marji, satis gelirleri veya magaza sayisi sorun."
                )

            if frame is not None and not frame.empty:
                st.markdown("**Ceyrek Bazli Tablo**")
                st.dataframe(_comparison_table_rows(frame), use_container_width=True, hide_index=True)
                st.download_button(
                    label="Trend Tablosunu CSV Indir",
                    data=_frame_to_csv_bytes(frame),
                    file_name=f"trend_{(qa.get('company') or 'ALL')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                chart_rows = _comparison_series_rows(frame)
                if chart_rows:
                    st.markdown("**Trend Grafigi**")
                    from pandas import DataFrame

                    st.line_chart(DataFrame(chart_rows), x="quarter", y="value")

            if missing_quarters:
                missing_txt = ", ".join(missing_quarters)
                st.warning(f"Bazi ceyreklerde metrik bulunamadi: {missing_txt}")

            if not qa["found"]:
                sources = result.get("top_sources", [])
                with st.expander("Nerelerde arandi?"):
                    if sources:
                        for source in sources:
                            st.write(
                                f"- {_source_label(str(source.get('doc_id', '')), str(source.get('quarter', '')), source.get('page', ''), str(source.get('section_title', '(no heading)')))}"
                            )
                    else:
                        st.write("Kaynak yok")
        else:
            primary_answer = _extract_primary_answer(qa["question"], qa["chunks"])
            if qa["found"] and primary_answer:
                source_chunk = primary_answer["source"]
                st.markdown("**Ana Yanit**")
                st.markdown(f"### {primary_answer['label']}: `{primary_answer['value']}`")
                st.caption(
                    "Kaynak: "
                    + _source_label(
                        source_chunk.doc_id,
                        source_chunk.quarter,
                        source_chunk.page,
                        source_chunk.section_title,
                    )
                )

            summary_items = _summary_items_for_display(qa["answer_text"])
            if summary_items:
                if primary_answer:
                    with st.expander("Detayli aciklama"):
                        for item in summary_items:
                            st.markdown(f"- {item}")
                else:
                    st.markdown("**Cevap Ozeti**")
                    for item in summary_items:
                        st.markdown(f"- {item}")

            technical_details = _technical_lines(qa["answer_text"])
            if technical_details:
                with st.expander("Teknik detaylar"):
                    for detail in technical_details:
                        st.write(f"- {detail}")

            if not qa["found"]:
                sources = _top_sources(qa["chunks"], limit=5)
                with st.expander("Nerelerde arandi?"):
                    if sources:
                        for source in sources:
                            st.write(
                                f"- {_source_label(source['doc_id'], source['quarter'], source['page'], source['section_title'])}"
                            )
                    else:
                        st.write("Kaynak yok")

    st.subheader("Kanitlar")
    if qa.get("mode") == "cross_company":
        cross_result = qa.get("cross_company_result") or {}
        records = cross_result.get("evidence", [])
        if not records:
            st.info("Uygun kanıt bulunamadı.")
        else:
            for idx, record in enumerate(records, start=1):
                with st.container(border=True):
                    st.markdown(f"**Kanit {idx}**")
                    st.caption(
                        _source_label(
                            str(record.get("doc_id", "")),
                            str(record.get("quarter", "")),
                            record.get("page", ""),
                            str(record.get("section_title", "(no heading)")),
                        )
                    )
                    st.markdown("**Kisa Alinti**")
                    st.write(_clean_chunk_text_for_display(str(record.get("excerpt", ""))))
                    verify_status = str(record.get("verify_status", "FAIL"))
                    st.caption(f"Verify: {verify_status}")
                    verify_warnings = list(record.get("verify_warnings", []))
                    if verify_warnings:
                        st.caption("Verify warnings: " + ", ".join(verify_warnings))
    elif qa.get("mode") == "comparison":
        comparison_result = qa.get("comparison_result") or {}
        records = comparison_result.get("records", [])
        if not records:
            st.info("Uygun kanıt bulunamadı.")
        else:
            for idx, record in enumerate(records, start=1):
                with st.container(border=True):
                    st.markdown(f"**Kanit {idx}**")
                    st.caption(
                        _source_label(
                            str(record.get("doc_id", "")),
                            str(record.get("quarter", "")),
                            record.get("page", ""),
                            str(record.get("section_title", "(no heading)")),
                        )
                    )
                    st.markdown(
                        f"**Aykilan Deger:** `{record.get('value_raw')}` (unit: {record.get('unit')}, multiplier: {record.get('multiplier')})"
                    )
                    verify_status = str(record.get("verify_status", "FAIL"))
                    st.caption(f"Verify: {verify_status}")
                    verify_warnings = list(record.get("verify_warnings", []))
                    if verify_warnings:
                        st.caption("Verify warnings: " + ", ".join(verify_warnings))
                    st.markdown("**Kisa Alinti**")
                    st.write(_clean_chunk_text_for_display(str(record.get("excerpt", ""))))
    else:
        if not qa["chunks"]:
            st.info("Uygun kanıt bulunamadı.")
        else:
            for idx, chunk in enumerate(qa["chunks"], start=1):
                with st.container(border=True):
                    st.markdown(f"**Kanit {idx}**")
                    st.caption(_source_label(chunk.doc_id, chunk.quarter, chunk.page, chunk.section_title))
                    st.markdown("**Kisa Alinti**")
                    st.write(_short_excerpt(chunk.text, max_chars=320))
                    with st.expander("Tam metni goster"):
                        st.text(_clean_chunk_text_for_display(chunk.text))

    export_text = _markdown_export(
        question=qa["question"],
        retriever_name=qa["retriever"],
        parsed=qa["parsed"],
        answer_text=qa["answer_text"],
        chunks=qa["chunks"],
        comparison_result=qa.get("comparison_result"),
        mode=str(qa.get("mode", "standard")),
    )
    export_name = f"rag_fin_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    st.download_button(
        label="Yanitlari Markdown Olarak Indir",
        data=export_text,
        file_name=export_name,
        mime="text/markdown",
    )


def _render_metrics_tab() -> None:
    st.subheader("3) Performans Olcumu")
    st.caption("Gold soru seti uzerinde v1-v6 retrieval performansi.")

    with st.expander("Metrikler ne anlama geliyor?"):
        st.write("- `hit@1`: Dogru kaynak ilk sirada bulundu mu?")
        st.write("- `hit@3`: Dogru kaynak ilk 3 sonucta var mi?")
        st.write("- `hit@5`: Dogru kaynak ilk 5 sonucta var mi?")
        st.write("- `MRR@5`: Dogru sonucun sirasi ne kadar yuksek?")
        st.write("- `quarter@1`: Top1 sonucu dogru ceyrekte mi?")

    if st.button("metrics_report Calistir"):
        try:
            from src.metrics import run_metrics_report

            with st.spinner("metrics_report calisiyor..."):
                report = run_metrics_report(
                    gold_file=CONFIG.evaluation.gold_file,
                    detailed_output=DEFAULT_DETAILED_OUTPUT,
                    summary_output=DEFAULT_SUMMARY_OUTPUT,
                    week6_summary_output=CONFIG.evaluation.week6_summary_output,
                )
            st.session_state["metrics_summary"] = report["summary"]
            st.success("metrics_report tamamlandi.")
        except Exception as exc:
            st.error(f"metrics_report calisamadi: {exc}")

    summary = st.session_state.get("metrics_summary")
    if summary is None:
        summary = _load_metrics_summary_from_disk()

    if not summary:
        st.info("Henuz metrics sonucu yok. 'metrics_report Calistir' ile baslatin.")
        return

    st.write(f"Generated at: {summary.get('generated_at', '-')}")
    st.write(f"Dataset: {summary.get('dataset', '-')}")
    st.dataframe(_metrics_rows(summary), use_container_width=True, hide_index=True)


def _render_dashboard_tab() -> None:
    st.subheader("4) Dashboard")
    st.caption("Sirket saglik panosu: KPI kartlari, ozet ve degisim analizi")

    companies = _available_companies()
    selected_company = st.selectbox(
        "Sirket",
        options=companies,
        index=0,
        key="dashboard_company",
        format_func=_company_display_name,
    )

    with st.spinner("KPI tablosu hazirlaniyor..."):
        ratio_result = build_ratio_table(
            question=(
                "Q1 Q2 Q3 Q4 net kar brut kar favok satis gelirleri "
                "faaliyet nakit akisi capex serbest nakit akisi "
                "net marj favok marji brut kar marji magaza sayisi trend"
            ),
            retriever=_retriever_v3(),
            company=selected_company,
            top_k_initial=max(CONFIG.retrieval.v3_top_k_initial, 30),
            top_k_final=max(CONFIG.retrieval.top_k_final, 12),
            alpha=CONFIG.retrieval.alpha_v3,
        )

    frame = ratio_result.get("frame")
    confidence_map = ratio_result.get("confidence_map", {})
    extraction_cfg = getattr(CONFIG, "extraction", None)
    low_conf_threshold = float(getattr(extraction_cfg, "low_confidence_threshold", 0.55))
    if frame is None or frame.empty:
        st.warning("Dashboard icin yeterli veri bulunamadi.")
        return

    latest = frame.iloc[-1]
    prev = frame.iloc[-2] if len(frame) >= 2 else None

    def _value_or_na(value: Any, is_percent: bool = False, is_money: bool = False) -> str:
        if value is None or _is_nan(value):
            return "-"
        if is_percent:
            return f"%{float(value):.2f}".replace(".", ",")
        numeric_value = float(value)
        if is_money:
            abs_value = abs(numeric_value)
            if abs_value >= 1_000_000_000:
                return f"{numeric_value / 1_000_000_000:.2f}".replace(".", ",") + " mlr TL"
            if abs_value >= 1_000_000:
                return f"{numeric_value / 1_000_000:.2f}".replace(".", ",") + " mn TL"
            return f"{numeric_value:,.0f}".replace(",", ".") + " TL"
        return f"{numeric_value:,.0f}".replace(",", ".")

    def _delta(current: Any, previous: Any, is_percent: bool = False, is_money: bool = False) -> Optional[str]:
        if (
            current is None
            or previous is None
            or _is_nan(current)
            or _is_nan(previous)
        ):
            return None
        delta_val = float(current) - float(previous)
        sign = "+" if delta_val > 0 else "-" if delta_val < 0 else ""
        if is_percent:
            return f"{sign}{abs(delta_val):.2f}%".replace(".", ",")
        if is_money:
            abs_delta = abs(delta_val)
            if abs_delta >= 1_000_000_000:
                return f"{sign}{abs_delta / 1_000_000_000:.2f}".replace(".", ",") + " mlr TL"
            if abs_delta >= 1_000_000:
                return f"{sign}{abs_delta / 1_000_000:.2f}".replace(".", ",") + " mn TL"
            return f"{sign}{abs_delta:,.0f}".replace(",", ".") + " TL"
        return f"{sign}{abs(delta_val):,.0f}".replace(",", ".")

    def _net_kar_delta(current: Any, previous: Any) -> Optional[str]:
        if (
            current is None
            or previous is None
            or _is_nan(current)
            or _is_nan(previous)
        ):
            return None
        delta_val = float(current) - float(previous)
        if abs(delta_val) < 1e-9:
            return "Yatay"
        direction = "Iyilesme" if delta_val > 0 else "Kotulesme"
        abs_delta = abs(delta_val)
        if abs_delta >= 1_000_000_000:
            amount = f"{abs_delta / 1_000_000_000:.2f}".replace(".", ",") + " mlr TL"
        elif abs_delta >= 1_000_000:
            amount = f"{abs_delta / 1_000_000:.2f}".replace(".", ",") + " mn TL"
        else:
            amount = f"{abs_delta:,.0f}".replace(",", ".") + " TL"
        return f"{direction}: {amount}"

    def _latest_metric_pair(column: str) -> Tuple[Any, Any, Optional[str], Optional[str]]:
        valid = frame[["quarter", column]].dropna(subset=[column])
        if valid.empty:
            return None, None, None, None
        current_row = valid.iloc[-1]
        current_value = current_row[column]
        current_quarter = str(current_row["quarter"])
        previous_value: Any = None
        previous_quarter: Optional[str] = None
        if len(valid) >= 2:
            previous_row = valid.iloc[-2]
            previous_value = previous_row[column]
            previous_quarter = str(previous_row["quarter"])
        return current_value, previous_value, current_quarter, previous_quarter

    def _confidence_detail(metric_key: str, quarter: Optional[str]) -> Dict[str, Any]:
        if not quarter:
            return {}
        return dict(confidence_map.get(metric_key, {}).get(quarter, {}) or {})

    def _confidence_value(metric_key: str, quarter: Optional[str]) -> Optional[float]:
        detail = _confidence_detail(metric_key, quarter)
        raw_conf = detail.get("confidence")
        if raw_conf is None:
            return None
        try:
            return float(raw_conf)
        except Exception:
            return None

    def _render_low_conf_badge(column: Any, metric_key: str, quarter: Optional[str]) -> None:
        conf = _confidence_value(metric_key, quarter)
        if conf is None or conf >= low_conf_threshold:
            return
        column.markdown(
            "<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
            "background:#fff3cd;color:#8a6d3b;font-size:12px;font-weight:600'>Low confidence</span>",
            unsafe_allow_html=True,
        )

    def _verify_status(metric_key: str, quarter: Optional[str]) -> str:
        detail = _confidence_detail(metric_key, quarter)
        return str(detail.get("verify_status", "FAIL"))

    def _render_verify_badge(column: Any, metric_key: str, quarter: Optional[str]) -> None:
        status = _verify_status(metric_key, quarter)
        if status == "PASS":
            color = "#d1f2eb"
            text = "#0b5345"
        elif status == "WARN":
            color = "#fff3cd"
            text = "#8a6d3b"
        else:
            color = "#f8d7da"
            text = "#842029"
        column.markdown(
            f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
            f"background:{color};color:{text};font-size:12px;font-weight:600'>Verify: {status}</span>",
            unsafe_allow_html=True,
        )

    net_kar_cur, net_kar_prev, net_kar_q, _ = _latest_metric_pair("net_kar")
    ciro_cur, ciro_prev, ciro_q, _ = _latest_metric_pair("satis_gelirleri")
    net_margin_cur, net_margin_prev, net_margin_q, _ = _latest_metric_pair("net_margin")
    favok_margin_cur, favok_margin_prev, favok_margin_q, _ = _latest_metric_pair("favok_margin")
    brut_margin_cur, brut_margin_prev, brut_margin_q, _ = _latest_metric_pair("brut_kar_marji")
    rev_growth_cur, rev_growth_prev, rev_growth_q, _ = _latest_metric_pair("revenue_growth_qoq")
    store_growth_cur, store_growth_prev, store_growth_q, _ = _latest_metric_pair("store_growth_qoq")

    row_top = st.columns(4)
    row_bottom = st.columns(3)

    net_kar_label = "Net zarar" if net_kar_cur is not None and not _is_nan(net_kar_cur) and float(net_kar_cur) < 0 else "Net kar"
    row_top[0].metric(
        f"{net_kar_label} ({net_kar_q or '-'})",
        _value_or_na(net_kar_cur, is_money=True),
        _net_kar_delta(net_kar_cur, net_kar_prev),
        delta_color="off",
    )
    _render_low_conf_badge(row_top[0], "net_kar", net_kar_q)
    _render_verify_badge(row_top[0], "net_kar", net_kar_q)
    row_top[0].caption("QoQ kurali: delta, ok yerine iyilesme/kotulesme metniyle gosterilir.")
    row_top[1].metric(
        f"Ciro / Satis ({ciro_q or '-'})",
        _value_or_na(ciro_cur, is_money=True),
        _delta(ciro_cur, ciro_prev, is_money=True),
    )
    _render_low_conf_badge(row_top[1], "satis_gelirleri", ciro_q)
    _render_verify_badge(row_top[1], "satis_gelirleri", ciro_q)
    row_top[2].metric(
        f"Net marj ({net_margin_q or '-'})",
        _value_or_na(net_margin_cur, is_percent=True),
        _delta(net_margin_cur, net_margin_prev, is_percent=True),
    )
    _render_low_conf_badge(row_top[2], "net_margin", net_margin_q)
    _render_verify_badge(row_top[2], "net_margin", net_margin_q)
    row_top[3].metric(
        f"FAVOK marji ({favok_margin_q or '-'})",
        _value_or_na(favok_margin_cur, is_percent=True),
        _delta(favok_margin_cur, favok_margin_prev, is_percent=True),
    )
    _render_low_conf_badge(row_top[3], "favok_margin", favok_margin_q)
    _render_verify_badge(row_top[3], "favok_margin", favok_margin_q)

    row_bottom[0].metric(
        f"Brut kar marji ({brut_margin_q or '-'})",
        _value_or_na(brut_margin_cur, is_percent=True),
        _delta(brut_margin_cur, brut_margin_prev, is_percent=True),
    )
    _render_low_conf_badge(row_bottom[0], "brut_kar_marji", brut_margin_q)
    _render_verify_badge(row_bottom[0], "brut_kar_marji", brut_margin_q)
    row_bottom[1].metric(
        f"Revenue growth ({rev_growth_q or '-'})",
        _value_or_na(rev_growth_cur, is_percent=True),
        _delta(rev_growth_cur, rev_growth_prev, is_percent=True),
    )
    _render_low_conf_badge(row_bottom[1], "revenue_growth_qoq", rev_growth_q)
    _render_verify_badge(row_bottom[1], "revenue_growth_qoq", rev_growth_q)
    row_bottom[2].metric(
        f"Store growth ({store_growth_q or '-'})",
        _value_or_na(store_growth_cur, is_percent=True),
        _delta(store_growth_cur, store_growth_prev, is_percent=True),
    )
    _render_low_conf_badge(row_bottom[2], "store_growth_qoq", store_growth_q)
    _render_verify_badge(row_bottom[2], "store_growth_qoq", store_growth_q)

    net_margin_val = net_margin_cur
    if net_margin_val is not None and not _is_nan(net_margin_val):
        if float(net_margin_val) >= CONFIG.health.net_margin_green_min:
            net_margin_color = "green"
            net_margin_label = "Iyi"
        elif float(net_margin_val) >= CONFIG.health.net_margin_yellow_min:
            net_margin_color = "orange"
            net_margin_label = "Orta"
        else:
            net_margin_color = "red"
            net_margin_label = "Zayif"
        st.markdown(
            f"Net marj durumu: <span style='color:{net_margin_color};font-weight:700'>{net_margin_label}</span>",
            unsafe_allow_html=True,
        )

    def _score(value: Any, green_min: float, yellow_min: float) -> int:
        if value is None or _is_nan(value):
            return 0
        val = float(value)
        if val >= green_min:
            return 2
        if val >= yellow_min:
            return 1
        return 0

    health_score = 0
    health_score += _score(net_margin_cur, CONFIG.health.net_margin_green_min, CONFIG.health.net_margin_yellow_min)
    health_score += _score(favok_margin_cur, CONFIG.health.favok_margin_green_min, CONFIG.health.favok_margin_yellow_min)
    health_score += _score(rev_growth_cur, CONFIG.health.revenue_growth_green_min, CONFIG.health.revenue_growth_yellow_min)
    health_score += _score(store_growth_cur, CONFIG.health.store_growth_green_min, CONFIG.health.store_growth_yellow_min)
    avg_score = health_score / 4.0
    if avg_score >= 1.5:
        health_label = "GREEN"
        health_color = "green"
    elif avg_score >= 0.8:
        health_label = "YELLOW"
        health_color = "orange"
    else:
        health_label = "RED"
        health_color = "red"

    st.markdown(
        f"### Health Label: <span style='color:{health_color}'>{health_label}</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Why? (Confidence detaylari)", expanded=False):
        confidence_rows = [
            ("Net kar", "net_kar", net_kar_q),
            ("Ciro / Satis", "satis_gelirleri", ciro_q),
            ("Net marj", "net_margin", net_margin_q),
            ("FAVOK marji", "favok_margin", favok_margin_q),
            ("Brut kar marji", "brut_kar_marji", brut_margin_q),
            ("Revenue growth", "revenue_growth_qoq", rev_growth_q),
            ("Store growth", "store_growth_qoq", store_growth_q),
        ]
        for label, metric_key, quarter in confidence_rows:
            detail = _confidence_detail(metric_key, quarter)
            conf = _confidence_value(metric_key, quarter)
            verify_status = str(detail.get("verify_status", "FAIL"))
            conf_label = "-" if conf is None else f"{conf:.2f}"
            st.markdown(
                f"**{label} ({quarter or '-'})** - confidence: `{conf_label}` | verify: `{verify_status}`"
            )
            reasons = list(detail.get("reasons", []))
            verify_warnings = list(detail.get("verify_warnings", []))
            evidences = list(detail.get("evidence", []))
            if reasons:
                for reason in reasons:
                    st.write(f"- {reason}")
            if verify_warnings:
                st.caption("Verify warnings:")
                for warning in verify_warnings:
                    st.caption(f"- {warning}")
            if evidences:
                st.caption("Kanit:")
                for ev in evidences:
                    st.caption(f"- {ev}")

    with st.expander("KPI Geri Bildirim (Dogrulama Dongusu)", expanded=False):
        st.caption("Her KPI icin Dogru/Yanlis secin. Yanlis ise dogru degeri veya Bulunamadi secenegini girin.")
        feedback_rows = [
            ("net_kar", net_kar_q, _value_or_na(net_kar_cur, is_money=True)),
            ("satis_gelirleri", ciro_q, _value_or_na(ciro_cur, is_money=True)),
            ("net_margin", net_margin_q, _value_or_na(net_margin_cur, is_percent=True)),
            ("favok_margin", favok_margin_q, _value_or_na(favok_margin_cur, is_percent=True)),
            ("brut_kar_marji", brut_margin_q, _value_or_na(brut_margin_cur, is_percent=True)),
            ("revenue_growth_qoq", rev_growth_q, _value_or_na(rev_growth_cur, is_percent=True)),
            ("store_growth_qoq", store_growth_q, _value_or_na(store_growth_cur, is_percent=True)),
        ]
        with st.form("dashboard_feedback_form", clear_on_submit=False):
            feedback_inputs: List[Dict[str, Any]] = []
            for metric_key, quarter, extracted in feedback_rows:
                st.markdown(f"**{metric_key} ({quarter or '-'})** - `{extracted}`")
                verdict = st.radio(
                    f"{metric_key} geri bildirim",
                    options=["atla", "dogru", "yanlis"],
                    horizontal=True,
                    key=f"fb_verdict_{metric_key}",
                    label_visibility="collapsed",
                )
                user_value: Optional[str] = None
                mark_missing = False
                if verdict == "yanlis":
                    col_fb1, col_fb2 = st.columns([3, 1])
                    user_value = col_fb1.text_input(
                        f"{metric_key} dogru deger",
                        key=f"fb_value_{metric_key}",
                        placeholder="Ornek: 4,8% veya 2.350 mn TL",
                        label_visibility="collapsed",
                    ).strip()
                    mark_missing = col_fb2.checkbox("Bulunamadi", key=f"fb_missing_{metric_key}")
                feedback_inputs.append(
                    {
                        "metric": metric_key,
                        "quarter": quarter,
                        "extracted": extracted,
                        "verdict": verdict,
                        "user_value": user_value,
                        "mark_missing": mark_missing,
                        "evidence_ref": (
                            ((confidence_map.get(metric_key, {}).get(quarter, {}) or {}).get("evidence", []) or [None])[0]
                        ),
                    }
                )
            submitted = st.form_submit_button("Geri Bildirimleri Kaydet")

        if submitted:
            saved = 0
            for row in feedback_inputs:
                verdict = str(row["verdict"])
                if verdict == "atla":
                    continue
                user_value = row["user_value"]
                if row["mark_missing"]:
                    user_value = "Bulunamadi"
                _append_feedback_log(
                    company=selected_company,
                    quarter=row["quarter"],
                    metric=row["metric"],
                    extracted_value=row["extracted"],
                    user_value=user_value,
                    evidence_ref=row["evidence_ref"],
                    verdict=verdict,
                )
                saved += 1
            if saved:
                st.success(f"{saved} geri bildirim kaydedildi.")
            else:
                st.info("Kaydedilecek geri bildirim secilmedi.")

    st.markdown("**Finansal Ozet (Executive Summary)**")
    summary_bullets = build_executive_summary(ratio_result, max_bullets=5)
    for bullet in summary_bullets:
        st.markdown(f"- {bullet}")

    st.markdown("**Degisim Tespiti (Son 2 Ceyrek)**")
    changes = detect_last_quarter_changes(ratio_result)
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.markdown("**Iyilesenler**")
        if changes["improved"]:
            for item in changes["improved"]:
                st.write(f"- {item}")
        else:
            st.write("- Yok")
    with ch2:
        st.markdown("**Kotulesenler**")
        if changes["worsened"]:
            for item in changes["worsened"]:
                st.write(f"- {item}")
        else:
            st.write("- Yok")
    with ch3:
        st.markdown("**Yatay Kalanlar**")
        if changes["flat"]:
            for item in changes["flat"]:
                st.write(f"- {item}")
        else:
            st.write("- Yok")

    st.markdown("**Ratio Tablosu**")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        label="Ratio Tablosunu CSV Indir",
        data=_frame_to_csv_bytes(frame),
        file_name=f"ratio_{selected_company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    try:
        from pandas import DataFrame

        if frame["net_kar"].notna().any():
            st.markdown("**Net Kar Trendi**")
            st.line_chart(DataFrame({"quarter": frame["quarter"], "net_kar": frame["net_kar"]}), x="quarter", y="net_kar")
        if frame["satis_gelirleri"].notna().any():
            st.markdown("**Ciro / Satis Gelirleri Trendi**")
            st.line_chart(
                DataFrame({"quarter": frame["quarter"], "satis_gelirleri": frame["satis_gelirleri"]}),
                x="quarter",
                y="satis_gelirleri",
            )
        if frame["net_margin"].notna().any():
            st.markdown("**Net Marj Trendi**")
            st.line_chart(DataFrame({"quarter": frame["quarter"], "net_margin": frame["net_margin"]}), x="quarter", y="net_margin")
        if frame["favok_margin"].notna().any():
            st.markdown("**FAVOK Marji Trendi**")
            st.line_chart(DataFrame({"quarter": frame["quarter"], "favok_margin": frame["favok_margin"]}), x="quarter", y="favok_margin")
        if frame["brut_kar_marji"].notna().any():
            st.markdown("**Brut Kar Marji Trendi**")
            st.line_chart(
                DataFrame({"quarter": frame["quarter"], "brut_kar_marji": frame["brut_kar_marji"]}),
                x="quarter",
                y="brut_kar_marji",
            )
    except Exception as exc:
        st.info(f"Trend grafik olusturulamadi: {exc}")


def _format_money_short(value: Any, currency: str = "TL") -> str:
    if value is None or _is_nan(value):
        return "-"
    currency_label = str(currency or "TL").upper()
    numeric = float(value)
    abs_value = abs(numeric)
    if currency_label != "TL" and abs_value >= 1_000_000:
        return f"{numeric / 1_000_000:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + f" mn {currency_label}"
    if abs_value >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f}".replace(".", ",") + f" mlr {currency_label}"
    if abs_value >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}".replace(".", ",") + f" mn {currency_label}"
    return f"{numeric:,.0f}".replace(",", ".") + f" {currency_label}"


def _format_pct_short(value: Any) -> str:
    if value is None or _is_nan(value):
        return "-"
    return f"%{float(value):.2f}".replace(".", ",")


def _render_money_bar_chart(slot: Any, frame: Any, value_col: str, title: str, currency: str = "TL") -> None:
    slot.markdown(f"**{title}**")
    if frame is None or value_col not in frame.columns:
        slot.info(f"{title} icin veri yok.")
        return
    chart_frame = frame[["quarter", value_col]].dropna().copy()
    if chart_frame.empty:
        slot.info(f"{title} icin veri yok.")
        return

    currency_label = str(currency or "TL").upper()
    if currency_label == "TL":
        scale_divisor = 1_000_000_000.0
        unit_short = "mlr"
        unit_label = "milyar"
    else:
        # Non-TL currencies are shown in millions for readability.
        scale_divisor = 1_000_000.0
        unit_short = "mn"
        unit_label = "milyon"
    chart_frame["value_scaled"] = chart_frame[value_col].astype(float) / scale_divisor
    slot.caption(f"Birim: {unit_label} {currency_label}")

    if alt is None:
        slot.bar_chart(chart_frame, x="quarter", y="value_scaled")
        return

    chart = (
        alt.Chart(chart_frame)
        .mark_bar(color="#7db9e8", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("quarter:N", title="Ceyrek", sort=["Q1", "Q2", "Q3", "Q4"]),
            y=alt.Y("value_scaled:Q", title=f"{unit_label.title()} {currency_label}"),
            tooltip=[
                alt.Tooltip("quarter:N", title="Ceyrek"),
                alt.Tooltip("value_scaled:Q", title=f"Deger ({unit_short} {currency_label})", format=",.2f"),
            ],
        )
        .properties(height=260)
    )
    slot.altair_chart(chart, use_container_width=True)


def _latest_and_prev(frame: Any, column: str, period: str) -> Tuple[Any, Any, Optional[str]]:
    valid = frame[["quarter", column]].dropna(subset=[column]) if frame is not None else None
    if valid is None or valid.empty:
        return None, None, None
    if period in {"Q1", "Q2", "Q3", "Q4"}:
        row = valid[valid["quarter"] == period]
        if row.empty:
            return None, None, period
        current = row.iloc[-1][column]
        prev_candidates = valid[valid["quarter"] < period]
        prev = prev_candidates.iloc[-1][column] if not prev_candidates.empty else None
        return current, prev, period
    current_row = valid.iloc[-1]
    prev = valid.iloc[-2][column] if len(valid) >= 2 else None
    return current_row[column], prev, str(current_row["quarter"])


def _record_for_metric_quarter(
    metric_records: Dict[str, List[Dict[str, Any]]],
    metric_key: str,
    quarter: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not quarter:
        return None
    search_keys = [metric_key]
    if metric_key == "net_margin":
        search_keys = ["net_kar_marji", "net_margin"]
    elif metric_key == "favok_margin":
        search_keys = ["favok_marji", "favok_margin"]

    for key in search_keys:
        for record in metric_records.get(key, []):
            if str(record.get("quarter")) == str(quarter):
                return record
    return None


def _split_excerpt_lines(excerpt: str, max_lines: int = 3) -> List[str]:
    compact = " ".join(str(excerpt or "").split())
    if not compact:
        return []
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", compact) if item.strip()]
    if len(sentences) >= 2:
        return sentences[:max(2, max_lines)]

    wrapped = [item.strip() for item in textwrap.wrap(compact, width=95) if item.strip()]
    if len(wrapped) >= 2:
        return wrapped[:max(2, max_lines)]
    if not wrapped:
        return []
    line = wrapped[0]
    if len(line) > 65:
        split_pos = line.rfind(" ", 20, len(line) - 20)
        if split_pos > 0:
            return [line[:split_pos].strip(), line[split_pos + 1 :].strip()][:max(2, max_lines)]
    return wrapped[:max_lines]


def _llm_assistant_enabled() -> bool:
    llm_cfg = getattr(CONFIG, "llm_assistant", None) or getattr(CONFIG, "llm_commentary", None)
    return bool(getattr(llm_cfg, "enabled", False))


def _render_commentary_box(
    commentary: Dict[str, Any],
    *,
    title: str = "Kisa yorum",
    expanded: bool = False,
) -> None:
    if not commentary:
        return
    error_msg = str(commentary.get("_error", "")).strip()
    model_name = str(commentary.get("_model", "")).strip()
    has_content = _commentary_has_content(commentary)
    if not has_content and not error_msg:
        return
    with st.expander(title, expanded=expanded):
        if has_content:
            headline = str(commentary.get("headline", "")).strip()
            bullets = [str(item).strip() for item in list(commentary.get("bullets", [])) if str(item).strip()]
            risk_note = str(commentary.get("risk_note", "")).strip()
            next_question = str(commentary.get("next_question", "")).strip()
            if headline:
                st.markdown(f"**{headline}**")
            for bullet in bullets:
                st.markdown(f"- {bullet}")
            if risk_note:
                st.caption(f"Risk notu: {risk_note}")
            if next_question:
                st.caption(f"Sonraki soru: {next_question}")
            if model_name:
                st.caption(f"Model: `{model_name}`")
        if error_msg:
            st.error(f"AI Hata: {error_msg}")


def _render_overview_page() -> None:
    st.subheader("Genel Bakış")
    st.caption("KPI kartlari, trendler ve guvenilirlik detayi")

    if not _has_indexed_reports():
        with st.container(border=True):
            st.markdown("### Getting Started")
            st.write("1. Raporlar sayfasindan PDF yukleyin.")
            st.write("2. Sirket / ceyrek / yil otomatik algilansin.")
            st.write("3. Tek tusla ice alma + indexleme yapin.")
            st.write("4. Bu sayfada KPI ve trendleri gorun.")
            st.write("5. Soru Sor sayfasinda sorularinizi yazin.")
            st.caption("Hizli demo icin: `ragfin-demo` veya `python -m ragfin.demo`")

            action_col1, action_col2, _ = st.columns([1, 1, 2])
            if action_col1.button("Raporlar sayfasina git", key="overview_go_reports"):
                st.session_state["nav_page"] = "Reports"
                st.rerun()
            if action_col2.button("Run Demo", key="overview_run_demo", type="primary"):
                try:
                    from src.demo import bootstrap_sample_into_config

                    with st.spinner("Demo veri seti hazirlaniyor..."):
                        summary = bootstrap_sample_into_config(config_path=CONFIG.path, clean_workspace=False)
                    _clear_cached_retrievers()
                    st.success(
                        "Demo dataset yuklendi: "
                        + f"{summary.get('pages_loaded', 0)} sayfa, "
                        + f"{(summary.get('index_summary') or {}).get('indexed_chunks', 0)} chunk indexlendi."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Run Demo basarisiz: {exc}")
                    st.info("Terminalden deneyin: `ragfin-demo` veya `python -m ragfin.demo`")

            sample_pdf = _find_sample_pdf()
            if st.button("Load sample report", key="overview_load_sample", type="primary"):
                if sample_pdf is None:
                    st.info("Repo icinde ornek PDF bulunamadi. Raporlar sayfasindan kendi PDF dosyanizi yukleyin.")
                else:
                    target = RAW_DIR / sample_pdf.name
                    copied = False
                    if sample_pdf.resolve() != target.resolve():
                        target.write_bytes(sample_pdf.read_bytes())
                        copied = True
                    ok, message = _run_ingest_and_index_v2(saved_files=[target.name if copied else sample_pdf.name])
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

            if sample_pdf is None:
                st.caption("Ornek dosya bulunamadi. Lutfen manuel PDF yukleyin.")
            else:
                st.caption(f"Ornek dosya hazir: {sample_pdf.name}")
        return

    companies = _available_companies()
    if not companies:
        st.info("Henuz sirket verisi bulunamadi. Raporlar sayfasindan PDF yukleyin.")
        return

    top_col1, top_col2 = st.columns([2, 1])
    selected_company = top_col1.selectbox(
        "Sirket",
        companies,
        key="overview_company",
        format_func=_company_display_name,
    )
    period = top_col2.selectbox("Donem", ["Latest", "Q1", "Q2", "Q3", "Q4"], index=0, key="overview_period")

    settings = _get_ui_settings()
    ratio_result = build_ratio_table(
        question=(
            "Q1 Q2 Q3 Q4 net kar brut kar favok satis gelirleri "
            "faaliyet nakit akisi capex serbest nakit akisi "
            "net marj favok marji brut kar marji magaza sayisi trend"
        ),
        retriever=_retriever_v3(),
        company=selected_company,
        top_k_initial=max(int(settings.get("top_k_initial_v3", CONFIG.retrieval.v3_top_k_initial)), 30),
        top_k_final=max(int(settings.get("top_k_final", CONFIG.retrieval.top_k_final)), 12),
        alpha=float(settings.get("alpha_v3", CONFIG.retrieval.alpha_v3)),
    )
    frame = ratio_result.get("frame")
    confidence_map = ratio_result.get("confidence_map", {})
    metric_records = ratio_result.get("metric_records", {})

    if frame is None or frame.empty:
        st.info("Secili sirket icin KPI verisi bulunamadi.")
        return

    def confidence_detail(metric_key: str, quarter: Optional[str]) -> Dict[str, Any]:
        if not quarter:
            return {}
        return dict(confidence_map.get(metric_key, {}).get(quarter, {}) or {})

    def metric_currency(metric_key: str, quarter: Optional[str]) -> str:
        record = _record_for_metric_quarter(metric_records, metric_key, quarter)
        if record:
            candidate = str(record.get("currency", "")).strip().upper()
            if candidate:
                return candidate
        for record in metric_records.get(metric_key, []):
            candidate = str(record.get("currency", "")).strip().upper()
            if candidate:
                return candidate
        return "TL"

    def delta_text(cur: Any, prev: Any, pct: bool = False, money: bool = False, currency: str = "TL") -> str:
        if cur is None or prev is None or _is_nan(cur) or _is_nan(prev):
            return "-"
        d = float(cur) - float(prev)
        sign = "+" if d > 0 else "-" if d < 0 else ""
        if pct:
            return f"{sign}{abs(d):.2f}%".replace(".", ",")
        if money:
            return sign + _format_money_short(abs(d), currency=currency)
        return f"{sign}{abs(d):,.0f}".replace(",", ".")

    def render_trust(column: Any, metric_key: str, quarter: Optional[str]) -> None:
        detail = confidence_detail(metric_key, quarter)
        badge = trust_badge_html(detail.get("confidence"), detail.get("verify_status"))
        column.markdown(badge, unsafe_allow_html=True)
        trust = trust_level(detail.get("confidence"), detail.get("verify_status"))
        if trust["level"] == "Low":
            column.caption("Bu deger tabloda kaymis olabilir; kaniti acip dogrulayin.")

    net_kar_cur, net_kar_prev, net_kar_q = _latest_and_prev(frame, "net_kar", period)
    brut_kar_cur, brut_kar_prev, brut_kar_q = _latest_and_prev(frame, "brut_kar", period)
    sales_cur, sales_prev, sales_q = _latest_and_prev(frame, "satis_gelirleri", period)
    favok_cur, favok_prev, favok_q = _latest_and_prev(frame, "favok", period)
    operating_cf_cur, operating_cf_prev, operating_cf_q = _latest_and_prev(frame, "faaliyet_nakit_akisi", period)
    capex_cur, capex_prev, capex_q = _latest_and_prev(frame, "capex", period)
    free_cf_cur, free_cf_prev, free_cf_q = _latest_and_prev(frame, "serbest_nakit_akisi", period)
    net_margin_cur, net_margin_prev, net_margin_q = _latest_and_prev(frame, "net_margin", period)
    favok_margin_cur, favok_margin_prev, favok_margin_q = _latest_and_prev(frame, "favok_margin", period)
    brut_margin_cur, brut_margin_prev, brut_margin_q = _latest_and_prev(frame, "brut_kar_marji", period)
    store_cur, store_prev, store_q = _latest_and_prev(frame, "magaza_sayisi", period)
    net_kar_currency = metric_currency("net_kar", net_kar_q)
    brut_kar_currency = metric_currency("brut_kar", brut_kar_q)
    sales_currency = metric_currency("satis_gelirleri", sales_q)
    favok_currency = metric_currency("favok", favok_q)
    operating_cf_currency = metric_currency("faaliyet_nakit_akisi", operating_cf_q)
    capex_currency = metric_currency("capex", capex_q)
    free_cf_currency = metric_currency("serbest_nakit_akisi", free_cf_q)

    detail_state_key = "overview_selected_kpi"

    def select_detail(
        metric_key: str,
        quarter: Optional[str],
        label: str,
        value_display: str,
        unit: str,
        currency: Optional[str] = None,
    ) -> None:
        st.session_state[detail_state_key] = {
            "company": selected_company,
            "metric": metric_key,
            "quarter": quarter,
            "label": label,
            "value_display": value_display,
            "unit": unit,
            "currency": str(currency or ""),
        }

    card_rows = st.columns(4)
    net_label = "Net zarar" if net_kar_cur is not None and not _is_nan(net_kar_cur) and float(net_kar_cur) < 0 else "Net kar"
    net_display = _format_money_short(net_kar_cur, currency=net_kar_currency)
    card_rows[0].metric(
        f"{net_label} ({net_kar_q or '-'})",
        net_display,
        delta_text(net_kar_cur, net_kar_prev, money=True, currency=net_kar_currency),
    )
    render_trust(card_rows[0], "net_kar", net_kar_q)
    if card_rows[0].button("Details", key=f"ov_detail_net_kar_{selected_company}_{net_kar_q}_{period}"):
        select_detail("net_kar", net_kar_q, net_label, net_display, "money", currency=net_kar_currency)

    brut_kar_display = _format_money_short(brut_kar_cur, currency=brut_kar_currency)
    card_rows[1].metric(
        f"Brut kar ({brut_kar_q or '-'})",
        brut_kar_display,
        delta_text(brut_kar_cur, brut_kar_prev, money=True, currency=brut_kar_currency),
    )
    render_trust(card_rows[1], "brut_kar", brut_kar_q)
    if card_rows[1].button("Details", key=f"ov_detail_brut_kar_{selected_company}_{brut_kar_q}_{period}"):
        select_detail("brut_kar", brut_kar_q, "Brut kar", brut_kar_display, "money", currency=brut_kar_currency)

    sales_display = _format_money_short(sales_cur, currency=sales_currency)
    card_rows[2].metric(
        f"Satis Gelirleri ({sales_q or '-'})",
        sales_display,
        delta_text(sales_cur, sales_prev, money=True, currency=sales_currency),
    )
    render_trust(card_rows[2], "satis_gelirleri", sales_q)
    if card_rows[2].button("Details", key=f"ov_detail_sales_{selected_company}_{sales_q}_{period}"):
        select_detail("satis_gelirleri", sales_q, "Satis Gelirleri", sales_display, "money", currency=sales_currency)

    favok_display = _format_money_short(favok_cur, currency=favok_currency)
    card_rows[3].metric(
        f"FAVOK ({favok_q or '-'})",
        favok_display,
        delta_text(favok_cur, favok_prev, money=True, currency=favok_currency),
    )
    render_trust(card_rows[3], "favok", favok_q)
    if card_rows[3].button("Details", key=f"ov_detail_favok_{selected_company}_{favok_q}_{period}"):
        select_detail("favok", favok_q, "FAVOK", favok_display, "money", currency=favok_currency)

    card_rows2 = st.columns(4)
    net_margin_display = _format_pct_short(net_margin_cur)
    card_rows2[0].metric(
        f"Net marj ({net_margin_q or '-'})",
        net_margin_display,
        delta_text(net_margin_cur, net_margin_prev, pct=True),
    )
    render_trust(card_rows2[0], "net_margin", net_margin_q)
    if card_rows2[0].button("Details", key=f"ov_detail_net_margin_{selected_company}_{net_margin_q}_{period}"):
        select_detail("net_margin", net_margin_q, "Net marj", net_margin_display, "%")

    favok_margin_display = _format_pct_short(favok_margin_cur)
    card_rows2[1].metric(
        f"FAVOK marji ({favok_margin_q or '-'})",
        favok_margin_display,
        delta_text(favok_margin_cur, favok_margin_prev, pct=True),
    )
    render_trust(card_rows2[1], "favok_margin", favok_margin_q)
    if card_rows2[1].button("Details", key=f"ov_detail_favok_margin_{selected_company}_{favok_margin_q}_{period}"):
        select_detail("favok_margin", favok_margin_q, "FAVOK marji", favok_margin_display, "%")

    brut_margin_display = _format_pct_short(brut_margin_cur)
    card_rows2[2].metric(
        f"Brut kar marji ({brut_margin_q or '-'})",
        brut_margin_display,
        delta_text(brut_margin_cur, brut_margin_prev, pct=True),
    )
    render_trust(card_rows2[2], "brut_kar_marji", brut_margin_q)
    if card_rows2[2].button("Details", key=f"ov_detail_brut_margin_{selected_company}_{brut_margin_q}_{period}"):
        select_detail("brut_kar_marji", brut_margin_q, "Brut kar marji", brut_margin_display, "%")

    store_display = "-" if store_cur is None or _is_nan(store_cur) else f"{float(store_cur):,.0f}".replace(",", ".")
    card_rows2[3].metric(f"Magaza sayisi ({store_q or '-'})", store_display, delta_text(store_cur, store_prev))
    render_trust(card_rows2[3], "magaza_sayisi", store_q)
    if card_rows2[3].button("Details", key=f"ov_detail_store_{selected_company}_{store_q}_{period}"):
        select_detail("magaza_sayisi", store_q, "Magaza sayisi", store_display, "count")

    card_rows3 = st.columns(3)
    operating_cf_display = _format_money_short(operating_cf_cur, currency=operating_cf_currency)
    card_rows3[0].metric(
        f"Faaliyet nakit akisi ({operating_cf_q or '-'})",
        operating_cf_display,
        delta_text(operating_cf_cur, operating_cf_prev, money=True, currency=operating_cf_currency),
    )
    render_trust(card_rows3[0], "faaliyet_nakit_akisi", operating_cf_q)
    if card_rows3[0].button("Details", key=f"ov_detail_operating_cf_{selected_company}_{operating_cf_q}_{period}"):
        select_detail(
            "faaliyet_nakit_akisi",
            operating_cf_q,
            "Faaliyet nakit akisi",
            operating_cf_display,
            "money",
            currency=operating_cf_currency,
        )

    capex_display = _format_money_short(capex_cur, currency=capex_currency)
    card_rows3[1].metric(
        f"CAPEX ({capex_q or '-'})",
        capex_display,
        delta_text(capex_cur, capex_prev, money=True, currency=capex_currency),
    )
    render_trust(card_rows3[1], "capex", capex_q)
    if card_rows3[1].button("Details", key=f"ov_detail_capex_{selected_company}_{capex_q}_{period}"):
        select_detail("capex", capex_q, "CAPEX", capex_display, "money", currency=capex_currency)

    free_cf_display = _format_money_short(free_cf_cur, currency=free_cf_currency)
    card_rows3[2].metric(
        f"Serbest nakit akisi ({free_cf_q or '-'})",
        free_cf_display,
        delta_text(free_cf_cur, free_cf_prev, money=True, currency=free_cf_currency),
    )
    render_trust(card_rows3[2], "serbest_nakit_akisi", free_cf_q)
    if card_rows3[2].button("Details", key=f"ov_detail_free_cf_{selected_company}_{free_cf_q}_{period}"):
        select_detail("serbest_nakit_akisi", free_cf_q, "Serbest nakit akisi", free_cf_display, "money", currency=free_cf_currency)

    selected_detail = st.session_state.get(detail_state_key) or {}
    if selected_detail and selected_detail.get("company") == selected_company:
        metric_key = str(selected_detail.get("metric", ""))
        quarter = selected_detail.get("quarter")
        label = str(selected_detail.get("label", metric_key))
        value_display = str(selected_detail.get("value_display", "-"))
        unit = str(selected_detail.get("unit", "-"))
        currency = str(selected_detail.get("currency", "")).strip().upper() or metric_currency(metric_key, quarter)
        unit_display = currency if unit == "money" else unit
        detail = confidence_detail(metric_key, quarter)
        trust = trust_level(detail.get("confidence"), detail.get("verify_status"))

        with st.container(border=True):
            st.markdown(f"### {label} - Nereden Geldi?")
            st.markdown(f"**Deger:** `{value_display}`  |  **Donem:** `{quarter or '-'}`  |  **Birim:** `{unit_display}`")
            st.markdown(trust_badge_html(detail.get("confidence"), detail.get("verify_status")), unsafe_allow_html=True)
            if trust["level"] == "Low":
                st.warning(trust["hint"])

            if metric_key in {"net_margin", "favok_margin", "serbest_nakit_akisi"}:
                if metric_key == "net_margin":
                    formula = "net_margin = net_kar / satis_gelirleri * 100"
                    input_metrics = [("Net kar", "net_kar"), ("Satis Gelirleri", "satis_gelirleri")]
                elif metric_key == "serbest_nakit_akisi":
                    formula = "serbest_nakit_akisi = faaliyet_nakit_akisi - capex"
                    input_metrics = [("Faaliyet nakit akisi", "faaliyet_nakit_akisi"), ("CAPEX", "capex")]
                else:
                    formula = "favok_margin = favok / satis_gelirleri * 100"
                    input_metrics = [("FAVOK", "favok"), ("Satis Gelirleri", "satis_gelirleri")]
                st.markdown("**Formula**")
                st.code(formula)
                st.markdown("**Girdi metrikleri ve kanitlar**")
                for input_label, input_metric in input_metrics:
                    row = frame[frame["quarter"] == quarter]
                    input_value = None if row.empty else row.iloc[-1].get(input_metric)
                    input_currency = metric_currency(input_metric, quarter)
                    input_display = (
                        _format_money_short(input_value, currency=input_currency)
                        if input_metric in {"net_kar", "satis_gelirleri", "favok", "faaliyet_nakit_akisi", "capex", "serbest_nakit_akisi"}
                        else str(input_value)
                    )
                    record = _record_for_metric_quarter(metric_records, input_metric, quarter)
                    st.markdown(f"- **{input_label}:** `{input_display}`")
                    if record:
                        rec_year = _extract_year_from_doc_id(str(record.get("doc_id", ""))) or "-"
                        st.caption(
                            "  Kaynak: "
                            + f"{record.get('company', selected_company)} | {rec_year} | {record.get('quarter', '-')} | "
                            + f"s.{record.get('page', '-')} | {record.get('section_title', '(no heading)')}"
                        )
                        excerpt_lines = _split_excerpt_lines(str(record.get("excerpt", "")), max_lines=3)
                        for line in excerpt_lines:
                            st.caption("  " + line)
                    else:
                        st.caption("  Kaynak kaydi bulunamadi.")
            else:
                record = _record_for_metric_quarter(metric_records, metric_key, quarter)
                if record:
                    rec_year = _extract_year_from_doc_id(str(record.get("doc_id", ""))) or "-"
                    st.markdown("**Citation**")
                    st.caption(
                        f"{record.get('company', selected_company)} | {rec_year} | {record.get('quarter', '-')} | "
                        f"s.{record.get('page', '-')} | {record.get('section_title', '(no heading)')}"
                    )
                    st.markdown("**Evidence excerpt**")
                    excerpt_lines = _split_excerpt_lines(str(record.get("excerpt", "")), max_lines=4)
                    for line in excerpt_lines:
                        st.write(f"- {line}")
                else:
                    st.info("Bu KPI icin dogrudan kayit bulunamadi. Guven detayindan kanitlari kontrol edin.")

            st.markdown("**Feedback**")
            form_key = f"overview_kpi_feedback_{selected_company}_{metric_key}_{quarter}"
            with st.form(form_key, clear_on_submit=False):
                verdict = st.radio(
                    "Bu KPI degeri dogru mu?",
                    options=["Dogru", "Yanlis", "Emin degilim"],
                    horizontal=True,
                )
                corrected_value = ""
                note = ""
                if verdict == "Yanlis":
                    corrected_value = st.text_input("Dogru deger", placeholder="Orn: 4,80% veya 11,37 mlr TL")
                    note = st.text_input("Not (opsiyonel)", placeholder="Kisa aciklama")
                submitted = st.form_submit_button("Geri bildirim gonder")

            if submitted:
                verdict_map = {"Dogru": "dogru", "Yanlis": "yanlis", "Emin degilim": "not_sure"}
                verdict_value = verdict_map[verdict]
                evidence_ref = None
                record = _record_for_metric_quarter(metric_records, metric_key, quarter)
                if record:
                    evidence_ref = _source_label(
                        str(record.get("doc_id", "")),
                        str(record.get("quarter", "")),
                        record.get("page", "-"),
                        str(record.get("section_title", "(no heading)")),
                    )
                _append_feedback_log(
                    company=selected_company,
                    quarter=quarter,
                    metric=metric_key,
                    extracted_value=value_display,
                    user_value=corrected_value.strip() or None,
                    evidence_ref=evidence_ref,
                    verdict=verdict_value,
                    note=note.strip() or None,
                )
                if verdict_value in {"dogru", "yanlis"}:
                    api_payload = {
                        "company": selected_company,
                        "quarter": quarter,
                        "metric": metric_key,
                        "extracted_value": value_display,
                        "user_value": corrected_value.strip() or None,
                        "evidence_ref": evidence_ref,
                        "verdict": verdict_value,
                    }
                    _post_feedback_to_api(api_payload)
                st.success("Thanks - this will improve future extraction.")

    st.markdown("**Finansal Buyukluk Trendleri (Sutun)**")
    bar_col1, bar_col2, bar_col3, bar_col4 = st.columns(4)

    _render_money_bar_chart(bar_col1, frame=frame, value_col="net_kar", title="Net kar trendi", currency=metric_currency("net_kar", net_kar_q))
    _render_money_bar_chart(bar_col2, frame=frame, value_col="brut_kar", title="Brut kar trendi", currency=metric_currency("brut_kar", brut_kar_q))
    _render_money_bar_chart(
        bar_col3,
        frame=frame,
        value_col="satis_gelirleri",
        title="Ciro trendi",
        currency=metric_currency("satis_gelirleri", sales_q),
    )
    _render_money_bar_chart(bar_col4, frame=frame, value_col="favok", title="FAVOK trendi", currency=metric_currency("favok", favok_q))

    st.markdown("**Nakit Akisi Trendleri (Sutun)**")
    cash_col1, cash_col2, cash_col3 = st.columns(3)
    _render_money_bar_chart(
        cash_col1,
        frame=frame,
        value_col="faaliyet_nakit_akisi",
        title="Faaliyet nakit akisi trendi",
        currency=metric_currency("faaliyet_nakit_akisi", operating_cf_q),
    )
    _render_money_bar_chart(cash_col2, frame=frame, value_col="capex", title="CAPEX trendi", currency=metric_currency("capex", capex_q))
    _render_money_bar_chart(
        cash_col3,
        frame=frame,
        value_col="serbest_nakit_akisi",
        title="Serbest nakit akisi trendi",
        currency=metric_currency("serbest_nakit_akisi", free_cf_q),
    )

    st.markdown("**Marj Trendleri (Cizgi)**")
    line_col1, line_col2, line_col3 = st.columns(3)

    with line_col1:
        st.markdown("**Net marj trendi**")
        net_margin_frame = frame[["quarter", "net_margin"]].dropna()
        if net_margin_frame.empty:
            st.info("Net marj trendi icin veri yok.")
        else:
            st.caption("Birim: %")
            st.line_chart(net_margin_frame, x="quarter", y="net_margin")

    with line_col2:
        st.markdown("**FAVOK marji trendi**")
        favok_margin_frame = frame[["quarter", "favok_margin"]].dropna()
        if favok_margin_frame.empty:
            st.info("FAVOK marji trendi icin veri yok.")
        else:
            st.caption("Birim: %")
            st.line_chart(favok_margin_frame, x="quarter", y="favok_margin")

    with line_col3:
        st.markdown("**Brut kar marji trendi**")
        brut_margin_frame = frame[["quarter", "brut_kar_marji"]].dropna()
        if brut_margin_frame.empty:
            st.info("Brut kar marji trendi icin veri yok.")
        else:
            st.caption("Birim: %")
            st.line_chart(brut_margin_frame, x="quarter", y="brut_kar_marji")

    st.markdown("**Bu ceyrekte ne degisti?**")
    changes = detect_last_quarter_changes(ratio_result)
    c1, c2, c3 = st.columns(3)
    c1.write("Iyilesenler")
    c1.write(", ".join(changes["improved"]) if changes["improved"] else "-")
    c2.write("Kotulesenler")
    c2.write(", ".join(changes["worsened"]) if changes["worsened"] else "-")
    c3.write("Yatay")
    c3.write(", ".join(changes["flat"]) if changes["flat"] else "-")

    if _llm_assistant_enabled():
        period_hint = period if period in {"Q1", "Q2", "Q3", "Q4"} else None
        model_col, btn_col = st.columns([2, 1])
        current_model = str(settings.get("llm_model", "")).strip() or str(DEFAULT_UI_SETTINGS["llm_model"])
        model_options = list(AI_MODEL_OPTIONS)
        if current_model not in model_options:
            model_options.insert(0, current_model)
        model_idx = model_options.index(current_model) if current_model in model_options else 0
        selected_ai_model = model_col.selectbox(
            "AI Model",
            options=model_options,
            index=model_idx,
            key=f"overview_model_select::{selected_company}::{period}",
        )
        settings["llm_model"] = selected_ai_model
        commentary_key = (
            f"overview_ai_commentary::{selected_company}::{period_hint or 'Latest'}::{selected_ai_model}"
        )
        if btn_col.button("AI Assistanta Sor", key=f"overview_ai_button::{selected_company}::{period}"):
            answer_payload = _overview_to_commentary_answer_payload(
                ratio_result=ratio_result,
                period_hint=period_hint,
                company=selected_company,
            )
            with st.spinner("AI yorum hazirlaniyor..."):
                st.session_state[commentary_key] = _post_commentary_to_api(
                    question=f"{selected_company} {period_hint or 'Latest'} ceyreginde ne degisti?",
                    answer_payload=answer_payload,
                    company=selected_company,
                    quarter=period_hint,
                    model_override=selected_ai_model,
                )
        overview_commentary = dict(st.session_state.get(commentary_key) or {})
        _render_commentary_box(overview_commentary, title="Kisa yorum", expanded=True)
        if _commentary_has_content(overview_commentary):
            st.caption("Bu sadece yorumdur; sayilar kanitlardan gelir.")

    with st.expander("Guven & Kanit Detayi", expanded=False):
        for metric_key, quarter in [
            ("net_kar", net_kar_q),
            ("brut_kar", brut_kar_q),
            ("satis_gelirleri", sales_q),
            ("favok", favok_q),
            ("faaliyet_nakit_akisi", operating_cf_q),
            ("capex", capex_q),
            ("serbest_nakit_akisi", free_cf_q),
            ("net_margin", net_margin_q),
            ("favok_margin", favok_margin_q),
            ("brut_kar_marji", brut_margin_q),
            ("magaza_sayisi", store_q),
        ]:
            detail = confidence_detail(metric_key, quarter)
            trust = trust_level(detail.get("confidence"), detail.get("verify_status"))
            conf_raw = detail.get("confidence")
            conf_display = "-" if conf_raw is None else f"{float(conf_raw):.2f}"
            st.markdown(
                f"**{metric_key} ({quarter or '-'})** | guven: `{trust['label']}` | confidence: `{conf_display}`"
            )
            warnings = list(detail.get("verify_warnings", []))
            if warnings:
                st.caption("Uyarilar: " + ", ".join(warnings))
            evidences = list(detail.get("evidence", []))
            for ev in evidences[:2]:
                st.caption(f"- {ev}")


def _render_companies_page() -> None:
    st.subheader("Şirketler")
    companies = _available_companies()
    if not companies:
        st.info("Şirket listesi boş. Önce rapor yükleyin.")
        return

    period = st.selectbox("Dönem", ["Latest", "Q1", "Q2", "Q3", "Q4"], index=0, key="companies_period")
    metric = st.selectbox(
        "Gösterilecek KPI",
        [
            "net_kar",
            "brut_kar",
            "satis_gelirleri",
            "favok",
            "faaliyet_nakit_akisi",
            "capex",
            "serbest_nakit_akisi",
            "net_margin",
            "favok_margin",
            "brut_kar_marji",
            "magaza_sayisi",
        ],
        index=0,
        key="companies_metric",
    )
    settings = _get_ui_settings()
    rows: List[Dict[str, Any]] = []
    for company in companies:
        result = build_ratio_table(
            question=(
                "Q1 Q2 Q3 Q4 net kar brut kar favok satis gelirleri "
                "faaliyet nakit akisi capex serbest nakit akisi "
                "net marj favok marji brut kar marji magaza sayisi trend"
            ),
            retriever=_retriever_v3(),
            company=company,
            top_k_initial=max(int(settings.get("top_k_initial_v3", CONFIG.retrieval.v3_top_k_initial)), 30),
            top_k_final=max(int(settings.get("top_k_final", CONFIG.retrieval.top_k_final)), 12),
            alpha=float(settings.get("alpha_v3", CONFIG.retrieval.alpha_v3)),
        )
        frame = result.get("frame")
        if frame is None or frame.empty:
            rows.append({"company": _company_display_name(company), "quarter": "-", "value": "-", "trust": "Low"})
            continue
        current, _, q = _latest_and_prev(frame, metric, period)
        metric_records = result.get("metric_records", {}) or {}
        metric_currency = "TL"
        for record in metric_records.get(metric, []):
            if q and str(record.get("quarter")) != str(q):
                continue
            candidate = str(record.get("currency", "")).strip().upper()
            if candidate:
                metric_currency = candidate
                break
        value_display = (
            _format_pct_short(current)
            if "marj" in metric
            else _format_money_short(current, currency=metric_currency)
            if metric in {"net_kar", "brut_kar", "satis_gelirleri", "favok", "faaliyet_nakit_akisi", "capex", "serbest_nakit_akisi"}
            else ("-" if current is None or _is_nan(current) else f"{float(current):,.0f}".replace(",", "."))
        )
        confidence_detail = (result.get("confidence_map", {}).get(metric, {}).get(q or "", {}) or {})
        trust = trust_level(confidence_detail.get("confidence"), confidence_detail.get("verify_status")).get("label", "Dusuk")
        rows.append({"company": _company_display_name(company), "quarter": q or "-", "value": value_display, "trust": trust})

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Karşılaştırma görünümü: şirket seçip Overview sayfasında detay trendleri inceleyin.")


def _render_reports_page() -> None:
    st.subheader("Raporlar")
    st.caption("Yeni dosya yükleyin ve tek adımda ingest + incremental index v2 çalıştırın.")

    uploaded = st.file_uploader("PDF Yükle", type=["pdf"], accept_multiple_files=True, key="reports_upload")
    if uploaded:
        detected_rows = [_detect_pdf_metadata_from_name(getattr(item, "name", "")) for item in uploaded]
        st.dataframe(detected_rows, use_container_width=True, hide_index=True)

    primary = st.button("Yeni Dosyaları İçe Al + Incremental Indexle", type="primary")
    if primary:
        saved = _save_uploaded_pdfs(uploaded or [])
        ok, message = _run_ingest_and_index_v2(saved_files=saved)
        if ok:
            st.success(message)
        else:
            st.error(message)

    docs_rows = _build_indexed_docs_rows()
    if docs_rows:
        st.markdown("**İndekslenen Dokümanlar**")
        st.dataframe(docs_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Henüz rapor bulunmuyor.")

    with st.expander("Gelişmiş İşlemler (Reindex)", expanded=False):
        col1, col2, col3 = st.columns(3)
        if col1.button("Incremental Index v2 (Hızlı)"):
            try:
                used_incremental = True
                try:
                    from src.index import build_index_v2_incremental as _index_v2_fn
                except Exception:
                    from src.index import build_index_v2 as _index_v2_fn

                    used_incremental = False

                summary = _index_v2_fn(
                    raw_dir=RAW_DIR,
                    processed_dir=PROCESSED_DIR,
                    collection_name=DEFAULT_COLLECTION_NAME_V2,
                    chunk_size=CONFIG.chunking.v2.chunk_size,
                    overlap=CONFIG.chunking.v2.overlap,
                )
                _clear_cached_retrievers()
                _append_ingest_log(
                    action="index_v2_incremental",
                    status="ok",
                    details={"mode": "incremental" if used_incremental else "full_v2_fallback", **summary},
                )
                if used_incremental:
                    st.success("Incremental index v2 tamamlandı.")
                else:
                    st.warning("Incremental import bulunamadi; full reindex v2 calistirildi.")
            except Exception as exc:
                _append_ingest_log(action="index_v2_incremental", status="error", details={"error": str(exc)})
                st.error(f"Incremental index v2 başarısız: {exc}")
        if col2.button("Full Reindex v2"):
            try:
                from src.index import build_index_v2

                summary = build_index_v2(
                    raw_dir=RAW_DIR,
                    processed_dir=PROCESSED_DIR,
                    collection_name=DEFAULT_COLLECTION_NAME_V2,
                    chunk_size=CONFIG.chunking.v2.chunk_size,
                    overlap=CONFIG.chunking.v2.overlap,
                )
                _clear_cached_retrievers()
                _append_ingest_log(action="reindex_v2", status="ok", details=summary)
                st.success("Reindex v2 tamamlandı.")
            except Exception as exc:
                _append_ingest_log(action="reindex_v2", status="error", details={"error": str(exc)})
                st.error(f"Reindex v2 başarısız: {exc}")
        if col3.button("Reindex v1"):
            try:
                from src.index import build_index

                summary = build_index(
                    raw_dir=RAW_DIR,
                    processed_dir=PROCESSED_DIR,
                    collection_name=DEFAULT_COLLECTION_NAME,
                    chunk_size=CONFIG.chunking.v1.chunk_size,
                    overlap=CONFIG.chunking.v1.overlap,
                )
                _clear_cached_retrievers()
                _append_ingest_log(action="reindex_v1", status="ok", details=summary)
                st.success("Reindex v1 tamamlandı.")
            except Exception as exc:
                _append_ingest_log(action="reindex_v1", status="error", details={"error": str(exc)})
                st.error(f"Reindex v1 başarısız: {exc}")

    with st.expander("Son İçe Alma Logları", expanded=False):
        rows = _load_recent_ingest_logs(limit=20)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Kayıt bulunamadı.")


def _render_ask_page() -> None:
    st.subheader("Soru Sor")
    st.caption("Sorunuzu yazın, kısa yanıtı alın. Kanıtlar varsayılan olarak kapalı gelir.")
    try:
        from src.demo import load_demo_questions

        demo_questions = load_demo_questions()
    except Exception:
        demo_questions = []
    if demo_questions:
        with st.container(border=True):
            st.caption("Demo Script (hazir sorular)")
            demo_cols = st.columns(3)
            for idx, item in enumerate(demo_questions[:5]):
                col = demo_cols[idx % 3]
                if col.button(item, key=f"ask_demo_question_{idx}"):
                    st.session_state["ask_product_question"] = item
                    st.rerun()

    companies = _available_companies()
    settings = _get_ui_settings()

    question = st.text_area(
        "Sorunuz",
        key="ask_product_question",
        height=90,
        placeholder="Orn: 2025 ucuncu ceyrek net kar kac?",
    )
    col1, col2 = st.columns([2, 1])
    selected_company = col1.selectbox(
        "Şirket",
        ["TUMU"] + companies,
        index=0,
        format_func=_company_display_name,
    )
    company_filter = None if selected_company == "TUMU" else selected_company
    retriever_name = str(settings.get("retriever", "v3"))
    with col2.expander("Advanced", expanded=False):
        retriever_name = st.selectbox("Retriever", ["v3", "v2", "v1", "v4", "v5", "v6"], index=["v3", "v2", "v1", "v4", "v5", "v6"].index(retriever_name))
        settings["retriever"] = retriever_name

    if st.button("Yanıtla", type="primary"):
        if not question.strip():
            st.warning("Lütfen bir soru girin.")
        else:
            parsed = parse_query(question)
            query_type = str(parsed.get("signals", {}).get("query_type"))
            cross_company_mode = is_cross_company_query(question, available_companies=companies)
            comparison_mode = _is_comparison_mode(question, query_type) and not cross_company_mode
            qa_payload: Dict[str, Any] = {
                "question": question,
                "company": company_filter,
                "retriever": retriever_name,
                "parsed": parsed,
                "mode": "standard",
                "found": False,
                "answer_text": "",
                "chunks": [],
                "comparison_result": None,
                "cross_company_result": None,
            }
            try:
                if cross_company_mode:
                    mentioned = detect_company_mentions(question, available_companies=companies)
                    selected = [c for c in ([company_filter] if company_filter else []) + mentioned if c]
                    if len(selected) < 2:
                        selected = companies[:3]
                    result = run_cross_company_comparison(
                        question=question,
                        retriever=_retriever_v3(),
                        companies=selected,
                        top_k_initial=int(settings.get("top_k_initial_v3", CONFIG.retrieval.v3_top_k_initial)),
                        top_k_final=int(settings.get("top_k_final", CONFIG.retrieval.top_k_final)),
                        alpha=float(settings.get("alpha_v3", CONFIG.retrieval.alpha_v3)),
                    )
                    qa_payload["mode"] = "cross_company"
                    qa_payload["cross_company_result"] = result
                    qa_payload["found"] = bool(result.get("found"))
                    qa_payload["answer_text"] = result.get("message", "Karsilastirma tamamlandi.")
                elif comparison_mode:
                    result = _run_comparison_pipeline(question=question, company=company_filter)
                    qa_payload["mode"] = "comparison"
                    qa_payload["comparison_result"] = result
                    qa_payload["found"] = bool(result.get("found"))
                    qa_payload["answer_text"] = "\n".join(_comparison_markdown_lines(result))
                    for quarter_chunks in (result.get("quarter_chunks", {}) or {}).values():
                        qa_payload["chunks"].extend(list(quarter_chunks))
                else:
                    chunks = _retrieve(question=question, retriever_name=retriever_name, company=company_filter)
                    answer_text = _answer_engine().answer(question=question, chunks=chunks)
                    qa_payload["mode"] = "standard"
                    qa_payload["found"] = _is_found_answer(answer_text)
                    qa_payload["answer_text"] = answer_text
                    qa_payload["chunks"] = chunks
                st.session_state.pop("ask_ai_commentary_payload", None)
                st.session_state.pop("ask_ai_commentary_signature", None)
                st.session_state["product_last_qa"] = qa_payload
            except Exception as exc:
                st.error(f"Sorgu çalıştırılamadı: {exc}")

    qa = st.session_state.get("product_last_qa")
    if not qa:
        st.info("Bir soru yazarak başlayın.")
        return

    with st.container(border=True):
        if qa["found"]:
            st.success("Yanıt hazır")
        else:
            st.warning("Bu soru icin dokumanda net bir kanit bulunamadi.")
            st.caption("Daha iyi sonuc almak icin asagidaki orneklerden birini deneyin:")
            suggestion_cols = st.columns(3)
            for idx, suggestion in enumerate(ASK_NOT_FOUND_SUGGESTIONS):
                col = suggestion_cols[idx % 3]
                if col.button(suggestion, key=f"ask_not_found_suggestion_{idx}"):
                    st.session_state["ask_product_question"] = suggestion
                    st.rerun()
        if qa["mode"] == "cross_company":
            result = qa.get("cross_company_result") or {}
            st.markdown(f"**{result.get('message', '-') }**")
            frame = result.get("frame")
            if frame is not None and not frame.empty:
                st.dataframe(frame, use_container_width=True, hide_index=True)
        elif qa["mode"] == "comparison":
            result = qa.get("comparison_result") or {}
            for line in _comparison_markdown_lines(result):
                st.markdown(line)
        else:
            question_text = str(qa.get("question", ""))
            chunks = qa.get("chunks", []) or []
            primary_answer = (
                _extract_primary_answer_with_fallback(
                    question=question_text,
                    chunks=chunks,
                    company=qa.get("company"),
                )
                if qa.get("found")
                else None
            )

            if primary_answer:
                source_chunk = primary_answer["source"]
                st.markdown("**Ana cevap**")
                label_text = html.escape(str(primary_answer["label"]))
                value_chip = _primary_value_chip_html(str(primary_answer["value"]))
                st.markdown(f"<h3 style='margin-top:0.2rem'>{label_text}: {value_chip}</h3>", unsafe_allow_html=True)
                st.caption(
                    "Kaynak: "
                    + _source_label(
                        source_chunk.doc_id,
                        source_chunk.quarter,
                        source_chunk.page,
                        source_chunk.section_title,
                    )
                )

            summary_items = [
                line
                for line in _summary_items_for_display(str(qa.get("answer_text", "")))[:6]
                if not _is_low_information_summary(line)
            ]
            if summary_items:
                if primary_answer:
                    with st.expander("Detayli aciklama", expanded=False):
                        for line in summary_items:
                            st.markdown(f"- {line}")
                else:
                    for line in summary_items:
                        st.markdown(f"- {line}")
            elif qa.get("found") and not primary_answer:
                st.info("Sayisal deger acik secilemedi. Asagidaki kanitlardan kontrol edebilirsiniz.")

    ask_commentary: Dict[str, Any] = {}
    if _llm_assistant_enabled():
        commentary_company: Optional[str] = None
        if qa.get("mode") == "cross_company":
            result = qa.get("cross_company_result") or {}
            commentary_company = str(result.get("best_company", "")).strip().upper() or None
        elif qa.get("company"):
            commentary_company = str(qa.get("company", "")).strip().upper() or None
        else:
            mentions = detect_company_mentions(str(qa.get("question", "")), available_companies=companies)
            if len(mentions) == 1:
                commentary_company = str(mentions[0]).strip().upper()

        selected_ai_model = str(settings.get("llm_model", "")).strip() or str(DEFAULT_UI_SETTINGS["llm_model"])
        answer_payload = _qa_to_commentary_answer_payload(qa, primary_answer=primary_answer if qa.get("mode") == "standard" else None)
        payload_company = str(answer_payload.get("company", "")).strip().upper() or None
        if payload_company:
            commentary_company = payload_company
        payload_quarter = str((answer_payload.get("parsed", {}) or {}).get("quarter") or "").strip() or None
        commentary_quarter = payload_quarter or str((qa.get("parsed", {}) or {}).get("quarter") or "").strip() or None
        evidence_signature = []
        for row in list(answer_payload.get("evidence") or [])[:3]:
            evidence_signature.append(
                {
                    "doc_id": str(row.get("doc_id", "")),
                    "page": row.get("page"),
                    "quarter": str(row.get("quarter", "")),
                    "company": str(row.get("company", "")),
                }
            )
        signature = json.dumps(
            {
                "question": str(qa.get("question", "")),
                "company": commentary_company,
                "quarter": commentary_quarter,
                "model": selected_ai_model,
                "mode": str(qa.get("mode", "standard")),
                "found": bool(qa.get("found")),
                "answer_text": str(qa.get("answer_text", ""))[:400],
                "evidence": evidence_signature,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if st.session_state.get("ask_ai_commentary_signature") != signature:
            st.session_state["ask_ai_commentary_signature"] = signature
            st.session_state.pop("ask_ai_commentary_payload", None)

        if qa.get("found"):
            model_col, ai_col, ai_note_col = st.columns([2, 1, 2])
            ask_current_model = str(settings.get("llm_model", "")).strip() or str(DEFAULT_UI_SETTINGS["llm_model"])
            ask_model_options = list(AI_MODEL_OPTIONS)
            if ask_current_model not in ask_model_options:
                ask_model_options.insert(0, ask_current_model)
            ask_model_idx = ask_model_options.index(ask_current_model) if ask_current_model in ask_model_options else 0
            selected_ai_model = model_col.selectbox(
                "AI Model",
                options=ask_model_options,
                index=ask_model_idx,
                key="ask_model_select",
            )
            settings["llm_model"] = selected_ai_model
            if ai_col.button("AI Assistanta Sor", key="ask_ai_assistant_button"):
                with st.spinner("AI yorum hazirlaniyor..."):
                    st.session_state["ask_ai_commentary_payload"] = _post_commentary_to_api(
                        question=str(qa.get("question", "")),
                        answer_payload=answer_payload,
                        company=commentary_company,
                        quarter=commentary_quarter,
                        model_override=selected_ai_model,
                    )
            ask_commentary = dict(st.session_state.get("ask_ai_commentary_payload") or {})
            if _commentary_has_content(ask_commentary):
                ai_note_col.caption("Bu sadece yorumdur; sayilar kanitlardan gelir.")
        else:
            st.session_state.pop("ask_ai_commentary_payload", None)

    _render_commentary_box(ask_commentary, title="Kisa yorum", expanded=False)

    with st.expander("Kaynaklar / Kanıtlar", expanded=False):
        if qa["mode"] == "cross_company":
            for row in (qa.get("cross_company_result", {}) or {}).get("evidence", [])[:10]:
                st.caption(
                    _source_label(
                        str(row.get("doc_id", "")),
                        str(row.get("quarter", "")),
                        row.get("page"),
                        str(row.get("section_title", "(no heading)")),
                    )
                )
                st.write(_short_excerpt(str(row.get("excerpt", ""))))
        elif qa["mode"] == "comparison":
            for row in (qa.get("comparison_result", {}) or {}).get("records", [])[:10]:
                st.caption(
                    _source_label(
                        str(row.get("doc_id", "")),
                        str(row.get("quarter", "")),
                        row.get("page"),
                        str(row.get("section_title", "(no heading)")),
                    )
                )
                st.write(_short_excerpt(str(row.get("excerpt", ""))))
        else:
            chunks = qa.get("chunks", [])
            for chunk in chunks[:10]:
                st.caption(_source_label(chunk.doc_id, chunk.quarter, chunk.page, chunk.section_title))
                st.write(_short_excerpt(chunk.text))

    export_text = _markdown_export(
        question=str(qa.get("question", "")),
        retriever_name=str(qa.get("retriever", "v3")),
        parsed=qa.get("parsed", {"signals": {}}),
        answer_text=str(qa.get("answer_text", "")),
        chunks=qa.get("chunks", []),
        comparison_result=qa.get("comparison_result"),
        mode=str(qa.get("mode", "standard")),
    )
    st.download_button(
        "Markdown indir",
        data=export_text,
        file_name=f"rag_fin_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )


def _render_kap_financials_page() -> None:
    def _kap_sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
        try:
            return int(row.get("year", 0)), int(row.get("period", 0))
        except Exception:
            quarter = str(row.get("quarter", "0Q0")).upper()
            match = re.search(r"(20\d{2})Q([1-4])", quarter)
            if match:
                return int(match.group(1)), int(match.group(2))
            return (0, 0)

    def _kap_period_label(row: Dict[str, Any]) -> str:
        try:
            year = int(row.get("year", 0))
            period = int(row.get("period", 0))
            if year > 0 and period in {1, 2, 3, 4}:
                return f"{year}/{period * 3}"
        except Exception:
            pass
        quarter = str(row.get("quarter", "")).upper()
        match = re.search(r"(20\d{2})Q([1-4])", quarter)
        if match:
            return f"{match.group(1)}/{int(match.group(2)) * 3}"
        return str(row.get("quarter", "-"))

    def _kap_period_sort_key(label: str) -> Tuple[int, int]:
        match = re.search(r"(20\d{2})/(\d{1,2})", str(label))
        if match:
            return int(match.group(1)), int(match.group(2))
        return (0, 0)

    def _kap_to_thousand(value: Any) -> Optional[float]:
        if value is None or _is_nan(value):
            return None
        try:
            return float(value) / 1_000.0
        except Exception:
            return None

    def _kap_format_thousand(value: Any) -> str:
        scaled = _kap_to_thousand(value)
        if scaled is None:
            return "-"
        return f"{scaled:,.0f}".replace(",", ".")

    def _kap_change_pct(current: Any, previous: Any) -> Optional[float]:
        cur = _kap_to_thousand(current)
        prev = _kap_to_thousand(previous)
        if cur is None or prev is None:
            return None
        if abs(prev) < 1e-9:
            return None
        return ((cur - prev) / abs(prev)) * 100.0

    def _kap_change_html(change_pct: Optional[float]) -> str:
        if change_pct is None or _is_nan(change_pct):
            return "<span style='color:#94a3b8'>-</span>"
        val = float(change_pct)
        if val > 0:
            color = "#22c55e"
        elif val < 0:
            color = "#ef4444"
        else:
            color = "#94a3b8"
        return f"<span style='color:{color};font-weight:700'>% {val:+.0f}</span>"

    def _kap_render_summary_table(
        *,
        title: str,
        unit_label: str,
        current_label: str,
        previous_label: str,
        rows: List[Tuple[str, Any, Any]],
    ) -> None:
        body_rows: List[str] = []
        for label, current_value, previous_value in rows:
            body_rows.append(
                "<tr>"
                f"<td style='padding:10px 12px;border-top:1px solid #1f2937;font-weight:600'>{html.escape(label)}</td>"
                f"<td style='padding:10px 12px;border-top:1px solid #1f2937;text-align:right'>{_kap_format_thousand(current_value)}</td>"
                f"<td style='padding:10px 12px;border-top:1px solid #1f2937;text-align:right'>{_kap_format_thousand(previous_value)}</td>"
                f"<td style='padding:10px 12px;border-top:1px solid #1f2937;text-align:right'>{_kap_change_html(_kap_change_pct(current_value, previous_value))}</td>"
                "</tr>"
            )
        table_html = (
            "<div style='border:1px solid #1f2937;border-radius:12px;overflow:hidden;background:#0b1220'>"
            f"<div style='padding:10px 12px;font-weight:700;border-bottom:1px solid #1f2937'>{html.escape(title)} <span style='font-weight:500;color:#94a3b8'>{html.escape(unit_label)}</span></div>"
            "<table style='width:100%;border-collapse:collapse'>"
            "<thead><tr>"
            "<th style='padding:10px 12px;text-align:left;color:#cbd5e1'>Kalem</th>"
            f"<th style='padding:10px 12px;text-align:right;color:#cbd5e1'>{html.escape(current_label)}</th>"
            f"<th style='padding:10px 12px;text-align:right;color:#cbd5e1'>{html.escape(previous_label)}</th>"
            "<th style='padding:10px 12px;text-align:right;color:#cbd5e1'>%</th>"
            "</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    def _kap_has_metric(rows: List[Dict[str, Any]], metric_key: str) -> bool:
        for row in rows:
            value = dict(row.get("metrics") or {}).get(metric_key)
            if value is None or _is_nan(value):
                value = dict(row.get("metrics_quarterly") or {}).get(metric_key)
            if value is None or _is_nan(value):
                value = dict(row.get("metrics_ytd") or {}).get(metric_key)
            if value is not None and not _is_nan(value):
                return True
        return False

    flow_metrics = {
        "satis_gelirleri",
        "brut_kar",
        "favok",
        "net_kar",
        "faaliyet_nakit_akisi",
        "capex",
        "serbest_nakit_akisi",
    }

    def _kap_resolve_metric_value(
        rows_sorted: List[Dict[str, Any]],
        idx: int,
        metric_key: str,
        *,
        as_quarterly_flow: bool,
    ) -> Optional[float]:
        row = rows_sorted[idx]
        metrics_point = dict(row.get("metrics") or {})
        metrics_q = dict(row.get("metrics_quarterly") or metrics_point)
        metrics_ytd = dict(row.get("metrics_ytd") or {})

        if as_quarterly_flow and metric_key in flow_metrics:
            ytd_value = metrics_ytd.get(metric_key)
            if ytd_value is not None and not _is_nan(ytd_value):
                prev_ytd: Optional[float] = None
                year = int(row.get("year", 0))
                period = int(row.get("period", 0))
                for prev_idx in range(idx - 1, -1, -1):
                    prev_row = rows_sorted[prev_idx]
                    if int(prev_row.get("year", 0)) != year:
                        break
                    prev_metrics_ytd = dict(prev_row.get("metrics_ytd") or {})
                    prev_candidate = prev_metrics_ytd.get(metric_key)
                    if prev_candidate is not None and not _is_nan(prev_candidate):
                        prev_ytd = float(prev_candidate)
                        break
                if period > 1 and prev_ytd is not None:
                    return float(ytd_value) - prev_ytd
                return float(ytd_value)
            q_value = metrics_q.get(metric_key)
            if q_value is not None and not _is_nan(q_value):
                return float(q_value)
            return None

        point_value = metrics_point.get(metric_key)
        if point_value is not None and not _is_nan(point_value):
            return float(point_value)
        q_value = metrics_q.get(metric_key)
        if q_value is not None and not _is_nan(q_value):
            return float(q_value)
        ytd_value = metrics_ytd.get(metric_key)
        if ytd_value is not None and not _is_nan(ytd_value):
            return float(ytd_value)
        return None

    def _kap_build_series_points(
        rows: List[Dict[str, Any]],
        value_builder: Any,
    ) -> List[Dict[str, Any]]:
        rows_sorted = sorted(rows, key=_kap_sort_key)
        period_to_value: Dict[str, float] = {}
        for idx, row in enumerate(rows_sorted):
            value = value_builder(rows_sorted, idx, row)
            if value is None or _is_nan(value):
                continue
            period_to_value[_kap_period_label(row)] = float(value)
        points = [{"period": period, "value": value} for period, value in period_to_value.items()]
        return sorted(points, key=lambda item: _kap_period_sort_key(str(item["period"])))[-5:]

    def _render_kap_quarterly_chart(
        slot: Any,
        *,
        rows: List[Dict[str, Any]],
        metric_key: str,
        title: str,
        currency: str,
    ) -> None:
        currency_code = str(currency or "TL").upper()
        if currency_code == "TL":
            scale_divisor = 1_000_000_000.0
            unit_short = "mlr"
            unit_label = "milyar"
        else:
            scale_divisor = 1_000_000.0
            unit_short = "mn"
            unit_label = "milyon"
        raw_points = _kap_build_series_points(
            rows,
            lambda rows_sorted, idx, _row: _kap_resolve_metric_value(
                rows_sorted,
                idx,
                metric_key,
                as_quarterly_flow=True,
            ),
        )
        points = [
            {"period": str(item["period"]), "value_scaled": float(item["value"]) / scale_divisor}
            for item in raw_points
        ]
        slot.markdown(f"**{title}**")
        slot.caption(f"Birim: {unit_label} {currency_code}")
        if not points:
            slot.info("Veri yok.")
            return
        try:
            from pandas import DataFrame

            chart_data = DataFrame(points)
        except Exception:
            chart_data = points
        if alt is None:
            try:
                slot.bar_chart(chart_data, x="period", y="value_scaled")
            except Exception:
                slot.write(points)
            return
        sort_order = [str(item["period"]) for item in points]
        chart = (
            alt.Chart(chart_data if not isinstance(chart_data, list) else alt.Data(values=points))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("period:N", title="", sort=sort_order),
                y=alt.Y("value_scaled:Q", title=""),
                color=alt.condition(
                    "datum.value_scaled < 0",
                    alt.value("#ef4444"),
                    alt.value("#76a9e2"),
                ),
                tooltip=[
                    alt.Tooltip("period:N", title="Donem"),
                    alt.Tooltip("value_scaled:Q", title=f"Deger ({unit_short} {currency_code})", format=",.2f"),
                ],
            )
            .properties(height=250)
        )
        slot.altair_chart(chart, use_container_width=True)

    def _render_kap_ratio_line_chart(
        slot: Any,
        *,
        points: List[Dict[str, Any]],
        title: str,
        value_label: str,
    ) -> None:
        slot.markdown(f"**{title}**")
        if not points:
            slot.info("Veri yok.")
            return
        try:
            from pandas import DataFrame

            chart_data = DataFrame(points)
        except Exception:
            chart_data = points
        if alt is None:
            try:
                slot.line_chart(chart_data, x="period", y="value")
            except Exception:
                slot.write(points)
            return
        sort_order = [str(item["period"]) for item in points]
        chart = (
            alt.Chart(chart_data if not isinstance(chart_data, list) else alt.Data(values=points))
            .mark_line(color="#f97316", point=True)
            .encode(
                x=alt.X("period:N", title="", sort=sort_order),
                y=alt.Y("value:Q", title=""),
                tooltip=[
                    alt.Tooltip("period:N", title="Donem"),
                    alt.Tooltip("value:Q", title=value_label, format=",.2f"),
                ],
            )
            .properties(height=250)
        )
        slot.altair_chart(chart, use_container_width=True)

    def _kap_margin_points(
        rows: List[Dict[str, Any]],
        *,
        numerator_key: str,
    ) -> List[Dict[str, Any]]:
        raw = _kap_build_series_points(
            rows,
            lambda rows_sorted, idx, _row: (
                (num / den) * 100.0
                if (num := _kap_resolve_metric_value(rows_sorted, idx, numerator_key, as_quarterly_flow=True)) is not None
                and (den := _kap_resolve_metric_value(rows_sorted, idx, "satis_gelirleri", as_quarterly_flow=True)) is not None
                and abs(float(den)) > 1e-9
                else None
            ),
        )
        return [{"period": str(item["period"]), "value": float(item["value"])} for item in raw]

    def _kap_cari_oran_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raw = _kap_build_series_points(
            rows,
            lambda rows_sorted, idx, _row: (
                (assets / liabilities)
                if (assets := _kap_resolve_metric_value(rows_sorted, idx, "donen_varliklar", as_quarterly_flow=False)) is not None
                and (liabilities := _kap_resolve_metric_value(rows_sorted, idx, "kisa_vadeli_yukumlulukler", as_quarterly_flow=False)) is not None
                and abs(float(liabilities)) > 1e-9
                else None
            ),
        )
        return [{"period": str(item["period"]), "value": float(item["value"])} for item in raw]

    def _kap_roe_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raw = _kap_build_series_points(
            rows,
            lambda rows_sorted, idx, _row: (
                (net_profit / equity) * 100.0
                if (net_profit := _kap_resolve_metric_value(rows_sorted, idx, "net_kar", as_quarterly_flow=True)) is not None
                and (equity := _kap_resolve_metric_value(rows_sorted, idx, "ozkaynaklar", as_quarterly_flow=False)) is not None
                and abs(float(equity)) > 1e-9
                else None
            ),
        )
        return [{"period": str(item["period"]), "value": float(item["value"])} for item in raw]

    def _kap_to_commentary_answer_payload(
        *,
        rows_sorted: List[Dict[str, Any]],
        company_code: str,
        latest_row: Dict[str, Any],
        previous_quarter_row: Optional[Dict[str, Any]],
        previous_year_row: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not rows_sorted:
            return {"found": False}

        latest_period_label = _kap_period_label(latest_row)
        income_like_keys = {
            "satis_gelirleri",
            "brut_kar",
            "favok",
            "net_kar",
            "faaliyet_nakit_akisi",
            "capex",
            "serbest_nakit_akisi",
            "faiz_gelirleri",
            "faiz_giderleri",
            "net_ucret_komisyon_gelirleri",
            "net_faaliyet_kari",
        }
        metric_keys = [
            "net_kar",
            "satis_gelirleri",
            "brut_kar",
            "favok",
            "faaliyet_nakit_akisi",
            "capex",
            "serbest_nakit_akisi",
            "ozkaynaklar",
            "toplam_varliklar",
            "finansal_borclar",
            "net_borc",
            "faiz_gelirleri",
            "faiz_giderleri",
            "net_ucret_komisyon_gelirleri",
            "net_faaliyet_kari",
            "finansal_varliklar_net",
            "krediler",
            "mevduatlar",
            "beklenen_zarar_karsiliklari",
        ]

        def _resolve_metric_for_row(row: Optional[Dict[str, Any]], metric_key: str) -> Optional[float]:
            if not row:
                return None
            metrics_point = dict(row.get("metrics") or {})
            metrics_ytd = dict(row.get("metrics_ytd") or {})
            raw: Any
            if metric_key in income_like_keys:
                raw = metrics_ytd.get(metric_key)
                if raw is None or _is_nan(raw):
                    raw = metrics_point.get(metric_key)
            else:
                raw = metrics_point.get(metric_key)
                if raw is None or _is_nan(raw):
                    raw = metrics_ytd.get(metric_key)
            if raw is None or _is_nan(raw):
                return None
            try:
                return float(raw)
            except Exception:
                return None

        metrics: Dict[str, Any] = {}
        deltas: Dict[str, Any] = {}
        for metric_key in metric_keys:
            cur_val = _resolve_metric_for_row(latest_row, metric_key)
            if cur_val is None:
                continue
            metrics[metric_key] = cur_val
            prev_q_val = _resolve_metric_for_row(previous_quarter_row, metric_key)
            if prev_q_val is not None:
                deltas[f"{metric_key}_qoq"] = cur_val - prev_q_val
            prev_y_val = _resolve_metric_for_row(previous_year_row, metric_key)
            if prev_y_val is not None:
                deltas[f"{metric_key}_yoy"] = cur_val - prev_y_val

        ratio_builders = {
            "brut_kar_marji": lambda: _kap_margin_points(rows_sorted, numerator_key="brut_kar"),
            "favok_marji": lambda: _kap_margin_points(rows_sorted, numerator_key="favok"),
            "net_kar_marji": lambda: _kap_margin_points(rows_sorted, numerator_key="net_kar"),
            "cari_oran": lambda: _kap_cari_oran_points(rows_sorted),
            "ozkaynak_karliligi": lambda: _kap_roe_points(rows_sorted),
        }
        ratios: Dict[str, Any] = {}
        for ratio_key, build_points in ratio_builders.items():
            points = list(build_points() or [])
            if not points:
                continue
            points_sorted = sorted(points, key=lambda item: _kap_period_sort_key(str(item.get("period", ""))))
            values_by_period: Dict[str, float] = {}
            for item in points_sorted:
                period_label = str(item.get("period", "")).strip()
                value = item.get("value")
                if not period_label or value is None or _is_nan(value):
                    continue
                try:
                    values_by_period[period_label] = float(value)
                except Exception:
                    continue
            if not values_by_period:
                continue
            ratio_latest_period = str(points_sorted[-1].get("period", "")).strip()
            if not ratio_latest_period:
                continue
            ratio_latest_value = values_by_period.get(ratio_latest_period)
            if ratio_latest_value is None:
                continue
            ratios[ratio_key] = ratio_latest_value

            if len(points_sorted) >= 2:
                prev_period = str(points_sorted[-2].get("period", "")).strip()
                prev_value = values_by_period.get(prev_period)
                if prev_value is not None:
                    deltas[f"{ratio_key}_qoq"] = ratio_latest_value - prev_value

            ratio_match = re.search(r"(20\d{2})/(\d{1,2})", ratio_latest_period)
            if ratio_match:
                yoy_period = f"{int(ratio_match.group(1)) - 1}/{int(ratio_match.group(2))}"
                yoy_value = values_by_period.get(yoy_period)
                if yoy_value is not None:
                    deltas[f"{ratio_key}_yoy"] = ratio_latest_value - yoy_value

        # --- QoQ / YoY direction signals (separate) ---
        qoq_signals: Dict[str, str] = {}
        yoy_signals: Dict[str, str] = {}
        for delta_key, delta_val in deltas.items():
            if delta_key.endswith("_qoq"):
                base = delta_key[: -len("_qoq")]
                qoq_signals[base] = "artis" if delta_val > 0 else ("azalis" if delta_val < 0 else "yatay")
            elif delta_key.endswith("_yoy"):
                base = delta_key[: -len("_yoy")]
                yoy_signals[base] = "artis" if delta_val > 0 else ("azalis" if delta_val < 0 else "yatay")

        # --- QoQ / YoY change percentages ---
        qoq_pct: Dict[str, float] = {}
        yoy_pct: Dict[str, float] = {}
        for metric_key in metric_keys:
            cur_val = metrics.get(metric_key)
            if cur_val is None:
                continue
            prev_q_val = _resolve_metric_for_row(previous_quarter_row, metric_key)
            if prev_q_val is not None and abs(float(prev_q_val)) > 1e-9:
                qoq_pct[metric_key] = round(((float(cur_val) - float(prev_q_val)) / abs(float(prev_q_val))) * 100.0, 1)
            prev_y_val = _resolve_metric_for_row(previous_year_row, metric_key)
            if prev_y_val is not None and abs(float(prev_y_val)) > 1e-9:
                yoy_pct[metric_key] = round(((float(cur_val) - float(prev_y_val)) / abs(float(prev_y_val))) * 100.0, 1)

        # --- FCF (Serbest Nakit Akisi) special signal ---
        fcf_signal: Dict[str, Any] = {}
        fcf_val = metrics.get("serbest_nakit_akisi")
        if fcf_val is not None:
            fcf_signal["sign"] = "pozitif" if float(fcf_val) >= 0 else "negatif"
            # Annual total context – sum all quarters of the same fiscal year
            latest_year = int(latest_row.get("year", 0))
            latest_period = int(latest_row.get("period", 0))
            fcf_annual_sum: float = 0.0
            fcf_annual_count: int = 0
            for row in rows_sorted:
                if int(row.get("year", 0)) == latest_year:
                    row_fcf = _resolve_metric_for_row(row, "serbest_nakit_akisi")
                    if row_fcf is not None:
                        fcf_annual_sum += float(row_fcf)
                        fcf_annual_count += 1
            if fcf_annual_count >= 2:
                fcf_signal["annual_sum"] = round(fcf_annual_sum, 0)
                fcf_signal["annual_sign"] = "pozitif" if fcf_annual_sum >= 0 else "negatif"
                fcf_signal["quarters_in_sum"] = fcf_annual_count
            if latest_period == 4:
                fcf_signal["is_year_end"] = True
            # Operating CF and capex split for context
            op_cf = metrics.get("faaliyet_nakit_akisi")
            capex = metrics.get("capex")
            if op_cf is not None:
                fcf_signal["operating_cf_sign"] = "pozitif" if float(op_cf) >= 0 else "negatif"
            if capex is not None:
                fcf_signal["capex_sign"] = "pozitif" if float(capex) >= 0 else "negatif"

        evidence_rows: List[Dict[str, Any]] = []
        focus_metrics = [
            ("net_kar", "Net Donem Kari"),
            ("satis_gelirleri", "Satislar"),
            ("favok", "FAVOK"),
            ("serbest_nakit_akisi", "Serbest Nakit Akisi"),
            ("faiz_gelirleri", "Faiz Gelirleri"),
            ("ozkaynaklar", "Ozkaynaklar"),
        ]
        for row in rows_sorted[-3:]:
            period_label = _kap_period_label(row)
            snippets: List[str] = []
            for metric_key, metric_label in focus_metrics:
                value = _resolve_metric_for_row(row, metric_key)
                if value is None:
                    continue
                snippets.append(f"{metric_label}: {value:,.0f}")
                if len(snippets) >= 3:
                    break
            if snippets:
                evidence_rows.append({"excerpt": f"{period_label} | " + " | ".join(snippets)})
            if len(evidence_rows) >= 5:
                break

        # Compose previous-period labels for prompt context
        prev_q_label = _kap_period_label(previous_quarter_row) if previous_quarter_row else None
        prev_y_label = _kap_period_label(previous_year_row) if previous_year_row else None

        return {
            "found": bool(metrics or ratios),
            "commentary_mode": "kap",
            "company": company_code,
            "parsed": {"quarter": latest_period_label},
            "metrics": metrics,
            "ratios": ratios,
            "deltas": deltas,
            "qoq_signals": qoq_signals,
            "yoy_signals": yoy_signals,
            "qoq_pct": qoq_pct,
            "yoy_pct": yoy_pct,
            "fcf_signal": fcf_signal,
            "prev_quarter_label": prev_q_label,
            "prev_year_label": prev_y_label,
            "confidence_map": {},
            "verify_map": {},
            "evidence": evidence_rows,
            "answer": {"found": bool(metrics or ratios), "verify_status": "PASS", "bullets": ["KAP ceyrek ozeti hazir."]},
        }

    st.subheader("KAP Finansallari")
    st.caption("Resmi KAP finansal bildirimleri. Bu panel, extractor akisini degistirmez; ayri kaynak olarak gosterir.")

    kap_cfg = getattr(CONFIG, "kap", None)
    if kap_cfg is None:
        st.info("KAP konfigurasyonu bulunamadi.")
        return
    if not bool(kap_cfg.enabled):
        st.info("KAP paneli kapali (`kap.enabled=false`). Ayarlardan acabilirsiniz.")
        return

    companies = _available_companies()
    # KAP sekmesi icin, rapor yuklenmemis olsa da secilebilecek bir BIST-30 cekirdek listesi sun.
    # Tekilleştirme görüntü etiketine göre yapılır (örn. BIM + BIMAS => tek BIM).
    merged_candidates: List[str] = list(companies) + list(KAP_DEFAULT_BIST30_COMPANIES)
    display_to_raw: Dict[str, str] = {}
    for candidate in merged_candidates:
        raw_norm = str(candidate or "").strip().upper()
        if not raw_norm:
            continue
        display = _company_display_name(raw_norm)
        if not display or display == "-":
            continue
        current = display_to_raw.get(display)
        if current is None:
            display_to_raw[display] = raw_norm
            continue
        current_norm = str(current).strip().upper()
        # Prefer exact canonical code first, otherwise keep the shorter alias.
        if raw_norm == display and current_norm != display:
            display_to_raw[display] = raw_norm
        elif current_norm != display and len(raw_norm) < len(current_norm):
            display_to_raw[display] = raw_norm
    merged_companies = [display_to_raw[key] for key in sorted(display_to_raw.keys())]

    selected_company = st.selectbox(
        "Sirket",
        options=merged_companies,
        key="kap_company_selector",
        format_func=_company_display_name,
    )
    top_left, top_right = st.columns([1, 4])
    force_refresh = top_left.button("KAP Verisini Yenile", key="kap_refresh")
    if force_refresh:
        top_right.caption("KAP cache yenileniyor...")

    with st.spinner("KAP verisi getiriliyor..."):
        snapshot = fetch_kap_company_snapshot(
            company=selected_company,
            cfg=kap_cfg,
            processed_dir=CONFIG.paths.processed_dir,
            force_refresh=force_refresh,
            max_quarters=10,
        )

    if not snapshot.get("ok"):
        st.warning(f"KAP verisi alinamadi: {snapshot.get('error', 'bilinmeyen hata')}")
        return

    if snapshot.get("cache_stale"):
        st.warning(
            "Canli KAP cagrisi hata verdi; son cache'lenmis veri gosteriliyor. "
            "Detay: " + str(snapshot.get("error", "bilinmeyen hata"))
        )
    elif snapshot.get("cache_hit"):
        st.caption("Kaynak: KAP (cache).")
    else:
        st.caption("Kaynak: KAP (canli).")

    quarter_rows = sorted(list(snapshot.get("quarters") or []), key=_kap_sort_key)
    # Revised disclosures may duplicate the same quarter. Keep the latest entry per period.
    period_dedup: Dict[str, Dict[str, Any]] = {}
    for row in quarter_rows:
        period_dedup[_kap_period_label(row)] = row
    quarter_rows = sorted(list(period_dedup.values()), key=_kap_sort_key)
    if not quarter_rows:
        st.info("KAP tarafinda son ceyrek verisi bulunamadi.")
        return

    latest = quarter_rows[-1]
    previous = quarter_rows[-2] if len(quarter_rows) >= 2 else {}

    def _kap_find_prev_year_same_period(
        rows: List[Dict[str, Any]],
        reference_row: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            ref_year = int(reference_row.get("year", 0))
            ref_period = int(reference_row.get("period", 0))
        except Exception:
            return None
        target_year = ref_year - 1
        for item in reversed(rows):
            try:
                year = int(item.get("year", 0))
                period = int(item.get("period", 0))
            except Exception:
                continue
            if year == target_year and period == ref_period:
                return item
        return None

    previous_income = _kap_find_prev_year_same_period(quarter_rows, latest) or previous
    previous_balance = previous

    latest_label = _kap_period_label(latest)
    previous_income_label = _kap_period_label(previous_income) if previous_income else "-"
    previous_balance_label = _kap_period_label(previous_balance) if previous_balance else "-"
    display_currency = str(latest.get("currency", "TL")).upper()

    header_col1, header_col2, header_col3 = st.columns([5, 1, 1])
    company_title = str(snapshot.get("company_title") or selected_company)
    header_col1.markdown(f"### {company_title} Ozet Finansallar")
    header_col2.markdown(f"`{display_currency}`")
    source_url = str(snapshot.get("source_url", "")).strip()
    if source_url:
        header_col3.markdown(f"[Piyasalar]({source_url})")

    left_table_rows: List[Tuple[str, Any, Any]] = []
    right_table_rows: List[Tuple[str, Any, Any]] = []
    latest_income_metrics = dict(latest.get("metrics_ytd") or latest.get("metrics") or {})
    previous_income_metrics = (
        dict(previous_income.get("metrics_ytd") or previous_income.get("metrics") or {}) if previous_income else {}
    )
    latest_balance_metrics = dict(latest.get("metrics") or latest.get("metrics_ytd") or {})
    previous_balance_metrics = (
        dict(previous_balance.get("metrics") or previous_balance.get("metrics_ytd") or {}) if previous_balance else {}
    )

    def _has_value(metrics: Dict[str, Any], metric_key: str) -> bool:
        value = metrics.get(metric_key)
        return value is not None and not _is_nan(value)

    stock_code = str(snapshot.get("stock_code") or selected_company or "").strip().upper()
    company_title_norm = str(snapshot.get("company_title") or "").upper()
    is_bank_by_identity = stock_code in KAP_BANK_TICKERS or "BANK" in company_title_norm

    has_faiz_gelirleri = _has_value(latest_income_metrics, "faiz_gelirleri")
    has_faiz_giderleri = _has_value(latest_income_metrics, "faiz_giderleri")
    has_net_ucret = _has_value(latest_income_metrics, "net_ucret_komisyon_gelirleri")
    has_net_faaliyet = _has_value(latest_income_metrics, "net_faaliyet_kari")
    has_mevduat = _has_value(latest_balance_metrics, "mevduatlar")
    has_fin_varliklar = _has_value(latest_balance_metrics, "finansal_varliklar_net")
    has_beklenen_zarar = _has_value(latest_balance_metrics, "beklenen_zarar_karsiliklari")

    # Heuristic is intentionally strict to avoid non-bank false positives
    # from generic labels such as "faiz giderleri" or "krediler".
    has_bank_income_core = has_faiz_gelirleri and (has_faiz_giderleri or has_net_ucret or has_net_faaliyet)
    has_bank_balance_core = has_mevduat or has_fin_varliklar or has_beklenen_zarar
    is_bank_like = is_bank_by_identity or (has_bank_income_core and has_bank_balance_core)

    income_row_defs: List[Tuple[str, str]]
    balance_row_defs: List[Tuple[str, str]]
    if is_bank_like:
        income_row_defs = [
            ("Faiz Gelirleri", "faiz_gelirleri"),
            ("Faiz Giderleri (-)", "faiz_giderleri"),
            ("Net Ucret Komisyon Gelirleri", "net_ucret_komisyon_gelirleri"),
            ("Net Faaliyet Kari (Zarari)", "net_faaliyet_kari"),
            ("Net Donem Kari", "net_kar"),
        ]
        balance_row_defs = [
            ("Finansal Varliklar (Net)", "finansal_varliklar_net"),
            ("Krediler", "krediler"),
            ("Mevduatlar", "mevduatlar"),
            ("Beklenen Zarar Karsiliklari", "beklenen_zarar_karsiliklari"),
            ("Ozkaynaklar", "ozkaynaklar"),
        ]
    else:
        income_row_defs = [
            ("Satislar", "satis_gelirleri"),
            ("Brut Kar", "brut_kar"),
            ("FAVOK", "favok"),
            ("Net Donem Kari", "net_kar"),
            ("Faaliyet Nakit Akisi", "faaliyet_nakit_akisi"),
            ("Serbest Nakit Akisi", "serbest_nakit_akisi"),
        ]
        balance_row_defs = [
            ("Donen Varliklar", "donen_varliklar"),
            ("Duran Varliklar", "duran_varliklar"),
            ("Toplam Varliklar", "toplam_varliklar"),
            ("Finansal Borclar", "finansal_borclar"),
            ("Net Borc", "net_borc"),
            ("Ozkaynaklar", "ozkaynaklar"),
        ]

    for label, metric_key in income_row_defs:
        left_table_rows.append(
            (
                label,
                latest_income_metrics.get(metric_key),
                previous_income_metrics.get(metric_key),
            )
        )

    for label, metric_key in balance_row_defs:
        right_table_rows.append(
            (
                label,
                latest_balance_metrics.get(metric_key),
                previous_balance_metrics.get(metric_key),
            )
        )

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        _kap_render_summary_table(
            title="Ozet Gelir Tablosu",
            unit_label=f"Bin {display_currency}",
            current_label=latest_label,
            previous_label=previous_income_label,
            rows=left_table_rows,
        )
    with table_col2:
        _kap_render_summary_table(
            title="Ozet Bilanco",
            unit_label=f"Bin {display_currency}",
            current_label=latest_label,
            previous_label=previous_balance_label,
            rows=right_table_rows,
        )

    st.markdown("**Ceyreklik Grafikler**")
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    chart1_metric = "satis_gelirleri"
    chart1_title = "Ceyreklik Satislar"
    if is_bank_like and _kap_has_metric(quarter_rows, "faiz_gelirleri"):
        chart1_metric = "faiz_gelirleri"
        chart1_title = "Ceyreklik Faiz Gelirleri"
    _render_kap_quarterly_chart(
        chart_col1,
        rows=quarter_rows,
        metric_key=chart1_metric,
        title=chart1_title,
        currency=display_currency,
    )
    if is_bank_like and _kap_has_metric(quarter_rows, "net_faaliyet_kari"):
        chart2_metric = "net_faaliyet_kari"
        chart2_title = "Ceyreklik Net Faaliyet Kari"
    elif is_bank_like and _kap_has_metric(quarter_rows, "net_ucret_komisyon_gelirleri"):
        chart2_metric = "net_ucret_komisyon_gelirleri"
        chart2_title = "Ceyreklik Net Ucret Komisyon Geliri"
    else:
        chart2_metric = "favok" if _kap_has_metric(quarter_rows, "favok") else "brut_kar"
        chart2_title = "Ceyreklik FAVOK" if chart2_metric == "favok" else "Ceyreklik Brut Kar"
    _render_kap_quarterly_chart(
        chart_col2,
        rows=quarter_rows,
        metric_key=chart2_metric,
        title=chart2_title,
        currency=display_currency,
    )
    _render_kap_quarterly_chart(
        chart_col3,
        rows=quarter_rows,
        metric_key="net_kar",
        title="Ceyreklik Net Kar",
        currency=display_currency,
    )

    if _kap_has_metric(quarter_rows, "serbest_nakit_akisi"):
        free_cf_col, _, _ = st.columns(3)
        _render_kap_quarterly_chart(
            free_cf_col,
            rows=quarter_rows,
            metric_key="serbest_nakit_akisi",
            title="Ceyreklik Serbest Nakit Akisi",
            currency=display_currency,
        )

    st.markdown("**Oran Grafikler (Cizgi)**")
    ratio_row1_col1, ratio_row1_col2, ratio_row1_col3 = st.columns(3)
    ratio_row2_col1, ratio_row2_col2, ratio_row2_col3 = st.columns(3)

    _render_kap_ratio_line_chart(
        ratio_row1_col1,
        points=_kap_margin_points(quarter_rows, numerator_key="brut_kar"),
        title="Brut Kar Marji (Ceyreklik)",
        value_label="Deger (%)",
    )
    _render_kap_ratio_line_chart(
        ratio_row1_col2,
        points=_kap_margin_points(quarter_rows, numerator_key="favok"),
        title="FAVOK/FVAOK Marji (Ceyreklik)",
        value_label="Deger (%)",
    )
    _render_kap_ratio_line_chart(
        ratio_row1_col3,
        points=_kap_margin_points(quarter_rows, numerator_key="net_kar"),
        title="Net Kar Marji (Ceyreklik)",
        value_label="Deger (%)",
    )
    _render_kap_ratio_line_chart(
        ratio_row2_col1,
        points=_kap_cari_oran_points(quarter_rows),
        title="Cari Oran",
        value_label="Deger (x)",
    )
    _render_kap_ratio_line_chart(
        ratio_row2_col2,
        points=_kap_roe_points(quarter_rows),
        title="Ozkaynak Karliligi (ROE, Ceyreklik)",
        value_label="Deger (%)",
    )
    ratio_row2_col3.empty()

    if _llm_assistant_enabled():
        settings = _get_ui_settings()
        model_col, btn_col = st.columns([2, 1])
        current_model = str(settings.get("llm_model", "")).strip() or str(DEFAULT_UI_SETTINGS["llm_model"])
        model_options = list(AI_MODEL_OPTIONS)
        if current_model not in model_options:
            model_options.insert(0, current_model)
        model_idx = model_options.index(current_model) if current_model in model_options else 0
        selected_ai_model = model_col.selectbox(
            "AI Model",
            options=model_options,
            index=model_idx,
            key=f"kap_model_select::{selected_company}",
        )
        settings["llm_model"] = selected_ai_model
        commentary_key = f"kap_ai_commentary::{selected_company}::{latest_label}::{selected_ai_model}"
        if btn_col.button("AI Assistanta Sor", key=f"kap_ai_button::{selected_company}"):
            previous_year_for_commentary = _kap_find_prev_year_same_period(quarter_rows, latest)
            commentary_payload = _kap_to_commentary_answer_payload(
                rows_sorted=quarter_rows,
                company_code=str(snapshot.get("stock_code") or selected_company),
                latest_row=latest,
                previous_quarter_row=previous if previous else None,
                previous_year_row=previous_year_for_commentary,
            )
            with st.spinner("AI yorum hazirlaniyor..."):
                st.session_state[commentary_key] = _post_commentary_to_api(
                    question=f"{_company_display_name(selected_company)} {latest_label} KAP finansallarinda neler degisti?",
                    answer_payload=commentary_payload,
                    company=str(snapshot.get("stock_code") or selected_company),
                    year=str(latest.get("year", "") or ""),
                    quarter=latest_label,
                    model_override=selected_ai_model,
                )
        kap_commentary = dict(st.session_state.get(commentary_key) or {})
        _render_commentary_box(kap_commentary, title="Kisa yorum", expanded=True)
        if _commentary_has_content(kap_commentary):
            st.caption("Bu sadece yorumdur; sayilar KAP verilerinden gelir.")

    fetched_at = str(snapshot.get("fetched_at", "")).replace("T", " ").replace("+00:00", " UTC")
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Ticker", str(snapshot.get("stock_code") or "-"))
    meta_col2.metric("KAP Guncellenme", fetched_at or "-")
    meta_col3.metric("Disclosures", len(quarter_rows))

    with st.expander("Resmi Finansal Bildirimler (Son Ceyrekler)", expanded=False):
        disclosure_rows: List[Dict[str, Any]] = []
        for row in sorted(quarter_rows, key=_kap_sort_key, reverse=True):
            disclosure_rows.append(
                {
                    "quarter": row.get("quarter", "-"),
                    "publish_date": row.get("publish_date", "-"),
                    "disclosure_index": row.get("disclosure_index", "-"),
                    "title": row.get("title", "-"),
                    "pdf_url": row.get("pdf_url", "-"),
                }
            )
        st.dataframe(disclosure_rows, use_container_width=True, hide_index=True)

    st.caption(
        "Not: Bu sekme KAP'tan gelen resmi finansal tablo verilerini ayri gostermek icindir. "
        "Overview ekranindaki mevcut extractor akisi degistirilmemistir."
    )
    with st.expander("KAP Ham Detay (Debug)", expanded=False):
        st.json(snapshot)


def _render_settings_page() -> None:
    st.subheader("Ayarlar (Advanced)")
    settings = _get_ui_settings()
    col1, col2, col3 = st.columns(3)
    col1.selectbox("Varsayılan retriever", ["v3", "v2", "v1", "v4", "v5", "v6"], key="settings_retriever")
    col2.number_input("top_k_final", min_value=1, max_value=30, key="settings_top_k_final")
    col3.checkbox("Debug candidates", key="settings_debug_candidates")

    settings["retriever"] = st.session_state.get("settings_retriever", settings["retriever"])
    settings["top_k_final"] = int(st.session_state.get("settings_top_k_final", settings["top_k_final"]))
    settings["show_debug_candidates"] = bool(st.session_state.get("settings_debug_candidates", settings["show_debug_candidates"]))

    with st.expander("Retrieval parametreleri", expanded=False):
        settings["top_k_initial_v2"] = st.number_input("v2 top_k_initial", min_value=1, max_value=100, value=int(settings["top_k_initial_v2"]))
        settings["top_k_initial_v3"] = st.number_input("v3 top_k_initial", min_value=1, max_value=100, value=int(settings["top_k_initial_v3"]))
        settings["top_k_vector_v5"] = st.number_input("v5 top_k_vector", min_value=1, max_value=100, value=int(settings["top_k_vector_v5"]))
        settings["top_k_bm25_v5"] = st.number_input("v5 top_k_bm25", min_value=1, max_value=100, value=int(settings["top_k_bm25_v5"]))
        settings["top_k_candidates_v6"] = st.number_input("v6 cross_top_n", min_value=1, max_value=100, value=int(settings["top_k_candidates_v6"]))
        settings["alpha_v2"] = st.slider("alpha_v2", 0.0, 1.0, float(settings["alpha_v2"]), 0.01)
        settings["alpha_v3"] = st.slider("alpha_v3", 0.0, 1.0, float(settings["alpha_v3"]), 0.01)
        settings["beta_v5"] = st.slider("beta_v5", 0.0, 2.0, float(settings["beta_v5"]), 0.01)

    with st.expander("AI Assistant Modeli", expanded=False):
        current_model = str(settings.get("llm_model", "")).strip() or str(DEFAULT_UI_SETTINGS["llm_model"])
        model_options = list(AI_MODEL_OPTIONS) + [AI_MODEL_CUSTOM_SENTINEL]
        selected_option = current_model if current_model in AI_MODEL_OPTIONS else AI_MODEL_CUSTOM_SENTINEL
        model_choice = st.selectbox(
            "Yorum modeli",
            model_options,
            index=model_options.index(selected_option),
            key="settings_llm_model_choice",
        )
        if model_choice == AI_MODEL_CUSTOM_SENTINEL:
            custom_model = st.text_input(
                "Custom model id",
                value=str(settings.get("llm_model_custom", current_model)),
                key="settings_llm_model_custom",
                placeholder="orn: openai/gpt-oss-120b:free",
            ).strip()
            settings["llm_model_custom"] = custom_model
            if custom_model:
                settings["llm_model"] = custom_model
        else:
            settings["llm_model"] = model_choice
            settings["llm_model_custom"] = ""
        st.caption(
            "Secilen model sadece AI yorum katmaninda kullanilir. "
            "Deterministik extraction/retrieval akisi degismez."
        )

    with st.expander("Eval / Benchmark Araçları", expanded=False):
        c1, c2, c3 = st.columns(3)
        if c1.button("metrics_report çalıştır"):
            from src.metrics import run_metrics_report

            with st.spinner("metrics_report çalışıyor..."):
                report = run_metrics_report(
                    gold_file=CONFIG.evaluation.gold_file,
                    multi_company_gold_file=CONFIG.evaluation.gold_multicompany_file,
                    detailed_output=CONFIG.evaluation.detailed_output,
                    summary_output=CONFIG.evaluation.summary_output,
                    week6_summary_output=CONFIG.evaluation.week6_summary_output,
                    top_k=int(settings["top_k_final"]),
                    top_k_initial_v2=int(settings["top_k_initial_v2"]),
                    top_k_initial_v3=int(settings["top_k_initial_v3"]),
                    top_k_initial_v5_vector=int(settings["top_k_vector_v5"]),
                    top_k_initial_v5_bm25=int(settings["top_k_bm25_v5"]),
                    top_k_candidates_v6=int(settings["top_k_candidates_v6"]),
                    alpha_v2=float(settings["alpha_v2"]),
                    alpha_v3=float(settings["alpha_v3"]),
                    beta_v5=float(settings["beta_v5"]),
                )
            st.session_state["metrics_summary"] = report["summary"]
            st.success("metrics_report tamamlandı.")
        if c2.button("latency_bench çalıştır"):
            from src.latency_benchmark import run_latency_benchmark

            with st.spinner("latency_bench çalışıyor..."):
                report = run_latency_benchmark(
                    gold_file=CONFIG.evaluation.gold_file,
                    output_file=CONFIG.evaluation.latency_output,
                    sample_size=CONFIG.evaluation.latency_sample_size,
                    top_k=int(settings["top_k_final"]),
                    top_k_initial_v3=int(settings["top_k_initial_v3"]),
                    top_k_initial_v5_vector=int(settings["top_k_vector_v5"]),
                    top_k_initial_v5_bm25=int(settings["top_k_bm25_v5"]),
                    top_k_candidates_v6=int(settings["top_k_candidates_v6"]),
                    alpha_v3=float(settings["alpha_v3"]),
                    beta_v5=float(settings["beta_v5"]),
                )
            st.success(f"latency_bench tamamlandı: {report.get('output_file')}")
        if c3.button("coverage_audit (ALL)"):
            from src.coverage_audit import run_coverage_audit

            with st.spinner("coverage_audit çalışıyor..."):
                report = run_coverage_audit(
                    company=None,
                    gold_file=CONFIG.evaluation.gold_multicompany_file,
                    top_k_initial_v3=int(settings["top_k_initial_v3"]),
                    top_k=int(settings["top_k_final"]),
                    alpha_v3=float(settings["alpha_v3"]),
                )
            st.success(f"coverage_audit tamamlandı: {report.get('output_file')}")


def main() -> None:
    _ensure_paths()
    st.set_page_config(page_title="Bilanço Asistanı", layout="wide")
    st.title("Bilanço Asistanı")
    st.caption("Finansal raporlar için yerel, kanıt odaklı analiz deneyimi")

    if not _is_supported_python():
        st.error(_runtime_help_message())
        st.code(r".\.venv39\Scripts\python.exe -m streamlit run app/ui.py")
        st.stop()

    _get_ui_settings()
    st.sidebar.title("Navigasyon")
    pages = ["Overview", "Companies", "Reports", "Ask"]
    if getattr(CONFIG, "kap", None) is not None and bool(getattr(CONFIG.kap, "enabled", False)):
        pages.append("KAP")
    pages.append("Settings")
    if st.session_state.get("nav_page") not in pages:
        st.session_state["nav_page"] = "Overview"

    page = st.sidebar.radio(
        "Sayfalar",
        pages,
        index=pages.index(str(st.session_state.get("nav_page", "Overview"))),
        format_func=lambda x: {
            "Overview": "Genel Bakış",
            "Companies": "Şirketler",
            "Reports": "Raporlar",
            "Ask": "Soru Sor",
            "KAP": "KAP Finansallari",
            "Settings": "Ayarlar",
        }[x],
        key="sidebar_page",
    )
    st.session_state["nav_page"] = page

    st.sidebar.caption("Varsayılan üretim modu: v3")
    st.sidebar.caption("60 sn: Raporlar -> Genel Bakış -> Soru Sor")

    if page == "Overview":
        _render_overview_page()
    elif page == "Companies":
        _render_companies_page()
    elif page == "Reports":
        _render_reports_page()
    elif page == "Ask":
        _render_ask_page()
    elif page == "KAP":
        _render_kap_financials_page()
    else:
        _render_settings_page()


if __name__ == "__main__":
    main()
