from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.config import load_config
from src.metrics_extractor import (
    QUARTER_ORDER,
    METRIC_DEFINITIONS,
    aggregate_metric_across_quarters,
    build_metric_query,
    infer_metric_from_question,
    metric_display_name,
    normalize_for_match,
)
from src.validators import validate_ratios

if TYPE_CHECKING:
    from src.retrieve import RetrievedChunk, RetrieverV3

COMPARISON_HINTS = (
    "karsilastir",
    "hangisi daha iyi",
    "karsilastirma",
    "en iyi",
)
BASE_METRICS = (
    "net_kar",
    "brut_kar",
    "satis_gelirleri",
    "favok",
    "faaliyet_nakit_akisi",
    "capex",
    "serbest_nakit_akisi",
    "brut_kar_marji",
    "magaza_sayisi",
)
MARGIN_METRICS = ("net_kar_marji", "favok_marji")
DEFAULT_COMPARISON_TARGET = "net_margin"

try:
    _RATIO_CONFIG = load_config(Path("config.yaml"))
    RATIO_SELF_VERIFY_PP_THRESHOLD = float(_RATIO_CONFIG.extraction.ratio_self_verify_pp_threshold)
except Exception:  # pragma: no cover
    RATIO_SELF_VERIFY_PP_THRESHOLD = 10.0


def normalize_company_name(company: Optional[str]) -> Optional[str]:
    if not company:
        return None
    normalized = re.sub(r"\s+", " ", str(company)).strip()
    if not normalized:
        return None
    return normalized.upper()


def _quarter_bucket(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    payload = str(value).strip().upper()
    if payload in QUARTER_ORDER:
        return payload
    match = re.search(r"Q([1-4])", payload)
    if match:
        return f"Q{match.group(1)}"
    return None


@lru_cache(maxsize=32)
def _load_local_company_quarter_chunks(company: str) -> Dict[str, Tuple["RetrievedChunk", ...]]:
    """
    Fallback pool from local chunk file. Used only to fill missing quarters.
    """
    company_norm = normalize_company_name(company)
    if not company_norm:
        return {}

    try:
        cfg = load_config(Path("config.yaml"))
        chunk_file = cfg.paths.chunks_v2_file
    except Exception:
        chunk_file = Path("data/processed/chunks_v2.jsonl")

    if not chunk_file.exists():
        return {}

    from src.retrieve import RetrievedChunk

    by_quarter: Dict[str, List["RetrievedChunk"]] = {quarter: [] for quarter in QUARTER_ORDER}
    with chunk_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            row_company = normalize_company_name(str(row.get("company", "")))
            if row_company != company_norm:
                continue
            bucket = _quarter_bucket(row.get("quarter"))
            if not bucket:
                continue
            by_quarter[bucket].append(
                RetrievedChunk(
                    text=str(row.get("text", "")),
                    distance=0.0,
                    score=1.0,
                    final_score=1.0,
                    vector_score=0.0,
                    lexical_boost=0.0,
                    doc_id=str(row.get("doc_id", "")),
                    company=str(row.get("company", "")),
                    quarter=str(row.get("quarter", "")),
                    year=row.get("year"),
                    page=int(row.get("page") or 0),
                    chunk_id=str(row.get("chunk_id", "")),
                    section_title=str(row.get("section_title") or "(no heading)"),
                    block_type=str(row.get("block_type") or "text"),
                    chunk_version=str(row.get("chunk_version") or "v2"),
                )
            )

    return {quarter: tuple(items) for quarter, items in by_quarter.items() if items}


def _fill_missing_metric_from_local_chunks(
    metric: str,
    frame: pd.DataFrame,
    records: List[Dict[str, Any]],
    local_quarter_chunks: Dict[str, Tuple["RetrievedChunk", ...]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if frame is None or frame.empty or not local_quarter_chunks:
        return frame, records
    if "quarter" not in frame.columns or "value" not in frame.columns:
        return frame, records

    updated = frame.copy()
    record_by_quarter = {str(item.get("quarter")): item for item in records}
    had_change = False

    for quarter in QUARTER_ORDER:
        row = updated[updated["quarter"] == quarter]
        current_value: Optional[float] = None
        if not row.empty:
            raw_value = row.iloc[-1].get("value")
            current_value = None if raw_value is None or pd.isna(raw_value) else float(raw_value)
        if current_value is not None:
            continue

        chunks = list(local_quarter_chunks.get(quarter, ()))
        if not chunks:
            continue

        fallback_frame, fallback_records = aggregate_metric_across_quarters(
            quarter_chunks={quarter: chunks},
            metric=metric,
        )
        if fallback_frame is None or fallback_frame.empty:
            continue
        fallback_row = fallback_frame[fallback_frame["quarter"] == quarter]
        if fallback_row.empty:
            continue
        value = fallback_row.iloc[-1].get("value")
        if value is None or pd.isna(value):
            continue

        if row.empty:
            updated = pd.concat([updated, fallback_row.iloc[[-1]]], ignore_index=True)
        else:
            idx = row.index[-1]
            for col in updated.columns:
                if col in fallback_row.columns:
                    updated.at[idx, col] = fallback_row.iloc[-1].get(col)

        if quarter not in record_by_quarter:
            fallback_record = next(
                (item for item in fallback_records if str(item.get("quarter")) == quarter),
                None,
            )
            if fallback_record:
                patched = dict(fallback_record)
                reasons = list(patched.get("reasons", []))
                reasons.append("local_chunk_fallback")
                patched["reasons"] = reasons
                records.append(patched)
                record_by_quarter[quarter] = patched
        had_change = True

    if had_change:
        updated["quarter"] = pd.Categorical(updated["quarter"], categories=QUARTER_ORDER, ordered=True)
        updated = updated.sort_values("quarter").reset_index(drop=True)
        updated["quarter"] = updated["quarter"].astype(str)

    return updated, records


def _prefer_store_count_from_local_chunks(
    frame: pd.DataFrame,
    records: List[Dict[str, Any]],
    local_quarter_chunks: Dict[str, Tuple["RetrievedChunk", ...]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Prefer consolidated total-store candidates from local quarter chunks when
    retrieved candidates point to like-for-like/store-traffic rows.
    """
    if frame is None or frame.empty or not local_quarter_chunks:
        return frame, records
    if "quarter" not in frame.columns or "value" not in frame.columns:
        return frame, records

    local_frame, local_records = aggregate_metric_across_quarters(
        quarter_chunks={quarter: list(chunks) for quarter, chunks in local_quarter_chunks.items()},
        metric="magaza_sayisi",
    )
    if local_frame is None or local_frame.empty:
        return frame, records

    local_record_by_quarter = {str(item.get("quarter")): item for item in local_records}
    record_by_quarter = {str(item.get("quarter")): item for item in records}
    updated = frame.copy()
    changed = False

    for quarter in QUARTER_ORDER:
        local_record = local_record_by_quarter.get(quarter)
        if not local_record:
            continue
        local_value = local_record.get("value")
        if local_value is None:
            continue
        try:
            local_value_f = float(local_value)
        except Exception:
            continue

        current_record = record_by_quarter.get(quarter)
        current_value_f: Optional[float] = None
        current_reasons: List[str] = []
        if current_record is not None:
            try:
                current_value_f = float(current_record.get("value"))
            except Exception:
                current_value_f = None
            current_reasons = [str(item) for item in current_record.get("reasons", [])]

        replace = False
        if current_record is None or current_value_f is None:
            replace = True
        elif "store_like_for_like_context_penalty" in current_reasons:
            replace = True
        elif local_value_f >= max(current_value_f + 200.0, current_value_f * 1.03):
            local_reasons = {str(item) for item in local_record.get("reasons", [])}
            if "store_total_context_bonus" in local_reasons:
                replace = True

        if not replace:
            continue

        local_row = local_frame[local_frame["quarter"] == quarter]
        if local_row.empty:
            continue
        local_row_data = local_row.iloc[-1]
        row = updated[updated["quarter"] == quarter]
        if row.empty:
            updated = pd.concat([updated, local_row.iloc[[-1]]], ignore_index=True)
        else:
            idx = row.index[-1]
            for col in updated.columns:
                if col in local_row.columns:
                    updated.at[idx, col] = local_row_data.get(col)

        patched_record = dict(local_record)
        patched_reasons = [str(item) for item in patched_record.get("reasons", [])]
        patched_reasons.append("local_store_total_override")
        patched_record["reasons"] = patched_reasons

        if current_record is None:
            records.append(patched_record)
        else:
            for i, record in enumerate(records):
                if str(record.get("quarter")) == quarter:
                    records[i] = patched_record
                    break
        record_by_quarter[quarter] = patched_record
        changed = True

    if changed:
        updated["quarter"] = pd.Categorical(updated["quarter"], categories=QUARTER_ORDER, ordered=True)
        updated = updated.sort_values("quarter").reset_index(drop=True)
        updated["quarter"] = updated["quarter"].astype(str)
    return updated, records


def detect_company_mentions(
    question: str,
    available_companies: Optional[Sequence[str]] = None,
) -> List[str]:
    norm_q = normalize_for_match(question)
    mentioned: List[str] = []
    candidates = available_companies or []
    for raw_company in candidates:
        company = normalize_company_name(raw_company)
        if not company:
            continue
        company_norm = normalize_for_match(company)
        if company_norm and company_norm in norm_q and company not in mentioned:
            mentioned.append(company)
    return mentioned


def is_cross_company_query(
    question: str,
    available_companies: Optional[Sequence[str]] = None,
) -> bool:
    norm_q = normalize_for_match(question)
    has_hint = any(hint in norm_q for hint in COMPARISON_HINTS)
    mentions = detect_company_mentions(question, available_companies=available_companies)
    return has_hint or len(mentions) >= 2


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return (numerator / denominator) * 100.0


def _safe_growth(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def _extract_metric_values(frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    rows: Dict[str, Optional[float]] = {}
    if frame is None or frame.empty:
        return rows
    for _, row in frame.iterrows():
        quarter = str(row.get("quarter"))
        value = row.get("value")
        rows[quarter] = float(value) if value is not None and not pd.isna(value) else None
    return rows


def _last_non_null(values: Dict[str, Optional[float]]) -> Tuple[Optional[str], Optional[float]]:
    for quarter in reversed(QUARTER_ORDER):
        value = values.get(quarter)
        if value is not None:
            return quarter, value
    return None, None


def _metric_target_from_question(question: str) -> str:
    metric = infer_metric_from_question(question)
    if metric == "net_kar_marji":
        return "net_margin"
    if metric == "favok_marji":
        return "favok_margin"
    if metric == "brut_kar_marji":
        return "brut_kar_marji"
    if metric in BASE_METRICS:
        return metric
    if "marj" in normalize_for_match(question):
        return "net_margin"
    return DEFAULT_COMPARISON_TARGET


def _retrieve_quarter_metric_chunks(
    metric: str,
    question: str,
    retriever: "RetrieverV3",
    company: Optional[str],
    top_k_initial: int,
    top_k_final: int,
    alpha: float,
) -> Dict[str, Sequence["RetrievedChunk"]]:
    quarter_chunks: Dict[str, Sequence["RetrievedChunk"]] = {}
    for quarter in QUARTER_ORDER:
        metric_query = build_metric_query(metric, quarter, question)
        top_k_final_effective = top_k_final
        if metric == "magaza_sayisi":
            metric_query = (
                f"{metric_query} toplam magaza magazasi bulunmaktadir "
                "magaza sayilari ozet tablo"
            ).strip()
            top_k_final_effective = max(top_k_final, 20)
        try:
            chunks = retriever.retrieve_with_query_awareness(
                query=metric_query,
                top_k_initial=top_k_initial,
                top_k_final=top_k_final_effective,
                alpha=alpha,
                quarter_override=quarter,
                company_override=company,
                allow_quarter_fallback=False,
            )
        except TypeError:
            # Backward compatibility for test doubles / legacy retrievers
            # that do not yet accept the `allow_quarter_fallback` kwarg.
            chunks = retriever.retrieve_with_query_awareness(
                query=metric_query,
                top_k_initial=top_k_initial,
                top_k_final=top_k_final_effective,
                alpha=alpha,
                quarter_override=quarter,
                company_override=company,
            )
        quarter_chunks[quarter] = chunks
    return quarter_chunks


def build_ratio_table(
    question: str,
    retriever: "RetrieverV3",
    company: Optional[str] = None,
    top_k_initial: int = 30,
    top_k_final: int = 12,
    alpha: float = 0.35,
) -> Dict[str, Any]:
    company_norm = normalize_company_name(company)
    local_company_chunks: Dict[str, Tuple["RetrievedChunk", ...]] = {}
    can_use_local_fallback = bool(
        company_norm
        and hasattr(retriever, "collection")
        and hasattr(retriever, "client")
    )
    if can_use_local_fallback and company_norm:
        local_company_chunks = _load_local_company_quarter_chunks(company_norm)
    metric_frames: Dict[str, pd.DataFrame] = {}
    metric_values: Dict[str, Dict[str, Optional[float]]] = {}
    metric_records: Dict[str, List[Dict[str, Any]]] = {}
    metric_record_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    confidence_map: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for metric in BASE_METRICS + MARGIN_METRICS:
        quarter_chunks = _retrieve_quarter_metric_chunks(
            metric=metric,
            question=question,
            retriever=retriever,
            company=company_norm,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            alpha=alpha,
        )
        frame, records = aggregate_metric_across_quarters(quarter_chunks=quarter_chunks, metric=metric)
        if local_company_chunks:
            if metric == "magaza_sayisi":
                frame, records = _prefer_store_count_from_local_chunks(
                    frame=frame,
                    records=records,
                    local_quarter_chunks=local_company_chunks,
                )
            else:
                frame, records = _fill_missing_metric_from_local_chunks(
                    metric=metric,
                    frame=frame,
                    records=records,
                    local_quarter_chunks=local_company_chunks,
                )
        metric_frames[metric] = frame
        metric_values[metric] = _extract_metric_values(frame)
        metric_records[metric] = records
        metric_record_index[metric] = {str(record.get("quarter")): record for record in records}
        confidence_map[metric] = {}
        for quarter in QUARTER_ORDER:
            record = metric_record_index[metric].get(quarter)
            if record:
                confidence_map[metric][quarter] = {
                    "confidence": record.get("confidence"),
                    "reasons": list(record.get("reasons", [])),
                    "evidence": [
                        f"[{record.get('doc_id')} | {record.get('quarter')} | {record.get('page')} | {record.get('section_title')}]"
                    ],
                    "source_metric": metric,
                    "verify_status": record.get("verify_status", "FAIL"),
                    "verify_warnings": list(record.get("verify_warnings", [])),
                    "verify_checks": list(record.get("verify_checks", [])),
                }
            else:
                confidence_map[metric][quarter] = {
                    "confidence": None,
                    "reasons": ["veri_bulunamadi"],
                    "evidence": [],
                    "source_metric": metric,
                    "verify_status": "FAIL",
                    "verify_warnings": ["veri_bulunamadi"],
                    "verify_checks": [],
                }

    rows: List[Dict[str, Any]] = []
    previous_sales: Optional[float] = None
    previous_store: Optional[float] = None
    previous_free_cf: Optional[float] = None
    previous_sales_conf: Optional[float] = None
    previous_store_conf: Optional[float] = None
    previous_free_cf_conf: Optional[float] = None

    for quarter in QUARTER_ORDER:
        net_kar = metric_values["net_kar"].get(quarter)
        brut_kar = metric_values["brut_kar"].get(quarter)
        sales = metric_values["satis_gelirleri"].get(quarter)
        favok = metric_values["favok"].get(quarter)
        operating_cf = metric_values["faaliyet_nakit_akisi"].get(quarter)
        capex_raw = metric_values["capex"].get(quarter)
        capex = abs(float(capex_raw)) if capex_raw is not None else None
        direct_free_cf = metric_values["serbest_nakit_akisi"].get(quarter)
        brut_margin = metric_values["brut_kar_marji"].get(quarter)
        direct_net_margin = metric_values["net_kar_marji"].get(quarter)
        direct_favok_margin = metric_values["favok_marji"].get(quarter)
        stores = metric_values["magaza_sayisi"].get(quarter)

        computed_net_margin = _safe_ratio(net_kar, sales)
        computed_favok_margin = _safe_ratio(favok, sales)
        computed_free_cf = (operating_cf - capex) if operating_cf is not None and capex is not None else None
        net_kar_conf = confidence_map["net_kar"][quarter]["confidence"]
        brut_kar_conf = confidence_map["brut_kar"][quarter]["confidence"]
        sales_conf = confidence_map["satis_gelirleri"][quarter]["confidence"]
        favok_conf = confidence_map["favok"][quarter]["confidence"]
        operating_cf_conf = confidence_map["faaliyet_nakit_akisi"][quarter]["confidence"]
        capex_conf = confidence_map["capex"][quarter]["confidence"]
        direct_free_cf_conf = confidence_map["serbest_nakit_akisi"][quarter]["confidence"]
        brut_conf = confidence_map["brut_kar_marji"][quarter]["confidence"]
        store_conf = confidence_map["magaza_sayisi"][quarter]["confidence"]
        direct_net_margin_conf = confidence_map["net_kar_marji"][quarter]["confidence"]
        direct_favok_margin_conf = confidence_map["favok_marji"][quarter]["confidence"]

        def _derived_free_cf_payload() -> Tuple[Optional[float], Optional[float], List[str], List[str], str, List[str]]:
            if computed_free_cf is not None and operating_cf_conf is not None and capex_conf is not None:
                derived_conf = min(float(operating_cf_conf), float(capex_conf))
                derived_reasons = ["hesaplandi: faaliyet_nakit_akisi - capex"]
                derived_evidence = (
                    list(confidence_map["faaliyet_nakit_akisi"][quarter]["evidence"])
                    + list(confidence_map["capex"][quarter]["evidence"])
                )
                source_statuses = [
                    str(confidence_map["faaliyet_nakit_akisi"][quarter].get("verify_status", "FAIL")),
                    str(confidence_map["capex"][quarter].get("verify_status", "FAIL")),
                ]
                if all(status == "PASS" for status in source_statuses):
                    derived_status = "PASS"
                    derived_warnings: List[str] = []
                elif any(status == "FAIL" for status in source_statuses):
                    derived_status = "WARN"
                    derived_warnings = ["derived_from_low_quality_source"]
                else:
                    derived_status = "WARN"
                    derived_warnings = ["derived_metric_check_required"]
                return computed_free_cf, derived_conf, derived_reasons, derived_evidence, derived_status, derived_warnings
            return None, None, ["hesap_icin_veri_eksik"], [], "FAIL", ["hesap_icin_veri_eksik"]

        free_cf, free_cf_conf, free_cf_reasons, free_cf_evidence, free_cf_verify_status, free_cf_verify_warnings = _derived_free_cf_payload()
        if direct_free_cf is not None:
            direct_status = str(confidence_map["serbest_nakit_akisi"][quarter].get("verify_status", "FAIL")).upper()
            if direct_status == "PASS" or free_cf is None:
                free_cf = direct_free_cf
                free_cf_conf = direct_free_cf_conf
                free_cf_reasons = list(confidence_map["serbest_nakit_akisi"][quarter]["reasons"])
                free_cf_evidence = list(confidence_map["serbest_nakit_akisi"][quarter]["evidence"])
                free_cf_verify_status = direct_status
                free_cf_verify_warnings = list(confidence_map["serbest_nakit_akisi"][quarter].get("verify_warnings", []))

        def _derived_net_margin_payload() -> Tuple[Optional[float], Optional[float], List[str], List[str], str, List[str]]:
            if computed_net_margin is not None and net_kar_conf is not None and sales_conf is not None:
                derived_conf = min(float(net_kar_conf), float(sales_conf))
                derived_reasons = ["hesaplandi: net_kar / satis_gelirleri"]
                derived_evidence = (
                    list(confidence_map["net_kar"][quarter]["evidence"])
                    + list(confidence_map["satis_gelirleri"][quarter]["evidence"])
                )
                source_statuses = [
                    str(confidence_map["net_kar"][quarter].get("verify_status", "FAIL")),
                    str(confidence_map["satis_gelirleri"][quarter].get("verify_status", "FAIL")),
                ]
                if all(status == "PASS" for status in source_statuses):
                    derived_status = "PASS"
                    derived_warnings: List[str] = []
                elif any(status == "FAIL" for status in source_statuses):
                    derived_status = "WARN"
                    derived_warnings = ["derived_from_low_quality_source"]
                else:
                    derived_status = "WARN"
                    derived_warnings = ["derived_metric_check_required"]
                return computed_net_margin, derived_conf, derived_reasons, derived_evidence, derived_status, derived_warnings
            return None, None, ["hesap_icin_veri_eksik"], [], "FAIL", ["hesap_icin_veri_eksik"]

        net_margin, net_margin_conf, net_margin_reasons, net_margin_evidence, net_margin_verify_status, net_margin_verify_warnings = _derived_net_margin_payload()
        if direct_net_margin is not None:
            net_margin = direct_net_margin
            net_margin_conf = direct_net_margin_conf
            net_margin_reasons = list(confidence_map["net_kar_marji"][quarter]["reasons"])
            net_margin_evidence = list(confidence_map["net_kar_marji"][quarter]["evidence"])
            net_margin_verify_status = str(confidence_map["net_kar_marji"][quarter].get("verify_status", "FAIL"))
            net_margin_verify_warnings = list(confidence_map["net_kar_marji"][quarter].get("verify_warnings", []))
            if computed_net_margin is not None:
                deviation_pp = abs(float(direct_net_margin) - float(computed_net_margin))
                if deviation_pp > RATIO_SELF_VERIFY_PP_THRESHOLD:
                    # Try fallback: alternative direct-margin candidates first, then computed margin.
                    fallback_used = False
                    direct_record = metric_record_index.get("net_kar_marji", {}).get(quarter)
                    if direct_record:
                        direct_candidates = list(direct_record.get("candidates", []))
                        best_candidate: Optional[Dict[str, Any]] = None
                        best_dev = float("inf")
                        for candidate in direct_candidates:
                            if not bool(candidate.get("validation_ok")):
                                continue
                            cand_val = candidate.get("value")
                            if cand_val is None or pd.isna(cand_val):
                                continue
                            dev = abs(float(cand_val) - float(computed_net_margin))
                            if dev < best_dev:
                                best_dev = dev
                                best_candidate = candidate
                        if best_candidate is not None and best_dev <= RATIO_SELF_VERIFY_PP_THRESHOLD:
                            net_margin = float(best_candidate["value"])
                            net_margin_conf = direct_net_margin_conf
                            net_margin_reasons = list(net_margin_reasons) + [
                                "direct_margin_consistency_fail",
                                "margin_fallback_alt_direct_candidate",
                            ]
                            net_margin_verify_status = "WARN"
                            net_margin_verify_warnings = list(net_margin_verify_warnings) + [
                                "direct_vs_computed_margin_deviation",
                            ]
                            fallback_used = True
                    if not fallback_used:
                        derived_value, derived_conf, derived_reasons, derived_evidence, derived_status, derived_warnings = _derived_net_margin_payload()
                        net_margin = derived_value
                        net_margin_conf = derived_conf
                        net_margin_reasons = list(derived_reasons) + ["direct_margin_consistency_fail"]
                        net_margin_evidence = list(derived_evidence)
                        net_margin_verify_status = derived_status if derived_status != "PASS" else "WARN"
                        net_margin_verify_warnings = list(derived_warnings) + [
                            "direct_vs_computed_margin_deviation",
                        ]

        def _derived_favok_margin_payload() -> Tuple[Optional[float], Optional[float], List[str], List[str], str, List[str]]:
            if computed_favok_margin is not None and favok_conf is not None and sales_conf is not None:
                derived_conf = min(float(favok_conf), float(sales_conf))
                derived_reasons = ["hesaplandi: favok / satis_gelirleri"]
                derived_evidence = (
                    list(confidence_map["favok"][quarter]["evidence"])
                    + list(confidence_map["satis_gelirleri"][quarter]["evidence"])
                )
                source_statuses = [
                    str(confidence_map["favok"][quarter].get("verify_status", "FAIL")),
                    str(confidence_map["satis_gelirleri"][quarter].get("verify_status", "FAIL")),
                ]
                if all(status == "PASS" for status in source_statuses):
                    derived_status = "PASS"
                    derived_warnings: List[str] = []
                elif any(status == "FAIL" for status in source_statuses):
                    derived_status = "WARN"
                    derived_warnings = ["derived_from_low_quality_source"]
                else:
                    derived_status = "WARN"
                    derived_warnings = ["derived_metric_check_required"]
                return computed_favok_margin, derived_conf, derived_reasons, derived_evidence, derived_status, derived_warnings
            return None, None, ["hesap_icin_veri_eksik"], [], "FAIL", ["hesap_icin_veri_eksik"]

        favok_margin, favok_margin_conf, favok_margin_reasons, favok_margin_evidence, favok_margin_verify_status, favok_margin_verify_warnings = _derived_favok_margin_payload()
        if direct_favok_margin is not None:
            favok_margin = direct_favok_margin
            favok_margin_conf = direct_favok_margin_conf
            favok_margin_reasons = list(confidence_map["favok_marji"][quarter]["reasons"])
            favok_margin_evidence = list(confidence_map["favok_marji"][quarter]["evidence"])
            favok_margin_verify_status = str(confidence_map["favok_marji"][quarter].get("verify_status", "FAIL"))
            favok_margin_verify_warnings = list(confidence_map["favok_marji"][quarter].get("verify_warnings", []))
            if computed_favok_margin is not None:
                deviation_pp = abs(float(direct_favok_margin) - float(computed_favok_margin))
                if deviation_pp > RATIO_SELF_VERIFY_PP_THRESHOLD:
                    fallback_used = False
                    direct_record = metric_record_index.get("favok_marji", {}).get(quarter)
                    if direct_record:
                        direct_candidates = list(direct_record.get("candidates", []))
                        best_candidate: Optional[Dict[str, Any]] = None
                        best_dev = float("inf")
                        for candidate in direct_candidates:
                            if not bool(candidate.get("validation_ok")):
                                continue
                            cand_val = candidate.get("value")
                            if cand_val is None or pd.isna(cand_val):
                                continue
                            dev = abs(float(cand_val) - float(computed_favok_margin))
                            if dev < best_dev:
                                best_dev = dev
                                best_candidate = candidate
                        if best_candidate is not None and best_dev <= RATIO_SELF_VERIFY_PP_THRESHOLD:
                            favok_margin = float(best_candidate["value"])
                            favok_margin_conf = direct_favok_margin_conf
                            favok_margin_reasons = list(favok_margin_reasons) + [
                                "direct_margin_consistency_fail",
                                "margin_fallback_alt_direct_candidate",
                            ]
                            favok_margin_verify_status = "WARN"
                            favok_margin_verify_warnings = list(favok_margin_verify_warnings) + [
                                "direct_vs_computed_margin_deviation",
                            ]
                            fallback_used = True
                    if not fallback_used:
                        derived_value, derived_conf, derived_reasons, derived_evidence, derived_status, derived_warnings = _derived_favok_margin_payload()
                        favok_margin = derived_value
                        favok_margin_conf = derived_conf
                        favok_margin_reasons = list(derived_reasons) + ["direct_margin_consistency_fail"]
                        favok_margin_evidence = list(derived_evidence)
                        favok_margin_verify_status = derived_status if derived_status != "PASS" else "WARN"
                        favok_margin_verify_warnings = list(derived_warnings) + [
                            "direct_vs_computed_margin_deviation",
                        ]

        revenue_growth_qoq = _safe_growth(sales, previous_sales)
        store_growth_qoq = _safe_growth(stores, previous_store)
        cash_flow_growth_qoq = _safe_growth(free_cf, previous_free_cf)
        revenue_growth_conf = (
            min(float(sales_conf), float(previous_sales_conf))
            if revenue_growth_qoq is not None and sales_conf is not None and previous_sales_conf is not None
            else None
        )
        store_growth_conf = (
            min(float(store_conf), float(previous_store_conf))
            if store_growth_qoq is not None and store_conf is not None and previous_store_conf is not None
            else None
        )
        cash_flow_growth_conf = (
            min(float(free_cf_conf), float(previous_free_cf_conf))
            if cash_flow_growth_qoq is not None and free_cf_conf is not None and previous_free_cf_conf is not None
            else None
        )
        revenue_growth_verify_status = (
            "PASS"
            if revenue_growth_conf is not None
            and confidence_map["satis_gelirleri"][quarter].get("verify_status") == "PASS"
            else "FAIL"
        )
        if revenue_growth_verify_status == "FAIL" and revenue_growth_qoq is not None:
            revenue_growth_verify_status = "WARN"
        store_growth_verify_status = (
            "PASS"
            if store_growth_conf is not None
            and confidence_map["magaza_sayisi"][quarter].get("verify_status") == "PASS"
            else "FAIL"
        )
        if store_growth_verify_status == "FAIL" and store_growth_qoq is not None:
            store_growth_verify_status = "WARN"
        cash_flow_growth_verify_status = "PASS" if cash_flow_growth_conf is not None else "FAIL"
        if cash_flow_growth_verify_status == "FAIL" and cash_flow_growth_qoq is not None:
            cash_flow_growth_verify_status = "WARN"

        row = {
            "company": company_norm or "ALL",
            "quarter": quarter,
            "net_kar": net_kar,
            "brut_kar": brut_kar,
            "satis_gelirleri": sales,
            "favok": favok,
            "faaliyet_nakit_akisi": operating_cf,
            "capex": capex,
            "serbest_nakit_akisi": free_cf,
            "brut_kar_marji": brut_margin,
            "magaza_sayisi": stores,
            "net_margin": net_margin,
            "favok_margin": favok_margin,
            "revenue_growth_qoq": revenue_growth_qoq,
            "store_growth_qoq": store_growth_qoq,
            "cash_flow_growth_qoq": cash_flow_growth_qoq,
            "net_kar_confidence": net_kar_conf,
            "brut_kar_confidence": brut_kar_conf,
            "satis_gelirleri_confidence": sales_conf,
            "favok_confidence": favok_conf,
            "faaliyet_nakit_akisi_confidence": operating_cf_conf,
            "capex_confidence": capex_conf,
            "serbest_nakit_akisi_confidence": free_cf_conf,
            "brut_kar_marji_confidence": brut_conf,
            "magaza_sayisi_confidence": store_conf,
            "net_margin_confidence": net_margin_conf,
            "favok_margin_confidence": favok_margin_conf,
            "revenue_growth_qoq_confidence": revenue_growth_conf,
            "store_growth_qoq_confidence": store_growth_conf,
            "cash_flow_growth_qoq_confidence": cash_flow_growth_conf,
            "net_kar_verify_status": confidence_map["net_kar"][quarter].get("verify_status"),
            "brut_kar_verify_status": confidence_map["brut_kar"][quarter].get("verify_status"),
            "satis_gelirleri_verify_status": confidence_map["satis_gelirleri"][quarter].get("verify_status"),
            "favok_verify_status": confidence_map["favok"][quarter].get("verify_status"),
            "faaliyet_nakit_akisi_verify_status": confidence_map["faaliyet_nakit_akisi"][quarter].get("verify_status"),
            "capex_verify_status": confidence_map["capex"][quarter].get("verify_status"),
            "serbest_nakit_akisi_verify_status": free_cf_verify_status,
            "brut_kar_marji_verify_status": confidence_map["brut_kar_marji"][quarter].get("verify_status"),
            "magaza_sayisi_verify_status": confidence_map["magaza_sayisi"][quarter].get("verify_status"),
            "net_margin_verify_status": net_margin_verify_status,
            "favok_margin_verify_status": favok_margin_verify_status,
            "revenue_growth_qoq_verify_status": revenue_growth_verify_status,
            "store_growth_qoq_verify_status": store_growth_verify_status,
            "cash_flow_growth_qoq_verify_status": cash_flow_growth_verify_status,
        }

        ratio_validation = validate_ratios(row)
        if not ratio_validation.get("ok", True):
            for flag in ratio_validation.get("flags", []):
                if flag == "brut_kar_marji_aralik_disi":
                    row["brut_kar_marji"] = None
                    row["brut_kar_marji_confidence"] = None
                elif flag == "net_marj_aralik_disi":
                    row["net_margin"] = None
                    row["net_margin_confidence"] = None
                elif flag == "favok_marji_aralik_disi":
                    row["favok_margin"] = None
                    row["favok_margin_confidence"] = None
            row["ratio_flags"] = list(ratio_validation.get("flags", []))
        else:
            row["ratio_flags"] = []

        confidence_values = [
            row["net_kar_confidence"],
            row["brut_kar_confidence"],
            row["satis_gelirleri_confidence"],
            row["favok_confidence"],
            row["faaliyet_nakit_akisi_confidence"],
            row["capex_confidence"],
            row["serbest_nakit_akisi_confidence"],
            row["net_margin_confidence"],
            row["favok_margin_confidence"],
            row["brut_kar_marji_confidence"],
        ]
        confidence_values = [float(v) for v in confidence_values if v is not None]
        row["overall_confidence"] = min(confidence_values) if confidence_values else None

        confidence_map.setdefault("net_margin", {})[quarter] = {
            "confidence": row["net_margin_confidence"],
            "reasons": net_margin_reasons,
            "evidence": net_margin_evidence,
            "source_metric": "net_kar_marji" if direct_net_margin is not None else "derived",
            "verify_status": net_margin_verify_status,
            "verify_warnings": net_margin_verify_warnings,
            "verify_checks": [],
        }
        confidence_map.setdefault("favok_margin", {})[quarter] = {
            "confidence": row["favok_margin_confidence"],
            "reasons": favok_margin_reasons,
            "evidence": favok_margin_evidence,
            "source_metric": "favok_marji" if direct_favok_margin is not None else "derived",
            "verify_status": favok_margin_verify_status,
            "verify_warnings": favok_margin_verify_warnings,
            "verify_checks": [],
        }
        confidence_map.setdefault("revenue_growth_qoq", {})[quarter] = {
            "confidence": row["revenue_growth_qoq_confidence"],
            "reasons": ["hesaplandi: satis_gelirleri_qoq"] if row["revenue_growth_qoq_confidence"] is not None else ["hesap_icin_veri_eksik"],
            "evidence": list(confidence_map["satis_gelirleri"][quarter]["evidence"]),
            "source_metric": "derived",
            "verify_status": revenue_growth_verify_status,
            "verify_warnings": [] if revenue_growth_verify_status == "PASS" else ["derived_metric_check_required"],
            "verify_checks": [],
        }
        confidence_map.setdefault("store_growth_qoq", {})[quarter] = {
            "confidence": row["store_growth_qoq_confidence"],
            "reasons": ["hesaplandi: magaza_sayisi_qoq"] if row["store_growth_qoq_confidence"] is not None else ["hesap_icin_veri_eksik"],
            "evidence": list(confidence_map["magaza_sayisi"][quarter]["evidence"]),
            "source_metric": "derived",
            "verify_status": store_growth_verify_status,
            "verify_warnings": [] if store_growth_verify_status == "PASS" else ["derived_metric_check_required"],
            "verify_checks": [],
        }
        confidence_map.setdefault("serbest_nakit_akisi", {})[quarter] = {
            "confidence": row["serbest_nakit_akisi_confidence"],
            "reasons": free_cf_reasons,
            "evidence": free_cf_evidence,
            "source_metric": "serbest_nakit_akisi" if direct_free_cf is not None else "derived",
            "verify_status": free_cf_verify_status,
            "verify_warnings": free_cf_verify_warnings,
            "verify_checks": [],
        }
        confidence_map.setdefault("cash_flow_growth_qoq", {})[quarter] = {
            "confidence": row["cash_flow_growth_qoq_confidence"],
            "reasons": ["hesaplandi: serbest_nakit_akisi_qoq"] if row["cash_flow_growth_qoq_confidence"] is not None else ["hesap_icin_veri_eksik"],
            "evidence": list(confidence_map["serbest_nakit_akisi"][quarter]["evidence"]),
            "source_metric": "derived",
            "verify_status": cash_flow_growth_verify_status,
            "verify_warnings": [] if cash_flow_growth_verify_status == "PASS" else ["derived_metric_check_required"],
            "verify_checks": [],
        }

        rows.append(row)
        previous_sales = sales if sales is not None else previous_sales
        previous_store = stores if stores is not None else previous_store
        previous_free_cf = free_cf if free_cf is not None else previous_free_cf
        previous_sales_conf = sales_conf if sales_conf is not None else previous_sales_conf
        previous_store_conf = store_conf if store_conf is not None else previous_store_conf
        previous_free_cf_conf = free_cf_conf if free_cf_conf is not None else previous_free_cf_conf

    frame = pd.DataFrame(rows)
    found = bool(
        frame.dropna(
            subset=[
                "net_kar",
                "brut_kar",
                "satis_gelirleri",
                "favok",
                "faaliyet_nakit_akisi",
                "capex",
                "serbest_nakit_akisi",
                "magaza_sayisi",
            ],
            how="all",
        ).shape[0]
    )

    latest = frame.iloc[-1] if not frame.empty else None
    overall_confidence: Optional[float] = None
    if latest is not None and latest.get("overall_confidence") is not None and not pd.isna(latest.get("overall_confidence")):
        overall_confidence = float(latest.get("overall_confidence"))
    explanation: List[str] = []
    if latest is not None:
        explanation.append(f"Son ceyrek: {latest['quarter']}")
        if latest.get("brut_kar") is not None and not pd.isna(latest.get("brut_kar")):
            explanation.append(f"Brut kar: {float(latest['brut_kar']):,.0f}".replace(",", "."))
        if latest.get("brut_kar_marji") is not None and not pd.isna(latest.get("brut_kar_marji")):
            explanation.append(f"Brut kar marji: %{float(latest['brut_kar_marji']):.2f}")
        if latest.get("net_margin") is not None and not pd.isna(latest.get("net_margin")):
            explanation.append(f"Net marj: %{float(latest['net_margin']):.2f}")
        if latest.get("favok_margin") is not None and not pd.isna(latest.get("favok_margin")):
            explanation.append(f"FAVOK marji: %{float(latest['favok_margin']):.2f}")
        if latest.get("serbest_nakit_akisi") is not None and not pd.isna(latest.get("serbest_nakit_akisi")):
            explanation.append(f"Serbest nakit akisi: {float(latest['serbest_nakit_akisi']):,.0f}".replace(",", "."))

    return {
        "company": company_norm or "ALL",
        "frame": frame,
        "found": found,
        "metric_frames": metric_frames,
        "metric_records": metric_records,
        "confidence_map": confidence_map,
        "overall_confidence": overall_confidence,
        "explanation": explanation,
    }


def _target_value_from_row(row: pd.Series, target: str) -> Optional[float]:
    value = row.get(target)
    if value is None or pd.isna(value):
        return None
    return float(value)


def run_cross_company_comparison(
    question: str,
    retriever: "RetrieverV3",
    companies: Sequence[str],
    top_k_initial: int = 30,
    top_k_final: int = 12,
    alpha: float = 0.35,
) -> Dict[str, Any]:
    company_list = [c for c in (normalize_company_name(item) for item in companies) if c]
    company_list = list(dict.fromkeys(company_list))
    if len(company_list) < 2:
        return {
            "found": False,
            "target": _metric_target_from_question(question),
            "frame": pd.DataFrame(),
            "best_company": None,
            "best_value": None,
            "company_results": {},
            "evidence": [],
            "message": "Karsilastirma icin en az iki sirket gerekli.",
        }

    target = _metric_target_from_question(question)
    company_results: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []

    for company in company_list:
        result = build_ratio_table(
            question=question,
            retriever=retriever,
            company=company,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            alpha=alpha,
        )
        company_results[company] = result
        frame = result["frame"]
        confidence_map = result.get("confidence_map", {})
        quarter: Optional[str] = None
        value: Optional[float] = None
        confidence: Optional[float] = None
        if frame is not None and not frame.empty:
            for _, row in frame.iloc[::-1].iterrows():
                metric_value = _target_value_from_row(row, target)
                if metric_value is not None:
                    quarter = str(row["quarter"])
                    value = metric_value
                    confidence_detail = confidence_map.get(target, {}).get(quarter, {})
                    confidence_raw = confidence_detail.get("confidence")
                    confidence = float(confidence_raw) if confidence_raw is not None else None
                    break
        rows.append(
            {
                "company": company,
                "target": target,
                "quarter": quarter,
                "value": value,
                "confidence": confidence,
            }
        )

        if target in METRIC_DEFINITIONS:
            for record in result["metric_records"].get(target, []):
                evidence.append(record)
        elif target == "net_margin":
            evidence.extend(result["metric_records"].get("net_kar", []))
            evidence.extend(result["metric_records"].get("satis_gelirleri", []))
        elif target == "favok_margin":
            evidence.extend(result["metric_records"].get("favok", []))
            evidence.extend(result["metric_records"].get("satis_gelirleri", []))

    frame = pd.DataFrame(rows)
    valid = frame.dropna(subset=["value"])
    if valid.empty:
        return {
            "found": False,
            "target": target,
            "frame": frame,
            "best_company": None,
            "best_value": None,
            "company_results": company_results,
            "evidence": evidence,
            "message": "Karsilastirma icin yeterli metrik bulunamadi.",
        }

    best_idx = valid["value"].idxmax()
    best_company = str(valid.loc[best_idx, "company"])
    best_value = float(valid.loc[best_idx, "value"])

    return {
        "found": True,
        "target": target,
        "target_label": metric_display_name(target) if target in METRIC_DEFINITIONS else target,
        "frame": frame,
        "best_company": best_company,
        "best_value": best_value,
        "best_confidence": float(valid.loc[best_idx, "confidence"]) if not pd.isna(valid.loc[best_idx, "confidence"]) else None,
        "company_results": company_results,
        "evidence": evidence,
        "message": f"En iyi performans: {best_company}",
    }


def build_executive_summary(
    ratio_result: Dict[str, Any],
    max_bullets: int = 5,
) -> List[str]:
    frame: pd.DataFrame = ratio_result.get("frame", pd.DataFrame())
    metric_records: Dict[str, List[Dict[str, Any]]] = ratio_result.get("metric_records", {})
    company = str(ratio_result.get("company", "ALL"))
    bullets: List[str] = []

    if frame is None or frame.empty:
        return [
            "Finansal ozet icin yeterli veri bulunamadi.",
            "Net kar bilgisi bulunamadi.",
            "Net marj bilgisi bulunamadi.",
            "FAVOK marji bilgisi bulunamadi.",
            "Ciro/magaza degisim bilgisi bulunamadi.",
        ][:max_bullets]

    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) >= 2 else None
    latest_q = str(latest.get("quarter", "-"))

    def _fmt_float(value: Any, digits: int = 2) -> str:
        if value is None or pd.isna(value):
            return "Bulunamadi"
        return f"{float(value):.{digits}f}".replace(".", ",")

    def _fmt_int(value: Any) -> str:
        if value is None or pd.isna(value):
            return "Bulunamadi"
        return f"{float(value):,.0f}".replace(",", ".")

    def _metric_citation(metric: str, quarter: str) -> str:
        for record in metric_records.get(metric, []):
            if str(record.get("quarter")) == quarter:
                return (
                    f"[{record.get('doc_id')} | {record.get('company', company)} | "
                    f"{record.get('quarter')} | {record.get('page')}]"
                )
        return "[Kanit bulunamadi]"

    net_kar = latest.get("net_kar")
    brut_kar = latest.get("brut_kar")
    operating_cf = latest.get("faaliyet_nakit_akisi")
    capex = latest.get("capex")
    free_cf = latest.get("serbest_nakit_akisi")
    brut_marj = latest.get("brut_kar_marji")
    net_marj = latest.get("net_margin")
    favok_marj = latest.get("favok_margin")
    rev_growth = latest.get("revenue_growth_qoq")
    store_growth = latest.get("store_growth_qoq")
    cash_flow_growth = latest.get("cash_flow_growth_qoq")

    bullets.append(
        f"{latest_q} doneminde net kar: {_fmt_int(net_kar)} (Kanit: {_metric_citation('net_kar', latest_q)})."
    )
    bullets.append(
        f"Brut kar: {_fmt_int(brut_kar)} (Kanit: {_metric_citation('brut_kar', latest_q)})."
    )
    bullets.append(
        f"Brut kar marji: %{_fmt_float(brut_marj)} | Net marj: %{_fmt_float(net_marj)} | FAVOK marji: %{_fmt_float(favok_marj)}."
    )
    bullets.append(
        f"Faaliyet nakit akisi: {_fmt_int(operating_cf)} | CAPEX: {_fmt_int(capex)} | Serbest nakit akisi: {_fmt_int(free_cf)}."
    )
    bullets.append(
        f"Ciro QoQ degisimi: %{_fmt_float(rev_growth)} | Magaza QoQ degisimi: %{_fmt_float(store_growth)} | Nakit akisi QoQ: %{_fmt_float(cash_flow_growth)}."
    )

    if previous is not None:
        prev_q = str(previous.get("quarter", "-"))
        latest_net = latest.get("net_kar")
        prev_net = previous.get("net_kar")
        if latest_net is not None and prev_net is not None and not pd.isna(latest_net) and not pd.isna(prev_net):
            direction = "artis" if float(latest_net) > float(prev_net) else "azalis" if float(latest_net) < float(prev_net) else "yatay"
            bullets.append(f"Net kar {prev_q} -> {latest_q} doneminde {direction} gosterdi.")
        else:
            bullets.append(f"{prev_q} -> {latest_q} net kar karsilastirmasi icin veri eksik.")
    else:
        bullets.append("Ceyrekler arasi degisim yorumu icin onceki ceyrek verisi eksik.")

    known_values = [
        value
        for value in (brut_marj, net_marj, favok_marj, rev_growth, store_growth, cash_flow_growth, free_cf)
        if value is not None and not pd.isna(value)
    ]
    if known_values:
        bullets.append("Genel gorunum: temel KPI'larda en az birinde olumlu sinyal mevcut.")
    else:
        bullets.append("Genel gorunum: KPI verileri parcali, yorum sinirli.")

    return bullets[:max_bullets]


def detect_last_quarter_changes(ratio_result: Dict[str, Any]) -> Dict[str, List[str]]:
    frame: pd.DataFrame = ratio_result.get("frame", pd.DataFrame())
    metrics = {
        "net_kar": "Net kar",
        "brut_kar": "Brut kar",
        "faaliyet_nakit_akisi": "Faaliyet nakit akisi",
        "serbest_nakit_akisi": "Serbest nakit akisi",
        "brut_kar_marji": "Brut kar marji",
        "net_margin": "Net marj",
        "favok_margin": "FAVOK marji",
        "revenue_growth_qoq": "Ciro buyumesi (QoQ)",
        "store_growth_qoq": "Magaza buyumesi (QoQ)",
        "cash_flow_growth_qoq": "Nakit akisi buyumesi (QoQ)",
    }
    if frame is None or frame.empty:
        return {"improved": [], "worsened": [], "flat": []}

    # Ignore synthetic/empty quarter rows (e.g. Q4 exists but all metrics are NaN).
    valid = frame.dropna(subset=list(metrics.keys()), how="all")
    if len(valid) < 2:
        return {"improved": [], "worsened": [], "flat": []}

    latest = valid.iloc[-1]
    previous = valid.iloc[-2]
    improved: List[str] = []
    worsened: List[str] = []
    flat: List[str] = []

    for key, label in metrics.items():
        cur = latest.get(key)
        prev = previous.get(key)
        if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
            continue
        delta = float(cur) - float(prev)
        if abs(delta) < 1e-6:
            flat.append(label)
        elif delta > 0:
            improved.append(label)
        else:
            worsened.append(label)

    return {"improved": improved, "worsened": worsened, "flat": flat}
