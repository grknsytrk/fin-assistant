from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.metrics_extractor import normalize_for_match
from src.ratio_engine import build_ratio_table


@dataclass
class _DummyChunk:
    text: str
    company: str = "BIM"
    quarter: str = "2025Q1"
    year: int = 2025
    page: int = 1
    chunk_id: str = "c1"
    section_title: str = "Gelir Tablosu"
    block_type: str = "table_like"
    doc_id: str = "dummy.pdf"


class _DummyRetriever:
    def __init__(self, payload: Dict[Tuple[str, str], List[_DummyChunk]]) -> None:
        self.payload = payload

    def retrieve_with_query_awareness(
        self,
        query: str,
        top_k_initial: int = 20,
        top_k_final: int = 5,
        alpha: float = 0.35,
        quarter_override: Optional[str] = None,
        company_override: Optional[str] = None,
    ) -> List[_DummyChunk]:
        query_norm = normalize_for_match(query)
        metric = "__none__"
        if "net kar marji" in query_norm:
            metric = "net_kar_marji"
        elif "favok marji" in query_norm:
            metric = "favok_marji"
        elif "brut kar marji" in query_norm:
            metric = "brut_kar_marji"
        elif "serbest nakit akisi" in query_norm:
            metric = "serbest_nakit_akisi"
        elif "faaliyetlerden elde edilen nakit akisi" in query_norm:
            metric = "faaliyet_nakit_akisi"
        elif "yatirim harcamalari" in query_norm:
            metric = "capex"
        elif "brut kar" in query_norm:
            metric = "brut_kar"
        elif "net kar" in query_norm:
            metric = "net_kar"
        elif "favok" in query_norm:
            metric = "favok"
        elif "satislar" in query_norm:
            metric = "satis_gelirleri"
        elif "magaza sayisi" in query_norm:
            metric = "magaza_sayisi"
        key = (metric, str(quarter_override or "Q1").upper())
        return list(self.payload.get(key, []))


def test_ratio_self_verification_falls_back_from_inconsistent_direct_margin() -> None:
    payload = {
        ("net_kar", "Q1"): [_DummyChunk(text="net kar 1000000", chunk_id="nk1", quarter="2025Q1")],
        ("satis_gelirleri", "Q1"): [_DummyChunk(text="satis gelirleri 10000000", chunk_id="sg1", quarter="2025Q1")],
        ("favok", "Q1"): [_DummyChunk(text="favok 1500000", chunk_id="fv1", quarter="2025Q1")],
        ("net_kar_marji", "Q1"): [
            _DummyChunk(
                text="net kar marji 40 net kar marji 11",
                chunk_id="nkm1",
                quarter="2025Q1",
            )
        ],
        ("favok_marji", "Q1"): [_DummyChunk(text="favok marji 15", chunk_id="fvm1", quarter="2025Q1")],
        ("brut_kar_marji", "Q1"): [_DummyChunk(text="brut kar marji 20", chunk_id="bkm1", quarter="2025Q1")],
        ("magaza_sayisi", "Q1"): [_DummyChunk(text="magaza sayisi 1200", chunk_id="ms1", quarter="2025Q1")],
    }
    retriever = _DummyRetriever(payload)
    result = build_ratio_table(question="q1 oranlar", retriever=retriever, company="BIM")
    frame = result["frame"]
    q1_row = frame[frame["quarter"] == "Q1"].iloc[0]
    # Computed net margin is 10%. Direct 40% should be rejected and fallback to 11%.
    assert round(float(q1_row["net_margin"]), 2) == 11.0
    assert str(q1_row["net_margin_verify_status"]).upper() in {"WARN", "PASS"}


def test_free_cash_flow_is_computed_from_operating_cash_flow_and_capex() -> None:
    payload = {
        ("faaliyet_nakit_akisi", "Q1"): [
            _DummyChunk(
                text="faaliyetlerden elde edilen nakit akisi 2000000",
                chunk_id="ocf1",
                quarter="2025Q1",
                block_type="text",
            )
        ],
        ("capex", "Q1"): [
            _DummyChunk(
                text="yatirim harcamalari 500000",
                chunk_id="capex1",
                quarter="2025Q1",
                block_type="text",
            )
        ],
    }
    retriever = _DummyRetriever(payload)
    result = build_ratio_table(question="q1 nakit akisi", retriever=retriever, company="BIM")
    frame = result["frame"]
    q1_row = frame[frame["quarter"] == "Q1"].iloc[0]
    assert round(float(q1_row["capex"]), 2) == 500000.0
    assert round(float(q1_row["serbest_nakit_akisi"]), 2) == 1500000.0
    assert str(q1_row["serbest_nakit_akisi_verify_status"]).upper() in {"PASS", "WARN"}
