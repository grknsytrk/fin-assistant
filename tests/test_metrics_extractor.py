from dataclasses import dataclass

from src.metrics_extractor import (
    aggregate_metric_across_quarters,
    extract_metric_with_candidates,
    parse_tr_number,
)


def test_metrics_extractor_parse_number_tr() -> None:
    assert parse_tr_number("1.234,56") == 1234.56
    assert parse_tr_number("1,234.56") == 1234.56
    assert parse_tr_number("11.367") == 11367.0
    assert parse_tr_number("5,7") == 5.7
    assert parse_tr_number("-123,4") == -123.4
    assert parse_tr_number("(123,4)") == -123.4


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


def test_metrics_extractor_million_billion_normalization() -> None:
    chunk1 = _DummyChunk(text="Net kar 1.234 milyon TL")
    res1 = extract_metric_with_candidates([chunk1], metric="net_kar", quarter="Q1")
    assert res1["selected"] is not None
    assert float(res1["selected"]["value"]) == 1_234_000_000.0

    chunk2 = _DummyChunk(text="FAVOK 2,5 milyar TL")
    res2 = extract_metric_with_candidates([chunk2], metric="favok", quarter="Q1")
    assert res2["selected"] is not None
    assert float(res2["selected"]["value"]) == 2_500_000_000.0


def test_metrics_extractor_prefers_latest_year_column() -> None:
    chunk = _DummyChunk(
        text="2024 2025 net kar 2.830.000.000 3.760.000.000",
        section_title="(no heading)",
        block_type="text",
    )
    result = extract_metric_with_candidates([chunk], metric="net_kar", quarter="Q1")
    assert result["selected"] is not None
    assert float(result["selected"]["value"]) == 3_760_000_000.0


def test_metrics_extractor_sanity_fallback_next_candidate() -> None:
    chunk = _DummyChunk(text="FAVOK marji 250 FAVOK marji 5")
    result = extract_metric_with_candidates([chunk], metric="favok_marji", quarter="Q1")
    assert result["selected"] is not None
    assert float(result["selected"]["value"]) == 5.0


def test_metrics_extractor_implicit_million_table_scaling() -> None:
    chunk = _DummyChunk(
        text="2024 2025 net satislar 67.651 70.873",
        section_title="Ozet Konsolide Gelir Tablosu",
        block_type="table_like",
    )
    result = extract_metric_with_candidates([chunk], metric="satis_gelirleri", quarter="Q3")
    assert result["selected"] is not None
    assert float(result["selected"]["value"]) >= 67_000_000_000.0


def test_metrics_extractor_negative_with_minus_and_parenthesis() -> None:
    minus_chunk = _DummyChunk(text="net kar -123,4 milyon TL", section_title="Gelir Tablosu", block_type="table_like")
    minus_result = extract_metric_with_candidates([minus_chunk], metric="net_kar", quarter="Q1")
    assert minus_result["selected"] is not None
    assert float(minus_result["selected"]["value"]) == -123_400_000.0

    paren_chunk = _DummyChunk(text="net kar (123,4) milyon TL", section_title="Gelir Tablosu", block_type="table_like")
    paren_result = extract_metric_with_candidates([paren_chunk], metric="net_kar", quarter="Q1")
    assert paren_result["selected"] is not None
    assert float(paren_result["selected"]["value"]) == -123_400_000.0


def test_metrics_extractor_negative_with_zarar_context() -> None:
    zarar_chunk = _DummyChunk(
        text="net donem zarari 123,4 milyon TL",
        section_title="Gelir Tablosu",
        block_type="table_like",
    )
    result = extract_metric_with_candidates([zarar_chunk], metric="net_kar", quarter="Q1")
    assert result["selected"] is not None
    assert float(result["selected"]["value"]) == -123_400_000.0


def test_structured_table_reconstruction_aligns_quarter_column() -> None:
    chunk = _DummyChunk(
        text=(
            "Q1 2025 Q2 2025 Q3 2025\n"
            "donem net kari 1000000 2000000 3000000\n"
        ),
        quarter="2025Q3",
        section_title="(no heading)",
        block_type="text",
    )
    result = extract_metric_with_candidates([chunk], metric="net_kar", quarter="Q3")
    assert result["selected"] is not None
    assert float(result["selected"]["value"]) == 3_000_000.0


def test_cross_quarter_consistency_prefers_non_outlier_candidate() -> None:
    q1_chunk = _DummyChunk(text="magaza sayisi 1000", quarter="2025Q1", chunk_id="q1")
    q2_chunk = _DummyChunk(text="magaza sayisi 9000 magaza sayisi 1100", quarter="2025Q2", chunk_id="q2")
    q3_chunk = _DummyChunk(text="magaza sayisi 1200", quarter="2025Q3", chunk_id="q3")
    frame, records = aggregate_metric_across_quarters(
        quarter_chunks={
            "Q1": [q1_chunk],
            "Q2": [q2_chunk],
            "Q3": [q3_chunk],
        },
        metric="magaza_sayisi",
    )
    q2_row = frame[frame["quarter"] == "Q2"].iloc[0]
    assert float(q2_row["value"]) == 1100.0
    assert q2_row["trend_consistency_score"] is not None


def test_expected_range_sanity_rejects_outlier() -> None:
    chunk = _DummyChunk(
        text="net kar 500000000000",
        section_title="Gelir Tablosu",
        block_type="table_like",
    )
    result = extract_metric_with_candidates([chunk], metric="net_kar", quarter="Q1")
    assert result["selected"] is None
    assert result["candidates"]
    assert result["candidates"][0]["validation_reason"] in {
        "config_expected_range_max_ustunde",
        "tl_degeri_olasi_degilden_buyuk",
    }
