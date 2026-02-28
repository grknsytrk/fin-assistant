from src.query_parser import parse_query


def test_query_parser_quarters() -> None:
    assert parse_query("2025 birinci çeyrek net kar")["quarter"] == "Q1"
    assert parse_query("ilk yarıyıl FAVÖK marjı")["quarter"] == "Q2"
    assert parse_query("dokuz aylık satış trendi")["quarter"] == "Q3"
    assert parse_query("2025 dorduncu ceyrek net kar")["quarter"] == "Q4"
