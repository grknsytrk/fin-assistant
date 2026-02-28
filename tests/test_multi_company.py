from src.ingest import parse_company_from_name, parse_year_from_name
from src.ratio_engine import detect_company_mentions, is_cross_company_query


def test_parse_company_and_year() -> None:
    assert parse_company_from_name("BIM 2025 3. Ceyrek Faaliyet Raporu") == "BIM"
    assert parse_company_from_name("MIGROS_2025_Q2_report") == "MIGROS"
    assert parse_company_from_name("NETCAD_4Q2025_gelir_tablosu") == "NETCAD"
    assert parse_company_from_name("2025Q4_NETCAD_gelir_tablosu") == "NETCAD"
    assert parse_company_from_name("NETCAD4Q2025_gelir_tablosu") == "NETCAD"
    assert parse_company_from_name("ACME2025Q1financial_results") == "ACME"
    assert parse_year_from_name("2025 3. Ceyrek Faaliyet Raporu") == 2025


def test_cross_company_detection() -> None:
    companies = ["BIM", "MIGROS", "SOK"]
    question = "BIM ve MIGROS net karini karsilastir"
    mentioned = detect_company_mentions(question, companies)
    assert "BIM" in mentioned and "MIGROS" in mentioned
    assert is_cross_company_query(question, companies) is True
