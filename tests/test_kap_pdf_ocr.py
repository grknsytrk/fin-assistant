from __future__ import annotations

import io
import json

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app import fund_service
from app import kap_pdf_ocr


def _parse(text: str) -> list[dict]:
    return fund_service._parse_kap_holdings_pdf_text(
        text, fund_code="TST", report_date="2026-08-31", source_url=None,
    )


@pytest.mark.parametrize(
    "values",
    [
        "25,000,000.000 91,750,000.00 18.02 6.75",
        "25.000.000,000 91.750.000,00 18,02 6,75",
    ],
)
def test_column_table_uses_total_percentage_and_preserves_number_locale(values: str) -> None:
    rows = _parse(f"""
III-FON PORTFOY DEGERI TABLOSU
Sermaye Piyasasi Araclari Ihracci Isin Kodu Nominal Degeri Toplam Deger Grup(%) Toplam(%)
A.Pay
ALGYO ALARKO GAYRIMENKUL YATIRIM ORTAKLIGI A.S. TRAALGYO91Q5 {values}
ISCTR TURKIYE IS BANKASI A.S. TRAISCTR91N2 0.507 6.41 0.00 0.00
IV-FON TOPLAM DEGER TABLOSU
""")
    by_code = {row["asset_code"]: row for row in rows}
    assert by_code["ALGYO"]["weight"] == pytest.approx(6.75)
    assert by_code["ALGYO"]["weight_quality"] == "ok"
    assert by_code["ALGYO"]["amount"] == 25_000_000
    assert by_code["ALGYO"]["market_value"] == 91_750_000
    assert by_code["ALGYO"]["asset_name"] == "ALARKO GAYRIMENKUL YATIRIM ORTAKLIGI A.S."
    assert by_code["ISCTR"]["amount"] == pytest.approx(0.507)
    assert by_code["ISCTR"]["market_value"] == pytest.approx(6.41)


def test_column_table_keeps_foreign_and_fund_rows_without_totals_or_collateral() -> None:
    rows = _parse("""
III-FONPORTFOYDEGERITABLOSU
Ihracci Isin Kodu Nominal Degeri Toplam Deger Grup(%) Toplam(%)
A.Pay
ALGYO ALARKO TRAALGYO91Q5 25,000,000.000 91,750,000.00 18.02 6.75
A.GrupToplam 25,000,000.000 91,750,000.00 6.75 6.75
C.Turev Araclar
VIOP
VIOP F_TUPRS0926 -2,500.000 0.00 0.00 0.00
D.Yab.Ser.Piy.Araclari
NVD.US US38747R6291 325,000.000 63,018,267.81 56.78 4.64
G.DigerVarliklar
FON
NSY NEO PORTFOY YONETIMI A.S. TRYVVGS01048 39,643,132.000 56,840,005.52 71.69 4.18
FON Toplam 39,643,132.000 56,840,005.52 4.18 4.18
T.REPO
ALGYO TRAALGYO91Q5 1,000.000 5,000.00 100.00 -0.50
DVZ
DVZ USD 1,088.540 52,439.87 87.87 0.00
Fon Portfoy Degeri 1,358,442,889.38
IV-FONTOPLAMDEGERTABLOSU
A.Pay
AKBNK AKBANK TRAAKBNK91N6 1,000.000 10,000.00 100.00 99.00
""")
    by_code = {row["asset_code"]: row for row in rows}
    assert set(by_code) == {"ALGYO", "NVD.US", "NSY"}
    assert by_code["ALGYO"]["weight"] == 6.75
    assert by_code["NVD.US"]["asset_type"] == "foreign_equity"
    assert by_code["NVD.US"]["provider_symbol"] == "NVD"
    assert by_code["NSY"]["asset_type"] == "fund"
    assert by_code["NSY"]["weight"] == 4.18


def test_column_table_rejects_unreadable_weight_without_consuming_next_row() -> None:
    rows = _parse("""
III-FON PORTFOY DEGERI TABLOSU
Isin Kodu Nominal Degeri Toplam Deger Grup(%) Toplam(%)
A.Pay
ALGYO ALARKO TRAALGYO91Q5 25,000,000.000 91,750,000.00 18.02
ALKIM ALKIM KIMYA TRAALKIM91E0 3,350,000.000 51,858,000.00 10.19 3.82
""")
    assert [(row["asset_code"], row["weight"]) for row in rows] == [("ALKIM", 3.82)]


def test_native_table_keeps_wrapped_foreign_rows_and_isins_across_auxiliary_tables() -> None:
    rows = _parse("""
III-FON PORTFOY DEGERI TABLOSU
Hisse Yabanci
ADI US ANALOG
DEVICES
INC
10.919,00 381,977647 09/07/26 365,130000 192.064.720,67 1,01 1,00USD 1,00
US0326541051
AMAT US AMAT US 9.760,00 641,979383 06/07/26 460,410000 216.477.010,28 1,14 1,13USD 1,13US0382221051
BABA US ALIBABA
GROUP
HOLDING
LTD
159.621,00 133,659281 03/08/26 115,760000 890.155.257,43 4,70 4,63USD 4,64
US01609W1027
V-AY ICINDE YAPILAN GIDERLER
Mevzuat Geregi Yapilmasi Zorunlu Tescil
ve Ilan Giderleri
3.567,00 0,0000 %
IV-FON TOPLAM DEGERI TABLOSU
D-)DIGER VARLIKLAR
0,00
FON TOPLAM DEGERI 19.169.450.688,99 100,00 %
TOPLAM
(FPD GORE)
ISIN KODU
MRVL US MARVELL
TECHNOLO
GY
16.582,00 257,093434 07/07/26 213,070000 170.206.614,14 0,90 0,89USD 0,89
US5738741041
MSFT US MICROSOF
T CORP
78.366,00 432,359227 28/07/26 509,310000 1.922.768.944,59 10,15 9,98USD 10,03US5949181045
MU US MICRON
TECHNOLO
GY
38.784,00 661,343012 19/08/26 940,860000 1.757.902.643,35 9,28 9,14USD 9,17
US5951121038
NOW US SERVICEN
OW INC
82.285,00 107,679879 05/08/26 145,000000 574.785.616,21 3,03 2,99USD 3,00US81762P1021
VI-ALIM SATIM ISLEMLERI
""")
    by_code = {row["asset_code"]: row for row in rows}
    assert set(by_code) == {"ADI", "AMAT", "BABA", "MRVL", "MSFT", "MU", "NOW"}
    assert all(row["asset_type"] == "foreign_equity" for row in rows)
    assert by_code["ADI"]["isin"] == "US0326541051"
    assert by_code["AMAT"]["isin"] == "US0382221051"
    assert by_code["BABA"]["isin"] == "US01609W1027"
    assert by_code["BABA"]["asset_name"] == "ALIBABA GROUP HOLDING LTD"
    assert by_code["MSFT"]["weight"] == pytest.approx(10.03)
    assert by_code["MSFT"]["provider_symbol"] == "MSFT"
    assert by_code["MU"]["weight"] == pytest.approx(9.17)
    assert by_code["NOW"]["weight"] == pytest.approx(3.00)


@pytest.mark.parametrize("values", [
    "12.550,000    3.748.494,72    8,936493    53,860",
    "12,550.000    3,748,494.72    8.936493    53.860",
])
def test_numbered_nominal_value_table_reads_ratio_before_purchase_price(values: str) -> None:
    rows = _parse(f"""
3 - FON PORTFOY DEGERI TABLOSU
A - HISSE SENETLERI    Ihracci    Nominal    Rayic Deger    Oran (%)    Birim Alis Fiyati
1. Banka Istirak ve Hissedarlari
2. Diger Ortakliklar
3. Yabanci Hisse Senetleri
1 ADIB AED    UAE-Abu Dhabi Islami    {values}
2 BOUBYAN KWD    Kuwait-Boubyan Bank    184,000    19.055,12    0,045428    38,720
3 ADNOCDRI UH AED    ADNOC Drilling Compa    28.000,000    2.118.184,01    5,049797    62,407
4 - TOPLAM DEGERI TABLOSU
A - HISSE SENETLERI    Ihracci    Nominal    Rayic Deger    Oran (%)    Birim Alis Fiyati
3. Yabanci Hisse Senetleri
4 MSFT USD    Microsoft    100,000    1.000,00    99,000000    53,860
""")
    by_code = {row["asset_code"]: row for row in rows}
    assert set(by_code) == {"ADIB", "BOUBYAN", "ADNOCDRI"}
    assert all(row["asset_type"] == "foreign_equity" for row in rows)
    assert by_code["ADIB"]["asset_name"] == "UAE-Abu Dhabi Islami"
    assert by_code["ADIB"]["weight"] == pytest.approx(8.936493)
    assert by_code["ADIB"]["amount"] == 12_550
    assert by_code["ADIB"]["market_value"] == pytest.approx(3_748_494.72)
    assert by_code["BOUBYAN"]["amount"] == 184
    assert by_code["ADNOCDRI"]["report_currency"] == "AED"


def test_nominal_value_table_excludes_other_sections_and_keeps_funds_without_cost_column() -> None:
    rows = _parse("""
3 - FON PORTFOY DEGERI TABLOSU
A - HISSE SENETLERI    Ihracci    Nominal    Rayic Deger    Oran (%)
1 THYAO    TURK HAVA YOLLARI    100,00    1.000,00    2,345678
C - TERS REPO    Ihracci    Nominal    Rayic Deger    Oran (%)
1 THYAO    TURK HAVA YOLLARI    100,00    1.000,00    -4,000000
S - KATILIM HESABI    Ihracci    Nominal    Rayic Deger    Oran (%)
1 AKBNK    AKBANK    100,00    1.000,00    4,000000
Y - KATILMA PAYLARI    Ihracci    Nominal    Rayic Deger    Oran (%)
1 NSY    NEO PORTFOY YEDINCI SERBEST FON    100,00    1.000,00    3,456789
4 - TOPLAM DEGERI TABLOSU
""")
    by_code = {row["asset_code"]: row for row in rows}
    assert set(by_code) == {"THYAO", "NSY"}
    assert by_code["THYAO"]["weight"] == pytest.approx(2.345678)
    assert by_code["THYAO"]["asset_type"] == "local_equity"
    assert by_code["NSY"]["asset_type"] == "fund"
    assert by_code["NSY"]["weight"] == pytest.approx(3.456789)


def test_foreign_numbered_table_does_not_guess_us_identity_from_non_us_labels() -> None:
    from app.api import _foreign_holding_provider_symbol, _quote_map_for_foreign_holdings

    rows = _parse("""
3 - FON PORTFOY DEGERI TABLOSU
A - HISSE SENETLERI    Ihracci    Nominal    Rayic Deger    Oran (%)    Birim Alis Fiyati
3. Yabanci Hisse Senetleri
1 STC SAR    Saudi Arabia-Saudi T    3.318,000    1.903.874,52    4,538878    269,044
2 THYAO AED    FOREIGN ISSUER    100,000    1.000,00    1,000000    10,000
""")
    assert all(row["asset_type"] == "foreign_equity" for row in rows)
    assert all(_foreign_holding_provider_symbol(row) == "" for row in rows)
    assert _quote_map_for_foreign_holdings(rows) == {}
    assert rows[0]["provider_symbol_status"] == "unresolved"
    assert rows[0]["report_currency"] == "SAR"


def test_ocr_rebuilds_wrapped_name_in_the_same_security_row() -> None:
    def box(left: float, top: float, width: float, text: str) -> list:
        return [[[left, top], [left + width, top], [left + width, top + 20], [left, top + 20]], text, 0.99]

    result = [
        box(260, 92, 150, "ORTAKLIGI A.S."),
        box(150, 80, 80, "ALGYO"),
        box(262, 68, 360, "ALARKO GAYRIMENKUL YATIRIM"),
        box(660, 80, 190, "TRAALGYO91Q5"),
        box(900, 80, 160, "25,000,000.000"),
        box(1100, 80, 160, "91,750,000.00"),
        box(1300, 80, 60, "18.02"),
        box(1450, 80, 60, "6.75"),
    ]
    text = kap_pdf_ocr.ocr_result_text(list(reversed(result)))
    assert text.splitlines() == [
        "ALGYO\tALARKO GAYRIMENKUL YATIRIM ORTAKLIGI A.S.\tTRAALGYO91Q5\t25,000,000.000\t91,750,000.00\t18.02\t6.75"
    ]


def _pdf_with_text_and_blank_page() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 10 Tf 10 10 Td (Native portfolio report with a readable text layer on its first page.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_blank_page(width=200, height=200)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_extraction_ocr_only_missing_pages_and_keeps_native_text(monkeypatch) -> None:
    data = _pdf_with_text_and_blank_page()
    calls = []

    def ocr(pdf_data: bytes, indices: list[int]) -> dict[int, str]:
        calls.append(indices)
        assert pdf_data.startswith(b"%PDF")
        return {1: "Scanned portfolio table"}

    monkeypatch.setattr(kap_pdf_ocr, "extract_pdf_ocr_pages", ocr)
    text = fund_service._extract_kap_pdf_text(b"java-wrapper" + data)
    assert calls == [[1]]
    assert text.startswith("KAP_PDF_OCR\n")
    assert "Native portfolio report" in text
    assert "Scanned portfolio table" in text

    writer = PdfWriter()
    writer.add_page(PdfReader(io.BytesIO(data)).pages[0])
    output = io.BytesIO()
    writer.write(output)
    calls.clear()
    assert "Native portfolio report" in fund_service._extract_kap_pdf_text(output.getvalue())
    assert calls == []


@pytest.mark.parametrize("schema_version", [1, fund_service.KAP_HOLDINGS_ATTACHMENT_TEXT_CACHE_VERSION])
def test_empty_attachment_text_cache_is_recovered_and_successful_ocr_is_reused(
    monkeypatch, tmp_path, schema_version: int,
) -> None:
    path = tmp_path / "attachment.json"
    path.write_text(json.dumps({"schema_version": schema_version, "obj_id": "pdf", "text": ""}), encoding="utf-8")
    monkeypatch.setattr(fund_service, "_holdings_attachment_text_path", lambda *_: path)
    downloads = []
    monkeypatch.setattr(fund_service, "_kap_download_attachment", lambda obj: downloads.append(obj) or b"pdf")
    monkeypatch.setattr(fund_service, "_extract_kap_pdf_text", lambda _: "KAP_PDF_OCR\nRecovered table")
    attachment = {"objId": "pdf", "fileName": "monthly.pdf"}
    text, meta = fund_service._kap_attachment_text_from_cache(tmp_path, 1, attachment)
    assert text == "KAP_PDF_OCR\nRecovered table"
    assert meta["attachment_text_cache_hit"] is False
    text_again, meta_again = fund_service._kap_attachment_text_from_cache(tmp_path, 1, attachment)
    assert text_again == text
    assert meta_again["attachment_text_cache_hit"] is True
    assert downloads == ["pdf"]
