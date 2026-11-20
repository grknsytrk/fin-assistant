from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api_module_facade import api_module
from app import cache as cache_module
from app import fund_service as fund_service_module
from app.api import app
from api_test_helpers import (
    _get_ready_fund,
    _patch_holdings_cache_path,
    _pps_foreign_holdings_text,
    _reset_flow_state,
)


def test_api_fund_holdings_unavailable_when_kap_fund_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(fund_service_module, "_kap_search_fund_metadata", lambda _fund_code: None)
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/YAC/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["positions"] == []
    assert payload["source_metadata"]["parse_status"] == "unavailable"


def test_api_fund_holdings_parses_kap_reports_and_deltas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    latest_text = """
FON PORTFÖY DEĞERİ TABLOSU
Hisse Senedi
DSTKF Destek Finans Faktoring A.S. 1.000,00 10,00 10.000,00 23,60TRYTRADSTKFXXX
TERA Tera Yatirim Menkul Degerler A.S. 2.000,00 5,00 20.000,00 12,86TRYTRATERAXXX
BIMAS Bim Birlesik Magazalar A.S. 100,00 5,00 500,00 1,25TRYTRABIMASXXX
"""
    previous_text = """
FON PORTFÖY DEĞERİ TABLOSU
Hisse Senedi
DSTKF Destek Finans Faktoring A.S. 1.000,00 10,00 10.000,00 20,45TRYTRADSTKFXXX
TERA Tera Yatirim Menkul Degerler A.S. 2.000,00 5,00 20.000,00 24,31TRYTRATERAXXX
AKBNK Akbank T.A.S. 300,00 2,00 600,00 3,00TRYTRAAKBNKXXX
"""

    def fake_detail(disclosure_index: int) -> Dict[str, Any]:
        if disclosure_index == 200:
            return {
                "disclosure": {"disclosureBasic": {"disclosureIndex": 200, "year": 2026, "donem": 4}},
                "attachments": [{"objId": "latest", "fileName": "TST_2026.04.pdf", "fileExtension": "pdf"}],
            }
        return {
            "disclosure": {"disclosureBasic": {"disclosureIndex": 100, "year": 2026, "donem": 3}},
            "attachments": [{"objId": "previous", "fileName": "TST_2026.03.pdf", "fileExtension": "pdf"}],
        }

    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "TST", "fund_oid": "fund-oid", "fund_name": "Test Fon"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "01.05.2026 10:00:00"}},
            {"disclosureBasic": {"disclosureIndex": 100, "publishDate": "01.04.2026 10:00:00"}},
        ],
    )
    monkeypatch.setattr(fund_service_module, "_kap_fetch_report_detail", fake_detail)
    monkeypatch.setattr(fund_service_module, "_kap_download_attachment", lambda obj_id: str(obj_id).encode("utf-8"))
    monkeypatch.setattr(
        fund_service_module,
        "_extract_kap_pdf_text",
        lambda data: latest_text if data == b"latest" else previous_text,
    )
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/TST/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    positions = {item["asset_code"]: item for item in payload["positions"]}
    assert positions["DSTKF"]["weight"] == 23.6
    assert positions["DSTKF"]["previous_weight"] == 20.45
    assert positions["DSTKF"]["weight_change"] == pytest.approx(3.15)
    assert positions["DSTKF"]["change_status"] == "increased"
    assert positions["TERA"]["change_status"] == "decreased"
    assert positions["BIMAS"]["change_status"] == "new"
    assert positions["AKBNK"]["change_status"] == "removed"
    assert payload["source_metadata"]["latest_report"]["file_name"] == "TST_2026.04.pdf"


def test_api_fund_holdings_adds_optional_sector_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 23.6,
                "previous_weight": None,
                "weight_change": None,
                "change_status": "unchanged",
                "amount": None,
                "market_value": None,
                "report_date": "2026-04-30",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.9,
            },
            {
                "fund_code": "TLY",
                "asset_code": "BIMAS",
                "asset_name": "Bilinmeyen Hisse",
                "asset_type": "local_equity",
                "weight": 1.0,
                "previous_weight": None,
                "weight_change": None,
                "change_status": "unchanged",
                "amount": None,
                "market_value": None,
                "report_date": "2026-04-30",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.9,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-20T11:59:30+00:00",
            "as_of": "2026-04-30",
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "latest_report": {"disclosure_index": 1601574, "report_date": "2026-04-30"},
            "previous_report": {"disclosure_index": 1583104, "report_date": "2026-03-31"},
            "disclosure_check": {"checked_at": "2026-05-20T11:59:30+00:00", "ttl_seconds": 60},
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        api_module,
        "_fund_holding_sector_map",
        lambda: (
            {"DSTKF": {"sector_code": "XFINK", "sector_label": "Finansal Kiralama Faktoring"}},
            {"cache_hit": False, "symbol_count": 1, "source": "test", "source_date": "2026-04-30", "warnings": []},
        ),
    )
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/TLY/holdings")

    assert response.status_code == 200
    positions = {item["asset_code"]: item for item in response.json()["positions"]}
    assert positions["DSTKF"]["sector_code"] == "XFINK"
    assert positions["DSTKF"]["sector_label"] == "Finansal Kiralama Faktoring"

    direct_payload = dict(cache_payload)
    direct_payload["positions"] = [cache_payload["positions"][1]]
    direct = api_module._enrich_fund_holdings_with_daily_market_data(direct_payload, "TLY")
    assert direct["positions"][0]["sector_code"] is None
    assert direct["positions"][0]["sector_label"] is None


def test_api_fund_holdings_reuses_monthly_report_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 23.63,
                "previous_weight": 20.43,
                "weight_change": 3.2,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": False,
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "latest_report": {"disclosure_index": 1601574, "report_date": "2026-04-30", "source_url": "https://www.kap.org.tr/tr/Bildirim/1601574"},
            "previous_report": {"disclosure_index": 1583104, "report_date": "2026-03-31", "source_url": "https://www.kap.org.tr/tr/Bildirim/1583104"},
            "disclosure_check": {"checked_at": "2026-05-19T00:00:00+00:00", "ttl_seconds": 21600},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 19, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "refresh_fund_holdings",
        lambda *_args, **_kwargs: pytest.fail("monthly holdings cache should be reused"),
    )
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/TLY/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["positions"][0]["asset_code"] == "DSTKF"
    assert payload["source_metadata"]["cache_hit"] is True
    assert payload["source_metadata"]["cache_policy"] == "monthly_report"


def test_fund_holdings_disclosure_check_skips_pdf_when_report_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    positions = [
        {
            "fund_code": "TLY",
            "asset_code": "DSTKF",
            "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
            "asset_type": "local_equity",
            "weight": 23.63,
            "previous_weight": 20.43,
            "weight_change": 3.2,
            "change_status": "increased",
            "report_date": "2026-04-30",
            "previous_report_date": "2026-03-31",
            "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
            "source_type": "kap_pdf",
            "parse_confidence": 0.82,
        }
    ]
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": positions,
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "latest_report": {"disclosure_index": 1601574, "report_date": "2026-04-30"},
            "previous_report": {"disclosure_index": 1583104, "report_date": "2026-03-31"},
            "disclosure_check": {"checked_at": "2026-05-01T00:00:00+00:00", "ttl_seconds": 21600},
            "positions_hash": fund_service_module._holdings_positions_hash(positions),
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "TLY", "fund_oid": "fund-oid", "fund_name": "TLY"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 1601574, "publishDate": "06.05.2026 09:00:00"}},
            {"disclosureBasic": {"disclosureIndex": 1583104, "publishDate": "05.04.2026 09:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_fetch_report_detail",
        lambda _disclosure_index: pytest.fail("unchanged disclosure should not fetch attachment detail"),
    )

    payload = fund_service_module.get_fund_holdings_payload(tmp_path, "TLY")

    assert payload["positions"][0]["asset_code"] == "DSTKF"
    assert payload["source_metadata"]["cache_hit"] is True
    assert payload["source_metadata"]["static_cache_hit"] is True
    assert payload["source_metadata"]["disclosure_check"]["latest_disclosure_index"] == 1601574


def test_fund_holdings_picks_up_new_disclosure_index_after_short_check_ttl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cached_positions = [
        {
            "fund_code": "TLY",
            "asset_code": "DSTKF",
            "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
            "asset_type": "local_equity",
            "weight": 20.0,
            "report_date": "2026-08-31",
            "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1657116",
            "source_type": "kap_pdf",
            "parse_confidence": 0.9,
        }
    ]
    cached_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": cached_positions,
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-09-02T16:56:15+03:00",
            "as_of": "2026-08-31",
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "latest_report": {
                "disclosure_index": 1657116,
                "report_date": "2026-08-31",
                "source_url": "https://www.kap.org.tr/tr/Bildirim/1657116",
            },
            "previous_report": {
                "disclosure_index": 1642306,
                "report_date": "2026-07-31",
                "source_url": "https://www.kap.org.tr/tr/Bildirim/1642306",
            },
            "disclosure_check": {
                "checked_at": "2026-09-09T15:44:56+03:00",
                "ttl_seconds": 60,
            },
            "positions_hash": fund_service_module._holdings_positions_hash(cached_positions),
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cached_payload), encoding="utf-8")
    monkeypatch.setattr(
        fund_service_module,
        "_utc_now",
        lambda: datetime(2026, 9, 9, 15, 16, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "TLY", "fund_oid": "fund-oid", "fund_name": "TLY"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 1661170, "publishDate": "09.09.2026 18:00:07"}},
            {"disclosureBasic": {"disclosureIndex": 1657116, "publishDate": "02.09.2026 11:02:16"}},
        ],
    )
    detail_calls: List[int] = []

    def fake_detail(disclosure_index: int) -> Dict[str, Any]:
        detail_calls.append(disclosure_index)
        return {"disclosureIndex": disclosure_index}

    def fake_positions(_processed_dir: Any, detail: Dict[str, Any], *, fund_code: str) -> Any:
        disclosure_index = int(detail["disclosureIndex"])
        return [
            {
                "fund_code": fund_code,
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 25.0 if disclosure_index == 1661170 else 20.0,
                "report_date": "2026-08-31",
                "source_report_url": f"https://www.kap.org.tr/tr/Bildirim/{disclosure_index}",
                "source_type": "kap_pdf",
                "parse_confidence": 0.9,
            }
        ], {
            "disclosure_index": disclosure_index,
            "file_name": "TLY_2026.08.pdf",
            "report_date": "2026-08-31",
            "source_url": f"https://www.kap.org.tr/tr/Bildirim/{disclosure_index}",
        }

    monkeypatch.setattr(fund_service_module, "_kap_fetch_report_detail", fake_detail)
    monkeypatch.setattr(fund_service_module, "_kap_report_positions", fake_positions)

    payload = fund_service_module.get_fund_holdings_payload(tmp_path, "TLY")

    assert detail_calls == [1661170, 1657116]
    assert payload["positions"][0]["weight"] == 25.0
    assert payload["source_metadata"]["latest_report"]["disclosure_index"] == 1661170
    assert payload["source_metadata"]["disclosure_check"]["latest_disclosure_index"] == 1661170
    assert payload["source_metadata"]["cache_hit"] is False


def test_fund_holdings_parser_version_reuses_cached_attachment_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    (tmp_path / "TLY.json").write_text(
        json.dumps(
            {
                "fund_code": "TLY",
                "status": "ok",
                "positions": [],
                "source": "kap_portfolio_allocation_report",
                "message": None,
                "source_metadata": {
                    "fetched_at": "2026-05-01T00:00:00+00:00",
                    "parse_status": "ok",
                    "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION - 1,
                    "latest_report": {"disclosure_index": 1601574, "report_date": "2026-04-30"},
                },
            }
        ),
        encoding="utf-8",
    )
    text_cache_path = tmp_path / "attachment_text.json"
    text_cache_path.write_text(
        json.dumps(
            {
                "schema_version": fund_service_module.KAP_HOLDINGS_ATTACHMENT_TEXT_CACHE_VERSION,
                "disclosure_index": 1601574,
                "obj_id": "pdf-obj",
                "file_name": "TLY_2026.04.pdf",
                "fetched_at": "2026-05-06T09:00:00+00:00",
                "text": """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
AKBNK AKBANK T.A.Ş. 18.653.248,00 69,718033 30/04/26 73,200000 1.365.417.753,60 4,66 3,67TL 80100511 3,67TRAAKBNK91N6
""",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "TLY", "fund_oid": "fund-oid", "fund_name": "TLY"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 1601574, "publishDate": "06.05.2026 09:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_fetch_report_detail",
        lambda _disclosure_index: {
            "disclosureBasic": {"disclosureIndex": 1601574, "year": 2026, "donem": 4},
            "attachments": [{"fileExtension": "pdf", "objId": "pdf-obj", "fileName": "TLY_2026.04.pdf"}],
        },
    )
    monkeypatch.setattr(
        fund_service_module,
        "_holdings_attachment_text_path",
        lambda _processed_dir, _disclosure_index, _obj_id: text_cache_path,
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_download_attachment",
        lambda _obj_id: pytest.fail("cached attachment text should avoid PDF download"),
    )

    payload = fund_service_module.get_fund_holdings_payload(tmp_path, "TLY")

    assert payload["source_metadata"]["latest_report"]["attachment_text_cache_hit"] is True
    assert payload["positions"][0]["asset_code"] == "AKBNK"
    assert payload["source_metadata"]["parser_version"] == fund_service_module.KAP_HOLDINGS_PARSE_VERSION


def test_api_fund_holdings_enriches_daily_market_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "TLY",
                "asset_code": "PKZ",
                "asset_name": "PUSULA PORTFÖY KUZEY HİSSE SENEDİ SERBEST FON",
                "asset_type": "fund",
                "weight": 10.0,
                "previous_weight": 8.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "TLY",
                "asset_code": "MISS",
                "asset_name": "Eksik Hisse",
                "asset_type": "local_equity",
                "weight": 5.0,
                "previous_weight": 4.0,
                "weight_change": 1.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "TLY",
                "asset_code": "OLD",
                "asset_name": "Çıkan Hisse",
                "asset_type": "local_equity",
                "weight": 0.0,
                "previous_weight": 2.0,
                "weight_change": -2.0,
                "change_status": "removed",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": False,
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:59:30+00:00", "ttl_seconds": 60},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))

    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols, **_kwargs: {
            "DSTKF": {"price": 40.0, "currency": "TRY", "change_pct": 2.5, "as_of": "2026-05-20T10:00:00+00:00"},
            "OLD": {"price": 9.0, "currency": "TRY", "change_pct": 10.0, "as_of": "2026-05-20T10:00:00+00:00"},
        },
    )
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {
            "rows": [
                {"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0},
                {"fund_code": "PKZ", "as_of": "2026-05-20", "price": 12.34, "daily_return": 1.2},
            ]
        },
    )
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/TLY/holdings")

    assert response.status_code == 200
    payload = response.json()
    positions = {item["asset_code"]: item for item in payload["positions"]}
    assert positions["DSTKF"]["price"] == 40.0
    assert positions["DSTKF"]["return_pct"] == 2.5
    assert positions["DSTKF"]["estimated_exposure_value"] == 20_000_000.0
    assert positions["DSTKF"]["estimated_pnl_value"] == 500_000.0
    assert positions["DSTKF"]["estimated_fund_return_contribution_pct"] == 0.5
    assert positions["PKZ"]["price"] == 12.34
    assert positions["PKZ"]["return_pct"] == 1.2
    assert positions["PKZ"]["estimated_pnl_value"] == 120_000.0
    assert positions["OLD"]["estimated_fund_return_contribution_pct"] is None
    assert payload["portfolio_effect"]["estimated_return_pct"] == pytest.approx(0.62)
    assert payload["portfolio_effect"]["estimated_pnl_value"] == 620_000.0
    assert payload["portfolio_effect"]["priced_weight"] == 30.0
    assert payload["portfolio_effect"]["missing_weight"] == 5.0


def test_api_fund_holdings_enriches_foreign_assets_with_yahoo_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"GTZ": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(api_module, "_quote_map_for_holding_stocks", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))

    def fake_yahoo_quote(symbol: str) -> Dict[str, Any]:
        if symbol == "SIVR":
            return {
                "ok": True,
                "price": 71.5,
                "currency": "USD",
                "change_pct": 1.4,
                "as_of": "2026-05-20T10:00:00+00:00",
                "long_name": "abrdn Physical Silver Shares ETF",
            }
        return {"ok": False, "error": "not_found"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_yahoo_quote)
    payload = {
        "fund_code": "GTZ",
        "status": "ok",
        "positions": [
            {
                "asset_code": "SIVRUS",
                "asset_name": "ABERDEEN STANDARD PHYSICAL SILVER SHARES ETF",
                "asset_type": "foreign_fund",
                "asset_region": "foreign",
                "provider_symbol": "SIVR",
                "weight": 10.0,
                "previous_weight": 8.0,
            },
            {
                "asset_code": "MISSUS",
                "asset_name": "Çözülemeyen Yabancı Hisse",
                "asset_type": "foreign_equity",
                "asset_region": "foreign",
                "provider_symbol": "MISS",
                "weight": 5.0,
                "previous_weight": 4.0,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "GTZ")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["SIVRUS"]["price"] == 71.5
    assert positions["SIVRUS"]["price_currency"] == "USD"
    assert positions["SIVRUS"]["return_pct"] == 1.4
    assert positions["SIVRUS"]["return_source"] == "yahoo_finance_chart"
    assert positions["SIVRUS"]["provider_name"] == "abrdn Physical Silver Shares ETF"
    assert positions["SIVRUS"]["asset_name"] == "abrdn Physical Silver Shares ETF"
    assert positions["SIVRUS"]["tefas_tradable"] is False
    assert positions["SIVRUS"]["detail_clickable"] is False
    assert positions["MISSUS"]["return_pct"] is None
    assert positions["MISSUS"]["return_source"] is None
    assert positions["MISSUS"]["detail_clickable"] is False
    assert enriched["portfolio_effect"]["estimated_return_pct"] == pytest.approx(0.14)
    assert enriched["portfolio_effect"]["priced_weight"] == 10.0
    assert enriched["portfolio_effect"]["missing_weight"] == 5.0
    enrichment_meta = enriched["source_metadata"]["daily_market_enrichment"]
    assert enrichment_meta["foreign_quote_count"] == 1
    assert enrichment_meta["foreign_quote_missing_count"] == 1


def test_api_fund_holdings_marks_inner_funds_as_tefas_tradable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Inner fund-type positions must expose ``tefas_tradable`` so the
    frontend can decide whether to make the row clickable."""
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "IIE",
        "status": "ok",
        "positions": [
            {
                "fund_code": "IIE",
                "asset_code": "PKZ",
                "asset_name": "PUSULA PORTFÖY KUZEY HİSSE SENEDİ SERBEST FON",
                "asset_type": "fund",
                "weight": 5.0,
                "previous_weight": 4.0,
                "weight_change": 1.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "IIE",
                "asset_code": "ITH",
                "asset_name": "İSTANBUL PORTFÖY YÖNETİMİ A.Ş. ONE LIFE VENTURES",
                "asset_type": "fund",
                "weight": 2.5,
                "previous_weight": 2.85,
                "weight_change": -0.35,
                "change_status": "decreased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
            {
                "fund_code": "IIE",
                "asset_code": "BSOKE",
                "asset_name": "BATISÖKE SÖKE ÇİMENTO SANAYİİ T.A.Ş.",
                "asset_type": "local_equity",
                "weight": 71.0,
                "previous_weight": 58.2,
                "weight_change": 12.8,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": False,
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:59:30+00:00", "ttl_seconds": 60},
            "warnings": [],
        },
    }
    (tmp_path / "IIE.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda symbols, **_kwargs: {})
    # PKZ is in the TEFAS-open snapshot, ITH is not.
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {
            "rows": [
                {"fund_code": "IIE", "as_of": "2026-05-20", "aum": 100_000_000.0},
                {
                    "fund_code": "PKZ",
                    "as_of": "2026-05-20",
                    "price": 12.34,
                    "daily_return": 1.2,
                    "founder_company": "PUSULA PORTFÖY",
                },
            ]
        },
    )
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/IIE/holdings")

    assert response.status_code == 200
    payload = response.json()
    positions = {item["asset_code"]: item for item in payload["positions"]}
    # PKZ is in the TEFAS funds snapshot → tradable.
    assert positions["PKZ"]["tefas_tradable"] is True
    assert positions["PKZ"]["provider_name"] == "PUSULA PORTFÖY"
    assert positions["PKZ"]["logo_symbol"] == "PUSULA PORTFÖY"
    # ITH is not in the snapshot → not tradable.
    assert positions["ITH"]["tefas_tradable"] is False
    # Stock positions must not carry the flag.
    assert positions["BSOKE"]["tefas_tradable"] is None


def test_api_fund_holdings_keeps_leveraged_weights_in_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {
            "GOOD": {"price": 20.0, "currency": "TRY", "change_pct": 2.5, "as_of": "2026-05-20T10:00:00+00:00"},
            "LEVR": {"price": 10.0, "currency": "TRY", "change_pct": 10.0, "as_of": "2026-05-20T10:00:00+00:00"},
        },
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "LEVR", "asset_name": "Kaldıraçlı Pozisyon", "asset_type": "local_equity", "weight": 127.5, "previous_weight": 45.2},
            {"asset_code": "GOOD", "asset_name": "Geçerli", "asset_type": "local_equity", "weight": 20.0, "previous_weight": 18.0},
            {"asset_code": "MISS", "asset_name": "Eksik", "asset_type": "local_equity", "weight": 5.0, "previous_weight": 4.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert enriched["status"] == "ok"
    assert positions["LEVR"]["weight"] == 127.5
    assert positions["LEVR"]["weight_quality"] == "ok"
    assert positions["LEVR"]["estimated_fund_return_contribution_pct"] == pytest.approx(12.75)
    assert positions["GOOD"]["estimated_fund_return_contribution_pct"] == 0.5
    assert enriched["portfolio_effect"]["estimated_return_pct"] == pytest.approx(13.25)
    assert enriched["portfolio_effect"]["priced_weight"] == 147.5
    assert enriched["portfolio_effect"]["missing_weight"] == 5.0
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "gross_exposure"
    assert quality["raw_total_weight"] == 152.5
    assert quality["adjusted_total_weight"] == 152.5


def test_api_fund_holdings_normalizes_high_confidence_basis_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {
            "AAA": {"price": 10.0, "currency": "TRY", "change_pct": 1.0, "as_of": "2026-05-20T10:00:00+00:00"},
            "BBB": {"price": 20.0, "currency": "TRY", "change_pct": 2.0, "as_of": "2026-05-20T10:00:00+00:00"},
            "CCC": {"price": 30.0, "currency": "TRY", "change_pct": 3.0, "as_of": "2026-05-20T10:00:00+00:00"},
        },
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "AAA", "asset_name": "AAA", "asset_type": "local_equity", "weight": 4000.0, "previous_weight": 3500.0},
            {"asset_code": "BBB", "asset_name": "BBB", "asset_type": "local_equity", "weight": 3500.0, "previous_weight": 3500.0},
            {"asset_code": "CCC", "asset_name": "CCC", "asset_type": "local_equity", "weight": 2500.0, "previous_weight": 3000.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert enriched["status"] == "ok"
    assert positions["AAA"]["weight"] == 40.0
    assert positions["AAA"]["raw_weight"] == 4000.0
    assert positions["AAA"]["weight_quality"] == "normalized"
    assert positions["AAA"]["weight_change"] == 5.0
    assert enriched["portfolio_effect"]["priced_weight"] == 100.0
    assert enriched["portfolio_effect"]["missing_weight"] == 0.0
    assert enriched["portfolio_effect"]["estimated_return_pct"] == pytest.approx(1.85)
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "ok"
    assert quality["normalized_position_count"] == 3
    assert quality["normalization"]["action"] == "basis_points_to_percent"


def test_api_fund_holdings_normalizes_fractional_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {},
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "AAA", "asset_name": "AAA", "asset_type": "local_equity", "weight": 0.40, "previous_weight": 0.35},
            {"asset_code": "BBB", "asset_name": "BBB", "asset_type": "local_equity", "weight": 0.35, "previous_weight": 0.35},
            {"asset_code": "CCC", "asset_name": "CCC", "asset_type": "local_equity", "weight": 0.25, "previous_weight": 0.30},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["AAA"]["weight"] == 40.0
    assert positions["AAA"]["raw_weight"] == 0.40
    assert positions["AAA"]["weight_quality"] == "normalized"
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "ok"
    assert quality["normalized_position_count"] == 3
    assert quality["normalization"]["action"] == "fraction_to_percent"
    assert quality["adjusted_total_weight"] == pytest.approx(100.0)


def test_api_fund_holdings_normalizes_per_position_basis_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {},
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    # Most rows are normal percents, but a single row leaks basis points.
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "GOOD1", "asset_name": "GOOD1", "asset_type": "local_equity", "weight": 30.0, "previous_weight": 30.0},
            {"asset_code": "GOOD2", "asset_name": "GOOD2", "asset_type": "local_equity", "weight": 30.0, "previous_weight": 28.0},
            {"asset_code": "BPLEAK", "asset_name": "BPLEAK", "asset_type": "local_equity", "weight": 4000.0, "previous_weight": 3500.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["GOOD1"]["weight"] == 30.0
    assert positions["GOOD1"]["weight_quality"] == "ok"
    assert positions["BPLEAK"]["weight"] == 40.0
    assert positions["BPLEAK"]["raw_weight"] == 4000.0
    assert positions["BPLEAK"]["weight_quality"] == "normalized"
    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["normalization"]["action"] == "none"
    assert quality["normalized_position_count"] == 1
    assert quality["adjusted_total_weight"] == pytest.approx(100.0)
    assert quality["status"] == "ok"


def test_api_fund_holdings_hides_astronomical_contract_number_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(api_module, "_quote_map_for_holding_stocks", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 80100517.0,
                "previous_weight": 20.45,
                "weight_change": 80100496.55,
            },
            {
                "asset_code": "GOOD",
                "asset_name": "Geçerli",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["DSTKF"]["weight"] is None
    assert positions["DSTKF"]["weight_quality"] == "invalid"
    assert positions["DSTKF"]["raw_weight"] == 80100517.0
    assert positions["DSTKF"]["weight_change"] is None
    assert "sözleşme" in positions["DSTKF"]["weight_warning"]
    assert positions["GOOD"]["weight"] == 20.0
    assert enriched["source_metadata"]["holdings_quality"]["invalid_position_count"] == 1
    assert enriched["source_metadata"]["holdings_quality"]["adjusted_total_weight"] == pytest.approx(20.0)


def test_api_fund_holdings_recomputes_delta_for_fallback_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(api_module, "_quote_map_for_holding_stocks", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 6.90,
                "weight_quality": "fallback",
                "weight_warning": "KAP satırındaki son FTD yüzdesi okunamadı.",
                "previous_weight": 20.45,
                "previous_weight_quality": "ok",
                "weight_change": 80100496.55,
            },
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    position = enriched["positions"][0]
    assert position["weight"] == pytest.approx(6.90)
    assert position["previous_weight"] == pytest.approx(20.45)
    assert position["weight_change"] == pytest.approx(-13.55)
    assert position["weight_quality"] == "fallback"
    assert position["weight_warning"]
    assert enriched["source_metadata"]["holdings_quality"]["fallback_position_count"] == 1


def test_api_fund_holdings_flags_gross_exposure_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": False, "row_count": 1}),
    )
    monkeypatch.setattr(
        api_module,
        "_quote_map_for_holding_stocks",
        lambda _symbols: {},
    )
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", lambda: ({}, {"symbol_count": 0}))
    payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {"asset_code": "AAA", "asset_name": "AAA", "asset_type": "local_equity", "weight": 71.0, "previous_weight": 60.0},
            {"asset_code": "BBB", "asset_name": "BBB", "asset_type": "local_equity", "weight": 66.0, "previous_weight": 50.0},
            {"asset_code": "CCC", "asset_name": "CCC", "asset_type": "local_equity", "weight": 20.0, "previous_weight": 30.0},
        ],
        "source": "kap_portfolio_allocation_report",
        "source_metadata": {"source": "kap_portfolio_allocation_report", "as_of": "2026-04-30", "warnings": []},
    }

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(payload, "TLY")

    quality = enriched["source_metadata"]["holdings_quality"]
    assert quality["status"] == "gross_exposure"
    assert quality["adjusted_total_weight"] == pytest.approx(157.0)
    assert quality["normalization"]["action"] == "none"
    # Raw weights remain reported to the UI.
    positions = {item["asset_code"]: item for item in enriched["positions"]}
    assert positions["AAA"]["weight"] == 71.0
    assert positions["BBB"]["weight"] == 66.0


def test_api_fund_holdings_enriches_tpkgy_from_gefas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "TPKGYF",
                "asset_name": "TERA PORTFÖY KONUT ALFA KATILIM GAYRİMENKUL YATIRIM FONU",
                "asset_type": "fund",
                "weight": 10.0,
                "previous_weight": 8.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "amount": None,
                "market_value": None,
                "price": None,
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": None,
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "stale": False,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:59:30+00:00", "ttl_seconds": 60},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {"rows": [{"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0}]},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_gefas_gyf_quote",
        lambda symbol: {
            "price": 8043.164427,
            "currency": "TRY",
            "change_pct": 0.21,
            "as_of": "2026-05-18",
            "source": "gefas_gyf",
        }
        if api_module._gefas_gyf_config(symbol)
        else None,
    )
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/TLY/holdings")

    assert response.status_code == 200
    payload = response.json()
    position = payload["positions"][0]
    assert position["price"] == 8043.164427
    assert position["return_pct"] == 0.21
    assert position["return_source"] == "gefas_gyf"
    assert position["return_as_of"] == "2026-05-18"
    assert position["estimated_exposure_value"] == 10_000_000.0
    assert position["estimated_pnl_value"] == 21_000.0
    assert position["estimated_fund_return_contribution_pct"] == pytest.approx(0.021)
    assert payload["portfolio_effect"]["estimated_return_pct"] == pytest.approx(0.021)
    assert payload["portfolio_effect"]["estimated_pnl_value"] == 21_000.0
    assert payload["portfolio_effect"]["priced_weight"] == 10.0
    assert payload["portfolio_effect"]["missing_weight"] == 0.0
    assert payload["source_metadata"]["daily_market_enrichment"]["gefas_gyf_quote_count"] == 1


def test_api_fund_holdings_refreshes_quotes_after_short_response_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:59:30+00:00", "ttl_seconds": 60},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {"rows": [{"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0}]},
    )
    calls = {"count": 0}

    def fake_price_map(_symbols: List[str], **_kwargs: Any) -> Dict[str, Dict[str, Any]]:
        calls["count"] += 1
        price = 40.0 + calls["count"]
        return {"DSTKF": {"price": price, "currency": "TRY", "change_pct": 1.0, "as_of": "2026-05-20T10:00:00+00:00"}}

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_price_map)
    client = TestClient(app)

    first = _get_ready_fund(client, "/funds/TLY/holdings").json()
    cached = _get_ready_fund(client, "/funds/TLY/holdings").json()
    assert cached["positions"][0]["price"] == first["positions"][0]["price"]
    assert calls["count"] == 1
    cache_module.get_cache().delete(
        f"api:fund-holdings:TLY:response:v{api_module._FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION}"
    )
    second = _get_ready_fund(client, "/funds/TLY/holdings").json()

    assert first["positions"][0]["price"] == 41.0
    assert second["positions"][0]["price"] == 42.0
    assert calls["count"] == 2


def test_api_fund_holdings_live_returns_small_market_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    cache_payload = {
        "fund_code": "TLY",
        "status": "ok",
        "positions": [
            {
                "fund_code": "TLY",
                "asset_code": "DSTKF",
                "asset_name": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "asset_type": "local_equity",
                "weight": 20.0,
                "previous_weight": 18.0,
                "weight_change": 2.0,
                "change_status": "increased",
                "report_date": "2026-04-30",
                "previous_report_date": "2026-03-31",
                "source_report_url": "https://www.kap.org.tr/tr/Bildirim/1601574",
                "source_type": "kap_pdf",
                "parse_confidence": 0.82,
            }
        ],
        "source": "kap_portfolio_allocation_report",
        "message": None,
        "source_metadata": {
            "source": "kap_portfolio_allocation_report",
            "fetched_at": "2026-05-01T00:00:00+00:00",
            "as_of": "2026-04-30",
            "cache_hit": True,
            "parse_status": "ok",
            "parser_version": fund_service_module.KAP_HOLDINGS_PARSE_VERSION,
            "disclosure_check": {"checked_at": "2026-05-20T11:59:30+00:00", "ttl_seconds": 60},
            "warnings": [],
        },
    }
    (tmp_path / "TLY.json").write_text(json.dumps(cache_payload), encoding="utf-8")
    monkeypatch.setattr(fund_service_module, "_utc_now", lambda: datetime(2026, 5, 20, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(
        fund_service_module,
        "refresh_fund_holdings",
        lambda *_args, **_kwargs: pytest.fail("live holdings should reuse static KAP cache"),
    )
    monkeypatch.setattr(
        fund_service_module,
        "load_funds_snapshot",
        lambda _processed_dir: {"rows": [{"fund_code": "TLY", "as_of": "2026-05-20", "aum": 100_000_000.0}]},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols, **_kwargs: {
            "DSTKF": {"price": 40.0, "currency": "TRY", "change_pct": 2.5, "as_of": "2026-05-20T10:00:00+00:00"}
        },
    )
    client = TestClient(app)

    response = client.get("/funds/TLY/holdings/live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "daily_market_enrichment"
    assert payload["positions"] == [
        {
            "asset_code": "DSTKF",
            "price": 40.0,
            "price_currency": "TRY",
            "return_pct": 2.5,
            "return_source": "infoyatirim_live_quote",
            "return_as_of": "2026-05-20T10:00:00+00:00",
            "estimated_exposure_value": 20_000_000.0,
            "estimated_pnl_value": 500_000.0,
            "estimated_fund_return_contribution_pct": 0.5,
        }
    ]
    assert payload["portfolio_effect"]["estimated_return_pct"] == 0.5
    assert payload["source_metadata"]["static_cache_hit"] is True


def test_kap_holdings_parser_handles_name_first_portfolio_rows() -> None:
    text = """
III- FON PORTFÖY DEĞERİ 
TABLOSU
Vade İhraçcı Nominal Değeri Rayiç Değeri %
bold bold bold
Isin Kodu
:Ğ) YABANCI HİSSE 
SENETLERİ
APPLE US 24,247.00 398,638,321.90 2.90US0378331005AAPL.O
Analog Devices Inc 21,812.00 423,197,258.90 3.08US0326541051ADI
BROADCOM Ltd 49,845.00 866,515,530.52 6.31US11135F1012A VGO.O
Alnylam Pharmaceuticals Inc. - 
Common Stock
9,972.00 123,272,844.69 0.90US02043Q1076ALNY .O
06/10/2026 10:18:22 Sayfa 1 / 8
Taiwan Semiconductor 
Manufacturing Co.Ltd.
16,139.00 361,139,252.60 2.63US8740391003TSM
719,503.00 13,166,216,457.12 95.87TOPLAM
:S) V ADELİ 
MEVDUAT/YABANCI PARA 
01.10.2026 Yapı ve Kredi Bankası 4,760,000.00 232,915,247.05 1.70Vadeli Mevduat YP gün 1
232,915,247.05 1.70TOPLAM
:Ş) YATIRIM FONU KATILMA 
PAYI
redi Portföy Yönetimi A.Ş. Frwrd 
Birinci Girişim Sermayesi Yatırım 
Fonu
5,964,506.00 11,255,380.69 0.08YHG
YAPI KREDI PORTFOY PARA 
PIYASASI FONU
164,084,794.00 310,480,426.78 2.26YLB
YAPI KREDI PORTFOY PARA 
PIYASASI FONU
370,008.00 679,999.96 0.01YLB
333,926,038.44 2.43TOPLAM
:AF) SERTIFIKALAR
TOPLAM
IV- FON TOPLAM DEĞERİ TABLOSU
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="YAY",
        report_date="2026-09-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert set(by_code) == {"AAPL", "ADI", "AVGO", "ALNY", "TSM", "YHG", "YLB"}
    assert by_code["AAPL"]["asset_type"] == "foreign_equity"
    assert by_code["AAPL"]["asset_name"] == "APPLE US"
    assert by_code["AAPL"]["isin"] == "US0378331005"
    assert by_code["AAPL"]["provider_symbol"] == "AAPL"
    assert by_code["AAPL"]["weight"] == pytest.approx(2.90)
    assert by_code["AVGO"]["weight"] == pytest.approx(6.31)
    assert by_code["AVGO"]["isin"] == "US11135F1012"
    assert by_code["AVGO"]["provider_symbol"] == "AVGO"
    assert by_code["ALNY"]["asset_name"] == "Alnylam Pharmaceuticals Inc. - Common Stock"
    assert by_code["ALNY"]["provider_symbol"] == "ALNY"
    assert by_code["TSM"]["asset_name"] == "Taiwan Semiconductor Manufacturing Co.Ltd."
    assert by_code["TSM"]["weight"] == pytest.approx(2.63)
    assert by_code["YHG"]["asset_type"] == "fund"
    assert by_code["YHG"]["asset_name"].startswith("redi Portföy Yönetimi A.Ş. Frwrd")
    assert by_code["YHG"]["weight"] == pytest.approx(0.08)
    assert by_code["YLB"]["weight"] == pytest.approx(2.27)
    assert by_code["YLB"]["amount"] == pytest.approx(164_454_802.0)
    assert "PAYI" not in by_code
    assert all(position["asset_type"] != "cash_fx" for position in positions)


def test_kap_holdings_parser_keeps_only_equities_and_funds() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
TRHOL Tera Finansal Yatirimlar Holding A.S. 1.000,00 10,00 10.000,00 5,98TRYTRATACYO91Q7
TEHOL Tera Yatirim Teknoloji Holding A.S. 1.000,00 10,00 10.000,00 2,46TRYTRAGLOBL91Q0
BORÇLANMA SENETLERİ
TRFTRYBE2614 Tera Yatirim Bankasi A.S. 05/10/26 154 0,00 3.540.000,00 100,248100 29/04/26 46,423707 102,968512 3.645.085,09 94,42 2,87TL 813652091 2,90TRFTRYBE2614
T.REPO
AC2 - (BORSA DISI) PARDUS PORTFÖY YÖNETİMİ A.Ş. 04/05/26 0 43,00 6.530.630,99 43,000000 30/04/26 1.828.523,76 6.530.630,99 43,000000 6.530.630,99 19,40 5,14TL 5,20TRYA1PY00081
MEVDUAT
TÜRKİYE VAKIFLAR BANKASI T.A.O. 04/05/26 0 40,25 2.019.927,00 30/04/26 2.028.837,29 40,250000 2.028.837,29 33,49 1,60TL 1,61
V-AY İÇİNDE YAPILAN GİDERLER
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert set(by_code) == {"TRHOL", "TEHOL"}
    assert by_code["TRHOL"]["asset_type"] == "local_equity"
    assert by_code["TEHOL"]["asset_type"] == "local_equity"
    assert "AC2" not in by_code
    assert "TÜRK" not in by_code
    assert "TÜRKİYE" not in by_code


def test_kap_holdings_parser_keeps_new_ipo_symbols_and_numeric_company_names() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
KARCL KARDEMİR
ÇELİK
SANAYİ A.Ş.
35.000.000,00 111,756316 31/08/26 151,500000 5.302.500.000,00 7,26 6,35TL 80100511 7,09TREKDMR00027
CITAS ÇİTLEKÇİ
MAĞAZACILIK GIDA A.Ş.
8.319.732,00 106,155658 26/08/26 128,000000 1.064.925.696,00 1,45 1,27TL 80100511 1,42TRECTLK00027
BINHO 1000
YATIRIMLAR HOLDİNG A.Ş.
12.154.577,00 10,153135 26/08/26 9,740000 118.385.579,98 0,16 0,14TL 80100511 0,16TREBINH00024
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="THF",
        report_date="2026-08-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["KARCL"]["asset_type"] == "local_equity"
    assert by_code["KARCL"]["weight"] == pytest.approx(7.09)
    assert by_code["CITAS"]["asset_type"] == "local_equity"
    assert by_code["CITAS"]["weight"] == pytest.approx(1.42)
    assert by_code["BINHO"]["asset_type"] == "local_equity"
    assert by_code["BINHO"]["weight"] == pytest.approx(0.16)


def test_kap_holdings_normalizer_keeps_unlisted_local_equity(tmp_path: Any) -> None:
    positions = [
        {
            "fund_code": "THF",
            "asset_code": "KARCL",
            "asset_name": "KARDEMİR ÇELİK SANAYİ A.Ş.",
            "asset_type": "local_equity",
            "weight": 7.09,
        },
        {
            "fund_code": "THF",
            "asset_code": "KALAN",
            "asset_name": "GÜN VADE TARİHİ İHRAÇCI KURUMMENKUL KIYMET NET DÖNÜŞ TUTARI DÖVİZ CİNSİ ISIN KODU KARCL",
            "asset_type": "local_equity",
            "weight": 1.6,
        },
    ]

    normalized = fund_service_module._normalize_holding_positions_for_response(
        tmp_path,
        positions,
        fund_code="THF",
    )

    assert normalized[0]["asset_code"] == "KARCL"
    assert normalized[0]["asset_type"] == "local_equity"
    assert normalized[0]["asset_name"] == "KARDEMİR ÇELİK SANAYİ A.Ş."
    assert all(row["asset_code"] != "KALAN" for row in normalized)


def test_kap_holdings_parser_ignores_table_headers_and_splits_funds() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
TOPLAM
(FPD
GÖRE)
GRUP
(%)TOPLAM DEĞERGÜNLÜK BR
DEĞER
REPO TEMİNAT
TUTARI
DÖVİZ
CİNSİ
BORSA
SÖZLEŞM
E NO
ISIN KODU
HİSSE SENETLERİ
Hisse Türk
ORANI
VADEYE
SAYISI
AKBNK AKBANK
T.A.Ş.
18.653.248,00 69,718033 30/04/26 73,200000 1.365.417.753,60 4,66 3,67TL 80100511 3,67TRAAKBNK91N6
T.REPO
AC2 - (BORSA DISI) PARDUS PORTFÖY YÖNETİMİ A.Ş. 04/05/26 0 43,00 6.530.630,99 43,000000 30/04/26 1.828.523,76 6.530.630,99 43,000000 6.530.630,99 19,40 5,14TL 5,20TRYA1PY00081
Y.Fonu Türk
PKZ-PUSULA
PORTFÖY KUZEY
HİSSE SENEDİ
SERBEST FON (TL)
(HİSSE SENEDİ
YOĞUN FON)
PUSULA
PORTFÖY
YÖNETİMİ
A.Ş.
296.567.139,00 7,259739 24/04/26 8,213626 2.435.891.563,64 37,90 6,55TL 6,55TRYPSLP00093
PNU - PUSULA
PORTFÖY İKİNCİ
PARA PİYASASI (TL)
FONU
PUSULA
PORTFÖY
YÖNETİMİ
A.Ş.
1.937.627.250,00 1,042159 29/04/26 1,065972 2.065.456.394,94 32,14 5,56TL 5,55TRYPSLP00168
TRT131027T36 HAZİNE 0 10.000,00 1,00 10.000,00 0,68TL 0,68TRT131027T36
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="PHE",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["AKBNK"]["asset_type"] == "local_equity"
    assert by_code["AKBNK"]["asset_name"] == "AKBANK T.A.Ş."
    assert by_code["PKZ"]["asset_type"] == "fund"
    assert by_code["PNU"]["asset_type"] == "fund"
    assert "CİNSİ" not in by_code
    assert "ORANI" not in by_code
    assert "VADEYE" not in by_code
    assert "SAYISI" not in by_code
    assert "AC2" not in by_code
    assert "TRT131027T36" not in by_code


def test_kap_holdings_parser_discards_page_break_name_fragments() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DEPOSU
TİCARET
VE SANAYİ
A.Ş.
SELEC SELÇUK
ECZA
DEPOSU
TİCARET
VE SANAYİ
A.Ş.
646.963,00 263,176177 31/08/26 333,500000 215.762.160,50 0,29 0,26TL 80100511 0,29
TRESLEC00014
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="THF",
        report_date="2026-08-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert "DEPOSU" not in by_code
    assert by_code["SELEC"]["weight"] == pytest.approx(0.29)


def test_kap_holdings_parser_stops_before_trade_flow_and_keeps_fund_rows() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FİNANS FAKTORİNG A.Ş.
297.000,00 2.100,000000 30/04/26 2.105,000000 625.185.000,00 8,90 7,42TL 80100511 7,42TREDSTK00016
ALKLC ALTINKILIÇ GIDA VE SÜT SANAYİ TİCARET A.Ş.
100.000,00 33,120000 30/04/26 34,000000 3.400.000,00 5,68 4,74TL 80100511 4,74TREALTC00018
28.628.328,84 616.050.828,67 84,88100,00 GRUP TOPLAMI 91,28
VIOP Nakit Teminatı
VIOP Nakit Teminatı 511.675,39 511.675,39 100,00 0,07
IV-FON TOPLAM DEĞERİ TABLOSU
AÇIKLAMA TUTAR ORAN%
A-)FON PORTFÖY DEĞERİ
B-)HAZIR DEĞERLER
725.771.076,08
36.919,65
FON TOPLAM DEĞERİ 674.990.430,71 100,00 %
DİĞER
Y.Fonu Türk
IDL AKTİF PORTFÖY
PARA PİYASASI (TL)
FONU
AKTİF PORTFÖY YÖNETİM A.Ş
10.721.000,00 5,296661 30/04/26 5,348747 57.343.916,59 52,51 7,90TL 8,50TRYMKFT00190
NKL AKTİF PORTFÖY
KISA VADELİ
SERBEST(TL) FON
AKTİF PORTFÖY YÖNETİM A.Ş
16.524.000,00 3,101976 27/04/26 3,138747 51.864.655,43 47,49 7,15TL 7,68TRYMKFT00281
725.771.076,08FON PORTFÖY DEĞERİ 100,00
Nisan-2026
HRZ-AKTİF PORTFÖY BIST HALKA ARZ ŞİRKETLERİ HİSSE SENEDİ (TL) FONU
VII-PORTFÖYDEN SATIŞLAR
KIYMET VADE IŞLEM TARIHI FIYAT İŞLEM DEĞERI NOMINAL DEĞERI
A) HİSSE SENETLERİ(SATIŞLAR)
HISSE TÜRK
06/04/26 30.072,00OBAMS 8,480 255.010,56OBA MAKARNACILIK SANAYI VE TICARET A.Ş.
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="HRZ",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1605671",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert set(by_code) == {"DSTKF", "ALKLC", "IDL", "NKL"}
    assert by_code["DSTKF"]["asset_type"] == "local_equity"
    assert by_code["IDL"]["asset_type"] == "fund"
    assert by_code["IDL"]["weight"] == 8.5
    assert by_code["NKL"]["asset_type"] == "fund"
    assert by_code["NKL"]["weight"] == 7.68
    assert "VII" not in by_code
    assert "OBAMS" not in by_code


def test_kap_holdings_parser_resumes_repeated_table_after_vi_section() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FAKTORİNG A.Ş.
10.414.435,00 799,526624 30/03/26 1.882,000000 19.599.966.670,00 33,07 20,29TL 80100517 20,43TREDSTF00012
VI-A-GEÇEN AY İÇİNDE RÜÇHAN HAKKI KULLANIMI
Sirketin Unvani BDL-BDZ Tarihi Nominal Değeriİşlem Tipi TUTAR
V-AY İÇİNDE YAPILAN GİDERLER
AÇIKLAMA TUTAR ORAN%
IV-FON TOPLAM DEĞERİ TABLOSU
A-)FON PORTFÖY DEĞERİ
96.581.119.176,39
FON PORTFÖY DEĞERİ TABLOSU
TOPLAM
(FPD
GÖRE)
GRUP
(%)TOPLAM DEĞERGÜNLÜK BR
DEĞER
ISIN KODU
DİĞER
Y.Fonu Türk
TPKGY TERA
Portföy A.Ş
49.559,00 117.828,44326
6
23/03/26 175.774,06000
0
8.711.186.639,54 82,52 9,02TL 9,08TRYTALP00036
TPKGY TERA
Portföy A.Ş
01/04/26 1 8,00 174.000,00000
0
30/03/26 175.774,06000
0
1.406.192,48 0,01 0,00TL 0,00TRYTALP00036
96.581.119.176,39FON PORTFÖY DEĞERİ 100,00
VII-PORTFÖYDEN SATIŞLAR
A) HİSSE SENETLERİ(SATIŞLAR)
HISSE TÜRK
06/04/26 30.072,00OBAMS 8,480 255.010,56OBA MAKARNACILIK SANAYI VE TICARET A.Ş.
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TLY",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1583104",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["DSTKF"]["asset_type"] == "local_equity"
    assert by_code["TPKGY"]["asset_type"] == "fund"
    assert by_code["TPKGY"]["weight"] == pytest.approx(9.08)
    assert "OBAMS" not in by_code


def test_kap_holdings_parser_uses_safe_fallback_when_borsa_number_replaces_ftd() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FİNANS FAKTORİNG A.Ş.
18.340,00 1.817,671842 28/04/26 2.730,000000 50.068.200,00 8,12 6,90TL 80100517TREDSTF00012
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url=None,
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["DSTKF"]["weight"] == pytest.approx(6.90)
    assert by_code["DSTKF"]["weight"] < 1000
    assert by_code["DSTKF"]["weight_quality"] == "fallback"
    assert "yaklaşık" in by_code["DSTKF"]["weight_warning"]


def test_kap_holdings_parser_prefers_real_ftd_after_separated_borsa_number() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
DSTKF DESTEK FİNANS FAKTORİNG A.Ş.
18.340,00 1.817,671842 28/04/26 2.730,000000 50.068.200,00 8,12 6,90TL 80_100_5 7,42TREDSTF00012
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url=None,
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["DSTKF"]["weight"] == pytest.approx(7.42)
    assert by_code["DSTKF"]["weight_quality"] == "ok"
    assert by_code["DSTKF"]["weight_warning"] is None


def test_kap_holdings_parser_drops_orphan_page_header_before_position() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
KONTR Kontrolmatik
Teknoloji
1.627.564,00 17,215647 22/12/25 8,620000 14.029.601,68 0,02 0,01TL 80100511 0,01TREKNTR00013
Mart-2026
TLY-TERA PORTFÖY BİRİNCİ SERBEST FON
TOPLAM
(FPD
GÖRE)
GRUP
(%)TOPLAM DEĞERGÜNLÜK BR
DEĞER
REPO TEMİNAT
TUTARI
DÖVİZ
CİNSİ
BORSA
SÖZLEŞM
E NO
TOPLAM
(FTD
GÖRE)
ISIN KODU
Enerji ve
Mühendislik
A.Ş.
PEKGY PEKER
GAYRİMEN
KUL
YATIRIM
ORTAKLIĞI
A.Ş
506.822.154,00 12,515982 31/03/26 14,300000 7.247.556.802,20 12,23 7,50TL 80100511 7,55TREPEGY00022
PEKGY PEKER
GAYRİMEN
KUL
YATIRIM
ORTAKLIĞI
A.Ş
389.798,00 12,518149 31/03/26 14,300000 5.574.111,40 0,01 0,01TL 80100511 0,01TREPEGY00022
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TLY",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["PEKGY"]["weight"] == 7.56
    assert by_code["PEKGY"]["asset_type"] == "local_equity"
    assert "TLY" not in by_code
    assert "ENERJI" not in by_code


def test_kap_holdings_parser_skips_collateral_lender_prefix() -> None:
    """``Tem.Ver.``/``Teminat Veren`` rows must not invent a phantom code or
    swallow the next position into their buffer."""
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
HİSSE SENETLERİ
Hisse Türk
BSOKE BATISÖKE
SÖKE
ÇİMENTO
SANAYİİ
T.A.Ş.
1.325.828,00 22,365319 31/03/26 32,400000 42.956.827,20 2,77 3,67TL 80100511 3,88TRABSOKE91F5
BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
186.952,00 4,532224 31/03/26 5,950000 1.112.325,13 0,07 0,10TL 80100511 0,10TRABTCIM91F5
Tem.Ver. BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
28.363.048,00 4,532224 31/03/26 5,950000 168.760.174,87 10,89 14,41TL 80100511TRABTCIM91F5
GRTHO GRAINTUR
K HOLDİNG
A.Ş.
3.104.500,00 113,537810 31/03/26 226,000000 701.617.000,00 45,26 59,93TL 63,31TREGRNT00029
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="IIE",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1584779",
    )

    by_code = {position["asset_code"]: position for position in positions}
    # The ``Tem.Ver.`` continuation row and the main BSOKE/GRTHO rows must
    # all surface, none of them as a phantom ``TEM.VER`` code.
    assert "TEM.VER" not in by_code
    assert "TEM" not in by_code
    assert by_code["BSOKE"]["asset_type"] == "local_equity"
    assert by_code["BSOKE"]["weight"] == 3.88
    # The collateral leg has the larger absolute weight (14.41 vs 0.10) so
    # it must be the one kept after de-duplication, NOT a negative leg.
    assert by_code["BTCIM"]["asset_type"] == "local_equity"
    assert by_code["BTCIM"]["weight"] == 14.41
    # GRTHO must not have been swallowed into the previous buffer; its main
    # equity weight must remain intact.
    assert by_code["GRTHO"]["asset_type"] == "local_equity"
    assert by_code["GRTHO"]["weight"] == 63.31


def test_kap_holdings_parser_excludes_repo_collateral_using_known_stock_ticker() -> None:
    """REPO/Mevduat collateral rows must not be promoted to ``local_equity``
    even when the leading token is a known BIST stock code, otherwise the
    negative collateral leg overwrites the real holding."""
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
HİSSE SENETLERİ
Hisse Türk
BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
28.363.048,00 4,532224 31/03/26 5,950000 168.760.174,87 10,89 14,41TL 80100511TRABTCIM91F5
REPO
BTCIM 01/04/26 0 43,00 100.117.808,22 43,000000 31/03/26 16.207.456,00 100.117.808,22 43,000000 -100.117.808,22 57,14 -8,5581313833 -9,03
BTCIM 01/04/26 0 43,00 75.088.356,16 43,000000 31/03/26 12.155.592,00 75.088.356,16 43,000000 -75.088.356,16 42,86 -6,4181313833 -6,78
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="IIE",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    # Only the genuine equity holding may surface; the REPO collateral
    # legs must stay in non-holding bucket and therefore be filtered out.
    assert "BTCIM" in by_code
    assert by_code["BTCIM"]["asset_type"] == "local_equity"
    assert by_code["BTCIM"]["weight"] == 14.41
    # No negative-weight ghost row is allowed for the same ticker.
    assert all(
        position["weight"] is None or position["weight"] >= 0
        for position in positions
    ), positions


def test_kap_holdings_parser_handles_glued_borsa_code_before_isin() -> None:
    """ISIN tail recognition must tolerate borsa/sözleşme codes glued
    between the currency token and the ISIN (e.g. ``14,41TL 80100511TRABTCIM91F5``).

    Without this tolerance, ``_kap_row_complete`` returns False for the row
    and the buffer keeps absorbing the next position's lines."""
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
Hisse Türk
BTCIM BATIÇİM
BATI
ANADOLU
ÇİMENTO
SANAYİİ
A.Ş.
28.363.048,00 4,532224 30/04/26 5,950000 168.760.174,87 10,89 14,41TL 80100511TRABTCIM91F5
BSOKE BATISÖKE
SÖKE
ÇİMENTO
SANAYİİ
T.A.Ş.
1.325.828,00 22,365319 30/04/26 32,400000 42.956.827,20 2,77 3,67TL 80100511 3,88TRABSOKE91F5
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="TST",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    # The ISIN tail with glued borsa code must still complete the BTCIM
    # row, otherwise its lines bleed into BSOKE and one of them disappears.
    assert {"BTCIM", "BSOKE"}.issubset(by_code.keys())
    assert by_code["BTCIM"]["weight"] == 14.41
    assert by_code["BSOKE"]["weight"] == 3.88


def test_kap_holdings_parser_sums_foreign_etf_lots_and_ignores_totals() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
A.PAY
GMSTR.F QNB FİNANS PORTFÖY GÜMÜŞ ETF TRYFNBK00030 0.00% 0 100,000.00 252.29 03.04.2025 0.00% 0 0 666.25 66,625,000.00 3.33% 0.44%
GMSTR.F QNB FİNANS PORTFÖY GÜMÜŞ ETF TRYFNBK00030 0.00% 0 30,000.00 252.29 03.04.2025 0.00% 0 0 666.25 19,987,500.00 1.00% 0.13%
Ana Grup Toplamı 130,000.00 86,612,500.00 4.33% 0.57%
c.YABANCI SERMAYE PİYASASI ARAÇLARI
BORSA YATIRIM FONgARI
SIVRUS ABERDEEN STANDARD INVESTMENTS US0032641088 0.00% 0 10,810.00 29.58 23.09.2024 0.00% 0 0 69.58 33,828,691.20 0.32% 0.22%
SIVRUS ABERDEEN STANDARD INVESTMENTS US0032641088 0.00% 0 15,612.00 30.27 03.10.2024 0.00% 0 0 69.58 48,856,015.45 0.46% 0.32%
Toplam 26,422.00 82,684,706.65 0.78% 0.54%
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="GTZ",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert "ANA" not in by_code
    assert by_code["GMSTR.F"]["asset_type"] == "fund"
    assert by_code["GMSTR.F"]["weight"] == pytest.approx(0.57)
    assert by_code["GMSTR.F"]["market_value"] == pytest.approx(86_612_500.0)
    assert by_code["SIVRUS"]["asset_type"] == "foreign_fund"
    assert by_code["SIVRUS"]["asset_region"] == "foreign"
    assert by_code["SIVRUS"]["provider_symbol"] == "SIVR"
    assert by_code["SIVRUS"]["detail_clickable"] is False
    assert by_code["SIVRUS"]["weight"] == pytest.approx(0.54)
    assert by_code["SIVRUS"]["market_value"] == pytest.approx(82_684_706.65)


def test_kap_holdings_parser_uses_known_foreign_etf_names_for_gtz() -> None:
    text = """
III-FON PORTFÖY DEĞERİ TABLOSU
c.YABANCI SERMAYE PİYASASI ARAÇLARI
BORSA YATIRIM FONLARI
ZSIGEUSW SWISSCANTO FONcSgEITUNG AG CH0183135992 0.00% 0 1.00 0 0 0 0 2,499,057,068.78 16.34%
SVUSASW UBS FUNc MANAGEMENT(SWITZERgAN CH0118929048 0.00% 0 1.00 0 0 0 0 2,189,418,481.25 14.34%
HUZCN HORİZONS ETFS MANAGEMENT CANAc CA37964K1012 0.00% 0 1.00 0 0 0 0 302,340,570.63 1.98%
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="GTZ",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["ZSIGEUSW"]["asset_name"] == "Swisscanto (CH) Silver ETF"
    assert by_code["ZSIGEUSW"]["provider_symbol"] == "ZSIGEU.SW"
    assert by_code["SVUSASW"]["asset_name"] == "UBS Silver ETF USD acc"
    assert by_code["SVUSASW"]["provider_symbol"] == "SVUSA.SW"
    assert by_code["HUZCN"]["asset_name"] == "Global X Silver ETF"
    assert by_code["HUZCN"]["provider_symbol"] == "HUZ.TO"


def test_kap_holdings_parser_handles_pps_foreign_equity_and_continued_sections() -> None:
    positions = fund_service_module._parse_kap_holdings_pdf_text(
        _pps_foreign_holdings_text(),
        fund_code="PPS",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert {"XOM", "OVV", "AEM", "B", "AA", "FCX", "EWZ", "SQQQ", "PJL", "PTN"}.issubset(by_code)
    assert not {"CORP", "INC", "CMN", "AMERICA", "HOLDINGS", "MINERALS", "SATIMLARDA", "USD"}.intersection(by_code)
    assert by_code["XOM"]["asset_type"] == "foreign_equity"
    assert by_code["B"]["asset_type"] == "foreign_equity"
    assert by_code["XOM"]["provider_symbol"] == "XOM"
    assert by_code["B"]["provider_symbol"] == "B"
    assert by_code["XOM"]["detail_clickable"] is False
    assert by_code["EWZ"]["asset_type"] == "foreign_fund"
    assert by_code["SQQQ"]["asset_type"] == "foreign_fund"
    assert by_code["EWZ"]["provider_symbol"] == "EWZ"
    assert by_code["PJL"]["asset_type"] == "fund"
    assert by_code["PTN"]["asset_type"] == "fund"
    assert by_code["XOM"]["weight"] == pytest.approx(16.59)
    assert by_code["EWZ"]["weight"] == pytest.approx(9.79)
    assert by_code["PJL"]["weight"] == pytest.approx(18.74)


def test_kap_holdings_parser_keeps_foreign_symbol_split_from_isin() -> None:
    text = """
III-FON PORTFOY DEGERI TABLOSU
Hisse Yabanci
QCOM
US7475251036
QUALCOMM INC
6.500,00 147,549223 24/04/26 178,445000 52.159.437,81 5,90 5,03USD 5,03US7475251036
VI-ALIM SATIM ISLEMLERI
"""

    positions = fund_service_module._parse_kap_holdings_pdf_text(
        text,
        fund_code="CPU",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )

    by_code = {position["asset_code"]: position for position in positions}
    assert by_code["QCOM"]["asset_type"] == "foreign_equity"
    assert by_code["QCOM"]["provider_symbol"] == "QCOM"
    assert by_code["QCOM"]["asset_name"] == "QUALCOMM INC"
    assert by_code["QCOM"]["weight"] == pytest.approx(5.03)
    assert "US7475251036" not in by_code


def test_kap_holdings_parser_merges_pps_previous_month_without_bogus_removed() -> None:
    latest = fund_service_module._parse_kap_holdings_pdf_text(
        _pps_foreign_holdings_text(),
        fund_code="PPS",
        report_date="2026-04-30",
        source_url="https://www.kap.org.tr/tr/Bildirim/1",
    )
    previous = fund_service_module._parse_kap_holdings_pdf_text(
        _pps_foreign_holdings_text(
            xom_weight=13.61,
            ovv_weight=6.67,
            aem_weight=5.68,
            b_weight=5.55,
            aa_weight=3.83,
            fcx_weight=0.52,
            ewz_weight=3.44,
            sqqq_weight=0.0,
            pjl_weight=21.02,
            ptn_weight=3.90,
        ),
        fund_code="PPS",
        report_date="2026-03-31",
        source_url="https://www.kap.org.tr/tr/Bildirim/2",
    )

    merged = fund_service_module._merge_holding_positions(
        latest,
        previous,
        latest_report_date="2026-04-30",
        previous_report_date="2026-03-31",
    )

    by_code = {position["asset_code"]: position for position in merged}
    assert by_code["XOM"]["previous_weight"] == pytest.approx(13.61)
    assert by_code["XOM"]["weight_change"] == pytest.approx(2.98)
    assert by_code["XOM"]["change_status"] == "increased"
    assert by_code["PJL"]["previous_weight"] == pytest.approx(21.02)
    assert by_code["PJL"]["weight_change"] == pytest.approx(-2.28)
    assert not {"CORP", "INC", "CMN", "AMERICA", "HOLDINGS", "MINERALS", "SATIMLARDA", "USD"}.intersection(by_code)


def test_kap_holdings_normalizer_corrects_fund_code_ocr_with_reference_data(tmp_path: Any) -> None:
    cache_dir = tmp_path / "funds_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "fund_code": "GTL",
                        "name": "GARANTİ PORTFÖY BİRİNCİ PARA PİYASASI (TL) FONU",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    positions = [
        {
            "fund_code": "GTZ",
            "asset_code": "GTG",
            "asset_name": "GARANTİ PORTFÖY BİRİNCİ PARA PİYASASI (TL) FONU",
            "asset_type": "fund",
            "weight": 2.73,
        }
    ]

    normalized = fund_service_module._normalize_holding_positions_for_response(
        tmp_path,
        positions,
        fund_code="GTZ",
    )

    assert normalized[0]["asset_code"] == "GTL"
    assert normalized[0]["asset_name"] == "GARANTİ PORTFÖY BİRİNCİ PARA PİYASASI (TL) FONU"


def test_api_fund_holdings_partial_when_pdf_not_parsed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_holdings_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        fund_service_module,
        "_kap_search_fund_metadata",
        lambda _fund_code: {"fund_code": "BAD", "fund_oid": "fund-oid", "fund_name": "Bad Fon"},
    )
    monkeypatch.setattr(fund_service_module, "_kap_portfolio_subject_oid", lambda _fund_oid: "subject-oid")
    monkeypatch.setattr(
        fund_service_module,
        "_kap_list_portfolio_disclosures",
        lambda _fund_oid, _subject_oid: [
            {"disclosureBasic": {"disclosureIndex": 200, "publishDate": "01.05.2026 10:00:00"}},
        ],
    )
    monkeypatch.setattr(
        fund_service_module,
        "_kap_fetch_report_detail",
        lambda _idx: {
            "disclosure": {"disclosureBasic": {"disclosureIndex": 200, "year": 2026, "donem": 4}},
            "attachments": [{"objId": "latest", "fileName": "BAD_2026.04.pdf", "fileExtension": "pdf"}],
        },
    )
    monkeypatch.setattr(fund_service_module, "_kap_download_attachment", lambda _obj_id: b"latest")
    monkeypatch.setattr(fund_service_module, "_extract_kap_pdf_text", lambda _data: "parse edilemeyen metin")
    client = TestClient(app)

    response = _get_ready_fund(client, "/funds/BAD/holdings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["positions"] == []
    assert payload["source_metadata"]["parse_status"] == "partial"


def test_gefas_http_client_pooling_parallel_metrics_and_urlopen_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import time
    import urllib.request
    from app.funds_api import holdings_enrichment as he_module

    # 1. Verify persistent httpx.Client reuse and teardown via close_shared_http_clients()
    he_module.close_gefas_http_client()
    client1, is_ephemeral1 = he_module._get_gefas_http_client()
    client2, is_ephemeral2 = he_module._get_gefas_http_client()
    assert is_ephemeral1 is False
    assert is_ephemeral2 is False
    assert client1 is client2
    assert not client1.is_closed

    fund_service_module.close_shared_http_clients()
    assert client1.is_closed
    assert he_module._GEFAS_HTTP_CLIENT is None

    # 2. Verify urllib.request.urlopen monkeypatch seam continues to work
    urlopen_urls: List[str] = []

    class _FakeUrlopenResp:
        def __enter__(self) -> "_FakeUrlopenResp":
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def read(self) -> bytes:
            return b'{"datas": [8100.5], "labels": ["18.05.2026"]}'

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=8: urlopen_urls.append(req.full_url) or _FakeUrlopenResp(),
    )
    chart_via_seam = api_module._fetch_gefas_gyf_chart("TRYTALP00036", 0)
    assert chart_via_seam["datas"] == [8100.5]
    assert len(urlopen_urls) == 1

    # 3. Verify _fetch_gefas_gyf_quote fetches metric=0 and metric=2 in parallel
    api_module._GEFAS_GYF_QUOTE_CACHE.clear()
    monkeypatch.setattr(api_module, "_shared_cache_get_dict", lambda _key: None)
    monkeypatch.setattr(api_module, "_shared_cache_set", lambda *_args, **_kwargs: None)

    lock = threading.Lock()
    active = 0
    peak_concurrency = 0

    def fake_chart(isin: str, metric: int) -> Dict[str, Any]:
        nonlocal active, peak_concurrency
        with lock:
            active += 1
            if active > peak_concurrency:
                peak_concurrency = active
        try:
            time.sleep(0.03)
            if metric == 0:
                return {"datas": [8043.16], "labels": ["18.05.2026"]}
            return {"datas": [0.25], "labels": ["18.05.2026"]}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(api_module, "_fetch_gefas_gyf_chart", fake_chart)
    quote = api_module._fetch_gefas_gyf_quote("TPKGYF")
    assert quote is not None
    assert peak_concurrency == 2
    assert quote["price"] == pytest.approx(8043.16)
    assert quote["change_pct"] == pytest.approx(0.25)


def test_enrich_fund_holdings_runs_four_legs_and_cold_caches_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import time

    api_module._FUND_HOLDING_SECTOR_MAP_CACHE.clear()
    api_module._FOREIGN_HOLDING_QUOTE_CACHE.clear()
    api_module._GEFAS_GYF_QUOTE_CACHE.clear()
    monkeypatch.setattr(api_module, "_shared_cache_get_dict", lambda _key: None)
    monkeypatch.setattr(api_module, "_shared_cache_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api_module,
        "_fund_snapshot_row_map_with_meta",
        lambda: ({"TLY": {"as_of": "2026-05-20", "aum": 100_000_000.0}}, {"cache_hit": True, "row_count": 1}),
    )

    # 1. Verify _fund_holding_sector_map cold build runs remaining sectors in parallel and preserves priority order
    sector_lock = threading.Lock()
    sector_active = 0
    sector_peak = 0
    first_priority = api_module._FUND_HOLDING_SECTOR_PRIORITY[0]
    second_priority = api_module._FUND_HOLDING_SECTOR_PRIORITY[1]

    def fake_bist_universe(sector_code: str) -> Dict[str, Any]:
        nonlocal sector_active, sector_peak
        with sector_lock:
            sector_active += 1
            if sector_active > sector_peak:
                sector_peak = sector_active
        try:
            time.sleep(0.015)
            # Both first_priority and second_priority contain OVERLAP -> first_priority must win!
            symbols = ["OVERLAP", f"SYM_{sector_code}"] if sector_code in {first_priority, second_priority} else [f"SYM_{sector_code}"]
            return {"symbols": symbols, "source": "test_bist", "source_date": "2026-05-20"}
        finally:
            with sector_lock:
                sector_active -= 1

    monkeypatch.setattr(api_module, "get_bist_index_universe", fake_bist_universe, raising=False)
    sector_map, sector_meta = api_module._fund_holding_sector_map()
    assert sector_peak >= 2
    assert sector_map["OVERLAP"]["sector_code"] == first_priority
    assert sector_meta["cache_hit"] is False

    # 2. Verify _quote_map_for_foreign_holdings fetches multiple Yahoo misses in parallel
    yahoo_lock = threading.Lock()
    yahoo_active = 0
    yahoo_peak = 0

    def fake_yahoo_quote(symbol: str) -> Dict[str, Any]:
        nonlocal yahoo_active, yahoo_peak
        with yahoo_lock:
            yahoo_active += 1
            if yahoo_active > yahoo_peak:
                yahoo_peak = yahoo_active
        try:
            time.sleep(0.025)
            return {
                "ok": True,
                "price": 100.0,
                "currency": "USD",
                "change_pct": 1.5,
                "as_of": "2026-05-20T10:00:00+00:00",
                "long_name": f"Asset {symbol}",
            }
        finally:
            with yahoo_lock:
                yahoo_active -= 1

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_yahoo_quote)
    foreign_quotes = api_module._quote_map_for_foreign_holdings(
        [
            {"asset_code": "AAPL", "asset_type": "foreign_equity", "provider_symbol": "AAPL"},
            {"asset_code": "MSFT", "asset_type": "foreign_equity", "provider_symbol": "MSFT"},
            {"asset_code": "NVDA", "asset_type": "foreign_equity", "provider_symbol": "NVDA"},
        ]
    )
    assert yahoo_peak >= 2
    assert set(foreign_quotes.keys()) == {"AAPL", "MSFT", "NVDA"}

    # 3. Verify _enrich_fund_holdings_with_daily_market_data executes all 4 legs concurrently
    leg_lock = threading.Lock()
    leg_active = 0
    leg_peak = 0

    def _enter_leg() -> None:
        nonlocal leg_active, leg_peak
        with leg_lock:
            leg_active += 1
            if leg_active > leg_peak:
                leg_peak = leg_active
        time.sleep(0.03)

    def _exit_leg() -> None:
        nonlocal leg_active
        with leg_lock:
            leg_active -= 1

    def slow_stocks(_symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        _enter_leg()
        try:
            return {"DSTKF": {"price": 40.0, "currency": "TRY", "change_pct": 2.0, "as_of": "2026-05-20"}}
        finally:
            _exit_leg()

    def slow_gefas(_code: str) -> Optional[Dict[str, Any]]:
        _enter_leg()
        try:
            return {"price": 8000.0, "currency": "TRY", "change_pct": 0.5, "as_of": "2026-05-20", "source": "gefas_gyf"}
        finally:
            _exit_leg()

    def slow_foreign(_positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        _enter_leg()
        try:
            return {"AAPL": {"price": 190.0, "currency": "USD", "change_pct": 1.0, "as_of": "2026-05-20", "source": "yahoo_finance_chart"}}
        finally:
            _exit_leg()

    def slow_sector() -> Any:
        _enter_leg()
        try:
            return ({"DSTKF": {"sector_code": "XFINK", "sector_label": "Faktoring"}}, {"cache_hit": True, "symbol_count": 1})
        finally:
            _exit_leg()

    monkeypatch.setattr(api_module, "_quote_map_for_holding_stocks", slow_stocks)
    monkeypatch.setattr(api_module, "_fetch_gefas_gyf_quote", slow_gefas)
    monkeypatch.setattr(api_module, "_quote_map_for_foreign_holdings", slow_foreign)
    monkeypatch.setattr(api_module, "_fund_holding_sector_map", slow_sector)

    enriched = api_module._enrich_fund_holdings_with_daily_market_data(
        {
            "fund_code": "TLY",
            "status": "ok",
            "positions": [
                {"asset_code": "DSTKF", "asset_name": "Destek", "asset_type": "local_equity", "weight": 20.0},
                {"asset_code": "TPKGYF", "asset_name": "Tera GYF", "asset_type": "fund", "weight": 10.0},
                {"asset_code": "AAPL", "asset_name": "Apple", "asset_type": "foreign_equity", "provider_symbol": "AAPL", "weight": 10.0},
            ],
            "source": "kap_portfolio_allocation_report",
            "source_metadata": {"as_of": "2026-04-30", "warnings": []},
        },
        "TLY",
    )
    assert leg_peak == 4
    assert enriched["portfolio_effect"]["priced_weight"] == 40.0
