from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from api_module_facade import api_module
from app import cache as cache_module
from app import fund_service as fund_service_module
from app import kap_service as kap_service_module
from src import kap_vyk_client


@pytest.fixture(autouse=True)
def _reset_flow_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGFIN_CACHE_BACKEND", raising=False)
    monkeypatch.delenv("RAGFIN_REDIS_URL", raising=False)
    monkeypatch.delenv("RAGFIN_CACHE_NAMESPACE", raising=False)
    monkeypatch.setenv("RAGFIN_ADMIN_REFRESH_TOKEN", "test-admin-token")
    cache_module.reset_cache_for_tests()
    api_module._FLOW_CACHE.clear()
    api_module._WATCH_CACHE.clear()
    api_module._WATCH_GLOBAL_CACHE.clear()
    api_module._STOCKS_CACHE.clear()
    api_module._UNIVERSE_CACHE.clear()
    api_module._MARKET_PRICE_CACHE.clear()
    api_module._MARKET_PRICE_SNAPSHOT_REFRESH_EVENT.set()
    api_module._clear_market_price_snapshot_failure()
    api_module._INFOYATIRIM_STOCK_PAGE_CACHE.clear()
    api_module._STOCK_RETURN_BASE_CACHE.clear()
    api_module._MARKET_STOCK_CARD_CHART_CACHE.clear()
    api_module._STOCK_CARD_VALUATION_CACHE.clear()
    api_module._MARKET_INDICES_CACHE.clear()
    api_module._MARKET_INDEX_DETAIL_CACHE.clear()
    api_module._MARKET_INDEX_QUOTE_CACHE.clear()
    api_module._MARKET_INDEX_INTRADAY_CACHE.clear()
    api_module._MARKET_INDEX_RETURN_CACHE.clear()
    api_module._COMMODITY_CACHE.clear()
    api_module._FX_CACHE.clear()
    api_module._FX_RETURN_CACHE.clear()
    api_module._KAP_MARKET_METADATA_CACHE.clear()
    api_module._STOCK_CARD_FINANCIAL_SNAPSHOT_CACHE.clear()
    api_module._FUND_HOLDING_SECTOR_MAP_CACHE.clear()
    api_module._ISYATIRIM_BASIC_SUMMARY_CACHE.clear()
    api_module._ISYATIRIM_BASIC_SUMMARY_PERSISTED_SIGNATURES.clear()
    api_module._GEFAS_GYF_QUOTE_CACHE.clear()
    api_module._FOREIGN_HOLDING_QUOTE_CACHE.clear()
    api_module._FUND_SNAPSHOT_ROW_MAP_CACHE.clear()
    fund_service_module.reset_fund_caches_for_tests()
    kap_service_module._BIST_UNIVERSE_CACHE.clear()
    kap_vyk_client.reset_caches_for_tests()
    yield
    cache_module.reset_cache_for_tests()


def _get_ready_fund(client, path):
    deadline = time.monotonic() + 3
    while True:
        response = client.get(path)
        if response.status_code != 200 or (
            response.json().get("status") != "pending" and not response.json().get("refresh_pending")
        ):
            return response
        assert time.monotonic() < deadline, "fund response did not finish"
        time.sleep(0.005)


def _patch_holdings_cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(
        fund_service_module,
        "_holdings_path",
        lambda _processed_dir, fund_code: tmp_path / f"{fund_code}.json",
    )
    monkeypatch.setattr(
        fund_service_module,
        "_holdings_attachment_text_path",
        lambda _processed_dir, disclosure_index, obj_id: tmp_path / f"attachment_{disclosure_index}_{obj_id}.json",
    )
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", lambda _symbol: {})
    monkeypatch.setattr(api_module, "_fetch_gefas_gyf_quote", lambda _symbol: None)
    monkeypatch.setattr(
        api_module,
        "_fund_holding_sector_map",
        lambda: ({}, {"cache_hit": False, "symbol_count": 0, "source": None, "source_date": None, "warnings": []}),
    )


def _pps_foreign_holdings_text(
    *,
    xom_weight: float = 16.59,
    ovv_weight: float = 8.84,
    aem_weight: float = 6.38,
    b_weight: float = 5.33,
    aa_weight: float = 4.93,
    fcx_weight: float = 0.78,
    ewz_weight: float = 9.79,
    sqqq_weight: float = 4.23,
    pjl_weight: float = 18.74,
    ptn_weight: float = 5.74,
) -> str:
    return f"""
III-FON PORTFOY DEGERI TABLOSU
Hisse Senedi Yabanci
AA US EQUITY UNITED
STATAES
OF
AMERICA
34.500,000 75,660000 30/04/26 75,660000 89.101.014,30 1,35 {aa_weight:.2f}US0138721065
AEM US EQUITY AGNICO-
EAGLE
MINES
LIMITED
CMN
14.950,000 179,560000 30/04/26 179,560000 91.980.683,85 1,39 {aem_weight:.2f}CA0084741085
B US EQUITY BARRICK
MINING
CORP
60.000,000 42,240000 30/04/26 42,240000 86.753.088,00 1,31 {b_weight:.2f}CA06849F1080
FCX US EQUITY FREEPORT
-
MCMORAN
INC
6.000,000 65,350000 30/04/26 65,350000 13.416.798,00 0,20 {fcx_weight:.2f}US35671D8570
OVV US EQUITY OVINTIV
INC
64.000,000 55,860000 30/04/26 55,860000 122.462.784,00 1,85 {ovv_weight:.2f}US69047Q1022
XOM US EQUITY EXXON
MOBIL
CORP
47.550,000 147,190000 30/04/26 147,190000 239.585.281,35 3,62 {xom_weight:.2f}US30231G1022
V-AY ICINDE YAPILAN GIDERLER
ACIKLAMA TUTAR ORAN%
Satimlarda Odenen Komisyonlar 359.188,88 0,1776 %
IV-FON TOPLAM DEGERI TABLOSU
DIGER
Borsa Y.Fonu Yabanci
EWZ US EQUITY ISHARES MSCI BRAZIL ETF 102.875,000 36,370000 30/04/26 36,370000 128.117.496,88 1,94 {ewz_weight:.2f}US4642864007
SQQQ US EQUITY PROSHARES ULTRAPRO SHORT QQQ 32.500,000 38,320000 30/04/26 38,320000 42.665.480,00 0,65 {sqqq_weight:.2f}US74347G4322
Y.Fonu Turk
PJL PHILLIP PORTFOY PARA PIYASASI FONU 1.250.000,000 4,742700 30/04/26 4,742700 5.928.375,00 0,09 {pjl_weight:.2f}TRYPHPY00016
PTN PHILLIP PORTFOY ALTIN FONU 540.000,000 1,618300 30/04/26 1,618300 873.882,00 0,01 {ptn_weight:.2f}TRYPHPY00107
Doviz
USD FED 7.855.647,60 44,758437 30/04/26 44,969200 353.262.188,05 0,00 0,00USD 17,86USD
VI-ALIM SATIM ISLEMLERI
"""


def _overview_commentary_payload() -> Dict[str, Any]:
    return {
        "company": "BIMAS",
        "company_title": "BIM BIRLESIK MAGAZALAR A.S.",
        "latest_period": "2025/12",
        "history_context": {
            "company_kind": "generic",
            "quarters": [
                {
                    "label": "2024/3",
                    "year": 2024,
                    "period": 1,
                    "metrics": {
                        "satis_gelirleri": 150_000_000_000.0,
                        "favok": 9_800_000_000.0,
                        "net_kar": 4_100_000_000.0,
                        "faaliyet_nakit_akisi": 3_500_000_000.0,
                        "serbest_nakit_akisi": 2_100_000_000.0,
                        "ozkaynaklar": 52_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.53,
                        "net_kar_marji": 2.73,
                        "roe": 7.88,
                        "cari_oran": 1.28,
                        "net_borc_ozkaynak": 0.42,
                        "nakit_donusum": 0.51,
                    },
                },
                {
                    "label": "2024/6",
                    "year": 2024,
                    "period": 2,
                    "metrics": {
                        "satis_gelirleri": 162_000_000_000.0,
                        "favok": 10_200_000_000.0,
                        "net_kar": 4_600_000_000.0,
                        "faaliyet_nakit_akisi": 3_800_000_000.0,
                        "serbest_nakit_akisi": 2_300_000_000.0,
                        "ozkaynaklar": 54_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.3,
                        "net_kar_marji": 2.84,
                        "roe": 8.52,
                        "cari_oran": 1.3,
                        "net_borc_ozkaynak": 0.4,
                        "nakit_donusum": 0.5,
                    },
                },
                {
                    "label": "2024/9",
                    "year": 2024,
                    "period": 3,
                    "metrics": {
                        "satis_gelirleri": 171_000_000_000.0,
                        "favok": 10_900_000_000.0,
                        "net_kar": 5_000_000_000.0,
                        "faaliyet_nakit_akisi": 4_200_000_000.0,
                        "serbest_nakit_akisi": 2_500_000_000.0,
                        "ozkaynaklar": 56_500_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.37,
                        "net_kar_marji": 2.92,
                        "roe": 8.85,
                        "cari_oran": 1.31,
                        "net_borc_ozkaynak": 0.39,
                        "nakit_donusum": 0.5,
                    },
                },
                {
                    "label": "2024/12",
                    "year": 2024,
                    "period": 4,
                    "metrics": {
                        "satis_gelirleri": 180_000_000_000.0,
                        "favok": 11_500_000_000.0,
                        "net_kar": 5_300_000_000.0,
                        "faaliyet_nakit_akisi": 4_500_000_000.0,
                        "serbest_nakit_akisi": 2_700_000_000.0,
                        "ozkaynaklar": 59_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.39,
                        "net_kar_marji": 2.94,
                        "roe": 8.98,
                        "cari_oran": 1.33,
                        "net_borc_ozkaynak": 0.38,
                        "nakit_donusum": 0.51,
                    },
                },
                {
                    "label": "2025/3",
                    "year": 2025,
                    "period": 1,
                    "metrics": {
                        "satis_gelirleri": 158_000_000_000.0,
                        "favok": 10_400_000_000.0,
                        "net_kar": 4_500_000_000.0,
                        "faaliyet_nakit_akisi": 3_900_000_000.0,
                        "serbest_nakit_akisi": 2_250_000_000.0,
                        "ozkaynaklar": 63_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.58,
                        "net_kar_marji": 2.85,
                        "roe": 7.14,
                        "cari_oran": 1.36,
                        "net_borc_ozkaynak": 0.4,
                        "nakit_donusum": 0.5,
                    },
                },
                {
                    "label": "2025/6",
                    "year": 2025,
                    "period": 2,
                    "metrics": {
                        "satis_gelirleri": 168_000_000_000.0,
                        "favok": 10_950_000_000.0,
                        "net_kar": 4_900_000_000.0,
                        "faaliyet_nakit_akisi": 4_050_000_000.0,
                        "serbest_nakit_akisi": 2_350_000_000.0,
                        "ozkaynaklar": 66_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.52,
                        "net_kar_marji": 2.92,
                        "roe": 7.42,
                        "cari_oran": 1.34,
                        "net_borc_ozkaynak": 0.45,
                        "nakit_donusum": 0.48,
                    },
                },
                {
                    "label": "2025/9",
                    "year": 2025,
                    "period": 3,
                    "metrics": {
                        "satis_gelirleri": 190_000_000_000.0,
                        "favok": 11_900_000_000.0,
                        "net_kar": 5_450_000_000.0,
                        "faaliyet_nakit_akisi": 4_500_000_000.0,
                        "serbest_nakit_akisi": 2_600_000_000.0,
                        "ozkaynaklar": 69_000_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.26,
                        "net_kar_marji": 2.87,
                        "roe": 7.9,
                        "cari_oran": 1.31,
                        "net_borc_ozkaynak": 0.46,
                        "nakit_donusum": 0.48,
                    },
                },
                {
                    "label": "2025/12",
                    "year": 2025,
                    "period": 4,
                    "metrics": {
                        "satis_gelirleri": 210_000_000_000.0,
                        "favok": 13_700_000_000.0,
                        "net_kar": 6_100_000_000.0,
                        "faaliyet_nakit_akisi": 4_900_000_000.0,
                        "serbest_nakit_akisi": 3_050_000_000.0,
                        "ozkaynaklar": 72_500_000_000.0,
                    },
                    "ratios": {
                        "favok_marji": 6.52,
                        "net_kar_marji": 2.9,
                        "roe": 8.41,
                        "cari_oran": 1.29,
                        "net_borc_ozkaynak": 0.53,
                        "nakit_donusum": 0.5,
                    },
                },
            ],
        },
        "overview_payload": {
            "income_summary": [
                {
                    "key": "satis_gelirleri",
                    "label": "Satislar",
                    "current_period": "2025/12",
                    "current_value": 721_060_000_000.0,
                    "current_display": "721,06 Milyar TL",
                    "base_period": "2024/12",
                    "base_value": 680_070_000_000.0,
                    "base_display": "680,07 Milyar TL",
                    "pct_change": 6.03,
                    "pct_display": "% 6",
                }
            ],
            "balance_summary": [
                {
                    "key": "net_borc",
                    "label": "Net Borc",
                    "current_period": "2025/12",
                    "current_value": 38_570_000_000.0,
                    "current_display": "38,57 Milyar TL",
                    "base_period": "2025/9",
                    "base_value": 31_120_000_000.0,
                    "base_display": "31,12 Milyar TL",
                    "pct_change": 23.94,
                    "pct_display": "% 24",
                }
            ],
            "charts": [
                {
                    "title": "Ceyreklik Satislar",
                    "kind": "bar",
                    "series": [
                        {"label": "2025/9", "value": 190_000_000_000.0, "display": "190,00 Milyar TL"},
                        {"label": "2025/12", "value": 210_000_000_000.0, "display": "210,00 Milyar TL"},
                    ],
                }
            ],
        },
    }


def _flow_item(
    *,
    idx: str,
    category: str,
    published_at: str,
    source: str = "Finansal Rapor",
    symbol: str = "BIMAS",
    id_prefix: str = "vyk",
) -> Dict[str, Any]:
    return {
        "id": f"{id_prefix}-{idx}",
        "source": source,
        "symbol": symbol,
        "stock_codes": [symbol],
        "title": f"Bildirim {idx}",
        "subject": f"Konu {idx}",
        "published_at": published_at,
        "category": category,
        "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{idx}",
    }


def _stub_flow_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    public: Optional[List[Dict[str, Any]]] = None,
    local: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[int]]:
    """Replace both upstream flow sources with deterministic stubs."""
    calls: Dict[str, List[int]] = {"public": [0], "local": [0]}

    def fake_public(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        calls["public"][0] += 1
        return list(public or [])

    def fake_local() -> List[Dict[str, Any]]:
        calls["local"][0] += 1
        return list(local or [])

    monkeypatch.setattr(api_module, "_fetch_kap_public_disclosures", fake_public)
    monkeypatch.setattr(api_module, "_local_flow_items_from_cache", fake_local)
    return calls


def _watch_quote(
    *,
    price: float,
    prev_close: float,
    currency: str = "TRY",
    as_of: str = "2026-04-20T09:30:00+00:00",
) -> Dict[str, Any]:
    change = round(price - prev_close, 4)
    change_pct = round((change / prev_close) * 100, 2)
    return {
        "ok": True,
        "price": price,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "currency": currency,
        "market_state": "REGULAR",
        "as_of": as_of,
    }


def _fx_payload_for_watch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "items": items,
        "source": "yahoo_finance_chart",
        "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
        "as_of": "2026-04-20T09:30:00+00:00",
    }


def _commodity_payload_for_watch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "items": items,
        "source": "yahoo_finance_chart",
        "delay_note": "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).",
        "as_of": "2026-04-20T09:30:00+00:00",
    }


def _iso_hours_ago(hours: float) -> str:
    dt = datetime.now() - timedelta(hours=hours)
    # VYK `time` format: "dd.mm.YYYY HH:MM:SS"
    return dt.strftime("%d.%m.%Y %H:%M:%S")


def _stub_vyk_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    last_index: int,
    disclosures: List[Dict[str, Any]],
    details: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """Replace kap_vyk_client helpers so we can drive _fetch_kap_vyk_feed."""
    calls: Dict[str, List[Any]] = {"last": [0], "list": [], "detail": [], "members": [0]}

    def fake_last(_cfg: Any) -> int:
        calls["last"][0] += 1
        return last_index

    def fake_list(_cfg: Any, *, start_index: int, **_kwargs: Any) -> List[Dict[str, Any]]:
        calls["list"].append(start_index)
        return list(disclosures)

    def fake_detail(_cfg: Any, disclosure_index: Any, *, file_type: str = "data") -> Optional[Dict[str, Any]]:
        idx = str(disclosure_index or "").strip()
        calls["detail"].append(idx)
        return details.get(idx)

    def fake_members(_cfg: Any) -> Dict[str, Dict[str, Any]]:
        calls["members"][0] += 1
        return {}

    monkeypatch.setattr(
        "src.kap_vyk_client.get_last_disclosure_index", fake_last, raising=False
    )
    monkeypatch.setattr(
        "src.kap_vyk_client.list_disclosures_batch", fake_list, raising=False
    )
    monkeypatch.setattr(
        "src.kap_vyk_client.get_disclosure_detail", fake_detail, raising=False
    )
    monkeypatch.setattr(
        "src.kap_vyk_client.build_company_lookup", fake_members, raising=False
    )
    monkeypatch.setattr("src.kap_vyk_client.is_enabled", lambda _cfg: True, raising=False)
    return calls


def _list_row(idx: int, disclosure_class: str = "ODA", disclosure_type: str = "ODA") -> Dict[str, Any]:
    return {
        "disclosureIndex": str(idx),
        "disclosureClass": disclosure_class,
        "disclosureType": disclosure_type,
        "title": "Örnek Şirket A.Ş.",
        "companyId": "1",
    }


def _detail(
    idx: int,
    *,
    hours_ago: float,
    subject_tr: str = "Örnek Konu",
    stock: str = "BIMAS",
    disclosure_class: str = "ODA",
    disclosure_type: str = "ODA",
) -> Dict[str, Any]:
    return {
        "disclosureIndex": str(idx),
        "disclosureClass": disclosure_class,
        "disclosureType": disclosure_type,
        "senderExchCodes": [stock],
        "senderTitle": "Örnek Şirket A.Ş.",
        "subject": {"tr": subject_tr, "en": "Sample"},
        "summary": {"tr": subject_tr, "en": "Sample"},
        "time": _iso_hours_ago(hours_ago),
    }
