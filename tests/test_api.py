from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.api import app
from src import kap_vyk_client


@pytest.fixture(autouse=True)
def _reset_flow_state() -> None:
    api_module._FLOW_CACHE.clear()
    api_module._WATCH_CACHE.clear()
    kap_vyk_client.reset_caches_for_tests()


def test_api_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_api_feedback() -> None:
    client = TestClient(app)
    response = client.post(
        "/feedback",
        json={
            "company": "BIM",
            "quarter": "Q1",
            "metric": "net_kar",
            "extracted_value": "1,23 mlr TL",
            "user_value": "1,20 mlr TL",
            "evidence_ref": "[doc|Q1|5|gelir tablosu]",
            "verdict": "yanlis",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "feedback_saved"


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
    vyk: Optional[List[Dict[str, Any]]] = None,
    local: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[int]]:
    """Replace both upstream flow sources with deterministic stubs."""
    calls: Dict[str, List[int]] = {"vyk": [0], "local": [0]}

    def fake_vyk(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        calls["vyk"][0] += 1
        return list(vyk or [])

    def fake_local() -> List[Dict[str, Any]]:
        calls["local"][0] += 1
        return list(local or [])

    monkeypatch.setattr(api_module, "_fetch_kap_vyk_feed", fake_vyk)
    monkeypatch.setattr(api_module, "_local_flow_items_from_cache", fake_local)
    return calls


def test_market_flow_prefers_vyk_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        vyk=[
            _flow_item(
                idx="1230901",
                category="finansal_rapor",
                published_at="2026-04-19T11:00:00",
            ),
            _flow_item(
                idx="1230900",
                category="ozel_durum",
                published_at="2026-04-19T10:00:00",
                source="Özel Durum",
                symbol="ASELS",
            ),
        ],
        local=[
            _flow_item(
                idx="9",
                category="finansal_rapor",
                published_at="2026-04-18T09:00:00",
                id_prefix="local",
            )
        ],
    )

    client = TestClient(app)
    response = client.get("/market/flow", params={"limit": 10})
    assert response.status_code == 200
    payload = response.json()

    assert payload["source"] == "kap_vyk"
    assert payload["degraded_mode"] is False
    assert payload["multi_category"] is True
    assert payload.get("warning") in (None, "")

    ids = [row["id"] for row in payload["items"]]
    assert ids == ["vyk-1230901", "vyk-1230900"]

    # VYK cevap verdiginde yerel cache'e hic dokunulmamali; akis kalabalaklasmasin.
    assert calls["vyk"][0] == 1
    assert calls["local"][0] == 0


def test_market_flow_falls_back_to_local_when_vyk_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        vyk=[],
        local=[
            _flow_item(
                idx="42",
                category="finansal_rapor",
                published_at="2026-04-19T08:00:00",
                id_prefix="local",
            )
        ],
    )

    client = TestClient(app)
    response = client.get("/market/flow", params={"limit": 10})
    assert response.status_code == 200
    payload = response.json()

    assert payload["source"] == "local_cache"
    assert payload["degraded_mode"] is True
    assert payload["multi_category"] is False
    assert payload["warning"]
    assert calls["vyk"][0] == 1
    assert calls["local"][0] == 1
    assert [row["id"] for row in payload["items"]] == ["local-42"]


def test_market_flow_reuses_cache_on_repeat_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        vyk=[
            _flow_item(
                idx="1",
                category="finansal_rapor",
                published_at="2026-04-19T11:00:00",
            )
        ],
    )

    client = TestClient(app)
    for _ in range(3):
        response = client.get("/market/flow", params={"limit": 5})
        assert response.status_code == 200

    # Flow cache 180 sn, degraded_mode=False oldugu icin ikinci ve ucuncu
    # istekler upstream'e hic ulasmamali.
    assert calls["vyk"][0] == 1


def test_market_flow_scales_detail_budget_with_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Kullanici UI'dan daha fazla kayit istedikçe backend VYK'ye gonderdigi
    # detay butcesini de buyutmeli; bu sayede 'kayit sayisi' secicisi gercekten
    # feed'i genisletiyor.
    seen_budgets: List[int] = []

    def fake_vyk(**kwargs: Any) -> List[Dict[str, Any]]:
        seen_budgets.append(int(kwargs.get("detail_budget") or 0))
        return [
            _flow_item(
                idx=str(i),
                category="finansal_rapor",
                published_at=f"2026-04-19T{10 + (i % 6):02d}:00:00",
            )
            for i in range(1, 6)
        ]

    monkeypatch.setattr(api_module, "_fetch_kap_vyk_feed", fake_vyk)
    monkeypatch.setattr(api_module, "_local_flow_items_from_cache", lambda: [])

    client = TestClient(app)
    client.get("/market/flow", params={"limit": 25})
    client.get("/market/flow", params={"limit": 500})

    assert seen_budgets, "VYK feed cagrilmadi"
    # 25 ve 500 ayri cache kovalarina dustugu icin iki ayri upstream cagri olmali
    # ve ikinci cagrinin butcesi ilkinden buyuk olmali.
    assert len(seen_budgets) == 2
    assert seen_budgets[0] < seen_budgets[1]
    assert seen_budgets[1] >= 500


def test_market_flow_refresh_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        vyk=[
            _flow_item(
                idx="1",
                category="finansal_rapor",
                published_at="2026-04-19T11:00:00",
            )
        ],
    )

    client = TestClient(app)
    client.get("/market/flow", params={"limit": 5})
    client.get("/market/flow", params={"limit": 5, "refresh": "true"})

    assert calls["vyk"][0] == 2


def test_market_flow_category_filter_applies_to_vyk_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_flow_sources(
        monkeypatch,
        vyk=[
            _flow_item(idx="1", category="ozel_durum", published_at="2026-04-19T12:00:00"),
            _flow_item(idx="2", category="finansal_rapor", published_at="2026-04-19T13:00:00"),
            _flow_item(idx="3", category="genel_kurul", published_at="2026-04-19T14:00:00"),
        ],
    )

    client = TestClient(app)
    response = client.get("/market/flow", params={"limit": 10, "category": "ozel_durum"})
    assert response.status_code == 200
    payload = response.json()
    cats = {row["category"] for row in payload["items"]}
    assert cats == {"ozel_durum"}


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


def test_market_watch_success_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        if yahoo_symbol == "XU100.IS":
            return _watch_quote(price=10235.0, prev_close=10195.0)
        if yahoo_symbol == "XU030.IS":
            return _watch_quote(price=11312.0, prev_close=11300.0)
        return {"ok": False, "error": "unsupported_symbol"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(
        api_module,
        "_market_fx_payload",
        lambda: _fx_payload_for_watch(
            [
                {
                    "symbol": "USD/TRY",
                    "label": "Amerikan Doları",
                    "yahoo_symbol": "USDTRY=X",
                    "price": 38.52,
                    "prev_close": 38.40,
                    "change": 0.12,
                    "change_pct": 0.31,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "EUR/TRY",
                    "label": "Euro",
                    "yahoo_symbol": "EURTRY=X",
                    "price": 41.80,
                    "prev_close": 41.72,
                    "change": 0.08,
                    "change_pct": 0.19,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        api_module,
        "_market_commodities_payload",
        lambda: _commodity_payload_for_watch(
            [
                {
                    "symbol": "BRENT",
                    "label": "Brent Petrol",
                    "yahoo_symbol": "BZ=F",
                    "price": 86.12,
                    "prev_close": 85.70,
                    "change": 0.42,
                    "change_pct": 0.49,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "ALTIN",
                    "label": "Altın (Ons)",
                    "yahoo_symbol": "GC=F",
                    "price": 2412.5,
                    "prev_close": 2404.0,
                    "change": 8.5,
                    "change_pct": 0.35,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "GUMUS",
                    "label": "Gümüş (Ons)",
                    "yahoo_symbol": "SI=F",
                    "price": 30.2,
                    "prev_close": 30.0,
                    "change": 0.2,
                    "change_pct": 0.67,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
                {
                    "symbol": "DOGALGAZ",
                    "label": "Doğal Gaz",
                    "yahoo_symbol": "NG=F",
                    "price": 2.45,
                    "prev_close": 2.41,
                    "change": 0.04,
                    "change_pct": 1.66,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                },
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()

    assert payload["source"] == "yahoo_finance_chart"
    assert "15dk" in payload["delay_note"]

    sections = payload["sections"]
    assert [row["symbol"] for row in sections["indices"]] == ["XU100", "XU030"]
    assert [row["symbol"] for row in sections["fx"]] == ["USD/TRY", "EUR/TRY"]
    assert [row["symbol"] for row in sections["commodities"]] == ["BRENT", "ALTIN", "GUMUS", "DOGALGAZ"]


def test_market_watch_index_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_quote(yahoo_symbol: str) -> Dict[str, Any]:
        calls.append(yahoo_symbol)
        if yahoo_symbol == "XU100.IS":
            return {"ok": False, "error": "primary_failed"}
        if yahoo_symbol == "^XU100":
            return _watch_quote(price=10235.0, prev_close=10195.0)
        if yahoo_symbol == "XU030.IS":
            return _watch_quote(price=11312.0, prev_close=11300.0)
        return {"ok": False, "error": "unsupported_symbol"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", lambda: _fx_payload_for_watch([]))
    monkeypatch.setattr(api_module, "_market_commodities_payload", lambda: _commodity_payload_for_watch([]))

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()

    indices = {row["symbol"]: row for row in payload["sections"]["indices"]}
    assert indices["XU100"]["yahoo_symbol"] == "^XU100"
    assert "XU100.IS" in calls
    assert "^XU100" in calls


def test_market_watch_reuses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    counters: Dict[str, List[int]] = {
        "quote": [0],
        "fx": [0],
        "commodity": [0],
    }

    def fake_quote(_yahoo_symbol: str) -> Dict[str, Any]:
        counters["quote"][0] += 1
        return _watch_quote(price=100.0, prev_close=99.0)

    def fake_fx() -> Dict[str, Any]:
        counters["fx"][0] += 1
        return _fx_payload_for_watch([])

    def fake_commodity() -> Dict[str, Any]:
        counters["commodity"][0] += 1
        return _commodity_payload_for_watch([])

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(api_module, "_market_fx_payload", fake_fx)
    monkeypatch.setattr(api_module, "_market_commodities_payload", fake_commodity)

    client = TestClient(app)
    first = client.get("/market/watch")
    second = client.get("/market/watch")

    assert first.status_code == 200
    assert second.status_code == 200
    assert counters["quote"][0] == 2
    assert counters["fx"][0] == 1
    assert counters["commodity"][0] == 1


def test_market_watch_handles_partial_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_quote(_yahoo_symbol: str) -> Dict[str, Any]:
        return {"ok": False, "error": "provider_down"}

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)
    monkeypatch.setattr(
        api_module,
        "_market_fx_payload",
        lambda: _fx_payload_for_watch(
            [
                {
                    "symbol": "USD/TRY",
                    "label": "Amerikan Doları",
                    "yahoo_symbol": "USDTRY=X",
                    "price": 38.52,
                    "prev_close": 38.40,
                    "change": 0.12,
                    "change_pct": 0.31,
                    "currency": "TRY",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        api_module,
        "_market_commodities_payload",
        lambda: _commodity_payload_for_watch(
            [
                {
                    "symbol": "BRENT",
                    "label": "Brent Petrol",
                    "yahoo_symbol": "BZ=F",
                    "price": 86.12,
                    "prev_close": 85.70,
                    "change": 0.42,
                    "change_pct": 0.49,
                    "currency": "USD",
                    "market_state": "REGULAR",
                    "as_of": "2026-04-20T09:30:00+00:00",
                    "error": None,
                }
            ]
        ),
    )

    client = TestClient(app)
    response = client.get("/market/watch")
    assert response.status_code == 200
    payload = response.json()
    sections = payload["sections"]

    index_xu100 = next(row for row in sections["indices"] if row["symbol"] == "XU100")
    fx_eur = next(row for row in sections["fx"] if row["symbol"] == "EUR/TRY")
    commodity_altin = next(row for row in sections["commodities"] if row["symbol"] == "ALTIN")

    assert index_xu100["price"] is None
    assert index_xu100["error"]
    assert fx_eur["price"] is None
    assert fx_eur["error"] == "instrument_not_found"
    assert commodity_altin["price"] is None
    assert commodity_altin["error"] == "instrument_not_found"


# region VYK feed unit tests ------------------------------------------------


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


def test_vyk_feed_drops_items_outside_24h_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pencere icinde en az 1 kayit varsa pencere disi kalanlar atilir.
    disclosures = [_list_row(101), _list_row(100)]
    details = {
        "101": _detail(101, hours_ago=2),
        "100": _detail(100, hours_ago=48),
    }
    _stub_vyk_client(
        monkeypatch,
        last_index=101,
        disclosures=disclosures,
        details=details,
    )

    items = api_module._fetch_kap_vyk_feed()

    ids = [row["id"] for row in items]
    assert ids == ["vyk-101"]


def test_vyk_feed_falls_back_when_window_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pencere icinde hic kayit yoksa, detayi alinmis en yeni kayitlari yine
    # gosteriyoruz. Test gateway'i eski sabit veri dondurdugunde akisi bos
    # birakmamak icin bu davranis kritik.
    disclosures = [_list_row(101), _list_row(100)]
    details = {
        "101": _detail(101, hours_ago=48),
        "100": _detail(100, hours_ago=72),
    }
    _stub_vyk_client(
        monkeypatch,
        last_index=101,
        disclosures=disclosures,
        details=details,
    )

    items = api_module._fetch_kap_vyk_feed()

    ids = [row["id"] for row in items]
    assert sorted(ids) == ["vyk-100", "vyk-101"]


def test_vyk_feed_respects_detail_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    disclosures = [_list_row(i) for i in range(110, 100, -1)]
    details = {str(i): _detail(i, hours_ago=1) for i in range(101, 111)}
    calls = _stub_vyk_client(
        monkeypatch,
        last_index=110,
        disclosures=disclosures,
        details=details,
    )

    items = api_module._fetch_kap_vyk_feed(detail_budget=3, list_pages=1)

    # Budget is the hard cap on detail fetches; feed length must not exceed it.
    assert len(items) <= 3
    assert len(calls["detail"]) <= 3


def test_vyk_feed_empty_when_credentials_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Disable the client entirely and make sure no network helpers are invoked.
    monkeypatch.setattr("src.kap_vyk_client.is_enabled", lambda _cfg: False, raising=False)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("VYK upstream cagrilmamali")

    monkeypatch.setattr(
        "src.kap_vyk_client.get_last_disclosure_index", _boom, raising=False
    )
    assert api_module._fetch_kap_vyk_feed() == []


# endregion
