from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api_module_facade import api_module
from app import kap_service as kap_service_module
from app.api import app
from src.nvidia_commentary import NvidiaCommentaryError
from api_test_helpers import (
    _detail,
    _flow_item,
    _list_row,
    _overview_commentary_payload,
    _reset_flow_state,
    _stub_flow_sources,
    _stub_vyk_client,
)


def test_kap_snapshot_response_cache_revalidates_without_public_bypass(monkeypatch):
    queued = []
    monkeypatch.setattr(api_module, "_cached_kap_snapshot_response", lambda _: None)
    monkeypatch.setattr(api_module, "_schedule_swr_revalidation", lambda **kw: queued.append(kw) or True)
    client = TestClient(app)
    first = client.get("/kap/snapshot", params={"company": "AKBNK", "max_quarters": 20})
    assert first.status_code == 200
    assert first.json()["pending"] is True
    assert len(queued) == 1
    key = api_module._kap_snapshot_response_cache_key("AKBNK", 20)
    entry = api_module._swr_cache_entry({"ok": True, "quarters": []}, fresh_ttl_seconds=300, stale_ttl_seconds=1800)
    api_module._shared_cache_set(key, entry, 1800)
    second = client.get("/kap/snapshot", params={"company": "AKBNK", "max_quarters": 20})
    assert second.json()["response_cache_hit"] is True
    assert len(queued) == 1
    refreshed = client.get("/kap/snapshot", params={"company": "AKBNK", "max_quarters": 20, "refresh": True})
    assert refreshed.json()["refresh_pending"] is True
    assert len(queued) == 2


@pytest.mark.parametrize("periods,values,expected", [
    ([(2025, 4), (2026, 1)], [1, 2], None),
    ([(2025, 2), (2025, 3), (2025, 4), (2026, 1)], [1, 2, 3, 4], 10),
    ([(2025, 1), (2025, 3), (2025, 4), (2026, 1)], [1, 2, 3, 4], None),
    ([(2025, 3), (2025, 3), (2025, 4), (2026, 1)], [1, 2, 3, 4], None),
    ([(2025, 2), (2025, 3), (2025, 4), (2026, 1)], [1, None, 3, 4], None),
    ([(2025, 2), (2025, 3), (2025, 4), (2026, 1)], [0, 0, 0, 0], 0),
    ([(2025, 2), (2025, 3), (2025, 4), (2026, 1)], [1, float("inf"), 3, 4], None),
])
def test_ttm_requires_complete_consecutive_quarterly_flows(periods, values, expected):
    quarters = [{"year": y, "period": p, "metrics_quarterly": {"net_kar": v},
                 "metrics": {"net_kar": 999}} for (y, p), v in zip(periods, values)]
    assert api_module._build_ttm_sum(quarters, "net_kar") == expected


@pytest.mark.parametrize("kind", ["bank", "insurance"])
def test_sector_ratios_cannot_be_restored_by_provider_fallback(kind):
    snapshot = {"company_kind": kind, "ttm_net_kar": 10, "ttm_favok": 5,
                "net_borc": 20, "ozkaynaklar": 50}
    result = api_module._resolve_market_card_valuation_from_cached_data({
        "financial_snapshot": snapshot, "provider_ratios": {"fd_favok": 99},
        "fallback_financial_ratios": {"net_borc_favok": 88},
    }, market_cap=100)
    assert result["fk"] == 10
    assert result["pd_dd"] == 2
    assert result["fd_favok"] is None
    assert result["net_borc_favok"] is None
    assert result["enterprise_value"] is None
    valuation = api_module._build_kap_valuation_payload(
        snapshot={"company_kind": kind, "quarters": []}, price_payload={},
        isyatirim_payload={"ok": True, "fd_favok": 12, "fd_favok_prim_iskonto_pct": 4},
    )
    assert valuation["fd_favok"] is None
    assert valuation["fd_favok_prim_iskonto_pct"] is None


def test_kap_disk_response_does_not_wait_for_slow_provider(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(api_module, "_MARKET_SWR_REVALIDATION_EXECUTOR", executor)
    calls = []
    def slow(*args, **kwargs):
        calls.append(args)
        entered.set()
        release.wait(3)
        return {"ok": True, "quarters": []}
    monkeypatch.setattr(api_module, "_cached_kap_snapshot_response", lambda _: {"ok": True, "quarters": [{"quarter": "2026Q1"}]})
    monkeypatch.setattr(api_module, "_build_kap_snapshot_response", slow)
    try:
        result = TestClient(app).get("/kap/snapshot", params={"company": "TEST", "max_quarters": 5})
        assert result.json()["response_cache_status"] == "disk_hit"
        assert entered.wait(1)
        assert not release.is_set()
        # A second depth shares the same background lease.
        second = TestClient(app).get("/kap/snapshot", params={"company": "TEST", "max_quarters": 20})
        assert second.json()["refresh_pending"] is True
        assert len(calls) == 1
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_kap_snapshot_normalization_exposes_identity_company_kind() -> None:
    payload = kap_service_module.normalize_snapshot_for_frontend(
        {
            "ok": True,
            "company": "TUPRS",
            "stock_code": "TUPRS",
            "company_title": "TÜPRAŞ-TÜRKİYE PETROL RAFİNERİLERİ A.Ş.",
            "quarters": [],
        }
    )

    assert payload["company_kind"] == "generic"


def test_kap_overview_models_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/kap/overview-models")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert isinstance(data["providers"], list)


def test_kap_overview_commentary_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        assert api_key == "dummy-nvidia-key"
        assert model == "minimaxai/minimax-m2.7"
        assert normalized_payload["company"] == "BIMAS"
        assert base_analysis["scorecard"]["overall_score"] > 0
        assert debug_trace is not None
        return (
            '{"headline":"Satis ivmesi korunuyor",'
            '"bullets":["Satislar yillik bazda artarken net borc yukselmis."],'
            '"risk_note":"Nakit akisi ve borc trendi izlenmeli.",'
            '"watch_metrics":["Net borc","FAVOK marji"],'
            '"summary":"Genel gorunum dengeli, karlilik tarafi daha destekleyici.",'
            '"seasonality_note":"Son ceyrek kendi mevsimsel bandindan belirgin kopmuyor.",'
            '"score_adjustments":{"overall_adjustment":0.3,"subscores":['
            '{"key":"buyume","adjustment":0.4,"summary":"Satis ve kar buyumesi ayni ceyrek bazina gore destekleyici."},'
            '{"key":"karlilik","adjustment":0.2,"summary":"Marjlar baz etkisine ragmen korunuyor."},'
            '{"key":"bilanco","adjustment":-0.2,"summary":"Borcluluk tarafi biraz daha dikkat gerektiriyor."},'
            '{"key":"nakit_akisi","adjustment":0.1,"summary":"Nakit donusumu zayiflamadan suruyor."}'
            ']}}'
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_AI_MODEL", raising=False)
    monkeypatch.setenv("OVERVIEW_AI_EXPOSE_DEBUG_TRACE", "1")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["headline"] == "Satis ivmesi korunuyor"
    assert payload["watch_metrics"] == ["Net borc", "FAVOK marji"]
    assert payload["model_used"] == "minimaxai/minimax-m2.7"
    assert payload["scorecard"]["score_source"] == "ai_adjusted"
    assert len(payload["scorecard"]["subscores"]) == 4
    assert any("validation:" in item for item in payload.get("debug_trace", []))


def test_kap_overview_commentary_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        assert api_key == "dummy-nvidia-key"
        assert model == "meta/llama-4-maverick-17b-128e-instruct"
        assert normalized_payload["model"] == "meta/llama-4-maverick-17b-128e-instruct"
        assert base_analysis["scorecard"]["score_source"] == "deterministic_only"
        return (
            '{"headline":"Model override calisti",'
            '"bullets":["Secilen model backend tarafina ulasti."],'
            '"risk_note":"",'
            '"watch_metrics":["Net borc"],'
            '"summary":"Model override ile AI adjustment akisi calisti.",'
            '"seasonality_note":"Mevsimsellik etkisi sinirli.",'
            '"score_adjustments":{"overall_adjustment":0.1,"subscores":['
            '{"key":"buyume","adjustment":0.1,"summary":"Buyume sinyalleri hafif olumlu."},'
            '{"key":"karlilik","adjustment":0.0,"summary":"Karlilik notu buyuk olcude korundu."},'
            '{"key":"bilanco","adjustment":0.0,"summary":"Bilanco notu sabit kaldi."},'
            '{"key":"nakit_akisi","adjustment":0.0,"summary":"Nakit akisi notu sabit kaldi."}'
            ']}}'
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.setenv("OVERVIEW_AI_EXPOSE_DEBUG_TRACE", "1")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    payload = _overview_commentary_payload()
    payload["model"] = "meta/llama-4-maverick-17b-128e-instruct"

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model_used"] == "meta/llama-4-maverick-17b-128e-instruct"
    assert body["scorecard"]["score_source"] == "ai_adjusted"


def test_kap_overview_commentary_does_not_truncate_model_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_headline = "Uzun baslik " + ("tam cumle devam ediyor " * 20) + "ve burada bitiyor."
    long_bullet = "Uzun madde " + ("kesilmeden devam eden finansal yorum " * 20) + "tamamlandi."
    long_risk = "Uzun risk notu " + ("cumle bolunmeden ilerliyor " * 20) + "tamamlandi."
    long_summary = "Uzun skor ozeti " + ("modelin gerekcesi kesilmeden aktariliyor " * 20) + "tamamlandi."
    long_subscore = "Uzun alt skor yorumu " + ("detayli nedenler kesilmeden korunuyor " * 20) + "tamamlandi."

    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        return json.dumps(
            {
                "headline": long_headline,
                "bullets": [long_bullet],
                "risk_note": long_risk,
                "watch_metrics": ["Net borc"],
                "summary": long_summary,
                "seasonality_note": long_summary,
                "score_adjustments": {
                    "overall_adjustment": 0.1,
                    "subscores": [
                        {"key": "buyume", "adjustment": 0.0, "summary": long_subscore},
                        {"key": "karlilik", "adjustment": 0.0, "summary": long_subscore},
                        {"key": "bilanco", "adjustment": 0.0, "summary": long_subscore},
                        {"key": "nakit_akisi", "adjustment": 0.0, "summary": long_subscore},
                    ],
                },
            },
            ensure_ascii=False,
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.setenv("OVERVIEW_AI_EXPOSE_DEBUG_TRACE", "1")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["headline"] == long_headline
    assert body["bullets"] == [long_bullet]
    assert body["risk_note"] == long_risk
    assert body["scorecard"]["summary"] == long_summary
    assert body["scorecard"]["seasonality_note"] == long_summary
    assert all(item["summary"] == long_subscore for item in body["scorecard"]["subscores"])


def test_kap_overview_commentary_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.delenv("EVREN_API_KEY", raising=False)
    monkeypatch.setenv("OVERVIEW_AI_EXPOSE_DEBUG_TRACE", "1")

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "NVIDIA_API_KEY" in payload["error"]
    assert payload["scorecard"]["score_source"] == "deterministic_only"
    assert any("NVIDIA_API_KEY bulunamadi" in item for item in payload.get("debug_trace", []))


def test_kap_overview_commentary_provider_error_includes_debug_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        raise NvidiaCommentaryError("simule provider hatasi")

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.setenv("OVERVIEW_AI_EXPOSE_DEBUG_TRACE", "1")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["error"] == "simule provider hatasi"
    assert payload["scorecard"]["score_source"] == "ai_failed_fallback"
    assert any("AI fallback" in item for item in payload.get("debug_trace", []))


def test_kap_overview_commentary_invalid_ai_adjustments_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(
        normalized_payload: Dict[str, Any],
        base_analysis: Dict[str, Any],
        api_key: str,
        model: str,
        *,
        debug_trace: Optional[List[str]] = None,
    ) -> str:
        return (
            '{"headline":"Eksik adjustment",'
            '"bullets":["Model yorum yazdi ama adjustment semasi eksik."],'
            '"risk_note":"",'
            '"watch_metrics":["Net borc"]}'
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia-key")
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.setenv("OVERVIEW_AI_EXPOSE_DEBUG_TRACE", "1")
    monkeypatch.setattr("src.nvidia_commentary._call_nvidia_chat", fake_call)

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=_overview_commentary_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["scorecard"]["score_source"] == "ai_failed_fallback"
    assert "score_adjustments" in payload["error"]


def test_kap_overview_commentary_rejects_large_body() -> None:
    client = TestClient(app)
    response = client.post(
        "/kap/overview-commentary",
        content=b'{"company":"' + (b"x" * (api_module.MAX_REQUEST_BYTES + 1)) + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_kap_overview_commentary_rejects_unexpected_top_level() -> None:
    payload = _overview_commentary_payload()
    payload["unexpected"] = True

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "beklenmeyen alan" in response.json()["detail"]


def test_kap_overview_commentary_requires_history_context() -> None:
    payload = _overview_commentary_payload()
    payload.pop("history_context")

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "history_context" in response.json()["detail"]


def test_kap_overview_commentary_rejects_unsupported_model() -> None:
    payload = _overview_commentary_payload()
    payload["model"] = "unsupported/model"

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "desteklenmeyen model" in response.json()["detail"]


def test_kap_overview_commentary_rejects_series_limit() -> None:
    payload = _overview_commentary_payload()
    payload["overview_payload"]["charts"][0]["series"] = [
        {"label": f"2025/{idx}", "value": float(idx), "display": str(idx)}
        for idx in range(11)
    ]

    client = TestClient(app)
    response = client.post("/kap/overview-commentary", json=payload)

    assert response.status_code == 422
    assert "en fazla 10" in response.json()["detail"]


def test_kap_overview_commentary_cancels_provider_task_on_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = False

    async def fake_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal cancelled
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            await asyncio.sleep(0)
            return True

    async def run_case() -> None:
        with pytest.raises(HTTPException) as exc_info:
            await api_module._run_overview_commentary_until_done_or_disconnected(
                DisconnectingRequest(),  # type: ignore[arg-type]
                {"company": "BIMAS"},
            )
        assert exc_info.value.status_code == 499

    monkeypatch.setattr(api_module, "generate_overview_commentary", fake_generate)
    asyncio.run(run_case())
    assert cancelled is True


def test_market_flow_prefers_public_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[
            _flow_item(
                idx="1230901",
                category="finansal_rapor",
                published_at="2026-04-19T11:00:00",
                id_prefix="public",
            ),
            _flow_item(
                idx="1230900",
                category="ozel_durum",
                published_at="2026-04-19T10:00:00",
                source="Özel Durum",
                symbol="ASELS",
                id_prefix="public",
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

    assert payload["source"] == "kap_public_website"
    assert payload["degraded_mode"] is False
    assert payload["multi_category"] is True
    assert payload.get("warning") in (None, "")

    ids = [row["id"] for row in payload["items"]]
    assert ids == ["public-1230901", "public-1230900"]

    assert calls["public"][0] == 1
    assert calls["local"][0] == 0


def test_market_flow_falls_back_to_local_when_public_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[],
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
    assert calls["public"][0] == 1
    assert calls["local"][0] == 1
    assert [row["id"] for row in payload["items"]] == ["local-42"]


def test_market_flow_reuses_cache_on_repeat_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[
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
    assert calls["public"][0] == 1


def test_market_flow_scales_detail_budget_with_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Kullanici UI'dan daha fazla kayit istedikçe backend VYK'ye gonderdigi
    # detay butcesini de buyutmeli; bu sayede 'kayit sayisi' secicisi gercekten
    # feed'i genisletiyor.
    seen_limits: List[int] = []

    def fake_public(**kwargs: Any) -> List[Dict[str, Any]]:
        seen_limits.append(int(kwargs.get("max_items") or 0))
        return [
            _flow_item(
                idx=str(i),
                category="finansal_rapor",
                published_at=f"2026-04-19T{10 + (i % 6):02d}:00:00",
            )
            for i in range(1, 6)
        ]

    monkeypatch.setattr(api_module, "_fetch_kap_public_disclosures", fake_public)
    monkeypatch.setattr(api_module, "_local_flow_items_from_cache", lambda: [])

    client = TestClient(app)
    client.get("/market/flow", params={"limit": 25})
    client.get("/market/flow", params={"limit": 500})

    assert seen_limits, "Public feed cagrilmadi"
    assert len(seen_limits) == 2
    assert seen_limits[0] < seen_limits[1]
    assert seen_limits[1] >= 500


def test_market_flow_refresh_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_flow_sources(
        monkeypatch,
        public=[
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

    assert calls["public"][0] == 2


def test_market_flow_category_filter_applies_to_public_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_flow_sources(
        monkeypatch,
        public=[
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


def test_market_flow_empty_persisted_category_does_not_return_unfiltered_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_flow_sources(
        monkeypatch,
        public=[
            _flow_item(idx="1", category="finansal_rapor", published_at="2026-04-19T12:00:00"),
        ],
    )
    monkeypatch.setattr(api_module._kap_flow_store, "store_is_configured", lambda: True)

    response = TestClient(app).get("/market/flow", params={"limit": 10, "category": "kar_payi"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == []
    assert payload["has_more"] is False
    assert payload["next_cursor"] is None


def test_market_flow_rejects_two_pagination_directions() -> None:
    response = TestClient(app).get(
        "/market/flow",
        params={"before": "cursor-a", "after": "cursor-b"},
    )

    assert response.status_code == 400


def test_admin_kap_refresh_requires_token_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    refresh_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        api_module,
        "_refresh_kap_flow_store",
        lambda **kwargs: refresh_calls.append(kwargs) or {"status": "ok", "stored_count": 2},
    )
    client = TestClient(app)

    unauthorized = client.post("/admin/kap/refresh")
    authorized = client.post(
        "/admin/kap/refresh?fast=1",
        headers={"Authorization": "Bearer test-admin-token"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["stored_count"] == 2
    assert refresh_calls == [{"fast": True}]


def test_kap_flow_head_schedules_only_one_stale_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: List[Any] = []

    class FakeExecutor:
        def submit(self, function: Any) -> None:
            submitted.append(function)

    monkeypatch.setattr(api_module._kap_flow_store, "store_is_configured", lambda: True)
    monkeypatch.setattr(api_module, "_KAP_FLOW_REFRESH_EXECUTOR", FakeExecutor())
    monkeypatch.setattr(
        api_module._kap_flow_store,
        "read_flow_head",
        lambda **_kwargs: {
            "latest_cursor": "cursor-1",
            "as_of": "2026-09-09T10:00:00+00:00",
            "last_successful_refresh": None,
            "refresh_status": "unknown",
        },
    )
    api_module._kap_flow_store.update_flow_status(
        refresh_status="ok",
        refresh_pending=False,
        last_successful_refresh=(datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat(),
    )

    client = TestClient(app)
    first = client.get("/market/flow/head")
    second = client.get("/market/flow/head")

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(submitted) == 1
    assert first.json()["latest_cursor"] == "cursor-1"
    assert first.json()["refresh_pending"] is True
    assert "güncelleniyor" in first.json()["warning"]


def test_kap_flow_head_schedules_live_refresh_before_stale_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: List[Any] = []
    refresh_calls: List[Dict[str, Any]] = []

    class FakeExecutor:
        def submit(self, function: Any) -> None:
            submitted.append(function)

    monkeypatch.setattr(api_module._kap_flow_store, "store_is_configured", lambda: True)
    monkeypatch.setattr(api_module, "_KAP_FLOW_REFRESH_EXECUTOR", FakeExecutor())
    monkeypatch.setattr(
        api_module,
        "_refresh_kap_flow_store",
        lambda **kwargs: refresh_calls.append(kwargs),
    )
    monkeypatch.setattr(
        api_module._kap_flow_store,
        "read_flow_head",
        lambda **_kwargs: {
            "latest_cursor": "cursor-live",
            "as_of": "2026-09-09T10:00:00+00:00",
            "last_successful_refresh": None,
            "refresh_status": "unknown",
        },
    )
    api_module._kap_flow_store.update_flow_status(
        refresh_status="running",
        refresh_pending=True,
        last_successful_refresh=(datetime.now(timezone.utc) - timedelta(seconds=45)).isoformat(),
    )

    response = TestClient(app).get("/market/flow/head")

    assert response.status_code == 200
    assert len(submitted) == 1
    assert response.json()["latest_cursor"] == "cursor-live"
    assert response.json()["refresh_pending"] is True
    submitted[0]()
    assert refresh_calls == [{"fast": True}]


def test_kap_flow_head_does_not_refresh_within_live_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: List[Any] = []

    class FakeExecutor:
        def submit(self, function: Any) -> None:
            submitted.append(function)

    monkeypatch.setattr(api_module._kap_flow_store, "store_is_configured", lambda: True)
    monkeypatch.setattr(api_module, "_KAP_FLOW_REFRESH_EXECUTOR", FakeExecutor())
    api_module._kap_flow_store.update_flow_status(
        refresh_status="ok",
        refresh_pending=False,
        last_successful_refresh=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
    )

    response = TestClient(app).get("/market/flow/head")

    assert response.status_code == 200
    assert submitted == []
    assert response.json()["refresh_pending"] is False


def test_kap_flow_refresh_failure_keeps_last_successful_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module._kap_flow_store, "database_enabled", lambda: False)
    api_module._kap_flow_store.persist_flow_items(
        [_flow_item(idx="stable", category="ozel_durum", published_at="2026-09-09T10:00:00+00:00")],
        source="kap_public_website",
    )
    previous = api_module._kap_flow_store.get_flow_status()

    def fail_refresh(**_kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(api_module, "_build_market_flow_payload", fail_refresh)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        api_module._refresh_kap_flow_store()

    status = api_module._kap_flow_store.get_flow_status()
    page = api_module._kap_flow_store.read_flow_page(limit=1)
    assert status["refresh_status"] == "failed"
    assert status["refresh_pending"] is False
    assert status["last_successful_refresh"] == previous["last_successful_refresh"]
    assert page is not None
    assert page["items"][0]["id"] == "vyk-stable"


def test_parse_kap_public_result_page_maps_related_symbols() -> None:
    page = """
    <table><tbody>
      <tr id="notification1">
        <td><input id="1660456"/><label>checkbox</label></td>
        <td>1</td>
        <td><div>Bugün</div><div>18:51</div></td>
        <td>TERA, TRA</td>
        <td>TERA YATIRIM MENKUL DEĞERLER A.Ş.<span>/ NETGL</span></td>
        <td>DG</td>
        <td>Fiyat Tespit Raporu</td>
        <td>Net Global raporu</td>
        <td>-</td>
      </tr>
    </tbody></table>
    """

    rows = api_module._parse_kap_public_result_page(page, 10)

    assert len(rows) == 1
    assert rows[0]["id"] == "kap-1660456"
    assert rows[0]["symbol"] == "NETGL"
    assert rows[0]["stock_codes"] == ["TERA", "TRA"]
    assert rows[0]["related_symbols"] == ["NETGL"]
    assert rows[0]["source"] == "Diğer Bildirim"
    assert rows[0]["category"] == "diger"
    assert rows[0]["title"] == "Fiyat Tespit Raporu"
    assert rows[0]["published_at"].endswith("18:51:00+03:00")


def test_parse_kap_public_result_page_prefers_embedded_absolute_publish_date() -> None:
    page = r'''
    <script>\"publishDate\":\"08.09.2026 23:57:04\",\"disclosureIndex\":1660529</script>
    <table><tbody>
      <tr id="notification1">
        <td><input id="1660529"/></td>
        <td>1</td>
        <td><div>Bugün</div><div>23:57</div></td>
        <td>ALKLC</td>
        <td>ALTINKILIÇ GIDA VE SÜT SANAYİ TİCARET A.Ş.</td>
        <td>ÖDA</td>
        <td>Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim</td>
        <td>Bedelsiz Sermaye Artırımı Hakkında</td>
      </tr>
    </tbody></table>
    '''

    rows = api_module._parse_kap_public_result_page(page, 10)

    assert len(rows) == 1
    assert rows[0]["published_at"] == "2026-09-08T23:57:04+03:00"


def test_parse_kap_public_result_page_classifies_oda_subjects() -> None:
    page = """
    <table><tbody>
      <tr id="notification1">
        <td><input id="1660457"/></td>
        <td>1</td>
        <td><div>Bugün</div><div>18:51</div></td>
        <td>GUBRF</td>
        <td>GÜBRE FABRİKALARI A.Ş.</td>
        <td>ÖDA</td>
        <td>Genel Kurul İşlemlerine İlişkin Bildirim</td>
        <td>2026 Olağan Genel Kurul Toplantısı</td>
      </tr>
      <tr id="notification2">
        <td><input id="1660458"/></td>
        <td>2</td>
        <td><div>Bugün</div><div>18:50</div></td>
        <td>EPLAS</td>
        <td>EGEPLAS A.Ş.</td>
        <td>ÖDA</td>
        <td>Kar Payı Dağıtım İşlemlerine İlişkin Bildirim</td>
        <td>Kar Payı Dağıtımına İlişkin Yönetim Kurulu Teklifi</td>
      </tr>
    </tbody></table>
    """

    rows = api_module._parse_kap_public_result_page(page, 10)

    assert [row["category"] for row in rows] == ["genel_kurul", "kar_payi"]


def test_parse_kap_api_disclosures_maps_criteria_rows() -> None:
    rows = api_module._parse_kap_api_disclosures(
        [
            {
                "publishDate": "09.09.2026 00:54:31",
                "disclosureIndex": 1660531,
                "disclosureType": "CA",
                "subject": "Genel Kurul İşlemlerine İlişkin Bildirim",
                "summary": "2025 Yılı Genel Kurul Toplantı Çağrısı",
                "stockCodes": "UMPAS",
            },
            {
                "publishDate": "08.09.2026 22:10:00",
                "disclosureIndex": 1660500,
                "disclosureType": "CA",
                "subject": "Kar Payı Dağıtımı",
                "summary": "Kâr payı dağıtım kararı",
                "stockCodes": "THYAO",
            },
            {
                "publishDate": "08.09.2026 21:10:00",
                "disclosureIndex": 1660499,
                "disclosureType": "FR",
                "subject": "Finansal Rapor",
                "summary": "2026/6 Aylık Finansal Rapor",
                "stockCodes": "ASELS",
            },
        ],
        50,
    )

    assert [row["category"] for row in rows] == [
        "genel_kurul",
        "kar_payi",
        "finansal_rapor",
    ]
    assert rows[0]["id"] == "kap-1660531"
    assert rows[0]["symbol"] == "UMPAS"
    assert rows[0]["source"] == "Genel Kurul"


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


def test_normalization_does_not_promote_ambiguous_metrics_to_quarterly_flows():
    result = kap_service_module.normalize_snapshot_for_frontend({
        "ok": True, "company": "TEST", "quarters": [{
            "year": 2026, "period": 2, "quarter": "2026Q2",
            "metrics": {"net_kar": 123}, "metrics_ytd": {"net_kar": 123},
        }],
    })
    assert result["quarters"][0]["metrics_quarterly"]["net_kar"]["value"] is None


def test_cached_kap_normalization_never_calls_inflation_provider(monkeypatch):
    monkeypatch.setattr(kap_service_module, "_load_monthly_inflation_rates", lambda: pytest.fail("cached response contacted TCMB"))
    result = kap_service_module.normalize_snapshot_for_frontend({
        "ok": True, "company": "TEST", "quarters": [
            {"year": 2026, "period": p, "quarter": f"2026Q{p}",
             "metrics": {"ozkaynaklar": 100}, "metrics_quarterly": {"net_kar": 10}}
            for p in (1, 2, 3)]}, allow_network=False)
    assert len(result["quarters"]) == 3
