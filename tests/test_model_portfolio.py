from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import app.api as api_mod
from app.model_portfolio_service import (
    compound_relative_return_pct,
    compute_target_progress_pct,
    reset_admin_rate_limit_for_tests,
    reset_model_portfolio_cache_for_tests,
)


def test_financial_math_helpers() -> None:
    # Compound alpha: ((1 + 0.50) / (1 + 0.25) - 1) * 100 = 20.0%
    assert compound_relative_return_pct(50.0, 25.0) == 20.0
    assert compound_relative_return_pct(10.0, 10.0) == 0.0

    # Target progress bar: Entry 100 -> Current 160 -> Target 200 => 60%
    assert compute_target_progress_pct(100.0, 160.0, 200.0) == 60.0
    assert compute_target_progress_pct(100.0, 220.0, 200.0) == 120.0


from dataclasses import replace

def test_model_portfolio_lifecycle_and_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_model_portfolio_cache_for_tests()
    reset_admin_rate_limit_for_tests()
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        api_mod,
        "CONFIG",
        replace(api_mod.CONFIG, paths=replace(api_mod.CONFIG.paths, processed_dir=tmp_path)),
    )
    monkeypatch.setenv("RAGFIN_ADMIN_TOKEN", "test-founder-secret")

    # Pre-warm _STOCKS_CACHE and _STOCK_CARD_VALUATION_CACHE to prove zero upstream calls
    upstream_price_calls = {"count": 0}
    upstream_history_calls = {"count": 0}

    def fail_if_called(symbols: List[str], **kwargs: Any) -> Dict[str, Any]:
        upstream_price_calls["count"] += 1
        return {}

    monkeypatch.setattr(api_mod, "_fetch_market_price_map", fail_if_called)

    api_mod._STOCKS_CACHE["payload:XUTUM"] = {
        "_ts": 9999999999.0,
        "entry": {
            "fresh_until": 9999999999.0,
            "stale_until": 9999999999.0,
            "payload": {
                "as_of": "2026-10-18T12:00:00Z",
                "rows": [
                    {
                        "symbol": "THYAO",
                        "company": "THYAO",
                        "price": 330.0,
                        "change": 6.0,
                        "change_pct": 1.85,
                        "market_cap": 455400000000.0,
                    },
                    {
                        "symbol": "BIMAS",
                        "company": "BIMAS",
                        "price": 550.0,
                        "change": -5.0,
                        "change_pct": -0.9,
                        "market_cap": 330000000000.0,
                    },
                ],
            },
        },
    }
    monkeypatch.setattr(api_mod, "_model_portfolio_cached_xu100", lambda: 11000.0)
    monkeypatch.setattr(
        api_mod,
        "_resolve_market_card_valuation",
        lambda sym, market_cap=None: {
            "fk": 5.4 if sym == "THYAO" else 14.2,
            "fd_favok": 4.8 if sym == "THYAO" else 9.1,
            "pd_dd": 1.1 if sym == "THYAO" else 4.3,
            "net_borc_favok": 1.2,
            "valuation_financial_period": "2026/06",
        },
    )

    def fake_series(kind: str, symbol: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        upstream_history_calls["count"] += 1
        if kind == "index":
            return [
                {"date": "2026-09-01", "value": 10000.0},
                {"date": "2026-10-01", "value": 10500.0},
            ]
        return [
            {"date": "2026-09-01", "value": 300.0},
            {"date": "2026-10-01", "value": 315.0},
        ]

    monkeypatch.setattr(api_mod, "_model_portfolio_comparison_series", fake_series)

    client = TestClient(api_mod.app)

    # 1. Unauthorized POST rejected
    unauth = client.post(
        "/model-portfolio/positions",
        json={"symbol": "THYAO", "entry_price": 300.0, "target_price": 450.0, "weight_pct": 25.0},
    )
    assert unauth.status_code == 401

    # 2. Verify admin token
    verify_res = client.post(
        "/model-portfolio/admin/verify",
        headers={"X-Admin-Token": "test-founder-secret"},
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["ok"] is True
    session_token = verify_res.json()["session_token"]
    auth_headers = {"Authorization": f"Bearer {session_token}"}

    # 3. Create position using warm cache
    create_res = client.post(
        "/model-portfolio/positions",
        headers=auth_headers,
        json={
            "symbol": "THYAO",
            "entry_date": "2026-09-01",
            "entry_price": 300.0,
            "entry_xu100": 10000.0,
            "target_price": 450.0,
            "weight_pct": 25.0,
            "thesis": "Güçlü yolcu trafiği ve kargo gelirleri ile çarpan iskontosu.",
            "catalysts": ["Çarpan İskontosu", "İhracat / Döviz Geliri"],
            "risk_note": "Petrol fiyatlarında ani yükseliş.",
        },
    )
    assert create_res.status_code == 200
    payload = create_res.json()
    assert upstream_price_calls["count"] == 0  # Zero upstream quote fetch!
    assert payload["summary"]["active_count"] == 1
    assert payload["summary"]["total_weight_pct"] == 25.0
    assert payload["summary"]["cash_weight_pct"] == 75.0

    pos = payload["active_positions"][0]
    assert pos["symbol"] == "THYAO"
    assert pos["current_price"] == 330.0
    assert pos["return_pct"] == 10.0  # 300 -> 330 = +10%
    assert pos["xu100_return_pct"] == 10.0  # 10000 -> 11000 = +10%
    assert pos["relative_return_pct"] == 0.0
    assert pos["target_progress_pct"] == 20.0  # (330-300)/(450-300) = 20%
    assert pos["entry_multiples"]["fk"] == 5.4

    # 4. Symbols endpoint (<1ms from state cache)
    syms_res = client.get("/model-portfolio/symbols")
    assert syms_res.status_code == 200
    assert syms_res.json()["symbols"] == ["THYAO"]

    # 5. Performance endpoint caches by version + day
    perf1 = client.get("/model-portfolio/performance")
    assert perf1.status_code == 200
    assert perf1.json()["cache_status"] == "miss"
    first_call_count = upstream_history_calls["count"]
    assert first_call_count == 2  # XU100 + THYAO

    perf2 = client.get("/model-portfolio/performance")
    assert perf2.status_code == 200
    assert perf2.json()["cache_status"] == "memory_hit"
    assert upstream_history_calls["count"] == first_call_count  # Zero extra history calls!

    # 6. Revise target price & weight
    pos_id = pos["id"]
    upd_res = client.put(
        f"/model-portfolio/positions/{pos_id}",
        headers=auth_headers,
        json={
            "target_price": 480.0,
            "weight_pct": 30.0,
            "revision_note": "3Ç beklentisiyle hedef fiyat yükseltildi",
        },
    )
    assert upd_res.status_code == 200
    updated_pos = upd_res.json()["active_positions"][0]
    assert updated_pos["target_price"] == 480.0
    assert updated_pos["weight_pct"] == 30.0
    assert len(updated_pos["revisions"]) == 1
    assert updated_pos["revisions"][0]["prev_target_price"] == 450.0

    # 7. Close position and verify Hit Rate
    close_res = client.post(
        f"/model-portfolio/positions/{pos_id}/close",
        headers=auth_headers,
        json={"exit_price": 360.0, "exit_xu100": 10500.0, "close_reason": "Hedef Fiyata Yakınsama"},
    )
    assert close_res.status_code == 200
    closed_data = close_res.json()
    assert closed_data["summary"]["active_count"] == 0
    assert closed_data["summary"]["closed_count"] == 1
    assert closed_data["summary"]["closed_hit_rate_pct"] == 100.0
    assert closed_data["closed_positions"][0]["beat_benchmark"] is True

    # 8. Brute-force protection locks out after 5 consecutive failed attempts
    for _ in range(5):
        bad_res = client.post(
            "/model-portfolio/admin/verify",
            headers={"X-Admin-Token": "wrong-secret"},
        )
        assert bad_res.status_code == 401

    locked_res = client.post(
        "/model-portfolio/admin/verify",
        headers={"X-Admin-Token": "test-founder-secret"},
    )
    assert locked_res.status_code == 429


def test_bist_market_hours_and_sse_delta_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime, timedelta, timezone
    import json

    trt = timezone(timedelta(hours=3))
    # Wednesday 14:30 TRT -> Open
    wed_open = datetime(2026, 10, 14, 14, 30, tzinfo=trt)
    assert api_mod._is_bist_market_open(wed_open) is True
    assert api_mod._effective_market_quotes_fresh_ttl(wed_open) == 3

    # Wednesday 20:00 TRT -> Closed (Off-hours)
    wed_night = datetime(2026, 10, 14, 20, 0, tzinfo=trt)
    assert api_mod._is_bist_market_open(wed_night) is False
    assert api_mod._effective_market_quotes_fresh_ttl(wed_night) >= 300

    # Sunday 12:00 TRT -> Closed (Weekend)
    sun_noon = datetime(2026, 10, 18, 12, 0, tzinfo=trt)
    assert api_mod._is_bist_market_open(sun_noon) is False

    # Delta computation only returns changed rows
    rows_tick_1 = [
        {"company": "BIMAS", "price": 510.0, "change": 5.0, "change_pct": 0.99, "volume": 1000.0, "market_cap": 300e9},
        {"company": "THYAO", "price": 310.0, "change": 2.0, "change_pct": 0.65, "volume": 2000.0, "market_cap": 420e9},
    ]
    changed_1, snap_1 = api_mod._compute_market_stocks_delta({}, rows_tick_1)
    assert len(changed_1) == 2

    rows_tick_2 = [
        {"company": "BIMAS", "price": 512.5, "change": 7.5, "change_pct": 1.48, "volume": 1200.0, "market_cap": 301e9},
        {"company": "THYAO", "price": 310.0, "change": 2.0, "change_pct": 0.65, "volume": 2000.0, "market_cap": 420e9},
    ]
    changed_2, snap_2 = api_mod._compute_market_stocks_delta(snap_1, rows_tick_2)
    assert len(changed_2) == 1
    assert changed_2[0]["company"] == "BIMAS"
    assert changed_2[0]["price"] == 512.5

    # Tick 3 with no changes emits 0 rows
    changed_3, _ = api_mod._compute_market_stocks_delta(snap_2, rows_tick_2)
    assert changed_3 == []

    # Verify GET /market/stream?index=XUTUM&once=true SSE endpoint
    monkeypatch.setattr(
        api_mod,
        "_market_stocks_payload",
        lambda index_name="XUTUM", force_refresh=False: {
            "index": index_name,
            "as_of": "2026-10-18T15:00:00+00:00",
            "rows": rows_tick_2,
        },
    )
    client = TestClient(api_mod.app)
    res = client.get("/market/stream?index=XUTUM&once=true")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    body_text = res.text
    assert "event: stocks_delta" in body_text
    data_line = next(line for line in body_text.splitlines() if line.startswith("data: "))
    parsed = json.loads(data_line[len("data: "):])
    assert parsed["index"] == "XUTUM"
    assert parsed["count"] == 2
    assert parsed["rows"][0]["company"] == "BIMAS"


