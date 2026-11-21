from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api_module_facade import api_module as api_mod
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
    assert first_call_count == 3  # XU100 + XU030 + THYAO

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
    assert api_mod._effective_market_quotes_fresh_ttl(wed_open) == api_mod._MARKET_QUOTES_FRESH_TTL

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


def test_market_stream_skips_delta_for_reused_rows_and_updates_new_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import json

    from app.routers import market as market_router

    rows_first = [
        {"company": "BIMAS", "price": 510.0, "change_pct": 0.99, "volume": 1000.0},
        {"company": "THYAO", "price": 310.0, "change_pct": 0.65, "volume": 2000.0},
    ]
    rows_updated = [
        {"company": "BIMAS", "price": 512.5, "change_pct": 1.48, "volume": 1200.0},
        {"company": "THYAO", "price": 310.0, "change_pct": 0.65, "volume": 2000.0},
    ]
    snapshot_rows = iter([rows_first, rows_first, rows_updated])
    delta_calls: List[List[Dict[str, Any]]] = []
    original_delta = market_router._compute_market_stocks_delta

    def fake_market_stocks_payload(*, index_name: str = "XUTUM") -> Dict[str, Any]:
        return {
            "index": index_name,
            "as_of": "2026-10-31T10:00:00+00:00",
            "rows": next(snapshot_rows),
        }

    def count_delta_calls(
        previous: Dict[str, tuple[Any, ...]],
        rows: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, tuple[Any, ...]]]:
        delta_calls.append(rows)
        return original_delta(previous, rows)

    async def no_sleep(_seconds: float) -> None:
        return None

    class FakeRequest:
        checks = 0

        async def is_disconnected(self) -> bool:
            self.checks += 1
            return self.checks > 3

    monkeypatch.setattr(market_router, "_market_stocks_payload", fake_market_stocks_payload)
    monkeypatch.setattr(market_router, "_compute_market_stocks_delta", count_delta_calls)
    monkeypatch.setattr(market_router, "_is_bist_market_open", lambda: True)
    monkeypatch.setattr(market_router.asyncio, "sleep", no_sleep)

    async def collect_stream() -> str:
        response = await market_router.market_stream(FakeRequest(), index="XUTUM", once=False)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
        return "".join(chunks)

    body = asyncio.run(collect_stream())

    assert delta_calls == [rows_first, rows_updated]
    events = [part for part in body.split("event: stocks_delta\n") if part.strip()]
    assert len(events) == 2
    event_data = [
        json.loads(next(line[6:] for line in event.splitlines() if line.startswith("data: ")))
        for event in events
    ]
    assert event_data[0]["count"] == 2
    assert event_data[1]["count"] == 1
    assert event_data[1]["rows"][0]["company"] == "BIMAS"
    assert event_data[1]["rows"][0]["price"] == 512.5


def _write_model_portfolio_state(tmp_path: Path, positions: List[Dict[str, Any]], version: int) -> Path:
    state_path = tmp_path / "model_portfolio" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "version": version,
                "updated_at": "2026-02-01T00:00:00+00:00",
                "positions": positions,
                "changelog": [],
            }
        ),
        encoding="utf-8",
    )
    return state_path


def _performance_test_series(series: Dict[str, Dict[str, float]]):
    calls: List[tuple[str, str, date, date]] = []

    def fake_series(kind: str, symbol: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        calls.append((kind, symbol, start_date, end_date))
        return [{"date": d, "value": v} for d, v in sorted(series.get(symbol, {}).items())]

    return fake_series, calls


def test_performance_series_keeps_a_closed_position_until_its_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import model_portfolio_service as mps

    reset_model_portfolio_cache_for_tests()
    monkeypatch.setattr(mps, "read_json_cache", lambda path: None)
    positions = [
        {
            "id": "pos-thyao",
            "symbol": "THYAO",
            "status": "active",
            "weight_pct": 50.0,
            "entry_date": "2026-01-05",
            "entry_price": 100.0,
        },
        {
            "id": "pos-bimas",
            "symbol": "BIMAS",
            "status": "closed",
            "weight_pct": 50.0,
            "entry_date": "2026-01-05",
            "entry_price": 100.0,
            "closed_date": "2026-01-07",
            "exit_price": 130.0,
        },
    ]
    _write_model_portfolio_state(tmp_path, positions, version=41)

    fake_series, calls = _performance_test_series(
        {
            "XU100": {d: 100.0 for d in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")},
            "XU030": {d: 100.0 for d in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")},
            "THYAO": {d: 100.0 for d in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")},
            "BIMAS": {"2026-01-05": 100.0, "2026-01-06": 100.0, "2026-01-07": 120.0},
        }
    )

    payload = mps.build_model_portfolio_performance_payload(
        tmp_path,
        fetch_series_fn=fake_series,
        quote_map={},
        current_xu100=None,
    )

    points = payload["points"]
    assert [p["date"] for p in points] == [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
    ]
    by_date = {p["date"]: p for p in points}
    # The closed position is still flat on the entry day and the day after.
    assert by_date["2026-01-06"]["portfolio_return_pct"] == 0.0
    # On the exit day the realized move is booked at exit_price (130), not the
    # last market print (120): half the sleeve gains 30% => +15%.
    assert by_date["2026-01-07"]["portfolio_return_pct"] == 15.0
    # After the close the position no longer contributes, but the realized gain stays.
    assert by_date["2026-01-09"]["portfolio_return_pct"] == 15.0

    # The closed symbol is only fetched through its exit date.
    closed_calls = [c for c in calls if c[1] == "BIMAS"]
    assert len(closed_calls) == 1
    assert closed_calls[0][3] == date(2026, 1, 7)
    assert all(c[3] > date(2026, 1, 7) for c in calls if c[1] == "THYAO")


def test_performance_series_supports_a_portfolio_with_only_closed_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import model_portfolio_service as mps

    reset_model_portfolio_cache_for_tests()
    monkeypatch.setattr(mps, "read_json_cache", lambda path: None)
    positions = [
        {
            "id": "pos-only-closed",
            "symbol": "ONLY",
            "status": "closed",
            "weight_pct": 40.0,
            "entry_date": "2026-01-05",
            "entry_price": 100.0,
            # Saturday close: no price bar exists for this date, so the exit
            # must still be booked on the chart by the series builder.
            "closed_date": "2026-01-10",
            "exit_price": 110.0,
        }
    ]
    _write_model_portfolio_state(tmp_path, positions, version=42)

    fake_series, _ = _performance_test_series(
        {
            "XU100": {d: 100.0 for d in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")},
            "XU030": {d: 100.0 for d in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")},
            "ONLY": {d: 100.0 for d in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")},
        }
    )

    payload = mps.build_model_portfolio_performance_payload(
        tmp_path,
        fetch_series_fn=fake_series,
        quote_map={},
        current_xu100=None,
    )

    assert payload["cache_status"] == "miss"
    assert payload["points"], "a fully closed portfolio must still expose its realized track record"
    assert payload["points"][-1]["date"] == "2026-01-10"
    assert payload["points"][-1]["portfolio_return_pct"] == 4.0


def test_load_model_portfolio_state_uses_conditional_cache_and_honors_read_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import model_portfolio_service as mps

    reset_model_portfolio_cache_for_tests()
    calls: list[Any] = []
    remote_payload = {
        "version": 7,
        "updated_at": "2026-11-20T10:00:00+00:00",
        "positions": [
            {
                "id": "pos-1",
                "symbol": "THYAO",
                "status": "active",
                "weight_pct": 100.0,
                "entry_date": "2026-11-01",
                "entry_price": 300.0,
            }
        ],
        "changelog": [],
    }
    remote_stamp = ("2026-11-20T10:00:00+00:00", 7)

    def fake_conditional(_path: Path, *, known_stamp=None):
        calls.append(known_stamp)
        if known_stamp == remote_stamp:
            return ("not_modified", None, remote_stamp)
        return ("hit", dict(remote_payload), remote_stamp)

    monkeypatch.setattr(mps, "read_json_cache_conditional", fake_conditional)

    first = mps.load_model_portfolio_state(tmp_path)
    assert first["version"] == 7
    assert first["positions"][0]["symbol"] == "THYAO"
    assert calls == [None]

    reloaded = mps.load_model_portfolio_state(tmp_path, force_reload=True)
    assert reloaded["version"] == 7
    assert calls == [None, remote_stamp]

    direct_calls: list[Path] = []
    monkeypatch.setattr(mps, "read_json_cache_conditional", mps._DEFAULT_READ_JSON_CACHE_CONDITIONAL)
    monkeypatch.setattr(mps, "read_json_cache", lambda path: direct_calls.append(path) or None)
    mps.load_model_portfolio_state(tmp_path, force_reload=True)
    assert len(direct_calls) == 1
    assert len(calls) == 2
    reset_model_portfolio_cache_for_tests()


def test_add_to_existing_position_averages_cost_preserves_analysis_and_tracks_historical_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import model_portfolio_service as mps

    reset_model_portfolio_cache_for_tests()
    reset_admin_rate_limit_for_tests()
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(mps, "read_json_cache", lambda path: None)
    monkeypatch.setattr(
        api_mod,
        "CONFIG",
        replace(api_mod.CONFIG, paths=replace(api_mod.CONFIG.paths, processed_dir=tmp_path)),
    )
    monkeypatch.setenv("RAGFIN_ADMIN_TOKEN", "test-founder-secret")

    monkeypatch.setattr(api_mod, "_fetch_market_price_map", lambda symbols, **kwargs: {})
    api_mod._STOCKS_CACHE["payload:XUTUM"] = {
        "_ts": 9999999999.0,
        "entry": {
            "fresh_until": 9999999999.0,
            "stale_until": 9999999999.0,
            "payload": {
                "as_of": "2026-10-23T12:00:00Z",
                "rows": [
                    {
                        "symbol": "BIMAS",
                        "company": "BIMAS",
                        "price": 374.25,
                        "change": -5.0,
                        "change_pct": -1.32,
                        "market_cap": 225000000000.0,
                    }
                ],
            },
        },
    }
    monkeypatch.setattr(api_mod, "_model_portfolio_cached_xu100", lambda: 11131.32)
    monkeypatch.setattr(
        api_mod,
        "_resolve_market_card_valuation",
        lambda sym, market_cap=None: {
            "fk": 14.2,
            "fd_favok": 9.1,
            "pd_dd": 4.3,
            "net_borc_favok": 0.5,
            "valuation_financial_period": "2026/06",
        },
    )

    client = TestClient(api_mod.app)
    verify_res = client.post(
        "/model-portfolio/admin/verify",
        headers={"X-Admin-Token": "test-founder-secret"},
    )
    assert verify_res.status_code == 200
    auth_headers = {"Authorization": f"Bearer {verify_res.json()['session_token']}"}

    create_res = client.post(
        "/model-portfolio/positions",
        headers=auth_headers,
        json={
            "symbol": "BIMAS",
            "entry_date": "2026-10-21",
            "entry_price": 385.75,
            "entry_xu100": 11654.28,
            "target_price": 570.0,
            "weight_pct": 20.0,
            "thesis": "Original thesis",
            "risk_note": "Original risk",
            "catalysts": ["Marj Genişlemesi"],
        },
    )
    assert create_res.status_code == 200
    pos_id = create_res.json()["active_positions"][0]["id"]

    add_res = client.post(
        f"/model-portfolio/positions/{pos_id}/add",
        headers=auth_headers,
        json={
            "add_weight_pct": 5.0,
            "add_price": 374.25,
            "add_date": "2026-10-23",
            "add_xu100": 11131.32,
            "note": "Kademeli ekleme",
        },
    )
    assert add_res.status_code == 200
    payload = add_res.json()
    pos = payload["active_positions"][0]
    assert pos["weight_pct"] == 25.0
    assert pos["entry_price"] == 383.45
    assert pos["entry_xu100"] == 11549.69
    assert pos["target_price"] == 570.0
    assert pos["thesis"] == "Original thesis"
    assert pos["risk_note"] == "Original risk"
    assert pos["catalysts"] == ["Marj Genişlemesi"]
    assert payload["summary"]["total_weight_pct"] == 25.0

    assert len(pos["revisions"]) == 1
    rev = pos["revisions"][0]
    assert rev["date"] == "2026-10-23"
    assert rev["prev_weight_pct"] == 20.0
    assert rev["new_weight_pct"] == 25.0
    assert rev["add_weight_pct"] == 5.0
    assert rev["add_price"] == 374.25
    assert rev["prev_entry_price"] == 385.75
    assert rev["new_entry_price"] == 383.45
    assert rev["note"] == "Kademeli ekleme"

    log_entry = payload["changelog"][0]
    assert log_entry["action"] == "weight_revised"
    assert log_entry["symbol"] == "BIMAS"
    assert log_entry["details"]["new_weight_pct"] == 25.0
    assert log_entry["details"]["new_entry_price"] == 383.45

    exceed_res = client.post(
        f"/model-portfolio/positions/{pos_id}/add",
        headers=auth_headers,
        json={
            "add_weight_pct": 80.0,
            "add_price": 374.25,
            "add_date": "2026-10-23",
        },
    )
    assert exceed_res.status_code == 400

    fake_series, _ = _performance_test_series(
        {
            "XU100": {
                "2026-10-21": 100.0,
                "2026-10-22": 100.0,
                "2026-10-23": 100.0,
                "2026-10-24": 100.0,
            },
            "XU030": {
                "2026-10-21": 100.0,
                "2026-10-22": 100.0,
                "2026-10-23": 100.0,
                "2026-10-24": 100.0,
            },
            "BIMAS": {
                "2026-10-21": 100.0,
                "2026-10-22": 110.0,
                "2026-10-23": 110.0,
                "2026-10-24": 132.0,
            },
        }
    )
    perf_payload = mps.build_model_portfolio_performance_payload(
        tmp_path,
        fetch_series_fn=fake_series,
        quote_map={},
        current_xu100=None,
    )
    by_date = {p["date"]: p for p in perf_payload["points"]}
    assert by_date["2026-10-22"]["portfolio_return_pct"] == 2.0
    assert by_date["2026-10-23"]["portfolio_return_pct"] == 2.0
    assert by_date["2026-10-24"]["portfolio_return_pct"] == 7.1
    reset_model_portfolio_cache_for_tests()



