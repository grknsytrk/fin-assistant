from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from api_module_facade import api_module
from app import cache as cache_module
from app import database as database_module
from app import kap_service as kap_service_module
from app.api import app
from api_test_helpers import _reset_flow_state


def test_market_universe_is_metadata_only_and_batch_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda index, force_refresh=False: {
            "index": str(index).upper(),
            "symbols": ["YKBNK", "BIMAS"],
            "count": 2,
            "source": "test",
            "source_url": None,
            "source_date": None,
            "fetched_at": "2026-08-21T10:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(
        api_module,
        "get_instruments",
        lambda *_args, **_kwargs: {
            "YKBNK": {
                "symbol": "YKBNK",
                "name": "Yapı ve Kredi Bankası A.Ş.",
                "source": "reference_data",
                "logo_url": "https://example.test/ykbnk.svg",
                "logo_source": "manual",
                "metadata": {"latest_quarter": "2026Q2", "has_kap_cache": True},
            }
        },
    )
    monkeypatch.setattr(
        api_module,
        "get_financial_snapshots_map",
        lambda _processed_dir, _symbols: {},
    )
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda *_args: pytest.fail("market universe should use the SQL snapshot map"),
    )
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda *_args, **_kwargs: pytest.fail("universe must not fetch quotes"))
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: pytest.fail("universe must not fetch summaries"))

    payload = api_module._market_universe_payload(index_name="XU100", force_refresh=True)
    rows = {row["symbol"]: row for row in payload["rows"]}

    assert rows["YKBNK"]["name"] == "Yapı ve Kredi Bankası A.Ş."
    assert rows["YKBNK"]["logo_url"] == "https://example.test/ykbnk.svg"
    assert rows["YKBNK"]["latest_quarter"] == "2026Q2"
    assert rows["YKBNK"]["has_kap_cache"] is True
    assert rows["BIMAS"]["name"] == "BIMAS"
    assert rows["BIMAS"]["price"] is None


def test_market_stock_search_uses_metadata_without_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "_market_universe_payload", lambda **_kwargs: {
        "universe": {"fetched_at": "2026-08-21T10:00:00+00:00"},
        "rows": [
            {"symbol": "YKBNK", "company": "YKBNK", "name": "Yapı ve Kredi Bankası A.Ş."},
            {"symbol": "BIMAS", "company": "BIMAS", "name": "BİM Birleşik Mağazalar A.Ş."},
        ],
    })
    monkeypatch.setattr(api_module, "_fetch_market_price_map", lambda *_args, **_kwargs: pytest.fail("search must not fetch quotes"))

    response = TestClient(app).get("/market/stocks/search", params={"q": "ykbnk", "limit": 20})

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["rows"][0]["symbol"] == "YKBNK"


def test_market_stocks_serves_stale_quote_when_upstream_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_entry = {
        "payload": {"index": "XUTUM", "rows": [{"company": "YKBNK", "price": 25.0}], "benchmarks": {}, "as_of": "old"},
        "fresh_until": time.time() - 10,
        "stale_until": time.time() + 120,
    }
    backend = cache_module.get_cache()
    backend.set("api:market:stocks:XUTUM:v4", stale_entry, ttl_seconds=120)
    monkeypatch.setattr(api_module, "_build_market_stocks_payload", lambda **_kwargs: {
        "index": "XUTUM",
        "rows": [{"company": "YKBNK", "price": None}],
        "benchmarks": {},
        "as_of": "new",
    })

    payload = api_module._market_stocks_payload(index_name="XUTUM")

    assert payload["stale"] is True
    assert payload["quote_status"] == "stale"
    assert payload["rows"][0]["price"] == 25.0
    assert backend.get("api:market:stocks:XUTUM:v4") == stale_entry


def test_market_stocks_flags_aged_quotes_as_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build fed by an aged quote snapshot must not be reported as fresh."""
    monkeypatch.setattr(api_module, "_build_market_stocks_payload", lambda **_kwargs: {
        "index": "XUTUM",
        "rows": [{"company": "YKBNK", "price": 25.0}],
        "benchmarks": {},
        "as_of": "old",
        "quote_stale": True,
    })

    payload = api_module._market_stocks_payload(index_name="XUTUM")

    assert payload["stale"] is True
    assert payload["quote_status"] == "stale"
    assert "quote_stale" not in payload

    entry = cache_module.get_cache().get("api:market:stocks:XUTUM:v4")
    assert entry["quote_stale"] is True
    assert "quote_stale" not in entry["payload"]

    # The process-local hit must keep reporting the aged quotes as stale.
    monkeypatch.setattr(
        api_module,
        "_build_market_stocks_payload",
        lambda **_kwargs: pytest.fail("a fresh envelope entry should not rebuild"),
    )
    cached = api_module._market_stocks_payload(index_name="XUTUM")
    assert cached["stale"] is True
    assert cached["quote_status"] == "stale"
    assert "quote_stale" not in cached


def test_market_stocks_caches_no_quote_result_for_three_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def no_quote_build(**_kwargs: Any) -> Dict[str, Any]:
        calls.append("build")
        return {
            "index": "XUTUM",
            "rows": [{"company": "YKBNK", "symbol": "YKBNK", "price": None}],
            "benchmarks": {},
            "as_of": "new",
        }

    monkeypatch.setattr(api_module, "_MARKET_QUOTES_NOQUOTE_TTL_SECONDS", 1)
    monkeypatch.setattr(api_module, "_build_market_stocks_payload", no_quote_build)
    monkeypatch.setattr(
        api_module,
        "_market_unavailable_stocks_payload",
        lambda index_name: {"index": index_name, "rows": [], "benchmarks": {}},
    )

    first = api_module._market_stocks_payload(index_name="XUTUM")
    second = api_module._market_stocks_payload(index_name="XUTUM")

    assert first["quote_status"] == "unavailable"
    assert second["quote_status"] == "unavailable"
    assert len(calls) == 1

    time.sleep(1.05)
    third = api_module._market_stocks_payload(index_name="XUTUM")
    assert third["quote_status"] == "unavailable"
    assert len(calls) == 2


def test_market_stocks_does_not_negative_cache_empty_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def empty_universe_build(**_kwargs: Any) -> Dict[str, Any]:
        calls.append("build")
        return {"index": "XUTUM", "rows": [], "benchmarks": {}, "as_of": "now"}

    monkeypatch.setattr(api_module, "_build_market_stocks_payload", empty_universe_build)

    payload = api_module._market_stocks_payload(index_name="XUTUM")

    assert payload["quote_status"] == "fresh"
    assert cache_module.get_cache().get("api:market:stocks:XUTUM:no-quote:v1") is None
    assert calls == ["build"]


def test_market_stocks_success_clears_no_quote_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cache_module.get_cache()
    no_quote_key = "api:market:stocks:XUTUM:no-quote:v1"

    def successful_build(**_kwargs: Any) -> Dict[str, Any]:
        # Model a marker written by a concurrent failed build before this
        # successful result is stored.
        backend.set(no_quote_key, {"status": "no_quote"}, ttl_seconds=3)
        return {
            "index": "XUTUM",
            "rows": [{"company": "YKBNK", "symbol": "YKBNK", "price": 25.0}],
            "benchmarks": {},
            "as_of": "now",
        }

    monkeypatch.setattr(api_module, "_build_market_stocks_payload", successful_build)

    payload = api_module._market_stocks_payload(index_name="XUTUM")

    assert payload["quote_status"] == "fresh"
    assert backend.get(no_quote_key) is None


def test_market_stocks_force_refresh_skips_build_during_negative_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cache_module.get_cache()
    backend.set(
        "api:market:stocks:XUTUM:no-quote:v1",
        {"status": "no_quote"},
        ttl_seconds=3,
    )
    monkeypatch.setattr(
        api_module,
        "_schedule_swr_revalidation",
        lambda **_kwargs: pytest.fail("negative cache should suppress revalidation"),
    )
    monkeypatch.setattr(
        api_module,
        "_build_market_stocks_payload",
        lambda **_kwargs: pytest.fail("negative cache should suppress a new build"),
    )
    monkeypatch.setattr(
        api_module,
        "_market_unavailable_stocks_payload",
        lambda index_name: {"index": index_name, "rows": [], "benchmarks": {}},
    )

    payload = api_module._market_stocks_payload(index_name="XUTUM", force_refresh=True)

    assert payload["quote_status"] == "unavailable"
    assert payload["refresh_pending"] is False


def test_market_indices_use_shared_stale_while_revalidate_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_row(index_code: str, **_kwargs: Any) -> Dict[str, Any]:
        calls.append(index_code)
        return {"symbol": index_code, "price": 100.0}

    monkeypatch.setattr(api_module, "_market_index_row", fake_row)

    first = api_module._market_indices_payload()
    second = api_module._market_indices_payload()

    assert len(calls) == len(api_module._MARKET_INDEX_ORDER)
    assert first["cache_status"] == "miss"
    assert second["cache_status"] == "local_hit"
    assert second["stale"] is False


def test_market_fx_shared_cache_collapses_quote_fanout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_quote(symbol: str) -> Dict[str, Any]:
        calls.append(symbol)
        return {
            "ok": True,
            "price": 10.0,
            "prev_close": 9.0,
            "change": 1.0,
            "change_pct": 11.1,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-09-06T10:00:00+00:00",
        }

    monkeypatch.setattr(api_module, "_fetch_yahoo_quote", fake_quote)

    first = api_module._market_fx_payload()
    second = api_module._market_fx_payload()

    assert len(calls) == len(api_module._FX_DIRECT_MAP)
    assert first["cache_status"] == "miss"
    assert second["cache_status"] == "local_hit"


def test_comparison_history_cache_excludes_id_and_label(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_history(
        asset: Any,
        *,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "id": asset.id or "default",
            "kind": "stock",
            "symbol": "YKBNK",
            "label": "Yapı Kredi",
            "points": [{"date": start_date.isoformat(), "value": 10.0}],
            "source": "test",
            "error": None,
        }

    monkeypatch.setattr(api_module, "_stock_comparison_history", fake_history)
    client = TestClient(app)
    body = {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "assets": [{"id": "first", "kind": "stock", "symbol": "YKBNK", "label": "İlk etiket"}],
    }
    first = client.post("/market/comparison-history", json=body)
    body["assets"][0]["id"] = "second"
    body["assets"][0]["label"] = "İkinci etiket"
    second = client.post("/market/comparison-history", json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == 1
    assert second.json()["assets"][0]["id"] == "second"
    assert second.json()["assets"][0]["label"] == "İkinci etiket"


def test_fetch_market_price_map_parses_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(database_module, "database_enabled", lambda: False)
    html = """
    <table><tbody id="tableBody">
      <tr data-symbol="A1CAP">
        <th scope="row">A1CAP</th>
        <td>A1 CAPITAL</td>
        <td class="price" data-val="14.18">14.18</td>
        <td class="change" data-val="-0.21">-0.21</td>
        <td class="percent" data-val="-1.46">-1.46 %</td>
        <td>108.750.000,50</td>
        <td class="previousClose" data-val="14.39">14.39</td>
      </tr>
    </tbody></table>
    """

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return html.encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())

    payload = api_module._fetch_market_price_map(["A1CAP"])

    assert payload["A1CAP"]["price"] == 14.18
    assert payload["A1CAP"]["change_pct"] == -1.46
    assert payload["A1CAP"]["volume"] == 108750000.50


def test_fetch_market_price_map_reuses_one_snapshot_across_market_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(database_module, "database_enabled", lambda: False)
    html = """
    <table><tbody id="tableBody">
      <tr data-symbol="A1CAP">
        <td>A1 CAPITAL</td>
        <td class="price" data-val="14.18">14.18</td>
        <td class="change" data-val="-0.21">-0.21</td>
        <td class="percent" data-val="-1.46">-1.46 %</td>
        <td>108.750.000</td>
        <td class="previousClose" data-val="14.39">14.39</td>
      </tr>
      <tr data-symbol="BIMAS">
        <td>BIM BIRLESIK MAGAZALAR</td>
        <td class="price" data-val="760.00">760.00</td>
        <td class="change" data-val="-3.00">-3.00</td>
        <td class="percent" data-val="-0.39">-0.39 %</td>
        <td>2.813.888.143</td>
        <td class="previousClose" data-val="763.00">763.00</td>
      </tr>
    </tbody></table>
    """

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return html.encode("utf-8")

    requested_urls: List[str] = []

    def fake_urlopen(request: Any, **_kwargs: Any) -> FakeResponse:
        requested_urls.append(str(getattr(request, "full_url", request)))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    first = api_module._fetch_market_price_map(["A1CAP"], index_name="XU100")
    second = api_module._fetch_market_price_map(["BIMAS"], index_name="XU030")

    assert requested_urls == ["https://infoyatirim.com/canli-borsa"]
    assert first["A1CAP"]["as_of"] == second["BIMAS"]["as_of"]
    assert first["A1CAP"]["price"] == 14.18
    assert second["BIMAS"]["price"] == 760.0


def test_fetch_market_price_map_uses_shared_redis_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = api_module._market_price_snapshot_payload(
        {
            "A1CAP": {"price": 14.18, "currency": "TRY", "as_of": "2026-10-31T10:00:00+00:00"},
            "BIMAS": {"price": 760.0, "currency": "TRY", "as_of": "2026-10-31T10:00:00+00:00"},
        }
    )
    shared_keys: List[str] = []
    monkeypatch.setattr(database_module, "database_enabled", lambda: False)
    monkeypatch.setattr(
        api_module,
        "_cache_status",
        lambda: {"cache_backend": "redis", "cache_redis_fallback": False},
    )
    monkeypatch.setattr(
        api_module,
        "_shared_cache_get_dict",
        lambda key: shared_keys.append(key) or snapshot,
    )
    monkeypatch.setattr(
        api_module,
        "_build_market_price_snapshot",
        lambda _symbols: pytest.fail("fresh shared snapshot should avoid upstream fetch"),
    )

    payload = api_module._fetch_market_price_map(["A1CAP"])

    assert shared_keys == ["api:market:price-snapshot:XUTUM:v1"]
    assert list(payload) == ["A1CAP"]
    assert payload["A1CAP"]["price"] == 14.18


def test_fetch_market_price_map_uses_postgres_snapshot_when_redis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = api_module._market_price_snapshot_payload(
        {"A1CAP": {"price": 14.18, "currency": "TRY", "as_of": "2026-10-31T10:00:00+00:00"}}
    )
    read_paths: List[Any] = []
    writes: List[Any] = []

    monkeypatch.setattr(database_module, "database_enabled", lambda: True)
    monkeypatch.setattr(database_module, "read_json_cache", lambda path: read_paths.append(path) or snapshot)
    monkeypatch.setattr(database_module, "write_json_cache", lambda path, value: writes.append((path, value)))
    monkeypatch.setattr(
        api_module,
        "_cache_status",
        lambda: {"cache_backend": "redis", "cache_redis_fallback": True},
    )
    monkeypatch.setattr(
        api_module,
        "_shared_cache_get_dict",
        lambda _key: pytest.fail("Redis fallback should use the Postgres snapshot"),
    )
    monkeypatch.setattr(
        api_module,
        "_build_market_price_snapshot",
        lambda _symbols: pytest.fail("fresh Postgres snapshot should avoid upstream fetch"),
    )

    payload = api_module._fetch_market_price_map(["A1CAP"])

    assert read_paths == [api_module._MARKET_PRICE_DATABASE_CACHE_PATH]
    assert writes == []
    assert payload["A1CAP"]["price"] == 14.18


def test_fetch_market_price_map_coalesces_concurrent_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent requests must produce exactly one upstream build and one write."""
    snapshot_store: Dict[str, Any] = {"payload": None, "writes": 0}
    state_lock = threading.Lock()
    first_read_barrier = threading.Barrier(2)
    initial_reads = 0
    build_calls: List[int] = []

    def fake_database_read() -> Optional[Dict[str, Any]]:
        nonlocal initial_reads
        with state_lock:
            synchronize = snapshot_store["payload"] is None and initial_reads < 2
            if synchronize:
                initial_reads += 1
            payload = snapshot_store["payload"]
        if synchronize:
            first_read_barrier.wait(timeout=3)
        return payload

    def fake_database_write(payload: Dict[str, Any]) -> None:
        with state_lock:
            snapshot_store["payload"] = payload
            snapshot_store["writes"] += 1

    def fake_build(_symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        with state_lock:
            build_calls.append(1)
        time.sleep(0.05)
        return {"A1CAP": {"price": 14.18, "currency": "TRY", "as_of": "2026-10-31T10:00:00+00:00"}}

    monkeypatch.setattr(database_module, "database_enabled", lambda: True)
    monkeypatch.setattr(api_module, "_market_price_snapshot_database_read", fake_database_read)
    monkeypatch.setattr(api_module, "_market_price_snapshot_database_write", fake_database_write)
    monkeypatch.setattr(
        api_module,
        "_cache_status",
        lambda: {"cache_backend": "memory", "cache_redis_fallback": True},
    )
    monkeypatch.setattr(api_module, "_build_market_price_snapshot", fake_build)
    api_module._MARKET_PRICE_CACHE.clear()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: api_module._fetch_market_price_map(["A1CAP"]), range(2)))

    assert results == [results[0], results[0]]
    assert results[0]["A1CAP"]["price"] == 14.18
    assert len(build_calls) == 1
    assert snapshot_store["writes"] == 1

    # A later request is served from process memory without a rebuild or a read.
    api_module._MARKET_PRICE_CACHE.clear()
    cached = api_module._fetch_market_price_map(["A1CAP"])
    assert cached["A1CAP"]["price"] == 14.18
    assert len(build_calls) == 1

    # An aged-out snapshot is refreshed instead of being trusted forever.
    snapshot_store["payload"]["created_at"] = time.time() - 10_000
    api_module._MARKET_PRICE_CACHE.clear()
    refreshed = api_module._fetch_market_price_map(["A1CAP"])
    assert refreshed["A1CAP"]["price"] == 14.18
    assert len(build_calls) == 2
    assert snapshot_store["writes"] == 2


def test_fetch_market_price_map_falls_back_when_shared_stores_fail_and_does_not_cache_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls: List[int] = []
    write_attempts: List[Any] = []

    def unavailable_database_write(path: Any, payload: Dict[str, Any]) -> None:
        write_attempts.append((path, payload))
        raise OSError("db down")

    monkeypatch.setattr(database_module, "database_enabled", lambda: True)
    monkeypatch.setattr(database_module, "read_json_cache", lambda _path: (_ for _ in ()).throw(OSError("db down")))
    monkeypatch.setattr(database_module, "write_json_cache", unavailable_database_write)
    monkeypatch.setattr(
        api_module,
        "_cache_status",
        lambda: {"cache_backend": "memory", "cache_redis_fallback": True},
    )
    monkeypatch.setattr(
        api_module,
        "_build_market_price_snapshot",
        lambda _symbols: build_calls.append(1)
        or {"A1CAP": {"price": 14.18, "currency": "TRY", "as_of": "2026-10-31T10:00:00+00:00"}},
    )

    payload = api_module._fetch_market_price_map(["A1CAP"])

    assert payload["A1CAP"]["price"] == 14.18
    assert len(build_calls) == 1
    assert len(write_attempts) == 1

    api_module._MARKET_PRICE_CACHE.clear()
    monkeypatch.setattr(api_module, "_build_market_price_snapshot", lambda _symbols: None)
    failed_payload = api_module._fetch_market_price_map(["A1CAP"])
    assert failed_payload == {}
    assert len(write_attempts) == 1


def test_fetch_market_price_map_builds_when_database_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No persisted snapshot and no lease: the process-local single-flight builds once."""
    builds: List[int] = []
    monkeypatch.setattr(database_module, "database_enabled", lambda: True)
    monkeypatch.setattr(database_module, "read_json_cache", lambda _path: None)
    monkeypatch.setattr(database_module, "write_json_cache", lambda *_args: None)
    monkeypatch.setattr(
        api_module,
        "_cache_status",
        lambda: {"cache_backend": "memory", "cache_redis_fallback": True},
    )
    monkeypatch.setattr(
        api_module,
        "_build_market_price_snapshot",
        lambda _symbols: builds.append(1)
        or {"A1CAP": {"price": 14.18, "currency": "TRY", "as_of": "2026-10-31T10:00:00+00:00"}},
    )
    api_module._MARKET_PRICE_CACHE.clear()

    payload = api_module._fetch_market_price_map(["A1CAP"])

    assert payload["A1CAP"]["price"] == 14.18
    assert len(builds) == 1


def test_fetch_market_price_map_provider_failure_is_not_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: List[int] = []
    writes: List[Any] = []
    monkeypatch.setattr(database_module, "database_enabled", lambda: False)
    monkeypatch.setattr(
        api_module,
        "_cache_status",
        lambda: {"cache_backend": "redis", "cache_redis_fallback": False},
    )
    monkeypatch.setattr(api_module, "_shared_cache_get_dict", lambda _key: None)
    monkeypatch.setattr(api_module, "_shared_cache_set", lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr(
        api_module,
        "_build_market_price_snapshot",
        lambda _symbols: builds.append(1) or None,
    )

    payload = api_module._fetch_market_price_map(["A1CAP"])

    assert payload == {}
    assert len(builds) == 1
    assert writes == []


def test_extract_infoyatirim_stock_page_quote_parses_single_stock_page() -> None:
    html = """
    <section>
      <h1>TRABZONSPOR SPORTİF (TSPOR)</h1>
      <div>Son İşlem Fiyatı 1.03₺</div>
      <div>Satış 1.04₺</div>
      <div>Günlük Değişim % 0.00%</div>
      <div>Günlük Hacim (TL) 358,697,300₺</div>
      <div>Günlük Değişim (TL) 0.00₺</div>
      <div>Piyasa Değeri 7,724,999,785</div>
      <div>F/K 23.86</div>
      <div>PD/DD 0.77</div>
      <div>FD/FAVÖK 11.20</div>
    </section>
    """

    payload = api_module._extract_infoyatirim_stock_page_quote("TSPOR", html)

    assert payload["price"] == 1.03
    assert payload["change_pct"] == 0.0
    assert payload["change"] == 0.0
    assert payload["volume"] == 358_697_300.0
    assert payload["market_cap"] == 7_724_999_785.0
    assert payload["fk"] == 23.86
    assert payload["pd_dd"] == 0.77
    assert payload["fd_favok"] == 11.2
    assert payload["currency"] == "TRY"
    assert api_module._extract_infoyatirim_stock_page_quote("TSPOR", "<html></html>") == {}


def test_fetch_market_price_map_uses_stock_page_fallback_for_missing_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(database_module, "database_enabled", lambda: False)
    html = """
    <table><tbody id="tableBody">
      <tr data-symbol="A1CAP">
        <th scope="row">A1CAP</th>
        <td>A1 CAPITAL</td>
        <td class="price" data-val="14.18">14.18</td>
        <td class="change" data-val="-0.21">-0.21</td>
        <td class="percent" data-val="-1.46">-1.46 %</td>
        <td>108.750.000,50</td>
      </tr>
    </tbody></table>
    """

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return html.encode("utf-8")

    fallback_calls: List[str] = []

    def fake_fallback(symbol: str) -> Dict[str, Any]:
        fallback_calls.append(symbol)
        if symbol == "EGEEN":
            return {
                "price": 6665.0,
                "currency": "TRY",
                "change": -17.35,
                "change_pct": -0.26,
                "volume": 212_043_600.0,
                "market_state": "",
                "as_of": "2026-04-25T09:00:00+00:00",
            }
        return {}

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(api_module, "_fetch_infoyatirim_stock_page_quote", fake_fallback)

    payload = api_module._fetch_market_price_map(["A1CAP", "EGEEN", "TSPOR"])

    assert fallback_calls == ["EGEEN", "TSPOR"]
    assert payload["A1CAP"]["price"] == 14.18
    assert payload["EGEEN"]["price"] == 6665.0
    assert payload["EGEEN"]["change_pct"] == -0.26
    assert payload["EGEEN"]["volume"] == 212_043_600.0
    assert "TSPOR" not in payload


def test_extract_isyatirim_basic_summary_map_parses_market_caps() -> None:
    html = """
    <table class="dataTable" data-csvname="temelozet">
      <thead>
        <tr>
          <th>Kod</th>
          <th>Hisse Adı</th>
          <th>Sektör</th>
          <th>Kapanış (TL)</th>
          <th>Piyasa Değeri (mn TL)</th>
          <th>Piyasa Değeri (mn $)</th>
          <th>Halka Açıklık Oranı (%)</th>
          <th>Sermaye (mn TL)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>ASTOR</td><td>Astor Enerji</td><td>Elektrik</td><td>221,50</td><td>221.057,0</td><td>4.919,5</td><td>42,7</td><td>998,0</td></tr>
        <tr><td>GARAN</td><td>Garanti Bankası</td><td>Bankacılık</td><td>138,00</td><td>579.600,0</td><td>12.898,7</td><td>14,0</td><td>4.200,0</td></tr>
        <tr><td>MGROS</td><td>Migros</td><td>Perakende</td><td>638,50</td><td>115.603,1</td><td>2.572,7</td><td>50,8</td><td>181,1</td></tr>
        <tr><td>TUPRS</td><td>Tüpraş</td><td>Petrol</td><td>269,00</td><td>518.308,0</td><td>11.534,6</td><td>48,6</td><td>1.926,8</td></tr>
      </tbody>
    </table>
    """

    payload = api_module._extract_isyatirim_basic_summary_map(html)

    assert payload["ASTOR"]["market_cap"] == 221_057_000_000.0
    assert payload["GARAN"]["market_cap"] == 579_600_000_000.0
    assert payload["MGROS"]["market_cap"] == 115_603_100_000.0
    assert payload["TUPRS"]["market_cap"] == 518_308_000_000.0
    assert payload["ASTOR"]["fdpo"] == 0.427
    assert payload["ASTOR"]["shares_outstanding"] == 998_000_000.0
    assert api_module._market_cap_from_quote_and_meta(
        {"price": 2.0},
        {},
        {"shares_outstanding": 10.0},
    ) == 20.0


def test_isyatirim_summary_persists_once_per_distinct_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    current_payload = [{"AAA": {"market_cap": 100.0, "sector": "Enerji"}}]
    persisted = []
    monkeypatch.setattr(api_module, "_shared_cache_get_dict", lambda _key: {"items": current_payload[0]})
    monkeypatch.setattr(api_module, "_shared_cache_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api_module,
        "upsert_provider_fundamentals_bulk",
        lambda _processed_dir, items: persisted.append(dict(items)),
    )

    first = api_module._fetch_isyatirim_basic_summary_map()
    api_module._ISYATIRIM_BASIC_SUMMARY_CACHE.clear()
    second = api_module._fetch_isyatirim_basic_summary_map()
    current_payload[0] = {"AAA": {"market_cap": 110.0, "sector": "Enerji"}}
    api_module._ISYATIRIM_BASIC_SUMMARY_CACHE.clear()
    changed = api_module._fetch_isyatirim_basic_summary_map()

    assert first == second == {"AAA": {"market_cap": 100.0, "sector": "Enerji"}}
    assert changed == {"AAA": {"market_cap": 110.0, "sector": "Enerji"}}
    assert persisted == [first, changed]


def test_isyatirim_summary_persistence_retries_after_database_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = {"AAA": {"market_cap": 100.0}}
    upsert_attempts = []
    monkeypatch.setattr(api_module, "_shared_cache_get_dict", lambda _key: {"items": items})
    monkeypatch.setattr(api_module, "_shared_cache_set", lambda *_args, **_kwargs: None)

    def flaky_upsert(_processed_dir, _items):
        upsert_attempts.append(True)
        if len(upsert_attempts) == 1:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(api_module, "upsert_provider_fundamentals_bulk", flaky_upsert)

    assert api_module._fetch_isyatirim_basic_summary_map() == items
    api_module._ISYATIRIM_BASIC_SUMMARY_CACHE.clear()
    assert api_module._fetch_isyatirim_basic_summary_map() == items

    assert len(upsert_attempts) == 2


def test_isyatirim_summary_concurrent_persistence_is_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = {"AAA": {"market_cap": 100.0}}
    upsert_attempts = []
    attempts_lock = threading.Lock()
    start_barrier = threading.Barrier(8)

    def slow_upsert(_processed_dir, _items):
        with attempts_lock:
            upsert_attempts.append(True)
        time.sleep(0.02)

    monkeypatch.setattr(api_module, "upsert_provider_fundamentals_bulk", slow_upsert)

    def persist_once(_index: int) -> None:
        start_barrier.wait(timeout=2)
        api_module._persist_isyatirim_basic_summary_if_changed(items)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(persist_once, range(8)))

    assert len(upsert_attempts) == 1


def test_index_weight_inputs_use_isyatirim_fdpo_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "_latest_share_count_from_kap_cache", lambda _symbol: None)
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "ASTOR": {
                "shares_outstanding": 998_000_000.0,
                "fdpo": 0.427,
            },
        },
    )

    payload = api_module._index_weight_inputs_for_symbol("ASTOR")

    assert payload["shares_outstanding"] == 998_000_000.0
    assert payload["fdpo"] == 0.427
    assert payload["weight_coefficient"] == 1.0


def test_bist_index_report_parser_and_fallback_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    csv_payload = """BILESEN KODU;BULTEN_ADI;ENDEKS KODU;ENDEKS ADI;ENDEKS INGILIZCE ADI;TARIH(GG/AA/YYYY)
CONSTITUENT CODE;CONSTITUENT NAME;INDEX CODE;INDEX NAME IN TURKISH;INDEX NAME IN ENGLISH;DATE(DD/MM/YYYY)
AAA.E;AAA TEST;XUTUM;BIST TUM;BIST ALL SHARES;29/04/2026
BBB.E;BBB TEST;XUTUM;BIST TUM;BIST ALL SHARES;29/04/2026
AAA.E;AAA TEST;XU100;BIST 100;BIST 100;29/04/2026
BNK.E;BANK TEST;XBANK;BIST BANKA;BIST BANKS;29/04/2026
"""
    parsed = kap_service_module.parse_bist_index_report_csv(csv_payload)

    assert parsed["XUTUM"]["symbols"] == ["AAA", "BBB"]
    assert parsed["XUTUM"]["source_date"] == "29/04/2026"
    assert parsed["XU100"]["symbols"] == ["AAA"]
    assert parsed["XBANK"]["symbols"] == ["BNK"]

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return csv_payload.encode("utf-8-sig")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    sector_payload = kap_service_module.get_bist_index_universe("XBANK", force_refresh=True)

    assert sector_payload["index"] == "XBANK"
    assert sector_payload["symbols"] == ["BNK"]
    assert sector_payload["fallback_used"] is False
    kap_service_module._BIST_UNIVERSE_CACHE.clear()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("shared BIST universe cache should satisfy request"),
    )
    cached_sector = kap_service_module.get_bist_index_universe("XBANK")
    assert cached_sector["symbols"] == ["BNK"]
    assert cached_sector["cache_hit"] is True

    def fail_urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    payload = kap_service_module.get_bist_index_universe("XUTUM", force_refresh=True)

    assert payload["index"] == "XUTUM"
    assert payload["fallback_used"] is True
    assert payload["source"] == "borsa_istanbul_csv_fallback_snapshot"
    assert payload["count"] == len(payload["symbols"])
    assert "AEFES" in payload["symbols"]


def test_market_stocks_payload_extends_rows_and_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    price_calls: List[tuple[str, ...]] = []

    monkeypatch.setattr("app.kap_service.get_bist100_companies", lambda: ["A1CAP", "AEFES"])
    monkeypatch.setattr("app.kap_service.get_bist30_companies", lambda: ["AEFES"])
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda index, force_refresh=False: {
            "index": str(index).upper(),
            "symbols": ["AEFES"] if str(index).upper() == "XU030" else ["A1CAP", "AEFES"],
            "count": 1 if str(index).upper() == "XU030" else 2,
            "source": "test",
            "source_url": "test://bist",
            "source_date": "29/04/2026",
            "fetched_at": "2026-04-29T12:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda *_args: pytest.fail("market stock rows should use the SQL snapshot map"),
    )
    monkeypatch.setattr(
        api_module,
        "get_financial_snapshots_map",
        lambda _processed_dir, _symbols: {
            "A1CAP": {
                "latest_quarter": "2026Q1",
                "has_kap_cache": True,
                "shares_outstanding": 100_000_000.0,
            },
            "AEFES": {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
        },
    )

    def fake_prices(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
        price_calls.append(tuple(symbols))
        return {
            "A1CAP": {
                "price": 14.18,
                "currency": "TRY",
                "change": -0.21,
                "change_pct": -1.46,
                "volume": 108750000.0,
                "as_of": "2026-04-24T12:00:00+00:00",
            },
            "AEFES": {
                "price": 19.0,
                "currency": "TRY",
                "change": 0.2,
                "change_pct": 1.06,
                "volume": None,
                "as_of": "2026-04-24T12:00:00+00:00",
            },
        }

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_prices)
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "A1CAP": {"market_cap": 999_000_000_000.0, "shares_outstanding": 1_000_000.0},
            "AEFES": {"market_cap": 38_000_000_000.0, "shares_outstanding": 2_000_000_000.0},
        },
    )

    def fake_return_bases(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            if symbol == "A1CAP":
                result[symbol] = {
                    "base_1w": 14.0,
                    "base_1m": 13.0,
                    "base_3m": 12.0,
                    "base_6m": 10.0,
                    "base_ytd": 11.0,
                    "base_1y": 7.0,
                }
            elif symbol == "XU100":
                result[symbol] = {
                    "base_1w": 100.0,
                    "base_1m": 95.0,
                    "base_3m": 90.0,
                    "base_6m": 80.0,
                    "base_ytd": 85.0,
                    "base_1y": 70.0,
                    "latest_close": 110.0,
                    "as_of": "2026-04-24T12:00:00+00:00",
                }
            elif symbol == "XU030":
                result[symbol] = {
                    "base_1w": 200.0,
                    "base_1m": 190.0,
                    "base_3m": 180.0,
                    "base_6m": 160.0,
                    "base_ytd": 170.0,
                    "base_1y": 140.0,
                    "latest_close": 220.0,
                    "as_of": "2026-04-24T12:00:00+00:00",
                }
        return result

    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", fake_return_bases)
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda index: {
            "symbol": index,
            "label": f"BIST {index}",
            "price": 110.0 if index == "XU100" else 220.0,
            "prev_close": 110.0 if index == "XU100" else 220.0,
            "change": 0.0,
            "change_pct": 0.0,
            "as_of": "2026-04-24T12:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"market stock rows should not resolve logos over KAP: {symbol}"),
    )
    api_module._STOCKS_CACHE.clear()

    client = TestClient(app)
    first = client.get("/market/stocks?index=XU100")
    second = client.get("/market/stocks?index=XU100")
    third = client.get("/market/stocks?index=XU030")
    fourth = client.get("/market/stocks?index=XU030")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert fourth.status_code == 200
    assert price_calls == [("A1CAP", "AEFES"), ("AEFES",)]

    first_payload = first.json()
    assert first_payload["index"] == "XU100"
    assert first_payload["benchmarks"]["XU100"]["return_1w_pct"] == 10.0
    assert first_payload["benchmarks"]["XU030"]["return_1w_pct"] == 10.0

    rows = first_payload["rows"]
    a1cap = rows[0]
    assert a1cap["company"] == "A1CAP"
    assert a1cap["volume"] == 108750000.0
    assert a1cap["market_cap"] == 1418000000.0
    assert a1cap["logo_url"] is None
    assert a1cap["logo_source"] is None
    assert a1cap["return_1w_pct"] == 1.29
    assert a1cap["return_1y_pct"] == 102.57
    assert rows[1]["market_cap"] == 38000000000.0
    assert rows[1]["return_1w_pct"] is None
    assert third.json()["index"] == "XU030"
    assert [row["company"] for row in third.json()["rows"]] == ["AEFES"]


def test_xutum_market_stocks_uses_cache_only_side_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.kap_service.get_bist_all_companies", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda _index, force_refresh=False: {
            "index": "XUTUM",
            "symbols": ["AAA", "BBB"],
            "count": 2,
            "source": "test",
            "source_url": "test://bist",
            "source_date": "29/04/2026",
            "fetched_at": "2026-04-29T12:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})

    def fake_price_map(symbols: List[str], *, index_name: str = "XU100") -> Dict[str, Dict[str, Any]]:
        assert index_name == "XUTUM"
        return {
            symbol: {
                "price": 10.0,
                "currency": "TRY",
                "change": 0.1,
                "change_pct": 1.0,
                "volume": 1000.0,
                "as_of": "2026-04-29T12:00:00+00:00",
            }
            for symbol in symbols
        }

    monkeypatch.setattr(api_module, "_fetch_market_price_map", fake_price_map)

    def fake_return_bases(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        assert symbols == ["XUTUM", "XU100", "XU030"]
        return {}

    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", fake_return_bases)
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"XUTUM should not resolve logos over KAP: {symbol}"),
    )

    payload = api_module._market_stocks_payload(index_name="XUTUM", force_refresh=True)

    assert payload["index"] == "XUTUM"
    assert [row["company"] for row in payload["rows"]] == ["AAA", "BBB"]
    assert payload["rows"][0]["logo_url"] is None
    assert payload["rows"][0]["return_1w_pct"] is None
    assert payload["universe"]["fallback_used"] is False


def test_legacy_xu030_payload_does_not_resolve_kap_logos(monkeypatch: pytest.MonkeyPatch) -> None:
    api_module._XU030_CACHE.clear()
    monkeypatch.setattr("app.kap_service.get_bist30_companies", lambda: ["AKBNK"])
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols: {
            "AKBNK": {
                "price": 64.0,
                "currency": "TRY",
                "change": 1.0,
                "change_pct": 1.59,
                "as_of": "2026-04-25T11:00:00+00:00",
            }
        },
    )
    monkeypatch.setattr(api_module, "_fill_prices_via_yahoo", lambda _symbols, base_map: base_map)
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(
        api_module,
        "_kap_logo_payload_for_symbol",
        lambda symbol: pytest.fail(f"legacy XU030 should not resolve logos over KAP: {symbol}"),
    )

    payload = api_module._xu030_payload()

    assert payload["index"] == "XU030"
    assert payload["rows"][0]["company"] == "AKBNK"
    assert payload["rows"][0]["logo_url"] is None
    assert payload["rows"][0]["logo_source"] is None


def test_infoyatirim_stock_page_fallback_populates_xu100_rows_and_index_impact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_html = """
    <table><tbody id="tableBody">
      <tr data-symbol="A1CAP">
        <td>A1 CAPITAL</td>
        <td class="price" data-val="14.18">14.18</td>
        <td class="change" data-val="-0.21">-0.21</td>
        <td class="percent" data-val="-1.46">-1.46 %</td>
        <td>108.750.000</td>
      </tr>
    </tbody></table>
    """

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return list_html.encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr("app.kap_service.get_bist100_companies", lambda: ["EGEEN"])
    monkeypatch.setattr(
        api_module,
        "_load_cached_kap_market_metadata",
        lambda _cache_dir, _symbol: {"latest_quarter": None, "has_kap_cache": False, "shares_outstanding": None},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_infoyatirim_stock_page_quote",
        lambda symbol: {
            "price": 6665.0,
            "currency": "TRY",
            "change": -17.35,
            "change_pct": -0.26,
            "volume": 212_043_600.0,
            "market_cap": 20_994_800_000.0,
            "market_state": "",
            "as_of": "2026-04-25T09:00:00+00:00",
        }
        if symbol == "EGEEN"
        else {},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_isyatirim_basic_summary_map",
        lambda: {
            "EGEEN": {
                "market_cap": 20_994_800_000.0,
                "shares_outstanding": 3_200_000.0,
                "fdpo": 0.358,
            },
        },
    )
    monkeypatch.setattr(api_module, "_fetch_stock_return_bases_bulk", lambda _symbols: {})
    monkeypatch.setattr(api_module, "_latest_share_count_from_kap_cache", lambda _symbol: None)
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda _index_code: {
            "symbol": "XU100",
            "label": "BIST 100",
            "yahoo_symbol": "XU100.IS",
            "price": 1000.0,
            "prev_close": 1002.6,
            "change": -2.6,
            "change_pct": -0.26,
            "high": 1010.0,
            "low": 990.0,
            "volume": 1_000_000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-04-25T09:00:00+00:00",
            "error": None,
        },
    )
    monkeypatch.setattr(api_module, "_fetch_index_return_bases", lambda _index_code: {})
    monkeypatch.setattr(api_module, "_index_intraday_payload", lambda _index_code: {"line_points": []})

    client = TestClient(app)
    stocks_response = client.get("/market/stocks?index=XU100&refresh=true")
    index_response = client.get("/market/indices/XU100?refresh=true")

    assert stocks_response.status_code == 200
    stock_row = stocks_response.json()["rows"][0]
    assert stock_row["company"] == "EGEEN"
    assert stock_row["price"] == 6665.0
    assert stock_row["change_pct"] == -0.26
    assert stock_row["volume"] == 212_043_600.0

    assert index_response.status_code == 200
    index_row = index_response.json()["constituents"][0]
    assert index_row["symbol"] == "EGEEN"
    assert index_row["weight_pct"] == 100.0
    assert index_row["point_effect"] == -2.6


def test_market_stocks_rejects_unknown_index() -> None:
    client = TestClient(app)
    response = client.get("/market/stocks?index=XU050")

    assert response.status_code == 400
    assert "Desteklenmeyen endeks" in response.json()["detail"]


def test_market_stocks_benchmarks_uses_live_quote_and_calculates_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bases = {
        "XUTUM": {
            "base_1w": 9500.0,
            "base_1m": 9000.0,
            "base_3m": 8500.0,
            "base_6m": 8000.0,
            "base_ytd": 8800.0,
            "base_1y": 7000.0,
            "latest_close": 10000.0,
            "as_of": "2026-11-06T18:10:00+03:00",
        },
        "XU100": {
            "base_1w": 9500.0,
            "base_1m": 9000.0,
            "base_3m": 8500.0,
            "base_6m": 8000.0,
            "base_ytd": 8800.0,
            "base_1y": 7000.0,
            "latest_close": 10000.0,
            "as_of": "2026-11-06T18:10:00+03:00",
        },
        "XU030": {
            "base_1w": 10500.0,
            "base_1m": 10000.0,
            "base_3m": 9500.0,
            "base_6m": 9000.0,
            "base_ytd": 9800.0,
            "base_1y": 8000.0,
            "latest_close": 11000.0,
            "as_of": "2026-11-06T18:10:00+03:00",
        },
    }
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_return_bases_bulk",
        lambda symbols: {s: fake_bases[s] for s in symbols if s in fake_bases},
    )

    def fake_live_quote(index_name: str) -> Dict[str, Any]:
        if index_name == "XU100":
            return {
                "symbol": "XU100",
                "label": "BIST 100",
                "yahoo_symbol": "XU100.IS",
                "price": 10500.0,
                "prev_close": 10000.0,
                "change": 500.0,
                "change_pct": 5.0,
                "high": 10550.0,
                "low": 9950.0,
                "volume": 120_000_000.0,
                "currency": "TRY",
                "market_state": "REGULAR",
                "as_of": "2026-11-07T14:30:00+03:00",
                "error": None,
            }
        return {
            "symbol": index_name,
            "label": f"BIST {index_name}",
            "yahoo_symbol": f"{index_name}.IS",
            "price": 11500.0 if index_name == "XU030" else 10200.0,
            "prev_close": 11000.0 if index_name == "XU030" else 10000.0,
            "change": 500.0 if index_name == "XU030" else 200.0,
            "change_pct": 4.55 if index_name == "XU030" else 2.0,
            "as_of": "2026-11-07T14:30:00+03:00",
            "error": None,
        }

    monkeypatch.setattr(api_module, "_fetch_index_quote", fake_live_quote)

    benchmarks = api_module._market_stock_benchmarks()

    assert "XU100" in benchmarks
    xu100 = benchmarks["XU100"]
    # Check that price, prev_close, change, change_pct fields are present
    assert xu100["price"] == 10500.0
    assert xu100["prev_close"] == 10000.0
    assert xu100["change"] == 500.0
    assert xu100["change_pct"] == 5.0
    assert xu100["as_of"] == "2026-11-07T14:30:00+03:00"

    # Verify non-zero intraday returns calculated from live quote 10500.0 instead of latest_close 10000.0
    # (10500 - 9500) / 9500 * 100 = 10.53%
    # With latest_close (10000 - 9500) / 9500 * 100 = 5.26%
    assert xu100["return_1w_pct"] == 10.53
    assert xu100["return_1m_pct"] == 16.67
    assert xu100["return_1y_pct"] == 50.0


def test_market_stocks_benchmarks_fallback_to_latest_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bases = {
        "XUTUM": {
            "base_1w": 9500.0,
            "latest_close": 10000.0,
            "as_of": "2026-11-06T18:10:00+03:00",
        },
        "XU100": {
            "base_1w": 9500.0,
            "latest_close": 10000.0,
            "as_of": "2026-11-06T18:10:00+03:00",
        },
        "XU030": {
            "base_1w": 10500.0,
            "latest_close": 11000.0,
            "as_of": "2026-11-06T18:10:00+03:00",
        },
    }
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_return_bases_bulk",
        lambda symbols: {s: fake_bases[s] for s in symbols if s in fake_bases},
    )

    # Simulate upstream quote failure (empty quote or non-positive price)
    monkeypatch.setattr(
        api_module,
        "_fetch_index_quote",
        lambda index_name: {
            "symbol": index_name,
            "price": None,
            "prev_close": None,
            "change": None,
            "change_pct": None,
            "error": "upstream_timeout",
        },
    )

    benchmarks = api_module._market_stock_benchmarks()

    assert "XU100" in benchmarks
    xu100 = benchmarks["XU100"]
    # Fallback price is latest_close (10000.0), prev_close is latest_close
    assert xu100["price"] == 10000.0
    assert xu100["prev_close"] == 10000.0
    assert xu100["change"] is None
    assert xu100["change_pct"] is None
    # Returns calculated from latest_close: (10000 - 9500) / 9500 * 100 = 5.26%
    assert xu100["return_1w_pct"] == 5.26
    assert xu100["as_of"] == "2026-11-06T18:10:00+03:00"


def test_market_watch_and_indices_consume_unified_index_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_module._WATCH_CACHE.clear()
    api_module._MARKET_INDEX_QUOTE_CACHE.clear()
    api_module._MARKET_INDICES_CACHE.clear()

    quote_calls = []

    def tracking_fetch_index_quote(index_code: str) -> Dict[str, Any]:
        quote_calls.append(index_code)
        return {
            "symbol": index_code,
            "label": f"BIST {index_code}",
            "yahoo_symbol": f"{index_code}.IS",
            "price": 10555.0,
            "prev_close": 10400.0,
            "change": 155.0,
            "change_pct": 1.49,
            "high": 10600.0,
            "low": 10350.0,
            "volume": 50_000_000.0,
            "currency": "TRY",
            "market_state": "REGULAR",
            "as_of": "2026-11-07T15:00:00+03:00",
            "error": None,
        }

    monkeypatch.setattr(api_module, "_fetch_index_quote", tracking_fetch_index_quote)
    monkeypatch.setattr(api_module, "_fetch_index_return_bases", lambda _code: {})
    monkeypatch.setattr(api_module, "_market_fx_payload", lambda: {"items": []})
    monkeypatch.setattr(api_module, "_market_commodities_payload", lambda: {"items": []})

    client = TestClient(app)

    # 1. Call /market/watch
    watch_res = client.get("/market/watch?refresh=true")
    assert watch_res.status_code == 200
    watch_indices = {r["symbol"]: r for r in watch_res.json()["sections"]["indices"]}
    assert "XU100" in watch_indices
    assert watch_indices["XU100"]["price"] == 10555.0
    assert watch_indices["XU100"]["change"] == 155.0
    assert watch_indices["XU100"]["change_pct"] == 1.49

    # 2. Call /market/indices
    indices_res = client.get("/market/indices?refresh=true")
    assert indices_res.status_code == 200
    indices_rows = {r["symbol"]: r for r in indices_res.json()["rows"]}
    assert "XU100" in indices_rows
    assert indices_rows["XU100"]["price"] == 10555.0
    assert indices_rows["XU100"]["change"] == 155.0
    assert indices_rows["XU100"]["change_pct"] == 1.49

    # Both endpoints received identical quote data
    assert watch_indices["XU100"]["price"] == indices_rows["XU100"]["price"]
    assert watch_indices["XU100"]["change"] == indices_rows["XU100"]["change"]
    assert watch_indices["XU100"]["change_pct"] == indices_rows["XU100"]["change_pct"]


def test_market_stocks_stream_emits_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_market_stocks_payload",
        lambda index_name="XUTUM": {
            "index": index_name,
            "as_of": "2026-11-07T15:00:00+03:00",
            "rows": [
                {
                    "symbol": "THYAO",
                    "price": 310.0,
                    "change": 5.0,
                    "volume": 1000.0,
                    "as_of": "2026-11-07T15:00:00+03:00",
                }
            ],
            "benchmarks": {
                "XU100": {
                    "price": 10500.0,
                    "prev_close": 10000.0,
                    "change": 500.0,
                    "change_pct": 5.0,
                    "return_1w_pct": 10.53,
                    "as_of": "2026-11-07T15:00:00+03:00",
                }
            },
        },
    )

    client = TestClient(app)
    response = client.get("/market/stream?index=XUTUM&once=true")
    assert response.status_code == 200

    content = response.text
    assert "event: stocks_delta" in content
    # Parse json line after data:
    for line in content.split("\n"):
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            assert "benchmarks" in payload
            assert payload["benchmarks"]["XU100"]["price"] == 10500.0
            assert payload["benchmarks"]["XU100"]["change_pct"] == 5.0
            break
    else:
        pytest.fail("stocks_delta event data not found in stream response")


def test_fetch_stock_return_bases_bulk_batches_spark_and_caches_over_512(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: List[str] = []
    timestamps = [
        datetime(2025, 11, 10, tzinfo=timezone.utc).timestamp(),
        datetime(2026, 11, 6, tzinfo=timezone.utc).timestamp(),
        datetime(2026, 11, 13, tzinfo=timezone.utc).timestamp(),
    ]

    class FakeResp:
        def __init__(self, body: Dict[str, Any]) -> None:
            self._body = json.dumps(body).encode("utf-8")

        def __enter__(self) -> "FakeResp":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    def fake_urlopen(request: Any, **_kwargs: Any) -> FakeResp:
        url = request.full_url
        requested_urls.append(url)
        if "/v7/finance/spark" in url:
            return FakeResp({
                "spark": {
                    "result": [
                        {
                            "symbol": "AAA.IS",
                            "response": [{
                                "timestamp": timestamps,
                                "indicators": {"quote": [{"close": [10.0, 20.0, 25.0]}]},
                            }],
                        }
                    ]
                }
            })
        return FakeResp({
            "chart": {
                "result": [{
                    "timestamp": timestamps,
                    "indicators": {"quote": [{"close": [50.0, 80.0, 100.0], "high": [50.0, 80.0, 100.0], "low": [50.0, 80.0, 100.0]}]},
                }]
            }
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(api_module._market_database, "read_json_cache", lambda *_a, **_kw: pytest.fail("should not read Supabase"))
    monkeypatch.setattr(api_module._market_database, "write_json_cache", lambda *_a, **_kw: pytest.fail("should not write Supabase"))

    assert api_module._STOCK_RETURN_BASE_CACHE.max_entries >= 1024
    res = api_module._fetch_stock_return_bases_bulk(["AAA", "BBB"])

    assert any("/v7/finance/spark" in u for u in requested_urls)
    assert res["AAA"]["base_1w"] == 20.0
    assert res["AAA"]["latest_close"] == 25.0
    # BBB was absent from Spark response and cleanly fell back to v8 chart
    assert res["BBB"]["base_1w"] == 80.0
    assert res["BBB"]["latest_close"] == 100.0

    # Subsequent cached XUTUM lookup hits bulk cache without network or Supabase calls
    requested_urls.clear()
    cached = api_module._cached_stock_return_bases_bulk(["AAA", "BBB"])
    assert requested_urls == []
    assert cached["AAA"]["base_1w"] == 20.0
    assert cached["BBB"]["base_1w"] == 80.0


def test_warm_stock_return_bases_sync_populates_and_refreshes_xutum_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.kap_service.get_bist_all_companies", lambda: ["AAA"])
    monkeypatch.setattr(
        "app.kap_service.get_bist_index_universe",
        lambda _idx, force_refresh=False: {
            "index": "XUTUM",
            "symbols": ["AAA"],
            "count": 1,
            "source": "test",
            "source_url": "test://bist",
            "source_date": "14/11/2026",
            "fetched_at": "2026-11-14T12:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(api_module, "_fetch_isyatirim_basic_summary_map", lambda: {})
    monkeypatch.setattr(
        api_module,
        "_fetch_market_price_map",
        lambda symbols, index_name="XUTUM": {
            s: {"price": 120.0, "currency": "TRY", "change": 2.0, "change_pct": 1.69, "volume": 1000.0, "as_of": "2026-11-14T12:00:00+00:00"}
            for s in symbols
        },
    )
    monkeypatch.setattr(api_module, "_market_stock_benchmarks", lambda: {})
    monkeypatch.setattr(api_module, "get_financial_snapshots_map", lambda *_a, **_kw: {})

    # First XUTUM request is cold for return bases
    cold_payload = api_module._market_stocks_payload(index_name="XUTUM")
    assert cold_payload["rows"][0]["return_1w_pct"] is None

    # Inject Spark-style bases during warmup
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_return_bases_spark_batch",
        lambda _syms: {},
    )
    monkeypatch.setattr(
        api_module,
        "_fetch_stock_return_bases",
        lambda sym: {"base_1w": 100.0, "base_1m": 80.0, "latest_close": 120.0} if sym == "AAA" else {},
    )

    summary = api_module._warm_stock_return_bases_sync(["AAA"])
    assert summary == {"requested": 1, "populated": 1}

    # Cached XUTUM payload was refreshed in-place with populated returns
    warmed_payload = api_module._market_stocks_payload(index_name="XUTUM")
    assert warmed_payload["rows"][0]["return_1w_pct"] == 20.0
    assert warmed_payload["rows"][0]["return_1m_pct"] == 50.0
