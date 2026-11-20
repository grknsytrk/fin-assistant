from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from app import fund_service


@pytest.fixture(autouse=True)
def _isolated_price_points_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(fund_service, "FUND_PRICE_POINTS_CACHE_TTL_SECONDS", 60)
    fund_service._reset_fund_price_points_cache_for_tests()
    yield
    fund_service._reset_fund_price_points_cache_for_tests()


def test_price_points_cache_normalizes_key_and_returns_copy(tmp_path, monkeypatch) -> None:
    calls = []

    def load(_processed_dir, code, *, start_date, end_date, normalized_sources):
        calls.append((code, start_date, end_date, normalized_sources))
        return [{"fund_code": code, "metadata": {"label": "original"}}]

    monkeypatch.setattr(fund_service, "_read_fund_price_points_uncached", load)

    first = fund_service.read_fund_price_points(
        tmp_path,
        " tly ",
        sources=["TEFAS", "tefas"],
    )
    first[0]["metadata"]["label"] = "mutated by caller"
    second = fund_service.read_fund_price_points(tmp_path, "TLY", sources=["tefas"])

    assert len(calls) == 1
    assert calls[0] == ("TLY", None, None, ("tefas",))
    assert second[0]["metadata"]["label"] == "original"


def test_price_points_cache_varies_by_date_and_source(tmp_path, monkeypatch) -> None:
    calls = []

    def load(_processed_dir, code, *, start_date, end_date, normalized_sources):
        calls.append((code, start_date, end_date, normalized_sources))
        return []

    monkeypatch.setattr(fund_service, "_read_fund_price_points_uncached", load)
    fund_service.read_fund_price_points(tmp_path, "TLY")
    fund_service.read_fund_price_points(
        tmp_path,
        "TLY",
        start_date=date(2026, 1, 1),
    )
    fund_service.read_fund_price_points(tmp_path, "TLY", sources=["fintables"])

    assert len(calls) == 3


def test_expired_price_points_entry_is_reloaded(tmp_path, monkeypatch) -> None:
    calls = 0

    def load(_processed_dir, code, *, start_date, end_date, normalized_sources):
        nonlocal calls
        calls += 1
        return [{"fund_code": code, "price": float(calls)}]

    monkeypatch.setattr(fund_service, "_read_fund_price_points_uncached", load)
    fund_service.read_fund_price_points(tmp_path, "TLY")
    with fund_service._FUND_PRICE_POINTS_CACHE_LOCK:
        key = next(iter(fund_service._FUND_PRICE_POINTS_CACHE))
        fund_service._FUND_PRICE_POINTS_CACHE[key].expires_at = 0

    refreshed = fund_service.read_fund_price_points(tmp_path, "TLY")

    assert calls == 2
    assert refreshed[0]["price"] == 2


def test_successful_price_write_invalidates_cached_history(tmp_path, monkeypatch) -> None:
    calls = []

    def load(_processed_dir, code, *, start_date, end_date, normalized_sources):
        calls.append(code)
        return [{"fund_code": code, "price": float(len(calls))}]

    monkeypatch.setattr(fund_service, "_read_fund_price_points_uncached", load)

    first = fund_service.read_fund_price_points(tmp_path, "TLY")
    assert first[0]["price"] == 1

    result = fund_service.upsert_fund_price_points(
        tmp_path,
        [{"fund_code": "TLY", "date": "2026-10-20", "price": 10.0}],
        source="tefasfon_funds",
    )
    assert result["upserted_count"] == 1

    second = fund_service.read_fund_price_points(tmp_path, "TLY")
    assert second[0]["price"] == 2
    assert len(calls) == 2


def test_concurrent_identical_misses_share_one_load(tmp_path, monkeypatch) -> None:
    calls = 0
    started = threading.Event()
    release = threading.Event()

    def load(_processed_dir, code, *, start_date, end_date, normalized_sources):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=3)
        return [{"fund_code": code, "price": 1.0}]

    monkeypatch.setattr(fund_service, "_read_fund_price_points_uncached", load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(fund_service.read_fund_price_points, tmp_path, "TLY")
        assert started.wait(timeout=3)
        second = pool.submit(fund_service.read_fund_price_points, tmp_path, "TLY")
        time.sleep(0.05)
        release.set()
        assert first.result(timeout=3) == second.result(timeout=3)

    assert calls == 1


def test_oversized_result_is_not_cached(tmp_path, monkeypatch) -> None:
    calls = 0

    def load(_processed_dir, code, *, start_date, end_date, normalized_sources):
        nonlocal calls
        calls += 1
        return [{"fund_code": code, "large": "payload"}]

    monkeypatch.setattr(fund_service, "_read_fund_price_points_uncached", load)
    monkeypatch.setattr(fund_service, "FUND_PRICE_POINTS_CACHE_MAX_ENTRY_BYTES", 1)

    fund_service.read_fund_price_points(tmp_path, "TLY")
    fund_service.read_fund_price_points(tmp_path, "TLY")

    assert calls == 2
    assert fund_service._FUND_PRICE_POINTS_CACHE_METRICS["oversize_skips"] == 2


def test_failed_price_points_load_is_not_cached(tmp_path, monkeypatch) -> None:
    calls = 0

    def load(_processed_dir, code, *, start_date, end_date, normalized_sources):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary database failure")
        return [{"fund_code": code, "price": 1.0}]

    monkeypatch.setattr(fund_service, "_read_fund_price_points_uncached", load)
    with pytest.raises(RuntimeError, match="temporary database failure"):
        fund_service.read_fund_price_points(tmp_path, "TLY")

    assert fund_service.read_fund_price_points(tmp_path, "TLY")[0]["price"] == 1.0
    assert calls == 2


def test_price_points_cache_respects_entry_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fund_service, "FUND_PRICE_POINTS_CACHE_MAX_ENTRIES", 1)
    monkeypatch.setattr(
        fund_service,
        "_read_fund_price_points_uncached",
        lambda _dir, code, **_kwargs: [{"fund_code": code}],
    )

    fund_service.read_fund_price_points(tmp_path, "AAA")
    fund_service.read_fund_price_points(tmp_path, "BBB")

    assert len(fund_service._FUND_PRICE_POINTS_CACHE) == 1
    assert next(iter(fund_service._FUND_PRICE_POINTS_CACHE))[1] == "BBB"
    assert fund_service._FUND_PRICE_POINTS_CACHE_METRICS["evictions"] == 1


def test_noop_price_point_upsert_does_not_bump_listing_revision(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager

    next_rowcount = 0
    bump_calls = 0
    invalidate_calls = 0

    class _FakeCursor:
        def __init__(self, rowcount: int) -> None:
            self.rowcount = rowcount

    class _FakeConn:
        def execute(self, _sql, _params=()):
            return _FakeCursor(next_rowcount)

        def commit(self) -> None:
            pass

    @contextmanager
    def fake_connect(_processed_dir):
        yield _FakeConn()

    monkeypatch.setattr(fund_service, "database_enabled", lambda: True)
    monkeypatch.setattr(fund_service, "_connect_fund_prices_db", fake_connect)
    monkeypatch.setattr(fund_service, "_bump_fund_listing_price_revision", lambda _conn: _inc("bump"))
    monkeypatch.setattr(fund_service, "_invalidate_fund_price_memory_caches", lambda: _inc("inv"))

    def _inc(kind: str) -> None:
        nonlocal bump_calls, invalidate_calls
        if kind == "bump":
            bump_calls += 1
        else:
            invalidate_calls += 1

    sample = [{"fund_code": "TLY", "date": "2026-10-20", "price": 10.0}]
    noop_result = fund_service.upsert_fund_price_points(tmp_path, sample, source="tefasfon_funds")
    assert noop_result["upserted_count"] == 1
    assert bump_calls == 0
    assert invalidate_calls == 0

    next_rowcount = 1
    changed_result = fund_service.upsert_fund_price_points(tmp_path, sample, source="tefasfon_funds")
    assert changed_result["upserted_count"] == 1
    assert bump_calls == 1
    assert invalidate_calls == 1


def test_read_fund_price_points_range_postgres_caches_and_omits_metadata(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager

    price_gen = 4
    queries: list[tuple[str, list]] = []
    fail_next = False

    class _FakeCursor:
        def __init__(self, rowcount: int = 1) -> None:
            self.rowcount = rowcount

    class _FakePgConn:
        is_postgres = True

        def execute(self, sql: str, params=()):
            normalized_sql = " ".join(sql.split())
            if normalized_sql.startswith("INSERT INTO fund_prices"):
                return _FakeCursor(1)
            if "INSERT INTO fund_price_warnings" in normalized_sql:
                return _FakeCursor(0)
            if fail_next:
                raise RuntimeError("transient db error")
            queries.append((normalized_sql, list(params)))
            return [
                {
                    "fund_code": "TLY",
                    "date": "2026-04-01",
                    "source": "tefasfon_funds",
                    "price": 10.5,
                    "daily_return": 1.2,
                    "aum": 2_000_000.0,
                    "investor_count": 500,
                    "share_count": 190_000.0,
                    "fetched_at": "2026-04-01T18:00:00+00:00",
                    "updated_at": "2026-04-01T18:00:00+00:00",
                }
            ]

        def commit(self) -> None:
            pass

    @contextmanager
    def fake_connect(_processed_dir):
        yield _FakePgConn()

    monkeypatch.setattr(fund_service, "database_enabled", lambda: True)
    monkeypatch.setattr(fund_service, "_connect_fund_prices_db", fake_connect)
    monkeypatch.setattr(
        fund_service,
        "fund_listing_revision",
        lambda _p: {"catalog_generation": 1, "price_generation": price_gen},
    )
    monkeypatch.setattr(fund_service, "_bump_fund_listing_price_revision", lambda _conn: None)

    fail_next = True
    assert (
        fund_service.read_fund_price_points_range(
            tmp_path, "TLY", start_date=date(2026, 4, 1), end_date=date(2026, 4, 10)
        )
        == []
    )
    fail_next = False

    first = fund_service.read_fund_price_points_range(
        tmp_path, "TLY", start_date=date(2026, 4, 1), end_date=date(2026, 4, 10)
    )
    second = fund_service.read_fund_price_points_range(
        tmp_path, "TLY", start_date=date(2026, 4, 1), end_date=date(2026, 4, 10)
    )
    assert len(queries) == 1
    sql, params = queries[0]
    select_part = "SELECT" + sql.split("SELECT", 1)[1]
    assert "SELECT DISTINCT ON (date) " in select_part
    assert "metadata_json" not in select_part
    assert len(params) == sql.count("?")
    assert first == second
    assert first[0]["aum"] == 2_000_000.0
    assert "tefas_open" not in first[0]

    fund_service.upsert_fund_price_points(
        tmp_path,
        [{"fund_code": "TLY", "date": "2026-04-02", "price": 10.8}],
        source="tefasfon_funds",
    )
    price_gen = 5
    fund_service.read_fund_price_points_range(
        tmp_path, "TLY", start_date=date(2026, 4, 1), end_date=date(2026, 4, 10)
    )
    assert len(queries) == 2


def test_preferred_and_uncached_postgres_queries_behavior_and_6h_ttl(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager
    from app.funds import price_store

    queries: list[tuple[str, list]] = []

    class _FakePgConn:
        is_postgres = True

        def execute(self, sql: str, params=()):
            normalized_sql = " ".join(sql.split())
            queries.append((normalized_sql, list(params)))
            return [
                {
                    "fund_code": "GEN",
                    "date": "2026-04-01",
                    "source": "fintables_udf_history",
                    "price": 100.0,
                    "daily_return": None,
                    "aum": None,
                    "investor_count": None,
                    "share_count": None,
                    "metadata_json": '{"tefas_open": true}',
                    "fetched_at": "2026-04-01T18:00:00+00:00",
                    "updated_at": "2026-04-01T18:00:00+00:00",
                },
                {
                    "fund_code": "GEN",
                    "date": "2026-04-01",
                    "source": "tefasfon_funds",
                    "price": 101.0,
                    "daily_return": 1.0,
                    "aum": 3_500_000.0,
                    "investor_count": 77,
                    "share_count": 35_000.0,
                    "metadata_json": '{"tefas_open": true}',
                    "fetched_at": "2026-04-01T18:00:00+00:00",
                    "updated_at": "2026-04-01T18:00:00+00:00",
                },
            ]

    @contextmanager
    def fake_connect(_processed_dir):
        yield _FakePgConn()

    monkeypatch.setattr(fund_service, "database_enabled", lambda: True)
    monkeypatch.setattr(fund_service, "FUND_PRICE_POINTS_CACHE_TTL_SECONDS", 3600)
    monkeypatch.setattr(fund_service, "_connect_fund_prices_db", fake_connect)
    monkeypatch.setattr(
        fund_service,
        "fund_listing_revision",
        lambda _p: {"catalog_generation": 1, "price_generation": 9},
    )

    preferred = fund_service._read_preferred_fund_price_points(tmp_path, "GEN")
    assert preferred[0]["price"] == 100.0
    assert preferred[0]["aum"] == 3_500_000.0
    assert preferred[0]["investor_count"] == 77
    assert "tefas_open" not in preferred[0]
    pref_sql, pref_params = queries[0]
    assert "SELECT DISTINCT ON (date, CASE WHEN source IN" in pref_sql
    assert "metadata_json" not in pref_sql
    assert len(pref_params) == pref_sql.count("?")

    cached_key = next(iter(fund_service._FUND_PRICE_POINTS_CACHE))
    assert cached_key[0] == "postgres:gen=9"
    assert price_store._effective_fund_price_points_cache_ttl_seconds() >= 21600
    fund_service._read_preferred_fund_price_points(tmp_path, "GEN")
    assert len(queries) == 1

    uncached_points = fund_service.read_fund_price_points(tmp_path, "GEN")
    assert len(queries) == 2
    uncached_sql, uncached_params = queries[1]
    assert uncached_sql.startswith("SELECT DISTINCT ON (date) ")
    assert "metadata_json" in uncached_sql
    assert len(uncached_params) == uncached_sql.count("?")
    assert uncached_points[0]["tefas_open"] is True


def test_apply_daily_return_overrides_postgres_memo_survives_60s_and_fences_on_price_gen(tmp_path, monkeypatch) -> None:
    from app.funds import price_store

    fund_service._clear_all_fund_memory_caches()
    current_time = 10_000.0
    price_gen = 1
    override_calls = 0

    monkeypatch.setattr(fund_service, "database_enabled", lambda: True)
    monkeypatch.setattr(fund_service, "FUNDS_LATEST_HISTORY_OVERLAY_TTL_SECONDS", 900)
    monkeypatch.setattr(price_store.time, "time", lambda: current_time)
    monkeypatch.setattr(
        fund_service,
        "fund_listing_revision",
        lambda _p: {"catalog_generation": 1, "price_generation": price_gen},
    )

    def fake_overrides(_dir, rows, **_kwargs):
        nonlocal override_calls
        override_calls += 1
        return {
            (row["fund_code"], row["as_of"]): 1.5
            for row in rows
        }

    monkeypatch.setattr(fund_service, "_daily_return_overrides_from_price_history", fake_overrides)

    rows = [
        {"fund_code": f"F{i:02d}", "as_of": "2026-04-01", "price": 10.0, "daily_return": None}
        for i in range(12)
    ]
    first = price_store._apply_daily_return_overrides(tmp_path, rows, include_missing=True)
    assert override_calls == 1
    assert first[0]["daily_return"] == 1.5

    current_time += 120.0
    second = price_store._apply_daily_return_overrides(tmp_path, rows, include_missing=True)
    assert override_calls == 1
    assert second[0]["daily_return"] == 1.5

    price_gen = 2
    price_store._apply_daily_return_overrides(tmp_path, rows, include_missing=True)
    assert override_calls == 2
    fund_service._clear_all_fund_memory_caches()
