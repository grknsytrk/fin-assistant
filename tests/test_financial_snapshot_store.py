from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

from app import financial_snapshot_store as store


def _kap_payload(net_profit: float) -> dict:
    return {
        "ok": True,
        "company": "AAA",
        "stock_code": "AAA",
        "company_title": "AAA A.Ş.",
        "company_kind": "generic",
        "fetched_at": "2026-10-17T12:00:00+00:00",
        "quarters": [
            {
                "year": 2025,
                "period": 4,
                "quarter": "2025Q4",
                "metrics": {
                    "ozkaynaklar": 200.0,
                    "net_borc": 50.0,
                    "toplam_varliklar": 300.0,
                },
                "metrics_ytd": {
                    "net_kar": net_profit,
                    "satis_gelirleri": 400.0,
                    "favok": 80.0,
                    "serbest_nakit_akisi": 25.0,
                },
            }
        ],
    }


def test_sync_kap_cache_skips_unchanged_and_preserves_provider_fallback(tmp_path: Path) -> None:
    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    cache_dir = processed_dir / "kap_cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "AAA.json"
    cache_path.write_text(json.dumps(_kap_payload(40.0)), encoding="utf-8")

    store.upsert_provider_fundamentals_bulk(
        processed_dir,
        {
            "AAA": {
                "sector": "Enerji",
                "shares_outstanding": 999_000_000.0,
                "free_float_pct": 42.7,
                "fk": 9.5,
                "pd_dd": 1.8,
                "fd_favok": 6.2,
                "fd_satis": 1.3,
                "provider_period": "2026Q2",
            }
        },
    )

    first = store.sync_financial_snapshots_from_kap_cache(processed_dir)
    second = store.sync_financial_snapshots_from_kap_cache(processed_dir)
    snapshot = store.get_financial_snapshots_map(processed_dir, ["AAA"])["AAA"]

    assert first["upserted_count"] == 1
    assert second["upserted_count"] == 0
    assert second["skipped_unchanged_count"] == 1
    assert snapshot["has_kap_cache"] is True
    assert snapshot["latest_quarter"] == "2025Q4"
    assert snapshot["ttm_net_kar"] == 40.0
    assert snapshot["ttm_satis_gelirleri"] == 400.0
    assert snapshot["ttm_favok"] == 80.0
    assert snapshot["roe_pct"] == 20.0
    assert snapshot["favok_margin_pct"] == 20.0
    # KAP's missing share count must not erase İş Yatırım's fallback.
    assert snapshot["shares_outstanding"] == 999_000_000.0
    assert snapshot["sector"] == "Enerji"
    assert snapshot["provider_fk"] == 9.5
    assert snapshot["provider_period"] == "2026Q2"

    old_mtime = cache_path.stat().st_mtime_ns
    cache_path.write_text(json.dumps(_kap_payload(45.0)), encoding="utf-8")
    new_mtime = max(cache_path.stat().st_mtime_ns, old_mtime + 1_000_000_000)
    os.utime(cache_path, ns=(new_mtime, new_mtime))
    changed = store.sync_financial_snapshots_from_kap_cache(processed_dir)
    refreshed = store.get_financial_snapshots_map(processed_dir, ["AAA"])["AAA"]

    assert changed["upserted_count"] == 1
    assert refreshed["ttm_net_kar"] == 45.0
    assert refreshed["provider_fk"] == 9.5
    assert refreshed["sector"] == "Enerji"
    store.reset_financial_snapshot_store_for_tests()


def test_finite_number_accepts_postgres_decimal_values() -> None:
    assert store._finite_number(Decimal("12.75")) == 12.75
    assert store._finite_number(Decimal("NaN")) is None


def test_financial_snapshot_memory_cache_lives_1800_seconds_and_expires(tmp_path: Path, monkeypatch) -> None:
    assert store._SNAPSHOT_MEMORY_CACHE_TTL_SECONDS == 1800.0
    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Enerji"}})

    class FakeTime:
        now = 100.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.now

    monkeypatch.setattr(store, "time", FakeTime)
    original_connection = store._connection
    reads = {"count": 0}

    def counted_connection(path: Path):
        reads["count"] += 1
        return original_connection(path)

    monkeypatch.setattr(store, "_connection", counted_connection)
    assert "AAA" in store.get_financial_snapshots_map(processed_dir)
    FakeTime.now = 1899.0
    assert "AAA" in store.get_financial_snapshots_map(processed_dir)
    assert reads["count"] == 1

    FakeTime.now = 1901.0
    assert "AAA" in store.get_financial_snapshots_map(processed_dir)
    assert reads["count"] == 2
    store.reset_financial_snapshot_store_for_tests()


def test_env_float_parses_snapshot_cache_ttl_override(monkeypatch) -> None:
    assert store._env_float("RAGFIN_TEST_TTL", 1800.0) == 1800.0

    monkeypatch.setenv("RAGFIN_TEST_TTL", "60.0")
    assert store._env_float("RAGFIN_TEST_TTL", 1800.0) == 60.0

    monkeypatch.setenv("RAGFIN_TEST_TTL", "  120.5  ")
    assert store._env_float("RAGFIN_TEST_TTL", 1800.0) == 120.5

    monkeypatch.setenv("RAGFIN_TEST_TTL", "invalid-float")
    assert store._env_float("RAGFIN_TEST_TTL", 1800.0) == 1800.0

    monkeypatch.setenv("RAGFIN_TEST_TTL", "")
    assert store._env_float("RAGFIN_TEST_TTL", 1800.0) == 1800.0


def test_single_worker_invalidation_increments_version_and_clears_cache(tmp_path: Path) -> None:
    from app.cache import get_cache

    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Enerji"}})

    # Initial read populates local memory cache
    data = store.get_financial_snapshots_map(processed_dir, ["AAA"])
    assert data["AAA"]["sector"] == "Enerji"
    key = store._cache_key(processed_dir)
    assert key in store._SNAPSHOT_MEMORY_CACHE

    v1 = store.get_snapshot_cache_version()
    assert v1
    assert get_cache().get(store._SNAPSHOT_VERSION_KEY) == v1

    # Invalidate
    bumped_v = store._invalidate_memory_cache(processed_dir)
    assert bumped_v
    assert len(store._SNAPSHOT_MEMORY_CACHE) == 0

    v2 = store.get_snapshot_cache_version()
    assert v2 != v1
    assert v2 == bumped_v
    assert get_cache().get(store._SNAPSHOT_VERSION_KEY) == v2
    store.reset_financial_snapshot_store_for_tests()


def test_local_memory_cache_invalidated_when_redis_version_changes(tmp_path: Path) -> None:
    from app.cache import get_cache

    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Enerji"}})

    # Read into local cache
    initial = store.get_financial_snapshots_map(processed_dir, ["AAA"])
    assert initial["AAA"]["sector"] == "Enerji"

    # Directly mutate DB without invalidating local cache
    with store._connection(processed_dir) as conn:
        conn.execute("UPDATE company_financial_snapshots SET sector = 'Teknoloji' WHERE symbol = 'AAA'")
        conn.commit()

    # Still returns stale cache because version hasn't changed and TTL not expired
    cached = store.get_financial_snapshots_map(processed_dir, ["AAA"])
    assert cached["AAA"]["sector"] == "Enerji"

    # Simulate another worker bumping the shared Redis version
    new_version = "worker-bump-9999"
    get_cache().set(store._SNAPSHOT_VERSION_KEY, new_version, ttl_seconds=604800)

    # Next read detects version mismatch and reloads from DB
    refreshed = store.get_financial_snapshots_map(processed_dir, ["AAA"])
    assert refreshed["AAA"]["sector"] == "Teknoloji"

    # Memory cache now records the new version
    key = store._cache_key(processed_dir)
    assert store._SNAPSHOT_MEMORY_CACHE[key][1] == new_version
    store.reset_financial_snapshot_store_for_tests()


def test_version_bump_invalidates_cache_across_simulated_multi_worker_instances(tmp_path: Path, monkeypatch) -> None:
    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Enerji"}})

    # Worker 1 and Worker 2 each have independent in-memory caches
    worker1_cache: dict = {}
    worker2_cache: dict = {}

    # Worker 1 reads
    monkeypatch.setattr(store, "_SNAPSHOT_MEMORY_CACHE", worker1_cache)
    data1 = store.get_financial_snapshots_map(processed_dir, ["AAA"])
    assert data1["AAA"]["sector"] == "Enerji"
    assert len(worker1_cache) == 1

    # Worker 2 reads
    monkeypatch.setattr(store, "_SNAPSHOT_MEMORY_CACHE", worker2_cache)
    data2 = store.load_financial_snapshots_map(processed_dir, ["AAA"])
    assert data2["AAA"]["sector"] == "Enerji"
    assert len(worker2_cache) == 1

    # Worker 1 updates data and invalidates
    monkeypatch.setattr(store, "_SNAPSHOT_MEMORY_CACHE", worker1_cache)
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Savunma"}})
    assert len(worker1_cache) == 0  # Worker 1 cleared its own cache

    # Worker 2 still has its old in-memory cache with sector 'Enerji'
    assert len(worker2_cache) == 1
    key = store._cache_key(processed_dir)
    assert worker2_cache[key][2]["AAA"]["sector"] == "Enerji"

    # Worker 2 reads again using load_financial_snapshots_map / read_financial_snapshots_map
    monkeypatch.setattr(store, "_SNAPSHOT_MEMORY_CACHE", worker2_cache)
    fresh_worker2 = store.read_financial_snapshots_map(processed_dir, ["AAA"])

    # Worker 2 automatically detected version mismatch and loaded the updated sector from DB
    assert fresh_worker2["AAA"]["sector"] == "Savunma"
    assert worker2_cache[key][2]["AAA"]["sector"] == "Savunma"
    store.reset_financial_snapshot_store_for_tests()


def test_missing_or_expired_version_key_triggers_db_reload(tmp_path: Path) -> None:
    from app.cache import get_cache

    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Havacilik"}})

    # Read into memory
    first = store.get_financial_snapshots_map(processed_dir, ["AAA"])
    assert first["AAA"]["sector"] == "Havacilik"

    # Directly modify DB
    with store._connection(processed_dir) as conn:
        conn.execute("UPDATE company_financial_snapshots SET sector = 'Ulasim' WHERE symbol = 'AAA'")
        conn.commit()

    # Expire/delete version key from cache backend
    get_cache().delete(store._SNAPSHOT_VERSION_KEY)

    # Read again: detects missing version, generates new version, and reloads from DB
    second = store.read_financial_snapshots_map(processed_dir, ["AAA"])
    assert second["AAA"]["sector"] == "Ulasim"
    new_v = get_cache().get(store._SNAPSHOT_VERSION_KEY)
    assert new_v is not None
    store.reset_financial_snapshot_store_for_tests()


def test_graceful_fallback_when_cache_backend_raises(tmp_path: Path, monkeypatch) -> None:
    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Finans"}})

    # Simulate broken / unreachable cache backend
    def failing_get_cache():
        raise RuntimeError("Redis connection failed")

    import app.cache
    monkeypatch.setattr(app.cache, "get_cache", failing_get_cache)

    # Read should not raise, fallback to in-memory version tracking
    data = store.get_financial_snapshots_map(processed_dir, ["AAA"])
    assert data["AAA"]["sector"] == "Finans"

    # Cached in memory
    assert "AAA" in store.load_financial_snapshots_map(processed_dir)

    # Invalidate should also not raise
    bumped = store._invalidate_memory_cache(processed_dir)
    assert bumped
    assert len(store._SNAPSHOT_MEMORY_CACHE) == 0

    store.reset_financial_snapshot_store_for_tests()


def test_aliases_match_target_functions() -> None:
    assert store.load_financial_snapshots_map is store.get_financial_snapshots_map
    assert store.read_financial_snapshots_map is store.get_financial_snapshots_map
    assert store.invalidate_financial_snapshots_cache is store._invalidate_memory_cache
    assert store._get_snapshot_cache_version is store.get_snapshot_cache_version


def test_expired_memory_ttl_with_unchanged_db_stamp_skips_full_table_scan(
    tmp_path: Path, monkeypatch
) -> None:
    from contextlib import contextmanager

    store.reset_financial_snapshot_store_for_tests()
    processed_dir = tmp_path / "processed"
    store.upsert_provider_fundamentals_bulk(processed_dir, {"AAA": {"sector": "Enerji"}})

    class FakeTime:
        now = 100.0

        @classmethod
        def monotonic(cls) -> float:
            return cls.now

    monkeypatch.setattr(store, "time", FakeTime)
    original_connection = store._connection
    queries: list[str] = []

    class QueryTrackingConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql: str, *args, **kwargs):
            queries.append(" ".join(sql.split()))
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    @contextmanager
    def tracked_connection(path: Path):
        with original_connection(path) as conn:
            yield QueryTrackingConn(conn)

    monkeypatch.setattr(store, "_connection", tracked_connection)

    assert store.get_financial_snapshots_map(processed_dir)["AAA"]["sector"] == "Enerji"
    full_scans = [q for q in queries if "SELECT * FROM company_financial_snapshots" in q]
    assert len(full_scans) == 1

    # Advance past 1800s TTL without changing version or DB rows: should only run lightweight stamp query
    FakeTime.now = 1905.0
    assert store.get_financial_snapshots_map(processed_dir)["AAA"]["sector"] == "Enerji"
    full_scans_after = [q for q in queries if "SELECT * FROM company_financial_snapshots" in q]
    stamp_checks = [q for q in queries if "MAX(updated_at)" in q]
    assert len(full_scans_after) == 1
    assert len(stamp_checks) == 1
    store.reset_financial_snapshot_store_for_tests()

