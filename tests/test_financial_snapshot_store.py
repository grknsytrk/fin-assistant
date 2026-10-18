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


def test_financial_snapshot_memory_cache_lives_60_seconds_and_expires(tmp_path: Path, monkeypatch) -> None:
    assert store._SNAPSHOT_MEMORY_CACHE_TTL_SECONDS == 60.0
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
    FakeTime.now = 159.0
    assert "AAA" in store.get_financial_snapshots_map(processed_dir)
    assert reads["count"] == 1

    FakeTime.now = 161.0
    assert "AAA" in store.get_financial_snapshots_map(processed_dir)
    assert reads["count"] == 2
    store.reset_financial_snapshot_store_for_tests()
