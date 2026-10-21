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
