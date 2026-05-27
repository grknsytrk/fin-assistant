from __future__ import annotations

import time
import threading

import pytest

from app import cache as cache_module


@pytest.fixture(autouse=True)
def _reset_backend(monkeypatch):
    """Each test starts with a clean backend selection."""

    monkeypatch.delenv("RAGFIN_CACHE_BACKEND", raising=False)
    monkeypatch.delenv("RAGFIN_REDIS_URL", raising=False)
    monkeypatch.delenv("RAGFIN_CACHE_NAMESPACE", raising=False)
    cache_module.reset_cache_for_tests()
    yield
    cache_module.reset_cache_for_tests()


def test_default_backend_is_memory() -> None:
    backend = cache_module.get_cache()
    assert backend.name == "memory"


def test_memory_cache_get_set_with_ttl() -> None:
    backend = cache_module.get_cache()
    backend.set("k", {"hello": "world"}, ttl_seconds=2)
    assert backend.get("k") == {"hello": "world"}


def test_memory_cache_returns_none_after_expiry(monkeypatch) -> None:
    backend = cache_module.InMemoryCache()
    backend.set("k", "v", ttl_seconds=1)
    real_time = time.time
    monkeypatch.setattr(cache_module.time, "time", lambda: real_time() + 5)
    assert backend.get("k") is None


def test_unknown_backend_falls_back_to_memory(monkeypatch) -> None:
    monkeypatch.setenv("RAGFIN_CACHE_BACKEND", "fancy-but-unknown")
    cache_module.reset_cache_for_tests()
    backend = cache_module.get_cache()
    assert backend.name == "memory"


def test_redis_backend_falls_back_when_unavailable(monkeypatch) -> None:
    """If the redis client cannot reach the server we keep the app running."""

    class _ExplodingRedis:
        class Redis:
            @staticmethod
            def from_url(*_args, **_kwargs):
                class _Client:
                    def ping(self):
                        raise RuntimeError("boom")

                return _Client()

    monkeypatch.setenv("RAGFIN_CACHE_BACKEND", "redis")
    monkeypatch.setenv("RAGFIN_REDIS_URL", "redis://localhost:1")
    monkeypatch.setattr(cache_module, "_BACKEND", None)
    # Replace the redis import target inside RedisCache.__init__.
    import sys

    sys.modules["redis"] = _ExplodingRedis  # type: ignore[assignment]
    try:
        backend = cache_module.get_cache()
    finally:
        sys.modules.pop("redis", None)
    assert backend.name == "memory"
    status = cache_module.cache_status()
    assert status["cache_redis_fallback"] is True
    assert "boom" in str(status["cache_last_error"])


def test_delete_prefix_invalidates_matching_keys() -> None:
    backend = cache_module.get_cache()
    backend.set("api:funds:a", 1, ttl_seconds=60)
    backend.set("api:funds:b", 2, ttl_seconds=60)
    backend.set("api:other", 3, ttl_seconds=60)

    assert backend.delete_prefix("api:funds:") == 2
    assert backend.get("api:funds:a") is None
    assert backend.get("api:funds:b") is None
    assert backend.get("api:other") == 3


def test_cached_decorator_returns_cached_payload() -> None:
    calls = {"n": 0}

    @cache_module.cached(key_fn=lambda value: f"key:{value}", ttl_seconds=60)
    def expensive(value: str) -> dict:
        calls["n"] += 1
        return {"echo": value}

    first = expensive("hi")
    second = expensive("hi")
    third = expensive("there")
    assert first == {"echo": "hi"}
    assert second == {"echo": "hi"}
    assert third == {"echo": "there"}
    assert calls["n"] == 2


def test_cached_decorator_skips_when_predicate_matches() -> None:
    calls = {"n": 0}

    @cache_module.cached(
        key_fn=lambda *, value: f"k:{value}",
        ttl_seconds=60,
        skip_when=lambda *, value: value == "no-cache",
    )
    def compute(*, value: str) -> dict:
        calls["n"] += 1
        return {"value": value}

    compute(value="ok")
    compute(value="ok")
    compute(value="no-cache")
    compute(value="no-cache")
    assert calls["n"] == 3  # cached once for "ok", bypassed twice for "no-cache"


def test_cached_decorator_single_flight_shares_one_miss() -> None:
    calls = {"n": 0}
    errors = []
    results = []
    start = threading.Event()

    @cache_module.cached(
        key_fn=lambda value: f"single:{value}",
        ttl_seconds=60,
        single_flight=True,
        lock_timeout=2,
    )
    def expensive(value: str) -> dict:
        calls["n"] += 1
        time.sleep(0.1)
        return {"echo": value, "call": calls["n"]}

    def worker() -> None:
        try:
            start.wait(timeout=1)
            results.append(expensive("same"))
        except Exception as exc:  # pragma: no cover - debugging aid
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=3)

    assert not errors
    assert len(results) == 6
    assert results == [{"echo": "same", "call": 1}] * 6
    assert calls["n"] == 1


def test_memory_lock_serialises_concurrent_callers() -> None:
    backend = cache_module.InMemoryCache()
    with backend.lock("snapshot", timeout=1) as first:
        assert first is True
        with backend.lock("snapshot", timeout=0.05) as second:
            assert second is False
