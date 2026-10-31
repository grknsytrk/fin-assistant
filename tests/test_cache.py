from __future__ import annotations

import sys
import threading
import time

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


def test_memory_cache_set_if_absent_is_atomic() -> None:
    backend = cache_module.InMemoryCache()
    assert backend.set_if_absent("job", "first", ttl_seconds=60) is True
    assert backend.set_if_absent("job", "second", ttl_seconds=60) is False
    assert backend.get("job") == "first"


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


def test_redis_backend_cools_down_when_unavailable_and_recovers(monkeypatch) -> None:
    """A startup outage should use memory temporarily and retry after cooldown."""

    clients = []
    class _ExplodingRedis:
        class Redis:
            @staticmethod
            def from_url(*_args, **_kwargs):
                class _Client:
                    get_calls = 0

                    def ping(self):
                        raise RuntimeError("boom")

                    def get(self, _key):
                        self.get_calls += 1
                        return None

                client = _Client()
                clients.append(client)
                return client

    monkeypatch.setenv("RAGFIN_CACHE_BACKEND", "redis")
    monkeypatch.setenv("RAGFIN_REDIS_URL", "redis://localhost:1")
    monkeypatch.setenv("RAGFIN_REDIS_COOLDOWN_SECONDS", "30")
    monkeypatch.setattr(cache_module, "_BACKEND", None)
    # Replace the redis import target inside RedisCache.__init__.
    import sys

    sys.modules["redis"] = _ExplodingRedis  # type: ignore[assignment]
    try:
        backend = cache_module.get_cache()
    finally:
        sys.modules.pop("redis", None)
    assert backend.name == "redis"
    status = cache_module.cache_status()
    assert status["cache_redis_fallback"] is True
    assert "boom" in str(status["cache_last_error"])
    assert status["cache_redis_circuit"]["open"] is True

    assert backend.get("api:funds:AAA") is None
    assert clients[0].get_calls == 0
    assert backend.metrics_snapshot()["by_key_prefix"]["funds"]["cooldown_skips"] == 1

    backend._cooldown_until = time.monotonic() - 1
    assert backend.get("api:funds:AAA") is None
    assert clients[0].get_calls == 1
    assert backend.circuit_snapshot()["open"] is False


def test_redis_runtime_error_suppresses_retries_during_cooldown(monkeypatch, caplog) -> None:
    class _FailingRedis:
        class Redis:
            @staticmethod
            def from_url(*_args, **_kwargs):
                class _Client:
                    get_calls = 0

                    def ping(self):
                        return True

                    def get(self, _key):
                        self.get_calls += 1
                        raise RuntimeError("quota exceeded")

                client = _Client()
                _FailingRedis.client = client
                return client

    monkeypatch.setenv("RAGFIN_CACHE_BACKEND", "redis")
    monkeypatch.setenv("RAGFIN_REDIS_COOLDOWN_SECONDS", "60")
    monkeypatch.setattr(cache_module, "_BACKEND", None)
    import sys

    sys.modules["redis"] = _FailingRedis  # type: ignore[assignment]
    try:
        backend = cache_module.get_cache()
    finally:
        sys.modules.pop("redis", None)

    with caplog.at_level("WARNING", logger=cache_module.logger.name):
        assert backend.get("api:kap:flow:latest") is None
        assert backend.get("api:kap:flow:latest") is None

    assert _FailingRedis.client.get_calls == 1
    assert backend.circuit_snapshot()["open"] is True
    family = backend.metrics_snapshot()["by_key_prefix"]["kap_flow_store"]
    assert family["get_calls"] == 2
    assert family["get_remote_attempts"] == 1
    assert family["get_errors"] == 1
    assert family["cooldown_skips"] == 1
    assert sum("redis cache get failed" in record.getMessage() for record in caplog.records) == 1


def test_delete_prefix_invalidates_matching_keys() -> None:
    backend = cache_module.get_cache()
    backend.set("api:funds:a", 1, ttl_seconds=60)
    backend.set("api:funds:b", 2, ttl_seconds=60)
    backend.set("api:other", 3, ttl_seconds=60)

    assert backend.delete_prefix("api:funds:") == 2
    assert backend.get("api:funds:a") is None
    assert backend.get("api:funds:b") is None
    assert backend.get("api:other") == 3


def _redis_backend_for_test(monkeypatch, client):
    class _FakeRedisModule:
        class Redis:
            @staticmethod
            def from_url(*_args, **_kwargs):
                return client

    monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule)
    return cache_module.RedisCache(url="redis://test", namespace="test")


def test_redis_invalidation_reports_confirmed_deletes_and_preserves_legacy_contracts(monkeypatch) -> None:
    class _Client:
        delete_calls = []

        def ping(self):
            return True

        def delete(self, *keys):
            self.delete_calls.append(keys)
            return len(keys)

        def scan_iter(self, *, match, count):
            assert match == "test:api:funds:*"
            assert count == 200
            return iter(["test:api:funds:shared"])

    client = _Client()
    backend = _redis_backend_for_test(monkeypatch, client)

    key_result = backend.delete_with_status("api:funds:single")
    prefix_result = backend.delete_prefix_with_status("api:funds:")

    assert key_result.complete is True
    assert prefix_result.complete is True
    assert prefix_result.deleted_count == 1
    assert backend.delete("api:funds:legacy") is None
    assert backend.delete_prefix("api:funds:") == 1


def test_redis_invalidation_reports_deferred_during_circuit_cooldown_and_clears_local_fallback(monkeypatch) -> None:
    class _Client:
        delete_calls = []

        def ping(self):
            return True

        def delete(self, *keys):
            self.delete_calls.append(keys)
            return len(keys)

        def scan_iter(self, *, match, count):
            assert count == 200
            return iter(["test:api:funds:shared"])

    client = _Client()
    backend = _redis_backend_for_test(monkeypatch, client)
    backend._circuit_open = True
    backend._cooldown_until = time.monotonic() + 30
    backend._fallback.set("api:funds:single", "stale", ttl_seconds=60)
    backend._fallback.set("api:funds:row", "stale", ttl_seconds=60)

    key_result = backend.delete_with_status("api:funds:single")
    prefix_result = backend.delete_prefix_with_status("api:funds:")

    assert key_result.complete is False
    assert prefix_result.complete is False
    assert backend._fallback.get("api:funds:single") is None
    assert backend._fallback.get("api:funds:row") is None
    assert client.delete_calls == []


def test_redis_invalidation_reports_deferred_after_remote_delete_errors(monkeypatch) -> None:
    class _Client:
        def ping(self):
            return True

        def delete(self, *_keys):
            raise RuntimeError("redis unavailable")

        def scan_iter(self, *, match, count):
            assert match == "test:api:funds:*"
            assert count == 200
            return iter(["test:api:funds:shared"])

    backend = _redis_backend_for_test(monkeypatch, _Client())
    backend._fallback.set("api:funds:single", "stale", ttl_seconds=60)
    backend._fallback.set("api:funds:row", "stale", ttl_seconds=60)

    key_result = backend.delete_with_status("api:funds:single")
    assert key_result.complete is False
    assert backend._fallback.get("api:funds:single") is None

    backend._circuit_open = False
    backend._cooldown_until = 0.0
    backend._half_open_probe = False
    prefix_result = backend.delete_prefix_with_status("api:funds:")
    assert prefix_result.complete is False
    assert backend._fallback.get("api:funds:row") is None


def test_redis_prefix_invalidation_streams_matches_in_bounded_delete_batches(monkeypatch) -> None:
    keys = [f"test:api:funds:{index}" for index in range(450)]

    class _Client:
        def __init__(self) -> None:
            self.delete_batches = []
            self.scan_counts = []

        def ping(self):
            return True

        def scan_iter(self, *, match, count):
            assert match == "test:api:funds:*"
            self.scan_counts.append(count)
            return iter(keys)

        def delete(self, *batch):
            self.delete_batches.append(batch)
            return len(batch)

    client = _Client()
    backend = _redis_backend_for_test(monkeypatch, client)

    result = backend.delete_prefix_with_status("api:funds:")

    assert result.complete is True
    assert result.deleted_count == 450
    assert client.scan_counts == [200]
    assert [len(batch) for batch in client.delete_batches] == [200, 200, 50]


def test_redis_prefix_invalidation_stays_deferred_after_a_late_batch_error(monkeypatch) -> None:
    keys = [f"test:api:funds:{index}" for index in range(450)]

    class _Client:
        def __init__(self) -> None:
            self.delete_batches = []

        def ping(self):
            return True

        def scan_iter(self, *, match, count):
            assert match == "test:api:funds:*"
            assert count == 200
            return iter(keys)

        def delete(self, *batch):
            self.delete_batches.append(batch)
            if len(self.delete_batches) == 2:
                raise RuntimeError("redis delete failed")
            return len(batch)

    client = _Client()
    backend = _redis_backend_for_test(monkeypatch, client)
    backend._fallback.set("api:funds:local", "stale", ttl_seconds=60)

    result = backend.delete_prefix_with_status("api:funds:")

    assert result.complete is False
    assert result.deleted_count == 201
    assert [len(batch) for batch in client.delete_batches] == [200, 200]
    assert backend._fallback.get("api:funds:local") is None


def test_json_helpers_ignore_non_dict_values() -> None:
    backend = cache_module.get_cache()
    backend.set("json:bad", ["not", "a", "dict"], ttl_seconds=60)

    assert cache_module.get_json_dict("json:bad") is None
    assert cache_module.set_json("json:ok", {"hello": "world"}, ttl_seconds=60) is True
    assert cache_module.get_json_dict("json:ok") == {"hello": "world"}


def test_get_or_set_returns_cache_hit_flag() -> None:
    calls = {"n": 0}

    def factory() -> dict:
        calls["n"] += 1
        return {"call": calls["n"]}

    first, first_hit = cache_module.get_or_set("compute:key", ttl_seconds=60, factory=factory)
    second, second_hit = cache_module.get_or_set("compute:key", ttl_seconds=60, factory=factory)

    assert first == {"call": 1}
    assert first_hit is False
    assert second == {"call": 1}
    assert second_hit is True
    assert calls["n"] == 1


def test_l1_l2_cached_populates_local_cache_from_shared_cache() -> None:
    local_cache: dict[str, dict] = {}
    cache_module.set_json("shared:key", {"value": 42}, ttl_seconds=60)

    value, hit = cache_module.l1_l2_cached(
        local_cache,
        "local:key",
        ttl_seconds=60,
        l2_key="shared:key",
        factory=lambda: {"value": 99},
    )

    assert value == {"value": 42}
    assert hit is True
    assert local_cache["local:key"]["data"] == {"value": 42}


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


def test_single_flight_waiter_uses_bounded_backoff_and_returns_before_timeout(monkeypatch) -> None:
    clock = [0.0]
    sleeps = []

    class _Backend:
        reads = 0

        def get(self, _key):
            self.reads += 1
            return {"value": "ready"} if self.reads == 3 else None

        def set_if_absent(self, *_args, **_kwargs):
            return False

    backend = _Backend()
    monkeypatch.setattr(cache_module, "_BACKEND", backend)
    monkeypatch.setattr(cache_module.time, "time", lambda: clock[0])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(cache_module.time, "sleep", fake_sleep)

    value, status = cache_module.get_or_set_single_flight(
        "test:single-flight:waiter",
        ttl_seconds=30,
        factory=lambda: pytest.fail("a waiter must not run the factory"),
        lock_ttl_seconds=30,
        wait_timeout_seconds=1,
        allow_cached=False,
    )

    assert value == {"value": "ready"}
    assert status == "coalesced"
    assert sleeps == pytest.approx([0.05, 0.075, 0.1125])
    assert backend.reads == 3


def test_single_flight_eight_second_wait_uses_at_most_35_cache_polls(monkeypatch) -> None:
    clock = [0.0]
    sleeps = []

    class _Backend:
        reads = 0

        def get(self, _key):
            self.reads += 1
            return None

        def set_if_absent(self, *_args, **_kwargs):
            return False

    backend = _Backend()
    monkeypatch.setattr(cache_module, "_BACKEND", backend)
    monkeypatch.setattr(cache_module.time, "time", lambda: clock[0])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(cache_module.time, "sleep", fake_sleep)

    value, status = cache_module.get_or_set_single_flight(
        "test:single-flight:timeout",
        ttl_seconds=30,
        factory=lambda: pytest.fail("a pending waiter must not run the factory"),
        lock_ttl_seconds=30,
        wait_timeout_seconds=8,
        allow_cached=False,
    )

    assert value is None
    assert status == "pending"
    assert clock[0] == 8.0
    assert sleeps[:4] == pytest.approx([0.05, 0.075, 0.1125, 0.16875])
    assert max(sleeps) <= 0.25
    assert backend.reads <= 35
    assert backend.reads == len(sleeps)


def test_cached_decorator_waiter_returns_value_and_keeps_timeout_factory_fallback(monkeypatch) -> None:
    from contextlib import contextmanager

    clock = [0.0]

    class _Backend:
        def __init__(self, ready_after: int | None) -> None:
            self.ready_after = ready_after
            self.reads = 0
            self.values = {}

        def get(self, key):
            self.reads += 1
            if self.ready_after is not None and self.reads >= self.ready_after:
                return {"value": "ready"}
            return self.values.get(key)

        def set(self, key, value, *, ttl_seconds):
            self.values[key] = value
            return True

        @contextmanager
        def lock(self, _key, *, timeout=30.0):
            yield False

    monkeypatch.setattr(cache_module.time, "time", lambda: clock[0])
    monkeypatch.setattr(cache_module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    ready_backend = _Backend(ready_after=3)
    monkeypatch.setattr(cache_module, "_BACKEND", ready_backend)
    ready_factory_calls = []

    @cache_module.cached(
        key_fn=lambda: "test:decorator:ready",
        ttl_seconds=30,
        single_flight=True,
        lock_timeout=1,
    )
    def ready_factory():
        ready_factory_calls.append(True)
        return {"value": "factory"}

    assert ready_factory() == {"value": "ready"}
    assert ready_factory_calls == []

    clock[0] = 0.0
    timeout_backend = _Backend(ready_after=None)
    monkeypatch.setattr(cache_module, "_BACKEND", timeout_backend)
    timeout_factory_calls = []

    @cache_module.cached(
        key_fn=lambda: "test:decorator:timeout",
        ttl_seconds=30,
        single_flight=True,
        lock_timeout=0.1,
    )
    def timeout_factory():
        timeout_factory_calls.append(True)
        return {"value": "factory"}

    assert timeout_factory() == {"value": "factory"}
    assert timeout_factory_calls == [True]
    assert timeout_backend.values["test:decorator:timeout"] == {"value": "factory"}


def test_memory_lock_serialises_concurrent_callers() -> None:
    backend = cache_module.InMemoryCache()
    with backend.lock("snapshot", timeout=1) as first:
        assert first is True
        with backend.lock("snapshot", timeout=0.05) as second:
            assert second is False
