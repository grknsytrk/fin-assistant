from __future__ import annotations

import copy
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app import cache as cache_module
from app import fund_service, json_cache


@pytest.fixture(autouse=True)
def _isolated_read_state(monkeypatch):
    """Keep the module-level read bookkeeping from leaking between tests."""

    monkeypatch.setattr(json_cache, "JSON_CACHE_LOCAL_READ_TTL_SECONDS", 60)
    monkeypatch.setattr(json_cache, "JSON_CACHE_METRICS_FLUSH_SECONDS", 3600.0)
    with json_cache._JSON_CACHE_READ_STATE_LOCK:
        json_cache._JSON_CACHE_READ_STATE.clear()
    with json_cache._JSON_CACHE_METRICS_LOCK:
        for key in json_cache._JSON_CACHE_METRICS:
            json_cache._JSON_CACHE_METRICS[key] = 0
        json_cache._JSON_CACHE_METRICS_BY_READER.clear()
        json_cache._JSON_CACHE_METRICS_SINCE = time.monotonic()
    try:
        yield
    finally:
        with json_cache._JSON_CACHE_READ_STATE_LOCK:
            json_cache._JSON_CACHE_READ_STATE.clear()


def _metrics() -> dict[str, int]:
    with json_cache._JSON_CACHE_METRICS_LOCK:
        return dict(json_cache._JSON_CACHE_METRICS)


class _FakeRemoteCache:
    def __init__(self, payload=None) -> None:
        self.payload = payload
        self.calls: list[Path] = []
        self._lock = threading.Lock()

    def __call__(self, path: Path):
        with self._lock:
            self.calls.append(path)
            payload = copy.deepcopy(self.payload)
        return payload


def _expire_read_state(path: Path) -> None:
    key = str(path)
    with json_cache._JSON_CACHE_READ_STATE_LOCK:
        read_at, signature = json_cache._JSON_CACHE_READ_STATE[key]
        json_cache._JSON_CACHE_READ_STATE[key] = (
            read_at - json_cache.JSON_CACHE_LOCAL_READ_TTL_SECONDS - 1,
            signature,
        )


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_json_serves_local_copy_without_remote_within_ttl(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "rows": ["remote"]})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    first = fund_service._read_json(path)
    second = fund_service._read_json(path)

    assert first == {"status": "ok", "rows": ["remote"]}
    assert second == first
    assert len(remote.calls) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == first


def test_read_json_revalidates_remotely_after_ttl(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "version": 1})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    assert fund_service._read_json(path) == {"status": "ok", "version": 1}
    remote.payload = {"status": "ok", "version": 2}

    assert fund_service._read_json(path) == {"status": "ok", "version": 1}
    assert len(remote.calls) == 1

    _expire_read_state(path)
    assert fund_service._read_json(path) == {"status": "ok", "version": 2}
    assert len(remote.calls) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok", "version": 2}


def test_read_json_revalidates_when_local_file_changed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "version": 1})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    fund_service._read_json(path)
    _write_json_file(path, {"status": "ok", "version": "local-edit"})
    remote.payload = {"status": "ok", "version": 2}

    assert fund_service._read_json(path) == {"status": "ok", "version": 2}
    assert len(remote.calls) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok", "version": 2}


def test_read_json_does_not_rewrite_identical_local_copy(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "rows": []})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    fund_service._read_json(path)
    first_mtime = path.stat().st_mtime_ns

    _expire_read_state(path)
    assert fund_service._read_json(path) == {"status": "ok", "rows": []}

    assert len(remote.calls) == 2
    assert path.stat().st_mtime_ns == first_mtime


def test_read_json_uses_local_copy_on_remote_miss_and_caches_the_miss(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    _write_json_file(path, {"status": "ok", "rows": ["local"]})
    remote = _FakeRemoteCache(None)
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    assert fund_service._read_json(path) == {"status": "ok", "rows": ["local"]}
    assert fund_service._read_json(path) == {"status": "ok", "rows": ["local"]}
    assert len(remote.calls) == 1

    _expire_read_state(path)
    assert fund_service._read_json(path) == {"status": "ok", "rows": ["local"]}
    assert len(remote.calls) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok", "rows": ["local"]}


def test_read_json_caches_miss_for_missing_local_and_remote(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "missing.json"
    remote = _FakeRemoteCache(None)
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    assert fund_service._read_json(path) is None
    assert fund_service._read_json(path) is None

    assert len(remote.calls) == 1
    assert not path.exists()
    with json_cache._JSON_CACHE_READ_STATE_LOCK:
        assert json_cache._JSON_CACHE_READ_STATE[str(path)][1] is None


def test_read_json_with_zero_ttl_always_consults_remote(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "version": 1})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)
    monkeypatch.setattr(json_cache, "JSON_CACHE_LOCAL_READ_TTL_SECONDS", 0)

    assert fund_service._read_json(path) == {"status": "ok", "version": 1}
    remote.payload = {"status": "ok", "version": 2}

    assert fund_service._read_json(path) == {"status": "ok", "version": 2}
    assert len(remote.calls) == 2


def test_write_local_json_makes_next_read_local_only(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "source": "remote"})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    fund_service._write_local_json(path, {"status": "ok", "source": "writer"})

    assert fund_service._read_json(path) == {"status": "ok", "source": "writer"}
    assert remote.calls == []


def test_read_json_coalesces_concurrent_remote_reads(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "shared.json"
    remote = _FakeRemoteCache({"status": "ok", "source": "remote"})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    # Warm the key, then force the TTL to expire so every thread has to
    # revalidate; only one of them may hit Supabase.
    assert fund_service._read_json(path) == {"status": "ok", "source": "remote"}
    _expire_read_state(path)

    remote_started = threading.Event()
    release_remote = threading.Event()
    original_remote = remote

    def slow_remote(cache_path: Path):
        payload = original_remote(cache_path)
        remote_started.set()
        assert release_remote.wait(2)
        return payload

    monkeypatch.setattr(fund_service, "read_json_cache", slow_remote)

    barrier = threading.Barrier(6)

    def worker():
        barrier.wait()
        return fund_service._read_json(path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(worker) for _ in range(6)]
        assert remote_started.wait(2)
        release_remote.set()
        results = [future.result(timeout=5) for future in futures]

    assert results == [{"status": "ok", "source": "remote"}] * 6
    assert len(remote.calls) == 2  # one warm-up read plus one revalidation
    assert _metrics()["local_hits"] == 5


def test_read_json_keeps_cache_files_independent(tmp_path, monkeypatch) -> None:
    path_a = tmp_path / "cache" / "a.json"
    path_b = tmp_path / "cache" / "b.json"
    payloads = {
        str(path_a): {"fund": "AAA"},
        str(path_b): {"fund": "BBB"},
    }
    calls: list[Path] = []

    def remote(cache_path: Path):
        calls.append(cache_path)
        return copy.deepcopy(payloads[str(cache_path)])

    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    assert fund_service._read_json(path_a) == {"fund": "AAA"}
    assert fund_service._read_json(path_b) == {"fund": "BBB"}
    assert fund_service._read_json(path_a) == {"fund": "AAA"}
    assert fund_service._read_json(path_b) == {"fund": "BBB"}

    assert calls == [path_a, path_b]


@pytest.mark.parametrize(
    ("backend_name", "unavailable"),
    [("redis", False), ("memory", True)],
)
def test_json_cache_reads_ignore_response_cache_backend_state(
    tmp_path, monkeypatch, backend_name, unavailable
) -> None:
    """The JSON L1 must behave the same whether Redis works or fell back."""

    class StubResponseCache:
        name = backend_name
        last_error = "redis unavailable" if unavailable else None

    monkeypatch.setattr(cache_module, "_BACKEND", StubResponseCache())
    monkeypatch.setattr(cache_module, "_REQUESTED_BACKEND", "redis")

    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "rows": ["cached"]})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    assert fund_service._read_json(path) == {"status": "ok", "rows": ["cached"]}
    assert fund_service._read_json(path) == {"status": "ok", "rows": ["cached"]}

    assert len(remote.calls) == 1
    metrics = _metrics()
    assert metrics["remote_reads"] == 1
    assert metrics["local_hits"] == 1


def test_json_cache_metrics_record_hits_and_single_remote_read(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "rows": ["x"]})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    fund_service._read_json(path)
    fund_service._read_json(path)
    fund_service._read_json(path)

    metrics = _metrics()
    assert metrics["remote_reads"] == 1
    assert metrics["remote_hits"] == 1
    assert metrics["remote_misses"] == 0
    assert metrics["fallback_local"] == 0
    assert metrics["local_hits"] == 2
    assert metrics["coalesced_hits"] == 0
    assert metrics["remote_bytes"] > 0


def test_json_cache_health_metrics_group_payload_bytes_by_safe_prefix(tmp_path) -> None:
    path = tmp_path / "processed" / "funds_cache" / "AAA.json"
    remote = _FakeRemoteCache({"fund_code": "AAA", "price": 12.5})
    before = json_cache.json_cache_metrics_status()

    assert json_cache.read_json_cache_local_first(path, remote_reader=remote) == remote.payload
    assert json_cache.read_json_cache_local_first(path, remote_reader=remote) == remote.payload

    after = json_cache.json_cache_metrics_status()
    family = after["by_key_prefix"]["funds_cache"]
    before_family = before["by_key_prefix"].get("funds_cache", {})
    assert family["remote_reads"] - before_family.get("remote_reads", 0) == 1
    assert family["remote_hits"] - before_family.get("remote_hits", 0) == 1
    assert family["local_hits"] - before_family.get("local_hits", 0) == 1
    assert family["approx_remote_payload_bytes"] > before_family.get(
        "approx_remote_payload_bytes", 0
    )
    assert str(path) not in str(after)


def test_read_json_uses_local_copy_when_remote_raises(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    _write_json_file(path, {"status": "ok", "rows": ["local"]})

    def failing_remote(_cache_path: Path):
        raise RuntimeError("supabase unavailable")

    monkeypatch.setattr(fund_service, "read_json_cache", failing_remote)

    assert fund_service._read_json(path) == {"status": "ok", "rows": ["local"]}
    metrics = _metrics()
    assert metrics["remote_misses"] == 1
    assert metrics["fallback_local"] == 1


def test_json_cache_metrics_record_remote_miss_and_local_fallback(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    _write_json_file(path, {"status": "ok", "rows": ["local"]})
    monkeypatch.setattr(fund_service, "read_json_cache", _FakeRemoteCache(None))

    assert fund_service._read_json(path) == {"status": "ok", "rows": ["local"]}

    metrics = _metrics()
    assert metrics["remote_reads"] == 1
    assert metrics["remote_hits"] == 0
    assert metrics["remote_misses"] == 1
    assert metrics["fallback_local"] == 1
    assert metrics["local_hits"] == 0
    assert metrics["remote_bytes"] == 0


def test_json_cache_metrics_flush_emits_one_anonymous_aggregate(
    tmp_path, monkeypatch, caplog
) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "rows": ["x"]})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)

    fund_service._read_json(path)
    fund_service._read_json(path)
    fund_service._read_json(path)

    monkeypatch.setattr(json_cache, "JSON_CACHE_METRICS_FLUSH_SECONDS", 1.0)
    with json_cache._JSON_CACHE_METRICS_LOCK:
        json_cache._JSON_CACHE_METRICS_SINCE -= 2
    with caplog.at_level(logging.INFO, logger=json_cache.LOGGER.name):
        fund_service._read_json(path)

    messages = [
        record.getMessage()
        for record in caplog.records
        if "json cache reads" in record.getMessage()
        and "json cache reads by_reader" not in record.getMessage()
    ]

    assert len(messages) == 1
    message = messages[0]
    assert "local_hits=3" in message
    assert "remote_reads=1" in message
    assert "remote_hits=1" in message
    assert "remote_misses=0" in message
    assert "fallback_local=0" in message
    assert "approx_payload_kib=" in message
    assert str(path) not in message
    assert "fund.json" not in message
    assert any(
        "json cache reads by_reader" in record.getMessage()
        and "reader=funds" in record.getMessage()
        and "approx_payload_kib=" in record.getMessage()
        for record in caplog.records
    )

    assert _metrics()["local_hits"] == 0
    assert _metrics()["remote_reads"] == 0


def test_read_state_eviction_stays_bounded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(json_cache, "JSON_CACHE_READ_STATE_MAX_ENTRIES", 3)

    paths = [tmp_path / f"cache-{index}.json" for index in range(4)]
    _write_json_file(paths[0], {"index": 0})
    for path in paths:
        json_cache.remember_json_cache_read(path)

    with json_cache._JSON_CACHE_READ_STATE_LOCK:
        keys = list(json_cache._JSON_CACHE_READ_STATE)

    assert len(keys) == 3
    assert str(paths[0]) not in keys
    assert [str(path) for path in paths[1:]] == keys
