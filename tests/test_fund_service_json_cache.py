from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app import fund_service


@pytest.fixture(autouse=True)
def _isolated_read_state(monkeypatch):
    """Keep the module-level read bookkeeping from leaking between tests."""

    monkeypatch.setattr(fund_service, "JSON_CACHE_LOCAL_READ_TTL_SECONDS", 60)
    with fund_service._JSON_CACHE_READ_STATE_LOCK:
        fund_service._JSON_CACHE_READ_STATE.clear()
    try:
        yield
    finally:
        with fund_service._JSON_CACHE_READ_STATE_LOCK:
            fund_service._JSON_CACHE_READ_STATE.clear()


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
    with fund_service._JSON_CACHE_READ_STATE_LOCK:
        read_at, signature = fund_service._JSON_CACHE_READ_STATE[key]
        fund_service._JSON_CACHE_READ_STATE[key] = (
            read_at - fund_service.JSON_CACHE_LOCAL_READ_TTL_SECONDS - 1,
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
    with fund_service._JSON_CACHE_READ_STATE_LOCK:
        assert fund_service._JSON_CACHE_READ_STATE[str(path)][1] is None


def test_read_json_with_zero_ttl_always_consults_remote(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cache" / "fund.json"
    remote = _FakeRemoteCache({"status": "ok", "version": 1})
    monkeypatch.setattr(fund_service, "read_json_cache", remote)
    monkeypatch.setattr(fund_service, "JSON_CACHE_LOCAL_READ_TTL_SECONDS", 0)

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
    remote_started = threading.Event()
    release_remote = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def slow_remote(cache_path: Path):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        remote_started.set()
        assert release_remote.wait(2)
        return {"status": "ok", "source": "remote"}

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
    assert call_count == 1


def test_read_state_eviction_stays_bounded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fund_service, "_JSON_CACHE_READ_STATE_MAX_ENTRIES", 3)

    paths = [tmp_path / f"cache-{index}.json" for index in range(4)]
    _write_json_file(paths[0], {"index": 0})
    for path in paths:
        fund_service._remember_json_cache_read(path)

    with fund_service._JSON_CACHE_READ_STATE_LOCK:
        keys = list(fund_service._JSON_CACHE_READ_STATE)

    assert len(keys) == 3
    assert str(paths[0]) not in keys
    assert [str(path) for path in paths[1:]] == keys
