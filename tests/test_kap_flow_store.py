from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app import cache as cache_module
from app import kap_flow_store as store


def _item(index: int, minute: int) -> dict:
    return {
        "id": f"kap-{index}",
        "disclosure_id": str(index),
        "source": "KAP",
        "symbol": "THYAO",
        "stock_codes": ["THYAO"],
        "title": f"Bildirim {index}",
        "published_at": datetime(2026, 9, 9, 10, minute, tzinfo=timezone.utc).isoformat(),
        "category": "ozel_durum",
        "kap_url": f"https://www.kap.org.tr/tr/Bildirim/{index}",
    }


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGFIN_CACHE_BACKEND", raising=False)
    monkeypatch.delenv("RAGFIN_REDIS_URL", raising=False)
    cache_module.reset_cache_for_tests()
    yield
    cache_module.reset_cache_for_tests()


def test_cursor_round_trip_and_invalid_values() -> None:
    cursor = store.encode_cursor("2026-09-09T10:12:00Z", "kap:12")

    assert store.decode_cursor(cursor) == ("2026-09-09T10:12:00+00:00", "kap:12")
    assert store.decode_cursor("not-a-cursor") is None


def test_redis_fallback_persists_head_and_pages_older_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "database_enabled", lambda: False)

    status = store.persist_flow_items(
        [_item(1, 1), _item(2, 2), _item(3, 3)],
        source="kap_public_website",
        as_of="2026-09-09T10:03:00+00:00",
    )
    assert status["stored_count"] == 3

    first = store.read_flow_page(limit=2)
    assert first is not None
    assert [item["id"] for item in first["items"]] == ["kap-3", "kap-2"]
    assert first["has_more"] is True

    older = store.read_flow_page(limit=2, before=first["next_cursor"])
    assert older is not None
    assert [item["id"] for item in older["items"]] == ["kap-1"]
    assert older["has_more"] is False


def test_database_read_failure_uses_bounded_redis_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "database_enabled", lambda: False)
    store.persist_flow_items([_item(1, 1)], source="kap_public_website")
    monkeypatch.setattr(store, "database_enabled", lambda: True)
    monkeypatch.setattr(store, "read_kap_flow_events", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("db down")))

    page = store.read_flow_page(limit=1)

    assert page is not None
    assert page["items"][0]["id"] == "kap-1"


def test_persist_flow_items_skips_unchanged_db_upsert_and_read_flow_head_avoids_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store.clear_persisted_flow_hashes()
    monkeypatch.setattr(store, "database_enabled", lambda: True)

    upsert_calls: list[list[dict]] = []
    head_db_calls = 0

    def fake_upsert(rows: list[dict]) -> dict[str, int]:
        upsert_calls.append(list(rows))
        return {"upserted_count": len(rows)}

    def fake_read_head(*, category=None) -> dict[str, str]:
        nonlocal head_db_calls
        head_db_calls += 1
        return {
            "event_key": "kap-2",
            "published_at": "2026-09-09T10:02:00+00:00",
        }

    monkeypatch.setattr(store, "upsert_kap_flow_events", fake_upsert)
    monkeypatch.setattr(store, "read_kap_flow_head", fake_read_head)
    monkeypatch.setattr(
        store,
        "read_kap_flow_events",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("head check must not read full payload rows")),
    )

    items = [_item(1, 1), _item(2, 2)]
    first_status = store.persist_flow_items(
        items,
        source="kap_public_website",
        as_of="2026-09-09T10:02:00+00:00",
    )
    assert first_status["stored_count"] == 2
    assert len(upsert_calls) == 1
    assert len(upsert_calls[0]) == 2

    head1 = store.read_flow_head()
    assert head1["latest_cursor"] is not None
    assert head_db_calls == 1

    second_status = store.persist_flow_items(
        items,
        source="kap_public_website",
        as_of="2026-09-09T10:02:00+00:00",
    )
    assert second_status["stored_count"] == 2
    # Identical items must not re-upsert or invalidate the cached flow head cursor.
    assert len(upsert_calls) == 1

    head2 = store.read_flow_head()
    assert head2["latest_cursor"] == head1["latest_cursor"]
    assert head_db_calls == 1
    store.clear_persisted_flow_hashes()


def test_persist_flow_items_skips_unchanged_redis_event_list_gets_and_sets(monkeypatch) -> None:
    from app.cache import InMemoryCache

    store.clear_persisted_flow_hashes()
    inner = InMemoryCache()
    flow_gets = 0
    flow_sets = 0

    class _CountingCache:
        name = "redis"

        def get(self, key: str):
            nonlocal flow_gets
            if key == store.FLOW_REDIS_STORE_KEY:
                flow_gets += 1
            return inner.get(key)

        def set(self, key: str, value, ttl_seconds=None):
            nonlocal flow_sets
            if key == store.FLOW_REDIS_STORE_KEY:
                flow_sets += 1
            inner.set(key, value, ttl_seconds=ttl_seconds)

        def delete(self, key: str) -> None:
            inner.delete(key)

        def delete_prefix(self, prefix: str) -> None:
            inner.delete_prefix(prefix)

    counting = _CountingCache()
    monkeypatch.setattr(store, "get_cache", lambda: counting)
    monkeypatch.setattr(store, "database_enabled", lambda: True)
    monkeypatch.setattr(store, "upsert_kap_flow_events", lambda rows: {"upserted_count": len(rows)})

    items = [
        {
            "id": "510001",
            "disclosureIndex": 510001,
            "published_at": "2026-09-09T10:05:00+00:00",
            "company": "THYAO",
            "title": "İlk duyuru",
            "category": "ODA",
        }
    ]
    store.persist_flow_items(items, source="kap_public_website", as_of="2026-09-09T10:05:00+00:00")
    assert flow_gets == 1
    assert flow_sets == 1

    # Repeated 1-minute cron ticks with unchanged items must not GET or SET the full event list.
    for _ in range(3):
        store.persist_flow_items(items, source="kap_public_website", as_of="2026-09-09T10:06:00+00:00")

    assert flow_gets == 1
    assert flow_sets == 1
    store.clear_persisted_flow_hashes()
