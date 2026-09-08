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
