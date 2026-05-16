import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from src import kap_fetcher
from src.kap_fetcher import (
    KAP_BROWSER_USER_AGENT,
    KAP_CACHE_SCHEMA_VERSION,
    _http_get_json,
    _list_company_disclosures,
    _pick_metric_value,
    fetch_kap_company_snapshot,
)


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _cfg(user_agent: str = "ragfin-test/1.0") -> SimpleNamespace:
    return SimpleNamespace(user_agent=user_agent, timeout_seconds=1.0)


def test_http_get_json_uses_browser_user_agent_before_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_user_agents: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del timeout
        headers = {key.lower(): value for key, value in request.header_items()}  # type: ignore[attr-defined]
        seen_user_agents.append(str(headers.get("user-agent", "")))
        return _FakeResponse([{"ok": True}])

    monkeypatch.setattr(kap_fetcher.urllib.request, "urlopen", fake_urlopen)

    payload = _http_get_json("https://example.test/kap", _cfg())

    assert payload == [{"ok": True}]
    assert seen_user_agents == [KAP_BROWSER_USER_AGENT]


def test_http_get_json_stops_after_429_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_user_agents: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del timeout
        headers = {key.lower(): value for key, value in request.header_items()}  # type: ignore[attr-defined]
        seen_user_agents.append(str(headers.get("user-agent", "")))
        raise urllib.error.HTTPError(
            "https://example.test/kap",
            429,
            "Request Limit Exceeded",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(kap_fetcher.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(kap_fetcher.time, "sleep", lambda _seconds: None)

    with pytest.raises(urllib.error.HTTPError):
        _http_get_json("https://example.test/kap", _cfg())

    assert seen_user_agents == [KAP_BROWSER_USER_AGENT, KAP_BROWSER_USER_AGENT]


def test_list_company_disclosures_stops_after_requested_period_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_http_get_json(url: str, _cfg_obj: object) -> list[dict[str, object]]:
        calls.append(url)
        if "/2026/" in url:
            return [
                {
                    "mkkMemberOid": "oid",
                    "stockCode": "MGROS",
                    "pdOid": "pd-2026-1",
                    "disclosureIndex": 1601478,
                    "year": 2026,
                    "period": 1,
                    "title": "MIGROS TICARET A.S.",
                }
            ]
        if "/2025/" in url:
            return [
                {
                    "mkkMemberOid": "oid",
                    "stockCode": "MGROS",
                    "pdOid": "pd-2025-4",
                    "disclosureIndex": 1566129,
                    "year": 2025,
                    "period": 4,
                    "title": "MIGROS TICARET A.S.",
                }
            ]
        raise AssertionError(f"unexpected year fetch: {url}")

    monkeypatch.setattr(kap_fetcher, "_utc_now", lambda: kap_fetcher.datetime(2026, 5, 11, tzinfo=kap_fetcher.timezone.utc))
    monkeypatch.setattr(kap_fetcher, "_http_get_json", fake_http_get_json)
    monkeypatch.setattr(kap_fetcher.time, "sleep", lambda _seconds: None)

    rows = _list_company_disclosures("oid", _cfg(), max_periods=2)

    assert [row["period"] for row in rows] == [1, 4]
    assert len(calls) == 2


def test_fetch_snapshot_uses_ticker_alias_cache_when_live_list_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cache_dir = tmp_path / "kap_cache"
    cache_dir.mkdir()
    (cache_dir / "ENKAI.json").write_text(
        json.dumps(
            {
                "ok": True,
                "schema_version": KAP_CACHE_SCHEMA_VERSION,
                "company": "ENKAI",
                "stock_code": "ENKAI",
                "member_oid": "oid-enka",
                "fetched_at": "2026-05-08T18:12:50+00:00",
                "quarters": [{"quarter": "2026Q1", "year": 2026, "period": 1}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        kap_fetcher,
        "_resolve_member",
        lambda _company, _cfg_obj: {"mkk_member_oid": "oid-enka", "title": "ENKA INSAAT VE SANAYI A.S."},
    )
    monkeypatch.setattr(kap_fetcher, "_list_company_disclosures", lambda **_kwargs: [])

    payload = fetch_kap_company_snapshot(
        company="ENKA",
        cfg=SimpleNamespace(cache_ttl_hours=0.0, timeout_seconds=1.0, user_agent="ragfin-test/1.0"),
        processed_dir=tmp_path,
        force_refresh=False,
        max_quarters=20,
    )

    assert payload["ok"] is True
    assert payload["cache_hit"] is True
    assert payload["cache_stale"] is True
    assert payload["stock_code"] == "ENKAI"
    assert payload["quarters"][0]["quarter"] == "2026Q1"


def test_net_kar_prefers_parent_or_net_profit_rows() -> None:
    rows = [
        {
            "label_norm": "surdurulen faaliyetler donem kari (zarari)",
            "body_index": 1,
            "col_order": 4,
            "value": 34_628_000_000.0,
        },
        {
            "label_norm": "net donem kari veya zarari",
            "body_index": 0,
            "col_order": 4,
            "value": 22_001_000_000.0,
        },
        {
            "label_norm": "ana ortaklik paylari",
            "body_index": 1,
            "col_order": 4,
            "value": 22_001_000_000.0,
        },
        {
            "label_norm": "kontrol gucu olmayan paylar",
            "body_index": 1,
            "col_order": 4,
            "value": 50_000_000_000.0,
        },
    ]

    picked = _pick_metric_value("net_kar", rows, period=4)
    assert picked == 22_001_000_000.0


def test_net_kar_falls_back_when_only_continued_operations_exists() -> None:
    rows = [
        {
            "label_norm": "surdurulen faaliyetler donem kari (zarari)",
            "body_index": 1,
            "col_order": 4,
            "value": 3_500_000_000.0,
        }
    ]

    picked = _pick_metric_value("net_kar", rows, period=3)
    assert picked == 3_500_000_000.0
