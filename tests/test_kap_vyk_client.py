from __future__ import annotations

import io
import urllib.error

import pytest

from src import kap_vyk_client
from src.config import KapConfig


@pytest.fixture(autouse=True)
def _reset_kap_vyk_caches() -> None:
    kap_vyk_client.reset_caches_for_tests()


def _kap_cfg(
    *,
    base_url: str,
    auth_mode: str = "auto",
    token_url: str = "",
    api_key: str = "key-123",
    api_secret: str = "secret-456",
) -> KapConfig:
    return KapConfig(
        enabled=True,
        timeout_seconds=5.0,
        cache_ttl_hours=0.0,
        user_agent="ragfin-test/1.0",
        api_key=api_key,
        api_secret=api_secret,
        vyk_base_url=base_url,
        vyk_auth_mode=auth_mode,
        vyk_token_url=token_url,
    )


def _http_error(url: str, code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, f"HTTP {code}", hdrs=None, fp=io.BytesIO(body))


def test_request_json_dev_auto_uses_basic_without_generate_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _kap_cfg(base_url="https://apigwdev.mkk.com.tr/api/vyk", auth_mode="auto")
    calls = []

    def fake_open_payload(url: str, *, cfg: KapConfig, headers: dict[str, str]):
        calls.append((url, headers["Authorization"]))
        return {"lastDisclosureIndex": 42}, {}

    monkeypatch.setattr(kap_vyk_client, "_open_payload", fake_open_payload)

    payload = kap_vyk_client._request_json(  # type: ignore[attr-defined]
        "https://apigwdev.mkk.com.tr/api/vyk/lastDisclosureIndex",
        cfg=cfg,
    )

    assert payload == {"lastDisclosureIndex": 42}
    assert len(calls) == 1
    assert calls[0][0].endswith("/lastDisclosureIndex")
    assert calls[0][1].startswith("Basic ")


def test_request_json_token_mode_fetches_and_uses_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _kap_cfg(base_url="https://apigw.mkk.com.tr/api/vyk", auth_mode="token")
    calls = []
    token = "header.payload.signature-token"

    def fake_open_payload(url: str, *, cfg: KapConfig, headers: dict[str, str]):
        calls.append((url, headers["Authorization"]))
        if "/generateToken" in url:
            return {"token": token}, {}
        assert headers["Authorization"] == f"Bearer {token}"
        return {"lastDisclosureIndex": 99}, {}

    monkeypatch.setattr(kap_vyk_client, "_open_payload", fake_open_payload)

    payload = kap_vyk_client._request_json(  # type: ignore[attr-defined]
        "https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex",
        cfg=cfg,
    )

    assert payload == {"lastDisclosureIndex": 99}
    assert len(calls) == 2
    assert calls[0][0] == "https://apigw.mkk.com.tr/auth/generateToken?apiKey=key-123"
    assert calls[0][1].startswith("Basic ")
    assert calls[1] == (
        "https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex",
        f"Bearer {token}",
    )


def test_request_json_token_mode_supports_api_key_only_generate_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _kap_cfg(
        base_url="https://apigw.mkk.com.tr/api/vyk",
        auth_mode="token",
        api_secret="",
    )
    calls = []
    token = "header.payload.signature-token"

    def fake_open_payload(url: str, *, cfg: KapConfig, headers: dict[str, str]):
        calls.append((url, headers.get("Authorization", "")))
        if "/auth/generateToken" in url:
            assert "apiKey=key-123" in url
            assert headers.get("Authorization") is None
            return {"token": token}, {}
        assert headers.get("Authorization") == f"Bearer {token}"
        return {"lastDisclosureIndex": 51}, {}

    monkeypatch.setattr(kap_vyk_client, "_open_payload", fake_open_payload)

    payload = kap_vyk_client._request_json(  # type: ignore[attr-defined]
        "https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex",
        cfg=cfg,
    )

    assert payload == {"lastDisclosureIndex": 51}
    assert calls[0][0] == "https://apigw.mkk.com.tr/auth/generateToken?apiKey=key-123"
    assert calls[1] == (
        "https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex",
        f"Bearer {token}",
    )


def test_request_json_auto_falls_back_to_basic_when_token_endpoint_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _kap_cfg(base_url="https://apigw.mkk.com.tr/api/vyk", auth_mode="auto")
    calls = []

    def fake_open_payload(url: str, *, cfg: KapConfig, headers: dict[str, str]):
        calls.append((url, headers["Authorization"]))
        if "/generateToken" in url:
            raise _http_error(url, 404)
        assert headers["Authorization"].startswith("Basic ")
        return {"lastDisclosureIndex": 77}, {}

    monkeypatch.setattr(kap_vyk_client, "_open_payload", fake_open_payload)

    payload = kap_vyk_client._request_json(  # type: ignore[attr-defined]
        "https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex",
        cfg=cfg,
    )

    assert payload == {"lastDisclosureIndex": 77}
    assert calls[0][0] == "https://apigw.mkk.com.tr/auth/generateToken?apiKey=key-123"
    assert calls[1][0] == "https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex"
    assert calls[1][1].startswith("Basic ")


def test_request_json_refreshes_token_after_er006(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _kap_cfg(base_url="https://apigw.mkk.com.tr/api/vyk", auth_mode="token")
    calls = []
    issued_tokens = iter(("token-one-value", "token-two-value"))

    def fake_open_payload(url: str, *, cfg: KapConfig, headers: dict[str, str]):
        calls.append((url, headers["Authorization"]))
        if "/generateToken" in url:
            return {"token": next(issued_tokens)}, {}
        if headers["Authorization"] == "Bearer token-one-value":
            return {"code": "ER006", "message": "Token has expired"}, {}
        assert headers["Authorization"] == "Bearer token-two-value"
        return {"lastDisclosureIndex": 123}, {}

    monkeypatch.setattr(kap_vyk_client, "_open_payload", fake_open_payload)

    payload = kap_vyk_client._request_json(  # type: ignore[attr-defined]
        "https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex",
        cfg=cfg,
    )

    assert payload == {"lastDisclosureIndex": 123}
    assert [url for url, _ in calls].count("https://apigw.mkk.com.tr/auth/generateToken?apiKey=key-123") == 2
    assert ("https://apigw.mkk.com.tr/api/vyk/lastDisclosureIndex", "Bearer token-two-value") in calls
