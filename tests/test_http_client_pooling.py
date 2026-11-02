from __future__ import annotations

import decimal
from typing import Any, Dict, List
import httpx
import pytest
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient

from app.api import app
from app.etag import ORJSONResponse
from app import fund_service


def test_fastapi_app_uses_orjson_as_default_response_class() -> None:
    assert app.router.default_response_class is ORJSONResponse
    health_route = next(r for r in app.routes if getattr(r, "path", "") == "/health")
    assert health_route.response_class is ORJSONResponse


def test_fastapi_endpoints_serialize_with_orjson() -> None:
    test_app = FastAPI(default_response_class=ORJSONResponse)
    test_app.add_api_route("/api/test-orjson-decimal", lambda: {"val": decimal.Decimal("123.456"), "status": "ok"})
    client = TestClient(test_app)
    response = client.get("/api/test-orjson-decimal")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"val": 123.456, "status": "ok"}


def test_unmocked_kap_client_is_cached_and_reusable() -> None:
    fund_service.close_shared_http_clients()
    try:
        client1 = fund_service._get_kap_client()
        client2 = fund_service._get_kap_client()
        assert client1 is client2
        assert isinstance(client1, httpx.Client)
        assert not client1.is_closed
    finally:
        fund_service.close_shared_http_clients()


def test_unmocked_fintables_proxy_client_is_cached_and_reusable() -> None:
    fund_service.close_shared_http_clients()
    try:
        client1 = fund_service._get_fintables_proxy_client()
        client2 = fund_service._get_fintables_proxy_client()
        assert client1 is client2
        assert isinstance(client1, httpx.Client)
        assert not client1.is_closed
    finally:
        fund_service.close_shared_http_clients()


def test_close_shared_http_clients_closes_sockets_and_resets_pool() -> None:
    fund_service.close_shared_http_clients()
    client_kap = fund_service._get_kap_client()
    client_proxy = fund_service._get_fintables_proxy_client()

    assert not client_kap.is_closed
    assert not client_proxy.is_closed

    fund_service.close_shared_http_clients()

    assert client_kap.is_closed
    assert client_proxy.is_closed
    assert fund_service._KAP_CLIENT is None
    assert fund_service._FINTABLES_PROXY_CLIENT is None

    # Next call creates a fresh client
    client_new = fund_service._get_kap_client()
    try:
        assert client_new is not client_kap
        assert not client_new.is_closed
    finally:
        fund_service.close_shared_http_clients()


def test_mock_safety_bypasses_pool_when_httpx_client_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    created_instances: List[Any] = []

    class MockHttpxClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            self.is_closed = False
            created_instances.append(self)

        def close(self) -> None:
            self.is_closed = True

    monkeypatch.setattr(fund_service.httpx, "Client", MockHttpxClient)

    # KAP client getter should return a new instance every time, never caching it
    client1 = fund_service._get_kap_client()
    client2 = fund_service._get_kap_client()
    assert client1 is not client2
    assert isinstance(client1, MockHttpxClient)
    assert isinstance(client2, MockHttpxClient)
    assert fund_service._KAP_CLIENT is None

    # Fintables proxy client getter should also return a new instance every time
    client3 = fund_service._get_fintables_proxy_client(timeout_seconds=5.0)
    client4 = fund_service._get_fintables_proxy_client(timeout_seconds=5.0)
    assert client3 is not client4
    assert isinstance(client3, MockHttpxClient)
    assert client3.kwargs["timeout"] == 5.0
    assert fund_service._FINTABLES_PROXY_CLIENT is None


def test_kap_get_json_retries_once_on_remote_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Any:
            calls.append((url, kwargs))
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("Server disconnected without response")
            return httpx.Response(
                200,
                json={"result": "success"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(fund_service.httpx, "Client", MockClient)

    result = fund_service._kap_get_json("/tr/api/test")
    assert len(calls) == 2
    assert result == {"result": "success"}


def test_kap_get_text_retries_once_on_remote_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Any:
            calls.append((url, kwargs))
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("idle timeout")
            return httpx.Response(
                200,
                text="<html>ok</html>",
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(fund_service.httpx, "Client", MockClient)

    result = fund_service._kap_get_text("/tr/test-page")
    assert len(calls) == 2
    assert result == "<html>ok</html>"


def test_kap_get_bytes_retries_once_on_remote_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Any:
            calls.append((url, kwargs))
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("idle timeout")
            return httpx.Response(
                200,
                content=b"%PDF-1.4...",
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(fund_service.httpx, "Client", MockClient)

    result = fund_service._kap_get_bytes("/tr/api/file/download/123")
    assert len(calls) == 2
    assert result == b"%PDF-1.4..."


def test_kap_get_fails_if_remote_protocol_error_persists_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Any:
            calls.append(url)
            raise httpx.RemoteProtocolError("persistent disconnect")

    monkeypatch.setattr(fund_service.httpx, "Client", MockClient)

    with pytest.raises(httpx.RemoteProtocolError):
        fund_service._kap_get_json("/tr/api/broken")

    assert len(calls) == 2


def test_kap_post_json_does_not_retry_remote_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> Any:
            calls.append(url)
            raise httpx.RemoteProtocolError("socket closed")

    monkeypatch.setattr(fund_service.httpx, "Client", MockClient)

    with pytest.raises(httpx.RemoteProtocolError):
        fund_service._kap_post_json("/tr/api/search", {"keyword": "test"})

    # POST must not be retried because it is not idempotent
    assert len(calls) == 1


def test_fintables_proxy_payload_retries_once_on_remote_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fund_service, "FINTABLES_PROXY_BASE_URL", "https://proxy.example.com")
    monkeypatch.setattr(fund_service, "FINTABLES_PROXY_TOKEN", "test-token")

    calls = []

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Any:
            calls.append((url, kwargs))
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("keepalive dropped")
            return httpx.Response(
                200,
                json={"s": "ok", "t": [1700000000], "c": [10.5]},
                headers={"content-type": "application/json"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(fund_service.httpx, "Client", MockClient)

    result = fund_service._fintables_proxy_payload(
        endpoint="https://fintables.com/barbar/udf/history",
        params={"symbol": "TI2"},
        timeout_seconds=8.0,
        context="test proxy history",
    )

    assert len(calls) == 2
    assert calls[0][0] == "https://proxy.example.com/internal/fintables/udf/history"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-token"
    assert calls[0][1]["timeout"] == 8.0
    assert result == {"s": "ok", "t": [1700000000], "c": [10.5]}


def test_concurrent_threads_client_getter_and_teardown() -> None:
    """Stress test concurrent access to client getters during rapid teardown."""
    import concurrent.futures

    errors: List[Exception] = []

    def worker(worker_id: int) -> None:
        try:
            for _ in range(5):
                client = fund_service._get_kap_client()
                assert isinstance(client, httpx.Client)
                if worker_id % 2 == 0:
                    fund_service.close_shared_http_clients()
        except Exception as exc:
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        concurrent.futures.wait(futures)

    fund_service.close_shared_http_clients()
    assert errors == []


def test_kap_request_get_keeps_headers_dynamic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class MockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Any:
            calls.append((url, kwargs))
            if "pdf" in url:
                return httpx.Response(200, content=b"%PDF", request=httpx.Request("GET", url))
            elif "html" in url:
                return httpx.Response(200, text="ok", request=httpx.Request("GET", url))
            return httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(fund_service.httpx, "Client", MockClient)

    fund_service._kap_get_json("/test/json")
    fund_service._kap_get_text("/test/html")
    fund_service._kap_get_bytes("/test/pdf")

    assert len(calls) == 3
    assert calls[0][1]["headers"]["Accept"] == "application/json, text/plain, */*"
    assert calls[1][1]["headers"]["Accept"] == "text/html,*/*"
    assert calls[2][1]["headers"]["Accept"] == "application/pdf,*/*"
