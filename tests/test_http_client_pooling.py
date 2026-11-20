from __future__ import annotations

from datetime import date
import decimal
import json
import threading
import time
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


def test_unmocked_tefas_direct_client_is_cached_and_reusable() -> None:
    fund_service.close_shared_http_clients()
    fund_service.reset_fund_caches_for_tests()
    try:
        client1 = fund_service._get_tefas_direct_client()
        client2 = fund_service._get_tefas_direct_client()
        assert client1 is client2
        assert isinstance(client1, httpx.Client)
        assert not client1.is_closed
        assert fund_service._is_shared_tefas_direct_client(client1) is True

        key = fund_service._tefas_direct_prime_key(client1, "https://www.tefas.gov.tr/tr/fon-verileri", "YAT")
        fund_service._mark_tefas_direct_primed(key)
        assert fund_service._is_tefas_direct_primed(key) is True

        fund_service.close_shared_http_clients()
        assert client1.is_closed
        assert fund_service._TEFAS_DIRECT_CLIENT is None
        assert fund_service._is_tefas_direct_primed(key) is False
    finally:
        fund_service.close_shared_http_clients()
        fund_service.reset_fund_caches_for_tests()


def test_tefas_direct_client_mock_safety_bypasses_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    fund_service.close_shared_http_clients()

    class MockDirectClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.is_closed = False

        def close(self) -> None:
            self.is_closed = True

    monkeypatch.setattr(fund_service.httpx, "Client", MockDirectClient)
    try:
        client1 = fund_service._get_tefas_direct_client(timeout_seconds=7.5)
        client2 = fund_service._get_tefas_direct_client(timeout_seconds=7.5)
        assert client1 is not client2
        assert isinstance(client1, MockDirectClient)
        assert client1.kwargs["timeout"] == 7.5
        assert fund_service._TEFAS_DIRECT_CLIENT is None
        assert fund_service._is_shared_tefas_direct_client(client1) is False
    finally:
        fund_service.close_shared_http_clients()


def test_tefas_direct_session_prime_ttl_skips_repeat_get_and_reprimes_on_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fund_service.close_shared_http_clients()
    fund_service.reset_fund_caches_for_tests()
    calls: List[tuple[str, str]] = []
    fail_post_with_403 = False
    client_holder: Dict[str, Any] = {}

    class FakeResponse:
        def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
            self.status_code = status_code
            self.headers = {"content-type": "application/json"}
            self.content = json.dumps(payload).encode("utf-8")

    class SharedMockClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.is_closed = False

        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            calls.append(("GET", url))
            return FakeResponse(200, {})

        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            nonlocal fail_post_with_403
            calls.append(("POST", url))
            if fail_post_with_403:
                fail_post_with_403 = False
                return FakeResponse(403, {"error": "session expired"})
            body = kwargs["json"]
            row_date = f"{body['basTarih'][:4]}-{body['basTarih'][4:6]}-{body['basTarih'][6:8]}"
            return FakeResponse(
                200,
                {
                    "resultList": [
                        {"fonKodu": "TLY", "tarih": row_date, "fiyat": 12.5, "portfoyBuyukluk": 1000.0}
                    ],
                    "toplamSayfa": 1,
                },
            )

        def close(self) -> None:
            self.is_closed = True

    shared_mock = SharedMockClient()
    monkeypatch.setattr(fund_service, "_get_tefas_direct_client", lambda *_a, **_k: shared_mock)
    monkeypatch.setattr(fund_service, "_is_shared_tefas_direct_client", lambda c: c is shared_mock)
    monkeypatch.setattr(fund_service, "TEFAS_DIRECT_PRIME_TTL_SECONDS", 600.0)
    monkeypatch.setattr(fund_service, "TEFAS_DIRECT_CHUNK_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(fund_service, "TEFAS_HTTP_RETRY_ATTEMPTS", 2)
    client_holder["client"] = shared_mock

    try:
        tefas = fund_service.TefasClient()
        rows_first = tefas.fetch_funds_direct(
            start_date=date(2026, 2, 15),
            end_date=date(2026, 3, 5),
            fund_codes=["TLY"],
        )
        assert len(rows_first) == 2
        assert [c[0] for c in calls] == ["GET", "POST", "POST"]
        assert shared_mock.is_closed is False

        calls.clear()
        fail_post_with_403 = True
        rows_second = tefas.fetch_funds_direct(
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 5),
            fund_codes=["TLY"],
        )
        assert len(rows_second) == 1
        # Warm cache skips GET -> hits 403 -> invalidates -> re-primes 1 GET -> retries POST.
        assert [c[0] for c in calls] == ["POST", "GET", "POST"]
    finally:
        fund_service._evict_tefas_direct_prime_entries(client_holder["client"])
        fund_service.reset_fund_caches_for_tests()


def test_tefas_post_json_retries_immediately_on_remote_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    class FakeClient:
        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            calls.append(url)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("stale keepalive socket")
            return httpx.Response(
                200,
                json={"data": [{"fonKodu": "TLY", "tarih": "2026-06-24", "fiyat": 5.0}]},
                headers={"content-type": "application/json"},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(fund_service, "_get_tefas_client", lambda *_a, **_k: FakeClient())
    monkeypatch.setattr(fund_service, "TEFAS_HTTP_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(fund_service, "TEFAS_HTTP_RETRY_BASE_SECONDS", 5.0)
    monkeypatch.setattr(fund_service, "TEFAS_DIRECT_CHUNK_DELAY_SECONDS", 0.0)

    started = time.monotonic()
    rows = fund_service.TefasClient().fetch_fund_range(
        fund_code="TLY",
        start_date=date(2026, 6, 24),
        end_date=date(2026, 6, 24),
    )
    elapsed = time.monotonic() - started
    assert len(calls) == 2
    assert len(rows) == 1
    assert elapsed < 1.0


def test_fintables_sticky_transport_routes_directly_to_proxy_and_recovers_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fund_service.reset_fund_caches_for_tests()
    direct_blocked = True
    proxy_ok = True
    events: List[str] = []

    class FakeHttpxClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            if "/internal/fintables/" in url:
                events.append("proxy")
                if not proxy_ok:
                    return httpx.Response(
                        502,
                        content=b"bad gateway",
                        headers={"content-type": "text/plain"},
                        request=httpx.Request("GET", url),
                    )
                return httpx.Response(
                    200,
                    json={"1w": {"prev_close_date": "2026-06-23T21:00:00Z", "prev_close": 10.0, "high": 11.0, "low": 9.0}},
                    headers={"content-type": "application/json"},
                    request=httpx.Request("GET", url),
                )
            events.append("direct")
            if direct_blocked:
                return httpx.Response(
                    403,
                    content=b"<html><title>Just a moment...</title>cloudflare</html>",
                    headers={"content-type": "text/html"},
                    request=httpx.Request("GET", url),
                )
            return httpx.Response(
                200,
                json={"1w": {"prev_close_date": "2026-06-24T21:00:00Z", "prev_close": 12.0, "high": 13.0, "low": 11.0}},
                headers={"content-type": "application/json"},
                request=httpx.Request("GET", url),
            )

    def fail_curl(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        events.append("curl")
        raise fund_service.FintablesUpstreamError("curl failed")

    def fail_curl_cffi(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        events.append("curl_cffi")
        raise fund_service.FintablesUpstreamError("curl_cffi failed")

    monkeypatch.setattr(fund_service.httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(fund_service, "_fintables_curl_payload", fail_curl)
    monkeypatch.setattr(fund_service, "_fintables_curl_cffi_payload", fail_curl_cffi)
    monkeypatch.setattr(fund_service, "FINTABLES_PROXY_BASE_URL", "https://edge.example.workers.dev")
    monkeypatch.setattr(fund_service, "FINTABLES_PROXY_TOKEN", "secret-token")
    monkeypatch.setattr(fund_service, "FINTABLES_STICKY_TRANSPORT_TTL_SECONDS", 900.0)

    try:
        client = fund_service.FintablesClient()
        res1 = client.fetch_yield_summary("TLY")
        assert res1["periods"]["1w"]["prev_close"] == 10.0
        assert events == ["direct", "curl", "curl_cffi", "proxy"]
        assert fund_service._get_fintables_sticky_transport() == "proxy"

        events.clear()
        res2 = client.fetch_yield_summary("TLY")
        assert res2["periods"]["1w"]["prev_close"] == 10.0
        # Sticky transport bypasses direct, curl, and curl_cffi completely!
        assert events == ["proxy"]

        # When proxy fails and direct recovers, sticky is cleared and direct answers cleanly.
        events.clear()
        proxy_ok = False
        direct_blocked = False
        res3 = client.fetch_yield_summary("TLY")
        assert res3["periods"]["1w"]["prev_close"] == 12.0
        assert events == ["proxy", "direct"]
        assert fund_service._get_fintables_sticky_transport() is None
    finally:
        fund_service.reset_fund_caches_for_tests()


def test_fintables_curl_cffi_session_is_pooled_and_closed_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from curl_cffi import requests as curl_requests

    fund_service.close_shared_http_clients()
    created: List[Any] = []

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.closed = False
            self.calls = 0
            created.append(self)

        def get(self, url: str, **kwargs: Any) -> Any:
            self.calls += 1

            class _Resp:
                status_code = 200
                headers = {"content-type": "application/json"}
                content = b'{"s":"ok","t":[],"c":[]}'

            return _Resp()

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(curl_requests, "Session", FakeSession)
    try:
        payload1 = fund_service._fintables_curl_cffi_payload(
            "https://gate.fintables.com/barbar/udf/history",
            params={"symbol": "TLY"},
            headers={"Accept": "application/json"},
            timeout_seconds=5.0,
            context="test cffi pool 1",
        )
        payload2 = fund_service._fintables_curl_cffi_payload(
            "https://gate.fintables.com/barbar/udf/history",
            params={"symbol": "MAC"},
            headers={"Accept": "application/json"},
            timeout_seconds=5.0,
            context="test cffi pool 2",
        )
        assert payload1 == {"s": "ok", "t": [], "c": []}
        assert payload2 == {"s": "ok", "t": [], "c": []}
        assert len(created) == 1
        assert created[0].kwargs["impersonate"] == "chrome131"
        assert created[0].calls == 2
        assert created[0].closed is False

        fund_service.close_shared_http_clients()
        assert created[0].closed is True
        assert fund_service._FINTABLES_CURL_CFFI_SESSION is None
    finally:
        fund_service.close_shared_http_clients()


def test_tefas_pacer_enforces_concurrency_backoff_decay_and_zero_delay_bypass() -> None:
    from app.funds.providers import _TefasPacer

    pacer = _TefasPacer(max_concurrency=2, min_interval_seconds=0.02, max_interval_seconds=0.16)
    active_counts: List[int] = []
    lock = threading.Lock()

    def _worker() -> None:
        with pacer.slot(bypass_delay=True):
            with lock:
                active_counts.append(pacer.stats()["active_slots"])
            time.sleep(0.02)

    threads = [threading.Thread(target=_worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max(active_counts) <= 2
    assert pacer.stats()["peak_concurrency"] <= 2

    # Backoff on 429 doubles up to ceiling; success decays toward base interval.
    assert pacer.stats()["current_interval"] == pytest.approx(0.02)
    pacer.record_rate_limit()
    assert pacer.stats()["current_interval"] == pytest.approx(0.10)  # max(0.02, 0.05) * 2
    pacer.record_rate_limit()
    assert pacer.stats()["current_interval"] == pytest.approx(0.16)  # capped at max_interval
    pacer.record_success()
    assert pacer.stats()["current_interval"] == pytest.approx(0.136)

    # Zero-delay bypass must not sleep even after rate limit pushback.
    started = time.monotonic()
    with pacer.slot(bypass_delay=True):
        pass
    assert time.monotonic() - started < 0.05

    pacer.reset()
    assert pacer.stats()["current_interval"] == pytest.approx(0.02)
    assert pacer.stats()["rate_limit_hits"] == 0
