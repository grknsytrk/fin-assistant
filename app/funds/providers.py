"""Upstream TEFAS and Fintables provider clients, adapters, and normalizers."""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

if TYPE_CHECKING:
    import httpx
    from app.fund_service import FundSnapshotFetchResult

_ORIGINAL_EXPORTS: Dict[str, Any] = {}
_TEFASFON_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="tefasfon-worker",
)
atexit.register(_TEFASFON_EXECUTOR.shutdown, wait=False)


def _fund_service() -> Any:
    """Resolve ``app.fund_service`` lazily to avoid import cycles and honor monkeypatches."""
    mod = sys.modules.get("app.fund_service")
    if mod is not None:
        return mod
    import app.fund_service as fund_service

    return fund_service


def _seam(name: str) -> Any:
    """Resolve a moved symbol via ``app.fund_service`` unless patched locally on this module."""
    local_obj = globals()[name]
    orig_obj = _ORIGINAL_EXPORTS.get(name, local_obj)
    if local_obj is not orig_obj:
        return local_obj
    return getattr(_fund_service(), name, local_obj)


def _unix_timestamp_for_date(value: date, *, end_of_day: bool = False) -> int:
    if end_of_day:
        moment = datetime(value.year, value.month, value.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        moment = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(moment.timestamp())


def _looks_like_html_challenge(text: str) -> bool:
    lower_text = text[:3000].lower()
    return (
        "<html" in lower_text
        or "just a moment" in lower_text
        or "cloudflare" in lower_text
        or "request rejected" in lower_text
        or 'window["bobcmn"]' in lower_text
        or "failureconfig" in lower_text
    )


def _decode_fintables_json_response(
    status_code: int,
    headers: Dict[str, str],
    body: bytes,
    *,
    context: str,
) -> Dict[str, Any]:
    fs = _fund_service()
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    text = body.decode("utf-8", errors="replace").strip()
    if status_code >= 400:
        if "html" in content_type or _seam("_looks_like_html_challenge")(text):
            raise fs.FintablesUpstreamError(f"{context}: {fs.FINTABLES_GATE_BLOCKED_MESSAGE}")
        raise fs.FintablesUpstreamError(f"{context} HTTP {status_code}")
    if not text:
        raise fs.FintablesFormatError(f"{context} empty response")
    if "html" in content_type or _seam("_looks_like_html_challenge")(text):
        raise fs.FintablesUpstreamError(f"{context}: {fs.FINTABLES_GATE_BLOCKED_MESSAGE}")
    if content_type and "json" not in content_type and not text.startswith(("{", "[")):
        raise fs.FintablesFormatError(f"{context} unexpected content-type: {content_type}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise fs.FintablesFormatError(f"{context} JSON parse failed") from exc
    if not isinstance(payload, dict):
        raise fs.FintablesFormatError(f"{context} response is not an object")
    return payload


def _curl_executable() -> Optional[str]:
    return shutil.which("curl.exe") or shutil.which("curl")


def _fintables_curl_payload(
    url: str,
    *,
    params: Dict[str, Any],
    headers: Dict[str, str],
    timeout_seconds: float,
    context: str,
) -> Dict[str, Any]:
    fs = _fund_service()
    curl = _seam("_curl_executable")()
    if not curl:
        raise fs.FintablesUpstreamError(f"{context}: curl fallback is unavailable")
    query = urlencode({key: value for key, value in params.items() if value is not None})
    full_url = f"{url}?{query}" if query else url
    marker = "__RAGFIN_HTTP_STATUS__:"
    command = [
        curl,
        "-L",
        "--compressed",
        "-sS",
        "--max-time",
        str(max(1, int(timeout_seconds) + 5)),
        "-w",
        f"\n{marker}%{{http_code}}",
    ]
    for key, value in headers.items():
        header_name = str(key).strip()
        if header_name and value is not None:
            command.extend(["-H", f"{header_name}: {value}"])
    command.append(full_url)
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=timeout_seconds + 8)
    except (OSError, subprocess.SubprocessError) as exc:
        raise fs.FintablesUpstreamError(f"{context}: curl fallback failed: {exc}") from exc
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if marker not in stdout:
        detail = stderr or f"curl exit {completed.returncode}"
        raise fs.FintablesUpstreamError(f"{context}: curl fallback did not return HTTP status: {detail}")
    body, raw_status = stdout.rsplit(marker, 1)
    try:
        status_code = int(raw_status.strip()[-3:])
    except ValueError as exc:
        raise fs.FintablesUpstreamError(f"{context}: curl fallback returned invalid HTTP status") from exc
    return _seam("_decode_fintables_json_response")(
        status_code,
        {"content-type": "application/json"},
        body.strip().encode("utf-8"),
        context=context,
    )


def _fintables_curl_cffi_payload(
    url: str,
    *,
    params: Dict[str, Any],
    headers: Dict[str, str],
    timeout_seconds: float,
    context: str,
) -> Dict[str, Any]:
    """Use a browser-shaped TLS client when the shell curl fallback is absent."""
    fs = _fund_service()
    try:
        from curl_cffi import requests as curl_requests  # type: ignore
    except Exception as exc:
        raise fs.FintablesUpstreamError(f"{context}: curl_cffi fallback is unavailable") from exc

    use_direct_get = not getattr(curl_requests.get, "__module__", "").startswith("curl_cffi")
    try:
        if use_direct_get:
            response = curl_requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout_seconds,
                allow_redirects=True,
                impersonate="chrome131",
            )
        else:
            session = fs._get_fintables_curl_cffi_session()
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout_seconds,
                allow_redirects=True,
            )
    except Exception as exc:
        raise fs.FintablesUpstreamError(f"{context}: curl_cffi fallback failed: {exc}") from exc
    return _seam("_decode_fintables_json_response")(
        response.status_code,
        dict(response.headers),
        response.content,
        context=context,
    )


def _fintables_proxy_payload(
    endpoint: str,
    *,
    params: Dict[str, Any],
    timeout_seconds: float,
    context: str,
) -> Dict[str, Any]:
    """Fetch through the authenticated edge proxy when the HF egress is blocked."""
    fs = _fund_service()
    if not fs.FINTABLES_PROXY_BASE_URL or not fs.FINTABLES_PROXY_TOKEN:
        raise fs.FintablesUpstreamError(f"{context}: Fintables proxy is not configured")
    if endpoint.rstrip("/").endswith("/barbar/udf/history"):
        proxy_path = "/internal/fintables/udf/history"
    elif endpoint.rstrip("/").endswith("/barbar/server/yield"):
        proxy_path = "/internal/fintables/yield-summary"
    else:
        raise fs.FintablesUpstreamError(f"{context}: unsupported Fintables proxy endpoint")
    proxy_url = f"{fs.FINTABLES_PROXY_BASE_URL}{proxy_path}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {fs.FINTABLES_PROXY_TOKEN}",
    }
    effective_timeout = timeout_seconds if timeout_seconds is not None else fs.FINTABLES_TIMEOUT_SECONDS
    client = fs._get_fintables_proxy_client(timeout_seconds)
    try:
        try:
            response = client.get(
                proxy_url,
                params=params,
                headers=headers,
                timeout=effective_timeout,
            )
        except fs.httpx.RemoteProtocolError:
            response = client.get(
                proxy_url,
                params=params,
                headers=headers,
                timeout=effective_timeout,
            )
    except fs.httpx.HTTPError as exc:
        raise fs.FintablesUpstreamError(f"{context}: Fintables proxy request failed: {exc}") from exc
    return _seam("_decode_fintables_json_response")(
        response.status_code,
        dict(response.headers),
        response.content,
        context=context,
    )


def _decode_tefas_json_response(
    status_code: int,
    headers: Dict[str, str],
    body: bytes,
    *,
    context: str,
) -> Dict[str, Any]:
    fs = _fund_service()
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    text = body.decode("utf-8", errors="replace").strip()
    if status_code >= 400:
        if status_code == 429:
            raise fs.TefasRateLimitError(
                f"{context} HTTP 429",
                retry_after_seconds=_seam("_tefas_retry_after_seconds")(headers),
            )
        raise fs.TefasUpstreamError(f"{context} HTTP {status_code}")
    if not text:
        raise fs.TefasFormatError(f"{context} empty response")
    if "html" in content_type or _seam("_looks_like_html_challenge")(text):
        raise fs.TefasUpstreamError(f"{context} HTML/WAF response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise fs.TefasFormatError(f"{context} JSON parse failed") from exc
    if not isinstance(payload, dict):
        raise fs.TefasFormatError(f"{context} response is not an object")
    return payload


def _tefas_retry_after_seconds(headers: Dict[str, str]) -> Optional[float]:
    raw_value = str(headers.get("retry-after") or headers.get("Retry-After") or "").strip()
    if not raw_value:
        return None
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _tefas_retry_delay_seconds(attempt: int, retry_after_seconds: Optional[float] = None) -> float:
    fs = _fund_service()
    if retry_after_seconds is not None:
        # Retry-After is a provider instruction, not an exponential-backoff
        # hint.  Keep a separately configurable safety ceiling so a malformed
        # multi-day header cannot pin a worker indefinitely.
        return min(max(0.0, retry_after_seconds), max(0.0, fs.TEFAS_HTTP_RETRY_AFTER_MAX_SECONDS))
    base = max(0.0, fs.TEFAS_HTTP_RETRY_BASE_SECONDS)
    delay = base * (2 ** max(0, attempt - 1))
    bounded = min(delay, max(0.0, fs.TEFAS_HTTP_RETRY_MAX_SECONDS))
    return bounded * fs.random.uniform(0.8, 1.2)


def _is_tefas_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, _fund_service().TefasRateLimitError) or "http 429" in str(exc).lower()


def _normalize_tefas_fund_list_payload(payload: Dict[str, Any], *, source_url: str) -> List[Dict[str, Any]]:
    fs = _fund_service()
    error_message = str(payload.get("errorMessage") or "").strip()
    if error_message and not payload.get("resultList"):
        raise fs.TefasUpstreamError(f"TEFAS fund list error: {error_message}")
    raw_rows = payload.get("data") or payload.get("Data") or payload.get("resultList") or []
    if raw_rows is None:
        return []
    if not isinstance(raw_rows, list):
        raise fs.TefasFormatError("TEFAS fund list rows are not a list")
    rows: List[Dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        point = fs._normalize_history_row(raw)
        if not point:
            continue
        point["source"] = fs.TEFAS_LIST_SNAPSHOT_SOURCE
        point["source_url"] = source_url
        rows.append(point)
    return rows


def _normalize_fintables_udf_history_payload(
    payload: Dict[str, Any],
    *,
    fund_code: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    fs = _fund_service()
    normalized_code = fs.normalize_fund_code(fund_code)
    status = str(payload.get("s") or "").strip().lower()
    if status != "ok":
        error = payload.get("errmsg") or payload.get("error") or status
        raise fs.FintablesUpstreamError(f"Fintables UDF history error: {error or 'missing status'}")

    timestamps = payload.get("t")
    closes = payload.get("c")
    if not isinstance(timestamps, list) or not isinstance(closes, list):
        raise fs.FintablesFormatError("Fintables UDF history missing t/c arrays")
    if len(timestamps) != len(closes):
        raise fs.FintablesFormatError("Fintables UDF history t/c length mismatch")

    points: List[Dict[str, Any]] = []
    for index, raw_timestamp in enumerate(timestamps):
        point_date = fs._fund_date(raw_timestamp)
        if not point_date:
            continue
        parsed_date = date.fromisoformat(point_date)
        if parsed_date < start_date or parsed_date > end_date:
            continue
        close = closes[index] if index < len(closes) else None
        price = fs._coerce_float(close)
        raw_point = {"t": raw_timestamp, "c": close}
        points.append(
            {
                "fund_code": normalized_code,
                "date": point_date,
                "price": price,
                "source": fs.FINTABLES_UDF_HISTORY_SOURCE,
                "source_url": fs.FINTABLES_UDF_HISTORY_ENDPOINT,
                "raw": {
                    "symbol": normalized_code,
                    "resolution": "D",
                    "point": raw_point,
                },
            }
        )
    return points


def _normalize_fintables_yield_summary_payload(payload: Dict[str, Any], *, fund_code: str) -> Dict[str, Any]:
    fs = _fund_service()
    normalized_code = fs.normalize_fund_code(fund_code)
    periods: Dict[str, Dict[str, Any]] = {}
    for key in ("1w", "1m", "3m", "6m", "ytd", "1y", "3y", "5y", "oldest"):
        raw_period = payload.get(key)
        if not isinstance(raw_period, dict):
            continue
        periods[key] = {
            "prev_close_date": fs._fund_date(raw_period.get("prev_close_date")),
            "prev_close": fs._coerce_float(raw_period.get("prev_close")),
            "high": fs._coerce_float(raw_period.get("high")),
            "low": fs._coerce_float(raw_period.get("low")),
        }
    return {
        "fund_code": normalized_code,
        "source": "fintables_yield_summary",
        "source_url": fs.FINTABLES_YIELD_SUMMARY_ENDPOINT,
        "periods": periods,
        "raw": payload,
    }


class FintablesClient:
    def __init__(
        self,
        *,
        udf_history_endpoint: Optional[str] = None,
        yield_summary_endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        fs = _fund_service()
        self.udf_history_endpoint = (
            udf_history_endpoint if udf_history_endpoint is not None else fs.FINTABLES_UDF_HISTORY_ENDPOINT
        )
        self.yield_summary_endpoint = (
            yield_summary_endpoint if yield_summary_endpoint is not None else fs.FINTABLES_YIELD_SUMMARY_ENDPOINT
        )
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else fs.FINTABLES_TIMEOUT_SECONDS

    def _headers(self, fund_code: str) -> Dict[str, str]:
        fs = _fund_service()
        normalized_code = fs.normalize_fund_code(fund_code)
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://fintables.com",
            "Referer": f"{fs.FINTABLES_FUND_BASE_URL}/{normalized_code}",
            "User-Agent": fs.FINTABLES_USER_AGENT,
        }
        cookie = os.getenv("RAGFIN_FINTABLES_COOKIE", "").strip()
        authorization = os.getenv("RAGFIN_FINTABLES_AUTHORIZATION", "").strip()
        if cookie:
            headers["Cookie"] = cookie
        if authorization:
            headers["Authorization"] = authorization
        extra_headers = os.getenv("RAGFIN_FINTABLES_EXTRA_HEADERS_JSON", "").strip()
        if extra_headers:
            try:
                parsed_headers = json.loads(extra_headers)
            except json.JSONDecodeError as exc:
                raise fs.FintablesFormatError("RAGFIN_FINTABLES_EXTRA_HEADERS_JSON is not valid JSON") from exc
            if not isinstance(parsed_headers, dict):
                raise fs.FintablesFormatError("RAGFIN_FINTABLES_EXTRA_HEADERS_JSON must be an object")
            for key, value in parsed_headers.items():
                header_name = str(key).strip()
                if header_name and value is not None:
                    headers[header_name] = str(value)
        return headers

    def _curl_headers(self, fund_code: str) -> Dict[str, str]:
        headers = dict(self._headers(fund_code))
        headers.pop("Origin", None)
        headers.pop("Referer", None)
        headers["Accept"] = "*/*"
        headers["Cache-Control"] = "no-cache"
        if "User-Agent" not in headers or not headers["User-Agent"].strip():
            headers["User-Agent"] = "PostmanRuntime/7.51.0"
        return headers

    def _get_json(self, endpoint: str, *, params: Dict[str, Any], fund_code: str, context: str) -> Dict[str, Any]:
        fs = _fund_service()
        proxy_configured = bool(fs.FINTABLES_PROXY_BASE_URL and fs.FINTABLES_PROXY_TOKEN)

        def _run_direct() -> Dict[str, Any]:
            client = fs._get_fintables_client(self.timeout_seconds)
            headers = self._headers(fund_code)
            try:
                try:
                    response = client.get(endpoint, params=params, headers=headers, timeout=self.timeout_seconds)
                except fs.httpx.RemoteProtocolError:
                    response = client.get(endpoint, params=params, headers=headers, timeout=self.timeout_seconds)
            except (fs.httpx.TimeoutException, fs.httpx.ConnectError) as exc:
                if fs.FINTABLES_CURL_FALLBACK_ENABLED:
                    raise fs.FintablesUpstreamError(
                        f"{context}: {fs.FINTABLES_GATE_BLOCKED_MESSAGE} ({exc})"
                    ) from exc
                raise
            finally:
                if client is not fs._FINTABLES_CLIENT:
                    _close_ephemeral_client(client)
            return _seam("_decode_fintables_json_response")(
                response.status_code,
                dict(response.headers),
                response.content,
                context=context,
            )

        def _run_curl(curl_headers: Dict[str, str]) -> Dict[str, Any]:
            return _seam("_fintables_curl_payload")(
                endpoint,
                params=params,
                headers=curl_headers,
                timeout_seconds=self.timeout_seconds,
                context=context,
            )

        def _run_curl_cffi(curl_headers: Dict[str, str]) -> Dict[str, Any]:
            return _seam("_fintables_curl_cffi_payload")(
                endpoint,
                params=params,
                headers=curl_headers,
                timeout_seconds=self.timeout_seconds,
                context=context,
            )

        def _run_proxy() -> Dict[str, Any]:
            return _seam("_fintables_proxy_payload")(
                endpoint,
                params=params,
                timeout_seconds=self.timeout_seconds,
                context=context,
            )

        tried: set[str] = set()
        prior_errors: Dict[str, Exception] = {}

        if fs.FINTABLES_CURL_FALLBACK_ENABLED:
            sticky = fs._get_fintables_sticky_transport()
            if sticky == "proxy" and proxy_configured:
                try:
                    return _run_proxy()
                except fs.FintablesUpstreamError as exc:
                    fs._clear_fintables_sticky_transport()
                    tried.add("proxy")
                    prior_errors["proxy"] = exc
            elif sticky == "curl_cffi":
                try:
                    return _run_curl_cffi(self._curl_headers(fund_code))
                except fs.FintablesUpstreamError as exc:
                    fs._clear_fintables_sticky_transport()
                    tried.add("curl_cffi")
                    prior_errors["curl_cffi"] = exc
            elif sticky == "curl":
                try:
                    return _run_curl(self._curl_headers(fund_code))
                except fs.FintablesUpstreamError as exc:
                    fs._clear_fintables_sticky_transport()
                    tried.add("curl")
                    prior_errors["curl"] = exc

        try:
            return _run_direct()
        except fs.FintablesUpstreamError as exc:
            if not fs.FINTABLES_CURL_FALLBACK_ENABLED or fs.FINTABLES_GATE_BLOCKED_MESSAGE not in str(exc):
                raise
            curl_headers = self._curl_headers(fund_code)
            candidates: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
                ("curl", lambda: _run_curl(curl_headers)),
                ("curl_cffi", lambda: _run_curl_cffi(curl_headers)),
            ]
            if proxy_configured:
                candidates.append(("proxy", _run_proxy))
            errors: List[Exception] = []
            for transport_name, runner in candidates:
                if transport_name in tried:
                    errors.append(prior_errors[transport_name])
                    continue
                try:
                    payload = runner()
                except fs.FintablesUpstreamError as fb_exc:
                    errors.append(fb_exc)
                    continue
                fs._set_fintables_sticky_transport(transport_name)
                return payload
            raise fs.FintablesUpstreamError("; ".join(str(err) for err in errors)) from errors[-1]

    def fetch_udf_history(self, fund_code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        fs = _fund_service()
        normalized_code = fs.normalize_fund_code(fund_code)
        if not normalized_code or start_date > end_date:
            return []
        params = {
            "symbol": normalized_code,
            "resolution": "D",
            "from": _seam("_unix_timestamp_for_date")(start_date),
            "to": _seam("_unix_timestamp_for_date")(end_date, end_of_day=True),
        }
        try:
            payload = self._get_json(
                self.udf_history_endpoint,
                params=params,
                fund_code=normalized_code,
                context="Fintables UDF history",
            )
        except fs.httpx.HTTPError as exc:
            raise fs.FintablesUpstreamError(f"Fintables UDF history request failed: {exc}") from exc
        return _seam("_normalize_fintables_udf_history_payload")(
            payload,
            fund_code=normalized_code,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_yield_summary(self, fund_code: str) -> Dict[str, Any]:
        fs = _fund_service()
        normalized_code = fs.normalize_fund_code(fund_code)
        if not normalized_code:
            return {
                "fund_code": "",
                "source": "fintables_yield_summary",
                "source_url": self.yield_summary_endpoint,
                "periods": {},
                "raw": {},
            }
        try:
            payload = self._get_json(
                self.yield_summary_endpoint,
                params={"code": normalized_code},
                fund_code=normalized_code,
                context="Fintables yield summary",
            )
        except fs.httpx.HTTPError as exc:
            raise fs.FintablesUpstreamError(f"Fintables yield summary request failed: {exc}") from exc
        return _seam("_normalize_fintables_yield_summary_payload")(payload, fund_code=normalized_code)


def fetch_fintables_udf_history(fund_code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    return _seam("FintablesClient")().fetch_udf_history(fund_code, start_date, end_date)


def fetch_fintables_yield_summary(fund_code: str) -> Dict[str, Any]:
    return _seam("FintablesClient")().fetch_yield_summary(fund_code)


def _tefasfon_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def _tefasfon_adapter_version() -> Optional[str]:
    try:
        return importlib_metadata.version("tefasfon")
    except Exception:
        return None


def _clean_dataframe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _dataframe_records(value: Any, *, context: str) -> List[Dict[str, Any]]:
    fs = _fund_service()
    if value is None:
        return []
    try:
        if bool(getattr(value, "empty", False)):
            return []
    except Exception:
        pass
    try:
        records = value.to_dict(orient="records")
    except Exception as exc:
        raise fs.TefasFormatError(f"{context} did not return a DataFrame-like object") from exc
    if not isinstance(records, list):
        raise fs.TefasFormatError(f"{context} records are not a list")
    cleaned: List[Dict[str, Any]] = []
    clean_value = _seam("_clean_dataframe_value")
    for record in records:
        if isinstance(record, dict):
            cleaned.append({str(key): clean_value(item) for key, item in record.items()})
    return cleaned


def _tefasfon_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    source: str,
    fund_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    fs = _fund_service()
    adapter_version = _seam("_tefasfon_adapter_version")()
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        point = dict(row)
        point["source"] = source
        point["source_url"] = fs.TEFASFON_SOURCE_URL
        if fund_type:
            point.setdefault("fund_type_code", fund_type)
        if adapter_version:
            point.setdefault("adapter_version", adapter_version)
        normalized_rows.append(point)
    return normalized_rows


def _merge_tefasfon_returns(
    fund_rows: Iterable[Dict[str, Any]],
    return_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    fs = _fund_service()
    normalize = fs.normalize_fund_code
    returns_by_code = {
        normalize(str(row.get("fonKodu") or row.get("fund_code") or "")): row
        for row in return_rows
        if isinstance(row, dict) and normalize(str(row.get("fonKodu") or row.get("fund_code") or ""))
    }
    merged_rows: List[Dict[str, Any]] = []
    for row in fund_rows:
        if not isinstance(row, dict):
            continue
        merged = dict(row)
        code = normalize(str(merged.get("fonKodu") or merged.get("fund_code") or ""))
        return_row = returns_by_code.get(code)
        if return_row:
            for key, value in return_row.items():
                if key in {"source", "source_url"}:
                    continue
                if value is not None and (merged.get(key) is None or key.startswith("getiri") or key == "riskDegeri"):
                    merged[key] = value
            merged["returns_source"] = fs.TEFASFON_RETURNS_SOURCE
        merged_rows.append(merged)
    return merged_rows


def _range_returns_by_code(return_rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    fs = _fund_service()
    normalize = fs.normalize_fund_code
    coerce_float = fs._coerce_float
    returns: Dict[str, float] = {}
    for row in return_rows:
        if not isinstance(row, dict):
            continue
        code = normalize(str(row.get("fonKodu") or row.get("fund_code") or ""))
        value = coerce_float(row.get("getiriOrani"))
        if code and value is not None:
            returns[code] = value
    return returns


def _merge_tefasfon_range_returns(
    fund_rows: Iterable[Dict[str, Any]],
    *,
    daily_return_rows: Iterable[Dict[str, Any]] = (),
    weekly_return_rows: Iterable[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    fs = _fund_service()
    normalize = fs.normalize_fund_code
    by_code_fn = _seam("_range_returns_by_code")
    daily_by_code = by_code_fn(daily_return_rows)
    weekly_by_code = by_code_fn(weekly_return_rows)
    merged_rows: List[Dict[str, Any]] = []
    for row in fund_rows:
        if not isinstance(row, dict):
            continue
        merged = dict(row)
        code = normalize(str(merged.get("fonKodu") or merged.get("fund_code") or ""))
        daily_return = daily_by_code.get(code)
        weekly_return = weekly_by_code.get(code)
        if daily_return is not None:
            merged["daily_return"] = daily_return
            merged["gunlukGetiri"] = daily_return
        if weekly_return is not None:
            merged["return_1w"] = weekly_return
            merged["getiri1h"] = weekly_return
        if daily_return is not None or weekly_return is not None:
            merged["range_returns_source"] = fs.TEFASFON_RETURNS_SOURCE
        merged_rows.append(merged)
    return merged_rows


def _coerce_tefas_percentage(value: Any) -> Optional[float]:
    """Convert tefasfon's Turkish-decimal percentage strings (e.g. ``"1,65"``)
    or numeric values to a float in percentage units. Returns ``None`` when
    the value is missing or non-numeric."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:  # NaN
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    text = text.replace("%", "").replace(" ", "")
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


# Stopaj (withholding tax) by fund category. Source: GVK Geçici 67. madde and
# the Resmî Gazete updates on tax rates for collective investment vehicles.
# Hisse senedi yoğun fonlar (HSY) %0, normal yatırım fonları %10, eurobond /
# döviz fonları için ayrı oranlar mevcut. Yatırımcı tarafında nihai oran fonu
# elde tutma süresine göre değişebileceği için "varsayılan" oranı veriyoruz ve
# UI bunu olduğu gibi gösteriyor.
_FUND_TAX_RULES = (
    # ETFler ve borsa yatırım fonları %0 stopaj
    (("borsa yatırım", "byf", "exchange"), "%0"),
    # Hisse senedi yoğun fonlar — HSYF kapsamına girer ve %0 stopaja tabidir
    (("hisse senedi yoğun", "hisse yoğun", "hisse senedi şemsiye", "hsyf"), "%0"),
    # Emeklilik yatırım fonları için stopaj uygulanmaz
    (("emeklilik", "pension"), "—"),
    # Serbest fonlar normal yatırım fonu rejimine tabidir → %10
    (("serbest",), "%10"),
    # Para piyasası, kıymetli maden ve karma fonlar varsayılan %10 stopaja tabidir
    (("para piyasası", "kıymetli maden", "katılım", "değişken", "karma", "borçlanma", "fon sepeti"), "%10"),
)


def _fund_tax_info(fund_type: Optional[str]) -> Optional[str]:
    if not fund_type:
        return None
    text = str(fund_type).strip().lower()
    if not text:
        return None
    for keywords, rate in _seam("_FUND_TAX_RULES"):
        if any(keyword in text for keyword in keywords):
            return rate
    # Düzenlenmiş "yatırım fonu" / "şemsiye fon" kategorileri varsayılan olarak %10
    if "yatırım fonu" in text or "şemsiye" in text:
        return "%10"
    return None


def _merge_tefasfon_management_fees(
    fund_rows: Iterable[Dict[str, Any]],
    fee_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach management fee / total expense ratio fields from a tefasfon
    ``get_returns(basis="MB")`` payload to fund snapshot rows."""

    fs = _fund_service()
    normalize = fs.normalize_fund_code
    coerce_pct = _seam("_coerce_tefas_percentage")
    fees_by_code: Dict[str, Dict[str, Any]] = {}
    for row in fee_rows:
        if not isinstance(row, dict):
            continue
        code = normalize(str(row.get("fonKodu") or row.get("fund_code") or ""))
        if not code:
            continue
        applied = coerce_pct(row.get("uygulananYu1Y"))
        prospectus = coerce_pct(row.get("fonIcTuzukYu1G"))
        ter = coerce_pct(row.get("fonTopGiderKesoran"))
        annual_return = coerce_pct(row.get("yillikGetiri"))
        if applied is None and prospectus is None and ter is None:
            continue
        fees_by_code[code] = {
            "management_fee_applied": applied,
            "management_fee_prospectus": prospectus,
            "total_expense_ratio": ter,
            "annual_return_pct": annual_return,
        }
    merged_rows: List[Dict[str, Any]] = []
    for row in fund_rows:
        if not isinstance(row, dict):
            continue
        merged = dict(row)
        code = normalize(str(merged.get("fonKodu") or merged.get("fund_code") or ""))
        info = fees_by_code.get(code)
        if info:
            for key, value in info.items():
                if value is not None and merged.get(key) is None:
                    merged[key] = value
            merged["management_fee_source"] = fs.TEFASFON_RETURNS_SOURCE
        merged_rows.append(merged)
    return merged_rows


def _yield_periods_from_points(points: Iterable[Dict[str, Any]], normalized_code: str) -> Dict[str, Dict[str, Any]]:
    fs = _fund_service()
    ordered = fs._valid_performance_points(points, normalized_code)
    if not ordered:
        return {}
    latest = ordered[-1]
    latest_date = date.fromisoformat(latest["date"])
    period_targets = {
        "1w": latest_date - timedelta(days=7),
        "1m": latest_date - timedelta(days=30),
        "3m": latest_date - timedelta(days=90),
        "6m": latest_date - timedelta(days=180),
        "ytd": date(latest_date.year, 1, 1),
        "1y": latest_date - timedelta(days=365),
    }
    periods: Dict[str, Dict[str, Any]] = {}
    for key, target in period_targets.items():
        base = fs._point_on_or_before(ordered, target)
        period_points = [
            point
            for point in ordered
            if target <= date.fromisoformat(point["date"]) <= latest_date and fs._coerce_float(point.get("price")) is not None
        ]
        prices = [float(point["price"]) for point in period_points if fs._coerce_float(point.get("price")) is not None]
        if not base and not prices:
            continue
        periods[key] = {
            "prev_close_date": base.get("date") if base else None,
            "prev_close": fs._coerce_float(base.get("price")) if base else None,
            "high": max(prices) if prices else None,
            "low": min(prices) if prices else None,
        }
    oldest = ordered[0]
    periods["oldest"] = {
        "prev_close_date": oldest.get("date"),
        "prev_close": fs._coerce_float(oldest.get("price")),
        "high": max(float(point["price"]) for point in ordered),
        "low": min(float(point["price"]) for point in ordered),
    }
    return periods


def _yield_periods_from_return_row(
    row: Dict[str, Any],
    *,
    latest_price: Optional[float],
    latest_date: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    fs = _fund_service()
    price = fs._coerce_float(latest_price)
    if price is None or price <= 0:
        return {}
    period_columns = {
        "1w": "getiri1h",
        "1m": "getiri1a",
        "3m": "getiri3a",
        "6m": "getiri6a",
        "ytd": "getiriyb",
        "1y": "getiri1y",
        "3y": "getiri3y",
        "5y": "getiri5y",
    }
    periods: Dict[str, Dict[str, Any]] = {}
    for period, column in period_columns.items():
        return_pct = fs._coerce_float(row.get(column))
        if return_pct is None:
            continue
        denominator = 1.0 + (return_pct / 100.0)
        if denominator <= 0:
            continue
        periods[period] = {
            "prev_close_date": None,
            "prev_close": price / denominator,
            "high": None,
            "low": None,
            "latest_date": latest_date,
            "return_pct": return_pct,
        }
    return periods


def _is_tefasfon_adapter_unavailable(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "timed out" in message
        or "not installed" in message
        or "cannot be imported" in message
        or "getter internals are unavailable" in message
    )


def _generate_date_chunks(
    start_date: date,
    end_date: date,
    max_chunk_days: Optional[int] = None,
) -> List[Tuple[date, date]]:
    """Partition [start_date, end_date] into contiguous, non-overlapping slices of at most max_chunk_days."""
    if start_date > end_date:
        return []
    effective_max = _fund_service().TEFAS_HISTORY_CHUNK_MAX_DAYS if max_chunk_days is None else max_chunk_days
    span_limit = max(1, int(effective_max))
    chunks: List[Tuple[date, date]] = []
    current_start = start_date
    while current_start <= end_date:
        current_end = min(end_date, current_start + timedelta(days=span_limit - 1))
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    return chunks


class TefasFonClient:
    def __init__(self, *, fund_types: Optional[Iterable[str]] = None) -> None:
        fs = _fund_service()
        effective_types = fs.TEFAS_FUND_TYPES if fund_types is None else fund_types
        normalized_types = [
            str(item).strip().upper()
            for item in effective_types
            if str(item).strip().upper() in fs._TEFAS_ALLOWED_FUND_TYPES
        ]
        self.fund_types = tuple(normalized_types) or ("SEC",)

    def _module(self) -> Any:
        fs = _fund_service()
        try:
            import tefasfon  # type: ignore
        except Exception as exc:
            raise fs.TefasUpstreamError("tefasfon package is not installed or cannot be imported") from exc
        return tefasfon

    def _call_dataframe(self, function_name: str, *, context: str, **kwargs: Any) -> List[Dict[str, Any]]:
        fs = _fund_service()
        module = self._module()
        function = getattr(module, function_name, None)
        if function is None:
            raise fs.TefasFormatError(f"tefasfon missing {function_name}")
        timeout_seconds = max(1.0, fs.TEFAS_TIMEOUT_SECONDS)
        future = _TEFASFON_EXECUTOR.submit(function, **kwargs)
        try:
            frame = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise fs.TefasUpstreamError(f"{context} timed out after {timeout_seconds:g}s") from exc
        except Exception as exc:
            raise fs.TefasUpstreamError(f"{context} failed: {exc}") from exc
        return _seam("_dataframe_records")(frame, context=context)

    def _call_records(self, callback: Callable[[], List[Dict[str, Any]]], *, context: str) -> List[Dict[str, Any]]:
        fs = _fund_service()
        timeout_seconds = max(1.0, fs.TEFAS_TIMEOUT_SECONDS)
        future = _TEFASFON_EXECUTOR.submit(callback)
        try:
            records = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise fs.TefasUpstreamError(f"{context} timed out after {timeout_seconds:g}s") from exc
        except Exception as exc:
            raise fs.TefasUpstreamError(f"{context} failed: {exc}") from exc
        return records

    def fetch_funds(
        self,
        *,
        start_date: date,
        end_date: date,
        fund_codes: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if start_date > end_date:
            return []
        fs = _fund_service()
        normalize = fs.normalize_fund_code
        codes = sorted({normalize(code) for code in list(fund_codes or []) if normalize(code)})
        rows: List[Dict[str, Any]] = []
        format_date = _seam("_tefasfon_date")
        for fund_type in self.fund_types:
            kwargs: Dict[str, Any] = {
                "fund_type": fund_type,
                "start_date": format_date(start_date),
                "end_date": format_date(end_date),
            }
            if codes:
                kwargs["fund_codes"] = codes
            try:
                records = self._call_dataframe("get_funds", context="tefasfon get_funds", **kwargs)
            except fs.TefasUpstreamError as exc:
                if codes and _seam("_is_tefasfon_adapter_unavailable")(exc):
                    fallback = _seam("TefasClient")().fetch_fund_history(
                        fund_codes=codes,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    rows.extend(fallback)
                    continue
                raise
            if codes and not records:
                fallback = _seam("TefasClient")().fetch_fund_history(
                    fund_codes=codes,
                    start_date=start_date,
                    end_date=end_date,
                )
                if fallback:
                    rows.extend(fallback)
                    continue
            rows.extend(_seam("_tefasfon_rows")(records, source=fs.TEFASFON_FUNDS_SOURCE, fund_type=fund_type))
        return rows

    def fetch_returns(
        self,
        *,
        fund_codes: Optional[Iterable[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        fs = _fund_service()
        normalize = fs.normalize_fund_code
        codes = sorted({normalize(code) for code in list(fund_codes or []) if normalize(code)})
        rows: List[Dict[str, Any]] = []
        format_date = _seam("_tefasfon_date")
        for fund_type in self.fund_types:
            kwargs: Dict[str, Any] = {"fund_type": fund_type, "basis": "RB"}
            if start_date and end_date:
                kwargs["start_date"] = format_date(start_date)
                kwargs["end_date"] = format_date(end_date)
            if codes:
                kwargs["fund_codes"] = codes
            records = self._call_dataframe("get_returns", context="tefasfon get_returns", **kwargs)
            rows.extend(_seam("_tefasfon_rows")(records, source=fs.TEFASFON_RETURNS_SOURCE, fund_type=fund_type))
        return rows

    def fetch_management_fees(
        self,
        *,
        fund_codes: Optional[Iterable[str]] = None,
        lookback_days: int = 21,
        as_of: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch management-fee/expense-ratio rows via tefasfon get_returns(basis='MB').

        The ``basis='MB'`` endpoint exposes ``uygulananYu1Y`` (applied annual fee),
        ``fonIcTuzukYu1G`` (prospectus annual fee) and ``fonTopGiderKesoran``
        (total expense ratio). Tefas requires a date window for this call, and
        the API fans out one request per month inside the window. We keep the
        window small (3 weeks by default) so the call is fast while still
        guaranteeing a recent business-day match.
        """
        fs = _fund_service()
        end = as_of or fs.date.today()
        start = end - timedelta(days=max(7, int(lookback_days)))
        normalize = fs.normalize_fund_code
        codes = sorted({normalize(code) for code in list(fund_codes or []) if normalize(code)})
        rows: List[Dict[str, Any]] = []
        format_date = _seam("_tefasfon_date")
        for fund_type in self.fund_types:
            kwargs: Dict[str, Any] = {
                "fund_type": fund_type,
                "basis": "MB",
                "start_date": format_date(start),
                "end_date": format_date(end),
            }
            if codes:
                kwargs["fund_codes"] = codes
            try:
                records = self._call_dataframe(
                    "get_returns",
                    context="tefasfon get_returns MB",
                    **kwargs,
                )
            except fs.TefasUpstreamError:
                continue
            rows.extend(_seam("_tefasfon_rows")(records, source=fs.TEFASFON_RETURNS_SOURCE, fund_type=fund_type))
        return rows

    def fetch_daily_funds_snapshot(self, as_of: date) -> List[Dict[str, Any]]:
        fs = _fund_service()
        rows: List[Dict[str, Any]] = []
        direct_errors: List[str] = []
        for fund_type in self.fund_types:
            try:
                rows.extend(self._fetch_funds_snapshot_direct(as_of, fund_type=fund_type))
            except fs.TefasUpstreamError as exc:
                direct_errors.append(str(exc))
        if rows:
            return rows
        # The package-level getter can silently stop at its first 1000-row
        # page. Use the paginated direct client before the legacy one-page
        # fallback so catalogue refreshes can recover the full universe.
        direct_client = _seam("TefasClient")()
        paginated_rows: List[Dict[str, Any]] = []
        for fund_type in self.fund_types:
            try:
                paginated_rows.extend(
                    direct_client.fetch_funds_direct(
                        start_date=as_of,
                        end_date=as_of,
                        fund_type=fund_type,
                    )
                )
            except fs.TefasUpstreamError as exc:
                direct_errors.append(str(exc))
        if paginated_rows:
            return paginated_rows
        try:
            fallback_rows = direct_client.fetch_fund_list_snapshot(target_date=as_of)
        except fs.TefasUpstreamError as exc:
            if direct_errors:
                direct_errors.append(str(exc))
            elif _seam("_is_tefasfon_adapter_unavailable")(exc):
                raise
            else:
                return []
        else:
            if fallback_rows:
                return fallback_rows
        if direct_errors:
            raise fs.TefasUpstreamError("; ".join(direct_errors))
        return []

    def fetch_portfolio(
        self,
        *,
        fund_code: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        fs = _fund_service()
        normalize = fs.normalize_fund_code
        normalized = normalize(fund_code)
        if not normalized or start_date > end_date:
            return []
        rows: List[Dict[str, Any]] = []
        direct_rate_limit_error: Optional[BaseException] = None
        direct_request_errors: List[str] = []
        for fund_type in self.fund_types:
            try:
                rows.extend(
                    self._fetch_portfolio_direct_request(
                        normalized,
                        start_date,
                        end_date,
                        fund_type=fund_type,
                    )
                )
            except fs.TefasRateLimitError as exc:
                direct_rate_limit_error = exc
                break
            except fs.TefasUpstreamError as exc:
                direct_request_errors.append(str(exc))
                continue
        if rows:
            return rows
        if direct_rate_limit_error is not None:
            raise direct_rate_limit_error
        if not direct_request_errors:
            # Direct TEFAS endpoint returned an authoritative valid empty payload;
            # skip redundant tefasfon/getter retries against the same endpoint.
            return []

        # Keep the package's getter implementation as a compatibility fallback.
        # TEFAS occasionally changes the public payload contract; this prevents a
        # direct-client change from turning a temporary mismatch into empty UI data.
        direct_errors: List[str] = []
        for fund_type in self.fund_types:
            try:
                rows.extend(self._fetch_portfolio_direct(normalized, start_date, end_date, fund_type=fund_type))
            except fs.TefasUpstreamError as exc:
                direct_errors.append(str(exc))
        if rows or not direct_errors:
            return rows
        format_date = _seam("_tefasfon_date")
        for fund_type in self.fund_types:
            records = self._call_dataframe(
                "get_portfolio",
                context="tefasfon get_portfolio",
                fund_type=fund_type,
                start_date=format_date(start_date),
                end_date=format_date(end_date),
                fund_codes=[normalized],
            )
            normalized_rows = _seam("_tefasfon_rows")(records, source=fs.TEFASFON_PORTFOLIO_SOURCE, fund_type=fund_type)
            coded_rows = [
                (row, normalize(str(row.get("fonKodu") or row.get("fund_code") or "")))
                for row in normalized_rows
            ]
            if any(code for _row, code in coded_rows):
                rows.extend(row for row, code in coded_rows if code == normalized)
            else:
                rows.extend(normalized_rows)
        return rows

    def _fetch_portfolio_direct_request(
        self,
        fund_code: str,
        start_date: date,
        end_date: date,
        *,
        fund_type: str,
    ) -> List[Dict[str, Any]]:
        rows = _seam("TefasClient")().fetch_portfolio_direct(
            fund_code=fund_code,
            start_date=start_date,
            end_date=end_date,
            fund_type=fund_type,
        )
        for row in rows:
            if isinstance(row, dict):
                row.setdefault("adapter_used", "direct_tefas_request")
        return rows

    def _fetch_funds_snapshot_direct(
        self,
        as_of: date,
        *,
        fund_type: str,
    ) -> List[Dict[str, Any]]:
        fs = _fund_service()
        fund_type_code = str(fund_type).strip().upper()

        def fetch_records() -> List[Dict[str, Any]]:
            from tefasfon import getter as tefas_getter  # type: ignore

            fon_tipi = tefas_getter._FUND_TIPI[fund_type_code]
            portal_url = tefas_getter._FUND_PORTAL[fund_type_code]
            fund_url_param = tefas_getter._FUND_URL_PARAM[fund_type_code]
            endpoint = tefas_getter._API_ENDPOINT["general_information"]
            as_of_iso = as_of.strftime("%Y-%m-%d")
            session = tefas_getter._new_session(portal_url, as_of_iso, as_of_iso, fund_url_param)
            try:
                base_payload: Dict[str, Any] = {
                    "fonTipi": fon_tipi,
                    "fonKodu": None,
                    "aramaMetni": None,
                    "fonTurKod": None,
                    "fonGrubu": None,
                    "sfonTurKod": None,
                    "basTarih": as_of.strftime("%Y%m%d"),
                    "bitTarih": as_of.strftime("%Y%m%d"),
                    "basSira": 1,
                    "bitSira": 1,
                    "fonTurAciklama": None,
                    "dil": "TR",
                    "kurucuKod": None,
                }
                post_headers = {
                    "Content-Type": "application/json",
                    "Referer": f"{portal_url}?startDate={as_of_iso}&endDate={as_of_iso}",
                    "Origin": "https://www.tefas.gov.tr",
                }
                page_size = int(getattr(tefas_getter, "_PAGE_SIZE", 1000))
                page_delay = float(getattr(tefas_getter, "_PAGE_DELAY", 1.0))
                retry_delay = float(getattr(tefas_getter, "_RETRY_DELAY", 5.0))
                max_retries = int(getattr(tefas_getter, "_MAX_RETRIES", 3))
                collected: List[Dict[str, Any]] = []
                seen_codes: set[str] = set()
                page_index = 0
                total_pages: Optional[int] = None
                reported_total: Optional[int] = None
                empty_streak = 0
                full_page_without_terminator = False

                def post_with_retries(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                    last_err: Optional[Exception] = None
                    for attempt in range(max(1, max_retries)):
                        try:
                            response = session.post(endpoint, json=payload, headers=post_headers, timeout=30)
                        except Exception as exc:  # network errors, dns, ssl, etc.
                            last_err = exc
                            response = None
                        if response is not None and response.text and response.text.strip():
                            try:
                                return response.json()
                            except Exception as exc:
                                last_err = exc
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                    if last_err is not None:
                        raise last_err
                    return None

                while True:
                    bas_sira = page_index * page_size + 1
                    payload = {
                        **base_payload,
                        "basSira": bas_sira,
                        "bitSira": bas_sira + page_size - 1,
                    }
                    body = post_with_retries(payload)
                    if not body:
                        if page_index > 0 and total_pages == 1:
                            # A full first page can be followed by an empty page
                            # when the endpoint's page count is unreliable.
                            if reported_total is None and full_page_without_terminator:
                                raise fs.TefasUpstreamError(
                                    f"tefasfon snapshot for {fund_type_code} {as_of.isoformat()} ended on a full page "
                                    "without an authoritative total row count"
                                )
                            break
                        raise fs.TefasUpstreamError(
                            f"tefasfon snapshot page {page_index + 1} returned empty body for {fund_type_code} {as_of.isoformat()}"
                        )
                    error_message = body.get("errorMessage") or body.get("errorCode")
                    if error_message:
                        raise fs.TefasUpstreamError(
                            f"tefasfon snapshot page {page_index + 1} for {fund_type_code} {as_of.isoformat()} failed: {error_message}"
                        )
                    if total_pages is None:
                        candidate = body.get("toplamSayfa") or body.get("totalPages") or 1
                        try:
                            total_pages = max(1, int(candidate))
                        except (TypeError, ValueError):
                            total_pages = 1
                    if reported_total is None:
                        for key in (
                            "toplamKayit",
                            "toplamSayi",
                            "totalCount",
                            "totalRecords",
                            "totalRows",
                            "total_count",
                            "count",
                        ):
                            value = body.get(key)
                            if value in (None, ""):
                                continue
                            try:
                                reported_total = max(0, int(float(value)))
                            except (TypeError, ValueError):
                                continue
                            break
                    rows: Optional[List[Any]] = None
                    for key in ("resultList", "data", "Data", "result", "Result", "rows", "items"):
                        value = body.get(key)
                        if isinstance(value, list):
                            rows = value
                            break
                    if rows is None:
                        rows = body if isinstance(body, list) else []
                    if not isinstance(rows, list):
                        rows = []
                    added = 0
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        fund_code = fs.normalize_fund_code(
                            str(row.get("fonKodu") or row.get("fund_code") or "")
                        )
                        if fund_code:
                            if fund_code in seen_codes:
                                continue
                            seen_codes.add(fund_code)
                        collected.append(row)
                        added += 1
                    if added == 0:
                        empty_streak += 1
                    else:
                        empty_streak = 0
                    page_index += 1
                    if reported_total is not None and len(collected) >= reported_total:
                        break
                    if len(rows) == page_size:
                        full_page_without_terminator = True
                    else:
                        if not rows and reported_total is None and full_page_without_terminator:
                            raise fs.TefasUpstreamError(
                                f"tefasfon snapshot for {fund_type_code} {as_of.isoformat()} ended on a full page "
                                "without an authoritative total row count"
                            )
                        full_page_without_terminator = False
                    if total_pages is not None and page_index >= total_pages:
                        # TEFAS has returned toplamSayfa=1 for a full page even
                        # when more rows were available. Probe the next page in
                        # that specific case; short pages remain terminal.
                        if total_pages > 1 or len(rows) < page_size:
                            break
                    if added < page_size and len(rows) < page_size:
                        # Server reported fewer pages than expected; trust the actual payload size.
                        break
                    if empty_streak >= 2:
                        raise fs.TefasUpstreamError(
                            f"tefasfon snapshot for {fund_type_code} {as_of.isoformat()} stalled after page {page_index}"
                        )
                    if page_delay > 0:
                        time.sleep(page_delay)
                if reported_total is not None and len(collected) < reported_total:
                    raise fs.TefasUpstreamError(
                        f"tefasfon snapshot for {fund_type_code} {as_of.isoformat()} was incomplete "
                        f"({len(collected)} rows vs reported {reported_total})"
                    )
                if total_pages and total_pages > 1 and page_index < total_pages:
                    raise fs.TefasUpstreamError(
                        f"tefasfon snapshot for {fund_type_code} {as_of.isoformat()} aborted at page {page_index}/{total_pages}"
                    )
                if reported_total is None and full_page_without_terminator:
                    raise fs.TefasUpstreamError(
                        f"tefasfon snapshot for {fund_type_code} {as_of.isoformat()} ended on a full page "
                        "without an authoritative total row count"
                    )
                return collected
            finally:
                if hasattr(session, "close"):
                    try:
                        session.close()
                    except Exception:
                        pass

        records = self._call_records(
            fetch_records,
            context=f"tefasfon direct funds snapshot {as_of.isoformat()}",
        )
        return _seam("_tefasfon_rows")(records, source=fs.TEFASFON_FUNDS_SOURCE, fund_type=fund_type_code)

    def _fetch_portfolio_direct(
        self,
        fund_code: str,
        start_date: date,
        end_date: date,
        *,
        fund_type: str,
    ) -> List[Dict[str, Any]]:
        fs = _fund_service()
        try:
            from tefasfon import getter as tefas_getter  # type: ignore
        except Exception as exc:
            raise fs.TefasUpstreamError("tefasfon getter internals are unavailable") from exc
        try:
            fund_type_code = str(fund_type).strip().upper()
            fon_tipi = tefas_getter._FUND_TIPI[fund_type_code]
            portal_url = tefas_getter._FUND_PORTAL[fund_type_code]
            fund_url_param = tefas_getter._FUND_URL_PARAM[fund_type_code]
            endpoint = tefas_getter._API_ENDPOINT["portfolio_breakdown"]
            start_iso = start_date.strftime("%Y-%m-%d")
            end_iso = end_date.strftime("%Y-%m-%d")
            session = tefas_getter._new_session(portal_url, start_iso, end_iso, fund_url_param)
            try:
                base_payload = {
                    "fonTipi": fon_tipi,
                    "fonKodu": fund_code,
                    "aramaMetni": fund_code,
                    "fonTurKod": None,
                    "fonGrubu": None,
                    "sfonTurKod": None,
                    "basTarih": start_date.strftime("%Y%m%d"),
                    "bitTarih": end_date.strftime("%Y%m%d"),
                    "basSira": 1,
                    "bitSira": 1000,
                    "fonTurAciklama": None,
                    "dil": "TR",
                    "kurucuKod": None,
                }
                records = tefas_getter._get_all_pages(session, endpoint, base_payload, portal_url, start_iso, end_iso)
            finally:
                if hasattr(session, "close"):
                    try:
                        session.close()
                    except Exception:
                        pass
        except Exception as exc:
            raise fs.TefasUpstreamError(f"tefasfon direct portfolio failed: {exc}") from exc
        rows = _seam("_tefasfon_rows")(records, source=fs.TEFASFON_PORTFOLIO_SOURCE, fund_type=fund_type_code)
        coded_rows = [
            (row, fs.normalize_fund_code(str(row.get("fonKodu") or row.get("fund_code") or "")))
            for row in rows
        ]
        if any(code for _row, code in coded_rows):
            return [row for row, code in coded_rows if code == fund_code]
        return rows

    def fetch_history(self, fund_code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        fs = _fund_service()
        normalize = fs.normalize_fund_code
        normalized = normalize(fund_code)
        if not normalized or start_date > end_date:
            return []

        def matching_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [
                row
                for row in rows
                if isinstance(row, dict)
                and normalize(str(row.get("fonKodu") or row.get("fund_code") or "")) == normalized
            ]

        rows_by_date: Dict[str, Dict[str, Any]] = {}
        direct_client = _seam("TefasClient")()
        direct_error: Optional[Any] = None

        def add_rows(rows: Iterable[Dict[str, Any]]) -> None:
            for row in matching_rows(rows):
                point_date = fs._fund_date(fs._first_present(row, "date", "tarih", "TARIH", "TARIHSTR"))
                if point_date:
                    if point_date in rows_by_date:
                        rows_by_date[point_date].update(row)
                    else:
                        rows_by_date[point_date] = dict(row)

        # 1. Primary: single-layer direct TEFAS query (calendar-month chunked inside TefasClient)
        try:
            direct_rows = direct_client.fetch_fund_history(
                fund_codes=[normalized],
                start_date=start_date,
                end_date=end_date,
            )
            matched_direct = matching_rows(direct_rows)
            if matched_direct:
                add_rows(matched_direct)
        except fs.TefasRateLimitError:
            raise
        except fs.TefasUpstreamError as exc:
            direct_error = exc
            fs.LOGGER.warning(
                "TEFAS direct history failed for %s [%s to %s]: %s",
                normalized,
                start_date.isoformat(),
                end_date.isoformat(),
                exc,
            )

        if rows_by_date:
            return [rows_by_date[point_date] for point_date in sorted(rows_by_date)]

        # 2. Fallback: bounded <= 90-day chunks when using tefasfon.get_funds
        chunks = _seam("_generate_date_chunks")(start_date, end_date, max_chunk_days=fs.TEFAS_HISTORY_CHUNK_MAX_DAYS)
        chunk_errors: List[Any] = []
        for chunk_start, chunk_end in chunks:
            try:
                fallback_rows = self.fetch_funds(
                    start_date=chunk_start,
                    end_date=chunk_end,
                    fund_codes=[normalized],
                )
                matched_fallback = matching_rows(fallback_rows)
                if matched_fallback:
                    add_rows(matched_fallback)
            except fs.TefasRateLimitError:
                raise
            except fs.TefasUpstreamError as exc:
                fs.LOGGER.warning(
                    "TEFAS fallback history chunk failed for %s [%s to %s]: %s",
                    normalized,
                    chunk_start.isoformat(),
                    chunk_end.isoformat(),
                    exc,
                )
                chunk_errors.append(exc)

        if rows_by_date:
            return [rows_by_date[point_date] for point_date in sorted(rows_by_date)]
        if chunk_errors:
            raise chunk_errors[-1]
        if direct_error is not None:
            raise direct_error
        return []

    def fetch_latest_fund_list_snapshot_result(
        self,
        *,
        as_of: date,
        lookback_days: Optional[int] = None,
        enrich: bool = True,
    ) -> FundSnapshotFetchResult:
        fs = _fund_service()
        effective_lookback = fs.FUNDS_MAX_LOOKBACK_CALENDAR_DAYS if lookback_days is None else lookback_days
        warnings: List[str] = []
        upstream_errors: List[str] = []
        attempted_dates: List[str] = []
        empty_dates: List[str] = []
        requested = as_of.isoformat()
        window = min(fs.FUNDS_MAX_LOOKBACK_CALENDAR_DAYS, max(1, int(effective_lookback)))
        consecutive_errors = 0
        for offset in range(window):
            target_date = as_of - timedelta(days=offset)
            target_text = target_date.isoformat()
            attempted_dates.append(target_text)
            try:
                # ``tefasfon.get_funds`` can return only its first 1000-row
                # page without raising an error. The daily snapshot path uses
                # the paginated request implementation instead.
                fund_rows = self.fetch_daily_funds_snapshot(target_date)
            except fs.TefasUpstreamError as exc:
                message = f"tefasfon_funds failed for {target_text}: {exc}"
                warnings.append(message)
                upstream_errors.append(message)
                consecutive_errors += 1
                if _seam("_is_tefasfon_adapter_unavailable")(exc) or _seam("_is_tefas_rate_limit")(exc) or consecutive_errors >= 3:
                    break
                continue
            if not fund_rows:
                consecutive_errors = 0
                empty_dates.append(target_text)
                continue
            if not enrich:
                fund_rows, skipped_closed, _skipped_unknown = fs._filter_tefas_open_rows(fund_rows)
                if skipped_closed:
                    warnings.append(f"tefas_open_only skipped {skipped_closed} closed fund rows")
                if fs.TEFAS_OPEN_ONLY and not fund_rows:
                    empty_dates.append(target_text)
                    continue
            else:
                previous_business_day = fs._previous_turkey_market_business_day(target_date)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=3,
                    thread_name_prefix="tefas-enrich",
                ) as executor:
                    returns_fut = executor.submit(self.fetch_returns)
                    fees_fut = executor.submit(self.fetch_management_fees, as_of=target_date)
                    daily_fut = executor.submit(
                        self.fetch_returns,
                        start_date=previous_business_day,
                        end_date=target_date,
                    )
                    weekly_fut = executor.submit(
                        self.fetch_returns,
                        start_date=target_date - timedelta(days=7),
                        end_date=target_date,
                    )
                    try:
                        fund_rows = _seam("_merge_tefasfon_returns")(fund_rows, returns_fut.result())
                    except fs.TefasUpstreamError as exc:
                        warnings.append(f"tefasfon_returns failed: {exc}")
                    try:
                        fund_rows = _seam("_merge_tefasfon_management_fees")(fund_rows, fees_fut.result())
                    except fs.TefasUpstreamError as exc:
                        warnings.append(f"tefasfon_management_fees failed: {exc}")

                    daily_return_rows: List[Dict[str, Any]] = []
                    weekly_return_rows: List[Dict[str, Any]] = []
                    try:
                        daily_return_rows = daily_fut.result()
                    except fs.TefasUpstreamError as exc:
                        warnings.append(f"tefasfon_range_returns failed: {exc}")
                    try:
                        weekly_return_rows = weekly_fut.result()
                    except fs.TefasUpstreamError as exc:
                        warnings.append(f"tefasfon_range_returns failed: {exc}")
                    if daily_return_rows or weekly_return_rows:
                        fund_rows = _seam("_merge_tefasfon_range_returns")(
                            fund_rows,
                            daily_return_rows=daily_return_rows,
                            weekly_return_rows=weekly_return_rows,
                        )
                open_rows, skipped_closed, _skipped_unknown = fs._filter_tefas_open_rows(fund_rows)
                if skipped_closed:
                    warnings.append(f"tefas_open_only skipped {skipped_closed} closed fund rows")
                if fs.TEFAS_OPEN_ONLY and not open_rows:
                    empty_dates.append(target_text)
                    continue
                fund_rows = open_rows
            resolved_dates = [
                fs._fund_date(fs._first_present(row, "date", "tarih", "TARIH", "TARIHSTR"))
                for row in fund_rows
                if isinstance(row, dict)
            ]
            resolved = max((item for item in resolved_dates if item), default=target_text)
            return fs.FundSnapshotFetchResult(
                rows=list(fund_rows),
                warnings=warnings,
                resolution_status="available" if offset == 0 else "resolved_previous_date",
                requested_as_of=requested,
                resolved_as_of=resolved,
                attempted_dates=attempted_dates,
                empty_dates=empty_dates,
                upstream_errors=upstream_errors,
                source=fs.TEFASFON_FUNDS_SOURCE,
            )
        return fs.FundSnapshotFetchResult(
            warnings=warnings,
            resolution_status="upstream_unavailable" if upstream_errors else "not_published",
            requested_as_of=requested,
            attempted_dates=attempted_dates,
            empty_dates=empty_dates,
            upstream_errors=upstream_errors,
            source=fs.TEFASFON_FUNDS_SOURCE,
        )

    def fetch_latest_fund_list_snapshot(
        self,
        *,
        as_of: date,
        lookback_days: Optional[int] = None,
        enrich: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        result = self.fetch_latest_fund_list_snapshot_result(
            as_of=as_of,
            lookback_days=lookback_days,
            enrich=enrich,
        )
        return result.rows, result.warnings

    def fetch_latest_portfolio(
        self,
        fund_code: str,
        *,
        as_of: date,
        lookback_days: int = 10,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        fs = _fund_service()
        warnings: List[str] = []
        start_date = as_of - timedelta(days=max(1, lookback_days))
        try:
            rows = self.fetch_portfolio(fund_code=fund_code, start_date=start_date, end_date=as_of)
        except fs.TefasUpstreamError as exc:
            warnings.append(f"tefasfon_portfolio failed for {start_date.isoformat()}..{as_of.isoformat()}: {exc}")
            return [], warnings
        if not rows:
            warnings.append(f"tefasfon_portfolio returned no rows for {start_date.isoformat()}..{as_of.isoformat()}")
            return [], warnings
        dated_rows: List[Tuple[str, Dict[str, Any]]] = []
        for row in rows:
            row_date = fs._fund_date(row.get("tarih") or row.get("TARIH") or row.get("date"))
            if row_date:
                dated_rows.append((row_date, row))
        if dated_rows:
            latest_date = max(d for d, _ in dated_rows)
            return [row for d, row in dated_rows if d == latest_date], warnings
        return rows, warnings

    def fetch_yield_summary(
        self,
        fund_code: str,
        *,
        as_of: Optional[date] = None,
        latest_price: Optional[float] = None,
        latest_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        fs = _fund_service()
        normalized = fs.normalize_fund_code(fund_code)
        raw_returns: List[Dict[str, Any]] = []
        periods: Dict[str, Dict[str, Any]] = {}
        points: List[Dict[str, Any]] = []
        history_error: Optional[BaseException] = None
        effective_end = as_of or fs.date.today()
        raw_returns = self.fetch_returns(fund_codes=[normalized])
        from_return_row = _seam("_yield_periods_from_return_row")
        if raw_returns:
            periods.update(
                from_return_row(
                    raw_returns[0],
                    latest_price=latest_price,
                    latest_date=latest_date,
                )
            )
        if "1w" not in periods:
            try:
                weekly_returns = self.fetch_returns(
                    fund_codes=[normalized],
                    start_date=effective_end - timedelta(days=7),
                    end_date=effective_end,
                )
                if weekly_returns:
                    weekly_row = dict(weekly_returns[0])
                    weekly_row["getiri1h"] = weekly_row.get("getiriOrani")
                    periods.update(
                        from_return_row(
                            weekly_row,
                            latest_price=latest_price,
                            latest_date=latest_date,
                        )
                    )
            except fs.TefasUpstreamError as exc:
                history_error = exc

        if not periods or "1w" not in periods:
            effective_start = effective_end - timedelta(days=max(370, fs.FUNDS_AUTO_FETCH_LOOKBACK_DAYS))
            try:
                points = self.fetch_history(normalized, effective_start, effective_end)
            except fs.TefasUpstreamError as exc:
                history_error = exc
                points = []
            history_periods = _seam("_yield_periods_from_points")(points, normalized)
            for key, value in history_periods.items():
                if key not in periods or periods[key].get("prev_close") is None:
                    periods[key] = value
        if not periods:
            if history_error is not None:
                raise history_error
            raise fs.TefasFormatError("tefasfon_funds returned no usable history for yield summary")
        return {
            "fund_code": normalized,
            "source": fs.TEFASFON_RETURNS_SOURCE if raw_returns else fs.TEFASFON_FUNDS_SOURCE,
            "source_url": fs.TEFASFON_SOURCE_URL,
            "periods": periods,
            "raw": {
                "points_count": len(points),
                "returns": raw_returns[:1],
            },
        }


def _close_ephemeral_client(client: Any) -> None:
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


class _TefasPacer:
    """Bounded concurrency semaphore + adaptive inter-request gate for TEFAS upstream requests."""

    def __init__(
        self,
        *,
        max_concurrency: int = 3,
        min_interval_seconds: float = 0.25,
        max_interval_seconds: float = 2.0,
    ) -> None:
        capacity = max(1, int(max_concurrency))
        base = max(0.0, float(min_interval_seconds))
        ceiling = max(base, float(max_interval_seconds))
        self._max_concurrency = capacity
        self._base_interval = base
        self._max_interval = ceiling
        self._current_interval = base
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._state_lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._rate_limit_hits = 0
        self._success_count = 0
        self._active_slots = 0
        self._peak_concurrency = 0

    def acquire_pace_delay(self, *, bypass: bool = False) -> float:
        if bypass:
            return 0.0
        with self._state_lock:
            interval = self._current_interval
            if interval <= 0:
                return 0.0
            now = time.monotonic()
            wait = max(0.0, self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at) + interval
        if wait > 0:
            time.sleep(wait)
        return wait

    @contextlib.contextmanager
    def slot(self, *, bypass_delay: bool = False):
        self._semaphore.acquire()
        with self._state_lock:
            self._active_slots += 1
            if self._active_slots > self._peak_concurrency:
                self._peak_concurrency = self._active_slots
        try:
            self.acquire_pace_delay(bypass=bypass_delay)
            yield
        finally:
            with self._state_lock:
                self._active_slots = max(0, self._active_slots - 1)
            self._semaphore.release()

    def record_rate_limit(self, retry_after_seconds: Optional[float] = None) -> None:
        fs = _fund_service()
        cap = max(0.0, float(getattr(fs, "TEFAS_HTTP_RETRY_AFTER_MAX_SECONDS", 300.0)))
        with self._state_lock:
            self._rate_limit_hits += 1
            self._current_interval = min(
                self._max_interval,
                max(self._base_interval, self._current_interval, 0.05) * 2.0,
            )
            pushback = self._current_interval
            if retry_after_seconds is not None and retry_after_seconds > 0:
                pushback = max(pushback, min(float(retry_after_seconds), cap))
            self._next_allowed_at = max(self._next_allowed_at, time.monotonic() + pushback)

    def record_success(self) -> None:
        with self._state_lock:
            self._success_count += 1
            if self._current_interval > self._base_interval:
                self._current_interval = max(self._base_interval, self._current_interval * 0.85)

    def reset(self) -> None:
        with self._state_lock:
            self._current_interval = self._base_interval
            self._next_allowed_at = 0.0
            self._rate_limit_hits = 0
            self._success_count = 0
            self._peak_concurrency = 0

    def stats(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "max_concurrency": self._max_concurrency,
                "base_interval": self._base_interval,
                "max_interval": self._max_interval,
                "current_interval": self._current_interval,
                "rate_limit_hits": self._rate_limit_hits,
                "success_count": self._success_count,
                "active_slots": self._active_slots,
                "peak_concurrency": self._peak_concurrency,
            }


class TefasClient:
    def __init__(
        self,
        *,
        funds_list_endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        fs = _fund_service()
        self.funds_list_endpoint = (
            funds_list_endpoint if funds_list_endpoint is not None else fs.TEFAS_FUNDS_LIST_ENDPOINT
        )
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else fs.TEFAS_TIMEOUT_SECONDS

    def _headers(self, *, referer: Optional[str] = None) -> Dict[str, str]:
        fs = _fund_service()
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Content-Type": "application/json",
            "Origin": fs.TEFAS_BASE_URL,
            "Referer": referer or f"{fs.TEFAS_BASE_URL}/fon-karsilastirma",
            "User-Agent": fs.FINTABLES_USER_AGENT,
        }

    def _post_json(
        self,
        client: httpx.Client,
        *,
        endpoint: str,
        body: Dict[str, Any],
        context: str,
        referer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST one TEFAS JSON page with shared retry/rate-limit handling."""

        fs = _fund_service()
        attempts = max(1, fs.TEFAS_HTTP_RETRY_ATTEMPTS)
        retry_delay = _seam("_tefas_retry_delay_seconds")
        decode_response = _seam("_decode_tefas_json_response")
        pacer = getattr(fs, "_TEFAS_PACER", None)
        keepalive_retried = False
        attempt = 1
        while attempt <= attempts:
            bypass_delay = float(getattr(fs, "TEFAS_DIRECT_CHUNK_DELAY_SECONDS", 0.0)) <= 0.0
            try:
                if pacer is not None:
                    with pacer.slot(bypass_delay=bypass_delay):
                        response = client.post(
                            endpoint,
                            json=body,
                            headers=self._headers(referer=referer),
                            timeout=self.timeout_seconds,
                        )
                else:
                    response = client.post(
                        endpoint,
                        json=body,
                        headers=self._headers(referer=referer),
                        timeout=self.timeout_seconds,
                    )
            except fs.httpx.RemoteProtocolError as exc:
                if not keepalive_retried:
                    keepalive_retried = True
                    continue
                if attempt >= attempts:
                    raise fs.TefasUpstreamError(f"{context} request failed: {exc}") from exc
                delay = retry_delay(attempt)
                attempt += 1
                if delay > 0:
                    time.sleep(delay)
                continue
            except fs.httpx.HTTPError as exc:
                if attempt >= attempts:
                    raise fs.TefasUpstreamError(f"{context} request failed: {exc}") from exc
                delay = retry_delay(attempt)
                attempt += 1
                if delay > 0:
                    time.sleep(delay)
                continue
            try:
                decoded = decode_response(
                    response.status_code,
                    dict(response.headers),
                    response.content,
                    context=context,
                )
            except fs.TefasRateLimitError as exc:
                if pacer is not None:
                    pacer.record_rate_limit(exc.retry_after_seconds)
                if attempt >= attempts:
                    raise
                delay = retry_delay(attempt, exc.retry_after_seconds)
                attempt += 1
                if delay > 0:
                    time.sleep(delay)
                continue
            if pacer is not None:
                pacer.record_success()
            return decoded
        raise fs.TefasUpstreamError(f"{context} request failed")

    def _fund_list_body(
        self,
        *,
        fund_type: str = "YAT",
        target_date: Optional[date] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        fund_code: Optional[str] = None,
        start_row: int = 1,
        end_row: Optional[int] = None,
    ) -> Dict[str, Any]:
        fs = _fund_service()
        effective_end_row = fs.TEFAS_FUNDS_LIST_PAGE_SIZE if end_row is None else end_row
        effective_start = start_date or target_date
        effective_end = end_date or target_date or effective_start
        if effective_start is None or effective_end is None:
            raise ValueError("target_date or start/end date is required")
        normalized_code = fs.normalize_fund_code(fund_code) or None
        return {
            "fonTipi": fund_type,
            "fonKodu": normalized_code,
            "aramaMetni": normalized_code,
            "fonGrubu": None,
            "basTarih": effective_start.strftime("%Y%m%d"),
            "bitTarih": effective_end.strftime("%Y%m%d"),
            "fonTurKod": None,
            "sfonTurKod": None,
            "basSira": start_row,
            "bitSira": effective_end_row,
            "fonTurAciklama": None,
            "dil": "TR",
            "kurucuKod": None,
        }

    def _post_fund_list(self, *, body: Dict[str, Any], context: str) -> Dict[str, Any]:
        fs = _fund_service()
        client = fs._get_tefas_client(self.timeout_seconds)
        try:
            return self._post_json(
                client,
                endpoint=self.funds_list_endpoint,
                body=body,
                context=context,
            )
        finally:
            if client is not fs._TEFAS_CLIENT:
                _close_ephemeral_client(client)

    @staticmethod
    def _direct_fund_config(fund_type: str) -> Tuple[str, str, Optional[str]]:
        fs = _fund_service()
        base_url = fs.TEFAS_BASE_URL
        configs = {
            "SEC": (
                "YAT",
                f"{base_url}/tr/fon-verileri",
                "YAT",
            ),
            "PEN": (
                "EMK",
                f"{base_url}/tr/fon-verileri",
                "EMK",
            ),
            "ETF": (
                "BYF",
                f"{base_url}/tr/fon-verileri",
                "BYF",
            ),
            "RE": (
                "GYF",
                f"{base_url}/tr/gayrimenkul-fonlari",
                None,
            ),
            "VC": (
                "GSYF",
                f"{base_url}/tr/girisim-sermayesi-fonlari",
                None,
            ),
        }
        normalized = str(fund_type or "").strip().upper()
        if normalized not in configs:
            raise fs.TefasFormatError(f"unsupported direct TEFAS fund type: {fund_type}")
        return configs[normalized]

    @staticmethod
    def _direct_monthly_chunks(start_date: date, end_date: date) -> List[Tuple[date, date]]:
        chunks: List[Tuple[date, date]] = []
        current = date(start_date.year, start_date.month, 1)
        while current <= end_date:
            next_month = (
                date(current.year + 1, 1, 1)
                if current.month == 12
                else date(current.year, current.month + 1, 1)
            )
            chunks.append((max(start_date, current), min(end_date, next_month - timedelta(days=1))))
            current = next_month
        return chunks

    @staticmethod
    def _direct_payload_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("resultList", "data", "Data", "result", "Result", "rows", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return []

    @staticmethod
    def _direct_payload_page_count(payload: Dict[str, Any]) -> Optional[int]:
        for key in ("toplamSayfa", "totalPages", "totalPage", "pageCount"):
            value = payload.get(key)
            if value in (None, ""):
                continue
            try:
                return max(1, int(float(value)))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _direct_payload_total_count(payload: Dict[str, Any]) -> Optional[int]:
        for key in (
            "toplamKayit",
            "toplamSayi",
            "totalCount",
            "totalRecords",
            "totalRows",
            "total_count",
            "count",
        ):
            value = payload.get(key)
            if value in (None, ""):
                continue
            try:
                return max(0, int(float(value)))
            except (TypeError, ValueError):
                continue
        return None

    def _prime_direct_session(
        self,
        client: httpx.Client,
        *,
        portal_url: str,
        fund_url_param: Optional[str],
        start_date: Optional[date],
        end_date: Optional[date],
        force: bool = False,
    ) -> bool:
        fs = _fund_service()
        key = fs._tefas_direct_prime_key(client, portal_url, fund_url_param)
        if not force and fs._is_tefas_direct_primed(key):
            return False
        params: Dict[str, str] = {}
        if fund_url_param:
            params["fundType"] = fund_url_param
        if start_date:
            params["startDate"] = start_date.isoformat()
        if end_date:
            params["endDate"] = end_date.isoformat()
        headers = {
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "User-Agent": fs.FINTABLES_USER_AGENT,
        }
        with fs._get_tefas_direct_prime_key_lock(key):
            if not force and fs._is_tefas_direct_primed(key):
                return False
            try:
                try:
                    response = client.get(
                        portal_url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                except fs.httpx.RemoteProtocolError:
                    response = client.get(
                        portal_url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
            except fs.httpx.HTTPError:
                # The portal GET is only used to establish the normal TEFAS session.
                # The JSON endpoint can still be usable when the HTML page is slow.
                return True
            if getattr(response, "status_code", 200) < 400:
                fs._mark_tefas_direct_primed(key)
            return True

    def _fetch_direct_pages_with_reprime(
        self,
        client: httpx.Client,
        *,
        portal_url: str,
        fund_url_param: Optional[str],
        start_date: Optional[date],
        end_date: Optional[date],
        endpoint: str,
        base_payload: Dict[str, Any],
        referer: str,
        context: str,
        reject_unverified_full_page: bool = False,
    ) -> List[Dict[str, Any]]:
        fs = _fund_service()
        primed_now = self._prime_direct_session(
            client,
            portal_url=portal_url,
            fund_url_param=fund_url_param,
            start_date=start_date,
            end_date=end_date,
        )
        try:
            return self._fetch_direct_pages(
                client,
                endpoint=endpoint,
                base_payload=base_payload,
                referer=referer,
                context=context,
                reject_unverified_full_page=reject_unverified_full_page,
            )
        except fs.TefasUpstreamError as exc:
            message_lower = str(exc).lower()
            if primed_now or ("http 403" not in message_lower and "html/waf" not in message_lower):
                raise
            fs._invalidate_tefas_direct_prime(
                client,
                portal_url=portal_url,
                fund_url_param=fund_url_param,
            )
            self._prime_direct_session(
                client,
                portal_url=portal_url,
                fund_url_param=fund_url_param,
                start_date=start_date,
                end_date=end_date,
                force=True,
            )
            return self._fetch_direct_pages(
                client,
                endpoint=endpoint,
                base_payload=base_payload,
                referer=referer,
                context=context,
                reject_unverified_full_page=reject_unverified_full_page,
            )

    def _fetch_direct_pages(
        self,
        client: httpx.Client,
        *,
        endpoint: str,
        base_payload: Dict[str, Any],
        referer: str,
        context: str,
        reject_unverified_full_page: bool = False,
    ) -> List[Dict[str, Any]]:
        fs = _fund_service()
        page_size = max(1, fs.TEFAS_DIRECT_PAGE_SIZE)
        start_row = 1
        page_number = 0
        total_pages: Optional[int] = None
        reported_total: Optional[int] = None
        all_rows: List[Dict[str, Any]] = []
        full_page_without_terminator = False

        while page_number < 1000:
            payload = dict(base_payload)
            payload["basSira"] = start_row
            payload["bitSira"] = start_row + page_size - 1
            response = self._post_json(
                client,
                endpoint=endpoint,
                body=payload,
                context=f"{context} page {page_number + 1}",
                referer=referer,
            )
            rows = self._direct_payload_rows(response)
            if not rows:
                break
            all_rows.extend(rows)
            page_number += 1
            total_pages = total_pages or self._direct_payload_page_count(response)
            reported_total = reported_total or self._direct_payload_total_count(response)
            if reported_total is not None and len(all_rows) >= reported_total:
                break
            if len(rows) == page_size:
                full_page_without_terminator = True
            if total_pages is not None and page_number >= total_pages and len(rows) < page_size:
                break
            # TEFAS sometimes reports toplamSayfa=1 for a full first page.
            # Probe the next cursor instead of treating that page as complete.
            if len(rows) < page_size:
                full_page_without_terminator = False
                break
            start_row += page_size
        if reported_total is not None and len(all_rows) < reported_total:
            raise fs.TefasUpstreamError(
                f"{context} was incomplete ({len(all_rows)} rows vs reported {reported_total})"
            )
        if total_pages is not None and page_number < total_pages:
            raise fs.TefasUpstreamError(
                f"{context} aborted at page {page_number}/{total_pages}"
            )
        if reject_unverified_full_page and reported_total is None and full_page_without_terminator:
            raise fs.TefasUpstreamError(
                f"{context} ended on a full page without an authoritative total row count"
            )
        return all_rows

    @staticmethod
    def _tag_direct_rows(
        rows: Iterable[Dict[str, Any]],
        *,
        source: str,
        source_url: str,
        fund_type: str,
    ) -> List[Dict[str, Any]]:
        tagged: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            point = dict(row)
            point["source"] = source
            point["source_url"] = source_url
            point.setdefault("fund_type_code", fund_type)
            tagged.append(point)
        return tagged

    @staticmethod
    def _filter_direct_codes(rows: Iterable[Dict[str, Any]], codes: Iterable[str]) -> List[Dict[str, Any]]:
        normalize = _fund_service().normalize_fund_code
        normalized_codes = {
            normalize(code)
            for code in codes
            if normalize(code)
        }
        if not normalized_codes:
            return list(rows)
        materialized = list(rows)
        coded_rows = [
            (
                row,
                normalize(str(row.get("fonKodu") or row.get("fund_code") or row.get("FONKODU") or "")),
            )
            for row in materialized
            if isinstance(row, dict)
        ]
        if not any(code for _row, code in coded_rows):
            return materialized
        return [row for row, code in coded_rows if code in normalized_codes]

    def fetch_funds_direct(
        self,
        *,
        start_date: date,
        end_date: date,
        fund_type: str = "SEC",
        fund_codes: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch TEFAS general-information rows without importing tefasfon."""

        if start_date > end_date:
            return []
        fs = _fund_service()
        normalize = fs.normalize_fund_code
        fund_type_code, portal_url, fund_url_param = self._direct_fund_config(fund_type)
        codes = sorted({normalize(code) for code in list(fund_codes or []) if normalize(code)})
        endpoint = self.funds_list_endpoint
        rows: List[Dict[str, Any]] = []
        chunks = self._direct_monthly_chunks(start_date, end_date)
        client = fs._get_tefas_direct_client(self.timeout_seconds)
        is_shared = fs._is_shared_tefas_direct_client(client)
        try:
            for chunk_start, chunk_end in chunks:
                code = codes[0] if len(codes) == 1 else None
                base_payload = self._fund_list_body(
                    fund_type=fund_type_code,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    fund_code=code,
                    end_row=fs.TEFAS_DIRECT_PAGE_SIZE,
                )
                referer = f"{portal_url}?startDate={chunk_start.isoformat()}&endDate={chunk_end.isoformat()}"
                rows.extend(
                    self._fetch_direct_pages_with_reprime(
                        client,
                        portal_url=portal_url,
                        fund_url_param=fund_url_param,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        endpoint=endpoint,
                        base_payload=base_payload,
                        referer=referer,
                        context=f"direct TEFAS funds {fund_type_code} {chunk_start.isoformat()} {chunk_end.isoformat()}",
                        reject_unverified_full_page=True,
                    )
                )
        finally:
            if not is_shared:
                fs._evict_tefas_direct_prime_entries(client)
                _close_ephemeral_client(client)
        filtered = self._filter_direct_codes(rows, codes)
        return self._tag_direct_rows(
            filtered,
            source=fs.TEFAS_DIRECT_FUNDS_SOURCE,
            source_url=endpoint,
            fund_type=fund_type_code,
        )

    def fetch_returns_direct(
        self,
        *,
        basis: str = "RB",
        fund_type: str = "SEC",
        fund_codes: Optional[Iterable[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch RB/SB/MB return data directly from TEFAS JSON endpoints."""

        fs = _fund_service()
        normalized_basis = str(basis or "RB").strip().upper()
        if normalized_basis not in {"RB", "SB", "MB"}:
            raise fs.TefasFormatError(f"unsupported direct TEFAS returns basis: {basis}")
        if (start_date is None) != (end_date is None):
            raise fs.TefasFormatError("direct TEFAS returns requires both start_date and end_date")
        if start_date and end_date and start_date > end_date:
            return []
        normalize = fs.normalize_fund_code
        fund_type_code, portal_url, fund_url_param = self._direct_fund_config(fund_type)
        codes = sorted({normalize(code) for code in list(fund_codes or []) if normalize(code)})
        endpoint = {
            "RB": fs.TEFAS_RETURNS_RB_ENDPOINT,
            "SB": fs.TEFAS_RETURNS_SB_ENDPOINT,
            "MB": fs.TEFAS_RETURNS_MB_ENDPOINT,
        }[normalized_basis]

        if normalized_basis == "RB" and start_date is None:
            chunks: List[Tuple[Optional[date], Optional[date]]] = [(None, None)]
        elif normalized_basis == "RB":
            chunks = [(start_date, end_date)]  # type: ignore[list-item]
        else:
            chunks = [
                (chunk_start, chunk_end)
                for chunk_start, chunk_end in self._direct_monthly_chunks(start_date, end_date)  # type: ignore[arg-type]
            ]

        rows: List[Dict[str, Any]] = []
        client = fs._get_tefas_direct_client(self.timeout_seconds)
        is_shared = fs._is_shared_tefas_direct_client(client)
        try:
            for chunk_start, chunk_end in chunks:
                prime_start = chunk_start or fs.date.today()
                prime_end = chunk_end or prime_start
                if normalized_basis == "RB":
                    if chunk_start is None:
                        base_payload: Dict[str, Any] = {
                            "dil": "TR",
                            "fonTipi": fund_type_code,
                            "kurucuKodu": None,
                            "sfonTurKod": None,
                            "fonTurAciklama": None,
                            "islem": 1,
                            "fonTurKod": None,
                            "fonGrubu": None,
                            "donemGetiri1a": "1",
                            "donemGetiri3a": "1",
                            "donemGetiri6a": "1",
                            "donemGetiri1y": "1",
                            "donemGetiriyb": "1",
                            "donemGetiri3y": "1",
                            "donemGetiri5y": "1",
                            "basTarih": None,
                            "bitTarih": None,
                            "calismaTipi": 2,
                            "getiriOrani": "1",
                        }
                    else:
                        base_payload = {
                            "dil": "TR",
                            "fonTipi": fund_type_code,
                            "kurucuKodu": None,
                            "sfonTurKod": None,
                            "fonTurAciklama": None,
                            "islem": 1,
                            "fonTurKod": None,
                            "fonGrubu": None,
                            "donemGetiri1a": "0",
                            "donemGetiri3a": "0",
                            "donemGetiri6a": "0",
                            "donemGetiri1y": "0",
                            "donemGetiriyb": "0",
                            "donemGetiri3y": "0",
                            "donemGetiri5y": "0",
                            "basTarih": chunk_start.strftime("%Y%m%d"),
                            "bitTarih": chunk_end.strftime("%Y%m%d"),  # type: ignore[union-attr]
                            "calismaTipi": 1,
                            "getiriOrani": "1",
                        }
                else:
                    base_payload = self._fund_list_body(
                        start_date=chunk_start,
                        end_date=chunk_end,
                        fund_type=fund_type_code,
                        end_row=fs.TEFAS_DIRECT_PAGE_SIZE,
                    )
                referer = f"{portal_url}?startDate={prime_start.isoformat()}&endDate={prime_end.isoformat()}"
                rows.extend(
                    self._fetch_direct_pages_with_reprime(
                        client,
                        portal_url=portal_url,
                        fund_url_param=fund_url_param,
                        start_date=prime_start,
                        end_date=prime_end,
                        endpoint=endpoint,
                        base_payload=base_payload,
                        referer=referer,
                        context=f"direct TEFAS returns {normalized_basis} {fund_type_code}",
                    )
                )
        finally:
            if not is_shared:
                fs._evict_tefas_direct_prime_entries(client)
                _close_ephemeral_client(client)
        filtered = self._filter_direct_codes(rows, codes)
        return self._tag_direct_rows(
            filtered,
            source=fs.TEFAS_DIRECT_RETURNS_SOURCE,
            source_url=endpoint,
            fund_type=fund_type_code,
        )

    def fetch_management_fees_direct(
        self,
        *,
        fund_type: str = "SEC",
        fund_codes: Optional[Iterable[str]] = None,
        as_of: Optional[date] = None,
        lookback_days: int = 21,
    ) -> List[Dict[str, Any]]:
        end_date = as_of or _fund_service().date.today()
        start_date = end_date - timedelta(days=max(7, int(lookback_days)))
        return self.fetch_returns_direct(
            basis="MB",
            fund_type=fund_type,
            fund_codes=fund_codes,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_portfolio_direct(
        self,
        *,
        fund_code: str,
        start_date: date,
        end_date: date,
        fund_type: str = "SEC",
    ) -> List[Dict[str, Any]]:
        """Fetch historical asset-allocation rows directly from TEFAS."""

        fs = _fund_service()
        normalized_code = fs.normalize_fund_code(fund_code)
        if not normalized_code or start_date > end_date:
            return []
        fund_type_code, portal_url, fund_url_param = self._direct_fund_config(fund_type)
        endpoint = fs.TEFAS_PORTFOLIO_ENDPOINT
        rows: List[Dict[str, Any]] = []
        chunks = self._direct_monthly_chunks(start_date, end_date)
        client = fs._get_tefas_direct_client(self.timeout_seconds)
        is_shared = fs._is_shared_tefas_direct_client(client)
        try:
            for chunk_start, chunk_end in chunks:
                base_payload = self._fund_list_body(
                    fund_type=fund_type_code,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    fund_code=normalized_code,
                    end_row=fs.TEFAS_DIRECT_PAGE_SIZE,
                )
                referer = f"{portal_url}?startDate={chunk_start.isoformat()}&endDate={chunk_end.isoformat()}"
                rows.extend(
                    self._fetch_direct_pages_with_reprime(
                        client,
                        portal_url=portal_url,
                        fund_url_param=fund_url_param,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        endpoint=endpoint,
                        base_payload=base_payload,
                        referer=referer,
                        context=f"direct TEFAS portfolio {normalized_code}",
                    )
                )
        finally:
            if not is_shared:
                fs._evict_tefas_direct_prime_entries(client)
                _close_ephemeral_client(client)
        filtered = self._filter_direct_codes(rows, [normalized_code])
        return self._tag_direct_rows(
            filtered,
            source=fs.TEFAS_DIRECT_PORTFOLIO_SOURCE,
            source_url=endpoint,
            fund_type=fund_type_code,
        )

    def fetch_fund_list_snapshot(self, *, target_date: date) -> List[Dict[str, Any]]:
        payload = self._post_fund_list(
            body=self._fund_list_body(target_date=target_date),
            context="TEFAS fund list",
        )
        return _seam("_normalize_tefas_fund_list_payload")(payload, source_url=self.funds_list_endpoint)

    def fetch_fund_range(self, *, fund_code: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        fs = _fund_service()
        normalize = fs.normalize_fund_code
        normalized_code = normalize(fund_code)
        if not normalized_code or start_date > end_date:
            return []
        payload = self._post_fund_list(
            body=self._fund_list_body(
                start_date=start_date,
                end_date=end_date,
                fund_code=normalized_code,
            ),
            context="TEFAS fund range",
        )
        rows = _seam("_normalize_tefas_fund_list_payload")(payload, source_url=self.funds_list_endpoint)
        result = []
        for row in rows:
            code = normalize(str(row.get("fund_code") or row.get("fonKodu") or ""))
            if code == normalized_code:
                normalized = dict(row)
                normalized["source"] = fs.TEFASFON_FUNDS_SOURCE
                result.append(normalized)
        return result

    def fetch_fund_history(
        self,
        *,
        fund_codes: Iterable[str],
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        fs = _fund_service()
        normalize = fs.normalize_fund_code
        codes = {normalize(code) for code in list(fund_codes or []) if normalize(code)}
        if not codes or start_date > end_date:
            return []
        if len(codes) == 1:
            # TEFAS accepts a fund-specific range request, but rejects a
            # range longer than one calendar month. Do not fall back to one
            # request per business day: split the range at calendar-month
            # boundaries and merge the returned daily rows. Consecutive fixed
            # 31-day chunks can cross February/March and silently lose the
            # first part of the next month because TEFAS rejects that request.
            # This is both faster and fixes newly-launched funds whose local
            # cache has no older points yet.
            normalized_code = next(iter(codes))
            chunk_days = max(2, min(31, int(fs.TEFAS_FUND_HISTORY_CHUNK_DAYS)))
            chunk_rows: List[Dict[str, Any]] = []
            chunk_errors: List[Any] = []
            chunks: List[Tuple[date, date]] = []
            current = start_date
            while current <= end_date:
                next_month = (
                    date(current.year + 1, 1, 1)
                    if current.month == 12
                    else date(current.year, current.month + 1, 1)
                )
                calendar_month_end = next_month - timedelta(days=1)
                chunk_end = min(
                    end_date,
                    calendar_month_end,
                    current + timedelta(days=chunk_days - 1),
                )
                chunks.append((current, chunk_end))
                current = chunk_end + timedelta(days=1)
            max_workers = max(1, int(getattr(fs, "TEFAS_HISTORY_MAX_WORKERS", 3)))
            if len(chunks) <= 1 or max_workers <= 1:
                for chunk_start, chunk_end in chunks:
                    try:
                        chunk_rows.extend(
                            self.fetch_fund_range(
                                fund_code=normalized_code,
                                start_date=chunk_start,
                                end_date=chunk_end,
                            )
                        )
                    except fs.TefasRateLimitError:
                        raise
                    except fs.TefasUpstreamError as exc:
                        # A single empty/temporarily bad month must not erase the
                        # valid months around it. If every chunk fails, surface the
                        # last upstream error to the caller for its normal fallback.
                        chunk_errors.append(exc)
            else:
                stop_event = threading.Event()
                worker_count = min(max_workers, len(chunks))

                def _fetch_chunk(chunk_start: date, chunk_end: date) -> Tuple[str, Any]:
                    if stop_event.is_set():
                        return ("skipped", None)
                    try:
                        rows = self.fetch_fund_range(
                            fund_code=normalized_code,
                            start_date=chunk_start,
                            end_date=chunk_end,
                        )
                        return ("ok", rows)
                    except fs.TefasRateLimitError as exc:
                        stop_event.set()
                        return ("rate_limit", exc)
                    except fs.TefasUpstreamError as exc:
                        return ("error", exc)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="tefas-history",
                ) as executor:
                    futures = [
                        executor.submit(_fetch_chunk, chunk_start, chunk_end)
                        for chunk_start, chunk_end in chunks
                    ]
                    outcomes = [future.result() for future in futures]

                for status, payload in outcomes:
                    if status == "rate_limit":
                        raise payload
                for status, payload in outcomes:
                    if status == "ok":
                        chunk_rows.extend(payload)
                    elif status == "error":
                        chunk_errors.append(payload)

            if chunk_rows:
                by_date: Dict[str, Dict[str, Any]] = {}
                for row in chunk_rows:
                    point_date = fs._fund_date(fs._first_present(row, "date", "tarih", "TARIH", "TARIHSTR"))
                    if point_date:
                        by_date[point_date] = row
                if by_date:
                    return [by_date[key] for key in sorted(by_date)]
            if chunk_errors:
                raise chunk_errors[-1]
            return []
        rows: List[Dict[str, Any]] = []
        current = start_date
        while current <= end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue
            snapshot_rows = self.fetch_fund_list_snapshot(target_date=current)
            for row in snapshot_rows:
                code = normalize(str(row.get("fund_code") or row.get("fonKodu") or ""))
                if code in codes:
                    normalized = dict(row)
                    normalized["source"] = fs.TEFASFON_FUNDS_SOURCE
                    rows.append(normalized)
            if fs.FUNDS_WEB_HISTORY_SLEEP_SECONDS > 0:
                time.sleep(fs.FUNDS_WEB_HISTORY_SLEEP_SECONDS)
            current += timedelta(days=1)
        return rows

    def fetch_latest_fund_list_snapshot_result(
        self,
        *,
        as_of: date,
        lookback_days: Optional[int] = None,
    ) -> FundSnapshotFetchResult:
        fs = _fund_service()
        effective_lookback = fs.FUNDS_MAX_LOOKBACK_CALENDAR_DAYS if lookback_days is None else lookback_days
        warnings: List[str] = []
        upstream_errors: List[str] = []
        attempted_dates: List[str] = []
        empty_dates: List[str] = []
        requested = as_of.isoformat()
        window = min(fs.FUNDS_MAX_LOOKBACK_CALENDAR_DAYS, max(1, int(effective_lookback)))
        consecutive_errors = 0
        is_rate_limit = _seam("_is_tefas_rate_limit")
        for offset in range(window):
            target_date = as_of - timedelta(days=offset)
            target_text = target_date.isoformat()
            attempted_dates.append(target_text)
            try:
                rows = self.fetch_fund_list_snapshot(target_date=target_date)
            except fs.TefasUpstreamError as exc:
                message = f"tefas_fund_list failed for {target_text}: {exc}"
                warnings.append(message)
                upstream_errors.append(message)
                consecutive_errors += 1
                if is_rate_limit(exc) or consecutive_errors >= 3:
                    break
                continue
            if not rows:
                consecutive_errors = 0
                empty_dates.append(target_text)
                continue
            resolved_dates = [
                fs._fund_date(fs._first_present(row, "date", "tarih", "TARIH", "TARIHSTR"))
                for row in rows
                if isinstance(row, dict)
            ]
            return fs.FundSnapshotFetchResult(
                rows=list(rows),
                warnings=warnings,
                resolution_status="available" if offset == 0 else "resolved_previous_date",
                requested_as_of=requested,
                resolved_as_of=max((item for item in resolved_dates if item), default=target_text),
                attempted_dates=attempted_dates,
                empty_dates=empty_dates,
                upstream_errors=upstream_errors,
                source=fs.TEFAS_DIRECT_FUNDS_SOURCE,
            )
        return fs.FundSnapshotFetchResult(
            warnings=warnings,
            resolution_status="upstream_unavailable" if upstream_errors else "not_published",
            requested_as_of=requested,
            attempted_dates=attempted_dates,
            empty_dates=empty_dates,
            upstream_errors=upstream_errors,
            source=fs.TEFAS_DIRECT_FUNDS_SOURCE,
        )

    def fetch_latest_fund_list_snapshot(
        self,
        *,
        as_of: date,
        lookback_days: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        result = self.fetch_latest_fund_list_snapshot_result(as_of=as_of, lookback_days=lookback_days)
        return result.rows, result.warnings


__all__ = [
    "_unix_timestamp_for_date",
    "_looks_like_html_challenge",
    "_decode_fintables_json_response",
    "_curl_executable",
    "_fintables_curl_payload",
    "_fintables_curl_cffi_payload",
    "_fintables_proxy_payload",
    "_decode_tefas_json_response",
    "_tefas_retry_after_seconds",
    "_tefas_retry_delay_seconds",
    "_is_tefas_rate_limit",
    "_normalize_tefas_fund_list_payload",
    "_normalize_fintables_udf_history_payload",
    "_normalize_fintables_yield_summary_payload",
    "FintablesClient",
    "fetch_fintables_udf_history",
    "fetch_fintables_yield_summary",
    "_tefasfon_date",
    "_tefasfon_adapter_version",
    "_clean_dataframe_value",
    "_dataframe_records",
    "_tefasfon_rows",
    "_merge_tefasfon_returns",
    "_range_returns_by_code",
    "_merge_tefasfon_range_returns",
    "_coerce_tefas_percentage",
    "_FUND_TAX_RULES",
    "_fund_tax_info",
    "_merge_tefasfon_management_fees",
    "_yield_periods_from_points",
    "_yield_periods_from_return_row",
    "_is_tefasfon_adapter_unavailable",
    "_generate_date_chunks",
    "_TefasPacer",
    "TefasFonClient",
    "TefasClient",
]

_ORIGINAL_EXPORTS.update({name: globals()[name] for name in __all__})
