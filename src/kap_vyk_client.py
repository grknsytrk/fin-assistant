"""Thin REST client for the official KAP VYK (Veri Yayin Kanali) endpoints.

Auth modes:
    * Dev/test gateway: Basic auth generally works directly.
    * Prod gateway: KAP docs require `generateToken` before the data methods
        (apiKey query param; optional Basic fallback when api_secret exists).
    The client supports both and can auto-select by environment.

Design notes:
  * Stdlib only (urllib) to avoid new runtime dependencies.
  * Long TTL in-memory caches on hot endpoints (members, disclosureDetail)
    so repeated flow refreshes do not stampede the gateway.
  * Every upstream failure is swallowed and returns an empty result so the
    /market/flow handler can degrade gracefully to the local cache.
"""
from __future__ import annotations

import base64
import json
import time
from threading import Lock
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from src.config import KapConfig

# region caches (process-local)
_DETAIL_CACHE: Dict[str, Dict[str, Any]] = {}
_DETAIL_CACHE_TTL_SECONDS = 6 * 3600  # published disclosures are immutable in practice

_MEMBERS_CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}
_MEMBERS_CACHE_TTL_SECONDS = 6 * 3600

_LAST_INDEX_CACHE: Dict[str, Any] = {"value": None, "ts": 0.0}
_LAST_INDEX_CACHE_TTL_SECONDS = 15

_TOKEN_CACHE: Dict[str, Any] = {"value": None, "expires_at": 0.0}
_TOKEN_CACHE_LOCK = Lock()
_TOKEN_FALLBACK_TTL_SECONDS = 23 * 3600
_TOKEN_GRACE_SECONDS = 60
_TOKEN_KEYS = {
    "token",
    "access_token",
    "accessToken",
    "bearerToken",
    "bearer_token",
    "jwt",
    "jwtToken",
    "id_token",
}
# endregion


def _basic_header(api_key: str, api_secret: str) -> str:
    token = f"{api_key}:{api_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(token).decode("ascii")


def _non_prod_base_url(base_url: str) -> bool:
    host = (urllib.parse.urlparse(base_url or "").hostname or "").lower()
    return any(flag in host for flag in ("dev", "test", "apitest"))


def _resolve_auth_mode(cfg: "KapConfig") -> str:
    mode = str(getattr(cfg, "vyk_auth_mode", "") or "").strip().lower()
    if mode in {"basic", "token"}:
        return mode
    return "basic" if _non_prod_base_url(getattr(cfg, "vyk_base_url", "")) else "token"


def _auth_schemes(cfg: "KapConfig") -> List[str]:
    configured = str(getattr(cfg, "vyk_auth_mode", "") or "").strip().lower()
    if configured == "basic":
        return ["basic"]
    if configured == "token":
        return ["bearer"]
    # `auto`: non-prod ortamlarda mevcut davranisi koru; prod'da token once gelsin.
    if _non_prod_base_url(getattr(cfg, "vyk_base_url", "")):
        return ["basic", "bearer"]
    return ["bearer", "basic"]


def _request_headers(cfg: "KapConfig", authorization: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Accept-Language": "tr",
        "User-Agent": cfg.user_agent,
    }
    if authorization:
        headers["Authorization"] = authorization
    return headers


def _request_timeout(cfg: "KapConfig") -> float:
    return float(getattr(cfg, "timeout_seconds", 10.0) or 10.0)


def _parse_payload(raw: str) -> Any:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def _open_payload(url: str, *, cfg: "KapConfig", headers: Dict[str, str]) -> tuple[Any, Dict[str, str]]:
    request = urllib.request.Request(url, method="GET", headers=headers)
    with urllib.request.urlopen(request, timeout=_request_timeout(cfg)) as response:
        raw = response.read().decode("utf-8", errors="replace")
        response_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    return _parse_payload(raw), response_headers


def _looks_like_token(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token or " " in token or len(token) < 12:
        return None
    return token


def _extract_token_from_headers(headers: Dict[str, str]) -> Optional[str]:
    for key in ("authorization", "x-auth-token", "token", "x-access-token"):
        candidate = _looks_like_token(headers.get(key))
        if candidate:
            return candidate
    return None


def _extract_token_from_payload(payload: Any) -> Optional[str]:
    direct = _looks_like_token(payload)
    if direct:
        return direct
    if isinstance(payload, dict):
        for key in _TOKEN_KEYS:
            if key in payload:
                candidate = _extract_token_from_payload(payload.get(key))
                if candidate:
                    return candidate
        for key, value in payload.items():
            if "token" in str(key).lower():
                candidate = _extract_token_from_payload(value)
                if candidate:
                    return candidate
        for value in payload.values():
            candidate = _extract_token_from_payload(value)
            if candidate:
                return candidate
    if isinstance(payload, list):
        for value in payload:
            candidate = _extract_token_from_payload(value)
            if candidate:
                return candidate
    return None


def _decode_token_expiry(token: str) -> Optional[float]:
    parts = str(token or "").split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception:
        return None
    try:
        exp = float(payload.get("exp"))
    except (TypeError, ValueError):
        return None
    return exp if exp > time.time() else None


def _clear_token_cache() -> None:
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE["value"] = None
        _TOKEN_CACHE["expires_at"] = 0.0


def _gateway_root_from_vyk_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(str(base_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _with_api_key_query(url: str, api_key: str) -> str:
    key = str(api_key or "").strip()
    if not key:
        return url
    parsed = urllib.parse.urlparse(str(url or "").strip())
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if any(str(name).lower() == "apikey" for name, _ in pairs):
        return url
    pairs.append(("apiKey", key))
    query = urllib.parse.urlencode(pairs)
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment)
    )


def _resolve_token_url(cfg: "KapConfig") -> str:
    explicit = str(getattr(cfg, "vyk_token_url", "") or "").strip()
    if explicit:
        return _with_api_key_query(explicit, getattr(cfg, "api_key", ""))

    gateway_root = _gateway_root_from_vyk_base_url(getattr(cfg, "vyk_base_url", ""))
    if gateway_root:
        return _with_api_key_query(f"{gateway_root}/auth/generateToken", getattr(cfg, "api_key", ""))

    fallback = f"{str(getattr(cfg, 'vyk_base_url', '')).rstrip('/')}/generateToken"
    return _with_api_key_query(fallback, getattr(cfg, "api_key", ""))


def _generate_bearer_token(cfg: "KapConfig") -> str:
    token_authorization = None
    if str(getattr(cfg, "api_secret", "") or "").strip():
        token_authorization = _basic_header(cfg.api_key, cfg.api_secret)
    payload, headers = _open_payload(
        _resolve_token_url(cfg),
        cfg=cfg,
        headers=_request_headers(cfg, token_authorization),
    )
    token = _extract_token_from_headers(headers) or _extract_token_from_payload(payload)
    if not token:
        raise ValueError("generateToken yaniti icinde token bulunamadi")
    expires_at = _decode_token_expiry(token) or (time.time() + _TOKEN_FALLBACK_TTL_SECONDS)
    _TOKEN_CACHE["value"] = token
    _TOKEN_CACHE["expires_at"] = expires_at
    return token


def _get_bearer_token(cfg: "KapConfig", *, force_refresh: bool = False) -> str:
    with _TOKEN_CACHE_LOCK:
        cached = str(_TOKEN_CACHE.get("value") or "").strip()
        expires_at = float(_TOKEN_CACHE.get("expires_at") or 0.0)
        if not force_refresh and cached and expires_at > (time.time() + _TOKEN_GRACE_SECONDS):
            return cached
        return _generate_bearer_token(cfg)


def _error_payload(exc: urllib.error.HTTPError) -> Any:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    return _parse_payload(raw)


def _payload_requests_new_token(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("code") or "").strip().upper() == "ER006":
        return True
    try:
        text = json.dumps(payload, ensure_ascii=False).lower()
    except Exception:
        return False
    return (
        "token has expired" in text
        or "token gecerlilik suresi bitmistir" in text
        or "token geçerlilik süresi bitmiştir" in text
    )


def _http_error_requests_new_token(exc: urllib.error.HTTPError) -> bool:
    if int(getattr(exc, "code", 0) or 0) in {401, 403}:
        return True
    return _payload_requests_new_token(_error_payload(exc))


def is_enabled(cfg: "KapConfig") -> bool:
    """Return True iff VYK credentials + base URL are fully configured."""
    return bool(
        getattr(cfg, "api_key", "")
        and getattr(cfg, "vyk_base_url", "")
    )


def _request_json(url: str, *, cfg: "KapConfig") -> Any:
    last_error: Optional[BaseException] = None
    for scheme in _auth_schemes(cfg):
        force_refresh = False
        max_attempts = 2 if scheme == "bearer" else 1
        for _ in range(max_attempts):
            try:
                authorization = (
                    f"Bearer {_get_bearer_token(cfg, force_refresh=force_refresh)}"
                    if scheme == "bearer"
                    else _basic_header(cfg.api_key, cfg.api_secret)
                )
                payload, _ = _open_payload(
                    url,
                    cfg=cfg,
                    headers=_request_headers(cfg, authorization),
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if scheme == "bearer" and _http_error_requests_new_token(exc):
                    _clear_token_cache()
                    force_refresh = True
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                break

            if scheme == "bearer" and _payload_requests_new_token(payload):
                _clear_token_cache()
                force_refresh = True
                continue
            return payload

    if last_error is not None:
        raise last_error
    return None


def get_last_disclosure_index(cfg: "KapConfig") -> Optional[int]:
    """Return the most recently published disclosureIndex (short TTL)."""
    if not is_enabled(cfg):
        return None
    now = time.time()
    cached = _LAST_INDEX_CACHE
    if cached.get("value") is not None and (now - float(cached.get("ts") or 0.0)) < _LAST_INDEX_CACHE_TTL_SECONDS:
        return int(cached["value"])
    try:
        payload = _request_json(f"{cfg.vyk_base_url}/lastDisclosureIndex", cfg=cfg)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    try:
        value = int(str((payload or {}).get("lastDisclosureIndex") or "").strip())
    except (TypeError, ValueError):
        return None
    cached["value"] = value
    cached["ts"] = now
    return value


def list_disclosures_batch(
    cfg: "KapConfig",
    *,
    start_index: int,
    disclosure_types: Optional[str] = None,
    disclosure_class: Optional[str] = None,
    company_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch up to 50 disclosures beginning at `start_index`.

    Mirrors the `/disclosures` service on the VYK gateway. Each call returns
    at most 50 rows so callers that need a wider window must paginate.
    """
    if not is_enabled(cfg):
        return []
    params: Dict[str, str] = {"disclosureIndex": str(int(start_index))}
    if disclosure_types:
        params["disclosureTypes"] = disclosure_types
    if disclosure_class:
        params["disclosureClass"] = disclosure_class
    if company_id:
        params["companyId"] = str(company_id)
    query = urllib.parse.urlencode(params)
    url = f"{cfg.vyk_base_url}/disclosures?{query}"
    try:
        payload = _request_json(url, cfg=cfg)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def get_disclosure_detail(
    cfg: "KapConfig",
    disclosure_index: Any,
    *,
    file_type: str = "data",
) -> Optional[Dict[str, Any]]:
    """Fetch `/disclosureDetail/{id}` with a long-TTL in-memory cache."""
    if not is_enabled(cfg):
        return None
    idx = str(disclosure_index or "").strip()
    if not idx:
        return None

    cache_key = f"{idx}:{file_type}"
    now = time.time()
    cached = _DETAIL_CACHE.get(cache_key)
    if cached and (now - float(cached.get("_ts") or 0.0)) < _DETAIL_CACHE_TTL_SECONDS:
        data = cached.get("data")
        return dict(data) if isinstance(data, dict) else None

    params = urllib.parse.urlencode({"fileType": file_type})
    url = f"{cfg.vyk_base_url}/disclosureDetail/{urllib.parse.quote(idx)}?{params}"
    try:
        payload = _request_json(url, cfg=cfg)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    _DETAIL_CACHE[cache_key] = {"_ts": now, "data": payload}
    return dict(payload)


def list_members(cfg: "KapConfig") -> List[Dict[str, Any]]:
    """Return the full KAP member list (long-TTL cached)."""
    if not is_enabled(cfg):
        return []
    now = time.time()
    cache = _MEMBERS_CACHE
    data = cache.get("data")
    if isinstance(data, list) and (now - float(cache.get("ts") or 0.0)) < _MEMBERS_CACHE_TTL_SECONDS:
        return [dict(row) for row in data if isinstance(row, dict)]
    try:
        payload = _request_json(f"{cfg.vyk_base_url}/members", cfg=cfg)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    normalized = [dict(row) for row in payload if isinstance(row, dict)]
    cache["data"] = normalized
    cache["ts"] = now
    return [dict(row) for row in normalized]


def build_company_lookup(cfg: "KapConfig") -> Dict[str, Dict[str, Any]]:
    """Return a `companyId -> member` dict derived from `list_members`."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in list_members(cfg):
        cid = str(row.get("id") or "").strip()
        if cid:
            lookup[cid] = row
    return lookup


def reset_caches_for_tests() -> None:
    """Test helper: clear all module-level caches."""
    _DETAIL_CACHE.clear()
    _MEMBERS_CACHE["data"] = None
    _MEMBERS_CACHE["ts"] = 0.0
    _LAST_INDEX_CACHE["value"] = None
    _LAST_INDEX_CACHE["ts"] = 0.0
    _clear_token_cache()
