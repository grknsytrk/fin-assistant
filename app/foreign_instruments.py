"""Resolve foreign security identifiers without treating an ISIN as a ticker."""

from __future__ import annotations

import atexit
import logging
import os
import re
import threading
from typing import Any, Callable, Optional

import httpx

LOGGER = logging.getLogger(__name__)
_ORIGINAL_HTTPX_CLIENT = httpx.Client
_OPENFIGI_CLIENT_LOCK = threading.Lock()
_OPENFIGI_CLIENT: Optional[httpx.Client] = None
OPENFIGI_TIMEOUT_SECONDS = float(os.getenv("RAGFIN_OPENFIGI_TIMEOUT_SECONDS", "6"))
_ISIN_PATTERN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_EXCHANGE_SUFFIXES = {"US": "", "CN": ".TO", "SW": ".SW", "LN": ".L", "FP": ".PA", "GY": ".DE", "NA": ".AS"}
_ISIN_HOME_EXCHANGES = {"US": "US", "CA": "CN", "CH": "SW", "GB": "LN", "FR": "FP", "DE": "GY", "NL": "NA"}


def is_isin(value: Any) -> bool:
    return bool(_ISIN_PATTERN.fullmatch(str(value or "").strip().upper()))


def _identity_from_mapping(mapping: dict[str, Any], isin: str) -> dict[str, Any]:
    candidates = [
        row for row in mapping.get("data", [])
        if isinstance(row, dict)
        and row.get("marketSector") == "Equity"
        and str(row.get("exchCode") or "") in _EXCHANGE_SUFFIXES
        and str(row.get("ticker") or "").strip()
        and str(row.get("name") or "").strip()
    ]
    home = _ISIN_HOME_EXCHANGES.get(isin[:2])
    preferred = [row for row in candidates if row.get("exchCode") == home]
    if preferred:
        candidates = preferred
    identities = {}
    for row in candidates:
        exchange = row["exchCode"]
        ticker = str(row["ticker"]).strip().upper().replace("/", "-")
        if exchange == "US":
            ticker = ticker.replace(".", "-")
        if is_isin(ticker) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", ticker):
            continue
        symbol = ticker + _EXCHANGE_SUFFIXES[exchange]
        identities[symbol] = {
            "provider_symbol": symbol,
            "name": str(row["name"]).strip(),
            "source": "openfigi",
        }
    # Ambiguous listings must not silently attach another security's quotes.
    return next(iter(identities.values())) if len(identities) == 1 else {}


def _get_openfigi_client(
    *,
    timeout: Optional[float] = None,
    headers: Optional[dict[str, str]] = None,
) -> tuple[httpx.Client, bool]:
    """Return a pooled persistent OpenFIGI client, or a fresh ephemeral mock in tests."""
    effective_timeout = OPENFIGI_TIMEOUT_SECONDS if timeout is None else timeout
    if httpx.Client is not _ORIGINAL_HTTPX_CLIENT:
        return httpx.Client(timeout=effective_timeout, headers=dict(headers or {})), True
    global _OPENFIGI_CLIENT
    with _OPENFIGI_CLIENT_LOCK:
        if _OPENFIGI_CLIENT is None or _OPENFIGI_CLIENT.is_closed:
            _OPENFIGI_CLIENT = httpx.Client(
                timeout=effective_timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0),
            )
        return _OPENFIGI_CLIENT, False


def close_openfigi_client() -> None:
    """Close the shared OpenFIGI HTTP connection pool cleanly."""
    global _OPENFIGI_CLIENT
    with _OPENFIGI_CLIENT_LOCK:
        client = _OPENFIGI_CLIENT
        _OPENFIGI_CLIENT = None
    if client is not None:
        try:
            client.close()
        except Exception:
            LOGGER.debug("error closing OpenFIGI HTTP client", exc_info=True)


atexit.register(close_openfigi_client)


def resolve_foreign_holdings(
    positions: list[dict[str, Any]],
    *,
    cache_get: Callable[[str], Any],
    cache_set: Callable[..., Any],
) -> list[dict[str, Any]]:
    rows = [dict(position) for position in positions]
    pending: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("asset_type") not in {"foreign_equity", "foreign_fund"}:
            continue
        code = str(row.get("asset_code") or "").strip().upper()
        isin = str(row.get("isin") or (code if is_isin(code) else "")).strip().upper()
        if not is_isin(isin):
            continue
        row["isin"] = isin
        symbol = str(row.get("provider_symbol") or row.get("logo_symbol") or code).strip().upper()
        name = str(row.get("asset_name") or "").strip()
        if symbol and not is_isin(symbol) and not is_isin(name):
            continue
        if is_isin(symbol):
            row["provider_symbol"] = None
            if is_isin(row.get("logo_symbol")):
                row["logo_symbol"] = None
        pending.setdefault(isin, []).append(row)

    missing = []
    for isin in pending:
        cached = cache_get(f"api:funds:foreign-identity:{isin}:v1")
        if isinstance(cached, dict):
            identities[isin] = cached
        else:
            missing.append(isin)
    if missing:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("RAGFIN_OPENFIGI_API_KEY", "").strip()
        if api_key:
            headers["X-OPENFIGI-APIKEY"] = api_key
        client, is_ephemeral = _get_openfigi_client(timeout=OPENFIGI_TIMEOUT_SECONDS, headers=headers)
        try:
            # Five jobs also fit older unauthenticated Mapping API limits.
            for start in range(0, len(missing), 5):
                batch = missing[start:start + 5]
                payload = [{"idType": "ID_ISIN", "idValue": isin, "marketSecDes": "Equity"} for isin in batch]
                try:
                    try:
                        response = client.post(
                            "https://api.openfigi.com/v3/mapping",
                            json=payload,
                            headers=headers,
                            timeout=OPENFIGI_TIMEOUT_SECONDS,
                        )
                    except httpx.RemoteProtocolError:
                        response = client.post(
                            "https://api.openfigi.com/v3/mapping",
                            json=payload,
                            headers=headers,
                            timeout=OPENFIGI_TIMEOUT_SECONDS,
                        )
                    response.raise_for_status()
                    mappings = response.json()
                    if not isinstance(mappings, list) or len(mappings) != len(batch):
                        raise ValueError("OpenFIGI returned an incomplete mapping response")
                    for isin, mapping in zip(batch, mappings):
                        identity = _identity_from_mapping(mapping, isin) if isinstance(mapping, dict) else {}
                        identities[isin] = identity
                        cache_set(
                            f"api:funds:foreign-identity:{isin}:v1", identity,
                            ttl_seconds=30 * 24 * 60 * 60 if identity else 60 * 60,
                        )
                except (httpx.HTTPError, ValueError):
                    LOGGER.warning("Foreign holding identifier mapping unavailable")
                    for isin in missing[start:]:
                        cache_set(f"api:funds:foreign-identity:{isin}:v1", {}, ttl_seconds=60)
                    break
        finally:
            if is_ephemeral and hasattr(client, "close"):
                client.close()

    for isin, targets in pending.items():
        identity = identities.get(isin) or {}
        if not identity.get("provider_symbol"):
            continue
        for row in targets:
            row["provider_symbol"] = identity["provider_symbol"]
            row["provider_symbol_status"] = "resolved"
            row["logo_symbol"] = identity["provider_symbol"]
            row["asset_name"] = identity["name"]
            row["identity_source"] = identity["source"]
    return rows
