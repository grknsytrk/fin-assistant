"""Resolve foreign security identifiers without treating an ISIN as a ticker."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable

import httpx

LOGGER = logging.getLogger(__name__)
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
        with httpx.Client(timeout=6, headers=headers) as client:
            # Five jobs also fit older unauthenticated Mapping API limits.
            for start in range(0, len(missing), 5):
                batch = missing[start:start + 5]
                try:
                    response = client.post(
                        "https://api.openfigi.com/v3/mapping",
                        json=[{"idType": "ID_ISIN", "idValue": isin, "marketSecDes": "Equity"} for isin in batch],
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
