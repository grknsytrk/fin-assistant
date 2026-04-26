#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "logo_probe_report.json"

# Keep this map focused on fallback-needing stocks; expand as needed.
STOCK_DOMAIN_MAP: Dict[str, str] = {
    "AEFES": "anadoluefes.com",
    "AKBNK": "akbank.com",
    "ASELS": "aselsan.com",
    "BIMAS": "bim.com.tr",
    "EREGL": "erdemir.com.tr",
    "GARAN": "garanti.com.tr",
    "ISCTR": "isbank.com.tr",
    "KCHOL": "koc.com.tr",
    "MGROS": "migroskurumsal.com",
    "PGSUS": "flypgs.com",
    "SAHOL": "sabanci.com",
    "SISE": "sisecam.com.tr",
    "TCELL": "turkcell.com.tr",
    "TAVHL": "tavhavalimanlari.com.tr",
    "THYAO": "thy.com",
    "TOASO": "tofas.com.tr",
    "TTKOM": "turktelekom.com.tr",
    "TUPRS": "tupras.com.tr",
    "VAKBN": "vakifbank.com.tr",
    "YKBNK": "yapikredi.com.tr",
    "ZOREN": "zorluenerji.com.tr",
}

TRADINGVIEW_SLUG_MAP: Dict[str, str] = {
    "SP500": "spx",
    "NASDAQ": "nasdaq",
    "DOW": "dow",
    "CAC40": "cac",
    "XU100": "xu100",
    "XU030": "xu030",
    "BRENT": "brent",
    "WTI": "crude-oil",
    "ALTIN": "gold",
    "GUMUS": "silver",
    "DOGALGAZ": "natural-gas",
    "AKBNK": "akbank",
    "AEFES": "anadolu-efes",
    "ASELS": "aselsan",
    "BIMAS": "bim",
    "ENKAI": "enka-insaat",
    "EREGL": "eregli-demir",
    "KCHOL": "koc",
    "PGSUS": "pegasus",
    "TAVHL": "tav-havalimanlari",
    "THYAO": "turkish-airlines",
    "TUPRS": "tupras",
}


def _is_ok_image(status: int, content_type: str, content_len: int, *, min_bytes: int = 1200) -> bool:
    if status != 200:
        return False
    if "image" not in (content_type or "").lower() and "svg" not in (content_type or "").lower():
        return False
    return content_len >= min_bytes


def _safe_get(session: requests.Session, url: str, timeout: float = 10.0) -> Dict[str, Any]:
    try:
        res = session.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        return {
            "ok": True,
            "status": int(res.status_code),
            "content_type": str(res.headers.get("content-type", "")),
            "bytes": len(res.content),
            "text": res.text[:200],
            "json": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "content_type": "",
            "bytes": 0,
            "text": str(exc),
            "json": None,
        }


def _oid_for_symbol(session: requests.Session, symbol: str) -> Optional[str]:
    try:
        payload = session.get(
            f"https://www.kap.org.tr/tr/api/member/filter/{symbol}",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if payload.status_code != 200:
            return None
        rows = payload.json()
    except Exception:
        return None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            oid = str(row.get("mkkMemberOid") or "").strip()
            if oid:
                return oid
    return None


def _probe_symbol(
    session: requests.Session,
    symbol: str,
    logo_dev_token: str,
    include_tradingview: bool,
) -> Dict[str, Any]:
    symbol = symbol.strip().upper()
    row: Dict[str, Any] = {
        "symbol": symbol,
        "mkk_member_oid": None,
        "kap_logo": {"url": None, "status": None, "bytes": 0, "content_type": "", "ok": False},
        "logo_dev": {"url": None, "status": None, "bytes": 0, "content_type": "", "ok": False},
        "tradingview": {"url": None, "status": None, "bytes": 0, "content_type": "", "ok": False},
        "final_source": "monogram",
    }

    oid = _oid_for_symbol(session, symbol)
    row["mkk_member_oid"] = oid

    if oid:
        kap_url = f"https://www.kap.org.tr/tr/api/member/logo/{oid}"
        kap_resp = _safe_get(session, kap_url)
        kap_ok = _is_ok_image(kap_resp["status"] or 0, kap_resp["content_type"], kap_resp["bytes"])
        row["kap_logo"] = {
            "url": kap_url,
            "status": kap_resp["status"],
            "bytes": kap_resp["bytes"],
            "content_type": kap_resp["content_type"],
            "ok": kap_ok,
        }

    domain = STOCK_DOMAIN_MAP.get(symbol)
    if logo_dev_token and domain:
        logo_dev_url = f"https://img.logo.dev/{domain}?token={logo_dev_token}"
        logo_dev_resp = _safe_get(session, logo_dev_url)
        logo_dev_ok = _is_ok_image(logo_dev_resp["status"] or 0, logo_dev_resp["content_type"], logo_dev_resp["bytes"])
        row["logo_dev"] = {
            "url": logo_dev_url,
            "status": logo_dev_resp["status"],
            "bytes": logo_dev_resp["bytes"],
            "content_type": logo_dev_resp["content_type"],
            "ok": logo_dev_ok,
        }

    if include_tradingview:
        slug = TRADINGVIEW_SLUG_MAP.get(symbol, symbol.lower())
        tv_url = f"https://s3-symbol-logo.tradingview.com/{slug}.svg"
        tv_resp = _safe_get(session, tv_url)
        tv_ok = _is_ok_image(tv_resp["status"] or 0, tv_resp["content_type"], tv_resp["bytes"], min_bytes=500)
        row["tradingview"] = {
            "url": tv_url,
            "status": tv_resp["status"],
            "bytes": tv_resp["bytes"],
            "content_type": tv_resp["content_type"],
            "ok": tv_ok,
        }

    if row["kap_logo"]["ok"]:
        row["final_source"] = "kap"
    elif row["logo_dev"]["ok"]:
        row["final_source"] = "logo_dev"
    elif row["tradingview"]["ok"]:
        row["final_source"] = "tradingview"
    else:
        row["final_source"] = "monogram"

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe logo coverage and fallback chain.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols; empty = BIST100")
    parser.add_argument("--token", default=os.getenv("VITE_LOGO_DEV_TOKEN", ""), help="Logo.dev public token")
    parser.add_argument("--include-tradingview", action="store_true", help="Probe TradingView fallback coverage")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    if args.symbols.strip():
        symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    else:
        from app.kap_service import get_bist100_companies

        symbols = get_bist100_companies()

    session = requests.Session()
    rows: List[Dict[str, Any]] = []
    started = time.time()

    for symbol in symbols:
        rows.append(
            _probe_symbol(
                session=session,
                symbol=symbol,
                logo_dev_token=str(args.token or "").strip(),
                include_tradingview=bool(args.include_tradingview),
            )
        )

    total = len(rows)
    kap_ok = sum(1 for row in rows if row["kap_logo"]["ok"])
    logo_dev_ok = sum(1 for row in rows if row["logo_dev"]["ok"])
    tv_ok = sum(1 for row in rows if row["tradingview"]["ok"])
    fallback_monogram = sum(1 for row in rows if row["final_source"] == "monogram")

    kap_empty = [
        row["symbol"]
        for row in rows
        if row["kap_logo"]["url"] and not row["kap_logo"]["ok"]
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol_count": total,
        "elapsed_ms": int((time.time() - started) * 1000),
        "summary": {
            "kap_ok": kap_ok,
            "kap_coverage_pct": round((kap_ok / total) * 100, 2) if total else 0.0,
            "logo_dev_ok": logo_dev_ok,
            "logo_dev_coverage_pct": round((logo_dev_ok / total) * 100, 2) if total else 0.0,
            "tradingview_ok": tv_ok,
            "tradingview_coverage_pct": round((tv_ok / total) * 100, 2) if total else 0.0,
            "monogram_fallback_count": fallback_monogram,
        },
        "kap_empty_or_invalid_symbols": kap_empty,
        "rows": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Symbols: {total}")
    print(f"KAP ok: {kap_ok}/{total}")
    if str(args.token or "").strip():
        print(f"Logo.dev ok: {logo_dev_ok}/{total}")
    else:
        print("Logo.dev token not provided; Logo.dev probe skipped.")
    if args.include_tradingview:
        print(f"TradingView ok: {tv_ok}/{total}")
    else:
        print("TradingView probe skipped.")
    print(f"Monogram fallback: {fallback_monogram}")
    print(f"Report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
