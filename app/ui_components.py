from __future__ import annotations

from typing import Any, Dict, Optional


def trust_level(confidence: Optional[float], verify_status: Optional[str]) -> Dict[str, Any]:
    status = str(verify_status or "").strip().upper()
    level = "Medium"

    if status == "FAIL":
        level = "Low"
    elif status == "WARN":
        level = "Medium"
    elif status == "PASS":
        level = "High"

    if confidence is not None:
        try:
            conf = float(confidence)
        except Exception:
            conf = None
        if conf is not None:
            if conf < 0.45:
                level = "Low"
            elif conf < 0.75 and level == "High":
                level = "Medium"

    if level == "High":
        return {
            "level": level,
            "label": "Yuksek",
            "icon": "[+]",
            "bg": "#d1fae5",
            "fg": "#065f46",
            "hint": "",
        }
    if level == "Medium":
        return {
            "level": level,
            "label": "Orta",
            "icon": "[~]",
            "bg": "#fef3c7",
            "fg": "#92400e",
            "hint": "",
        }
    return {
        "level": level,
        "label": "Dusuk",
        "icon": "[!]",
        "bg": "#fee2e2",
        "fg": "#991b1b",
        "hint": "Bu deger tabloda kaymis olabilir; kaniti acip dogrulayin.",
    }


def trust_badge_html(confidence: Optional[float], verify_status: Optional[str]) -> str:
    trust = trust_level(confidence=confidence, verify_status=verify_status)
    return (
        "<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{trust['bg']};color:{trust['fg']};font-size:12px;font-weight:600'>"
        f"{trust['icon']} {trust['label']} guven"
        "</span>"
    )
