from __future__ import annotations

import re
from typing import Dict, List, Optional

TR_NORMALIZE_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

Q1_PATTERNS = [
    re.compile(r"\b1\.?\s*ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bbirinci\s+ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bilk\s+ceyrek(?:te|de|da|in|i|e|a)?\b"),
]
Q2_PATTERNS = [
    re.compile(r"\b2\.?\s*ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bikinci\s+ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bilk\s+yariyil(?:da|de|in|i|a|e)?\b"),
    re.compile(r"\byariyil(?:da|de|in|i|a|e)?\b"),
]
Q3_PATTERNS = [
    re.compile(r"\b3\.?\s*ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bucuncu\s+ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bdokuz\s+aylik(?:ta|te|in|i|a|e)?\b"),
    re.compile(r"\b9\s+aylik(?:ta|te|in|i|a|e)?\b"),
    re.compile(r"\bilk\s+9\s+ay\b"),
    re.compile(r"\bilk\s+dokuz\s+ay\b"),
]
Q4_PATTERNS = [
    re.compile(r"\b4\.?\s*ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bdorduncu\s+ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\bson\s+ceyrek(?:te|de|da|in|i|e|a)?\b"),
    re.compile(r"\b12\s+aylik(?:ta|te|in|i|a|e)?\b"),
    re.compile(r"\btam\s+yil(?:da|de|in|i|a|e)?\b"),
]

NUMERIC_KEYWORDS = {
    "kac",
    "tutar",
    "milyar",
    "milyon",
    "ciro",
    "hasilat",
    "satis",
    "nakit",
    "nakit akisi",
    "capex",
    "yatirim",
    "tl",
    "yuzde",
    "marj",
    "oran",
    "ne kadar",
    "%",
}
TREND_KEYWORDS = {
    "artti",
    "artmis",
    "artis",
    "azaldi",
    "azalmis",
    "azalis",
    "degisim",
    "degisti",
    "seyir",
    "trend",
    "yukseldi",
    "geriledi",
}
QUALITATIVE_KEYWORDS = {
    "risk",
    "strateji",
    "beklenti",
    "gorunum",
    "degerlendirme",
    "degerlendirme",
    "politika",
    "yaklasim",
}
KPI_KEYWORDS = {
    "magaza",
    "kpi",
    "online",
    "eticaret",
    "e-ticaret",
    "calisan",
    "personel",
    "adet",
    "sayi",
}


def _normalize(text: str) -> str:
    lowered = text.lower().translate(TR_NORMALIZE_MAP)
    lowered = re.sub(r"[^\w%\s]", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _find_matched_phrases(normalized_question: str, patterns: List[re.Pattern]) -> List[str]:
    matched: List[str] = []
    for pattern in patterns:
        for match in pattern.finditer(normalized_question):
            phrase = match.group(0).strip()
            if phrase and phrase not in matched:
                matched.append(phrase)
    return matched


def infer_query_type(question: str) -> str:
    normalized = _normalize(question)

    if any(keyword in normalized for keyword in TREND_KEYWORDS):
        return "trend"
    if any(keyword in normalized for keyword in QUALITATIVE_KEYWORDS):
        return "qualitative"
    if any(keyword in normalized for keyword in KPI_KEYWORDS):
        return "kpi"
    if any(keyword in normalized for keyword in NUMERIC_KEYWORDS):
        return "numeric"
    return "other"


def parse_query(question: str) -> Dict[str, object]:
    normalized = _normalize(question)
    year_match = YEAR_PATTERN.search(normalized)
    year: Optional[str] = year_match.group(1) if year_match else None

    q1_hits = _find_matched_phrases(normalized, Q1_PATTERNS)
    q2_hits = _find_matched_phrases(normalized, Q2_PATTERNS)
    q3_hits = _find_matched_phrases(normalized, Q3_PATTERNS)
    q4_hits = _find_matched_phrases(normalized, Q4_PATTERNS)

    quarter: Optional[str] = None
    if q4_hits:
        quarter = "Q4"
    elif q3_hits:
        quarter = "Q3"
    elif q2_hits:
        quarter = "Q2"
    elif q1_hits:
        quarter = "Q1"

    query_type = infer_query_type(question)
    signals = {
        "normalized_question": normalized,
        "year": year,
        "matched_q1_phrases": q1_hits,
        "matched_q2_phrases": q2_hits,
        "matched_q3_phrases": q3_hits,
        "matched_q4_phrases": q4_hits,
        "query_type": query_type,
    }
    return {
        "quarter": quarter,
        "signals": signals,
    }
