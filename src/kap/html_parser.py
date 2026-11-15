"""HTML/XBRL row parsing and Turkish text normalization helpers for KAP disclosures."""

from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

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

_LABEL_PATTERN = re.compile(
    r'<div class="gwt-Label multi-language-content content-tr"[^>]*>(.*?)</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
_VALUE_PATTERN = re.compile(
    r'<td class="taxonomy-context-value col-order-class-(\d+)"[^>]*>\s*'
    r"<div>\s*<div[^>]*title=\"([^\"]+)\"",
    flags=re.IGNORECASE | re.DOTALL,
)


class _LegacyKapFinancialRowParser(HTMLParser):
    """Read the pre-2018 KAP taxonomy table layout used in older reports."""

    def __init__(self, *, body_index: int, unit_multiplier: float) -> None:
        super().__init__(convert_charrefs=True)
        self.body_index = body_index
        self.unit_multiplier = unit_multiplier
        self.rows: List[Dict[str, Any]] = []
        self._row: Optional[Dict[str, Any]] = None
        self._row_depth = 0
        self._cell_depth = 0
        self._cell_kind: Optional[str] = None
        self._div_depth = 0
        self._label_depth: Optional[int] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        classes = set(str(attributes.get("class") or "").split())

        if tag == "tr":
            if self._row is None and {"data-input-row", "abstract-row"} & classes:
                self._row = {
                    "label_parts": [],
                    "values": [],
                    "value_title": None,
                    "value_text": [],
                }
                self._row_depth = 1
                self._cell_depth = 0
                self._cell_kind = None
            elif self._row is not None:
                self._row_depth += 1
            return

        if self._row is None:
            return

        if tag == "td":
            if self._cell_depth == 0:
                if "taxonomy-field-title" in classes:
                    self._cell_kind = "label"
                elif "taxonomy-context-value" in classes:
                    self._cell_kind = "value"
                    self._row["value_title"] = None
                    self._row["value_text"] = []
                else:
                    self._cell_kind = "other"
            self._cell_depth += 1
            return

        if tag == "div":
            self._div_depth += 1
            if self._cell_kind == "label" and "content-tr" in classes:
                self._label_depth = self._div_depth
            elif (
                self._cell_kind == "value"
                and self._row.get("value_title") is None
                and attributes.get("title")
            ):
                self._row["value_title"] = attributes["title"]

    def handle_endtag(self, tag: str) -> None:
        if self._row is None:
            return

        if tag == "div" and self._div_depth > 0:
            if self._label_depth == self._div_depth:
                self._label_depth = None
            self._div_depth -= 1
            return

        if tag == "td" and self._cell_depth > 0:
            self._cell_depth -= 1
            if self._cell_depth == 0:
                if self._cell_kind == "label":
                    self._row["label"] = " ".join(self._row["label_parts"]).strip()
                elif self._cell_kind == "value":
                    self._row["values"].append(
                        self._row.get("value_title") or "".join(self._row["value_text"])
                    )
                self._cell_kind = None
            return

        if tag == "tr" and self._row_depth > 0:
            self._row_depth -= 1
            if self._row_depth == 0:
                label = str(self._row.get("label") or "").strip()
                for column_offset, raw_value in enumerate(self._row["values"]):
                    parsed_value = _parse_numeric_token(raw_value)
                    if not label or parsed_value is None:
                        continue
                    self.rows.append(
                        {
                            "body_index": self.body_index,
                            "label": label,
                            "label_norm": _normalize(label),
                            "col_order": 4 + column_offset,
                            "value": float(parsed_value) * self.unit_multiplier,
                        }
                    )
                self._row = None
                self._cell_kind = None
                self._cell_depth = 0
                self._div_depth = 0
                self._label_depth = None

    def handle_data(self, data: str) -> None:
        if self._row is None:
            return
        if self._cell_kind == "label" and self._label_depth is not None:
            self._row["label_parts"].append(data)
        elif self._cell_kind == "value":
            self._row["value_text"].append(data)


def _normalize(text: str) -> str:
    lowered = str(text or "").strip().lower().translate(TR_NORMALIZE_MAP)
    lowered = unicodedata.normalize("NFKD", lowered)
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    return " ".join(lowered.split())


def _clean_html_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(raw or ""))
    text = html.unescape(text)
    return " ".join(text.split())


def _parse_numeric_token(raw: str) -> Optional[float]:
    token = str(raw or "").strip()
    if not token:
        return None

    token = token.replace("\xa0", "").replace(" ", "")
    negative = False
    if token.startswith("(") and token.endswith(")"):
        negative = True
        token = token[1:-1]
    if token.startswith("-"):
        negative = True
        token = token[1:]
    if token.startswith("+"):
        token = token[1:]

    token = re.sub(r"[^0-9\.,]", "", token)
    if not token:
        return None

    # Normalize decimal/thousand separators.
    if "." in token and "," in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        if token.count(",") > 1:
            token = token.replace(",", "")
        else:
            left, right = token.split(",", 1)
            if len(right) == 3 and left:
                token = left + right
            else:
                token = left + "." + right
    elif "." in token:
        if token.count(".") > 1:
            token = token.replace(".", "")
        else:
            left, right = token.split(".", 1)
            if len(right) == 3 and left:
                token = left + right

    try:
        value = float(token)
    except Exception:
        return None
    if negative:
        value = -abs(value)
    return value


def _parse_unit_info(html_block: str) -> Dict[str, Any]:
    raw = ""
    multiplier = 1.0
    currency = ""

    unit_match = re.search(
        r"Sunum Para Birimi</td>\s*<td>(.*?)</td>",
        html_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if unit_match:
        raw = _clean_html_text(unit_match.group(1))

    unit_norm = _normalize(raw)
    if "1.000.000.000" in raw:
        multiplier = 1_000_000_000.0
    elif "1.000.000" in raw:
        multiplier = 1_000_000.0
    elif "1.000" in raw:
        multiplier = 1_000.0
    elif "milyar" in unit_norm:
        multiplier = 1_000_000_000.0
    elif "milyon" in unit_norm:
        multiplier = 1_000_000.0
    elif "bin" in unit_norm:
        multiplier = 1_000.0

    upper_raw = raw.upper()
    for code in ("TRY", "TL", "EUR", "USD"):
        if code in upper_raw:
            currency = "TL" if code == "TRY" else code
            break
    if not currency:
        currency = "TL"

    return {"raw": raw, "multiplier": multiplier, "currency": currency}


def _extract_rows_from_disclosure_body(body_html: str, body_index: int, unit_multiplier: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in _VALUE_PATTERN.finditer(body_html):
        col_order = int(match.group(1))
        parsed_value = _parse_numeric_token(match.group(2))
        if parsed_value is None:
            continue
        raw_value = float(parsed_value)
        pre = body_html[max(0, match.start() - 2800) : match.start()]
        labels = [_clean_html_text(item) for item in _LABEL_PATTERN.findall(pre)]
        if not labels:
            continue
        label = labels[-1].strip()
        if not label:
            continue
        label_norm = _normalize(label)
        if label_norm in {
            "",
            "cari donem",
            "onceki donem",
            "dipnot referansi",
        }:
            continue
        rows.append(
            {
                "body_index": body_index,
                "label": label,
                "label_norm": label_norm,
                "col_order": col_order,
                "value": raw_value * unit_multiplier,
            }
        )
    if not rows:
        legacy_parser = _LegacyKapFinancialRowParser(
            body_index=body_index,
            unit_multiplier=unit_multiplier,
        )
        legacy_parser.feed(body_html)
        rows = legacy_parser.rows
    return rows

