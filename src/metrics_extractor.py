from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.autoverify import auto_verify_metric
from src.config import load_config
from src.validators import validate_metric_value

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.retrieve import RetrievedChunk, RetrieverV3

QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]

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

COMPARISON_KEYWORDS = {
    "trend",
    "artis",
    "azalis",
    "karsilastir",
    "karsilastirma",
    "karsilastirmasi",
    "degisim",
    "ceyrekler",
    "q1",
    "q2",
    "q3",
    "q1 q2 q3",
}
LOSS_CONTEXT_WORDS = (
    "zarar",
    "zarari",
    "net donem zarari",
    "donem zarari",
)
MONETARY_METRICS = {
    "net_kar",
    "brut_kar",
    "favok",
    "satis_gelirleri",
    "faaliyet_nakit_akisi",
    "capex",
    "serbest_nakit_akisi",
}
STORE_CONTEXT_KEYWORDS = (
    "magaza",
    "sube",
    "store",
    "outlet",
)
STORE_TOTAL_CONTEXT_KEYWORDS = (
    "toplam magaza",
    "toplam sube",
    "magazasi bulunmaktadir",
    "magaza sayilari ozet tablo",
)
STORE_LFL_CONTEXT_KEYWORDS = (
    "ayni magaza",
    "like for like",
    "magaza basi",
    "trafik",
    "sepet hacmi",
    "tekabul etmektedir",
)

try:
    _CONFIG = load_config(Path("config.yaml"))
    METRIC_DICTIONARY_PATH = _CONFIG.extraction.metrics_dictionary_file
    TOP_CANDIDATES_DEFAULT = int(_CONFIG.extraction.top_candidates)
    TREND_DEVIATION_THRESHOLD_PCT = float(_CONFIG.extraction.trend_deviation_threshold_pct)
    EXPECTED_RANGES = dict(_CONFIG.extraction.expected_ranges)
except Exception:  # pragma: no cover
    METRIC_DICTIONARY_PATH = Path("data/dictionaries/metrics_tr.yaml")
    TOP_CANDIDATES_DEFAULT = 5
    TREND_DEVIATION_THRESHOLD_PCT = 300.0
    EXPECTED_RANGES: Dict[str, Dict[str, float]] = {}

TABLE_NUMBER_PATTERN = re.compile(r"\(?[\-]?\d[\d\.,]*\)?")
TABLE_PERIOD_PATTERN = re.compile(
    r"(?:20\d{2}\s*[qQ][1-4]|[qQ][1-4]\s*20\d{2}|(?:[1-4]\.?[cq]|12\.?a(?:y)?|9\.?a(?:y)?|6\.?a|1\.?yy|1y|fy|yillik)\s*20\d{2}|20\d{2}\s*(?:[1-4]\.?[cq]|12\.?a(?:y)?|9\.?a(?:y)?|6\.?a|1\.?yy|1y|fy|yillik)|[qQ][1-4]|[1-4]\.?[cq]|12\.?a(?:y)?|9\.?a(?:y)?|6\.?a|1\.?yy|1y|fy|yillik|ilk\s+yariyil|dokuz\s+aylik|dorduncu\s+ceyrek)",
    re.IGNORECASE,
)

METRIC_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "net_kar": {
        "label": "Net kar",
        "unit": "TL",
        "query": "net kar",
        "keyword_hints": (
            "net kar",
            "ana ortakliga ait net donem kari",
            "ana ortaklik payi",
            "ana ortaklara dusen kar",
            "zarar",
            "net donem zarari",
        ),
        "forbidden_hints": ("marj", "orani"),
        "patterns": [
            re.compile(r"net\s+kar\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"ana\s+ortakliga\s+ait\s+net\s+donem\s+kari\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"ana\s+ortakl[ıi]k\s+pay(?:lar[ıi])?\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(r"ana\s+ortakl[ıi]k\s+pay(?:lar[ıi])?\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"net\s+donem\s+zarar[ıi]\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"donem\s+net\s+kar[ıi]\s+veya\s+zarar[ıi]\s+(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"donem\s+kar[ıi]\s+veya\s+zarar[ıi]\s+(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}[^\n]{0,70}net\s+kar\s+milyon\s+tl",
                re.IGNORECASE,
            ),
            re.compile(
                r"net\s+kar\s+milyon\s+tl[^\d]{0,40}(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"net\s+donem\s+zarar[ıi]\s+milyon\s+tl[^\d]{0,40}(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"donem\s+net\s+kar[ıi](?:\s+\(?zarar[ıi]\)?)?\s*(\(?[\-]?\d[\d\.,]*\)?)\s*(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"donem\s+kar[ıi](?:\s+\(?zarar[ıi]\)?)?\s*(\(?[\-]?\d[\d\.,]*\)?)\s*(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(?:1c|2c|3c|1q|2q|3q|1yy|1y|6a|9a)\s*20\d{2}\s+(?:1c|2c|3c|1q|2q|3q|1yy|1y|6a|9a)\s*20\d{2}[^\n]{0,120}net\s+kar\s*\(mtl\)",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("gelir tablosu", "ozet konsolide gelir", "ceyreksel gelir tablosu"),
    },
    "favok": {
        "label": "FAVOK",
        "unit": "TL",
        "query": "favok",
        "keyword_hints": ("favok", "fvaok"),
        "forbidden_hints": ("marj", "orani"),
        "patterns": [
            re.compile(r"(?:fvaok|favok)\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"(?:fvaok|favok)[^\d]{0,30}(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}[^\n]{0,70}(?:konsolide\s+)?(?:fvaok|favok)\s+milyon\s+tl",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:konsolide\s+)?(?:fvaok|favok)\s+milyon\s+tl[^\d]{0,40}(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("gelir tablosu", "ozet konsolide gelir"),
    },
    "faaliyet_nakit_akisi": {
        "label": "Faaliyet nakit akisi",
        "unit": "TL",
        "query": "faaliyetlerden elde edilen nakit akisi",
        "keyword_hints": (
            "faaliyetlerden elde edilen nakit",
            "isletme faaliyetlerinden saglanan net nakit",
            "faaliyetlerden elde edilen nakit akisi",
            "isletme faaliyetlerinden elde edilen nakit",
            "operating cash flow",
            "faaliyet nakit akisi",
        ),
        "forbidden_hints": ("yatirim faaliyetlerinde kullanilan", "finansman faaliyetlerinden"),
        "patterns": [
            re.compile(
                r"(?:faal[iı]yetlerden|i[sş]letme\s+faal[iı]yetlerinden)\s+(?:elde\s+edilen|sa[gğ]lanan)\s+(?:net\s+)?nakit(?:\s+ak[ıi][sş][ıi])?[^\d\-]{0,24}(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:operating\s+cash\s+flow)[^\d\-]{0,20}(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}[^\n]{0,120}(?:faal[iı]yetlerden|i[sş]letme\s+faal[iı]yetlerinden)[^\n]{0,80}nakit",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("nakit akis tablosu", "nakit akisi", "cash flow"),
    },
    "capex": {
        "label": "Yatirim harcamalari (CAPEX)",
        "unit": "TL",
        "query": "yatirim harcamalari",
        "keyword_hints": (
            "yatirim harcamalari",
            "capex",
            "yatirim faaliyetlerinde kullanilan net nakit",
            "yat har satis",
        ),
        "forbidden_hints": ("finansman faaliyetlerinden",),
        "patterns": [
            re.compile(
                r"(?:yat[ıi]r[ıi]m\s+harcam(?:a|alar[ıi]|as[ıi])|capex)[^\d\-]{0,24}(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"yat[ıi]r[ıi]m\s+faal[iı]yetlerinde\s+kullan[ıi]lan\s+net\s+nakit[^\d\-]{0,24}(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:yat\.?\s*har\.?\s*/\s*sat[ıi][sş])[^\d\-]{0,24}(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("nakit akis tablosu", "nakit akisi", "yatirim harcamalari", "cash flow"),
    },
    "serbest_nakit_akisi": {
        "label": "Serbest nakit akisi",
        "unit": "TL",
        "query": "serbest nakit akisi",
        "keyword_hints": ("serbest nakit akisi", "free cash flow", "fcf"),
        "forbidden_hints": (),
        "patterns": [
            re.compile(
                r"(?:serbest\s+nakit\s+ak[ıi][sş][ıi]|free\s+cash\s+flow|\bfcf\b)[^\d\-]{0,24}(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("nakit akis tablosu", "nakit akisi", "cash flow"),
    },
    "brut_kar": {
        "label": "Brut kar",
        "unit": "TL",
        "query": "brut kar",
        "keyword_hints": ("brut kar", "gross profit"),
        "forbidden_hints": ("marj", "orani"),
        "patterns": [
            re.compile(r"brut\s+kar\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"gross\s+profit\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"brut\s+kar[^0-9]{0,80}(\(?[\-]?\d[\d\.,]*\)?)\s*(?:milyon|milyar)?\s*tl",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}[^\n]{0,80}brut\s+kar",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("gelir tablosu", "ozet konsolide gelir"),
    },
    "brut_kar_marji": {
        "label": "Brut kar marji",
        "unit": "%",
        "query": "brut kar marji",
        "keyword_hints": ("brut kar marji",),
        "forbidden_hints": (),
        "patterns": [
            re.compile(r"brut\s+kar\s+marji\s*%?\s*(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"brut\s+kar\s+marji[^%]{0,120}%\s*(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
        ],
        "section_hints": ("gelir tablosu", "ozet konsolide gelir"),
    },
    "net_kar_marji": {
        "label": "Net kar marji",
        "unit": "%",
        "query": "net kar marji",
        "keyword_hints": ("net kar marji",),
        "forbidden_hints": (),
        "patterns": [
            re.compile(r"net\s+kar\s+marji\s*%?\s*(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"net\s+kar\s+marji[^%]{0,120}%\s*(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"%\s*[\-]?\d[\d\.,]*\s*%\s*[\-]?\d[\d\.,]*\s*%\s*(\(?[\-]?\d[\d\.,]*\)?)\s*(?:3c|9a)\s*20\d{2}[^\n]{0,120}net\s+satislar[^\n]{0,120}(?:fvaok|favok)\s+marji\s+net\s+kar\s+marji",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("gelir tablosu", "ozet konsolide gelir"),
    },
    "favok_marji": {
        "label": "FAVOK marji",
        "unit": "%",
        "query": "favok marji",
        "keyword_hints": ("favok marji", "fvaok marji"),
        "forbidden_hints": (),
        "patterns": [
            re.compile(r"(?:fvaok|favok)\s+marji\s*%?\s*(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"(?:fvaok|favok)\s+marji[^%]{0,120}%\s*(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"%\s*[\-]?\d[\d\.,]*\s*%\s*(\(?[\-]?\d[\d\.,]*\)?)\s*%\s*[\-]?\d[\d\.,]*\s*(?:3c|9a)\s*20\d{2}[^\n]{0,120}net\s+satislar[^\n]{0,120}(?:fvaok|favok)\s+marji\s+net\s+kar\s+marji",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("gelir tablosu", "ozet konsolide gelir"),
    },
    "satis_gelirleri": {
        "label": "Satis gelirleri",
        "unit": "TL",
        "query": "satislar",
        "keyword_hints": ("satis", "satis gelir", "hasilat", "ciro"),
        "forbidden_hints": ("marj", "orani"),
        "patterns": [
            re.compile(r"sat[ıi]slar?\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"sat[ıi]s\s+gelir(?:leri)?\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"hasilat\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"ciro[^\d]{0,20}(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"toplam\s+faaliyet\s+gelir(?:leri|i)\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"toplam\s+ciro\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(r"konsolide\s+ciro\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"(?:net\s+)?sat[ıi]slar?[^0-9]{0,80}(\(?[\-]?\d[\d\.,]*\)?)\s*(?:milyon|milyar)?\s*tl",
                re.IGNORECASE,
            ),
            re.compile(r"ciro[^0-9]{0,80}(\(?[\-]?\d[\d\.,]*\)?)\s*(?:milyon|milyar)?\s*tl", re.IGNORECASE),
            re.compile(
                r"(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}\s+(?:9a|1yy|3c|2c|1c)\s+20\d{2}[^\n]{0,80}net\s+sat[ıi]slar",
                re.IGNORECASE,
            ),
            re.compile(
                r"net\s+sat[ıi]slar[^\d]{0,50}(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("gelir tablosu", "ozet konsolide gelir"),
    },
    "magaza_sayisi": {
        "label": "Magaza sayisi",
        "unit": "count",
        "query": "toplam magaza sayisi",
        "keyword_hints": ("magaza", "toplam"),
        "forbidden_hints": (),
        "patterns": [
            re.compile(
                r"toplam\s+(?:magaza|sube)(?:\s+sayis[ıi])?\s*(\(?[\-]?\d[\d\.,]*\)?)",
                re.IGNORECASE,
            ),
            re.compile(r"(\(?[\-]?\d[\d\.,]*\)?)\s*(?:\([^\)]*\)\s*)?magazas[ıi]\s+bulunmaktad[ıi]r", re.IGNORECASE),
            re.compile(r"(?:\()?\s*([\-]?\d[\d\.,]*)\s*(?:\))?\s+magaza(?:ya)?\s+tekabul", re.IGNORECASE),
            re.compile(r"magaza\s+sayis[ıi]\s+(\(?[\-]?\d[\d\.,]*\)?)", re.IGNORECASE),
            re.compile(
                r"(\(?[\-]?\d[\d\.,]*\)?)\s+(\(?[\-]?\d[\d\.,]*\)?)\s+(?:1c|2c|3c|1q|2q|3q|1yy|1y|6a|9a)\s*20\d{2}\s+(?:1c|2c|3c|1q|2q|3q|1yy|1y|6a|9a)\s*20\d{2}[^\n]{0,80}magaza\s+sayis[ıi]",
                re.IGNORECASE,
            ),
        ],
        "section_hints": ("magaza sayilari", "magaza", "hakkinda"),
    },
}


def _load_metrics_dictionary(path: Path = METRIC_DICTIONARY_PATH) -> Dict[str, Dict[str, Any]]:
    if yaml is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return {}
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    cleaned: Dict[str, Dict[str, Any]] = {}
    for metric_name, row in metrics.items():
        if not isinstance(row, dict):
            continue
        cleaned[str(metric_name)] = row
    return cleaned


def _normalize_dict_text(text: str) -> str:
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    tr_normalized = without_marks.translate(TR_NORMALIZE_MAP)
    tr_normalized = re.sub(r"[^\w%\s\.,:\-\(\)]", " ", tr_normalized, flags=re.UNICODE)
    tr_normalized = re.sub(r"\s+", " ", tr_normalized).strip()
    return tr_normalized


def _tuple_unique(items: Sequence[str]) -> Tuple[str, ...]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        norm = _normalize_dict_text(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return tuple(ordered)


def _apply_dictionary_overrides(
    definitions: Dict[str, Dict[str, Any]],
    dictionary_rows: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for metric, base in definitions.items():
        row = dict(base)
        dictionary_row = dictionary_rows.get(metric, {})

        if dictionary_row.get("label"):
            row["label"] = str(dictionary_row["label"])
        if dictionary_row.get("unit"):
            row["unit"] = str(dictionary_row["unit"])
        if dictionary_row.get("query"):
            row["query"] = str(dictionary_row["query"])

        base_synonyms = [str(item) for item in row.get("keyword_hints", ())]
        if row.get("label"):
            base_synonyms.append(str(row["label"]))
        if row.get("query"):
            base_synonyms.append(str(row["query"]))
        dict_synonyms = [
            str(item)
            for item in dictionary_row.get("synonyms", [])
            if isinstance(item, str)
        ]
        row["synonyms"] = _tuple_unique(base_synonyms + dict_synonyms)
        row["keyword_hints"] = row["synonyms"]

        base_sections = [str(item) for item in row.get("section_hints", ())]
        dict_sections = [
            str(item)
            for item in dictionary_row.get("section_hints", [])
            if isinstance(item, str)
        ]
        row["section_hints"] = _tuple_unique(base_sections + dict_sections)
        merged[metric] = row
    return merged


METRIC_DEFINITIONS = _apply_dictionary_overrides(
    METRIC_DEFINITIONS,
    _load_metrics_dictionary(),
)


def normalize_for_match(text: str) -> str:
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    tr_normalized = without_marks.translate(TR_NORMALIZE_MAP)
    tr_normalized = re.sub(r"[^\w%\s\.,:\-\(\)]", " ", tr_normalized, flags=re.UNICODE)
    tr_normalized = re.sub(r"\s+", " ", tr_normalized).strip()
    return tr_normalized


def quarter_label(raw_quarter: str) -> str:
    candidate = str(raw_quarter).upper()
    if candidate.endswith("Q1") or candidate == "Q1":
        return "Q1"
    if candidate.endswith("Q2") or candidate == "Q2":
        return "Q2"
    if candidate.endswith("Q3") or candidate == "Q3":
        return "Q3"
    if candidate.endswith("Q4") or candidate == "Q4":
        return "Q4"
    return candidate


def parse_tr_number(raw: str) -> Optional[float]:
    payload = raw.strip().replace(" ", "")
    if not payload:
        return None

    parenthesis_negative = payload.startswith("(") and payload.endswith(")")
    if parenthesis_negative:
        payload = payload[1:-1]
    sign = -1.0 if payload.startswith("-") else 1.0
    if parenthesis_negative:
        sign = -1.0
    payload = payload.lstrip("+-")
    if not payload:
        return None

    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", payload):
        normalized = payload.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", payload):
        normalized = payload.replace(",", "")
    elif payload.count(",") == 1 and payload.count(".") == 0:
        left, right = payload.split(",", 1)
        normalized = f"{left}.{right}" if len(right) <= 2 else f"{left}{right}"
    elif payload.count(".") > 1 and payload.count(",") == 0:
        normalized = payload.replace(".", "")
    elif "," in payload and "." in payload:
        if payload.rfind(",") > payload.rfind("."):
            normalized = payload.replace(".", "").replace(",", ".")
        else:
            normalized = payload.replace(",", "")
    else:
        normalized = payload.replace(",", ".")

    try:
        return sign * float(normalized)
    except ValueError:
        return None


def _apply_negative_signal(
    metric: str,
    parsed_value: float,
    raw_value: str,
    context_norm: str,
    before_match_norm: str,
) -> float:
    if parsed_value < 0:
        return parsed_value
    raw_compact = raw_value.strip()
    if raw_compact.startswith("(") and raw_compact.endswith(")"):
        return -abs(parsed_value)

    if metric not in MONETARY_METRICS:
        return parsed_value
    combined = f"{before_match_norm} {context_norm}".strip()
    # Labels like "Dönem net karı veya zararı" are neutral headings and do not
    # imply that the numeric value is negative.
    if re.search(r"kar[ıi]\s*(?:veya|/)\s*\(?zarar[ıi]\)?", combined):
        return parsed_value
    if any(signal in combined for signal in LOSS_CONTEXT_WORDS):
        return -abs(parsed_value)
    return parsed_value


def format_metric_value(value: Optional[float], unit: str, currency: str = "TL") -> str:
    if value is None:
        return "Bulunamadi"
    if unit == "count":
        return f"{value:,.0f}".replace(",", ".") + " adet"
    if unit == "%":
        return "%" + f"{value:.2f}".replace(".", ",")
    if unit == "TL":
        currency_label = (currency or "TL").upper()
        abs_value = abs(value)
        if abs_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}".replace(".", ",") + f" milyar {currency_label}"
        if abs_value >= 1_000_000:
            return f"{value / 1_000_000:.2f}".replace(".", ",") + f" milyon {currency_label}"
        return f"{value:,.0f}".replace(",", ".") + f" {currency_label}"
    return str(value)


def metric_display_name(metric: str) -> str:
    config = METRIC_DEFINITIONS.get(metric, {})
    return str(config.get("label", metric))


def metric_unit(metric: str) -> str:
    config = METRIC_DEFINITIONS.get(metric, {})
    return str(config.get("unit", "TL"))


def _expected_range_for_metric(metric: str) -> Optional[Dict[str, float]]:
    bounds = EXPECTED_RANGES.get(metric)
    if not isinstance(bounds, dict):
        return None
    if "min" not in bounds and "max" not in bounds:
        return None
    result: Dict[str, float] = {}
    if "min" in bounds:
        result["min"] = float(bounds["min"])
    if "max" in bounds:
        result["max"] = float(bounds["max"])
    return result if result else None


def _looks_like_table_chunk(text_norm: str, block_type: str) -> bool:
    if block_type == "table_like":
        return True
    lines = [line.strip() for line in text_norm.splitlines() if line.strip()]
    if len(lines) < 2:
        # Some PDF extractors flatten table rows into a single long line.
        # Detect those chunks heuristically via dense numeric + period tokens.
        number_count = len(TABLE_NUMBER_PATTERN.findall(text_norm))
        period_count = len(TABLE_PERIOD_PATTERN.findall(text_norm))
        return number_count >= 8 and period_count >= 2
    numeric_rich = sum(1 for line in lines if len(TABLE_NUMBER_PATTERN.findall(line)) >= 2)
    period_rich = sum(1 for line in lines if len(TABLE_PERIOD_PATTERN.findall(line)) >= 2)
    spaced = sum(1 for line in lines if len(re.findall(r"\s{2,}", line)) >= 1)
    return numeric_rich >= 2 or (numeric_rich >= 1 and period_rich >= 1) or (numeric_rich >= 2 and spaced >= 1)


def _period_token_to_quarter(token: str) -> Optional[str]:
    payload = re.sub(r"[^a-z0-9]+", "", normalize_for_match(token))
    if "q1" in payload or "1c" in payload or "1q" in payload:
        return "Q1"
    if (
        "q2" in payload
        or "2c" in payload
        or "2q" in payload
        or "6a" in payload
        or "1yy" in payload
        or "1y" in payload
        or "ilkyariyil" in payload
    ):
        return "Q2"
    if "q3" in payload or "3c" in payload or "3q" in payload or "9a" in payload or "9ay" in payload or "dokuzaylik" in payload:
        return "Q3"
    if (
        "q4" in payload
        or "4c" in payload
        or "4q" in payload
        or "12a" in payload
        or "12ay" in payload
        or "yillik" in payload
        or "fy" in payload
        or "dorduncuceyrek" in payload
    ):
        return "Q4"
    return None


def _extract_period_columns(line_norm: str) -> List[Dict[str, Optional[str]]]:
    token_iter = list(
        re.finditer(
            r"\b(?:20\d{2}|q[1-4](?:\s*20\d{2}|\d{2})?|[1-4]\.?[cq](?:\s*20\d{2}|\d{2})?|12\.?a(?:y)?(?:\s*20\d{2}|\d{2})?|9\.?a(?:y)?(?:\s*20\d{2}|\d{2})?|6\.?a(?:\s*20\d{2}|\d{2})?|1\.?yy(?:\s*20\d{2}|\d{2})?|1y(?:\s*20\d{2}|\d{2})?|fy(?:\s*20\d{2}|\d{2})?|yillik|ilk\s+yariyil|dokuz\s+aylik|dorduncu\s+ceyrek)\b",
            line_norm,
            flags=re.IGNORECASE,
        )
    )
    if token_iter:
        periods: List[Dict[str, Optional[str]]] = []
        last_year: Optional[str] = None
        for match in token_iter:
            token_norm = match.group(0).strip()
            year_value: Optional[str] = None
            if re.fullmatch(r"20\d{2}", token_norm):
                year_value = token_norm
                last_year = token_norm
                periods.append(
                    {
                        "token": token_norm,
                        "quarter": None,
                        "year": year_value,
                    }
                )
                continue

            year_match = re.search(r"20\d{2}", token_norm)
            if year_match:
                year_value = year_match.group(0)
            else:
                compact = token_norm.replace(" ", "")
                yy_match = re.search(r"(\d{2})$", compact)
                if yy_match:
                    year_value = f"20{yy_match.group(1)}"
            if year_value:
                last_year = year_value
            elif last_year:
                year_value = last_year

            periods.append(
                {
                    "token": token_norm,
                    "quarter": _period_token_to_quarter(token_norm),
                    "year": year_value,
                }
            )
        return periods

    tokens = TABLE_PERIOD_PATTERN.findall(line_norm)
    periods: List[Dict[str, Optional[str]]] = []
    for token in tokens:
        token_norm = str(token)
        year_match = re.search(r"20\d{2}", token_norm)
        quarter = _period_token_to_quarter(token_norm)
        periods.append(
            {
                "token": token_norm,
                "quarter": quarter,
                "year": year_match.group(0) if year_match else None,
            }
        )
    return periods


def _choose_aligned_number_index(
    metric: str,
    quarter: str,
    row_numbers: Sequence[str],
    period_columns: Sequence[Dict[str, Optional[str]]],
) -> int:
    if not row_numbers:
        return 0
    limit = min(len(row_numbers), len(period_columns))
    if limit > 0:
        # First preference: explicit quarter alignment.
        quarter_matches: List[Tuple[int, int]] = []
        for idx in range(limit):
            if period_columns[idx].get("quarter") == quarter:
                year_raw = period_columns[idx].get("year")
                year_num = int(year_raw) if year_raw and str(year_raw).isdigit() else -1
                quarter_matches.append((idx, year_num))
        if quarter_matches:
            quarter_matches.sort(key=lambda item: (item[1], item[0]), reverse=True)
            return quarter_matches[0][0]
        # Second preference: latest year column.
        year_candidates: List[Tuple[int, int]] = []
        for idx in range(limit):
            year_raw = period_columns[idx].get("year")
            if year_raw and str(year_raw).isdigit():
                year_candidates.append((idx, int(year_raw)))
        if year_candidates:
            year_candidates.sort(key=lambda item: item[1], reverse=True)
            return year_candidates[0][0]

    # Conservative fallback: margins often appear as last percentage column.
    if metric in {"net_kar_marji", "favok_marji", "brut_kar_marji"}:
        return len(row_numbers) - 1
    return len(row_numbers) - 1


def _structured_table_candidates(
    *,
    metric: str,
    quarter: str,
    chunk: "RetrievedChunk",
    text_norm: str,
    section_norm: str,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not _looks_like_table_chunk(text_norm=text_norm, block_type=str(getattr(chunk, "block_type", "text"))):
        return candidates

    metric_synonyms = [str(item) for item in METRIC_DEFINITIONS.get(metric, {}).get("synonyms", ())]
    metric_synonyms = [normalize_for_match(item) for item in metric_synonyms if item]
    if not metric_synonyms:
        metric_synonyms = [normalize_for_match(metric)]

    lines = [line.strip() for line in text_norm.splitlines() if line.strip()]
    raw_lines = [line.strip() for line in str(getattr(chunk, "text", "")).splitlines() if line.strip()]
    if not lines:
        return candidates

    for idx, line in enumerate(lines):
        if not any(syn in line for syn in metric_synonyms):
            continue
        number_matches = list(TABLE_NUMBER_PATTERN.finditer(line))
        if len(number_matches) < 2:
            continue

        forbidden_hints = [normalize_for_match(str(item)) for item in METRIC_DEFINITIONS.get(metric, {}).get("forbidden_hints", ())]
        if forbidden_hints and any(hint and hint in line for hint in forbidden_hints):
            continue

        header_periods: List[Dict[str, Optional[str]]] = []
        header_line_norm = ""
        header_line_index: Optional[int] = None
        for back in range(max(0, idx - 15), idx + 1):
            periods = _extract_period_columns(lines[back])
            if len(periods) >= 2:
                if (not header_periods) or (len(periods) > len(header_periods)) or (
                    len(periods) == len(header_periods) and (header_line_index is None or back > header_line_index)
                ):
                    header_periods = periods
                    header_line_norm = lines[back]
                    header_line_index = back

        number_tokens = [match.group(0) for match in number_matches]
        aligned_idx = _choose_aligned_number_index(
            metric=metric,
            quarter=quarter,
            row_numbers=number_tokens,
            period_columns=header_periods,
        )
        aligned_idx = max(0, min(aligned_idx, len(number_matches) - 1))
        match = number_matches[aligned_idx]
        raw_value = match.group(0).strip()
        parsed = parse_tr_number(raw_value)
        if parsed is None:
            continue

        header_context = lines[idx - 1] if idx > 0 else ""
        raw_line = raw_lines[idx] if idx < len(raw_lines) else ""
        raw_header_context = raw_lines[idx - 1] if idx > 0 and (idx - 1) < len(raw_lines) else ""
        raw_period_header = (
            raw_lines[header_line_index] if header_line_index is not None and header_line_index < len(raw_lines) else ""
        )
        raw_context = f"{raw_period_header} {raw_header_context} {raw_line}".strip()
        context_norm = f"{header_line_norm} {header_context} {line}".strip()
        if forbidden_hints and any(hint and hint in context_norm for hint in forbidden_hints):
            continue
        before_match_norm = line[: match.start()]
        unit = str(METRIC_DEFINITIONS.get(metric, {}).get("unit", "TL"))
        if unit == "TL" and "%" in context_norm and not any(token in context_norm for token in ("tl", "milyon", "milyar", "mn", "mlr")):
            continue
        parsed = _apply_negative_signal(
            metric=metric,
            parsed_value=float(parsed),
            raw_value=raw_value,
            context_norm=context_norm,
            before_match_norm=before_match_norm,
        )
        multiplier = _unit_multiplier(
            metric=metric,
            context_norm=context_norm,
            text_norm=text_norm,
            local_context_norm=line[max(0, match.start() - 14) : min(len(line), match.end() + 24)],
            raw_context=raw_context,
            raw_text=str(getattr(chunk, "text", "")),
        )
        currency = _detect_currency(
            metric=metric,
            context_norm=context_norm,
            text_norm=text_norm,
            local_context_norm=line[max(0, match.start() - 14) : min(len(line), match.end() + 24)],
            raw_context=raw_context,
            raw_text=str(getattr(chunk, "text", "")),
        )
        multiplier = _implicit_table_multiplier(
            metric=metric,
            unit=unit,
            current_multiplier=multiplier,
            parsed_value=float(parsed),
            raw_value=raw_value,
            block_type="table_like",
            section_norm=section_norm,
            text_norm=text_norm,
        )
        scaled_value = float(parsed) * multiplier
        candidates.append(
            {
                "raw_value": raw_value,
                "scaled_value": scaled_value,
                "context_norm": context_norm,
                "before_match_norm": before_match_norm,
                "pattern_index": -(1000 + idx),
                "line_index": idx,
                "source": "table_reconstruction",
                "multiplier": multiplier,
                "unit": unit,
                "currency": currency,
            }
        )

    return candidates


def is_comparison_query(question: str) -> bool:
    normalized = normalize_for_match(question)
    return any(keyword in normalized for keyword in COMPARISON_KEYWORDS)


def infer_metric_from_question(question: str) -> Optional[str]:
    normalized = normalize_for_match(question)

    # Margin metrics first to prevent collisions like "net kar marji" -> "net_kar".
    priority = [
        "net_kar_marji",
        "favok_marji",
        "brut_kar_marji",
        "serbest_nakit_akisi",
        "faaliyet_nakit_akisi",
        "capex",
        "brut_kar",
        "net_kar",
        "favok",
        "satis_gelirleri",
        "magaza_sayisi",
    ]
    for metric in priority:
        metric_cfg = METRIC_DEFINITIONS.get(metric, {})
        synonyms = metric_cfg.get("synonyms", ())
        for synonym in synonyms:
            if synonym and synonym in normalized:
                return metric

    # Conservative fallback with legacy heuristics.
    if re.search(r"\bnet\s*kar\s*marj", normalized):
        return "net_kar_marji"
    if re.search(r"\b(?:fvaok|favok)\s*marj", normalized):
        return "favok_marji"
    if re.search(r"\bbrut\s*kar\s*marj", normalized):
        return "brut_kar_marji"
    if re.search(r"\b(serbest\s*nakit|free\s*cash\s*flow|\bfcf\b)", normalized):
        return "serbest_nakit_akisi"
    if re.search(r"\b((isletme|faaliyet)\s*faaliyetlerinden.*nakit|faaliyetlerden.*nakit|operating\s*cash\s*flow)", normalized):
        return "faaliyet_nakit_akisi"
    if re.search(r"\b(capex|yatirim\s*harcama|yat\.?\s*har\.?)", normalized):
        return "capex"
    if re.search(r"\bbrut\s*kar\b", normalized):
        return "brut_kar"
    if re.search(r"\b(net\s*donem\s*zarar|zarar)\b", normalized):
        return "net_kar"
    if re.search(r"\bnet\s*kar\b", normalized):
        return "net_kar"
    if re.search(r"\b(?:fvaok|favok)\b", normalized):
        return "favok"
    if re.search(r"\b(satis|hasilat|ciro)\b", normalized):
        return "satis_gelirleri"
    if re.search(r"\bmagaza\b", normalized):
        return "magaza_sayisi"
    return None


def build_metric_query(metric: str, quarter: str, question: str) -> str:
    quarter_map = {
        "Q1": "birinci ceyrek",
        "Q2": "ikinci ceyrek",
        "Q3": "ucuncu ceyrek",
        "Q4": "dorduncu ceyrek",
    }
    quarter_phrase = quarter_map.get(quarter, quarter)
    metric_phrase = str(METRIC_DEFINITIONS.get(metric, {}).get("query", metric))
    synonym_hints_raw = [
        str(item)
        for item in METRIC_DEFINITIONS.get(metric, {}).get("synonyms", ())
        if isinstance(item, str)
    ]
    section_hints_raw = [
        str(item)
        for item in METRIC_DEFINITIONS.get(metric, {}).get("section_hints", ())
        if isinstance(item, str)
    ]
    synonym_hints: List[str] = []
    seen = set()
    for hint in synonym_hints_raw + section_hints_raw:
        hint_norm = normalize_for_match(hint).strip()
        if not hint_norm or hint_norm in seen:
            continue
        if hint_norm == normalize_for_match(metric_phrase):
            continue
        if len(hint_norm) < 4:
            continue
        seen.add(hint_norm)
        synonym_hints.append(hint_norm)
        if len(synonym_hints) >= 6:
            break
    hint_payload = " ".join(synonym_hints)
    return f"2025 {quarter_phrase} {metric_phrase} {hint_payload} {question}".strip()


def _detect_currency(
    metric: str,
    context_norm: str,
    text_norm: str,
    local_context_norm: Optional[str] = None,
    raw_context: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> str:
    if metric not in MONETARY_METRICS:
        return "TL"

    raw_scope = f"{raw_context or ''} {raw_text or ''}".lower()
    norm_scope = f"{local_context_norm or ''} {context_norm or ''} {text_norm or ''}".lower()

    # Strong signals with symbols first.
    if any(token in raw_scope for token in ("€", "eur", "euro", "avro")):
        return "EUR"
    if any(token in raw_scope for token in ("usd", "dolar", "$")):
        return "USD"
    if "£" in raw_scope or "gbp" in raw_scope or "sterlin" in raw_scope:
        return "GBP"
    if any(token in raw_scope for token in ("tl", "try", "turk lira", "turk lirasi", "lira")):
        return "TL"

    if re.search(r"\b(eur|euro|avro)\b", norm_scope):
        return "EUR"
    if re.search(r"\b(usd|dolar)\b", norm_scope):
        return "USD"
    if re.search(r"\b(gbp|sterlin)\b", norm_scope):
        return "GBP"
    if re.search(r"\b(tl|try|lira)\b", norm_scope):
        return "TL"
    return "TL"


def _unit_multiplier(
    metric: str,
    context_norm: str,
    text_norm: str,
    local_context_norm: Optional[str] = None,
    raw_context: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> float:
    unit = str(METRIC_DEFINITIONS.get(metric, {}).get("unit", "TL"))
    if unit == "%" or metric == "magaza_sayisi":
        return 1.0

    raw_local = (raw_context or "").lower()
    raw_all = f"{raw_local} {raw_text or ''}".lower()
    if re.search(r"(?:€\s*m|m\s*€|\beur\s*m\b|\bm\s*eur\b|\bavro\s*m\b|\bm\s*avro\b)", raw_all):
        return 1_000_000.0
    if re.search(r"(?:€\s*bn|bn\s*€|\beur\s*bn\b|\bbn\s*eur\b|\bavro\s*bn\b|\bbn\s*avro\b)", raw_all):
        return 1_000_000_000.0
    if re.search(r"(?:\$\s*m|m\s*\$|\busd\s*m\b|\bm\s*usd\b|\bdolar\s*m\b|\bm\s*dolar\b)", raw_all):
        return 1_000_000.0
    if re.search(r"(?:\$\s*bn|bn\s*\$|\busd\s*bn\b|\bbn\s*usd\b|\bdolar\s*bn\b|\bbn\s*dolar\b)", raw_all):
        return 1_000_000_000.0
    if re.search(r"(?:£\s*m|m\s*£|\bgbp\s*m\b|\bm\s*gbp\b|\bsterlin\s*m\b|\bm\s*sterlin\b)", raw_all):
        return 1_000_000.0
    if re.search(r"(?:£\s*bn|bn\s*£|\bgbp\s*bn\b|\bbn\s*gbp\b|\bsterlin\s*bn\b|\bbn\s*sterlin\b)", raw_all):
        return 1_000_000_000.0
    if re.search(r"\btlm\b", raw_all):
        return 1_000_000.0
    if re.search(r"\(\s*[€$£]\s*m\s*\)|\(\s*m\s*[€$£]\s*\)", raw_all):
        return 1_000_000.0

    local = local_context_norm or ""
    if re.search(r"\btlm\b", local):
        return 1_000_000.0
    if re.search(r"\bmlr\b", local):
        return 1_000_000_000.0
    if re.search(r"\bmn\b", local):
        return 1_000_000.0
    if "milyar" in local:
        return 1_000_000_000.0
    if "milyon" in local:
        return 1_000_000.0
    if "bin" in local:
        return 1_000.0

    if re.search(r"\btlm\b", context_norm):
        return 1_000_000.0
    if re.search(r"\bmlr\b", context_norm):
        return 1_000_000_000.0
    if re.search(r"\bmn\b", context_norm):
        return 1_000_000.0
    has_milyar = "milyar" in context_norm
    has_milyon = "milyon" in context_norm
    if has_milyar and not has_milyon:
        return 1_000_000_000.0
    if has_milyon and not has_milyar:
        return 1_000_000.0
    if has_milyar and has_milyon:
        # Mixed contexts are common in slide tables; prefer conservative scaling.
        return 1_000_000.0

    if "milyar tl" in text_norm or "milyar" in text_norm:
        return 1_000_000_000.0
    if "milyon tl" in text_norm or "milyon" in text_norm:
        return 1_000_000.0
    if re.search(r"\btlm\b", text_norm):
        return 1_000_000.0
    if re.search(r"\bmlr\s*tl\b", text_norm) or re.search(r"\bmlr\b", text_norm):
        return 1_000_000_000.0
    if re.search(r"\bmn\s*tl\b", text_norm) or re.search(r"\bmn\b", text_norm):
        return 1_000_000.0
    return 1.0


def _implicit_table_multiplier(
    metric: str,
    unit: str,
    current_multiplier: float,
    parsed_value: float,
    raw_value: str,
    block_type: str,
    section_norm: str,
    text_norm: str,
) -> float:
    if unit != "TL" or current_multiplier != 1.0:
        return current_multiplier
    if metric not in MONETARY_METRICS:
        return current_multiplier
    if block_type != "table_like":
        return current_multiplier
    has_year_signal = bool(re.search(r"20\d{2}", text_norm))
    has_fin_table_section = any(
        hint in section_norm for hint in ("gelir tablosu", "ozet konsolide gelir", "nakit akis", "cash flow", "finansal")
    )
    if not has_year_signal and not has_fin_table_section:
        return current_multiplier

    abs_value = abs(parsed_value)
    if abs_value < 100.0 or abs_value > 1_000_000.0:
        return current_multiplier

    # Quarter-rich tables often omit explicit units in each row.
    # If the chunk carries period columns, assume million-scale by default.
    compact_text = re.sub(r"[^a-z0-9]+", "", text_norm.lower())
    if any(token in compact_text for token in ("q1", "q2", "q3", "q4", "1c", "2c", "3c", "4c", "1q", "2q", "3q", "4q", "12a", "12ay", "9a", "9ay", "6a", "1yy", "1y")):
        return 1_000_000.0

    if not has_fin_table_section:
        return current_multiplier

    # Values like 67.651 in financial tables are often million TL units.
    if "." in raw_value or "," in raw_value or abs_value >= 1000.0:
        return 1_000_000.0
    return current_multiplier


def _candidate_score(
    metric: str,
    section_norm: str,
    context_norm: str,
    chunk_text: str,
    block_type: str,
    rank: int,
) -> Tuple[float, List[str]]:
    score = max(0.0, 22.0 - float(rank))
    reasons: List[str] = []
    unit = str(METRIC_DEFINITIONS.get(metric, {}).get("unit", "TL"))
    hints = METRIC_DEFINITIONS.get(metric, {}).get("section_hints", ())

    if any(hint in section_norm for hint in hints):
        score += 14.0
        reasons.append("section_title_match")
    if "gelir tablosuna gelen" in section_norm:
        score -= 30.0
        reasons.append("section_contribution_penalty")
    if block_type == "table_like":
        score += 10.0
        reasons.append("table_like_boost")

    metric_synonyms = METRIC_DEFINITIONS.get(metric, {}).get("synonyms", ())
    label = normalize_for_match(str(METRIC_DEFINITIONS.get(metric, {}).get("label", metric)))
    query = normalize_for_match(str(METRIC_DEFINITIONS.get(metric, {}).get("query", metric)))
    same_window_exact = False
    if label and label in context_norm:
        score += 22.0
        reasons.append("label_exact_match")
        same_window_exact = True
    elif query and query in context_norm:
        score += 20.0
        reasons.append("query_exact_match")
        same_window_exact = True

    if not same_window_exact and any(hint in context_norm for hint in metric_synonyms):
        score += 14.0
        reasons.append("label_synonym_match")

    if not same_window_exact:
        metric_tokens = set(re.findall(r"[a-z0-9]+", f"{label} {query}"))
        context_tokens = set(re.findall(r"[a-z0-9]+", context_norm))
        overlap = len(metric_tokens.intersection(context_tokens))
        if overlap >= 2:
            score += 8.0
            reasons.append("label_fuzzy_match")

    if metric == "magaza_sayisi":
        if "magaza sayilari" in section_norm:
            score += 40.0
            reasons.append("magaza_section_bonus")
        if "toplam" in context_norm:
            score += 10.0
            reasons.append("toplam_keyword_bonus")
    elif unit == "%":
        if "%" in context_norm or "marji" in context_norm:
            score += 8.0
            reasons.append("percent_context_match")
    else:
        if any(token in context_norm for token in ("tl", "milyon", "milyar", "eur", "usd", "avro", "dolar")):
            score += 8.0
            reasons.append("currency_context_match")

    line_score, line_reason = _line_proximity_score(metric=metric, text=chunk_text, context_norm=context_norm)
    score += line_score
    reasons.append(line_reason)
    return score, reasons


def _line_proximity_score(metric: str, text: str, context_norm: str) -> Tuple[float, str]:
    lines = [normalize_for_match(line) for line in text.splitlines() if line.strip()]
    if not lines:
        return 2.0, "window_proximity"

    synonyms = [str(item) for item in METRIC_DEFINITIONS.get(metric, {}).get("synonyms", ())]
    synonyms = [normalize_for_match(item) for item in synonyms if item]
    if not synonyms:
        synonyms = [normalize_for_match(metric)]

    if any(syn in context_norm for syn in synonyms):
        return 14.0, "same_line_proximity"
    for line in lines:
        if any(syn in line for syn in synonyms):
            return 6.0, "near_line_proximity"
    return 2.0, "window_proximity"


def _has_store_context(*segments: str) -> bool:
    combined = normalize_for_match(" ".join(segment for segment in segments if segment))
    return any(keyword in combined for keyword in STORE_CONTEXT_KEYWORDS)


def _is_consolidated_context(context_norm: str, section_norm: str) -> bool:
    scope = f"{context_norm} {section_norm}"
    return any(
        token in scope
        for token in (
            "toplam",
            "konsolide",
            "ana ortak",
            "anaortak",
            "ufrs",
            "gelir tablosu",
        )
    )


def _metric_specific_adjustment(
    metric: str,
    before_match_norm: str,
    context_norm: str,
    text_norm: str,
    scaled_value: float,
    multiplier: float,
) -> float:
    adjustment = 0.0

    if metric == "net_kar":
        if "net satislar" in before_match_norm or "satis buyumesi" in before_match_norm:
            adjustment -= 18.0
        if "donem net kari" in context_norm or "ana ortakliga ait net donem kari" in context_norm:
            adjustment += 14.0
        if any(token in context_norm for token in ("ana ortak", "anaortak", "ana ortaklik pay", "ana ortaklara dusen")):
            adjustment += 24.0
        if "gelir tablosuna gelen" in context_norm:
            adjustment -= 24.0
        if multiplier >= 1_000_000.0 and scaled_value > 25_000_000_000.0:
            adjustment -= 20.0
    elif metric == "magaza_sayisi":
        if scaled_value < 300.0 and "magaza" in context_norm:
            # Small values in pages that also contain 4-digit counts are often
            # opening counts, not total store count.
            if re.search(r"\b\d{4,5}\b", text_norm):
                adjustment -= 24.0
        if scaled_value >= 1000.0:
            adjustment += 10.0
    elif metric == "capex":
        # CAPEX often appears in cash-flow sections; store/percentage contexts
        # are usually not a monetary CAPEX value.
        if "magaza" in context_norm:
            adjustment -= 40.0
        if "%" in context_norm and abs(scaled_value) < 1_000_000.0:
            adjustment -= 26.0

    return adjustment


def _is_year_like_store_value(
    scaled_value: float,
    before_match_norm: str,
    context_norm: str,
) -> bool:
    if scaled_value < 1900.0 or scaled_value > 2100.0:
        return False
    rounded = int(round(scaled_value))
    scope = f"{before_match_norm} {context_norm}"
    years = [int(item) for item in re.findall(r"\b20\d{2}\b", scope)]
    return rounded in years


def _prefer_latest_year_value(
    metric: str,
    unit: str,
    text_norm: str,
    match_start: int,
    match_end: int,
    default_value: float,
) -> float:
    if unit != "TL" or metric not in {"net_kar", "favok", "satis_gelirleri", "brut_kar"}:
        return default_value

    before = text_norm[max(0, match_start - 140) : match_start]
    years = re.findall(r"20\d{2}", before)
    if len(years) < 2:
        return default_value

    first_year, second_year = years[-2], years[-1]
    if first_year >= second_year:
        return default_value

    # In slides that list 2024 then 2025, the first value after metric can be prior year.
    tail = text_norm[match_end : min(len(text_norm), match_end + 28)]
    next_value_match = re.search(r"\s+(\(?[\-]?\d[\d\.,]*\)?)", tail)
    if not next_value_match:
        return default_value
    next_value = parse_tr_number(next_value_match.group(1))
    if next_value is None:
        return default_value
    return float(next_value)


def _quarter_year_alignment_score(
    chunk: "RetrievedChunk",
    quarter: str,
    context_norm: str,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []
    chunk_quarter = quarter_label(str(getattr(chunk, "quarter", "")))
    expected = quarter_label(quarter)
    if chunk_quarter and expected and chunk_quarter == expected:
        score += 8.0
        reasons.append("quarter_alignment")
    elif chunk_quarter and expected and chunk_quarter != expected:
        score -= 8.0
        reasons.append("quarter_mismatch_penalty")

    chunk_year = getattr(chunk, "year", None)
    if chunk_year in (2025, "2025"):
        score += 2.0
        reasons.append("year_alignment")
    elif "2025" in context_norm:
        score += 1.0
        reasons.append("year_context_match")
    return score, reasons


_QUARTER_CONTEXT_TOKENS: Dict[str, Tuple[str, ...]] = {
    "Q1": ("q1", "1c", "1q", "ucaylik", "birinciceyrek"),
    "Q2": ("q2", "2c", "2q", "6a", "1yy", "1y", "ilkyariyil", "ikinciceyrek"),
    "Q3": ("q3", "3c", "3q", "9a", "9ay", "dokuzaylik", "ucuncuceyrek"),
    "Q4": ("q4", "4c", "4q", "12a", "12ay", "yillik", "fy", "dorduncuceyrek"),
}


def _quarter_context_score(quarter: str, context_norm: str, before_match_norm: str) -> Tuple[float, List[str]]:
    expected = quarter_label(quarter)
    if expected not in _QUARTER_CONTEXT_TOKENS:
        return 0.0, []

    combined = f"{before_match_norm} {context_norm}".strip()
    combined_compact = re.sub(r"[^a-z0-9]+", "", combined)
    target_tokens = _QUARTER_CONTEXT_TOKENS[expected]
    other_tokens: List[str] = []
    for q, tokens in _QUARTER_CONTEXT_TOKENS.items():
        if q != expected:
            other_tokens.extend(tokens)

    score = 0.0
    reasons: List[str] = []
    has_target = any((token in combined) or (token in combined_compact) for token in target_tokens)
    has_other = any((token in combined) or (token in combined_compact) for token in other_tokens)

    if has_target:
        score += 18.0
        reasons.append("quarter_token_match")
    elif has_other:
        score -= 16.0
        reasons.append("other_quarter_token_penalty")

    annual_pair_header = bool(re.search(r"\b20\d{2}\s+20\d{2}\b", combined))
    annual_hint = annual_pair_header or ("yillik" in combined) or ("fy" in combined)
    if annual_hint and not has_target:
        score -= 22.0 if expected == "Q4" else 12.0
        reasons.append("annual_header_penalty")

    return score, reasons


def _confidence_from_candidates(
    selected_index: int,
    candidates: Sequence[Dict[str, Any]],
) -> float:
    if not candidates:
        return 0.0
    selected = candidates[selected_index]
    selected_score = float(selected.get("score", 0.0))
    competitor_score = float(candidates[0].get("score", 0.0))
    if selected_index == 0 and len(candidates) > 1:
        competitor_score = float(candidates[1].get("score", 0.0))
    margin = selected_score - competitor_score

    base = 0.20 + max(0.0, min(selected_score / 110.0, 0.55))
    margin_bonus = max(0.0, min(margin / 40.0, 0.25))
    fallback_penalty = min(0.30, 0.10 * selected_index)
    confidence = base + margin_bonus - fallback_penalty
    return max(0.0, min(1.0, confidence))


def extract_metric_with_candidates(
    chunks: Sequence["RetrievedChunk"],
    metric: str,
    quarter: str,
    top_n: int = TOP_CANDIDATES_DEFAULT,
    use_structured_table_reconstruction: bool = True,
    use_expected_range_sanity: bool = True,
) -> Dict[str, Any]:
    config = METRIC_DEFINITIONS.get(metric)
    if not config:
        return {"selected": None, "candidates": []}

    all_candidates: List[Dict[str, Any]] = []
    forbidden_hints = tuple(str(item) for item in config.get("forbidden_hints", ()))
    expected_range = _expected_range_for_metric(metric) if use_expected_range_sanity else None

    for rank, chunk in enumerate(chunks):
        text = str(getattr(chunk, "text", ""))
        section = str(getattr(chunk, "section_title", ""))
        block_type = str(getattr(chunk, "block_type", "text"))
        text_norm = normalize_for_match(text)
        text_norm_lines = "\n".join(normalize_for_match(line) for line in text.splitlines())
        section_norm = normalize_for_match(section)
        if metric == "net_kar" and "gelir tablosuna gelen net k" in section_norm:
            continue
        table_like_detected = _looks_like_table_chunk(text_norm=text_norm_lines, block_type=block_type)
        effective_block_type = "table_like" if table_like_detected else block_type

        for pattern_idx, pattern in enumerate(config["patterns"]):
            for match in pattern.finditer(text_norm):
                group_index = match.lastindex or 1
                raw_value = match.group(group_index).strip()
                value = parse_tr_number(raw_value)
                if value is None:
                    continue

                start = max(0, match.start() - 90)
                end = min(len(text_norm), match.end() + 90)
                context_norm = text_norm[start:end]
                local_start = max(0, match.start() - 14)
                local_end = min(len(text_norm), match.end() + 24)
                local_context_norm = text_norm[local_start:local_end]
                raw_context = text[max(0, start) : min(len(text), end)]
                before_start = max(0, match.start() - 120)
                before_match_norm = text_norm[before_start:match.start()]
                if metric == "magaza_sayisi":
                    store_window = text_norm[max(0, match.start() - 80) : min(len(text_norm), match.end() + 80)]
                    if not _has_store_context(store_window, section_norm):
                        continue

                unit = str(config["unit"])
                value = _prefer_latest_year_value(
                    metric=metric,
                    unit=unit,
                    text_norm=text_norm,
                    match_start=match.start(),
                    match_end=match.end(),
                    default_value=float(value),
                )
                value = _apply_negative_signal(
                    metric=metric,
                    parsed_value=float(value),
                    raw_value=raw_value,
                    context_norm=context_norm,
                    before_match_norm=before_match_norm,
                )
                multiplier = _unit_multiplier(
                    metric=metric,
                    context_norm=context_norm,
                    text_norm=text_norm,
                    local_context_norm=local_context_norm,
                    raw_context=raw_context,
                    raw_text=text,
                )
                currency = _detect_currency(
                    metric=metric,
                    context_norm=context_norm,
                    text_norm=text_norm,
                    local_context_norm=local_context_norm,
                    raw_context=raw_context,
                    raw_text=text,
                )
                implicit_multiplier = _implicit_table_multiplier(
                    metric=metric,
                    unit=unit,
                    current_multiplier=multiplier,
                    parsed_value=float(value),
                    raw_value=raw_value,
                    block_type=effective_block_type,
                    section_norm=section_norm,
                    text_norm=text_norm,
                )
                implicit_multiplier_used = implicit_multiplier != multiplier
                if implicit_multiplier_used:
                    multiplier = implicit_multiplier
                scaled_value = float(value) * multiplier

                candidate_score, reasons = _candidate_score(
                    metric=metric,
                    section_norm=section_norm,
                    context_norm=context_norm,
                    chunk_text=text,
                    block_type=effective_block_type,
                    rank=rank,
                )
                if table_like_detected and block_type != "table_like":
                    reasons.append("table_like_detected_from_text")
                if implicit_multiplier_used:
                    reasons.append("implicit_million_table_multiplier")
                candidate_score += _metric_specific_adjustment(
                    metric=metric,
                    before_match_norm=before_match_norm,
                    context_norm=context_norm,
                    text_norm=text_norm,
                    scaled_value=scaled_value,
                    multiplier=multiplier,
                )
                if metric == "magaza_sayisi":
                    if _is_year_like_store_value(
                        scaled_value=scaled_value,
                        before_match_norm=before_match_norm,
                        context_norm=context_norm,
                    ):
                        candidate_score -= 70.0
                        reasons.append("year_token_as_store_penalty")
                    if any(token in context_norm for token in STORE_LFL_CONTEXT_KEYWORDS):
                        candidate_score -= 34.0
                        reasons.append("store_like_for_like_context_penalty")
                    if any(token in context_norm for token in STORE_TOTAL_CONTEXT_KEYWORDS):
                        candidate_score += 26.0
                        reasons.append("store_total_context_bonus")
                    if (
                        scaled_value < 5000.0
                        and any(token in context_norm for token in ("tl", "gunluk", "magaza basi"))
                    ):
                        candidate_score -= 28.0
                        reasons.append("store_count_currency_context_penalty")
                if metric in MONETARY_METRICS:
                    number_density = len(TABLE_NUMBER_PATTERN.findall(context_norm))
                    if number_density >= 8 and not any(
                        token in context_norm for token in ("toplam", "ana ortak", "konsolide", "ufrs")
                    ):
                        candidate_score -= 14.0
                        reasons.append("decomposition_table_penalty")
                qy_score, qy_reasons = _quarter_year_alignment_score(
                    chunk=chunk,
                    quarter=quarter,
                    context_norm=context_norm,
                )
                candidate_score += qy_score
                reasons.extend(qy_reasons)
                quarter_ctx_score, quarter_ctx_reasons = _quarter_context_score(
                    quarter=quarter,
                    context_norm=context_norm,
                    before_match_norm=before_match_norm,
                )
                candidate_score += quarter_ctx_score
                reasons.extend(quarter_ctx_reasons)

                immediate_span = text_norm[match.start() : min(len(text_norm), match.start() + 26)]
                if any(hint in immediate_span for hint in forbidden_hints):
                    candidate_score -= 16.0
                    reasons.append("forbidden_hint_penalty")

                if unit == "TL" and "baz puan" in context_norm:
                    candidate_score -= 22.0
                    reasons.append("baz_puan_penalty")
                if unit == "TL" and "%" in context_norm:
                    candidate_score -= 10.0
                    reasons.append("percent_near_tl_penalty")

                validation = validate_metric_value(
                    metric=metric,
                    value=scaled_value,
                    unit=unit,
                    expected_range=expected_range,
                )
                validation_ok = bool(validation.get("ok"))
                validation_reason = str(validation.get("reason", "ok"))
                if metric in {"net_kar", "satis_gelirleri", "favok"}:
                    if (
                        "decomposition_table_penalty" in reasons
                        and "quarter_token_match" not in reasons
                        and not _is_consolidated_context(context_norm, section_norm)
                    ):
                        validation_ok = False
                        validation_reason = "decomposition_context_rejected"
                if validation_ok:
                    reasons.append("sanity_ok")
                else:
                    reasons.append(f"sanity_fail:{validation_reason}")

                all_candidates.append(
                    {
                        "company": str(getattr(chunk, "company", "BIM")),
                        "quarter": quarter,
                        "metric": metric,
                        "value": scaled_value,
                        "unit": unit,
                        "doc_id": str(getattr(chunk, "doc_id", "")),
                        "page": int(getattr(chunk, "page", 0)),
                        "section_title": section,
                        "chunk_id": str(getattr(chunk, "chunk_id", "")),
                        "block_type": block_type,
                        "value_raw": float(value),
                        "multiplier": multiplier,
                        "currency": currency,
                        "excerpt": " ".join(text.split())[:360],
                        "score": float(candidate_score),
                        "rank": int(rank),
                        "pattern_index": int(pattern_idx),
                        "reasons": reasons,
                        "validation_ok": validation_ok,
                        "validation_reason": validation_reason,
                    }
                )

        if use_structured_table_reconstruction:
            structured_candidates = _structured_table_candidates(
                metric=metric,
                quarter=quarter,
                chunk=chunk,
                text_norm=text_norm_lines,
                section_norm=section_norm,
            )
            for idx, seed in enumerate(structured_candidates):
                scaled_value = float(seed["scaled_value"])
                context_norm = str(seed["context_norm"])
                before_match_norm = str(seed["before_match_norm"])
                raw_value = str(seed["raw_value"])
                multiplier = float(seed.get("multiplier", 1.0))
                unit = str(seed.get("unit", config["unit"]))
                currency = str(seed.get("currency", "TL"))
                if metric == "magaza_sayisi" and not _has_store_context(context_norm, section_norm):
                    continue

                candidate_score, reasons = _candidate_score(
                    metric=metric,
                    section_norm=section_norm,
                    context_norm=context_norm,
                    chunk_text=text,
                    block_type="table_like",
                    rank=rank,
                )
                reasons.append("structured_table_row_reconstruction")
                candidate_score += 6.0
                candidate_score += _metric_specific_adjustment(
                    metric=metric,
                    before_match_norm=before_match_norm,
                    context_norm=context_norm,
                    text_norm=text_norm,
                    scaled_value=scaled_value,
                    multiplier=multiplier,
                )
                if metric == "magaza_sayisi":
                    if _is_year_like_store_value(
                        scaled_value=scaled_value,
                        before_match_norm=before_match_norm,
                        context_norm=context_norm,
                    ):
                        candidate_score -= 70.0
                        reasons.append("year_token_as_store_penalty")
                    if any(token in context_norm for token in STORE_LFL_CONTEXT_KEYWORDS):
                        candidate_score -= 34.0
                        reasons.append("store_like_for_like_context_penalty")
                    if any(token in context_norm for token in STORE_TOTAL_CONTEXT_KEYWORDS):
                        candidate_score += 26.0
                        reasons.append("store_total_context_bonus")
                    if (
                        scaled_value < 5000.0
                        and any(token in context_norm for token in ("tl", "gunluk", "magaza basi"))
                    ):
                        candidate_score -= 28.0
                        reasons.append("store_count_currency_context_penalty")
                if metric in MONETARY_METRICS:
                    number_density = len(TABLE_NUMBER_PATTERN.findall(context_norm))
                    if number_density >= 8 and not any(
                        token in context_norm for token in ("toplam", "ana ortak", "konsolide", "ufrs")
                    ):
                        candidate_score -= 14.0
                        reasons.append("decomposition_table_penalty")
                qy_score, qy_reasons = _quarter_year_alignment_score(
                    chunk=chunk,
                    quarter=quarter,
                    context_norm=context_norm,
                )
                candidate_score += qy_score
                reasons.extend(qy_reasons)
                quarter_ctx_score, quarter_ctx_reasons = _quarter_context_score(
                    quarter=quarter,
                    context_norm=context_norm,
                    before_match_norm=before_match_norm,
                )
                candidate_score += quarter_ctx_score
                reasons.extend(quarter_ctx_reasons)

                validation = validate_metric_value(
                    metric=metric,
                    value=scaled_value,
                    unit=unit,
                    expected_range=expected_range,
                )
                validation_ok = bool(validation.get("ok"))
                validation_reason = str(validation.get("reason", "ok"))
                if metric in {"net_kar", "satis_gelirleri", "favok"}:
                    if (
                        "decomposition_table_penalty" in reasons
                        and "quarter_token_match" not in reasons
                        and not _is_consolidated_context(context_norm, section_norm)
                    ):
                        validation_ok = False
                        validation_reason = "decomposition_context_rejected"
                if validation_ok:
                    reasons.append("sanity_ok")
                else:
                    reasons.append(f"sanity_fail:{validation_reason}")

                all_candidates.append(
                    {
                        "company": str(getattr(chunk, "company", "BIM")),
                        "quarter": quarter,
                        "metric": metric,
                        "value": scaled_value,
                        "unit": unit,
                        "doc_id": str(getattr(chunk, "doc_id", "")),
                        "page": int(getattr(chunk, "page", 0)),
                        "section_title": section,
                        "chunk_id": str(getattr(chunk, "chunk_id", "")),
                        "block_type": "table_like",
                        "value_raw": float(parse_tr_number(raw_value) or 0.0),
                        "multiplier": multiplier,
                        "currency": currency,
                        "excerpt": " ".join(text.split())[:360],
                        "score": float(candidate_score),
                        "rank": int(rank),
                        "pattern_index": int(seed.get("pattern_index", -(2000 + idx))),
                        "reasons": reasons,
                        "validation_ok": validation_ok,
                        "validation_reason": validation_reason,
                    }
                )

    all_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    candidate_limit = max(1, int(top_n))
    if metric == "magaza_sayisi":
        # Store-count extraction is noisy in some table layouts; keep a wider
        # candidate window to avoid selecting small unrelated integers.
        candidate_limit = max(candidate_limit, 12)
    top_candidates = all_candidates[:candidate_limit]
    selection_candidates: List[Dict[str, Any]] = list(top_candidates)

    selected_index: Optional[int] = next(
        (idx for idx, candidate in enumerate(selection_candidates) if bool(candidate.get("validation_ok"))),
        None,
    )
    if selected_index is None:
        fallback_valid = next(
            (candidate for candidate in all_candidates[candidate_limit:] if bool(candidate.get("validation_ok"))),
            None,
        )
        if fallback_valid is not None:
            selection_candidates.append(fallback_valid)
            selected_index = len(selection_candidates) - 1

    if selected_index is not None and quarter in QUARTER_ORDER:
        quarter_token_candidate_indices = [
            idx
            for idx, candidate in enumerate(selection_candidates)
            if bool(candidate.get("validation_ok")) and "quarter_token_match" in list(candidate.get("reasons", []))
        ]
        if quarter_token_candidate_indices:
            best_quarter_idx = max(
                quarter_token_candidate_indices,
                key=lambda idx: float(selection_candidates[idx].get("score", 0.0)),
            )
            if "quarter_token_match" not in list(selection_candidates[selected_index].get("reasons", [])):
                selected_index = best_quarter_idx

    selected: Optional[Dict[str, Any]] = None
    if selected_index is not None:
        if metric == "magaza_sayisi":
            current_candidate = selection_candidates[selected_index]
            current_value = float(current_candidate.get("value", 0.0))
            if current_value < 300.0:
                switched_to_large = False
                for idx, candidate in enumerate(selection_candidates):
                    if not bool(candidate.get("validation_ok")):
                        continue
                    candidate_value = float(candidate.get("value", 0.0))
                    if candidate_value >= 500.0:
                        selected_index = idx
                        switched_to_large = True
                        break
                if not switched_to_large:
                    fallback_large = next(
                        (
                            candidate
                            for candidate in all_candidates
                            if bool(candidate.get("validation_ok")) and float(candidate.get("value", 0.0)) >= 500.0
                        ),
                        None,
                    )
                    if fallback_large is not None:
                        if fallback_large not in selection_candidates:
                            selection_candidates.append(fallback_large)
                            selected_index = len(selection_candidates) - 1
                            switched_to_large = True
                if switched_to_large:
                    # Marked later in selected reasons.
                    pass
            current_candidate = selection_candidates[selected_index]
            current_reasons = set(str(item) for item in current_candidate.get("reasons", []))
            if "store_total_context_bonus" not in current_reasons:
                total_idx: Optional[int] = None
                total_value = current_value
                for idx, candidate in enumerate(selection_candidates):
                    if not bool(candidate.get("validation_ok")):
                        continue
                    reasons = set(str(item) for item in candidate.get("reasons", []))
                    if "store_total_context_bonus" not in reasons:
                        continue
                    candidate_value = float(candidate.get("value", 0.0))
                    if candidate_value > total_value:
                        total_idx = idx
                        total_value = candidate_value
                if total_idx is not None and total_value >= max(current_value + 200.0, current_value * 1.03):
                    selected_index = total_idx
        selected = dict(selection_candidates[selected_index])
        selected_reasons = list(selected.get("reasons", []))
        if selected_index >= len(top_candidates):
            selected_reasons.append("valid_candidate_outside_top_n")
        if metric == "magaza_sayisi":
            try:
                if float(selected.get("value", 0.0)) >= 500.0:
                    selected_reasons.append("store_count_total_preferred")
            except Exception:
                pass
        if selected_index > 0:
            selected_reasons.append("sanity_fallback_next_candidate")
        confidence = _confidence_from_candidates(selected_index, selection_candidates)
        selected["confidence"] = confidence
        if confidence < 0.45:
            selected_reasons.append("low_margin_confidence")
        selected["reasons"] = selected_reasons
        verify = auto_verify_metric(
            metric=metric,
            selected=selected,
            candidates=selection_candidates,
            quarter=quarter,
        )
        selected["verify_status"] = verify.get("status", "FAIL")
        selected["verify_checks"] = list(verify.get("checks", []))
        selected["verify_warnings"] = list(verify.get("warnings", []))
        selected["verify_reasons"] = list(verify.get("reasons", []))
        selected["verify_alternate_value"] = verify.get("alternate_value")
        selected["candidates"] = selection_candidates
        selected["candidate_count"] = len(selection_candidates)

    return {
        "selected": selected,
        "candidates": top_candidates,
    }


def extract_metric_from_chunks(
    chunks: Sequence["RetrievedChunk"],
    metric: str,
    quarter: str,
) -> Optional[Dict[str, Any]]:
    result = extract_metric_with_candidates(
        chunks=chunks,
        metric=metric,
        quarter=quarter,
        top_n=TOP_CANDIDATES_DEFAULT,
    )
    return result.get("selected")


def _trend_deviation_pct(value: Optional[float], anchor: Optional[float]) -> Optional[float]:
    if value is None or anchor is None:
        return None
    if abs(float(anchor)) < 1e-9:
        return None
    # Sign flips can be real (profit <-> loss), do not over-penalize.
    if float(value) * float(anchor) < 0:
        return None
    return abs(float(value) - float(anchor)) / abs(float(anchor)) * 100.0


def _trend_consistency_score(value: Optional[float], prev_value: Optional[float], next_value: Optional[float]) -> Tuple[float, Optional[float]]:
    deviations = [
        deviation
        for deviation in (
            _trend_deviation_pct(value, prev_value),
            _trend_deviation_pct(value, next_value),
        )
        if deviation is not None
    ]
    if not deviations:
        return 1.0, None
    worst = max(deviations)
    score = max(0.0, min(1.0, 1.0 - (worst / max(TREND_DEVIATION_THRESHOLD_PCT, 1.0))))
    return score, worst


def _promote_candidate_from_list(
    metric: str,
    quarter: str,
    candidates: Sequence[Dict[str, Any]],
    candidate_index: int,
    reason: str,
) -> Dict[str, Any]:
    selected = dict(candidates[candidate_index])
    selected_reasons = list(selected.get("reasons", []))
    selected_reasons.append(reason)
    if candidate_index > 0:
        selected_reasons.append("sanity_fallback_next_candidate")
    if metric == "magaza_sayisi":
        try:
            if float(selected.get("value", 0.0)) >= 500.0:
                selected_reasons.append("store_count_total_preferred")
        except Exception:
            pass
    confidence = _confidence_from_candidates(candidate_index, candidates)
    selected["confidence"] = confidence
    if confidence < 0.45:
        selected_reasons.append("low_margin_confidence")
    selected["reasons"] = selected_reasons
    verify = auto_verify_metric(
        metric=metric,
        selected=selected,
        candidates=candidates,
        quarter=quarter,
    )
    selected["verify_status"] = verify.get("status", "FAIL")
    selected["verify_checks"] = list(verify.get("checks", []))
    selected["verify_warnings"] = list(verify.get("warnings", []))
    selected["verify_reasons"] = list(verify.get("reasons", []))
    selected["verify_alternate_value"] = verify.get("alternate_value")
    selected["candidates"] = list(candidates)
    selected["candidate_count"] = len(candidates)
    return selected


def aggregate_metric_across_quarters(
    quarter_chunks: Dict[str, Sequence["RetrievedChunk"]],
    metric: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    extracted_records: List[Dict[str, Any]] = []

    quarter_payload: Dict[str, Dict[str, Any]] = {}
    for quarter in QUARTER_ORDER:
        chunks = quarter_chunks.get(quarter, [])
        extraction = extract_metric_with_candidates(
            chunks=chunks,
            metric=metric,
            quarter=quarter,
            top_n=TOP_CANDIDATES_DEFAULT,
        )
        top_candidates = list(extraction.get("candidates", []))
        record = extraction.get("selected")
        invalid_all = bool(top_candidates) and all(not bool(c.get("validation_ok")) for c in top_candidates)
        quarter_payload[quarter] = {
            "record": dict(record) if record else None,
            "candidates": top_candidates,
            "invalid_all": invalid_all,
        }

    # Cross-quarter consistency pass: if a selected value jumps too much,
    # try the next valid candidate in the same quarter.
    for idx, quarter in enumerate(QUARTER_ORDER):
        payload = quarter_payload[quarter]
        record = payload.get("record")
        if not record:
            continue
        current_value = float(record.get("value", 0.0))
        prev_record = quarter_payload.get(QUARTER_ORDER[idx - 1], {}).get("record") if idx > 0 else None
        next_record = quarter_payload.get(QUARTER_ORDER[idx + 1], {}).get("record") if idx < len(QUARTER_ORDER) - 1 else None
        prev_value = float(prev_record["value"]) if prev_record and prev_record.get("value") is not None else None
        next_value = float(next_record["value"]) if next_record and next_record.get("value") is not None else None
        consistency_score, worst_deviation = _trend_consistency_score(current_value, prev_value, next_value)
        record["trend_consistency_score"] = consistency_score

        if worst_deviation is None or worst_deviation <= TREND_DEVIATION_THRESHOLD_PCT:
            continue

        candidates = list(payload.get("candidates", []))
        if not candidates:
            warnings = list(record.get("verify_warnings", []))
            warnings.append("trend_outlier_suspected")
            record["verify_warnings"] = warnings
            if str(record.get("verify_status", "")).upper() == "PASS":
                record["verify_status"] = "WARN"
            continue

        selected_candidate_idx = 0
        for candidate_idx, candidate in enumerate(candidates):
            if (
                candidate.get("doc_id") == record.get("doc_id")
                and candidate.get("page") == record.get("page")
                and abs(float(candidate.get("value", 0.0)) - float(record.get("value", 0.0))) < 1e-9
            ):
                selected_candidate_idx = candidate_idx
                break

        best_idx: Optional[int] = None
        best_deviation = worst_deviation
        for candidate_idx, candidate in enumerate(candidates):
            if not bool(candidate.get("validation_ok")):
                continue
            if candidate_idx == selected_candidate_idx:
                continue
            cand_value = float(candidate.get("value", 0.0))
            _, cand_worst = _trend_consistency_score(cand_value, prev_value, next_value)
            if cand_worst is None:
                continue
            if cand_worst < best_deviation:
                best_deviation = cand_worst
                best_idx = candidate_idx
                if cand_worst <= TREND_DEVIATION_THRESHOLD_PCT:
                    break

        if best_idx is not None and best_deviation < worst_deviation:
            promoted = _promote_candidate_from_list(
                metric=metric,
                quarter=quarter,
                candidates=candidates,
                candidate_index=best_idx,
                reason="trend_consistency_fallback",
            )
            promoted_score, _ = _trend_consistency_score(float(promoted.get("value", 0.0)), prev_value, next_value)
            promoted["trend_consistency_score"] = promoted_score
            quarter_payload[quarter]["record"] = promoted
        else:
            warnings = list(record.get("verify_warnings", []))
            warnings.append("trend_outlier_suspected")
            record["verify_warnings"] = warnings
            if str(record.get("verify_status", "")).upper() == "PASS":
                record["verify_status"] = "WARN"

    for quarter in QUARTER_ORDER:
        payload = quarter_payload.get(quarter, {})
        record = payload.get("record")
        invalid_all = bool(payload.get("invalid_all", False))
        if record:
            extracted_records.append(record)
            citation = f"[{record['doc_id']} | {quarter} | {record['page']} | {record['section_title']}]"
            rows.append(
                {
                    "company": record.get("company"),
                    "quarter": quarter,
                    "metric": metric_display_name(metric),
                    "value": record["value"],
                    "unit": record["unit"],
                    "currency": str(record.get("currency", "TL")),
                    "value_display": format_metric_value(
                        float(record["value"]),
                        str(record["unit"]),
                        str(record.get("currency", "TL")),
                    ),
                    "citation": citation,
                    "confidence": record.get("confidence"),
                    "verify_status": record.get("verify_status"),
                    "verify_warnings": list(record.get("verify_warnings", [])),
                    "trend_consistency_score": record.get("trend_consistency_score", 1.0),
                    "invalid_all_candidates": False,
                }
            )
        else:
            rows.append(
                {
                    "company": None,
                    "quarter": quarter,
                    "metric": metric_display_name(metric),
                    "value": None,
                    "unit": metric_unit(metric),
                    "currency": "TL",
                    "value_display": "Bulunamadi",
                    "citation": None,
                    "confidence": None,
                    "verify_status": "FAIL",
                    "verify_warnings": [],
                    "trend_consistency_score": None,
                    "invalid_all_candidates": invalid_all,
                }
            )

    frame = pd.DataFrame(rows)
    numeric_values = pd.to_numeric(frame["value"], errors="coerce")
    frame["abs_change"] = numeric_values.diff()
    frame["pct_change"] = numeric_values.pct_change(fill_method=None) * 100.0

    def _direction(delta: Any) -> Optional[str]:
        if pd.isna(delta):
            return None
        if float(delta) > 0:
            return "increase"
        if float(delta) < 0:
            return "decrease"
        return "flat"

    frame["direction"] = frame["abs_change"].apply(_direction)
    return frame, extracted_records


def compute_overall_change(frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    valid = frame.dropna(subset=["value"])
    if len(valid) < 2:
        return {
            "abs_change": None,
            "pct_change": None,
            "direction": None,
        }

    start = float(valid.iloc[0]["value"])
    end = float(valid.iloc[-1]["value"])
    abs_change = end - start
    pct_change = None if start == 0 else (abs_change / start) * 100.0
    direction = "increase" if abs_change > 0 else "decrease" if abs_change < 0 else "flat"
    return {
        "abs_change": abs_change,
        "pct_change": pct_change,
        "direction": direction,
    }


def collect_top_sources(
    quarter_chunks: Dict[str, Sequence["RetrievedChunk"]],
    limit_total: int = 6,
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()
    for quarter in QUARTER_ORDER:
        for chunk in quarter_chunks.get(quarter, []):
            key = (
                str(getattr(chunk, "doc_id", "")),
                str(getattr(chunk, "company", "BIM")),
                str(getattr(chunk, "quarter", "")),
                int(getattr(chunk, "page", 0)),
                str(getattr(chunk, "section_title", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "doc_id": key[0],
                    "company": key[1],
                    "quarter": key[2],
                    "page": key[3],
                    "section_title": key[4] or "(no heading)",
                }
            )
            if len(sources) >= limit_total:
                return sources
    return sources


def run_cross_quarter_comparison(
    question: str,
    retriever: "RetrieverV3",
    top_k_initial: int = 20,
    top_k_final: int = 8,
    alpha: float = 0.35,
    company: Optional[str] = None,
) -> Dict[str, Any]:
    metric = infer_metric_from_question(question)
    quarter_chunks: Dict[str, Sequence["RetrievedChunk"]] = {}

    for quarter in QUARTER_ORDER:
        metric_query = build_metric_query(metric, quarter, question) if metric else question
        chunks = retriever.retrieve_with_query_awareness(
            query=metric_query,
            top_k_initial=top_k_initial,
            top_k_final=top_k_final,
            alpha=alpha,
            quarter_override=quarter,
            company_override=company,
        )
        quarter_chunks[quarter] = chunks

    if not metric:
        return {
            "metric": None,
            "frame": pd.DataFrame(
                [
                    {
                        "quarter": quarter,
                        "metric": None,
                        "value": None,
                        "unit": None,
                        "value_display": "Bulunamadi",
                        "citation": None,
                        "abs_change": None,
                        "pct_change": None,
                        "direction": None,
                    }
                    for quarter in QUARTER_ORDER
                ]
            ),
            "records": [],
            "overall_change": {"abs_change": None, "pct_change": None, "direction": None},
            "missing_quarters": QUARTER_ORDER.copy(),
            "found": False,
            "top_sources": collect_top_sources(quarter_chunks=quarter_chunks),
            "quarter_chunks": quarter_chunks,
        }

    frame, records = aggregate_metric_across_quarters(
        quarter_chunks=quarter_chunks,
        metric=metric,
    )
    overall_change = compute_overall_change(frame)
    missing_quarters = [
        str(row["quarter"]) for _, row in frame.iterrows() if pd.isna(row.get("value"))
    ]
    found = bool(records)

    return {
        "company": company,
        "metric": metric,
        "frame": frame,
        "records": records,
        "overall_change": overall_change,
        "missing_quarters": missing_quarters,
        "found": found,
        "top_sources": collect_top_sources(quarter_chunks=quarter_chunks),
        "quarter_chunks": quarter_chunks,
    }
