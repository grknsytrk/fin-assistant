from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)

QUARTER_PATTERN_TR = re.compile(
    r"(?P<year>20\d{2}).*?(?P<quarter>[1-4])\.\s*[cçCÇ]eyrek",
    re.IGNORECASE,
)
QUARTER_PATTERN_CQ_FIRST = re.compile(
    r"(?P<quarter>[1-4])\s*[cq]\s*[-_ ]?\s*(?P<year>20\d{2})",
    re.IGNORECASE,
)
QUARTER_PATTERN_CQ_LAST = re.compile(
    r"(?P<year>20\d{2})\s*[-_ ]?\s*(?P<quarter>[1-4])\s*[cq]",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"(20\d{2})")
COMPANY_ALIASES: Dict[str, str] = {
    "bim": "BIM",
    "bimas": "BIM",
    "bim birlesik magazalar": "BIM",
    "file": "FILE",
    "migros": "MIGROS",
    "sok": "SOK",
    "sok marketler": "SOK",
    "a101": "A101",
    # BIST aliases for broader company auto-detection from file names.
    "akbank": "AKBNK",
    "akbnk": "AKBNK",
    "aselsan": "ASELS",
    "asels": "ASELS",
    "emlak konut": "EKGYO",
    "ekgyo": "EKGYO",
    "enka": "ENKAI",
    "enkai": "ENKAI",
    "eregli": "EREGL",
    "erdemir": "EREGL",
    "eregl": "EREGL",
    "garanti": "GARAN",
    "garan": "GARAN",
    "is bankasi": "ISCTR",
    "isbank": "ISCTR",
    "isctr": "ISCTR",
    "koc holding": "KCHOL",
    "kchol": "KCHOL",
    "kozal": "TRALT",
    "koza altin": "TRALT",
    "tralt": "TRALT",
    "petkim": "PETKM",
    "petkm": "PETKM",
    "sabanci": "SAHOL",
    "sahol": "SAHOL",
    "sisecam": "SISE",
    "sise": "SISE",
    "thy": "THYAO",
    "thyao": "THYAO",
    "turk hava yollari": "THYAO",
    "tofas": "TOASO",
    "toaso": "TOASO",
    "tupras": "TUPRS",
    "tuprs": "TUPRS",
    "yapi kredi": "YKBNK",
    "ykbnk": "YKBNK",
}
COMPANY_QUARTER_TOKEN_PATTERN = re.compile(
    r"\b(?:q[1-4]|[1-4]q|[1-4]c|c[1-4]|[1-4]\.?\s*[cç]eyrek)\b",
    re.IGNORECASE,
)
COMPANY_COMPACT_PERIOD_PATTERN = re.compile(
    r"(?:20\d{2})?(?:q[1-4]|c[1-4]|[1-4]q|[1-4]c)(?:20\d{2})?",
    re.IGNORECASE,
)
COMPANY_VERSION_PATTERN = re.compile(r"\bv\d+\b", re.IGNORECASE)
COMPANY_NOISE_TOKENS = {
    "sunum",
    "yatirimci",
    "investor",
    "presentation",
    "webcast",
    "report",
    "raporu",
    "faaliyet",
    "gelir",
    "tablosu",
    "financial",
    "results",
    "tr",
    "turkce",
    "english",
}


@dataclass
class PageDocument:
    doc_id: str
    company: str
    quarter: str
    year: Optional[int]
    page: int
    text: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def list_pdf_files(raw_dir: Path) -> List[Path]:
    return sorted(raw_dir.glob("*.pdf"))


def parse_quarter_from_name(name: str) -> str:
    match = None
    for pattern in (QUARTER_PATTERN_TR, QUARTER_PATTERN_CQ_FIRST, QUARTER_PATTERN_CQ_LAST):
        match = pattern.search(name)
        if match:
            break
    if match:
        year = match.group("year")
        quarter = match.group("quarter")
        return f"{year}Q{quarter}"
    return "UNKNOWN"


def parse_year_from_name(name: str) -> Optional[int]:
    quarter_match = None
    for pattern in (QUARTER_PATTERN_TR, QUARTER_PATTERN_CQ_FIRST, QUARTER_PATTERN_CQ_LAST):
        quarter_match = pattern.search(name)
        if quarter_match:
            break
    if quarter_match:
        return int(quarter_match.group("year"))
    year_match = YEAR_PATTERN.search(name)
    if year_match:
        return int(year_match.group(1))
    return None


def parse_company_from_name(name: str) -> str:
    lowered = name.lower().strip()
    for alias, canonical in COMPANY_ALIASES.items():
        if alias in lowered:
            return canonical

    def _normalize_candidate(raw: str) -> str:
        text = str(raw or "").lower()
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\b20\d{2}\s*[qc]\s*[1-4]\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b[1-4]\s*[qc]\s*20\d{2}\b", " ", text, flags=re.IGNORECASE)
        text = YEAR_PATTERN.sub(" ", text)
        text = COMPANY_QUARTER_TOKEN_PATTERN.sub(" ", text)
        text = COMPANY_VERSION_PATTERN.sub(" ", text)
        parts: List[str] = []
        for token in text.split():
            token_norm = re.sub(r"[^a-z0-9çğıöşü]", "", token, flags=re.IGNORECASE)
            token_norm = COMPANY_COMPACT_PERIOD_PATTERN.sub("", token_norm)
            if not token_norm:
                continue
            if token_norm in COMPANY_NOISE_TOKENS:
                continue
            parts.append(token_norm)
        return " ".join(parts).strip()

    candidates: List[str] = [lowered]

    year_match = YEAR_PATTERN.search(lowered)
    if year_match:
        prefix = lowered[: year_match.start()].strip(" -_")
        suffix = lowered[year_match.end() :].strip(" -_")
        if prefix:
            candidates.insert(0, prefix)
        if suffix:
            candidates.append(suffix)
    else:
        prefix = ""

    for pattern in (QUARTER_PATTERN_TR, QUARTER_PATTERN_CQ_FIRST, QUARTER_PATTERN_CQ_LAST):
        match = pattern.search(lowered)
        if not match:
            continue
        left = lowered[: match.start()].strip(" -_")
        right = lowered[match.end() :].strip(" -_")
        if left:
            candidates.insert(0, left)
        if right:
            candidates.append(right)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_candidate(candidate)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        return normalized.upper()

    if year_match:
        prefix = lowered[: year_match.start()].strip(" -_")
        if prefix:
            compact = re.sub(r"\s+", " ", prefix).strip()
            return compact.upper()

    if "faaliyet raporu" in lowered:
        return "BIM"
    return "UNKNOWN"


def clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00A0", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_page_text(page) -> str:
    extracted = ""
    try:
        extracted = page.extract_text() or ""
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Sayfa metni extract_text ile okunamadı: %s", exc)

    if not extracted.strip():
        try:
            extracted = page.extract_text(extraction_mode="layout") or ""
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Sayfa metni layout fallback ile de okunamadı: %s", exc)

    if not extracted.strip():
        return ""

    return clean_text(extracted)


def extract_pdf_pages(pdf_path: Path) -> List[PageDocument]:
    reader = PdfReader(str(pdf_path))
    doc_id = pdf_path.stem
    company = parse_company_from_name(pdf_path.stem)
    quarter = parse_quarter_from_name(pdf_path.stem)
    year = parse_year_from_name(pdf_path.stem)

    page_documents: List[PageDocument] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = extract_page_text(page)
        if not text:
            LOGGER.info("Boş/okunamayan sayfa: %s - sayfa %d", pdf_path.name, page_num)
        page_documents.append(
            PageDocument(
                doc_id=doc_id,
                company=company,
                quarter=quarter,
                year=year,
                page=page_num,
                text=text,
            )
        )
    return page_documents


def save_pages_jsonl(pages: List[PageDocument], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page.to_dict(), ensure_ascii=False) + "\n")


def load_pages_jsonl(input_file: Path) -> List[PageDocument]:
    pages: List[PageDocument] = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            raw_year = raw.get("year")
            try:
                year = int(raw_year) if raw_year is not None and str(raw_year).strip() else None
            except Exception:
                year = parse_year_from_name(str(raw.get("doc_id", "")))
            pages.append(
                PageDocument(
                    doc_id=str(raw["doc_id"]),
                    company=str(raw.get("company", "BIM")),
                    quarter=str(raw["quarter"]),
                    year=year,
                    page=int(raw["page"]),
                    text=str(raw["text"]),
                )
            )
    return pages


def ingest_raw_pdfs(
    raw_dir: Path,
    output_file: Optional[Path] = None,
) -> Tuple[List[PageDocument], Dict[str, object]]:
    pdf_files = list_pdf_files(raw_dir)
    pages: List[PageDocument] = []
    pages_per_pdf: Dict[str, int] = {}
    companies_found = set()

    for pdf_path in pdf_files:
        pdf_pages = extract_pdf_pages(pdf_path)
        pages.extend(pdf_pages)
        pages_per_pdf[pdf_path.name] = len(pdf_pages)
        if pdf_pages:
            companies_found.add(pdf_pages[0].company)

    if output_file is not None:
        save_pages_jsonl(pages, output_file)

    summary: Dict[str, object] = {
        "num_pdfs": len(pdf_files),
        "pdf_files": [p.name for p in pdf_files],
        "pages_per_pdf": pages_per_pdf,
        "total_pages": len(pages),
        "companies": sorted(companies_found),
        "num_companies": len(companies_found),
    }
    return pages, summary


def summarize_pages_for_ingest(pages: List[PageDocument]) -> Dict[str, object]:
    pages_per_doc: Dict[str, int] = {}
    companies_found = set()
    for row in pages:
        pages_per_doc[row.doc_id] = pages_per_doc.get(row.doc_id, 0) + 1
        if row.company:
            companies_found.add(str(row.company).upper())

    return {
        "num_pdfs": len(pages_per_doc),
        "pdf_files": sorted(pages_per_doc.keys()),
        "pages_per_pdf": pages_per_doc,
        "total_pages": len(pages),
        "companies": sorted(companies_found),
        "num_companies": len(companies_found),
    }


def ingest_page_fixtures(
    fixtures_file: Path,
    output_file: Optional[Path] = None,
) -> Tuple[List[PageDocument], Dict[str, object]]:
    pages = load_pages_jsonl(fixtures_file)
    if output_file is not None:
        save_pages_jsonl(pages, output_file)
    return pages, summarize_pages_for_ingest(pages)
