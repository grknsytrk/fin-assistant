from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ingest import PageDocument

HEADING_KEYWORDS = [
    "RİSK",
    "RISK",
    "FİNANS",
    "FINANS",
    "ÖZET",
    "OZET",
    "YÖNET",
    "YONET",
    "FAALİYET",
    "FAALIYET",
    "PERFORMANS",
    "SONUÇ",
    "SONUC",
    "DEĞERLEND",
    "DEGERLEND",
    "BEKLENT",
    "STRATEJ",
    "YATIRIM",
    "KURUMSAL",
    "SÜRDÜR",
    "SURDUR",
    "PAZAR",
    "SEKTÖR",
    "SEKTOR",
]
NUMBERED_HEADING_PATTERN = re.compile(r"^\s*\d{1,2}(\.\d{1,2})*[\)\.]?\s+\S+")
NUMERIC_TOKEN_PATTERN = re.compile(r"\d[\d\.,]*")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
LEADING_FRAGMENT_SAFE_TOKENS = {
    "ve",
    "bu",
    "bir",
    "ile",
    "ancak",
    "fakat",
    "ama",
    "gibi",
    "hem",
    "ise",
    "sonra",
    "icin",
    "için",
    "olarak",
}


@dataclass
class TextChunk:
    doc_id: str
    company: str
    quarter: str
    year: Optional[int]
    page: int
    chunk_id: str
    text: str

    def metadata(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "company": self.company,
            "quarter": self.quarter,
            "year": self.year,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SectionBlock:
    section_title: str
    section_text: str
    start_page: int


@dataclass
class TextChunkV2:
    doc_id: str
    company: str
    quarter: str
    year: Optional[int]
    page: int
    chunk_id: str
    text: str
    section_title: str
    block_type: str = "text"
    chunk_version: str = "v2"

    def metadata(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "company": self.company,
            "quarter": self.quarter,
            "year": self.year,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "section_title": self.section_title,
            "block_type": self.block_type,
            "chunk_version": self.chunk_version,
        }

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def chunk_page(
    page_doc: PageDocument,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size pozitif olmalı")
    if overlap < 0:
        raise ValueError("overlap negatif olamaz")
    if overlap >= chunk_size:
        raise ValueError("overlap, chunk_size'dan küçük olmalı")

    text = page_doc.text.strip()
    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[TextChunk] = []
    raw_start = 0
    chunk_index = 1

    while raw_start < len(text):
        start = _snap_start_boundary(text, raw_start)
        if start >= len(text):
            break
        end = _snap_end_boundary(text, min(len(text), start + chunk_size))
        chunk_text = _clean_leading_fragment(text[start:end].strip())
        if chunk_text:
            if chunks:
                prev = chunks[-1].text
                too_small_overlap_tail = len(chunk_text) < max(40, overlap // 2) and chunk_text in prev
                if chunk_text == prev or too_small_overlap_tail:
                    if end >= len(text):
                        break
                    raw_start += step
                    continue
            chunk_id = f"{page_doc.doc_id}-p{page_doc.page}-c{chunk_index}"
            chunks.append(
                TextChunk(
                    doc_id=page_doc.doc_id,
                    company=page_doc.company,
                    quarter=page_doc.quarter,
                    year=page_doc.year,
                    page=page_doc.page,
                    chunk_id=chunk_id,
                    text=chunk_text,
                )
            )
            chunk_index += 1

        if end >= len(text):
            break
        raw_start += step

    return chunks


def chunk_documents(
    pages: List[PageDocument],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_size=chunk_size, overlap=overlap))
    return chunks


def save_chunks_jsonl(chunks: List[TextChunk], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def _is_word_char(ch: str) -> bool:
    return ch.isalnum()


def _snap_start_boundary(text: str, start: int) -> int:
    idx = max(0, min(start, len(text)))
    if idx >= len(text):
        return idx

    # If start is in the middle of a token, move to next token boundary.
    if idx > 0 and _is_word_char(text[idx - 1]) and _is_word_char(text[idx]):
        while idx < len(text) and _is_word_char(text[idx]):
            idx += 1

    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _snap_end_boundary(text: str, end: int) -> int:
    idx = max(0, min(end, len(text)))
    if idx <= 0 or idx >= len(text):
        return idx

    # If end is in the middle of a token, extend to token end.
    if _is_word_char(text[idx - 1]) and _is_word_char(text[idx]):
        while idx < len(text) and _is_word_char(text[idx]):
            idx += 1
    return idx


def _clean_leading_fragment(text: str) -> str:
    payload = text.lstrip()
    if not payload:
        return payload

    match = re.match(r"^([^\s]+)(\s+.*)$", payload, flags=re.DOTALL)
    if not match:
        return payload

    first_token = match.group(1)
    rest = match.group(2).lstrip()
    token_alpha = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", first_token)
    token_lower = token_alpha.lower()
    if not token_lower:
        return payload

    looks_fragment = (
        len(token_lower) <= 7
        and token_lower not in LEADING_FRAGMENT_SAFE_TOKENS
        and token_alpha[:1].islower()
        and first_token.endswith((",", ".", ";", ":"))
    )
    if looks_fragment and rest:
        return rest
    return payload


def _char_fallback_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size pozitif olmalı")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap, 0 ile chunk_size arasinda olmali")

    chunks: List[str] = []
    step = chunk_size - overlap
    raw_start = 0
    payload = text.strip()
    while raw_start < len(payload):
        start = _snap_start_boundary(payload, raw_start)
        if start >= len(payload):
            break
        end = _snap_end_boundary(payload, min(len(payload), start + chunk_size))
        piece = _clean_leading_fragment(payload[start:end].strip())
        if piece:
            if chunks:
                prev = chunks[-1]
                too_small_overlap_tail = len(piece) < max(40, overlap // 2) and piece in prev
                if piece == prev or too_small_overlap_tail:
                    if end >= len(payload):
                        break
                    raw_start += step
                    continue
            chunks.append(piece)
        if end >= len(payload):
            break
        raw_start += step
    return chunks


def _is_all_caps_heading(line: str) -> bool:
    candidate = line.strip()
    if not (4 <= len(candidate) <= 80):
        return False
    if candidate != candidate.upper():
        return False

    letters = [ch for ch in candidate if ch.isalpha()]
    if not letters:
        return False

    letter_or_space = sum(1 for ch in candidate if ch.isalpha() or ch.isspace())
    ratio = letter_or_space / max(len(candidate), 1)
    return ratio >= 0.70


def _is_heading_line(line: str) -> bool:
    candidate = line.strip()
    if not candidate:
        return False
    if NUMBERED_HEADING_PATTERN.match(candidate):
        first_num_match = re.match(r"^\s*(\d+)", candidate)
        if first_num_match:
            first_num = int(first_num_match.group(1))
            if 1 <= first_num <= 12:
                return True
    if _is_all_caps_heading(candidate):
        return True

    # Keyword-based heading rule is intentionally conservative to avoid
    # classifying long narrative sentences as headings.
    if len(candidate) > 80:
        return False
    if "," in candidate or "." in candidate:
        return False
    if len(candidate.split()) > 8:
        return False

    upper_line = candidate.upper()
    return any(keyword in upper_line for keyword in HEADING_KEYWORDS)


def split_page_into_sections(page_doc: PageDocument) -> List[SectionBlock]:
    payload = page_doc.text.strip()
    if not payload:
        return []

    lines = page_doc.text.splitlines()
    saw_heading = False
    current_title = "(no heading)"
    current_lines: List[str] = []
    sections: List[SectionBlock] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped and _is_heading_line(stripped):
            saw_heading = True
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(
                    SectionBlock(
                        section_title=current_title,
                        section_text=section_text,
                        start_page=page_doc.page,
                    )
                )
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(raw_line)

    tail_text = "\n".join(current_lines).strip()
    if tail_text:
        sections.append(
            SectionBlock(
                section_title=current_title,
                section_text=tail_text,
                start_page=page_doc.page,
            )
        )
    elif saw_heading and current_title != "(no heading)":
        sections.append(
            SectionBlock(
                section_title=current_title,
                section_text=current_title,
                start_page=page_doc.page,
            )
        )

    if not saw_heading:
        return [
            SectionBlock(
                section_title="(no heading)",
                section_text=payload,
                start_page=page_doc.page,
            )
        ]
    return sections


def _split_paragraphs(section_text: str) -> List[str]:
    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_PATTERN.split(section_text) if p.strip()]
    if paragraphs:
        return paragraphs
    compact = section_text.strip()
    return [compact] if compact else []


def is_table_like_paragraph(paragraph: str) -> bool:
    numeric_tokens = NUMERIC_TOKEN_PATTERN.findall(paragraph)
    if len(numeric_tokens) < 3:
        return False

    many_spaces = len(re.findall(r"\s{2,}", paragraph)) >= 2
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    short_numeric_lines = sum(
        1 for line in lines if len(line) <= 60 and NUMERIC_TOKEN_PATTERN.search(line)
    )
    multi_short_lines = short_numeric_lines >= 3
    numeric_rich_lines = sum(1 for line in lines if len(NUMERIC_TOKEN_PATTERN.findall(line)) >= 2)
    has_year_headers = bool(re.search(r"20\d{2}", paragraph)) and numeric_rich_lines >= 2
    has_period_headers = bool(re.search(r"\b(?:q[1-4]|[123]c|[123]q|9a|6a|1yy)\b", paragraph, flags=re.IGNORECASE))
    return many_spaces or multi_short_lines or numeric_rich_lines >= 3 or (has_year_headers and has_period_headers)


def _build_paragraph_groups(paragraphs: List[str], chunk_size: int) -> List[str]:
    groups: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                groups.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            groups.extend(_char_fallback_chunks(paragraph, chunk_size=chunk_size, overlap=0))
            continue

        projected = current_len + (2 if current else 0) + len(paragraph)
        if current and projected > chunk_size:
            groups.append("\n\n".join(current).strip())
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = projected

    if current:
        groups.append("\n\n".join(current).strip())
    return [group for group in groups if group]


def _apply_overlap(groups: List[str], chunk_size: int, overlap: int) -> List[str]:
    if not groups:
        return []
    if overlap <= 0:
        return groups

    merged: List[str] = []
    previous = ""
    for group in groups:
        if previous:
            tail = previous[-overlap:]
            if tail and _is_word_char(tail[0]):
                boundary_match = re.search(r"[\s,.;:!?()\[\]\-]", tail)
                if boundary_match:
                    tail = tail[boundary_match.end() :]
            tail = tail.strip()

            if tail:
                allowed_tail = max(0, chunk_size - len(group) - 2)
                if allowed_tail < len(tail):
                    tail = tail[-allowed_tail:] if allowed_tail > 0 else ""

            candidate = f"{tail}\n\n{group}".strip() if tail else group
            if len(candidate) > chunk_size:
                candidate = candidate[:chunk_size].rstrip()
            merged.append(candidate)
        else:
            merged.append(group)
        previous = merged[-1]
    return merged


def _add_v2_chunks(
    target: List[TextChunkV2],
    chunk_texts: List[str],
    page_doc: PageDocument,
    section_title: str,
    section_idx: int,
    block_type: str,
    chunk_seq: int,
) -> int:
    for chunk_text in chunk_texts:
        cleaned = _clean_leading_fragment(chunk_text.strip())
        if not cleaned:
            continue
        if (
            len(cleaned) < 15
            and len(cleaned.split()) < 3
            and len(NUMERIC_TOKEN_PATTERN.findall(cleaned)) < 2
        ):
            continue
        chunk_id = f"{page_doc.doc_id}-p{page_doc.page}-s{section_idx}-c{chunk_seq}"
        target.append(
            TextChunkV2(
                doc_id=page_doc.doc_id,
                company=page_doc.company,
                quarter=page_doc.quarter,
                year=page_doc.year,
                page=page_doc.page,
                chunk_id=chunk_id,
                text=cleaned,
                section_title=section_title,
                block_type=block_type,
            )
        )
        chunk_seq += 1
    return chunk_seq


def chunk_documents_v2(
    pages: List[PageDocument],
    chunk_size: int = 900,
    overlap: int = 150,
) -> Tuple[List[TextChunkV2], Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size pozitif olmalı")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap, 0 ile chunk_size arasinda olmali")

    chunks: List[TextChunkV2] = []
    chunks_per_pdf: Dict[str, int] = {}
    sections_detected = 0
    table_like_count = 0

    for page in pages:
        sections = split_page_into_sections(page)
        sections_detected += len(sections)
        page_chunk_seq = 1

        for section_idx, section in enumerate(sections, start=1):
            paragraphs = _split_paragraphs(section.section_text)
            if not paragraphs:
                continue

            narrative_paragraphs: List[str] = []
            for paragraph in paragraphs:
                if is_table_like_paragraph(paragraph):
                    if narrative_paragraphs:
                        groups = _build_paragraph_groups(narrative_paragraphs, chunk_size=chunk_size)
                        text_chunks = _apply_overlap(groups, chunk_size=chunk_size, overlap=overlap)
                        page_chunk_seq = _add_v2_chunks(
                            target=chunks,
                            chunk_texts=text_chunks,
                            page_doc=page,
                            section_title=section.section_title,
                            section_idx=section_idx,
                            block_type="text",
                            chunk_seq=page_chunk_seq,
                        )
                        narrative_paragraphs = []

                    table_chunks = _char_fallback_chunks(
                        paragraph,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    table_like_count += len(table_chunks)
                    page_chunk_seq = _add_v2_chunks(
                        target=chunks,
                        chunk_texts=table_chunks,
                        page_doc=page,
                        section_title=section.section_title,
                        section_idx=section_idx,
                        block_type="table_like",
                        chunk_seq=page_chunk_seq,
                    )
                else:
                    narrative_paragraphs.append(paragraph)

            if narrative_paragraphs:
                groups = _build_paragraph_groups(narrative_paragraphs, chunk_size=chunk_size)
                text_chunks = _apply_overlap(groups, chunk_size=chunk_size, overlap=overlap)
                page_chunk_seq = _add_v2_chunks(
                    target=chunks,
                    chunk_texts=text_chunks,
                    page_doc=page,
                    section_title=section.section_title,
                    section_idx=section_idx,
                    block_type="text",
                    chunk_seq=page_chunk_seq,
                )

    for chunk in chunks:
        chunks_per_pdf[chunk.doc_id] = chunks_per_pdf.get(chunk.doc_id, 0) + 1

    stats: Dict[str, Any] = {
        "sections_detected": sections_detected,
        "table_like_count": table_like_count,
        "chunks_per_pdf": chunks_per_pdf,
    }
    return chunks, stats


def save_chunks_v2_jsonl(chunks: List[TextChunkV2], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
