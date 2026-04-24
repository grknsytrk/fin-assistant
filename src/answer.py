from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    from src.retrieve import RetrievedChunk

NUMBER_PATTERN = re.compile(r"\d[\d\.,]*")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
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
QUESTION_STOPWORDS = {
    "ve",
    "ile",
    "mi",
    "mı",
    "mu",
    "mü",
    "ne",
    "nedir",
    "neler",
    "nasil",
    "hangi",
    "kac",
    "kaç",
    "kadar",
    "yuzde",
    "yüzde",
    "gore",
    "göre",
    "ilk",
    "ceyrek",
    "yarıyıl",
    "yariyil",
    "aylik",
    "yil",
}


class AnswerAdapter(ABC):
    @abstractmethod
    def generate(self, question: str, chunks: Sequence["RetrievedChunk"]) -> str:
        raise NotImplementedError


class LocalLLMAdapter(AnswerAdapter):
    def __init__(self, llm_client: object) -> None:
        self.llm_client = llm_client

    def generate(self, question: str, chunks: Sequence["RetrievedChunk"]) -> str:
        raise NotImplementedError(
            "LocalLLMAdapter henüz yapılandırılmadı. RulesBasedAnswerAdapter kullanılmalı."
        )


class RulesBasedAnswerAdapter(AnswerAdapter):
    def __init__(self, max_distance: float = 0.45, min_keyword_coverage: float = 0.3) -> None:
        self.max_distance = max_distance
        self.min_keyword_coverage = min_keyword_coverage

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        normalized = unicodedata.normalize("NFKD", lowered)
        without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized_tr = without_marks.translate(TR_NORMALIZE_MAP)
        normalized_tr = re.sub(r"[^\w\s%]", " ", normalized_tr)
        normalized_tr = re.sub(r"\s+", " ", normalized_tr).strip()
        return normalized_tr

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(cls._normalize_text(text))

    @classmethod
    def _question_keywords(cls, question: str) -> List[str]:
        keywords: List[str] = []
        for token in cls._tokenize(question):
            if token in QUESTION_STOPWORDS:
                continue
            if token.isdigit():
                continue
            if len(token) < 3:
                continue
            keywords.append(token)
        return keywords

    @classmethod
    def _chunk_keyword_set(cls, chunks: Sequence["RetrievedChunk"]) -> set:
        corpus_tokens = set()
        for chunk in chunks:
            corpus_tokens.update(cls._tokenize(chunk.text))
            corpus_tokens.update(cls._tokenize(chunk.section_title))
        return corpus_tokens

    @staticmethod
    def _format_citation(chunk: "RetrievedChunk") -> str:
        section_title = (chunk.section_title or "(no heading)").strip()
        return f"[{chunk.doc_id}, {chunk.quarter}, {chunk.page}, {section_title}]"

    @staticmethod
    def _extract_numbers(chunks: Sequence["RetrievedChunk"], max_items: int = 12) -> List[str]:
        numbers: List[str] = []
        seen = set()
        for chunk in chunks:
            for match in NUMBER_PATTERN.findall(chunk.text):
                if match not in seen:
                    seen.add(match)
                    numbers.append(match)
                if len(numbers) >= max_items:
                    return numbers
        return numbers

    @staticmethod
    def _extract_quote(text: str, max_chars: int = 220) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        quote = " ".join(lines[:2])
        if len(quote) > max_chars:
            return quote[: max_chars - 3].rstrip() + "..."
        return quote

    def _is_found(self, question: str, chunks: Sequence["RetrievedChunk"]) -> bool:
        if not chunks:
            return False
        best_distance = min(chunk.distance for chunk in chunks)
        if best_distance > self.max_distance:
            return False

        question_keywords = self._question_keywords(question)
        if not question_keywords:
            return True

        corpus_tokens = self._chunk_keyword_set(chunks)
        overlap_count = sum(1 for token in question_keywords if token in corpus_tokens)
        if overlap_count == 0:
            return False

        coverage = overlap_count / len(question_keywords)
        if len(question_keywords) >= 3 and overlap_count < 2 and coverage < self.min_keyword_coverage:
            return False
        return True

    @classmethod
    def _find_best_answer_line(cls, question: str, chunks: Sequence["RetrievedChunk"]) -> Optional[str]:
        """Find the line in the most relevant chunk that best matches the question keywords and contains a number."""
        q_keywords = cls._question_keywords(question)
        if not q_keywords:
            return None

        best_line: Optional[str] = None
        best_score = -1

        for chunk in chunks:
            lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
            for line in lines:
                line_tokens = set(cls._tokenize(line))
                # Count how many question keywords appear in this line
                overlap = sum(1 for kw in q_keywords if kw in line_tokens)
                if overlap == 0:
                    continue
                # Line must contain at least one number
                numbers = NUMBER_PATTERN.findall(line)
                if not numbers:
                    continue
                # Prefer lines with more keyword overlap
                score = overlap * 10 + len(numbers)
                if score > best_score:
                    best_score = score
                    best_line = line

        return best_line

    @classmethod
    def _extract_direct_answer(cls, question: str, chunks: Sequence["RetrievedChunk"]) -> Optional[str]:
        """Try to extract a direct 'Label: Value' answer from the best matching line."""
        best_line = cls._find_best_answer_line(question, chunks)
        if not best_line:
            return None

        # Try to find the most relevant number on the line
        numbers = NUMBER_PATTERN.findall(best_line)
        if not numbers:
            return None

        # Build a clean label from the question
        q_keywords = cls._question_keywords(question)
        label_parts = []
        for word in question.split():
            normalized = cls._normalize_text(word)
            tokens = TOKEN_PATTERN.findall(normalized)
            if tokens and tokens[0] not in QUESTION_STOPWORDS and len(tokens[0]) >= 2:
                label_parts.append(word)
        label = " ".join(label_parts).strip().title() if label_parts else question.strip().title()

        # Pick the most significant number (longest, most likely to be a real value)
        best_number = max(numbers, key=lambda n: len(n.replace(".", "").replace(",", "")))

        return f"{label}: {best_number}"

    def generate(self, question: str, chunks: Sequence["RetrievedChunk"]) -> str:
        searched_pages = []
        seen_pages = set()
        for chunk in chunks:
            citation = self._format_citation(chunk)
            if citation not in seen_pages:
                seen_pages.add(citation)
                searched_pages.append(citation)

        if not self._is_found(question=question, chunks=chunks):
            lines = [
                "- Dokümanda bulunamadı.",
                f"- Aranan sayfalar: {', '.join(searched_pages) if searched_pages else 'Yok'}",
                "",
                "Evidence",
                "- Uygun kanıt bulunamadı.",
            ]
            return "\n".join(lines)

        direct_answer = self._extract_direct_answer(question, chunks)
        numeric_candidates = self._extract_numbers(chunks)

        summary_lines = []
        if direct_answer:
            summary_lines.append(f"- {direct_answer}")
        else:
            summary_lines.append("- Yanıt: İlgili içerik aşağıdaki kanıtlarda bulundu.")
        if numeric_candidates:
            summary_lines.append(f"- Sayısal adaylar: {', '.join(numeric_candidates)}")

        evidence_lines = ["", "Evidence"]
        for chunk in chunks:
            quote = self._extract_quote(chunk.text)
            evidence_lines.append(
                f'- {self._format_citation(chunk)} "{quote}"'
            )

        return "\n".join(summary_lines + evidence_lines)


class AnswerEngine:
    def __init__(self, adapter: Optional[AnswerAdapter] = None) -> None:
        self.adapter = adapter or RulesBasedAnswerAdapter()

    def answer(self, question: str, chunks: Sequence["RetrievedChunk"]) -> str:
        return self.adapter.generate(question=question, chunks=chunks)
