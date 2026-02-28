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

        numeric_candidates = self._extract_numbers(chunks)
        summary_lines = [
            f"- Soru: {question}",
            "- Yanıt: İlgili içerik aşağıdaki kanıtlarda bulundu.",
            (
                f"- Sayısal adaylar: {', '.join(numeric_candidates)}"
                if numeric_candidates
                else "- Sayısal adaylar: Belirgin değer ayıklanamadı."
            ),
        ]

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
