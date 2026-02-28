from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import chromadb
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "ChromaDB import edilemedi. Python 3.9-3.12 kullanin ve requirements.txt kurulumunu tekrar yapin."
    ) from exc

from src.embeddings import E5Embedder
from src.index import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_COLLECTION_NAME_V2,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
QUARTER_PATTERN = re.compile(r"^(20\d{2})?Q?([1-4])$", re.IGNORECASE)
CURRENCY_OR_PERCENT_PATTERN = re.compile(r"(?:\btl\b|\bmilyon\b|\bmilyar\b|%)", re.IGNORECASE)
DEFAULT_CHUNKS_V2_JSONL = Path("data/processed/chunks_v2.jsonl")
QUERY_SYNONYM_EXPANSIONS = {
    "ciro": ("satis", "satislar", "hasilat", "gelir", "net satislar"),
    "hasilat": ("satis", "satislar", "ciro", "gelir"),
}
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
LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "CrossEncoder import edilemedi. sentence-transformers kurulumunu kontrol edin."
        ) from exc

    return CrossEncoder(model_name_or_path=model_name)


def normalize_for_match(text: str) -> str:
    return text.lower().translate(TR_NORMALIZE_MAP)


def normalize_quarter_filter(quarter: Optional[str]) -> Optional[str]:
    if not quarter:
        return None
    candidate = quarter.strip().upper()
    matched = QUARTER_PATTERN.match(candidate)
    if not matched:
        return candidate

    year = matched.group(1) or "2025"
    quarter_num = matched.group(2)
    return f"{year}Q{quarter_num}"


def normalize_company_filter(company: Optional[str]) -> Optional[str]:
    if not company:
        return None
    candidate = str(company).strip()
    if not candidate:
        return None
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate.upper()


@dataclass
class RetrievedChunk:
    text: str
    distance: float
    score: float
    doc_id: str
    company: str
    quarter: str
    year: Optional[int]
    page: int
    chunk_id: str
    section_title: str = "(no heading)"
    block_type: str = "text"
    chunk_version: str = "v1"
    vector_score: float = 0.0
    lexical_boost: float = 0.0
    final_score: float = 0.0

    def citation(self) -> str:
        return f"[{self.doc_id}, {self.quarter}, {self.page}]"

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


class Retriever:
    def __init__(
        self,
        chroma_path: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> None:
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = E5Embedder(model_name=model_name)

    @staticmethod
    def _is_collection_not_found_error(exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        return ("notfound" in name and "collection" in message) or (
            "collection" in message and "does not exist" in message
        )

    def _refresh_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = normalize_for_match(text)
        return TOKEN_PATTERN.findall(normalized)

    def _expand_query_for_retrieval(self, query: str) -> str:
        tokens = self._tokenize(query)
        expansions: List[str] = []
        for token in tokens:
            if token.endswith("ler") or token.endswith("lar"):
                root = token[:-3]
                if len(root) >= 3:
                    expansions.append(root)
            if token in QUERY_SYNONYM_EXPANSIONS:
                expansions.extend(QUERY_SYNONYM_EXPANSIONS[token])
        if not expansions:
            return query
        unique_expansions = list(dict.fromkeys(expansions))
        expansion_text = " ".join(unique_expansions)
        return f"{query} {expansion_text}".strip()

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if abs(max_score - min_score) < 1e-9:
            return [max(0.0, min(1.0, score)) for score in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _query_collection(
        self,
        query: str,
        n_results: int,
        quarter: Optional[str] = None,
        company: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        filter_quarter = normalize_quarter_filter(quarter)
        filter_company = normalize_company_filter(company)
        apply_quarter_in_python = bool(filter_quarter and filter_company)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        # NOTE: Avoid `$and` filter path due stability issues seen on some
        # Chroma/Rust builds. When both company+quarter are requested, query by
        # company and apply quarter filtering in Python.
        if filter_company:
            query_kwargs["where"] = {"company": filter_company}
            if apply_quarter_in_python:
                query_kwargs["n_results"] = max(int(n_results), 1) * 6
        elif filter_quarter:
            query_kwargs["where"] = {"quarter": filter_quarter}

        result = None
        for attempt in range(2):
            try:
                result = self.collection.query(
                    **query_kwargs,
                )
                break
            except Exception as exc:
                if attempt == 0 and self._is_collection_not_found_error(exc):
                    LOGGER.warning(
                        "Collection handle stale; refreshing collection and retrying query. "
                        "collection=%s",
                        self.collection_name,
                    )
                    self._refresh_collection()
                    continue
                raise
        if result is None:
            return []
        documents = result.get("documents", [[]])[0]
        # Never drop an explicit company filter: cross-company fallback can return
        # a wrong firm's evidence and break groundedness.
        if filter_company and not documents:
            return []

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: List[RetrievedChunk] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            distance = float(dist)
            score = 1.0 - distance
            raw_year = meta.get("year")
            try:
                year = int(raw_year) if raw_year is not None and str(raw_year).strip() else None
            except Exception:
                year = None
            retrieved.append(
                RetrievedChunk(
                    text=str(doc),
                    distance=distance,
                    score=score,
                    doc_id=str(meta.get("doc_id", "")),
                    company=str(meta.get("company", "BIM")),
                    quarter=str(meta.get("quarter", "")),
                    year=year,
                    page=int(meta.get("page", 0)),
                    chunk_id=str(meta.get("chunk_id", "")),
                    section_title=str(meta.get("section_title", "(no heading)")),
                    block_type=str(meta.get("block_type", "text")),
                    chunk_version=str(meta.get("chunk_version", "v1")),
                    vector_score=score,
                    lexical_boost=0.0,
                    final_score=score,
                )
            )

        if apply_quarter_in_python and filter_quarter:
            retrieved = [
                row
                for row in retrieved
                if normalize_quarter_filter(str(row.quarter)) == filter_quarter
            ]
            if n_results > 0:
                retrieved = retrieved[:n_results]

        return retrieved

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        quarter: Optional[str] = None,
        company: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        results = self._query_collection(
            query=query,
            n_results=top_k,
            quarter=quarter,
            company=company,
        )
        # Fallback: if company+quarter combo yields nothing, retry without quarter
        if not results and company and quarter:
            results = self._query_collection(
                query=query,
                n_results=top_k,
                quarter=None,
                company=company,
            )
        return results

    def _compute_lexical_boost(self, query_tokens: List[str], chunk_text: str) -> float:
        chunk_norm = normalize_for_match(chunk_text)
        chunk_tokens = set(TOKEN_PATTERN.findall(chunk_norm))

        token_matches = sum(1 for token in query_tokens if token in chunk_tokens)
        boost = float(min(token_matches, 8))

        if "kar" in query_tokens and "net kar" in chunk_norm:
            boost += 2.0

        query_has_favok = any(token in {"favok", "fvaok"} for token in query_tokens)
        if query_has_favok and ("favok" in chunk_norm or "fvaok" in chunk_norm):
            boost += 2.0

        query_has_risk = any(token.startswith("risk") for token in query_tokens)
        if query_has_risk and "risk" in chunk_norm:
            boost += 1.5

        return boost

    def retrieve_with_boost(
        self,
        query: str,
        top_k_initial: int = 15,
        top_k_final: int = 5,
        alpha: float = 0.35,
        quarter: Optional[str] = None,
        company: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        expanded_query = self._expand_query_for_retrieval(query)
        candidates = self._query_collection(
            query=expanded_query,
            n_results=top_k_initial,
            quarter=quarter,
            company=company,
        )
        # Fallback: if company+quarter combo yields nothing, retry without quarter
        if not candidates and company and quarter:
            candidates = self._query_collection(
                query=expanded_query,
                n_results=top_k_initial,
                quarter=None,
                company=company,
            )
        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        normalized_vector_scores = self._normalize_scores([item.score for item in candidates])

        reranked: List[RetrievedChunk] = []
        for item, normalized_vector in zip(candidates, normalized_vector_scores):
            lexical_boost = self._compute_lexical_boost(query_tokens=query_tokens, chunk_text=item.text)
            item.vector_score = normalized_vector
            item.lexical_boost = lexical_boost
            item.final_score = normalized_vector + (alpha * lexical_boost)
            reranked.append(item)

        reranked.sort(key=lambda row: row.final_score, reverse=True)
        return reranked[:top_k_final]


class RetrieverV2(Retriever):
    def __init__(
        self,
        chroma_path: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME_V2,
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> None:
        super().__init__(
            chroma_path=chroma_path,
            collection_name=collection_name,
            model_name=model_name,
        )


class RetrieverV3(RetrieverV2):
    QUAL_SECTION_KEYWORDS = (
        "risk",
        "strateji",
        "beklenti",
        "gorunum",
        "degerlend",
        "yonetim",
        "finansman",
        "surdurulebilirlik",
    )
    KPI_TEXT_KEYWORDS = (
        "magaza",
        "personel",
        "calisan",
        "online",
        "eticaret",
        "e ticaret",
        "adet",
    )
    TREND_TEXT_KEYWORDS = (
        "degisim",
        "artis",
        "azalis",
        "artti",
        "azaldi",
        "marj",
    )
    NUMERIC_QUERY_HINTS = (
        "kac",
        "tutar",
        "milyar",
        "milyon",
        "tl",
        "marj",
        "%",
        "ne kadar",
    )
    QUAL_QUERY_HINTS = (
        "risk",
        "strateji",
        "beklenti",
        "gorunum",
        "degerlendirme",
        "yaklasim",
    )

    @staticmethod
    def _contains_any(text: str, keywords: tuple) -> bool:
        return any(keyword in text for keyword in keywords)

    def _query_type_boost(self, query_type: str, question_norm: str, item: RetrievedChunk) -> float:
        text_norm = normalize_for_match(item.text)
        section_norm = normalize_for_match(item.section_title)
        boost = 0.0

        is_numeric_query = query_type == "numeric" or (
            query_type == "other"
            and self._contains_any(
                question_norm,
                self.NUMERIC_QUERY_HINTS,
            )
        )
        is_qualitative_query = query_type == "qualitative" or self._contains_any(
            question_norm,
            self.QUAL_QUERY_HINTS,
        )

        if is_numeric_query:
            if item.block_type == "table_like":
                boost += 2.2
            if CURRENCY_OR_PERCENT_PATTERN.search(text_norm):
                boost += 1.3

        if query_type == "trend":
            if self._contains_any(text_norm, self.TREND_TEXT_KEYWORDS):
                boost += 1.6
            if item.block_type == "table_like":
                boost += 0.8

        if is_qualitative_query:
            if self._contains_any(section_norm, self.QUAL_SECTION_KEYWORDS):
                boost += 2.0
            if self._contains_any(text_norm, self.QUAL_QUERY_HINTS):
                boost += 1.2

        if query_type == "kpi":
            if self._contains_any(text_norm, self.KPI_TEXT_KEYWORDS):
                boost += 1.6
            if self._contains_any(section_norm, self.KPI_TEXT_KEYWORDS):
                boost += 2.2
            if item.block_type == "table_like":
                boost += 1.2
            if "magaza" in question_norm and ("magaza" in text_norm or "magaza" in section_norm):
                boost += 2.8
            if "calisan" in question_norm and ("calisan" in text_norm or "personel" in text_norm):
                boost += 2.4
            if ("online" in question_norm or "eticaret" in question_norm) and (
                "online" in text_norm or "eticaret" in text_norm or "e ticaret" in text_norm
            ):
                boost += 2.4

        return boost

    def retrieve_with_query_awareness(
        self,
        query: str,
        top_k_initial: int = 20,
        top_k_final: int = 5,
        alpha: float = 0.35,
        quarter_override: Optional[str] = None,
        company_override: Optional[str] = None,
        allow_quarter_fallback: bool = True,
    ) -> List[RetrievedChunk]:
        from src.query_parser import parse_query

        parsed = parse_query(query)
        auto_quarter = parsed.get("quarter")
        quarter = quarter_override or auto_quarter
        company = company_override
        question_norm = normalize_for_match(query)
        query_type = str(parsed.get("signals", {}).get("query_type", "qualitative"))

        expanded_query = self._expand_query_for_retrieval(query)
        candidates = self._query_collection(
            query=expanded_query,
            n_results=top_k_initial,
            quarter=quarter,
            company=company,
        )
        # If company filter is set but quarter+company combo yields nothing,
        # retry with only the company filter (handles docs with quarter=UNKNOWN).
        if allow_quarter_fallback and not candidates and company and quarter:
            LOGGER.info(
                "Quarter+company filter returned no results; retrying without quarter filter "
                "(company=%s, quarter=%s)",
                company,
                quarter,
            )
            candidates = self._query_collection(
                query=expanded_query,
                n_results=top_k_initial,
                quarter=None,
                company=company,
            )
        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        normalized_vector_scores = self._normalize_scores([item.score for item in candidates])

        reranked: List[RetrievedChunk] = []
        for item, normalized_vector in zip(candidates, normalized_vector_scores):
            lexical_boost = self._compute_lexical_boost(
                query_tokens=query_tokens,
                chunk_text=item.text,
            )
            lexical_boost += self._query_type_boost(
                query_type=query_type,
                question_norm=question_norm,
                item=item,
            )
            item.vector_score = normalized_vector
            item.lexical_boost = lexical_boost
            item.final_score = normalized_vector + (alpha * lexical_boost)
            reranked.append(item)

        reranked.sort(key=lambda row: row.final_score, reverse=True)
        return reranked[:top_k_final]


class RetrieverBM25:
    def __init__(
        self,
        chunks_file: Path = DEFAULT_CHUNKS_V2_JSONL,
        auto_quarter_filter: bool = True,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "rank-bm25 import edilemedi. requirements.txt kurulumunu kontrol edin."
            ) from exc

        self.chunks_file = Path(chunks_file)
        if not self.chunks_file.exists():
            raise FileNotFoundError(f"BM25 corpus dosyasi bulunamadi: {self.chunks_file}")

        self.auto_quarter_filter = auto_quarter_filter
        self._rows = self._load_rows(self.chunks_file)
        self._corpus_tokens = [self._tokenize_for_bm25(row["text"], row["section_title"]) for row in self._rows]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    @staticmethod
    def _load_rows(path: Path) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                raw_year = payload.get("year")
                try:
                    year = int(raw_year) if raw_year is not None and str(raw_year).strip() else None
                except Exception:
                    year = None
                rows.append(
                    {
                        "doc_id": str(payload.get("doc_id", "")),
                        "company": str(payload.get("company", "BIM")),
                        "quarter": str(payload.get("quarter", "")),
                        "year": year,
                        "page": int(payload.get("page", 0)),
                        "chunk_id": str(payload.get("chunk_id", "")),
                        "text": str(payload.get("text", "")),
                        "section_title": str(payload.get("section_title", "(no heading)")),
                        "block_type": str(payload.get("block_type", "text")),
                        "chunk_version": str(payload.get("chunk_version", "v2")),
                    }
                )
        return rows

    @staticmethod
    def _tokenize_for_bm25(text: str, section_title: str = "") -> List[str]:
        merged = f"{section_title} {text}".strip()
        normalized = normalize_for_match(merged)
        return TOKEN_PATTERN.findall(normalized)

    @staticmethod
    def _to_retrieved_chunk(row: Dict[str, object], raw_score: float, normalized_score: float) -> RetrievedChunk:
        raw_year = row.get("year")
        try:
            year = int(raw_year) if raw_year is not None and str(raw_year).strip() else None
        except Exception:
            year = None
        return RetrievedChunk(
            text=str(row["text"]),
            distance=float(max(0.0, 1.0 - normalized_score)),
            score=float(raw_score),
            doc_id=str(row["doc_id"]),
            company=str(row.get("company", "BIM")),
            quarter=str(row["quarter"]),
            year=year,
            page=int(row["page"]),
            chunk_id=str(row["chunk_id"]),
            section_title=str(row["section_title"]),
            block_type=str(row["block_type"]),
            chunk_version=str(row["chunk_version"]),
            vector_score=0.0,
            lexical_boost=float(normalized_score),
            final_score=float(normalized_score),
        )

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if abs(max_score - min_score) < 1e-9:
            if max_score <= 0:
                return [0.0 for _ in scores]
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        quarter: Optional[str] = None,
        company: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        from src.query_parser import parse_query

        candidate_quarter = quarter
        if self.auto_quarter_filter and not candidate_quarter:
            parsed = parse_query(query)
            candidate_quarter = parsed.get("quarter")
        quarter_filter = normalize_quarter_filter(candidate_quarter)
        company_filter = normalize_company_filter(company)

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scored_rows: List[Tuple[float, Dict[str, object]]] = []
        for idx, score in enumerate(raw_scores):
            row = self._rows[idx]
            if quarter_filter and normalize_quarter_filter(str(row.get("quarter", ""))) != quarter_filter:
                continue
            if company_filter and normalize_company_filter(str(row.get("company", ""))) != company_filter:
                continue
            scored_rows.append((float(score), row))

        if not scored_rows:
            return []

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        top_rows = scored_rows[:top_k]
        normalized = self._normalize_scores([score for score, _ in top_rows])
        results = [
            self._to_retrieved_chunk(row=row, raw_score=score, normalized_score=norm_score)
            for (score, row), norm_score in zip(top_rows, normalized)
        ]
        return results


class RetrieverV5Hybrid(RetrieverV3):
    def __init__(
        self,
        chroma_path: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME_V2,
        model_name: str = "intfloat/multilingual-e5-small",
        chunks_file: Path = DEFAULT_CHUNKS_V2_JSONL,
    ) -> None:
        super().__init__(
            chroma_path=chroma_path,
            collection_name=collection_name,
            model_name=model_name,
        )
        self.bm25 = RetrieverBM25(chunks_file=chunks_file, auto_quarter_filter=False)

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if abs(max_score - min_score) < 1e-9:
            return [1.0 if score > 0 else 0.0 for score in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def retrieve_with_hybrid(
        self,
        query: str,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        top_k_final: int = 5,
        beta: float = 0.6,
        alpha_v3: float = 0.35,
        quarter_override: Optional[str] = None,
        company_override: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        from src.query_parser import parse_query

        parsed = parse_query(query)
        quarter = quarter_override or parsed.get("quarter")
        company = company_override

        vector_results = self.retrieve_with_query_awareness(
            query=query,
            top_k_initial=top_k_vector,
            top_k_final=top_k_vector,
            alpha=alpha_v3,
            quarter_override=quarter,
            company_override=company,
        )
        bm25_results = self.bm25.retrieve(
            query=query,
            top_k=top_k_bm25,
            quarter=quarter,
            company=company,
        )

        merged: Dict[str, RetrievedChunk] = {}
        vector_score_map: Dict[str, float] = {}
        bm25_score_map: Dict[str, float] = {}

        for row in vector_results:
            merged[row.chunk_id] = row
            vector_score_map[row.chunk_id] = float(row.vector_score)

        for row in bm25_results:
            if row.chunk_id not in merged:
                merged[row.chunk_id] = row
            bm25_score_map[row.chunk_id] = float(row.final_score)

        vector_norm = self._normalize_scores([vector_score_map.get(chunk_id, 0.0) for chunk_id in merged.keys()])
        bm25_norm = self._normalize_scores([bm25_score_map.get(chunk_id, 0.0) for chunk_id in merged.keys()])

        reranked: List[RetrievedChunk] = []
        for idx, chunk_id in enumerate(merged.keys()):
            item = merged[chunk_id]
            vec_score = vector_norm[idx]
            bm25_score = bm25_norm[idx]
            item.vector_score = vec_score
            item.lexical_boost = bm25_score
            item.final_score = vec_score + (beta * bm25_score)
            item.distance = max(0.0, 1.0 - item.final_score)
            reranked.append(item)

        reranked.sort(key=lambda row: row.final_score, reverse=True)
        return reranked[:top_k_final]


class RetrieverV6Cross(RetrieverV5Hybrid):
    def __init__(
        self,
        chroma_path: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME_V2,
        model_name: str = "intfloat/multilingual-e5-small",
        chunks_file: Path = DEFAULT_CHUNKS_V2_JSONL,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        super().__init__(
            chroma_path=chroma_path,
            collection_name=collection_name,
            model_name=model_name,
            chunks_file=chunks_file,
        )
        self.cross_encoder_model = cross_encoder_model

    def retrieve_with_cross_encoder(
        self,
        query: str,
        top_k_candidates: int = 15,
        top_k_final: int = 5,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
        beta: float = 0.6,
        alpha_v3: float = 0.35,
        quarter_override: Optional[str] = None,
        company_override: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        candidates = self.retrieve_with_hybrid(
            query=query,
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
            top_k_final=top_k_candidates,
            beta=beta,
            alpha_v3=alpha_v3,
            quarter_override=quarter_override,
            company_override=company_override,
        )
        if not candidates:
            return []

        cross_encoder = _get_cross_encoder(self.cross_encoder_model)
        pairs = [(query, item.text) for item in candidates]
        raw_scores = list(cross_encoder.predict(pairs, show_progress_bar=False))
        normalized_scores = self._normalize_scores([float(score) for score in raw_scores])

        reranked: List[RetrievedChunk] = []
        for item, raw, norm in zip(candidates, raw_scores, normalized_scores):
            item.score = float(raw)
            item.final_score = float(norm)
            item.distance = max(0.0, 1.0 - item.final_score)
            reranked.append(item)

        reranked.sort(key=lambda row: row.final_score, reverse=True)
        return reranked[:top_k_final]
