from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


class E5Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        prefixed = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        vector = self.model.encode(
            [f"query: {query}"],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

