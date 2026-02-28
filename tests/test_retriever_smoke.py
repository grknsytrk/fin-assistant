from pathlib import Path

import pytest

from src.config import load_config
from src.retrieve import RetrieverV3


def test_retriever_smoke() -> None:
    cfg = load_config(Path("config.yaml"))
    sqlite_file = cfg.chroma.dir / "chroma.sqlite3"
    if not cfg.chroma.dir.exists() or not sqlite_file.exists():
        pytest.skip("Chroma index bulunamadi; smoke testi atlandi.")

    retriever = RetrieverV3(
        chroma_path=cfg.chroma.dir,
        collection_name=cfg.chroma.collection_v2,
        model_name=cfg.models.embedding,
    )
    rows = retriever.retrieve_with_query_awareness(
        query="2025 ucuncu ceyrek net kar kac?",
        top_k_initial=5,
        top_k_final=1,
        alpha=cfg.retrieval.alpha_v3,
    )
    assert isinstance(rows, list)
