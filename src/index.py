from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

try:
    import chromadb
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "ChromaDB import edilemedi. Python 3.9-3.12 kullanin ve requirements.txt kurulumunu tekrar yapin."
    ) from exc

from src.chunking import (
    TextChunk,
    TextChunkV2,
    chunk_documents,
    chunk_documents_v2,
    save_chunks_jsonl,
    save_chunks_v2_jsonl,
)
from src.embeddings import E5Embedder
from src.ingest import (
    extract_pdf_pages,
    ingest_raw_pdfs,
    list_pdf_files,
    load_pages_jsonl,
    save_pages_jsonl,
    summarize_pages_for_ingest,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "bimas_faaliyet_2025"
DEFAULT_COLLECTION_NAME_V2 = "bimas_faaliyet_2025_v2"
DEFAULT_CHROMA_DIR = Path("data/processed/chroma")
DEFAULT_PAGES_JSONL = Path("data/processed/pages.jsonl")
DEFAULT_CHUNKS_JSONL = Path("data/processed/chunks.jsonl")
DEFAULT_CHUNKS_V2_JSONL = Path("data/processed/chunks_v2.jsonl")
DEFAULT_INDEX_MANIFEST_V2 = Path("data/processed/index_manifest_v2.json")


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be scalar (str/int/float/bool)."""
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            sanitized[key] = value
            continue
        if isinstance(value, int):
            sanitized[key] = value
            continue
        if isinstance(value, float):
            if math.isfinite(value):
                sanitized[key] = value
            continue
        if isinstance(value, str):
            sanitized[key] = value
            continue
        sanitized[key] = str(value)
    return sanitized


def _clear_collection(collection) -> int:
    existing_count = collection.count()
    if existing_count == 0:
        return 0
    existing = collection.get(limit=existing_count, include=[])
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    return len(existing_ids)


def _collect_collection_ids_for_doc_ids(collection, doc_ids: Set[str]) -> List[str]:
    if not doc_ids:
        return []
    existing_count = collection.count()
    if existing_count == 0:
        return []
    existing = collection.get(limit=existing_count, include=["metadatas"])
    existing_ids = existing.get("ids", [])
    metadatas = existing.get("metadatas", [])
    deletable: List[str] = []
    for chunk_id, metadata in zip(existing_ids, metadatas):
        meta = metadata or {}
        if str(meta.get("doc_id", "")) in doc_ids:
            deletable.append(str(chunk_id))
    return deletable


def _index_chunks(collection, chunks: List[Any], embedder: E5Embedder, batch_size: int = 64) -> int:
    indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [chunk.chunk_id for chunk in batch]
        documents = [chunk.text for chunk in batch]
        metadatas = [_sanitize_metadata(chunk.metadata()) for chunk in batch]
        embeddings = embedder.embed_documents(documents)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        indexed += len(batch)
        LOGGER.info("Indexing progress: %d/%d", indexed, len(chunks))
    return indexed


def _load_chunks_v2_jsonl(path: Path) -> List[TextChunkV2]:
    if not path.exists():
        return []
    chunks: List[TextChunkV2] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            raw_year = raw.get("year")
            try:
                year = int(raw_year) if raw_year is not None and str(raw_year).strip() else None
            except Exception:
                year = None
            chunks.append(
                TextChunkV2(
                    doc_id=str(raw.get("doc_id", "")),
                    company=str(raw.get("company", "")),
                    quarter=str(raw.get("quarter", "")),
                    year=year,
                    page=int(raw.get("page", 0)),
                    chunk_id=str(raw.get("chunk_id", "")),
                    text=str(raw.get("text", "")),
                    section_title=str(raw.get("section_title", "(no heading)")),
                    block_type=str(raw.get("block_type", "text")),
                    chunk_version=str(raw.get("chunk_version", "v2")),
                )
            )
    return chunks


def _file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "doc_id": path.stem,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"files": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"files": {}}
    files = payload.get("files", {})
    if not isinstance(files, dict):
        files = {}
    return {"files": files}


def _save_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pages_per_pdf_from_pages(pages: Iterable[Any], pdf_files: List[Path]) -> Dict[str, int]:
    counts_by_doc: Dict[str, int] = {}
    for page in pages:
        doc_id = str(getattr(page, "doc_id", ""))
        if not doc_id:
            continue
        counts_by_doc[doc_id] = counts_by_doc.get(doc_id, 0) + 1
    rows: Dict[str, int] = {}
    for pdf in pdf_files:
        rows[pdf.name] = int(counts_by_doc.get(pdf.stem, 0))
    return rows


def build_index(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 900,
    overlap: int = 150,
) -> Dict[str, object]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pages_output = processed_dir / DEFAULT_PAGES_JSONL.name
    chunks_output = processed_dir / DEFAULT_CHUNKS_JSONL.name
    chroma_dir = processed_dir / DEFAULT_CHROMA_DIR.name

    pages, ingest_summary = ingest_raw_pdfs(raw_dir=raw_dir, output_file=pages_output)
    chunks = chunk_documents(pages, chunk_size=chunk_size, overlap=overlap)
    save_chunks_jsonl(chunks, chunks_output)

    embedder = E5Embedder()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    deleted = _clear_collection(collection)
    if deleted:
        LOGGER.info("Collection temizlendi. Silinen kayıt: %d", deleted)

    indexed_count = _index_chunks(collection=collection, chunks=chunks, embedder=embedder)
    final_count = collection.count()

    summary: Dict[str, object] = {
        "num_pdfs": ingest_summary["num_pdfs"],
        "pdf_files": ingest_summary["pdf_files"],
        "pages_per_pdf": ingest_summary["pages_per_pdf"],
        "total_pages": ingest_summary["total_pages"],
        "total_chunks": len(chunks),
        "indexed_chunks": indexed_count,
        "collection_count": final_count,
        "indexing_success": final_count == len(chunks),
        "collection_name": collection_name,
        "chroma_path": str(chroma_dir),
        "pages_output": str(pages_output),
        "chunks_output": str(chunks_output),
    }
    return summary


def build_index_v2_incremental(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    collection_name: str = DEFAULT_COLLECTION_NAME_V2,
    chunk_size: int = 900,
    overlap: int = 150,
) -> Dict[str, object]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pages_output = processed_dir / DEFAULT_PAGES_JSONL.name
    chunks_output = processed_dir / DEFAULT_CHUNKS_V2_JSONL.name
    chroma_dir = processed_dir / DEFAULT_CHROMA_DIR.name
    manifest_output = processed_dir / DEFAULT_INDEX_MANIFEST_V2.name

    pdf_files = list_pdf_files(raw_dir)
    current_signatures: Dict[str, Dict[str, Any]] = {
        pdf.name: _file_signature(pdf) for pdf in pdf_files
    }

    previous_manifest = _load_manifest(manifest_output)
    previous_files: Dict[str, Dict[str, Any]] = dict(previous_manifest.get("files", {}))

    changed_files: List[str] = []
    unchanged_files: List[str] = []
    for filename, signature in current_signatures.items():
        previous = previous_files.get(filename)
        if previous == signature:
            unchanged_files.append(filename)
        else:
            changed_files.append(filename)
    removed_files = sorted(set(previous_files.keys()) - set(current_signatures.keys()))

    changed_doc_ids = {current_signatures[name]["doc_id"] for name in changed_files}
    removed_doc_ids = {
        str(previous_files.get(name, {}).get("doc_id", Path(name).stem)) for name in removed_files
    }
    affected_doc_ids = set(changed_doc_ids) | set(removed_doc_ids)

    changed_pages: List[Any] = []
    for pdf in pdf_files:
        if pdf.name not in changed_files:
            continue
        changed_pages.extend(extract_pdf_pages(pdf))

    existing_pages = load_pages_jsonl(pages_output) if pages_output.exists() else []
    kept_pages = [page for page in existing_pages if str(getattr(page, "doc_id", "")) not in affected_doc_ids]
    all_pages = kept_pages + changed_pages
    save_pages_jsonl(all_pages, pages_output)

    changed_chunks_v2, changed_chunk_stats = chunk_documents_v2(
        changed_pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    existing_chunks_v2 = _load_chunks_v2_jsonl(chunks_output) if chunks_output.exists() else []
    kept_chunks_v2 = [
        chunk for chunk in existing_chunks_v2 if str(getattr(chunk, "doc_id", "")) not in affected_doc_ids
    ]
    all_chunks_v2 = kept_chunks_v2 + changed_chunks_v2
    save_chunks_v2_jsonl(all_chunks_v2, chunks_output)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection_count_before = collection.count()
    bootstrap_full = False
    if collection_count_before == 0 and all_chunks_v2 and not changed_chunks_v2:
        # Collection was dropped externally while local pages/chunks files still exist.
        # Bootstrap from local chunks to keep incremental mode resilient.
        bootstrap_full = True
        changed_chunks_v2 = list(all_chunks_v2)
        changed_files = [pdf.name for pdf in pdf_files]
        affected_doc_ids = {str(getattr(chunk, "doc_id", "")) for chunk in all_chunks_v2}
        LOGGER.info(
            "V2 incremental bootstrap: empty collection detected, re-indexing all local chunks (%d).",
            len(changed_chunks_v2),
        )

    deleted_ids = _collect_collection_ids_for_doc_ids(collection=collection, doc_ids=affected_doc_ids)
    if deleted_ids:
        collection.delete(ids=deleted_ids)
        LOGGER.info("V2 incremental: silinen chunk: %d", len(deleted_ids))

    indexed_count = 0
    if changed_chunks_v2:
        embedder = E5Embedder()
        indexed_count = _index_chunks(collection=collection, chunks=changed_chunks_v2, embedder=embedder)
    final_count = collection.count()

    _save_manifest(manifest_output, {"files": current_signatures})
    ingest_summary = summarize_pages_for_ingest(all_pages)
    pages_per_pdf = _pages_per_pdf_from_pages(all_pages, pdf_files)

    summary: Dict[str, object] = {
        "mode": "incremental",
        "bootstrap_full": bootstrap_full,
        "num_pdfs": len(pdf_files),
        "pdf_files": [p.name for p in pdf_files],
        "pages_per_pdf": pages_per_pdf,
        "total_pages": len(all_pages),
        "total_chunks": len(all_chunks_v2),
        "indexed_chunks": indexed_count,
        "collection_count": final_count,
        "indexing_success": final_count == len(all_chunks_v2),
        "collection_name": collection_name,
        "chroma_path": str(chroma_dir),
        "pages_output": str(pages_output),
        "chunks_output": str(chunks_output),
        "sections_detected": changed_chunk_stats["sections_detected"],
        "table_like_count": changed_chunk_stats["table_like_count"],
        "chunks_per_pdf": changed_chunk_stats["chunks_per_pdf"],
        "companies": ingest_summary["companies"],
        "changed_files": sorted(changed_files),
        "unchanged_files": sorted(unchanged_files),
        "removed_files": sorted(removed_files),
        "affected_doc_ids": sorted(affected_doc_ids),
        "deleted_chunks": len(deleted_ids),
    }
    return summary


def build_index_v2(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    collection_name: str = DEFAULT_COLLECTION_NAME_V2,
    chunk_size: int = 900,
    overlap: int = 150,
) -> Dict[str, object]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pages_output = processed_dir / DEFAULT_PAGES_JSONL.name
    chunks_output = processed_dir / DEFAULT_CHUNKS_V2_JSONL.name
    chroma_dir = processed_dir / DEFAULT_CHROMA_DIR.name

    pages, ingest_summary = ingest_raw_pdfs(raw_dir=raw_dir, output_file=pages_output)
    chunks_v2, chunking_stats = chunk_documents_v2(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    save_chunks_v2_jsonl(chunks_v2, chunks_output)

    embedder = E5Embedder()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    deleted = _clear_collection(collection)
    if deleted:
        LOGGER.info("V2 collection temizlendi. Silinen kayıt: %d", deleted)

    indexed_count = _index_chunks(collection=collection, chunks=chunks_v2, embedder=embedder)
    final_count = collection.count()

    summary: Dict[str, object] = {
        "num_pdfs": ingest_summary["num_pdfs"],
        "pdf_files": ingest_summary["pdf_files"],
        "pages_per_pdf": ingest_summary["pages_per_pdf"],
        "total_pages": ingest_summary["total_pages"],
        "total_chunks": len(chunks_v2),
        "indexed_chunks": indexed_count,
        "collection_count": final_count,
        "indexing_success": final_count == len(chunks_v2),
        "collection_name": collection_name,
        "chroma_path": str(chroma_dir),
        "pages_output": str(pages_output),
        "chunks_output": str(chunks_output),
        "sections_detected": chunking_stats["sections_detected"],
        "table_like_count": chunking_stats["table_like_count"],
        "chunks_per_pdf": chunking_stats["chunks_per_pdf"],
    }
    return summary


def build_index_v2_from_pages(
    pages_file: Path,
    processed_dir: Path = Path("data/processed"),
    collection_name: str = DEFAULT_COLLECTION_NAME_V2,
    chunk_size: int = 900,
    overlap: int = 150,
) -> Dict[str, object]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_output = processed_dir / DEFAULT_CHUNKS_V2_JSONL.name
    chroma_dir = processed_dir / DEFAULT_CHROMA_DIR.name

    pages = load_pages_jsonl(pages_file)
    ingest_summary = summarize_pages_for_ingest(pages)

    chunks_v2, chunking_stats = chunk_documents_v2(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    save_chunks_v2_jsonl(chunks_v2, chunks_output)

    embedder = E5Embedder()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    deleted = _clear_collection(collection)
    if deleted:
        LOGGER.info("V2 collection temizlendi. Silinen kayıt: %d", deleted)

    indexed_count = _index_chunks(collection=collection, chunks=chunks_v2, embedder=embedder)
    final_count = collection.count()

    summary: Dict[str, object] = {
        "num_pdfs": ingest_summary["num_pdfs"],
        "pdf_files": ingest_summary["pdf_files"],
        "pages_per_pdf": ingest_summary["pages_per_pdf"],
        "total_pages": ingest_summary["total_pages"],
        "total_chunks": len(chunks_v2),
        "indexed_chunks": indexed_count,
        "collection_count": final_count,
        "indexing_success": final_count == len(chunks_v2),
        "collection_name": collection_name,
        "chroma_path": str(chroma_dir),
        "pages_output": str(pages_file),
        "chunks_output": str(chunks_output),
        "sections_detected": chunking_stats["sections_detected"],
        "table_like_count": chunking_stats["table_like_count"],
        "chunks_per_pdf": chunking_stats["chunks_per_pdf"],
        "companies": ingest_summary["companies"],
    }
    return summary
