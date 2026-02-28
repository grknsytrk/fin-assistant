ROLE:
You are an AI coding agent. Build Stage-1 project: “RAG-Fin v1” (Financial Report Analysis Assistant) for BİMAS 2025 Q1/Q2/Q3 activity report PDFs.
Goal: A working local RAG pipeline that answers ONLY from the PDFs, always shows sources (pdf file + page), and avoids hallucination.

HARD REQUIREMENTS (must):
1) Local-first: do NOT require external paid APIs. Use open-source components (PyPDF/pypdf, ChromaDB, sentence-transformers).
2) Ingestion must preserve page numbers. Every chunk must have metadata: {doc_id, quarter, page, chunk_id}.
3) Retrieval must return top-k chunks with their sources. 
4) Answering must be “grounded”: 
   - If info is not found in retrieved chunks, respond: “Dokümanda bulunamadı” and list what pages were searched.
   - Never invent numbers.
5) Output format must be consistent:
   - Summary answer (bullets)
   - Evidence section with citations: [doc_id, quarter, page]
   - Quote small excerpts (max ~2 lines each) from the retrieved chunks as evidence.

PROJECT STRUCTURE:
Create repo structure:
rag-fin/
  data/raw/              # already contains PDFs
  data/processed/
  src/
    ingest.py
    chunking.py
    index.py
    retrieve.py
    answer.py
    cli.py
  eval/
    questions.jsonl
  README.md
  requirements.txt

TASKS (execute in order):
A) Detect the PDFs in data/raw. Print filenames found.
B) Implement PDF text extraction with page granularity.
   - Use pypdf (preferred) and fall back gracefully if a page returns empty text.
   - Clean text: normalize spaces, remove null bytes, collapse excessive newlines.
C) Chunking v1 (Week-1 simple): 
   - Character-based chunking with overlap.
   - Default: chunk_size ~900 chars, overlap ~150 chars.
D) Embeddings:
   - Use SentenceTransformer: "intfloat/multilingual-e5-small" (fast + TR-friendly).
E) Vector DB:
   - Use ChromaDB PersistentClient stored at data/processed/chroma
   - Collection name: "bimas_faaliyet_2025"
F) Build retrieval CLI:
   - Command: python -m src.cli query "net kâr"
   - Prints top 5 results with: distance/score, doc_id, quarter, page, chunk_id, and first 400 chars of text.
G) Build answering module:
   - Use a local LLM if available (optional), but if no local LLM is configured, implement a rules-based answerer that:
     - returns retrieved evidence only,
     - extracts numeric candidates via regex (e.g., \d[\d\.\,]*),
     - never fabricates.
   - Preferred: implement an adapter layer so later we can plug a real LLM.
H) Add eval scaffold:
   - Create eval/questions.jsonl with 30 finance questions (TR).
   - Create a script to run retrieval for each question and store which pages were retrieved (for manual review).

QUALITY CHECKS:
- Run ingestion and indexing end-to-end and print:
  - number of PDFs
  - number of pages extracted per PDF
  - number of chunks created
  - indexing success
- Run at least 5 queries and show outputs.

DELIVERABLES:
1) Code + requirements.txt
2) README.md with:
   - setup instructions (venv, pip install -r requirements.txt)
   - how to ingest/index
   - how to query
   - limitations (tables may be imperfect in v1)
3) Ensure everything runs on Windows.

IMPORTANT:
- Do not ask the user questions unless absolutely blocked.
- If a library fails, choose the next best alternative and document it in README.
- Keep the code clean, typed where reasonable, and with clear logs.
