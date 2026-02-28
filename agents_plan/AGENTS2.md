ROLE:
You are an AI coding agent. Implement Week-2 improvements for the existing Stage-1 project “RAG-Fin v1” (BİMAS 2025 Q1/Q2/Q3 activity reports).
Goal: Increase retrieval accuracy (not just “works”), with heading-aware chunking, richer metadata, and a simple rerank/boost layer—while keeping the system grounded (no hallucinations).

CONTEXT:
Project already has:
- page extraction → data/processed/pages.jsonl
- char-based chunking → data/processed/chunks.jsonl
- embeddings: intfloat/multilingual-e5-small
- chroma collection: data/processed/chroma, name: bimas_faaliyet_2025
- CLI: python -m src.cli query "..."
- AnswerAdapter + rules-based grounded answerer
- eval/questions.jsonl + eval runner output file

DO NOT BREAK Week-1:
Keep old pipeline runnable. Add Week-2 as opt-in flags or new functions/modules, and update README.

TASKS (Week-2):

1) Heading-aware chunking (core)
Implement a new chunker option that is aware of section headings and paragraph boundaries.

HEADING DETECTION RULES (TR finance PDFs):
Treat a line as a heading if it matches one of:
- ALL CAPS line (Turkish letters included) length 4–80 and mostly letters/spaces
- Numbered headings: r"^\s*\d+(\.\d+)*[\)\.]?\s+\S+"
- Common finance headings keywords (case-insensitive contains):
  ["RİSK", "RISK", "FİNANS", "FINANS", "ÖZET", "OZET", "YÖNET", "YONET",
   "FAALİYET", "FAALIYET", "PERFORMANS", "SONUÇ", "SONUC",
   "DEĞERLEND", "DEGERLEND", "BEKLENT", "STRATEJ", "YATIRIM",
   "KURUMSAL", "SÜRDÜR", "SURDUR", "PAZAR", "SEKTÖR", "SEKTOR"]

BEHAVIOR:
- Split each page into lines.
- Build sections: when a heading is detected, start a new section.
- Each section has: section_title, section_text, start_page (current page), and keep page number for citation.
- If no heading exists on a page, treat it as a single unnamed section ("(no heading)").

Then chunk inside each section:
- Prefer paragraph-based chunking: split on blank lines first.
- Merge paragraphs until approx target_size (token/char) with overlap.
- Keep char-based fallback.

OUTPUT:
- Write a new file: data/processed/chunks_v2.jsonl
- Each chunk must include metadata:
  {doc_id, quarter, page, chunk_id, section_title, block_type, chunk_version:"v2"}
- block_type default: "text"

2) Table-like block separation (simple heuristic)
Within each section/page, detect “table-like” paragraphs:
- A paragraph is table-like if:
  - It contains >= 3 numeric tokens (regex: \d[\d\.\,]*)
  - AND has many spaces/alignment OR multiple short lines (>=3 lines) with numbers
When detected:
- Store it as separate chunk(s) with block_type="table_like"
- Do NOT mix table-like paragraph with narrative paragraph in the same chunk.

3) Indexing v2 (new collection OR metadata toggle)
Option A (preferred): Create a new Chroma collection:
- name: "bimas_faaliyet_2025_v2"
- stored under same chroma directory
Option B: Same collection but include chunk_version and allow filtering.

Implement whichever is simpler/cleaner, but do not delete old collection.
Index chunks_v2 with the same embedding model.

4) Retrieval improvements: hybrid boost + optional filter
Add a rerank/boost layer after vector retrieval:
- Retrieve top_k=15 from Chroma first.
- Compute a lexical boost score:
  - Tokenize query (TR-friendly simple split + lowercase + strip punctuation)
  - For each chunk: add +1 per exact token match (cap at e.g., 8)
  - Add extra boost if query contains “kâr/kar” and chunk contains “net kâr/net kar”
  - Add extra boost if query contains “FAVÖK/FAVOK”
- Final score = normalized_vector_score + alpha*lexical_boost
- Sort and return top_k_final=5.

Also support:
- If user passes --quarter Q1/Q2/Q3, filter results by metadata quarter (where possible).
- Add CLI command: 
  python -m src.cli query_v2 "net kâr" --quarter Q3

5) Update Answer module to include section_title
In the Evidence section, show:
- [doc_id, quarter, page, section_title]
Include short excerpt as before.

6) Eval update
Update eval runner to run both:
- baseline (v1)
- improved (v2)
For each question store:
- top pages (v1 vs v2)
- top section_titles (v2)
Write to: data/processed/eval_retrieval_comparison.jsonl

7) README update
Add:
- How to run v2 ingestion/chunking/index
- How to query v2
- What changed vs v1
- Known limitations

ACCEPTANCE CHECKS:
- Running v2 should produce chunks_v2.jsonl and index into new collection.
- Query “net kâr”, “riskler”, “mağaza sayısı”, “satış gelirleri”, “FAVÖK marjı” should show more coherent sections and better page targeting than v1.
- Must remain grounded; if evidence is missing, answer says “Dokümanda bulunamadı”.

IMPORTANT:
- Do not ask the user questions unless blocked.
- Keep code clean, minimal dependencies.
- Ensure Windows compatibility.
- Log key stats: chunks count per PDF, table_like count, sections detected count.
