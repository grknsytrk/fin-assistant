# RAG-Fin

Local-first financial report assistant that ingests quarterly PDF reports, builds ChromaDB-backed indexes, and returns grounded financial answers with evidence metadata through UI, API, and CLI interfaces.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Status](#project-status)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [CLI Reference](#cli-reference)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

## Overview

RAG-Fin processes quarterly financial report PDFs from `data/raw/`, extracts structured page records, chunks and indexes them in ChromaDB, and serves answers grounded in the source documents. Each answer carries evidence metadata (`doc_id`, `company`, `quarter`, `page`, `confidence`) so users can trace claims back to source pages.

Supported query modes: single-metric extraction, trend analysis across quarters, and cross-company comparison. An optional LLM commentary layer (via OpenRouter) can generate natural-language summaries on top of the structured output.

Intended for engineers and financial analysts who need auditable, traceable answers from PDF-based financial disclosures. RAG-Fin is not a managed cloud product; it runs locally by default.

## Features

- Ingest PDF reports into structured page-level JSONL records with company/quarter metadata.
- Build retrieval indexes using v1 (basic vector) and v2 (chunked vector) pipelines, with incremental v2 updates.
- Retrieve evidence through six retriever variants: v1 (vector), v2 (vector + lexical rerank), v3 (query-parsed hybrid with quarter filtering), v4 (BM25 lexical), v5 (vector + BM25 hybrid), v6 (cross-encoder rerank).
- Extract financial metrics (net kar, FAVOK, satis gelirleri, brut kar, etc.) with confidence scores and range-based sanity checks.
- Compute trend tables across quarters and ratio tables (margins, growth rates) with self-verification.
- Compare companies side-by-side when the query contains cross-company intent.
- Export trend and ratio tables as downloadable CSV via `GET /export`.
- Generate optional LLM commentary (OpenRouter) with multi-model fallback and real-number prompting.
- Capture user correction feedback into `data/processed/feedback.jsonl` for review loops.
- Fetch live KAP (Public Disclosure Platform) company snapshots with configurable caching.

## Project Status

- **Version:** 0.16.0
- **Python:** >=3.9, <3.13
- **CI:** GitHub Actions workflow (`ci-week16.yml`) runs pytest, compileall, deterministic demo preparation, and smoke metric checks on every push and PR.

## Architecture

```text
PDF files (data/raw/)
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Ingestion  │────▶│  Chunking &  │────▶│    ChromaDB      │
│ src/ingest  │     │  Indexing     │     │  (vector store)  │
│             │     │  src/index    │     │                  │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                  │
                              ┌────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   Retrieval      │
                    │ src/retrieve     │  v1..v6 retrievers
                    │ src/query_parser │  intent + quarter parsing
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌────────────┐ ┌────────────┐ ┌────────────────┐
      │  Answer    │ │  Metrics   │ │  Ratio Engine  │
      │ src/answer │ │  Extractor │ │ src/ratio_eng. │
      └─────┬──────┘ └─────┬──────┘ └──────┬─────────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                   ┌─────────────────┐
                   │   Interfaces    │
                   │ Streamlit UI    │  app/ui.py
                   │ FastAPI         │  app/api.py
                   │ Product CLI     │  ragfin/*
                   │ Engineering CLI │  src/cli.py
                   └─────────────────┘
```

1. **Ingestion** — `src/ingest.py` reads PDFs from `data/raw/`, extracts page text, detects company/quarter/year from filenames, and writes `data/processed/pages.jsonl`.
2. **Chunking & Indexing** — `src/index.py` splits pages into overlapping chunks (configurable size/overlap) and upserts them into ChromaDB collections using `intfloat/multilingual-e5-small` embeddings.
3. **Retrieval** — `src/retrieve.py` provides multiple retrieval strategies. `src/query_parser.py` adds intent detection, synonym expansion, and quarter-aware filtering.
4. **Answer & Analysis** — `src/answer.py` produces rules-based grounded answers. `src/metrics_extractor.py` extracts specific financial metrics with confidence. `src/ratio_engine.py` builds trend/ratio tables and cross-company comparisons.
5. **Commentary (optional)** — `src/commentary.py` sends structured answer payloads to an OpenRouter LLM for natural-language summaries in Turkish.
6. **KAP Integration** — `src/kap_fetcher.py` fetches live company data from the KAP (Public Disclosure Platform) API with local caching.

## Directory Structure

```text
.
├── app/                        # Streamlit UI and FastAPI application
│   ├── api.py                  # FastAPI endpoints (/ask, /export, /ingest, …)
│   ├── ui.py                   # Streamlit multi-page UI
│   └── ui_components.py        # Shared UI widgets (trust badges, etc.)
├── ragfin/                     # Product CLI entrypoints (installed as console_scripts)
│   ├── cli.py                  # ragfin demo | doctor | ui | api
│   ├── api.py                  # ragfin-api launcher
│   ├── demo.py                 # ragfin-demo launcher
│   ├── doctor.py               # ragfin-doctor launcher
│   └── ui.py                   # ragfin-ui launcher
├── src/                        # Core library modules
│   ├── answer.py               # Rules-based answer generation
│   ├── chunking.py             # Text chunking utilities
│   ├── cli.py                  # Engineering CLI (ingest, index, query, eval, …)
│   ├── commentary.py           # Optional LLM commentary via OpenRouter
│   ├── config.py               # YAML + env config loader (AppConfig dataclass)
│   ├── coverage_audit.py       # Per-company metric coverage audit
│   ├── embeddings.py           # E5 embedding wrapper
│   ├── eval_runner.py          # Evaluation pipeline runner
│   ├── index.py                # ChromaDB index builder (v1, v2, incremental)
│   ├── ingest.py               # PDF ingestion to pages.jsonl
│   ├── kap_fetcher.py          # KAP public disclosure API client
│   ├── metrics_extractor.py    # Financial metric extraction with confidence
│   ├── query_parser.py         # Query intent, quarter, and company parsing
│   ├── ratio_engine.py         # Trend/ratio tables + cross-company comparison
│   ├── retrieve.py             # Retriever implementations (v1–v6)
│   └── validators.py           # Ratio and extraction validators
├── data/
│   ├── raw/                    # Input PDF reports (place files here)
│   ├── processed/              # Generated outputs (pages, chunks, chroma, evals)
│   ├── demo/                   # Deterministic demo workspace
│   ├── demo_bundle/            # Demo fixtures and gold question sets
│   └── dictionaries/           # Metric name dictionaries (metrics_tr.yaml)
├── eval/                       # Evaluation gold question sets and runner
├── tests/                      # Pytest test suite
├── scripts/                    # Utility scripts (demo smoke check, etc.)
├── config.yaml                 # Default runtime configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies (pytest, httpx)
├── pyproject.toml              # Package metadata and console_scripts
├── Dockerfile                  # API container image (Python 3.11-slim)
├── Makefile                    # Convenience targets (demo, doctor, ui, api, test)
└── LICENSE                     # MIT License
```

## Getting Started

### Prerequisites

- **Python 3.9–3.12** (3.11 recommended; checked by `ragfin doctor`)
- **pip**
- **Git**
- Optional: Docker (for containerized API deployment)

### Quickstart

Shortest path to a running local demo with sample data:

```bash
git clone <repo-url> && cd rag-fin
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install and run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
ragfin demo
```

This prepares a deterministic demo workspace under `data/demo/` and launches the Streamlit UI at `http://127.0.0.1:8501`.

For full setup details, see [Installation](#installation).

## Installation

### Local installation (recommended)

```bash
python -m venv .venv
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt   # adds pytest, httpx
pip install -e .                       # installs ragfin CLI entrypoints
```

Verify the installation:

```bash
ragfin doctor
```

`ragfin doctor` checks Python version, required imports (`chromadb`, `sentence_transformers`, `streamlit`, `fastapi`, etc.), model availability, data paths, and Chroma collection status.

### Alternative: Docker (API only)

```bash
docker build -t ragfin-api .
docker run --rm -p 8000:8000 --env-file .env ragfin-api
```

The container runs `uvicorn app.api:app --host 0.0.0.0 --port 8000` on Python 3.11-slim. It exposes only the FastAPI service, not the Streamlit UI.

## Configuration

### Runtime config file

All runtime parameters are read from `config.yaml` at the repository root. Override the config path with the `RAGFIN_CONFIG` environment variable.

Key sections in `config.yaml`:

| Section | Controls |
|---|---|
| `paths` | Input/output directories (`raw_dir`, `processed_dir`, file paths) |
| `chroma` | ChromaDB storage directory and collection names |
| `chunking` | Chunk size and overlap for v1/v2 pipelines |
| `retrieval` | Top-k values and alpha/beta blending weights per retriever |
| `models` | Embedding model (`intfloat/multilingual-e5-small`) and cross-encoder model |
| `extraction` | Metric dictionary, confidence thresholds, expected value ranges |
| `kap` | KAP fetcher toggle, timeout, cache TTL, user-agent |
| `llm_assistant` | LLM commentary toggle, provider, model, timeout, temperature |
| `health` | Dashboard health indicator thresholds (margin, growth) |

### Environment variables

A `.env` file in the working directory or repository root is loaded automatically.

| Variable | Required | Default | Description | Example |
|---|---|---|---|---|
| `RAGFIN_CONFIG` | No | `config.yaml` | Override config file path | `RAGFIN_CONFIG=data/demo/config.demo.yaml` |
| `OPENROUTER_API_KEY` | Only for LLM commentary | *(empty)* | OpenRouter API key for LLM commentary | `OPENROUTER_API_KEY=sk-or-v1-xxxx` |
| `RAGFIN_LLM_ASSISTANT_ENABLED` | No | `false` | Enable LLM commentary generation | `true` |
| `RAGFIN_LLM_ASSISTANT_PROVIDER` | No | `openrouter` | LLM provider name | `openrouter` |
| `RAGFIN_LLM_ASSISTANT_MODEL` | No | `arcee-ai/trinity-large-preview:free` | LLM model identifier | `meta-llama/llama-3.3-70b-instruct:free` |
| `RAGFIN_LLM_ASSISTANT_MAX_TOKENS` | No | *(null)* | Max token cap for commentary | `512` |
| `RAGFIN_LLM_ASSISTANT_TIMEOUT_S` | No | `8` | LLM request timeout in seconds | `10` |
| `RAGFIN_LLM_ASSISTANT_TEMPERATURE` | No | `0.2` | LLM sampling temperature | `0.2` |
| `RAGFIN_LLM_ASSISTANT_REASONING_ENABLED` | No | `true` | Enable reasoning mode flag | `true` |
| `RAGFIN_KAP_ENABLED` | No | `true` | Enable KAP fetcher integration | `false` |
| `RAGFIN_KAP_TIMEOUT_SECONDS` | No | `10` | HTTP timeout for KAP requests | `12` |
| `RAGFIN_KAP_CACHE_TTL_HOURS` | No | `24` | Cache TTL for KAP responses | `24` |
| `RAGFIN_KAP_USER_AGENT` | No | `ragfin-kap-fetcher/1.0 (+local-first)` | User-Agent header for KAP requests | `ragfin-kap-fetcher/1.0 (+my-org)` |

`RAGFIN_LLM_COMMENTARY_*` aliases are also accepted by the config loader.

**Never commit real API keys.** Store secrets in `.env` and add it to `.gitignore`.

## Usage

### Run the Streamlit UI

```bash
ragfin ui
```

Or directly:

```bash
python -m ragfin.ui
```

Opens a multi-page financial assistant UI at `http://127.0.0.1:8501`. Pages include dashboard overview, single-question Q&A, trend analysis, and ratio/comparison views.

### Run the FastAPI service

```bash
ragfin api --host 0.0.0.0 --port 8000 --reload
```

Verify it is running:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### Ask a grounded question (API)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"2025 ucuncu ceyrek net kar kac?","retriever":"v3","mode":"single","company":"BIM"}'
```

Response contains `answer`, `parsed` (extracted metric, quarter, company), `evidence` (source chunks with page numbers), and `debug` fields.

### Export tables as CSV

```bash
curl "http://127.0.0.1:8000/export?type=trend&company=BIM" -o trend.csv
curl "http://127.0.0.1:8000/export?type=ratio&company=BIM" -o ratios.csv
```

### Ingest your own data

Place PDF reports in `data/raw/`, then:

```bash
python -m src.cli ingest
python -m src.cli index_v2
```

Or via the API:

```bash
curl -X POST http://127.0.0.1:8000/ingest
curl -X POST http://127.0.0.1:8000/index -H "Content-Type: application/json" -d '{"version":"v2"}'
```

## API Reference

**Base URL (local default):** `http://127.0.0.1:8000`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check. Returns `{"status":"ok"}`. |
| `GET` | `/stats` | Dataset and index counters, available companies. |
| `POST` | `/ingest` | Ingest PDFs from the configured `paths.raw_dir`. |
| `POST` | `/index` | Build or rebuild index. Body: `{"version":"v1"}` or `{"version":"v2"}`. |
| `POST` | `/ask` | Answer a question. Body: `AskRequest` (see below). |
| `POST` | `/commentary` | Generate LLM commentary from an answer payload. |
| `POST` | `/feedback` | Store user correction feedback to JSONL. |
| `GET` | `/export` | Download trend or ratio table as CSV. Params: `type=trend|ratio`, `company`. |

### `POST /ask` request body

```json
{
  "question": "BIM ve MIGROS net karini karsilastir",
  "retriever": "v3",
  "mode": "single",
  "company": null
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | *(required)* | Natural-language financial question (Turkish or English) |
| `retriever` | enum | `"v3"` | One of `v1`, `v2`, `v3`, `v5`, `v6` |
| `mode` | enum | `"single"` | `"single"` for point queries, `"trend"` for multi-quarter |
| `company` | string \| null | `null` | Filter to a specific company; `null` for auto-detection |

### `POST /feedback` request body

```json
{
  "company": "BIM",
  "quarter": "Q1",
  "metric": "net_kar",
  "extracted_value": "1,23 mlr TL",
  "user_value": "1,20 mlr TL",
  "evidence_ref": "[doc|Q1|5|gelir tablosu]",
  "verdict": "yanlis"
}
```

## CLI Reference

### Product CLI (`ragfin`)

Installed as a console script via `pip install -e .`.

```bash
ragfin --version   # prints version (0.16.0)
ragfin --help
```

| Command | Description |
|---|---|
| `ragfin demo` | Prepare deterministic demo workspace and launch Streamlit UI. Flags: `--prepare-only`, `--no-clean`, `--host`, `--port`. |
| `ragfin doctor` | Run environment diagnostics (Python version, imports, models, paths, Chroma). |
| `ragfin ui` | Start Streamlit UI (`app/ui.py`). |
| `ragfin api` | Start FastAPI service. Flags: `--host`, `--port`, `--reload`. |

### Engineering CLI (`python -m src.cli`)

Lower-level commands for data pipeline and evaluation tasks.

```bash
python -m src.cli --help
```

| Command | Example |
|---|---|
| `ingest` | `python -m src.cli ingest` |
| `index_v2` | `python -m src.cli index_v2` |
| `index_v2_incremental` | `python -m src.cli index_v2_incremental` |
| `query_v3` | `python -m src.cli query_v3 "2025 Q3 net kar?" --company BIM` |
| `ask_v3` | `python -m src.cli ask_v3 "Q1 Q2 Q3 net kar trendi nasil?" --company BIM` |
| `coverage_audit` | `python -m src.cli coverage_audit --company MIGROS` |
| `metrics_report` | `python -m src.cli metrics_report` |
| `doctor` | `python -m src.cli doctor` |

## Development

### Local developer setup

```bash
python -m venv .venv
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### Makefile targets

```bash
make demo      # prepare demo workspace and launch UI
make doctor    # run environment diagnostics
make ui        # start Streamlit UI
make api       # start FastAPI service
make test      # run pytest
```

### Key development dependencies

| Package | Purpose |
|---|---|
| `pytest` (>=8.2) | Test runner |
| `httpx` (>=0.27) | Async HTTP client for API tests |

## Testing

### Run the test suite

```bash
python -m pytest -q
```

Tests cover: API endpoints, auto-verify logic, commentary generation, config loading, demo packaging, KAP fetcher, metric extraction, multi-company queries, query parsing, ratio engine, retriever smoke tests, and validators.

### CI parity

The CI workflow (`.github/workflows/ci-week16.yml`) runs these steps on every push:

```bash
python -m compileall src app ragfin       # syntax check all modules
python -m pytest -q                        # unit and integration tests
python -m ragfin.demo --prepare-only       # deterministic demo workspace
# smoke metric evaluation on demo dataset
python -m src.cli metrics_report \
  --gold-file data/demo_bundle/gold_questions_demo.jsonl \
  --multi-company-gold-file data/demo_bundle/gold_questions_demo_multicompany.jsonl \
  --detailed-output data/demo/processed/eval_metrics_detailed.jsonl \
  --summary-output data/demo/processed/eval_metrics_summary.json \
  --week6-summary-output data/demo/processed/eval_metrics_week6.json \
  --top-k 3 --top-k-initial-v2 8 --top-k-initial-v3 12 \
  --top-k-initial-v5-vector 12 --top-k-initial-v5-bm25 12 \
  --top-k-candidates-v6 8
python scripts/check_demo_metrics_smoke.py data/demo/processed/eval_metrics_summary.json
```

Run these locally before opening a PR to match CI behavior.

## Deployment

### Docker (API)

```bash
docker build -t ragfin-api .
docker run --rm -p 8000:8000 --env-file .env ragfin-api
```

The image uses `python:3.11-slim`, installs `requirements.txt`, and runs uvicorn on port 8000. Mount a volume or copy PDF data into the container as needed.

### Direct (API)

```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Streamlit UI

```bash
ragfin ui
```

Streamlit runs on port 8501 by default. For production use, place it behind a reverse proxy with TLS.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ragfin doctor` shows `python_version: FAIL` | Python version outside 3.9–3.12 | Create a virtual environment with a supported Python (e.g., 3.11) and reinstall dependencies. |
| `ragfin doctor` reports `import_chromadb: FAIL` or `import_sentence_transformers: FAIL` | Missing dependencies in the active environment | Activate the correct venv and run `pip install -r requirements.txt`. |
| `POST /ask` returns `Dokümanda bulunamadı` | Data not ingested/indexed, or overly restrictive company filter | Run `python -m src.cli ingest` then `python -m src.cli index_v2`. Check `GET /stats` for available companies. Retry without the `company` filter. |
| `POST /commentary` returns empty payload | LLM assistant disabled or API key missing | Set `RAGFIN_LLM_ASSISTANT_ENABLED=true` and provide `OPENROUTER_API_KEY` in `.env`. Restart the service. |
| ChromaDB errors on startup | Corrupted or version-mismatched Chroma data | Delete `data/processed/chroma/` and re-run `python -m src.cli index_v2`. |
| Streamlit fails to start | Port 8501 already in use, or Streamlit not installed | Check for conflicting processes on port 8501. Verify `streamlit` is installed (`pip list \| grep streamlit`). |

## Contributing

1. **Report issues** — Open a GitHub issue with a clear description of the bug or proposal.
2. **Branch** — Create a short-lived branch from `main` with a descriptive name (e.g., `fix/quarter-filter-bug`).
3. **Commit** — Keep commits focused. Use clear messages that describe intent.
4. **Local checks** — Run these before opening a PR:
   ```bash
   python -m compileall src app ragfin
   python -m pytest -q
   python -m ragfin.demo --prepare-only
   ```
5. **Open a PR** with:
   - Problem statement and scope
   - Testing evidence (commands and outcomes)
   - Config or data migration notes, if applicable

## Security

- Store secrets (`OPENROUTER_API_KEY`) in `.env`. Never commit real keys.
- Review `data/raw/` contents before committing — PDFs may contain private or licensed data.
- The FastAPI service has **no built-in authentication or authorization**. When exposing beyond localhost, place it behind your own auth layer and TLS termination.

## License

Licensed under the [MIT License](LICENSE).
