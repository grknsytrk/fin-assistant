# RAG-Fin Command Reference

Use this file for exact command patterns while running the workflows in `SKILL.md`.

## Core Start Paths

```powershell
# Deterministic demo (recommended quick check)
python -m ragfin.demo

# Environment diagnostics
python -m ragfin.doctor

# Product surfaces
python -m ragfin.ui
python -m ragfin.api --host 0.0.0.0 --port 8000
```

## Data Pipeline

```powershell
# Extract pages from raw PDFs
python -m src.cli --config config.yaml ingest

# Full v2 index build
python -m src.cli --config config.yaml index_v2

# Incremental v2 index build (changed files only)
python -m src.cli --config config.yaml index_v2_incremental
```

## Retrieval and Grounded Answers

```powershell
# Inspect retrieval first
python -m src.cli --config config.yaml query_v3 "2025 ucuncu ceyrek net kar kac?" --company BIM

# Then run grounded answer path
python -m src.cli --config config.yaml ask_v3 "2025 ucuncu ceyrek net kar kac?" --company BIM
```

## Evaluation and Coverage

```powershell
# Retrieval metrics and benchmark outputs
python -m src.cli --config config.yaml metrics_report

# Coverage diagnostics by company
python -m src.cli --config config.yaml coverage_audit --company MIGROS

# Phrase suggestions for missing dictionary coverage
python -m src.cli --config config.yaml dict_suggest --company SOK
```

## Windows venv Variant

```powershell
.\.venv39\Scripts\python.exe -m ragfin.demo
.\.venv39\Scripts\python.exe -m src.cli --config config.yaml index_v2
```
