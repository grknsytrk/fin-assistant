---
name: rag-fin-ops
description: Operate and troubleshoot the RAG-Fin local-first pipeline in this repository. Use when requests involve demo setup, ingestion and indexing, retrieval and answer verification, API or UI startup, coverage or evaluation runs, or environment diagnostics via ragfin and src.cli commands.
---

# RAG-Fin Ops

## Overview

Use this skill to select and execute the right operational workflow for this repo.
Prefer deterministic `python -m ...` commands from the project root and keep outputs evidence-driven.

## Workflow Selection

1. Run environment checks when dependency, path, or model issues appear.
2. Run deterministic demo flow when the user wants a fast product walkthrough.
3. Run ingest and index workflows when raw PDFs changed.
4. Run retrieval and answer debug workflows when quality is questioned.
5. Run coverage and eval workflows when quality measurement or gap analysis is requested.
6. Start UI or API only after environment and index health look good.

## Baseline Rules

- Execute commands from repository root.
- Use Python `3.9` to `3.12` (`chromadb` is not compatible with `3.14`).
- Prefer `.venv39` on Windows when available.
- Read [references/commands.md](references/commands.md) for exact command variants.

## Playbooks

### 1. Recover Environment

- Run `python -m ragfin.doctor` or `ragfin doctor`.
- Reinstall dependencies and rerun doctor if import checks fail.
- Fix writable path issues before ingestion or indexing.

### 2. Run Deterministic Demo

- Run `python -m ragfin.demo` for full demo preparation and UI launch.
- Run `python -m ragfin.demo --prepare-only` for fixture ingest and index only.
- Use this path for fast smoke validation.

### 3. Ingest and Index Real Data

- Run `python -m src.cli --config config.yaml ingest`.
- Run `python -m src.cli --config config.yaml index_v2` for full rebuild.
- Run `python -m src.cli --config config.yaml index_v2_incremental` for changed PDFs only.
- Validate PDF, page, chunk, and index counters before query testing.

### 4. Debug Retrieval and Answers

- Start with retrieval visibility:
  - `python -m src.cli --config config.yaml query_v3 "<soru>" --company <KOD>`
- Continue with grounded answer path:
  - `python -m src.cli --config config.yaml ask_v3 "<soru>" --company <KOD>`
- Inspect retrieved chunk metadata before changing retrieval logic.

### 5. Run Coverage and Evaluation

- Run `python -m src.cli --config config.yaml metrics_report`.
- Run `python -m src.cli --config config.yaml coverage_audit --company <KOD>`.
- Run `python -m src.cli --config config.yaml dict_suggest --company <KOD>`.
- Use generated outputs under `data/processed/` for iteration.

### 6. Start Product Surfaces

- Start UI with `python -m ragfin.ui`.
- Start API with `python -m ragfin.api --host 0.0.0.0 --port 8000`.
- Keep config context explicit when switching demo vs main workspace.

## Execution Constraints

- Keep answers grounded in retrieved evidence and avoid fabricated numeric values.
- Prefer the smallest command set that verifies or fixes the issue.
- Report key run outputs and the next action after each operation.
- Explicitly state when a command changes persistent index or processed artifacts.
