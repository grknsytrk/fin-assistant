from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML gerekli. requirements.txt kurulumunu kontrol edin.") from exc

from src.config import load_config
from src.index import build_index_v2_from_pages
from src.ingest import ingest_page_fixtures

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_BUNDLE_DIR = REPO_ROOT / "data" / "demo_bundle"
DEMO_FIXTURE_FILE = DEMO_BUNDLE_DIR / "pages_fixture.jsonl"
DEMO_GOLD_FILE = DEMO_BUNDLE_DIR / "gold_questions_demo.jsonl"
DEMO_GOLD_MULTI_FILE = DEMO_BUNDLE_DIR / "gold_questions_demo_multicompany.jsonl"
DEMO_QUESTIONS_FILE = DEMO_BUNDLE_DIR / "demo_questions.txt"
DEMO_ROOT = REPO_ROOT / "data" / "demo"
DEMO_CONFIG_PATH = DEMO_ROOT / "config.demo.yaml"


def _demo_config_payload() -> Dict[str, object]:
    metrics_dict_file = (REPO_ROOT / "data" / "dictionaries" / "metrics_tr.yaml").resolve()
    return {
        "paths": {
            "raw_dir": "raw",
            "processed_dir": "processed",
            "pages_file": "processed/pages.jsonl",
            "chunks_v1_file": "processed/chunks.jsonl",
            "chunks_v2_file": "processed/chunks_v2.jsonl",
            "ui_log_file": "processed/ui_logs.jsonl",
        },
        "chroma": {
            "dir": "processed/chroma",
            "collection_v1": "ragfin_demo_v1",
            "collection_v2": "ragfin_demo_v2",
        },
        "evaluation": {
            "gold_file": str(DEMO_GOLD_FILE.resolve()),
            "gold_multicompany_file": str(DEMO_GOLD_MULTI_FILE.resolve()),
            "detailed_output": "processed/eval_metrics_detailed.jsonl",
            "summary_output": "processed/eval_metrics_summary.json",
            "week6_summary_output": "processed/eval_metrics_week6.json",
            "error_output": "processed/error_analysis.jsonl",
            "latency_output": "processed/latency_benchmark.json",
            "latency_sample_size": 6,
        },
        "extraction": {
            "metrics_dictionary_file": str(metrics_dict_file),
        },
    }


def write_demo_config(config_path: Path = DEMO_CONFIG_PATH) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_demo_config_payload(), f, allow_unicode=False, sort_keys=False)
    return config_path


def _clear_workspace(config: AppConfig) -> None:
    for target in (config.paths.raw_dir, config.paths.processed_dir):
        if target.exists():
            shutil.rmtree(target)


def bootstrap_sample_into_config(
    config_path: Path,
    *,
    clean_workspace: bool = False,
) -> Dict[str, object]:
    cfg = load_config(config_path)
    if clean_workspace:
        _clear_workspace(cfg)

    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    pages, ingest_summary = ingest_page_fixtures(
        fixtures_file=DEMO_FIXTURE_FILE,
        output_file=cfg.paths.pages_file,
    )
    index_summary = build_index_v2_from_pages(
        pages_file=cfg.paths.pages_file,
        processed_dir=cfg.paths.processed_dir,
        collection_name=cfg.chroma.collection_v2,
        chunk_size=cfg.chunking.v2.chunk_size,
        overlap=cfg.chunking.v2.overlap,
    )
    return {
        "config_path": str(cfg.path),
        "pages_loaded": len(pages),
        "ingest_summary": ingest_summary,
        "index_summary": index_summary,
    }


def prepare_demo_workspace(
    *,
    config_path: Path = DEMO_CONFIG_PATH,
    clean_workspace: bool = True,
) -> Dict[str, object]:
    written = write_demo_config(config_path)
    result = bootstrap_sample_into_config(
        config_path=written,
        clean_workspace=clean_workspace,
    )
    result["demo_root"] = str(config_path.parent)
    return result


def launch_demo_ui(
    *,
    config_path: Path = DEMO_CONFIG_PATH,
    host: str = "127.0.0.1",
    port: int = 8501,
) -> int:
    env = os.environ.copy()
    env["RAGFIN_CONFIG"] = str(config_path.resolve())
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str((REPO_ROOT / "app" / "ui.py").resolve()),
        "--server.address",
        host,
        "--server.port",
        str(int(port)),
    ]
    print(f"[demo] config: {config_path.resolve()}")
    print(f"[demo] streamlit: http://{host}:{port}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


def _print_summary(summary: Dict[str, object]) -> None:
    print("[demo] workspace hazirlandi")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_demo_questions() -> list[str]:
    if not DEMO_QUESTIONS_FILE.exists():
        return [
            "BIM 2025 ucuncu ceyrek net kar kac?",
            "MIGROS 2025 ikinci ceyrek ciro ne kadar?",
            "SOK 2025 birinci ceyrek net donem zarari kac?",
            "Q1 Q2 Q3 FAVOK marji trendi nasil?",
            "BIM ve MIGROS net karini karsilastir.",
        ]
    rows: list[str] = []
    with DEMO_QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            question = line.strip()
            if question:
                rows.append(question)
    return rows[:5]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-Fin deterministic demo runner")
    parser.add_argument("--config-out", default=str(DEMO_CONFIG_PATH), help="Demo config dosyasi")
    parser.add_argument("--prepare-only", action="store_true", help="Sadece fixture ingest + index yap")
    parser.add_argument("--no-clean", action="store_true", help="Demo workspace temizligini atla")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_out = Path(args.config_out)
    summary = prepare_demo_workspace(
        config_path=config_out,
        clean_workspace=not args.no_clean,
    )
    _print_summary(summary)
    if args.prepare_only:
        return 0
    return launch_demo_ui(config_path=config_out, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
