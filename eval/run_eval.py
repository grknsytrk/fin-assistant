from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_PY = (3, 9)
MAX_PY_EXCLUSIVE = (3, 14)


def _validate_python_version() -> None:
    current = sys.version_info[:2]
    if not (MIN_PY <= current < MAX_PY_EXCLUSIVE):
        major, minor = current
        raise SystemExit(
            "Bu proje icin Python 3.9-3.12 kullanin. "
            f"Mevcut surum: {major}.{minor}. "
            "Ornek: .\\.venv39\\Scripts\\python.exe eval\\run_eval.py"
        )


def main() -> None:
    _validate_python_version()
    from src.eval_runner import run_retrieval_eval_comparison

    parser = argparse.ArgumentParser(description="Run retrieval comparison evaluation (v1 vs v2)")
    parser.add_argument("--questions-file", default="eval/questions.jsonl")
    parser.add_argument("--output-file", default="data/processed/eval_retrieval_comparison.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-initial-v2", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    summary = run_retrieval_eval_comparison(
        questions_file=Path(args.questions_file),
        output_file=Path(args.output_file),
        top_k=args.top_k,
        top_k_initial_v2=args.top_k_initial_v2,
        alpha=args.alpha,
    )
    print(f"Soru sayisi: {summary['questions_count']}")
    print(f"Top-k (final): {summary['top_k']}")
    print(f"Top-k initial (v2): {summary['top_k_initial_v2']}")
    print(f"Alpha: {summary['alpha']}")
    print(f"Cikti dosyasi: {summary['output_file']}")


if __name__ == "__main__":
    main()
