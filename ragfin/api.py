from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG-Fin FastAPI service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api:app",
        "--host",
        args.host,
        "--port",
        str(int(args.port)),
    ]
    if args.reload:
        cmd.append("--reload")
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
