from __future__ import annotations

import argparse
import sys

from ragfin import __version__
from ragfin.api import main as api_main
from ragfin.demo import main as demo_main
from ragfin.doctor import main as doctor_main
from ragfin.ui import main as ui_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragfin", description="RAG-Fin product CLI")
    parser.add_argument("--version", action="version", version=f"ragfin {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Prepare deterministic sample workspace and launch UI")
    demo.add_argument("--prepare-only", action="store_true")
    demo.add_argument("--no-clean", action="store_true")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8501)
    demo.add_argument("--config-out", default="data/demo/config.demo.yaml")

    doctor = sub.add_parser("doctor", help="Run environment diagnostics")
    doctor.add_argument("--config", default="config.yaml")

    ui = sub.add_parser("ui", help="Run Streamlit UI")
    ui.add_argument("extra", nargs=argparse.REMAINDER)

    api = sub.add_parser("api", help="Run FastAPI service")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return demo_main(
            [
                "--config-out",
                str(args.config_out),
                "--host",
                str(args.host),
                "--port",
                str(args.port),
            ]
            + (["--prepare-only"] if args.prepare_only else [])
            + (["--no-clean"] if args.no_clean else [])
        )
    if args.command == "doctor":
        return doctor_main(["--config", str(args.config)])
    if args.command == "ui":
        return ui_main(args.extra)
    if args.command == "api":
        payload = ["--host", str(args.host), "--port", str(args.port)]
        if args.reload:
            payload.append("--reload")
        return api_main(payload)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
