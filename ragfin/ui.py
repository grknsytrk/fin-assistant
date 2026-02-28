from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str((repo_root / "app" / "ui.py").resolve()),
    ]
    if argv:
        cmd.extend(argv)
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
