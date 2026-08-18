"""Hugging Face Gradio Space entrypoint for the FastAPI application.

The Gradio SDK runner executes this file.  We keep the entrypoint separate
from the ``app`` package so importing ``app.api`` cannot be shadowed by an
``app.py`` module at the repository root.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    # A Space does not have a persistent local runtime and may be restarted;
    # scheduled collection is handled outside the web process in production.
    os.environ.setdefault("RAGFIN_FUND_COLLECTOR_ENABLED", "0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
