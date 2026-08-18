"""Hugging Face Gradio Space entrypoint for the FastAPI application.

The Gradio SDK runner executes this file.  The Space uses a small Gradio API
probe to satisfy ZeroGPU startup detection while the existing FastAPI routes
are registered on the same server.
"""

from __future__ import annotations

import os

from starlette.middleware.gzip import GZipMiddleware
import uvicorn

os.environ.setdefault("RAGFIN_FUND_COLLECTOR_ENABLED", "0")

from app.api import app as api_app, bootstrap_application_storage

try:
    import gradio as gr
    import spaces
except ImportError:  # pragma: no cover - only the HF Gradio runtime provides this
    gr = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


if gr is not None and spaces is not None:
    demo = gr.Server()
    # The API router is mounted into Gradio's FastAPI server on Spaces, so
    # middleware registered on app.api does not wrap these requests.
    demo.add_middleware(GZipMiddleware, minimum_size=1024)
    demo.include_router(api_app.router)

    @spaces.GPU
    @demo.api(name="probe")
    def _hf_zero_gpu_probe(value: str) -> str:
        """Satisfy ZeroGPU startup detection; the API itself is CPU-bound."""

        return value
else:
    demo = None


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    # ``include_router`` does not run api_app's FastAPI lifespan on Gradio
    # Spaces. Run the same schema/cache bootstrap before serving routes.
    bootstrap_application_storage()
    if demo is not None:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            prevent_thread_lock=False,
        )
        return

    uvicorn.run(
        api_app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
