"""Hugging Face Gradio Space entrypoint for the FastAPI application.

The Gradio SDK runner executes this file.  The Space uses a small Gradio API
probe to satisfy ZeroGPU startup detection while the existing FastAPI routes
are registered on the same server.
"""

from __future__ import annotations

import atexit
import logging
import os

import uvicorn

os.environ.setdefault("RAGFIN_FUND_COLLECTOR_ENABLED", "0")

from app.api import (
    app as api_app,
    bootstrap_application_storage,
    close_postgres_pool,
    close_shared_http_clients,
    configure_http_middleware,
)
from app.cache import get_cache
from app.etag import ORJSONResponse


def _shutdown_runtime_resources() -> None:
    for cleanup_fn in (close_postgres_pool, close_shared_http_clients):
        try:
            cleanup_fn()
        except Exception:
            pass
    try:
        cache_backend = get_cache()
        close_fn = getattr(cache_backend, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass


atexit.register(_shutdown_runtime_resources)

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
    configure_http_middleware(demo)
    try:
        demo.include_router(api_app.router, default_response_class=ORJSONResponse)
    except TypeError:
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
    # Storage bootstrap runs before Gradio/Uvicorn configures logging. Enable
    # its concise cache-hydration metric so startup egress can be inspected in
    # Hugging Face Space logs without turning on verbose application logging.
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger("app.database").setLevel(logging.INFO)
    # ``include_router`` does not run api_app's FastAPI lifespan on Gradio
    # Spaces. Run the same schema/cache bootstrap before serving routes.
    # Postgres is an optional cold-start seed: when it is unreachable the API
    # must still come up and serve from process/disk caches.
    try:
        bootstrap_application_storage()
    except Exception:
        logging.getLogger("app.api").exception(
            "application storage bootstrap failed; serving from local caches only"
        )
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
