"""Modularized market service helpers.

Each submodule resolves patchable seams and shared mutable state at call time
via ``_market_service()`` / ``_seam(...)`` so ``app.market_service`` remains
the single owner for caches, locks, TTLs, and test monkeypatches.
"""

from __future__ import annotations

from typing import Any


def _market_service() -> Any:
    import app.market_service as ms

    return ms


def _seam(name: str) -> Any:
    return getattr(_market_service(), name)
