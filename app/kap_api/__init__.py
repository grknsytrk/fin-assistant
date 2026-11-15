"""Modularized KAP API service helpers.

Each submodule resolves patchable seams and shared mutable state at call time
via ``_kas()`` so ``app.kap_api_service`` remains the single owner for caches,
locks, TTLs, and test monkeypatches.
"""

from __future__ import annotations

import sys
from typing import Any


def _kas() -> Any:
    if "app.kap_api_service" in sys.modules:
        return sys.modules["app.kap_api_service"]
    import app.kap_api_service as kas

    return kas
