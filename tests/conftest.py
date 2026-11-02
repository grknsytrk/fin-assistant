from __future__ import annotations

import pytest

from app.database import close_postgres_pool


@pytest.fixture(autouse=True)
def _isolate_test_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests run against isolated local storage rather than live deployment database."""
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    close_postgres_pool()
    yield
    close_postgres_pool()
