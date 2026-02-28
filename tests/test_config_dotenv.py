from __future__ import annotations

from pathlib import Path

from src import config as config_module


def test_load_dotenv_file_sets_env(monkeypatch, tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OPENROUTER_API_KEY=test-key\n"
        "RAGFIN_LLM_COMMENTARY_ENABLED=true\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RAGFIN_LLM_COMMENTARY_ENABLED", raising=False)
    config_module._LOADED_DOTENV_FILES.clear()

    loaded = config_module.load_dotenv_file(dotenv)
    assert loaded is not None
    assert loaded.resolve() == dotenv.resolve()
    assert config_module.os.getenv("OPENROUTER_API_KEY") == "test-key"
    assert config_module.os.getenv("RAGFIN_LLM_COMMENTARY_ENABLED") == "true"


def test_resolve_config_path_reads_dotenv_override(monkeypatch, tmp_path: Path) -> None:
    override_cfg = tmp_path / "override.yaml"
    override_cfg.write_text("paths:\n  raw_dir: data/raw\n", encoding="utf-8")
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"RAGFIN_CONFIG={override_cfg}\n", encoding="utf-8")

    monkeypatch.delenv("RAGFIN_CONFIG", raising=False)
    config_module._LOADED_DOTENV_FILES.clear()

    resolved = config_module.resolve_config_path(tmp_path / "config.yaml")
    assert resolved == override_cfg.resolve()

