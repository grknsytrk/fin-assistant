from __future__ import annotations

import argparse
import importlib
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import AppConfig, load_config

MIN_PY = (3, 9)
MAX_PY = (3, 13)  # exclusive


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    action: str = ""


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="PASS", detail=detail)


def _fail(name: str, detail: str, action: str) -> CheckResult:
    return CheckResult(name=name, status="FAIL", detail=detail, action=action)


def _warn(name: str, detail: str, action: str = "") -> CheckResult:
    return CheckResult(name=name, status="WARN", detail=detail, action=action)


def _check_python() -> CheckResult:
    import sys

    current = sys.version_info[:2]
    if MIN_PY <= current < MAX_PY:
        return _ok("python_version", f"{current[0]}.{current[1]}")
    return _fail(
        "python_version",
        f"{current[0]}.{current[1]}",
        "Python 3.9-3.12 kullanin (ornek: .venv39).",
    )


def _check_import(module_name: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
        return _ok(f"import_{module_name}", "import edildi")
    except Exception as exc:
        return _fail(
            f"import_{module_name}",
            f"import hatasi: {exc}",
            f"`pip install -r requirements.txt` ile `{module_name}` kurulumunu dogrulayin.",
        )


def _check_model_local(cfg: AppConfig) -> CheckResult:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        return _fail(
            "embedding_model_local",
            f"sentence-transformers import edilemedi: {exc}",
            "requirements kurulumunu tamamlayin.",
        )

    model_name = cfg.models.embedding
    try:
        SentenceTransformer(model_name, local_files_only=True)
        return _ok("embedding_model_local", f"lokalde hazir: {model_name}")
    except Exception as exc:
        return _warn(
            "embedding_model_local",
            f"lokal model bulunamadi: {model_name} ({exc})",
            "Ilk indexleme sirasinda model indirilecek. Offline ortam icin modeli onceden cache edin.",
        )


def _check_path_writable(path: Path, label: str) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="ragfin_doctor_", dir=str(path), delete=True):
            pass
        return _ok(label, str(path))
    except Exception as exc:
        return _fail(label, f"yazilamiyor: {path} ({exc})", f"Klasor izinlerini kontrol edin: {path}")


def _check_chroma(cfg: AppConfig) -> CheckResult:
    try:
        import chromadb
    except Exception as exc:
        return _fail("chroma_collection", f"chromadb import hatasi: {exc}", "chromadb kurulumunu dogrulayin.")

    try:
        cfg.chroma.dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(cfg.chroma.dir))
        temp_name = f"doctor_tmp_{int(time.time())}"
        collection = client.get_or_create_collection(name=temp_name)
        collection.add(
            ids=["doctor_test_id"],
            documents=["doctor test"],
            metadatas=[{"source": "doctor"}],
            embeddings=[[0.1, 0.2, 0.3]],
        )
        count = collection.count()
        client.delete_collection(name=temp_name)
        return _ok("chroma_collection", f"test collection olusturuldu (count={count})")
    except Exception as exc:
        return _fail(
            "chroma_collection",
            f"collection test basarisiz: {exc}",
            f"Chroma klasorunu ve sqlite izinlerini kontrol edin: {cfg.chroma.dir}",
        )


def run_doctor(config_path: Optional[Path] = None) -> List[CheckResult]:
    cfg = load_config(config_path)
    results: List[CheckResult] = []
    results.append(_check_python())
    results.append(_check_import("chromadb"))
    results.append(_check_import("sentence_transformers"))
    results.append(_check_model_local(cfg))
    results.append(_check_path_writable(cfg.paths.raw_dir, "path_raw_dir"))
    results.append(_check_path_writable(cfg.paths.processed_dir, "path_processed_dir"))
    results.append(_check_path_writable(cfg.chroma.dir, "path_chroma_dir"))
    results.append(_check_chroma(cfg))
    return results


def _print_table(results: List[CheckResult]) -> None:
    print("RAG-Fin Doctor")
    print("-" * 96)
    print(f"{'CHECK':30} {'STATUS':8} {'DETAIL':54}")
    print("-" * 96)
    for row in results:
        detail = row.detail[:54]
        print(f"{row.name[:30]:30} {row.status[:8]:8} {detail:54}")
    print("-" * 96)
    actions = [row for row in results if row.status != "PASS" and row.action]
    if actions:
        print("Actionable next steps:")
        for row in actions:
            print(f"- [{row.name}] {row.action}")
    else:
        print("Tum kontroller basarili.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-Fin doctor checks")
    parser.add_argument("--config", default=os.getenv("RAGFIN_CONFIG", "config.yaml"))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_doctor(Path(args.config))
    _print_table(results)
    has_fail = any(item.status == "FAIL" for item in results)
    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
