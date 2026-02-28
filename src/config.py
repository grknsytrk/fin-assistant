from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML import edilemedi. requirements.txt kurulumunu kontrol edin."
    ) from exc


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "pages_file": "data/processed/pages.jsonl",
        "chunks_v1_file": "data/processed/chunks.jsonl",
        "chunks_v2_file": "data/processed/chunks_v2.jsonl",
        "ui_log_file": "data/processed/ui_logs.jsonl",
    },
    "chroma": {
        "dir": "data/processed/chroma",
        "collection_v1": "bimas_faaliyet_2025",
        "collection_v2": "bimas_faaliyet_2025_v2",
    },
    "chunking": {
        "v1": {
            "chunk_size": 900,
            "overlap": 150,
        },
        "v2": {
            "chunk_size": 900,
            "overlap": 150,
        },
    },
    "retrieval": {
        "top_k_final": 5,
        "v2_top_k_initial": 15,
        "v3_top_k_initial": 20,
        "v5_top_k_vector": 20,
        "v5_top_k_bm25": 20,
        "v6_cross_top_n": 15,
        "alpha_v2": 0.35,
        "alpha_v3": 0.35,
        "beta_v5": 0.60,
    },
    "models": {
        "embedding": "intfloat/multilingual-e5-small",
        "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    "evaluation": {
        "gold_file": "eval/gold_questions.jsonl",
        "gold_multicompany_file": "eval/gold_questions_multicompany.jsonl",
        "detailed_output": "data/processed/eval_metrics_detailed.jsonl",
        "summary_output": "data/processed/eval_metrics_summary.json",
        "week6_summary_output": "data/processed/eval_metrics_week6.json",
        "error_output": "data/processed/error_analysis.jsonl",
        "latency_output": "data/processed/latency_benchmark.json",
        "latency_sample_size": 20,
    },
    "health": {
        "net_margin_green_min": 3.0,
        "net_margin_yellow_min": 1.5,
        "favok_margin_green_min": 5.0,
        "favok_margin_yellow_min": 3.0,
        "revenue_growth_green_min": 3.0,
        "revenue_growth_yellow_min": 0.0,
        "store_growth_green_min": 1.0,
        "store_growth_yellow_min": 0.0,
    },
    "extraction": {
        "metrics_dictionary_file": "data/dictionaries/metrics_tr.yaml",
        "top_candidates": 5,
        "low_confidence_threshold": 0.55,
        "trend_deviation_threshold_pct": 300.0,
        "ratio_self_verify_pp_threshold": 10.0,
        "expected_ranges": {
            "net_kar": {"min": -80_000_000_000, "max": 120_000_000_000},
            "brut_kar": {"min": -120_000_000_000, "max": 500_000_000_000},
            "satis_gelirleri": {"min": 1_000_000_000, "max": 2_000_000_000_000},
            "favok": {"min": -120_000_000_000, "max": 300_000_000_000},
            "faaliyet_nakit_akisi": {"min": -400_000_000_000, "max": 400_000_000_000},
            "capex": {"min": -250_000_000_000, "max": 250_000_000_000},
            "serbest_nakit_akisi": {"min": -400_000_000_000, "max": 400_000_000_000},
            "magaza_sayisi": {"min": 100, "max": 100_000},
            "net_kar_marji": {"min": -80, "max": 80},
            "favok_marji": {"min": -80, "max": 80},
            "brut_kar_marji": {"min": -20, "max": 90},
        },
    },
    "kap": {
        "enabled": True,
        "timeout_seconds": 10.0,
        "cache_ttl_hours": 24.0,
        "user_agent": "ragfin-kap-fetcher/1.0 (+local-first)",
    },
    "llm_commentary": {
        "enabled": False,
        "provider": "openrouter",
        "model": "arcee-ai/trinity-large-preview:free",
        "max_tokens": None,
        "timeout_s": 8.0,
        "temperature": 0.2,
        "reasoning_enabled": True,
    },
    "llm_assistant": {
        "enabled": False,
        "provider": "openrouter",
        "model": "arcee-ai/trinity-large-preview:free",
        "max_tokens": None,
        "timeout_s": 8.0,
        "temperature": 0.2,
        "reasoning_enabled": True,
    },
}
CONFIG_ENV_VAR = "RAGFIN_CONFIG"
_LOADED_DOTENV_FILES: Set[Path] = set()


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    pages_file: Path
    chunks_v1_file: Path
    chunks_v2_file: Path
    ui_log_file: Path


@dataclass(frozen=True)
class ChromaConfig:
    dir: Path
    collection_v1: str
    collection_v2: str


@dataclass(frozen=True)
class ChunkingVersionConfig:
    chunk_size: int
    overlap: int


@dataclass(frozen=True)
class ChunkingConfig:
    v1: ChunkingVersionConfig
    v2: ChunkingVersionConfig


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_final: int
    v2_top_k_initial: int
    v3_top_k_initial: int
    v5_top_k_vector: int
    v5_top_k_bm25: int
    v6_cross_top_n: int
    alpha_v2: float
    alpha_v3: float
    beta_v5: float


@dataclass(frozen=True)
class ModelsConfig:
    embedding: str
    cross_encoder: str


@dataclass(frozen=True)
class EvaluationConfig:
    gold_file: Path
    gold_multicompany_file: Path
    detailed_output: Path
    summary_output: Path
    week6_summary_output: Path
    error_output: Path
    latency_output: Path
    latency_sample_size: int


@dataclass(frozen=True)
class HealthConfig:
    net_margin_green_min: float
    net_margin_yellow_min: float
    favok_margin_green_min: float
    favok_margin_yellow_min: float
    revenue_growth_green_min: float
    revenue_growth_yellow_min: float
    store_growth_green_min: float
    store_growth_yellow_min: float


@dataclass(frozen=True)
class ExtractionConfig:
    metrics_dictionary_file: Path
    top_candidates: int
    low_confidence_threshold: float
    trend_deviation_threshold_pct: float
    ratio_self_verify_pp_threshold: float
    expected_ranges: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class KapConfig:
    enabled: bool
    timeout_seconds: float
    cache_ttl_hours: float
    user_agent: str


@dataclass(frozen=True)
class LLMCommentaryConfig:
    enabled: bool
    provider: str
    model: str
    max_tokens: Optional[int]
    timeout_s: float
    temperature: float
    reasoning_enabled: bool


@dataclass(frozen=True)
class AppConfig:
    path: Path
    paths: PathsConfig
    chroma: ChromaConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    health: HealthConfig
    extraction: ExtractionConfig
    kap: KapConfig
    llm_commentary: LLMCommentaryConfig
    llm_assistant: LLMCommentaryConfig


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_dotenv_file(dotenv_path: Optional[Path] = None) -> Optional[Path]:
    candidates: list[Path] = []
    if dotenv_path is not None:
        candidates.append(Path(dotenv_path))
    else:
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parents[1] / ".env")

    seen: Set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists() or not resolved.is_file():
            continue
        if resolved in _LOADED_DOTENV_FILES:
            return resolved

        with resolved.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if raw.startswith("export "):
                    raw = raw[len("export ") :].strip()
                if "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                os.environ.setdefault(key, value)

        _LOADED_DOTENV_FILES.add(resolved)
        return resolved
    return None


def _as_path(raw: Any, base_dir: Path) -> Path:
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _require_positive_int(name: str, value: Any) -> int:
    payload = int(value)
    if payload <= 0:
        raise ValueError(f"{name} pozitif olmali: {payload}")
    return payload


def _optional_positive_int(name: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"none", "null"}:
        return None
    payload = int(raw)
    if payload == 0:
        return None
    if payload < 0:
        raise ValueError(f"{name} negatif olamaz: {payload}")
    return payload


def _require_non_negative_int(name: str, value: Any) -> int:
    payload = int(value)
    if payload < 0:
        raise ValueError(f"{name} negatif olamaz: {payload}")
    return payload


def _require_positive_float(name: str, value: Any) -> float:
    payload = float(value)
    if payload <= 0:
        raise ValueError(f"{name} pozitif olmali: {payload}")
    return payload


def _require_non_negative_float(name: str, value: Any) -> float:
    payload = float(value)
    if payload < 0:
        raise ValueError(f"{name} negatif olamaz: {payload}")
    return payload


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "on", "evet"}


def _build_config(raw: Dict[str, Any], path: Path) -> AppConfig:
    base_dir = path.parent.resolve()

    paths = raw["paths"]
    chroma = raw["chroma"]
    chunking = raw["chunking"]
    retrieval = raw["retrieval"]
    models = raw["models"]
    evaluation = raw["evaluation"]
    health = raw["health"]
    extraction = raw.get("extraction", {})
    kap = raw.get("kap", {})
    llm_commentary = raw.get("llm_commentary", {})
    llm_assistant = raw.get("llm_assistant", llm_commentary)

    chunk_v1_size = _require_positive_int("chunking.v1.chunk_size", chunking["v1"]["chunk_size"])
    chunk_v1_overlap = _require_non_negative_int("chunking.v1.overlap", chunking["v1"]["overlap"])
    chunk_v2_size = _require_positive_int("chunking.v2.chunk_size", chunking["v2"]["chunk_size"])
    chunk_v2_overlap = _require_non_negative_int("chunking.v2.overlap", chunking["v2"]["overlap"])
    if chunk_v1_overlap >= chunk_v1_size:
        raise ValueError("chunking.v1.overlap, chunk_size'dan kucuk olmali")
    if chunk_v2_overlap >= chunk_v2_size:
        raise ValueError("chunking.v2.overlap, chunk_size'dan kucuk olmali")

    retrieval_cfg = RetrievalConfig(
        top_k_final=_require_positive_int("retrieval.top_k_final", retrieval["top_k_final"]),
        v2_top_k_initial=_require_positive_int("retrieval.v2_top_k_initial", retrieval["v2_top_k_initial"]),
        v3_top_k_initial=_require_positive_int("retrieval.v3_top_k_initial", retrieval["v3_top_k_initial"]),
        v5_top_k_vector=_require_positive_int("retrieval.v5_top_k_vector", retrieval["v5_top_k_vector"]),
        v5_top_k_bm25=_require_positive_int("retrieval.v5_top_k_bm25", retrieval["v5_top_k_bm25"]),
        v6_cross_top_n=_require_positive_int("retrieval.v6_cross_top_n", retrieval["v6_cross_top_n"]),
        alpha_v2=_require_non_negative_float("retrieval.alpha_v2", retrieval["alpha_v2"]),
        alpha_v3=_require_non_negative_float("retrieval.alpha_v3", retrieval["alpha_v3"]),
        beta_v5=_require_non_negative_float("retrieval.beta_v5", retrieval["beta_v5"]),
    )

    health_cfg = HealthConfig(
        net_margin_green_min=float(health["net_margin_green_min"]),
        net_margin_yellow_min=float(health["net_margin_yellow_min"]),
        favok_margin_green_min=float(health["favok_margin_green_min"]),
        favok_margin_yellow_min=float(health["favok_margin_yellow_min"]),
        revenue_growth_green_min=float(health["revenue_growth_green_min"]),
        revenue_growth_yellow_min=float(health["revenue_growth_yellow_min"]),
        store_growth_green_min=float(health["store_growth_green_min"]),
        store_growth_yellow_min=float(health["store_growth_yellow_min"]),
    )
    if health_cfg.net_margin_green_min < health_cfg.net_margin_yellow_min:
        raise ValueError("health.net_margin_green_min, yellow_min'den buyuk/esit olmali")
    if health_cfg.favok_margin_green_min < health_cfg.favok_margin_yellow_min:
        raise ValueError("health.favok_margin_green_min, yellow_min'den buyuk/esit olmali")
    if health_cfg.revenue_growth_green_min < health_cfg.revenue_growth_yellow_min:
        raise ValueError("health.revenue_growth_green_min, yellow_min'den buyuk/esit olmali")
    if health_cfg.store_growth_green_min < health_cfg.store_growth_yellow_min:
        raise ValueError("health.store_growth_green_min, yellow_min'den buyuk/esit olmali")

    extraction_cfg = ExtractionConfig(
        metrics_dictionary_file=_as_path(
            extraction.get("metrics_dictionary_file", DEFAULT_CONFIG["extraction"]["metrics_dictionary_file"]),
            base_dir,
        ),
        top_candidates=_require_positive_int(
            "extraction.top_candidates",
            extraction.get("top_candidates", DEFAULT_CONFIG["extraction"]["top_candidates"]),
        ),
        low_confidence_threshold=_require_non_negative_float(
            "extraction.low_confidence_threshold",
            extraction.get(
                "low_confidence_threshold",
                DEFAULT_CONFIG["extraction"]["low_confidence_threshold"],
            ),
        ),
        trend_deviation_threshold_pct=_require_positive_float(
            "extraction.trend_deviation_threshold_pct",
            extraction.get(
                "trend_deviation_threshold_pct",
                DEFAULT_CONFIG["extraction"]["trend_deviation_threshold_pct"],
            ),
        ),
        ratio_self_verify_pp_threshold=_require_positive_float(
            "extraction.ratio_self_verify_pp_threshold",
            extraction.get(
                "ratio_self_verify_pp_threshold",
                DEFAULT_CONFIG["extraction"]["ratio_self_verify_pp_threshold"],
            ),
        ),
        expected_ranges={},
    )
    if extraction_cfg.low_confidence_threshold > 1.0:
        raise ValueError("extraction.low_confidence_threshold 0..1 araliginda olmali")
    if extraction_cfg.ratio_self_verify_pp_threshold > 100.0:
        raise ValueError("extraction.ratio_self_verify_pp_threshold makul aralikta olmali")

    expected_ranges_raw = extraction.get(
        "expected_ranges",
        DEFAULT_CONFIG["extraction"].get("expected_ranges", {}),
    )
    expected_ranges: Dict[str, Dict[str, float]] = {}
    if isinstance(expected_ranges_raw, dict):
        for metric_name, bounds in expected_ranges_raw.items():
            if not isinstance(bounds, dict):
                continue
            if "min" not in bounds or "max" not in bounds:
                continue
            min_val = float(bounds["min"])
            max_val = float(bounds["max"])
            if min_val > max_val:
                raise ValueError(f"extraction.expected_ranges.{metric_name} min > max")
            expected_ranges[str(metric_name)] = {"min": min_val, "max": max_val}

    extraction_cfg = ExtractionConfig(
        metrics_dictionary_file=extraction_cfg.metrics_dictionary_file,
        top_candidates=extraction_cfg.top_candidates,
        low_confidence_threshold=extraction_cfg.low_confidence_threshold,
        trend_deviation_threshold_pct=extraction_cfg.trend_deviation_threshold_pct,
        ratio_self_verify_pp_threshold=extraction_cfg.ratio_self_verify_pp_threshold,
        expected_ranges=expected_ranges,
    )

    kap_defaults = DEFAULT_CONFIG["kap"]
    kap_enabled = _as_bool(kap.get("enabled", kap_defaults["enabled"]))
    kap_timeout_seconds = _require_positive_float(
        "kap.timeout_seconds",
        kap.get("timeout_seconds", kap_defaults["timeout_seconds"]),
    )
    kap_cache_ttl_hours = _require_positive_float(
        "kap.cache_ttl_hours",
        kap.get("cache_ttl_hours", kap_defaults["cache_ttl_hours"]),
    )
    kap_user_agent = str(kap.get("user_agent", kap_defaults["user_agent"])).strip() or str(
        kap_defaults["user_agent"]
    )
    if kap_timeout_seconds > 120.0:
        raise ValueError("kap.timeout_seconds 1..120 araliginda olmali")
    if kap_cache_ttl_hours > 24 * 30:
        raise ValueError("kap.cache_ttl_hours makul aralikta olmali (<= 720)")

    env_kap_enabled = os.getenv("RAGFIN_KAP_ENABLED", "").strip()
    if env_kap_enabled:
        kap_enabled = _as_bool(env_kap_enabled)
    env_kap_timeout = os.getenv("RAGFIN_KAP_TIMEOUT_SECONDS", "").strip()
    if env_kap_timeout:
        kap_timeout_seconds = _require_positive_float("RAGFIN_KAP_TIMEOUT_SECONDS", env_kap_timeout)
    env_kap_ttl = os.getenv("RAGFIN_KAP_CACHE_TTL_HOURS", "").strip()
    if env_kap_ttl:
        kap_cache_ttl_hours = _require_positive_float("RAGFIN_KAP_CACHE_TTL_HOURS", env_kap_ttl)
    kap_user_agent = os.getenv("RAGFIN_KAP_USER_AGENT", "").strip() or kap_user_agent

    if kap_timeout_seconds > 120.0:
        raise ValueError("RAGFIN_KAP_TIMEOUT_SECONDS 1..120 araliginda olmali")
    if kap_cache_ttl_hours > 24 * 30:
        raise ValueError("RAGFIN_KAP_CACHE_TTL_HOURS makul aralikta olmali (<= 720)")

    kap_cfg = KapConfig(
        enabled=kap_enabled,
        timeout_seconds=kap_timeout_seconds,
        cache_ttl_hours=kap_cache_ttl_hours,
        user_agent=kap_user_agent,
    )

    llm_defaults = DEFAULT_CONFIG.get("llm_assistant", DEFAULT_CONFIG["llm_commentary"])
    llm_enabled = _as_bool(llm_assistant.get("enabled", llm_defaults["enabled"]))
    llm_provider = str(llm_assistant.get("provider", llm_defaults["provider"])).strip()
    llm_model = str(llm_assistant.get("model", llm_defaults["model"])).strip()
    llm_max_tokens = _optional_positive_int(
        "llm_assistant.max_tokens",
        llm_assistant.get("max_tokens", llm_defaults["max_tokens"]),
    )
    llm_timeout_s = _require_positive_float(
        "llm_assistant.timeout_s",
        llm_assistant.get("timeout_s", llm_defaults.get("timeout_s", 8.0)),
    )
    llm_temperature = _require_non_negative_float(
        "llm_assistant.temperature",
        llm_assistant.get("temperature", llm_defaults["temperature"]),
    )
    llm_reasoning_enabled = _as_bool(
        llm_assistant.get(
            "reasoning_enabled",
            llm_defaults["reasoning_enabled"],
        )
    )
    if llm_timeout_s > 120.0:
        raise ValueError("llm_assistant.timeout_s 1..120 araliginda olmali")
    if llm_temperature > 2.0:
        raise ValueError("llm_assistant.temperature 0..2 araliginda olmali")

    env_enabled = os.getenv("RAGFIN_LLM_ASSISTANT_ENABLED", "").strip() or os.getenv(
        "RAGFIN_LLM_COMMENTARY_ENABLED", ""
    ).strip()
    if env_enabled:
        llm_enabled = _as_bool(env_enabled)
    llm_provider = (
        os.getenv("RAGFIN_LLM_ASSISTANT_PROVIDER", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_PROVIDER", "").strip()
        or llm_provider
    )
    llm_model = (
        os.getenv("RAGFIN_LLM_ASSISTANT_MODEL", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_MODEL", "").strip()
        or llm_model
    )
    env_max_tokens = (
        os.getenv("RAGFIN_LLM_ASSISTANT_MAX_TOKENS", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_MAX_TOKENS", "").strip()
    )
    if env_max_tokens:
        llm_max_tokens = _optional_positive_int(
            "RAGFIN_LLM_ASSISTANT_MAX_TOKENS",
            env_max_tokens,
        )
    env_timeout = (
        os.getenv("RAGFIN_LLM_ASSISTANT_TIMEOUT_S", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_TIMEOUT_S", "").strip()
    )
    if env_timeout:
        llm_timeout_s = _require_positive_float("RAGFIN_LLM_ASSISTANT_TIMEOUT_S", env_timeout)
    llm_temperature = _require_non_negative_float(
        "RAGFIN_LLM_ASSISTANT_TEMPERATURE",
        os.getenv(
            "RAGFIN_LLM_ASSISTANT_TEMPERATURE",
            os.getenv("RAGFIN_LLM_COMMENTARY_TEMPERATURE", str(llm_temperature)),
        ),
    )
    if llm_timeout_s > 120.0:
        raise ValueError("RAGFIN_LLM_ASSISTANT_TIMEOUT_S 1..120 araliginda olmali")
    if llm_temperature > 2.0:
        raise ValueError("RAGFIN_LLM_ASSISTANT_TEMPERATURE 0..2 araliginda olmali")
    env_reasoning_enabled = (
        os.getenv("RAGFIN_LLM_ASSISTANT_REASONING_ENABLED", "").strip()
        or os.getenv("RAGFIN_LLM_COMMENTARY_REASONING_ENABLED", "").strip()
    )
    if env_reasoning_enabled:
        llm_reasoning_enabled = _as_bool(env_reasoning_enabled)

    llm_cfg = LLMCommentaryConfig(
        enabled=llm_enabled,
        provider=llm_provider,
        model=llm_model,
        max_tokens=llm_max_tokens,
        timeout_s=llm_timeout_s,
        temperature=llm_temperature,
        reasoning_enabled=llm_reasoning_enabled,
    )

    return AppConfig(
        path=path.resolve(),
        paths=PathsConfig(
            raw_dir=_as_path(paths["raw_dir"], base_dir),
            processed_dir=_as_path(paths["processed_dir"], base_dir),
            pages_file=_as_path(paths["pages_file"], base_dir),
            chunks_v1_file=_as_path(paths["chunks_v1_file"], base_dir),
            chunks_v2_file=_as_path(paths["chunks_v2_file"], base_dir),
            ui_log_file=_as_path(paths["ui_log_file"], base_dir),
        ),
        chroma=ChromaConfig(
            dir=_as_path(chroma["dir"], base_dir),
            collection_v1=str(chroma["collection_v1"]),
            collection_v2=str(chroma["collection_v2"]),
        ),
        chunking=ChunkingConfig(
            v1=ChunkingVersionConfig(chunk_size=chunk_v1_size, overlap=chunk_v1_overlap),
            v2=ChunkingVersionConfig(chunk_size=chunk_v2_size, overlap=chunk_v2_overlap),
        ),
        retrieval=retrieval_cfg,
        models=ModelsConfig(
            embedding=str(models["embedding"]),
            cross_encoder=str(models["cross_encoder"]),
        ),
        evaluation=EvaluationConfig(
            gold_file=_as_path(evaluation["gold_file"], base_dir),
            gold_multicompany_file=_as_path(
                evaluation.get("gold_multicompany_file", DEFAULT_CONFIG["evaluation"]["gold_multicompany_file"]),
                base_dir,
            ),
            detailed_output=_as_path(evaluation["detailed_output"], base_dir),
            summary_output=_as_path(evaluation["summary_output"], base_dir),
            week6_summary_output=_as_path(evaluation["week6_summary_output"], base_dir),
            error_output=_as_path(evaluation["error_output"], base_dir),
            latency_output=_as_path(evaluation["latency_output"], base_dir),
            latency_sample_size=_require_positive_int("evaluation.latency_sample_size", evaluation["latency_sample_size"]),
        ),
        health=health_cfg,
        extraction=extraction_cfg,
        kap=kap_cfg,
        llm_commentary=llm_cfg,
        llm_assistant=llm_cfg,
    )


def resolve_config_path(config_path: Optional[Path] = None) -> Path:
    if config_path is not None:
        cfg_candidate = Path(config_path)
        cfg_parent = cfg_candidate if cfg_candidate.is_dir() else cfg_candidate.parent
        if str(cfg_parent).strip():
            load_dotenv_file(cfg_parent / ".env")
    load_dotenv_file()

    env_override = os.getenv(CONFIG_ENV_VAR, "").strip()
    if env_override:
        return Path(env_override).resolve()
    return (config_path or Path("config.yaml")).resolve()


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    path = resolve_config_path(config_path)
    payload = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config dosyasi root seviyede object olmali")
        payload = _deep_merge(payload, loaded)
    return _build_config(payload, path)
