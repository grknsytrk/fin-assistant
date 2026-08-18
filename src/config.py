from __future__ import annotations

import copy
import os
import re
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
        "processed_dir": "data/processed",
    },
    "kap": {
        "enabled": True,
        "timeout_seconds": 10.0,
        "cache_ttl_hours": 0.0,
        "user_agent": "ragfin-kap-fetcher/1.0 (+local-first)",
        "api_key": "",
        "api_secret": "",
        "vyk_base_url": "https://apigwdev.mkk.com.tr/api/vyk",
        "vyk_auth_mode": "auto",
        "vyk_token_url": "",
    },
}

CONFIG_ENV_VAR = "RAGFIN_CONFIG"
_LOADED_DOTENV_FILES: Set[Path] = set()


@dataclass(frozen=True)
class PathsConfig:
    processed_dir: Path


@dataclass(frozen=True)
class KapConfig:
    enabled: bool
    timeout_seconds: float
    cache_ttl_hours: float
    user_agent: str
    api_key: str
    api_secret: str
    vyk_base_url: str
    vyk_auth_mode: str
    vyk_token_url: str


@dataclass(frozen=True)
class AppConfig:
    path: Path
    paths: PathsConfig
    kap: KapConfig


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
                if "=" in raw:
                    key, value = raw.split("=", 1)
                elif ":" in raw:
                    key, value = raw.split(":", 1)
                    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key.strip()):
                        continue
                else:
                    continue
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "evet"}


def _positive_float(name: str, value: Any) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} pozitif olmali: {parsed}")
    return parsed


def _non_negative_float(name: str, value: Any) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} negatif olamaz: {parsed}")
    return parsed


def _build_config(raw: Dict[str, Any], path: Path) -> AppConfig:
    base_dir = path.parent.resolve()
    paths = raw.get("paths") or {}
    kap = raw.get("kap") or {}
    kap_defaults = DEFAULT_CONFIG["kap"]

    timeout_seconds = _positive_float(
        "kap.timeout_seconds",
        kap.get("timeout_seconds", kap_defaults["timeout_seconds"]),
    )
    cache_ttl_hours = _non_negative_float(
        "kap.cache_ttl_hours",
        kap.get("cache_ttl_hours", kap_defaults["cache_ttl_hours"]),
    )
    if timeout_seconds > 120.0:
        raise ValueError("kap.timeout_seconds 1..120 araliginda olmali")
    if cache_ttl_hours > 24 * 30:
        raise ValueError("kap.cache_ttl_hours makul aralikta olmali (<= 720; 0=canli KAP)")

    kap_enabled = _as_bool(kap.get("enabled", kap_defaults["enabled"]))
    env_enabled = os.getenv("RAGFIN_KAP_ENABLED", "").strip()
    if env_enabled:
        kap_enabled = _as_bool(env_enabled)
    env_timeout = os.getenv("RAGFIN_KAP_TIMEOUT_SECONDS", "").strip()
    if env_timeout:
        timeout_seconds = _positive_float("RAGFIN_KAP_TIMEOUT_SECONDS", env_timeout)
    env_ttl = os.getenv("RAGFIN_KAP_CACHE_TTL_HOURS", "").strip()
    if env_ttl:
        cache_ttl_hours = _non_negative_float("RAGFIN_KAP_CACHE_TTL_HOURS", env_ttl)
    if timeout_seconds > 120.0:
        raise ValueError("RAGFIN_KAP_TIMEOUT_SECONDS 1..120 araliginda olmali")
    if cache_ttl_hours > 24 * 30:
        raise ValueError("RAGFIN_KAP_CACHE_TTL_HOURS makul aralikta olmali (<= 720; 0=canli KAP)")

    user_agent = str(kap.get("user_agent", kap_defaults["user_agent"])).strip() or kap_defaults[
        "user_agent"
    ]
    api_key = str(kap.get("api_key", kap_defaults["api_key"])).strip()
    api_secret = str(kap.get("api_secret", kap_defaults["api_secret"])).strip()
    vyk_base_url = str(kap.get("vyk_base_url", kap_defaults["vyk_base_url"])).strip().rstrip("/")
    vyk_auth_mode = str(kap.get("vyk_auth_mode", kap_defaults["vyk_auth_mode"])).strip().lower() or "auto"
    vyk_token_url = str(kap.get("vyk_token_url", kap_defaults["vyk_token_url"])).strip().rstrip("/")

    api_key = os.getenv("RAGFIN_KAP_API_KEY", "").strip() or api_key
    api_secret = os.getenv("RAGFIN_KAP_API_SECRET", "").strip() or api_secret
    vyk_base_url = os.getenv("RAGFIN_KAP_VYK_BASE_URL", "").strip().rstrip("/") or vyk_base_url
    vyk_auth_mode = os.getenv("RAGFIN_KAP_VYK_AUTH_MODE", "").strip().lower() or vyk_auth_mode
    vyk_token_url = os.getenv("RAGFIN_KAP_VYK_TOKEN_URL", "").strip().rstrip("/") or vyk_token_url
    if vyk_auth_mode not in {"auto", "basic", "token"}:
        raise ValueError("RAGFIN_KAP_VYK_AUTH_MODE auto/basic/token olmali")

    return AppConfig(
        path=path.resolve(),
        paths=PathsConfig(
            processed_dir=_as_path(
                paths.get("processed_dir", DEFAULT_CONFIG["paths"]["processed_dir"]),
                base_dir,
            )
        ),
        kap=KapConfig(
            enabled=kap_enabled,
            timeout_seconds=timeout_seconds,
            cache_ttl_hours=cache_ttl_hours,
            user_agent=user_agent,
            api_key=api_key,
            api_secret=api_secret,
            vyk_base_url=vyk_base_url,
            vyk_auth_mode=vyk_auth_mode,
            vyk_token_url=vyk_token_url,
        ),
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
