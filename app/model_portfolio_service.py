"""Model Portfolio persistence, enrichment, and cached performance calculations.

Designed to reuse existing in-memory and shared caches (`_STOCKS_CACHE`,
`_STOCK_CARD_VALUATION_CACHE`, and `_comparison_history_cached_asset`) without
triggering duplicate upstream requests.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.database import read_json_cache, write_json_cache

LOGGER = logging.getLogger(__name__)

import re
from urllib.parse import unquote

_STATE_LOCK = threading.RLock()
_MODEL_PORTFOLIO_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
_MODEL_PORTFOLIO_PERF_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
_PERF_MEMORY_TTL_SECONDS = 15 * 60

_AUTH_LOCK = threading.Lock()
_FAILED_ADMIN_ATTEMPTS: Dict[str, List[float]] = {}
_ADMIN_MAX_FAILED_ATTEMPTS = 5
_ADMIN_LOCKOUT_WINDOW_SECONDS = 10 * 60
_SYMBOL_RE = re.compile(r"^[A-Z0-9.]{1,12}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    return date.today().isoformat()


def _state_file_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / "model_portfolio" / "state.json"


def _cache_scope_key(processed_dir: Path) -> str:
    try:
        return str(Path(processed_dir).resolve())
    except Exception:
        return str(processed_dir)


def _clean_token_value(raw: Optional[str]) -> str:
    val = unquote(str(raw or "").strip()).strip()
    if val.upper().startswith("RAGFIN_ADMIN_TOKEN="):
        val = val.split("=", 1)[1].strip()
    elif val.upper().startswith("RAGFIN_ADMIN_REFRESH_TOKEN="):
        val = val.split("=", 1)[1].strip()
    return val.strip('"').strip("'").strip()


def _read_dotenv_admin_token() -> str:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return ""
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() in {"RAGFIN_ADMIN_TOKEN", "RAGFIN_ADMIN_REFRESH_TOKEN"}:
                cleaned = _clean_token_value(val)
                if cleaned:
                    return cleaned
    except Exception:
        pass
    return ""


def expected_admin_token() -> str:
    primary = _clean_token_value(os.getenv("RAGFIN_ADMIN_TOKEN"))
    if primary:
        return primary
    dotenv_token = _read_dotenv_admin_token()
    if dotenv_token:
        return dotenv_token
    secondary = _clean_token_value(os.getenv("RAGFIN_ADMIN_REFRESH_TOKEN"))
    if secondary:
        return secondary
    return ""


def is_admin_locked_out(client_ip: str) -> bool:
    ip = str(client_ip or "unknown").strip() or "unknown"
    now = time.monotonic()
    with _AUTH_LOCK:
        attempts = [ts for ts in _FAILED_ADMIN_ATTEMPTS.get(ip, []) if now - ts < _ADMIN_LOCKOUT_WINDOW_SECONDS]
        if attempts:
            _FAILED_ADMIN_ATTEMPTS[ip] = attempts
        else:
            _FAILED_ADMIN_ATTEMPTS.pop(ip, None)
        return len(attempts) >= _ADMIN_MAX_FAILED_ATTEMPTS


def record_admin_auth_result(client_ip: str, success: bool) -> None:
    ip = str(client_ip or "unknown").strip() or "unknown"
    now = time.monotonic()
    with _AUTH_LOCK:
        if success:
            _FAILED_ADMIN_ATTEMPTS.pop(ip, None)
            return
        attempts = [ts for ts in _FAILED_ADMIN_ATTEMPTS.get(ip, []) if now - ts < _ADMIN_LOCKOUT_WINDOW_SECONDS]
        attempts.append(now)
        _FAILED_ADMIN_ATTEMPTS[ip] = attempts


def reset_admin_rate_limit_for_tests() -> None:
    with _AUTH_LOCK:
        _FAILED_ADMIN_ATTEMPTS.clear()


def verify_admin_token(token: Optional[str]) -> bool:
    candidate = _clean_token_value(token)
    if not candidate:
        return False
    expected = expected_admin_token()
    if not expected:
        return False
    return secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def _default_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _now_iso(),
        "positions": [],
        "changelog": [],
    }


def _normalize_state(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return _default_state()
    positions = raw.get("positions")
    changelog = raw.get("changelog")
    version = raw.get("version")
    try:
        version_int = max(1, int(version))
    except (TypeError, ValueError):
        version_int = 1
    return {
        "version": version_int,
        "updated_at": str(raw.get("updated_at") or _now_iso()),
        "positions": [dict(item) for item in positions if isinstance(item, dict)] if isinstance(positions, list) else [],
        "changelog": [dict(item) for item in changelog if isinstance(item, dict)] if isinstance(changelog, list) else [],
    }


def load_model_portfolio_state(processed_dir: Path, *, force_reload: bool = False) -> Dict[str, Any]:
    scope = _cache_scope_key(processed_dir)
    with _STATE_LOCK:
        if not force_reload and scope in _MODEL_PORTFOLIO_STATE_CACHE:
            cached = _MODEL_PORTFOLIO_STATE_CACHE[scope]
            return {
                "version": cached["version"],
                "updated_at": cached["updated_at"],
                "positions": [dict(p) for p in cached["positions"]],
                "changelog": [dict(c) for c in cached["changelog"]],
            }

        path = _state_file_path(processed_dir)
        local_state: Optional[Dict[str, Any]] = None
        if path.exists():
            try:
                local_state = _normalize_state(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                LOGGER.warning("Failed to read local model portfolio state from %s", path, exc_info=True)

        remote_raw = read_json_cache(path)
        remote_state = _normalize_state(remote_raw) if isinstance(remote_raw, dict) else None

        state: Optional[Dict[str, Any]] = local_state
        if remote_state is not None:
            if local_state is None or (
                int(remote_state.get("version") or 0),
                str(remote_state.get("updated_at") or ""),
            ) > (
                int(local_state.get("version") or 0),
                str(local_state.get("updated_at") or ""),
            ):
                state = remote_state
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    LOGGER.debug("Could not mirror remote model portfolio state locally", exc_info=True)
            elif (
                int(local_state.get("version") or 0),
                str(local_state.get("updated_at") or ""),
            ) > (
                int(remote_state.get("version") or 0),
                str(remote_state.get("updated_at") or ""),
            ):
                threading.Thread(target=write_json_cache, args=(path, local_state), daemon=True).start()

        if state is None:
            state = _default_state()

        _MODEL_PORTFOLIO_STATE_CACHE[scope] = state
        return {
            "version": state["version"],
            "updated_at": state["updated_at"],
            "positions": [dict(p) for p in state["positions"]],
            "changelog": [dict(c) for c in state["changelog"]],
        }


def _save_model_portfolio_state(processed_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    scope = _cache_scope_key(processed_dir)
    path = _state_file_path(processed_dir)
    normalized = _normalize_state(state)
    normalized["version"] = int(normalized.get("version") or 1) + 1
    normalized["updated_at"] = _now_iso()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)

    _MODEL_PORTFOLIO_STATE_CACHE[scope] = normalized
    threading.Thread(
        target=write_json_cache,
        args=(path, dict(normalized)),
        name="model-portfolio-remote-sync",
        daemon=True,
    ).start()
    return {
        "version": normalized["version"],
        "updated_at": normalized["updated_at"],
        "positions": [dict(p) for p in normalized["positions"]],
        "changelog": [dict(c) for c in normalized["changelog"]],
    }


def reset_model_portfolio_cache_for_tests() -> None:
    with _STATE_LOCK:
        _MODEL_PORTFOLIO_STATE_CACHE.clear()
        _MODEL_PORTFOLIO_PERF_MEMORY_CACHE.clear()
    reset_admin_rate_limit_for_tests()


def get_active_model_portfolio_symbols(processed_dir: Path) -> Dict[str, Any]:
    state = load_model_portfolio_state(processed_dir)
    symbols: List[str] = []
    seen = set()
    for pos in state["positions"]:
        if str(pos.get("status") or "active") != "active":
            continue
        sym = str(pos.get("symbol") or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    return {
        "symbols": symbols,
        "count": len(symbols),
        "version": state["version"],
        "updated_at": state["updated_at"],
    }


def _safe_float(val: Any) -> Optional[float]:
    if isinstance(val, bool) or val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if not (num == num) or num in (float("inf"), float("-inf")):
        return None
    return num


def _positive_float(val: Any) -> Optional[float]:
    num = _safe_float(val)
    return num if num is not None and num > 0 else None


def compound_relative_return_pct(
    asset_return_pct: Optional[float],
    benchmark_return_pct: Optional[float],
) -> Optional[float]:
    """Compute institutional compound relative return (Alpha): ((1+Ra)/(1+Rb) - 1)*100."""
    if asset_return_pct is None or benchmark_return_pct is None:
        return None
    denom = 1.0 + (benchmark_return_pct / 100.0)
    if denom <= 0:
        return None
    numer = 1.0 + (asset_return_pct / 100.0)
    return round(((numer / denom) - 1.0) * 100.0, 2)


def compute_return_pct(current_price: Optional[float], base_price: Optional[float]) -> Optional[float]:
    if current_price is None or base_price is None or current_price <= 0 or base_price <= 0:
        return None
    return round(((current_price - base_price) / base_price) * 100.0, 2)


def compute_target_progress_pct(
    entry_price: Optional[float],
    current_price: Optional[float],
    target_price: Optional[float],
) -> Optional[float]:
    if entry_price is None or current_price is None or target_price is None:
        return None
    span = target_price - entry_price
    if abs(span) < 1e-6:
        return 100.0 if current_price >= target_price else 0.0
    raw = ((current_price - entry_price) / span) * 100.0
    return round(max(-100.0, min(150.0, raw)), 1)


def _holding_days(entry_date_str: Optional[str], end_date_str: Optional[str] = None) -> int:
    if not entry_date_str:
        return 0
    try:
        start_d = date.fromisoformat(str(entry_date_str)[:10])
    except ValueError:
        return 0
    if end_date_str:
        try:
            end_d = date.fromisoformat(str(end_date_str)[:10])
        except ValueError:
            end_d = date.today()
    else:
        end_d = date.today()
    return max(0, (end_d - start_d).days)


def _normalize_catalysts(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    result: List[str] = []
    seen = set()
    for item in raw:
        label = " ".join(str(item or "").split()).strip()
        if label and label.casefold() not in seen:
            seen.add(label.casefold())
            result.append(label[:48])
        if len(result) >= 8:
            break
    return result


def _normalize_multiples(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"fk": None, "fd_favok": None, "pd_dd": None, "net_borc_favok": None, "period": None}
    return {
        "fk": _safe_float(raw.get("fk")),
        "fd_favok": _safe_float(raw.get("fd_favok")),
        "pd_dd": _safe_float(raw.get("pd_dd")),
        "net_borc_favok": _safe_float(raw.get("net_borc_favok")),
        "period": str(raw.get("period") or raw.get("valuation_financial_period") or "").strip() or None,
    }


def create_position(
    processed_dir: Path,
    payload: Dict[str, Any],
    *,
    company_name: Optional[str] = None,
    logo_url: Optional[str] = None,
    logo_source: Optional[str] = None,
    entry_multiples: Optional[Dict[str, Any]] = None,
    entry_xu100: Optional[float] = None,
) -> Dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol or not _SYMBOL_RE.match(symbol):
        raise ValueError("Geçerli bir hisse kodu (symbol) girilmelidir.")

    entry_price = _positive_float(payload.get("entry_price"))
    if entry_price is None:
        raise ValueError("Giriş fiyatı (entry_price) 0'dan büyük olmalıdır.")

    target_price = _positive_float(payload.get("target_price"))
    if target_price is None:
        raise ValueError("Hedef fiyat (target_price) 0'dan büyük olmalıdır.")

    weight_pct = _positive_float(payload.get("weight_pct"))
    if weight_pct is None or weight_pct > 100.0:
        raise ValueError("Portföy ağırlığı (weight_pct) 0 ile 100 arasında olmalıdır.")

    entry_date_raw = str(payload.get("entry_date") or _today_iso()).strip()[:10]
    try:
        date.fromisoformat(entry_date_raw)
    except ValueError as exc:
        raise ValueError("Giriş tarihi YYYY-AA-GG formatında olmalıdır.") from exc

    with _STATE_LOCK:
        state = load_model_portfolio_state(processed_dir)
        active_positions = [p for p in state["positions"] if p.get("status") == "active"]
        if any(str(p.get("symbol") or "").upper() == symbol for p in active_positions):
            raise ValueError(f"{symbol} zaten aktif model portföyde yer alıyor.")

        current_total_weight = sum(float(p.get("weight_pct") or 0.0) for p in active_positions)
        if round(current_total_weight + weight_pct, 4) > 100.0:
            remaining = max(0.0, round(100.0 - current_total_weight, 2))
            raise ValueError(
                f"Toplam portföy ağırlığı %100'ü aşamaz. Kalan kullanılabilir ağırlık: %{remaining}"
            )

        now = _now_iso()
        position_id = uuid.uuid4().hex[:12]
        resolved_xu100 = _positive_float(payload.get("entry_xu100")) or _positive_float(entry_xu100)
        resolved_multiples = _normalize_multiples(payload.get("entry_multiples") or entry_multiples)

        position = {
            "id": position_id,
            "symbol": symbol,
            "company_name": (str(payload.get("company_name") or company_name or symbol).strip() or symbol)[:120],
            "sector": (str(payload.get("sector") or "").strip()[:64]) or None,
            "logo_url": logo_url,
            "logo_source": logo_source,
            "status": "active",
            "entry_date": entry_date_raw,
            "entry_price": round(entry_price, 4),
            "entry_xu100": round(resolved_xu100, 2) if resolved_xu100 else None,
            "weight_pct": round(weight_pct, 2),
            "target_price": round(target_price, 4),
            "horizon": str(payload.get("horizon") or "6-12 Ay").strip()[:32] or "6-12 Ay",
            "rating": str(payload.get("rating") or "Endeks Üstü Getiri").strip()[:48] or "Endeks Üstü Getiri",
            "thesis": str(payload.get("thesis") or "").strip()[:4000],
            "catalysts": _normalize_catalysts(payload.get("catalysts")),
            "risk_note": str(payload.get("risk_note") or "").strip()[:1500],
            "entry_multiples": resolved_multiples,
            "revisions": [],
            "closed_date": None,
            "exit_price": None,
            "exit_xu100": None,
            "close_reason": None,
            "created_at": now,
            "updated_at": now,
        }

        changelog_entry = {
            "id": uuid.uuid4().hex[:12],
            "date": entry_date_raw,
            "created_at": now,
            "action": "added",
            "symbol": symbol,
            "company_name": position["company_name"],
            "summary": (
                f"{symbol} %{position['weight_pct']:.1f} ağırlık ve "
                f"{position['target_price']:.2f} ₺ hedef fiyat ile model portföye eklendi."
            ),
            "details": {
                "entry_price": position["entry_price"],
                "target_price": position["target_price"],
                "weight_pct": position["weight_pct"],
            },
        }

        state["positions"].insert(0, position)
        state["changelog"].insert(0, changelog_entry)
        state["changelog"] = state["changelog"][:200]
        _save_model_portfolio_state(processed_dir, state)
        return position


def update_position(
    processed_dir: Path,
    position_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    with _STATE_LOCK:
        state = load_model_portfolio_state(processed_dir)
        target_idx = next(
            (i for i, p in enumerate(state["positions"]) if str(p.get("id")) == str(position_id)),
            None,
        )
        if target_idx is None:
            raise KeyError("Model portföy pozisyonu bulunamadı.")

        position = dict(state["positions"][target_idx])
        prev_target = float(position.get("target_price") or 0.0)
        prev_weight = float(position.get("weight_pct") or 0.0)

        new_target = (
            _positive_float(payload.get("target_price"))
            if "target_price" in payload and payload.get("target_price") is not None
            else prev_target
        )
        new_weight = (
            _positive_float(payload.get("weight_pct"))
            if "weight_pct" in payload and payload.get("weight_pct") is not None
            else prev_weight
        )
        if new_target is None or new_target <= 0:
            raise ValueError("Hedef fiyat 0'dan büyük olmalıdır.")
        if new_weight is None or new_weight <= 0 or new_weight > 100.0:
            raise ValueError("Portföy ağırlığı 0 ile 100 arasında olmalıdır.")

        if position.get("status") == "active":
            other_weights = sum(
                float(p.get("weight_pct") or 0.0)
                for i, p in enumerate(state["positions"])
                if i != target_idx and p.get("status") == "active"
            )
            if round(other_weights + new_weight, 4) > 100.0:
                remaining = max(0.0, round(100.0 - other_weights, 2))
                raise ValueError(
                    f"Toplam portföy ağırlığı %100'ü aşamaz. Bu pozisyon için maksimum ağırlık: %{remaining}"
                )

        if "entry_price" in payload and payload.get("entry_price") is not None:
            ep = _positive_float(payload.get("entry_price"))
            if ep is None:
                raise ValueError("Giriş fiyatı 0'dan büyük olmalıdır.")
            position["entry_price"] = round(ep, 4)

        if "entry_date" in payload and payload.get("entry_date"):
            ed = str(payload.get("entry_date")).strip()[:10]
            date.fromisoformat(ed)
            position["entry_date"] = ed

        if "entry_xu100" in payload and payload.get("entry_xu100") is not None:
            ex = _positive_float(payload.get("entry_xu100"))
            position["entry_xu100"] = round(ex, 2) if ex else None

        if "sector" in payload:
            position["sector"] = (str(payload.get("sector") or "").strip()[:64]) or None
        if "horizon" in payload and payload.get("horizon") is not None:
            position["horizon"] = str(payload.get("horizon") or "").strip()[:32] or "6-12 Ay"
        if "rating" in payload and payload.get("rating") is not None:
            position["rating"] = str(payload.get("rating") or "").strip()[:48] or "Endeks Üstü Getiri"
        if "thesis" in payload and payload.get("thesis") is not None:
            position["thesis"] = str(payload.get("thesis") or "").strip()[:4000]
        if "catalysts" in payload and payload.get("catalysts") is not None:
            position["catalysts"] = _normalize_catalysts(payload.get("catalysts"))
        if "risk_note" in payload and payload.get("risk_note") is not None:
            position["risk_note"] = str(payload.get("risk_note") or "").strip()[:1500]
        if "entry_multiples" in payload and isinstance(payload.get("entry_multiples"), dict):
            position["entry_multiples"] = _normalize_multiples(payload.get("entry_multiples"))

        now = _now_iso()
        today = _today_iso()
        target_changed = abs(new_target - prev_target) >= 0.005
        weight_changed = abs(new_weight - prev_weight) >= 0.01
        revision_note = str(payload.get("revision_note") or "").strip()[:500]

        position["target_price"] = round(new_target, 4)
        position["weight_pct"] = round(new_weight, 2)
        position["updated_at"] = now

        if target_changed or weight_changed or revision_note:
            revisions = list(position.get("revisions") or [])
            revisions.insert(
                0,
                {
                    "id": uuid.uuid4().hex[:10],
                    "date": today,
                    "created_at": now,
                    "prev_target_price": round(prev_target, 4),
                    "new_target_price": round(new_target, 4),
                    "prev_weight_pct": round(prev_weight, 2),
                    "new_weight_pct": round(new_weight, 2),
                    "note": revision_note or None,
                },
            )
            position["revisions"] = revisions[:50]

            symbol = position["symbol"]
            changes: List[str] = []
            action = "updated"
            if target_changed:
                action = "target_revised"
                changes.append(f"hedef fiyatı {prev_target:.2f} ₺ → {new_target:.2f} ₺")
            if weight_changed:
                action = "weight_revised" if not target_changed else "target_revised"
                changes.append(f"ağırlığı %{prev_weight:.1f} → %{new_weight:.1f}")
            summary_body = " ve ".join(changes) if changes else "yatırım notu"
            summary = f"{symbol} {summary_body} olarak güncellendi."
            if revision_note:
                summary = f"{summary} ({revision_note})"

            state["changelog"].insert(
                0,
                {
                    "id": uuid.uuid4().hex[:12],
                    "date": today,
                    "created_at": now,
                    "action": action,
                    "symbol": symbol,
                    "company_name": position.get("company_name") or symbol,
                    "summary": summary,
                    "details": {
                        "prev_target_price": prev_target,
                        "new_target_price": new_target,
                        "prev_weight_pct": prev_weight,
                        "new_weight_pct": new_weight,
                        "note": revision_note or None,
                    },
                },
            )
            state["changelog"] = state["changelog"][:200]

        state["positions"][target_idx] = position
        _save_model_portfolio_state(processed_dir, state)
        return position


def close_position(
    processed_dir: Path,
    position_id: str,
    payload: Dict[str, Any],
    *,
    current_price_fallback: Optional[float] = None,
    current_xu100_fallback: Optional[float] = None,
) -> Dict[str, Any]:
    with _STATE_LOCK:
        state = load_model_portfolio_state(processed_dir)
        target_idx = next(
            (i for i, p in enumerate(state["positions"]) if str(p.get("id")) == str(position_id)),
            None,
        )
        if target_idx is None:
            raise KeyError("Model portföy pozisyonu bulunamadı.")

        position = dict(state["positions"][target_idx])
        if position.get("status") == "closed":
            raise ValueError("Bu pozisyon zaten kapatılmış.")

        exit_price = _positive_float(payload.get("exit_price")) or _positive_float(current_price_fallback)
        if exit_price is None:
            raise ValueError("Çıkış fiyatı (exit_price) 0'dan büyük olmalıdır.")

        closed_date_raw = str(payload.get("closed_date") or _today_iso()).strip()[:10]
        date.fromisoformat(closed_date_raw)

        exit_xu100 = _positive_float(payload.get("exit_xu100")) or _positive_float(current_xu100_fallback)
        close_reason = str(payload.get("close_reason") or "Hedef Fiyata Ulaştı / Kâr Realizasyonu").strip()[:240]
        now = _now_iso()

        position["status"] = "closed"
        position["closed_date"] = closed_date_raw
        position["exit_price"] = round(exit_price, 4)
        position["exit_xu100"] = round(exit_xu100, 2) if exit_xu100 else None
        position["close_reason"] = close_reason
        position["updated_at"] = now

        realized_ret = compute_return_pct(exit_price, _positive_float(position.get("entry_price")))
        ret_str = f"%+{realized_ret:.1f}" if realized_ret is not None and realized_ret >= 0 else f"%{realized_ret or 0.0:.1f}"
        symbol = position["symbol"]

        state["changelog"].insert(
            0,
            {
                "id": uuid.uuid4().hex[:12],
                "date": closed_date_raw,
                "created_at": now,
                "action": "closed",
                "symbol": symbol,
                "company_name": position.get("company_name") or symbol,
                "summary": f"{symbol} {exit_price:.2f} ₺ fiyat ve {ret_str} getiri ile model portföyden çıkarıldı ({close_reason}).",
                "details": {
                    "entry_price": position.get("entry_price"),
                    "exit_price": position["exit_price"],
                    "return_pct": realized_ret,
                    "close_reason": close_reason,
                },
            },
        )
        state["changelog"] = state["changelog"][:200]
        state["positions"][target_idx] = position
        _save_model_portfolio_state(processed_dir, state)
        return position


def delete_position(processed_dir: Path, position_id: str) -> Dict[str, Any]:
    with _STATE_LOCK:
        state = load_model_portfolio_state(processed_dir)
        target_idx = next(
            (i for i, p in enumerate(state["positions"]) if str(p.get("id")) == str(position_id)),
            None,
        )
        if target_idx is None:
            raise KeyError("Model portföy pozisyonu bulunamadı.")
        removed = state["positions"].pop(target_idx)
        _save_model_portfolio_state(processed_dir, state)
        return removed


def build_model_portfolio_payload(
    processed_dir: Path,
    *,
    quote_map: Dict[str, Dict[str, Any]],
    instrument_map: Dict[str, Dict[str, Any]],
    current_xu100: Optional[float],
    valuation_resolver: Callable[[str, Optional[float]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the fast (<15ms) Model Portfolio snapshot using already-cached quotes & multiples."""
    state = load_model_portfolio_state(processed_dir)
    active_rows: List[Dict[str, Any]] = []
    closed_rows: List[Dict[str, Any]] = []

    for raw_pos in state["positions"]:
        symbol = str(raw_pos.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        inst = instrument_map.get(symbol) or {}
        quote = quote_map.get(symbol) or {}
        company_name = (
            str(inst.get("name") or "").strip()
            or str(raw_pos.get("company_name") or "").strip()
            or symbol
        )
        logo_url = str(inst.get("logo_url") or raw_pos.get("logo_url") or "").strip() or None
        logo_source = str(inst.get("logo_source") or raw_pos.get("logo_source") or "").strip() or None

        entry_price = _positive_float(raw_pos.get("entry_price"))
        target_price = _positive_float(raw_pos.get("target_price"))
        weight_pct = _positive_float(raw_pos.get("weight_pct")) or 0.0
        entry_xu100 = _positive_float(raw_pos.get("entry_xu100"))
        status = str(raw_pos.get("status") or "active")

        if status == "closed":
            exit_price = _positive_float(raw_pos.get("exit_price"))
            exit_xu100 = _positive_float(raw_pos.get("exit_xu100")) or current_xu100
            return_pct = compute_return_pct(exit_price, entry_price)
            xu100_ret_pct = compute_return_pct(exit_xu100, entry_xu100)
            rel_ret_pct = compound_relative_return_pct(return_pct, xu100_ret_pct)
            beat_benchmark = (
                rel_ret_pct > 0
                if rel_ret_pct is not None
                else (return_pct > 0 if return_pct is not None else False)
            )
            closed_rows.append(
                {
                    **raw_pos,
                    "symbol": symbol,
                    "company_name": company_name,
                    "logo_url": logo_url,
                    "logo_source": logo_source,
                    "current_price": exit_price,
                    "daily_change_pct": None,
                    "return_pct": return_pct,
                    "xu100_return_pct": xu100_ret_pct,
                    "relative_return_pct": rel_ret_pct,
                    "beat_benchmark": beat_benchmark,
                    "holding_days": _holding_days(raw_pos.get("entry_date"), raw_pos.get("closed_date")),
                    "entry_multiples": _normalize_multiples(raw_pos.get("entry_multiples")),
                    "current_multiples": _normalize_multiples(raw_pos.get("entry_multiples")),
                }
            )
            continue

        live_price = _positive_float(quote.get("price"))
        effective_price = live_price if live_price is not None else entry_price
        daily_change_pct = _safe_float(quote.get("change_pct"))
        market_cap = _positive_float(quote.get("market_cap"))

        valuation = valuation_resolver(symbol, market_cap) if symbol else {}
        current_multiples = _normalize_multiples(valuation)

        return_pct = compute_return_pct(effective_price, entry_price)
        xu100_ret_pct = compute_return_pct(current_xu100, entry_xu100)
        rel_ret_pct = compound_relative_return_pct(return_pct, xu100_ret_pct)
        target_upside_pct = compute_return_pct(target_price, effective_price)
        target_progress_pct = compute_target_progress_pct(entry_price, effective_price, target_price)
        target_reached = (
            effective_price is not None
            and target_price is not None
            and effective_price >= target_price
        )
        attribution_pct = (
            round((weight_pct * return_pct) / 100.0, 2)
            if return_pct is not None
            else 0.0
        )

        active_rows.append(
            {
                **raw_pos,
                "symbol": symbol,
                "company_name": company_name,
                "logo_url": logo_url,
                "logo_source": logo_source,
                "current_price": effective_price,
                "daily_change_pct": daily_change_pct,
                "price_as_of": quote.get("price_as_of") or quote.get("as_of"),
                "market_cap": market_cap,
                "return_pct": return_pct,
                "xu100_return_pct": xu100_ret_pct,
                "relative_return_pct": rel_ret_pct,
                "target_upside_pct": target_upside_pct,
                "target_progress_pct": target_progress_pct,
                "target_reached": target_reached,
                "attribution_pct": attribution_pct,
                "holding_days": _holding_days(raw_pos.get("entry_date")),
                "entry_multiples": _normalize_multiples(raw_pos.get("entry_multiples")),
                "current_multiples": current_multiples,
            }
        )

    active_rows.sort(key=lambda r: (-float(r.get("weight_pct") or 0.0), str(r.get("symbol") or "")))
    closed_rows.sort(key=lambda r: (str(r.get("closed_date") or ""), str(r.get("symbol") or "")), reverse=True)

    total_weight_pct = round(sum(float(r.get("weight_pct") or 0.0) for r in active_rows), 2)
    cash_weight_pct = round(max(0.0, 100.0 - total_weight_pct), 2)

    weighted_return_sum = sum(
        float(r.get("weight_pct") or 0.0) * float(r.get("return_pct") or 0.0)
        for r in active_rows
        if r.get("return_pct") is not None
    )
    portfolio_return_pct = round(weighted_return_sum / 100.0, 2) if active_rows else None
    invested_return_pct = (
        round(weighted_return_sum / total_weight_pct, 2)
        if total_weight_pct > 0 and active_rows
        else None
    )

    xu100_weighted_pairs = [
        (float(r.get("weight_pct") or 0.0), float(r.get("xu100_return_pct")))
        for r in active_rows
        if r.get("xu100_return_pct") is not None
    ]
    xu100_weight_denom = sum(w for w, _ in xu100_weighted_pairs)
    xu100_return_pct = (
        round(sum(w * ret for w, ret in xu100_weighted_pairs) / xu100_weight_denom, 2)
        if xu100_weight_denom > 0
        else None
    )
    relative_return_pct = compound_relative_return_pct(invested_return_pct, xu100_return_pct)

    upside_pairs = [
        (float(r.get("weight_pct") or 0.0), float(r.get("target_upside_pct")))
        for r in active_rows
        if r.get("target_upside_pct") is not None
    ]
    upside_weight_denom = sum(w for w, _ in upside_pairs)
    weighted_target_upside_pct = (
        round(sum(w * u for w, u in upside_pairs) / upside_weight_denom, 2)
        if upside_weight_denom > 0
        else None
    )

    closed_count = len(closed_rows)
    closed_win_count = sum(1 for r in closed_rows if r.get("beat_benchmark"))
    closed_hit_rate_pct = (
        round((closed_win_count / closed_count) * 100.0, 1)
        if closed_count > 0
        else None
    )
    closed_returns = [float(r["return_pct"]) for r in closed_rows if r.get("return_pct") is not None]
    closed_relatives = [float(r["relative_return_pct"]) for r in closed_rows if r.get("relative_return_pct") is not None]
    closed_avg_return_pct = round(sum(closed_returns) / len(closed_returns), 2) if closed_returns else None
    closed_avg_relative_pct = round(sum(closed_relatives) / len(closed_relatives), 2) if closed_relatives else None

    all_dates = [
        str(p.get("entry_date"))[:10]
        for p in state["positions"]
        if p.get("entry_date")
    ]
    inception_date = min(all_dates) if all_dates else None

    return {
        "version": state["version"],
        "updated_at": state["updated_at"],
        "as_of": _now_iso(),
        "current_xu100": current_xu100,
        "summary": {
            "active_count": len(active_rows),
            "closed_count": closed_count,
            "total_weight_pct": total_weight_pct,
            "cash_weight_pct": cash_weight_pct,
            "portfolio_return_pct": portfolio_return_pct,
            "invested_return_pct": invested_return_pct,
            "xu100_return_pct": xu100_return_pct,
            "relative_return_pct": relative_return_pct,
            "weighted_target_upside_pct": weighted_target_upside_pct,
            "closed_hit_rate_pct": closed_hit_rate_pct,
            "closed_win_count": closed_win_count,
            "closed_avg_return_pct": closed_avg_return_pct,
            "closed_avg_relative_pct": closed_avg_relative_pct,
            "inception_date": inception_date,
        },
        "active_positions": active_rows,
        "closed_positions": closed_rows,
        "changelog": state["changelog"],
    }


def build_model_portfolio_performance_payload(
    processed_dir: Path,
    *,
    fetch_series_fn: Callable[[str, str, date, date], List[Dict[str, Any]]],
    quote_map: Dict[str, Dict[str, Any]],
    current_xu100: Optional[float],
    shared_cache_get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    shared_cache_set: Optional[Callable[[str, Dict[str, Any], int], None]] = None,
) -> Dict[str, Any]:
    """Build canonical daily Model Portfolio vs XU100 series with version+day caching.

    All range filters (1M, 3M, 6M, YTD, 1Y, ALL) are sliced client-side from this
    canonical response so switching ranges never triggers an extra network request.
    """
    state = load_model_portfolio_state(processed_dir)
    active_positions = [p for p in state["positions"] if str(p.get("status") or "active") == "active"]
    today = date.today()
    today_iso = today.isoformat()
    version = state["version"]

    if not active_positions:
        return {
            "version": version,
            "start_date": today_iso,
            "end_date": today_iso,
            "points": [],
            "as_of": _now_iso(),
            "cache_status": "empty",
        }

    entry_dates: List[date] = []
    for pos in active_positions:
        try:
            entry_dates.append(date.fromisoformat(str(pos.get("entry_date"))[:10]))
        except ValueError:
            continue
    earliest_entry = min(entry_dates) if entry_dates else (today - timedelta(days=365))
    canonical_start = min(earliest_entry, today - timedelta(days=365))

    cache_key = f"api:model-portfolio:performance:v1:ver={version}:day={today_iso}"
    now_ts = time.time()

    mem_entry = _MODEL_PORTFOLIO_PERF_MEMORY_CACHE.get(cache_key)
    base_series_data: Optional[Dict[str, Any]] = None
    cache_status = "miss"

    if mem_entry and now_ts - float(mem_entry.get("_ts") or 0) < _PERF_MEMORY_TTL_SECONDS:
        base_series_data = mem_entry.get("data")
        cache_status = "memory_hit"
    elif shared_cache_get is not None:
        shared_hit = shared_cache_get(cache_key)
        if isinstance(shared_hit, dict) and isinstance(shared_hit.get("sorted_dates"), list):
            base_series_data = shared_hit
            _MODEL_PORTFOLIO_PERF_MEMORY_CACHE[cache_key] = {"_ts": now_ts, "data": shared_hit}
            cache_status = "shared_hit"

    if base_series_data is None:
        symbols = sorted({str(p.get("symbol") or "").strip().upper() for p in active_positions if p.get("symbol")})
        from concurrent.futures import ThreadPoolExecutor

        def fetch_one(item: Tuple[str, str]) -> Tuple[str, List[Dict[str, Any]]]:
            kind, sym = item
            return sym if kind == "stock" else "XU100", fetch_series_fn(kind, sym, canonical_start, today)

        tasks: List[Tuple[str, str]] = [("index", "XU100")] + [("stock", sym) for sym in symbols]
        series_by_symbol: Dict[str, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            for key, pts in pool.map(fetch_one, tasks):
                date_map: Dict[str, float] = {}
                for pt in pts or []:
                    d_str = str(pt.get("date") or "")[:10]
                    val = _positive_float(pt.get("value"))
                    if d_str and val is not None:
                        date_map[d_str] = val
                series_by_symbol[key] = date_map

        all_dates_set = set(series_by_symbol.get("XU100", {}).keys())
        for sym in symbols:
            all_dates_set.update(series_by_symbol.get(sym, {}).keys())
        sorted_dates = sorted(d for d in all_dates_set if canonical_start.isoformat() <= d <= today_iso)

        base_series_data = {
            "sorted_dates": sorted_dates,
            "series_by_symbol": series_by_symbol,
        }
        if sorted_dates:
            _MODEL_PORTFOLIO_PERF_MEMORY_CACHE[cache_key] = {"_ts": now_ts, "data": base_series_data}
            if shared_cache_set is not None:
                shared_cache_set(cache_key, base_series_data, _PERF_MEMORY_TTL_SECONDS)

    sorted_dates = list(base_series_data.get("sorted_dates") or [])
    raw_series_map: Dict[str, Dict[str, float]] = {
        k: dict(v) for k, v in (base_series_data.get("series_by_symbol") or {}).items()
    }

    # Overlay today's live cached prices without mutating the historical cache entry
    live_Overlay_dates = set(sorted_dates)
    has_live_today = False
    if current_xu100 is not None and current_xu100 > 0:
        raw_series_map.setdefault("XU100", {})[today_iso] = float(current_xu100)
        has_live_today = True
    for pos in active_positions:
        sym = str(pos.get("symbol") or "").strip().upper()
        live_p = _positive_float((quote_map.get(sym) or {}).get("price"))
        if live_p is not None:
            raw_series_map.setdefault(sym, {})[today_iso] = live_p
            has_live_today = True
    if has_live_today and today_iso not in live_Overlay_dates:
        sorted_dates.append(today_iso)
        sorted_dates.sort()

    if not sorted_dates:
        return {
            "version": version,
            "start_date": canonical_start.isoformat(),
            "end_date": today_iso,
            "points": [],
            "as_of": _now_iso(),
            "cache_status": cache_status,
        }

    # Forward-fill missing trading days per symbol & compute daily weighted index (base 100)
    symbols = [str(p.get("symbol") or "").strip().upper() for p in active_positions if p.get("symbol")]
    last_price: Dict[str, Optional[float]] = {sym: None for sym in symbols}
    last_xu100: Optional[float] = None

    portfolio_index = 100.0
    xu100_index = 100.0
    points: List[Dict[str, Any]] = []

    for idx, d_str in enumerate(sorted_dates):
        xu_val = raw_series_map.get("XU100", {}).get(d_str)
        prev_xu = last_xu100
        if xu_val is not None and xu_val > 0:
            last_xu100 = xu_val

        if idx > 0 and prev_xu is not None and last_xu100 is not None and prev_xu > 0:
            xu_daily_ret = (last_xu100 - prev_xu) / prev_xu
            xu100_index = xu100_index * (1.0 + xu_daily_ret)

        weighted_daily_ret = 0.0
        active_weight_sum = 0.0
        for pos in active_positions:
            sym = str(pos.get("symbol") or "").strip().upper()
            weight = float(pos.get("weight_pct") or 0.0)
            entry_d = str(pos.get("entry_date") or "")[:10]
            sym_val = raw_series_map.get(sym, {}).get(d_str)
            prev_sym = last_price.get(sym)
            if sym_val is not None and sym_val > 0:
                last_price[sym] = sym_val
            cur_sym = last_price.get(sym)

            # A stock contributes from its entry_date onwards (or across the chart if already active)
            if d_str >= entry_d and idx > 0 and prev_sym is not None and cur_sym is not None and prev_sym > 0:
                stock_daily_ret = (cur_sym - prev_sym) / prev_sym
                weighted_daily_ret += weight * stock_daily_ret
                active_weight_sum += weight
            elif d_str < entry_d and idx > 0 and prev_sym is not None and cur_sym is not None and prev_sym > 0:
                # Before earliest entry date (e.g., 1Y lookback), show hypothetical current basket trajectory
                stock_daily_ret = (cur_sym - prev_sym) / prev_sym
                weighted_daily_ret += weight * stock_daily_ret
                active_weight_sum += weight

        if idx > 0 and active_weight_sum > 0:
            # Normalize daily return by active stock sleeve weight so 100 TL invested tracks stock performance cleanly
            portfolio_daily_ret = weighted_daily_ret / active_weight_sum
            portfolio_index = portfolio_index * (1.0 + portfolio_daily_ret)

        port_ret_pct = round(portfolio_index - 100.0, 2)
        xu_ret_pct = round(xu100_index - 100.0, 2)
        rel_pct = compound_relative_return_pct(port_ret_pct, xu_ret_pct)

        points.append(
            {
                "date": d_str,
                "portfolio_value": round(portfolio_index, 2),
                "xu100_value": round(xu100_index, 2),
                "portfolio_return_pct": port_ret_pct,
                "xu100_return_pct": xu_ret_pct,
                "relative_return_pct": rel_pct if rel_pct is not None else 0.0,
            }
        )

    return {
        "version": version,
        "start_date": points[0]["date"] if points else canonical_start.isoformat(),
        "end_date": points[-1]["date"] if points else today_iso,
        "points": points,
        "as_of": _now_iso(),
        "cache_status": cache_status,
    }
