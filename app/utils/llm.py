"""
utils.llm
~~~~~~~~~
Task-routed LLM access.  Edit config/task_routing.toml to change which
provider/model handles each task — no Python code changes required.

Public API
----------
  generate_for_task(task, prompt, response_model)  → LLMResponse
  merged_status(provider, key_id)                  → list[dict]
  ALL_CLIENTS                                       → list[LLMClient]
  Prompt                                            → re-export from infrakit
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from infrakit.core.config.loader import load, load_env
from infrakit.llm import LLMClient, Prompt  # re-export Prompt for convenience

from app.utils.logger import get_logger

log = get_logger(__name__)


# ── app config ────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    if Path("config/config.ini").exists():
        return load("config/config.ini", env_override=True, cast_values=True, env_file=".env", interpolate=True)
    if Path(".env").exists():
        return load_env(".env", cast_values=True)
    if Path("config.yaml").exists():
        return load("config.yaml")
    if Path("config.json").exists():
        return load("config.json")
    return {}


_cfg = _load_cfg()


def _key_list(val: object) -> list[str]:
    """Wrap a bare string key in a list; skip invalid entries."""
    if not val:
        return []
    if isinstance(val, list):
        return [v for v in val if isinstance(v, str) and v.strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


# ── shared keys + common config ───────────────────────────────────────────────

_cfg_state  = _cfg.get("LLM_STATE_DIR")
_state_root = Path(_cfg_state) if _cfg_state else Path.home() / ".infrakit" / "llm"

_keys = {
    "gemini_keys": _key_list(_cfg.get("gemini_api", {}).get("api_keys", [])),
    "openai_keys": _key_list(_cfg.get("openai_api", {}).get("api_keys", [])),
    "groq_keys":   _key_list(_cfg.get("groq_api",   {}).get("api_keys", [])),
}

_common = dict(
    mode=_cfg.get("LLM_MODE", "async"),
    max_concurrent=int(_cfg.get("LLM_CONCURRENCY", 3)),
    quota_file=_cfg.get("LLM_QUOTA_FILE") or None,
    openai_model=_cfg.get("OPENAI_MODEL") or "gpt-4o-mini",
    groq_model=_cfg.get("groq_api", {}).get("model") or "llama-3.3-70b-versatile",
)

_DEFAULT_GEMINI = _cfg.get("GEMINI_MODEL") or "gemini-2.5-flash"


# ── task routing config ───────────────────────────────────────────────────────

_ROUTING_PATH = Path("config/task_routing.toml")

# Hard-coded defaults — used when task_routing.toml is missing.
_DEFAULT_ROUTING: dict = {
    "scoring":    {"provider": "gemini", "model": "gemini-3.5-flash",        "fallback": {"provider": "gemini", "model": "gemini-2.5-flash"}},
    "analysis":   {"provider": "gemini", "model": "gemini-3.5-flash",        "fallback": {"provider": "gemini", "model": "gemini-2.5-flash"}},
    "jd_parsing": {"provider": "gemini", "model": "gemini-2.5-flash"},
    "editing":    {"provider": "gemini", "model": "gemini-2.5-flash"},
    "generation": {"provider": "groq",   "model": "llama-3.3-70b-versatile", "fallback": {"provider": "gemini", "model": "gemini-2.5-flash-lite"}},
}


def _load_task_routing() -> dict:
    if not _ROUTING_PATH.exists():
        log.warning("[llm] %s not found — using built-in defaults", _ROUTING_PATH)
        return _DEFAULT_ROUTING
    with _ROUTING_PATH.open("rb") as f:
        data = tomllib.load(f)
    tasks = data.get("tasks", {})
    if not tasks:
        log.warning("[llm] task_routing.toml has no [tasks] entries — using built-in defaults")
        return _DEFAULT_ROUTING
    log.info("[llm] loaded task routing from %s: %s", _ROUTING_PATH, sorted(tasks))
    return tasks


# ── client pool ───────────────────────────────────────────────────────────────

@dataclass
class _Route:
    primary_provider:  str
    primary_client:    LLMClient
    fallback_provider: Optional[str]
    fallback_client:   Optional[LLMClient]  # None when infrakit handles cross-provider fallback


def _safe_dir(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def _build_client(
    primary_provider:  str,
    primary_model:     str,
    fallback_provider: Optional[str],
    fallback_model:    Optional[str],
) -> LLMClient:
    """
    Build one LLMClient for the given (primary, fallback) provider/model pair.

    Cross-provider fallback (e.g. groq → gemini): both providers go into
    fallback_order so infrakit handles the switch automatically.

    Same-provider fallback: primary_client covers only the primary model;
    the caller builds a separate fallback_client and switches at the Python level.
    """
    providers = {primary_provider}
    if fallback_provider:
        providers.add(fallback_provider)

    # Start with common config, then only include provider-model keys that this
    # client actually uses.  Leaving groq_model/openai_model on clients that
    # never call those providers causes infrakit to create ModelState entries for
    # them, producing duplicate model rows in merged_status().
    kw = {k: v for k, v in _common.items() if k not in ("groq_model", "openai_model")}

    if "gemini" in providers:
        kw["gemini_model"] = (
            primary_model  if primary_provider  == "gemini" else
            (fallback_model or _DEFAULT_GEMINI)
        )
    if "groq" in providers:
        kw["groq_model"] = (
            primary_model  if primary_provider  == "groq" else
            (fallback_model or _common["groq_model"])
        )
    if "openai" in providers:
        kw["openai_model"] = (
            primary_model  if primary_provider  == "openai" else
            (fallback_model or _common["openai_model"])
        )

    cross_provider = fallback_provider is not None and fallback_provider != primary_provider
    fallback_order = [primary_provider, fallback_provider] if cross_provider else [primary_provider]

    dir_name = _safe_dir(primary_model)
    if cross_provider and fallback_model:
        dir_name += f"__{_safe_dir(fallback_model)}"

    return LLMClient(
        keys=_keys,
        storage_dir=str(_state_root / dir_name),
        fallback_order=fallback_order,
        **kw,
    )


def _build_routes(task_cfg: dict) -> tuple[dict[str, _Route], list[LLMClient]]:
    """
    Parse task config → build route table and a deduplicated client pool.
    Tasks that share the same (provider, model, fallback) signature reuse one client.
    """
    pool:   dict[tuple, LLMClient] = {}
    routes: dict[str, _Route]      = {}

    def _get_or_create(pp: str, pm: str, fp: Optional[str] = None, fm: Optional[str] = None) -> LLMClient:
        key = (pp, pm, fp, fm)
        if key not in pool:
            pool[key] = _build_client(pp, pm, fp, fm)
        return pool[key]

    for task_name, cfg in task_cfg.items():
        pp = cfg["provider"]
        pm = cfg["model"]
        fb = cfg.get("fallback")
        fp = fb["provider"] if fb else None
        fm = fb["model"]    if fb else None

        cross_provider = fb is not None and fp != pp

        if cross_provider:
            # One client; infrakit auto-falls back across providers via fallback_order
            primary_client = _get_or_create(pp, pm, fp, fm)
            routes[task_name] = _Route(
                primary_provider=pp,
                primary_client=primary_client,
                fallback_provider=None,
                fallback_client=None,
            )
        else:
            # Same-provider (or no fallback): build primary, optionally a separate fallback
            primary_client  = _get_or_create(pp, pm)
            fallback_client = _get_or_create(fp, fm) if fb else None
            routes[task_name] = _Route(
                primary_provider=pp,
                primary_client=primary_client,
                fallback_provider=fp,
                fallback_client=fallback_client,
            )

    return routes, list(pool.values())


_task_cfg            = _load_task_routing()
_routes, ALL_CLIENTS = _build_routes(_task_cfg)


# ── public API ────────────────────────────────────────────────────────────────

def generate_for_task(task: str, prompt: Prompt, response_model=None):
    """
    Generate a response using the model configured for `task` in task_routing.toml.

    Fallback is handled transparently:
    - Cross-provider: infrakit switches providers automatically.
    - Same-provider:  this function calls the fallback client when the primary
                      returns response.error or a schema mismatch.

    Returns:  LLMResponse (always); check .schema_matched / .parsed on the result.
    Raises:   ValueError  — unknown task name.
              RuntimeError — all configured models are unavailable (no valid keys).
    """
    if task not in _routes:
        raise ValueError(
            f"Unknown task '{task}'. Defined tasks: {sorted(_routes.keys())}"
        )

    route = _routes[task]

    # Primary attempt
    resp = None
    try:
        resp = route.primary_client.generate(
            prompt, provider=route.primary_provider, response_model=response_model
        )
    except RuntimeError as exc:
        log.warning("[llm] task=%s primary unavailable: %s", task, exc)

    # Same-provider fallback (cross-provider is handled inside primary_client)
    if route.fallback_client is not None:
        needs_fallback = (
            resp is None
            or bool(resp.error)
            or (response_model is not None and (not resp.schema_matched or resp.parsed is None))
        )
        if needs_fallback:
            reason = (resp.error if resp else None) or "schema mismatch / no response"
            log.info("[llm] task=%s primary failed (%s) → fallback", task, reason)
            try:
                resp = route.fallback_client.generate(
                    prompt, provider=route.fallback_provider, response_model=response_model
                )
            except RuntimeError as exc:
                log.warning("[llm] task=%s fallback also unavailable: %s", task, exc)
                if resp is None:
                    raise RuntimeError(
                        f"All configured models for task '{task}' are unavailable"
                    ) from exc

    if resp is None:
        raise RuntimeError(
            f"All configured models for task '{task}' are unavailable — check API keys and quotas"
        )

    return resp


def merged_status(provider: str | None = None, key_id: str | None = None) -> list[dict]:
    """
    Merge status from all clients, combining model rows per (provider, key_id).
    Used by the settings UI to show one card per key with all active models listed.
    """
    seen: dict[tuple[str, str], dict] = {}

    for client in ALL_CLIENTS:
        try:
            entries = client.status(provider=provider, key_id=key_id)
        except Exception:
            continue
        for entry in entries:
            k = (entry["provider"], entry["key_id"])
            if k not in seen:
                seen[k] = {**entry, "models": list(entry.get("models", []))}
            else:
                # Merge models — skip any model name already present so that
                # clients sharing the same underlying key never produce
                # duplicate rows (which would trigger React key warnings).
                existing = {m.get("model") for m in seen[k]["models"]}
                for m in entry.get("models", []):
                    if m.get("model") not in existing:
                        seen[k]["models"].append(m)
                        existing.add(m.get("model"))
                if entry.get("status") == "active":
                    seen[k]["status"] = "active"
                if entry.get("current_rpm", 0) > seen[k].get("current_rpm", 0):
                    seen[k]["current_rpm"] = entry["current_rpm"]

    return list(seen.values())


__all__ = ["generate_for_task", "ALL_CLIENTS", "merged_status", "Prompt"]
