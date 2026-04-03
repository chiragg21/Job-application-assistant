"""
utils.llm
~~~~~~~~~
Thin wrapper that boots the infrakit LLM client once and exports it.

Reads all configuration from the project config file (.env / config.yaml /
config.json) via infrakit.config — no raw os.getenv calls.

The client reads key state from ``~/.infrakit/llm/`` by default, and
loads quota limits from ``~/.infrakit/llm/quotas.json`` if that file
exists.  Both paths can be overridden via LLM_STATE_DIR / LLM_QUOTA_FILE.

Usage
-----
    from utils.llm import llm, Prompt

    response = llm.generate(Prompt(user="Summarise this text: ..."), provider="openai")
    print(response.content)

    # structured output
    from pydantic import BaseModel

    class Summary(BaseModel):
        title: str
        bullets: list[str]

    response = llm.generate(
        Prompt(system="Return only JSON.", user="Summarise: ..."),
        provider="openai",
        response_model=Summary,
    )
    if response.schema_matched:
        print(response.parsed.bullets)

    # async batch (inside an async function)
    batch = await llm.async_batch_generate(prompts, provider="gemini")
"""

import json
from pathlib import Path

from infrakit.core.config.loader import load, load_env
from infrakit.llm import LLMClient, Prompt  # re-export Prompt for convenience

# ── config loading ────────────────────────────────────────────────────────────

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

# ── key loading ───────────────────────────────────────────────────────────────
# Keys are read from the project config or from a local keys.json file.
# Never commit real API keys — use .env or your secret manager.


# ── client singleton ──────────────────────────────────────────────────────────

llm: LLMClient = LLMClient(
    keys={'gemini_keys':_cfg.get("gemini_api", {}).get("api_keys", []),
          'openai_keys':_cfg.get("openai_api", {}).get("api_keys", [])},
    # storage_dir and quota_file default to ~/.infrakit/llm/
    # override via LLM_STATE_DIR / LLM_QUOTA_FILE in your config file:
    storage_dir=_cfg.get("LLM_STATE_DIR") or None,
    quota_file=_cfg.get("LLM_QUOTA_FILE") or None,
    mode=_cfg.get("LLM_MODE", "async"),            # "async" | "threaded"
    max_concurrent=int(_cfg.get("LLM_CONCURRENCY", 3)),
    openai_model=_cfg.get("OPENAI_MODEL") or None,
    gemini_model=_cfg.get("GEMINI_MODEL") or None,
)

__all__ = ["llm", "Prompt"]
