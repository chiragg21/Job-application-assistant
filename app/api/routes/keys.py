# app/api/routes/keys.py
#
# API key management endpoints.
#
#   GET    /keys/status          → live per-key metrics (merged across all clients)
#   PATCH  /keys/quota           → set rpm/token limits on all clients
#   POST   /keys                 → add a key at runtime + persist to data/user_keys.json
#   DELETE /keys/{provider}/{key_id} → remove a key at runtime + remove from persistence

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.utils.llm import ALL_CLIENTS, merged_status
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/keys", tags=["Key Management"])

_USER_KEYS_PATH = Path("data/user_keys.json")
_keys_lock = threading.Lock()

_VALID_PROVIDERS = {"gemini", "groq", "openai"}


# ── Persistence helpers ───────────────────────────────────────────────────────

def _load_raw() -> list[dict]:
    if not _USER_KEYS_PATH.exists():
        return []
    try:
        with _USER_KEYS_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_raw(entries: list[dict]) -> None:
    _USER_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _USER_KEYS_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def load_user_keys() -> None:
    """
    Called at startup: register all user-added keys with the running LLM client.
    Keys are stored in data/user_keys.json so they survive server restarts.
    """
    entries = _load_raw()
    for entry in entries:
        for client in ALL_CLIENTS:
            try:
                client.add_key(entry["provider"], entry["key"])
            except Exception as exc:
                log.warning("[keys] failed to load persisted key into client: %s", exc)
        log.info("[keys] loaded persisted key provider=%s id=%.8s…", entry["provider"], entry["key"])
    if entries:
        log.info("[keys] loaded %d user key(s) from disk", len(entries))


# ── Request / response models ─────────────────────────────────────────────────

class QuotaPatchRequest(BaseModel):
    provider:           str
    key_id:             str
    model:              Optional[str] = None   # None = applies to all models on this key
    rpm_limit:          Optional[int] = None
    tpm_limit:          Optional[int] = None
    daily_token_limit:  Optional[int] = None
    reset_hour_utc:     int = 0


class AddKeyRequest(BaseModel):
    provider: str   # "gemini" | "groq" | "openai"
    key:      str


class AddKeyResponse(BaseModel):
    added:   bool
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _any_client():
    """Return the first available client — used for operations that apply globally."""
    if not ALL_CLIENTS:
        raise RuntimeError("No LLM clients configured")
    return ALL_CLIENTS[0]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
def get_key_status(provider: Optional[str] = None):
    """
    Return live status for all registered API keys.

    Optional query param `provider` filters to a single provider
    (e.g. `?provider=gemini`).

    Each entry in the returned list contains:
      provider, key_id, status, rpm_limit, current_rpm,
      models[{model, status, current_tpm, day_token_total, daily_remaining,
              total_tokens, total_requests, total_errors}],
      recent_meta (last 5 request latencies / outcomes)
    """
    try:
        return merged_status(provider=provider)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/quota")
def patch_key_quota(req: QuotaPatchRequest):
    """
    Set or update quota limits for a specific API key.

    `key_id` is the first 8 characters of the raw key (as shown by /keys/status).
    `model` is optional — omit to apply limits to all models on this key.
    """
    try:
        from infrakit.llm import QuotaConfig
        quota = QuotaConfig(
            model=req.model,
            rpm_limit=req.rpm_limit,
            tpm_limit=req.tpm_limit,
            daily_token_limit=req.daily_token_limit,
            reset_hour_utc=req.reset_hour_utc,
        )
        for client in ALL_CLIENTS:
            client.set_quota(req.provider, req.key_id, quota)
        log.info("[keys] quota updated provider=%s key_id=%s", req.provider, req.key_id)
        return {"updated": True}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("", response_model=AddKeyResponse)
def add_key(req: AddKeyRequest):
    """
    Add a new API key at runtime.

    The key is immediately available for requests (no restart needed).
    It is also persisted to `data/user_keys.json` so it survives restarts.
    """
    provider = req.provider.lower().strip()
    if provider not in _VALID_PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported provider '{provider}'. Valid: {sorted(_VALID_PROVIDERS)}",
        )
    if not req.key.strip():
        raise HTTPException(status_code=422, detail="Key must not be empty.")
    try:
        for client in ALL_CLIENTS:
            client.add_key(provider, req.key)
        with _keys_lock:
            entries = _load_raw()
            if not any(e["key"] == req.key and e["provider"] == provider for e in entries):
                entries.append({"provider": provider, "key": req.key})
                _save_raw(entries)
        log.info("[keys] added key provider=%s id=%.8s…", provider, req.key)
        return AddKeyResponse(added=True, message=f"{provider} key registered successfully.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/{provider}/{key_id}/toggle")
def toggle_key(provider: str, key_id: str, active: bool = True):
    """
    Manually activate or deactivate a key.

    Pass `?active=false` to suspend the key — it will be skipped for all
    requests but stays registered.  Pass `?active=true` to re-enable it.
    The key's daily token counter and model stats are left untouched.
    """
    from infrakit.llm.models import ModelStatus

    provider = provider.lower()
    if provider not in _VALID_PROVIDERS:
        raise HTTPException(status_code=422, detail=f"Unknown provider: {provider}")

    import time
    found = False
    for client in ALL_CLIENTS:
        km = client._km  # type: ignore[attr-defined]
        with km._lock:  # type: ignore[attr-defined]
            ks = next((k for k in km._states.get(provider, []) if k.key_id == key_id), None)
            if ks is None:
                continue
            found = True
            for ms in ks.model_states.values():
                if active:
                    ms.status = ModelStatus.ACTIVE
                    ms.deactivated_at = None
                else:
                    ms.status = ModelStatus.INACTIVE
                    ms.deactivated_at = time.time()
            km._persist()  # type: ignore[attr-defined]

    if not found:
        raise HTTPException(status_code=404, detail=f"Key '{key_id}' not found for provider '{provider}'.")

    log.info("[keys] toggle provider=%s key_id=%s active=%s", provider, key_id, active)
    return {"active": active, "key_id": key_id}


@router.delete("/{provider}/{key_id}")
def remove_key(provider: str, key_id: str):
    """
    Remove a key by provider and key_id prefix.

    `key_id` is the first 8 characters as shown by /keys/status.
    The key is deregistered immediately and removed from the persistence file.
    """
    try:
        for client in ALL_CLIENTS:
            client.remove_key(provider, key_id)
        with _keys_lock:
            entries = _load_raw()
            before  = len(entries)
            entries = [
                e for e in entries
                if not (e["provider"] == provider and e["key"].startswith(key_id))
            ]
            if len(entries) < before:
                _save_raw(entries)
        log.info("[keys] removed key provider=%s key_id=%s", provider, key_id)
        return {"removed": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
