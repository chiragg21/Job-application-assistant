"""
utils.logger
~~~~~~~~~~~~
Thin wrapper that boots the infrakit logger once and exports ``get_logger``.

Reads LOG_DIR, LOG_STRATEGY, LOG_STREAM, LOG_FORMAT, and LOG_LEVEL from the
project config file (.env / config.yaml / config.json) via infrakit.config.

Usage
-----
    from utils.logger import get_logger

    log = get_logger(__name__)
    log.info("hello")
"""

from pathlib import Path
from infrakit.core.config.loader import load, load_env
from infrakit.core.logger import setup, get_logger  # re-export get_logger

_booted = False


def _load_cfg() -> dict:
    if Path(".env").exists():
        return load_env(".env", cast_values=True)
    if Path("config.yaml").exists():
        return load("config.yaml")
    if Path("config.json").exists():
        return load("config.json")
    return {}


def _boot() -> None:
    global _booted
    if _booted:
        return
    cfg = _load_cfg()
    setup(
        log_dir=cfg.get("LOG_DIR", "logs"),
        strategy=cfg.get("LOG_STRATEGY", "date"),
        stream=cfg.get("LOG_STREAM", "stdout"),
        fmt=cfg.get("LOG_FORMAT", "human"),
        level=cfg.get("LOG_LEVEL", "DEBUG"),
    )
    _booted = True


_boot()

__all__ = ["get_logger"]
