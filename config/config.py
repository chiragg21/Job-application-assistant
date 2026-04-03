from pathlib import Path
from infrakit.core.config.loader import load

_CONFIG_PATH = Path(__file__).parent / "config.ini"
_ENV_PATH    = Path(__file__).parent.parent / ".env"

_cache: dict | None = None


def get_config_dict() -> dict:
    """
    Load config.ini with .env overrides.
    Returns the same nested dict as before: {section: {key: value}}.
    Result is cached after the first call.
    """
    global _cache
    if _cache is None:
        _cache = load(
            _CONFIG_PATH,
            env_file=_ENV_PATH,
            interpolate=True,
            cast_values=True,
        )
    return _cache
