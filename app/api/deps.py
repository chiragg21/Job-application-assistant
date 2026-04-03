# app/api/deps.py
#
# FastAPI dependency providers shared across routers.

from __future__ import annotations

from app.utils import SQLHandler, get_logger

log = get_logger(__name__)

_sql: SQLHandler | None = None


def get_db() -> SQLHandler:
    global _sql
    if _sql is None:
        _sql = SQLHandler()
    return _sql
