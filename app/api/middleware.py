"""
app/api/middleware.py
---------------------
JWT authentication dependency for FastAPI.

Usage in routes:
    from app.api.middleware import get_current_user, CurrentUser

    @router.get("/protected")
    def endpoint(user: CurrentUser):
        ...   # user.id is the DB user_id

Dev mode:
    Set DEV_AUTH=true in .env — all requests resolve to user_id=1 with no token
    required.  This is the default until Google OAuth credentials are configured.
"""

from __future__ import annotations

import os
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, HTTPException, status
from pydantic import BaseModel

from app.utils.logger import get_logger

log = get_logger(__name__)

JWT_SECRET  = os.environ.get("JWT_SECRET",  "dev-secret-change-before-hosting")
JWT_ALG     = "HS256"
DEV_AUTH    = os.environ.get("DEV_AUTH", "true").lower() in ("1", "true", "yes")


class AuthUser(BaseModel):
    id:             int
    email:          str | None = None
    name:           str | None = None
    avatar_url:     str | None = None
    email_verified: bool       = False


def get_current_user(
    access_token: Annotated[str | None, Cookie()] = None,
) -> AuthUser:
    """
    FastAPI dependency.  Returns the authenticated user from the JWT cookie.
    In DEV_AUTH mode, always returns a mock user with id=1.
    """
    if DEV_AUTH:
        return AuthUser(id=1, email="dev@local", name="Dev User")

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = jwt.decode(access_token, JWT_SECRET, algorithms=[JWT_ALG])
        return AuthUser(
            id=int(payload["sub"]),
            email=payload.get("email"),
            name=payload.get("name"),
            avatar_url=payload.get("picture"),
            email_verified=bool(payload.get("email_verified", False)),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# Type alias for DI
CurrentUser = Annotated[AuthUser, Depends(get_current_user)]


def make_access_token(user_id: int, email: str, name: str, picture: str = "") -> str:
    """Sign a short-lived JWT for the given user."""
    import time
    payload = {
        "sub":            str(user_id),
        "email":          email,
        "name":           name,
        "picture":        picture,
        "email_verified": True,
        "iat":            int(time.time()),
        "exp":            int(time.time()) + 60 * 60 * 24 * 30,   # 30 days
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
