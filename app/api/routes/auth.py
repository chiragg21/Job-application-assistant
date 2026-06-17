"""
app/api/routes/auth.py
-----------------------
Google OAuth 2.0 login flow.

  GET  /auth/login            → redirect to Google consent screen
  GET  /auth/callback         → handle Google callback, issue JWT cookie
  POST /auth/logout           → clear cookie
  GET  /auth/me               → return current user info (requires JWT cookie)

Requires env vars:
  GOOGLE_CLIENT_ID      — from Google Cloud Console
  GOOGLE_CLIENT_SECRET  — from Google Cloud Console
  OAUTH_REDIRECT_URI    — e.g. http://localhost:8000/auth/callback
  JWT_SECRET            — random string, keep secret
  FRONTEND_URL          — where to redirect after login (e.g. http://localhost:3000)

Set DEV_AUTH=true (default) to skip OAuth entirely.
"""

from __future__ import annotations

import os
import time

from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.middleware import make_access_token, get_current_user, CurrentUser
from app.utils.sqlite_handler import SQLHandler
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI   = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
FRONTEND_URL         = os.environ.get("FRONTEND_URL", "http://localhost:3000")
DEV_AUTH             = os.environ.get("DEV_AUTH", "true").lower() in ("1", "true", "yes")

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL  = "https://www.googleapis.com/oauth2/v3/userinfo"


@router.get("/login")
async def login(request: Request):
    """Redirect the browser to Google's OAuth consent screen."""
    if DEV_AUTH:
        # Dev mode — auto-login as user 1
        token = make_access_token(1, "dev@local", "Dev User")
        resp = RedirectResponse(url=FRONTEND_URL)
        resp.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=60*60*24*30)
        return resp

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID in .env")

    client = AsyncOAuth2Client(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        redirect_uri=OAUTH_REDIRECT_URI,
    )
    uri, state = client.create_authorization_url(
        GOOGLE_AUTHORIZE_URL,
        scope="openid email profile",
    )
    resp = RedirectResponse(url=uri)
    resp.set_cookie("oauth_state", state, httponly=True, samesite="lax", max_age=600)
    return resp


@router.get("/callback")
async def callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Exchange the code for tokens, upsert user in DB, issue JWT cookie."""
    if error:
        return RedirectResponse(url=f"{FRONTEND_URL}/login?error={error}")

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google OAuth not configured")

    saved_state = request.cookies.get("oauth_state", "")
    client = AsyncOAuth2Client(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        redirect_uri=OAUTH_REDIRECT_URI,
        state=saved_state,
    )

    try:
        token = await client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        resp_userinfo = await client.get(GOOGLE_USERINFO_URL)
        userinfo = resp_userinfo.json()
    except Exception as exc:
        log.error("OAuth token exchange failed: %s", exc)
        raise HTTPException(status_code=400, detail="OAuth exchange failed")

    oauth_sub     = userinfo["sub"]
    email         = userinfo.get("email", "")
    name          = userinfo.get("name", "")
    avatar_url    = userinfo.get("picture", "")
    email_verified = userinfo.get("email_verified", False)

    db = SQLHandler()
    # Find or create user by oauth_sub
    user = db.fetch_one("users", filters={"oauth_sub": oauth_sub})
    if user is None:
        user_id = db.add_one("users", {
            "name":           name,
            "email":          email,
            "oauth_sub":      oauth_sub,
            "email_verified": int(email_verified),
            "avatar_url":     avatar_url,
            "last_login_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    else:
        user_id = user["id"]
        db.update_data("users", {
            "email":          email,
            "email_verified": int(email_verified),
            "avatar_url":     avatar_url,
            "last_login_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, {"id": user_id})

    access_token = make_access_token(user_id, email, name, avatar_url)
    resp = RedirectResponse(url=FRONTEND_URL)
    resp.delete_cookie("oauth_state")
    resp.set_cookie("access_token", access_token, httponly=True, samesite="lax", max_age=60*60*24*30)
    return resp


@router.post("/logout")
def logout():
    resp = JSONResponse({"logged_out": True})
    resp.delete_cookie("access_token")
    return resp


@router.get("/me")
def me(user: CurrentUser):
    return {
        "id":             user.id,
        "email":          user.email,
        "name":           user.name,
        "avatar_url":     user.avatar_url,
        "email_verified": user.email_verified,
    }
