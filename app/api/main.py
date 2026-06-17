# app/api/main.py
#
# FastAPI application entry point.
#
# Start with:
#   uv run uvicorn app.api.main:app --reload

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import jd, resume, edit, generate, score, keys, library, auth, applications, skills
from app.utils import get_logger

log = get_logger(__name__)


def _prewarm_embeddings() -> None:
    """
    Load the sentence-transformer weights once at startup so the first real
    request is not delayed.  Runs in a thread-pool executor to avoid blocking
    the event loop.
    """
    try:
        from app.core.embeddings import prewarm
        prewarm()
        log.info("[startup] embedding model pre-warmed")
    except Exception as exc:
        log.warning("[startup] embedding pre-warm failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Job-Assistant API starting up")
    from app.graph.edit_graph import evict_old_threads, _checkpointer
    from app.api.routes.keys import load_user_keys
    _checkpointer.setup()   # ensure SQLite checkpoint tables exist
    evict_old_threads()
    load_user_keys()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _prewarm_embeddings)
    yield
    log.info("Job-Assistant API shutting down")


app = FastAPI(
    title="Job Assistant API",
    description=(
        "Resume tailoring system: parse JDs and resumes, retrieve relevant "
        "sections, interactively edit with LLM suggestions, score against "
        "the JD, and generate cover letters / emails / outreach messages."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ----------------------------------------------------------------------
# Request tracing — logs every request's arrival and completion so we can
# see in the console which calls reach the server and whether they finish
# (a request that logs "→ START" but never "← DONE" is hanging server-side;
#  a page that hangs with NO "→ START" line means the request never left the
#  browser, e.g. connection-pool exhaustion).
# ----------------------------------------------------------------------
@app.middleware("http")
async def trace_requests(request: Request, call_next):
    # Use print(flush=True) rather than the logger: uvicorn reconfigures
    # Python logging on startup and can swallow handler output, whereas a
    # flushed print is guaranteed to land in the uvicorn console live.
    t0 = time.perf_counter()
    print(f"[trace] >> START {request.method} {request.url.path}", flush=True)
    try:
        response = await call_next(request)
    except Exception as exc:
        dt = (time.perf_counter() - t0) * 1000
        print(f"[trace] !! ERROR {request.method} {request.url.path}  "
              f"({dt:.0f} ms): {exc!r}", flush=True)
        raise
    dt = (time.perf_counter() - t0) * 1000
    print(f"[trace] << DONE  {request.method} {request.url.path}  "
          f"{response.status_code}  ({dt:.0f} ms)", flush=True)
    return response


# Allow all origins for local dev; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------
app.include_router(jd.router)
app.include_router(resume.router)
app.include_router(edit.router)
app.include_router(generate.router)
app.include_router(score.router)
app.include_router(keys.router)
app.include_router(library.router)
app.include_router(auth.router)
app.include_router(applications.router)
app.include_router(skills.router)


# -----------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------
@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Job Assistant API",
        "docs":    "/docs",
        "redoc":   "/redoc",
    }
