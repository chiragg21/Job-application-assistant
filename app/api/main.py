# app/api/main.py
#
# FastAPI application entry point.
#
# Start with:
#   uv run uvicorn app.api.main:app --reload

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import jd, resume, edit, generate, score, keys
from app.utils import get_logger

log = get_logger(__name__)


def _prewarm_embeddings() -> None:
    """
    Instantiate all pipeline singletons and run a dummy embed so the
    420 MB sentence-transformer model is loaded before the first real
    request arrives.  Runs in a thread-pool executor to avoid blocking
    the event loop.
    """
    try:
        from app.core.pipeline import _get_jd_parser, _get_resume_parser, _get_retriever
        jd_parser     = _get_jd_parser()
        resume_parser = _get_resume_parser()
        _get_retriever()
        # Trigger actual model weights load via a no-op embed call
        resume_parser.ef(["warmup"])
        jd_parser.ef(["warmup"])
        log.info("[startup] embedding model pre-warmed")
    except Exception as exc:
        log.warning("[startup] embedding pre-warm failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Job-Assistant API starting up")
    from app.graph.edit_graph import evict_old_threads
    from app.api.routes.keys import load_user_keys
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
