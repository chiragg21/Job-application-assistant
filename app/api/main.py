# app/api/main.py
#
# FastAPI application entry point.
#
# Start with:
#   uv run uvicorn app.api.main:app --reload

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import jd, resume, edit, generate, score
from app.utils import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Job-Assistant API starting up")
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
