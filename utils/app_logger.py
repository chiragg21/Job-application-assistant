"""
logger.py
---------
Centralized logging setup for job-application-assistant.

Usage:
    from logger import get_logger

    log = get_logger(__name__)
    log.info("JD ingested", extra={"job_id": 42, "company": "Google"})
    log.warning("Low similarity score", extra={"score": 0.61})
    log.error("ChromaDB insert failed", extra={"collection": "jd_chunks"})

Log files:
    logs/app.log      — all levels, JSON format, rotates at 5MB (keeps 3)
    logs/error.log    — ERROR and above only, for quick debugging
    logs/llm.log      — LLM call tracking only (tokens, model, cost proxy)

Environment variable LOG_LEVEL overrides default (INFO):
    LOG_LEVEL=DEBUG python main.py
"""

import json
import logging
import logging.handlers
import os
import pathlib
import traceback
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_DIR = pathlib.Path("logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Separate logger name for LLM calls so they go to llm.log
LLM_LOGGER_NAME = "llm"


# ---------------------------------------------------------------------------
# Custom JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Emits one JSON object per line — easy to grep, parse, or ship to
    any log aggregator later (Loki, Datadog, etc.) without reformatting.

    Each line looks like:
    {
        "ts": "2025-02-28T10:32:01Z",
        "level": "INFO",
        "logger": "pipelines.ingest_jd",
        "message": "JD ingested",
        "job_id": 42,
        "company": "Google"
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra= fields passed by the caller
        skip_keys = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "levelname", "levelno", "lineno",
            "message", "module", "msecs", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "taskName", "thread", "threadName",
        }
        for key, val in record.__dict__.items():
            if key not in skip_keys:
                log_obj[key] = val

        # Attach exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_obj["stack"] = self.formatStack(record.stack_info)

        return json.dumps(log_obj, default=str)


# ---------------------------------------------------------------------------
# Readable Console Formatter
# ---------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):
    """
    Human-readable format for terminal output with color coding by level.
    Extra fields from extra={} are appended as key=value pairs.
    """

    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    SKIP_KEYS = {
        "args", "asctime", "created", "exc_info", "exc_text",
        "filename", "funcName", "levelname", "levelno", "lineno",
        "message", "module", "msecs", "msg", "name", "pathname",
        "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<8}{self.RESET}"
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        base = f"{ts} {level} [{record.name}] {record.getMessage()}"

        # Append extra fields
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self.SKIP_KEYS
        }
        if extras:
            extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extras.items())
            base += extra_str

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

_initialized = False


def _setup_logging():
    """Run once — sets up handlers on the root logger."""
    global _initialized
    if _initialized:
        return

    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter individually

    # -- Console handler (INFO+ by default, respects LOG_LEVEL) -----------
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    # -- app.log: rotating JSON, all levels --------------------------------
    app_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(JSONFormatter())
    root.addHandler(app_handler)

    # -- error.log: ERROR and above only -----------------------------------
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "error.log",
        maxBytes=2 * 1024 * 1024,   # 2 MB
        backupCount=2,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root.addHandler(error_handler)

    # -- llm.log: dedicated handler for LLM tracking ----------------------
    llm_file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "llm.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    llm_file_handler.setLevel(logging.DEBUG)
    llm_file_handler.setFormatter(JSONFormatter())

    llm_logger = logging.getLogger(LLM_LOGGER_NAME)
    llm_logger.addHandler(llm_file_handler)
    llm_logger.propagate = True  # still shows on console + app.log

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Call this at the top of every module:
        log = get_logger(__name__)
    """
    _setup_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# LLM Call Logger — structured helper
# ---------------------------------------------------------------------------

def log_llm_call(
    *,
    pipeline: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    job_id: int | None = None,
    application_id: int | None = None,
    output_type: str | None = None,
    error: str | None = None,
):
    """
    Structured logger for every LLM API call.
    Writes to llm.log + app.log + console.

    Usage:
        log_llm_call(
            pipeline="cover_letter",
            model="qwen/qwen-2.5-72b-instruct",
            prompt_tokens=512,
            completion_tokens=340,
            job_id=42,
            application_id=7,
            output_type="cover_letter",
        )
    """
    _setup_logging()
    llm_log = logging.getLogger(LLM_LOGGER_NAME)

    total_tokens = input_tokens + output_tokens
    level = logging.ERROR if error else logging.INFO
    message = f"LLM call {'failed' if error else 'completed'} [{pipeline}]"

    llm_log.log(
        level,
        message,
        extra={
            "pipeline": pipeline,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "job_id": job_id,
            "application_id": application_id,
            "output_type": output_type,
            "error": error,
        },
    )


# ---------------------------------------------------------------------------
# Exception logger helper
# ---------------------------------------------------------------------------

def log_exception(logger: logging.Logger, message: str, **extra):
    """
    Log a full exception with traceback and any extra context.
    Call inside an except block — captures current exception automatically.

    Usage:
        try:
            ...
        except Exception:
            log_exception(log, "ChromaDB insert failed", collection="jd_chunks")
    """
    logger.error(
        message,
        exc_info=True,
        stack_info=False,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log = get_logger(__name__)

    log.debug("Debug message — only in app.log unless LOG_LEVEL=DEBUG")
    log.info("App started")
    log.info("JD ingested", extra={"job_id": 42, "company": "Google", "role": "ML Engineer"})
    log.warning("Similarity below threshold", extra={"score": 0.61, "threshold": 0.85})
    log.error("SQLite write failed", extra={"table": "applications"})

    log_llm_call(
        pipeline="cover_letter",
        model="qwen/qwen-2.5-72b-instruct",
        input_tokens=512,
        output_tokens=340,
        job_id=42,
        application_id=7,
        output_type="cover_letter",
    )

    try:
        raise ValueError("Schema mismatch in resume_edits")
    except Exception:
        log_exception(log, "Unexpected error during resume edit storage", table="resume_edits")

    print("\nCheck logs/ folder for app.log, error.log, llm.log")