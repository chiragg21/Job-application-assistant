# app/core/jd_extractor.py
#
# Extract job-description text from a URL via Jina Reader (r.jina.ai).
# Free, no API key required, handles JS-rendered pages
# (LinkedIn, Indeed, Glassdoor, Naukri, etc.)

from __future__ import annotations

import httpx

from app.utils.logger import get_logger

log = get_logger(__name__)

_JINA_BASE = "https://r.jina.ai/"
_TIMEOUT   = 30   # seconds


def extract_jd_from_url(url: str) -> str:
    """
    Fetch readable plain text from a job-posting URL via Jina Reader.

    Strips navigation, ads, and JS boilerplate — returns the raw article/job
    text suitable for passing directly to parse_jd().

    Raises ValueError on HTTP errors, timeouts, or empty content.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    target = f"{_JINA_BASE}{url}"
    log.info("[jd_extractor] fetching via Jina Reader: %s", url)

    try:
        resp = httpx.get(
            target,
            timeout=_TIMEOUT,
            headers={
                "Accept":          "text/plain",
                "X-Return-Format": "text",
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(
            f"Jina Reader returned HTTP {exc.response.status_code} for {url}"
        ) from exc
    except httpx.TimeoutException:
        raise ValueError(f"Jina Reader timed out after {_TIMEOUT}s for {url}")
    except httpx.RequestError as exc:
        raise ValueError(f"Network error fetching {url}: {exc}") from exc

    text = resp.text.strip()
    if not text:
        raise ValueError(f"Jina Reader returned empty content for {url}")

    log.info("[jd_extractor] extracted %d chars from %s", len(text), url)
    return text
