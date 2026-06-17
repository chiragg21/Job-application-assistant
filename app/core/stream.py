# app/core/stream.py
#
# Per-thread SSE event queue.
#
# Used by _resume_suggestions() to push section-level events as each
# parallel LLM call completes, and by GET /edit/{thread_id}/stream to
# tail those events and serve them as Server-Sent Events.
#
# Lifecycle
# ---------
#   reset(thread_id)  — called at the START of every generate_suggestions
#                       pass; discards any stale queue from a previous pass
#                       and installs a fresh one.
#   put(thread_id, event)  — called from worker threads as each section
#                            finishes; no-op when thread_id is None (avoids
#                            conditional checks in the caller).
#   get_or_create(thread_id) — called by the SSE endpoint to obtain the
#                              queue (creating it if the producer started
#                              first and already called put()).
#   discard(thread_id)  — called in the SSE generator's finally block to
#                         release the queue once the stream ends.

import queue
import threading

_queues: dict[str, queue.Queue] = {}
_lock = threading.Lock()


def reset(thread_id: str) -> None:
    """
    Clear any pending events for a new generation pass.

    IMPORTANT: this drains the existing queue *in place* rather than
    replacing it with a new object.  The SSE endpoint captures the queue
    reference once at connect time (it connects *before* the producer
    starts), so swapping in a fresh Queue here would orphan that consumer:
    it would block on the old, now-abandoned queue and never receive the
    terminal ``done`` event — leaving the SSE connection open forever and
    eventually exhausting the browser's per-host connection pool.
    """
    with _lock:
        q = _queues.get(thread_id)
        if q is None:
            _queues[thread_id] = queue.Queue()
            return
    # Drain outside the lock; producers/consumers only touch the queue itself.
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def get_or_create(thread_id: str) -> queue.Queue:
    """Return the queue for this thread, creating it if it doesn't exist yet."""
    with _lock:
        if thread_id not in _queues:
            _queues[thread_id] = queue.Queue()
        return _queues[thread_id]


def put(thread_id: str | None, event: dict) -> None:
    """Push an event.  No-op when thread_id is None (streaming not requested)."""
    if thread_id is None:
        return
    get_or_create(thread_id).put(event)


def discard(thread_id: str) -> None:
    """Remove the queue (called from the SSE endpoint on close/error)."""
    with _lock:
        _queues.pop(thread_id, None)
