"""Cooperative OCR deadline, isolated per synchronous request/thread context.

Native calls are checked before/after, not forcibly cancelled. Tesseract's
subprocess timeout and Paddle lock acquisition can use the remaining budget.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import time

from app.core.config import get_settings
from app.services.errors import OCR_TIMEOUT, PipelineError

_deadline: ContextVar[float | None] = ContextVar("ocr_deadline", default=None)


def remaining_seconds(*, limit: float | None = None) -> float | None:
    deadline = _deadline.get()
    if deadline is None:
        return limit
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PipelineError(OCR_TIMEOUT, "OCR request deadline exceeded", component="pipeline")
    return remaining if limit is None else min(remaining, limit)


@contextmanager
def ocr_deadline():
    deadline = time.monotonic() + get_settings().ocr_request_timeout_seconds
    previous = _deadline.get()
    token = _deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        yield
        remaining_seconds()
    finally:
        _deadline.reset(token)


@contextmanager
def deadline_lock(lock):
    """Wait only within the active budget; never release while native work runs."""
    timeout = remaining_seconds()
    acquired = lock.acquire() if timeout is None else lock.acquire(timeout=timeout)
    if not acquired:
        raise PipelineError(OCR_TIMEOUT, "OCR deadline exceeded waiting for Paddle lock", component="paddle")
    try:
        remaining_seconds()
        yield
        remaining_seconds()
    finally:
        lock.release()
