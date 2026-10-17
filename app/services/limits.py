"""Process-local admission limit shared by legacy and hybrid OCR requests."""
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock

from fastapi import HTTPException
from app.core.config import get_settings
from app.services.deadline import ocr_deadline


_lock = Lock()
_limiter = None


def _semaphore(capacity: int) -> BoundedSemaphore:
    global _limiter
    with _lock:
        if _limiter is None or _limiter[0] != capacity:
            _limiter = (capacity, BoundedSemaphore(capacity))
        return _limiter[1]


@contextmanager
def ocr_slot():
    semaphore = _semaphore(get_settings().max_concurrent_ocr)
    if not semaphore.acquire(blocking=False):
        raise HTTPException(503, detail={"code": "OCR_BUSY", "message": "OCR capacity exceeded; retry later"},
                            headers={"Retry-After": "1"})
    try:
        with ocr_deadline():
            yield
    finally:
        semaphore.release()
