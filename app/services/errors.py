"""API error model. No stack traces to client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


INVALID_INPUT = "INVALID_INPUT"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
MODEL_LANGUAGE_UNSUPPORTED = "MODEL_LANGUAGE_UNSUPPORTED"
OCR_TIMEOUT = "OCR_TIMEOUT"
DETECTION_FAILED = "DETECTION_FAILED"
RECOGNITION_FAILED = "RECOGNITION_FAILED"
DOCUMENT_PARSE_FAILED = "DOCUMENT_PARSE_FAILED"
INTERNAL_ERROR = "INTERNAL_ERROR"

_STATUS = {
    INVALID_INPUT: 400,
    FILE_TOO_LARGE: 413,
    UNSUPPORTED_FILE_TYPE: 400,
    CLASSIFICATION_FAILED: 422,
    MODEL_UNAVAILABLE: 503,
    MODEL_LANGUAGE_UNSUPPORTED: 503,
    OCR_TIMEOUT: 504,
    DETECTION_FAILED: 500,
    RECOGNITION_FAILED: 500,
    DOCUMENT_PARSE_FAILED: 500,
    INTERNAL_ERROR: 500,
}


@dataclass
class PipelineError(Exception):
    code: str
    message: str
    component: str | None = None

    @property
    def status_code(self) -> int:
        return _STATUS.get(self.code, 500)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": 'OCR processing failed' if self.status_code == 500 else self.message,
                "component": self.component,
            }
        }
