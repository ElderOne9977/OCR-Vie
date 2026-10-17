"""PaddleOCR PP-OCRv5 Latin recognizer adapter (polygon-aware, perspective rectify).

Verified against paddleocr 3.7.0:
- ``TextRecognition(model_name=rec_model)`` then ``predict(crop_image)``
  returns dict-like results with keys ``rec_text`` (list of strings)
  and ``rec_score`` (list of floats).
- Accepts BGR or RGB numpy array or PIL Image.

This is a line recognizer. The router segments pages with PaddleDetector,
rectifies each polygon and feeds its crop here; it never submits a full page.

Lazy import only. Absent paddleocr -> status ``unavailable``, never fake ``ready``.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

from app.recognizers.base import TextRecognizer
from app.schemas.ocr import ComponentStatus, RecognitionResult
from app.services.errors import MODEL_UNAVAILABLE, MODEL_LANGUAGE_UNSUPPORTED, OCR_TIMEOUT, RECOGNITION_FAILED, PipelineError
from app.services.normalization import normalize_text
from app.services.paddle_models import resolve_rec_model

logger = logging.getLogger("vietnamese-ocr.paddle_recognizer")


class PaddleRecognizer(TextRecognizer):
    name = "paddle"

    def __init__(self) -> None:
        self._status: ComponentStatus = "not_loaded"
        self._load_error: str | None = None
        from app.services.paddle_runtime import is_available
        if not is_available():
            self._status = "unavailable"

    def status(self, lang: str | None = None) -> ComponentStatus:
        if lang is not None:
            from app.services.paddle_runtime import component_status
            model = resolve_rec_model(lang)
            return component_status('TextRecognition', model) if model else 'unavailable'
        # Load state is process-global (predictor cache), so a freshly created
        # adapter must report READY once any sibling instance loaded the model.
        if self._status == "not_loaded":
            from app.services.paddle_runtime import is_any_model_loaded, get_load_error

            if is_any_model_loaded("TextRecognition"):
                self._status = "ready"
            elif get_load_error('TextRecognition', resolve_rec_model('vie+eng')):
                return 'load_failed'
        return self._status

    def ensure_ready(self, lang: str = "vie+eng") -> bool:
        """Trigger lazy load; return True only when the model is really loaded."""
        self._ensure_loaded(lang)
        if self._status == "ready":
            from app.services.paddle_runtime import get_recognition_model
            get_recognition_model(lang)  # Validate the requested language even after warmup.
        return self._status == "ready"

    def _ensure_loaded(self, lang: str = "vie+eng") -> None:
        """Lazy-load the recognizer model and update status accordingly."""
        if self._status in ("ready", "loading", "unavailable", "load_failed"):
            return
        self._status = "loading"
        try:
            from app.services.paddle_runtime import get_recognition_model

            _ = get_recognition_model(lang)
            self._status = "ready"
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, PipelineError) and exc.code == OCR_TIMEOUT:
                self._status = "not_loaded"
                raise
            if isinstance(exc, PipelineError) and exc.code == MODEL_LANGUAGE_UNSUPPORTED:
                # Weights loaded successfully; only this language is unsupported.
                self._status = "ready"
                raise
            self._status = "load_failed"
            self._load_error = str(exc)
            logger.warning("PaddleRecognizer load failed: %s", exc)

    def recognize(
        self,
        image,
        *,
        lang: str = "vie+eng",
        psm: int = 6,
    ) -> RecognitionResult:
        # Document preprocessing → binary single-channel; Paddle needs 3-channel BGR.
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image.squeeze(axis=2), cv2.COLOR_GRAY2BGR)
        self._ensure_loaded(lang)
        if self._status != "ready":
            raise PipelineError(
                MODEL_UNAVAILABLE,
                "Paddle recognizer not ready ({}). Install paddleocr or check MODELS_DIR.".format(
                    self._status
                ),
                component=self.name,
            )
        from app.services.paddle_runtime import get_recognition_model, predict

        rec_model_name = resolve_rec_model(lang)
        if rec_model_name is None:
            raise PipelineError(
                MODEL_UNAVAILABLE,
                f"Unsupported language for Paddle PP-OCRv5: {lang!r}",
                component=self.name,
            )
        predictor = get_recognition_model(lang)
        try:
            results = predict(predictor, image)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("paddle recognition failed")
            raise PipelineError(RECOGNITION_FAILED, f"Recognition failed: {exc}", component=self.name) from exc

        texts: list[str] = []
        scores: list[float] = []
        for result in results:
            rec_text_list = result.get("rec_text", [])
            rec_score_list = result.get("rec_score", [])
            if not isinstance(rec_text_list, list):
                rec_text_list = [rec_text_list]
            if not isinstance(rec_score_list, list):
                rec_score_list = [rec_score_list]
            for i in range(max(len(rec_text_list), len(rec_score_list))):
                txt = str(rec_text_list[i]).strip() if i < len(rec_text_list) else ""
                sc = float(rec_score_list[i]) if i < len(rec_score_list) else 0.0
                if txt:
                    texts.append(txt)
                    scores.append(sc)

        raw_text = "\n".join(texts)
        normalized_text, _norm_info = normalize_text(raw_text)
        raw_confidence = sum(scores) / len(scores) if scores else None
        ocr_confidence = round(raw_confidence, 4) if raw_confidence is not None else None
        return RecognitionResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            ocr_confidence=ocr_confidence,
            engine="paddle_v5_latin",
            model=rec_model_name,
            raw_confidence=raw_confidence,
        )
