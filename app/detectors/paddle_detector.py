"""PaddleOCR PP-OCRv5 text detector adapter (polygon regions).

Verified against paddleocr 3.7.0: ``TextDetection(model_name=..., **det_args)``
then ``predict(image)`` returns dict-like results with keys
``dt_polys`` (list of Nx2 polygons) and ``dt_scores`` (list of floats).

Lazy import only. Absent paddleocr -> status ``unavailable``, never fake ``ready``.
"""

from __future__ import annotations

import logging
from app.detectors.base import TextDetector
from app.schemas.ocr import ComponentStatus, DetectedRegion
from app.services.errors import MODEL_UNAVAILABLE, OCR_TIMEOUT, PipelineError
from app.services.polygon_utils import polygon_to_bbox

logger = logging.getLogger("vietnamese-ocr.paddle_detector")


class PaddleDetector(TextDetector):
    name = "paddle_detector"

    def __init__(self) -> None:
        self._status: ComponentStatus = "not_loaded"
        self._load_error: str | None = None
        from app.services.paddle_runtime import is_available
        if not is_available():
            self._status = "unavailable"

    def status(self) -> ComponentStatus:
        # Load state is process-global (predictor cache), so a freshly created
        # adapter must report READY once any sibling instance loaded the model.
        if self._status == "not_loaded":
            from app.services.paddle_runtime import component_status
            from app.services.paddle_models import resolve_det_model
            return component_status('TextDetection', resolve_det_model(None))
        return self._status

    def ensure_ready(self) -> bool:
        """Trigger lazy load; return True only when the model is really loaded."""
        self._ensure_loaded()
        return self._status == "ready"

    def _ensure_loaded(self) -> None:
        """Lazy-load the detector model and update status accordingly."""
        if self._status in ("ready", "loading", "unavailable", "load_failed"):
            return
        self._status = "loading"
        try:
            from app.services.paddle_runtime import get_detector_model

            _ = get_detector_model()
            self._status = "ready"
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, PipelineError) and exc.code == OCR_TIMEOUT:
                self._status = "not_loaded"
                raise
            self._status = "load_failed"
            self._load_error = str(exc)
            logger.warning("PaddleDetector load failed: %s", exc)

    def detect(self, image) -> list[DetectedRegion]:
        self._ensure_loaded()
        if self._status != "ready":
            raise PipelineError(
                MODEL_UNAVAILABLE,
                "Paddle detector not ready ({}). Install paddleocr or check MODELS_DIR.".format(
                    self._status
                ),
                component=self.name,
            )
        from app.services.paddle_runtime import get_detector_model, predict

        predictor = get_detector_model()
        results = predict(predictor, image)
        regions: list[DetectedRegion] = []
        for result in results:
            polys = result.get("dt_polys")
            scores = result.get("dt_scores")
            if polys is None:
                continue
            for idx, poly in enumerate(polys):
                points = _to_points(poly)
                if len(points) < 3:
                    continue
                confidence = None
                if scores is not None and idx < len(scores):
                    confidence = round(float(scores[idx]), 4)
                regions.append(
                    DetectedRegion(
                        polygon=points,
                        bbox=polygon_to_bbox(points),
                        detection_confidence=confidence,
                    )
                )
        return regions


def _to_points(poly) -> list[list[float]]:
    """Normalize a paddle polygon (np.ndarray | list) to list[[x, y], ...]."""
    try:
        arr = poly.tolist() if hasattr(poly, "tolist") else list(poly)
    except (TypeError, ValueError):
        return []
    points: list[list[float]] = []
    for point in arr:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    return points
