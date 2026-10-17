"""Adapter status + neural fallback (mocked; no paddle required)."""
from unittest.mock import MagicMock, patch

import numpy as np

from app.detectors.paddle_detector import PaddleDetector
from app.recognizers.paddle_recognizer import PaddleRecognizer
from app.schemas.ocr import BBox, DetectedRegion


def test_paddle_detector_unavailable_without_paddle():
    with patch("importlib.util.find_spec", return_value=None):
        d = PaddleDetector()
        assert d.status() == "unavailable"


def test_paddle_recognizer_unavailable_without_paddle():
    with patch("importlib.util.find_spec", return_value=None):
        r = PaddleRecognizer()
        assert r.status() == "unavailable"


def test_neural_scene_fallback_when_detector_unavailable(monkeypatch):
    # Simulate OCR_SCENE_DETECTOR=paddle but detector absent -> router falls back to tesseract path
    from app.core.config import reset_settings
    from app.pipelines.router import OCRRouter

    monkeypatch.setenv("OCR_SCENE_DETECTOR", "paddle")
    monkeypatch.setenv("OCR_SCENE_RECOGNIZER", "tesseract")
    reset_settings()
    try:
        router = OCRRouter.__new__(OCRRouter)
        # Minimal stub classifier to avoid needing real image
        from app.schemas.ocr import ClassificationResult

        router.classifier = MagicMock()
        router.classifier.classify.return_value = ClassificationResult(
            route="scene", confidence=1.0, reason="forced", classifier="manual"
        )
        # Fake image
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        from app.schemas.ocr import TimingMs

        warnings: list[str] = []
        timing = TimingMs()
        # Mock detector to be unavailable
        with patch("app.pipelines.router.get_detector") as mock_det:
            m = MagicMock()
            m.status.return_value = "unavailable"
            m.name = "paddle_detector"
            mock_det.return_value = m
            # Call _run_scene_neural directly
            result = router._run_scene_neural(
                img,
                router.classifier.classify.return_value,
                recognizer="tesseract",
                lang="vie+eng",
                psm=6,
                filename=None,
                warnings=warnings,
                timing=timing,
                started=0.0,
            )
            assert result is None
            # _run_scene_neural reports paddle unavailability; caller _run_scene
            # decides fallback vs MODEL_UNAVAILABLE and appends PADDLE_UNAVAILABLE...
            assert any(("unavailable" in w or "not usable" in w) for w in warnings)
    finally:
        reset_settings()
        monkeypatch.delenv("OCR_SCENE_DETECTOR", raising=False)
        monkeypatch.delenv("OCR_SCENE_RECOGNIZER", raising=False)
