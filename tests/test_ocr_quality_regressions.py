"""Regressions for page/line contracts and cache-independent engine selection."""
import cv2
import numpy as np
import pytest
from app.core.config import Settings
from app.pipelines.router import OCRRouter
from app.schemas.ocr import BBox, DetectedRegion, RecognitionResult
from app.services.errors import PipelineError


class Recognizer:
    def __init__(self, name, state="ready"):
        self.name, self.state, self.inputs = name, state, []

    def status(self):
        return self.state

    def ensure_ready(self, lang="vie+eng"):
        if self.state == "not_loaded":
            self.state = "ready"
        return self.state == "ready"

    def recognize(self, image, *, lang, psm):
        self.inputs.append((image.shape, psm))
        text = "Xin chào Việt Nam"
        engine = "paddle_v5_latin" if self.name == "paddle" else self.name
        return RecognitionResult(text, text, .95, engine)


class Detector:
    name = "paddle_detector"
    def status(self): return "ready"
    def ensure_ready(self): return True
    def detect(self, image):
        return [DetectedRegion([[10, 10], [210, 10], [210, 40], [10, 40]], BBox(10, 10, 200, 30), .9)]


@pytest.fixture
def pipeline(monkeypatch):
    from app.detectors.full_frame_detector import FullFrameDetector
    paddle = Recognizer("paddle", "not_loaded")
    tess = Recognizer("tesseract")
    monkeypatch.setattr("app.pipelines.router.get_settings", lambda: Settings())
    monkeypatch.setattr("app.pipelines.router.get_recognizer", lambda name: tess if name == "tesseract" else paddle)
    monkeypatch.setattr("app.pipelines.router.get_detector", lambda name: FullFrameDetector() if name == "full_frame" else Detector())
    _, png = cv2.imencode(".png", np.full((1400, 1000, 3), 255, np.uint8))
    return OCRRouter(), png.tobytes(), paddle, tess


def test_document_auto_is_stable_across_paddle_warmup(pipeline):
    router, png, paddle, tess = pipeline
    cold = router.process(png)
    paddle.state = "ready"
    warm = router.process(png)
    assert cold.engines == warm.engines
    assert warm.engines["recognizer"] == "tesseract"
    assert not cold.fallback_used and not cold.warnings
    assert not paddle.inputs


@pytest.mark.parametrize("state", ["not_loaded", "ready"])
def test_explicit_document_paddle_uses_color_line_crops(pipeline, state):
    router, png, paddle, tess = pipeline
    paddle.state = state
    out = router.process(png, route="document", recognizer="paddle_v5_latin", return_structure=True)
    assert out.engines["detector"] == "paddle_detector"
    assert out.engines["recognizer"] == "paddle_v5_latin"
    assert paddle.inputs[0][0] == (30, 200, 3)
    assert not tess.inputs
    assert out.provenance.route == "document"
    assert out.document is not None


def test_explicit_document_paddle_unavailable_does_not_switch_engine(pipeline):
    router, png, paddle, tess = pipeline
    paddle.state = "unavailable"
    with pytest.raises(PipelineError) as err:
        router.process(png, route="document", recognizer="paddle_v5_latin")
    assert err.value.status_code == 503
    assert not tess.inputs


def test_partial_neural_error_does_not_expose_exception(pipeline, monkeypatch):
    router, png, paddle, tess = pipeline
    recognize = paddle.recognize
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('private secret path')
        return recognize(*args, **kwargs)
    detect = Detector.detect
    monkeypatch.setattr(Detector, 'detect', lambda self, image: detect(self, image) * 2)
    monkeypatch.setattr(paddle, 'recognize', fail)
    out = router.process(png, route='scene', recognizer='paddle_v5_latin', lang='eng')
    assert out.warnings
    assert out.text
    assert 'private secret path' not in str(out.warnings)


@pytest.mark.parametrize("route", ["scene", "document"])
def test_hybrid_uses_detector_and_tesseract_psm7(pipeline, route):
    router, png, paddle, tess = pipeline
    out = router.process(png, route=route, recognizer="paddle_tesseract")
    assert out.engines["detector"] == "paddle_detector"
    assert out.provenance.recognizer == "tesseract"
    assert tess.inputs == [((30, 200, 3), 7)]
    assert not paddle.inputs


def test_tesseract_single_pass_preserves_lines_and_normalizes_confidence(monkeypatch):
    from app.recognizers.tesseract_recognizer import TesseractRecognizer
    calls = []
    def data(*args, **kwargs):
        calls.append(kwargs)
        return dict(text=["", "Xin", "chào", "Việt", "Nam"], conf=[-1, 90, 80, 100, 90],
                    page_num=[1]*5, block_num=[0,1,1,1,1], par_num=[0,1,1,1,1], line_num=[0,1,1,2,2])
    monkeypatch.setattr("pytesseract.image_to_data", data)
    monkeypatch.setattr("pytesseract.image_to_string", lambda *a, **k: pytest.fail("second OCR pass"))
    result = TesseractRecognizer().recognize(np.zeros((60, 200, 3), np.uint8))
    assert result.raw_text == "Xin chào\nViệt Nam"
    assert result.ocr_confidence == .9
    assert result.raw_confidence == 90.0
    assert len(calls) == 1


@pytest.mark.parametrize("scene,document", [("auto", "tesseract"), ("paddle_tesseract", "paddle_tesseract")])
def test_health_reports_document_and_configured_scene(monkeypatch, scene, document):
    from app.api.health import health_ready
    from tests.test_engine_policy import _patch_health
    _patch_health(monkeypatch, paddle_ready=True, tesseract_ready=False)
    monkeypatch.setattr("app.api.health.get_settings", lambda: Settings(scene_recognizer=scene, document_recognizer=document))
    report = health_ready()
    assert report["primary_document_pipeline"]["status"] == "unavailable"
    if scene == "paddle_tesseract":
        assert report["ready"] is False
        assert report["primary_scene_pipeline"]["recognizer"] == "tesseract"
