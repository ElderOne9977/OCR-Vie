"""Detector tradeoffs and resource budgets; native inference is stubbed only at its boundary."""
import numpy as np
import pytest

from app.core.config import Settings, reset_settings
from fastapi import HTTPException
from tests.test_ocr_quality_regressions import pipeline


@pytest.fixture(autouse=True)
def fresh_settings():
    reset_settings()
    yield
    reset_settings()


@pytest.mark.parametrize("side", [768, 960, 1280])
def test_detector_side_reaches_native_constructor(monkeypatch, side):
    from app.services import paddle_runtime
    monkeypatch.setenv("DETECTOR_LIMIT_SIDE_LEN", str(side))
    calls = []
    monkeypatch.setattr(paddle_runtime, "_get_or_create_predictor",
                        lambda *args: calls.append(args) or object())
    paddle_runtime.get_detector_model()
    assert calls[0][2]["limit_side_len"] == side
    assert calls[0][2]["limit_type"] == "max"


def test_detector_default_preserves_resolution():
    assert Settings().detector_limit_side_len == 960


@pytest.mark.parametrize("side", [0, 31, 4001])
def test_detector_side_rejects_out_of_range(monkeypatch, side):
    monkeypatch.setenv("DETECTOR_LIMIT_SIDE_LEN", str(side))
    with pytest.raises(ValueError, match="detector_limit_side_len"):
        Settings.from_env()


@pytest.mark.parametrize("polygon", [
    [[0, 0], [10, 0], [10, 10], [0, 10]],
    [[0, 0], [10, 0], [5, 10]],
])
def test_crop_budget_is_checked_before_allocation(monkeypatch, polygon):
    from app.services.polygon_utils import crop_polygon
    calls = []
    monkeypatch.setattr("cv2.warpPerspective", lambda *a, **k: calls.append(a))
    with pytest.raises(HTTPException) as error:
        crop_polygon(np.zeros((20, 20, 3), np.uint8), polygon, max_pixels=99)
    assert error.value.status_code == 413
    assert not calls


def test_exact_crop_budget_preserves_output():
    from app.services.polygon_utils import crop_polygon
    image = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    expected = crop_polygon(image, polygon)
    assert np.array_equal(crop_polygon(image, polygon, max_pixels=100), expected)
    assert crop_polygon(image, [], max_pixels=0) is None


@pytest.mark.parametrize("batch_mode", [True, False])
def test_router_checks_cumulative_crop_budget_before_second_warp(pipeline, monkeypatch, batch_mode):
    from tests.test_ocr_quality_regressions import Detector
    import cv2
    router, image, _, tess = pipeline
    detector = Detector()
    region = detector.detect(None)[0]  # 6000 pixels per crop.
    monkeypatch.setattr(detector, "detect", lambda image: [region, region])
    monkeypatch.setattr("app.pipelines.router.get_detector", lambda name: detector)
    monkeypatch.setattr("app.pipelines.router.get_settings", lambda: Settings(max_image_pixels=10000))
    if batch_mode:
        tess.recognize_many = lambda *a, **k: pytest.fail("Over-budget crops must not start a batch")
    calls = []
    warp = cv2.warpPerspective
    monkeypatch.setattr(cv2, "warpPerspective", lambda *a, **k: calls.append(a) or warp(*a, **k))
    with pytest.raises(HTTPException) as error:
        router.process(image, route="scene", recognizer="paddle_tesseract")
    assert error.value.status_code == 413
    assert len(calls) == 1


def test_tesseract_calls_share_remaining_request_budget(monkeypatch):
    from app.services.limits import ocr_slot
    from app.ocr_service import run_tesseract
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "10")
    now = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    timeouts = []
    def native(source, **kwargs):
        timeouts.append(kwargs["timeout"])
        return "text"
    with ocr_slot():
        assert run_tesseract(native, "one") == "text"
        now[0] += 6
        assert run_tesseract(native, "two") == "text"
    assert timeouts == [10, 4]
    run_tesseract(native, "outside")
    assert timeouts[-1] == 30  # Request context must not leak.


def test_expired_request_does_not_start_another_batch(monkeypatch):
    from pathlib import Path
    from app.services.limits import ocr_slot
    from app.services.errors import PipelineError
    from app.recognizers.tesseract_recognizer import TesseractRecognizer
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("TESSERACT_BATCH_SIZE", "1")
    now, calls, dirs = [100.0], [], []
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    def native(source, **kwargs):
        calls.append(kwargs["timeout"])
        dirs.append(Path(source).parent)
        now[0] += 4
        return dict(text=[""], conf=[-1], page_num=[1], block_num=[0], par_num=[0], line_num=[0])
    monkeypatch.setattr("pytesseract.image_to_data", native)
    with pytest.raises(PipelineError) as error:
        with ocr_slot():
            TesseractRecognizer().recognize_many([np.zeros((10, 10), np.uint8)] * 3)
    assert error.value.code == "OCR_TIMEOUT"
    assert calls == [7, 3]
    assert all(not d.exists() for d in dirs)


def test_expired_request_skips_native_operation_and_releases_slot(monkeypatch):
    from app.services.limits import ocr_slot
    from app.services.errors import PipelineError
    from app.ocr_service import run_tesseract
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("MAX_CONCURRENT_OCR", "1")
    now, calls = [100.0], []
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    with pytest.raises(PipelineError) as error:
        with ocr_slot():
            now[0] += 11
            run_tesseract(lambda *a, **k: calls.append(a), "image")
    assert error.value.status_code == 504
    assert not calls
    with ocr_slot():
        assert run_tesseract(lambda *a, **k: "ok", "image") == "ok"


def test_predictor_result_after_deadline_is_not_success(monkeypatch):
    from app.services.limits import ocr_slot
    from app.services.paddle_runtime import predict
    from app.services.errors import PipelineError
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "10")
    now = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    class Native:
        def predict(self, image):
            now[0] += 11
            yield image
    with pytest.raises(PipelineError) as error:
        with ocr_slot():
            predict(Native(), "image")
    assert error.value.code == "OCR_TIMEOUT"
    # Predictor lock and request context are both released on failure.
    from app.services import paddle_runtime
    assert paddle_runtime._predict_lock.acquire(blocking=False)
    paddle_runtime._predict_lock.release()


def test_predictor_lock_wait_uses_request_deadline(monkeypatch):
    import threading
    from app.services import paddle_runtime
    from app.services.limits import ocr_slot
    from app.services.errors import PipelineError
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "0.02")
    calls = []
    class Native:
        def predict(self, image):
            calls.append(image)
            return [image]
    paddle_runtime._predict_lock.acquire()
    timer = threading.Timer(.1, paddle_runtime._predict_lock.release)
    timer.start()
    try:
        with pytest.raises(PipelineError) as error:
            with ocr_slot():
                paddle_runtime.predict(Native(), "image")
        assert error.value.code == "OCR_TIMEOUT"
        assert not calls
    finally:
        timer.join()


@pytest.mark.parametrize("endpoint,fn,result", [("/ocr", "ocr_image", "text"),
                                                ("/ocr/data", "ocr_image_data", [])])
def test_legacy_deadline_at_return_is_mapped_to_504(monkeypatch, endpoint, fn, result):
    from fastapi.testclient import TestClient
    from app.main import app
    from tests.test_performance_regressions import png
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "10")
    now = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    def late(*args, **kwargs):
        now[0] += 11
        return result
    monkeypatch.setattr(f"app.main.{fn}", late)
    response = TestClient(app).post(endpoint, files={"file": ("x.png", png(), "image/png")})
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "OCR_TIMEOUT"


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_request_deadline_rejects_invalid_setting(monkeypatch, seconds):
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", seconds)
    with pytest.raises(ValueError, match="ocr_request_timeout_seconds"):
        Settings.from_env()


def test_expired_request_does_not_start_model_construction(monkeypatch):
    import sys
    from types import SimpleNamespace
    from app.services import paddle_runtime
    from app.services.limits import ocr_slot
    from app.services.errors import PipelineError
    now, calls = [100.0], []
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "10")
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    for name, value in [("_CACHE", {}), ("_LOAD_ERRORS", {}), ("_LOADED", set()), ("_MODEL_META", {})]:
        monkeypatch.setattr(paddle_runtime, name, value)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=lambda **k: calls.append(k) or object()))
    with pytest.raises(PipelineError) as error:
        with ocr_slot():
            now[0] += 11
            paddle_runtime._get_or_create_predictor("TextDetection", "test_model", {})
    assert error.value.code == "OCR_TIMEOUT"
    assert not calls


@pytest.mark.parametrize("component,getter", [("detector", "get_detector_model"),
                                             ("recognizer", "get_recognition_model")])
def test_initialization_timeout_is_not_a_model_load_failure(monkeypatch, component, getter):
    from app.services import paddle_runtime
    from app.detectors.paddle_detector import PaddleDetector
    from app.recognizers.paddle_recognizer import PaddleRecognizer
    from app.services.errors import PipelineError
    monkeypatch.setattr(paddle_runtime, "is_available", lambda: True)
    def timeout(*a, **k):
        raise PipelineError("OCR_TIMEOUT", "Deadline exceeded")
    monkeypatch.setattr(paddle_runtime, getter, timeout)
    adapter = PaddleDetector() if component == "detector" else PaddleRecognizer()
    with pytest.raises(PipelineError) as error:
        adapter.ensure_ready()
    assert error.value.code == "OCR_TIMEOUT"
    assert adapter._status == "not_loaded"
    assert adapter._load_error is None


def test_paddle_recognizer_preserves_timeout_error(monkeypatch):
    from app.services import paddle_runtime
    from app.recognizers.paddle_recognizer import PaddleRecognizer
    from app.services.errors import PipelineError
    monkeypatch.setattr(paddle_runtime, "get_recognition_model", lambda *a, **k: object())
    def timeout(*a, **k):
        raise PipelineError("OCR_TIMEOUT", "Deadline exceeded")
    monkeypatch.setattr(paddle_runtime, "predict", timeout)
    rec = PaddleRecognizer()
    rec._status = "ready"
    with pytest.raises(PipelineError) as error:
        rec.recognize(np.zeros((10, 20, 3), np.uint8), lang="en")
    assert error.value.status_code == 504


def test_model_initialization_lock_wait_obeys_deadline(monkeypatch):
    import threading
    from app.services import paddle_runtime
    from app.services.limits import ocr_slot
    from app.services.errors import PipelineError
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "0.02")
    key = paddle_runtime._make_key("TextDetection", "test_model", {})
    monkeypatch.setattr(paddle_runtime, "_CACHE", {})
    paddle_runtime._lock.acquire()
    def release():
        paddle_runtime._CACHE[key] = object()
        paddle_runtime._lock.release()
    timer = threading.Timer(.1, release)
    timer.start()
    try:
        with pytest.raises(PipelineError):
            with ocr_slot():
                paddle_runtime._get_or_create_predictor("TextDetection", "test_model", {})
        assert timer.is_alive()  # Timed out before the busy constructor released its lock.
    finally:
        timer.join()


def test_expired_request_skips_crop_warp(monkeypatch):
    from app.services.limits import ocr_slot
    from app.services.polygon_utils import crop_polygon
    from app.services.errors import PipelineError
    monkeypatch.setenv("OCR_REQUEST_TIMEOUT_SECONDS", "10")
    now, calls = [100.0], []
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    def warp(image, matrix, size, **kwargs):
        calls.append(size)
        return np.zeros((size[1], size[0], 3), np.uint8)
    monkeypatch.setattr("cv2.warpPerspective", warp)
    with pytest.raises(PipelineError):
        with ocr_slot():
            now[0] += 11
            crop_polygon(np.zeros((20, 20, 3), np.uint8), [[0, 0], [10, 0], [10, 10], [0, 10]])
    assert not calls
