"""Resource, language and batch contracts; no Paddle models needed."""
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import reset_settings
from app.main import app
from app.recognizers.tesseract_recognizer import TesseractRecognizer
from tests.test_ocr_quality_regressions import pipeline


@pytest.fixture(autouse=True)
def settings_reset():
    reset_settings()
    yield
    reset_settings()


def png(size=(40, 20)):
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def test_upload_reads_only_configured_limit_plus_one(monkeypatch):
    from app.api.ocr import _read_upload
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "10")
    stream = io.BytesIO(b"x" * 1000)
    upload = UploadFile(file=stream, headers={"content-type": "image/png"})
    with pytest.raises(HTTPException) as error:
        _read_upload(upload)
    assert error.value.status_code == 413
    assert stream.tell() == 11


@pytest.mark.parametrize("decoder", ["app.preprocessing.common.decode_bytes_to_bgr", "app.image_processing.decode_image"])
def test_pixel_limit_applies_before_full_decode(monkeypatch, decoder):
    import importlib
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "100")
    module, name = decoder.rsplit(".", 1)
    fn = getattr(importlib.import_module(module), name)
    calls = []
    monkeypatch.setattr(cv2, "imdecode", lambda *args: calls.append(args) or np.zeros((20, 40, 3), np.uint8))
    with pytest.raises(HTTPException) as error:
        fn(png())
    assert error.value.status_code == 413
    assert not calls


def test_concurrent_legacy_request_is_rejected_and_slot_released(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_OCR", "1")
    entered, release = threading.Event(), threading.Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return "Xin chào"
    monkeypatch.setattr("app.main.ocr_image", blocked)
    def request():
        return TestClient(app).post("/ocr", files={"file": ("x.png", png(), "image/png")})
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request)
        assert entered.wait(5)
        try:
            second = pool.submit(request)
            # Release even when old implementation admits both requests.
            try:
                response = second.result(timeout=1)
            except TimeoutError:
                release.set()
                response = second.result(timeout=5)
            assert response.status_code == 503
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
    assert request().status_code == 200


def test_runtime_applies_configured_device_and_threads(monkeypatch):
    from app.services import paddle_runtime as runtime
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("CPU_THREADS", "3")
    calls = []
    monkeypatch.setattr(runtime, "_get_or_create_predictor", lambda *args: calls.append(args) or object())
    runtime.get_detector_model()
    assert calls[0][2]["device"] == "cpu"
    assert calls[0][2]["cpu_threads"] == 3


def test_document_gray_keeps_small_intensity_details():
    from app.preprocessing.document import preprocess
    image = np.full((20, 20, 3), 255, np.uint8)
    image[10, 10] = 90
    image[10, 11] = 170
    gray = preprocess(image)
    assert gray[10, 10] == 90
    assert gray[10, 11] == 170


def test_incomplete_paddle_decoder_rejects_vietnamese(monkeypatch):
    from app.services import paddle_runtime as runtime
    decoder = SimpleNamespace(paddlex_predictor=SimpleNamespace(post_op=SimpleNamespace(character=list("abcđư"))))
    monkeypatch.setattr(runtime, "_get_or_create_predictor", lambda *args: decoder)
    from app.services.errors import PipelineError
    with pytest.raises(PipelineError, match="Vietnamese") as error:
        runtime.get_recognition_model("vie+eng")
    assert error.value.status_code == 503


def test_warm_paddle_adapter_checks_language_before_detection(monkeypatch):
    from app.services import paddle_runtime as runtime
    from app.recognizers.paddle_recognizer import PaddleRecognizer
    from app.services.errors import PipelineError
    decoder = SimpleNamespace(paddlex_predictor=SimpleNamespace(post_op=SimpleNamespace(character=list("abc"))))
    monkeypatch.setattr(runtime, "_get_or_create_predictor", lambda *args: decoder)
    rec = PaddleRecognizer()
    rec._status = "ready"
    with pytest.raises(PipelineError) as error:
        rec.ensure_ready("vie")
    assert error.value.code == "MODEL_LANGUAGE_UNSUPPORTED"


@pytest.mark.parametrize("recognizer", ["auto", "AUTO", ""])
def test_vietnamese_auto_uses_hybrid(pipeline, recognizer):
    router, image, paddle, tess = pipeline
    out = router.process(image, route="scene", lang="vie+eng", recognizer=recognizer)
    assert out.engines["recognizer"] == "tesseract"
    assert out.engines["detector"] == "paddle_detector"
    assert not paddle.inputs
    assert not out.fallback_used


def test_region_limit_rejects_instead_of_truncating(pipeline, monkeypatch):
    from tests.test_ocr_quality_regressions import Detector
    router, image, _, tess = pipeline
    det = Detector()
    region = det.detect(None)[0]
    monkeypatch.setattr(det, "detect", lambda image: [region] * 257)
    monkeypatch.setattr("app.pipelines.router.get_detector", lambda name: det)
    with pytest.raises(HTTPException) as error:
        router.process(image, route="scene", recognizer="paddle_tesseract")
    assert error.value.status_code == 413
    assert not tess.inputs


def test_health_does_not_import_paddle(monkeypatch):
    import builtins
    from app.api.health import health_ready
    original = builtins.__import__
    calls = []
    def tracked(name, *args, **kwargs):
        if name == "paddleocr":
            calls.append(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", tracked)
    health_ready()
    assert not calls


def test_batch_maps_blank_middle_page_and_per_page_confidence(monkeypatch):
    rec = TesseractRecognizer()
    assert callable(getattr(rec, "recognize_many", None))
    calls, paths = [], []
    def data(source, **kwargs):
        from pathlib import Path
        calls.append(kwargs)
        paths.append(Path(source).parent)
        images = Path(source).read_text(encoding="utf-8").splitlines()
        assert len(images) == 3
        # Saved input must be RGB, not OpenCV BGR.
        assert Image.open(images[0]).getpixel((2, 2)) == (30, 20, 10)
        assert Image.open(images[0]).getpixel((0, 0)) == (255, 255, 255)
        return dict(text=["", "Xin", "chào", "", "", "Việt"], conf=[-1, 90, 80, -1, -1, 100],
                    page_num=[1, 1, 1, 2, 3, 3], block_num=[0, 1, 1, 0, 0, 1],
                    par_num=[0, 1, 1, 0, 0, 1], line_num=[0, 1, 1, 0, 0, 1])
    monkeypatch.setattr("pytesseract.image_to_data", data)
    images = [np.full((20, 40, 3), [10, 20, 30], np.uint8)] * 3
    results = rec.recognize_many(images, lang="vie", psm=7)
    assert [r.raw_text for r in results] == ["Xin chào", "", "Việt"]
    assert [r.ocr_confidence for r in results] == [.85, None, 1.0]
    assert len(calls) == 1
    assert calls[0]["timeout"] > 0
    assert not paths[0].exists()


def test_single_recognition_passes_timeout(monkeypatch):
    monkeypatch.setenv("OCR_TIMEOUT_SECONDS", "2")
    calls = []
    def data(*args, **kwargs):
        calls.append(kwargs)
        return {"text": []}
    monkeypatch.setattr("pytesseract.image_to_data", data)
    TesseractRecognizer().recognize(np.zeros((20, 40), np.uint8))
    assert calls[0]["timeout"] == 2


def test_timeout_returns_explicit_error(monkeypatch):
    def data(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")
    monkeypatch.setattr("pytesseract.image_to_data", data)
    from app.services.errors import PipelineError
    with pytest.raises(PipelineError) as error:
        TesseractRecognizer().recognize(np.zeros((20, 40), np.uint8))
    assert error.value.status_code == 504


def test_batch_rejects_missing_page_instead_of_shifting_results(monkeypatch):
    rec = TesseractRecognizer()
    assert callable(getattr(rec, "recognize_many", None))
    monkeypatch.setattr("pytesseract.image_to_data", lambda *a, **k: dict(text=["Việt"], conf=[90],
                        page_num=[2], block_num=[1], par_num=[1], line_num=[1]))
    from app.services.errors import PipelineError
    with pytest.raises(PipelineError) as error:
        rec.recognize_many([np.zeros((20, 40), np.uint8)] * 2)
    assert error.value.code == "RECOGNITION_FAILED"


@pytest.mark.parametrize("endpoint,function", [("/ocr", "image_to_string"), ("/ocr/data", "image_to_data")])
def test_legacy_timeout_has_same_504_contract(monkeypatch, endpoint, function):
    def timeout(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")
    monkeypatch.setattr(f"pytesseract.{function}", timeout)
    response = TestClient(app).post(endpoint, files={"file": ("x.png", png(), "image/png")})
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "OCR_TIMEOUT"


def test_warmup_defaults_to_vietnamese_hybrid(monkeypatch):
    from scripts import warmup_models
    monkeypatch.setattr(warmup_models, "_print_env", lambda: None)
    monkeypatch.setattr(warmup_models, "_report_model", lambda *args: True)
    monkeypatch.setattr("app.detectors.paddle_detector.PaddleDetector.ensure_ready", lambda self: True)
    monkeypatch.setattr("app.recognizers.tesseract_recognizer.TesseractRecognizer.status", lambda self: "ready")
    monkeypatch.setattr("app.recognizers.paddle_recognizer.PaddleRecognizer.ensure_ready",
                        lambda *a, **k: pytest.fail("Default Vietnamese warmup must not load Latin recognizer"))
    assert warmup_models.main() == 0


def test_timing_reports_decode_and_initialization(pipeline):
    router, image, _, _ = pipeline
    out = router.process(image, route="scene", recognizer="paddle_tesseract")
    timing = out.timing_ms.to_dict()
    assert timing.get("decode") is not None
    assert timing.get("initialization") is not None
    assert timing["total"] >= timing["decode"] + timing["initialization"]


def test_paddle_language_aliases_work_with_tesseract(monkeypatch):
    calls = []
    monkeypatch.setattr("pytesseract.image_to_data", lambda *a, **k: calls.append(k) or {"text": []})
    result = TesseractRecognizer().recognize(np.zeros((20, 40), np.uint8), lang="vi+en")
    assert calls[0]["lang"] == "vie+eng"
    assert result.model == "vie+eng"


def test_admission_limiter_is_singleton_even_at_concurrent_cold_start(monkeypatch):
    from app.services import limits
    constructor = limits.BoundedSemaphore
    start = threading.Barrier(5)
    def slow_constructor(capacity):
        threading.Event().wait(.02)
        return constructor(capacity)
    monkeypatch.setattr(limits, "BoundedSemaphore", slow_constructor)
    def get():
        start.wait(timeout=5)
        return limits._semaphore(17)  # Fresh capacity, never used by OCR requests in tests.
    with ThreadPoolExecutor(max_workers=5) as pool:
        objects = list(pool.map(lambda _: get(), range(5)))
    assert len({id(obj) for obj in objects}) == 1


def test_predictor_lock_covers_generator_consumption():
    from app.services.paddle_runtime import predict
    state = {"active": 0, "peak": 0}
    start = threading.Barrier(4)
    class Predictor:
        def predict(self, image):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            threading.Event().wait(.01)
            yield image
            state["active"] -= 1
    model = Predictor()
    def call(index):
        start.wait(timeout=5)
        return predict(model, index)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(call, range(4))) == [[0], [1], [2], [3]]
    assert state["peak"] == 1


def test_batch_chunk_bound_and_cleanup_on_timeout(monkeypatch):
    from pathlib import Path
    from app.services.errors import PipelineError
    monkeypatch.setenv("TESSERACT_BATCH_SIZE", "2")
    counts, dirs = [], []
    def timeout(source, **kwargs):
        counts.append(len(Path(source).read_text().splitlines()))
        dirs.append(Path(source).parent)
        raise RuntimeError("Tesseract process timeout")
    monkeypatch.setattr("pytesseract.image_to_data", timeout)
    with pytest.raises(PipelineError) as error:
        TesseractRecognizer().recognize_many([np.zeros((20, 40), np.uint8)] * 3)
    assert error.value.status_code == 504
    assert counts == [2]
    assert not dirs[0].exists()


def test_status_reuses_native_tesseract_version_cache(monkeypatch):
    import pytesseract
    calls = []
    version = pytesseract.get_tesseract_version
    monkeypatch.setattr(version, "_result", version)
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: calls.append(a) or b"tesseract 5.4.0\n")
    rec = TesseractRecognizer()
    assert rec.status() == rec.status() == "ready"
    assert len(calls) == 1


def test_missing_tesseract_does_not_load_heavy_detector(pipeline, monkeypatch):
    from tests.test_ocr_quality_regressions import Detector
    from app.services.errors import PipelineError
    router, image, _, tess = pipeline
    tess.state = "unavailable"
    calls = []
    monkeypatch.setattr(Detector, "ensure_ready", lambda self: calls.append(True) or True)
    with pytest.raises(PipelineError) as error:
        router.process(image, route="scene", recognizer="paddle_tesseract")
    assert error.value.status_code == 503
    assert not calls
