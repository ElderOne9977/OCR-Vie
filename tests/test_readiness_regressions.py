from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.services import paddle_runtime as runtime
from app.services.paddle_models import DET_MODEL, LATIN_REC_MODEL, EN_REC_MODEL


def test_health_reports_missing_vie_and_shared_load_error(monkeypatch):
    from app.api.health import health_ready
    monkeypatch.setattr(runtime,'is_available',lambda:True)
    monkeypatch.setattr(runtime,'is_model_present',lambda model:False)
    monkeypatch.setattr('app.api.health.get_settings',lambda:Settings(allow_model_download=False))
    with patch.dict(runtime._LOAD_ERRORS,{('TextDetection',DET_MODEL):'failed'},clear=True),patch.object(runtime,'_LOADED',set()),patch('pytesseract.get_tesseract_version',return_value='5'),patch('pytesseract.get_languages',return_value=['eng']):
        body=health_ready()
        assert body['components']['tesseract']=='unavailable'
        assert body['components']['paddle_detector']=='load_failed'


def test_health_status_is_specific_to_requested_recognition_model(monkeypatch):
    from app.recognizers.paddle_recognizer import PaddleRecognizer
    monkeypatch.setattr(runtime,'is_available',lambda:True)
    monkeypatch.setattr(runtime,'is_model_present',lambda name:True)
    with patch.object(runtime,'_LOADED',{('TextRecognition',EN_REC_MODEL)}):
        assert PaddleRecognizer().status(lang='vie')=='not_loaded'
        assert PaddleRecognizer().status(lang='eng')=='ready'


def test_model_presence_requires_actual_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime,'model_dir',lambda name:tmp_path)
    (tmp_path/'README.md').write_text('not a model')
    assert not runtime.is_model_present(DET_MODEL)
    for name in ('inference.json','inference.pdiparams','inference.yml'):
        (tmp_path/name).write_bytes(b'valid')
    assert runtime.is_model_present(DET_MODEL)


def test_http_readiness_returns_503_without_loading(monkeypatch):
    from app.main import app
    monkeypatch.setattr(runtime,'is_available',lambda:False)
    response=TestClient(app).get('/health/ready')
    assert response.status_code==503
    assert response.json()['ready'] is False
