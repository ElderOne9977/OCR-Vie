import sys
from unittest.mock import patch


def test_smoke_requires_real_default_ocr_and_current_ui(monkeypatch):
    from scripts import smoke_test
    import json
    calls=[]
    def get(base,path):
        if path=='/':return 200,'<title>OCR-Vie</title>'
        if path=='/health/ready':return 200,json.dumps({'ready':True})
        return 200,'{}'
    def post(base,path,png):
        calls.append(path)
        return 503,'{"detail":{"code":"MODEL_UNAVAILABLE"}}'
    monkeypatch.setattr(smoke_test,'_get',get)
    monkeypatch.setattr(smoke_test,'_post_file',post)
    monkeypatch.setattr(sys,'argv',['smoke'])
    assert smoke_test.main()==1
    assert any('/v1/ocr' in path for path in calls)


def test_environment_check_returns_nonzero_for_missing_dependency(monkeypatch):
    from scripts import check_environment
    with patch('importlib.util.find_spec',return_value=None):
        assert check_environment.main()==1


def test_warmup_runs_inside_server_process(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app as application
    import app.pipelines.router
    calls=[]
    class Router:
        def process(self,*args,**kwargs):calls.append(kwargs)
    monkeypatch.setenv('OCR_WARMUP','true')
    monkeypatch.setattr(app.pipelines.router,'get_router',Router)
    monkeypatch.setattr('app.api.health.health_ready',lambda:{'ready':True})
    with TestClient(application) as client:
        assert client.get('/health/live').status_code==200
    assert len(calls)==1
