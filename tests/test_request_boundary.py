import asyncio
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings, Settings


@pytest.mark.parametrize('declared_length', [None, b'10'])
def test_chunked_upload_stops_before_full_multipart(monkeypatch, declared_length):
    import app.core.config as config
    import starlette.formparsers as multipart
    files = []
    factory = multipart.SpooledTemporaryFile
    def tracked_file(*args, **kwargs):
        file = factory(*args, **kwargs)
        files.append(file)
        return file
    monkeypatch.setattr(multipart, 'SpooledTemporaryFile', tracked_file)
    monkeypatch.setattr(config, '_settings', replace(get_settings(), max_upload_bytes=1100*1024))
    head=b'--audit\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\nContent-Type: image/png\r\n\r\n'
    chunks=[head]+[b'x'*65536]*40+[b'\r\n--audit--\r\n']
    received=[]; sent=[]
    async def run():
        iterator=iter(chunks)
        async def receive():
            chunk=next(iterator)
            received.append(chunk)
            return {'type':'http.request','body':chunk,'more_body':len(received)<len(chunks)}
        async def send(message):sent.append(message)
        headers=[(b'content-type',b'multipart/form-data; boundary=audit')]
        if declared_length is not None:
            headers.append((b'content-length', declared_length))
        scope=dict(type='http',asgi={'version':'3.0'},http_version='1.1',method='POST',scheme='http',
                   path='/v1/ocr',raw_path=b'/v1/ocr',query_string=b'',root_path='',
                   headers=headers,
                   client=('127.0.0.1',1),server=('test',80))
        await app(scope,receive,send)
    asyncio.run(run())
    assert next(m['status'] for m in sent if m['type']=='http.response.start')==413
    assert len(received)<len(chunks)
    assert files and all(file.closed for file in files), 'Interrupted parser must close temporary files'
    assert files[0]._rolled, 'Exercise a disk-spooled upload, not only memory buffer'


def test_total_body_boundary_accepts_exact_cap_rejects_next_byte(monkeypatch):
    import app.core.config as config
    from app.services.request_boundary import RequestBoundary
    monkeypatch.setattr(config, '_settings', replace(get_settings(), max_upload_bytes=64))
    async def downstream(scope, receive, send):
        message = await receive()
        assert len(message['body']) == 64 + 64*1024
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'ok'})
    async def run(size):
        sent = []
        async def receive():return {'type':'http.request','body':b'x'*size,'more_body':False}
        async def send(message):sent.append(message)
        await RequestBoundary(downstream)({'type':'http','method':'POST','headers':[]},receive,send)
        return sent[0]['status']
    assert asyncio.run(run(64+64*1024)) == 200
    assert asyncio.run(run(65+64*1024)) == 413


def test_body_cap_counts_other_files_and_fields(monkeypatch):
    import app.core.config as config
    monkeypatch.setattr(config, '_settings', replace(get_settings(), max_upload_bytes=64))
    response = TestClient(app).post('/v1/ocr', files=[
        ('file', ('a.png', b'valid', 'image/png')),
        ('other', ('b.png', b'x'*65536, 'image/png')),
        ('comment', (None, 'x'*128))])
    assert response.status_code == 413


@pytest.mark.parametrize('query',['response_format=typo','parser=typo','route=typo','recognizer=typo'])
def test_query_typo_rejected_before_ocr(query):
    response=TestClient(app).post('/v1/ocr?'+query,files={'file':('a.png',b'not-an-image','image/png')})
    assert response.status_code==422


def test_internal_exception_is_safe_and_traceable(monkeypatch):
    import app.api.ocr as ocr
    class Router:
        def process(self,*args,**kwargs):raise RuntimeError('private secret path')
    monkeypatch.setattr(ocr,'get_router',Router)
    response=TestClient(app).post('/v1/ocr',files={'file':('a.png',b'bytes','image/png')})
    assert response.status_code==500
    assert 'private secret' not in response.text
    assert response.headers.get('x-request-id')


@pytest.mark.parametrize('kwargs',[{'classifier_min_confidence':float('nan')},{'classifier_min_confidence':2},{'fallback_route':'oops'},
                                  {'scene_recognizer':'typo'}, {'scene_detector':'typo'}, {'document_parser':'typo'}])
def test_invalid_routing_config_fails_fast(kwargs):
    with pytest.raises(ValueError):Settings(**kwargs)
