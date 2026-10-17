import json
import threading
from http.server import HTTPServer

import httpx
import pytest

from benchmarks.annotate import make_handler


@pytest.fixture
def annotator(tmp_path):
    record = dict(id='a', image='a.png', split='test', text='A', regions=[])
    path = tmp_path / 'annotations.jsonl'
    path.write_text(json.dumps(record) + '\n', encoding='utf-8')
    (tmp_path / 'a.png').write_bytes(b'image')
    server = HTTPServer(('127.0.0.1', 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}', timeout=3) as client:
        yield client, path, record
    server.shutdown()
    server.server_close()
    thread.join()


def test_annotator_restricts_images_and_uses_revision(annotator):
    client, path, record = annotator
    assert client.get('/image/README.md').status_code == 404
    response = client.get('/api/list')
    revision = response.headers.get('etag')
    assert revision
    before = path.read_bytes()
    assert client.post('/api/save', json=record).status_code == 428
    assert client.post('/api/save', json=record, headers={'If-Match': 'stale'}).status_code == 409
    assert path.read_bytes() == before
    record['text'] = 'B'
    saved = client.post('/api/save', json=record, headers={'If-Match': revision})
    assert saved.status_code == 200
    assert saved.headers['etag'] != revision
    assert client.post('/api/save', json=record, headers={'If-Match': revision}).status_code == 409


def test_corrupt_dataset_blocks_list_and_save(annotator):
    client, path, record = annotator
    path.write_text('{bad json\n', encoding='utf-8')
    before = path.read_bytes()
    response = client.get('/api/list')
    assert response.status_code == 400
    assert 'line 1' in response.json()['error']
    assert client.post('/api/save', json=record, headers={'If-Match': 'stale'}).status_code != 200
    assert path.read_bytes() == before


def test_save_reports_split_write_failure_and_keeps_primary_data(annotator, monkeypatch):
    from pathlib import Path
    client, path, record = annotator
    revision = client.get('/api/list').headers['etag']
    original_write = Path.write_text
    def failing_write(p, *args, **kwargs):
        if p.parent.name == 'splits':
            raise OSError('disk failure')
        return original_write(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', failing_write)
    record['text'] = 'B'
    response = client.post('/api/save', json=record, headers={'If-Match':revision})
    assert response.status_code == 500
    assert 'reload' in response.json()['error'].lower()
    assert json.loads(path.read_text(encoding='utf-8'))['text'] == 'B'
