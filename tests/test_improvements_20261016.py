import io
import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas.ocr import BBox, DetectedRegion
from benchmarks.validate_dataset import validate


@pytest.mark.parametrize('endpoint', ['/v1/ocr', '/v1/ocr/text'])
def test_all_recognition_errors_are_not_empty_success(endpoint):
    detector = Mock()
    detector.name = 'paddle'
    detector.status.return_value = 'ready'
    detector.ensure_ready.return_value = True
    detector.detect.return_value = [DetectedRegion([[1,1],[80,1],[80,40],[1,40]], BBox(1,1,79,39))]
    rec = Mock()
    rec.name = 'paddle'
    rec.status.return_value = 'ready'
    rec.ensure_ready.return_value = True
    rec.recognize.side_effect = RuntimeError('private-path')
    image = io.BytesIO()
    Image.new('RGB', (100,60), 'white').save(image, format='PNG')
    with patch('app.pipelines.router.get_detector', return_value=detector), patch('app.pipelines.router.get_recognizer', return_value=rec):
        response = TestClient(app).post(endpoint+'?route=scene&recognizer=paddle&lang=eng', files={'file':('a.png',image.getvalue(),'image/png')})
    assert response.status_code == 500
    assert 'private-path' not in response.text


def test_classifier_internal_error_is_sanitized():
    image = io.BytesIO()
    Image.new('RGB', (100,60)).save(image, format='PNG')
    with patch('app.classifiers.document_scene_classifier.HeuristicDocumentSceneClassifier.classify', side_effect=RuntimeError('private-path')):
        response = TestClient(app).post('/v1/ocr', files={'file':('a.png',image.getvalue(),'image/png')})
    assert response.status_code == 500
    assert 'private-path' not in response.text


@pytest.mark.parametrize('change', [{'image':17}, {'split':[]}, {'id':[]}, {'tags':[[]]}])
def test_validator_rejects_bad_types_without_crash(tmp_path, change):
    Image.new('RGB',(10,10)).save(tmp_path/'a.png')
    record = dict(id='a',image='a.png',split='test',category='scene',language='vie',text='hello',**{})
    record.update(change)
    path=tmp_path/'annotations.jsonl'
    path.write_text(json.dumps(record),encoding='utf-8')
    assert validate(path) == 1


def test_split_membership_and_missing_ids_are_rejected(tmp_path):
    Image.new('RGB',(10,10)).save(tmp_path/'a.png')
    path=tmp_path/'annotations.jsonl'
    path.write_text(json.dumps(dict(id='a',image='a.png',split='test',category='scene',language='vie',text='x')),encoding='utf-8')
    (tmp_path/'splits').mkdir()
    (tmp_path/'splits'/'train.txt').write_text('a\n')
    assert validate(path) == 1
    (tmp_path/'splits'/'train.txt').write_text('')
    assert validate(path) == 1


def test_freeze_rejects_duplicate_images(tmp_path, monkeypatch):
    from benchmarks.freeze_testset import main
    Image.new('RGB',(10,10)).save(tmp_path/'a.png')
    rows=[dict(id=str(i),image='a.png',split='test',category='scene',language='vie',text='x',source_type='real',review_status='reviewed') for i in range(100)]
    (tmp_path/'annotations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    monkeypatch.setattr('sys.argv',['freeze','--data',str(tmp_path)])
    assert main() == 1
    assert not (tmp_path/'frozen').exists()


@pytest.mark.parametrize('key',['source_type','review_status'])
def test_freeze_rejects_invalid_metadata_without_crash(tmp_path,monkeypatch,key):
    from benchmarks.freeze_testset import main
    Image.new('RGB',(10,10)).save(tmp_path/'a.png')
    row=dict(id='a',image='a.png',split='test',category='scene',language='vie',text='x',source_type='real',review_status='reviewed')
    row[key]=[]
    (tmp_path/'annotations.jsonl').write_text(json.dumps(row),encoding='utf-8')
    monkeypatch.setattr('sys.argv',['freeze','--data',str(tmp_path),'--allow-preliminary'])
    assert main() == 1
