import hashlib
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("variant", ["clean", "rotation", "perspective"])
def test_synthetic_polygon_contains_rendered_ink(variant):
    from scripts.generate_ocr_samples import _render, SAMPLES
    for idx, (text, _) in enumerate(SAMPLES):
        image, poly = _render(text, variant, 42, idx)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
        ink = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 120
        assert np.count_nonzero(ink & (mask == 0)) / np.count_nonzero(ink) < .01, text


def test_synthetic_seed_independent_of_python_hash_seed():
    code = "from scripts.generate_ocr_samples import _render; import hashlib; a,p=_render('Ngay 02/10/2026','rotation',42,7); print(hashlib.sha256(a.tobytes()).hexdigest(),p)"
    outputs = [subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(seed)}) for seed in (1, 2)]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("detected", [False, True])
def test_scene_runner_scores_missing_and_wrong_text(tmp_path, monkeypatch, detected):
    from benchmarks import run_scene_benchmark as runner
    from app.schemas.ocr import BBox, DetectedRegion, RecognitionResult
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    cv2.imwrite(str(data / "images/a.png"), np.full((60, 200, 3), 255, np.uint8))
    poly = [[0, 0], [100, 0], [100, 30], [0, 30]]
    (data / "annotations.jsonl").write_text(json.dumps(dict(id="a", image="images/a.png", text="Xin chào", regions=[dict(polygon=poly, text="Xin chào")])) + "\n", encoding="utf-8")
    class Detector:
        def ensure_ready(self): return True
        def detect(self, image):
            return [DetectedRegion(poly, BBox(0, 0, 100, 30), .99)] if detected else []
    class Rec:
        def status(self): return "ready"
        def recognize(self, image, *, lang, psm=6):
            assert psm == 7
            return RecognitionResult("Sai chữ", "Sai chữ", .61, "tesseract")
    monkeypatch.setattr(runner, "_get_detector", Detector)
    monkeypatch.setattr(runner, "_get_recognizer", lambda _: (Rec(), "tesseract", "vie+eng"))
    monkeypatch.setattr(sys, "argv", ["bench", "--data", str(data), "--output", str(tmp_path / "out"), "--recognizer", "tesseract", '--mode', 'ablation'])
    assert runner.main() == 0
    report = json.loads(next((tmp_path / "out").glob("*/summary.json")).read_text(encoding="utf-8"))
    assert report["end_to_end"]["wer"] == 1.0
    assert report["end_to_end"]["cer"] > 0
    import csv
    rows = list(csv.DictReader(next((tmp_path / "out").glob("*/scene_results.csv")).open(encoding="utf-8")))
    assert float(rows[0]["wer"]) == 1.0
    rec_rows = list(csv.DictReader(next((tmp_path / "out").glob("*/recognition_results.csv")).open(encoding="utf-8")))
    if detected:
        assert float(rec_rows[0]["confidence"]) == .61
        assert report["confidence_bins"]["0.5-0.7"]["avg_cer"] > 0
    else:
        assert rec_rows[0]["pred_text"] == ""
        assert float(rec_rows[0]["cer"]) == 1.0


def test_recognition_runner_uses_line_psm_and_excludes_ignored(tmp_path, monkeypatch):
    from benchmarks import run_recognition_benchmark as runner
    from app.schemas.ocr import RecognitionResult
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images/a.png"), np.full((60, 200, 3), 255, np.uint8))
    poly = [[0, 0], [100, 0], [100, 30], [0, 30]]
    ann = dict(id="a", image="images/a.png", regions=[dict(polygon=poly, text="hello"), dict(polygon=poly, text="ignored", ignore=True)])
    (tmp_path / "annotations.jsonl").write_text(json.dumps(ann) + "\n", encoding="utf-8")
    class Rec:
        def status(self): return "ready"
        def recognize(self, image, *, lang, psm=6):
            assert psm == 7
            return RecognitionResult("hello", "hello", .9, "tesseract")
    monkeypatch.setattr(runner, "_get_recognizer", lambda _: (Rec(), "tesseract", "vie+eng"))
    monkeypatch.setattr(sys, "argv", ["bench", "--data", str(tmp_path), "--output", str(tmp_path / "out")])
    assert runner.main() == 0
    report = json.loads(next((tmp_path / "out").glob("*/summary.json")).read_text(encoding="utf-8"))
    assert report["metrics"]["cer"] == 0
    assert report["n_samples"] == 1


def test_scene_auto_calls_production_router(tmp_path, monkeypatch):
    from benchmarks import run_scene_benchmark as runner
    from types import SimpleNamespace
    from app.schemas.ocr import TimingMs, Provenance
    (tmp_path / 'images').mkdir()
    cv2.imwrite(str(tmp_path / 'images/a.png'), np.full((20, 50, 3), 255, np.uint8))
    (tmp_path / 'annotations.jsonl').write_text(json.dumps(dict(id='a', image='images/a.png', language='vie', text='Xin chào', regions=[]))+'\n', encoding='utf-8')
    calls = []
    class Router:
        def process(self, image_bytes, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(regions=[], text='Xin chào', timing_ms=TimingMs(detection=1, recognition=2, total=4), engines={}, provenance=Provenance('scene', None, 'paddle', 'tesseract', 'none'))
    import app.pipelines.router
    monkeypatch.setattr(app.pipelines.router, 'get_router', Router)
    monkeypatch.setattr(sys, 'argv', ['bench', '--data', str(tmp_path), '--output', str(tmp_path/'out'), '--recognizer', 'auto'])
    assert runner.main() == 0
    assert calls[0]['lang'] == 'vie' and calls[0]['recognizer'] == 'auto'
    report = json.loads(next((tmp_path/'out').glob('*/summary.json')).read_text(encoding='utf-8'))
    assert report['execution_mode'] == 'production_pipeline'
    assert report['end_to_end']['cer'] == 0


@pytest.mark.parametrize('mapped,wrong,expected', [(True,False,1), (True,True,.5), (False,False,None)])
def test_recognition_fields_require_explicit_mapping(tmp_path, monkeypatch, mapped, wrong, expected):
    from benchmarks import run_recognition_benchmark as runner
    from app.schemas.ocr import RecognitionResult
    (tmp_path/'images').mkdir()
    cv2.imwrite(str(tmp_path/'images/a.png'), np.full((40,100,3),255,np.uint8))
    regions=[dict(text=value, polygon=[[0,0],[50,0],[50,20],[0,20]], **({'field_key':key} if mapped else {}))
             for key,value in [('amount','100'),('tax_id','123')]]
    regions.append(dict(text='unrelated',polygon=[[50,0],[99,0],[99,20],[50,20]]))
    (tmp_path/'annotations.jsonl').write_text(json.dumps(dict(id='a',image='images/a.png',regions=regions,fields={'amount':'100','tax_id':'123'}))+'\n',encoding='utf-8')
    class Rec:
        count=0
        def status(self):return 'ready'
        def recognize(self,image,**kwargs):
            self.count+=1
            text='100' if self.count==1 else ('bad' if wrong else '123')
            return RecognitionResult(text,text,.9,'tesseract')
    monkeypatch.setattr(runner,'_get_recognizer',lambda name:(Rec(),'tesseract','vie+eng'))
    monkeypatch.setattr(sys,'argv',['bench','--data',str(tmp_path),'--output',str(tmp_path/'out')])
    assert runner.main()==0
    report=json.loads(next((tmp_path/'out').glob('*/summary.json')).read_text(encoding='utf-8'))
    assert report['metrics']['field_exact_match']==expected
    assert report['field_evaluation']['evaluated']==(2 if mapped else 0)


@pytest.mark.parametrize('mode', ['production', 'ablation'])
def test_scene_all_failed_exits_nonzero(tmp_path, monkeypatch, mode):
    from benchmarks import run_scene_benchmark as runner
    from app.schemas.ocr import BBox, DetectedRegion
    poly = [[0,0],[100,0],[100,30],[0,30]]
    cv2.imwrite(str(tmp_path/'a.png'), np.full((40,100,3),255,np.uint8))
    (tmp_path/'annotations.jsonl').write_text(json.dumps(dict(id='a',image='a.png',text='100',
        regions=[dict(text='100',polygon=poly)]))+'\n',encoding='utf-8')
    class Failing:
        def status(self):return 'ready'
        def process(self,*args,**kwargs):raise RuntimeError('Unavailable')
        def recognize(self,*args,**kwargs):raise RuntimeError('Unavailable')
    class Detector:
        def ensure_ready(self):return True
        def detect(self,image):return [DetectedRegion(poly,BBox(0,0,100,30),.9)]
    import app.pipelines.router
    monkeypatch.setattr(app.pipelines.router,'get_router',Failing)
    monkeypatch.setattr(runner,'_get_detector',Detector)
    monkeypatch.setattr(runner,'_get_recognizer',lambda _: (Failing(),'tesseract','vie+eng'))
    monkeypatch.setattr(sys,'argv',['bench','--data',str(tmp_path),'--output',str(tmp_path/'out'),'--mode',mode])
    assert runner.main() == 1


def test_requested_shapely_backend_does_not_fallback_when_missing(monkeypatch):
    import builtins
    from benchmarks.metrics import polygon_iou
    original = builtins.__import__
    def without_shapely(name,*args,**kwargs):
        if name.startswith('shapely'):
            raise ImportError('shapely not installed')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_shapely)
    square = [[0,0],[10,0],[10,10],[0,10]]
    with pytest.raises(ImportError, match='not installed'):
        polygon_iou(square,square,backend='shapely')


def test_production_all_failed_regions_does_not_count_http_success(tmp_path, monkeypatch):
    from benchmarks import run_scene_benchmark as runner
    from types import SimpleNamespace
    from app.schemas.ocr import TimingMs, Provenance
    poly = [[0,0],[50,0],[50,20],[0,20]]
    cv2.imwrite(str(tmp_path/'a.png'),np.full((20,50,3),255,np.uint8))
    (tmp_path/'annotations.jsonl').write_text(json.dumps(dict(id='a',image='a.png',text='hello',
        regions=[dict(text='hello',polygon=poly)]))+'\n',encoding='utf-8')
    class Router:
        def process(self,*args,**kwargs):
            return SimpleNamespace(regions=[SimpleNamespace(polygon=poly,raw_text='',ocr_confidence=None)],
                text='', timing_ms=TimingMs(total=1), provenance=Provenance('scene',None,'paddle','paddle','none'),
                warnings=['region 0 recognition failed'])
    import app.pipelines.router
    monkeypatch.setattr(app.pipelines.router,'get_router',Router)
    monkeypatch.setattr(sys,'argv',['bench','--data',str(tmp_path),'--output',str(tmp_path/'out')])
    assert runner.main() == 1


@pytest.mark.parametrize('mapped,wrong,expected', [(True,False,1), (True,True,.5), (False,False,None)])
def test_scene_fields_score_predictions_and_detection_misses(tmp_path, monkeypatch, mapped, wrong, expected):
    from benchmarks import run_scene_benchmark as runner
    from types import SimpleNamespace
    from app.schemas.ocr import TimingMs, Provenance
    polygons = [[[x,0],[x+30,0],[x+30,20],[x,20]] for x in (0,40,80)]
    regions = [dict(polygon=poly,text=value,**({'field_key':key} if mapped and key else {}))
               for poly,value,key in zip(polygons,['100','123','other'],['amount','tax_id',None])]
    cv2.imwrite(str(tmp_path/'a.png'),np.full((30,120,3),255,np.uint8))
    (tmp_path/'annotations.jsonl').write_text(json.dumps(dict(id='a',image='a.png',text='100 123 other',
        fields={'amount':'100','tax_id':'123'},regions=regions))+'\n',encoding='utf-8')
    class Router:
        def process(self,*args,**kwargs):
            # Omit second detection to prove a missed field is scored wrong.
            pairs = [(polygons[0],'100'),(polygons[2],'other')] if wrong else list(zip(polygons,['100','123','other']))
            return SimpleNamespace(regions=[SimpleNamespace(polygon=p,raw_text=t,ocr_confidence=.9) for p,t in pairs],
                text=' '.join(t for p,t in pairs),timing_ms=TimingMs(total=1),
                provenance=Provenance('scene',None,'paddle','tesseract','none'),warnings=[])
    import app.pipelines.router
    monkeypatch.setattr(app.pipelines.router,'get_router',Router)
    monkeypatch.setattr(sys,'argv',['bench','--data',str(tmp_path),'--output',str(tmp_path/'out')])
    assert runner.main() == 0
    report=json.loads(next((tmp_path/'out').glob('*/summary.json')).read_text(encoding='utf-8'))
    assert report['end_to_end']['field_exact_match'] == expected
    assert report['field_evaluation']['unmapped'] == (0 if mapped else 2)
