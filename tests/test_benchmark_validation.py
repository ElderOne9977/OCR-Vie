"""Phase 3 benchmark dataset validation + serialization tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from benchmarks.validate_dataset import validate


def _write_jsonl(tmpdir: Path, lines: list[dict]) -> Path:
    p = tmpdir / "annotations.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return p


def test_valid_minimal():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "images").mkdir()
        # create a tiny image
        import numpy as np, cv2
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp / "images" / "a.png"), img)
        lines = [{
            "id": "a", "image": "images/a.png", "split": "test", "category": "scene",
            "language": "vie", "text": "hello", "tags": ["synthetic"],
            "regions": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "hello"}],
        }]
        p = _write_jsonl(tmp, lines)
        assert validate(p) == 0


def test_duplicate_id_fails():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "images").mkdir()
        import numpy as np, cv2
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp / "images" / "a.png"), img)
        lines = [
            {"id": "a", "image": "images/a.png", "split": "test", "category": "scene", "language": "vie", "text": "x", "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "x"}]},
            {"id": "a", "image": "images/a.png", "split": "test", "category": "scene", "language": "vie", "text": "y", "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "y"}]},
        ]
        p = _write_jsonl(tmp, lines)
        assert validate(p) == 1


def test_missing_image_fails():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        lines = [{"id": "a", "image": "images/nope.png", "split": "test", "category": "scene", "language": "vie", "text": "x", "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "x"}]}]
        p = _write_jsonl(tmp, lines)
        assert validate(p) == 1


def test_invalid_split_fails():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "images").mkdir()
        import numpy as np, cv2
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp / "images" / "a.png"), img)
        lines = [{"id": "a", "image": "images/a.png", "split": "production", "category": "scene", "language": "vie", "text": "x", "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "x"}]}]
        p = _write_jsonl(tmp, lines)
        assert validate(p) == 1


def test_duplicate_image_sha256_across_splits_fails():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "images").mkdir()
        import numpy as np, cv2
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp / "images" / "a.png"), img)
        cv2.imwrite(str(tmp / "images" / "b.png"), img)
        lines = [
            {"id": "a", "image": "images/a.png", "split": "test", "category": "scene", "language": "vie", "text": "x", "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "x"}]},
            {"id": "b", "image": "images/b.png", "split": "train", "category": "scene", "language": "vie", "text": "y", "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "y"}]},
        ]
        p = _write_jsonl(tmp, lines)
        assert validate(p) == 1


def test_empty_text_fails():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "images").mkdir()
        import numpy as np, cv2
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp / "images" / "a.png"), img)
        lines = [{"id": "a", "image": "images/a.png", "split": "test", "category": "scene", "language": "vie", "text": "  ", "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "x"}]}]
        p = _write_jsonl(tmp, lines)
        assert validate(p) == 1


def test_serialization_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "images").mkdir()
        import numpy as np, cv2
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp / "images" / "a.png"), img)
        lines = [{"id": "a", "image": "images/a.png", "split": "test", "category": "scene", "language": "vie", "text": "hello", "tags": ["synthetic"], "regions": [{"polygon": [[0,0],[1,0],[1,1],[0,1]], "text": "hello"}]}]
        p = _write_jsonl(tmp, lines)
        assert validate(p) == 0
        # ensure jsonl is readable
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert rows[0]["id"] == "a"


def test_annotation_loaders_refuse_corrupt_lines(tmp_path):
    from benchmarks.annotate import _load_records
    from benchmarks.dataset_summary import _load_records as summary_load
    from benchmarks.audit_annotations import _load as audit_load
    path = tmp_path / 'annotations.jsonl'
    path.write_text('{"id":"a"}\n{bad json\n', encoding='utf-8')
    for loader, arg in ((_load_records, path), (summary_load, tmp_path), (audit_load, path)):
        with pytest.raises(ValueError, match='line 2'):
            loader(arg)


def test_annotation_save_rejects_stale_revision_and_bad_schema(tmp_path):
    from benchmarks.annotate import _save_records
    from benchmarks.dataset import sha256_file
    path = tmp_path / 'annotations.jsonl'
    path.write_text('{"id":"original"}\n', encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='changed'):
        _save_records(path, [], expected_revision='stale')
    with pytest.raises(ValueError):
        _save_records(path, [{'id': 'x', 'split': 'wrong'}], expected_revision=sha256_file(path))
    assert path.read_bytes() == before


def test_annotation_save_rejects_unknown_field_mapping(tmp_path):
    from benchmarks.annotate import _save_records
    path = tmp_path / 'annotations.jsonl'
    path.write_text('original', encoding='utf-8')
    row = dict(id='a', image='a.png', split='test', text='100', fields={'amount': '100'},
               regions=[dict(polygon=[[0,0],[10,0],[10,10],[0,10]], text='100', field_key='typo')])
    with pytest.raises(ValueError, match='field'):
        _save_records(path, [row])
    assert path.read_text() == 'original'


@pytest.mark.parametrize('region', [
    {'polygon': [[0,0],[float('nan'),0],[10,10]], 'text': 'x'},
    {'polygon': [[0,0],[1,1],[2,2]], 'text': 'x'},
    {'polygon': [[0,0],[10,0],[10,10]], 'text': 'x', 'field_key': 'missing'},
])
def test_validator_rejects_invalid_geometry_and_field_mapping(tmp_path, region):
    import cv2, numpy as np
    cv2.imwrite(str(tmp_path/'a.png'), np.zeros((20,20,3), np.uint8))
    row = dict(id='a',image='a.png',split='test',category='scene',language='vie',text='x',regions=[region])
    path = _write_jsonl(tmp_path,[row])
    assert validate(path) == 1
