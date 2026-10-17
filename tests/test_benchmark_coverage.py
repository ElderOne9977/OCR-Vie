"""Phase 3.2 coverage + freeze-manifest tests.

Covers:
  - coverage_report flags (SEVERE n<5, LOW n<10, ok)
  - real_reviewed_test_count gate
  - record_tags includes category + tags
  - build_manifest determinism + hash stability + no image bytes
  - freeze_testset refuses <100 without --allow-preliminary
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.dataset import (
    COVERAGE_LOW_THRESHOLD,
    COVERAGE_SEVERE_THRESHOLD,
    MIN_REAL_TEST_REVIEWED,
    ROOT,
    build_manifest,
    coverage_report,
    real_reviewed_test_count,
    record_tags,
)


def _rec(rid, *, tags=None, category="scene", split="test", source="real", review="reviewed", sha256=None):
    r = {"id": rid, "split": split, "category": category}
    if tags is not None:
        r["tags"] = tags
    if source == "synthetic":
        r["tags"] = (tags or []) + ["synthetic"]
    else:
        r["source_type"] = source
    r["review_status"] = review
    if sha256:
        r["sha256"] = sha256
    return r


# ---- thresholds sanity ----

def test_thresholds_are_documented_values():
    assert COVERAGE_SEVERE_THRESHOLD == 5
    assert COVERAGE_LOW_THRESHOLD == 10
    assert MIN_REAL_TEST_REVIEWED == 100


# ---- record_tags ----

def test_record_tags_includes_category_and_tags():
    t = record_tags(_rec("a", tags=["camera", "blur"], category="signage"))
    assert {"camera", "blur", "signage"} <= t


# ---- coverage_report ----

def test_coverage_report_severe_flag():
    recs = [_rec(f"a{i}", tags=["signage"]) for i in range(3)]
    rep = coverage_report(recs)
    assert rep["items"]["signage"]["n"] == 3
    assert rep["items"]["signage"]["flag"] == "SEVERE"


def test_coverage_report_low_flag():
    recs = [_rec(f"a{i}", tags=["blur"]) for i in range(7)]
    rep = coverage_report(recs)
    assert rep["items"]["blur"]["flag"] == "LOW"


def test_coverage_report_ok_flag():
    recs = [_rec(f"a{i}", tags=["camera"]) for i in range(12)]
    rep = coverage_report(recs)
    assert rep["items"]["camera"]["flag"] == "ok"


def test_coverage_report_zero_is_severe():
    rep = coverage_report([])
    assert rep["items"]["clean_print"]["n"] == 0
    assert rep["items"]["clean_print"]["flag"] == "SEVERE"


def test_coverage_report_never_suppresses():
    # advisory only: every plan item present in output even at n=0
    rep = coverage_report([])
    assert len(rep["items"]) >= 15


# ---- real_reviewed_test_count ----

def test_real_reviewed_test_count_gate():
    recs = [
        _rec("r1", split="test", source="real", review="reviewed"),
        _rec("r2", split="test", source="real", review="unreviewed"),
        _rec("r3", split="train", source="real", review="reviewed"),
        _rec("s1", split="test", source="synthetic", review="reviewed"),
    ]
    assert real_reviewed_test_count(recs) == 1


def test_synthetic_does_not_count():
    recs = [_rec(f"s{i}", source="synthetic", review="reviewed") for i in range(50)]
    assert real_reviewed_test_count(recs) == 0


# ---- build_manifest ----

def test_build_manifest_deterministic_hash():
    recs = [_rec("b", sha256="aa"), _rec("a", sha256="bb")]
    m1 = build_manifest(recs)
    m2 = build_manifest(list(reversed(recs)))
    assert m1["manifest_hash"] == m2["manifest_hash"]
    assert m1["count"] == 2
    # sorted by id
    assert [e["id"] for e in m1["entries"]] == ["a", "b"]


def test_build_manifest_hash_changes_with_content():
    a = build_manifest([_rec("a", sha256="aa")])
    b = build_manifest([_rec("a", sha256="cc")])
    assert a["manifest_hash"] != b["manifest_hash"]


def test_build_manifest_contains_no_image_bytes():
    m = build_manifest([_rec("a", sha256="aa")])
    blob = json.dumps(m)
    assert "base64" not in blob
    assert 'text' not in m['entries'][0]
    assert 'regions' not in m['entries'][0]


# ---- freeze gate ----

def test_freeze_refuses_below_100(tmp_path):
    data = tmp_path / "benchmark_data"
    data.mkdir()
    from PIL import Image
    for i in range(3):
        Image.new('RGB', (10,10), (i,0,0)).save(data / f'sample{i}.png')
    (data / "annotations.jsonl").write_text(
        "\n".join(json.dumps({**_rec(f"r{i}"), 'image':f'sample{i}.png','language':'vie','text':'x'}) for i in range(3)),
        encoding="utf-8",
    )
    r = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "freeze_testset.py"), "--data", str(data)],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "need 100" in (r.stderr + r.stdout)


def test_freeze_allow_preliminary_labels(tmp_path):
    data = tmp_path / "benchmark_data"
    data.mkdir()
    from PIL import Image
    for i in range(3):
        Image.new('RGB', (10,10), (i,0,0)).save(data / f'sample{i}.png')
    (data / "annotations.jsonl").write_text(
        "\n".join(json.dumps({**_rec(f"r{i}"), 'image': f'sample{i}.png', 'language':'vie', 'text':'x'}) for i in range(3)),
        encoding="utf-8",
    )
    r = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "freeze_testset.py"), "--data", str(data), "--allow-preliminary"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    out = json.loads((data / "frozen" / "testset_v0.1.json").read_text(encoding="utf-8"))
    assert out["label"] == "PRELIMINARY NOT REPRESENTATIVE"


def test_manifest_hash_tracks_gt_and_actual_image(tmp_path):
    import copy
    image = tmp_path / 'a.png'
    image.write_bytes(b'first')
    row = {**_rec('a'), 'image': 'a.png', 'text': 'A', 'regions': [], 'fields': {'amount': '100'}}
    original = build_manifest([row], tmp_path)
    changed = copy.deepcopy(row)
    changed['fields']['amount'] = '200'
    assert build_manifest([changed], tmp_path)['manifest_hash'] != original['manifest_hash']
    changed = copy.deepcopy(row)
    changed['text'] = 'B'
    assert build_manifest([changed], tmp_path)['manifest_hash'] != original['manifest_hash']
    image.write_bytes(b'second')
    assert build_manifest([row], tmp_path)['manifest_hash'] != original['manifest_hash']
    with pytest.raises(ValueError, match='SHA256'):
        build_manifest([{**row, 'sha256': 'stale'}], tmp_path)


def test_freeze_never_overwrites_existing_version(tmp_path, monkeypatch):
    from benchmarks import freeze_testset
    (tmp_path / 'annotations.jsonl').write_text('', encoding='utf-8')
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    target = frozen / 'testset_v0.1.json'
    target.write_bytes(b'original')
    monkeypatch.setattr(sys, 'argv', ['freeze', '--data', str(tmp_path), '--allow-preliminary'])
    assert freeze_testset.main() != 0
    assert target.read_bytes() == b'original'
