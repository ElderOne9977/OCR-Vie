"""Phase 3.1 dataset filtering + metadata tests.

Covers:
  - filter_records: split / source / reviewed_only / include_rejected
  - count_records: by_source / by_review / by_split / by_category / by_tag
  - record_source / record_review_status / record_annotation_level fallbacks
  - version metadata (benchmark_data/dataset.json) structure
  - import duplicate protection (existing_hashes)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.dataset import (
    ROOT,
    count_records,
    filter_records,
    record_annotation_level,
    record_review_status,
    record_source,
)


def _rec(rid, *, split="test", source_type=None, tags=None, review=None, category="scene"):
    r = {"id": rid, "split": split, "category": category}
    if source_type is not None:
        r["source_type"] = source_type
    if tags is not None:
        r["tags"] = tags
    if review is not None:
        r["review_status"] = review
    return r


# ---- record_source ----

def test_record_source_tag_synthetic_wins():
    assert record_source(_rec("a", source_type="real", tags=["synthetic"])) == "synthetic"


def test_record_source_source_type():
    assert record_source(_rec("a", source_type="real", tags=["camera"])) == "real"
    assert record_source(_rec("a", source_type="synthetic", tags=[])) == "synthetic"


def test_record_source_default_synthetic():
    assert record_source(_rec("a")) == "synthetic"


# ---- record_review_status ----

def test_record_review_status_known():
    assert record_review_status(_rec("a", review="reviewed")) == "reviewed"
    assert record_review_status(_rec("a", review="rejected")) == "rejected"


def test_record_review_status_default_unreviewed():
    assert record_review_status(_rec("a")) == "unreviewed"
    assert record_review_status(_rec("a", review="bogus")) == "unreviewed"


# ---- record_annotation_level ----

def test_record_annotation_level_explicit():
    r = _rec("a")
    r["annotation_level"] = "region"
    assert record_annotation_level(r) == "region"


def test_record_annotation_level_inferred():
    r = _rec("a")
    r["regions"] = [{"polygon": [[0, 0], [1, 0], [1, 1]], "text": "x"}]
    assert record_annotation_level(r) == "region"
    r2 = _rec("b")
    assert record_annotation_level(r2) == "image"


# ---- filter_records ----

def _mixed():
    return [
        _rec("r1", split="test", source_type="real", review="reviewed"),
        _rec("r2", split="test", source_type="real", review="unreviewed"),
        _rec("r3", split="test", source_type="real", review="rejected"),
        _rec("s1", split="test", source_type="synthetic", review="reviewed"),
        _rec("r4", split="train", source_type="real", review="reviewed"),
        _rec("r5", split="val", source_type="real", review="reviewed"),
    ]


def test_filter_no_args_returns_all():
    # rejected records are excluded by default (include_rejected=False)
    assert len(filter_records(_mixed())) == 5


def test_filter_by_split():
    out = filter_records(_mixed(), split="test", include_rejected=True)
    assert {r["id"] for r in out} == {"r1", "r2", "r3", "s1"}


def test_filter_by_source_real():
    out = filter_records(_mixed(), source="real")
    assert all(record_source(r) == "real" for r in out)
    assert "s1" not in {r["id"] for r in out}


def test_filter_by_source_synthetic():
    out = filter_records(_mixed(), source="synthetic")
    assert {r["id"] for r in out} == {"s1"}


def test_filter_reviewed_only():
    out = filter_records(_mixed(), reviewed_only=True)
    assert all(record_review_status(r) == "reviewed" for r in out)
    assert {r["id"] for r in out} == {"r1", "s1", "r4", "r5"}


def test_filter_rejected_excluded_by_default():
    out = filter_records(_mixed())
    assert "r3" not in {r["id"] for r in out}


def test_filter_include_rejected():
    out = filter_records(_mixed(), include_rejected=True)
    assert "r3" in {r["id"] for r in out}


def test_filter_official_selection():
    """Official quality reporting: test + real + reviewed."""
    out = filter_records(_mixed(), split="test", source="real", reviewed_only=True)
    assert {r["id"] for r in out} == {"r1"}


def test_filter_include_unreviewed_noop_when_reviewed_only():
    a = filter_records(_mixed(), reviewed_only=True)
    b = filter_records(_mixed(), reviewed_only=True, include_unreviewed=True)
    assert {r["id"] for r in a} == {r["id"] for r in b}


# ---- count_records ----

def test_count_records_shapes():
    c = count_records(_mixed())
    assert c["by_source"]["real"] == 5
    assert c["by_source"]["synthetic"] == 1
    assert c["by_review"]["reviewed"] == 4
    assert c["by_review"]["unreviewed"] == 1
    assert c["by_review"]["rejected"] == 1
    assert c["by_split"]["test"] == 4
    assert c["by_split"]["train"] == 1
    assert c["by_split"]["val"] == 1
    assert c["by_category"]["scene"] == 6


def test_count_records_tags():
    recs = [_rec("a", tags=["synthetic", "camera"]), _rec("b", tags=["camera"])]
    c = count_records(recs)
    assert c["by_tag"]["camera"] == 2
    assert c["by_tag"]["synthetic"] == 1


# ---- version metadata ----

def test_dataset_json_metadata_structure():
    p = ROOT / "benchmark_data" / "dataset.json"
    assert p.exists(), "benchmark_data/dataset.json must exist"
    meta = json.loads(p.read_text(encoding="utf-8"))
    for key in ("name", "version", "created_at", "counts", "schema_version"):
        assert key in meta, f"missing metadata key: {key}"
    counts = meta["counts"]
    assert "total" in counts
    assert "by_split" in counts
    assert set(counts["by_split"]) <= {"train", "val", "test"}
    # counts must match annotations.jsonl
    ann = ROOT / "benchmark_data" / "annotations.jsonl"
    if ann.exists():
        rows = [json.loads(l) for l in ann.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert counts["total"] == len(rows)
        live = count_records(rows)
        assert counts["by_split"] == live["by_split"]


def test_dataset_json_privacy_gate_documented():
    meta = json.loads((ROOT / "benchmark_data" / "dataset.json").read_text(encoding="utf-8"))
    assert meta.get("privacy_gate", {}).get("commit_required_privacy") == "safe"


def test_dataset_readme_documents_privacy_gate():
    text = (ROOT / "benchmark_data" / "README.md").read_text(encoding="utf-8")
    assert "Privacy Gate" in text
    assert "privacy" in text.lower()
    assert "--reviewed-only" in text


# ---- import duplicate protection ----

def test_import_existing_hashes_dedup():
    import tempfile
    import numpy as np
    import cv2
    from benchmarks.import_images import existing_hashes

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "images").mkdir()
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp / "images" / "a.png"), img)
        cv2.imwrite(str(tmp / "images" / "b.png"), img)  # identical bytes
        records = [
            {"id": "a", "image": "images/a.png"},
            {"id": "b", "image": "images/b.png"},
        ]
        hashes = existing_hashes(tmp, records)
        # identical images collapse to a single sha256
        assert len(hashes) == 1


def test_import_next_id_increments():
    from benchmarks.import_images import next_id

    existing = {"real_0001", "real_0002"}
    assert next_id(existing, "real") == "real_0003"
    assert next_id(set(), "real") == "real_0001"