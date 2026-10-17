#!/usr/bin/env python3
"""Dataset summary: real/synthetic, reviewed, splits, categories, diacritics, fields.

Fast, no OCR model. Works on any benchmark_data directory.
Usage: python benchmarks/dataset_summary.py --data benchmark_data
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "benchmark_data"


def _load_records(data_dir: Path) -> list[dict]:
    from benchmarks.dataset import load_annotations
    return load_annotations(data_dir / 'annotations.jsonl')


def _diacritic_coverage(records: list[dict]) -> dict:
    from benchmarks.dataset import VIET_SPECIAL_CHARS, VIET_TONES
    chars = VIET_SPECIAL_CHARS + [c.upper() for c in VIET_SPECIAL_CHARS]
    # include tone combining marks via NFD count
    char_count: Counter = Counter()
    sample_containing: Counter = Counter()
    tone_count: Counter = Counter()
    tone_names = VIET_TONES
    for rec in records:
        texts = [str(rec.get("text", ""))]
        for r in rec.get("regions") or []:
            if isinstance(r, dict):
                texts.append(str(r.get("text", "")))
        combined = " ".join(texts)
        seen: set[str] = set()
        for ch in combined:
            if ch in chars:
                char_count[ch] += 1
                seen.add(ch)
            # tone marks: decompose and count combining
            nfd = unicodedata.normalize("NFD", ch)
            for c in nfd:
                if c in tone_names:
                    tone_count[tone_names[c]] += 1
        for s in seen:
            sample_containing[s] += 1
        # also count combining tones in whole string NFD
        nfd_all = unicodedata.normalize("NFD", combined)
        for c in nfd_all:
            if c in tone_names:
                # already counted per char above? avoid double; keep per-string total separate
                pass
    return {
        "char_count": dict(char_count),
        "sample_containing": dict(sample_containing),
        "tone_count": dict(tone_count),
    }


def _critical_field_coverage(records: list[dict]) -> dict:
    from benchmarks.dataset import FIELD_KEYS
    field_sample: Counter = Counter()
    field_total = 0
    for rec in records:
        fields = rec.get("fields") or {}
        if not isinstance(fields, dict):
            continue
        for k in fields:
            field_sample[str(k)] += 1
            field_total += 1
    return {"by_key": dict(field_sample), "total_regions_with_fields": field_total}


def main() -> int:
    import argparse
    # Vietnamese diacritics (e.g. Đ) must not crash on Windows cp1252 consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--json", dest="json_out", default=None, help="Write JSON summary to file")
    args = ap.parse_args()
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        # try repo-root relative
        if (ROOT / args.data).exists():
            data_dir = (ROOT / args.data).resolve()
        else:
            data_dir = data_dir.resolve()
    if data_dir.is_file():
        data_dir = data_dir.parent

    try:
        records = _load_records(data_dir)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    if not records:
        print(f"No records in {data_dir / 'annotations.jsonl'}")
        return 1

    from benchmarks.dataset import filter_records, count_records, record_source, record_review_status

    counts = count_records(records)
    real = [r for r in records if record_source(r) == "real"]
    synth = [r for r in records if record_source(r) == "synthetic"]
    reviewed = [r for r in records if record_review_status(r) == "reviewed"]
    unreviewed = [r for r in records if record_review_status(r) == "unreviewed"]
    rejected = [r for r in records if record_review_status(r) == "rejected"]

    # resolution stats (if image_quality present else try read)
    widths, heights = [], []
    for rec in records:
        iq = rec.get("image_quality") or {}
        if iq.get("width") and iq.get("height"):
            widths.append(int(iq["width"])); heights.append(int(iq["height"]))

    cov = _diacritic_coverage(records)
    field_cov = _critical_field_coverage(records)

    # category plan coverage
    from benchmarks.dataset import CATEGORY_PLAN
    cat_counts = counts["by_category"]
    tag_counts = counts["by_tag"]

    # samples with/without diacritics, mixed vi-en
    diacritics = set("ăâêôơưđĂÂÊÔƠƯĐ")
    with_diac, without_diac, mixed = 0, 0, 0
    for rec in records:
        texts = " ".join([str(rec.get("text", ""))] + [str(r.get("text", "")) for r in rec.get("regions") or []])
        has_diac = any(c in texts for c in diacritics)
        if has_diac:
            with_diac += 1
        else:
            without_diac += 1
        # mixed: has both ascii english words and vietnamese diacritics
        lang = str(rec.get("language", ""))
        tags = rec.get("tags") or []
        if "mixed_vi_en" in tags or lang == "vie+eng" or (has_diac and any(ord(c) < 128 and c.isalpha() for c in texts)):
            # heuristic: if has diacritics and also pure ascii words, count mixed
            if has_diac:
                mixed += 1

    summary = {
        "total": len(records),
        "by_source": counts["by_source"],
        "by_review": counts["by_review"],
        "by_split": counts["by_split"],
        "by_category": counts["by_category"],
        "by_tag": counts["by_tag"],
        "real": len(real),
        "synthetic": len(synth),
        "reviewed": len(reviewed),
        "unreviewed": len(unreviewed),
        "rejected": len(rejected),
        "region_count": sum(len(r.get("regions") or []) for r in records),
        "diacritic_coverage": cov,
        "field_coverage": field_cov,
        "with_diacritics": with_diac,
        "without_diacritics": without_diac,
        "mixed_vi_en": mixed,
        "resolution": {
            "n_with_quality": len(widths),
            "avg_width": round(sum(widths)/len(widths), 1) if widths else None,
            "avg_height": round(sum(heights)/len(heights), 1) if heights else None,
        },
    }

    # pretty print
    print(f"=== DATASET SUMMARY: {data_dir} ===")
    print(f"total={len(records)}  real={len(real)}  synthetic={len(synth)}  reviewed={len(reviewed)}  unreviewed={len(unreviewed)}  rejected={len(rejected)}")
    print(f"splits: {counts['by_split']}")
    print(f"categories: {counts['by_category']}")
    print(f"tags: {counts['by_tag']}")
    print(f"regions: {summary['region_count']}")
    print(f"with_diacritics={with_diac}  without={without_diac}  mixed_vi_en={mixed}")
    print(f"diacritic char_count: {cov['char_count']}")
    print(f"diacritic sample_containing: {cov['sample_containing']}")
    print(f"tone_count: {cov['tone_count']}")
    print(f"fields: {field_cov}")
    if widths:
        print(f"resolution avg: {summary['resolution']['avg_width']}x{summary['resolution']['avg_height']}  n={len(widths)}")
    # coverage report: underrepresented categories (SEVERE n<5, LOW n<10)
    from benchmarks.dataset import coverage_report
    cov_rep = coverage_report(records)
    summary["category_plan_coverage"] = cov_rep
    print("\n--- category plan coverage ---")
    for cat, info in cov_rep["items"].items():
        print(f"  {cat:20s} {info['n']:3d}  {info['flag']}")
    print(f"  (SEVERE n<{cov_rep['severe_threshold']}, LOW n<{cov_rep['low_threshold']}; advisory only, data never suppressed)")
    # progress toward 100
    from benchmarks.dataset import MIN_REAL_TEST_REVIEWED
    real_reviewed_test = len([r for r in records if record_source(r) == "real" and record_review_status(r) == "reviewed" and r.get("split") == "test"])
    summary["real_reviewed_test"] = real_reviewed_test
    summary["min_real_test_reviewed"] = MIN_REAL_TEST_REVIEWED
    print(f"\nReal reviewed test: {real_reviewed_test} / {MIN_REAL_TEST_REVIEWED}  (initial target)")
    if real_reviewed_test < MIN_REAL_TEST_REVIEWED:
        print("  -> COLLECTING (add more real reviewed test samples)")
    else:
        print("  -> READY FOR FIRST BENCHMARK")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
