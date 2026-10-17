#!/usr/bin/env python3
"""Freeze the REAL test set v0.1 (ids + SHA256 + annotations hash + manifest hash).

Gate: requires >=100 REAL REVIEWED test samples. Refuses otherwise unless
--allow-preliminary (then labels the manifest PRELIMINARY NOT REPRESENTATIVE).

Writes benchmark_data/frozen/testset_v<version>.json. Does NOT run any OCR.
No image bytes are copied; only hashes. Safe to commit (no private content).

Usage:
  python benchmarks/freeze_testset.py --data benchmark_data
  python benchmarks/freeze_testset.py --data benchmark_data --version "0.1"
  python benchmarks/freeze_testset.py --data benchmark_data --allow-preliminary
"""
from __future__ import annotations
import argparse
import json
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "benchmark_data"


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze the REAL test set manifest.")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--version", default="0.1")
    ap.add_argument("--allow-preliminary", action="store_true", help="Allow <100 and label as PRELIMINARY.")
    args = ap.parse_args()
    data_dir = Path(args.data)
    ann_path = data_dir / "annotations.jsonl"
    from benchmarks.dataset import load_annotations
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.version):
        print('ERROR: Invalid version', file=sys.stderr)
        return 1
    version_file = data_dir / 'frozen' / f'testset_v{args.version}.json'
    if version_file.exists():
        print('ERROR: Version already exists; choose a new version', file=sys.stderr)
        return 1
    try:
        records = load_annotations(ann_path)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    from benchmarks.dataset import filter_records, build_manifest, MIN_REAL_TEST_REVIEWED

    from benchmarks.validate_dataset import validate
    if validate(ann_path) != 0:
        print('ERROR: Dataset validation failed; nothing frozen', file=sys.stderr)
        return 1

    sel = filter_records(records, split="test", source="real", reviewed_only=True)
    n = len(sel)
    if n < MIN_REAL_TEST_REVIEWED and not args.allow_preliminary:
        print(f"ERROR: Only {n} REAL REVIEWED test samples; need {MIN_REAL_TEST_REVIEWED} for official freeze.", file=sys.stderr)
        print("Use --allow-preliminary to label this as PRELIMINARY NOT REPRESENTATIVE.", file=sys.stderr)
        return 1

    try:
        manifest = build_manifest(sel, data_dir)
        if not sel or not all(entry['image_verified'] for entry in manifest['entries']):
            raise ValueError('Freeze requires nonempty records with verified images')
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    label = f"REAL TEST SET v{args.version}" if n >= MIN_REAL_TEST_REVIEWED else "PRELIMINARY NOT REPRESENTATIVE"
    manifest["label"] = label
    manifest["note"] = "Schema v2: verified image and annotation hashes; no image bytes or raw GT."

    frozen_dir = data_dir / "frozen"
    frozen_dir.mkdir(exist_ok=True)
    version_file = frozen_dir / f"testset_v{args.version}.json"
    try:
        with version_file.open('x', encoding='utf-8') as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
    except FileExistsError:
        print('ERROR: Version already exists', file=sys.stderr)
        return 1
    print(f"FROZEN MANIFEST: {version_file}")
    print(f"Label: {label}")
    print(f"Count: {manifest['count']} (real_reviewed_test)")
    print(f"Manifest hash: {manifest['manifest_hash']}")
    print("Safe to commit (contains IDs and SHA256 only; no private image bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
