#!/usr/bin/env python3
"""Import incoming images into benchmark_data with privacy gate and dedup.

Usage:
  python benchmarks/import_images.py --source incoming_images \
    --data benchmark_data --category camera --split test \
    --source-type real --privacy safe --tags signage diacritics_heavy

Creates unreviewed annotation stubs. Never overwrites existing files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA = ROOT / "benchmark_data"
INCOMING_DIR = ROOT / "benchmark_incoming"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def sha256_file(path: Path) -> str:
    from benchmarks.dataset import sha256_file as _f
    return _f(path)


def image_quality_metadata(path: Path) -> dict:
    """Compute diagnostic image descriptors (no OCR model, cv2 only)."""
    img = cv2.imread(str(path))
    if img is None:
        return {}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "width": w,
        "height": h,
        "megapixels": round((w * h) / 1_000_000, 3),
        "brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def existing_hashes(data_dir: Path, records: list[dict]) -> set[str]:
    hashes: set[str] = set()
    for rec in records:
        rel = rec.get("image", "")
        if not rel:
            continue
        p = Path(rel) if Path(rel).is_absolute() else data_dir / rel
        if p.exists():
            hashes.add(sha256_file(p))
    return hashes


def next_id(existing: set[str], prefix: str) -> str:
    i = 1
    while f"{prefix}_{i:04d}" in existing:
        i += 1
    return f"{prefix}_{i:04d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Directory of images to import")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--category", default="scene")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--source-type", default="real", choices=["real", "synthetic"])
    ap.add_argument("--privacy", default="unknown", choices=["safe", "restricted", "unknown"])
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--license", default="")
    ap.add_argument("--prefix", default=None, help="ID prefix; default = source-type")
    ap.add_argument("--notes", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = (ROOT / source_dir) if not source_dir.exists() else source_dir.resolve()
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    if not source_dir.is_dir():
        print(f"ERROR: source '{source_dir}' is not a directory", file=sys.stderr)
        return 1
    # Privacy gate: real data must be explicitly approved as safe.
    if args.source_type == "real" and args.privacy != "safe":
        print(
            "ERROR: --privacy safe required for --source-type real.\n"
            "Review for PII manually, then re-run with --privacy safe.",
            file=sys.stderr,
        )
        return 1

    img_dir = data_dir / "images"
    splits_dir = data_dir / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    ann_path = data_dir / "annotations.jsonl"

    records = _load_records(ann_path)
    seen_ids = {str(r.get("id", "")) for r in records}
    seen_hashes = existing_hashes(data_dir, records)
    prefix = args.prefix or args.source_type

    candidates = sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not candidates:
        print(f"No supported images found in {source_dir}", file=sys.stderr)
        return 1

    imported, skipped = 0, 0
    for src in candidates:
        h = sha256_file(src)
        if h in seen_hashes:
            print(f"SKIP duplicate {src.name} sha256={h[:12]}")
            skipped += 1
            continue
        rid = next_id(seen_ids, prefix)
        seen_ids.add(rid)
        dest_name = f"{rid}{src.suffix.lower()}"
        dest = img_dir / dest_name
        if dest.exists():
            print(f"SKIP exists {dest_name}")
            skipped += 1
            continue
        rec = {
            "id": rid,
            "image": f"images/{dest_name}",
            "split": args.split,
            "category": args.category,
            "language": "vie",
            "text": "",
            "tags": [args.source_type, *args.tags],
            "regions": [],
            "annotation_level": "image",
            "annotation_complete": False,
            "review_status": "unreviewed",
            "source_type": args.source_type,
            "privacy": args.privacy,
            "license": args.license,
            "sha256": h,
            "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if args.notes:
            rec["notes"] = args.notes
        if args.dry_run:
            print(f"DRY-RUN would import {src.name} -> {dest_name} id={rid}")
        else:
            shutil.copy2(src, dest)
            rec["image_quality"] = image_quality_metadata(dest)
            print(f"IMPORTED {src.name} -> {dest_name} id={rid} unreviewed")
        seen_hashes.add(h)
        records.append(rec)
        imported += 1

    if imported and not args.dry_run:
        # Rewrite the JSONL from the merged record list (never overwrite images).
        tmp = ann_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(ann_path)
        for split in ("train", "val", "test"):
            ids = [str(r["id"]) for r in records if r.get("split") == split]
            (splits_dir / f"{split}.txt").write_text(
                "".join(i + "\n" for i in ids), encoding="utf-8"
            )

    verb = "would import" if args.dry_run else "imported"
    print(f"\n{verb}: {imported}, skipped duplicates/existing: {skipped}")
    print("Next: annotate with `python benchmarks/annotate.py --data benchmark_data`, "
          "then mark reviewed, then audit + validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())