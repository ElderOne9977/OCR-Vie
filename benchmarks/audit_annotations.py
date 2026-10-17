#!/usr/bin/env python3
"""Audit benchmark_data annotations for completeness and consistency.

Checks (stdlib only, shapely optional):
  ERROR: missing text, duplicate id, invalid split, image missing, sha256 dup,
         polygon <3 pts / non-numeric / non-finite / zero area / out of bounds
  WARNING: extreme tiny/large boxes, duplicate text regions, overlapping dup
           polygons, unreviewed, rejected, no-diacritics samples, ignored regions
  INFO: category/tag/split distributions, region counts, privacy/license

Exit 1 on ERROR, 0 otherwise. Mirrors validate_dataset.py but richer QA.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "benchmark_data"

VALID_SPLITS = {"train", "val", "test"}


def _sha256_file(p: Path) -> str:
    from benchmarks.dataset import sha256_file
    return sha256_file(p)


def _poly_area(poly: list) -> float:
    try:
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
    except Exception:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _poly_bbox(poly: list) -> tuple[float, float, float, float]:
    xs = [float(pt[0]) for pt in poly]
    ys = [float(pt[1]) for pt in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    from benchmarks.dataset import load_annotations
    return load_annotations(path)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    data_dir = Path(args.data)
    if data_dir.is_file():
        data_dir = data_dir.parent
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir) if not (ROOT / args.data).exists() else (ROOT / args.data).resolve()
        if not data_dir.is_dir():
            data_dir = Path(args.data).resolve().parent if Path(args.data).suffix else Path(args.data).resolve()
    # normalize: if user passed file path
    if (data_dir / "annotations.jsonl").exists():
        ann_path = data_dir / "annotations.jsonl"
        base = data_dir
    else:
        ann_path = Path(args.data)
        base = ann_path.parent
        if ann_path.is_dir():
            ann_path = ann_path / "annotations.jsonl"
            base = Path(args.data)
    if not ann_path.exists():
        print(f"ERROR: {ann_path} not found", file=sys.stderr)
        return 1
    base = ann_path.parent

    try:
        records = _load(ann_path)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    errors, warnings, infos = [], [], []

    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    cat_counter: Counter = Counter()
    tag_counter: Counter = Counter()
    split_counter: Counter = Counter()
    diacritics = set("ăâêôơưđĂÂÊÔƠƯĐ")
    no_diacritic_ids: list[str] = []
    region_count = 0
    ignored_regions = 0
    unreviewed = 0
    rejected = 0

    for rec in records:
        rid = str(rec.get("id", "?"))
        if rid in seen_ids:
            errors.append(f"duplicate id '{rid}'")
        seen_ids.add(rid)
        split_counter[str(rec.get("split", "?"))] += 1
        cat_counter[str(rec.get("category", "?"))] += 1
        for t in rec.get("tags") or []:
            tag_counter[str(t)] += 1
        text = rec.get("text", "")
        if not isinstance(text, str) or not text.strip():
            # allow empty only when annotation incomplete and image-level pending
            if rec.get("annotation_complete"):
                errors.append(f"{rid}: text missing/empty but annotation_complete=true")
            else:
                warnings.append(f"{rid}: text empty (annotation incomplete)")
        # diacritics presence
        combined = " ".join([str(text)] + [str(r.get("text", "")) for r in rec.get("regions") or []])
        if not any(c in combined for c in diacritics):
            no_diacritic_ids.append(rid)
        # review status
        rs = rec.get("review_status", "unreviewed")
        if rs == "unreviewed":
            unreviewed += 1
        elif rs == "rejected":
            rejected += 1
        # image exists
        img_rel = rec.get("image", "")
        img_path = (base / img_rel).resolve() if isinstance(img_rel, str) and img_rel else None
        if img_path is None or not img_path.exists():
            # also try repo root
            alt = ROOT / str(img_rel) if img_rel else None
            if alt and alt.exists():
                img_path = alt
            else:
                errors.append(f"{rid}: image not found '{img_rel}'")
                img_path = None
        W = H = None
        if img_path and img_path.exists():
            h = _sha256_file(img_path)
            if h in seen_hashes and seen_hashes[h] != rid:
                errors.append(f"{rid}: duplicate image sha256 with '{seen_hashes[h]}'")
            else:
                seen_hashes[h] = rid
            # try read dims
            try:
                import cv2
                im = cv2.imread(str(img_path))
                if im is not None:
                    H, W = im.shape[:2]
            except Exception:
                pass
        # regions
        regions = rec.get("regions")
        if regions is None:
            warnings.append(f"{rid}: regions missing (annotation_level={rec.get('annotation_level')})")
            continue
        if not isinstance(regions, list):
            errors.append(f"{rid}: regions must be list")
            continue
        region_count += len(regions)
        # duplicate text regions
        texts = [str(r.get("text", "")) for r in regions if isinstance(r, dict)]
        dup_texts = [t for t, c in Counter(texts).items() if c > 1 and t.strip()]
        if dup_texts:
            warnings.append(f"{rid}: duplicate region texts {dup_texts[:3]}")
        for idx, r in enumerate(regions):
            if not isinstance(r, dict):
                errors.append(f"{rid} region {idx}: not an object")
                continue
            if r.get("ignore"):
                ignored_regions += 1
                continue
            poly = r.get("polygon")
            rtext = r.get("text")
            if poly is None or rtext is None:
                errors.append(f"{rid} region {idx}: missing polygon/text")
                continue
            if not isinstance(poly, list) or len(poly) < 3:
                errors.append(f"{rid} region {idx}: polygon needs >=3 points")
                continue
            pts_ok = True
            for p_idx, pt in enumerate(poly):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    errors.append(f"{rid} region {idx} pt {p_idx}: expected [x,y]")
                    pts_ok = False
                    break
                try:
                    x, y = float(pt[0]), float(pt[1])
                except Exception:
                    errors.append(f"{rid} region {idx} pt {p_idx}: non-numeric")
                    pts_ok = False
                    break
                if not math.isfinite(x) or not math.isfinite(y):
                    errors.append(f"{rid} region {idx} pt {p_idx}: non-finite")
                    pts_ok = False
                    break
                if W is not None and H is not None:
                    # allow small margin (-5) for near-bounds
                    if x < -5 or y < -5 or x > W + 5 or y > H + 5:
                        warnings.append(f"{rid} region {idx}: point out of image bounds [{x},{y}] vs {W}x{H}")
            if not pts_ok:
                continue
            area = _poly_area(poly)
            if area < 1e-6:
                errors.append(f"{rid} region {idx}: zero area")
                continue
            if W and H:
                img_area = float(W * H)
                if area < img_area * 0.0005:
                    warnings.append(f"{rid} region {idx}: extreme tiny box area={area:.1f} img={W}x{H}")
                if area > img_area * 0.95:
                    warnings.append(f"{rid} region {idx}: extreme large box area={area:.1f} img={W}x{H}")
            # self-intersection check via shapely if available
            try:
                from shapely.geometry import Polygon as _Poly
                p = _Poly(poly)
                if not p.is_valid:
                    warnings.append(f"{rid} region {idx}: self-intersecting polygon")
            except ImportError:
                pass
            except Exception:
                warnings.append(f"{rid} region {idx}: self-intersecting polygon")
        # overlapping duplicate polygon check (pairwise bbox IoU >0.95 same text)
        if isinstance(regions, list) and len(regions) > 1:
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    ri, rj = regions[i], regions[j]
                    if not isinstance(ri, dict) or not isinstance(rj, dict):
                        continue
                    if ri.get("ignore") or rj.get("ignore"):
                        continue
                    pi, pj = ri.get("polygon"), rj.get("polygon")
                    if not pi or not pj:
                        continue
                    try:
                        from benchmarks.metrics import polygon_iou
                        iou = polygon_iou(pi, pj)
                    except Exception:
                        iou = 0.0
                    if iou > 0.95 and str(ri.get("text", "")).strip() == str(rj.get("text", "")).strip():
                        warnings.append(f"{rid}: overlapping duplicate polygons {i}<->{j} iou={iou:.2f}")
                        break

    # split validation
    for rid in records:
        sp = rid.get("split")
        if sp not in VALID_SPLITS:
            errors.append(f"{rid.get('id')}: invalid split '{sp}'")

    # summary
    print("=== AUDIT ===")
    print(f"records: {len(records)}  regions: {region_count}  ignored_regions: {ignored_regions}")
    print(f"unreviewed: {unreviewed}  rejected: {rejected}  reviewed: {len(records)-unreviewed-rejected}")
    print(f"splits: {dict(split_counter)}")
    print(f"categories: {dict(cat_counter)}")
    print(f"tags: {dict(tag_counter)}")
    print(f"samples with no diacritics: {len(no_diacritic_ids)}")
    if no_diacritic_ids:
        print(f"  e.g. {no_diacritic_ids[:5]}")
    print("")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ERROR: {e}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:50]:
            print(f"  WARNING: {w}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings)-50} more warnings")
    if not errors and not warnings:
        print("No errors or warnings.")
    if errors:
        print(f"\nFAILED: {len(errors)} error(s)", file=sys.stderr)
        return 1
    if warnings:
        print(f"\nPASSED with {len(warnings)} warning(s)")
    else:
        print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
