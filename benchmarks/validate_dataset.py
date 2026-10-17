"""Validate benchmark_data/annotations.jsonl against Phase 3 spec.

Checks: JSONL syntax, unique id, image exists, valid split, text exists,
polygon geometry valid, field structure valid, no duplicate image SHA256 across splits.
Exits non-zero on failure. Stdlib + optional OpenCV (for geometry sanity only).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.dataset import validate_ground_truth
BENCH_DIR = ROOT / "benchmark_data"
ANNOTATIONS = BENCH_DIR / "annotations.jsonl"
VALID_SPLITS = {"train", "val", "test"}
VALID_LANGS = {"vie", "eng", "vie+eng", "en", "vi"}
OPTIONAL_FIELD_KEYS = {"amount", "tax_id", "document_id", "invoice_id", "mova"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _err(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"ERROR: {msg}", file=sys.stderr)


def validate(path: Path = ANNOTATIONS) -> int:
    errors: list[str] = []
    path = Path(path)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1
    # base dir = annotations.jsonl parent (so any dataset root works)
    base_dir = path.parent
    seen_ids: set[str] = set()
    record_splits: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}  # hash -> id
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or all(not l.strip() for l in lines):
        _err("annotations.jsonl empty", errors)
        return 1
    for idx, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            _err(f"line {idx}: invalid JSON: {e}", errors)
            continue
        if not isinstance(obj, dict):
            _err(f'line {idx}: expected object', errors)
            continue
        invalid = False
        for key in ('id', 'image', 'split', 'category', 'language', 'text'):
            if not isinstance(obj.get(key), str) or not obj[key].strip():
                _err(f'line {idx}: {key} must be a nonempty string', errors)
                invalid = True
        if invalid:
            continue
        try:
            validate_ground_truth(obj)
        except ValueError as exc:
            _err(f'line {idx}: {exc}', errors)
        # required fields
        for k in ("id", "image", "split", "category", "language", "text"):
            if k not in obj:
                _err(f"line {idx} id={obj.get('id','?')}: missing '{k}'", errors)
        rid = str(obj.get("id", f"line{idx}"))
        if rid in seen_ids:
            _err(f"line {idx}: duplicate id '{rid}'", errors)
        else:
            seen_ids.add(rid)
        split = obj.get("split")
        record_splits[rid] = split
        if split not in VALID_SPLITS:
            _err(f"line {idx} id={rid}: invalid split '{split}' expected train/val/test", errors)
        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            _err(f"line {idx} id={rid}: text missing/empty", errors)
        # image exists (relative to benchmark_data)
        img_rel = obj.get("image")
        if isinstance(img_rel, str):
            img_path = (base_dir / img_rel).resolve()
            # also allow repo-root-relative paths
            if not img_path.exists():
                alt = ROOT / img_rel
                if alt.exists():
                    img_path = alt
                else:
                    _err(f"line {idx} id={rid}: image not found '{img_rel}'", errors)
                    img_path = None
            if img_path and img_path.exists():
                try:
                    h = sha256_file(img_path)
                except Exception as e:
                    _err(f"line {idx} id={rid}: cannot hash image: {e}", errors)
                else:
                    if h in seen_hashes and seen_hashes[h] != rid:
                        _err(f"line {idx} id={rid}: duplicate image SHA256 {h[:8]}... also used by '{seen_hashes[h]}' (no duplicate across splits)", errors)
                    else:
                        seen_hashes.setdefault(h, rid)
        # regions
        regions = obj.get("regions", [])
        if not isinstance(regions, list):
            _err(f"line {idx} id={rid}: regions must be list", errors)
        else:
            for r_idx, r in enumerate(regions):
                if not isinstance(r, dict):
                    _err(f"line {idx} id={rid} region {r_idx}: not an object", errors)
                    continue
                poly = r.get("polygon")
                rtext = r.get("text")
                if poly is None or rtext is None:
                    _err(f"line {idx} id={rid} region {r_idx}: missing polygon/text", errors)
                    continue
                if not isinstance(poly, list) or len(poly) < 3:
                    _err(f"line {idx} id={rid} region {r_idx}: polygon needs >=3 points", errors)
                    continue
                for p_idx, pt in enumerate(poly):
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        _err(f"line {idx} id={rid} region {r_idx} pt {p_idx}: expected [x,y]", errors)
                        break
                    try:
                        float(pt[0]); float(pt[1])
                    except Exception:
                        _err(f"line {idx} id={rid} region {r_idx} pt {p_idx}: non-numeric", errors)
                        break
                if not isinstance(rtext, str):
                    _err(f"line {idx} id={rid} region {r_idx}: text must be string", errors)
        # fields optional
        fields = obj.get("fields")
        if fields is not None:
            if not isinstance(fields, dict):
                _err(f"line {idx} id={rid}: fields must be object", errors)
            else:
                for k, v in fields.items():
                    if k not in OPTIONAL_FIELD_KEYS and k not in {"amount","tax_id","document_id","field_amount","field_tax_id"}:
                        # allow any but warn if unexpected type
                        pass
                    if not isinstance(v, str):
                        _err(f"line {idx} id={rid} fields.{k}: must be string", errors)
        # tags optional but if present should be list
        tags = obj.get("tags")
        if tags is not None and (not isinstance(tags, list) or any(not isinstance(t, str) for t in tags)):
            _err(f"line {idx} id={rid}: tags must be list of strings", errors)
    # splits files consistency (if present)
    splits_dir = base_dir / "splits"
    if splits_dir.exists():
        all_split_ids: set[str] = set()
        for sp in VALID_SPLITS:
            fp = splits_dir / f"{sp}.txt"
            if not fp.exists():
                continue
            for line in fp.read_text(encoding="utf-8").splitlines():
                line=line.strip()
                if not line:
                    continue
                if line in all_split_ids:
                    _err(f"splits: duplicate id '{line}' across splits", errors)
                all_split_ids.add(line)
                if line not in seen_ids:
                    _err(f"splits/{sp}.txt: id '{line}' not in annotations.jsonl", errors)
                elif record_splits.get(line) != sp:
                    _err(f"splits/{sp}.txt: id '{line}' belongs to {record_splits.get(line)}", errors)
        for missing in seen_ids - all_split_ids:
            _err(f"splits: missing id '{missing}'", errors)
    if errors:
        print(f"\nValidation FAILED: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"Validation OK: {len(seen_ids)} records, {len(seen_hashes)} unique images")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BENCH_DIR))
    args = ap.parse_args()
    p = Path(args.data) / "annotations.jsonl" if Path(args.data).is_dir() else Path(args.data)
    raise SystemExit(validate(p))
