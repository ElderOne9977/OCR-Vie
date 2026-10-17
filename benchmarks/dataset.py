"""Dataset helpers: benchmark_data JSONL + splits + extended Phase 3.1 schema.

No binary data committed. Provides the canonical schema constants and the
single ``filter_records`` used by every benchmark CLI so REAL vs SYNTHETIC and
reviewed-only selection are consistent across tooling.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmark_data"
ANNOTATIONS = BENCH_DIR / "annotations.jsonl"

EXPECTED_LAYOUT = {
    "raw": "data/raw — original images, never modified",
    "processed": "data/processed — resized/normalized copies (derived)",
    "train": "data/train — images + *.txt/.json labels",
    "val": "data/val — images + *.txt/.json labels",
    "test": "data/test — images + *.txt/.json labels",
    "benchmark_data/images": "benchmark_data/images — synthetic safe samples",
    "benchmark_data/crops": "benchmark_data/crops — GT crops (derived)",
    "benchmark_data/annotations.jsonl": "JSONL per spec: id,image,split,category,language,text,tags,regions,fields",
    "benchmark_data/splits": "train.txt/val.txt/test.txt — ids per split",
    "benchmark_data/dataset.json": "dataset metadata: name, version, created_at, counts",
    "benchmark_incoming": "unreviewed incoming images (gitignored, private until approved)",
}

ANNOTATION_FORMAT = """
benchmark_data/annotations.jsonl — one JSON per line:
  {
    "id": "sample_001",
    "image": "images/sample_001.png",
    "split": "test",
    "category": "scene",
    "language": "vie",
    "text": "Xin chao Viet Nam",
    "tags": ["synthetic","clean"],
    "regions": [{"polygon": [[x,y],...], "text": "Xin chao"}],
    "fields": {"amount": "1.250.000"}  # optional
  }
splits/*.txt list ids, one per line. No duplicate image SHA256 across splits.
"""

VALID_SPLITS = {"train", "val", "test"}
VALID_CATEGORIES = {"scene", "document", "synthetic", "invoice", "id_card", "sign"}

# ---- Phase 3.1 schema vocabulary (single source of truth) ----
SOURCE_TYPES = {"real", "synthetic"}
REVIEW_STATUSES = {"unreviewed", "reviewed", "rejected"}
ANNOTATION_LEVELS = {"image", "region"}
PRIVACY_VALUES = {"safe", "restricted", "unknown"}
ILLEGIBLE_STATUSES = {"illegible", "partially_legible", "ignore"}

# Recommended initial REAL test-set category plan (targets, not requirements).
CATEGORY_PLAN = {
    "clean_print", "screenshot", "camera", "signage", "product_label",
    "small_text", "low_contrast", "blur", "jpeg_compression",
    "perspective", "rotation", "complex_background", "mixed_vi_en",
    "diacritics_heavy", "numeric_heavy", "identifier_heavy",
}

# Vietnamese base letters whose marks must be measured (UPPERCASE too).
VIET_SPECIAL_CHARS = ["ă", "â", "ê", "ô", "ơ", "ư", "đ"]
VIET_SPECIAL_CHARS_UPPER = ["Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ"]
# Canonical combining tone marks, labelled by tone name.
VIET_TONES = {
    "\u0301": "sắc", "\u0300": "huyền", "\u0309": "hỏi",
    "\u0303": "ngã", "\u0323": "nặng",
}

# Critical-text categories to track (use synthetic/fake identifiers for privacy).
CRITICAL_FIELD_KINDS = {
    "money": ["amount"],
    "date": ["date"],
    "phone": ["phone"],
    "tax_id": ["tax_id"],
    "document_id": ["document_id"],
    "invoice_code": ["invoice_code"],
    "contract_code": ["contract_code"],
    "serial": ["serial"],
    "sku": ["sku"],
}

FIELD_KEYS = sorted({k for ks in CRITICAL_FIELD_KINDS.values() for k in ks})

# ---- Coverage thresholds (Phase 3.2) ----
# Minimum number of reviewed REAL test samples for an official, representative
# benchmark. Below this the numbers are smoke-only / not representative.
MIN_REAL_TEST_REVIEWED = 100
# Per-category/tag coverage flags. n<SEVERE is "severely underrepresented",
# n<LOW is "low coverage". Flags are advisory (never suppress data).
COVERAGE_SEVERE_THRESHOLD = 5
COVERAGE_LOW_THRESHOLD = 10


def load_annotations(path: Path = ANNOTATIONS) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('expected an annotation object')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}: line {line_no}: {exc}') from exc
            rows.append(record)
    return rows


def validate_ground_truth(record: dict) -> None:
    """Shared finite geometry/field contract for save, validation and freeze."""
    for key, choices in (('source_type', SOURCE_TYPES), ('review_status', REVIEW_STATUSES),
                         ('annotation_level', ANNOTATION_LEVELS), ('privacy', PRIVACY_VALUES)):
        if key in record and (not isinstance(record[key], str) or record[key] not in choices):
            raise ValueError(f'Invalid {key}')
    fields = record.get('fields') or {}
    if not isinstance(fields, dict) or any(not isinstance(v, str) for v in fields.values()):
        raise ValueError('Invalid fields schema')
    regions = record.get('regions', [])
    if not isinstance(regions, list):
        raise ValueError('Regions must be a list')
    mapped = set()
    for region in regions:
        polygon = region.get('polygon') if isinstance(region, dict) else None
        if not isinstance(polygon, list) or len(polygon) < 3 or not isinstance(region.get('text'), str):
            raise ValueError('Invalid region schema')
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 or
               any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) for p in polygon):
            raise ValueError('Polygon coordinates must be finite numbers')
        area2 = sum(a[0]*b[1] - b[0]*a[1] for a,b in zip(polygon, polygon[1:]+polygon[:1]))
        if abs(area2) < 1e-6 or len({tuple(p) for p in polygon}) != len(polygon):
            raise ValueError('Degenerate polygon')
        key = region.get('field_key')
        if key and not region.get('ignore'):
            if not isinstance(key, str) or key not in fields or key in mapped:
                raise ValueError('Invalid or duplicate field mapping')
            mapped.add(key)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- Deterministic normalization policy (benchmark GT, never OCR pred) ----

def normalize_ws(text: str) -> str:
    """Documented deterministic whitespace normalization for benchmark scoring.

    Raw region GT keeps visible text as-is. Scoring normalizes:
      - tab -> space, CRLF -> LF -> space
      - collapse runs of whitespace to a single space
      - strip leading/trailing whitespace
    No punctuation, case, or numeric normalization. Raw text remains primary.
    """
    if not text:
        return ""
    s = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def record_source(rec: dict) -> str:
    """REAL vs SYNTHETIC for a record. tags>synthetic>source_type fallback."""
    tags = rec.get("tags") or []
    if isinstance(tags, list) and any(t == "synthetic" for t in tags):
        return "synthetic"
    if rec.get("source_type") in SOURCE_TYPES:
        return rec["source_type"]
    return "synthetic"


def record_review_status(rec: dict) -> str:
    """Review status; legacy records without the field are 'unreviewed'."""
    rs = rec.get("review_status")
    return rs if rs in REVIEW_STATUSES else "unreviewed"


def record_annotation_level(rec: dict) -> str:
    """Annotation level; inferred from regions when field missing."""
    al = rec.get("annotation_level")
    if al in ANNOTATION_LEVELS:
        return al
    return "region" if rec.get("regions") else "image"


def filter_records(
    records: list[dict],
    split: str | None = None,
    source: str | None = None,
    reviewed_only: bool = False,
    include_unreviewed: bool = False,
    include_rejected: bool = False,
) -> list[dict]:
    """Canonical selection used by all benchmark CLIs.

    - split=None -> all splits. split='test' -> only test records.
    - source in {'real','synthetic'} -> only that source. source=None -> all.
    - reviewed_only=True -> only review_status=='reviewed'.
      (replaced the old include-unreviewed default; include_unreviewed=True
      is accepted for CLI backward-compat and is a no-op when reviewed_only.)
    - include_rejected=False -> drop rejected records.
    Official quality reporting should always use:
      split='test', source='real', reviewed_only=True.
    """
    out: list[dict] = []
    for rec in records:
        if split is not None and rec.get("split") != split:
            continue
        if source is not None and record_source(rec) != source:
            continue
        if reviewed_only and record_review_status(rec) != "reviewed":
            continue
        if not include_rejected and record_review_status(rec) == "rejected":
            continue
        out.append(rec)
    return out


def count_records(records: list[dict]) -> dict:
    """Fast counts: by source, review status, split, category, tag."""
    by_source: dict[str, int] = {}
    by_review: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    for rec in records:
        s = record_source(rec)
        rs = record_review_status(rec)
        by_source[s] = by_source.get(s, 0) + 1
        by_review[rs] = by_review.get(rs, 0) + 1
        sp = str(rec.get("split", "?"))
        by_split[sp] = by_split.get(sp, 0) + 1
        cat = str(rec.get("category", "?"))
        by_category[cat] = by_category.get(cat, 0) + 1
        for t in rec.get("tags") or []:
            by_tag[str(t)] = by_tag.get(str(t), 0) + 1
    return {
        "by_source": by_source,
        "by_review": by_review,
        "by_split": by_split,
        "by_category": by_category,
        "by_tag": by_tag,
    }


# ---- Coverage + freeze helpers (Phase 3.2) ----

def record_tags(rec: dict) -> set[str]:
    """Tags plus category, so category-plan coverage counts both."""
    out: set[str] = set()
    for t in rec.get("tags") or []:
        out.add(str(t))
    cat = rec.get("category")
    if cat:
        out.add(str(cat))
    return out


def real_reviewed_test_count(records: list[dict]) -> int:
    """Count REAL + reviewed + split==test. The official coverage gate."""
    return sum(
        1
        for r in records
        if record_source(r) == "real"
        and record_review_status(r) == "reviewed"
        and r.get("split") == "test"
    )


def coverage_report(records: list[dict]) -> dict:
    """Per-category-plan counts with severe/low flags.

    Each plan item is matched against a record's tags OR category. Flags are
    advisory only; data is never suppressed. ``n<SEVERE`` => severely
    underrepresented, ``n<LOW`` => low coverage, else ok.
    """
    per: dict[str, int] = {}
    for rec in records:
        for item in record_tags(rec):
            if item in CATEGORY_PLAN:
                per[item] = per.get(item, 0) + 1
    items: dict[str, dict] = {}
    for item in sorted(CATEGORY_PLAN):
        n = per.get(item, 0)
        if n < COVERAGE_SEVERE_THRESHOLD:
            flag = "SEVERE"
        elif n < COVERAGE_LOW_THRESHOLD:
            flag = "LOW"
        else:
            flag = "ok"
        items[item] = {"n": n, "flag": flag}
    return {
        "severe_threshold": COVERAGE_SEVERE_THRESHOLD,
        "low_threshold": COVERAGE_LOW_THRESHOLD,
        "items": items,
    }


def build_manifest(records: list[dict], data_dir: Path | None = None) -> dict:
    """Fingerprint verified image bytes and canonical scoring GT without publishing GT."""
    base = Path(data_dir) if data_dir is not None else BENCH_DIR
    entries = []
    for rec in sorted(records, key=lambda r: str(r.get("id", ""))):
        validate_ground_truth(rec)
        rel = str(rec.get("image", ""))
        sha = rec.get("sha256")
        verified = False
        if rel:
            p = base / rel
            if not p.exists():
                p = ROOT / rel
            if not p.is_file():
                raise ValueError(f"Image missing for {rec.get('id')}: {rel}")
            actual = sha256_file(p)
            if sha and sha != actual:
                raise ValueError(f"Image SHA256 mismatch for {rec.get('id')}")
            sha = actual
            verified = True
        gt = {key: rec.get(key) for key in (
            'image', 'language', 'category', 'tags', 'text', 'regions', 'fields',
            'annotation_level', 'annotation_complete', 'reading_order',
        )}
        gt_json = json.dumps(gt, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        entries.append(
            {
                "id": rec.get("id"),
                "split": rec.get("split"),
                "sha256": sha,
                "source": record_source(rec),
                "review_status": record_review_status(rec),
                "annotation_sha256": hashlib.sha256(gt_json.encode('utf-8')).hexdigest(),
                "image_verified": verified,
            }
        )
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": 2,
        "count": len(entries),
        "manifest_hash": manifest_hash,
        "entries": entries,
    }


def model_fingerprints(model_names: list[str], languages: list[str]) -> dict:
    """Offline artifact inventory, outside OCR timings; never initializes models."""
    import os
    import pytesseract
    from app.services.paddle_runtime import model_dir
    from app.ocr_service import normalize_tesseract_lang
    models = {}
    for name in sorted(set(model_names)):
        directory = model_dir(name)
        models[name] = {f.name: sha256_file(f) for f in sorted(directory.glob('inference.*')) if f.is_file()}
    roots = [Path(os.environ['TESSDATA_PREFIX'])] if os.getenv('TESSDATA_PREFIX') else [
        Path(pytesseract.pytesseract.tesseract_cmd).parent / 'tessdata',
        Path('/usr/share/tesseract-ocr/5/tessdata'), Path('/usr/share/tessdata')]
    tess = {}
    for lang in sorted({part for lang in languages for part in normalize_tesseract_lang(lang).split('+')}):
        path = next((root/f'{lang}.traineddata' for root in roots if (root/f'{lang}.traineddata').is_file()), None)
        tess[lang] = sha256_file(path) if path else None
    return {'paddle_local_inventory': models, 'tesseract_traineddata': tess,
            'scope': 'local artifact hashes, not proof every listed model was used'}
