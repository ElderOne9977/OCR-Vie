# benchmark_data — Vietnamese OCR Benchmark Dataset (Phase 3.1)

## Privacy Gate

**WARNING:** This directory contains benchmark data that may include personal or sensitive information. Before committing to git:

1. Verify all records have `privacy: "safe"` in their annotations
2. Run: `python benchmarks/validate_dataset.py --data benchmark_data`
3. Ensure no `benchmark_incoming/` files are staged

The `.gitignore` protects private directories. Never commit without explicit `privacy=safe` approval.

## Data Organization

```
benchmark_data/
├── images/           # Original benchmark images (never modified)
├── annotations.jsonl # JSONL annotations per spec
├── splits/          # train.txt/val.txt/test.txt id lists
├── dataset.json     # Version metadata + counts
└── README.md        # This file
```

## Building the Dataset

### Step 1: Generate Synthetic Samples (Smoke Testing)

```bash
python scripts/generate_ocr_samples.py \
  --output benchmark_data \
  --count 12 \
  --seed 42 \
  --variants clean blur jpeg rotation perspective low_contrast small_text
```

- Uses only system fonts (no PII)
- Deterministic via `--seed`
- Tagged `synthetic` for easy filtering

### Step 2: Import Real Images (Optional)

```bash
python benchmarks/import_images.py \
  --source my_real_images/ \
  --data benchmark_data \
  --category camera \
  --split test \
  --source-type real \
  --privacy safe \
  --tags signage diacritics_heavy \
  --notes "Street signs from Ho Chi Minh City"
```

- Creates unreviewed annotation stubs
- Deduplicates by SHA256 hash
- Requires `--privacy safe` for real data

### Step 3: Annotate Regions

```bash
python benchmarks/annotate.py --data benchmark_data --port 8765
```

Then open http://localhost:8765/ to:
- Draw/edit polygon regions
- Enter GT text (verbatim, exact whitespace/punctuation)
- Mark reviewed/rejected status
- Set annotation level (image/region)
- Export back to `annotations.jsonl`

### Step 4: Validate

```bash
python benchmarks/validate_dataset.py --data benchmark_data
```

Checks: unique IDs, valid splits, text present, polygons valid, no duplicate SHA256 across splits.
Exit code 0 = OK, non-zero = error (fix before using in benchmark).

### Step 5: Audit & Summary

```bash
python benchmarks/audit_annotations.py --data benchmark_data
python benchmarks/dataset_summary.py --data benchmark_data --json benchmark_data/dataset_summary.json
```

Provides QA on geometry, duplicates, coverage of Vietnamese diacritics/tone marks, critical fields (`amount`, `tax_id`, etc.).

## Coverage Targets (Phase 3)

Initial target for official quality benchmark:

- ≥100 **real**, **reviewed**, **test** samples
- Balanced across category plan: `clean_print`, `camera`, `signage`, `diacritics_heavy`, `mixed_vi_en`, `low_contrast`, etc.

Per-category-plan flags (advisory, data is **never** suppressed):

- `n < 5` → **SEVERE** (severely underrepresented)
- `n < 10` → **LOW** (low coverage)
- otherwise → `ok`

Tracked items: `clean_print, screenshot, camera, signage, product_label, small_text, low_contrast, blur, jpeg_compression, perspective, rotation, complex_background, mixed_vi_en, diacritics_heavy, numeric_heavy, identifier_heavy`.

Check progress:
```bash
python benchmarks/dataset_summary.py --data benchmark_data | grep "Real reviewed test:"
```

## Freeze the Real Test Set (Phase 3.2)

Once `Real reviewed test: x / 100` reaches ≥100, freeze ids + SHA256 + splits +
annotations into an immutable manifest. `freeze_testset.py` refuses below 100
(unless `--allow-preliminary`, which labels the manifest
`PRELIMINARY NOT REPRESENTATIVE`). The manifest stores **only** ids/splits/sha256
— no image bytes — so it is safe to commit.

```bash
python benchmarks/freeze_testset.py --data benchmark_data --version 0.1
# -> benchmark_data/frozen/testset_v0.1.json  (label: REAL TEST SET v0.1)
```

Do **not** run the official benchmark on an unfrozen or <100 set; any run on a
smaller set must be labelled `PRELIMINARY NOT REPRESENTATIVE`.

## Benchmark CLI Filtering Flags

All Phase 3 benchmark runners support canonical filtering flags:

```bash
# Recognition-only benchmark
python benchmarks/run_recognition_benchmark.py \
  --data benchmark_data \
  --recognizer paddle_v5_latin \
  --split test \
  --source real \
  --reviewed-only \
  --preprocessing original

# Scene end-to-end benchmark
python benchmarks/run_scene_benchmark.py \
  --data benchmark_data \
  --recognizer paddle_v5_latin \
  --split test \
  --source real \
  --reviewed-only \
  --include-rejected  # Optional: include rejected records
```

| Flag | Description | Default | Notes |
|---|---|---|---|
| `--source` | Filter by `real` or `synthetic` | None (all) | Official reports use `--source real` |
| `--split` | Filter by `train`/`val`/`test` | None (all) | Quality benchmark uses `--split test` |
| `--reviewed-only` | Only `review_status == 'reviewed'` | False | **Recommended for official runs** |
| `--include-unreviewed` | Keep unreviewed | False | No-op when `--reviewed-only` is set |
| `--include-rejected` | Keep rejected records | False | Usually excluded from quality metrics |

## Schema (Phase 3.1)

Annotation record structure:

```json
{
  "id": "unique_id",
  "image": "images/sample.png",
  "split": "test",
  "category": "scene",
  "language": "vie",
  "text": "Xin chào Việt Nam",
  "tags": ["real", "camera", "diacritics_heavy"],
  "regions": [
    {
      "polygon": [[x1,y1], [x2,y2]],
      "text": "Xin chào",
      "ignore": false,
      "annotation_level": "region"
    }
  ],
  "fields": {
    "amount": "1.250.000",
    "tax_id": "012345678901"
  },
  "review_status": "unreviewed",
  "source_type": "real",
  "privacy": "safe",
  "sha256": "abc123...",
  "imported_at": "2026-10-03T12:00:00Z"
}
```

## Metrics Used in Benchmarking

- **CER** (Character Error Rate): `lev(pred, ref) / len(ref)`
- **WER** (Word Error Rate): Word-level Levenshtein ratio
- **Exact Match**: `strip(pred) == strip(ref)`
- **accent_stripped_cer**: CER after removing Vietnamese diacritics (NFD Mn filter + đ/Đ map)
- **diacritic_loss_rate**: `cer(pred, ref) - accent_stripped_cer(pred, ref)` clamped ≥ 0
- **Critical field exact match**: Amount/Tax ID/document ID character-perfect match
- **Detection IoU**: Polygon IoU via shapely if available, else bbox approximation (threshold 0.5)

Formula: Diacritic loss measures how much of CER is attributable to accent errors. For example:
- Reference: `"Xin chào Việt Nam"`
- Prediction: `"Xin chao Viet Nam"`
- Normal CER > 0, accent-stripped CER = 0 → diacritic_loss_rate = normal CER

## Status

- **Current version**: 0.3.1 (Phase 3.1 schema with filtering + privacy gate)
- **Current sample count**: 12 synthetic (SMOKE ONLY NOT REPRESENTATIVE)
- **Next milestone**: ≥100 real reviewed test samples for official quality benchmark
- **Blocked**: Real Vietnamese labeled dataset collection

## Documentation Links

- Main `README.md` §14 Benchmarking
- `PROJECT_LOG.md` §18 Hybrid Phase 3
- `benchmarks/metrics.py` – CER/WER/diacritic/IoU implementations
- `benchmarks/dataset.py` – Canonical filtering logic + schema constants