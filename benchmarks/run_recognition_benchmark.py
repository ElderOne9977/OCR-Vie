"""Recognition-only benchmark: GT-crop -> recognizer isolation.

Usage:
  python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer paddle_v5_latin
  python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer tesseract
Supports preprocessing ablation: original | gray | otsu (BGR variants, no prod change).
Writes timestamped benchmark_results/<run-id>/ with CSV/JSON/MD.
Stdlib + cv2/numpy/Pillow only. Raw_text primary, no correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from benchmarks.dataset import count_records, filter_records, load_annotations, build_manifest, real_reviewed_test_count
from importlib.metadata import version, PackageNotFoundError
from benchmarks.metrics import (
    accent_stripped_cer,
    cer,
    diacritic_loss_rate,
    exact_match,
    field_exact_match,
    wer,
)
from app.services.polygon_utils import crop_polygon

ROOT = Path(__file__).resolve().parents[1]


def _git_hash() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _load_annotations(data_dir: Path) -> list[dict]:
    p = data_dir / "annotations.jsonl" if (data_dir / "annotations.jsonl").exists() else data_dir
    if p.is_dir():
        p = p / "annotations.jsonl"
    return load_annotations(p)


def _apply_preprocessing(crop, mode: str):
    if crop is None:
        return crop
    if mode == "original":
        return crop
    if mode == "gray":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "otsu":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return crop


def _get_recognizer(name: str):
    key = (name or "auto").lower()
    if key in ("auto", "tesseract", "tesseract_vie_eng"):
        from app.recognizers.tesseract_recognizer import TesseractRecognizer
        return TesseractRecognizer(), "tesseract", "vie+eng"
    # paddle_v5_latin / paddle / auto -> paddle
    from app.recognizers.paddle_recognizer import PaddleRecognizer
    return PaddleRecognizer(), "paddle_v5_latin", "latin_PP-OCRv5_mobile_rec"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark_data")
    ap.add_argument("--recognizer", default="tesseract", choices=["paddle_v5_latin","paddle","tesseract","auto"])
    ap.add_argument("--preprocessing", default="original", choices=["original","gray","otsu"])
    ap.add_argument("--psm", type=int, default=7, help="Tesseract page segmentation for line crops")
    ap.add_argument("--output", default="benchmark_results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default=None, choices=["real","synthetic"])
    ap.add_argument("--split", default=None, choices=["train","val","test"])
    ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--include-rejected", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve() if not data_dir.exists() else data_dir.resolve()
        # fallback to ROOT/data_dir relative
        alt = ROOT / args.data
        if alt.exists():
            data_dir = alt.resolve()
    rows_ann = _load_annotations(data_dir)
    if not rows_ann:
        print(f"No annotations in {data_dir}", file=sys.stderr)
        return 1
    rows_ann = filter_records(rows_ann, split=args.split, source=args.source, reviewed_only=args.reviewed_only, include_unreviewed=args.include_unreviewed, include_rejected=args.include_rejected)
    if args.limit:
        rows_ann = rows_ann[:args.limit]
    if not rows_ann:
        print("No records after filtering", file=sys.stderr)
        return 1
    # bench dir
    bench_root = data_dir if data_dir.is_dir() else data_dir.parent
    manifest = build_manifest(rows_ann, bench_root)
    # recognizer
    rec, rec_label, rec_model = _get_recognizer(args.recognizer)
    languages = sorted({a.get("language", "vie+eng") for a in rows_ann})
    if rec_label == "tesseract":
        rec_model = " / ".join({"vi": "vie", "en": "eng"}.get(lang, lang) for lang in languages)
    # version meta
    import sys as _sys
    py_ver = _sys.version.split()[0]
    try:
        paddle_ver = version('paddlepaddle')
    except PackageNotFoundError:
        paddle_ver = "not_installed"
    try:
        po_ver = version('paddleocr')
    except PackageNotFoundError:
        po_ver = "not_installed"
    try:
        import pytesseract as _pt
        tesseract_ver = str(_pt.get_tesseract_version())
    except Exception:
        tesseract_ver = "not_available"
    from app.core.config import get_settings
    from benchmarks.dataset import model_fingerprints
    from app.services.paddle_models import resolve_rec_model, VIETNAMESE_CHARACTERS
    settings = get_settings()
    device = settings.device
    # detect/detector meta
    det_name = None  # GT crop recognition does not run a detector.
    det_model = None
    # timing
    t_load0 = time.perf_counter()
    load_ok = False
    try:
        if hasattr(rec, "ensure_ready"):
            load_ok = all(rec.ensure_ready(lang=lang) for lang in languages)
        else:
            load_ok = rec.status() == "ready"
    except Exception:
        load_ok = False
    model_load_ms = round((time.perf_counter() - t_load0) * 1000, 2)
    if not load_ok:
        print("Recognizer unavailable", file=sys.stderr)
        return 1
    # prepare output dir
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{ts}_recognition_{rec_label}_{args.preprocessing}"
    out_dir = ROOT / args.output / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    # iterate gt crops
    import unicodedata
    csv_rows = []
    errors = []
    cer_list, wer_list, accent_list, loss_list = [], [], [], []
    exact_cnt = 0
    field_total = 0
    field_ok = 0
    field_unmapped = 0
    successes = 0
    conf_bins = {"0.0-0.5": [0,0,0.0], "0.5-0.7": [0,0,0.0], "0.7-0.8": [0,0,0.0], "0.8-0.9": [0,0,0.0], "0.9-1.0": [0,0,0.0]}
    # Include tone-bearing variants (e.g. ệ/ấ), not just bare ă/â/ê.
    # Count deficits only; these are NOT edit-aligned character error rates.
    viet_chars = sorted(ch for ch in VIETNAMESE_CHARACTERS if not ch.isascii())
    viet_total = {c:0 for c in viet_chars}
    viet_miss = {c:0 for c in viet_chars}
    rec_times = []
    for ann in rows_ann:
        img_rel = ann.get("image", "")
        img_path = (bench_root / img_rel) if not Path(img_rel).is_absolute() else Path(img_rel)
        if not img_path.exists():
            alt2 = ROOT / img_rel
            if alt2.exists():
                img_path = alt2
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot decode benchmark image: {img_path}")
        regions = ann.get("regions") or [{"polygon": [[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0,image.shape[0]]], "text": ann.get("text","")}]
        mapped = [r.get('field_key') for r in regions if not r.get('ignore') and r.get('field_key')]
        fields = ann.get('fields') or {}
        if len(mapped) != len(set(mapped)) or any(key not in fields for key in mapped):
            raise ValueError(f"Invalid field mapping: {ann.get('id')}")
        field_unmapped += len(set(fields) - set(mapped))
        for r_idx, reg in enumerate(regions):
            if reg.get("ignore", False):
                continue
            gt = str(reg.get("text",""))
            poly = reg.get("polygon")
            crop = crop_polygon(image, poly) if poly else image
            if crop is None or crop.size == 0:
                raise ValueError(f"Invalid GT crop: {ann.get('id')} region {r_idx}")
            crop_p = _apply_preprocessing(crop, args.preprocessing)
            t0 = time.perf_counter()
            try:
                language = ann.get("language", "vie+eng")
                if rec_label == "tesseract":
                    language = {"vi": "vie", "en": "eng"}.get(language, language)
                res = rec.recognize(crop_p, lang=language, psm=args.psm)
                pred = res.raw_text or ""
                conf = res.ocr_confidence
                successes += 1
            except Exception as e:
                pred = ""
                conf = None
                errors.append({"id": ann.get("id"), "region": r_idx, "error": str(e)})
            dt = round((time.perf_counter() - t0)*1000,2)
            rec_times.append(dt)
            c = cer(pred, gt)
            w = wer(pred, gt)
            ac = accent_stripped_cer(pred, gt)
            loss = diacritic_loss_rate(pred, gt)
            ex = exact_match(pred, gt)
            cer_list.append(c); wer_list.append(w); accent_list.append(ac); loss_list.append(loss)
            if ex:
                exact_cnt += 1
            # field
            field_key = reg.get('field_key')
            if field_key:
                field_total += 1
                field_ok += int(field_exact_match(pred, str(fields[field_key])))
            # confidence bin
            if conf is not None:
                try:
                    cf = float(conf)
                    # All adapters now expose a 0..1 score, not a calibrated probability.
                    if cf < 0.5:
                        k="0.0-0.5"
                    elif cf < 0.7:
                        k="0.5-0.7"
                    elif cf < 0.8:
                        k="0.7-0.8"
                    elif cf < 0.9:
                        k="0.8-0.9"
                    else:
                        k="0.9-1.0"
                    conf_bins[k][0]+=1
                    conf_bins[k][1]+= int(ex)
                    conf_bins[k][2]+= c
                except Exception:
                    pass
            # viet breakdown
            for ch in viet_chars:
                cnt = unicodedata.normalize('NFC', gt).count(ch)
                if cnt:
                    viet_total[ch]+=cnt
                    # miss if pred missing that char
                    miss = cnt - unicodedata.normalize('NFC', pred).count(ch)
                    if miss>0:
                        viet_miss[ch]+=miss
            csv_rows.append({
                "id": ann.get("id"),
                "image": img_rel,
                "split": ann.get("split"),
                "category": ann.get("category"),
                "region_idx": r_idx,
                "gt_text": gt,
                "pred_text": pred,
                "cer": round(c,4),
                "wer": round(w,4),
                "exact_match": int(ex),
                "accent_stripped_cer": round(ac,4),
                "diacritic_loss_rate": round(loss,4),
                "confidence": conf,
                "recognition_ms": dt,
                "preprocessing": args.preprocessing,
            })
            if c > 0.3:
                errors.append({"id": ann.get("id"), "region": r_idx, "gt": gt, "pred": pred, "cer": round(c,4)})
    n = len(csv_rows)
    avg_cer = round(sum(cer_list)/len(cer_list),4) if cer_list else 0.0
    avg_wer = round(sum(wer_list)/len(wer_list),4) if wer_list else 0.0
    avg_accent = round(sum(accent_list)/len(accent_list),4) if accent_list else 0.0
    avg_loss = round(sum(loss_list)/len(loss_list),4) if loss_list else 0.0
    exact_rate = round(exact_cnt / n, 4) if n else 0.0
    cold_ms = rec_times[0] if rec_times else 0
    warm_avg = round(sum(rec_times[1:]) / len(rec_times[1:]),2) if len(rec_times)>1 else 0.0
    # warm vs cold separate
    # write recognition_results.csv
    import csv as _csv
    rec_csv = out_dir / "recognition_results.csv"
    with rec_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id","image","split","category","region_idx","gt_text","pred_text","cer","wer","exact_match","accent_stripped_cer","diacritic_loss_rate","confidence","recognition_ms","preprocessing"])
        w.writeheader()
        w.writerows(csv_rows)
    # scene_results.csv empty placeholder for compatibility? create minimal
    scene_csv = out_dir / "scene_results.csv"
    with scene_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id","image","gt_regions","pred_regions","matched","missed","fp","precision","recall","f1","mean_iou","cer","wer","detection_ms","recognition_ms","total_ms"])
        w.writeheader()
    # errors.csv
    err_csv = out_dir / "errors.csv"
    with err_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id","region","gt","pred","cer","error","type"])
        w.writeheader()
        for e in errors[:200]:
            w.writerow({k: e.get(k,"") for k in ["id","region","gt","pred","cer","error","type"]})
    # confidence bins summary
    bins_out = {}
    for k, (cnt, ex_cnt, cer_sum) in conf_bins.items():
        bins_out[k] = {"count": cnt, "exact_rate": round(ex_cnt/cnt,4) if cnt else 0.0, "avg_cer": round(cer_sum/cnt,4) if cnt else 0.0}
    # viet breakdown rates
    viet_break = {ch: {"total": viet_total[ch], "missed": viet_miss[ch], "miss_rate": round(viet_miss[ch]/viet_total[ch],4) if viet_total[ch] else 0.0} for ch in viet_chars if viet_total[ch]>0}
    # summary
    summary = {
        "run_id": run_id,
        "mode": "recognition",
        "recognizer": rec_label,
        "detector": det_name,
        "preprocessing": args.preprocessing,
        "psm": args.psm,
        "languages": languages,
        "data": str(data_dir),
        "n_samples": n,
        "n_images": len(rows_ann),
        "metrics": {"cer": avg_cer, "wer": avg_wer, "exact_match_rate": exact_rate, "accent_stripped_cer": avg_accent, "diacritic_loss_rate": avg_loss, "field_exact_match": round(field_ok/field_total,4) if field_total else None},
        "latency_ms": {"model_load_ms": model_load_ms, "cold_recognition_ms": cold_ms, "warm_avg_recognition_ms": warm_avg, "avg_recognition_ms": round(sum(rec_times)/len(rec_times),2) if rec_times else 0},
        "confidence_bins": bins_out,
        "vietnamese_breakdown": viet_break,
        "model_meta": {"python": py_ver, "paddlepaddle": paddle_ver, "paddleocr": po_ver, "detector": det_model, "recognizer": rec_model, "tesseract": tesseract_ver, "device": device, "cli": f"--recognizer {rec_label} --preprocessing {args.preprocessing}", "dataset": str(data_dir), "sample_count": n, "git_hash": _git_hash()},
        "selection": {"source": args.source, "split": args.split, "reviewed_only": args.reviewed_only, "include_rejected": args.include_rejected}, "dataset_counts": count_records(rows_ann), "warnings": ["REAL REVIEWED TEST COUNT <100: SMOKE ONLY NOT REPRESENTATIVE" ] if real_reviewed_test_count(rows_ann) < 100 else [],
        "cold_vs_warm": {"cold_ms": cold_ms, "warm_avg_ms": warm_avg},
        'field_evaluation': {'evaluated': field_total, 'unmapped': field_unmapped},
        'execution_mode': 'gt_crop_recognition_only',
        'dataset_manifest': manifest,
        'model_artifacts': model_fingerprints([m for lang in languages if rec_label != 'tesseract' and (m := resolve_rec_model(lang))], languages),
        'failed_regions': sum('error' in e for e in errors),
        'vietnamese_breakdown_scope': 'NFC character count deficits; not aligned error rate',
        'runtime_config': {'cpu_threads': settings.cpu_threads, 'detector_side': settings.detector_limit_side_len,
                           'batch_size': settings.tesseract_batch_size, 'psm': args.psm},
        'successful_regions': successes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # summary.md
    md = []
    md.append(f"# Recognition Benchmark {run_id}\n")
    md.append(f"- recognizer: {rec_label} ({rec_model}) preprocessing={args.preprocessing}\n")
    md.append(f"- data: {data_dir} samples={n} images={len(rows_ann)}\n")
    md.append(f"- model_load_ms={model_load_ms} cold={cold_ms} warm_avg={warm_avg}\n")
    md.append(f"- CER={avg_cer} WER={avg_wer} exact={exact_rate} accent_cer={avg_accent} diacritic_loss={avg_loss}\n")
    if summary['warnings']:
        md.append('\n> WARNING: REAL REVIEWED TEST COUNT <100 — SMOKE ONLY NOT REPRESENTATIVE\n')
    md.append(f"\n## Model Meta\n```json\n{json.dumps(summary['model_meta'], ensure_ascii=False, indent=2)}\n```\n")
    md.append(f"\n## Confidence bins\n```json\n{json.dumps(bins_out, ensure_ascii=False, indent=2)}\n```\n")
    if viet_break:
        md.append(f"\n## Vietnamese breakdown\n```json\n{json.dumps(viet_break, ensure_ascii=False, indent=2)}\n```\n")
    md.append(f"\nFormula: diacritic_loss_rate = cer(pred,ref) - accent_stripped_cer(pred,ref) clamped >=0\n")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Done: {out_dir} CER={avg_cer} WER={avg_wer} exact={exact_rate} n={n}")
    if summary['warnings']:
        print(summary['warnings'][0], file=sys.stderr)
    return 0 if successes else 1

if __name__ == "__main__":
    raise SystemExit(main())
