"""Reproduce synthetic 12-line CPU/RAM/quality and batch equivalence measurements.

Run in .venv311: python scripts/verify_ocr_performance.py --engine document
or --engine hybrid. psutil is optional; memory/CPU fields are null without it.
Does not install/download models or alter settings/datasets. Emits JSON stdout.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ALLOW_MODEL_DOWNLOAD"] = "false"
import cv2
import numpy as np
from app.pipelines.router import OCRRouter
from app.recognizers.tesseract_recognizer import TesseractRecognizer
from app.services.normalization import normalize_text
from benchmarks.metrics import cer, wer
from benchmarks.dataset import build_manifest, model_fingerprints
from app.core.config import get_settings
from app.services.paddle_models import DET_MODEL
from dataclasses import asdict

try:
    import psutil
except ImportError:
    psutil = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("document", "hybrid"), default="document")
    parser.add_argument("--lang", default="vie+eng")
    parser.add_argument('--repeats', type=int, default=30, help='Warm requests, in addition to one cold request')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = root / "benchmark_data_quality_v2"
    annotations = [json.loads(line) for line in (dataset / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:12]
    images = [cv2.imread(str(dataset / a["image"])) for a in annotations]
    assert len(images) == 12 and all(im is not None for im in images)
    page = np.full((1754, 1240, 3), 255, np.uint8)
    for i, image in enumerate(images):
        y = 60 + 130 * i
        page[y:y + image.shape[0], 60:60 + image.shape[1]] = image
    ok, encoded = cv2.imencode(".png", page)
    assert ok
    expected = "\n".join(a["text"] for a in annotations)
    router = OCRRouter()
    records = []
    proc = psutil.Process() if psutil else None
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    for repeat in range(args.repeats + 1):
        state = {"peak": 0, "children": set(), "child_cpu": {}}
        stop = threading.Event()
        def sample():
            while not stop.is_set():
                rss = 0
                for p in [proc] + proc.children(recursive=True):
                    try:
                        rss += p.memory_info().rss
                        if p.pid != proc.pid:
                            state["children"].add(p.pid)
                            cpu = p.cpu_times()
                            state["child_cpu"][p.pid] = cpu.user + cpu.system
                    except psutil.Error:
                        pass
                state["peak"] = max(state["peak"], rss)
                stop.wait(.02)
        sampler = threading.Thread(target=sample, daemon=True) if proc else None
        if sampler:
            sampler.start()
        cpu_start = proc.cpu_times() if proc else None
        start = time.perf_counter()
        try:
            out = router.process(encoded.tobytes(), route="document" if args.engine == "document" else "scene",
                                 recognizer="auto", lang=args.lang)
        finally:
            elapsed = time.perf_counter() - start
            stop.set()
            if sampler:
                sampler.join()
        cpu_end = proc.cpu_times() if proc else None
        records.append(dict(repeat=repeat, phase="cold" if repeat == 0 else "warm",
                            ms=round(elapsed * 1000, 1), cer=round(cer(out.text, expected), 4),
                            wer=round(wer(out.text, expected), 4), regions=len(out.regions),
                            peak_tree_rss_mib=round(state["peak"] / 2**20, 1) if proc else None,
                            rss_after_mib=round(proc.memory_info().rss / 2**20, 1) if proc else None,
                            subprocesses_sampled=len(state["children"]) if proc else None,
                            cpu_seconds_est=round(cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
                                                  + sum(state["child_cpu"].values()), 3) if proc else None,
                            timings=out.timing_ms.to_dict(), engines=out.engines))
    rec = TesseractRecognizer()
    # Same detected crops for per-call vs batch comparison; avoid detector timing confound.
    if args.engine == "hybrid":
        from app.detectors.paddle_detector import PaddleDetector
        from app.services.polygon_utils import crop_polygon, sort_reading_order
        detector = PaddleDetector()
        crops = [crop_polygon(page, d.polygon) for d in sort_reading_order(detector.detect(page))]
    else:
        crops = images
    assert all(c is not None and c.size for c in crops)
    def measure(operation):
        times, result = [], None
        for _ in range(3):
            start = time.perf_counter()
            result = operation()
            times.append(round((time.perf_counter() - start) * 1000, 1))
        return times, result
    individual_times, individual = measure(lambda: [rec.recognize(c, lang=args.lang, psm=7) for c in crops])
    batch_times, batch = measure(lambda: rec.recognize_many(crops, lang=args.lang, psm=7))
    assert [r.to_dict() for r in individual] == [r.to_dict() for r in batch], "Batch changed OCR outputs/confidence"
    blank_crops = [crops[0], np.full((60, 450, 3), 255, np.uint8), crops[-1]]
    blank_single = [rec.recognize(c, lang=args.lang, psm=7) for c in blank_crops]
    blank_batch = rec.recognize_many(blank_crops, lang=args.lang, psm=7)
    assert [r.to_dict() for r in blank_single] == [r.to_dict() for r in blank_batch], "Blank page shifted mapping"
    text, _ = normalize_text("\n".join(r.raw_text for r in batch if r.raw_text))
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(dict(engine=args.engine, lang=args.lang, dataset="synthetic_quality_v2_12_lines",
                         dataset_manifest=build_manifest(annotations, dataset), runtime_config=asdict(get_settings()),
                         model_artifacts=model_fingerprints([DET_MODEL] if args.engine == 'hybrid' else [], [args.lang]),
                         runs=records, individual_recognition_ms=individual_times, batch_recognition_ms=batch_times,
                         batch_same_as_individual=True, blank_mapping_preserved=True,
                         crop_cer=round(cer(text, expected), 4), assertions="passed"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
