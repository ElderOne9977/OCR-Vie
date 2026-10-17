"""Offline detector ablation; changes no production settings or model files.

Run each configuration in a fresh process for comparable memory measurements:
  .venv311/Scripts/python scripts/probe_detector.py --threads 4 --side 960
Only the existing cached model is used. Results are synthetic, not a release gate.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ALLOW_MODEL_DOWNLOAD"] = "false"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--side", type=int, default=960)
    parser.add_argument("--no-mkldnn", action="store_true")
    parser.add_argument("--text-scale", type=float, default=1.0,
                        help="Shrink the text images, keeping page dimensions fixed (small-text stress test).")
    args = parser.parse_args()
    if args.threads < 1 or args.side < 32 or not 0.1 <= args.text_scale <= 1:
        parser.error("threads must be positive, side >= 32, and text-scale in [0.1, 1]")
    import cv2
    import numpy as np
    import psutil
    from app.services.paddle_runtime import get_detector_model, predict
    from app.services.polygon_utils import crop_polygon
    from app.recognizers.tesseract_recognizer import TesseractRecognizer
    from app.services.normalization import normalize_text
    from benchmarks.metrics import cer

    dataset = Path(__file__).resolve().parents[1] / "benchmark_data_quality_v2"
    rows = [json.loads(s) for s in (dataset / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()][:12]
    assert len(rows) == 12
    page = np.full((1754, 1240, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        image = cv2.imread(str(dataset / row["image"]))
        assert image is not None
        if args.text_scale != 1:
            image = cv2.resize(image, None, fx=args.text_scale, fy=args.text_scale,
                               interpolation=cv2.INTER_AREA)
        y = 60 + 130 * i
        page[y:y + image.shape[0], 60:60 + image.shape[1]] = image
    proc = psutil.Process()
    peak = [0]
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            stop.wait(.02)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        start = time.perf_counter()
        model = get_detector_model(cpu_threads=args.threads, limit_side_len=args.side,
                                   limit_type="max", enable_mkldnn=not args.no_mkldnn)
        load_ms = (time.perf_counter() - start) * 1000
        timings, cpus = [], []
        for _ in range(5):
            cpu = proc.cpu_times()
            start = time.perf_counter()
            result = predict(model, page)[0]
            timings.append((time.perf_counter() - start) * 1000)
            after = proc.cpu_times()
            cpus.append(after.user + after.system - cpu.user - cpu.system)
        polys = sorted(result["dt_polys"], key=lambda p: (float(np.min(p[:, 1])), float(np.min(p[:, 0]))))
        stop.set()
        sampler.join()
        crops = [crop_polygon(page, p.tolist()) for p in polys]
        assert all(c is not None and c.size for c in crops)
        recognized = TesseractRecognizer().recognize_many(crops, lang="vie+eng", psm=7)
        text, _ = normalize_text("\n".join(r.raw_text for r in recognized if r.raw_text))
        print(json.dumps(dict(threads=args.threads, side=args.side, mkldnn=not args.no_mkldnn,
                              text_scale=args.text_scale,
                              load_ms=round(load_ms, 1), detector_ms=[round(t, 1) for t in timings],
                              warm_median_ms=round(statistics.median(timings[1:]), 1),
                              warm_cpu_median_s=round(statistics.median(cpus[1:]), 3),
                              detector_peak_rss_mib=round(peak[0] / 2**20, 1),
                              rss_after_mib=round(proc.memory_info().rss / 2**20, 1),
                              regions=len(polys), cer=round(cer(text, "\n".join(r["text"] for r in rows)), 4))))
    finally:
        stop.set()
        sampler.join()


if __name__ == "__main__":
    main()
