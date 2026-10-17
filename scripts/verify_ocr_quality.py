"""Run in a fresh Python 3.11 process to reproduce the cold/warm A4 regression.

Uses local synthetic fixtures and cached models. Emits JSON evidence to stdout;
does not change datasets or model settings.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from app.pipelines.router import OCRRouter
from benchmarks.metrics import cer, wer
from app.services.errors import MODEL_LANGUAGE_UNSUPPORTED, PipelineError


def main():
    root = Path(__file__).resolve().parents[1]
    anns = [json.loads(line) for line in (root / "benchmark_data/annotations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:4]
    page = np.full((1754, 1240, 3), 255, np.uint8)
    y = 100
    for ann in anns:
        image = cv2.imread(str(root / "benchmark_data" / ann["image"]))
        scale = min(1050 / image.shape[1], 1.8)
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        page[y:y+image.shape[0], 80:80+image.shape[1]] = image
        y += image.shape[0] + 100
    ok, encoded = cv2.imencode(".png", page)
    assert ok
    expected = "\n".join(a["text"] for a in anns)
    router = OCRRouter()
    results = []
    for label, route, recognizer in [
        ("auto_cold", "auto", "auto"),
        ("scene_paddle", "scene", "paddle_v5_latin"),
        ("auto_warm", "auto", "auto"),
        ("document_hybrid", "document", "paddle_tesseract"),
        ("document_paddle", "document", "paddle_v5_latin"),
    ]:
        try:
            out = router.process(encoded.tobytes(), route=route, recognizer=recognizer, filename="a4_probe.png")
        except PipelineError as error:
            assert recognizer == "paddle_v5_latin" and error.code == MODEL_LANGUAGE_UNSUPPORTED
            results.append(dict(case=label, rejected=True, error=error.to_dict()))
            continue
        results.append(dict(case=label, route=out.route, engines=out.engines, text=out.text,
                            regions=len(out.regions), cer=round(cer(out.text, expected), 4),
                            wer=round(wer(out.text, expected), 4), warnings=out.warnings,
                            fallback_used=out.fallback_used))
    cold, _, warm, hybrid, paddle = results
    assert cold["engines"] == warm["engines"]
    assert cold["text"] == warm["text"] and cold["text"]
    assert cold["engines"]["recognizer"] == "tesseract"
    assert not cold["fallback_used"] and not warm["warnings"]
    assert hybrid["regions"] == 4 and hybrid["text"]
    assert paddle.get("rejected") or (paddle["regions"] == 4 and paddle["text"])
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(dict(expected=expected, results=results, assertions="passed"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
