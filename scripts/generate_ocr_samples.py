#!/usr/bin/env python3
"""Generate safe synthetic OCR samples using only system fonts.

Deterministic via --seed. No personal data. Tagged `synthetic`.
Supports variants: clean, blur, jpeg, rotation, perspective, low_contrast, small_text.
Writes: benchmark_data/images/*.png, annotations.jsonl, splits/*.txt
Stdlib + Pillow + OpenCV + NumPy only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmark_data"

# Safe synthetic texts (no PII). Covers diacritics + numbers.
SAMPLES = [
    ("Xin chào Việt Nam", {"category": "scene"}),
    ("Hà Nội 2026", {}),
    ("Số 123 Đường Lê Lợi", {}),
    ("Tổng tiền: 1.250.000 VNĐ", {"fields": {"amount": "1.250.000"}}),
    ("Mã số thuế: 012345678901", {"fields": {"tax_id": "012345678901"}}),
    ("Hóa đơn 001/2026", {"fields": {"document_id": "001/2026"}}),
    ("Công ty TNHH ABC", {}),
    ("Ngày 02/10/2026", {}),
    ("Cửa hàng 24h", {}),
    ("Địa chỉ: 45 Phan Chu Trinh", {}),
    ("Số lượng: 10", {}),
    ("Thành phố Hồ Chí Minh", {}),
]


def _find_font(size: int = 32) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\ARIAL.TTF",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render(text: str, variant: str, seed: int, idx: int) -> tuple[np.ndarray, list[list[float]]]:
    stable_hash = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rnd = random.Random(seed + idx * 1000 + stable_hash % 1000)
    # variant param influences rendering
    font_size = 28
    if variant == "small_text":
        font_size = 14
    font = _find_font(font_size)
    # measure text
    tmp = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 24
    W, H = tw + pad * 2, th + pad * 2 + 10
    W = max(W, 320); H = max(H, 80)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    # low_contrast: lighter text / gray bg
    fill = (0, 0, 0)
    bg = (255, 255, 255)
    if variant == "low_contrast":
        img = Image.new("RGB", (W, H), (220, 220, 220))
        draw = ImageDraw.Draw(img)
        fill = (90, 90, 90)
    draw.text((pad, pad), text, font=font, fill=fill)
    # textbbox includes baseline offsets; retain them and a small ink margin.
    x0, y0, x1, y1 = draw.textbbox((pad, pad), text, font=font)
    poly = np.float32([[x0-2, y0-2], [x1+2, y0-2], [x1+2, y1+2], [x0-2, y1+2]])
    # rotation variant
    if variant == "rotation":
        angle = rnd.uniform(-8, 8)
        matrix = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)
        corners = cv2.transform(np.float32([[[0, 0], [W, 0], [W, H], [0, H]]]), matrix)[0]
        lower = np.floor(corners.min(axis=0))
        upper = np.ceil(corners.max(axis=0))
        matrix[:, 2] -= lower
        arr = cv2.warpAffine(np.array(img), matrix, tuple((upper-lower).astype(int)), flags=cv2.INTER_CUBIC, borderValue=(255,255,255))
        poly = cv2.transform(poly[None], matrix)[0]
        img = Image.fromarray(arr)
    # perspective variant (simple shear)
    if variant == "perspective":
        arr = np.array(img)
        h, w = arr.shape[:2]
        dx = int(w * 0.06)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[rnd.randint(0, dx), 0], [w - rnd.randint(0, dx), 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        arr = cv2.warpPerspective(arr, M, (w, h), borderValue=(255, 255, 255))
        poly = cv2.perspectiveTransform(poly[None], M)[0]
        img = Image.fromarray(arr)
    arr = np.array(img)
    # convert RGB -> BGR for opencv pipeline compat
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if variant == "blur":
        arr_bgr = cv2.GaussianBlur(arr_bgr, (5, 5), 1.2)
    if variant == "jpeg":
        _, enc = cv2.imencode(".jpg", arr_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        arr_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    h, w = arr_bgr.shape[:2]
    poly[:, 0] = np.clip(poly[:, 0], 0, w-1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h-1)
    return arr_bgr, poly.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="*", default=["clean"],
                    help="subset of clean blur jpeg rotation perspective low_contrast small_text")
    args = ap.parse_args()
    out = Path(args.output)
    if (out / "annotations.jsonl").exists():
        ap.error("Output already contains annotations; choose a new --output directory to preserve existing data.")
    img_dir = out / "images"
    crops_dir = out / "crops"
    splits_dir = out / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants if args.variants else ["clean"]
    n = min(args.count, len(SAMPLES))
    entries = []
    for i in range(n):
        text, extra = SAMPLES[i % len(SAMPLES)]
        variant = variants[i % len(variants)]
        # tag variant
        tags = ["synthetic", variant]
        arr, poly = _render(text, variant, args.seed, i)
        fname = f"synth_{i:03d}_{variant}.png"
        fpath = img_dir / fname
        cv2.imwrite(str(fpath), arr)
        entry = {
            "id": f"synth_{i:03d}",
            "image": f"images/{fname}",
            "split": "test" if i < n*0.6 else ("val" if i < n*0.85 else "train"),
            "category": extra.get("category", "scene"),
            "language": "vie",
            "text": text,
            "tags": tags,
            "regions": [{"polygon": poly, "text": text}],
        }
        if "fields" in extra:
            entry["fields"] = extra["fields"]
        entries.append(entry)
    # simple split assignment already; write annotations
    ann_path = out / "annotations.jsonl"
    with ann_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    # write splits
    for sp in ("train", "val", "test"):
        ids = [e["id"] for e in entries if e["split"] == sp]
        (splits_dir / f"{sp}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    # README
    (out / "README.md").write_text(
        "# benchmark_data — synthetic safe samples\n\n"
        "Generated by `scripts/generate_ocr_samples.py` using only system fonts.\n"
        "No personal data. Tagged `synthetic`. Deterministic via --seed.\n"
        "Variants: blur, jpeg, rotation, perspective, low_contrast, small_text.\n"
        "Do not commit sensitive documents.\n", encoding="utf-8")
    print(f"Wrote {len(entries)} samples to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
