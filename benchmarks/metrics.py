"""CER/WER/exact + Vietnamese diacritic + detection metrics. Stdlib only."""

from __future__ import annotations

import unicodedata


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character Error Rate. 0=perfect."""
    if not ref:
        return 0.0 if not pred else 1.0
    return _lev(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    """Word Error Rate."""
    pw, rw = pred.split(), ref.split()
    if not rw:
        return 0.0 if not pw else 1.0
    # word-level Levenshtein: treat each word as token
    # reuse _lev on token indices via dynamic programming on word lists
    m, n = len(pw), len(rw)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if pw[i - 1] == rw[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / len(rw)


def exact_match(pred: str, ref: str) -> bool:
    return pred.strip() == ref.strip()

# ---- Vietnamese diacritic helpers ----
_VIETNAMESE_BASE = {
    "ă": "a", "â": "a", "ê": "e", "ô": "o", "ơ": "o", "ư": "u", "đ": "d",
    "Ă": "A", "Â": "A", "Ê": "E", "Ô": "O", "Ơ": "O", "Ư": "U", "Đ": "D",
}

def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics via NFD + Mn filter; map đ/Đ explicitly.

    Keeps base letters, drops combining marks. Used for accent-stripped CER.
    Example: 'Xin chào Việt Nam' -> 'Xin chao Viet Nam'
    """
    if not text:
        return text
    # handle đ before NFD (đ decomposes differently)
    # NFD will map characters to base + combining; filter Mn
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # NFD maps đ -> d + combining? Actually đ is LATIN SMALL LETTER D WITH STROKE, NFD stays đ, so map.
    # Apply mapping for characters that survive.
    stripped = "".join(_VIETNAMESE_BASE.get(c, c) for c in stripped)
    # Re-compose to NFC for stability
    return unicodedata.normalize("NFC", stripped)


def accent_stripped_cer(pred: str, ref: str) -> float:
    """CER after stripping accents from both strings."""
    return cer(strip_accents(pred), strip_accents(ref))


def diacritic_loss_rate(pred: str, ref: str) -> float:
    """Fraction of CER attributable to diacritic loss.

    Formula: cer(pred, ref) - accent_stripped_cer(pred, ref)
    Clamped to >=0. When 0, base letters match; when >0, diacritics lost.
    For 'Xin chào Việt Nam' vs 'Xin chao Viet Nam': cer>0, accent_cer==0 => loss==cer.
    Documented explicitly; do not replace normal CER.
    """
    c = cer(pred, ref)
    ac = accent_stripped_cer(pred, ref)
    loss = c - ac
    return loss if loss > 0 else 0.0


# ---- IoU / detection metrics ----
def bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU for axis-aligned bboxes [x,y,w,h]."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def polygon_to_bbox_xywh(polygon: list[list[float]]) -> list[float]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def polygon_iou(a: list[list[float]], b: list[list[float]], *, backend: str = 'opencv') -> float:
    """Explicit polygon backend; invalid geometry never silently becomes bbox IoU."""
    if backend == 'bbox':
        return bbox_iou(polygon_to_bbox_xywh(a), polygon_to_bbox_xywh(b))
    if backend == 'shapely':
        from shapely.geometry import Polygon as _Poly  # type: ignore
        pa, pb = _Poly(a), _Poly(b)
        if not pa.is_valid or not pb.is_valid:
            raise ValueError('Invalid polygon')
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return inter / union if union else 0.0
    if backend != 'opencv':
        raise ValueError(f'Unknown IoU backend: {backend}')
    import cv2
    import numpy as np
    pa, pb = [np.asarray(points, np.float32).reshape(-1, 2) for points in (a, b)]
    if any(len(p) < 3 or not np.isfinite(p).all() or not cv2.isContourConvex(p) for p in (pa, pb)):
        raise ValueError('OpenCV IoU requires finite convex polygons; use shapely for concave GT')
    intersection = float(cv2.intersectConvexConvex(pa, pb)[0])
    union = cv2.contourArea(pa) + cv2.contourArea(pb) - intersection
    return intersection / union if union > 0 else 0.0


def detection_metrics(
    pred_polys: list[list[list[float]]],
    gt_polys: list[list[list[float]]],
    iou_threshold: float = 0.5,
    *, backend: str = 'opencv',
) -> dict:
    """Match pred<->GT by IoU greedily; return precision/recall/F1/IoU stats.

    Matching: each GT matched to at most one pred with max IoU >= threshold.
    """
    if not gt_polys and not pred_polys:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0, "missed": 0, "fp": 0, "mean_iou": 1.0}
    if not gt_polys:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0, "missed": 0, "fp": len(pred_polys), "mean_iou": 0.0}
    if not pred_polys:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0, "missed": len(gt_polys), "fp": 0, "mean_iou": 0.0}
    # Greedy matching
    used_pred = set()
    matched = 0
    ious: list[float] = []
    for gt in gt_polys:
        best_i, best_v = -1, 0.0
        for i, pred in enumerate(pred_polys):
            if i in used_pred:
                continue
            v = polygon_iou(gt, pred, backend=backend)
            if v > best_v:
                best_v, best_i = v, i
        if best_v >= iou_threshold and best_i != -1:
            matched += 1
            used_pred.add(best_i)
            ious.append(best_v)
    fp = len(pred_polys) - matched
    missed = len(gt_polys) - matched
    precision = matched / len(pred_polys) if pred_polys else 0.0
    recall = matched / len(gt_polys) if gt_polys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matched, "missed": missed, "fp": fp, "mean_iou": mean_iou}


# ---- critical field ----
def field_exact_match(pred: str, ref: str) -> bool:
    """Exact match for critical fields (strip, case-sensitive)."""
    return (pred or "").strip() == (ref or "").strip()


# Vietnamese tone marks set for error breakdown
VIET_TONE_MARKS = {"\u0301", "\u0300", "\u0309", "\u0303", "\u0323"}  # sắc huyền hỏi ngã nặng combining
VIET_DIACRITIC_CHARS = frozenset(["ă", "â", "ê", "ô", "ơ", "ư", "đ", "Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ"])
