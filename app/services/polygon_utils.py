"""Polygon helpers for scene OCR: bbox derivation and perspective rectification.

Pure OpenCV/NumPy; no paddle dependency. Paddle det output is BGR/RGB as given;
callers must NOT run Tesseract preprocessing on these crops.
"""

from __future__ import annotations

import numpy as np
import cv2
from fastapi import HTTPException

from app.schemas.ocr import BBox, DetectedRegion
from app.services.deadline import remaining_seconds


def polygon_to_bbox(polygon: list[list[float]]) -> BBox:
    """Axis-aligned bounding box that encloses the polygon."""
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    return BBox(
        x=int(round(x_min)),
        y=int(round(y_min)),
        width=max(0, int(round(x_max - x_min))),
        height=max(0, int(round(y_max - y_min))),
    )


def sort_reading_order(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """Group overlapping rows before sorting left-to-right; not column layout."""
    # ponytail: quadratic scan capped by MAX_REGIONS; add column layout only
    # when a labelled multi-column dataset demonstrates the need.
    rows = []
    for region in sorted(regions, key=lambda r: (r.bbox.y, r.bbox.x)):
        top, bottom = region.bbox.y, region.bbox.y + region.bbox.height
        for row in rows:
            overlap = min(bottom, row[1]) - max(top, row[0])
            if overlap > 0 and overlap >= .5 * min(bottom - top, row[1] - row[0]):
                row[0], row[1] = max(top, row[0]), min(bottom, row[1])
                row[2].append(region)
                break
        else:
            rows.append([top, bottom, [region]])
    return [r for row in rows for r in sorted(row[2], key=lambda r: r.bbox.x)]


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"order_quad expects 4 points, got {pts.shape[0]}")
    if not np.isfinite(pts).all() or len(np.unique(pts, axis=0)) != 4:
        raise ValueError("Quad must have four distinct finite points")
    centered = pts - pts.mean(axis=0)
    ordered = pts[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
    if not cv2.isContourConvex(ordered) or cv2.contourArea(ordered) < 1e-6:
        raise ValueError("Quad must be convex with nonzero area")
    # Sum ties at 45 degrees use y/x, independent of input permutation.
    start = np.lexsort((ordered[:, 0], ordered[:, 1], ordered.sum(axis=1)))[0]
    return np.roll(ordered, -int(start), axis=0)


def crop_polygon(image: np.ndarray, polygon: list[list[float]], *,
                 max_pixels: int | None = None) -> np.ndarray | None:
    """Perspective-rectify a (possibly rotated) text polygon into an upright crop.

    Falls back to a plain bbox crop for non-quad polygons. Returns None for
    degenerate regions. Input image is assumed BGR (OpenCV) and is not modified.
    Reject over-budget output before allocating a copy or perspective warp.
    """
    remaining_seconds()
    if image is None or image.size == 0:
        return None
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3 or not np.isfinite(pts).all():
        return None

    height, width = image.shape[:2]
    if pts.shape[0] != 4:
        bbox = polygon_to_bbox(polygon)
        x0 = max(0, bbox.x)
        y0 = max(0, bbox.y)
        x1 = min(width, bbox.x + bbox.width)
        y1 = min(height, bbox.y + bbox.height)
        if x1 <= x0 or y1 <= y0:
            return None
        if max_pixels is not None and (x1 - x0) * (y1 - y0) > max_pixels:
            raise HTTPException(413, detail="Total crop pixels exceed MAX_IMAGE_PIXELS")
        return image[y0:y1, x0:x1].copy()

    try:
        quad = order_quad(pts)
    except ValueError:
        return None
    top_w = np.linalg.norm(quad[1] - quad[0])
    bottom_w = np.linalg.norm(quad[2] - quad[3])
    left_h = np.linalg.norm(quad[3] - quad[0])
    right_h = np.linalg.norm(quad[2] - quad[1])
    out_w = int(round(max(top_w, bottom_w)))
    out_h = int(round(max(left_h, right_h)))
    if out_w < 2 or out_h < 2:
        return None
    if max_pixels is not None and out_w * out_h > max_pixels:
        raise HTTPException(413, detail="Total crop pixels exceed MAX_IMAGE_PIXELS")

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)
    if warped.size == 0:
        return None
    return warped
