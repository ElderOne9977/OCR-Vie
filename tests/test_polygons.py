import numpy as np
import itertools
import pytest

from app.schemas.ocr import BBox
from app.services.polygon_utils import crop_polygon, order_quad, polygon_to_bbox


def test_polygon_to_bbox():
    bbox = polygon_to_bbox([[10, 20], [30, 20], [30, 40], [10, 40]])
    assert bbox == BBox(x=10, y=20, width=20, height=20)


def test_order_quad_reorders_random_permutation():
    quad = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    perm = quad[[2, 0, 3, 1]]
    ordered = order_quad(perm)
    assert ordered.shape == (4, 2)
    # top-left should be (0,0)
    assert list(ordered[0]) == [0.0, 0.0]


def test_crop_polygon_axis_aligned():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:10, 5:10] = 255
    poly = [[5, 5], [10, 5], [10, 10], [5, 10]]
    crop = crop_polygon(img, poly)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_crop_polygon_degenerate_returns_none():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert crop_polygon(img, [[0, 0], [0, 0]]) is None
    assert crop_polygon(img, []) is None


def test_crop_polygon_tri_falls_back_to_bbox():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    # 3 points -> bbox fallback
    crop = crop_polygon(img, [[2, 2], [8, 2], [5, 8]])
    assert crop is not None
    assert crop.size > 0


def test_diamond_all_permutations_keep_four_corners():
    points = [(50, 0), (100, 50), (50, 100), (0, 50)]
    expected = order_quad(np.array(points, dtype=np.float32))
    for permutation in itertools.permutations(points):
        actual = order_quad(np.array(permutation, dtype=np.float32))
        assert len(np.unique(actual, axis=0)) == 4
        np.testing.assert_array_equal(actual, expected)
    crop = crop_polygon(np.full((101, 101, 3), 255, np.uint8), points)
    assert crop is not None and np.mean(crop) > 250


@pytest.mark.parametrize('angle', [44, 45, 46])
def test_rotated_quad_stays_convex(angle):
    import cv2
    points = np.float32([[[30, 40], [90, 40], [90, 60], [30, 60]]])
    rotated = cv2.transform(points, cv2.getRotationMatrix2D((60, 50), angle, 1))[0]
    ordered = order_quad(rotated[[2, 0, 3, 1]])
    assert len(np.unique(ordered, axis=0)) == 4
    assert cv2.isContourConvex(ordered)


@pytest.mark.parametrize('points', [
    [[0, 0], [0, 0], [5, 5], [0, 5]],
    [[0, 0], [1, 1], [2, 2], [3, 3]],
    [[0, 0], [5, 0], [float('nan'), 5], [0, 5]],
    [[0, 0], [10, 0], [2, 2], [0, 10]],
])
def test_invalid_quad_is_rejected_before_warp(points):
    assert crop_polygon(np.zeros((20, 20, 3), np.uint8), points) is None


def test_crop_preserves_colored_corner_orientation():
    image = np.zeros((30, 40, 3), np.uint8)
    image[5:15, 5:20] = (255, 0, 0)
    image[5:15, 20:35] = (0, 255, 0)
    image[15:25, 20:35] = (0, 0, 255)
    image[15:25, 5:20] = (255, 255, 255)
    crop = crop_polygon(image, [[35, 25], [5, 5], [5, 25], [35, 5]])
    assert crop is not None
    np.testing.assert_array_equal(crop[2, 2], (255, 0, 0))
    np.testing.assert_array_equal(crop[2, -3], (0, 255, 0))
    np.testing.assert_array_equal(crop[-3, -3], (0, 0, 255))
    np.testing.assert_array_equal(crop[-3, 2], (255, 255, 255))


def test_reading_order_groups_words_with_vertical_jitter():
    from app.services import polygon_utils
    from app.schemas.ocr import DetectedRegion
    right = DetectedRegion([], BBox(100, 9, 40, 20), .9)
    left = DetectedRegion([], BBox(10, 10, 40, 20), .9)
    below = DetectedRegion([], BBox(10, 40, 40, 20), .9)
    assert polygon_utils.sort_reading_order([right, below, left]) == [left, right, below]


def test_reading_order_is_row_major_not_inferred_column_layout():
    from app.services.polygon_utils import sort_reading_order
    from app.schemas.ocr import DetectedRegion
    regions = [DetectedRegion([], BBox(x, y, 40, 20), .9)
               for x,y in [(10,10),(200,10),(10,40),(200,40)]]
    assert sort_reading_order(list(reversed(regions))) == regions


def test_tall_region_does_not_bridge_separate_text_rows():
    from app.services.polygon_utils import sort_reading_order
    from app.schemas.ocr import DetectedRegion
    tall = DetectedRegion([], BBox(100,9,20,100),.9)
    first = DetectedRegion([], BBox(10,10,40,20),.9)
    second = DetectedRegion([], BBox(10,40,40,20),.9)
    assert sort_reading_order([second,tall,first]) == [first,tall,second]
