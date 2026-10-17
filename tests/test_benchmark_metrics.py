"""Phase 3 benchmark metrics tests: CER/WER/exact/accent/diacritic/IoU/region matching.

Includes required regressions:
  - 'Xin chào Việt Nam' vs 'Xin chao Viet Nam' -> normal CER > 0, accent-stripped == 0
  - '012345678901' vs '0123456789O1' -> field exact False
  - '1.250.000' vs '1.250.000' -> field exact True
"""
from __future__ import annotations

import pytest

from benchmarks.metrics import (
    accent_stripped_cer,
    bbox_iou,
    cer,
    detection_metrics,
    diacritic_loss_rate,
    exact_match,
    field_exact_match,
    polygon_iou,
    strip_accents,
    wer,
)


def test_cer_perfect_is_zero():
    assert cer("abc", "abc") == 0.0


def test_cer_substitution():
    assert cer("abd", "abc") == pytest.approx(1 / 3)


def test_wer_is_word_level_not_char():
    # one word wrong out of two -> 0.5 (char-level on joined string would differ)
    assert wer("hello world", "hello word") == pytest.approx(0.5)
    # joined-char distance would be 1/11; assert we are NOT doing that
    assert wer("hello world", "hello word") != pytest.approx(1 / 11)


def test_wer_perfect_zero():
    assert wer("a b c", "a b c") == 0.0


def test_exact_match_trims():
    assert exact_match(" xin chao ", "xin chao")
    assert not exact_match("xin chao", "xin chao viet")


def test_strip_accents_vietnamese():
    assert strip_accents("Xin chào Việt Nam") == "Xin chao Viet Nam"
    assert strip_accents("Đường") == "Duong"


def test_regression_diacritic_cer_vs_accent_stripped():
    ref = "Xin chào Việt Nam"
    pred = "Xin chao Viet Nam"
    assert cer(pred, ref) > 0  # normal CER counts lost diacritics
    assert accent_stripped_cer(pred, ref) == 0.0  # base letters correct
    assert diacritic_loss_rate(pred, ref) == pytest.approx(cer(pred, ref))


def test_diacritic_loss_rate_clamped_non_negative():
    # when pred is better than accent-stripped baseline, never negative
    assert diacritic_loss_rate("Xin chào Việt Nam", "Xin chào Việt Nam") >= 0.0


def test_bbox_iou_identical():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_bbox_iou_disjoint():
    assert bbox_iou([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0


def test_bbox_iou_half_overlap():
    # 10x10 shifted by 5 -> intersection 5x10=50, union 100+100-50=150
    assert bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)


def test_polygon_iou_identical():
    poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_iou(poly, poly) == pytest.approx(1.0)


def test_detection_metrics_perfect_match():
    poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
    m = detection_metrics([poly], [poly])
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["matched"] == 1
    assert m["missed"] == 0
    assert m["fp"] == 0


def test_detection_metrics_missed_and_fp():
    gt = [[0, 0], [10, 0], [10, 10], [0, 10]]
    far = [[100, 100], [110, 100], [110, 110], [100, 110]]
    m = detection_metrics([far], [gt])
    assert m["matched"] == 0
    assert m["missed"] == 1
    assert m["fp"] == 1
    assert m["recall"] == 0.0


def test_detection_metrics_empty_both():
    m = detection_metrics([], [])
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_regression_field_tax_id_exact_false():
    assert field_exact_match("0123456789O1", "012345678901") is False


def test_regression_field_amount_exact_true():
    assert field_exact_match("1.250.000", "1.250.000") is True


def test_polygon_backend_is_explicit_and_never_falls_back():
    diamond = [[5,0],[10,5],[5,10],[0,5]]
    square = [[0,0],[10,0],[10,10],[0,10]]
    assert polygon_iou(diamond,square,backend='opencv') == pytest.approx(.5)
    assert polygon_iou(diamond,square,backend='bbox') == 1
    with pytest.raises(ValueError):
        polygon_iou([[0,0],[10,0],[2,2],[0,10]],square,backend='opencv')
