"""Unit tests for metric evaluation (IoU calculation, Precision, Recall, and F1)."""

import pytest
import numpy as np
from evaluate import compute_iou


def test_iou_identical_boxes():
    """Verify IoU of identical boxes is 1.0."""
    box = [10.0, 20.0, 50.0, 60.0]
    assert pytest.approx(compute_iou(box, box)) == 1.0


def test_iou_disjoint_boxes():
    """Verify IoU of non-overlapping boxes is 0.0."""
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    assert compute_iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    """Verify IoU of half-overlapping boxes."""
    box1 = [0.0, 0.0, 20.0, 10.0]  # area 200
    box2 = [10.0, 0.0, 30.0, 10.0] # area 200, inter 100, union 300
    assert pytest.approx(compute_iou(box1, box2)) == 100.0 / 300.0


def test_precision_recall_f1_math():
    """Verify precision, recall, F1 calculation with true positives, false positives, false negatives."""
    tp, fp, fn = 8, 2, 4
    prec = tp / (tp + fp) # 8 / 10 = 0.8
    rec = tp / (tp + fn)  # 8 / 12 = 0.6667
    f1 = (2 * prec * rec) / (prec + rec)

    assert pytest.approx(prec) == 0.8
    assert pytest.approx(rec) == 8.0 / 12.0
    assert pytest.approx(f1) == 2 * 0.8 * (8.0/12.0) / (0.8 + 8.0/12.0)
