"""Unit tests for hflip TTA fusion and weighted box fusion."""

import numpy as np
import torch

from evaluate import _flip_xyxy, _nms_per_class
from ensemble_wbf import fuse_predictions, wbf_fuse_boxes


def test_flip_xyxy_mirrors_across_width():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 100.0, 10.0]])
    flipped = _flip_xyxy(boxes, 100.0)
    expected = torch.tensor([[50.0, 20.0, 90.0, 60.0], [0.0, 0.0, 100.0, 10.0]])
    assert torch.allclose(flipped, expected)
    assert torch.allclose(boxes, torch.tensor([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 100.0, 10.0]]))


def test_nms_keeps_highest_scoring_duplicate():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([1, 1, 1])
    kept_boxes, kept_scores, kept_labels = _nms_per_class(boxes, scores, labels, iou_thresh=0.5)
    assert len(kept_boxes) == 2
    assert torch.allclose(kept_scores, torch.tensor([0.9, 0.7]))
    assert len(kept_labels) == 2


def test_wbf_fuses_duplicate_boxes_with_weighted_coordinates():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]], dtype=np.float32)
    scores = np.array([0.9, 0.7], dtype=np.float32)
    fused_boxes, fused_scores = wbf_fuse_boxes(boxes, scores, iou_thresh=0.55)

    assert fused_boxes.shape == (1, 4)
    assert np.allclose(fused_boxes[0], np.average(boxes, axis=0, weights=scores), atol=1e-5)
    assert np.isclose(float(fused_scores[0]), 0.8, atol=1e-6)


def test_wbf_keeps_disjoint_boxes_separate():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 120.0, 120.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    fused_boxes, fused_scores = wbf_fuse_boxes(boxes, scores, iou_thresh=0.55)

    assert fused_boxes.shape == (2, 4)
    assert np.allclose(np.sort(fused_scores), np.array([0.8, 0.9], dtype=np.float32))


def test_wbf_handles_empty_input():
    fused_boxes, fused_scores = wbf_fuse_boxes(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    )
    assert fused_boxes.shape == (0, 4)
    assert fused_scores.shape == (0,)


def test_fuse_predictions_merges_across_models():
    preds_a = [{"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9}]
    preds_b = [
        {"image_id": 1, "category_id": 1, "bbox": [1.0, 0.0, 10.0, 10.0], "score": 0.7},
        {"image_id": 1, "category_id": 2, "bbox": [40.0, 40.0, 10.0, 10.0], "score": 0.6},
    ]
    fused, per_image = fuse_predictions([preds_a, preds_b], iou_thresh=0.55)

    assert len(per_image[1]) == 2
    class_one = [d for d in per_image[1] if d["category_id"] == 1]
    assert len(class_one) == 1
    assert np.isclose(class_one[0]["score"], 0.8, atol=1e-6)
    assert len(fused) == 2
