"""Unit tests for model architecture and policy configurations."""

import pytest
import torch
from model import get_detector_model, NUM_CLASSES, CLASS_NAMES, HAS_TORCHVISION

pytestmark = pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not installed in local test env")


def test_model_initialization():
    """Verify Faster R-CNN initialization and 5-class predictor head."""
    model = get_detector_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
        min_size=800,
        max_size=1333,
        box_score_thresh=0.001,
        box_detections_per_img=300,
        box_nms_thresh=0.5,
    )

    # Check head output classes
    cls_score = model.roi_heads.box_predictor.cls_score
    assert cls_score.out_features == 5
    assert len(CLASS_NAMES) == 5
    assert CLASS_NAMES[0] == "background"
    assert CLASS_NAMES[1] == "impacted_tooth"
    assert CLASS_NAMES[2] == "caries"
    assert CLASS_NAMES[3] == "periapical_lesion"
    assert CLASS_NAMES[4] == "deep_caries"


def test_model_forward_eval():
    """Verify forward pass in eval mode produces required dictionary keys."""
    model = get_detector_model(num_classes=5, pretrained=False)
    model.eval()

    dummy_img = torch.rand((3, 400, 600), dtype=torch.float32)
    with torch.no_grad():
        outputs = model([dummy_img])

    assert len(outputs) == 1
    out = outputs[0]
    assert "boxes" in out
    assert "labels" in out
    assert "scores" in out


def test_retinanet_initialization_and_forward():
    """Verify the RetinaNet comparison architecture builds a 5-class head and predicts."""
    model = get_detector_model(
        name="retinanet_resnet50_fpn_v2",
        num_classes=NUM_CLASSES,
        pretrained=False,
        min_size=128,
        max_size=128,
        box_detections_per_img=50,
    )
    model.eval()
    assert model.head.classification_head.num_classes == 5

    with torch.no_grad():
        outputs = model([torch.rand((3, 128, 160), dtype=torch.float32)])

    out = outputs[0]
    assert set(out.keys()) == {"boxes", "labels", "scores"}
    assert out["boxes"].shape[1] == 4
    assert out["labels"].max().item() <= 4
