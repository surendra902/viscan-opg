"""
Model definitions for Viscan OPG.

Architectures (selected by `model.name` in the config):
- fasterrcnn_resnet50_fpn_v2  (deployed baseline)
- retinanet_resnet50_fpn_v2   (architecture-comparison experiment)

Pretraining: COCO_V1 weights from torchvision.
Classes: 5 classes (0: background, 1: impacted_tooth, 2: caries, 3: periapical_lesion, 4: deep_caries).

License:
- Code & architecture: BSD-3-Clause (torchvision) for both detectors.
- Fine-tuned weights: Trained on DENTEX (CC BY-NC-SA 4.0).
  Commercial deployment requires retraining on a permissively licensed dataset
  or explicit permission from the DENTEX consortium.
"""

from functools import partial
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

try:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        retinanet_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    HAS_TORCHVISION = True
except ImportError:
    fasterrcnn_resnet50_fpn_v2 = None
    FasterRCNN_ResNet50_FPN_V2_Weights = None
    FastRCNNPredictor = None
    retinanet_resnet50_fpn_v2 = None
    RetinaNet_ResNet50_FPN_V2_Weights = None
    RetinaNetClassificationHead = None
    HAS_TORCHVISION = False

CLASS_NAMES = {
    0: "background",
    1: "impacted_tooth",
    2: "caries",
    3: "periapical_lesion",
    4: "deep_caries",
}

NUM_CLASSES = 5


def get_detector_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
    box_score_thresh: float = 0.001,
    box_detections_per_img: int = 300,
    box_nms_thresh: float = 0.5,
    rpn_score_thresh: float = 0.0,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_test: int = 1000,
    rpn_nms_thresh: float = 0.7,
    name: str = "fasterrcnn_resnet50_fpn_v2",
) -> nn.Module:
    """
    Constructs a torchvision detection model with a custom 5-class head.

    `name` selects the architecture:
    - "fasterrcnn_resnet50_fpn_v2" (default): two-stage, RPN + FPN
    - "retinanet_resnet50_fpn_v2": one-stage, focal loss
    """
    if "retinanet" in name.lower():
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        model = retinanet_resnet50_fpn_v2(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img,
        )
        # v2 uses GroupNorm(32) in the retinanet head; keep that norm for the new head
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels,
            num_anchors,
            num_classes,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        return model

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None

    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
        box_detections_per_img=box_detections_per_img,
        box_nms_thresh=box_nms_thresh,
        rpn_score_thresh=rpn_score_thresh,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
    )

    # Replace FastRCNNPredictor head with 5 classes (background + 4 findings)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def build_model_from_config(config: Dict[str, Any], pretrained: bool = True) -> nn.Module:
    """Builds model from a parsed YAML config dict."""
    model_cfg = config.get("model", {})
    return get_detector_model(
        name=model_cfg.get("name", "fasterrcnn_resnet50_fpn_v2"),
        num_classes=config.get("data", {}).get("num_classes", NUM_CLASSES),
        pretrained=pretrained,
        min_size=model_cfg.get("min_size", 800),
        max_size=model_cfg.get("max_size", 1333),
        box_score_thresh=model_cfg.get("box_score_thresh", 0.001),
        box_detections_per_img=model_cfg.get("box_detections_per_img", 300),
        box_nms_thresh=model_cfg.get("box_nms_thresh", 0.5),
        rpn_score_thresh=model_cfg.get("rpn_score_thresh", 0.0),
        rpn_pre_nms_top_n_test=model_cfg.get("rpn_pre_nms_top_n_test", 1000),
        rpn_post_nms_top_n_test=model_cfg.get("rpn_post_nms_top_n_test", 1000),
        rpn_nms_thresh=model_cfg.get("rpn_nms_thresh", 0.7),
    )


def load_model_weights(
    model: nn.Module,
    weights_path: str,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Loads checkpoint state dict into model."""
    checkpoint = torch.load(weights_path, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

