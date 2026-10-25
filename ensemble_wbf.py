"""
Weighted Box Fusion (WBF) ensemble evaluation for Viscan OPG.

Fuses the COCO prediction dumps of two or more detectors (see
`evaluate.py --dump_predictions`) and re-runs the same evaluation protocol:
COCO mAP overall/per class, validation-only threshold selection, per-class
P/R/F1, confusion matrix, and failure cases.

WBF is the fusion rule used by the top DENTEX 2023 entry (He et al.) and
DETDet (Choi et al.) to combine heterogeneous detectors; here it combines
Faster R-CNN and RetinaNet, both BSD-3-Clause torchvision models.

Usage:
    python ensemble_wbf.py \
        --gt data/dentex/annotations/instances_test.json \
        --images_dir data/dentex/images/test \
        --preds preds_frcnn/predictions_test.json preds_retinanet/predictions_test.json \
        --val-gt data/dentex/annotations/instances_val.json \
        --val-preds preds_frcnn/predictions_val.json preds_retinanet/predictions_val.json \
        --output_dir evaluation_ensemble

The test split is only read here for the single final scoring pass; the
threshold is selected on the validation split.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from evaluate import (
    evaluate_threshold_metrics,
    extract_failure_cases,
    run_coco_map,
)


def _xywh_to_xyxy(bbox: Sequence[float]) -> List[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def _xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_vector(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (n, 4) array of xyxy boxes."""
    if len(others) == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas_b = np.clip(others[:, 2] - others[:, 0], 0.0, None) * np.clip(
        others[:, 3] - others[:, 1], 0.0, None
    )
    union = area_a + areas_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def wbf_fuse_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuses same-class boxes with weighted box fusion.

    Boxes are visited in descending score order; a box joins the cluster whose
    fused box has the highest IoU >= iou_thresh. Fused coordinates are the
    score-weighted mean of the cluster members and the fused score is the mean
    member score (the standard WBF rule).
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = np.argsort(-scores)
    clusters: List[Dict[str, Any]] = []
    fused = np.zeros((0, 4), dtype=np.float64)

    for idx in order:
        box = boxes[idx].astype(np.float64)
        score = float(scores[idx])

        best_j, best_iou = -1, 0.0
        if len(clusters) > 0:
            ious = _iou_vector(box, fused)
            best_j = int(np.argmax(ious))
            best_iou = float(ious[best_j])

        if best_j >= 0 and best_iou >= iou_thresh:
            cluster = clusters[best_j]
            cluster["boxes"].append(box)
            cluster["scores"].append(score)
            weights = np.asarray(cluster["scores"])
            cluster["fused"] = np.average(np.asarray(cluster["boxes"]), axis=0, weights=weights)
            fused[best_j] = cluster["fused"]
        else:
            clusters.append({"boxes": [box], "scores": [score], "fused": box})
            fused = np.vstack([fused, box[None, :]])

    fused_boxes = np.asarray([c["fused"] for c in clusters], dtype=np.float32)
    fused_scores = np.asarray([float(np.mean(c["scores"])) for c in clusters], dtype=np.float32)
    return fused_boxes, fused_scores


def fuse_predictions(
    pred_lists: Sequence[List[Dict[str, Any]]],
    iou_thresh: float = 0.55,
    max_det: int = 300,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Fuses COCO-format prediction dumps and returns detections plus per-image view."""
    grouped: Dict[Tuple[int, int], Dict[str, List]] = defaultdict(
        lambda: {"boxes": [], "scores": []}
    )
    for preds in pred_lists:
        for det in preds:
            key = (int(det["image_id"]), int(det["category_id"]))
            grouped[key]["boxes"].append(_xywh_to_xyxy(det["bbox"]))
            grouped[key]["scores"].append(float(det["score"]))

    fused: List[Dict[str, Any]] = []
    per_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for (image_id, category_id) in sorted(grouped.keys()):
        payload = grouped[(image_id, category_id)]
        boxes, scores = wbf_fuse_boxes(
            np.asarray(payload["boxes"], dtype=np.float32),
            np.asarray(payload["scores"], dtype=np.float32),
            iou_thresh=iou_thresh,
        )
        for box, score in zip(boxes, scores):
            fused.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": _xyxy_to_xywh(box),
                "score": float(score),
            })
            per_image[image_id].append({
                "category_id": category_id,
                "bbox_xyxy": [float(v) for v in box],
                "score": float(score),
            })

    # Keep the strongest max_det detections per image so fusion cannot inflate recall artificially.
    trimmed: List[Dict[str, Any]] = []
    trimmed_per_image: Dict[int, List[Dict[str, Any]]] = {}
    for image_id, items in per_image.items():
        items = sorted(items, key=lambda d: -d["score"])[:max_det]
        trimmed_per_image[image_id] = items
        for item in items:
            trimmed.append({
                "image_id": image_id,
                "category_id": item["category_id"],
                "bbox": _xyxy_to_xywh(item["bbox_xyxy"]),
                "score": item["score"],
            })
    return trimmed, trimmed_per_image


def select_threshold(coco_val, val_per_image: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    sweep = []
    best_threshold, best_f1 = 0.70, -1.0
    for thr in np.arange(0.10, 0.91, 0.05):
        tau = float(round(thr, 2))
        res = evaluate_threshold_metrics(coco_val, val_per_image, conf_thresh=tau)
        sweep.append({"threshold": tau, "macro_f1": res["macro_f1"]})
        if res["macro_f1"] > best_f1:
            best_threshold, best_f1 = tau, res["macro_f1"]
    print(f"[VAL] Selected threshold {best_threshold:.2f} (fused val macro-F1={best_f1:.3f})")
    return {
        "mode": "val_sweep",
        "best_threshold": best_threshold,
        "best_val_macro_f1": round(float(best_f1), 4),
        "sweep": sweep,
    }


def run_ensemble(
    pred_paths: Sequence[str],
    gt_path: str,
    images_dir: str,
    val_pred_paths: Sequence[str],
    val_gt_path: str,
    output_dir: str = "evaluation_ensemble",
    iou_thresh: float = 0.55,
):
    from pycocotools.coco import COCO

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _load(paths):
        return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]

    test_preds = _load(pred_paths)
    val_preds = _load(val_pred_paths)

    coco_gt = COCO(gt_path)
    coco_val = COCO(val_gt_path)

    fused_test, test_per_image = fuse_predictions(test_preds, iou_thresh=iou_thresh)
    fused_val, val_per_image = fuse_predictions(val_preds, iou_thresh=iou_thresh)

    print(f"\n--- 1. Fused COCO mAP ({len(pred_paths)} members, IoU_fuse={iou_thresh}) ---")
    map_results = run_coco_map(coco_gt, fused_test)

    threshold_info = select_threshold(coco_val, val_per_image)
    chosen_tau = float(threshold_info["best_threshold"])

    print(f"\n--- 2. Fused test metrics at tau={chosen_tau:.2f} ---")
    thresh_results = evaluate_threshold_metrics(coco_gt, test_per_image, conf_thresh=chosen_tau)

    print("\n--- 3. Failure cases ---")
    failures = extract_failure_cases(
        coco_gt, Path(images_dir), test_per_image, out_path / "failures", conf_thresh=chosen_tau
    )

    report = {
        "model": "wbf_ensemble",
        "members": [str(p) for p in pred_paths],
        "iou_fuse_threshold": iou_thresh,
        "coco_map": map_results,
        "threshold_selection": threshold_info,
        "threshold_metrics": thresh_results,
        "cpu_latency_ms": None,
        "latency_note": "Not measured: fusion is post-processing; wall-clock is the sum of member model latencies.",
        "num_failure_cases_extracted": len(failures),
    }

    with open(out_path / "final_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE] Ensemble report saved to {out_path / 'final_evaluation_report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WBF ensemble of COCO prediction dumps")
    parser.add_argument("--preds", nargs="+", required=True, help="Test prediction dumps")
    parser.add_argument("--gt", required=True, help="Test COCO annotations")
    parser.add_argument("--images_dir", required=True, help="Test image directory (for failure cases)")
    parser.add_argument("--val-preds", nargs="+", required=True, help="Validation prediction dumps")
    parser.add_argument("--val-gt", required=True, help="Validation COCO annotations")
    parser.add_argument("--output_dir", default="evaluation_ensemble")
    parser.add_argument("--iou_thresh", type=float, default=0.55, help="WBF clustering IoU")
    args = parser.parse_args()

    run_ensemble(
        pred_paths=args.preds,
        gt_path=args.gt,
        images_dir=args.images_dir,
        val_pred_paths=args.val_preds,
        val_gt_path=args.val_gt,
        output_dir=args.output_dir,
        iou_thresh=args.iou_thresh,
    )
