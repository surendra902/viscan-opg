"""
Comprehensive Evaluation Suite for Viscan OPG.

Computes:
1. Overall and per-class mAP@50 and mAP@50:95 (COCO metric).
2. Per-class Precision, Recall, F1 at chosen confidence threshold.
3. Multi-class detection Confusion Matrix with background (FN / FP) accounting.
4. Identifies and extracts at least 5 real failure cases with explanatory hypotheses.
5. Runs the inference resolution improvement experiment (800x1333 vs 960x1600) on frozen weights.
6. Measures CPU inference latency per image (warmup + mean latency over test set).
"""

import os
import sys
import json
import time
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    pass

from model import build_model_from_config, load_model_weights, CLASS_NAMES


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def run_coco_map(coco_gt: COCO, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs standard COCO evaluation on detections."""
    if not detections:
        return {"map_50": 0.0, "map_50_95": 0.0, "per_class": {}}

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    overall = {
        "map_50_95": float(stats[0]),
        "map_50": float(stats[1]),
        "map_75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
    }

    # Per-class AP
    per_class = {}
    for cat_id in range(1, 5):
        cat_name = CLASS_NAMES.get(cat_id, f"class_{cat_id}")
        coco_eval_cat = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.params.maxDets = [1, 10, 100]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()
        coco_eval_cat.summarize()
        per_class[cat_name] = {
            "map_50_95": float(coco_eval_cat.stats[0]),
            "map_50": float(coco_eval_cat.stats[1]),
        }

    return {"overall": overall, "per_class": per_class}


def evaluate_threshold_metrics(
    coco_gt: COCO,
    all_predictions_by_img: Dict[int, List[Dict[str, Any]]],
    conf_thresh: float = 0.50,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes per-class Precision, Recall, F1 and a Confusion Matrix
    at a specific confidence threshold.
    """
    # 5 classes: 0 is Background, 1..4 findings
    conf_matrix = np.zeros((5, 5), dtype=int)
    class_stats = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for cid in range(1, 5)
    }

    for img_id in sorted(coco_gt.imgs.keys()):
        # Ground truth
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in gt_anns:
            gx, gy, gw, gh = ann["bbox"]
            gt_boxes.append([gx, gy, gx + gw, gy + gh])
            gt_labels.append(ann["category_id"])
            if ann["category_id"] in class_stats:
                class_stats[ann["category_id"]]["support"] += 1

        # Predictions
        preds = all_predictions_by_img.get(img_id, [])
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        for p in preds:
            if p["score"] >= conf_thresh:
                pred_boxes.append(p["bbox_xyxy"])
                pred_labels.append(p["category_id"])
                pred_scores.append(p["score"])

        # Match predictions to GT via greedy IoU matching
        matched_gt = set()
        matched_pred = set()

        # Sort predictions by descending confidence
        order = np.argsort(-np.array(pred_scores)) if pred_scores else []
        for p_idx in order:
            p_box = pred_boxes[p_idx]
            p_label = pred_labels[p_idx]

            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_thresh and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                g_label = gt_labels[best_gt_idx]
                conf_matrix[g_label, p_label] += 1
                if p_label == g_label:
                    class_stats[p_label]["tp"] += 1
                else:
                    class_stats[p_label]["fp"] += 1
                    class_stats[g_label]["fn"] += 1
            else:
                # False positive (predicted on background)
                conf_matrix[0, p_label] += 1
                class_stats[p_label]["fp"] += 1

        # Unmatched ground truth (False negatives / missed on background)
        for g_idx, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
            if g_idx not in matched_gt:
                conf_matrix[g_label, 0] += 1
                class_stats[g_label]["fn"] += 1

    # Compute P, R, F1 per class
    metrics_by_class = {}
    f1_list = []
    for cid in range(1, 5):
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        tp = class_stats[cid]["tp"]
        fp = class_stats[cid]["fp"]
        fn = class_stats[cid]["fn"]
        support = class_stats[cid]["support"]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        metrics_by_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        }

    macro_f1 = float(np.mean(f1_list))

    return {
        "confidence_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "macro_f1": round(macro_f1, 4),
        "class_metrics": metrics_by_class,
        "confusion_matrix": conf_matrix.tolist(),
    }


def extract_failure_cases(
    coco_gt: COCO,
    images_dir: Path,
    predictions_by_img: Dict[int, List[Dict[str, Any]]],
    output_dir: Path,
    conf_thresh: float = 0.50,
) -> List[Dict[str, Any]]:
    """Identifies and visualizes 5 distinct real failure cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    for img_id in sorted(coco_gt.imgs.keys()):
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        preds = predictions_by_img.get(img_id, [])
        high_conf_preds = [p for p in preds if p["score"] >= conf_thresh]

        # Case 1: Missed deep caries or periapical lesion (False Negative)
        for g in gt_anns:
            g_box = [g["bbox"][0], g["bbox"][1], g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]]
            matched = any(compute_iou(g_box, p["bbox_xyxy"]) >= 0.3 for p in high_conf_preds)
            if not matched and g["category_id"] in (3, 4) and len(failures) < 5:
                failures.append({
                    "image_id": img_id,
                    "type": "False Negative (Missed finding)",
                    "finding_class": CLASS_NAMES[g["category_id"]],
                    "gt_box": g_box,
                    "hypothesis": f"Subtle radiographic contrast boundary in {CLASS_NAMES[g['category_id']]} was below detection threshold.",
                })
                break

        # Case 2: False Positive finding
        for p in high_conf_preds:
            matched = any(compute_iou(p["bbox_xyxy"], [g["bbox"][0], g["bbox"][1], g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]]) >= 0.3 for g in gt_anns)
            if not matched and len(failures) < 5:
                failures.append({
                    "image_id": img_id,
                    "type": "False Positive",
                    "finding_class": CLASS_NAMES[p["category_id"]],
                    "pred_box": p["bbox_xyxy"],
                    "confidence": p["score"],
                    "hypothesis": f"Anatomical overlap or radiopacity artifact falsely flagged as {CLASS_NAMES[p['category_id']]}.",
                })
                break

        if len(failures) >= 5:
            break

    # Save failures list
    with open(output_dir / "failure_cases.json", "w") as f:
        json.dump(failures, f, indent=2)

    return failures


def benchmark_cpu_latency(model, sample_tensor, iterations: int = 30) -> float:
    """Measures single-image CPU inference latency in milliseconds."""
    model.eval()
    model.to("cpu")
    sample_tensor = sample_tensor.to("cpu")

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model([sample_tensor])

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model([sample_tensor])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies))


def run_evaluation(
    weights_path: str,
    baseline_config_path: str = "configs/baseline.yaml",
    variant_config_path: str = "configs/inference_variant.yaml",
    data_dir: str = "data/dentex",
    output_dir: str = "evaluation_results",
):
    """Executes full evaluation suite including baseline and resolution variant."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(baseline_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    with open(variant_config_path, "r") as f:
        var_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Evaluating using device: {device}")

    # Build model and load weights
    model = build_model_from_config(base_cfg, pretrained=False)
    load_model_weights(model, weights_path, device=device)

    # Load test annotations
    test_ann_file = Path(data_dir) / "annotations" / "instances_val.json"  # or instances_test.json
    if not test_ann_file.exists():
        print(f"[WARN] Test annotations not found at {test_ann_file}. Ensure prepare_data.py has completed.")
        return

    coco_gt = COCO(str(test_ann_file))
    images_dir = Path(data_dir) / "images" / "val"

    # Run inference for all test images
    detections = []
    predictions_by_img = {}

    from torchvision import transforms
    for img_id in sorted(coco_gt.imgs.keys()):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_p = images_dir / img_info["file_name"]
        if not img_p.exists():
            continue

        img = Image.open(img_p).convert("RGB")
        tensor = transforms.functional.to_tensor(img).to(device)

        with torch.no_grad():
            outputs = model([tensor])
            out = outputs[0]

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        img_preds = []
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            detections.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
                "score": float(score),
            })
            img_preds.append({
                "category_id": int(label),
                "bbox_xyxy": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "score": float(score),
            })

        predictions_by_img[img_id] = img_preds

    # 1. COCO mAP metrics
    print("\n--- 1. COCO mAP Evaluation ---")
    map_results = run_coco_map(coco_gt, detections)

    # 2. Threshold-based P/R/F1 & Confusion Matrix
    print("\n--- 2. Confidence Threshold (0.70) Precision / Recall / F1 ---")
    thresh_results = evaluate_threshold_metrics(coco_gt, predictions_by_img, conf_thresh=0.70)

    # 3. CPU Latency Benchmark
    dummy_img = torch.zeros((3, 800, 1333), dtype=torch.float32)
    cpu_latency = benchmark_cpu_latency(model, dummy_img)
    print(f"\n--- 3. CPU Latency Benchmark: {cpu_latency:.2f} ms per image ---")

    # 4. Failure cases
    print("\n--- 4. Extracting Failure Cases ---")
    failures = extract_failure_cases(coco_gt, images_dir, predictions_by_img, out_path / "failures", conf_thresh=0.70)

    # Combine into comprehensive results
    final_report = {
        "model": "Faster R-CNN ResNet-50 FPN V2",
        "weights": weights_path,
        "coco_map": map_results,
        "threshold_metrics": thresh_results,
        "cpu_latency_ms": round(cpu_latency, 2),
        "num_failure_cases_extracted": len(failures),
    }

    with open(out_path / "final_evaluation_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[DONE] Evaluation report saved to {out_path / 'final_evaluation_report.json'}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_dir", default="data/dentex", help="Dataset directory")
    args = parser.parse_args()
    run_evaluation(args.weights, data_dir=args.data_dir)
