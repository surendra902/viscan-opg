"""
Cascade crop refiner: a ConvNeXt-Tiny classifier that re-scores (and optionally
re-labels) detector boxes from a higher-resolution crop.

Why this stage: the measured evaluation shows two dominant errors.
  1. Class confusion, mostly deep_caries scored as caries (14 of 32 deep_caries
     test boxes landed in the caries column of the confusion matrix).
  2. Background false positives, mostly caries at the 0.70 operating threshold
     (124 background -> caries cells).
Both are classification failures on boxes the detector already found, so a crop
classifier trained at 224x224 on the lesion itself can act on them without
changing the proposal geometry.

Training data is built ONLY from the train split annotations (no val/test GT):
  * positives: every GT box cropped with random context/scale jitter,
  * negatives: random boxes with IoU < 0.05 against every GT box in the image,
    sized from that image's GT size distribution.

Score composition at apply time (documented, not tuned on test):
  rescore: score = sqrt(det_score * p_label)
  reclass: label = argmax over the 4 finding classes; boxes whose argmax is the
           background class are dropped; score = sqrt(det_score * p_label_new)

License: ConvNeXt-Tiny code/weights are BSD-3-Clause (torchvision).

Usage:
    python cascade_refine.py build --data_dir data/dentex --output_dir checkpoints/cascade
    python cascade_refine.py train --bank_dir checkpoints/cascade --output checkpoints/cascade/convnext_cascade.pth
    python cascade_refine.py apply --checkpoint checkpoints/cascade/convnext_cascade.pth \
        --preds preds_slice/predictions_val_tiles.json --split val --data_dir data/dentex \
        --mode rescore --output_dir preds_cascade
"""

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

CLASS_NAMES = {0: "background", 1: "impacted_tooth", 2: "caries", 3: "periapical_lesion", 4: "deep_caries"}
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Crop geometry
# --------------------------------------------------------------------------- #
def context_crop_box(box: Sequence[float], context: float, width: int, height: int,
                     shift: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float, float, float]:
    """Expands an xyxy box by `context`, applies a relative shift, clips to image."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx += shift[0] * bw
    cy += shift[1] * bh
    half_w = bw * context / 2.0
    half_h = bh * context / 2.0
    nx1 = max(0.0, cx - half_w)
    ny1 = max(0.0, cy - half_h)
    nx2 = min(float(width), cx + half_w)
    ny2 = min(float(height), cy + half_h)
    return nx1, ny1, nx2, ny2


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xywh_to_xyxy(bbox: Sequence[float]) -> List[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


# --------------------------------------------------------------------------- #
# Crop bank construction (train split only)
# --------------------------------------------------------------------------- #
def build_crop_bank(
    data_dir: str,
    output_dir: str,
    size: int = CROP_SIZE,
    context: Tuple[float, float] = (1.15, 1.45),
    negatives_ratio: float = 1.0,
    seed: int = 42,
) -> Dict:
    from pycocotools.coco import COCO

    data_path = Path(data_dir)
    coco = COCO(str(data_path / "annotations" / "instances_train.json"))
    images_dir = data_path / "images" / "train"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco.anns.values():
        anns_by_image[int(ann["image_id"])].append(ann)

    # Pass 1: decide the sample list so the memmap can be allocated up front.
    samples: List[Tuple[int, List[float], int, float]] = []  # (image_id, xyxy, label, context)
    for image_id in sorted(anns_by_image.keys()):
        img_info = coco.loadImgs(image_id)[0]
        width, height = float(img_info["width"]), float(img_info["height"])
        gt_boxes = [_xywh_to_xyxy(a["bbox"]) for a in anns_by_image[image_id]]
        for ann, gt in zip(anns_by_image[image_id], gt_boxes):
            ctx = rng.uniform(*context)
            samples.append((image_id, gt, int(ann["category_id"]), ctx))

        num_neg = int(round(len(gt_boxes) * negatives_ratio))
        sizes = [(g[2] - g[0], g[3] - g[1]) for g in gt_boxes]
        made, attempts = 0, 0
        while made < num_neg and attempts < num_neg * 50:
            attempts += 1
            bw, bh = sizes[rng.randrange(len(sizes))]
            scale = rng.uniform(0.7, 1.4)
            bw, bh = max(8.0, bw * scale), max(8.0, bh * scale)
            if bw >= width or bh >= height:
                continue
            x1 = rng.uniform(0.0, width - bw)
            y1 = rng.uniform(0.0, height - bh)
            cand = [x1, y1, x1 + bw, y1 + bh]
            if max(iou_xyxy(cand, g) for g in gt_boxes) >= 0.05:
                continue
            samples.append((image_id, cand, 0, rng.uniform(*context)))
            made += 1

    crops = np.lib.format.open_memmap(
        out_path / "crops_uint8.npy", mode="w+", dtype=np.uint8,
        shape=(len(samples), size, size, 3),
    )
    labels = np.zeros((len(samples),), dtype=np.int64)

    index = 0
    for image_id, items in samples_grouped(samples):
        img_info = coco.loadImgs(image_id)[0]
        img = Image.open(images_dir / img_info["file_name"]).convert("RGB")
        width, height = img.size
        for gt, label, ctx in items:
            crop_box = context_crop_box(gt, ctx, width, height)
            if crop_box[2] - crop_box[0] < 4 or crop_box[3] - crop_box[1] < 4:
                crop_box = (max(0.0, gt[0]), max(0.0, gt[1]),
                            min(float(width), gt[2]), min(float(height), gt[3]))
            crop = img.crop(tuple(int(round(v)) for v in crop_box)).resize((size, size), Image.BILINEAR)
            crops[index] = np.asarray(crop, dtype=np.uint8)
            labels[index] = label
            index += 1
        del img

    crops.flush()
    del crops
    np.save(out_path / "labels.npy", labels)

    counts = Counter(int(v) for v in labels)
    meta = {
        "source_split": "train",
        "num_samples": int(index),
        "counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(counts.items())},
        "crop_size": size,
        "context_range": [context[0], context[1]],
        "negatives_ratio": negatives_ratio,
        "seed": seed,
        "note": "Built only from train-split GT. Negative boxes have IoU < 0.05 with every GT box.",
    }
    with open(out_path / "crop_bank_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[CASCADE] crop bank: {index} crops {dict(meta['counts'])} -> {out_path}")
    return meta


def samples_grouped(samples: Sequence[Tuple]):
    grouped: Dict[int, List[Tuple]] = defaultdict(list)
    for image_id, box, label, ctx in samples:
        grouped[int(image_id)].append((box, label, ctx))
    for image_id in sorted(grouped.keys()):
        yield image_id, grouped[image_id]


class CropBank(Dataset):
    def __init__(self, bank_dir: str, train: bool = True):
        bank = Path(bank_dir)
        self.crops = np.load(bank / "crops_uint8.npy", mmap_mode="r")
        self.labels = np.load(bank / "labels.npy")
        self.train = train

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        arr = np.asarray(self.crops[idx], dtype=np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                arr = arr[:, ::-1, :]
            arr = arr * random.uniform(0.9, 1.1)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))), int(self.labels[idx])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_cascade(
    bank_dir: str,
    output_path: str,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 2,
    seed: int = 42,
    device: str = "",
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    bank = CropBank(bank_dir, train=True)
    with open(Path(bank_dir) / "crop_bank_meta.json", "r", encoding="utf-8") as f:
        bank_meta = json.load(f)

    counts = torch.zeros(5, dtype=torch.float64)
    for label, count in Counter(bank.labels.tolist()).items():
        counts[int(label)] = count
    inv = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
    class_weights = (inv / inv.sum() * 5.0).float()

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 5)
    model.to(device)

    loader = DataLoader(bank, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, persistent_workers=num_workers > 0)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    for epoch in range(epochs):
        model.train()
        running, seen, correct = 0.0, 0, 0
        t0 = time.perf_counter()
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(targets)
            seen += len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
        scheduler.step()
        entry = {
            "epoch": epoch + 1,
            "loss": round(running / max(1, seen), 4),
            "train_accuracy": round(correct / max(1, seen), 4),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        history.append(entry)
        print(f"[CASCADE] epoch {epoch + 1}/{epochs} loss={entry['loss']} acc={entry['train_accuracy']}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "num_classes": 5,
        "crop_size": bank_meta["crop_size"],
        "context": float(np.mean(bank_meta["context_range"])),
        "class_names": CLASS_NAMES,
        "pretrained": "IMAGENET1K_V1",
        "history": history,
        "bank_meta": bank_meta,
    }, out_path)
    print(f"[CASCADE] saved {out_path}")
    return {"history": history, "class_weights": class_weights.tolist()}


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def _load_classifier(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def _batch_probs(model: nn.Module, crops: torch.Tensor, device: torch.device,
                 batch_size: int) -> np.ndarray:
    probs = []
    for start in range(0, len(crops), batch_size):
        chunk = crops[start:start + batch_size].to(device)
        with torch.no_grad():
            logits = model(chunk)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 5), dtype=np.float32)


def _prepare_crops(img: Image.Image, boxes: Sequence[Sequence[float]], context: float,
                   size: int) -> Tuple[torch.Tensor, List[int]]:
    width, height = img.size
    arrays, keep = [], []
    for i, box in enumerate(boxes):
        crop_box = context_crop_box(box, context, width, height)
        if crop_box[2] - crop_box[0] < 4 or crop_box[3] - crop_box[1] < 4:
            continue
        crop = img.crop(tuple(int(round(v)) for v in crop_box)).resize((size, size), Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arrays.append(arr.transpose(2, 0, 1))
        keep.append(i)
    if not arrays:
        return torch.zeros((0, 3, size, size), dtype=torch.float32), keep
    return torch.from_numpy(np.ascontiguousarray(np.stack(arrays))), keep


def apply_cascade(
    checkpoint_path: str,
    preds_path: str,
    split: str,
    data_dir: str,
    output_dir: str,
    mode: str = "rescore",
    max_dets: int = 300,
    batch_size: int = 64,
) -> Dict:
    from pycocotools.coco import COCO

    if mode not in ("rescore", "reclass"):
        raise ValueError(f"Unknown mode: {mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_classifier(checkpoint_path, device)
    context = float(checkpoint["context"])
    size = int(checkpoint["crop_size"])

    data_path = Path(data_dir)
    images_dir = data_path / "images" / split
    coco = COCO(str(data_path / "annotations" / f"instances_{split}.json"))

    dets = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    by_image: Dict[int, List[Dict]] = defaultdict(list)
    for det in dets:
        by_image[int(det["image_id"])].append(det)

    # Keep the strongest `max_dets` boxes per image and category. COCO evaluation
    # truncates to 100 per category anyway, so this only bounds work.
    if max_dets > 0:
        for image_id, items in by_image.items():
            picked: List[Dict] = []
            for category_id in sorted({int(d["category_id"]) for d in items}):
                group = [d for d in items if int(d["category_id"]) == category_id]
                group.sort(key=lambda d: -float(d["score"]))
                picked.extend(group[:max_dets])
            by_image[image_id] = picked

    out: List[Dict] = []
    t0 = time.perf_counter()
    dropped_bg = 0
    changed_label = 0

    for image_id in sorted(by_image.keys()):
        items = by_image[image_id]
        img_info = coco.loadImgs(image_id)[0]
        img = Image.open(images_dir / img_info["file_name"]).convert("RGB")
        xyxy = [_xywh_to_xyxy(d["bbox"]) for d in items]
        crops, kept_idx = _prepare_crops(img, xyxy, context, size)
        probs = _batch_probs(model, crops, device, batch_size)
        del img

        prob_by_idx = {idx: probs[row] for row, idx in enumerate(kept_idx)}
        for i, det in enumerate(items):
            p = prob_by_idx.get(i)
            det_score = float(det["score"])
            if p is None:
                out.append(dict(det))
                continue
            if mode == "rescore":
                label = int(det["category_id"])
                score = det_score * float(p[label]) if 0 <= label < len(p) else det_score
            else:
                if int(np.argmax(p)) == 0:
                    dropped_bg += 1
                    continue
                finding_probs = p[1:5]
                label = int(np.argmax(finding_probs)) + 1
                if label != int(det["category_id"]):
                    changed_label += 1
                score = det_score * float(p[label])
            out.append({
                "image_id": int(det["image_id"]),
                "category_id": label,
                "bbox": [float(v) for v in det["bbox"]],
                "score": float(math.sqrt(max(0.0, min(1.0, score)))),
            })

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dump = out_path / f"predictions_{split}_{mode}.json"
    with open(dump, "w", encoding="utf-8") as f:
        json.dump(out, f)

    meta = {
        "mode": mode,
        "input": preds_path,
        "output": str(dump),
        "context": context,
        "crop_size": size,
        "max_dets_per_image_class": max_dets,
        "num_input": len(dets),
        "num_considered": sum(len(v) for v in by_image.values()),
        "num_output": len(out),
        "dropped_background": dropped_bg,
        "relabelled": changed_label,
        "seconds": round(time.perf_counter() - t0, 1),
        "score_rule": "sqrt(detector_score * classifier_probability)",
    }
    with open(out_path / f"cascade_meta_{split}_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[CASCADE] {mode}: {meta['num_input']} -> {meta['num_output']} dets "
          f"(dropped bg={dropped_bg}, relabelled={changed_label}) -> {dump}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny cascade crop refiner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--data_dir", default="data/dentex")
    p_build.add_argument("--output_dir", default="checkpoints/cascade")
    p_build.add_argument("--size", type=int, default=CROP_SIZE)
    p_build.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train")
    p_train.add_argument("--bank_dir", default="checkpoints/cascade")
    p_train.add_argument("--output", default="checkpoints/cascade/convnext_cascade.pth")
    p_train.add_argument("--epochs", type=int, default=4)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--num_workers", type=int, default=2)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--preds", required=True)
    p_apply.add_argument("--split", choices=["val", "test"], required=True)
    p_apply.add_argument("--data_dir", default="data/dentex")
    p_apply.add_argument("--output_dir", default="preds_cascade")
    p_apply.add_argument("--mode", choices=["rescore", "reclass"], default="rescore")
    p_apply.add_argument("--max_dets", type=int, default=300)

    args = parser.parse_args()
    if args.command == "build":
        build_crop_bank(args.data_dir, args.output_dir, size=args.size, seed=args.seed)
    elif args.command == "train":
        train_cascade(args.bank_dir, args.output, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers)
    else:
        apply_cascade(args.checkpoint, args.preds, args.split, args.data_dir,
                      args.output_dir, mode=args.mode, max_dets=args.max_dets)


if __name__ == "__main__":
    main()
