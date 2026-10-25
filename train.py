"""
Training pipeline for Viscan OPG.

Fine-tunes Faster R-CNN ResNet-50 FPN V2 on DENTEX dataset:
- Fixed random seed (42)
- Configuration-driven via YAML
- Data augmentation: horizontal flip (p=0.5) + ColorJitter (brightness/contrast ±0.15)
- CosineAnnealingLR scheduler
- AMP (automatic mixed precision) on CUDA
- Training & validation loss tracking
- Validation AP evaluation after every epoch
- Checkpoints saved: best_model.pth and last_model.pth
- Training curves saved as high-resolution PNG plots

License:
- Code: BSD-3-Clause (torchvision)
- Fine-tuned weights: trained on DENTEX (CC BY-NC-SA 4.0)
"""

import os
import sys
import yaml
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import torchvision
    from torchvision import transforms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:
    pass

from model import build_model_from_config, CLASS_NAMES


def set_seed(seed: int = 42):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DentexCocoDataset(Dataset):
    """PyTorch Dataset for COCO-formatted DENTEX data with augmentation."""

    def __init__(self, images_dir: str, annotation_file: str, is_train: bool = False):
        self.images_dir = Path(images_dir)
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self.is_train = is_train
        # Subtle brightness/contrast jitter for training (safe for radiographs)
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15) if is_train else None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.images_dir / img_info["file_name"]

        # Load RGB image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            raw_x_max = float(x) + float(w)
            raw_y_max = float(y) + float(h)
            x_min = max(0.0, float(x))
            y_min = max(0.0, float(y))
            x_max = min(float(width), raw_x_max)
            y_max = min(float(height), raw_y_max)

            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(ann["category_id"]))
                areas.append(float(ann.get("area", w * h)))
                iscrowd.append(int(ann.get("iscrowd", 0)))

        # Train-time augmentation
        if self.is_train:
            # Random horizontal flip (p=0.5)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_boxes = []
                for b in boxes:
                    flipped_boxes.append([width - b[2], b[1], width - b[0], b[3]])
                boxes = flipped_boxes

            # Color jitter (brightness/contrast ±0.15)
            if self.color_jitter is not None:
                img = self.color_jitter(img)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        # Convert PIL to Tensor [0.0, 1.0]
        img_tensor = transforms.functional.to_tensor(img)

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_coco(model, data_loader, coco_gt, device):
    """Runs validation evaluation and computes COCO mAP50."""
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    w = x_max - x_min
                    h = y_max - y_min
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x_min), float(y_min), float(w), float(h)],
                        "score": float(score),
                    })

    if not results:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] is AP @ 0.50:0.95, stats[1] is AP @ 0.50
    ap_50_95 = float(coco_eval.stats[0])
    ap_50 = float(coco_eval.stats[1])
    return ap_50, ap_50_95


def plot_curves(history: Dict[str, List[float]], output_dir: Path):
    """Plots and saves loss and AP progression curves as images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Loss Curve
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Total Loss")
    if "loss_classifier" in history:
        plt.plot(epochs, history["loss_classifier"], "r--", label="Classifier Loss")
    if "loss_box_reg" in history:
        plt.plot(epochs, history["loss_box_reg"], "g--", label="Box Reg Loss")
    plt.title("Training Loss vs. Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # 2. Validation mAP Curve
    plt.figure(figsize=(8, 5))
    val_epochs = range(1, len(history["val_map50"]) + 1)
    plt.plot(val_epochs, history["val_map50"], "g-s", label="Validation mAP@50")
    plt.plot(val_epochs, history["val_map50_95"], "m-^", label="Validation mAP@50:95")
    plt.title("Validation mAP vs. Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("mAP", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "val_ap_curve.png", dpi=300)
    plt.close()


def train(config_path: str, epochs_override: int = None):
    """Main training routine."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using training device: {device}")

    # Build model
    model = build_model_from_config(config, pretrained=True)
    model.to(device)

    # Prepare datasets & loaders
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    data_dir = Path(data_cfg.get("data_dir", "data/dentex"))

    train_dataset = DentexCocoDataset(
        images_dir=str(data_dir / "images" / "train"),
        annotation_file=str(data_dir / "annotations" / "instances_train.json"),
        is_train=True,
    )
    val_dataset = DentexCocoDataset(
        images_dir=str(data_dir / "images" / "val"),
        annotation_file=str(data_dir / "annotations" / "instances_val.json"),
        is_train=False,
    )

    batch_size = train_cfg.get("batch_size", 2)
    num_workers = train_cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Optimizer & Scheduler
    lr = train_cfg.get("lr", 0.005)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
    )

    epochs = train_cfg.get("epochs", 12)
    if epochs_override:
        print(f"[INFO] --epochs override: {epochs} -> {epochs_override} (not the reported protocol)")
        epochs = epochs_override
    scheduler_name = train_cfg.get("lr_scheduler", "CosineAnnealingLR")
    if scheduler_name == "CosineAnnealingLR":
        eta_min_factor = train_cfg.get("lr_eta_min_factor", 0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * eta_min_factor
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("lr_step_size", 5),
            gamma=train_cfg.get("lr_gamma", 0.1),
        )

    # AMP (automatic mixed precision)
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[INFO] AMP enabled: {use_amp}")
    print(f"[INFO] LR scheduler: {scheduler_name}")

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(train_cfg.get("output_dir", "runs/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "loss_classifier": [],
        "loss_box_reg": [],
        "val_map50": [],
        "val_map50_95": [],
    }

    best_map50 = 0.0

    print(f"\n[START] Beginning training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        cls_losses = []
        reg_losses = []

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(losses.item())
            # Faster R-CNN: loss_classifier / loss_box_reg; RetinaNet: classification / bbox_regression
            cls_key = next((k for k in ("loss_classifier", "classification") if k in loss_dict), None)
            reg_key = next((k for k in ("loss_box_reg", "bbox_regression") if k in loss_dict), None)
            if cls_key is not None:
                cls_losses.append(loss_dict[cls_key].item())
            if reg_key is not None:
                reg_losses.append(loss_dict[reg_key].item())

            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch}/{epochs}] Batch [{i+1}/{len(train_loader)}] Loss: {losses.item():.4f} lr: {cur_lr:.6f}")

        lr_scheduler.step()

        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)
        history["loss_classifier"].append(float(np.mean(cls_losses)) if cls_losses else 0.0)
        history["loss_box_reg"].append(float(np.mean(reg_losses)) if reg_losses else 0.0)

        # Validation evaluation
        print(f"\n[EVAL] Running validation evaluation for Epoch {epoch}...")
        val_map50, val_map50_95 = evaluate_coco(model, val_loader, val_dataset.coco, device)
        history["val_map50"].append(val_map50)
        history["val_map50_95"].append(val_map50_95)

        print(f"Epoch {epoch} Result -> Train Loss: {mean_loss:.4f} | Val mAP@50: {val_map50:.4f} | Val mAP@50:95: {val_map50_95:.4f}")

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_map50": val_map50,
            "config": config,
        }

        torch.save(checkpoint_data, checkpoint_dir / "last_model.pth")
        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save(checkpoint_data, checkpoint_dir / "best_model.pth")
            print(f"  [CHECKPOINT] New best validation mAP@50: {best_map50:.4f}. Saved best_model.pth.")

        # Update curves after each epoch
        plot_curves(history, output_dir)

    print(f"\n[DONE] Training complete. Best Val mAP@50: {best_map50:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to config yaml")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count for smoke tests only; omit for the reported protocol",
    )
    args = parser.parse_args()
    train(args.config, epochs_override=args.epochs)
