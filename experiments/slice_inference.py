"""
Sliced (tiling-aided) inference for the Viscan OPG detector.

Motivation, measured on this project: DENTEX images are 1,976-3,139 px wide and
976-1,536 px tall. With `min_size=800, max_size=1333` the detector sees them
downscaled by 0.425-0.675x (mean 0.474x), which shrinks small caries and
periapical lesions. Slicing runs the same weights on 800x800 crops taken at
native pixel scale (no downscaling inside a crop), then lets WBF merge the
full-image and per-tile detections.

This script only writes COCO prediction dumps, one file per view group:
    <out>/predictions_<split>_full.json    full-image pass (optionally hflip TTA)
    <out>/predictions_<split>_tiles.json   overlapping native-scale tiles
    <out>/slice_meta_<split>.json          tile grid, timings, device

Fuse / score them with `experiments/ensemble_wbf.py` so that the operating threshold is
still selected on the validation split only.

Sliced inference is reported in the literature to help small objects (SAHI,
arXiv 2202.06934, reports +5.1 to +6.8 AP over full-image inference on
VisDrone/xView with FCOS/VFNet/TOOD). Whether it helps here is NOT known until
the evaluation is run; this script makes no accuracy claim.

Usage:
    python experiments/slice_inference.py --weights checkpoints/best_model.pth \
        --config configs/baseline.yaml --split val \
        --data_dir data/dentex --output_dir preds_slice --views both
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import _flip_xyxy, _nms_per_class, _write_predictions
from model import build_model_from_config, load_model_weights

DetectFn = Callable[[torch.Tensor], Dict[str, np.ndarray]]


def generate_tiles(width: int, height: int, tile: int = 800,
                   overlap: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """Returns xyxy tile windows at native scale, covering the whole image."""
    tile_w = min(tile, width)
    tile_h = min(tile, height)
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def _starts(size: int, win: int) -> List[int]:
        if size <= win:
            return [0]
        starts = list(range(0, size - win + 1, stride))
        if starts[-1] != size - win:
            starts.append(size - win)
        return starts

    windows = []
    seen = set()
    for y0 in _starts(height, tile_h):
        for x0 in _starts(width, tile_w):
            key = (x0, y0, x0 + tile_w, y0 + tile_h)
            if key not in seen:
                seen.add(key)
                windows.append(key)
    return windows


def _clip_boxes(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    clipped = boxes.copy().astype(np.float64)
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, height)
    clipped[:, 2] = np.clip(clipped[:, 2], 0.0, width)
    clipped[:, 3] = np.clip(clipped[:, 3], 0.0, height)
    return clipped


def _detect_view(detect_fn: DetectFn, tensor: torch.Tensor, tta: str = "none",
                 score_floor: float = 0.001, max_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the detector on one view, optionally averaging in a mirrored pass."""
    out = detect_fn(tensor)
    boxes = np.asarray(out["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(out["scores"], dtype=np.float32).reshape(-1)
    labels = np.asarray(out["labels"], dtype=np.int64).reshape(-1)

    if tta == "hflip":
        flipped = detect_fn(torch.flip(tensor, dims=[-1]))
        width = float(tensor.shape[-1])
        fboxes = np.asarray(flipped["boxes"], dtype=np.float32).reshape(-1, 4)
        fscores = np.asarray(flipped["scores"], dtype=np.float32).reshape(-1)
        flabels = np.asarray(flipped["labels"], dtype=np.int64).reshape(-1)

        # Mirror into the original frame. torchvision already reports boxes in
        # the input view's pixel space, so the mirror width is the view width.
        mirrored = fboxes.copy()
        mirrored[:, 0] = width - fboxes[:, 2]
        mirrored[:, 2] = width - fboxes[:, 0]

        boxes_t = torch.from_numpy(np.concatenate([boxes, mirrored]))
        scores_t = torch.from_numpy(np.concatenate([scores, fscores]))
        labels_t = torch.from_numpy(np.concatenate([labels, flabels]))
        boxes_t, scores_t, labels_t = _nms_per_class(
            boxes_t, scores_t, labels_t, iou_thresh=0.50, max_det=max_dets
        )
        boxes, scores, labels = boxes_t.numpy(), scores_t.numpy(), labels_t.numpy()

    if score_floor > 0:
        keep = scores >= score_floor
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if max_dets > 0 and len(boxes) > max_dets:
        keep = np.argsort(-scores)[:max_dets]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    return boxes, scores, labels


def run_split(
    detect_fn: DetectFn,
    image_paths: Sequence[Tuple[int, str]],
    views: str = "both",
    tile: int = 800,
    overlap: float = 0.25,
    tta: str = "none",
    tile_tta: str = "none",
    max_dets_full: int = 300,
    max_dets_tile: int = 100,
    score_floor: float = 0.001,
) -> Tuple[Dict[str, List[Dict]], Dict]:
    """Produces COCO-format detections for the requested view groups."""
    dumps: Dict[str, List[Dict]] = {"full": [], "tiles": []}
    meta = {"num_images": 0, "num_tiles": 0, "ms_per_image": {}, "tile": tile, "overlap": overlap}
    timings = {"full": [], "tiles": []}
    tile_counts = []

    for image_id, path in image_paths:
        img = Image.open(path).convert("RGB")
        width, height = img.size
        meta["num_images"] += 1

        if views in ("full", "both"):
            t0 = time.perf_counter()
            tensor = transforms.functional.to_tensor(img)
            boxes, scores, labels = _detect_view(
                detect_fn, tensor, tta=tta, score_floor=score_floor, max_dets=max_dets_full
            )
            timings["full"].append((time.perf_counter() - t0) * 1000.0)
            _append_dets(dumps["full"], image_id, boxes, scores, labels)

        if views in ("tiles", "both"):
            t0 = time.perf_counter()
            windows = generate_tiles(width, height, tile=tile, overlap=overlap)
            tile_counts.append(len(windows))
            for (x0, y0, x1, y1) in windows:
                crop = img.crop((x0, y0, x1, y1))
                cw, ch = crop.size
                view_tensor = transforms.functional.to_tensor(crop)
                tboxes, tscores, tlabels = _detect_view(
                    detect_fn, view_tensor, tta=tile_tta,
                    score_floor=score_floor, max_dets=max_dets_tile,
                )
                tboxes = _clip_boxes(tboxes, cw, ch)
                tboxes[:, [0, 2]] += x0
                tboxes[:, [1, 3]] += y0
                tboxes = _clip_boxes(tboxes, width, height)
                keep = (tboxes[:, 2] - tboxes[:, 0] >= 1.0) & (tboxes[:, 3] - tboxes[:, 1] >= 1.0)
                _append_dets(dumps["tiles"], image_id, tboxes[keep], tscores[keep], tlabels[keep])
            timings["tiles"].append((time.perf_counter() - t0) * 1000.0)
            meta["num_tiles"] += len(windows)

    for key in ("full", "tiles"):
        if timings[key]:
            meta["ms_per_image"][key] = round(float(np.mean(timings[key])), 2)
    if tile_counts:
        meta["tiles_per_image_min"] = int(min(tile_counts))
        meta["tiles_per_image_max"] = int(max(tile_counts))
        meta["tiles_per_image_mean"] = round(float(np.mean(tile_counts)), 2)
    meta["detections"] = {k: len(v) for k, v in dumps.items()}
    meta["tta_full"] = tta
    meta["tta_tiles"] = tile_tta
    return dumps, meta


def _append_dets(target: List[Dict], image_id: int, boxes: np.ndarray,
                 scores: np.ndarray, labels: np.ndarray) -> None:
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = (float(v) for v in box)
        target.append({
            "image_id": int(image_id),
            "category_id": int(label),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })


def _load_model(config_path: str, weights_path: str, device: torch.device,
                max_dets_per_view: int):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg, pretrained=False)
    load_model_weights(model, weights_path, device=device)
    if hasattr(model, "roi_heads"):
        model.roi_heads.detections_per_img = max_dets_per_view
    elif hasattr(model, "detections_per_img"):
        model.detections_per_img = max_dets_per_view
    return model


def _make_detect_fn(model, device: torch.device) -> DetectFn:
    def detect(tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            out = model([tensor.to(device)])[0]
        return {
            "boxes": out["boxes"].cpu().numpy(),
            "scores": out["scores"].cpu().numpy(),
            "labels": out["labels"].cpu().numpy(),
        }
    return detect


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-image + native-scale tiled inference dumps")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--data_dir", default="data/dentex")
    parser.add_argument("--output_dir", default="preds_slice")
    parser.add_argument("--views", choices=["full", "tiles", "both"], default="both")
    parser.add_argument("--tile", type=int, default=800, help="Tile side in native pixels")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--tta", choices=["none", "hflip"], default="none",
                        help="hflip TTA on the full-image view")
    parser.add_argument("--tile_tta", choices=["none", "hflip"], default="none",
                        help="hflip TTA on tiles (doubles tile cost)")
    parser.add_argument("--max_dets_full", type=int, default=300,
                        help="Detections kept for the full-image view (COCO evaluates 100 per class)")
    parser.add_argument("--max_dets_tile", type=int, default=100,
                        help="Detections kept per tile")
    parser.add_argument("--limit", type=int, default=0, help="Only first N images (smoke test)")
    args = parser.parse_args()

    from pycocotools.coco import COCO

    data_dir = Path(args.data_dir)
    ann_file = data_dir / "annotations" / f"instances_{args.split}.json"
    images_dir = data_dir / "images" / args.split
    coco = COCO(str(ann_file))
    image_paths = [
        (img_id, str(images_dir / coco.loadImgs(img_id)[0]["file_name"]))
        for img_id in sorted(coco.imgs.keys())
    ]
    if args.limit:
        image_paths = image_paths[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SLICE] split={args.split} images={len(image_paths)} device={device} views={args.views}")
    model = _load_model(args.config, args.weights, device,
                        max_dets_per_view=max(args.max_dets_full, args.max_dets_tile))
    detect_fn = _make_detect_fn(model, device)

    dumps, meta = run_split(
        detect_fn, image_paths, views=args.views, tile=args.tile, overlap=args.overlap,
        tta=args.tta, tile_tta=args.tile_tta,
        max_dets_full=args.max_dets_full, max_dets_tile=args.max_dets_tile,
        score_floor=0.001,
    )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if args.views in ("full", "both"):
        _write_predictions(out_path / f"predictions_{args.split}_full.json", dumps["full"])
    if args.views in ("tiles", "both"):
        _write_predictions(out_path / f"predictions_{args.split}_tiles.json", dumps["tiles"])

    meta["device"] = str(device)
    meta["weights"] = args.weights
    meta["timing_note"] = (
        "Wall-clock per image on the stated device, excluding model load and JSON writing. "
        "Not a CPU-latency measurement."
    )
    with open(out_path / f"slice_meta_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SLICE] detections: {meta['detections']} | tiles: {meta.get('num_tiles')}")
    print(f"[SLICE] wrote dumps to {out_path}")


if __name__ == "__main__":
    main()
