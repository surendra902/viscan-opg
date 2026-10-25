"""Unit tests for the cascade crop refiner: geometry, crop bank, apply modes."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import cascade_refine as cr


def test_context_crop_box_expands_around_center():
    box = [100.0, 100.0, 200.0, 200.0]
    assert cr.context_crop_box(box, 2.0, 1000, 1000) == (50.0, 50.0, 250.0, 250.0)


def test_context_crop_box_clips_and_shifts():
    box = [0.0, 0.0, 100.0, 100.0]
    x1, y1, x2, y2 = cr.context_crop_box(box, 2.0, 150, 150)
    assert (x1, y1) == (0.0, 0.0)
    assert (x2, y2) == (150.0, 150.0)

    x1, y1, x2, y2 = cr.context_crop_box([50.0, 50.0, 100.0, 100.0], 1.0, 1000, 1000, shift=(1.0, 0.0))
    assert x1 == 100.0 and x2 == 150.0


def test_iou_xyxy_reference_values():
    assert cr.iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert cr.iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert cr.iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3, rel=1e-6)


@pytest.fixture()
def tiny_dataset(tmp_path):
    data_dir = tmp_path / "dentex"
    images_dir = data_dir / "images" / "train"
    ann_dir = data_dir / "annotations"
    images_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    images, anns = [], []
    ann_id = 1
    for i in range(3):
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 255, size=(300, 500, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images_dir / f"img_{i}.jpg")
        images.append({"id": i + 1, "file_name": f"img_{i}.jpg", "width": 500, "height": 300})
        for j, cat in enumerate([1, 2, 3]):
            anns.append({
                "id": ann_id, "image_id": i + 1, "category_id": cat,
                "bbox": [40.0 + 120 * j, 60.0, 60.0, 40.0], "area": 2400.0, "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"},
                       {"id": 3, "name": "c"}, {"id": 4, "name": "d"}],
    }
    (ann_dir / "instances_train.json").write_text(json.dumps(coco), encoding="utf-8")
    return data_dir


def test_build_crop_bank_counts_and_shapes(tiny_dataset, tmp_path):
    out = tmp_path / "bank"
    meta = cr.build_crop_bank(str(tiny_dataset), str(out), size=64, seed=7)

    crops = np.load(out / "crops_uint8.npy", mmap_mode="r")
    labels = np.load(out / "labels.npy")
    assert crops.shape == (meta["num_samples"], 64, 64, 3)
    assert crops.shape[0] == labels.shape[0]
    # 9 positives (3 images x 3 boxes) + 9 negatives (1:1 ratio).
    assert meta["num_samples"] == 18
    assert meta["counts"]["background"] == 9
    assert meta["counts"]["caries"] == 3


def _write_random_checkpoint(path: Path, size: int = 64, context: float = 1.3) -> None:
    model = cr.convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 5)
    torch.save({
        "model": model.state_dict(), "num_classes": 5, "crop_size": size,
        "context": context, "class_names": cr.CLASS_NAMES,
        "history": [], "bank_meta": {},
    }, path)


def _split_preds(tmp_path, with_dets=True):
    split_dir = tmp_path / "d"
    (split_dir / "images" / "val").mkdir(parents=True)
    (split_dir / "annotations").mkdir(parents=True)
    Image.new("RGB", (500, 300), (50, 50, 50)).save(split_dir / "images" / "val" / "i.jpg")
    coco = {
        "images": [{"id": 1, "file_name": "i.jpg", "width": 500, "height": 300}],
        "annotations": [],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    }
    (split_dir / "annotations" / "instances_val.json").write_text(json.dumps(coco))
    dets = [
        {"image_id": 1, "category_id": 2, "bbox": [100.0, 60.0, 60.0, 40.0], "score": 0.6},
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 30.0, 30.0], "score": 0.4},
    ] if with_dets else []
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps(dets))
    return split_dir, preds, dets


def test_apply_cascade_rescore_preserves_boxes_and_orders_scores(tmp_path):
    ckpt = tmp_path / "c.pth"
    _write_random_checkpoint(ckpt)
    split_dir, preds, dets = _split_preds(tmp_path)

    meta = cr.apply_cascade(str(ckpt), str(preds), "val", str(split_dir),
                            str(tmp_path / "out"), mode="rescore")

    out = json.loads((tmp_path / "out" / "predictions_val_rescore.json").read_text())
    assert meta["num_input"] == 2 and meta["num_output"] == 2
    # Per-class trimming groups the dump by category, so compare order-insensitively.
    before = sorted(dets, key=lambda d: d["bbox"])
    after = sorted(out, key=lambda d: d["bbox"])
    assert [d["bbox"] for d in after] == [d["bbox"] for d in before]
    assert [d["category_id"] for d in after] == [d["category_id"] for d in before]
    for b, a in zip(before, after):
        # score = sqrt(det_score * p), p in (0, 1]
        assert 0.0 < a["score"] <= np.sqrt(b["score"]) + 1e-6


def test_apply_cascade_reclass_can_drop_background(tmp_path, monkeypatch):
    ckpt = tmp_path / "c.pth"
    _write_random_checkpoint(ckpt)
    split_dir, preds, _ = _split_preds(tmp_path)

    class BackgroundOnly(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros((x.shape[0], 5), dtype=torch.float32)
            logits[:, 0] = 10.0
            return logits

    module = BackgroundOnly()
    monkeypatch.setattr(cr, "_load_classifier", lambda path, device: (module, {
        "context": 1.3, "crop_size": 64,
    }))

    meta = cr.apply_cascade(str(ckpt), str(preds), "val", str(split_dir),
                            str(tmp_path / "out"), mode="reclass")
    out = json.loads((tmp_path / "out" / "predictions_val_reclass.json").read_text())
    assert meta["dropped_background"] == meta["num_input"]
    assert out == []


def test_apply_cascade_handles_empty_dump(tmp_path):
    ckpt = tmp_path / "c.pth"
    _write_random_checkpoint(ckpt)
    split_dir, preds, _ = _split_preds(tmp_path, with_dets=False)
    meta = cr.apply_cascade(str(ckpt), str(preds), "val", str(split_dir), str(tmp_path / "out"))
    assert meta["num_output"] == 0
