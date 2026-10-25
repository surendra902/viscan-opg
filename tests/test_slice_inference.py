"""Unit tests for sliced inference (tiling, offsets, mirror TTA)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from evaluate import _flip_xyxy
from slice_inference import (
    _clip_boxes,
    _detect_view,
    generate_tiles,
    run_split,
)


def _fake_detector(boxes, scores=0.5, labels=1):
    def detect(tensor):
        n = len(boxes)
        return {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(n, 4),
            "scores": np.full((n,), scores, dtype=np.float32),
            "labels": np.full((n,), labels, dtype=np.int64),
        }
    return detect


def test_generate_tiles_covers_image_edges():
    tiles = generate_tiles(2830, 1367, tile=800, overlap=0.25)
    # stride = 600; x starts 0,600,1200,1800 then snapped to 2830-800=2030
    assert [t[0] for t in tiles if t[1] == 0] == [0, 600, 1200, 1800, 2030]
    assert [t[1] for t in tiles if t[0] == 0] == [0, 567]
    assert len(tiles) == 10
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= 2830
        assert 0 <= y1 < y2 <= 1367
        assert x2 - x1 == 800 and y2 - y1 == 800


def test_generate_tiles_small_image_is_single_full_tile():
    assert generate_tiles(500, 400, tile=800, overlap=0.25) == [(0, 0, 500, 400)]


def test_detect_view_hflip_merges_mirror_duplicate():
    # Horizontally centred box: its mirror in a 200 px view is the same box.
    detector = _fake_detector([[80.0, 50.0, 120.0, 90.0]], scores=0.9)
    tensor = torch.zeros(3, 120, 200)
    boxes, scores, labels = _detect_view(detector, tensor, tta="hflip", max_dets=300)
    assert boxes.shape == (1, 4)
    assert np.allclose(boxes[0], [80.0, 50.0, 120.0, 90.0])
    assert labels.tolist() == [1]


def test_flip_xyxy_used_for_mirroring_matches_view_width():
    boxes = torch.tensor([[10.0, 0.0, 30.0, 20.0]])
    assert torch.allclose(_flip_xyxy(boxes, 100.0), torch.tensor([[70.0, 0.0, 90.0, 20.0]]))


def test_clip_boxes_clamps_to_image():
    clipped = _clip_boxes(np.array([[-10.0, -5.0, 120.0, 90.0]]), 100.0, 80.0)
    assert np.allclose(clipped, [[0.0, 0.0, 100.0, 80.0]])


@pytest.fixture()
def synthetic_split(tmp_path):
    width, height = 2000, 1000
    images_dir = tmp_path / "images" / "val"
    images_dir.mkdir(parents=True)
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    Image.new("RGB", (width, height), (30, 30, 30)).save(images_dir / "img_1.jpg")
    coco = {
        "images": [{"id": 1, "file_name": "img_1.jpg", "width": width, "height": height}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 2,
            "bbox": [1000.0, 400.0, 100.0, 50.0], "area": 5000.0, "iscrowd": 0,
        }],
        "categories": [{"id": 1, "name": "impacted_tooth"}, {"id": 2, "name": "caries"}],
    }
    (ann_dir / "instances_val.json").write_text(json.dumps(coco), encoding="utf-8")
    return tmp_path, images_dir / "img_1.jpg"


def test_run_split_offsets_tile_detections_and_writes_full_view(synthetic_split):
    tmp_path, image_path = synthetic_split
    # Every view "detects" a 20x20 box at view-local (10, 10).
    detector = _fake_detector([[10.0, 10.0, 30.0, 30.0]], scores=0.8, labels=2)

    dumps, meta = run_split(
        detector, [(1, str(image_path))], views="both", tile=800, overlap=0.25,
        max_dets_full=300, max_dets_tile=100,
    )

    # 2000x1000 with 800 px tiles and stride 600 -> 3 x 2 = 6 tiles.
    assert meta["num_images"] == 1
    assert meta["num_tiles"] == 6
    assert len(dumps["full"]) == 1
    assert dumps["full"][0]["bbox"] == [10.0, 10.0, 20.0, 20.0]

    expected_origins = {
        (x0, y0) for (x0, y0, _, _) in generate_tiles(2000, 1000, 800, 0.25)
    }
    observed_origins = {(d["bbox"][0] - 10.0, d["bbox"][1] - 10.0) for d in dumps["tiles"]}
    assert len(dumps["tiles"]) == 6
    assert observed_origins == expected_origins
    assert meta["detections"] == {"full": 1, "tiles": 6}


def test_run_split_views_tiles_only(synthetic_split):
    _, image_path = synthetic_split
    detector = _fake_detector([[10.0, 10.0, 30.0, 30.0]], scores=0.8, labels=2)
    dumps, meta = run_split(detector, [(1, str(image_path))], views="tiles", tile=800, overlap=0.25)
    assert dumps["full"] == []
    assert len(dumps["tiles"]) == 6


def test_write_predictions_roundtrip(tmp_path):
    from evaluate import _write_predictions

    dets = [{"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.5}]
    out = tmp_path / "p.json"
    _write_predictions(out, dets)
    assert json.loads(Path(out).read_text()) == dets
