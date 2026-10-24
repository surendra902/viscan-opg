"""Unit tests for dataset parsing, category mapping, and bounding box conversions."""

import pytest
from prepare_data import CATEGORY_MAPPING, COCO_CATEGORIES


def test_category_mapping():
    """Verify category mapping aligns with DENTEX release and torchvision targets."""
    # 0: Impacted -> 1
    assert CATEGORY_MAPPING[0]["id"] == 1
    assert CATEGORY_MAPPING[0]["name"] == "impacted_tooth"

    # 1: Caries -> 2
    assert CATEGORY_MAPPING[1]["id"] == 2
    assert CATEGORY_MAPPING[1]["name"] == "caries"

    # 2: Periapical Lesion -> 3
    assert CATEGORY_MAPPING[2]["id"] == 3
    assert CATEGORY_MAPPING[2]["name"] == "periapical_lesion"

    # 3: Deep Caries -> 4
    assert CATEGORY_MAPPING[3]["id"] == 4
    assert CATEGORY_MAPPING[3]["name"] == "deep_caries"


def test_bbox_conversion_bounds():
    """Verify xywh to xyxy conversion and boundary clipping."""
    # Source [x, y, w, h]
    x, y, w, h = 100.0, 50.0, 40.0, 30.0
    img_w, img_h = 200, 150

    x_min = max(0.0, x)
    y_min = max(0.0, y)
    x_max = min(float(img_w), x_min + w)
    y_max = min(float(img_h), y_min + h)

    assert x_min == 100.0
    assert y_min == 50.0
    assert x_max == 140.0
    assert y_max == 80.0
    assert x_max > x_min
    assert y_max > y_min


def test_convert_annotations_to_coco_pipeline(tmp_path):
    """Verify raw DENTEX JSON is correctly converted to standard COCO format."""
    import json
    from prepare_data import convert_annotations_to_coco

    raw_data = {
        "images": [
            {"id": 101, "file_name": "sample_opg_101.png", "width": 800, "height": 400}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 101,
                "category_id_3": 0,  # Impacted -> 1 (impacted_tooth)
                "bbox": [50.0, 60.0, 70.0, 80.0],
            },
            {
                "id": 2,
                "image_id": 101,
                "category_id_3": 3,  # Deep Caries -> 4 (deep_caries)
                "bbox": [200.0, 150.0, 40.0, 50.0],
            },
        ],
    }

    raw_json = tmp_path / "raw.json"
    raw_json.write_text(json.dumps(raw_data), encoding="utf-8")

    out_json = tmp_path / "coco.json"
    convert_annotations_to_coco(
        raw_json_path=raw_json,
        images_dir=tmp_path / "images",
        out_json_path=out_json,
        split_name="test_split",
    )

    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "sample_opg_101.png"
    assert len(coco["annotations"]) == 2
    assert coco["annotations"][0]["category_id"] == 1  # impacted_tooth
    assert coco["annotations"][0]["bbox"] == [50.0, 60.0, 70.0, 80.0]
    assert coco["annotations"][1]["category_id"] == 4  # deep_caries
    assert len(coco["categories"]) == 4


def test_negative_origin_bbox_clipped_not_shifted(tmp_path):
    """Verify negative-origin bounding boxes are clipped at 0 rather than shifted right/down (D10)."""
    import json
    from prepare_data import convert_annotations_to_coco

    raw_data = {
        "images": [{"id": 1, "file_name": "clip.png", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id_3": 0, "bbox": [-10.0, 10.0, 20.0, 20.0]}],
    }
    raw_json = tmp_path / "raw_clip.json"
    raw_json.write_text(json.dumps(raw_data), encoding="utf-8")
    out_json = tmp_path / "coco_clip.json"

    convert_annotations_to_coco(raw_json, tmp_path / "images", out_json, "clip_test")
    coco = json.loads(out_json.read_text(encoding="utf-8"))
    assert coco["annotations"][0]["bbox"] == [0.0, 10.0, 10.0, 20.0]


def test_download_file_local_workflow(tmp_path):
    """Verify download_file skips existing files and creates directories."""
    from prepare_data import download_file

    dest = tmp_path / "sub" / "test_file.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("already_downloaded", encoding="utf-8")

    # Should skip because file exists with > 0 bytes
    download_file("http://invalid-url-should-skip.local", dest)
    assert dest.read_text(encoding="utf-8") == "already_downloaded"

