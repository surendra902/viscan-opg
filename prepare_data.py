"""
DENTEX Dataset Downloader, Validator, and COCO-Format Converter.

Official splits:
- Train: 705 images, 3,529 boxes across 4 finding classes
- Val: 50 images, 182 boxes
- Test: 250 images

Mapping (category_id_3 -> target_id):
- 0: Impacted -> 1 (impacted_tooth)
- 1: Caries -> 2 (caries)
- 2: Periapical Lesion -> 3 (periapical_lesion)
- 3: Deep Caries -> 4 (deep_caries)
Background: 0

Key join rule: Match annotations[].image_id -> images[].id -> images[].file_name.
Never infer filename from image ID.
"""

import os
import sys
import json
import zipfile
import urllib.request
from typing import Dict, Any, List, Tuple
from pathlib import Path

CATEGORY_MAPPING = {
    0: {"id": 1, "name": "impacted_tooth", "src_name": "Impacted"},
    1: {"id": 2, "name": "caries", "src_name": "Caries"},
    2: {"id": 3, "name": "periapical_lesion", "src_name": "Periapical Lesion"},
    3: {"id": 4, "name": "deep_caries", "src_name": "Deep Caries"},
}

COCO_CATEGORIES = [
    {"id": 1, "name": "impacted_tooth", "supercategory": "finding"},
    {"id": 2, "name": "caries", "supercategory": "finding"},
    {"id": 3, "name": "periapical_lesion", "supercategory": "finding"},
    {"id": 4, "name": "deep_caries", "supercategory": "finding"},
]

HF_BASE_URL = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/7b27ccc8e342dcb774f69adc6ca5e6c09fefce93/DENTEX"


def download_file(url: str, dest_path: Path, desc: str = ""):
    """Downloads a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[SKIP] {dest_path.name} already exists ({dest_path.stat().st_size} bytes).")
        return

    print(f"[DOWNLOAD] Downloading {desc or dest_path.name} from {url}...")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  Progress: {percent}% ({count * block_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print("\n  [DONE] Download complete.")


def extract_zip(zip_path: Path, extract_dir: Path, members_filter=None):
    """Extracts zip archive, optionally filtering members."""
    print(f"[EXTRACT] Extracting {zip_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if members_filter:
            members = [m for m in members if members_filter(m)]
        for member in members:
            # Skip checkpoint files
            if ".ipynb_checkpoints" in member:
                continue
            zf.extract(member, extract_dir)
    print(f"  [DONE] Extracted {len(members)} files.")


def convert_annotations_to_coco(
    raw_json_path: Path,
    images_dir: Path,
    out_json_path: Path,
    split_name: str,
) -> Dict[str, Any]:
    """
    Parses DENTEX raw JSON, validates categories and bounding boxes,
    matches image_id to images[].file_name, and exports standard COCO json.
    """
    print(f"[CONVERT] Converting {split_name} annotations from {raw_json_path}...")
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build image map: id -> image_dict
    image_map = {}
    for img in data.get("images", []):
        img_id = img["id"]
        fname = os.path.basename(img["file_name"])
        # Check if file exists in images_dir
        image_map[img_id] = {
            "id": img_id,
            "file_name": fname,
            "width": img.get("width", 0),
            "height": img.get("height", 0),
        }

    coco_images = []
    coco_annotations = []
    class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    ann_id_counter = 1

    valid_image_ids = set()

    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_map:
            continue

        # Extract disease diagnosis category (category_id_3 in DENTEX)
        cat_id_3 = ann.get("category_id_3")
        if cat_id_3 is None or cat_id_3 not in CATEGORY_MAPPING:
            continue

        target_cat_id = CATEGORY_MAPPING[cat_id_3]["id"]

        # Parse bbox [x, y, w, h]
        bbox_raw = ann.get("bbox", [])
        if len(bbox_raw) != 4:
            continue

        x, y, w, h = [float(v) for v in bbox_raw]
        if w <= 0 or h <= 0:
            continue

        # Convert to xyxy for bounds verification
        x_min = max(0.0, x)
        y_min = max(0.0, y)
        img_w = image_map[img_id]["width"]
        img_h = image_map[img_id]["height"]

        if img_w > 0:
            x_max = min(float(img_w), x_min + w)
        else:
            x_max = x_min + w

        if img_h > 0:
            y_max = min(float(img_h), y_min + h)
        else:
            y_max = y_min + h

        final_w = x_max - x_min
        final_h = y_max - y_min
        if final_w <= 1 or final_h <= 1:
            continue

        coco_annotations.append({
            "id": ann_id_counter,
            "image_id": img_id,
            "category_id": target_cat_id,
            "bbox": [x_min, y_min, final_w, final_h],
            "area": final_w * final_h,
            "iscrowd": 0,
        })
        ann_id_counter += 1
        class_counts[target_cat_id] += 1
        valid_image_ids.add(img_id)

    # Filter images to only those belonging to this split
    for img_id, img_info in image_map.items():
        coco_images.append(img_info)

    coco_output = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": COCO_CATEGORIES,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)

    print(f"  [SUMMARY] {split_name.upper()}: {len(coco_images)} images, {len(coco_annotations)} annotations.")
    for cid, count in class_counts.items():
        print(f"    Class '{CATEGORY_MAPPING[cid-1]['name']}': {count} boxes")

    return {
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "counts": class_counts,
    }


def prepare_dataset(dest_root: str = "data/dentex"):
    """Full preparation workflow for DENTEX data."""
    root = Path(dest_root)
    zips_dir = root / "zips"
    raw_dir = root / "raw"
    images_dir = root / "images"
    ann_dir = root / "annotations"

    for d in [zips_dir, raw_dir, images_dir, ann_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Download Validation Annotations JSON and Archives
    val_json_url = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/validation_triple.json"
    val_raw_json = raw_dir / "val.json"
    download_file(val_json_url, val_raw_json, "Validation Annotations JSON")

    # 2. Download Training ZIP & Validation ZIP
    train_zip = zips_dir / "training_data.zip"
    val_zip = zips_dir / "validation_data.zip"
    test_zip = zips_dir / "test_data.zip"

    download_file(f"{HF_BASE_URL}/training_data.zip", train_zip, "Training Images ZIP")
    download_file(f"{HF_BASE_URL}/validation_data.zip", val_zip, "Validation Images ZIP")
    download_file(f"{HF_BASE_URL}/test_data.zip", test_zip, "Test Images ZIP")

    # 3. Extract diagnosis-relevant images and training JSON from zip
    extract_zip(train_zip, root / "extracted_train", members_filter=lambda m: "quadrant-enumeration-disease" in m)
    extract_zip(val_zip, root / "extracted_val")
    extract_zip(test_zip, root / "extracted_test")

    train_raw_json = root / "extracted_train" / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"

    # 4. Convert and reconcile COCO format
    convert_annotations_to_coco(
        train_raw_json,
        images_dir / "train",
        ann_dir / "instances_train.json",
        "train",
    )
    convert_annotations_to_coco(
        val_raw_json,
        images_dir / "val",
        ann_dir / "instances_val.json",
        "val",
    )

    print("\n[COMPLETE] DENTEX dataset preparation completed successfully.")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dentex"
    prepare_dataset(out_dir)
