"""
Generates real sample bundles in samples/ by running the actual inference service
(service.DentalAnalysisService) with the trained weights on raw OPG images.

For each input image it writes:
    <name>_input.png      raw input (unmodified pixels)
    <name>_annotated.png  class-coloured boxes + labels (same renderer as the API)
    <name>_report.json    the exact JSON the /analyze endpoint returns

Usage:
    python generate_samples.py --weights checkpoints/best_model.pth \
        --images "C:/path/val_1.png" "C:/path/val_10.png" --output samples/ [--threshold 0.70]
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from service import DentalAnalysisService

DEFAULT_THRESHOLD = 0.70  # operating point selected by macro-F1 sweep on the validation set


def create_sample_bundles(
    weights: str,
    images: list,
    output: Path,
    threshold: float,
    prefix_start: int = 1,
    names: list | None = None,
):
    output.mkdir(parents=True, exist_ok=True)
    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    service = DentalAnalysisService(
        weights_path=weights,
        config={
            "model": {
                "name": "fasterrcnn_resnet50_fpn_v2",
                "min_size": 800,
                "max_size": 1333,
                "box_score_thresh": 0.001,  # must match training/eval config
            }
        },
        device="cpu",
        results_dir=str(output / "_tmp"),
        confidence_threshold=threshold,
    )
    if service.model is None:
        raise RuntimeError("Model failed to load; refusing to write placeholder samples.")

    if names is not None and len(names) != len(images):
        raise ValueError("--names must have the same number of entries as --images")

    for i, img_path in enumerate(images, start=prefix_start):
        img_path = Path(img_path)
        name = names[i - prefix_start] if names else f"sample_{i:02d}_{img_path.stem}"
        raw = img_path.read_bytes()

        report = service.analyze_bytes(raw, filename=img_path.name)

        # Move annotated image produced by the service to the final name
        tmp_annotated = output / "_tmp" / Path(report["annotated_image_url"]).name
        shutil.move(str(tmp_annotated), str(output / f"{name}_annotated.png"))

        input_path = output / f"{name}_input.png"
        if input_path.resolve() != img_path.resolve():
            Image.open(img_path).convert("RGB").save(input_path, format="PNG", optimize=True)

        report["image_id"] = name
        report["original_image_url"] = f"/samples/{name}_input.png"
        report["annotated_image_url"] = f"/samples/{name}_annotated.png"
        with open(output / f"{name}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[OK] {name}: {len(report['findings'])} findings, {report['inference_ms']} ms -> {report['summary']}")

    shutil.rmtree(output / "_tmp", ignore_errors=True)
    print(f"[DONE] Wrote {len(images)} sample bundles to {output}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="checkpoints/best_model.pth")
    ap.add_argument("--images", nargs="+", required=True, help="Raw OPG image paths (PNG/JPG/DICOM)")
    ap.add_argument("--output", default="samples")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--names", nargs="+", default=None, help="Output sample names (default: sample_NN_<input stem>)")
    args = ap.parse_args()
    create_sample_bundles(
        args.weights, args.images, Path(args.output), args.threshold, names=args.names
    )
