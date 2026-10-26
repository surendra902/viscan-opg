"""Render one held-out OPG under the four inference settings the report compares.

The four panels share the same shipped checkpoint (checkpoints/best_model.pth) and
the same drawing code as the app (service.DentalAnalysisService._draw_annotations).
Only the setting named in each panel title changes, so the differences are the
setting and nothing else:

  A  shipped        800x1333 input, score floor 0.70, one forward pass
  B  floor 0.30     identical to A except the floor, so the suppressed boxes show
  C  hflip TTA      A plus the mirrored pass and class-wise NMS used by evaluate.py
  D  960x1600       the same weights built with configs/inference_variant.yaml sizes,
                   which changes only the input geometry at inference time

Panel D is not the 960x1600 *training* run in the report; that one also changed the
batch size. Here the weights are fixed, so resolution is isolated.

Detections are computed at the native decoded resolution, exactly as the service does,
then the panel image and its boxes are scaled together so the labels stay legible.

Usage:
    python scripts/compare_model_outputs.py --image samples/sample_02_val_10_input.png
    python scripts/compare_model_outputs.py --image samples/sample_04_val_14_input.png \
        --output assets/eval/model_output_variants.png
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fdi  # noqa: E402
from evaluate import _build_and_load, _flip_xyxy, _nms_per_class  # noqa: E402
from model import CLASS_NAMES  # noqa: E402
from service import DentalAnalysisService  # noqa: E402

PANEL_GAP = 16
LABEL_BAND = 62


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _run(model: torch.nn.Module, tensor: torch.Tensor) -> Tuple[Any, Any, Any]:
    with torch.no_grad():
        out = model([tensor])[0]
    return out["boxes"], out["scores"], out["labels"]


def detect(model: torch.nn.Module, tensor: torch.Tensor, threshold: float, tta: bool) -> List[Dict[str, Any]]:
    """Boxes above the floor, in the same shape the service puts into findings."""
    width, height = tensor.shape[-1], tensor.shape[-2]
    boxes, scores, labels = _run(model, tensor)
    if tta:
        flipped_boxes, flipped_scores, flipped_labels = _run(model, torch.flip(tensor, dims=[-1]))
        boxes = torch.cat([boxes, _flip_xyxy(flipped_boxes, float(width))])
        scores = torch.cat([scores, flipped_scores])
        labels = torch.cat([labels, flipped_labels])
        boxes, scores, labels = _nms_per_class(boxes, scores, labels)

    table = fdi.load_table()
    findings: List[Dict[str, Any]] = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        if score < threshold:
            continue
        class_name = CLASS_NAMES.get(int(label), f"finding_{int(label)}")
        if class_name == "background":
            continue
        x_min = max(0.0, min(float(width), box[0]))
        y_min = max(0.0, min(float(height), box[1]))
        x_max = max(0.0, min(float(width), box[2]))
        y_max = max(0.0, min(float(height), box[3]))
        if x_max <= x_min or y_max <= y_min:
            continue
        clipped = [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)]
        guess = fdi.estimate_fdi(clipped, int(width), int(height), table)
        findings.append(
            {
                "class": class_name,
                "confidence": round(float(score), 2),
                "tooth_fdi": guess["fdi"] if guess else None,
                "bbox": clipped,
            }
        )
    findings.sort(key=lambda item: item["confidence"], reverse=True)
    return findings


def _scaled(findings: List[Dict[str, Any]], factor: float) -> List[Dict[str, Any]]:
    out = []
    for item in findings:
        copy = dict(item)
        copy["bbox"] = [round(value * factor, 1) for value in item["bbox"]]
        out.append(copy)
    return out


def panel(service: DentalAnalysisService, image: Image.Image, findings: List[Dict[str, Any]],
          title: str, note: str, panel_width: int) -> Image.Image:
    factor = panel_width / image.width
    size = (panel_width, max(1, round(image.height * factor)))
    small = image.resize(size, Image.LANCZOS)
    drawn = service._draw_annotations(small, _scaled(findings, factor))

    canvas = Image.new("RGB", (panel_width, size[1] + LABEL_BAND), "#ffffff")
    canvas.paste(drawn, (0, LABEL_BAND))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="#111111", font=_font(24))
    draw.text((8, 34), note, fill="#444444", font=_font(18))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="samples/sample_02_val_10_input.png")
    parser.add_argument("--weights", default="checkpoints/best_model.pth")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--variant-config", default="configs/inference_variant.yaml")
    parser.add_argument("--output", default="assets/eval/model_output_variants.png")
    parser.add_argument("--json", default="assets/eval/model_output_variants.json")
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_file():
        raise SystemExit(f"No image at {path}")
    torch.set_num_threads(args.threads)

    cfg, model = _build_and_load(args.config, args.weights, torch.device("cpu"))
    model.eval()
    vcfg, variant = _build_and_load(args.variant_config, args.weights, torch.device("cpu"))
    variant.eval()

    with tempfile.TemporaryDirectory() as tmp:
        service = DentalAnalysisService(weights_path=None, device="cpu", results_dir=tmp)

    image = Image.open(path).convert("RGB")
    tensor = transforms.functional.to_tensor(image)

    shipped = f"min {cfg['model']['min_size']} / max {cfg['model']['max_size']}"
    bigger = f"min {vcfg['model']['min_size']} / max {vcfg['model']['max_size']}"
    specs = [
        ("A  Shipped: floor 0.70, one pass", f"{shipped} input", model, shipped, 0.70, False),
        ("B  Same weights, floor 0.30", "what the 0.70 floor hides", model, shipped, 0.30, False),
        ("C  Shipped floor + hflip TTA", "two passes merged by class-wise NMS", model, shipped, 0.70, True),
        ("D  Same weights, larger input", f"{bigger}; weights unchanged", variant, bigger, 0.70, False),
    ]

    results: List[Dict[str, Any]] = []
    panels: List[Image.Image] = []
    for title, note, net, input_size, threshold, tta in specs:
        start = time.perf_counter()
        findings = detect(net, tensor, threshold, tta)
        elapsed = round((time.perf_counter() - start) * 1000.0, 1)
        tally: Dict[str, int] = {}
        for item in findings:
            tally[item["class"]] = tally.get(item["class"], 0) + 1
        counts = ", ".join(f"{value} {key}" for key, value in sorted(tally.items())) or "no boxes"
        results.append(
            {
                "panel": title.split("  ", 1)[1],
                "input": input_size,
                "score_floor": threshold,
                "tta": "hflip" if tta else "none",
                "config": Path(args.config if net is model else args.variant_config).name,
                "n_findings": len(findings),
                "per_class": tally,
                "pass_ms_cpu": elapsed,
                "findings": findings,
            }
        )
        panels.append(panel(service, image, findings, title, f"{counts} · {elapsed:.0f} ms", args.panel_width))
        print(f"{title:36} {len(findings):>3} boxes  {elapsed:>8.0f} ms  {counts}")

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    width = max(p.width for p in panels) * columns + PANEL_GAP
    height = max(p.height for p in panels) * rows + PANEL_GAP
    canvas = Image.new("RGB", (width, height), "#f4f4f4")
    for index, tile in enumerate(panels):
        x = (index % columns) * (tile.width + PANEL_GAP)
        y = (index // columns) * (tile.height + PANEL_GAP)
        canvas.paste(tile, (x, y))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="PNG")
    print(f"wrote {out} ({canvas.width}x{canvas.height})")

    payload = {
        "image": str(path),
        "image_size": [image.width, image.height],
        "weights": args.weights,
        "threads": args.threads,
        "note": (
            "Panels share checkpoints/best_model.pth and the app's own drawing code; only the named "
            "setting differs. pass_ms_cpu is a timed wall on this machine while other work ran, "
            "so compare the four numbers with each other, not with an unloaded session."
        ),
        "panels": results,
    }
    json_out = Path(args.json)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {json_out}")


if __name__ == "__main__":
    main()
