"""Measure CPU forward-pass latency across a set of OPG images.

This times model inference only (image tensor -> detections), matching the
benchmark in evaluate.py. It excludes file decoding, the input gate, annotation
drawing and PNG writes, which are included in the service's inference_ms field.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import _build_and_load


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_images(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        return [path]
    images = sorted(
        p
        for p in path.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and not p.stem.endswith(("_annotated", "_original"))
    )
    return images[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Image file or directory")
    parser.add_argument("--weights", default="checkpoints/best_model.pth")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="Write the full result as JSON here")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    images = find_images(Path(args.images), args.limit)
    if not images:
        raise SystemExit(f"No images found under {args.images}")
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    _, model = _build_and_load(args.config, str(weights), torch.device("cpu"))
    model.eval()

    rows = []
    with torch.no_grad():
        first = TF.to_tensor(Image.open(images[0]).convert("RGB"))
        for _ in range(2):
            model([first])

        for path in images:
            image = TF.to_tensor(Image.open(path).convert("RGB"))
            model([image])
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                model([image])
                samples.append((time.perf_counter() - start) * 1000.0)
            rows.append(
                {
                    "image": path.name,
                    "size": list(Image.open(path).size),
                    "ms": [round(v, 1) for v in samples],
                    "mean_ms": round(statistics.mean(samples), 1),
                }
            )

    values = sorted(v for row in rows for v in row["ms"])
    p95 = values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
    summary = {
        "scope": "model forward pass only; excludes decode, gate, drawing and PNG writes",
        "n_images": len(rows),
        "repeats_per_image": args.repeats,
        "threads": args.threads,
        "mean_ms": round(statistics.mean(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(min(values), 1),
        "max_ms": round(max(values), 1),
        "per_image": rows,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.uname().processor,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
        },
    }

    print(
        f"{len(rows)} images, {args.repeats} runs each, {args.threads} threads: "
        f"mean {summary['mean_ms']:.0f} ms, median {summary['median_ms']:.0f} ms, "
        f"p95 {summary['p95_ms']:.0f} ms, range {summary['min_ms']:.0f}-{summary['max_ms']:.0f} ms"
    )
    for row in rows:
        print(f"  {row['image']:<16} {row['mean_ms']:>8.0f} ms  {row['size'][0]}x{row['size'][1]}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
