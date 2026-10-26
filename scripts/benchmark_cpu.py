"""Measure CPU forward-pass latency across a set of OPG images.

This times model inference only (image tensor -> detections), matching the
benchmark in evaluate.py. It excludes file decoding, the input gate, annotation
drawing and PNG writes, which are included in the service's inference_ms field.

With --tta hflip the timed block is the one evaluate.py runs: a forward pass on
the image, a forward pass on its horizontal mirror, and class-wise NMS over the
merged boxes.

With --tta both the plain pass and the TTA pass are timed as paired runs of the
same image inside one process, so the TTA ratio is not distorted by drifting
between separate runs.

With --service-url the script loads no model and instead posts images to a
running server, timing the full /analyze request (decode, input gate, model
pass, annotation drawing and two PNG writes) as wall-clock and reading the
server's own inference_ms.

With --full-request the same whole-pipeline timing happens in this process, with
no HTTP server: the service object is built here and each image goes through
decode, gate, model, drawing and the two PNG writes. Both modes also record the
peak resident set of the process that did the work, sampled in a background
thread, so the memory figure is measured rather than estimated.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import _build_and_load, _flip_xyxy, _nms_per_class

try:
    import psutil
except ImportError:  # memory reporting degrades to None instead of failing the run
    psutil = None


class RssSampler:
    """Sample this process's resident set in a background thread.

    The peak is the largest sample seen, so a spike that starts and ends inside
    one interval is missed; the interval is recorded in the output with it.
    """

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "RssSampler":
        if psutil is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                self.samples.append(proc.memory_info().rss)
            except psutil.Error:
                break
            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, object]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {"memory_measured": False}
        mb = 1024 * 1024
        return {
            "memory_measured": True,
            "rss_sampling_s": self.interval_s,
            "n_rss_samples": len(self.samples),
            "peak_rss_mb": round(max(self.samples) / mb, 1),
            "final_rss_mb": round(self.samples[-1] / mb, 1),
        }


def memory_context() -> dict[str, object]:
    """Machine RAM and the process footprint at the point of the call."""
    out: dict[str, object] = {}
    if psutil is not None:
        out["machine_total_ram_mb"] = round(psutil.virtual_memory().total / (1024 * 1024))
        out["rss_now_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    return out


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


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "mean_ms": round(statistics.mean(ordered), 1),
        "median_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(min(ordered), 1),
        "max_ms": round(max(ordered), 1),
    }


def run_live(images: list[Path], base: str, repeats: int, timeout: float, output: str | None) -> None:
    """Time full /analyze requests against a running server; loads no model here."""
    import httpx

    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{base}/health")
        if health.status_code != 200 or not health.json().get("model_loaded"):
            raise SystemExit(f"{base}/health reports no loaded model; start the server first.")

        first = images[0].read_bytes()
        client.post(f"{base}/analyze", files={"file": (images[0].name, first, "image/png")})

        rows = []
        for path in images:
            data = path.read_bytes()
            for _ in range(repeats):
                start = time.perf_counter()
                response = client.post(f"{base}/analyze", files={"file": (path.name, data, "image/png")})
                wall_ms = (time.perf_counter() - start) * 1000.0
                body = response.json() if response.status_code == 200 else {}
                rows.append(
                    {
                        "image": path.name,
                        "http_status": response.status_code,
                        "server_inference_ms": body.get("inference_ms"),
                        "n_findings": len(body.get("findings", [])),
                        "wall_ms": round(wall_ms, 1),
                    }
                )

    summary = {
        "scope": (
            "full /analyze request through a running server: file decode, input gate, model pass, "
            "annotation drawing and two PNG writes. wall_ms adds the upload and JSON parse."
        ),
        "service_url": base,
        "server_threads": "torch default (the app sets no thread count)",
        "n_images": len(rows),
        "repeats_per_image": repeats,
        "per_request": rows,
        "server_inference": stats([r["server_inference_ms"] for r in rows if r["server_inference_ms"]]),
        "wall": stats([r["wall_ms"] for r in rows]),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.uname().processor,
            "python": platform.python_version(),
            "httpx": httpx.__version__,
        },
    }
    print(
        f"{len(rows)} requests, {repeats} per image: server inference_ms mean "
        f"{summary['server_inference']['mean_ms']:.0f} ms, range "
        f"{summary['server_inference']['min_ms']:.0f}-{summary['server_inference']['max_ms']:.0f} ms; "
        f"wall-clock mean {summary['wall']['mean_ms']:.0f} ms"
    )
    for row in rows:
        print(f"  {row['image']:<28} {row['server_inference_ms']:>8} ms  wall {row['wall_ms']:>8} ms")
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")


def run_full_request(images: list[Path], weights: Path, config: str, threads: int, repeats: int, output: str | None) -> None:
    """Time analyze_bytes() end to end in this process: decode, gate, model, drawing, PNG writes."""
    import yaml
    from service import DentalAnalysisService

    cfg = None
    config_path = Path(config)
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    threshold = float((cfg or {}).get("evaluation", {}).get("ui_confidence_threshold", 0.70))

    torch.set_num_threads(threads)
    with tempfile.TemporaryDirectory() as tmp:
        service = DentalAnalysisService(
            weights_path=str(weights),
            config=cfg,
            device="cpu",
            results_dir=tmp,
            confidence_threshold=threshold,
        )
        if service.model is None:
            raise SystemExit(f"No model loaded from {weights}; nothing to time.")
        loaded = memory_context()
        first = images[0].read_bytes()
        service.analyze_bytes(first, images[0].name)

        rows = []
        sampler = RssSampler().start()
        for path in images:
            data = path.read_bytes()
            for _ in range(repeats):
                start = time.perf_counter()
                body = service.analyze_bytes(data, path.name)
                rows.append(
                    {
                        "image": path.name,
                        "server_inference_ms": body.get("inference_ms"),
                        "n_findings": len(body.get("findings", [])),
                        "wall_ms": round((time.perf_counter() - start) * 1000.0, 1),
                    }
                )
        memory = sampler.stop()

    summary = {
        "scope": (
            "analyze_bytes() in this process, no HTTP server: file decode, input gate, model pass, "
            "annotation drawing and two PNG writes. wall_ms is the same work without a socket. "
            f"peak_rss_mb is the largest {memory.get('rss_sampling_s', 0)} s sample of this process, "
            "which holds the checkpoint and one image at a time."
        ),
        "mode": "in-process full request",
        "threads": threads,
        "confidence_threshold": threshold,
        "n_images": len(rows),
        "repeats_per_image": repeats,
        "per_request": rows,
        "service_inference": stats([r["server_inference_ms"] for r in rows if r["server_inference_ms"]]),
        "wall": stats([r["wall_ms"] for r in rows]),
        "memory_after_weight_load": loaded,
        "memory": memory,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.uname().processor,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "psutil": psutil.__version__ if psutil else None,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
        },
    }
    print(
        f"{len(rows)} requests, {repeats} per image, {threads} threads: inference_ms mean "
        f"{summary['service_inference']['mean_ms']:.0f}, range "
        f"{summary['service_inference']['min_ms']:.0f}-{summary['service_inference']['max_ms']:.0f}; "
        f"wall mean {summary['wall']['mean_ms']:.0f} ms; "
        f"peak RSS {memory.get('peak_rss_mb', 'n/a')} MB of "
        f"{loaded.get('machine_total_ram_mb', 'n/a')} MB RAM"
    )
    for row in rows:
        print(f"  {row['image']:<28} {row['server_inference_ms']:>8} ms  wall {row['wall_ms']:>8} ms")
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Image file or directory")
    parser.add_argument("--weights", default="checkpoints/best_model.pth")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--tta",
        choices=["none", "hflip", "both"],
        default="none",
        help="hflip adds a mirrored forward pass and the NMS merge used by evaluate.py; "
        "both times plain and hflip passes as paired runs in one process",
    )
    parser.add_argument("--output", help="Write the full result as JSON here")
    parser.add_argument(
        "--service-url",
        help="Time full /analyze requests against this running server instead of loading a model",
    )
    parser.add_argument(
        "--full-request",
        action="store_true",
        help="Time service.analyze_bytes() end to end in this process (decode, gate, model, drawing, PNG writes)",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout for --service-url")
    args = parser.parse_args()

    images = find_images(Path(args.images), args.limit)
    if not images:
        raise SystemExit(f"No images found under {args.images}")
    if args.service_url:
        run_live(images, args.service_url.rstrip("/"), args.repeats, args.timeout, args.output)
        return
    if args.full_request:
        run_full_request(images, Path(args.weights), args.config, args.threads, args.repeats, args.output)
        return

    torch.set_num_threads(args.threads)
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    _, model = _build_and_load(args.config, str(weights), torch.device("cpu"))
    model.eval()
    memory_at_load = memory_context()

    def plain_pass(image: torch.Tensor) -> None:
        model([image])

    def hflip_pass(image: torch.Tensor) -> None:
        width = float(image.shape[-1])
        out = model([image])[0]
        flipped_out = model([torch.flip(image, dims=[-1])])[0]
        boxes = torch.cat([out["boxes"], _flip_xyxy(flipped_out["boxes"], width)])
        scores = torch.cat([out["scores"], flipped_out["scores"]])
        labels = torch.cat([out["labels"], flipped_out["labels"]])
        _nms_per_class(boxes, scores, labels)

    passes = {
        "none": [plain_pass],
        "hflip": [hflip_pass],
        "both": [plain_pass, hflip_pass],
    }[args.tta]

    rows = []
    sampler = RssSampler().start()
    with torch.no_grad():
        first = TF.to_tensor(Image.open(images[0]).convert("RGB"))
        for fn in passes:
            for _ in range(2):
                fn(first)

        for path in images:
            image = TF.to_tensor(Image.open(path).convert("RGB"))
            for fn in passes:
                fn(image)
            samples: dict[str, list[float]] = {fn.__name__: [] for fn in passes}
            for repeat in range(args.repeats):
                # Alternate which pass runs first so neither always sees the colder cache.
                order = passes if repeat % 2 == 0 else list(reversed(passes))
                for fn in order:
                    start = time.perf_counter()
                    fn(image)
                    samples[fn.__name__].append((time.perf_counter() - start) * 1000.0)
            row = {
                "image": path.name,
                "size": list(Image.open(path).size),
                **{name: [round(v, 1) for v in values] for name, values in samples.items()},
                **{
                    f"mean_{name}": round(statistics.mean(values), 1)
                    for name, values in samples.items()
                },
            }
            if args.tta == "both":
                row["ratio"] = round(
                    statistics.mean(samples["hflip_pass"]) / statistics.mean(samples["plain_pass"]), 3
                )
            else:
                name = passes[0].__name__
                row["ms"] = row[name]
                row["mean_ms"] = row[f"mean_{name}"]
            rows.append(row)

    memory = sampler.stop()

    scope = {
        "none": "model forward pass only; excludes decode, gate, drawing and PNG writes",
        "hflip": "hflip TTA: two forward passes plus class-wise NMS; excludes decode, gate, drawing and PNG writes",
        "both": "paired in one process: plain pass and hflip TTA pass of the same image, order alternating",
    }[args.tta]
    summary = {
        "scope": scope,
        "tta": args.tta,
        "n_images": len(rows),
        "repeats_per_image": args.repeats,
        "threads": args.threads,
        "per_image": rows,
        "memory_after_weight_load": memory_at_load,
        "memory": memory,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.uname().processor,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "psutil": psutil.__version__ if psutil else None,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
        },
    }
    if args.tta == "both":
        plain = [v for row in rows for v in row["plain_pass"]]
        hflip = [v for row in rows for v in row["hflip_pass"]]
        ratios = [row["ratio"] for row in rows]
        summary["plain"] = stats(plain)
        summary["hflip"] = stats(hflip)
        summary["ratio"] = {
            "aggregate": round(statistics.mean(hflip) / statistics.mean(plain), 3),
            "per_image_mean": round(statistics.mean(ratios), 3),
            "per_image_median": round(statistics.median(ratios), 3),
            "per_image_min": round(min(ratios), 3),
            "per_image_max": round(max(ratios), 3),
        }
        print(
            f"{len(rows)} images, {args.repeats} paired runs each, {args.threads} threads: "
            f"plain mean {summary['plain']['mean_ms']:.0f} ms / median {summary['plain']['median_ms']:.0f} ms, "
            f"hflip mean {summary['hflip']['mean_ms']:.0f} ms / median {summary['hflip']['median_ms']:.0f} ms, "
            f"ratio aggregate {summary['ratio']['aggregate']} "
            f"(per-image median {summary['ratio']['per_image_median']}, "
            f"range {summary['ratio']['per_image_min']}-{summary['ratio']['per_image_max']})"
        )
        for row in rows:
            print(
                f"  {row['image']:<28} plain {row['mean_plain_pass']:>7.0f} ms  "
                f"hflip {row['mean_hflip_pass']:>7.0f} ms  x{row['ratio']:<6} "
                f"{row['size'][0]}x{row['size'][1]}"
            )
    else:
        values = sorted(v for row in rows for v in row["ms"])
        summary.update(stats(values))
        print(
            f"{len(rows)} images, {args.repeats} runs each, {args.threads} threads, tta={args.tta}: "
            f"mean {summary['mean_ms']:.0f} ms, median {summary['median_ms']:.0f} ms, "
            f"p95 {summary['p95_ms']:.0f} ms, range {summary['min_ms']:.0f}-{summary['max_ms']:.0f} ms"
        )
        for row in rows:
            print(f"  {row['image']:<16} {row['mean_ms']:>8.0f} ms  {row['size'][0]}x{row['size'][1]}")

    print(
        f"peak RSS {memory.get('peak_rss_mb', 'not measured')} MB of "
        f"{memory_at_load.get('machine_total_ram_mb', 'n/a')} MB RAM "
        f"(resident right after the weight load: {memory_at_load.get('rss_now_mb', 'n/a')} MB)"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
