"""Fit and measure the FDI tooth-number geometry table used by the 2D service.

DENTEX ships one 10.9 GB training archive, so the annotation JSON is read straight
out of it over HTTP range requests instead of downloading the archive.

Subcommands
-----------
fetch  download only the train annotation JSON member out of the pinned archive.
fit    reproduce the seed-42 split, pick the vertical weight on a one-in-five image
       holdout inside the 635 training images, fit the table on all of them, and
       write assets/fdi/fdi_geometry.json.
eval   measure the table: quadrant and tooth agreement on held-out validation
       ground-truth boxes, and, with a prediction dump, on detected boxes.

The 70 validation images are never used for fitting or tuning; tests/test_fdi.py
asserts that the shipped table records a train-only fit.

Nothing here reads the 50 held-out test images or their annotations.

Usage:
    python scripts/dentex_fdi.py fetch --output data/dentex/raw/train_quadrant_enumeration_disease.json
    python scripts/dentex_fdi.py fit --json data/dentex/raw/train_quadrant_enumeration_disease.json
    python scripts/dentex_fdi.py eval --json data/dentex/raw/train_quadrant_enumeration_disease.json \
        --preds preds_val/predictions_val.json
"""

import argparse
import io
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fdi

HF_REVISION = "7b27ccc8e342dcb774f69adc6ca5e6c09fefce93"
TRAIN_ZIP_URL = (
    "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/"
    f"{HF_REVISION}/DENTEX/training_data.zip"
)
TRAIN_JSON_MEMBER = "train_quadrant_enumeration_disease.json"
VAL_IMAGE_COUNT = 70
SEED = 42


class RangeFile(io.IOBase):
    """A seekable read-only view of one remote file, filled by HTTP Range requests."""

    def __init__(self, url: str, window: int = 8 * 1024 * 1024):
        self.url = url
        self.window = window
        self._size: Optional[int] = None
        self._pos = 0
        self._buf = b""
        self._buf_start = -1

    @property
    def size(self) -> int:
        if self._size is None:
            request = urllib.request.Request(self.url, method="HEAD")
            with urllib.request.urlopen(request) as response:  # nosec B310 - pinned public dataset
                length = response.headers.get("Content-Length")
                if length is None or int(length) < 200:
                    length = response.headers.get("X-Linked-Size")
                if length is None:
                    raise RuntimeError("Remote archive has no reported length.")
                self._size = int(length)
        return self._size

    def _fetch(self, start: int, stop: int) -> bytes:
        stop = min(stop, self.size) - 1
        if stop < start:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
        with urllib.request.urlopen(request) as response:  # nosec B310 - pinned public dataset
            if response.status != 206:
                raise RuntimeError(f"Remote server refused a byte range (HTTP {response.status}).")
            return response.read()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError(whence)
        self._pos = max(0, min(pos, self.size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        if self._buf_start > self._pos or self._pos + n > self._buf_start + len(self._buf):
            end = min(self.size, max(self._pos + n, self._pos + self.window))
            self._buf = self._fetch(self._pos, end)
            self._buf_start = self._pos
        offset = self._pos - self._buf_start
        chunk = self._buf[offset : offset + n]
        self._pos += len(chunk)
        return chunk

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def remote_member(url: str, member_suffix: str, dest: Path) -> int:
    with zipfile.ZipFile(RangeFile(url)) as archive:
        names = [n for n in archive.namelist() if n.endswith(member_suffix)]
        if not names:
            raise SystemExit(f"{member_suffix} not found in the remote archive.")
        data = archive.read(names[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"wrote {dest} ({len(data):,} bytes) from {names[0]}")
    return len(data)


def split_ids(data: Dict[str, Any]) -> Tuple[set, set]:
    """The same deterministic 635/70 split prepare_data.py applies to the train pool."""
    ids = sorted(img["id"] for img in data.get("images", []))
    rng = random.Random(SEED)
    rng.shuffle(ids)
    return set(ids[VAL_IMAGE_COUNT:]), set(ids[:VAL_IMAGE_COUNT])


def records(data: Dict[str, Any], allowed_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    dims = {img["id"]: (img.get("width", 0), img.get("height", 0)) for img in data.get("images", [])}
    out: List[Dict[str, Any]] = []
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        width, height = dims.get(image_id, (0, 0))
        if width <= 0 or height <= 0:
            continue
        if allowed_ids is not None and image_id not in allowed_ids:
            continue
        if ann.get("category_id_1") is None or ann.get("category_id_2") is None:
            continue
        x, y, w, h = ann["bbox"]
        u = min(max((x + w / 2.0) / width, 0.0), 1.0)
        v = min(max((y + h / 2.0) / height, 0.0), 1.0)
        out.append(
            {
                "u": u,
                "v": v,
                "quadrant": int(ann["category_id_1"]) + 1,
                "tooth": int(ann["category_id_2"]) + 1,
                "image_id": image_id,
                "diagnosis": int(ann.get("category_id_3", -1)),
                "box": [x, y, x + w, y + h],
                "width": width,
                "height": height,
            }
        )
    return out


def score(table: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    quadrants = teeth = exact = assigned = 0
    for row in rows:
        guess = fdi.assign_fdi(row["u"], row["v"], table)
        if guess is None:
            continue
        assigned += 1
        quadrants += guess["quadrant"] == row["quadrant"]
        teeth += guess["tooth"] == row["tooth"]
        exact += guess["fdi"] == row["quadrant"] * 10 + row["tooth"]
    n = max(assigned, 1)
    return {
        "n_assigned": assigned,
        "quadrant_accuracy": round(quadrants / n, 4),
        "tooth_accuracy": round(teeth / n, 4),
        "fdi_accuracy": round(exact / n, 4),
    }


def cmd_fetch(args: argparse.Namespace) -> None:
    dest = Path(args.output)
    if dest.is_file() and dest.stat().st_size > 0 and not args.force:
        print(f"[SKIP] {dest} already exists ({dest.stat().st_size:,} bytes).")
        return
    remote_member(args.url, TRAIN_JSON_MEMBER, dest)


def cmd_fit(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    train_ids, val_ids = split_ids(data)
    train_rows = records(data, train_ids)
    val_rows = records(data, val_ids)
    if not train_rows or not val_rows:
        raise SystemExit("Expected both train and validation records.")

    # Pick v_weight inside the train pool only (fit on 4/5 of the images, score the
    # remaining 1/5), so the validation numbers below are free of any selection.
    held_ids = {image_id for i, image_id in enumerate(sorted(train_ids)) if i % 5 == 0}
    fit_rows = [row for row in train_rows if row["image_id"] not in held_ids]
    tune_rows = [row for row in train_rows if row["image_id"] in held_ids]

    weights = args.v_weights or [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    trials = []
    for weight in weights:
        table = fdi.fit_table(fit_rows, v_weight=weight)
        measured = score(table, tune_rows)
        trials.append((weight, measured))
        print(
            f"  v_weight {weight:>4}: train-holdout quadrant {measured['quadrant_accuracy']:.3f} "
            f"tooth {measured['tooth_accuracy']:.3f} FDI {measured['fdi_accuracy']:.3f}"
        )
    best_weight, best_tune = max(trials, key=lambda t: (t[1]["fdi_accuracy"], t[1]["tooth_accuracy"]))

    table = fdi.fit_table(train_rows, v_weight=best_weight)
    val_scores = score(table, val_rows)
    table["tuning"] = {
        "selected_on": "a 1/5 image holdout inside the 635-image train pool",
        "n_fit_boxes": len(fit_rows),
        "n_tune_boxes": len(tune_rows),
        "scores_by_v_weight": [{"v_weight": w, **s} for w, s in trials],
        **best_tune,
    }
    table["validation"] = {
        "split": "val70 (the 70-image validation split from the same train pool)",
        "n_images": len(val_ids),
        "n_boxes": len(val_rows),
        "used_for": "measurement only; nothing here was selected on these boxes",
        **val_scores,
    }
    table["source"] = {
        "dataset": "DENTEX",
        "revision": HF_REVISION,
        "annotation_file": TRAIN_JSON_MEMBER,
        "fit_split": f"train (635 images of the 705-image pool, seed={SEED})",
        "fit_rule": "same split as prepare_data.py",
    }
    fdi.save_table(table, Path(args.output))
    print(
        f"fitted on {len(train_rows):,} train boxes; v_weight {best_weight}; "
        f"val FDI {val_scores['fdi_accuracy']:.3f}, quadrant {val_scores['quadrant_accuracy']:.3f}, "
        f"tooth {val_scores['tooth_accuracy']:.3f} -> {args.output}"
    )
    if args.write_eval:
        out = Path(args.write_eval)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"tuning": table["tuning"], "validation": table["validation"]}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {out}")


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def cmd_eval(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    _, val_ids = split_ids(data)
    val_rows = records(data, val_ids)
    table = fdi.load_table(Path(args.table))
    if table is None:
        raise SystemExit(f"No table at {args.table}; run fit first.")

    result: Dict[str, Any] = {
        "table": str(args.table),
        "table_source": table.get("source"),
        "oracle_ground_truth_boxes": score(table, val_rows),
    }

    if args.preds:
        preds = json.loads(Path(args.preds).read_text(encoding="utf-8"))
        by_image: Dict[int, List[Dict[str, Any]]] = {}
        for pred in preds.get("predictions", preds.get("annotations", [])):
            by_image.setdefault(int(pred["image_id"]), []).append(pred)
        dims = {img["id"]: (img["width"], img["height"]) for img in data.get("images", [])}
        matched: List[Dict[str, Any]] = []
        for row in val_rows:
            width, height = dims[row["image_id"]]
            best, best_iou = None, 0.0
            for pred in by_image.get(row["image_id"], []):
                x, y, w, h = pred["bbox"]
                box = [x, y, x + w, y + h]
                iou = _iou(box, row["box"])
                if iou > best_iou:
                    best, best_iou = box, iou
            if best is not None and best_iou >= args.iou:
                u, v = fdi.box_centre(best, width, height)
                matched.append({**row, "u": u, "v": v})
        result["detected_boxes"] = {
            "iou_match": args.iou,
            "n_gt_boxes": len(val_rows),
            "n_matched": len(matched),
            **score(table, matched),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--url", default=TRAIN_ZIP_URL)
    p_fetch.add_argument("--output", default="data/dentex/raw/train_quadrant_enumeration_disease.json")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_fetch)

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--json", required=True)
    p_fit.add_argument("--output", default=str(fdi.TABLE_PATH))
    p_fit.add_argument("--write-eval", default="assets/eval/fdi_geometry.json")
    p_fit.add_argument("--v-weights", type=float, nargs="*", default=None)
    p_fit.set_defaults(func=cmd_fit)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--json", required=True)
    p_eval.add_argument("--table", default=str(fdi.TABLE_PATH))
    p_eval.add_argument("--preds", help="COCO prediction dump for the validation split")
    p_eval.add_argument("--iou", type=float, default=0.5)
    p_eval.add_argument(
        "--output",
        default="assets/eval/fdi_validation.json",
        help="Separate from fit's --write-eval so re-measuring does not overwrite the tuning report",
    )
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
