"""FDI tooth number for a detected finding, estimated from arch geometry.

DENTEX labels every abnormal tooth with a quadrant digit (1-4) and a tooth digit
(1-8) in FDI notation, so each training box has a true tooth number even though
the detector predicts only the four finding classes. This module fits that
geometry on the training split's ground-truth box centres: one threshold splits
upper from lower, one splits the two sides of the arch, and within a quadrant the
tooth digit comes from the nearest fitted tooth centroid.

The result is a position-based guess, not a dental measurement, and it inherits
every detection error: a wrong box or a wrong class gives a wrong number. The
measured accuracy of the estimate is recorded in the table itself and in
assets/eval/fdi_geometry.json.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

TABLE_VERSION = 1
BASIS = "arch_geometry"
QUADRANT_DIGITS = (1, 2, 3, 4)
TOOTH_DIGITS = tuple(range(1, 9))
TABLE_PATH = Path(__file__).resolve().parent / "assets" / "fdi" / "fdi_geometry.json"


def box_centre(box: Sequence[float], width: int, height: int) -> tuple[float, float]:
    """Normalised (u, v) centre of an xyxy box, clipped into the image."""
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return min(max(cx / float(width), 0.0), 1.0), min(max(cy / float(height), 0.0), 1.0)


def quadrant_of(u: float, v: float, u_midline: float, v_arch: float) -> int:
    """FDI quadrant digit from arch position: 1 upper right, 2 upper left, 3 lower left, 4 lower right."""
    upper = v < v_arch
    patient_right = u < u_midline
    if upper:
        return 1 if patient_right else 2
    return 4 if patient_right else 3


def _best_threshold(values: Sequence[float], labels: Sequence[int], candidates: Sequence[float]) -> tuple[float, int]:
    """Cut position with the fewest label<2 vs label>=2 mistakes; prefix counts keep it linear."""
    if not candidates:
        return 0.5, len(labels)
    ordered = sorted(zip(values, labels))
    sorted_values = [value for value, _ in ordered]
    prefix_low = [0]
    for _, label in ordered:
        prefix_low.append(prefix_low[-1] + (1 if label < 2 else 0))
    total_low = prefix_low[-1]
    n = len(ordered)

    best, best_errors = candidates[0], n + 1
    for cut in candidates:
        k = bisect_right(sorted_values, cut)
        correct = prefix_low[k] + ((n - k) - (total_low - prefix_low[k]))
        errors = n - correct
        if errors < best_errors:
            best, best_errors = cut, errors
    return best, best_errors


def _candidates(values: Sequence[float]) -> List[float]:
    ordered = sorted(values)
    return [(a + b) / 2.0 for a, b in zip(ordered, ordered[1:]) if b > a]


def fit_table(
    records: Sequence[Mapping[str, float]],
    *,
    v_weight: float = 1.0,
    source: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an assignment table from ground-truth records with u, v, quadrant and tooth digits."""
    if not records:
        raise ValueError("fit_table needs at least one record")

    us = [float(r["u"]) for r in records]
    vs = [float(r["v"]) for r in records]
    side_labels = [1 if r["quadrant"] in (1, 4) else 2 for r in records]
    level_labels = [1 if r["quadrant"] in (1, 2) else 3 for r in records]

    u_midline, side_errors = _best_threshold(us, side_labels, _candidates(us))
    v_arch, level_errors = _best_threshold(vs, level_labels, _candidates(vs))

    centroids: Dict[str, Dict[str, List[float]]] = {}
    support: Dict[str, int] = {}
    for quadrant in QUADRANT_DIGITS:
        group = [r for r in records if int(r["quadrant"]) == quadrant]
        if not group:
            continue
        per_tooth: Dict[str, List[float]] = {}
        for tooth in TOOTH_DIGITS:
            digits = [r for r in group if int(r["tooth"]) == tooth]
            if digits:
                per_tooth[str(tooth)] = [
                    round(sum(float(r["u"]) for r in digits) / len(digits), 6),
                    round(sum(float(r["v"]) for r in digits) / len(digits), 6),
                ]
            support[f"{quadrant}{tooth}"] = len(digits)
        centroids[str(quadrant)] = per_tooth

    return {
        "version": TABLE_VERSION,
        "basis": BASIS,
        "v_weight": v_weight,
        "quadrant_split": {"u_midline": round(u_midline, 6), "v_arch": round(v_arch, 6)},
        "split_train_accuracy": {
            "side": round(1.0 - side_errors / len(records), 4),
            "level": round(1.0 - level_errors / len(records), 4),
        },
        "tooth_centroids": centroids,
        "support": support,
        "n_fit_boxes": len(records),
        "source": dict(source or {}),
    }


def assign_fdi(u: float, v: float, table: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Nearest fitted tooth centroid in normalised arch coordinates, or None if the quadrant has no support."""
    split = table["quadrant_split"]
    quadrant = quadrant_of(u, v, float(split["u_midline"]), float(split["v_arch"]))
    per_tooth = table["tooth_centroids"].get(str(quadrant)) or {}
    if not per_tooth:
        return None

    weight = float(table.get("v_weight", 1.0))
    scored = sorted(
        (
            (u - centroid[0]) ** 2 + (weight * (v - centroid[1])) ** 2,
            int(tooth),
        )
        for tooth, centroid in per_tooth.items()
    )
    best_sq, tooth = scored[0]
    runner_up_sq = scored[1][0] if len(scored) > 1 else math.inf
    best = math.sqrt(best_sq)
    runner_up = math.sqrt(runner_up_sq)
    margin = 1.0 if runner_up == math.inf else round((runner_up - best) / max(runner_up, 1e-9), 4)
    return {"fdi": quadrant * 10 + tooth, "quadrant": quadrant, "tooth": tooth, "margin": margin}


def estimate_fdi(
    box: Sequence[float],
    width: int,
    height: int,
    table: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Assign an FDI number to one detected box. Returns None when no table is loaded."""
    if table is None or width <= 0 or height <= 0:
        return None
    u, v = box_centre(box, width, height)
    return assign_fdi(u, v, table)


def load_table(path: Path = TABLE_PATH) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if table.get("version") != TABLE_VERSION or "tooth_centroids" not in table:
        return None
    return table


def save_table(table: Mapping[str, Any], path: Path = TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
