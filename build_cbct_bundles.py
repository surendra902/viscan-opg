"""Builds samples/cbct/<Scan>_bundle.zip for the demo scans.

Each bundle contains that scan's grayscale volume, its 5-class label map, a
provenance.json describing where the labels came from, and (for the scan that
has one) the Kaggle run_metrics.json. Nothing is copied between scans.

Usage:
    python build_cbct_bundles.py
"""
import json
import zipfile
from pathlib import Path

from cbct_pipeline import DEMO_SCAN_PROVENANCE

DIR = Path(__file__).resolve().parent / "samples" / "cbct"
SCANS = ("PostDentalSurgery", "PreDentalSurgery")


def _carried_members(scan: str) -> dict:
    """Keeps run_metrics.json (and any other non-volume member) from the current bundle."""
    out = DIR / f"{scan}_bundle.zip"
    carried = {}
    if not out.exists():
        return carried
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            if name.endswith((".nii", ".nii.gz")):
                continue
            carried[name] = zf.read(name)
    return carried


def build(scan: str) -> Path:
    vol = DIR / f"{scan}_Volume.nii.gz"
    seg = DIR / f"{scan}_Segmentation.nii.gz"
    if not vol.is_file() or not seg.is_file():
        raise FileNotFoundError(f"Missing {vol.name} or {seg.name} for {scan}.")

    provenance_info = DEMO_SCAN_PROVENANCE[scan]
    provenance = {
        "segmentation_source": provenance_info["source"],
        "model_inference_ran": provenance_info["source"] == "demo_model_labelmap",
        "detail": provenance_info["detail"],
        "volume_member": vol.name,
        "segmentation_member": seg.name,
    }
    carried = _carried_members(scan)
    carried.pop("provenance.json", None)

    out = DIR / f"{scan}_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        zf.write(vol, vol.name)
        zf.write(seg, seg.name)
        zf.writestr(
            "provenance.json",
            json.dumps(provenance, indent=2) + "\n",
            compress_type=zipfile.ZIP_DEFLATED,
        )
        for name, data in sorted(carried.items()):
            zf.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    return out


if __name__ == "__main__":
    for scan in SCANS:
        path = build(scan)
        print(f"built {path.name} ({path.stat().st_size:,} bytes)")
