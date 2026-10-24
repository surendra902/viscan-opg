"""Builds samples/cbct/<Scan>_bundle.zip (grayscale volume + DentalSegmentator label map) for the demo.

Each bundle is one real, distinct scan. The service computes slices, findings and 3D meshes from the bundle's
own data - nothing is copied from another scan.

Usage:
    python build_cbct_bundles.py                      # (re)builds PostDentalSurgery from its loose files
    # PreDentalSurgery: run kaggle_cbct_scan2_notebook.ipynb on Kaggle and drop PreDentalSurgery_bundle.zip
    # into samples/cbct/.
"""
import zipfile
from pathlib import Path

DIR = Path(__file__).resolve().parent / "samples" / "cbct"


def build(scan: str) -> Path:
    vol = DIR / f"{scan}_Volume.nii.gz"
    seg = DIR / f"{scan}_Segmentation.nii.gz"
    out = DIR / f"{scan}_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:  # NIfTI is already gzip-compressed
        zf.write(vol, vol.name)
        zf.write(seg, seg.name)
    return out


if __name__ == "__main__":
    print(build("PostDentalSurgery"))
