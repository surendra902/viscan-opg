"""Regenerates samples/cbct artifacts from the shipped demo bundles.

Every PNG, STL, GLB and JSON is produced by running the real service on that
bundle, so the artifacts cannot diverge from the bundle contents. Demo
provenance is attached server-side (the labels are pre-segmented; no live
model inference is claimed for these requests).

Usage:
    python generate_cbct_samples.py
"""
import json
import shutil
from pathlib import Path

from cbct_pipeline import DEMO_SCAN_PROVENANCE
from service import DentalAnalysisService

SAMPLES_DIR = Path(__file__).resolve().parent / "samples" / "cbct"


def regenerate(scan: str, service: DentalAnalysisService, tmp_root: Path) -> None:
    bundle = SAMPLES_DIR / f"{scan}_bundle.zip"
    if not bundle.is_file():
        raise FileNotFoundError(bundle)

    tmp_results = tmp_root / scan
    tmp_results.mkdir(parents=True, exist_ok=True)
    service.results_dir = tmp_results

    report = service.analyze_cbct_bytes(
        bundle.read_bytes(),
        bundle.name,
        provenance_override=DEMO_SCAN_PROVENANCE[scan],
    )

    def produced(url: str) -> Path:
        return tmp_results / Path(url).name

    targets = {
        "original_image_url": f"{scan}_cbct_original.png",
        "annotated_image_url": f"{scan}_cbct_ortho.png",
        "mesh_glb_url": f"{scan}_scene.glb",
    }
    for key, filename in targets.items():
        shutil.copyfile(produced(report[key]), SAMPLES_DIR / filename)
        report[key] = f"/results/{filename}"

    for cname, url in report["mesh_stl_urls"].items():
        filename = f"{scan}_{cname}.stl"
        shutil.copyfile(produced(url), SAMPLES_DIR / filename)
        report["mesh_stl_urls"][cname] = f"/results/{filename}"

    report["image_id"] = scan
    (SAMPLES_DIR / f"{scan}_cbct_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[OK] {scan}: source={report['segmentation_source']} "
        f"demo={report['demo']} inference_ran={report['model']['inference_ran_in_this_request']} "
        f"total_ms={report['timings_ms']['total']}"
    )
    for finding in report["findings"]:
        print(f"     {finding['hemimandible']}: {finding['min_distance_mm']} mm")


if __name__ == "__main__":
    tmp_root = SAMPLES_DIR / "_tmp_generate"
    shutil.rmtree(tmp_root, ignore_errors=True)
    service = DentalAnalysisService(
        weights_path=None,
        results_dir=str(tmp_root),
        device="cpu",
    )
    try:
        for scan in sorted(DEMO_SCAN_PROVENANCE):
            regenerate(scan, service, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
