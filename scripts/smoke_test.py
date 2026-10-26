"""End-to-end smoke test for a running Viscan server.

Covers the cases a reviewer is most likely to try: a sample with findings, an FDI
number and provenance block on every finding, a sample with none, a wrong file
type, a corrupt DICOM, a colour non-dental image, a continuous-tone 2:1 grayscale
image and a synthetic MONOCHROME1 DICOM that carries a patient name.

Start the app first (python app.py or docker compose up), then run:
    python scripts/smoke_test.py
"""

import argparse
import io
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def analyse(client: httpx.Client, base: str, name: str, data: bytes, content_type: str):
    return client.post(
        f"{base}/analyze",
        files={"file": (name, data, content_type)},
    )


def grayscale_2to1_png() -> bytes:
    """Continuous-tone grayscale 2:1 image that is not a dental radiograph.

    It satisfies every rule the heuristic gate checks (wide aspect, no colour,
    mid-tone mass, continuous histogram), which is the point: the gate cannot
    tell it apart from a radiograph. Recorded as a documented false accept.
    """
    height, width = 800, 1600
    yy, xx = np.mgrid[0:height, 0:width]
    rng = np.random.default_rng(3)
    base = 90.0 + 60.0 * np.sin(xx / 210.0) + 40.0 * np.cos(yy / 130.0)
    array = np.clip(base + rng.normal(0.0, 12.0, (height, width)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def monochrome1_dicom() -> bytes:
    """Synthetic MONOCHROME1 DICOM with a patient name, to prove it never leaks."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    rows, cols = 800, 1600
    rng = np.random.default_rng(7)
    gradient = np.linspace(40.0, 210.0, cols, dtype=np.float32)[None, :]
    array = np.clip(gradient + rng.normal(0.0, 18.0, (rows, cols)), 0, 255).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "DOE^JOHN"
    ds.PatientID = "PHI-TEST-123"
    ds.Modality = "DX"
    ds.Rows, ds.Columns = array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = array.tobytes()

    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    passed = 0
    failed = 0

    def report(label: str, ok: bool, detail: str) -> None:
        nonlocal passed, failed
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    with httpx.Client(timeout=args.timeout) as client:
        try:
            health = client.get(f"{base}/health")
        except httpx.HTTPError as exc:
            print(f"Cannot reach {base}: {exc}")
            print("Start the server first: python app.py (or docker compose up).")
            return 1

        payload = health.json() if health.status_code == 200 else {}
        model_loaded = bool(payload.get("model_loaded"))
        report(
            "health",
            health.status_code == 200 and model_loaded,
            f"HTTP {health.status_code}, model_loaded={model_loaded}",
        )
        if not model_loaded:
            print("The 2D model is not loaded, so the upload cases will return 503.")
            return 1

        with_findings = SAMPLES / "sample_01_val_1_input.png"
        response = analyse(client, base, with_findings.name, with_findings.read_bytes(), "image/png")
        body = response.json() if response.status_code == 200 else {}
        findings = body.get("findings", [])
        report(
            "sample with findings",
            response.status_code == 200
            and body.get("modality") == "OPG"
            and len(findings) >= 1
            and bool(body.get("annotated_image_url")),
            f"HTTP {response.status_code}, {len(findings)} findings, {body.get('summary', '')}",
        )

        numbered = [item for item in findings if isinstance(item.get("tooth_fdi"), int)]
        numbering = body.get("tooth_numbering") or {}
        validation = numbering.get("held_out_validation") or {}
        report(
            "tooth numbers attached",
            len(findings) >= 1
            and len(numbered) == len(findings)
            and numbering.get("basis") == "arch_geometry"
            and validation.get("fdi_exact") is not None,
            f"{len(numbered)}/{len(findings)} findings carry an FDI number; "
            f"basis={numbering.get('basis')}, held-out exact={validation.get('fdi_exact')}",
        )

        no_findings = SAMPLES / "sample_03_val_11_input.png"
        response = analyse(client, base, no_findings.name, no_findings.read_bytes(), "image/png")
        body = response.json() if response.status_code == 200 else {}
        report(
            "no findings",
            response.status_code == 200
            and body.get("findings") == []
            and "No pathological findings" in body.get("summary", ""),
            f"HTTP {response.status_code}, {len(body.get('findings', []))} findings",
        )

        response = analyse(client, base, "notes.txt", b"this is not an image\n", "text/plain")
        report(
            "wrong file type",
            response.status_code == 400,
            f"HTTP {response.status_code}, {response.json().get('detail', '')[:100]}",
        )

        response = analyse(client, base, "broken.dcm", b"\x00" * 256, "application/dicom")
        report(
            "corrupt DICOM",
            response.status_code == 422,
            f"HTTP {response.status_code}, {response.json().get('detail', '')[:100]}",
        )

        buffer = io.BytesIO()
        Image.new("RGB", (512, 512), (220, 20, 20)).save(buffer, format="PNG")
        response = analyse(client, base, "red_square.png", buffer.getvalue(), "image/png")
        report(
            "colour non-dental image",
            response.status_code == 422,
            f"HTTP {response.status_code}, {response.json().get('detail', '')[:100]}",
        )

        # Documents a known limitation: a continuous-tone grayscale 2:1 image
        # passes the heuristic gate and reaches the detector (README Section 12).
        response = analyse(client, base, "grayscale_2to1.png", grayscale_2to1_png(), "image/png")
        body = response.json() if response.status_code == 200 else {}
        report(
            "grayscale non-dental image",
            response.status_code == 200,
            f"HTTP {response.status_code}, {len(body.get('findings', []))} findings "
            "(accepted by the heuristic gate; documented false accept)",
        )

        response = analyse(client, base, "synthetic_mono1.dcm", monochrome1_dicom(), "application/dicom")
        leaked = [token for token in ("DOE", "JOHN", "PHI-TEST-123") if token in response.text]
        report(
            "MONOCHROME1 DICOM",
            response.status_code == 200 and not leaked,
            f"HTTP {response.status_code}, patient identifiers in output: {leaked or 'none'}",
        )

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
