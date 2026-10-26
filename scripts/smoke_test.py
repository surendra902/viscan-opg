"""End-to-end smoke test for a running Viscan server.

Covers the five cases a reviewer is most likely to try: a sample with findings,
a sample with none, a wrong file type, a corrupt DICOM and a non-dental image.

Start the app first (python app.py or docker compose up), then run:
    python scripts/smoke_test.py
"""

import argparse
import io
import sys
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def analyse(client: httpx.Client, base: str, name: str, data: bytes, content_type: str):
    return client.post(
        f"{base}/analyze",
        files={"file": (name, data, content_type)},
    )


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
            "non-dental image",
            response.status_code == 422,
            f"HTTP {response.status_code}, {response.json().get('detail', '')[:100]}",
        )

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
