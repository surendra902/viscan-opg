"""Unit tests for FastAPI endpoints, fail-closed weight checks, input gate, and JSON response schemas."""

import io
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import app as app_module
from app import app, service


@pytest.fixture
def client():
    return TestClient(app)


def _make_continuous_opg_png() -> bytes:
    """Creates a synthetic continuous-tone grayscale image passing radiographic checks."""
    arr = np.linspace(40, 210, 800 * 400, dtype=np.uint8).reshape((400, 800))
    img = Image.fromarray(arr, mode="L")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def test_health_endpoint(client):
    """Verify /health endpoint accurately reports model_loaded and status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    expected_status = "healthy" if data["model_loaded"] else "degraded"
    assert data["status"] == expected_status


def test_analyze_missing_weights_returns_503(client, monkeypatch):
    """Verify POST /analyze fails closed with HTTP 503 when detector weights are absent (D01)."""
    monkeypatch.setattr(service, "model", None)
    response = client.post(
        "/analyze",
        files={"file": ("test_opg.png", _make_continuous_opg_png(), "image/png")},
    )
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower() or "unavailable" in response.json()["detail"].lower()


def test_analyze_valid_image_with_loaded_model(client, monkeypatch):
    """Verify POST /analyze with loaded detector returns HTTP 200 and full required schema."""
    class DummyDetector:
        def __call__(self, images):
            return [{
                "boxes": torch.tensor([[100.0, 80.0, 220.0, 190.0]], dtype=torch.float32),
                "scores": torch.tensor([0.91], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.int64),
            }]

    monkeypatch.setattr(service, "model", DummyDetector())

    response = client.post(
        "/analyze",
        files={"file": ("test_opg.png", _make_continuous_opg_png(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    # Schema checks
    assert "image_id" in data
    assert data["modality"] == "OPG"
    assert "model" in data
    assert data["model"]["name"] == "fasterrcnn_resnet50_fpn_v2"
    assert "inference_ms" in data
    assert isinstance(data["findings"], list)
    assert len(data["findings"]) == 1
    assert data["findings"][0]["class"] == "caries"
    assert data["findings"][0]["confidence"] == 0.91
    assert "summary" in data
    assert "original_image_url" in data
    assert "annotated_image_url" in data
    assert data["disclaimer"] == "AI-assisted findings for dentist review. Not a diagnosis."


def test_analyze_rejects_grayscale_text_document(client):
    """Verify non-dental grayscale text document is rejected with HTTP 422 (D02)."""
    image = Image.new("RGB", (1000, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 980, 90), fill="black")
    draw.text((30, 40), "SYNTHETIC DOCUMENT - NOT A RADIOGRAPH", fill="white")
    for y in range(110, 480, 35):
        draw.text((30, y), "This is a text document with no dental anatomy. " * 3, fill="black")
        draw.line((30, y + 20, 950, y + 20), fill="black", width=3)
    bio = io.BytesIO()
    image.save(bio, format="PNG")

    response = client.post(
        "/analyze",
        files={"file": ("document.png", bio.getvalue(), "image/png")},
    )
    assert response.status_code == 422


def test_analyze_empty_file_rejected(client):
    """Verify empty file upload returns 400."""
    response = client.post(
        "/analyze",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_analyze_oversized_upload_returns_413(client, monkeypatch):
    """Verify an upload above the configured cap is rejected before any model work (HTTP 413)."""
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1024)
    response = client.post(
        "/analyze",
        files={"file": ("oversized_opg.png", b"\x89PNG\r\n\x1a\n" + b"0" * 4096, "image/png")},
    )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()
