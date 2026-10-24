"""Unit tests for FastAPI endpoints and JSON response schemas."""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health endpoint returns status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_analyze_valid_dummy_image(client):
    """Verify POST /analyze with a valid grayscale image returns 200 and expected schema."""
    # Create valid synthetic grayscale dental-like image (wide aspect ratio and realistic dynamic range)
    arr = np.linspace(40, 210, 800 * 400, dtype=np.uint8).reshape((400, 800))
    img = Image.fromarray(arr, mode="L")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)

    response = client.post(
        "/analyze",
        files={"file": ("test_opg.png", bio.getvalue(), "image/png")},
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
    assert "summary" in data
    assert "annotated_image_url" in data
    assert data["disclaimer"] == "AI-assisted findings for dentist review. Not a diagnosis."


def test_analyze_empty_file_rejected(client):
    """Verify empty file upload returns 400."""
    response = client.post(
        "/analyze",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
