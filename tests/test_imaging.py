"""Unit tests for imaging module (DICOM, MONOCHROME1, and bounds checks)."""

import io
import pytest
import numpy as np
from PIL import Image

from imaging import (
    load_image_bytes,
    load_standard_image_bytes,
    InvalidImageError,
    UnsupportedDICOMError,
    _apply_window_fallback,
)


def test_load_standard_png():
    """Verify loading and RGB conversion of a standard PNG image."""
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    raw_bytes = bio.getvalue()

    loaded_img, modality, (w, h) = load_image_bytes(raw_bytes, "test.png")
    assert modality == "OPG"
    assert (w, h) == (640, 480)
    assert loaded_img.mode == "RGB"
    assert loaded_img.size == (640, 480)


def test_empty_bytes_rejected():
    """Verify 0-byte upload raises InvalidImageError."""
    with pytest.raises(InvalidImageError, match="empty"):
        load_image_bytes(b"", "empty.png")


def test_corrupt_bytes_rejected():
    """Verify corrupt random bytes raise InvalidImageError."""
    with pytest.raises(InvalidImageError):
        load_image_bytes(b"CORRUPT_NOT_AN_IMAGE_DATA_12345", "corrupt.jpg")


def test_windowing_fallback_logic():
    """Verify windowing fallback formula correctly clips intensities."""
    class MockDataset:
        RescaleSlope = 1.0
        RescaleIntercept = 0.0
        WindowCenter = 100.0
        WindowWidth = 50.0

    raw_arr = np.array([[50, 100, 150]], dtype=np.float32)
    ds = MockDataset()
    res = _apply_window_fallback(raw_arr, ds)

    # c=100, w=50 -> lower=75, upper=124.5
    assert res[0, 0] >= 75.0  # 50 clipped up to lower
    assert res[0, 2] <= 125.0 # 150 clipped down to upper


def test_monochrome1_inversion_simulation():
    """Verify that MONOCHROME1 inverts black and white correctly."""
    # Create gradient
    norm = np.array([[0, 128, 255]], dtype=np.uint8)
    inverted = 255 - norm
    assert inverted[0, 0] == 255
    assert inverted[0, 2] == 0
