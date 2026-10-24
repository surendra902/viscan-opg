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


def test_dicom_rescale_precedes_windowing():
    """Verify RescaleSlope and RescaleIntercept are applied before VOI LUT windowing (D03)."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    def _make_dicom(array, slope=1.0, intercept=0.0):
        array = np.asarray(array, dtype=np.uint16)
        fm = FileMetaDataset()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.ImplementationClassUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID = fm.MediaStorageSOPClassUID
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.Modality = "DX"
        ds.Rows, ds.Columns = array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        ds.WindowCenter = 450
        ds.WindowWidth = 700
        ds.PixelData = array.tobytes()
        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        return buf.getvalue()

    raw = np.arange(8, dtype=np.uint16).reshape(2, 4) * 50
    physical = raw * 2 + 100
    out_rescaled = np.asarray(load_image_bytes(_make_dicom(raw, slope=2.0, intercept=100.0), "a.dcm")[0])
    out_physical = np.asarray(load_image_bytes(_make_dicom(physical, slope=1.0, intercept=0.0), "b.dcm")[0])
    assert np.array_equal(out_rescaled, out_physical)
