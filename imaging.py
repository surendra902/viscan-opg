"""
DICOM and Image Ingestion Pipeline for Viscan OPG.

Handles:
- PNG, JPEG, and single-frame monochrome 2D DICOM (.dcm) files.
- Rescale slope/intercept, VOI LUT, and Window Center/Width.
- MONOCHROME1 inversion (white=low, black=high -> inverted to standard display).
- Rejection of multiframe, compressed/unsupported transfer syntaxes, and corrupted inputs.
- Safe patient metadata exclusion (no PHI retained in memory or outputs).
"""

import io
from typing import Tuple, Optional
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    HAS_PYDICOM = True
except ImportError:
    pydicom = None
    apply_modality_lut = None
    apply_voi_lut = None
    HAS_PYDICOM = False


class ImageProcessingError(Exception):
    """Base exception for imaging errors."""
    pass


class InvalidImageError(ImageProcessingError):
    """Raised when the input is corrupted or cannot be parsed."""
    pass


class UnsupportedDICOMError(ImageProcessingError):
    """Raised when DICOM format/modality/frames are unsupported."""
    pass


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit
MAX_PIXEL_DIM = 8192                 # Safety cap against decompression bombs


def load_image_bytes(data: bytes, filename: str = "") -> Tuple[Image.Image, str, Tuple[int, int]]:
    """
    Decodes raw image bytes (PNG, JPEG, or 2D DICOM) into a PIL RGB Image.

    Returns:
        (image_pil, modality, (original_width, original_height))
    """
    if len(data) == 0:
        raise InvalidImageError("Uploaded file is empty (0 bytes).")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(f"Uploaded file exceeds maximum size of {MAX_UPLOAD_BYTES // (1024*1024)}MB.")

    # Check if input is a DICOM file: standard 128-byte preamble or file extension
    is_dicom = False
    if len(data) > 132 and data[128:132] == b"DICM":
        is_dicom = True
    elif filename.lower().endswith(".dcm"):
        is_dicom = True

    if is_dicom:
        return load_dicom_bytes(data)
    else:
        return load_standard_image_bytes(data)


def load_standard_image_bytes(data: bytes) -> Tuple[Image.Image, str, Tuple[int, int]]:
    """Decodes PNG, JPEG, BMP, or WebP bytes to a PIL RGB image."""
    try:
        bio = io.BytesIO(data)
        img = Image.open(bio)
        img.load()  # Force decode to verify integrity
    except Exception as e:
        raise InvalidImageError(f"Unreadable or corrupted image file: {e}")

    width, height = img.size
    if width > MAX_PIXEL_DIM or height > MAX_PIXEL_DIM:
        raise InvalidImageError(f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({MAX_PIXEL_DIM}).")

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img, "OPG", (width, height)


def load_dicom_bytes(data: bytes) -> Tuple[Image.Image, str, Tuple[int, int]]:
    """
    Decodes single-frame 2D monochrome DICOM bytes into a PIL RGB image.
    Applies VOI LUT / windowing and MONOCHROME1 inversion.
    Excludes all patient metadata.
    """
    if not HAS_PYDICOM:
        raise ImageProcessingError("pydicom is not installed. Please install pydicom to handle DICOM files.")

    try:
        bio = io.BytesIO(data)
        ds = pydicom.dcmread(bio, force=True)
    except Exception as e:
        raise InvalidImageError(f"Failed to parse DICOM dataset: {e}")

    # Check pixel data presence
    if not hasattr(ds, "PixelData") or ds.PixelData is None:
        raise UnsupportedDICOMError("DICOM file contains no PixelData.")

    # Validate single frame
    num_frames = getattr(ds, "NumberOfFrames", 1)
    try:
        num_frames = int(num_frames)
    except (ValueError, TypeError):
        num_frames = 1

    if num_frames > 1:
        raise UnsupportedDICOMError(
            f"Multiframe DICOM ({num_frames} frames) is not supported in the 2D OPG pipeline."
        )

    # Validate photometric interpretation
    photo_interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photo_interp not in ("MONOCHROME1", "MONOCHROME2"):
        raise UnsupportedDICOMError(
            f"PhotometricInterpretation '{photo_interp}' is unsupported. Only MONOCHROME1 and MONOCHROME2 are supported."
        )

    try:
        pixel_array = ds.pixel_array
    except Exception as e:
        raise InvalidImageError(
            f"Cannot decode DICOM pixel data. Compressed transfer syntaxes require additional codecs: {e}"
        )

    # Handle shape (must be 2D)
    if pixel_array.ndim != 2:
        raise UnsupportedDICOMError(f"Expected 2D image, got array with shape {pixel_array.shape}.")

    orig_height, orig_width = pixel_array.shape
    if orig_width > MAX_PIXEL_DIM or orig_height > MAX_PIXEL_DIM:
        raise InvalidImageError(f"DICOM dimensions ({orig_width}x{orig_height}) exceed safety limit.")

    # Apply Modality LUT (RescaleSlope / RescaleIntercept) before VOI LUT / Windowing
    try:
        if apply_modality_lut is not None:
            modality_pixels = apply_modality_lut(pixel_array, ds)
        else:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            modality_pixels = pixel_array.astype(np.float32) * slope + intercept

        has_window = hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth")
        has_voi = hasattr(ds, "VOILUTSequence")
        if apply_voi_lut is not None and (has_window or has_voi):
            processed = apply_voi_lut(modality_pixels, ds)
        else:
            processed = _apply_window_fallback(pixel_array, ds)
    except Exception:
        processed = _apply_window_fallback(pixel_array, ds)

    # Normalize to 0 - 255 uint8
    min_val = float(np.min(processed))
    max_val = float(np.max(processed))
    if max_val > min_val:
        normalized = ((processed - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros_like(processed, dtype=np.uint8)

    # MONOCHROME1: 0 corresponds to white, maximum corresponds to black
    # Invert so 255 is white and 0 is black (standard display)
    if photo_interp == "MONOCHROME1":
        normalized = 255 - normalized

    # Convert grayscale 2D to 3-channel RGB PIL Image
    rgb_array = np.stack([normalized] * 3, axis=-1)
    img = Image.fromarray(rgb_array, mode="RGB")

    modality = str(getattr(ds, "Modality", "OPG"))
    if not modality or modality == "":
        modality = "OPG"

    return img, modality, (orig_width, orig_height)


def _apply_window_fallback(pixel_array: np.ndarray, ds) -> np.ndarray:
    """Fallback windowing computation when apply_voi_lut is unavailable."""
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    rescaled = pixel_array.astype(np.float32) * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        if hasattr(wc, "__iter__"):
            wc = wc[0]
        if hasattr(ww, "__iter__"):
            ww = ww[0]
        c = float(wc)
        w = float(ww)
        if w > 0:
            lower = c - 0.5 - (w - 1) / 2
            upper = c - 0.5 + (w - 1) / 2
            rescaled = np.clip(rescaled, lower, upper)

    return rescaled
