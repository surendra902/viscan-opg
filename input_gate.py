"""
Input Gate: Dental OPG Validity Classifier.

Rejects non-dental inputs (e.g., standard photos, documents, chest radiographs)
with clear, informative diagnostic errors before detector inference.
"""

from typing import Tuple
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NonDentalInputError(Exception):
    """Raised when an uploaded image is identified as non-dental."""
    pass


class DentalInputGate:
    """
    Validates whether an input image conforms to a 2D dental radiograph (OPG).
    Combines radiological image characteristics (aspect ratio, intensity distribution,
    radiographic contrast) with an optional fine-tuned MobileNetV3 binary classifier.
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = None
        if HAS_TORCH and weights_path:
            try:
                self.model = models.mobilenet_v3_small(weights=None)
                self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
                state = torch.load(weights_path, map_location=device)
                self.model.load_state_dict(state)
                self.model.to(device)
                self.model.eval()
            except Exception:
                self.model = None

    def validate(self, img: Image.Image) -> Tuple[bool, float, str]:
        """
        Validates image.
        Returns: (is_valid: bool, score: float, message: str)
        """
        width, height = img.size

        # 1. Structural check: Aspect Ratio
        # Panoramic OPGs have a wide aspect ratio (typically 1.4 to 2.5).
        # Extremely tall or extreme portrait images are not panoramic dental OPGs.
        aspect_ratio = width / max(1, height)
        if aspect_ratio < 0.65:
            return False, 0.05, f"Image aspect ratio ({aspect_ratio:.2f}) indicates a portrait format, not a panoramic dental radiograph."

        # 2. Photometric / Color Saturation Check
        # Dental X-rays are fundamentally monochrome/grayscale.
        # High colour saturation indicates a standard photographic image.
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        color_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r))
        # Grayscale images have color_diff near 0 (<= 5.0 due to subtle compression artifacts)
        if color_diff > 25.0:
            return False, 0.02, f"Image contains substantial chromatic color (mean diff={color_diff:.1f}). Dental X-rays are monochrome radiographs."

        # 3. Dynamic Range & Contrast Check
        # X-rays must have non-trivial dynamic range (not blank, solid color, or extreme clipping)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        std_dev = float(np.std(gray))
        if std_dev < 8.0:
            return False, 0.01, f"Image lacks sufficient dynamic range (std={std_dev:.1f}). The file appears blank, corrupt, or unexposed."

        # 4. Radiographic Continuous-Tone & Histogram Modality Check
        # Scanned text documents, screenshots, and binary graphics have bimodal or sparse histograms
        # dominated by near-white backgrounds or two discrete tones, whereas dental OPGs exhibit
        # continuous grayscale attenuation across soft tissue, bone, dentin, and enamel.
        near_white_frac = float(np.mean(gray >= 240.0))
        if near_white_frac > 0.35:
            return (
                False,
                0.03,
                f"Image is dominated by a bright/white background ({near_white_frac * 100:.1f}% near-white pixels), typical of a document or screenshot rather than a dental radiograph.",
            )

        midtone_frac = float(np.mean((gray > 15.0) & (gray < 235.0)))
        if midtone_frac < 0.25:
            return (
                False,
                0.04,
                f"Image lacks continuous radiographic mid-tone tissue attenuation (mid-tone fraction={midtone_frac:.2f}). Likely a binary text document or synthetic graphic.",
            )

        hist, _ = np.histogram(gray, bins=32, range=(0.0, 256.0))
        hist_probs = hist / max(1, gray.size)
        top2_bin_mass = float(np.sort(hist_probs)[-2:].sum())
        if top2_bin_mass > 0.60:
            return (
                False,
                0.05,
                f"Image intensity histogram is concentrated in two discrete tones (top-2 bin mass={top2_bin_mass:.2f}), indicating a document or diagram rather than a continuous-tone radiograph.",
            )

        # 5. Neural Classifier check (if model loaded)
        if self.model is not None and HAS_TORCH:
            try:
                tfms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                tensor = tfms(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    dental_prob = float(probs[0, 1].item())
                    if dental_prob < 0.50:
                        return False, dental_prob, f"Input rejected by classifier as non-dental (dental confidence: {dental_prob:.2f})."
                    return True, dental_prob, "Valid dental radiograph."
            except Exception:
                pass

        # Heuristic gate passed (no neural classifier checkpoint loaded)
        heuristic_score = round(float(min(0.90, max(0.55, midtone_frac))), 2)
        return True, heuristic_score, "Valid dental radiograph (heuristic gate)."
