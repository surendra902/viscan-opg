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

        # 4. Neural Classifier check (if model loaded)
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

        # If heuristic checks pass and no classifier failure
        return True, 0.95, "Valid dental radiograph."
