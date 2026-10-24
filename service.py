"""
Unified Inference Service for Viscan OPG.

Orchestrates:
1. Input decoding via imaging.py (handles DICOM windowing/MONOCHROME1 & PNG/JPEG)
2. Dental validity checking via input_gate.py
3. Detector inference on CPU/GPU
4. Coordinates preserved in original image pixel space
5. Class-colored annotations with class label and confidence
6. Deterministic human-readable summary
7. Strict JSON output schema with required disclaimer
"""

import os
import uuid
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from imaging import load_image_bytes, ImageProcessingError
from input_gate import DentalInputGate, NonDentalInputError
from model import build_model_from_config, load_model_weights, CLASS_NAMES

# Consistent visual palette for classes
CLASS_COLORS = {
    "impacted_tooth": "#00BFFF",     # Deep Sky Blue
    "caries": "#FFD700",             # Gold / Amber
    "periapical_lesion": "#BA55D3",  # Medium Orchid / Violet
    "deep_caries": "#FF4500",        # Orange Red
}

DISCLAIMER_TEXT = "AI-assisted findings for dentist review. Not a diagnosis."


class DentalAnalysisService:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        results_dir: str = "results",
        confidence_threshold: float = 0.70,
    ):
        self.device = torch.device(device) if HAS_TORCH else "cpu"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = confidence_threshold
        self.model = None
        self.input_gate = DentalInputGate(device=device)

        if HAS_TORCH and weights_path and os.path.exists(weights_path):
            cfg = config or {
                "model": {
                    "name": "fasterrcnn_resnet50_fpn_v2",
                    "min_size": 800,
                    "max_size": 1333,
                    "box_score_thresh": 0.05,
                }
            }
            self.model = build_model_from_config(cfg, pretrained=False)
            load_model_weights(self.model, weights_path, device=self.device)
            self.model.eval()

    def analyze_bytes(
        self,
        file_bytes: bytes,
        filename: str = "",
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs the complete analysis pipeline on uploaded file bytes."""
        t_start = time.perf_counter()
        image_id = f"opg_{uuid.uuid4().hex[:8]}"
        threshold = custom_threshold if custom_threshold is not None else self.conf_threshold

        # 1. Decode Image & Modality (strips patient metadata)
        img_pil, modality, (orig_w, orig_h) = load_image_bytes(file_bytes, filename)

        # 2. Gate check (Rejects non-dental inputs)
        is_valid, gate_score, gate_msg = self.input_gate.validate(img_pil)
        if not is_valid:
            raise NonDentalInputError(f"Input rejected: {gate_msg}")

        # 3. Model Inference
        findings = []
        if self.model is not None and HAS_TORCH:
            tensor = transforms.functional.to_tensor(img_pil).to(self.device)
            with torch.no_grad():
                outputs = self.model([tensor])
                out = outputs[0]

            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                score_val = float(score)
                if score_val < threshold:
                    continue

                label_int = int(label)
                class_name = CLASS_NAMES.get(label_int, f"finding_{label_int}")
                if class_name == "background":
                    continue

                # Ensure box stays within original image bounds
                x_min = max(0.0, min(float(orig_w), float(box[0])))
                y_min = max(0.0, min(float(orig_h), float(box[1])))
                x_max = max(0.0, min(float(orig_w), float(box[2])))
                y_max = max(0.0, min(float(orig_h), float(box[3])))

                if x_max > x_min and y_max > y_min:
                    findings.append({
                        "class": class_name,
                        "confidence": round(score_val, 2),
                        "tooth_fdi": None,
                        "bbox": [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)],
                    })

        # 4. Generate Annotated Image
        annotated_img = self._draw_annotations(img_pil, findings)
        annotated_filename = f"{image_id}_annotated.png"
        annotated_filepath = self.results_dir / annotated_filename
        annotated_img.save(annotated_filepath, format="PNG")

        # 5. Deterministic Human-Readable Summary
        summary = self._generate_summary(findings)

        inference_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

        return {
            "image_id": image_id,
            "modality": modality,
            "model": {
                "name": "fasterrcnn_resnet50_fpn_v2",
                "version": "2.0.0",
            },
            "inference_ms": inference_ms,
            "findings": findings,
            "summary": summary,
            "annotated_image_url": f"/results/{annotated_filename}",
            "disclaimer": DISCLAIMER_TEXT,
        }

    def _draw_annotations(self, img: Image.Image, findings: List[Dict[str, Any]]) -> Image.Image:
        """Draws bounding boxes, labels, and confidence overlays on image."""
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        w, h = draw_img.size

        # Dynamic stroke width based on image dimensions
        stroke_width = max(2, int(min(w, h) / 300))

        for item in findings:
            cls_name = item["class"]
            conf = item["confidence"]
            bbox = item["bbox"]
            color = CLASS_COLORS.get(cls_name, "#00FF00")

            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline=color, width=stroke_width)

            # Label banner
            text = f"{cls_name} {conf:.2f}"
            font_size = max(12, int(min(w, h) / 50))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((x0, y0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_y0 = max(0, y0 - text_h - 4)
            draw.rectangle([x0, label_y0, x0 + text_w + 6, label_y0 + text_h + 4], fill=color)
            draw.text((x0 + 3, label_y0 + 2), text, fill="#000000", font=font)

        return draw_img

    def _generate_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Derives a concise, deterministic summary from detected findings."""
        if not findings:
            return "No pathological findings detected at current confidence threshold."

        counts = {}
        for f in findings:
            c = f["class"]
            counts[c] = counts.get(c, 0) + 1

        parts = [f"{n} {c.replace('_', ' ')}" for c, n in sorted(counts.items())]
        total = len(findings)
        findings_str = "finding" if total == 1 else "findings"
        return f"{total} {findings_str}: {', '.join(parts)}."
