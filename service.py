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

import importlib.util
import os
import threading
import uuid
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from imaging import load_image_bytes, ImageProcessingError
from input_gate import DentalInputGate, NonDentalInputError
from model import build_model_from_config, load_model_weights, CLASS_NAMES, HAS_TORCHVISION
from cbct_pipeline import (
    is_cbct_bytes,
    load_cbct_input,
    postprocess_and_compute_findings,
    render_cbct_ortho_views,
    export_cbct_meshes,
    parse_nifti_bytes,
    write_nifti_gz_bytes,
    SEGMENTATION_SOURCES,
)

# Consistent visual palette for classes
CLASS_COLORS = {
    "impacted_tooth": "#00BFFF",     # Deep Sky Blue
    "caries": "#FFD700",             # Gold / Amber
    "periapical_lesion": "#BA55D3",  # Medium Orchid / Violet
    "deep_caries": "#FF4500",        # Orange Red
}

DISCLAIMER_TEXT = "AI-assisted findings for dentist review. Not a diagnosis."


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested but the required weights are not loaded."""
    pass


class CbctInferenceError(RuntimeError):
    """Raised when installed DentalSegmentator weights fail during a live CBCT run."""
    pass


def _rss_mb() -> Optional[float]:
    """Current process resident memory in MB, or None if it cannot be measured."""
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            return None


def _find_cbct_model_folder(weights_dir: Path) -> Optional[Path]:
    """
    Finds the nnU-Net model folder that actually contains fold_0/checkpoint_final.pth
    and dataset.json. Zenodo archives contain more than one dataset.json, so picking
    the first match would be wrong.
    """
    candidates = [
        p.parent for p in weights_dir.rglob("dataset.json")
        if (p.parent / "fold_0" / "checkpoint_final.pth").is_file()
    ]
    if not candidates:
        return None
    preferred = [c for c in candidates if "3d_fullres" in str(c).lower()]
    return sorted(preferred or candidates, key=lambda p: len(str(p)))[0]


def _missing_cbct_weights_message() -> str:
    return (
        "Raw CBCT segmentation requires nnunetv2 and the DentalSegmentator weights, which are not "
        "available in this runtime. Install with 'pip install -r requirements-cbct.txt', download "
        "Dataset112_DentalSegmentator_v100 from https://doi.org/10.5281/zenodo.10829675 (CC BY 4.0), "
        "and set VISCAN_CBCT_WEIGHTS_DIR to the extracted folder containing dataset.json and "
        "fold_0/checkpoint_final.pth. No demo data was substituted. Alternatively upload a 5-class "
        "NIfTI label map or a pre-segmented zip bundle."
    )


class DentalAnalysisService:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        results_dir: str = "results",
        confidence_threshold: float = 0.70,
    ):
        if device is None:
            device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if HAS_TORCH else "cpu"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = confidence_threshold
        self.model = None
        self.input_gate = DentalInputGate(device=device)
        # A 3D sliding-window run needs several GB of VRAM/RAM. Serialize CBCT requests
        # so two concurrent uploads cannot exhaust the worker; the queue timeout is HTTP-level.
        self._cbct_lock = threading.Lock()

        if HAS_TORCH and HAS_TORCHVISION and weights_path and os.path.exists(weights_path):
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
        """Runs the complete analysis pipeline on uploaded file bytes (2D OPG or 3D CBCT)."""
        if is_cbct_bytes(file_bytes, filename):
            return self.analyze_cbct_bytes(file_bytes, filename)

        t_start = time.perf_counter()
        image_id = f"opg_{uuid.uuid4().hex[:8]}"
        threshold = custom_threshold if custom_threshold is not None else self.conf_threshold

        # 1. Decode Image & Modality (strips patient metadata)
        img_pil, modality, (orig_w, orig_h) = load_image_bytes(file_bytes, filename)

        # 2. Gate check (Rejects non-dental inputs)
        is_valid, gate_score, gate_msg = self.input_gate.validate(img_pil)
        if not is_valid:
            raise NonDentalInputError(f"Input rejected: {gate_msg}")

        # 3. Model Inference (fail closed if model checkpoint is unavailable)
        if self.model is None or not HAS_TORCH:
            raise ModelNotLoadedError(
                "Detector model weights are not loaded. Provide a valid checkpoint via VISCAN_WEIGHTS_PATH or checkpoints/best_model.pth."
            )

        findings = []
        if transforms is not None:
            tensor = transforms.functional.to_tensor(img_pil).to(self.device)
        else:
            arr = np.asarray(img_pil.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(self.device)
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

        # 4. Save Original (decoded PNG for DICOM preview) and Annotated Image
        original_filename = f"{image_id}_original.png"
        original_filepath = self.results_dir / original_filename
        img_pil.save(original_filepath, format="PNG")

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
            "original_image_url": f"/results/{original_filename}",
            "annotated_image_url": f"/results/{annotated_filename}",
            "three_d_available": False,
            "three_d_note": (
                "No 3D model for a 2D OPG: a panoramic radiograph contains no volumetric data. "
                "Upload a CBCT volume (.nii, .nii.gz, or a zipped DICOM/NIfTI series) for 3D segmentation."
            ),
            "disclaimer": DISCLAIMER_TEXT,
        }

    def analyze_cbct_bytes(
        self,
        file_bytes: bytes,
        filename: str = "",
        provenance_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs 3D CBCT segmentation (live DentalSegmentator when the upload is a raw
        volume and weights are installed), nerve clearance, MPR previews and meshes.
        Any upload gets its own outputs; demo labels are never substituted.
        """
        with self._cbct_lock:
            return self._analyze_cbct_bytes(file_bytes, filename, provenance_override)

    def _analyze_cbct_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        provenance_override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        rss_before = _rss_mb()
        gpu_peak_allocated_mb: Optional[float] = None
        if HAS_TORCH and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        image_id = f"cbct_{uuid.uuid4().hex[:8]}"

        vol_xyz, seg_xyz, spacing_xyz, input_info = load_cbct_input(file_bytes, filename)
        t_loaded = time.perf_counter()

        inference_ran = False
        source = input_info.get("source")
        source_detail = input_info.get("detail") or ""

        if seg_xyz is None:
            cbct_weights_dir = os.getenv("VISCAN_CBCT_WEIGHTS_DIR", "").strip()
            if not cbct_weights_dir or not Path(cbct_weights_dir).is_dir():
                raise ModelNotLoadedError(_missing_cbct_weights_message())
            seg_xyz = self._run_live_nnunet_cbct(vol_xyz, spacing_xyz, Path(cbct_weights_dir))
            inference_ran = True
            source = "model_inference"
            source_detail = (
                "DentalSegmentator nnU-Net 3d_fullres (fold_0, pretrained as-is) executed on this "
                "uploaded volume in this request."
            )
        t_segmented = time.perf_counter()

        if provenance_override:
            override_source = provenance_override.get("source")
            if override_source in SEGMENTATION_SOURCES:
                source = override_source
            if provenance_override.get("detail"):
                source_detail = str(provenance_override["detail"])

        if source not in SEGMENTATION_SOURCES:
            source = "uploaded_labelmap" if seg_xyz is not None else None
        declared = input_info.get("declared_provenance")
        if declared and not inference_ran and not provenance_override:
            source_detail += (
                f" Bundle declaration (unverified): {declared['segmentation_source']}."
            )

        if seg_xyz.shape != vol_xyz.shape:
            message = f"Segmentation shape {seg_xyz.shape} does not match volume shape {vol_xyz.shape}."
            if inference_ran:
                raise CbctInferenceError(message)
            raise ImageProcessingError(message)

        seg_xyz = seg_xyz.astype(np.int16)
        labels = np.unique(seg_xyz)
        if labels.min() < 0 or labels.max() > 5:
            message = f"Segmentation contains labels outside 0..5: {labels.tolist()[:10]}."
            if inference_ran:
                raise CbctInferenceError(message)
            raise ImageProcessingError(message)
        if int(seg_xyz.max()) == 0:
            message = "Segmentation contains no labeled anatomical structures."
            if inference_ran:
                raise CbctInferenceError(message)
            raise ImageProcessingError(message)

        seg_clean, structure_stats, findings = postprocess_and_compute_findings(
            seg_xyz, spacing_xyz, apply_island_filter=True
        )
        t_postprocessed = time.perf_counter()

        # Every upload gets its own slices and its own meshes, computed from its own data.
        orig_filename = f"{image_id}_original.png"
        annot_filename = f"{image_id}_annotated.png"
        render_cbct_ortho_views(
            vol_xyz,
            seg_clean,
            findings,
            self.results_dir / orig_filename,
            self.results_dir / annot_filename,
        )
        t_rendered = time.perf_counter()
        mesh_glb_url, mesh_stl_urls = export_cbct_meshes(
            seg_clean, spacing_xyz, self.results_dir, image_id
        )
        t_meshed = time.perf_counter()

        if findings:
            clearance_parts = [
                f"{f['hemimandible'].title()} lower teeth-to-mandibular canal minimum clearance: {f['min_distance_mm']:.2f} mm"
                for f in findings
            ]
            summary = (
                "CBCT 5-structure segmentation complete (Maxilla, Mandible, Upper Teeth, Lower Teeth, Mandibular Canal). "
                + "; ".join(clearance_parts) + "."
            )
        else:
            summary = "CBCT 5-structure segmentation complete. No lower tooth-to-mandibular canal proximity detected."

        inference_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        rss_after = _rss_mb()
        if HAS_TORCH and self.device.type == "cuda" and torch.cuda.is_available():
            gpu_peak_allocated_mb = round(
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 1
            )

        nx, ny, nz = vol_xyz.shape
        has_grayscale = bool(input_info.get("has_grayscale_volume", True))
        demo = provenance_override is not None

        return {
            "image_id": image_id,
            "modality": "CBCT",
            "segmentation_source": source,
            "segmentation_source_detail": source_detail,
            "demo": demo,
            "model": {
                "name": "DentalSegmentator_nnUNetv2_3d_fullres",
                "version": "v1.0.0 (Dataset112_DentalSegmentator, Dot et al. 2024)",
                "weights_doi": "10.5281/zenodo.10829675",
                "license": "CC BY 4.0",
                "pretrained_as_is": True,
                "inference_ran_in_this_request": inference_ran,
            },
            "inference_ms": inference_ms,
            "timings_ms": {
                "load": round((t_loaded - t_start) * 1000.0, 1),
                "segmentation": (
                    round((t_segmented - t_loaded) * 1000.0, 1) if inference_ran else None
                ),
                "postprocess": round((t_postprocessed - t_segmented) * 1000.0, 1),
                "rendering": round((t_rendered - t_postprocessed) * 1000.0, 1),
                "meshing": round((t_meshed - t_rendered) * 1000.0, 1),
                "total": inference_ms,
            },
            "memory": {
                "process_rss_before_mb": rss_before,
                "process_rss_after_mb": rss_after,
                "process_rss_delta_mb": (
                    round(rss_after - rss_before, 1)
                    if rss_before is not None and rss_after is not None else None
                ),
                "gpu_peak_allocated_mb": gpu_peak_allocated_mb,
            },
            "volume_metadata": {
                "dimensions_voxels": [int(nx), int(ny), int(nz)],
                "spacing_mm": [float(s) for s in spacing_xyz],
                "has_grayscale_volume": has_grayscale,
                "segmented_structures": structure_stats,
            },
            "original_image_note": (
                None if has_grayscale else
                "Synthetic intensity rendering generated from label IDs; no grayscale volume was supplied."
            ),
            "findings": findings,
            "summary": summary,
            "original_image_url": f"/results/{orig_filename}",
            "annotated_image_url": f"/results/{annot_filename}",
            "mesh_glb_url": mesh_glb_url,
            "mesh_stl_urls": mesh_stl_urls,
            "three_d_available": True,
            "disclaimer": DISCLAIMER_TEXT,
        }

    def _run_live_nnunet_cbct(
        self,
        vol_xyz: np.ndarray,
        spacing_xyz: Tuple[float, float, float],
        weights_dir: Path,
    ) -> np.ndarray:
        """Invokes nnUNetPredictor on a 3D CBCT volume when nnunetv2 and Zenodo 10829675 weights are installed."""
        model_folder = _find_cbct_model_folder(weights_dir)
        if model_folder is None:
            raise ModelNotLoadedError(
                "VISCAN_CBCT_WEIGHTS_DIR is set, but no complete nnU-Net model folder was found. "
                "It must contain dataset.json and fold_0/checkpoint_final.pth. "
                + _missing_cbct_weights_message()
            )

        checkpoint_name = "checkpoint_final.pth"
        if not (model_folder / "fold_0" / checkpoint_name).is_file():
            checkpoint_name = "checkpoint_best.pth"
        if not (model_folder / "fold_0" / checkpoint_name).is_file():
            raise ModelNotLoadedError(
                "No checkpoint_final.pth or checkpoint_best.pth found in the model's fold_0 directory. "
                + _missing_cbct_weights_message()
            )

        try:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            os.environ.setdefault("nnUNet_raw", str(weights_dir))
            os.environ.setdefault("nnUNet_preprocessed", str(weights_dir))
            os.environ.setdefault("nnUNet_results", str(weights_dir))
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError as exc:
            raise ModelNotLoadedError(
                f"nnunetv2 is not installed in this runtime ({exc}). "
                + _missing_cbct_weights_message()
            ) from exc

        import tempfile
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_p = Path(tmp)
                in_file = tmp_p / "case_0000.nii.gz"
                out_file = tmp_p / "case.nii.gz"
                in_file.write_bytes(write_nifti_gz_bytes(vol_xyz, spacing_xyz))

                predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=False,
                    perform_everything_on_device=(self.device.type == "cuda"),
                    device=self.device,
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=False,
                )
                predictor.initialize_from_trained_model_folder(
                    str(model_folder),
                    use_folds=(0,),
                    checkpoint_name=checkpoint_name,
                )
                predictor.predict_from_files(
                    [[str(in_file)]],
                    [str(out_file)],
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=1,
                    num_processes_segmentation_export=1,
                )
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)

                if not out_file.is_file():
                    raise CbctInferenceError("DentalSegmentator did not produce a segmentation file.")
                seg_arr, _ = parse_nifti_bytes(out_file.read_bytes(), is_gzipped=True)
        except (ModelNotLoadedError, CbctInferenceError):
            raise
        except Exception as exc:
            raise CbctInferenceError(f"DentalSegmentator inference failed: {str(exc)[:500]}") from exc

        if seg_arr.shape != vol_xyz.shape:
            raise CbctInferenceError(
                f"DentalSegmentator returned shape {seg_arr.shape}, expected {vol_xyz.shape}."
            )
        labels = np.unique(seg_arr)
        if labels.min() < 0 or labels.max() > 5:
            raise CbctInferenceError(
                f"DentalSegmentator returned labels outside 0..5: {labels.tolist()[:10]}."
            )
        return seg_arr.astype(np.uint8)

    def cbct_readiness(self) -> Dict[str, Any]:
        """Reports whether live 3D inference can run here, without exposing filesystem paths."""
        try:
            nnunet_installed = importlib.util.find_spec("nnunetv2") is not None
        except (ImportError, ValueError):
            nnunet_installed = False
        weights_env = os.getenv("VISCAN_CBCT_WEIGHTS_DIR", "").strip()
        weights_dir = Path(weights_env) if weights_env else None
        weights_exists = bool(weights_dir and weights_dir.is_dir())
        model_folder_found = bool(weights_exists and _find_cbct_model_folder(weights_dir))
        ready = bool(nnunet_installed and weights_exists and model_folder_found)

        if ready:
            reason = "Live DentalSegmentator inference is available."
        elif not nnunet_installed and not weights_exists:
            reason = "nnunetv2 and DentalSegmentator weights are not installed; raw CBCT uploads return HTTP 503."
        elif not nnunet_installed:
            reason = "nnunetv2 is not installed; raw CBCT uploads return HTTP 503."
        elif not weights_exists:
            reason = "VISCAN_CBCT_WEIGHTS_DIR is not set to an existing folder; raw CBCT uploads return HTTP 503."
        else:
            reason = "No complete nnU-Net model folder (dataset.json + fold_0 checkpoint) was found; raw CBCT uploads return HTTP 503."

        return {
            "ready": ready,
            "nnunetv2_installed": nnunet_installed,
            "weights_dir_configured": weights_exists,
            "model_folder_found": model_folder_found,
            "device": str(self.device),
            "reason": reason,
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
