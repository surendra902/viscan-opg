# Viscan Dental OPG AI Assistant

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/surendra08/viscan-opg)
[![Model Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-surendra08%2Fviscan--opg--weights-green)](https://huggingface.co/surendra08/viscan-opg-weights)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

> **Clinical Notice:** AI-assisted findings for dentist review. Not a diagnosis.

**🚀 Live Interactive Demo:** [https://huggingface.co/spaces/surendra08/viscan-opg](https://huggingface.co/spaces/surendra08/viscan-opg)

An end-to-end clinical decision-support prototype for automated finding detection in panoramic dental radiographs (OPG). Detects 4 core pathological conditions with original pixel coordinates, confidence scores, deterministic summaries, and side-by-side visual clinical review.

---

## 1. Key Capabilities

- **Core 2D OPG Pathology Detection:** Detects `caries`, `deep_caries`, `periapical_lesion`, and `impacted_tooth` with original pixel coordinates and confidence scores.
- **Bonus 3D CBCT Volumetric Extension:** Accepts 3D CBCT volumes (`.nii.gz`, `.nii`, or `.zip` 3D DICOM/NIfTI series), segments 5 anatomical structures (`maxilla_upper_skull`, `mandible`, `upper_teeth`, `lower_teeth`, `mandibular_canal`) via **pretrained `DentalSegmentator` (`nnU-Net 3d_fullres`, CC BY 4.0, used as-is)**, computes exact 3D Euclidean millimeter clearance between lower teeth roots and the left/right mandibular canal (`0.71 mm` right / `1.58 mm` left on `PostDentalSurgery`; `2.12 mm` right / `1.50 mm` left on `PreDentalSurgery`), and exports interactive `.glb` / `.stl` 3D meshes rendered directly in the browser (`<model-viewer>` / WebGL).
- **Multi-Modality Ingestion:** Supports standard images (PNG, JPEG), single-frame 2D DICOM (`.dcm`) with Modality LUT + VOI LUT windowing and MONOCHROME1 polarity inversion, and 3D NIfTI / zipped 3D DICOM series with automatic patient de-identification.
- **Input Validity Gate:** Identifies and rejects non-dental, blank, or text-document uploads before inference.
- **Decision Support API & UI:** Unified FastAPI backend (`POST /analyze` and `POST /analyze/sample_cbct`) and interactive web UI for 2D & 3D clinician review.
- **Deterministic Reporting:** Structured JSON output matching clinical schema requirements (2D `bbox` or 3D `bbox_3d_voxel` / physical `mm` coordinates) with mandatory disclaimer.

---

## 2. Architecture & Design Decisions

- **Detector:** Faster R-CNN with ResNet-50 FPN V2 backbone initialized from COCO pretrained weights (`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`). 43.27M parameters (43.05M trainable).
- **Commercial Licensing:** Permissive **BSD-3-Clause** licensing across PyTorch / torchvision (avoiding the AGPL copyleft viral constraints of Ultralytics YOLO).
- **Evaluation Discipline:** Low score floor (`box_score_thresh=0.001`, `rpn_score_thresh=0.0`) inside detector to ensure unbiased COCO mAP curves before deployment thresholding (selected operating threshold: `0.70`).

---

## 3. Quickstart: Local Launch

### Option A: Local Python Environment
Requires Python 3.10+ (tested on 3.11).

```bash
# 1. Clone repository
git clone https://github.com/surendra902/viscan-opg.git
cd viscan-opg

# 2. Create virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download fine-tuned model weights (or use huggingface-cli / curl)
mkdir checkpoints
# Linux / macOS:
curl -L -o checkpoints/best_model.pth "https://huggingface.co/surendra08/viscan-opg-weights/resolve/main/best_model.pth"
# Windows PowerShell:
Invoke-WebRequest -Uri "https://huggingface.co/surendra08/viscan-opg-weights/resolve/main/best_model.pth" -OutFile "checkpoints\best_model.pth"

# 5. Launch web application (single command)
python app.py
```
Open **http://localhost:8000** in your browser.

### Option B: Docker Container
```bash
docker build -t viscan-opg .
docker run -p 8000:8000 viscan-opg
```

---

## 4. Pre-trained Weights & Integrity Verification

| File | Size | SHA256 / Checksum | Repository / Download | License |
|---|---|---|---|---|
| `best_model.pth` (2D OPG Faster R-CNN) | 173,466,789 bytes (173.5 MB) | `e0b49b12f48fc71ec1f583db08f107650659571e7c2b9cbd454dbfed13b46a1f` | [Hugging Face Hub](https://huggingface.co/surendra08/viscan-opg-weights) · [Direct Download](https://huggingface.co/surendra08/viscan-opg-weights/resolve/main/best_model.pth) | **CC BY-NC-SA 4.0** |
| `Dataset112_DentalSegmentator_v100.zip` (3D CBCT nnU-Net, used pretrained as-is) | 229,747,861 bytes (219.1 MB) | MD5: `b71cd5230168d28a4f71b078265b76be` | [Zenodo DOI 10.5281/zenodo.10829675](https://zenodo.org/records/10829675) (Dot et al., *J. Dentistry* 2024) | **CC BY 4.0** |

> **License Disclosure:**
> - The application source code, API service, Dockerfile, and architecture definitions are licensed under the permissive **BSD-3-Clause** license.
> - The fine-tuned weights (`best_model.pth`) were fine-tuned on the DENTEX MICCAI 2023 dataset and inherit the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license. Commercial clinical deployments must retrain the detector using an internal or commercially cleared dataset.

### Verify Checkpoint Integrity:
```bash
# Linux / macOS:
sha256sum checkpoints/best_model.pth
# Expected output:
# e0b49b12f48fc71ec1f583db08f107650659571e7c2b9cbd454dbfed13b46a1f  checkpoints/best_model.pth

# Windows PowerShell:
Get-FileHash checkpoints\best_model.pth -Algorithm SHA256
# Hash: E0B49B12F48FC71EC1F583DB08F107650659571E7C2B9CBD454DBFED13B46A1F
```

---

## 5. API Reference

### `POST /analyze`
Accepts a multipart file upload (`PNG`, `JPG`, or `.dcm`) and returns structured findings.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze?confidence_threshold=0.70" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_opg.png"
```

**Response (JSON):**
```json
{
  "image_id": "opg_a1b2c3d4",
  "modality": "OPG",
  "model": {
    "name": "fasterrcnn_resnet50_fpn_v2",
    "version": "2.0.0"
  },
  "inference_ms": 4618.0,
  "findings": [
    {
      "class": "deep_caries",
      "confidence": 0.84,
      "tooth_fdi": null,
      "bbox": [412.0, 288.0, 470.0, 351.0]
    },
    {
      "class": "periapical_lesion",
      "confidence": 0.76,
      "tooth_fdi": null,
      "bbox": [405.0, 360.0, 480.0, 435.0]
    }
  ],
  "summary": "2 findings: 1 deep caries, 1 periapical lesion.",
  "original_image_url": "/results/opg_a1b2c3d4_original.png",
  "annotated_image_url": "/results/opg_a1b2c3d4_annotated.png",
  "disclaimer": "AI-assisted findings for dentist review. Not a diagnosis."
}
```

### 3D CBCT Analysis (`POST /analyze` or `POST /analyze/sample_cbct`)
Accepts `.nii.gz`, `.nii`, or `.zip` (3D DICOM series or NIfTI bundle) and returns 5-class 3D segmentation statistics, 3D mandibular canal nerve clearance findings (`mm` + 3D voxel bounding boxes), orthogonal slice previews, and interactive `.glb` / `.stl` mesh URLs:
```bash
curl -X POST "http://localhost:8000/analyze" \
     -F "file=@samples/cbct/PostDentalSurgery_Segmentation.nii.gz"
```
**Response (3D CBCT JSON):**
```json
{
  "image_id": "PostDentalSurgery",
  "modality": "CBCT_3D",
  "model": {
    "name": "dentalsegmentator_nnunet_3d_fullres",
    "version": "1.0.0",
    "source": "Pretrained as-is (Dot et al., Journal of Dentistry 2024, Zenodo 10829675, CC BY 4.0)"
  },
  "volume_shape": [360, 360, 330],
  "voxel_spacing_mm": [0.5, 0.5, 0.5],
  "inference_ms": 131407.1,
  "postprocess_ms": 5816.3,
  "mesh_generation_ms": 938.6,
  "memory_mb": {
    "gpu_peak_allocated_mb": 4936.0,
    "cpu_rss_mb": 2641.8
  },
  "findings": [
    {
      "class": "mandibular_canal_to_lower_teeth_clearance",
      "hemimandible": "right",
      "min_distance_mm": 0.71,
      "closest_lower_tooth_voxel": [119, 151, 89],
      "closest_canal_voxel": [118, 151, 88],
      "bbox_3d_voxel": [108, 141, 78, 129, 161, 99]
    },
    {
      "class": "mandibular_canal_to_lower_teeth_clearance",
      "hemimandible": "left",
      "min_distance_mm": 1.58,
      "closest_lower_tooth_voxel": [235, 132, 75],
      "closest_canal_voxel": [236, 132, 72],
      "bbox_3d_voxel": [225, 122, 62, 246, 142, 85]
    }
  ],
  "mesh_glb_url": "/results/PostDentalSurgery_scene.glb",
  "disclaimer": "AI-assisted findings for dentist review. Not a diagnosis."
}
```

---

## 6. Dataset Preparation & Training

### Prepare Data
Downloads the official DENTEX (MICCAI 2023) benchmark, creates the deterministic `SEED=42` 3-way split (`635` train / `70` val from `training_data.zip`, and `50` held-out test from `validation_data.zip`), copies images into `images/train`, `images/val`, and `images/test`, and writes COCO JSON annotations (`instances_train.json`, `instances_val.json`, `instances_test.json`):
```bash
python prepare_data.py data/dentex
```

### Train Detector
```bash
python train.py --config configs/baseline.yaml
```
Outputs saved to `runs/baseline/` (`loss_curve.png`, `val_ap_curve.png`) and `checkpoints/best_model.pth`.

### Evaluate Model
```bash
python evaluate.py --weights checkpoints/best_model.pth --data_dir data/dentex
```
Computes:
- Overall & per-class mAP@50 and mAP@50:95
- Precision, Recall, and F1 at 0.70 confidence threshold
- 5x5 Confusion Matrix with background FN/FP accounting
- Extracts 5 real failure cases to `evaluation_results/failures/`
- Measures CPU latency per image

---

## 7. Testing

Run automated unit and integration tests:
```bash
pytest tests/ -v
```
All tests verify DICOM windowing, MONOCHROME1 handling, model predictor heads, coordinate bounds clipping, and API schemas.
