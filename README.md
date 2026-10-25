# Viscan Dental OPG AI Assistant

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/surendra08/viscan-opg)
[![Model Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-surendra08%2Fviscan--opg--weights-green)](https://huggingface.co/surendra08/viscan-opg-weights)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

> **Clinical Notice:** AI-assisted findings for dentist review. Not a diagnosis.

**🚀 Live Interactive Demo:** [https://huggingface.co/spaces/surendra08/viscan-opg](https://huggingface.co/spaces/surendra08/viscan-opg)

**Demo video (5–8 min, narrated):** Pending — to be recorded against this build and linked here.

The Space runs this FastAPI app with the Docker SDK on port 7860 and bakes in the 2D detector weights. On that CPU runtime the two pre-segmented CBCT demos and uploaded 5-class label maps render meshes, while a raw CBCT upload returns HTTP 503 because nnU-Net and the Zenodo weights are not installed there (`/health` reports `cbct.ready: false`).

An end-to-end clinical decision-support prototype for automated finding detection in panoramic dental radiographs (OPG). Detects 4 core pathological conditions with original pixel coordinates, confidence scores, deterministic summaries, and side-by-side visual clinical review. The same API also accepts 3D CBCT input; live nnU-Net inference is optional and the service fails closed when it is not installed.

---

## 1. Key Capabilities

- **Core 2D OPG Pathology Detection:** Detects `caries`, `deep_caries`, `periapical_lesion`, and `impacted_tooth` with original pixel coordinates and confidence scores.
- **Bonus 3D CBCT Volumetric Extension:** Accepts raw 3D CBCT (`.nii`, `.nii.gz`, or a ZIP with a 3D DICOM series/NIfTI pair), pre-segmented 5-class NIfTI label maps, and ZIP bundles containing a volume + label map. Raw volumes run **pretrained `DentalSegmentator` (`nnU-Net 3d_fullres` fold_0, CC BY 4.0, used as-is)** when `nnunetv2` and the Zenodo weights are installed; otherwise the request returns HTTP 503 with setup instructions and no demo data is substituted. The service computes the minimum 3D voxel-surface distance in millimetres between lower teeth (label 4) and the mandibular canal (label 5) in each half of the volume, and exports per-upload `.glb` / `.stl` meshes for browser review.
- **Honest 3D Provenance:** Every CBCT response reports `segmentation_source` (`model_inference`, `uploaded_labelmap`, `demo_reference_labelmap`, or `demo_model_labelmap`), whether inference ran in that request, per-stage timings, and process/GPU memory. A 2D OPG response explicitly reports `three_d_available: false`.
- **Multi-Modality Ingestion:** Supports standard images (PNG, JPEG), single-frame 2D DICOM (`.dcm`) with Modality LUT + VOI LUT windowing and MONOCHROME1 polarity inversion, and 3D NIfTI / zipped 3D DICOM series with automatic patient de-identification.
- **Input Validity Gate:** Identifies and rejects non-dental, blank, or text-document uploads before inference.
- **Decision Support API & UI:** Unified FastAPI backend (`POST /analyze` and `POST /analyze/sample_cbct`) and interactive web UI for 2D & 3D clinician review.
- **Deterministic Reporting:** Structured JSON output matching clinical schema requirements (2D `bbox` or 3D `bbox_3d_voxel` / physical `mm` coordinates) with mandatory disclaimer.

---

## 2. Architecture & Design Decisions

- **Detector:** Faster R-CNN with ResNet-50 FPN V2 backbone initialized from COCO pretrained weights (`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`). 43.27M parameters (43.05M trainable).
- **Commercial Licensing:** Permissive **BSD-3-Clause** licensing across PyTorch / torchvision (avoiding the AGPL copyleft viral constraints of Ultralytics YOLO).
- **Evaluation Discipline:** Low score floor (`box_score_thresh=0.001`, `rpn_score_thresh=0.0`) inside detector to ensure unbiased COCO mAP curves before deployment thresholding (selected operating threshold: `0.70`).
- **3D Routing & Provenance:** CBCT routing is content-based (gzip/NIfTI header or ZIP members), not filename-only. Raw volumes require installed weights; pre-segmented label maps are processed directly; demo bundles are identified server-side. A `provenance.json` supplied by an uploader is echoed as **unverified** and can never set `segmentation_source` to `model_inference`.
- **Concurrency & Fail-Closed Behavior:** CBCT jobs are serialized with a process lock because a sliding-window nnU-Net run needs several GB of RAM/VRAM. Missing weights raise HTTP 503 instead of falling back to a sample.

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
docker run -p 8000:8000 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  viscan-opg
```

The image does not contain model weights (`.dockerignore` excludes `*.pth`), so
mount a directory holding `best_model.pth` (see §4) at `/app/checkpoints`. With
that mount the image runs 2D OPG, pre-segmented CBCT label maps, and the demo
bundles; without it `/health` reports `model_loaded: false` and `/analyze`
returns HTTP 503. Raw CBCT inference is opt-in:

```bash
docker build --build-arg INSTALL_CBCT=true -t viscan-opg-cbct .
docker run -p 8000:8000 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  -v /path/to/cbct_weights:/cbct_weights:ro \
  -e VISCAN_CBCT_WEIGHTS_DIR=/cbct_weights \
  viscan-opg-cbct

# Or with the compose profile:
docker compose --profile cbct up viscan-cbct
```

### Option C: Enable Live Raw-CBCT Inference Locally

```bash
pip install -r requirements-cbct.txt

# Download Dataset112_DentalSegmentator_v100.zip from
# https://doi.org/10.5281/zenodo.10829675 (CC BY 4.0) and extract it.
export VISCAN_CBCT_WEIGHTS_DIR=/path/to/extracted/folder   # contains dataset.json + fold_0/checkpoint_final.pth

python app.py
```

Check readiness without uploading a scan:

```bash
curl http://localhost:8000/health
```

`/health` reports `cbct.ready`, `nnunetv2_installed`, `weights_dir_configured`,
`model_folder_found`, and a human-readable `reason`; the `cbct` block does not
echo the weights directory path. When `ready` is false, raw CBCT uploads return HTTP 503; pre-segmented
label maps and demo scans still work.

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

**Response (JSON, abridged — 2 of 8 findings — from `samples/sample_01_val_1_report.json`, measured on CPU):**
```json
{
  "image_id": "sample_01_val_1",
  "modality": "OPG",
  "model": {
    "name": "fasterrcnn_resnet50_fpn_v2",
    "version": "2.0.0"
  },
  "inference_ms": 8301.8,
  "findings": [
    {
      "class": "impacted_tooth",
      "confidence": 0.99,
      "tooth_fdi": null,
      "bbox": [1260.5, 176.1, 1391.4, 301.8]
    },
    {
      "class": "caries",
      "confidence": 0.98,
      "tooth_fdi": null,
      "bbox": [1142.7, 460.4, 1323.1, 663.3]
    }
  ],
  "summary": "8 findings: 5 caries, 3 impacted tooth.",
  "original_image_url": "/samples/sample_01_val_1_input.png",
  "annotated_image_url": "/samples/sample_01_val_1_annotated.png",
  "three_d_available": false,
  "three_d_note": "No 3D model for a 2D OPG: a panoramic radiograph contains no volumetric data. Upload a CBCT volume (.nii, .nii.gz, or a zipped DICOM/NIfTI series) for 3D segmentation.",
  "disclaimer": "AI-assisted findings for dentist review. Not a diagnosis."
}
```

**Size limits and malformed input.** Uploads above `VISCAN_MAX_UPLOAD_MB` (default `1024` MB) are rejected with HTTP `413` before decoding. Gzipped NIfTI decompression is capped at `VISCAN_MAX_NIFTI_DECOMPRESSED_MB` (default `2048` MB) and ZIP archives at `VISCAN_MAX_ZIP_UNCOMPRESSED_MB` (default `4096` MB), so a decompression bomb returns HTTP `400` with a limit message instead of exhausting memory. Wrong file types, corrupt images/DICOMs, and non-dental inputs return structured `4xx` errors; a valid image with no findings returns an empty `findings` list and the summary "No pathological findings detected at current confidence threshold."

### 3D CBCT Analysis (`POST /analyze` or `POST /analyze/sample_cbct`)

The service accepts three CBCT input modes and reports which one produced the labels:

| Input | Label source | `segmentation_source` | Inference in this request |
|---|---|---|---|
| Raw `.nii` / `.nii.gz` / ZIP with a 3D DICOM series or volume-only NIfTI | DentalSegmentator nnU-Net (requires optional stack + weights) | `model_inference` | `true` |
| 5-class NIfTI label map, or ZIP containing a volume + label map | Labels supplied by the uploader | `uploaded_labelmap` | `false` |
| `POST /analyze/sample_cbct?scan=PostDentalSurgery` / `PreDentalSurgery` | Server-side demo bundle | `demo_reference_labelmap` (Post) or `demo_model_labelmap` (Pre) | `false` |

```bash
# Raw volume, live inference (503 if the optional stack is not ready)
curl -X POST "http://localhost:8000/analyze" \
     -F "file=@my_cbct.nii.gz"

# Pre-segmented label map
curl -X POST "http://localhost:8000/analyze" \
     -F "file=@samples/cbct/PostDentalSurgery_Segmentation.nii.gz"

# Bundled demo (server labels its provenance)
curl -X POST "http://localhost:8000/analyze/sample_cbct?scan=PostDentalSurgery"
```

**Example response** (trimmed from [`samples/cbct/PostDentalSurgery_cbct_report.json`](samples/cbct/PostDentalSurgery_cbct_report.json); the mesh geometry and images are generated per request):
```json
{
  "image_id": "PostDentalSurgery",
  "modality": "CBCT",
  "segmentation_source": "demo_reference_labelmap",
  "segmentation_source_detail": "Shipped demo labels are the published SlicerDentalSegmentator reference mask for this scan (https://github.com/gaudot/SlicerDentalSegmentator), after this repository's 60 mm3 island filter was applied to labels 1-4 ...",
  "demo": true,
  "model": {
    "name": "DentalSegmentator_nnUNetv2_3d_fullres",
    "version": "v1.0.0 (Dataset112_DentalSegmentator, Dot et al. 2024)",
    "weights_doi": "10.5281/zenodo.10829675",
    "license": "CC BY 4.0",
    "pretrained_as_is": true,
    "inference_ran_in_this_request": false
  },
  "inference_ms": 9631.3,
  "timings_ms": {
    "load": 2427.9,
    "segmentation": null,
    "postprocess": 2454.9,
    "rendering": 269.0,
    "meshing": 4479.5,
    "total": 9631.3
  },
  "memory": {
    "process_rss_before_mb": 345.7,
    "process_rss_after_mb": 598.0,
    "process_rss_delta_mb": 252.3,
    "gpu_peak_allocated_mb": null
  },
  "volume_metadata": {
    "dimensions_voxels": [360, 360, 330],
    "spacing_mm": [0.5, 0.5, 0.5],
    "has_grayscale_volume": true,
    "segmented_structures": {
      "mandible": {"label_id": 2, "voxel_count": 309895, "volume_cm3": 38.74, "bbox_3d_voxel": [71, 66, 40, 297, 234, 207]},
      "mandibular_canal": {"label_id": 5, "voxel_count": 6655, "volume_cm3": 0.83, "bbox_3d_voxel": [98, 97, 61, 265, 196, 140]}
    }
  },
  "findings": [
    {
      "class": "mandibular_canal_to_lower_teeth_clearance",
      "hemimandible": "right",
      "min_distance_mm": 0.71,
      "closest_lower_tooth_voxel": [119, 153, 90],
      "closest_canal_voxel": [118, 153, 89],
      "closest_lower_tooth_mm": [59.5, 76.5, 45.0],
      "closest_canal_mm": [59.0, 76.5, 44.5],
      "bbox_3d_voxel": [108, 143, 79, 129, 163, 100]
    },
    {
      "class": "mandibular_canal_to_lower_teeth_clearance",
      "hemimandible": "left",
      "min_distance_mm": 1.58,
      "closest_lower_tooth_voxel": [235, 132, 75],
      "closest_canal_voxel": [236, 132, 72],
      "closest_lower_tooth_mm": [117.5, 66.0, 37.5],
      "closest_canal_mm": [118.0, 66.0, 36.0],
      "bbox_3d_voxel": [225, 122, 62, 246, 142, 85]
    }
  ],
  "summary": "CBCT 5-structure segmentation complete (Maxilla, Mandible, Upper Teeth, Lower Teeth, Mandibular Canal). Right lower teeth-to-mandibular canal minimum clearance: 0.71 mm; Left lower teeth-to-mandibular canal minimum clearance: 1.58 mm.",
  "original_image_url": "/results/PostDentalSurgery_cbct_original.png",
  "annotated_image_url": "/results/PostDentalSurgery_cbct_ortho.png",
  "mesh_glb_url": "/results/PostDentalSurgery_scene.glb",
  "mesh_stl_urls": {
    "mandible": "/results/PostDentalSurgery_mandible.stl"
  },
  "three_d_available": true,
  "disclaimer": "AI-assisted findings for dentist review. Not a diagnosis."
}
```

> The timing/memory values above are for a pre-segmented demo bundle on CPU, with
> no nnU-Net stage. External Kaggle Tesla T4 measurements for the full model run
> (131.4 s inference, 4,936 MB peak VRAM) and the Post model-vs-reference Dice
> are recorded in [`samples/cbct/PROVENANCE.md`](samples/cbct/PROVENANCE.md).

#### 3D limitations

- The distance is a minimum voxel-surface Euclidean distance between label 4 and
  label 5 after 60 mm³ island filtering; it inherits the 0.5 mm voxel grid and the
  segmentation error of the label map. It is not a substitute for nerve tracing.
- The demo `PostDentalSurgery` labels are the published reference mask after the
  60 mm³ island filter, not a model prediction; the demo `PreDentalSurgery` labels
  are model output without a reference mask. Both are disclosed per request.
- DICOM ZIP ingestion strips patient tags and keeps only voxels and spacing; it
  sorts slices along the acquisition normal and does not resample oblique gantry
  tilt.
- No claim is made that the two demo volumes belong to the same patient or that
  the Pre/Post volume differences reflect a clinical outcome.

---

## 6. Dataset Preparation & Training

### Prepare Data
Downloads the official DENTEX (MICCAI 2023) benchmark at pinned revision `7b27ccc8e342dcb774f69adc6ca5e6c09fefce93`, creates the deterministic `SEED=42` 3-way split (`635` train / `70` val from `training_data.zip`, and `50` held-out test from `validation_data.zip`), copies images into `images/train`, `images/val`, and `images/test`, and writes COCO JSON annotations (`instances_train.json`, `instances_val.json`, `instances_test.json`). Of DENTEX's 1,005 published images, the 250 challenge-test images have no public annotations and are not downloaded; the working cohort is the 755 labeled images (705 + 50):
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

### Architecture Comparison: RetinaNet
`configs/retinanet.yaml` runs the same data, seed, schedule, and 800×1333 input with RetinaNet ResNet-50 FPN V2 (one-stage, focal loss) instead of Faster R-CNN, so the test metrics isolate the architecture. Get comparable metrics with:
```bash
python train.py --config configs/retinanet.yaml
python evaluate.py --weights checkpoints/retinanet/best_model.pth --config configs/retinanet.yaml --output_dir evaluation_retinanet --skip_variant --select_threshold val
```
`RESEARCH_BENCHMARKS.md` records the verified DENTEX 2023 leaderboard numbers, how the top entries were built, and where this model sits relative to them (with the split differences stated).

### Test-Time Augmentation and WBF Ensemble
```bash
# hflip TTA for any configured detector; also dumps predictions for fusion
python evaluate.py --weights checkpoints/best_model.pth --data_dir data/dentex \
  --output_dir evaluation_tta --skip_variant --select_threshold val \
  --tta hflip --dump_predictions preds_frcnn_tta

# weighted box fusion of two detectors, threshold re-selected on the validation split
python ensemble_wbf.py \
  --gt data/dentex/annotations/instances_test.json --images_dir data/dentex/images/test \
  --preds preds_frcnn/predictions_test.json preds_retinanet/predictions_test.json \
  --val-gt data/dentex/annotations/instances_val.json \
  --val-preds preds_frcnn/predictions_val.json preds_retinanet/predictions_val.json \
  --output_dir evaluation_ensemble
```
`evaluate.py` reports single-pass CPU latency; the ensemble's wall-clock is the sum of its members' latencies and is not measured by the fusion script.

### Sliced Inference at Native Scale
DENTEX images are 1,976–3,139 px wide; full-image inference downscales them by 0.425–0.675× (mean 0.474×), which shrinks small lesions. This script runs the same weights on the full image and on overlapping 800×800 native-scale crops, and writes one COCO dump per view group so WBF can fuse them:
```bash
python slice_inference.py --weights checkpoints/best_model.pth --config configs/baseline.yaml \
  --split val --data_dir data/dentex --output_dir preds_slice --views both \
  --tile 800 --overlap 0.25 --tta hflip --max_dets 100

python ensemble_wbf.py \
  --gt data/dentex/annotations/instances_test.json --images_dir data/dentex/images/test \
  --preds preds_slice/predictions_test_full.json preds_slice/predictions_test_tiles.json \
  --val-gt data/dentex/annotations/instances_val.json \
  --val-preds preds_slice/predictions_val_full.json preds_slice/predictions_val_tiles.json \
  --output_dir evaluation_sliced
```
`slice_meta_<split>.json` records the tile grid, detections per view, and wall-clock per image on the device used. Slicing multiplies forward passes by the tile count (~10 per 2830×1367 image), so it is a benchmark artifact, not a replacement for the deployed single-pass model.

### Cascade Crop Refiner
A ConvNeXt-Tiny (BSD-3-Clause) classifier trained only on train-split GT crops re-scores detector boxes from a 1.3× context crop:
```bash
python cascade_refine.py build --data_dir data/dentex --output_dir checkpoints/cascade
python cascade_refine.py train --bank_dir checkpoints/cascade --output checkpoints/cascade/convnext_cascade.pth
python cascade_refine.py apply --checkpoint checkpoints/cascade/convnext_cascade.pth \
  --preds preds_slice/predictions_val_full.json --split val --data_dir data/dentex \
  --mode rescore --output_dir preds_cascade
```
`rescore` keeps box and label and sets `score = sqrt(det_score × p_label)`; `reclass` takes the argmax over the four findings, drops boxes whose argmax is background, and re-scores. Both variants are measured, not assumed.

### Kaggle Benchmark Notebook
`kaggle_benchmark_notebook.ipynb` runs the architecture comparison in one GPU session: it verifies the held-out split, validates the downloaded Faster R-CNN checkpoint against its published SHA-256, evaluates Faster R-CNN with and without TTA, trains and evaluates RetinaNet, and fuses both with WBF.

`kaggle_beat_benchmark_notebook.ipynb` runs the full ladder above (E1–E5): full-image, hflip TTA, native-scale sliced inference, RetinaNet, WBF over all members, and the crop refiner in both modes. It selects each pipeline on validation mAP, evaluates the test split once, and prints every measured row. Neither notebook reports a metric the run did not produce; no ensemble, sliced-inference, or refiner result is claimed in this repository until a run completes.

---

## 7. Sample Artifacts

- **2D OPG:** `samples/sample_01..05_*` each contain the input PNG, the annotated PNG, and the recorded `/analyze` JSON at threshold `0.70`. Regenerate with `python generate_samples.py --weights checkpoints/best_model.pth --images <five OPGs> --names sample_01_val_1 sample_02_val_10 sample_03_val_11 sample_04_val_14 sample_05_val_21 --output samples`.
- **3D CBCT:** `samples/cbct/` contains the two demo bundles, their MPR previews, 5 STLs, a GLB scene, the recorded demo response JSON, and [`PROVENANCE.md`](samples/cbct/PROVENANCE.md) with label sources, bundle hashes, external Kaggle measurements, and local demo timings. Regenerate with `python build_cbct_bundles.py && python generate_cbct_samples.py`.

---

## 8. Testing

Run automated unit and integration tests:
```bash
pytest tests/ -v
```
The suite covers DICOM windowing/MONOCHROME1 handling, coordinate clipping, API
schemas, NIfTI round-tripping, analytical nerve distances, STL/GLB export, and
the CBCT behavior described above: content-based routing, 2D responses never
advertising 3D, uploaded label maps never claiming inference, uploaded
`provenance.json` being untrusted, server-side demo provenance, DICOM patient-tag
stripping, raw CBCT returning 503 without weights, and a stubbed live-inference
path producing per-upload meshes. The stubbed test proves wiring and provenance,
not nnU-Net accuracy.
