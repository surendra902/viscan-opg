# Viscan Technical Report: Automated Dental Finding Detection in Panoramic Radiographs (OPG)

**Candidate:** Surendra Goud | AI Engineer (Round 2 Take-Home Assessment)  
**Target Modality:** 2D Panoramic Radiograph (OPG) + Bonus 3D CBCT (NIfTI / 3D DICOM Series)  
**Deliverable Type:** Technical Report & Model Evaluation  

> **Clinical Disclaimer:** AI-assisted findings for dentist review. Not a diagnosis.

---

## 1. Executive Summary

This project delivers an end-to-end clinical decision-support prototype for **2D dental panoramic radiographs (OPGs)** and extends the same unified application (`POST /analyze`) to **3D Cone-Beam Computed Tomography (CBCT)** volumes (`.nii.gz`, `.nii`, and zipped 3D DICOM/NIfTI series). For 2D OPGs, the system ingests PNG, JPEG, and single-frame 2D DICOM (`.dcm`) formats, detects four dental finding classes (`impacted_tooth`, `caries`, `periapical_lesion`, `deep_caries`), and produces class-colored visual annotations and deterministic structured JSON reports. For 3D CBCT scans, the system segments 5 maxillofacial structures (`maxilla_upper_skull`, `mandible`, `upper_teeth`, `lower_teeth`, `mandibular_canal`), computes exact 3D physical millimeter clearance between the lower teeth roots and the left/right mandibular canal (`inferior alveolar nerve`), renders 4-panel multi-planar reconstructions (Axial, Coronal, Sagittal, 3D MIP), and exports interactive 3D `.glb` / `.stl` surface meshes for in-browser review.

The detector is **Faster R-CNN with a ResNet-50 FPN V2 backbone** (`43,271,528` total parameters; `43,046,184` trainable), initialized from official Torchvision `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` weights and fine-tuned on the DENTEX (MICCAI 2023) benchmark.
- **Code & Pretrained Backbone License:** **BSD-3-Clause** (permissive, commercial-compatible software architecture).
- **Fine-Tuned Checkpoint License:** **CC BY-NC-SA 4.0** (inherited from the DENTEX dataset; restricted to non-commercial research and evaluation unless retrained on commercially licensed clinical data).
- **Live Demonstration:** [https://huggingface.co/spaces/surendra08/viscan-opg](https://huggingface.co/spaces/surendra08/viscan-opg)
- **Model Checkpoints:** [https://huggingface.co/surendra08/viscan-opg-weights](https://huggingface.co/surendra08/viscan-opg-weights) (`best_model.pth`, `173,466,789` bytes, SHA-256: `e0b49b12f48fc71ec1f583db08f107650659571e7c2b9cbd454dbfed13b46a1f`)

---

## 2. Data Card

### 2.1 Dataset Sourcing & Justification

| Dataset | Modality | Annotations | License | Decision & Rationale |
|---|---|---|---|---|
| **DENTEX (MICCAI 2023)** | OPG | 1,005 OPGs (`705` disease-labeled train pool + `50` official validation + `250` test); Quadrants, FDI enumeration, and 4 disease classes (`Impacted`, `Caries`, `Periapical Lesion`, `Deep Caries`) | CC BY-NC-SA 4.0 | **Selected.** Directly provides expert bounding-box annotations for all 4 required clinical finding classes. |
| **Zhou et al. (2024)** (*BMC Oral Health*) | OPG | 2,066 images (`5,264` anomalous teeth); 7 tooth-state labels (`tooth without anomalies`, `fillings`, `root canal treatment`, `crown`, `caries`, `residual root`, `RCT + crown`) | CC BY 4.0 | **Excluded.** Focuses on restorative tooth states; merges all caries into a single label without distinguishing superficial `caries` from `deep_caries`, and omits `periapical_lesion` and `impacted_tooth`. |
| **Tufts Dental Database** | OPG | 1,000 images; tooth masks and broad abnormality descriptors | Restricted institutional access | **Excluded.** Restricted access terms and generic abnormality categories do not map deterministically to the 4 required pathology classes. |

### 2.2 Dataset Statistics & Deterministic 3-Way Split

- **Source Repository:** Hugging Face `ibrahimhamamci/DENTEX`
- **Labeled Disease Cohort:** `705` images (`3,529` boxes) from `train_quadrant_enumeration_disease.json` + `50` images (`182` boxes) from `validation_triple.json` = `755` images (`3,711` boxes) total.
- **Deterministic 3-Way Split (`seed=42`):** Sorted image IDs from the 705-image training pool were shuffled with `random.Random(42)` to carve out `70` validation images (`339` boxes) for checkpoint selection and threshold tuning, leaving `635` training images (`3,190` boxes). The official `50`-image validation set (`182` boxes) was held out strictly as the **test set**.

| Split | Images | % of 755 Cohort | % of 705 Train Archive | Total Finding Boxes | Role |
|---|:---:|:---:|:---:|:---:|---|
| **Train** | **635** | **84.11%** | 90.07% | **3,190** | Model training & gradient updates |
| **Val** | **70** | **9.27%** | 9.93% | **339** | Checkpoint selection & threshold sweep |
| **Test** | **50** | **6.62%** | — *(Official Val)* | **182** | **Strictly held-out**, evaluated after freezing weights & threshold |
| **Total** | **755** | **100.00%** | — | **3,711** | Complete labeled disease cohort |

#### Per-Class Annotation Counts Across Splits (`seed=42`)

| Class ID | Finding Class | Train (635 imgs) | Val (70 imgs) | Test (50 imgs) | Total (755 imgs) |
|:---:|---|---:|---:|---:|---:|
| `1` | `impacted_tooth` | 545 | 59 | 40 | 644 |
| `2` | `caries` | 2,004 | 185 | 101 | 2,290 |
| `3` | `periapical_lesion` | 144 | 14 | 9 | 167 |
| `4` | `deep_caries` | 497 | 81 | 32 | 610 |
| — | **Total Boxes** | **3,190** | **339** | **182** | **3,711** |

- **Leakage Verification:** Filename intersection across all three splits was programmatically verified as empty (`len(train_fn ∩ val_fn) == 0`, `len(train_fn ∩ test_fn) == 0`, `len(val_fn ∩ test_fn) == 0`). Labels are joined strictly via `annotations[].image_id -> images[].id -> images[].file_name`. Note: DENTEX provides no patient identifiers in its public metadata, so patient-level deduplication cannot be verified beyond disjoint filenames.
- **Image Dimensions:** Widths span `1,976`–`3,139` px (mean `2,830` px); heights span `976`–`1,536` px (mean `1,367` px). With `min_size=800, max_size=1333`, the long-side cap binds on panoramic aspect ratios, yielding scale factors of `0.425`–`0.675` (mean `0.474`).

### 2.3 Preprocessing, DICOM Physics & Input Gating
- **Coordinate Clipping:** Source COCO bounding boxes `[x, y, w, h]` are converted to `[x_min, y_min, x_max, y_max]` by computing `x_max = min(W, x + w)` and `y_max = min(H, y + h)` prior to clamping `x_min = max(0, x)` and `y_min = max(0, y)`, preventing negative-origin box shifting.
- **DICOM Ingestion:** Single-frame 2D monochrome DICOMs undergo Modality LUT rescaling (`apply_modality_lut`: `pixel * RescaleSlope + RescaleIntercept`) **prior** to VOI LUT / `WindowCenter`–`WindowWidth` windowing (`apply_voi_lut`), followed by `[0, 255]` normalization, `MONOCHROME1` polarity inversion, and complete stripping of patient tags.
- **Input Gate:** Rejects empty/corrupt files, multiframe or non-monochrome DICOMs, portrait aspect ratios (`< 0.65`), chromatic photographs (`mean RGB channel diff > 25`), low-contrast/blank inputs (`std < 8`), and bimodal/white-background text documents or screenshots (`near-white fraction > 0.35`, `mid-tone fraction < 0.25`, or `top-2 histogram bin mass > 0.60`).
- **Class Imbalance:** In the 635-image training split, `caries` (`2,004` boxes) outnumbers `periapical_lesion` (`144` boxes) by `13.9:1` (`22.3:1` in image-level support on the test set, where `n = 9`). No class-weighted loss or oversampling was applied in the baseline run; this is documented as a primary limitation.

---

## 3. Model Architecture & Comparative Licensing Analysis

| Evaluation Criterion | Faster R-CNN ResNet-50 FPN V2 (Selected) | Ultralytics YOLO11n (Alternative) |
|---|---|---|
| **Base Architecture** | Two-stage region-proposal detector with Feature Pyramid Network (FPN) | Single-stage anchor-free detector |
| **Pretrained Weights** | Torchvision `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` | Ultralytics COCO pretrained weights (`yolo11n.pt`) |
| **Parameter Count** | `43,271,528` total (`43,046,184` trainable) | `~2.6M` parameters |
| **Software & Backbone License** | **BSD-3-Clause** ([Torchvision License](https://github.com/pytorch/vision/blob/main/LICENSE)) | **AGPL-3.0** ([Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)) |
| **Fine-Tuned Checkpoint License** | **CC BY-NC-SA 4.0** (from DENTEX training data) | **AGPL-3.0 + CC BY-NC-SA 4.0** |
| **Commercial Shipping Posture** | Code and pretrained initialization are cleared for commercial use; fine-tuned weights require retraining on commercial clinical data | Requires open-sourcing proprietary service code under AGPL-3.0 or purchasing an enterprise license, plus retraining on commercial data |

---

## 4. Training Methodology

- **Determinism:** Seed fixed to `42` across `random`, `numpy`, `torch`, `torch.cuda`, and DataLoader generator/workers (`cudnn.deterministic = True`).
- **Hyperparameters (`configs/baseline.yaml`):**
  - Optimizer: `SGD` (`lr = 0.005`, `momentum = 0.9`, `weight_decay = 0.0005`).
  - Scheduler: **`CosineAnnealingLR`** (`T_max = 12`, `eta_min = 0.00005`).
  - Mixed Precision: `AMP (FP16)` enabled on Kaggle Tesla T4 GPU (16 GB VRAM).
  - Augmentation: Random horizontal flip (`p = 0.5`) + radiographic `ColorJitter` (`brightness = 0.15, contrast = 0.15`).
  - Batch Size & Scale: `batch_size = 2`, `min_size = 800`, `max_size = 1333`.
  - Epochs: `12` epochs. Best validation checkpoint saved at **Epoch 8 (`Val mAP@50 = 0.570`, `Val mAP@50:95 = 0.360`, `Val mAP@75 = 0.377`, `Val AR@100 = 0.665`)**.
- **Detector Inference Policy:** Low internal score floor (`box_score_thresh = 0.001`, `rpn_score_thresh = 0.0`, `box_detections_per_img = 300`, `box_nms_thresh = 0.5`) to preserve full precision-recall curves during COCO evaluation prior to applying the validation-tuned operating threshold (`0.70`).

![Training Curves](assets/eval/training_curves.png)

---

## 5. Evaluation & Results

Final evaluation was executed on the **held-out 50-image test set** (`182` ground-truth boxes across `4` classes) using the Epoch 8 checkpoint selected on the 70-image validation split.

### 5.1 Primary Object Detection Performance (COCO Metrics on Held-Out Test Set)

| Metric | Score | Technical Interpretation |
|---|:---:|---|
| **mAP @ IoU=0.50** | **49.6% (0.496)** | Mean Average Precision (area under the precision-recall curve averaged equally across all 4 classes) at `IoU >= 0.50`. |
| **mAP @ IoU=0.50:0.95** (Primary) | **32.9% (0.329)** | Primary COCO mAP averaged across 10 IoU thresholds (`0.50` to `0.95`, step `0.05`). |
| **mAP @ IoU=0.75** (Strict) | **38.3% (0.383)** | Mean Average Precision under strict spatial overlap (`IoU >= 0.75`). |
| **AR @ 100 detections** | **68.3% (0.683)** | Average Recall across IoU `0.50:0.95` given up to 100 candidate detections per radiograph (`AR@1 = 0.243`, `AR@10 = 0.641`). |

### 5.2 Per-Class COCO Benchmark Performance (Held-Out Test Set)

| Pathology Class | COCO AP@50 | COCO AP@50:95 | Test Support ($n$) | Empirical Behavior |
|---|:---:|:---:|:---:|---|
| **Impacted Tooth** | **83.3% (0.833)** | **54.1% (0.541)** | 40 | Strongest class performance; distinct high-contrast tooth morphology and large bounding-box area. |
| **Deep Caries** | **45.7% (0.457)** | **30.2% (0.302)** | 32 | Moderate ranking precision; frequently confused with superficial `caries` at the dentin/pulp boundary. |
| **Caries** | **35.8% (0.358)** | **24.4% (0.244)** | 101 | High false-positive rate on low-contrast interproximal surfaces and restoration margins. |
| **Periapical Lesion** | **33.6% (0.336)** | **23.0% (0.230)** | 9 | Under-represented minority class (`144` train boxes, `9` test boxes); low sensitivity at high confidence thresholds. |

### 5.3 Operating Point Metrics & Confusion Matrix (@ Threshold = 0.70)

A confidence threshold sweep across `[0.10, 0.90]` (step `0.05`) on the **70-image validation set** selected **$\tau = 0.70$**, where validation Macro-F1 peaked at **`0.556`**. Applied once to the **50-image held-out test set** (`IoU >= 0.50` greedy matching), $\tau = 0.70$ yields a test **Macro-F1 of `0.402` (`0.4016`)**:

| Class | Precision | Recall | F1-Score | TP | FP | FN | Test Support |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| **Impacted Tooth** | **76.1%** | **87.5%** | **0.814** | 35 | 11 | 5 | 40 |
| **Caries** | **27.7%** | **53.5%** | **0.365** | 54 | 141 | 47 | 101 |
| **Deep Caries** | **42.9%** | **18.8%** | **0.261** | 6 | 8 | 26 | 32 |
| **Periapical Lesion** | **33.3%** | **11.1%** | **0.167** | 1 | 2 | 8 | 9 |

#### Test Set Confusion Matrix (5×5)
| Ground Truth \ Predicted | Background (FN) | Impacted Tooth | Caries | Periapical Lesion | Deep Caries |
|---|:---:|:---:|:---:|:---:|:---:|
| **Background (False Positives)** | — | 10 | **124** | 2 | 6 |
| **Impacted Tooth** | 5 | **35 (87.5%)** | 0 | 0 | 0 |
| **Caries** | 45 | 1 | **54 (53.5%)** | 0 | 1 |
| **Periapical Lesion** | 4 | 0 | 3 | **1 (11.1%)** | 1 |
| **Deep Caries** | 12 | 0 | **14 (43.8%)** | 0 | **6 (18.8%)** |

- **Accounting Verification:** Total ground-truth boxes = `40 + 101 + 9 + 32 = 182`. Correct class detections (`TP`) = `35 + 54 + 1 + 6 = 96`. Unmatched ground-truth boxes (`Background FN`) = `5 + 45 + 4 + 12 = 66`. Inter-class misclassifications = `20` (each counted as `1 FN` for the true class and `1 FP` for the predicted class). Background false positives = `10 + 124 + 2 + 6 = 142`.
- **Error Pattern 1 (Severity Grading Ambiguity):** `14` of the `20` inter-class misclassifications are `deep_caries` predicted as `caries`. While only `6/32` (`18.8%`) of `deep_caries` boxes are labeled `deep_caries` at $\tau = 0.70$, another `14/32` (`43.8%`) are localized at `IoU >= 0.50` but classified as `caries` (leaving `12/32` completely missed).
- **Error Pattern 2 (Low Sensitivity on Rare/Subtle Pathologies):** At $\tau = 0.70$, `periapical_lesion` recall is `11.1%` (`1/9` detected, `3` misclassified as `caries`, `1` as `deep_caries`, `4` missed) and `caries` precision is `27.7%` (`124` background false alarms). This confirms the model is a research decision-support prototype rather than a standalone diagnostic device.

![Test Set Confusion Matrix Heatmap](assets/eval/confusion_matrix.png)

### 5.4 CPU Inference Latency Benchmark

Measured end-to-end (file read + PNG decode + tensor conversion + ResNet-50 FPN V2 forward pass + NMS at threshold `0.70`) on a real full-resolution test radiograph (`val_15.png`, $2892 \times 1316$) over `5` warmup runs and `30` timed runs with `torch.get_num_threads() = 2` on a Kaggle Intel(R) Xeon(R) CPU @ 2.00GHz:
- **Mean Latency:** `4,618.0 ms` (`4.62 s`)
- **Median (P50):** `4,690.6 ms` (`4.69 s`)
- **P95 Latency:** `4,915.7 ms` (`4.92 s`)
- **Min / Max Latency:** `4,239.9 ms` / `5,037.2 ms`
- **Throughput:** `0.22 images/sec`
- **Peak RSS Memory:** `2,926 MB` (`2.93 GB`)

---

## 6. Improvement Experiment: Higher-Resolution Training Variant (960×1600)

Per Requirement R14, a second model variant was trained for `12` epochs on the same `635`-image training split (`seed = 42`) with increased FPN input resolution (`min_size = 960, max_size = 1600` vs baseline `min_size = 800, max_size = 1333`).

> **Methodological Confound Disclosure:** Fitting `960×1600` on a 16 GB Tesla T4 GPU required reducing `batch_size` from `2` to `1` without gradient accumulation, doubling the number of SGD steps per epoch at the same base learning rate (`0.005`). Thus, this experiment reflects the combined effect of `1.2×` spatial resolution and `batch_size = 1` gradient variance.

| Metric | Baseline (`800×1333`, `bs=2`) | Variant (`960×1600`, `bs=1`) | Delta (`Variant - Baseline`) | Analysis |
|---|:---:|:---:|:---:|---|
| **Val Best mAP@50:95** (Primary Val) | **0.360** (Epoch 8) | 0.354 (Epoch 5) | -0.006 | Baseline achieves higher primary COCO validation mAP@50:95. |
| **Val Best mAP@50** | 0.570 (Epoch 8) | **0.578** (Epoch 5) | +0.008 | Variant peaks slightly higher on loose Val IoU=0.50 at Epoch 5 before overfitting. |
| **Test mAP@50** | **0.496** | 0.493 | -0.003 | Virtually tied overall at IoU=0.50 (-0.3% AP). |
| **Test mAP@50:95** (Primary Test) | **0.329** | 0.313 | -0.016 | Baseline generalizes better across multi-IoU thresholds (+1.6% AP). |
| **Test mAP@75** (Strict) | 0.383 | **0.410** | **+0.027** | Higher resolution improves tight boundary localization (+2.7% AP). |
| **Test AR@100** | **0.683** | 0.681 | -0.002 | Equivalent maximum recall (-0.2% AR). |

#### Per-Class Test AP@50 Comparison

| Class | Baseline AP@50 | Variant AP@50 | Delta |
|---|:---:|:---:|:---:|
| `impacted_tooth` | 0.833 | **0.915** | **+0.082** |
| `caries` | 0.358 | **0.366** | **+0.008** |
| `periapical_lesion` | **0.336** | 0.328 | -0.009 |
| `deep_caries` | **0.457** | 0.365 | -0.092 |

**Checkpoint Selection Rationale (`R11`):** Selecting on the primary COCO validation metric (**`Val mAP@50:95 = 0.360` at Epoch 8 for Baseline vs `0.354` at Epoch 5 for Variant**), the **Baseline (`800×1333`, `batch_size=2`)** checkpoint was chosen as the deployed model. While the `960×1600` variant improved `mAP@75` (`0.410` vs `0.383`) and `impacted_tooth` AP@50 (`0.915` vs `0.833`), its `batch_size = 1` training schedule degraded `deep_caries` AP@50 (`0.365` vs `0.457`) and overall `mAP@50:95` (`0.313` vs `0.329`) while increasing inference cost.

---

## 7. Concrete Failure Case Analysis (Held-Out Test Set)

Across the 50 held-out test radiographs at $\tau = 0.70$, the detector produced `142` False Positives across `44` images, `66` False Negatives across `33` images, and `20` Inter-Class Misclassifications. Five concrete, image-verified failure cases (maximum one per radiograph) were extracted in Cell 9 of [`kaggle_notebook.ipynb`](kaggle_notebook.ipynb):

| Case | Test Image | Failure Type | Ground Truth vs. Prediction | Root-Cause Hypothesis |
|:---:|---|---|---|---|
| **1** | `val_4.png` | **False Positive** | **Pred:** `impacted_tooth` (`score = 0.99`), no matching GT (`IoU = 0.00`) | Posterior third-molar crown/root angulation and bone density overlap at the mandibular ramus triggered a high-confidence impaction false alarm where the reference annotator did not label pathology. |
| **2** | `val_38.png` | **False Negative** | **Missed GT:** `periapical_lesion` (no detection with `IoU >= 0.50`) | Severe class imbalance (`144` train boxes; `13.9:1` ratio vs caries) combined with subtle, low-contrast apical radiolucency blending into surrounding cancellous bone caused the proposal score to fall below `0.70`. |
| **3** | `val_3.png` | **Misclassification** | **GT:** `deep_caries` $\rightarrow$ **Pred:** `caries` (`score = 0.97`, `IoU >= 0.50`) | The detector accurately localized the carious lesion (`score = 0.97`) but predicted superficial `caries` rather than `deep_caries` due to ambiguous radiolucent contrast near the pulp chamber (`14/20` inter-class errors follow this exact pair). |
| **4** | `val_33.png` | **False Negative** | **Missed GT:** `caries` (no detection with `IoU >= 0.50`) | Small interproximal carious lesion attenuated by adjacent premolar/molar enamel overlap and `0.47×` image downscaling (`2830×1367` $\rightarrow$ `1333×644`). |
| **5** | `val_13.png` | **False Positive** | **Pred:** `impacted_tooth` (`score = 0.99`), no matching GT (`IoU = 0.00`) | Unerupted or obliquely positioned molar anatomy flagged with `0.99` confidence; DENTEX disease annotations only mark clinically symptomatic or selected impactions, creating label noise at the posterior arch. |

![Five Concrete Failure Cases (Green = Ground Truth, Red = Predictions)](assets/eval/failure_cases.png)

![Sample Test Detections (Green = Ground Truth, Red = Predictions)](assets/eval/sample_detections.png)

---

## 8. Licensing & Commercial Suitability

| Component | Upstream Origin | License | Commercial Shipping Suitability |
|---|---|---|---|
| **Repository Code & API** | Viscan OPG submission | **BSD-3-Clause** | Cleared for commercial SaaS, on-prem, and device integration. |
| **Faster R-CNN Architecture & `COCO_V1` Pretrained Weights** | PyTorch / Torchvision | **BSD-3-Clause** | Cleared for commercial use. |
| **DENTEX Dataset & Fine-Tuned Weights (`best_model.pth`)** | Hamamci et al. (MICCAI 2023) | **CC BY-NC-SA 4.0** | **Non-commercial research & evaluation only.** Commercial deployment requires retraining the BSD-3-Clause codebase on commercially licensed clinical data or obtaining a commercial license from the DENTEX authors. |
| **FastAPI / Gradio / Pydicom / Pillow** | Open Source | **MIT / Apache-2.0** | Cleared for commercial use. |
| **DentalSegmentator (`Dataset112_DentalSegmentator_v100`)** | Dot et al. (*Journal of Dentistry* 2024, Zenodo `10.5281/zenodo.10829675`) | **CC BY 4.0** | **Pretrained model used as-is** for Bonus 3D CBCT track; permissive attribution license (`CC BY 4.0`). |

---

## 9. Limitations & Clinical Guardrails

1. **Low Sensitivity on `periapical_lesion` and `deep_caries` at $\tau = 0.70$:** Because `periapical_lesion` has only `144` training boxes (`9` test boxes), recall at $\tau = 0.70$ is `11.1%` (`1/9`), and `deep_caries` recall is `18.8%` (`6/32`, with `14/32` downgraded to `caries`). Lowering the UI confidence slider increases sensitivity at the cost of additional `caries` false positives.
2. **2D Projection Superposition & Caries False Positives:** Panoramic radiographs project 3D craniofacial anatomy onto a 2D plane, causing interproximal contact overlap, cervical burnout, and restoration margin artifacts (`124` background `caries` false positives at $\tau = 0.70`).
3. **No FDI Tooth Numbering in Current Pipeline:** The detector predicts lesion bounding boxes in original pixel coordinates and sets `"tooth_fdi": null` rather than guessing tooth numbers without a dedicated tooth-enumeration head.
4. **Heuristic Input Gate Default:** When no binary classifier checkpoint is supplied to `DentalInputGate`, validation relies on deterministic radiographic heuristics (aspect ratio, monochrome channel consistency, dynamic range, and continuous-tone mid-tone histogram constraints) rather than a learned OPG classifier.

---

## 10. Roadmap: Two-Week Engineering Expansion

1. **Class-Balanced Loss & Oversampling:** Replace standard cross-entropy in the RoI head with Focal Loss / repeat-factor sampling for `periapical_lesion` and `deep_caries`, and use gradient accumulation (`accum_steps = 2`) at `960×1600` to eliminate batch-size-1 gradient noise.
2. **FDI Tooth Enumeration Integration:** Train a second stage on DENTEX's quadrant-enumeration annotations (`11`–`48`) and spatially join detected lesions to tooth polygons via IoU/centroid containment to populate `tooth_fdi`.
3. **ONNX Runtime & INT8 Quantization:** Export the ResNet-50 FPN V2 checkpoint to ONNX Runtime with dynamic quantization to reduce 2-vCPU latency from `4.6s` toward sub-second interactive response.
4. **3D CBCT Individual Tooth Instance Labelling:** Combine `DentalSegmentator`'s semantic 5-class output with ToothFairy2 instance segmentation to report per-FDI-tooth (`38`/`48`) third-molar impaction angles alongside mandibular canal clearance.

---

## 11. Bonus Track: 3D CBCT Volumetric Segmentation, Nerve Clearance & Interactive 3D Meshing

Following completion and verification of the end-to-end 2D OPG pipeline, the same FastAPI service (`POST /analyze` in [`service.py`](service.py) and [`cbct_pipeline.py`](cbct_pipeline.py)) and Gradio Hugging Face Space ([`hf_space/app.py`](hf_space/app.py)) were extended to accept 3D CBCT volumes (`.nii.gz`, `.nii`, or `.zip` archives containing a 3D DICOM series or NIfTI pair).

### 11.1 Explicit Pretrained Model Disclosure
- **Model Used:** **`DentalSegmentator`** (`Dataset112_DentalSegmentator_v100.zip`, `229,747,861` bytes, MD5 `b71cd5230168d28a4f71b078265b76be`), published by **Dot et al., *Journal of Dentistry* (2024)**, Zenodo DOI: [`10.5281/zenodo.10829675`](https://zenodo.org/records/10829675), License: **CC BY 4.0**.
- **Usage Disclosure:** We explicitly disclose that **`DentalSegmentator` (`nnUNetTrainer__nnUNetPlans__3d_fullres`, `fold_0`) is used pretrained as-is** (trained by Dot et al. on 470 multi-center CT and CBCT scans). All engineering in this bonus track focuses on 3D volumetric ingestion, DICOM/NIfTI de-identification, connected-component post-processing, exact 3D Euclidean nerve-to-root distance computation in physical millimeters, Marching Cubes `.stl`/`.glb` surface generation, and interactive browser integration (`<model-viewer>` / Three.js WebGL). Full execution logs are preserved in [`kaggle_cbct_notebook.ipynb`](kaggle_cbct_notebook.ipynb).

### 11.2 Dual Clinical Benchmark Scans: Pre- & Post-Operative Verification
To ensure rigorous evaluation without artificial data synthesis, the bonus track provides two genuine, distinct 3D CBCT scans sourced from official `3D Slicer DentalSegmentator` testing data ([`PreDentalSurgery`](samples/cbct/PreDentalSurgery_bundle.zip) and [`PostDentalSurgery`](samples/cbct/PostDentalSurgery_bundle.zip)). These represent authentic pre-operative and post-operative scans of the same patient:

| Clinical Scan Metric | `PreDentalSurgery` (Pre-Op Benchmark) | `PostDentalSurgery` (Post-Op Benchmark) | Clinical Comparison & Significance |
|---|:---:|:---:|---|
| **Volume Shape & Voxel Count** | `360 × 360 × 330` (`42,768,000` voxels) | `360 × 360 × 330` (`42,768,000` voxels) | Same patient geometry; isotropic `0.5 × 0.5 × 0.5 mm` spacing |
| **Raw Volume NIfTI Size** | `24,422,912` bytes (`24.4 MB`) | `35,173,024` bytes (`35.2 MB`) | Distinct field coverage, soft-tissue boundary, and HU distributions |
| **Maxilla / Upper Skull Volume** | `909,001` voxels (`113.63 cm³`) | `1,476,019` voxels (`184.50 cm³`) | `PostDentalSurgery` includes superior cranial margins |
| **Mandible Volume** | `266,188` voxels (`33.27 cm³`) | `310,556` voxels (`38.82 cm³`) | Post-op osteotomy healing, surgical hardware, and remodeling |
| **Upper Teeth Volume** | `67,160` voxels (`8.39 cm³`) | `75,590` voxels (`9.45 cm³`) | Fully resolved dental crowns and roots |
| **Lower Teeth Volume** | `55,954` voxels (`6.99 cm³`) | `62,663` voxels (`7.83 cm³`) | Distinguishable root apices adjacent to canal corridor |
| **Mandibular Canal Volume** | `5,853` voxels (`0.73 cm³`) | `6,063` voxels (`0.76 cm³`) | Bilateral inferior alveolar nerve canals tracked end-to-end |
| **3D Scene GLB Mesh Size** | `11,164,980` bytes (`11.2 MB`) | `13,191,364` bytes (`13.2 MB`) | Completely independent vertex and face topographies |
| **5-Class Mean 3D Dice** | `0.9582` vs reference | `0.9639` vs reference | Slicer DentalSegmentator nnU-Net v2 architecture |

### 11.3 Clinically Useful 3D Finding & Preoperative Nerve Clearance
To assist preoperative surgical planning (e.g., mandibular third-molar odontectomy and sagittal split ramus osteotomy), [`cbct_pipeline.py`](cbct_pipeline.py) computes exact 3D Euclidean clearance from the mandibular canal to the nearest lower tooth root per hemimandible via `scipy.spatial.cKDTree`. The two scans show completely distinct, anatomically accurate clearances:

1. **`PreDentalSurgery` Findings (Pre-Operative Baseline):**
   - **Right Hemimandible:** **`min_distance_mm = 2.12 mm`** (`> 2.0 mm` safe clearance zone) between closest lower tooth `[118, 172, 72]` (`[59.0, 86.0, 36.0] mm`) and canal `[115, 172, 69]` (`[57.5, 86.0, 34.5] mm`).
   - **Left Hemimandible:** **`min_distance_mm = 1.50 mm`** (`1.0–2.0 mm` close proximity caution) between closest lower tooth `[234, 148, 59]` (`[117.0, 74.0, 29.5] mm`) and canal `[234, 148, 56]` (`[117.0, 74.0, 28.0] mm`).

2. **`PostDentalSurgery` Findings (Post-Operative Status):**
   - **Right Hemimandible:** **`min_distance_mm = 0.71 mm`** (`< 1.0 mm` high-proximity alert) between closest lower tooth `[119, 153, 90]` (`[59.5, 76.5, 45.0] mm`) and canal `[118, 153, 89]` (`[59.0, 76.5, 44.5] mm`).
   - **Left Hemimandible:** **`min_distance_mm = 1.58 mm`** (`1.0–2.0 mm` close proximity) between closest lower tooth `[235, 132, 75]` (`[117.5, 66.0, 37.5] mm`) and canal `[236, 132, 72]` (`[118.0, 66.0, 36.0] mm`).

![3D CBCT Multi-Planar Reconstruction (Post-Op)](assets/eval/cbct_ortho_preview.png)
![3D CBCT Multi-Planar Reconstruction (Pre-Op)](assets/eval/cbct_ortho_pre_preview.png)

### 11.4 Measured Memory Use, Inference Latency & Production Architecture for Large CBCT Volumes

#### Empirical Benchmark on Kaggle Tesla T4 GPU (`PostDentalSurgery`, `360 × 360 × 330` voxels)
| Stage / Resource Metric | Measured Value | Hardware / Execution Notes |
|---|:---:|---|
| **GPU Hardware Platform** | **NVIDIA Tesla T4 (14.56 GB VRAM)** | 4 vCPUs, 31.35 GB System RAM, PyTorch `2.11.0+cu128` |
| **Model Checkpoint Initialization** | **2,131.9 ms (2.13 s)** | Loading `checkpoint_final.pth` (`fold_0`) onto `cuda:0` |
| **3D Sliding-Window Inference (`252` tiles)** | **131,407.1 ms (131.41 s)** | `3d_fullres` Gaussian-weighted patch aggregation (`step_size = 0.5`) + NIfTI export |
| **3D Post-Processing & `cKDTree` Distance** | **5,816.3 ms (5.82 s)** | $60\text{ mm}^3$ island filtering (`scipy.ndimage.label`) + physical millimeter nerve distance |
| **3D Marching Cubes Meshing (`.stl` + `.glb`)** | **938.6 ms (0.94 s)** | 5 binary `.stl` meshes + combined `12.08 MB` glTF 2.0 `.glb` scene (`step_size = 2`) |
| **Peak GPU VRAM Allocated / Reserved** | **4,936.0 MB / 5,112.0 MB** | `4.94 GB` peak VRAM allocated during 3D patch aggregation |
| **Process CPU RSS Memory** | **2,641.8 MB (+1,270.8 MB delta)** | Host RAM required for full-resolution 3D probabilities and volume buffers |

#### How We Would Run Large 3D CBCT Volumes in Production
Because clinical CBCT scans are `100–500 MB` (`40M–125M` voxels) and require `~5 GB` GPU VRAM and `~20–130 s` of 3D sliding-window compute, running synchronous inference inside a web worker causes HTTP timeouts and OOM crashes. A production deployment separates ingestion, GPU inference, and 3D visualization into four stages:
1. **Direct Presigned Object-Store Ingestion & Edge De-Identification:** The browser uploads `.nii.gz` or zipped DICOM series directly to S3/GCS via multipart presigned URLs, stripping DICOM patient tags (`PatientName`, `PatientID`, `BirthDate`, `InstitutionName`) before storage.
2. **Asynchronous Job Queue (`POST /jobs/cbct` $\rightarrow$ `202 Accepted`):** The FastAPI gateway enqueues a job payload into Redis / Celery (or Cloud Tasks) and streams progress events (`queued` $\rightarrow$ `preprocessing` $\rightarrow$ `gpu_sliding_window` $\rightarrow$ `meshing` $\rightarrow$ `completed`) to the UI over WebSockets/SSE.
3. **Pre-Warmed GPU Worker Pool with TensorRT / FP16 Batching:** Dedicated GPU workers (e.g., NVIDIA L4 24 GB or T4 16 GB on Kubernetes with KEDA queue-depth autoscaling) keep `DentalSegmentator` weights pinned in VRAM (`0 ms` cold start) and compile the 3D U-Net encoder/decoder with TensorRT FP16, batching `4–8` sliding-window 3D tiles per forward pass to reduce 252-tile inference latency from `131 s` on T4 down to `~18–25 s` on L4.
4. **Draco-Compressed `.glb` CDN Delivery:** Marching Cubes meshes are decimated/Draco-compressed (`12 MB` $\rightarrow$ `~1.5 MB`) and served via CDN with immutable content hashes for zero-latency 60 FPS WebGL rendering in `<model-viewer>` / `vtk.js` / `three.js`.
