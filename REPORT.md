# Viscan Technical Report: Automated Dental Finding Detection in Panoramic Radiographs (OPG)

**Candidate:** Surendra Goud | AI Engineer (Round 2 Take-Home Assessment)  
**Target Modality:** 2D Panoramic Radiograph (OPG)  
**Deliverable Type:** Technical Report & Model Evaluation  

> **Clinical Disclaimer:** AI-assisted findings for dentist review. Not a diagnosis.

---

## 1. Executive Summary

This project delivers an end-to-end clinical decision-support prototype for 2D dental panoramic radiographs (OPGs). The system ingests OPGs in PNG, JPEG, and single-frame 2D DICOM (`.dcm`) formats, detects four dental finding classes (`impacted_tooth`, `caries`, `periapical_lesion`, `deep_caries`), and produces class-colored visual annotations and deterministic structured JSON reports for dentist review.

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
4. **3D CBCT Volumetric Extension:** Add a separate async worker pipeline using 3D patch-based nnU-Net (`DentalSegmentator`) for NIfTI / multiframe 3D CBCT volumes.
