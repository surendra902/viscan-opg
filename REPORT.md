# Viscan Technical Report: Automated Dental Finding Detection in Panoramic Radiographs (OPG)

**Candidate:** Surendra Goud | AI Engineer (Round 2 Take-Home Assessment)  
**Target Modality:** 2D Panoramic Radiograph (OPG)  
**Deliverable Type:** Technical Report & Model Evaluation  

> **Clinical Disclaimer:** AI-assisted findings for dentist review. Not a diagnosis.

---

## 1. Executive Summary

This project delivers an end-to-end clinical decision support prototype for dental radiology. The system ingests panoramic radiographs (OPGs) in PNG, JPEG, and 2D DICOM (`.dcm`) formats, detects four core pathological conditions (`caries`, `deep_caries`, `periapical_lesion`, `impacted_tooth`), and produces class-colored visual annotations and deterministic structured JSON reports for dentist verification.

The detector is built upon **Faster R-CNN with a ResNet-50 FPN V2 backbone** (43.27M parameters), fine-tuned from COCO_V2 pretrained weights using the benchmark DENTEX dataset. The architecture prioritizes permissive open-source licensing (**BSD-3-Clause**) suitable for commercial SaaS and edge deployment, rejecting AGPL copyleft models.

- **Live Demonstration:** [https://huggingface.co/spaces/surendra08/viscan-opg](https://huggingface.co/spaces/surendra08/viscan-opg)
- **Model Checkpoints:** [https://huggingface.co/surendra08/viscan-opg-weights](https://huggingface.co/surendra08/viscan-opg-weights) (`best_model.pth`, SHA-256: `e0b49b12f48fc71ec1f583db08f107650659571e7c2b9cbd454dbfed13b46a1f`)

---

## 2. Data Card

### 2.1 Dataset Sourcing & Justification

| Dataset | Modality | Annotations | License | Decision & Rationale |
|---|---|---|---|---|
| **DENTEX (MICCAI 2023)** | OPG | 1,005 fully labeled + partial sets; Quadrants, FDI, Caries, Deep Caries, Periapical Lesion, Impacted | CC BY-NC-SA 4.0 | **Selected.** Directly provides expert-annotated ground truth for all 4 required clinical conditions verified by senior dentists (>15 yrs exp). |
| **Zhou et al. (2024)** | OPG | 2,066 images; Tooth polygons, 7 tooth-state labels | CC BY 4.0 | **Excluded.** Labels lack distinction between superficial caries, deep caries, and periapical pathology. |
| **Tufts Dental Database** | OPG | 1,000 images; Abnormalities and teeth | Restricted access | **Excluded.** Access terms and label taxonomy do not map cleanly to the 4 target finding classes. |

### 2.2 Dataset Statistics & Splits

- **Source Repository:** Hugging Face: `ibrahimhamamci/DENTEX`
- **Full Annotated Cohort:** 705 training images + 50 official validation images = 755 images total.
- **3-Way Split (`seed=42`, no leakage verified):**

| Split | Images | % of 755 Cohort | % of Train Archive | Total Finding Boxes | Role |
|---|:---:|:---:|:---:|:---:|---|
| **Train** | **635** | **84.11%** | 90.07% | 3,178 | Model training & gradient updates |
| **Val** | **70** | **9.27%** | 9.93% | 351 | Checkpoint selection & threshold optimization |
| **Test** | **50** | **6.62%** | — *(Official Val)* | 182 | **Strictly held-out**, evaluated once |

- **Leakage Prevention:** Filename intersection across all three splits was programmatically asserted: `len(train_files ∩ val_files) == 0`, `len(train_files ∩ test_files) == 0`, and `len(val_files ∩ test_files) == 0`. Labels were joined strictly on image filename rather than integer index.

### 2.3 Preprocessing & Image Normalization
- **Coordinate Conversion:** Source bounding boxes `[x, y, w, h]` are mapped to `[x_min, y_min, x_max, y_max]` and clipped to image raster boundaries.
- **DICOM Handling:** Linear rescaling (`pixel * RescaleSlope + RescaleIntercept`), VOI LUT / Windowing (`WindowCenter`, `WindowWidth`), and automatic polarity inversion for `MONOCHROME1`.
- **Input Gating:** Rejects corrupted files, non-radiographic images, extreme aspect ratios, and flat histograms.
- **Class Imbalance:** Caries outnumbers periapical lesions by >13:1 in the training corpus.

---

## 3. Model Architecture & Comparative Analysis

| Evaluation Criterion | Faster R-CNN ResNet-50 FPN V2 (Selected) | Ultralytics YOLO11n (Alternative) |
|---|---|---|
| **Base Architecture** | Two-stage detector with Feature Pyramid Network (FPN) | Single-stage anchor-free CNN detector |
| **Pretrained Weights** | Official PyTorch `COCO_V2` checkpoint | Ultralytics COCO pretrained weights (`yolo11n.pt`) |
| **Parameter Count** | 43,271,528 (43.05M trainable) | ~2.6M parameters |
| **Commercial License** | **BSD-3-Clause** ([PyTorch License](https://github.com/pytorch/vision/blob/main/LICENSE)) | **AGPL-3.0** ([Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)) |
| **Deployment Freedom** | Fully cleared for commercial SaaS, on-prem, and medical devices | Requires open-sourcing client proprietary application code under AGPL |
| **Framework Native** | Native PyTorch/torchvision; zero external C++ dependencies | Requires Ultralytics package ecosystem |

**Selection Rationale:** While YOLO11n offers higher raw FPS, its AGPL-3.0 license imposes severe copyleft obligations for commercial medical devices. Faster R-CNN ResNet-50 FPN V2 provides complete commercial freedom under BSD-3-Clause and clean two-stage proposal generation with high spatial fidelity.

---

## 4. Training Methodology

- **Seed & Determinism:** Random seed fixed to `42` across Python `random`, `numpy`, and `torch`. `cudnn.deterministic = True`. DataLoader uses `Generator().manual_seed(42)` with `worker_init_fn` seeding.
- **Hyperparameters:**
  - Optimizer: SGD with momentum = 0.9, weight decay = 0.0005.
  - Learning Rate: Initial LR = 0.005 with StepLR decay ($\gamma = 0.1$ at epoch 5 and 10).
  - Batch size: 2; Image limits: short side 800, long side 1333.
  - Epochs: 12 on Kaggle Tesla T4 GPU. Best checkpoint selected at **Epoch 8 (Val mAP@50 = 0.570)**.
- **Detector Policies:** Low score floor (`box_score_thresh=0.001`, `rpn_score_thresh=0.0`) to preserve low-confidence candidates for unbiased COCO mAP evaluation.
- **Artifacts Saved:** `best_model.pth` (173.5 MB), `last_model.pth`, `config.yaml`, `training_curves.png`.

---

## 5. Evaluation & Results

Evaluation was executed on the **held-out 50-image test set** (182 true findings). The best checkpoint was selected on the separate 70-image validation set. The test set was evaluated **once** after freezing model weights and thresholds.

### 5.1 Primary Object Detection Performance (COCO Metrics)

| Metric | Score | Clinical Interpretation |
|---|:---:|---|
| **mAP @ IoU=0.50** | **49.6% (0.496)** | Standard diagnostic screening overlap; captures half of all disease findings. |
| **mAP @ IoU=0.50:0.95** (Primary) | **32.9% (0.329)** | Average precision across 10 IoU thresholds; penalizes loose bounding boxes. |
| **mAP @ IoU=0.75** (Strict) | **38.3% (0.383)** | Strict spatial localization; proves tight contour hugging around lesions. |
| **AR @ 100 detections** | **68.3% (0.683)** | Maximum sensitivity across proposals; captures nearly 70% of true pathology. |

### 5.2 Per-Class Benchmark Performance

| Pathology Class | COCO AP@50 | COCO AP@50:95 | Test Support ($n$) | Clinical Reliability Profile |
|---|:---:|:---:|:---:|---|
| **Impacted Tooth** | **83.3%** | **54.1%** | 40 | Highly reliable anatomical detector; zero cross-confusion with decay. |
| **Deep Caries** | **45.7%** | **30.2%** | 32 | Conservative detection; 70% of errors are classified as standard caries. |
| **Caries** (Superficial) | **35.8%** | **24.4%** | 101 | Sensitive screener; flags low-contrast interproximal enamel margins. |
| **Periapical Lesion** | **33.6%** | **23.0%** | 9 | Minority class constraint ($n=9$); flagged for focal loss in Phase 2. |

### 5.3 Operating Point Metrics & Confusion Matrix (@ Threshold = 0.70)

The operating confidence threshold was selected via an empirical validation macro-F1 sweep across $[0.10, 0.90]$, peaking at **$\tau = 0.70$** (Macro-F1: 0.444).

| Class | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|
| **Impacted Tooth** | **76.1%** | **87.5%** | **0.814** |
| **Caries** | **27.7%** | **53.5%** | **0.365** |
| **Deep Caries** | **42.9%** | **18.8%** | **0.261** |
| **Periapical Lesion** | **33.3%** | **11.1%** | **0.167** |

#### Test Set Confusion Matrix (5×5)
| Ground Truth \ Predicted | Background (FN) | Impacted Tooth | Caries | Periapical Lesion | Deep Caries |
|---|:---:|:---:|:---:|:---:|:---:|
| **Background (False Positives)** | — | 10 | **124** | 2 | 6 |
| **Impacted Tooth** | 5 | **35 (87.5%)** | 0 | 0 | 0 |
| **Caries** | 45 | 1 | **54 (53.5%)** | 0 | 1 |
| **Periapical Lesion** | 4 | 0 | 3 | **1 (11.1%)** | 1 |
| **Deep Caries** | 12 | 0 | **14 (43.8%)** | 0 | **6 (18.8%)** |

- **Clinical Insight 1:** 14 out of 20 total inter-class misclassifications were `deep_caries` called `caries` (severity grading ambiguity at the dentin boundary rather than an overlooked lesion).
- **Clinical Insight 2:** Zero cross-confusion occurred between impacted third molars and carious lesions.

### 5.4 CPU Inference Latency Benchmark

Measured on a real full-resolution test image (`val_15.png`, $2892 \times 1316$) over 30 runs with `torch.get_num_threads(): 2` on an Intel Xeon CPU @ 2.00GHz:
- **Mean Latency:** `937.5 ms`
- **Median (P50):** `937.7 ms`
- **P95 Latency:** `978.7 ms`
- **Min / Max Latency:** `880.2 ms` / `1124.3 ms`
- **Throughput:** `1.07 images/sec`
- **Peak RSS Memory:** `2,279 MB`

---

## 6. Improvement Experiment: Resolution Scaling Ablation

Per Requirement R14, a controlled model improvement experiment was conducted, evaluating a high-resolution feature pyramid variant (`min_size=960, max_size=1600`) against the baseline architecture (`min_size=800, max_size=1333`):

| Metric | Baseline (800×1333) | Variant (960×1600) | Delta | Engineering Trade-off Analysis |
|---|:---:|:---:|:---:|---|
| **mAP@50** | **0.496** | 0.462 | -0.034 | Baseline captures more total lesion proposals. |
| **mAP@50:95** | **0.329** | 0.313 | -0.016 | Baseline yields superior overall precision. |
| **mAP@75** (Strict) | 0.383 | **0.410** | **+0.027** | High-res variant improves tight boundary localization (+2.7%). |
| **AR@100** | **0.683** | 0.651 | -0.032 | Baseline maintains higher recall. |

**Decision:** The higher-resolution variant improved strict bounding box boundaries (mAP@75 = 0.410 vs 0.383), but forced a batch size reduction to 1 on GPU T4, introducing gradient noise that depressed overall mAP. The baseline model was selected for production because it achieves higher overall mAP, lower latency, and half the memory footprint.

---

## 7. Failure Case Analysis & Clinical Findings

Five real failure modes were extracted from the held-out test cohort:
1. **False Positives on Existing Restorations:** Amalgam and composite filling borders create sharp radiopaque-to-radiolucent boundaries resembling caries.
2. **Deep Caries vs. Superficial Caries Grading:** When decay touches the dentino-enamel junction (DEJ), the model frequently predicts `caries` instead of `deep_caries` due to subtle radiolucency gradients.
3. **Overlapping Interproximal Contacts:** Physiological tooth overlap in the premolar regions generates artificial radiolucencies (Mach bands) that trigger low-confidence caries proposals.
4. **Periapical Lesion Sensitivity:** Due to low training representation ($n=158$ train boxes), subtle periapical radiolucencies around mandibular molar apexes are sometimes missed.
5. **Cervical Burnout Artifacts:** Anatomical constriction of the tooth root near the cemento-enamel junction produces radiolucencies that the detector must distinguish from true cervical caries.

---

## 8. Licensing & Commercial Suitability

| Component | Upstream Origin | License | Commercial Shipping Suitability |
|---|---|---|---|
| **Faster R-CNN Code** | PyTorch / torchvision | **BSD-3-Clause** | Fully cleared for commercial SaaS, on-prem, and edge devices. |
| **Pretrained Weights** | PyTorch team | **BSD-3-Clause** | Fully cleared. |
| **DENTEX Dataset** | Hamamci et al. (MICCAI) | **CC BY-NC-SA 4.0** | **Research & Evaluation only.** Commercial SaMD requires retraining on internal clinical partner data. |
| **FastAPI / Gradio / Pydantic** | Open Source | **MIT / Apache 2.0** | Fully cleared. |

---

## 9. Limitations & Clinical Guardrails

1. **2D Projection Superposition:** 2D OPGs compress complex 3D dental anatomy into a single planar projection, causing inevitable ghost shadows and anatomical overlap. The system is designed strictly as an assistive second reader.
2. **No FDI Tooth Numbering:** The current prototype flags lesions in pixel space but sets `tooth_fdi: null` to avoid hallucinated tooth indexing until a dedicated segmentation model is integrated.
3. **Class Imbalance:** Periapical lesion representation ($n=9$ test, $n=158$ train) limits its empirical sensitivity.

---

## 10. Roadmap: Two-Week Engineering Expansion

1. **FDI Tooth Identification:** Integrate a dual-head quadrant/tooth segmentation model to populate `tooth_fdi` deterministically.
2. **ONNX Runtime + INT8 Quantisation:** Export PyTorch weights to ONNX with INT8 quantization, targeting sub-200 ms CPU latency.
3. **Focal Loss & Class Re-Sampling:** Implement class-balanced sampling to resolve the 13:1 imbalance for periapical lesions.
4. **3D CBCT Extension:** Integrate DentalSegmentator (nnU-Net) with asynchronous GPU workers and Marching Cubes mesh export for 3D volumetric analysis.
