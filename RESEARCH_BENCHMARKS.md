# External Benchmarks, Gap Analysis, and the Viscan Benchmark Architecture

Verified on 2026-10-25. Every number below is either (a) copied from a cited table, or (b) measured by this
repository and already recorded in [`REPORT.md`](REPORT.md). No value is estimated, extrapolated, or
reconstructed. Where a comparison is not like-for-like, that is stated next to the comparison.

---

## 1. What the official benchmark actually measures

Source: DENTEX challenge report, arXiv:2305.19112 (HTML v2), Tables I–II and the evaluation-metric section.

- The challenge has three tasks: quadrant detection, FDI enumeration, and diagnosis (the four disease classes
  `caries`, `deep_caries`, `periapical_lesion`, `impacted`). Diagnosis is the task our four-class detector
  corresponds to.
- Split: **705 train / 50 validation / 250 challenge test**. The 250 test annotations are not public.
- Metric: COCO-style **AP averaged over 10 IoU thresholds from 0.50 to 0.95**, plus AP50, AP75 and AR.
  AP/AP50/AP75 are reported as percentages, AR as a decimal.
- The report's Table II contains submitted algorithms, reference runs of published detectors (entries named
  after the detector's original author), and one baseline entry marked with `*`.

## 2. Official DENTEX 2023 diagnosis leaderboard (verbatim values)

Table II, Diagnosis columns only. Percentages as published; `AR` decimal as published.

| Entry (as listed) | AR | AP | AP50 | AP75 |
|---|---:|---:|---:|---:|
| Hamamci I. (`HierarchicalDet`, baseline `*`) | 0.691 | **37.6** | **60.2** | 44.0 |
| Mei S. | 0.5972 | 37.44 | 56.72 | **45.49** |
| He L. | 0.5921 | 37.06 | 54.31 | 44.77 |
| Chen S. | 0.644 | 37.0 | 58.1 | 42.6 |
| Choi K. | **0.7585** | 36.48 | 57.30 | 42.84 |
| Carion N. (DETR reference) | 0.514 | 33.4 | 52.8 | 41.7 |
| Ren S. (Faster R-CNN reference) | 0.533 | 33.2 | 54.3 | 38.0 |
| Lin T.Y. (RetinaNet reference) | 0.587 | 32.5 | 54.2 | 35.6 |
| Wu H. | 0.6062 | 30.63 | 48.36 | 36.32 |
| van Nistelrooij N. | 0.6489 | 30.15 | 50.38 | 34.22 |
| Dascalu T. | 0.3033 | 14.03 | 26.02 | 13.54 |

Readings that follow directly from the table:

- The five best diagnosis entries span **36.48 – 37.6 AP**, a 1.12-point band.
- The best entry is **+4.4 AP / +5.9 AP50 / +6.0 AP75** above the Faster R-CNN reference in the same table.
- Faster R-CNN and RetinaNet reference runs are **0.7 AP apart** (33.2 vs 32.5).

## 3. Methodology behind the strongest entries (from the report's method descriptions)

| Entry | Approach | Diagnosis result |
|---|---|---|
| Hamamci I. (baseline) | Diffusion-based hierarchical detector, Swin-Transformer backbone + FPN; decoder refines noisy boxes. | 37.6 AP / 60.2 AP50 |
| Mei S. (YOLOrtho) | Single-stage, modified YOLOv8 with coordinate convolutions, **four independent binary classification heads**, Hungarian linear-sum assignment. | 37.44 AP / 56.72 AP50 / 45.49 AP75 |
| He L. | Three-stage hybrid: DINO (ResNet-50), U-Net / SE U-Net segmentation, DiffusionDet for quadrants, YOLOv8; **weighted box fusion (WBF)** over models. | 37.06 AP / 54.31 AP50 |
| Chen S. (DiffusionDet) | Diffusion-based detector with FPN. | 37.0 AP / 58.1 AP50 |
| Choi K. (DETDet) | Two-stage: Mask R-CNN with Swin-T for enumeration; **ensemble of DiffusionDet and DINO** for diagnosis; closest-box-center matching. | 36.48 AP / 57.30 AP50 / 0.7585 AR |

Common ingredients among the top five: a modern backbone (Swin / DINO), **model fusion** (WBF or ensembles)
in at least two entries, and hierarchy or enumeration context in at least two. The single-stage YOLOv8-based
entry is the cheapest of them to deploy.

## 4. Our measured results (this repository)

Source: `REPORT.md` §5, produced by `evaluate.py` on the **50-image official validation split, used as the
held-out test set** (182 boxes), with the threshold selected on the separate 70-image validation split.

| Metric | Value |
|---|---:|
| mAP@50 | 0.496 |
| mAP@50:95 | 0.329 |
| mAP@75 | 0.383 |
| AR@100 | 0.683 |
| Operating threshold (selected on val) | 0.70 |
| Val macro-F1 at that threshold | 0.556 |
| Test macro-F1 | 0.4016 |
| CPU latency per image | 4,618 ms |

Per-class test AP@50: `impacted_tooth` 0.833, `deep_caries` 0.457, `caries` 0.358,
`periapical_lesion` 0.336.

The 960×1600 variant (same weights protocol, `batch_size=1`) measured 0.493 AP50 / 0.313 AP / 0.410 AP75.

## 5. Our model versus the published leaderboard

**These are not head-to-head numbers.** Our test images are the 50 public validation images; the leaderboard
uses the 250 hidden challenge-test images. Our training split is 635 images (70 held out of the 705 pool);
challenge entries train on the 705-image pool with unknown schedules. The metric definition is the same.

| Comparison | Ours | DENTEX Faster R-CNN reference | Delta |
|---|---:|---:|---:|
| AP (0.50:0.95) | 32.9 | 33.2 | −0.3 |
| AP50 | 49.6 | 54.3 | −4.7 |
| AP75 | 38.3 | 38.0 | +0.3 |
| AR@100 | 68.3 | 53.3 | +15.0 |

What can be said without overreach:

1. Our AP and AP75 are within 0.3 points of the challenge's Faster R-CNN reference run. Our AP50 is
   4.7 points lower while our AR@100 is 15 points higher.
2. That pattern — same strict-IoU quality, more detections retained, worse loose-IoU ranking — points at
   **classification/ranking**, not localization. Our own confusion matrix is consistent with it: 14 of 32
   `deep_caries` boxes are localized at IoU ≥ 0.50 but labeled `caries`, and `caries` carries 124 background
   false positives at τ = 0.70. This is a hypothesis supported by our measured error counts, not a proven cause.
3. The leaderboard's own spread says the architecture ceiling on this benchmark is about **+4 AP over a
   Faster R-CNN reference**. Expecting a single architecture swap to deliver much more than that is not
   supported by the published results.

## 6. Architecture decision for this assignment

### Rejected options and why

| Option | Why not adopted |
|---|---|
| Ultralytics YOLOv8/YOLO11 | Promising accuracy (Mei et al. rank 2 on diagnosis with a modified YOLOv8), but **AGPL-3.0** for code and weights. This repository is BSD-3-Clause; adding it would change the commercial-shipping posture documented in `REPORT.md` §3 and §8. Kept as a paper comparison only. |
| DINO / DiffusionDet / Swin hierarchical | Strongest published results, but no BSD torchvision implementation, materially higher CPU cost than the current 4.6 s budget, and a training pipeline we cannot validate here. |
| Two-stage tooth → finding cascade | The literature's dominant pattern (DETDet, HierarchicalDet, He et al.), but it needs FDI enumeration labels, a new label-conversion path, and a `tooth_fdi` field in the API. Deferred as roadmap, not claimed as done. |

### Adopted benchmark architecture (all BSD-3-Clause)

```
Faster R-CNN R50-FPN-V2 @800×1333 ──┐
                                    ├─ full-image + native-scale tiled views ─ hflip TTA ─ NMS ─ WBF (IoU 0.55)
RetinaNet   R50-FPN-V2 @800×1333 ──┘        │
                                            ├─ ConvNeXt-Tiny crop refiner (rescore / reclass) ─ val-selected τ ─ test
```

- **Members:** the deployed Faster R-CNN checkpoint and a RetinaNet trained on the identical split, seed,
  schedule and input scale (`configs/retinanet.yaml`). Both torchvision, BSD-3-Clause.
- **Views:** each detector runs on the full image and on overlapping 800×800 native-scale crops
  (`slice_inference.py`). Full-image inference downscales DENTEX images by 0.425–0.675× (mean 0.474×);
  800×800 crops are processed at 1:1 inside `min_size=800, max_size=1333`, so lesion pixels are not thrown away.
- **TTA:** original + horizontally flipped inference, boxes mirrored back, merged with class-wise NMS.
- **Fusion:** weighted box fusion, the rule used by He et al. and DETDet.
- **Refiner:** a ConvNeXt-Tiny classifier trained on train-split GT crops (positives) and IoU<0.05 random
  crops (background) re-scores boxes as `sqrt(det_score × p_class)`; a second variant also re-labels the box
  and drops boxes whose argmax is background. It targets the two measured error modes: `deep_caries`→`caries`
  confusion and `caries` background false positives.
- **Threshold:** still selected only on the 70-image validation split, then applied once to the test split.
- **Deployment posture:** the app keeps the single Faster R-CNN model; every added stage is a benchmark artifact
  unless its measured test gain justifies the extra CPU cost. Sliced inference multiplies forward passes by the
  tile count (~10 per image at 2830×1367), which is recorded per run in `slice_meta_<split>.json`.

Evidence for each component, and what is *not* evidence:

- WBF and ensembles appear in two top-five diagnosis entries → grounds the fusion choice.
- TTA has **no DENTEX evidence** in the sources reviewed; it is included as a measured ablation, not as an
  expected win.
- Slicing has external evidence on other domains (SAHI, arXiv 2202.06934: +6.8 AP FCOS, +5.1 VFNet,
  +5.3 TOOD on VisDrone/xView). It has **no DENTEX evidence** here; it is a measured ablation.
- The crop refiner is a standard two-stage pattern, but no published DENTEX entry reports this exact rescorer.
- The claim that any of this beats our 0.496 AP50 is **unverified** until the notebook is run. Nothing in
  this document asserts it.

## 7. Experiment ladder and decision rule

| # | Experiment | Command | Decision rule |
|---|---|---|---|
| E1 | Architecture: RetinaNet vs Faster R-CNN | `train.py --config configs/retinanet.yaml`, then `evaluate.py --select_threshold val --skip_variant` | Adopt RetinaNet only if test AP50 or AP improves; report the loss otherwise. |
| E2 | hflip TTA ablation | `evaluate.py ... --tta hflip --dump_predictions preds_a` | Keep only if test AP50/AP improves; latency doubles by construction. |
| E3 | WBF ensemble | `ensemble_wbf.py` over the two members' dumps | Adopt for benchmark if AP50/AP improves; adopt for deployment only if latency is acceptable. |
| E4 | Native-scale sliced inference | `slice_inference.py --views both --tile 800 --overlap 0.25`, fuse with `ensemble_wbf.py` | Keep only if fused AP50/AP improves over E2/E3 on test. |
| E5 | ConvNeXt-Tiny crop refiner | `cascade_refine.py build/train/apply`, fuse the rescored dump | Keep only if fused AP50/AP improves; `reclass` must also not lower per-class AP. |
| E6 | Tooth → finding cascade (deferred) | requires enumeration labels + API `tooth_fdi` | Revisit after E1–E5. |

Pipeline selection uses **validation mAP only**; the test split is read once per pipeline at the end. The notebook
reports every measured row, including the ones that lost.

The Kaggle notebook `kaggle_beat_benchmark_notebook.ipynb` runs E1–E5 in one session and prints the comparison
table from the JSON files each run writes.

## 8. Licences

- torchvision code and both detector architectures: BSD-3-Clause.
- DENTEX annotations/images: CC BY-NC-SA 4.0; fine-tuned weights inherit the non-commercial restriction.
- No AGPL component is added by the TTA or WBF code in this repository.

## 9. Explicit non-claims

- No state-of-the-art claim on DENTEX or on our split.
- No claim that our 50-image test results are comparable to the challenge's 250-image test results.
- No claim that TTA, WBF, slicing or the crop refiner improves our metrics until the measured JSON says so.
- No claim that sliced inference reproduces SAHI's reported gains; that evidence is from VisDrone/xView.
- No claim about the leaders' exact training recipes beyond what the challenge report states.

## Sources

- DENTEX challenge report (tasks, splits, metric definition, Table I and Table II, method descriptions):
  https://arxiv.org/html/2305.19112v2
- He et al., integrated segmentation and detection (DINO, U-Net, WBF): https://arxiv.org/abs/2308.14161
- Mei et al., YOLOrtho (modified YOLOv8, four binary heads, Hungarian assignment): https://arxiv.org/abs/2308.05967
- Choi et al., DETDet (Mask R-CNN + Swin-T enumeration, DiffusionDet/DINO diagnosis ensemble):
  https://arxiv.org/abs/2308.14070
- Challenge site and data description: https://dentex.grand-challenge.org/ and https://dentex.grand-challenge.org/data/
- Internal measured results: [`REPORT.md`](REPORT.md) §5–§6
