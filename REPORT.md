# Viscan: Automated Dental Finding Detection in Panoramic X-rays (OPG)

Technical report for the AI Engineer (Round 2) take-home assessment. Main 2D track, plus a bonus 3D CBCT track.

**Candidate:** Surendra

| Field | Value |
|---|---|
| Report version | 1.0 |
| Date | 27 October 2026 |
| Code commit | `31c424cea648ffbeb459fc275eaeca23ea8c6277` (documentation was edited after this commit) |
| Weights | `best_model.pth`, 173,466,789 bytes, SHA-256 `e0b49b12f48fc71ec1f583db08f107650659571e7c2b9cbd454dbfed13b46a1f` |
| Live demo | [huggingface.co/spaces/surendra08/viscan-opg](https://huggingface.co/spaces/surendra08/viscan-opg) |
| Weights download | [huggingface.co/surendra08/viscan-opg-weights](https://huggingface.co/surendra08/viscan-opg-weights) |
| Demo video (5–8 min) | Pending — to be recorded against this build |

> **Clinical disclaimer:** AI-assisted findings for dentist review. Not a diagnosis. This build is not clinically ready and has not been validated on any clinical data.

---

## 1. Summary

I built a service that takes a panoramic dental X-ray, draws boxes around four findings (impacted tooth, caries, periapical lesion, deep caries), and returns an annotated image plus a JSON report. The same endpoint also accepts 3D CBCT scans (§8).

The detector is Faster R-CNN (ResNet-50 backbone, 43.3 M parameters), fine-tuned for 12 epochs on DENTEX 2023. On 50 held-out X-rays it scores **mAP@50 0.496** and **mAP@50:95 0.329**. At the confidence setting I chose (0.70), average F1 across the four classes is 0.402. On a CPU with 2 threads it takes **4.6 seconds** per X-ray.

The best score I measured is the same model run twice, once on the image and once on its mirror: mAP@50 rises from 0.496 to 0.528. I still ship the single pass, because the mirror run doubles the compute and its F1 at the chosen setting is slightly lower (0.399 against 0.402). §5.5 lists every configuration I ran — the baseline plus nine alternatives, one of which improved detection.

> **In plain words**
> - The model draws boxes around four kinds of dental finding on a panoramic X-ray.
> - On 50 X-rays it had never seen, it scores about 50 out of 100 on the standard test (mAP@50), where 100 would mean every marked finding found with no false alarms.
> - It is good at impacted teeth, weak at deep decay, and weaker still at infections at the root tip.
> - It takes about 4.6 seconds per X-ray on a 2-core CPU, so the demo handles one image at a time.
> - This is research code. It is not a diagnosis, and a dentist must check every box.

**Terms used in this report**

| Term | Meaning |
|---|---|
| Precision | Of the boxes the model drew, the share that were correct |
| Recall | Of the findings a dentist marked, the share the model found |
| F1 | One score combining precision and recall; 1.0 is perfect |
| IoU | How much a predicted box overlaps the true box; 0.50 means about half |
| mAP | Average precision across classes. `@50` needs 50% overlap; `@50:95` averages stricter overlaps from 50% to 95% and is my main score |
| TTA | Test-time augmentation: run the model on the image and its mirror, then merge |
| NMS / WBF | Two ways to merge overlapping boxes: keep the best and drop the rest (NMS), or average them (weighted box fusion, WBF) |

---

## 2. Intended use and safety

This is a research prototype for a dentist-facing review workflow. It highlights candidate regions and returns boxes with class labels and confidence scores. It does not diagnose. The confidence score ranks findings; it is not the probability of disease. `tooth_fdi` (tooth number) is always `null` rather than guessed, and patient tags are stripped from DICOM files. The 3D track reports a straight-line gap in millimetres between two segmented surfaces, not a traced nerve or a risk score. If the 3D model files are missing, the service refuses raw 3D uploads with HTTP 503 and never substitutes demo data.

**Licensing restriction:** my fine-tuned weights inherit the non-commercial licence of DENTEX and cannot be used commercially as they are (§7).

---

## 3. Data card

**Dataset choice.** DENTEX (MICCAI 2023, CC BY-NC-SA 4.0) is the only public dataset I found with expert boxes for all four required findings, so it is the only one I used. I excluded two alternatives: Zhou et al. (2024) merges all caries into one label and has no impaction label, and the Tufts database is available only under restricted institutional terms. Source: Hugging Face `ibrahimhamamci/DENTEX`, pinned to revision `7b27ccc8e342dcb774f69adc6ca5e6c09fefce93`.

**Size.** DENTEX has 705 labelled training images, 50 official validation images and 250 challenge-test images with no public labels. I did not use the 250. My working set is 755 labelled images with 3,711 boxes, split with seed 42.

| Split | Images | Boxes | Role |
|---|---:|---:|---|
| Train | 635 | 3,190 | Teaching the model |
| Validation | 70 | 339 | Choosing the best epoch and the confidence setting |
| Test | 50 | 182 | Final scores only; untouched until model and threshold were frozen |

| Class | Train | Val | Test |
|---|---:|---:|---:|
| `impacted_tooth` | 545 | 59 | 40 |
| `caries` | 2,004 | 185 | 101 |
| `periapical_lesion` | 144 | 14 | 9 |
| `deep_caries` | 497 | 81 | 32 |

Caries outnumbers periapical lesions about 14 to 1 in training, and the rarest class has only 9 test boxes. Every detector and setting in this report was scored on exactly this split; §5.5 is the full comparison.

**Avoiding leakage.** The 70 validation images were drawn from the 705-image archive with a seeded shuffle. The test set is the official DENTEX validation set. I checked in code that no filename appears in two splits. DENTEX has no patient IDs, so I cannot rule out one patient appearing in two splits.

**Preparation.** Boxes `[x, y, width, height]` are clipped to the image and converted to `[x_min, y_min, x_max, y_max]`. No classes were merged. Images are shrunk to fit 800 × 1333, about 0.47× on average, which matters for the tiling experiment in §5.5. DICOM files go through the standard contrast settings (modality LUT, then VOI LUT), MONOCHROME1 images are flipped, and an input check rejects blank, corrupt, multi-frame, colour, portrait or screenshot-like files.

**Class imbalance.** I applied no class weights or oversampling in the main run. That is one of the main limitations (§6.2).

The raw data is not in the repository; `prepare_data.py` downloads and prepares it.

---

## 4. Model choice and training

I compared two options and trained Faster R-CNN. I also trained RetinaNet for comparison (§5.5).

| | **Faster R-CNN R50-FPN V2** (used) | Ultralytics YOLO11n (not trained) |
|---|---|---|
| Type | Two-stage detector | Single-stage detector |
| Size | 43.3 M parameters | about 2.6 M |
| Code and starting weights | BSD-3-Clause | AGPL-3.0 |
| Fine-tuned model | CC BY-NC-SA 4.0 (from the data) | AGPL-3.0 plus CC BY-NC-SA 4.0 |

I chose Faster R-CNN for two reasons. The DENTEX reference run is a Faster R-CNN, so I have a like-for-like check, and its licence is permissive, whereas AGPL would force a YOLO-based service to be open-sourced. The cost is a larger, slower model.

**Training setup** (`configs/baseline.yaml`): fine-tuned from COCO-pretrained weights, seed 42 everywhere with deterministic mode, SGD (learning rate 0.005, momentum 0.9, weight decay 0.0005), cosine schedule over 12 epochs, batch size 2, mixed precision on a Tesla T4. The best checkpoint is **epoch 8** (validation mAP@50 0.570, mAP@50:95 0.360). The model keeps low-scoring boxes internally so the precision-recall curves are complete, and the confidence setting is applied afterwards.

![Figure 1: training and validation curves](assets/eval/training_curves.png)

*Figure 1: validation mAP@50 peaks at epoch 8 while training loss keeps falling, so I kept epoch 8 and not the last epoch.*

**Reproduce:**

```bash
python prepare_data.py data/dentex
python train.py --config configs/baseline.yaml
python evaluate.py --weights checkpoints/best_model.pth --data_dir data/dentex
```

---

## 5. Results (50 held-out images, 182 boxes)

### 5.1 Overall scores

| Metric | Score |
|---|---:|
| mAP@50 | 0.496 |
| **mAP@50:95 (main)** | **0.329** |
| mAP@75 | 0.383 |
| AR@100 (recall) | 0.683 |

The drop from 0.496 at 50% overlap to 0.383 at 75% overlap suggests boxes are usually in the right place but not drawn tightly.

For context, the DENTEX 2023 report lists 36.5–37.6 AP for its top five entries and 33.2 AP for its Faster R-CNN reference, on a hidden 250-image set. My 0.329 is on a different 50-image public set, so the two are not comparable.

### 5.2 Per class, and the confidence setting

I swept confidence from 0.10 to 0.90 on the 70 validation images and picked **0.70**, where the average F1 across classes peaked (0.556). I applied it once to the test set. Test average F1 is 0.402, noticeably lower than validation, partly because 50 images is small and partly because one global setting suits the rare classes badly.

| Class | Test boxes | AP@50 | AP@50:95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `impacted_tooth` | 40 | 0.833 | 0.541 | 76.1% | 87.5% | 0.81 |
| `caries` | 101 | 0.358 | 0.244 | 27.7% | 53.5% | 0.37 |
| `deep_caries` | 32 | 0.457 | 0.302 | 42.9% | 18.8% | 0.26 |
| `periapical_lesion` | 9 | 0.336 | 0.230 | 33.3% | 11.1% | 0.17 |

Impacted teeth are usable. The other three are weak, and that gap (0.81 down to 0.17) is the clearest result in this report. Average F1 is the plain mean of the four values: (0.814 + 0.365 + 0.261 + 0.167) / 4 = 0.402.

Those four columns use the class-aware matching rule in [`evaluate.py`](evaluate.py), where a box matched at IoU ≥ 0.50 but given the wrong class counts against both classes. Concretely: caries precision of 27.7% is 54 correct boxes out of 195 drawn, so 141 false positives — 124 on background, 14 deep-caries boxes and 3 periapical boxes. Caries recall of 53.5% is 54 of 101, so 47 misses — 45 never found, 1 labelled deep caries and 1 labelled impacted tooth. Every class decomposes the same way.

### 5.3 Confusion matrix

Rows are the truth, columns are the prediction.

| Truth \ Predicted | Nothing | Impacted | Caries | Periapical | Deep caries |
|---|---:|---:|---:|---:|---:|
| Nothing there (false alarm) | — | 10 | 124 | 2 | 6 |
| `impacted_tooth` | 5 | **35** | 0 | 0 | 0 |
| `caries` | 45 | 1 | **54** | 0 | 1 |
| `periapical_lesion` | 4 | 0 | 3 | **1** | 1 |
| `deep_caries` | 12 | 0 | 14 | 0 | **6** |

The totals reconcile: 182 true boxes = 96 found + 66 missed entirely + 20 swapped between classes, and 142 false alarms land where nothing was marked.

Two patterns explain most of the errors. First, 14 of the 32 deep-caries boxes are found in the right place but called ordinary caries. The line between the two is subjective and depends on how close the decay runs to the nerve, which is hard to see on a flat X-ray. Second, about nine in ten false alarms (124 of 142) are caries, likely on low-contrast gaps between teeth and around fillings, though this run does not check that anatomically.

![Figure 2: test-set confusion matrix](assets/eval/confusion_matrix.png)

*Figure 2: the same matrix as a heatmap; the bright off-diagonal cell is the deep-to-shallow caries confusion.*

### 5.4 CPU speed

End to end (file read, decoding, model, box merging) on a full-size scan (2892 × 1316), with 5 warm-up and 30 timed runs, 2 threads on an Intel Xeon at 2.0 GHz: **mean 4.6 s** (median 4.7 s, 95th percentile 4.9 s, range 4.2–5.0 s), peak memory 2.9 GB. That is fine for a single-user demo and too slow for a clinic queue. GPU speed for this model was not measured.

### 5.5 Comparison of all models and settings

Everything below uses the same 635 / 70 / 50 split, seed and evaluator, from one Kaggle T4 session of [`kaggle_beat_benchmark_notebook.ipynb`](kaggle_beat_benchmark_notebook.ipynb). The run first reproduced the shipped baseline (AP 0.3293, AP50 0.4961, AP75 0.3830, AR@100 0.6834, F1 0.4016), so the comparison starts from a number that matches. Each setting's confidence threshold and each pipeline were chosen on validation, and the test column was read once.

| Setting | What changed | Val AP | Test AP | Test AP50 | Test AP75 | Test AR@100 | Δ AP50 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Baseline (shipped)** | 800 × 1333, one pass | 0.360 | 0.329 | 0.496 | 0.383 | 0.683 | — |
| Higher resolution | 960 × 1600, batch size 1 | 0.354¹ | 0.313 | 0.493 | 0.410 | 0.681 | −0.003 |
| Mirror averaging (TTA) | Same weights, image plus mirror | 0.369 | 0.345 | 0.528 | 0.398 | 0.685 | +0.032 |
| Native-scale tiles | 800 × 800 crops, merged with full view | 0.196 | 0.219 | 0.356 | 0.252 | 0.664 | −0.140 |
| RetinaNet | One-stage model, same recipe | 0.078² | 0.053 | 0.095 | 0.056 | 0.514 | −0.401 |
| Tiles + RetinaNet | Weighted box fusion | 0.145 | 0.179 | 0.302 | 0.197 | 0.646 | −0.194 |
| Tiles + crop checker (re-score) | ConvNeXt-Tiny re-scores each box | 0.224 | 0.221 | 0.359 | 0.247 | 0.656 | −0.137 |
| Tiles + RetinaNet + checker (re-score) | Both | 0.186 | 0.194 | 0.317 | 0.208 | 0.661 | −0.179 |
| Tiles + crop checker (re-label) | Checker re-labels each box | 0.138 | 0.170 | 0.272 | 0.190 | 0.557 | −0.224 |
| Tiles + RetinaNet + checker (re-label) | Both | 0.080 | 0.144 | 0.236 | 0.166 | 0.608 | −0.260 |

¹ Validation AP for that run at its selected epoch (5 of 12). ² This cell is the training loop's validation mAP@50 (0.0783 at epoch 12), not mAP@50:95, so it is not on the same scale as the others; it is here only to show how far that member had converged.

**What this shows.** One of nine alternatives improved detection (mirror averaging, +0.032 AP50), none improved the setting I would actually run, and the rest lost between 0.003 and 0.401 AP50.

- **Why I still ship the baseline.** Mirror averaging doubles the forward passes, and at its own best threshold (0.80) its F1 is 0.399 against 0.402. The higher-resolution run drew tighter boxes (AP75 0.383 → 0.410) and found impacted teeth better (AP50 0.833 → 0.915), but lost on overall AP and on deep caries (0.457 → 0.365). Its batch size also changed, so resolution is not isolated.
- **Why tiling failed (my best explanation, not proven).** An 800 px tile is shown to the model at about twice the object size it learned (0.474× in training), so every tile is an unusual view, and the tile pass produced about 373 candidates per image, more than the 300-box cap. AP50 fell to 0.356.
- **RetinaNet was undertrained, not shown to be worse.** Its validation score was still rising at epoch 12, so the architecture is untested rather than rejected. Fusing it with the tiles cost 0.054 AP50 against the tiles alone, which says nothing about fusion as an idea.
- **The crop checker** helped slightly when re-scoring (0.219 → 0.221) and hurt when re-labelling: it changed 1,850 of 3,303 test boxes, dropped 277 as background, and AP50 fell to 0.272. Its reported 92.1% accuracy is on its own training crops ([`cascade_refine.py:276`](cascade_refine.py#L276)), not held-out data.
- **Stacking compounds the error.** The two three-stage fusions landed at 0.194 and 0.144 AP50, below the tiled detector (0.356) and below the checker alone (0.359 and 0.272).

At each pipeline's own validation-selected threshold, the setting a user would actually run:

| Pipeline | Threshold from validation | Test F1 | Test AP50 |
|---|---:|---:|---:|
| Baseline (shipped) | 0.70 | 0.4016 | 0.496 |
| Mirror averaging (TTA) | 0.80 | 0.3988 | 0.528 |
| Native-scale tiles | 0.35 | 0.290 | 0.356 |
| Tiles + RetinaNet | 0.30 | 0.266 | 0.302 |
| Tiles + crop checker (re-score) | 0.40 | 0.354 | 0.359 |
| Tiles + crop checker (re-label) | 0.25 | 0.318 | 0.272 |

None beats the baseline's 0.402 at its own operating point; mirror averaging comes closest while still gaining 0.032 AP50.

Per-class thresholds were not run — one global setting per pipeline. The two three-stage fusions were scored on detection AP only. A side-by-side panel of ground truth, baseline and the best alternative would need prediction dumps re-rendered from that Kaggle session; the rendered comparisons that exist here are Figures 3 and 4.

### 5.6 The models behind these numbers

Four models were trained for this report. Only the first one is deployed.

| Model | Role | How it was trained | Best test score |
|---|---|---|---|
| Faster R-CNN, ResNet-50 FPN V2, 800 × 1333 | The shipped detector | 43.3 M parameters, 12 epochs on 635 images, best epoch 8 | AP50 0.496, AP 0.329 |
| Faster R-CNN, ResNet-50 FPN V2, 960 × 1600 | Resolution check | Same data and seed, batch size 1, best epoch 5 | AP50 0.493, AP 0.313 |
| RetinaNet, ResNet-50 FPN V2 | Architecture check | Same recipe and 12 epochs; scored with mirror averaging | AP50 0.095, AP 0.053 |
| ConvNeXt-Tiny crop checker | Re-scores or re-labels boxes proposed by the tiled detector | 6,380 crops: 3,190 annotated boxes plus 3,190 background crops | 92.1% accuracy on its own training crops; effect on AP in §5.5 |

Two of these are detectors doing the same job, which is the only clean architecture comparison I have. The two-stage Faster R-CNN was better by a wide margin: 0.496 AP50 against 0.095. That gap is larger than the architecture alone explains, because RetinaNet was still improving when its 12 epochs ran out, so the comparison reflects the schedule as much as the design. On this data, with the training each of them received, Faster R-CNN is the better model I have evidence for.

The crop checker is not a detector. It looks at a box someone else proposed and either re-scores it or re-labels it; as a re-scorer on the tiled detector it moved slice AP from 0.219 to 0.221, and as a re-labeller it dropped AP50 to 0.272. Its 92.1% is training accuracy on the crop bank, not held-out accuracy.

Everything else in §5.5 — mirroring, tiling, weighted box fusion — is a change in how the shipped model is run, not a new model.

---

## 6. Failures and limitations

### 6.1 Five failure cases

At the 0.70 setting the model made 142 false alarms (in 44 of 50 images), missed 66 boxes entirely (in 33 images) and swapped 20 classes. Five cases were exported, at most one per image, in Cell 9 of [`kaggle_notebook.ipynb`](kaggle_notebook.ipynb). The explanations are hypotheses, since I ran no experiment to test causes.

| # | Image | What happened | Likely reason |
|---|---|---|---|
| 1 | `val_4` | False alarm: impacted tooth, score 0.99, IoU 0.00 | Confident about back-of-jaw anatomy the annotators did not mark. If DENTEX labels only some impactions, unlabelled third molars explain it. |
| 2 | `val_38` | Missed a periapical lesion | Rare class (144 training boxes, AP50 0.336), and only 1 of 9 was found at this setting, so the miss is typical rather than odd. |
| 3 | `val_3` | Deep caries called caries, score 0.97 | Box is right, severity is wrong. 14 of 20 swaps are this pair, which points at the deep/shallow line and not at localisation. |
| 4 | `val_33` | Missed a caries | Caries recall is 53.5%, and the scan is shrunk 0.47×, so a small gap-between-teeth cavity is the expected casualty. |
| 5 | `val_13` | False alarm: impacted tooth, score 0.99, IoU 0.00 | Same error as case 1, same confidence. The repeat suggests this is the model's most confident mistake, not a one-off. |

The exported cases are dominated by the model's most confident error, while its most frequent error is only visible in the totals.

| Error | Count | Breakdown |
|---|---:|---|
| False alarms on background | 142 in 44 images | caries 124, impacted 10, deep caries 6, periapical 2 |
| Missed entirely | 66 in 33 images | caries 45, deep caries 12, impacted 5, periapical 4 |
| Swapped between classes | 20 | deep→caries 14, periapical→caries 3, periapical→deep 1, caries→deep 1, caries→impacted 1 |

Nine of every ten false alarms are caries, and seven of every ten swaps collapse deep caries into ordinary caries.

![Figure 3: the five failure cases](assets/eval/failure_cases.png)

*Figure 3: ground truth in green, predictions in red. Panels 1 and 5 are both confident impacted-tooth false alarms; panel 3 is the deep-to-shallow swap.*

![Figure 4: sample test detections](assets/eval/sample_detections.png)

*Figure 4: a typical test scan. The extra red boxes are the caries false-alarm pattern counted above.*

### 6.2 Limitations

§5.5 holds the measured comparisons; this table records what they mean for use.

| Limitation | What I measured | Consequence |
|---|---|---|
| Rare classes | Recall 11.1% (periapical) and 18.8% (deep caries) at 0.70 | Close to unusable at the default setting; lowering it finds more, with more false alarms |
| Deep vs shallow labels | 14 of 32 deep boxes mislabelled | Severity cannot be trusted even when the box is right |
| Caries false alarms | 124 of 142; precision 27.7% | The model is a highlighter, not a filter; expect a heavy review load |
| No class balancing, one global threshold | Not done in any run | Rare classes are penalised twice |
| 2D projection overlap | Not measured | Overlapping anatomy is confounded with lesions and cannot be separated in 2D |
| No tooth numbers | `tooth_fdi` is always `null` | Findings cannot be keyed to a specific tooth |
| Small test set | 50 images, 9 periapical boxes, no patient IDs | Per-class numbers are noisy and patient overlap between splits cannot be ruled out |
| Rule-based input check | No learned classifier | May reject unusual but valid X-rays |
| One dataset | DENTEX only | Performance on other clinics' scanners is unknown |

The first four need new training. The rest need new data, a new model, or a product decision — none of them is fixed by tuning the demo.

---

## 7. Licensing

| Component | Licence | Commercial use |
|---|---|---|
| My code and API, and torchvision Faster R-CNN with COCO weights | BSD-3-Clause | Allowed |
| DENTEX data and my fine-tuned `best_model.pth` | CC BY-NC-SA 4.0 | **Not allowed as it stands.** Needs retraining on commercially licensed data. |
| Runtime libraries (PyTorch, FastAPI, pydicom, NumPy, SciPy, scikit-image, trimesh, nibabel, nnU-Net v2, etc.) | MIT / BSD / Apache-2.0 / HPND | Allowed, with attribution |
| DentalSegmentator weights (3D) and 3D Slicer demo scans | CC BY 4.0 / Slicer BSD-style | Allowed; confirm data reuse with the Slicer project |
| Ultralytics YOLO (not used) | AGPL-3.0 | Only if the service is open-sourced or an enterprise licence is bought |

The code could ship in a commercial product. The trained 2D weights could not. The restriction comes entirely from DENTEX.

---

## 8. Bonus: 3D CBCT

The same `/analyze` endpoint accepts `.nii`, `.nii.gz` and ZIP archives of a 3D scan. I use **DentalSegmentator** (Dot et al., 2024) **as-is, without any training of mine**, to segment five structures: upper jaw, mandible, upper teeth, lower teeth and mandibular canal. My work is the loading, patient-tag removal, 60 mm³ island cleanup, distance measurement, 3D mesh export and the browser viewer.

The reported finding is the smallest straight-line gap in millimetres between the lower teeth and the nerve canal, per half of the scan. On the Post-surgery demo scan the gaps are 0.71 mm and 1.58 mm; on the Pre-surgery scan, 2.12 mm and 1.50 mm. The two halves are a voxel midpoint split, not yet mapped to the patient's real left and right, and the gap is not a traced nerve path or a surgical risk score. On a Tesla T4 one full run takes 131 s and uses 4.9 GB of GPU memory; the model prediction matches the published reference mask at mean Dice 0.9639 on the Post scan.

The free Space has no 3D model, so a raw upload there returns HTTP 503 and only the two bundled pre-segmented scans work. The production plan (queued jobs, warm GPU workers, compressed meshes) is a design in [`README.md`](README.md) §7, not a benchmark.

![Figure 5: CBCT multi-planar reconstruction, Post scan](assets/eval/cbct_ortho_preview.png)

![Figure 5: CBCT multi-planar reconstruction, Pre scan](assets/eval/cbct_ortho_pre_preview.png)

*Figure 5: axial, coronal, sagittal and coronal MIP panels for the Post (top) and Pre (bottom) demo volumes.*

---

## 9. What I would do with two more weeks

| Priority | Work | Expected effect | Effort |
|---|---|---|---|
| 1 | Per-class confidence settings, tuned on validation, no retraining | Likely recovers rare-class recall at some cost to caries precision; not measured | 1–2 days |
| 2 | Class-balanced loss and oversampling for periapical and deep caries | The only plausible route to improving 11% and 19% recall; unmeasured | about 1 week |
| 3 | RetinaNet with a schedule suited to 635 images | Current run is undertrained, so the architecture is untested | 2–3 GPU days |
| 4 | ONNX export with INT8 quantisation | Target is well under the 4.6 s now; unmeasured | 3–5 days |
| 5 | Tooth-number (FDI) stage to fill `tooth_fdi` | Turns boxes into per-tooth findings | about 2 weeks |

Mirror averaging on a GPU deployment is the cheap extra (+0.032 AP50 for twice the compute); its CPU cost was never measured.

---

## 10. References

- Hamamci et al., "DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays", MICCAI 2023.
- Dot et al., "DentalSegmentator", *Journal of Dentistry*, 2024. Zenodo 10.5281/zenodo.10829675.
- Isensee et al., "nnU-Net", *Nature Methods*, 2021.
- Torchvision detection models (Faster R-CNN, RetinaNet) and COCO weights, BSD-3-Clause.
- 3D Slicer, "CBCT Dental Surgery" sample data.
- Akyon et al., "Slicing Aided Hyper Inference", arXiv:2202.06934. Solovyev et al., "Weighted Boxes Fusion", 2021. Lin et al., Microsoft COCO, ECCV 2014 (`pycocotools`).

*AI tools used are listed in [AI_DISCLOSURE.md](AI_DISCLOSURE.md).*
