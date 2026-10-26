# AI Tool Disclosure Note

**Candidate:** Surendra Goud  
**Role:** AI Engineer (Round 2 Take-Home Assessment)  
**Date:** October 2026  

---

## 1. Tools used, and what each did

| Tool | Vendor | What I used it for |
|---|---|---|
| Google Antigravity | Google | Code scaffolding and boilerplate: the FastAPI backend and its inline HTML/JS UI, PyTorch Dataset collate functions, and `pycocotools` metric wrappers. Debugging, pytest authoring, and report formatting. |
| Qoder Desktop | Qoder | Repository audit, Docker and README review, `scripts/smoke_test.py` authoring, the tooth-numbering estimator (`fdi.py`, `scripts/dentex_fdi.py`, `tests/test_fdi.py`), the paired CPU latency mode in `scripts/benchmark_cpu.py`, the `REPORT.md` → PDF renderer and its four-page check (`scripts/render_report_pdf.py`), and review and rewording of this submission, including this conversation. |

No other AI assistants were used. Every training run, evaluation and latency number in the submission came from the executed notebooks and scripts named in the report; no metric was produced by a language model.

## 2. Scope of AI usage

1. **Scaffolding and Boilerplate:** Drafting boilerplate code for the FastAPI backend and its inline HTML/JS UI, PyTorch Dataset collate functions, and `pycocotools` metric calculation wrappers.
2. **Codebase Verification and Audit Remediation:** Cross-checking notebook execution outputs (`kaggle_notebook.ipynb`, `kaggle_beat_benchmark_notebook.ipynb`) against repository scripts (`prepare_data.py`, `train.py`, `evaluate.py`, `imaging.py`, `input_gate.py`, `service.py`, `app.py`), fixing DICOM Modality LUT before VOI LUT ordering, negative-origin bounding box clipping, HTTP 503 fail-closed behavior when model weights are missing, input gate radiographic heuristics, and the first-run checkpoint download and SHA-256 verification path.
3. **Documentation and Test Fixtures:** Formatting `REPORT.md` from metrics and plots extracted directly from the executed Kaggle notebooks, and writing the `pytest` suite and `scripts/smoke_test.py`.
4. **Feature code:** `fdi.py`, `scripts/dentex_fdi.py` and `tests/test_fdi.py` (the arch-geometry tooth-number estimator, its label fetch and its measurements) were drafted with Qoder and reviewed line by line. Every number quoted for that feature is printed by the script itself and stored in `assets/fdi/fdi_geometry.json` and `assets/eval/fdi_geometry.json`.

## 3. Human Engineering and Accountability

- **System Architecture and Design Decisions:** Selecting Faster R-CNN ResNet-50 FPN V2 (`COCO_V1`) for two-stage detection under BSD-3-Clause code licensing (with explicit `CC BY-NC-SA 4.0` disclosure for DENTEX-derived weights), designing the 3-way dataset split (`635` train / `70` val / `50` test), executing training and evaluation on Kaggle Tesla T4 GPU, and reviewing all technical claims and failure modes were directed and verified by the candidate.
- **Tooth numbering:** answering "which tooth is this finding on" with a geometry table fitted on DENTEX's own quadrant and tooth-position labels, instead of training a second detector, and keeping the result labelled an estimate in the JSON, on the drawn box and in the limits section, were the candidate's calls. So was the decision not to add classes DENTEX does not label.
- **Code Comprehension:** The candidate understands and can explain every function, class, metric, and architectural trade-off in this repository during the technical walkthrough.
