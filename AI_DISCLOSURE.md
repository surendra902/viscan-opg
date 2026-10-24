# AI Tool Disclosure Note

**Candidate:** Surendra Goud  
**Role:** AI Engineer (Round 2 Take-Home Assessment)  
**Date:** October 2026  

---

## 1. AI Coding Assistants Used
- **Google Antigravity AI Assistant** (code scaffolding, debugging, test authoring, and report formatting)
- **Hermes / Automated Audit & Review Agent** (independent static/dynamic audit verification and contract testing)

## 2. Scope of AI Usage
The assistant was used in a collaborative pair-programming and code-audit capacity for:
1. **Scaffolding and Boilerplate:** Drafting boilerplate code for the FastAPI backend, Gradio interface, PyTorch Dataset collate functions, and `pycocotools` metric calculation wrappers.
2. **Codebase Verification & Audit Remediation:** Cross-checking notebook execution outputs (`kaggle_notebook.ipynb`) against repository scripts (`prepare_data.py`, `train.py`, `evaluate.py`, `imaging.py`, `input_gate.py`, `service.py`, `app.py`), fixing DICOM Modality LUT precede-VOI-LUT ordering, negative-origin bounding box clipping, HTTP 503 fail-closed behavior when model weights are missing, and input gate radiographic heuristics.
3. **Documentation, Report Generation & Test Fixtures:** Formatting `REPORT.md` and ReportLab PDF generation scripts (`generate_pdf_report.py`, `generate_dataset_evaluation_pdf.py`) using verified metrics and plot outputs extracted directly from the executed Kaggle training notebook, and writing `pytest` unit and regression tests.

## 3. Human Engineering & Accountability
- **System Architecture & Design Decisions:** Selecting Faster R-CNN ResNet-50 FPN V2 (`COCO_V1`) for two-stage detection under BSD-3-Clause code licensing (with explicit `CC BY-NC-SA 4.0` disclosure for DENTEX-derived weights), designing the 3-way dataset split (`635` train / `70` val / `50` test), executing training and evaluation on Kaggle Tesla T4 GPU, and reviewing all technical claims and failure modes were directed and verified by the candidate.
- **Code Comprehension:** The candidate understands and can explain every function, class, metric, and architectural trade-off in this repository during the technical walkthrough.
