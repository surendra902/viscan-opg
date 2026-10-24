# AI Tool Disclosure Note

**Candidate:** Surendra Goud  
**Role:** AI Engineer (Round 2 Take-Home Assessment)  
**Date:** October 2026  

---

## 1. AI Coding Assistants Used
- **Google Antigravity AI Assistant (Claude 3.7 / Gemini 2.5 / DeepMind pairing engine)**

## 2. Scope of AI Usage
The assistant was used in a collaborative pair-programming capacity for:
1. **Scaffolding and Boilerplate:** Drafting boilerplate code for the FastAPI backend, PyTorch Dataset collate functions, and standard pycocotools metric calculation wrappers.
2. **Schema Verification & AST Audits:** Inspecting constructor defaults and API schemas (e.g., verifying torchvision Faster R-CNN RPN score thresholds and FastRCNNPredictor head configurations).
3. **Drafting Documentation & Test Fixtures:** Generating structured markdown tables, Mermaid architecture diagrams, and pytest unit test cases for DICOM windowing and bounding box coordinate bounds clipping.

## 3. Human Engineering & Accountability
- **System Architecture & Design Decisions:** The decision to select Faster R-CNN ResNet-50 FPN V2 for stronger feature extraction while maintaining BSD-3-Clause code compliance (rejecting AGPL YOLO), the design of the DICOM MONOCHROME1 ingestion pipeline, the selection of the DENTEX MICCAI benchmark, and the choice of the inference-resolution scaling experiment were authored, evaluated, and directed by the candidate.
- **Code Comprehension:** The candidate understands and can explain every function, class, formula, and hyperparameter in this repository during the technical walkthrough call.
