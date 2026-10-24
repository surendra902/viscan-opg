"""
FastAPI Application and Interactive Web UI for Viscan OPG.

Features:
- POST /analyze endpoint accepting multipart file uploads (PNG, JPEG, 2D DICOM).
- Side-by-side UI showing original vs annotated images, findings list, and download buttons.
- Robust error handling for corrupt files, non-dental uploads, and oversized payloads.
- Strict patient privacy enforcement (no PHI persisted or logged).
- Single-command execution: python app.py or uvicorn app:app.
"""

import os
import io
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from service import DentalAnalysisService, DISCLAIMER_TEXT, ModelNotLoadedError
from imaging import ImageProcessingError, InvalidImageError, UnsupportedDICOMError
from input_gate import NonDentalInputError

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI application
app = FastAPI(
    title="Viscan OPG Dental Radiology Analysis API",
    description="Automated 2D OPG finding detection for dentist clinical decision support.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount results directory for serving generated annotated images
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Initialize analysis service
weights_file = os.getenv("VISCAN_WEIGHTS_PATH", "checkpoints/best_model.pth")
service = DentalAnalysisService(
    weights_path=weights_file if os.path.exists(weights_file) else None,
    results_dir=str(RESULTS_DIR),
)


@app.get("/health")
def health_check():
    """Health status and model readiness endpoint."""
    model_ready = service.model is not None
    return {
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "device": str(service.device),
        "results_dir": str(RESULTS_DIR),
    }


@app.post("/analyze")
async def analyze_opg(
    file: UploadFile = File(..., description="Dental OPG (PNG, JPG, 2D DICOM .dcm) or 3D CBCT (.nii, .nii.gz, DICOM .zip)"),
    confidence_threshold: Optional[float] = Query(0.70, ge=0.01, le=1.0, description="Detection confidence threshold"),
):
    """
    Analyzes an uploaded 2D dental panoramic radiograph (OPG) or 3D CBCT scan.
    Returns structured findings, deterministic summary, annotated image URL,
    and (for 3D CBCT) 3D coordinates, nerve clearance in mm, and .glb / .stl mesh URLs.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = service.analyze_bytes(
            file_bytes=content,
            filename=file.filename or "uploaded_image",
            custom_threshold=confidence_threshold,
        )
        return JSONResponse(status_code=200, content=result)

    except HTTPException:
        raise
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {str(e)}")
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    except UnsupportedDICOMError as e:
        raise HTTPException(status_code=422, detail=f"Unsupported DICOM format: {str(e)}")
    except NonDentalInputError as e:
        raise HTTPException(status_code=422, detail=f"Input rejected: {str(e)}")
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal analysis failure: {str(e)}")


@app.post("/analyze/sample_cbct")
def analyze_sample_cbct():
    """Runs the 3D CBCT post-processing, nerve-distance, and 3D mesh pipeline on the verified CBCT sample."""
    sample_path = Path("samples/cbct/PostDentalSurgery_Segmentation.nii.gz")
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample CBCT volume not found in samples/cbct/.")
    result = service.analyze_cbct_bytes(sample_path.read_bytes(), sample_path.name)
    return JSONResponse(status_code=200, content=result)


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Interactive side-by-side web UI for dentists and evaluators."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Viscan Dental OPG AI Assistant</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --border: #334155;
      --caries: #FFD700;
      --deep-caries: #FF4500;
      --periapical: #BA55D3;
      --impacted: #00BFFF;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .header {
      max-width: 1300px;
      margin: 0 auto 24px auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    .header h1 { font-size: 24px; font-weight: 700; color: var(--primary); }
    .disclaimer-banner {
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid #f59e0b;
      color: #fbbf24;
      padding: 10px 16px;
      border-radius: 8px;
      margin-bottom: 24px;
      font-size: 14px;
      text-align: center;
      max-width: 1300px;
      margin-left: auto;
      margin-right: auto;
    }
    .main-grid {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 24px;
      max-width: 1300px;
      margin: 0 auto;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }
    .card h2 { font-size: 18px; margin-bottom: 16px; color: var(--text); }
    .upload-area {
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 24px;
      text-align: center;
      cursor: pointer;
      margin-bottom: 16px;
      transition: border-color 0.2s;
    }
    .upload-area:hover { border-color: var(--primary); }
    input[type="file"] { display: none; }
    .slider-group { margin-bottom: 16px; }
    .slider-group label { display: block; font-size: 13px; color: var(--text-muted); margin-bottom: 4px; }
    .btn {
      display: block;
      width: 100%;
      background: var(--primary);
      color: #0f172a;
      border: none;
      padding: 12px;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.2s;
      font-size: 15px;
    }
    .btn:hover { background: var(--primary-hover); }
    .btn:disabled { background: var(--border); color: var(--text-muted); cursor: not-allowed; }
    .comparison-container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    .image-box {
      background: #020617;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
      min-height: 280px;
      display: flex;
      flex-direction: column;
    }
    .image-box-title {
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(255,255,255,0.05);
      border-bottom: 1px solid var(--border);
      color: var(--text-muted);
    }
    .image-preview {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 8px;
    }
    .image-preview img {
      max-width: 100%;
      max-height: 380px;
      object-fit: contain;
      border-radius: 4px;
    }
    .findings-card {
      margin-top: 16px;
    }
    .finding-badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
      color: #000;
      margin-right: 6px;
    }
    .bg-impacted { background: var(--impacted); }
    .bg-caries { background: var(--caries); }
    .bg-periapical { background: var(--periapical); }
    .bg-deep-caries { background: var(--deep-caries); }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
    th { color: var(--text-muted); }
    .actions-bar {
      display: flex;
      gap: 12px;
      margin-top: 16px;
    }
    .action-btn {
      flex: 1;
      background: rgba(255,255,255,0.1);
      color: var(--text);
      text-decoration: none;
      text-align: center;
      padding: 10px;
      border-radius: 6px;
      font-size: 13px;
      border: 1px solid var(--border);
      cursor: pointer;
    }
    .action-btn:hover { background: rgba(255,255,255,0.2); }
    .status-msg {
      margin-top: 12px;
      font-size: 13px;
      padding: 8px 12px;
      border-radius: 6px;
      display: none;
    }
    .status-error { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .status-info { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
  </style>
</head>
<body>
  <div class="header">
    <h1>Viscan Dental OPG AI Assistant</h1>
    <span style="font-size: 13px; color: var(--text-muted);">Faster R-CNN ResNet-50 FPN V2</span>
  </div>

  <div class="disclaimer-banner">
    ⚠️ <strong>Clinical Notice:</strong> AI-assisted findings for dentist review. Not a diagnosis. All bounding boxes and labels must be verified by a licensed dental professional.
  </div>

  <div class="main-grid">
    <!-- Left: Controls -->
    <div class="card">
      <h2>Upload 2D OPG or 3D CBCT</h2>
      <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
        <p style="font-size: 14px; font-weight: 600;" id="fileNameDisplay">Select OPG or 3D CBCT Scan</p>
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">2D: PNG, JPG, DICOM (.dcm) · 3D CBCT: NIfTI (.nii, .nii.gz) or DICOM (.zip)</p>
      </div>
      <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.dcm,.nii,.gz,.zip">

      <div class="slider-group">
        <label for="confSlider">Confidence Threshold: <span id="confVal">0.70</span></label>
        <input type="range" id="confSlider" min="0.10" max="0.90" step="0.05" value="0.70" style="width: 100%;">
      </div>

      <button class="btn" id="analyzeBtn" disabled>Run Inference</button>
      <button class="btn" id="sampleCbctBtn" style="margin-top: 8px; background: #0d9488; color: #fff;">Load Verified 3D CBCT Sample</button>
      <div class="status-msg" id="statusBox"></div>

      <div style="margin-top: 24px; font-size: 12px; color: var(--text-muted);">
        <p style="font-weight: 600; margin-bottom: 6px;">2D OPG Conditions (Faster R-CNN):</p>
        <p><span class="finding-badge bg-caries">Caries</span> Interproximal/occlusal</p>
        <p style="margin-top: 4px;"><span class="finding-badge bg-deep-caries">Deep Caries</span> Pulpal proximity</p>
        <p style="margin-top: 4px;"><span class="finding-badge bg-periapical">Periapical Lesion</span> Apical radiolucency</p>
        <p style="margin-top: 4px;"><span class="finding-badge bg-impacted">Impacted Tooth</span> Eruption obstruction</p>
        <p style="font-weight: 600; margin: 12px 0 6px 0;">3D CBCT Structures (DentalSegmentator):</p>
        <p>Maxilla · Mandible · Upper/Lower Teeth · Mandibular Canal + 3D Nerve Clearance (mm)</p>
      </div>
    </div>

    <!-- Right: Results -->
    <div class="card">
      <h2>Side-by-Side Analysis</h2>
      <div class="comparison-container">
        <div class="image-box">
          <div class="image-box-title">Original Radiograph / Multi-Planar CBCT</div>
          <div class="image-preview" id="origPreview">
            <span style="color: var(--text-muted); font-size: 13px;">No radiograph loaded</span>
          </div>
        </div>
        <div class="image-box">
          <div class="image-box-title">Annotated Findings / 5-Class CBCT Overlay</div>
          <div class="image-preview" id="annotPreview">
            <span style="color: var(--text-muted); font-size: 13px;">Awaiting analysis</span>
          </div>
        </div>
      </div>

      <!-- Interactive 3D CBCT Mesh Viewer (shown when modality === 'CBCT') -->
      <div id="cbct3dSection" style="display: none; margin-top: 18px; background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
          <h3 style="font-size: 15px; color: var(--primary);">Interactive 3D CBCT Surface Mesh (Marching Cubes — Orbit / Zoom / Pan)</h3>
          <span style="font-size: 12px; color: var(--text-muted);">Pretrained DentalSegmentator (nnU-Net v2, CC BY 4.0)</span>
        </div>
        <model-viewer id="cbctModelViewer" src="" alt="3D CBCT Dental Segmentation" camera-controls auto-rotate shadow-intensity="1" style="width: 100%; height: 420px; background-color: #090d16; border-radius: 6px;"></model-viewer>
      </div>

      <div class="findings-card">
        <h3 style="font-size: 16px; margin-bottom: 8px;">Clinical Summary</h3>
        <p id="summaryText" style="font-size: 14px; color: var(--text-muted);">Upload an OPG radiograph or 3D CBCT volume to generate findings.</p>
        
        <table id="findingsTable" style="display: none;">
          <thead>
            <tr>
              <th>Finding Class</th>
              <th>Confidence / Distance</th>
              <th>2D Box / 3D Voxel Coordinates</th>
              <th>Tooth FDI / Hemimandible</th>
            </tr>
          </thead>
          <tbody id="findingsBody"></tbody>
        </table>
      </div>

      <div class="actions-bar" id="downloadBar" style="display: none; flex-wrap: wrap; gap: 10px;">
        <a class="action-btn" id="downloadImgBtn" target="_blank" download>Download Annotated PNG</a>
        <a class="action-btn" id="downloadJsonBtn" target="_blank" download>Download JSON Report</a>
        <a class="action-btn" id="downloadGlbBtn" target="_blank" download style="display: none; background: #0d9488;">Download 3D Scene (.glb)</a>
      </div>
    </div>
  </div>

  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
  <script>
    let selectedFile = null;
    let latestResult = null;

    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const sampleCbctBtn = document.getElementById('sampleCbctBtn');
    const confSlider = document.getElementById('confSlider');
    const confVal = document.getElementById('confVal');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const statusBox = document.getElementById('statusBox');
    const origPreview = document.getElementById('origPreview');
    const annotPreview = document.getElementById('annotPreview');
    const summaryText = document.getElementById('summaryText');
    const findingsTable = document.getElementById('findingsTable');
    const findingsBody = document.getElementById('findingsBody');
    const downloadBar = document.getElementById('downloadBar');
    const downloadImgBtn = document.getElementById('downloadImgBtn');
    const downloadJsonBtn = document.getElementById('downloadJsonBtn');
    const downloadGlbBtn = document.getElementById('downloadGlbBtn');
    const cbct3dSection = document.getElementById('cbct3dSection');
    const cbctModelViewer = document.getElementById('cbctModelViewer');

    confSlider.oninput = () => { confVal.textContent = parseFloat(confSlider.value).toFixed(2); };

    fileInput.onchange = (e) => {
      if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        fileNameDisplay.textContent = selectedFile.name;
        analyzeBtn.disabled = false;
        showStatus('', false);

        // Clear any stale results from a previous analysis
        latestResult = null;
        annotPreview.innerHTML = '<span style="color: var(--text-muted); font-size: 13px;">Awaiting analysis</span>';
        summaryText.textContent = 'Upload an OPG radiograph or 3D CBCT volume to generate findings.';
        findingsBody.innerHTML = '';
        findingsTable.style.display = 'none';
        downloadBar.style.display = 'none';
        cbct3dSection.style.display = 'none';

        const lower = selectedFile.name.toLowerCase();
        if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
          const reader = new FileReader();
          reader.onload = (re) => {
            origPreview.innerHTML = `<img src="${re.target.result}" alt="Original">`;
          };
          reader.readAsDataURL(selectedFile);
        } else {
          origPreview.innerHTML = `<span style="color: var(--text-muted); font-size: 13px;">Medical volume loaded: ${selectedFile.name} (preview rendered upon analysis)</span>`;
        }
      }
    };

    sampleCbctBtn.onclick = async () => {
      sampleCbctBtn.disabled = true;
      showStatus('Running 3D CBCT post-processing & mesh pipeline on PostDentalSurgery sample...', true, false);
      try {
        const resp = await fetch('/analyze/sample_cbct', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Sample CBCT analysis failed.');
        latestResult = data;
        displayResults(data);
        showStatus(`3D CBCT analysis complete in ${data.inference_ms}ms.`, true, false);
      } catch (err) {
        showStatus(err.message, true, true);
      } finally {
        sampleCbctBtn.disabled = false;
      }
    };

    analyzeBtn.onclick = async () => {
      if (!selectedFile) return;
      analyzeBtn.disabled = true;
      showStatus('Analyzing scan...', true, false);

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const resp = await fetch(`/analyze?confidence_threshold=${confSlider.value}`, {
          method: 'POST',
          body: formData
        });

        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || 'Analysis request failed.');
        }

        latestResult = data;
        displayResults(data);
        showStatus(`Analysis complete in ${data.inference_ms}ms.`, true, false);
      } catch (err) {
        showStatus(err.message, true, true);
      } finally {
        analyzeBtn.disabled = false;
      }
    };

    function displayResults(data) {
      if (data.original_image_url) {
        origPreview.innerHTML = `<img src="${data.original_image_url}" alt="Original">`;
      }
      summaryText.textContent = data.summary;
      annotPreview.innerHTML = `<img src="${data.annotated_image_url}" alt="Annotated">`;

      if (data.modality === 'CBCT' && data.mesh_glb_url) {
        cbct3dSection.style.display = 'block';
        cbctModelViewer.src = data.mesh_glb_url;
        downloadGlbBtn.style.display = 'inline-block';
        downloadGlbBtn.href = data.mesh_glb_url;
        downloadGlbBtn.download = `${data.image_id}_scene.glb`;
      } else {
        cbct3dSection.style.display = 'none';
        downloadGlbBtn.style.display = 'none';
      }

      findingsBody.innerHTML = '';
      if (data.findings.length > 0) {
        findingsTable.style.display = 'table';
        data.findings.forEach(f => {
          const tr = document.createElement('tr');
          const badgeClass = 'bg-' + f.class.replace(/_/g, '-');
          const metricStr = f.min_distance_mm !== undefined
            ? `<strong>${f.min_distance_mm.toFixed(2)} mm</strong>`
            : `<strong>${(f.confidence * 100).toFixed(0)}%</strong>`;
          const boxArr = f.bbox_3d_voxel || f.bbox || [];
          const locStr = f.hemimandible
            ? `${f.hemimandible.toUpperCase()} (voxel ${JSON.stringify(f.closest_lower_tooth_voxel)})`
            : (f.tooth_fdi !== null ? f.tooth_fdi : '<em>null</em>');
          tr.innerHTML = `
            <td><span class="finding-badge ${badgeClass}">${f.class}</span></td>
            <td>${metricStr}</td>
            <td><code>[${boxArr.join(', ')}]</code></td>
            <td>${locStr}</td>
          `;
          findingsBody.appendChild(tr);
        });
      } else {
        findingsTable.style.display = 'none';
      }

      downloadBar.style.display = 'flex';
      downloadImgBtn.href = data.annotated_image_url;
      downloadImgBtn.download = `${data.image_id}_annotated.png`;

      const jsonBlob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      downloadJsonBtn.href = URL.createObjectURL(jsonBlob);
      downloadJsonBtn.download = `${data.image_id}_report.json`;
    }

    function showStatus(msg, show, isError = false) {
      if (!show) { statusBox.style.display = 'none'; return; }
      statusBox.style.display = 'block';
      statusBox.textContent = msg;
      statusBox.className = 'status-msg ' + (isError ? 'status-error' : 'status-info');
    }
  </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
