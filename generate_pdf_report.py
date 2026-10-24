"""
Generates an executive, publication-quality Clinical Evaluation PDF Report
for the Viscan Dental OPG AI Assistant.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
    PageBreak,
)

PDF_OUTPUT_1 = Path(r"c:\Users\Errachandhanam\Videos\1\viscan-opg\Viscan_OPG_Model_Evaluation_Report.pdf")
PDF_OUTPUT_2 = Path(r"C:\Users\Errachandhanam\Downloads\Viscan_OPG_Model_Evaluation_Report.pdf")

doc = SimpleDocTemplate(
    str(PDF_OUTPUT_1),
    pagesize=letter,
    leftMargin=0.5 * inch,
    rightMargin=0.5 * inch,
    topMargin=0.5 * inch,
    bottomMargin=0.5 * inch,
)

styles = getSampleStyleSheet()

# Custom Color Palette
PRIMARY = colors.HexColor("#1E3A8A")     # Dark Navy Blue
SECONDARY = colors.HexColor("#0284C7")   # Medical Cyan / Azure
ACCENT = colors.HexColor("#0D9488")      # Teal
DARK_TEXT = colors.HexColor("#0F172A")   # Slate 900
MUTED_TEXT = colors.HexColor("#475569")  # Slate 600
BG_LIGHT = colors.HexColor("#F8FAFC")    # Slate 50
BORDER_COLOR = colors.HexColor("#CBD5E1") # Slate 300

# Typography Styles
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=PRIMARY,
    spaceAfter=4,
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=11,
    leading=15,
    textColor=SECONDARY,
    spaceAfter=8,
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=17,
    textColor=PRIMARY,
    spaceBefore=10,
    spaceAfter=6,
)

body_style = ParagraphStyle(
    "DocBody",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9,
    leading=13,
    textColor=DARK_TEXT,
    spaceAfter=5,
)

meta_style = ParagraphStyle(
    "MetaText",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=12,
    textColor=MUTED_TEXT,
)

small_style = ParagraphStyle(
    "SmallNote",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=7.5,
    leading=10,
    textColor=MUTED_TEXT,
)

table_header_style = ParagraphStyle(
    "TableHeader",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8.5,
    leading=11,
    textColor=colors.white,
    alignment=1,  # Center
)

table_cell_style = ParagraphStyle(
    "TableCell",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=11,
    textColor=DARK_TEXT,
)

table_cell_bold = ParagraphStyle(
    "TableCellBold",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8,
    leading=11,
    textColor=DARK_TEXT,
)

disclaimer_style = ParagraphStyle(
    "DisclaimerText",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#991B1B"),
)

story = []

# Header / Title Block
story.append(Paragraph("Viscan Dental OPG AI Assistant", title_style))
story.append(Paragraph("Comprehensive Model Evaluation, Benchmarking & Clinical Metrics Report", subtitle_style))

# Quick Meta Card
meta_data = [
    [
        Paragraph("<b>Model:</b> Faster R-CNN ResNet-50 FPN V2 (43.27M params, COCO_V1 init)<br/><b>Benchmark:</b> DENTEX (MICCAI 2023) — Held-Out Test Set (50 images, 182 findings)", meta_style),
        Paragraph("<b>Operating Threshold:</b> 0.70 (Val Macro-F1=0.556; Test Macro-F1=0.402)<br/><b>Live Demo:</b> huggingface.co/spaces/surendra08/viscan-opg<br/><b>Weights SHA256:</b> e0b49b12f48fc71ec1f583db08f107650659571e7c2b9cbd454dbfed13b46a1f", meta_style),
    ]
]
meta_table = Table(meta_data, colWidths=[3.7 * inch, 3.8 * inch])
meta_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
    ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 6),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(meta_table)
story.append(Spacer(1, 8))

# 1. Primary COCO Metrics
story.append(Paragraph("1. Primary Benchmark Metrics (Held-Out Test Set: 50 images)", h1_style))

coco_summary_data = [
    [Paragraph("mAP @ IoU=0.50", table_header_style), Paragraph("mAP @ IoU=0.50:0.95 (Primary)", table_header_style), Paragraph("mAP @ IoU=0.75 (Strict)", table_header_style), Paragraph("Average Recall (AR@100)", table_header_style)],
    [Paragraph("<b>49.6%</b> (0.496)", table_cell_style), Paragraph("<b>32.9%</b> (0.329)", table_cell_style), Paragraph("<b>38.3%</b> (0.383)", table_cell_style), Paragraph("<b>68.3%</b> (0.683)", table_cell_style)],
]
coco_table = Table(coco_summary_data, colWidths=[1.875 * inch] * 4)
coco_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 5),
    ("BACKGROUND", (0, 1), (-1, 1), BG_LIGHT),
]))
story.append(coco_table)
story.append(Spacer(1, 8))

# 2. Per-Class Metrics Table
story.append(Paragraph("2. Per-Class Benchmark & Clinical Operating Metrics (Threshold = 0.70)", h1_style))

class_data = [
    [
        Paragraph("Pathology Class", table_header_style),
        Paragraph("COCO AP@50", table_header_style),
        Paragraph("COCO AP@50:95", table_header_style),
        Paragraph("Precision", table_header_style),
        Paragraph("Recall", table_header_style),
        Paragraph("F1-Score", table_header_style),
        Paragraph("Test Support", table_header_style),
        Paragraph("Clinical Reliability", table_header_style),
    ],
    [
        Paragraph("<b>Impacted Tooth</b>", table_cell_bold),
        Paragraph("83.3%", table_cell_style),
        Paragraph("54.1%", table_cell_style),
        Paragraph("<b>76.1%</b>", table_cell_style),
        Paragraph("<b>87.5%</b>", table_cell_style),
        Paragraph("<b>0.814</b>", table_cell_style),
        Paragraph("40", table_cell_style),
        Paragraph("<b>High Reliability:</b> 87.5% sensitivity with few false alarms.", table_cell_style),
    ],
    [
        Paragraph("<b>Deep Caries</b>", table_cell_bold),
        Paragraph("45.7%", table_cell_style),
        Paragraph("30.2%", table_cell_style),
        Paragraph("<b>42.9%</b>", table_cell_style),
        Paragraph("<b>18.8%</b>", table_cell_style),
        Paragraph("<b>0.261</b>", table_cell_style),
        Paragraph("32", table_cell_style),
        Paragraph("Conservative detector; 70% of errors are classified as standard caries.", table_cell_style),
    ],
    [
        Paragraph("<b>Caries</b>", table_cell_bold),
        Paragraph("35.8%", table_cell_style),
        Paragraph("24.4%", table_cell_style),
        Paragraph("<b>27.7%</b>", table_cell_style),
        Paragraph("<b>53.5%</b>", table_cell_style),
        Paragraph("<b>0.365</b>", table_cell_style),
        Paragraph("101", table_cell_style),
        Paragraph("Sensitive screener (captures over half); prone to enamel border FPs.", table_cell_style),
    ],
    [
        Paragraph("<b>Periapical Lesion</b>", table_cell_bold),
        Paragraph("33.6%", table_cell_style),
        Paragraph("23.0%", table_cell_style),
        Paragraph("<b>33.3%</b>", table_cell_style),
        Paragraph("<b>11.1%</b>", table_cell_style),
        Paragraph("<b>0.167</b>", table_cell_style),
        Paragraph("9", table_cell_style),
        Paragraph("Small sample size (n=9); requires focal loss or oversampling.", table_cell_style),
    ],
]

class_table = Table(
    class_data,
    colWidths=[1.1 * inch, 0.75 * inch, 0.85 * inch, 0.7 * inch, 0.65 * inch, 0.65 * inch, 0.7 * inch, 2.1 * inch],
)
class_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 4),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
]))
story.append(class_table)
story.append(Spacer(1, 8))

# 3. Confusion Matrix Table
story.append(Paragraph("3. Test Set Confusion Matrix (Operating Threshold = 0.70)", h1_style))

cm_data = [
    [
        Paragraph("Ground Truth \\ Predicted", table_header_style),
        Paragraph("Background (FN)", table_header_style),
        Paragraph("Impacted Tooth", table_header_style),
        Paragraph("Caries", table_header_style),
        Paragraph("Periapical", table_header_style),
        Paragraph("Deep Caries", table_header_style),
    ],
    [Paragraph("<b>Background (False Positives)</b>", table_cell_bold), Paragraph("—", table_cell_style), Paragraph("10", table_cell_style), Paragraph("<b>124</b>", table_cell_style), Paragraph("2", table_cell_style), Paragraph("6", table_cell_style)],
    [Paragraph("<b>Impacted Tooth</b>", table_cell_bold), Paragraph("5", table_cell_style), Paragraph("<b>35</b>", table_cell_style), Paragraph("0", table_cell_style), Paragraph("0", table_cell_style), Paragraph("0", table_cell_style)],
    [Paragraph("<b>Caries</b>", table_cell_bold), Paragraph("45", table_cell_style), Paragraph("1", table_cell_style), Paragraph("<b>54</b>", table_cell_style), Paragraph("0", table_cell_style), Paragraph("1", table_cell_style)],
    [Paragraph("<b>Periapical Lesion</b>", table_cell_bold), Paragraph("4", table_cell_style), Paragraph("0", table_cell_style), Paragraph("3", table_cell_style), Paragraph("<b>1</b>", table_cell_style), Paragraph("1", table_cell_style)],
    [Paragraph("<b>Deep Caries</b>", table_cell_bold), Paragraph("12", table_cell_style), Paragraph("0", table_cell_style), Paragraph("<b>14</b>", table_cell_style), Paragraph("0", table_cell_style), Paragraph("<b>6</b>", table_cell_style)],
]

cm_table = Table(cm_data, colWidths=[2.2 * inch, 1.1 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch])
cm_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 4),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
]))
story.append(cm_table)
story.append(Spacer(1, 8))

# 4. Diagnostic Key Findings & Engineering Analysis
story.append(Paragraph("4. Diagnostic & Error Analysis Takeaways", h1_style))
findings_text = """
• <b>Zero Anatomical Cross-Confusion:</b> Impacted teeth had 0 cross-classifications with carious or periapical lesions (35/40 TP, 5 FN).<br/>
• <b>Boundary Severity Distinction:</b> Out of 20 total inter-class misclassifications, 14 are <code>deep_caries</code> predicted as <code>caries</code> (severity grading ambiguity near the dentin/pulp border; 6/32 TP as deep_caries, 14/32 localized as caries, 12/32 missed).<br/>
• <b>False Alarm Screener Profile:</b> 124 of the 142 background False Positives occur on low-contrast caries proposals near interproximal contacts and existing restorations.<br/>
• <b>5 Concrete Test Failure Cases:</b> Case 1 <code>val_4.png</code> (FP <code>impacted_tooth</code> 0.99), Case 2 <code>val_38.png</code> (FN missed <code>periapical_lesion</code>), Case 3 <code>val_3.png</code> (Misclass <code>deep_caries</code>→<code>caries</code> 0.97), Case 4 <code>val_33.png</code> (FN missed <code>caries</code>), Case 5 <code>val_13.png</code> (FP <code>impacted_tooth</code> 0.99).<br/>
• <b>Resolution Variant Finding:</b> The 960x1600 variant (bs=1) improved strict mAP@75 (0.410 vs 0.383), while baseline (800x1333, bs=2) achieved higher primary Val mAP@50:95 (0.360 vs 0.354) and Test mAP@50:95 (0.329 vs 0.313).
"""
story.append(Paragraph(findings_text, body_style))
story.append(Spacer(1, 6))

# 5. Visual Plots Section (Side by Side Images)
story.append(Paragraph("5. Empirical Visualizations (Training Dynamics & Dental Test Confusion Matrix)", h1_style))

REPO_ROOT = Path(__file__).resolve().parent
curve_img_path = str(REPO_ROOT / "assets" / "eval" / "training_curves.png")
cm_img_path = str(REPO_ROOT / "assets" / "eval" / "confusion_matrix.png")

images_row = []
if os.path.exists(curve_img_path) and os.path.exists(cm_img_path):
    images_row = [
        [
            RLImage(curve_img_path, width=4.3 * inch, height=1.6 * inch),
            RLImage(cm_img_path, width=3.1 * inch, height=1.6 * inch),
        ]
    ]
    img_table = Table(images_row, colWidths=[4.4 * inch, 3.1 * inch])
    img_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 6))

# 6. Latency, Hardware & Regulatory Compliance
story.append(Paragraph("6. System Specifications, Latency & Regulatory Compliance", h1_style))

spec_data = [
    [
        Paragraph("<b>Parameter Count:</b> 43,271,528 (43.05M trainable)<br/><b>Model Architecture:</b> Faster R-CNN ResNet-50 FPN V2 (COCO_V1)<br/><b>Checkpoint Size:</b> 173.5 MB (173,466,789 bytes)<br/><b>Training Epochs:</b> 12 (Best: Epoch 8, Val mAP@50=0.570, Val mAP@50:95=0.360)", body_style),
        Paragraph("<b>CPU Inference Latency:</b> 4,618.0 ms mean, 4,915.7 ms P95 (2-vCPU Xeon @ 2.00GHz)<br/><b>Memory Footprint:</b> 2,926 MB Peak RSS (on 2892x1316 OPG)<br/><b>Code License:</b> BSD-3-Clause (Torchvision / Commercial Permissive)<br/><b>Weights License:</b> CC BY-NC-SA 4.0 (DENTEX Non-Commercial Research)", body_style),
    ]
]
spec_table = Table(spec_data, colWidths=[3.75 * inch, 3.75 * inch])
spec_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
    ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(spec_table)
story.append(Spacer(1, 6))

# Regulatory Clinical Disclaimer Box
disclaimer_data = [
    [
        Paragraph(
            "<b>REGULATORY CLINICAL NOTICE:</b> This AI decision-support system is designed for retrospective and paired clinical evaluation by licensed dental professionals. It does not constitute a primary diagnostic device (FDA SaMD Tier II / MDR Class IIa considerations apply). Outputs are intended as visual assistive prompts for dentist review.",
            disclaimer_style,
        )
    ]
]
disclaimer_table = Table(disclaimer_data, colWidths=[7.5 * inch])
disclaimer_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FCA5A5")),
    ("PADDING", (0, 0), (-1, -1), 5),
]))
story.append(disclaimer_table)

# =========================================================================
# PAGE 2: BONUS 3D CBCT VOLUMETRIC EXTENSION
# =========================================================================
story.append(PageBreak())
story.append(Paragraph("7. Bonus Track: 3D CBCT Volumetric Segmentation & Nerve Clearance", h1_style))

cbct_intro = """
<b>Explicit Pretrained Model Disclosure:</b> The 3D CBCT bonus pipeline integrates <b>DentalSegmentator</b> (<code>Dataset112_DentalSegmentator_v100</code>, Dot et al., <i>Journal of Dentistry</i> 2024, Zenodo DOI <code>10.5281/zenodo.10829675</code>, License: <b>CC BY 4.0</b>, <code>nnUNetTrainer__nnUNetPlans__3d_fullres</code>, <code>fold_0</code>; the authors report training on 470 CT/CBCT scans) <b>pretrained as-is</b>, with no fine-tuning. Engineering contributions cover NIfTI / 3D DICOM series ingestion, patient tag de-identification, 60 mm³ 3D connected-component post-processing, minimum label-surface distance in physical millimetres (<code>scipy.spatial.cKDTree</code>), Marching Cubes <code>.stl</code> / <code>.glb</code> surface meshing, and browser WebGL integration (<code>&lt;model-viewer&gt;</code>).
"""
story.append(Paragraph(cbct_intro, body_style))
story.append(Spacer(1, 4))

cbct_seg_data = [
    [
        Paragraph("Class ID", table_header_style),
        Paragraph("Anatomical Structure", table_header_style),
        Paragraph("Voxels (Demo Reference Mask)", table_header_style),
        Paragraph("Volume (cm³)", table_header_style),
        Paragraph("Model Dice vs. Reference", table_header_style),
        Paragraph("Mesh Vertices", table_header_style),
        Paragraph("Mesh Faces", table_header_style),
    ],
    [Paragraph("1", table_cell_style), Paragraph("<b>maxilla_upper_skull</b>", table_cell_bold), Paragraph("1,446,339", table_cell_style), Paragraph("180.79", table_cell_style), Paragraph("<b>0.9768</b>", table_cell_bold), Paragraph("706,776", table_cell_style), Paragraph("353,388", table_cell_style)],
    [Paragraph("2", table_cell_style), Paragraph("<b>mandible</b>", table_cell_bold), Paragraph("309,895", table_cell_style), Paragraph("38.74", table_cell_style), Paragraph("<b>0.9849</b>", table_cell_bold), Paragraph("141,192", table_cell_style), Paragraph("70,596", table_cell_style)],
    [Paragraph("3", table_cell_style), Paragraph("<b>upper_teeth</b>", table_cell_bold), Paragraph("75,300", table_cell_style), Paragraph("9.41", table_cell_style), Paragraph("<b>0.9867</b>", table_cell_bold), Paragraph("153,088", table_cell_style), Paragraph("76,544", table_cell_style)],
    [Paragraph("4", table_cell_style), Paragraph("<b>lower_teeth</b>", table_cell_bold), Paragraph("63,081", table_cell_style), Paragraph("7.89", table_cell_style), Paragraph("<b>0.9745</b>", table_cell_bold), Paragraph("139,056", table_cell_style), Paragraph("69,528", table_cell_style)],
    [Paragraph("5", table_cell_style), Paragraph("<b>mandibular_canal</b>", table_cell_bold), Paragraph("6,655", table_cell_style), Paragraph("0.83", table_cell_style), Paragraph("<b>0.8968</b>", table_cell_bold), Paragraph("29,792", table_cell_style), Paragraph("14,896", table_cell_style)],
    [Paragraph("—", table_cell_style), Paragraph("<b>Total / Mean</b>", table_cell_bold), Paragraph("<b>1,901,270</b>", table_cell_bold), Paragraph("<b>237.66</b>", table_cell_bold), Paragraph("<b>0.9639</b>", table_cell_bold), Paragraph("<b>1,169,904</b>", table_cell_bold), Paragraph("<b>584,952</b>", table_cell_bold)],
]

cbct_seg_table = Table(cbct_seg_data, colWidths=[0.6 * inch, 1.5 * inch, 1.15 * inch, 0.9 * inch, 1.15 * inch, 1.1 * inch, 1.1 * inch])
cbct_seg_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 4),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
]))
story.append(cbct_seg_table)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Voxel counts, volumes and mesh counts are measured from the reference label map shipped in the "
    "<code>PostDentalSurgery</code> demo bundle. The Dice column compares a separate model prediction with "
    "that reference; the demo request itself runs no inference. Mesh counts are from the regenerated "
    "<code>.glb</code> scene.",
    small_style,
))
story.append(Spacer(1, 6))

cbct_findings_text = """
• <b>3D Finding: Minimum Label-Surface Distance (PostDentalSurgery, 360×360×330 @ 0.5 mm):</b><br/>
&nbsp;&nbsp;– <b>Side A (voxel x &lt; midpoint):</b> <b>min_distance_mm = 0.71 mm</b> between <code>closest_lower_tooth_voxel=[119, 153, 90]</code> (<code>[59.5, 76.5, 45.0] mm</code>) and <code>closest_canal_voxel=[118, 153, 89]</code> (<code>[59.0, 76.5, 44.5] mm</code>), <code>bbox_3d_voxel=[108, 143, 79, 129, 163, 100]</code>.<br/>
&nbsp;&nbsp;– <b>Side B (voxel x ≥ midpoint):</b> <b>min_distance_mm = 1.58 mm</b> between <code>closest_lower_tooth_voxel=[235, 132, 75]</code> (<code>[117.5, 66.0, 37.5] mm</code>) and <code>closest_canal_voxel=[236, 132, 72]</code> (<code>[118.0, 66.0, 36.0] mm</code>), <code>bbox_3d_voxel=[225, 122, 62, 246, 142, 85]</code>.<br/>
&nbsp;&nbsp;– <b>Caveat:</b> the side is a volume-side index, not verified patient laterality; the value is a voxel-surface distance on a 0.5 mm grid, not a nerve-centreline trace or a validated surgical risk score. No diagnostic threshold is applied.<br/>
• <b>Measured Memory &amp; Latency (Kaggle NVIDIA Tesla T4 GPU, 14.56 GB VRAM, PyTorch 2.11.0+cu128):</b><br/>
&nbsp;&nbsp;– <b>3D Sliding-Window Segmentation (252 tiles):</b> <code>131,407.1 ms</code> (<code>131.41 s</code>) | <b>Model Init:</b> <code>2,131.9 ms</code><br/>
&nbsp;&nbsp;– <b>3D Post-Processing + cKDTree Distance:</b> <code>5,816.3 ms</code> (<code>5.82 s</code>) | <b>Marching Cubes Meshing (.stl + .glb):</b> <code>938.6 ms</code> (<code>0.94 s</code>)<br/>
&nbsp;&nbsp;– <b>Peak GPU VRAM Allocated:</b> <code>4,936.0 MB</code> (<code>4.94 GB</code>) | <b>Process CPU RSS Memory:</b> <code>2,641.8 MB</code> (<code>+1,270.8 MB</code> delta)<br/>
• <b>Production Approach for Large CBCT Volumes (design proposal, not benchmarked in this submission):</b> (1) Direct presigned S3/GCS multipart upload with edge DICOM de-identification; (2) Async Redis/Celery task queue returning <code>202 Accepted</code> + WebSocket progress; (3) Autoscaled GPU workers with pre-warmed weights and serialized per-GPU jobs; TensorRT/FP16 and tile batching are possible optimizations, but no speed-up is claimed here; (4) Decimated/Draco-compressed <code>.glb</code> meshes served via CDN. The current exporter writes uncompressed GLB (28.7–35.1 MB for the two demo scans).
"""
story.append(Paragraph(cbct_findings_text, body_style))
story.append(Spacer(1, 6))

cbct_img_path = str(REPO_ROOT / "assets" / "eval" / "cbct_ortho_preview.png")
if os.path.exists(cbct_img_path):
    story.append(RLImage(cbct_img_path, width=6.6 * inch, height=2.85 * inch))

# Build PDF
doc.build(story)
print(f"Generated PDF successfully at: {PDF_OUTPUT_1}")

# Also copy to Downloads folder for immediate user access
import shutil
shutil.copy(str(PDF_OUTPUT_1), str(PDF_OUTPUT_2))
print(f"Copied to Downloads at: {PDF_OUTPUT_2}")
