"""
Generate publication-quality 3-Page PDF report answering:
1. Dataset usage (DENTEX vs Zhou vs Tufts)
2. Partition percentages (Train: 84.11%, Val: 9.27%, Test: 6.62%)
3. Precise quantitative evaluation metrics, threshold optimization, confusion matrix, and ablations.
"""

import os
import shutil
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
from reportlab.pdfgen import canvas

REPO_ROOT = Path(__file__).resolve().parent
PDF_OUTPUT_1 = REPO_ROOT / "Viscan_Dataset_Usage_and_Evaluation_Report.pdf"
PDF_OUTPUT_2 = Path(r"C:\Users\Errachandhanam\Downloads\Viscan_Dataset_Usage_and_Evaluation_Report.pdf")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(32, 762, "Viscan Dental OPG AI — Dataset Usage, Partitioning & Quantitative Evaluation Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(32, 756, 580, 756)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(32, 28, 580, 28)
        self.drawString(32, 18, "CONFIDENTIAL & PROPRIETARY — TAKE-HOME ASSESSMENT AUDIT — VISCAN DENTAL AI")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(580, 18, page_str)
        self.restoreState()


doc = SimpleDocTemplate(
    str(PDF_OUTPUT_1),
    pagesize=letter,
    leftMargin=0.45 * inch,
    rightMargin=0.45 * inch,
    topMargin=0.38 * inch,
    bottomMargin=0.42 * inch,
)

styles = getSampleStyleSheet()

# Colors
PRIMARY = colors.HexColor("#1E3A8A")     # Navy Blue
SECONDARY = colors.HexColor("#0284C7")   # Azure / Medical Blue
ACCENT = colors.HexColor("#0D9488")      # Teal
DARK_TEXT = colors.HexColor("#0F172A")   # Slate 900
MUTED_TEXT = colors.HexColor("#475569")  # Slate 600
BG_LIGHT = colors.HexColor("#F8FAFC")    # Slate 50
BG_ALT = colors.HexColor("#F1F5F9")      # Slate 100
BORDER_COLOR = colors.HexColor("#CBD5E1")  # Slate 300
HIGHLIGHT_GREEN = colors.HexColor("#ECFDF5")
HIGHLIGHT_RED = colors.HexColor("#FEF2F2")

# Styles
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=19,
    textColor=PRIMARY,
    spaceAfter=2,
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=9.5,
    leading=13,
    textColor=SECONDARY,
    spaceAfter=6,
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10.5,
    leading=13.5,
    textColor=PRIMARY,
    spaceBefore=5,
    spaceAfter=3,
)

body_style = ParagraphStyle(
    "DocBody",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=11,
    textColor=DARK_TEXT,
    spaceAfter=3,
)

table_header_style = ParagraphStyle(
    "TableHeader",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7.5,
    leading=9.5,
    textColor=colors.white,
    alignment=1,
)

table_cell_style = ParagraphStyle(
    "TableCell",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.2,
    leading=9.5,
    textColor=DARK_TEXT,
)

table_cell_center = ParagraphStyle(
    "TableCellCenter",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.2,
    leading=9.5,
    textColor=DARK_TEXT,
    alignment=1,
)

table_cell_bold = ParagraphStyle(
    "TableCellBold",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7.2,
    leading=9.5,
    textColor=DARK_TEXT,
)

table_cell_bold_center = ParagraphStyle(
    "TableCellBoldCenter",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7.2,
    leading=9.5,
    textColor=DARK_TEXT,
    alignment=1,
)

badge_selected = ParagraphStyle(
    "BadgeSelected",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7.2,
    leading=9.5,
    textColor=colors.HexColor("#065F46"),
    alignment=1,
)

badge_excluded = ParagraphStyle(
    "BadgeExcluded",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7.2,
    leading=9.5,
    textColor=colors.HexColor("#991B1B"),
    alignment=1,
)

callout_style = ParagraphStyle(
    "CalloutText",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.6,
    leading=10.8,
    textColor=colors.HexColor("#1E293B"),
)

story = []

# =========================================================================
# PAGE 1: EXECUTIVE RESPONSE & DATASET SELECTION / PARTITIONING
# =========================================================================

story.append(Paragraph("Viscan Dental OPG AI Assistant — Technical Evaluation Audit", title_style))
story.append(Paragraph("Dataset Sourcing Analysis, Partition Percentages & Precise Quantitative Evaluation", subtitle_style))

# Executive Summary Banner Box
summary_box_data = [
    [
        Paragraph(
            "<b>EXECUTIVE AUDIT SUMMARY:</b><br/>"
            "• <b>Did we use all datasets mentioned in the assessment?</b> <b>NO.</b> Out of three candidate public datasets reviewed (DENTEX, Zhou et al., Tufts), only <b>DENTEX (MICCAI 2023)</b> was selected. Zhou et al. and Tufts were systematically excluded because their annotations lack clinical separation of the 4 required disease classes.<br/>"
            "• <b>Exact Dataset Split Percentages:</b> From the 755-image labeled cohort, <b>84.11% (635 images, 3,190 boxes)</b> was used for Training, <b>9.27% (70 images, 339 boxes)</b> was carved out for Validation (checkpoint selection & threshold tuning), and <b>6.62% (50 images, 182 boxes)</b> was strictly held out as the Test Set (evaluated only once). Zero filename overlap confirmed across splits.<br/>"
            "• <b>Evaluation Results:</b> The Faster R-CNN ResNet-50 FPN V2 achieved <b>mAP@50 = 49.6%</b>, <b>mAP@50:95 = 32.9%</b>, and <b>mAP@75 = 38.3%</b> on held-out test data. Operating at validation-tuned threshold <b>0.70</b> (Val Macro-F1 = 0.556, Test Macro-F1 = 0.402), <code>impacted_tooth</code> achieved <b>76.1% precision / 87.5% recall (F1=0.814)</b>, while <code>caries</code> captured <b>53.5% (54/101) of test decay lesions</b>.",
            callout_style,
        )
    ]
]
summary_table = Table(summary_box_data, colWidths=[7.6 * inch])
summary_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3B82F6")),
    ("PADDING", (0, 0), (-1, -1), 4.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(summary_table)
story.append(Spacer(1, 4))

# SECTION 1
story.append(Paragraph("1. Assessment Dataset Review & Justification (Why We Did Not Use All Datasets)", h1_style))
story.append(Paragraph(
    "The assessment specification (R06) required reviewing and comparing public expert-annotated dental OPG datasets to select the optimal training foundation. Three primary datasets were evaluated:",
    body_style
))

dataset_review_data = [
    [
        Paragraph("Dataset Name", table_header_style),
        Paragraph("Modality / Size", table_header_style),
        Paragraph("Annotation Taxonomy", table_header_style),
        Paragraph("License", table_header_style),
        Paragraph("Status", table_header_style),
        Paragraph("Engineering Rationale & R03 Compatibility", table_header_style),
    ],
    [
        Paragraph("<b>DENTEX<br/>(MICCAI 2023)</b>", table_cell_bold),
        Paragraph("Panoramic OPG<br/>1,005 images total<br/>(755 labeled cohort)", table_cell_style),
        Paragraph("Hierarchical triple annotations:<br/>• Quadrants (1–4)<br/>• FDI tooth enumeration (11–48)<br/>• <b>4 Clinical Findings:</b> Caries, Deep Caries, Periapical Lesion, Impacted Tooth", table_cell_style),
        Paragraph("CC BY-NC-SA 4.0", table_cell_center),
        Paragraph("<b>SELECTED</b>", badge_selected),
        Paragraph("<b>Directly fulfills Requirement R03.</b> Provides multi-institutional expert ground truth verified by senior dentists (>15 yrs exp) for all four target pathology classes. High-quality bounding boxes allow native two-stage object detection.", table_cell_style),
    ],
    [
        Paragraph("<b>Zhou et al.<br/>(2024)</b>", table_cell_bold),
        Paragraph("Panoramic OPG<br/>2,066 images", table_cell_style),
        Paragraph("Instance segmentation polygons with 7 tooth-state classes (tooth without anomalies, fillings, root canal treatment (RCT), crown, caries, residual root, RCT + crown).", table_cell_style),
        Paragraph("CC BY 4.0", table_cell_center),
        Paragraph("<b>EXCLUDED</b>", badge_excluded),
        Paragraph("<b>Incompatible taxonomy.</b> Labels focus on restorative status and general tooth state. Does not distinguish superficial caries from deep caries and lacks impacted tooth and periapical lesion classes.", table_cell_style),
    ],
    [
        Paragraph("<b>Tufts Dental<br/>Database</b>", table_cell_bold),
        Paragraph("Panoramic OPG<br/>1,000 images", table_cell_style),
        Paragraph("Tooth masks and broad abnormality regions (radiolucency, radiopacity, eye-tracking maps).", table_cell_style),
        Paragraph("Restricted Institutional", table_cell_center),
        Paragraph("<b>EXCLUDED</b>", badge_excluded),
        Paragraph("<b>Restricted access & unaligned taxonomy.</b> Access requires institutional agreement. Abnormality categories are generic radiographic descriptors that cannot be deterministically mapped to the four required clinical entities.", table_cell_style),
    ],
]

dataset_table = Table(dataset_review_data, colWidths=[1.1 * inch, 0.95 * inch, 1.8 * inch, 0.75 * inch, 0.75 * inch, 2.25 * inch])
dataset_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 2.8),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("BACKGROUND", (0, 1), (-1, 1), HIGHLIGHT_GREEN),
    ("BACKGROUND", (0, 2), (-1, 2), colors.white),
    ("BACKGROUND", (0, 3), (-1, 3), BG_ALT),
]))
story.append(dataset_table)
story.append(Spacer(1, 4))

# SECTION 2
story.append(Paragraph("2. Dataset Partitioning, Usage Percentages & Leakage Prevention", h1_style))
story.append(Paragraph(
    "To guarantee methodological integrity (R07, R11), the labeled DENTEX corpus was partitioned into a 3-way split using fixed random seed 42. "
    "The official 50-image validation archive was designated as the <b>untouched held-out test set</b> and evaluated once. "
    "The 705-image training archive was split into 635 training images and 70 validation images for epoch checkpoint selection.",
    body_style
))

split_data = [
    [
        Paragraph("Partition", table_header_style),
        Paragraph("Image Count", table_header_style),
        Paragraph("% of Labeled Cohort (755)", table_header_style),
        Paragraph("% of Train Archive (705)", table_header_style),
        Paragraph("Total Finding Boxes", table_header_style),
        Paragraph("Role & Evaluation Discipline", table_header_style),
    ],
    [
        Paragraph("<b>Training Set</b>", table_cell_bold),
        Paragraph("635", table_cell_bold_center),
        Paragraph("<b>84.11%</b>", table_cell_bold_center),
        Paragraph("<b>90.07%</b>", table_cell_center),
        Paragraph("3,190", table_cell_center),
        Paragraph("Model parameter updates, loss backpropagation, and multi-scale data augmentation (horizontal flip, scale jitter).", table_cell_style),
    ],
    [
        Paragraph("<b>Validation Set</b>", table_cell_bold),
        Paragraph("70", table_cell_bold_center),
        Paragraph("<b>9.27%</b>", table_cell_bold_center),
        Paragraph("<b>9.93%</b>", table_cell_center),
        Paragraph("339", table_cell_center),
        Paragraph("Checkpoint selection (Epoch 8 best Val mAP@50=0.570, mAP@50:95=0.360) and threshold sweep ([0.10, 0.90]).", table_cell_style),
    ],
    [
        Paragraph("<b>Held-Out Test Set</b>", table_cell_bold),
        Paragraph("50", table_cell_bold_center),
        Paragraph("<b>6.62%</b>", table_cell_bold_center),
        Paragraph("— (Official Val)", table_cell_center),
        Paragraph("182", table_cell_center),
        Paragraph("<b>Strictly held-out.</b> Evaluated once after freezing model weights and operating threshold. Zero tuning leakage.", table_cell_style),
    ],
    [
        Paragraph("<b>Total Active Cohort</b>", table_cell_bold),
        Paragraph("<b>755</b>", table_cell_bold_center),
        Paragraph("<b>100.0%</b>", table_cell_bold_center),
        Paragraph("—", table_cell_center),
        Paragraph("<b>3,711</b>", table_cell_bold_center),
        Paragraph("Complete expert-annotated 2D OPG cohort used in the project.", table_cell_style),
    ],
]

split_table = Table(split_data, colWidths=[1.2 * inch, 0.85 * inch, 1.4 * inch, 1.25 * inch, 1.1 * inch, 1.8 * inch])
split_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 2.8),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("BACKGROUND", (0, 1), (-1, 1), colors.white),
    ("BACKGROUND", (0, 2), (-1, 2), BG_ALT),
    ("BACKGROUND", (0, 3), (-1, 3), HIGHLIGHT_GREEN),
    ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#E2E8F0")),
]))
story.append(split_table)
story.append(Spacer(1, 3))

leakage_text = """
<b>Split Disjointness Verification:</b> Zero filename intersection was programmatically asserted across all three splits (<code>len(train_files intersect val_files) == 0</code>, <code>len(train_files intersect test_files) == 0</code>, <code>len(val_files intersect test_files) == 0</code>). 
Note: Public DENTEX releases strip DICOM patient headers prior to publication (`train_0.png`..`train_704.png`, `val_0.png`..`val_49.png`), so patient-level deduplication cannot be verified beyond image-level disjointness.
"""
story.append(Paragraph(leakage_text, body_style))

# PAGE BREAK TO PAGE 2
story.append(PageBreak())

# =========================================================================
# PAGE 2: PRECISE EVALUATION METHODOLOGY & BENCHMARK METRICS
# =========================================================================

story.append(Paragraph("3. Quantitative Output Evaluation Methodology & Primary Metrics", h1_style))
story.append(Paragraph(
    "Model performance was evaluated on the held-out 50-image test set (182 true findings) using official COCO evaluation standards "
    "(IoU threshold range 0.50 to 0.95 with step 0.05). Below are the measured metrics:",
    body_style
))

# COCO primary table
coco_kpi_data = [
    [
        Paragraph("Evaluation Metric", table_header_style),
        Paragraph("Measured Value", table_header_style),
        Paragraph("Standard / Target", table_header_style),
        Paragraph("Clinical & Technical Interpretation", table_header_style),
    ],
    [
        Paragraph("<b>mAP @ IoU=0.50</b>", table_cell_bold),
        Paragraph("<b>49.6% (0.496)</b>", table_cell_bold_center),
        Paragraph("Diagnostic Screening Standard", table_cell_center),
        Paragraph("Area under the precision-recall curve averaged across all 4 classes at 50% spatial overlap (IoU >= 0.50) against expert bounding boxes.", table_cell_style),
    ],
    [
        Paragraph("<b>mAP @ IoU=0.50:0.95</b> (Primary)", table_cell_bold),
        Paragraph("<b>32.9% (0.329)</b>", table_cell_bold_center),
        Paragraph("Official COCO Primary Metric", table_cell_center),
        Paragraph("Average precision integrated across 10 IoU thresholds (0.50 to 0.95). Stringent measure penalizing loose boundaries on small dental findings.", table_cell_style),
    ],
    [
        Paragraph("<b>mAP @ IoU=0.75</b> (Strict)", table_cell_bold),
        Paragraph("<b>38.3% (0.383)</b>", table_cell_bold_center),
        Paragraph("High-Precision Boundary Metric", table_cell_center),
        Paragraph("Evaluates localization tightness under strict 75% IoU overlap requirements.", table_cell_style),
    ],
    [
        Paragraph("<b>Average Recall (AR @ 100)</b>", table_cell_bold),
        Paragraph("<b>68.3% (0.683)</b>", table_cell_bold_center),
        Paragraph("Top-100 Proposal Sensitivity", table_cell_center),
        Paragraph("Upper-bound recall across up to 100 candidate detections per OPG (before confidence thresholding). At operating threshold 0.70, 96/182 (52.7%) of GT boxes are matched.", table_cell_style),
    ],
]

coco_kpi_table = Table(coco_kpi_data, colWidths=[1.8 * inch, 1.2 * inch, 1.4 * inch, 3.2 * inch])
coco_kpi_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 3.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
]))
story.append(coco_kpi_table)
story.append(Spacer(1, 5))

story.append(Paragraph("4. Per-Class Benchmark Performance & Operating Metrics (@ Threshold = 0.70)", h1_style))
story.append(Paragraph(
    "To select a deployment threshold without test leakage, a threshold sweep (0.10 to 0.90) was performed on the 70-image validation set. "
    "A confidence threshold of <b>0.70</b> was selected to maximize Validation Macro-F1 (<b>0.556</b>), achieving <b>0.402</b> Macro-F1 on the held-out test set:",
    body_style
))

per_class_data = [
    [
        Paragraph("Pathology Class", table_header_style),
        Paragraph("COCO AP50", table_header_style),
        Paragraph("COCO AP50:95", table_header_style),
        Paragraph("Precision", table_header_style),
        Paragraph("Recall", table_header_style),
        Paragraph("F1-Score", table_header_style),
        Paragraph("Test Support (n)", table_header_style),
        Paragraph("Observed Clinical Behavior & Error Profile", table_header_style),
    ],
    [
        Paragraph("<b>Impacted Tooth</b>", table_cell_bold),
        Paragraph("83.3%", table_cell_center),
        Paragraph("54.1%", table_cell_center),
        Paragraph("<b>76.1%</b>", table_cell_bold_center),
        Paragraph("<b>87.5%</b>", table_cell_bold_center),
        Paragraph("<b>0.814</b>", table_cell_bold_center),
        Paragraph("40", table_cell_center),
        Paragraph("<b>Strongest Class:</b> Distinct morphology of impacted third molars yields zero cross-confusion with decay (35/40 recalled, 10 background FPs).", table_cell_style),
    ],
    [
        Paragraph("<b>Caries</b> (Superficial)", table_cell_bold),
        Paragraph("35.8%", table_cell_center),
        Paragraph("24.4%", table_cell_center),
        Paragraph("<b>27.7%</b>", table_cell_bold_center),
        Paragraph("<b>53.5%</b>", table_cell_bold_center),
        Paragraph("<b>0.365</b>", table_cell_bold_center),
        Paragraph("101", table_cell_center),
        Paragraph("<b>High False-Alarm Rate:</b> Recalls 54/101 lesions (45 missed, 2 misclassified), with 124 background false positives on interproximal shading and restorations.", table_cell_style),
    ],
    [
        Paragraph("<b>Deep Caries</b>", table_cell_bold),
        Paragraph("45.7%", table_cell_center),
        Paragraph("30.2%", table_cell_center),
        Paragraph("<b>42.9%</b>", table_cell_bold_center),
        Paragraph("<b>18.8%</b>", table_cell_bold_center),
        Paragraph("<b>0.261</b>", table_cell_bold_center),
        Paragraph("32", table_cell_center),
        Paragraph("<b>Adjacent Severity Confusion:</b> 6/32 recalled as deep caries, while 14/32 (43.8%) are detected at IoU>=0.50 but labeled as superficial <code>caries</code>.", table_cell_style),
    ],
    [
        Paragraph("<b>Periapical Lesion</b>", table_cell_bold),
        Paragraph("33.6%", table_cell_center),
        Paragraph("23.0%", table_cell_center),
        Paragraph("<b>33.3%</b>", table_cell_bold_center),
        Paragraph("<b>11.1%</b>", table_cell_bold_center),
        Paragraph("<b>0.167</b>", table_cell_bold_center),
        Paragraph("9", table_cell_center),
        Paragraph("<b>Severe Minority Class:</b> Only 9 true lesions in test set (144 in train; 14 in val). 1 recalled, 4 missed, 4 misclassified as caries/deep caries.", table_cell_style),
    ],
]

per_class_table = Table(
    per_class_data,
    colWidths=[1.1 * inch, 0.7 * inch, 0.8 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 2.4 * inch],
)
per_class_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 3),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
]))
story.append(per_class_table)
story.append(Spacer(1, 5))

story.append(Paragraph("5. Confusion Matrix & Clinical Failure Mode Analysis", h1_style))
story.append(Paragraph(
    "A 5x5 detection confusion matrix was computed on the 50-image test set at threshold 0.70 and IoU >= 0.50:",
    body_style
))

cm_matrix_data = [
    [
        Paragraph("Ground Truth \\ Predicted", table_header_style),
        Paragraph("Background (FN)", table_header_style),
        Paragraph("Impacted Tooth", table_header_style),
        Paragraph("Caries", table_header_style),
        Paragraph("Periapical", table_header_style),
        Paragraph("Deep Caries", table_header_style),
    ],
    [Paragraph("<b>Background (False Positives)</b>", table_cell_bold), Paragraph("—", table_cell_center), Paragraph("10", table_cell_center), Paragraph("<b>124</b>", table_cell_bold_center), Paragraph("2", table_cell_center), Paragraph("6", table_cell_center)],
    [Paragraph("<b>Impacted Tooth</b>", table_cell_bold), Paragraph("5", table_cell_center), Paragraph("<b>35 (87.5%)</b>", table_cell_bold_center), Paragraph("0", table_cell_center), Paragraph("0", table_cell_center), Paragraph("0", table_cell_center)],
    [Paragraph("<b>Caries</b>", table_cell_bold), Paragraph("45", table_cell_center), Paragraph("1", table_cell_center), Paragraph("<b>54 (53.5%)</b>", table_cell_bold_center), Paragraph("0", table_cell_center), Paragraph("1", table_cell_center)],
    [Paragraph("<b>Periapical Lesion</b>", table_cell_bold), Paragraph("4", table_cell_center), Paragraph("0", table_cell_center), Paragraph("3", table_cell_center), Paragraph("<b>1 (11.1%)</b>", table_cell_bold_center), Paragraph("1", table_cell_center)],
    [Paragraph("<b>Deep Caries</b>", table_cell_bold), Paragraph("12", table_cell_center), Paragraph("0", table_cell_center), Paragraph("<b>14 (43.8%)</b>", table_cell_bold_center), Paragraph("0", table_cell_center), Paragraph("<b>6 (18.8%)</b>", table_cell_bold_center)],
]

cm_table = Table(cm_matrix_data, colWidths=[2.3 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch, 1.1 * inch])
cm_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 3),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
]))
story.append(cm_table)
story.append(Spacer(1, 3))

cm_takeaways = """
<b>Key Error Findings:</b><br/>
1. <b>Zero Impacted-to-Decay Confusion:</b> 0 ground-truth impacted teeth were misclassified as decay or periapical lesions (35 TP, 5 FN).<br/>
2. <b>Severity Grading Ambiguity:</b> 14 out of 20 inter-class misclassifications occurred when <code>deep_caries</code> was predicted as <code>caries</code>.<br/>
3. <b>False Positive Concentration:</b> 124 of 142 false positives belong to <code>caries</code> on low-contrast interproximal enamel and cervical burnout regions.
"""
story.append(Paragraph(cm_takeaways, body_style))

# PAGE BREAK TO PAGE 3
story.append(PageBreak())

# =========================================================================
# PAGE 3: CONTROLLED EXPERIMENTATION, BENCHMARKS & VERIFIED DELIVERABLES
# =========================================================================

story.append(Paragraph("6. Architecture Resolution Experiment & Engineering Trade-Offs", h1_style))
story.append(Paragraph(
    "Per Requirement R14, a trained model comparison experiment evaluated a higher-resolution feature pyramid variant "
    "(min_size=960, max_size=1600, batch_size=1) against the baseline architecture (min_size=800, max_size=1333, batch_size=2):",
    body_style
))

ablation_data = [
    [
        Paragraph("Model Configuration", table_header_style),
        Paragraph("Input Scale / Batch", table_header_style),
        Paragraph("Effective Scale on OPG", table_header_style),
        Paragraph("mAP@50", table_header_style),
        Paragraph("mAP@50:95", table_header_style),
        Paragraph("mAP@75 (Strict)", table_header_style),
        Paragraph("Engineering Outcome & Rationale", table_header_style),
    ],
    [
        Paragraph("<b>Baseline Architecture</b><br/>(ResNet-50 FPN V2)", table_cell_bold),
        Paragraph("800 × 1333<br/>(batch_size=2)", table_cell_center),
        Paragraph("1333 × 607<br/>(Long-side cap binds)", table_cell_center),
        Paragraph("<b>49.6%</b>", table_cell_bold_center),
        Paragraph("<b>32.9%</b>", table_cell_bold_center),
        Paragraph("38.3%", table_cell_center),
        Paragraph("<b>Selected as Final Model</b> via primary Val mAP@50:95 (0.360 vs 0.354). Higher test mAP@50 (+0.3%) and mAP@50:95 (+1.6%), with 45.7% Deep Caries AP50.", table_cell_style),
    ],
    [
        Paragraph("<b>High-Resolution Variant</b><br/>(ResNet-50 FPN V2)", table_cell_bold),
        Paragraph("960 × 1600<br/>(batch_size=1)", table_cell_center),
        Paragraph("1600 × 728<br/>(1.2× resolution)", table_cell_center),
        Paragraph("49.3% (-0.3%)", table_cell_center),
        Paragraph("31.3% (-1.6%)", table_cell_center),
        Paragraph("<b>41.0% (+2.7%)</b>", table_cell_bold_center),
        Paragraph("Improved strict mAP@75 (+2.7%) and Impacted Tooth AP50 (91.5%), but batch_size=1 without LR rescaling degraded Deep Caries AP50 (36.5%, -9.2%). Two-variable confound noted.", table_cell_style),
    ],
]

ablation_table = Table(ablation_data, colWidths=[1.4 * inch, 1.0 * inch, 1.1 * inch, 0.7 * inch, 0.75 * inch, 0.85 * inch, 1.8 * inch])
ablation_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 3.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("BACKGROUND", (0, 1), (-1, 1), HIGHLIGHT_GREEN),
    ("BACKGROUND", (0, 2), (-1, 2), colors.white),
]))
story.append(ablation_table)
story.append(Spacer(1, 5))

# SECTION 7: PLOTS
story.append(Paragraph("7. Empirical Evidence Visualizations (Training Curves & Dental Confusion Matrix)", h1_style))

curve_img_path = str(REPO_ROOT / "assets" / "eval" / "training_curves.png")
cm_img_path = str(REPO_ROOT / "assets" / "eval" / "confusion_matrix.png")

if os.path.exists(curve_img_path) and os.path.exists(cm_img_path):
    images_row = [
        [
            RLImage(curve_img_path, width=4.35 * inch, height=1.65 * inch),
            RLImage(cm_img_path, width=3.15 * inch, height=1.65 * inch),
        ]
    ]
    img_table = Table(images_row, colWidths=[4.4 * inch, 3.2 * inch])
    img_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 5))

# SECTION 8: HARDWARE & SYSTEM BENCHMARK
story.append(Paragraph("8. Hardware Benchmarks, Licensing & Live Deliverables", h1_style))

spec_data = [
    [
        Paragraph("<b>Hardware & CPU Latency Benchmark</b><br/>"
                  "• <b>Platform:</b> Intel Xeon @ 2.00GHz (2 vCPUs, 2 threads)<br/>"
                  "• <b>Test Image:</b> Real OPG <code>val_15.png</code> (2892 × 1316)<br/>"
                  "• <b>Mean Latency:</b> 4,618.0 ms (Throughput: 0.22 img/sec)<br/>"
                  "• <b>P95 Latency:</b> 4,915.7 ms (Max: 5,037.2 ms)<br/>"
                  "• <b>Memory Footprint:</b> 2,926 MB Peak RSS", body_style),
        Paragraph("<b>Licensing & Open Source Governance</b><br/>"
                  "• <b>Architecture:</b> Faster R-CNN ResNet-50 FPN V2<br/>"
                  "• <b>Code License:</b> <b>BSD-3-Clause</b> (Permissive, commercial ready)<br/>"
                  "• <b>Pretrained Source:</b> PyTorch Torchvision COCO_V1<br/>"
                  "• <b>Weights License:</b> CC BY-NC-SA 4.0 (DENTEX)<br/>"
                  "• <b>Production Note:</b> Commercial deployment requires retraining on commercial data", body_style),
    ]
]
spec_table = Table(spec_data, colWidths=[3.8 * inch, 3.8 * inch])
spec_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
    ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
    ("PADDING", (0, 0), (-1, -1), 4.5),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(spec_table)
story.append(Spacer(1, 5))

# Verified Deliverable URLs Table
urls_data = [
    [
        Paragraph("<b>Live Cloud Demonstration:</b>", table_cell_bold),
        Paragraph("<b>https://huggingface.co/spaces/surendra08/viscan-opg</b> (Status: Active / 200 OK)", table_cell_style),
    ],
    [
        Paragraph("<b>Weights LFS Repository:</b>", table_cell_bold),
        Paragraph("<b>https://huggingface.co/surendra08/viscan-opg-weights</b> (SHA-256: <code>e0b49b12f48f...</code>)", table_cell_style),
    ],
    [
        Paragraph("<b>Local Standalone Service:</b>", table_cell_bold),
        Paragraph("FastAPI backend (<code>POST /analyze</code>) with side-by-side web UI & Docker support (port 8000)", table_cell_style),
    ],
]
urls_table = Table(urls_data, colWidths=[2.2 * inch, 5.4 * inch])
urls_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#86EFAC")),
    ("PADDING", (0, 0), (-1, -1), 3.5),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(urls_table)
story.append(Spacer(1, 5))

# Regulatory Clinical Notice
disclaimer_data = [
    [
        Paragraph(
            "<b>REGULATORY CLINICAL NOTICE:</b> This software is a clinical decision-support research prototype developed for "
            "paired evaluation alongside licensed dental practitioners. It is not an FDA-cleared or CE-marked medical device. "
            "All visual annotations, bounding boxes, and clinical summaries must be verified by a qualified dentist.",
            ParagraphStyle("DisclStyle", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.2, leading=10, textColor=colors.HexColor("#991B1B")),
        )
    ]
]
disclaimer_table = Table(disclaimer_data, colWidths=[7.6 * inch])
disclaimer_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), HIGHLIGHT_RED),
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FCA5A5")),
    ("PADDING", (0, 0), (-1, -1), 3.5),
]))
story.append(disclaimer_table)

# Build Document with NumberedCanvas
doc.build(story, canvasmaker=NumberedCanvas)
print(f"Generated PDF successfully at: {PDF_OUTPUT_1}")

# Copy to Downloads if accessible
try:
    shutil.copy(str(PDF_OUTPUT_1), str(PDF_OUTPUT_2))
    print(f"Copied to Downloads at: {PDF_OUTPUT_2}")
except Exception as exc:
    print(f"Note: Could not copy to Downloads: {exc}")
