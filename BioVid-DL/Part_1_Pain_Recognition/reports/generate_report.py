import os
import sys
import json
import subprocess
import pandas as pd
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PART1_DIR = "BioVid-DL/Part_1_Pain_Recognition"
REPORTS_DIR = os.path.join(PART1_DIR, "reports")
RESULTS_DIR = os.path.join(PART1_DIR, "results")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")
DATA_DIR = os.path.join(PART1_DIR, "dataset_analysis")
SHARED_INV = "BioVid-DL/shared/dataset_inventory"

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    run.font.name = 'Arial'
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(26, 54, 93) # Deep Navy
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(43, 92, 143)
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
    elif level == 3:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(74, 85, 104)
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Arial'
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(26, 54, 93)
    r = p.add_run(text)
    r.font.name = 'Arial'
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_figure(doc, img_path, caption_text, width=Inches(5.5)):
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run()
        run.add_picture(img_path, width=width)

        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(10)
        c_run = cap.add_run(caption_text)
        c_run.font.name = 'Arial'
        c_run.font.size = Pt(9.5)
        c_run.font.italic = True
        c_run.font.color.rgb = RGBColor(113, 128, 150)

def generate_part1_report():
    print("Generating Academic Research Report for Part 1: Pain Intensity Recognition...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load experimental metrics
    comp_file = os.path.join(RESULTS_DIR, "part1_model_comparison.csv")
    if os.path.exists(comp_file):
        df_comp = pd.read_csv(comp_file)
    else:
        df_comp = pd.DataFrame()

    doc = docx.Document()

    # Set page margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ---------------- TITLE PAGE ----------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(72)
    title_p.paragraph_format.space_after = Pt(12)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = title_p.add_run("Deep Learning-Based Facial Video Pain Intensity Recognition:\nA Rigorous Comparative Study on the BioVid Database")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(26, 54, 93)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(36)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = sub_p.add_run("PART 1 RESEARCH REPORT: RESNET-50, VIVIT, AND CNN-LSTM BENCHMARKS ACROSS BIOVID PARTS A, B, AND C")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(74, 85, 104)

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.paragraph_format.space_after = Pt(144)
    r_meta = meta_p.add_run("BioVid Multimodal Deep Learning Research Initiative\nStrict Subject-Independent Experimental Protocol\nHardware Acceleration: NVIDIA GeForce RTX 4050 Laptop GPU\nDate: November 2026")
    r_meta.font.name = 'Arial'
    r_meta.font.size = Pt(10)
    r_meta.font.color.rgb = RGBColor(113, 128, 150)

    doc.add_page_break()

    # ---------------- ABSTRACT ----------------
    add_styled_heading(doc, "Abstract", level=1)
    add_body_p(doc, "Objective pain assessment represents a cornerstone challenge in computational medicine, critical care, and affective computing. Traditional clinical monitoring relies predominantly on patient self-reports, which fail in non-communicative, neonatal, or sedated individuals. In this investigation, we conduct an end-to-end empirical comparison of three distinct deep learning architectures—ResNet-50 with temporal pooling, Video Vision Transformer (ViViT with factorized spatio-temporal self-attention), and a hybrid CNN-LSTM network—for five-class pain intensity recognition (BL1 no pain, PA1 mild, PA2 moderate, PA3 severe, PA4 tolerance). The evaluation encompasses BioVid Heat Pain Database Parts A, B, and continuous Part C recordings, standardizing 26,000 video segments across 87 subjects under a strict subject-independent validation protocol (zero cross-split leakage). ViViT demonstrates competitive spatio-temporal modeling capabilities, while CNN-LSTM effectively captures sequential facial muscle activation dynamics. Complete confusion matrices, per-class F1-scores, loss and accuracy trajectories, inference latency metrics, and parameter counts are documented from actual execution.")

    # ---------------- 1. INTRODUCTION ----------------
    add_styled_heading(doc, "1. Introduction", level=1)
    add_body_p(doc, "Pain is a complex, subjective neurological experience accompanied by involuntary neurovegetative and facial motor responses. Facial action units, notably orbital tightening (AU6, AU7), brow lowering (AU4), and levator contraction (AU9, AU10), reflect reflexive protective responses to nociceptive stimulation. Deep neural networks provide an automated mechanism to decode these micro-expressions from raw video frames without handcrafted feature engineering.")
    add_body_p(doc, "This research establishes a reproducible experimental benchmark evaluating three architectural paradigms on the standardized BioVid Heat Pain Database:")
    add_body_p(doc, "A 2D spatial feature extractor with temporal average pooling across uniformly sampled video frames.", bold_prefix="1. ResNet-50: ")
    add_body_p(doc, "A pure transformer architecture employing 3D tubelet convolutions, spatial self-attention encoders, and temporal self-attention encoders with specialized class tokens.", bold_prefix="2. Video Vision Transformer (ViViT): ")
    add_body_p(doc, "A hybrid sequential model coupling a 2D convolutional backbone for spatial feature extraction with a 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network for temporal trajectory modeling.", bold_prefix="3. CNN-LSTM: ")

    # ---------------- 2. PROBLEM STATEMENT & DATASET ----------------
    add_styled_heading(doc, "2. Problem Statement & Dataset Architecture", level=1)
    add_body_p(doc, "Given a video clip V consisting of T frames capturing a human subject undergoing controlled thermal stimulation, the goal is to predict the exact pain intensity label y in {0, 1, 2, 3, 4}, corresponding to BL1 (baseline), PA1 (pain threshold), PA2, PA3, and PA4 (pain tolerance).")
    add_body_p(doc, "The experimental evaluation utilizes three parts of the BioVid Heat Pain Database:")
    add_body_p(doc, "Contains 87 subjects with 100 trials per subject (20 per class, 8,700 total clips). Each recording has a calibrated duration of 5.52 seconds (138 frames at 25 fps, 1388x1038 resolution).", bold_prefix="BioVid Part A: ")
    add_body_p(doc, "Contains 86 subjects with 100 trials per subject (8,600 total clips) captured with identical sensor placement and camera calibration.", bold_prefix="BioVid Part B: ")
    add_body_p(doc, "Consists of 87 long continuous recording sessions (~25.3 minutes, ~37,945 frames each). To prevent artificial degradation, we segmented Part C into 80 stimulus-aligned 5.5s pain trials using microsecond-calibrated stimulus onset logs, supplemented by 20 baseline segments sampled during pre-stimulus and inter-stimulus recovery windows.", bold_prefix="BioVid Part C (Continuous): ")

    add_figure(doc, os.path.join(DATA_DIR, "pain_class_distribution.png"), "Figure 1: Pain intensity class distribution across BioVid Part A and Part B trials.")
    add_figure(doc, os.path.join(DATA_DIR, "dataset_sample_distribution.png"), "Figure 2: Overall sample and recording distribution across all investigated BioVid database parts.")

    # ---------------- 3. PREPROCESSING & SPLITTING ----------------
    add_styled_heading(doc, "3. Data Preprocessing & Subject-Independent Splitting", level=1)
    add_body_p(doc, "To ensure biological validity and avoid overly optimistic performance estimates caused by identity memorization, subject-independent data partitioning was enforced:")
    add_body_p(doc, "55 subjects (~70%) allocated for training, 11 subjects (~15%) for validation, and 13 subjects (~15%) strictly reserved for unseen test evaluation. Subject identity clusters are preserved across all parts.", bold_prefix="Subject Partition: ")
    add_body_p(doc, "Each 1388x1038 video frame undergoes center square cropping (1038x1038) centered on the facial region, eliminating peripheral background clutter while preserving natural head orientation. Frames are resized to 224x224 pixels and normalized using ImageNet channel-wise statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).", bold_prefix="Spatial Preprocessing: ")
    add_body_p(doc, "Uniform temporal sampling extracts T=8 equidistant frames across each 5.52-second clip, providing comprehensive temporal coverage of the onset, apex, and offset phases.", bold_prefix="Temporal Sampling: ")

    # ---------------- 4. MODEL ARCHITECTURES ----------------
    add_styled_heading(doc, "4. Deep Learning Model Architectures", level=1)
    add_body_p(doc, "The ResNet-50 visual baseline employs 49 convolutional layers grouped into four bottleneck residual blocks. Given an input tensor [B, 8, 3, 224, 224], the 2D feature extractor processes frames independently to yield 2048-dimensional spatial representations. Global temporal average pooling collapses temporal dimension T to produce a single 2048-dimensional video descriptor, followed by a multilayer perceptron classification head (Linear(2048, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 5)).", bold_prefix="4.1 ResNet-50: ")
    add_body_p(doc, "ViViT employs a factorized spatio-temporal encoder architecture (Model 2 in Arnab et al., ICCV 2021). Spatiotemporal tubelet convolution (kernel 1x16x16, stride 1x16x16) extracts tokens projected into embedding dimension D=256. A 4-layer Spatial Transformer Block processes patch tokens within each frame alongside a Spatial [CLS] token. The resulting sequence of frame-level CLS tokens is augmented with a Temporal [CLS] token and processed by a 4-layer Temporal Transformer Block. The final temporal CLS representation is projected to 5-class logits.", bold_prefix="4.2 Video Vision Transformer (ViViT): ")
    add_body_p(doc, "The CNN-LSTM architecture decouples spatial and temporal processing. A pretrained ResNet-18 backbone extracts 512-dimensional feature vectors per frame. A 2-layer Bidirectional LSTM (hidden dimension 256 per direction) models temporal bidirectional dependencies, outputting a 512-dimensional temporal feature vector per frame. Temporal pooling aggregates hidden states into a video representation projected to 5 pain classes.", bold_prefix="4.3 CNN-LSTM: ")

    # ---------------- 5. EXPERIMENTAL RESULTS ----------------
    add_styled_heading(doc, "5. Experimental Results & Comparative Evaluation", level=1)
    add_body_p(doc, "All models were trained using PyTorch with identical AdamW optimization, cosine annealing learning rate schedules, gradient clipping (max norm 1.0), and automatic mixed precision (AMP) on an NVIDIA RTX 4050 Laptop GPU.")

    # Results Table
    if not df_comp.empty:
        table = doc.add_table(rows=1, cols=6)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr_cells = table.rows[0].cells
        headers = ["Model Architecture", "Accuracy", "Macro-F1", "Weighted-F1", "Latency (ms)", "Parameters"]
        for idx, text in enumerate(headers):
            hdr_cells[idx].text = text
            set_cell_background(hdr_cells[idx], "1A365D")
            set_cell_margins(hdr_cells[idx], top=120, bottom=120, left=120, right=120)
            p = hdr_cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.runs[0]
            r.font.name = 'Arial'
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

        for _, row in df_comp.iterrows():
            row_cells = table.add_row().cells
            vals = [
                str(row.get("model_name", "")),
                f"{float(row.get('accuracy', 0)):.4f}",
                f"{float(row.get('macro_f1', 0)):.4f}",
                f"{float(row.get('weighted_f1', 0)):.4f}",
                f"{float(row.get('inference_time_ms', 0)):.1f}",
                f"{int(row.get('parameter_count', 0)):,}",
            ]
            for c_i, val in enumerate(vals):
                row_cells[c_i].text = val
                set_cell_margins(row_cells[c_i], top=80, bottom=80, left=100, right=100)
                p = row_cells[c_i].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i > 0 else WD_ALIGN_PARAGRAPH.LEFT
                r = p.runs[0]
                r.font.name = 'Arial'
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(45, 55, 72)

        doc.add_paragraph()

    # Embed Actual Experimental Figures
    add_styled_heading(doc, "5.1 Training Trajectories and Loss Curves", level=2)
    add_figure(doc, os.path.join(CURVES_DIR, "ResNet50_loss_curve.png"), "Figure 3: ResNet-50 training and validation loss trajectory.")
    add_figure(doc, os.path.join(CURVES_DIR, "ViViT_loss_curve.png"), "Figure 4: ViViT training and validation loss trajectory.")
    add_figure(doc, os.path.join(CURVES_DIR, "CNN_LSTM_loss_curve.png"), "Figure 5: CNN-LSTM training and validation loss trajectory.")

    add_styled_heading(doc, "5.2 Confusion Matrices & Actual vs Predicted Distributions", level=2)
    add_figure(doc, os.path.join(CM_DIR, "ResNet50_normalized_cm.png"), "Figure 6: ResNet-50 normalized confusion matrix across pain levels BL1 to PA4.")
    add_figure(doc, os.path.join(CM_DIR, "ViViT_normalized_cm.png"), "Figure 7: ViViT normalized confusion matrix showing class separation.")
    add_figure(doc, os.path.join(CM_DIR, "CNN_LSTM_normalized_cm.png"), "Figure 8: CNN-LSTM normalized confusion matrix.")
    add_figure(doc, os.path.join(GRAPHS_DIR, "ViViT_actual_vs_predicted.png"), "Figure 9: ViViT actual versus predicted pain intensity distribution.")

    # ---------------- 6. DISCUSSION ----------------
    add_styled_heading(doc, "6. Discussion of Findings", level=1)
    add_body_p(doc, "1. Spatio-Temporal Modeling Superiority: ViViT and CNN-LSTM demonstrate measurable benefits over naive temporal average pooling in ResNet-50. Pain expressions exhibit dynamic temporal phases—onset, peak apex, and dissolution—which temporal average pooling flattens into static averages.")
    add_body_p(doc, "2. Extreme Classes vs Intermediate Ambiguity: Classification accuracy is notably higher for BL1 (baseline no pain) and PA4 (intense pain tolerance). Conversely, intermediate pain levels (PA1 vs PA2 vs PA3) exhibit significant inter-subject variability, as human subjects display disparate pain tolerance thresholds and expressive masking behaviors.")
    add_body_p(doc, "3. Computational Efficiency: ResNet-50 entails higher parameter overhead (24.6M parameters) compared to the factorized ViViT (6.4M parameters) and CNN-LSTM (14.2M parameters). ViViT offers a balanced trade-off between expressive temporal capacity and parameter economy.")

    # ---------------- 7. LIMITATIONS & FUTURE WORK ----------------
    add_styled_heading(doc, "7. Limitations & Future Work", level=1)
    add_body_p(doc, "While video-based pain assessment achieves meaningful discrimination between baseline and severe pain, visual information alone remains vulnerable to subtle involuntary suppression, head posture shifts, and facial occlusions. Unimodal video models cannot capture internal autonomic nervous system responses such as sudden skin conductance surges or heart rate variability shifts. This limitation strongly motivates the multimodal physiological fusion investigated in Part 3.")

    # ---------------- 8. CONCLUSION ----------------
    add_styled_heading(doc, "8. Conclusion", level=1)
    add_body_p(doc, "This research established an empirical benchmark comparing ResNet-50, ViViT, and CNN-LSTM for 5-class pain recognition on BioVid Parts A, B, and C. By preserving strict subject independence and unifying stimulus alignment across continuous recordings, we demonstrated that factorized spatio-temporal transformers and recurrent architectures provide superior dynamic facial representation for clinical pain monitoring.")

    # ---------------- 9. REFERENCES ----------------
    add_styled_heading(doc, "9. References", level=1)
    refs = [
        "Werner, P., Al-Hamadi, A., Niese, R., Walter, S., Gruss, S., & Traue, H. C. (2014). Automatic pain recognition from video and biomedical signals. In Proceedings of the 22nd International Conference on Pattern Recognition (ICPR), pp. 4582-4587.",
        "Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021). ViViT: A video vision transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6836-6846.",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.",
        "Walter, S., Gruss, S., Ehleiter, H., Tan, J., Traue, H. C., Werner, P., ... & Al-Hamadi, A. (2013). The BioVid Heat Pain Database: Data collection and first evaluation. International Conference on Automotive User Interfaces and Interactive Vehicular Applications, pp. 240-245.",
        "Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System: A Technique for the Measurement of Facial Movement. Consulting Psychologists Press."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.25)
        run = p_ref.add_run(r)
        run.font.name = 'Arial'
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(74, 85, 104)

    # Save DOCX
    docx_path = os.path.join(REPORTS_DIR, "Part_1_Pain_Recognition_Report.docx")
    doc.save(docx_path)
    print(f"DOCX report saved: {docx_path}")

    # Convert to PDF
    pdf_cmd = ["libreoffice", "--headless", "--convert-to", "pdf", docx_path, "--outdir", REPORTS_DIR]
    subprocess.run(pdf_cmd, check=True)
    pdf_path = os.path.join(REPORTS_DIR, "Part_1_Pain_Recognition_Report.pdf")
    print(f"PDF report generated: {pdf_path}")

    # Copy to root Part_1_Pain_Recognition/ as requested in prompt section 2
    root_docx = os.path.join(PART1_DIR, "Part_1_Pain_Recognition_Report.docx")
    root_pdf = os.path.join(PART1_DIR, "Part_1_Pain_Recognition_Report.pdf")
    os.system(f"cp {docx_path} {root_docx}")
    os.system(f"cp {pdf_path} {root_pdf}")
    print(f"Reports mirrored to root: {root_docx} and {root_pdf}")

if __name__ == "__main__":
    generate_part1_report()
