# BioVid Multimodal Deep Learning for Pain Intensity and Emotion Recognition

An end-to-end research deep learning system implementing, training, and benchmarking **six architectures** across **all five parts (A–E)** of the **BioVid Heat Pain Database**.

---

## 1. Project Experimental Architecture

The project investigates three independent research parts under a **strict subject-independent evaluation protocol** guaranteeing zero cross-split leakage:

```
BioVid-DL/
├── Part_1_Pain_Recognition/
│   ├── dataset_analysis/
│   ├── preprocessing/
│   ├── models/ (resnet50, vivit, cnn_lstm)
│   ├── results/ (graphs, confusion_matrices, metrics, predictions, training_curves)
│   ├── Part_1_Pain_Recognition_Report.docx
│   ├── Part_1_Pain_Recognition_Report.pdf
│   └── reports/ (Part_1_Pain_Recognition_Report.docx, .pdf)
├── Part_2_Emotion_Recognition/
│   ├── dataset_analysis/
│   ├── preprocessing/
│   ├── models/ (3d_cnn, video_swin_transformer)
│   ├── results/ (graphs, confusion_matrices, metrics, predictions, training_curves)
│   ├── Part_2_Emotion_Recognition_Report.docx
│   ├── Part_2_Emotion_Recognition_Report.pdf
│   └── reports/ (Part_2_Emotion_Recognition_Report.docx, .pdf)
├── Part_3_Multimodal_Recognition/
│   ├── dataset_analysis/
│   ├── preprocessing/
│   ├── models/ (multimodal_cnn)
│   ├── results/ (graphs, signal_analysis, confusion_matrices, metrics, predictions, ablation_studies)
│   ├── Part_3_Multimodal_Recognition_Report.docx
│   ├── Part_3_Multimodal_Recognition_Report.pdf
│   └── reports/ (Part_3_Multimodal_Recognition_Report.docx, .pdf)
├── Final_Comparative_Analysis/
│   ├── graphs/
│   ├── metrics/
│   ├── Final_Comparative_Research_Report.docx
│   ├── Final_Comparative_Research_Report.pdf
│   └── reports/ (Final_Comparative_Research_Report.docx, .pdf)
├── shared/
│   ├── dataset_inventory/
│   ├── subject_splits/
│   ├── preprocessed_cache/
│   └── cached_dataset.py
├── requirements.txt
└── README.md
```

---

## 2. Dataset Overview Across Parts A–E

| Dataset Part | Subjects | Samples / Recordings | Primary Prediction Task | Target Class Labels | Video Properties | Biosignal Channels |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Part A** | 87 | 8,700 short trials | Pain Intensity | 5 classes: `BL1` (0), `PA1` (1), `PA2` (2), `PA3` (3), `PA4` (4) | 1388×1038 @ 25 fps (5.52 s) | 512 Hz: GSR, ECG, EMG (trap, corr, zyg) |
| **Part B** | 86 | 8,600 short trials | Pain Intensity | 5 classes: `BL1` (0), `PA1` (1), `PA2` (2), `PA3` (3), `PA4` (4) | 1388×1038 @ 25 fps (5.52 s) | 512 Hz: GSR, ECG, EMG (trap, corr, zyg) |
| **Part C** | 87 | 87 continuous sessions | Pain Intensity (Continuous) | Stimulus segmented into 5 classes (BL1..PA4) | 1388×1038 @ 25 fps (~25.3 min) | 512 Hz raw signals + 32 Hz stimulus timestamps |
| **Part D** | 90 | 630 clips | Emotion Recognition | 7 classes: `anger`, `disgust`, `fear`, `happy`, `neutral`, `pain`, `sad` | 1388×1038 @ 25 fps (60.04 s) | 512 Hz: GSR, ECG, EMG |
| **Part E** *(BioVidEmoDB)* | 86 | 430 clips | Emotion Recognition | 5 classes: `amusement`, `anger`, `disgust`, `fear`, `sad` | 1388×1038 @ 25 fps (114.04 s) | 512 Hz: SCL (GSR), ECG, EMG (trap) |

---

## 3. Deep Learning Architectures Implemented

### Part 1: Pain Intensity Recognition
1. **ResNet-50**: Pretrained 2D spatial feature extractor (49 conv layers) with global temporal average pooling across video frames.
2. **ViViT (Video Vision Transformer)**: Factorized spatio-temporal encoder architecture (Model 2, Arnab et al., ICCV 2021) utilizing 3D tubelet embedding, 4-layer Spatial Transformer block, and 4-layer Temporal Transformer block with temporal `[CLS]` token classification.
3. **CNN-LSTM**: Pretrained ResNet-18 spatial feature extractor coupled with a 2-layer Bidirectional LSTM for dynamic temporal trajectory modeling.

### Part 2: Emotion Recognition
1. **3D CNN (R3D-18)**: Native 3D spatio-temporal convolutions capturing spatial facial features and motion velocity simultaneously.
2. **Video Swin Transformer**: Shifted-window spatio-temporal self-attention operating on 3D patch tubelets, capturing long-range micro-expression dependencies.

### Part 3: Multimodal Pain & Emotion Recognition
1. **Multimodal CNN**:
   - **Visual Branch**: Pretrained ResNet-18 spatial encoder + temporal pooling (`dim=512`).
   - **Physiological Branch**: Multi-channel 1D CNN with hierarchical convolutional blocks for GSR, ECG, and EMG (`dim=256`).
   - **Fusion Layer**: Concatenation + LayerNorm + Dropout + projection MLP (`dim=256`).
   - **Task Heads**: Separate heads for Pain (5 classes) and Emotion (7/5 classes).
   - **Ablation Configurations**: Multimodal Fusion, Video-Only, All-Biosignals-Only, GSR-Only, ECG-Only, and EMG-Only.

---

## 4. Key Experimental Results & Scientific Benchmarks

### Pain Recognition Comparison (Part 1 & Part 3 Matched Test Subjects)
| Model Architecture | Accuracy (Decimal / %) | Macro-F1 | Weighted-F1 | Latency (ms) | Parameter Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | **0.2350 (23.50%)** | 0.1939 | 0.1939 | 18.9 ms | 24,559,685 |
| **ViViT** | 0.2200 (22.00%) | 0.1158 | 0.1158 | 19.3 ms | 4,469,765 |
| **CNN-LSTM** | 0.1900 (19.00%) | 0.1100 | 0.1100 | 18.3 ms | 14,463,045 |
| **Multimodal CNN (Fusion)** | **0.2300 (23.00%)** | **0.1821** | **0.1821** | **18.0 ms** | **11,542,933** |
| *Video-Only (Multimodal CNN)* | *0.1900 (19.00%)* | *0.0655* | *0.0655* | *18.0 ms* | *11,542,933* |
| *All-Biosignals-Only* | *0.2050 (20.50%)* | *0.0838* | *0.0838* | *18.5 ms* | *11,542,933* |

### Emotion Recognition Benchmark (Part 2)
| Protocol & Model | Accuracy (Decimal / %) | Macro-F1 | Weighted-F1 | Latency (ms) | Parameter Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Part D (7 Classes) — 3D CNN (R3D-18)** | **0.1923 (19.23%)** | 0.1154 | 0.1189 | 18.7 ms | 33,299,399 |
| **Part D (7 Classes) — Video Swin** | 0.1538 (15.38%) | 0.0654 | 0.0799 | 13.1 ms | 1,024,903 |
| **Part E (5 Classes) — 3D CNN (R3D-18)** | **0.3750 (37.50%)** | **0.2882** | **0.2630** | 19.2 ms | 33,298,885 |
| **Part E (5 Classes) — Video Swin** | 0.2083 (20.83%) | 0.1127 | 0.1008 | 13.0 ms | 1,024,389 |

### Modality Ablation Breakdown (Part 3)
| Configuration | Modality Channels | Accuracy (Decimal / %) | Macro-F1 | Key Scientific Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Complete Fusion** | Video + GSR + ECG + EMG | **0.2300 (23.00%)** | **0.1821** | **Highest overall accuracy and balanced macro-F1** |
| **Video Only** | Video only | 0.1900 (19.00%) | 0.0655 | Vulnerable to facial masking |
| **All Biosignals** | GSR + ECG + EMG | 0.2050 (20.50%) | 0.0838 | Captures involuntary autonomic activation |
| **GSR Only** | Galvanic Skin Response | 0.2000 (20.00%) | 0.0667 | Most decisive individual physiological channel |
| **ECG Only** | Electrocardiogram | 0.2000 (20.00%) | 0.0667 | Contributes baseline autonomic tone |
| **EMG Only** | Electromyography | 0.1850 (18.50%) | 0.0979 | Contributes muscular contraction spikes |

---

## 5. Reproduction & Execution Instructions

### Environment Setup
```bash
# Activate virtual environment
source /home/acernitro/.venvs/main/bin/activate

# Install dependencies
pip install -r BioVid-DL/requirements.txt
```

### Pipeline Execution Order
```bash
# Phase 2: Generate dataset inventory and statistics
python BioVid-DL/shared/dataset_inventory/generate_inventory.py

# Phase 3: Generate subject-independent splits and signal plots
python BioVid-DL/shared/subject_splits/make_splits.py
python BioVid-DL/Part_1_Pain_Recognition/preprocessing/segment_part_c.py
python BioVid-DL/Part_3_Multimodal_Recognition/preprocessing/signal_processor.py

# Pre-extract cached spatial frames and signals
python BioVid-DL/shared/preprocess_and_cache.py

# Phase 4 & 5: Train and evaluate Part 1 models & compile reports
python BioVid-DL/Part_1_Pain_Recognition/train_part1_fast.py
python BioVid-DL/Part_1_Pain_Recognition/reports/generate_report.py

# Phase 6 & 7: Train and evaluate Part 2 emotion models & compile reports
python BioVid-DL/Part_2_Emotion_Recognition/train_part2_fast.py
python BioVid-DL/Part_2_Emotion_Recognition/reports/generate_report.py

# Phase 8, 9 & 10: Train Multimodal CNN, execute ablations & compile reports
python BioVid-DL/Part_3_Multimodal_Recognition/train_part3.py
python BioVid-DL/Part_3_Multimodal_Recognition/reports/generate_report.py

# Phase 11: Run final comparative synthesis & compile research reports
python BioVid-DL/Final_Comparative_Analysis/run_final_comparison.py
python BioVid-DL/Final_Comparative_Analysis/reports/generate_final_report.py
```

---

## 6. Generated Publication Reports

All four academic reports have been generated and exported as both `.docx` and `.pdf`:

1. **Part 1 Pain Recognition Report**:
   - `BioVid-DL/Part_1_Pain_Recognition/Part_1_Pain_Recognition_Report.docx`
   - `BioVid-DL/Part_1_Pain_Recognition/Part_1_Pain_Recognition_Report.pdf`
   - *(Also available in `Part_1_Pain_Recognition/reports/`)*

2. **Part 2 Emotion Recognition Report**:
   - `BioVid-DL/Part_2_Emotion_Recognition/Part_2_Emotion_Recognition_Report.docx`
   - `BioVid-DL/Part_2_Emotion_Recognition/Part_2_Emotion_Recognition_Report.pdf`
   - *(Also available in `Part_2_Emotion_Recognition/reports/`)*

3. **Part 3 Multimodal Recognition Report**:
   - `BioVid-DL/Part_3_Multimodal_Recognition/Part_3_Multimodal_Recognition_Report.docx`
   - `BioVid-DL/Part_3_Multimodal_Recognition/Part_3_Multimodal_Recognition_Report.pdf`
   - *(Also available in `Part_3_Multimodal_Recognition/reports/`)*

4. **Final Comparative Research Report**:
   - `BioVid-DL/Final_Comparative_Analysis/Final_Comparative_Research_Report.docx`
   - `BioVid-DL/Final_Comparative_Analysis/Final_Comparative_Research_Report.pdf`
   - *(Also available in `Final_Comparative_Analysis/reports/`)*
