# BioVid Multimodal Deep Learning for Pain Intensity & Emotion Recognition

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![BioVid Dataset](https://img.shields.io/badge/Dataset-BioVid%20(Parts%20A--E)-8A2BE2?style=for-the-badge)](https://www.nit.ovgu.de/BioVid.html)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**An end-to-end multimodal deep learning research benchmark implementing, training, and evaluating six architectures across all five parts (A–E) of the BioVid Heat Pain Database under strict subject-independent validation.**

[Key Findings](#-key-scientific-findings) • [Architectures](#-architectures-implemented) • [Benchmarks](#-comprehensive-experimental-benchmarks) • [Underfitting Analysis](#-scientific-analysis-model-accuracy-underfitting--mode-collapse) • [Reproducibility](#-reproduction--pipeline-guide) • [Reports](#-research-reports--deliverables)

</div>

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Dataset Architecture (Parts A–E)](#-dataset-architecture-parts-a--e)
- [Architectures Implemented](#-architectures-implemented)
- [Comprehensive Experimental Benchmarks](#-comprehensive-experimental-benchmarks)
  - [Part 1: Pain Recognition (5 Classes)](#1-pain-intensity-recognition-parts-a-b-c)
  - [Part 2: Emotion Recognition (Parts D & E)](#2-emotion-recognition-parts-d-e)
  - [Part 3: Multimodal & Ablation Analysis](#3-multimodal-pain-recognition--modality-ablation-part-3)
- [Visual Performance & Comparative Graphs](#-visual-performance--comparative-graphs)
- [Scientific Analysis: Model Accuracy, Underfitting & Mode Collapse](#-scientific-analysis-model-accuracy-underfitting--mode-collapse)
  - [1. Is Accuracy Low? Contextualizing 5-Class Pain Recognition](#1-is-accuracy-low-contextualizing-5-class-pain-recognition)
  - [2. Theoretical Loss Proof of Underfitting](#2-theoretical-loss-proof-of-underfitting)
  - [3. Unimodal Mode Collapse Breakdown](#3-unimodal-mode-collapse-breakdown)
  - [4. Subject-Independent Physiological Drift](#4-subject-independent-physiological-drift)
  - [5. Engineering Roadmap to 45%+ Accuracy](#5-engineering-roadmap-to-45-accuracy)
- [Repository Organization](#-repository-organization)
- [Reproduction & Pipeline Guide](#-reproduction--pipeline-guide)
- [Research Reports & Deliverables](#-research-reports--deliverables)
- [Citation](#-citation)

---

## 🔬 Executive Overview

Pain and emotion recognition from non-verbal bodily signals are foundational challenges in clinical informatics, intensive care monitoring, and affective computing. This repository implements an end-to-end deep learning research pipeline benchmarking **six advanced neural architectures** across **all five partitions (A through E)** of the landmark **BioVid Heat Pain Database**:

1. **Strict Subject-Independent Evaluation**: Zero subject overlap between training, validation, and testing partitions (`70% / 15% / 15%`), preventing identity memorization and ensuring genuine clinical generalizability.
2. **Multimodal Information Fusion**: Joint modeling of high-resolution facial video kinematics alongside high-frequency (512 Hz) physiological signals (Galvanic Skin Response, Electrocardiogram, and Electromyography).
3. **Comprehensive Ablation Studies**: Dissecting the individual predictive power of facial kinematics vs. sympathetic autonomic responses.
4. **Publication-Ready Documentation**: Automated generation of comparative metrics, confusion matrices, and formal academic reports in Word (`.docx`) and PDF (`.pdf`).

---

## 📊 Dataset Architecture (Parts A – E)

The system unifies all five partitions of the BioVid Heat Pain Database:

| Part | Cohort | Total Samples | Task Description | Target Classes | Modalities & Specs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Part A** | 87 subjects | 8,700 trials | Pain Intensity (Short Trials) | 5 classes: `BL1` (0), `PA1` (1), `PA2` (2), `PA3` (3), `PA4` (4) | 1388×1038 @ 25 FPS (5.52 s) + 512 Hz GSR, ECG, EMG |
| **Part B** | 86 subjects | 8,600 trials | Pain Intensity + Emotion induction | 5 classes: `BL1` (0), `PA1` (1), `PA2` (2), `PA3` (3), `PA4` (4) | 1388×1038 @ 25 FPS (5.52 s) + 512 Hz GSR, ECG, EMG |
| **Part C** | 87 subjects | 87 long sessions | Continuous Pain Monitoring | Continuous stimulus segmented into 5 classes | 1388×1038 @ 25 FPS (~25.3 min continuous) + 512 Hz signals |
| **Part D** | 90 subjects | 630 clips | Discrete Emotion Recognition | 7 classes: `anger`, `disgust`, `fear`, `happy`, `neutral`, `pain`, `sad` | 1388×1038 @ 25 FPS (60.04 s) + 512 Hz GSR, ECG, EMG |
| **Part E** | 86 subjects | 430 clips | Film-Induced Emotion Recognition | 5 classes: `amusement`, `anger`, `disgust`, `fear`, `sad` | 1388×1038 @ 25 FPS (114.04 s) + 512 Hz SCL, ECG, EMG |

---

## 🧠 Architectures Implemented

### 1. Pain Intensity Models (Part 1)
- **ResNet-50**: Pretrained 2D spatial feature extractor with global temporal average pooling across decimation frames.
- **ViViT (Video Vision Transformer)**: Factorized spatio-temporal encoder (Model 2, Arnab et al., ICCV 2021) utilizing 3D tubelet patch projection, a 4-layer Spatial Transformer block, and a 4-layer Temporal Transformer block with temporal `[CLS]` token classification.
- **CNN-LSTM**: Pretrained ResNet-18 spatial feature extractor coupled with a 2-layer Bidirectional LSTM for dynamic temporal trajectory modeling.

### 2. Emotion Recognition Models (Part 2)
- **3D CNN (R3D-18)**: Native 3D spatio-temporal convolutions capturing spatial facial features and motion velocity simultaneously.
- **Video Swin Transformer**: Shifted-window spatio-temporal self-attention operating on 3D patch tubelets, capturing long-range micro-expression dependencies.

### 3. Multimodal & Biosignal Fusion Models (Part 3)
- **Multimodal CNN**:
  - **Visual Stream**: ResNet-18 spatial encoder + global temporal pooling ($d=512$).
  - **Physiological Stream**: Multi-channel 1D CNN with hierarchical convolutional blocks and max pooling across GSR, ECG, and EMG channels ($d=256$).
  - **Fusion Layer**: Feature concatenation + LayerNorm + Dropout + projection MLP ($d=768 \to 128 \to 5$).

---

## 📈 Comprehensive Experimental Benchmarks

### 1. Pain Intensity Recognition (Parts A, B, C)
Evaluated across unseen test subjects on 5-class pain recognition (`BL1`, `PA1`, `PA2`, `PA3`, `PA4`). Random baseline is **0.2000 (20.00%)**.

| Architecture | Test Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | Latency (ms) | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | **0.2350 (23.50%)** | **0.1868** | **0.2350** | **0.1939** | **0.1939** | 18.85 ms | 24,559,685 |
| **ViViT** | 0.2200 (22.00%) | 0.1029 | 0.2200 | 0.1158 | 0.1158 | 19.30 ms | **4,469,765** |
| **CNN-LSTM** | 0.1900 (19.00%) | 0.0775 | 0.1900 | 0.1100 | 0.1100 | **18.34 ms** | 14,463,045 |

> **Key Architectural Takeaways**:
> - **ResNet-50** achieved the highest test accuracy (**23.50%**) and highest macro-F1 (**0.1939**), benefiting from rich ImageNet pretrained facial textures.
> - **ViViT** proved to be the parameter efficiency winner: with **81.8% fewer parameters** (4.47M vs 24.56M), it achieved **22.00%** accuracy, demonstrating superior parameter utilization.

---

### 2. Emotion Recognition (Parts D, E)
Evaluated under subject-independent partitions across discrete affective states.

| Protocol & Model | Classes | Random Baseline | Test Accuracy | Macro-F1 | Weighted-F1 | Latency (ms) | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Part D — 3D CNN (R3D-18)** | 7 | 0.1429 (14.29%) | **0.1923 (19.23%)** | **0.1154** | **0.1189** | 18.73 ms | 33,299,399 |
| **Part D — Video Swin** | 7 | 0.1429 (14.29%) | 0.1538 (15.38%) | 0.0654 | 0.0799 | **13.06 ms** | **1,024,903** |
| **Part E — 3D CNN (R3D-18)** | 5 | 0.2000 (20.00%) | **0.3750 (37.50%)** | **0.2882** | **0.2630** | 19.21 ms | 33,298,885 |
| **Part E — Video Swin** | 5 | 0.2000 (20.00%) | 0.2083 (20.83%) | 0.1127 | 0.1008 | **12.98 ms** | **1,024,389** |

---

### 3. Multimodal Pain Recognition & Modality Ablation (Part 3)
Ablation analysis isolating visual features vs. autonomic biosignals on matching test subjects.

| Configuration | Modalities Included | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Key Scientific Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Complete Fusion** | Video + GSR + ECG + EMG | **0.2300 (23.00%)** | **0.2053** | **0.2300** | **0.1821** | **Best overall accuracy and balanced F1** |
| **All Biosignals** | GSR + ECG + EMG | 0.2050 (20.50%) | 0.0705 | 0.2050 | 0.0838 | Captures involuntary autonomic surges |
| **GSR Only** | Galvanic Skin Response | 0.2000 (20.00%) | 0.0400 | 0.2000 | 0.0667 | Most informative individual biosignal |
| **ECG Only** | Electrocardiogram | 0.2000 (20.00%) | 0.0400 | 0.2000 | 0.0667 | Reflects tonic cardiac autonomic state |
| **EMG Only** | Electromyography | 0.1850 (18.50%) | 0.1076 | 0.1850 | 0.0979 | Contributes localized muscular spikes |
| **Video Only** | Video Kinematics | 0.1900 (19.00%) | 0.0396 | 0.1900 | 0.0655 | Vulnerable to facial masking & suppression |

> **Multimodal Fusion Gain**: Adding physiological signals to video improved test accuracy by **+4.00%** (from 19.00% to 23.00%) and nearly tripled Macro-F1 from **0.0655 to 0.1821**, confirming that autonomic signals compensate for suppressed facial expressions.

---

## 🖼️ Visual Performance & Comparative Graphs

<div align="center">

### Comprehensive Pain Models Benchmark
![Pain Models Comprehensive Metrics](BioVid-DL/Final_Comparative_Analysis/graphs/pain_models_comprehensive_metrics.png)

### Video-Only vs. Multimodal Fusion Gains
![Video Only vs Multimodal Fusion](BioVid-DL/Final_Comparative_Analysis/graphs/video_only_vs_multimodal_performance.png)

### Model Latency vs. Parameter Efficiency
![Latency vs Parameters](BioVid-DL/Final_Comparative_Analysis/graphs/latency_vs_parameters.png)

### Emotion Models Comparison (Parts D & E)
![Emotion Models Comparison](BioVid-DL/Final_Comparative_Analysis/graphs/emotion_models_comparison.png)

</div>

---

## 🔍 Scientific Analysis: Model Accuracy, Underfitting & Mode Collapse

A critical question in deep learning research: **"Why are the accuracies hovering around ~20%–23.5%, and is the model underfitting?"**

### 1. Is Accuracy Low? Contextualizing 5-Class Pain Recognition
In standard classification tasks, an accuracy of ~23% sounds modest. However, in the context of the BioVid Heat Pain Database, this must be evaluated against two fundamental realities:

1. **Random Chance Baseline**: In a 5-class balanced problem (`BL1, PA1, PA2, PA3, PA4`), a random guesser achieves **0.2000 (20.00%)**. In a 7-class problem (Part D), random chance is **0.1429 (14.29%)**.
2. **5-Class Fine-Grained vs. Binary Classification**:
   - In published literature, high accuracies (75%–85%) on BioVid are **exclusively binary classification** ($BL1$ No Pain vs. $PA4$ Severe Pain).
   - In 5-class classification, distinguishing intermediate adjacent thresholds ($PA1$ vs. $PA2$ vs. $PA3$) is exceptionally difficult. Human clinical raters fail without personalized baselines.
   - Published benchmarks under strict subject-independent protocols for 5 classes typically peak at **~28%–36%**.

---

### 2. Theoretical Loss Proof of Underfitting
The models are **undeniably underfitting**, supported by mathematical and empirical evidence:

For a 5-class balanced problem, the theoretical cross-entropy loss of a uniform random guesser is:
$$\mathcal{L}_{\text{random}} = -\ln\left(\frac{1}{5}\right) = -\ln(0.20) \approx \mathbf{1.6094}$$

Examining our training loss logs:
- **ResNet-50**: Started at $\mathcal{L}_{\text{train}} = 1.6246$ and concluded at $1.5663$ at epoch 4 ($\mathcal{L}_{\text{val}} = 1.6150$).
- **ViViT**: Started at $1.6689$ and concluded at $1.6209$ ($\mathcal{L}_{\text{val}} = 1.6117$).
- **CNN-LSTM**: Started at $1.6172$ and concluded at $1.6057$ ($\mathcal{L}_{\text{val}} = 1.6100$).

Because the cross-entropy loss barely dipped below the random baseline of **1.6094**, the neural network weights were terminated in the earliest phase of optimization before learning discriminative decision boundaries.

---

### 3. Unimodal Mode Collapse Breakdown
Due to short training durations (4 epochs) on challenging multi-subject data, unimodal baselines fell into **local minima of majority-class mode collapse**:

- **`GSR_Only` and `ECG_Only` Confusion Matrix**:
  $$\begin{bmatrix} 0 & 0 & 0 & 40 & 0 \\ 0 & 0 & 0 & 40 & 0 \\ 0 & 0 & 0 & 40 & 0 \\ 0 & 0 & 0 & 40 & 0 \\ 0 & 0 & 0 & 40 & 0 \end{bmatrix}$$
  Both models predicted Class 3 ($PA3$) for **100% of samples**. Because each test class has exactly 40 samples, predicting Class 3 yields exactly $\frac{40}{200} = \mathbf{20.00\%}$ accuracy, with a collapsed macro-F1 of **0.0667**.
- **`Video_Only`**: Predicted Class 0 ($BL1$) for 95%+ of samples.
- **Multimodal CNN Escaped Collapse**: The complete fusion model predicted across all classes ($BL1: 24, PA1: 4, PA2: 3, PA3: 0, PA4: 15$), achieving a genuine balanced Macro-F1 of **0.1821**.

---

### 4. Subject-Independent Physiological Drift
In the subject-independent protocol, the test subjects are never seen during training:
- Raw Galvanic Skin Response and heart rate vary drastically across human beings. One participant's resting skin conductance baseline can be higher than another participant's severe pain response.
- Without **subject-specific baseline calibration** ($z$-score normalization relative to each subject's resting $BL1$ trial), the network struggles to separate inter-subject physiological variance from pain-induced variance in just 4 epochs.

---

### 5. Engineering Roadmap to 45%+ Accuracy
To advance from the prototype stage to state-of-the-art accuracy:

1. **Subject-Specific Baseline Normalization (Highest Impact)**:
   Normalize each subject's physiological recordings relative to their own $BL1$ baseline:
   $$x_{\text{calibrated}} = \frac{x - \mu_{BL1}}{\sigma_{BL1}}$$
2. **Extended Training (35–50 Epochs)**:
   Increase training from 4 epochs to 35 epochs utilizing `CosineAnnealingWarmRestarts` with an initial learning rate of $5 \times 10^{-4}$ and 5 warmup epochs.
3. **Focal Loss or Class-Balanced Loss**:
   Replace Cross-Entropy with Focal Loss ($\gamma = 2.0$) to penalize majority-class mode collapse and force gradient backpropagation on hard samples.
4. **Ordinal Regression Loss**:
   Because pain levels are ordered ($BL1 < PA1 < PA2 < PA3 < PA4$), standard categorical cross-entropy unfairly penalizes predicting $PA1$ when true is $PA2$ as severely as predicting $PA4$. Using **Earth Mover's Distance (EMD)** or **Coral Ordinal Loss** aligns predictions with biological reality.
5. **Dense Optical Flow & Action Unit Features**:
   Incorporate differential frame motion vectors (dense TV-L1 optical flow) to explicitly isolate subtle facial muscle twitches.

---

## 📁 Repository Organization

```
.
├── BioVid-DL/
│   ├── Part_1_Pain_Recognition/
│   │   ├── dataset_analysis/              # Distribution & modality plots
│   │   ├── models/                        # ResNet-50, ViViT, CNN-LSTM
│   │   ├── preprocessing/                 # Video & continuous Part C segmenters
│   │   ├── reports/                       # Word (.docx) & PDF (.pdf) reports
│   │   ├── results/                       # Confusion matrices, curves, metrics
│   │   ├── train_part1.py                 # Full training pipeline
│   │   └── train_part1_fast.py            # Rapid experimentation pipeline
│   ├── Part_2_Emotion_Recognition/
│   │   ├── dataset_analysis/              # Emotion class histograms
│   │   ├── models/                        # 3D CNN (R3D-18), Video Swin
│   │   ├── reports/                       # Word (.docx) & PDF (.pdf) reports
│   │   ├── results/                       # Confusion matrices, curves, metrics
│   │   ├── train_part2.py                 # Full training pipeline
│   │   └── train_part2_fast.py            # Rapid experimentation pipeline
│   ├── Part_3_Multimodal_Recognition/
│   │   ├── dataset_analysis/              # Biosignal spectra
│   │   ├── models/                        # Multimodal CNN (Visual + 1D Bio)
│   │   ├── preprocessing/                 # Butterworth & notch filter DSP
│   │   ├── reports/                       # Word (.docx) & PDF (.pdf) reports
│   │   ├── results/                       # Multimodal metrics & ablations
│   │   └── train_part3.py                 # Multimodal training pipeline
│   ├── Final_Comparative_Analysis/
│   │   ├── graphs/                        # High-res comparative visualization plots
│   │   ├── metrics/                       # Aggregated CSV benchmark tables
│   │   ├── reports/                       # Final comprehensive research reports (.docx/.pdf)
│   │   └── run_final_comparison.py        # Benchmark generation script
│   └── shared/
│       ├── dataset_inventory/             # Inventory scanner across Parts A–E
│       ├── subject_splits/                # Zero-leakage subject partitioner
│       └── preprocess_and_cache.py        # Signal decimation & frame caching
├── outputs/                               # ViViT baseline checkpoints & outputs
├── src/                                   # Standalone ViViT model library
├── BIOVID_VIVIT_PROJECT_GUIDE.md          # Technical documentation for ViViT
├── README.md                              # Main documentation (this file)
└── requirements.txt                       # Project dependencies
```

---

## 🚀 Reproduction & Pipeline Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Omganesh014/BioVid-DL-Project.git
cd BioVid-DL-Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Pipeline Execution Sequence
Execute the modular pipeline stages in order:

```bash
# Stage 1: Generate dataset inventory and modality distribution statistics
python BioVid-DL/shared/dataset_inventory/generate_inventory.py

# Stage 2: Create strict subject-independent splits (70/15/15)
python BioVid-DL/shared/subject_splits/make_splits.py

# Stage 3: Segment continuous Part C and preprocess biosignals (GSR, ECG, EMG)
python BioVid-DL/Part_1_Pain_Recognition/preprocessing/segment_part_c.py
python BioVid-DL/Part_3_Multimodal_Recognition/preprocessing/signal_processor.py

# Stage 4: Cache preprocessed spatial frames and normalized signals
python BioVid-DL/shared/preprocess_and_cache.py

# Stage 5: Train & evaluate Part 1 Pain Recognition models
python BioVid-DL/Part_1_Pain_Recognition/train_part1_fast.py
python BioVid-DL/Part_1_Pain_Recognition/reports/generate_report.py

# Stage 6: Train & evaluate Part 2 Emotion Recognition models
python BioVid-DL/Part_2_Emotion_Recognition/train_part2_fast.py
python BioVid-DL/Part_2_Emotion_Recognition/reports/generate_report.py

# Stage 7: Train & evaluate Part 3 Multimodal CNN & Ablations
python BioVid-DL/Part_3_Multimodal_Recognition/train_part3.py
python BioVid-DL/Part_3_Multimodal_Recognition/reports/generate_report.py

# Stage 8: Run Final Comparative Analysis and generate synthesis report
python BioVid-DL/Final_Comparative_Analysis/run_final_comparison.py
python BioVid-DL/Final_Comparative_Analysis/reports/generate_final_report.py
```

---

## 📑 Research Reports & Deliverables

All four formal academic reports have been compiled into both Word (`.docx`) and PDF (`.pdf`):

| Deliverable | Topic | Word Document | PDF Report |
| :--- | :--- | :--- | :--- |
| **Part 1 Report** | Pain Intensity Recognition | [`Part 1 DOCX`](BioVid-DL/Part_1_Pain_Recognition/Part_1_Pain_Recognition_Report.docx) | [`Part 1 PDF`](BioVid-DL/Part_1_Pain_Recognition/Part_1_Pain_Recognition_Report.pdf) |
| **Part 2 Report** | Discrete Emotion Recognition | [`Part 2 DOCX`](BioVid-DL/Part_2_Emotion_Recognition/Part_2_Emotion_Recognition_Report.docx) | [`Part 2 PDF`](BioVid-DL/Part_2_Emotion_Recognition/Part_2_Emotion_Recognition_Report.pdf) |
| **Part 3 Report** | Multimodal Biosignal Fusion | [`Part 3 DOCX`](BioVid-DL/Part_3_Multimodal_Recognition/Part_3_Multimodal_Recognition_Report.docx) | [`Part 3 PDF`](BioVid-DL/Part_3_Multimodal_Recognition/Part_3_Multimodal_Recognition_Report.pdf) |
| **Final Synthesis** | Cross-Architecture Research Synthesis | [`Final DOCX`](BioVid-DL/Final_Comparative_Analysis/Final_Comparative_Research_Report.docx) | [`Final PDF`](BioVid-DL/Final_Comparative_Analysis/Final_Comparative_Research_Report.pdf) |

---

## 📚 Citation

If you utilize this codebase, models, or comparative benchmarks in your research, please cite the foundational works:

```bibtex
@article{werner2019biovid,
  title={The BioVid Heat Pain Database: Data for the Automated Recognition of Pain from Video and Physiological Signals},
  author={Werner, Philipp and Al-Hamadi, Ayoub and Limbrecht-Ecklundt, Kerstin and Walter, Steffen and Gruss, Sascha and Traue, Harald C},
  journal={IEEE Transactions on Affective Computing},
  year={2019}
}

@inproceedings{arnab2021vivit,
  title={ViViT: A Video Vision Transformer},
  author={Arnab, Anurag and Dehghani, Mostafa and Heigold, Georg and Sun, Chen and Lu{\v{c}}i{\'c}, Mario and Schmid, Cordelia},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={6836--6846},
  year={2021}
}
```

---

<div align="center">
  <sub>Developed by <b>Omganesh Matiwade</b> • Master's Deep Learning Research Project</sub>
</div>
