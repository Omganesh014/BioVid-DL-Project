# BioVid Deep Learning Project: ViViT Pain Recognition

Automated, objective pain assessment using Video Vision Transformers (ViViT Model 2: Factorized Encoder) trained on the official **BioVid Heat Pain Database (Part A)**.

---

## 📌 Project Overview
This repository contains the end-to-end deep learning pipeline for pain recognition from facial video sequences. The architecture implements strict subject-level isolation to prevent data leakage and utilizes factorized spatial-temporal transformers for accurate pain level classification across 5 calibrated classes:
- **BL1** (Baseline / No pain)
- **PA1** (Pain threshold)
- **PA2** (Mild pain)
- **PA3** (Moderate pain)
- **PA4** (Severe pain / Pain tolerance)

For in-depth theoretical background, dataset design, ViViT mathematical formulation, and architecture diagrams, see [BIOVID_VIVIT_PROJECT_GUIDE.md](BIOVID_VIVIT_PROJECT_GUIDE.md).

---

## 🏗️ Repository Structure

```
├── BIOVID_VIVIT_PROJECT_GUIDE.md   # Architectural reference & deep documentation
├── run_pipeline.py                 # Master script to execute all pipeline stages
├── step1_inspect_dataset.py        # Dataset structure verification & codec inspection
├── step2_build_manifest_and_splits.py # Zero-leakage subject-level split generator
├── step3_visualize_samples.py      # Sample inspection and grid visualizer
├── step4_trace_single_video.py     # Single video decoding & tensor trace
├── step5_memory_test.py            # VRAM & forward/backward memory benchmark
├── step6_train_and_evaluate.py     # ViViT training, evaluation, & metrics export
├── src/
│   ├── config.py                   # Centralized hyperparameter and path config
│   ├── dataset.py                  # PyTorch Dataset for BioVid video sequences
│   ├── utils.py                    # Metric utilities, confusion matrix, logging
│   └── models/
│       └── vivit.py                # ViViT Model 2 (Factorized Spatial-Temporal Encoder)
├── outputs/
│   ├── checkpoints/                # Model weights (best_model.pt, last_model.pt)
│   ├── evaluation/                 # Confusion matrix and evaluation plots
│   ├── logs/                       # Metrics, test predictions, and train history
│   ├── manifest/                   # biovid_manifest.csv, train.csv, val.csv, test.csv
│   └── visualizations/             # Sample grid visualizations
└── sample_inspection/              # Cropped sample inspection frames
```

---

## 🚀 Pipeline Workflow

To run the complete pipeline sequentially:
```bash
python run_pipeline.py
```

Or execute individual steps:
1. **Dataset Inspection**: `python step1_inspect_dataset.py`
2. **Build Manifests & Splits**: `python step2_build_manifest_and_splits.py`
3. **Visualize Video Frames**: `python step3_visualize_samples.py`
4. **Trace Single Sample**: `python step4_trace_single_video.py`
5. **Memory Benchmark**: `python step5_memory_test.py`
6. **Train & Evaluate**: `python step6_train_and_evaluate.py`
