#!/usr/bin/env python3
"""
BioVid Pain Recognition Pipeline with ViViT (Video Vision Transformer)
Master orchestrator executing prompt.md Steps 1 through 20 end-to-end.
"""

import argparse
import sys
import time

from step1_inspect_dataset import run_step1_to_step9_inspection
from step2_build_manifest_and_splits import build_manifest_and_splits
from step3_visualize_samples import generate_visual_inspection_grid
from step4_trace_single_video import trace_and_verify_single_video
from step5_memory_test import run_memory_test
from step6_train_and_evaluate import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End BioVid ViViT Pipeline (prompt.md)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for training (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8, safe for RTX 4050 6GB)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers (default: 2)",
    )
    parser.add_argument(
        "--max_train_subjects",
        type=int,
        default=None,
        help="Optionally limit subjects for faster execution",
    )
    parser.add_argument(
        "--max_samples_per_subject",
        type=int,
        default=None,
        help="Optionally limit samples per subject for faster execution",
    )

    args = parser.parse_args()

    pipeline_start = time.time()
    print("=" * 80)
    print("STARTING FULL END-TO-END BIOVID ViViT PIPELINE (prompt.md)")
    print("=" * 80)

    # STAGE 1: Steps 1 - 9
    print("\n>>> STAGE 1: DATASET INSPECTION & RECONNAISSANCE (Steps 1 - 9)")
    run_step1_to_step9_inspection()

    # STAGE 2: Steps 10 - 11
    print("\n>>> STAGE 2: MANIFEST & SUBJECT-LEVEL SPLITTING (Steps 10 - 11)")
    build_manifest_and_splits()

    # STAGE 3: Step 12
    print("\n>>> STAGE 3: VISUAL DATA INSPECTION (Step 12)")
    generate_visual_inspection_grid()

    # STAGE 4: Steps 13 - 16
    print("\n>>> STAGE 4: ViViT TRACING & SINGLE-SAMPLE VERIFICATION (Steps 13 - 16)")
    trace_and_verify_single_video()

    # STAGE 5: Step 17
    print("\n>>> STAGE 5: GPU MEMORY TEST ON RTX 4050 (Step 17)")
    run_memory_test()

    # STAGE 6: Steps 18 - 19
    print("\n>>> STAGE 6: MODEL TRAINING & UNSEEN SUBJECT EVALUATION (Steps 18 - 19)")
    train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        max_train_subjects=args.max_train_subjects,
        max_samples_per_subject=args.max_samples_per_subject,
    )

    total_time = time.time() - pipeline_start
    print("=" * 80)
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
