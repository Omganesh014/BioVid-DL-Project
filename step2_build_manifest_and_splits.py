import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import config
from src.dataset import LABEL_MAP


def build_manifest_and_splits():
    print("=" * 80)
    print("STEP 10: BUILD THE BIOVID DATASET MANIFEST")
    print("=" * 80)

    samples_csv = config.samples_csv_path
    if not samples_csv.exists():
        raise FileNotFoundError(f"Part A samples.csv not found: {samples_csv}")

    raw_df = pd.read_csv(samples_csv, sep="\t")
    print(f"Loaded {len(raw_df)} sample records from {samples_csv}")

    part_a_root = config.part_a_root
    records = []
    missing_videos = []

    for idx, row in raw_df.iterrows():
        sub_id = int(row["subject_id"])
        sub_name = str(row["subject_name"])
        sample_id = int(row["sample_id"])
        sample_name = str(row["sample_name"])
        class_id = int(row["class_id"])
        class_name = str(row["class_name"])

        # Strict check on class mapping
        if class_id not in LABEL_MAP or LABEL_MAP[class_id] != class_name:
            raise ValueError(
                f"Label integrity violated at row {idx}: class_id={class_id}, class_name={class_name}"
            )

        video_path = part_a_root / "video" / sub_name / f"{sample_name}.mp4"
        if not video_path.exists():
            missing_videos.append(str(video_path))

        records.append({
            "subject_id": sub_id,
            "subject_name": sub_name,
            "trial_id": sample_id,
            "sample_name": sample_name,
            "video_path": str(video_path),
            "label": class_id,
            "class_name": class_name,
            "fps": config.raw_fps,
            "frame_count": config.raw_frame_count,
            "duration": round(config.raw_frame_count / config.raw_fps, 2),
        })

    if missing_videos:
        raise FileNotFoundError(
            f"Found {len(missing_videos)} missing video files! First: {missing_videos[0]}"
        )

    manifest_df = pd.DataFrame(records)
    manifest_path = config.manifest_dir / "biovid_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Successfully generated manifest with {len(manifest_df)} rows at: {manifest_path}")

    print("\n" + "=" * 80)
    print("STEP 11: SUBJECT-LEVEL SPLIT (TRAIN / VAL / TEST)")
    print("=" * 80)

    # Extract all unique subject IDs
    unique_subjects = sorted(manifest_df["subject_name"].unique())
    num_subjects = len(unique_subjects)
    print(f"Total unique subjects: {num_subjects}")

    # Set seed for reproducible split
    rng = np.random.RandomState(config.seed)
    shuffled_subjects = rng.permutation(unique_subjects)

    n_train = int(num_subjects * config.train_ratio)
    n_val = int(num_subjects * config.val_ratio)
    n_test = num_subjects - n_train - n_val

    train_subjects = set(shuffled_subjects[:n_train])
    val_subjects = set(shuffled_subjects[n_train : n_train + n_val])
    test_subjects = set(shuffled_subjects[n_train + n_val :])

    print(f"Split breakdown: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)} subjects")

    # STRICT ASSERTIONS AGAINST DATA LEAKAGE
    print("Verifying subject isolation (ZERO leakage)...")
    assert len(train_subjects & val_subjects) == 0, "DATA LEAKAGE: Subject in train and val!"
    assert len(train_subjects & test_subjects) == 0, "DATA LEAKAGE: Subject in train and test!"
    assert len(val_subjects & test_subjects) == 0, "DATA LEAKAGE: Subject in val and test!"
    assert len(train_subjects | val_subjects | test_subjects) == num_subjects, "Missing subjects!"
    print("-> ZERO SUBJECT LEAKAGE CONFIRMED: Train, Val, and Test sets have mutually disjoint subjects.")

    # Create subset DataFrames
    train_df = manifest_df[manifest_df["subject_name"].isin(train_subjects)].copy()
    val_df = manifest_df[manifest_df["subject_name"].isin(val_subjects)].copy()
    test_df = manifest_df[manifest_df["subject_name"].isin(test_subjects)].copy()

    # Verify duplicate trial IDs per subject
    for split_name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dups = df_split.duplicated(subset=["subject_name", "sample_name"]).sum()
        assert dups == 0, f"Found {dups} duplicate trials in {split_name} split!"

    total_samples = len(train_df) + len(val_df) + len(test_df)
    assert total_samples == len(manifest_df), f"Sample count mismatch: {total_samples} vs {len(manifest_df)}"

    print(f"Train samples: {len(train_df)} ({len(train_df)//5} per class across 5 classes)")
    print(f"Val samples:   {len(val_df)} ({len(val_df)//5} per class across 5 classes)")
    print(f"Test samples:  {len(test_df)} ({len(test_df)//5} per class across 5 classes)")

    # Save splits
    train_path = config.manifest_dir / "train.csv"
    val_path = config.manifest_dir / "val.csv"
    test_path = config.manifest_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    print("=" * 80)
    print("STEP 10 & 11 COMPLETE: MANIFEST AND SUBJECT SPLITS SUCCESSFULLY GENERATED!\n")


if __name__ == "__main__":
    build_manifest_and_splits()
