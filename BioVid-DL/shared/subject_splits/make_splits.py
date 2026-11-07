import os
import json
import random
import pandas as pd
import numpy as np

SPLITS_DIR = "BioVid-DL/shared/subject_splits"
os.makedirs(SPLITS_DIR, exist_ok=True)
INV_PATH = "BioVid-DL/shared/dataset_inventory/dataset_inventory.csv"

def generate_subject_splits():
    print("Generating strictly subject-independent splits across all BioVid parts...")
    df = pd.read_csv(INV_PATH)

    # Collect unique subject identifiers
    # For Part A/B/C/D, subject_name format is e.g. 071309_w_21
    # For Part E, subject_name format is e.g. 120713-09_w_21 -> canonical key: suffix 09_w_21 or base
    def get_canonical_subject(row):
        sname = str(row["subject_name"])
        if row["dataset_part"] == "PartE":
            parts = sname.split("-")
            if len(parts) >= 2:
                return parts[1] # e.g. 09_w_21
            return sname
        # For A/B/C/D e.g. 071309_w_21
        if len(sname) >= 11 and "_" in sname:
            return sname[4:] # e.g. 09_w_21
        return sname

    df["canonical_subject"] = df.apply(get_canonical_subject, axis=1)

    unique_subjects = sorted(df["canonical_subject"].unique())
    print(f"Total unique canonical subject clusters identified: {len(unique_subjects)}")

    # Set deterministic random seed
    random.seed(42)
    np.random.seed(42)
    shuffled_subjects = list(unique_subjects)
    random.shuffle(shuffled_subjects)

    n_total = len(shuffled_subjects)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_subs = set(shuffled_subjects[:n_train])
    val_subs = set(shuffled_subjects[n_train:n_train + n_val])
    test_subs = set(shuffled_subjects[n_train + n_val:])

    print(f"Subject partition: {len(train_subs)} Train, {len(val_subs)} Val, {len(test_subs)} Test")

    # Map back to row-level split
    def assign_split(sub):
        if sub in train_subs:
            return "train"
        elif sub in val_subs:
            return "val"
        else:
            return "test"

    df["split"] = df["canonical_subject"].apply(assign_split)

    # Save summary
    splits_meta = {
        "train_subjects": sorted(list(train_subs)),
        "val_subjects": sorted(list(val_subs)),
        "test_subjects": sorted(list(test_subs)),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_by_part": df.groupby(["dataset_part", "split"]).size().unstack(fill_value=0).to_dict(),
    }

    with open(os.path.join(SPLITS_DIR, "subject_splits.json"), "w") as f:
        json.dump(splits_meta, f, indent=2)

    df.to_csv(os.path.join(SPLITS_DIR, "inventory_with_splits.csv"), index=False)
    print("Subject-independent splits saved successfully.")
    print("Distribution by part and split:")
    print(df.groupby(["dataset_part", "split"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    generate_subject_splits()
