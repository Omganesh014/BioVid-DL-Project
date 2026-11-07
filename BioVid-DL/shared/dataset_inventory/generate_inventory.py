import os
import glob
import json
import re
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

DB_ROOT = "/run/media/acernitro/Ventoy/DB"
SHARED_INV_DIR = "BioVid-DL/shared/dataset_inventory"
os.makedirs(SHARED_INV_DIR, exist_ok=True)

def parse_part_e_subject(filename):
    base = os.path.basename(filename).replace(".mp4", "").replace("_bio.csv", "")
    parts = base.split("-")
    if len(parts) >= 2:
        return parts[0] + "-" + parts[1]
    return parts[0]

def build_inventory():
    print("Generating comprehensive dataset inventory across Parts A, B, C, D, E...")
    records = []

    # Part A
    part_a_samples = os.path.join(DB_ROOT, "PartA/starting_point/samples.csv")
    if os.path.exists(part_a_samples):
        df_a = pd.read_csv(part_a_samples, sep=r'\s*;\s*|\t|,', engine='python')
        for _, row in df_a.iterrows():
            vid_path = os.path.join(DB_ROOT, f"PartA/video/{row['subject_name']}/{row['sample_name']}.mp4")
            bio_f = os.path.join(DB_ROOT, f"PartA/biosignals_filtered/{row['subject_name']}/{row['sample_name']}_bio.csv")
            bio_r = os.path.join(DB_ROOT, f"PartA/biosignals_raw/{row['subject_name']}/{row['sample_name']}_bio.csv")
            records.append({
                "dataset_part": "PartA",
                "subject_id": int(row["subject_id"]),
                "subject_name": str(row["subject_name"]),
                "sample_name": str(row["sample_name"]),
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "task": "pain",
                "video_path": vid_path if os.path.exists(vid_path) else "",
                "bio_filtered_path": bio_f if os.path.exists(bio_f) else "",
                "bio_raw_path": bio_r if os.path.exists(bio_r) else "",
                "duration_sec": 5.52,
                "fps": 25.0,
                "frame_count": 138,
                "bio_sampling_hz": 512,
                "channels": "gsr,ecg,emg_trapezius,emg_corrugator,emg_zygomaticus",
            })

    # Part B
    part_b_samples = os.path.join(DB_ROOT, "PartB/starting_point/samples.csv")
    if os.path.exists(part_b_samples):
        df_b = pd.read_csv(part_b_samples, sep=r'\s*;\s*|\t|,', engine='python')
        for _, row in df_b.iterrows():
            vid_path = os.path.join(DB_ROOT, f"PartB/video/{row['subject_name']}/{row['sample_name']}.mp4")
            bio_f = os.path.join(DB_ROOT, f"PartB/biosignals_filtered/{row['subject_name']}/{row['sample_name']}_bio.csv")
            bio_r = os.path.join(DB_ROOT, f"PartB/biosignals_raw/{row['subject_name']}/{row['sample_name']}_bio.csv")
            records.append({
                "dataset_part": "PartB",
                "subject_id": int(row["subject_id"]),
                "subject_name": str(row["subject_name"]),
                "sample_name": str(row["sample_name"]),
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "task": "pain",
                "video_path": vid_path if os.path.exists(vid_path) else "",
                "bio_filtered_path": bio_f if os.path.exists(bio_f) else "",
                "bio_raw_path": bio_r if os.path.exists(bio_r) else "",
                "duration_sec": 5.52,
                "fps": 25.0,
                "frame_count": 138,
                "bio_sampling_hz": 512,
                "channels": "gsr,ecg,emg_trapezius,emg_corrugator,emg_zygomaticus",
            })

    # Part C
    part_c_samples = os.path.join(DB_ROOT, "PartC/starting_point/samples.csv")
    if os.path.exists(part_c_samples):
        df_c = pd.read_csv(part_c_samples, sep=r'\s*;\s*|\t|,', engine='python')
        for _, row in df_c.iterrows():
            vid_path = os.path.join(DB_ROOT, f"PartC/video/{row['subject_name']}.mp4")
            bio_r = os.path.join(DB_ROOT, f"PartC/biosignals_raw/{row['subject_name']}.csv")
            stim_path = os.path.join(DB_ROOT, f"PartC/stimulus/{row['subject_name']}.csv")
            lbl_path = os.path.join(DB_ROOT, f"PartC/label/{row['subject_name']}.csv")
            records.append({
                "dataset_part": "PartC",
                "subject_id": int(row["subject_id"]),
                "subject_name": str(row["subject_name"]),
                "sample_name": str(row["subject_name"]),
                "class_id": -1,
                "class_name": "continuous_pain",
                "task": "pain_continuous",
                "video_path": vid_path if os.path.exists(vid_path) else "",
                "bio_filtered_path": "",
                "bio_raw_path": bio_r if os.path.exists(bio_r) else "",
                "stimulus_path": stim_path if os.path.exists(stim_path) else "",
                "label_path": lbl_path if os.path.exists(lbl_path) else "",
                "duration_sec": 1517.8,
                "fps": 25.0,
                "frame_count": 37945,
                "bio_sampling_hz": 512,
                "channels": "gsr,ecg,emg_trapezius,emg_corrugator,emg_zygomaticus",
            })

    # Part D
    part_d_samples = os.path.join(DB_ROOT, "PartD/starting_point/samples.csv")
    d_label_map = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "pain", 6: "sad"}
    if os.path.exists(part_d_samples):
        df_d = pd.read_csv(part_d_samples, sep=r'\s*;\s*|\t|,', engine='python')
        for _, row in df_d.iterrows():
            vid_path = os.path.join(DB_ROOT, f"PartD/video/{row['sample_name']}.mp4")
            bio_r = os.path.join(DB_ROOT, f"PartD/biosignals_raw/{row['sample_name']}_bio.csv")
            cid = int(row["class_id"])
            records.append({
                "dataset_part": "PartD",
                "subject_id": int(row["subject_id"]),
                "subject_name": str(row["subject_name"]),
                "sample_name": str(row["sample_name"]),
                "class_id": cid,
                "class_name": d_label_map.get(cid, str(cid)),
                "task": "emotion",
                "video_path": vid_path if os.path.exists(vid_path) else "",
                "bio_filtered_path": "",
                "bio_raw_path": bio_r if os.path.exists(bio_r) else "",
                "duration_sec": 60.04,
                "fps": 25.0,
                "frame_count": 1501,
                "bio_sampling_hz": 512,
                "channels": "gsr,ecg,emg_trapezius,emg_corrugator,emg_zygomaticus",
            })

    # Part E
    vids_e = sorted(glob.glob(os.path.join(DB_ROOT, "PartE - BioVidEmoDB/videos_frontal/*.mp4")))
    e_label_map = {"amusement": 0, "anger": 1, "disgust": 2, "fear": 3, "sad": 4}
    for idx, v in enumerate(vids_e):
        fname = os.path.basename(v).replace(".mp4", "")
        parts = fname.split("-")
        emo = parts[-1]
        subj = parse_part_e_subject(v)
        bio_r = os.path.join(DB_ROOT, f"PartE - BioVidEmoDB/bio_raw/{fname}_bio.csv")
        records.append({
            "dataset_part": "PartE",
            "subject_id": idx // 5 + 1,
            "subject_name": subj,
            "sample_name": fname,
            "class_id": e_label_map.get(emo, -1),
            "class_name": emo,
            "task": "emotion",
            "video_path": v,
            "bio_filtered_path": "",
            "bio_raw_path": bio_r if os.path.exists(bio_r) else "",
            "duration_sec": 114.04,
            "fps": 25.0,
            "frame_count": 2851,
            "bio_sampling_hz": 512,
            "channels": "scl,ecg,emg_trap",
        })

    inv_df = pd.DataFrame(records)
    csv_path = os.path.join(SHARED_INV_DIR, "dataset_inventory.csv")
    json_path = os.path.join(SHARED_INV_DIR, "dataset_inventory.json")
    inv_df.to_csv(csv_path, index=False)
    
    # Summary dictionary
    summary = {
        "total_records": len(inv_df),
        "parts": inv_df["dataset_part"].value_counts().to_dict(),
        "tasks": inv_df["task"].value_counts().to_dict(),
        "subjects_per_part": inv_df.groupby("dataset_part")["subject_name"].nunique().to_dict(),
        "video_verified": int((inv_df["video_path"] != "").sum()),
        "bio_raw_verified": int((inv_df["bio_raw_path"] != "").sum()),
        "bio_filtered_verified": int((inv_df["bio_filtered_path"] != "").sum()),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Inventory saved successfully.")
    print("Summary:", json.dumps(summary, indent=2))
    return inv_df

def plot_dataset_statistics(df):
    print("Plotting scientific dataset statistics...")
    # 1. Samples per Dataset Part
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["dataset_part"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_r")
    ax.set_title("BioVid Dataset: Sample Distribution Across Parts A–E", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples / Recordings", fontsize=12)
    ax.set_xlabel("Dataset Part", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(SHARED_INV_DIR, "dataset_sample_distribution.png"), dpi=300)
    plt.close(fig)

    # 2. Pain Class Distribution (Parts A & B)
    pain_df = df[df["dataset_part"].isin(["PartA", "PartB"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=pain_df, x="class_name", hue="dataset_part", ax=ax, palette=["#2b5c8f", "#d95f02"])
    ax.set_title("Part 1: Pain Intensity Class Distribution (Part A vs Part B)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pain Intensity Class", fontsize=12)
    ax.set_ylabel("Trial Count", fontsize=12)
    ax.legend(title="Dataset Part")
    plt.tight_layout()
    fig.savefig(os.path.join(SHARED_INV_DIR, "pain_class_distribution.png"), dpi=300)
    plt.close(fig)

    # 3. Emotion Category Distribution (Part D & Part E)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    df_d = df[df["dataset_part"] == "PartD"]
    sns.countplot(data=df_d, x="class_name", ax=ax1, order=["anger", "disgust", "fear", "happy", "neutral", "pain", "sad"], palette="viridis")
    ax1.set_title("Part D: Discrete Emotion Distribution (7 Classes)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Emotion Class", fontsize=11)
    ax1.set_ylabel("Clip Count", fontsize=11)
    ax1.tick_params(axis='x', rotation=30)

    df_e = df[df["dataset_part"] == "PartE"]
    sns.countplot(data=df_e, x="class_name", ax=ax2, order=["amusement", "anger", "disgust", "fear", "sad"], palette="magma")
    ax2.set_title("Part E (BioVidEmoDB): Emotion Distribution (5 Classes)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Emotion Class", fontsize=11)
    ax2.set_ylabel("Clip Count", fontsize=11)
    ax2.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(SHARED_INV_DIR, "emotion_class_distribution.png"), dpi=300)
    plt.close(fig)

    # 4. Modality Availability Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    mods = {
        "Video Only": (df["video_path"] != "").sum() - (df["bio_raw_path"] != "").sum(),
        "Video + Raw Bio": ((df["video_path"] != "") & (df["bio_raw_path"] != "")).sum(),
        "Video + Filtered Bio": ((df["video_path"] != "") & (df["bio_filtered_path"] != "")).sum(),
    }
    ax.bar(mods.keys(), mods.values(), color=["#4e79a7", "#59a14f", "#edc948"])
    ax.set_title("Modality Availability Across Entire BioVid DB", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(SHARED_INV_DIR, "modality_availability.png"), dpi=300)
    plt.close(fig)

    # Copy relevant graphs into respective dataset_analysis folders
    for dest in ["BioVid-DL/Part_1_Pain_Recognition/dataset_analysis",
                 "BioVid-DL/Part_2_Emotion_Recognition/dataset_analysis",
                 "BioVid-DL/Part_3_Multimodal_Recognition/dataset_analysis"]:
        os.makedirs(dest, exist_ok=True)
        os.system(f"cp {SHARED_INV_DIR}/*.png {dest}/")

    print("All dataset statistics and graphs created and distributed.")

if __name__ == "__main__":
    df = build_inventory()
    plot_dataset_statistics(df)
