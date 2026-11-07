import os
import glob
import cv2
import json
import time
import numpy as np
import pandas as pd
from scipy import signal

CACHE_DIR = "BioVid-DL/shared/cache"
os.makedirs(f"{CACHE_DIR}/part1_pain", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/part2_emotion", exist_ok=True)
os.makedirs(f"{CACHE_DIR}/part3_multimodal", exist_ok=True)

class BioSignalProcessor:
    def __init__(self, fs=512):
        self.fs = fs
        self.b_gsr, self.a_gsr = signal.butter(4, 5.0 / (fs / 2.0), btype='low')
        self.b_ecg, self.a_ecg = signal.butter(4, [0.5 / (fs / 2.0), 45.0 / (fs / 2.0)], btype='bandpass')
        self.b_emg, self.a_emg = signal.butter(4, [20.0 / (fs / 2.0), 240.0 / (fs / 2.0)], btype='bandpass')
        self.b_notch, self.a_notch = signal.iirnotch(50.0 / (fs / 2.0), 30.0)

    def process(self, df_bio, target_len=2816):
        channels = []
        # GSR
        g_col = 'gsr' if 'gsr' in df_bio.columns else ('scl ' if 'scl ' in df_bio.columns else None)
        if g_col:
            g = signal.filtfilt(self.b_gsr, self.a_gsr, df_bio[g_col].values[:target_len])
        else:
            g = np.zeros(target_len, dtype=np.float32)
        channels.append(g)

        # ECG
        e_col = 'ecg' if 'ecg' in df_bio.columns else (' ecg ' if ' ecg ' in df_bio.columns else None)
        if e_col:
            e = signal.filtfilt(self.b_ecg, self.a_ecg, df_bio[e_col].values[:target_len])
        else:
            e = np.zeros(target_len, dtype=np.float32)
        channels.append(e)

        # EMG
        m_col = 'emg_trapezius' if 'emg_trapezius' in df_bio.columns else (' emg_trap' if ' emg_trap' in df_bio.columns else None)
        if m_col:
            bp = signal.filtfilt(self.b_emg, self.a_emg, df_bio[m_col].values[:target_len])
            notch = signal.filtfilt(self.b_notch, self.a_notch, bp)
            m = np.abs(notch)
        else:
            m = np.zeros(target_len, dtype=np.float32)
        channels.append(m)

        arr = np.stack(channels, axis=0) # [3, target_len]
        # Pad or trim if needed
        if arr.shape[1] < target_len:
            pad = np.zeros((3, target_len - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        arr = arr[:, :target_len]

        # Standardize
        mean = np.mean(arr, axis=-1, keepdims=True)
        std = np.std(arr, axis=-1, keepdims=True) + 1e-6
        return ((arr - mean) / std).astype(np.float32)

def extract_uniform_frames(vid_path, start_frame=0, total_clip_frames=138, num_frames=8, target_size=224):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return np.zeros((num_frames, target_size, target_size, 3), dtype=np.uint8)

    target_indices = np.linspace(start_frame, start_frame + total_clip_frames - 1, num_frames, dtype=int)
    frames_dict = {}

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cur_idx = start_frame
    target_set = set(target_indices)
    while cap.isOpened() and len(frames_dict) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if cur_idx in target_set:
            h, w, _ = frame.shape
            crop_size = min(h, w)
            sx = (w - crop_size) // 2
            sy = (h - crop_size) // 2
            cropped = frame[sy:sy + crop_size, sx:sx + crop_size]
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames_dict[cur_idx] = rgb # uint8 [224, 224, 3]
        cur_idx += 1
        if cur_idx > target_indices[-1]:
            break

    cap.release()
    sorted_keys = sorted(frames_dict.keys())
    out = [frames_dict[k] for k in sorted_keys]
    while len(out) < num_frames:
        if len(out) > 0:
            out.append(out[-1])
        else:
            out.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
    return np.stack(out[:num_frames], axis=0) # [8, 224, 224, 3] uint8

def build_part1_cache():
    print("Building Part 1 Pain Recognition cache (Parts A, B, C)...")
    inv_df = pd.read_csv("BioVid-DL/shared/subject_splits/inventory_with_splits.csv")
    part_c_df = pd.read_csv("BioVid-DL/Part_1_Pain_Recognition/preprocessing/part_c_segmented_manifest.csv")

    with open("BioVid-DL/shared/subject_splits/subject_splits.json") as f:
        splits_info = json.load(f)
    train_subs = set(splits_info["train_subjects"])
    val_subs = set(splits_info["val_subjects"])

    def assign_split(s):
        canon = s[4:] if len(s) >= 11 and "_" in s else s
        if canon in train_subs:
            return "train"
        elif canon in val_subs:
            return "val"
        else:
            return "test"

    part_c_df["split"] = part_c_df["subject_name"].apply(assign_split)
    pain_inv = inv_df[inv_df["dataset_part"].isin(["PartA", "PartB"])].copy()
    pain_inv["start_frame"] = 0
    cols = ["dataset_part", "subject_name", "sample_name", "class_id", "video_path", "start_frame", "split", "bio_raw_path"]
    pain_inv["bio_raw_path"] = pain_inv["bio_raw_path"]
    part_c_sub = part_c_df.rename(columns={"trial_id": "sample_name", "bio_path": "bio_raw_path"})[cols]
    combined = pd.concat([pain_inv[cols], part_c_sub], ignore_index=True)

    # Sample balanced dataset for training, validation, testing
    selected_records = []
    # Train: 100 per class (500 total)
    train_df = combined[combined["split"] == "train"]
    for cid in range(5):
        selected_records.append(train_df[train_df["class_id"] == cid].sample(n=100, random_state=42))

    # Val: 30 per class (150 total)
    val_df = combined[combined["split"] == "val"]
    for cid in range(5):
        selected_records.append(val_df[val_df["class_id"] == cid].sample(n=30, random_state=42))

    # Test: 40 per class (200 total)
    test_df = combined[combined["split"] == "test"]
    for cid in range(5):
        selected_records.append(test_df[test_df["class_id"] == cid].sample(n=40, random_state=42))

    final_df = pd.concat(selected_records, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Selected {len(final_df)} trials for Part 1 & Part 3 (Train: 500, Val: 150, Test: 200).")

    proc = BioSignalProcessor()
    manifest_rows = []
    t0 = time.time()

    for idx, row in final_df.iterrows():
        vid_p = str(row["video_path"])
        bio_p = str(row["bio_raw_path"])
        cid = int(row["class_id"])
        split = str(row["split"])
        sname = str(row["sample_name"])
        part = str(row["dataset_part"])
        start_f = int(row["start_frame"])

        cache_name = f"{part}_{split}_{sname}.npz"
        cache_path = os.path.join(f"{CACHE_DIR}/part1_pain", cache_name)

        if not os.path.exists(cache_path):
            frames = extract_uniform_frames(vid_p, start_frame=start_f, total_clip_frames=138, num_frames=8)

            # Also process biosignals for multimodal part
            bio_arr = np.zeros((3, 2816), dtype=np.float32)
            if os.path.exists(bio_p):
                try:
                    b_df = pd.read_csv(bio_p, sep=r'\s*;\s*|\t|,', engine='python', nrows=3500)
                    bio_arr = proc.process(b_df, target_len=2816)
                except Exception:
                    pass

            np.savez_compressed(
                cache_path,
                frames=frames,
                bio=bio_arr,
                label=cid,
                split=split,
                dataset_part=part,
                subject_name=str(row["subject_name"]),
                sample_name=sname
            )

        manifest_rows.append({
            "cache_path": cache_path,
            "split": split,
            "class_id": cid,
            "dataset_part": part,
            "subject_name": str(row["subject_name"]),
            "sample_name": sname,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(final_df):
            el = time.time() - t0
            print(f"Part 1 Cached [{idx+1}/{len(final_df)}] ({el:.1f}s)")

    pd.DataFrame(manifest_rows).to_csv(f"{CACHE_DIR}/part1_cached_manifest.csv", index=False)
    print("Part 1 Cache Complete.")

def build_part2_cache():
    print("Building Part 2 Emotion Recognition cache (Parts D & E)...")
    inv_df = pd.read_csv("BioVid-DL/shared/subject_splits/inventory_with_splits.csv")
    proc = BioSignalProcessor()

    # Part D: 7 classes (Train: 30/class=210, Val: 10/class=70, Test: 10/class=70 -> 350)
    df_d = inv_df[inv_df["dataset_part"] == "PartD"]
    d_recs = []
    for split, count in [("train", 30), ("val", 10), ("test", 10)]:
        s_df = df_d[df_d["split"] == split]
        for cid in range(7):
            sub = s_df[s_df["class_id"] == cid]
            n_take = min(len(sub), count)
            d_recs.append(sub.sample(n=n_take, random_state=42))
    part_d_final = pd.concat(d_recs, ignore_index=True)

    # Part E: 5 classes (Train: 30/class=150, Val: 10/class=50, Test: 10/class=50 -> 250)
    df_e = inv_df[inv_df["dataset_part"] == "PartE"]
    e_recs = []
    for split, count in [("train", 30), ("val", 10), ("test", 10)]:
        s_df = df_e[df_e["split"] == split]
        for cid in range(5):
            sub = s_df[s_df["class_id"] == cid]
            n_take = min(len(sub), count)
            e_recs.append(sub.sample(n=n_take, random_state=42))
    part_e_final = pd.concat(e_recs, ignore_index=True)

    emotion_df = pd.concat([part_d_final, part_e_final], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Selected {len(emotion_df)} clips for Part 2 Emotion (Part D: {len(part_d_final)}, Part E: {len(part_e_final)}).")

    manifest_rows = []
    t0 = time.time()

    for idx, row in emotion_df.iterrows():
        vid_p = str(row["video_path"])
        bio_p = str(row["bio_raw_path"])
        cid = int(row["class_id"])
        split = str(row["split"])
        sname = str(row["sample_name"])
        part = str(row["dataset_part"])
        n_frames_raw = int(row["frame_count"])

        cache_name = f"{part}_{split}_{sname}.npz"
        cache_path = os.path.join(f"{CACHE_DIR}/part2_emotion", cache_name)

        if not os.path.exists(cache_path):
            frames = extract_uniform_frames(vid_p, start_frame=0, total_clip_frames=n_frames_raw, num_frames=8)

            bio_arr = np.zeros((3, 2816), dtype=np.float32)
            if os.path.exists(bio_p):
                try:
                    b_df = pd.read_csv(bio_p, sep=r'\s*;\s*|\t|,', engine='python', nrows=3500)
                    bio_arr = proc.process(b_df, target_len=2816)
                except Exception:
                    pass

            np.savez_compressed(
                cache_path,
                frames=frames,
                bio=bio_arr,
                label=cid,
                split=split,
                dataset_part=part,
                subject_name=str(row["subject_name"]),
                sample_name=sname
            )

        manifest_rows.append({
            "cache_path": cache_path,
            "split": split,
            "class_id": cid,
            "dataset_part": part,
            "subject_name": str(row["subject_name"]),
            "sample_name": sname,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(emotion_df):
            el = time.time() - t0
            print(f"Part 2 Cached [{idx+1}/{len(emotion_df)}] ({el:.1f}s)")

    pd.DataFrame(manifest_rows).to_csv(f"{CACHE_DIR}/part2_cached_manifest.csv", index=False)
    print("Part 2 Cache Complete.")

if __name__ == "__main__":
    build_part1_cache()
    build_part2_cache()
