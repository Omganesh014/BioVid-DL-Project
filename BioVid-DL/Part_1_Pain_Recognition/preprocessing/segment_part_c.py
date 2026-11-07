import os
import glob
import pandas as pd
import numpy as np

DB_ROOT = "/run/media/acernitro/Ventoy/DB"
OUT_CSV = "BioVid-DL/Part_1_Pain_Recognition/preprocessing/part_c_segmented_manifest.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def generate_part_c_segments():
    print("Segmenting Part C continuous recordings into stimulus-aligned 5.5s trials...")
    stim_files = sorted(glob.glob(os.path.join(DB_ROOT, "PartC/stimulus/*.csv")))
    records = []

    for s_file in stim_files:
        subject_name = os.path.basename(s_file).replace(".csv", "")
        vid_path = os.path.join(DB_ROOT, f"PartC/video/{subject_name}.mp4")
        bio_path = os.path.join(DB_ROOT, f"PartC/biosignals_raw/{subject_name}.csv")

        if not os.path.exists(vid_path) or not os.path.exists(bio_path):
            continue

        df_stim = pd.read_csv(s_file, sep=r'\s*;\s*|\t|,', engine='python')

        # 1. Pain Stimulus Trials (80 per subject: 20 each of PA1, PA2, PA3, PA4)
        for trial_idx, row in df_stim.iterrows():
            t_sec = float(row["time"]) / 1e6
            lbl = int(row["label"])
            start_frame = int(t_sec * 25.0)
            records.append({
                "dataset_part": "PartC",
                "subject_name": subject_name,
                "trial_id": f"pain_{trial_idx+1:03d}",
                "start_time_sec": t_sec,
                "duration_sec": 5.52,
                "start_frame": start_frame,
                "num_frames": 138,
                "class_id": lbl,
                "class_name": f"PA{lbl}",
                "video_path": vid_path,
                "bio_path": bio_path,
                "stimulus_onset_us": int(row["time"]),
            })

        # 2. Baseline Trials (20 per subject: BL1, class 0)
        # Sample baseline intervals before first stimulus and between stimuli during calm phases
        first_stim_sec = float(df_stim.iloc[0]["time"]) / 1e6
        # Pre-stimulus baseline
        records.append({
            "dataset_part": "PartC",
            "subject_name": subject_name,
            "trial_id": "bl1_000",
            "start_time_sec": max(2.0, first_stim_sec - 10.0),
            "duration_sec": 5.52,
            "start_frame": int(max(2.0, first_stim_sec - 10.0) * 25.0),
            "num_frames": 138,
            "class_id": 0,
            "class_name": "BL1",
            "video_path": vid_path,
            "bio_path": bio_path,
            "stimulus_onset_us": int(max(2.0, first_stim_sec - 10.0) * 1e6),
        })

        # Inter-stimulus baseline intervals (sample 19 more evenly across the 79 inter-stimulus gaps)
        gap_indices = np.linspace(0, len(df_stim) - 2, 19, dtype=int)
        for b_i, g_idx in enumerate(gap_indices, start=1):
            t_curr = float(df_stim.iloc[g_idx]["time"]) / 1e6
            t_next = float(df_stim.iloc[g_idx + 1]["time"]) / 1e6
            # Sample in the middle of recovery period before next stimulus onset (e.g. 8s after current onset)
            bl_onset = t_curr + 8.5
            if bl_onset + 5.52 < t_next:
                start_f = int(bl_onset * 25.0)
            else:
                bl_onset = t_next - 6.0
                start_f = int(bl_onset * 25.0)

            records.append({
                "dataset_part": "PartC",
                "subject_name": subject_name,
                "trial_id": f"bl1_{b_i:03d}",
                "start_time_sec": bl_onset,
                "duration_sec": 5.52,
                "start_frame": start_f,
                "num_frames": 138,
                "class_id": 0,
                "class_name": "BL1",
                "video_path": vid_path,
                "bio_path": bio_path,
                "stimulus_onset_us": int(bl_onset * 1e6),
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Part C segmented manifest saved to {OUT_CSV}")
    print(f"Total trials created: {len(df_out)}")
    print("Class distribution:\n", df_out["class_name"].value_counts())

if __name__ == "__main__":
    generate_part_c_segments()
