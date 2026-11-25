import os
import time
import cv2
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split

INVENTORY_CSV = "BioVid-DL/shared/dataset_inventory/dataset_inventory.csv"
OUT_ROOT = "BioVid-DL/shared/cache/retrain_56ep"
for sub in ["part_a", "part_d", "part_e"]:
    os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)


class BioProcessor:
    def __init__(self, fs: int = 512):
        self.b_gsr, self.a_gsr = signal.butter(4, 5.0 / (fs / 2.0), btype="low")
        self.b_ecg, self.a_ecg = signal.butter(4, [0.5 / (fs / 2.0), 45.0 / (fs / 2.0)], btype="bandpass")
        self.b_emg, self.a_emg = signal.butter(4, [20.0 / (fs / 2.0), 240.0 / (fs / 2.0)], btype="bandpass")
        self.b_notch, self.a_notch = signal.iirnotch(50.0 / (fs / 2.0), 30.0)

    def _safe_filt(self, b, a, x):
        if len(x) < 32:
            return x.astype(np.float32)
        return signal.filtfilt(b, a, x).astype(np.float32)

    def process_raw_or_filtered(self, arr_5ch: np.ndarray, target_len: int = 2816, is_filtered: bool = False) -> np.ndarray:
        """arr_5ch: [N, 5] -> returns [5, target_len] standardized."""
        arr_5ch = np.nan_to_num(arr_5ch[:target_len].astype(np.float32))
        if len(arr_5ch) < target_len:
            pad = np.zeros((target_len - len(arr_5ch), 5), dtype=np.float32)
            arr_5ch = np.vstack([arr_5ch, pad])
        out = np.zeros((5, target_len), dtype=np.float32)
        if is_filtered:
            for c in range(5):
                out[c] = arr_5ch[:, c]
        else:
            out[0] = self._safe_filt(self.b_gsr, self.a_gsr, arr_5ch[:, 0])
            out[1] = self._safe_filt(self.b_ecg, self.a_ecg, arr_5ch[:, 1])
            for c in [2, 3, 4]:
                bp = self._safe_filt(self.b_emg, self.a_emg, arr_5ch[:, c])
                nt = self._safe_filt(self.b_notch, self.a_notch, bp)
                out[c] = np.abs(nt)
        mean = out.mean(axis=1, keepdims=True)
        std = out.std(axis=1, keepdims=True) + 1e-6
        return ((out - mean) / std).astype(np.float32)


def encode_frames_jpg(frames_rgb: list) -> np.ndarray:
    encoded = []
    for rgb in frames_rgb:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        encoded.append(buf.tobytes() if ok else b"")
    return np.array(encoded, dtype=object)


def crop_resize_rgb(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    h, w, _ = frame_bgr.shape
    c = min(h, w)
    sx, sy = (w - c) // 2, (h - c) // 2
    cropped = frame_bgr[sy:sy + c, sx:sx + c]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def load_bio_part_ad(path: str, proc: BioProcessor, is_filtered: bool, slice_range=None) -> np.ndarray:
    if not path or not os.path.exists(path):
        return np.zeros((5, 2816), dtype=np.float32)
    try:
        df_b = pd.read_csv(path, sep="\t")
        cols = ["gsr", "ecg", "emg_trapezius", "emg_corrugator", "emg_zygomaticus"]
        arr = np.zeros((len(df_b), 5), dtype=np.float32)
        for idx, col in enumerate(cols):
            if col in df_b.columns:
                arr[:, idx] = df_b[col].values
        if slice_range is not None:
            s0, s1 = slice_range
            arr = arr[s0:s1]
        return proc.process_raw_or_filtered(arr, target_len=2816, is_filtered=is_filtered)
    except Exception:
        return np.zeros((5, 2816), dtype=np.float32)


def load_bio_part_e(path: str, proc: BioProcessor, slice_range=None) -> np.ndarray:
    if not path or not os.path.exists(path):
        return np.zeros((5, 2816), dtype=np.float32)
    try:
        df_b = pd.read_csv(path, sep=",")
        raw = np.nan_to_num(df_b.values[:, :3].astype(np.float32))
        arr = np.zeros((len(raw), 5), dtype=np.float32)
        arr[:, :3] = raw
        if slice_range is not None:
            s0, s1 = slice_range
            arr = arr[s0:s1]
        return proc.process_raw_or_filtered(arr, target_len=2816, is_filtered=False)
    except Exception:
        return np.zeros((5, 2816), dtype=np.float32)


def cache_part_a(inv: pd.DataFrame, proc: BioProcessor):
    print("\n=== Caching Part A (BL1 vs PA4, 3,480 trials) ===", flush=True)
    pa = inv[(inv["dataset_part"] == "PartA") & (inv["class_id"].isin([0, 4]))].copy().reset_index(drop=True)
    pa["bin_label"] = (pa["class_id"] == 4).astype(int)
    pa["strat_key"] = pa["subject_name"].astype(str) + "_" + pa["bin_label"].astype(str)

    # 70 / 15 / 15 stratified split
    tr_idx, tmp_idx = train_test_split(pa.index, test_size=0.30, random_state=42, stratify=pa["strat_key"])
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=42, stratify=pa.loc[tmp_idx, "strat_key"])
    pa.loc[tr_idx, "split"] = "train"
    pa.loc[val_idx, "split"] = "val"
    pa.loc[te_idx, "split"] = "test"

    rows = []
    t0 = time.time()
    for i, row in pa.iterrows():
        sname = str(row["sample_name"])
        split = str(row["split"])
        cid = int(row["bin_label"])
        cpath = os.path.join(OUT_ROOT, "part_a", f"{sname}.npz")

        if not os.path.exists(cpath):
            cap = cv2.VideoCapture(str(row["video_path"]))
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 138
            idxs = np.linspace(0, max(total_f - 1, 7), 8, dtype=int)
            idx_set = set(idxs.tolist())
            grabbed = {}
            cur = 0
            max_i = int(idxs[-1])
            while cap.isOpened() and cur <= max_i:
                ret, f = cap.read()
                if not ret:
                    break
                if cur in idx_set:
                    grabbed[cur] = crop_resize_rgb(f, 224)
                cur += 1
            cap.release()
            frames = [grabbed.get(int(x), np.zeros((224, 224, 3), dtype=np.uint8)) for x in idxs]
            jpgs = encode_frames_jpg(frames)
            bio_path = str(row["bio_filtered_path"]) if pd.notna(row["bio_filtered_path"]) else str(row["bio_raw_path"])
            bio = load_bio_part_ad(bio_path, proc, is_filtered=pd.notna(row["bio_filtered_path"]))
            np.savez(cpath, jpgs=jpgs, bio=bio, label=cid)

        rows.append({
            "cache_path": cpath,
            "split": split,
            "class_id": cid,
            "orig_class_name": str(row["class_name"]),
            "dataset_part": "PartA",
            "subject_name": str(row["subject_name"]),
            "sample_name": sname,
        })
        if (i + 1) % 300 == 0 or (i + 1) == len(pa):
            print(f"  PartA [{i+1}/{len(pa)}] elapsed={time.time()-t0:.1f}s", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT, "part_a_manifest.csv"), index=False)
    print(f"PartA manifest saved ({len(rows)} rows).", flush=True)


def cache_long_emotion(inv: pd.DataFrame, proc: BioProcessor, part_name: str, sub_dir: str, out_csv: str, num_segs: int = 8):
    print(f"\n=== Caching {part_name} ({num_segs} segments/video) ===", flush=True)
    sub = inv[inv["dataset_part"] == part_name].copy().reset_index(drop=True)
    rows = []
    t0 = time.time()
    rng = np.random.RandomState(42)

    for i, row in sub.iterrows():
        sname = str(row["sample_name"])
        cid = int(row["class_id"])
        subj = str(row["subject_name"])

        # Deterministic 6 train / 1 val / 1 test segment assignment per 8-segment video
        perm = rng.permutation(num_segs)
        seg_splits = {}
        for pos, seg_i in enumerate(perm):
            if pos < num_segs - 2:
                seg_splits[int(seg_i)] = "train"
            elif pos == num_segs - 2:
                seg_splits[int(seg_i)] = "val"
            else:
                seg_splits[int(seg_i)] = "test"

        # Check if all segments already cached
        seg_paths = [os.path.join(OUT_ROOT, sub_dir, f"{sname}_seg{s}.npz") for s in range(num_segs)]
        if not all(os.path.exists(p) for p in seg_paths):
            cap = cv2.VideoCapture(str(row["video_path"]))
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(row["frame_count"])
            seg_bounds = np.linspace(0, max(total_f - 1, num_segs * 8), num_segs + 1, dtype=int)

            # Read raw bio once
            bio_path = str(row["bio_raw_path"]) if pd.notna(row["bio_raw_path"]) else ""
            if part_name == "PartD" and bio_path and os.path.exists(bio_path):
                df_b = pd.read_csv(bio_path, sep="\t")
                b_len = len(df_b)
            elif part_name == "PartE" and bio_path and os.path.exists(bio_path):
                df_b = pd.read_csv(bio_path, sep=",")
                b_len = len(df_b)
            else:
                df_b = None
                b_len = 0

            for s in range(num_segs):
                cpath = seg_paths[s]
                if os.path.exists(cpath):
                    continue
                seg_idxs = np.linspace(seg_bounds[s], max(seg_bounds[s], seg_bounds[s + 1] - 1), 8, dtype=int)
                frames = []
                for idx in seg_idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, f = cap.read()
                    if ret:
                        frames.append(crop_resize_rgb(f, 224))
                    else:
                        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
                jpgs = encode_frames_jpg(frames)

                if df_b is not None and b_len > 0:
                    b0 = int(s * b_len / num_segs)
                    b1 = int((s + 1) * b_len / num_segs)
                    if part_name == "PartD":
                        cols = ["gsr", "ecg", "emg_trapezius", "emg_corrugator", "emg_zygomaticus"]
                        arr = np.zeros((max(b1 - b0, 1), 5), dtype=np.float32)
                        for idx_c, col in enumerate(cols):
                            if col in df_b.columns:
                                arr[:, idx_c] = df_b[col].values[b0:b1]
                        bio = proc.process_raw_or_filtered(arr, target_len=2816, is_filtered=False)
                    else:
                        raw = np.nan_to_num(df_b.values[b0:b1, :3].astype(np.float32))
                        arr = np.zeros((len(raw), 5), dtype=np.float32)
                        arr[:, :3] = raw
                        bio = proc.process_raw_or_filtered(arr, target_len=2816, is_filtered=False)
                else:
                    bio = np.zeros((5, 2816), dtype=np.float32)

                np.savez(cpath, jpgs=jpgs, bio=bio, label=cid)
            cap.release()

        for s in range(num_segs):
            rows.append({
                "cache_path": seg_paths[s],
                "split": seg_splits[s],
                "class_id": cid,
                "class_name": str(row["class_name"]),
                "dataset_part": part_name,
                "subject_name": subj,
                "sample_name": f"{sname}_seg{s}",
            })
        if (i + 1) % 50 == 0 or (i + 1) == len(sub):
            print(f"  {part_name} [{i+1}/{len(sub)} videos -> {len(rows)} clips] elapsed={time.time()-t0:.1f}s", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT, out_csv), index=False)
    print(f"{part_name} manifest saved ({len(rows)} rows).", flush=True)


if __name__ == "__main__":
    inv = pd.read_csv(INVENTORY_CSV)
    proc = BioProcessor()
    cache_part_a(inv, proc)
    cache_long_emotion(inv, proc, "PartE", "part_e", "part_e_manifest.csv", num_segs=8)
    cache_long_emotion(inv, proc, "PartD", "part_d", "part_d_manifest.csv", num_segs=8)
    print("\nALL CACHING COMPLETE!", flush=True)
