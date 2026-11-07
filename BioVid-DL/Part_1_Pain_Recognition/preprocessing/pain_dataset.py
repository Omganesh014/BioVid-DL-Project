import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class BioVidPainDataset(Dataset):
    """
    Video Dataset for Pain Intensity Recognition (Parts A, B, C).
    Extracts uniformly sampled frames with center cropping and ImageNet normalization.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        num_frames: int = 8,
        target_size: int = 224,
        max_samples: int = None,
    ):
        super().__init__()
        if max_samples and len(df) > max_samples:
            self.df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def _read_frames(self, vid_path, start_frame=0, total_clip_frames=138):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            # Return blank tensor if corrupt
            return np.zeros((self.num_frames, 3, self.target_size, self.target_size), dtype=np.float32)

        # Uniform frame indices within [start_frame, start_frame + total_clip_frames]
        target_indices = np.linspace(start_frame, start_frame + total_clip_frames - 1, self.num_frames, dtype=int)
        frames_dict = {}

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        cur_idx = start_frame
        target_set = set(target_indices)
        while cap.isOpened() and len(frames_dict) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx in target_set:
                # Preprocess frame
                h, w, _ = frame.shape
                crop_size = min(h, w)
                sx = (w - crop_size) // 2
                sy = (h - crop_size) // 2
                cropped = frame[sy:sy + crop_size, sx:sx + crop_size]
                resized = cv2.resize(cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                norm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                frames_dict[cur_idx] = norm.transpose(2, 0, 1) # [3, H, W]
            cur_idx += 1
            if cur_idx > target_indices[-1]:
                break

        cap.release()

        # If any frames missed, pad with duplicates or zeros
        sorted_keys = sorted(frames_dict.keys())
        out_frames = [frames_dict[k] for k in sorted_keys]
        while len(out_frames) < self.num_frames:
            if len(out_frames) > 0:
                out_frames.append(out_frames[-1])
            else:
                out_frames.append(np.zeros((3, self.target_size, self.target_size), dtype=np.float32))

        return np.stack(out_frames[:self.num_frames], axis=0) # [T, C, H, W]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_path = str(row["video_path"])
        label = int(row["class_id"])
        start_frame = int(row.get("start_frame", 0))

        frames = self._read_frames(vid_path, start_frame=start_frame, total_clip_frames=138)
        tensor = torch.from_numpy(frames).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        meta = {
            "dataset_part": str(row.get("dataset_part", "PartA")),
            "subject_name": str(row.get("subject_name", "")),
            "sample_name": str(row.get("sample_name", row.get("trial_id", ""))),
            "class_id": label,
        }
        return tensor, label_tensor, meta
