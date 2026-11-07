import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PART_D_CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "pain", "sad"]
PART_E_CLASSES = ["amusement", "anger", "disgust", "fear", "sad"]

class BioVidEmotionDataset(Dataset):
    """
    Video Dataset for Emotion Recognition on BioVid Part D (7 classes) and Part E (5 classes).
    Extracts uniformly sampled frames across full recording duration with center square cropping.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_part: str = "PartD",
        num_frames: int = 8,
        target_size: int = 224,
    ):
        super().__init__()
        self.df = df[df["dataset_part"] == dataset_part].reset_index(drop=True)
        self.dataset_part = dataset_part
        self.num_frames = num_frames
        self.target_size = target_size
        self.class_names = PART_D_CLASSES if dataset_part == "PartD" else PART_E_CLASSES

    def __len__(self):
        return len(self.df)

    def _read_frames(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return np.zeros((self.num_frames, 3, self.target_size, self.target_size), dtype=np.float32)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            total_frames = max(total_frames, self.num_frames)

        target_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        target_set = set(target_indices)
        frames_dict = {}

        cur_idx = 0
        while cap.isOpened() and len(frames_dict) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx in target_set:
                h, w, _ = frame.shape
                crop_size = min(h, w)
                sx = (w - crop_size) // 2
                sy = (h - crop_size) // 2
                cropped = frame[sy:sy + crop_size, sx:sx + crop_size]
                resized = cv2.resize(cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                norm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                frames_dict[cur_idx] = norm.transpose(2, 0, 1)
            cur_idx += 1
            if cur_idx > target_indices[-1]:
                break

        cap.release()

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

        frames = self._read_frames(vid_path)
        tensor = torch.from_numpy(frames).float() # [T, C, H, W]
        label_tensor = torch.tensor(label, dtype=torch.long)

        meta = {
            "dataset_part": self.dataset_part,
            "subject_name": str(row.get("subject_name", "")),
            "sample_name": str(row.get("sample_name", "")),
            "class_id": label,
            "class_name": self.class_names[label] if label < len(self.class_names) else str(label),
        }
        return tensor, label_tensor, meta
