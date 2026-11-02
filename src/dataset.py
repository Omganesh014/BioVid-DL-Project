import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import config


# Strict verified mapping from BioVid Part A metadata
LABEL_MAP: Dict[int, str] = {
    0: "BL1",  # Baseline (no pain)
    1: "PA1",  # Pain intensity 1
    2: "PA2",  # Pain intensity 2
    3: "PA3",  # Pain intensity 3
    4: "PA4",  # Pain intensity 4
}

CLASS_TO_ID: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}

# ImageNet normalization constants for RGB video
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_temporal_indices(total_frames: int, num_samples: int = 16) -> List[int]:
    """
    Uniform temporal sampling across the entire video duration.
    For total_frames=138 and num_samples=16:
    produces 16 evenly spaced indices covering the full 5.52s trial.
    """
    if total_frames < num_samples:
        raise ValueError(
            f"Video frame count ({total_frames}) is less than requested samples ({num_samples})!"
        )
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
    return indices


def preprocess_frame(
    frame: np.ndarray,
    target_size: int = 224,
    crop_mode: str = "center_square",
) -> np.ndarray:
    """
    Preprocess a raw video frame (H x W x C, BGR):
    1. Center square crop (1038x1038 from 1388x1038) to retain subject face/head perfectly
       without stretching, distortion, or background margin clutter.
    2. Resize to target_size x target_size (e.g. 224x224).
    3. Convert BGR to RGB.
    4. Normalize pixel values to [0, 1] then ImageNet standard score.
    Returns: float32 ndarray of shape (C, target_size, target_size)
    """
    h, w, c = frame.shape
    if crop_mode == "center_square":
        crop_size = min(h, w)
        sx = (w - crop_size) // 2
        sy = (h - crop_size) // 2
        cropped = frame[sy : sy + crop_size, sx : sx + crop_size]
    else:
        cropped = frame

    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize: (x/255 - mean) / std
    norm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose H x W x C -> C x H x W
    return norm.transpose(2, 0, 1)


def read_video_frames(
    video_path: Union[str, Path],
    num_frames: int = 16,
    target_size: int = 224,
    crop_mode: str = "center_square",
) -> np.ndarray:
    """
    Read sampled frames from a BioVid MP4 video file.
    Uses sequential reading to guarantee fast H.264 decoding without seek penalties.
    Returns: ndarray of shape (num_frames, 3, target_size, target_size)
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback if frame count metadata is missing: read sequentially
        total_frames = config.raw_frame_count

    target_indices = set(get_temporal_indices(total_frames, num_frames))
    frames_dict: Dict[int, np.ndarray] = {}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            processed = preprocess_frame(frame, target_size=target_size, crop_mode=crop_mode)
            frames_dict[frame_idx] = processed
            if len(frames_dict) == num_frames:
                break
        frame_idx += 1

    cap.release()

    if len(frames_dict) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames but decoded {len(frames_dict)} from {video_path}. "
            "Pipeline stops to prevent feeding incomplete or corrupted data."
        )

    # Sort frames in original chronological order
    sorted_indices = sorted(frames_dict.keys())
    ordered_frames = [frames_dict[i] for i in sorted_indices]
    return np.stack(ordered_frames, axis=0)  # [T, C, H, W]


class BioVidDataset(Dataset):
    """
    PyTorch Dataset for BioVid Heat Pain Database (Part A).
    Loads actual video files based on manifest DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_frames: int = 16,
        target_size: int = 224,
        crop_mode: str = "center_square",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.target_size = target_size
        self.crop_mode = crop_mode

        # Strict validation of labels
        for idx, label in enumerate(self.df["label"]):
            if label not in LABEL_MAP:
                raise ValueError(
                    f"Unexpected label {label} at row {idx}! Expected one of {list(LABEL_MAP.keys())}."
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int]]]:
        row = self.df.iloc[idx]
        vid_path = row["video_path"]
        label = int(row["label"])

        # Load video tensor [T, C, H, W]
        frames = read_video_frames(
            vid_path,
            num_frames=self.num_frames,
            target_size=self.target_size,
            crop_mode=self.crop_mode,
        )

        tensor = torch.from_numpy(frames).float()  # [T, C, H, W]
        label_tensor = torch.tensor(label, dtype=torch.long)

        meta = {
            "subject_id": int(row["subject_id"]),
            "subject_name": str(row["subject_name"]),
            "trial_id": int(row.get("trial_id", row.get("sample_id", 0))),
            "sample_name": str(row["sample_name"]),
            "video_path": str(vid_path),
        }

        return tensor, label_tensor, meta


def create_dataloader(
    df: pd.DataFrame,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    num_frames: int = 16,
    target_size: int = 224,
    crop_mode: str = "center_square",
) -> DataLoader:
    """
    Factory function for DataLoader.
    Batches will have shape:
      tensors: [B, T, C, H, W]
      labels:  [B]
    """
    dataset = BioVidDataset(
        df=df,
        num_frames=num_frames,
        target_size=target_size,
        crop_mode=crop_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
