import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

class BioVidCachedDataset(Dataset):
    """
    High-performance Cached Dataset for BioVid Pain and Emotion Experiments.
    Reads pre-extracted .npz records with instantaneous disk I/O,
    converting uint8 frames to normalized float32 tensors with on-the-fly regularization.
    """
    def __init__(self, manifest_csv: str, split: str = None, dataset_part: str = None, is_training: bool = False):
        super().__init__()
        df = pd.read_csv(manifest_csv)
        if split:
            df = df[df["split"] == split]
        if dataset_part:
            df = df[df["dataset_part"] == dataset_part]
        self.df = df.reset_index(drop=True)
        self.split = split
        self.is_training = is_training or (split == "train")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = row["cache_path"]

        data = np.load(cache_path)
        # frames: uint8 [T, 224, 224, 3]
        raw_frames = data["frames"] # [T, H, W, C]
        
        # On-the-fly Data Augmentation during training
        if self.is_training:
            # 1. Random horizontal flip
            if np.random.rand() > 0.5:
                raw_frames = np.flip(raw_frames, axis=2) # flip along width W

        # Transpose to [T, C, H, W]
        transposed = raw_frames.transpose(0, 3, 1, 2)
        
        # Normalize: (x/255.0 - mean) / std
        norm_frames = (transposed.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        if self.is_training:
            # 2. Subtle brightness/contrast jitter (+- 8%)
            jitter = np.random.uniform(0.92, 1.08)
            norm_frames = norm_frames * jitter

        vid_tensor = torch.from_numpy(norm_frames.copy()).float() # [T, 3, 224, 224]
        
        bio_arr = data["bio"] if "bio" in data else np.zeros((3, 2816), dtype=np.float32)
        if self.is_training and "bio" in data:
            # 3. Low-amplitude Gaussian noise to prevent signal memorization (SNR > 30dB)
            noise = np.random.normal(0, 0.015, bio_arr.shape).astype(np.float32)
            bio_arr = bio_arr + noise

        bio_tensor = torch.from_numpy(bio_arr).float() # [3, 2816]

        label = int(row["class_id"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        meta = {
            "dataset_part": str(row["dataset_part"]),
            "subject_name": str(row["subject_name"]),
            "sample_name": str(row["sample_name"]),
            "class_id": label,
        }
        return vid_tensor, bio_tensor, label_tensor, meta

def get_dataloaders(manifest_csv, batch_size=8, dataset_part=None, num_workers=2):
    train_ds = BioVidCachedDataset(manifest_csv, split="train", dataset_part=dataset_part, is_training=True)
    val_ds = BioVidCachedDataset(manifest_csv, split="val", dataset_part=dataset_part, is_training=False)
    test_ds = BioVidCachedDataset(manifest_csv, split="test", dataset_part=dataset_part, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
