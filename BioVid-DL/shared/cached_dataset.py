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
    converting uint8 frames to normalized float32 tensors on the fly.
    """
    def __init__(self, manifest_csv: str, split: str = None, dataset_part: str = None):
        super().__init__()
        df = pd.read_csv(manifest_csv)
        if split:
            df = df[df["split"] == split]
        if dataset_part:
            df = df[df["dataset_part"] == dataset_part]
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = row["cache_path"]

        data = np.load(cache_path)
        # frames: uint8 [T, 224, 224, 3]
        raw_frames = data["frames"] # [T, H, W, C]
        # Transpose to [T, C, H, W]
        transposed = raw_frames.transpose(0, 3, 1, 2)
        # Normalize: (x/255.0 - mean) / std
        norm_frames = (transposed.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        vid_tensor = torch.from_numpy(norm_frames).float() # [T, 3, 224, 224]
        bio_arr = data["bio"] if "bio" in data else np.zeros((3, 2816), dtype=np.float32)
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

def get_dataloaders(manifest_csv, batch_size=16, dataset_part=None, num_workers=2):
    train_ds = BioVidCachedDataset(manifest_csv, split="train", dataset_part=dataset_part)
    val_ds = BioVidCachedDataset(manifest_csv, split="val", dataset_part=dataset_part)
    test_ds = BioVidCachedDataset(manifest_csv, split="test", dataset_part=dataset_part)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
