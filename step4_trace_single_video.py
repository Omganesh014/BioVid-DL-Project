from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import config
from src.dataset import (
    LABEL_MAP,
    get_temporal_indices,
    preprocess_frame,
    read_video_frames,
)
from src.models.vivit import ViViT, ViViTConfig
from src.utils import get_device, set_seed


def trace_and_verify_single_video():
    set_seed(42)
    device = get_device()

    print("=" * 80)
    print("STEP 15: TRACE ONE ACTUAL BIOVID VIDEO THROUGH ViViT PIPELINE")
    print("=" * 80)

    # 1. Select one real video from manifest
    manifest_p = config.manifest_dir / "biovid_manifest.csv"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    df = pd.read_csv(manifest_p)
    sample_row = df.iloc[0]

    video_path = sample_row["video_path"]
    true_label = int(sample_row["label"])
    class_name = sample_row["class_name"]
    sub_name = sample_row["subject_name"]
    sample_name = sample_row["sample_name"]

    print(f"Tracing video: {video_path}")
    print(f"Subject: {sub_name} | Sample: {sample_name}")
    print(f"True Label: {true_label} ({class_name})")

    # Trace video loading
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("\n--- Pipeline Step-by-Step Trace ---")
    print(f"1. Original video:             {video_path}")
    print(f"   Resolution & FPS:           {w}x{h} @ {fps:.2f} fps")
    print(f"2. Frame count:                {total_frames} frames ({total_frames/fps:.2f} seconds)")

    sampled_indices = get_temporal_indices(total_frames, config.num_frames)
    print(f"3. Sampled frame indices (T={config.num_frames}): {sampled_indices}")

    raw_frames = read_video_frames(
        video_path,
        num_frames=config.num_frames,
        target_size=config.image_size,
        crop_mode=config.crop_mode,
    )
    print(f"4. Sampled & preprocessed:     numpy array shape = {raw_frames.shape} (T x C x H x W)")
    print(f"   Value range:                min={raw_frames.min():.3f}, max={raw_frames.max():.3f}, dtype={raw_frames.dtype}")

    # Build tensor of shape [B, T, C, H, W] with B=1
    tensor_input = torch.from_numpy(raw_frames).unsqueeze(0).float().to(device)
    print(f"5. Batch tensor shape:         {tuple(tensor_input.shape)} [B, T, C, H, W]")

    # Initialize ViViT model
    model_cfg = ViViTConfig(
        num_frames=config.num_frames,
        image_size=config.image_size,
        patch_size=config.patch_size,
        tubelet_t=config.tubelet_t,
        in_channels=3,
        embed_dim=config.embed_dim,
        spatial_layers=config.spatial_layers,
        temporal_layers=config.temporal_layers,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )

    model = ViViT(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total ({trainable_params:,} trainable)")

    # Trace inside ViViT
    print("\n--- ViViT Layer-by-Layer Internal Tensor Shapes ---")
    layer_shapes = model.trace_pipeline(tensor_input)
    for layer_name, shape in layer_shapes.items():
        print(f"   {layer_name:32s} -> shape: {shape}")

    print("\n" + "=" * 80)
    print("STEP 16: VERIFY MODEL WITH ONE REAL SAMPLE (FORWARD + BACKWARD)")
    print("=" * 80)

    # 1. Forward pass
    model.train()
    outputs = model(tensor_input)
    print(f"1. Output logits shape:        {tuple(outputs.shape)} (Expected: [1, {config.num_classes}])")
    assert outputs.shape == (1, config.num_classes), f"Output shape mismatch: {outputs.shape}"

    # 2. Verify label
    target_tensor = torch.tensor([true_label], dtype=torch.long, device=device)
    print(f"2. Target label:               {target_tensor.item()} ({LABEL_MAP[target_tensor.item()]})")
    assert target_tensor.item() in LABEL_MAP, f"Invalid label: {target_tensor.item()}"

    # 3. Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, target_tensor)
    loss_val = loss.item()
    print(f"3. Computed CrossEntropyLoss:  {loss_val:.4f}")

    # 4. Check for NaN / Inf in loss and logits
    assert not torch.isnan(outputs).any(), "NaN detected in logits!"
    assert not torch.isinf(outputs).any(), "Inf detected in logits!"
    assert not np.isnan(loss_val), "NaN detected in loss!"
    assert not np.isinf(loss_val), "Inf detected in loss!"
    print("4. NaN / Inf Check:            PASSED (Logits & loss are finite and clean)")

    # 5. Backward pass
    model.zero_grad()
    loss.backward()
    print("5. Backward propagation:       COMPLETED successfully")

    # 6. Verify gradients
    grad_norms = []
    has_zero_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has None gradient!")
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}!"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}!"
            g_norm = param.grad.data.norm(2).item()
            grad_norms.append((name, g_norm))

    total_grad_norm = (sum(g ** 2 for _, g in grad_norms)) ** 0.5
    print(f"6. Total gradient norm:        {total_grad_norm:.4f}")
    print(f"   Max parameter gradient:     {max(g for _, g in grad_norms):.4f}")
    print(f"   Min parameter gradient:     {min(g for _, g in grad_norms):.6f}")
    print("7. Gradient Verification:      PASSED (All gradients valid, non-null, finite)")

    # 7. Optimizer step test
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.step()
    print("8. Optimizer step:             COMPLETED successfully")
    print("=" * 80)
    print("STEP 15 & 16 VALIDATION COMPLETE: PIPELINE 100% OPERATIONAL!\n")


if __name__ == "__main__":
    trace_and_verify_single_video()
