import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import config
from src.dataset import LABEL_MAP, create_dataloader
from src.models.vivit import ViViT, ViViTConfig
from src.utils import (
    compute_metrics,
    get_device,
    get_gpu_memory_info,
    plot_confusion_matrix,
    set_seed,
)


def save_environment_config(
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    train_samples: int,
    val_samples: int,
    test_samples: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    output_dir: Path,
) -> None:
    """Saves config.yaml and label_mapping.json as requested in Step 18."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"

    cfg_dict = {
        "hardware": {
            "gpu": device_name,
            "pytorch_version": torch.__version__,
            "cuda_version": cuda_ver,
            "amp_enabled": config.use_amp,
        },
        "model": {
            "architecture": "ViViT (Factorized Spatio-Temporal Encoder)",
            "num_frames": config.num_frames,
            "image_resolution": [config.image_size, config.image_size],
            "patch_size": config.patch_size,
            "tubelet_t": config.tubelet_t,
            "embed_dim": config.embed_dim,
            "spatial_layers": config.spatial_layers,
            "temporal_layers": config.temporal_layers,
            "num_heads": config.num_heads,
            "mlp_dim": config.mlp_dim,
            "dropout": config.dropout,
            "num_classes": config.num_classes,
        },
        "dataset": {
            "name": "BioVid Heat Pain Database (Part A)",
            "total_dataset_size": train_samples + val_samples + test_samples,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "train_subjects_count": len(train_subjects),
            "val_subjects_count": len(val_subjects),
            "test_subjects_count": len(test_subjects),
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": epochs,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss_function": "CrossEntropyLoss",
            "gradient_clip_norm": config.gradient_clip_norm,
            "seed": config.seed,
        },
    }

    yaml_path = output_dir / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved configuration metadata: {yaml_path}")

    # Save label mapping
    mapping_path = output_dir / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(LABEL_MAP, f, indent=2)
    print(f"Saved label mapping: {mapping_path}")


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Runs evaluation over a DataLoader and computes metrics and error analysis DF."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_confs = []
    all_sub_ids = []
    all_trial_ids = []
    all_sample_names = []

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast("cuda", enabled=config.use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            probs = F.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            total_loss += loss.item() * len(y)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

            all_sub_ids.extend(meta["subject_id"].numpy().tolist() if isinstance(meta["subject_id"], torch.Tensor) else meta["subject_id"])
            all_trial_ids.extend(meta["trial_id"].numpy().tolist() if isinstance(meta["trial_id"], torch.Tensor) else meta["trial_id"])
            all_sample_names.extend(meta["sample_name"])

    avg_loss = total_loss / max(len(all_targets), 1)
    y_true_arr = np.array(all_targets)
    y_pred_arr = np.array(all_preds)

    metrics = compute_metrics(y_true_arr, y_pred_arr)
    metrics["eval_loss"] = avg_loss

    # DataFrame for error analysis (Step 19)
    pred_df = pd.DataFrame({
        "subject_id": all_sub_ids,
        "trial_id": all_trial_ids,
        "sample_name": all_sample_names,
        "true_label": y_true_arr,
        "predicted_label": y_pred_arr,
        "confidence": np.round(all_confs, 4),
        "correct": (y_true_arr == y_pred_arr),
    })

    return avg_loss, metrics, pred_df


def train_and_evaluate(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_workers: int = 2,
    max_train_subjects: Optional[int] = None,
    max_samples_per_subject: Optional[int] = None,
):
    print("=" * 80)
    print("STEP 18: TRAINING ViViT ON BIOVID DATASET")
    print("=" * 80)

    set_seed(config.seed)
    device = get_device()

    # Load splits
    train_df = pd.read_csv(config.manifest_dir / "train.csv")
    val_df = pd.read_csv(config.manifest_dir / "val.csv")
    test_df = pd.read_csv(config.manifest_dir / "test.csv")

    # Optional subset filtering for fast demonstration / testing
    if max_train_subjects is not None:
        selected_train_subs = train_df["subject_name"].unique()[:max_train_subjects]
        train_df = train_df[train_df["subject_name"].isin(selected_train_subs)].copy()
        selected_val_subs = val_df["subject_name"].unique()[:max(2, max_train_subjects // 4)]
        val_df = val_df[val_df["subject_name"].isin(selected_val_subs)].copy()
        selected_test_subs = test_df["subject_name"].unique()[:max(2, max_train_subjects // 4)]
        test_df = test_df[test_df["subject_name"].isin(selected_test_subs)].copy()

    if max_samples_per_subject is not None:
        per_class = max(1, max_samples_per_subject // config.num_classes)
        train_df = train_df.groupby(["subject_name", "label"]).head(per_class).copy()
        val_df = val_df.groupby(["subject_name", "label"]).head(per_class).copy()
        test_df = test_df.groupby(["subject_name", "label"]).head(per_class).copy()

    train_subjects = sorted(train_df["subject_name"].unique().tolist())
    val_subjects = sorted(val_df["subject_name"].unique().tolist())
    test_subjects = sorted(test_df["subject_name"].unique().tolist())

    print(f"Active training subjects:   {len(train_subjects)} ({len(train_df)} samples)")
    print(f"Active validation subjects: {len(val_subjects)} ({len(val_df)} samples)")
    print(f"Active testing subjects:    {len(test_subjects)} ({len(test_df)} samples)")

    # Assert ZERO leakage
    assert len(set(train_subjects) & set(val_subjects)) == 0, "DATA LEAKAGE between Train and Val!"
    assert len(set(train_subjects) & set(test_subjects)) == 0, "DATA LEAKAGE between Train and Test!"
    assert len(set(val_subjects) & set(test_subjects)) == 0, "DATA LEAKAGE between Val and Test!"

    # Save environment config
    save_environment_config(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        train_samples=len(train_df),
        val_samples=len(val_df),
        test_samples=len(test_df),
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
        output_dir=config.logs_dir,
    )

    # Create DataLoaders
    train_loader = create_dataloader(
        train_df,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        num_frames=config.num_frames,
        target_size=config.image_size,
        crop_mode=config.crop_mode,
    )
    val_loader = create_dataloader(
        val_df,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        num_frames=config.num_frames,
        target_size=config.image_size,
        crop_mode=config.crop_mode,
    )
    test_loader = create_dataloader(
        test_df,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        num_frames=config.num_frames,
        target_size=config.image_size,
        crop_mode=config.crop_mode,
    )

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    best_val_f1 = -1.0
    history = []

    print(f"\nStarting training loop for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (x, y, _) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=config.use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == y).sum().item()
            total_train += len(y)

        train_loss = running_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        # Validation
        val_loss, val_metrics, _ = evaluate_model(model, val_loader, criterion, device)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        scheduler.step()
        epoch_sec = time.time() - epoch_start
        gpu_mem = get_gpu_memory_info()

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_sec:5.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.1f}% Macro-F1: {val_f1:.4f} | "
            f"GPU VRAM: {gpu_mem['allocated_mb']:.0f} MB"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_weighted_f1": round(val_metrics["weighted_f1"], 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "time_sec": round(epoch_sec, 2),
        })

        # Save Best Model Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_ckpt_path = config.checkpoints_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_accuracy": val_acc,
                "config": model_cfg,
            }, best_ckpt_path)
            print(f"  -> Saved new best model checkpoint to: {best_ckpt_path}")

    # Save Last Model Checkpoint
    last_ckpt_path = config.checkpoints_dir / "last_model.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_cfg,
    }, last_ckpt_path)
    print(f"\nSaved last model checkpoint to: {last_ckpt_path}")

    # Save Training History CSV
    history_df = pd.DataFrame(history)
    history_path = config.logs_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to: {history_path}")

    print("\n" + "=" * 80)
    print("STEP 19: EVALUATION ON UNSEEN TEST SUBJECTS")
    print("=" * 80)

    # Load best model for evaluation
    best_ckpt = torch.load(config.checkpoints_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (Val Macro-F1: {best_ckpt['val_macro_f1']:.4f})")

    test_loss, test_metrics, test_preds_df = evaluate_model(
        model, test_loader, criterion, device
    )

    print(f"\n--- Unseen Test Set Metrics ({len(test_preds_df)} samples) ---")
    print(f"Test Loss:             {test_loss:.4f}")
    print(f"Test Accuracy:         {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:       {test_metrics['macro_precision']:.4f}")
    print(f"Weighted Precision:    {test_metrics['weighted_precision']:.4f}")
    print(f"Macro Recall:          {test_metrics['macro_recall']:.4f}")
    print(f"Weighted Recall:       {test_metrics['weighted_recall']:.4f}")
    print(f"Macro F1 Score:        {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score:     {test_metrics['weighted_f1']:.4f}")

    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 55)
    for cname, pcm in test_metrics["per_class"].items():
        print(f"{cname:<8} | {pcm['precision']:<10.4f} | {pcm['recall']:<10.4f} | {pcm['f1_score']:<10.4f} | {pcm['support']:<8d}")

    print("\n--- Confusion Matrix ---")
    cm_arr = np.array(test_metrics["confusion_matrix"])
    print(cm_arr)

    # Plot and save confusion matrix
    cm_plot_path = config.evaluation_dir / "confusion_matrix.png"
    class_names = [LABEL_MAP[i] for i in range(config.num_classes)]
    plot_confusion_matrix(cm_arr, class_names, cm_plot_path)
    print(f"\nSaved confusion matrix figure to: {cm_plot_path}")

    # Save test metrics JSON
    metrics_json_path = config.logs_dir / "test_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved test metrics JSON to: {metrics_json_path}")

    # Save detailed prediction CSV for error analysis (Step 19)
    preds_csv_path = config.logs_dir / "test_predictions.csv"
    test_preds_df.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions error analysis CSV to: {preds_csv_path}")

    print("=" * 80)
    print("STEP 18 & 19 TRAINING & EVALUATION COMPLETE: ALL ARTIFACTS SAVED!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioVid ViViT Training & Evaluation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Worker threads")
    parser.add_argument("--max_train_subjects", type=int, default=None, help="Limit subjects for quick run")
    parser.add_argument("--max_samples_per_subject", type=int, default=None, help="Limit samples per subject")
    args = parser.parse_args()

    train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        max_train_subjects=args.max_train_subjects,
        max_samples_per_subject=args.max_samples_per_subject,
    )
