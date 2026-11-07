import os
import sys
import time
import copy
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("BioVid-DL/shared")
sys.path.append("BioVid-DL/Part_3_Multimodal_Recognition/models/multimodal_cnn")
sys.path.append("BioVid-DL/Part_1_Pain_Recognition/utils")

from cached_dataset import BioVidCachedDataset
from multimodal_cnn import MultimodalCNN
from evaluator import evaluate_and_save, save_training_curves, count_parameters

RESULTS_DIR = "BioVid-DL/Part_3_Multimodal_Recognition/results"
TRAIN_CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")
ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation_studies")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
for d in [ABLATION_DIR, CKPT_DIR, TRAIN_CURVES_DIR]:
    os.makedirs(d, exist_ok=True)

PAIN_CLASSES = ["BL1", "PA1", "PA2", "PA3", "PA4"]

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()

def train_multimodal_model(
    model,
    train_loader,
    val_loader,
    epochs=35,
    lr=1e-4,
    device="cuda",
    accum_steps=4,
    patience=7,
):
    print(f"\n{'='*40}\nTraining Multimodal CNN on {device} (Max Epochs: {epochs}, Effective Batch: {train_loader.batch_size * accum_steps})\n{'='*40}", flush=True)
    model = model.to(device)
    criterion = FocalLossWithLabelSmoothing(gamma=1.5, label_smoothing=0.05)

    # Differential learning rate
    param_groups = [
        {"params": model.video_cnn.parameters(), "lr": lr * 0.15},
        {"params": model.bio_cnn.parameters(), "lr": lr},
        {"params": model.fusion.parameters(), "lr": lr},
        {"params": model.pain_head.parameters(), "lr": lr},
    ]

    optimizer = AdamW(param_groups, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        optimizer.zero_grad()

        for step_idx, (videos, bios, labels, _) in enumerate(train_loader):
            videos, bios, labels = videos.to(device), bios.to(device), labels.to(device)

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(video=videos, bio=bios, task="pain", modality_mode="multimodal")
                    loss = criterion(outputs, labels) / accum_steps
                scaler.scale(loss).backward()
                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(video=videos, bio=bios, task="pain", modality_mode="multimodal")
                loss = criterion(outputs, labels) / accum_steps
                loss.backward()
                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * accum_steps * videos.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for videos, bios, labels, _ in val_loader:
                videos, bios, labels = videos.to(device), bios.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(video=videos, bio=bios, task="pain", modality_mode="multimodal")
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(video=videos, bio=bios, task="pain", modality_mode="multimodal")
                    loss = criterion(outputs, labels)
                v_loss += loss.item() * videos.size(0)
                preds = torch.argmax(outputs, dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(CKPT_DIR, "Multimodal_CNN_best.pt"))
            patience_counter = 0
            is_best = " [BEST CHECKPOINT SAVED]"
        else:
            patience_counter += 1

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) — Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}{is_best}", flush=True)

        if patience_counter >= patience:
            print(f"--> Early stopping triggered at Epoch {epoch}. Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}", flush=True)
            break

    print(f"Restoring best weights for Multimodal CNN (Lowest Val Loss: {best_val_loss:.4f})...", flush=True)
    model.load_state_dict(best_weights)
    save_training_curves(history, "Multimodal_CNN", TRAIN_CURVES_DIR)
    return model

def eval_multimodal_ablation(model, test_loader, mode="multimodal", bio_channel_mask=None, config_name="Complete_Fusion", device="cuda"):
    model.eval()
    actuals, preds, probs, metas = [], [], [], []
    t0 = time.time()

    with torch.no_grad():
        for videos, bios, labels, meta in test_loader:
            videos, bios = videos.to(device), bios.to(device)

            if bio_channel_mask is not None:
                mask = torch.tensor(bio_channel_mask, device=device).view(1, -1, 1).float()
                bios = bios * mask

            with torch.amp.autocast('cuda'):
                outputs = model(video=videos, bio=bios, task="pain", modality_mode=mode)
                batch_probs = torch.softmax(outputs, dim=1)
            batch_preds = torch.argmax(outputs, dim=1)

            actuals.extend(labels.cpu().numpy().tolist())
            preds.extend(batch_preds.cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy().tolist())

            for i in range(len(labels)):
                metas.append({
                    "dataset_part": meta["dataset_part"][i],
                    "subject_name": meta["subject_name"][i],
                    "sample_name": meta["sample_name"][i],
                })

    dt = time.time() - t0
    inf_time_ms = (dt / len(actuals)) * 1000.0 if len(actuals) > 0 else 0.0
    param_count = count_parameters(model)

    metrics = evaluate_and_save(
        actuals,
        preds,
        probs,
        metas,
        config_name,
        param_count,
        inf_time_ms,
        RESULTS_DIR
    )
    return metrics

def run_part3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_csv = "BioVid-DL/shared/cache/part1_cached_manifest.csv"

    train_ds = BioVidCachedDataset(manifest_csv, split="train", is_training=True)
    val_ds = BioVidCachedDataset(manifest_csv, split="val", is_training=False)
    test_ds = BioVidCachedDataset(manifest_csv, split="test", is_training=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # 1. Train Multimodal CNN
    model = MultimodalCNN(num_bio_channels=3, num_pain_classes=5, num_emotion_classes=7, pretrained=True, dropout=0.3)
    model = train_multimodal_model(model, train_loader, val_loader, epochs=35, lr=1.2e-4, device=device, accum_steps=4, patience=7)

    # 2. Comprehensive Ablation Studies
    print("\n--- Running Multimodal Ablation Studies on Unseen Test Subjects ---", flush=True)
    ablation_results = []

    # Config 1: Complete Fusion
    m1 = eval_multimodal_ablation(model, test_loader, mode="multimodal", config_name="Complete_Multimodal_Fusion", device=device)
    ablation_results.append({
        "Configuration": "Complete Fusion",
        "Modality Channels": "Video + GSR + ECG + EMG",
        "Accuracy": m1["accuracy"],
        "Macro_Precision": m1["macro_precision"],
        "Macro_Recall": m1["macro_recall"],
        "Macro_F1": m1["macro_f1"],
        "Weighted_F1": m1["weighted_f1"],
        "MAE": m1["mae"],
        "QWK": m1["qwk"],
    })

    # Config 2: Video Only
    m2 = eval_multimodal_ablation(model, test_loader, mode="video_only", config_name="Video_Only", device=device)
    ablation_results.append({
        "Configuration": "Video Only",
        "Modality Channels": "Video only (Biosignals zeroed)",
        "Accuracy": m2["accuracy"],
        "Macro_Precision": m2["macro_precision"],
        "Macro_Recall": m2["macro_recall"],
        "Macro_F1": m2["macro_f1"],
        "Weighted_F1": m2["weighted_f1"],
        "MAE": m2["mae"],
        "QWK": m2["qwk"],
    })

    # Config 3: All Biosignals Only
    m3 = eval_multimodal_ablation(model, test_loader, mode="bio_only", config_name="All_Biosignals_Only", device=device)
    ablation_results.append({
        "Configuration": "All Biosignals",
        "Modality Channels": "GSR + ECG + EMG (Video zeroed)",
        "Accuracy": m3["accuracy"],
        "Macro_Precision": m3["macro_precision"],
        "Macro_Recall": m3["macro_recall"],
        "Macro_F1": m3["macro_f1"],
        "Weighted_F1": m3["weighted_f1"],
        "MAE": m3["mae"],
        "QWK": m3["qwk"],
    })

    # Config 4: GSR Only (Channel 0)
    m4 = eval_multimodal_ablation(model, test_loader, mode="bio_only", bio_channel_mask=[1.0, 0.0, 0.0], config_name="GSR_Only", device=device)
    ablation_results.append({
        "Configuration": "GSR Only",
        "Modality Channels": "Galvanic Skin Response only",
        "Accuracy": m4["accuracy"],
        "Macro_Precision": m4["macro_precision"],
        "Macro_Recall": m4["macro_recall"],
        "Macro_F1": m4["macro_f1"],
        "Weighted_F1": m4["weighted_f1"],
        "MAE": m4["mae"],
        "QWK": m4["qwk"],
    })

    # Config 5: ECG Only (Channel 1)
    m5 = eval_multimodal_ablation(model, test_loader, mode="bio_only", bio_channel_mask=[0.0, 1.0, 0.0], config_name="ECG_Only", device=device)
    ablation_results.append({
        "Configuration": "ECG Only",
        "Modality Channels": "Electrocardiogram only",
        "Accuracy": m5["accuracy"],
        "Macro_Precision": m5["macro_precision"],
        "Macro_Recall": m5["macro_recall"],
        "Macro_F1": m5["macro_f1"],
        "Weighted_F1": m5["weighted_f1"],
        "MAE": m5["mae"],
        "QWK": m5["qwk"],
    })

    # Config 6: EMG Only (Channel 2)
    m6 = eval_multimodal_ablation(model, test_loader, mode="bio_only", bio_channel_mask=[0.0, 0.0, 1.0], config_name="EMG_Only", device=device)
    ablation_results.append({
        "Configuration": "EMG Only",
        "Modality Channels": "Electromyography only",
        "Accuracy": m6["accuracy"],
        "Macro_Precision": m6["macro_precision"],
        "Macro_Recall": m6["macro_recall"],
        "Macro_F1": m6["macro_f1"],
        "Weighted_F1": m6["weighted_f1"],
        "MAE": m6["mae"],
        "QWK": m6["qwk"],
    })

    # Save summary
    df_abl = pd.DataFrame(ablation_results)
    df_abl.to_csv(os.path.join(ABLATION_DIR, "multimodal_ablation_summary.csv"), index=False)
    print("\nMultimodal Ablation Summary Table:", flush=True)
    print(df_abl[["Configuration", "Accuracy", "Macro_F1", "MAE", "QWK"]], flush=True)

if __name__ == "__main__":
    run_part3()
