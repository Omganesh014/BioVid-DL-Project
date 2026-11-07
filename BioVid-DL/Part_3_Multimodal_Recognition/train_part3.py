import os
import sys
import time
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
os.makedirs(ABLATION_DIR, exist_ok=True)

PAIN_CLASSES = ["BL1", "PA1", "PA2", "PA3", "PA4"]

def train_multimodal_model(model, train_loader, val_loader, epochs=5, lr=1e-4, device="cuda"):
    print(f"\n{'='*35}\nTraining Multimodal CNN (Video + Biosignals)\n{'='*35}", flush=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for videos, bios, labels, _ in train_loader:
            videos, bios, labels = videos.to(device), bios.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(video=videos, bio=bios, task="pain", modality_mode="multimodal")
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(video=videos, bio=bios, task="pain", modality_mode="multimodal")
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * videos.size(0)
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
        print(f"Epoch [{epoch}/{epochs}] ({dt:.2f}s) — Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", flush=True)

    save_training_curves(history, "Multimodal_CNN", TRAIN_CURVES_DIR)
    return model

def eval_ablation_mode(model, test_loader, mode_name, mask_channels=None, device="cuda"):
    """
    Evaluates model in different ablation settings:
    - 'Multimodal_Fusion' (all video + all biosignals)
    - 'Video_Only'
    - 'All_Biosignals_Only'
    - 'GSR_Only' (channels 0 active, others zeroed)
    - 'ECG_Only' (channel 1 active, others zeroed)
    - 'EMG_Only' (channel 2 active, others zeroed)
    """
    model.eval()
    actuals, preds, probs, metas = [], [], [], []
    t0 = time.time()

    with torch.no_grad():
        for videos, bios, labels, meta in test_loader:
            videos, bios = videos.to(device), bios.to(device)

            # Apply channel masks if requested
            if mask_channels is not None:
                masked_bio = torch.zeros_like(bios)
                for ch in mask_channels:
                    masked_bio[:, ch:ch+1, :] = bios[:, ch:ch+1, :]
            else:
                masked_bio = bios

            with torch.amp.autocast('cuda'):
                if mode_name == "Video_Only":
                    outputs = model(video=videos, bio=None, task="pain", modality_mode="video_only")
                elif mode_name in ["All_Biosignals_Only", "GSR_Only", "ECG_Only", "EMG_Only"]:
                    outputs = model(video=None, bio=masked_bio, task="pain", modality_mode="bio_only")
                else: # Multimodal Fusion
                    outputs = model(video=videos, bio=masked_bio, task="pain", modality_mode="multimodal")

                batch_probs = torch.softmax(outputs, dim=1)

            batch_preds = torch.argmax(outputs, dim=1)
            actuals.extend(labels.numpy().tolist())
            preds.extend(batch_preds.cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy().tolist())

            for i in range(len(labels)):
                metas.append({
                    "dataset_part": meta["dataset_part"][i],
                    "subject_name": meta["subject_name"][i],
                    "sample_name": meta["sample_name"][i],
                })

    dt = time.time() - t0
    inf_time_ms = (dt / len(actuals)) * 1000.0
    param_count = count_parameters(model)

    metrics = evaluate_and_save(
        actuals,
        preds,
        probs,
        metas,
        mode_name,
        param_count,
        inf_time_ms,
        RESULTS_DIR,
        class_names=PAIN_CLASSES
    )
    return metrics

def plot_ablation_results(df_ablation):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    sns.barplot(data=df_ablation, x="model_name", y="accuracy", ax=ax1, palette="mako")
    ax1.set_title("Modality Ablation: Accuracy Comparison", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Modality Configuration", fontsize=11)
    ax1.set_ylabel("Classification Accuracy", fontsize=11)
    ax1.tick_params(axis='x', rotation=35)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=10, fontweight="bold")

    # Macro-F1 comparison
    sns.barplot(data=df_ablation, x="model_name", y="macro_f1", ax=ax2, palette="rocket")
    ax2.set_title("Modality Ablation: Macro-F1 Score Comparison", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Modality Configuration", fontsize=11)
    ax2.set_ylabel("Macro-F1 Score", fontsize=11)
    ax2.tick_params(axis='x', rotation=35)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(ABLATION_DIR, "modality_ablation_comparison.png"), dpi=300)
    plt.close(fig)
    print("Modality ablation graph saved successfully.", flush=True)

def run_part3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_csv = "BioVid-DL/shared/cache/part1_cached_manifest.csv"

    train_ds = BioVidCachedDataset(manifest_csv, split="train")
    val_ds = BioVidCachedDataset(manifest_csv, split="val")
    test_ds = BioVidCachedDataset(manifest_csv, split="test")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Multimodal CNN
    model = MultimodalCNN(num_bio_channels=3, num_pain_classes=5, num_emotion_classes=7, pretrained=True)
    model = train_multimodal_model(model, train_loader, val_loader, epochs=5, lr=1e-4, device=device)

    # Modality Ablations on matched test set
    print("\nRunning Rigorous Modality Ablations on Matched Test Set...", flush=True)
    m_fusion = eval_ablation_mode(model, test_loader, "Complete_Multimodal_Fusion", mask_channels=None, device=device)
    m_video = eval_ablation_mode(model, test_loader, "Video_Only", device=device)
    m_bio = eval_ablation_mode(model, test_loader, "All_Biosignals_Only", mask_channels=[0, 1, 2], device=device)
    m_gsr = eval_ablation_mode(model, test_loader, "GSR_Only", mask_channels=[0], device=device)
    m_ecg = eval_ablation_mode(model, test_loader, "ECG_Only", mask_channels=[1], device=device)
    m_emg = eval_ablation_mode(model, test_loader, "EMG_Only", mask_channels=[2], device=device)

    ablation_records = [m_fusion, m_video, m_bio, m_gsr, m_ecg, m_emg]
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(os.path.join(ABLATION_DIR, "modality_ablation_metrics.csv"), index=False)
    plot_ablation_results(df_ablation)

    print("\nPart 3 Modality Ablation Summary:", flush=True)
    print(df_ablation[["model_name", "accuracy", "macro_f1", "weighted_f1", "inference_time_ms"]], flush=True)

if __name__ == "__main__":
    run_part3()
