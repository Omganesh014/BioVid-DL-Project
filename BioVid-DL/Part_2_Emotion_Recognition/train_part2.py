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

sys.path.append("BioVid-DL/Part_2_Emotion_Recognition/preprocessing")
sys.path.append("BioVid-DL/Part_2_Emotion_Recognition/models")
sys.path.append("BioVid-DL/Part_1_Pain_Recognition/utils") # Re-use evaluation utilities

from emotion_dataset import BioVidEmotionDataset, PART_D_CLASSES, PART_E_CLASSES
from r3d import VideoR3D18
from video_swin import VideoSwinTransformer
from evaluator import evaluate_and_save, save_training_curves, count_parameters

RESULTS_DIR = "BioVid-DL/Part_2_Emotion_Recognition/results"
TRAIN_CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")

def train_emotion_model(model, model_name, train_loader, val_loader, epochs=4, lr=1e-4, device="cuda"):
    print(f"\n{'='*25}\nTraining {model_name} on {device}\n{'='*25}")
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

        for videos, labels, _ in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * videos.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = total_loss / (total + 1e-9)
        train_acc = correct / (total + 1e-9)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for videos, labels, _ in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(videos)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                v_loss += loss.item() * videos.size(0)
                preds = torch.argmax(outputs, dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / (v_total + 1e-9)
        val_acc = v_correct / (v_total + 1e-9)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch}/{epochs}] ({dt:.1f}s) — Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    save_training_curves(history, model_name, TRAIN_CURVES_DIR)
    return model

def eval_emotion_model(model, model_name, test_loader, class_names, device="cuda"):
    model.eval()
    actuals, preds, probs, metas = [], [], [], []
    t_start = time.time()

    with torch.no_grad():
        for videos, labels, meta in test_loader:
            videos = videos.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(videos)
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

    t_total = time.time() - t_start
    inf_time_ms = (t_total / max(1, len(actuals))) * 1000.0
    param_count = count_parameters(model)

    metrics = evaluate_and_save(
        actuals,
        preds,
        probs,
        metas,
        model_name,
        param_count,
        inf_time_ms,
        RESULTS_DIR,
        class_names=class_names
    )
    return metrics

def run_part_2_experiments():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inv_df = pd.read_csv("BioVid-DL/shared/subject_splits/inventory_with_splits.csv")

    # ---------------- EXPERIMENT 1: Part D (7 Classes) ----------------
    print("\n" + "="*50)
    print("PART 2 — EXPERIMENT 1: Part D Emotion Recognition (7 Categories)")
    print("="*50)
    df_d = inv_df[inv_df["dataset_part"] == "PartD"]
    train_d = BioVidEmotionDataset(df_d[df_d["split"] == "train"], dataset_part="PartD", num_frames=8)
    val_d = BioVidEmotionDataset(df_d[df_d["split"] == "val"], dataset_part="PartD", num_frames=8)
    test_d = BioVidEmotionDataset(df_d[df_d["split"] == "test"], dataset_part="PartD", num_frames=8)

    loader_train_d = DataLoader(train_d, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    loader_val_d = DataLoader(val_d, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    loader_test_d = DataLoader(test_d, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # 1. 3D CNN (R3D-18) on Part D
    m_r3d_d = VideoR3D18(num_classes=7, pretrained=True)
    m_r3d_d = train_emotion_model(m_r3d_d, "PartD_3D_CNN_R3D18", loader_train_d, loader_val_d, epochs=4, lr=1e-4, device=device)
    metrics_r3d_d = eval_emotion_model(m_r3d_d, "PartD_3D_CNN_R3D18", loader_test_d, PART_D_CLASSES, device=device)

    # 2. Video Swin Transformer on Part D
    m_swin_d = VideoSwinTransformer(num_classes=7, embed_dim=128, depths=(2, 2), num_heads=(4, 8))
    m_swin_d = train_emotion_model(m_swin_d, "PartD_Video_Swin", loader_train_d, loader_val_d, epochs=4, lr=1e-4, device=device)
    metrics_swin_d = eval_emotion_model(m_swin_d, "PartD_Video_Swin", loader_test_d, PART_D_CLASSES, device=device)

    # ---------------- EXPERIMENT 2: Part E (5 Classes) ----------------
    print("\n" + "="*50)
    print("PART 2 — EXPERIMENT 2: Part E BioVidEmoDB Recognition (5 Categories)")
    print("="*50)
    df_e = inv_df[inv_df["dataset_part"] == "PartE"]
    train_e = BioVidEmotionDataset(df_e[df_e["split"] == "train"], dataset_part="PartE", num_frames=8)
    val_e = BioVidEmotionDataset(df_e[df_e["split"] == "val"], dataset_part="PartE", num_frames=8)
    test_e = BioVidEmotionDataset(df_e[df_e["split"] == "test"], dataset_part="PartE", num_frames=8)

    loader_train_e = DataLoader(train_e, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    loader_val_e = DataLoader(val_e, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    loader_test_e = DataLoader(test_e, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # 1. 3D CNN on Part E
    m_r3d_e = VideoR3D18(num_classes=5, pretrained=True)
    m_r3d_e = train_emotion_model(m_r3d_e, "PartE_3D_CNN_R3D18", loader_train_e, loader_val_e, epochs=4, lr=1e-4, device=device)
    metrics_r3d_e = eval_emotion_model(m_r3d_e, "PartE_3D_CNN_R3D18", loader_test_e, PART_E_CLASSES, device=device)

    # 2. Video Swin Transformer on Part E
    m_swin_e = VideoSwinTransformer(num_classes=5, embed_dim=128, depths=(2, 2), num_heads=(4, 8))
    m_swin_e = train_emotion_model(m_swin_e, "PartE_Video_Swin", loader_train_e, loader_val_e, epochs=4, lr=1e-4, device=device)
    metrics_swin_e = eval_emotion_model(m_swin_e, "PartE_Video_Swin", loader_test_e, PART_E_CLASSES, device=device)

    # Summary table
    comp_df = pd.DataFrame([metrics_r3d_d, metrics_swin_d, metrics_r3d_e, metrics_swin_e])
    comp_df.to_csv(os.path.join(RESULTS_DIR, "part2_emotion_model_comparison.csv"), index=False)
    print("\nPart 2 Emotion Model Comparison:")
    print(comp_df[["model_name", "accuracy", "macro_f1", "weighted_f1", "inference_time_ms", "parameter_count"]])

if __name__ == "__main__":
    run_part_2_experiments()
