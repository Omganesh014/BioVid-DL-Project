import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np

sys.path.append("BioVid-DL/shared")
sys.path.append("BioVid-DL/Part_2_Emotion_Recognition/models/3d_cnn")
sys.path.append("BioVid-DL/Part_2_Emotion_Recognition/models/video_swin_transformer")
sys.path.append("BioVid-DL/Part_1_Pain_Recognition/utils")

from cached_dataset import get_dataloaders
from r3d import VideoR3D18
from video_swin import VideoSwinTransformer
from evaluator import evaluate_and_save, save_training_curves, count_parameters

RESULTS_DIR = "BioVid-DL/Part_2_Emotion_Recognition/results"
TRAIN_CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")
PART_D_CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "pain", "sad"]
PART_E_CLASSES = ["amusement", "anger", "disgust", "fear", "sad"]

def train_model(model, model_name, train_loader, val_loader, epochs=5, lr=1e-4, device="cuda"):
    print(f"\n{'='*30}\nTraining {model_name} on {device}\n{'='*30}", flush=True)
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

        for videos, _, labels, _ in train_loader:
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
            for videos, _, labels, _ in val_loader:
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
        print(f"Epoch [{epoch}/{epochs}] ({dt:.2f}s) — Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", flush=True)

    save_training_curves(history, model_name, TRAIN_CURVES_DIR)
    return model

def eval_model(model, model_name, test_loader, class_names, device="cuda"):
    model.eval()
    actuals, preds, probs, metas = [], [], [], []
    t0 = time.time()

    with torch.no_grad():
        for videos, _, labels, meta in test_loader:
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

    dt = time.time() - t0
    inf_time_ms = (dt / max(1, len(actuals))) * 1000.0
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

def run_part2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_csv = "BioVid-DL/shared/cache/part2_cached_manifest.csv"

    # Part D Loaders (7 classes)
    t_d, v_d, te_d = get_dataloaders(manifest_csv, batch_size=4, dataset_part="PartD")
    # 1. 3D CNN (R3D-18) on Part D
    m_r3d_d = VideoR3D18(num_classes=7, pretrained=True)
    m_r3d_d = train_model(m_r3d_d, "PartD_3D_CNN_R3D18", t_d, v_d, epochs=4, lr=1e-4, device=device)
    metrics_r3d_d = eval_model(m_r3d_d, "PartD_3D_CNN_R3D18", te_d, PART_D_CLASSES, device=device)

    # 2. Video Swin on Part D
    m_swin_d = VideoSwinTransformer(num_classes=7, embed_dim=128, depths=(2, 2), num_heads=(4, 8))
    m_swin_d = train_model(m_swin_d, "PartD_Video_Swin", t_d, v_d, epochs=4, lr=1e-4, device=device)
    metrics_swin_d = eval_model(m_swin_d, "PartD_Video_Swin", te_d, PART_D_CLASSES, device=device)

    # Part E Loaders (5 classes)
    t_e, v_e, te_e = get_dataloaders(manifest_csv, batch_size=4, dataset_part="PartE")
    # 3. 3D CNN on Part E
    m_r3d_e = VideoR3D18(num_classes=5, pretrained=True)
    m_r3d_e = train_model(m_r3d_e, "PartE_3D_CNN_R3D18", t_e, v_e, epochs=4, lr=1e-4, device=device)
    metrics_r3d_e = eval_model(m_r3d_e, "PartE_3D_CNN_R3D18", te_e, PART_E_CLASSES, device=device)

    # 4. Video Swin on Part E
    m_swin_e = VideoSwinTransformer(num_classes=5, embed_dim=128, depths=(2, 2), num_heads=(4, 8))
    m_swin_e = train_model(m_swin_e, "PartE_Video_Swin", t_e, v_e, epochs=4, lr=1e-4, device=device)
    metrics_swin_e = eval_model(m_swin_e, "PartE_Video_Swin", te_e, PART_E_CLASSES, device=device)

    # Summary
    comp_df = pd.DataFrame([metrics_r3d_d, metrics_swin_d, metrics_r3d_e, metrics_swin_e])
    comp_df.to_csv(os.path.join(RESULTS_DIR, "part2_emotion_model_comparison.csv"), index=False)
    print("\nPart 2 Emotion Model Comparison Summary:", flush=True)
    print(comp_df[["model_name", "accuracy", "macro_f1", "weighted_f1", "inference_time_ms", "parameter_count"]], flush=True)

if __name__ == "__main__":
    run_part2()
