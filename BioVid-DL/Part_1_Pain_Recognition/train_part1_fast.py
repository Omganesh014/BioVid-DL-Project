import os
import sys
import time
import copy
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np

sys.path.append("BioVid-DL/shared")
sys.path.append("BioVid-DL/Part_1_Pain_Recognition/models")
sys.path.append("BioVid-DL/Part_1_Pain_Recognition/utils")

from cached_dataset import get_dataloaders
from resnet50.resnet50 import VideoResNet50
from cnn_lstm.cnn_lstm import VideoCNNLSTM
from vivit.vivit import ViViT, ViViTConfig
from evaluator import evaluate_and_save, save_training_curves, count_parameters

RESULTS_DIR = "BioVid-DL/Part_1_Pain_Recognition/results"
TRAIN_CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

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

def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    epochs=35,
    lr=1e-4,
    device="cuda",
    accum_steps=4,
    patience=7,
):
    print(f"\n{'='*40}\nTraining {model_name} on {device} (Max Epochs: {epochs}, Accum: {accum_steps}, Effective Batch: {train_loader.batch_size * accum_steps})\n{'='*40}", flush=True)
    model = model.to(device)
    criterion = FocalLossWithLabelSmoothing(gamma=1.5, label_smoothing=0.05)

    # Differential learning rates for fine-tuning
    if hasattr(model, "feature_extractor"): # ResNet50
        param_groups = [
            {"params": model.feature_extractor.parameters(), "lr": lr * 0.15},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    elif hasattr(model, "cnn") and hasattr(model, "lstm"): # CNN-LSTM
        param_groups = [
            {"params": model.cnn.parameters(), "lr": lr * 0.15},
            {"params": model.lstm.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

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

        for step_idx, (videos, _, labels, _) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, labels) / accum_steps
                scaler.scale(loss).backward()
                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(videos)
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

        # Validation phase after every epoch
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
            torch.save(best_weights, os.path.join(CKPT_DIR, f"{model_name}_best.pt"))
            patience_counter = 0
            is_best = " [BEST CHECKPOINT SAVED]"
        else:
            patience_counter += 1

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) — Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}{is_best}", flush=True)

        if patience_counter >= patience:
            print(f"--> Early stopping triggered at Epoch {epoch}. Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}", flush=True)
            break

    # Restore peak weights
    print(f"Restoring best weights for {model_name} (Lowest Val Loss: {best_val_loss:.4f})...", flush=True)
    model.load_state_dict(best_weights)
    save_training_curves(history, model_name, TRAIN_CURVES_DIR)
    return model

def eval_model(model, model_name, test_loader, device="cuda"):
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
    inf_time_ms = (dt / len(actuals)) * 1000.0
    param_count = count_parameters(model)

    metrics = evaluate_and_save(
        actuals,
        preds,
        probs,
        metas,
        model_name,
        param_count,
        inf_time_ms,
        RESULTS_DIR
    )
    return metrics

def run_part1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_csv = "BioVid-DL/shared/cache/part1_cached_manifest.csv"
    train_loader, val_loader, test_loader = get_dataloaders(manifest_csv, batch_size=8)

    # 1. ResNet-50
    m_resnet = VideoResNet50(num_classes=5, pretrained=True)
    m_resnet = train_model(m_resnet, "ResNet50", train_loader, val_loader, epochs=35, lr=1.2e-4, device=device, accum_steps=4, patience=7)
    metrics_resnet = eval_model(m_resnet, "ResNet50", test_loader, device=device)

    # 2. ViViT
    vivit_cfg = ViViTConfig(num_frames=8, image_size=224, patch_size=16, embed_dim=256, spatial_layers=4, temporal_layers=4, num_heads=4, mlp_dim=512, num_classes=5)
    m_vivit = ViViT(vivit_cfg)
    m_vivit = train_model(m_vivit, "ViViT", train_loader, val_loader, epochs=35, lr=1e-4, device=device, accum_steps=4, patience=7)
    metrics_vivit = eval_model(m_vivit, "ViViT", test_loader, device=device)

    # 3. CNN-LSTM
    m_cnnlstm = VideoCNNLSTM(num_classes=5, hidden_dim=256, pretrained=True)
    m_cnnlstm = train_model(m_cnnlstm, "CNN_LSTM", train_loader, val_loader, epochs=35, lr=1.2e-4, device=device, accum_steps=4, patience=7)
    metrics_cnnlstm = eval_model(m_cnnlstm, "CNN_LSTM", test_loader, device=device)

    # Summary
    comp_df = pd.DataFrame([metrics_resnet, metrics_vivit, metrics_cnnlstm])
    comp_df.to_csv(os.path.join(RESULTS_DIR, "part1_model_comparison.csv"), index=False)
    print("\nPart 1 Model Comparison Summary:", flush=True)
    print(comp_df[["model_name", "accuracy", "macro_f1", "weighted_f1", "mae", "qwk", "inference_time_ms", "parameter_count"]], flush=True)

if __name__ == "__main__":
    run_part1()
