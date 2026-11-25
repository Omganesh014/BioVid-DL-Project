import os
import sys
import time
import copy
import json
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.models.video as vm
import timm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error,
)

sys.path.append("BioVid-DL/Part_1_Pain_Recognition/models")
sys.path.append("BioVid-DL/Part_1_Pain_Recognition/utils")
sys.path.append("BioVid-DL/Part_2_Emotion_Recognition/models/3d_cnn")
sys.path.append("BioVid-DL/Part_3_Multimodal_Recognition/models/multimodal_cnn")

from resnet50.resnet50 import VideoResNet50
from cnn_lstm.cnn_lstm import VideoCNNLSTM
from vivit.vivit import ViViT, ViViTConfig
from r3d import VideoR3D18
from multimodal_cnn import MultimodalCNN
from evaluator import evaluate_and_save, save_training_curves, count_parameters

EPOCHS = 56
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_ROOT = "BioVid-DL/shared/cache/retrain_56ep"

PAIN_BIN_CLASSES = ["BL1", "PA4"]
PART_D_CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "pain", "sad"]
PART_E_CLASSES = ["amusement", "anger", "disgust", "fear", "sad"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FastJpgNpzDataset(Dataset):
    def __init__(self, manifest_csv: str, split: str, is_training: bool = False, out_size: int = 224):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.is_training = is_training
        self.out_size = out_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["cache_path"], allow_pickle=True)
        jpgs = data["jpgs"]
        frames = []
        for raw_bytes in jpgs:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                rgb = np.zeros((self.out_size, self.out_size, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if self.out_size != rgb.shape[0]:
                    rgb = cv2.resize(rgb, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)
        vid = np.stack(frames, axis=0)  # [T, H, W, 3]

        if self.is_training:
            if np.random.rand() > 0.5:
                vid = np.flip(vid, axis=2)

        vid_t = vid.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        vid_t = (vid_t - IMAGENET_MEAN) / IMAGENET_STD

        if self.is_training:
            vid_t = vid_t * np.random.uniform(0.92, 1.08)

        bio = data["bio"].astype(np.float32)
        if self.is_training:
            bio = bio + np.random.normal(0, 0.015, bio.shape).astype(np.float32)

        meta = {
            "dataset_part": str(row["dataset_part"]),
            "subject_name": str(row["subject_name"]),
            "sample_name": str(row["sample_name"]),
        }
        return (
            torch.from_numpy(vid_t.copy()).float(),
            torch.from_numpy(bio.copy()).float(),
            torch.tensor(int(row["class_id"]), dtype=torch.long),
            meta,
        )


def build_pretrained_vivit(num_classes: int = 2) -> ViViT:
    cfg = ViViTConfig(
        num_frames=8,
        image_size=224,
        patch_size=16,
        tubelet_t=1,
        embed_dim=384,
        spatial_layers=12,
        temporal_layers=2,
        num_heads=6,
        mlp_dim=1536,
        dropout=0.1,
        num_classes=num_classes,
    )
    model = ViViT(cfg)
    vit = timm.create_model("vit_small_patch16_224", pretrained=True)
    with torch.no_grad():
        model.tubelet_embed.projection.weight.copy_(vit.patch_embed.proj.weight.unsqueeze(2))
        model.tubelet_embed.projection.bias.copy_(vit.patch_embed.proj.bias)
        model.spatial_cls_token.copy_(vit.cls_token)
        model.spatial_pos_embed.copy_(vit.pos_embed)
        for i in range(12):
            vb = vit.blocks[i]
            mb = model.spatial_blocks[i]
            mb.norm1.weight.copy_(vb.norm1.weight)
            mb.norm1.bias.copy_(vb.norm1.bias)
            mb.attn.qkv.weight.copy_(vb.attn.qkv.weight)
            mb.attn.qkv.bias.copy_(vb.attn.qkv.bias)
            mb.attn.proj.weight.copy_(vb.attn.proj.weight)
            mb.attn.proj.bias.copy_(vb.attn.proj.bias)
            mb.norm2.weight.copy_(vb.norm2.weight)
            mb.norm2.bias.copy_(vb.norm2.bias)
            mb.mlp[0].weight.copy_(vb.mlp.fc1.weight)
            mb.mlp[0].bias.copy_(vb.mlp.fc1.bias)
            mb.mlp[3].weight.copy_(vb.mlp.fc2.weight)
            mb.mlp[3].bias.copy_(vb.mlp.fc2.bias)
        model.spatial_norm.weight.copy_(vit.norm.weight)
        model.spatial_norm.bias.copy_(vit.norm.bias)
    return model


class PretrainedVideoSwin(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.backbone = vm.swin3d_t(weights=vm.Swin3D_T_Weights.DEFAULT)
        in_f = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> [B, C, T, H, W]
        return self.backbone(x.permute(0, 2, 1, 3, 4))


def get_cosine_with_warmup(optimizer, total_epochs: int, warmup_epochs: int = 3):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = float(ep - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_56_epochs(
    model,
    model_name: str,
    train_loader,
    val_loader,
    results_dir: str,
    lr: float = 1.5e-4,
    is_multimodal: bool = False,
    task: str = "pain",
    bio_channels: int = 5,
    accum_steps: int = 2,
):
    print(f"\n{'='*72}\nTraining {model_name} for {EPOCHS} epochs on {DEVICE} (lr={lr}, accum={accum_steps})\n{'='*72}", flush=True)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Differential learning rate for pretrained CNN/ViT backbones vs new heads
    if hasattr(model, "feature_extractor"):
        params = [
            {"params": model.feature_extractor.parameters(), "lr": lr * 0.25},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    elif hasattr(model, "cnn") and hasattr(model, "lstm"):
        params = [
            {"params": model.cnn.parameters(), "lr": lr * 0.25},
            {"params": model.lstm.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    elif hasattr(model, "spatial_blocks"):
        spatial_p = (
            list(model.tubelet_embed.parameters())
            + [model.spatial_cls_token, model.spatial_pos_embed]
            + list(model.spatial_blocks.parameters())
            + list(model.spatial_norm.parameters())
        )
        temporal_p = (
            [model.temporal_cls_token, model.temporal_pos_embed]
            + list(model.temporal_blocks.parameters())
            + list(model.temporal_norm.parameters())
            + list(model.head.parameters())
        )
        params = [
            {"params": spatial_p, "lr": lr * 0.2},
            {"params": temporal_p, "lr": lr},
        ]
    elif hasattr(model, "video_cnn"):
        params = [
            {"params": model.video_cnn.parameters(), "lr": lr * 0.25},
            {"params": model.bio_cnn.parameters(), "lr": lr},
            {"params": model.fusion.parameters(), "lr": lr},
            {"params": model.pain_head.parameters(), "lr": lr},
            {"params": model.emotion_head.parameters(), "lr": lr},
            {"params": model.video_only_head.parameters(), "lr": lr},
            {"params": model.bio_only_head.parameters(), "lr": lr},
        ]
    else:
        params = [{"params": model.parameters(), "lr": lr}]

    optimizer = AdamW(params, weight_decay=1e-2)
    scheduler = get_cosine_with_warmup(optimizer, EPOCHS, warmup_epochs=3)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = os.path.join(results_dir, "checkpoints")
    curves_dir = os.path.join(results_dir, "training_curves")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_corr = tr_tot = 0
        optimizer.zero_grad(set_to_none=True)
        n_steps = len(train_loader)
        for step_i, (videos, bios, labels, _) in enumerate(train_loader):
            videos = videos.to(DEVICE, non_blocking=True)
            bios = bios[:, :bio_channels, :].to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if is_multimodal:
                    out = model(video=videos, bio=bios, task=task, modality_mode="multimodal")
                    loss = criterion(out, labels)
                    if task == "pain":
                        out_v = model(video=videos, bio=bios, task=task, modality_mode="video_only")
                        out_b = model(video=videos, bio=bios, task=task, modality_mode="bio_only")
                        loss = loss + 0.5 * criterion(out_v, labels) + 0.5 * criterion(out_b, labels)
                else:
                    out = model(videos)
                    loss = criterion(out, labels)
                scaled_loss = loss / accum_steps
            scaler.scale(scaled_loss).backward()
            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == n_steps:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tr_loss += loss.item() * labels.size(0)
            tr_corr += (out.argmax(1) == labels).sum().item()
            tr_tot += labels.size(0)

        scheduler.step()
        model.eval()
        va_loss = va_corr = va_tot = 0
        with torch.no_grad():
            for videos, bios, labels, _ in val_loader:
                videos = videos.to(DEVICE, non_blocking=True)
                bios = bios[:, :bio_channels, :].to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    if is_multimodal:
                        out = model(video=videos, bio=bios, task=task, modality_mode="multimodal")
                    else:
                        out = model(videos)
                    loss = criterion(out, labels)
                va_loss += loss.item() * labels.size(0)
                va_corr += (out.argmax(1) == labels).sum().item()
                va_tot += labels.size(0)

        ep_tr_loss = tr_loss / tr_tot
        ep_tr_acc = tr_corr / tr_tot
        ep_va_loss = va_loss / va_tot
        ep_va_acc = va_corr / va_tot
        history["train_loss"].append(ep_tr_loss)
        history["train_acc"].append(ep_tr_acc)
        history["val_loss"].append(ep_va_loss)
        history["val_acc"].append(ep_va_acc)

        mark = ""
        if (ep_va_acc > best_val_acc) or (ep_va_acc == best_val_acc and ep_va_loss < best_val_loss):
            best_val_acc = ep_va_acc
            best_val_loss = ep_va_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(ckpt_dir, f"{model_name}_best.pt"))
            mark = " *"

        if epoch == 1 or epoch % 7 == 0 or epoch == EPOCHS:
            print(
                f"  Epoch [{epoch:02d}/{EPOCHS}] ({time.time()-t0:.1f}s) | "
                f"TrLoss={ep_tr_loss:.4f} TrAcc={ep_tr_acc*100:.2f}% | "
                f"VaLoss={ep_va_loss:.4f} VaAcc={ep_va_acc*100:.2f}%{mark}",
                flush=True,
            )

    model.load_state_dict(best_weights)
    save_training_curves(history, model_name, curves_dir)
    print(f"Finished {model_name}: Best Val Acc = {best_val_acc*100:.2f}%", flush=True)
    return model, history


def evaluate_model(
    model,
    model_name: str,
    test_loader,
    results_dir: str,
    class_names: list,
    is_multimodal: bool = False,
    task: str = "pain",
    mode: str = "multimodal",
    bio_channel_mask=None,
    bio_channels: int = 5,
):
    model.eval()
    actuals, preds, probs, metas = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for videos, bios, labels, meta in test_loader:
            videos = videos.to(DEVICE, non_blocking=True)
            bios = bios[:, :bio_channels, :].to(DEVICE, non_blocking=True)
            if bio_channel_mask is not None:
                m = torch.tensor(bio_channel_mask, device=DEVICE).view(1, -1, 1).float()
                bios = bios * m
            with torch.amp.autocast("cuda"):
                if is_multimodal:
                    out = model(video=videos, bio=bios, task=task, modality_mode=mode)
                else:
                    out = model(videos)
                p = torch.softmax(out, dim=1)
            pr = out.argmax(1)
            actuals.extend(labels.numpy().tolist())
            preds.extend(pr.cpu().numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())
            for i in range(len(labels)):
                metas.append({
                    "dataset_part": meta["dataset_part"][i],
                    "subject_name": meta["subject_name"][i],
                    "sample_name": meta["sample_name"][i],
                })
    inf_ms = ((time.time() - t0) / max(1, len(actuals))) * 1000.0
    pcount = count_parameters(model)
    metrics = evaluate_and_save(
        actuals, preds, probs, metas, model_name, pcount, inf_ms, results_dir, class_names=class_names
    )
    return metrics, actuals, preds


def main():
    set_seed(42)
    print(f"Running 56-Epoch Full Retraining Pipeline on {DEVICE} ({torch.cuda.get_device_name(0)})", flush=True)

    # ==========================================================================
    # PART 1: PAIN RECOGNITION (Part A BL1 vs PA4, 3,480 clips)
    # ==========================================================================
    pa_csv = os.path.join(CACHE_ROOT, "part_a_manifest.csv")
    tr_a = DataLoader(FastJpgNpzDataset(pa_csv, "train", True, 224), batch_size=8, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_a = DataLoader(FastJpgNpzDataset(pa_csv, "val", False, 224), batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    te_a = DataLoader(FastJpgNpzDataset(pa_csv, "test", False, 224), batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    p1_dir = "BioVid-DL/Part_1_Pain_Recognition/results"

    m_res = VideoResNet50(num_classes=2, pretrained=True, dropout=0.3)
    m_res, _ = train_56_epochs(m_res, "ResNet50", tr_a, va_a, p1_dir, lr=2e-4)
    met_res, _, _ = evaluate_model(m_res, "ResNet50", te_a, p1_dir, PAIN_BIN_CLASSES)
    del m_res; torch.cuda.empty_cache()

    m_vivit = build_pretrained_vivit(num_classes=2)
    m_vivit, hist_vivit = train_56_epochs(m_vivit, "ViViT", tr_a, va_a, p1_dir, lr=1.5e-4)
    met_vivit, y_true_a, y_pred_a = evaluate_model(m_vivit, "ViViT", te_a, p1_dir, PAIN_BIN_CLASSES)
    del m_vivit; torch.cuda.empty_cache()

    m_lstm = VideoCNNLSTM(num_classes=2, hidden_dim=256, pretrained=True, dropout=0.3)
    m_lstm, _ = train_56_epochs(m_lstm, "CNN_LSTM", tr_a, va_a, p1_dir, lr=2e-4)
    met_lstm, _, _ = evaluate_model(m_lstm, "CNN_LSTM", te_a, p1_dir, PAIN_BIN_CLASSES)
    del m_lstm; torch.cuda.empty_cache()

    pd.DataFrame([met_res, met_vivit, met_lstm]).to_csv(os.path.join(p1_dir, "part1_model_comparison.csv"), index=False)

    # ==========================================================================
    # PART 3: MULTIMODAL CNN ON PART A PAIN + 6 ABLATIONS
    # ==========================================================================
    p3_dir = "BioVid-DL/Part_3_Multimodal_Recognition/results"
    os.makedirs(os.path.join(p3_dir, "ablation_studies"), exist_ok=True)
    m_mm_a = MultimodalCNN(num_bio_channels=5, num_pain_classes=2, num_emotion_classes=7, pretrained=True, dropout=0.3)
    m_mm_a, hist_mm_a = train_56_epochs(m_mm_a, "Multimodal_CNN", tr_a, va_a, p3_dir, lr=2e-4, is_multimodal=True, task="pain", bio_channels=5)

    abl_configs = [
        ("Complete Fusion", "Video + GSR + ECG + EMG (5-ch)", "Complete_Multimodal_Fusion", "multimodal", None),
        ("Video Only", "Video only (Biosignals zeroed)", "Video_Only", "video_only", None),
        ("All Biosignals", "GSR + ECG + EMG (Video zeroed)", "All_Biosignals_Only", "bio_only", None),
        ("GSR Only", "Galvanic Skin Response only", "GSR_Only", "bio_only", [1.0, 0.0, 0.0, 0.0, 0.0]),
        ("ECG Only", "Electrocardiogram only", "ECG_Only", "bio_only", [0.0, 1.0, 0.0, 0.0, 0.0]),
        ("EMG Only", "Electromyography (Trap/Corr/Zyg)", "EMG_Only", "bio_only", [0.0, 0.0, 1.0, 1.0, 1.0]),
    ]
    abl_rows = []
    met_mm_a = None
    for cname, ch_desc, fn, mode, mask in abl_configs:
        m_res_abl, _, _ = evaluate_model(
            m_mm_a, fn, te_a, p3_dir, PAIN_BIN_CLASSES,
            is_multimodal=True, task="pain", mode=mode, bio_channel_mask=mask, bio_channels=5
        )
        if cname == "Complete Fusion":
            met_mm_a = copy.deepcopy(m_res_abl)
            met_mm_a["model_name"] = "Multimodal_CNN"
        abl_rows.append({
            "Configuration": cname,
            "Modality Channels": ch_desc,
            "Accuracy": m_res_abl["accuracy"],
            "Macro_Precision": m_res_abl["macro_precision"],
            "Macro_Recall": m_res_abl["macro_recall"],
            "Macro_F1": m_res_abl["macro_f1"],
            "Weighted_F1": m_res_abl["weighted_f1"],
            "MAE": m_res_abl["mae"],
            "QWK": m_res_abl["qwk"],
        })
    pd.DataFrame(abl_rows).to_csv(os.path.join(p3_dir, "ablation_studies/multimodal_ablation_summary.csv"), index=False)
    del m_mm_a; torch.cuda.empty_cache()

    # ==========================================================================
    # PART 2: EMOTION RECOGNITION (Part D 7-Class & Part E 5-Class)
    # ==========================================================================
    p2_dir = "BioVid-DL/Part_2_Emotion_Recognition/results"
    pd_csv = os.path.join(CACHE_ROOT, "part_d_manifest.csv")
    pe_csv = os.path.join(CACHE_ROOT, "part_e_manifest.csv")

    tr_d = DataLoader(FastJpgNpzDataset(pd_csv, "train", True, 112), batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_d = DataLoader(FastJpgNpzDataset(pd_csv, "val", False, 112), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    te_d = DataLoader(FastJpgNpzDataset(pd_csv, "test", False, 112), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    tr_e = DataLoader(FastJpgNpzDataset(pe_csv, "train", True, 112), batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_e = DataLoader(FastJpgNpzDataset(pe_csv, "val", False, 112), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    te_e = DataLoader(FastJpgNpzDataset(pe_csv, "test", False, 112), batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Part D: R3D-18, Video Swin, Multimodal CNN
    m_r3d_d = VideoR3D18(num_classes=7, pretrained=True, dropout=0.3)
    m_r3d_d, _ = train_56_epochs(m_r3d_d, "PartD_3D_CNN_R3D18", tr_d, va_d, p2_dir, lr=2e-4, accum_steps=1)
    met_r3d_d, _, _ = evaluate_model(m_r3d_d, "PartD_3D_CNN_R3D18", te_d, p2_dir, PART_D_CLASSES)
    del m_r3d_d; torch.cuda.empty_cache()

    m_swin_d = PretrainedVideoSwin(num_classes=7)
    m_swin_d, _ = train_56_epochs(m_swin_d, "PartD_Video_Swin", tr_d, va_d, p2_dir, lr=1.5e-4, accum_steps=1)
    met_swin_d, _, _ = evaluate_model(m_swin_d, "PartD_Video_Swin", te_d, p2_dir, PART_D_CLASSES)
    del m_swin_d; torch.cuda.empty_cache()

    m_mm_d = MultimodalCNN(num_bio_channels=5, num_pain_classes=2, num_emotion_classes=7, pretrained=True, dropout=0.3)
    m_mm_d, _ = train_56_epochs(m_mm_d, "PartD_Multimodal_CNN", tr_d, va_d, p2_dir, lr=2e-4, is_multimodal=True, task="emotion", bio_channels=5, accum_steps=1)
    met_mm_d, _, _ = evaluate_model(m_mm_d, "PartD_Multimodal_CNN", te_d, p2_dir, PART_D_CLASSES, is_multimodal=True, task="emotion", bio_channels=5)
    del m_mm_d; torch.cuda.empty_cache()

    # Part E: R3D-18, Video Swin, Multimodal CNN
    m_r3d_e = VideoR3D18(num_classes=5, pretrained=True, dropout=0.3)
    m_r3d_e, _ = train_56_epochs(m_r3d_e, "PartE_3D_CNN_R3D18", tr_e, va_e, p2_dir, lr=2e-4, accum_steps=1)
    met_r3d_e, _, _ = evaluate_model(m_r3d_e, "PartE_3D_CNN_R3D18", te_e, p2_dir, PART_E_CLASSES)
    del m_r3d_e; torch.cuda.empty_cache()

    m_swin_e = PretrainedVideoSwin(num_classes=5)
    m_swin_e, _ = train_56_epochs(m_swin_e, "PartE_Video_Swin", tr_e, va_e, p2_dir, lr=1.5e-4, accum_steps=1)
    met_swin_e, _, _ = evaluate_model(m_swin_e, "PartE_Video_Swin", te_e, p2_dir, PART_E_CLASSES)
    del m_swin_e; torch.cuda.empty_cache()

    m_mm_e = MultimodalCNN(num_bio_channels=3, num_pain_classes=2, num_emotion_classes=5, pretrained=True, dropout=0.3)
    m_mm_e, _ = train_56_epochs(m_mm_e, "PartE_Multimodal_CNN", tr_e, va_e, p2_dir, lr=2e-4, is_multimodal=True, task="emotion", bio_channels=3, accum_steps=1)
    met_mm_e, _, _ = evaluate_model(m_mm_e, "PartE_Multimodal_CNN", te_e, p2_dir, PART_E_CLASSES, is_multimodal=True, task="emotion", bio_channels=3)
    del m_mm_e; torch.cuda.empty_cache()

    p2_comp = pd.DataFrame([met_r3d_d, met_swin_d, met_r3d_e, met_swin_e])
    p2_comp.to_csv(os.path.join(p2_dir, "part2_emotion_model_comparison.csv"), index=False)

    # ==========================================================================
    # FINAL COMPARATIVE ANALYSIS & OUTPUTS LOGS
    # ==========================================================================
    final_dir = "BioVid-DL/Final_Comparative_Analysis/metrics"
    os.makedirs(final_dir, exist_ok=True)
    pd.DataFrame([met_res, met_vivit, met_lstm, met_mm_a]).to_csv(os.path.join(final_dir, "final_pain_models_comparison.csv"), index=False)
    p2_comp.to_csv(os.path.join(final_dir, "final_emotion_models_comparison.csv"), index=False)

    # Summary JSON for all 6 models across tasks
    summary_56 = {
        "epochs": EPOCHS,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "part1_pain_BL1_vs_PA4": {
            "ResNet50": met_res["accuracy"],
            "ViViT": met_vivit["accuracy"],
            "CNN_LSTM": met_lstm["accuracy"],
            "Multimodal_CNN": met_mm_a["accuracy"],
        },
        "part2_emotion_PartD_7cls": {
            "3D_CNN_R3D18": met_r3d_d["accuracy"],
            "Video_Swin": met_swin_d["accuracy"],
            "Multimodal_CNN": met_mm_d["accuracy"],
        },
        "part2_emotion_PartE_5cls": {
            "3D_CNN_R3D18": met_r3d_e["accuracy"],
            "Video_Swin": met_swin_e["accuracy"],
            "Multimodal_CNN": met_mm_e["accuracy"],
        },
        "full_metrics": {
            "ResNet50": met_res,
            "ViViT": met_vivit,
            "CNN_LSTM": met_lstm,
            "Multimodal_CNN_PartA": met_mm_a,
            "PartD_3D_CNN_R3D18": met_r3d_d,
            "PartD_Video_Swin": met_swin_d,
            "PartD_Multimodal_CNN": met_mm_d,
            "PartE_3D_CNN_R3D18": met_r3d_e,
            "PartE_Video_Swin": met_swin_e,
            "PartE_Multimodal_CNN": met_mm_e,
        },
    }
    with open(os.path.join(final_dir, "retrain_56ep_summary.json"), "w") as f:
        json.dump(summary_56, f, indent=2)

    # Update outputs/logs/training_history.csv & test_metrics.json with real ViViT/Multimodal 56-ep runs
    hist_df = pd.DataFrame({
        "epoch": list(range(1, EPOCHS + 1)),
        "train_loss": [round(x, 4) for x in hist_mm_a["train_loss"]],
        "train_acc": [round(x, 4) for x in hist_mm_a["train_acc"]],
        "val_loss": [round(x, 4) for x in hist_mm_a["val_loss"]],
        "val_acc": [round(x, 4) for x in hist_mm_a["val_acc"]],
    })
    os.makedirs("outputs/logs", exist_ok=True)
    hist_df.to_csv("outputs/logs/training_history.csv", index=False)

    print("\n" + "=" * 78, flush=True)
    print("ALL 6 MODELS 56-EPOCH REAL RETRAINING COMPLETE!", flush=True)
    print("=" * 78, flush=True)
    print("Part A Pain (BL1 vs PA4):", summary_56["part1_pain_BL1_vs_PA4"], flush=True)
    print("Part D Emotion (7-Cls) :", summary_56["part2_emotion_PartD_7cls"], flush=True)
    print("Part E Emotion (5-Cls) :", summary_56["part2_emotion_PartE_5cls"], flush=True)


if __name__ == "__main__":
    main()
