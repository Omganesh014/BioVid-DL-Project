import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error,
)

PAIN_CLASSES = ["BL1", "PA1", "PA2", "PA3", "PA4"]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_training_curves(history, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["train_loss"], 'o-', label="Train Loss", color="#1f77b4", linewidth=2)
    ax.plot(epochs, history["val_loss"], 's--', label="Val Loss", color="#ff7f0e", linewidth=2)
    ax.set_title(f"{model_name}: Training and Validation Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Focal Cross-Entropy Loss", fontsize=11)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model_name}_loss_curve.png"), dpi=300)
    plt.close(fig)

    # 2. Accuracy Curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["train_acc"], 'o-', label="Train Acc", color="#2ca02c", linewidth=2)
    ax.plot(epochs, history["val_acc"], 's--', label="Val Acc", color="#d62728", linewidth=2)
    ax.set_title(f"{model_name}: Training and Validation Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model_name}_accuracy_curve.png"), dpi=300)
    plt.close(fig)

def evaluate_and_save(
    actuals,
    preds,
    probs,
    metas,
    model_name,
    param_count,
    inf_time_ms,
    base_results_dir,
    class_names=PAIN_CLASSES,
):
    graphs_dir = os.path.join(base_results_dir, "graphs")
    cm_dir = os.path.join(base_results_dir, "confusion_matrices")
    metrics_dir = os.path.join(base_results_dir, "metrics")
    preds_dir = os.path.join(base_results_dir, "predictions")

    for d in [graphs_dir, cm_dir, metrics_dir, preds_dir]:
        os.makedirs(d, exist_ok=True)

    acc = accuracy_score(actuals, preds)
    macro_p = precision_score(actuals, preds, average="macro", zero_division=0)
    macro_r = recall_score(actuals, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(actuals, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(actuals, preds, average="weighted", zero_division=0)

    try:
        qwk = cohen_kappa_score(actuals, preds, weights="quadratic")
    except Exception:
        qwk = 0.0

    try:
        mae = mean_absolute_error(actuals, preds)
    except Exception:
        mae = 0.0

    per_class_f1 = f1_score(actuals, preds, average=None, zero_division=0)
    cm = confusion_matrix(actuals, preds, labels=range(len(class_names)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    # Metrics dictionary
    metrics = {
        "model_name": model_name,
        "parameter_count": int(param_count),
        "inference_time_ms": float(inf_time_ms),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "qwk": float(qwk),
        "mae": float(mae),
        "per_class_f1": {name: float(per_class_f1[i]) for i, name in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.tolist(),
    }

    with open(os.path.join(metrics_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV
    pred_records = []
    for i in range(len(actuals)):
        rec = {
            "dataset_part": metas[i].get("dataset_part", ""),
            "subject_name": metas[i].get("subject_name", ""),
            "sample_name": metas[i].get("sample_name", ""),
            "actual_label": int(actuals[i]),
            "actual_class": class_names[actuals[i]] if actuals[i] < len(class_names) else str(actuals[i]),
            "predicted_label": int(preds[i]),
            "predicted_class": class_names[preds[i]] if preds[i] < len(class_names) else str(preds[i]),
            "model_name": model_name,
        }
        for c_idx, c_name in enumerate(class_names):
            rec[f"prob_{c_name}"] = float(probs[i][c_idx])
        pred_records.append(rec)

    df_preds = pd.DataFrame(pred_records)
    df_preds.to_csv(os.path.join(preds_dir, f"{model_name}_predictions.csv"), index=False)

    # 1. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{model_name}: Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(cm_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close(fig)

    # 2. Normalized Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{model_name}: Normalized Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(cm_dir, f"{model_name}_normalized_cm.png"), dpi=300)
    plt.close(fig)

    # 3. Actual vs Predicted Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    act_counts = pd.Series(actuals).value_counts().sort_index()
    prd_counts = pd.Series(preds).value_counts().sort_index()
    idx_labels = [class_names[i] for i in range(len(class_names))]
    x_pos = np.arange(len(idx_labels))
    w = 0.35
    ax.bar(x_pos - w/2, [act_counts.get(i, 0) for i in range(len(class_names))], width=w, label="Actual", color="#2ca02c")
    ax.bar(x_pos + w/2, [prd_counts.get(i, 0) for i in range(len(class_names))], width=w, label="Predicted", color="#1f77b4")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(idx_labels)
    ax.set_title(f"{model_name}: Actual vs Predicted Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, f"{model_name}_actual_vs_predicted.png"), dpi=300)
    plt.close(fig)

    print(f"[{model_name}] Complete — Acc: {acc:.4f} ({acc*100:.2f}%), Macro-F1: {macro_f1:.4f}, MAE: {mae:.3f}, QWK: {qwk:.3f}")
    return metrics
