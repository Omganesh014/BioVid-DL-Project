import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

PRED_DIR_P1 = "BioVid-DL/Part_1_Pain_Recognition/results/predictions"
PRED_DIR_P3 = "BioVid-DL/Part_3_Multimodal_Recognition/results/predictions"
OUTPUT_METRICS_DIR = "BioVid-DL/Final_Comparative_Analysis/metrics"
OUTPUT_GRAPHS_DIR = "BioVid-DL/Final_Comparative_Analysis/graphs"

os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
os.makedirs(OUTPUT_GRAPHS_DIR, exist_ok=True)

models = {
    "Multimodal CNN (Complete Fusion)": os.path.join(PRED_DIR_P3, "Complete_Multimodal_Fusion_predictions.csv"),
    "All Biosignals Only": os.path.join(PRED_DIR_P3, "All_Biosignals_Only_predictions.csv"),
    "Video Only Baseline": os.path.join(PRED_DIR_P3, "Video_Only_predictions.csv"),
    "ResNet-50": os.path.join(PRED_DIR_P1, "ResNet50_predictions.csv"),
    "ViViT": os.path.join(PRED_DIR_P1, "ViViT_predictions.csv"),
    "CNN-LSTM": os.path.join(PRED_DIR_P1, "CNN_LSTM_predictions.csv")
}

records = []
plt.figure(figsize=(8, 6), dpi=300)

for name, path in models.items():
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    # Filter for BL1 (No Pain) vs PA4 (Severe Pain)
    df_b = df[df["actual_class"].isin(["BL1", "PA4"])].copy()
    y_true = (df_b["actual_class"] == "PA4").astype(int).values
    
    # Normalized score between PA4 and BL1
    p_pa4 = df_b["prob_PA4"].values
    p_bl1 = df_b["prob_BL1"].values
    y_score = p_pa4 / (p_pa4 + p_bl1 + 1e-8)
    y_pred = (y_score >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    records.append({
        "model_name": name,
        "binary_task": "BL1 (No Pain) vs PA4 (Severe Pain)",
        "accuracy": acc,
        "precision": prec,
        "recall_severe_pain": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": str(cm)
    })
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
plt.ylabel("True Positive Rate (Severe Pain Recall)", fontsize=11, fontweight="bold")
plt.title("ROC Curves: Clinical Severe Pain Alert (BL1 No Pain vs PA4 Severe Pain)", fontsize=12, fontweight="bold", pad=12)
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_GRAPHS_DIR, "binary_pain_roc_curve.png")
plt.savefig(roc_path)
plt.close()

df_binary = pd.DataFrame(records)
csv_path = os.path.join(OUTPUT_METRICS_DIR, "binary_pain_alert_comparison.csv")
df_binary.to_csv(csv_path, index=False)
print("Binary benchmark metrics saved:", csv_path)
print("ROC Curve saved:", roc_path)

# Generate Confusion Matrix for Multimodal CNN Binary Alert
df_mm = pd.read_csv(models["Multimodal CNN (Complete Fusion)"])
df_mm_b = df_mm[df_mm["actual_class"].isin(["BL1", "PA4"])].copy()
y_true_mm = (df_mm_b["actual_class"] == "PA4").astype(int).values
y_score_mm = df_mm_b["prob_PA4"].values / (df_mm_b["prob_PA4"].values + df_mm_b["prob_BL1"].values + 1e-8)
y_pred_mm = (y_score_mm >= 0.5).astype(int)
cm_mm = confusion_matrix(y_true_mm, y_pred_mm)

plt.figure(figsize=(6, 5), dpi=300)
sns.heatmap(cm_mm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted No Pain", "Predicted Severe Pain"],
            yticklabels=["Actual No Pain (BL1)", "Actual Severe Pain (PA4)"],
            annot_kws={"size": 14, "weight": "bold"})
plt.title("Multimodal CNN: Clinical Emergency Confusion Matrix\n(Accuracy: 65.0% | Severe Pain Recall: 75.0% | AUC: 70.4%)", fontsize=11, fontweight="bold", pad=12)
plt.ylabel("True Clinical State", fontsize=10, fontweight="bold")
plt.xlabel("Predicted Emergency Alert", fontsize=10, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_GRAPHS_DIR, "binary_pain_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print("Binary Confusion Matrix saved:", cm_path)
