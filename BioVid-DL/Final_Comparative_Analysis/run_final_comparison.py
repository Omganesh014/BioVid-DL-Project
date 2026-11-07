import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FINAL_DIR = "BioVid-DL/Final_Comparative_Analysis"
GRAPHS_DIR = os.path.join(FINAL_DIR, "graphs")
METRICS_DIR = os.path.join(FINAL_DIR, "metrics")
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def run_comparative_synthesis():
    print("Synthesizing Final Comparative Research Benchmarks across all experiments...", flush=True)

    # 1. Load Part 1 Pain Metrics
    p1_file = "BioVid-DL/Part_1_Pain_Recognition/results/part1_model_comparison.csv"
    df_p1 = pd.read_csv(p1_file) if os.path.exists(p1_file) else pd.DataFrame()

    # 2. Load Part 2 Emotion Metrics
    p2_file = "BioVid-DL/Part_2_Emotion_Recognition/results/part2_emotion_model_comparison.csv"
    df_p2 = pd.read_csv(p2_file) if os.path.exists(p2_file) else pd.DataFrame()

    # 3. Load Part 3 Multimodal Ablation Metrics
    p3_file = "BioVid-DL/Part_3_Multimodal_Recognition/results/ablation_studies/modality_ablation_metrics.csv"
    df_p3 = pd.read_csv(p3_file) if os.path.exists(p3_file) else pd.DataFrame()

    # Assemble Pain Models Comparison: ResNet50, ViViT, CNN_LSTM, Multimodal_CNN
    pain_models = []
    if not df_p1.empty:
        for _, r in df_p1.iterrows():
            pain_models.append(r.to_dict())

    if not df_p3.empty:
        # Multimodal fusion row
        fusion_row = df_p3[df_p3["model_name"] == "Complete_Multimodal_Fusion"]
        if not fusion_row.empty:
            d = fusion_row.iloc[0].to_dict()
            d["model_name"] = "Multimodal_CNN"
            pain_models.append(d)

    df_pain_comp = pd.DataFrame(pain_models)
    df_pain_comp.to_csv(os.path.join(METRICS_DIR, "final_pain_models_comparison.csv"), index=False)

    # Assemble Emotion Models Comparison
    df_p2.to_csv(os.path.join(METRICS_DIR, "final_emotion_models_comparison.csv"), index=False)

    # ---------------- GENERATE PUBLICATION COMPARATIVE GRAPHS ----------------
    # 1. Pain Recognition Accuracy, Precision, Recall, Macro-F1 Comparison
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    metrics_list = [("accuracy", "Accuracy", ax1 := axes[0, 0], "Blues_d"),
                    ("macro_precision", "Macro Precision", axes[0, 1], "Greens_d"),
                    ("macro_recall", "Macro Recall", axes[1, 0], "Oranges_d"),
                    ("macro_f1", "Macro-F1 Score", axes[1, 1], "Purples_d")]

    for m_col, m_title, ax, pal in metrics_list:
        sns.barplot(data=df_pain_comp, x="model_name", y=m_col, ax=ax, palette=pal)
        ax.set_title(f"Pain Models: {m_title} Comparison", fontsize=12, fontweight="bold")
        ax.set_xlabel("Architecture", fontsize=10)
        ax.set_ylabel(m_title, fontsize=10)
        ax.tick_params(axis='x', rotation=25)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "pain_models_comprehensive_metrics.png"), dpi=300)
    plt.close(fig)

    # 2. Emotion Models Comparison: Part D & Part E
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_p2, x="model_name", y="accuracy", ax=ax1, palette="viridis")
    ax1.set_title("Emotion Recognition: Accuracy Comparison", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Model & Protocol", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.tick_params(axis='x', rotation=30)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight="bold")

    sns.barplot(data=df_p2, x="model_name", y="macro_f1", ax=ax2, palette="magma")
    ax2.set_title("Emotion Recognition: Macro-F1 Comparison", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Model & Protocol", fontsize=10)
    ax2.set_ylabel("Macro-F1", fontsize=10)
    ax2.tick_params(axis='x', rotation=30)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "emotion_models_comparison.png"), dpi=300)
    plt.close(fig)

    # 3. Video-Only vs Complete Multimodal Performance (Key Thesis Validation)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not df_p3.empty:
        comp_v_m = df_p3[df_p3["model_name"].isin(["Video_Only", "All_Biosignals_Only", "Complete_Multimodal_Fusion"])]
        sns.barplot(data=comp_v_m, x="model_name", y="accuracy", ax=ax, palette=["#4e79a7", "#59a14f", "#e15759"])
        ax.set_title("Primary Research Hypothesis: Unimodal vs Multimodal Fusion", fontsize=13, fontweight="bold")
        ax.set_xlabel("Input Modality Paradigm", fontsize=11)
        ax.set_ylabel("Pain Classification Accuracy", fontsize=11)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(GRAPHS_DIR, "video_only_vs_multimodal_performance.png"), dpi=300)
        plt.close(fig)

    # 4. Latency vs Parameter Count Scatter
    all_models = []
    for df, task in [(df_pain_comp, "Pain"), (df_p2, "Emotion")]:
        if not df.empty:
            for _, r in df.iterrows():
                all_models.append({
                    "model_name": r["model_name"],
                    "parameters_m": r["parameter_count"] / 1e6,
                    "inference_time_ms": r["inference_time_ms"],
                    "accuracy": r["accuracy"],
                    "task": task
                })
    df_all = pd.DataFrame(all_models)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(data=df_all, x="parameters_m", y="inference_time_ms", hue="task", size="accuracy",
                    sizes=(100, 400), ax=ax, palette=["#1f77b4", "#ff7f0e"])
    ax.set_title("Computational Efficiency: Model Parameters vs Inference Latency (RTX 4050)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trainable Parameters (Millions)", fontsize=11)
    ax.set_ylabel("Inference Time per Video Clip (ms)", fontsize=11)
    for _, r in df_all.iterrows():
        ax.annotate(r["model_name"], (r["parameters_m"], r["inference_time_ms"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "latency_vs_parameters.png"), dpi=300)
    plt.close(fig)

    print("Final comparative figures generated and saved successfully.", flush=True)

if __name__ == "__main__":
    run_comparative_synthesis()
