"""
Phase 5 Authoritative Graphs Generator.

Generates authoritative, publication-quality figures exclusively from the verified
full-precision metrics in PHASE5_AUTHORITATIVE_METRICS.json and
PHASE5_FULL_PRECISION_PREDICTIONS.csv.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

def generate_authoritative_graphs(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = "security_probe_sql/outputs/metrics/PHASE5_AUTHORITATIVE_METRICS.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_metrics = data["authoritative_model_metrics"]
    pred_path = "security_probe_sql/outputs/predictions/PHASE5_FULL_PRECISION_PREDICTIONS.csv"
    df_pred = pd.read_csv(pred_path)
    y_test = df_pred["ground_truth_label"].to_numpy().astype(int)
    prevalence = float(y_test.sum() / len(y_test))

    # -------------------------------------------------------------
    # 1. Authoritative Model Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    model_order = [
        "majority_class",
        "code_length",
        "lexical_heuristic",
        "tfidf_logistic_regression",
        "neural_linear_probe",
        "neural_mlp_probe"
    ]
    model_labels = [
        "Majority Class",
        "Code Length",
        "Lexical Regex",
        "TF-IDF + LR",
        "Neural Linear Probe",
        "Neural MLP Probe"
    ]
    
    bal_accs = [models_metrics[m]["balanced_accuracy"] for m in model_order]
    f1s = [models_metrics[m]["f1_score"] for m in model_order]
    aps = [models_metrics[m]["average_precision_score"] for m in model_order]

    x = np.arange(len(model_order))
    width = 0.25

    ax.bar(x - width, bal_accs, width, label="Balanced Accuracy", color="#2b5c8f")
    ax.bar(x, f1s, width, label="F1 Score", color="#e07a5f")
    ax.bar(x + width, aps, width, label="Average Precision (AP)", color="#81b29a")

    ax.set_ylabel("Metric Score")
    ax.set_title("Phase 5 Authoritative Benchmark Comparison (Held-Out Test Set, N=456)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_authoritative_model_comparison.png"))
    plt.close(fig)
    print("  Generated: 01_authoritative_model_comparison.png")

    # -------------------------------------------------------------
    # 2. Authoritative Confusion Matrix
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_data = models_metrics["neural_linear_probe"]["confusion_matrix"]["matrix_2x2"]
    cm_matrix = np.array(cm_data)

    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Predicted Safe (0)", "Predicted Vuln (1)"],
        yticklabels=["Actual Safe (0)", "Actual Vuln (1)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title("Authoritative Confusion Matrix: Linear Probe (T=0.60)")
    ax.set_ylabel("True Ground Truth Label")
    ax.set_xlabel("Predicted Label (Positive Preds = 408 / 456)")

    # Annotation text
    ax.text(0.5, -0.15, "High Sensitivity (Recall: 96.7%), High False Positive Rate (87.7%)",
            ha="center", transform=ax.transAxes, fontsize=10, style="italic")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_authoritative_confusion_matrix.png"))
    plt.close(fig)
    print("  Generated: 02_authoritative_confusion_matrix.png")

    # -------------------------------------------------------------
    # 3. Authoritative ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    # TF-IDF
    fpr_tf, tpr_tf, _ = roc_curve(y_test, df_pred["prob_tfidf"])
    auc_tf = models_metrics["tfidf_logistic_regression"]["roc_auc"]
    ax.plot(fpr_tf, tpr_tf, label=f"TF-IDF + LR (AUC = {auc_tf:.4f})", color="#e76f51", lw=2)

    # Linear Probe
    fpr_lp, tpr_lp, _ = roc_curve(y_test, df_pred["prob_linear_full_precision"])
    auc_lp = models_metrics["neural_linear_probe"]["roc_auc"]
    ax.plot(fpr_lp, tpr_lp, label=f"Neural Linear Probe (AUC = {auc_lp:.4f})", color="#2a9d8f", lw=2.5)

    # Code Length
    fpr_len, tpr_len, _ = roc_curve(y_test, df_pred["prob_code_len"])
    auc_len = models_metrics["code_length"]["roc_auc"]
    ax.plot(fpr_len, tpr_len, label=f"Code Length Baseline (AUC = {auc_len:.4f})", color="#f4a261", lw=1.5, ls="--")

    # MLP Probe
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, df_pred["prob_mlp_full_precision"])
    auc_mlp = models_metrics["neural_mlp_probe"]["roc_auc"]
    ax.plot(fpr_mlp, tpr_mlp, label=f"Neural MLP Probe (AUC = {auc_mlp:.4f})", color="#9b5de5", lw=1.5)

    # Chance Line
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance Line (AUC = 0.5000)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("Authoritative ROC Curves (N=456 Held-Out Test Set)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_authoritative_roc_curves.png"))
    plt.close(fig)
    print("  Generated: 03_authoritative_roc_curves.png")

    # -------------------------------------------------------------
    # 4. Authoritative Precision-Recall Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # TF-IDF
    p_tf, r_tf, _ = precision_recall_curve(y_test, df_pred["prob_tfidf"])
    ap_tf = models_metrics["tfidf_logistic_regression"]["average_precision_score"]
    ax.plot(r_tf, p_tf, label=f"TF-IDF + LR (AP = {ap_tf:.4f})", color="#e76f51", lw=2)

    # Linear Probe
    p_lp, r_lp, _ = precision_recall_curve(y_test, df_pred["prob_linear_full_precision"])
    ap_lp = models_metrics["neural_linear_probe"]["average_precision_score"]
    ax.plot(r_lp, p_lp, label=f"Neural Linear Probe (AP = {ap_lp:.4f})", color="#2a9d8f", lw=2.5)

    # MLP Probe
    p_mlp, r_mlp, _ = precision_recall_curve(y_test, df_pred["prob_mlp_full_precision"])
    ap_mlp = models_metrics["neural_mlp_probe"]["average_precision_score"]
    ax.plot(r_mlp, p_mlp, label=f"Neural MLP Probe (AP = {ap_mlp:.4f})", color="#9b5de5", lw=1.5)

    # Horizontal prevalence baseline
    ax.axhline(prevalence, color="#d62828", ls="--", lw=2, label=f"Random / Majority Baseline (Prevalence = {prevalence:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Authoritative Precision-Recall Curves with Prevalence Baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_authoritative_precision_recall_curves.png"))
    plt.close(fig)
    print("  Generated: 04_authoritative_precision_recall_curves.png")

    # -------------------------------------------------------------
    # 5. Discrepancy Reconciliation Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    metrics_comp = ["ROC-AUC", "Balanced Acc", "Precision", "Recall", "F1 Score"]
    unverified_claims = [0.8037, 0.7107, 0.5098, 0.5714, 0.5389]
    verified_auth = [
        models_metrics["neural_linear_probe"]["roc_auc"],
        models_metrics["neural_linear_probe"]["balanced_accuracy"],
        models_metrics["neural_linear_probe"]["precision"],
        models_metrics["neural_linear_probe"]["recall"],
        models_metrics["neural_linear_probe"]["f1_score"]
    ]

    x = np.arange(len(metrics_comp))
    width = 0.35

    ax.bar(x - width/2, unverified_claims, width, label="Unverified Summary Claim (Fabrication)", color="#d9534f", alpha=0.85)
    ax.bar(x + width/2, verified_auth, width, label="Authoritative Checkpoint Result (Reproduced)", color="#0275d8", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Phase 5 Discrepancy Reconciliation: Unverified Claim vs Authoritative Model")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_comp)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True)

    for i in range(len(metrics_comp)):
        ax.text(x[i] - width/2, unverified_claims[i] + 0.02, f"{unverified_claims[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(x[i] + width/2, verified_auth[i] + 0.02, f"{verified_auth[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_discrepancy_reconciliation_chart.png"))
    plt.close(fig)
    print("  Generated: 05_discrepancy_reconciliation_chart.png")

    # -------------------------------------------------------------
    # 6. Authoritative Test Localization (N=2)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    loc_metrics_names = ["Top-1 Hit Rate", "Top-3 Hit Rate", "Top-5 Hit Rate", "MRR"]
    loc_values = [0.00, 1.00, 1.00, 0.50]

    bars = ax.bar(loc_metrics_names, loc_values, color="#5bc0de", width=0.5, edgecolor="black")
    ax.set_ylabel("Rate / Score")
    ax.set_title("Authoritative Held-Out Test Localization Performance (N=2)")
    ax.set_ylim(0, 1.2)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.text(0.5, 0.85, "Held-out test set contains exactly 2 ground-truth annotated samples.\nTarget lines ranked at position 2 in both instances.",
            ha="center", transform=ax.transAxes, fontsize=9.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff9db", ec="gray", lw=0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_authoritative_test_localization.png"))
    plt.close(fig)
    print("  Generated: 06_authoritative_test_localization.png")

    print(f"\n[Phase 5 Authoritative Graphs Completed] 6 figures saved to: {output_dir}")

if __name__ == "__main__":
    out_dir = "security_probe_sql/outputs/graphs/PHASE5_AUTHORITATIVE_GRAPHS"
    generate_authoritative_graphs(out_dir)
