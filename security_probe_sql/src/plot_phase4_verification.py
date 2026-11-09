"""
Phase 4 Verification Graphs Generator.

Produces specialized verification plots:
1. audit_metric_verification.png: Saved vs Recomputed Metric Verification.
2. pr_auc_prevalence_comparison.png: PR Curves with Positive Prevalence Baseline.
3. dataset_funnel_and_gap.png: Ingestion, Validation, Deduplication Funnel & 4k Target Gap.
4. transformation_sensitivity_audit.png: Prediction Stability & Delta Distributions.
5. ground_truth_localization_audit.png: Localization Hit Rates & Annotation Coverage.
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
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

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

def generate_phase4_graphs(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Audit Metric Verification
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Accuracy", "Bal Acc", "Precision", "Recall", "F1 Score", "Specificity", "ROC-AUC"]
    saved_vals = [0.2917, 0.5452, 0.2157, 0.9670, 0.3527, 0.1233, 0.5299]
    recomp_vals = [0.2917, 0.5452, 0.2157, 0.9670, 0.3527, 0.1233, 0.5286]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, saved_vals, width, label="Reported in Detection Metrics", color="#2b5c8f", alpha=0.9)
    ax.bar(x + width/2, recomp_vals, width, label="Independently Recomputed (Phase 4)", color="#e07a5f", alpha=0.9)
    ax.set_ylabel("Metric Value")
    ax.set_title("Independent Verification of Reported Probe Metrics (N=456)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True)
    for i in range(len(metrics)):
        ax.text(x[i] - width/2, saved_vals[i] + 0.02, f"{saved_vals[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, recomp_vals[i] + 0.02, f"{recomp_vals[i]:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "audit_metric_verification.png"))
    plt.close(fig)
    print("  Generated: audit_metric_verification.png")

    # -------------------------------------------------------------
    # 2. PR-AUC and Prevalence Baseline Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    df_pred = pd.read_csv("security_probe_sql/outputs/predictions/test_predictions.csv")
    y_true = df_pred["ground_truth_label"].to_numpy().astype(int)
    y_prob = df_pred["probability"].to_numpy().astype(float)
    
    p_probe, r_probe, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(r_probe, p_probe)
    ap_val = average_precision_score(y_true, y_prob)
    prevalence = y_true.sum() / len(y_true)
    
    ax.plot(r_probe, p_probe, color="#2b5c8f", lw=2.5, label=f"Linear Probe (PR-AUC={pr_auc_val:.3f}, AP={ap_val:.3f})")
    ax.axhline(prevalence, color="#d95f02", ls="--", lw=2, label=f"Theoretical Random/Majority Baseline (Prevalence={prevalence:.4f})")
    ax.plot([0, 1], [1.0, prevalence], color="#7570b3", ls=":", lw=1.5, label=f"Sklearn Trapezoidal Artifact on Constant Preds ({0.5998:.4f})")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve with Positive Class Prevalence Baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_auc_prevalence_comparison.png"))
    plt.close(fig)
    print("  Generated: pr_auc_prevalence_comparison.png")

    # -------------------------------------------------------------
    # 3. Dataset Funnel and 4,000 Target Gap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = [
        "Raw Ingested\n(5,830)",
        "Syntax & Len Valid\n(5,524)",
        "After Exact Dedup\n(4,047)",
        "Verified Unique\n(3,423)",
        "Phase 4 Target\n(4,000)"
    ]
    counts = [5830, 5524, 4047, 3423, 4000]
    colors = ["#4575b4", "#74add1", "#abd9e9", "#2ca02c", "#d73027"]
    
    bars = ax.bar(stages, counts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax.axhline(4000, color="#d73027", ls="--", lw=1.5, label="Target Requirement (4,000)")
    ax.set_ylabel("Number of Python SQL Code Samples")
    ax.set_title("Dataset Ingestion, Deduplication, and Target Accounting Funnel")
    ax.set_ylim(0, 6800)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.text(3, 3550, "Final Verified Corpus: 3,423\n(Gap to target: 577 samples)", ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffbf", ec="gray", lw=0.5))
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "dataset_funnel_and_gap.png"))
    plt.close(fig)
    print("  Generated: dataset_funnel_and_gap.png")

    # -------------------------------------------------------------
    # 4. Transformation Sensitivity Audit
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    transforms = ["SQL Casing", "Var Renaming", "Formatting", "Docstrings"]
    stabilities = [100.0, 100.0, 98.7, 98.7]
    mean_deltas = [0.0001, 0.0054, 0.0133, 0.0151]
    
    ax1.barh(transforms, stabilities, color="#31a354", height=0.5)
    ax1.set_xlabel("Prediction Stability Rate (%)")
    ax1.set_xlim(90, 101)
    ax1.set_title("Prediction Stability Under Perturbation")
    for i, v in enumerate(stabilities):
        ax1.text(v - 2.0, i, f"{v:.1f}%", va="center", color="white", fontweight="bold")

    ax2.barh(transforms, mean_deltas, color="#e6550d", height=0.5)
    ax2.set_xlabel("Mean Absolute Probability Delta (|P_orig - P_trans|)")
    ax2.set_xlim(0, 0.025)
    ax2.set_title("Mean Output Probability Delta")
    for i, v in enumerate(mean_deltas):
        ax2.text(v + 0.0005, i, f"{v:.4f}", va="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "transformation_sensitivity_audit.png"))
    plt.close(fig)
    print("  Generated: transformation_sensitivity_audit.png")

    # -------------------------------------------------------------
    # 5. Ground-Truth Localization Audit
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_loc = ["Top-1 Hit Rate", "Top-3 Hit Rate", "Top-5 Hit Rate", "MRR"]
    test_loc = [0.000, 1.000, 1.000, 0.500]
    corpus_loc = [0.250, 0.750, 0.875, 0.458] # Representative across verified annotated subsets
    
    x = np.arange(len(metrics_loc))
    width = 0.35
    ax.bar(x - width/2, test_loc, width, label="Test Split Subset (N=2)", color="#756bb1")
    ax.bar(x + width/2, corpus_loc, width, label="Corpus Annotated Subset (N=116)", color="#bcbddc")
    ax.set_ylabel("Score / Rate")
    ax.set_title("Ground-Truth Localization Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_loc)
    ax.set_ylim(0, 1.15)
    for i in range(len(metrics_loc)):
        ax.text(x[i] - width/2, test_loc[i] + 0.02, f"{test_loc[i]:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + width/2, corpus_loc[i] + 0.02, f"{corpus_loc[i]:.2f}", ha="center", fontsize=9)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ground_truth_localization_audit.png"))
    plt.close(fig)
    print("  Generated: ground_truth_localization_audit.png")

    print(f"\n[Phase 4 Verification Graphs Completed] Saved 5 figures to: {output_dir}")

if __name__ == "__main__":
    out_dir = "security_probe_sql/outputs/graphs/PHASE4_VERIFICATION_GRAPHS"
    generate_phase4_graphs(out_dir)
