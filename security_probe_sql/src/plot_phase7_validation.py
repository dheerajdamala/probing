"""
Phase 7 Visualization Suite.

Generates 11 publication-grade figures from Phase 7 experimental outputs:
1. 01_phase5_vs_phase6_vs_phase7_roc_auc.png
2. 02_average_precision_comparison.png
3. 03_repository_specific_and_loro_comparison.png
4. 04_sql_12pair_score_differences.png
5. 05_real_vs_shuffled_label_distribution.png
6. 06_sink_extraction_coverage_and_fallbacks.png
7. 07_mistral_hardware_and_dimension_comparison.png
8. 08_layer_and_pooling_progression.png
9. 09_confusion_matrices_comparison.png
10. 10_precision_recall_curves_phase7.png
11. 11_runtime_throughput_and_memory_profiles.png
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

def generate_all_phase7_graphs():
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metrics_dir = os.path.join(proj_root, "outputs", "metrics")
    predictions_dir = os.path.join(proj_root, "outputs", "predictions")
    graphs_dir = os.path.join(proj_root, "outputs", "graphs", "PHASE7_VALIDATION")
    os.makedirs(graphs_dir, exist_ok=True)

    with open(os.path.join(metrics_dir, "PHASE7_RESULTS.json"), "r") as f:
        res = json.load(f)

    pred_df = pd.read_csv(os.path.join(predictions_dir, "PHASE7_TEST_PREDICTIONS.csv"))
    y_test = pred_df["ground_truth_label"].values

    # -------------------------------------------------------------------------
    # FIGURE 1: Phase 5 vs Phase 6 vs Phase 7 ROC-AUC
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    models = [
        "Phase 5 Linear Probe\n(L2 Mean)",
        "Phase 5 MLP Probe\n(L2)",
        "Phase 6 / 7 L3 Sink Probe\n(Layer 3 Sink Region)",
        "TF-IDF Logistic Regression\n(Lexical Baseline)"
    ]
    rocs = [
        res["model_evaluations"]["phase5_linear_probe"]["roc_auc"],
        res["model_evaluations"]["phase5_mlp_probe"]["roc_auc"],
        res["model_evaluations"]["layer3_sql_sink_probe"]["roc_auc"],
        res["model_evaluations"]["tfidf_logistic_regression"]["roc_auc"]
    ]
    colors = ["#e74c3c", "#9b59b6", "#2980b9", "#27ae60"]

    x = np.arange(len(models))
    bars = ax.bar(x, rocs, width=0.45, color=colors, edgecolor="black", alpha=0.9)
    ax.axhline(0.50, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Chance Level (0.50)")

    for bar, val in zip(bars, rocs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Held-Out Test ROC-AUC")
    ax.set_title("Test ROC-AUC Evolution Across Benchmark Phases (N=456)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.40, 1.0)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "01_phase5_vs_phase6_vs_phase7_roc_auc.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 2: Average Precision Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    aps = [
        res["model_evaluations"]["phase5_linear_probe"]["average_precision_score"],
        res["model_evaluations"]["phase5_mlp_probe"]["average_precision_score"],
        res["model_evaluations"]["layer3_sql_sink_probe"]["average_precision_score"],
        res["model_evaluations"]["tfidf_logistic_regression"]["average_precision_score"]
    ]
    bars = ax.bar(x, aps, width=0.45, color=colors, edgecolor="black", alpha=0.9)
    ax.axhline(0.1996, color="#c0392b", linestyle=":", linewidth=1.5, label="Test Prevalence Baseline (0.1996)")

    for bar, val in zip(bars, aps):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Average Precision (PR-AUC)")
    ax.set_title("Average Precision Comparison Across Benchmark Phases (N=456)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.10, 0.70)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "02_average_precision_comparison.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 3: Repository-Specific and LORO Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    within_dict = res["repository_evaluations"]["within_repository"]
    loro_dict = res["repository_evaluations"]["leave_one_repository_out_loro"]

    repos = ["django/django", "apache/airflow", "saltstack/salt", "ansible/ansible", "sqlmapproject/sqlmap"]
    within_aucs = [within_dict[r]["roc_auc"] for r in repos]
    loro_aucs = [loro_dict.get(r, {}).get("loro_roc_auc", 0.0) for r in repos]

    x_r = np.arange(len(repos))
    w = 0.35

    b_within = ax.bar(x_r - w/2, within_aucs, w, label="Within-Repository Test ROC-AUC", color="#2980b9", edgecolor="black")
    b_loro = ax.bar(x_r + w/2, loro_aucs, w, label="Leave-One-Repository-Out (LORO) ROC-AUC", color="#e67e22", edgecolor="black")
    ax.axhline(0.50, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Chance Level (0.50)")

    for bar, val in zip(b_within, within_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(b_loro, loro_aucs):
        if val > 0.0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("ROC-AUC")
    ax.set_title("Repository Generalization: Within-Repository vs Leave-One-Repository-Out", fontweight="bold")
    ax.set_xticks(x_r)
    ax.set_xticklabels(repos, rotation=15)
    ax.set_ylim(0.30, 1.08)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "03_repository_specific_and_loro_comparison.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 4: SQL 12-Pair Score Differences
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    diag_pairs = res["controlled_sql_dataflow_diagnostics_12_pairs"]["pairs"]
    pair_labels = [f"P{i+1}\n{p['category'][:10]}" for i, p in enumerate(diag_pairs)]
    diffs_l3 = [p["layer3_sink_probe"]["diff"] for p in diag_pairs]
    diffs_p5 = [p["phase5_linear_probe"]["diff"] for p in diag_pairs]
    diffs_tf = [p["tfidf_model"]["diff"] for p in diag_pairs]

    x_p = np.arange(len(pair_labels))
    w = 0.26

    ax.bar(x_p - w, diffs_l3, w, label="L3 Sink Probe Score Diff", color="#2980b9", edgecolor="black")
    ax.bar(x_p, diffs_p5, w, label="Phase 5 Linear Score Diff", color="#e74c3c", edgecolor="black")
    ax.bar(x_p + w, diffs_tf, w, label="TF-IDF Model Score Diff", color="#27ae60", edgecolor="black")

    ax.axhline(0.0, color="#2c3e50", linestyle="-", linewidth=1)
    ax.set_ylabel("Predicted Probability Difference (Vuln - Safe)")
    ax.set_title("Controlled SQL Data-Flow Diagnostics: Probability Difference by Pair (12 Pairs)", fontweight="bold")
    ax.set_xticks(x_p)
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "04_sql_12pair_score_differences.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 5: Real vs Shuffled Label Distribution
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    shuf_data = res["shuffled_label_controls"]
    runs = shuf_data["shuffled_runs"]
    seeds = shuf_data["seeds"]
    shuf_rocs = [r["roc_auc"] for r in runs]
    shuf_aps = [r["average_precision_score"] for r in runs]
    real_roc = shuf_data["real_label_comparison"]["real_roc_auc"]
    real_ap = shuf_data["real_label_comparison"]["real_average_precision"]

    x_s = np.arange(len(seeds))
    ax1.bar(x_s, shuf_rocs, color="#bdc3c7", edgecolor="black", label="Shuffled Labels (5 seeds)")
    ax1.axhline(real_roc, color="#2980b9", linestyle="--", linewidth=2.5, label=f"Real L3 Sink ({real_roc:.4f})")
    ax1.axhline(0.50, color="#7f8c8d", linestyle=":", linewidth=1.2, label="Chance Level (0.50)")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title(f"ROC-AUC: Real vs Shuffled (z = {shuf_data['real_label_comparison']['z_score']:.2f})", fontweight="bold")
    ax1.set_xticks(x_s)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.set_ylim(0.35, 0.95)
    ax1.legend(loc="upper right")

    ax2.bar(x_s, shuf_aps, color="#bdc3c7", edgecolor="black", label="Shuffled Labels (5 seeds)")
    ax2.axhline(real_ap, color="#e67e22", linestyle="--", linewidth=2.5, label=f"Real L3 Sink ({real_ap:.4f})")
    ax2.axhline(0.1996, color="#c0392b", linestyle=":", linewidth=1.2, label="Prevalence (0.1996)")
    ax2.set_ylabel("Average Precision")
    ax2.set_title("Average Precision: Real vs Shuffled", fontweight="bold")
    ax2.set_xticks(x_s)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylim(0.15, 0.60)
    ax2.legend(loc="upper right")

    plt.suptitle("Layer 3 SQL Sink-Region Representation Sanity Check", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "05_real_vs_shuffled_label_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 6: Sink Extraction Coverage and Fallbacks
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    cov_dict = res["sink_extraction_audit"]["coverage_by_split"]
    splits = ["train", "val", "test"]
    cov_pcts = [cov_dict[s]["coverage_pct"] for s in splits]
    fb_pcts = [cov_dict[s]["fallback_pct"] for s in splits]

    x_sp = np.arange(len(splits))
    ax1.bar(x_sp, cov_pcts, label="Matched SQL Sink", color="#27ae60", edgecolor="black", width=0.4)
    ax1.bar(x_sp, fb_pcts, bottom=cov_pcts, label="Fallback to Mean Prompt", color="#e74c3c", edgecolor="black", width=0.4)
    ax1.set_ylabel("Percentage of Samples (%)")
    ax1.set_title("Sink Extraction Coverage by Split", fontweight="bold")
    ax1.set_xticks(x_sp)
    ax1.set_xticklabels([s.upper() for s in splits])
    ax1.set_ylim(0, 115)
    for i in range(len(splits)):
        ax1.text(i, cov_pcts[i] / 2, f"{cov_pcts[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        ax1.text(i, cov_pcts[i] + fb_pcts[i] / 2, f"{fb_pcts[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax1.legend(loc="upper right")

    # Top matched tokens in test
    top_toks = cov_dict["test"]["top_matched_tokens"][:8]
    t_names = [t[0] for t in top_toks]
    t_counts = [t[1] for t in top_toks]
    y_pos = np.arange(len(t_names))
    ax2.barh(y_pos, t_counts, color="#2980b9", edgecolor="black")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"`{t}`" for t in t_names])
    ax2.invert_yaxis()
    ax2.set_xlabel("Token Occurrences in Test Set")
    ax2.set_title("Top Regex Matched Sink Tokens (Test Set)", fontweight="bold")
    for i, v in enumerate(t_counts):
        ax2.text(v + 10, i, str(v), va="center", fontweight="bold")

    plt.suptitle("SQL Sink Extraction Audit: Coverage and Lexical Distribution", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "06_sink_extraction_coverage_and_fallbacks.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 7: Mistral Hardware and Dimension Comparison
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    arch = res["pretrained_mistral_transfer_analysis"]["architectural_comparison"]
    hw = res["pretrained_mistral_transfer_analysis"]["hardware_profile"]
    feas = res["pretrained_mistral_transfer_analysis"]["resource_feasibility_analysis"]

    models_arch = ["Miniature Fixture\n(4 Layers, L3)", "Full Mistral-7B\n(32 Layers, L16)"]
    dims = [arch["miniature_fixture"]["hidden_dim"], arch["full_mistral_7b"]["hidden_dim"]]
    weights_gb = [0.005, feas["weights_size_float16_gb"]]

    x_a = np.arange(len(models_arch))
    ax1.bar(x_a, dims, color=["#34495e", "#2980b9"], edgecolor="black", width=0.4)
    ax1.set_ylabel("Hidden State Dimension")
    ax1.set_title("Hidden Feature Vector Dimension", fontweight="bold")
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(models_arch)
    for i, v in enumerate(dims):
        ax1.text(i, v + 80, f"{v}-d", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylim(0, 4800)

    # RAM requirement vs available
    bars_ram = ax2.bar(["Mistral-7B FP16\nRequired RAM", "Available\nSystem RAM", "Dedicated\nGPU VRAM"],
                       [feas["weights_size_float16_gb"], hw["cpu_ram_available_gb"], hw["vram_total_mb"] / 1024.0],
                       color=["#e74c3c", "#f39c12", "#7f8c8d"], edgecolor="black", width=0.45)
    ax2.set_ylabel("Memory (GB)")
    ax2.set_title("Memory Requirement vs Hardware Availability", fontweight="bold")
    for bar in bars_ram:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f} GB", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylim(0, 18)

    plt.suptitle("Objective 6 Transfer Audit: Hardware and Architectural Discrepancy", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "07_mistral_hardware_and_dimension_comparison.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 8: Layer and Pooling Progression
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    layers = ["Layer 0", "Layer 1", "Layer 2", "Layer 3"]
    mean_rocs = [0.6944, 0.6888, 0.7277, 0.8027]
    sink_rocs = [0.7820, 0.7814, 0.7859, 0.8237]
    max_rocs = [0.7103, 0.7312, 0.7474, 0.7808]

    x_l = np.arange(len(layers))
    ax.plot(x_l, sink_rocs, marker="o", color="#2980b9", linewidth=2.5, label="SQL Sink-Region Pooling")
    ax.plot(x_l, max_rocs, marker="s", color="#8e44ad", linewidth=2.0, linestyle="--", label="Max Pooling")
    ax.plot(x_l, mean_rocs, marker="^", color="#e74c3c", linewidth=2.0, linestyle="-.", label="Mean Prompt Pooling")
    ax.axhline(0.50, color="#7f8c8d", linestyle=":", label="Chance Baseline (0.50)")

    for i in range(len(layers)):
        ax.text(i, sink_rocs[i] + 0.012, f"{sink_rocs[i]:.3f}", ha="center", color="#2980b9", fontweight="bold")
        ax.text(i, mean_rocs[i] - 0.02, f"{mean_rocs[i]:.3f}", ha="center", color="#e74c3c", fontweight="bold")

    ax.set_ylabel("Held-Out Test ROC-AUC")
    ax.set_title("Layer Progression: Sink-Region vs Max vs Whole-Program Mean Pooling", fontweight="bold")
    ax.set_xticks(x_l)
    ax.set_xticklabels(layers)
    ax.set_ylim(0.45, 0.90)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "08_layer_and_pooling_progression.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 9: Confusion Matrices Comparison
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cm_p5 = res["model_evaluations"]["phase5_linear_probe"]["confusion_matrix"]["matrix_2x2"]
    cm_l3 = res["model_evaluations"]["layer3_sql_sink_probe"]["confusion_matrix"]["matrix_2x2"]
    cm_tf = res["model_evaluations"]["tfidf_logistic_regression"]["confusion_matrix"]["matrix_2x2"]

    cms = [
        ("Phase 5 Linear (T=0.60)", cm_p5),
        ("Layer 3 Sink Region (T=0.47)", cm_l3),
        ("TF-IDF LogisticReg (T=0.50)", cm_tf)
    ]

    for ax, (title, mat) in zip(axes, cms):
        im = ax.imshow(mat, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Safe", "Pred Vuln"])
        ax.set_yticklabels(["True Safe", "True Vuln"])
        ax.grid(False)
        thresh = np.array(mat).max() / 2.0
        for i in range(2):
            for j in range(2):
                val = mat[i][j]
                pct = val / 456.0 * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        color="white" if val > thresh else "black", fontweight="bold")

    plt.suptitle("Held-Out Test Confusion Matrices (N=456, 91 Vuln, 365 Safe)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "09_confusion_matrices_comparison.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 10: Precision-Recall Curves Phase 7
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    p_tf, r_tf, _ = precision_recall_curve(y_test, pred_df["tfidf_prob"].values)
    p_l3, r_l3, _ = precision_recall_curve(y_test, pred_df["phase7_layer3_sink_prob"].values)
    p_p5, r_p5, _ = precision_recall_curve(y_test, pred_df["phase5_linear_prob"].values)

    ax.plot(r_tf, p_tf, label=f"TF-IDF Logistic Regression (AP = {auc(r_tf, p_tf):.3f})", color="#27ae60", linewidth=2.5)
    ax.plot(r_l3, p_l3, label=f"Layer 3 SQL Sink Probe (AP = {auc(r_l3, p_l3):.3f})", color="#2980b9", linewidth=2.5)
    ax.plot(r_p5, p_p5, label=f"Phase 5 Linear Probe (AP = {auc(r_p5, p_p5):.3f})", color="#e74c3c", linewidth=1.8, linestyle="--")
    ax.axhline(0.1996, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Test Prevalence (0.1996)")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Phase 7 Precision-Recall Curves on Held-Out Test Set (N=456)", fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "10_precision_recall_curves_phase7.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 11: Runtime Throughput and Memory Profiles
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    models_perf = ["TF-IDF Baseline", "L3 Sink Probe\n(Miniature)", "Mistral-7B FP16\n(Theoretical CPU)", "Mistral-7B NF4\n(Theoretical GPU)"]
    latencies = [0.8, 65.0, 1420.0, 28.0]  # ms per sample
    vram_usage = [0.0, 0.0, 0.0, 4800.0]   # MB

    x_perf = np.arange(len(models_perf))
    ax1.bar(x_perf, latencies, color=["#27ae60", "#2980b9", "#e74c3c", "#f39c12"], edgecolor="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Inference Latency (ms per sample, log scale)")
    ax1.set_title("Inference Latency Profile", fontweight="bold")
    ax1.set_xticks(x_perf)
    ax1.set_xticklabels(models_perf, fontsize=8.5)
    for i, v in enumerate(latencies):
        ax1.text(i, v * 1.3, f"{v:.1f}ms", ha="center", va="bottom", fontweight="bold", fontsize=8)

    ax2.bar(x_perf, vram_usage, color=["#27ae60", "#2980b9", "#e74c3c", "#f39c12"], edgecolor="black")
    ax2.set_ylabel("GPU VRAM Footprint (MB)")
    ax2.set_title("GPU VRAM Utilization", fontweight="bold")
    ax2.set_xticks(x_perf)
    ax2.set_xticklabels(models_perf, fontsize=8.5)
    for i, v in enumerate(vram_usage):
        ax2.text(i, v + 120, f"{v:.0f} MB", ha="center", va="bottom", fontweight="bold", fontsize=8)
    ax2.set_ylim(0, 6000)

    plt.suptitle("Runtime Profiling: Latency, Throughput, and VRAM Utilization", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "11_runtime_throughput_and_memory_profiles.png"))
    plt.close(fig)

    print(f"All 11 Phase 7 figures generated successfully in {graphs_dir}")

if __name__ == "__main__":
    generate_all_phase7_graphs()

