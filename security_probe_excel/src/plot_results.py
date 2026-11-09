"""
Plotting script for publication-grade graphs of the Excel-compatible SQL Injection Probe benchmark.
Generates ROC curves, PR curves, Confusion Matrices, and Stratified Performance charts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_all():
    base_dir = r"C:\Users\dheer\Desktop\isae\security_probe_excel"
    metrics_path = os.path.join(base_dir, "outputs", "metrics", "EXCEL_PROBE_RESULTS.json")
    pred_path = os.path.join(base_dir, "outputs", "predictions", "test_predictions.csv")
    graphs_dir = os.path.join(base_dir, "outputs", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
        print(f"Results not found yet: {metrics_path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    df = pd.read_csv(pred_path)
    y_true = df["y_true"].values
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for model_col, name, color in [
        ("prob_sink", "Model B: SQL Sink Probe", "#1f77b4"),
        ("prob_tfidf", "Baseline E: TF-IDF Logistic Regression", "#2ca02c")
    ]:
        if model_col in df.columns:
            fpr, tpr, _ = roc_curve(y_true, df[model_col].values)
            auc = metrics["model_evaluations"]["linear_probe_sql_sink_pooling"]["roc_auc"] if "sink" in model_col else metrics["model_evaluations"]["tfidf_logistic_regression"]["roc_auc"]
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, lw=2)
            
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("ROC Curves: SQL Injection Probe vs Baselines", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "01_roc_curves_comparison.png"))
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for model_col, name, color in [
        ("prob_sink", "Model B: SQL Sink Probe", "#1f77b4"),
        ("prob_tfidf", "Baseline E: TF-IDF Logistic Regression", "#2ca02c")
    ]:
        if model_col in df.columns:
            prec, rec, _ = precision_recall_curve(y_true, df[model_col].values)
            ap = metrics["model_evaluations"]["linear_probe_sql_sink_pooling"]["average_precision"] if "sink" in model_col else metrics["model_evaluations"]["tfidf_logistic_regression"]["average_precision"]
            plt.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", color=color, lw=2)
            
    prev = float(np.mean(y_true))
    plt.axhline(prev, color="red", linestyle="--", label=f"Positive Prevalence ({prev:.3f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves: SQL Injection Detection", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "02_precision_recall_curves.png"))
    plt.close()
    
    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for idx, (m_key, title) in enumerate([
        ("linear_probe_sql_sink_pooling", "Model B: SQL Sink Probe"),
        ("tfidf_logistic_regression", "Baseline E: TF-IDF")
    ]):
        cm_dict = metrics["model_evaluations"][m_key]["confusion_matrix"]
        cm_matrix = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
        sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=["Safe", "Vulnerable"], yticklabels=["Safe", "Vulnerable"])
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)
    plt.suptitle("Test Set Confusion Matrices (N = 706)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "03_confusion_matrices.png"))
    plt.close()
    
    # 4. Stratified: Case Type
    ct_data = metrics.get("stratified_evaluations", {}).get("case_type", {})
    if ct_data:
        ct_df = pd.DataFrame(ct_data).T.reset_index().rename(columns={"index": "case_type"})
        plt.figure(figsize=(10, 5), dpi=300)
        sns.barplot(data=ct_df, x="case_type", y="roc_auc", palette="Blues_d")
        plt.title("Probe Discrimination (ROC-AUC) by Case Type", fontsize=13, fontweight="bold")
        plt.ylabel("ROC-AUC", fontsize=11)
        plt.xlabel("Case Type", fontsize=11)
        plt.ylim(0, 1.05)
        for i, row in ct_df.iterrows():
            plt.text(i, row["roc_auc"] + 0.02, f"{row['roc_auc']:.2f}\n(N={int(row['count'])})", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "04_stratified_performance_by_case_type.png"))
        plt.close()
        
    # 5. Stratified: DB API
    db_data = metrics.get("stratified_evaluations", {}).get("db_api", {})
    if db_data:
        db_df = pd.DataFrame(db_data).T.reset_index().rename(columns={"index": "db_api"})
        plt.figure(figsize=(10, 5), dpi=300)
        sns.barplot(data=db_df, x="db_api", y="roc_auc", palette="Greens_d")
        plt.title("Probe Discrimination (ROC-AUC) Across Database APIs", fontsize=13, fontweight="bold")
        plt.ylabel("ROC-AUC", fontsize=11)
        plt.xlabel("Database API", fontsize=11)
        plt.ylim(0, 1.05)
        for i, row in db_df.iterrows():
            plt.text(i, row["roc_auc"] + 0.02, f"{row['roc_auc']:.2f}\n(N={int(row['count'])})", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "05_stratified_performance_by_db_api.png"))
        plt.close()
        
    # 6. Stratified: Query Style
    qs_data = metrics.get("stratified_evaluations", {}).get("query_style", {})
    if qs_data:
        qs_df = pd.DataFrame(qs_data).T.reset_index().rename(columns={"index": "query_style"})
        plt.figure(figsize=(10, 5), dpi=300)
        sns.barplot(data=qs_df, x="query_style", y="roc_auc", palette="Purples_d")
        plt.title("Probe Discrimination (ROC-AUC) Across Query Construction Styles", fontsize=13, fontweight="bold")
        plt.ylabel("ROC-AUC", fontsize=11)
        plt.xlabel("Query Style", fontsize=11)
        plt.xticks(rotation=25)
        plt.ylim(0, 1.05)
        for i, row in qs_df.iterrows():
            plt.text(i, row["roc_auc"] + 0.02, f"{row['roc_auc']:.2f}\n(N={int(row['count'])})", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "06_stratified_performance_by_query_style.png"))
        plt.close()
        
    print(f"Generated all 6 publication figures in: {graphs_dir}")

if __name__ == "__main__":
    plot_all()

