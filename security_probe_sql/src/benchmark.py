"""
Benchmarking and Baselines Evaluation Module for Python SQL Injection Security Probe.

Evaluates the trained probe against 5 baselines on the held-out test partition:
1. Majority-Class Baseline
2. Code-Length Baseline
3. TF-IDF + Logistic Regression Baseline
4. Lexical SQL Injection Heuristic Baseline
5. Neural Linear Representation Probe (Primary)
6. Neural MLP Representation Probe (Comparison)

Saves:
- outputs/metrics/detection_metrics.json
- outputs/metrics/localization_metrics.json
- outputs/predictions/test_predictions.csv
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
import yaml
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.model.load_model import load_model_and_tokenizer
from security_probe_sql.src.feature_extractor import extract_code_representations
from security_probe_sql.src.probe_model import SecurityProbe
from security_probe_sql.src.inference import SQLInjectionProbeAnalyzer
from security_probe_sql.src.evaluation import (
    compute_detection_metrics,
    compute_localization_metrics,
)


def load_dataset_records(filepath: str) -> List[Dict[str, Any]]:
    """Loads records from jsonl file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# Lexical regex heuristic for SQL injection patterns
LEXICAL_SQLI_REGEX = re.compile(
    r'(execute\s*\(\s*f["\']|'
    r'execute\s*\(\s*["\'].*?%s.*?["\']\s*%\s*|'
    r'execute\s*\(\s*["\'].*?\{.*?["\']\.format|'
    r'execute\s*\(\s*["\'].*?\+\s*\w+|'
    r'raw\s*\(\s*f["\']|'
    r'raw\s*\(\s*["\'].*?%s.*?["\']\s*%\s*|'
    r'select\s+.*?WHERE.*?["\']\s*\+\s*\w+|'
    r'select\s+.*?WHERE.*?f["\'])',
    re.IGNORECASE
)


def run_benchmark_suite(config_path: str) -> Dict[str, Any]:
    """Runs all 6 models/baselines on identical held-out test data."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, cfg["outputs"]["models_dir"])
    metrics_dir = os.path.join(project_root, cfg["outputs"]["metrics_dir"])
    pred_dir = os.path.join(project_root, cfg["outputs"]["predictions_dir"])
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # 1. Load Data Splits
    train_path = os.path.join(project_root, cfg["data"]["train_split"])
    val_path = os.path.join(project_root, cfg["data"]["val_split"])
    test_path = os.path.join(project_root, cfg["data"]["test_split"])

    train_records = load_dataset_records(train_path)
    val_records = load_dataset_records(val_path)
    test_records = load_dataset_records(test_path)

    print(f"\n[Benchmark Suite] Evaluating on {len(test_records)} held-out test samples...")

    y_train = np.array([r["label"] for r in train_records], dtype=int)
    y_val = np.array([r["label"] for r in val_records], dtype=int)
    y_test = np.array([r["label"] for r in test_records], dtype=int)

    train_codes = [r["code"] for r in train_records]
    val_codes = [r["code"] for r in val_records]
    test_codes = [r["code"] for r in test_records]

    results: Dict[str, Any] = {}

    # ==========================================
    # Baseline 1: Majority Class Baseline
    # ==========================================
    print("Evaluating Baseline 1: Majority Class...")
    majority_label = int(np.bincount(y_train).argmax())
    pred_maj = np.full_like(y_test, majority_label)
    prob_maj = np.full(len(y_test), float(majority_label))
    results["majority_class"] = compute_detection_metrics(y_test, pred_maj, prob_maj)

    # ==========================================
    # Baseline 2: Code Length Baseline
    # ==========================================
    print("Evaluating Baseline 2: Code Length...")
    len_train = np.array([[len(c)] for c in train_codes], dtype=float)
    len_val = np.array([[len(c)] for c in val_codes], dtype=float)
    len_test = np.array([[len(c)] for c in test_codes], dtype=float)

    len_clf = LogisticRegression(class_weight="balanced", random_state=42)
    len_clf.fit(len_train, y_train)
    prob_len = len_clf.predict_proba(len_test)[:, 1]
    pred_len = len_clf.predict(len_test)
    results["code_length"] = compute_detection_metrics(y_test, pred_len, prob_len)

    # ==========================================
    # Baseline 3: TF-IDF + Logistic Regression
    # ==========================================
    print("Evaluating Baseline 3: TF-IDF + Logistic Regression...")
    tfidf = TfidfVectorizer(max_features=2500, token_pattern=r'(?u)\b\w+\b|[^\w\s]')
    X_tfidf_tr = tfidf.fit_transform(train_codes)
    X_tfidf_te = tfidf.transform(test_codes)

    tfidf_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    tfidf_clf.fit(X_tfidf_tr, y_train)
    prob_tfidf = tfidf_clf.predict_proba(X_tfidf_te)[:, 1]
    pred_tfidf = tfidf_clf.predict(X_tfidf_te)
    results["tfidf_logistic_regression"] = compute_detection_metrics(y_test, pred_tfidf, prob_tfidf)

    # ==========================================
    # Baseline 4: Lexical SQL Injection Heuristic
    # ==========================================
    print("Evaluating Baseline 4: Lexical SQLi Heuristic...")
    pred_lexical = np.array([1 if LEXICAL_SQLI_REGEX.search(c) else 0 for c in test_codes], dtype=int)
    prob_lexical = pred_lexical.astype(float)
    results["lexical_heuristic"] = compute_detection_metrics(y_test, pred_lexical, prob_lexical)

    # ==========================================
    # Baseline 5 & 6: Neural Representation Probes
    # ==========================================
    is_mock = cfg["model"].get("mock", True)
    target_layer = cfg["model"]["mock_target_layer"] if is_mock else cfg["model"]["target_layer"]
    model, tokenizer, _ = load_model_and_tokenizer(mock=is_mock)

    # Extract or load test features
    test_feat_cache = os.path.join(models_dir, f"test_features_layer_{target_layer}.npz")
    if os.path.exists(test_feat_cache):
        print(f"Loading cached test features from {test_feat_cache}...")
        test_npz = np.load(test_feat_cache)
        X_test = test_npz["X"]
    else:
        print(f"Extracting test representations (layer {target_layer})...")
        X_test_list = []
        for idx, r in enumerate(test_records):
            rep_dict = extract_code_representations(
                model=model,
                tokenizer=tokenizer,
                code=r["code"],
                layer_idx=target_layer,
                pooling="mean_prompt",
            )
            X_test_list.append(rep_dict["pooled_rep"])
        X_test = np.array(X_test_list, dtype=np.float32)
        np.savez_compressed(test_feat_cache, X=X_test, y=y_test)

    # Load fitted probes
    linear_path = os.path.join(models_dir, "probe_head_linear.pkl")
    mlp_path = os.path.join(models_dir, "probe_head_mlp.pkl")

    probe_linear = SecurityProbe.load(linear_path)
    probe_mlp = SecurityProbe.load(mlp_path)

    # Evaluate Primary Linear Probe
    print("Evaluating Model 5: Primary Linear Representation Probe...")
    prob_probe_linear = probe_linear.predict_proba(X_test)
    pred_probe_linear = probe_linear.predict(X_test) # uses optimal validation threshold
    results["neural_linear_probe"] = compute_detection_metrics(y_test, pred_probe_linear, prob_probe_linear)
    results["neural_linear_probe"]["selected_threshold"] = probe_linear.decision_threshold

    # Evaluate Comparison MLP Probe
    print("Evaluating Model 6: Comparison MLP Representation Probe...")
    prob_probe_mlp = probe_mlp.predict_proba(X_test)
    pred_probe_mlp = probe_mlp.predict(X_test)
    results["neural_mlp_probe"] = compute_detection_metrics(y_test, pred_probe_mlp, prob_probe_mlp)
    results["neural_mlp_probe"]["selected_threshold"] = probe_mlp.decision_threshold

    # ==========================================
    # Per-Source Performance Breakdown
    # ==========================================
    per_source_metrics = {}
    sources = set(r.get("source_dataset", "unknown") for r in test_records)
    for src in sources:
        indices = [i for i, r in enumerate(test_records) if r.get("source_dataset") == src]
        if len(indices) >= 5:
            src_y = y_test[indices]
            src_pred = pred_probe_linear[indices]
            src_prob = prob_probe_linear[indices]
            per_source_metrics[src] = compute_detection_metrics(src_y, src_pred, src_prob)

    results["per_source_breakdown"] = per_source_metrics

    # Save detection metrics
    det_metrics_path = os.path.join(metrics_dir, "detection_metrics.json")
    with open(det_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detection metrics to: {det_metrics_path}")

    # ==========================================
    # Localization Evaluation (Ground-Truth Subsets)
    # ==========================================
    print("\nEvaluating Line-Level Localization on Ground-Truth Annotated Test Samples...")
    analyzer = SQLInjectionProbeAnalyzer(
        probe=probe_linear,
        model=model,
        tokenizer=tokenizer,
        layer_idx=target_layer,
    )

    annotated_test = []
    annotated_findings = []
    all_predictions_rows = []

    for idx, r in enumerate(test_records):
        analysis = analyzer.analyze(r["code"])
        is_annotated = r.get("ground_truth_spans_available") or bool(r.get("vulnerable_lines"))
        if is_annotated:
            annotated_test.append(r)
            annotated_findings.append(analysis["findings"])

        # Per-sample prediction record
        all_predictions_rows.append({
            "sample_id": r["sample_id"],
            "source_dataset": r.get("source_dataset", "unknown"),
            "repository": r.get("repository", "unknown"),
            "ground_truth_label": r["label"],
            "predicted_label": int(analysis["is_vulnerable"]),
            "probability": analysis["probability"],
            "outcome": analysis["outcome"],
            "top_suspicious_line": analysis["findings"][0]["start_line"] if analysis["findings"] else None,
            "top_suspicious_score": analysis["findings"][0]["score"] if analysis["findings"] else None,
            "ground_truth_spans_available": is_annotated,
            "ground_truth_lines": str(r.get("vulnerable_lines", [])),
        })

    loc_metrics = compute_localization_metrics(annotated_test, annotated_findings)
    loc_metrics["total_test_samples"] = len(test_records)
    loc_metrics["ground_truth_annotated_in_test"] = len(annotated_test)

    loc_metrics_path = os.path.join(metrics_dir, "localization_metrics.json")
    with open(loc_metrics_path, "w", encoding="utf-8") as f:
        json.dump(loc_metrics, f, indent=2)
    print(f"Saved localization metrics to: {loc_metrics_path}")
    print(f"  Top-1 Hit Rate: {loc_metrics.get('top_1_hit_rate', 0.0)}")
    print(f"  Top-3 Hit Rate: {loc_metrics.get('top_3_hit_rate', 0.0)}")
    print(f"  MRR: {loc_metrics.get('mean_reciprocal_rank', 0.0)}")

    # Save per-sample test predictions CSV
    df_preds = pd.DataFrame(all_predictions_rows)
    preds_csv_path = os.path.join(pred_dir, "test_predictions.csv")
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions CSV to: {preds_csv_path}")

    return {
        "detection_metrics": results,
        "localization_metrics": loc_metrics,
        "predictions_csv": preds_csv_path,
    }


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_file = os.path.join(project_root, "config.yaml")
    run_benchmark_suite(cfg_file)
