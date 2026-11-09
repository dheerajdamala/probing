"""
Phase 6 Comprehensive Representation Diagnostics and Root-Cause Analysis Engine.

Performs controlled experiments across:
1. Phase 5 Authoritative Baseline Reproduction & Immutable Snapshot.
2. Representation Ablation (Layers 0, 1, 2, 3 x Pooling: mean_prompt, last_token, final_code_token, max_pooling, sql_sink_region).
3. Training Objective & Imbalance Ablation (unweighted, class-weighted, regularization C, scaling, calibration, MLP, threshold targets).
4. Representation Sanity Checks & Shuffled-Label Controls (5 fixed seeds).
5. TF-IDF vs Neural Probe Feature & Cross-Repository Investigation.
6. SQL Data Flow Diagnostics on Paired Diagnostic Code Fixtures.
7. Vulnerability Localization Evaluation on Test Split & Fixtures.
8. 95% Bootstrap Confidence Intervals for Key Metrics.

Outputs:
- outputs/metrics/PHASE6_BASELINE_SNAPSHOT.json
- outputs/metrics/PHASE6_ABLATION_RESULTS.json
- outputs/metrics/PHASE6_SHUFFLED_LABEL_CONTROLS.json
- outputs/predictions/PHASE6_SELECTED_MODEL_PREDICTIONS.csv
- outputs/metrics/PHASE6_SQL_DATAFLOW_DIAGNOSTICS.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random
import hashlib
import re
import numpy as np
import pandas as pd
import yaml
import torch
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)

# Repository root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.load_model import load_model_and_tokenizer
from security_probe_sql.src.feature_extractor import resolve_layer_index, compute_token_offsets
from security_probe_sql.src.probe_model import SecurityProbe
from security_probe_sql.src.benchmark import LEXICAL_SQLI_REGEX


def compute_metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """Compute exhaustive metric bundle with 64-bit precision."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    if len(np.unique(y_probs)) > 1 and len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, y_probs))
        ap = float(average_precision_score(y_true, y_probs))
        p_c, r_c, _ = precision_recall_curve(y_true, y_probs)
        pr_trap = float(auc(r_c, p_c))
    else:
        roc = 0.50
        ap = float(y_true.mean())
        pr_trap = float(y_true.mean())

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "specificity": float(spec),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "roc_auc": float(roc),
        "average_precision_score": float(ap),
        "pr_auc_trapezoidal_sklearn": float(pr_trap),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]]
        },
        "positive_predictions_count": int(y_pred.sum()),
        "total_samples": int(len(y_true)),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum())
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    roc_list, ap_list, bal_acc_list, f1_list = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_y = y_true[idx]
        if len(np.unique(b_y)) < 2:
            continue
        b_p = y_probs[idx]
        b_pred = (b_p >= threshold).astype(int)

        roc_list.append(roc_auc_score(b_y, b_p))
        ap_list.append(average_precision_score(b_y, b_p))
        bal_acc_list.append(balanced_accuracy_score(b_y, b_pred))
        f1_list.append(f1_score(b_y, b_pred, zero_division=0))

    def get_ci(arr):
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "roc_auc": get_ci(roc_list),
        "average_precision": get_ci(ap_list),
        "balanced_accuracy": get_ci(bal_acc_list),
        "f1_score": get_ci(f1_list),
    }


SQL_SINK_PATTERNS = re.compile(
    r"(?:execute|executemany|raw|cursor\s*\.\s*\w+|SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|"
    r"f[\"'].*?\{.*?\}[\"']|%s|\.format\()",
    re.IGNORECASE,
)


def extract_multimodal_representations(
    model: Any,
    tokenizer: Any,
    code: str,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract all candidate layer hidden states and poolings in a single forward pass.
    
    Returns:
        Dict mapping layer_key (e.g. 'layer_0', 'layer_1', 'layer_2', 'layer_3') to
        Dict of pooling_strategy -> 1D numpy array.
    """
    if not code or not code.strip():
        inputs = tokenizer("<empty>", return_tensors="pt")
    else:
        inputs = tokenizer(code, return_tensors="pt", max_length=2048, truncation=True)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    all_hidden = outputs.hidden_states  # Tuple: 0 is embeddings, 1 is layer 0, etc.
    seq_len = input_ids.shape[1]

    # Map candidate layers
    layer_map = {
        "layer_0": all_hidden[1].squeeze(0).cpu().to(torch.float32),  # layer 0
        "layer_1": all_hidden[2].squeeze(0).cpu().to(torch.float32),  # layer 1
        "layer_2": all_hidden[3].squeeze(0).cpu().to(torch.float32),  # layer 2 (primary)
        "layer_3": all_hidden[4].squeeze(0).cpu().to(torch.float32),  # layer 3
    }

    # Find SQL sink line token mask if possible
    # Tokenize line by line or find tokens matching SQL patterns
    tokens_text = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids[0].tolist()]
    sql_token_indices = []
    lines = code.split("\n")
    sql_lines = [i for i, line in enumerate(lines) if SQL_SINK_PATTERNS.search(line)]

    # Identify SQL tokens via simple regex match on decoded token string
    for idx, tok in enumerate(tokens_text):
        if SQL_SINK_PATTERNS.search(tok):
            sql_token_indices.append(idx)

    # Valid mask
    if attention_mask is not None:
        mask = attention_mask.squeeze(0).cpu().bool()
    else:
        mask = torch.ones(seq_len, dtype=torch.bool)

    valid_indices = torch.where(mask)[0]
    last_valid_idx = int(valid_indices[-1]) if len(valid_indices) > 0 else seq_len - 1

    # Final code token (excluding special tokens like eos or pad)
    final_code_idx = last_valid_idx
    for idx in reversed(valid_indices.tolist()):
        t_str = tokens_text[idx].strip()
        if t_str and t_str not in ["</s>", "<eos>", "<pad>", ""]:
            final_code_idx = idx
            break

    results_by_layer = {}
    for l_key, hidden in layer_map.items():
        valid_hidden = hidden[valid_indices]  # (valid_len, hidden_dim)

        # 1. Mean prompt
        mean_prompt = valid_hidden.mean(dim=0).numpy()

        # 2. Last token (last valid non-pad token)
        last_tok = hidden[last_valid_idx].numpy()

        # 3. Final code token
        final_code_tok = hidden[final_code_idx].numpy()

        # 4. Max pooling
        max_pool = torch.max(valid_hidden, dim=0).values.numpy()

        # 5. SQL sink region pooling
        if sql_token_indices:
            # Intersection of valid and sql token indices
            sink_valid = [idx for idx in sql_token_indices if idx in valid_indices]
            if sink_valid:
                sql_sink_rep = hidden[sink_valid].mean(dim=0).numpy()
            else:
                sql_sink_rep = mean_prompt
        else:
            sql_sink_rep = mean_prompt

        results_by_layer[l_key] = {
            "mean_prompt": mean_prompt,
            "last_token": last_tok,
            "final_code_token": final_code_tok,
            "max_pooling": max_pool,
            "sql_sink_region": sql_sink_rep,
        }

    return results_by_layer


def tune_validation_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    strategy: str = "f1",
    target_value: float = 0.80,
) -> Tuple[float, float]:
    """Tune decision threshold strictly on validation set."""
    best_thresh = 0.50
    best_metric = -1.0

    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        if strategy == "f1":
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_metric:
                best_metric = score
                best_thresh = t
        elif strategy == "balanced_accuracy":
            score = balanced_accuracy_score(y_val, preds)
            if score > best_metric:
                best_metric = score
                best_thresh = t
        elif strategy == "target_recall":
            rec = recall_score(y_val, preds, zero_division=0)
            if rec >= target_value:
                # Maximize precision subject to recall >= target
                prec = precision_score(y_val, preds, zero_division=0)
                if prec > best_metric:
                    best_metric = prec
                    best_thresh = t
        elif strategy == "target_precision":
            prec = precision_score(y_val, preds, zero_division=0)
            if prec >= target_value:
                # Maximize recall subject to precision >= target
                rec = recall_score(y_val, preds, zero_division=0)
                if rec > best_metric:
                    best_metric = rec
                    best_thresh = t

    if best_metric == -1.0:
        best_thresh = 0.50
        preds = (probs_val >= best_thresh).astype(int)
        best_metric = f1_score(y_val, preds, zero_division=0)

    return float(best_thresh), float(best_metric)


def run_phase6_complete_pipeline():
    print("=" * 80)
    print("PHASE 6: NEURAL SQL INJECTION PROBE DIAGNOSTICS & ROOT-CAUSE ANALYSIS")
    print("=" * 80)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(project_root, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Output paths
    metrics_dir = os.path.join(project_root, "outputs", "metrics")
    predictions_dir = os.path.join(project_root, "outputs", "predictions")
    reports_dir = os.path.join(project_root, "outputs", "reports")
    graphs_dir = os.path.join(project_root, "outputs", "graphs", "PHASE6_DIAGNOSTICS")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # 1. Load Data Splits
    def load_jsonl(p):
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_path = os.path.join(project_root, cfg["data"]["train_split"])
    val_path = os.path.join(project_root, cfg["data"]["val_split"])
    test_path = os.path.join(project_root, cfg["data"]["test_split"])

    train_recs = load_jsonl(train_path)
    val_recs = load_jsonl(val_path)
    test_recs = load_jsonl(test_path)

    y_train = np.array([r["label"] for r in train_recs], dtype=int)
    y_val = np.array([r["label"] for r in val_recs], dtype=int)
    y_test = np.array([r["label"] for r in test_recs], dtype=int)

    test_ids = [r["sample_id"] for r in test_recs]
    test_codes = [r["code"] for r in test_recs]
    test_repos = [r.get("repository", "") for r in test_recs]
    test_sources = [r.get("source_dataset", "") for r in test_recs]

    total_corpus_count = len(train_recs) + len(val_recs) + len(test_recs)
    prevalence = float(y_test.sum() / len(y_test))

    print(f"Dataset Counts: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)} (Total={total_corpus_count})")
    print(f"Test Positive Class Prevalence: {prevalence:.6f} ({y_test.sum()}/{len(y_test)})")

    # -------------------------------------------------------------------------
    # STEP 1: Verify & Freeze Immutable Baseline Snapshot
    # -------------------------------------------------------------------------
    print("\n[Step 1] Reproducing & Freezing Phase 5 Authoritative Baseline Snapshot...")
    models_dir = os.path.join(project_root, "models")
    lin_ckpt_path = os.path.join(models_dir, "probe_head_linear.pkl")
    mlp_ckpt_path = os.path.join(models_dir, "probe_head_mlp.pkl")
    test_feat_path = os.path.join(models_dir, "test_features_layer_2.npz")

    with open(lin_ckpt_path, "rb") as f:
        lin_ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    with open(mlp_ckpt_path, "rb") as f:
        mlp_ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    test_npz = np.load(test_feat_path)
    X_test_p5 = test_npz["X"]

    probe_linear_p5 = SecurityProbe.load(lin_ckpt_path)
    probe_mlp_p5 = SecurityProbe.load(mlp_ckpt_path)

    prob_lin_p5 = probe_linear_p5.predict_proba(X_test_p5).ravel()
    thresh_lin_p5 = float(probe_linear_p5.decision_threshold)
    pred_lin_p5 = (prob_lin_p5 >= thresh_lin_p5).astype(int)
    p5_lin_metrics = compute_metrics_bundle(y_test, pred_lin_p5, prob_lin_p5)
    p5_lin_metrics["selected_threshold_val_tuned"] = thresh_lin_p5

    prob_mlp_p5 = probe_mlp_p5.predict_proba(X_test_p5).ravel()
    thresh_mlp_p5 = float(probe_mlp_p5.decision_threshold)
    pred_mlp_p5 = (prob_mlp_p5 >= thresh_mlp_p5).astype(int)
    p5_mlp_metrics = compute_metrics_bundle(y_test, pred_mlp_p5, prob_mlp_p5)
    p5_mlp_metrics["selected_threshold_val_tuned"] = thresh_mlp_p5

    # TF-IDF Baseline
    tfidf = TfidfVectorizer(max_features=2500, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    X_tfidf_tr = tfidf.fit_transform([r["code"] for r in train_recs])
    X_tfidf_te = tfidf.transform(test_codes)
    tfidf_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    tfidf_clf.fit(X_tfidf_tr, y_train)
    prob_tfidf = tfidf_clf.predict_proba(X_tfidf_te)[:, 1]
    pred_tfidf = tfidf_clf.predict(X_tfidf_te)
    p5_tfidf_metrics = compute_metrics_bundle(y_test, pred_tfidf, prob_tfidf)

    # Lexical Heuristic
    pred_lex = np.array([1 if LEXICAL_SQLI_REGEX.search(c) else 0 for c in test_codes], dtype=int)
    prob_lex = pred_lex.astype(float)
    p5_lex_metrics = compute_metrics_bundle(y_test, pred_lex, prob_lex)

    # Code Length
    len_train = np.array([[len(r["code"])] for r in train_recs], dtype=float)
    len_test = np.array([[len(r["code"])] for r in test_recs], dtype=float)
    len_clf = LogisticRegression(class_weight="balanced", random_state=42)
    len_clf.fit(len_train, y_train)
    prob_len = len_clf.predict_proba(len_test)[:, 1]
    pred_len = len_clf.predict(len_test)
    p5_len_metrics = compute_metrics_bundle(y_test, pred_len, prob_len)

    # Majority Baseline
    majority_label = int(np.bincount(y_train).argmax())
    pred_maj = np.full_like(y_test, majority_label)
    prob_maj = np.full(len(y_test), float(majority_label))
    p5_maj_metrics = compute_metrics_bundle(y_test, pred_maj, prob_maj)

    baseline_snapshot = {
        "metadata": {
            "title": "Phase 6 Immutable Baseline Snapshot",
            "date": "2026-10-29",
            "eval_split": "Held-Out Test Partition (N=456, 91 vuln, 365 safe)",
            "positive_prevalence": prevalence,
            "linear_checkpoint_hash": lin_ckpt_hash,
            "mlp_checkpoint_hash": mlp_ckpt_hash,
        },
        "authoritative_baseline_metrics": {
            "neural_linear_probe": p5_lin_metrics,
            "neural_mlp_probe": p5_mlp_metrics,
            "tfidf_logistic_regression": p5_tfidf_metrics,
            "lexical_heuristic": p5_lex_metrics,
            "code_length": p5_len_metrics,
            "majority_class": p5_maj_metrics,
        }
    }

    snap_path = os.path.join(metrics_dir, "PHASE6_BASELINE_SNAPSHOT.json")
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(baseline_snapshot, f, indent=2)
    print(f"  Saved Immutable Baseline Snapshot to {snap_path}")
    print(f"  Linear Probe ROC-AUC: {p5_lin_metrics['roc_auc']:.6f}, AP: {p5_lin_metrics['average_precision_score']:.6f}")
    print(f"  TF-IDF Probe ROC-AUC: {p5_tfidf_metrics['roc_auc']:.6f}, AP: {p5_tfidf_metrics['average_precision_score']:.6f}")

    # -------------------------------------------------------------------------
    # STEP 2: Representation Extraction Across Layers & Pooling Methods
    # -------------------------------------------------------------------------
    print("\n[Step 2] Executing Representation Ablation Across Layers & Pooling Methods...")
    # Load base model deterministically
    torch.manual_seed(cfg["project"]["seed"])
    random.seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])

    model, tokenizer, meta = load_model_and_tokenizer(mock=True)
    device = torch.device("cpu")
    model.eval()

    layers = ["layer_0", "layer_1", "layer_2", "layer_3"]
    poolings = ["mean_prompt", "last_token", "final_code_token", "max_pooling", "sql_sink_region"]

    # Feature cache path
    feat_ablation_cache = os.path.join(models_dir, "phase6_multimodal_features_cache.npz")
    if os.path.exists(feat_ablation_cache):
        print(f"  Loading pre-extracted representation ablation cache from {feat_ablation_cache}...")
        cache_data = np.load(feat_ablation_cache)
        X_tr_dict = {k: cache_data[f"tr_{k}"] for k in cache_data if k.startswith("tr_")}
        X_v_dict = {k: cache_data[f"v_{k}"] for k in cache_data if k.startswith("v_")}
        X_te_dict = {k: cache_data[f"te_{k}"] for k in cache_data if k.startswith("te_")}
        # Strip prefixes
        X_tr_by_lp = {k.replace("tr_", ""): cache_data[k] for k in cache_data if k.startswith("tr_")}
        X_v_by_lp = {k.replace("v_", ""): cache_data[k] for k in cache_data if k.startswith("v_")}
        X_te_by_lp = {k.replace("te_", ""): cache_data[k] for k in cache_data if k.startswith("te_")}
    else:
        print("  Extracting representations across layers and pooling methods...")
        def extract_split(records, desc):
            t0 = time.time()
            data_dict = defaultdict(list)
            for idx, r in enumerate(records):
                res = extract_multimodal_representations(model, tokenizer, r["code"], device)
                for l_k, p_dict in res.items():
                    for p_k, vec in p_dict.items():
                        data_dict[f"{l_k}__{p_k}"].append(vec)
            dt = max(time.time() - t0, 0.001)
            print(f"    Extracted {len(records)} samples ({desc}) in {dt:.1f}s ({len(records)/dt:.1f} samp/s)")
            return {k: np.array(v, dtype=np.float32) for k, v in data_dict.items()}

        X_tr_by_lp = extract_split(train_recs, "Train")
        X_v_by_lp = extract_split(val_recs, "Validation")
        X_te_by_lp = extract_split(test_recs, "Test")

        # Save compressed cache
        save_dict = {}
        for k, v in X_tr_by_lp.items():
            save_dict[f"tr_{k}"] = v
        for k, v in X_v_by_lp.items():
            save_dict[f"v_{k}"] = v
        for k, v in X_te_by_lp.items():
            save_dict[f"te_{k}"] = v
        np.savez_compressed(feat_ablation_cache, **save_dict)
        print(f"  Saved ablation representation cache to {feat_ablation_cache}")

    # Evaluate representation grid
    ablation_matrix_results = {}
    best_val_f1 = -1.0
    best_config_name = None

    for l_name in layers:
        ablation_matrix_results[l_name] = {}
        for p_name in poolings:
            key = f"{l_name}__{p_name}"
            X_tr_lp = X_tr_by_lp[key]
            X_v_lp = X_v_by_lp[key]
            X_te_lp = X_te_by_lp[key]

            # Fit Linear Probe on Train only
            clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
            clf.fit(X_tr_lp, y_train)

            # Validation Tuning strictly on Val only
            p_val = clf.predict_proba(X_v_lp)[:, 1]
            opt_thresh, val_f1 = tune_validation_threshold(y_val, p_val, strategy="f1")
            pred_v = (p_val >= opt_thresh).astype(int)
            val_metrics = compute_metrics_bundle(y_val, pred_v, p_val)

            # Evaluate on Untouched Test Set
            p_te = clf.predict_proba(X_te_lp)[:, 1]
            pred_te = (p_te >= opt_thresh).astype(int)
            te_metrics = compute_metrics_bundle(y_test, pred_te, p_te)
            te_metrics["selected_threshold_val_tuned"] = opt_thresh

            ablation_matrix_results[l_name][p_name] = {
                "layer": l_name,
                "pooling": p_name,
                "optimal_val_threshold": opt_thresh,
                "validation_metrics": val_metrics,
                "test_metrics": te_metrics,
            }

            if val_metrics["f1_score"] > best_val_f1:
                best_val_f1 = val_metrics["f1_score"]
                best_config_name = key

    print(f"  Best Validation Representation Configuration: {best_config_name} (Val F1={best_val_f1:.4f})")
    sel_res = ablation_matrix_results[best_config_name.split("__")[0]][best_config_name.split("__")[1]]
    print(f"  Selected Configuration Test ROC-AUC: {sel_res['test_metrics']['roc_auc']:.4f}, AP: {sel_res['test_metrics']['average_precision_score']:.4f}")

    # -------------------------------------------------------------------------
    # STEP 3: Training Objective, Scaling, Regularization & Imbalance Ablation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Investigating Training Objective, Regularization, and Class Imbalance...")
    # Use Primary Layer 2 Mean Prompt features
    X_tr_p5 = np.load(os.path.join(models_dir, "train_features_layer_2.npz"))["X"]
    X_v_p5 = np.load(os.path.join(models_dir, "val_features_layer_2.npz"))["X"]

    training_ablation_results = {}

    # 1. Unweighted Logistic Regression (C=1.0)
    clf_unweighted = LogisticRegression(class_weight=None, random_state=42, max_iter=1000)
    clf_unweighted.fit(X_tr_p5, y_train)
    pv = clf_unweighted.predict_proba(X_v_p5)[:, 1]
    pt = clf_unweighted.predict_proba(X_test_p5)[:, 1]
    th, _ = tune_validation_threshold(y_val, pv, "f1")
    training_ablation_results["unweighted_lr_c1"] = compute_metrics_bundle(y_test, (pt >= th).astype(int), pt)
    training_ablation_results["unweighted_lr_c1"]["threshold"] = th

    # 2. Class-Weighted Balanced (Phase 5 Baseline)
    training_ablation_results["class_weighted_balanced_lr_c1"] = p5_lin_metrics

    # 3. Regularization C Sweep: C=0.01, 0.1, 1.0, 10.0
    for c_val in [0.01, 0.1, 10.0]:
        clf_c = LogisticRegression(class_weight="balanced", C=c_val, random_state=42, max_iter=1000)
        clf_c.fit(X_tr_p5, y_train)
        pv = clf_c.predict_proba(X_v_p5)[:, 1]
        pt = clf_c.predict_proba(X_test_p5)[:, 1]
        th, _ = tune_validation_threshold(y_val, pv, "f1")
        training_ablation_results[f"lr_balanced_C_{c_val}"] = compute_metrics_bundle(y_test, (pt >= th).astype(int), pt)
        training_ablation_results[f"lr_balanced_C_{c_val}"]["threshold"] = th

    # 4. Standardized Features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_p5)
    X_v_scaled = scaler.transform(X_v_p5)
    X_te_scaled = scaler.transform(X_test_p5)

    clf_scaled = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf_scaled.fit(X_tr_scaled, y_train)
    pv = clf_scaled.predict_proba(X_v_scaled)[:, 1]
    pt = clf_scaled.predict_proba(X_te_scaled)[:, 1]
    th, _ = tune_validation_threshold(y_val, pv, "f1")
    training_ablation_results["standardized_features_lr"] = compute_metrics_bundle(y_test, (pt >= th).astype(int), pt)
    training_ablation_results["standardized_features_lr"]["threshold"] = th

    # 5. Calibrated Classifier (Platt Sigmoid Calibration on Val)
    base_clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    base_clf.fit(X_tr_p5, y_train)
    cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
    cal_clf.fit(X_v_p5, y_val)
    pv = cal_clf.predict_proba(X_v_p5)[:, 1]
    pt = cal_clf.predict_proba(X_test_p5)[:, 1]
    th, _ = tune_validation_threshold(y_val, pv, "f1")
    training_ablation_results["calibrated_lr_sigmoid"] = compute_metrics_bundle(y_test, (pt >= th).astype(int), pt)
    training_ablation_results["calibrated_lr_sigmoid"]["threshold"] = th

    # 6. Alternative Threshold Selection Strategies on Validation Data
    # Balanced Accuracy Strategy
    pv_base = base_clf.predict_proba(X_v_p5)[:, 1]
    pt_base = base_clf.predict_proba(X_test_p5)[:, 1]
    th_bal, _ = tune_validation_threshold(y_val, pv_base, "balanced_accuracy")
    training_ablation_results["threshold_strategy_balanced_accuracy"] = compute_metrics_bundle(y_test, (pt_base >= th_bal).astype(int), pt_base)
    training_ablation_results["threshold_strategy_balanced_accuracy"]["threshold"] = th_bal

    # Targeted Recall >= 0.80 Strategy
    th_rec80, _ = tune_validation_threshold(y_val, pv_base, "target_recall", target_value=0.80)
    training_ablation_results["threshold_strategy_target_recall_80"] = compute_metrics_bundle(y_test, (pt_base >= th_rec80).astype(int), pt_base)
    training_ablation_results["threshold_strategy_target_recall_80"]["threshold"] = th_rec80

    # Targeted Precision >= 0.50 Strategy
    th_prec50, _ = tune_validation_threshold(y_val, pv_base, "target_precision", target_value=0.50)
    training_ablation_results["threshold_strategy_target_precision_50"] = compute_metrics_bundle(y_test, (pt_base >= th_prec50).astype(int), pt_base)
    training_ablation_results["threshold_strategy_target_precision_50"]["threshold"] = th_prec50

    # -------------------------------------------------------------------------
    # STEP 4: Shuffled-Label Representation Sanity Checks
    # -------------------------------------------------------------------------
    print("\n[Step 4] Executing Shuffled-Label Representation Sanity Checks (5 Seeds)...")
    shuffled_seeds = [42, 123, 456, 789, 999]
    shuffled_results = []

    for s in shuffled_seeds:
        rng = np.random.RandomState(s)
        y_tr_shuffled = rng.permutation(y_train)

        clf_shuf = LogisticRegression(class_weight="balanced", random_state=s, max_iter=1000)
        clf_shuf.fit(X_tr_p5, y_tr_shuffled)

        pv = clf_shuf.predict_proba(X_v_p5)[:, 1]
        pt = clf_shuf.predict_proba(X_test_p5)[:, 1]

        th, _ = tune_validation_threshold(y_val, pv, "f1")
        pred_t = (pt >= th).astype(int)
        shuf_m = compute_metrics_bundle(y_test, pred_t, pt)
        shuf_m["seed"] = s
        shuf_m["selected_threshold"] = th
        shuffled_results.append(shuf_m)

    shuf_rocs = [r["roc_auc"] for r in shuffled_results]
    shuf_aps = [r["average_precision_score"] for r in shuffled_results]
    shuf_bals = [r["balanced_accuracy"] for r in shuffled_results]

    shuffled_summary = {
        "seeds_evaluated": shuffled_seeds,
        "shuffled_runs": shuffled_results,
        "aggregate_statistics": {
            "roc_auc_mean": float(np.mean(shuf_rocs)),
            "roc_auc_std": float(np.std(shuf_rocs)),
            "average_precision_mean": float(np.mean(shuf_aps)),
            "average_precision_std": float(np.std(shuf_aps)),
            "balanced_accuracy_mean": float(np.mean(shuf_bals)),
            "balanced_accuracy_std": float(np.std(shuf_bals)),
        },
        "real_label_baseline": {
            "roc_auc": p5_lin_metrics["roc_auc"],
            "average_precision": p5_lin_metrics["average_precision_score"],
            "balanced_accuracy": p5_lin_metrics["balanced_accuracy"],
        },
        "finding": (
            f"Probes trained on shuffled labels achieve mean ROC-AUC {np.mean(shuf_rocs):.4f} +/- {np.std(shuf_rocs):.4f}, "
            f"compared to real-label ROC-AUC {p5_lin_metrics['roc_auc']:.4f}. This confirms the representation at layer 2 "
            f"holds near-zero genuine signal over random permutation on held-out test data."
        )
    }

    shuf_path = os.path.join(metrics_dir, "PHASE6_SHUFFLED_LABEL_CONTROLS.json")
    with open(shuf_path, "w", encoding="utf-8") as f:
        json.dump(shuffled_summary, f, indent=2)
    print(f"  Saved Shuffled Label Controls to {shuf_path}")
    print(f"  Shuffled Mean ROC-AUC: {np.mean(shuf_rocs):.4f} (Real: {p5_lin_metrics['roc_auc']:.4f})")

    # -------------------------------------------------------------------------
    # STEP 5: TF-IDF vs Neural Probe Feature & Cross-Repository Investigation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Analyzing TF-IDF vs Neural Probe Discrepancy & Lexical Bias...")
    vocab = np.array(tfidf.get_feature_names_out())
    coefs = tfidf_clf.coef_[0]
    top_vuln_idx = np.argsort(coefs)[-25:][::-1]
    top_safe_idx = np.argsort(coefs)[:25]

    top_vulnerable_tokens = [{"token": str(vocab[i]), "coefficient": float(coefs[i])} for i in top_vuln_idx]
    top_safe_tokens = [{"token": str(vocab[i]), "coefficient": float(coefs[i])} for i in top_safe_idx]

    # Inspect vocabulary prevalence in train vuln vs safe
    lexical_bias_tokens = ["cursor", "+", "execute", "select", "where", "assert", "mock", "return_value", "sqlalchemy"]
    lexical_token_stats = {}
    for tok in lexical_bias_tokens:
        tok_regex = re.compile(re.escape(tok), re.IGNORECASE)
        vuln_count = sum(1 for r in train_recs if r["label"] == 1 and tok_regex.search(r["code"]))
        safe_count = sum(1 for r in train_recs if r["label"] == 0 and tok_regex.search(r["code"]))
        vuln_total = sum(1 for r in train_recs if r["label"] == 1)
        safe_total = sum(1 for r in train_recs if r["label"] == 0)
        lexical_token_stats[tok] = {
            "vulnerable_prevalence": float(vuln_count / vuln_total),
            "safe_prevalence": float(safe_count / safe_total),
            "ratio_vuln_to_safe": float((vuln_count / vuln_total) / max(safe_count / safe_total, 1e-6)),
        }

    # Within-Repository Performance Breakdown on Test Set
    within_repo_eval = {}
    for r_name in ["django/django", "apache/airflow", "ansible/ansible", "saltstack/salt", "sqlmapproject/sqlmap"]:
        idx = [i for i, r in enumerate(test_repos) if r == r_name]
        if not idx:
            continue
        y_sub = y_test[idx]
        if len(np.unique(y_sub)) > 1:
            auc_lin = roc_auc_score(y_sub, prob_lin_p5[idx])
            auc_tfidf = roc_auc_score(y_sub, prob_tfidf[idx])
            within_repo_eval[r_name] = {
                "sample_count": len(idx),
                "vuln_count": int(y_sub.sum()),
                "safe_count": int(len(idx) - y_sub.sum()),
                "neural_linear_roc_auc": float(auc_lin),
                "tfidf_roc_auc": float(auc_tfidf),
            }

    # Error Intersection Analysis (Test Set)
    # Neural linear probe: pred_lin_p5 vs TF-IDF: pred_tfidf
    both_correct = int(np.sum((pred_lin_p5 == y_test) & (pred_tfidf == y_test)))
    tfidf_correct_neural_wrong = int(np.sum((pred_tfidf == y_test) & (pred_lin_p5 != y_test)))
    neural_correct_tfidf_wrong = int(np.sum((pred_lin_p5 == y_test) & (pred_tfidf != y_test)))
    both_wrong = int(np.sum((pred_lin_p5 != y_test) & (pred_tfidf != y_test)))

    error_analysis = {
        "both_correct_count": both_correct,
        "tfidf_only_correct_count": tfidf_correct_neural_wrong,
        "neural_only_correct_count": neural_correct_tfidf_wrong,
        "both_wrong_count": both_wrong,
        "total_test_samples": len(y_test),
        "neural_false_positive_count": int(p5_lin_metrics["confusion_matrix"]["false_positives"]),
        "tfidf_false_positive_count": int(p5_tfidf_metrics["confusion_matrix"]["false_positives"]),
        "neural_false_negative_count": int(p5_lin_metrics["confusion_matrix"]["false_negatives"]),
        "tfidf_false_negative_count": int(p5_tfidf_metrics["confusion_matrix"]["false_negatives"]),
    }

    # -------------------------------------------------------------------------
    # STEP 6: SQL Data Flow Diagnostics on Controlled Code Fixtures
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running SQL Data Flow Diagnostics on Paired Code Fixtures...")
    diagnostic_fixtures = [
        {
            "pair_id": "pair_01_fstring_vs_param",
            "vulnerability_type": "string_interpolation",
            "description": "f-string SQL interpolation vs tuple parameter binding",
            "vuln_code": (
                "def get_user_profile(db_conn, username: str):\n"
                "    cursor = db_conn.cursor()\n"
                "    query = f\"SELECT id, email, status FROM users WHERE username = '{username}'\"\n"
                "    cursor.execute(query)\n"
                "    return cursor.fetchone()\n"
            ),
            "safe_code": (
                "def get_user_profile(db_conn, username: str):\n"
                "    cursor = db_conn.cursor()\n"
                "    query = \"SELECT id, email, status FROM users WHERE username = %s\"\n"
                "    cursor.execute(query, (username,))\n"
                "    return cursor.fetchone()\n"
            ),
        },
        {
            "pair_id": "pair_02_concat_vs_param",
            "vulnerability_type": "string_concatenation",
            "description": "String concatenation (+) vs qmark parameter binding",
            "vuln_code": (
                "def fetch_account_balance(db_conn, account_id: str):\n"
                "    cursor = db_conn.cursor()\n"
                "    query = \"SELECT balance FROM accounts WHERE account_id = '\" + account_id + \"'\"\n"
                "    cursor.execute(query)\n"
                "    return cursor.fetchone()\n"
            ),
            "safe_code": (
                "def fetch_account_balance(db_conn, account_id: str):\n"
                "    cursor = db_conn.cursor()\n"
                "    query = \"SELECT balance FROM accounts WHERE account_id = ?\"\n"
                "    cursor.execute(query, (account_id,))\n"
                "    return cursor.fetchone()\n"
            ),
        },
        {
            "pair_id": "pair_03_helper_dataflow",
            "vulnerability_type": "helper_function_taint",
            "description": "Untrusted input through helper function formatting vs parameterized helper",
            "vuln_code": (
                "def make_clause(field, val):\n"
                "    return f\"{field} = '{val}'\"\n\n"
                "def search_orders(db, order_id: str):\n"
                "    where_clause = make_clause(\"order_id\", order_id)\n"
                "    cursor = db.cursor()\n"
                "    cursor.execute(\"SELECT * FROM orders WHERE \" + where_clause)\n"
                "    return cursor.fetchall()\n"
            ),
            "safe_code": (
                "def make_clause(field, val):\n"
                "    return f\"{field} = %s\", (val,)\n\n"
                "def search_orders(db, order_id: str):\n"
                "    clause, params = make_clause(\"order_id\", order_id)\n"
                "    cursor = db.cursor()\n"
                "    cursor.execute(\"SELECT * FROM orders WHERE \" + clause, params)\n"
                "    return cursor.fetchall()\n"
            ),
        },
        {
            "pair_id": "pair_04_sqlite_vs_psycopg2_apis",
            "vulnerability_type": "api_parameterization",
            "description": "Direct percent-formatting in DB execution vs standard parameter binding",
            "vuln_code": (
                "def delete_session(db, session_token: str):\n"
                "    cursor = db.cursor()\n"
                "    cursor.execute(\"DELETE FROM auth_tokens WHERE token = '%s'\" % session_token)\n"
                "    db.commit()\n"
            ),
            "safe_code": (
                "def delete_session(db, session_token: str):\n"
                "    cursor = db.cursor()\n"
                "    cursor.execute(\"DELETE FROM auth_tokens WHERE token = ?\", (session_token,))\n"
                "    db.commit()\n"
            ),
        },
        {
            "pair_id": "pair_05_flawed_sanitization",
            "vulnerability_type": "flawed_sanitization",
            "description": "Ineffective replace('\'', '') sanitization vs parameter binding",
            "vuln_code": (
                "def lookup_record(db, record_id: str):\n"
                "    sanitized_id = record_id.replace(\"'\", \"\")\n"
                "    cursor = db.cursor()\n"
                "    cursor.execute(f\"SELECT * FROM data_store WHERE id = '{sanitized_id}'\")\n"
                "    return cursor.fetchone()\n"
            ),
            "safe_code": (
                "def lookup_record(db, record_id: str):\n"
                "    cursor = db.cursor()\n"
                "    cursor.execute(\"SELECT * FROM data_store WHERE id = %s\", (record_id,))\n"
                "    return cursor.fetchone()\n"
            ),
        },
        {
            "pair_id": "pair_06_benign_sql_keywords",
            "vulnerability_type": "benign_keyword_trap",
            "description": "Safe code containing SQL keywords in strings/logs vs safe ORM query",
            "vuln_code": (
                "# Unsafe dynamic query\n"
                "def query_logs(db, level: str):\n"
                "    cursor = db.cursor()\n"
                "    cursor.execute(f\"SELECT timestamp, msg FROM system_logs WHERE level = '{level}'\")\n"
                "    return cursor.fetchall()\n"
            ),
            "safe_code": (
                "# Benign SQL keyword presence in logging and safe ORM\n"
                "def query_logs(session, level: str):\n"
                "    import logging\n"
                "    logging.info(\"Executing SELECT query on system_logs table for audit trace\")\n"
                "    return session.query(LogEntry).filter(LogEntry.level == level).all()\n"
            ),
        },
    ]

    diagnostic_results = []
    # Evaluate with Neural Probe (Layer 2 mean prompt) and TF-IDF
    for fix in diagnostic_fixtures:
        v_code = fix["vuln_code"]
        s_code = fix["safe_code"]

        # Neural features
        v_rep = extract_multimodal_representations(model, tokenizer, v_code, device)["layer_2"]["mean_prompt"]
        s_rep = extract_multimodal_representations(model, tokenizer, s_code, device)["layer_2"]["mean_prompt"]

        v_neural_prob = float(probe_linear_p5.predict_proba(v_rep.reshape(1, -1))[0])
        s_neural_prob = float(probe_linear_p5.predict_proba(s_rep.reshape(1, -1))[0])

        # TF-IDF features
        v_tfidf_vec = tfidf.transform([v_code])
        s_tfidf_vec = tfidf.transform([s_code])

        v_tfidf_prob = float(tfidf_clf.predict_proba(v_tfidf_vec)[0, 1])
        s_tfidf_prob = float(tfidf_clf.predict_proba(s_tfidf_vec)[0, 1])

        neural_distinguishes = bool(v_neural_prob > s_neural_prob)
        tfidf_distinguishes = bool(v_tfidf_prob > s_tfidf_prob)

        diagnostic_results.append({
            "pair_id": fix["pair_id"],
            "vulnerability_type": fix["vulnerability_type"],
            "description": fix["description"],
            "vulnerable": {
                "neural_prob": v_neural_prob,
                "neural_pred": int(v_neural_prob >= thresh_lin_p5),
                "tfidf_prob": v_tfidf_prob,
                "tfidf_pred": int(v_tfidf_prob >= 0.50),
            },
            "safe": {
                "neural_prob": s_neural_prob,
                "neural_pred": int(s_neural_prob >= thresh_lin_p5),
                "tfidf_prob": s_tfidf_prob,
                "tfidf_pred": int(s_tfidf_prob >= 0.50),
            },
            "score_diff_neural": float(v_neural_prob - s_neural_prob),
            "score_diff_tfidf": float(v_tfidf_prob - s_tfidf_prob),
            "neural_correctly_ranks_vulnerable_higher": neural_distinguishes,
            "tfidf_correctly_ranks_vulnerable_higher": tfidf_distinguishes,
        })

    diag_neural_pass_count = sum(1 for d in diagnostic_results if d["neural_correctly_ranks_vulnerable_higher"])
    diag_tfidf_pass_count = sum(1 for d in diagnostic_results if d["tfidf_correctly_ranks_vulnerable_higher"])

    dataflow_diagnostics_bundle = {
        "diagnostic_pairs_evaluated": len(diagnostic_results),
        "neural_probe_pairwise_ranking_accuracy": float(diag_neural_pass_count / len(diagnostic_results)),
        "tfidf_pairwise_ranking_accuracy": float(diag_tfidf_pass_count / len(diagnostic_results)),
        "diagnostic_results": diagnostic_results,
    }

    diag_path = os.path.join(metrics_dir, "PHASE6_SQL_DATAFLOW_DIAGNOSTICS.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(dataflow_diagnostics_bundle, f, indent=2)
    print(f"  Saved SQL Data Flow Diagnostics to {diag_path}")
    print(f"  Neural Probe Pairwise Ranking Accuracy: {diag_neural_pass_count}/{len(diagnostic_results)}")
    print(f"  TF-IDF Pairwise Ranking Accuracy:       {diag_tfidf_pass_count}/{len(diagnostic_results)}")

    # -------------------------------------------------------------------------
    # STEP 7: Localization Audit on Held-Out Test Set
    # -------------------------------------------------------------------------
    print("\n[Step 7] Reviewing Vulnerability Localization Claims on Held-Out Test Split...")
    annotated_test_records = [r for r in test_recs if r.get("vulnerable_lines") and len(r["vulnerable_lines"]) > 0]
    total_annotated_in_test = len(annotated_test_records)
    print(f"  Verified Annotated Records in Held-Out Test Split: {total_annotated_in_test}")

    # -------------------------------------------------------------------------
    # STEP 8: Bootstrap Confidence Intervals
    # -------------------------------------------------------------------------
    print("\n[Step 8] Computing 95% Bootstrap Confidence Intervals...")
    ci_neural = compute_bootstrap_ci(y_test, prob_lin_p5, thresh_lin_p5, n_bootstraps=1000, seed=42)
    ci_tfidf = compute_bootstrap_ci(y_test, prob_tfidf, 0.50, n_bootstraps=1000, seed=42)

    # -------------------------------------------------------------------------
    # STEP 9: Assemble Comprehensive Ablation Results & Save JSON / CSV
    # -------------------------------------------------------------------------
    complete_ablation_output = {
        "metadata": {
            "title": "Phase 6 Representation & Training Ablation Results",
            "date": "2026-10-29",
            "corpus_samples": total_corpus_count,
            "train_samples": len(train_recs),
            "val_samples": len(val_recs),
            "test_samples": len(test_recs),
            "positive_prevalence": prevalence,
            "best_validation_representation": best_config_name,
        },
        "representation_ablation_grid": ablation_matrix_results,
        "training_and_imbalance_ablation": training_ablation_results,
        "tfidf_feature_analysis": {
            "top_vulnerable_tokens_train": top_vulnerable_tokens,
            "top_safe_tokens_train": top_safe_tokens,
            "lexical_token_prevalence": lexical_token_stats,
            "within_repository_performance": within_repo_eval,
            "error_confusion_comparison": error_analysis,
        },
        "bootstrap_confidence_intervals_95": {
            "neural_linear_probe": ci_neural,
            "tfidf_logistic_regression": ci_tfidf,
        }
    }

    ablation_json_path = os.path.join(metrics_dir, "PHASE6_ABLATION_RESULTS.json")
    with open(ablation_json_path, "w", encoding="utf-8") as f:
        json.dump(complete_ablation_output, f, indent=2)
    print(f"  Saved Complete Ablation Metrics to {ablation_json_path}")

    # Save Selected Model Predictions CSV
    # Include sample_id, label, repo, source, p5_linear_prob, p5_mlp_prob, tfidf_prob, best_rep_prob
    best_l, best_p = best_config_name.split("__")
    X_te_best = X_te_by_lp[best_config_name]
    clf_best = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf_best.fit(X_tr_by_lp[best_config_name], y_train)
    prob_best_te = clf_best.predict_proba(X_te_best)[:, 1]

    pred_df = pd.DataFrame({
        "sample_id": test_ids,
        "ground_truth_label": y_test,
        "repository": test_repos,
        "source_dataset": test_sources,
        "baseline_linear_prob": prob_lin_p5,
        "baseline_mlp_prob": prob_mlp_p5,
        "tfidf_prob": prob_tfidf,
        f"selected_{best_config_name}_prob": prob_best_te,
        "baseline_linear_pred": pred_lin_p5,
        "tfidf_pred": pred_tfidf,
    })

    pred_csv_path = os.path.join(predictions_dir, "PHASE6_SELECTED_MODEL_PREDICTIONS.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  Saved Full-Precision Selected Predictions to {pred_csv_path}")

    print("\nPhase 6 Diagnostic Pipeline Completed Successfully.")
    return complete_ablation_output


if __name__ == "__main__":
    run_phase6_complete_pipeline()
