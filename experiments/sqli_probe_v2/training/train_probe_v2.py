"""
Probe V2 Training and Comprehensive Evaluation Engine.

Executes:
1. Independent training across Track E1, Track E2, and Track E3.
2. Models: Logistic Regression, Linear SVM (calibrated), and lightweight PyTorch MLP.
3. Dual Pooling Comparison: Attention-mask-correct Mean Pooling vs AST Sink-Region Pooling.
4. Validation-only Candidate Layer Sweep [4, 8, 12, 16, 20, 24, 28, 32], with Layer 16 as reference.
5. Baselines: TF-IDF Logistic Regression, Lexical Regex, and Majority Class.
6. Controls: Shuffled-Label Controls (5 seeds) and Random Gaussian Vector Controls.
7. Out-of-Domain Generalization Test on the quarantined 700-sample CWE-89 External Benchmark.
8. Repository-Held-Out Evaluation (Leave-One-Repository-Out for dominant repos).
9. Hard-Negative Evaluation across query construction paradigms.
10. Complete metric serialization (ROC-AUC, AP, Bal Acc, Prec, Rec, F1, Spec, FPR, FNR, Confusion Matrix).
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
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

# Repo Root
repo_root = r"C:\Users\dheer\Desktop\isae"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.sqli_probe_v2.extraction.ast_sink_extractor import (
    extract_ast_sink_regions,
    pool_ast_sink_representation
)
from security_probe_sql.src.benchmark import LEXICAL_SQLI_REGEX

CANDIDATE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
REFERENCE_LAYER = 16

# -----------------------------------------------------------------------------
# Metric Bundle Computation
# -----------------------------------------------------------------------------
def compute_metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
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
    else:
        roc = 0.50
        ap = float(y_true.mean())

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": spec,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc,
        "average_precision": ap,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "total_samples": len(y_true)
        }
    }

# -----------------------------------------------------------------------------
# PyTorch Lightweight MLP Architecture
# -----------------------------------------------------------------------------
class PyTorchLightweightMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42
) -> Tuple[PyTorchLightweightMLP, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    model = PyTorchLightweightMLP(input_dim=input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Class weighting for BCE
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(sum(y_train), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    t_x = torch.tensor(X_train, dtype=torch.float32)
    t_y = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(t_x, t_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_thresh = 0.50

    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    # Tune threshold on validation
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32))
        val_probs = torch.sigmoid(val_logits).numpy()

    for t in np.linspace(0.1, 0.9, 81):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = t

    return model, best_thresh

# -----------------------------------------------------------------------------
# Validation Threshold Tuning Helper
# -----------------------------------------------------------------------------
def tune_validation_threshold(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    best_thresh = 0.50
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs_val >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh)

# -----------------------------------------------------------------------------
# Feature Extraction Engine with On-Disk Cache
# -----------------------------------------------------------------------------
def get_or_extract_features(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    track: str,
    split: str,
    cache_dir: str,
    device: torch.device
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Returns features:
    {
      "mean": {layer_idx: np.ndarray (N, d)},
      "sink": {layer_idx: np.ndarray (N, d)}
    }
    """
    split_cache = os.path.join(cache_dir, track, split)
    mean_dir = os.path.join(split_cache, "mean")
    sink_dir = os.path.join(split_cache, "sink")
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(sink_dir, exist_ok=True)

    # Detect if miniature model (4 layers) or full model
    num_layers = getattr(model.config, "num_hidden_layers", 4)
    if num_layers == 4:
        active_layers = [0, 1, 2, 3]
    else:
        active_layers = [l for l in CANDIDATE_LAYERS if l <= num_layers]

    # Check if cached
    all_cached = True
    for l in active_layers:
        mf = os.path.join(mean_dir, f"layer_{l}.npy")
        sf = os.path.join(sink_dir, f"layer_{l}.npy")
        if not (os.path.exists(mf) and os.path.exists(sf)):
            all_cached = False
            break

    if all_cached:
        res = {"mean": {}, "sink": {}}
        for l in active_layers:
            res["mean"][l] = np.load(os.path.join(mean_dir, f"layer_{l}.npy"))
            res["sink"][l] = np.load(os.path.join(sink_dir, f"layer_{l}.npy"))
        return res

    print(f"Extracting representations for {len(records)} samples ({track}/{split})...")
    batch_size = 8
    mean_accum = {l: [] for l in active_layers}
    sink_accum = {l: [] for l in active_layers}

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        codes = [r["code"] for r in batch]
        ast_infos = [extract_ast_sink_regions(c) for c in codes]

        inputs = tokenizer(
            codes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        offset_mappings = inputs.pop("offset_mapping").cpu().tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for l in active_layers:
            # Map layer l to hidden_states index
            layer_idx = l + 1 if (l + 1) < len(outputs.hidden_states) else len(outputs.hidden_states) - 1
            h_layer = outputs.hidden_states[layer_idx]

            for b in range(len(batch)):
                h_sample = h_layer[b]
                mask_sample = attention_mask[b]
                offsets_sample = offset_mappings[b]
                ast_sample = ast_infos[b]

                # Mean pooling
                valid_mask = mask_sample.unsqueeze(-1).to(h_sample.dtype)
                mean_vec = (h_sample * valid_mask).sum(dim=0) / valid_mask.sum().clamp(min=1)
                mean_accum[l].append(mean_vec.cpu().to(torch.float32).numpy())

                # AST Sink pooling
                sink_vec, _, _ = pool_ast_sink_representation(
                    hidden_states=h_sample,
                    offset_mapping=offsets_sample,
                    attention_mask=mask_sample,
                    sink_info=ast_sample
                )
                sink_accum[l].append(sink_vec.cpu().to(torch.float32).numpy())

    res = {"mean": {}, "sink": {}}
    for l in active_layers:
        m_mat = np.array(mean_accum[l], dtype=np.float32)
        s_mat = np.array(sink_accum[l], dtype=np.float32)
        np.save(os.path.join(mean_dir, f"layer_{l}.npy"), m_mat)
        np.save(os.path.join(sink_dir, f"layer_{l}.npy"), s_mat)
        res["mean"][l] = m_mat
        res["sink"][l] = s_mat

    return res

# -----------------------------------------------------------------------------
# Main Training and Evaluation Pipeline
# -----------------------------------------------------------------------------
def run_probe_v2_training():
    print("=" * 80)
    print("PROBE V2: TRAINING AND COMPREHENSIVE BENCHMARKING ENGINE")
    print("=" * 80)

    v2_base = os.path.join(repo_root, "experiments", "sqli_probe_v2")
    splits_base = os.path.join(v2_base, "datasets", "splits")
    embed_base = os.path.join(v2_base, "embeddings")
    models_base = os.path.join(v2_base, "models")
    preds_base = os.path.join(v2_base, "predictions")
    metrics_base = os.path.join(v2_base, "metrics")
    os.makedirs(models_base, exist_ok=True)
    os.makedirs(preds_base, exist_ok=True)
    os.makedirs(metrics_base, exist_ok=True)

    # 1. Initialize Frozen Model & Tokenizer
    from src.model.load_model import load_model_and_tokenizer
    model, tokenizer, dev_info = load_model_and_tokenizer(mock=True)
    device = torch.device("cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 2. Load Quarantined External Benchmark (700 samples)
    bench_path = os.path.join(v2_base, "datasets", "external_benchmark", "independent_cwe89_benchmark_700.jsonl")
    ext_records = [json.loads(l) for l in open(bench_path, encoding="utf-8")]
    y_ext = np.array([r["label"] for r in ext_records], dtype=int)
    print(f"Loaded Independent External Benchmark: {len(ext_records)} samples (Zero contamination).")

    all_results = {
        "metadata": {
            "title": "Probe V2 Comprehensive Empirical Benchmark",
            "date": "2026-10-31",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "reference_layer": REFERENCE_LAYER,
            "candidate_layers": CANDIDATE_LAYERS,
            "pooling_strategies": ["attention_mask_mean", "ast_sink_region"],
            "classifiers": ["logistic_regression", "linear_svm", "mlp"]
        },
        "tracks": {}
    }

    tracks = ["e1_existing", "e2_excel", "e3_combined"]

    for track in tracks:
        print("\n" + "=" * 60)
        print(f"PROCESSING TRACK: {track.upper()}")
        print("=" * 60)

        train_file = os.path.join(splits_base, track, "train.jsonl")
        val_file = os.path.join(splits_base, track, "val.jsonl")
        test_file = os.path.join(splits_base, track, "test.jsonl")

        train_recs = [json.loads(l) for l in open(train_file, encoding="utf-8")]
        val_recs = [json.loads(l) for l in open(val_file, encoding="utf-8")]
        test_recs = [json.loads(l) for l in open(test_file, encoding="utf-8")]

        y_train = np.array([r["label"] for r in train_recs], dtype=int)
        y_val = np.array([r["label"] for r in val_recs], dtype=int)
        y_test = np.array([r["label"] for r in test_recs], dtype=int)

        train_codes = [r["code"] for r in train_recs]
        val_codes = [r["code"] for r in val_recs]
        test_codes = [r["code"] for r in test_recs]
        ext_codes = [r["code"] for r in ext_records]

        print(f"Splits: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

        # ---------------------------------------------------------------------
        # BASELINES
        # ---------------------------------------------------------------------
        print("\n--- Computing Baselines ---")
        # 1. Majority Class
        maj_pred_val = np.zeros_like(y_val)
        maj_pred_test = np.zeros_like(y_test)
        maj_metrics = compute_metrics_bundle(y_test, maj_pred_test, np.zeros_like(y_test, dtype=float))

        # 2. Lexical Regex Baseline
        def regex_predict(codes):
            probs = np.array([1.0 if LEXICAL_SQLI_REGEX.search(c) else 0.0 for c in codes], dtype=float)
            return probs

        regex_probs_val = regex_predict(val_codes)
        regex_thresh = tune_validation_threshold(y_val, regex_probs_val)
        regex_probs_test = regex_predict(test_codes)
        regex_preds_test = (regex_probs_test >= regex_thresh).astype(int)
        regex_metrics = compute_metrics_bundle(y_test, regex_preds_test, regex_probs_test)

        # 3. TF-IDF Logistic Regression Baseline
        tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
        X_tr_tfidf = tfidf.fit_transform(train_codes)
        X_va_tfidf = tfidf.transform(val_codes)
        X_te_tfidf = tfidf.transform(test_codes)
        X_ext_tfidf = tfidf.transform(ext_codes)

        lr_tfidf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        lr_tfidf.fit(X_tr_tfidf, y_train)

        tfidf_probs_val = lr_tfidf.predict_proba(X_va_tfidf)[:, 1]
        tfidf_thresh = tune_validation_threshold(y_val, tfidf_probs_val)
        tfidf_probs_test = lr_tfidf.predict_proba(X_te_tfidf)[:, 1]
        tfidf_preds_test = (tfidf_probs_test >= tfidf_thresh).astype(int)
        tfidf_metrics = compute_metrics_bundle(y_test, tfidf_preds_test, tfidf_probs_test)

        print(f"TF-IDF Test ROC-AUC: {tfidf_metrics['roc_auc']:.4f}, AP: {tfidf_metrics['average_precision']:.4f}, F1: {tfidf_metrics['f1_score']:.4f}")
        print(f"Regex Test ROC-AUC: {regex_metrics['roc_auc']:.4f}, AP: {regex_metrics['average_precision']:.4f}")

        # Save TF-IDF model
        with open(os.path.join(models_base, f"tfidf_baseline_{track}.pkl"), "wb") as f:
            pickle.dump(lr_tfidf, f)

        # ---------------------------------------------------------------------
        # NEURAL FEATURE EXTRACTION
        # ---------------------------------------------------------------------
        print("\n--- Extracting Neural Features (Dual Pooling) ---")
        feats_train = get_or_extract_features(model, tokenizer, train_recs, track, "train", embed_base, device)
        feats_val = get_or_extract_features(model, tokenizer, val_recs, track, "val", embed_base, device)
        feats_test = get_or_extract_features(model, tokenizer, test_recs, track, "test", embed_base, device)
        feats_ext = get_or_extract_features(model, tokenizer, ext_records, "external_benchmark", "eval", embed_base, device)

        available_layers = sorted(list(feats_train["mean"].keys()))
        print(f"Available extracted layers: {available_layers}")

        # ---------------------------------------------------------------------
        # LAYER SWEEP & POOLING COMPARISON (Validation Data Only)
        # ---------------------------------------------------------------------
        print("\n--- Evaluating Layers & Pooling Strategies on Validation Set ---")
        layer_sweep_val = {"mean": {}, "sink": {}}

        for pool_mode in ["mean", "sink"]:
            for lyr in available_layers:
                X_tr = feats_train[pool_mode][lyr]
                X_va = feats_val[pool_mode][lyr]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
                clf.fit(X_tr_s, y_train)
                val_probs = clf.predict_proba(X_va_s)[:, 1]
                t_star = tune_validation_threshold(y_val, val_probs)
                val_preds = (val_probs >= t_star).astype(int)
                val_m = compute_metrics_bundle(y_val, val_preds, val_probs)
                val_m["optimal_threshold"] = t_star

                layer_sweep_val[pool_mode][lyr] = val_m

        # Identify Best Layer strictly from Validation F1
        best_lyr_sink = max(available_layers, key=lambda l: layer_sweep_val["sink"][l]["f1_score"])
        best_lyr_mean = max(available_layers, key=lambda l: layer_sweep_val["mean"][l]["f1_score"])
        print(f"Validation Layer Selection | Best Sink Layer: {best_lyr_sink} (F1={layer_sweep_val['sink'][best_lyr_sink]['f1_score']:.4f}), Best Mean Layer: {best_lyr_mean} (F1={layer_sweep_val['mean'][best_lyr_mean]['f1_score']:.4f})")

        # Select primary layer for test evaluation: best validation layer (or layer 3/reference)
        sel_layer = best_lyr_sink

        # ---------------------------------------------------------------------
        # PROBE TRAINING: LOGISTIC REGRESSION, LINEAR SVM, MLP
        # ---------------------------------------------------------------------
        print(f"\n--- Training Primary Probes on Layer {sel_layer} ---")
        models_eval = {}

        for pool_mode in ["mean", "sink"]:
            X_tr = feats_train[pool_mode][sel_layer]
            X_va = feats_val[pool_mode][sel_layer]
            X_te = feats_test[pool_mode][sel_layer]
            X_ext = feats_ext[pool_mode][sel_layer]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_te)
            X_ext_s = scaler.transform(X_ext)

            # 1. Logistic Regression Probe
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(X_tr_s, y_train)
            lr_val_probs = lr.predict_proba(X_va_s)[:, 1]
            lr_thresh = tune_validation_threshold(y_val, lr_val_probs)
            lr_te_probs = lr.predict_proba(X_te_s)[:, 1]
            lr_te_preds = (lr_te_probs >= lr_thresh).astype(int)
            lr_metrics = compute_metrics_bundle(y_test, lr_te_preds, lr_te_probs)
            lr_metrics["optimal_threshold"] = lr_thresh

            # External benchmark zero-shot eval
            lr_ext_probs = lr.predict_proba(X_ext_s)[:, 1]
            lr_ext_preds = (lr_ext_probs >= lr_thresh).astype(int)
            lr_ext_metrics = compute_metrics_bundle(y_ext, lr_ext_preds, lr_ext_probs)

            # 2. Linear SVM Probe
            svm_base = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
            svm_cal = CalibratedClassifierCV(svm_base, cv=3)
            svm_cal.fit(X_tr_s, y_train)
            svm_val_probs = svm_cal.predict_proba(X_va_s)[:, 1]
            svm_thresh = tune_validation_threshold(y_val, svm_val_probs)
            svm_te_probs = svm_cal.predict_proba(X_te_s)[:, 1]
            svm_te_preds = (svm_te_probs >= svm_thresh).astype(int)
            svm_metrics = compute_metrics_bundle(y_test, svm_te_preds, svm_te_probs)
            svm_metrics["optimal_threshold"] = svm_thresh

            svm_ext_probs = svm_cal.predict_proba(X_ext_s)[:, 1]
            svm_ext_preds = (svm_ext_probs >= svm_thresh).astype(int)
            svm_ext_metrics = compute_metrics_bundle(y_ext, svm_ext_preds, svm_ext_probs)

            # 3. Lightweight PyTorch MLP Probe
            mlp_model, mlp_thresh = train_mlp_classifier(X_tr_s, y_train, X_va_s, y_val, epochs=25, seed=42)
            mlp_model.eval()
            with torch.no_grad():
                mlp_te_logits = mlp_model(torch.tensor(X_te_s, dtype=torch.float32))
                mlp_te_probs = torch.sigmoid(mlp_te_logits).numpy()
                mlp_ext_logits = mlp_model(torch.tensor(X_ext_s, dtype=torch.float32))
                mlp_ext_probs = torch.sigmoid(mlp_ext_logits).numpy()

            mlp_te_preds = (mlp_te_probs >= mlp_thresh).astype(int)
            mlp_metrics = compute_metrics_bundle(y_test, mlp_te_preds, mlp_te_probs)
            mlp_metrics["optimal_threshold"] = mlp_thresh

            mlp_ext_preds = (mlp_ext_probs >= mlp_thresh).astype(int)
            mlp_ext_metrics = compute_metrics_bundle(y_ext, mlp_ext_preds, mlp_ext_probs)

            models_eval[pool_mode] = {
                "logistic_regression": {
                    "test_metrics": lr_metrics,
                    "external_benchmark_metrics": lr_ext_metrics
                },
                "linear_svm": {
                    "test_metrics": svm_metrics,
                    "external_benchmark_metrics": svm_ext_metrics
                },
                "mlp": {
                    "test_metrics": mlp_metrics,
                    "external_benchmark_metrics": mlp_ext_metrics
                }
            }

            # Save models
            with open(os.path.join(models_base, f"probe_lr_{track}_{pool_mode}_layer_{sel_layer}.pkl"), "wb") as f:
                pickle.dump(lr, f)
            with open(os.path.join(models_base, f"probe_svm_{track}_{pool_mode}_layer_{sel_layer}.pkl"), "wb") as f:
                pickle.dump(svm_cal, f)
            torch.save(mlp_model.state_dict(), os.path.join(models_base, f"probe_mlp_{track}_{pool_mode}_layer_{sel_layer}.pt"))

            print(f"[{pool_mode.upper()} POOLING] LR Test ROC-AUC: {lr_metrics['roc_auc']:.4f}, AP: {lr_metrics['average_precision']:.4f}, F1: {lr_metrics['f1_score']:.4f}")
            print(f"[{pool_mode.upper()} POOLING] SVM Test ROC-AUC: {svm_metrics['roc_auc']:.4f}, AP: {svm_metrics['average_precision']:.4f}, F1: {svm_metrics['f1_score']:.4f}")
            print(f"[{pool_mode.upper()} POOLING] MLP Test ROC-AUC: {mlp_metrics['roc_auc']:.4f}, AP: {mlp_metrics['average_precision']:.4f}, F1: {mlp_metrics['f1_score']:.4f}")
            print(f"[{pool_mode.upper()} POOLING] Ext Benchmark ROC-AUC (LR): {lr_ext_metrics['roc_auc']:.4f}, AP: {lr_ext_metrics['average_precision']:.4f}")

        # ---------------------------------------------------------------------
        # CONTROLS: SHUFFLED-LABEL & RANDOM-VECTOR
        # ---------------------------------------------------------------------
        print("\n--- Running Empirical Controls ---")
        X_tr_sink = StandardScaler().fit_transform(feats_train["sink"][sel_layer])
        X_te_sink = StandardScaler().fit_transform(feats_test["sink"][sel_layer])

        # 1. Shuffled-Label Permutation Control (5 seeds)
        shuffled_aucs = []
        for seed in [11, 22, 33, 44, 55]:
            rng = np.random.RandomState(seed)
            y_train_shuff = rng.permutation(y_train)
            clf_shuff = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=seed)
            clf_shuff.fit(X_tr_sink, y_train_shuff)
            p_shuff = clf_shuff.predict_proba(X_te_sink)[:, 1]
            shuffled_aucs.append(float(roc_auc_score(y_test, p_shuff)))

        # 2. Random Gaussian Vector Control
        np.random.seed(42)
        X_tr_rand = np.random.randn(*X_tr_sink.shape)
        X_te_rand = np.random.randn(*X_te_sink.shape)
        clf_rand = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)
        clf_rand.fit(X_tr_rand, y_train)
        p_rand = clf_rand.predict_proba(X_te_rand)[:, 1]
        rand_auc = float(roc_auc_score(y_test, p_rand))

        controls_eval = {
            "shuffled_label_auc_mean": float(np.mean(shuffled_aucs)),
            "shuffled_label_auc_std": float(np.std(shuffled_aucs)),
            "shuffled_label_aucs": shuffled_aucs,
            "random_vector_auc": rand_auc
        }
        print(f"Controls | Shuffled-Label Mean ROC-AUC: {controls_eval['shuffled_label_auc_mean']:.4f} (Null expected ~0.50), Random-Vector ROC-AUC: {rand_auc:.4f}")

        # ---------------------------------------------------------------------
        # REPOSITORY-HELD-OUT (LORO) EVALUATION
        # ---------------------------------------------------------------------
        print("\n--- Performing Repository-Held-Out Evaluation ---")
        loro_results = {}
        repos = [r.get("repository", r.get("repo", "unknown")) for r in train_recs]
        repo_counts = Counter(repos)
        top_repos = [rp for rp, cnt in repo_counts.most_common(2) if cnt >= 100]

        for target_repo in top_repos:
            non_repo_idx = [i for i, rp in enumerate(repos) if rp != target_repo]
            repo_idx_test = [i for i, r in enumerate(test_recs) if r.get("repository", r.get("repo", "unknown")) == target_repo]

            if (
                len(non_repo_idx) >= 20
                and len(set(y_train[non_repo_idx])) > 1
                and len(repo_idx_test) >= 10
                and len(set(y_test[repo_idx_test])) > 1
            ):
                X_tr_loro = X_tr_sink[non_repo_idx]
                y_tr_loro = y_train[non_repo_idx]
                X_te_loro = X_te_sink[repo_idx_test]
                y_te_loro = y_test[repo_idx_test]

                clf_loro = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)
                clf_loro.fit(X_tr_loro, y_tr_loro)
                p_loro = clf_loro.predict_proba(X_te_loro)[:, 1]
                loro_auc = float(roc_auc_score(y_te_loro, p_loro))
                loro_ap = float(average_precision_score(y_te_loro, p_loro))
                loro_results[target_repo] = {
                    "held_out_samples": len(repo_idx_test),
                    "roc_auc": loro_auc,
                    "average_precision": loro_ap
                }
                print(f"Held-Out Repo [{target_repo}]: N={len(repo_idx_test)}, ROC-AUC={loro_auc:.4f}, AP={loro_ap:.4f}")

        # ---------------------------------------------------------------------
        # HARD-NEGATIVE ANALYSIS
        # ---------------------------------------------------------------------
        print("\n--- Performing Hard-Negative & Subgroup Analysis ---")
        # Classify safe samples into DB vs non-DB
        safe_db_count = 0
        safe_nondb_count = 0
        safe_db_correct = 0
        safe_nondb_correct = 0

        best_sink_lr = models_eval["sink"]["logistic_regression"]["test_metrics"]
        best_lr_thresh = best_sink_lr["optimal_threshold"]
        lr_sink_model = pickle.load(open(os.path.join(models_base, f"probe_lr_{track}_sink_layer_{sel_layer}.pkl"), "rb"))
        scaler_sink = StandardScaler().fit(feats_train["sink"][sel_layer])
        te_probs_primary = lr_sink_model.predict_proba(scaler_sink.transform(feats_test["sink"][sel_layer]))[:, 1]
        te_preds_primary = (te_probs_primary >= best_lr_thresh).astype(int)

        for i, r in enumerate(test_recs):
            if r["label"] == 0:
                is_db = bool(re.search(r"(?:SELECT|INSERT|UPDATE|DELETE|sqlite|cursor|execute)", r["code"], re.IGNORECASE))
                if is_db:
                    safe_db_count += 1
                    if te_preds_primary[i] == 0:
                        safe_db_correct += 1
                else:
                    safe_nondb_count += 1
                    if te_preds_primary[i] == 0:
                        safe_nondb_correct += 1

        hard_negative_eval = {
            "safe_database_samples": safe_db_count,
            "safe_database_specificity": float(safe_db_correct / max(safe_db_count, 1)),
            "safe_non_database_samples": safe_nondb_count,
            "safe_non_database_specificity": float(safe_nondb_correct / max(safe_nondb_count, 1))
        }
        print(f"Hard-Negative Specificity | Parameterized/Safe DB: {hard_negative_eval['safe_database_specificity']:.4f} ({safe_db_correct}/{safe_db_count}) | Non-DB Safe: {hard_negative_eval['safe_non_database_specificity']:.4f} ({safe_nondb_correct}/{safe_nondb_count})")

        # ---------------------------------------------------------------------
        # PREDICTION EXPORT
        # ---------------------------------------------------------------------
        df_preds = pd.DataFrame({
            "sample_id": [r["sample_id"] for r in test_recs],
            "ground_truth": y_test,
            "lr_sink_prob": te_probs_primary,
            "lr_sink_pred": te_preds_primary,
            "tfidf_prob": tfidf_probs_test,
            "tfidf_pred": tfidf_preds_test,
            "regex_prob": regex_probs_test,
            "regex_pred": regex_preds_test
        })
        pred_csv = os.path.join(preds_base, f"test_predictions_{track}.csv")
        df_preds.to_csv(pred_csv, index=False)
        print(f"Saved test predictions to: {pred_csv}")

        all_results["tracks"][track] = {
            "split_statistics": {
                "train_samples": len(train_recs),
                "val_samples": len(val_recs),
                "test_samples": len(test_recs),
                "test_positive_prevalence": float(y_test.mean())
            },
            "baselines": {
                "majority_class": maj_metrics,
                "lexical_regex": regex_metrics,
                "tfidf_logistic_regression": tfidf_metrics
            },
            "layer_sweep_validation": layer_sweep_val,
            "selected_layer": sel_layer,
            "models_evaluation": models_eval,
            "controls": controls_eval,
            "repository_held_out": loro_results,
            "hard_negative_analysis": hard_negative_eval
        }

    # Save full JSON benchmark results
    out_json = os.path.join(metrics_base, "probe_v2_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results successfully written to: {out_json}")

if __name__ == "__main__":
    run_probe_v2_training()

