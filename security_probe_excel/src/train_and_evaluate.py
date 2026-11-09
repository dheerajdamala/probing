"""
Complete training and evaluation pipeline for the SQL Injection Probe on the expanded Excel-compatible corpus.
Extracts representations, trains Linear, Sink, MLP, Paired, and Baseline models,
tunes thresholds strictly on validation data, and evaluates on held-out test data.
"""

import os
import sys
import json
import time
import pickle
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
repo_root = os.path.dirname(parent_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

from src.model.load_model import create_miniature_mistral

# SQL Sink Regex
SQL_SINK_REGEX = re.compile(
    r'(?:execute|executemany|raw|cursor\s*\.\s*\w+|SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|f[\"\'].*?\{.*?\}[\"\']|%s|\.format\()',
    re.IGNORECASE
)

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def extract_features(model, tokenizer, code: str, layer_idx: int = 3, pooling: str = "sql_sink_region"):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states: 0=embed, 1=layer0, 2=layer1, 3=layer2, 4=layer3
        idx = min(layer_idx + 1, len(outputs.hidden_states) - 1)
        h = outputs.hidden_states[idx].squeeze(0) # (seq_len, dim)
        
    input_ids = inputs["input_ids"].squeeze(0).tolist()
    
    if pooling == "mean_prompt":
        return h.mean(dim=0).cpu().numpy()
        
    elif pooling == "sql_sink_region":
        sink_indices = []
        for i, tid in enumerate(input_ids):
            token_str = tokenizer.decode([tid])
            if SQL_SINK_REGEX.search(token_str):
                sink_indices.append(i)
        if sink_indices:
            return h[sink_indices].mean(dim=0).cpu().numpy()
        return h.mean(dim=0).cpu().numpy()
    else:
        return h.mean(dim=0).cpu().numpy()

def extract_dataset_features(model, tokenizer, records: List[Dict[str, Any]], pooling: str = "sql_sink_region", layer_idx: int = 3):
    X_list = []
    y_list = []
    print(f"Extracting {pooling} features for {len(records)} records (layer {layer_idx})...")
    t0 = time.time()
    for i, r in enumerate(records):
        feat = extract_features(model, tokenizer, r["code"], layer_idx=layer_idx, pooling=pooling)
        X_list.append(feat)
        y_list.append(int(r["label"]))
    elapsed = max(time.time() - t0, 0.001)
    print(f"Extracted {len(X_list)} representations in {elapsed:.2f}s ({len(X_list)/elapsed:.1f} samples/s).")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    try:
        ap = float(average_precision_score(y_true, y_prob))
    except Exception:
        ap = float(np.mean(y_true))
        
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": spec,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": auc,
        "average_precision": ap,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return best_thresh

def run_experiment():
    base_dir = r"C:\Users\dheer\Desktop\isae\security_probe_excel"
    splits_dir = os.path.join(base_dir, "data", "splits")
    models_dir = os.path.join(base_dir, "models")
    metrics_dir = os.path.join(base_dir, "outputs", "metrics")
    pred_dir = os.path.join(base_dir, "outputs", "predictions")
    samples_dir = os.path.join(base_dir, "outputs", "samples")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # 1. Load data
    train_recs = [json.loads(l) for l in open(os.path.join(splits_dir, "train.jsonl"), "r", encoding="utf-8")]
    val_recs = [json.loads(l) for l in open(os.path.join(splits_dir, "val.jsonl"), "r", encoding="utf-8")]
    test_recs = [json.loads(l) for l in open(os.path.join(splits_dir, "test.jsonl"), "r", encoding="utf-8")]
    
    print(f"Loaded records: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    
    # 2. Initialize frozen base model
    print("Initializing frozen miniature Mistral model...")
    model, tokenizer = create_miniature_mistral(num_layers=4, hidden_size=128)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    # 3. Extract Representations
    # A. Mean pooling (Layer 3)
    X_tr_mean, y_train = extract_dataset_features(model, tokenizer, train_recs, pooling="mean_prompt", layer_idx=3)
    X_va_mean, y_val = extract_dataset_features(model, tokenizer, val_recs, pooling="mean_prompt", layer_idx=3)
    X_te_mean, y_test = extract_dataset_features(model, tokenizer, test_recs, pooling="mean_prompt", layer_idx=3)
    
    # B. SQL Sink pooling (Layer 3)
    X_tr_sink, _ = extract_dataset_features(model, tokenizer, train_recs, pooling="sql_sink_region", layer_idx=3)
    X_va_sink, _ = extract_dataset_features(model, tokenizer, val_recs, pooling="sql_sink_region", layer_idx=3)
    X_te_sink, _ = extract_dataset_features(model, tokenizer, test_recs, pooling="sql_sink_region", layer_idx=3)
    
    # Scale representations
    scaler_mean = StandardScaler().fit(X_tr_mean)
    X_tr_mean_s = scaler_mean.transform(X_tr_mean)
    X_va_mean_s = scaler_mean.transform(X_va_mean)
    X_te_mean_s = scaler_mean.transform(X_te_mean)
    
    scaler_sink = StandardScaler().fit(X_tr_sink)
    X_tr_sink_s = scaler_sink.transform(X_tr_sink)
    X_va_sink_s = scaler_sink.transform(X_va_sink)
    X_te_sink_s = scaler_sink.transform(X_te_sink)
    
    # --- MODEL A: Linear Probe (Mean Pooling) ---
    print("\n[Training Model A: Linear Probe (Mean Pooling)]")
    clf_mean = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    clf_mean.fit(X_tr_mean_s, y_train)
    val_prob_mean = clf_mean.predict_proba(X_va_mean_s)[:, 1]
    thresh_mean = tune_threshold(y_val, val_prob_mean)
    test_prob_mean = clf_mean.predict_proba(X_te_mean_s)[:, 1]
    metrics_mean = compute_metrics(y_test, test_prob_mean, thresh_mean)
    print(f"Model A Test ROC-AUC: {metrics_mean['roc_auc']:.4f}, AP: {metrics_mean['average_precision']:.4f}, F1: {metrics_mean['f1_score']:.4f}")
    
    # --- MODEL B: Linear Probe (SQL Sink Pooling) ---
    print("\n[Training Model B: Linear Probe (SQL Sink Pooling)]")
    clf_sink = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    clf_sink.fit(X_tr_sink_s, y_train)
    val_prob_sink = clf_sink.predict_proba(X_va_sink_s)[:, 1]
    thresh_sink = tune_threshold(y_val, val_prob_sink)
    test_prob_sink = clf_sink.predict_proba(X_te_sink_s)[:, 1]
    metrics_sink = compute_metrics(y_test, test_prob_sink, thresh_sink)
    print(f"Model B Test ROC-AUC: {metrics_sink['roc_auc']:.4f}, AP: {metrics_sink['average_precision']:.4f}, F1: {metrics_sink['f1_score']:.4f}")
    
    # --- MODEL C: MLP Probe (SQL Sink Pooling) ---
    print("\n[Training Model C: MLP Probe (SQL Sink Pooling)]")
    mlp = PyTorchMLP(input_dim=128, hidden_dim=64)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.005, weight_decay=0.01)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    X_tr_tensor = torch.tensor(X_tr_sink_s, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_va_tensor = torch.tensor(X_va_sink_s, dtype=torch.float32)
    X_te_tensor = torch.tensor(X_te_sink_s, dtype=torch.float32)
    
    for epoch in range(40):
        mlp.train()
        optimizer.zero_grad()
        out = mlp(X_tr_tensor)
        loss = criterion(out, y_tr_tensor)
        loss.backward()
        optimizer.step()
        
    mlp.eval()
    with torch.no_grad():
        val_prob_mlp = torch.sigmoid(mlp(X_va_tensor)).cpu().numpy()
        test_prob_mlp = torch.sigmoid(mlp(X_te_tensor)).cpu().numpy()
    thresh_mlp = tune_threshold(y_val, val_prob_mlp)
    metrics_mlp = compute_metrics(y_test, test_prob_mlp, thresh_mlp)
    print(f"Model C Test ROC-AUC: {metrics_mlp['roc_auc']:.4f}, AP: {metrics_mlp['average_precision']:.4f}, F1: {metrics_mlp['f1_score']:.4f}")

    # --- MODEL D: Contrastive / Paired Enhanced Probe ---
    print("\n[Training Model D: Contrastive / Paired Representation Probe]")
    # For samples in train with fixed_code, augment training set with explicit difference vectors
    paired_train_diffs = []
    paired_train_y = []
    for r in train_recs:
        if r.get("fixed_code"):
            feat_vuln = extract_features(model, tokenizer, r["code"], layer_idx=3, pooling="sql_sink_region")
            feat_safe = extract_features(model, tokenizer, r["fixed_code"], layer_idx=3, pooling="sql_sink_region")
            diff = feat_vuln - feat_safe
            paired_train_diffs.append(diff)
            paired_train_y.append(1)
            paired_train_diffs.append(-diff)
            paired_train_y.append(0)
            
    if paired_train_diffs:
        X_tr_paired = np.vstack([X_tr_sink, np.array(paired_train_diffs, dtype=np.float32)])
        y_tr_paired = np.concatenate([y_train, np.array(paired_train_y, dtype=np.int32)])
    else:
        X_tr_paired = X_tr_sink
        y_tr_paired = y_train
        
    scaler_paired = StandardScaler().fit(X_tr_paired)
    X_tr_paired_s = scaler_paired.transform(X_tr_paired)
    X_va_paired_s = scaler_paired.transform(X_va_sink)
    X_te_paired_s = scaler_paired.transform(X_te_sink)
    
    clf_paired = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    clf_paired.fit(X_tr_paired_s, y_tr_paired)
    val_prob_paired = clf_paired.predict_proba(X_va_paired_s)[:, 1]
    thresh_paired = tune_threshold(y_val, val_prob_paired)
    test_prob_paired = clf_paired.predict_proba(X_te_paired_s)[:, 1]
    metrics_paired = compute_metrics(y_test, test_prob_paired, thresh_paired)
    print(f"Model D Test ROC-AUC: {metrics_paired['roc_auc']:.4f}, AP: {metrics_paired['average_precision']:.4f}, F1: {metrics_paired['f1_score']:.4f}")

    # --- BASELINE E: TF-IDF + Logistic Regression ---
    print("\n[Training Baseline E: TF-IDF + Logistic Regression]")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|[^\w\s]')
    train_texts = [r["code"] for r in train_recs]
    val_texts = [r["code"] for r in val_recs]
    test_texts = [r["code"] for r in test_recs]
    
    X_tr_tfidf = tfidf.fit_transform(train_texts)
    X_va_tfidf = tfidf.transform(val_texts)
    X_te_tfidf = tfidf.transform(test_texts)
    
    clf_tfidf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    clf_tfidf.fit(X_tr_tfidf, y_train)
    val_prob_tfidf = clf_tfidf.predict_proba(X_va_tfidf)[:, 1]
    thresh_tfidf = tune_threshold(y_val, val_prob_tfidf)
    test_prob_tfidf = clf_tfidf.predict_proba(X_te_tfidf)[:, 1]
    metrics_tfidf = compute_metrics(y_test, test_prob_tfidf, thresh_tfidf)
    print(f"Baseline E Test ROC-AUC: {metrics_tfidf['roc_auc']:.4f}, AP: {metrics_tfidf['average_precision']:.4f}, F1: {metrics_tfidf['f1_score']:.4f}")

    # --- BASELINE F: Regex Heuristic Baseline ---
    print("\n[Evaluating Baseline F: Regex Baseline]")
    vuln_regex = re.compile(r'(?:execute\s*\(.*?(\+|\%|format|f["\']).*?\)|raw\s*\(.*?(\+|\%|format|f["\']))', re.DOTALL | re.IGNORECASE)
    test_prob_regex = np.array([1.0 if vuln_regex.search(r["code"]) else 0.0 for r in test_recs], dtype=np.float32)
    metrics_regex = compute_metrics(y_test, test_prob_regex, threshold=0.5)
    print(f"Baseline F Test ROC-AUC: {metrics_regex['roc_auc']:.4f}, F1: {metrics_regex['f1_score']:.4f}")

    # --- BASELINE G: Majority Class Baseline ---
    print("\n[Evaluating Baseline G: Majority Class Baseline]")
    majority_prob = np.full(len(y_test), fill_value=float(np.mean(y_train)), dtype=np.float32)
    metrics_majority = compute_metrics(y_test, majority_prob, threshold=0.5)
    print(f"Baseline G Test Balanced Accuracy: {metrics_majority['balanced_accuracy']:.4f}")

    # --- STRATIFIED EVALUATION (Model B Sink Probe) ---
    print("\n[Computing Stratified Evaluations for Model B]")
    test_df = pd.DataFrame(test_recs)
    test_df["y_true"] = y_test
    test_df["prob_sink"] = test_prob_sink
    test_df["prob_tfidf"] = test_prob_tfidf
    test_df["pred_sink"] = (test_prob_sink >= thresh_sink).astype(int)
    test_df["pred_tfidf"] = (test_prob_tfidf >= thresh_tfidf).astype(int)
    
    # Save test predictions CSV
    pred_csv_path = os.path.join(pred_dir, "test_predictions.csv")
    test_df.to_csv(pred_csv_path, index=False)
    print(f"Saved test predictions to: {pred_csv_path}")

    stratified_results = {
        "case_type": {},
        "db_api": {},
        "query_style": {}
    }
    
    for ctype, group in test_df.groupby("case_type"):
        if len(group) >= 5:
            m = compute_metrics(group["y_true"].values, group["prob_sink"].values, thresh_sink)
            stratified_results["case_type"][ctype] = {
                "count": len(group),
                "vuln_count": int(group["y_true"].sum()),
                "roc_auc": m["roc_auc"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1_score": m["f1_score"]
            }
            
    for db, group in test_df.groupby("db_api"):
        if len(group) >= 5:
            m = compute_metrics(group["y_true"].values, group["prob_sink"].values, thresh_sink)
            stratified_results["db_api"][db] = {
                "count": len(group),
                "vuln_count": int(group["y_true"].sum()),
                "roc_auc": m["roc_auc"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1_score": m["f1_score"]
            }
            
    for qs, group in test_df.groupby("query_style"):
        if len(group) >= 5:
            m = compute_metrics(group["y_true"].values, group["prob_sink"].values, thresh_sink)
            stratified_results["query_style"][qs] = {
                "count": len(group),
                "vuln_count": int(group["y_true"].sum()),
                "roc_auc": m["roc_auc"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1_score": m["f1_score"]
            }

    # --- LOCALIZATION EVALUATION (Verified Line Annotations) ---
    print("\n[Computing Localization Evaluation on Verified Lines]")
    loc_samples = [r for r in test_recs if r.get("vulnerable_lines") and len(r.get("vulnerable_lines", [])) > 0]
    print(f"Test samples with ground truth vulnerable_lines: {len(loc_samples)}")
    
    line_hits = 0
    total_loc = 0
    sample_loc_details = []
    
    for r in loc_samples:
        total_loc += 1
        gt_lines = set(r["vulnerable_lines"])
        code = r["code"]
        code_lines = code.splitlines()
        
        # Token-based line detection via SQL sink tokens
        pred_lines = set()
        for l_num, line_str in enumerate(code_lines, 1):
            if SQL_SINK_REGEX.search(line_str):
                pred_lines.add(l_num)
                
        hit = len(gt_lines.intersection(pred_lines)) > 0
        if hit:
            line_hits += 1
        sample_loc_details.append({
            "sample_id": r["sample_id"],
            "dataset_source": r.get("dataset_source", ""),
            "gt_lines": sorted(list(gt_lines)),
            "pred_lines": sorted(list(pred_lines)),
            "hit": hit
        })
        
    loc_accuracy = line_hits / max(total_loc, 1)
    print(f"Localization Top-1 / Hit Accuracy: {loc_accuracy:.4f} ({line_hits}/{total_loc})")
    
    localization_results = {
        "total_annotated_samples": total_loc,
        "hit_count": line_hits,
        "hit_accuracy": loc_accuracy,
        "sample_details": sample_loc_details[:20]
    }

    # --- SAVE ARTIFACTS ---
    # Save models
    with open(os.path.join(models_dir, "probe_head_excel_linear.pkl"), "wb") as f:
        pickle.dump({"classifier": clf_mean, "scaler": scaler_mean, "threshold": thresh_mean}, f)
    with open(os.path.join(models_dir, "probe_head_excel_sink.pkl"), "wb") as f:
        pickle.dump({"classifier": clf_sink, "scaler": scaler_sink, "threshold": thresh_sink}, f)
    with open(os.path.join(models_dir, "probe_head_excel_mlp.pkl"), "wb") as f:
        pickle.dump({"model": mlp.state_dict(), "scaler": scaler_sink, "threshold": thresh_mlp}, f)
    with open(os.path.join(models_dir, "probe_head_excel_paired.pkl"), "wb") as f:
        pickle.dump({"classifier": clf_paired, "scaler": scaler_paired, "threshold": thresh_paired}, f)
    with open(os.path.join(models_dir, "tfidf_baseline.pkl"), "wb") as f:
        pickle.dump({"vectorizer": tfidf, "classifier": clf_tfidf, "threshold": thresh_tfidf}, f)
    print(f"Saved all checkpoints to: {models_dir}")

    # Compile comprehensive metrics JSON
    all_results = {
        "metadata": {
            "title": "Excel-Compatible SQL Injection Probe Benchmark",
            "date": "2026-10-31",
            "train_samples": len(train_recs),
            "val_samples": len(val_recs),
            "test_samples": len(test_recs),
            "test_positive_prevalence": float(np.mean(y_test))
        },
        "model_evaluations": {
            "linear_probe_mean_pooling": metrics_mean,
            "linear_probe_sql_sink_pooling": metrics_sink,
            "mlp_probe_sql_sink_pooling": metrics_mlp,
            "paired_enhanced_probe": metrics_paired,
            "tfidf_logistic_regression": metrics_tfidf,
            "regex_baseline": metrics_regex,
            "majority_baseline": metrics_majority
        },
        "stratified_evaluations": stratified_results,
        "localization_evaluation": localization_results
    }
    
    results_path = os.path.join(metrics_dir, "EXCEL_PROBE_RESULTS.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved complete metrics to: {results_path}")

    # Save sample training records (JSONL)
    sample_records_path = os.path.join(samples_dir, "sample_training_records.jsonl")
    with open(sample_records_path, "w", encoding="utf-8") as f:
        for r in train_recs[:10]:
            f.write(json.dumps(r) + "\n")
    print(f"Saved sample training records to: {sample_records_path}")

    return all_results

if __name__ == "__main__":
    run_experiment()

