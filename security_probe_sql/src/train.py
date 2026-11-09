"""
Training Pipeline for Python SQL Injection Representation Probe.

Extracts representations from frozen LLM, fits primary Linear (Logistic Regression)
and comparison MLP probe heads, and tunes decision threshold strictly on held-out
validation data.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import time
import yaml
import numpy as np
import torch
from typing import Any, Dict, List, Tuple

from src.model.load_model import load_model_and_tokenizer
from security_probe_sql.src.feature_extractor import extract_code_representations
from security_probe_sql.src.probe_model import SecurityProbe
from security_probe_sql.src.evaluation import compute_detection_metrics


def load_dataset_split(filepath: str) -> List[Dict[str, Any]]:
    """Load jsonl dataset records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_split_features(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    layer_idx: int = 16,
    batch_desc: str = "split",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract code representations for all samples in a split."""
    print(f"Extracting hidden-state features for {len(records)} samples ({batch_desc}, layer {layer_idx})...")
    X_list = []
    y_list = []
    sample_ids = []

    t0 = time.time()
    for idx, r in enumerate(records):
        code = r.get("code", "")
        lbl = int(r.get("label", 0))
        sid = r.get("sample_id", f"sample_{idx}")

        rep_dict = extract_code_representations(
            model=model,
            tokenizer=tokenizer,
            code=code,
            layer_idx=layer_idx,
            pooling="mean_prompt",
        )
        X_list.append(rep_dict["pooled_rep"])
        y_list.append(lbl)
        sample_ids.append(sid)

    t1 = time.time()
    elapsed = max(t1 - t0, 0.001)
    print(f"  Extracted {len(X_list)} representations in {elapsed:.2f} s ({len(X_list)/elapsed:.1f} samples/sec)")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, sample_ids


def run_training_pipeline(config_path: str) -> Dict[str, Any]:
    """Execute complete representation extraction, probe training, and validation tuning."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, cfg["outputs"]["models_dir"])
    metrics_dir = os.path.join(project_root, cfg["outputs"]["metrics_dir"])
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Load Data Splits
    train_path = os.path.join(project_root, cfg["data"]["train_split"])
    val_path = os.path.join(project_root, cfg["data"]["val_split"])

    train_records = load_dataset_split(train_path)
    val_records = load_dataset_split(val_path)

    print(f"[Training Pipeline] Loaded {len(train_records)} train, {len(val_records)} validation samples.")

    # 2. Load Frozen Base Model (Mock for fast test, Full for GPU production)
    is_mock = cfg["model"].get("mock", True)
    target_layer = cfg["model"]["mock_target_layer"] if is_mock else cfg["model"]["target_layer"]
    print(f"Loading base LLM (mock={is_mock}, layer={target_layer})...")
    model, tokenizer, _ = load_model_and_tokenizer(mock=is_mock)

    # Ensure model is strictly frozen
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # 3. Extract or Load Cached Representations
    train_feat_cache = os.path.join(models_dir, f"train_features_layer_{target_layer}.npz")
    val_feat_cache = os.path.join(models_dir, f"val_features_layer_{target_layer}.npz")

    if os.path.exists(train_feat_cache) and os.path.exists(val_feat_cache):
        print(f"Loading cached features from {train_feat_cache}...")
        train_npz = np.load(train_feat_cache)
        X_train, y_train = train_npz["X"], train_npz["y"]
        val_npz = np.load(val_feat_cache)
        X_val, y_val = val_npz["X"], val_npz["y"]
    else:
        X_train, y_train, _ = extract_split_features(model, tokenizer, train_records, target_layer, "train")
        X_val, y_val, _ = extract_split_features(model, tokenizer, val_records, target_layer, "validation")
        np.savez_compressed(train_feat_cache, X=X_train, y=y_train)
        np.savez_compressed(val_feat_cache, X=X_val, y=y_val)
        print(f"Cached extracted features to {train_feat_cache}")

    # 4. Train Primary Linear Probe (Logistic Regression)
    print("\nFitting Primary Linear Probe Head (Logistic Regression)...")
    probe_linear = SecurityProbe(
        architecture="linear",
        regularization_c=cfg["probe"]["linear"]["c_regularization"],
        random_state=cfg["project"]["seed"],
        selected_layer=target_layer,
        uncertainty_margin=cfg["threshold"]["uncertainty_margin"],
    )
    probe_linear.fit(X_train, y_train)

    # Validation Threshold Tuning for Linear Probe
    val_tune_results = probe_linear.tune_threshold(X_val, y_val, metric="f1")
    opt_thresh = val_tune_results["optimal_threshold"]
    print(f"  Tuned Optimal Validation Threshold: {opt_thresh:.4f} (Val F1: {val_tune_results['validation_f1']:.4f})")

    # Save Linear Probe
    linear_path = os.path.join(models_dir, "probe_head_linear.pkl")
    probe_linear.save(linear_path)
    print(f"  Saved Linear Probe to {linear_path}")

    # 5. Train Comparison MLP Probe Head
    print("\nFitting Comparison MLP Probe Head (2-layer PyTorch)...")
    probe_mlp = SecurityProbe(
        architecture="mlp",
        random_state=cfg["project"]["seed"],
        selected_layer=target_layer,
        uncertainty_margin=cfg["threshold"]["uncertainty_margin"],
        mlp_hidden_dim=cfg["probe"]["mlp"]["hidden_dim"],
        mlp_epochs=cfg["probe"]["mlp"]["epochs"],
        mlp_lr=cfg["probe"]["mlp"]["learning_rate"],
    )
    probe_mlp.fit(X_train, y_train)

    # Validation Threshold Tuning for MLP Probe
    val_mlp_tune = probe_mlp.tune_threshold(X_val, y_val, metric="f1")
    print(f"  Tuned Optimal Validation Threshold for MLP: {val_mlp_tune['optimal_threshold']:.4f} (Val F1: {val_mlp_tune['validation_f1']:.4f})")

    mlp_path = os.path.join(models_dir, "probe_head_mlp.pkl")
    probe_mlp.save(mlp_path)
    print(f"  Saved MLP Probe to {mlp_path}")

    # 6. Candidate Layer Comparison on Validation Set
    candidate_layers = cfg["model"]["mock_candidate_layers"] if is_mock else cfg["model"]["candidate_layers"]
    layer_comparison_results = {}
    print(f"\nEvaluating candidate layers {candidate_layers} on validation data...")

    for l_cand in candidate_layers:
        if l_cand == target_layer:
            # Reuse already extracted representations
            cand_X_tr, cand_y_tr = X_train, y_train
            cand_X_v, cand_y_v = X_val, y_val
        else:
            cand_X_tr, cand_y_tr, _ = extract_split_features(model, tokenizer, train_records, l_cand, f"train_l{l_cand}")
            cand_X_v, cand_y_v, _ = extract_split_features(model, tokenizer, val_records, l_cand, f"val_l{l_cand}")

        cand_probe = SecurityProbe(
            architecture="linear",
            regularization_c=1.0,
            random_state=cfg["project"]["seed"],
            selected_layer=l_cand,
        )
        cand_probe.fit(cand_X_tr, cand_y_tr)
        t_res = cand_probe.tune_threshold(cand_X_v, cand_y_v)
        val_preds = cand_probe.predict(cand_X_v)
        val_probs = cand_probe.predict_proba(cand_X_v)
        metrics = compute_detection_metrics(cand_y_v, val_preds, val_probs)
        layer_comparison_results[f"layer_{l_cand}"] = {
            "layer_index": l_cand,
            "optimal_threshold": t_res["optimal_threshold"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
        print(f"  Layer {l_cand} -> Bal Acc: {metrics['balanced_accuracy']}, F1: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}")

    training_summary = {
        "base_model": cfg["model"]["base_model"],
        "mock_mode": is_mock,
        "primary_layer": target_layer,
        "linear_probe": {
            "optimal_threshold": opt_thresh,
            "validation_f1": val_tune_results["validation_f1"],
            "model_path": linear_path,
        },
        "mlp_probe": {
            "optimal_threshold": val_mlp_tune["optimal_threshold"],
            "validation_f1": val_mlp_tune["validation_f1"],
            "model_path": mlp_path,
        },
        "layer_comparison": layer_comparison_results,
        "threshold_curve": val_tune_results["threshold_curve"],
    }

    summary_file = os.path.join(metrics_dir, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print(f"\n[Training Completed] Summary saved to: {summary_file}")
    return training_summary


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_file = os.path.join(project_root, "config.yaml")
    run_training_pipeline(cfg_file)
