"""
Phase 5 Automated Consistency and Reproducibility Tests.

Validates that:
1. PHASE5_AUTHORITATIVE_METRICS.json matches recomputed metrics exactly.
2. PHASE5_FULL_PRECISION_PREDICTIONS.csv aligns 1-to-1 with test.jsonl IDs.
3. Checkpoint SHA-256 hashes match recorded values.
4. Majority class average precision strictly equals test-set positive prevalence.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "metrics", "PHASE5_AUTHORITATIVE_METRICS.json")
PRED_PATH = os.path.join(PROJECT_ROOT, "outputs", "predictions", "PHASE5_FULL_PRECISION_PREDICTIONS.csv")
TEST_JSONL = os.path.join(PROJECT_ROOT, "data", "processed", "splits", "test.jsonl")
LIN_CKPT = os.path.join(PROJECT_ROOT, "models", "probe_head_linear.pkl")

def test_authoritative_artifacts_exist():
    assert os.path.exists(METRICS_PATH), "Authoritative metrics JSON missing!"
    assert os.path.exists(PRED_PATH), "Full precision predictions CSV missing!"
    assert os.path.exists(TEST_JSONL), "Test split jsonl missing!"
    assert os.path.exists(LIN_CKPT), "Linear probe checkpoint missing!"

def test_prediction_sample_ids_align_with_test_split():
    df = pd.read_csv(PRED_PATH)
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    assert len(df) == len(test_records), f"Prediction count {len(df)} != test records {len(test_records)}"
    for idx, r in enumerate(test_records):
        assert df.iloc[idx]["sample_id"] == r["sample_id"]
        assert df.iloc[idx]["ground_truth_label"] == r["label"]

def test_recomputed_metrics_match_authoritative_json():
    df = pd.read_csv(PRED_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        auth_data = json.load(f)

    lin_metrics = auth_data["authoritative_model_metrics"]["neural_linear_probe"]
    y_true = df["ground_truth_label"].to_numpy().astype(int)
    y_pred = df["pred_linear"].to_numpy().astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    assert tn == lin_metrics["confusion_matrix"]["true_negatives"]
    assert fp == lin_metrics["confusion_matrix"]["false_positives"]
    assert fn == lin_metrics["confusion_matrix"]["false_negatives"]
    assert tp == lin_metrics["confusion_matrix"]["true_positives"]

    assert abs(accuracy_score(y_true, y_pred) - lin_metrics["accuracy"]) < 1e-4
    assert abs(balanced_accuracy_score(y_true, y_pred) - lin_metrics["balanced_accuracy"]) < 1e-4
    assert abs(precision_score(y_true, y_pred, zero_division=0) - lin_metrics["precision"]) < 1e-4
    assert abs(recall_score(y_true, y_pred, zero_division=0) - lin_metrics["recall"]) < 1e-4
    assert abs(f1_score(y_true, y_pred, zero_division=0) - lin_metrics["f1_score"]) < 1e-4

def test_checkpoint_hash_matches_recorded():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        auth_data = json.load(f)
    expected_hash = auth_data["benchmark_metadata"]["linear_checkpoint"]["sha256"]

    with open(LIN_CKPT, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    assert actual_hash == expected_hash, f"Checkpoint hash mismatch: {actual_hash} != {expected_hash}"

def test_majority_baseline_ap_equals_prevalence():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        auth_data = json.load(f)
    prevalence = auth_data["benchmark_metadata"]["positive_class_prevalence"]
    maj_ap = auth_data["authoritative_model_metrics"]["majority_class"]["average_precision_score"]

    assert abs(maj_ap - prevalence) < 1e-5, f"Majority AP {maj_ap} != prevalence {prevalence}"
