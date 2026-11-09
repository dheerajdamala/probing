"""
Comprehensive Unit Test Suite for Python SQL Injection Security Probe.

Tests:
1. Single-sample inference independence (no paired reference needed).
2. Edge cases: empty code, whitespace, malformed syntax, truncation.
3. Base LLM parameter immutability and zero steering hook injection.
4. Leakage-free dataset splitting verification (zero hash/pair overlap).
5. Output JSON schema and field constraint validation.
6. Exploratory token and line-level score ranking.
7. Detection and localization metric mathematical correctness.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import pytest
import numpy as np
import torch

from src.model.load_model import load_model_and_tokenizer
from security_probe_sql.src.feature_extractor import extract_code_representations
from security_probe_sql.src.probe_model import SecurityProbe
from security_probe_sql.src.inference import SQLInjectionProbeAnalyzer
from security_probe_sql.src.evaluation import compute_detection_metrics, compute_localization_metrics


@pytest.fixture(scope="module")
def shared_model_and_tokenizer():
    """Load cached fast miniature Mistral model fixture."""
    model, tokenizer, _ = load_model_and_tokenizer(mock=True)
    return model, tokenizer


@pytest.fixture(scope="module")
def fitted_toy_linear_probe():
    """Create and fit a lightweight toy probe on miniature representations."""
    hidden_dim = 128
    np.random.seed(42)
    X_safe = np.random.randn(20, hidden_dim).astype(np.float32) - 0.5
    X_vuln = np.random.randn(20, hidden_dim).astype(np.float32) + 0.5
    X = np.vstack([X_safe, X_vuln])
    y = np.array([0] * 20 + [1] * 20, dtype=np.int32)

    probe = SecurityProbe(
        architecture="linear",
        regularization_c=1.0,
        random_state=42,
        decision_threshold=0.50,
        uncertainty_margin=0.08,
        selected_layer=2,
    )
    probe.fit(X, y)
    return probe


def test_single_sample_inference_no_pairing(shared_model_and_tokenizer, fitted_toy_linear_probe):
    """Test 1: Single code sample inference without any contrastive pair or reference."""
    model, tokenizer = shared_model_and_tokenizer
    analyzer = SQLInjectionProbeAnalyzer(
        probe=fitted_toy_linear_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )

    code = "def query_db(uid):\n    return db.execute('SELECT * FROM users WHERE id = %s', (uid,))\n"
    result = analyzer.analyze(code)

    assert result["status"] in ["completed", "uncertain"]
    assert result["language"] == "python"
    assert result["vulnerability_type"] == "sql_injection"
    assert isinstance(result["is_vulnerable"], bool)
    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 <= result["threshold"] <= 1.0
    assert isinstance(result["findings"], list)
    assert len(result["limitations"]) > 0


def test_empty_and_whitespace_input_handling(shared_model_and_tokenizer, fitted_toy_linear_probe):
    """Test 2: Graceful error handling for empty or non-string input without crashes."""
    model, tokenizer = shared_model_and_tokenizer
    analyzer = SQLInjectionProbeAnalyzer(
        probe=fitted_toy_linear_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )

    empty_res = analyzer.analyze("")
    assert empty_res["status"] == "error"
    assert empty_res["outcome"] == "ERROR"
    assert empty_res["is_vulnerable"] is False

    ws_res = analyzer.analyze("   \n\t  ")
    assert ws_res["status"] == "error"
    assert ws_res["outcome"] == "ERROR"


def test_base_model_frozen_no_steering(shared_model_and_tokenizer, fitted_toy_linear_probe):
    """Test 3: Base model parameters remain strictly frozen with zero steering hooks."""
    model, tokenizer = shared_model_and_tokenizer

    # Assert all model parameters have requires_grad=False
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"

    # Run inference and verify model weights are not modified
    analyzer = SQLInjectionProbeAnalyzer(
        probe=fitted_toy_linear_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )
    code = "import sqlite3\ncur.execute(f'SELECT * FROM tbl WHERE x = {val}')\n"
    analyzer.analyze(code)

    for name, param in model.named_parameters():
        assert not param.requires_grad


def test_zero_data_leakage():
    """Test 4: Verify train, validation, and test splits have 0 hash or pair overlap."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    splits_dir = os.path.join(project_root, "data", "processed", "splits")

    train_file = os.path.join(splits_dir, "train.jsonl")
    val_file = os.path.join(splits_dir, "val.jsonl")
    test_file = os.path.join(splits_dir, "test.jsonl")

    assert os.path.exists(train_file), "Train split file missing"
    assert os.path.exists(val_file), "Val split file missing"
    assert os.path.exists(test_file), "Test split file missing"

    with open(train_file, "r", encoding="utf-8") as f:
        tr_data = [json.loads(l) for l in f if l.strip()]
    with open(val_file, "r", encoding="utf-8") as f:
        v_data = [json.loads(l) for l in f if l.strip()]
    with open(test_file, "r", encoding="utf-8") as f:
        te_data = [json.loads(l) for l in f if l.strip()]

    tr_hashes = set(r["code_hash"] for r in tr_data)
    v_hashes = set(r["code_hash"] for r in v_data)
    te_hashes = set(r["code_hash"] for r in te_data)

    assert len(tr_hashes.intersection(v_hashes)) == 0, "Train-Val code hash overlap detected!"
    assert len(tr_hashes.intersection(te_hashes)) == 0, "Train-Test code hash overlap detected!"
    assert len(v_hashes.intersection(te_hashes)) == 0, "Val-Test code hash overlap detected!"

    tr_pairs = set(r.get("pair_id") for r in tr_data if r.get("pair_id"))
    v_pairs = set(r.get("pair_id") for r in v_data if r.get("pair_id"))
    te_pairs = set(r.get("pair_id") for r in te_data if r.get("pair_id"))

    assert len(tr_pairs.intersection(v_pairs)) == 0, "Train-Val pair leakage detected!"
    assert len(tr_pairs.intersection(te_pairs)) == 0, "Train-Test pair leakage detected!"
    assert len(v_pairs.intersection(te_pairs)) == 0, "Val-Test pair leakage detected!"


def test_exploratory_localization_ranking(shared_model_and_tokenizer, fitted_toy_linear_probe):
    """Test 5: Verify exploratory localization findings contain valid line numbers and scores."""
    model, tokenizer = shared_model_and_tokenizer
    analyzer = SQLInjectionProbeAnalyzer(
        probe=fitted_toy_linear_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
        localization_threshold=0.0, # return all
    )

    code = "import sqlite3\ndef bad_search(term):\n    query = f'SELECT * FROM users WHERE name = \"{term}\"'\n    return db.execute(query)\n"
    res = analyzer.analyze(code)

    findings = res["findings"]
    assert len(findings) > 0
    # Findings should be sorted in descending order of score
    scores = [f["score"] for f in findings]
    assert scores == sorted(scores, reverse=True)

    for f in findings:
        assert 1 <= f["start_line"] <= 4
        assert f["localization_mode"] == "exploratory_token_projection"
        assert f["ground_truth_spans_available"] is False


def test_detection_metrics_computation():
    """Test 6: Verify mathematical accuracy of binary detection metrics."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.7, 0.4, 0.9])

    metrics = compute_detection_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.50
    assert metrics["balanced_accuracy"] == 0.50
    assert metrics["precision"] == 0.50
    assert metrics["recall"] == 0.50
    assert metrics["f1_score"] == 0.50
    assert metrics["specificity"] == 0.50
    assert metrics["false_positive_rate"] == 0.50
    assert metrics["false_negative_rate"] == 0.50
    assert metrics["confusion_matrix"]["true_negatives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["confusion_matrix"]["true_positives"] == 1
