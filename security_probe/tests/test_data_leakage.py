"""Unit tests for data leakage prevention and evaluation label integrity."""

from __future__ import annotations

import pytest
import numpy as np

from security_probe.src.train import audit_split_leakage, validate_and_load_records
from security_probe.src.evaluation import compute_detection_metrics


def test_real_dataset_split_leakage():
    """Verify that actual train, val, and test splits contain no duplicate code."""
    train_records = validate_and_load_records("data/processed/train.json")
    val_records = validate_and_load_records("data/processed/val.json")
    test_records = validate_and_load_records("data/processed/test.json")

    audit = audit_split_leakage(train_records, val_records, test_records)

    assert audit["leakage_detected"] is False, f"Data leakage detected across splits: {audit}"
    assert audit["train_val_duplicates"] == 0
    assert audit["train_test_duplicates"] == 0
    assert audit["val_test_duplicates"] == 0


def test_synthetic_leakage_detection():
    """Verify that audit_split_leakage accurately catches intentional cross-split duplicate code."""
    train_mock = [{"code": "def foo(): pass", "semantic_label": 0}]
    val_mock = [{"code": "def foo(): pass", "semantic_label": 0}] # Duplicate!
    test_mock = [{"code": "def bar(): pass", "semantic_label": 1}]

    audit = audit_split_leakage(train_mock, val_mock, test_mock)

    assert audit["leakage_detected"] is True
    assert audit["train_val_duplicates"] == 1
    assert audit["train_test_duplicates"] == 0


def test_evaluation_rejects_missing_and_unevaluable_labels():
    """Test 8: Evaluation does not silently count missing or unevaluable labels as correct."""
    # Empty ground truth must return an error dictionary, not 1.0 accuracy
    res_empty = compute_detection_metrics(np.array([]), np.array([]))
    assert "error" in res_empty

    # Verify that missing label in record raises a KeyError during validation
    invalid_record = [{"code": "def foo(): pass"}] # missing 'semantic_label'
    import tempfile, json
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
        json.dump({"probe_records": invalid_record}, tmp)
        tmp_name = tmp.name

    with pytest.raises(KeyError, match="semantic_label"):
        validate_and_load_records(tmp_name)
