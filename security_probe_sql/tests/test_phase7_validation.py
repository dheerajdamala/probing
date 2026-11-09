"""
Automated Unit Tests for Phase 7 Validation, SQL Sink Extraction, and Checkpoint Verification.
"""

from __future__ import annotations

import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def phase7_paths():
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return {
        "proj_root": proj_root,
        "metrics_json": os.path.join(proj_root, "outputs", "metrics", "PHASE7_RESULTS.json"),
        "pred_csv": os.path.join(proj_root, "outputs", "predictions", "PHASE7_TEST_PREDICTIONS.csv"),
        "graphs_dir": os.path.join(proj_root, "outputs", "graphs", "PHASE7_VALIDATION"),
        "ckpt_path": os.path.join(proj_root, "models", "probe_head_layer3_sql_sink.pkl"),
        "test_split": os.path.join(proj_root, "data", "processed", "splits", "test.jsonl"),
    }


def test_phase7_artifacts_exist(phase7_paths):
    """Test that all Phase 7 JSON metrics, prediction CSV, and graphs exist."""
    assert os.path.exists(phase7_paths["metrics_json"]), "Missing PHASE7_RESULTS.json"
    assert os.path.getsize(phase7_paths["metrics_json"]) > 0, "PHASE7_RESULTS.json is empty"

    assert os.path.exists(phase7_paths["pred_csv"]), "Missing PHASE7_TEST_PREDICTIONS.csv"
    assert os.path.getsize(phase7_paths["pred_csv"]) > 0, "PHASE7_TEST_PREDICTIONS.csv is empty"

    assert os.path.exists(phase7_paths["ckpt_path"]), "Missing probe_head_layer3_sql_sink.pkl"

    expected_graphs = [
        "01_phase5_vs_phase6_vs_phase7_roc_auc.png",
        "02_average_precision_comparison.png",
        "03_repository_specific_and_loro_comparison.png",
        "04_sql_12pair_score_differences.png",
        "05_real_vs_shuffled_label_distribution.png",
        "06_sink_extraction_coverage_and_fallbacks.png",
        "07_mistral_hardware_and_dimension_comparison.png",
        "08_layer_and_pooling_progression.png",
        "09_confusion_matrices_comparison.png",
        "10_precision_recall_curves_phase7.png",
        "11_runtime_throughput_and_memory_profiles.png",
    ]
    for g in expected_graphs:
        gp = os.path.join(phase7_paths["graphs_dir"], g)
        assert os.path.exists(gp), f"Missing graph: {g}"
        assert os.path.getsize(gp) > 1000, f"Graph file unexpectedly small: {g}"


def test_phase7_predictions_align_with_test_split(phase7_paths):
    """Verify that predictions match the held-out test split exactly."""
    test_recs = []
    with open(phase7_paths["test_split"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_recs.append(json.loads(line))

    assert len(test_recs) == 456
    expected_ids = [r["sample_id"] for r in test_recs]
    expected_labels = [r["label"] for r in test_recs]

    df = pd.read_csv(phase7_paths["pred_csv"])
    assert len(df) == 456
    assert list(df["sample_id"]) == expected_ids
    assert list(df["ground_truth_label"]) == expected_labels
    assert df["phase7_layer3_sink_prob"].between(0.0, 1.0).all()


def test_layer3_sink_probe_metrics(phase7_paths):
    """Verify that Phase 7 metrics match authoritative reproduced figures."""
    with open(phase7_paths["metrics_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    l3_m = data["model_evaluations"]["layer3_sql_sink_probe"]
    assert pytest.approx(l3_m["roc_auc"], rel=1e-3) == 0.8237
    assert pytest.approx(l3_m["average_precision_score"], rel=1e-3) == 0.5007
    assert pytest.approx(l3_m["balanced_accuracy"], rel=1e-3) == 0.7558
    assert pytest.approx(l3_m["f1_score"], rel=1e-3) == 0.5551
    assert l3_m["confusion_matrix"]["matrix_2x2"] == [[279, 86], [23, 68]]


def test_shuffled_controls_significance(phase7_paths):
    """Verify that real-label Layer 3 sink probe is statistically superior to shuffled noise."""
    with open(phase7_paths["metrics_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    shuf = data["shuffled_label_controls"]
    assert len(shuf["seeds"]) == 5
    mean_shuf_roc = shuf["aggregate_statistics"]["roc_auc_mean"]
    real_roc = shuf["real_label_comparison"]["real_roc_auc"]
    z_score = shuf["real_label_comparison"]["z_score"]

    assert 0.40 <= mean_shuf_roc <= 0.55
    assert real_roc > mean_shuf_roc + 0.20
    assert z_score > 5.0, f"Expected z-score > 5.0, got {z_score}"


def test_12pair_diagnostics_coverage(phase7_paths):
    """Verify that 12 pairs were evaluated and ranking accuracy is reported."""
    with open(phase7_paths["metrics_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    diag = data["controlled_sql_dataflow_diagnostics_12_pairs"]
    assert diag["total_pairs_evaluated"] == 12
    assert len(diag["pairs"]) == 12
    assert diag["tfidf_ranking_accuracy"] == 1.0
    # Neural probes fail on most pairs due to lack of dataflow syntax awareness
    assert diag["layer3_sink_ranking_accuracy"] <= 0.60


def test_mistral_hardware_audit_boundary(phase7_paths):
    """Verify that transfer analysis correctly audited hardware boundary."""
    with open(phase7_paths["metrics_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    transfer = data["pretrained_mistral_transfer_analysis"]
    hw = transfer["hardware_profile"]
    feas = transfer["resource_feasibility_analysis"]

    assert hw["cuda_available"] is False
    assert feas["can_execute_float16_locally"] is False
    assert "STOP_AND_DOCUMENT_LIMITATION" in feas["protocol_action"]


def test_localization_audit_sample_count(phase7_paths):
    """Verify localization audit correctly reports strictly 2 test annotations."""
    with open(phase7_paths["metrics_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    loc = data["localization_ground_truth_audit"]
    assert loc["annotated_test_sample_count"] == 2
    assert len(loc["samples_evaluated"]) == 2

