"""
Unit Tests for Phase 6 Diagnostics, Ablation Artifacts, and Baseline Freezing.
"""

from __future__ import annotations

import os
import json
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def base_paths():
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return {
        "proj_root": proj_root,
        "metrics_dir": os.path.join(proj_root, "outputs", "metrics"),
        "predictions_dir": os.path.join(proj_root, "outputs", "predictions"),
        "graphs_dir": os.path.join(proj_root, "outputs", "graphs", "PHASE6_DIAGNOSTICS"),
        "test_split": os.path.join(proj_root, "data", "processed", "splits", "test.jsonl"),
    }


def test_phase6_artifacts_exist(base_paths):
    """Test that all required Phase 6 JSON metrics, CSV predictions, and graphs exist."""
    m_dir = base_paths["metrics_dir"]
    g_dir = base_paths["graphs_dir"]
    p_dir = base_paths["predictions_dir"]

    required_metrics = [
        "PHASE6_BASELINE_SNAPSHOT.json",
        "PHASE6_ABLATION_RESULTS.json",
        "PHASE6_SHUFFLED_LABEL_CONTROLS.json",
        "PHASE6_SQL_DATAFLOW_DIAGNOSTICS.json",
    ]
    for rm in required_metrics:
        p = os.path.join(m_dir, rm)
        assert os.path.exists(p), f"Missing metric artifact: {p}"
        assert os.path.getsize(p) > 0, f"Empty metric artifact: {p}"

    pred_csv = os.path.join(p_dir, "PHASE6_SELECTED_MODEL_PREDICTIONS.csv")
    assert os.path.exists(pred_csv), f"Missing prediction CSV: {pred_csv}"
    assert os.path.getsize(pred_csv) > 0, f"Empty prediction CSV: {pred_csv}"

    required_graphs = [
        "01_roc_auc_and_ap_across_representations.png",
        "02_layer_and_pooling_heatmap.png",
        "03_precision_recall_curves.png",
        "04_confusion_matrices_at_thresholds.png",
        "05_validation_threshold_tradeoffs.png",
        "06_shuffled_label_controls.png",
        "07_performance_by_data_source.png",
        "08_paired_sql_dataflow_diagnostics.png",
        "09_runtime_and_memory_profiles.png",
    ]
    for rg in required_graphs:
        gp = os.path.join(g_dir, rg)
        assert os.path.exists(gp), f"Missing graph: {gp}"
        assert os.path.getsize(gp) > 1000, f"Graph file unexpectedly small: {gp}"


def test_phase6_predictions_align_with_test_split(base_paths):
    """Verify that predictions match the held-out test split exactly."""
    test_recs = []
    with open(base_paths["test_split"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_recs.append(json.loads(line))

    assert len(test_recs) == 456, f"Expected 456 test records, got {len(test_recs)}"
    expected_ids = [r["sample_id"] for r in test_recs]
    expected_labels = [r["label"] for r in test_recs]

    pred_csv = os.path.join(base_paths["predictions_dir"], "PHASE6_SELECTED_MODEL_PREDICTIONS.csv")
    df = pd.read_csv(pred_csv)

    assert len(df) == 456, f"Expected 456 prediction rows, got {len(df)}"
    assert list(df["sample_id"]) == expected_ids, "Sample IDs do not align exactly with test split"
    assert list(df["ground_truth_label"]) == expected_labels, "Labels do not match test split"

    # Verify probability bounds
    for col in ["baseline_linear_prob", "baseline_mlp_prob", "tfidf_prob"]:
        assert df[col].between(0.0, 1.0).all(), f"Probabilities in {col} outside [0, 1]"


def test_baseline_snapshot_metrics_consistency(base_paths):
    """Verify that Phase 6 baseline snapshot matches Phase 5 authoritative baseline."""
    snap_p = os.path.join(base_paths["metrics_dir"], "PHASE6_BASELINE_SNAPSHOT.json")
    with open(snap_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    lin_m = data["authoritative_baseline_metrics"]["neural_linear_probe"]
    assert pytest.approx(lin_m["roc_auc"], rel=1e-4) == 0.529941
    assert pytest.approx(lin_m["average_precision_score"], rel=1e-4) == 0.225890
    assert pytest.approx(lin_m["balanced_accuracy"], rel=1e-4) == 0.545160

    tf_m = data["authoritative_baseline_metrics"]["tfidf_logistic_regression"]
    assert pytest.approx(tf_m["roc_auc"], rel=1e-4) == 0.884149
    assert pytest.approx(tf_m["average_precision_score"], rel=1e-4) == 0.594973


def test_shuffled_label_sanity_check(base_paths):
    """Verify that shuffled label controls are near chance and evaluate expected variance."""
    shuf_p = os.path.join(base_paths["metrics_dir"], "PHASE6_SHUFFLED_LABEL_CONTROLS.json")
    with open(shuf_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data["shuffled_runs"]
    assert len(runs) == 5, f"Expected 5 shuffled seeds, got {len(runs)}"
    # Aggregate mean across shuffled runs should be close to chance (0.50) within finite sample noise
    mean_roc = data["aggregate_statistics"]["roc_auc_mean"]
    assert 0.45 <= mean_roc <= 0.58, f"Shuffled aggregate mean ROC-AUC out of chance range: {mean_roc}"
    for r in runs:
        # Individual random permutation draws can fluctuate due to finite test set variance
        assert 0.35 <= r["roc_auc"] <= 0.68, f"Shuffled ROC-AUC out of reasonable variance bounds: {r['roc_auc']}"


def test_dataflow_diagnostics_structure(base_paths):
    """Verify that all 6 diagnostic pairs were evaluated and recorded."""
    diag_p = os.path.join(base_paths["metrics_dir"], "PHASE6_SQL_DATAFLOW_DIAGNOSTICS.json")
    with open(diag_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["diagnostic_pairs_evaluated"] == 6
    assert len(data["diagnostic_results"]) == 6
    assert data["tfidf_pairwise_ranking_accuracy"] == 1.0  # TF-IDF passes 6/6
