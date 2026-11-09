"""Comprehensive experimental runner for Security Probe Phase 2 validation."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.model.load_model import load_model_and_tokenizer
from security_probe.src.feature_extractor import extract_code_representations
from security_probe.src.train import extract_split_features, validate_and_load_records, audit_split_leakage
from security_probe.src.probe_model import SecurityProbe
from security_probe.src.evaluation import compute_detection_metrics, find_optimal_threshold, compute_per_cwe_metrics
from security_probe.src.inference import SecurityProbeAnalyzer


def run_all_phase2():
    print("=" * 80)
    print("STARTING SECURITY PROBE PHASE 2 EMPIRICAL VALIDATION")
    print("=" * 80)

    # 1. Load data
    train_pr = validate_and_load_records("data/processed/train.json")
    val_pr = validate_and_load_records("data/processed/val.json")
    test_pr = validate_and_load_records("data/processed/test.json")

    print(f"Loaded splits: Train={len(train_pr)}, Val={len(val_pr)}, Test={len(test_pr)}")

    # 2. Load model
    t_load_start = time.time()
    model, tokenizer, model_info = load_model_and_tokenizer(mock=True)
    model_load_time = round(time.time() - t_load_start, 4)
    print(f"Model loaded in {model_load_time}s ({model_info.get('device')})")

    layer_idx = 2 # Midpoint for 4-layer miniature model

    # 3. Extract representations
    print("\n--- Extracting Features ---")
    X_train, y_train, cwe_train = extract_split_features(model, tokenizer, train_pr, layer_idx=layer_idx)
    X_val, y_val, cwe_val = extract_split_features(model, tokenizer, val_pr, layer_idx=layer_idx)
    X_test, y_test, cwe_test = extract_split_features(model, tokenizer, test_pr, layer_idx=layer_idx)

    # 4. Train Probe
    print("\n--- Training Probe on Train Split Only ---")
    probe = SecurityProbe(
        architecture="linear",
        regularization_c=1.0,
        random_state=42,
        selected_layer=layer_idx,
    )
    probe.fit(X_train, y_train, cwe_labels=cwe_train)

    # 5. Tune Threshold on Validation Data Only
    print("\n--- Tuning Decision Threshold on Validation Data Only ---")
    val_probs = probe.predict_proba(X_val)
    best_thresh, best_val_f1 = find_optimal_threshold(y_val, val_probs)
    print(f"Selected Threshold: {best_thresh:.2f} (Validation F1: {best_val_f1:.4f})")
    probe.decision_threshold = best_thresh

    # 6. Single Evaluation on Held-Out Test Set
    print("\n--- Evaluating Once on Held-Out Test Split ---")
    test_probs = probe.predict_proba(X_test)
    test_preds = probe.predict(X_test)

    probe_metrics = compute_detection_metrics(y_test, test_preds, test_probs)
    per_cwe = compute_per_cwe_metrics(list(y_test), list(test_preds), cwe_test)

    print("Probe Test Metrics:", probe_metrics)
    print("Per-CWE Breakdown:", per_cwe)

    # 7. Baselines on Test Set
    print("\n--- Evaluating Baselines on Test Split ---")
    # Baseline 1: Majority Class (predict all 0 or all 1)
    maj_pred = np.zeros_like(y_test)
    maj_metrics = compute_detection_metrics(y_test, maj_pred)

    # Baseline 2: Code Length Classifier (trained on train, evaluated on test)
    X_len_train = np.array([[len(r["code"])] for r in train_pr])
    X_len_test = np.array([[len(r["code"])] for r in test_pr])
    clf_len = LogisticRegression().fit(X_len_train, y_train)
    len_probs = clf_len.predict_proba(X_len_test)[:, 1]
    len_preds = clf_len.predict(X_len_test)
    len_metrics = compute_detection_metrics(y_test, len_preds, len_probs)

    # Baseline 3: TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=100)
    X_tfidf_train = tfidf.fit_transform([r["code"] for r in train_pr])
    X_tfidf_test = tfidf.transform([r["code"] for r in test_pr])
    clf_tfidf = LogisticRegression(C=1.0).fit(X_tfidf_train, y_train)
    tfidf_probs = clf_tfidf.predict_proba(X_tfidf_test)[:, 1]
    tfidf_preds = clf_tfidf.predict(X_tfidf_test)
    tfidf_metrics = compute_detection_metrics(y_test, tfidf_preds, tfidf_probs)

    print("Majority Class Baseline:", maj_metrics)
    print("Length-Only Baseline:   ", len_metrics)
    print("TF-IDF Baseline:        ", tfidf_metrics)

    # 8. Generate Per-Sample Test Predictions CSV
    analyzer = SecurityProbeAnalyzer(
        probe=probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
    )

    test_pred_rows = []
    for idx, r in enumerate(test_pr):
        code = r["code"]
        gt_label = int(r["semantic_label"])
        cwe = str(r.get("cwe", "CWE-UNKNOWN"))
        p_id = str(r.get("parent_id", "unknown"))
        sample_id = str(r.get("id", f"test_{idx}"))

        res = analyzer.analyze(code)

        test_pred_rows.append({
            "sample_id": sample_id,
            "parent_id": p_id,
            "cwe": cwe,
            "ground_truth_label": gt_label,
            "predicted_label": int(test_preds[idx]),
            "predicted_probability": round(float(test_probs[idx]), 4),
            "status": res["status"],
            "confidence": res["confidence"],
            "predicted_cwe": res["predicted_cwe"],
            "no_finding": res["no_finding"],
            "findings_count": len(res["findings"]),
            "code_chars": len(code),
            "code_snippet": code.replace("\n", "\\n")[:100] + "...",
        })

    df_test_preds = pd.DataFrame(test_pred_rows)
    test_preds_csv_path = "security_probe/outputs/TEST_PREDICTIONS.csv"
    df_test_preds.to_csv(test_preds_csv_path, index=False)
    print(f"\nSaved test predictions to {test_preds_csv_path}")

    # 9. Save DETECTION_METRICS.json
    detection_report = {
        "dataset_summary": {
            "train_samples": len(train_pr),
            "val_samples": len(val_pr),
            "test_samples": len(test_pr),
            "train_class_distribution": {"safe_0": int((y_train == 0).sum()), "vuln_1": int((y_train == 1).sum())},
            "val_class_distribution": {"safe_0": int((y_val == 0).sum()), "vuln_1": int((y_val == 1).sum())},
            "test_class_distribution": {"safe_0": int((y_test == 0).sum()), "vuln_1": int((y_test == 1).sum())},
        },
        "model_configuration": {
            "base_model": model_info.get("model_id", "mock-miniature-mistral"),
            "probe_architecture": "linear (LogisticRegression L2)",
            "selected_layer": layer_idx,
            "pooling_strategy": "mean_prompt",
            "feature_dimension": int(X_train.shape[1]),
            "regularization_c": 1.0,
            "decision_threshold": best_thresh,
            "uncertainty_margin": probe.uncertainty_margin,
            "threshold_selection_source": "validation_split_only",
        },
        "probe_detection_metrics": probe_metrics,
        "per_cwe_metrics": per_cwe,
        "baseline_comparisons": {
            "majority_class_baseline": maj_metrics,
            "length_only_baseline": len_metrics,
            "tfidf_logistic_regression_baseline": tfidf_metrics,
        },
        "error_and_unevaluable_counts": {
            "total_test_samples": len(test_pr),
            "successful_inferences": len(test_pred_rows),
            "failed_inferences": 0,
            "uncertain_predictions": int((df_test_preds["status"] == "UNCERTAIN").sum()),
            "missing_labels": 0,
        },
    }

    metrics_json_path = "security_probe/outputs/DETECTION_METRICS.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(detection_report, f, indent=2)
    print(f"Saved detection metrics to {metrics_json_path}")

    # 10. Runtime Benchmark
    print("\n--- Running Runtime Benchmarking ---")
    import psutil
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 * 1024)

    # Test latency on short, medium, long snippets
    short_code = "def add(x, y):\n    return x + y\n"
    medium_code = (
        "def process_user_data(cursor, user_id: str, is_admin: bool):\n"
        "    if not user_id:\n"
        "        raise ValueError('Missing user ID')\n"
        "    query = 'SELECT username, email, role FROM users WHERE id = %s'\n"
        "    cursor.execute(query, (user_id,))\n"
        "    record = cursor.fetchone()\n"
        "    if record and is_admin:\n"
        "        return {'status': 'success', 'data': record}\n"
        "    return None\n"
    )
    long_code = medium_code * 8 # ~80 lines

    benchmarks = {}
    for name, snippet in [("short_code", short_code), ("medium_code", medium_code), ("long_code", long_code)]:
        extract_latencies = []
        probe_latencies = []
        end_to_end_latencies = []

        for _ in range(20):
            t0 = time.time()
            ext = extract_code_representations(model, tokenizer, snippet, layer_idx=layer_idx)
            t_ext = time.time() - t0

            t1 = time.time()
            _ = probe.predict_proba(ext["pooled_rep"])
            _ = probe.score_token_hidden_states(ext["token_hidden_states"])
            t_probe = time.time() - t1

            t2 = time.time()
            _ = analyzer.analyze(snippet)
            t_e2e = time.time() - t2

            extract_latencies.append(t_ext)
            probe_latencies.append(t_probe)
            end_to_end_latencies.append(t_e2e)

        benchmarks[name] = {
            "code_lines": len(snippet.splitlines()),
            "code_chars": len(snippet),
            "avg_feature_extraction_ms": round(float(np.mean(extract_latencies) * 1000), 2),
            "std_feature_extraction_ms": round(float(np.std(extract_latencies) * 1000), 2),
            "avg_probe_prediction_ms": round(float(np.mean(probe_latencies) * 1000), 2),
            "avg_end_to_end_ms": round(float(np.mean(end_to_end_latencies) * 1000), 2),
        }

    ram_after = process.memory_info().rss / (1024 * 1024)

    benchmark_report = {
        "hardware_environment": {
            "platform": sys.platform,
            "cuda_available": False,
            "device": "CPU (Intel / AMD host)",
            "python_version": sys.version.split()[0],
            "note": "Local environment is CPU host; Colab/Kaggle T4 GPU profile documented in report.",
        },
        "model_loading": {
            "model_id": model_info.get("model_id"),
            "load_time_sec": model_load_time,
            "parameters": model_info.get("num_parameters"),
            "device": "cpu",
        },
        "memory_profile": {
            "process_rss_mb_before": round(ram_before, 2),
            "process_rss_mb_after": round(ram_after, 2),
            "memory_increase_mb": round(ram_after - ram_before, 2),
            "gpu_vram_peak_mb": None,
        },
        "latency_benchmarks": benchmarks,
    }

    benchmark_json_path = "security_probe/outputs/RUNTIME_BENCHMARK.json"
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"Saved runtime benchmark to {benchmark_json_path}")

    # 11. Shortcut & Leakage Audit Data
    print("\n--- Compiling Shortcut & Leakage Analysis ---")
    leakage = audit_split_leakage(train_pr, val_pr, test_pr)

    splits_stats = {}
    for name, split in [("train", train_pr), ("val", val_pr), ("test", test_pr)]:
        safe_lens = [len(r["code"]) for r in split if r["semantic_label"] == 0]
        vuln_lens = [len(r["code"]) for r in split if r["semantic_label"] == 1]
        safe_lines = [len(r["code"].splitlines()) for r in split if r["semantic_label"] == 0]
        vuln_lines = [len(r["code"].splitlines()) for r in split if r["semantic_label"] == 1]
        splits_stats[name] = {
            "safe_chars_mean": round(float(np.mean(safe_lens)), 1),
            "safe_chars_std": round(float(np.std(safe_lens)), 1),
            "vuln_chars_mean": round(float(np.mean(vuln_lens)), 1),
            "vuln_chars_std": round(float(np.std(vuln_lens)), 1),
            "char_length_diff": round(float(np.mean(safe_lens) - np.mean(vuln_lens)), 1),
            "safe_lines_mean": round(float(np.mean(safe_lines)), 1),
            "vuln_lines_mean": round(float(np.mean(vuln_lines)), 1),
            "parents_count": len(set(r.get("parent_id") for r in split)),
        }

    print("Shortcut stats:", splits_stats)
    print("Leakage audit:", leakage)
    print("\n=== PHASE 2 EXPERIMENTS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_all_phase2()
