"""Training script for supervised security probe.

Loads representation datasets, verifies data integrity and leakage guards,
trains lightweight linear and MLP probes, tunes validation decision thresholds,
and exports serialized models and comprehensive metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from security_probe.src.evaluation import compute_detection_metrics, compute_per_cwe_metrics, find_optimal_threshold
from security_probe.src.feature_extractor import extract_code_representations
from security_probe.src.probe_model import SecurityProbe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path | str = "security_probe/config.yaml") -> Dict[str, Any]:
    """Load configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_and_load_records(file_path: Path | str) -> List[Dict[str, Any]]:
    """Validate existence and schema of dataset JSON files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset file missing: {path.resolve()}\n"
            "Please ensure datasets exist in data/processed/ or specify valid paths in config.yaml."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON file {path}: {e}")

    # Extract records: prioritize 'probe_records', fallback to paired conversion
    if isinstance(data, dict) and "probe_records" in data:
        records = data["probe_records"]
    elif isinstance(data, dict) and "paired" in data:
        records = []
        for pair in data["paired"]:
            p_id = pair.get("id", "unknown")
            cwe = pair.get("cwe", "CWE-UNKNOWN")
            prompt = pair.get("prompt", "")
            safe_comp = pair.get("safe_completion", "")
            vuln_comp = pair.get("vulnerable_completion", "")
            records.append({
                "id": f"{p_id}_safe",
                "parent_id": p_id,
                "cwe": cwe,
                "code": prompt + safe_comp,
                "semantic_label": 0,
            })
            records.append({
                "id": f"{p_id}_vuln",
                "parent_id": p_id,
                "cwe": cwe,
                "code": prompt + vuln_comp,
                "semantic_label": 1,
            })
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            f"Dataset at {path} does not match expected schema. Must contain 'probe_records' or 'paired'."
        )

    if not records:
        raise ValueError(f"Dataset at {path} contains 0 records.")

    # Validate essential fields
    for idx, r in enumerate(records):
        if "code" not in r:
            raise KeyError(f"Record at index {idx} in {path} is missing required field 'code'.")
        if "semantic_label" not in r:
            raise KeyError(f"Record at index {idx} in {path} is missing required field 'semantic_label'.")

    return records


def audit_split_leakage(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit train, validation, and test splits for code duplicate leakage."""
    train_codes = {r["code"].strip() for r in train_records if r.get("code")}
    val_codes = {r["code"].strip() for r in val_records if r.get("code")}
    test_codes = {r["code"].strip() for r in test_records if r.get("code")}

    train_val_overlap = train_codes.intersection(val_codes)
    train_test_overlap = train_codes.intersection(test_codes)
    val_test_overlap = val_codes.intersection(test_codes)

    leakage_detected = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    return {
        "leakage_detected": leakage_detected,
        "train_val_duplicates": len(train_val_overlap),
        "train_test_duplicates": len(train_test_overlap),
        "val_test_duplicates": len(val_test_overlap),
    }


def extract_split_features(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    layer_idx: int,
    pooling: str = "mean_prompt",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract representations from frozen model for a dataset split."""
    X_list = []
    y_list = []
    cwe_list = []

    for item in records:
        code = item["code"]
        label = int(item["semantic_label"])
        cwe = str(item.get("cwe", "CWE-UNKNOWN"))

        res = extract_code_representations(
            model=model,
            tokenizer=tokenizer,
            code=code,
            layer_idx=layer_idx,
            pooling=pooling,
        )
        X_list.append(res["pooled_rep"])
        y_list.append(label)
        cwe_list.append(cwe)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, cwe_list


def run_training_pipeline(
    config_path: str = "security_probe/config.yaml",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute full probe training and evaluation workflow."""
    cfg = load_config(config_path)
    p_cfg = cfg["probe"]
    paths = cfg["paths"]

    # 1. Load and validate data splits
    train_records = validate_and_load_records(paths["train_data"])
    val_records = validate_and_load_records(paths["val_data"])
    test_records = validate_and_load_records(paths["test_data"])

    # 2. Check for data leakage
    leakage_audit = audit_split_leakage(train_records, val_records, test_records)
    if leakage_audit["leakage_detected"]:
        logger.warning(f"Data leakage detected across splits: {leakage_audit}")

    # 3. Model setup (load model if not provided)
    if model is None or tokenizer is None:
        from src.model.load_model import load_model_and_tokenizer
        logger.info(f"Loading base model: {p_cfg['model_id']}")
        model, tokenizer = load_model_and_tokenizer(model_id=p_cfg["model_id"])

    layer_idx = p_cfg.get("selected_layer", 16)

    # 4. Extract representations
    logger.info("Extracting training representations...")
    X_train, y_train, cwe_train = extract_split_features(
        model, tokenizer, train_records, layer_idx, pooling=p_cfg.get("pooling_strategy", "mean_prompt")
    )
    logger.info("Extracting validation representations...")
    X_val, y_val, cwe_val = extract_split_features(
        model, tokenizer, val_records, layer_idx, pooling=p_cfg.get("pooling_strategy", "mean_prompt")
    )
    logger.info("Extracting test representations...")
    X_test, y_test, cwe_test = extract_split_features(
        model, tokenizer, test_records, layer_idx, pooling=p_cfg.get("pooling_strategy", "mean_prompt")
    )

    # 5. Initialize and fit probe
    probe = SecurityProbe(
        architecture=p_cfg.get("architecture", "linear"),
        regularization_c=p_cfg.get("regularization_c", 1.0),
        random_state=p_cfg.get("random_seed", 42),
        decision_threshold=p_cfg.get("decision_threshold", 0.50),
        uncertainty_margin=p_cfg.get("uncertainty_margin", 0.08),
        selected_layer=layer_idx,
    )
    probe.fit(X_train, y_train, cwe_labels=cwe_train, max_iter=p_cfg.get("max_iter", 1000))

    # 6. Tune decision threshold on validation set
    val_probs = probe.predict_proba(X_val)
    best_thresh, best_val_f1 = find_optimal_threshold(y_val, val_probs)
    probe.decision_threshold = best_thresh
    logger.info(f"Optimal validation threshold: {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")

    # 7. Evaluate on held-out test set
    test_probs = probe.predict_proba(X_test)
    test_preds = (test_probs >= probe.decision_threshold).astype(int)
    test_metrics = compute_detection_metrics(y_test, test_preds, test_probs)
    per_cwe_metrics = compute_per_cwe_metrics(list(y_test), list(test_preds), cwe_test)

    # 8. Serialize probe and metrics
    model_save_dir = Path(paths["model_save_dir"])
    model_save_dir.mkdir(parents=True, exist_ok=True)
    probe_file = model_save_dir / "probe_model.pkl"
    probe.save(probe_file)

    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "probe_evaluation_metrics.json"

    evaluation_report = {
        "model_id": p_cfg.get("model_id"),
        "selected_layer": layer_idx,
        "feature_dim": int(X_train.shape[1]),
        "architecture": probe.architecture,
        "optimal_threshold": best_thresh,
        "validation_f1": best_val_f1,
        "test_metrics": test_metrics,
        "per_cwe_metrics": per_cwe_metrics,
        "data_leakage_audit": leakage_audit,
        "sample_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info(f"Saved evaluation metrics to {metrics_file}.")
    return evaluation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Security Probe")
    parser.add_argument("--config", type=str, default="security_probe/config.yaml")
    args = parser.parse_args()
    run_training_pipeline(config_path=args.config)
