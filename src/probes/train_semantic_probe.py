"""Semantic probe training module.

Trains linear (logistic regression) and MLP probes to predict program
semantic properties (e.g. source-to-sink unsanitized flow) directly
from intermediate Transformer hidden states.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model.extract_activations import extract_layer_hidden_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_probe_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    probe_records: List[Dict[str, Any]],
    layer_idx: int,
    pooling: str = "last_token",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target labels y for a specific layer."""
    X_list = []
    y_list = []

    for item in probe_records:
        code = item["code"]
        label = item["semantic_label"]

        acts = extract_layer_hidden_states(
            model=model,
            tokenizer=tokenizer,
            text=code,
            target_layers=[layer_idx],
            pooling=pooling,
        )
        vec = acts[layer_idx].numpy()
        X_list.append(vec)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a linear L2-regularized logistic regression probe."""
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_probe_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Evaluate probe performance including calibration and balanced accuracy."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds

    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    brier = brier_score_loss(y, probs)

    try:
        roc_auc = roc_auc_score(y, probs)
    except Exception:
        roc_auc = 0.5 # Default if single class or undefined

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "brier_score": round(float(brier), 4),
    }


def train_and_evaluate_layer_probes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_file: Path | str = "data/processed/train.json",
    val_file: Path | str = "data/processed/val.json",
    layers: Optional[List[int]] = None,
    output_dir: Path | str = "outputs/probes",
    pooling: str = "last_token",
) -> Dict[int, Dict[str, Any]]:
    """Train and evaluate probes across specified layers."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(train_file, "r", encoding="utf-8") as f:
        train_records = json.load(f)["probe_records"]
    with open(val_file, "r", encoding="utf-8") as f:
        val_records = json.load(f)["probe_records"]

    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    results: Dict[int, Dict[str, Any]] = {}

    for l in layers:
        logger.info(f"Extracting features for Layer {l} probe...")
        X_train, y_train = extract_probe_dataset(model, tokenizer, train_records, layer_idx=l, pooling=pooling)
        X_val, y_val = extract_probe_dataset(model, tokenizer, val_records, layer_idx=l, pooling=pooling)

        logger.info(f"Training probe for Layer {l} on {len(y_train)} samples...")
        probe = train_linear_probe(X_train, y_train)

        train_metrics = evaluate_probe_model(probe, X_train, y_train)
        val_metrics = evaluate_probe_model(probe, X_val, y_val)

        # Baseline comparison: random chance (50% for balanced safe/vuln pairs)
        majority_label = int(np.round(np.mean(y_train)))
        baseline_preds = np.full_like(y_val, majority_label)
        baseline_acc = float(accuracy_score(y_val, baseline_preds))

        results[l] = {
            "layer": l,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "baseline_accuracy": round(baseline_acc, 4),
            "num_features": X_train.shape[1],
        }

        # Save probe artifact
        probe_file = out_path / f"probe_layer_{l}.pkl"
        with open(probe_file, "wb") as f:
            pickle.dump(probe, f)

        logger.info(
            f"Layer {l} Probe Results -> Val Acc: {val_metrics['accuracy']:.2%} | "
            f"Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f} "
            f"(Baseline: {baseline_acc:.2%})"
        )

    summary_file = out_path / "probe_layer_sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    train_and_evaluate_layer_probes(m, tok, layers=[0, 1, 2, 3])
