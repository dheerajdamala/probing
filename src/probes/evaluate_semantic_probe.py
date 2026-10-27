"""Semantic probe evaluation on held-out test data with calibration analysis."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.probes.train_semantic_probe import evaluate_probe_model, extract_probe_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_test_split(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    probe_file: Path | str,
    layer_idx: int,
    test_file: Path | str = "data/processed/test.json",
    pooling: str = "last_token",
) -> Dict[str, Any]:
    """Evaluate trained probe on held-out test set."""
    with open(probe_file, "rb") as f:
        probe = pickle.load(f)

    with open(test_file, "r", encoding="utf-8") as f:
        test_records = json.load(f)["probe_records"]

    X_test, y_test = extract_probe_dataset(model, tokenizer, test_records, layer_idx=layer_idx, pooling=pooling)
    metrics = evaluate_probe_model(probe, X_test, y_test)

    preds = probe.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test, preds).tolist()

    out = {
        "layer": layer_idx,
        "num_test_samples": len(y_test),
        "metrics": metrics,
        "confusion_matrix": conf_mat,
        "classification_report": report,
    }

    logger.info(
        f"Test Evaluation (Layer {layer_idx}) -> Accuracy: {metrics['accuracy']:.2%}, "
        f"F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}, Brier: {metrics['brier_score']:.4f}"
    )
    return out


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    res = evaluate_test_split(m, tok, "outputs/probes/probe_layer_2.pkl", layer_idx=2)
    print("Test Probe Eval:", res["metrics"])
