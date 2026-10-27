"""Semantic evaluation module using trained probes."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model.extract_activations import extract_layer_hidden_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_semantic_probe_on_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    code: str,
    probe_file: Path | str,
    layer_idx: int = 2,
    pooling: str = "last_token",
) -> Dict[str, Any]:
    """Score code completion using trained semantic probe."""
    p_path = Path(probe_file)
    if not p_path.exists():
        return {
            "probe_available": False,
            "predicted_semantic_label": None,
            "confidence": None,
        }

    with open(p_path, "rb") as f:
        probe = pickle.load(f)

    acts = extract_layer_hidden_states(model, tokenizer, code, target_layers=[layer_idx], pooling=pooling)
    vec = acts[layer_idx].numpy().reshape(1, -1)

    pred_label = int(probe.predict(vec)[0])
    prob_vuln = float(probe.predict_proba(vec)[0, 1]) if hasattr(probe, "predict_proba") else float(pred_label)

    return {
        "probe_available": True,
        "layer_idx": layer_idx,
        "predicted_semantic_label": pred_label, # 0 = safe/sanitized, 1 = unsanitized flow
        "prob_unsanitized_flow": round(prob_vuln, 4),
        "is_predicted_safe": (pred_label == 0),
    }


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    res = evaluate_semantic_probe_on_completion(m, tok, "def test():\n    pass\n", "outputs/probes/probe_layer_2.pkl")
    print("Semantic Probe Score on completion:", res)
