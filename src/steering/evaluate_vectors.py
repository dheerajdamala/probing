"""Evaluation module for candidate security vectors on held-out data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model.extract_activations import extract_layer_hidden_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_vector_separation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    vector_bundle: Dict[str, Dict[int, torch.Tensor]],
    val_pairs: List[Dict[str, Any]],
    pooling: str = "last_token",
) -> Dict[int, Dict[str, Any]]:
    """Evaluate projection separation and ranking accuracy on held-out validation pairs."""
    sec_vectors = vector_bundle["security_vectors"]
    rand_vectors = vector_bundle.get("random_vectors", {})
    layers = sorted(list(sec_vectors.keys()))

    results: Dict[int, Dict[str, Any]] = {}

    for l in layers:
        v_sec = sec_vectors[l]
        v_rand = rand_vectors.get(l, None)

        sec_margins = []
        rand_margins = []
        sec_correct = 0
        rand_correct = 0

        for item in val_pairs:
            prompt = item["prompt"]
            safe_code = prompt + item["safe_completion"]
            vuln_code = prompt + item["vulnerable_completion"]

            safe_acts = extract_layer_hidden_states(model, tokenizer, safe_code, target_layers=[l], pooling=pooling)[l]
            vuln_acts = extract_layer_hidden_states(model, tokenizer, vuln_code, target_layers=[l], pooling=pooling)[l]

            # Security vector projections
            proj_safe = torch.dot(safe_acts, v_sec).item()
            proj_vuln = torch.dot(vuln_acts, v_sec).item()
            margin = proj_safe - proj_vuln
            sec_margins.append(margin)
            if margin > 0:
                sec_correct += 1

            # Random vector projections
            if v_rand is not None:
                r_proj_safe = torch.dot(safe_acts, v_rand).item()
                r_proj_vuln = torch.dot(vuln_acts, v_rand).item()
                r_margin = r_proj_safe - r_proj_vuln
                rand_margins.append(r_margin)
                if r_margin > 0:
                    rand_correct += 1

        n = len(val_pairs)
        results[l] = {
            "num_val_pairs": n,
            "security_vector": {
                "mean_margin": round(float(sum(sec_margins) / n), 4) if n > 0 else 0.0,
                "pairwise_accuracy": round(sec_correct / n, 4) if n > 0 else 0.0,
            },
            "random_vector_control": {
                "mean_margin": round(float(sum(rand_margins) / n), 4) if n > 0 else 0.0,
                "pairwise_accuracy": round(rand_correct / n, 4) if n > 0 else 0.0,
            } if v_rand is not None else None
        }

        logger.info(
            f"Layer {l} | Security Vector Margin: {results[l]['security_vector']['mean_margin']:.4f} "
            f"(Acc: {results[l]['security_vector']['pairwise_accuracy']:.2%}) | "
            f"Random Margin: {results[l]['random_vector_control']['mean_margin']:.4f} "
            f"(Acc: {results[l]['random_vector_control']['pairwise_accuracy']:.2%})"
        )

    return results


def run_vector_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    vector_file: Path | str = "outputs/security_vectors.pt",
    val_file: Path | str = "data/processed/val.json",
    output_report_file: Path | str = "outputs/vector_evaluation_report.json",
) -> Dict[str, Any]:
    """Load cached vectors, run validation, and write report."""
    bundle = torch.load(vector_file, weights_only=True)
    with open(val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_pairs = val_data.get("paired", [])

    results = evaluate_vector_separation(model, tokenizer, bundle, val_pairs)

    report_path = Path(output_report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved vector evaluation report to {report_path}")
    return results


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    run_vector_evaluation(m, tok)
