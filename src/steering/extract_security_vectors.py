"""Security vector extraction from contrastive safe vs. vulnerable representations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model.extract_activations import extract_layer_hidden_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_security_vector_from_pairs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: List[Dict[str, Any]],
    target_layers: List[int],
    method: str = "mean_diff",
    pooling: str = "last_token",
    normalize: bool = True,
    random_seed: int = 42,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Extract candidate security direction vectors across target layers.

    Returns:
        Dict containing:
          - "security_vectors": {layer_idx: Tensor(hidden_dim)}
          - "random_vectors": {layer_idx: Tensor(hidden_dim)}
    """
    logger.info(f"Extracting security vectors from {len(pairs)} pairs across layers {target_layers} using {method}...")

    # Collect differences per layer: layer_idx -> List[Tensor]
    diffs_per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}

    for idx, item in enumerate(pairs):
        prompt = item["prompt"]
        safe_code = prompt + item["safe_completion"]
        vuln_code = prompt + item["vulnerable_completion"]

        safe_acts = extract_layer_hidden_states(model, tokenizer, safe_code, target_layers=target_layers, pooling=pooling)
        vuln_acts = extract_layer_hidden_states(model, tokenizer, vuln_code, target_layers=target_layers, pooling=pooling)

        for l in target_layers:
            diff = safe_acts[l] - vuln_acts[l]
            diffs_per_layer[l].append(diff)

    security_vectors: Dict[int, torch.Tensor] = {}
    random_vectors: Dict[int, torch.Tensor] = {}

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    for l in target_layers:
        diff_stack = torch.stack(diffs_per_layer[l], dim=0) # shape: (N, hidden_dim)

        if method == "mean_diff":
            vec = diff_stack.mean(dim=0)
        elif method == "pca_diff":
            diff_np = diff_stack.numpy()
            diff_centered = diff_np - diff_np.mean(axis=0, keepdims=True)
            u, s, vt = np.linalg.svd(diff_centered, full_matrices=False)
            vec = torch.tensor(vt[0], dtype=torch.float32)
            # Orient towards safe by checking sign of dot product with mean
            mean_vec = diff_stack.mean(dim=0)
            if torch.dot(vec, mean_vec) < 0:
                vec = -vec
        else:
            raise ValueError(f"Unknown vector extraction method: {method}")

        if normalize:
            norm = torch.norm(vec, p=2)
            vec = vec / (norm + 1e-9)

        security_vectors[l] = vec

        # Control vector: random Gaussian vector of identical dimension and norm
        rand_vec = torch.randn_like(vec)
        if normalize:
            rand_vec = rand_vec / (torch.norm(rand_vec, p=2) + 1e-9)
        random_vectors[l] = rand_vec

    logger.info(f"Successfully computed security and random vectors for {len(target_layers)} layers.")
    return {
        "security_vectors": security_vectors,
        "random_vectors": random_vectors,
    }


def extract_and_save_vectors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_file: Path | str = "data/processed/train.json",
    output_path: Path | str = "outputs/security_vectors.pt",
    target_layers: Optional[List[int]] = None,
    method: str = "mean_diff",
    pooling: str = "last_token",
    normalize: bool = True,
) -> Path:
    """Run full extraction pipeline and save vectors to disk."""
    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("paired", [])

    if target_layers is None:
        target_layers = list(range(model.config.num_hidden_layers))

    vectors_bundle = compute_security_vector_from_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        target_layers=target_layers,
        method=method,
        pooling=pooling,
        normalize=normalize,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors_bundle, out_file)
    logger.info(f"Saved security vector bundle to {out_file}")
    return out_file


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    extract_and_save_vectors(m, tok, target_layers=[0, 1, 2, 3])
