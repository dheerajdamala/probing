"""
Feature Extraction Module for Python SQL Injection Security Probe.

Extracts pooled representations and token-level hidden states from frozen
Transformer language models without modifying model parameters or introducing
steering interventions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def resolve_layer_index(model: Any, requested_layer: int) -> int:
    """Validate layer index against model depth and return valid index."""
    num_layers = None
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_layers = len(model.transformer.h)

    if num_layers is None:
        return requested_layer

    if requested_layer < 0 or requested_layer >= num_layers:
        fallback = min(max(0, requested_layer), num_layers - 1)
        logger.warning(
            f"Requested layer {requested_layer} out of bounds for model with {num_layers} layers. "
            f"Falling back to layer {fallback}."
        )
        return fallback

    return requested_layer


def compute_token_offsets(
    tokenizer: PreTrainedTokenizerBase,
    code: str,
    input_ids: torch.Tensor,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Compute character span offsets (start_char, end_char) for each token."""
    token_ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
    token_strings = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]

    # Fast tokenizer offset mapping attempt
    try:
        encoded = tokenizer(code, return_offsets_mapping=True, add_special_tokens=True)
        if "offset_mapping" in encoded and encoded["offset_mapping"] is not None:
            raw_offsets = encoded["offset_mapping"]
            if len(raw_offsets) == len(token_ids):
                return token_strings, raw_offsets
    except Exception:
        pass

    # Robust character alignment fallback
    offsets: List[Tuple[int, int]] = []
    current_idx = 0
    code_len = len(code)

    for tok_str in token_strings:
        cleaned_tok = tok_str.lstrip(" ").replace("<s>", "").replace("</s>", "")
        if not cleaned_tok:
            offsets.append((current_idx, current_idx))
            continue

        found_idx = code.find(cleaned_tok, current_idx)
        if found_idx != -1:
            start = found_idx
            end = found_idx + len(cleaned_tok)
            offsets.append((start, end))
            current_idx = end
        else:
            offsets.append((current_idx, min(current_idx + len(cleaned_tok), code_len)))
            current_idx = min(current_idx + len(cleaned_tok), code_len)

    return token_strings, offsets


def extract_code_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    code: str,
    layer_idx: int = 16,
    pooling: str = "mean_prompt",
    device: Optional[torch.device] = None,
    compute_offsets: bool = False,
) -> Dict[str, Any]:
    """Extract both pooled code-level representation and per-token hidden states.
    
    Args:
        model: Frozen pretrained language model.
        tokenizer: Tokenizer.
        code: Input Python source code string.
        layer_idx: Target hidden state layer index.
        pooling: "mean_prompt" or "last_token".
        device: Torch device.
        compute_offsets: Whether to compute character offsets (slower, needed for token localization).
        
    Returns:
        Dict containing:
            - "pooled_rep": 1D numpy array of shape (hidden_dim,)
            - "token_hidden_states": 2D torch.Tensor of shape (seq_len, hidden_dim) on CPU
            - "token_strings": List of decoded token strings (or empty list)
            - "token_offsets": List of (start_char, end_char) tuples (or empty list)
            - "layer_idx": Validated layer index
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    actual_layer = resolve_layer_index(model, layer_idx)

    # Empty or whitespace handling
    if not code or not code.strip():
        inputs = tokenizer("<empty>", return_tensors="pt")
    else:
        inputs = tokenizer(code, return_tensors="pt", max_length=2048, truncation=True)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Strictly read-only forward pass: No gradients, no steering interventions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    all_hidden = outputs.hidden_states
    num_layers = len(all_hidden) - 1
    safe_layer = min(actual_layer, num_layers - 1)

    layer_hidden = all_hidden[safe_layer + 1] # +1 because 0 is embeddings
    token_hidden = layer_hidden.squeeze(0).cpu().to(torch.float32)

    # Compute pooled representation
    if pooling == "last_token":
        pooled = token_hidden[-1].numpy()
    elif pooling == "mean_prompt":
        if attention_mask is not None:
            mask = attention_mask.squeeze(0).cpu().unsqueeze(-1)
            sum_rep = (token_hidden * mask).sum(dim=0)
            count = torch.clamp(mask.sum(), min=1.0)
            pooled = (sum_rep / count).numpy()
        else:
            pooled = token_hidden.mean(dim=0).numpy()
    else:
        pooled = token_hidden.mean(dim=0).numpy()

    if compute_offsets:
        token_strings, token_offsets = compute_token_offsets(tokenizer, code, input_ids.cpu())
    else:
        token_strings, token_offsets = [], []

    return {
        "pooled_rep": pooled,
        "token_hidden_states": token_hidden,
        "token_strings": token_strings,
        "token_offsets": token_offsets,
        "layer_idx": safe_layer,
    }
