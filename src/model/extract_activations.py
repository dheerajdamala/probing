"""Hidden-state extraction and activation caching module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_layer_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    target_layers: Optional[List[int]] = None,
    pooling: str = "last_token",
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Forward-pass hidden-state extractor returning pooled layer representations.

    Args:
        model: Pre-trained causal LM (e.g. Mistral).
        tokenizer: Model tokenizer.
        text: Prompt or complete code sequence.
        target_layers: List of layer indices to extract. If None, extracts all layers.
        pooling: "last_token", "mean_prompt", or "none".
        device: Computation device.

    Returns:
        Dict mapping layer_idx -> extracted tensor on CPU.
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # outputs.hidden_states is a tuple of (embedding_output, layer_0, layer_1, ..., layer_N)
    all_hidden = outputs.hidden_states
    num_layers = len(all_hidden) - 1

    if target_layers is None:
        target_layers = list(range(num_layers))

    extracted: Dict[int, torch.Tensor] = {}

    for l_idx in target_layers:
        if l_idx < 0 or l_idx >= num_layers:
            raise ValueError(f"Layer index {l_idx} is out of bounds for model with {num_layers} layers.")

        # hidden_states[l_idx + 1] corresponds to output of layer l_idx
        layer_rep = all_hidden[l_idx + 1] # shape: (batch_size, seq_len, hidden_dim)

        if pooling == "last_token":
            # Select vector at the last valid token
            vec = layer_rep[:, -1, :].squeeze(0).cpu().to(torch.float32)
        elif pooling == "mean_prompt":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(layer_rep)
                sum_rep = torch.sum(layer_rep * mask_expanded, dim=1)
                lens = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1.0)
                vec = (sum_rep / lens).squeeze(0).cpu().to(torch.float32)
            else:
                vec = layer_rep.mean(dim=1).squeeze(0).cpu().to(torch.float32)
        elif pooling == "none":
            vec = layer_rep.squeeze(0).cpu().to(torch.float32)
        else:
            raise ValueError(f"Unknown pooling strategy: '{pooling}'")

        extracted[l_idx] = vec

    return extracted


def cache_activations(
    activations: Dict[str, Dict[int, torch.Tensor]],
    output_path: Path | str,
) -> Path:
    """Save dictionary of activations to disk."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Flatten structure for torch.save: key format "{sample_id}_layer_{l_idx}"
    flat_tensors: Dict[str, torch.Tensor] = {}
    for sample_id, layer_dict in activations.items():
        for l_idx, tensor in layer_dict.items():
            flat_tensors[f"{sample_id}_layer_{l_idx}"] = tensor

    torch.save(flat_tensors, out_file)
    logger.info(f"Cached {len(flat_tensors)} activation tensors to {out_file}")
    return out_file


def load_cached_activations(cache_path: Path | str) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load cached activations back into nested dictionary."""
    in_file = Path(cache_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Activation cache not found at: {in_file}")

    flat_tensors: Dict[str, torch.Tensor] = torch.load(in_file, weights_only=True)
    nested: Dict[str, Dict[int, torch.Tensor]] = {}

    for k, v in flat_tensors.items():
        # format: "{sample_id}_layer_{l_idx}"
        parts = k.rsplit("_layer_", 1)
        if len(parts) == 2:
            s_id, l_str = parts[0], parts[1]
            l_idx = int(l_str)
            nested.setdefault(s_id, {})[l_idx] = v

    return nested


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    res = extract_layer_hidden_states(m, tok, "def test():\n    pass\n", target_layers=[0, 1, 2, 3])
    for l, t in res.items():
        print(f"Layer {l} representation shape: {t.shape}")
