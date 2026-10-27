"""Steered code generation pipeline.

Executes generation under activation intervention, logs parameters,
measures latency, and verifies output stability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model.generate import generate_code_completion
from src.steering.activation_hook import ActivationSteeringHook

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_with_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    steering_vector: Optional[torch.Tensor] = None,
    layer_idx: int = 2,
    alpha: float = 1.0,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    do_sample: bool = False,
    condition_name: str = "steered",
) -> Dict[str, Any]:
    """Generate completion under fixed activation steering intervention.

    If steering_vector is None or alpha == 0.0, executes unsteered baseline.
    """
    vector_norm = float(torch.norm(steering_vector, p=2).item()) if steering_vector is not None else 0.0

    if steering_vector is None or alpha == 0.0:
        # Unsteered generation
        gen_result = generate_code_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        gen_result.update({
            "condition": condition_name,
            "steered": False,
            "layer_idx": layer_idx,
            "alpha": 0.0,
            "vector_norm": vector_norm,
            "interventions_count": 0,
        })
        return gen_result

    # Attach hook and generate
    with ActivationSteeringHook(
        model=model,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        alpha=alpha,
    ) as hook:
        gen_result = generate_code_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    # Health checks on steered completion
    completion = gen_result["completion"]
    is_malformed = len(completion.strip()) == 0 or len(set(completion.split())) < 2

    gen_result.update({
        "condition": condition_name,
        "steered": True,
        "layer_idx": layer_idx,
        "alpha": alpha,
        "vector_norm": vector_norm,
        "interventions_count": hook.interventions_count,
        "is_malformed": is_malformed,
    })

    return gen_result


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    v = torch.randn(m.config.hidden_size)
    res = generate_with_steering(m, tok, "def add(a, b):\n", steering_vector=v, layer_idx=1, alpha=1.5, max_new_tokens=16)
    print("Steered Gen Result:", res["completion"], f"(interventions: {res['interventions_count']})")
