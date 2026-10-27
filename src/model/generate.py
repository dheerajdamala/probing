"""Code generation module with latency profiling and deterministic decoding."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_code_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Generate completion for a code prompt with timing diagnostics."""
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = input_ids.shape[1]

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample and temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
    elapsed_sec = time.perf_counter() - start_time

    generated_tokens = output_ids[0, prompt_len:]
    num_generated = len(generated_tokens)
    completion_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    tokens_per_sec = (num_generated / elapsed_sec) if elapsed_sec > 0 else 0.0

    return {
        "prompt": prompt,
        "completion": completion_text,
        "full_text": full_text,
        "prompt_tokens": prompt_len,
        "generated_tokens": num_generated,
        "elapsed_sec": round(elapsed_sec, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Sequential or batch generation for multiple prompts."""
    results = []
    for p in prompts:
        res = generate_code_completion(model, tokenizer, p, **kwargs)
        results.append(res)
    return results


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    res = generate_code_completion(m, tok, "def hello_world():\n", max_new_tokens=16)
    print("Generated result:", res["completion"], f"({res['tokens_per_sec']} t/s)")
