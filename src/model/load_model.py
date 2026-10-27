"""Model and tokenizer loader with hardware awareness and precision control.

Supports real Hugging Face Mistral models as well as lightweight
mock Mistral instances for deterministic testing and dry-runs.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_optimal_device(requested: str = "auto") -> torch.device:
    """Resolve optimal hardware device."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested == "cpu":
        return torch.device("cpu")
    elif requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def get_model_memory_usage(model: PreTrainedModel) -> Dict[str, Any]:
    """Measure model parameter count and memory footprint."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size_bytes + buffer_size_bytes) / (1024 * 1024)

    return {
        "num_parameters": num_params,
        "num_trainable": num_trainable,
        "total_size_mb": round(total_size_mb, 2),
        "total_size_gb": round(total_size_mb / 1024, 3),
        "dtype": str(next(model.parameters()).dtype),
    }


def create_miniature_mistral(
    num_layers: int = 4,
    hidden_size: int = 128,
    intermediate_size: int = 256,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    vocab_size: int = 32000,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Create a miniature, fast in-memory Mistral model for unit tests and CI."""
    tokenizer = AutoTokenizer.from_pretrained("wandb/pruned_mistral")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MistralConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or 1,
        eos_token_id=tokenizer.eos_token_id or 2,
    )
    model = MistralForCausalLM(config)
    model.eval()

    return model, tokenizer


def load_model_and_tokenizer(
    model_id: str = "wandb/pruned_mistral",
    device: str = "auto",
    torch_dtype: Optional[str] = None,
    mock: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Dict[str, Any]]:
    """Load model and tokenizer with memory profiling."""
    start_time = time.time()

    if mock:
        logger.info("Initializing synthetic miniature Mistral model for testing/dry-run...")
        model, tokenizer = create_miniature_mistral()
        device_obj = torch.device("cpu")
        model.to(device_obj)
        load_time = time.time() - start_time
        meta = get_model_memory_usage(model)
        meta["load_time_sec"] = round(load_time, 3)
        meta["device"] = str(device_obj)
        meta["mock"] = True
        return model, tokenizer, meta

    device_obj = get_optimal_device(device)
    logger.info(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve dtype
    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if device_obj.type == "cuda" else torch.float32

    logger.info(f"Loading model '{model_id}' onto {device_obj} with dtype {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device_obj)
    model.eval()

    load_time = time.time() - start_time
    meta = get_model_memory_usage(model)
    meta["load_time_sec"] = round(load_time, 3)
    meta["device"] = str(device_obj)
    meta["mock"] = False

    logger.info(
        f"Model loaded successfully in {meta['load_time_sec']}s: "
        f"{meta['num_parameters']:,} parameters ({meta['total_size_mb']} MB) on {meta['device']}"
    )

    return model, tokenizer, meta


if __name__ == "__main__":
    # Test miniature model instantiation
    m, tok, info = load_model_and_tokenizer(mock=True)
    print("Mock Model Meta:", info)
