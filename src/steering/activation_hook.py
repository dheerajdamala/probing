"""Activation steering hook implementation for Transformer residual streams.

Provides clean forward-hook intervention and context management:
h_modified = h_original + alpha * security_vector
Safely attaches and detaches hooks without modifying pre-trained model weights.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ActivationSteeringHook:
    """Context-managed forward hook for static activation steering.

    Injects:
        h_modified = h_original + alpha * steering_vector
    at the specified Transformer decoder layer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        steering_vector: torch.Tensor,
        alpha: float = 1.0,
        normalize_intervention: bool = False,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.alpha = float(alpha)
        self.normalize_intervention = normalize_intervention
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # Verify layer index existence
        self.layer_module = self._resolve_layer_module(model, layer_idx)

        # Ensure steering vector is on matching device and dtype
        device = next(self.layer_module.parameters()).device
        dtype = next(self.layer_module.parameters()).dtype
        if dtype in (torch.uint8, torch.int8):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.steering_vector = steering_vector.to(device=device, dtype=dtype)

        # Logging tracking
        self.interventions_count = 0
        self.last_input_shape = None
        self.last_output_shape = None

    def _resolve_layer_module(self, model: PreTrainedModel, layer_idx: int) -> nn.Module:
        """Resolve layer module across standard causal architectures (Mistral/Llama/GPT)."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError(f"Unsupported model architecture for layer resolution: {type(model)}")

        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer {layer_idx} out of range [0, {len(layers) - 1}]")

        return layers[layer_idx]

    def _hook_fn(
        self,
        module: nn.Module,
        inputs: Tuple[Any, ...],
        output: Union[torch.Tensor, Tuple[Any, ...]],
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        """Hook callback modifying intermediate residual states."""
        self.interventions_count += 1

        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        self.last_input_shape = hidden_states.shape

        # Calculate perturbation: alpha * v (broadcasted across batch and sequence)
        perturbation = (self.alpha * self.steering_vector).to(dtype=hidden_states.dtype, device=hidden_states.device)

        # Check for numerical stability
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            logger.warning(f"NaN/Inf detected in layer {self.layer_idx} hidden states prior to steering.")

        modified_hidden = hidden_states + perturbation

        self.last_output_shape = modified_hidden.shape

        if is_tuple:
            return (modified_hidden,) + output[1:]
        return modified_hidden

    def register(self) -> ActivationSteeringHook:
        """Attach hook to model."""
        if self.hook_handle is None:
            self.hook_handle = self.layer_module.register_forward_hook(self._hook_fn)
            logger.debug(f"Attached steering hook at layer {self.layer_idx} (alpha={self.alpha})")
        return self

    def remove(self) -> None:
        """Remove hook safely."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.debug(f"Removed steering hook from layer {self.layer_idx}")

    def __enter__(self) -> ActivationSteeringHook:
        return self.register()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    v = torch.randn(m.config.hidden_size)
    with ActivationSteeringHook(m, layer_idx=1, steering_vector=v, alpha=2.0) as hook:
        inputs = tok("def test():", return_tensors="pt")
        out = m(**inputs)
        print(f"Hook successfully fired {hook.interventions_count} times! Shape: {hook.last_output_shape}")
