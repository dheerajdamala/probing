"""Unit tests for activation steering hooks and steered generation."""

from __future__ import annotations

import copy
import pytest
import torch

from src.model.load_model import load_model_and_tokenizer
from src.steering.activation_hook import ActivationSteeringHook
from src.steering.steered_generate import generate_with_steering


@pytest.fixture(scope="module")
def mock_model_and_tokenizer():
    """Load fast miniature Mistral model fixture."""
    model, tokenizer, _ = load_model_and_tokenizer(mock=True)
    return model, tokenizer


def test_hook_lifecycle_and_weight_immutability(mock_model_and_tokenizer):
    """Test hook attaches, fires, detaches, and does not alter model weights."""
    model, tokenizer = mock_model_and_tokenizer
    hidden_size = model.config.hidden_size
    v = torch.randn(hidden_size)

    # Snapshot weights before hook
    param_before = next(model.parameters()).clone()

    inputs = tokenizer("def foo():\n", return_tensors="pt")

    # Baseline forward pass
    with torch.no_grad():
        out_baseline = model(**inputs).logits

    # Steered forward pass with context manager
    with ActivationSteeringHook(model, layer_idx=1, steering_vector=v, alpha=2.0) as hook:
        assert hook.hook_handle is not None
        with torch.no_grad():
            out_steered = model(**inputs).logits
        assert hook.interventions_count > 0

    # Verify hook removed
    assert hook.hook_handle is None

    # Forward pass after hook removal should match baseline exactly
    with torch.no_grad():
        out_after = model(**inputs).logits

    assert torch.allclose(out_baseline, out_after, atol=1e-5), "Activations differed after hook removal!"

    # Verify model weights remained untouched
    param_after = next(model.parameters())
    assert torch.equal(param_before, param_after), "Model parameters were modified by steering hook!"


def test_intervention_mathematical_shift(mock_model_and_tokenizer):
    """Test that activation hook shifts residual stream output by alpha * v."""
    model, tokenizer = mock_model_and_tokenizer
    layer_idx = 1
    hidden_size = model.config.hidden_size

    # Controlled steering vector
    v = torch.ones(hidden_size) * 0.5
    alpha = 3.0

    inputs = tokenizer("def test():\n", return_tensors="pt")

    captured_inputs = []
    def monitor_pre_hook(mod, inp):
        # Input to next layer is the steered output of the previous layer
        t = inp[0]
        captured_inputs.append(t.clone())

    # Monitor input arriving at layer 2 (which comes from layer 1)
    handle = model.model.layers[layer_idx + 1].register_forward_pre_hook(monitor_pre_hook)

    # 1. Unhooked forward pass
    with torch.no_grad():
        model(**inputs)

    # 2. Hooked forward pass with steering
    with ActivationSteeringHook(model, layer_idx=layer_idx, steering_vector=v, alpha=alpha):
        with torch.no_grad():
            model(**inputs)

    handle.remove()

    assert len(captured_inputs) == 2, f"Expected 2 passes, got {len(captured_inputs)}"
    unhooked_in = captured_inputs[0]
    hooked_in = captured_inputs[1]

    expected_shift = alpha * v
    actual_shift = (hooked_in - unhooked_in)[0, 0, :]

    assert torch.allclose(actual_shift, expected_shift, atol=1e-4), "Intervention did not match alpha * v!"


def test_steered_generate_execution(mock_model_and_tokenizer):
    """Test generation pipeline executes under unsteered and steered conditions."""
    model, tokenizer = mock_model_and_tokenizer
    hidden_size = model.config.hidden_size
    v = torch.randn(hidden_size)
    prompt = "def calculate_sum(a, b):\n"

    # 1. Unsteered
    res_unsteered = generate_with_steering(
        model, tokenizer, prompt, steering_vector=None, layer_idx=1, alpha=0.0, max_new_tokens=8
    )
    assert not res_unsteered["steered"]
    assert res_unsteered["interventions_count"] == 0
    assert len(res_unsteered["completion"]) > 0

    # 2. Steered with security vector
    res_steered = generate_with_steering(
        model, tokenizer, prompt, steering_vector=v, layer_idx=1, alpha=2.0, max_new_tokens=8
    )
    assert res_steered["steered"]
    assert res_steered["interventions_count"] > 0
    assert isinstance(res_steered["completion"], str)
