"""Unit tests for single-sample security probe inference."""

from __future__ import annotations

import copy
import pytest
import torch
import numpy as np

from src.model.load_model import load_model_and_tokenizer
from security_probe.src.feature_extractor import extract_code_representations
from security_probe.src.probe_model import SecurityProbe
from security_probe.src.inference import SecurityProbeAnalyzer


@pytest.fixture(scope="module")
def mock_model_and_tokenizer():
    """Load cached fast miniature Mistral model fixture."""
    model, tokenizer, _ = load_model_and_tokenizer(mock=True)
    return model, tokenizer


@pytest.fixture(scope="module")
def fitted_toy_probe(mock_model_and_tokenizer):
    """Create and fit a lightweight toy probe on miniature representations."""
    model, tokenizer = mock_model_and_tokenizer
    hidden_dim = model.config.hidden_size # 128

    # Create synthetic training set
    np.random.seed(42)
    X_safe = np.random.randn(10, hidden_dim).astype(np.float32) - 0.5
    X_vuln = np.random.randn(10, hidden_dim).astype(np.float32) + 0.5
    X = np.vstack([X_safe, X_vuln])
    y = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    cwes = ["CWE-89"] * 5 + ["CWE-119"] * 5 + ["CWE-89"] * 5 + ["CWE-119"] * 5

    probe = SecurityProbe(
        architecture="linear",
        regularization_c=1.0,
        random_state=42,
        decision_threshold=0.50,
        uncertainty_margin=0.08,
        selected_layer=2,
    )
    probe.fit(X, y, cwe_labels=cwes)
    return probe


def test_single_sample_inference_no_pairing(mock_model_and_tokenizer, fitted_toy_probe):
    """Test 1: A single code sample can be analyzed without any paired example or reference."""
    model, tokenizer = mock_model_and_tokenizer
    analyzer = SecurityProbeAnalyzer(
        probe=fitted_toy_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )

    code = "def query_db(uid):\n    return db.execute('SELECT * FROM users WHERE id = %s', (uid,))\n"
    result = analyzer.analyze(code)

    assert isinstance(result, dict)
    assert "status" in result
    assert "is_vulnerable" in result
    assert "probability" in result
    assert "confidence" in result
    assert "findings" in result
    assert "metadata" in result
    assert result["metadata"]["localization_mode"] == "exploratory_token_projection"
    # Verify no second sample was required or mentioned
    assert "pair" not in result
    assert "contrastive" not in result["metadata"]


def test_vulnerable_and_benign_both_processed(mock_model_and_tokenizer, fitted_toy_probe):
    """Test 2: Both vulnerable and benign code samples can be processed independently."""
    model, tokenizer = mock_model_and_tokenizer
    analyzer = SecurityProbeAnalyzer(
        probe=fitted_toy_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )

    sample_benign = "def add(a, b):\n    return a + b\n"
    sample_vuln = "def run_cmd(user_arg):\n    import os\n    os.system('cat ' + user_arg)\n"

    res_b = analyzer.analyze(sample_benign)
    res_v = analyzer.analyze(sample_vuln)

    assert res_b["status"] in ["BENIGN", "NO_FINDING", "UNCERTAIN", "VULNERABLE"]
    assert res_v["status"] in ["BENIGN", "NO_FINDING", "UNCERTAIN", "VULNERABLE"]
    assert 0.0 <= res_b["probability"] <= 1.0
    assert 0.0 <= res_v["probability"] <= 1.0


def test_empty_and_malformed_input_safety(mock_model_and_tokenizer, fitted_toy_probe):
    """Test 4: Empty and syntactically malformed input are handled safely without crashing."""
    model, tokenizer = mock_model_and_tokenizer
    analyzer = SecurityProbeAnalyzer(
        probe=fitted_toy_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )

    # Empty string
    res_empty = analyzer.analyze("")
    assert res_empty["status"] == "NO_FINDING"
    assert res_empty["is_vulnerable"] is False
    assert res_empty["no_finding"] is True
    assert res_empty["findings"] == []

    # Whitespace only
    res_ws = analyzer.analyze("   \n\t  \n")
    assert res_ws["status"] == "NO_FINDING"
    assert res_ws["is_vulnerable"] is False

    # Malformed Python code (syntax error)
    malformed = "def func_broken(:\n   for while True return %%%"
    res_malformed = analyzer.analyze(malformed)
    assert res_malformed["status"] in ["BENIGN", "VULNERABLE", "UNCERTAIN", "NO_FINDING"]
    assert "probability" in res_malformed


def test_base_model_weights_remain_unchanged(mock_model_and_tokenizer, fitted_toy_probe):
    """Test 5: The base LLM's weights remain frozen and unchanged after probe inference."""
    model, tokenizer = mock_model_and_tokenizer
    param_before = next(model.parameters()).clone()

    analyzer = SecurityProbeAnalyzer(
        probe=fitted_toy_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )

    code = "def sample():\n    return 42\n"
    _ = analyzer.analyze(code)

    param_after = next(model.parameters())
    assert torch.equal(param_before, param_after), "Base model weights were modified during probe analysis!"


def test_probe_does_not_invoke_steering_hooks(mock_model_and_tokenizer, fitted_toy_probe):
    """Test 6: The probe does not register or invoke existing activation steering hooks."""
    model, tokenizer = mock_model_and_tokenizer
    analyzer = SecurityProbeAnalyzer(
        probe=fitted_toy_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
    )

    # Count registered forward hooks on all model modules
    initial_hooks = sum(len(m._forward_hooks) for m in model.modules())

    code = "def test_fn(x):\n    return x * 2\n"
    _ = analyzer.analyze(code)

    final_hooks = sum(len(m._forward_hooks) for m in model.modules())
    assert final_hooks == initial_hooks, "Forward hooks were modified on the model by the probe!"

    # Verify no ActivationSteeringHook was invoked or attached
    for m in model.modules():
        for h in m._forward_hooks.values():
            hook_target = getattr(h, "__self__", None)
            assert type(hook_target).__name__ != "ActivationSteeringHook", "ActivationSteeringHook found on model!"


def test_uncertain_output_schema_validity(mock_model_and_tokenizer, fitted_toy_probe):
    """Test 7: The output schema is valid even when the result falls into the uncertain band."""
    model, tokenizer = mock_model_and_tokenizer
    # Set wide uncertainty margin to force UNCERTAIN status
    analyzer = SecurityProbeAnalyzer(
        probe=fitted_toy_probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
        decision_threshold=0.50,
        uncertainty_margin=0.49, # Any prob in [0.01, 0.99] becomes UNCERTAIN
    )

    code = "def check_val(x):\n    return x > 0\n"
    result = analyzer.analyze(code)

    assert result["status"] == "UNCERTAIN"
    assert result["is_vulnerable"] is None
    assert "probability" in result
    assert "confidence" in result
    assert "findings" in result
    assert "line_scores" in result
    assert isinstance(result["metadata"], dict)
