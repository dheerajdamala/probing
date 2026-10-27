"""Unit tests for security vector extraction and probe training."""

from __future__ import annotations

import pytest
import torch

from src.model.load_model import load_model_and_tokenizer
from src.probes.train_semantic_probe import train_linear_probe, evaluate_probe_model
from src.steering.extract_security_vectors import compute_security_vector_from_pairs


@pytest.fixture(scope="module")
def mock_model_and_tok():
    model, tokenizer, _ = load_model_and_tokenizer(mock=True)
    return model, tokenizer


def test_vector_extraction(mock_model_and_tok):
    """Test contrastive security vector extraction and normalization."""
    model, tokenizer = mock_model_and_tok
    dummy_pairs = [
        {
            "id": "pair_01",
            "prompt": "def query_db(uid):\n",
            "safe_completion": "    return cursor.execute('SELECT * FROM u WHERE id=%s', (uid,))\n",
            "vulnerable_completion": "    return cursor.execute(f'SELECT * FROM u WHERE id={uid}')\n",
        }
    ]

    res = compute_security_vector_from_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=dummy_pairs,
        target_layers=[0, 1],
        method="mean_diff",
        normalize=True,
    )

    sec_vectors = res["security_vectors"]
    rand_vectors = res["random_vectors"]

    assert 0 in sec_vectors and 1 in sec_vectors
    assert 0 in rand_vectors and 1 in rand_vectors

    # Verify unit norm
    norm_sec = torch.norm(sec_vectors[0], p=2).item()
    norm_rand = torch.norm(rand_vectors[0], p=2).item()
    assert abs(norm_sec - 1.0) < 1e-4
    assert abs(norm_rand - 1.0) < 1e-4


def test_probe_training_and_evaluation():
    """Test linear probe training on synthetic feature matrix."""
    import numpy as np

    N = 20
    d = 16
    np.random.seed(42)
    # Class 0 centered at -1, Class 1 centered at +1
    X0 = np.random.randn(N // 2, d) - 1.0
    X1 = np.random.randn(N // 2, d) + 1.0
    X = np.vstack([X0, X1])
    y = np.array([0] * (N // 2) + [1] * (N // 2))

    probe = train_linear_probe(X, y, C=1.0)
    metrics = evaluate_probe_model(probe, X, y)

    assert metrics["accuracy"] >= 0.90
    assert metrics["f1"] >= 0.90
    assert metrics["roc_auc"] >= 0.90
