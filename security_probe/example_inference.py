"""Minimal runnable single-sample inference example for SecurityProbe.

Demonstrates analyzing a single code snippet without requiring a paired example,
contrastive baseline, or steering interventions.

Usage:
    python security_probe/example_inference.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.model.load_model import load_model_and_tokenizer
from security_probe.src.inference import SecurityProbeAnalyzer
from security_probe.src.probe_model import SecurityProbe


def run_example():
    print("=" * 70)
    print("Security Probe: Single-Sample Vulnerability Analysis Example")
    print("=" * 70)

    # 1. Load model and tokenizer (using fast miniature fixture for local demonstration)
    print("\n1. Loading base model (frozen feature extractor)...")
    model, tokenizer, info = load_model_and_tokenizer(mock=True)
    print(f"   Loaded model: {info.get('num_parameters', 0):,} params on {info.get('device')}")

    # 2. Initialize a trained/fitted probe
    # (If a saved probe exists on disk, load it; otherwise initialize a calibrated probe)
    probe_path = Path("security_probe/models/probe_model.pkl")
    if probe_path.exists():
        print(f"\n2. Loading serialized probe from {probe_path}...")
        probe = SecurityProbe.load(probe_path)
    else:
        print("\n2. Initializing demonstration probe...")
        import numpy as np
        hidden_dim = model.config.hidden_size
        np.random.seed(42)
        # Synthetic calibration fit
        X = np.vstack([np.random.randn(10, hidden_dim) - 0.5, np.random.randn(10, hidden_dim) + 0.5])
        y = np.array([0] * 10 + [1] * 10, dtype=np.int32)
        probe = SecurityProbe(
            architecture="linear",
            regularization_c=1.0,
            decision_threshold=0.50,
            uncertainty_margin=0.08,
            selected_layer=2,
        )
        probe.fit(X, y, cwe_labels=["CWE-89"] * 10 + ["CWE-119"] * 10)

    # 3. Create Analyzer
    analyzer = SecurityProbeAnalyzer(
        probe=probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=2,
        finding_threshold=0.55,
    )

    # 4. Input: A SINGLE code sample (no paired safe/vulnerable version)
    sample_code = (
        "def get_user_records(cursor, user_id: str):\n"
        "    # Construct and execute database query\n"
        "    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n"
        "    cursor.execute(query)\n"
        "    return cursor.fetchall()\n"
    )

    print("\n3. Analyzing single input code sample:")
    print("-" * 50)
    print(sample_code.strip())
    print("-" * 50)

    # 5. Execute inference
    result = analyzer.analyze(sample_code)

    # 6. Display structured results
    print("\n4. Structured Analysis Result:")
    print(f"   Status:                 {result['status']}")
    print(f"   Is Vulnerable:          {result['is_vulnerable']}")
    print(f"   Calibrated Probability: {result['probability']}")
    print(f"   Confidence Score:       {result['confidence']}")
    print(f"   Predicted CWE:          {result['predicted_cwe']}")
    print(f"   No Finding Flag:        {result['no_finding']}")

    print("\n5. Line-Level Exploratory Scores:")
    for l_num, score in result["line_scores"].items():
        bar = "#" * int(score * 20)
        print(f"   Line {l_num:2d} [{score:.3f}] | {bar:<20}")

    if result["findings"]:
        print("\n6. Suspected Vulnerable Findings:")
        for idx, finding in enumerate(result["findings"], 1):
            print(f"   Finding #{idx}: Line {finding['line_number']} (Score: {finding['score']:.3f})")
            print(f"     Content: {finding['line_content'].strip()}")
    else:
        print("\n6. Suspected Vulnerable Findings: None (all below finding threshold)")

    print("\n" + "=" * 70)
    print("Localization Notice: Dataset has whole-program labels only.")
    print("Line scores represent exploratory representation projections, not ground-truth spans.")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
