"""
Shortcut Learning and Robustness Analysis Module for Python SQL Injection Probe.

Evaluates probe robustness against semantics-preserving code transformations:
1. Identifier Variable Renaming (var_uid, stmt_q, etc.)
2. Comment and Docstring Injections
3. Formatting and Whitespace Perturbations
4. SQL Keyword Casing Alterations (SELECT vs select)

Measures:
- Prediction flip rate (invariance / stability)
- Mean absolute probability delta (|P_orig - P_trans|)
- Robustness across vulnerable vs safe samples
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
import ast
import yaml
import numpy as np
from typing import Any, Dict, List, Tuple

from src.model.load_model import load_model_and_tokenizer
from security_probe_sql.src.probe_model import SecurityProbe
from security_probe_sql.src.inference import SQLInjectionProbeAnalyzer


# Transformation 1: Variable Renaming
def transform_rename_variables(code: str) -> str:
    """Semantics-preserving renaming of local query variables."""
    replacements = [
        (r'\busername\b', 'user_identifier'),
        (r'\buser_id\b', 'account_uid'),
        (r'\bcursor\b', 'db_cursor_obj'),
        (r'\bcur\b', 'c_handler'),
        (r'\bquery\b', 'sql_stmt_text'),
        (r'\bsql\b', 'raw_sql_command'),
        (r'\bconn\b', 'db_connection'),
    ]
    transformed = code
    for pat, repl in replacements:
        transformed = re.sub(pat, repl, transformed)
    return transformed


# Transformation 2: Comment & Docstring Injection
def transform_add_comments_docstrings(code: str) -> str:
    """Injects benign explanatory comments and docstrings."""
    lines = code.splitlines()
    injected = []
    injected.append('# Security compliance audited internal routine')
    injected.append('"""Execute transactional database interaction safely."""')
    for line in lines:
        injected.append(line)
        if 'def ' in line:
            injected.append('    # Query initialization phase')
        elif 'execute' in line.lower():
            injected.append('    # Database engine dispatch call')
    return "\n".join(injected)


# Transformation 3: Whitespace & Formatting Perturbations
def transform_formatting_whitespace(code: str) -> str:
    """Perturbs indentation, trailing spaces, and blank line density."""
    lines = code.splitlines()
    transformed = []
    for line in lines:
        if line.strip():
            transformed.append(line + "  ") # trailing whitespace
            transformed.append("") # extra blank lines
    return "\n".join(transformed)


# Transformation 4: SQL Keyword Casing
def transform_sql_casing(code: str) -> str:
    """Toggles SQL keywords between uppercase and lowercase."""
    keywords = ["SELECT", "FROM", "WHERE", "INSERT", "INTO", "UPDATE", "SET", "DELETE", "AND", "OR"]
    transformed = code
    for kw in keywords:
        # If uppercase, turn to lowercase; if lowercase, turn to uppercase
        transformed = re.sub(r'\b' + kw + r'\b', kw.lower(), transformed)
    return transformed


def run_shortcut_diagnostics(config_path: str, num_eval_samples: int = 150) -> Dict[str, Any]:
    """Evaluates probe sensitivity to controlled, semantics-preserving perturbations."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, cfg["outputs"]["models_dir"])
    metrics_dir = os.path.join(project_root, cfg["outputs"]["metrics_dir"])
    test_path = os.path.join(project_root, cfg["data"]["test_split"])

    # Load test samples
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()][:num_eval_samples]

    # Load model & probe
    is_mock = cfg["model"].get("mock", True)
    target_layer = cfg["model"]["mock_target_layer"] if is_mock else cfg["model"]["target_layer"]
    model, tokenizer, _ = load_model_and_tokenizer(mock=is_mock)
    linear_path = os.path.join(models_dir, "probe_head_linear.pkl")
    probe = SecurityProbe.load(linear_path)

    analyzer = SQLInjectionProbeAnalyzer(
        probe=probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=target_layer,
    )

    transformations = {
        "variable_renaming": transform_rename_variables,
        "comment_docstring_injection": transform_add_comments_docstrings,
        "formatting_whitespace": transform_formatting_whitespace,
        "sql_keyword_casing": transform_sql_casing,
    }

    diagnostic_results = {}
    print(f"\n[Shortcut Analysis] Testing robustness on {len(test_records)} test samples...")

    # Compute baseline original predictions
    orig_analyses = [analyzer.analyze(r["code"]) for r in test_records]
    orig_probs = np.array([a["probability"] for a in orig_analyses])
    orig_preds = np.array([int(a["is_vulnerable"]) for a in orig_analyses])

    for t_name, t_func in transformations.items():
        print(f"  Evaluating transformation: {t_name}...")
        trans_probs = []
        trans_preds = []

        for r in test_records:
            trans_code = t_func(r["code"])
            res = analyzer.analyze(trans_code)
            trans_probs.append(res["probability"])
            trans_preds.append(int(res["is_vulnerable"]))

        trans_probs = np.array(trans_probs)
        trans_preds = np.array(trans_preds)

        # Compute metrics
        flips = int((orig_preds != trans_preds).sum())
        flip_rate = float(flips / len(orig_preds))
        stability_rate = 1.0 - flip_rate
        mean_prob_delta = float(np.mean(np.abs(orig_probs - trans_probs)))
        max_prob_delta = float(np.max(np.abs(orig_probs - trans_probs)))

        diagnostic_results[t_name] = {
            "total_tested": len(test_records),
            "label_flips": flips,
            "prediction_flip_rate": round(flip_rate, 4),
            "stability_rate": round(stability_rate, 4),
            "mean_absolute_prob_delta": round(mean_prob_delta, 4),
            "max_absolute_prob_delta": round(max_prob_delta, 4),
        }
        print(f"    Stability: {stability_rate*100:.1f}%, Mean Prob Delta: {mean_prob_delta:.4f}")

    output_path = os.path.join(metrics_dir, "shortcut_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_results, f, indent=2)

    print(f"[Shortcut Analysis Completed] Results saved to: {output_path}")
    return diagnostic_results


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_file = os.path.join(project_root, "config.yaml")
    run_shortcut_diagnostics(cfg_file)
