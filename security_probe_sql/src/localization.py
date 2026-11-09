"""
Exploratory Token-Level Projection and Suspicious Span Localization Module.

Projects frozen per-token hidden states onto the trained probe decision vector
to produce token-level vulnerability scores and aggregate them into line-level
and span-level candidate findings.

CRITICAL RESEARCH NOTICE:
Because the probe is trained on whole-program labels, token-level scores represent
post-hoc linear exploratory projections onto the representation decision boundary,
NOT ground-truth supervised localization, unless verified against line annotations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def char_to_line_col(code: str, char_idx: int) -> Tuple[int, int]:
    """Convert 0-based character index to 1-based (line_num, col_num)."""
    if char_idx <= 0:
        return (1, 1)
    if char_idx > len(code):
        char_idx = len(code)

    prefix = code[:char_idx]
    lines = prefix.split("\n")
    line_num = len(lines)
    col_num = len(lines[-1]) + 1
    return (line_num, col_num)


def compute_token_scores(
    token_hidden_states: torch.Tensor,
    weight_vector: np.ndarray,
    bias_scalar: float,
    scaler: StandardScaler,
) -> np.ndarray:
    """Project per-token hidden states onto probe decision vector.
    
    Args:
        token_hidden_states: Tensor of shape (seq_len, hidden_dim).
        weight_vector: 1D array of shape (hidden_dim,).
        bias_scalar: Float bias term.
        scaler: StandardScaler fitted during probe training.
        
    Returns:
        1D numpy array of probabilities s_t in [0, 1] for each token.
    """
    H = token_hidden_states.cpu().numpy() # (seq_len, hidden_dim)
    # Standardize token states using training feature statistics
    H_scaled = scaler.transform(H)
    # Linear projection + sigmoid activation
    logits = H_scaled @ weight_vector + bias_scalar
    # Numerically stable sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    return probs


def aggregate_line_scores(
    code: str,
    token_strings: List[str],
    token_offsets: List[Tuple[int, int]],
    token_scores: np.ndarray,
    aggregation: str = "max",
) -> List[Dict[str, Any]]:
    """Aggregate token scores to source code lines.
    
    Args:
        code: Full input source code.
        token_strings: List of token strings.
        token_offsets: List of (start_char, end_char).
        token_scores: 1D array of token probabilities.
        aggregation: "max" or "mean".
        
    Returns:
        List of dicts for each line with line_number, line_content, score, and token details.
    """
    code_lines = code.split("\n")
    line_data: List[Dict[str, Any]] = [
        {"line_number": idx + 1, "line_content": line, "token_scores": []}
        for idx, line in enumerate(code_lines)
    ]

    for tok_str, (start_char, end_char), score in zip(token_strings, token_offsets, token_scores):
        if not tok_str.strip():
            continue
        start_line, _ = char_to_line_col(code, start_char)
        if 1 <= start_line <= len(line_data):
            line_data[start_line - 1]["token_scores"].append(float(score))

    # Compute aggregate score per line
    for item in line_data:
        scores = item["token_scores"]
        if scores:
            if aggregation == "max":
                item["score"] = float(np.max(scores))
            elif aggregation == "mean":
                item["score"] = float(np.mean(scores))
            else:
                item["score"] = float(np.max(scores))
        else:
            item["score"] = 0.0

    return line_data


def extract_candidate_findings(
    code: str,
    line_data: List[Dict[str, Any]],
    token_strings: List[str],
    token_offsets: List[Tuple[int, int]],
    token_scores: np.ndarray,
    score_threshold: float = 0.45,
    max_findings: int = 5,
    ground_truth_spans_available: bool = False,
) -> List[Dict[str, Any]]:
    """Rank and extract top suspicious code regions based on exploratory token projections."""
    sorted_lines = sorted(
        [l for l in line_data if l["score"] >= score_threshold and l["line_content"].strip()],
        key=lambda x: x["score"],
        reverse=True,
    )

    findings: List[Dict[str, Any]] = []
    seen_lines = set()

    for l_info in sorted_lines[:max_findings]:
        l_num = l_info["line_number"]
        if l_num in seen_lines:
            continue
        seen_lines.add(l_num)

        # Find character span for this line
        line_content = l_info["line_content"]
        start_col = 1
        end_col = len(line_content) + 1

        findings.append({
            "start_line": l_num,
            "end_line": l_num,
            "start_column": start_col,
            "end_column": end_col,
            "score": round(float(l_info["score"]), 4),
            "line_snippet": line_content.strip(),
            "localization_mode": "exploratory_token_projection",
            "ground_truth_spans_available": ground_truth_spans_available,
        })

    return findings
