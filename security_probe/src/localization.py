"""Localization module for mapping token representations to source lines and spans.

Aggregates token-level probe projections into line-level findings and character/column spans.
Provides transparent documentation that token-level scoring from whole-program training
represents an exploratory representation projection, not ground-truth span annotations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def char_to_line_col(text: str, char_idx: int) -> Tuple[int, int]:
    """Convert 0-indexed character offset into 1-indexed (line_number, column_number)."""
    if char_idx <= 0:
        return 1, 1
    clamped_idx = min(char_idx, len(text))
    prefix = text[:clamped_idx]
    lines = prefix.splitlines(keepends=True)
    if not lines:
        return 1, 1
    if prefix.endswith(("\n", "\r")):
        return len(lines) + 1, 1
    return len(lines), len(lines[-1]) + 1


def map_tokens_to_lines(
    code: str,
    token_strings: List[str],
    token_offsets: List[Tuple[int, int]],
    token_scores: np.ndarray,
    aggregation: str = "max",
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    """Map token vulnerability scores to source code lines and detailed token objects.
    
    Args:
        code: Original source code string.
        token_strings: Decoded tokens from tokenizer.
        token_offsets: List of (start_char, end_char) offsets.
        token_scores: 1D numpy array of token scores.
        aggregation: "max" or "mean" aggregation across tokens on each line.
        
    Returns:
        Tuple of:
            - Dict mapping line_number (1-indexed) -> line_score (float)
            - List of token detail dictionaries
    """
    token_details: List[Dict[str, Any]] = []
    line_scores_raw: Dict[int, List[float]] = {}

    code_lines = code.splitlines() if code else [""]
    num_lines = max(1, len(code_lines))

    # Initialize all lines with score 0.0
    for l_num in range(1, num_lines + 1):
        line_scores_raw[l_num] = []

    for idx, (tok_str, (start_c, end_c)) in enumerate(zip(token_strings, token_offsets)):
        score = float(token_scores[idx]) if idx < len(token_scores) else 0.0
        start_line, start_col = char_to_line_col(code, start_c)
        end_line, end_col = char_to_line_col(code, end_c)

        detail = {
            "token_index": idx,
            "token": tok_str,
            "score": round(score, 4),
            "start_char": start_c,
            "end_char": end_c,
            "start_line": start_line,
            "start_col": start_col,
            "end_line": end_line,
            "end_col": end_col,
        }
        token_details.append(detail)

        # Assign token score to its line
        if start_line in line_scores_raw:
            line_scores_raw[start_line].append(score)

    line_scores: Dict[int, float] = {}
    for l_num, scores in line_scores_raw.items():
        if not scores:
            line_scores[l_num] = 0.0
        elif aggregation == "max":
            line_scores[l_num] = round(float(max(scores)), 4)
        elif aggregation == "mean":
            line_scores[l_num] = round(float(sum(scores) / len(scores)), 4)
        else:
            line_scores[l_num] = round(float(max(scores)), 4)

    return line_scores, token_details


def extract_suspected_findings(
    code: str,
    line_scores: Dict[int, float],
    token_details: List[Dict[str, Any]],
    finding_threshold: float = 0.60,
) -> List[Dict[str, Any]]:
    """Extract candidate vulnerable regions and lines exceeding the finding threshold."""
    code_lines = code.splitlines() if code else [""]
    findings: List[Dict[str, Any]] = []

    for l_num, score in line_scores.items():
        if score >= finding_threshold:
            line_content = code_lines[l_num - 1] if l_num - 1 < len(code_lines) else ""
            # Filter tokens on this line
            line_tokens = [t for t in token_details if t["start_line"] == l_num]
            start_col = min([t["start_col"] for t in line_tokens]) if line_tokens else 1
            end_col = max([t["end_col"] for t in line_tokens]) if line_tokens else len(line_content) + 1

            findings.append({
                "line_number": l_num,
                "line_content": line_content,
                "score": score,
                "start_col": start_col,
                "end_col": end_col,
                "trigger_tokens": [
                    {"token": t["token"], "score": t["score"]}
                    for t in line_tokens if t["score"] >= finding_threshold
                ],
            })

    # Sort findings by score descending
    findings.sort(key=lambda x: x["score"], reverse=True)
    return findings
