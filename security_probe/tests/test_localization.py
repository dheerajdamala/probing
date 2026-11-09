"""Unit tests for probe localization, character-to-line mapping, and overlap metrics."""

from __future__ import annotations

import pytest
import numpy as np

from security_probe.src.localization import (
    char_to_line_col,
    map_tokens_to_lines,
    extract_suspected_findings,
)
from security_probe.src.evaluation import evaluate_localization_overlap


def test_char_to_line_col_correspondence():
    """Test 3: Returned line and column positions correspond accurately to original source."""
    code = (
        "def foo(x):\n"      # Line 1: 12 chars + '\n' (char 0 to 12)
        "    y = x + 1\n"    # Line 2: 14 chars + '\n' (char 13 to 27)
        "    return y\n"     # Line 3: 13 chars + '\n' (char 28 to 41)
    )

    # Start of line 1
    l, c = char_to_line_col(code, 0)
    assert l == 1 and c == 1

    # 'foo' on line 1 (char 4)
    l, c = char_to_line_col(code, 4)
    assert l == 1 and c == 5

    # Start of line 2 (char 12, right after '\n')
    l, c = char_to_line_col(code, 12)
    assert l == 2 and c == 1

    # 'y' on line 2 (char 16: after 4 spaces)
    l, c = char_to_line_col(code, 16)
    assert l == 2 and c == 5


def test_token_score_line_aggregation():
    """Test aggregation of token scores into line-level findings."""
    code = (
        "def query_db(uid):\n"
        "    q = f'SELECT * FROM users WHERE id = {uid}'\n"
        "    return db.execute(q)\n"
    )

    # 4 synthetic tokens: 1 on line 1, 2 on line 2, 1 on line 3
    token_strings = ["def", "q", "{uid}", "return"]
    # Offsets in code:
    # "def" at 0..3 (line 1)
    # "q" at 23..24 (line 2)
    # "{uid}" at 56..61 (line 2)
    # "return" at 69..75 (line 3)
    token_offsets = [(0, 3), (23, 24), (56, 61), (69, 75)]
    token_scores = np.array([0.1, 0.7, 0.95, 0.2], dtype=np.float32)

    line_scores, token_details = map_tokens_to_lines(
        code=code,
        token_strings=token_strings,
        token_offsets=token_offsets,
        token_scores=token_scores,
        aggregation="max",
    )

    assert line_scores[1] == 0.1
    # Line 2 max of 0.7 and 0.95 is 0.95
    assert line_scores[2] == 0.95
    assert line_scores[3] == 0.2

    # Extract findings exceeding threshold 0.8
    findings = extract_suspected_findings(
        code=code,
        line_scores=line_scores,
        token_details=token_details,
        finding_threshold=0.80,
    )

    assert len(findings) == 1
    assert findings[0]["line_number"] == 2
    assert findings[0]["score"] == 0.95
    assert "SELECT" in findings[0]["line_content"]


def test_localization_overlap_metrics():
    """Test precision, recall, and IoU overlap on synthetic localization ground truth."""
    # 1. Both empty (true negative line localization)
    res_empty = evaluate_localization_overlap(set(), set())
    assert res_empty["precision"] == 1.0
    assert res_empty["recall"] == 1.0
    assert res_empty["iou"] == 1.0

    # 2. Perfect match
    res_perfect = evaluate_localization_overlap({2, 3}, {2, 3})
    assert res_perfect["precision"] == 1.0
    assert res_perfect["recall"] == 1.0
    assert res_perfect["iou"] == 1.0

    # 3. Partial overlap (Pred: {2, 3, 4}, GT: {3, 4, 5})
    # Intersection = {3, 4} (size 2), Union = {2, 3, 4, 5} (size 4)
    res_partial = evaluate_localization_overlap({2, 3, 4}, {3, 4, 5})
    assert res_partial["precision"] == round(2 / 3, 4)
    assert res_partial["recall"] == round(2 / 3, 4)
    assert res_partial["iou"] == 0.5

    # 4. Completely disjoint (Pred: {1}, GT: {5})
    res_disjoint = evaluate_localization_overlap({1}, {5})
    assert res_disjoint["precision"] == 0.0
    assert res_disjoint["recall"] == 0.0
    assert res_disjoint["iou"] == 0.0
