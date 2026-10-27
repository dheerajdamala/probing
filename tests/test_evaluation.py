"""Unit tests for security pattern analysis, sandbox execution, and code extraction."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness
from src.evaluation.run_benchmarks import extract_clean_code_completion
from src.evaluation.security_eval import evaluate_code_security


def test_security_rules_cwe89():
    """Test SQL injection pattern detection."""
    safe_code = "def get_u(cursor, uid):\n    cursor.execute('SELECT * FROM u WHERE id = %s', (uid,))"
    vuln_code = "def get_u(cursor, uid):\n    cursor.execute(f'SELECT * FROM u WHERE id = {uid}')"

    res_safe = evaluate_code_security(safe_code, "CWE-89")
    res_vuln = evaluate_code_security(vuln_code, "CWE-89")

    assert res_safe["is_secure"]
    assert not res_safe["is_vulnerable"]

    assert not res_vuln["is_secure"]
    assert res_vuln["is_vulnerable"]


def test_security_rules_cwe78():
    """Test command injection pattern detection."""
    safe_code = "import subprocess\nsubprocess.run(['ls', '-l'], shell=False)"
    vuln_code = "import subprocess\nsubprocess.run(f'ls {user_dir}', shell=True)"

    res_safe = evaluate_code_security(safe_code, "CWE-78")
    res_vuln = evaluate_code_security(vuln_code, "CWE-78")

    assert res_safe["is_secure"]
    assert res_vuln["is_vulnerable"]


def test_cwe119_symmetric_behavior():
    """Test that gibberish text without indexing is not flagged as a buffer overflow."""
    gibberish = "def foo():\n    issтротротро"
    res = evaluate_code_security(gibberish, "CWE-119")
    assert not res["is_vulnerable"], "Gibberish without index access was flagged as vulnerable!"
    assert not res["is_secure"]

    # Explicit unchecked indexing
    unsafe_idx = "def copy(src, s, l):\n    return src[s : s + l]\n"
    res_unsafe = evaluate_code_security(unsafe_idx, "CWE-119")
    assert res_unsafe["is_vulnerable"]
    assert not res_unsafe["is_secure"]

    # Explicit checked indexing
    safe_idx = "def copy(src, s, l):\n    if s < 0 or s + l > len(src): raise IndexError()\n    return src[s : s + l]\n"
    res_safe = evaluate_code_security(safe_idx, "CWE-119")
    assert not res_safe["is_vulnerable"]
    assert res_safe["is_secure"]


def test_sandbox_imports():
    """Regression test: verify sandbox supports imports needed by benchmark prompts."""
    code_typing = "from typing import List\ndef f(x: List[int]) -> int:\n    return sum(x)\n"
    res1 = evaluate_functional_correctness(code_typing, [{"args": [[1, 2, 3]], "expected_result": 6}])
    assert res1["syntax_valid"]
    assert res1["functional_passed"], f"Typing import failed: {res1.get('error')}"

    code_html = "import html\ndef f(name: str) -> str:\n    return html.escape(name)\n"
    res2 = evaluate_functional_correctness(code_html, [{"args": ["<script>"], "expected_result": "&lt;script&gt;"}])
    assert res2["syntax_valid"]
    assert res2["functional_passed"], f"Html import failed: {res2.get('error')}"


def test_extract_clean_code_completion():
    """Test stripping markdown fences and truncating trailing definitions."""
    prompt = "def add(a, b):\n"
    raw_md = "```python\n    return a + b\n```\nHere is an explanation."
    clean = extract_clean_code_completion(prompt, raw_md)
    assert clean.strip() == "return a + b"
    assert "```" not in clean


def test_ground_truth_test_cases():
    """Verify that all ground-truth safe completions pass functional and security tests."""
    test_file = Path("data/processed/test.json")
    if not test_file.exists():
        pytest.skip("data/processed/test.json not found")

    with open(test_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)["paired"]

    for item in pairs:
        full_safe_code = item["prompt"] + item["safe_completion"]
        test_cases = item.get("test_cases", [])

        # Functional correctness check
        func_res = evaluate_functional_correctness(full_safe_code, test_cases)
        assert func_res["syntax_valid"], f"Safe code syntax invalid for {item['id']}: {func_res.get('error')}"
        assert func_res["functional_passed"], f"Safe code failed test cases for {item['id']}: {func_res.get('error')}"

        # Security check
        sec_res = evaluate_code_security(full_safe_code, item["cwe"])
        assert sec_res["is_secure"], f"Safe code failed security rule for {item['id']}"
        assert not sec_res["is_vulnerable"], f"Safe code flagged as vulnerable for {item['id']}"
