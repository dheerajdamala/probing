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


def test_sqlite3_sandbox_import():
    """Verify sqlite3 import is allowed in sandbox for SQL query tasks."""
    code = (
        "import sqlite3\n"
        "def query_db(cursor, uid: str):\n"
        "    cursor.execute('SELECT 1')\n"
        "    return cursor.fetchone()\n"
    )
    res = evaluate_functional_correctness(code, [{"args": ["dummy_cursor", "admin"], "expected_result": {"id": "1", "username": "alice", "role": "admin"}}])
    assert res["syntax_valid"]
    assert res["functional_passed"], f"sqlite3 sandbox import failed: {res.get('error')}"


def test_markdown_fence_def_extraction():
    """Regression test: verify that leading newlines in markdown code blocks do not discard def."""
    from src.evaluation.run_benchmarks import assemble_full_code

    prompt = "def lookup_user(cursor, uid: str):\n    \"\"\"Query user.\"\"\"\n"
    raw_md = (
        "```python\n"
        "def lookup_user(cursor, uid: str):\n"
        "    query = f\"SELECT * FROM users WHERE id = '{uid}'\"\n"
        "    cursor.execute(query)\n"
        "    return cursor.fetchone()\n"
        "```\n"
    )
    clean = extract_clean_code_completion(prompt, raw_md)
    assert clean.strip(), "clean_completion was unexpectedly empty!"
    assert "def lookup_user" in clean

    full = assemble_full_code(prompt, clean)
    syn_valid, syn_err = check_syntax(full)
    assert syn_valid, f"Syntax invalid: {syn_err}"

    res = evaluate_functional_correctness(full, [{"args": ["dummy_cursor", "admin"], "expected_result": {"id": "1", "username": "alice", "role": "admin"}}])
    assert res["functional_passed"], f"Functional evaluation failed: {res.get('error')}"


def test_evaluator_status_model_handling():
    """Verify status model correctly assigns MISSING_COMPLETION, EXECUTION_ERROR, PASS, and FAIL."""
    from src.evaluation.run_benchmarks import resolve_evaluation_status

    prompt = "def foo(x):\n"
    
    # 1. Missing completion
    s1 = resolve_evaluation_status(prompt=prompt, raw_comp="", clean_comp="", syn_valid=True, syn_err=None, fn_eval={"functional_passed": False}, sec_eval={"is_secure": False, "is_vulnerable": False})
    assert s1["evaluation_status"] == "MISSING_COMPLETION"
    assert s1["is_secure"] is None
    assert s1["is_vulnerable"] is None

    # 2. Syntax / Execution error
    s2 = resolve_evaluation_status(prompt=prompt, raw_comp="invalid syntax :::", clean_comp="invalid syntax :::", syn_valid=False, syn_err="SyntaxError", fn_eval={"functional_passed": False}, sec_eval={"is_secure": False, "is_vulnerable": False})
    assert s2["evaluation_status"] == "EXECUTION_ERROR"
    assert s2["is_secure"] is None
    assert s2["is_vulnerable"] is None

    # 3. Functional pass + Security pass = PASS
    s3 = resolve_evaluation_status(prompt=prompt, raw_comp="    return x", clean_comp="    return x", syn_valid=True, syn_err=None, fn_eval={"functional_passed": True}, sec_eval={"is_secure": True, "is_vulnerable": False})
    assert s3["evaluation_status"] == "PASS"
    assert s3["joint_passed"] is True

    # 4. Functional pass + Security fail = FAIL
    s4 = resolve_evaluation_status(prompt=prompt, raw_comp="    return x", clean_comp="    return x", syn_valid=True, syn_err=None, fn_eval={"functional_passed": True}, sec_eval={"is_secure": False, "is_vulnerable": True})
    assert s4["evaluation_status"] == "FAIL"
    assert s4["joint_passed"] is False
