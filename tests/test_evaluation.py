"""Unit tests for security pattern analysis and functional execution."""

from __future__ import annotations

import pytest

from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness
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


def test_functional_correctness_isolated_execution():
    """Test safe sandbox execution and assertion testing."""
    code = "def multiply(a, b):\n    return a * b\n"
    test_cases = [
        {"args": [3, 4], "expected_result": 12},
        {"args": [0, 5], "expected_result": 0},
    ]

    res = evaluate_functional_correctness(code, test_cases, function_name="multiply")
    assert res["syntax_valid"]
    assert res["functional_passed"]
    assert res["pass_rate"] == 1.0


def test_syntax_error_handling():
    """Test handling of invalid syntax."""
    malformed_code = "def broken_fn(:\n  return"
    syntax_ok, err = check_syntax(malformed_code)
    assert not syntax_ok
    assert err is not None
