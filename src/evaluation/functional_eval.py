"""Functional evaluation and isolated test execution module."""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MockCursor:
    """Mock database cursor for safe in-memory test execution."""

    def __init__(self):
        self.last_query = None
        self.last_params = None
        self.rowcount = 1

    def execute(self, query: str, params=None):
        self.last_query = query
        self.last_params = params

    def fetchone(self):
        return {"id": "1", "username": "alice", "role": "admin"}

    def fetchall(self):
        return [{"id": "1", "name": "Item A", "price": 10.0}]


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Verify that generated code forms a syntactically valid Python AST."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
    except Exception as e:
        return False, str(e)


def evaluate_functional_correctness(
    code: str,
    test_cases: List[Dict[str, Any]],
    function_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute code in an isolated dictionary scope with mocked safe fixtures."""
    syntax_valid, syntax_err = check_syntax(code)
    if not syntax_valid:
        return {
            "syntax_valid": False,
            "functional_passed": False,
            "tests_run": 0,
            "tests_passed": 0,
            "pass_rate": 0.0,
            "error": syntax_err,
        }

    # Safe sandbox globals
    safe_globals = {
        "__builtins__": {
            "len": len,
            "range": range,
            "print": lambda *args: None,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "Exception": Exception,
            "ValueError": ValueError,
            "IndexError": IndexError,
            "KeyError": KeyError,
            "PermissionError": PermissionError,
            "TypeError": TypeError,
        },
        "dummy_cursor": MockCursor(),
    }
    local_scope: Dict[str, Any] = {}

    try:
        exec(code, safe_globals, local_scope)
    except Exception as e:
        return {
            "syntax_valid": True,
            "functional_passed": False,
            "tests_run": 0,
            "tests_passed": 0,
            "pass_rate": 0.0,
            "error": f"Exec error: {type(e).__name__}: {str(e)}",
        }

    # If no test cases are specified, compilation and definition is considered successful
    if not test_cases:
        return {
            "syntax_valid": True,
            "functional_passed": True,
            "tests_run": 0,
            "tests_passed": 0,
            "pass_rate": 1.0,
            "error": None,
        }

    tests_run = 0
    tests_passed = 0

    # Locate defined function in local scope
    target_fn = None
    if function_name and function_name in local_scope:
        target_fn = local_scope[function_name]
    else:
        # Pick first callable defined in local_scope
        for k, v in local_scope.items():
            if callable(v):
                target_fn = v
                break

    if target_fn is None:
        return {
            "syntax_valid": True,
            "functional_passed": False,
            "tests_run": len(test_cases),
            "tests_passed": 0,
            "pass_rate": 0.0,
            "error": "No callable function found in executed scope",
        }

    for tc in test_cases:
        tests_run += 1
        args = tc.get("args", [])
        resolved_args = []
        for a in args:
            if a == "dummy_cursor":
                resolved_args.append(MockCursor())
            else:
                resolved_args.append(a)

        try:
            res = target_fn(*resolved_args)
            if "expected_result" in tc:
                if res == tc["expected_result"]:
                    tests_passed += 1
            elif "expected_contains" in tc:
                if isinstance(res, str) and tc["expected_contains"] in res:
                    tests_passed += 1
            else:
                # If execution completed without throwing unexpected exception
                tests_passed += 1
        except Exception:
            pass

    pass_rate = (tests_passed / tests_run) if tests_run > 0 else 0.0
    all_passed = (tests_passed == tests_run) and tests_run > 0

    return {
        "syntax_valid": True,
        "functional_passed": all_passed,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "pass_rate": round(pass_rate, 4),
        "error": None,
    }


if __name__ == "__main__":
    sample_code = "def add_nums(a, b):\n    return a + b\n"
    res = evaluate_functional_correctness(sample_code, [{"args": [2, 3], "expected_result": 5}])
    print("Functional test eval:", res)
