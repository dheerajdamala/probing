"""Functional evaluation and isolated test execution module."""

from __future__ import annotations

import ast
import html
import inspect
import logging
import math
import os
import re
import types
import typing
from typing import Any, Dict, List, Optional, Tuple

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


class MockCompletedProcess:
    """Mock subprocess.run return object."""

    def __init__(self, stdout="file1.txt\nfile2.txt\n", returncode=0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class MockSubprocessModule:
    """Safe mock of subprocess module preventing real shell execution."""

    CompletedProcess = MockCompletedProcess

    @staticmethod
    def run(cmd, shell=False, capture_output=False, text=False, **kwargs):
        stdout = "file1.txt\nfile2.txt\n" if "ls" in str(cmd) else "output"
        return MockCompletedProcess(stdout=stdout, returncode=0)

    @staticmethod
    def Popen(cmd, shell=False, **kwargs):
        return MockCompletedProcess()

    @staticmethod
    def check_output(cmd, shell=False, text=False, **kwargs):
        return "file1.txt\nfile2.txt\n" if "ls" in str(cmd) else "output"

    @staticmethod
    def getoutput(cmd):
        return "file1.txt\nfile2.txt\n" if "ls" in str(cmd) else "output"


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Verify that generated code forms a syntactically valid Python AST."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
    except Exception as e:
        return False, str(e)


def create_safe_sandbox() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create restricted sandbox globals and safe import resolver."""
    import pathlib
    allowed_modules = {
        "typing": typing,
        "html": html,
        "os": os,
        "os.path": os.path,
        "pathlib": pathlib,
        "subprocess": MockSubprocessModule(),
        "math": math,
        "re": re,
    }

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base_name = name.split(".")[0]
        if name in allowed_modules:
            return allowed_modules[name]
        elif base_name in allowed_modules:
            return allowed_modules[base_name]
        else:
            raise ImportError(f"Import of module '{name}' is restricted in sandbox.")

    safe_builtins = {
        "__import__": safe_import,
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
        "bytes": bytes,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "all": all,
        "any": any,
        "enumerate": enumerate,
        "zip": zip,
        "slice": slice,
        "map": map,
        "filter": filter,
        "round": round,
        "sorted": sorted,
        "reversed": reversed,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "hasattr": hasattr,
        "getattr": getattr,
        "open": lambda *args, **kwargs: _mock_open(*args, **kwargs),
        "Exception": Exception,
        "ValueError": ValueError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "PermissionError": PermissionError,
        "TypeError": TypeError,
    }

    safe_globals = {
        "__builtins__": safe_builtins,
        "typing": typing,
        "html": html,
        "os": os,
        "pathlib": pathlib,
        "Path": pathlib.Path,
        "subprocess": MockSubprocessModule(),
        "dummy_cursor": MockCursor(),
    }
    return safe_globals, {}


def _mock_open(path, mode='r', **kwargs):
    """In-memory mock file context manager."""
    class MockFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return "sample document content"
        def write(self, data):
            return len(data)
    return MockFile()


def evaluate_functional_correctness(
    code: str,
    test_cases: List[Dict[str, Any]],
    function_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute code in an isolated dictionary scope with safe fixtures."""
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

    # Extract function names defined in the AST
    try:
        tree = ast.parse(code)
        ast_fn_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    except Exception:
        ast_fn_names = []

    safe_globals, local_scope = create_safe_sandbox()

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

    # Locate defined target function in local scope
    target_fn = None
    if function_name and function_name in local_scope:
        target_fn = local_scope[function_name]
    elif ast_fn_names:
        # Check AST-defined functions first (prefer the last defined function)
        for fn_name in reversed(ast_fn_names):
            cand = local_scope.get(fn_name)
            if isinstance(cand, types.FunctionType):
                target_fn = cand
                break

    # Fallback to any user-defined FunctionType (excluding classes and imported types)
    if target_fn is None:
        for k, v in local_scope.items():
            if isinstance(v, types.FunctionType) and not k.startswith("_"):
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

    tests_run = 0
    tests_passed = 0
    last_failure_reason = None

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
                else:
                    last_failure_reason = f"Result {repr(res)} != expected {repr(tc['expected_result'])}"
            elif "expected_contains" in tc:
                if isinstance(res, str) and tc["expected_contains"] in res:
                    tests_passed += 1
                else:
                    last_failure_reason = f"Result {repr(res)} did not contain {repr(tc['expected_contains'])}"
            else:
                # Function must execute and return a non-None value representing computed output
                if res is not None:
                    tests_passed += 1
                else:
                    last_failure_reason = "Function returned None (body incomplete or missing return statement)"
        except Exception as e:
            last_failure_reason = f"Test execution raised {type(e).__name__}: {str(e)}"

    pass_rate = (tests_passed / tests_run) if tests_run > 0 else 0.0
    all_passed = (tests_passed == tests_run) and tests_run > 0

    return {
        "syntax_valid": True,
        "functional_passed": all_passed,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "pass_rate": round(pass_rate, 4),
        "error": None if all_passed else (last_failure_reason or f"Failed {tests_run - tests_passed}/{tests_run} test cases"),
    }
