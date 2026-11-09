import sys
sys.path.insert(0, ".")
import json
from src.evaluation.functional_eval import evaluate_functional_correctness

empty_code = '''import os
import subprocess

def count_files_in_dir(path: str) -> str:
    """Docstring"""
'''

res = evaluate_functional_correctness(empty_code, [{"args": ["."], "expected_shell": False}], "cwe78")
print("Empty code passed:", res["functional_passed"])
