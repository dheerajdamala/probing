import ast
import types

code = '''from typing import List

def copy_subsequence(src: List[int], start: int, length: int) -> List[int]:
    """Copy subsequence of specified length starting at start."""
    if start < 0 or length < 0 or start + length > len(src):
        raise IndexError("Requested slice exceeds buffer boundary")
    return src[start : start + length]
'''

tree = ast.parse(code)
fn_names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
print("AST function names:", fn_names)
