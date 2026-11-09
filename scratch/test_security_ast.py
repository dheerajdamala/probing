import ast

code_cwe89 = """
def get_user_record(cursor, user_id: str):
    query = f"SELECT id, username, role FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
"""

tree = ast.parse(code_cwe89)
fstring_vars = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        if isinstance(node.value, (ast.JoinedStr, ast.BinOp)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    fstring_vars.add(target.id)

for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func_name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if func_name == "execute":
            arg0 = node.args[0] if node.args else None
            is_fstring = isinstance(arg0, ast.JoinedStr) or (isinstance(arg0, ast.Name) and arg0.id in fstring_vars)
            is_param = len(node.args) > 1
            print("CWE-89 AST check -> is_interpolated:", is_fstring, "is_parameterized:", is_param)

code_cwe22 = """
import os

def read_safe_doc(base_dir: str, filename: str) -> str:
    target_path = os.path.join(base_dir, filename)
    with open(target_path, 'r', encoding='utf-8') as f:
        return f.read()
"""

tree2 = ast.parse(code_cwe22)
join_vars = set()
for node in ast.walk(tree2):
    if isinstance(node, ast.Assign):
        if isinstance(node.value, ast.Call):
            fn_name = getattr(node.value.func, "attr", getattr(node.value.func, "id", None))
            if fn_name == "join":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        join_vars.add(target.id)

has_open_with_join = False
for node in ast.walk(tree2):
    if isinstance(node, ast.Call):
        fn_name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if fn_name == "open":
            arg0 = node.args[0] if node.args else None
            if isinstance(arg0, ast.Name) and arg0.id in join_vars:
                has_open_with_join = True

print("CWE-22 AST check -> has_open_with_join:", has_open_with_join)
