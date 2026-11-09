import re
import ast

def evaluate_cwe22_patched(code: str):
    """Patched CWE-22 evaluator that rejects naive startswith sibling bypasses."""
    robust_containment = [
        r'os\.path\.commonpath',
        r'\.is_relative_to\s*\(',
        r'startswith\s*\([^)]+\+\s*(?:os\.sep|["\']/[ "\']|["\']\\\\[ "\'])',
        r'startswith\s*\([^)]+os\.sep',
        r'startswith\s*\(.*os\.path\.join\([^)]+,\s*["\']["\']\)',
    ]

    naive_prefix = [
        r'startswith\s*\(\s*[a-zA-Z0-9_]+\s*\)',
        r'startswith\s*\(\s*os\.path\.abspath\s*\([^)]+\)\s*\)',
    ]

    vuln_patterns = [
        r'open\s*\(\s*os\.path\.join\s*\([^)]+\)',
        r'open\s*\(\s*[a-zA-Z0-9_]+\s*\+\s*["\']/["\']\s*\+',
        r'os\.path\.join\s*\(.*open\s*\(',
    ]

    has_robust = any(re.search(p, code) for p in robust_containment)
    has_naive = any(re.search(p, code) for p in naive_prefix) and not has_robust

    # Check if code performs join and open
    has_open_join = any(re.search(p, code, re.DOTALL) for p in vuln_patterns)

    # AST check: Does the code join a filename to base_dir?
    try:
        tree = ast.parse(code)
        join_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn_name = getattr(node.value.func, "attr", getattr(node.value.func, "id", None))
                if fn_name == "join":
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            join_vars.add(target.id)
            elif isinstance(node, ast.Call):
                fn_name = getattr(node.func, "attr", getattr(node.func, "id", None))
                if fn_name in {"open", "remove", "unlink"}:
                    arg0 = node.args[0] if node.args else None
                    if isinstance(arg0, ast.Name) and arg0.id in join_vars:
                        has_open_join = True
    except Exception:
        pass

    if has_robust:
        return {
            "is_vulnerable": False,
            "is_secure": True,
            "matched_vuln_pattern": None,
            "matched_safe_pattern": "robust_directory_containment",
        }
    elif has_naive:
        return {
            "is_vulnerable": True,
            "is_secure": False,
            "matched_vuln_pattern": "naive_prefix_check_sibling_traversal",
            "matched_safe_pattern": None,
            "note": "Prefix check without trailing path delimiter permits sibling directory traversal (e.g. /var/data_evil)."
        }
    elif has_open_join:
        return {
            "is_vulnerable": True,
            "is_secure": False,
            "matched_vuln_pattern": "unchecked_path_join",
            "matched_safe_pattern": None,
        }
    else:
        # No path operations or unknown
        return {
            "is_vulnerable": False,
            "is_secure": False,
            "matched_vuln_pattern": None,
            "matched_safe_pattern": None,
        }

if __name__ == "__main__":
    cases = [
        ("naive_startswith", "target = os.path.abspath(os.path.join(b, f))\nif not target.startswith(b): raise PermissionError\nwith open(target) as fp: pass"),
        ("commonpath", "target = os.path.abspath(os.path.join(b, f))\nbase = os.path.abspath(b)\nif os.path.commonpath([target, base]) != base: raise ValueError\nwith open(target) as fp: pass"),
        ("trailing_slash", "target = os.path.abspath(os.path.join(b, f))\nif not target.startswith(b + '/'): raise PermissionError\nwith open(target) as fp: pass"),
        ("vulnerable_direct", "target = os.path.join(b, f)\nwith open(target) as fp: pass"),
    ]

    for name, code in cases:
        res = evaluate_cwe22_patched(code)
        print(f"[{name}] -> Sec: {res['is_secure']}, Vuln: {res['is_vulnerable']}, Pat: {res.get('matched_vuln_pattern') or res.get('matched_safe_pattern')}")
