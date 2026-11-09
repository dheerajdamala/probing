def robust_extract_clean_code_completion(prompt: str, raw_completion: str) -> str:
    """Robustly extract and clean code completion from model generation.
    
    Preserves relative indentation, strips markdown fences, conversational text,
    and handles both full function re-declarations and indented body completions.
    """
    code = raw_completion
    
    # 1. Extract from markdown fences if present
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0]
        
    # Strip only leading/trailing blank lines, preserving column 0 indentation of lines
    code = code.strip("\r\n")
    if not code.strip():
        return ""
        
    lines = code.splitlines()
    cleaned_lines = []
    seen_first_def = False
    
    for line in lines:
        stripped = line.strip()
        # Truncate at subsequent functions, classes, or main blocks
        if line and not line.startswith(" ") and not line.startswith("\t"):
            if stripped.startswith("if __name__"):
                break
            if stripped.startswith("def "):
                if seen_first_def:
                    break
                seen_first_def = True
            elif stripped.startswith("class "):
                if seen_first_def or cleaned_lines:
                    break
        elif stripped.startswith("def ") and not seen_first_def:
            seen_first_def = True
            
        cleaned_lines.append(line)
        
    cleaned = "\n".join(cleaned_lines).rstrip() + "\n"
    return cleaned


def assemble_full_code(prompt: str, completion: str) -> str:
    """Assemble executable full code from prompt and extracted completion."""
    comp = completion.strip()
    if not comp:
        return prompt.strip() + "\n"
        
    # If the completion already defines the function (starts with def or imports then def),
    # use the completion directly so we don't duplicate function headers
    if comp.startswith("def ") or ("\ndef " in comp and not comp.startswith("    ")):
        return comp + "\n"
        
    # If completion does not start with indentation, indent each non-empty line by 4 spaces
    lines = completion.splitlines()
    first_non_empty = next((l for l in lines if l.strip()), "")
    if first_non_empty and not first_non_empty.startswith(" ") and not first_non_empty.startswith("\t"):
        indented_lines = ["    " + l if l.strip() else l for l in lines]
        return prompt.rstrip() + "\n" + "\n".join(indented_lines) + "\n"
        
    # If completion already starts with indentation, append directly to prompt
    return prompt.rstrip() + "\n" + completion.lstrip("\r\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness

    prompt = '''def lookup_user_data_v000(cursor, user_id: str):
    """Query users record by user_id from fintech_banking database."""
'''

    raw_cases = [
        # Case 1: Model repeats def inside markdown
        '''```python
def lookup_user_data_v000(cursor, user_id: str):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
```''',
        # Case 2: Model imports sqlite3 then defines function
        '''```python
import sqlite3

def lookup_user_data_v000(cursor, user_id: str):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
```''',
        # Case 3: Conversational text then markdown
        '''Here is the implementation:
```python
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
```''',
        # Case 4: Indented body only
        '''    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
''',
        # Case 5: def without markdown
        '''def lookup_user_data_v000(cursor, user_id: str):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
''',
    ]

    test_cases = [{'args': ['dummy_cursor', "'admin'"], 'expected_result': {'id': '1', 'username': 'alice', 'role': 'admin'}}]

    for i, raw in enumerate(raw_cases, 1):
        cleaned = robust_extract_clean_code_completion(prompt, raw)
        full = assemble_full_code(prompt, cleaned)
        syn, err = check_syntax(full)
        fn = evaluate_functional_correctness(full, test_cases, "lookup_user_data_v000")
        print(f"CASE {i}: syn={syn}, fn_pass={fn['functional_passed']}, err={fn.get('error')}")
        if not syn:
            print(f"FAILED CODE CASE {i}:\n{repr(full)}")


