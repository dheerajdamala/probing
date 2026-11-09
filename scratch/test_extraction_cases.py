import sys
sys.path.insert(0, ".")
from src.evaluation.run_benchmarks import extract_clean_code_completion
from src.evaluation.functional_eval import evaluate_functional_correctness, check_syntax

prompt = '''def lookup_user_data_v000(cursor, user_id: str):
    """Query users record by user_id from fintech_banking database."""
'''

test_cases_raw = [
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
    cursor.execute("SELECT 1")
    return cursor.fetchone()
```''',

    # Case 3: Model outputs conversational text then code
    '''Here is the completed function:
```python
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
```''',

    # Case 4: Model outputs only indented body
    '''    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
''',

    # Case 5: Model starts with def without markdown
    '''def lookup_user_data_v000(cursor, user_id: str):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    return cursor.fetchone()
''',
]

task_test_cases = [{'args': ['dummy_cursor', "'admin'"], 'expected_result': {'id': '1', 'username': 'alice', 'role': 'admin'}}]

for i, raw in enumerate(test_cases_raw, 1):
    cleaned = extract_clean_code_completion(prompt, raw)
    full_code = prompt + cleaned
    syn_valid, syn_err = check_syntax(full_code)
    fn_eval = evaluate_functional_correctness(full_code, task_test_cases, "lookup_user_data_v000")
    print(f"=== CASE {i} ===")
    print("CLEANED:", repr(cleaned))
    print("FULL CODE:\n" + full_code)
    print("SYNTAX:", syn_valid, syn_err)
    print("FN EVAL:", fn_eval)
    print()
