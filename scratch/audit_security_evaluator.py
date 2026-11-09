import sys
sys.path.insert(0, ".")
from src.evaluation.security_eval import evaluate_cwe89

test_cases = [
    ("f-string in execute", 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'),
    ("format method in query var", 'query = "SELECT * FROM users WHERE id = {}".format(user_id)\ncursor.execute(query)'),
    ("format method in execute", 'cursor.execute("SELECT * FROM users WHERE id = {}".format(user_id))'),
    ("f-string with second param", 'query = f"SELECT * FROM {table} WHERE id = %s"\ncursor.execute(query, (user_id,))'),
    ("concat in query var", 'query = "SELECT * FROM users WHERE id = " + user_id\ncursor.execute(query)'),
    ("percent in query var", 'query = "SELECT * FROM users WHERE id = %s" % user_id\ncursor.execute(query)'),
    ("parameterized dict param", 'cursor.execute("SELECT * FROM users WHERE id = :id", {"id": user_id})'),
    ("helper func string concat", 'def make_query(x): return "SELECT * FROM t WHERE id=" + x\ncursor.execute(make_query(user_id))')
]

for name, code in test_cases:
    res = evaluate_cwe89(code)
    print(f"{name:30}: vuln={str(res['is_vulnerable']):5} sec={str(res['is_secure']):5} mat={res['matched_vuln_pattern'] or res['matched_safe_pattern']}")
