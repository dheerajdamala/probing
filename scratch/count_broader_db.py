import pandas as pd
import re
import ast

db_sql_regex = re.compile(
    r'(\bselect\b[\s\S]*?\bfrom\b|\binsert\s+into\b|\bupdate\b[\s\S]*?\bset\b|\bdelete\s+from\b|'
    r'\bcursor\b|\bsqlite3\b|\bpsycopg2\b|\bsqlalchemy\b|\bpymysql\b|\bmysqldb\b|'
    r'\.execute\(|\.executemany\(|\.raw\(|\.raw_query\(|\.objects\.raw\(|'
    r'db\.session|connection\.cursor|\.fetchall\(|\.fetchone\()',
    re.IGNORECASE
)

for name in ["pycode_vul_train.csv", "pycode_vul_test.csv"]:
    df = pd.read_csv(f"security_probe_sql/data/raw/{name}")
    v_col = 'vulnerable_function_source' if 'vulnerable_function_source' in df.columns else 'function_code'
    p_col = 'patched_function_source'
    
    v_match = df[v_col].astype(str).apply(lambda s: bool(db_sql_regex.search(s)))
    p_match = df[p_col].astype(str).apply(lambda s: bool(db_sql_regex.search(s)))
    
    print(f"{name}: total {len(df)} rows")
    print(f"  vulnerable matches: {v_match.sum()}")
    print(f"  patched matches: {p_match.sum()}")
    print(f"  either matches: {(v_match | p_match).sum()}")
