import requests
import io
import pandas as pd
import ast
import re

sql_regex = re.compile(
    r'(select\s+[\s\S]*?\s+from|insert\s+into|update\s+[\s\S]*?\s+set|delete\s+from|'
    r'cursor\.execute|sqlite3|psycopg2|sqlalchemy|raw\(|execute_sql|\.raw_query|'
    r'db\.session\.execute|cursor\(\)\.execute|\.execute\(["\']\s*(select|insert|update|delete|create|drop))',
    re.IGNORECASE
)

url = "https://huggingface.co/datasets/MickyMike/cvefixes_bigvul/raw/main/test.csv"
print("Reading MickyMike test.csv...")
df = pd.read_csv(url)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:8])
print("Languages/CWEs in columns:", [c for c in df.columns if 'lang' in c.lower() or 'cwe' in c.lower()])
if 'language' in df.columns:
    print("Languages:", df['language'].value_counts())
elif 'lang' in df.columns:
    print("Languages:", df['lang'].value_counts())

def is_python(code):
    try:
        ast.parse(str(code))
        return True
    except:
        return False

# Check if any column has python code
for c in ['processed_func', 'func_before', 'code', 'function_code']:
    if c in df.columns:
        py_mask = df[c].apply(is_python)
        print(f"Col {c} valid Python: {py_mask.sum()}")
        has_sql = df[c].astype(str).apply(lambda s: bool(sql_regex.search(s)))
        print(f"Col {c} has SQL: {has_sql.sum()}")
        print(f"Col {c} valid Python + SQL: {(py_mask & has_sql).sum()}")
