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

url = "https://huggingface.co/datasets/ayshajavd/code-security-vulnerability-dataset/resolve/main/data/test-00000-of-00001.parquet"
print("Downloading ayshajavd test parquet (13MB)...")
r = requests.get(url, timeout=60)
if r.status_code == 200:
    df = pd.read_parquet(io.BytesIO(r.content))
    print("ayshajavd test shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample:\n", df.head(2))
    if 'language' in df.columns:
        print("Languages:\n", df['language'].value_counts())
    df_py = df[df['language'].astype(str).str.lower() == 'python']
    print(f"Python rows: {len(df_py)}")
    if len(df_py) > 0:
        has_sql = df_py['code'].astype(str).apply(lambda s: bool(sql_regex.search(s)))
        print(f"Python rows with SQL: {has_sql.sum()}")
        if 'cwe' in df_py.columns:
            print("CWE values in Python SQL:\n", df_py[has_sql]['cwe'].value_counts())
