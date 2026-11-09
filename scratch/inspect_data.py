import os
import re
import pandas as pd

sql_regex = re.compile(
    r'(select\s+[\s\S]*?\s+from|insert\s+into|update\s+[\s\S]*?\s+set|delete\s+from|'
    r'cursor\.execute|sqlite3|psycopg2|sqlalchemy|raw\(|execute_sql|\.raw_query|'
    r'db\.session\.execute|cursor\(\)\.execute|\.execute\(["\']\s*(select|insert|update|delete|create|drop))',
    re.IGNORECASE
)

raw_dir = "security_probe_sql/data/raw"

print("--- Inspecting cvefixes_pyvul_merged.parquet ---")
f = os.path.join(raw_dir, "cvefixes_pyvul_merged.parquet")
if os.path.exists(f):
    df = pd.read_parquet(f)
    print("Shape:", df.shape)
    has_b = df["code_before"].astype(str).apply(lambda s: bool(sql_regex.search(s)))
    has_a = df["code_after"].astype(str).apply(lambda s: bool(sql_regex.search(s)))
    print("Has SQL in code_before:", has_b.sum())
    print("Has SQL in code_after:", has_a.sum())
    print("Has SQL in either:", (has_b | has_a).sum())
    cwe89 = df["cwe_id"].astype(str).str.contains("89") | df["cwe_name"].astype(str).str.contains("SQL", case=False, na=False)
    print("CWE-89 rows:", cwe89.sum())

print("\n--- Inspecting detectvul_cvefixes.parquet ---")
f2 = os.path.join(raw_dir, "detectvul_cvefixes.parquet")
if os.path.exists(f2):
    df2 = pd.read_parquet(f2)
    print("Shape:", df2.shape)
    def check_detectvul(row):
        lines = row.get("raw_lines")
        if lines is None or not hasattr(lines, "__iter__"):
            return False
        code = "\n".join(str(l) for l in lines)
        return bool(sql_regex.search(code))
    has_sql_d = df2.apply(check_detectvul, axis=1)
    print("Has SQL:", has_sql_d.sum())

print("\n--- Inspecting scthornton_web_train.parquet ---")
f3 = os.path.join(raw_dir, "scthornton_web_train.parquet")
if os.path.exists(f3):
    df3 = pd.read_parquet(f3)
    print("Shape:", df3.shape)
    def check_scthornton(row):
        c = str(row.get("context", ""))
        return bool(sql_regex.search(c))
    has_sql_sc = df3.apply(check_scthornton, axis=1)
    print("Has SQL:", has_sql_sc.sum())
