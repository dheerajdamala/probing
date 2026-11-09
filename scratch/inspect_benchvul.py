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

print("Checking BenchVul...")
url = "https://huggingface.co/datasets/yikun-li/BenchVul/resolve/main/BenchVul.csv"
# Read first 1000 rows to check schema
r = requests.get(url, headers={"Range": "bytes=0-500000"}, timeout=30)
if r.status_code in (200, 206):
    lines = r.text.splitlines()
    print("BenchVul header:", lines[0])
    print("Line 1:", lines[1][:200])

print("\nChecking TitanVul...")
url2 = "https://huggingface.co/datasets/yikun-li/TitanVul/resolve/main/TitanVul.csv"
r2 = requests.get(url2, headers={"Range": "bytes=0-500000"}, timeout=30)
if r2.status_code in (200, 206):
    lines2 = r2.text.splitlines()
    print("TitanVul header:", lines2[0])
    print("Line 1:", lines2[1][:200])

print("\nChecking Mars203020/secmutbench data tree...")
r3 = requests.get("https://huggingface.co/api/datasets/Mars203020/secmutbench/tree/main/data", timeout=30)
if r3.status_code == 200:
    for item in r3.json():
        print(" ", item.get("path"), item.get("size"))
