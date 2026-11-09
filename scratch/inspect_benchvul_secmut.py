import requests
import io
import json
import pandas as pd
import ast
import re

sql_regex = re.compile(
    r'(select\s+[\s\S]*?\s+from|insert\s+into|update\s+[\s\S]*?\s+set|delete\s+from|'
    r'cursor\.execute|sqlite3|psycopg2|sqlalchemy|raw\(|execute_sql|\.raw_query|'
    r'db\.session\.execute|cursor\(\)\.execute|\.execute\(["\']\s*(select|insert|update|delete|create|drop))',
    re.IGNORECASE
)

print("--- Inspecting Mars203020/secmutbench ---")
url_sec = "https://huggingface.co/datasets/Mars203020/secmutbench/resolve/main/data/dataset.json"
r_sec = requests.get(url_sec, timeout=30)
if r_sec.status_code == 200:
    data = json.loads(r_sec.content)
    print(f"secmutbench type: {type(data)}, length: {len(data)}")
    if isinstance(data, list) and len(data) > 0:
        print("Keys of first element:", data[0].keys())
        sqli_samples = [d for d in data if '89' in str(d.get('cwe', '')) or '89' in str(d.get('cwe_id', '')) or 'sql' in str(d).lower()]
        print(f"CWE-89 / SQL candidate samples in secmutbench: {len(sqli_samples)}")
        if sqli_samples:
            print("Sample 0:", {k: str(v)[:80] for k, v in sqli_samples[0].items()})

print("\n--- Inspecting BenchVul for Python ---")
# Let's check size of BenchVul.csv
r_head = requests.head("https://huggingface.co/datasets/yikun-li/BenchVul/resolve/main/BenchVul.csv", timeout=15)
print("BenchVul.csv size:", r_head.headers.get("Content-Length"))
