import pandas as pd
import json
import ast
import re
import hashlib

def compute_code_hash(code: str) -> str:
    norm = "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()

sql_pattern = re.compile(
    r'(select\s+[\s\S]*?\s+from|insert\s+into|update\s+[\s\S]*?\s+set|delete\s+from|'
    r'cursor\.execute|sqlite3|psycopg2|sqlalchemy|raw\(|execute_sql|\.raw_query|'
    r'db\.session\.execute|cursor\(\)\.execute|\.execute\(|'
    r'\.objects\.filter|\.objects\.raw|session\.query|\.execute_many)',
    re.IGNORECASE
)

sqli_vuln_pattern = re.compile(
    r'(execute\s*\(\s*f["\']|'
    r'execute\s*\(\s*["\'].*?%s.*?["\']\s*%\s*|'
    r'execute\s*\(\s*["\'].*?\{.*?["\']\.format|'
    r'execute\s*\(\s*["\'].*?\+\s*\w+|'
    r'raw\s*\(\s*f["\']|'
    r'raw\s*\(\s*["\'].*?%s.*?["\']\s*%\s*|'
    r'select\s+.*?WHERE.*?["\']\s*\+\s*\w+|'
    r'select\s+.*?WHERE.*?f["\'])',
    re.IGNORECASE
)

# Load current unique hashes from processed dataset
with open("security_probe_sql/data/processed/validated_sql_dataset.jsonl", "r", encoding="utf-8") as f:
    existing_hashes = {json.loads(line)["code_hash"] for line in f if line.strip()}
print(f"Current validated dataset size: {len(existing_hashes)} lines")

# Let's inspect pycode_vul
unique_new_snippets = {}

for split, csv_name in [("train", "pycode_vul_train.csv"), ("test", "pycode_vul_test.csv")]:
    df = pd.read_csv(f"security_probe_sql/data/raw/{csv_name}")
    v_col = 'vulnerable_function_source' if 'vulnerable_function_source' in df.columns else 'function_code'
    p_col = 'patched_function_source'
    
    for idx, row in df.iterrows():
        cwe = str(row.get('cwe_ids', '')) + " " + str(row.get('predicted_cwe_ids', ''))
        is_cwe89 = '89' in cwe
        
        for col_type, col in [("vuln", v_col), ("patch", p_col)]:
            code = str(row.get(col, '')) if pd.notna(row.get(col)) else ''
            if len(code.strip()) < 25:
                continue
            if not (is_cwe89 or sql_pattern.search(code)):
                continue
            try:
                ast.parse(code)
            except:
                try:
                    ast.parse("def _wrapper():\n" + "\n".join("    " + l for l in code.splitlines()))
                except:
                    continue
            h = compute_code_hash(code)
            if h not in existing_hashes and h not in unique_new_snippets:
                is_vuln = 1 if (is_cwe89 and col_type == "vuln") or sqli_vuln_pattern.search(code) else 0
                unique_new_snippets[h] = {
                    "sample_id": f"pycode_vul_{split}_{idx}_{col_type}",
                    "source": "S-AIR-L/PyCode-Vul",
                    "label": is_vuln,
                    "code_len": len(code)
                }

print(f"Found {len(unique_new_snippets)} additional valid, unique Python SQL snippets in PyCode-Vul!")
v_cnt = sum(1 for v in unique_new_snippets.values() if v["label"] == 1)
s_cnt = sum(1 for v in unique_new_snippets.values() if v["label"] == 0)
print(f"  Vulnerable: {v_cnt}, Safe: {s_cnt}")
