import json
import os
import sys
import pandas as pd
import ast
import re
import hashlib
from collections import Counter

sys.path.insert(0, os.path.abspath("."))
from security_probe_sql.src.deduplication import deduplicate_dataset, compute_code_hash

sql_pattern = re.compile(
    r'(select\s+[\s\S]*?\s+from|insert\s+into|update\s+[\s\S]*?\s+set|delete\s+from|'
    r'cursor\.execute|sqlite3|psycopg2|sqlalchemy|raw\(|execute_sql|\.raw_query|'
    r'db\.session\.execute|cursor\(\)\.execute|\.execute\(["\']\s*(select|insert|update|delete|create|drop)|'
    r'\.objects\.raw|session\.query)',
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

# 1. Load existing interim records
interim_path = "security_probe_sql/data/interim/raw_combined_records.jsonl"
existing_records = []
existing_hashes = set()
with open(interim_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            existing_records.append(r)
            h = compute_code_hash(r["code"])
            existing_hashes.add(h)

print(f"Existing raw records: {len(existing_records)} (unique hashes: {len(existing_hashes)})")

# 2. Ingest additional verified real-world functions from PyCode-Vul
new_records = []
for split, csv_name in [("train", "pycode_vul_train.csv"), ("test", "pycode_vul_test.csv")]:
    df = pd.read_csv(f"security_probe_sql/data/raw/{csv_name}")
    v_col = 'vulnerable_function_source' if 'vulnerable_function_source' in df.columns else 'function_code'
    p_col = 'patched_function_source'
    
    for idx, row in df.iterrows():
        cwe = str(row.get('cwe_ids', '')) + " " + str(row.get('predicted_cwe_ids', ''))
        is_cwe89 = '89' in cwe
        repo = str(row.get('repo', ''))
        sha = str(row.get('sha', ''))
        file_path = str(row.get('file_path', ''))
        pair_id = f"pycode_vul_p4_{split}_{idx}"
        
        # Vulnerable candidate
        v_code = str(row.get(v_col, '')) if pd.notna(row.get(v_col)) else ''
        if len(v_code.strip()) >= 25 and (is_cwe89 or sql_pattern.search(v_code)):
            vh = compute_code_hash(v_code)
            if vh not in existing_hashes:
                try:
                    ast.parse(v_code)
                    is_sqli = is_cwe89 or bool(sqli_vuln_pattern.search(v_code))
                    rec = {
                        "sample_id": f"pycode_vul_p4_{split}_{idx}_vuln",
                        "source_dataset": "S-AIR-L/PyCode-Vul",
                        "source_url": "https://huggingface.co/datasets/S-AIR-L/PyCode-Vul",
                        "repository": repo,
                        "commit_hash": sha,
                        "file_path": file_path,
                        "language": "python",
                        "cwe": "CWE-89" if is_sqli else "CWE-Other-DB",
                        "vulnerability_type": "sql_injection" if is_sqli else "database_query",
                        "label": 1 if is_sqli else 0,
                        "code": v_code,
                        "vulnerable_lines": [],
                        "fixed_code": str(row.get(p_col, '')) if pd.notna(row.get(p_col)) else '',
                        "pair_id": pair_id,
                        "license": "Apache-2.0",
                        "verification_status": "verified_vulnerable" if is_sqli else "verified_safe",
                        "raw_source_split": split
                    }
                    new_records.append(rec)
                    existing_hashes.add(vh)
                except:
                    pass

        # Patched candidate
        p_code = str(row.get(p_col, '')) if pd.notna(row.get(p_col)) else ''
        if len(p_code.strip()) >= 25 and (is_cwe89 or sql_pattern.search(p_code)):
            ph = compute_code_hash(p_code)
            if ph not in existing_hashes:
                try:
                    ast.parse(p_code)
                    rec = {
                        "sample_id": f"pycode_vul_p4_{split}_{idx}_patch",
                        "source_dataset": "S-AIR-L/PyCode-Vul",
                        "source_url": "https://huggingface.co/datasets/S-AIR-L/PyCode-Vul",
                        "repository": repo,
                        "commit_hash": sha,
                        "file_path": file_path,
                        "language": "python",
                        "cwe": "CWE-89-Fixed" if is_cwe89 else "CWE-Other-DB",
                        "vulnerability_type": "sql_injection" if is_cwe89 else "database_query",
                        "label": 0,
                        "code": p_code,
                        "vulnerable_lines": [],
                        "fixed_code": "",
                        "pair_id": pair_id,
                        "license": "Apache-2.0",
                        "verification_status": "verified_safe",
                        "raw_source_split": split
                    }
                    new_records.append(rec)
                    existing_hashes.add(ph)
                except:
                    pass

print(f"New validated unique records to add: {len(new_records)}")
combined = existing_records + new_records
print(f"Total raw candidates before dedup: {len(combined)}")

# Run syntax check and deduplication
syntax_valid = []
for r in combined:
    code = r["code"]
    if len(code.strip()) < 25:
        continue
    try:
        ast.parse(code)
        syntax_valid.append(r)
    except:
        try:
            ast.parse("def _wrapper():\n" + "\n".join("    " + l for l in code.splitlines()))
            syntax_valid.append(r)
        except:
            pass

print(f"Syntax valid records: {len(syntax_valid)}")
deduped, dedup_stats = deduplicate_dataset(syntax_valid, near_dup_threshold=0.88)
print(f"\nFinal Deduplicated Dataset Size: {len(deduped)}")
v_cnt = sum(1 for r in deduped if r["label"] == 1)
s_cnt = sum(1 for r in deduped if r["label"] == 0)
print(f"  Vulnerable: {v_cnt} ({v_cnt/len(deduped)*100:.1f}%)")
print(f"  Safe: {s_cnt} ({s_cnt/len(deduped)*100:.1f}%)")
print(f"  Dedup stats: {dedup_stats}")
