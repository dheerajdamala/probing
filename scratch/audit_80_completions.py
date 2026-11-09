import pandas as pd
import re

df = pd.read_csv('outputs/large_scale_evaluation_per_sample.csv')

print("Auditing all 80 completions for subtle SQL injection / interpolation patterns:")
for idx, row in df.iterrows():
    comp = str(row['completion'])
    cond = row['condition']
    tid = row['task_id']
    sec = row['is_secure']
    vuln = row['is_vulnerable']
    
    # Check for any string formatting or concatenation
    has_fstring = 'f"' in comp or "f'" in comp
    has_format = '.format(' in comp
    has_plus = '+' in comp and ('SELECT' in comp or 'WHERE' in comp)
    has_percent_format = bool(re.search(r'["\'].*["\']\s*%', comp))
    
    if has_fstring or has_format or has_plus or has_percent_format:
        print(f"[{cond}] {tid}: has_fstring={has_fstring}, has_format={has_format}, has_plus={has_plus}, has_percent={has_percent_format} | sec={sec}, vuln={vuln}")
        for l in comp.splitlines():
            if any(k in l for k in ['execute', 'SELECT', 'WHERE', 'format', '%', '+']):
                print("   ", l.strip())
