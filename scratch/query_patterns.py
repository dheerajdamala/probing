import pandas as pd
import re

df = pd.read_csv('outputs/large_scale_evaluation_per_sample.csv')

query_types = {}
for idx, row in df.iterrows():
    comp = str(row['completion'])
    exec_lines = [l.strip() for l in comp.splitlines() if 'execute' in l]
    line_repr = '; '.join(exec_lines)
    # Simplify line_repr
    simplified = re.sub(r'["\'].*?["\']', '"SQL"', line_repr)
    query_types[simplified] = query_types.get(simplified, 0) + 1

print("Query execution patterns across 80 completions:")
for p, count in query_types.items():
    print(f"{count:2d} samples: {p}")
