import pandas as pd

df = pd.read_csv('outputs/large_scale_evaluation_per_sample.csv')
t16 = df[df['task_id'] == 'task_cwe89_0016']
for idx, row in t16.iterrows():
    print(f"[{row['condition']}]")
    print("Functional passed:", row['functional_passed'])
    print(row['completion'])
    print("-" * 40)
