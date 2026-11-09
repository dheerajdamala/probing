import pandas as pd
import json

df = pd.read_csv('outputs/large_scale_evaluation_per_sample.csv')

print("=================================================================")
print("AUDIT FOR task_cwe89_0006")
print("=================================================================")
t6 = df[df['task_id'] == 'task_cwe89_0006']
for idx, row in t6.iterrows():
    print(f"Condition: {row['condition']}")
    print(f"Functional passed: {row['functional_passed']}")
    print(f"Functional error: {row['functional_error']}")
    print(f"Is secure: {row['is_secure']}")
    print(f"Is vulnerable: {row['is_vulnerable']}")
    print("Completion:")
    print(row['completion'])
    print("-" * 50)

print("\n=================================================================")
print("AUDIT FOR task_cwe89_0015")
print("=================================================================")
t15 = df[df['task_id'] == 'task_cwe89_0015']
for idx, row in t15.iterrows():
    print(f"Condition: {row['condition']}")
    print(f"Functional passed: {row['functional_passed']}")
    print(f"Functional error: {row['functional_error']}")
    print(f"Is secure: {row['is_secure']}")
    print(f"Is vulnerable: {row['is_vulnerable']}")
    print("Completion:")
    print(row['completion'])
    print("-" * 50)

print("\n=================================================================")
print("AUDIT FOR task_cwe89_0004")
print("=================================================================")
t4 = df[df['task_id'] == 'task_cwe89_0004']
for idx, row in t4.iterrows():
    print(f"Condition: {row['condition']}")
    print(f"Functional passed: {row['functional_passed']}")
    print(f"Functional error: {row['functional_error']}")
    print(f"Is secure: {row['is_secure']}")
    print(f"Is vulnerable: {row['is_vulnerable']}")
    print(f"Matched pattern: {row['matched_pattern']}")
    print("Completion:")
    print(row['completion'])
    print("-" * 50)
