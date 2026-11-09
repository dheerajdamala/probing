import json
import pandas as pd

# Load test split
test_samples = []
with open("security_probe_sql/data/processed/splits/test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        test_samples.append(json.loads(line))

annotated_test = [s for s in test_samples if s.get("ground_truth_spans_available") or (s.get("vulnerable_lines") and len(s.get("vulnerable_lines")) > 0)]
print(f"Total test samples: {len(test_samples)}")
print(f"Annotated test samples: {len(annotated_test)}")

for idx, s in enumerate(annotated_test):
    print(f"\n--- Annotated Sample {idx+1}: {s.get('sample_id')} ---")
    print(f"  Source: {s.get('source_dataset')}")
    print(f"  Repo: {s.get('repository')}")
    print(f"  Label: {s.get('label')}")
    print(f"  Vulnerable lines: {s.get('vulnerable_lines')}")
    print(f"  Span details: {s.get('span_details')}")
    code = s.get('code', '')
    lines = code.splitlines()
    print(f"  Total code lines: {len(lines)}")
    for vl in s.get('vulnerable_lines', []):
        if 1 <= vl <= len(lines):
            print(f"    Line {vl}: {lines[vl-1].strip()}")
