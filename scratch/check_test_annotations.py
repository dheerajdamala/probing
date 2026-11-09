import json

with open('experiments/sqli_probe_v3_full_mistral/data/e2_excel/test.jsonl') as f:
    recs = [json.loads(line) for line in f]

print(f"Total test samples: {len(recs)}")
sinks_present = [r for r in recs if r.get('sink') and r.get('sink') != 'None']
vuln_lines_present = [r for r in recs if r.get('vulnerable_lines') and r.get('vulnerable_lines') not in ('None', '[]')]
query_styles = set(r.get('query_style') for r in recs)

print(f"Samples with non-None 'sink': {len(sinks_present)}")
print(f"Samples with non-empty 'vulnerable_lines': {len(vuln_lines_present)}")
print(f"Unique query_style values: {query_styles}")

for r in recs[:3]:
    print("-" * 50)
    print(f"ID: {r['sample_id']}, Label: {r['label']}, Query Style: {r.get('query_style')}")
    print(f"Sink: {r.get('sink')}")
    print(f"Vulnerable Lines: {r.get('vulnerable_lines')}")
    print(f"Code preview:\n{r['code'][:150]}")

