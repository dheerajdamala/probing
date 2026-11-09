import json

with open('experiments/sqli_probe_v3_full_mistral/data/e2_excel/test.jsonl') as f:
    recs = {r['sample_id']: r for r in (json.loads(line) for line in f)}

with open('experiments/sqli_probe_v3_full_mistral/results/error_analysis.json') as f:
    ea = json.load(f)

print('=== FALSE POSITIVES (Safe code classified as Vulnerable) ===')
for sid in ea['fp_sample_ids']:
    r = recs[sid]
    print(f"Sample ID: {sid} | DB API: {r.get('db_api')} | Case Type: {r.get('case_type')} | Description: {r.get('task_description')}")
    print("Code snippet:\n" + r.get('code')[:250] + "\n" + "-"*40)

print('\n=== FALSE NEGATIVES (Vulnerable code classified as Safe) ===')
for sid in ea['fn_sample_ids']:
    r = recs[sid]
    print(f"Sample ID: {sid} | DB API: {r.get('db_api')} | Case Type: {r.get('case_type')} | Description: {r.get('task_description')}")
    print("Code snippet:\n" + r.get('code')[:250] + "\n" + "-"*40)

