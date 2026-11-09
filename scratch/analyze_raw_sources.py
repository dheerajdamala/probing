import json
from collections import Counter

counts = Counter()
vuln_counts = Counter()
safe_counts = Counter()

with open("security_probe_sql/data/interim/raw_combined_records.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        src = r.get("source_dataset", "unknown")
        counts[src] += 1
        if r.get("label") == 1:
            vuln_counts[src] += 1
        else:
            safe_counts[src] += 1

print("--- RAW COMBINED RECORDS BY SOURCE ---")
for src, cnt in counts.most_common():
    print(f"{src:45s}: total={cnt:5d} (vuln={vuln_counts[src]:4d}, safe={safe_counts[src]:4d})")
print(f"Total raw: {sum(counts.values())}")
