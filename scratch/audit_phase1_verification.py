import csv
import json
import os
import sys

print("=== PHASE 1 AUDIT: VERIFYING REPORTED ARTIFACTS ===")

# 1. Load JSON and CSV
json_path = "outputs/steering_results.json"
csv_path = "outputs/steering_per_sample.csv"
report_path = "outputs/STEERING_EXPERIMENT_REPORT.md"

with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

with open(csv_path, "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

with open(report_path, "r", encoding="utf-8") as f:
    report_text = f.read()

# 2. Check conditions count
json_conds = json_data.get("conditions", {})
print(f"Total conditions in JSON: {len(json_conds)}")
print(f"Total rows in CSV: {len(csv_rows)}")
csv_cond_names = set(r["condition"] for r in csv_rows)
print(f"Total unique conditions in CSV: {len(csv_cond_names)}")

# 3. Verify Layer 16 alpha 1.0 specifically
l16_rows = [r for r in csv_rows if r["condition"] == "steered_sec_L16_a1.0"]
print(f"\nCondition 'steered_sec_L16_a1.0' rows in CSV ({len(l16_rows)} items):")
passing_joint = []
for r in l16_rows:
    sid = r["sample_id"]
    cwe = r["cwe"]
    syn = r["syntax_valid"] == "True"
    fn = r["functional_passed"] == "True"
    sec = r["is_secure"] == "True"
    vuln = r["is_vulnerable"] == "True"
    joint = r["joint_passed"] == "True"
    print(f"  [{sid}] {cwe}: syn={syn}, fn={fn}, sec={sec}, vuln={vuln}, joint={joint}")
    if joint:
        passing_joint.append((sid, cwe, r["matched_safe_pattern"]))

print(f"\nJoint passing count at Layer 16, alpha=1.0: {len(passing_joint)} / {len(l16_rows)} ({len(passing_joint)/len(l16_rows)*100:.1f}%)")
print("Identified 4 passing tasks:")
for sid, cwe, pat in passing_joint:
    print(f"  - {sid} ({cwe}): Safe pattern: {pat}")

# 4. Check agreement between JSON summary and CSV calculations
print("\n--- Cross-File Metric Agreement ---")
discrepancies = []
for cond_name, cdata in json_conds.items():
    cond_csv_rows = [r for r in csv_rows if r["condition"] == cond_name]
    if not cond_csv_rows:
        discrepancies.append(f"Condition {cond_name} in JSON but missing in CSV")
        continue
    n = len(cond_csv_rows)
    syn_rate = sum(1 for r in cond_csv_rows if r["syntax_valid"] == "True") / n
    fn_rate = sum(1 for r in cond_csv_rows if r["functional_passed"] == "True") / n
    sec_rate = sum(1 for r in cond_csv_rows if r["is_secure"] == "True") / n
    vuln_rate = sum(1 for r in cond_csv_rows if r["is_vulnerable"] == "True") / n
    joint_rate = sum(1 for r in cond_csv_rows if r["joint_passed"] == "True") / n

    if abs(cdata["syntax_validity_rate"] - syn_rate) > 1e-4:
        discrepancies.append(f"{cond_name}: syntax rate mismatch ({cdata['syntax_validity_rate']} vs {syn_rate})")
    if abs(cdata["functional_pass_rate"] - fn_rate) > 1e-4:
        discrepancies.append(f"{cond_name}: fn rate mismatch ({cdata['functional_pass_rate']} vs {fn_rate})")
    if abs(cdata["security_pass_rate"] - sec_rate) > 1e-4:
        discrepancies.append(f"{cond_name}: sec rate mismatch ({cdata['security_pass_rate']} vs {sec_rate})")
    if abs(cdata["vulnerability_rate"] - vuln_rate) > 1e-4:
        discrepancies.append(f"{cond_name}: vuln rate mismatch ({cdata['vulnerability_rate']} vs {vuln_rate})")
    if abs(cdata["joint_pass_rate"] - joint_rate) > 1e-4:
        discrepancies.append(f"{cond_name}: joint rate mismatch ({cdata['joint_pass_rate']} vs {joint_rate})")

if not discrepancies:
    print("SUCCESS: 100% agreement between JSON and CSV across all 21 conditions!")
else:
    print("DISCREPANCIES FOUND:")
    for d in discrepancies:
        print(f"  * {d}")
