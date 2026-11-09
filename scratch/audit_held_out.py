import sys
sys.path.insert(0, ".")
import json
from src.evaluation.functional_eval import evaluate_functional_correctness, check_syntax
from src.evaluation.security_eval import evaluate_code_security

with open("data/processed/held_out_evaluation.json") as f:
    d = json.load(f)["paired"]

print(f"=== CALIBRATION AUDIT OF HELD-OUT TASKS (N = {len(d)}) ===")
all_safe_pass = True
all_vuln_detected = True

for item in d:
    fid = item["id"]
    cwe = item["cwe"]
    prompt = item["prompt"]
    safe_code = prompt + item["safe_completion"]
    vuln_code = prompt + item["vulnerable_completion"]

    safe_syn, _ = check_syntax(safe_code)
    safe_fn = evaluate_functional_correctness(safe_code, item["test_cases"], fid)
    safe_sec = evaluate_code_security(safe_code, cwe)

    vuln_syn, _ = check_syntax(vuln_code)
    vuln_fn = evaluate_functional_correctness(vuln_code, item["test_cases"], fid)
    vuln_sec = evaluate_code_security(vuln_code, cwe)

    safe_ok = safe_syn and safe_fn["functional_passed"] and safe_sec["is_secure"] and not safe_sec["is_vulnerable"]
    vuln_ok = vuln_syn and vuln_fn["functional_passed"] and vuln_sec["is_vulnerable"] and not vuln_sec["is_secure"]

    print(f"[{fid}] {cwe}: Safe OK={safe_ok} (syn={safe_syn}, fn={safe_fn['functional_passed']}, sec={safe_sec['is_secure']}) | Vuln OK={vuln_ok} (vuln={vuln_sec['is_vulnerable']})")

    if not safe_ok:
        all_safe_pass = False
        print(f"   SAFE FAILURE REASON: fn={safe_fn.get('error')}, sec={safe_sec}")
    if not vuln_ok:
        all_vuln_detected = False
        print(f"   VULN FAILURE REASON: fn={vuln_fn.get('error')}, sec={vuln_sec}")

print(f"\nAll 10 safe references pass: {all_safe_pass}")
print(f"All 10 vulnerable controls detected: {all_vuln_detected}")
