import sys
sys.path.insert(0, ".")
import json
from src.evaluation.functional_eval import evaluate_functional_correctness, check_syntax
from src.evaluation.security_eval import evaluate_code_security

with open("data/processed/test.json") as f:
    d = json.load(f)["paired"]

print("=== AUDIT OF REFERENCE IMPLEMENTATIONS ===")
all_safe_pass = True
all_vuln_detected = True

for item in d:
    fid = item["id"]
    cwe = item["cwe"]
    prompt = item["prompt"]
    safe_code = prompt + item["safe_completion"]
    vuln_code = prompt + item["vulnerable_completion"]
    
    # Safe eval
    safe_syn, _ = check_syntax(safe_code)
    safe_fn = evaluate_functional_correctness(safe_code, item["test_cases"], fid)
    safe_sec = evaluate_code_security(safe_code, cwe)
    
    # Vuln eval
    vuln_syn, _ = check_syntax(vuln_code)
    vuln_fn = evaluate_functional_correctness(vuln_code, item["test_cases"], fid)
    vuln_sec = evaluate_code_security(vuln_code, cwe)
    
    print(f"[{fid}] {cwe}:")
    print(f"  SAFE: syn={safe_syn}, fn={safe_fn['functional_passed']}, sec={safe_sec['is_secure']}, vuln={safe_sec['is_vulnerable']}")
    print(f"  VULN: syn={vuln_syn}, fn={vuln_fn['functional_passed']}, sec={vuln_sec['is_secure']}, vuln={vuln_sec['is_vulnerable']}")
    
    if not (safe_syn and safe_fn["functional_passed"] and safe_sec["is_secure"] and not safe_sec["is_vulnerable"]):
        all_safe_pass = False
    if not (vuln_sec["is_vulnerable"] and not vuln_sec["is_secure"]):
        all_vuln_detected = False

print("\n--- Summary ---")
print(f"All safe references pass syntax, functional, and security evaluators: {all_safe_pass}")
print(f"All vulnerable references detected as vulnerable: {all_vuln_detected}")
