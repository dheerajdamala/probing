"""Generate per-sample diagnostic CSV from benchmark results."""

import csv
import json
from pathlib import Path

with open("outputs/benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

csv_path = Path("outputs/per_sample_diagnostics.csv")

fieldnames = [
    "condition",
    "sample_id",
    "cwe",
    "syntax_valid",
    "functional_passed",
    "is_secure",
    "is_vulnerable",
    "syntax_error",
    "functional_error",
    "matched_vuln_pattern",
    "matched_safe_pattern",
    "root_cause_category",
    "completion_snippet",
]

rows = []
for cond_name, cond in data["conditions"].items():
    for item in cond["per_item_results"]:
        syntax_ok = item["functional"].get("syntax_valid", False)
        func_ok = item["functional"].get("functional_passed", False)
        sec_ok = item["security"].get("is_secure", False)
        vuln_ok = item["security"].get("is_vulnerable", False)
        err = item["functional"].get("error") or ""

        syntax_err = err if not syntax_ok else ""
        func_err = err if syntax_ok and not func_ok else ""

        # Root cause categorization
        if "ImportError: __import__ not found" in err:
            category = "SANDBOX_IMPORT_BUG"
        elif "invalid syntax" in err or "SyntaxError" in err:
            category = "MOCK_WEIGHTS_MALFORMED_SYNTAX"
        elif vuln_ok and item["cwe"] == "CWE-119" and not syntax_ok:
            category = "ASYMMETRIC_CWE119_DEFAULT_VULNERABLE"
        else:
            category = "MOCK_UNINITIALIZED_MODEL"

        comp_clean = item["completion"].encode("ascii", "backslashreplace").decode("ascii").replace("\n", "\\n")
        if len(comp_clean) > 80:
            comp_clean = comp_clean[:80] + "..."

        rows.append({
            "condition": cond_name,
            "sample_id": item["id"],
            "cwe": item["cwe"],
            "syntax_valid": syntax_ok,
            "functional_passed": func_ok,
            "is_secure": sec_ok,
            "is_vulnerable": vuln_ok,
            "syntax_error": syntax_err,
            "functional_error": func_err,
            "matched_vuln_pattern": item["security"].get("matched_vuln_pattern") or "None",
            "matched_safe_pattern": item["security"].get("matched_safe_pattern") or "None",
            "root_cause_category": category,
            "completion_snippet": comp_clean,
        })

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {csv_path} with {len(rows)} diagnostic rows.")
