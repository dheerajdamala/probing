import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from src.evaluation.functional_eval import evaluate_functional_correctness

with open("data/processed/test.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)["paired"]

item = pairs[0]
full_safe_code = item["prompt"] + item["safe_completion"]
print("Code:\n", full_safe_code)
print("Test cases:", item["test_cases"])

# Let's run exec directly to see what exception gets raised
from src.evaluation.functional_eval import create_safe_sandbox
safe_globals, local_scope = create_safe_sandbox()
exec(full_safe_code, safe_globals, local_scope)
print("Local scope keys:", local_scope.keys())
fn = local_scope.get("copy_subsequence")
try:
    ret = fn([10, 20, 30, 40], 1, 2)
    print("Return value:", ret)
except Exception as e:
    import traceback
    traceback.print_exc()

res = evaluate_functional_correctness(full_safe_code, item["test_cases"])
print("Eval result:", res)
