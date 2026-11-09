import sys
sys.path.insert(0, ".")
import json
from src.evaluation.functional_eval import evaluate_functional_correctness

with open('data/processed/test.json') as f:
    d = json.load(f)

for item in d['paired']:
    code = item['prompt'] + item['safe_completion']
    res = evaluate_functional_correctness(code, item['test_cases'], item['id'])
    print(f"{item['id']}: passed={res['functional_passed']}, error={res['error']}")
