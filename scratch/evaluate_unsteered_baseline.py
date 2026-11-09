import sys
sys.path.insert(0, ".")
import json
import time
import torch
from src.model.load_model import load_model_and_tokenizer
from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness
from src.evaluation.security_eval import evaluate_code_security
from src.evaluation.run_benchmarks import extract_clean_code_completion

# 1. Simple deterministic programming tasks
SIMPLE_TASKS = [
    {
        "id": "simple_add_integers",
        "name": "Add two integers",
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of two integers a and b.\"\"\"\n",
        "signature": "add(a: int, b: int) -> int",
        "test_cases": [
            {"args": [2, 3], "expected_result": 5},
            {"args": [-1, 1], "expected_result": 0},
            {"args": [10, -5], "expected_result": 5},
        ]
    },
    {
        "id": "simple_reverse_string",
        "name": "Reverse a string",
        "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed version of string s.\"\"\"\n",
        "signature": "reverse_string(s: str) -> str",
        "test_cases": [
            {"args": ["hello"], "expected_result": "olleh"},
            {"args": ["world"], "expected_result": "dlrow"},
            {"args": [""], "expected_result": ""},
        ]
    },
    {
        "id": "simple_max_element",
        "name": "Return maximum element in list",
        "prompt": "from typing import List\n\ndef max_element(numbers: List[int]) -> int:\n    \"\"\"Return the maximum element in non-empty list numbers.\"\"\"\n",
        "signature": "max_element(numbers: List[int]) -> int",
        "test_cases": [
            {"args": [[1, 5, 3, 9, 2]], "expected_result": 9},
            {"args": [[-10, -3, -20]], "expected_result": -3},
            {"args": [[42]], "expected_result": 42},
        ]
    },
    {
        "id": "simple_is_palindrome",
        "name": "Check if string is a palindrome",
        "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if string s is a palindrome, False otherwise.\"\"\"\n",
        "signature": "is_palindrome(s: str) -> bool",
        "test_cases": [
            {"args": ["racecar"], "expected_result": True},
            {"args": ["hello"], "expected_result": False},
            {"args": ["aba"], "expected_result": True},
            {"args": ["abc"], "expected_result": False},
        ]
    }
]

def run_unsteered_baseline_evaluation(model_id="axolotl-ai-co/tiny-mistral-25m"):
    print(f"Loading unsteered baseline model '{model_id}'...")
    start_load = time.time()
    model, tokenizer, meta = load_model_and_tokenizer(model_id=model_id, device="cpu")
    model.eval()
    load_time = time.time() - start_load
    print(f"Model ready in {load_time:.2f}s ({meta['num_parameters']:,} params).")

    results = {
        "model_id": model_id,
        "device": meta["device"],
        "num_parameters": meta["num_parameters"],
        "dtype": meta["dtype"],
        "simple_tasks": [],
        "security_tasks": [],
        "simple_task_pass_rate": 0.0,
        "security_functional_pass_rate": 0.0,
        "security_pass_rate": 0.0,
        "vulnerability_rate": 0.0,
        "syntax_validity_rate": 0.0,
    }

    # --- Part 1: Simple Tasks ---
    print("\n=== Evaluating Simple Deterministic Tasks ===")
    simple_passed = 0
    for task in SIMPLE_TASKS:
        prompt = task["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt")
        input_token_count = encoded["input_ids"].shape[1]
        
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        output_token_count = out.shape[1] - input_token_count
        raw_completion = tokenizer.decode(out[0][input_token_count:], skip_special_tokens=True)
        clean_completion = extract_clean_code_completion(prompt, raw_completion)
        full_code = prompt + clean_completion

        syn_valid, syn_err = check_syntax(full_code)
        fn_eval = evaluate_functional_correctness(full_code, task["test_cases"], task["id"])

        if fn_eval["functional_passed"]:
            simple_passed += 1

        record = {
            "id": task["id"],
            "name": task["name"],
            "signature": task["signature"],
            "prompt": prompt,
            "raw_completion": raw_completion,
            "clean_completion": clean_completion,
            "full_code": full_code,
            "input_tokens": int(input_token_count),
            "output_tokens": int(output_token_count),
            "latency_sec": round(elapsed, 4),
            "tokens_per_sec": round(output_token_count / max(elapsed, 0.001), 2),
            "syntax_valid": syn_valid,
            "syntax_error": syn_err,
            "functional_passed": fn_eval["functional_passed"],
            "tests_run": fn_eval["tests_run"],
            "tests_passed": fn_eval["tests_passed"],
            "functional_error": fn_eval.get("error"),
        }
        results["simple_tasks"].append(record)
        print(f"[{task['id']}] syn={syn_valid}, fn_passed={fn_eval['functional_passed']}, error={fn_eval.get('error') or syn_err}")

    results["simple_task_pass_rate"] = round(simple_passed / len(SIMPLE_TASKS), 4)

    # --- Part 2: Security Benchmark Tasks ---
    print("\n=== Evaluating Security Benchmark Prompts ===")
    with open("data/processed/test.json") as f:
        security_items = json.load(f)["paired"]

    sec_passed = 0
    vuln_count = 0
    fn_passed = 0
    syntax_passed = 0

    for item in security_items:
        prompt = item["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt")
        input_token_count = encoded["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        output_token_count = out.shape[1] - input_token_count
        raw_completion = tokenizer.decode(out[0][input_token_count:], skip_special_tokens=True)
        clean_completion = extract_clean_code_completion(prompt, raw_completion)
        full_code = prompt + clean_completion

        syn_valid, syn_err = check_syntax(full_code)
        fn_eval = evaluate_functional_correctness(full_code, item["test_cases"], item["id"])
        sec_eval = evaluate_code_security(full_code, item["cwe"])

        if syn_valid:
            syntax_passed += 1
        if fn_eval["functional_passed"]:
            fn_passed += 1
        if sec_eval["is_secure"]:
            sec_passed += 1
        if sec_eval["is_vulnerable"]:
            vuln_count += 1

        record = {
            "id": item["id"],
            "cwe": item["cwe"],
            "prompt": prompt,
            "raw_completion": raw_completion,
            "clean_completion": clean_completion,
            "full_code": full_code,
            "input_tokens": int(input_token_count),
            "output_tokens": int(output_token_count),
            "latency_sec": round(elapsed, 4),
            "tokens_per_sec": round(output_token_count / max(elapsed, 0.001), 2),
            "syntax_valid": syn_valid,
            "syntax_error": syn_err,
            "functional_passed": fn_eval["functional_passed"],
            "functional_error": fn_eval.get("error"),
            "is_secure": sec_eval["is_secure"],
            "is_vulnerable": sec_eval["is_vulnerable"],
            "matched_vuln_pattern": sec_eval["matched_vuln_pattern"],
            "matched_safe_pattern": sec_eval["matched_safe_pattern"],
        }
        results["security_tasks"].append(record)
        print(f"[{item['id']}] syn={syn_valid}, fn={fn_eval['functional_passed']}, sec={sec_eval['is_secure']}, vuln={sec_eval['is_vulnerable']}")

    n_sec = len(security_items)
    results["security_functional_pass_rate"] = round(fn_passed / n_sec, 4)
    results["security_pass_rate"] = round(sec_passed / n_sec, 4)
    results["vulnerability_rate"] = round(vuln_count / n_sec, 4)
    results["syntax_validity_rate"] = round(syntax_passed / n_sec, 4)

    # Save to outputs/baseline_results.json
    out_file = "outputs/baseline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved machine-readable baseline results to {out_file}")

    return results

if __name__ == "__main__":
    run_unsteered_baseline_evaluation()
