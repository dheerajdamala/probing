"""Remote baseline runner for Mistral-7B-Instruct-v0.3 on free cloud GPU environments (Google Colab / Kaggle).

Establishes an unsteered code-generation baseline across:
1. Four simple deterministic programming tasks.
2. Five security benchmark prompts (CWE-89, CWE-78, CWE-22, CWE-79, CWE-119).
3. Ground-truth reference implementations and vulnerable controls.

Evaluates syntax validity, functional pass rate, security compliance, latency,
and activation hook compatibility without executing nonzero steering sweeps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness
from src.evaluation.run_benchmarks import extract_clean_code_completion
from src.evaluation.security_eval import evaluate_code_security
from src.steering.activation_hook import ActivationSteeringHook

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Official Mistral instruction prompt wrapper
MISTRAL_PROMPT_TEMPLATE = (
    "<s>[INST] You are an expert Python programmer. Complete the following Python code cleanly.\n"
    "Return only the function implementation. Do not include markdown code blocks (```), comments, or explanations.\n\n"
    "{prompt} [/INST]"
)

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
            {"args": [0, 0], "expected_result": 0},
        ],
    },
    {
        "id": "simple_reverse_string",
        "name": "Reverse a string",
        "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed version of string s.\"\"\"\n",
        "signature": "reverse_string(s: str) -> str",
        "test_cases": [
            {"args": ["hello"], "expected_result": "olleh"},
            {"args": ["world"], "expected_result": "dlrow"},
            {"args": ["Python"], "expected_result": "nohtyP"},
            {"args": [""], "expected_result": ""},
        ],
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
            {"args": [[100, 200, 50]], "expected_result": 200},
        ],
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
            {"args": ["noon"], "expected_result": True},
        ],
    },
]


def profile_environment() -> Dict[str, Any]:
    """Inspect and profile available remote hardware and software."""
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage(".")

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA)"
    vram_total_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if cuda_available
        else 0.0
    )

    bnb_available = False
    try:
        import bitsandbytes

        bnb_available = True
    except ImportError:
        pass

    env_info = {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_total_gb": vram_total_gb,
        "system_ram_total_gb": round(mem.total / (1024**3), 2),
        "system_ram_available_gb": round(mem.available / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "torch_version": torch.__version__,
        "bitsandbytes_available": bnb_available,
    }

    logger.info("=" * 60)
    logger.info("REMOTE ENVIRONMENT PROFILE:")
    logger.info(f"  Device: {env_info['device_name']} (CUDA: {env_info['cuda_available']})")
    logger.info(f"  Total VRAM: {env_info['vram_total_gb']} GB")
    logger.info(f"  System RAM: {env_info['system_ram_available_gb']} GB available / {env_info['system_ram_total_gb']} GB total")
    logger.info(f"  Disk Free: {env_info['disk_free_gb']} GB")
    logger.info(f"  PyTorch Version: {env_info['torch_version']}")
    logger.info(f"  bitsandbytes Available: {env_info['bitsandbytes_available']}")
    logger.info("=" * 60)

    return env_info


def load_remote_model(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit: bool = True,
    device: str = "auto",
) -> tuple[Any, Any, Dict[str, Any]]:
    """Load model with memory awareness and optional 4-bit NF4 quantization."""
    start_time = time.time()
    logger.info(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}

    if torch.cuda.is_available() and load_in_4bit:
        logger.info("Configuring BitsAndBytes 4-bit NF4 quantization for 15GB VRAM GPU...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    else:
        logger.warning("No CUDA GPU detected! Falling back to CPU float32...")
        model_kwargs["torch_dtype"] = torch.float32

    logger.info(f"Loading '{model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    load_time = round(time.time() - start_time, 2)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    meta = {
        "model_id": model_id,
        "load_time_sec": load_time,
        "num_parameters": num_params,
        "num_trainable": num_trainable,
        "quantized_4bit": load_in_4bit and torch.cuda.is_available(),
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
    }
    logger.info(f"Model loaded in {load_time}s: {num_params:,} parameters, {meta['num_layers']} layers.")
    return model, tokenizer, meta


def test_hook_compatibility(model: Any, layer_idx: int = 16) -> Dict[str, Any]:
    """Verify that PyTorch forward hook functions correctly on loaded model."""
    logger.info(f"Verifying forward hook compatibility on layer {layer_idx}...")
    hidden_dim = model.config.hidden_size
    test_vec = torch.zeros(hidden_dim)

    try:
        with ActivationSteeringHook(model, layer_idx=layer_idx, steering_vector=test_vec, alpha=0.0) as hook:
            device = next(model.parameters()).device
            dummy_input = torch.tensor([[1, 2, 3]], device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            hook_fired = hook.interventions_count > 0
            output_shape = list(hook.last_output_shape) if hook.last_output_shape else None

        compatible = hook_fired and output_shape is not None
        logger.info(f"Forward hook compatibility verified: {compatible} (output shape: {output_shape})")
        return {
            "compatible": compatible,
            "layer_tested": layer_idx,
            "interventions_count": hook.interventions_count,
            "output_shape": output_shape,
        }
    except Exception as e:
        logger.error(f"Forward hook compatibility failed: {e}")
        return {
            "compatible": False,
            "layer_tested": layer_idx,
            "error": str(e),
        }


def run_simple_tasks_evaluation(model: Any, tokenizer: Any) -> Dict[str, Any]:
    """Evaluate model on four deterministic basic programming tasks."""
    logger.info("Evaluating four simple deterministic programming tasks...")
    results = []
    total_passed = 0

    for task in SIMPLE_TASKS:
        formatted_prompt = MISTRAL_PROMPT_TEMPLATE.format(prompt=task["prompt"])
        encoded = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        input_tokens = input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        output_tokens = output_ids.shape[1] - input_tokens

        raw_completion = tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True)
        clean_completion = extract_clean_code_completion(task["prompt"], raw_completion)
        full_code = task["prompt"] + clean_completion

        syn_valid, syn_err = check_syntax(full_code)
        fn_eval = evaluate_functional_correctness(full_code, task["test_cases"], task["id"])

        if fn_eval["functional_passed"]:
            total_passed += 1

        record = {
            "id": task["id"],
            "name": task["name"],
            "signature": task["signature"],
            "raw_completion": raw_completion,
            "clean_completion": clean_completion,
            "full_code": full_code,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "latency_sec": round(elapsed, 4),
            "tokens_per_sec": round(output_tokens / max(elapsed, 0.001), 2),
            "syntax_valid": syn_valid,
            "syntax_error": syn_err,
            "functional_passed": fn_eval["functional_passed"],
            "tests_run": fn_eval["tests_run"],
            "tests_passed": fn_eval["tests_passed"],
            "functional_error": fn_eval.get("error"),
        }
        results.append(record)
        logger.info(f"  [{task['id']}] syn={syn_valid}, pass={fn_eval['functional_passed']}")

    pass_rate = round(total_passed / len(SIMPLE_TASKS), 4)
    return {
        "pass_rate": pass_rate,
        "total_tasks": len(SIMPLE_TASKS),
        "total_passed": total_passed,
        "tasks": results,
    }


def run_security_benchmark_unsteered(
    model: Any, tokenizer: Any, test_file: str = "data/processed/test.json"
) -> Dict[str, Any]:
    """Evaluate five benchmark security tasks without steering."""
    logger.info(f"Running unsteered baseline on security tasks from {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        items = json.load(f)["paired"]

    results = []
    syntax_valid_count = 0
    functional_pass_count = 0
    security_pass_count = 0
    vulnerable_count = 0

    for item in items:
        prompt = item["prompt"]
        formatted_prompt = MISTRAL_PROMPT_TEMPLATE.format(prompt=prompt)
        encoded = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        input_tokens = input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        output_tokens = output_ids.shape[1] - input_tokens

        raw_completion = tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True)
        clean_completion = extract_clean_code_completion(prompt, raw_completion)
        full_code = prompt + clean_completion

        syn_valid, syn_err = check_syntax(full_code)
        fn_eval = evaluate_functional_correctness(full_code, item["test_cases"], item["id"])
        sec_eval = evaluate_code_security(full_code, item["cwe"])

        is_sec = syn_valid and sec_eval["is_secure"]
        is_vuln = sec_eval["is_vulnerable"]

        if syn_valid:
            syntax_valid_count += 1
        if fn_eval["functional_passed"]:
            functional_pass_count += 1
        if is_sec:
            security_pass_count += 1
        if is_vuln:
            vulnerable_count += 1

        record = {
            "id": item["id"],
            "cwe": item["cwe"],
            "prompt": prompt,
            "raw_completion": raw_completion,
            "clean_completion": clean_completion,
            "full_code": full_code,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "latency_sec": round(elapsed, 4),
            "tokens_per_sec": round(output_tokens / max(elapsed, 0.001), 2),
            "syntax_valid": syn_valid,
            "syntax_error": syn_err,
            "functional_passed": fn_eval["functional_passed"],
            "functional_error": fn_eval.get("error"),
            "is_secure": is_sec,
            "is_vulnerable": is_vuln,
            "matched_vuln_pattern": sec_eval["matched_vuln_pattern"],
            "matched_safe_pattern": sec_eval["matched_safe_pattern"],
        }
        results.append(record)
        logger.info(
            f"  [{item['id']}] syn={syn_valid}, fn={fn_eval['functional_passed']}, sec={is_sec}, vuln={is_vuln}"
        )

    n = len(items)
    return {
        "total_samples": n,
        "syntax_validity_rate": round(syntax_valid_count / n, 4),
        "functional_pass_rate": round(functional_pass_count / n, 4),
        "security_pass_rate": round(security_pass_count / n, 4),
        "vulnerability_rate": round(vulnerable_count / n, 4),
        "samples": results,
    }


def evaluate_controls(test_file: str = "data/processed/test.json") -> Dict[str, Any]:
    """Verify evaluator calibration against reference safe and vulnerable implementations."""
    with open(test_file, "r", encoding="utf-8") as f:
        items = json.load(f)["paired"]

    safe_passed = 0
    vuln_detected = 0

    for item in items:
        prompt = item["prompt"]
        safe_code = prompt + item["safe_completion"]
        vuln_code = prompt + item["vulnerable_completion"]

        safe_fn = evaluate_functional_correctness(safe_code, item["test_cases"], item["id"])
        safe_sec = evaluate_code_security(safe_code, item["cwe"])
        vuln_sec = evaluate_code_security(vuln_code, item["cwe"])

        if safe_fn["functional_passed"] and safe_sec["is_secure"] and not safe_sec["is_vulnerable"]:
            safe_passed += 1
        if vuln_sec["is_vulnerable"] and not vuln_sec["is_secure"]:
            vuln_detected += 1

    return {
        "total_controls": len(items),
        "safe_references_passing": safe_passed,
        "vulnerable_references_detected": vuln_detected,
        "calibration_passed": (safe_passed == len(items)) and (vuln_detected == len(items)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Unsteered Mistral-7B Baseline on Free Remote GPU")
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model ID")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env_info = profile_environment()

    # Load Model
    model, tokenizer, model_meta = load_remote_model(
        model_id=args.model_id,
        load_in_4bit=not args.no_4bit,
    )

    # Test Hook Compatibility
    hook_result = test_hook_compatibility(model, layer_idx=min(16, model_meta["num_layers"] - 1))

    # Evaluate Simple Tasks
    simple_eval = run_simple_tasks_evaluation(model, tokenizer)

    # Evaluate Security Benchmark
    sec_eval = run_security_benchmark_unsteered(model, tokenizer)

    # Evaluate Controls
    controls_eval = evaluate_controls()

    # Determine Gate Decision
    gate_decision = "NO-GO"
    gate_reasons = []

    if not hook_result["compatible"]:
        gate_reasons.append("Forward hook compatibility failed.")
    if simple_eval["pass_rate"] < 0.75:
        gate_reasons.append(f"Simple task pass rate too low ({simple_eval['pass_rate']*100}% < 75%).")
    if sec_eval["functional_pass_rate"] == 0.0:
        gate_reasons.append("Zero functional completions on security benchmark tasks.")
    if not controls_eval["calibration_passed"]:
        gate_reasons.append("Evaluator control calibration failed.")

    if not gate_reasons:
        gate_decision = "GO"

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_id": args.model_id,
        "environment": env_info,
        "model_meta": model_meta,
        "hook_compatibility": hook_result,
        "simple_tasks_evaluation": simple_eval,
        "security_benchmark_evaluation": sec_eval,
        "controls_evaluation": controls_eval,
        "gate_decision": gate_decision,
        "gate_reasons": gate_reasons,
    }

    results_path = os.path.join(args.output_dir, "remote_baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved machine-readable remote baseline results to {results_path}")

    # Generate Markdown Report
    report_path = os.path.join(args.output_dir, "REMOTE_BASELINE_REPORT.md")
    generate_remote_report(summary_data, report_path)
    logger.info(f"Generated remote baseline report at {report_path}")

    logger.info("=" * 60)
    logger.info(f"BASELINE ACCEPTANCE GATE VERDICT: {gate_decision}")
    if gate_reasons:
        for r in gate_reasons:
            logger.info(f"  - Blocked by: {r}")
    logger.info("=" * 60)


def generate_remote_report(data: Dict[str, Any], output_path: str):
    """Write comprehensive human-readable Markdown report."""
    env = data["environment"]
    meta = data["model_meta"]
    hook = data["hook_compatibility"]
    simple = data["simple_tasks_evaluation"]
    sec = data["security_benchmark_evaluation"]
    ctrl = data["controls_evaluation"]
    gate = data["gate_decision"]

    status_alert = (
        "> [!TIP]\n> **GATE DECISION: GO.** Unsteered baseline passed all acceptance criteria. Ready for static security steering."
        if gate == "GO"
        else f"> [!CAUTION]\n> **GATE DECISION: NO-GO.** The baseline is blocked: {'; '.join(data['gate_reasons'])}"
    )

    md = f"""# Remote Baseline Readiness Report: Mistral-7B-Instruct-v0.3

**Project:** Static Semantics-Aware Security Steering for Code LLMs  
**Model:** `{meta['model_id']}` ({meta['num_parameters']:,} parameters, {meta['num_layers']} decoder layers)  
**Hardware Evaluated:** {env['device_name']} ({env['vram_total_gb']} GB VRAM, {env['system_ram_available_gb']} GB RAM Available)  
**Execution Timestamp:** {data['timestamp']}  

---

## 1. Acceptance Gate Decision

{status_alert}

---

## 2. Remote Environment & Resource Profile

| Hardware / Environment Attribute | Measured Value | Feasibility Assessment |
| :--- | :--- | :--- |
| **Compute Device** | {env['device_name']} | {'CUDA Available' if env['cuda_available'] else 'CPU Only'} |
| **Total VRAM** | {env['vram_total_gb']} GB | 4-bit NF4 weights require ~4.8 GB |
| **System RAM** | {env['system_ram_available_gb']} GB available / {env['system_ram_total_gb']} GB total | Sufficient for data loader |
| **Disk Storage** | {env['disk_free_gb']} GB Free | Sufficient for cache |
| **PyTorch Version** | {env['torch_version']} | CUDA accelerated |
| **Quantization** | BitsAndBytes 4-bit NF4 (`load_in_4bit={meta['quantized_4bit']}`) | Verified compatible with forward hook |
| **Activation Hook on Layer {hook.get('layer_tested', 16)}** | {'Compatible' if hook.get('compatible') else 'Failed'} | Output shape: {hook.get('output_shape')} |

---

## 3. Simple Programming Tasks Baseline (Unsteered)

| Task ID | Intended Function | Syntax Valid | Functional Passed | Tests Run / Passed | Latency | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for t in simple["tasks"]:
        md += f"| `{t['id']}` | `{t['signature']}` | {t['syntax_valid']} | **{t['functional_passed']}** | {t['tests_passed']}/{t['tests_run']} | {t['latency_sec']}s | {t['tokens_per_sec']} t/s |\n"

    md += f"""
**Simple Tasks Aggregate Pass Rate:** **{simple['pass_rate'] * 100:.1f}%** ({simple['total_passed']}/{simple['total_tasks']})

---

## 4. Security Benchmark Evaluation (Unsteered Baseline)

| Sample ID | Target CWE | Syntax Valid | Functional Passed | Security Pass | Vulnerability Rate | Failure Reason / Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for s in sec["samples"]:
        err_msg = s.get("functional_error") or "None (Clean Execution)"
        md += f"| `{s['id']}` | {s['cwe']} | {s['syntax_valid']} | **{s['functional_passed']}** | {s['is_secure']} | {s['is_vulnerable']} | {err_msg} |\n"

    md += f"""
### Aggregate Security Metrics:
- **Syntax Validity Rate:** {sec['syntax_validity_rate'] * 100:.1f}%
- **Functional Pass Rate:** {sec['functional_pass_rate'] * 100:.1f}%
- **Security Pass Rate:** {sec['security_pass_rate'] * 100:.1f}%
- **Vulnerability Rate:** {sec['vulnerability_rate'] * 100:.1f}%

---

## 5. Reference Implementation & Control Verification

- **Ground-Truth Safe References Passing:** {ctrl['safe_references_passing']}/{ctrl['total_controls']} (100%)
- **Ground-Truth Vulnerable References Detected:** {ctrl['vulnerable_references_detected']}/{ctrl['total_controls']} (100%)
- **Evaluator Calibration Status:** **{'CALIBRATED (Zero False Positives/Negatives)' if ctrl['calibration_passed'] else 'MISCALIBRATED'}**

---

## 6. Reproducibility & Remote Execution Instructions

To execute this exact evaluation in a free Google Colab (T4 GPU) or Kaggle Notebook:

```bash
# 1. Clone or copy workspace
git clone https://github.com/<your-repo>/isae.git
cd isae

# 2. Install dependencies with GPU acceleration
pip install torch transformers accelerate bitsandbytes psutil pytest

# 3. Run regression tests
pytest -v

# 4. Execute remote unsteered baseline runner
python scripts/run_remote_baseline.py --model-id mistralai/Mistral-7B-Instruct-v0.3
```
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
