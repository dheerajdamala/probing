"""Benchmark runner executing multi-condition evaluation on held-out test splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.evaluation.functional_eval import evaluate_functional_correctness
from src.evaluation.report_results import generate_markdown_report
from src.evaluation.security_eval import evaluate_code_security
from src.evaluation.semantic_eval import evaluate_semantic_probe_on_completion
from src.steering.steered_generate import generate_with_steering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_clean_code_completion(prompt: str, raw_completion: str) -> str:
    """Robustly extract and clean code completion from model generation.
    
    Preserves relative indentation, strips markdown fences, conversational text,
    and handles both full function re-declarations and indented body completions.
    """
    code = raw_completion
    
    # 1. Extract from markdown fences if present
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0]
        
    # Strip only leading/trailing blank lines, preserving column 0 indentation of lines
    code = code.strip("\r\n")
    if not code.strip():
        return ""
        
    lines = code.splitlines()
    cleaned_lines = []
    seen_first_def = False
    
    for line in lines:
        stripped = line.strip()
        # Truncate at subsequent functions, classes, or main blocks
        if line and not line.startswith(" ") and not line.startswith("\t"):
            if stripped.startswith("if __name__"):
                break
            if stripped.startswith("def "):
                if seen_first_def:
                    break
                seen_first_def = True
            elif stripped.startswith("class "):
                if seen_first_def or cleaned_lines:
                    break
        elif stripped.startswith("def ") and not seen_first_def:
            seen_first_def = True
            
        cleaned_lines.append(line)
        
    cleaned = "\n".join(cleaned_lines).rstrip() + "\n"
    return cleaned


def assemble_full_code(prompt: str, completion: str) -> str:
    """Assemble executable full code from prompt and extracted completion."""
    comp = completion.strip()
    if not comp:
        return prompt.strip() + "\n"
        
    # If the completion already defines the function (starts with def or imports then def),
    # use the completion directly so we don't duplicate function headers
    if comp.startswith("def ") or ("\ndef " in comp and not comp.startswith("    ")):
        return comp + "\n"
        
    # If completion does not start with indentation, indent each non-empty line by 4 spaces
    lines = completion.splitlines()
    first_non_empty = next((l for l in lines if l.strip()), "")
    if first_non_empty and not first_non_empty.startswith(" ") and not first_non_empty.startswith("\t"):
        indented_lines = ["    " + l if l.strip() else l for l in lines]
        return prompt.rstrip() + "\n" + "\n".join(indented_lines) + "\n"
        
    # If completion already starts with indentation, append directly to prompt
    return prompt.rstrip() + "\n" + completion.lstrip("\r\n")


def resolve_evaluation_status(
    prompt: str,
    raw_comp: str,
    clean_comp: str,
    syn_valid: bool,
    syn_err: Optional[str],
    fn_eval: Dict[str, Any],
    sec_eval: Dict[str, Any],
    skipped: bool = False,
) -> Dict[str, Any]:
    """Resolve explicit evaluation status: PASS, FAIL, NOT_EVALUATED, EXECUTION_ERROR, MISSING_COMPLETION."""
    if skipped:
        return {
            "evaluation_status": "NOT_EVALUATED",
            "functional_passed": False,
            "is_secure": None,
            "is_vulnerable": None,
            "joint_passed": False,
            "error_reason": "Skipped or pending evaluation",
        }

    if not raw_comp or not raw_comp.strip() or not clean_comp or not clean_comp.strip():
        return {
            "evaluation_status": "MISSING_COMPLETION",
            "functional_passed": False,
            "is_secure": None,
            "is_vulnerable": None,
            "joint_passed": False,
            "error_reason": "Model produced empty or whitespace-only completion",
        }

    if not syn_valid:
        return {
            "evaluation_status": "EXECUTION_ERROR",
            "functional_passed": False,
            "is_secure": None,
            "is_vulnerable": None,
            "joint_passed": False,
            "error_reason": f"SyntaxError: {syn_err}",
        }

    fn_err = fn_eval.get("error")
    if fn_err and ("Exec error" in str(fn_err) or "ImportError" in str(fn_err)):
        return {
            "evaluation_status": "EXECUTION_ERROR",
            "functional_passed": False,
            "is_secure": None,
            "is_vulnerable": None,
            "joint_passed": False,
            "error_reason": fn_err,
        }

    fn_pass = fn_eval.get("functional_passed", False)
    is_sec = sec_eval.get("is_secure", False)
    is_vuln = sec_eval.get("is_vulnerable", False)

    if fn_pass and is_sec:
        return {
            "evaluation_status": "PASS",
            "functional_passed": True,
            "is_secure": True,
            "is_vulnerable": False,
            "joint_passed": True,
            "error_reason": None,
        }
    else:
        reasons = []
        if not fn_pass:
            reasons.append(fn_err or "Functional tests failed")
        if not is_sec:
            reasons.append("Vulnerable code pattern matched" if is_vuln else "Missing security guard")
        return {
            "evaluation_status": "FAIL",
            "functional_passed": fn_pass,
            "is_secure": is_sec,
            "is_vulnerable": is_vuln,
            "joint_passed": False,
            "error_reason": "; ".join(reasons),
        }


def run_condition_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_pairs: List[Dict[str, Any]],
    condition_name: str,
    steering_vector: Optional[torch.Tensor] = None,
    layer_idx: int = 2,
    alpha: float = 0.0,
    secure_prompt_prefix: Optional[str] = None,
    probe_file: Optional[Path | str] = None,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """Run evaluation for a single condition across all test items."""
    logger.info(f"Running benchmark condition: {condition_name} (alpha={alpha}, layer={layer_idx})...")

    results_per_item = []
    total_sec_passed = 0
    total_vulnerable = 0
    total_func_passed = 0
    total_syntax_valid = 0
    total_latencies = []
    total_tps = []

    for item in test_pairs:
        prompt = item["prompt"]
        if secure_prompt_prefix:
            model_prompt = f"# Instruction: {secure_prompt_prefix}\n" + prompt
        else:
            model_prompt = prompt

        cwe = item["cwe"]
        test_cases = item.get("test_cases", [])

        # Generate completion
        gen_res = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=model_prompt,
            steering_vector=steering_vector,
            layer_idx=layer_idx,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            condition_name=condition_name,
        )

        clean_completion = extract_clean_code_completion(prompt, gen_res["completion"])
        full_code = prompt + clean_completion

        # 1. Security Evaluation
        sec_res = evaluate_code_security(full_code, cwe)
        if sec_res["is_secure"]:
            total_sec_passed += 1
        if sec_res["is_vulnerable"]:
            total_vulnerable += 1

        # 2. Functional Evaluation
        func_res = evaluate_functional_correctness(full_code, test_cases)
        if func_res["syntax_valid"]:
            total_syntax_valid += 1
        if func_res["functional_passed"]:
            total_func_passed += 1

        # 3. Semantic Probe Evaluation
        sem_res = {}
        if probe_file and Path(probe_file).exists():
            sem_res = evaluate_semantic_probe_on_completion(
                model, tokenizer, full_code, probe_file=probe_file, layer_idx=layer_idx
            )

        total_latencies.append(gen_res["elapsed_sec"])
        total_tps.append(gen_res["tokens_per_sec"])

        results_per_item.append({
            "id": item["id"],
            "cwe": cwe,
            "completion": gen_res["completion"],
            "security": sec_res,
            "functional": func_res,
            "semantic": sem_res,
            "latency_sec": gen_res["elapsed_sec"],
            "tokens_per_sec": gen_res["tokens_per_sec"],
        })

    n = len(test_pairs)
    summary = {
        "condition": condition_name,
        "steered": (steering_vector is not None and alpha != 0.0),
        "alpha": alpha,
        "layer_idx": layer_idx,
        "total_samples": n,
        "security_pass_rate": round(total_sec_passed / n, 4) if n > 0 else 0.0,
        "vulnerability_rate": round(total_vulnerable / n, 4) if n > 0 else 0.0,
        "functional_pass_rate": round(total_func_passed / n, 4) if n > 0 else 0.0,
        "syntax_validity_rate": round(total_syntax_valid / n, 4) if n > 0 else 0.0,
        "mean_latency_sec": round(sum(total_latencies) / n, 4) if n > 0 else 0.0,
        "mean_tokens_per_sec": round(sum(total_tps) / n, 2) if n > 0 else 0.0,
        "per_item_results": results_per_item,
    }

    logger.info(
        f"Condition '{condition_name}' -> Sec Pass: {summary['security_pass_rate']:.1%}, "
        f"Vuln Rate: {summary['vulnerability_rate']:.1%}, Func Pass: {summary['functional_pass_rate']:.1%}, "
        f"Syntax: {summary['syntax_validity_rate']:.1%}"
    )
    return summary


def run_full_benchmark_suite(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    vector_bundle_file: Path | str = "outputs/security_vectors.pt",
    test_file: Path | str = "data/processed/test.json",
    probe_dir: Path | str = "outputs/probes",
    output_dir: Path | str = "outputs",
    layer_idx: int = 2,
    alphas: Optional[List[float]] = None,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """Execute complete benchmark suite across all conditions."""
    if alphas is None:
        alphas = [0.5, 1.0, 2.0, 3.0]

    with open(test_file, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)["paired"]

    # Load vectors
    v_bundle = torch.load(vector_bundle_file, weights_only=True)
    sec_vec = v_bundle["security_vectors"].get(layer_idx, None)
    rand_vec = v_bundle["random_vectors"].get(layer_idx, None)

    probe_file = Path(probe_dir) / f"probe_layer_{layer_idx}.pkl"

    conditions_results = {}

    # 1. Unsteered Baseline
    conditions_results["unsteered_baseline"] = run_condition_benchmark(
        model=model,
        tokenizer=tokenizer,
        test_pairs=test_pairs,
        condition_name="unsteered_baseline",
        steering_vector=None,
        layer_idx=layer_idx,
        alpha=0.0,
        probe_file=probe_file,
        max_new_tokens=max_new_tokens,
    )

    # 2. Secure Prompt Baseline
    conditions_results["secure_prompt_baseline"] = run_condition_benchmark(
        model=model,
        tokenizer=tokenizer,
        test_pairs=test_pairs,
        condition_name="secure_prompt_baseline",
        steering_vector=None,
        layer_idx=layer_idx,
        alpha=0.0,
        secure_prompt_prefix="Ensure secure programming practices, input validation, and no injection vulnerabilities.",
        probe_file=probe_file,
        max_new_tokens=max_new_tokens,
    )

    # 3. Random Vector Control
    if rand_vec is not None:
        conditions_results["random_vector_control_alpha_1.0"] = run_condition_benchmark(
            model=model,
            tokenizer=tokenizer,
            test_pairs=test_pairs,
            condition_name="random_vector_control_alpha_1.0",
            steering_vector=rand_vec,
            layer_idx=layer_idx,
            alpha=1.0,
            probe_file=probe_file,
            max_new_tokens=max_new_tokens,
        )

    # 4. Security Steered Conditions across Alphas
    if sec_vec is not None:
        for a in alphas:
            c_name = f"steered_security_alpha_{a}"
            conditions_results[c_name] = run_condition_benchmark(
                model=model,
                tokenizer=tokenizer,
                test_pairs=test_pairs,
                condition_name=c_name,
                steering_vector=sec_vec,
                layer_idx=layer_idx,
                alpha=a,
                probe_file=probe_file,
                max_new_tokens=max_new_tokens,
            )

    full_report_data = {
        "model_id": getattr(model.config, "_name_or_path", "Mistral"),
        "eval_split": "test",
        "num_prompts": len(test_pairs),
        "target_layer": layer_idx,
        "conditions": conditions_results,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_json = out_path / "benchmark_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)

    # Generate Markdown report
    generate_markdown_report(full_report_data, output_path=out_path / "BENCHMARK_REPORT.md")

    return full_report_data


if __name__ == "__main__":
    from src.model.load_model import load_model_and_tokenizer
    m, tok, _ = load_model_and_tokenizer(mock=True)
    run_full_benchmark_suite(m, tok, layer_idx=1, alphas=[1.0, 2.0])
