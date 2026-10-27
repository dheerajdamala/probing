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
            effective_prompt = f"# Instruction: {secure_prompt_prefix}\n" + prompt
        else:
            effective_prompt = prompt

        cwe = item["cwe"]
        test_cases = item.get("test_cases", [])

        # Generate completion
        gen_res = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=effective_prompt,
            steering_vector=steering_vector,
            layer_idx=layer_idx,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            condition_name=condition_name,
        )

        full_code = effective_prompt + gen_res["completion"]

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
