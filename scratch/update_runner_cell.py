%%writefile scripts/run_large_scale_evaluation.py
"""Large-Scale Static Security Steering Evaluation Runner (3,500 Tasks / 14,000 Executions).

Executes genuine model generation across 4 core conditions:
1. unsteered_baseline (alpha = 0.0)
2. hooked_alpha_0 (alpha = 0.0)
3. learned_vector_layer16 (alpha = 1.0)
4. random_vector_layer16 (alpha = 1.0, seed = 42)

Features:
- Checkpointing & idempotent resumption (skips already evaluated (condition, task_id) records).
- Throughput and latency tracking (tokens/sec, sec/task, dynamic ETA).
- Strict non-simulation policy: Calls model.generate() directly; never substitutes canned fixtures.
- Evaluates with patched security evaluator (preventing CWE-22 sibling traversal false positives).
- Tracks completed, failed, skipped, and pending tasks separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# Ensure repository root is on sys.path before importing internal packages
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness
from src.evaluation.run_benchmarks import assemble_full_code, extract_clean_code_completion, resolve_evaluation_status
from src.evaluation.security_eval import evaluate_code_security
from src.steering.activation_hook import ActivationSteeringHook
from src.steering.extract_security_vectors import compute_security_vector_from_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_tokenizer_4bit(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3") -> Tuple[Any, Any]:
    """Load model with 4-bit NF4 quantization for 15GB T4 GPU."""
    logger.info(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Configuring BitsAndBytes 4-bit NF4 quantization for GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer

MISTRAL_PROMPT_TEMPLATE = (
    "<s>[INST] You are an expert Python programmer. Complete the following Python code cleanly.\n"
    "Return only the function implementation. Do not include markdown code blocks (```), comments, or explanations.\n\n"
    "{prompt} [/INST]"
)


def load_existing_checkpoint(checkpoint_csv: str) -> Tuple[Set[Tuple[str, str]], List[Dict[str, Any]]]:
    """Load existing evaluated task records for idempotent resumption."""
    completed_pairs: Set[Tuple[str, str]] = set()
    existing_records: List[Dict[str, Any]] = []

    if os.path.exists(checkpoint_csv):
        with open(checkpoint_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            for row in reader:
                completed_pairs.add((row["condition"], row["task_id"]))
                existing_records.append(row)
        logger.info(f"Loaded {len(existing_records)} existing records from checkpoint {checkpoint_csv}.")
    return completed_pairs, existing_records


def append_record_to_csv(csv_path: str, record: Dict[str, Any], write_header: bool = False) -> None:
    """Append a single evaluated record to CSV for real-time persistence."""
    fieldnames = [
        "condition", "condition_type", "layer", "alpha", "task_id", "cwe",
        "evaluation_status", "syntax_valid", "syntax_error", "functional_passed", "functional_error",
        "is_secure", "is_vulnerable", "joint_passed", "matched_pattern",
        "latency_sec", "input_tokens", "output_tokens", "tokens_per_sec",
        "execution_status", "raw_completion", "completion"
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def run_evaluation_suite(
    model: Any,
    tokenizer: Any,
    steering_vector: torch.Tensor,
    random_vector: torch.Tensor,
    tasks: List[Dict[str, Any]],
    output_dir: str = "outputs",
    checkpoint_file: str = "large_scale_evaluation_per_sample.csv",
    layer_idx: int = 16,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """Execute evaluation across all conditions with checkpointing and throughput profiling."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_csv = os.path.join(output_dir, checkpoint_file)

    completed_pairs, existing_records = load_existing_checkpoint(checkpoint_csv)
    write_header = not os.path.exists(checkpoint_csv)

    conditions = [
        {"name": "unsteered_baseline", "type": "baseline", "alpha": 0.0, "vector": None},
        {"name": "hooked_alpha_0", "type": "control_alpha_0", "alpha": 0.0, "vector": steering_vector},
        {"name": "learned_vector_layer16", "type": "learned_steering", "alpha": 1.0, "vector": steering_vector},
        {"name": "random_vector_layer16", "type": "random_control", "alpha": 1.0, "vector": random_vector},
    ]

    total_tasks = len(tasks)
    total_planned_executions = total_tasks * len(conditions)
    executed_count = 0
    skipped_count = len(completed_pairs)
    failed_count = 0

    latencies: List[float] = []
    throughputs: List[float] = []

    logger.info(
        f"Target Benchmark: {total_tasks} tasks across {len(conditions)} conditions "
        f"({total_planned_executions} total executions). Already completed: {skipped_count}."
    )

    t_suite_start = time.time()

    for cond in conditions:
        c_name = cond["name"]
        c_type = cond["type"]
        c_alpha = cond["alpha"]
        c_vec = cond["vector"]

        logger.info(f"--- Evaluating Condition: {c_name} (alpha={c_alpha}) ---")

        for idx, item in enumerate(tasks):
            t_id = item["id"]
            if (c_name, t_id) in completed_pairs:
                continue

            prompt = item["prompt"]
            formatted_prompt = MISTRAL_PROMPT_TEMPLATE.format(prompt=prompt)
            encoded = tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            input_tokens = input_ids.shape[1]

            t0 = time.time()
            try:
                if c_alpha == 0.0 or c_vec is None:
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=96,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                else:
                    with ActivationSteeringHook(model, layer_idx=layer_idx, steering_vector=c_vec, alpha=c_alpha):
                        with torch.no_grad():
                            output_ids = model.generate(
                                input_ids=input_ids,
                                max_new_tokens=96,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                elapsed = time.time() - t0
                output_tokens = output_ids.shape[1] - input_tokens
                tokens_sec = round(output_tokens / max(elapsed, 1e-4), 2)
                raw_comp = tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True)
                clean_comp = extract_clean_code_completion(prompt, raw_comp)
                exec_status = "SUCCESS"
            except Exception as e:
                elapsed = time.time() - t0
                output_tokens = 0
                tokens_sec = 0.0
                clean_comp = ""
                exec_status = f"FAILED: {type(e).__name__}: {str(e)}"
                failed_count += 1
                logger.error(f"Inference failed for {t_id} under {c_name}: {e}")

            full_code = assemble_full_code(prompt, clean_comp)
            syn_valid, syn_err = check_syntax(full_code)
            fn_eval = evaluate_functional_correctness(full_code, item.get("test_cases", []), item["id"])
            sec_eval = evaluate_code_security(full_code, item["cwe"])

            status_res = resolve_evaluation_status(
                prompt=prompt,
                raw_comp=raw_comp,
                clean_comp=clean_comp,
                syn_valid=syn_valid,
                syn_err=syn_err,
                fn_eval=fn_eval,
                sec_eval=sec_eval,
                skipped=False,
            )

            fn_pass = status_res["functional_passed"]
            is_sec = status_res["is_secure"]
            is_vuln = status_res["is_vulnerable"]
            joint_pass = status_res["joint_passed"]
            eval_status = status_res["evaluation_status"]

            matched_pat = sec_eval.get("matched_safe_pattern") or sec_eval.get("matched_vuln_pattern")

            record = {
                "condition": c_name,
                "condition_type": c_type,
                "layer": layer_idx if c_vec is not None else None,
                "alpha": c_alpha,
                "task_id": t_id,
                "cwe": item["cwe"],
                "evaluation_status": eval_status,
                "syntax_valid": syn_valid,
                "syntax_error": syn_err,
                "functional_passed": fn_pass,
                "functional_error": fn_eval.get("error"),
                "is_secure": is_sec,
                "is_vulnerable": is_vuln,
                "joint_passed": joint_pass,
                "matched_pattern": matched_pat,
                "latency_sec": round(elapsed, 4),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "tokens_per_sec": tokens_sec,
                "execution_status": exec_status,
                "raw_completion": raw_comp,
                "completion": clean_comp
            }

            append_record_to_csv(checkpoint_csv, record, write_header=write_header)
            write_header = False
            completed_pairs.add((c_name, t_id))
            executed_count += 1
            latencies.append(elapsed)
            throughputs.append(tokens_sec)

            if executed_count % batch_size == 0:
                avg_lat = sum(latencies[-batch_size:]) / batch_size
                avg_tok_sec = sum(throughputs[-batch_size:]) / batch_size
                rem_tasks = total_planned_executions - len(completed_pairs)
                eta_minutes = round((rem_tasks * avg_lat) / 60, 1)
                logger.info(
                    f"[{c_name}] Completed {len(completed_pairs)}/{total_planned_executions} "
                    f"({round(len(completed_pairs)/total_planned_executions*100, 1)}%). "
                    f"Throughput: {avg_tok_sec:.1f} tok/s ({avg_lat:.3f} s/task). ETA: {eta_minutes} min."
                )

    total_time = round(time.time() - t_suite_start, 2)
    logger.info(f"Suite completed in {total_time}s. Total records saved: {len(completed_pairs)}.")
    return {
        "total_planned": total_planned_executions,
        "completed": len(completed_pairs),
        "executed_this_run": executed_count,
        "failed": failed_count,
        "total_time_sec": total_time,
        "checkpoint_file": checkpoint_csv
    }


from src.steering.extract_security_vectors import compute_security_vector_from_pairs


def main():
    parser = argparse.ArgumentParser(description="Large-Scale 3,500 Task Steering Evaluator")
    parser.add_argument("--tasks-file", type=str, default="data/processed/expanded_evaluation_3k.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--preflight", type=int, default=None, help="Run a small preflight subset (e.g. 20-50 tasks)")
    args = parser.parse_args()

    # Hardware check
    if not torch.cuda.is_available():
        logger.error(
            "HARDWARE GATE ERROR: CUDA GPU is not available on this host. "
            "Mistral-7B-Instruct-v0.3 in 4-bit NF4 requires CUDA bitsandbytes kernels. "
            "In accordance with project policy: Stopping at preflight stage. "
            "Report the blocker explicitly; do not substitute mock outputs."
        )
        sys.exit(1)

    logger.info("CUDA GPU detected. Initializing benchmark run.")

    # 1. Load evaluation tasks
    with open(args.tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data

    if args.preflight is not None:
        tasks = tasks[:args.preflight]
        logger.info(f"PREFLIGHT MODE ACTIVE: Running {len(tasks)} tasks across 4 conditions ({len(tasks)*4} generations).")
    else:
        logger.info(f"FULL BENCHMARK MODE ACTIVE: Loaded {len(tasks)} tasks.")

    # 2. Load model and tokenizer
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    logger.info(f"Loading {model_id} in 4-bit NF4...")
    model, tokenizer = load_model_and_tokenizer_4bit(model_id=model_id)

    # 3. Compute static security vector at Layer 16
    layer_idx = 16
    with open("data/processed/train.json", "r", encoding="utf-8") as f:
        train_pairs = json.load(f)["paired"]

    logger.info(f"Computing static security vector at Layer {layer_idx} from {len(train_pairs)} training pairs...")
    vector_bundle = compute_security_vector_from_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=train_pairs,
        target_layers=[layer_idx],
        method="mean_diff",
        pooling="last_token",
        normalize=True,
    )
    learned_sec_vector = vector_bundle["security_vectors"][layer_idx]
    random_vector = vector_bundle["random_vectors"][layer_idx]

    logger.info(f"Security vector Layer {layer_idx} norm: {learned_sec_vector.norm().item():.4f}")
    logger.info(f"Random vector Layer {layer_idx} norm: {random_vector.norm().item():.4f}")

    # 4. Execute evaluation suite
    summary = run_evaluation_suite(
        model=model,
        tokenizer=tokenizer,
        steering_vector=learned_sec_vector,
        random_vector=random_vector,
        tasks=tasks,
        output_dir=args.output_dir,
        layer_idx=layer_idx,
    )

    logger.info("=== Execution Summary ===")
    logger.info(f"Completed: {summary['completed']}/{summary['total_planned']} generations.")
    logger.info(f"Total Wall-Clock Time: {summary['total_time_sec']}s")
    logger.info(f"Checkpoint File: {summary['checkpoint_file']}")


if __name__ == "__main__":
    main()
