"""Results reporting and statistical aggregation module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_markdown_report(results_data: Dict[str, Any], output_path: Path | str = "outputs/BENCHMARK_REPORT.md") -> str:
    """Format experiment results into a clean markdown report."""
    conditions: Dict[str, Any] = results_data.get("conditions", {})

    lines = [
        "# Empirical Benchmark & Steering Evaluation Report",
        "",
        "**Project:** Static Semantics-Aware Security Steering for Code LLMs  ",
        f"**Model:** {results_data.get('model_id', 'Pruned Mistral')}  ",
        f"**Evaluation Split:** {results_data.get('eval_split', 'test')} ({results_data.get('num_prompts', 0)} prompts)  ",
        "",
        "---",
        "",
        "## 1. Primary Comparative Results",
        "",
        "| Experimental Condition | Security Pass Rate | Vulnerability Rate | Functional Pass Rate | Syntax Validity | Latency (sec) | Throughput (t/s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cond_name, stats in conditions.items():
        sec_pass = f"{stats.get('security_pass_rate', 0.0) * 100:.1f}%"
        vuln_rate = f"{stats.get('vulnerability_rate', 0.0) * 100:.1f}%"
        func_pass = f"{stats.get('functional_pass_rate', 0.0) * 100:.1f}%"
        syntax_val = f"{stats.get('syntax_validity_rate', 0.0) * 100:.1f}%"
        latency = f"{stats.get('mean_latency_sec', 0.0):.3f}"
        tps = f"{stats.get('mean_tokens_per_sec', 0.0):.1f}"

        lines.append(f"| **{cond_name}** | {sec_pass} | {vuln_rate} | {func_pass} | {syntax_val} | {latency} | {tps} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Key Findings & Observations",
        "",
        f"- **Vulnerability Suppression:** Comparing unsteered baseline against security-steered conditions demonstrates the impact of residual stream perturbations.",
        f"- **Functional Preservation:** Tracked whether steering preserved functional syntax or induced token breakdown at elevated steering strengths.",
        f"- **Control Robustness:** The random vector baseline controls for generic activation noise, confirming whether improvements are direction-specific.",
        "",
        "---",
        "",
        "## 3. Condition Details and Hyperparameters",
        "",
    ])

    for cond_name, stats in conditions.items():
        lines.append(f"### Condition: `{cond_name}`")
        lines.append(f"- **Steered:** {stats.get('steered', False)}")
        lines.append(f"- **Layer:** {stats.get('layer_idx', 'N/A')}")
        lines.append(f"- **Alpha ($\\alpha$):** {stats.get('alpha', 0.0)}")
        lines.append(f"- **Total Samples:** {stats.get('total_samples', 0)}")
        lines.append("")

    report_str = "\n".join(lines)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_str)

    logger.info(f"Generated benchmark report at {out_file}")
    return report_str


if __name__ == "__main__":
    dummy_data = {
        "model_id": "wandb/pruned_mistral",
        "eval_split": "test",
        "num_prompts": 5,
        "conditions": {
            "unsteered_baseline": {
                "security_pass_rate": 0.40,
                "vulnerability_rate": 0.60,
                "functional_pass_rate": 0.80,
                "syntax_validity_rate": 1.00,
                "mean_latency_sec": 0.45,
                "mean_tokens_per_sec": 65.2,
                "steered": False,
                "alpha": 0.0,
            },
            "steered_sec_alpha_1.5": {
                "security_pass_rate": 0.80,
                "vulnerability_rate": 0.20,
                "functional_pass_rate": 0.80,
                "syntax_validity_rate": 1.00,
                "mean_latency_sec": 0.46,
                "mean_tokens_per_sec": 64.8,
                "steered": True,
                "alpha": 1.5,
                "layer_idx": 2,
            }
        }
    }
    generate_markdown_report(dummy_data)
