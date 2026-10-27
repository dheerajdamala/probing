"""End-to-end experiment orchestration script.

Executes vector extraction, probe training, static activation steering,
and full evaluation benchmarking across all conditions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from src.data.download_datasets import download_or_generate_dataset
from src.data.prepare_dataset import prepare_splits
from src.data.validate_dataset import validate_splits
from src.evaluation.report_results import generate_markdown_report
from src.evaluation.run_benchmarks import run_full_benchmark_suite
from src.model.load_model import load_model_and_tokenizer
from src.probes.evaluate_semantic_probe import evaluate_test_split
from src.probes.train_semantic_probe import train_and_evaluate_layer_probes
from src.steering.evaluate_vectors import run_vector_evaluation
from src.steering.extract_security_vectors import extract_and_save_vectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    mock: bool = False,
    model_id: str = "wandb/pruned_mistral",
    device: str = "auto",
    target_layer: Optional[int] = None,
    alphas: Optional[list] = None,
    dry_run: bool = False,
) -> None:
    """Run full research pipeline from dataset verification through benchmark reporting."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATASET VERIFICATION & PREPARATION")
    logger.info("=" * 60)
    download_or_generate_dataset()
    prepare_splits()
    if not validate_splits():
        logger.error("Dataset validation failed! Aborting pipeline.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"STEP 2: MODEL LOADING (mock={mock}, model_id={model_id})")
    logger.info("=" * 60)
    model, tokenizer, meta = load_model_and_tokenizer(model_id=model_id, device=device, mock=mock)
    logger.info(f"Model ready: {meta['num_parameters']:,} parameters on {meta['device']}")

    num_layers = model.config.num_hidden_layers
    if target_layer is None:
        target_layer = min(2, num_layers - 1)
    sweep_layers = list(range(num_layers))

    logger.info("=" * 60)
    logger.info("STEP 3: SECURITY VECTOR EXTRACTION (STAGE E)")
    logger.info("=" * 60)
    vector_file = "outputs/security_vectors.pt"
    extract_and_save_vectors(
        model=model,
        tokenizer=tokenizer,
        train_file="data/processed/train.json",
        output_path=vector_file,
        target_layers=sweep_layers,
    )
    logger.info("Evaluating vector separation on validation split...")
    run_vector_evaluation(
        model=model,
        tokenizer=tokenizer,
        vector_file=vector_file,
        val_file="data/processed/val.json",
    )

    logger.info("=" * 60)
    logger.info("STEP 4: SEMANTIC PROBE TRAINING & EVALUATION (STAGE F)")
    logger.info("=" * 60)
    train_and_evaluate_layer_probes(
        model=model,
        tokenizer=tokenizer,
        train_file="data/processed/train.json",
        val_file="data/processed/val.json",
        layers=sweep_layers,
    )
    test_probe_file = f"outputs/probes/probe_layer_{target_layer}.pkl"
    evaluate_test_split(
        model=model,
        tokenizer=tokenizer,
        probe_file=test_probe_file,
        layer_idx=target_layer,
        test_file="data/processed/test.json",
    )

    if dry_run:
        logger.info("Dry-run requested: skipping generation benchmark.")
        return

    logger.info("=" * 60)
    logger.info("STEP 5: STATIC ACTIVATION STEERING BENCHMARKS (STAGE G & H)")
    logger.info("=" * 60)
    if alphas is None:
        alphas = [0.5, 1.0, 2.0, 3.0]

    benchmark_data = run_full_benchmark_suite(
        model=model,
        tokenizer=tokenizer,
        vector_bundle_file=vector_file,
        test_file="data/processed/test.json",
        probe_dir="outputs/probes",
        output_dir="outputs",
        layer_idx=target_layer,
        alphas=alphas,
        max_new_tokens=48,
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Report generated: outputs/BENCHMARK_REPORT.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ISAE Security Steering Pipeline")
    parser.add_argument("--mock", action="store_true", help="Use lightweight synthetic Mistral instance")
    parser.add_argument("--model-id", type=str, default="wandb/pruned_mistral", help="Hugging Face Model ID")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--target-layer", type=int, default=None, help="Target layer for steering")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction and probe without generation")
    args = parser.parse_args()

    run_pipeline(
        mock=args.mock,
        model_id=args.model_id,
        device=args.device,
        target_layer=args.target_layer,
        dry_run=args.dry_run,
    )
