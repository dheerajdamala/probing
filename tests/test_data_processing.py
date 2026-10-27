"""Unit tests for dataset loading, preparation, and validation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.download_datasets import download_or_generate_dataset
from src.data.prepare_dataset import prepare_splits
from src.data.validate_dataset import validate_splits


def test_dataset_pipeline(tmp_path):
    """Test full data generation, splitting, and validation flow in temporary directory."""
    raw_dir = tmp_path / "raw"
    raw_file = download_or_generate_dataset(output_dir=raw_dir)
    assert raw_file.exists()

    with open(raw_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    assert len(corpus) > 0

    proc_dir = tmp_path / "processed"
    summary = prepare_splits(raw_file=raw_file, output_dir=proc_dir)
    assert summary["total_pairs"] == len(corpus)
    assert summary["train_pairs"] > 0
    assert summary["val_pairs"] > 0
    assert summary["test_pairs"] > 0

    # Validation
    is_valid = validate_splits(processed_dir=proc_dir)
    assert is_valid, "Dataset splits failed validation checks!"
