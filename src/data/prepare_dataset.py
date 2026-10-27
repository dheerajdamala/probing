"""Dataset preparation and train/val/test splitting.

Splits paired examples into train, validation, and test partitions
with leakage prevention across task variants, and formats data for both
contrastive steering vector extraction and semantic probe supervision.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_splits(
    raw_file: Path | str = "data/raw/isae_paired_corpus.json",
    output_dir: Path | str = "data/processed",
    config_file: Path | str = "configs/dataset.yaml",
) -> Dict[str, Any]:
    """Partition paired dataset into train, val, and test splits."""
    raw_path = Path(raw_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "r", encoding="utf-8") as f:
        corpus: List[Dict[str, Any]] = json.load(f)

    cfg = {}
    if Path(config_file).exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("dataset", {})

    seed = cfg.get("random_seed", 42)
    train_ratio = cfg.get("splits", {}).get("train_ratio", 0.60)
    val_ratio = cfg.get("splits", {}).get("val_ratio", 0.20)

    # Group by CWE to ensure stratified distribution across splits
    cwe_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in corpus:
        cwe = item["cwe"]
        cwe_groups.setdefault(cwe, []).append(item)

    train_set: List[Dict[str, Any]] = []
    val_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    rng = random.Random(seed)

    for cwe, items in sorted(cwe_groups.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n and n > 2:
            n_train = n - 2
            n_val = 1

        train_part = shuffled[:n_train]
        val_part = shuffled[n_train : n_train + n_val]
        test_part = shuffled[n_train + n_val :]

        # Guarantee at least 1 test item per CWE if possible
        if not test_part and len(val_part) > 1:
            test_part = [val_part.pop()]
        elif not test_part and len(train_part) > 1:
            test_part = [train_part.pop()]

        train_set.extend(train_part)
        val_set.extend(val_part)
        test_set.extend(test_part)

    def to_probe_records(paired_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten paired examples into individual code samples for probe training."""
        records = []
        for item in paired_list:
            records.append({
                "id": f"{item['id']}_safe",
                "parent_id": item["id"],
                "cwe": item["cwe"],
                "code": item["prompt"] + item["safe_completion"],
                "prompt": item["prompt"],
                "completion": item["safe_completion"],
                "semantic_label": item["semantic_label_safe"],
                "semantic_target": item["semantic_target"],
                "is_safe": True
            })
            records.append({
                "id": f"{item['id']}_vuln",
                "parent_id": item["id"],
                "cwe": item["cwe"],
                "code": item["prompt"] + item["vulnerable_completion"],
                "prompt": item["prompt"],
                "completion": item["vulnerable_completion"],
                "semantic_label": item["semantic_label_vuln"],
                "semantic_target": item["semantic_target"],
                "is_safe": False
            })
        return records

    splits_data = {
        "train": {
            "paired": train_set,
            "probe_records": to_probe_records(train_set)
        },
        "val": {
            "paired": val_set,
            "probe_records": to_probe_records(val_set)
        },
        "test": {
            "paired": test_set,
            "probe_records": to_probe_records(test_set)
        }
    }

    for split_name, content in splits_data.items():
        split_file = out_dir / f"{split_name}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        logger.info(
            f"Saved {split_name} split to {split_file} "
            f"({len(content['paired'])} pairs, {len(content['probe_records'])} probe records)"
        )

    summary = {
        "total_pairs": len(corpus),
        "total_probe_samples": len(corpus) * 2,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "test_pairs": len(test_set),
        "cwes": sorted(list(cwe_groups.keys())),
        "seed": seed
    }

    summary_file = out_dir / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    prepare_splits()
