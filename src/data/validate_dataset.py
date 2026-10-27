"""Dataset validation and integrity checking script.

Verifies schema compliance, prevents data leakage between splits,
validates syntax, and ensures balance in contrastive pairs.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_splits(processed_dir: Path | str = "data/processed") -> bool:
    """Run comprehensive validation on train, val, and test splits."""
    proc_path = Path(processed_dir)
    splits = ["train", "val", "test"]

    seen_pair_ids: Dict[str, str] = {}
    seen_prompts: Dict[str, str] = {}
    all_passed = True

    for split in splits:
        split_file = proc_path / f"{split}.json"
        if not split_file.exists():
            logger.error(f"Missing split file: {split_file}")
            return False

        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paired: List[Dict[str, Any]] = data.get("paired", [])
        probe_records: List[Dict[str, Any]] = data.get("probe_records", [])

        logger.info(f"Validating {split}: {len(paired)} pairs, {len(probe_records)} probe records")

        if len(paired) == 0:
            logger.error(f"Split {split} has 0 pairs!")
            all_passed = False

        if len(probe_records) != len(paired) * 2:
            logger.error(f"Probe records ({len(probe_records)}) != 2 * pairs ({len(paired)}) in {split}")
            all_passed = False

        for pair in paired:
            pair_id = pair.get("id")
            if not pair_id:
                logger.error("Found pair without 'id'")
                all_passed = False
                continue

            # Leakage check: no pair ID may appear in multiple splits
            if pair_id in seen_pair_ids:
                logger.error(f"Data leakage! Pair {pair_id} in {split} was already in {seen_pair_ids[pair_id]}")
                all_passed = False
            seen_pair_ids[pair_id] = split

            # Leakage check: prompt uniqueness across splits
            prompt = pair.get("prompt", "").strip()
            if prompt in seen_prompts and seen_prompts[prompt] != split:
                logger.error(f"Prompt leakage across splits: '{prompt[:30]}...' in {split} and {seen_prompts[prompt]}")
                all_passed = False
            seen_prompts[prompt] = split

            # Syntax validation of code compositions
            safe_code = pair["prompt"] + pair["safe_completion"]
            vuln_code = pair["prompt"] + pair["vulnerable_completion"]

            try:
                ast.parse(safe_code)
            except SyntaxError as e:
                logger.error(f"Syntax error in safe code for {pair_id}: {e}")
                all_passed = False

            try:
                ast.parse(vuln_code)
            except SyntaxError as e:
                logger.error(f"Syntax error in vulnerable code for {pair_id}: {e}")
                all_passed = False

            # Semantic label check
            if pair.get("semantic_label_safe") != 0 or pair.get("semantic_label_vuln") != 1:
                logger.error(f"Invalid semantic labels in pair {pair_id}")
                all_passed = False

    if all_passed:
        logger.info("Dataset validation passed! Zero leakage and 100% valid Python syntax verified.")
    else:
        logger.error("Dataset validation failed with errors.")

    return all_passed


if __name__ == "__main__":
    import sys
    success = validate_splits()
    sys.exit(0 if success else 1)
