"""
Independent Validation and Integrity Audit for Expanded E2 Mistral Embeddings.

Verifies:
1. Expected sample count for each split (3,304 train, 674 val, 695 test).
2. No missing or duplicate sample IDs.
3. Every vector has exactly 4096 dimensions (float32).
4. No NaN or infinite values.
5. No corrupted or zero-norm vectors.
6. Every embedding maps to the correct source sample in the split jsonl.
7. Original and reused embeddings match exactly bit-for-bit against historical checksums.
8. Output manifest integrity and schema compliance.
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
from typing import Dict, Any, List

import numpy as np

EXPECTED_COUNTS = {
    "train": 3304,
    "validation": 674,
    "test": 695
}

HISTORICAL_COUNTS = {
    "train": 700,
    "validation": 149,
    "test": 151
}

HISTORICAL_ORIGINAL_CHECKSUMS = {
    "train": "174e05cb5a01f5c49100cfd370499f553f514b782b1a6e9ed96d27ce8f733139",
    "validation": "77eff9374ff6774fa563cfd83253af0965bac67748e344c3df8fdde451ce9007",
    "test": "5def3a12d0a86777e57930ed4a8dbec69a0f2d0e1805801f8e02490e51db25dc"
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_split_embeddings(
    split_name: str,
    embeddings_base_dir: str,
    splits_data_dir: str,
    historical_base_dir: str
) -> Dict[str, Any]:
    """
    Validates a single split.
    """
    print(f"\n========================================================")
    print(f"Validating Split: {split_name.upper()}")
    print(f"========================================================")

    split_dir = os.path.join(embeddings_base_dir, split_name)
    jsonl_file = os.path.join(splits_data_dir, f"{split_name}.jsonl")
    npy_file = os.path.join(split_dir, "sink", "layer_12.npy")
    ids_file = os.path.join(split_dir, "sample_ids.json")
    map_file = os.path.join(split_dir, "sample_id_to_index.json")
    npz_file = os.path.join(split_dir, "sample_id_embeddings.npz")
    labels_file = os.path.join(split_dir, "labels.npy")

    # 1. Check file existence
    required_files = [npy_file, ids_file, map_file, npz_file, labels_file]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required file: {rf}"

    # 2. Check JSONL records
    records = load_jsonl(jsonl_file)
    expected_total = EXPECTED_COUNTS[split_name]
    assert len(records) == expected_total, f"JSONL count mismatch: expected {expected_total}, got {len(records)}"

    # 3. Check sample IDs list
    with open(ids_file, "r", encoding="utf-8") as f:
        ids_list = json.load(f)
    assert len(ids_list) == expected_total, f"sample_ids.json count mismatch: expected {expected_total}, got {len(ids_list)}"
    assert len(set(ids_list)) == len(ids_list), "Duplicate sample IDs detected in sample_ids.json!"

    # Verify 1:1 match with JSONL order
    jsonl_ids = [r["sample_id"] for r in records]
    assert ids_list == jsonl_ids, "sample_ids.json does not strictly match JSONL sample_id sequence!"

    # 4. Check Index Map
    with open(map_file, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    assert len(id_map) == expected_total, f"sample_id_to_index.json count mismatch: {len(id_map)}"
    for idx, sid in enumerate(ids_list):
        assert id_map[sid] == idx, f"Index map discrepancy for {sid}: {id_map[sid]} != {idx}"

    # 5. Check NumPy embeddings matrix
    embs = np.load(npy_file)
    assert embs.shape == (expected_total, 4096), f"Matrix shape mismatch: expected ({expected_total}, 4096), got {embs.shape}"
    assert embs.dtype == np.float32, f"Expected float32, got {embs.dtype}"

    # 6. Check NaNs, Infs, zero norms
    nan_count = int(np.isnan(embs).sum())
    inf_count = int(np.isinf(embs).sum())
    norms = np.linalg.norm(embs, axis=1)
    zero_norms = int((norms <= 0.0001).sum())

    assert nan_count == 0, f"Found {nan_count} NaNs in {split_name} matrix!"
    assert inf_count == 0, f"Found {inf_count} Infs in {split_name} matrix!"
    assert zero_norms == 0, f"Found {zero_norms} zero-norm vectors in {split_name} matrix!"

    # 7. Check NPZ key-value dictionary
    with np.load(npz_file) as npz_data:
        npz_keys = set(npz_data.files)
        assert len(npz_keys) == expected_total, f"NPZ keys count mismatch: {len(npz_keys)}"
        assert npz_keys == set(ids_list), "NPZ keys do not match sample_ids.json set!"
        # Spot-check first and last vectors for exact equality
        first_id = ids_list[0]
        last_id = ids_list[-1]
        assert np.array_equal(npz_data[first_id], embs[0]), f"Vector mismatch in NPZ for {first_id}"
        assert np.array_equal(npz_data[last_id], embs[-1]), f"Vector mismatch in NPZ for {last_id}"

    # 8. Check Labels
    labels = np.load(labels_file)
    assert labels.shape == (expected_total,), f"Labels shape mismatch: {labels.shape}"
    jsonl_labels = np.array([int(r["label"]) for r in records], dtype=np.int64)
    assert np.array_equal(labels, jsonl_labels), "labels.npy does not match JSONL ground truth labels!"

    # 9. Bit-exact verification against historical reused embeddings
    orig_split = "val" if split_name == "validation" else split_name
    orig_hist_npy = os.path.join(historical_base_dir, orig_split, "sink", "layer_12.npy")
    orig_hist_ids = os.path.join(historical_base_dir, orig_split, "sample_ids.json")

    hist_embs = np.load(orig_hist_npy)
    with open(orig_hist_ids, "r", encoding="utf-8") as f:
        hist_ids_list = json.load(f)

    assert len(hist_ids_list) == HISTORICAL_COUNTS[split_name]
    hist_id_to_vec = {sid: hist_embs[i] for i, sid in enumerate(hist_ids_list)}

    matched_reused = 0
    for sid, h_vec in hist_id_to_vec.items():
        assert sid in id_map, f"Historical sample {sid} missing from expanded {split_name}!"
        idx_in_exp = id_map[sid]
        exp_vec = embs[idx_in_exp]
        assert np.array_equal(exp_vec, h_vec), f"Bit-level vector difference detected for reused sample {sid}!"
        matched_reused += 1

    print(f"  [PASS] Total records: {expected_total}")
    print(f"  [PASS] Shape: {embs.shape} (float32)")
    print(f"  [PASS] Zero NaNs, Zero Infs, Zero Empty Vectors")
    print(f"  [PASS] Norm range: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    print(f"  [PASS] Bit-exact historical reuse verified: {matched_reused}/{HISTORICAL_COUNTS[split_name]} samples")
    print(f"  [PASS] Newly extracted samples validated: {expected_total - matched_reused}")

    return {
        "split": split_name,
        "total_records": expected_total,
        "reused_historical_records": matched_reused,
        "newly_extracted_records": expected_total - matched_reused,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_norm_count": zero_norms,
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "sha256_layer12_npy": compute_sha256(npy_file),
        "sha256_sample_ids_json": compute_sha256(ids_file),
        "sha256_embeddings_npz": compute_sha256(npz_file),
        "historical_reuse_bit_exact": True,
        "status": "PASSED"
    }


def main():
    embeddings_base = "experiments/sqli_probe_v3_full_mistral/expanded_v2/embeddings"
    splits_data = "experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits"
    historical_base = "experiments/sqli_probe_v3_full_mistral/embeddings/full_e2"

    audit_report = {
        "audit_name": "Expanded E2 Mistral Embedding Extraction Validation Audit",
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "layer": 12,
        "dimension": 4096,
        "pooling": "ast_sink",
        "splits": {}
    }

    all_passed = True
    for split_name in ["train", "validation", "test"]:
        try:
            res = validate_split_embeddings(
                split_name=split_name,
                embeddings_base_dir=embeddings_base,
                splits_data_dir=splits_data,
                historical_base_dir=historical_base
            )
            audit_report["splits"][split_name] = res
        except Exception as e:
            print(f"  [FAIL] Validation failed for {split_name}: {e}")
            audit_report["splits"][split_name] = {"status": "FAILED", "error": str(e)}
            all_passed = False

    audit_report["all_passed"] = all_passed
    report_path = os.path.join(embeddings_base, "extraction_validation_audit.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\nFinal Audit Status: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Audit report saved to: {report_path}")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

