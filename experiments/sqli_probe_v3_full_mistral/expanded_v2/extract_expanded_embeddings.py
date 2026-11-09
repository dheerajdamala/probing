"""
Expanded E2 Mistral Embedding Extraction Engine for Probe V3.

Extracts genuine mistralai/Mistral-7B-Instruct-v0.3 Layer 12 AST sink-pooled representations
(d=4096) for the expanded E2 dataset (N=4,673: 3,304 train, 674 val, 695 test).

Methodology strictly preserves Probe V3 Phase 1-3 specifications:
- Model: mistralai/Mistral-7B-Instruct-v0.3
- Quantization: 4-bit NF4 (bitsandbytes)
- Target Layer: Layer 12 (1-indexed hidden states)
- Pooling: AST SQL sink region character-to-token projection with fallback to attention-mean
- Output Dimension: 4,096
- Checksum & ID Auditing: Every vector is explicitly mapped to sample_id
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

# Ensure parent and extraction directories are in path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
workspace_dir = os.path.abspath(os.path.join(parent_dir, "..", ".."))

for p in [script_dir, parent_dir, workspace_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from extraction.ast_sink_extractor import (
        extract_ast_sink_regions,
        pool_ast_sink_representation
    )
except ImportError:
    from experiments.sqli_probe_v3_full_mistral.extraction.ast_sink_extractor import (
        extract_ast_sink_regions,
        pool_ast_sink_representation
    )

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYER = 12
EMBEDDING_DIM = 4096


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


def verify_and_load_model(device: str = "cuda") -> Tuple[Any, Any]:
    """
    Loads Mistral-7B-Instruct-v0.3 in 4-bit NF4.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {MODEL_NAME} with 4-bit NF4 on {device}...")
    if device == "cuda" and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        print("WARNING: CUDA not available or CPU requested. Loading in float32 / bfloat16 on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    model.eval()
    print("Model and tokenizer successfully loaded.")
    return model, tokenizer


def extract_batch_representations(
    model: Any,
    tokenizer: Any,
    batch_records: List[Dict[str, Any]],
    target_layer: int = TARGET_LAYER
) -> List[Tuple[str, np.ndarray, int]]:
    """
    Extracts Layer 12 AST sink representations for a batch of records.
    Returns list of (sample_id, vector_4096, label).
    """
    codes = [r["code"] for r in batch_records]
    sample_ids = [r["sample_id"] for r in batch_records]
    labels = [int(r["label"]) for r in batch_records]

    # 1. AST Analysis for each sample
    ast_infos = [extract_ast_sink_regions(c) for c in codes]

    # 2. Tokenization with offsets
    inputs = tokenizer(
        codes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    offset_mappings = inputs.pop("offset_mapping").cpu().tolist()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    attention_mask = inputs["attention_mask"]

    # 3. Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 4. Hidden state at target_layer
    h_layer = outputs.hidden_states[target_layer]

    results = []
    for b_idx in range(len(batch_records)):
        h_sample = h_layer[b_idx]
        mask_sample = attention_mask[b_idx]
        offsets_sample = offset_mappings[b_idx]
        ast_sample = ast_infos[b_idx]

        sink_vec, _, _ = pool_ast_sink_representation(
            hidden_states=h_sample,
            offset_mapping=offsets_sample,
            attention_mask=mask_sample,
            sink_info=ast_sample
        )
        vec_np = sink_vec.cpu().to(torch.float32).numpy()

        assert vec_np.shape == (EMBEDDING_DIM,), f"Vector dimension mismatch: {vec_np.shape}"
        assert not np.isnan(vec_np).any(), f"NaN in vector for {sample_ids[b_idx]}"
        assert not np.isinf(vec_np).any(), f"Inf in vector for {sample_ids[b_idx]}"

        results.append((sample_ids[b_idx], vec_np, labels[b_idx]))

    return results


def run_extraction(
    records: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    out_npz_path: str,
    batch_size: int = 8,
    save_interval: int = 50
) -> Dict[str, np.ndarray]:
    """
    Extracts embeddings for records with incremental saving.
    """
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    embeddings_dict: Dict[str, np.ndarray] = {}

    # Check if partial exists to resume
    if os.path.exists(out_npz_path):
        print(f"Loading existing embeddings from {out_npz_path} to resume...")
        with np.load(out_npz_path) as data:
            for k in data.files:
                embeddings_dict[k] = data[k]
        print(f"Loaded {len(embeddings_dict)} already extracted vectors.")

    unextracted = [r for r in records if r["sample_id"] not in embeddings_dict]
    print(f"Total to extract: {len(unextracted)} / {len(records)} records.")

    if not unextracted:
        print("All records already extracted!")
        return embeddings_dict

    t0 = time.time()
    for i in range(0, len(unextracted), batch_size):
        batch = unextracted[i:i + batch_size]
        batch_results = extract_batch_representations(model, tokenizer, batch)

        for sid, vec, _ in batch_results:
            embeddings_dict[sid] = vec

        done = min(i + batch_size, len(unextracted))
        if done % (batch_size * 10) == 0 or done == len(unextracted):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 0.001)
            eta = (len(unextracted) - done) / max(rate, 0.001) / 60
            print(f"  [{done}/{len(unextracted)}] ({done/len(unextracted)*100:.1f}%) "
                  f"- Rate: {rate:.1f} samples/s - Elapsed: {elapsed/60:.1f}m - ETA: {eta:.1f}m")

        # Save checkpoint periodically
        if (i // batch_size + 1) % save_interval == 0 or done == len(unextracted):
            np.savez_compressed(out_npz_path, **embeddings_dict)

    np.savez_compressed(out_npz_path, **embeddings_dict)
    print(f"Saved {len(embeddings_dict)} vectors to {out_npz_path}")
    return embeddings_dict


def merge_and_build_split(
    split_name: str,
    split_jsonl_path: str,
    historical_npz_path: str,
    new_extracted_npz_path: str,
    out_dir: str
) -> Dict[str, Any]:
    """
    Merges historical and newly extracted embeddings to construct the complete
    split dataset in out_dir/{split_name}/.
    """
    target_split_dir = os.path.join(out_dir, split_name)
    os.makedirs(os.path.join(target_split_dir, "sink"), exist_ok=True)

    # 1. Load historical embeddings
    hist_dict = {}
    if os.path.exists(historical_npz_path):
        with np.load(historical_npz_path) as h_data:
            for k in h_data.files:
                hist_dict[k] = h_data[k]
    print(f"[{split_name}] Historical vectors available: {len(hist_dict)}")

    # 2. Load new extracted embeddings
    new_dict = {}
    if os.path.exists(new_extracted_npz_path):
        with np.load(new_extracted_npz_path) as n_data:
            for k in n_data.files:
                new_dict[k] = n_data[k]
    print(f"[{split_name}] Newly extracted vectors available: {len(new_dict)}")

    # 3. Read official split records line by line
    records = load_jsonl(split_jsonl_path)
    total_records = len(records)
    print(f"[{split_name}] Total records in split manifest: {total_records}")

    ordered_sample_ids = []
    ordered_labels = []
    ordered_vectors = []
    id_to_index = {}
    merged_dict = {}

    reused_count = 0
    new_count = 0
    missing_ids = []

    for idx, rec in enumerate(records):
        sid = rec["sample_id"]
        label = int(rec["label"])
        ordered_sample_ids.append(sid)
        ordered_labels.append(label)
        id_to_index[sid] = idx

        if sid in hist_dict:
            vec = hist_dict[sid]
            reused_count += 1
        elif sid in new_dict:
            vec = new_dict[sid]
            new_count += 1
        else:
            missing_ids.append(sid)
            continue

        ordered_vectors.append(vec)
        merged_dict[sid] = vec

    if missing_ids:
        print(f"ERROR: [{split_name}] Missing embeddings for {len(missing_ids)} samples!")
        print(f"First 10 missing: {missing_ids[:10]}")
        raise ValueError(f"Missing {len(missing_ids)} embeddings in {split_name}")

    vectors_matrix = np.array(ordered_vectors, dtype=np.float32)
    labels_array = np.array(ordered_labels, dtype=np.int64)

    # Assertions
    assert vectors_matrix.shape == (total_records, EMBEDDING_DIM), f"Matrix shape mismatch: {vectors_matrix.shape}"
    assert labels_array.shape == (total_records,), f"Labels shape mismatch: {labels_array.shape}"
    assert not np.isnan(vectors_matrix).any(), "NaN in merged matrix"
    assert not np.isinf(vectors_matrix).any(), "Inf in merged matrix"

    # Save output artifacts
    npy_path = os.path.join(target_split_dir, "sink", f"layer_{TARGET_LAYER}.npy")
    ids_path = os.path.join(target_split_dir, "sample_ids.json")
    map_path = os.path.join(target_split_dir, "sample_id_to_index.json")
    npz_path = os.path.join(target_split_dir, "sample_id_embeddings.npz")
    labels_path = os.path.join(target_split_dir, "labels.npy")

    np.save(npy_path, vectors_matrix)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ordered_sample_ids, f, indent=2)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(id_to_index, f, indent=2)
    np.savez_compressed(npz_path, **merged_dict)
    np.save(labels_path, labels_array)

    stats = {
        "split": split_name,
        "total_samples": total_records,
        "reused_historical": reused_count,
        "newly_extracted": new_count,
        "missing_count": len(missing_ids),
        "vector_shape": list(vectors_matrix.shape),
        "sha256_layer12_npy": compute_sha256(npy_path),
        "sha256_sample_ids_json": compute_sha256(ids_path),
        "sha256_embeddings_npz": compute_sha256(npz_path),
        "min_norm": float(np.linalg.norm(vectors_matrix, axis=1).min()),
        "max_norm": float(np.linalg.norm(vectors_matrix, axis=1).max()),
        "mean_norm": float(np.linalg.norm(vectors_matrix, axis=1).mean()),
        "vulnerable_count": int((labels_array == 1).sum()),
        "safe_count": int((labels_array == 0).sum())
    }

    print(f"[{split_name}] Completed merge: {total_records} samples ({reused_count} reused, {new_count} new).")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Expanded E2 Mistral Embedding Extraction Engine")
    parser.add_argument("--mode", choices=["verify", "extract", "merge", "all"], default="all",
                        help="Execution mode: verify, extract, merge, or all")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all",
                        help="Target split")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for forward pass")
    parser.add_argument("--data_dir", default="experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits",
                        help="Directory containing expanded split jsonl files")
    parser.add_argument("--output_dir", default="experiments/sqli_probe_v3_full_mistral/expanded_v2/embeddings",
                        help="Output directory for embeddings")
    args = parser.parse_args()

    split_configs = [
        {
            "name": "train",
            "full_jsonl": os.path.join(args.data_dir, "train.jsonl"),
            "unextracted_jsonl": os.path.join(args.data_dir, "unextracted_train_2604.jsonl"),
            "hist_npz": os.path.join(args.output_dir, "historical", "train_historical_700.npz"),
            "new_npz": os.path.join(args.output_dir, "new_extracted", "train_new_2604.npz"),
            "expected_total": 3304,
            "expected_hist": 700,
            "expected_new": 2604
        },
        {
            "name": "validation",
            "full_jsonl": os.path.join(args.data_dir, "validation.jsonl"),
            "unextracted_jsonl": os.path.join(args.data_dir, "unextracted_val_525.jsonl"),
            "hist_npz": os.path.join(args.output_dir, "historical", "validation_historical_149.npz"),
            "new_npz": os.path.join(args.output_dir, "new_extracted", "validation_new_525.npz"),
            "expected_total": 674,
            "expected_hist": 149,
            "expected_new": 525
        },
        {
            "name": "test",
            "full_jsonl": os.path.join(args.data_dir, "test.jsonl"),
            "unextracted_jsonl": os.path.join(args.data_dir, "unextracted_test_544.jsonl"),
            "hist_npz": os.path.join(args.output_dir, "historical", "test_historical_151.npz"),
            "new_npz": os.path.join(args.output_dir, "new_extracted", "test_new_544.npz"),
            "expected_total": 695,
            "expected_hist": 151,
            "expected_new": 544
        }
    ]

    selected_splits = split_configs if args.split == "all" else [c for c in split_configs if c["name"] == args.split]

    model, tokenizer = None, None
    if args.mode in ["verify", "extract", "all"]:
        model, tokenizer = verify_and_load_model(device=args.device)

        if args.mode == "verify":
            print("\n--- Running Single-Sample Verification ---")
            sample_code = "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')"
            test_rec = [{"sample_id": "test_verify_001", "code": sample_code, "label": 1}]
            res = extract_batch_representations(model, tokenizer, test_rec)
            sid, vec, lab = res[0]
            print(f"Verified sample {sid}: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
            print("Verification passed successfully.")
            return

    if args.mode in ["extract", "all"]:
        print("\n--- Starting Representation Extraction ---")
        for sc in selected_splits:
            print(f"\nProcessing Split: {sc['name']}")
            records = load_jsonl(sc["unextracted_jsonl"])
            run_extraction(records, model, tokenizer, sc["new_npz"], batch_size=args.batch_size)

    if args.mode in ["merge", "all"]:
        print("\n--- Merging Historical and New Embeddings ---")
        summary_manifest = {
            "model": MODEL_NAME,
            "layer": TARGET_LAYER,
            "dimension": EMBEDDING_DIM,
            "splits": {}
        }
        for sc in selected_splits:
            stats = merge_and_build_split(
                split_name=sc["name"],
                split_jsonl_path=sc["full_jsonl"],
                historical_npz_path=sc["hist_npz"],
                new_extracted_npz_path=sc["new_npz"],
                out_dir=args.output_dir
            )
            summary_manifest["splits"][sc["name"]] = stats

        summary_path = os.path.join(args.output_dir, "expanded_v2_extraction_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_manifest, f, indent=2)
        print(f"\nSummary manifest saved to: {summary_path}")


if __name__ == "__main__":
    main()

