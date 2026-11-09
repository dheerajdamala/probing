"""
Production Embedding Extraction Engine for Probe V2.

Extracts representations from frozen Mistral-7B-Instruct-v0.3:
- Candidate Layers: [4, 8, 12, 16, 20, 24, 28, 32] (Layer 16 as reference)
- Dual Pooling: Attention-mask-correct Mean Pooling AND AST-aware Sink-Region Pooling
  computed from the exact same hidden-state forward pass.
- Validates dimensions (N, 4096), zero NaNs, zero Infs, and strict sample ID alignment.
- Supports both GPU execution (4-bit NF4 for T4) and dry-run verification mode.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional

repo_root = r"C:\Users\dheer\Desktop\isae"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.sqli_probe_v2.extraction.ast_sink_extractor import (
    extract_ast_sink_regions,
    pool_ast_sink_representation
)

CANDIDATE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
REFERENCE_LAYER = 16

def load_split_records(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def extract_split_embeddings(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    output_dir: str,
    track_name: str,
    split_name: str,
    batch_size: int = 8,
    is_dry_run: bool = False
) -> Dict[str, Any]:
    mean_dir = os.path.join(output_dir, "mean")
    sink_dir = os.path.join(output_dir, "sink")
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(sink_dir, exist_ok=True)

    sample_ids = [r["sample_id"] for r in records]
    labels = np.array([int(r["label"]) for r in records], dtype=np.int32)
    
    # Save labels and sample IDs
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    with open(os.path.join(output_dir, "sample_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sample_ids, f, indent=2)

    hidden_dim = 4096 if not is_dry_run else 128
    actual_layers = CANDIDATE_LAYERS if not is_dry_run else [0, 1, 2, 3]

    layer_mean_accum = {l: [] for l in actual_layers}
    layer_sink_accum = {l: [] for l in actual_layers}
    
    sinks_detected_count = 0
    fallback_count = 0
    t0 = time.time()

    print(f"Extracting representations for {len(records)} samples ({track_name} / {split_name}) across {len(actual_layers)} layers...")

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        codes = [r["code"] for r in batch]
        
        # 1. Compute AST sink info for all samples in batch
        ast_infos = [extract_ast_sink_regions(c) for c in codes]
        for info in ast_infos:
            if info["has_ast_sink"]:
                sinks_detected_count += 1
            else:
                fallback_count += 1

        # 2. Tokenize with attention mask and offsets
        inputs = tokenizer(
            codes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        offset_mappings = inputs.pop("offset_mapping").cpu().tolist()
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"] # (batch_size, seq_len)

        # 3. Single forward pass computing all layer hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 4. Extract both pooling representations across all candidate layers
        for l in actual_layers:
            # Hugging Face index: index 0 is embeddings, layer l is index l + 1
            layer_idx = l + 1 if (l + 1) < len(outputs.hidden_states) else len(outputs.hidden_states) - 1
            h_layer = outputs.hidden_states[layer_idx] # (batch_size, seq_len, hidden_dim)

            for b_idx in range(len(batch)):
                h_sample = h_layer[b_idx] # (seq_len, hidden_dim)
                mask_sample = attention_mask[b_idx] # (seq_len,)
                offsets_sample = offset_mappings[b_idx] # list of (start, end)
                ast_sample = ast_infos[b_idx]

                # A. Attention-mask-correct Mean Pooling
                valid_mask = mask_sample.unsqueeze(-1).to(h_sample.dtype) # (seq_len, 1)
                mean_vec = (h_sample * valid_mask).sum(dim=0) / valid_mask.sum().clamp(min=1)
                layer_mean_accum[l].append(mean_vec.cpu().to(torch.float32).numpy())

                # B. AST-Aware Sink Pooling (from identical hidden states)
                sink_vec, _, _ = pool_ast_sink_representation(
                    hidden_states=h_sample,
                    offset_mapping=offsets_sample,
                    attention_mask=mask_sample,
                    sink_info=ast_sample
                )
                layer_sink_accum[l].append(sink_vec.cpu().to(torch.float32).numpy())

    elapsed = max(time.time() - t0, 0.001)
    throughput = len(records) / elapsed
    print(f"Extraction complete in {elapsed:.2f}s ({throughput:.1f} samples/sec).")

    # 5. Validate integrity & Save per layer
    layer_audit = {}
    for l in actual_layers:
        mean_matrix = np.array(layer_mean_accum[l], dtype=np.float32)
        sink_matrix = np.array(layer_sink_accum[l], dtype=np.float32)

        # Verify shapes
        assert mean_matrix.shape == (len(records), hidden_dim), f"Shape mismatch in mean layer {l}: {mean_matrix.shape}"
        assert sink_matrix.shape == (len(records), hidden_dim), f"Shape mismatch in sink layer {l}: {sink_matrix.shape}"

        # Verify no NaNs or Infs
        assert not np.isnan(mean_matrix).any(), f"NaN detected in mean matrix layer {l}"
        assert not np.isinf(mean_matrix).any(), f"Inf detected in mean matrix layer {l}"
        assert not np.isnan(sink_matrix).any(), f"NaN detected in sink matrix layer {l}"
        assert not np.isinf(sink_matrix).any(), f"Inf detected in sink matrix layer {l}"

        # Save matrices
        mean_file = os.path.join(mean_dir, f"layer_{l}.npy")
        sink_file = os.path.join(sink_dir, f"layer_{l}.npy")
        np.save(mean_file, mean_matrix)
        np.save(sink_file, sink_matrix)

        layer_audit[f"layer_{l}"] = {
            "mean_shape": list(mean_matrix.shape),
            "sink_shape": list(sink_matrix.shape),
            "mean_norm": float(np.linalg.norm(mean_matrix, axis=1).mean()),
            "sink_norm": float(np.linalg.norm(sink_matrix, axis=1).mean()),
            "mean_storage_bytes": int(mean_matrix.nbytes),
            "sink_storage_bytes": int(sink_matrix.nbytes),
            "nan_count": 0,
            "inf_count": 0
        }

    report = {
        "track": track_name,
        "split": split_name,
        "sample_count": len(records),
        "hidden_dimension": hidden_dim,
        "candidate_layers": actual_layers,
        "reference_layer": REFERENCE_LAYER if not is_dry_run else 3,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_sec": throughput,
        "sinks_detected": sinks_detected_count,
        "fallbacks_used": fallback_count,
        "sink_coverage_pct": (sinks_detected_count / len(records)) * 100 if len(records) > 0 else 0,
        "layer_audit": layer_audit
    }

    report_path = os.path.join(output_dir, "extraction_integrity_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Integrity report saved to: {report_path}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Probe V2 Embedding Extraction Engine")
    parser.add_argument("--track", choices=["e1_existing", "e2_excel", "e3_combined", "pilot"], default="pilot")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    v2_base = os.path.join(repo_root, "experiments", "sqli_probe_v2")
    splits_base = os.path.join(v2_base, "datasets", "splits")
    embed_base = os.path.join(v2_base, "embeddings")

    # Detect execution environment
    has_cuda = torch.cuda.is_available()
    print(f"Probe V2 Extraction Engine | CUDA available: {has_cuda}")

    if not has_cuda:
        print("Running in CPU Dry-Run / Validation Fixture mode using miniature model...")
        import importlib.util
        load_model_path = os.path.join(repo_root, "src", "model", "load_model.py")
        spec = importlib.util.spec_from_file_location("root_load_model", load_model_path)
        root_load_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_load_model)
        model, tokenizer = root_load_model.create_miniature_mistral(num_layers=4, hidden_size=128)
        model.eval()
        is_dry_run = True
    else:
        print(f"Loading {args.model_name} in 4-bit NF4 on GPU...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        model.eval()
        is_dry_run = False

    track = args.track
    if track == "pilot":
        # Run small validation pilot on test split
        split_file = os.path.join(splits_base, "e2_excel", "test.jsonl")
        recs = load_split_records(split_file)[:25]
        out_d = os.path.join(embed_base, "pilot_validation")
        extract_split_embeddings(model, tokenizer, recs, out_d, "pilot_validation", "test_pilot", args.batch_size, is_dry_run)
    else:
        splits = [args.split] if args.split != "all" else ["train", "val", "test"]
        for s in splits:
            split_file = os.path.join(splits_base, track, f"{s}.jsonl")
            recs = load_split_records(split_file)
            out_d = os.path.join(embed_base, track, s)
            extract_split_embeddings(model, tokenizer, recs, out_d, track, s, args.batch_size, is_dry_run)

if __name__ == "__main__":
    main()

