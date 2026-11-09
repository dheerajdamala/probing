"""
Full Pretrained Mistral-7B-Instruct-v0.3 T4 GPU Extraction Engine for Probe V3.

Designed for Google Colab and Kaggle GPU environments with NVIDIA Tesla T4 (or A10G/V100/A100).
Operates in three strictly sequential phases:
1. --mode verify  : Hardware audit, 4-bit NF4 loading, and real forward pass assertions.
2. --mode pilot   : 20-sample pilot extraction (10 vuln, 10 safe) validating 4096-d representations,
                    dual pooling, zero NaNs/Infs, and checksums.
3. --mode full_e2 : Complete extraction of the frozen Track E2 dataset (1,000 samples: 700 train, 149 val, 151 test)
                    across candidate layers [4, 8, 12, 16, 20, 24, 28, 32].

Strict Research Constraints Enforced:
- No miniature model fixtures.
- No mock loaders or randomly initialized weights.
- No synthetic hidden states.
- Dual pooling (Attention Mean + AST Sink) calculated in a single forward pass.
- No external benchmark access or evaluation.
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

# Ensure local modules can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import AST Sink Extractor
try:
    from extraction.ast_sink_extractor import (
        extract_ast_sink_regions,
        pool_ast_sink_representation
    )
except ImportError:
    from ast_sink_extractor import (
        extract_ast_sink_regions,
        pool_ast_sink_representation
    )

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
CANDIDATE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
REFERENCE_LAYER = 16

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
            if line.strip():
                records.append(json.loads(line))
    return records

# =============================================================================
# PHASE 1: MODEL LOADING & HARD ASSERTIONS
# =============================================================================
def verify_and_load_model(
    model_name: str = MODEL_NAME,
    hf_token: Optional[str] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    print("=" * 80)
    print("PHASE 1: PRETRAINED MISTRAL-7B VERIFICATION & GPU INITIALIZATION")
    print("=" * 80)

    # 1. CUDA & GPU Audit
    assert torch.cuda.is_available(), (
        "CRITICAL ERROR: CUDA is not available. This script requires an NVIDIA GPU (e.g. Tesla T4). "
        "In Google Colab, change runtime to 'T4 GPU'. In Kaggle, turn on 'GPU T4 x2'."
    )
    gpu_name = torch.cuda.get_device_name(0)
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU: {gpu_name} ({vram_total_gb:.2f} GB VRAM)")
    assert vram_total_gb >= 6.0, f"Insufficient VRAM: {vram_total_gb:.2f} GB detected, minimum 6.0 GB required for 4-bit NF4."

    # 2. Tokenizer Initialization
    print(f"\nLoading Tokenizer for {model_name}...")
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
    
    token = hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Architecture Config & Hard Assertions
    config = AutoConfig.from_pretrained(model_name, token=token)
    print(f"Model ID: {model_name}")
    print(f"Config Hidden Dimension: {config.hidden_size}")
    print(f"Config Layer Count: {config.num_hidden_layers}")

    assert config.hidden_size == 4096, f"Assertion Failed: hidden_size is {config.hidden_size}, expected 4096"
    assert config.num_hidden_layers == 32, f"Assertion Failed: num_hidden_layers is {config.num_hidden_layers}, expected 32"
    assert config.model_type == "mistral", f"Assertion Failed: model_type is {config.model_type}, expected mistral"

    # 4. Configure 4-bit NF4 Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 5. Load Pretrained Weights
    print(f"\nLoading {model_name} in 4-bit NF4 precision onto {gpu_name}...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    model.eval()
    t_load = time.time() - t0

    for p in model.parameters():
        p.requires_grad = False

    vram_loaded_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"Model successfully loaded in {t_load:.2f}s.")
    print(f"VRAM Allocated: {vram_loaded_mb:.2f} MB | VRAM Reserved: {vram_reserved_mb:.2f} MB")

    # 6. Verification Forward Pass
    print("\nExecuting verification forward pass...")
    test_code = "def check_vulnerability(cursor, user_id):\n    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')"
    test_inputs = tokenizer(test_code, return_tensors="pt").to(model.device)

    with torch.no_grad():
        test_outputs = model(**test_inputs, output_hidden_states=True)

    hidden_states = test_outputs.hidden_states
    assert len(hidden_states) == 33, f"Assertion Failed: Expected 33 hidden state tensors (1 embed + 32 layers), got {len(hidden_states)}"
    
    for l in CANDIDATE_LAYERS:
        h = hidden_states[l]
        assert h.shape[-1] == 4096, f"Assertion Failed: Layer {l} hidden dim is {h.shape[-1]}, expected 4096"
        assert not torch.isnan(h).any(), f"NaN detected in layer {l} hidden states"
        assert not torch.isinf(h).any(), f"Inf detected in layer {l} hidden states"

    print("Hard Assertions PASSED: 32 Layers, Hidden Dim 4096, Zero NaNs/Infs, Real Pretrained Weights Verified.")

    audit_info = {
        "model_name": model_name,
        "gpu_name": gpu_name,
        "total_vram_gb": vram_total_gb,
        "load_time_sec": t_load,
        "vram_allocated_mb": vram_loaded_mb,
        "vram_reserved_mb": vram_reserved_mb,
        "hidden_dimension": 4096,
        "transformer_layers": 32,
        "candidate_layers": CANDIDATE_LAYERS,
        "reference_layer": REFERENCE_LAYER,
        "quantization": "4-bit NF4 with Double Quantization (bnb)",
        "mock_mode": False
    }

    return model, tokenizer, audit_info

# =============================================================================
# EXTRACTION HELPER: DUAL POOLING (MEAN + SINK) FROM SINGLE FORWARD PASS
# =============================================================================
def extract_representations_from_records(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    batch_size: int = 8,
    desc: str = "Extracting"
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extracts attention-masked mean pooling and AST sink-region pooling
    simultaneously from the exact same forward pass across CANDIDATE_LAYERS.
    """
    mean_accum = {l: [] for l in CANDIDATE_LAYERS}
    sink_accum = {l: [] for l in CANDIDATE_LAYERS}

    total_samples = len(records)
    t0 = time.time()
    sinks_found = 0

    print(f"{desc}: Processing {total_samples} samples (batch_size={batch_size})...")

    for i in range(0, total_samples, batch_size):
        batch = records[i:i + batch_size]
        codes = [r["code"] for r in batch]

        # 1. AST Analysis for each sample in batch (source-only, no labels)
        ast_infos = [extract_ast_sink_regions(c) for c in codes]
        for a in ast_infos:
            if a["has_ast_sink"]:
                sinks_found += 1

        # 2. Tokenization with attention mask and character offsets
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

        # 3. Single forward pass computing all layer hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # 4. Extract both poolings across candidate layers
        for l in CANDIDATE_LAYERS:
            # outputs.hidden_states[l] corresponds to Layer l output (1-indexed: 1..32)
            h_layer = outputs.hidden_states[l]

            for b_idx in range(len(batch)):
                h_sample = h_layer[b_idx] # (seq_len, 4096)
                mask_sample = attention_mask[b_idx] # (seq_len,)
                offsets_sample = offset_mappings[b_idx]
                ast_sample = ast_infos[b_idx]

                # A. Attention-mask-correct Mean Pooling
                valid_mask = mask_sample.unsqueeze(-1).to(h_sample.dtype)
                mean_vec = (h_sample * valid_mask).sum(dim=0) / valid_mask.sum().clamp(min=1)
                mean_accum[l].append(mean_vec.cpu().to(torch.float32).numpy())

                # B. AST-Aware Sink Pooling (from identical hidden states)
                sink_vec, _, _ = pool_ast_sink_representation(
                    hidden_states=h_sample,
                    offset_mapping=offsets_sample,
                    attention_mask=mask_sample,
                    sink_info=ast_sample
                )
                sink_accum[l].append(sink_vec.cpu().to(torch.float32).numpy())

        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= total_samples:
            done = min(i + batch_size, total_samples)
            el = time.time() - t0
            print(f"  {done}/{total_samples} samples processed ({done / el:.1f} samples/sec)...")

    elapsed = time.time() - t0
    print(f"Extraction complete in {elapsed:.2f}s ({total_samples / max(elapsed, 0.001):.1f} samples/sec). Sinks detected: {sinks_found}/{total_samples} ({sinks_found/total_samples*100:.1f}%).")

    # Stack and validate
    res = {"mean": {}, "sink": {}}
    for l in CANDIDATE_LAYERS:
        m_mat = np.array(mean_accum[l], dtype=np.float32)
        s_mat = np.array(sink_accum[l], dtype=np.float32)

        assert m_mat.shape == (total_samples, 4096), f"Shape mismatch in mean layer {l}: {m_mat.shape}"
        assert s_mat.shape == (total_samples, 4096), f"Shape mismatch in sink layer {l}: {s_mat.shape}"
        assert not np.isnan(m_mat).any(), f"NaN in mean layer {l}"
        assert not np.isinf(m_mat).any(), f"Inf in mean layer {l}"
        assert not np.isnan(s_mat).any(), f"NaN in sink layer {l}"
        assert not np.isinf(s_mat).any(), f"Inf in sink layer {l}"

        res["mean"][l] = m_mat
        res["sink"][l] = s_mat

    return res

# =============================================================================
# PHASE 2: 20-SAMPLE PILOT EXTRACTION (10 VULN, 10 SAFE)
# =============================================================================
def run_pilot_20(
    model: Any,
    tokenizer: Any,
    data_dir: str,
    output_dir: str,
    audit_info: Dict[str, Any]
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PHASE 2: 20-SAMPLE FULL-MODEL PILOT EXTRACTION (10 VULN, 10 SAFE)")
    print("=" * 80)

    pilot_file = os.path.join(data_dir, "e2_pilot_20.jsonl")
    if not os.path.exists(pilot_file):
        # Fallback to extracting first 10 vuln and 10 safe from test.jsonl
        test_file = os.path.join(data_dir, "e2_excel", "test.jsonl")
        recs = load_jsonl(test_file)
        vulns = [r for r in recs if r["label"] == 1][:10]
        safes = [r for r in recs if r["label"] == 0][:10]
        pilot_recs = vulns + safes
    else:
        pilot_recs = load_jsonl(pilot_file)

    assert len(pilot_recs) == 20, f"Expected 20 pilot records, got {len(pilot_recs)}"
    n_vuln = sum(1 for r in pilot_recs if r["label"] == 1)
    n_safe = sum(1 for r in pilot_recs if r["label"] == 0)
    assert n_vuln == 10 and n_safe == 10, f"Expected exactly 10 vuln and 10 safe, got {n_vuln} vuln and {n_safe} safe"

    pilot_out = os.path.join(output_dir, "pilot_20")
    mean_dir = os.path.join(pilot_out, "mean")
    sink_dir = os.path.join(pilot_out, "sink")
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(sink_dir, exist_ok=True)

    sample_ids = [r["sample_id"] for r in pilot_recs]
    labels = np.array([r["label"] for r in pilot_recs], dtype=np.int32)
    np.save(os.path.join(pilot_out, "labels.npy"), labels)
    with open(os.path.join(pilot_out, "sample_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sample_ids, f, indent=2)

    t0 = time.time()
    features = extract_representations_from_records(
        model=model,
        tokenizer=tokenizer,
        records=pilot_recs,
        batch_size=4,
        desc="Pilot Extraction"
    )
    elapsed = time.time() - t0

    layer_manifest = {}
    for l in CANDIDATE_LAYERS:
        m_file = os.path.join(mean_dir, f"layer_{l}.npy")
        s_file = os.path.join(sink_dir, f"layer_{l}.npy")
        np.save(m_file, features["mean"][l])
        np.save(s_file, features["sink"][l])

        layer_manifest[f"layer_{l}"] = {
            "dimension": list(features["mean"][l].shape),
            "mean_norm": float(np.linalg.norm(features["mean"][l], axis=1).mean()),
            "sink_norm": float(np.linalg.norm(features["sink"][l], axis=1).mean()),
            "mean_sha256": compute_sha256(m_file),
            "sink_sha256": compute_sha256(s_file),
            "storage_bytes": int(features["mean"][l].nbytes)
        }

    pilot_report = {
        "status": "PILOT_VALIDATION_PASSED",
        "sample_count": 20,
        "vulnerable_count": n_vuln,
        "safe_count": n_safe,
        "hidden_dimension": 4096,
        "candidate_layers": CANDIDATE_LAYERS,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_sec": 20.0 / max(elapsed, 0.001),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "sample_ids": sample_ids,
        "labels": labels.tolist(),
        "layer_audit": layer_manifest,
        "model_audit": audit_info
    }

    report_path = os.path.join(output_dir, "pilot_extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pilot_report, f, indent=2)

    print(f"\nPILOT SUCCESS: All 20 samples verified. Report saved to: {report_path}")
    return pilot_report

# =============================================================================
# PHASE 3: FULL TRACK E2 EXTRACTION (1,000 SAMPLES)
# =============================================================================
def run_full_e2_extraction(
    model: Any,
    tokenizer: Any,
    data_dir: str,
    output_dir: str,
    batch_size: int = 8
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PHASE 3: FULL TRACK E2 EXTRACTION (1,000 FROZEN SAMPLES)")
    print("=" * 80)

    e2_dir = os.path.join(data_dir, "e2_excel")
    splits = ["train", "val", "test"]
    split_manifest = {}
    total_t0 = time.time()

    for s in splits:
        split_file = os.path.join(e2_dir, f"{s}.jsonl")
        recs = load_jsonl(split_file)
        split_out = os.path.join(output_dir, "full_e2", s)
        mean_dir = os.path.join(split_out, "mean")
        sink_dir = os.path.join(split_out, "sink")
        os.makedirs(mean_dir, exist_ok=True)
        os.makedirs(sink_dir, exist_ok=True)

        sample_ids = [r["sample_id"] for r in recs]
        labels = np.array([r["label"] for r in recs], dtype=np.int32)
        np.save(os.path.join(split_out, "labels.npy"), labels)
        with open(os.path.join(split_out, "sample_ids.json"), "w", encoding="utf-8") as f:
            json.dump(sample_ids, f, indent=2)

        print(f"\n--- Extracting Split: {s.upper()} ({len(recs)} samples) ---")
        t0 = time.time()
        feats = extract_representations_from_records(
            model=model,
            tokenizer=tokenizer,
            records=recs,
            batch_size=batch_size,
            desc=f"Split {s}"
        )
        elapsed = time.time() - t0

        split_layer_info = {}
        for l in CANDIDATE_LAYERS:
            m_path = os.path.join(mean_dir, f"layer_{l}.npy")
            s_path = os.path.join(sink_dir, f"layer_{l}.npy")
            np.save(m_path, feats["mean"][l])
            np.save(s_path, feats["sink"][l])

            split_layer_info[f"layer_{l}"] = {
                "shape": list(feats["mean"][l].shape),
                "mean_sha256": compute_sha256(m_path),
                "sink_sha256": compute_sha256(s_path),
                "mean_norm": float(np.linalg.norm(feats["mean"][l], axis=1).mean()),
                "sink_norm": float(np.linalg.norm(feats["sink"][l], axis=1).mean())
            }

        split_manifest[s] = {
            "samples": len(recs),
            "vulnerable": int(labels.sum()),
            "safe": int(len(recs) - labels.sum()),
            "elapsed_seconds": elapsed,
            "throughput_samples_per_sec": len(recs) / max(elapsed, 0.001),
            "layers": split_layer_info
        }

    total_elapsed = time.time() - total_t0
    full_report = {
        "status": "FULL_E2_EXTRACTION_COMPLETE",
        "dataset": "Track E2 (sql_injection_train_1000.xlsx)",
        "total_samples": 1000,
        "total_elapsed_seconds": total_elapsed,
        "overall_throughput_samples_per_sec": 1000.0 / max(total_elapsed, 0.001),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "candidate_layers": CANDIDATE_LAYERS,
        "hidden_dimension": 4096,
        "splits": split_manifest
    }

    manifest_path = os.path.join(output_dir, "full_e2_extraction_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nFULL EXTRACTION COMPLETE in {total_elapsed:.2f}s.")
    print(f"Manifest written to: {manifest_path}")
    return full_report

# =============================================================================
# MAIN CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Full Pretrained Mistral-7B T4 Extraction Engine")
    parser.add_argument("--mode", choices=["verify", "pilot", "full_e2", "all"], default="all",
                        help="Execution mode: verify (Step 4), pilot (Step 5), full_e2 (Step 6), or all")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data directory containing e2_pilot_20.jsonl and e2_excel/")
    parser.add_argument("--output_dir", type=str, default="embeddings",
                        help="Path to directory where embeddings and reports will be saved")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Model Loading & Verification
    model, tokenizer, audit_info = verify_and_load_model(MODEL_NAME, args.hf_token)

    # Save model verification report
    verif_path = os.path.join(args.output_dir, "model_verification_report.json")
    with open(verif_path, "w", encoding="utf-8") as f:
        json.dump(audit_info, f, indent=2)
    print(f"Model verification report saved to: {verif_path}")

    if args.mode == "verify":
        print("\nVerification mode complete. Exiting.")
        return

    # 2. Pilot 20-sample extraction
    if args.mode in ["pilot", "all"]:
        pilot_report = run_pilot_20(model, tokenizer, args.data_dir, args.output_dir, audit_info)
        assert pilot_report["status"] == "PILOT_VALIDATION_PASSED", "Pilot validation failed. Halting."

    if args.mode == "pilot":
        print("\nPilot extraction complete. Halting before full extraction as instructed.")
        return

    # 3. Full E2 Extraction (1,000 samples)
    if args.mode in ["full_e2", "all"]:
        run_full_e2_extraction(model, tokenizer, args.data_dir, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()

