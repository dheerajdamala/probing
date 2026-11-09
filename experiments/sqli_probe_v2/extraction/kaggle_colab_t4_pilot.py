"""
Full Pretrained Mistral-7B-Instruct-v0.3 T4 GPU Extraction Pilot & Benchmark Script.

Designed for Google Colab or Kaggle Notebooks with NVIDIA T4 GPU (15–16 GB VRAM).
Measures:
1. Exact GPU memory allocation and reservation (VRAM) in 4-bit NF4.
2. Inference latency and throughput (samples/sec, tokens/sec) across batch sizes (1, 4, 8).
3. Embedding matrix dimensions (N, 4096) across candidate layers [4, 8, 12, 16, 20, 24, 28, 32].
4. Projected full-run runtime and disk storage requirements for Tracks E1, E2, E3, and E_ext.
"""

import os
import sys
import time
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Candidate layers for 32-layer Mistral-7B
CANDIDATE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
REFERENCE_LAYER = 16

def run_t4_pilot(model_name="mistralai/Mistral-7B-Instruct-v0.3", batch_sizes=[1, 4, 8]):
    print(f"=== Starting Pretrained Mistral-7B Pilot on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ===")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This script is intended to run on a T4 GPU node.")
        print("Switch to a Google Colab T4 or Kaggle GPU runtime to execute full extraction.")
        return

    # 1. Measure initial VRAM baseline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_start_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # 2. Configure 4-bit NF4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Loading {model_name} in 4-bit NF4...")
    t_load_0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    t_load = time.time() - t_load_0
    
    vram_model_loaded_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    vram_model_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    print(f"Model loaded in {t_load:.2f}s.")
    print(f"VRAM Allocated: {vram_model_loaded_mb:.2f} MB | VRAM Reserved: {vram_model_reserved_mb:.2f} MB")
    
    # Representative 20-sample pilot batch
    pilot_codes = [
        "def query_user(conn, uid):\n    cur = conn.cursor()\n    cur.execute(f'SELECT * FROM users WHERE id = {uid}')\n    return cur.fetchone()",
        "def query_user_safe(conn, uid):\n    cur = conn.cursor()\n    cur.execute('SELECT * FROM users WHERE id = %s', (uid,))\n    return cur.fetchone()",
        "def delete_item(conn, name):\n    query = 'DELETE FROM items WHERE name = \"' + name + '\"'\n    conn.cursor().execute(query)",
        "def get_orders(session, cid):\n    return session.query(Order).filter(Order.customer_id == cid).all()",
        "from typing import List\nimport os\ndef list_dir(p: str) -> List[str]:\n    return os.listdir(p)"
    ] * 4  # 20 samples total
    
    benchmark_results = {
        "model_name": model_name,
        "device": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
        "model_load_time_sec": t_load,
        "vram_model_allocated_mb": vram_model_loaded_mb,
        "vram_model_reserved_mb": vram_model_reserved_mb,
        "batch_benchmarks": {}
    }
    
    # 3. Benchmark batch sizes
    for bs in batch_sizes:
        print(f"\nBenchmarking batch size {bs}...")
        torch.cuda.reset_peak_memory_stats()
        
        batches = [pilot_codes[i:i + bs] for i in range(0, len(pilot_codes), bs)]
        t_batch_0 = time.time()
        
        layer_outputs = {l: [] for l in CANDIDATE_LAYERS}
        
        for batch in batches:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=False
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
            # Extract across candidate layers
            for l in CANDIDATE_LAYERS:
                # outputs.hidden_states[l + 1] corresponds to layer l
                h = outputs.hidden_states[l + 1] # (batch_size, seq_len, 4096)
                # Apply attention mask for mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                layer_outputs[l].append(pooled.cpu().to(torch.float32).numpy())
                
        elapsed = time.time() - t_batch_0
        throughput = len(pilot_codes) / elapsed
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
        
        print(f"Batch size {bs}: {throughput:.2f} samples/sec | Peak VRAM: {peak_vram_mb:.2f} MB")
        
        benchmark_results["batch_benchmarks"][f"bs_{bs}"] = {
            "batch_size": bs,
            "elapsed_seconds": elapsed,
            "samples_per_second": throughput,
            "peak_vram_allocated_mb": peak_vram_mb,
            "peak_vram_reserved_mb": peak_reserved_mb
        }
        
    # 4. Dimension & Storage Projections
    sample_embedding = layer_outputs[REFERENCE_LAYER][0]
    embedding_dim = sample_embedding.shape[-1]
    
    # Storage per sample per layer in FP32 (4096 * 4 bytes = 16.384 KB)
    bytes_per_vector = embedding_dim * 4
    
    projections = {
        "embedding_dimension": embedding_dim,
        "bytes_per_sample_per_layer": bytes_per_vector,
        "candidate_layer_count": len(CANDIDATE_LAYERS),
        "tracks": {
            "e1_existing_corpus": {
                "sample_count": 2973,
                "projected_extraction_minutes_bs8": (2973 / benchmark_results["batch_benchmarks"]["bs_8"]["samples_per_second"]) / 60,
                "storage_mb_per_layer": (2973 * bytes_per_vector) / (1024 * 1024),
                "total_storage_mb_8_layers": (2973 * bytes_per_vector * len(CANDIDATE_LAYERS)) / (1024 * 1024)
            },
            "e2_excel_1000": {
                "sample_count": 1000,
                "projected_extraction_minutes_bs8": (1000 / benchmark_results["batch_benchmarks"]["bs_8"]["samples_per_second"]) / 60,
                "storage_mb_per_layer": (1000 * bytes_per_vector) / (1024 * 1024),
                "total_storage_mb_8_layers": (1000 * bytes_per_vector * len(CANDIDATE_LAYERS)) / (1024 * 1024)
            },
            "e3_combined_corpus": {
                "sample_count": 3973,
                "projected_extraction_minutes_bs8": (3973 / benchmark_results["batch_benchmarks"]["bs_8"]["samples_per_second"]) / 60,
                "storage_mb_per_layer": (3973 * bytes_per_vector) / (1024 * 1024),
                "total_storage_mb_8_layers": (3973 * bytes_per_vector * len(CANDIDATE_LAYERS)) / (1024 * 1024)
            },
            "e_ext_independent_benchmark": {
                "sample_count": 700,
                "projected_extraction_minutes_bs8": (700 / benchmark_results["batch_benchmarks"]["bs_8"]["samples_per_second"]) / 60,
                "storage_mb_per_layer": (700 * bytes_per_vector) / (1024 * 1024),
                "total_storage_mb_8_layers": (700 * bytes_per_vector * len(CANDIDATE_LAYERS)) / (1024 * 1024)
            }
        }
    }
    benchmark_results["projections"] = projections
    
    with open("t4_mistral_pilot_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print("\nBenchmark complete. Saved to t4_mistral_pilot_benchmark.json")

if __name__ == "__main__":
    run_t4_pilot()

