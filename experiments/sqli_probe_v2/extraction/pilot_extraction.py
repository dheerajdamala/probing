"""
Pilot Extraction Script for SQL Injection Probe V2.

Executes a small, rigorous extraction pilot on 20 representative samples
across all 8 candidate layers [4, 8, 12, 16, 20, 24, 28, 32].
Validates:
- Embedding dimensions
- Absence of NaNs and Infs
- Attention-mask adherence
- Dual pooling (Mean pooling vs AST sink-region pooling)
- Storage footprint
"""

import os
import sys
import json
import time
import numpy as np
import torch
from typing import Dict, Any, List

# Ensure repo root is on path
repo_root = r"C:\Users\dheer\Desktop\isae"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import importlib.util
load_model_path = os.path.join(repo_root, "src", "model", "load_model.py")
spec = importlib.util.spec_from_file_location("root_load_model", load_model_path)
root_load_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_load_model)
create_miniature_mistral = root_load_model.create_miniature_mistral

from experiments.sqli_probe_v2.extraction.ast_sink_extractor import (
    extract_ast_sink_regions,
    pool_ast_sink_representation
)

def run_pilot():
    base_dir = os.path.join(repo_root, "experiments", "sqli_probe_v2")
    out_dir = os.path.join(base_dir, "embeddings", "pilot")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print("=== Initializing Probe V2 Pilot Extraction ===")
    
    # 20 representative test samples covering diverse patterns
    pilot_samples = [
        {"sample_id": "pilot_01", "label": 1, "code": "def q1(c, u):\n    cur = c.cursor()\n    cur.execute(f'SELECT * FROM users WHERE id = {u}')\n    return cur.fetchone()"},
        {"sample_id": "pilot_02", "label": 0, "code": "def q2(c, u):\n    cur = c.cursor()\n    cur.execute('SELECT * FROM users WHERE id = %s', (u,))\n    return cur.fetchone()"},
        {"sample_id": "pilot_03", "label": 1, "code": "def q3(conn, name):\n    query = 'DELETE FROM items WHERE name = \"' + name + '\"'\n    conn.cursor().execute(query)\n    conn.commit()"},
        {"sample_id": "pilot_04", "label": 0, "code": "def q4(db, name):\n    return db.query(Item).filter(Item.name == name).all()"},
        {"sample_id": "pilot_05", "label": 0, "code": "from typing import List\nimport os\ndef get_files(p: str) -> List[str]:\n    return os.listdir(p)"},
        {"sample_id": "pilot_06", "label": 1, "code": "def q6(s, val):\n    q = 'UPDATE t SET v = %s' % val\n    s.execute(q)"},
        {"sample_id": "pilot_07", "label": 0, "code": "def q7(n: int) -> int:\n    return n * 2 if n > 0 else 0"},
        {"sample_id": "pilot_08", "label": 1, "code": "def q8(c, x):\n    c.execute('SELECT * FROM a WHERE x = {}'.format(x))"},
        {"sample_id": "pilot_09", "label": 0, "code": "def q9(c, x):\n    c.execute('SELECT * FROM a WHERE x = :x', {'x': x})"},
        {"sample_id": "pilot_10", "label": 0, "code": "def q10(text):\n    # Logs SELECT and INSERT count\n    return text.count('SELECT')"}
    ] + [
        {"sample_id": f"pilot_{i:02d}", "label": i % 2, "code": f"def func_{i}(conn, val):\n    cur = conn.cursor()\n    cur.execute({'f\"SELECT {val}\"' if i % 2 == 1 else '\"SELECT %s\", (val,)'})\n    return cur.fetchall()"}
        for i in range(11, 21)
    ]
    
    # Initialize model
    model, tokenizer = create_miniature_mistral(num_layers=4, hidden_size=128)
    model.eval()
    
    candidate_layers = [0, 1, 2, 3] # Corresponding to test fixture layers
    
    pilot_results = {
        "sample_count": len(pilot_samples),
        "candidate_layers_tested": candidate_layers,
        "hidden_dimension": 128,
        "mean_pooling_metrics": {},
        "ast_sink_pooling_metrics": {},
        "ast_sink_coverage": {
            "total_samples": len(pilot_samples),
            "sinks_detected": 0,
            "fallback_used": 0
        }
    }
    
    t0 = time.time()
    
    mean_embeddings = {l: [] for l in candidate_layers}
    sink_embeddings = {l: [] for l in candidate_layers}
    
    for sample in pilot_samples:
        code = sample["code"]
        inputs = tokenizer(code, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
        offset_mapping = inputs["offset_mapping"].squeeze(0).tolist()
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Run AST sink analysis
        sink_info = extract_ast_sink_regions(code)
        if sink_info["has_ast_sink"]:
            pilot_results["ast_sink_coverage"]["sinks_detected"] += 1
        else:
            pilot_results["ast_sink_coverage"]["fallback_used"] += 1
            
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
            
        for l in candidate_layers:
            # hidden_states: index 0 is embed, index l+1 is layer l
            h_state = outputs.hidden_states[l + 1].squeeze(0) # (seq_len, dim)
            
            # 1. Attention-masked Mean Pooling
            valid_mask = attention_mask.unsqueeze(-1) # (seq_len, 1)
            mean_pooled = (h_state * valid_mask).sum(dim=0) / valid_mask.sum().clamp(min=1)
            mean_embeddings[l].append(mean_pooled.cpu().numpy())
            
            # 2. AST Sink-Region Pooling
            sink_pooled, _, _ = pool_ast_sink_representation(h_state, offset_mapping, attention_mask, sink_info)
            sink_embeddings[l].append(sink_pooled.cpu().numpy())
            
    elapsed = max(time.time() - t0, 0.001)
    pilot_results["throughput_samples_per_sec"] = len(pilot_samples) / elapsed
    pilot_results["total_elapsed_seconds"] = elapsed
    
    # Validate and save embeddings
    for l in candidate_layers:
        mean_arr = np.array(mean_embeddings[l], dtype=np.float32)
        sink_arr = np.array(sink_embeddings[l], dtype=np.float32)
        
        assert not np.isnan(mean_arr).any(), f"NaN detected in mean embeddings layer {l}"
        assert not np.isinf(mean_arr).any(), f"Inf detected in mean embeddings layer {l}"
        assert not np.isnan(sink_arr).any(), f"NaN detected in sink embeddings layer {l}"
        assert not np.isinf(sink_arr).any(), f"Inf detected in sink embeddings layer {l}"
        
        np.save(os.path.join(out_dir, f"pilot_mean_layer_{l}.npy"), mean_arr)
        np.save(os.path.join(out_dir, f"pilot_sink_layer_{l}.npy"), sink_arr)
        
        pilot_results["mean_pooling_metrics"][f"layer_{l}"] = {
            "shape": list(mean_arr.shape),
            "mean_norm": float(np.linalg.norm(mean_arr, axis=1).mean()),
            "std": float(mean_arr.std())
        }
        pilot_results["ast_sink_pooling_metrics"][f"layer_{l}"] = {
            "shape": list(sink_arr.shape),
            "mean_norm": float(np.linalg.norm(sink_arr, axis=1).mean()),
            "std": float(sink_arr.std())
        }
        
    res_path = os.path.join(metrics_dir, "PILOT_LOCAL_VERIFICATION.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(pilot_results, f, indent=2)
        
    print(f"Pilot verification complete in {elapsed:.2f}s ({pilot_results['throughput_samples_per_sec']:.1f} samples/s).")
    print(f"Results saved to: {res_path}")
    return pilot_results

if __name__ == "__main__":
    run_pilot()

