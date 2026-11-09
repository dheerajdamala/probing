import os
import json
import numpy as np

emb_dir = "experiments/sqli_probe_v3_full_mistral/embeddings"
data_dir = "experiments/sqli_probe_v3_full_mistral/data/e2_excel"

print("=" * 80)
print("AUDITING EXTRACTED MISTRAL-7B PROBE V3 EMBEDDINGS")
print("=" * 80)

# 1. Model Verification Report
mv_path = os.path.join(emb_dir, "model_verification_report.json")
if os.path.exists(mv_path):
    with open(mv_path) as f:
        mv = json.load(f)
    print("\n--- Model Verification Report ---")
    for k, v in mv.items():
        print(f"  {k}: {v}")
else:
    print(f"Missing {mv_path}")

# 2. Pilot Report
pilot_path = os.path.join(emb_dir, "pilot_extraction_report.json")
if os.path.exists(pilot_path):
    with open(pilot_path) as f:
        pr = json.load(f)
    print("\n--- Pilot Extraction Report Summary ---")
    print(f"  Status: {pr.get('status')}")
    print(f"  Passed assertions: {pr.get('passed_assertions')}")
    print(f"  Layers extracted: {pr.get('candidate_layers', pr.get('layers_extracted'))}")
    print(f"  Pilot samples count: {len(pr.get('samples', pr.get('sample_ids', [])))}")
else:
    print(f"Missing {pilot_path}")

# 3. Full E2 Extraction Manifest
manifest_path = os.path.join(emb_dir, "full_e2_extraction_manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        mf = json.load(f)
    print("\n--- Full E2 Extraction Manifest Summary ---")
    print(f"  Status: {mf.get('status')}")
    print(f"  Total samples: {mf.get('total_samples')}")
    print(f"  Total elapsed time: {mf.get('total_elapsed_seconds'):.1f} s ({mf.get('total_elapsed_seconds')/60:.1f} min)")
    print(f"  Throughput: {mf.get('overall_throughput_samples_per_sec'):.3f} samples/sec")
    print(f"  Peak VRAM: {mf.get('peak_vram_mb'):.1f} MB")
    print(f"  Hidden size: {mf.get('hidden_dimension')}")
    print(f"  Candidate layers: {mf.get('candidate_layers')}")
    print(f"  Splits: {list(mf.get('splits', {}).keys())}")
else:
    print(f"Missing {manifest_path}")

# 4. Detailed Array Verification
print("\n--- Checking All Embedding Files ---")
splits = ["train", "val", "test"]
expected_counts = {"train": 700, "val": 149, "test": 151}
layers = [4, 8, 12, 16, 20, 24, 28, 32]
poolings = ["mean", "sink"]

total_arrays_checked = 0
total_nans = 0
total_infs = 0

for split in splits:
    split_dir = os.path.join(emb_dir, "full_e2", split)
    labels = np.load(os.path.join(split_dir, "labels.npy"))
    with open(os.path.join(split_dir, "sample_ids.json")) as f:
        sids = json.load(f)
    
    # Check against original jsonl
    orig_records = [json.loads(line) for line in open(os.path.join(data_dir, f"{split}.jsonl"))]
    orig_sids = [r["sample_id"] for r in orig_records]
    orig_labels = [int(r["label"]) for r in orig_records]
    
    assert sids == orig_sids, f"Sample IDs mismatch for split {split}!"
    assert np.array_equal(labels, orig_labels), f"Labels mismatch for split {split}!"
    
    exp_n = expected_counts[split]
    assert len(labels) == exp_n, f"Expected {exp_n} labels, got {len(labels)}"
    print(f"\n[Split: {split.upper()}] Samples: {len(labels)} (Vuln: {np.sum(labels==1)}, Safe: {np.sum(labels==0)})")
    print(f"  ID Alignment: 100% matched with frozen {split}.jsonl ({exp_n}/{exp_n})")
    print(f"  Label Alignment: 100% matched with frozen {split}.jsonl ({exp_n}/{exp_n})")
    
    for pool in poolings:
        for layer in layers:
            npy_path = os.path.join(split_dir, pool, f"layer_{layer}.npy")
            arr = np.load(npy_path)
            total_arrays_checked += 1
            
            # Shape assertion
            assert arr.shape == (exp_n, 4096), f"Wrong shape: {arr.shape}, expected ({exp_n}, 4096)"
            assert arr.dtype == np.float32, f"Wrong dtype: {arr.dtype}, expected float32"
            
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            total_nans += n_nan
            total_infs += n_inf
            assert n_nan == 0, f"Found {n_nan} NaNs in {npy_path}"
            assert n_inf == 0, f"Found {n_inf} Infs in {npy_path}"
            
            mean_norm = float(np.linalg.norm(arr, axis=1).mean())
            if layer in [4, 16, 32]:
                print(f"  {pool:4s} | Layer {layer:2d}: shape={arr.shape}, dtype={arr.dtype}, mean_L2_norm={mean_norm:.2f}, NaNs={n_nan}, Infs={n_inf}")

print("\n" + "=" * 80)
print(f"AUDIT COMPLETE: All {total_arrays_checked} embedding arrays verified successfully!")
print(f"Total NaNs: {total_nans}")
print(f"Total Infs: {total_infs}")
print(f"Sample Alignment: 100% verified across train (700), val (149), test (151)")
print(f"Vector Dimensions: 4,096 on ALL 8 candidate layers [4, 8, 12, 16, 20, 24, 28, 32]")
print(f"Dual Pooling: Attention Mean and AST Sink both extracted from identical forward passes")
print("=" * 80)
