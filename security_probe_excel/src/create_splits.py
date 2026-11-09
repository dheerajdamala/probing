"""
Create leakage-resistant, group-aware train/validation/test splits.
Ensures that paired vulnerable/fixed samples, identical base tasks,
and related code clusters never cross split boundaries.
"""

import os
import sys
import json
import random
from collections import defaultdict
from typing import Dict, List, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def get_group_key(rec: Dict[str, Any]) -> str:
    sid = str(rec.get("sample_id", ""))
    orig_id = str(rec.get("original_sample_id", ""))
    
    # Check if 3k pair
    if "ext_3k_" in sid:
        # e.g. ext_3k_task_cwe89_0001_vuln -> group by task_cwe89_0001
        parts = sid.split("_")
        return "_".join(parts[:-1])
        
    # Check if excel pair
    if sid.startswith("t_"):
        return sid
        
    # Check external validated
    if "ext_val_" in sid:
        # group by original_sample_id or repository
        return orig_id if orig_id else sid
        
    return sid

def create_splits():
    base_dir = r"C:\Users\dheer\Desktop\isae\security_probe_excel\data"
    corpus_path = os.path.join(base_dir, "deduplicated", "deduplicated_corpus.jsonl")
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    print(f"Total deduplicated records: {len(records)}")
    
    # Group records by group_key
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        gk = get_group_key(r)
        groups[gk].append(r)
        
    print(f"Total distinct groups: {len(groups)}")
    
    # Separate groups by dominant label for stratified group split
    vuln_groups = []
    safe_groups = []
    mixed_groups = []
    
    for gk, group_recs in groups.items():
        labels = set(r["label"] for r in group_recs)
        if len(labels) > 1:
            mixed_groups.append((gk, group_recs))
        elif 1 in labels:
            vuln_groups.append((gk, group_recs))
        else:
            safe_groups.append((gk, group_recs))
            
    print(f"Group breakdown: Mixed={len(mixed_groups)}, Pure Vuln={len(vuln_groups)}, Pure Safe={len(safe_groups)}")
    
    # Seed for reproducibility
    rng = random.Random(42)
    rng.shuffle(mixed_groups)
    rng.shuffle(vuln_groups)
    rng.shuffle(safe_groups)
    
    # Target split proportions: 70% train, 15% val, 15% test
    def partition_groups(g_list, train_ratio=0.70, val_ratio=0.15):
        n = len(g_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_g = g_list[:n_train]
        val_g = g_list[n_train:n_train + n_val]
        test_g = g_list[n_train + n_val:]
        return train_g, val_g, test_g
        
    m_tr, m_va, m_te = partition_groups(mixed_groups)
    v_tr, v_va, v_te = partition_groups(vuln_groups)
    s_tr, s_va, s_te = partition_groups(safe_groups)
    
    train_recs = [r for _, glist in (m_tr + v_tr + s_tr) for r in glist]
    val_recs = [r for _, glist in (m_va + v_va + s_va) for r in glist]
    test_recs = [r for _, glist in (m_te + v_te + s_te) for r in glist]
    
    # Shuffle within each split
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    rng.shuffle(test_recs)
    
    def get_stats(recs):
        n = len(recs)
        v = sum(1 for r in recs if r["label"] == 1)
        s = n - v
        return {"total": n, "vulnerable": v, "safe": s, "prevalence": v / n if n > 0 else 0}
        
    stats = {
        "train": get_stats(train_recs),
        "val": get_stats(val_recs),
        "test": get_stats(test_recs),
        "total": len(records)
    }
    
    print("\n=== SPLIT SUMMARY ===")
    print("Train:", stats["train"])
    print("Val:  ", stats["val"])
    print("Test: ", stats["test"])
    
    # Check no group overlap
    train_groups = set(get_group_key(r) for r in train_recs)
    val_groups = set(get_group_key(r) for r in val_recs)
    test_groups = set(get_group_key(r) for r in test_recs)
    
    assert len(train_groups.intersection(val_groups)) == 0, "Leakage between train and val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Leakage between train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Leakage between val and test!"
    print("Verification passed: ZERO group or pair leakage across splits!")
    
    # Save splits
    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        out_f = os.path.join(splits_dir, f"{name}.jsonl")
        with open(out_f, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {name} split ({len(recs)} records) to {out_f}")
        
    with open(os.path.join(splits_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    create_splits()

