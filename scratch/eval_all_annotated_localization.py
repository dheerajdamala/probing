import json
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import yaml
import numpy as np
import pandas as pd

from security_probe_sql.src.inference import SQLInjectionProbeAnalyzer

config_path = "security_probe_sql/configs/probe.yaml"
analyzer = SQLInjectionProbeAnalyzer(config_path=config_path, model_type="linear")

annotated_samples = []
with open("security_probe_sql/data/processed/validated_sql_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("ground_truth_spans_available") or (r.get("vulnerable_lines") and len(r.get("vulnerable_lines")) > 0):
            annotated_samples.append(r)

print(f"Total annotated samples in corpus: {len(annotated_samples)}")

top_1_hits = 0
top_3_hits = 0
top_5_hits = 0
reciprocal_ranks = []
details = []

for idx, sample in enumerate(annotated_samples):
    code = sample["code"]
    true_lines = set(sample.get("vulnerable_lines", []))
    if not true_lines and sample.get("span_details"):
        sd = sample["span_details"]
        if "start_line" in sd:
            true_lines.add(int(sd["start_line"]))
            
    if not true_lines:
        continue
        
    analysis = analyzer.analyze(code, sample_id=sample["sample_id"])
    line_scores = analysis["line_scores"] # list of dicts: [{"line_number": L, "score": S, ...}]
    ranked_lines = [item["line_number"] for item in line_scores]
    
    # Calculate rank of first true line
    found_rank = None
    for r_idx, l_num in enumerate(ranked_lines):
        if l_num in true_lines:
            found_rank = r_idx + 1
            break
            
    if found_rank is not None:
        reciprocal_ranks.append(1.0 / found_rank)
        if found_rank == 1:
            top_1_hits += 1
        if found_rank <= 3:
            top_3_hits += 1
        if found_rank <= 5:
            top_5_hits += 1
    else:
        reciprocal_ranks.append(0.0)
        
    details.append({
        "sample_id": sample["sample_id"],
        "source": sample["source_dataset"],
        "true_lines": list(true_lines),
        "rank": found_rank,
        "top_line_predicted": ranked_lines[0] if ranked_lines else None,
        "code_len": len(ranked_lines)
    })

n = len(details)
print(f"Evaluated {n} annotated samples:")
print(f"  Top-1 Hit Rate: {top_1_hits / n:.4f} ({top_1_hits}/{n})")
print(f"  Top-3 Hit Rate: {top_3_hits / n:.4f} ({top_3_hits}/{n})")
print(f"  Top-5 Hit Rate: {top_5_hits / n:.4f} ({top_5_hits}/{n})")
print(f"  Mean Reciprocal Rank (MRR): {np.mean(reciprocal_ranks):.4f}")
