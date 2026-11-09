"""
Phase 4 Transformation Sensitivity Audit Script.

Audits probe prediction stability across 4 semantics-preserving transformations:
1. Identifier variable renaming
2. Benign comment & docstring injection
3. Whitespace & indentation formatting
4. SQL keyword casing

Saves exact code pairs, probabilities, label flips, and AST validation status
to outputs/predictions/PHASE4_TRANSFORMATION_AUDIT.json.
"""

from __future__ import annotations

import os
import sys
import json
import ast
import re
import yaml
import numpy as np
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.load_model import load_model_and_tokenizer
from security_probe_sql.src.probe_model import SecurityProbe
from security_probe_sql.src.inference import SQLInjectionProbeAnalyzer
from security_probe_sql.src.shortcut_analysis import (
    transform_rename_variables,
    transform_add_comments_docstrings,
    transform_formatting_whitespace,
    transform_sql_casing,
)

def run_transformation_audit(config_path: str, max_samples: int = 150) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, cfg["outputs"]["models_dir"])
    test_path = os.path.join(project_root, cfg["data"]["test_split"])
    
    probe_path = os.path.join(models_dir, "probe_head_linear.pkl")
    probe = SecurityProbe.load(probe_path)
    
    is_mock = cfg["model"].get("mock", True)
    target_layer = cfg["model"]["mock_target_layer"] if is_mock else cfg["model"]["target_layer"]
    print("[Transformation Audit] Loading model and tokenizer...")
    model, tokenizer, _ = load_model_and_tokenizer(mock=is_mock)
    
    analyzer = SQLInjectionProbeAnalyzer(
        probe=probe,
        model=model,
        tokenizer=tokenizer,
        layer_idx=target_layer,
    )
    
    test_records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))
                
    test_subset = test_records[:max_samples]
    print(f"[Transformation Audit] Running audit across {len(test_subset)} test samples...")
    
    transform_registry = {
        "variable_renaming": transform_rename_variables,
        "comment_docstring_injection": transform_add_comments_docstrings,
        "formatting_whitespace": transform_formatting_whitespace,
        "sql_keyword_casing": transform_sql_casing,
    }
    
    audit_results: Dict[str, Any] = {}
    detailed_pairs: Dict[str, List[Dict[str, Any]]] = {}
    
    for t_name, t_func in transform_registry.items():
        print(f"  Auditing transformation: {t_name}...")
        deltas = []
        flips = 0
        pairs = []
        
        for sample in test_subset:
            orig_code = sample["code"]
            trans_code = t_func(orig_code)
            
            # Check AST preservation
            orig_ast_valid = False
            trans_ast_valid = False
            try:
                ast.parse(orig_code)
                orig_ast_valid = True
            except:
                pass
            try:
                ast.parse(trans_code)
                trans_ast_valid = True
            except:
                pass
                
            orig_res = analyzer.analyze(orig_code)
            trans_res = analyzer.analyze(trans_code)
            
            prob_orig = float(orig_res["probability"])
            prob_trans = float(trans_res["probability"])
            pred_orig = 1 if orig_res["is_vulnerable"] else 0
            pred_trans = 1 if trans_res["is_vulnerable"] else 0
            
            delta = abs(prob_orig - prob_trans)
            is_flip = (pred_orig != pred_trans)
            if is_flip:
                flips += 1
            deltas.append(delta)
            
            # Truncation check
            orig_tok_len = len(tokenizer.encode(orig_code))
            trans_tok_len = len(tokenizer.encode(trans_code))
            
            pairs.append({
                "sample_id": sample["sample_id"],
                "source_dataset": sample["source_dataset"],
                "ground_truth_label": sample["label"],
                "prob_orig": round(prob_orig, 4),
                "prob_trans": round(prob_trans, 4),
                "pred_orig": pred_orig,
                "pred_trans": pred_trans,
                "abs_delta": round(delta, 4),
                "label_flipped": is_flip,
                "orig_token_count": orig_tok_len,
                "trans_token_count": trans_tok_len,
                "semantic_syntax_preserved": (orig_ast_valid == trans_ast_valid),
                "sample_code_snippet_orig": orig_code[:120].strip(),
                "sample_code_snippet_trans": trans_code[:120].strip(),
            })
            
        mean_delta = float(np.mean(deltas))
        max_delta = float(np.max(deltas))
        flip_rate = float(flips / len(test_subset))
        stability_rate = 1.0 - flip_rate
        
        audit_results[t_name] = {
            "samples_tested": len(test_subset),
            "label_flips": flips,
            "prediction_flip_rate": round(flip_rate, 4),
            "stability_rate": round(stability_rate, 4),
            "mean_absolute_prob_delta": round(mean_delta, 4),
            "max_absolute_prob_delta": round(max_delta, 4),
        }
        detailed_pairs[t_name] = pairs
        print(f"    Stability: {stability_rate*100:.1f}%, Mean Delta: {mean_delta:.4f}")

    # Save detailed pairs
    pred_dir = os.path.join(project_root, cfg["outputs"]["predictions_dir"])
    metrics_dir = os.path.join(project_root, cfg["outputs"]["metrics_dir"])
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    pairs_out = os.path.join(pred_dir, "PHASE4_TRANSFORMATION_AUDIT.json")
    with open(pairs_out, "w", encoding="utf-8") as f:
        json.dump(detailed_pairs, f, indent=2)
        
    metrics_out = os.path.join(metrics_dir, "PHASE4_TRANSFORMATION_METRICS.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
        
    print(f"\n[Transformation Audit Completed]")
    print(f"  Detailed pairs saved to: {pairs_out}")
    print(f"  Metrics summary saved to: {metrics_out}")
    return audit_results

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_file = os.path.join(project_root, "config.yaml")
    run_transformation_audit(cfg_file, max_samples=150)
