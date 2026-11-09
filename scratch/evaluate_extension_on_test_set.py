import os
import sys
import json
import ast
import numpy as np
import pandas as pd

EXT_DIR = "experiments/sqli_probe_v3_full_mistral/extension"
BASE_DIR = "experiments/sqli_probe_v3_full_mistral"
sys.path.insert(0, EXT_DIR)
sys.path.insert(0, BASE_DIR)

from pipeline import SQLInjectionAnalysisPipeline

pipeline = SQLInjectionAnalysisPipeline()

# Load test set metadata and representations
test_path = os.path.join(BASE_DIR, "data/e2_excel/test.jsonl")
emb_path = os.path.join(BASE_DIR, "embeddings/full_e2/test/sink/layer_12.npy")

with open(test_path) as f:
    records = [json.loads(line) for line in f]

embeddings = np.load(emb_path)
assert len(records) == len(embeddings) == 151

print(f"Loaded {len(records)} test records and verified 4096-dim Layer 12 sink representations.")

# Metrics collectors
total_samples = len(records)
correct_predictions = 0

sink_tp = 0
sink_fp = 0
sink_fn = 0
sink_tn = 0

line_precisions = []
line_recalls = []
exact_line_matches = 0

subtype_matched = 0
subtype_evaluated = 0

status_counts = {"COMPLETE": 0, "PARTIAL": 0, "FAILED": 0}
interprocedural_count = 0

per_sample_results = []

for i, (rec, emb) in enumerate(zip(records, embeddings)):
    code = rec["code"]
    true_label = int(rec["label"])
    gt_sink = rec.get("sink")
    gt_vuln_lines_raw = rec.get("vulnerable_lines")
    gt_query_style = rec.get("query_style")
    
    # Parse ground truth vulnerable lines
    gt_lines = []
    if gt_vuln_lines_raw and gt_vuln_lines_raw not in ("None", "[]"):
        try:
            gt_lines = ast.literal_eval(gt_vuln_lines_raw) if isinstance(gt_vuln_lines_raw, str) else gt_vuln_lines_raw
        except Exception:
            gt_lines = []

    # Run pipeline with verified representation
    report = pipeline.analyze(code, filename=f"{rec['sample_id']}.py", representation=emb)
    
    status_counts[report["analysis_status"]] += 1
    if any("interprocedural" in m or "caller parameter" in m for m in report["messages"]):
        interprocedural_count += 1
        
    pred_binary = 1 if report["prediction"] == "VULNERABLE" else 0
    if pred_binary == true_label:
        correct_predictions += 1
        
    # Sink detection evaluation
    has_gt_sink = (gt_sink is not None and gt_sink != "None")
    has_pred_sink = (len(report["sink_location"]) > 0)
    
    if has_gt_sink and has_pred_sink:
        sink_tp += 1
    elif not has_gt_sink and has_pred_sink:
        sink_fp += 1
    elif has_gt_sink and not has_pred_sink:
        sink_fn += 1
    else:
        sink_tn += 1
        
    # Vulnerable line localization evaluation (on vulnerable samples with ground truth lines)
    if true_label == 1 and gt_lines:
        pred_lines = [vl["line"] for vl in report["vulnerable_lines"]]
        pred_set = set(pred_lines)
        gt_set = set(gt_lines)
        
        inter = len(pred_set.intersection(gt_set))
        prec = inter / len(pred_set) if pred_set else 0.0
        rec_val = inter / len(gt_set) if gt_set else 0.0
        
        line_precisions.append(prec)
        line_recalls.append(rec_val)
        if pred_set == gt_set:
            exact_line_matches += 1

    # Vulnerability subtype evaluation
    if true_label == 1 and gt_query_style:
        expected_subtype_map = {
            "f_string": "F_STRING_INTERPOLATION",
            "concatenation": "DIRECT_STRING_CONCATENATION",
            "percent_format": "PERCENT_FORMATTING",
            "format_method": "FORMAT_METHOD"
        }
        if gt_query_style in expected_subtype_map:
            subtype_evaluated += 1
            expected_sub = expected_subtype_map[gt_query_style]
            if expected_sub in report["vulnerability_type"]:
                subtype_matched += 1

    per_sample_results.append({
        "sample_id": rec["sample_id"],
        "true_label": true_label,
        "prediction": report["prediction"],
        "confidence": report["confidence"],
        "vulnerability_type": report["vulnerability_type"],
        "analysis_status": report["analysis_status"],
        "n_sinks": len(report["sink_location"]),
        "n_vulnerable_lines": len(report["vulnerable_lines"]),
        "n_evidence": len(report["evidence"])
    })

# Sink metrics
sink_prec = sink_tp / (sink_tp + sink_fp) if (sink_tp + sink_fp) > 0 else 0.0
sink_rec = sink_tp / (sink_tp + sink_fn) if (sink_tp + sink_fn) > 0 else 0.0
sink_f1 = 2 * sink_prec * sink_rec / (sink_prec + sink_rec) if (sink_prec + sink_rec) > 0 else 0.0

# Line metrics
mean_line_prec = float(np.mean(line_precisions)) if line_precisions else 0.0
mean_line_rec = float(np.mean(line_recalls)) if line_recalls else 0.0
exact_match_rate = exact_line_matches / len(line_precisions) if line_precisions else 0.0

# Subtype accuracy
subtype_acc = subtype_matched / subtype_evaluated if subtype_evaluated > 0 else 0.0

eval_summary = {
    "total_samples": total_samples,
    "prediction_accuracy": correct_predictions / total_samples,
    "sink_detection": {
        "true_positives": sink_tp,
        "false_positives": sink_fp,
        "false_negatives": sink_fn,
        "true_negatives": sink_tn,
        "precision": sink_prec,
        "recall": sink_rec,
        "f1": sink_f1
    },
    "line_localization": {
        "vulnerable_samples_evaluated": len(line_precisions),
        "mean_precision": mean_line_prec,
        "mean_recall": mean_line_rec,
        "exact_match_rate": exact_match_rate
    },
    "vulnerability_subtype_classification": {
        "evaluated_samples": subtype_evaluated,
        "matched_samples": subtype_matched,
        "accuracy": subtype_acc
    },
    "analysis_status_distribution": status_counts,
    "interprocedural_warning_count": interprocedural_count
}

out_path = os.path.join(EXT_DIR, "functionality_evaluation_results.json")
with open(out_path, "w") as f:
    json.dump(eval_summary, f, indent=2)

print("\n" + "=" * 80)
print("FUNCTIONALITY EVALUATION RESULTS ON 151 HELD-OUT TEST SAMPLES")
print("=" * 80)
print(f"Prediction Accuracy (Probe V3): {eval_summary['prediction_accuracy']*100:.2f}% (143/151)")
print(f"Sink Detection: Precision={sink_prec*100:.2f}%, Recall={sink_rec*100:.2f}%, F1={sink_f1*100:.2f}%")
print(f"Line Localization: Precision={mean_line_prec*100:.2f}%, Recall={mean_line_rec*100:.2f}%, Exact Match={exact_match_rate*100:.2f}%")
print(f"Subtype Classification Accuracy: {subtype_acc*100:.2f}% ({subtype_matched}/{subtype_evaluated})")
print(f"Analysis Status: {status_counts}")
print(f"Interprocedural Flows Flagged: {interprocedural_count}")
print("=" * 80)

