"""
Comprehensive Adversarial Evaluation Runner for SQL Injection Probe V3 Extension.
Evaluates the interprocedural analysis engine across all 100 adversarial cases.
Computes detection metrics, localization precision/recall, false positive rate,
per-category breakdowns, and records detailed per-case diagnostic outcomes.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXT_DIR, ".."))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline import SQLInjectionAnalysisPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(BASE_DIR, "adversarial_cases")
MANIFEST_PATH = os.path.join(BASE_DIR, "adversarial_manifest.json")
RESULTS_PATH = os.path.join(BASE_DIR, "adversarial_evaluation_results.json")

def run_adversarial_eval():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipeline = SQLInjectionAnalysisPipeline()

    detailed_results = []
    category_stats = {}

    tp, fp, tn, fn = 0, 0, 0, 0
    sink_tp, sink_fn = 0, 0
    vuln_line_tp, vuln_line_fp, vuln_line_fn = 0, 0, 0
    status_matches = 0

    total_start = time.time()

    for item in manifest:
        cid = item["id"]
        cat = item["category"]
        is_project = item["is_project"]
        rel_fn = item["filename"]
        target_path = os.path.join(CASES_DIR, rel_fn)
        expected_label = item["expected_label"]
        expected_status = item["expected_status"]
        expected_sink = item.get("expected_sink_line")
        expected_vuln_lines = set(item.get("expected_vuln_lines", []))
        expected_subtype = item.get("expected_subtype")

        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "latencies_ms": []
            }

        category_stats[cat]["total"] += 1

        t0 = time.time()
        if is_project:
            report = pipeline.analyze_project(target_path)
        else:
            with open(target_path, "r", encoding="utf-8") as f_obj:
                code_str = f_obj.read()
            report = pipeline.analyze(code_str, filename=os.path.basename(rel_fn))
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        category_stats[cat]["latencies_ms"].append(elapsed_ms)

        pred_label = report["prediction"]
        analysis_status = report["analysis_status"]

        # Classification correctness
        is_pred_correct = (pred_label == expected_label)
        if is_pred_correct:
            category_stats[cat]["correct"] += 1

        if expected_label == "VULNERABLE":
            if pred_label == "VULNERABLE":
                tp += 1
                category_stats[cat]["tp"] += 1
            else:
                fn += 1
                category_stats[cat]["fn"] += 1
        elif expected_label == "SAFE":
            if pred_label == "SAFE":
                tn += 1
                category_stats[cat]["tn"] += 1
            else:
                fp += 1
                category_stats[cat]["fp"] += 1
        elif expected_label == "UNCERTAIN":
            if pred_label == "UNCERTAIN":
                tn += 1
                category_stats[cat]["tn"] += 1
            else:
                fp += 1
                category_stats[cat]["fp"] += 1

        # Status check
        if analysis_status == expected_status:
            status_matches += 1

        # Sink check
        detected_sinks = [s["line"] for s in report.get("sink_location", [])]
        sink_match = False
        if expected_sink is not None:
            if expected_sink in detected_sinks:
                sink_tp += 1
                sink_match = True
            else:
                sink_fn += 1
        else:
            if not detected_sinks:
                sink_match = True

        # Vulnerable line check
        detected_vuln_lines = {vl["line"] for vl in report.get("vulnerable_lines", [])}
        for l in detected_vuln_lines:
            if l in expected_vuln_lines:
                vuln_line_tp += 1
            else:
                vuln_line_fp += 1
        for l in expected_vuln_lines:
            if l not in detected_vuln_lines:
                vuln_line_fn += 1

        # Subtype check
        reported_types = report.get("vulnerability_type", [])
        subtype_match = False
        if expected_subtype:
            subtype_match = expected_subtype in reported_types
        else:
            subtype_match = (len(reported_types) == 0 or reported_types == ["STATIC_OR_SAFE_SQL"])

        detailed_results.append({
            "id": cid,
            "category": cat,
            "expected_label": expected_label,
            "predicted_label": pred_label,
            "prediction_correct": is_pred_correct,
            "expected_status": expected_status,
            "actual_status": analysis_status,
            "status_correct": (analysis_status == expected_status),
            "confidence": report.get("confidence", 0.0),
            "vulnerability_types": reported_types,
            "expected_subtype": expected_subtype,
            "subtype_matched": subtype_match,
            "expected_sink": expected_sink,
            "detected_sinks": detected_sinks,
            "sink_matched": sink_match,
            "expected_vuln_lines": sorted(list(expected_vuln_lines)),
            "detected_vuln_lines": sorted(list(detected_vuln_lines)),
            "latency_ms": elapsed_ms,
            "findings_count": len(report.get("findings", [])),
            "messages": report.get("messages", [])
        })

    total_time_ms = round((time.time() - total_start) * 1000, 2)
    total_samples = len(manifest)

    precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
    recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    f1 = round(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0, 4)
    accuracy = round((tp + tn) / total_samples, 4)

    sink_recall = round(sink_tp / (sink_tp + sink_fn) if (sink_tp + sink_fn) > 0 else 1.0, 4)
    vuln_line_prec = round(vuln_line_tp / (vuln_line_tp + vuln_line_fp) if (vuln_line_tp + vuln_line_fp) > 0 else 0.0, 4)
    vuln_line_rec = round(vuln_line_tp / (vuln_line_tp + vuln_line_fn) if (vuln_line_tp + vuln_line_fn) > 0 else 0.0, 4)
    vuln_line_f1 = round(2 * vuln_line_prec * vuln_line_rec / (vuln_line_prec + vuln_line_rec) if (vuln_line_prec + vuln_line_rec) > 0 else 0.0, 4)

    # Category summaries
    cat_summary = {}
    for cat, data in category_stats.items():
        tot = data["total"]
        cor = data["correct"]
        cat_summary[cat] = {
            "total_cases": tot,
            "correct_predictions": cor,
            "accuracy": round(cor / tot if tot > 0 else 0.0, 4),
            "tp": data["tp"],
            "fp": data["fp"],
            "tn": data["tn"],
            "fn": data["fn"],
            "mean_latency_ms": round(sum(data["latencies_ms"]) / len(data["latencies_ms"]) if data["latencies_ms"] else 0.0, 2)
        }

    # Identify any misclassified cases or defects
    defects = [
        r for r in detailed_results if not r["prediction_correct"]
    ]

    output_data = {
        "evaluation_title": "Adversarial Evaluation of SQL Injection Probe V3 Interprocedural Engine",
        "total_adversarial_cases": total_samples,
        "total_elapsed_ms": total_time_ms,
        "mean_latency_per_case_ms": round(total_time_ms / total_samples, 2),
        "overall_classification_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            }
        },
        "localization_metrics": {
            "sink_detection_recall": sink_recall,
            "vulnerable_line_precision": vuln_line_prec,
            "vulnerable_line_recall": vuln_line_rec,
            "vulnerable_line_f1": vuln_line_f1
        },
        "status_accuracy": round(status_matches / total_samples, 4),
        "category_performance": cat_summary,
        "defects_count": len(defects),
        "defects": defects,
        "detailed_results": detailed_results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("=================================================================")
    print("       ADVERSARIAL EVALUATION COMPLETED ACROSS 100 CASES         ")
    print("=================================================================")
    print(f"Overall Accuracy:  {accuracy * 100:.2f}% ({tp+tn}/{total_samples})")
    print(f"Precision:         {precision * 100:.2f}%")
    print(f"Recall:            {recall * 100:.2f}%")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Confusion Matrix:  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Sink Recall:       {sink_recall * 100:.2f}%")
    print(f"Vulnerable Line:   Prec={vuln_line_prec * 100:.2f}%, Rec={vuln_line_rec * 100:.2f}%, F1={vuln_line_f1:.4f}")
    print(f"Total Time:        {total_time_ms:.2f} ms ({total_time_ms / total_samples:.2f} ms/case)")
    print(f"Defects Discovered:{len(defects)}")
    print("=================================================================")

    if defects:
        print("\nDefect Summary:")
        for d in defects:
            print(f"  - [{d['id']}] Expected: {d['expected_label']}, Predicted: {d['predicted_label']} (Cat: {d['category']})")

    return output_data

if __name__ == "__main__":
    run_adversarial_eval()

