"""
Evaluation Metrics Module for Python SQL Injection Security Probe.

Computes comprehensive detection and localization metrics:
- Accuracy, Balanced Accuracy, Precision, Recall, F1, Specificity
- ROC-AUC, PR-AUC, Confusion Matrix, FPR, FNR
- Per-source performance breakdown
- Line-level localization metrics on ground-truth annotated samples
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute all required binary detection metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_probs: Predicted probabilities P(y=1|x) in [0, 1].
        
    Returns:
        Dict of computed numerical evaluation results.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    roc_auc = None
    pr_auc = None

    if y_probs is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = 0.50
        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
            pr_auc = float(auc(r_curve, p_curve))
        except Exception:
            pr_auc = float(np.mean(y_true))

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "total_samples": len(y_true),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum()),
    }


def compute_localization_metrics(
    annotated_test_samples: List[Dict[str, Any]],
    predicted_findings_list: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute localization accuracy against genuine ground-truth span annotations.
    
    Args:
        annotated_test_samples: Samples that contain verified vulnerable_lines.
        predicted_findings_list: List of predicted findings for each sample.
        
    Returns:
        Top-1, Top-3, Top-5 hit rates and Mean Reciprocal Rank (MRR).
    """
    if not annotated_test_samples:
        return {
            "total_annotated_samples": 0,
            "status": "no_ground_truth_annotations_in_subset",
        }

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks = []

    for sample, findings in zip(annotated_test_samples, predicted_findings_list):
        gt_lines = set(sample.get("vulnerable_lines", []))
        if not gt_lines and sample.get("span_details"):
            s_line = sample["span_details"].get("start_line")
            if s_line:
                gt_lines.add(s_line)

        if not gt_lines:
            continue

        predicted_lines = [f["start_line"] for f in findings]

        # Top-k checks
        hit_top1 = len(predicted_lines) > 0 and (predicted_lines[0] in gt_lines)
        hit_top3 = any(l in gt_lines for l in predicted_lines[:3])
        hit_top5 = any(l in gt_lines for l in predicted_lines[:5])

        if hit_top1:
            top1_hits += 1
        if hit_top3:
            top3_hits += 1
        if hit_top5:
            top5_hits += 1

        # MRR
        rr = 0.0
        for rank, p_line in enumerate(predicted_lines, 1):
            if p_line in gt_lines:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    num_eval = max(len(reciprocal_ranks), 1)
    return {
        "total_annotated_evaluated": len(reciprocal_ranks),
        "top_1_hit_rate": round(top1_hits / num_eval, 4),
        "top_3_hit_rate": round(top3_hits / num_eval, 4),
        "top_5_hit_rate": round(top5_hits / num_eval, 4),
        "mean_reciprocal_rank": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0,
    }
