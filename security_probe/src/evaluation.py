"""Evaluation metrics and validation utilities for security probes.

Computes detection metrics (P, R, F1, PR-AUC, Brier score), per-CWE breakdowns,
synthetic/ground-truth localization benchmarks, and reliability/audit summaries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive binary detection performance metrics."""
    # Ensure inputs are valid
    if len(y_true) == 0:
        return {"error": "Empty ground-truth labels"}

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
    }

    if y_prob is not None:
        try:
            brier = float(brier_score_loss(y_true, y_prob))
            metrics["brier_score"] = round(brier, 4)
        except Exception:
            metrics["brier_score"] = None

        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob))
                precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = float(auc(recall_arr, precision_arr))
                metrics["roc_auc"] = round(roc_auc, 4)
                metrics["pr_auc"] = round(pr_auc, 4)
            else:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidate_thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Find decision threshold maximizing F1 score strictly on validation data."""
    if candidate_thresholds is None:
        candidate_thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 17)]

    best_thresh = 0.50
    best_f1 = -1.0

    for thresh in candidate_thresholds:
        preds = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return float(best_thresh), round(float(best_f1), 4)


def compute_per_cwe_metrics(
    y_true: List[int],
    y_pred: List[int],
    cwe_labels: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute detection metrics broken down by individual CWE category."""
    per_cwe: Dict[str, Dict[str, Any]] = {}
    unique_cwes = sorted(list(set(cwe_labels)))

    for cwe in unique_cwes:
        indices = [i for i, c in enumerate(cwe_labels) if c == cwe]
        if not indices:
            continue
        sub_true = np.array([y_true[i] for i in indices])
        sub_pred = np.array([y_pred[i] for i in indices])

        sub_acc = float(accuracy_score(sub_true, sub_pred))
        sub_f1 = float(f1_score(sub_true, sub_pred, zero_division=0))
        sub_rec = float(recall_score(sub_true, sub_pred, zero_division=0))

        per_cwe[cwe] = {
            "sample_count": len(indices),
            "vulnerable_count": int(sub_true.sum()),
            "accuracy": round(sub_acc, 4),
            "recall": round(sub_rec, 4),
            "f1": round(sub_f1, 4),
        }

    return per_cwe


def evaluate_localization_overlap(
    predicted_lines: Set[int],
    ground_truth_lines: Set[int],
) -> Dict[str, float]:
    """Evaluate line-level localization precision, recall, F1, and IoU overlap.
    
    Used when line-level ground-truth annotations are provided.
    """
    # Case 1: Both sets empty (true negative line localization)
    if not predicted_lines and not ground_truth_lines:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "true_negatives": 1,
        }

    # Case 2: One set empty, the other non-empty
    if not predicted_lines or not ground_truth_lines:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "true_negatives": 0,
        }

    intersection = len(predicted_lines.intersection(ground_truth_lines))
    union = len(predicted_lines.union(ground_truth_lines))

    prec = intersection / len(predicted_lines)
    rec = intersection / len(ground_truth_lines)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "true_negatives": 0,
    }
