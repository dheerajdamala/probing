import os
import sys
import json
import csv
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Add project root and extension to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, EXT_DIR)

from pipeline import SQLInjectionAnalysisPipeline

class LightweightMLP(nn.Module):
    def __init__(self, input_dim=4096, hidden1=256, hidden2=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def main():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    detailed_reports_dir = os.path.join(demo_dir, "detailed_sample_reports")
    os.makedirs(detailed_reports_dir, exist_ok=True)

    print("=" * 70)
    print("PROBE V3 TEST DEMO: VERIFICATION & ANALYSIS")
    print("=" * 70)

    # 1. Paths
    emb_path = os.path.join(BASE_DIR, "embeddings", "full_e2", "test", "sink", "layer_12.npy")
    lbl_path = os.path.join(BASE_DIR, "embeddings", "full_e2", "test", "labels.npy")
    test_jsonl = os.path.join(BASE_DIR, "data", "e2_excel", "test.jsonl")
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "MLP_sink_layer_12.pt")
    scaler_path = os.path.join(BASE_DIR, "checkpoints", "MLP_sink_layer_12_scaler.joblib")
    orig_pred_path = os.path.join(BASE_DIR, "predictions", "test_predictions.csv")

    # 2. Load embeddings and labels
    X = np.load(emb_path)
    y = np.load(lbl_path)
    print(f"Loaded embeddings: {X.shape}, labels: {y.shape}")

    # Load test metadata and source code
    samples = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {test_jsonl}")

    # 3. Load scaler and checkpoint
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    model = LightweightMLP()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Neural inference
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy()

    # Verify against historical predictions
    orig_probs = []
    with open(orig_pred_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_probs.append(float(row["Best_Validation_Selected"]))
    orig_probs = np.array(orig_probs)
    max_diff = np.max(np.abs(probs - orig_probs))
    print(f"Max absolute difference from saved test predictions: {max_diff:.8e}")
    assert max_diff < 1e-6, "Predictions do not match historical values!"

    threshold = 0.36
    preds = (probs >= threshold).astype(int)

    # 5. Calculate Metrics
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    spec = tn / (tn + fp)
    f1 = f1_score(y, preds)
    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)

    metrics = {
        "dataset_name": "E2_test",
        "sample_count": len(y),
        "vulnerable_count": int(np.sum(y == 1)),
        "safe_count": int(np.sum(y == 0)),
        "threshold": threshold,
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        },
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }

    print("\n--- METRICS SUMMARY ---")
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Precision:   {prec * 100:.2f}%")
    print(f"Recall:      {rec * 100:.2f}%")
    print(f"Specificity: {spec * 100:.2f}%")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Save metrics JSON
    metrics_file = os.path.join(demo_dir, "test_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # 6. Save all predictions CSV
    all_preds_file = os.path.join(demo_dir, "all_test_predictions.csv")
    with open(all_preds_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "sample_id", "ground_truth", "vulnerable_prob", "safe_prob",
            "prediction", "confidence", "correct", "error_type"
        ])
        for idx in range(len(samples)):
            s = samples[idx]
            sid = s.get("sample_id", s.get("id", f"sample_{idx}"))
            gt = int(y[idx])
            prob_vuln = float(probs[idx])
            prob_safe = 1.0 - prob_vuln
            pred = int(preds[idx])
            conf = max(prob_vuln, prob_safe)
            correct = (gt == pred)
            
            error_type = "NONE"
            if not correct:
                if gt == 0 and pred == 1:
                    error_type = "FALSE_POSITIVE"
                elif gt == 1 and pred == 0:
                    error_type = "FALSE_NEGATIVE"

            writer.writerow([
                idx, sid,
                "VULNERABLE" if gt == 1 else "SAFE",
                f"{prob_vuln:.4f}", f"{prob_safe:.4f}",
                "VULNERABLE" if pred == 1 else "SAFE",
                f"{conf:.4f}",
                "YES" if correct else "NO",
                error_type
            ])
    print(f"Saved all 151 predictions to {all_preds_file}")

    # 7. Detailed Analysis on Representative Samples
    selected_ids = ["t_0157", "t_0075", "t_0358", "t_0843", "t_0867", "t_0543"]
    pipeline = SQLInjectionAnalysisPipeline()

    selected_details = []

    for sid in selected_ids:
        idx = next(i for i, s in enumerate(samples) if s.get("sample_id", s.get("id")) == sid)
        s = samples[idx]
        code = s["code"]
        gt = int(y[idx])
        prob_vuln = float(probs[idx])
        prob_safe = 1.0 - prob_vuln
        pred = int(preds[idx])
        conf = max(prob_vuln, prob_safe)

        # Run pipeline
        report = pipeline.analyze(code, filename=f"{sid}.py")
        
        sample_info = {
            "sample_id": sid,
            "ground_truth": "VULNERABLE" if gt == 1 else "SAFE",
            "neural_probe": {
                "prediction": "VULNERABLE" if pred == 1 else "SAFE",
                "vulnerable_prob": prob_vuln,
                "safe_prob": prob_safe,
                "confidence": conf,
                "threshold": threshold,
                "correct": (gt == pred),
                "error_type": "FALSE_POSITIVE" if (gt == 0 and pred == 1) else ("FALSE_NEGATIVE" if (gt == 1 and pred == 0) else "NONE")
            },
            "static_dataflow_analysis": {
                "prediction": report["prediction"],
                "confidence": report["confidence"],
                "vulnerability_type": report["vulnerability_type"],
                "sink_location": report["sink_location"],
                "vulnerable_lines": report["vulnerable_lines"],
                "evidence": report["evidence"]
            },
            "source_code": code,
            "full_report": report
        }

        # Save individual JSON report
        rep_path = os.path.join(detailed_reports_dir, f"{sid}_report.json")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(sample_info, f, indent=2)

        selected_details.append(sample_info)
        print(f"Processed sample {sid}: GT={sample_info['ground_truth']}, Neural={sample_info['neural_probe']['prediction']}, Static={report['prediction']}")

    # Save aggregated selected details
    sel_file = os.path.join(demo_dir, "selected_sample_details.json")
    with open(sel_file, "w", encoding="utf-8") as f:
        json.dump(selected_details, f, indent=2)
    print(f"Saved {len(selected_details)} selected sample details to {sel_file}")

if __name__ == "__main__":
    main()
