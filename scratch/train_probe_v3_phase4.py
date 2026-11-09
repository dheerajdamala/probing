import os
import json
import time
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, brier_score_loss
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure output directories exist
os.makedirs("experiments/sqli_probe_v3_full_mistral/results", exist_ok=True)
os.makedirs("experiments/sqli_probe_v3_full_mistral/checkpoints", exist_ok=True)
os.makedirs("experiments/sqli_probe_v3_full_mistral/predictions", exist_ok=True)

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "experiments/sqli_probe_v3_full_mistral/data/e2_excel"
EMB_DIR = "experiments/sqli_probe_v3_full_mistral/embeddings/full_e2"

CANDIDATE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
POOLING_STRATEGIES = ["mean", "sink"]

# ---------------------------------------------------------
# 1. Load Labels, Sample IDs, and Raw Text Data
# ---------------------------------------------------------
print("Loading frozen dataset metadata and text...")
train_records = [json.loads(l) for l in open(os.path.join(DATA_DIR, "train.jsonl"))]
val_records = [json.loads(l) for l in open(os.path.join(DATA_DIR, "val.jsonl"))]
test_records = [json.loads(l) for l in open(os.path.join(DATA_DIR, "test.jsonl"))]

y_train = np.array([int(r["label"]) for r in train_records], dtype=int)
y_val = np.array([int(r["label"]) for r in val_records], dtype=int)
y_test = np.array([int(r["label"]) for r in test_records], dtype=int)

train_ids = [r["sample_id"] for r in train_records]
val_ids = [r["sample_id"] for r in val_records]
test_ids = [r["sample_id"] for r in test_records]

train_code = [r["code"] for r in train_records]
val_code = [r["code"] for r in val_records]
test_code = [r["code"] for r in test_records]

print(f"Loaded Train: {len(y_train)} (Vuln: {sum(y_train==1)}), Val: {len(y_val)} (Vuln: {sum(y_val==1)}), Test: {len(y_test)} (Vuln: {sum(y_test==1)})")

# ---------------------------------------------------------
# Helper: Metrics & Bootstrap CI
# ---------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }
    return metrics

def bootstrap_ci(y_true, y_prob, threshold, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_metrics = {k: [] for k in ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "precision", "recall", "specificity"]}
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_true = y_true[idx]
        b_prob = y_prob[idx]
        if len(np.unique(b_true)) < 2:
            continue
        b_pred = (b_prob >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(b_true, b_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        boot_metrics["roc_auc"].append(roc_auc_score(b_true, b_prob))
        boot_metrics["pr_auc"].append(average_precision_score(b_true, b_prob))
        boot_metrics["f1"].append(f1_score(b_true, b_pred, zero_division=0))
        boot_metrics["balanced_accuracy"].append(balanced_accuracy_score(b_true, b_pred))
        boot_metrics["precision"].append(precision_score(b_true, b_pred, zero_division=0))
        boot_metrics["recall"].append(recall_score(b_true, b_pred, zero_division=0))
        boot_metrics["specificity"].append(spec)
        
    ci_results = {}
    for k, v in boot_metrics.items():
        ci_results[k] = {
            "mean": float(np.mean(v)),
            "ci_95_low": float(np.percentile(v, 2.5)),
            "ci_95_high": float(np.percentile(v, 97.5))
        }
    return ci_results

# ---------------------------------------------------------
# PyTorch MLP Definition
# ---------------------------------------------------------
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

def train_mlp(X_train, y_train, X_val, y_val, epochs=40, lr=1e-3, weight_decay=1e-2, batch_size=32, seed=42):
    torch.manual_seed(seed)
    model = LightweightMLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    
    best_val_auc = -1.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(y_val, val_probs)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    model.load_state_dict(best_state)
    return model

print("Setup completed.")
