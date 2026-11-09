import os
import json
import time
import re
import numpy as np
import pandas as pd
import joblib
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
BASE_DIR = "experiments/sqli_probe_v3_full_mistral"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
PRED_DIR = os.path.join(BASE_DIR, "predictions")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = os.path.join(BASE_DIR, "data/e2_excel")
EMB_DIR = os.path.join(BASE_DIR, "embeddings/full_e2")

CANDIDATE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
POOLING_STRATEGIES = ["mean", "sink"]

# ---------------------------------------------------------
# 1. Load Labels, Sample IDs, and Raw Text Data
# ---------------------------------------------------------
print("=" * 80)
print("PROBE V3 PHASE 4: TRAINING & EVALUATION ON FULL MISTRAL-7B REPRESENTATIONS")
print("=" * 80)

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

print(f"Loaded Track E2 Splits:")
print(f"  Train: {len(y_train)} samples ({np.sum(y_train==1)} vuln, {np.sum(y_train==0)} safe)")
print(f"  Val:   {len(y_val)} samples ({np.sum(y_val==1)} vuln, {np.sum(y_val==0)} safe)")
print(f"  Test:  {len(y_test)} samples ({np.sum(y_test==1)} vuln, {np.sum(y_test==0)} safe)")

# ---------------------------------------------------------
# 2. Metric Computation & Bootstrap CI
# ---------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def find_optimal_threshold(y_true, y_prob):
    best_thresh = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)
    return best_thresh, best_f1

def bootstrap_ci(y_true, y_prob, threshold, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_metrics = {k: [] for k in ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "precision", "recall", "specificity", "accuracy"]}
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        b_true = y_true[idx]
        b_prob = y_prob[idx]
        if len(np.unique(b_true)) < 2:
            continue
        b_pred = (b_prob >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(b_true, b_pred, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        boot_metrics["roc_auc"].append(roc_auc_score(b_true, b_prob))
        boot_metrics["pr_auc"].append(average_precision_score(b_true, b_prob))
        boot_metrics["f1"].append(f1_score(b_true, b_pred, zero_division=0))
        boot_metrics["balanced_accuracy"].append(balanced_accuracy_score(b_true, b_pred))
        boot_metrics["precision"].append(precision_score(b_true, b_pred, zero_division=0))
        boot_metrics["recall"].append(recall_score(b_true, b_pred, zero_division=0))
        boot_metrics["specificity"].append(spec)
        boot_metrics["accuracy"].append(accuracy_score(b_true, b_pred))
        
    ci_results = {}
    for k, v in boot_metrics.items():
        ci_results[k] = {
            "mean": float(np.mean(v)),
            "ci_95_low": float(np.percentile(v, 2.5)),
            "ci_95_high": float(np.percentile(v, 97.5))
        }
    return ci_results

# ---------------------------------------------------------
# 3. Model Architectures
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

def train_mlp_model(X_tr, y_tr, X_va, y_va, epochs=35, lr=1e-3, weight_decay=1e-2, batch_size=32, seed=42):
    torch.manual_seed(seed)
    model = LightweightMLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    
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
            val_logits = model(X_va_t)
            val_probs = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(y_va, val_probs)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    model.load_state_dict(best_state)
    return model

# ---------------------------------------------------------
# 4. Lexical Regex Baseline Implementation
# ---------------------------------------------------------
SQLI_REGEX_PATTERNS = [
    r"(?i)(execute|executemany|raw|cursor\.\w+)\s*\(\s*(f[\"'].*?[\"']|[\"'].*?%s.*?[\"']\s*%)", # f-string or % formatting inside execute
    r"(?i)(execute|executemany|raw|cursor\.\w+)\s*\(\s*[\"'].*?[\"']\s*\+\s*", # string concatenation inside execute
    r"(?i)(execute|executemany|raw|cursor\.\w+)\s*\(\s*[\"'].*?\.format\(", # .format() inside execute
    r"(?i)SELECT\s+.*?\s+FROM\s+.*?\s+WHERE\s+.*?(\+|%|\.format|f[\"'])", # SQL query concatenation in string
    r"(?i)INSERT\s+INTO\s+.*?\s+VALUES\s*\(.*?(\+|%|\.format|f[\"'])",
    r"(?i)UPDATE\s+.*?\s+SET\s+.*?(\+|%|\.format|f[\"'])",
    r"(?i)DELETE\s+FROM\s+.*?\s+WHERE\s+.*?(\+|%|\.format|f[\"'])",
]
compiled_sqli_regex = [re.compile(p) for p in SQLI_REGEX_PATTERNS]

def predict_regex(code_list):
    probs = []
    preds = []
    for code in code_list:
        matched = any(p.search(code) is not None for p in compiled_sqli_regex)
        preds.append(1 if matched else 0)
        probs.append(0.95 if matched else 0.05)
    return np.array(preds), np.array(probs)

# ---------------------------------------------------------
# 5. PHASE 4A: VALIDATION LAYER & POOLING SWEEP
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 4A: SYSTEMATIC VALIDATION LAYER & POOLING SWEEP")
print("=" * 80)

sweep_results = []
trained_models = {}
val_predictions = {}

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

for pool in POOLING_STRATEGIES:
    for layer in CANDIDATE_LAYERS:
        t0 = time.time()
        # Load embeddings
        X_tr_raw = np.load(os.path.join(EMB_DIR, "train", pool, f"layer_{layer}.npy"))
        X_va_raw = np.load(os.path.join(EMB_DIR, "val", pool, f"layer_{layer}.npy"))
        
        # Fit scaler exclusively on train
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)
        
        config_key_base = f"{pool}_layer_{layer}"
        
        # --- Model 1: Logistic Regression ---
        best_lr_auc = -1.0
        best_lr_model = None
        best_lr_C = None
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=1000, random_state=SEED, solver='lbfgs')
            clf.fit(X_tr, y_train)
            p_va = clf.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_val, p_va)
            if auc > best_lr_auc:
                best_lr_auc = auc
                best_lr_model = clf
                best_lr_C = c
                
        p_lr_va = best_lr_model.predict_proba(X_va)[:, 1]
        t_lr, f1_lr = find_optimal_threshold(y_val, p_lr_va)
        metrics_lr = compute_metrics(y_val, (p_lr_va >= t_lr).astype(int), p_lr_va)
        metrics_lr.update({
            "model": "LogisticRegression", "layer": layer, "pooling": pool,
            "best_hyperparam": f"C={best_lr_C}", "threshold": t_lr
        })
        sweep_results.append(metrics_lr)
        trained_models[f"LR_{config_key_base}"] = (best_lr_model, scaler, t_lr)
        val_predictions[f"LR_{config_key_base}"] = p_lr_va
        
        # --- Model 2: Linear SVM ---
        best_svm_auc = -1.0
        best_svm_model = None
        best_svm_C = None
        for c in C_GRID:
            svm = LinearSVC(C=c, dual=False, max_iter=2000, random_state=SEED)
            svm.fit(X_tr, y_train)
            df_va = svm.decision_function(X_va)
            # Platt scaling via sigmoid on decision function
            p_va = 1.0 / (1.0 + np.exp(-df_va))
            auc = roc_auc_score(y_val, p_va)
            if auc > best_svm_auc:
                best_svm_auc = auc
                best_svm_model = svm
                best_svm_C = c
                
        df_svm_va = best_svm_model.decision_function(X_va)
        p_svm_va = 1.0 / (1.0 + np.exp(-df_svm_va))
        t_svm, f1_svm = find_optimal_threshold(y_val, p_svm_va)
        metrics_svm = compute_metrics(y_val, (p_svm_va >= t_svm).astype(int), p_svm_va)
        metrics_svm.update({
            "model": "LinearSVM", "layer": layer, "pooling": pool,
            "best_hyperparam": f"C={best_svm_C}", "threshold": t_svm
        })
        sweep_results.append(metrics_svm)
        trained_models[f"SVM_{config_key_base}"] = (best_svm_model, scaler, t_svm)
        val_predictions[f"SVM_{config_key_base}"] = p_svm_va
        
        # --- Model 3: Lightweight MLP ---
        mlp_model = train_mlp_model(X_tr, y_train, X_va, y_val, epochs=35, seed=SEED)
        mlp_model.eval()
        with torch.no_grad():
            p_mlp_va = torch.sigmoid(mlp_model(torch.tensor(X_va, dtype=torch.float32))).numpy()
        t_mlp, f1_mlp = find_optimal_threshold(y_val, p_mlp_va)
        metrics_mlp = compute_metrics(y_val, (p_mlp_va >= t_mlp).astype(int), p_mlp_va)
        metrics_mlp.update({
            "model": "LightweightMLP", "layer": layer, "pooling": pool,
            "best_hyperparam": "lr=1e-3,wd=1e-2,epochs=35", "threshold": t_mlp
        })
        sweep_results.append(metrics_mlp)
        trained_models[f"MLP_{config_key_base}"] = (mlp_model, scaler, t_mlp)
        val_predictions[f"MLP_{config_key_base}"] = p_mlp_va
        
        elapsed = time.time() - t0
        print(f"[{pool.upper():4s} | Layer {layer:2d}] LR AUC: {metrics_lr['roc_auc']:.4f} (F1: {metrics_lr['f1']:.4f}) | "
              f"SVM AUC: {metrics_svm['roc_auc']:.4f} (F1: {metrics_svm['f1']:.4f}) | "
              f"MLP AUC: {metrics_mlp['roc_auc']:.4f} (F1: {metrics_mlp['f1']:.4f}) | {elapsed:.1f}s")

df_sweep = pd.DataFrame(sweep_results)
df_sweep.to_csv(os.path.join(RESULTS_DIR, "validation_sweep_results.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "validation_sweep_results.json"), "w") as f:
    json.dump(sweep_results, f, indent=2)

# Save validation predictions
df_val_preds = pd.DataFrame(val_predictions)
df_val_preds.insert(0, "sample_id", val_ids)
df_val_preds.insert(1, "true_label", y_val)
df_val_preds.to_csv(os.path.join(PRED_DIR, "validation_predictions.csv"), index=False)

# ---------------------------------------------------------
# 6. Model Selection from Validation Results
# ---------------------------------------------------------
# Find optimal configuration strictly on validation data
prefix_map = {
    "LogisticRegression": "LR",
    "LinearSVM": "SVM",
    "LightweightMLP": "MLP"
}

best_row = df_sweep.loc[df_sweep["roc_auc"].idxmax()]
best_model_name = best_row["model"]
best_layer = int(best_row["layer"])
best_pooling = str(best_row["pooling"])
best_ck_id = f"{prefix_map[best_model_name]}_{best_pooling}_layer_{best_layer}"

print("\n" + "=" * 80)
print("VALIDATION SELECTION RESULTS:")
print(f"  Best Validation Model:    {best_model_name}")
print(f"  Best Validation Layer:    Layer {best_layer}")
print(f"  Best Validation Pooling:  {best_pooling}")
print(f"  Best Validation ROC-AUC:  {best_row['roc_auc']:.4f}")
print(f"  Best Validation F1:       {best_row['f1']:.4f} (Threshold: {best_row['threshold']:.2f})")
print(f"  Reference Configuration:  Layer 16 (Mean & Sink)")
print("=" * 80)

# ---------------------------------------------------------
# 7. PHASE 4B: BASELINES & CONTROLS (Fitted on Train, Validated)
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 4B: BASELINES & CONTROLS EVALUATION")
print("=" * 80)

# Baseline 1: TF-IDF + Logistic Regression
tfidf = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2))
X_tr_tfidf = tfidf.fit_transform(train_code)
X_va_tfidf = tfidf.transform(val_code)
X_te_tfidf = tfidf.transform(test_code)

clf_tfidf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
clf_tfidf.fit(X_tr_tfidf, y_train)
p_tfidf_va = clf_tfidf.predict_proba(X_va_tfidf)[:, 1]
t_tfidf, f1_tfidf = find_optimal_threshold(y_val, p_tfidf_va)
metrics_tfidf_va = compute_metrics(y_val, (p_tfidf_va >= t_tfidf).astype(int), p_tfidf_va)
print(f"Baseline: TF-IDF + LR | Val ROC-AUC: {metrics_tfidf_va['roc_auc']:.4f} | Val F1: {metrics_tfidf_va['f1']:.4f}")

# Baseline 2: Lexical Regex Heuristic
p_regex_pred_va, p_regex_prob_va = predict_regex(val_code)
metrics_regex_va = compute_metrics(y_val, p_regex_pred_va, p_regex_prob_va)
print(f"Baseline: Lexical Regex | Val ROC-AUC: {metrics_regex_va['roc_auc']:.4f} | Val F1: {metrics_regex_va['f1']:.4f}")

# Baseline 3: Majority Class Classifier
majority_class = int(np.bincount(y_train).argmax()) # 1 (vulnerable: 385/700)
p_maj_va = np.full_like(y_val, float(majority_class), dtype=float)
metrics_maj_va = compute_metrics(y_val, np.full_like(y_val, majority_class), p_maj_va)
print(f"Baseline: Majority Class | Val ROC-AUC: {metrics_maj_va['roc_auc']:.4f} | Val F1: {metrics_maj_va['f1']:.4f}")

# Control 1: Shuffled-Label Controls (5 random seeds)
# Shuffles train labels only, keeping representation and validation split completely intact
shuffled_seeds = [42, 43, 44, 45, 46]
shuffled_val_metrics = []
# Use best configuration representations
X_tr_best_raw = np.load(os.path.join(EMB_DIR, "train", best_pooling, f"layer_{best_layer}.npy"))
X_va_best_raw = np.load(os.path.join(EMB_DIR, "val", best_pooling, f"layer_{best_layer}.npy"))
scaler_best = StandardScaler()
X_tr_best = scaler_best.fit_transform(X_tr_best_raw)
X_va_best = scaler_best.transform(X_va_best_raw)

for s in shuffled_seeds:
    rng = np.random.RandomState(s)
    y_tr_shuffled = rng.permutation(y_train)
    clf_shuffled = LogisticRegression(C=1.0, max_iter=1000, random_state=s)
    clf_shuffled.fit(X_tr_best, y_tr_shuffled)
    p_shuf_va = clf_shuffled.predict_proba(X_va_best)[:, 1]
    m_shuf = compute_metrics(y_val, (p_shuf_va >= 0.5).astype(int), p_shuf_va)
    m_shuf["seed"] = s
    shuffled_val_metrics.append(m_shuf)

mean_shuf_auc = np.mean([m["roc_auc"] for m in shuffled_val_metrics])
print(f"Control: Shuffled-Label (5 seeds) | Mean Val ROC-AUC: {mean_shuf_auc:.4f} (Expected ~0.5000)")

# Control 2: Random Gaussian Feature Vector Control
# Replace representations with pure N(0, I) Gaussian vectors
rng_gauss = np.random.RandomState(SEED)
X_tr_gauss = rng_gauss.randn(len(y_train), 4096)
X_va_gauss = rng_gauss.randn(len(y_val), 4096)
clf_gauss = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
clf_gauss.fit(X_tr_gauss, y_train)
p_gauss_va = clf_gauss.predict_proba(X_va_gauss)[:, 1]
metrics_gauss_va = compute_metrics(y_val, (p_gauss_va >= 0.5).astype(int), p_gauss_va)
print(f"Control: Random Gaussian Vectors | Val ROC-AUC: {metrics_gauss_va['roc_auc']:.4f} (Expected ~0.5000)")

# ---------------------------------------------------------
# 8. PHASE 4C: STRICT FINAL TEST SET EVALUATION
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 4C: STRICT ONE-SHOT TEST SET EVALUATION")
print("=" * 80)

test_results = {}
test_predictions_dict = {}

# Evaluate configurations on Test:
# 1. Optimal Configuration (selected via validation)
# 2. Reference Configuration: Layer 16 Mean
# 3. Reference Configuration: Layer 16 Sink
# 4. TF-IDF + Logistic Regression
# 5. Lexical Regex
# 6. Majority Class
# 7. Shuffled-Label Control (seed 42)
# 8. Random Vector Control

eval_configs = [
    ("Best_Validation_Selected", best_model_name, best_pooling, best_layer, best_ck_id),
    ("Reference_Layer16_Mean", "LogisticRegression", "mean", 16, "LR_mean_layer_16"),
    ("Reference_Layer16_Sink", "LogisticRegression", "sink", 16, "LR_sink_layer_16"),
]

for label_name, m_type, pool, lyr, ck_id in eval_configs:
    model_obj, sc, thresh = trained_models[ck_id]
    X_te_raw = np.load(os.path.join(EMB_DIR, "test", pool, f"layer_{lyr}.npy"))
    X_te = sc.transform(X_te_raw)
    
    if m_type == "LogisticRegression":
        p_te = model_obj.predict_proba(X_te)[:, 1]
    elif m_type == "LinearSVM":
        df_te = model_obj.decision_function(X_te)
        p_te = 1.0 / (1.0 + np.exp(-df_te))
    elif m_type == "LightweightMLP":
        model_obj.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(model_obj(torch.tensor(X_te, dtype=torch.float32))).numpy()
            
    pred_te = (p_te >= thresh).astype(int)
    m_test = compute_metrics(y_test, pred_te, p_te)
    ci_test = bootstrap_ci(y_test, p_te, thresh, n_bootstraps=1000, seed=SEED)
    m_test["bootstrap_ci_95"] = ci_test
    m_test["threshold"] = thresh
    m_test["model"] = m_type
    m_test["layer"] = lyr
    m_test["pooling"] = pool
    
    test_results[label_name] = m_test
    test_predictions_dict[label_name] = p_te
    print(f"[{label_name}] ROC-AUC: {m_test['roc_auc']:.4f} (95% CI: [{ci_test['roc_auc']['ci_95_low']:.4f}, {ci_test['roc_auc']['ci_95_high']:.4f}]) | "
          f"PR-AUC: {m_test['pr_auc']:.4f} | F1: {m_test['f1']:.4f} | Rec: {m_test['recall']:.4f} | Spec: {m_test['specificity']:.4f}")

# Evaluate Baseline 1: TF-IDF on Test
p_tfidf_te = clf_tfidf.predict_proba(X_te_tfidf)[:, 1]
pred_tfidf_te = (p_tfidf_te >= t_tfidf).astype(int)
m_tfidf_te = compute_metrics(y_test, pred_tfidf_te, p_tfidf_te)
ci_tfidf_te = bootstrap_ci(y_test, p_tfidf_te, t_tfidf, n_bootstraps=1000, seed=SEED)
m_tfidf_te["bootstrap_ci_95"] = ci_tfidf_te
m_tfidf_te["threshold"] = t_tfidf
test_results["Baseline_TFIDF_LR"] = m_tfidf_te
test_predictions_dict["Baseline_TFIDF_LR"] = p_tfidf_te
print(f"[Baseline_TFIDF_LR] ROC-AUC: {m_tfidf_te['roc_auc']:.4f} (95% CI: [{ci_tfidf_te['roc_auc']['ci_95_low']:.4f}, {ci_tfidf_te['roc_auc']['ci_95_high']:.4f}]) | "
      f"PR-AUC: {m_tfidf_te['pr_auc']:.4f} | F1: {m_tfidf_te['f1']:.4f} | Rec: {m_tfidf_te['recall']:.4f} | Spec: {m_tfidf_te['specificity']:.4f}")

# Evaluate Baseline 2: Lexical Regex on Test
p_regex_pred_te, p_regex_prob_te = predict_regex(test_code)
m_regex_te = compute_metrics(y_test, p_regex_pred_te, p_regex_prob_te)
ci_regex_te = bootstrap_ci(y_test, p_regex_prob_te, 0.5, n_bootstraps=1000, seed=SEED)
m_regex_te["bootstrap_ci_95"] = ci_regex_te
m_regex_te["threshold"] = 0.5
test_results["Baseline_Lexical_Regex"] = m_regex_te
test_predictions_dict["Baseline_Lexical_Regex"] = p_regex_prob_te
print(f"[Baseline_Lexical_Regex] ROC-AUC: {m_regex_te['roc_auc']:.4f} | PR-AUC: {m_regex_te['pr_auc']:.4f} | F1: {m_regex_te['f1']:.4f} | Rec: {m_regex_te['recall']:.4f} | Spec: {m_regex_te['specificity']:.4f}")

# Evaluate Baseline 3: Majority Class on Test
p_maj_te = np.full_like(y_test, float(majority_class), dtype=float)
pred_maj_te = np.full_like(y_test, majority_class)
m_maj_te = compute_metrics(y_test, pred_maj_te, p_maj_te)
test_results["Baseline_Majority_Class"] = m_maj_te
test_predictions_dict["Baseline_Majority_Class"] = p_maj_te
print(f"[Baseline_Majority_Class] ROC-AUC: {m_maj_te['roc_auc']:.4f} | F1: {m_maj_te['f1']:.4f}")

# Evaluate Control 1: Shuffled-Label Control on Test
X_te_best_raw = np.load(os.path.join(EMB_DIR, "test", best_pooling, f"layer_{best_layer}.npy"))
X_te_best = scaler_best.transform(X_te_best_raw)
clf_shuf_42 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf_shuf_42.fit(X_tr_best, np.random.RandomState(42).permutation(y_train))
p_shuf_te = clf_shuf_42.predict_proba(X_te_best)[:, 1]
m_shuf_te = compute_metrics(y_test, (p_shuf_te >= 0.5).astype(int), p_shuf_te)
test_results["Control_Shuffled_Labels_Seed42"] = m_shuf_te
test_predictions_dict["Control_Shuffled_Labels_Seed42"] = p_shuf_te
print(f"[Control_Shuffled_Labels_Seed42] ROC-AUC: {m_shuf_te['roc_auc']:.4f} | F1: {m_shuf_te['f1']:.4f}")

# Evaluate Control 2: Random Vector Control on Test
X_te_gauss = rng_gauss.randn(len(y_test), 4096)
p_gauss_te = clf_gauss.predict_proba(X_te_gauss)[:, 1]
m_gauss_te = compute_metrics(y_test, (p_gauss_te >= 0.5).astype(int), p_gauss_te)
test_results["Control_Random_Gaussian_Vectors"] = m_gauss_te
test_predictions_dict["Control_Random_Gaussian_Vectors"] = p_gauss_te
print(f"[Control_Random_Gaussian_Vectors] ROC-AUC: {m_gauss_te['roc_auc']:.4f} | F1: {m_gauss_te['f1']:.4f}")

# Save Test Results and Predictions
with open(os.path.join(RESULTS_DIR, "test_evaluation_results.json"), "w") as f:
    json.dump(test_results, f, indent=2)

df_test_preds = pd.DataFrame(test_predictions_dict)
df_test_preds.insert(0, "sample_id", test_ids)
df_test_preds.insert(1, "true_label", y_test)
df_test_preds.to_csv(os.path.join(PRED_DIR, "test_predictions.csv"), index=False)

# ---------------------------------------------------------
# 9. PHASE 4D: DETAILED ERROR ANALYSIS ON TEST SET
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 4D: DETAILED TEST ERROR ANALYSIS")
print("=" * 80)

best_p_te = test_predictions_dict["Best_Validation_Selected"]
best_pred_te = (best_p_te >= test_results["Best_Validation_Selected"]["threshold"]).astype(int)

df_test_meta = pd.DataFrame(test_records)
df_test_meta["label"] = df_test_meta["label"].astype(int)
df_test_meta["predicted_label"] = best_pred_te
df_test_meta["predicted_prob"] = best_p_te
df_test_meta["is_correct"] = (df_test_meta["predicted_label"] == df_test_meta["label"])

# Analyze False Positives and False Negatives
fp_samples = df_test_meta[(df_test_meta["label"] == 0) & (df_test_meta["predicted_label"] == 1)]
fn_samples = df_test_meta[(df_test_meta["label"] == 1) & (df_test_meta["predicted_label"] == 0)]

print(f"Total Test Samples: {len(df_test_meta)}")
print(f"Total False Positives: {len(fp_samples)}")
print(f"Total False Negatives: {len(fn_samples)}")

# Breakdown by case_type
case_type_breakdown = {}
for ct, group in df_test_meta.groupby("case_type"):
    y_g = group["label"].astype(int).values
    p_g = group["predicted_prob"].values
    pred_g = group["predicted_label"].values
    acc = accuracy_score(y_g, pred_g)
    case_type_breakdown[ct] = {
        "n_samples": len(group),
        "accuracy": float(acc),
        "false_positives": int(np.sum((y_g == 0) & (pred_g == 1))),
        "false_negatives": int(np.sum((y_g == 1) & (pred_g == 0)))
    }
    print(f"Case Type: {ct:20s} | N={len(group):3d} | Acc: {acc:.4f} | FP={case_type_breakdown[ct]['false_positives']}, FN={case_type_breakdown[ct]['false_negatives']}")

# Breakdown by db_api
db_api_breakdown = {}
for api, group in df_test_meta.groupby("db_api"):
    y_g = group["label"].astype(int).values
    pred_g = group["predicted_label"].values
    acc = accuracy_score(y_g, pred_g)
    db_api_breakdown[api] = {
        "n_samples": len(group),
        "accuracy": float(acc),
        "false_positives": int(np.sum((y_g == 0) & (pred_g == 1))),
        "false_negatives": int(np.sum((y_g == 1) & (pred_g == 0)))
    }

error_analysis_data = {
    "total_test_samples": len(df_test_meta),
    "total_fp": len(fp_samples),
    "total_fn": len(fn_samples),
    "fp_sample_ids": fp_samples["sample_id"].tolist(),
    "fn_sample_ids": fn_samples["sample_id"].tolist(),
    "case_type_breakdown": case_type_breakdown,
    "db_api_breakdown": db_api_breakdown
}

with open(os.path.join(RESULTS_DIR, "error_analysis.json"), "w") as f:
    json.dump(error_analysis_data, f, indent=2)

# Save sample-level error analysis CSV
df_test_meta[["sample_id", "label", "predicted_label", "predicted_prob", "is_correct", "case_type", "db_api", "query_style"]].to_csv(
    os.path.join(RESULTS_DIR, "test_sample_error_analysis.csv"), index=False
)

# ---------------------------------------------------------
# 10. Save Model Checkpoints
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING MODEL CHECKPOINTS")
print("=" * 80)

# Save best model
best_model_obj, best_sc, best_t = trained_models[best_ck_id]
if isinstance(best_model_obj, nn.Module):
    torch.save(best_model_obj.state_dict(), os.path.join(CHECKPOINT_DIR, f"{best_ck_id}.pt"))
else:
    joblib.dump(best_model_obj, os.path.join(CHECKPOINT_DIR, f"{best_ck_id}.joblib"))
joblib.dump(best_sc, os.path.join(CHECKPOINT_DIR, f"{best_ck_id}_scaler.joblib"))

# Save reference models (Layer 16 Mean and Sink)
for ref_key in ["LR_mean_layer_16", "LR_sink_layer_16"]:
    ref_obj, ref_sc, ref_t = trained_models[ref_key]
    joblib.dump(ref_obj, os.path.join(CHECKPOINT_DIR, f"{ref_key}.joblib"))
    joblib.dump(ref_sc, os.path.join(CHECKPOINT_DIR, f"{ref_key}_scaler.joblib"))

# Save TF-IDF Baseline model
joblib.dump(clf_tfidf, os.path.join(CHECKPOINT_DIR, "baseline_tfidf_lr.joblib"))
joblib.dump(tfidf, os.path.join(CHECKPOINT_DIR, "baseline_tfidf_vectorizer.joblib"))

print("All checkpoints saved successfully!")
print("=" * 80)
print("TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY.")
print("=" * 80)

