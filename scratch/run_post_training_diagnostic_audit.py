import os
import json
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression

BASE_DIR = "experiments/sqli_probe_v3_full_mistral"
DATA_DIR = os.path.join(BASE_DIR, "data/e2_excel")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRED_DIR = os.path.join(BASE_DIR, "predictions")

print("=" * 80)
print("STARTING SCIENTIFIC DIAGNOSTIC AUDIT OF PROBE V3")
print("=" * 80)

# Load data splits
train_records = [json.loads(l) for l in open(os.path.join(DATA_DIR, "train.jsonl"))]
val_records = [json.loads(l) for l in open(os.path.join(DATA_DIR, "val.jsonl"))]
test_records = [json.loads(l) for l in open(os.path.join(DATA_DIR, "test.jsonl"))]

y_train = np.array([int(r["label"]) for r in train_records])
y_val = np.array([int(r["label"]) for r in val_records])
y_test = np.array([int(r["label"]) for r in test_records])

df_test_preds = pd.read_csv(os.path.join(PRED_DIR, "test_predictions.csv"))

audit_data = {}

# -----------------------------------------------------------------------------
# INVESTIGATION 1: RANDOM GAUSSIAN CONTROL MONTE CARLO PERMUTATION ANALYSIS
# -----------------------------------------------------------------------------
print("\n--- Running Investigation 1: Random Gaussian Control Audit ---")
# Check distribution of ROC-AUC under pure random Gaussian features (d=4096, N_tr=700, N_te=151)
n_runs = 200 # 200 independent full training runs of LR on Gaussian features
gauss_test_aucs = []
gauss_test_aps = []

for r in range(n_runs):
    rng = np.random.RandomState(1000 + r)
    X_tr_g = rng.randn(len(y_train), 4096)
    X_te_g = rng.randn(len(y_test), 4096)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=r)
    clf.fit(X_tr_g, y_train)
    p_te_g = clf.predict_proba(X_te_g)[:, 1]
    gauss_test_aucs.append(float(roc_auc_score(y_test, p_te_g)))
    gauss_test_aps.append(float(average_precision_score(y_test, p_te_g)))

orig_gauss_auc = float(roc_auc_score(y_test, df_test_preds["Control_Random_Gaussian_Vectors"]))
orig_gauss_ap = float(average_precision_score(y_test, df_test_preds["Control_Random_Gaussian_Vectors"]))

# Calculate empirical p-value for the original single-seed run
empirical_p_auc = float(np.mean([a >= orig_gauss_auc for a in gauss_test_aucs]))

audit_data["investigation_1_gaussian_control"] = {
    "single_seed_reported_auc": orig_gauss_auc,
    "single_seed_reported_ap": orig_gauss_ap,
    "monte_carlo_runs": n_runs,
    "monte_carlo_mean_auc": float(np.mean(gauss_test_aucs)),
    "monte_carlo_std_auc": float(np.std(gauss_test_aucs)),
    "monte_carlo_min_auc": float(np.min(gauss_test_aucs)),
    "monte_carlo_max_auc": float(np.max(gauss_test_aucs)),
    "monte_carlo_95_interval": [float(np.percentile(gauss_test_aucs, 2.5)), float(np.percentile(gauss_test_aucs, 97.5))],
    "monte_carlo_mean_ap": float(np.mean(gauss_test_aps)),
    "empirical_p_value_of_reported_auc": empirical_p_auc,
    "standard_error_theoretical": float(np.sqrt(1 / (12 * 83 * 68) * (83 + 68 + 1))), # Mann-Whitney SE ~ 0.047
}
print(f"Reported Gaussian Control AUC: {orig_gauss_auc:.4f}")
print(f"Monte Carlo 200 runs Mean AUC: {np.mean(gauss_test_aucs):.4f} +/- {np.std(gauss_test_aucs):.4f} (Range: [{np.min(gauss_test_aucs):.4f}, {np.max(gauss_test_aucs):.4f}])")
print(f"Empirical p-value of 0.6013 under null: {empirical_p_auc:.4f}")

# -----------------------------------------------------------------------------
# INVESTIGATION 2: SHUFFLED-LABEL CONTROLS (5 SEEDS)
# -----------------------------------------------------------------------------
print("\n--- Running Investigation 2: Shuffled-Label Control Audit ---")
# Check how shuffled labels behaved across 5 seeds on train -> val
# Also evaluate the 5 seeds on test to document null consistency
shuffled_test_aucs = []
shuffled_seeds = [42, 43, 44, 45, 46]

# Load best representations (Layer 12 Sink)
X_tr_best = np.load(os.path.join(BASE_DIR, "embeddings/full_e2/train/sink/layer_12.npy"))
X_te_best = np.load(os.path.join(BASE_DIR, "embeddings/full_e2/test/sink/layer_12.npy"))

for s in shuffled_seeds:
    rng = np.random.RandomState(s)
    y_tr_shuf = rng.permutation(y_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=s)
    clf.fit(X_tr_best, y_tr_shuf)
    p_te_shuf = clf.predict_proba(X_te_best)[:, 1]
    shuffled_test_aucs.append({
        "seed": s,
        "test_roc_auc": float(roc_auc_score(y_test, p_te_shuf)),
        "test_pr_auc": float(average_precision_score(y_test, p_te_shuf))
    })

audit_data["investigation_2_shuffled_labels"] = {
    "seeds_evaluated": shuffled_seeds,
    "shuffled_test_results": shuffled_test_aucs,
    "mean_test_auc": float(np.mean([x["test_roc_auc"] for x in shuffled_test_aucs])),
    "std_test_auc": float(np.std([x["test_roc_auc"] for x in shuffled_test_aucs]))
}
print(f"Shuffled-label 5-seed Mean Test AUC: {audit_data['investigation_2_shuffled_labels']['mean_test_auc']:.4f}")

# -----------------------------------------------------------------------------
# INVESTIGATION 3: DATASET LEAKAGE AUDIT ON TRACK E2
# -----------------------------------------------------------------------------
print("\n--- Running Investigation 3: Dataset Leakage Audit ---")
def normalize_code(c):
    c = re.sub(r'#.*', '', c)
    c = re.sub(r'""".*?"""', '', c, flags=re.DOTALL)
    c = re.sub(r"'''.*?'''", '', c, flags=re.DOTALL)
    c = re.sub(r'\s+', ' ', c).strip()
    return c

# 1. Exact duplicate code
tr_codes = [normalize_code(r["code"]) for r in train_records]
va_codes = [normalize_code(r["code"]) for r in val_records]
te_codes = [normalize_code(r["code"]) for r in test_records]

exact_dup_tr_va = set(tr_codes).intersection(set(va_codes))
exact_dup_tr_te = set(tr_codes).intersection(set(te_codes))
exact_dup_va_te = set(va_codes).intersection(set(te_codes))

# 2. Near duplicate (token Jaccard > 0.85)
def get_tokens(c):
    return set(re.findall(r'\b\w+\b', c.lower()))

tr_tokens = [get_tokens(c) for c in tr_codes]
te_tokens = [get_tokens(c) for c in te_codes]

near_dups_tr_te = []
for i, t_te in enumerate(te_tokens):
    for j, t_tr in enumerate(tr_tokens):
        inter = len(t_te.intersection(t_tr))
        union = len(t_te.union(t_tr))
        if union > 0 and (inter / union) >= 0.85:
            near_dups_tr_te.append({
                "test_id": test_records[i]["sample_id"],
                "train_id": train_records[j]["sample_id"],
                "jaccard": inter / union
            })

# 3. Fixed / vulnerable pairs leakage check
tr_orig_ids = set([r.get("original_sample_id") for r in train_records if r.get("original_sample_id")])
te_orig_ids = set([r.get("original_sample_id") for r in test_records if r.get("original_sample_id")])
orig_id_overlap = tr_orig_ids.intersection(te_orig_ids)

# 4. Metadata label correlations
df_all = pd.DataFrame(train_records + val_records + test_records)
df_all["label"] = df_all["label"].astype(int)
df_all["code_len"] = df_all["code"].apply(len)
df_all["n_lines"] = df_all["n_lines"].astype(float)

correlations = {}
for col in ["case_type", "db_api", "query_style", "dataset_source"]:
    if col in df_all.columns:
        ct = pd.crosstab(df_all[col], df_all["label"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        min_dim = min(ct.shape) - 1
        if min_dim > 0:
            cramers_v = float(np.sqrt(chi2 / (len(df_all) * min_dim)))
        else:
            cramers_v = 0.0
        correlations[col] = {
            "chi2": float(chi2), "p_value": float(p), "cramers_v": float(cramers_v)
        }

# Length correlation
r_len, p_len = stats.pointbiserialr(df_all["label"], df_all["code_len"])
correlations["code_len"] = {"point_biserial_r": float(r_len), "p_value": float(p_len)}

audit_data["investigation_3_leakage"] = {
    "exact_duplicates": {
        "train_val": len(exact_dup_tr_va),
        "train_test": len(exact_dup_tr_te),
        "val_test": len(exact_dup_va_te)
    },
    "near_duplicates_train_test_jaccard_gt_085": len(near_dups_tr_te),
    "original_sample_id_overlap_train_test": len(orig_id_overlap),
    "metadata_label_correlations": correlations
}
print(f"Exact duplicates Train-Test: {len(exact_dup_tr_te)}")
print(f"Near duplicates Train-Test: {len(near_dups_tr_te)}")
print(f"Metadata Cramers V (case_type): {correlations.get('case_type', {}).get('cramers_v', 0):.4f}")
print(f"Metadata Cramers V (db_api): {correlations.get('db_api', {}).get('cramers_v', 0):.4f}")
print(f"Length correlation: r={r_len:.4f}")

# -----------------------------------------------------------------------------
# INVESTIGATION 4: AST SINK POOLING BEHAVIOR
# -----------------------------------------------------------------------------
print("\n--- Running Investigation 4: AST Sink Pooling Audit ---")
import sys
sys.path.insert(0, BASE_DIR)
from extraction.ast_sink_extractor import extract_ast_sink_regions

all_recs = train_records + val_records + test_records
sink_stats = {
    "vuln": {"total": 0, "has_sink": 0, "fallback_to_mean": 0, "sink_lengths": []},
    "safe": {"total": 0, "has_sink": 0, "fallback_to_mean": 0, "sink_lengths": []}
}

for r in all_recs:
    lbl = "vuln" if int(r["label"]) == 1 else "safe"
    sink_stats[lbl]["total"] += 1
    sink_info = extract_ast_sink_regions(r["code"])
    if sink_info["has_ast_sink"]:
        sink_stats[lbl]["has_sink"] += 1
        sink_len = sum(end - start for start, end in sink_info["sink_spans"])
        sink_stats[lbl]["sink_lengths"].append(sink_len)
    else:
        sink_stats[lbl]["fallback_to_mean"] += 1

vuln_sink_rate = sink_stats["vuln"]["has_sink"] / sink_stats["vuln"]["total"]
safe_sink_rate = sink_stats["safe"]["has_sink"] / sink_stats["safe"]["total"]
mean_vuln_len = float(np.mean(sink_stats["vuln"]["sink_lengths"]))
mean_safe_len = float(np.mean(sink_stats["safe"]["sink_lengths"]))

audit_data["investigation_4_ast_pooling"] = {
    "vulnerable": {
        "total": sink_stats["vuln"]["total"],
        "sink_detected_count": sink_stats["vuln"]["has_sink"],
        "sink_coverage_rate": vuln_sink_rate,
        "mean_sink_char_length": mean_vuln_len
    },
    "safe": {
        "total": sink_stats["safe"]["total"],
        "sink_detected_count": sink_stats["safe"]["has_sink"],
        "sink_coverage_rate": safe_sink_rate,
        "mean_sink_char_length": mean_safe_len
    },
    "coverage_difference": float(abs(vuln_sink_rate - safe_sink_rate))
}
print(f"AST Sink Coverage - Vuln: {vuln_sink_rate*100:.2f}%, Safe: {safe_sink_rate*100:.2f}%")
print(f"Mean Sink Span - Vuln: {mean_vuln_len:.1f} chars, Safe: {mean_safe_len:.1f} chars")

# -----------------------------------------------------------------------------
# INVESTIGATION 5: PAIRED STATISTICAL COMPARISONS (MISTRAL VS TF-IDF)
# -----------------------------------------------------------------------------
print("\n--- Running Investigation 5: Paired Statistical Comparison ---")
p_mistral = df_test_preds["Best_Validation_Selected"].values
p_tfidf = df_test_preds["Baseline_TFIDF_LR"].values

t_mistral = 0.36
t_tfidf = 0.51

pred_mistral = (p_mistral >= t_mistral).astype(int)
pred_tfidf = (p_tfidf >= t_tfidf).astype(int)

# 1. Paired Bootstrap for Delta ROC-AUC
rng_boot = np.random.RandomState(42)
n_boot = 2000
delta_aucs = []
delta_aps = []

for _ in range(n_boot):
    idx = rng_boot.randint(0, len(y_test), len(y_test))
    b_true = y_test[idx]
    if len(np.unique(b_true)) < 2:
        continue
    auc_m = roc_auc_score(b_true, p_mistral[idx])
    auc_t = roc_auc_score(b_true, p_tfidf[idx])
    delta_aucs.append(auc_m - auc_t)
    
    ap_m = average_precision_score(b_true, p_mistral[idx])
    ap_t = average_precision_score(b_true, p_tfidf[idx])
    delta_aps.append(ap_m - ap_t)

delta_auc_mean = float(np.mean(delta_aucs))
delta_auc_ci = [float(np.percentile(delta_aucs, 2.5)), float(np.percentile(delta_aucs, 97.5))]
p_val_delta_auc = float(np.mean([d <= 0 for d in delta_aucs]) * 2) # Two-sided bootstrap p-value

delta_ap_mean = float(np.mean(delta_aps))
delta_ap_ci = [float(np.percentile(delta_aps, 2.5)), float(np.percentile(delta_aps, 97.5))]

# 2. McNemar's Test
# Contingency table:
# b: Mistral correct, TF-IDF wrong
# c: Mistral wrong, TF-IDF correct
correct_mistral = (pred_mistral == y_test)
correct_tfidf = (pred_tfidf == y_test)

b = int(np.sum(correct_mistral & (~correct_tfidf)))
c = int(np.sum((~correct_mistral) & correct_tfidf))
both_correct = int(np.sum(correct_mistral & correct_tfidf))
both_wrong = int(np.sum((~correct_mistral) & (~correct_tfidf)))

# McNemar test statistic with continuity correction: (|b - c| - 1)^2 / (b + c)
if (b + c) > 0:
    mcnemar_stat = float((abs(b - c) - 1)**2 / (b + c))
    mcnemar_p = float(1 - stats.chi2.cdf(mcnemar_stat, 1))
else:
    mcnemar_stat = 0.0
    mcnemar_p = 1.0

# 3. Brier Score & Calibration
brier_mistral = float(brier_score_loss(y_test, p_mistral))
brier_tfidf = float(brier_score_loss(y_test, p_tfidf))

audit_data["investigation_5_paired_comparison"] = {
    "delta_roc_auc": {
        "mean": delta_auc_mean,
        "ci_95": delta_auc_ci,
        "p_value_two_sided": p_val_delta_auc,
        "zero_in_ci": bool(delta_auc_ci[0] <= 0 <= delta_auc_ci[1])
    },
    "delta_pr_auc": {
        "mean": delta_ap_mean,
        "ci_95": delta_ap_ci,
        "zero_in_ci": bool(delta_ap_ci[0] <= 0 <= delta_ap_ci[1])
    },
    "mcnemar_test": {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mistral_only_correct": b,
        "tfidf_only_correct": c,
        "mcnemar_statistic": mcnemar_stat,
        "p_value": mcnemar_p,
        "statistically_significant": bool(mcnemar_p < 0.05)
    },
    "brier_scores": {
        "mistral": brier_mistral,
        "tfidf": brier_tfidf,
        "improvement_pct": float((brier_tfidf - brier_mistral) / brier_tfidf * 100)
    }
}
print(f"Paired Delta ROC-AUC (Mistral - TFIDF): {delta_auc_mean:+.4f} (95% CI: [{delta_auc_ci[0]:+.4f}, {delta_auc_ci[1]:+.4f}])")
print(f"Paired Delta PR-AUC: {delta_ap_mean:+.4f} (95% CI: [{delta_ap_ci[0]:+.4f}, {delta_ap_ci[1]:+.4f}])")
print(f"McNemar's test: b={b}, c={c}, p={mcnemar_p:.4f} (Significant: {mcnemar_p < 0.05})")
print(f"Brier score: Mistral {brier_mistral:.4f} vs TF-IDF {brier_tfidf:.4f} (Improvement: {audit_data['investigation_5_paired_comparison']['brier_scores']['improvement_pct']:.1f}%)")

# Save diagnostic data
with open(os.path.join(RESULTS_DIR, "post_training_diagnostic_audit.json"), "w") as f:
    json.dump(audit_data, f, indent=2)

print("\nSaved diagnostic results to post_training_diagnostic_audit.json")
print("=" * 80)
