# Workstream A: Phase 4A.1 Hybrid Evaluation Reconciliation & Independent Verification

**Project:** Python SQL Injection Detection (CWE-89)  
**Execution Timestamp:** 2026-11-09 06:44:08 UTC  
**Git Commit:** `24b7d6eecaac8daa0759309c4caeec3602811da3`  
**Artifact Directory:** `experiments/sqli_probe_v3_full_mistral/expanded_v2/hybrid_verification/`  

---

## Executive Summary

Phase 4A.1 was commissioned to independently audit, reconcile, and mathematically verify the results from Phase 4A directly from per-sample predictions.

### Verdict on Prior Phase 4A Findings:
**FULLY REPRODUCED AND VERIFIED.**
Every metric, matrix count, and percentage reported in Phase 4A has been verified bit-for-bit from underlying per-sample records. All apparent numerical variations between tables have been reconciled down to unambiguous mathematical definitions.

---

## 1. Verified Neural Baseline ($N = 695$)

- **Total Test Samples:** 695 (203 Vulnerable, 492 Safe)
- **Neural Confusion Matrix at $\tau = 0.66$:**
  $$\text{TP} = 195, \quad \text{FP} = 244, \quad \text{TN} = 248, \quad \text{FN} = 8$$
- **Precision:** $44.42\%$ (195 / 439)
- **Recall:** $96.06\%$ (195 / 203)
- **Specificity:** $50.41\%$ (248 / 492)
- **$F_1$ Score:** $0.6075$
- **ROC-AUC:** $0.8977$
- **Average Precision:** $0.7456$

Every prediction maps 1:1 to its test sample ID. There are zero duplicated or missing IDs.

---

## 2. Reconciled Status Counts & Discrepancy Resolution

The audit resolved all surface discrepancies:

| Evaluation Dimension | Metric / Column | Value | Mathematical Definition |
| :--- | :--- | :---: | :--- |
| **Conservative Cross-Tabulation** | Uncertain Samples | **42** | Count of samples where `conservative_prediction == 'UNCERTAIN'` (Resolved subset: $695 - 42 = 653$). |
| **Conservative Completeness** | Partial Snippets | **44** | Count of snippets where `conservative_status == 'PARTIAL'` ($651 	ext{ Complete} + 44 	ext{ Partial} = 695$). Exactly 2 snippets had an established vulnerable sink but auxiliary unresolvable depth in a secondary function. |
| **Operational Classifier** | Uncertain Outputs | **16** | Samples where `operational_prediction == 'UNCERTAIN'` (Resolved subset: $695 - 16 = 679$). Operational fallback resolves 26 borderline cases as safe based on absence of execution sinks. |

---

## 3. Verified Neural vs. Static Cross-Tabulation

Reconstructed directly from per-sample predictions:

| Neural Outcome ($\tau=0.66$) | Confirmed Safe (Static) | Confirmed Vulnerable (Static) | Uncertain / Unresolved (Static) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **False Positives (FP)** | **135 (55.33%)** | 98 (40.16%) | 11 (4.51%) | **244** |
| **True Positives (TP)** | **6 (3.08%)** | **181 (92.82%)** | 8 (4.10%) | **195** |
| **False Negatives (FN)** | 2 (25.00%) | **4 (50.00%)** | 2 (25.00%) | **8** |
| **True Negatives (TN)** | 212 (85.48%) | 15 (6.05%) | 21 (8.47%) | **248** |
| **Total** | **355 (51.08%)** | **298 (42.88%)** | **42 (6.04%)** | **695** |

### Verified Core Answers:
1. **False-Positive Elimination:** Static analysis definitively eliminates **135 out of 244 neural false positives (55.33%)**.
2. **True-Positive Suppression Risk:** Only **6 out of 195 true positives (3.08%)** are incorrectly cleared as safe.
3. **False-Negative Detection:** Static analysis independently detects **4 out of 8 false negatives (50.00%)** completely missed by the neural probe.

---

## 4. Verified Candidate Hybrid Policies

| Policy | Confirmed Alerts | Likely Safe | Uncertain (Triage) | False Positives | False Negatives | Precision | Automated Recall | Total Coverage (incl. Triage) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy A: Neural-Only ($\tau=0.66$)** | 439 | 256 | 0 | 244 | 8 | 44.42% | 96.06% | 96.06% |
| **Policy B: Static-Only (Operational)** | 293 | 386 | 16 | 109 | 8 | 62.80% | 90.64% | 96.06% |
| **Policy C: Evidence-Confirmation** | 279 | 214 | 202 | **98** | **2** | **64.87%** | 89.16% | 99.01% |
| **Policy D: High-Recall Hybrid Triage** | 298 | 378 | **19** | 113 | 10 | 62.08% | **91.13%** | **95.07%** |

---

## 5. Metadata and Provenance Signatures

All verified outputs and cryptographic hashes have been registered in:  
`experiments/sqli_probe_v3_full_mistral/expanded_v2/hybrid_verification/artifact_hashes.json`
