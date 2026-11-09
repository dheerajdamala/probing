# Reconciliation of Status Counts and Classification Metrics

## 1. Discrepancy in Conservative Uncertainty Counts: 42 vs. 44

In the Phase 4A report, two separate counts were cited:
- **42 Uncertain samples** in the Neural/Static Cross-Tabulation and Resolved Subset ($N = 695 - 42 = 653$).
- **44 Partial samples** in the Static Completeness Summary ($651 	ext{ Complete} + 44 	ext{ Partial} = 695$).

### Root-Cause Resolution:
The divergence is caused by a subtle, valid distinction between **Prediction Outcome** versus **Analysis Completeness**:
- **Conservative Prediction Outcome (`conservative_prediction`):**  
  Classifies a sample into `VULNERABLE`, `SAFE`, or `UNCERTAIN`. A sample is classified as `VULNERABLE` if an explicit unsafe source-to-sink dataflow path is proved.
- **Analysis Completeness Status (`conservative_status`):**  
  Classifies whether all paths in the file were resolved (`COMPLETE`) or if any call edge/depth was cut off (`PARTIAL`).

There are **exactly 2 samples** in the test set where:
1. An explicit unsafe SQL injection path was unequivocally proved (`conservative_prediction = 'VULNERABLE'`).
2. An unrelated secondary function in the same snippet contained an unresolved call edge (`conservative_status = 'PARTIAL'`).

**Audit Verdict:**
- The cross-tabulation table correctly evaluated the **42 uncertain samples** that lacked definitive classification.
- The completeness table correctly reported all **44 snippets** that had partial call depth.
- Both tables are mathematically correct under their respective definitions.

---

## 2. Discrepancy in Operational vs. Conservative Uncertainty: 16 vs. 42

- Conservative Evaluation: **42 Uncertain samples** (Resolved subset $N = 653$).
- Operational Pipeline: **16 Uncertain samples** (Resolved subset $N = 679$).

### Root-Cause Resolution:
- The **Conservative Evaluator** strictly enforces that any code snippet with an unparameterized database sink and unresolved caller context must remain `UNCERTAIN`. It refuses to assume safety.
- The **Operational Pipeline Classifier** (`pipeline.py`) incorporates a heuristic fallback: if no tainted query construction patterns are detected at the immediate sink site, it operationally marks the snippet as `SAFE` even if backward caller tracing was incomplete.
- This resolves 26 borderline samples operationally, expanding coverage from $93.96\%$ to $97.70\%$, but increases false positives by 4 samples (from 109 to 113).
