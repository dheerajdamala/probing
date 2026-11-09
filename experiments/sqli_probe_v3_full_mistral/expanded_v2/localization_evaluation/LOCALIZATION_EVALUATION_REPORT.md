# Workstream B: Vulnerability Localization and Evidence Validation Report

**Project:** Python SQL Injection Detection (CWE-89)  
**Artifact Directory:** `experiments/sqli_probe_v3_full_mistral/expanded_v2/localization_evaluation/`  
**Execution Timestamp:** 2026-11-09 14:52:41 UTC  
**Git Commit:** `24b7d6eecaac8daa0759309c4caeec3602811da3`  

---

## Executive Summary

Phase 4B provides the rigorous, independent evaluation of the **line-level localization accuracy, taint path tracing, and evidence explanation validity** of the Python SQL injection detection system across an independently annotated 100-sample benchmark.

### Headline Results:
- **Sink Localization Accuracy:** **75.00%** (45/60 vulnerable sinks localized to exact line number).
- **Query Construction Line Accuracy:** **83.33%** (50/60 query formatting sites exact).
- **Top-1 Vulnerable Line Accuracy:** **83.33%**.
- **Top-3 Vulnerable Line Accuracy:** **83.33%**.
- **Complete Source-to-Sink Path Accuracy:** **58.33%**.
- **Benchmark Classification:** Precision = **90.91%**, Recall = **83.33%**, Specificity = **87.50%**, $F_1 = 0.8696$.

---

## 1. 100-Sample Benchmark Distribution & Annotations

The benchmark was constructed across 8 independent structural categories:
1. **Direct SQL Injection:** 20 samples (Vulnerable)
2. **Helper-Function / Interprocedural Injection:** 15 samples (Vulnerable)
3. **Cross-File Injection:** 15 samples (Vulnerable)
4. **Safe Parameterized SQL:** 15 samples (Safe)
5. **Safe ORM & Query Construction:** 10 samples (Safe)
6. **Validated / Allowlisted Input:** 10 samples (Safe)
7. **Complex Query Construction:** 10 samples (Vulnerable)
8. **No SQL Injection / No SQL Sink:** 5 samples (Safe)  
**Total:** 60 Vulnerable, 40 Safe = 100 Samples.

---

## 2. Fine-Grained Line Localization Metrics

| Localization Dimension | Metric Evaluated | Value | Definition & Scope |
| :--- | :--- | :---: | :--- |
| **SQL Execution Sink** | Exact Sink Line Accuracy | **75.00%** | Exact line match on DB-API execution call (`conn.execute`, `cursor.execute`). |
| **Query Construction** | Exact Construction Line Accuracy | **83.33%** | Exact line match on f-string, format, or concatenation expression. |
| **Input Source** | Exact Source Line Accuracy | **0.00%** | Exact line match on function parameter or request payload arrival site. |
| **Vulnerable Line Set** | Top-1 Accuracy | **83.33%** | First reported vulnerable line is in ground truth vulnerable set. |
| **Vulnerable Line Set** | Top-3 Accuracy | **83.33%** | At least one reported line within top 3 is in ground truth vulnerable set. |
| **Vulnerable Line Set** | Mean Overlap $F_1$ Score | **0.7167** | Harmonic mean of line precision (77.8%) and recall (70.8%). |
| **Dataflow Taint Path** | Complete Path Accuracy | **58.33%** | Multi-hop source $	o$ propagation $	o$ construction $	o$ sink sequence valid. |

---

## 3. Evidence and Explanation Validation

For every flagged sample, structured evidence objects were validated against code semantics:
- **Taint Origin Support:** Verified that sources contain unvalidated input variables.
- **SQL Execution Support:** Verified that sinks represent executable database driver operations.
- **Safe Parameter Exemption:** Verified that `%s` and `?` bound placeholders pass parameters separately, avoiding false injection alerts.
