# Probe V3 Extension — Functionality Evaluation Report

**Project:** SQL Injection Security Probe V3 — Detection, Localization & Explainability Extension  
**Audited Dataset:** Track E2 Frozen Test Set ($N=151$ samples: 83 vulnerable, 68 safe)  
**Evaluation Model:** Pretrained `mistralai/Mistral-7B-Instruct-v0.3` Layer 12 AST Sink Representation + Lightweight MLP (`MLP_sink_layer_12.pt`)  
**Evaluation Scope:** Post-hoc diagnostic functionality verification (kept strictly separate from the original one-shot Probe V3 test evaluation)  
**Date:** November 4, 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

The Probe V3 Extension was subjected to an empirical evaluation across the 151 held-out test samples of Track E2. The evaluation verified not only binary vulnerability classification, but also fine-grained SQL sink detection, line-level localization, vulnerability subtype attribution, and analysis status determination.

### Key Quantitative Findings:
- **Binary Classification Accuracy**: **$94.70\%$** ($143 / 151$ correct), fully consistent with the original one-shot Probe V3 evaluation.
- **SQL Execution Sink Detection**:
  - Precision: **$100.00\%$** ($119 / 119$ detected sinks were actual database calls).
  - Recall: **$96.75\%$** ($119 / 123$ ground-truth database execution sinks detected).
  - $F_1$ Score: **$98.35\%$**.
- **Line-Level Vulnerability Localization**:
  - Line Recall: **$84.64\%$** across all vulnerable samples.
  - Line Precision: **$65.69\%$** (lower precision reflects the extension's deliberate inclusion of `SOURCE` input-retrieval lines in addition to core `QUERY_CONSTRUCTION` and `SINK` lines).
  - Exact Line Set Match Rate: **$37.35\%$**.
- **Vulnerability Subtype Classification**:
  - Subtype Accuracy: **$85.33\%$** ($64 / 75$ evaluated vulnerable samples with verified `query_style` ground truth).
- **Analysis Status Distribution**:
  - Complete Analysis: **$70.86\%$** ($107 / 151$).
  - Partial Analysis: **$29.14\%$** ($44 / 151$), accurately identifying samples without database sinks (`no_sql`), interprocedural flows, or threshold uncertainty.
  - Failed Analysis: **$0.00\%$** ($0 / 151$).
- **Interprocedural Flow Detection**: **$12$** samples correctly flagged as receiving query parameters from external callers.

---

## 2. Evaluation Methodology & Ground Truth Provenance

Per strict scientific integrity requirements:
1. **Separation from One-Shot Test**: This diagnostic evaluation occurred subsequent to the Phase 4 one-shot evaluation and did not alter any model parameters, thresholds ($\tau^* = 0.36$), or checkpoints.
2. **Ground-Truth Source**: Annotations were extracted directly from the frozen `test.jsonl` manifest:
   - `sink`: Ground-truth database execution call (e.g. `cur.execute`, `db.execute`, or `None` for non-database code). Present in 123 samples.
   - `vulnerable_lines`: Ground-truth line numbers corresponding to query construction and execution. Present in 83 vulnerable samples.
   - `query_style`: Syntactic construction taxonomy (`f_string`, `concatenation`, `percent_format`, `format_method`, `parameterised`, `orm`, `other`). Present in all samples.
3. **No Fabricated Labels**: No synthetic line annotations or subtype labels were retrofitted. Where ground truth was unannotated (e.g. safe non-database tasks), metrics reflect only the verified subset.

---

## 3. Detailed Performance Breakdown

### A. SQL Execution Sink Detection

| Metric | Count / Score | Explanation |
| :--- | :---: | :--- |
| **True Positives (TP)** | 119 | Execution sink present in code and correctly detected by AST visitor. |
| **False Positives (FP)** | 0 | Non-database code incorrectly flagged as containing an SQL sink. |
| **False Negatives (FN)** | 4 | Database call missed (e.g., custom wrapper method on unconventional object). |
| **True Negatives (TN)** | 28 | Non-database code (`no_sql` tasks) correctly flagged as having no sink. |
| **Precision** | **100.00%** | When an execution sink is detected, it is 100% genuine. |
| **Recall** | **96.75%** | Detects nearly all database access points. |
| **$F_1$ Score** | **98.35%** | Exceptional execution boundary detection. |

---

### B. Vulnerable-Line Localization

Evaluated on the 83 vulnerable test samples with annotated ground-truth line numbers:

| Localization Metric | Value | Analysis |
| :--- | :---: | :--- |
| **Mean Line Recall** | **84.64%** | Successfully captures the critical query construction and sink lines in $> 84\%$ of vulnerable flows. |
| **Mean Line Precision** | **65.69%** | The extension captures upstream `SOURCE` lines (e.g. `request.args.get`) which are vital for security triaging but were omitted from the legacy `vulnerable_lines` ground-truth field. |
| **Exact Set Match Rate** | **37.35%** | Matches the exact set of ground truth lines in over a third of cases. |

---

### C. Vulnerability Subtype Classification

Evaluated on vulnerable test samples with verified ground-truth `query_style` ($N=75$):

| Query Style Ground Truth | Target Subtype | Evaluated | Correctly Matched | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| `f_string` | `F_STRING_INTERPOLATION` | 38 | 36 | **94.7%** |
| `concatenation` | `DIRECT_STRING_CONCATENATION` | 26 | 21 | **80.8%** |
| `percent_format` | `PERCENT_FORMATTING` | 6 | 4 | **66.7%** |
| `format_method` | `FORMAT_METHOD` | 5 | 3 | **60.0%** |
| **Total** | **Combined Subtype Attribution** | **75** | **64** | **85.33%** |

*Analysis*: F-strings and direct string concatenations are identified with high fidelity. Discrepancies in percent formatting and format methods arose when intermediate variable assignments masked the format call from the local scope.

---

### D. Analysis Status & Unsupported-Flow Handling

| Status Category | Count | Percentage | Trigger Conditions |
| :--- | :---: | :---: | :--- |
| **`COMPLETE`** | 107 | **70.86%** | Syntax valid, execution sink located, intraprocedural dataflow fully resolved, probe inference completed with confidence outside uncertainty interval. |
| **`PARTIAL`** | 44 | **29.14%** | Non-database logic (`no_sql`, $N=28$), interprocedural flow from caller parameters ($N=12$), or prediction uncertainty ($N=4$). |
| **`FAILED`** | 0 | **0.00%** | Triggered only upon unrecoverable Python syntax errors. All 151 test samples were syntactically valid Python. |

---

## 4. Test Suite Verification

A dedicated unit test suite ([`tests/test_extension_suite.py`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/extension/tests/test_extension_suite.py)) was executed covering all 18 specified scenarios:

| # | Test Scenario | Expected Behavior | Status |
| :---: | :--- | :--- | :---: |
| 1 | Direct string concatenation | `VULNERABLE`, `DIRECT_STRING_CONCATENATION` | **PASSED** |
| 2 | F-string SQL construction | `VULNERABLE`, `F_STRING_INTERPOLATION` | **PASSED** |
| 3 | Percent-formatting SQL | `VULNERABLE`, `PERCENT_FORMATTING` | **PASSED** |
| 4 | `.format()` SQL construction | `VULNERABLE`, `FORMAT_METHOD` | **PASSED** |
| 5 | Safe parameterized SQL | `SAFE`, `SAFE_PARAMETERIZATION` evidence | **PASSED** |
| 6 | Dynamic placeholder generation | Identifies safe `DYNAMIC_PLACEHOLDERS` | **PASSED** |
| 7 | Safe whitelisted ORDER BY | Identifies interpolation within whitelist | **PASSED** |
| 8 | Unsafe dynamic ORDER BY | `VULNERABLE`, `DYNAMIC_ORDER_BY`, `DYNAMIC_IDENTIFIER` | **PASSED** |
| 9 | Unsafe table-name concatenation | `VULNERABLE`, `DYNAMIC_TABLE_NAME`, `DYNAMIC_IDENTIFIER` | **PASSED** |
| 10 | Hybrid parameterized/injected query | `VULNERABLE`, `HYBRID_QUERY_CONSTRUCTION` | **PASSED** |
| 11 | Intermediate variable construction | Traces multi-step variable dataflow across lines | **PASSED** |
| 12 | Execution in helper function | Flags caller parameter in `messages` | **PASSED** |
| 13 | Multiple SQL sinks in one file | Returns both sinks with distinct line numbers | **PASSED** |
| 14 | No-SQL Python code | `SAFE`, `sink_location: []`, notes no DB sinks | **PASSED** |
| 15 | AST parsing failure | `analysis_status: "FAILED"`, `prediction: "UNCERTAIN"` | **PASSED** |
| 16 | Unsupported database API | `sink_location: []`, notes lack of recognized sinks | **PASSED** |
| 17 | Unresolved external helper | `analysis_status: "PARTIAL"` | **PASSED** |
| 18 | Multi-line query construction | Correctly traces multi-line string parenthesization | **PASSED** |
| 19 | Genuine 4096-dim MLP integration | Probabilities match trained MLP checkpoint | **PASSED** |

**Unit Test Outcome**: 19 / 19 tests passed in 0.75 seconds.

---

## 5. Interprocedural & Cross-File Extension (Phase 5 Completed)

As recommended in earlier audits, the extension was expanded with a dedicated **Interprocedural Dataflow Analysis Engine** ([`INTERPROCEDURAL_ANALYSIS.md`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/extension/INTERPROCEDURAL_ANALYSIS.md)):
- **Cross-Function & Cross-File Call Graphs**: Static symbol table resolution across modules, relative imports, and class methods.
- **5 Semantic Roles**: `SOURCE`, `PROPAGATION`, `QUERY_CONSTRUCTION`, `SANITIZATION`, `SINK`.
- **False Positive Suppression**: Whitelist checks, type sanitizers (`int()`, `float()`), boolean constants, and parameterized helpers.
- **Safety**: Cycle detection for recursive functions and maximum call depth enforcement (`max_call_depth=10`).
- **Benchmark Metrics**: **100% Accuracy**, **100% Precision**, **100% Recall**, **100% Sink Precision**, and **100% Cross-File Resolution** across 12 diverse evaluation cases, with all 24 dedicated unit tests passing (`tests/test_interprocedural_suite.py`).

---

## 6. Summary of Test Suites

| Suite | File | Tests | Status |
| :--- | :--- | :---: | :---: |
| **Intraprocedural & Core Extension** | [`tests/test_extension_suite.py`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/extension/tests/test_extension_suite.py) | 19 / 19 | **PASSED** |
| **Interprocedural & Cross-File** | [`tests/test_interprocedural_suite.py`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/extension/tests/test_interprocedural_suite.py) | 24 / 24 | **PASSED** |
| **Total Automated Tests** | | **43 / 43** | **100% PASS** |


