# Phase 4B vs Phase 5 Metric Discrepancy & Root Cause Analysis

## Project
**Python SQL Injection Detection (CWE-89)** using Frozen Mistral-7B Representations and Interprocedural Static Analysis.
**Audit Phase**: Phase 5.1 (Reproducibility & Metric Consistency Audit).

---

## 1. Executive Summary

During the Phase 5 development diagnostic evaluation, a noticeable discrepancy emerged between the metrics originally reported for the frozen baseline analyzer (V1) in Phase 4B and the frozen V1 baseline metrics reported in Phase 5:

| Metric | Phase 4B Reported (Frozen V1) | Phase 5 Reported (Frozen V1) | Observed Discrepancy |
|---|---|---|---|
| **Sink Exact Accuracy** | **75.00% (45 / 60)** | **100.00% (60 / 60)** | **+25.00% (15 samples)** |
| **Construction Line Accuracy** | **83.33% (50 / 60)** | **100.00% (60 / 60)** | **+16.67% (10 samples)** |
| **Source-to-Sink Path Accuracy** | **58.33% (35 / 60)** | **100.00% (60 / 60)** | **+41.67% (25 samples)** |
| **Vulnerable-Line Top-1 Accuracy** | **83.33% (50 / 60)** | **83.33% (50 / 60)** | **0.00% (Identical)** |
| **Vulnerable-Line Top-3 Accuracy** | **83.33% (50 / 60)** | **83.33% (50 / 60)** | **0.00% (Identical)** |
| **Source Line Accuracy** | **0.00% (0 / 60)** | **0.00% (0 / 60)** | **0.00% (Identical)** |

This independent audit investigated the exact root causes behind these discrepancies across nine specific evaluation vectors. 

### Core Finding
The discrepancies are **not** caused by benchmark changes, sample modifications, or ground-truth tampering. The underlying 100 benchmark sample files and their ground-truth annotations are **100% bit-for-bit identical** (verified by SHA-256).

Instead, the discrepancies are entirely attributable to **two distinct evaluation methodology differences** between the evaluation scripts:
1. **Evaluation Script Defect in Phase 4B (Line 448 Key Typo)**: In `run_phase_4b_localization.py`, multi-file project analysis attempted to read `analysis_res.get("sinks", [])`. However, `SQLInjectionAnalysisPipeline.analyze_project()` returns the key `"sink_location"`. Because of this key mismatch, all 15 multi-file cross-file samples received an empty sink list (`[]`), causing sink accuracy to artificially drop from 100% (60/60) to 75% (45/60).
2. **Construction Line Extraction Criteria (Strict Findings vs. Evidence Fallback)**: In Phase 4B, construction lines were strictly extracted from `finding.vulnerable_lines` with `role == "QUERY_CONSTRUCTION"`. For the 10 "Complex query construction" samples, V1's interprocedural analyzer failed to resolve the loop-carried dataflow and returned `SAFE`, which caused `pipeline.py` to suppress `vulnerable_lines = []`. In Phase 5, the diagnostic script extracted construction lines from the broader `evidence` array, which contained `DIRECT_STRING_CONCATENATION` for all 10 complex samples, shifting construction accuracy from 83.33% (50/60) to 100% (60/60).

---

## 2. Systematic Investigation Across Evaluation Vectors

### Vector 1: Were the benchmark files identical?
**Verdict**: **YES (100% Identical).**
- All 100 benchmark sample directories and Python scripts were hashed using SHA-256.
- The manifest hash `bfe5c25e37089184b79b503052e1ea25d28751a794ca5a7ec2e52c9b7a02481f` is identical across Phase 4B and Phase 5.
- No source code files were added, deleted, or modified.

### Vector 2: Were ground-truth annotations identical?
**Verdict**: **YES (100% Identical).**
- File: `expanded_v2/localization_evaluation/ground_truth_annotations.json`
- SHA-256: `7c9f66f54f1a12b9ca1334aa51ef6ce35df6c3ad55efc4f106c4696167824bcb`
- Line-by-line verification confirmed that ground-truth sets (`sink_lines`, `construction_lines`, `source_lines`, `vulnerable_lines`) have not changed.

### Vector 3: Did sample IDs and file paths match?
**Verdict**: **YES (100% Identical).**
- Exactly 100 sample IDs (`loc_direct_001` through `loc_safe_noval_005`) match across both manifests.

### Vector 4: Was the same evaluation code used?
**Verdict**: **NO.**
- Phase 4B used `run_phase_4b_localization.py`.
- Phase 5 used `phase5_analyzer_improvements/run_phase5_evaluation.py`.
- Key algorithmic differences existed in how dictionary fields were read from pipeline reports.

### Vector 5: Did denominators change?
**Verdict**: **NO.**
- Both evaluations used exactly $N = 60$ vulnerable samples as the denominator for vulnerability localization metrics, and $N = 40$ safe samples for false-positive rates.

### Vector 6: Did metric definitions change?
**Verdict**: **YES.**
- **Construction Line Extraction**:
  - Phase 4B Definition: $\text{Line} \in \{vl.\text{line} \mid vl \in \text{findings.vulnerable\_lines} \land vl.\text{role} == \text{"QUERY\_CONSTRUCTION"}\}$.
  - Phase 5 Definition: $\text{Line} \in \{ev.\text{line} \mid ev \in \text{evidence} \land ev.\text{category} \in \text{UNSAFE\_CONSTRUCTIONS}\} \cup \{vl.\text{line} \mid vl \in \text{vulnerable\_lines}\}$.
- **Path Accuracy**:
  - Phase 4B Definition: $\text{len}(\text{path\_steps}) \ge 2 \land \text{exact\_sink\_match} \land (\text{exact\_constr\_match} \lor \text{exact\_source\_match})$.
  - Phase 5 Definition: $\text{len}(\text{path\_steps}) \ge 2 \land \text{sink\_match} \land \text{constr\_match}$.

### Vector 7: Were missing predictions excluded?
**Verdict**: **NO.**
- In both evaluations, all 100 samples were evaluated. Zero samples were excluded or dropped.

### Vector 8: Were the same path-matching rules used?
**Verdict**: **NO.**
- As detailed in Vector 6, path correctness in Phase 4B was tightly coupled to finding-level `exact_constr_match`. Because Phase 4B had key and classification failures on 25 samples, its path accuracy was recorded as $35 / 60 = 58.33\%$.

### Vector 9: Did the previous Phase 4B report contain an aggregation error?
**Verdict**: **YES (An Evaluation Script Implementation Bug).**
- Specifically, line 448 of `run_phase_4b_localization.py` queried `analysis_res.get("sinks", [])`. Because `analyze_project` outputs `"sink_location"`, this single-line typo invalidated sink extraction for all 15 multi-file cross-file samples.

---

## 3. Dissecting the 25 Discrepancy Samples

The discrepancy between Phase 4B and Phase 5 is entirely accounted for by two distinct cohorts totaling 25 samples:

### Cohort A: 15 Cross-File Injection Samples (`loc_crossfile_001` – `015`)
- **Category**: Cross-file injection (multi-file projects).
- **Phase 4B Outcome**: Sink match = `False` (0/15), Path match = `False` (0/15).
- **Phase 5 Outcome**: Sink match = `True` (15/15), Path match = `True` (15/15).
- **Cause**:
  In `run_phase_4b_localization.py`:
  ```python
  # Line 448 (Multi-file branch):
  reported_sinks = analysis_res.get("sinks", [])  # BUG: returns []!
  # Compare with Line 460 (Single-file branch):
  reported_sinks = analysis_res.get("sink_location", [])  # CORRECT!
  ```
  The pipeline report dictionary returned by `analyze_project()` uses the key `"sink_location"`. The analyzer *did* successfully find the sink at line 6 of `db_ops.py`, but the evaluation script looked under `"sinks"`, extracted an empty list, and recorded a failure.
- **Correction**: When evaluated using `"sink_location"`, V1 achieves **15 / 15 sink hits** on cross-file samples.

### Cohort B: 10 Complex Query Construction Samples (`loc_complex_001` – `010`)
- **Category**: Complex query construction (loops and augmented assignments).
- **Phase 4B Outcome**: Construction match = `False` (0/10), Path match = `False` (0/10).
- **Phase 5 Outcome**: Construction match = `True` (10/10), Path match = `True` (10/10).
- **Cause**:
  In frozen V1, the intraprocedural dataflow tracer recognized the concatenation at line 5 (`query += ...`) and emitted evidence `DIRECT_STRING_CONCATENATION`. However, V1's interprocedural analysis lacked loop mutation tracking and concluded `SAFE`. Under `pipeline.py`, when a prediction is `SAFE`, `final_vulnerable_lines` is cleared to `[]`.
  - Phase 4B inspected only `vulnerable_lines`, which was empty $\rightarrow$ recorded 0/10.
  - Phase 5 inspected `evidence`, which retained line 5 $\rightarrow$ recorded 10/10.
- **Correction**: Under strict finding/vulnerable_lines extraction, V1 genuinely failed to report construction lines for these 10 samples (50/60 = 83.33%). V2's improved loop and mutation analysis genuinely resolves this, identifying line 5 as `SQL_CONSTRUCTION` in `vulnerable_lines` and achieving **60 / 60 (100.0%)**.

---

## 4. Reconstructed Metric Matrix Across Standards

To provide total clarity, we recomputed all metrics under:
1. **P4B Original Buggy Script Logic**: Reproduces Phase 4B results exactly.
2. **Corrected Strict Logic**: Fixes the line 448 typo while enforcing strict `vulnerable_lines` extraction.
3. **Phase 5 Evidence Fallback Logic**: Uses the broader evidence-level line extraction.
4. **Improved V2 Strict Logic**: Evaluates the improved analyzer under strict `vulnerable_lines` criteria.

| Metric | P4B Original Buggy (V1) | Corrected Strict (V1) | P5 Evidence Fallback (V1) | Improved V2 Strict |
|---|---|---|---|---|
| **Sink Line Accuracy** | 45 / 60 (75.00%) | **60 / 60 (100.00%)** | 60 / 60 (100.00%) | **60 / 60 (100.00%)** |
| **Construction Line Accuracy** | 50 / 60 (83.33%) | 50 / 60 (83.33%) | 60 / 60 (100.00%) | **60 / 60 (100.00%)** |
| **Source-to-Sink Path Accuracy** | 35 / 60 (58.33%) | 50 / 60 (83.33%) | 60 / 60 (100.00%) | **60 / 60 (100.00%)** |
| **Top-1 Vulnerable Line Accuracy** | 50 / 60 (83.33%) | 50 / 60 (83.33%) | 50 / 60 (83.33%) | **60 / 60 (100.00%)** |
| **Top-3 Vulnerable Line Accuracy** | 50 / 60 (83.33%) | 50 / 60 (83.33%) | 50 / 60 (83.33%) | **60 / 60 (100.00%)** |
| **Safe Sample False Positives** | 5 / 40 (12.50%) | 5 / 40 (12.50%) | 5 / 40 (12.50%) | **0 / 40 (0.00%)** |

---

## 5. Audit Conclusion

1. **Phase 4B's 75% Sink Accuracy was an Artifact**: The frozen V1 analyzer always identified 100% of sinks; the 75% was caused by the `"sinks"` vs `"sink_location"` dictionary lookup typo in the multi-file branch of the evaluation script.
2. **V2 Provides Genuine, Verifiable Gains**: Under the strict, rigorous Phase 4B criteria:
   - Construction line accuracy improved from **83.33% (50/60) to 100.00% (60/60)**.
   - Top-1 vulnerable line localization improved from **83.33% (50/60) to 100.00% (60/60)** (+10 samples).
   - Safe-sample false positives dropped from **5 / 40 (12.50%) to 0 / 40 (0.00%)**.
   - Balanced classification accuracy on the benchmark improved from **85.42% to 100.00%**.

