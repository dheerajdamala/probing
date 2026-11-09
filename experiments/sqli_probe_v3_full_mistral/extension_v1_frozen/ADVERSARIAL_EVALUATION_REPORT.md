# Adversarial Evaluation Report: SQL Injection Probe V3 Interprocedural Engine

**Project:** SQL Injection Security Probe V3 — Interprocedural Dataflow Analysis Engine  
**Evaluation Scope:** Adversarial Stress Testing across 100 Non-Trivial Scenarios  
**Date:** November 4, 2026  
**Status:** Completed, Audited & Empirically Documented  

---

## 1. Executive Summary

Following the initial implementation of the Interprocedural, Cross-Function and Cross-File Dataflow Analysis Engine (which achieved 100% on a basic 12-case pilot), we conducted a rigorous, adversarial stress test. The goal was **not** to artificially preserve 100% metrics, but to deliberately challenge the engine against complex real-world Python patterns and identify its genuine structural boundaries.

### Benchmark Highlights:
- **Test Corpus**: **100 distinct, non-trivial test programs** spanning 10 architectural categories.
- **Initial Adversarial Accuracy**: **80.00%** (20 genuine defects / edge-case failures discovered).
- **Hardened Adversarial Accuracy**: **87.00%** (87 / 100 correct) after implementing minimal, justified fixes backed by 7 dedicated regression tests.
- **Precision**: **91.04%** ($61 / 67$).
- **Recall**: **89.71%** ($61 / 68$).
- **$F_1$ Score**: **0.9037**.
- **Sink Detection Recall**: **85.57%** ($83 / 97$ sinks).
- **Execution Latency**: **0.88 ms / case** ($87.68\text{ ms}$ total across all 100 cases).
- **Total Automated Unit Tests Passing**: **50 / 50 (100%)** across 3 test suites.

---

## 2. Baseline Preservation & Integrity Audit

Prior to creating the adversarial benchmark, cryptographic SHA-256 hashes were recorded to verify that existing model weights, scalers, embeddings, and historical reports remained untouched:

| Artifact | Path | SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| **Probe V3 Checkpoint** | `checkpoints/MLP_sink_layer_12.pt` | `a096e55e2dc26b6d62476818ff30c1717cfb7d8139c3fc9b7c2b05a55daf479d` | **UNTOUCHED** |
| **Probe Scaler** | `checkpoints/MLP_sink_layer_12_scaler.joblib` | `222f28a962647290582a79777fdc953ba6b91b38d7284bddcec9b5c37c598d6a` | **UNTOUCHED** |
| **Track E2 Test Embeddings** | `embeddings/full_e2/test/sink/layer_12.npy` | `5def3a12d0a86777e57930ed4a8dbec69a0f2d0e1805801f8e02490e51db25dc` | **UNTOUCHED** |
| **Track E2 Test Labels** | `embeddings/full_e2/test/labels.npy` | `f2ad079cf7bfbbac312b1e89d7eb69889a18220ca48a9e10a7e974ca955538dd` | **UNTOUCHED** |
| **Phase 4 Training Report** | `02_PROBE_V3_E2_TRAINING_AND_EVALUATION_REPORT.md` | `370826960965a226c8c93f8e4a6024c912c4e53e8cb8885bad70048cdf78eed9` | **UNTOUCHED** |
| **Phase 4 Audit Report** | `03_PROBE_V3_POST_TRAINING_AUDIT.md` | `ac72e09f8bd5bbc86cbf01009d2c654e097a5388ad23c96add04cd5549656d98` | **UNTOUCHED** |

---

## 3. Adversarial Benchmark Architecture (100 Cases)

The benchmark is located at `evaluation/adversarial_cases/` with ground-truth annotations decoupled into `evaluation/adversarial_manifest.json`.

```
evaluation/adversarial_cases/
├── vulnerable/     (Cases 01-45, 66-72, 75, 77-82, 85-87, 91-94, 100)
├── safe/           (Cases 46-65, 73-74, 76, 83-84, 95)
├── partial/        (Cases 88-90)
└── edge_cases/     (Cases 96-99)
```

### Performance by Category (10 Categories)

| # | Category | Total Cases | Correct | TP | FP | TN | FN | Accuracy | Mean Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Direct and Indirect SQL Injection** | 15 | 13 | 13 | 0 | 0 | 2 | **86.67%** | 0.93 ms |
| 2 | **Cross-Function Propagation** | 15 | 14 | 14 | 0 | 0 | 1 | **93.33%** | 0.65 ms |
| 3 | **Cross-File Propagation** | 15 | 14 | 14 | 0 | 0 | 1 | **93.33%** | 2.12 ms |
| 4 | **Safe Parameterized Queries** | 10 | 9 | 0 | 1 | 9 | 0 | **90.00%** | 0.68 ms |
| 5 | **Sanitization and Validation** | 10 | 8 | 0 | 2 | 8 | 0 | **80.00%** | 0.72 ms |
| 6 | **Dynamic SQL Identifiers** | 10 | 10 | 8 | 0 | 2 | 0 | **100.00%** | 0.75 ms |
| 7 | **Control-Flow and Reassignment** | 10 | 6 | 4 | 2 | 2 | 2 | **60.00%** | 0.61 ms |
| 8 | **Recursion and Unresolved Calls** | 5 | 5 | 2 | 0 | 3 | 0 | **100.00%** | 0.81 ms |
| 9 | **ORM & Database API Variations** | 5 | 3 | 3 | 1 | 0 | 1 | **60.00%** | 0.74 ms |
| 10 | **Syntax & Unsupported Constructs** | 5 | 5 | 3 | 0 | 2 | 0 | **100.00%** | 0.67 ms |
| **Total** | | **100** | **87** | **61** | **6** | **26** | **7** | **87.00%** | **0.88 ms** |

---

## 4. Discovered Defects & Justified Fixes

When the benchmark was first executed, **20 defects** were exposed. Following the strict requirement:
> *"If a genuine defect is discovered, document it first, create a failing regression test, then implement the smallest justified fix."*

We created [`tests/test_adversarial_regressions.py`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/extension/tests/test_adversarial_regressions.py), verified that all 7 regression tests failed, and then implemented minimal fixes:

### Fixed Defects:
1. **Defect 1: `.format_map()` omission**: `format_map()` was not handled alongside `.format()`.
   - *Fix:* Added `format_map` to `_check_expression_for_sql_construction` and `_trace_expression`.
2. **Defect 2 & 3: Peewee `execute_sql` & Django `extra(where=[...])` sinks**:
   - *Fix:* Added `execute_sql`, `extra`, `execute_query` to `KNOWN_SQL_METHODS`, added `database` and `objects` to `KNOWN_DB_RECEIVERS`, and extracted `where` keyword arguments.
3. **Defect 4: `str(uuid.UUID(...))` wrapper masking sanitizer**:
   - *Fix:* Unwrapped outer `str()` calls in `_trace_expression` to inspect inner sanitizer calls (`UUID`, etc.).
4. **Defect 5: Subclass method inheritance resolution**:
   - *Fix:* Added `class_bases` tracking to `ProjectASTGraph` so calls to inherited methods (`self.run_query()` defined on `BaseDB`) resolve correctly to parent class symbols.
5. **Defect 6: Empty file crash / uncertain status**:
   - *Fix:* Added empty code check returning `SAFE` with status `COMPLETE` directly.
6. **Defect 7: Unresolved calls / depth limit treating unknown flows as safe**:
   - *Fix:* In `pipeline.py`, when findings have `resolution_status in ("PARTIAL", "UNRESOLVED")`, report `prediction = "UNCERTAIN"` rather than falling back to `SAFE`.

---

## 5. Remaining Genuine Weaknesses & Architectural Boundaries (13 Cases)

The remaining 13 defects reflect genuine, well-understood trade-offs between lightweight AST-level analysis and heavyweight SSA/CFG intermediate representations:

### A. Flow-Insensitive Variable Reassignment (4 cases)
- **`ADV_076` & `ADV_084` (Overwrites)**: `q = f"...{untrusted}"; q = "SELECT 1"`. In AST analysis, `assignments["q"]` collects all assignment nodes across the function without order execution tracking. The analyzer flags the earlier tainted assignment.
- **`ADV_077` (Conditional Overwrite)**: Initial safe query overwritten inside an `if` block.
- **`ADV_082` (Bypassed Sanitizer Branch)**: One branch applies `int()`, but the alternate branch does not. The flow-insensitive engine sees the sanitizing call in the function and treats it as sanitized.
- *Root Cause:* AST lacks a Static Single Assignment (SSA) form and control-flow dominator tree.

### B. Dynamic Python Metaprogramming & Aliasing (3 cases)
- **`ADV_008` (Function Pointer Aliasing)**: `run = cursor.execute; run(sql)`. The receiver is a local variable, not an attribute on a cursor object.
- **`ADV_030` (`*args` Unpacking)**: Variable arguments passed via tuple indexing `args[0]`.
- **`ADV_045` (Indirect Multi-Hop Re-Export)**: Re-export across 3 module hops without direct local symbol mapping.
- *Root Cause:* Requires interprocedural alias analysis and points-to graph.

### C. String Concatenation Idioms (1 case)
- **`ADV_009` (`"".join(...)`)**: Building queries via list join rather than binary addition or f-strings.
- *Root Cause:* Join method arguments are list literals, requiring list-element string synthesis.

### D. Advanced Sanitization Patterns (2 cases)
- **`ADV_063` (Dictionary Mapping)**: `col = MAPPING.get(user_input, 'id')`.
- **`ADV_064` (Regex Precondition Guard)**: `if re.match(r'^\d+$', raw): ...`.
- *Root Cause:* Requires semantic expression modeling for dictionary `.get()` fallback and regex pattern semantics.

### E. ORM & Dynamic Parameter In-Clauses (3 cases)
- **`ADV_052` (Dynamic IN-Clause Placeholders)**: `placeholders = ', '.join(['%s'] * len(ids))`.
- **`ADV_091` & `ADV_095` (SQLAlchemy `text(...)` Wrapper)**: `text()` encapsulates the SQL string before passing to `session.execute()`.
- *Root Cause:* The wrapper function separates the AST string literal from the execution sink receiver.

---

## 6. Comprehensive Test Suite Summary

All unit test suites execute in **under 1.05 seconds** with **100% pass rate**:

```powershell
python -m unittest discover -s experiments/sqli_probe_v3_full_mistral/extension/tests -p "test_*.py"
----------------------------------------------------------------------
Ran 50 tests in 1.005s

OK
```

1. **`test_extension_suite.py`**: **19 / 19 PASSED** (Core intraprocedural & schema compliance)
2. **`test_interprocedural_suite.py`**: **24 / 24 PASSED** (Multi-step, cross-file, class methods, cycles)
3. **`test_adversarial_regressions.py`**: **7 / 7 PASSED** (Regressions for discovered adversarial defects)
4. **Total Verified Unit Tests**: **50 / 50 PASSED**

---

## 7. Conclusions & Recommendations

1. **Adversarial Validation Confirms Practical Strength**: Under deliberately aggressive stress testing, the engine maintained **87.00% accuracy** and **0.9037 F1**, with **100% accuracy** on dynamic SQL identifiers, recursion cycle detection, and syntax edge cases.
2. **Preservation of Pretrained Mistral V3**: All neural probe checkpoints and feature representations remain identical to their validated pre-test state.
3. **Future Extension Roadmap**:
   - Implement an SSA-based intraprocedural CFG to resolve temporal variable overwrites and conditional sanitizer branches.
   - Introduce alias tracking for function pointers (`run = cursor.execute`).
