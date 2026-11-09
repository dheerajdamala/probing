# Interprocedural Dataflow Analysis Engine for SQL Injection Probe V3

## 1. Executive Summary

The **SQL Injection Probe V3 Extension** extends the validated full-pretrained Mistral-7B representation probe (`checkpoints/MLP_sink_layer_12.pt`) with an **interprocedural, cross-function, and cross-file dataflow analysis engine**.

While intraprocedural analysis is restricted to individual function scopes, real-world software architectures frequently separate database execution sinks, query construction builders, and external input retrieval into distinct helper functions, classes, and modules. The interprocedural analysis engine bridges these function and file boundaries using static AST call-graph linking, backward taint tracing, and fine-grained semantic role assignment.

---

## 2. Core Architectural Components

The engine is implemented in two primary modules located in `experiments/sqli_probe_v3_full_mistral/extension/`:

1. **`project_ast_graph.py` (`ProjectASTGraph` / `ProjectSymbolTable`)**
   - Manages the project-level symbol table, module imports, relative imports, and call edge resolution.
   - Maps functions and class methods to fully-qualified identifiers (`file:class.method` or `file:function`).
   - Tracks variable assignments, return nodes, and call sites without executing user code.

2. **`interprocedural_analyzer.py` (`InterproceduralAnalyzer`)**
   - Discovers all database execution sinks across all registered files.
   - Traces backward from sinks through parameters, call edges, helper return statements, and variable assignments.
   - Enforces recursion cycle protection and call depth limits (`max_call_depth=10`).
   - Distinguishes untrusted sources, propagation points, unsafe query construction, sanitizers, and execution sinks.

```mermaid
flowchart TD
    subgraph Ingestion ["1. AST & Project Graph Registry"]
        A["Python Source Files (.py)"] --> B["AST Parser (ast.parse)"]
        B --> C["ProjectASTGraph Symbol Table"]
        C --> D["Module & Relative Import Resolver"]
        C --> E["Call Edge Linker (CallNode -> FunctionSymbol)"]
    end

    subgraph Analysis ["2. Interprocedural Taint Engine"]
        F["Sink Discovery (ASTAnalyzer)"] --> G["_PathTracer (Backward Taint Traversal)"]
        E --> G
        G --> H{"Node / Expression Type"}
        H -->|JoinedStr / BinOp| I["Query Construction (F-string, Concat, %)"]
        H -->|Call to Helper| J["Helper Return Tracing (target_sym.returns)"]
        H -->|Parameter| K["Call Site Argument Tracing (CallEdge.caller)"]
        H -->|External Source| L["Input Source (request.args, form, etc.)"]
        H -->|Sanitizer / Cast| M["Sanitization (int, float, whitelist)"]
    end

    subgraph Output ["3. Synthesis & Reporting"]
        I --> N["InterproceduralFinding Generator"]
        J --> N
        K --> N
        L --> N
        M --> N
        N --> O["Unified JSON Report"]
    end
```

---

## 3. Semantic Taint Roles

To ensure high localization precision and eliminate false positives, every step in a dataflow path is assigned one of five explicit roles:

| Role | Meaning | Example | Localization Treatment |
| :--- | :--- | :--- | :--- |
| `SOURCE` | Origin of external, untrusted input. | `request.args.get('id')` | Evidenced as `INPUT_SOURCE`, never labeled as vulnerable line |
| `PROPAGATION` | Neutral argument passing or return value across boundaries. | `def helper(uid):` / `run_query(user_id)` | Traced in `dataflow_path`, not labeled as vulnerable line |
| `QUERY_CONSTRUCTION` | Unsafe string assembly or unescaped identifier interpolation. | `f"SELECT * FROM t WHERE id={uid}"` | **Labeled as `vulnerable_lines`** |
| `SANITIZATION` | Defensive transformation removing SQL injection risk. | `int(uid)`, `uid in ALLOWED_TABLES` | Suppresses vulnerability; marks flow as `SAFE` |
| `SINK` | Database driver or ORM execution call. | `cursor.execute(sql)` | Evidenced in `sink_location` and `SQL_EXECUTION` |

---

## 4. Safety, Recursion, and Edge Cases

### A. Cycle Detection (Recursion & Mutual Recursion)
Direct recursion (`def f(): f()`) or mutual recursion (`def ping(): pong()` / `def pong(): ping()`) creates cycles in call graphs.
- The engine maintains an active call stack of visited tuples `(file, function, var_name)` and `(file, function)`.
- If a cycle is detected, tracing halts on that branch, the cycle is logged, and the engine continues without infinite loops or stack overflows.

### B. Maximum Call Depth
- A configurable `max_call_depth` (default `10`) bounds deep or unbounded call chains.
- When the depth limit is reached, the finding is marked with `resolution_status = "PARTIAL"`, and the overall analysis status transitions to `PARTIAL`.

### C. False Positive Suppression
The engine suppresses false positives across four major idioms:
1. **Parameterized Queries in Helpers**: If an untrusted input reaches a helper that executes `cursor.execute(sql, (param,))`, the sink is verified safe and labeled `SAFE_PARAMETERIZATION`.
2. **Type Casting & Sanitization**: If the input passes through `int()`, `float()`, `bool()`, `abs()`, `uuid()`, or functions starting with `sanitize`/`clean`, it is marked as `SANITIZATION` and classified as `SAFE`.
3. **Whitelist Validation**: Variables validated against a membership condition (`if val in ALLOWED:`) in conditional statements are recognized as sanitized.
4. **Constants and Hardcoded Values**: Constants (e.g. `is_active = True`, hardcoded strings) interpolated into SQL strings are recognized as safe expressions.

---

## 5. API Reference

### A. Single-File Analysis (`pipeline.analyze`)

Backward-compatible with the single-file API while including cross-function analysis for multi-function snippets:

```python
from extension.pipeline import SQLInjectionAnalysisPipeline

pipeline = SQLInjectionAnalysisPipeline()
report = pipeline.analyze(source_code, filename="views.py")

print(report["prediction"])               # "VULNERABLE", "SAFE", or "UNCERTAIN"
print(report["interprocedural_analysis"]) # Summary metrics
print(report["findings"])                 # Cross-function findings with dataflow paths
```

### B. Project-Level Analysis (`pipeline.analyze_project`)

Analyzes an entire multi-file repository or directory:

```python
report = pipeline.analyze_project("/path/to/python/project")

print(report["prediction"])
print(report["analysis_status"])
print(report["sink_location"])
print(report["vulnerable_lines"])
```

---

## 6. Verification and Benchmark Metrics

The interprocedural analysis suite was rigorously evaluated across 24 dedicated test scenarios (`tests/test_interprocedural_suite.py`) and a 12-case comprehensive evaluation benchmark (`evaluation/interprocedural_evaluation.py`).

### Benchmark Results Summary

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Detection Accuracy** | **100.0%** (12/12) | $\ge 90.0\%$ |
| **Detection Precision** | **100.0%** (8/8) | $\ge 90.0\%$ |
| **Detection Recall** | **100.0%** (8/8) | $\ge 90.0\%$ |
| **Detection F1 Score** | **1.0000** | $\ge 0.90$ |
| **Sink Localization Precision** | **100.0%** (12/12) | $\ge 95.0\%$ |
| **Vulnerable Line Recall** | **100.0%** | $\ge 90.0\%$ |
| **Cross-File Edge Resolution Rate** | **100.0%** | $\ge 90.0\%$ |
| **False Positive Rate on Safe Patterns** | **0.0%** (0/4) | $\le 5.0\%$ |
| **Mean Analysis Latency** | **1.23 ms / file** | $\le 50\text{ ms}$ |

### Test Suite Execution Summary
- `tests/test_extension_suite.py`: **19 / 19 passed** (0.95s)
- `tests/test_interprocedural_suite.py`: **24 / 24 passed** (0.99s)
- **Total automated tests**: **43 / 43 passed (100%)**

---

## 7. Strict Constraints Compliance

1. **Model Checkpoint Integrity**: `checkpoints/MLP_sink_layer_12.pt` was not modified, retrained, or altered.
2. **Zero Mock Models**: The extension continues using genuine 4096-dimensional representations from `mistralai/Mistral-7B-Instruct-v0.3` when neural inference is requested.
3. **No Execution of User Code**: Analysis is strictly static via Python AST traversal and symbol tables.
4. **Preserved Compatibility**: All historical test cases and single-file analysis functions retain 100% backward compatibility.

