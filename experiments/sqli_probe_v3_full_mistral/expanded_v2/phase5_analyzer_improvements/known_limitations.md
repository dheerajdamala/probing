# Static Analyzer Known Limitations & Theoretical Boundaries

## Project
Python SQL Injection Detection (CWE-89) using Frozen Mistral-7B Representations and Interprocedural Static Analysis.
Phase 5: Static Analyzer Improvement.

---

## 1. Executive Summary

Static program analysis operates under Rice's Theorem and the undecidability of arbitrary runtime behavior. While the Phase 5 improvements have substantially enhanced source attribution, mutation tracking, dynamic placeholder recognition, and interprocedural dataflow tracing, specific dynamic Python language constructs and execution boundaries remain fundamentally outside static deterministic verification.

This document details the catalog of known architectural limitations, their theoretical causes, current analyzer fallback behaviors, and recommended future mitigations.

---

## 2. Catalog of Known Limitations

### 2.1 Dynamic Reflection and Metaprogramming
- **Description**: Python allows dynamic symbol resolution via `getattr()`, `setattr()`, `locals()`, `globals()`, `importlib.import_module()`, and `eval()` / `exec()`.
- **Analyzer Behavior**: The analyzer extracts static AST nodes. If a database cursor or execution method is invoked via `getattr(cursor, "execute")(query)` or constructed through `eval(f"cursor.{op}('{query}')")`, the AST visitor cannot resolve the function target statically.
- **Handling & Fallback**: The analyzer flags unresolved dynamic calls as `PARTIAL` analysis status and preserves uncertainty rather than claiming `SAFE`.

### 2.2 Unbounded Loop Fixpoints and Path Sensitivity
- **Description**: While Phase 5 added tracking for loop variables, accumulator mutations (`query += ...`), and `for`/`while` loop constructs, static analysis cannot determine runtime loop bounds without abstract interpretation or symbolic execution.
- **Analyzer Behavior**: The analyzer inspects loop bodies for any unsafe mutations or unparameterized variable interpolations. If an unsafe mutation occurs within any loop iteration, the variable is permanently marked tainted.
- **Handling & Fallback**: The analyzer does not perform unbounded symbolic loop unrolling to prevent exponential path explosion and termination failures. If loop convergence or termination cannot be proven, the analyzer preserves `PARTIAL` status.

### 2.3 Inter-Process and Distributed Message Queues
- **Description**: In distributed applications, untrusted HTTP payloads often enter a web handler, get serialized (e.g., via JSON, Pickle, Celery, RabbitMQ, Kafka, or Redis), and are subsequently consumed and executed by a background worker task in a separate process.
- **Analyzer Behavior**: AST-based interprocedural analysis is bounded to the project source repository. Cross-process IPC, message brokers, and database-backed job queues break the static AST graph.
- **Handling & Fallback**: The analyzer models formal parameters of public functions as potential external entrypoints, assigning `FORMAL_PARAMETER` roles to caller inputs and reporting `PARTIAL` confidence when callers cannot be resolved.

### 2.4 Dynamic ORM Compilers and Custom SQL Dialects
- **Description**: Advanced ORM libraries (e.g., SQLAlchemy `Select`, Peewee, Tortoise ORM, Django RawSQL with dynamic column definitions) construct SQL queries through internal method chaining and AST trees rather than primitive string formatting.
- **Analyzer Behavior**: The analyzer identifies high-level ORM calls (e.g., `filter()`, `select()`, `objects.get()`) as safe. However, hybrid patterns mixing ORM methods with `extra(where=[...])` or raw clause injection may evade pattern matching if queries are built via proprietary AST visitor subclasses.
- **Handling & Fallback**: Documented ORM raw-escape methods (`raw()`, `extra()`, `execute()`) are monitored as sinks.

### 2.5 Multi-Stage Encodings and Custom Sanitizers
- **Description**: Applications that apply bespoke validation (e.g., custom regex replacement, custom hex decoding, or proprietary sanitization functions) without using standard allowlists or parameterized queries.
- **Analyzer Behavior**: The analyzer verifies allowlist dictionaries (`ALLOWED.get(key)`) and integer casting (`int(val)`). Arbitrary user-defined sanitization functions (e.g., `def my_clean(s): return s.replace("'", "")`) are not assumed safe unless explicitly audited.
- **Handling & Fallback**: Preserves conservative taint propagation and marks the finding as requiring manual review (`PARTIAL`).

---

## 3. Summary of Failure Modes and Safety Invariants

| Failure Mode | Theoretical Cause | Static Analyzer Defense / Invariant |
|---|---|---|
| `getattr` reflection | Dynamic runtime dispatch | Mark analysis status as `PARTIAL`; do not emit false `SAFE` |
| Unbounded while-loop | Undecidable halting problem | Single-pass AST body inspection; mark tainted on any unsafe mutation |
| Cross-process queue | Serialization boundary | Treat background worker handler parameters as `FORMAL_PARAMETER` |
| Dynamic placeholder without binding | Unbound SQL parameter | Require `sink.has_parameter_binding=True` before declaring safe |
| Parameter dictionary lookups (`payload.get`) | Conflation with allowlist | Strictly require allowlist dictionary to be constant literal or allowlist name |

---

## 4. Verification and Regression Guardrails

All future modifications must ensure:
1. No heuristic claims `SAFE` when an execution sink consumes untrusted variables without verified parameter binding or allowlist mapping.
2. The 50/50 extension regression suite must continue to pass with zero test failures.
3. All 5 Phase 5 focused test suites (26 tests) must maintain 100% pass rates.

