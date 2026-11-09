# Representative Localization Failure Cases

This document details cases where the static analyzer exhibited localization limitations, missing cross-file hops or failing to localize multi-clause query builders.

---

### Case 1: Cross-File Import Taint Hop
- **Sample ID:** `loc_crossfile_001`
- **Category:** Cross-file injection
- **Ground Truth:** Source `api.py`:3, Sink `db_ops.py`:6
- **Analyzer Limitation:** When resolving imports without an explicit package manifest, `ProjectASTGraph` resolves function calls but occasionally fails to propagate backward argument indices across separate module roots, resulting in partial path tracing.
- **Root Cause:** Import aliasing without full virtual environment path mapping.

---

### Case 2: Multi-Clause Complex Query Loop
- **Sample ID:** `loc_complex_001`
- **Category:** Complex query construction
- **Ground Truth:** Construction Line 5 (`query += " AND status = '" + val + "'`), Sink Line 6
- **Analyzer Limitation:** Intraprocedural tracer identifies the sink at Line 6, but inside a `for` loop body, AST assignment tracing attributes the construction line to the initial assignment at Line 2 rather than the augmented assignment inside the loop block.
- **Root Cause:** Flow-insensitive intra-statement assignment tracking in iterative loop blocks.
