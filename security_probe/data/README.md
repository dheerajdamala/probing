# Security Probe Data Directory

This directory documents data requirements and references external datasets.

## Dataset Policy
- To avoid storage duplication, datasets are referenced from the existing repository root at `data/processed/`:
  - `data/processed/train.json`
  - `data/processed/val.json`
  - `data/processed/test.json`
- Each record in `probe_records` contains:
  - `code`: Full Python code snippet
  - `semantic_label`: 1 (vulnerable / insecure) or 0 (secure / benign)
  - `cwe`: CWE classification (e.g. CWE-119, CWE-89, CWE-78, etc.)
  - `parent_id`: Parent pair identifier used for leakage prevention

## Critical Note on Ground-Truth Spans
- The current datasets provide **whole-program semantic labels** only.
- **Span-level, line-level, or token-level ground truth is NOT present** in the dataset.
- Token and line localization metrics in this prototype are based on **exploratory representation projection** (projecting per-token hidden states onto the trained probe vector).
- We do not fabricate or invent ground-truth span annotations.
