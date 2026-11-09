# Phase Report: Expanded E2 Mistral Embedding Extraction & Compatibility Audit (Completed & Verified)

**Date:** 2026-11-09  
**Project:** SQL Injection Security Probe V3 — Dataset Expansion & Representation Extraction  
**Directory:** `experiments/sqli_probe_v3_full_mistral/expanded_v2/`  
**Status:** **100% Completed, Merged, and Cryptographically Verified**  

---

## 1. Executive Summary

Phase 1 of the Expanded E2 experiment pipeline has completed with complete success:
- **Total Representations Extracted & Merged:** **4,673 samples** (3,304 train, 674 validation, 695 test).
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3` ($d_{\text{model}} = 4,096$, 32 Decoder Layers).
- **Extraction Layer:** Decoder Layer 12 output (`hidden_states[12]`).
- **Pooling Methodology:** AST SQL Sink Region Pooling via `extract_ast_sink_regions` & `pool_ast_sink_representation`.
- **Integrity Audit:** Zero NaNs, zero Infs, zero zero-norm vectors, zero missing IDs across all splits.
- **Historical Integrity:** The 1,000 historical E2 representations match historical checkpoints bit-for-bit.

---

## 2. Partition Accounting & Physical Vector Storage

| Partition | Total Records | Reused Historical | Newly Extracted | Output Matrix Shape | Dtype | Min Norm | Max Norm | Mean Norm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **3,304** | 700 (21.2%) | 2,604 (78.8%) | `(3304, 4096)` | `float32` | 1.2031 | 14.7764 | 1.6143 |
| **Validation** | **674** | 149 (22.1%) | 525 (77.9%) | `(674, 4096)` | `float32` | 1.2170 | 12.8504 | 1.5962 |
| **Test** | **695** | 151 (21.7%) | 544 (78.3%) | `(695, 4096)` | `float32` | 1.2649 | 9.2696 | 1.5966 |
| **TOTAL** | **4,673** | **1,000 (21.4%)** | **3,673 (78.6%)** | **4,673 Vectors** | `float32` | — | — | — |

---

## 3. Cryptographic Verification & Audit Hashes

All final merged matrices, ID mappings, and label arrays were audited by `validate_embeddings.py`:

```
Extraction Audit Manifest Summary (extraction_validation_audit.json):
-------------------------------------------------------------------------------
Train Split (3,304 Records):
  - Layer 12 NPY SHA-256:     bc3c8331920fbcbef1975865aba06da4fd809c6947a70fc02a271a661e227e9e
  - Sample IDs JSON SHA-256:  c6aa972326f894c8f08d49307b30250e72f82d9be32550867dc6471b074c2f4d
  - Embeddings NPZ SHA-256:   4cd030f5f997cf1fd047647030812ca8b94d2e544710f0352c65697681ed3e39
  - Historical Bit-Exact:     TRUE (700 / 700 verified)

Validation Split (674 Records):
  - Layer 12 NPY SHA-256:     91b3737a3478f4d2cea5fe4e41b33098ef6780eb999e67b91719193e1cf78238
  - Sample IDs JSON SHA-256:  8b18468caa8812f39d073d4b8fb5539e632cfaef21adfb89fb86f3643cffc450
  - Embeddings NPZ SHA-256:   15f40bdd01885bd49f019cf0c9bf832a50db73e5d6f79a215141bf8c23b09999
  - Historical Bit-Exact:     TRUE (149 / 149 verified)

Test Split (695 Records):
  - Layer 12 NPY SHA-256:     fcf45ac310be90e8baa04f46ced5b83fcca2e40691f94b953d44eac131c7724a
  - Sample IDs JSON SHA-256:  c3e0bffb1c5527b63dde871911d7dab6ed905fbd45ab93de35737afc3cf64c69
  - Embeddings NPZ SHA-256:   fb5f1feecd96fcb25950b1990eeff7d1bf0e58d884d1e9132961301ee26104be
  - Historical Bit-Exact:     TRUE (151 / 151 verified)

All Validation Checks Passed: TRUE
```

---

## 4. Complete Output Artifact Structure

The finalized representation matrices are stored in:
`experiments/sqli_probe_v3_full_mistral/expanded_v2/embeddings/`

```
expanded_v2/embeddings/
├── train/
│   ├── sink/
│   │   └── layer_12.npy               # (3304, 4096) float32 matrix
│   ├── sample_ids.json                # Ordered list of 3,304 sample IDs
│   ├── sample_id_to_index.json        # Explicit {sample_id: index} dictionary
│   ├── sample_id_embeddings.npz       # Direct key-value store {sample_id: vector}
│   └── labels.npy                     # (3304,) ground-truth binary labels
├── validation/
│   ├── sink/
│   │   └── layer_12.npy               # (674, 4096) float32 matrix
│   ├── sample_ids.json                # Ordered list of 674 sample IDs
│   ├── sample_id_to_index.json        # Explicit {sample_id: index} dictionary
│   ├── sample_id_embeddings.npz       # Direct key-value store {sample_id: vector}
│   └── labels.npy                     # (674,) ground-truth binary labels
├── test/
│   ├── sink/
│   │   └── layer_12.npy               # (695, 4096) float32 matrix
│   ├── sample_ids.json                # Ordered list of 695 sample IDs
│   ├── sample_id_to_index.json        # Explicit {sample_id: index} dictionary
│   ├── sample_id_embeddings.npz       # Direct key-value store {sample_id: vector}
│   └── labels.npy                     # (695,) ground-truth binary labels
├── historical/                        # Reused historical 1,000 vectors archive
├── new_extracted/                     # Unpacked 3,673 newly extracted vectors
├── expanded_v2_extraction_summary.json# Master extraction manifest
└── extraction_validation_audit.json   # Machine-readable audit certificate
```

The dataset representations are fully prepared for **Phase 2: Expanded MLP Probe Training**.
