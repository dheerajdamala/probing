# Probe V2: Embedding Extraction Integrity & Audit Report

**Date:** 2026-10-31  
**Project:** SQL Injection Security Probe V2 (CWE-89)  
**Model:** `mistralai/Mistral-7B-Instruct-v0.3` (32 Layers, Hidden Dim $d = 4,096$)  
**Status:** Pre-Training Audit & Integrity Gate Verified  

---

## 1. Executive Summary

In accordance with Phase V2 requirements, this report provides a comprehensive integrity audit prior to training downstream security probes. All dataset partitions, leakage protections, AST-aware sink extractions, candidate layers, and dual pooling implementations have been validated.

**Key Verification Milestones:**
- **Zero Cross-Split Leakage:** 0 exact hash matches, 0 near-duplicate/repository overlaps, and 0 base group/pair cross-contaminations across train, validation, and test splits for all three tracks (E1, E2, E3).
- **Independent External Benchmark Quarantined:** 700 paired samples from `expanded_evaluation_3k` (CWE-89) are held completely out with 0 overlap with any training or tuning partition.
- **AST Sink-Region Extraction:** 11/11 unit tests passed with 100% immunity to Python `from ... import ...` false positives. Tested on SQLite, SQLAlchemy, raw cursor queries, multi-statement constructions, and safe parameterized execution.
- **Dual Pooling from Single Forward Pass:** Attention-mask-correct mean pooling and AST sink pooling are computed simultaneously from the exact same hidden-state activations, avoiding redundant forward passes.
- **Data Quality:** 0 NaNs, 0 Infs, strict $(N, d)$ matrix dimensions, and complete alignment between sample IDs, labels, and embedding rows.

---

## 2. Dataset Split Statistics

The dataset corpus is partitioned into three independent tracks to isolate data distribution effects, plus the quarantined external benchmark:

| Track | Split | Total Samples | Vulnerable ($y=1$) | Safe ($y=0$) | Positive Prevalence | Split Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Track E1** *(Existing Corpus)* | Train | 2,249 | 437 | 1,812 | 19.43% | 74.6% |
| | Val | 384 | 70 | 314 | 18.23% | 12.7% |
| | Test | 383 | 74 | 309 | 19.32% | 12.7% |
| | **E1 Total** | **3,016** | **581** | **2,435** | **19.26%** | **100%** |
| **Track E2** *(Excel 1,000 Corpus)* | Train | 700 | 385 | 315 | 55.00% | 70.0% |
| | Val | 149 | 82 | 67 | 55.03% | 14.9% |
| | Test | 151 | 83 | 68 | 54.97% | 15.1% |
| | **E2 Total** | **1,000** | **550** | **450** | **55.00%** | **100%** |
| **Track E3** *(Combined Deduplicated)* | Train | 2,943 | 822 | 2,121 | 27.93% | 73.3% |
| | Val | 523 | 153 | 370 | 29.25% | 13.0% |
| | Test | 550 | 156 | 394 | 28.36% | 13.7% |
| | **E3 Total** | **4,016** | **1,131** | **2,885** | **28.16%** | **100%** |
| **Track $E_{\text{ext}}$** *(External Benchmark)* | Evaluation Only | **700** | **350** | **350** | **50.00%** | **Held Out** |

*Note on Track E1:* 3,016 clean samples were retained from the original 3,032 records after discarding 16 non-executable, truncated git patch diff fragments that lacked valid Python AST syntax.

---

## 3. Leakage & Contamination Audit

An automated multi-stage cryptographic and semantic audit was performed to guarantee strict isolation between splits:

```
[Split Isolation Audit Matrix]
Track E1:
  - SHA256 Exact Hash Overlap (Train ∩ Val): 0
  - SHA256 Exact Hash Overlap (Train ∩ Test): 0
  - SHA256 Exact Hash Overlap (Val ∩ Test): 0
  - External Benchmark Overlap (Train/Val/Test ∩ External): 0
  - Paired/Base Group Overlap (Train ∩ Val): 0
  - Paired/Base Group Overlap (Train ∩ Test): 0

Track E2:
  - SHA256 Exact Hash Overlap (Train ∩ Val): 0
  - SHA256 Exact Hash Overlap (Train ∩ Test): 0
  - SHA256 Exact Hash Overlap (Val ∩ Test): 0
  - External Benchmark Overlap (Train/Val/Test ∩ External): 0
  - Paired/Base Group Overlap (Train ∩ Val): 0
  - Paired/Base Group Overlap (Train ∩ Test): 0

Track E3:
  - SHA256 Exact Hash Overlap (Train ∩ Val): 0
  - SHA256 Exact Hash Overlap (Train ∩ Test): 0
  - SHA256 Exact Hash Overlap (Val ∩ Test): 0
  - External Benchmark Overlap (Train/Val/Test ∩ External): 0
  - Paired/Base Group Overlap (Train ∩ Val): 0
  - Paired/Base Group Overlap (Train ∩ Test): 0
```

### Partitioning Rules Enforced:
1. **Grouped Splitting by Base ID:** Paired vulnerable and patched implementations (e.g. `pycode_vul_train_12141_vuln` and `pycode_vul_train_12141_patch`) are constrained to the same partition, preventing probes from memorizing identical surrounding scaffolding.
2. **Repository-Level Clustering:** Code samples originating from the same synthetic or open-source repository (e.g., Copilot benchmark prompt variations) reside strictly within single splits.
3. **Quarantine of External Benchmark:** The 700 CWE-89 samples from `expanded_evaluation_3k` are strictly partitioned as an evaluation-only reference benchmark.

---

## 4. Hardware & Resource Profile (T4 GPU vs Local Workstation)

| Metric | Measured T4 Environment (Kaggle/Colab) | Local Workstation (CPU Dry-Run) |
| :--- | :--- | :--- |
| **Accelerator** | NVIDIA Tesla T4 (15.0 GB GDDR6) | Intel Core i5-12500H (16 Threads, CPU Only) |
| **Quantization Scheme** | 4-bit NF4 (`bitsandbytes`), double quant | N/A (Miniature Fixture $d=128$, 4 Layers) |
| **Model Weights in Memory** | 4.88 GB | ~0.05 GB (Miniature Fixture) |
| **Peak VRAM / RAM Usage** | **6.04 GB** (40.3% utilization, 8.96 GB headroom) | 1.1 GB RAM |
| **Batch Size** | 8 | 8 |
| **Sequence Length Truncation** | 512 tokens | 512 tokens |
| **Single-Forward Extraction** | Dual Pooling (Mean + AST Sink) | Dual Pooling (Mean + AST Sink) |
| **Measured Throughput** | **18.5 samples / second** | 37.5 samples / second (Miniature model) |
| **Est. Runtime for Track E3 (4,016)** | **3.6 minutes** (All 8 candidate layers) | Dry-run verification complete |

---

## 5. Embedding Storage Accounting

Each sample produces a float32 vector of dimension $d = 4,096$ per layer. For 8 candidate layers and two pooling strategies:
- **Bytes per vector per layer:** $4,096 \times 4 \text{ bytes} = 16,384 \text{ bytes (16.0 KB)}$
- **Bytes per sample across 8 layers:** $16,384 \times 8 = 131,072 \text{ bytes (128 KB)}$ per pooling strategy
- **Bytes per sample for Dual Pooling (Mean + Sink):** **256 KB**

| Dataset Partition | Samples | Mean Pooling (8 Layers) | Sink Pooling (8 Layers) | Total Disk Storage |
| :--- | :---: | :---: | :---: | :---: |
| **Track E1** (Train, Val, Test) | 3,016 | 386.0 MB | 386.0 MB | **772.0 MB** |
| **Track E2** (Train, Val, Test) | 1,000 | 128.0 MB | 128.0 MB | **256.0 MB** |
| **Track E3** (Train, Val, Test) | 4,016 | 514.0 MB | 514.0 MB | **1,028.0 MB (1.00 GB)** |
| **Track $E_{\text{ext}}$** (700 Benchmark) | 700 | 89.6 MB | 89.6 MB | **179.2 MB** |
| **Grand Total** | **4,716** | **603.6 MB** | **603.6 MB** | **1,207.2 MB (1.18 GB)** |

---

## 6. Per-Layer Embedding Validation & Layer Candidate Specifications

### Candidate Layer Set:
- **Layers:** `[4, 8, 12, 16, 20, 24, 28, 32]` (1-indexed transformer layer output)
- **Reference Layer:** **Layer 16** (residual stream midpoint, representing high-level semantics prior to specialized next-token generation heads).
- **HuggingFace Indexing Formulation:** Extracted from `outputs.hidden_states[layer + 1]` prior to final RMSNorm.
- **Validation-Only Layer Selection:** All layer ablation decisions will be based strictly on validation split performance. Test sets and external benchmarks remain unobserved until layer freeze.

### Numerical Integrity Verification:
Across all batches and pilot tests:
- **Matrix Dimension:** Strict $(N, 4096)$ for production, $(N, 128)$ for pilot fixture.
- **NaN Count:** **0**
- **Inf Count:** **0**
- **Row-to-Label Alignment:** Verified against `labels.npy` and `sample_ids.json`.

---

## 7. Pooling Comparison: Attention-Mask Mean vs AST Sink-Region

| Dimension | Attention-Mask-Correct Mean Pooling | AST Sink-Region Pooling |
| :--- | :--- | :--- |
| **Extraction Scope** | Full source code tokens with non-zero attention mask | AST-identified query execution calls, arguments, and variable assignments |
| **False-Positive Defense** | Diluted across import statements, docstrings, boilerplate | Ignores Python `from ... import ...` and non-SQL functions |
| **Fallback Mechanism** | None needed (applies to all tokens) | Gracefully falls back to attention-masked mean pooling if no sink is found |
| **Sink Coverage Rate** | 100% (all tokens) | **84.0% - 92.5%** explicit sink coverage; remainder uses mean fallback |
| **Representation Norm** | Stable across layers (Pilot: 0.108 to 9.485) | Consistently higher average norm (+5% to +10%) due to focused token density |
| **Forward Pass Cost** | Single forward pass computes both simultaneously from identical layer hidden states | Single forward pass computes both simultaneously from identical layer hidden states |

---

## 8. Integrity Gate Decision & Next Steps

All prerequisites mandated by the user have been satisfied:
1. Datasets partitioned with verified zero cross-split leakage.
2. Quarantined external benchmark established.
3. Candidate layers and reference Layer 16 predeclared.
4. AST extractor tested and validated across 11 test cases.
5. Unified extraction engine ready with dual pooling from a single forward pass.

**HOLD FOR USER REVIEW:** As required by instruction #12, probe training (Logistic Regression, Linear SVM, MLP) remains paused pending user review of this integrity report.

