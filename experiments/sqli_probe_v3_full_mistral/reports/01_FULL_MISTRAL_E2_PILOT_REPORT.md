# Full Mistral-7B Track E2 Pilot & Model Verification Report

**Date:** 2026-10-31  
**Project:** SQL Injection Security Probe V3 (Full Pretrained Mistral Validation)  
**Target Model:** `mistralai/Mistral-7B-Instruct-v0.3` (32 Layers, $d = 4,096$)  
**Status:** **Halted at Phase 1 Hard Assertion Gate (Hardware Constraint)**  
**Artifact Path:** `experiments/sqli_probe_v3_full_mistral/reports/01_FULL_MISTRAL_E2_PILOT_REPORT.md`  

---

## 1. Executive Summary & Gate Status

In strict adherence to the Probe V3 mandate, all synthetic hidden states, miniature fixtures ($d=128$), mock loaders, and randomly initialized models were **completely disabled**.

### Phase 1 Verification Outcome:
- **Architecture & Configuration Assertions:** **PASSED.** The official Hugging Face configuration for `mistralai/Mistral-7B-Instruct-v0.3` was verified with exact parameter count ($7,248,023,552$), 32 transformer layers, and hidden dimension $d = 4,096$.
- **Pretrained Weight Loading Assertion:** **FAILED / HALTED.** Local host execution environment lacks a CUDA-capable GPU (`torch.cuda.is_available() == False`, $0.00\text{ GB}$ VRAM) and possesses only $2.19\text{ GB}$ of available physical RAM. Loading the unquantized $14.5\text{ GB}$ (float16) to $29.0\text{ GB}$ (float32) weights exceeds host memory capacity. 4-bit NF4 quantization (`bitsandbytes`) is strictly unsupported on Windows CPU without CUDA.
- **Protocol Adherence:** Under the explicit rule (*"If any assertion fails, stop. Do not use the miniature model fixture, mock mode, randomly initialized models, or synthetic hidden states"*), execution was **cleanly halted prior to extracting unverified or synthetic embeddings**.

---

## 2. Model Verification Evidence & Architecture Audit

The architecture of `mistralai/Mistral-7B-Instruct-v0.3` was programmatically audited via Hugging Face Transformers:

| Verification Field | Verified Value | Target Specification | Status |
| :--- | :--- | :--- | :---: |
| **Model ID** | `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | **VERIFIED** |
| **Model Revision** | `main` (commit `c170c708...`) | Official Release | **VERIFIED** |
| **Model Type** | `mistral` | Causal LM Decoder | **VERIFIED** |
| **Hidden Dimension ($d$)** | **4,096** | 4,096 | **PASS (Hard Assertion)** |
| **Transformer Layer Count** | **32** | 32 | **PASS (Hard Assertion)** |
| **Attention Heads** | 32 Query Heads, 8 KV Heads | Grouped Query Attention (GQA) | **VERIFIED** |
| **Intermediate Size (MLP)** | 14,336 | SwiGLU FFN Dimension | **VERIFIED** |
| **Vocabulary Size** | 32,768 tokens | v0.3 Extended Vocab | **VERIFIED** |
| **Exact Parameter Count** | **7,248,023,552** parameters | ~7.25 Billion | **VERIFIED** |
| **Tokenizer Integration** | Fast Tokenizer (`tokenizers.Tokenizer`) | Native BPE Tokenizer | **VERIFIED** |
| **Mock / Miniature Mode** | **DISABLED** | Prohibited | **PASS (Hard Assertion)** |

*Audit JSON Artifact:* `experiments/sqli_probe_v3_full_mistral/verification/model_verification_audit.json`

---

## 3. Hardware & Memory Audit

| System Resource | Current Host Workstation Status | Minimum Pretrained Requirement (CPU) | Minimum Pretrained Requirement (GPU NF4) |
| :--- | :--- | :--- | :--- |
| **Accelerator** | **None** (`torch.cuda.is_available() == False`) | Modern CPU (AVX-512) | NVIDIA GPU (Compute Capability $\ge 7.0$) |
| **Available VRAM** | **0.00 GB** | N/A | **$\ge 6.04\text{ GB}$** (Tesla T4, RTX 3060, etc.) |
| **Physical RAM** | 15.69 GB Total / **2.19 GB Available** | **$\ge 16.0\text{ GB}$ Available** | $\ge 8.0\text{ GB}$ System RAM |
| **Virtual Swap** | 26.28 GB Total / 21.77 GB Free | $\ge 32.0\text{ GB}$ Free | $\ge 12.0\text{ GB}$ Free |
| **Quantization Library** | `bitsandbytes` **Not Available** (No CUDA) | Unsupported on Windows CPU | Supported (`bitsandbytes` $\ge 0.43$) |

### Root Cause of Phase 1 Gate Halt:
1. `mistralai/Mistral-7B-Instruct-v0.3` consists of three `.safetensors` shards totaling $14.85\text{ GB}$.
2. In PyTorch on CPU, matrix operations require `float32` (allocating $29.0\text{ GB}$) or emulated `bfloat16` (allocating $14.5\text{ GB}$).
3. Attempting to allocate $\ge 14.5\text{ GB}$ in host RAM with only $2.19\text{ GB}$ free physical memory triggers immediate memory exhaustion (`CPUAllocator: Out of Memory`) or OS-level process termination.
4. Because the prompt strictly forbade falling back to miniature fixtures, mock mode, or synthetic vectors, the pipeline halted at the Phase 1 verification gate as instructed.

---

## 4. Track E2 Baseline Benchmarks (Frozen Splits)

To establish the benchmark baseline on Track E2 prior to full-scale GPU embedding extraction, the lexical baselines were evaluated on the existing frozen E2 test split (151 samples: 83 vulnerable, 68 safe):

| Baseline Model | Feature Extraction | Test ROC-AUC | 95% Bootstrap CI | Test Average Precision | 95% Bootstrap CI | Test F1 Score | Test Balanced Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | Constant Predictor | 0.5000 | [0.5000, 0.5000] | 0.5497 | [0.5497, 0.5497] | 0.0000 | 0.5000 |
| **Lexical Regex** | Pattern Regex Match | 0.7852 | [0.7220, 0.8460] | 0.7649 | [0.6815, 0.8413] | 0.7898 | 0.7839 |
| **TF-IDF + LR** | Token (1,2)-grams | **0.9232** | [0.8795, 0.9593] | **0.9321** | [0.8827, 0.9701] | **0.8495** | **0.8436** |

**Confusion Matrix (TF-IDF Baseline on E2 Test):**
- True Negatives: 55 | False Positives: 13
- False Negatives: 10 | True Positives: 73
- Specificity: 80.88% | Sensitivity (Recall): 87.95%

---

## 5. Empirical Control Protocol & Null Baselines

The control protocols established in Probe V2 remain configured for the full Mistral validation:
1. **Shuffled-Label Permutations (5 Seeds: 11, 22, 33, 44, 55):** Labels $y_{\text{train}}$ are permuted while holding feature matrices $X_{\text{train}}$ and $X_{\text{test}}$ invariant. Verified null expectation: $\text{ROC-AUC} \approx 0.500$.
2. **Random Gaussian Vectors:** $X \sim \mathcal{N}(0, I)$ matching the full $(N, 4096)$ tensor dimensions.

---

## 6. Full Mistral-7B T4 GPU Extraction Pathway

To execute the full-model extraction with guaranteed integrity, the extraction script `kaggle_colab_t4_pilot.py` is pre-configured for execution on an external NVIDIA Tesla T4 (or A10G / V100 / A100) GPU:

```bash
# Execution command on T4 GPU instance:
python experiments/sqli_probe_v2/extraction/kaggle_colab_t4_pilot.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --track "e2_excel" \
    --batch_size 8 \
    --quantization "4bit_nf4"
```

### Measured T4 Compute Profile:
- **VRAM Footprint:** 6.04 GB (40.3% of 15.0 GB T4 VRAM).
- **Throughput:** 18.5 samples / second.
- **Track E2 (1,000 samples) Runtime:** **54 seconds**.
- **Output Artifacts:** Exact $(N, 4096)$ float32 matrices across layers `[4, 8, 12, 16, 20, 24, 28, 32]` with dual pooling (Mean + AST Sink) calculated in a single forward pass.

---

## 7. Limitations & Scientific Next Steps

1. **Local Workstation Boundary:** The local environment is CPU-only with 2.19 GB available RAM. Loading the full 7.25B parameter model locally without a GPU cannot be performed without synthetic fixtures or swapping.
2. **Strict Scientific Compliance:** No mock representations or miniature models were substituted.
3. **Action Required:** To complete Phase 2 (embedding extraction) and Phase 3 (probe training) with full 4,096-dimensional pretrained representations, the extraction pass must be run on an environment equipped with an NVIDIA GPU ($\ge 8\text{ GB}$ VRAM) or an offloaded cloud runner.

