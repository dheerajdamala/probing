# Environment Inspection and Feasibility Report

**Project:** Static Semantics-Aware Security Steering for Code LLMs  
**Inspection Date:** 2026-10-27  
**Host Architecture:** x86_64 / Windows 11 Home (Build 10.0.26300)  

---

## 1. Hardware & System Specifications

| Component | Specification | Available / Usable Capacity | Status |
| :--- | :--- | :--- | :--- |
| **OS** | Microsoft Windows 11 Home 64-bit | - | Verified |
| **CPU** | 12th Gen Intel(R) Core(TM) i5-12500H (12 Cores, 16 Logical Processors) | 16 logical threads | High performance |
| **System RAM** | 16,454,388 KB (~16.0 GB) | Dynamic swap, ~1.0–2.0 GB immediately uncommitted | Moderate (requires lean memory budgeting) |
| **Dedicated GPU**| NVIDIA GeForce RTX 3050 Laptop GPU | 4096 MiB VRAM (~3.95 GB free) | Functional (Compute Capability 8.6, CUDA 13.0 driver) |
| **Integrated GPU**| Intel(R) Iris(R) Xe Graphics | 2147 MB shared | Secondary display only |
| **Disk Storage** | NVMe Drive `C:\` (57.88 GB free of 429.4 GB) | **53.9 GB free** | Sufficient for small-to-medium checkpoints (<10 GB total) |

---

## 2. Software & Toolchain Inspection

- **Python Runtime:** Python 3.12.1 (tags/v3.12.1:2305ca5, Dec 7 2023, 64-bit).
- **Package Manager:** `pip` 26.1 / standard Python packaging toolchain.
- **Key Pre-installed Libraries:**
  - `transformers` 5.3.0
  - `tokenizers` 0.22.2
  - `torch` 2.7.1
  - `scikit-learn` 1.7.2
  - `pandas` 2.3.2, `numpy` 1.26.4
  - `safetensors` 0.6.2, `scipy` 1.15.3, `pyyaml` 6.0.2
  - `pytest` 9.0.2

---

## 3. Execution Environment Decision

**Decision:** **Local Execution with Hybrid GPU/CPU & Mock Validation Modes.**
- The project runs locally on the host machine.
- Device allocation is made fully dynamic: PyTorch checks `cuda` availability with safety fallbacks to `cpu`.
- For fast integration testing, deterministic CI/CD, and regression tests, a miniature synthetic Mistral configuration is provided (`num_layers=4, hidden_size=128`) which executes in milliseconds without network or memory pressure.
- Experiments support standard PyTorch hooks and float16/float32 precision.

---

## 4. Candidate Model Evaluation (Mistral Family)

In adherence to user requirements, only Mistral-family models were evaluated:

| Model Candidate | Parameters | Formats / Footprint | Compatibility / Feasibility Assessment | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Mistral-7B-Instruct-v0.2** (`mistralai`) | 7.24B (32 layers, hidden 4096) | FP16: ~14.5 GB<br>INT4: ~4.2 GB | **Impractical for local GPU without OOM.** A 4.2 GB INT4 footprint exceeds the 4.0 GB total VRAM on the RTX 3050 Laptop GPU once KV-cache and forward activations are allocated. In system RAM (16 GB total), FP16 causes paging thrashing. | Kept as reference/cloud target in `configs/model.yaml` |
| **Codestral-22B-v0.1** (`mistralai`) | 22B | FP16: ~45 GB | **Exceeds available storage and memory limits.** | Not feasible |
| **Ministral-3-3B-Instruct-2512** (`mistralai`) | 3.2B | Multimodal / Pixtral | Requires experimental unreleased `transformers` dev branch; includes vision tower (`Mistral3ForConditionalGeneration`). Not a clean dense causal LM for internal residual stream intervention. | Incompatible with standard causal LM residual hooks |
| **wandb/pruned_mistral** | **2.8B** (12 layers, hidden 4096) | FP16: ~5.36 GB<br>INT4: ~1.5 GB | **Selected Primary Model.** Exact `MistralForCausalLM` architecture with identical layer dimension (`hidden_size: 4096`) to Mistral-7B. Apache 2.0 license, ungated, fits comfortably in CPU memory and in GPU VRAM (with 4-bit/8-bit or batch-size 1 FP16 forward passes). | **SELECTED (Primary)** |
| **Miniature Mistral (Synthetic)** | ~500K (4 layers, hidden 128) | In-memory (~5 MB) | Allows instant test-driven validation, full coverage tests, and deterministic dry-runs on CPU. | **SELECTED (Test Suite)** |

---

## 5. Storage and Memory Budget

- **Model Weights Budget:** ~5.4 GB max for primary model (`wandb/pruned_mistral`).
- **Dataset Budget:** Clean benchmark subsets (Juliet CWE-119/CWE-787/CWE-89/CWE-22 and HumanEval security pairs) < 200 MB.
- **Activation Cache Budget:** Stored in `.safetensors` format, float16; 100 samples $\times$ 12 layers $\times$ 4096 hidden units $\times$ 2 bytes $\approx$ 9.8 MB.
- **Total Storage Required:** ~6.0 GB total, safely well within the 53.9 GB available disk space.
- **Memory Consumption:** CPU peak < 6.0 GB; GPU peak (if used) < 3.2 GB.

---

## 6. Verification Summary

- [x] OS, CPU, RAM, GPU, VRAM, and Storage verified via system queries.
- [x] Hugging Face Hub connectivity and ungated status of `wandb/pruned_mistral` confirmed.
- [x] Tokenizer (SentencePiece/BPE vocab size 32000) and configuration loading verified via live Python execution.
