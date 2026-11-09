# Probe V3 Expanded E2 Mistral-7B GPU Execution Guide

**Target Architecture:** `mistralai/Mistral-7B-Instruct-v0.3` ($d=4,096$, 32 Layers)  
**Extraction Layer:** Layer 12 AST SQL Sink Pooling  
**Expanded Dataset:** $N = 4,673$ total samples (3,304 train, 674 val, 695 test)  
**Historical Reused Samples:** 1,000 samples (700 train, 149 val, 151 test) — verified bit-for-bit compatible  
**Samples Requiring GPU Extraction:** 3,673 samples (2,604 train, 525 val, 544 test)  
**Execution Environment:** NVIDIA Tesla T4 / A10G / A100 GPU (Google Colab / Kaggle Notebooks)  

---

## 1. Executive Summary & Methodology

The Probe V3 architecture relies on genuine 4096-dimensional representations extracted from Layer 12 of `mistralai/Mistral-7B-Instruct-v0.3` using AST-aware SQL sink region pooling.

To extract representations for the 4,673-sample expanded E2 dataset while preserving 100% scientific integrity:
1. **Historical 1,000 Samples Reused:** The 1,000 samples from the historical E2 benchmark (`t_0001`–`t_1000`) were audited and verified to be bit-exact identical in source code and labels to the corresponding entries in the expanded splits. Their Layer 12 AST sink representations have been migrated and verified with cryptographic SHA-256 hashes in `embeddings/historical/`.
2. **Remaining 3,673 Samples Extracted on GPU:** The 3,673 newly collected samples are extracted using the exact same tokenizer, 4-bit NF4 quantization, and Layer 12 AST sink pooling pipeline on a remote GPU.
3. **Deterministic Merge & Strict Verification:** The extraction engine merges the historical and new representations into final `train`, `validation`, and `test` suites where every vector is explicitly bound to its `sample_id`.

---

## 2. Remote GPU Resource Requirements

| Resource | Value | Status on Tesla T4 (15 GB) | Status on A100 (40 GB) |
| :--- | :--- | :--- | :--- |
| **Model Weights (4-bit NF4)** | ~4.88 GB | Supported | Supported |
| **Peak VRAM (Batch Size = 8)** | ~6.04 GB | ~9 GB Headroom (60% Free) | ~34 GB Headroom (85% Free) |
| **System RAM** | ~3.5 GB | Fully Supported | Fully Supported |
| **Throughput (Tesla T4)** | ~0.45 s / sample (~2.2 samples/s) | ~28 min total for 3,673 samples | ~8 min total on A100 |

---

## 3. Step-by-Step GPU Execution Options

### Option A: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `experiments/sqli_probe_v3_full_mistral/expanded_v2/probe_v3_expanded_t4.ipynb`.
3. Select **Runtime** > **Change runtime type** > **Hardware accelerator** > **T4 GPU**.
4. Run all cells sequentially:
   - **Cell 1–3:** Verifies T4 GPU and unpacks `expanded_v2_t4_bundle.zip` (included directly in the notebook as a self-extracting payload).
   - **Cell 4–5:** Authenticates and runs single-sample verification of Mistral-7B in 4-bit NF4 with AST sink extraction.
   - **Cell 6:** Extracts 4,096-d representations for the 3,673 new samples across train (2,604), validation (525), and test (544).
   - **Cell 7:** Automatically packages `mistral_probe_v3_expanded_v2_embeddings.zip` for download.
5. Download `mistral_probe_v3_expanded_v2_embeddings.zip` and place it in the workspace root (`c:\Users\dheer\Desktop\isae\`).

---

### Option B: Kaggle Notebooks
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook** > **File** > **Upload Notebook** and upload `probe_v3_expanded_t4.ipynb`.
3. In the sidebar under **Notebook Options**:
   - **Accelerator:** Select `GPU T4 x1` (or `GPU P100`).
   - **Internet:** Turn **ON** (required to load `mistralai/Mistral-7B-Instruct-v0.3`).
4. Click **Run All** (or run cells one by one).
5. When complete, download `mistral_probe_v3_expanded_v2_embeddings.zip` from the output directory.

---

## 4. Local Ingestion, Merge & Integrity Validation

Once `mistral_probe_v3_expanded_v2_embeddings.zip` is placed in the workspace, run:

```bash
# 1. Unpack extracted new embeddings into new_extracted/
python -c "
import zipfile, os
os.makedirs('experiments/sqli_probe_v3_full_mistral/expanded_v2/embeddings/new_extracted', exist_ok=True)
with zipfile.ZipFile('mistral_probe_v3_expanded_v2_embeddings.zip') as z:
    z.extractall('experiments/sqli_probe_v3_full_mistral/expanded_v2/embeddings/new_extracted')
"

# 2. Merge historical and new representations into final split suites
python experiments/sqli_probe_v3_full_mistral/expanded_v2/extract_expanded_embeddings.py --mode merge

# 3. Execute strict mathematical validation audit
python experiments/sqli_probe_v3_full_mistral/expanded_v2/validate_embeddings.py
```

The validation audit verifies:
- Exactly 3,304 train, 674 validation, and 695 test samples.
- Every vector has dimension 4,096 with zero NaNs, zero Infs, and zero zero-norm vectors.
- Bit-exact float equality between reused historical vectors and Probe V3 historical checkpoints.
- Strict 1:1 mapping between `sample_ids.json` and `data/e2_expanded/splits/{split}.jsonl`.

