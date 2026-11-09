# Probe V3: Full Pretrained Mistral-7B T4 GPU Execution Guide

**Date:** 2026-11-03  
**Target Architecture:** `mistralai/Mistral-7B-Instruct-v0.3` ($d=4,096$, 32 Layers)  
**Execution Environment:** NVIDIA Tesla T4 GPU (Google Colab / Kaggle Notebooks)  
**Corpus:** Frozen Track E2 Dataset (1,000 Samples: 700 Train, 149 Val, 151 Test)  
**Artifact Directory:** `experiments/sqli_probe_v3_full_mistral/`  

---

## 1. Required Files for Upload

To ensure the remote GPU environment is 100% self-contained and reproducible without recreating splits, upload the following bundle:

```
probe_v3_execution_bundle/
├── kaggle_colab_t4_pilot.py       # Main extraction and verification engine
├── probe_v3_t4.ipynb              # Complete executable Jupyter Notebook
├── requirements_gpu.txt           # GPU-compatible dependency specifications
├── extraction/
│   └── ast_sink_extractor.py      # Standalone AST-aware SQL sink region extractor
└── data/
    ├── e2_pilot_20.jsonl          # 20 frozen pilot samples (10 vuln, 10 safe)
    ├── e2_excel/
    │   ├── train.jsonl            # 700 frozen training samples
    │   ├── val.jsonl              # 149 frozen validation samples
    │   └── test.jsonl             # 151 frozen test samples
    └── manifests/
        └── probe_v2_split_manifest.json # Authoritative freeze manifest
```

> [!NOTE]
> All files in `data/e2_excel/` are byte-identical copies of the Probe V2 frozen splits. No datasets or manifests are recreated.

---

## 2. Resource Estimates (Clearly Marked as Estimates)

| Resource | Estimated Consumption | Limit / Availability on T4 Instance | Safety Margin |
| :--- | :--- | :--- | :--- |
| **GPU VRAM (Model Weights)** | **~4.88 GB (Estimate)** | 15.00 GB (Tesla T4) | ~10.12 GB Free |
| **GPU Peak VRAM (Batch Size 8)** | **~6.04 GB (Estimate)** | 15.00 GB (Tesla T4) | ~8.96 GB Free (59.7% Headroom) |
| **System Physical RAM** | **~3.20 GB (Estimate)** | 12.70 GB (Colab) / 13.00 GB (Kaggle) | ~9.50 GB Free |
| **Disk Space (Model Shards)** | **~14.85 GB (Estimate)** | ~78.00 GB (Colab) / ~73.00 GB (Kaggle) | ~60.00 GB Free |
| **Disk Space (E2 Embeddings)** | **~256.00 MB (Estimate)** | ~78.00 GB (Colab) / ~73.00 GB (Kaggle) | Negligible |
| **Extraction Throughput** | **~18.5 samples/sec (Estimate)** | Single forward pass dual pooling | Batch Size 8 |
| **Total Runtime for Full E2** | **~54 seconds (Estimate)** | 1,000 samples across 8 layers | Fast execution |

---

## 3. Step-by-Step Notebook Execution Order

Follow these exact steps in order. Each block corresponds directly to a cell in `probe_v3_t4.ipynb`.

### Step 1: Environment & GPU Check
Verify that an NVIDIA GPU with at least 6.0 GB VRAM is active.

```python
import sys, psutil, torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

assert torch.cuda.is_available(), "CRITICAL: CUDA is unavailable. Switch runtime to GPU (T4)."

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
assert vram_gb >= 6.0, f"Insufficient VRAM: {vram_gb:.2f} GB detected, minimum 6.0 GB required."
```

### Step 2: Install Compatible GPU Dependencies
Install `transformers`, `accelerate`, and `bitsandbytes` without upgrading PyTorch.

```bash
!pip install -q "transformers>=4.40.0,<4.47.0" "accelerate>=0.29.0" "bitsandbytes>=0.43.0" "safetensors>=0.4.2" "sentencepiece>=0.2.0" "huggingface_hub>=0.22.0" "scikit-learn>=1.4.0"
```

### Step 3: Verify bitsandbytes CUDA Compatibility
Ensure the 4-bit NF4 kernel executes properly on the active GPU.

```python
import bitsandbytes as bnb
import torch

print(f"bitsandbytes version: {bnb.__version__}")
linear4 = bnb.nn.Linear4bit(128, 128, bias=False, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16).cuda()
dummy_out = linear4(torch.randn(2, 128, dtype=torch.bfloat16).cuda())
assert dummy_out.shape == (2, 128), "bitsandbytes forward pass failed!"
print("bitsandbytes 4-bit NF4 is fully operational on this GPU.")
```

### Step 4: Hugging Face Authentication (Optional / Rate-Limit Bypass)
If running unauthenticated triggers Hub rate limits, supply a read token:

```python
from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
    print("Authenticated with Hugging Face.")
else:
    print("Proceeding without explicit token.")
```

### Step 5: Verify Data Files and Split Counts
Ensure the frozen dataset splits are present and unmodified.

```python
import os, json

for p in ["data/e2_pilot_20.jsonl", "data/e2_excel/train.jsonl", "data/e2_excel/val.jsonl", "data/e2_excel/test.jsonl"]:
    assert os.path.exists(p), f"Missing dataset file: {p}"

train_n = len([json.loads(l) for l in open("data/e2_excel/train.jsonl")])
val_n = len([json.loads(l) for l in open("data/e2_excel/val.jsonl")])
test_n = len([json.loads(l) for l in open("data/e2_excel/test.jsonl")])
pilot_n = len([json.loads(l) for l in open("data/e2_pilot_20.jsonl")])

assert train_n == 700 and val_n == 149 and test_n == 151, "Frozen split sample count mismatch!"
assert pilot_n == 20, "Pilot record count mismatch!"
print(f"Data Verified: Train={train_n}, Val={val_n}, Test={test_n}, Pilot={pilot_n}")
```

---

## 4. How to Execute the Model Verification (Step 4)

Run the verification phase. This loads `mistralai/Mistral-7B-Instruct-v0.3` in 4-bit NF4 onto the GPU, tests hard assertions on dimensions (hidden dimension = 4,096, 32 layers), and executes a test forward pass:

```bash
!python kaggle_colab_t4_pilot.py --mode verify --data_dir data --output_dir embeddings
```

**Expected Verification Output:**
```
PHASE 1: PRETRAINED MISTRAL-7B VERIFICATION & GPU INITIALIZATION
Detected GPU: Tesla T4 (15.00 GB VRAM)
Model ID: mistralai/Mistral-7B-Instruct-v0.3
Config Hidden Dimension: 4096
Config Layer Count: 32
Model successfully loaded in ~35-50s.
VRAM Allocated: ~4880 MB | VRAM Reserved: ~5120 MB
Hard Assertions PASSED: 32 Layers, Hidden Dim 4096, Zero NaNs/Infs, Real Pretrained Weights Verified.
Model verification report saved to: embeddings/model_verification_report.json
```

---

## 5. How to Execute the 20-Sample Pilot (Step 5)

Extract the candidate layers `[4, 8, 12, 16, 20, 24, 28, 32]` for 10 vulnerable and 10 safe samples with dual pooling:

```bash
!python kaggle_colab_t4_pilot.py --mode pilot --data_dir data --output_dir embeddings
```

### Inspect Pilot Report & Audit Gate:

```python
import json

with open("embeddings/pilot_extraction_report.json") as f:
    report = json.load(f)

print(f"Pilot Status: {report['status']}")
print(f"Sample Count: {report['sample_count']} (Vuln: {report['vulnerable_count']}, Safe: {report['safe_count']})")
print(f"Throughput: {report['throughput_samples_per_sec']:.2f} samples/sec")

for lyr, info in report["layer_audit"].items():
    print(f"  {lyr}: Shape={info['dimension']}, Mean Norm={info['mean_norm']:.3f}, Sink Norm={info['sink_norm']:.3f}")

assert report["status"] == "PILOT_VALIDATION_PASSED", "STOP: Pilot validation failed! Do not proceed to full dataset."
print("GATE PASSED: Proceeding to full extraction is authorized.")
```

---

## 6. How to Execute the Full E2 Extraction (Step 6)

Only after the pilot passes, extract all 1,000 samples across the frozen train, validation, and test splits:

```bash
!python kaggle_colab_t4_pilot.py --mode full_e2 --data_dir data --output_dir embeddings --batch_size 8
```

### Expected Output Summary:
```
PHASE 3: FULL TRACK E2 EXTRACTION (1,000 FROZEN SAMPLES)
--- Extracting Split: TRAIN (700 samples) ---
Extraction complete in ~38.0s (18.4 samples/sec).
--- Extracting Split: VAL (149 samples) ---
Extraction complete in ~8.1s (18.5 samples/sec).
--- Extracting Split: TEST (151 samples) ---
Extraction complete in ~8.2s (18.4 samples/sec).
FULL EXTRACTION COMPLETE in ~54.3s.
Manifest written to: embeddings/full_e2_extraction_manifest.json
```

---

## 7. Packaging & Archive Creation for Download

Compress the extracted matrices, sample ID maps, and manifests into an archive:

```bash
!zip -r mistral_probe_v3_e2_embeddings.zip embeddings/full_e2/ embeddings/*.json
```

In Google Colab, trigger direct browser download:
```python
from google.colab import files
files.download("mistral_probe_v3_e2_embeddings.zip")
```

In Kaggle, the file will be available under `/kaggle/working/mistral_probe_v3_e2_embeddings.zip` for direct download from the output panel.

---

## 8. Post-Extraction Workflow (Back on Local Workstation)

Once downloaded back to the workstation:
1. Unzip into `experiments/sqli_probe_v3_full_mistral/embeddings/full_e2/`.
2. Execute the downstream probe training script (Logistic Regression, Linear SVM) on the true 4,096-dimensional representations.
3. Compare against the TF-IDF, Regex, and Majority Class baselines on identical sample IDs.
4. Run 5-seed shuffled-label and random-vector controls.

