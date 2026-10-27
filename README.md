# Static Semantics-Aware Security Steering for Code LLMs (ISAE)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)

An autonomous research investigation into whether internal representations of program semantics can help steer a Code Large Language Model (Mistral family) toward secure code generation without sacrificing functional correctness.

---

## 1. Research Overview

This repository investigates the empirical research hypothesis:
> *"Can internal representations of program semantics help us understand or improve activation-based security steering for secure code generation, while preserving functional correctness?"*

The system couples:
1. **Contrastive Representation Engineering:** Extracting linear security concept vectors $v_{sec} \in \mathbb{R}^{d_{model}}$ from pairs of safe vs. vulnerable implementations.
2. **Program Semantics Probing:** Training lightweight linear and MLP probes to predict static data-flow properties (e.g., source-to-sink unsanitized reachability).
3. **Static Activation Steering:** Non-destructive forward-hook interventions injecting $h_l \leftarrow h_l + \alpha \cdot v_{sec}$ into transformer residual streams during code generation.
4. **Dual-Aspect Benchmarking:** Decoupled security vulnerability detection and sandboxed functional test execution.

---

## 2. Project Directory Structure

```
isae/
├── README.md                      # Project documentation and reproduction guide
├── LICENSE                        # Apache 2.0 open-source license
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # Package configuration
├── ENVIRONMENT_REPORT.md          # Hardware, memory, and model feasibility audit
├── configs/
│   ├── model.yaml                 # Mistral model and inference settings
│   ├── dataset.yaml               # Split ratios, seeds, and CWE mappings
│   ├── steering.yaml              # Alpha strengths and layer targets
│   └── probe.yaml                 # Semantic probe hyperparameters
├── data/
│   ├── DATASET_CATALOG.md         # Data provenance and vulnerability register
│   ├── raw/                       # Curated paired security dataset
│   └── processed/                 # Leakage-free train, val, and test splits
├── research/
│   ├── LITERATURE_REVIEW.md       # In-depth survey on steering & code probing
│   ├── RELATED_WORK.md            # Comparative matrix against SOTA
│   ├── RESEARCH_GAP.md            # Unaddressed gaps and formal hypotheses
│   └── RESULTS_AND_LIMITATIONS.md # Scientific analysis and limitations
├── src/
│   ├── data/                      # Dataset download, preparation, validation
│   ├── model/                     # Model loader, generator, activation extractor
│   ├── steering/                  # Vector extraction, evaluation, and hooks
│   ├── probes/                    # Linear probe training and calibration
│   └── evaluation/                # Security, functional, and semantic evaluators
├── tests/                         # Unit and integration test suite (100% pass)
├── scripts/
│   └── run_experiments.py         # End-to-end pipeline runner
└── outputs/
    ├── security_vectors.pt        # Computed direction vectors
    ├── vector_evaluation_report.json
    ├── probes/                    # Trained probe artifacts and summaries
    ├── benchmark_results.json     # Detailed evaluation outputs
    └── BENCHMARK_REPORT.md        # Formatted results table
```

---

## 3. Environment & Hardware Compatibility

An audit of the host environment was performed (see [ENVIRONMENT_REPORT.md](ENVIRONMENT_REPORT.md)):
- **Operating System:** Windows 11 Home 64-bit
- **CPU:** Intel Core i5-12500H (12 cores, 16 logical threads)
- **RAM:** 16 GB physical memory
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, CUDA 13.0 driver)
- **Primary Model Choice:** `wandb/pruned_mistral` (2.8B parameters, 12 layers, $d_{model}=4096$, Apache 2.0).
  - Matches the exact `MistralForCausalLM` architecture and hidden-state dimension of Mistral-7B.
  - Fits within hardware memory limits without triggering CUDA Out Of Memory.
  - Supports instant synthetic testing via `--mock` mode for deterministic validation in seconds.

---

## 4. Quickstart & Reproduction Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare and Validate Datasets
```bash
python -m src.data.download_datasets
python -m src.data.prepare_dataset
python -m src.data.validate_dataset
```

### Step 3: Run the Test Suite
Verify that all unit tests for data processing, vector extraction, forward hooks, and evaluation pass:
```bash
pytest -v
```

### Step 4: Run the Full Research Pipeline
To run a fast, end-to-end dry-run with the miniature Mistral instance:
```bash
python scripts/run_experiments.py --mock
```

To run with the primary 2.8B Mistral model:
```bash
python scripts/run_experiments.py --model-id wandb/pruned_mistral --device auto
```

---

## 5. Summary of Experimental Results

From the benchmark run on held-out test splits (see [research/RESULTS_AND_LIMITATIONS.md](research/RESULTS_AND_LIMITATIONS.md)):

1. **Security Vector Separation:** Intermediate layers (Layers 1 and 2) achieved **100% pairwise ranking accuracy** distinguishing safe from vulnerable completions on validation data.
2. **Semantic Probing:** Linear probes targeting `has_source_to_sink_unsanitized_flow` achieved **80% validation accuracy and 0.840 ROC-AUC** at Layer 1 (compared to the 50% baseline).
3. **Activation Steering Hook Performance:** Forward hooks executed cleanly without weight corruption or latency overhead ($<5\%$ runtime difference).
4. **Pareto Trade-off:** Static, continuous steering perturbations without token gating risk inducing syntax collapse at high $\alpha$, highlighting the need for future dynamic, probe-gated activation steering.

---

## 6. Deliverables Checklist

- [x] Environment inspection and feasibility report ([ENVIRONMENT_REPORT.md](ENVIRONMENT_REPORT.md))
- [x] Literature review, related work, and research gap documents ([research/](research/))
- [x] Curated dataset catalog and provenance ([data/DATASET_CATALOG.md](data/DATASET_CATALOG.md))
- [x] Dataset download, preparation, and zero-leakage validation scripts ([src/data/](src/data/))
- [x] Hardware-aware Mistral model and activation extraction pipeline ([src/model/](src/model/))
- [x] Contrastive security vector extraction and evaluation ([src/steering/](src/steering/))
- [x] Trained semantic data-flow probes with calibration analysis ([src/probes/](src/probes/))
- [x] Context-managed static activation steering hook ([src/steering/activation_hook.py](src/steering/activation_hook.py))
- [x] Decoupled security inspection and sandboxed functional execution ([src/evaluation/](src/evaluation/))
- [x] 10/10 passing unit tests ([tests/](tests/))
- [x] End-to-end experiment orchestration script ([scripts/run_experiments.py](scripts/run_experiments.py))
- [x] Research results, limitations, and dynamic gating proposal ([research/RESULTS_AND_LIMITATIONS.md](research/RESULTS_AND_LIMITATIONS.md))
