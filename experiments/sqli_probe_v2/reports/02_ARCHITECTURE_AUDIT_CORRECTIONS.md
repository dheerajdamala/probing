# Probe V2: Architecture Audit Corrections & Full Mistral Specification

**Project:** SQL Injection Security Probe V2 (`experiments/sqli_probe_v2/`)  
**Date:** October 31, 2026  
**Auditor:** Automated Empirical Audit Suite  

---

## 1. Executive Summary & Critical Corrections

This document establishes formal architectural corrections regarding model scope, representation dimensions, hidden layer indexing, and layer selection protocols.

### Critical Correction on Model Attribution
> **Correction Notice:** The ROC-AUC of **0.8237** and Average Precision of **0.5007** reported in Phase 6/7 were computed using a **128-dimensional miniature Transformer unit fixture** (`create_miniature_mistral`, 4 layers, $d=128$, randomly initialized weights) running locally on CPU. That score represents the discriminatory capacity of localized lexical sink pooling over miniature embeddings, **not full pretrained Mistral-7B**. 
>
> In Probe V2, all production claims and representation analyses must be derived from the actual pretrained **`mistralai/Mistral-7B-Instruct-v0.3`** model ($d=4,096$, 32 layers) executing on GPU compute infrastructure.

---

## 2. Full Pretrained Mistral Architectural Specification

| Architectural Property | Miniature Experimental Fixture | Target Production Model (`Mistral-7B-Instruct-v0.3`) |
|---|---|---|
| **Hugging Face Model ID** | Synthetic (`create_miniature_mistral`) | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Parameter Count** | ~500,000 | **7.24 Billion** |
| **Number of Transformer Layers** | 4 | **32** |
| **Hidden Representation Dimension ($d$)** | **128** | **4,096** |
| **Intermediate Size (MLP)** | 512 | 14,336 |
| **Attention Heads / KV Heads** | 4 / 2 (GQA) | 32 / 8 (Grouped Query Attention) |
| **Vocabulary Size** | 32,000 | 32,768 |
| **Weight Initialization** | Random Gaussian (Seed 42) | Official Pretrained & Instruction-Tuned Weights |
| **Primary Execution Platform** | Local Windows CPU Workstation | Cloud GPU Workstation (Kaggle / Colab T4 / A10G) |
| **Quantization Format on T4** | N/A (FP32) | 4-bit NF4 (`bitsandbytes`, double quantization) |

---

## 3. Layer Sweep & Hidden-State Indexing Specification

### 1. Predeclared Candidate Layers
To systematically evaluate the depth progression of security representations, V2 predeclares 8 candidate layers spanning the transformer's depth:
$$\text{Candidate Layers} = [4, 8, 12, 16, 20, 24, 28, 32]$$
- **Reference Configuration:** **Layer 16** (exact mid-network residual stream corresponding to the project's security-steering intervention vector).
- **Early Layers (4, 8):** Capture lexical syntax and local token adjacency.
- **Mid Layers (12, 16, 20):** Capture semantic role assignment, variable bindings, and inter-procedural argument flow.
- **Deep Layers (24, 28, 32):** Capture task-level decision-making and instruction following.

### 2. Exact Hugging Face Tensor Indexing
In the Hugging Face `transformers` implementation (`MistralForCausalLM`), the tuple returned by `outputs.hidden_states` has length `num_hidden_layers + 1` (33 elements for 32 layers):
- `outputs.hidden_states[0]`: Embedding layer output (`model.embed_tokens`).
- `outputs.hidden_states[1]`: Output of Decoder Block 0 (Layer 0).
- `outputs.hidden_states[l]`: Output of Decoder Block $l - 1$ (Layer $l - 1$).
- `outputs.hidden_states[17]`: **Layer 16** (Decoder Block 15 output).
- `outputs.hidden_states[32]`: **Layer 31** (Final Decoder Block output).
- `outputs.hidden_states[33]`: (If present in certain versions, post-final-norm).

### 3. Residual Stream Normalization Point
- In `MistralForCausalLM`, each decoder block outputs directly into the residual stream:
  $$\mathbf{x}_{l} = \mathbf{x}_{l-1} + \text{MLP}(\text{RMSNorm}(\mathbf{x}_{l-1} + \text{Attn}(\text{RMSNorm}(\mathbf{x}_{l-1}))))$$
- The extracted representations $\mathbf{h}^{(l)}$ are taken directly from the **unnormalized residual stream** at the exit of block $l$.
- The final layer normalization (`model.norm`, RMSNorm) is applied only before the final LM head (`lm_head`). For all intermediate candidate layers ($l \in \{4, 8, 12, 16, 20, 24, 28\}$), representations are raw residual stream states before final normalization.

---

## 4. Layer Selection & Threshold Tuning Protocol

To prevent test-set data leakage and overfitting:
1. **Zero Test Data in Selection:** The held-out test split is strictly locked and inaccessible during layer selection, pooling selection, and threshold tuning.
2. **Optimization Objective:** The optimal layer $l^*$ and optimal decision threshold $\tau^*$ are chosen strictly by evaluating the **validation split Average Precision (PR-AUC)**:
   $$l^* = \arg\max_{l \in \{4, 8, \dots, 32\}} \text{AveragePrecision}(\text{Probe}^{(l)}, \mathcal{D}_{\text{val}})$$
3. **Single Test Evaluation:** Only the probe corresponding to the validation-selected layer $l^*$ and threshold $\tau^*$ is evaluated on the held-out test split.

