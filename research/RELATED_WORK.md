# Related Work and Comparative Analysis

**Project:** Static Semantics-Aware Security Steering for Code LLMs  
**Date:** 2026-10-27  

---

## 1. Taxonomy of Related Works

```
                                  Secure Code Generation
                                            │
        ┌───────────────────────────────────┼──────────────────────────────────┐
        │                                   │                                  │
 Prompt-Level / Context              Model Retraining                 Internal Interventions
 - Secure System Prompts             - SFT on Safe Datasets           - Linear Activation Steering
 - In-Context Safe Examples          - DPO / RLHF Alignment           - Contrastive RepE
 - RAG with Static Rules             - LoRA Adapters                  - Dynamic Runtime Hooks
```

---

## 2. Detailed Comparative Matrix

| Work / Method | Target Domain | Intervention Level | Preserves Weights? | Functional Verification | Key Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pearce et al. (2021)** *Asleep at the Keyboard* | C / Python (CWEs) | Post-hoc empirical audit | N/A (Black-box) | CodeQL / Compiler | Merely observational; no intervention proposed. |
| **Zou et al. (2023)** *Representation Engineering (RepE)* | Natural Language (Honesty, Morality, Emotion) | Internal residual stream ($h + \alpha v$) | Yes | Multi-choice QA accuracy | Focused on high-level conversational attributes, not syntactic/algorithmic execution. |
| **Turner et al. (2023)** *Activation Addition* | Natural Language (Topic, sentiment) | Intermediate layer residual addition | Yes | Perplexity / Sentiment score | Causes linguistic degradation at elevated $\alpha$; not evaluated on code. |
| **Troshin & Chirkova (2022)** *AST Probing* | Code Transformers (CodeBERT, GraphCodeBERT) | Diagnostic linear probing | Yes (Read-only) | Probe $R^2$ / Accuracy | Diagnostic only; no steering or generative intervention. |
| **SecCodePLT (2024)** | Python / C++ (CWE suites) | Prompt-based & fine-tuning evaluation | No (for FT) | Unit tests + Semgrep | Does not investigate model hidden states or activation vectors. |
| **Our Proposed Framework (ISAE)** | Code LLMs (Mistral Causal LM) | **Static residual stream steering ($h_l + \alpha v_{sec}$) combined with semantic data-flow probing** | **Yes (Zero-shot parameter freeze)** | **Dual: Isolated Security Scanner + Functional Unit Test Suites** | Evaluated on fixed-layer static steering baseline before dynamic gating. |

---

## 3. Analysis of Existing Implementations

1. **RepE Reference Implementations:**
   - Standard RepE packages implement PCA over contrastive question-answer pairs (`User: Tell a lie vs Tell the truth`).
   - *Limitation when applied to code:* Code tokens have rigid indentation and syntactic dependencies. Indiscriminate token pooling across the entire prompt introduces noise from variable names and comment tokens.
   - *Our adaptation:* Explicit token selection and pooling strategies (e.g., sink-site pooling vs prompt-end pooling) to isolate the decision boundary where the vulnerability is introduced.

2. **Secure Code Benchmarks:**
   - Most benchmarks either run static pattern matchers (which produce false positives on harmless variable names) or run code unsafely.
   - *Our implementation:* Strict separation of security static analysis (checking for known insecure APIs and unparameterized query sinks) and isolated execution for functional test cases.
