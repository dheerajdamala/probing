# Comprehensive Literature Review

**Research Topic:** Static Semantics-Aware Security Steering for Code Large Language Models (LLMs)  
**Status:** Completed Literature Review  
**Date:** 2026-10-27  

---

## 1. Introduction and Foundations

Large Language Models trained on code (such as Mistral, CodeLlama, StarCoder, and DeepSeek-Coder) have demonstrated remarkable capabilities in source code completion and synthesis. However, empirical studies consistently reveal that up to 40% of generated completions contain critical security vulnerabilities matching common weakness enumerations (e.g., CWE-79 Cross-Site Scripting, CWE-89 SQL Injection, CWE-119/787 Buffer Errors, CWE-22 Path Traversal) (Pearce et al., 2021; Asleep at the Keyboard).

Traditional approaches to mitigate insecure code generation rely on:
1. **Reinforcement Learning from Human/AI Feedback (RLHF/RLAIF):** Computationally expensive and prone to reward hacking or catastrophic forgetting of programming syntax.
2. **System Prompting / Few-Shot In-Context Demonstrations:** Sensitive to prompt formatting, adds latency and context window overhead, and frequently overridden by downstream generation tokens.
3. **Post-Hoc Static Analysis and Filtering:** Rejects or attempts to re-generate completions, multiplying inference latency.

Recent advances in mechanistic interpretability offer a distinct paradigm: **Internal Representation Engineering and Activation Steering**.

---

## 2. Activation Steering and Representation Engineering

### 2.1 Theoretical Mechanism
Activation addition (Turner et al., 2023) and Representation Engineering (RepE; Zou et al., 2023) posit that concepts, behaviors, and latent intents are represented as linear directions in the intermediate residual stream of transformer blocks:

$$h_l' = h_l + \alpha \cdot v_{concept}$$

Where:
- $h_l \in \mathbb{R}^{d_{model}}$ is the hidden representation at layer $l$.
- $v_{concept} \in \mathbb{R}^{d_{model}}$ is a normalized concept steering vector computed via contrastive activation pairs $(x_{safe}, x_{vuln})$.
- $\alpha \in \mathbb{R}$ is the steering intensity multiplier.

### 2.2 Vector Extraction Paradigms
- **Mean Difference (Contrastive Centroid Difference):**
  $$v = \frac{1}{|S_{pos}|} \sum_{x \in S_{pos}} h_l(x) - \frac{1}{|S_{neg}|} \sum_{x \in S_{neg}} h_l(x)$$
- **Principal Component Analysis (First Difference PC):**
  Applying PCA to the paired difference vectors $\Delta h_i = h_l(x_{pos}^{(i)}) - h_l(x_{neg}^{(i)})$ captures the axis of maximal variance differentiating the concepts.
- **Support Vector Machine / Logistic Boundary:** Fitting a linear separator on latent states to identify the separating hyperplane normal vector.

### 2.3 Steering Externalities and Safety Trade-offs
Recent investigations (Wang et al., 2024; Arditi et al., 2024) observe that indiscriminate activation steering often incurs severe "steering externalities":
- Degraded functional syntax (token collapse, repetitive loops, or malformed ASTs).
- Over-defensiveness (refusing valid, benign programming prompts).
- Erasure of nuanced algorithmic logic required to satisfy unit tests.

---

## 3. Internal Representations of Program Semantics and Probing

### 3.1 What Do Code LLMs Represent?
A fundamental question in neural code intelligence is whether transformers learn emergent program semantics (control flow, data flow, variable states) or superficial token n-grams.
- **Syntax and Type Probing:** Troshin & Chirkova (2022) and subsequent AST probing works showed that intermediate transformer layers linearly encode abstract syntax tree depth and variable scope.
- **Latent Execution Traces:** Recent studies into execution state tracking (Jin et al., 2024; Ma et al., 2024) demonstrate that hidden states in middle-to-late layers maintain linear approximations of variable values and taint-flow pathways during forward token processing.
- **Source-to-Sink Data Flow:** Security properties in code are governed by program semantics—specifically whether untrusted input flows into sensitive execution sinks (e.g., `strcpy`, `system`, SQL query formatting) without passing through a sanitization sanitizer.

### 3.2 Linear vs. Non-linear Probes
Probing evaluates whether property $Y$ is accessible from representation $H$:
$$P(Y|H) = \sigma(W \cdot H + b)$$
High probing accuracy with a linear classifier demonstrates that the property is encoded as a linear subspace, matching the mathematical premise of linear activation steering.

---

## 4. Benchmarking Secure Code Generation

To rigorously evaluate secure code generation, benchmarks must decouple security analysis from functional validation:
- **Juliet Test Suite (NIST SARD):** Standardized C/C++ and Java test suites with explicit safe (`good()`) and vulnerable (`bad()`) functions covering 118 CWE categories.
- **SecurityEval & SecCodePLT:** Python-focused benchmarks evaluating LLM completions against Bandit/Semgrep static analysis and unit tests.
- **CyberSecEval (Meta, 2023/2024):** Evaluates risk in autocomplete settings across multiple languages.

---

## 5. Summary of State-of-the-Art

| Domain | Established Methods | Key Findings |
| :--- | :--- | :--- |
| **Activation Steering** | RepE (Zou et al.), Activation Addition (Turner et al.) | Shifts model attitude and sentiment; highly sensitive to layer selection and $\alpha$. |
| **Code Probing** | TypeProbe, Execution State Probing | Syntax encoded in early layers; relational data flow and semantic types emerge in middle layers. |
| **Code Security Mitigations** | Fine-tuning, LoRA, RLHF, Prompt Guarding | High retraining cost; prompting degrades with long prefixes; vulnerability rate remains $>20\%$. |
| **Functionality Preservation** | HumanEval pass@k, unit test suites | Critical bottleneck; aggressive safety alignment frequently causes functional regression. |
