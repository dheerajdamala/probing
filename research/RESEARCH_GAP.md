# Research Gap and Hypothesis Formulation

**Project:** Static Semantics-Aware Security Steering for Code LLMs  
**Date:** 2026-10-27  

---

## 1. Unaddressed Research Questions

Despite significant separate advancements in (a) activation steering for natural language alignment and (b) probing neural representations for code understanding, critical gaps remain at their intersection:

### Gap 1: Absence of Security-Specific Steering Directions in Code LLMs
Previous representation engineering work has largely focused on conversational attributes (truthfulness, toxicity, emotion, refusal). It remains unproven whether **code security** (e.g., memory safety, input sanitization, parameterized queries) forms a coherent, linearly steerable concept vector in the residual stream of modern code-capable causal language models like Mistral.

### Gap 2: The Security–Functionality Pareto Frontier Under Activation Perturbations
Unlike natural language—where slight semantic drift produces valid, readable prose—code has zero syntax tolerance. Adding an activation vector $v_{security}$ risks disrupting syntax tokens (indentation, parentheses, keywords) or core functional algorithms. Prior works have not quantified the **Pareto frontier** between security vulnerability reduction and functional test suite pass rate under varying steering strengths $\alpha$.

### Gap 3: Disconnect Between Static Semantic Understanding and Steering Efficacy
While diagnostic probing can demonstrate that a model's hidden states encode whether a source-to-sink data flow is sanitized, existing steering techniques blindly apply a single static vector without assessing whether the model already possesses the internal representation of that semantic property. Understanding how internal semantic probe accuracy correlates with layer-wise steering effectiveness is an open empirical question.

---

## 2. Formal Research Hypothesis

We formulate the core empirical hypothesis as follows:

> **Hypothesis $H_1$:** Contrastive representations of safe versus vulnerable code completions isolate a directional vector $v_{sec} \in \mathbb{R}^{d_{model}}$ in the intermediate residual layers of a Mistral architecture, such that static intervention:
> $$h_l \leftarrow h_l + \alpha \cdot v_{sec}$$
> during autoregressive generation reduces the incidence of security vulnerabilities without collapsing functional syntactic validity.

> **Diagnostic Hypothesis $H_2$:** The layers exhibiting the highest linear probing accuracy for underlying semantic data-flow properties (e.g., presence of sanitization / bound checks) correspond to the layers where static activation steering produces the most favorable security-to-functionality trade-off.

---

## 3. Scope of Initial Phase (Static System)

To maintain scientific rigor:
1. We implement and evaluate **static activation steering** with fixed layers, fixed vectors, and fixed strengths ($\alpha$).
2. We implement a standalone **semantic probe** to inspect layer representations without coupling it as a dynamic controller.
3. We establish strict control conditions:
   - Baseline unsteered model.
   - Steered model across $\alpha \in \{0.5, 1.0, 2.0, 3.0, 5.0\}$.
   - Random vector control (steering with orthogonal random directions).
   - In-context secure prompting baseline.
