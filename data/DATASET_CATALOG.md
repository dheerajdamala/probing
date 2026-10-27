# Dataset Catalog & Provenance Register

**Project:** Static Semantics-Aware Security Steering for Code LLMs  
**Last Updated:** 2026-10-27  

---

## 1. Candidate Datasets Investigated

| Dataset Name | Source / Repository | License | Language(s) | Vulnerability Coverage | Contrastive Pairs? | Semantic Labels? | Suitability Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NIST SARD / Juliet Test Suite** | NIST Software Assurance Reference Dataset | Public Domain | C/C++, Java | CWE-119, CWE-120, CWE-787, CWE-89 | Yes (`good()` vs `bad()`) | Yes (Control-flow paths) | Selected for baseline reference; adapted to standardized paired completions. |
| **SecurityEval Benchmark** | GitHub (`spsanderson/SecurityEval`) | MIT License | Python | 138 CWE patterns | Prompt-only (single completion) | Inferred via Semgrep | Prompts utilized for evaluation; completions augmented with contrastive safe/insecure pairs. |
| **SecCodePLT** | Hugging Face / GitHub | Apache-2.0 | Python, C++ | Top-25 CWEs | Some datasets paired | Partial | High quality; verified for evaluation splits. |
| **Curated ISAE Semantic Benchmark** | Project Repository (`data/processed/`) | Apache-2.0 | Python | CWE-89, CWE-78, CWE-22, CWE-79, CWE-119 | **Yes (1:1 Strictly Paired)** | **Yes (Binary source-to-sink taint reachability)** | **Selected Primary Dataset** |

---

## 2. Primary Dataset Specification: `isae_paired_security_v1`

### 2.1 Design and Construction
To rigorously evaluate activation steering and train semantic probes, data must satisfy four conditions:
1. **Strictly Paired Contrast:** Each prompt $p_i$ is accompanied by a ground-truth safe completion $c_{i, safe}$ and a ground-truth vulnerable completion $c_{i, vuln}$ having identical functionality, signature, and intent.
2. **Explicit Semantic Labels:** Each completion is annotated with a formal static semantic property: `has_source_to_sink_unsanitized_flow` ($y \in \{0, 1\}$).
3. **Executable Functional Test Cases:** Each prompt includes automated test cases to evaluate functional correctness independently of security.
4. **Vulnerability Detection Rules:** Each prompt includes an automated security rule (AST and pattern inspector) to detect insecure patterns without executing hostile payloads.

### 2.2 CWE Distribution
- **CWE-89 (SQL Injection):** Unsanitized query string interpolation vs. parameterized query binding.
- **CWE-78 (OS Command Injection):** String concatenation in `os.system`/`subprocess.Popen(..., shell=True)` vs. tokenized list execution.
- **CWE-22 (Path Traversal):** Unchecked relative file path joining vs. `os.path.commonpath` / resolved base directory verification.
- **CWE-79 (Cross-Site Scripting):** Raw string interpolation into HTML output vs. HTML escaping (`html.escape`).
- **CWE-119 / CWE-787 (Buffer / Array Bounds Error):** Unchecked index access vs. explicit boundary and length guard.

### 2.3 Split Methodology & Leakage Prevention
To prevent data contamination and spurious memorization:
- **Grouping:** All variations of a specific programming task or CWE template are grouped strictly within the same split.
- **Stratified Split:** 60% Train (for steering vector extraction and probe training), 20% Validation (for hyperparameter tuning: $\alpha$, layer selection), 20% Test (held-out evaluation).
- **Provenance:** Fully reproducible generation and validation via `src/data/download_datasets.py` and `src/data/validate_dataset.py`.
