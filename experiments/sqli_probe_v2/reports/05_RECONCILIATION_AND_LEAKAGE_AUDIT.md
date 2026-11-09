# Probe V2: Dataset Reconciliation, Storage Accounting, and Leakage Audit

**Date:** 2026-10-31  
**Project:** SQL Injection Security Probe V2 (CWE-89)  
**Status:** Approved Pre-Training Reconciliation  

---

## 1. Reconciliation of Track E1 Sample Count (2,973 vs 3,016)

### Root Cause of the 43-Sample Difference:
The historical `validated_sql_dataset.jsonl` contains **3,032** samples (591 vulnerable, 2,441 safe).

1. **Initial Heuristic Estimate (Report 01):**
   - In exploratory Phase 4 data validation, an aggressive preliminary regex and line-warning filter had flagged **59 samples** (53 vulnerable and 6 safe), yielding an estimated clean count of $3,032 - 59 = 2,973$ samples (538 vulnerable, 2,435 safe).
   - These 53 vulnerable snippets had raised Python `SyntaxWarning: invalid escape sequence '\A'` (and `\s`, `\$`) when scanned in raw docstrings or regex patterns.
2. **Authoritative AST Parsing (`create_probe_v2_splits.py`):**
   - When rigorous Python compilation via `ast.parse()` was executed, only **16 samples** produced fatal `SyntaxError`s (10 vulnerable and 6 safe).
   - All 16 fatal syntax errors were git diff hunks from `cvefixes` and `felixwangg` that suffered from `unexpected indent (<unknown>, line 1)` due to being cropped mid-block.
   - The remaining **43 samples** (all vulnerable) were verified to be **100% syntactically valid Python code**. Python raised non-fatal string escape warnings, but successfully constructed full Abstract Syntax Trees.
3. **Audit of the 43 Rescued Samples:**
   - **Provenance:** 32 from `S-AIR-L/PyCode-Vul` and 11 from `Zenodo-5225651-Copilot-CWE`.
   - **Labels:** 100% confirmed CWE-89 vulnerable (`label = 1`).
   - **Sample ID Examples:** `pycode_vul_train_9981_vuln`, `pycode_vul_train_7170_vuln`, `zenodo_5225651_experiments_dop_cwe-89_unsubscribe_scenario_meta_tabs_copilot_19.py`.
   - **Resolution:** These 43 genuine, syntactically valid vulnerable samples were correctly retained rather than discarded, establishing the authoritative Track E1 corpus at **3,016 samples** (581 vulnerable, 2,435 safe).

---

## 2. Reconciled Embedding Storage Calculations

Each embedding representation uses float32 precision across $d = 4,096$:
$$\text{Size per layer} = 4,096 \times 4 \text{ bytes} = 16,384 \text{ bytes (16.0 KB)}$$
$$\text{Size across 8 candidate layers} = 16.0 \text{ KB} \times 8 = 128.0 \text{ KB per pooling method}$$
$$\text{Dual Pooling (Mean + AST Sink)} = 256.0 \text{ KB per sample}$$

### Physical Storage vs. Deduplicated Unique Storage:

Because Track E3 is the combined union of Track E1 and Track E2 ($3,016 + 1,000 = 4,016$), we clearly distinguish between physical duplicated disk consumption and unique deduplicated representation storage:

| Corpus | Sample Count | Pooling Methods | Candidate Layers | Physical Storage (Duplicated) | Deduplicated Unique Storage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Track E1** *(Existing)* | 3,016 | Dual (Mean + Sink) | 8 | 772.1 MB | 772.1 MB |
| **Track E2** *(Excel 1,000)* | 1,000 | Dual (Mean + Sink) | 8 | 256.0 MB | 256.0 MB |
| **Track E3** *(Combined)* | 4,016 | Dual (Mean + Sink) | 8 | 1,028.1 MB | **0.0 MB** *(100% subset of E1+E2)* |
| **Track $E_{\text{ext}}$** *(700 Benchmark)* | 700 | Dual (Mean + Sink) | 8 | 179.2 MB | 179.2 MB |
| **Total Project Footprint** | **4,716 Unique** | Dual (Mean + Sink) | 8 | **2,235.4 MB (2.24 GB)** | **1,207.3 MB (1.21 GB)** |

- **Physical Materialization:** 2.24 GB if Track E3 is duplicated into separate directory trees.
- **Deduplicated / Indexed Storage:** 1.21 GB when Track E3 references E1 and E2 via sample ID index maps.

---

## 3. Explicit Repository-Level & Near-Duplicate Leakage Audit

Exact cryptographic SHA256 checking ensures 0 exact string leaks. However, code datasets can suffer from semantic and repository-level leakage. We conducted an explicit near-duplicate and repository audit:

### 3.1 Repository Distribution and Overlap Statistics

| Metric | Track E1 | Track E2 | Track E3 |
| :--- | :---: | :---: | :---: |
| **Total Unique Repositories** | 31 | 1 | 32 |
| **Train / Val / Test Repositories** | 28 / 16 / 15 | 1 / 1 / 1 | 29 / 16 / 17 |
| **Train-Val Repo Overlap** | 15 / 16 (93.8%) | 1 / 1 (100.0%) | 15 / 16 (93.8%) |
| **Train-Test Repo Overlap** | 13 / 15 (86.7%) | 1 / 1 (100.0%) | 15 / 17 (88.2%) |
| **Dominant Repositories** | Airflow (692), Django (617), Copilot (474), Sqlmap (354) | Curated Synthetic (1,000) | Airflow (692), Django (617), Curated (1,000) |

**Key Finding:** In naturalistic datasets like E1, while paired samples (`_vuln` and `_patch`) are strictly quarantined to the same split, different modules/commits from large repositories like `apache/airflow` and `django/django` inevitably exist across splits. This makes a **repository-held-out evaluation** essential to detect whether probes learn repository-specific syntax rather than SQL injection semantics.

### 3.2 Token-Level Near-Duplicate (Jaccard) Distribution

Token n-gram Jaccard similarity between Test code and Train code:

| Metric | Track E1 | Track E2 | Track E3 |
| :--- | :---: | :---: | :---: |
| **Mean Test-to-Train Jaccard** | **0.596** | **0.405** | **0.465** |
| **Median Test-to-Train Jaccard** | 0.645 | 0.400 | 0.386 |
| **95th Percentile Jaccard** | 0.974 | 0.571 | 0.953 |
| **Max Jaccard** | 1.000 (Scaffolding match) | 0.750 | 0.992 |
| **Samples with Jaccard $\ge 0.85$** | **28.0%** | **0.0%** | **20.0%** |

**Interpretation:**
- **Track E1** exhibits high near-duplicate similarity (28% with Jaccard $\ge 0.85$) due to synthetic Copilot prompt templates sharing identical wrapper boilerplate code.
- **Track E2** exhibits substantially greater code diversity (0% near-duplicates $\ge 0.85$, max 0.750).
- **Track E3** dilutes the templated clustering of E1 with the diverse structures of E2.

---

## 4. Training Protocol & Mitigation Plan

To address these empirical findings during probe training:
1. **Repository-Held-Out Evaluation:** A dedicated Leave-One-Repository-Out (LORO) evaluation will be executed, holding out `apache/airflow` and `django/django` entirely to evaluate out-of-distribution generalization.
2. **Hard-Negative Evaluation:** A dedicated test on safe database queries (parameterized SQL, ORM queries) will be run to verify whether the probe relies on the mere presence of SQL keywords.
3. **Controls:** Shuffled-label controls (5 random seeds) and random Gaussian vector controls will be run alongside all neural probes to establish true null baselines.
4. **Validation-Only Tuning:** All hyperparameters ($C$, layer selection, decision threshold) will be tuned exclusively on the validation split; test sets and the external 700-sample benchmark will remain unobserved until final freeze.

