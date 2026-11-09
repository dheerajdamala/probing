# Dataset Card: PySQLi-4k Benchmark & Training Dataset

## 1. Dataset Summary

The **PySQLi-4k** dataset is an empirical, multi-source collection of Python source code functions and scripts specifically centered on **SQL query construction, database interaction, and SQL injection vulnerabilities (CWE-89)**.

The dataset was curated from 11 distinct public repositories, research benchmarks, and commit archives (including Hugging Face Datasets, Zenodo archives, and open-source vulnerability databases). Every sample preserves its source dataset, repository, commit SHA, file path, original license, and vulnerability label.

## 2. Dataset Counts and Breakdown

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Raw Records Ingested** | **5,405** | Cumulative candidates from 11 public sources |
| **Total Python Source Records** | **5,405** | 100% verified Python code |
| **Total Database / SQL Relevant** | **5,405** | Filtered by SQL execution and database driver usage |
| **Total After Exact Code Deduplication** | **3,773** | Normalized SHA-256 code hash deduplication |
| **Total Unique Records (Final Validated)** | **3,032** | After near-duplicate Jaccard filtering and pair preservation |
| **Validated Vulnerable Samples (`label=1`)** | **591** | SQL query string formatting, unparameterized concatenation, CWE-89 commits |
| **Validated Non-Vulnerable Samples (`label=0`)** | **2,441** | Parameterized queries, ORM queries, patched security commits |
| **Samples with Ground-Truth Span Annotations** | **116** | CodeQL AST alerts from Zenodo-5225651 & DetectVul |
| **Excluded Records** | **2,373** | 306 syntax/length errors, 1,326 exact duplicates, 741 near-duplicates |

### Target Accounting Statement

The project specification aimed for at least 4,000 unique qualifying examples from public repositories. Following rigorous multi-source harvesting across 11 public datasets and strict deduplication (which removed 2,067 duplicate or unparseable samples), the final validated unique corpus stands at **3,032 unique Python SQL samples**. In accordance with Section 3, Rule 10 ("never claim that the target was met unless the final validated dataset actually meets it"), we report this exact measured count of 3,032 unique records and document the empirical constraints of public Python SQLi availability.

## 3. Contributing Sources and Provenance

| Source Name | Repository / DOI | License | Ingested Count | Unique Kept |
| :--- | :--- | :--- | :--- | :--- |
| **S-AIR-L/PyCode-Vul** | Hugging Face (`S-AIR-L/PyCode-Vul`) | Apache-2.0 | 4,547 | 2,492 |
| **Zenodo Copilot CWE Scenarios** | Zenodo DOI: `10.5281/zenodo.5225651` | CC-BY-4.0 | 484 | 338 |
| **janhvi-ag/cvefixes_pyvul_merged**| Hugging Face (`janhvi-ag/cvefixes_pyvul_merged`) | MIT | 40 | 38 |
| **cmonplz/Python_Vulnerability_Remediation** | Hugging Face (`cmonplz/Python_Vulnerability_Remediation`) | MIT | 146 | 92 |
| **maddyrucos/code_vulnerability_python** | Hugging Face (`maddyrucos/code_vulnerability_python`) | MIT | 67 | 45 |
| **DetectVul/CVEFixes** | Hugging Face (`DetectVul/CVEFixes`) | Apache-2.0 | 44 | 14 |
| **Meta CyberSecEval** | Hugging Face (`purpcode/CyberSecEval-SCG`) | CC-BY-NC-4.0 | 34 | 7 |
| **scthornton/securecode** | Hugging Face (`scthornton/securecode`) | Apache-2.0 | 21 | 3 |
| **felixwangg/pyvul_curated_minus** | Hugging Face (`felixwangg/pyvul_curated_minus`) | MIT | 8 | 1 |
| **Bandit / CodeQL Reference Suite** | GitHub (`PyCQA/bandit`) | Apache-2.0 | 12 | 1 |
| **s2e-lab/SecurityEval** | Hugging Face (`s2e-lab/SecurityEval`) | Apache-2.0 | 2 | 1 |
| **Total** | | | **5,405** | **3,032** |

## 4. Leakage-Resistant Split Protocol

Partitions were created using cluster-grouped stratified splitting:
- **Grouping Unit:** `(repository, pair_id)`. Vulnerable code and its patched remediation pair are guaranteed to reside within the identical partition.
- **Split Proportions:** 70% Train, 15% Validation, 15% Test.
- **Leakage Verification:** 
  - Cross-split exact code hash overlap: **0**
  - Cross-split pair ID overlap: **0**
  - Cross-split near-duplicate overlap: **0**

### Partition Statistics

| Split | Total Samples | Vulnerable (`label=1`) | Non-Vulnerable (`label=0`) | Vulnerability Rate | Span Annotated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2,122 | 404 | 1,718 | 19.04% | 85 |
| **Validation** | 454 | 96 | 358 | 21.15% | 15 |
| **Test (Held-Out)** | 456 | 91 | 365 | 19.96% | 16 |
| **Combined** | **3,032** | **591** | **2,441** | **19.49%** | **116** |

## 5. Label Convention and Schema

- `1`: Vulnerable to SQL injection (unparameterized string formatting, dynamic SQL concatenation, vulnerable commit).
- `0`: Non-vulnerable to SQL injection (parameterized bind variables, safe ORM query, security patch).
