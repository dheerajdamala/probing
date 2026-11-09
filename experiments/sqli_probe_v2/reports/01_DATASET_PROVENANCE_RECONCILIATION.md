# Probe V2: Dataset Provenance & Membership Reconciliation

**Project:** SQL Injection Security Probe V2 (`experiments/sqli_probe_v2/`)  
**Date:** October 31, 2026  
**Auditor:** Automated Empirical Audit Suite  

---

## 1. Executive Summary & Reconciliation Ledger

A rigorous audit was conducted to reconcile the inconsistent dataset counts reported across historical project documents: **3,032**, **3,423**, **700**, **1,000**, and **4,673**.

The table below provides the authoritative record-level mapping, origin, and disposition of each figure:

| Count | Primary Physical Artifact | Origin & Provenance | Vulnerable / Safe Breakdown | Role in Probe V2 |
|---|---|---|---|---|
| **3,032** | `security_probe_sql/data/processed/validated_sql_dataset.jsonl` | Phase 4 initial deduplication of 5,405 raw records from 9 sources (PyCode-Vul, Copilot-CWE, CVEFixes, Bandit, etc.). | 591 Vulnerable / 2,441 Safe (19.5% vuln) | **Track E1 Corpus** (Existing Validated Dataset). |
| **3,423** | `security_probe_sql/outputs/metrics/PHASE4_DATASET_COUNTS.json` | Phase 4 secondary expansion: included 391 additional safe database queries from PyCode-Vul (5,830 raw candidates ingested). Exactly the same 591 vulnerable samples as the 3,032 corpus. | 591 Vulnerable / 2,832 Safe (17.3% vuln) | Historical Phase 4/5 audit record. |
| **700** | `data/processed/expanded_evaluation_3k.json` (CWE-89 subset) | 350 paired synthetic coding tasks created for evaluating LLM security steering. Each task has 1 vulnerable and 1 safe completion ($350 \times 2 = 700$). | 350 Vulnerable / 350 Safe (50.0% vuln) | **Independent External Benchmark ($E_{\text{ext}}$)**. Kept strictly out of training to serve as an unpolluted out-of-domain evaluation benchmark. |
| **1,000** | `sql_injection_train_1000.xlsx` | New reference Excel workbook with 15-column schema, verified fixes, and vulnerable line annotations. | 550 Vulnerable / 450 Safe (55.0% vuln) | **Track E2 Corpus** (Reference Excel Dataset). |
| **4,673** | Normalized merge ($1,000 + 700 + 3,032 - 59$) | Aggregate of Excel (1,000), 3k benchmark (700), and validated dataset (3,032), minus 59 syntax errors quarantined from git diff hunks. | 1,438 Vulnerable / 3,235 Safe (30.8% vuln) | Transitional exploratory aggregate. |
| **3,973** | Clean Deduplicated Merge ($E_1 + E_2$) | **The Authoritative Track E3 Training Corpus**: Merges E1 (3,032 - 59 syntax errors = 2,973) and E2 (1,000) with 0 syntax errors, 0 duplicates, and 0 label conflicts, leaving the 700-example benchmark untouched. | 1,088 Vulnerable / 2,885 Safe (27.4% vuln) | **Track E3 Corpus** (Combined Training Dataset). |

---

## 2. Forensic Audit of Candidate Subsets

### 1. The 3,032 vs 3,423 Discrepancy
- In Phase 4, the initial automated filtering script wrote `validated_sql_dataset.jsonl` with **3,032** samples (591 vulnerable, 2,441 safe).
- Subsequently, the audit script `generate_phase4_dataset_counts.py` scanned an expanded raw cache of 5,830 records that contained an additional 391 safe database queries from `S-AIR-L/PyCode-Vul` ($2,441 + 391 = 2,832$ safe). This brought the count in the JSON report to **3,423**.
- **Crucial finding:** The underlying serialized file committed and used for Phase 5, Phase 6, and Phase 7 feature extractions was `validated_sql_dataset.jsonl` containing **3,032** records. The 591 vulnerable samples are 100% identical between both counts.

### 2. The 700-Example CWE-89 Benchmark (`expanded_evaluation_3k.json`)
- In `data/processed/expanded_evaluation_3k.json`, exactly 350 tasks belong to CWE-89 (SQL Injection). Each task contains a `prompt`, `vulnerable_completion`, and `safe_completion`.
- When parsed into individual code samples, these yield exactly 350 vulnerable samples and 350 safe samples (**700 total**).
- **Benchmark Contamination Risk:** These 700 samples were designed as an independent evaluation benchmark for the security steering vector. Incorporating them into the probe's training corpus would contaminate the benchmark and prevent valid zero-shot evaluation.
- **Resolution:** **The 700 samples are strictly quarantined from training** and designated exclusively as **Independent External Benchmark ($E_{\text{ext}}$)**.

### 3. The 1,000-Example Excel Dataset (`sql_injection_train_1000.xlsx`)
- Contains 1,000 tabular records: 400 normal vulnerable, 150 hard positive vulnerable, 200 safe DB, 150 hard negative safe, 100 no-SQL safe.
- Binary labels: 550 vulnerable (`label=1`), 450 safe (`label=0`).
- Code overlap check: Cryptographic SHA-256 hash comparison between the 1,000 Excel code samples and the 3,032 samples in `validated_sql_dataset.jsonl` revealed **0 exact overlaps** and **0 near-duplicate matches**. The Excel dataset represents 1,000 completely independent Python SQL injection examples.

---

## 3. Authoritative Experimental Corpus Partitioning for V2

Under the reconciled framework, Probe V2 establishes three strictly partitioned training corpora plus one independent external benchmark:

| Experiment Track | Corpus Composition | Total Records | Vulnerable | Safe | Role & Purpose |
|---|---|---|---|---|---|
| **Track E1** | `validated_sql_dataset.jsonl` (minus 59 syntax errors) | **2,973** | 538 | 2,435 | Tests reproducibility of Phase 5/6 results using the improved architecture. |
| **Track E2** | `sql_injection_train_1000.xlsx` | **1,000** | 550 | 450 | Evaluates training signal and linear separability of the new curated Excel dataset in isolation. |
| **Track E3** | Deduplicated Merge of E1 + E2 | **3,973** | 1,088 | 2,885 | Measures transfer performance and scaling benefits of combining naturalistic repository code with curated patterns. |
| **Track $E_{\text{ext}}$** | `expanded_evaluation_3k.json` (CWE-89 subset) | **700** | 350 | 350 | **Independent Out-of-Domain Benchmark**. Never seen during training or threshold tuning; evaluated once as a zero-shot generalization test. |

---

## 4. Verification Check
- Zero group leakage between paired samples.
- Zero contamination of the 700-example external benchmark into any training split.
- 100% of samples in E1, E2, E3, and $E_{\text{ext}}$ are syntactically valid Python code parseable by `ast.parse()`.

