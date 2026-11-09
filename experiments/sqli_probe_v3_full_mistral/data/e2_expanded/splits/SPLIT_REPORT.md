# E2 Expanded V2: Leakage-Resistant Dataset Partition Report

## 1. Executive Summary & Partition Overview

In accordance with the strict leakage-prevention mandates, the expanded SQL injection dataset (**4,673 records**) has been partitioned into reliable, group-aware, and class-stratified **Train**, **Validation**, and **Test** splits.

- **Total Records:** 4,673 samples (1,438 vulnerable, 3,235 safe; natural prevalence: **30.77%**)
- **Target Proportions:** Approximately 70% Train, 15% Validation, 15% Test.
- **Achieved Proportions:** 70.70% Train, 14.42% Validation, 14.87% Test.
- **Historical E2 Test Isolation:** The original 151-sample E2 test set is **100% preserved inside the new Test split** (`is_historical_test: true`) with zero leakage into Train or Validation.
- **Cross-Split Leakage:** **Zero** sample ID overlap, **zero** exact code overlap, **zero** normalized whitespace/comment overlap, **zero** repository group overlap, **zero** vulnerable/fixed pair overlap, and **zero** near-duplicate ($\text{Jaccard} \ge 0.85$) overlap.

---

## 2. Partition Summary Statistics

| Partition | Total Samples | % of Dataset | Vulnerable | Safe | Vulnerability Prevalence |
|---|---|---|---|---|---|
| **Train** | **3,304** | 70.70% | 1,104 | 2,200 | 33.41% |
| **Validation** | **674** | 14.42% | 131 | 543 | 19.44% |
| **Test** | **695** | 14.87% | 203 | 492 | 29.21% |
| **Total** | **4,673** | 100.00% | **1,438** | **3,235** | **30.77%** |

### Test Split Composition
The 695-sample Test partition is composed of two explicitly labeled subsets:
1. **Historical E2 Test Benchmark (`is_historical_test: true`):** 151 samples (83 vulnerable, 68 safe). Matches historical Phase 4 evaluation bit-for-bit.
2. **Expanded External Test Benchmark (`is_historical_test: false`):** 544 samples (120 vulnerable, 424 safe) derived from completely independent external project repositories (`asleep-at-the-keyboard-nyu`, `PyCQA/bandit`, `scthornton/securecode`, `psf/requests`, `cvefixes_annotated`, `kbase/metrics`, `tornadoweb/tornado`).

---

## 3. Group-Level Leakage Prevention Methodology

Grouping was enforced using a multi-layered Disjoint Set Union (Union-Find) hierarchy:

### 3.1 Repository & Project Grouping
All samples from the same GitHub repository/project domain were strictly constrained to a single partition:
- **Train Repositories:** `apache/airflow` (692), `django/django` (617), `sqlmapproject/sqlmap` + `andresriancho/w3af` (406), `mlflow/mlflow` (158), `volatilityfoundation/volatility` (9), `viakondratiuk/cash-machine` (4), `yaml/pyyaml` (3), `paramiko/paramiko` (3), `john5223/bottle-auth` (2), `tiredtyrant/flairbot` (2), `rl-institut/NESP2` (2), `cve_CVE-2018-25088` (2), `cve_CVE-2021-42134` (1), `cve_CVE-2016-2513` (1), `cve_CVE-2022-41892` (1), `cve_` (1).
- **Validation Repositories:** `saltstack/salt` (279), `ansible/ansible` (161), `jupyter/notebook` (42), `synthetic_dpo_remediation` (22), `pallets/flask` (18), `s2e-lab/SecurityEval` (2), `cve_CVE-2011-4953` (1).
- **Test Repositories:** `asleep-at-the-keyboard-nyu` (474), `PyCQA/bandit` (28), `scthornton/securecode` (21), `psf/requests` (11), `cvefixes_annotated` (6), `kbase/metrics` (2), `tornadoweb/tornado` (2).

### 3.2 Cross-Project Copied Code & Tool Engines
- Automated similarity analysis detected that `andresriancho/w3af` includes code borrowed from `sqlmapproject/sqlmap`.
- To prevent near-duplicate leakage across repositories, `andresriancho/w3af` (52 samples) and `sqlmapproject/sqlmap` (354 samples) were merged into a single component (**406 samples**) placed entirely in **Train**.

### 3.3 Synthetic Benchmark Task & Structural Template Clustering
- All 700 samples from `expanded_evaluation_3k` (`ext_3k_task_cwe89_*`) were found to share identical parameterized prompt templates (differing only by identifier numbers like `_v000` vs `_v230`).
- To prevent template clone leakage, all 700 samples (350 vulnerable/safe pairs) were grouped together and assigned exclusively to **Train**.

### 3.4 Vulnerable / Fixed Code Pairs
- All 1,691 pairs from `validated_sql_dataset` and 350 pairs from `expanded_evaluation_3k` are guaranteed to remain within the same partition. Zero pairs are split across boundaries.

---

## 4. Historical Test Isolation & Benchmark Preservation

The original E2 dataset (1,000 samples) maintains its historical partition assignments:
- **Historical Train (700 samples):** 100% retained in the new Train split.
- **Historical Validation (149 samples):** 100% retained in the new Validation split.
- **Historical Test (151 samples):** 100% retained in the new Test split.

### Verification of Historical Test Immunity:
- Exact code hash overlap with historical test: **0**
- Normalized code hash overlap with historical test: **0**
- Token-level Jaccard similarity ($\ge 0.85$) with historical test: **0**
- Leaked into Train: **0**
- Leaked into Validation: **0**

---

## 5. Comprehensive Leakage Audit Results

The audit script executed automated pairwise cross-split checks across all $4,673$ records:

| Check Category | Validation Rule | Checked Pairs | Violations Detected | Status |
|---|---|---|---|---|
| **Sample ID Overlap** | $\text{Split}_A \cap \text{Split}_B = \emptyset$ | All pairs | 0 | **PASSED** |
| **Exact Code Hash** | $\text{SHA256}(\text{code})_A \cap \text{SHA256}(\text{code})_B = \emptyset$ | All pairs | 0 | **PASSED** |
| **Normalized Hash** | $\text{SHA256}(\text{norm})_A \cap \text{SHA256}(\text{norm})_B = \emptyset$ | All pairs | 0 | **PASSED** |
| **Repository Groups** | $\text{Repo}_A \cap \text{Repo}_B = \emptyset$ | All pairs | 0 | **PASSED** |
| **Vulnerable/Fixed Pairs** | $\text{Pair}_A \cap \text{Pair}_B = \emptyset$ | All pairs | 0 | **PASSED** |
| **Template Families** | $\text{Template}_A \cap \text{Template}_B = \emptyset$ | All pairs | 0 | **PASSED** |
| **Near-Duplicate Similarity** | $\text{Jaccard}(\text{tokens}) \ge 0.85$ | $3,304 \times 674$, $3,304 \times 695$, $674 \times 695$ | 0 | **PASSED** |
| **Historical Test Integrity** | 151 historical test samples isolated | Test split only | 0 | **PASSED** |

---

## 6. Deliverable Directory & File Manifest

Location: `experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/`

1. [`train.jsonl`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/train.jsonl): 3,304 training records in exact E2 schema.
2. [`validation.jsonl`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/validation.jsonl): 674 validation records in exact E2 schema.
3. [`val.jsonl`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/val.jsonl): Convenience alias identical to `validation.jsonl`.
4. [`test.jsonl`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/test.jsonl): 695 test records in exact E2 schema.
5. [`split_manifest.json`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/split_manifest.json): Full index mapping every `sample_id` to its split, group ID, and historical test flag.
6. [`grouping_report.json`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/grouping_report.json): Breakdown of every group, cluster size, and split assignment.
7. [`leakage_audit.json`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/leakage_audit.json): Cryptographic and statistical verification evidence confirming zero cross-split leakage.
8. [`SPLIT_REPORT.md`](file:///c:/Users/dheer/Desktop/isae/experiments/sqli_probe_v3_full_mistral/data/e2_expanded/splits/SPLIT_REPORT.md): This report.

---

## 7. Artifact Integrity Sign-Off

- [x] All 4,673 records accounted for across the three partitions ($3,304 + 674 + 695 = 4,673$).
- [x] All original 1,000 E2 records accounted for and preserved in their historical splits ($700 + 149 + 151 = 1,000$).
- [x] Original E2 schema strictly preserved (18 JSON keys, exact data types and null handling).
- [x] Existing loader (`SQLInjectionRecord.from_dict` and `load_jsonl`) loads all three split files without modifications.
- [x] Original historical files in `experiments/sqli_probe_v3_full_mistral/data/e2_excel/` untouched.
- [x] Pretrained checkpoints, scalers, embeddings, and historical reports untouched.
- [x] Zero model training and zero embedding extraction performed during this task.

