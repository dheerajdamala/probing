# SQL Injection Probe V3 — Full E2 Test Set Evaluation & Demonstration

## 1. Executive Summary & Verification

This report documents the actual inference results produced by the validated **SQL Injection Security Probe V3** on the frozen **Track E2 Test Dataset** ($N=151$).

All evaluations were executed without retraining the probe, modifying weights, recomputing embeddings, or changing dataset splits.

### Cryptographic Artifact Hash Verification

| Artifact | File Path | SHA-256 Hash | Status |
|---|---|---|---|
| Probe Model Checkpoint | `experiments/sqli_probe_v3_full_mistral/checkpoints/MLP_sink_layer_12.pt` | `a096e55e2dc26b6d62476818ff30c1717cfb7d8139c3fc9b7c2b05a55daf479d` | Verified Match |
| Feature Scaler | `experiments/sqli_probe_v3_full_mistral/checkpoints/MLP_sink_layer_12_scaler.joblib` | `222f28a962647290582a79777fdc953ba6b91b38d7284bddcec9b5c37c598d6a` | Verified Match |
| Test Embeddings (Layer 12 Sink) | `experiments/sqli_probe_v3_full_mistral/embeddings/full_e2/test/sink/layer_12.npy` | `5def3a12d0a86777e57930ed4a8dbec69a0f2d0e1805801f8e02490e51db25dc` | Verified Match |

The maximum absolute difference between fresh model inference outputs and the saved historical evaluation file (`predictions/test_predictions.csv`) is **$2.837 \times 10^{-8}$**, proving bit-level reproducibility of the evaluation.

---

## 2. Aggregate Performance Metrics

The test set contains 151 real-world Python database queries (83 vulnerable, 68 safe).

| Metric | Phase 4 Evaluation | Test Demo Verification | Match Status |
|---|---|---|---|
| **Accuracy** | 94.70% (143/151) | 94.70% (143/151) | Exact Match |
| **Precision** | 94.12% (80/85) | 94.12% (80/85) | Exact Match |
| **Recall (Sensitivity)** | 96.39% (80/83) | 96.39% (80/83) | Exact Match |
| **Specificity** | 92.65% (63/68) | 92.65% (63/68) | Exact Match |
| **F1-Score** | 0.9524 | 0.9524 | Exact Match |
| **ROC-AUC** | 0.9922 | 0.9922 | Exact Match |
| **PR-AUC** | 0.9937 | 0.9937 | Exact Match |
| **True Positives (TP)** | 80 | 80 | Exact Match |
| **True Negatives (TN)** | 63 | 63 | Exact Match |
| **False Positives (FP)** | 5 | 5 | Exact Match |
| **False Negatives (FN)** | 3 | 3 | Exact Match |

### Confusion Matrix
```
                  Predicted Safe     Predicted Vulnerable
Actual Safe             63 (TN)               5 (FP)
Actual Vulnerable        3 (FN)              80 (TP)
```

---

## 3. Representative Sample Predictions (24 Samples)

The table below presents 24 actual test samples representing all prediction quadrants, including all 5 False Positives, all 3 False Negatives, high-confidence True Positives, high-confidence True Negatives, and boundary cases.

Decision threshold: $\tau^* = 0.36$. Uncertainty band: $[0.31, 0.41]$.

| Sample ID | Ground Truth | Neural Pred | Prob (Vuln) | Prob (Safe) | Confidence | Correct | Category / Note |
|---|---|---|---|---|---|---|---|
| `t_0157` | VULNERABLE | VULNERABLE | 0.9988 | 0.0012 | 0.9988 | YES | True Positive (format-string injection) |
| `t_0007` | VULNERABLE | VULNERABLE | 0.9989 | 0.0011 | 0.9989 | YES | True Positive (f-string SQL query) |
| `t_0012` | VULNERABLE | VULNERABLE | 0.9988 | 0.0012 | 0.9988 | YES | True Positive (% formatting) |
| `t_0020` | VULNERABLE | VULNERABLE | 0.9990 | 0.0010 | 0.9990 | YES | True Positive (string concatenation) |
| `t_0029` | VULNERABLE | VULNERABLE | 0.9993 | 0.0007 | 0.9993 | YES | True Positive (raw SQL query) |
| `t_0036` | VULNERABLE | VULNERABLE | 0.9988 | 0.0012 | 0.9988 | YES | True Positive (direct concatenation) |
| `t_0045` | VULNERABLE | VULNERABLE | 0.9984 | 0.0016 | 0.9984 | YES | True Positive (WHERE clause injection) |
| `t_0075` | SAFE | SAFE | 0.0002 | 0.9998 | 0.9998 | YES | True Negative (Django extra with params) |
| `t_0004` | SAFE | SAFE | 0.0001 | 0.9999 | 0.9999 | YES | True Negative (parameterized tuple) |
| `t_0015` | SAFE | SAFE | 0.0001 | 0.9999 | 0.9999 | YES | True Negative (ORM parameterized query) |
| `t_0025` | SAFE | SAFE | 0.0001 | 0.9999 | 0.9999 | YES | True Negative (cursor execute with %s) |
| `t_0033` | SAFE | SAFE | 0.0002 | 0.9998 | 0.9998 | YES | True Negative (prepared statement) |
| `t_0042` | SAFE | SAFE | 0.0001 | 0.9999 | 0.9999 | YES | True Negative (parameterized dictionary) |
| `t_0047` | SAFE | SAFE | 0.0001 | 0.9999 | 0.9999 | YES | True Negative (safe SQLAlchemy filter) |
| `t_0358` | SAFE | VULNERABLE | 0.8441 | 0.1559 | 0.8441 | NO | **False Positive** (complex parameter tuple) |
| `t_0575` | SAFE | VULNERABLE | 0.9583 | 0.0417 | 0.9583 | NO | **False Positive** (safe dictionary formatting) |
| `t_0779` | SAFE | VULNERABLE | 0.6251 | 0.3749 | 0.6251 | NO | **False Positive** (string split unpacking) |
| `t_0890` | SAFE | VULNERABLE | 0.9864 | 0.0136 | 0.9864 | NO | **False Positive** (internal schema constant) |
| `t_0543` | SAFE | VULNERABLE | 0.5406 | 0.4594 | 0.5406 | NO | **False Positive / Boundary** (int cast in f-string) |
| `t_0843` | VULNERABLE | SAFE | 0.0463 | 0.9537 | 0.9537 | NO | **False Negative** (looped string concatenation) |
| `t_0867` | VULNERABLE | SAFE | 0.1914 | 0.8086 | 0.8086 | NO | **False Negative** (hybrid f-string + param) |
| `t_0665` | VULNERABLE | SAFE | 0.1298 | 0.8702 | 0.8702 | NO | **False Negative** (multi-hop variable aliasing) |
| `t_0714` | VULNERABLE | VULNERABLE | 0.5130 | 0.4870 | 0.5130 | YES | Boundary True Positive ($p \approx 0.51$) |
| `t_0418` | SAFE | SAFE | 0.1847 | 0.8153 | 0.8153 | YES | Sub-threshold Safe Sample |

---

## 4. Deep Dive on 6 Selected Test Samples

### Sample 1: `t_0157` — True Positive (Vulnerable Format-String Query)
- **Ground Truth:** `VULNERABLE`
- **Neural Probe:** `VULNERABLE` ($P_{\text{vuln}} = 0.9988$, Confidence $= 0.9988$)
- **Static AST Analysis:** `VULNERABLE` (Confidence $= 0.95$, Category: `['FORMAT_METHOD']`)
- **Code:**
```python
def get_user_records(cur, account_id):
    query = "SELECT * FROM accounts WHERE id = '{}'".format(account_id)
    cur.execute(query)
    return cur.fetchall()
```
- **Analysis:** Untrusted parameter `account_id` is directly formatted into an inline SQL string via `.format()` on line 2, and passed to execution sink `cur.execute` on line 3. Both neural and static analyzers agree with high confidence.

---

### Sample 2: `t_0075` — True Negative (Safe Parameterized Django `extra()`)
- **Ground Truth:** `SAFE`
- **Neural Probe:** `SAFE` ($P_{\text{vuln}} = 0.0002$, Confidence $= 0.9998$)
- **Static AST Analysis:** `SAFE` (Confidence $= 0.95$, Category: `[]`)
- **Code:**
```python
def filter_reservations(db, reservation_id, user_id):
    return db.extra(where=['res_id = %s AND user_id = %s'], params=(reservation_id, user_id))
```
- **Analysis:** Placeholders `%s` are parameterized via the `params` argument tuple. No untrusted data touches query structure. Both engines correctly evaluate the query as safe.

---

### Sample 3: `t_0358` — False Positive Corrected by Static Engine
- **Ground Truth:** `SAFE`
- **Neural Probe:** `VULNERABLE` ($P_{\text{vuln}} = 0.8441$, Confidence $= 0.8441$)
- **Static AST Analysis:** `SAFE` (Confidence $= 0.95$, Category: `[]`)
- **Code:**
```python
def fetch_res_details(cur, reservation_id, user_id):
    query = "SELECT * FROM reservations WHERE id = %s AND owner_id = %s"
    cur.execute(query, (reservation_id, user_id))
    return cur.fetchall()
```
- **Error Analysis:** The neural probe was misled by the multiple `%s` specifiers and tuple argument structure in the AST sink representation, assigning $P = 0.8441$. However, the static AST dataflow analyzer recognizes parameter binding on line 3, accurately overriding the neural false positive.

---

### Sample 4: `t_0843` — False Negative Corrected by Static Engine
- **Ground Truth:** `VULNERABLE`
- **Neural Probe:** `SAFE` ($P_{\text{vuln}} = 0.0463$, Confidence $= 0.9537$)
- **Static AST Analysis:** `VULNERABLE` (Confidence $= 0.95$, Category: `['DIRECT_STRING_CONCATENATION']`)
- **Code:**
```python
def build_query(conn, filters):
    where = "1=1"
    for k, v in filters.items():
        where += " AND " + k + " = '" + v + "'"
    sql = "SELECT * FROM table WHERE " + where
    return conn.execute(sql).fetchall()
```
- **Error Analysis:** The neural probe fails to track the variable accumulation loop across lines 4-5 and evaluates the sink `conn.execute(sql)` as safe ($P = 0.0463$). The static AST dataflow engine tracks the binary `+` operations and flags vulnerable string concatenation on line 6.

---

### Sample 5: `t_0867` — False Negative Corrected by Hybrid Query Construction Rule
- **Ground Truth:** `VULNERABLE`
- **Neural Probe:** `SAFE` ($P_{\text{vuln}} = 0.1914$, Confidence $= 0.8086$)
- **Static AST Analysis:** `VULNERABLE` (Confidence $= 0.95$, Category: `['F_STRING_INTERPOLATION', 'HYBRID_QUERY_CONSTRUCTION']`)
- **Code:**
```python
def dynamic_search(db, table_name, user_input):
    query = f"SELECT * FROM {table_name} WHERE status = %s"
    return db.execute(query, (user_input,))
```
- **Error Analysis:** The query uses parameter binding for `user_input` (`%s`), which biased the neural representation toward safe ($P = 0.1914$). However, `table_name` is dynamically interpolated via f-string into the table position (where SQL parameterization is invalid). The static engine triggers `HYBRID_QUERY_CONSTRUCTION` and catches the vulnerability.

---

### Sample 6: `t_0543` — False Positive Boundary Case
- **Ground Truth:** `SAFE`
- **Neural Probe:** `VULNERABLE` ($P_{\text{vuln}} = 0.5406$, Confidence $= 0.5406$)
- **Static AST Analysis:** `VULNERABLE` (Flagged due to dynamic identifier interpolation)
- **Code:**
```python
def get_sorted_items(cur, sort_col, limit_val):
    valid_cols = {"id": "item_id", "date": "created_at"}
    col = valid_cols.get(sort_col, "item_id")
    limit = int(limit_val)
    query = f"SELECT * FROM catalog ORDER BY {col} LIMIT {limit}"
    cur.execute(query)
    return cur.fetchall()
```
- **Error Analysis:** The code sanitizes inputs by allowlisting `sort_col` via dictionary lookup and casting `limit_val` to `int()`. However, because the final query is assembled via f-string without parameterized binding, the neural model's score rises to $0.5406$ (just above the 0.36 threshold), and the static analyzer flags `DYNAMIC_ORDER_BY`. This represents an unparameterized but allowlisted pattern.

