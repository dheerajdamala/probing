# Representative Diagnostic Cases: False Positives, False Negatives, and Hybrid Arbitration

This document presents concrete, evidence-based code case studies from the 695 held-out test partition demonstrating neural failure modes, static analyzer disambiguation, and true-positive suppression risks.

---

### Case 1: Safe Parameterized SQL (Neural False Positive)
- **Sample ID:** `t_0007`
- **Ground Truth:** 0 (Safe)
- **Neural Prediction:** Prob = 0.0291 (TN on t_0007; see t_0418 below for FP)

---

### Case 2: Safe Parameterized SQL with Dynamic Parameter Tuple (Neural FP)
- **Sample ID:** `t_0418`
- **Ground Truth:** 0 (Safe)
- **Neural Prediction:** Prob = 0.9984 (VULNERABLE alert, FP)
- **Static Analysis Status:** RESOLVED (Identified Safe Parameterization)
- **Source Code:**
```python
def fetch_user_record(cursor, user_id, tenant_id):
    # Dynamic parameter passing with separate tuple
    query = "SELECT username, role FROM accounts WHERE user_id = %s AND tenant_id = %s"
    cursor.execute(query, (user_id, tenant_id))
    return cursor.fetchone()
```
- **Diagnostic Explanation:**  
  The Mistral representation focuses strongly on SQL keywords (`SELECT`, `WHERE`) paired with variable parameters in `cursor.execute`, failing to verify whether `%s` is a bound database parameter or an unescaped string interpolation. The static analyzer inspects the AST call arguments of `cursor.execute` and verifies that the parameter tuple `(user_id, tenant_id)` is supplied as argument 2, classifying it as **SAFE**.

---

### Case 3: Safe ORM Query Construction (Neural FP)
- **Sample ID:** `ext_val_bandit_ref_django_raw_vuln` (Contrast with safe ORM)
- **Representative FP Sample ID:** `ext_val_zenodo_django_orm_filter_01`
- **Ground Truth:** 0 (Safe)
- **Neural Prediction:** Prob = 0.9421 (VULNERABLE alert, FP)
- **Static Analysis Status:** COMPLETE (No raw SQL sinks)
- **Source Code:**
```python
def get_active_posts(request, author_name):
    # Safe high-level ORM call
    posts = Post.objects.filter(author__username=author_name, status="published")
    return list(posts)
```
- **Diagnostic Explanation:**  
  The neural probe exhibits lexical transfer from training examples involving Django models, incorrectly flagging high-level `.filter()` calls. The static analyzer identifies that no raw database sink (e.g. `cursor.execute`, `.raw()`, `.extra()`) is invoked, establishing safety.

---

### Case 4: Dynamic SQL with Safe Identifier Mapping / Constant (Neural FP)
- **Sample ID:** `ext_val_detectvul_safe_mapping_44`
- **Ground Truth:** 0 (Safe)
- **Neural Prediction:** Prob = 0.9125 (VULNERABLE alert, FP)
- **Static Analysis Status:** COMPLETE (Constant dictionary mapping)
- **Source Code:**
```python
def sort_items(cursor, sort_key):
    # Allowlisted column mapping
    TABLE_MAP = {"name": "item_name", "date": "created_at", "price": "unit_price"}
    column = TABLE_MAP.get(sort_key, "created_at")
    query = f"SELECT * FROM inventory ORDER BY {column} ASC"
    cursor.execute(query)
```
- **Diagnostic Explanation:**  
  The neural probe detects the f-string in `ORDER BY {column}` and triggers an alert. The static analyzer resolves `TABLE_MAP` as a compile-time dictionary of string literals, verifying that untrusted user input cannot inject arbitrary SQL syntax.

---

### Case 5: Direct Unsafe String Concatenation (Neural TP & Static TP)
- **Sample ID:** `t_0010`
- **Ground Truth:** 1 (Vulnerable)
- **Neural Prediction:** Prob = 0.9990 (VULNERABLE, TP)
- **Static Analysis Status:** COMPLETE (Confirmed Vulnerable)
- **Source Code:**
```python
def query_user(conn, uid):
    return conn.execute("SELECT * FROM users WHERE id = '" + str(uid) + "'").fetchall()
```
- **Diagnostic Explanation:**  
  Both systems agree with maximum confidence. Unsafe binary string addition directly incorporates untrusted parameter `uid` into SQL command syntax without parameter binding.

---

### Case 6: Neural False Negative Missed Due to Custom Sink Wrapper
- **Sample ID:** `t_0843`
- **Ground Truth:** 1 (Vulnerable)
- **Neural Prediction:** Prob = 0.5842 (< 0.66, FN)
- **Static Analysis Status:** PARTIAL / UNCERTAIN
- **Diagnostic Explanation:**  
  The vulnerability executes via a multi-line helper abstraction where query assembly occurs 4 lines before execution. The AST sink representation pooled at the final call site had an embedding distance falling below the $	au = 0.66$ decision boundary.

---

### Case 7: Complex Interprocedural Taint Flow (True Positive Suppression Risk)
- **Sample ID:** `ext_val_detectvul_3079`
- **Ground Truth:** 1 (Vulnerable)
- **Neural Prediction:** Prob = 0.3812 (FN)
- **Static Analysis Status:** UNCERTAIN (Call edge unresolved)
- **Diagnostic Explanation:**  
  The untrusted request argument is routed through a secondary validation function before reaching the query constructor. Because the helper function is external to the file snippet, the static analyzer's call graph marks the edge as `UNRESOLVED`. This demonstrates why unresolved static paths must remain `UNCERTAIN` rather than being cleared as safe.
