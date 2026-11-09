# Representative Localization Success Cases

This document details representative examples where the AST and interprocedural static analyzer successfully identified exact vulnerable lines, input sources, and execution sinks.

---

### Case 1: Direct SQL Injection Exact Line Localization
- **Sample ID:** `loc_direct_001`
- **Category:** Direct SQL injection
- **Ground Truth:** Source Line 1 (`user_input`), Construction Line 3, Sink Line 5
- **Analyzer Output:**
  - Sinks Identified: Line 5 (`cursor.execute`)
  - Construction Lines: Line 3 (`F_STRING_INTERPOLATION`)
  - Top-1 Vulnerable Line: Line 3
  - Top-3 Vulnerable Lines: [3, 5]
  - Exact Line Set Match: **TRUE** (100% overlap)
- **Code:**
```python
1: def search_users_1(conn, user_input):
2:     # Line 2: Untrusted input directly interpolated
3:     sql = f"SELECT * FROM users WHERE username = '{user_input}'"
4:     cursor = conn.cursor()
5:     return cursor.execute(sql).fetchall()
```

---

### Case 2: Interprocedural Function Boundary Taint Tracing
- **Sample ID:** `loc_interproc_001`
- **Category:** Helper-function/interprocedural injection
- **Ground Truth:** Source Line 4 (`user_tag`), Propagation Line 5, Construction Line 2, Sink Line 6
- **Analyzer Output:**
  - Path Step 1: Line 4 (Source `user_tag`)
  - Path Step 2: Line 5 (Propagation into `build_query_1`)
  - Path Step 3: Line 2 (Query construction via f-string)
  - Path Step 4: Line 6 (Sink execution via `conn.execute`)
  - Exact Path Match: **TRUE**
- **Code:**
```python
1: def build_query_1(filter_val):
2:     return f"SELECT * FROM records WHERE status = '{filter_val}'"
3: 
4: def run_search_1(conn, user_tag):
5:     q = build_query_1(user_tag)
6:     return conn.execute(q).fetchall()
```

---

### Case 3: Safe Parameterized SQL Correctly Cleared
- **Sample ID:** `loc_safe_param_001`
- **Category:** Safe parameterized SQL
- **Ground Truth:** Safe (Separate tuple parameter binding at Line 4)
- **Analyzer Output:**
  - Prediction: `SAFE`
  - Evidence: `Query passes parameters via separate argument.`
  - Vulnerable Lines Flagged: `[]` (None)
  - Correctness: **TRUE** (No false alarms generated)
- **Code:**
```python
1: def get_user_profile_1(conn, user_id):
2:     sql = "SELECT id, username, email FROM users WHERE id = ?"
3:     cursor = conn.cursor()
4:     cursor.execute(sql, (user_id,))
5:     return cursor.fetchone()
```

---

### Case 4: High-Level Safe ORM Operation
- **Sample ID:** `loc_safe_orm_001`
- **Category:** Safe ORM and query construction
- **Ground Truth:** Safe (Django `.filter()` query)
- **Analyzer Output:**
  - Prediction: `SAFE`
  - Sinks Detected: `[]` (None)
  - Correctness: **TRUE**
- **Code:**
```python
1: def list_user_articles_1(request, author_id):
2:     articles = Article.objects.filter(author_id=author_id, is_published=True)
3:     return list(articles.values('id', 'title'))
```

---

### Case 5: Validated / Allowlisted Dynamic Identifier
- **Sample ID:** `loc_safe_val_001`
- **Category:** Validated/allowlisted input
- **Ground Truth:** Safe (Column mapped from static dictionary `ALLOWED`)
- **Analyzer Output:**
  - Prediction: `SAFE`
  - Sinks Detected: Line 4
  - Evidence: Verified constant dictionary lookup on Line 3
  - Correctness: **TRUE**
- **Code:**
```python
1: def get_sorted_logs_1(cursor, sort_field):
2:     ALLOWED = {'time': 'timestamp', 'level': 'log_level', 'id': 'log_id'}
3:     col = ALLOWED.get(sort_field, 'timestamp')
4:     cursor.execute(f"SELECT * FROM logs ORDER BY {col} ASC")
5:     return cursor.fetchall()
```
