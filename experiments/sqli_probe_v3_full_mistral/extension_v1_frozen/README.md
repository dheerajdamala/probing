# Probe V3 — SQL Injection Detection, Localization & Explainability Extension

A modular extension for the SQL Injection Security Probe V3 that accepts raw Python source code, performs AST-aware SQL sink and dataflow analysis, extracts Layer 12 representations from `mistralai/Mistral-7B-Instruct-v0.3`, and produces a structured, explainable JSON vulnerability report.

---

## 1. System Architecture

```
                                [Raw Python Source Code]
                                           │
                                           ▼
                                 ┌──────────────────┐
                                 │  Input Handler   │ ──> (Syntax Check, Line Normalization)
                                 └──────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
          ┌──────────────────────┐                   ┌───────────────────────────┐
          │     AST Analyzer     │                   │ Representation Extractor  │
          │ (Sink & Clause Parse)│                   │ (Mistral-7B Layer 12 Sink)│
          └──────────────────────┘                   └───────────────────────────┘
                     │                                           │
                     ▼                                           ▼
          ┌──────────────────────┐                   ┌───────────────────────────┐
          │ Query Dataflow Tracer│                   │      Probe Predictor      │
          │ (Intraprocedural)    │                   │   (Lightweight MLP, 0.36) │
          └──────────────────────┘                   └───────────────────────────┘
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼
                              ┌───────────────────────────┐
                              │ Vulnerability Classifier  │ ──> (Subtype Taxonomy)
                              │     & Evidence Builder    │ ──> (Line-Anchored Proof)
                              └───────────────────────────┘
                                           │
                                           ▼
                              ┌───────────────────────────┐
                              │     Report Generator      │
                              └───────────────────────────┘
                                           │
                                           ▼
                                [Structured JSON Output]
```

### Component Breakdown
1. **`input_handler.py`**: Normalizes carriage returns, validates syntax with `ast.parse`, and catches syntax errors without crashing.
2. **`ast_analyzer.py`**: Locates database execution sinks (`.execute()`, `.raw()`, `.query()`, etc.), identifies parameter binding arguments, detects dynamic placeholders, and classifies clause targets (`ORDER BY`, table names).
3. **`query_dataflow.py`**: Traces backward from execution sinks to identify intermediate query-construction assignments and external input retrieval lines (`request.args`, function parameters).
4. **`representation_extractor.py`**: Connects to the frozen `mistralai/Mistral-7B-Instruct-v0.3` model to extract Layer 12 AST sink-region pooled representations ($d=4,096$). Disallows mock and miniature fixtures.
5. **`probe_predictor.py`**: Runs inference using the validated `MLP_sink_layer_12.pt` weights and `MLP_sink_layer_12_scaler.joblib`, evaluating against the validation threshold $\tau^* = 0.36$ with an uncertainty band ($\pm 0.05$).
6. **`vulnerability_classifier.py`**: Classifies code into the 11 supported vulnerability construction subtypes based on verifiable source code evidence.
7. **`evidence_builder.py`**: Synthesizes line-specific observations categorized by support type (`AST`, `DATAFLOW`, `HEURISTIC`).
8. **`report_generator.py`**: Serializes results into the unified JSON schema.
9. **`pipeline.py`**: Orchestrates the entire detection, localization, and explainability workflow.

---

## 2. Supported Vulnerability Construction Types

- `DIRECT_STRING_CONCATENATION`: Binary `+` concatenation incorporating untrusted variables.
- `F_STRING_INTERPOLATION`: Python f-strings formatting variables directly into SQL.
- `PERCENT_FORMATTING`: Python `%` modulo operator formatting query strings.
- `FORMAT_METHOD`: `.format()` method formatting query strings.
- `DYNAMIC_IDENTIFIER`: Generic dynamic construction of SQL identifiers.
- `DYNAMIC_TABLE_NAME`: Direct injection into `FROM`, `INTO`, or `TABLE` clauses.
- `DYNAMIC_COLUMN_NAME`: Dynamic construction of selected or projected columns.
- `DYNAMIC_ORDER_BY`: Injection into `ORDER BY` clauses.
- `HYBRID_QUERY_CONSTRUCTION`: Queries that mix valid parameter placeholders (`?`, `%s`) with inline dynamic string interpolation.
- `OTHER`: Unspecified dynamic query construction pattern.
- `UNCONFIRMED`: High probe vulnerability confidence without an identifiable AST syntax construction pattern.

---

## 3. Example Usage & Output

### Example Input (`example_vulnerable.py`)
```python
def lookup_customer(request):
    account_id = request.args.get("id")
    query = "SELECT name, email FROM customers WHERE account_id = '" + account_id + "'"
    cursor.execute(query)
    return cursor.fetchone()
```

### Python Execution
```python
from extension.pipeline import SQLInjectionAnalysisPipeline

pipeline = SQLInjectionAnalysisPipeline()
code = open("example_vulnerable.py").read()
report = pipeline.analyze(code, filename="example_vulnerable.py")
print(json.dumps(report, indent=2))
```

### Example Structured JSON Output
```json
{
  "analysis_status": "COMPLETE",
  "messages": [],
  "prediction": "VULNERABLE",
  "confidence": 0.9639,
  "probabilities": {
    "vulnerable": 0.9639,
    "safe": 0.0361
  },
  "vulnerability_type": [
    "DIRECT_STRING_CONCATENATION"
  ],
  "sink_location": [
    {
      "function": "cursor.execute",
      "file": "example_vulnerable.py",
      "line": 4,
      "column": 4
    }
  ],
  "vulnerable_lines": [
    {
      "line": 2,
      "role": "SOURCE",
      "reason": "External HTTP request parameter retrieved via request.args.get."
    },
    {
      "line": 3,
      "role": "QUERY_CONSTRUCTION",
      "reason": "SQL string assigned to 'query' constructed using dynamic formatting or concatenation."
    },
    {
      "line": 4,
      "role": "SINK",
      "reason": "SQL query reaches execution method 'cursor.execute'."
    }
  ],
  "evidence": [
    {
      "line": 2,
      "category": "INPUT_SOURCE",
      "description": "External HTTP request parameter retrieved via request.args.get.",
      "support": "DATAFLOW"
    },
    {
      "line": 3,
      "category": "DIRECT_STRING_CONCATENATION",
      "description": "SQL query constructed via direct string concatenation (+). Affects SQL clause: WHERE.",
      "support": "AST"
    },
    {
      "line": 4,
      "category": "SQL_EXECUTION",
      "description": "SQL execution sink 'cursor.execute' invoked without parameter bindings.",
      "support": "AST"
    }
  ]
}
```

---

## 4. Execution Commands

### Run Dedicated Unit Test Suite
```powershell
python -m unittest experiments/sqli_probe_v3_full_mistral/extension/tests/test_extension_suite.py
```

### Run Benchmark Functionality Evaluation
```powershell
python scratch/evaluate_extension_on_test_set.py
```

---

## 5. Known Limitations

1. **Intraprocedural Scope**: The dataflow tracer operates strictly within individual functions. When a query variable is passed into an external helper function across modules, the pipeline reports `analysis_status: "PARTIAL"` and notes the interprocedural boundary.
2. **Local Hardware Constraint**: Generating representations for new arbitrary code requires GPU acceleration (NVIDIA T4 or better) to load `mistralai/Mistral-7B-Instruct-v0.3`. In CPU-only environments, the pipeline can analyze AST patterns or process pre-extracted representations. Mock models and synthetic vectors are strictly prohibited.
3. **Symbolic Whitelist Invariants**: Dictionary lookups (e.g. `order = {"date": "created_at"}[sort]`) and conditional sanitization guards may be flagged as dynamic string formatting if complex branching masks the whitelist invariant.

