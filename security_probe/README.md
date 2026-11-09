# Security Probe: Isolated Representation-Based Security Analyzer

An isolated, modular security probe prototype designed to analyze a **single code snippet** at inference time. It predicts code-level vulnerability, candidate CWE categories, and exploratory token/line-level vulnerability projections from the frozen representations of a pretrained Code LLM.

---

## 1. Key Design Principles

1. **Strict Single-Sample Inference**:
   The analyzer accepts exactly **one** code snippet (`result = analyzer.analyze(code)`). It does **not** require a paired safe version, a paired vulnerable version, or a contrastive comparison at inference time.
2. **Frozen Base Model**:
   The underlying Transformer is treated strictly as a read-only feature extractor (`torch.no_grad()`). Model weights are never modified, fine-tuned, or updated.
3. **No Activation Steering**:
   This prototype is a pure diagnostic probe. It does not invoke, attach, or modify the project's activation steering hooks or residual stream perturbations during generation.
4. **Honest Localization Boundary**:
   The available datasets provide **whole-program semantic labels only**. No ground-truth token or span annotations exist in the dataset. Token- and line-level scores are derived via **exploratory representation projection** (projecting per-token hidden states onto the probe's decision hyperplane). We explicitly document that these scores are exploratory and do not constitute learned ground-truth span localization.
5. **Configurable Architecture & Thresholds**:
   Operates across configurable Transformer layers (e.g. Layer 16 for Mistral-7B, Layer 2 for miniature test fixtures). Decision thresholds and uncertainty margins are calibrated strictly on validation data.

---

## 2. Directory Structure

```text
security_probe/
├── README.md                           # System overview and usage guide
├── config.yaml                         # Configurable probe and model hyperparameters
├── requirements.txt                    # Python dependencies
├── example_inference.py                # Minimal runnable single-sample inference demo
├── conftest.py                         # Pytest configuration and path resolution
├── src/
│   ├── feature_extractor.py            # Forward-pass hidden state extraction & token offset mapping
│   ├── probe_model.py                  # Linear & MLP probe architectures with serialization
│   ├── localization.py                 # Character-to-line mapping & exploratory finding extraction
│   ├── inference.py                    # High-level SecurityProbeAnalyzer API
│   ├── train.py                        # Dataset loading, leakage auditing, training, and threshold tuning
│   └── evaluation.py                   # Detection, per-CWE, and synthetic localization metrics
├── tests/
│   ├── test_inference.py               # Unit tests for single-sample inference, safety, and model freezing
│   ├── test_localization.py            # Unit tests for line/col offsets and overlap metrics
│   └── test_data_leakage.py            # Unit tests for split leakage guards and label integrity
├── data/
│   └── README.md                       # Data policy and lack of span annotations notice
├── models/
│   └── README.md                       # Model artifact serialization guidelines
└── outputs/
    ├── README.md                       # Output artifacts index
    ├── PROBE_DESIGN_REPORT.md          # Comprehensive design and architectural report
    └── PROBE_TEST_RESULTS.txt          # Pytest execution verification logs
```

---

## 3. Quickstart & Usage

### 3.1 Single-Sample Inference Example

Run the self-contained demonstration script:
```bash
python security_probe/example_inference.py
```

### 3.2 Programmatic API Usage

```python
from src.model.load_model import load_model_and_tokenizer
from security_probe.src.inference import SecurityProbeAnalyzer
from security_probe.src.probe_model import SecurityProbe

# 1. Load frozen base model
model, tokenizer, _ = load_model_and_tokenizer(model_id="mistralai/Mistral-7B-Instruct-v0.3")

# 2. Load trained probe artifact
probe = SecurityProbe.load("security_probe/models/probe_model.pkl")

# 3. Initialize analyzer
analyzer = SecurityProbeAnalyzer(
    probe=probe,
    model=model,
    tokenizer=tokenizer,
    layer_idx=16,
    finding_threshold=0.60,
)

# 4. Analyze a single code sample
code_snippet = """
def lookup_user(cursor, username: str):
    query = f"SELECT * FROM users WHERE name = {username}"
    cursor.execute(query)
    return cursor.fetchone()
"""

result = analyzer.analyze(code_snippet)

print("Status:", result["status"])             # 'VULNERABLE', 'BENIGN', 'UNCERTAIN', or 'NO_FINDING'
print("Probability:", result["probability"])   # Calibrated float in [0.0, 1.0]
print("Findings:", result["findings"])         # List of suspected lines/spans
```

---

## 4. Output Schema

The `analyze()` method returns a structured dictionary:

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | `str` | Decision status: `VULNERABLE`, `BENIGN`, `UNCERTAIN`, `NO_FINDING`, or `ERROR` |
| `is_vulnerable` | `bool \| None` | `True` if vulnerable, `False` if benign, `None` if uncertain/error |
| `probability` | `float \| None` | Calibrated vulnerability probability $P(\text{vulnerable} \mid x)$ |
| `confidence` | `float` | Distance from decision boundary normalized to $[0.0, 1.0]$ |
| `predicted_cwe` | `str \| None` | Predicted CWE identifier (e.g. `CWE-89`, `CWE-119`) |
| `cwe_probabilities` | `dict \| None` | Full probability distribution across known CWE classes |
| `findings` | `list` | Suspected vulnerable lines exceeding `finding_threshold` with line/col offsets |
| `line_scores` | `dict` | Mapping of line numbers (1-indexed) to exploratory risk scores |
| `token_scores` | `list` | Token-by-token exploratory projections with start/end character offsets |
| `no_finding` | `bool` | `True` when no line exceeds the candidate finding threshold |
| `metadata` | `dict` | Layer index, architecture, thresholds, and localization caveats |

---

## 5. Running Tests

Run the complete isolated test suite:
```bash
python -m pytest security_probe/tests -v
```

All 12 unit tests verify single-sample isolation, boundary safety, model immutability, lack of steering hooks, line/col accuracy, and data leakage guards.
