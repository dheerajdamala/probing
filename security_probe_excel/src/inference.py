"""
Inference Pipeline for Python SQL Injection Security Probe.
Accepts raw Python source code and returns structured security findings.
Does not require dataset metadata or auxiliary training columns at inference time.
"""

import os
import sys
import re
import pickle
from typing import Dict, Any, List, Optional, Tuple

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
repo_root = os.path.dirname(parent_dir)

import importlib.util
load_model_path = os.path.join(repo_root, "src", "model", "load_model.py")
spec = importlib.util.spec_from_file_location("root_load_model", load_model_path)
root_load_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_load_model)
create_miniature_mistral = root_load_model.create_miniature_mistral

import numpy as np
import torch

SQL_SINK_REGEX = re.compile(
    r'(?:execute|executemany|raw|cursor\s*\.\s*\w+|SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|f[\"\'].*?\{.*?\}[\"\']|%s|\.format\()',
    re.IGNORECASE
)

class SQLInjectionInferenceProbe:
    """Production inference wrapper for SQL Injection detection."""

    def __init__(self, model_checkpoint_path: Optional[str] = None):
        base_dir = r"C:\Users\dheer\Desktop\isae\security_probe_excel"
        if model_checkpoint_path is None:
            model_checkpoint_path = os.path.join(base_dir, "models", "probe_head_excel_sink.pkl")
            
        self.checkpoint_path = model_checkpoint_path
        self._load_classifier()
        
        # Load frozen base model & tokenizer for representation extraction
        self.model, self.tokenizer = create_miniature_mistral(num_layers=4, hidden_size=128)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_classifier(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")
        with open(self.checkpoint_path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.threshold = float(data.get("threshold", 0.5))

    def extract_features(self, code: str) -> np.ndarray:
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Layer 3 hidden state (index 4)
            h = outputs.hidden_states[min(4, len(outputs.hidden_states) - 1)].squeeze(0)
            
        input_ids = inputs["input_ids"].squeeze(0).tolist()
        sink_indices = []
        for i, tid in enumerate(input_ids):
            token_str = self.tokenizer.decode([tid])
            if SQL_SINK_REGEX.search(token_str):
                sink_indices.append(i)
                
        if sink_indices:
            rep = h[sink_indices].mean(dim=0).cpu().numpy()
        else:
            rep = h.mean(dim=0).cpu().numpy()
            
        return rep.reshape(1, -1)

    def detect_suspicious_lines_and_sinks(self, code: str) -> Tuple[List[int], List[str]]:
        suspicious_lines = []
        detected_sinks = []
        lines = code.splitlines()
        for idx, line in enumerate(lines, 1):
            if SQL_SINK_REGEX.search(line):
                suspicious_lines.append(idx)
                match = re.search(r'(\b\w+\.(?:execute|executemany|raw|query)\b)', line)
                if match and match.group(1) not in detected_sinks:
                    detected_sinks.append(match.group(1))
        return suspicious_lines, detected_sinks

    def predict(self, source_code: str) -> Dict[str, Any]:
        """Predict SQL injection vulnerability from raw source code."""
        if not source_code or not source_code.strip():
            return {
                "prediction": "safe",
                "is_vulnerable": False,
                "confidence": 1.0,
                "vulnerability_probability": 0.0,
                "cwe_classification": None,
                "suspicious_lines": [],
                "detected_sinks": [],
                "explanation": "Empty or whitespace-only code sample."
            }

        # Extract features & predict probability
        feat = self.extract_features(source_code)
        feat_scaled = self.scaler.transform(feat)
        prob = float(self.classifier.predict_proba(feat_scaled)[0, 1])
        is_vuln = bool(prob >= self.threshold)
        
        suspicious_lines, detected_sinks = self.detect_suspicious_lines_and_sinks(source_code)
        
        if is_vuln:
            cwe = "CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')"
            explanation = (
                f"Vulnerability detected with probability {prob:.4f} (threshold: {self.threshold:.2f}). "
                f"The probe identified database query execution constructs with potentially unescaped interpolation or concatenation."
            )
        else:
            cwe = None
            explanation = (
                f"Code predicted safe with vulnerability probability {prob:.4f} (threshold: {self.threshold:.2f}). "
                f"Database calls appear properly parameterized, safe, or absent."
            )

        return {
            "prediction": "vulnerable" if is_vuln else "safe",
            "is_vulnerable": is_vuln,
            "confidence": float(abs(prob - 0.5) * 2),
            "vulnerability_probability": prob,
            "decision_threshold": self.threshold,
            "cwe_classification": cwe,
            "suspicious_lines": suspicious_lines,
            "detected_sinks": detected_sinks,
            "explanation": explanation
        }

if __name__ == "__main__":
    import numpy as np
    sample_code_vuln = '''def get_user(conn, username):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")
    return cursor.fetchone()
'''
    sample_code_safe = '''def get_user(conn, username):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = %s", (username,))
    return cursor.fetchone()
'''
    print("Testing inference wrapper...")
    probe = SQLInjectionInferenceProbe()
    res_v = probe.predict(sample_code_vuln)
    print("\nVulnerable Sample Result:")
    print(res_v)
    res_s = probe.predict(sample_code_safe)
    print("\nSafe Sample Result:")
    print(res_s)

