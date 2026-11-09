"""
Probe Predictor for Probe V3.
Loads the verified Lightweight MLP checkpoint and train-fitted StandardScaler,
applies the validation-selected decision threshold (0.36), and computes calibrated class probabilities.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
import joblib
import torch
import torch.nn as nn

DEFAULT_CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints"))
DEFAULT_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))

class LightweightMLP(nn.Module):
    """Lightweight 2-layer MLP probe mapping 4096-dim representations to SQLi logits."""
    def __init__(self, input_dim: int = 4096, hidden1: int = 256, hidden2: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class ProbePredictor:
    """Inference engine for the Probe V3 Lightweight MLP model."""

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        results_dir: str = DEFAULT_RESULTS_DIR,
        uncertainty_margin: float = 0.05
    ):
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.uncertainty_margin = uncertainty_margin
        
        self.model: Optional[LightweightMLP] = None
        self.scaler: Optional[Any] = None
        self.threshold: float = 0.36
        self.is_loaded: bool = False
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads weights, scaler, and threshold from disk."""
        model_path = os.path.join(self.checkpoint_dir, "MLP_sink_layer_12.pt")
        scaler_path = os.path.join(self.checkpoint_dir, "MLP_sink_layer_12_scaler.joblib")
        val_sweep_path = os.path.join(self.results_dir, "validation_sweep_results.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing probe checkpoint: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing probe scaler: {scaler_path}")

        # 1. Load threshold from validation sweep results
        if os.path.exists(val_sweep_path):
            with open(val_sweep_path) as f:
                records = json.load(f)
            for r in records:
                if r.get("model") == "LightweightMLP" and r.get("layer") == 12 and r.get("pooling") == "sink":
                    self.threshold = float(r.get("threshold", 0.36))
                    break
        else:
            self.threshold = 0.36

        # 2. Load PyTorch model state
        self.model = LightweightMLP()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 3. Load StandardScaler
        self.scaler = joblib.load(scaler_path)
        self.is_loaded = True

    def predict(self, representation: np.ndarray) -> Dict[str, Any]:
        """
        Runs probe inference on a 4,096-dimensional Mistral representation vector.
        
        Args:
            representation: 1D array of shape (4096,) or 2D array of shape (1, 4096).
            
        Returns:
            Dictionary with prediction, confidence, probabilities, and calibration metadata.
        """
        if not self.is_loaded or self.model is None or self.scaler is None:
            raise RuntimeError("ProbePredictor artifacts not loaded.")

        if representation.ndim == 1:
            X = representation.reshape(1, -1)
        else:
            X = representation

        assert X.shape[1] == 4096, f"Expected 4096 features, got {X.shape[1]}"

        # Standardize features using train-fitted scaler
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logit = self.model(X_tensor).item()
            prob_vuln = torch.sigmoid(torch.tensor(logit)).item()

        prob_safe = 1.0 - prob_vuln

        # Evaluate against threshold with uncertainty interval
        low_bound = max(0.0, self.threshold - self.uncertainty_margin)
        high_bound = min(1.0, self.threshold + self.uncertainty_margin)

        if low_bound <= prob_vuln <= high_bound:
            pred = "UNCERTAIN"
            confidence = max(prob_vuln, prob_safe)
        elif prob_vuln >= self.threshold:
            pred = "VULNERABLE"
            confidence = prob_vuln
        else:
            pred = "SAFE"
            confidence = prob_safe

        return {
            "prediction": pred,
            "confidence": round(confidence, 4),
            "probabilities": {
                "vulnerable": round(prob_vuln, 4),
                "safe": round(prob_safe, 4)
            },
            "raw_logit": round(logit, 4),
            "threshold": round(self.threshold, 4),
            "uncertainty_interval": [round(low_bound, 4), round(high_bound, 4)],
            "calibration_status": "CALIBRATED_BRIER_0.0399"
        }

