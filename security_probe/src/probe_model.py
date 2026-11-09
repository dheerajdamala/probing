"""Probe model module for representation-based security analysis.

Implements lightweight linear and MLP probes trained on frozen LLM representations.
Supports whole-code vulnerability classification, optional CWE category prediction,
calibrated probability estimation, and token-level projection.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class SecurityProbe:
    """Supervised representation probe for code security analysis."""

    def __init__(
        self,
        architecture: str = "linear",
        regularization_c: float = 1.0,
        random_state: int = 42,
        decision_threshold: float = 0.50,
        uncertainty_margin: float = 0.08,
        selected_layer: int = 16,
    ):
        self.architecture = architecture
        self.regularization_c = float(regularization_c)
        self.random_state = int(random_state)
        self.decision_threshold = float(decision_threshold)
        self.uncertainty_margin = float(uncertainty_margin)
        self.selected_layer = int(selected_layer)

        self.scaler = StandardScaler()
        self.classifier: Optional[LogisticRegression] = None
        self.cwe_classifier: Optional[LogisticRegression] = None
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None
        self.cwe_classes_: Optional[List[str]] = None
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cwe_labels: Optional[List[str]] = None,
        max_iter: int = 1000,
    ) -> SecurityProbe:
        """Fit probe on training representations X and binary labels y (0=benign, 1=vulnerable)."""
        if len(X) == 0:
            raise ValueError("Cannot train probe on empty dataset.")

        # Fit feature scaler
        X_scaled = self.scaler.fit_transform(X)

        unique_y = np.unique(y)
        if len(unique_y) < 2:
            logger.warning(f"Training dataset has only one class ({unique_y}). Minimum 2 classes required.")

        # Fit binary vulnerability classifier
        self.classifier = LogisticRegression(
            C=self.regularization_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.classifier.fit(X_scaled, y)
        self.classes_ = self.classifier.classes_

        # Fit optional CWE multi-class classifier
        if cwe_labels is not None and len(set(cwe_labels)) > 1:
            self.cwe_classes_ = sorted(list(set(cwe_labels)))
            self.cwe_classifier = LogisticRegression(
                C=self.regularization_c,
                penalty="l2",
                solver="lbfgs",
                max_iter=max_iter,
                random_state=self.random_state,
                class_weight="balanced",
            )
            self.cwe_classifier.fit(X_scaled, cwe_labels)

        self.is_fitted = True
        logger.info(
            f"Fitted SecurityProbe ({self.architecture}) on {X.shape[0]} samples (dim={X.shape[1]}). "
            f"Classes: {self.classes_}."
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probability of vulnerability P(y=1|x)."""
        if not self.is_fitted or self.classifier is None:
            raise RuntimeError("Probe must be fitted before predict_proba.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)
        # Class 1 probability
        if 1 in self.classes_:
            idx = np.where(self.classes_ == 1)[0][0]
            return probs[:, idx]
        return probs[:, -1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary vulnerability label using calibrated decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.decision_threshold).astype(int)

    def predict_cwe(self, X: np.ndarray) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """Predict CWE category and class probability distribution."""
        if self.cwe_classifier is None:
            return None, None
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        pred_cwe = self.cwe_classifier.predict(X_scaled)[0]
        probs = self.cwe_classifier.predict_proba(X_scaled)[0]
        dist = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.cwe_classifier.classes_, probs)}
        return pred_cwe, dist

    def score_token_hidden_states(self, token_hidden_states: torch.Tensor) -> np.ndarray:
        """Project token hidden states through the linear decision boundary.
        
        Args:
            token_hidden_states: Tensor of shape (seq_len, hidden_dim).
            
        Returns:
            1D numpy array of token vulnerability scores in [0.0, 1.0].
        """
        if not self.is_fitted or self.classifier is None:
            raise RuntimeError("Probe must be fitted before scoring tokens.")

        # Convert to numpy and scale
        tokens_np = token_hidden_states.cpu().numpy()
        tokens_scaled = self.scaler.transform(tokens_np)

        # Compute decision function: w^T x + b
        logits = self.classifier.decision_function(tokens_scaled)
        scores = sigmoid(logits)
        return scores

    def save(self, filepath: Path | str) -> Path:
        """Serialize probe weights, scaler, and metadata to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "architecture": self.architecture,
            "regularization_c": self.regularization_c,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "uncertainty_margin": self.uncertainty_margin,
            "selected_layer": self.selected_layer,
            "is_fitted": self.is_fitted,
            "classes_": self.classes_,
            "cwe_classes_": self.cwe_classes_,
            "scaler": self.scaler,
            "classifier": self.classifier,
            "cwe_classifier": self.cwe_classifier,
            "training_metadata": self.training_metadata,
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Saved SecurityProbe to {out_path}.")
        return out_path

    @classmethod
    def load(cls, filepath: Path | str) -> SecurityProbe:
        """Deserialize probe from disk."""
        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"Probe artifact not found at {in_path}.")

        with open(in_path, "rb") as f:
            payload = pickle.load(f)

        probe = cls(
            architecture=payload.get("architecture", "linear"),
            regularization_c=payload.get("regularization_c", 1.0),
            random_state=payload.get("random_state", 42),
            decision_threshold=payload.get("decision_threshold", 0.50),
            uncertainty_margin=payload.get("uncertainty_margin", 0.08),
            selected_layer=payload.get("selected_layer", 16),
        )
        probe.is_fitted = payload.get("is_fitted", False)
        probe.classes_ = payload.get("classes_")
        probe.cwe_classes_ = payload.get("cwe_classes_")
        probe.scaler = payload.get("scaler")
        probe.classifier = payload.get("classifier")
        probe.cwe_classifier = payload.get("cwe_classifier")
        probe.training_metadata = payload.get("training_metadata", {})
        return probe
