"""
Probe Models for Python SQL Injection Security Probing.

Implements:
1. LinearProbeHead: Logistic Regression classifier on pooled hidden representations.
2. MLPProbeHead: Multi-Layer Perceptron classifier for comparison.
3. SecurityProbe: Unified wrapper supporting validation-based threshold tuning,
   decision vector extraction for token projection, and persistence.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler


class PyTorchMLP(nn.Module):
    """2-layer MLP classifier head."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SecurityProbe:
    """Supervised probe head for SQL injection vulnerability detection."""

    def __init__(
        self,
        architecture: str = "linear",
        regularization_c: float = 1.0,
        random_state: int = 42,
        decision_threshold: float = 0.50,
        uncertainty_margin: float = 0.08,
        selected_layer: int = 16,
        mlp_hidden_dim: int = 128,
        mlp_epochs: int = 40,
        mlp_lr: float = 0.001,
    ):
        self.architecture = architecture.lower()
        self.regularization_c = regularization_c
        self.random_state = random_state
        self.decision_threshold = decision_threshold
        self.uncertainty_margin = uncertainty_margin
        self.selected_layer = selected_layer
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_epochs = mlp_epochs
        self.mlp_lr = mlp_lr

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.classifier: Any = None
        self.weight_vector: Optional[np.ndarray] = None
        self.bias_scalar: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SecurityProbe":
        """Fit probe head on training representations."""
        X_scaled = self.scaler.fit_transform(X)

        if self.architecture == "linear":
            self.classifier = LogisticRegression(
                C=self.regularization_c,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
            self.classifier.fit(X_scaled, y)
            self.weight_vector = self.classifier.coef_[0] # (hidden_dim,)
            self.bias_scalar = float(self.classifier.intercept_[0])

        elif self.architecture == "mlp":
            torch.manual_seed(self.random_state)
            input_dim = X.shape[1]
            model = PyTorchMLP(input_dim, self.mlp_hidden_dim)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.mlp_lr, weight_decay=0.01)
            
            # Account for class imbalance in loss
            pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            model.train()
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            for epoch in range(self.mlp_epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            self.classifier = model
            # Approximate linear surrogate decision vector from first layer for localization
            w1 = model.net[0].weight.detach().cpu().numpy() # (hidden_dim_mlp, input_dim)
            w2 = model.net[3].weight.detach().cpu().numpy() # (1, hidden_dim_mlp)
            self.weight_vector = (w2 @ w1)[0]
            self.bias_scalar = float(model.net[3].bias.detach().cpu().item())

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns vulnerability probabilities P(y=1|x)."""
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before predict_proba().")
        X_scaled = self.scaler.transform(X)

        if self.architecture == "linear":
            return self.classifier.predict_proba(X_scaled)[:, 1]
        elif self.architecture == "mlp":
            self.classifier.eval()
            with torch.no_grad():
                logits = self.classifier(torch.tensor(X_scaled, dtype=torch.float32))
                probs = torch.sigmoid(logits).cpu().numpy()
            return probs

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels using decision threshold."""
        thresh = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def tune_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray, metric: str = "f1"
    ) -> Dict[str, Any]:
        """Tuning decision threshold strictly on held-out validation data."""
        val_probs = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

        best_thresh = 0.50
        best_score = -1.0
        threshold_curve = []

        # Evaluate candidate thresholds
        test_thresholds = np.linspace(0.10, 0.90, 81)
        for t in test_thresholds:
            preds = (val_probs >= t).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            threshold_curve.append({
                "threshold": float(t),
                "f1": float(score),
                "precision": float(np.mean(y_val[preds == 1])) if preds.sum() > 0 else 0.0,
                "recall": float(np.mean(preds[y_val == 1])) if y_val.sum() > 0 else 0.0,
            })
            if score > best_score:
                best_score = score
                best_thresh = float(t)

        self.decision_threshold = best_thresh
        return {
            "optimal_threshold": best_thresh,
            "validation_f1": best_score,
            "threshold_curve": threshold_curve,
        }

    def get_decision_vector(self) -> Tuple[np.ndarray, float, StandardScaler]:
        """Returns parameters needed for token-level exploratory projection."""
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before extracting decision vector.")
        return self.weight_vector, self.bias_scalar, self.scaler

    def save(self, filepath: str) -> None:
        """Persist fitted probe parameters and configuration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "SecurityProbe":
        """Load persisted probe."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
