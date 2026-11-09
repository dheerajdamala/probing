"""
Inference Module for Single-Sample Python SQL Injection Probing.

Provides an inference API that accepts a single Python source code sample
at test time, without requiring any paired reference, contrastive pair, or
safe variant.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from security_probe_sql.src.feature_extractor import extract_code_representations
from security_probe_sql.src.localization import (
    aggregate_line_scores,
    compute_token_scores,
    extract_candidate_findings,
)
from security_probe_sql.src.probe_model import SecurityProbe

logger = logging.getLogger(__name__)


class SQLInjectionProbeAnalyzer:
    """Production inference interface for single-sample Python SQL injection probing."""

    def __init__(
        self,
        probe: SecurityProbe,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        localization_threshold: float = 0.45,
    ):
        self.probe = probe
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx if layer_idx is not None else probe.selected_layer
        self.device = device
        self.localization_threshold = localization_threshold

        # Ensure base LLM is frozen
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze a single Python source code string for SQL injection.
        
        Args:
            code: Source code string to inspect.
            
        Returns:
            JSON-compatible dictionary matching the required output schema.
        """
        # 1. Edge Case: Empty or Non-String Input
        if not isinstance(code, str) or not code.strip():
            return {
                "status": "error",
                "language": "python",
                "vulnerability_type": "sql_injection",
                "is_vulnerable": False,
                "probability": 0.0,
                "threshold": self.probe.decision_threshold,
                "outcome": "ERROR",
                "findings": [],
                "limitations": [
                    "Input code was empty, whitespace-only, or not a string. No analysis performed."
                ],
            }

        # 2. Extract Hidden State Representations
        try:
            rep_dict = extract_code_representations(
                model=self.model,
                tokenizer=self.tokenizer,
                code=code,
                layer_idx=self.layer_idx,
                pooling="mean_prompt",
                device=self.device,
                compute_offsets=True,
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                "status": "error",
                "language": "python",
                "vulnerability_type": "sql_injection",
                "is_vulnerable": False,
                "probability": 0.0,
                "threshold": self.probe.decision_threshold,
                "outcome": "ERROR",
                "findings": [],
                "limitations": [f"Tokenizer or model forward pass failed: {str(e)}"],
            }

        pooled_rep = rep_dict["pooled_rep"].reshape(1, -1)
        token_hidden = rep_dict["token_hidden_states"]
        token_strings = rep_dict["token_strings"]
        token_offsets = rep_dict["token_offsets"]

        # 3. Supervised Program-Level Detection Probability
        prob = float(self.probe.predict_proba(pooled_rep)[0])
        thresh = float(self.probe.decision_threshold)
        margin = float(self.probe.uncertainty_margin)

        # Determine Outcome Status
        if prob >= (thresh + margin):
            is_vulnerable = True
            outcome = "VULNERABLE"
        elif prob <= (thresh - margin):
            is_vulnerable = False
            outcome = "NON_VULNERABLE"
        else:
            is_vulnerable = prob >= thresh
            outcome = "UNCERTAIN"

        # 4. Exploratory Token-Level Projection and Span Localization
        weight_vec, bias_val, scaler = self.probe.get_decision_vector()
        token_scores = compute_token_scores(token_hidden, weight_vec, bias_val, scaler)
        line_data = aggregate_line_scores(code, token_strings, token_offsets, token_scores, aggregation="max")

        findings = extract_candidate_findings(
            code=code,
            line_data=line_data,
            token_strings=token_strings,
            token_offsets=token_offsets,
            token_scores=token_scores,
            score_threshold=self.localization_threshold,
            max_findings=5,
            ground_truth_spans_available=False,
        )

        limitations = [
            "Program vulnerability was detected from frozen LLM hidden-state representations.",
            "Decision threshold was selected strictly on held-out validation data.",
            "Token-level scores represent exploratory linear projections; ground-truth line annotations were not supervised during token scoring.",
        ]
        if outcome == "UNCERTAIN":
            limitations.append(
                f"Prediction falls within the uncertainty margin [{thresh - margin:.2f}, {thresh + margin:.2f}]."
            )

        return {
            "status": "completed" if outcome != "UNCERTAIN" else "uncertain",
            "language": "python",
            "vulnerability_type": "sql_injection",
            "is_vulnerable": is_vulnerable,
            "probability": round(prob, 4),
            "threshold": round(thresh, 4),
            "outcome": outcome,
            "findings": findings,
            "limitations": limitations,
            "metadata": {
                "model_layer": rep_dict["layer_idx"],
                "total_tokens": len(token_strings),
                "total_lines": len(code.splitlines()),
            },
        }
