"""Inference pipeline for single-sample representation-based security probing.

Provides a unified interface:
    result = probe_analyzer.analyze(code)
Accepts a single code snippet without requiring paired safe/vulnerable versions,
contrastive comparisons, or activation steering interventions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from security_probe.src.feature_extractor import extract_code_representations
from security_probe.src.localization import extract_suspected_findings, map_tokens_to_lines
from security_probe.src.probe_model import SecurityProbe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SecurityProbeAnalyzer:
    """End-to-end single-sample security probe analyzer."""

    def __init__(
        self,
        probe: SecurityProbe,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        layer_idx: Optional[int] = None,
        pooling_strategy: str = "mean_prompt",
        decision_threshold: Optional[float] = None,
        uncertainty_margin: Optional[float] = None,
        finding_threshold: float = 0.60,
    ):
        self.probe = probe
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx if layer_idx is not None else probe.selected_layer
        self.pooling_strategy = pooling_strategy
        self.decision_threshold = decision_threshold if decision_threshold is not None else probe.decision_threshold
        self.uncertainty_margin = uncertainty_margin if uncertainty_margin is not None else probe.uncertainty_margin
        self.finding_threshold = float(finding_threshold)

        # Ensure model is in eval mode and parameters do not require grad
        if hasattr(self.model, "eval"):
            self.model.eval()

    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze a single code sample and predict vulnerability, CWE, and exploratory spans.
        
        Args:
            code: Source code string to inspect.
            
        Returns:
            Structured dictionary with detection, localization, and reliability metadata.
        """
        # Edge case 1: Empty or whitespace-only code
        if not code or not code.strip():
            return {
                "status": "NO_FINDING",
                "is_vulnerable": False,
                "probability": 0.0,
                "confidence": 1.0,
                "predicted_cwe": None,
                "cwe_probabilities": None,
                "findings": [],
                "line_scores": {1: 0.0},
                "token_scores": [],
                "no_finding": True,
                "metadata": {
                    "layer": self.layer_idx,
                    "architecture": self.probe.architecture,
                    "decision_threshold": self.decision_threshold,
                    "uncertainty_margin": self.uncertainty_margin,
                    "finding_threshold": self.finding_threshold,
                    "code_length": 0,
                    "token_count": 0,
                    "localization_mode": "exploratory_token_projection",
                    "note": "Input was empty or whitespace-only.",
                },
            }

        try:
            # 1. Extract representations from frozen base model (strictly read-only)
            ext_res = extract_code_representations(
                model=self.model,
                tokenizer=self.tokenizer,
                code=code,
                layer_idx=self.layer_idx,
                pooling=self.pooling_strategy,
            )
            pooled_rep = ext_res["pooled_rep"]
            token_hidden = ext_res["token_hidden_states"]
            token_strings = ext_res["token_strings"]
            token_offsets = ext_res["token_offsets"]
            actual_layer = ext_res["layer_idx"]

            # 2. Overall vulnerability detection
            prob = float(self.probe.predict_proba(pooled_rep)[0])

            # 3. CWE category prediction
            pred_cwe, cwe_dist = self.probe.predict_cwe(pooled_rep)

            # 4. Token-level projection and line-level mapping
            token_scores = self.probe.score_token_hidden_states(token_hidden)
            line_scores, token_details = map_tokens_to_lines(
                code=code,
                token_strings=token_strings,
                token_offsets=token_offsets,
                token_scores=token_scores,
                aggregation="max",
            )

            # 5. Extract candidate findings
            findings = extract_suspected_findings(
                code=code,
                line_scores=line_scores,
                token_details=token_details,
                finding_threshold=self.finding_threshold,
            )

            # 6. Determine structured status and confidence
            # Confidence is calibrated distance from the neutral decision threshold
            confidence = round(float(abs(prob - self.decision_threshold) * 2.0), 4)

            # Check uncertainty region
            lower_bound = self.decision_threshold - self.uncertainty_margin
            upper_bound = self.decision_threshold + self.uncertainty_margin

            if lower_bound <= prob <= upper_bound:
                status = "UNCERTAIN"
                is_vuln = None
            elif prob > upper_bound:
                status = "VULNERABLE"
                is_vuln = True
            else:
                status = "BENIGN"
                is_vuln = False

            has_findings = len(findings) > 0
            no_finding = not has_findings

            # If overall score is safe and there are no findings, explicit NO_FINDING status
            if status == "BENIGN" and no_finding:
                status = "NO_FINDING"

            return {
                "status": status,
                "is_vulnerable": is_vuln,
                "probability": round(prob, 4),
                "confidence": confidence,
                "predicted_cwe": pred_cwe,
                "cwe_probabilities": cwe_dist,
                "findings": findings,
                "line_scores": line_scores,
                "token_scores": [
                    {"token": t["token"], "score": t["score"], "line": t["start_line"], "col": t["start_col"]}
                    for t in token_details
                ],
                "no_finding": no_finding,
                "metadata": {
                    "layer": actual_layer,
                    "architecture": self.probe.architecture,
                    "decision_threshold": self.decision_threshold,
                    "uncertainty_margin": self.uncertainty_margin,
                    "finding_threshold": self.finding_threshold,
                    "code_length": len(code),
                    "token_count": len(token_strings),
                    "localization_mode": "exploratory_token_projection",
                    "ground_truth_spans_available": False,
                    "note": "Token and line localization is an exploratory representation projection.",
                },
            }

        except Exception as e:
            logger.error(f"Error during probe inference: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "is_vulnerable": None,
                "probability": None,
                "confidence": 0.0,
                "predicted_cwe": None,
                "cwe_probabilities": None,
                "findings": [],
                "line_scores": {},
                "token_scores": [],
                "no_finding": True,
                "error": f"{type(e).__name__}: {str(e)}",
                "metadata": {
                    "layer": self.layer_idx,
                    "architecture": self.probe.architecture,
                    "decision_threshold": self.decision_threshold,
                    "uncertainty_margin": self.uncertainty_margin,
                    "finding_threshold": self.finding_threshold,
                    "code_length": len(code) if code else 0,
                    "token_count": 0,
                    "localization_mode": "exploratory_token_projection",
                },
            }
