"""
Report Generator for SQL Injection Detection, Localization & Explainability.
Formats analyzed components into the unified, schema-compliant JSON report.
"""

from typing import List, Dict, Any, Optional

class ReportGenerator:
    """Assembles all diagnostic, localization, and classification artifacts into unified report."""

    def __init__(self):
        pass

    def generate_report(
        self,
        analysis_status: str,
        messages: List[str],
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        vulnerability_type: List[str],
        sinks: List[Any],
        vulnerable_lines: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        interprocedural_analysis: Optional[Dict[str, Any]] = None,
        findings: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Constructs the unified report dictionary.
        
        Args:
            analysis_status: 'COMPLETE', 'PARTIAL', or 'FAILED'.
            messages: List of warnings, limitations, or explanations.
            prediction: 'VULNERABLE', 'SAFE', or 'UNCERTAIN'.
            confidence: Float between 0.0 and 1.0.
            probabilities: Dict with 'vulnerable' and 'safe' keys.
            vulnerability_type: List of supported vulnerability category strings.
            sinks: List of SinkLocation objects or sink dictionaries.
            vulnerable_lines: List of line localization dictionaries.
            evidence: List of structured evidence dictionaries.
            interprocedural_analysis: Optional interprocedural analysis summary dictionary.
            findings: Optional list of structured cross-function findings.
            
        Returns:
            Dictionary matching the required JSON schema.
        """
        # Format sink locations
        formatted_sinks = []
        for s in sinks:
            if isinstance(s, dict):
                formatted_sinks.append(s)
            else:
                formatted_sinks.append({
                    "function": getattr(s, "function", "execute"),
                    "file": getattr(s, "file", "snippet.py"),
                    "line": getattr(s, "line", 0),
                    "column": getattr(s, "column", 0)
                })

        report = {
            "analysis_status": analysis_status,
            "messages": messages,
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "vulnerable": round(float(probabilities.get("vulnerable", 0.0)), 4),
                "safe": round(float(probabilities.get("safe", 0.0)), 4)
            },
            "vulnerability_type": vulnerability_type,
            "sink_location": formatted_sinks,
            "vulnerable_lines": vulnerable_lines,
            "evidence": evidence
        }

        if interprocedural_analysis is not None:
            report["interprocedural_analysis"] = interprocedural_analysis
        if findings is not None:
            report["findings"] = findings

        return report

