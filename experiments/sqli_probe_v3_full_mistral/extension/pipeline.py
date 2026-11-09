"""
End-to-End SQL Injection Detection, Localization & Explainability Pipeline.
Orchestrates AST parsing, dataflow tracing, Mistral Probe V3 inference,
vulnerability classification, and structured report generation.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    from .input_handler import InputHandler, ProcessedInput
    from .ast_analyzer import ASTAnalyzer, ASTAnalysisResult
    from .query_dataflow import QueryDataflowTracer, DataflowResult
    from .representation_extractor import RepresentationExtractor
    from .probe_predictor import ProbePredictor
    from .vulnerability_classifier import VulnerabilityClassifier
    from .evidence_builder import EvidenceBuilder
    from .report_generator import ReportGenerator
    from .interprocedural_analyzer import InterproceduralAnalyzer, InterproceduralAnalysisResult
except ImportError:
    from input_handler import InputHandler, ProcessedInput
    from ast_analyzer import ASTAnalyzer, ASTAnalysisResult
    from query_dataflow import QueryDataflowTracer, DataflowResult
    from representation_extractor import RepresentationExtractor
    from probe_predictor import ProbePredictor
    from vulnerability_classifier import VulnerabilityClassifier
    from evidence_builder import EvidenceBuilder
    from report_generator import ReportGenerator
    from interprocedural_analyzer import InterproceduralAnalyzer, InterproceduralAnalysisResult

class SQLInjectionAnalysisPipeline:
    """Integrated SQL Injection Detection and Explainability System."""

    def __init__(
        self,
        probe_predictor: Optional[ProbePredictor] = None,
        representation_extractor: Optional[RepresentationExtractor] = None,
        uncertainty_margin: float = 0.05
    ):
        self.input_handler = InputHandler()
        self.ast_analyzer = ASTAnalyzer()
        self.vulnerability_classifier = VulnerabilityClassifier()
        self.evidence_builder = EvidenceBuilder()
        self.report_generator = ReportGenerator()
        self.interprocedural_analyzer = InterproceduralAnalyzer()
        
        # Load probe predictor if available
        if probe_predictor is not None:
            self.probe_predictor = probe_predictor
        else:
            try:
                self.probe_predictor = ProbePredictor(uncertainty_margin=uncertainty_margin)
            except Exception as e:
                self.probe_predictor = None

        # Load representation extractor
        if representation_extractor is not None:
            self.rep_extractor = representation_extractor
        else:
            self.rep_extractor = RepresentationExtractor()

    def analyze(
        self,
        source_code: str,
        filename: Optional[str] = "snippet.py",
        representation: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyzes Python source code for SQL injection vulnerabilities.
        
        Args:
            source_code: Python code string.
            filename: Source file name.
            representation: Optional pre-extracted 4096-dim Layer 12 representation.
            
        Returns:
            Structured JSON report complying with the extension specification.
        """
        messages = []
        fn = filename or "snippet.py"

        # Empty source code check
        if not source_code.strip():
            return self.report_generator.generate_report(
                analysis_status="COMPLETE",
                messages=["Empty source code provided; no database operations detected."],
                prediction="SAFE",
                confidence=1.0,
                probabilities={"vulnerable": 0.0, "safe": 1.0},
                vulnerability_type=[],
                sinks=[],
                vulnerable_lines=[],
                evidence=[]
            )

        # 1. Input Ingestion & Syntax Check
        inp: ProcessedInput = self.input_handler.process(source_code, filename=fn)
        if not inp.syntax_valid:
            return self.report_generator.generate_report(
                analysis_status="FAILED",
                messages=[f"Syntax Error: {inp.syntax_error_message}"],
                prediction="UNCERTAIN",
                confidence=0.5,
                probabilities={"vulnerable": 0.5, "safe": 0.5},
                vulnerability_type=["UNCONFIRMED"],
                sinks=[],
                vulnerable_lines=[],
                evidence=[]
            )

        # 2. AST Sink & Query Construction Analysis
        ast_res: ASTAnalysisResult = self.ast_analyzer.analyze(inp.ast_tree, filename=fn, lines=inp.lines)
        if not ast_res.has_sinks:
            messages.append("No database execution sinks (e.g. execute, raw, query) detected in source code.")

        # 3. Intraprocedural Dataflow Tracing
        tracer = QueryDataflowTracer(inp, ast_res)
        df_res: DataflowResult = tracer.trace_dataflow()
        messages.extend(df_res.messages)

        # 4. Neural Probe Prediction
        pred_label = "UNCERTAIN"
        confidence = 0.5
        probabilities = {"vulnerable": 0.5, "safe": 0.5}
        probe_available = False

        # Check if representation vector was provided directly
        if representation is not None and self.probe_predictor is not None:
            try:
                probe_res = self.probe_predictor.predict(representation)
                pred_label = probe_res["prediction"]
                confidence = probe_res["confidence"]
                probabilities = probe_res["probabilities"]
                probe_available = True
            except Exception as e:
                messages.append(f"Probe inference error: {str(e)}")
        elif self.rep_extractor.model is not None and self.probe_predictor is not None:
            # Full model is loaded on GPU; extract representation dynamically
            vec, rep_meta = self.rep_extractor.extract_from_code(source_code)
            if vec is not None:
                probe_res = self.probe_predictor.predict(vec)
                pred_label = probe_res["prediction"]
                confidence = probe_res["confidence"]
                probabilities = probe_res["probabilities"]
                probe_available = True
            else:
                messages.append(rep_meta.get("reason", "Representation extraction failed."))
        else:
            # Neither representation nor GPU model available.
            # Derive prediction from AST construction patterns without mock vectors.
            messages.append(
                "Full Mistral-7B model not loaded in local environment (requires GPU). "
                "Prediction evaluated using AST syntax patterns."
            )
            has_unsafe_patterns = any(
                p.pattern_type in {
                    "F_STRING_INTERPOLATION", "DIRECT_STRING_CONCATENATION",
                    "PERCENT_FORMATTING", "FORMAT_METHOD"
                } and not (p.is_safe_placeholder and any(s.has_parameter_binding for s in ast_res.sinks))
                for p in ast_res.construction_patterns
            )
            if ast_res.has_sinks:
                if has_unsafe_patterns:
                    pred_label = "VULNERABLE"
                    confidence = 0.90
                    probabilities = {"vulnerable": 0.90, "safe": 0.10}
                else:
                    pred_label = "SAFE"
                    confidence = 0.90
                    probabilities = {"vulnerable": 0.10, "safe": 0.90}
            else:
                pred_label = "SAFE"
                confidence = 0.95
                probabilities = {"vulnerable": 0.05, "safe": 0.95}

        # 5. Vulnerability Subtype Classification
        vuln_types = self.vulnerability_classifier.classify(
            ast_result=ast_res,
            dataflow_result=df_res,
            probe_prediction=pred_label,
            is_safe_code=(pred_label == "SAFE")
        )

        # 6. Structured Evidence Generation
        evidence = self.evidence_builder.build_evidence(
            ast_result=ast_res,
            dataflow_result=df_res,
            processed_input=inp
        )

        # 7. Interprocedural Call Graph & Taint Tracing
        interproc_res = self.interprocedural_analyzer.analyze_single_file(source_code, filename=fn)
        if interproc_res.findings:
            any_interproc_vuln = any(f.is_vulnerable for f in interproc_res.findings)
            if any_interproc_vuln:
                if not probe_available:
                    pred_label = "VULNERABLE"
                    confidence = max(confidence, 0.90)
                    probabilities = {"vulnerable": 0.90, "safe": 0.10}
                for f in interproc_res.findings:
                    if f.is_vulnerable:
                        if f.vulnerability_type not in vuln_types:
                            vuln_types.append(f.vulnerability_type)
                        for vl in f.vulnerable_lines:
                            if not any(x.get("line") == vl.get("line") for x in df_res.vulnerable_lines):
                                df_res.vulnerable_lines.append(vl)
                        for ev in f.evidence:
                            if not any(x.get("line") == ev.get("line") and x.get("category") == ev.get("category") for x in evidence):
                                evidence.append(ev)
            elif all(not f.is_vulnerable for f in interproc_res.findings):
                has_unresolved_sink_flow = any(
                    f.resolution_status in ("PARTIAL", "UNRESOLVED") for f in interproc_res.findings
                )
                if has_unresolved_sink_flow:
                    pred_label = "UNCERTAIN"
                    confidence = 0.50
                    probabilities = {"vulnerable": 0.50, "safe": 0.50}
                    vuln_types = ["UNCONFIRMED"]
                elif not probe_available:
                    pred_label = "SAFE"
                    confidence = max(confidence, 0.90)
                    probabilities = {"vulnerable": 0.10, "safe": 0.90}
                    vuln_types = []

        messages.extend(interproc_res.messages)
        messages = list(dict.fromkeys(messages))

        # Filter vulnerable lines: if SAFE, no lines are vulnerable
        final_vulnerable_lines = df_res.vulnerable_lines if pred_label == "VULNERABLE" else []

        # 8. Analysis Status Determination
        if not ast_res.has_sinks or df_res.is_interprocedural or pred_label == "UNCERTAIN" or not probe_available or interproc_res.analysis_status == "PARTIAL":
            status = "PARTIAL"
        else:
            status = "COMPLETE"

        # 9. Report Generation
        return self.report_generator.generate_report(
            analysis_status=status,
            messages=messages,
            prediction=pred_label,
            confidence=confidence,
            probabilities=probabilities,
            vulnerability_type=vuln_types,
            sinks=ast_res.sinks,
            vulnerable_lines=final_vulnerable_lines,
            evidence=evidence,
            interprocedural_analysis=interproc_res.to_dict(),
            findings=[f.to_dict() for f in interproc_res.findings]
        )

    def analyze_project(
        self,
        project_path: str,
        representation_provider: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyzes an entire multi-file Python project directory for cross-file SQL injection vulnerabilities.
        
        Args:
            project_path: Path to the root directory of the project.
            representation_provider: Optional provider for neural probe representations.
            
        Returns:
            Structured JSON report aggregating cross-file findings and call graphs.
        """
        interproc_res = self.interprocedural_analyzer.analyze_project(project_path)
        
        all_sinks = []
        all_vulnerable_lines = []
        all_evidence = []
        all_vuln_types = set()
        seen_lines = set()
        seen_ev = set()
        is_any_vulnerable = False

        for f in interproc_res.findings:
            if f.is_vulnerable:
                is_any_vulnerable = True
                all_vuln_types.add(f.vulnerability_type)
            elif f.vulnerability_type:
                all_vuln_types.add(f.vulnerability_type)

            for s in f.sink_lines:
                all_sinks.append(s)

            for vl in f.vulnerable_lines:
                k = (vl.get("file"), vl.get("line"))
                if k not in seen_lines:
                    seen_lines.add(k)
                    all_vulnerable_lines.append(vl)

            for ev in f.evidence:
                k = (ev.get("line"), ev.get("category"))
                if k not in seen_ev:
                    seen_ev.add(k)
                    all_evidence.append(ev)

        if is_any_vulnerable:
            pred_label = "VULNERABLE"
            confidence = 0.90
            probabilities = {"vulnerable": 0.90, "safe": 0.10}
        elif all_sinks:
            pred_label = "SAFE"
            confidence = 0.90
            probabilities = {"vulnerable": 0.10, "safe": 0.90}
        else:
            pred_label = "SAFE"
            confidence = 0.95
            probabilities = {"vulnerable": 0.05, "safe": 0.95}

        status = interproc_res.analysis_status
        if not getattr(self, "probe_predictor", None):
            status = "PARTIAL"

        return self.report_generator.generate_report(
            analysis_status=status,
            messages=sorted(list(set(interproc_res.messages))),
            prediction=pred_label,
            confidence=confidence,
            probabilities=probabilities,
            vulnerability_type=sorted(list(all_vuln_types)) if all_vuln_types else ["STATIC_OR_SAFE_SQL"],
            sinks=all_sinks,
            vulnerable_lines=all_vulnerable_lines,
            evidence=all_evidence,
            interprocedural_analysis=interproc_res.to_dict(),
            findings=[f.to_dict() for f in interproc_res.findings]
        )
