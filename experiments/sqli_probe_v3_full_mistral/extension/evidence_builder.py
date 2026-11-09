"""
Evidence Builder for SQL Injection Analysis.
Synthesizes line-anchored evidence items categorized by syntactic and dataflow support.
"""

from typing import List, Dict, Any, Optional

class EvidenceBuilder:
    """Builds structured evidence entries anchored to actual source code lines."""

    def __init__(self):
        pass

    def build_evidence(
        self,
        ast_result,
        dataflow_result,
        processed_input
    ) -> List[Dict[str, Any]]:
        """
        Constructs verifiable evidence entries.
        
        Args:
            ast_result: ASTAnalysisResult.
            dataflow_result: DataflowResult.
            processed_input: ProcessedInput.
            
        Returns:
            List of evidence dictionaries: line, category, description, support.
        """
        evidence_items: List[Dict[str, Any]] = []
        seen_keys = set()

        # 1. SQL Execution Sinks
        for s in ast_result.sinks:
            func_desc = s.function or "Database execution call"
            param_desc = " with parameter bindings" if s.has_parameter_binding else " without parameter bindings"
            desc = f"SQL execution sink '{func_desc}' invoked{param_desc}."
            key = (s.line, "SQL_EXECUTION")
            if key not in seen_keys:
                seen_keys.add(key)
                evidence_items.append({
                    "line": s.line,
                    "category": "SQL_EXECUTION",
                    "description": desc,
                    "support": "AST"
                })

            if s.has_parameter_binding:
                key_param = (s.line, "SAFE_PARAMETERIZATION")
                if key_param not in seen_keys:
                    seen_keys.add(key_param)
                    evidence_items.append({
                        "line": s.line,
                        "category": "SAFE_PARAMETERIZATION",
                        "description": "SQL statement passes parameters as a separate argument to execution sink.",
                        "support": "AST"
                    })

        # 2. Query Construction Patterns
        for p in ast_result.construction_patterns:
            cat = p.pattern_type
            if p.is_safe_placeholder:
                cat = "DYNAMIC_PLACEHOLDERS"
                desc = "Parameter placeholders dynamically generated in query string (safe pattern)."
            elif cat == "F_STRING_INTERPOLATION":
                desc = "Query string constructed via f-string expression with interpolated variables."
            elif cat == "DIRECT_STRING_CONCATENATION":
                desc = "Query string constructed via direct binary string concatenation (+)."
            elif cat == "PERCENT_FORMATTING":
                desc = "Query string formatted using Python % modulo interpolation."
            elif cat == "FORMAT_METHOD":
                desc = "Query string formatted using .format() method."
            else:
                desc = p.description

            key = (p.line, cat)
            if key not in seen_keys:
                seen_keys.add(key)
                evidence_items.append({
                    "line": p.line,
                    "category": cat,
                    "description": desc,
                    "support": "AST"
                })

        # 3. Dataflow Traces (Sources, Formal Parameters, Intermediate Flows & Construction)
        for vl in dataflow_result.vulnerable_lines:
            line_num = vl["line"]
            role = vl["role"]
            if role in ("ORIGINAL_SOURCE", "SOURCE"):
                key = (line_num, "INPUT_SOURCE")
                if key not in seen_keys:
                    seen_keys.add(key)
                    evidence_items.append({
                        "line": line_num,
                        "category": "INPUT_SOURCE",
                        "description": vl["reason"],
                        "support": "DATAFLOW"
                    })
            elif role == "FORMAL_PARAMETER":
                key = (line_num, "FORMAL_PARAMETER")
                if key not in seen_keys:
                    seen_keys.add(key)
                    evidence_items.append({
                        "line": line_num,
                        "category": "FORMAL_PARAMETER",
                        "description": vl["reason"],
                        "support": "DATAFLOW"
                    })
            elif role == "PROPAGATION":
                key = (line_num, "INTERMEDIATE_DATAFLOW")
                if key not in seen_keys:
                    seen_keys.add(key)
                    evidence_items.append({
                        "line": line_num,
                        "category": "INTERMEDIATE_DATAFLOW",
                        "description": vl["reason"],
                        "support": "DATAFLOW"
                    })
            elif role in ("SQL_CONSTRUCTION", "QUERY_CONSTRUCTION"):
                key = (line_num, "INTERMEDIATE_DATAFLOW")
                if key not in seen_keys:
                    seen_keys.add(key)
                    evidence_items.append({
                        "line": line_num,
                        "category": "INTERMEDIATE_DATAFLOW",
                        "description": vl["reason"],
                        "support": "DATAFLOW"
                    })

        # Sort evidence by line number
        evidence_items.sort(key=lambda x: x["line"])
        return evidence_items
