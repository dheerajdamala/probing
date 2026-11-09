"""
Interprocedural Dataflow Analysis Engine for SQL Injection Detection.
Traces taint, query construction, and parameterization across function and file boundaries.
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    from .project_ast_graph import ProjectASTGraph, FunctionSymbol, CallEdge
    from .ast_analyzer import ASTAnalyzer, ASTAnalysisResult, SinkLocation
except ImportError:
    from project_ast_graph import ProjectASTGraph, FunctionSymbol, CallEdge
    from ast_analyzer import ASTAnalyzer, ASTAnalysisResult, SinkLocation

@dataclass
class PathStep:
    step_number: int
    file: str
    line: int
    role: str # SOURCE, PROPAGATION, QUERY_CONSTRUCTION, SANITIZATION, SINK
    function: str
    description: str
    code_snippet: str

@dataclass
class InterproceduralFinding:
    vulnerability_type: str
    is_vulnerable: bool
    source_lines: List[Dict[str, Any]] = field(default_factory=list)
    propagation_lines: List[Dict[str, Any]] = field(default_factory=list)
    vulnerable_lines: List[Dict[str, Any]] = field(default_factory=list)
    sink_lines: List[Dict[str, Any]] = field(default_factory=list)
    dataflow_path: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    resolution_status: str = "RESOLVED" # RESOLVED, PARTIAL, UNRESOLVED
    confidence_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vulnerability_type": self.vulnerability_type,
            "is_vulnerable": self.is_vulnerable,
            "source_lines": self.source_lines,
            "propagation_lines": self.propagation_lines,
            "vulnerable_lines": self.vulnerable_lines,
            "sink_lines": self.sink_lines,
            "dataflow_path": self.dataflow_path,
            "evidence": self.evidence,
            "resolution_status": self.resolution_status,
            "confidence_note": self.confidence_note
        }

@dataclass
class InterproceduralAnalysisResult:
    analysis_status: str # COMPLETE, PARTIAL, FAILED
    findings: List[InterproceduralFinding] = field(default_factory=list)
    functions_analyzed: int = 0
    call_edges_resolved: int = 0
    call_edges_unresolved: int = 0
    maximum_depth_reached: int = 0
    messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_status": self.analysis_status,
            "findings": [f.to_dict() for f in self.findings],
            "functions_analyzed": self.functions_analyzed,
            "call_edges_resolved": self.call_edges_resolved,
            "call_edges_unresolved": self.call_edges_unresolved,
            "maximum_depth_reached": self.maximum_depth_reached,
            "messages": self.messages
        }

class InterproceduralAnalyzer:
    """Interprocedural, cross-function and cross-file dataflow analysis engine."""

    def __init__(self, max_call_depth: int = 10):
        self.max_call_depth = max_call_depth
        self.ast_analyzer = ASTAnalyzer()

    def analyze_project(self, project_dir: str) -> InterproceduralAnalysisResult:
        """Analyzes all Python files in a directory for cross-file SQL injection paths."""
        graph = ProjectASTGraph(root_dir=project_dir)

        # 1. Discover all .py files
        py_files = []
        for root, _, files in os.walk(project_dir):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))

        if not py_files:
            return InterproceduralAnalysisResult(
                analysis_status="COMPLETE",
                findings=[],
                messages=["No Python source files found in specified project directory."]
            )

        # 2. Parse and register all files
        syntax_errors = []
        for pf in py_files:
            try:
                with open(pf, "r", encoding="utf-8", errors="replace") as f_obj:
                    code = f_obj.read()
                tree = ast.parse(code, filename=pf)
                graph.register_file(pf, code, tree)
            except SyntaxError as e:
                syntax_errors.append(f"SyntaxError in {pf}: {str(e)}")

        if len(graph.file_ast_trees) == 0:
            return InterproceduralAnalysisResult(
                analysis_status="FAILED",
                findings=[],
                messages=syntax_errors or ["No valid Python source files could be parsed."]
            )

        # 3. Resolve call edges across project
        graph.resolve_all_call_edges()

        # 4. Perform cross-function taint tracing
        result = self._run_taint_tracing(graph)
        if syntax_errors:
            result.messages.extend(syntax_errors)
            result.analysis_status = "PARTIAL"
        return result

    def analyze_single_file(self, code: str, filename: str = "snippet.py") -> InterproceduralAnalysisResult:
        """Analyzes a single code string containing multiple functions."""
        graph = ProjectASTGraph()
        try:
            tree = ast.parse(code, filename=filename)
            graph.register_file(filename, code, tree)
        except SyntaxError as e:
            return InterproceduralAnalysisResult(
                analysis_status="FAILED",
                findings=[],
                messages=[f"SyntaxError in {filename}: {str(e)}"]
            )

        graph.resolve_all_call_edges()
        return self._run_taint_tracing(graph)

    def _run_taint_tracing(self, graph: ProjectASTGraph) -> InterproceduralAnalysisResult:
        findings: List[InterproceduralFinding] = []
        messages: List[str] = []
        max_depth_seen = 0

        # Step 1: Detect all sinks across all registered files
        sinks_found: List[Tuple[str, SinkLocation, Optional[FunctionSymbol]]] = []
        for f_path, tree in graph.file_ast_trees.items():
            lines = graph.file_lines[f_path]
            ast_res = self.ast_analyzer.analyze(tree, filename=f_path, lines=lines)
            for sink in ast_res.sinks:
                # Find enclosing function
                enclosing_func = self._find_enclosing_function(graph, f_path, sink.line)
                sinks_found.append((f_path, sink, enclosing_func))

        if not sinks_found:
            return InterproceduralAnalysisResult(
                analysis_status="COMPLETE",
                findings=[],
                functions_analyzed=len(graph.functions_by_qualname),
                call_edges_resolved=graph.resolved_edges_count,
                call_edges_unresolved=graph.unresolved_edges_count,
                maximum_depth_reached=0,
                messages=["No SQL execution sinks detected across analyzed functions."]
            )

        # Step 2: For each sink, trace backward across function boundaries
        for f_path, sink, enc_func in sinks_found:
            tracer = _PathTracer(graph, self.max_call_depth)
            finding = tracer.trace_sink_backwards(f_path, sink, enc_func)
            findings.append(finding)
            if tracer.max_depth_reached > max_depth_seen:
                max_depth_seen = tracer.max_depth_reached
            messages.extend(tracer.messages)

        # Determine overall analysis status
        status = "COMPLETE"
        if max_depth_seen >= self.max_call_depth or any(f.resolution_status in ("PARTIAL", "UNRESOLVED") for f in findings):
            status = "PARTIAL"
        if graph.unresolved_edges_count > 0 and any(f.is_vulnerable for f in findings):
            # If there are unresolved edges in relevant paths
            pass

        return InterproceduralAnalysisResult(
            analysis_status=status,
            findings=findings,
            functions_analyzed=len(graph.functions_by_qualname),
            call_edges_resolved=graph.resolved_edges_count,
            call_edges_unresolved=graph.unresolved_edges_count,
            maximum_depth_reached=max_depth_seen,
            messages=sorted(list(set(messages)))
        )

    def _find_enclosing_function(self, graph: ProjectASTGraph, file_path: str, line: int) -> Optional[FunctionSymbol]:
        for qn, sym in graph.functions_by_qualname.items():
            if os.path.abspath(sym.file) == os.path.abspath(file_path):
                if sym.start_line <= line <= sym.end_line:
                    return sym
        return None


class _PathTracer:
    """Internal helper to trace taint and query assembly backwards from a sink."""

    def __init__(self, graph: ProjectASTGraph, max_depth: int):
        self.graph = graph
        self.max_depth = max_depth
        self.max_depth_reached = 0
        self.messages: List[str] = []
        self.call_stack: List[str] = []

    def trace_sink_backwards(
        self,
        file_path: str,
        sink: SinkLocation,
        enclosing_func: Optional[FunctionSymbol]
    ) -> InterproceduralFinding:
        sink_snippet = self._get_line_snippet(file_path, sink.line)
        func_name = enclosing_func.name if enclosing_func else "<module>"

        sink_loc = {
            "file": os.path.basename(file_path),
            "line": sink.line,
            "column": sink.column,
            "role": "SINK",
            "function": func_name,
            "evidence": sink_snippet
        }

        # Check parameter binding directly at sink
        is_safe_param = sink.has_parameter_binding
        query_var = sink.query_var_name
        query_node = sink.query_node

        steps: List[PathStep] = []
        vulnerable_lines: List[Dict[str, Any]] = []
        source_lines: List[Dict[str, Any]] = []
        propagation_lines: List[Dict[str, Any]] = []
        sink_lines: List[Dict[str, Any]] = [sink_loc]
        evidence_items: List[Dict[str, Any]] = []

        # Evidence for sink
        evidence_items.append({
            "line": sink.line,
            "category": "SQL_EXECUTION",
            "description": f"SQL execution call '{sink.function}' invoked.",
            "support": "AST"
        })
        if is_safe_param:
            evidence_items.append({
                "line": sink.line,
                "category": "SAFE_PARAMETERIZATION",
                "description": "Query passes parameters via separate argument.",
                "support": "AST"
            })

        # Trace query construction backwards
        vuln_type = "UNCONFIRMED"
        has_unsafe_construction = False
        resolution_status = "RESOLVED"

        construction_info = self._trace_expression(
            node=query_node,
            var_name=query_var,
            curr_func=enclosing_func,
            curr_file=file_path,
            depth=1,
            steps=steps,
            vulnerable_lines=vulnerable_lines,
            source_lines=source_lines,
            propagation_lines=propagation_lines,
            evidence_items=evidence_items
        )

        has_unsafe_construction = construction_info.get("is_unsafe", False)
        inferred_type = construction_info.get("vuln_type", "OTHER")
        if construction_info.get("partial_reason"):
            resolution_status = "PARTIAL"
            self.messages.append(construction_info["partial_reason"])
        if self.max_depth_reached >= self.max_depth:
            resolution_status = "PARTIAL"
            self.messages.append(f"Maximum call depth ({self.max_depth}) exceeded during tracing.")

        # Final vulnerability evaluation
        if has_unsafe_construction:
            vuln_type = inferred_type
            # Check for hybrid
            if is_safe_param:
                vuln_type = "HYBRID_QUERY_CONSTRUCTION"
            is_vuln = True
        elif is_safe_param:
            vuln_type = "SAFE_PARAMETERIZATION"
            is_vuln = False
        elif not source_lines and not vulnerable_lines:
            vuln_type = "STATIC_OR_SAFE_SQL"
            is_vuln = False
        else:
            vuln_type = "UNCONFIRMED"
            is_vuln = False

        # Add sink step to path
        steps.append(PathStep(
            step_number=len(steps) + 1,
            file=os.path.basename(file_path),
            line=sink.line,
            role="SINK",
            function=func_name,
            description=f"Query is executed via '{sink.function}'.",
            code_snippet=sink_snippet
        ))

        # Convert steps to dicts
        path_dicts = [
            {
                "step": s.step_number,
                "file": s.file,
                "line": s.line,
                "role": s.role,
                "function": s.function,
                "description": s.description,
                "code": s.code_snippet
            }
            for s in steps
        ]

        return InterproceduralFinding(
            vulnerability_type=vuln_type,
            is_vulnerable=is_vuln,
            source_lines=source_lines,
            propagation_lines=propagation_lines,
            vulnerable_lines=vulnerable_lines,
            sink_lines=sink_lines,
            dataflow_path=path_dicts,
            evidence=evidence_items,
            resolution_status=resolution_status
        )

    def _trace_expression(
        self,
        node: Optional[ast.AST],
        var_name: Optional[str],
        curr_func: Optional[FunctionSymbol],
        curr_file: str,
        depth: int,
        steps: List[PathStep],
        vulnerable_lines: List[Dict[str, Any]],
        source_lines: List[Dict[str, Any]],
        propagation_lines: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if depth > self.max_depth:
            self.max_depth_reached = depth
            return {"is_unsafe": False, "partial_reason": f"Maximum call depth ({self.max_depth}) exceeded during tracing."}

        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

        func_name = curr_func.name if curr_func else "<module>"
        file_base = os.path.basename(curr_file)

        # 1. If variable name is given, trace its assignment in curr_func or caller parameters
        if var_name:
            # Cycle detection in variable traversal
            cycle_key = f"{file_base}:{func_name}:{var_name}"
            if cycle_key in self.call_stack:
                return {"is_unsafe": False, "partial_reason": f"Recursive call cycle detected: {cycle_key}"}
            self.call_stack.append(cycle_key)

            res = self._trace_variable(
                var_name=var_name,
                curr_func=curr_func,
                curr_file=curr_file,
                depth=depth,
                steps=steps,
                vulnerable_lines=vulnerable_lines,
                source_lines=source_lines,
                propagation_lines=propagation_lines,
                evidence_items=evidence_items
            )
            self.call_stack.pop()
            return res

        if node is None:
            return {"is_unsafe": False}

        # 2. Direct AST expression checks
        # Constants (boolean flags, integers, etc.)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float)) or node.value is None:
                return {"is_unsafe": False, "is_constant": True}

        # Subscripts (e.g. request.args['id'] or data['user'])
        if isinstance(node, ast.Subscript):
            if self._is_external_source_expr(node):
                line = node.lineno
                snippet = self._get_line_snippet(curr_file, line)
                source_lines.append({
                    "file": file_base,
                    "line": line,
                    "column": node.col_offset,
                    "role": "SOURCE",
                    "function": func_name,
                    "evidence": snippet
                })
                steps.append(PathStep(
                    step_number=len(steps) + 1,
                    file=file_base,
                    line=line,
                    role="SOURCE",
                    function=func_name,
                    description="Untrusted input retrieved from external source subscript.",
                    code_snippet=snippet
                ))
                evidence_items.append({
                    "line": line,
                    "category": "INPUT_SOURCE",
                    "description": "External subscript access.",
                    "support": "DATAFLOW"
                })
                return {"is_unsafe": False, "has_source": True}
            elif isinstance(node.value, ast.Name):
                return self._trace_variable(
                    var_name=node.value.id,
                    curr_func=curr_func,
                    curr_file=curr_file,
                    depth=depth + 1,
                    steps=steps,
                    vulnerable_lines=vulnerable_lines,
                    source_lines=source_lines,
                    propagation_lines=propagation_lines,
                    evidence_items=evidence_items
                )

        # Sanitization calls (e.g. int(x), escape(x), clean(x), str(uuid.UUID(x)))
        if isinstance(node, ast.Call):
            call_fn_name = ""
            if isinstance(node.func, ast.Name):
                call_fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_fn_name = node.func.attr
            
            # Check str() wrapping a sanitizer
            if call_fn_name == "str" and node.args and isinstance(node.args[0], ast.Call):
                inner_fn_name = getattr(node.args[0].func, "id", "") or getattr(node.args[0].func, "attr", "")
                if inner_fn_name in ("UUID", "uuid", "uuid4", "int", "float", "bool", "abs", "round"):
                    call_fn_name = f"str({inner_fn_name})"

            if call_fn_name in ("int", "float", "bool", "abs", "round", "uuid", "uuid4", "UUID", "escape") or \
               call_fn_name.startswith("str(") or \
               call_fn_name.startswith("sanitize") or call_fn_name.startswith("clean"):
                line = node.lineno
                snippet = self._get_line_snippet(curr_file, line)
                steps.append(PathStep(
                    step_number=len(steps) + 1,
                    file=file_base,
                    line=line,
                    role="SANITIZATION",
                    function=func_name,
                    description=f"Input is sanitized or safely converted via '{call_fn_name}()'.",
                    code_snippet=snippet
                ))
                evidence_items.append({
                    "line": line,
                    "category": "SANITIZATION",
                    "description": f"Sanitization via '{call_fn_name}' removes SQL injection risk.",
                    "support": "AST"
                })
                return {"is_unsafe": False, "is_sanitized": True}

        # Dict literals (e.g. {"id": request.args['id']})
        if isinstance(node, ast.Dict):
            is_any_dict_unsafe = False
            for v in node.values:
                res = self._trace_expression(
                    node=v,
                    var_name=None,
                    curr_func=curr_func,
                    curr_file=curr_file,
                    depth=depth + 1,
                    steps=steps,
                    vulnerable_lines=vulnerable_lines,
                    source_lines=source_lines,
                    propagation_lines=propagation_lines,
                    evidence_items=evidence_items
                )
                if res.get("is_unsafe") or res.get("has_source"):
                    is_any_dict_unsafe = True
            return {"is_unsafe": is_any_dict_unsafe, "has_source": is_any_dict_unsafe}

        # A. F-String
        if isinstance(node, ast.JoinedStr):
            formatted_vals = [v for v in node.values if isinstance(v, ast.FormattedValue)]
            line = node.lineno
            snippet = self._get_line_snippet(curr_file, line)
            
            if not formatted_vals:
                return {"is_unsafe": False, "is_constant": True}

            # Trace variables inside the f-string first to check taint/sanitization
            has_taint = False
            all_safe_or_sanitized = True
            for fv in formatted_vals:
                if isinstance(fv.value, ast.Constant) and isinstance(fv.value.value, (bool, int, float)):
                    continue
                for sub_name in self._extract_names(fv.value):
                    var_res = self._trace_expression(
                        node=None,
                        var_name=sub_name,
                        curr_func=curr_func,
                        curr_file=curr_file,
                        depth=depth + 1,
                        steps=steps,
                        vulnerable_lines=vulnerable_lines,
                        source_lines=source_lines,
                        propagation_lines=propagation_lines,
                        evidence_items=evidence_items
                    )
                    if var_res.get("is_sanitized") or var_res.get("is_constant"):
                        continue
                    else:
                        has_taint = True
                        all_safe_or_sanitized = False

            if not has_taint or all_safe_or_sanitized:
                return {"is_unsafe": False, "is_sanitized": True}

            clause = self._detect_sql_clause_from_joinedstr(node)
            vuln_type = "F_STRING_INTERPOLATION"
            if clause == "ORDER_BY":
                vuln_type = "DYNAMIC_ORDER_BY"
            elif clause == "TABLE_NAME":
                vuln_type = "DYNAMIC_TABLE_NAME"

            steps.append(PathStep(
                step_number=len(steps) + 1,
                file=file_base,
                line=line,
                role="QUERY_CONSTRUCTION",
                function=func_name,
                description=f"Unsafe SQL constructed using f-string interpolation ({clause or 'WHERE'}).",
                code_snippet=snippet
            ))
            vulnerable_lines.append({
                "file": file_base,
                "line": line,
                "column": node.col_offset,
                "role": "QUERY_CONSTRUCTION",
                "function": func_name,
                "evidence": snippet
            })
            evidence_items.append({
                "line": line,
                "category": vuln_type,
                "description": f"F-string incorporates unescaped variable into SQL query string.",
                "support": "AST"
            })
            return {"is_unsafe": True, "vuln_type": vuln_type}

        # B. Direct String Concatenation (+)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            line = node.lineno
            snippet = self._get_line_snippet(curr_file, line)

            # Trace operands
            has_taint = False
            for sub_name in self._extract_names(node.right):
                var_res = self._trace_expression(
                    node=None,
                    var_name=sub_name,
                    curr_func=curr_func,
                    curr_file=curr_file,
                    depth=depth + 1,
                    steps=steps,
                    vulnerable_lines=vulnerable_lines,
                    source_lines=source_lines,
                    propagation_lines=propagation_lines,
                    evidence_items=evidence_items
                )
                if var_res.get("is_sanitized") or var_res.get("is_constant"):
                    continue
                else:
                    has_taint = True

            if not has_taint and self._extract_names(node.right):
                return {"is_unsafe": False, "is_sanitized": True}

            clause = self._detect_sql_clause_from_binop(node)
            vuln_type = "DIRECT_STRING_CONCATENATION"
            if clause == "ORDER_BY":
                vuln_type = "DYNAMIC_ORDER_BY"
            elif clause == "TABLE_NAME":
                vuln_type = "DYNAMIC_TABLE_NAME"

            steps.append(PathStep(
                step_number=len(steps) + 1,
                file=file_base,
                line=line,
                role="QUERY_CONSTRUCTION",
                function=func_name,
                description=f"Unsafe SQL constructed via direct string concatenation (+) ({clause or 'WHERE'}).",
                code_snippet=snippet
            ))
            vulnerable_lines.append({
                "file": file_base,
                "line": line,
                "column": node.col_offset,
                "role": "QUERY_CONSTRUCTION",
                "function": func_name,
                "evidence": snippet
            })
            evidence_items.append({
                "line": line,
                "category": vuln_type,
                "description": "Direct string concatenation builds SQL statement.",
                "support": "AST"
            })
            return {"is_unsafe": True, "vuln_type": vuln_type}

        # C. Percent Formatting (%)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            line = node.lineno
            snippet = self._get_line_snippet(curr_file, line)
            steps.append(PathStep(
                step_number=len(steps) + 1,
                file=file_base,
                line=line,
                role="QUERY_CONSTRUCTION",
                function=func_name,
                description="Unsafe SQL constructed using Python % string formatting.",
                code_snippet=snippet
            ))
            vulnerable_lines.append({
                "file": file_base,
                "line": line,
                "column": node.col_offset,
                "role": "QUERY_CONSTRUCTION",
                "function": func_name,
                "evidence": snippet
            })
            return {"is_unsafe": True, "vuln_type": "PERCENT_FORMATTING"}

        # D. .format() or .format_map() method
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "format_map"):
            line = node.lineno
            snippet = self._get_line_snippet(curr_file, line)
            steps.append(PathStep(
                step_number=len(steps) + 1,
                file=file_base,
                line=line,
                role="QUERY_CONSTRUCTION",
                function=func_name,
                description="Unsafe SQL constructed using .format() interpolation.",
                code_snippet=snippet
            ))
            vulnerable_lines.append({
                "file": file_base,
                "line": line,
                "column": node.col_offset,
                "role": "QUERY_CONSTRUCTION",
                "function": func_name,
                "evidence": snippet
            })
            return {"is_unsafe": True, "vuln_type": "FORMAT_METHOD"}

        # E. Function Call producing query or input (e.g. build_query(...) or get_input(...))
        elif isinstance(node, ast.Call):
            return self._trace_call_return(
                call_node=node,
                curr_func=curr_func,
                curr_file=curr_file,
                depth=depth,
                steps=steps,
                vulnerable_lines=vulnerable_lines,
                source_lines=source_lines,
                propagation_lines=propagation_lines,
                evidence_items=evidence_items
            )

        # F. Name node
        elif isinstance(node, ast.Name):
            return self._trace_variable(
                var_name=node.id,
                curr_func=curr_func,
                curr_file=curr_file,
                depth=depth,
                steps=steps,
                vulnerable_lines=vulnerable_lines,
                source_lines=source_lines,
                propagation_lines=propagation_lines,
                evidence_items=evidence_items
            )

        return {"is_unsafe": False}

    def _trace_variable(
        self,
        var_name: str,
        curr_func: Optional[FunctionSymbol],
        curr_file: str,
        depth: int,
        steps: List[PathStep],
        vulnerable_lines: List[Dict[str, Any]],
        source_lines: List[Dict[str, Any]],
        propagation_lines: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        func_name = curr_func.name if curr_func else "<module>"
        file_base = os.path.basename(curr_file)

        # Whitelist validation check in current function
        if self._is_whitelisted_in_func(var_name, curr_func):
            line = curr_func.start_line if curr_func else 1
            snippet = self._get_line_snippet(curr_file, line)
            steps.append(PathStep(
                step_number=len(steps) + 1,
                file=file_base,
                line=line,
                role="SANITIZATION",
                function=func_name,
                description=f"Variable '{var_name}' is validated against a whitelist/comparator in '{func_name}'.",
                code_snippet=snippet
            ))
            evidence_items.append({
                "line": line,
                "category": "WHITELIST_VALIDATION",
                "description": f"Variable '{var_name}' is validated against an allowed set.",
                "support": "AST"
            })
            return {"is_unsafe": False, "is_sanitized": True}

        # A. Check if assigned locally inside curr_func
        if curr_func and var_name in curr_func.assignments:
            assigns = curr_func.assignments[var_name]
            for val_node, l, c in assigns:
                snippet = self._get_line_snippet(curr_file, l)
                # Check if this assignment is an external input source (e.g. request.args.get or request.args['id'])
                if self._is_external_source_expr(val_node):
                    source_lines.append({
                        "file": file_base,
                        "line": l,
                        "column": c,
                        "role": "SOURCE",
                        "function": func_name,
                        "evidence": snippet
                    })
                    steps.append(PathStep(
                        step_number=len(steps) + 1,
                        file=file_base,
                        line=l,
                        role="SOURCE",
                        function=func_name,
                        description=f"Untrusted input retrieved from external source into variable '{var_name}'.",
                        code_snippet=snippet
                    ))
                    evidence_items.append({
                        "line": l,
                        "category": "INPUT_SOURCE",
                        "description": f"External input assigned to '{var_name}'.",
                        "support": "DATAFLOW"
                    })
                    return {"is_unsafe": False, "has_source": True}

                # Otherwise, trace the assigned value expression
                return self._trace_expression(
                    node=val_node,
                    var_name=None,
                    curr_func=curr_func,
                    curr_file=curr_file,
                    depth=depth + 1,
                    steps=steps,
                    vulnerable_lines=vulnerable_lines,
                    source_lines=source_lines,
                    propagation_lines=propagation_lines,
                    evidence_items=evidence_items
                )

        # B. Check if var_name is a parameter of curr_func
        if curr_func and var_name in curr_func.params:
            param_idx = curr_func.params.index(var_name)
            param_cycle_key = f"{file_base}:{func_name}:param:{var_name}"
            if param_cycle_key in self.call_stack:
                return {
                    "is_unsafe": False,
                    "partial_reason": f"Recursive parameter cycle detected: {param_cycle_key}"
                }
            self.call_stack.append(param_cycle_key)

            steps.append(PathStep(
                step_number=len(steps) + 1,
                file=file_base,
                line=curr_func.start_line,
                role="PROPAGATION",
                function=func_name,
                description=f"Value propagated into parameter '{var_name}' of '{func_name}'.",
                code_snippet=self._get_line_snippet(curr_file, curr_func.start_line)
            ))
            propagation_lines.append({
                "file": file_base,
                "line": curr_func.start_line,
                "column": 0,
                "role": "PROPAGATION",
                "function": func_name,
                "evidence": f"def {func_name}(..., {var_name}, ...)"
            })

            # CROSS-FUNCTION RESOLUTION: Find all call sites that invoke curr_func
            call_edges = [e for e in self.graph.call_edges if e.resolved_target == curr_func]
            if not call_edges:
                self.call_stack.pop()
                return {
                    "is_unsafe": False,
                    "partial_reason": f"No call sites found for function '{curr_func.qualname}'; parameter '{var_name}' is unconstrained."
                }

            # Trace argument at each call site
            is_any_unsafe = False
            is_any_sanitized = False
            inferred_type = "OTHER"
            collected_partial = None
            for edge in call_edges:
                caller_sym = self.graph.functions_by_qualname.get(edge.caller_qualname)
                caller_file = edge.caller_file
                call_node = edge.call_node

                arg_node = None
                if param_idx < len(call_node.args):
                    arg_node = call_node.args[param_idx]
                else:
                    for kw in call_node.keywords:
                        if kw.arg == var_name:
                            arg_node = kw.value

                if arg_node:
                    steps.append(PathStep(
                        step_number=len(steps) + 1,
                        file=os.path.basename(caller_file),
                        line=edge.call_line,
                        role="PROPAGATION",
                        function=caller_sym.name if caller_sym else "<module>",
                        description=f"Argument passed to '{func_name}' at call site.",
                        code_snippet=self._get_line_snippet(caller_file, edge.call_line)
                    ))
                    arg_res = self._trace_expression(
                        node=arg_node,
                        var_name=None,
                        curr_func=caller_sym,
                        curr_file=caller_file,
                        depth=depth + 1,
                        steps=steps,
                        vulnerable_lines=vulnerable_lines,
                        source_lines=source_lines,
                        propagation_lines=propagation_lines,
                        evidence_items=evidence_items
                    )
                    if arg_res.get("partial_reason"):
                        collected_partial = arg_res["partial_reason"]
                    if arg_res.get("is_sanitized"):
                        is_any_sanitized = True
                    if arg_res.get("is_unsafe"):
                        is_any_unsafe = True
                        inferred_type = arg_res.get("vuln_type", "OTHER")

            self.call_stack.pop()
            res_dict = {
                "is_unsafe": is_any_unsafe,
                "is_sanitized": is_any_sanitized and not is_any_unsafe,
                "vuln_type": inferred_type
            }
            if collected_partial:
                res_dict["partial_reason"] = collected_partial
            return res_dict

        return {"is_unsafe": False}

    def _trace_call_return(
        self,
        call_node: ast.Call,
        curr_func: Optional[FunctionSymbol],
        curr_file: str,
        depth: int,
        steps: List[PathStep],
        vulnerable_lines: List[Dict[str, Any]],
        source_lines: List[Dict[str, Any]],
        propagation_lines: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Check if direct external input call, e.g. request.args.get(...)
        if self._is_external_source_expr(call_node):
            line = call_node.lineno
            snippet = self._get_line_snippet(curr_file, line)
            source_lines.append({
                "file": os.path.basename(curr_file),
                "line": line,
                "column": call_node.col_offset,
                "role": "SOURCE",
                "function": curr_func.name if curr_func else "<module>",
                "evidence": snippet
            })
            steps.append(PathStep(
                step_number=len(steps) + 1,
                file=os.path.basename(curr_file),
                line=line,
                role="SOURCE",
                function=curr_func.name if curr_func else "<module>",
                description="Untrusted input retrieved from external source.",
                code_snippet=snippet
            ))
            return {"is_unsafe": False, "has_source": True}

        # Resolve target function symbol
        target_name = self.graph._extract_call_target_name(call_node)
        caller_qn = curr_func.qualname if curr_func else None
        target_sym = self.graph._resolve_target(curr_file, target_name, caller_qualname=caller_qn)

        if not target_sym:
            return {
                "is_unsafe": False,
                "partial_reason": f"External or unresolvable call to '{target_name}' at line {call_node.lineno}."
            }

        # Check call recursion cycle
        call_cycle_key = f"{target_sym.file}:{target_sym.name}"
        if call_cycle_key in self.call_stack:
            return {
                "is_unsafe": False,
                "partial_reason": f"Recursive call cycle detected: {call_cycle_key}"
            }
        self.call_stack.append(call_cycle_key)

        # Helper returns: trace all return statements of target_sym
        steps.append(PathStep(
            step_number=len(steps) + 1,
            file=os.path.basename(curr_file),
            line=call_node.lineno,
            role="PROPAGATION",
            function=curr_func.name if curr_func else "<module>",
            description=f"Invokes helper function '{target_sym.name}'.",
            code_snippet=self._get_line_snippet(curr_file, call_node.lineno)
        ))

        if not target_sym.returns:
            self.call_stack.pop()
            return {"is_unsafe": False}

        is_any_unsafe = False
        is_any_sanitized = False
        inferred_type = "OTHER"
        collected_partial = None

        for ret in target_sym.returns:
            if ret.value:
                ret_snippet = self._get_line_snippet(target_sym.file, ret.lineno)
                steps.append(PathStep(
                    step_number=len(steps) + 1,
                    file=os.path.basename(target_sym.file),
                    line=ret.lineno,
                    role="PROPAGATION",
                    function=target_sym.name,
                    description=f"Helper '{target_sym.name}' returns computed value.",
                    code_snippet=ret_snippet
                ))
                ret_res = self._trace_expression(
                    node=ret.value,
                    var_name=None,
                    curr_func=target_sym,
                    curr_file=target_sym.file,
                    depth=depth + 1,
                    steps=steps,
                    vulnerable_lines=vulnerable_lines,
                    source_lines=source_lines,
                    propagation_lines=propagation_lines,
                    evidence_items=evidence_items
                )
                if ret_res.get("partial_reason"):
                    collected_partial = ret_res["partial_reason"]
                if ret_res.get("is_sanitized"):
                    is_any_sanitized = True
                if ret_res.get("is_unsafe"):
                    is_any_unsafe = True
                    inferred_type = ret_res.get("vuln_type", "OTHER")

        self.call_stack.pop()
        res_dict = {
            "is_unsafe": is_any_unsafe,
            "is_sanitized": is_any_sanitized and not is_any_unsafe,
            "vuln_type": inferred_type
        }
        if collected_partial:
            res_dict["partial_reason"] = collected_partial
        return res_dict

    def _is_external_source_expr(self, node: ast.AST) -> bool:
        """Identifies calls and subscripts that read untrusted external inputs."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("input", "raw_input"):
                return True
            if isinstance(node.func, ast.Attribute):
                caller_str = ""
                if isinstance(node.func.value, ast.Name):
                    caller_str = node.func.value.id
                elif isinstance(node.func.value, ast.Attribute):
                    caller_str = f"{getattr(node.func.value.value, 'id', '')}.{node.func.value.attr}"
                if any(x in caller_str.lower() for x in ("request", "req", "args", "form", "json", "params")):
                    return True
        elif isinstance(node, ast.Subscript):
            val = node.value
            if isinstance(val, ast.Attribute):
                attr_name = val.attr.lower()
                val_val = getattr(val.value, "id", "").lower()
                if any(x in attr_name or x in val_val for x in ("request", "req", "args", "form", "json", "params", "query")):
                    return True
            elif isinstance(val, ast.Name):
                if any(x in val.id.lower() for x in ("request", "req", "args", "form", "json", "params")):
                    return True
        return False

    def _is_whitelisted_in_func(self, var_name: str, func_sym: Optional[FunctionSymbol]) -> bool:
        """Detects whether a variable has been validated by an if-statement comparator (whitelist/membership)."""
        if not func_sym or not func_sym.node:
            return False
        for node in ast.walk(func_sym.node):
            if isinstance(node, ast.If):
                for child in ast.walk(node.test):
                    if isinstance(child, ast.Compare):
                        names = {n.id for n in ast.walk(child) if isinstance(n, ast.Name)}
                        if var_name in names:
                            return True
        return False

    def _extract_names(self, node: ast.AST) -> Set[str]:
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names

    def _get_line_snippet(self, file_path: str, line_num: int) -> str:
        lines = self.graph.file_lines.get(os.path.abspath(file_path), [])
        if 1 <= line_num <= len(lines):
            return lines[line_num - 1].strip()
        return ""

    def _detect_sql_clause_from_joinedstr(self, node: ast.JoinedStr) -> Optional[str]:
        parts = []
        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                parts.append(val.value.upper())
            elif isinstance(val, ast.FormattedValue):
                parts.append("{VAR}")
        s = " ".join(parts)
        if "ORDER BY" in s:
            return "ORDER_BY"
        if re.search(r"\bFROM\s+[A-Za-z0-9_]*\{|\bINTO\s+[A-Za-z0-9_]*\{|\bTABLE\s+[A-Za-z0-9_]*\{", s):
            return "TABLE_NAME"
        return "WHERE"

    def _detect_sql_clause_from_binop(self, node: ast.BinOp) -> Optional[str]:
        left_str = ""
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            left_str = node.left.value.upper()
        if "ORDER BY" in left_str:
            return "ORDER_BY"
        if re.search(r"\bFROM\s*[\"']?\s*\+|\bINTO\s*[\"']?\s*\+|\bTABLE\s*[\"']?\s*\+", left_str):
            return "TABLE_NAME"
        return "WHERE"

