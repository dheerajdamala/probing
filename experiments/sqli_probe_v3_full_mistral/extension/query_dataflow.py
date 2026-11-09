"""
Intraprocedural Query Dataflow and Line Localization Engine.
Traces dataflow from input sources through intermediate query constructions to SQL execution sinks.
"""

import ast
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class VulnerableLine:
    line: int
    role: str  # ORIGINAL_SOURCE, FORMAL_PARAMETER, PROPAGATION, SQL_CONSTRUCTION, SQL_SINK
    reason: str
    source_expr: Optional[str] = None

@dataclass
class DataflowResult:
    vulnerable_lines: List[Dict[str, Any]] = field(default_factory=list)
    has_unresolved_dataflow: bool = False
    is_interprocedural: bool = False
    messages: List[str] = field(default_factory=list)
    source_lines: List[Dict[str, Any]] = field(default_factory=list)
    formal_parameter_lines: List[Dict[str, Any]] = field(default_factory=list)
    construction_lines: List[Dict[str, Any]] = field(default_factory=list)
    sink_lines: List[Dict[str, Any]] = field(default_factory=list)

class QueryDataflowTracer:
    """Performs intraprocedural backwards taint and assignment tracing for SQL queries."""

    def __init__(self, processed_input, ast_result):
        self.code = processed_input.source_code
        self.lines = processed_input.lines
        self.ast_tree = processed_input.ast_tree
        self.ast_result = ast_result

    def trace_dataflow(self) -> DataflowResult:
        if not self.ast_tree or not self.ast_result.has_sinks:
            return DataflowResult(
                vulnerable_lines=[],
                has_unresolved_dataflow=False,
                is_interprocedural=False,
                messages=[]
            )

        vulnerable_lines_map: Dict[int, Dict[str, Any]] = {}
        messages: List[str] = []
        is_interprocedural = False
        has_unresolved = False

        # Gather all function parameters and function definition lines
        func_params: Dict[str, Set[str]] = {}
        func_def_lines: Dict[str, int] = {}
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = {arg.arg for arg in node.args.args if arg.arg != "self"}
                func_params[node.name] = params
                func_def_lines[node.name] = node.lineno

        for sink in self.ast_result.sinks:
            # 1. Register SINK line
            sink_line = sink.line
            sink_func = sink.function or "SQL execution call"
            vulnerable_lines_map[sink_line] = {
                "line": sink_line,
                "role": "SQL_SINK",
                "reason": f"SQL query reaches execution method '{sink_func}'."
            }

            # 2. Trace query variable if query is not an inline literal
            query_var = sink.query_var_name
            visited_vars = set()

            if query_var:
                self._trace_variable_assignments(
                    var_name=query_var,
                    before_line=sink_line,
                    vulnerable_lines_map=vulnerable_lines_map,
                    visited_vars=visited_vars,
                    func_params=func_params,
                    func_def_lines=func_def_lines,
                    messages=messages
                )
            else:
                # Query was passed directly as an inline expression in the call
                if sink.query_node:
                    q_line = getattr(sink.query_node, "lineno", sink_line)
                    if self._is_dynamic_expression(sink.query_node):
                        vulnerable_lines_map[q_line] = {
                            "line": q_line,
                            "role": "SQL_CONSTRUCTION",
                            "reason": "Inline SQL statement constructed dynamically with variables or formatting."
                        }

        # Check for input sources like request.args / input()
        self._find_external_input_sources(vulnerable_lines_map)

        # Sort lines by line number
        sorted_lines = sorted(vulnerable_lines_map.values(), key=lambda x: x["line"])

        # Detect interprocedural limitations
        if any("formal parameter" in m or "caller parameter" in m or "interprocedural" in m for m in messages):
            is_interprocedural = True
            has_unresolved = True

        # Split into distinct role collections
        source_lines = [vl for vl in sorted_lines if vl["role"] == "ORIGINAL_SOURCE"]
        formal_param_lines = [vl for vl in sorted_lines if vl["role"] == "FORMAL_PARAMETER"]
        construction_lines = [vl for vl in sorted_lines if vl["role"] == "SQL_CONSTRUCTION"]
        sink_lines = [vl for vl in sorted_lines if vl["role"] == "SQL_SINK"]

        # Genuine vulnerability sites (SQL_CONSTRUCTION, SQL_SINK) form vulnerable_lines,
        # with SQL_CONSTRUCTION prioritized at Top-1.
        actual_vuln_lines = [vl for vl in sorted_lines if vl["role"] in {"SQL_CONSTRUCTION", "SQL_SINK"}]
        actual_vuln_lines.sort(key=lambda x: (0 if x["role"] == "SQL_CONSTRUCTION" else 1, x["line"]))

        return DataflowResult(
            vulnerable_lines=actual_vuln_lines,
            has_unresolved_dataflow=has_unresolved,
            is_interprocedural=is_interprocedural,
            messages=messages,
            source_lines=source_lines,
            formal_parameter_lines=formal_param_lines,
            construction_lines=construction_lines,
            sink_lines=sink_lines
        )

    def _trace_variable_assignments(
        self,
        var_name: str,
        before_line: int,
        vulnerable_lines_map: Dict[int, Dict[str, Any]],
        visited_vars: Set[str],
        func_params: Dict[str, Set[str]],
        func_def_lines: Dict[str, int],
        messages: List[str]
    ):
        if var_name in visited_vars:
            return
        visited_vars.add(var_name)

        assignments = self.ast_result.assignments.get(var_name, [])
        # Find assignments before this line sorted in reverse line order
        relevant_assigns = [(node, l, c) for node, l, c in assignments if l <= before_line]
        relevant_assigns.sort(key=lambda x: x[1], reverse=True)

        if not relevant_assigns:
            # Check if var is a function parameter from caller
            is_param = False
            for f_name, p_set in func_params.items():
                if var_name in p_set:
                    is_param = True
                    def_line = func_def_lines.get(f_name, 1)
                    if def_line not in vulnerable_lines_map:
                        vulnerable_lines_map[def_line] = {
                            "line": def_line,
                            "role": "FORMAL_PARAMETER",
                            "reason": f"Formal parameter '{var_name}' of function '{f_name}' receives caller-supplied input."
                        }
                    messages.append(
                        f"Variable '{var_name}' originates from caller parameter (formal parameter) of function '{f_name}'; "
                        f"interprocedural analysis required to verify external caller dataflow."
                    )
            if not is_param:
                messages.append(f"Assignment to query variable '{var_name}' could not be traced intraprocedurally.")
            return

        for node, l, c in relevant_assigns:
            if self._is_dynamic_expression(node):
                if l not in vulnerable_lines_map:
                    vulnerable_lines_map[l] = {
                        "line": l,
                        "role": "SQL_CONSTRUCTION",
                        "reason": f"SQL string assigned/mutated into '{var_name}' constructed using dynamic formatting or concatenation."
                    }
            elif self._is_string_literal(node):
                # Safe static query initialization - do not mark as SQL_CONSTRUCTION
                pass
            else:
                # Intermediate propagation or function call
                if l not in vulnerable_lines_map:
                    vulnerable_lines_map[l] = {
                        "line": l,
                        "role": "PROPAGATION",
                        "reason": f"Intermediate assignment to '{var_name}'."
                    }

            # Recursively trace any variables inside this expression
            sub_vars = self._extract_names_from_expr(node)
            for sv in sub_vars:
                if sv != var_name:
                    self._trace_variable_assignments(
                        var_name=sv,
                        before_line=l,
                        vulnerable_lines_map=vulnerable_lines_map,
                        visited_vars=visited_vars,
                        func_params=func_params,
                        func_def_lines=func_def_lines,
                        messages=messages
                    )

    def _is_dynamic_expression(self, node: ast.AST) -> bool:
        """Returns True if the AST node represents dynamic string construction."""
        if isinstance(node, ast.JoinedStr):
            return any(isinstance(v, ast.FormattedValue) for v in node.values)
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Mod)):
                return True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"format", "join"}:
                return True
        return False

    def _is_string_literal(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return True
        return False

    def _extract_names_from_expr(self, node: ast.AST) -> Set[str]:
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names

    def _find_external_input_sources(self, vulnerable_lines_map: Dict[int, Dict[str, Any]]):
        """Scans for lines reading from common HTTP request objects or user inputs."""
        for node in ast.walk(self.ast_tree):
            line = getattr(node, "lineno", None)
            if not line or line in vulnerable_lines_map:
                continue

            # Check for request.args, request.form, request.json, request.GET, request.POST, input()
            is_source = False
            src_desc = ""

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "input":
                    is_source = True
                    src_desc = "Standard user input retrieved via input()."
                elif isinstance(node.func, ast.Attribute):
                    val_str = ""
                    if isinstance(node.func.value, ast.Attribute):
                        val_str = f"{getattr(node.func.value.value, 'id', '')}.{node.func.value.attr}"
                    elif isinstance(node.func.value, ast.Name):
                        val_str = node.func.value.id

                    if "request" in val_str or "req" in val_str or node.func.attr in {"get", "getlist", "json"}:
                        is_source = True
                        src_desc = f"External HTTP request parameter retrieved via {val_str}.{node.func.attr}."
            elif isinstance(node, ast.Subscript):
                val_str = ""
                if isinstance(node.value, ast.Attribute):
                    val_str = f"{getattr(node.value.value, 'id', '')}.{node.value.attr}"
                elif isinstance(node.value, ast.Name):
                    val_str = node.value.id

                if "request" in val_str or "req" in val_str:
                    is_source = True
                    src_desc = f"External HTTP request parameter retrieved via {val_str}[...]."

            if is_source:
                vulnerable_lines_map[line] = {
                    "line": line,
                    "role": "ORIGINAL_SOURCE",
                    "reason": src_desc
                }
