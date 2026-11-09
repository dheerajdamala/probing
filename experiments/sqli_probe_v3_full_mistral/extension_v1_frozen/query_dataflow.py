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
    role: str  # SOURCE, QUERY_CONSTRUCTION, SINK
    reason: str

@dataclass
class DataflowResult:
    vulnerable_lines: List[Dict[str, Any]] = field(default_factory=list)
    has_unresolved_dataflow: bool = False
    is_interprocedural: bool = False
    messages: List[str] = field(default_factory=list)

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

        # Gather all function parameters across AST to know which variables come from callers
        func_params: Dict[str, Set[str]] = {} # func_name -> set(param_names)
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = {arg.arg for arg in node.args.args if arg.arg != "self"}
                func_params[node.name] = params

        for sink in self.ast_result.sinks:
            # 1. Register SINK line
            sink_line = sink.line
            sink_func = sink.function or "SQL execution call"
            vulnerable_lines_map[sink_line] = {
                "line": sink_line,
                "role": "SINK",
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
                    messages=messages
                )
            else:
                # Query was passed directly as an inline expression in the call
                if sink.query_node:
                    q_line = getattr(sink.query_node, "lineno", sink_line)
                    if self._is_dynamic_expression(sink.query_node):
                        vulnerable_lines_map[q_line] = {
                            "line": q_line,
                            "role": "QUERY_CONSTRUCTION",
                            "reason": "Inline SQL statement constructed dynamically with variables or formatting."
                        }

        # Check for input sources like request.args / input()
        self._find_external_input_sources(vulnerable_lines_map)

        # Sort lines by line number
        sorted_lines = sorted(vulnerable_lines_map.values(), key=lambda x: x["line"])

        # Detect interprocedural limitations
        if any("caller parameter" in m or "interprocedural" in m for m in messages):
            is_interprocedural = True

        return DataflowResult(
            vulnerable_lines=sorted_lines,
            has_unresolved_dataflow=has_unresolved,
            is_interprocedural=is_interprocedural,
            messages=messages
        )

    def _trace_variable_assignments(
        self,
        var_name: str,
        before_line: int,
        vulnerable_lines_map: Dict[int, Dict[str, Any]],
        visited_vars: Set[str],
        func_params: Dict[str, Set[str]],
        messages: List[str]
    ):
        if var_name in visited_vars:
            return
        visited_vars.add(var_name)

        assignments = self.ast_result.assignments.get(var_name, [])
        # Find assignments before this line
        relevant_assigns = [(node, l, c) for node, l, c in assignments if l <= before_line]

        if not relevant_assigns:
            # Check if var is a function parameter from caller
            is_param = False
            for f_name, p_set in func_params.items():
                if var_name in p_set:
                    is_param = True
                    messages.append(
                        f"Variable '{var_name}' originates from caller parameter of function '{f_name}'; "
                        f"intraprocedural analysis cannot verify external caller dataflow."
                    )
            if not is_param:
                messages.append(f"Assignment to query variable '{var_name}' could not be traced intraprocedurally.")
            return

        for node, l, c in relevant_assigns:
            if self._is_dynamic_expression(node):
                if l not in vulnerable_lines_map:
                    vulnerable_lines_map[l] = {
                        "line": l,
                        "role": "QUERY_CONSTRUCTION",
                        "reason": f"SQL string assigned to '{var_name}' constructed using dynamic formatting or concatenation."
                    }
            elif self._is_string_literal(node):
                # Safe static query assignment
                pass

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

            # Check for request.args.get, request.form, request.json, input()
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

                    if "request" in val_str or "req" in val_str or node.func.attr in {"get", "getlist"}:
                        is_source = True
                        src_desc = f"External HTTP request parameter retrieved via {val_str}.{node.func.attr}."

            if is_source:
                vulnerable_lines_map[line] = {
                    "line": line,
                    "role": "SOURCE",
                    "reason": src_desc
                }

