"""
AST-based SQL Sink and Query Construction Analyzer for Python.
Identifies execution sinks, parameterization styles, and syntactic construction patterns.
"""

import ast
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

# Recognized database execution methods
KNOWN_SQL_METHODS = {
    "execute", "executemany", "raw", "select", "query",
    "fetch", "fetchall", "fetchone", "scalar", "scalars",
    "execute_sql", "extra", "execute_query"
}

# Recognized database receiver object names
KNOWN_DB_RECEIVERS = {
    "cursor", "cur", "conn", "connection", "db", "session",
    "engine", "self", "adapter", "client", "r", "db_session",
    "objects", "database"
}

@dataclass
class SinkLocation:
    function: str
    file: str
    line: int
    column: int
    end_line: int
    end_column: int
    query_node: Optional[ast.AST] = None
    query_var_name: Optional[str] = None
    has_parameter_binding: bool = False
    parameter_node: Optional[ast.AST] = None

@dataclass
class ConstructionPattern:
    pattern_type: str  # DIRECT_STRING_CONCATENATION, F_STRING_INTERPOLATION, PERCENT_FORMATTING, FORMAT_METHOD, DYNAMIC_IDENTIFIER, SAFE_DYNAMIC_PLACEHOLDERS, etc.
    line: int
    column: int
    description: str
    node: ast.AST
    is_safe_placeholder: bool = False
    is_whitelisted: bool = False
    target_clause: Optional[str] = None  # ORDER_BY, TABLE_NAME, COLUMN_NAME, WHERE

@dataclass
class ASTAnalysisResult:
    has_sinks: bool
    sinks: List[SinkLocation] = field(default_factory=list)
    construction_patterns: List[ConstructionPattern] = field(default_factory=list)
    query_variables: Set[str] = field(default_factory=set)
    assignments: Dict[str, List[Tuple[ast.AST, int, int]]] = field(default_factory=dict) # var_name -> [(node, line, col)]
    placeholder_variables: Set[str] = field(default_factory=set)
    whitelisted_variables: Set[str] = field(default_factory=set)
    messages: List[str] = field(default_factory=list)

class SQLASTVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines: List[str], defined_funcs: Optional[Set[str]] = None):
        self.filename = filename
        self.lines = lines
        self.defined_funcs: Set[str] = defined_funcs or set()
        self.sinks: List[SinkLocation] = []
        self.patterns: List[ConstructionPattern] = []
        self.query_vars: Set[str] = set()
        self.assignments: Dict[str, List[Tuple[ast.AST, int, int]]] = {}
        self.placeholder_vars: Set[str] = set()
        self.whitelisted_vars: Set[str] = set()
        self.messages: List[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Ignore 'from ... import ...' statements so 'from' is never confused with SQL
        pass

    def visit_Assign(self, node: ast.Assign):
        # Record variable assignments
        for target in node.targets:
            target_names = self._extract_names(target)
            for var_name in target_names:
                if var_name not in self.assignments:
                    self.assignments[var_name] = []
                self.assignments[var_name].append((node.value, node.lineno, node.col_offset))

                # Check if assigned value is dynamic placeholder generation
                if self._is_dynamic_placeholder_expr(node.value):
                    self.placeholder_vars.add(var_name)

                # Check if assigned value is an allowlist lookup or mapping
                if self._is_allowlist_lookup_expr(node.value):
                    self.whitelisted_vars.add(var_name)

                # Check if the assigned value looks like an SQL construction
                self._check_expression_for_sql_construction(node.value, assigned_var=var_name)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        # Record augmented assignments (e.g. query += "...")
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            if var_name not in self.assignments:
                self.assignments[var_name] = []
            self.assignments[var_name].append((node.value, node.lineno, node.col_offset))
            self._check_expression_for_sql_construction(
                node.value,
                assigned_var=var_name,
                is_aug=True,
                target_line=node.lineno
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        target_names = self._extract_names(node.target)
        for t_name in target_names:
            if t_name not in self.assignments:
                self.assignments[t_name] = []
            self.assignments[t_name].append((node.iter, node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        target_names = self._extract_names(node.target)
        for t_name in target_names:
            if t_name not in self.assignments:
                self.assignments[t_name] = []
            self.assignments[t_name].append((node.iter, node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        # Check if condition validates an allowlist
        if isinstance(node.test, ast.Compare):
            for op in node.test.ops:
                if isinstance(op, ast.In):
                    left_names = self._extract_names(node.test.left)
                    for ln in left_names:
                        self.whitelisted_vars.add(ln)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check if call is an SQL execution sink
        sink_loc = self._detect_sql_sink(node)
        if sink_loc:
            self.sinks.append(sink_loc)
            if sink_loc.query_node:
                self._check_expression_for_sql_construction(sink_loc.query_node, at_sink=True)
                if isinstance(sink_loc.query_node, ast.Name):
                    self.query_vars.add(sink_loc.query_node.id)
                    sink_loc.query_var_name = sink_loc.query_node.id
        self.generic_visit(node)

    def _detect_sql_sink(self, node: ast.Call) -> Optional[SinkLocation]:
        """Determine if a Call AST node represents a database query execution sink."""
        func_name = ""
        is_sink = False
        
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr.lower()
            if method_name in KNOWN_SQL_METHODS:
                # Check receiver
                receiver_name = ""
                if isinstance(node.func.value, ast.Name):
                    receiver_name = node.func.value.id.lower()
                elif isinstance(node.func.value, ast.Attribute):
                    receiver_name = node.func.value.attr.lower()
                
                # If receiver is a known DB object or method is unmistakably execute/raw/executemany
                if receiver_name in KNOWN_DB_RECEIVERS or method_name in {"execute", "executemany", "raw"}:
                    is_sink = True
                    rec_str = node.func.value.id if isinstance(node.func.value, ast.Name) else "db"
                    func_name = f"{rec_str}.{node.func.attr}"
        elif isinstance(node.func, ast.Name):
            # Direct function call like raw_sql(query) - but exclude user-defined functions in current project
            if node.func.id not in self.defined_funcs and node.func.id.lower() in {"execute_sql", "raw_sql"}:
                is_sink = True
                func_name = node.func.id

        if not is_sink:
            return None

        # Extract query argument and parameter argument
        query_node = None
        param_node = None
        has_params = False

        if node.args:
            query_node = node.args[0]
            if len(node.args) > 1:
                has_params = True
                param_node = node.args[1]

        # Also check keyword arguments
        for kw in node.keywords:
            if kw.arg in {"sql", "query", "statement"}:
                query_node = kw.value
            elif kw.arg == "where":
                # Django QuerySet.extra(where=[...])
                if isinstance(kw.value, (ast.List, ast.Tuple)) and kw.value.elts:
                    query_node = kw.value.elts[0]
                else:
                    query_node = kw.value
            elif kw.arg in {"params", "parameters", "args"}:
                has_params = True
                param_node = kw.value

        end_l = getattr(node, "end_lineno", node.lineno)
        end_c = getattr(node, "end_col_offset", node.col_offset + 10)

        return SinkLocation(
            function=func_name,
            file=self.filename,
            line=node.lineno,
            column=node.col_offset,
            end_line=end_l,
            end_column=end_c,
            query_node=query_node,
            has_parameter_binding=has_params,
            parameter_node=param_node
        )

    def _is_dynamic_placeholder_expr(self, node: ast.AST) -> bool:
        """Check if an AST expression produces dynamic SQL parameter placeholders."""
        if node is None:
            return False

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            if node.args:
                arg0 = node.args[0]
                if self._is_placeholder_multiplication(arg0):
                    return True
                elif isinstance(arg0, ast.GeneratorExp):
                    for child in ast.walk(arg0):
                        if isinstance(child, ast.Constant) and child.value in {"%s", "?", "$1"}:
                            return True
                elif isinstance(arg0, (ast.List, ast.Tuple)):
                    for elt in arg0.elts:
                        if isinstance(elt, ast.Constant) and elt.value in {"%s", "?", "$1"}:
                            return True
        elif self._is_placeholder_multiplication(node):
            return True
        return False

    def _is_allowlist_lookup_expr(self, node: ast.AST) -> bool:
        """Check if an AST expression retrieves a value via an allowlist dictionary or clamp."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            dict_name = getattr(node.func.value, "id", "")
            if dict_name:
                # Do not treat generic parameter/input dicts as allowlists
                if any(x in dict_name.lower() for x in ("request", "req", "args", "form", "json", "params", "payload", "data", "body", "input")):
                    return False
                if any(kw in dict_name.lower() for kw in ("allow", "white", "valid", "safe", "mapping")):
                    return True
        elif isinstance(node, ast.IfExp):
            if isinstance(node.test, ast.Compare):
                return any(isinstance(op, ast.In) for op in node.test.ops)
        return False

    def _extract_names(self, node: ast.AST) -> Set[str]:
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names

    def _check_expression_for_sql_construction(
        self,
        node: ast.AST,
        assigned_var: Optional[str] = None,
        at_sink: bool = False,
        is_aug: bool = False,
        target_line: Optional[int] = None
    ):
        """Analyze an AST expression to identify how an SQL query is constructed."""
        if node is None:
            return

        eff_line = target_line if target_line is not None else getattr(node, "lineno", 1)
        eff_col = getattr(node, "col_offset", 0)

        # 1. F-String Interpolation: ast.JoinedStr
        if isinstance(node, ast.JoinedStr):
            formatted_values = [v for v in node.values if isinstance(v, ast.FormattedValue)]
            if formatted_values:
                string_text = self._extract_joinedstr_text(node)
                is_sql = self._contains_sql_keywords(string_text)
                target_clause = self._detect_sql_clause(string_text)

                # Check if ALL formatted values are safe placeholders
                all_placeholders = all(
                    (isinstance(v.value, ast.Name) and v.value.id in self.placeholder_vars) or
                    self._is_dynamic_placeholder_expr(v.value)
                    for v in formatted_values
                )
                # Check if ALL formatted values are allowlisted
                all_whitelisted = len(self.whitelisted_vars) > 0 and all(
                    (isinstance(v.value, ast.Name) and v.value.id in self.whitelisted_vars)
                    for v in formatted_values
                )

                if all_placeholders:
                    pattern_type = "SAFE_DYNAMIC_PLACEHOLDERS"
                    desc = "Parameter placeholders dynamically generated in query string (safe pattern)."
                    is_safe_placeholder = True
                else:
                    pattern_type = "F_STRING_INTERPOLATION"
                    desc = "SQL query string constructed using f-string interpolation."
                    if target_clause:
                        desc += f" Affects SQL clause: {target_clause}."
                    is_safe_placeholder = False

                self.patterns.append(ConstructionPattern(
                    pattern_type=pattern_type,
                    line=eff_line,
                    column=eff_col,
                    description=desc,
                    node=node,
                    is_safe_placeholder=is_safe_placeholder,
                    is_whitelisted=all_whitelisted,
                    target_clause=target_clause
                ))

        # 2. Binary Add (Concatenation): ast.BinOp with op=ast.Add
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            text_repr = self._get_binop_string_representation(node)
            is_placeholder = self._is_placeholder_multiplication(node)
            target_clause = self._detect_sql_clause(text_repr)

            pattern_type = "DIRECT_STRING_CONCATENATION"
            desc = "SQL query constructed via direct string concatenation (+)."
            if target_clause:
                desc += f" Affects SQL clause: {target_clause}."

            self.patterns.append(ConstructionPattern(
                pattern_type=pattern_type,
                line=eff_line,
                column=eff_col,
                description=desc,
                node=node,
                is_safe_placeholder=is_placeholder,
                target_clause=target_clause
            ))

        # 2b. Augmented assignment with variable name or call: query += val
        elif is_aug and (isinstance(node, ast.Name) or isinstance(node, ast.Call)):
            self.patterns.append(ConstructionPattern(
                pattern_type="DIRECT_STRING_CONCATENATION",
                line=eff_line,
                column=eff_col,
                description=f"SQL query variable '{assigned_var}' mutated via augmented assignment (+=).",
                node=node,
                is_safe_placeholder=False
            ))

        # 3. Percent Formatting: ast.BinOp with op=ast.Mod (%)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            text_repr = self._get_node_text(node.left)
            target_clause = self._detect_sql_clause(text_repr)

            is_placeholder = (
                (isinstance(node.right, ast.Name) and node.right.id in self.placeholder_vars) or
                self._is_dynamic_placeholder_expr(node.right)
            )

            if is_placeholder:
                pattern_type = "SAFE_DYNAMIC_PLACEHOLDERS"
                desc = "Parameter placeholders dynamically generated in query string (safe pattern)."
                is_safe = True
            else:
                pattern_type = "PERCENT_FORMATTING"
                desc = "SQL query constructed using Python % string formatting."
                if target_clause:
                    desc += f" Affects SQL clause: {target_clause}."
                is_safe = False

            self.patterns.append(ConstructionPattern(
                pattern_type=pattern_type,
                line=eff_line,
                column=eff_col,
                description=desc,
                node=node,
                is_safe_placeholder=is_safe,
                target_clause=target_clause
            ))

        # 4. .format() or .format_map() Method Call: ast.Call
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "format_map"):
            text_repr = self._get_node_text(node.func.value)
            target_clause = self._detect_sql_clause(text_repr)

            all_placeholders = (
                all(
                    (isinstance(a, ast.Name) and a.id in self.placeholder_vars) or
                    self._is_dynamic_placeholder_expr(a)
                    for a in node.args
                ) and
                all(
                    (isinstance(kw.value, ast.Name) and kw.value.id in self.placeholder_vars) or
                    self._is_dynamic_placeholder_expr(kw.value)
                    for kw in node.keywords
                ) and (len(node.args) > 0 or len(node.keywords) > 0)
            )

            if all_placeholders:
                pattern_type = "SAFE_DYNAMIC_PLACEHOLDERS"
                desc = "Parameter placeholders dynamically generated in query string (safe pattern)."
                is_safe = True
            else:
                pattern_type = "FORMAT_METHOD"
                desc = "SQL query constructed using .format() method call."
                if target_clause:
                    desc += f" Affects SQL clause: {target_clause}."
                is_safe = False

            self.patterns.append(ConstructionPattern(
                pattern_type=pattern_type,
                line=eff_line,
                column=eff_col,
                description=desc,
                node=node,
                is_safe_placeholder=is_safe,
                target_clause=target_clause
            ))

        # 5. Dynamic placeholder generation via join (e.g. ", ".join(["%s"] * len(...)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            is_placeholder = self._is_dynamic_placeholder_expr(node)
            if is_placeholder:
                self.patterns.append(ConstructionPattern(
                    pattern_type="SAFE_DYNAMIC_PLACEHOLDERS",
                    line=eff_line,
                    column=eff_col,
                    description="Dynamic generation of query parameter placeholders (e.g. %s or ?).",
                    node=node,
                    is_safe_placeholder=True
                ))

    def _extract_joinedstr_text(self, node: ast.JoinedStr) -> str:
        parts = []
        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                parts.append(val.value)
            elif isinstance(val, ast.FormattedValue):
                parts.append("{VAR}")
        return " ".join(parts)

    def _get_binop_string_representation(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._get_binop_string_representation(node.left) + " " + self._get_binop_string_representation(node.right)
        elif isinstance(node, ast.Name):
            return f"VAR_{node.id}"
        return ""

    def _get_node_text(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        return ""

    def _contains_sql_keywords(self, text: str) -> bool:
        if not text:
            return False
        upper = text.upper()
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "JOIN", "ORDER BY", "GROUP BY", "TABLE"]
        return any(re.search(rf"\b{kw}\b", upper) for kw in sql_keywords)

    def _detect_sql_clause(self, text: str) -> Optional[str]:
        if not text:
            return None
        upper = text.upper()
        if "ORDER BY" in upper:
            return "ORDER_BY"
        if re.search(r"\bFROM\s+[A-Za-z0-9_]*(\{|VAR_|\+)|\bFROM\s*[\"']?\\s*\+|\bINTO\s+[A-Za-z0-9_]*(\{|VAR_|\+)|\bTABLE\s+[A-Za-z0-9_]*(\{|VAR_|\+)", upper):
            return "TABLE_NAME"
        if re.search(r"\bWHERE\b", upper):
            return "WHERE"
        if re.search(r"\bSET\b", upper):
            return "SET"
        return None

    def _is_placeholder_multiplication(self, node: ast.AST) -> bool:
        """Check if AST node creates a list/tuple/string of placeholders, e.g. ['%s'] * len(ids) or ('?',) * n."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if isinstance(node.left, (ast.List, ast.Tuple)) and node.left.elts:
                elt = node.left.elts[0]
                if isinstance(elt, ast.Constant) and elt.value in {"%s", "?", "$1"}:
                    return True
            elif isinstance(node.left, ast.Constant) and node.left.value in {"%s", "?", "$1"}:
                return True
        elif isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            if all(isinstance(e, ast.Constant) and e.value in {"%s", "?", "$1"} for e in node.elts):
                return True
        return False


class ASTAnalyzer:
    """Public interface for AST analysis."""

    def __init__(self):
        pass

    def analyze(self, ast_tree: Optional[ast.AST], filename: str, lines: List[str]) -> ASTAnalysisResult:
        if ast_tree is None:
            return ASTAnalysisResult(
                has_sinks=False,
                sinks=[],
                construction_patterns=[],
                query_variables=set(),
                assignments={},
                placeholder_variables=set(),
                whitelisted_variables=set(),
                messages=["AST parsing was not successful; AST analysis skipped."]
            )

        # Pre-scan for defined functions
        defined_funcs = set()
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_funcs.add(node.name)

        visitor = SQLASTVisitor(filename=filename, lines=lines, defined_funcs=defined_funcs)
        visitor.visit(ast_tree)

        return ASTAnalysisResult(
            has_sinks=len(visitor.sinks) > 0,
            sinks=visitor.sinks,
            construction_patterns=visitor.patterns,
            query_variables=visitor.query_vars,
            assignments=visitor.assignments,
            placeholder_variables=visitor.placeholder_vars,
            whitelisted_variables=visitor.whitelisted_vars,
            messages=visitor.messages
        )
