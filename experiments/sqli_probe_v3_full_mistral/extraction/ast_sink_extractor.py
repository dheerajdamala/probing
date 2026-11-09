"""
AST-Aware, Source-Only SQL Sink Region Extractor for Python.

Identifies:
1. SQL execution calls (.execute, .raw, .executemany, session.query, etc.)
2. Query arguments passed to execution sinks.
3. Relevant upstream variable assignments constructing query strings (f-strings, concatenation, .format).
4. Maps character spans to subword token indices using tokenizer offset mappings.

Strict constraints:
- Operates strictly on raw source code string with zero access to ground-truth annotations.
- Completely immune to Python from-import false positives.
- Cleanly falls back to attention-masked mean pooling when no SQL sink exists.
"""

import ast
import re
from typing import Dict, Any, List, Tuple, Set, Optional
import torch

# Known database execution method names
DB_EXECUTION_METHODS = {
    "execute", "executemany", "raw", "select", "query",
    "fetch", "fetchall", "fetchone", "scalar", "scalars"
}

# Known database receiver names
DB_RECEIVER_NAMES = {
    "cursor", "cur", "conn", "connection", "db", "session",
    "engine", "self", "adapter", "client"
}

class SQLSinkVisitor(ast.NodeVisitor):
    """AST visitor that tracks SQL execution sinks and upstream query definitions."""

    def __init__(self, code: str):
        self.code = code
        self.lines = code.splitlines()
        self.sink_spans: List[Tuple[int, int]] = []
        self.sink_lines: Set[int] = set()
        self.query_vars: Set[str] = set()
        self.detected_sinks: List[str] = []
        self.assignments: Dict[str, Tuple[ast.AST, int, int]] = {} # var_name -> (node, start_char, end_char)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Explicitly ignore 'from ... import ...' statements
        # Do not treat 'from' as an SQL keyword!
        pass

    def visit_Assign(self, node: ast.Assign):
        # Track variable assignments that construct potential query strings
        start_char, end_char = self._get_node_span(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id] = (node, start_char, end_char)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        is_sink = False
        method_name = ""

        # Check attribute calls (e.g. cursor.execute, session.query)
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr.lower()
            receiver_name = ""
            if isinstance(node.func.value, ast.Name):
                receiver_name = node.func.value.id.lower()
            elif isinstance(node.func.value, ast.Attribute):
                receiver_name = node.func.value.attr.lower()

            if method_name in DB_EXECUTION_METHODS:
                is_sink = True
            elif receiver_name in DB_RECEIVER_NAMES:
                is_sink = True

        # Check direct function calls (e.g. execute(query))
        elif isinstance(node.func, ast.Name):
            method_name = node.func.id.lower()
            if method_name in DB_EXECUTION_METHODS:
                is_sink = True

        if is_sink:
            call_start, call_end = self._get_node_span(node)
            self.sink_spans.append((call_start, call_end))
            self._add_node_lines(node)
            self.detected_sinks.append(method_name)

            # Inspect arguments passed into the sink
            for arg in node.args:
                arg_start, arg_end = self._get_node_span(arg)
                self.sink_spans.append((arg_start, arg_end))
                self._add_node_lines(arg)

                # If argument is a variable name, track its definition
                if isinstance(arg, ast.Name):
                    var_name = arg.id
                    self.query_vars.add(var_name)
                    if var_name in self.assignments:
                        _, assign_start, assign_end = self.assignments[var_name]
                        self.sink_spans.append((assign_start, assign_end))
                        self._add_node_lines(self.assignments[var_name][0])

        self.generic_visit(node)

    def _get_node_span(self, node: ast.AST) -> Tuple[int, int]:
        """Compute character offset span for an AST node."""
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            # Compute char start from lineno and col_offset
            start = 0
            for i in range(node.lineno - 1):
                start += len(self.lines[i]) + 1
            start += getattr(node, "col_offset", 0)

            end = 0
            for i in range(node.end_lineno - 1):
                end += len(self.lines[i]) + 1
            end += getattr(node, "end_col_offset", len(self.lines[node.end_lineno - 1]))
            return max(0, start), min(len(self.code), end)
        return 0, 0

    def _add_node_lines(self, node: ast.AST):
        if hasattr(node, "lineno"):
            start_l = node.lineno
            end_l = getattr(node, "end_lineno", start_l)
            for l in range(start_l, end_l + 1):
                self.sink_lines.add(l)


def extract_ast_sink_regions(code: str) -> Dict[str, Any]:
    """Parse code using AST and extract sink spans, lines, and query variables."""
    if not code or not code.strip():
        return {
            "sink_spans": [],
            "sink_lines": [],
            "detected_sinks": [],
            "query_vars": [],
            "fallback_used": True,
            "has_ast_sink": False
        }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback if code snippet is an incomplete fragment
        return {
            "sink_spans": [],
            "sink_lines": [],
            "detected_sinks": [],
            "query_vars": [],
            "fallback_used": True,
            "has_ast_sink": False
        }

    visitor = SQLSinkVisitor(code)
    visitor.visit(tree)

    has_sinks = len(visitor.sink_spans) > 0
    return {
        "sink_spans": visitor.sink_spans,
        "sink_lines": sorted(list(visitor.sink_lines)),
        "detected_sinks": visitor.detected_sinks,
        "query_vars": sorted(list(visitor.query_vars)),
        "fallback_used": not has_sinks,
        "has_ast_sink": has_sinks
    }


def pool_ast_sink_representation(
    hidden_states: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    attention_mask: torch.Tensor,
    sink_info: Dict[str, Any]
) -> Tuple[torch.Tensor, List[int], bool]:
    """
    Pool hidden states over AST-extracted sink regions.
    
    Args:
        hidden_states: Tensor of shape (seq_len, hidden_dim)
        offset_mapping: List of (char_start, char_end) for each token
        attention_mask: Tensor of shape (seq_len,)
        sink_info: Output dictionary from extract_ast_sink_regions
        
    Returns:
        pooled_vector: Tensor of shape (hidden_dim,)
        selected_token_indices: List of matched token indices
        fallback_used: Boolean indicating if whole-prompt mean fallback was used
    """
    sink_spans = sink_info.get("sink_spans", [])
    selected_indices: List[int] = []

    if sink_spans:
        for idx, (t_start, t_end) in enumerate(offset_mapping):
            # Ignore special tokens with span (0, 0)
            if t_start == 0 and t_end == 0:
                continue
            # Respect attention mask
            if attention_mask[idx].item() == 0:
                continue

            for (s_start, s_end) in sink_spans:
                # Check for character span intersection
                if max(t_start, s_start) < min(t_end, s_end):
                    selected_indices.append(idx)
                    break

    if selected_indices:
        pooled = hidden_states[selected_indices].mean(dim=0)
        return pooled, selected_indices, False
    else:
        # Documented fallback: attention-masked mean pooling across all non-padding tokens
        valid_indices = (attention_mask == 1).nonzero(as_tuple=True)[0]
        if len(valid_indices) > 0:
            pooled = hidden_states[valid_indices].mean(dim=0)
        else:
            pooled = hidden_states.mean(dim=0)
        return pooled, valid_indices.tolist(), True

