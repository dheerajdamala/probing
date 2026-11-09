"""
Interprocedural Program Representation, Call Graph, and Cross-File AST Registry.
Parses multi-function and multi-file projects, maps function symbols, resolves imports,
and constructs interprocedural call graphs.
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

@dataclass
class CodeLocation:
    file: str
    line: int
    column: int
    end_line: int
    end_column: int

@dataclass
class FunctionSymbol:
    name: str
    qualname: str
    file: str
    node: ast.AST # FunctionDef or AsyncFunctionDef
    params: List[str]
    returns: List[ast.Return] = field(default_factory=list)
    calls: List[ast.Call] = field(default_factory=list)
    assignments: Dict[str, List[Tuple[ast.AST, int, int]]] = field(default_factory=dict)
    sinks: List[Any] = field(default_factory=list)
    sources: List[Tuple[int, str, str]] = field(default_factory=list) # (line, var_or_expr, description)
    is_method: bool = False
    parent_class: Optional[str] = None
    start_line: int = 0
    end_line: int = 0

@dataclass
class CallEdge:
    caller_qualname: str
    caller_file: str
    call_line: int
    call_node: ast.Call
    target_name: str
    resolved_target: Optional[FunctionSymbol] = None
    is_resolved: bool = False
    unresolved_reason: Optional[str] = None

class ProjectASTGraph:
    """Manages cross-file function symbols, imports, and call edge resolution."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir) if root_dir else None
        self.functions_by_qualname: Dict[str, FunctionSymbol] = {}
        self.functions_by_name: Dict[str, List[FunctionSymbol]] = {}
        self.file_ast_trees: Dict[str, ast.AST] = {}
        self.file_lines: Dict[str, List[str]] = {}
        # file_path -> {imported_name -> (target_module_path, original_exported_name)}
        self.file_imports: Dict[str, Dict[str, Tuple[str, str]]] = {}
        self.call_edges: List[CallEdge] = []
        self.class_bases: Dict[str, List[str]] = {}
        self.resolved_edges_count = 0
        self.unresolved_edges_count = 0

    def register_file(self, file_path: str, code: str, tree: ast.AST):
        norm_path = os.path.abspath(file_path)
        self.file_ast_trees[norm_path] = tree
        self.file_lines[norm_path] = code.splitlines()
        self.file_imports[norm_path] = {}

        # 1. Parse imports
        self._extract_imports(norm_path, tree)

        # 2. Extract function symbols and classes
        self._extract_symbols(norm_path, tree)

    def resolve_all_call_edges(self):
        """Resolves target FunctionSymbol for every Call node across all registered files."""
        self.resolved_edges_count = 0
        self.unresolved_edges_count = 0

        for edge in self.call_edges:
            caller_file = edge.caller_file
            target_name = edge.target_name

            resolved_func = self._resolve_target(caller_file, target_name, caller_qualname=edge.caller_qualname)
            if resolved_func:
                edge.resolved_target = resolved_func
                edge.is_resolved = True
                self.resolved_edges_count += 1
            else:
                edge.is_resolved = False
                edge.unresolved_reason = f"Unresolved target: {target_name}"
                self.unresolved_edges_count += 1

    def _extract_imports(self, file_path: str, tree: ast.AST):
        dir_name = os.path.dirname(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    # Attempt to resolve module file path
                    mod_path = self._resolve_module_path(alias.name, dir_name)
                    if mod_path:
                        self.file_imports[file_path][local_name] = (mod_path, "*")
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                mod_name = node.module or ""
                # Calculate base directory for relative imports
                base_dir = dir_name
                if level > 1:
                    for _ in range(level - 1):
                        base_dir = os.path.dirname(base_dir)

                target_mod_path = self._resolve_module_path(mod_name, base_dir)
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    self.file_imports[file_path][local_name] = (target_mod_path, alias.name)

    def _resolve_module_path(self, module_name: str, base_dir: str) -> Optional[str]:
        if not module_name:
            # e.g. from . import helper
            candidate = os.path.join(base_dir, "__init__.py")
            if os.path.isfile(candidate):
                return candidate
            return None

        parts = module_name.split(".")
        # Try relative to base_dir
        rel_py = os.path.join(base_dir, *parts) + ".py"
        if os.path.isfile(rel_py):
            return os.path.abspath(rel_py)

        rel_init = os.path.join(base_dir, *parts, "__init__.py")
        if os.path.isfile(rel_init):
            return os.path.abspath(rel_init)

        # Try relative to root_dir if configured
        if self.root_dir:
            root_py = os.path.join(self.root_dir, *parts) + ".py"
            if os.path.isfile(root_py):
                return os.path.abspath(root_py)
            root_init = os.path.join(self.root_dir, *parts, "__init__.py")
            if os.path.isfile(root_init):
                return os.path.abspath(root_init)

        return None

    def _extract_symbols(self, file_path: str, tree: ast.AST):
        class SymbolVisitor(ast.NodeVisitor):
            def __init__(self, outer):
                self.outer = outer
                self.class_stack: List[str] = []
                self.scope_stack: List[str] = []

            def visit_ClassDef(self, node: ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                self.outer.class_bases[node.name] = bases
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._handle_func(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self._handle_func(node)

            def _handle_func(self, node):
                parent_class = self.class_stack[-1] if self.class_stack else None
                is_method = parent_class is not None

                # Build qualified name
                scope_prefix = ".".join(self.scope_stack)
                if scope_prefix:
                    qual_prefix = f"{scope_prefix}."
                else:
                    qual_prefix = ""

                if parent_class:
                    qualname = f"{os.path.basename(file_path)}:{parent_class}.{node.name}"
                else:
                    qualname = f"{os.path.basename(file_path)}:{qual_prefix}{node.name}"

                params = [arg.arg for arg in node.args.args]
                if is_method and params and params[0] in ("self", "cls"):
                    params = params[1:]

                sym = FunctionSymbol(
                    name=node.name,
                    qualname=qualname,
                    file=file_path,
                    node=node,
                    params=params,
                    is_method=is_method,
                    parent_class=parent_class,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno)
                )

                # Inspect inner statements for returns, assignments, calls, and sources
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        sym.returns.append(child)
                    elif isinstance(child, ast.Assign):
                        for tgt in child.targets:
                            if isinstance(tgt, ast.Name):
                                if tgt.id not in sym.assignments:
                                    sym.assignments[tgt.id] = []
                                sym.assignments[tgt.id].append((child.value, child.lineno, child.col_offset))
                    elif isinstance(child, ast.Call):
                        sym.calls.append(child)
                        # Register call edge
                        target_name = self.outer._extract_call_target_name(child)
                        self.outer.call_edges.append(CallEdge(
                            caller_qualname=qualname,
                            caller_file=file_path,
                            call_line=child.lineno,
                            call_node=child,
                            target_name=target_name
                        ))

                self.outer.functions_by_qualname[qualname] = sym
                if node.name not in self.outer.functions_by_name:
                    self.outer.functions_by_name[node.name] = []
                self.outer.functions_by_name[node.name].append(sym)

                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()

        visitor = SymbolVisitor(self)
        visitor.visit(tree)

    def _extract_call_target_name(self, call_node: ast.Call) -> str:
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            val_name = ""
            if isinstance(call_node.func.value, ast.Name):
                val_name = call_node.func.value.id
            elif isinstance(call_node.func.value, ast.Attribute):
                val_name = f"{getattr(call_node.func.value.value, 'id', '')}.{call_node.func.value.attr}"
            if val_name:
                return f"{val_name}.{call_node.func.attr}"
            return call_node.func.attr
        return "UNKNOWN_CALL"

    def _resolve_target(self, caller_file: str, target_name: str, caller_qualname: Optional[str] = None) -> Optional[FunctionSymbol]:
        """Resolves target_name to a FunctionSymbol via local definitions, imports, or classes."""
        base_fn = os.path.basename(caller_file)

        # 1. Local function in the same file
        local_qualname = f"{base_fn}:{target_name}"
        if local_qualname in self.functions_by_qualname:
            return self.functions_by_qualname[local_qualname]

        # 2. Check if target_name is self.method or cls.method within a class
        if caller_qualname and ("." in target_name):
            parts = target_name.split(".")
            if parts[0] in ("self", "cls"):
                caller_sym = self.functions_by_qualname.get(caller_qualname)
                if caller_sym and caller_sym.parent_class:
                    target_method = parts[-1]
                    class_method_qualname = f"{base_fn}:{caller_sym.parent_class}.{target_method}"
                    if class_method_qualname in self.functions_by_qualname:
                        return self.functions_by_qualname[class_method_qualname]
                    # Check base classes (inheritance)
                    for base_cls in self.class_bases.get(caller_sym.parent_class, []):
                        base_method_qualname = f"{base_fn}:{base_cls}.{target_method}"
                        if base_method_qualname in self.functions_by_qualname:
                            return self.functions_by_qualname[base_method_qualname]

        # 3. Check if target_name is imported in caller_file
        imports = self.file_imports.get(caller_file, {})
        if target_name in imports:
            target_mod_path, orig_name = imports[target_name]
            if target_mod_path:
                target_base = os.path.basename(target_mod_path)
                import_qualname = f"{target_base}:{orig_name}"
                if import_qualname in self.functions_by_qualname:
                    return self.functions_by_qualname[import_qualname]

        # 4. Check attribute calls (e.g. module.func or Class.method)
        if "." in target_name:
            parts = target_name.split(".")
            prefix, attr = parts[0], parts[-1]
            if prefix in imports:
                mod_path, _ = imports[prefix]
                if mod_path:
                    target_base = os.path.basename(mod_path)
                    mod_qualname = f"{target_base}:{attr}"
                    if mod_qualname in self.functions_by_qualname:
                        return self.functions_by_qualname[mod_qualname]

            # Check if prefix is a class name
            for qn, sym in self.functions_by_qualname.items():
                if sym.parent_class == prefix and sym.name == attr:
                    return sym

        # 5. Fallback search by function name if unique across project
        candidates = self.functions_by_name.get(target_name, [])
        if len(candidates) == 1:
            return candidates[0]

        return None

# Backward compatibility alias
ProjectSymbolTable = ProjectASTGraph

