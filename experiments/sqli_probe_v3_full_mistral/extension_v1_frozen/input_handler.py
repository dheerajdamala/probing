"""
Input Handler for Python Source Code.
Handles ingestion, normalization, line mapping, and initial AST syntax validation.
"""

import ast
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ProcessedInput:
    source_code: str
    filename: str
    lines: List[str]
    syntax_valid: bool
    ast_tree: Optional[ast.AST] = None
    syntax_error_message: Optional[str] = None
    syntax_error_line: Optional[int] = None

class InputHandler:
    """Ingests and validates raw Python source code."""

    def __init__(self):
        pass

    def process(self, code: str, filename: Optional[str] = None) -> ProcessedInput:
        """
        Normalize and validate input code.
        
        Args:
            code: Raw Python code string.
            filename: Optional source file name (defaults to 'snippet.py').
            
        Returns:
            ProcessedInput with syntax check results and parsed AST tree if valid.
        """
        fn = filename or "snippet.py"
        
        if code is None:
            code = ""
            
        # Normalize carriage returns
        normalized_code = code.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_code.splitlines()

        if not normalized_code.strip():
            return ProcessedInput(
                source_code=normalized_code,
                filename=fn,
                lines=lines,
                syntax_valid=False,
                ast_tree=None,
                syntax_error_message="Empty source code provided",
                syntax_error_line=0
            )

        try:
            tree = ast.parse(normalized_code, filename=fn)
            return ProcessedInput(
                source_code=normalized_code,
                filename=fn,
                lines=lines,
                syntax_valid=True,
                ast_tree=tree,
                syntax_error_message=None,
                syntax_error_line=None
            )
        except SyntaxError as e:
            return ProcessedInput(
                source_code=normalized_code,
                filename=fn,
                lines=lines,
                syntax_valid=False,
                ast_tree=None,
                syntax_error_message=f"SyntaxError: {e.msg} at line {e.lineno}",
                syntax_error_line=e.lineno
            )
        except Exception as e:
            return ProcessedInput(
                source_code=normalized_code,
                filename=fn,
                lines=lines,
                syntax_valid=False,
                ast_tree=None,
                syntax_error_message=f"ParseError: {str(e)}",
                syntax_error_line=0
            )

