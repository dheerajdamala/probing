"""
Schema definition and record validation for the SQL Injection Security Probe.
Treats sql_injection_train_1000.xlsx as the reference authoritative schema.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Union, Any, Dict
import json
import ast
import re

# Reference categories from sql_injection_train_1000.xlsx
VALID_CASE_TYPES = {"normal", "safe_db", "hard_positive", "hard_negative", "no_sql"}
VALID_DB_APIS = {"sqlite3", "psycopg2", "sqlalchemy", "mysql", "django_orm", "other"}
VALID_QUERY_STYLES = {
    "concatenation", "parameterised", "f_string", "percent_format",
    "format_method", "orm", "constant", "other"
}
VALID_SOURCES = {"function_argument", "web_request", "file", "environment", "other"}
VALID_FIX_TYPES = {
    "parameterised_query", "orm_rewrite", "allow_list", "type_conversion", "other"
}
VALID_BANDS = {"1-5", "6-10", "11-20", "21-40", "41-100"}

def compute_band(n_lines: int) -> str:
    if n_lines <= 5:
        return "1-5"
    elif n_lines <= 10:
        return "6-10"
    elif n_lines <= 20:
        return "11-20"
    elif n_lines <= 40:
        return "21-40"
    else:
        return "41-100"

def parse_vulnerable_lines(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, str):
        v = v.strip()
        if not v or v == "[]":
            return []
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except Exception:
            pass
        # Fallback regex for comma-separated integers
        nums = re.findall(r"\d+", v)
        return [int(x) for x in nums]
    return []

@dataclass
class SQLInjectionRecord:
    # 1. Primary identification and labels
    sample_id: str
    code: str
    label: int  # 1 = vulnerable, 0 = safe
    
    # 2. Case categorization and task context
    case_type: str  # normal, safe_db, hard_positive, hard_negative, no_sql
    task_description: str
    db_api: str     # sqlite3, psycopg2, sqlalchemy, mysql, django_orm, other
    
    # 3. Code length and token metrics
    n_lines: int
    band: str       # 1-5, 6-10, 11-20, 21-40, 41-100
    n_tokens: int
    
    # 4. Stylistic and structural attributes
    query_style: str  # concatenation, parameterised, f_string, percent_format, format_method, orm, constant, other
    sink: Optional[str] = None
    source: Optional[str] = None
    
    # 5. Remediation and localization supervision
    fixed_code: Optional[str] = None
    vulnerable_lines: List[int] = field(default_factory=list)
    fix_type: Optional[str] = None
    
    # 6. Provenance and traceability
    dataset_source: str = "sql_injection_train_1000.xlsx"
    original_sample_id: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate record against reference schema rules."""
        errors = []
        if not self.sample_id:
            errors.append("Missing sample_id")
        if not self.code or not self.code.strip():
            errors.append("Empty code")
        else:
            try:
                ast.parse(self.code)
            except SyntaxError as e:
                errors.append(f"Invalid Python code syntax: {e}")
                
        if self.label not in (0, 1):
            errors.append(f"Invalid label: {self.label}")
            
        if self.case_type not in VALID_CASE_TYPES:
            errors.append(f"Unknown case_type: {self.case_type}")
            
        if self.label == 0:
            if self.case_type not in {"safe_db", "hard_negative", "no_sql"}:
                errors.append(f"Safe sample has unexpected case_type: {self.case_type}")
            if self.vulnerable_lines:
                errors.append(f"Safe sample has non-empty vulnerable_lines: {self.vulnerable_lines}")
        elif self.label == 1:
            if self.case_type not in {"normal", "hard_positive"}:
                errors.append(f"Vulnerable sample has unexpected case_type: {self.case_type}")
                
        if self.fixed_code:
            try:
                ast.parse(self.fixed_code)
            except SyntaxError as e:
                errors.append(f"Invalid Python fixed_code syntax: {e}")
                
        return errors
        
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["vulnerable_lines_str"] = json.dumps(self.vulnerable_lines)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SQLInjectionRecord":
        v_lines = parse_vulnerable_lines(d.get("vulnerable_lines"))
        code = str(d.get("code", ""))
        n_lines = d.get("n_lines")
        if n_lines is None or n_lines <= 0:
            n_lines = len(code.splitlines()) if code else 1
            
        band = d.get("band")
        if not band or band not in VALID_BANDS:
            band = compute_band(n_lines)
            
        n_tokens = d.get("n_tokens")
        if n_tokens is None or n_tokens <= 0:
            n_tokens = len(code.split())
            
        fixed_c = d.get("fixed_code")
        if fixed_c is not None and (pd_is_nan(fixed_c) or not str(fixed_c).strip()):
            fixed_c = None
            
        sink = d.get("sink")
        if sink is not None and (pd_is_nan(sink) or not str(sink).strip()):
            sink = None
            
        src = d.get("source")
        if src is not None and (pd_is_nan(src) or not str(src).strip()):
            src = None
            
        f_type = d.get("fix_type")
        if f_type is not None and (pd_is_nan(f_type) or not str(f_type).strip()):
            f_type = None

        return cls(
            sample_id=str(d.get("sample_id", "")),
            code=code,
            label=int(d.get("label", 0)),
            case_type=str(d.get("case_type", "normal" if d.get("label") == 1 else "safe_db")),
            task_description=str(d.get("task_description", "")),
            db_api=str(d.get("db_api", "other")),
            n_lines=int(n_lines),
            band=str(band),
            n_tokens=int(n_tokens),
            query_style=str(d.get("query_style", "other")),
            sink=str(sink) if sink else None,
            source=str(src) if src else None,
            fixed_code=str(fixed_c) if fixed_c else None,
            vulnerable_lines=v_lines,
            fix_type=str(f_type) if f_type else None,
            dataset_source=str(d.get("dataset_source", "sql_injection_train_1000.xlsx")),
            original_sample_id=str(d.get("original_sample_id", d.get("sample_id", "")))
        )

def pd_is_nan(val: Any) -> bool:
    if val is None:
        return True
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return True
    except Exception:
        pass
    if str(val).lower() in ("nan", "none", "null"):
        return True
    return False

