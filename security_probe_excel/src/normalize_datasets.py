"""
Normalization pipeline for SQL Injection datasets.
Maps external compatible datasets to the reference Excel schema,
performs deduplication, conflicting label detection, syntax validation,
and generates deduplicated and quarantined corpora.
"""

import os
import sys
import json
import ast
import hashlib
import re
import pandas as pd
from typing import Dict, Any, List, Set, Tuple

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.schema import (
    SQLInjectionRecord,
    compute_band,
    parse_vulnerable_lines,
    VALID_CASE_TYPES,
    VALID_DB_APIS,
    VALID_QUERY_STYLES
)

def hash_code(s: str) -> str:
    """Exact code hash after basic trimming."""
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()

def normalize_whitespace_code(s: str) -> str:
    """Strip comments and normalize internal whitespace for near-duplicate detection."""
    lines = []
    for line in s.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        # Remove trailing inline comment if any
        if "#" in line:
            parts = line.split("#", 1)
            line = parts[0].strip()
        if line:
            lines.append(" ".join(line.split()))
    return "\n".join(lines)

def hash_normalized_code(s: str) -> str:
    norm = normalize_whitespace_code(s)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def detect_db_api(code: str) -> str:
    code_lower = code.lower()
    if "sqlite3" in code_lower:
        return "sqlite3"
    elif "psycopg2" in code_lower or "pg8000" in code_lower:
        return "psycopg2"
    elif "sqlalchemy" in code_lower:
        return "sqlalchemy"
    elif "mysql" in code_lower or "pymysql" in code_lower:
        return "mysql"
    elif "django.db" in code_lower or "models." in code_lower:
        return "django_orm"
    return "other"

def detect_query_style(code: str) -> str:
    # f-string: f"..." or f'...'
    if re.search(r'f["\'].*?\{.*?\}["\']', code):
        return "f_string"
    # % format: %s or % (
    if re.search(r'%\s*\(|%s', code):
        return "percent_format"
    # format method: .format(
    if ".format(" in code:
        return "format_method"
    # concatenation: + inside execute
    if re.search(r'execute\s*\(.*?(\+|\+=).*?\)', code, re.DOTALL):
        return "concatenation"
    # orm: .filter( or .raw( or .objects.
    if re.search(r'\.(filter|raw|all|get|exclude)\(', code):
        return "orm"
    # parameterised: , ( or , [ or , params or , tuple
    if re.search(r'execute\s*\([^,]+,\s*[\(\[\{]', code):
        return "parameterised"
    return "other"

def extract_sink_name(code: str) -> str:
    match = re.search(r'(\b\w+\.(?:execute|executemany|raw|filter|query|fetch|fetchall|fetchone)\b)', code)
    if match:
        return match.group(1)
    if "execute" in code:
        return "execute"
    return None

def compute_diff_lines(code_vuln: str, code_safe: str) -> List[int]:
    """Identify line numbers in code_vuln that differ from code_safe."""
    v_lines = code_vuln.splitlines()
    s_lines = code_safe.splitlines()
    diff = []
    for i, line in enumerate(v_lines):
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            continue
        # Check if line exists in safe version
        if line not in s_lines:
            diff.append(i + 1)
    return diff

def load_excel_reference(excel_path: str) -> List[SQLInjectionRecord]:
    df = pd.read_excel(excel_path, sheet_name="rows_1000")
    records = []
    for _, row in df.iterrows():
        rec_dict = row.to_dict()
        rec_dict["dataset_source"] = "sql_injection_train_1000.xlsx"
        rec_dict["original_sample_id"] = str(row["sample_id"])
        record = SQLInjectionRecord.from_dict(rec_dict)
        records.append(record)
    return records

def load_expanded_3k_cwe89(json_path: str) -> List[SQLInjectionRecord]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data.get("tasks", [])
    sql_tasks = [t for t in tasks if "89" in str(t.get("cwe", "")) or "sql" in str(t.get("cwe_name", "")).lower()]
    
    records = []
    for t in sql_tasks:
        tid = t.get("id", "cwe89_unknown")
        prompt = t.get("prompt", "")
        vuln_comp = t.get("vulnerable_completion", "")
        safe_comp = t.get("safe_completion", "")
        
        code_vuln = (prompt + "\n" + vuln_comp).strip()
        code_safe = (prompt + "\n" + safe_comp).strip()
        
        db_api = detect_db_api(code_vuln)
        q_style_vuln = detect_query_style(code_vuln)
        q_style_safe = detect_query_style(code_safe)
        sink_vuln = extract_sink_name(code_vuln)
        sink_safe = extract_sink_name(code_safe)
        
        vuln_lines = compute_diff_lines(code_vuln, code_safe)
        if not vuln_lines:
            # Fallback to lines in vulnerable completion
            prompt_len = len(prompt.splitlines())
            vuln_lines = list(range(prompt_len + 1, len(code_vuln.splitlines()) + 1))
            
        # Vulnerable sample
        rec_vuln = SQLInjectionRecord(
            sample_id=f"ext_3k_{tid}_vuln",
            code=code_vuln,
            label=1,
            case_type="normal",
            task_description=f"CWE-89: {t.get('cwe_name', 'SQL Injection')} - {prompt[:80]}",
            db_api=db_api,
            n_lines=len(code_vuln.splitlines()),
            band=compute_band(len(code_vuln.splitlines())),
            n_tokens=len(code_vuln.split()),
            query_style=q_style_vuln,
            sink=sink_vuln,
            source="function_argument",
            fixed_code=code_safe,
            vulnerable_lines=vuln_lines,
            fix_type="parameterised_query",
            dataset_source="expanded_evaluation_3k",
            original_sample_id=tid
        )
        records.append(rec_vuln)
        
        # Safe sample
        rec_safe = SQLInjectionRecord(
            sample_id=f"ext_3k_{tid}_safe",
            code=code_safe,
            label=0,
            case_type="safe_db",
            task_description=f"CWE-89 Safe: {t.get('cwe_name', 'SQL Injection')} - {prompt[:80]}",
            db_api=db_api,
            n_lines=len(code_safe.splitlines()),
            band=compute_band(len(code_safe.splitlines())),
            n_tokens=len(code_safe.split()),
            query_style=q_style_safe if q_style_safe != "other" else "parameterised",
            sink=sink_safe,
            source=None,
            fixed_code=None,
            vulnerable_lines=[],
            fix_type=None,
            dataset_source="expanded_evaluation_3k",
            original_sample_id=tid
        )
        records.append(rec_safe)
        
    return records

def load_validated_sql_dataset(jsonl_path: str) -> List[SQLInjectionRecord]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sid = item.get("sample_id", "")
            code = item.get("code", "")
            lbl = int(item.get("label", 0))
            
            db_api = detect_db_api(code)
            q_style = detect_query_style(code)
            sink = extract_sink_name(code)
            v_lines = parse_vulnerable_lines(item.get("vulnerable_lines"))
            if lbl == 0:
                v_lines = []
                
            fixed_c = item.get("fixed_code")
            if fixed_c and not str(fixed_c).strip():
                fixed_c = None
                
            case_t = "normal" if lbl == 1 else "safe_db"
            
            rec = SQLInjectionRecord(
                sample_id=f"ext_val_{sid}",
                code=code,
                label=lbl,
                case_type=case_t,
                task_description=f"SQL sample from {item.get('repository', 'repo')}",
                db_api=db_api,
                n_lines=len(code.splitlines()),
                band=compute_band(len(code.splitlines())),
                n_tokens=len(code.split()),
                query_style=q_style,
                sink=sink,
                source="function_argument" if lbl == 1 else None,
                fixed_code=fixed_c,
                vulnerable_lines=v_lines,
                fix_type="parameterised_query" if fixed_c else None,
                dataset_source=item.get("source_dataset", "validated_sql_dataset"),
                original_sample_id=sid
            )
            records.append(rec)
    return records

def run_integration_pipeline():
    base_dir = r"C:\Users\dheer\Desktop\isae"
    excel_path = os.path.join(base_dir, "sql_injection_train_1000.xlsx")
    d3k_path = os.path.join(base_dir, "data", "processed", "expanded_evaluation_3k.json")
    val_path = os.path.join(base_dir, "security_probe_sql", "data", "processed", "validated_sql_dataset.jsonl")
    
    out_dir = os.path.join(base_dir, "security_probe_excel", "data")
    norm_dir = os.path.join(out_dir, "normalized")
    dedup_dir = os.path.join(out_dir, "deduplicated")
    quar_dir = os.path.join(out_dir, "quarantined")
    manifest_dir = os.path.join(out_dir, "manifests")
    
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(dedup_dir, exist_ok=True)
    os.makedirs(quar_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    print("Loading datasets...")
    excel_recs = load_excel_reference(excel_path)
    print(f"Loaded {len(excel_recs)} records from Excel reference.")
    
    d3k_recs = load_expanded_3k_cwe89(d3k_path)
    print(f"Loaded {len(d3k_recs)} records from expanded_evaluation_3k (CWE-89).")
    
    val_recs = load_validated_sql_dataset(val_path)
    print(f"Loaded {len(val_recs)} records from validated_sql_dataset.")
    
    # Save normalized sets
    all_raw_candidates = excel_recs + d3k_recs + val_recs
    print(f"Total raw candidates combined: {len(all_raw_candidates)}")
    
    # Validation & Deduplication Phase
    seen_exact_hashes: Dict[str, SQLInjectionRecord] = {}
    seen_norm_hashes: Dict[str, SQLInjectionRecord] = {}
    
    deduplicated_records: List[SQLInjectionRecord] = []
    quarantined_records: List[Dict[str, Any]] = []
    
    counts = {
        "total_input": len(all_raw_candidates),
        "excel_reference_count": len(excel_recs),
        "syntax_error_quarantined": 0,
        "conflicting_label_quarantined": 0,
        "exact_duplicate_skipped": 0,
        "near_duplicate_skipped": 0,
        "deduplicated_final": 0,
        "final_vulnerable": 0,
        "final_safe": 0
    }
    
    # Process Excel reference records FIRST to ensure they have highest priority
    ordered_candidates = excel_recs + d3k_recs + val_recs
    
    for rec in ordered_candidates:
        # 1. Syntax check
        validation_errors = rec.validate()
        syntax_errs = [e for e in validation_errors if "syntax" in e.lower() or "invalid python" in e.lower()]
        if syntax_errs:
            quarantined_records.append({
                "sample_id": rec.sample_id,
                "dataset_source": rec.dataset_source,
                "reason": "syntax_error",
                "details": syntax_errs,
                "code": rec.code
            })
            counts["syntax_error_quarantined"] += 1
            continue
            
        exact_h = hash_code(rec.code)
        norm_h = hash_normalized_code(rec.code)
        
        # 2. Check for conflicting label on identical code
        if exact_h in seen_exact_hashes:
            existing = seen_exact_hashes[exact_h]
            if existing.label != rec.label:
                quarantined_records.append({
                    "sample_id": rec.sample_id,
                    "dataset_source": rec.dataset_source,
                    "conflicting_sample_id": existing.sample_id,
                    "reason": "conflicting_label",
                    "existing_label": existing.label,
                    "new_label": rec.label,
                    "code": rec.code
                })
                counts["conflicting_label_quarantined"] += 1
                continue
            else:
                # Exact duplicate with same label
                counts["exact_duplicate_skipped"] += 1
                continue
                
        # 3. Near-duplicate check
        if norm_h in seen_norm_hashes:
            existing = seen_norm_hashes[norm_h]
            if existing.label != rec.label:
                quarantined_records.append({
                    "sample_id": rec.sample_id,
                    "dataset_source": rec.dataset_source,
                    "conflicting_sample_id": existing.sample_id,
                    "reason": "near_duplicate_conflicting_label",
                    "existing_label": existing.label,
                    "new_label": rec.label,
                    "code": rec.code
                })
                counts["conflicting_label_quarantined"] += 1
                continue
            else:
                # Near duplicate with same label
                counts["near_duplicate_skipped"] += 1
                continue
                
        # Clean unique record
        seen_exact_hashes[exact_h] = rec
        seen_norm_hashes[norm_h] = rec
        deduplicated_records.append(rec)
        if rec.label == 1:
            counts["final_vulnerable"] += 1
        else:
            counts["final_safe"] += 1
            
    counts["deduplicated_final"] = len(deduplicated_records)
    
    print("\n=== PIPELINE RESULTS ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
        
    # Write normalized corpus
    norm_path = os.path.join(norm_dir, "all_normalized_corpus.jsonl")
    with open(norm_path, "w", encoding="utf-8") as f:
        for r in all_raw_candidates:
            f.write(json.dumps(r.to_dict()) + "\n")
            
    # Write deduplicated corpus
    dedup_path = os.path.join(dedup_dir, "deduplicated_corpus.jsonl")
    with open(dedup_path, "w", encoding="utf-8") as f:
        for r in deduplicated_records:
            f.write(json.dumps(r.to_dict()) + "\n")
            
    # Write quarantined records
    quar_path = os.path.join(quar_dir, "quarantined_samples.jsonl")
    with open(quar_path, "w", encoding="utf-8") as f:
        for q in quarantined_records:
            f.write(json.dumps(q) + "\n")
            
    # Write manifest
    manifest_path = os.path.join(manifest_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
        
    print(f"\nDeduplicated corpus saved to: {dedup_path}")
    print(f"Quarantined records saved to: {quar_path}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    run_integration_pipeline()

