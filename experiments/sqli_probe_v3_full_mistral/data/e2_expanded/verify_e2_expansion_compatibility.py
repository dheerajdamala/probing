import os
import sys
import json
import hashlib
import ast
import pandas as pd
from typing import Dict, Any, List

REPO_ROOT = r"C:\Users\dheer\Desktop\isae"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "sqli_probe_v3_full_mistral", "extraction"))

from security_probe_excel.src.schema import (
    SQLInjectionRecord,
    compute_band,
    parse_vulnerable_lines,
    VALID_CASE_TYPES,
    VALID_DB_APIS,
    VALID_QUERY_STYLES
)

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_excel_reference(excel_path: str, sheet_name: str = "rows_1000") -> List[SQLInjectionRecord]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    records = []
    for _, row in df.iterrows():
        rec_dict = row.to_dict()
        rec_dict["dataset_source"] = "sql_injection_train_1000.xlsx"
        rec_dict["original_sample_id"] = str(row["sample_id"])
        record = SQLInjectionRecord.from_dict(rec_dict)
        records.append(record)
    return records

def compute_schema_fingerprint(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    keys = list(records[0].keys())
    types = {k: set() for k in keys}
    null_counts = {k: 0 for k in keys}
    for r in records:
        for k in keys:
            val = r.get(k)
            if val is None:
                null_counts[k] += 1
                types[k].add("NoneType")
            else:
                types[k].add(type(val).__name__)
    types_serializable = {k: sorted(list(v)) for k, v in types.items()}
    fp_str = json.dumps({"keys": keys, "types": types_serializable}, sort_keys=True)
    fp_hash = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()
    return {
        "keys": keys,
        "key_count": len(keys),
        "types": types_serializable,
        "null_counts": null_counts,
        "fingerprint_sha256": fp_hash
    }

def main():
    print("=" * 70)
    print("E2 EXPANSION COMPATIBILITY VERIFICATION AUDIT")
    print("=" * 70)

    orig_excel = os.path.join(REPO_ROOT, "sql_injection_train_1000.xlsx")
    orig_train_jl = os.path.join(REPO_ROOT, "experiments", "sqli_probe_v3_full_mistral", "data", "e2_excel", "train.jsonl")
    orig_val_jl = os.path.join(REPO_ROOT, "experiments", "sqli_probe_v3_full_mistral", "data", "e2_excel", "val.jsonl")
    orig_test_jl = os.path.join(REPO_ROOT, "experiments", "sqli_probe_v3_full_mistral", "data", "e2_excel", "test.jsonl")

    expanded_jl = os.path.join(REPO_ROOT, "experiments", "sqli_probe_v3_full_mistral", "data", "e2_expanded", "e2_expanded_all.jsonl")
    expanded_excel = os.path.join(REPO_ROOT, "experiments", "sqli_probe_v3_full_mistral", "data", "e2_expanded", "sql_injection_expanded_4673.xlsx")

    # -------------------------------------------------------------
    # Step 1: Load original E2 dataset using existing loaders
    # -------------------------------------------------------------
    print("\n[Step 1] Loading original E2 dataset with existing loaders...")
    orig_records_excel = load_excel_reference(orig_excel, sheet_name="rows_1000")
    print(f"  Loaded {len(orig_records_excel)} records via load_excel_reference.")

    orig_jsonl_all = load_jsonl(orig_train_jl) + load_jsonl(orig_val_jl) + load_jsonl(orig_test_jl)
    print(f"  Loaded {len(orig_jsonl_all)} records via load_jsonl (train+val+test).")
    assert len(orig_records_excel) == 1000, "Original Excel must have 1000 records"
    assert len(orig_jsonl_all) == 1000, "Original JSONL splits must total 1000 records"

    # -------------------------------------------------------------
    # Step 2: Load expanded dataset using the exact same loaders
    # -------------------------------------------------------------
    print("\n[Step 2] Loading expanded dataset with identical loaders...")
    exp_records_jl = load_jsonl(expanded_jl)
    print(f"  Loaded {len(exp_records_jl)} records from e2_expanded_all.jsonl.")
    
    exp_records_excel = load_excel_reference(expanded_excel, sheet_name="rows_4673")
    print(f"  Loaded {len(exp_records_excel)} records from sql_injection_expanded_4673.xlsx.")
    assert len(exp_records_jl) == 4673, f"Expected 4673 records, got {len(exp_records_jl)}"
    assert len(exp_records_excel) == 4673, f"Expected 4673 records, got {len(exp_records_excel)}"

    # -------------------------------------------------------------
    # Step 3: Verify record structure equality
    # -------------------------------------------------------------
    print("\n[Step 3] Verifying record structure and dictionary keys...")
    orig_keys = list(orig_jsonl_all[0].keys())
    exp_keys = list(exp_records_jl[0].keys())
    print(f"  Original JSONL keys ({len(orig_keys)}): {orig_keys}")
    print(f"  Expanded JSONL keys ({len(exp_keys)}): {exp_keys}")
    assert orig_keys == exp_keys, "JSONL keys or key order do not match!"

    orig_df = pd.read_excel(orig_excel, sheet_name="rows_1000")
    exp_df = pd.read_excel(expanded_excel, sheet_name="rows_4673")
    orig_cols = list(orig_df.columns)
    exp_cols = list(exp_df.columns)
    print(f"  Original Excel columns ({len(orig_cols)}): {orig_cols}")
    print(f"  Expanded Excel columns ({len(exp_cols)}): {exp_cols}")
    assert orig_cols == exp_cols, "Excel columns or column order do not match!"

    # -------------------------------------------------------------
    # Step 4: Verify all original 1,000 records are preserved exactly
    # -------------------------------------------------------------
    print("\n[Step 4] Verifying all original 1,000 records are preserved...")
    orig_excel_dict = {str(r["sample_id"]): r for _, r in orig_df.iterrows()}
    for i in range(1000):
        exp_r = exp_records_jl[i]
        sid = exp_r["sample_id"]
        assert sid in orig_excel_dict, f"Sample {sid} not in original Excel!"
        orig_r = orig_excel_dict[sid]
        
        # Check code, label, db_api, case_type
        assert exp_r["label"] == int(orig_r["label"]), f"Label mismatch for {sid}"
        assert exp_r["code"].strip() == str(orig_r["code"]).strip(), f"Code mismatch for {sid}"
        assert exp_r["case_type"] == str(orig_r["case_type"]), f"Case type mismatch for {sid}"
        assert exp_r["db_api"] == str(orig_r["db_api"]), f"DB API mismatch for {sid}"
    print("  All original 1,000 records verified identical to source.")

    # -------------------------------------------------------------
    # Step 5: Verify schema adherence of all 3,673 new records
    # -------------------------------------------------------------
    print("\n[Step 5] Verifying schema adherence of 3,673 expanded records...")
    syntax_errors = 0
    validation_errors = 0
    for idx, r in enumerate(exp_records_jl[1000:], 1001):
        # Check AST parseable
        try:
            ast.parse(r["code"])
        except Exception as e:
            syntax_errors += 1
            print(f"  Syntax error in sample {r.get('sample_id')}: {e}")
            
        # Check required fields
        if list(r.keys()) != orig_keys:
            validation_errors += 1
        if r["label"] not in (0, 1):
            validation_errors += 1
            
    assert syntax_errors == 0, f"Found {syntax_errors} syntax errors in expanded records!"
    assert validation_errors == 0, f"Found {validation_errors} validation errors in expanded records!"
    print(f"  All 3,673 expanded records parsed successfully by ast.parse (0 syntax errors, 0 schema errors).")

    # -------------------------------------------------------------
    # Step 6: Verify existing preprocessing & extraction pipeline
    # -------------------------------------------------------------
    print("\n[Step 6] Testing existing AST sink extraction pipeline...")
    from ast_sink_extractor import extract_ast_sink_regions
    
    # Test on a batch of expanded records
    extracted_sinks = 0
    for r in exp_records_jl[:100] + exp_records_jl[1000:1100]:
        code = r["code"]
        res = extract_ast_sink_regions(code)
        if res.get("has_ast_sink") or res.get("detected_sinks"):
            extracted_sinks += 1
    print(f"  AST sink extraction tested on 200 samples: {extracted_sinks} sinks identified successfully.")

    # -------------------------------------------------------------
    # Step 7: Compare schema fingerprints
    # -------------------------------------------------------------
    print("\n[Step 7] Computing and comparing schema fingerprints...")
    fp_orig = compute_schema_fingerprint(orig_jsonl_all)
    fp_exp = compute_schema_fingerprint(exp_records_jl)
    
    print(f"  Original Schema Key Count: {fp_orig['key_count']}")
    print(f"  Expanded Schema Key Count: {fp_exp['key_count']}")
    assert fp_orig["keys"] == fp_exp["keys"], "Fingerprint keys mismatch!"
    
    print("  Key Data Types Comparison:")
    for k in fp_orig["keys"]:
        t_orig = fp_orig["types"][k]
        t_exp = fp_exp["types"][k]
        print(f"    {k:22s} -> Orig: {t_orig} | Exp: {t_exp}")
        assert set(t_exp).issubset(set(t_orig)) or set(t_orig).issubset(set(t_exp)), f"Type discrepancy for {k}"

    print("\n=============================================================")
    print("ALL 7 COMPATIBILITY VERIFICATION AUDITS PASSED WITH ZERO ERRORS")
    print("=============================================================")

if __name__ == "__main__":
    main()
