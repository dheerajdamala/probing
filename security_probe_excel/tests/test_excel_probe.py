"""
Unit and integration tests for the Excel-compatible SQL Injection Probe experiment.
"""

import os
import sys
import json
import pytest
import pandas as pd

base_dir = r"C:\Users\dheer\Desktop\isae\security_probe_excel"
repo_root = r"C:\Users\dheer\Desktop\isae"
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from security_probe_excel.src.schema import SQLInjectionRecord, parse_vulnerable_lines, compute_band

def test_excel_file_exists_and_unmodified():
    excel_path = os.path.join(repo_root, "sql_injection_train_1000.xlsx")
    assert os.path.exists(excel_path), "Reference Excel file not found!"
    size = os.path.getsize(excel_path)
    assert size > 300000, f"Unexpected Excel size: {size}"
    df = pd.read_excel(excel_path, sheet_name="rows_1000")
    assert len(df) == 1000, f"Expected 1000 rows, found {len(df)}"
    assert "code" in df.columns and "label" in df.columns

def test_schema_validation():
    valid_rec = SQLInjectionRecord(
        sample_id="test_01",
        code="def foo(): pass",
        label=0,
        case_type="safe_db",
        task_description="test task",
        db_api="sqlite3",
        n_lines=1,
        band="1-5",
        n_tokens=4,
        query_style="parameterised"
    )
    errs = valid_rec.validate()
    assert len(errs) == 0, f"Validation errors on valid record: {errs}"

    d = valid_rec.to_dict()
    assert d["sample_id"] == "test_01"
    reconstructed = SQLInjectionRecord.from_dict(d)
    assert reconstructed.sample_id == valid_rec.sample_id
    assert reconstructed.label == valid_rec.label

def test_normalized_and_deduplicated_data_exists():
    dedup_path = os.path.join(base_dir, "data", "deduplicated", "deduplicated_corpus.jsonl")
    assert os.path.exists(dedup_path), "Deduplicated corpus file not found!"
    count = 0
    with open(dedup_path, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
    assert count >= 4000, f"Expected >= 4000 deduplicated records, found {count}"

def test_split_integrity_zero_leakage():
    splits_dir = os.path.join(base_dir, "data", "splits")
    train_f = os.path.join(splits_dir, "train.jsonl")
    val_f = os.path.join(splits_dir, "val.jsonl")
    test_f = os.path.join(splits_dir, "test.jsonl")
    
    assert os.path.exists(train_f) and os.path.exists(val_f) and os.path.exists(test_f)
    
    train_ids = set()
    for l in open(train_f, "r", encoding="utf-8"):
        train_ids.add(json.loads(l)["sample_id"])
    val_ids = set()
    for l in open(val_f, "r", encoding="utf-8"):
        val_ids.add(json.loads(l)["sample_id"])
    test_ids = set()
    for l in open(test_f, "r", encoding="utf-8"):
        test_ids.add(json.loads(l)["sample_id"])
        
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test"

def test_inference_pipeline():
    from security_probe_excel.src.inference import SQLInjectionInferenceProbe
    probe = SQLInjectionInferenceProbe()
    
    # 1. Vulnerable code sample
    code_vuln = '''def query_user(conn, uid):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE id = {uid}")
    return cur.fetchone()
'''
    res_v = probe.predict(code_vuln)
    assert "prediction" in res_v
    assert "vulnerability_probability" in res_v
    assert "suspicious_lines" in res_v
    assert len(res_v["suspicious_lines"]) > 0

    # 2. Empty code sample
    res_empty = probe.predict("")
    assert res_empty["prediction"] == "safe"
    assert res_empty["vulnerability_probability"] == 0.0

def test_checkpoints_and_metrics_exist():
    models_dir = os.path.join(base_dir, "models")
    metrics_path = os.path.join(base_dir, "outputs", "metrics", "EXCEL_PROBE_RESULTS.json")
    
    assert os.path.exists(os.path.join(models_dir, "probe_head_excel_linear.pkl"))
    assert os.path.exists(os.path.join(models_dir, "probe_head_excel_sink.pkl"))
    assert os.path.exists(os.path.join(models_dir, "probe_head_excel_mlp.pkl"))
    assert os.path.exists(os.path.join(models_dir, "tfidf_baseline.pkl"))
    assert os.path.exists(metrics_path)
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "model_evaluations" in data
    assert "linear_probe_sql_sink_pooling" in data["model_evaluations"]
    assert "mlp_probe_sql_sink_pooling" in data["model_evaluations"]
    assert "tfidf_logistic_regression" in data["model_evaluations"]
    assert data["model_evaluations"]["mlp_probe_sql_sink_pooling"]["roc_auc"] > 0.80

