"""
Interprocedural Evaluation Benchmark for SQL Injection Probe V3 Extension.
Evaluates detection metrics, sink/source localization, vulnerable line precision/recall,
cross-file resolution rate, and false positive suppression across realistic patterns.
"""

import os
import sys
import time
import json
import shutil
import tempfile
from typing import Dict, List, Any

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXT_DIR, ".."))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline import SQLInjectionAnalysisPipeline
from interprocedural_analyzer import InterproceduralAnalyzer

BENCHMARK_CASES = [
    {
        "id": "CASE_01_DIRECT_PARAM_PROPAGATION",
        "description": "Caller reads input and passes directly to helper SQL sink",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "is_cross_file": False,
        "code": (
            "def exec_query(param):\n"
            "    cursor.execute(f\"SELECT * FROM users WHERE id = '{param}'\")\n"
            "def handle():\n"
            "    u = request.args['id']\n"
            "    exec_query(u)\n"
        )
    },
    {
        "id": "CASE_02_HELPER_RETURN_QUERY",
        "description": "Helper constructs query string, caller executes it",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 6,
        "expected_vuln_lines": [2],
        "is_cross_file": False,
        "code": (
            "def build_sql(term):\n"
            "    return f\"SELECT * FROM items WHERE name = '{term}'\"\n"
            "def run():\n"
            "    t = request.args.get('term')\n"
            "    q = build_sql(t)\n"
            "    db.execute(q)\n"
        )
    },
    {
        "id": "CASE_03_HELPER_RETURN_INPUT",
        "description": "Helper retrieves input source, caller constructs and executes query",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 6,
        "expected_vuln_lines": [5],
        "is_cross_file": False,
        "code": (
            "def get_user_id():\n"
            "    return request.args['uid']\n"
            "def execute_flow():\n"
            "    uid = get_user_id()\n"
            "    query = \"SELECT * FROM profiles WHERE id = '\" + uid + \"'\"\n"
            "    cursor.execute(query)\n"
        )
    },
    {
        "id": "CASE_04_THREE_FUNCTION_CHAIN",
        "description": "3-stage dataflow chain from input to execution",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "is_cross_file": False,
        "code": (
            "def execute_final(q):\n"
            "    cursor.execute(q)\n"
            "def build_middle(val):\n"
            "    sql = f\"SELECT * FROM logs WHERE msg = '{val}'\"\n"
            "    execute_final(sql)\n"
            "def entry():\n"
            "    build_middle(request.form['msg'])\n"
        )
    },
    {
        "id": "CASE_05_SAFE_PARAM_HELPER",
        "description": "Helper safely binds parameter in execute sink",
        "expected_label": "SAFE",
        "expected_sink_line": 2,
        "expected_vuln_lines": [],
        "is_cross_file": False,
        "code": (
            "def execute_safe(user_id):\n"
            "    cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n"
            "def handle():\n"
            "    uid = request.args.get('id')\n"
            "    execute_safe(uid)\n"
        )
    },
    {
        "id": "CASE_06_SANITIZED_INT_HELPER",
        "description": "Helper casts untrusted input to int before query construction",
        "expected_label": "SAFE",
        "expected_sink_line": 6,
        "expected_vuln_lines": [],
        "is_cross_file": False,
        "code": (
            "def parse_num(raw_str):\n"
            "    return int(raw_str)\n"
            "def run():\n"
            "    n = parse_num(request.args['page'])\n"
            "    q = f\"SELECT * FROM pages WHERE page_num = {n}\"\n"
            "    cursor.execute(q)\n"
        )
    },
    {
        "id": "CASE_07_WHITELIST_VALIDATION",
        "description": "Variable validated against an allowed whitelist in caller",
        "expected_label": "SAFE",
        "expected_sink_line": 4,
        "expected_vuln_lines": [],
        "is_cross_file": False,
        "code": (
            "ALLOWED = ['asc', 'desc']\n"
            "def run(order):\n"
            "    if order in ALLOWED:\n"
            "        cursor.execute(f\"SELECT * FROM items ORDER BY id {order}\")\n"
            "def handle():\n"
            "    run(request.args.get('o'))\n"
        )
    },
    {
        "id": "CASE_08_BOOLEAN_CONSTANT",
        "description": "Interpolated variable is a boolean constant",
        "expected_label": "SAFE",
        "expected_sink_line": 3,
        "expected_vuln_lines": [],
        "is_cross_file": False,
        "code": (
            "def run_status():\n"
            "    is_active = True\n"
            "    cursor.execute(f\"SELECT * FROM accounts WHERE active = {is_active}\")\n"
        )
    },
    {
        "id": "CASE_09_CLASS_METHOD_FLOW",
        "description": "Object-oriented service method query execution",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 3,
        "expected_vuln_lines": [5],
        "is_cross_file": False,
        "code": (
            "class UserService:\n"
            "    def exec_raw(self, sql):\n"
            "        cursor.execute(sql)\n"
            "    def get_by_name(self, name):\n"
            "        stmt = f\"SELECT * FROM users WHERE name = '{name}'\"\n"
            "        self.exec_raw(stmt)\n"
            "def api():\n"
            "    svc = UserService()\n"
            "    svc.get_by_name(request.args['u'])\n"
        )
    },
    {
        "id": "CASE_10_DICT_PROPAGATION",
        "description": "Untrusted parameter passed inside dictionary payload",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 2,
        "expected_vuln_lines": [2],
        "is_cross_file": False,
        "code": (
            "def exec_dict(data):\n"
            "    cursor.execute(f\"SELECT * FROM items WHERE tag = '{data['tag']}'\")\n"
            "def handle():\n"
            "    payload = {'tag': request.args['t']}\n"
            "    exec_dict(payload)\n"
        )
    },
    {
        "id": "CASE_11_CROSS_FILE_BUILDER",
        "description": "Query built in external helper file and executed in main",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 5,
        "expected_vuln_lines": [2],
        "is_cross_file": True,
        "files": {
            "query_builder.py": (
                "def make_select(table, val):\n"
                "    return f\"SELECT * FROM {table} WHERE code = '{val}'\"\n"
            ),
            "main.py": (
                "from query_builder import make_select\n"
                "def handler():\n"
                "    val = request.args['code']\n"
                "    sql = make_select('products', val)\n"
                "    cursor.execute(sql)\n"
            )
        }
    },
    {
        "id": "CASE_12_CROSS_FILE_EXECUTOR",
        "description": "Query constructed in main and passed to external db executor",
        "expected_label": "VULNERABLE",
        "expected_sink_line": 2,
        "expected_vuln_lines": [4],
        "is_cross_file": True,
        "files": {
            "db_ops.py": (
                "def run_sql(stmt):\n"
                "    cursor.execute(stmt)\n"
            ),
            "main.py": (
                "from db_ops import run_sql\n"
                "def handler():\n"
                "    u = request.args['username']\n"
                "    query = \"SELECT * FROM staff WHERE user = '\" + u + \"'\"\n"
                "    run_sql(query)\n"
            )
        }
    }
]

def run_evaluation() -> Dict[str, Any]:
    pipeline = SQLInjectionAnalysisPipeline()
    results = []
    
    tp, fp, tn, fn = 0, 0, 0, 0
    sink_hits, sink_total = 0, 0
    vuln_line_tp, vuln_line_fp, vuln_line_fn = 0, 0, 0
    path_complete_count = 0
    cross_file_resolved = 0
    cross_file_total = 0

    start_time = time.time()

    for case in BENCHMARK_CASES:
        cid = case["id"]
        expected_label = case["expected_label"]
        is_cross_file = case["is_cross_file"]

        case_start = time.time()

        if is_cross_file:
            temp_dir = tempfile.mkdtemp(prefix=f"eval_{cid}_")
            cross_file_total += 1
            try:
                for fname, content in case["files"].items():
                    fpath = os.path.join(temp_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(content)
                report = pipeline.analyze_project(temp_dir)
                if report.get("interprocedural_analysis", {}).get("call_edges_resolved", 0) > 0:
                    cross_file_resolved += 1
            finally:
                shutil.rmtree(temp_dir)
        else:
            report = pipeline.analyze(case["code"], filename=f"{cid}.py")

        case_elapsed_ms = round((time.time() - case_start) * 1000, 2)
        pred_label = report["prediction"]

        # Classification metrics
        if expected_label == "VULNERABLE":
            if pred_label == "VULNERABLE":
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == "SAFE":
                tn += 1
            else:
                fp += 1

        # Sink localization
        expected_sink = case["expected_sink_line"]
        sink_total += 1
        detected_sinks = [s["line"] for s in report["sink_location"]]
        if expected_sink in detected_sinks:
            sink_hits += 1

        # Vulnerable line localization
        expected_vuln_lines = set(case["expected_vuln_lines"])
        detected_vuln_lines = {vl["line"] for vl in report["vulnerable_lines"]}

        for l in detected_vuln_lines:
            if l in expected_vuln_lines:
                vuln_line_tp += 1
            else:
                vuln_line_fp += 1
        for l in expected_vuln_lines:
            if l not in detected_vuln_lines:
                vuln_line_fn += 1

        # Path completeness: whether dataflow path links source/propagation to sink
        findings = report.get("findings", [])
        if findings:
            f = findings[0]
            roles = [step["role"] for step in f.get("dataflow_path", [])]
            if "SINK" in roles and ("SOURCE" in roles or "PROPAGATION" in roles or "QUERY_CONSTRUCTION" in roles):
                path_complete_count += 1

        results.append({
            "case_id": cid,
            "description": case["description"],
            "expected_label": expected_label,
            "predicted_label": pred_label,
            "prediction_correct": pred_label == expected_label,
            "confidence": report["confidence"],
            "vulnerability_type": report["vulnerability_type"],
            "sinks_found": detected_sinks,
            "vulnerable_lines_found": list(detected_vuln_lines),
            "expected_vuln_lines": list(expected_vuln_lines),
            "latency_ms": case_elapsed_ms,
            "status": report["analysis_status"]
        })

    total_time_ms = round((time.time() - start_time) * 1000, 2)
    total_samples = len(BENCHMARK_CASES)

    precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
    recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    f1 = round(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0, 4)
    accuracy = round((tp + tn) / total_samples, 4)

    sink_precision = round(sink_hits / sink_total if sink_total > 0 else 0.0, 4)
    vuln_line_prec = round(vuln_line_tp / (vuln_line_tp + vuln_line_fp) if (vuln_line_tp + vuln_line_fp) > 0 else 0.0, 4)
    vuln_line_rec = round(vuln_line_tp / (vuln_line_tp + vuln_line_fn) if (vuln_line_tp + vuln_line_fn) > 0 else 0.0, 4)
    vuln_line_f1 = round(2 * vuln_line_prec * vuln_line_rec / (vuln_line_prec + vuln_line_rec) if (vuln_line_prec + vuln_line_rec) > 0 else 0.0, 4)

    summary = {
        "evaluation_name": "Probe V3 Interprocedural Dataflow Analysis Evaluation",
        "benchmark_sample_count": total_samples,
        "total_elapsed_ms": total_time_ms,
        "mean_latency_per_sample_ms": round(total_time_ms / total_samples, 2),
        "classification_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
        },
        "localization_metrics": {
            "sink_localization_precision": sink_precision,
            "vulnerable_line_precision": vuln_line_prec,
            "vulnerable_line_recall": vuln_line_rec,
            "vulnerable_line_f1": vuln_line_f1
        },
        "dataflow_metrics": {
            "path_completeness_rate": round(path_complete_count / total_samples, 4),
            "cross_file_resolution_rate": round(cross_file_resolved / cross_file_total if cross_file_total > 0 else 1.0, 4)
        },
        "results": results
    }

    out_path = os.path.join(EXT_DIR, "evaluation", "interprocedural_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    run_evaluation()

