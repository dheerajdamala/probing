"""
Generate sample inference requests and structured responses covering diverse code patterns:
1. Vulnerable f-string query
2. Safe parameterized query
3. Vulnerable concatenation
4. Safe ORM query
5. Safe non-SQL code
6. Tainted helper function
"""

import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.inference import SQLInjectionInferenceProbe

def generate_samples():
    base_dir = r"C:\Users\dheer\Desktop\isae\security_probe_excel"
    samples_dir = os.path.join(base_dir, "outputs", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    probe = SQLInjectionInferenceProbe()
    
    test_cases = [
        {
            "case_id": "inf_01_fstring_vulnerable",
            "category": "f_string_interpolation",
            "description": "Direct f-string SQL query interpolation with untrusted user input.",
            "code": """def get_user_profile(conn, user_id):
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, username, email FROM users WHERE id = {user_id}")
    return cursor.fetchone()"""
        },
        {
            "case_id": "inf_02_parameterized_safe",
            "category": "db_api_parameterized",
            "description": "Safe parameterized tuple binding using %s placeholder.",
            "code": """def get_user_profile(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()"""
        },
        {
            "case_id": "inf_03_concat_vulnerable",
            "category": "string_concatenation",
            "description": "Unescaped string concatenation inside cursor execution.",
            "code": """def delete_project(conn, project_name):
    cur = conn.cursor()
    query = "DELETE FROM projects WHERE name = '" + project_name + "'"
    cur.execute(query)
    conn.commit()"""
        },
        {
            "case_id": "inf_04_django_orm_safe",
            "category": "orm_abstraction",
            "description": "Safe Django ORM query using filter() parameter encapsulation.",
            "code": """def fetch_active_orders(customer_id):
    from myapp.models import Order
    return Order.objects.filter(customer_id=customer_id, status='active')"""
        },
        {
            "case_id": "inf_05_no_sql_safe",
            "category": "non_database_logic",
            "description": "Benign non-SQL Python code performing string parsing.",
            "code": """def calculate_total_tax(items, tax_rate=0.07):
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    return round(subtotal * tax_rate, 2)"""
        },
        {
            "case_id": "inf_06_tainted_helper_vulnerable",
            "category": "inter_procedural_taint",
            "description": "Untrusted input routed through an insecure helper function formatting.",
            "code": """def _build_filter(val):
    return "WHERE status = '%s'" % val

def search_records(conn, status_input):
    clause = _build_filter(status_input)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs " + clause)
    return cursor.fetchall()"""
        }
    ]
    
    results = []
    print("Executing sample inference requests...")
    for tc in test_cases:
        pred_res = probe.predict(tc["code"])
        results.append({
            "case_id": tc["case_id"],
            "category": tc["category"],
            "description": tc["description"],
            "input_code": tc["code"],
            "inference_result": pred_res
        })
        print(f"  [{tc['case_id']}] -> Pred: {pred_res['prediction']}, Prob: {pred_res['vulnerability_probability']:.4f}")
        
    out_path = os.path.join(samples_dir, "sample_inference_requests_responses.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sample inference requests and responses to: {out_path}")

if __name__ == "__main__":
    generate_samples()

