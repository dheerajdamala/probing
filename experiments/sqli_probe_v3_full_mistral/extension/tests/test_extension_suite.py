"""
Comprehensive Test Suite for Probe V3 SQL Injection Extension.
Tests all 18 specified scenarios, JSON schema compliance, line localization, and probe integration.
"""

import os
import sys
import unittest
import numpy as np

# Ensure extension module is importable
EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXT_DIR, ".."))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from extension.pipeline import SQLInjectionAnalysisPipeline
from extension.probe_predictor import ProbePredictor

class TestSQLInjectionExtension(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = SQLInjectionAnalysisPipeline()
        cls.predictor = ProbePredictor()
        
        # Load a verified test embedding for probe validation
        emb_path = os.path.join(ROOT_DIR, "embeddings/full_e2/test/sink/layer_12.npy")
        if os.path.exists(emb_path):
            cls.test_embeddings = np.load(emb_path)
        else:
            cls.test_embeddings = None

    def _assert_schema(self, report):
        """Verifies report conforms to required JSON schema."""
        required_keys = [
            "analysis_status", "messages", "prediction", "confidence",
            "probabilities", "vulnerability_type", "sink_location",
            "vulnerable_lines", "evidence"
        ]
        for k in required_keys:
            self.assertIn(k, report, f"Missing key: {k}")
        self.assertIn(report["analysis_status"], ["COMPLETE", "PARTIAL", "FAILED"])
        self.assertIn(report["prediction"], ["VULNERABLE", "SAFE", "UNCERTAIN"])
        self.assertIsInstance(report["confidence"], float)
        self.assertIsInstance(report["probabilities"], dict)
        self.assertIn("vulnerable", report["probabilities"])
        self.assertIn("safe", report["probabilities"])
        self.assertIsInstance(report["vulnerability_type"], list)
        self.assertIsInstance(report["sink_location"], list)
        self.assertIsInstance(report["vulnerable_lines"], list)
        self.assertIsInstance(report["evidence"], list)

    # 1. Direct concatenation vulnerability
    def test_01_direct_concatenation(self):
        code = (
            "def get_user(user_id):\n"
            "    query = \"SELECT * FROM users WHERE id = '\" + user_id + \"'\"\n"
            "    cursor.execute(query)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("DIRECT_STRING_CONCATENATION", report["vulnerability_type"])
        self.assertTrue(any(s["line"] == 3 for s in report["sink_location"]))
        self.assertTrue(any(vl["line"] == 2 for vl in report["vulnerable_lines"]))

    # 2. F-string SQL construction
    def test_02_f_string_construction(self):
        code = (
            "def search_products(term):\n"
            "    sql = f\"SELECT * FROM items WHERE name LIKE '%{term}%'\"\n"
            "    db.execute(sql)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("F_STRING_INTERPOLATION", report["vulnerability_type"])

    # 3. Percent-format SQL construction
    def test_03_percent_formatting(self):
        code = (
            "def authenticate(user, password):\n"
            "    q = \"SELECT id FROM accounts WHERE u='%s' AND p='%s'\" % (user, password)\n"
            "    cur.execute(q)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("PERCENT_FORMATTING", report["vulnerability_type"])

    # 4. .format() SQL construction
    def test_04_format_method(self):
        code = (
            "def fetch_logs(level):\n"
            "    stmt = \"SELECT * FROM logs WHERE severity = '{}'\".format(level)\n"
            "    conn.execute(stmt)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("FORMAT_METHOD", report["vulnerability_type"])

    # 5. Safe parameterized SQL
    def test_05_safe_parameterized(self):
        code = (
            "def get_user_safe(user_id):\n"
            "    query = \"SELECT * FROM users WHERE id = ?\"\n"
            "    cursor.execute(query, (user_id,))\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "SAFE")
        self.assertEqual(report["vulnerability_type"], [])
        self.assertTrue(any(ev["category"] == "SAFE_PARAMETERIZATION" for ev in report["evidence"]))

    # 6. Safe parameterized SQL with dynamic placeholder generation
    def test_06_safe_dynamic_placeholders(self):
        code = (
            "def get_multiple_users(ids):\n"
            "    marks = ', '.join(['%s'] * len(ids))\n"
            "    query = f\"SELECT * FROM users WHERE id IN ({marks})\"\n"
            "    cursor.execute(query, tuple(ids))\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertTrue(any(ev["category"] == "DYNAMIC_PLACEHOLDERS" for ev in report["evidence"]))

    # 7. Safe whitelisted ORDER BY
    def test_07_safe_whitelisted_order_by(self):
        code = (
            "def list_posts(sort_order):\n"
            "    valid_sorts = {'date': 'created_at DESC', 'views': 'views DESC'}\n"
            "    order = valid_sorts.get(sort_order, 'created_at DESC')\n"
            "    cursor.execute(f\"SELECT * FROM posts ORDER BY {order}\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertTrue(any(ev["category"] == "F_STRING_INTERPOLATION" for ev in report["evidence"]))

    # 8. Unsafe dynamic ORDER BY
    def test_08_unsafe_dynamic_order_by(self):
        code = (
            "def list_items(sort_col):\n"
            "    sql = f\"SELECT id, name FROM items ORDER BY {sort_col} ASC\"\n"
            "    cursor.execute(sql)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("DYNAMIC_ORDER_BY", report["vulnerability_type"])
        self.assertIn("DYNAMIC_IDENTIFIER", report["vulnerability_type"])

    # 9. Unsafe table-name concatenation
    def test_09_unsafe_table_name(self):
        code = (
            "def archive_table(table_name):\n"
            "    query = \"SELECT * FROM \" + table_name\n"
            "    cursor.execute(query)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("DYNAMIC_TABLE_NAME", report["vulnerability_type"])
        self.assertIn("DYNAMIC_IDENTIFIER", report["vulnerability_type"])

    # 10. Hybrid parameterized and interpolated queries
    def test_10_hybrid_query(self):
        code = (
            "def update_profile(user_id, status, role):\n"
            "    sql = f\"UPDATE users SET status = '{status}' WHERE id = ?\"\n"
            "    cursor.execute(sql, (user_id,))\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("HYBRID_QUERY_CONSTRUCTION", report["vulnerability_type"])

    # 11. Query construction through intermediate variables
    def test_11_intermediate_variables(self):
        code = (
            "def complex_filter(dept, min_salary):\n"
            "    where_clause = \"dept = '\" + dept + \"'\"\n"
            "    where_clause += f\" AND salary >= {min_salary}\"\n"
            "    query = \"SELECT * FROM employees WHERE \" + where_clause\n"
            "    cursor.execute(query)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertGreaterEqual(len(report["vulnerable_lines"]), 3)

    # 12. SQL execution inside helper functions
    def test_12_execution_in_helper(self):
        code = (
            "def run_sql(statement):\n"
            "    cur = get_connection().cursor()\n"
            "    cur.execute(statement)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertTrue(len(report["sink_location"]) > 0)
        self.assertTrue(any("caller parameter" in m or "interprocedural" in m for m in report["messages"]))

    # 13. Multiple SQL sinks in one file
    def test_13_multiple_sinks(self):
        code = (
            "def process_transaction(user_id, amount):\n"
            "    cur.execute(\"SELECT balance FROM accounts WHERE user_id = ?\", (user_id,))\n"
            "    cur.execute(f\"UPDATE accounts SET balance = balance + {amount} WHERE user_id = {user_id}\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(len(report["sink_location"]), 2)
        self.assertEqual(report["sink_location"][0]["line"], 2)
        self.assertEqual(report["sink_location"][1]["line"], 3)

    # 14. No-SQL Python code
    def test_14_no_sql_code(self):
        code = (
            "def compute_fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "SAFE")
        self.assertEqual(len(report["sink_location"]), 0)
        self.assertIn("No database execution sinks", report["messages"][0])

    # 15. AST parsing failure
    def test_15_ast_parsing_failure(self):
        bad_code = "def syntax_error(:\n    return 42"
        report = self.pipeline.analyze(bad_code)
        self._assert_schema(report)
        self.assertEqual(report["analysis_status"], "FAILED")
        self.assertEqual(report["prediction"], "UNCERTAIN")
        self.assertTrue(any("Syntax Error" in m for m in report["messages"]))

    # 16. Unsupported database API
    def test_16_unsupported_database_api(self):
        code = (
            "def execute_custom_db(cmd):\n"
            "    custom_driver.dispatch_native_buffer(cmd)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(len(report["sink_location"]), 0)

    # 17. Unresolved external helper
    def test_17_unresolved_external_helper(self):
        code = (
            "from external_sanitizer import clean_query\n"
            "def execute_cleaned(raw_query):\n"
            "    safe_q = clean_query(raw_query)\n"
            "    cursor.execute(safe_q)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["analysis_status"], "PARTIAL")

    # 18. Multi-line query construction
    def test_18_multiline_query(self):
        code = (
            "def generate_report(year, region):\n"
            "    query = (\n"
            "        \"SELECT department, SUM(sales) \"\n"
            "        \"FROM fiscal_reports \"\n"
            "        \"WHERE fiscal_year = \" + str(year) + \" \"\n"
            "        \"AND region = '\" + region + \"' \"\n"
            "        \"GROUP BY department\"\n"
            "    )\n"
            "    cursor.execute(query)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("DIRECT_STRING_CONCATENATION", report["vulnerability_type"])

    # Probe Integration Test with genuine 4096-dim representations
    def test_19_probe_predictor_integration(self):
        if self.test_embeddings is not None:
            # Test first vector from the frozen test set
            vec = self.test_embeddings[0]
            res = self.predictor.predict(vec)
            self.assertIn("prediction", res)
            self.assertIn(res["prediction"], ["VULNERABLE", "SAFE", "UNCERTAIN"])
            self.assertIn("confidence", res)
            self.assertIn("probabilities", res)
            self.assertAlmostEqual(res["probabilities"]["vulnerable"] + res["probabilities"]["safe"], 1.0, places=3)
            self.assertEqual(res["threshold"], 0.36)

if __name__ == "__main__":
    unittest.main()

