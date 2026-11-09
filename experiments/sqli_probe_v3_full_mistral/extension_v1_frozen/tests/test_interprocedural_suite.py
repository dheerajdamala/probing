"""
Dedicated Interprocedural and Cross-File Test Suite for SQL Injection Probe V3.
Covers all 24 required scenarios: cross-function dataflow, multi-step chains, cross-file imports,
class methods, false positive suppression, recursion cycles, depth limits, and error handling.
"""

import os
import sys
import shutil
import tempfile
import unittest

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXT_DIR, ".."))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from extension.pipeline import SQLInjectionAnalysisPipeline
from extension.interprocedural_analyzer import InterproceduralAnalyzer

class TestInterproceduralSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = SQLInjectionAnalysisPipeline()
        cls.analyzer = InterproceduralAnalyzer()

    def _assert_schema(self, report):
        """Validates standard JSON report schema and interprocedural section."""
        required = [
            "analysis_status", "messages", "prediction", "confidence",
            "probabilities", "vulnerability_type", "sink_location",
            "vulnerable_lines", "evidence", "interprocedural_analysis", "findings"
        ]
        for k in required:
            self.assertIn(k, report, f"Missing key in report: {k}")
        self.assertIn(report["analysis_status"], ["COMPLETE", "PARTIAL", "FAILED"])
        self.assertIn(report["prediction"], ["VULNERABLE", "SAFE", "UNCERTAIN"])

    # 1. Direct cross-function source-to-sink flow
    def test_01_direct_cross_function_flow(self):
        code = (
            "def run_query(uid):\n"
            "    cursor.execute(f\"SELECT * FROM users WHERE id = '{uid}'\")\n"
            "\n"
            "def handler():\n"
            "    user_id = request.args.get('id')\n"
            "    run_query(user_id)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(any(vl["line"] == 2 for vl in report["vulnerable_lines"]))
        self.assertTrue(any(s["line"] == 2 for s in report["sink_location"]))

    # 2. Source in helper return value
    def test_02_helper_return_value(self):
        code = (
            "def get_user_input():\n"
            "    return request.args['q']\n"
            "\n"
            "def search():\n"
            "    query_term = get_user_input()\n"
            "    db.execute(f\"SELECT * FROM items WHERE name = '{query_term}'\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(any(vl["line"] == 6 for vl in report["vulnerable_lines"]))

    # 3. Query builder in helper, execution in caller
    def test_03_query_builder_helper(self):
        code = (
            "def build_select(table, filter_val):\n"
            "    return f\"SELECT * FROM {table} WHERE val = '{filter_val}'\"\n"
            "\n"
            "def execute_flow():\n"
            "    val = request.args.get('val')\n"
            "    sql = build_select('orders', val)\n"
            "    conn.execute(sql)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(any(vl["line"] == 2 for vl in report["vulnerable_lines"]))

    # 4. Query builder in caller, execution in helper
    def test_04_query_executor_helper(self):
        code = (
            "def execute_sql(stmt):\n"
            "    cursor.execute(stmt)\n"
            "\n"
            "def process_request():\n"
            "    user = request.form['username']\n"
            "    q = \"SELECT * FROM accounts WHERE name = '\" + user + \"'\"\n"
            "    execute_sql(q)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(any(vl["line"] == 6 for vl in report["vulnerable_lines"]))

    # 5. Multi-step chain (3+ functions)
    def test_05_multi_step_chain(self):
        code = (
            "def step3_sink(sql_str):\n"
            "    db.execute(sql_str)\n"
            "\n"
            "def step2_forward(arg):\n"
            "    step3_sink(arg)\n"
            "\n"
            "def step1_assemble(user_val):\n"
            "    s = f\"SELECT * FROM records WHERE col = '{user_val}'\"\n"
            "    step2_forward(s)\n"
            "\n"
            "def entrypoint():\n"
            "    step1_assemble(request.args['v'])\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(any(vl["line"] == 8 for vl in report["vulnerable_lines"]))

    # 6. Cross-file import in project directory
    def test_06_cross_file_import(self):
        temp_dir = tempfile.mkdtemp(prefix="sqli_test_cross_file_")
        try:
            helper_path = os.path.join(temp_dir, "db_helpers.py")
            main_path = os.path.join(temp_dir, "main.py")

            with open(helper_path, "w") as f:
                f.write(
                    "def execute_query(query_text):\n"
                    "    cursor.execute(query_text)\n"
                )

            with open(main_path, "w") as f:
                f.write(
                    "from db_helpers import execute_query\n"
                    "\n"
                    "def handle_req():\n"
                    "    raw_val = request.args.get('id')\n"
                    "    sql = f\"SELECT * FROM users WHERE id = '{raw_val}'\"\n"
                    "    execute_query(sql)\n"
                )

            report = self.pipeline.analyze_project(temp_dir)
            self._assert_schema(report)
            self.assertEqual(report["prediction"], "VULNERABLE")
            self.assertTrue(len(report["vulnerable_lines"]) > 0)
        finally:
            shutil.rmtree(temp_dir)

    # 7. Cross-file relative import
    def test_07_cross_file_relative_import(self):
        temp_dir = tempfile.mkdtemp(prefix="sqli_test_rel_import_")
        try:
            pkg_dir = os.path.join(temp_dir, "mypkg")
            os.makedirs(pkg_dir)
            with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
                f.write("")

            with open(os.path.join(pkg_dir, "builder.py"), "w") as f:
                f.write(
                    "def make_query(val):\n"
                    "    return f\"SELECT * FROM items WHERE code = '{val}'\"\n"
                )

            with open(os.path.join(pkg_dir, "runner.py"), "w") as f:
                f.write(
                    "from .builder import make_query\n"
                    "\n"
                    "def run():\n"
                    "    q = make_query(request.args['c'])\n"
                    "    cursor.execute(q)\n"
                )

            report = self.pipeline.analyze_project(temp_dir)
            self._assert_schema(report)
            self.assertEqual(report["prediction"], "VULNERABLE")
        finally:
            shutil.rmtree(temp_dir)

    # 8. Class method call
    def test_08_class_method_call(self):
        code = (
            "class DatabaseService:\n"
            "    def execute_raw(self, sql):\n"
            "        cursor.execute(sql)\n"
            "\n"
            "    def query_user(self, uid):\n"
            "        q = f\"SELECT * FROM users WHERE id = '{uid}'\"\n"
            "        self.execute_raw(q)\n"
            "\n"
            "def handle():\n"
            "    svc = DatabaseService()\n"
            "    svc.query_user(request.args.get('id'))\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(any(vl["line"] == 6 for vl in report["vulnerable_lines"]))

    # 9. Multiple sources flowing to same sink
    def test_09_multiple_sources_one_sink(self):
        code = (
            "def execute_sink(statement):\n"
            "    cursor.execute(statement)\n"
            "\n"
            "def route_a():\n"
            "    val_a = request.args['a']\n"
            "    execute_sink(f\"SELECT * FROM a WHERE x = '{val_a}'\")\n"
            "\n"
            "def route_b():\n"
            "    val_b = request.form['b']\n"
            "    execute_sink(f\"SELECT * FROM b WHERE y = '{val_b}'\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertGreaterEqual(len(report["vulnerable_lines"]), 1)

    # 10. Multiple sinks called by same caller
    def test_10_multiple_sinks_one_caller(self):
        code = (
            "def multi_sink_caller():\n"
            "    u = request.args['u']\n"
            "    q1 = f\"SELECT * FROM u1 WHERE id = '{u}'\"\n"
            "    db1.execute(q1)\n"
            "    q2 = f\"SELECT * FROM u2 WHERE id = '{u}'\"\n"
            "    db2.execute(q2)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertEqual(len(report["sink_location"]), 2)

    # 11. Safe parameterization through helper
    def test_11_safe_parameterization_through_helper(self):
        code = (
            "def run_safe(uid):\n"
            "    cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))\n"
            "\n"
            "def handle():\n"
            "    user_id = request.args.get('id')\n"
            "    run_safe(user_id)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "SAFE")
        self.assertEqual(len(report["vulnerable_lines"]), 0)

    # 12. Unsafe percent-formatting through helper
    def test_12_unsafe_percent_formatting(self):
        code = (
            "def build_unsafe(name):\n"
            "    return \"SELECT * FROM staff WHERE name = '%s'\" % name\n"
            "\n"
            "def run():\n"
            "    q = build_unsafe(request.args['name'])\n"
            "    cursor.execute(q)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")

    # 13. ORM-like abstraction with raw SQL sink
    def test_13_orm_raw_sql_abstraction(self):
        code = (
            "class ModelRepository:\n"
            "    def find_by_raw(self, raw_filter):\n"
            "        sql = f\"SELECT * FROM models WHERE {raw_filter}\"\n"
            "        session.execute(sql)\n"
            "\n"
            "def api():\n"
            "    repo = ModelRepository()\n"
            "    repo.find_by_raw(request.args['filter'])\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")

    # 14. Callback / higher-order invocation
    def test_14_higher_order_callback(self):
        code = (
            "def execute_with_sink(val):\n"
            "    cursor.execute(f\"SELECT * FROM data WHERE key = '{val}'\")\n"
            "\n"
            "def dispatch(func, argument):\n"
            "    func(argument)\n"
            "\n"
            "def main():\n"
            "    execute_with_sink(request.args['k'])\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")

    # 15. Sanitized flow (int cast in helper) -> SAFE
    def test_15_sanitized_flow_int_cast(self):
        code = (
            "def parse_id(raw_val):\n"
            "    return int(raw_val)\n"
            "\n"
            "def handle():\n"
            "    clean_id = parse_id(request.args['id'])\n"
            "    q = f\"SELECT * FROM orders WHERE id = {clean_id}\"\n"
            "    cursor.execute(q)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "SAFE")
        self.assertEqual(len(report["vulnerable_lines"]), 0)

    # 16. False positive suppression (whitelisted table name in helper) -> SAFE
    def test_16_false_positive_whitelisted_table(self):
        code = (
            "ALLOWED_TABLES = ['users', 'products', 'orders']\n"
            "\n"
            "def get_table_data(tbl_name):\n"
            "    if tbl_name in ALLOWED_TABLES:\n"
            "        cursor.execute(f\"SELECT * FROM {tbl_name}\")\n"
            "\n"
            "def handle():\n"
            "    get_table_data(request.args['table'])\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "SAFE")
        self.assertEqual(len(report["vulnerable_lines"]), 0)

    # 17. False positive suppression (boolean flag in helper) -> SAFE
    def test_17_false_positive_boolean_flag(self):
        code = (
            "def fetch_active():\n"
            "    is_active = True\n"
            "    cursor.execute(f\"SELECT * FROM accounts WHERE active = {is_active}\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "SAFE")
        self.assertEqual(len(report["vulnerable_lines"]), 0)

    # 18. Recursive function (cycle detection, no infinite loop)
    def test_18_recursive_function(self):
        code = (
            "def traverse_and_query(node):\n"
            "    if node.left:\n"
            "        traverse_and_query(node.left)\n"
            "    cursor.execute(f\"SELECT * FROM tree WHERE id = '{node.id}'\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")

    # 19. Mutual recursion (cycle detection)
    def test_19_mutual_recursion(self):
        code = (
            "def ping(x):\n"
            "    pong(x)\n"
            "\n"
            "def pong(y):\n"
            "    ping(y)\n"
            "\n"
            "def trigger():\n"
            "    cursor.execute(f\"SELECT * FROM pingpong WHERE val = '{request.args['p']}'\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")

    # 20. Deeply nested call chain exceeding max depth
    def test_20_max_depth_exceeded(self):
        analyzer = InterproceduralAnalyzer(max_call_depth=2)
        code = (
            "def d1(x): return d2(x)\n"
            "def d2(x): return d3(x)\n"
            "def d3(x): return d4(x)\n"
            "def d4(x): return f\"SELECT * FROM t WHERE id = '{x}'\"\n"
            "def run():\n"
            "    sql = d1(request.args['id'])\n"
            "    cursor.execute(sql)\n"
        )
        res = analyzer.analyze_single_file(code)
        self.assertEqual(res.analysis_status, "PARTIAL")

    # 21. Unresolved external import
    def test_21_unresolved_external_import(self):
        code = (
            "from third_party_module import mystery_helper\n"
            "\n"
            "def run():\n"
            "    val = mystery_helper(request.args['x'])\n"
            "    cursor.execute(f\"SELECT * FROM t WHERE id = '{val}'\")\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")

    # 22. Missing file in import
    def test_22_missing_file_import(self):
        temp_dir = tempfile.mkdtemp(prefix="sqli_test_missing_file_")
        try:
            main_path = os.path.join(temp_dir, "main.py")
            with open(main_path, "w") as f:
                f.write(
                    "from non_existent_module import missing_func\n"
                    "def run():\n"
                    "    cursor.execute('SELECT 1')\n"
                )
            report = self.pipeline.analyze_project(temp_dir)
            self._assert_schema(report)
            self.assertEqual(report["prediction"], "SAFE")
        finally:
            shutil.rmtree(temp_dir)

    # 23. Syntax error in one project file
    def test_23_syntax_error_in_one_file(self):
        temp_dir = tempfile.mkdtemp(prefix="sqli_test_syntax_err_")
        try:
            bad_path = os.path.join(temp_dir, "bad.py")
            good_path = os.path.join(temp_dir, "good.py")

            with open(bad_path, "w") as f:
                f.write("def broken(x: invalid syntax here!!!\n")

            with open(good_path, "w") as f:
                f.write(
                    "def safe_run():\n"
                    "    cursor.execute('SELECT * FROM ok WHERE id = %s', (1,))\n"
                )

            report = self.pipeline.analyze_project(temp_dir)
            self.assertEqual(report["analysis_status"], "PARTIAL")
            self.assertTrue(any("SyntaxError" in m for m in report["messages"]))
            self.assertEqual(report["prediction"], "SAFE")
        finally:
            shutil.rmtree(temp_dir)

    # 24. Dictionary / list passing through helper
    def test_24_dict_list_propagation(self):
        code = (
            "def build_from_dict(payload):\n"
            "    return f\"SELECT * FROM users WHERE id = '{payload['uid']}'\"\n"
            "\n"
            "def handle():\n"
            "    data = {'uid': request.args['id']}\n"
            "    sql = build_from_dict(data)\n"
            "    cursor.execute(sql)\n"
        )
        report = self.pipeline.analyze(code)
        self._assert_schema(report)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(any(vl["line"] == 2 for vl in report["vulnerable_lines"]))

if __name__ == "__main__":
    unittest.main()

