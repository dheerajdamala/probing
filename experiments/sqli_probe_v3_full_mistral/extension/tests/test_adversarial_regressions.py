"""
Regression Test Suite for Adversarial Defects Identified in SQL Injection Probe V3.
Documents discovered defects with targeted reproduction tests before implementing fixes.
"""

import os
import sys
import unittest

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(EXT_DIR, ".."))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline import SQLInjectionAnalysisPipeline

class TestAdversarialRegressions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = SQLInjectionAnalysisPipeline()

    # Defect 1: format_map not recognized
    def test_defect_format_map(self):
        code = (
            "def render_sql():\n"
            "    ctx = {'user': request.args['user']}\n"
            "    stmt = \"SELECT * FROM users WHERE username = '{user}'\".format_map(ctx)\n"
            "    conn.execute(stmt)\n"
        )
        report = self.pipeline.analyze(code)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertIn("FORMAT_METHOD", report["vulnerability_type"])

    # Defect 2: Peewee execute_sql not recognized as sink
    def test_defect_peewee_execute_sql(self):
        code = (
            "def raw_peewee(code):\n"
            "    db.execute_sql(f\"SELECT * FROM inventory WHERE code = '{code}'\")\n"
        )
        report = self.pipeline.analyze(code)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(len(report["sink_location"]) > 0)

    # Defect 3: Django extra() where clause not recognized as sink
    def test_defect_django_extra(self):
        code = (
            "def filter_name(val):\n"
            "    return Entry.objects.extra(where=[\"title = '\" + val + \"'\"])\n"
        )
        report = self.pipeline.analyze(code)
        self.assertEqual(report["prediction"], "VULNERABLE")
        self.assertTrue(len(report["sink_location"]) > 0)

    # Defect 4: str(uuid.UUID(...)) wrapped sanitizer
    def test_defect_uuid_str_wrapper(self):
        code = (
            "import uuid\n"
            "def fetch_session():\n"
            "    clean_uuid = str(uuid.UUID(request.args['token']))\n"
            "    sql = f\"SELECT * FROM sessions WHERE tok = '{clean_uuid}'\"\n"
            "    cursor.execute(sql)\n"
        )
        report = self.pipeline.analyze(code)
        self.assertEqual(report["prediction"], "SAFE")

    # Defect 5: Subclass method inheritance resolution
    def test_defect_subclass_method_inheritance(self):
        code = (
            "class BaseDB:\n"
            "    def run_query(self, query):\n"
            "        cursor.execute(query)\n"
            "class UserDB(BaseDB):\n"
            "    def find_user(self, name):\n"
            "        self.run_query(f\"SELECT * FROM users WHERE name = '{name}'\")\n"
            "def api():\n"
            "    UserDB().find_user(request.args['name'])\n"
        )
        report = self.pipeline.analyze(code)
        self.assertEqual(report["prediction"], "VULNERABLE")

    # Defect 6: Empty file handling
    def test_defect_empty_file_safe(self):
        code = ""
        report = self.pipeline.analyze(code)
        self.assertEqual(report["prediction"], "SAFE")
        self.assertEqual(report["analysis_status"], "COMPLETE")

    # Defect 7: Unresolved external import should be UNCERTAIN / PARTIAL
    def test_defect_unresolved_call_uncertain(self):
        code = (
            "from third_party_security_lib import mystery_sanitizer\n"
            "def process():\n"
            "    val = mystery_sanitizer(request.args['val'])\n"
            "    cursor.execute(val)\n"
        )
        report = self.pipeline.analyze(code)
        self.assertEqual(report["prediction"], "UNCERTAIN")
        self.assertEqual(report["analysis_status"], "PARTIAL")

if __name__ == "__main__":
    unittest.main()

