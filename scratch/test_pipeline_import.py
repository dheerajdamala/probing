import sys
import os

ext_dir = os.path.abspath("experiments/sqli_probe_v3_full_mistral/extension")
sys.path.insert(0, ext_dir)

from pipeline import SQLInjectionAnalysisPipeline

pipe = SQLInjectionAnalysisPipeline()
code = "def get_many(conn, ids):\n    marks = \",\".join(\"?\" for _ in ids)\n    return conn.execute(f\"SELECT id, name FROM tags WHERE id IN ({marks})\", list(ids)).fetchall()"
res = pipe.analyze(code, filename="test.py")
print("Status:", res["analysis_status"])
print("Prediction:", res["prediction"])
print("Vulnerability types:", res["vulnerability_type"])
print("Sinks:", res["sink_location"])
print("Findings:", res["findings"])
