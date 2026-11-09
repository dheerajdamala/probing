import requests
import json

datasets = ["yikun-li/TitanVul", "yikun-li/BenchVul", "Mars203020/secmutbench"]

for d in datasets:
    print(f"\n--- Checking {d} ---")
    r = requests.get(f"https://huggingface.co/api/datasets/{d}/tree/main", timeout=15)
    if r.status_code == 200:
        files = [f.get("path") for f in r.json() if isinstance(f, dict)]
        print(f"Files ({len(files)}): {files[:10]}")
    else:
        print(f"Status: {r.status_code}")
