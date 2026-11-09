import requests
import io
import pandas as pd

print("Checking MickyMike/cvefixes_bigvul...")
try:
    url = "https://huggingface.co/datasets/MickyMike/cvefixes_bigvul/raw/main/test.csv"
    r = requests.get(url, timeout=30)
    print("MickyMike test.csv status:", r.status_code, "len:", len(r.content))
except Exception as e:
    print("MickyMike error:", e)

print("\nChecking secbench-hf/SecBench tree...")
try:
    r = requests.get("https://huggingface.co/api/datasets/secbench-hf/SecBench/tree/main/data", timeout=30)
    if r.status_code == 200:
        for item in r.json():
            print(" ", item.get("path"), item.get("size"))
except Exception as e:
    print("SecBench error:", e)
