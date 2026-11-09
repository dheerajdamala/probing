import requests
import json

r = requests.get("https://huggingface.co/api/datasets/Kolega-Dev/RealVuln/tree/main", timeout=15)
if r.status_code == 200:
    for item in r.json():
        print(item.get("path"), item.get("size"))

r2 = requests.get("https://huggingface.co/api/datasets/Kolega-Dev/RealVuln-v2/tree/main", timeout=15)
if r2.status_code == 200:
    for item in r2.json():
        print(item.get("path"), item.get("size"))
