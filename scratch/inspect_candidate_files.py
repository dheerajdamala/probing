import requests
import json
import os

print("--- Inspecting ayshajavd/code-security-vulnerability-dataset data folder ---")
r = requests.get("https://huggingface.co/api/datasets/ayshajavd/code-security-vulnerability-dataset/tree/main/data", timeout=15)
if r.status_code == 200:
    for item in r.json():
        print(item.get("path"), item.get("size"))

print("\n--- Inspecting lemon42-ai/Code_Vulnerability_Labeled_Dataset ---")
r2 = requests.get("https://huggingface.co/api/datasets/lemon42-ai/Code_Vulnerability_Labeled_Dataset/tree/main", timeout=15)
if r2.status_code == 200:
    for item in r2.json():
        print(item.get("path"), item.get("size"))

print("\n--- Inspecting starsofchance/CVEfixes_v1.0.8 ---")
r3 = requests.get("https://huggingface.co/api/datasets/starsofchance/CVEfixes_v1.0.8/tree/main", timeout=15)
if r3.status_code == 200:
    for item in r3.json():
        print(item.get("path"), item.get("size"))
