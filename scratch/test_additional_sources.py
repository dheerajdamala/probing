import requests
import json

candidates = [
    "ayshajavd/code-security-vulnerability-dataset",
    "MickyMike/cvefixes_bigvul",
    "starsofchance/CVEfixes_v1.0.8",
    "CoQuIR/CVEFixes",
    "mmeberg/PyVulDet-NER",
    "lemon42-ai/Code_Vulnerability_Labeled_Dataset",
    "secbench-hf/SecBench",
    "hitoshura25/cvefixes",
    "xin1997/big-vul-cvefix-repair_all_only_input"
]

print("Checking candidate datasets on Hugging Face...")
for repo_id in candidates:
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        d = r.json()
        print(f"\n[FOUND] {repo_id}")
        print(f"  Description: {d.get('description', '')[:100]}...")
        # Get list of files
        files_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
        rf = requests.get(files_url, timeout=15)
        if rf.status_code == 200:
            files = [f.get('path') for f in rf.json() if isinstance(f, dict)]
            print(f"  Files ({len(files)}): {files[:6]}")
    else:
        print(f"[STATUS {r.status_code}] {repo_id}")
