import requests
import json
import os

print("--- Testing CodeQL CWE-089 Python test files on GitHub ---")
# CodeQL python test directory for CWE-089
api_url = "https://api.github.com/repos/github/codeql/contents/python/ql/test/query-tests/Security/CWE-089"
r = requests.get(api_url, timeout=15)
if r.status_code == 200:
    files = [f.get("name") for f in r.json() if f.get("name", "").endswith(".py")]
    print(f"CodeQL CWE-089 test python files ({len(files)}): {files}")
else:
    print(f"GitHub API status: {r.status_code}")

print("\n--- Testing DjanGoat / PyGoat on GitHub ---")
r_djangoat = requests.get("https://api.github.com/repos/adeyosemanputra/pygoat/contents/pygoat/introduction", timeout=15)
if r_djangoat.status_code == 200:
    files = [f.get("name") for f in r_djangoat.json() if f.get("name", "").endswith(".py")]
    print(f"PyGoat files ({len(files)}): {files}")
else:
    print(f"PyGoat status: {r_djangoat.status_code}")
