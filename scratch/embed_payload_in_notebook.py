import json
import base64
import io
import zipfile
import os

base_dir = "experiments/sqli_probe_v3_full_mistral"
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".jsonl") or f.endswith(".py") or (f.endswith(".json") and "report" in f) or (f.endswith(".json") and "manifest" in f):
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, base_dir).replace("\\", "/")
                zf.write(fp, rel)
                print(f"Zipped: {rel}")

b64_payload = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Total payload size: {len(b64_payload)} characters")

nb_path = os.path.join(base_dir, "probe_v3_t4.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

extract_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Step 0: Self-Extracting Probe V3 Code & Frozen E2 Datasets\n",
        "# Unpacks all extraction scripts, AST extractors, manifests, and data splits into the working directory.\n",
        "import base64, io, zipfile, os\n",
        "\n",
        f"PAYLOAD = \"\"\"{b64_payload}\"\"\"\n",
        "\n",
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD)), \"r\") as zf:\n",
        "    zf.extractall(\".\")\n",
        "\n",
        "print(\"SUCCESS: Unpacked all Probe V3 datasets, AST extractors, and scripts!\")\n",
        "if os.path.exists(\"data/e2_excel/train.jsonl\"):\n",
        "    print(f\"Verified data/e2_excel/train.jsonl: {os.path.getsize('data/e2_excel/train.jsonl')} bytes\")\n"
    ]
}

has_extract = any("Self-Extracting Probe V3" in "".join(c.get("source", [])) for c in nb["cells"])
if not has_extract:
    nb["cells"].insert(1, extract_cell)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated probe_v3_t4.ipynb with self-extracting payload cell.")
else:
    print("Self-extracting cell already present in probe_v3_t4.ipynb.")

