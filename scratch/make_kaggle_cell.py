import base64
import os

with open("isae_colab_bundle.zip", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

header = """# --- Step 1: Self-Extracting Repository & Dependencies ---
import base64
import os
import zipfile

bundle_b64 = \"\"\""""

footer = """\"\"\"

with open("bundle.zip", "wb") as f:
    f.write(base64.b64decode(bundle_b64))

with zipfile.ZipFile("bundle.zip", "r") as zf:
    zf.extractall(".")

print("Extracted all repository files and 3,500 tasks successfully!")

# Step 2: Install required packages & verify GPU
get_ipython().system("pip install -q torch transformers accelerate bitsandbytes pytest pyyaml")
get_ipython().system("nvidia-smi")
"""

with open("scratch/kaggle_setup_cell.py", "w", encoding="utf-8") as f:
    f.write(header + b64_data + footer)

print(f"Created scratch/kaggle_setup_cell.py: {os.path.getsize('scratch/kaggle_setup_cell.py') / 1024:.1f} KB")
