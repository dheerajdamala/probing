import json
import base64
import os

bundle_path = 'experiments/sqli_probe_v3_full_mistral/expanded_v2/expanded_v2_t4_bundle.zip'
with open(bundle_path, 'rb') as f:
    b64_payload = base64.b64encode(f.read()).decode('ascii')

nb = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Probe V3 Expanded E2 Mistral-7B GPU Extraction Engine\n',
                '\n',
                '**Architecture:** `mistralai/Mistral-7B-Instruct-v0.3` ($d=4,096$, 32 Layers)\n',
                '**Target Layer:** Layer 12 AST SQL Sink Pooling\n',
                '**Dataset:** Expanded E2 Dataset ($N=4,673$ total: 3,304 train, 674 val, 695 test)\n',
                '**Task:** Extract genuine 4096-d representations for 3,673 new samples (2,604 train, 525 val, 544 test).\n',
                '\n',
                '> [!IMPORTANT]\n',
                '> Use an environment with GPU acceleration enabled (Tesla T4, V100, A10G, or A100).\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Cell 1: Environment & GPU Hardware Verification\n',
                'Confirm NVIDIA GPU and CUDA runtime are active.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import sys, os, psutil, torch\n',
                '\n',
                'print(f"Python Version: {sys.version.split()[0]}")\n',
                'print(f"PyTorch Version: {torch.__version__}")\n',
                'print(f"CUDA Available: {torch.cuda.is_available()}")\n',
                'if torch.cuda.is_available():\n',
                '    device_name = torch.cuda.get_device_name(0)\n',
                '    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n',
                '    print(f"GPU Model: {device_name}")\n',
                '    print(f"GPU Total VRAM: {total_mem:.2f} GB")\n',
                '    assert total_mem >= 14.0, "Warning: Model requires >=14GB VRAM (Tesla T4 or better) for batch_size=8."\n',
                'else:\n',
                '    raise RuntimeError("CUDA is not available. Please enable GPU in Runtime > Change runtime type.")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Cell 2: Install Compatible Dependencies\n',
                'Install transformers, bitsandbytes, accelerate, and dependencies.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '!pip install -q "transformers>=4.40.0,<4.47.0" "accelerate>=0.29.0" "bitsandbytes>=0.43.0" "safetensors" "scikit-learn"\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Cell 3: Unpack Self-Contained Extraction Payload\n',
                'Unpacks scripts, AST sink extractor, and dataset split subsets.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import base64, io, zipfile, os\n',
                '\n',
                f'PAYLOAD = """{b64_payload}"""\n',
                '\n',
                'print("Unpacking self-contained extraction bundle...")\n',
                'buf = io.BytesIO(base64.b64decode(PAYLOAD))\n',
                'with zipfile.ZipFile(buf, "r") as z:\n',
                '    z.extractall(".")\n',
                '    print("Unpacked files:", z.namelist())\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Cell 4: Authenticate with Hugging Face (Optional/Recommended)\n',
                'If needed to download mistralai/Mistral-7B-Instruct-v0.3 without rate limits.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'from huggingface_hub import login\n',
                'import os\n',
                'hf_token = os.environ.get("HF_TOKEN")\n',
                'if hf_token:\n',
                '    login(token=hf_token)\n',
                '    print("Logged in with HF_TOKEN from environment.")\n',
                'else:\n',
                '    print("HF_TOKEN not set; proceeding with public access / cached model.")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Cell 5: Verify Model & AST Sink Forward Pass\n',
                'Loads Mistral-7B in 4-bit NF4 and validates a test sample.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '!python extract_expanded_embeddings.py --mode verify --device cuda\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Cell 6: Extract Representations for New Samples\n',
                'Runs extraction across the 3 unextracted sets (3,673 samples total).\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'os.makedirs("output/new_extracted", exist_ok=True)\n',
                '!python extract_expanded_embeddings.py --mode extract --data_dir data --output_dir output --batch_size 8 --device cuda\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Cell 7: Package Extracted Representations for Download\n',
                'Zips the extracted .npz files into mistral_probe_v3_expanded_v2_embeddings.zip.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import zipfile, glob\n',
                '\n',
                'out_zip = "mistral_probe_v3_expanded_v2_embeddings.zip"\n',
                'with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:\n',
                '    for f in glob.glob("output/new_extracted/*.npz"):\n',
                '        z.write(f, os.path.basename(f))\n',
                '        print("Added to zip:", f)\n',
                '\n',
                'print(f"Successfully created {out_zip} ({os.path.getsize(out_zip) / (1024*1024):.2f} MB)!")\n'
            ]
        }
    ],
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'provenance': []},
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 0
}

nb_path = 'experiments/sqli_probe_v3_full_mistral/expanded_v2/probe_v3_expanded_t4.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Created Jupyter Notebook at:', nb_path)

