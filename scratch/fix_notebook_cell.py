import json

nb_path = 'experiments/sqli_probe_v3_full_mistral/probe_v3_t4.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if 'bnb.nn.Linear4bit' in source:
        cell['source'] = [
            'import torch\n',
            'import bitsandbytes as bnb\n',
            'print(f"bitsandbytes version: {bnb.__version__}")\n',
            '\n',
            '# Test 4-bit linear layer on GPU\n',
            'linear4 = bnb.nn.Linear4bit(128, 128, bias=False, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()\n',
            'dummy_in = torch.randn(2, 128, dtype=torch.bfloat16).cuda()\n',
            'dummy_out = linear4(dummy_in)\n',
            'assert dummy_out.shape == (2, 128), "bitsandbytes 4-bit forward pass failed"\n',
            'print("SUCCESS: bitsandbytes 4-bit NF4 is fully functional on this GPU.")\n'
        ]
        print('Updated Linear4bit test cell.')

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

