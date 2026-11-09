import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.model.load_model import load_model_and_tokenizer
from src.model.generate import generate_code_completion

print("CUDA available:", torch.cuda.is_available())
m, tok, meta = load_model_and_tokenizer("wandb/pruned_mistral", device="auto", mock=False)
print("Loaded:", meta)

prompt = "def add(a: int, b: int) -> int:\n    \"\"\"Return sum of a and b.\"\"\"\n"
res = generate_code_completion(m, tok, prompt, max_new_tokens=24)
print("Completion:", repr(res["completion"]))
