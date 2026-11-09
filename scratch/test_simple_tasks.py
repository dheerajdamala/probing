import sys
sys.path.insert(0, ".")
import torch
from src.model.load_model import load_model_and_tokenizer
from src.evaluation.functional_eval import check_syntax

model, tokenizer, meta = load_model_and_tokenizer(model_id="axolotl-ai-co/tiny-mistral-25m", device="cpu")

simple_prompts = [
    ("identity", "def identity(x):\n    '''Return x'''\n    return "),
    ("square", "def square(x):\n    '''Return square of x'''\n    return "),
    ("is_even", "def is_even(n: int) -> bool:\n    '''Check if n is even'''\n    return "),
    ("add", "def add(a: int, b: int) -> int:\n    '''Return sum of a and b'''\n    return "),
    ("greet", "def greet(name: str) -> str:\n    '''Return greeting'''\n    return "),
]

print("=== Testing Simple Code Generation on tiny-mistral-25m ===")
for name, prompt in simple_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw_comp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    full_code = prompt + raw_comp
    syn_ok, syn_err = check_syntax(full_code)
    print(f"\nTask: {name}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Generated Completion: {repr(raw_comp)}")
    print(f"Syntax valid: {syn_ok}")
    if syn_err:
        print(f"Syntax error: {syn_err}")
