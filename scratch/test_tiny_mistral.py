import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "axolotl-ai-co/tiny-mistral-25m"
print(f"Loading {model_id}...")
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

prompt = "def add(a, b):\n    '''Return the sum of a and b.'''\n    return"
encoded = tok(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=25,
        do_sample=False
    )
print("GENERATION RESULT:")
print(tok.decode(out[0], skip_special_tokens=True))
