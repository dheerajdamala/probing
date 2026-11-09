import sys
sys.path.insert(0, ".")
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.run_benchmarks import extract_clean_code_completion
from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness
from src.evaluation.security_eval import evaluate_code_security

model_id = "axolotl-ai-co/tiny-mistral-25m"
print(f"Loading {model_id}...")
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

with open("data/processed/test.json") as f:
    test_data = json.load(f)["paired"]

print(f"\nLoaded {len(test_data)} test items.")
for item in test_data:
    prompt = item["prompt"]
    encoded = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    comp = tok.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    clean_comp = extract_clean_code_completion(prompt, comp)
    full_code = prompt + clean_comp
    syn_ok, syn_err = check_syntax(full_code)
    sec_eval = evaluate_code_security(full_code, item["cwe"])
    print(f"\n--- Item {item['id']} ({item['cwe']}) ---")
    print(f"Raw comp: {repr(comp[:80])}")
    print(f"Clean comp: {repr(clean_comp[:80])}")
    print(f"Syntax valid: {syn_ok} | Vulnerable: {sec_eval['is_vulnerable']} | Secure: {sec_eval['is_secure']}")
