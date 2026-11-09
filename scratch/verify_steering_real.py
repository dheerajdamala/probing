import sys
sys.path.insert(0, ".")
import torch
from src.model.load_model import load_model_and_tokenizer
from src.steering.activation_hook import ActivationSteeringHook

model, tokenizer, meta = load_model_and_tokenizer(model_id="axolotl-ai-co/tiny-mistral-25m", device="cpu")
model.eval()

prompt = "def get_user_by_username(cursor, username: str):\n    '''Query user'''\n"
inputs = tokenizer(prompt, return_tensors="pt")

# 1. Baseline unhooked
with torch.no_grad():
    base_out = model(**inputs)
    base_logits = base_out.logits
    base_gen = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)

# Create a normalized steering vector for layer 4
target_layer = 4
hidden_dim = model.config.hidden_size # 384
rand_vec = torch.randn(hidden_dim)
rand_vec = rand_vec / torch.norm(rand_vec)

# 2. Hooked at alpha = 0.0
with ActivationSteeringHook(model, layer_idx=target_layer, steering_vector=rand_vec, alpha=0.0):
    with torch.no_grad():
        a0_out = model(**inputs)
        a0_logits = a0_out.logits
        a0_gen = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)

diff_a0 = (base_logits - a0_logits).abs().max().item()
token_match_a0 = torch.equal(base_gen, a0_gen)
print(f"[Alpha = 0.0] Max Logits Difference: {diff_a0:.8e}")
print(f"[Alpha = 0.0] Tokens Match Baseline: {token_match_a0}")

# 3. Hooked at alpha = 1.5
with ActivationSteeringHook(model, layer_idx=target_layer, steering_vector=rand_vec, alpha=1.5):
    with torch.no_grad():
        a1_out = model(**inputs)
        a1_logits = a1_out.logits
        a1_gen = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)

diff_a1 = (base_logits - a1_logits).abs().max().item()
token_match_a1 = torch.equal(base_gen, a1_gen)
print(f"[Alpha = 1.5] Max Logits Difference: {diff_a1:.4f}")
print(f"[Alpha = 1.5] Tokens Match Baseline: {token_match_a1}")
print(f"Base Generation: {repr(tokenizer.decode(base_gen[0]))}")
print(f"Steered (a=1.5): {repr(tokenizer.decode(a1_gen[0]))}")
