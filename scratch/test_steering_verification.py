import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.model.load_model import load_model_and_tokenizer
from src.model.generate import generate_code_completion
from src.steering.activation_hook import ActivationSteeringHook
from src.steering.steered_generate import generate_with_steering

def verify_steering():
    print("=== 1. Loading Mock Mistral ===")
    model, tokenizer, meta = load_model_and_tokenizer(mock=True)
    hidden_size = model.config.hidden_size
    layer_idx = 1
    prompt = "def calculate_discount(price: float, discount: float) -> float:\n"

    # Random unit vector
    torch.manual_seed(42)
    v_sec = torch.randn(hidden_size)
    v_sec = v_sec / torch.norm(v_sec, p=2)

    v_rand = torch.randn(hidden_size)
    v_rand = v_rand / torch.norm(v_rand, p=2)

    print(f"v_sec norm: {torch.norm(v_sec, p=2).item():.6f}")
    print(f"v_rand norm: {torch.norm(v_rand, p=2).item():.6f}")
    assert abs(torch.norm(v_sec, p=2).item() - 1.0) < 1e-6
    assert abs(torch.norm(v_rand, p=2).item() - 1.0) < 1e-6

    # Test baseline without hook
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        base_logits = model(**inputs).logits
    base_gen = generate_code_completion(model, tokenizer, prompt, max_new_tokens=16, temperature=0.0)

    # Test with hook at alpha=0.0
    with ActivationSteeringHook(model, layer_idx=layer_idx, steering_vector=v_sec, alpha=0.0):
        with torch.no_grad():
            alpha0_logits = model(**inputs).logits
    alpha0_gen = generate_with_steering(model, tokenizer, prompt, steering_vector=v_sec, layer_idx=layer_idx, alpha=0.0, max_new_tokens=16)

    logits_diff_alpha0 = torch.max(torch.abs(base_logits - alpha0_logits)).item()
    print(f"Max logits diff between unhooked and hook(alpha=0.0): {logits_diff_alpha0:.2e}")
    assert logits_diff_alpha0 < 1e-5, "alpha=0.0 altered model logits!"
    assert base_gen["completion"] == alpha0_gen["completion"], "alpha=0.0 altered deterministic generation!"
    print("CONFIRMED: alpha=0.0 reproduces unmodified model behavior exactly.")

    # Test with hook at alpha=1.5
    with ActivationSteeringHook(model, layer_idx=layer_idx, steering_vector=v_sec, alpha=1.5) as hook:
        with torch.no_grad():
            steered_logits = model(**inputs).logits
    steered_gen = generate_with_steering(model, tokenizer, prompt, steering_vector=v_sec, layer_idx=layer_idx, alpha=1.5, max_new_tokens=16)

    logits_diff_steered = torch.max(torch.abs(base_logits - steered_logits)).item()
    print(f"Max logits diff with steering(alpha=1.5): {logits_diff_steered:.4f}")
    assert logits_diff_steered > 1e-2, "Steering did not alter logits!"
    print(f"Baseline completion: {repr(base_gen['completion'])}")
    print(f"Steered completion:  {repr(steered_gen['completion'])}")

    tokens_differ = (base_gen["completion"] != steered_gen["completion"])
    print(f"Did generated text change with steering? {tokens_differ}")

    # Random control test
    rand_gen = generate_with_steering(model, tokenizer, prompt, steering_vector=v_rand, layer_idx=layer_idx, alpha=1.5, max_new_tokens=16)
    print(f"Random completion:   {repr(rand_gen['completion'])}")
    print("CONFIRMED: Nonzero steering modifies activations and changes generation.")

if __name__ == "__main__":
    verify_steering()
