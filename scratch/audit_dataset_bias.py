import ast
import json
import difflib
from collections import Counter
from typing import List, Dict, Any, Set

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["paired"]

train_items = load_json("data/processed/train.json")
test_items = load_json("data/processed/test.json")
val_items = load_json("data/processed/val.json")
held_out_items = load_json("data/processed/held_out_evaluation.json")

print(f"Loaded: train={len(train_items)}, test={len(test_items)}, val={len(val_items)}, held_out={len(held_out_items)}")

def get_ast_node_types(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
        return [type(node).__name__ for node in ast.walk(tree)]
    except Exception:
        return []

def get_ast_structure_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
        class StructureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.seq = []
            def generic_visit(self, node):
                self.seq.append(type(node).__name__)
                super().generic_visit(node)
        v = StructureVisitor()
        v.visit(tree)
        return "-".join(v.seq)
    except Exception:
        return "PARSE_ERROR"

def extract_apis(code: str) -> Set[str]:
    apis = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    apis.add(f"{getattr(node.func.value, 'id', '')}.{node.func.attr}")
                elif isinstance(node.func, ast.Name):
                    apis.add(node.func.id)
    except Exception:
        pass
    return apis

# Audit details
splits = {
    "train": train_items,
    "pilot_test": test_items,
    "held_out": held_out_items,
}

all_tasks = []
for sname, items in splits.items():
    for it in items:
        it_copy = dict(it)
        it_copy["split"] = sname
        all_tasks.append(it_copy)

print(f"\nTotal tasks across splits: {len(all_tasks)}")

# 1. Exact duplicates
exact_prompts = [t["prompt"].strip() for t in all_tasks]
prompt_counts = Counter(exact_prompts)
duplicates = {k: v for k, v in prompt_counts.items() if v > 1}
print(f"Exact prompt duplicate count: {len(duplicates)}")

# 2. Normalized similarity between train and held-out / test
print("\n--- Structural and API Overlap Audit ---")
for t in all_tasks:
    t["apis_safe"] = extract_apis(t["prompt"] + t["safe_completion"])
    t["apis_vuln"] = extract_apis(t["prompt"] + t["vulnerable_completion"])
    t["ast_sig_safe"] = get_ast_structure_signature(t["prompt"] + t["safe_completion"])
    t["ast_sig_vuln"] = get_ast_structure_signature(t["prompt"] + t["vulnerable_completion"])

train_apis_safe = set().union(*[t["apis_safe"] for t in all_tasks if t["split"] == "train"])
train_apis_vuln = set().union(*[t["apis_vuln"] for t in all_tasks if t["split"] == "train"])

test_apis_safe = set().union(*[t["apis_safe"] for t in all_tasks if t["split"] == "pilot_test"])
held_out_apis_safe = set().union(*[t["apis_safe"] for t in all_tasks if t["split"] == "held_out"])

print(f"Train safe APIs: {train_apis_safe}")
print(f"Pilot test safe APIs: {test_apis_safe}")
print(f"Held-out safe APIs: {held_out_apis_safe}")

print(f"API overlap Train vs Pilot: {train_apis_safe & test_apis_safe}")
print(f"API overlap Train vs Held-Out: {train_apis_safe & held_out_apis_safe}")

# 3. Pairwise sequence similarity
print("\n--- Pairwise Prompt Similarity (Train vs Test / Held-Out) ---")
high_similarity_pairs = []
for tr in train_items:
    for te in test_items + held_out_items:
        sim = difflib.SequenceMatcher(None, tr["prompt"], te["prompt"]).ratio()
        if sim > 0.6:
            high_similarity_pairs.append((tr["id"], te["id"], round(sim, 3), tr["cwe"], te["cwe"]))

print(f"High prompt similarity pairs (>0.60): {len(high_similarity_pairs)}")
for tr_id, te_id, sim, c1, c2 in high_similarity_pairs:
    print(f"  [{c1}] {tr_id} <-> {te_id}: {sim*100:.1f}%")

# 4. Repair Idiom Repetition
print("\n--- Repair Idiom Repetition Analysis ---")
idioms_by_cwe = {}
for t in all_tasks:
    cwe = t["cwe"]
    if cwe not in idioms_by_cwe:
        idioms_by_cwe[cwe] = []
    # Identify idiom in safe completion
    safe_comp = t["safe_completion"]
    idiom = "other"
    if "cursor.execute" in safe_comp:
        idiom = "cursor_parameterized"
    elif "subprocess.run" in safe_comp and "shell=False" in safe_comp:
        idiom = "subprocess_shell_false"
    elif "html.escape" in safe_comp:
        idiom = "html_escape"
    elif "os.path.commonpath" in safe_comp or "os.path.abspath" in safe_comp:
        idiom = "path_canonicalization"
    elif "IndexError" in safe_comp or "len(" in safe_comp:
        idiom = "bounds_check"
    idioms_by_cwe[cwe].append((t["split"], t["id"], idiom))

for cwe, items in idioms_by_cwe.items():
    print(f"[{cwe}]:")
    idiom_counts = Counter(i[2] for i in items)
    for idiom, count in idiom_counts.items():
        print(f"  {idiom}: {count}/{len(items)} ({count/len(items)*100:.0f}%)")

# Save machine-readable overlap audit results
audit_results = {
    "total_tasks": len(all_tasks),
    "train_count": len(train_items),
    "pilot_count": len(test_items),
    "held_out_count": len(held_out_items),
    "exact_duplicates": len(duplicates),
    "api_overlap_train_test": list(train_apis_safe & test_apis_safe),
    "api_overlap_train_heldout": list(train_apis_safe & held_out_apis_safe),
    "high_prompt_similarity_pairs": [
        {"train_id": p[0], "eval_id": p[1], "similarity": p[2], "train_cwe": p[3], "eval_cwe": p[4]}
        for p in high_similarity_pairs
    ],
    "repair_idiom_repetition": {
        cwe: dict(Counter(i[2] for i in items)) for cwe, items in idioms_by_cwe.items()
    }
}

with open("outputs/dataset_overlap_results.json", "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print("\nSaved overlap audit to outputs/dataset_overlap_results.json")
