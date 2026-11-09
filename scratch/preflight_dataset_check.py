import json
import ast
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.functional_eval import check_syntax, evaluate_functional_correctness
from src.evaluation.security_eval import evaluate_code_security

def main():
    print("Running comprehensive preflight validation on expanded_evaluation_3k.json...")
    fpath = "data/processed/expanded_evaluation_3k.json"
    with open(fpath, "r", encoding="utf-8") as f:
        d = json.load(f)

    tasks = d["tasks"]
    total_tasks = len(tasks)
    print(f"Total tasks loaded: {total_tasks}")

    # 1. Unique IDs and Prompts
    ids = [t["id"] for t in tasks]
    prompts = [t["prompt"] for t in tasks]
    unique_ids = len(set(ids))
    unique_prompts = len(set(prompts))

    # 2. Syntax validation
    malformed_safe = []
    malformed_vuln = []
    for t in tasks:
        syn_safe, err_safe = check_syntax(t["prompt"] + t["safe_completion"])
        if not syn_safe:
            malformed_safe.append((t["id"], err_safe))
        syn_vuln, err_vuln = check_syntax(t["prompt"] + t["vulnerable_completion"])
        if not syn_vuln:
            malformed_vuln.append((t["id"], err_vuln))

    # 3. Train set leakage check
    with open("data/processed/train.json", "r", encoding="utf-8") as f:
        train_pairs = json.load(f)["paired"]
    train_prompts = set(t["prompt"].strip() for t in train_pairs)
    train_ids = set(t["id"] for t in train_pairs)
    leakage_prompts = [t["id"] for t in tasks if t["prompt"].strip() in train_prompts]
    leakage_ids = [t["id"] for t in tasks if t["id"] in train_ids]

    # 4. Distribution Analysis
    cwe_dist = Counter(t["cwe"] for t in tasks)
    domain_dist = Counter(t["domain"] for t in tasks)
    diff_dist = Counter(t["difficulty"] for t in tasks)

    # 5. Near-Duplicate Cluster Analysis
    # Cluster tasks by normalized AST structure (replacing identifiers and constants)
    class ASTNormalizer(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value="CONST"), node)
        def visit_FunctionDef(self, node):
            node.name = "FUNC"
            self.generic_visit(node)
            return node

    ast_clusters = defaultdict(list)
    for t in tasks:
        try:
            tree = ast.parse(t["prompt"] + t["safe_completion"])
            norm_tree = ASTNormalizer().visit(tree)
            ast.fix_missing_locations(norm_tree)
            norm_code = ast.unparse(norm_tree)
            ast_clusters[norm_code].append(t["id"])
        except Exception:
            pass

    print(f"Total distinct structural AST archetypes: {len(ast_clusters)}")
    cluster_sizes = Counter(len(v) for v in ast_clusters.values())
    print(f"Cluster size distribution: {dict(cluster_sizes)}")

    # 6. Stratified Executability & Label Calibration Test (Sample 100 tasks across all CWEs and domains)
    stratified_sample = []
    per_cwe = defaultdict(list)
    for t in tasks:
        per_cwe[t["cwe"]].append(t)
    for cwe, items in per_cwe.items():
        stratified_sample.extend(items[:10]) # 10 tasks per CWE x 10 CWEs = 100 tasks

    print(f"Testing executability and calibration on {len(stratified_sample)} stratified sample tasks...")
    strat_results = []
    fn_pass_count = 0
    sec_pass_count = 0
    vuln_detect_count = 0

    for t in stratified_sample:
        safe_code = t["prompt"] + t["safe_completion"]
        vuln_code = t["prompt"] + t["vulnerable_completion"]

        fn_res = evaluate_functional_correctness(safe_code, t.get("test_cases", []), t["id"])
        sec_safe = evaluate_code_security(safe_code, t["cwe"])
        sec_vuln = evaluate_code_security(vuln_code, t["cwe"])

        if fn_res["functional_passed"]: fn_pass_count += 1
        if sec_safe["is_secure"]: sec_pass_count += 1
        if sec_vuln["is_vulnerable"]: vuln_detect_count += 1

        strat_results.append({
            "task_id": t["id"],
            "cwe": t["cwe"],
            "domain": t["domain"],
            "functional_passed": fn_res["functional_passed"],
            "functional_error": fn_res.get("error"),
            "safe_is_secure": sec_safe["is_secure"],
            "safe_is_vuln": sec_safe["is_vulnerable"],
            "vuln_is_vuln": sec_vuln["is_vulnerable"]
        })

    print(f"Stratified Results (N={len(stratified_sample)}):")
    print(f"  Functional Pass Rate (Safe ref): {fn_pass_count}/{len(stratified_sample)} ({fn_pass_count/len(stratified_sample)*100:.1f}%)")
    print(f"  Security Pass Rate (Safe ref): {sec_pass_count}/{len(stratified_sample)} ({sec_pass_count/len(stratified_sample)*100:.1f}%)")
    print(f"  Vulnerability Detection (Vuln ref): {vuln_detect_count}/{len(stratified_sample)} ({vuln_detect_count/len(stratified_sample)*100:.1f}%)")

    # Save to outputs/EXPANDED_DATASET_PREFLIGHT.json
    preflight_json = {
        "timestamp": "2026-10-28 12:22:00 UTC",
        "dataset_file": fpath,
        "total_records": total_tasks,
        "unique_task_ids": unique_ids,
        "unique_prompts": unique_prompts,
        "exact_prompt_duplicates": total_tasks - unique_prompts,
        "malformed_safe_records": len(malformed_safe),
        "malformed_vuln_records": len(malformed_vuln),
        "train_set_leakage": {
            "prompt_overlap_count": len(leakage_prompts),
            "id_overlap_count": len(leakage_ids),
            "isolated": len(leakage_prompts) == 0 and len(leakage_ids) == 0
        },
        "distributions": {
            "cwe_distribution": dict(cwe_dist),
            "domain_distribution": dict(domain_dist),
            "difficulty_distribution": dict(diff_dist)
        },
        "structural_cluster_analysis": {
            "total_structural_ast_clusters": len(ast_clusters),
            "archetype_distribution": {f"cluster_size_{k}": v for k, v in cluster_sizes.items()}
        },
        "stratified_audit": {
            "sample_size": len(stratified_sample),
            "functional_pass_rate": round(fn_pass_count / len(stratified_sample), 4),
            "safe_reference_security_pass_rate": round(sec_pass_count / len(stratified_sample), 4),
            "vuln_reference_detection_rate": round(vuln_detect_count / len(stratified_sample), 4),
            "sample_records": strat_results[:10]
        }
    }

    with open("outputs/EXPANDED_DATASET_PREFLIGHT.json", "w", encoding="utf-8") as f:
        json.dump(preflight_json, f, indent=2)

    print("Saved outputs/EXPANDED_DATASET_PREFLIGHT.json")

if __name__ == "__main__":
    main()
