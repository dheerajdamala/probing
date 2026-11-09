import json

audit_data = {
    "timestamp": "2026-10-28 12:07:00 UTC",
    "audit_title": "Comprehensive Repository Dataset and Inference Sample Audit",
    "raw_dataset_records": 127,
    "unique_task_specifications": 122,
    "unique_vulnerability_scenarios": 12,
    "contrastive_training_pairs": 7,
    "distinct_cwe_categories": [
        "CWE-89",
        "CWE-78",
        "CWE-22",
        "CWE-79",
        "CWE-119",
        "CWE-20/119"
    ],
    "distinct_cwe_count": 6,
    "programming_languages": ["python"],
    "programming_language_count": 1,
    "inference_breakdown": {
        "baseline_unsteered_samples": 14,
        "learned_steering_samples": 25,
        "hook_alpha_0_control_samples": 5,
        "random_vector_control_samples": 25,
        "total_genuine_inference_samples": 69,
        "mocked_or_templated_samples": 2300,
        "unverified_samples": 0,
        "total_repository_completions": 2369
    },
    "evaluated_outcomes_genuine": {
        "total_genuine_evaluated": 20,
        "syntax_passing": 20,
        "functional_passing": 19,
        "security_passing_reported": 4,
        "security_passing_audited": 3,
        "joint_passing_audited": 3,
        "joint_pass_by_condition": {
            "baseline_unhooked": 0,
            "hook_alpha_0": 0,
            "steered_sec_L16_a1.0": 3,
            "random_control_L16_a1.0": 0
        }
    },
    "datasets_inventory": [
        {
            "file": "data/raw/isae_paired_corpus.json",
            "type": "raw_paired_corpus",
            "record_count": 17,
            "cwes": ["CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-119"]
        },
        {
            "file": "data/processed/train.json",
            "type": "training_split",
            "record_count": 7,
            "cwes": ["CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-119"]
        },
        {
            "file": "data/processed/val.json",
            "type": "validation_split",
            "record_count": 5,
            "cwes": ["CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-119"]
        },
        {
            "file": "data/processed/test.json",
            "type": "pilot_test_split",
            "record_count": 5,
            "cwes": ["CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-119"]
        },
        {
            "file": "data/processed/held_out_evaluation.json",
            "type": "generalization_split",
            "record_count": 10,
            "cwes": ["CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-119"]
        },
        {
            "file": "data/processed/expanded_evaluation.json",
            "type": "expanded_benchmark_specifications",
            "record_count": 100,
            "cwes": ["CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-20/119"]
        }
    ],
    "inference_artifacts_inventory": [
        {
            "file": "outputs/remote_baseline_results.json",
            "provenance": "GENUINE_REMOTE_GPU",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "samples": 9,
            "notes": "4 simple deterministic tasks + 5 security benchmark unsteered baseline runs"
        },
        {
            "file": "outputs/steering_per_sample.csv",
            "provenance": "GENUINE_REMOTE_GPU",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "samples": 45,
            "notes": "Sweep across Layer 16 with 4 alphas for learned and random, plus baseline"
        },
        {
            "file": "outputs/REAL_INFERENCE_PER_SAMPLE.csv",
            "provenance": "GENUINE_REMOTE_GPU",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "samples": 20,
            "notes": "5 benchmark tasks x 4 core conditions (baseline, alpha=0, steered L16 a1.0, random L16 a1.0)"
        },
        {
            "file": "outputs/expanded_evaluation_per_sample.csv",
            "provenance": "MOCKED_FIXTURE_BRANCH",
            "model": "None (Bypassed via --mock flag)",
            "samples": 2300,
            "notes": "100 tasks x 23 conditions assigned directly from dictionary fixtures"
        }
    ]
}

with open("outputs/DATASET_SIZE_AUDIT.json", "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2)

print("Saved outputs/DATASET_SIZE_AUDIT.json")
