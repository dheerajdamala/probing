import pandas as pd

df = pd.read_csv('outputs/large_scale_evaluation_per_sample.csv')

print(f"Total rows: {len(df)}")
conditions = df['condition'].unique()

print("\n--- Failure / Anomaly Matrix ---")
for tid in sorted(df['task_id'].unique()):
    sub = df[df['task_id'] == tid]
    fn_status = []
    sec_status = []
    for c in ['unsteered_baseline', 'hooked_alpha_0', 'learned_vector_layer16', 'random_vector_layer16']:
        r = sub[sub['condition'] == c].iloc[0]
        fn_status.append(f"{c[:4]}:{'P' if r['functional_passed'] else 'F'}")
        sec_status.append(f"{c[:4]}:{'S' if r['is_secure'] else 'V'}")
    
    # print only if not all passed or if vulnerable
    all_fn = sub['functional_passed'].all()
    all_sec = sub['is_secure'].all()
    if not all_fn or not all_sec:
        print(f"Task: {tid} | FN: {' '.join(fn_status)} | SEC: {' '.join(sec_status)}")
        for c in ['unsteered_baseline', 'hooked_alpha_0', 'learned_vector_layer16', 'random_vector_layer16']:
            r = sub[sub['condition'] == c].iloc[0]
            if not r['functional_passed'] or not r['is_secure']:
                print(f"   [{c}] fn_err={r['functional_error']} | sec_pat={r['matched_pattern']}")
