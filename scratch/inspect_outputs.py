import json

with open("outputs/benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for cond_name, cond in data["conditions"].items():
    print(f"\n==================== Condition: {cond_name} ====================")
    for item in cond["per_item_results"]:
        comp = item["completion"].encode('ascii', 'backslashreplace').decode('ascii').replace("\n", "\\n")
        if len(comp) > 60:
            comp = comp[:60] + "..."
        err = str(item["functional"].get("error")).encode('ascii', 'backslashreplace').decode('ascii')
        syntax_ok = item["functional"].get("syntax_valid")
        func_ok = item["functional"].get("functional_passed")
        sec_ok = item["security"].get("is_secure")
        vuln_ok = item["security"].get("is_vulnerable")
        print(f"[{item['cwe']}] {item['id']}: syn={syntax_ok}, func={func_ok}, sec={sec_ok}, vuln={vuln_ok}")
        print(f"   err: {err}")
        print(f"   comp: {comp}")
