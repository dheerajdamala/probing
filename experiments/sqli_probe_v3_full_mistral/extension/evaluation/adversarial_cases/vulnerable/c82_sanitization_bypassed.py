def bypass_flow(cond, raw_val):
    if cond:
        val = int(raw_val)
    else:
        val = raw_val
    cursor.execute(f"SELECT * FROM data WHERE v = {val}")
def api(): bypass_flow(False, request.args['v'])
