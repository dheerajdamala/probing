def unpack_flow():
    raw_a, safe_b = request.args['x'], 100
    sql = f"SELECT * FROM inventory WHERE batch = '{raw_a}'"
    cursor.execute(sql)
