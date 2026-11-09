def chain_assign():
    v1 = request.args['x']
    v2 = v1
    v3 = v2
    sql = f"SELECT * FROM logs WHERE entry = '{v3}'"
    cursor.execute(sql)
