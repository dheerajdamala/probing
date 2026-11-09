def ternary_flow(flag):
    val = request.args['u'] if flag else 'guest'
    sql = f"SELECT * FROM sessions WHERE user = '{val}'"
    cursor.execute(sql)
