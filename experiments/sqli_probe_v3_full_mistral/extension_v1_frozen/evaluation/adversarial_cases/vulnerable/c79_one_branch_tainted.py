def branch_tainted(cond):
    if cond:
        sql = f"SELECT * FROM u WHERE id = {request.args['id']}"
    else:
        sql = 'SELECT * FROM u WHERE id = 1'
    cursor.execute(sql)
