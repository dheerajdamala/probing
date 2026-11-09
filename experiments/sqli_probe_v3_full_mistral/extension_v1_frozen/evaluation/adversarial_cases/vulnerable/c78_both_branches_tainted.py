def both_tainted(cond):
    if cond:
        q = f"SELECT * FROM a WHERE x = '{request.args['x']}'"
    else:
        q = f"SELECT * FROM b WHERE y = '{request.args['y']}'"
    cursor.execute(q)
