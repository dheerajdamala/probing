def alias_exec(uid):
    run = cursor.execute
    sql = f"SELECT * FROM t WHERE id = {uid}"
    run(sql)
