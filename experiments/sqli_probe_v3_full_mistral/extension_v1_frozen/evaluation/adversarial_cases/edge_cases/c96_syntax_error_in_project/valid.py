def handler():
    sql = f"SELECT * FROM users WHERE id = '{request.args['id']}'"
    cursor.execute(sql)
