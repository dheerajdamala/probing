def get_user():
    clean_id = int(request.args['id'])
    sql = f"SELECT * FROM users WHERE id = {clean_id}"
    cursor.execute(sql)
