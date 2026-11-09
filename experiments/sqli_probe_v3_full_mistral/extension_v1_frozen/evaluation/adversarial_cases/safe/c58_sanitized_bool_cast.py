def query_active():
    is_active = bool(request.args.get('active'))
    sql = f"SELECT * FROM accounts WHERE active = {is_active}"
    cursor.execute(sql)
