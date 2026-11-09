def try_flow():
    try:
        val = request.args['secret']
    except KeyError:
        val = 'default'
    sql = f"SELECT * FROM secrets WHERE key = '{val}'"
    cursor.execute(sql)
