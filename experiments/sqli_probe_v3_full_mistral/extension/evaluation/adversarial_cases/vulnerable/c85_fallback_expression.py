def fallback_flow():
    param = request.args.get('term') or 'default'
    sql = f"SELECT * FROM search WHERE term = '{param}'"
    cursor.execute(sql)
