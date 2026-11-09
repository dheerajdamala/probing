def dict_flow():
    params = {'filter': request.args['filter']}
    sql = f"SELECT * FROM records WHERE filter = '{params['filter']}'"
    cursor.execute(sql)
