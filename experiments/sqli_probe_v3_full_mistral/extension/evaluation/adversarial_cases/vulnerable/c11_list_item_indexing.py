def list_flow():
    items = [request.args['param']]
    sql = f"SELECT * FROM items WHERE code = '{items[0]}'"
    cursor.execute(sql)
