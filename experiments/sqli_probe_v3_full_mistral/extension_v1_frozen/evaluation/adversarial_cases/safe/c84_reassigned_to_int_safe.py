def reassign_safe():
    var = request.args['x']
    var = 42
    sql = f"SELECT * FROM items WHERE id = {var}"
    cursor.execute(sql)
