def join_table(sec_tbl):
    sql = f"SELECT * FROM orders JOIN {sec_tbl} ON orders.id = {sec_tbl}.oid"
    cursor.execute(sql)
def handle(): join_table(request.args['sec'])
