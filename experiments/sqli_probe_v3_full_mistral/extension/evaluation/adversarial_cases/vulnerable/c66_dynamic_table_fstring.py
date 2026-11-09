def query_table(tbl):
    sql = f"SELECT * FROM {tbl} WHERE active = 1"
    cursor.execute(sql)
def handle(): query_table(request.args['table'])
