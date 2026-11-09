def select_column(col_name):
    sql = f"SELECT {col_name} FROM users"
    cursor.execute(sql)
def handle(): select_column(request.args['col'])
