import database_module
def handle():
    v = request.args['token']
    sql = f"SELECT * FROM auth WHERE tok = '{v}'"
    database_module.query_database(sql)
