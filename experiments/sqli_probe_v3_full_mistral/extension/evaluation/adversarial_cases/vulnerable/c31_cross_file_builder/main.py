from builder import build_user_sql
def run():
    u = request.args['id']
    sql = build_user_sql(u)
    cursor.execute(sql)
