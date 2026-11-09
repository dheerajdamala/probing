from queries import assemble_sql as build_fn
def run():
    sql = build_fn(request.args['id'])
    cursor.execute(sql)
