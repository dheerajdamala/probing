def render_sql():
    ctx = {'user': request.args['user']}
    stmt = "SELECT * FROM users WHERE username = '{user}'".format_map(ctx)
    conn.execute(stmt)
