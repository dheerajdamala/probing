def overwrite_safe():
    q = 'SELECT 1'
    if True:
        q = f"SELECT * FROM users WHERE id = {request.args['id']}"
    cursor.execute(q)
