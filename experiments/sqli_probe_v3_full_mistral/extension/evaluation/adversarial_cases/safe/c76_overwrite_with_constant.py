def overwrite_flow():
    q = f"SELECT * FROM users WHERE id = {request.args['id']}"
    q = "SELECT * FROM users WHERE id = 1"
    cursor.execute(q)
