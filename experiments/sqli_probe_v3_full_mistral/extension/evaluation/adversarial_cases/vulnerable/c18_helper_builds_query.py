def build_user_query(uid):
    return f"SELECT * FROM users WHERE id = {uid}"
def handle():
    u = request.args['id']
    q = build_user_query(u)
    cursor.execute(q)
