def query_user(user_id=None):
    cursor.execute(f"SELECT * FROM u WHERE id = {user_id}")
def api():
    query_user(user_id=request.args['id'])
