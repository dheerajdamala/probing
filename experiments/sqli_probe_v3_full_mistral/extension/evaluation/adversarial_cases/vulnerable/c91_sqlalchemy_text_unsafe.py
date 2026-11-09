def query_user(session, uid):
    stmt = text(f"SELECT * FROM users WHERE id = '{uid}'")
    session.execute(stmt)
def handle(s): query_user(s, request.args['id'])
