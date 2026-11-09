def safe_sqlalchemy(session, uid):
    stmt = text("SELECT * FROM users WHERE id = :id")
    session.execute(stmt, {'id': uid})
def handle(s): safe_sqlalchemy(s, request.args['id'])
