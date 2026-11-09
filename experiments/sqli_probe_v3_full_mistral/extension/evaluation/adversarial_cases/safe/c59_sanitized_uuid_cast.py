import uuid
def fetch_session():
    clean_uuid = str(uuid.UUID(request.args['token']))
    sql = f"SELECT * FROM sessions WHERE tok = '{clean_uuid}'"
    cursor.execute(sql)
