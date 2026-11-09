def clean_id(raw):
    return int(raw)
def run():
    safe_id = clean_id(request.args['id'])
    stmt = f"SELECT * FROM t WHERE id = {safe_id}"
    cursor.execute(stmt)
