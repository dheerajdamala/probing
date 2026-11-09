def fetch_param():
    return request.args.get('target')
def handler():
    target = fetch_param()
    stmt = f"SELECT * FROM nodes WHERE name = '{target}'"
    cursor.execute(stmt)
