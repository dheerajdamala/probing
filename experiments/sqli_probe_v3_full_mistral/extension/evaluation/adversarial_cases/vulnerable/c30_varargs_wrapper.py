def raw_exec(*args):
    cursor.execute(args[0])
def entry():
    uid = request.args['id']
    sql = f"SELECT * FROM accounts WHERE id = {uid}"
    raw_exec(sql)
