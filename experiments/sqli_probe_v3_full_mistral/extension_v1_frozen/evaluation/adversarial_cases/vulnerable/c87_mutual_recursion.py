def ping(x): pong(x)
def pong(y):
    if False: ping(y)
def entry():
    cursor.execute(f"SELECT * FROM t WHERE val = '{request.args['p']}'")
