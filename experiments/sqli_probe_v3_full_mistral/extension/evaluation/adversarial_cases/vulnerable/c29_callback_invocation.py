def db_callback(val):
    cursor.execute(f"SELECT * FROM entries WHERE v = '{val}'")
def dispatch(cb, arg):
    cb(arg)
def run():
    dispatch(db_callback, request.args['v'])
