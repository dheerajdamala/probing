from facade import exec_fn
def action():
    v = request.args['val']
    stmt = f"SELECT * FROM t WHERE val = '{v}'"
    exec_fn(stmt)
