from ..db import exec_query
def do_action():
    val = request.args['x']
    s = f"SELECT * FROM t WHERE val = '{val}'"
    exec_query(s)
