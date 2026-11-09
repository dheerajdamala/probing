from helpers import *
def entry():
    inp = request.args['q']
    s = f"SELECT * FROM docs WHERE q = '{inp}'"
    raw_query(s)
