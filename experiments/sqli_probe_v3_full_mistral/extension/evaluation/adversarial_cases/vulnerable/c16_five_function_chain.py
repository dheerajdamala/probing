def f5_sink(v):
    cursor.execute(f"SELECT * FROM t WHERE id = '{v}'")
def f4(v): f5_sink(v)
def f3(v): f4(v)
def f2(v): f3(v)
def f1(v): f2(v)
def entry(): f1(request.args['id'])
