def builder(val):
    return f"SELECT * FROM data WHERE key = '{val}'"
def runner(query):
    cursor.execute(query)
def controller():
    runner(builder(request.args['k']))
