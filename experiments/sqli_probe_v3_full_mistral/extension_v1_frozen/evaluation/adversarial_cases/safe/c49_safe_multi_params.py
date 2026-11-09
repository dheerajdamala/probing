def safe_multi(a, b):
    cursor.execute('SELECT * FROM t WHERE a = %s AND b = %s', (a, b))
def handle(): safe_multi(request.args['a'], request.args['b'])
