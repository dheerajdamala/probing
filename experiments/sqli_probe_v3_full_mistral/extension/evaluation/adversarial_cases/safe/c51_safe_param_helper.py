def run_safe(uid):
    cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))
def process():
    u = request.args.get('id')
    run_safe(u)
