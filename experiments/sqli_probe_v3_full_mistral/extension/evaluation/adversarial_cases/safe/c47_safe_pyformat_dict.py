def safe_pyformat(user_val):
    params = {'u': user_val}
    cursor.execute('SELECT * FROM accounts WHERE u = %(u)s', params)
def handle(): safe_pyformat(request.args['u'])
